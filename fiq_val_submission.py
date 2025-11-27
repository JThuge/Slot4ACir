import json
from argparse import ArgumentParser
from operator import itemgetter
from typing import List, Tuple, Dict
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from data_utils import FashionIQDataset, targetpad_transform, base_path
from utils import device, extract_index_blip_features
from lavis.models import load_model_and_preprocess
import argparse


def collate_fn(batch: list):
    """
    Discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def generate_fiq_test_submissions(file_name: str, blip_model, preprocess, txt_processors, rerank, exp_name):
    # Define the dataset and extract index features
    # split: str, dress_types: List[str], mode: str, preprocess
    dress_types = ['dress', 'shirt', 'toptee']
    for dress_type in dress_types:
        relative_val_dataset = FashionIQDataset('val', dress_type, 'relative', preprocess)
        classic_val_dataset = FashionIQDataset('val', dress_type, 'classic', preprocess)
        
        index_features, index_names = extract_index_blip_features(classic_val_dataset, blip_model)

        # Generate test prediction dicts for CIRR
        pairid_to_predictions = generate_fiq_test_dicts(relative_val_dataset, blip_model,
                                                                                    index_features, index_names, txt_processors, rerank)

        submission = dict()
        submission.update(pairid_to_predictions)

        # Define submission path
        from pathlib import Path
        submissions_folder_path = Path(f"./submission/FashionIQ/{dress_type}/{exp_name}")
        submissions_folder_path.mkdir(exist_ok=False, parents=True)

        if rerank:
            file_name = file_name + f'_{rerank}'

        print(f"Saving FashionIQ test predictions")
        with open(submissions_folder_path / f"recall_submission_{file_name}.json", 'w+') as file:
            json.dump(submission, file, sort_keys=True)


def generate_fiq_test_dicts(relative_test_dataset: FashionIQDataset, blip_model, index_features: torch.tensor,
                             index_names: List[str], txt_processors, rerank) \
        -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    
    # Generate predictions
    # predicted_sim, reference_names, group_members, pairs_id, = \
    pred_sim, target_names, reference_names, _ = \
        generate_fiq_test_predictions(blip_model, relative_test_dataset, index_names,
                                       index_features, txt_processors)

    print(f"Compute FashionIQ {relative_test_dataset.dress_types} validation results")

    # Compute the distances and sort the results
    distances = 1 - pred_sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    pids = list(range(len(sorted_index_names)))
    # Generate prediction dicts
    pairid_to_predictions = {str(int(pid)): prediction[:50].tolist() for (pid, prediction) in
                             zip(pids, sorted_index_names)}

    return pairid_to_predictions


def generate_fiq_test_predictions(blip_model, relative_val_dataset: FashionIQDataset,
                                 index_names: List[str], index_features, txt_processors, save_memory=False, textual_inversion=False) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Compute FashionIQ predictions on the validation set
    :param clip_model: CLIP model
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features and target names
    """
    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation predictions")

    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=16,
                                     num_workers=4, pin_memory=True, collate_fn=collate_fn,
                                     shuffle=False)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features[-1]))

    # Initialize predicted features and target names
    target_names = []
    reference_names_all = []
    distance = []
    captions_all = []

    for reference_names, batch_target_names, captions in tqdm(relative_val_loader):  # Load data

        # Concatenate the captions in a deterministic way
        flattened_captions: list = np.array(captions).T.flatten().tolist()
        if textual_inversion:
            prefix = 'a photo of $ that '
        else:
            prefix = ''
        input_captions = [
            f"{prefix}{flattened_captions[i].strip('.?, ').lower()} and {flattened_captions[i + 1].strip('.?, ').lower()}" for
            i in range(0, len(flattened_captions), 2)]
        input_captions = [txt_processors["eval"](caption) for caption in input_captions]
        # Compute the predicted features
        with torch.no_grad():
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if len(input_captions) == 1:
                reference_image_features = itemgetter(*reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            feature_curr = index_features[0]
            if save_memory:
                feature_curr = feature_curr.to(blip_model.device)
                reference_image_features = reference_image_features.to(blip_model.device)
            batch_distance = blip_model.inference(reference_image_features, feature_curr, input_captions)
            distance.append(batch_distance)
            captions_all += input_captions

        target_names.extend(batch_target_names)
        reference_names_all.extend(reference_names)
    
    distance = torch.vstack(distance)

    return distance, target_names, reference_names_all, captions_all


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = ArgumentParser()
    parser.add_argument("--blip-model-name", default="blip2_cir_cat", type=str)
    parser.add_argument("--model-path", type=str, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--backbone", type=str, default="pretrain", help="pretrain for vit-g, pretrain_vitL for vit-l")
    parser.add_argument("--rerank", type=str2bool, default=False)
    parser.add_argument("--save-name", type=str, default=False)

    args = parser.parse_args()
    # blip model
    blip_model, _, txt_processors = load_model_and_preprocess(name=args.blip_model_name, model_type=args.backbone, is_eval=False, device=device)
    
    checkpoint_path = args.model_path

    checkpoint = torch.load(checkpoint_path, map_location=device)
    msg = blip_model.load_state_dict(checkpoint[blip_model.__class__.__name__], strict=False)
    if len(msg) != 0:
        print("Missing keys {}".format(msg.missing_keys))

    input_dim = 224
    preprocess = targetpad_transform(1.25, input_dim)

    generate_fiq_test_submissions(f'{args.blip_model_name}', blip_model, preprocess, txt_processors, args.rerank, args.save_name)


if __name__ == '__main__':
    main()