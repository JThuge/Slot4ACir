# from comet_ml import Experiment
import json
import os
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, geometric_mean, harmonic_mean
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lavis.models import load_model_and_preprocess
from torch.optim.lr_scheduler import OneCycleLR

from data_utils import base_path, squarepad_transform, targetpad_transform, \
    CIRRDataset, FashionIQDataset
from utils import collate_fn, update_train_running_results_dict, set_train_bar_description_dict, extract_index_blip_features, \
    save_model, generate_randomized_fiq_caption, device, backup_code
from validate_blip import compute_cirr_val_metrics, compute_fiq_val_metrics


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def clip_finetune_fiq(train_dress_types: List[str], val_dress_types: List[str], exp_name: str, save_dir: str, dataset: str,
                      num_epochs: int, blip_model_name: str, backbone: str, learning_rate: float, batch_size: int,
                      validation_frequency: int, transform: str, save_training: bool, save_best: bool, save_memory: bool, llm_sampler: bool,
                      light_model_name: str, use_adapt: bool, num_slots: int, loss_setting: List[str], **kwargs):
    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(save_dir) / dataset / f"{exp_name}_{training_start}"
    training_path.mkdir(exist_ok=False, parents=True)
    backup_code('./', training_path)
    # Save all the hyperparameters on a file
    with open(training_path / "configs.json", 'w') as file:
        json.dump(vars(args), file, sort_keys=True, indent=4)    
    print(f"save-memory-in: {save_memory}")

    blip_model, _, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type=backbone, is_eval=False, device=device, light_model_name=light_model_name, use_adapt=use_adapt, num_slots=num_slots, loss_setting=loss_setting)
    
    update_method = getattr(blip_model, '_update_f_former', None)
    if callable(update_method):
        blip_model._update_f_former()

    input_dim = 224

    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    idx_to_dress_mapping = {}
    relative_val_datasets = []
    classic_val_datasets = []

    # Define the validation datasets
    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)
        relative_val_datasets.append(relative_val_dataset)
        classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)
        classic_val_datasets.append(classic_val_dataset)

    # Define the train datasets and the combining function
    relative_train_dataset = FashionIQDataset('train', train_dress_types, 'relative', preprocess, llm_sampler)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=kwargs['num_workers'], pin_memory=False, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)

    # Define the optimizer, the loss and the grad scaler
    optimizer = optim.AdamW(
        [{'params': filter(lambda p: p.requires_grad, blip_model.parameters()), 'lr': learning_rate,
        'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay':0.05}])
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, pct_start=1.5/num_epochs, div_factor=100., steps_per_epoch=len(relative_train_loader), epochs=num_epochs)

    scaler = torch.cuda.amp.GradScaler()

    # When save_best == True initialize the best result to zero
    if save_best:
        best_avg_recall = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # Start with the training loop
    print('Training loop started')
    for epoch in range(num_epochs):
        train_running_results = {'images_in_epoch': 0}
        train_bar = tqdm(relative_train_loader, ncols=150)
        for idx, (reference_images, target_images, captions) in enumerate(train_bar):
            images_in_batch = reference_images.size(0)
            step = len(train_bar) * epoch + idx

            optimizer.zero_grad()

            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)

            # Randomize the training caption in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1 (d) cap2
            flattened_captions: list = np.array(captions).T.flatten().tolist()
            captions = generate_randomized_fiq_caption(flattened_captions)
            captions = [txt_processors["eval"](caption) for caption in captions]
            blip_model.train()
            # Extract the features, compute the logits and the loss
            with torch.cuda.amp.autocast():
                loss_dict = blip_model({"image":reference_images, "target":target_images, "text_input":captions})
                loss = 0.
                for key in loss_dict.keys():
                    loss += loss_dict[key]

            # Backpropagate and update the weights
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            update_train_running_results_dict(train_running_results, loss_dict, images_in_batch)
            set_train_bar_description_dict(train_bar, epoch, num_epochs, train_running_results)

        loss_log_dict = {'epoch': epoch}
        for key in train_running_results.keys():
            if key != 'images_in_epoch':
                loss_log_dict[key] = float(
            train_running_results[key] / train_running_results['images_in_epoch'])
            # Training CSV logging
        training_log_frame = pd.concat(
            [training_log_frame,
                pd.DataFrame(data=loss_log_dict, index=[0])])
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            blip_model.eval()
            recalls_at10 = []
            recalls_at50 = []

            # Compute and log validation metrics for each validation dataset (which corresponds to a different FashionIQ category)
            for relative_val_dataset, classic_val_dataset, idx in zip(relative_val_datasets, classic_val_datasets,
                                                                        idx_to_dress_mapping):
             
                index_features, index_names = extract_index_blip_features(classic_val_dataset, blip_model, save_memory)
                recall_at10, recall_at50 = compute_fiq_val_metrics(relative_val_dataset, blip_model,
                                                                    index_features, index_names, txt_processors, save_memory)
                
                recalls_at10.append(recall_at10)
                recalls_at50.append(recall_at50)
                torch.cuda.empty_cache()

            results_dict = {}
            for i in range(len(recalls_at10)):
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]
            results_dict.update({
                f'average_recall_at10': mean(recalls_at10),
                f'average_recall_at50': mean(recalls_at50),
                f'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2
            })

            print(json.dumps(results_dict, indent=4))
          
            # Validation CSV logging
            log_dict = {'epoch': epoch}
            log_dict.update(results_dict)
            validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
            validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

            if save_training:
                if save_best and results_dict['average_recall'] > best_avg_recall:
                    best_avg_recall = results_dict['average_recall']
                    save_model('tuned_clip_best', epoch, blip_model, training_path)
                else:
                    save_model('tuned_clip_last', epoch, blip_model, training_path)


def clip_finetune_cirr(num_epochs: int, exp_name: str, dataset: str, blip_model_name: str, backbone: str, learning_rate: float, batch_size: int,
                       validation_frequency: int, transform: str, save_training: bool, save_best: bool, save_dir: str, llm_sampler: bool, 
                       light_model_name: str, use_adapt: bool, num_slots: int, loss_setting: List[str], **kwargs):
    """
    Fine-tune CLIP on the CIRR dataset using as combining function the image-text element-wise sum
    :param num_epochs: number of epochs
    :param blip_model_name: BLIP model you want to use: "RN50", "RN101", "RN50x4"...
    :param learning_rate: fine-tuning learning rate
    :param batch_size: batch size
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use. Should be in ['clip', 'squarepad', 'targetpad']. When
                targetpad is also required to provide `target_ratio` kwarg.
    :param save_training: when True save the weights of the Combiner network
    :param encoder: which CLIP encoder to fine-tune, should be in ['both', 'text', 'image']
    :param save_best: when True save only the weights of the best Combiner w.r.t three different averages of the metrics
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio`    
    :return: None
    """
    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(save_dir) / dataset / f"{exp_name}_{training_start}"
    training_path.mkdir(exist_ok=False, parents=True)
    backup_code('./', training_path)

    # Save all the hyperparameters on a file
    with open(training_path / "configs.json", 'w') as file:
        json.dump(vars(args), file, sort_keys=True, indent=4)

    blip_model, _, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type=backbone, is_eval=False, device=device, light_model_name=light_model_name, use_adapt=use_adapt, num_slots=num_slots, loss_setting=loss_setting)
    
    update_method = getattr(blip_model, '_update_f_former', None)
    if callable(update_method):
        blip_model._update_f_former()

    # clip_model.eval().float()  
    input_dim = 224

    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with ratio={target_ratio} preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    # Define the validation datasets
    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)

    # When fine-tuning only the text encoder we can precompute the index features since they do not change over the epochs
    # Define the train dataset and the comb ining function
    relative_train_dataset = CIRRDataset('train', 'relative', preprocess, llm_sampler)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, 
                                       batch_size=batch_size,
                                       num_workers=kwargs['num_workers'], 
                                       pin_memory=False, 
                                       collate_fn=collate_fn,
                                       drop_last=True, 
                                       shuffle=True)

    # Define the optimizer, the loss and the grad scaler
    optimizer = optim.AdamW([{'params': filter(lambda p: p.requires_grad, blip_model.parameters()), 'lr': learning_rate,
          'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay':0.05}])
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, pct_start=1/50, steps_per_epoch=len(relative_train_loader), epochs=num_epochs)
    scaler = torch.cuda.amp.GradScaler()

    if save_best:
        best_arithmetic = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        train_running_results = {'images_in_epoch': 0}
        train_bar = tqdm(relative_train_loader, ncols=150)
        for _, (reference_images, target_images, captions) in enumerate(train_bar):
            images_in_batch = reference_images.size(0)
            optimizer.zero_grad()

            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)

            captions = [txt_processors["eval"](caption) for caption in captions]
            
            blip_model.train()

            # Extract the features, compute the logits and the loss
            with torch.cuda.amp.autocast():
                loss_dict = blip_model({"image":reference_images, "target":target_images, "text_input":captions}) #"tar_input": tar_captions
                loss = 0.
                for key in loss_dict.keys():
                    if key != 'loss_itc':
                        loss += kwargs[key] * loss_dict[key]
                    else:
                        loss += loss_dict[key]

            # Backpropagate and update the weights
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            update_train_running_results_dict(train_running_results, loss_dict, images_in_batch)
            set_train_bar_description_dict(train_bar, epoch, num_epochs, train_running_results)


        loss_log_dict = {'epoch': epoch}
        for key in train_running_results.keys():
            if key != 'images_in_epoch':
                loss_log_dict[key] = float(
            train_running_results[key] / train_running_results['images_in_epoch'])
            # Training CSV logging
        training_log_frame = pd.concat([training_log_frame, pd.DataFrame(data=loss_log_dict, index=[0])])
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            blip_model.eval()
            val_index_features, val_index_names = extract_index_blip_features(classic_val_dataset, blip_model)
            results = compute_cirr_val_metrics(relative_val_dataset, blip_model, val_index_features,
                                                val_index_names, txt_processors)
            group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = results

            results_dict = {
                'group_recall_at1': group_recall_at1,
                'group_recall_at2': group_recall_at2,
                'group_recall_at3': group_recall_at3,
                'recall_at1': recall_at1,
                'recall_at5': recall_at5,
                'recall_at10': recall_at10,
                'recall_at50': recall_at50,
                'mean(R@5+R_s@1)': (group_recall_at1 + recall_at5) / 2,
                'arithmetic_mean': mean(results),
                'harmonic_mean': harmonic_mean(results),
                'geometric_mean': geometric_mean(results)
            }
            print(json.dumps(results_dict, indent=4))
            # Validation CSV logging
            log_dict = {'epoch': epoch}
            log_dict.update(results_dict)
            validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
            validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

            if save_training:
                if save_best and results_dict['arithmetic_mean'] > best_arithmetic:
                    best_arithmetic = results_dict['arithmetic_mean']
                    save_model('tuned_clip_arithmetic_best', epoch, blip_model, training_path)
                else:
                    # save last
                    save_model('tuned_clip_arithmetic_last', epoch, blip_model, training_path)
       

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--exp-name", type=str, help="Name of the current experiment", default='exp')
    parser.add_argument("--dataset", type=str, required=True, choices=["cirr", "fashioniq"])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-epochs", default=300, type=int, help="number training epochs")
    parser.add_argument("--blip-model-name", default="blip2_cir_cat", type=str, help="[blip2_cir_cat, blip2_cir, blip2_cir_align_prompt]")
    parser.add_argument("--backbone", type=str, default="pretrain", help="pretrain for vit-g, pretrain_vitL for vit-l")
    parser.add_argument("--learning-rate", default=2e-6, type=float, help="Learning rate")
    parser.add_argument("--batch-size", default=512, type=int, help="Batch size")
    parser.add_argument("--seed", default=42, type=int)
    
    # loss loss_itc is set to 1. by default
    parser.add_argument("--loss-dta", default=0.4, type=float)
    parser.add_argument("--loss-rtc", default=0.4, type=float)

    parser.add_argument("--validation-frequency", default=1, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str, help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--save-dir", type=str, default='./output', help="Save path of the experiment")
    parser.add_argument("--save-training", action='store_true', help="Whether save the training model")
    parser.add_argument("--save-best", action='store_true', help="Save only the best model during training")
    parser.add_argument("--save-memory", action='store_true', help="Memory effificent mode")
    parser.add_argument("--llm-sampler", action='store_true', help="Whether to use diversified llm sampler")

    # lightweight config
    parser.add_argument("--light-model-name", default='efficientnet-b0', type=str)
    parser.add_argument("--use-adapt", action='store_true')
    parser.add_argument("--num-slots", default=8, type=int)
    parser.add_argument("--loss-setting", default=['itc', 'dta'], type=str, nargs='+')

    args = parser.parse_args()
    # set_seed(args.seed)

    training_hyper_params = {
        "num_epochs": args.num_epochs,
        "exp_name": args.exp_name,
        "save_dir": args.save_dir,
        "dataset": args.dataset,
        "num_workers": args.num_workers,
        "blip_model_name": args.blip_model_name,
        "backbone": args.backbone,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "target_ratio": args.target_ratio,
        "save_training": args.save_training,
        "save_best": args.save_best,
        "loss_rtc": args.loss_rtc,
        "loss_dta": args.loss_dta,
        "save_memory": args.save_memory,
        "llm_sampler": args.llm_sampler,
        "light_model_name": args.light_model_name,
        "num_slots": args.num_slots,
        "use_adapt": args.use_adapt,
        "loss_setting": args.loss_setting
    }
    
    if args.dataset == 'cirr':
        clip_finetune_cirr(**training_hyper_params)
    else:
        training_hyper_params.update({
                                        'train_dress_types': ['dress', 'toptee', 'shirt'], 
                                        'val_dress_types': ['dress', 'toptee', 'shirt']
                                    })
        clip_finetune_fiq(**training_hyper_params)