"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    disabled_train
)
from lavis.models.blip_models.blip_outputs import BlipOutputFeatures
from lavis.models.slots_module.multi_head_slot_attention import SlotFusion
from efficientnet_pytorch import EfficientNet
from EfficientViT.classification.model.build import EfficientViT_M4, EfficientViT_M2
from mobilenet.mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@registry.register_model("blip2_light_cir")
class Blip2LightCir(Blip2Base):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        # ViT config
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        # query and cross att
        num_query_token=32,
        cross_attention_freq=2,
        # compute similarity
        embed_dim=256,
        max_txt_len=32,
        # lightweight visual encoder
        light_model_name="efficientnet-b0",
        # slot config
        num_slots=8,
        use_adapt=False,
        # loss config
        loss_setting=['itc', 'dta'],
    ):
        super().__init__()
        self.tokenizer = self.init_tokenizer()

        self.light_visual_encoder = self.init_light_vision_encoder(light_model_name, vit_precision)
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision)

        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, 
                                                            self.visual_encoder.num_features, 
                                                            cross_attention_freq)
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])
        
        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.max_txt_len = max_txt_len
               
        # slot initialization
        self.num_slots, self.use_adapt = num_slots, use_adapt
        self.fusion_slot = SlotFusion(num_slots=num_slots, 
                                      input_dim=self.visual_width, 
                                      emb_dim=self.Qformer.config.hidden_size,
                                      use_adapt=use_adapt)

        # loss settings
        self.loss_setting = loss_setting
        

    def forward(self, samples):
        image = samples["image"]
        target = samples["target"]
        text = samples["text_input"]
        loss_dict = dict()

        ###============== Query Side Feature Extraction ===================###
        image_embeds = self.light_visual_encoder.extract_features(image)
        b, c, h, w = image_embeds.shape
        image_embeds = image_embeds.reshape(b, h*w, c) # B,L,D

        if self.use_adapt:
            fusion_output, fusion_weight = self.fusion_slot(image_embeds)
            fusion_attn = fusion_weight
        else:
            fusion_output = self.fusion_slot(image_embeds)
            fusion_attn = torch.ones(fusion_output.size()[:-1], dtype=torch.long).to(self.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)

        fusion_mask = torch.cat([fusion_attn, text_tokens.attention_mask], dim=1)
        text_output = self.Qformer.bert(
            input_ids=text_tokens.input_ids,
            query_embeds= fusion_output,
            attention_mask=fusion_mask,
            return_dict=True
        )

        fusion_feats = F.normalize(self.text_proj(text_output.last_hidden_state[:, self.num_slots, :]), dim=-1)

        ###============== Lightweight Student Contrastive Loss ===================###
        taregt_embeds = self.ln_vision(self.visual_encoder(target))
        target_atts = torch.ones(taregt_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        ) 
        target_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=taregt_embeds,
            encoder_attention_mask=target_atts,
            use_cache=True,
            return_dict=True,
        )

        target_feats = F.normalize(
            self.vision_proj(target_output.last_hidden_state), dim=-1
        )
        sim_t2q = torch.matmul(
            fusion_feats.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
        ).squeeze()

        sim_i2t, _ = sim_t2q.max(-1)
        sim_i2t = sim_i2t / self.temp
        bs = image.size(0)
        targets = torch.linspace(0,  bs - 1, bs, dtype=int).to(image.device)

        loss_itc = F.cross_entropy(sim_i2t, targets)
        assert 'itc' in self.loss_setting, "The loss must contains a contrastive term"
        loss_dict['loss_itc'] = loss_itc

        ###============== Teacher Contrastive Loss (Q-former) ===================###
        if 'dta' in self.loss_setting:
            image_anchors = self.ln_vision(self.visual_encoder(image))
            image_anchors_att = torch.ones(image_anchors.size()[:-1], dtype=torch.long).to(image.device)

            attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
            anchor_output = self.Qformer.bert(
                input_ids=text_tokens.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_anchors,
                encoder_attention_mask=image_anchors_att,
                return_dict=True,
            )

            anchors = anchor_output.last_hidden_state[:, :query_tokens.size(1), :]
            anchors_for = self.Qformer.bert(
                input_ids=text_tokens.input_ids,
                query_embeds=anchors,
                attention_mask=attention_mask,
                return_dict=True
            )

            anchor_feats = F.normalize(self.text_proj(anchors_for.last_hidden_state[:, query_tokens.size(1), :]), dim=-1)

            sim_r2t = torch.matmul(anchor_feats.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)).squeeze()

            sim_r2t, _ = sim_r2t.max(-1)
            sim_r2t = sim_r2t / self.temp

            # teacher constrastive loss
            loss_rtc = F.cross_entropy(sim_r2t, targets)

            # distillation alignment (dta) loss
            loss_dta = F.mse_loss(fusion_output.mean(1), anchors.mean(1))

            loss_dict['loss_dta'] = loss_dta
            loss_dict['loss_rtc'] = loss_rtc

        return loss_dict


    def init_light_vision_encoder(self, light_model_name, precision):
        assert light_model_name in ["efficientnet-b0", "efficientnet-b1", "efficientnet-b2", 
                                "efficientvit-m4", "efficientvit-m2", "mobilenetv3-s", "mobilenetv3-l"]
        
        if light_model_name.startswith("efficientnet"):
            visual_encoder = EfficientNet.from_pretrained(light_model_name)
        elif light_model_name == 'efficientvit-m4':
            visual_encoder = EfficientViT_M4(pretrained='efficientvit_m4')
        elif light_model_name == 'efficientvit-m2':
            visual_encoder = EfficientViT_M2(pretrained='efficientvit_m2')
        elif light_model_name == 'mobilenetv3-s':
            visual_encoder = MobileNetV3_Small()
            visual_encoder.load_state_dict(torch.load("mobilenet/450_act3_mobilenetv3_small.pth", map_location='cpu'))
        elif light_model_name == 'mobilenetv3-l':
            visual_encoder = MobileNetV3_Large()
            visual_encoder.load_state_dict(torch.load("mobilenet/450_act3_mobilenetv3_large.pth", map_location='cpu'))
        else:
            raise NotImplementedError(f"The model {light_model_name} is not implemented yet.")

        if light_model_name.endswith(("b0", "b1")):
            self.visual_width = 1280
        elif light_model_name.endswith("b2"):
            self.visual_width = 1408
        elif light_model_name.endswith("m4"):
            self.visual_width = 384
        elif light_model_name.endswith("m2"):
            self.visual_width = 224
        elif light_model_name.endswith("s"):
            self.visual_width = 576
        elif light_model_name.endswith("l"):
            self.visual_width = 960

        self.light_model_name = light_model_name
        
        return visual_encoder
    

    @torch.no_grad()
    def inference(self, reference_embeds, target_feats, text):
        if self.use_adapt:
            fusion_output, fusion_weight = self.fusion_slot(reference_embeds)
            fusion_attn = fusion_weight
        else:
            fusion_output = self.fusion_slot(reference_embeds)
            fusion_attn = torch.ones(fusion_output.size()[:-1], dtype=torch.long).to(self.device)
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(reference_embeds.device)

        fusion_mask = torch.cat([fusion_attn, text_tokens.attention_mask], dim=1)
        text_output = self.Qformer.bert(
            input_ids=text_tokens.input_ids,
            query_embeds=fusion_output,
            attention_mask=fusion_mask,
            return_dict=True
        )

        fusion_feats = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, self.num_slots, :]), dim=-1
        )


        sim_t2q = torch.matmul(
            fusion_feats.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_i2t, _ = sim_t2q.max(-1)
        return sim_i2t
    

    @torch.no_grad()
    def extract_target_features(self, image, mode='mean'):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long
        ).to(self.device)
        query_tokens = self.query_tokens.expand(
            image_embeds_frozen.shape[0], -1, -1
        )

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_embeds = query_output.last_hidden_state

        # return image_embeds
        image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        reference_image_embeds = self.light_visual_encoder.extract_features(image)
        b, c, h, w = reference_image_embeds.shape
        reference_image_embeds = reference_image_embeds.reshape(b, h*w, c) # B,L,D

        return image_features, reference_image_embeds


    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        # lightweight model config
        light_model_name = cfg.get("light_model_name", "efficientnet-b0")
        use_adapt = cfg.get("use_adapt", False)
        num_slots = cfg.get("num_slots", 8)
        loss_setting = cfg.get("loss_setting", ['itc', 'dta'])

        print(f"Model Config: {cfg}")
        
        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
            light_model_name=light_model_name,
            use_adapt=use_adapt,
            num_slots=num_slots,
            loss_setting=loss_setting
        )
        model.load_checkpoint_from_config(cfg)

        return model