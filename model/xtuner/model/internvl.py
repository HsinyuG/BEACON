# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import torch
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import GenerationConfig
import math

from xtuner.registry import BUILDER

from .utils import (
    find_all_linear_names,
    get_peft_model_state_dict,
    guess_load_checkpoint,
    make_inputs_require_grad,
)

from xtuner.dataset.grounding_dino_pipeline import BoxPipeline

class InternVL_V1_5(BaseModel):
    def __init__(
        self,
        model_path,
        freeze_llm=False,
        freeze_visual_encoder=False,
        llm_lora=None,
        visual_encoder_lora=None,
        quantization_vit=False,
        quantization_llm=False,
        pretrained_pth=None,
    ):
        print_log("Start to load InternVL_V1_5 model.", logger="current")
        super().__init__()
        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.use_llm_lora = llm_lora is not None
        self.use_visual_encoder_lora = visual_encoder_lora is not None
        self.quantization_vit = quantization_vit
        self.quantization_llm = quantization_llm
        if quantization_vit:
            assert visual_encoder_lora is not None
        if quantization_llm:
            assert quantization_llm and llm_lora is not None

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if config.llm_config.model_type == "internlm2":
            config.llm_config.attn_implementation = "flash_attention_2"
        else:
            config.llm_config._attn_implementation = "flash_attention_2"

        if quantization_vit is False and quantization_llm is False:
            quantization = None
        else:
            llm_int8_skip_modules = ["mlp1"]
            if quantization_llm and not quantization_vit:
                llm_int8_skip_modules.append("vision_model")

            if quantization_vit and not quantization_llm:
                llm_int8_skip_modules.append("language_model")

            quantization_config = dict(
                type=BitsAndBytesConfig,
                llm_int8_skip_modules=llm_int8_skip_modules,
                load_in_4bit=True,
                load_in_8bit=False,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            quantization_clazz = quantization_config.pop("type")
            quantization = quantization_clazz(**quantization_config)

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization,
            config=config,
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.model.img_context_token_id = img_context_token_id

        if self.freeze_llm:
            self.model.language_model.requires_grad_(False)
        if self.freeze_visual_encoder:
            self.model.vision_model.requires_grad_(False)

        if hasattr(self.model.language_model, "enable_input_require_grads"):
            self.model.language_model.enable_input_require_grads()
        else:
            self.model.language_model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )

        self.gradient_checkpointing_enable()

        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora)

        if self.use_visual_encoder_lora:
            self._prepare_visual_encoder_for_lora(visual_encoder_lora)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f"Load pretrained weight from {pretrained_pth}")

        self._count = 0
        print_log(self, logger="current")
        print_log("InternVL_V1_5 construction is complete", logger="current")

    def _parse_lora_config(self, lora_config):
        if (
            isinstance(lora_config, dict)
            or isinstance(lora_config, Config)
            or isinstance(lora_config, ConfigDict)
        ):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self, lora_config, use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.model.language_model = prepare_model_for_kbit_training(
            self.model.language_model, use_activation_checkpointing
        )
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.model.language_model)
            lora_config.target_modules = modules
        self.model.language_model = get_peft_model(
            self.model.language_model, lora_config
        )

    def _prepare_visual_encoder_for_lora(self, lora_config):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.model.vision_model)
            lora_config.target_modules = modules
        self.model.vision_model = get_peft_model(self.model.vision_model, lora_config)

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.model.language_model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.model.language_model.gradient_checkpointing_disable()

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.model.vision_model, state_dict=state_dict
                )
            )
        elif not self.freeze_visual_encoder:
            to_return.update(
                {k: v for k, v in state_dict.items() if "model.vision_model." in k}
            )
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.model.language_model, state_dict=state_dict
                )
            )
        elif not self.freeze_llm:
            to_return.update(
                {k: v for k, v in state_dict.items() if "model.language_model." in k}
            )
        # Step 3. Projector
        to_return.update({k: v for k, v in state_dict.items() if "model.mlp1." in k})
        return to_return

    def init_weights(self):
        pass

    def forward(self, data, data_samples=None, mode="loss"):
        pixel_values = data["pixel_values"]

        if type(pixel_values) is list or pixel_values.ndim == 5:
            if type(pixel_values) is list:
                pixel_values = [
                    x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                ]
            # b*n, c, h, w
            concat_images = torch.cat(
                [image.to(self.model.vision_model.dtype) for image in pixel_values],
                dim=0,
            )
        else:
            raise NotImplementedError()

        input_ids = data["input_ids"]
        position_ids = data["position_ids"]
        attention_mask = data["attention_mask"]
        # sum is 0 are text
        image_flags = torch.sum(concat_images, dim=(1, 2, 3)) != 0
        image_flags = image_flags.long()

        labels = data["labels"]
        use_cache = False

        # Directly calling this code in LORA fine-tuning
        # will result in an error,so we must rewrite it.
        # TODO: Once the official is fixed, we can remove it.
        # outputs = self.model(input_ids=input_ids,
        #                      position_ids=position_ids,
        #                      attention_mask=attention_mask,
        #                      image_flags=image_flags,
        #                      pixel_values=concat_images,
        #                      labels=labels,
        #                      use_cache=use_cache)
        outputs = self._llm_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            pixel_values=concat_images,
            labels=labels,
            use_cache=use_cache,
        )
        loss_dict = {"loss": outputs.loss}
        return loss_dict

    def _llm_forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = (
            return_dict
            if return_dict is not None
            else self.model.config.use_return_dict
        )

        image_flags = image_flags.squeeze(-1)
        # We only added the clone code here to avoid the error.
        input_embeds = self.model.language_model.get_input_embeddings()(
            input_ids
        ).clone()

        vit_embeds = self.model.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        # if torch.distributed.get_rank() == 0 and self._count % 100 == 0:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        if rank == 0 and self._count % 100 == 0:
            print(
                f"dynamic ViT batch size: {vit_batch_size}, "
                f"images per sample: {vit_batch_size / B}, "
                f"dynamic token length: {N}"
            )
        self._count += 1

        input_ids = input_ids.reshape(B * N)
        selected = input_ids == self.model.img_context_token_id
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(
                -1, C
            )
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(
                f"warning: {e}, input_embeds[selected].shape="
                f"{input_embeds[selected].shape}, "
                f"vit_embeds.shape={vit_embeds.shape}"
            )
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(
                -1, self.model.language_model.config.vocab_size
            )
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

import torch.nn as nn
from torch.cuda.amp import autocast
import re

class InternVL_V1_5_NavTarget(InternVL_V1_5):
    def __init__(
        self,
        model_path,
        freeze_llm=False,
        freeze_visual_encoder=False,
        llm_lora=None,
        visual_encoder_lora=None,
        quantization_vit=False,
        quantization_llm=False,
        pretrained_pth=None,
        reg_weight=0.5,
        token_ce_weight=1.0,
        use_meta_action=False,
        use_pred_meta_action_onehot=False,
        fuse_visual_tokens=False,
        output_mode="VQA",
        output_head=None,
        use_pe3d=False,
        num_views=4,
        region_cls_freq=None,
        finetune_head=None,
    ):
        print_log("Start to load InternVL_V1_5 model.", logger="current")
        BaseModel.__init__(self)  # bypass parent’s init
        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.use_llm_lora = llm_lora is not None
        self.use_visual_encoder_lora = visual_encoder_lora is not None
        self.quantization_vit = quantization_vit
        self.quantization_llm = quantization_llm
        if quantization_vit:
            assert visual_encoder_lora is not None
        if quantization_llm:
            assert quantization_llm and llm_lora is not None

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if config.llm_config.model_type == "internlm2":
            config.llm_config.attn_implementation = "flash_attention_2"
        else:
            config.llm_config._attn_implementation = "flash_attention_2"

        if quantization_vit is False and quantization_llm is False:
            quantization = None
        else:
            llm_int8_skip_modules = ["mlp1"]
            if quantization_llm and not quantization_vit:
                llm_int8_skip_modules.append("vision_model")

            if quantization_vit and not quantization_llm:
                llm_int8_skip_modules.append("language_model")

            quantization_config = dict(
                type=BitsAndBytesConfig,
                llm_int8_skip_modules=llm_int8_skip_modules,
                load_in_4bit=True,
                load_in_8bit=False,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            quantization_clazz = quantization_config.pop("type")
            quantization = quantization_clazz(**quantization_config)

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization,
            config=config,
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.model.img_context_token_id = img_context_token_id

        if self.freeze_llm:
            self.model.language_model.requires_grad_(False)
        if self.freeze_visual_encoder:
            self.model.vision_model.requires_grad_(False)

        if hasattr(self.model.language_model, "enable_input_require_grads"):
            self.model.language_model.enable_input_require_grads()
        else:
            self.model.language_model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )

        self.gradient_checkpointing_enable()

        # # resize input embed before add llm lora
        self.added_special_token = False
        if tokenizer is not None:
            self.tokenizer = tokenizer
            # tokenizer_type = self.tokenizer['type']
            # del self.tokenizer['type']
            # self.tokenizer = tokenizer_type(**self.tokenizer)
            self._add_special_tokens() # NOTE: ablate this in VQA only and got better with this activated

        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora, add_special_tokens=self.added_special_token)
        else:
            raise NotImplementedError("LORA must be used in LLM for this model.")

        if self.use_visual_encoder_lora:
            self._prepare_visual_encoder_for_lora(visual_encoder_lora)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f"Load pretrained weight from {pretrained_pth}")

        # new added head
        hidden_size = self.model.language_model.config.hidden_size
        head_in = hidden_size + 16 if use_meta_action else hidden_size
        head_in = head_in + hidden_size if fuse_visual_tokens else head_in
        
        self.tgt_head = nn.Sequential(
            nn.Linear(head_in, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2),
        )

        self.tgt_head_2 = nn.Sequential(
            nn.Linear(head_in, 1024),
            nn.ReLU(),
            nn.Linear(1024, 16),
        )

        self.tgt_head_3 = nn.Sequential(
            nn.Linear(head_in, 1024),
            nn.ReLU(),
            nn.Linear(1024, 8),
        )

        self.tgt_head_polar = nn.Sequential(
            nn.Linear(head_in, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3),
        )

        self.tgt_head_waypoints = nn.Sequential(
            nn.Linear(head_in, 1024),
            nn.ReLU(),
            nn.Linear(1024, 12),  # predict 6 waypoints (x,y)
        )

        self.reg_weight = reg_weight
        self.token_ce_weight = token_ce_weight
        self.use_meta_action = use_meta_action
        self.fuse_visual_tokens = fuse_visual_tokens
        self.use_pred_meta_action_onehot = use_pred_meta_action_onehot
        self.output_mode = output_mode

        self.output_head = None
        if output_mode == "LSSAttn":
            self.output_head = BUILDER.build(output_head)
            assert self.output_head is not None, "output_head must be provided for LSSAttn mode"
        # end of new added head

        self.finetune_head = None
        if finetune_head is not None:
            self.finetune_head = BUILDER.build(finetune_head)
            assert self.output_mode == "Affordance", "finetune_head is only for Affordance mode"

        # # for positional embedding
        # self.use_pe3d = use_pe3d
        # if self.use_pe3d:
        #     self.num_views = num_views
        #     self.view_embed = nn.Embedding(num_views, hidden_size)

        # 3D positional encoding (LLaVA-3D style learned MLP; no voxel pooling)
        self.use_pe3d = use_pe3d
        self.invalid_depth_val = 10.0  # your dataset convention

        if self.use_pe3d:
            # pre-projector dim = input dim of InternVL projector (mlp1)
            # InternVL HF: mlp1 is nn.Sequential(LN, Linear, GELU, Linear)
            preproj_dim = self.model.mlp1[1].in_features

            self.pos3d_mlp = nn.Sequential(
                nn.Linear(3, preproj_dim),
                nn.LayerNorm(preproj_dim),
                nn.ReLU(),
                nn.Linear(preproj_dim, preproj_dim),
            )

            # match LLaVA-3D's Xavier init style
            for p in self.pos3d_mlp.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)


        # for oracle bev (lazy: the pkl is only read if run_sample() is called,
        # which happens solely in the optional freespace branch)
        from xtuner.perception_modules.oracle_bev import BEVMaskGenerator
        self.oracle_bev_generator = BEVMaskGenerator(
            ann_file=os.environ.get(
                "BEACON_ORACLE_BEV_ANN",
                "/path/to/Falcon/data/captures_v3_val_<scan>/scan<scan>_Landmark-RxR-dynamic.pkl",
            ),
            out_dir=os.environ.get(
                "BEACON_ORACLE_BEV_OUT",
                "/path/to/generate_bev_masks/traversable_masks",
            ),
            outer_half_xy=6.4,
            inner_half_xy=3.6,
            outer_pitch=0.2,
            inner_pitch=0.1,
            z_up=0.4,
            z_down=2.0,
            clearance_band=(-1.4, 0.0),
            support_band=(-1.6, -1.4),
            connectivity=4,
            corner_check=True,
            inner_vote_min=2,
        )
        # end of oracle bev

        # class balance
        if region_cls_freq is not None:
            counts = torch.tensor([v for v in region_cls_freq.values()], dtype=torch.float)  # order confirmed
            weights = 1.0 / torch.log(counts + 1e-3)
            weights = weights / weights.mean()
            self.class_weights = weights.tolist()
        else:
            self.class_weights = None

        self._log_trainable()

        # box pipeline
        # self.box_pipeline = BoxPipeline()

        self._count = 0
        print_log(self, logger="current")
        print_log("InternVL_V1_5 construction is complete", logger="current")

    def _prepare_llm_for_lora(self, lora_config, use_activation_checkpointing=True, add_special_tokens=False):
        lora_config = self._parse_lora_config(lora_config)
        self.model.language_model = prepare_model_for_kbit_training(
            self.model.language_model, use_activation_checkpointing
        )
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.model.language_model)
            lora_config.target_modules = modules
        self.model.language_model = get_peft_model(
            self.model.language_model, lora_config
        )
        if not add_special_tokens:
            return
        else:
            print("Unfreezing input embedding and lm_head for new special tokens...")
        # newly added to unfreeze the new tokens id to embedding and lm_head
        # for name, param in self.named_parameters():
        #     if 'tok_' in name or 'lm_head' in name:
        #         print("Unfrozen {} !!!".format(name))
        #         param.requires_grad_(True)
        #     if 'output.' in name and 'llm' in name and 'lora' not in name:
        #         print("Unfrozen {} !!!".format(name))
        #         param.requires_grad_(True)
        lm = self.model.language_model
        # input embeddings
        in_emb = lm.get_input_embeddings()
        in_emb.weight.requires_grad_(True)

        # output embeddings / lm head (may be lora-wrapped)
        out_emb = lm.get_output_embeddings()
        # some models expose .weight, some are wrapped and expose base_layer.weight
        if hasattr(out_emb, "weight"):
            out_emb.weight.requires_grad_(True)
        elif hasattr(out_emb, "base_layer") and hasattr(out_emb.base_layer, "weight"):
            out_emb.base_layer.weight.requires_grad_(True)

    def _add_special_tokens(self):
        assert hasattr(self, "tokenizer")

        target_tokens = ['[TGT]']
        region_tokens = [
            # '[FRONT_SMALL]',
            # '[FRONT_BIG]',
            # '[FRONTLEFT_SMALL]',
            # '[FRONTLEFT_BIG]',
            # '[LEFT_SMALL]',
            # '[LEFT_BIG]',
            # '[BACKLEFT_SMALL]',
            # '[BACKLEFT_BIG]',
            # '[BACK_SMALL]',
            # '[BACK_BIG]',
            # '[BACKRIGHT_SMALL]',
            # '[BACKRIGHT_BIG]',
            # '[RIGHT_SMALL]',
            # '[RIGHT_BIG]',
            # '[FRONTRIGHT_SMALL]',
            # '[FRONTRIGHT_BIG]',
        ]
        special_tokens = target_tokens + region_tokens
        self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.tgt_token_idx = self.tokenizer("[TGT]", add_special_tokens=False).input_ids[0]
        self.region_token_idxs = []
        for rt in region_tokens:
            idx = self.tokenizer(rt, add_special_tokens=False).input_ids[0]
            self.region_token_idxs.append(idx)

        self.model.language_model.resize_token_embeddings(len(self.tokenizer))

        # # it should already have the grad by enable_input_require_grads() we just double confirm this
        # for p in self.model.language_model.get_input_embeddings().parameters():
        #     p.requires_grad_(True)
        # # optional sanity check
        # assert self.model.language_model.get_input_embeddings().weight.requires_grad.all()
        self.added_special_token = True
        print(f"[TGT]: {self.tgt_token_idx}")
        print(f"Region tokens: {self.region_token_idxs}")
        print('****************************Add special tokens ********************************************')
        return

    def _log_trainable(self):
        emb_grad = self.model.language_model.get_input_embeddings().weight.requires_grad
        head_grad = self.model.language_model.get_output_embeddings().weight.requires_grad
        lora_trainable = [n for n, p in self.model.language_model.named_parameters() if "lora_" in n and p.requires_grad]
        non_lora_trainable = [n for n, p in self.model.language_model.named_parameters() if p.requires_grad and "lora_" not in n]

        print_log(
            f"[NavTarget] emb_grad={emb_grad}, head_grad={head_grad}, "
            f"trainable non-LoRA={len(non_lora_trainable)}, trainable LoRA={len(lora_trainable)}",
            logger="current",
        )

    def state_dict(self, *args, **kwargs): # tested works
        base_full = nn.Module.state_dict(self, *args, **kwargs)  # raw state with tgt_head
        filtered = super().state_dict(*args, **kwargs)    # current LoRA/projector filter
        # for k, v in base_full.items():
        #     if k.startswith("tgt_head."):
        #         filtered[k] = v
        for k, v in base_full.items():
            if not k.startswith("model.") and not k.startswith("data_preprocessor."):
                # everything outside model/data_preprocessor, e.g., tgt_head, output_head
                filtered[k] = v
        tok_key = "model.language_model.base_model.model.model.tok_embeddings.weight"
        out_key = "model.language_model.base_model.model.output.base_layer.weight"
        if tok_key in base_full:
            filtered[tok_key] = base_full[tok_key]
        else:
            raise KeyError(f"Cannot find {tok_key} in base_full state dict.")
        if out_key in base_full:
            filtered[out_key] = base_full[out_key]
        else:
            raise KeyError(f"Cannot find {out_key} in base_full state dict.")
        return filtered



    def _prompt_len_from_labels(self, labels, ignore_index=-100):
        # labels: [1, T]
        idx = (labels[0] != ignore_index).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            raise ValueError("No answer tokens in labels; cannot truncate for AR.")
        return idx[0].item()
    
    def _truncate_to_prompt(self, input_ids, attention_mask, position_ids, labels):
        prompt_len = self._prompt_len_from_labels(labels)
        # sanity: answer part should match input_ids for VQA single-turn
        assert torch.equal(input_ids[:, prompt_len:], labels[:, prompt_len:]), (
            "Input IDs and labels do not match in answer part during eval."
        )
        return (
            input_ids[:, :prompt_len],
            attention_mask[:, :prompt_len],
            position_ids[:, :prompt_len],
            prompt_len,
        )

    def _dir_ids_8_from_views(
        self, batch_size: int, fH: int, fW: int, device: torch.device
    ) -> torch.Tensor:
        # dir order: FRONT, FRONTLEFT, LEFT, BACKLEFT, BACK, BACKRIGHT, RIGHT, FRONTRIGHT
        assert fW % 4 == 0
        q = fW // 4
        h = fW // 2

        lq = slice(0, q)
        mid = slice(q, q + h)
        rq = slice(fW - q, fW)

        dir_ids = torch.empty((4, fH, fW), device=device, dtype=torch.long)

        # view 0: front
        dir_ids[0, :, lq] = 1
        dir_ids[0, :, mid] = 0
        dir_ids[0, :, rq] = 7
        # view 1: left
        dir_ids[1, :, lq] = 3
        dir_ids[1, :, mid] = 2
        dir_ids[1, :, rq] = 1
        # view 2: back
        dir_ids[2, :, lq] = 5
        dir_ids[2, :, mid] = 4
        dir_ids[2, :, rq] = 3
        # view 3: right
        dir_ids[3, :, lq] = 7
        dir_ids[3, :, mid] = 6
        dir_ids[3, :, rq] = 5

        dir_ids = dir_ids.unsqueeze(0).expand(batch_size, -1, -1, -1)
        return dir_ids.reshape(batch_size * 4, fH * fW)

    def _extract_feature_preproj(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Copy of HF InternVL extract_feature() but stops BEFORE self.model.mlp1."""
        if self.model.select_layer == -1:
            vit_embeds = self.model.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state
        else:
            vit_embeds = self.model.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[self.model.select_layer]

        vit_embeds = vit_embeds[:, 1:, :]  # drop CLS

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.model.pixel_shuffle(vit_embeds, scale_factor=self.model.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])  # (B*V, T, C_preproj)
        return vit_embeds


    def _xyz_ego_tokens(
        self,
        depth: torch.Tensor,        # (B, V, H, W) meters, invalid==10
        intrinsics: torch.Tensor,   # (B, V, 3, 3)
        cam2egos: torch.Tensor,     # (B, V, 4, 4) cam->ego
        token_len: int,             # T (e.g. 256)
        out_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
        xyz_flat:     (B*V, T, 3) ego coords (meters)
        invalid_flat: (B*V, T) bool, True if patch has no valid depth
        """
        B, V, H, W = depth.shape
        Ht = Wt = int(token_len ** 0.5)
        if Ht * Wt != token_len:
            raise ValueError(f"Expected square token grid, got token_len={token_len}.")

        ph = H // Ht
        pw = W // Wt
        if ph * Ht != H or pw * Wt != W:
            raise ValueError(f"Depth {H}x{W} not divisible by token grid {Ht}x{Wt}.")

        depth = depth.to(dtype=torch.float32)
        intrinsics = intrinsics.to(dtype=torch.float32)
        cam2egos = cam2egos.to(dtype=torch.float32)

        # per-token mean depth excluding invalid_depth_val
        d = depth.view(B, V, Ht, ph, Wt, pw)  # (B,V,Ht,ph,Wt,pw)
        valid = d != float(self.invalid_depth_val)
        cnt = valid.sum(dim=(3, 5))  # (B,V,Ht,Wt)
        d_sum = (d * valid).sum(dim=(3, 5))
        d_mean = torch.where(cnt > 0, d_sum / cnt.clamp_min(1), torch.zeros_like(d_sum))
        invalid = cnt == 0  # (B,V,Ht,Wt)

        device = depth.device

        # token cell center in pixel coords (0-indexed)
        u_cent = (torch.arange(Wt, device=device, dtype=torch.float32) + 0.5) * pw - 0.5  # (Wt,)
        v_cent = (torch.arange(Ht, device=device, dtype=torch.float32) + 0.5) * ph - 0.5  # (Ht,)
        vv, uu = torch.meshgrid(v_cent, u_cent, indexing="ij")  # (Ht,Wt)

        uu = uu[None, None].expand(B, V, -1, -1)
        vv = vv[None, None].expand(B, V, -1, -1)

        fx = intrinsics[:, :, 0, 0][..., None, None]
        fy = intrinsics[:, :, 1, 1][..., None, None]
        cx = intrinsics[:, :, 0, 2][..., None, None]
        cy = intrinsics[:, :, 1, 2][..., None, None]

        z = d_mean
        x = (uu - cx) / fx * z
        y = (vv - cy) / fy * z

        ones = torch.ones_like(z)
        p_cam = torch.stack([x, y, z, ones], dim=-1)  # (B,V,Ht,Wt,4)

        # cam -> ego
        p_ego = torch.einsum("bvij,bvhwj->bvhwi", cam2egos, p_cam)  # (B,V,Ht,Wt,4)
        xyz = p_ego[..., :3]  # (B,V,Ht,Wt,3)

        # invalid patches contribute zero PE
        xyz = xyz.masked_fill(invalid[..., None], 0.0)

        xyz_flat = xyz.reshape(B * V, Ht * Wt, 3).to(dtype=out_dtype)
        invalid_flat = invalid.reshape(B * V, Ht * Wt)
        return xyz_flat, invalid_flat



    def _llm_forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        data = None, # optional for additional geometry
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        pixel_values: [B, 4, 3, 448, 448]
        image_flags: [B*4] (bool), false for padding dummy images
        """
        return_dict = (
            return_dict
            if return_dict is not None
            else self.model.config.use_return_dict
        )

        image_flags = image_flags.squeeze(-1)
        # We only added the clone code here to avoid the error.
        input_embeds = self.model.language_model.get_input_embeddings()(
            input_ids
        ).clone()

        # vit_embeds = self.model.extract_feature(pixel_values) # B*4, (448/14/2)^2=256, H=2048
        # vit_embeds = vit_embeds[image_flags == 1]
        # newly added for positional embedding
        if self.use_pe3d:
            assert data is not None, "data must be provided for view embedding"
            depth = data["depth"].to(input_ids.device) # [B, 4, H, W]
            cam2imgs = data["cam2imgs"]
            cam2egos = data["cam2egos"]
            if isinstance(cam2imgs, dict):
                order = ["front", "left", "back", "right"]
                cam2imgs = torch.stack([cam2imgs[k] for k in order], dim=1)
                cam2egos = torch.stack([cam2egos[k] for k in order], dim=1)
            cam2imgs = cam2imgs.to(input_ids.device)
            cam2egos = cam2egos.to(input_ids.device)
            intrinsics = cam2imgs[:, :, :3, :3]
            # batch_size = input_ids.shape[0]
            # if self.num_views ==  4:
            #     view_ids = torch.arange(4, device=vit_embeds.device).repeat(batch_size) # B*4
            #     view_ids = view_ids[image_flags == 1]
            #     if view_ids.numel() != vit_embeds.shape[0]:
            #         raise ValueError(
            #             f"view_ids ({view_ids.numel()}) != vit_embeds ({vit_embeds.shape[0]}). "
            #             "Expected 4 views per sample in fixed order."
            #         )
            #     view_emb = self.view_embed(view_ids).to(vit_embeds.dtype) # B*4, H
            #     vit_embeds = vit_embeds + view_emb[:, None, :]  # B*4, 256, H
            # elif self.num_views == 8:
            #     if vit_embeds.shape[0] != batch_size * 4:
            #         raise ValueError(
            #             f"Expected 4 views per sample for 8-dir embedding, got {vit_embeds.shape[0]} "
            #             f"for batch_size {batch_size}."
            #         )
            #     t = vit_embeds.shape[1]
            #     fH = int(math.sqrt(t))
            #     fW = fH
            #     if fH * fW != t:
            #         raise ValueError(f"Expected square token grid, got t={t}.")
            #     dir_ids = self._dir_ids_8_from_views(
            #         batch_size, fH, fW, vit_embeds.device
            #     )
            #     dir_emb = self.view_embed(dir_ids).to(vit_embeds.dtype)
            #     vit_embeds = vit_embeds + dir_emb
            # else:
            #     raise NotImplementedError(f"num_views={self.num_views} must be 4 or 8.")
            vit_preproj = self._extract_feature_preproj(pixel_values)  # (B*4, T, C_preproj)
            T = vit_preproj.shape[1]

            xyz_flat, invalid_flat = self._xyz_ego_tokens(
                depth=depth,
                intrinsics=intrinsics,
                cam2egos=cam2egos,
                token_len=T,
                out_dtype=vit_preproj.dtype,
            )  # xyz_flat: (B*4,T,3)

            pos = self.pos3d_mlp(xyz_flat)  # (B*4, T, C_preproj)
            pos = pos.masked_fill(invalid_flat[..., None], 0.0)

            vit_preproj = vit_preproj + pos
            vit_embeds = self.model.mlp1(vit_preproj)  # (B*4, T, C_llm)
        else:
            vit_embeds = self.model.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        
        # end of newly added for positional embedding
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        # if torch.distributed.get_rank() == 0 and self._count % 100 == 0:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        if rank == 0 and self._count % 100 == 0:
            print(
                f"dynamic ViT batch size: {vit_batch_size}, "
                f"images per sample: {vit_batch_size / B}, "
                f"dynamic token length: {N}"
            )
        self._count += 1

        input_ids = input_ids.reshape(B * N)
        selected = input_ids == self.model.img_context_token_id
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(
                -1, C
            )
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(
                f"warning: {e}, input_embeds[selected].shape="
                f"{input_embeds[selected].shape}, "
                f"vit_embeds.shape={vit_embeds.shape}"
            )
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(
                -1, self.model.language_model.config.vocab_size
            )
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def _llm_forward_autoregressive(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        image_flags: torch.LongTensor,
        generation_config: Optional[GenerationConfig] = None,
        data=None, # optional for additional geometry
    ):
        # Same visual-token replacement as _llm_forward, then generate()
        image_flags = image_flags.squeeze(-1)
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()

        # vit_embeds = self.model.extract_feature(pixel_values)
        # vit_embeds = vit_embeds[image_flags == 1]
        # if self.use_pe3d:
        #     batch_size = input_ids.shape[0]
        #     if self.num_views ==  4:
        #         view_ids = torch.arange(4, device=vit_embeds.device).repeat(batch_size) # B*4
        #         view_ids = view_ids[image_flags == 1]
        #         if view_ids.numel() != vit_embeds.shape[0]:
        #             raise ValueError(
        #                 f"view_ids ({view_ids.numel()}) != vit_embeds ({vit_embeds.shape[0]}). "
        #                 "Expected 4 views per sample in fixed order."
        #             )
        #         view_emb = self.view_embed(view_ids).to(vit_embeds.dtype) # B*4, H
        #         vit_embeds = vit_embeds + view_emb[:, None, :]  # B*4, 256, H
        #     elif self.num_views == 8:
        #         if vit_embeds.shape[0] != batch_size * 4:
        #             raise ValueError(
        #                 f"Expected 4 views per sample for 8-dir embedding, got {vit_embeds.shape[0]} "
        #                 f"for batch_size {batch_size}."
        #             )
        #         t = vit_embeds.shape[1]
        #         fH = int(math.sqrt(t))
        #         fW = fH
        #         if fH * fW != t:
        #             raise ValueError(f"Expected square token grid, got t={t}.")
        #         dir_ids = self._dir_ids_8_from_views(
        #             batch_size, fH, fW, vit_embeds.device
        #         )
        #         dir_emb = self.view_embed(dir_ids).to(vit_embeds.dtype)
        #         vit_embeds = vit_embeds + dir_emb
        #     else:
        #         raise NotImplementedError(f"num_views={self.num_views} must be 4 or 8.")
        if self.use_pe3d:
            assert data is not None, "data must be provided for view embedding"
            depth = data["depth"].to(input_ids.device) # [B, 4, H, W]
            images = data['raw_img'] # (B, 4, H, W, 3) torch.uint8 BGR
            cam2imgs = data["cam2imgs"]
            cam2egos = data["cam2egos"]
            if isinstance(cam2imgs, dict):
                order = ["front", "left", "back", "right"]
                cam2imgs = torch.stack([cam2imgs[k] for k in order], dim=1)
                cam2egos = torch.stack([cam2egos[k] for k in order], dim=1)
            cam2imgs = cam2imgs.to(input_ids.device)
            cam2egos = cam2egos.to(input_ids.device)
            intrinsics = cam2imgs[:, :, :3, :3]
            vit_preproj = self._extract_feature_preproj(pixel_values)  # (B*4, T, C_preproj)
            T = vit_preproj.shape[1]

            xyz_flat, invalid_flat = self._xyz_ego_tokens(
                depth=depth,
                intrinsics=intrinsics,
                cam2egos=cam2egos,
                token_len=T,
                out_dtype=vit_preproj.dtype,
            )  # xyz_flat: (B*4,T,3)

            pos = self.pos3d_mlp(xyz_flat)  # (B*4, T, C_preproj)
            pos = pos.masked_fill(invalid_flat[..., None], 0.0)

            vit_preproj = vit_preproj + pos
            vit_embeds = self.model.mlp1(vit_preproj)  # (B*4, T, C_llm)
        else:
            vit_embeds = self.model.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]

        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        if rank == 0 and self._count % 100 == 0:
            print(
                f"dynamic ViT batch size: {vit_batch_size}, "
                f"images per sample: {vit_batch_size / B}, "
                f"dynamic token length: {N}"
            )
        self._count += 1

        flat_ids = input_ids.reshape(B * N)
        selected = flat_ids == self.model.img_context_token_id
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(
                f"warning: {e}, input_embeds[selected].shape="
                f"{input_embeds[selected].shape}, "
                f"vit_embeds.shape={vit_embeds.shape}"
            )
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        if generation_config is None:
            im_end_ids = self.tokenizer("<|im_end|>", add_special_tokens=False).input_ids
            assert len(im_end_ids) == 1, "<|im_end|> should be a single token id."
            im_end_id = im_end_ids[0]
            generation_config = GenerationConfig(
                max_new_tokens=32,
                do_sample=False,
                eos_token_id=im_end_id,
                pad_token_id=(
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else im_end_id
                ),
            )

        return self.model.language_model.generate(
            input_ids=input_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            use_cache=True,
        )

    def _circ_dist8(self, a: int, b: int) -> int:
        d = abs(a - b)
        return min(d, 8 - d)

    # def _success_at_1(self, move_dir_pred: str, range_tag_pred: str,
    #                 move_dir_gt: str, range_tag_gt: str):
    #     DIR2IDX = {"FRONT":0,"FRONTRIGHT":1,"RIGHT":2,"BACKRIGHT":3,"BACK":4,"BACKLEFT":5,"LEFT":6,"FRONTLEFT":7}
    #     dstep = self._circ_dist8(DIR2IDX[move_dir_pred], DIR2IDX[move_dir_gt])
    #     succ1_dir = int(dstep <= 1)
    #     succ1_dir_and_size = int((dstep <= 1) and (range_tag_pred == range_tag_gt))
    #     return dstep, succ1_dir, succ1_dir_and_size

    def _success_at_0_and_1(self,
                       move_dir_pred: str, range_tag_pred: str,
                       move_dir_gt: str, range_tag_gt: str):
        DIR2IDX = {
            "FRONT": 0, "FRONTRIGHT": 1, "RIGHT": 2, "BACKRIGHT": 3,
            "BACK": 4, "BACKLEFT": 5, "LEFT": 6, "FRONTLEFT": 7
        }
        dstep = self._circ_dist8(DIR2IDX[move_dir_pred], DIR2IDX[move_dir_gt])

        succ0_dir = int(dstep == 0)
        succ0_dir_and_size = int((dstep == 0) and (range_tag_pred == range_tag_gt))

        succ1_dir = int(dstep <= 1)
        succ1_dir_and_size = int((dstep <= 1) and (range_tag_pred == range_tag_gt))

        return dstep, succ0_dir, succ0_dir_and_size, succ1_dir, succ1_dir_and_size
    
    def _success_at_0_and_1_dir_only(self, pred_dir_id, gt_dir_id):
        dstep = self._circ_dist8(int(pred_dir_id), int(gt_dir_id))

        succ0_dir = int(dstep == 0)
        succ1_dir = int(dstep <= 1)

        return dstep, succ0_dir, succ1_dir
    
    def _circ_distN(self, a: int, b: int, N: int) -> int:
        """Circular distance on N bins."""
        d = abs(a - b)
        return min(d, N - d)

    def _success_at_0_1_2_bins(self, dir_pred_id: int, dir_gt_id: int, N: int = 48):
        """
        dir_pred_id, dir_gt_id: ints in [0, N-1]
        Returns: dstep, succ0, succ1, succ2
        """
        dstep = self._circ_distN(int(dir_pred_id), int(dir_gt_id), N)

        succ0 = int(dstep == 0)
        succ1 = int(dstep <= 1)
        succ2 = int(dstep <= 2)

        return dstep, succ0, succ1, succ2
    
    def parse_and_predict_meta_action(self, text: str, text_label: str, data: dict):
        text_compact = re.sub(r"[^A-Z]", "", text.upper())
        text_label_compact = re.sub(r"[^A-Z]", "", text_label.upper())
        try:
            move_dir = next(d for d in ["FRONTLEFT","FRONTRIGHT","BACKLEFT","BACKRIGHT","FRONT","BACK","LEFT","RIGHT"] if d in text_compact)
        except StopIteration:
            # if not move_dir:
            move_dir = "FRONT"  # default
        range_tag = "SMALL" if "SMALL" in text.upper() else "BIG" # default
        move_dir_label = next(d for d in ["FRONTLEFT","FRONTRIGHT","BACKLEFT","BACKRIGHT","FRONT","BACK","LEFT","RIGHT"] if d in text_label_compact)
        range_tag_label = "SMALL" if "SMALL" in text_label.upper() else "BIG"
        print(f"[Eval] Predicted: {text} (dir: {move_dir}, {range_tag}); "
                f"Label: {text_label} (dir: {move_dir_label}, {range_tag_label})")
                
        # dstep, succ1_dir, succ1_dir_and_size = self._success_at_1(
        #     move_dir, range_tag,
        #     move_dir_label, range_tag_label)
        dstep, succ0_dir, succ0_dir_and_size, succ1_dir, succ1_dir_and_size = self._success_at_0_and_1(
            move_dir, range_tag,
            move_dir_label, range_tag_label)
        return move_dir, move_dir_label, range_tag, range_tag_label, dstep, succ1_dir, succ1_dir_and_size, succ0_dir, succ0_dir_and_size
    
    def parse_pred_to_meta_action(self, pred_region, gt_region):
        result_template = "Move towards the {} region with a {} step."
        # parse predicted region token idx to direction and range
        pred_region_idx = pred_region.item()
        gt_region_idx = gt_region.item()
        region_token_map = {
            0: "Front Small",
            1: "Front Big",
            2: "FrontLeft Small",
            3: "FrontLeft Big",
            4: "Left Small",
            5: "Left Big",
            6: "BackLeft Small",
            7: "BackLeft Big",
            8: "Back Small",
            9: "Back Big",
            10: "BackRight Small",
            11: "BackRight Big",
            12: "Right Small",
            13: "Right Big",
            14: "FrontRight Small",
            15: "FrontRight Big",
        }
        pred_region_str = region_token_map.get(pred_region_idx, "Front Small")
        gt_region_str = region_token_map.get(gt_region_idx, "Front Small")
        pred_meta_action = result_template.format(
            pred_region_str.split()[0],
            pred_region_str.split()[1],
        )
        gt_meta_action = result_template.format(
            gt_region_str.split()[0],
            gt_region_str.split()[1],
        )
        return pred_meta_action, gt_meta_action

    
    def oracle_bev_predict_target(self, move_dir: str, range_tag: str, data: dict):        
        # oracle bev query
        base_yaw_deg = math.degrees(data['agent_yaw_hab_rad'][0].item())
        traversable_mask, fan_mask, best_idx, best_xy = self.oracle_bev_generator.oracle_free_point(
            yaw_deg=base_yaw_deg,
            target_token=data['viewpoint_hash'][0],
            direction=move_dir,
            step_size=range_tag,
            token_key="waypoint_token",
        )
        # # debug
        # self.oracle_bev_generator.save_vis(
        #     traversable_mask,
        #     fan_mask,
        #     best_xy,
        #     best_idx,
        #     save_path="debug_outputs/oracle_bev_traversable.png",
        # )
        # exit(0)
        return best_xy
    
    def oracle_bev_predict_target_from_dir48(self, dir_id: int, data: dict):
        base_yaw_deg = math.degrees(data['agent_yaw_hab_rad'][0].item())
        best_xy = self.oracle_bev_generator.oracle_free_xy_from_dir48(
            yaw_deg=base_yaw_deg,
            target_token=data['viewpoint_hash'][0],
            dir_id=dir_id,
            token_key="waypoint_token",
        )
        return best_xy

    def forward(self, data, data_samples=None, mode="loss"):
        # print("[debug] forward data keys:", data.keys())
        # for k, v in data.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
        #     elif isinstance(v, list):
        #         print(f"  {k}: list of length {len(v)}")
        #         for i, item in enumerate(v):
        #             if isinstance(item, torch.Tensor):
        #                 print(f"    [{i}]: shape={item.shape}, dtype={item.dtype}, device={item.device}")
        #             elif isinstance(item, dict):
        #                 print(f"    [{i}]: dict with keys {list(item.keys())}")
        #                 for dk, dv in item.items():
        #                     if isinstance(dv, torch.Tensor):
        #                         print(f"        {dk}: shape={dv.shape}, dtype={dv.dtype}, device={dv.device}")
        #                     else:
        #                         print(f"        {dk}: type={type(dv)}")
        #             else:
        #                 print(f"    [{i}]: type={type(item)}")
        #     elif isinstance(v, dict):
        #         print(f"  {k}: dict with keys {list(v.keys())}")
        #         for dk, dv in v.items():
        #             if isinstance(dv, torch.Tensor):
        #                 print(f"    {dk}: shape={dv.shape}, dtype={dv.dtype}, device={dv.device}")
        #             else:
        #                 print(f"    {dk}: type={type(dv)}")
        #     else:
        #         print(f"  {k}: type={type(v)}")
        # '''
        # [debug] forward data keys: dict_keys(['input_ids', 'attention_mask', 'position_ids', 'labels', 'pixel_values', 'image', 'conversations', 'scene', 'token', 'viewpoint_hash', 'agent_pos_hab', 'agent_yaw_hab_rad', 'cam_height', 'waypoints_hab', 'pano_path', 'depth_paths', 'cam2imgs', 'cam2egos', 'ego2global', 'p_goal', 'ep_idx'])
        # input_ids: shape=torch.Size([1, 1231]), dtype=torch.int64, device=cuda:0
        # attention_mask: shape=torch.Size([1, 1231]), dtype=torch.bool, device=cuda:0
        # position_ids: shape=torch.Size([1, 1231]), dtype=torch.int64, device=cuda:0
        # labels: shape=torch.Size([1, 1231]), dtype=torch.int64, device=cuda:0
        # pixel_values: shape=torch.Size([1, 4, 3, 448, 448]), dtype=torch.float32, device=cuda:0
        # image: list of length 1
        #     [0]: type=<class 'list'>
        # conversations: list of length 1
        #     [0]: type=<class 'list'>
        # scene: list of length 1
        #     [0]: type=<class 'str'>
        # token: list of length 1
        #     [0]: type=<class 'str'>
        # viewpoint_hash: list of length 1
        #     [0]: type=<class 'str'>
        # agent_pos_hab: shape=torch.Size([1, 3]), dtype=torch.float32, device=cuda:0
        # agent_yaw_hab_rad: shape=torch.Size([1]), dtype=torch.float32, device=cuda:0
        # cam_height: shape=torch.Size([1]), dtype=torch.float32, device=cuda:0
        # waypoints_hab: shape=torch.Size([1, 6, 3]), dtype=torch.float32, device=cuda:0
        # pano_path: list of length 1
        #     [0]: type=<class 'str'>
        # depth_paths: list of length 1
        #     [0]: type=<class 'str'>
        # cam2imgs: dict with keys ['front', 'left', 'back', 'right']
        #     front: shape=torch.Size([1, 4, 4]), dtype=torch.float32, device=cuda:0
        #     left: shape=torch.Size([1, 4, 4]), dtype=torch.float32, device=cuda:0
        #     back: shape=torch.Size([1, 4, 4]), dtype=torch.float32, device=cuda:0
        #     right: shape=torch.Size([1, 4, 4]), dtype=torch.float32, device=cuda:0
        # cam2egos: dict with keys ['front', 'left', 'back', 'right']
        #     front: shape=torch.Size([1, 4, 4]), dtype=torch.float32, device=cuda:0
        #     left: shape=torch.Size([1, 4, 4]), dtype=torch.float32, device=cuda:0
        #     back: shape=torch.Size([1, 4, 4]), dtype=torch.float32, device=cuda:0
        #     right: shape=torch.Size([1, 4, 4]), dtype=torch.float32, device=cuda:0
        # ego2global: shape=torch.Size([1, 4, 4]), dtype=torch.float32, device=cuda:0
        # p_goal: shape=torch.Size([1, 3]), dtype=torch.float32, device=cuda:0
        # ep_idx: shape=torch.Size([1]), dtype=torch.int64, device=cuda:0
        # '''

        pixel_values = data["pixel_values"]

        if type(pixel_values) is list or pixel_values.ndim == 5:
            if type(pixel_values) is list:
                pixel_values = [
                    x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                ]
            # b*n, c, h, w
            concat_images = torch.cat(
                [image.to(self.model.vision_model.dtype) for image in pixel_values],
                dim=0,
            )
        else:
            raise NotImplementedError()

        input_ids = data["input_ids"]
        position_ids = data["position_ids"]
        attention_mask = data["attention_mask"]
        # sum is 0 are text
        image_flags = torch.sum(concat_images, dim=(1, 2, 3)) != 0
        image_flags = image_flags.long()

        labels = data["labels"]
        use_cache = False

        # Directly calling this code in LORA fine-tuning
        # will result in an error,so we must rewrite it.
        # TODO: Once the official is fixed, we can remove it.
        # outputs = self.model(input_ids=input_ids,
        #                      position_ids=position_ids,
        #                      attention_mask=attention_mask,
        #                      image_flags=image_flags,
        #                      pixel_values=concat_images,
        #                      labels=labels,
        #                      use_cache=use_cache)

        variants = data['variant']
        if self.output_mode == "VQA":
            if self.training:
                out = self._llm_forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    image_flags=image_flags,
                    pixel_values=concat_images,
                    labels=labels,
                    use_cache=use_cache,
                    data=data,
                )
                loss_ce = out.loss # effectively averaged over #per_batch_chat_samples*#average_answer
                return {"loss": loss_ce, "loss_ce": loss_ce}
            else:
                assert len(input_ids) == 1, "Eval batch size should be 1."
                amp_ctx = autocast(dtype=torch.bfloat16)
                forward_prediction_reults = [] # mmengine requires this to be a list
                # truncate prompt using labels
                prompt_input_ids, prompt_attention_mask, prompt_position_ids, prompt_len = (
                    self._truncate_to_prompt(input_ids, attention_mask, position_ids, labels)
                )
                with amp_ctx:
                    gen_ids = self._llm_forward_autoregressive(
                        input_ids=prompt_input_ids,
                        attention_mask=prompt_attention_mask,
                        image_flags=image_flags,
                        pixel_values=concat_images,
                        data=data,
                    )
                gen_new_ids = gen_ids[:, prompt_len:]  # 1 x T'
                text = self.tokenizer.decode(gen_new_ids[0], skip_special_tokens=False)
                loss_ce = torch.tensor(0.0, device=input_ids.device) # no CE loss in AR eval; keep as 0 for ExternalMetric logging
                # Keep only valid vocab IDs
                label_ids = labels[0]
                valid = (label_ids >= 0) & (label_ids < self.tokenizer.vocab_size)
                text_label = self.tokenizer.decode(label_ids[valid], skip_special_tokens=False)

                move_dir, move_dir_label, range_tag, range_tag_label, dstep, succ1_dir, succ1_dir_and_size, succ0_dir, succ0_dir_and_size = self.parse_and_predict_meta_action(
                    text, text_label, data)
                # best_xy = self.oracle_bev_predict_target(
                #     move_dir=move_dir,
                #     range_tag=range_tag,
                #     data=data,
                # ) # BUG: sometimes this will crash cuz the work is too heavy? or some bugs? but we will not rely on this result in the end
                import numpy as np
                best_xy = np.array([0.0, 0.0])  # dummy placeholder
                # pred_aff = torch.zeros((128, 128), device=input_ids.device)  # dummy placeholder
                forward_prediction_reults.append(
                    {"p_pred": torch.from_numpy(best_xy).to(input_ids.device), "loss_ce": loss_ce, "loss": loss_ce, "text_pred": text, \
                        "dstep": dstep, "succ1_dir": succ1_dir, "succ1_dir_and_size": succ1_dir_and_size, \
                            "succ0_dir": succ0_dir, "succ0_dir_and_size": succ0_dir_and_size}
                )
                return forward_prediction_reults
        elif self.output_mode == "Affordance":
            predict_mode = 'waypoints' # region, xy, waypoints
            is_fine_tune = True
            if self.training:
                if is_fine_tune:
                    amp_ctx = autocast(dtype=torch.bfloat16)
                    with amp_ctx:
                        out = self._llm_forward(
                            input_ids=input_ids,
                            position_ids=position_ids,
                            attention_mask=attention_mask,
                            image_flags=image_flags,
                            pixel_values=concat_images,
                            labels=None,  # no labels in conversation
                            # labels=labels, # BUG
                            use_cache=use_cache,
                            output_hidden_states=True,   # we need hidden states for tgt token
                            return_dict=True, # we need dict output
                            data=data,
                        )
                    hidden = out.hidden_states[-1] # B x T x H
                    tgt_mask = (input_ids == self.tgt_token_idx) # B x T
                    tgt_state = hidden[tgt_mask] # B, H
                    B, N, fH, fW, H = hidden.shape[0], 4, 16, 16, hidden.shape[2]

                    img_mask = (input_ids == self.model.img_context_token_id) # B x T
                    assert img_mask.sum(-1).eq(16*16*4).all(), "Expected 16x16x4 image context tokens."
                    img_states = hidden[img_mask].contiguous().view(B, 4*16*16, H) # B x (16*16*4) x H
                    img_states = img_states.view(B, N, fH, fW, H)  # B x 4 x fH=16 x fW=16 x H # N=view first, row major

                    pred_r_theta = self.tgt_head_polar(tgt_state)  # [B, 3]
                    # finetune_starting_query_feat = tgt_state.detach()
                    finetune_starting_query_feat = tgt_state
                    finetune_starting_ctx_feat = img_states
                    # finetune_starting_ctx_feat = None
                    # finetune_starting_skip_result = pred_r_theta.detach()
                    finetune_starting_skip_result = None

                    depth = data["depth"].to(input_ids.device) # [B, 4, H, W]
                    images = data['raw_img'] # (B, 4, H, W, 3) torch.uint8 BGR
                    cam2imgs = data["cam2imgs"]
                    cam2egos = data["cam2egos"]
                    if isinstance(cam2imgs, dict):
                        order = ["front", "left", "back", "right"]
                        cam2imgs = torch.stack([cam2imgs[k] for k in order], dim=1)
                        cam2egos = torch.stack([cam2egos[k] for k in order], dim=1)
                    cam2imgs = cam2imgs.to(input_ids.device)
                    cam2egos = cam2egos.to(input_ids.device)
                    intrinsics = cam2imgs[:, :, :3, :3]

                    # # Method 1: predict waypoints directly
                    # pred_waypoints_xy = self.finetune_head(
                    #     finetune_starting_query_feat,
                    #     finetune_starting_skip_result,
                    #     depth,
                    #     images.permute(0,1,4,2,3),  # to B,4,3,H,W
                    #     intrinsics,
                    #     cam2egos,
                    # )   # [B, 6, 2]
                    # target_waypoint_xy = data['waypoints'][:, :, :2].to(pred_waypoints_xy.device)  # [B, 6, 2]
                    # loss_l1 = torch.nn.functional.l1_loss(pred_waypoints_xy, target_waypoint_xy)

                    # use_free_space_loss = False
                    # if use_free_space_loss:
                    #     occ_field = torch.sqrt(data["occ_field"].to(pred_waypoints_xy.device))  # [B, H, W], in agent local frame
                    #     B, num_wp, _ = pred_waypoints_xy.shape
                    #     pred_wp_xy_reshaped = pred_waypoints_xy.view(B * num_wp, 2)
                    #     loss_free_space = torch.nn.functional.grid_sample(
                    #         occ_field.unsqueeze(1).float().repeat_interleave(num_wp, dim=0),
                    #         torch.stack([(((pred_wp_xy_reshaped[:,1]-(-6.4))/0.2)/(occ_field.shape[2]-1)*2-1),
                    #                     (((pred_wp_xy_reshaped[:,0]-(-6.4))/0.2)/(occ_field.shape[1]-1)*2-1)], dim=-1).view(-1,1,1,2),
                    #         mode="bilinear", padding_mode="border", align_corners=True
                    #     ).view(-1) # [B * num_wp]
                    #     loss_free_space = torch.clamp(loss_free_space, max=5.0).mean()  # cap at 5.0 to avoid extreme outliers
                    #     loss = loss_l1 + 0.5 * loss_free_space
                    # else:
                    #     loss = loss_l1
                    #     loss_free_space = torch.tensor(0.0, device=loss_l1.device)

                    # return {"loss": loss, "loss_l1": loss_l1, "loss_free_space": loss_free_space}

                    # # Method 2: score on anchors
                    # pred_anchor_scores = self.finetune_head(
                    #     finetune_starting_query_feat,
                    #     finetune_starting_skip_result,
                    #     depth,
                    #     images.permute(0,1,4,2,3),  # to B,4,3,H,W
                    #     intrinsics,
                    #     cam2egos,
                    # )   # [B, num_anchors]
                    # target_anchor_idx, _ = self.finetune_head.find_best_waypoint_anchor(
                    #     data['waypoints'][:, :, :2].to(pred_anchor_scores.device)  # [B, 6, 2]
                    # )
                    # target_onehot = torch.zeros_like(pred_anchor_scores)
                    # target_onehot.scatter_(1, target_anchor_idx.unsqueeze(1), 1.0)
                    # # loss_cls = self.finetune_head.py_sigmoid_focal_loss(
                    # #     pred_anchor_scores,
                    # #     target_onehot,
                    # #     gamma=2.0,
                    # #     alpha=0.25,
                    # #     reduction='mean'
                    # # ) * 20.0  # scale up to match magnitude of other losses
                    # loss_cls = self.finetune_head.ade_soft_anchor_ce_loss(
                    #     pred_logits=pred_anchor_scores,
                    #     gt_waypoints_xy=data["waypoints"][:, :, :2].to(pred_anchor_scores.device),
                    #     sigma=0.5,
                    #     topk=None,   # or an int
                    # )
                    # return {"loss": loss_cls, "loss_cls": loss_cls}

                    # # Method 2: score on anchors, plus oracle offset
                    # pred_anchor_scores, pred_offsets = self.finetune_head(
                    #     finetune_starting_query_feat,
                    #     finetune_starting_skip_result,
                    #     depth,
                    #     images.permute(0,1,4,2,3),  # to B,4,3,H,W
                    #     intrinsics,
                    #     cam2egos,
                    # )   # [B, num_anchors]; [B, num_anchors, 6, 2]
                    # target_anchor_idx, target_anchor = self.finetune_head.find_best_waypoint_anchor(
                    #     data['waypoints'][:, :, :2].to(pred_anchor_scores.device)  # [B, 6, 2]
                    # ) # [B], [B, 6, 2]
                    # target_onehot = torch.zeros_like(pred_anchor_scores)
                    # target_onehot.scatter_(1, target_anchor_idx.unsqueeze(1), 1.0)
                    # # loss_cls = self.finetune_head.py_sigmoid_focal_loss(
                    # #     pred_anchor_scores,
                    # #     target_onehot,
                    # #     gamma=2.0,
                    # #     alpha=0.25,
                    # #     reduction='mean'
                    # # ) * 20.0  # scale up to match magnitude of other losses
                    # loss_cls = self.finetune_head.ade_soft_anchor_ce_loss(
                    #     pred_logits=pred_anchor_scores,
                    #     gt_waypoints_xy=data["waypoints"][:, :, :2].to(pred_anchor_scores.device),
                    #     sigma=0.5,
                    #     topk=None,   # or an int
                    # )
                    # oracle_selected_offsets = torch.gather(
                    #     pred_offsets,
                    #     1,
                    #     target_anchor_idx.view(-1, 1, 1).unsqueeze(-1).repeat(1, 1, pred_offsets.shape[2], 2)
                    # ).squeeze(1)  # [B, 6, 2]
                    # path_pred = target_anchor + oracle_selected_offsets  # [B, 6, 2]
                    # path_gt = data['waypoints'][:, :, :2].to(path_pred.device)  # [B, 6, 2]
                    # loss_l1 = torch.nn.functional.l1_loss(path_pred, path_gt)
                    # loss = 2 * loss_cls + loss_l1 # scale to mimic the original 100:1 when the cls is focal loss
                    # return {"loss": loss, "loss_cls": loss_cls, "loss_l1": loss_l1}

                    # Method 3: direct bev affordance
                    # print(data['traversable_mask'].shape) # (B, H, W)
                    # print(data['visible_mask'].shape)
                    # print(data['affordance_mask'].shape)
                    # exit(0)
                    use_occ_loss = False
                    use_diff_loss = False
                    use_prob_loss = False
                    affordance_label = data['affordance_mask'].to(input_ids.device).float()
                    traversable_mask = data['traversable_mask'].to(input_ids.device)
                    visible_mask = data['visible_mask'].to(input_ids.device)
                    affordance_gt_prob = data.get('affordance_gt_prob', None)
                    # # debug
                    # print(affordance_gt_prob.shape, affordance_gt_prob.min(), affordance_gt_prob.max(), affordance_gt_prob.mean())
                    # import numpy as np
                    # save_path = "debug_outputs/1d_prob_gt.npz"
                    # np.savez_compressed(
                    #     save_path,
                    #     affordance_gt_prob=affordance_gt_prob.cpu().numpy(),
                    #     affordance_label=affordance_label.cpu().numpy(),
                    #     traversable_mask=traversable_mask.cpu().numpy(),
                    #     visible_mask=visible_mask.cpu().numpy(),
                    # )
                    # exit(0)
                    # # end of debug
                    affordance_logits = self.finetune_head(
                        finetune_starting_query_feat,
                        finetune_starting_skip_result,
                        depth,
                        images.permute(0,1,4,2,3),  # to B,4,3,H,W
                        intrinsics,
                        cam2egos,
                        context_tokens=finetune_starting_ctx_feat,
                        affordance_mask=affordance_label, # only for debug
                        traversable_mask=traversable_mask, # only for debug
                        visible_mask=visible_mask, # only for debug
                        affordance_gt_prob=affordance_gt_prob.to(input_ids.device) if affordance_gt_prob is not None else None, # only for diffusion
                        gt_xy=data["p_goal"][:, :2].to(input_ids.device), # only for diffusion
                    )   # [B, H, W]

                    if affordance_logits.dim() == 3: # B, H, W
                        # affordance_label = affordance_label * traversable_mask  # only consider traversable area
                        use_pseudo_label = True
                        if not use_pseudo_label:
                            affordance_label = affordance_label * traversable_mask * visible_mask  # only consider traversable and visible area
                        else:
                            pseudo_bound = data["pseudo_bound"].to(input_ids.device).float()
                            from xtuner.dataset.internvl_dataset import SingleFrameLoader
                            import numpy as np
                            p_goal_xy = data["p_goal"][:, :2].to(input_ids.device).float()
                            x_min = y_min = -6.4
                            res = 0.1
                            rmax_cells = 10.0
                            B, Hm, Wm = pseudo_bound.shape
                            pseudo_aff = torch.zeros_like(pseudo_bound)
                            for b in range(B):
                                region = (pseudo_bound[b] > 0.5).detach().cpu().numpy().astype(bool)
                                x = float(p_goal_xy[b, 0].item())
                                y = float(p_goal_xy[b, 1].item())
                                i = int(round((x - (x_min + 0.5 * res)) / res))
                                j = int(round((y - (y_min + 0.5 * res)) / res))
                                if i < 0 or i >= Hm or j < 0 or j >= Wm:
                                    raise RuntimeError(
                                        f"p_goal out of BEV grid at b={b}: x={x:.3f}, y={y:.3f}, i={i}, j={j}, H={Hm}, W={Wm}"
                                    )
                                region_goal = region
                                if not bool(region_goal[i, j]):
                                    region_goal = region_goal.copy()
                                    region_goal[i, j] = True
                                _, dist = SingleFrameLoader.bounded_geodesic_gaussian_cells(
                                    region_goal, i, j, rmax_cells=rmax_cells, return_dist=True
                                )
                                pseudo_aff[b] = torch.from_numpy((dist <= rmax_cells).astype(np.float32)).to(pseudo_aff.device)
                            affordance_label = pseudo_aff

                            # # debug
                            # save_path = f'debug_outputs/1_pseudo_label.npz'
                            # np.savez_compressed(
                            #     save_path,
                            #     gt_affordance_label=data['affordance_mask'].cpu().numpy(),
                            #     traversable_mask=traversable_mask.cpu().numpy(),
                            #     visible_mask=visible_mask.cpu().numpy(),
                            #     affordance_label=affordance_label.cpu().numpy(),
                            #     pseudo_bound=pseudo_bound.cpu().numpy(),
                            #     p_goal_xy=p_goal_xy.cpu().numpy(),
                            # )
                            # exit(0)
                            # # end of debug

                        # # ablation version: changeable geodesic radius
                        # affordance_label = data['affordance_gt_bin'].to(input_ids.device).float()

                        # # debug
                        # affordance_gt_bin = data['affordance_gt_bin'].to(input_ids.device).float()
                        # try:
                        #     assert (affordance_gt_bin == affordance_label).all(), "Affordance GT mismatch with ann mask"
                        # except AssertionError as err:
                        #     print(f"AssertionError: {err}")
                        #     if not hasattr(self, "_debug_count"):
                        #         self._debug_count = 0
                        #     else:
                        #         self._debug_count += 1
                        #     save_path = f'debug_outputs/0_pipeline_{self._debug_count}.npz'
                        #     import numpy as np
                        #     np.savez_compressed(
                        #         save_path,
                        #         affordance_gt_bin=affordance_gt_bin.cpu().numpy(),
                        #         vis=visible_mask.cpu().numpy(),
                        #         trav=traversable_mask.cpu().numpy(),
                        #         aff=affordance_label.cpu().numpy(),
                        #     )
                        # # end of debug


                        visible_1d = visible_mask.view(visible_mask.shape[0], -1)  # [B, H*W]
                        affordance_1d = affordance_label.view(affordance_label.shape[0], -1)  # [B, H*W]
                        affordance_logits_1d = affordance_logits.view(affordance_logits.shape[0], -1)  # [B, H*W]
                        # loss_aff = torch.nn.functional.binary_cross_entropy_with_logits(
                        #     affordance_logits_1d[visible_1d.bool()], affordance_1d[visible_1d.bool()], reduction="mean"
                        # ) 
                        if not use_diff_loss:
                            if use_prob_loss:
                                affordance_label = affordance_gt_prob.to(input_ids.device)
                                affordance_1d = affordance_label.view(affordance_label.shape[0], -1)  # [B, H*W]
                                # loss_aff = torch.nn.functional.binary_cross_entropy_with_logits(
                                #     affordance_logits_1d, affordance_1d, reduction="mean"
                                # ) # TODO: focal loss if recall low
                                loss_aff = self.finetune_head.py_sigmoid_focal_loss(
                                    affordance_logits_1d, affordance_1d, gamma=2.0, alpha=0.25, reduction="mean"
                                )
                            else:
                                loss_aff = torch.nn.functional.binary_cross_entropy_with_logits(
                                    affordance_logits_1d, affordance_1d, reduction="mean"
                                )
                        else:
                            loss_aff_mse, loss_aff_bce = self.finetune_head.compute_diffusion_loss(
                                # affordance_gt_prob=affordance_gt_prob, # [B, H, W]
                                affordance_label=affordance_label, # [B, H, W]
                            )
                            loss_aff = loss_aff_mse + 0.0 * loss_aff_bce

                        if use_occ_loss:
                            obstacle_1d = (~traversable_mask.bool()).view(traversable_mask.shape[0], -1)
                            obstacle_1d = obstacle_1d & visible_1d.bool()
                            p_free = torch.sigmoid(affordance_logits_1d)
                            loss_forbid = (p_free[obstacle_1d].mean() if obstacle_1d.any() else p_free.new_tensor(0.0))

                        loss = loss_aff + 1.0 * loss_forbid if use_occ_loss else loss_aff
                        return {"loss": loss, "loss_aff": loss_aff, "loss_forbid": loss_forbid if use_occ_loss else torch.tensor(0.0, device=loss_aff.device), \
                                'loss_aff_mse': loss_aff_mse if use_diff_loss else torch.tensor(0.0, device=loss_aff.device), \
                                'loss_aff_bce': loss_aff_bce if use_diff_loss else torch.tensor(0.0, device=loss_aff.device)}

                    elif affordance_logits.dim() == 2: # output is directly [B, 2] for xy regression or B, 12
                        pred_xy = affordance_logits  # [B, 2]
                        if pred_xy.shape[1] ==2:
                            target_xy = data["p_goal"][:, :2].to(pred_xy.device)  # [B, 2]
                            loss_reg = torch.nn.functional.mse_loss(pred_xy, target_xy)
                            loss = loss_reg
                            return {"loss": loss, "loss_reg": loss_reg}
                        elif pred_xy.shape[1] == 12:
                            pred_waypoints_xy = pred_xy.view(-1, 6, 2)  # [B, 6, 2]
                            target_waypoint_xy = data['waypoints'][:, :, :2].to(pred_waypoints_xy.device)  # [B, 6, 2]
                            loss_l1 = torch.nn.functional.l1_loss(pred_waypoints_xy, target_waypoint_xy)
                            loss = loss_l1
                            return {"loss": loss, "loss_l1": loss_l1}

                    elif affordance_logits.dim() == 0: # already the loss
                        return {"loss": affordance_logits}
                    
                    elif affordance_logits.dim() == 1: # 3 losses
                        loss_cls_pos = affordance_logits[0]
                        loss_cls_neg = affordance_logits[1]
                        loss_reg = affordance_logits[2]
                        loss = 10 * (loss_cls_pos + 1 * loss_cls_neg) + loss_reg
                        return {"loss": loss, "loss_cls_pos": loss_cls_pos, "loss_cls_neg": loss_cls_neg, "loss_reg": loss_reg}
                
                    elif affordance_logits.dim() == 4: # per wp mask head B, num_wp, H, W
                        wp_mask = data.get("wp_mask", None)
                        # # debug
                        # save_path = "debug_outputs/2_wp_mask.npz"
                        # import numpy as np
                        # np.savez_compressed(
                        #     save_path,
                        #     wp_mask=wp_mask.cpu().numpy() if wp_mask is not None else None,
                        #     traversable_mask=traversable_mask.cpu().numpy(),
                        #     visible_mask=visible_mask.cpu().numpy(),
                        #     affordance_mask=affordance_label.cpu().numpy(),
                        # )
                        # exit(0)
                        # # end of debug
                        if wp_mask is None:
                            raise KeyError("Missing `wp_mask` in data for waypoint mask supervision.")
                        wp_mask = wp_mask.to(input_ids.device).float()  # (B,num_wp,H,W)
                        wp_mask = wp_mask * traversable_mask.unsqueeze(1) * visible_mask.unsqueeze(1) # only consider traversable and visible area
                        logits_wp = affordance_logits.to(input_ids.device).float()  # (B,num_wp,H,W)
                        print(f"pred shape min max: {logits_wp.shape} {logits_wp.min().item()} {logits_wp.max().item()}")
                        print(f"gt shape min max: {wp_mask.shape} {wp_mask.min().item()} {wp_mask.max().item()}")
                        assert (wp_mask.amin(dim=(2,3)) <= 1e-6).all() and (wp_mask.amax(dim=(2,3)) >= 1 - 1e-6).all(), \
                            "bug: some (b,k) mask is missing 0 or missing 1"
                        with torch.autocast(device_type="cuda", enabled=False):
                            # loss_wp = torch.nn.functional.binary_cross_entropy_with_logits(
                            #     logits_wp.reshape(B*6, 128, 128), wp_mask.reshape(B*6, 128, 128), reduction="mean"
                            # ) * 6.0 # BCE not working
                            loss_wp = self.finetune_head.py_sigmoid_focal_loss(
                                logits_wp.reshape(B*6, 128, 128), wp_mask.reshape(B*6, 128, 128), gamma=2.0, alpha=0.25, reduction="mean"
                            ) * 6.0
                        loss = loss_wp
                        return {"loss": loss, "loss_wp_mask": loss_wp}

                    # pred_occ = self.finetune_head(
                    #     depth, 
                    #     images.permute(0,1,4,2,3),  # to B,4,3,H,W
                    #     intrinsics,
                    #     cam2egos,
                    # )
                    # occ_label = data['occ_label'].to(pred_occ.device).to(pred_occ.dtype)
                    # loss_occ = torch.nn.functional.binary_cross_entropy_with_logits(
                    #     pred_occ, occ_label, reduction="mean"
                    # )
                    # loss = loss_occ
                    # return {"loss": loss, "loss_occ": loss_occ}

                    # pred_r_theta = self.finetune_head(
                    #     finetune_starting_query_feat, 
                    #     finetune_starting_ctx_feat, 
                    #     finetune_starting_skip_result,
                    #     depth, intrinsics
                    # )

                    # a, b, c = pred_r_theta[:, 0], pred_r_theta[:, 1], pred_r_theta[:, 2] # shape [B]
                    # pred_r = torch.sigmoid(c) * (6.4 - 1.0) + 1.0  # scale to [1, 6.4]
                    # u = torch.stack([a, b], dim=-1)
                    # u = torch.nn.functional.normalize(u, dim=-1, eps=1e-8)  # [B,2]
                    # pred_xy = pred_r.unsqueeze(-1) * u  # [B,2]
                    # pred_xy = self.finetune_head(
                    #     finetune_starting_query_feat, 
                    #     finetune_starting_ctx_feat, 
                    #     finetune_starting_skip_result,
                    #     depth, intrinsics
                    # )   # [B,2]
                    # target_xy = data["p_goal"][:, :2].to(pred_xy.device)  # [B, 2]
                    # loss_reg = torch.nn.functional.mse_loss(pred_xy, target_xy)
                    # occ_field = torch.sqrt(data["occ_field"].to(pred_xy.device))  # [B, H, W], in agent local frame
                    # loss_free_space = torch.nn.functional.grid_sample(
                    #     occ_field.unsqueeze(1).float(),
                    #     torch.stack([(((pred_xy[:,1]-(-6.4))/0.2)/(occ_field.shape[2]-1)*2-1),
                    #                 (((pred_xy[:,0]-(-6.4))/0.2)/(occ_field.shape[1]-1)*2-1)], dim=-1).view(-1,1,1,2),
                    #     mode="bilinear", padding_mode="border", align_corners=True
                    # ).view(-1) # [B]
                    # loss_free_space = torch.clamp(loss_free_space, max=5.0).mean()  # cap at 5.0 to avoid extreme outliers
                    # loss = 1*loss_reg + 1*loss_free_space
                    # return {"loss": loss, "loss_free_space": loss_free_space, "loss_reg": loss_reg}
                
                # # - Method 1, use RawChat balanced for training
                # variants = list(variants) if not isinstance(variants, list) else variants
                # num_rawchat = sum(1 for v in variants if v == "RawChat")
                # num_ste = sum(1 for v in variants if v == "SingleTokenEmbedding")
                # assert num_rawchat == num_ste, "Expected equal number of RawChat and SingleTokenEmbedding samples in a batch."
                # out = self._llm_forward(
                #     input_ids=input_ids,
                #     position_ids=position_ids,
                #     attention_mask=attention_mask,
                #     image_flags=image_flags,
                #     pixel_values=concat_images,
                #     labels=labels,
                #     use_cache=use_cache,
                #     output_hidden_states=True,   # we need hidden states for tgt token
                #     return_dict=True, # we need dict output
                # )
                # loss_ce = out.loss # effectively averaged over #per_batch_chat_samples*#average_answer_tokens
                # hidden = out.hidden_states[-1] # B x T x H
                # reg_losses = []
                # for i, var in enumerate(variants):
                #     if var == "RawChat":
                #         # for RawChat, we do not have target token, so skip
                #         continue
                #     elif var == "SingleTokenEmbedding":
                #         tgt_mask = (input_ids[i] == self.tgt_token_idx)
                #         assert tgt_mask.sum() == 1, \
                #             ValueError("Expected exactly one [TGT] token per sample in SingleTokenEmbedding variant.")
                #         tgt_state = hidden[i][tgt_mask].squeeze(0)  # [H]
                #         if self.fuse_visual_tokens:
                #             img_mask = (input_ids[i] == self.model.img_context_token_id)
                #             assert img_mask.sum() == 16*16*4, "Expected 16x16x4 image context tokens."
                #             img_pool = hidden[i][img_mask].mean(dim=0)  # global pool, [H]
                #             fused = torch.cat([tgt_state, img_pool], dim=-1)
                #         else:
                #             fused = tgt_state
                #         if to_predict_region:
                #             pred_region_logits = self.tgt_head_2(fused)  # [16]
                #             target_region = data["region_label"][i].to(pred_region_logits.device)  # scalar
                #             reg_losses.append(torch.nn.functional.cross_entropy(
                #                 pred_region_logits.unsqueeze(0), target_region.unsqueeze(0)
                #             ))
                #         else:
                #             pred_xy = self.tgt_head(fused)
                #             target_xy = data["p_goal"][i, :2].to(pred_xy.device)
                #             reg_losses.append(torch.nn.functional.mse_loss(pred_xy, target_xy))
                #     else:
                #         raise NotImplementedError(f"Unknown variant: {var}")
                # loss_reg = torch.stack(reg_losses).mean() # effectively averaged over #per_batch_ste_samples
                # # return values
                # loss = self.token_ce_weight * loss_ce + self.reg_weight * loss_reg
                # return {"loss": loss, "loss_ce": loss_ce, "loss_reg": loss_reg}
                # - Method 2, don't use RawChat balanced for training
                out = self._llm_forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    image_flags=image_flags,
                    pixel_values=concat_images,
                    labels=None,
                    use_cache=use_cache,
                    output_hidden_states=True,   # we need hidden states for tgt token
                    return_dict=True, # we need dict output
                )
                hidden = out.hidden_states[-1] # B x T x H
                tgt_mask = (input_ids == self.tgt_token_idx) # B x T
                assert tgt_mask.sum(-1).eq(1).all(), "Expected exactly one [TGT] per sample"
                # tgt_state = hidden[:, tgt_mask].squeeze(1)  # [B, H]
                tgt_state = torch.masked_select(hidden, tgt_mask.unsqueeze(-1)).view(hidden.size(0), -1)  # B,T,H and B,T,1 -> B*H -> B,H
                if predict_mode == 'region':
                    # - Method 2.1.1
                    # pred_region_logits = self.tgt_head_2(tgt_state)  # [B, 16]
                    # - Method 2.1.2
                    pred_region_logits = self.tgt_head_3(tgt_state)  # [B, 8]
                    target_region = data["region_label"].to(pred_region_logits.device)  # [B]
                    target_region = target_region // 2 # map 16-region to 8-region
                    loss_reg = torch.nn.functional.cross_entropy(pred_region_logits, target_region)
                    loss_ce = torch.tensor(0.0).to(input_ids.device)
                elif predict_mode == 'xy':
                    # method 2.2.1: direct xy regression
                    # pred_xy = self.tgt_head(tgt_state)  # [B, 2]
                    # target_xy = data["p_goal"][:, :2].to(pred_xy.device)  # [B, 2]
                    # loss_reg = torch.nn.functional.mse_loss(pred_xy, target_xy)

                    # # method 2.2.2: polar coordinate regression
                    # pred_r_theta = self.tgt_head_polar(tgt_state)  # [B, 3]
                    # a, b, c = pred_r_theta[:, 0], pred_r_theta[:, 1], pred_r_theta[:, 2] # shape [B]
                    # pred_r = torch.sigmoid(c) * (6.4 - 1.0) + 1.0  # scale to [1, 6.4]
                    # u = torch.stack([a, b], dim=-1)
                    # u = torch.nn.functional.normalize(u, dim=-1, eps=1e-8)  # [B,2]
                    # pred_xy = pred_r.unsqueeze(-1) * u  # [B,2]
                    # target_xy = data["p_goal"][:, :2].to(pred_xy.device)  # [B, 2]
                    # loss_reg = torch.nn.functional.mse_loss(pred_xy, target_xy)
                    # loss_ce = torch.tensor(0.0).to(input_ids.device)

                    # method 2.2.3 polar coordinate regression with collision penalty
                    use_free_space_loss = False
                    pred_r_theta = self.tgt_head_polar(tgt_state)  # [B, 3]
                    a, b, c = pred_r_theta[:, 0], pred_r_theta[:, 1], pred_r_theta[:, 2] # shape [B]
                    pred_r = torch.sigmoid(c) * (6.4 - 1.0) + 1.0  # scale to [1, 6.4]
                    u = torch.stack([a, b], dim=-1)
                    u = torch.nn.functional.normalize(u, dim=-1, eps=1e-8)  # [B,2]
                    pred_xy = pred_r.unsqueeze(-1) * u  # [B,2]
                    target_xy = data["p_goal"][:, :2].to(pred_xy.device)  # [B, 2]
                    loss_reg = torch.nn.functional.mse_loss(pred_xy, target_xy)
                    if use_free_space_loss:
                        # occ_field = torch.sqrt(data["occ_field"].to(pred_xy.device))  # [B, H, W], in agent local frame
                        occ_field = data["occ_field"].to(pred_xy.device).float()   # if occ_field already is your L1 meters penalty
                        # occ_field = (~data["traversable_mask"].to(pred_xy.device).bool()).float()  # [B,H,W], 0 ok, 1 bad
                        resolution = 0.1
                        # loss_free_space = torch.nn.functional.grid_sample(
                        #     occ_field.unsqueeze(1).float(),
                        #     torch.stack([(((pred_xy[:,1]-(-6.4))/resolution)/(occ_field.shape[2]-1)*2-1),
                        #                 (((pred_xy[:,0]-(-6.4))/resolution)/(occ_field.shape[1]-1)*2-1)], dim=-1).view(-1,1,1,2),
                        #     mode="bilinear", padding_mode="border", align_corners=True
                        # ).view(-1) # [B]
                        loss_free_space = torch.nn.functional.grid_sample(
                            occ_field.unsqueeze(1).float(),
                            torch.stack([(((pred_xy[:,1]-(-6.4))/resolution)/(occ_field.shape[2])*2-1),
                                        (((pred_xy[:,0]-(-6.4))/resolution)/(occ_field.shape[1])*2-1)], dim=-1).view(-1,1,1,2),
                            mode="bilinear", padding_mode="border", align_corners=False
                        ).view(-1)
                        loss_free_space = torch.clamp(loss_free_space, max=5.0).mean()  # cap at 5.0 to avoid extreme outliers
                    else:
                        loss_free_space = torch.tensor(0.0).to(pred_xy.device)
                    loss = 1*loss_reg + 0.5*loss_free_space
                    return {"loss": loss, "loss_free_space": loss_free_space, "loss_reg": loss_reg}
                elif predict_mode == 'waypoints':
                    pred_waypoints_xy = self.tgt_head_waypoints(tgt_state)  # [B, 12]
                    pred_waypoints_xy = pred_waypoints_xy.view(-1, 6, 2)  # [B, 6, 2]
                    target_waypoint_xy = data['waypoints'][:, :, :2].to(pred_waypoints_xy.device)  # [B, 6, 2]
                    loss_l1 = torch.nn.functional.l1_loss(pred_waypoints_xy, target_waypoint_xy)

                    use_free_space_loss = False
                    if use_free_space_loss:
                        occ_field = torch.sqrt(data["occ_field"].to(pred_waypoints_xy.device))  # [B, H, W], in agent local frame
                        B, num_wp, _ = pred_waypoints_xy.shape
                        pred_wp_xy_reshaped = pred_waypoints_xy.view(B * num_wp, 2)
                        loss_free_space = torch.nn.functional.grid_sample(
                            occ_field.unsqueeze(1).float().repeat_interleave(num_wp, dim=0),
                            torch.stack([(((pred_wp_xy_reshaped[:,1]-(-6.4))/0.2)/(occ_field.shape[2]-1)*2-1),
                                        (((pred_wp_xy_reshaped[:,0]-(-6.4))/0.2)/(occ_field.shape[1]-1)*2-1)], dim=-1).view(-1,1,1,2),
                            mode="bilinear", padding_mode="border", align_corners=True
                        ).view(-1) # [B * num_wp]
                        loss_free_space = torch.clamp(loss_free_space, max=5.0).mean()  # cap at 5.0 to avoid extreme outliers
                        loss = loss_l1 + 0.5 * loss_free_space

                    else:
                        loss_free_space = torch.tensor(0.0, device=loss_l1.device)
                        loss = loss_l1

                    return {"loss": loss, "loss_l1": loss_l1, "loss_free_space": loss_free_space}
                else:
                    raise NotImplementedError(f"Unknown predict_mode: {predict_mode}")

            else:
                if is_fine_tune:
                    amp_ctx = autocast(dtype=torch.bfloat16)
                    forward_prediction_reults = [] # mmengine requires this to be a list
                    with amp_ctx:
                        out = self._llm_forward(
                            input_ids=input_ids,
                            position_ids=position_ids,
                            attention_mask=attention_mask,
                            image_flags=image_flags,
                            pixel_values=concat_images,
                            labels=None,  # no labels in conversation
                            # labels=labels, # BUG
                            use_cache=use_cache,
                            output_hidden_states=True,   # we need hidden states for tgt token
                            return_dict=True, # we need dict output
                            data=data,
                        )
                    hidden = out.hidden_states[-1] # B x T x H
                    tgt_mask = (input_ids == self.tgt_token_idx) # B x T
                    tgt_state = hidden[tgt_mask] # B, H
                    B, N, fH, fW, H = hidden.shape[0], 4, 16, 16, hidden.shape[2]

                    img_mask = (input_ids == self.model.img_context_token_id) # B x T
                    assert img_mask.sum(-1).eq(16*16*4).all(), "Expected 16x16x4 image context tokens."
                    img_states = hidden[img_mask].contiguous().view(B, 4*16*16, H) # B x (16*16*4) x H
                    img_states = img_states.view(B, N, fH, fW, H)  # B x 4 x fH=16 x fW=16 x H # N=view first, row major

                    pred_r_theta = self.tgt_head_polar(tgt_state)  # [B, 3]
                    # finetune_starting_query_feat = tgt_state.detach()
                    finetune_starting_query_feat = tgt_state
                    finetune_starting_ctx_feat = img_states
                    # finetune_starting_ctx_feat = None
                    # finetune_starting_skip_result = pred_r_theta.detach()
                    finetune_starting_skip_result = None

                    depth = data["depth"].to(input_ids.device) # [B, 4, H, W]
                    images = data['raw_img'] # (B, 4, H, W, 3) torch.uint8 BGR
                    cam2imgs = data["cam2imgs"]
                    cam2egos = data["cam2egos"]
                    if isinstance(cam2imgs, dict):
                        order = ["front", "left", "back", "right"]
                        cam2imgs = torch.stack([cam2imgs[k] for k in order], dim=1)
                        cam2egos = torch.stack([cam2egos[k] for k in order], dim=1)
                    cam2imgs = cam2imgs.to(input_ids.device)
                    cam2egos = cam2egos.to(input_ids.device)
                    intrinsics = cam2imgs[:, :, :3, :3]

                    # # Method 1: predict waypoints directly
                    # pred_waypoints_xy = self.finetune_head(
                    #     finetune_starting_query_feat,
                    #     finetune_starting_skip_result,
                    #     depth,
                    #     images.permute(0,1,4,2,3),  # to B,4,3,H,W
                    #     intrinsics,
                    #     cam2egos,
                    # )   # [B, 6, 2]
                    # target_waypoint_xy = data['waypoints'][:, :, :2].to(pred_waypoints_xy.device)  # [B, 6, 2]
                    # loss_l1 = torch.nn.functional.l1_loss(pred_waypoints_xy, target_waypoint_xy)
                    # pred_xy = pred_waypoints_xy[0, -1] # shape [2]
                    # print(f"[Eval] Finetune predicted waypoint L1 loss: {loss_l1.item()}")

                    # use_free_space_loss = False
                    # if use_free_space_loss:
                    #     occ_field = torch.sqrt(data["occ_field"].to(pred_waypoints_xy.device))  # [B, H, W], in agent local frame
                    #     B, num_wp, _ = pred_waypoints_xy.shape
                    #     pred_wp_xy_reshaped = pred_waypoints_xy.view(B * num_wp, 2)
                    #     loss_free_space = torch.nn.functional.grid_sample(
                    #         occ_field.unsqueeze(1).float().repeat_interleave(num_wp, dim=0),
                    #         torch.stack([(((pred_wp_xy_reshaped[:,1]-(-6.4))/0.2)/(occ_field.shape[2]-1)*2-1),
                    #                     (((pred_wp_xy_reshaped[:,0]-(-6.4))/0.2)/(occ_field.shape[1]-1)*2-1)], dim=-1).view(-1,1,1,2),
                    #         mode="bilinear", padding_mode="border", align_corners=True
                    #     ).view(-1) # [B * num_wp]
                    #     loss_free_space = torch.clamp(loss_free_space, max=5.0).mean()  # cap at 5.0 to avoid extreme outliers
                    #     loss = loss_l1 + 0.5 * loss_free_space
                    #     print(f"[Eval] Finetune predicted waypoint free space loss: {loss_free_space.item()}")
                    # else:
                    #     loss = loss_l1
                    #     loss_free_space = torch.tensor(0.0, device=loss_l1.device)
                    
                    # return [{"p_pred": pred_xy, "loss_l1": loss_l1, "loss": loss, 'path_pred': pred_waypoints_xy[0], \
                    #                  "loss_free_space": loss_free_space, }]

                    # # Method 2: score on anchors
                    # pred_anchor_scores = self.finetune_head(
                    #     finetune_starting_query_feat,
                    #     finetune_starting_skip_result,
                    #     depth,
                    #     images.permute(0,1,4,2,3),  # to B,4,3,H,W
                    #     intrinsics,
                    #     cam2egos,
                    # )   # [B, num_anchors]
                    # target_anchor_idx, _ = self.finetune_head.find_best_waypoint_anchor(
                    #     data['waypoints'][:, :, :2].to(pred_anchor_scores.device)  # [B, 6, 2]
                    # ) # [B]
                    # target_onehot = torch.zeros_like(pred_anchor_scores)
                    # target_onehot.scatter_(1, target_anchor_idx.unsqueeze(1), 1.0)
                    # # loss_cls = self.finetune_head.py_sigmoid_focal_loss(
                    # #     pred_anchor_scores,
                    # #     target_onehot,
                    # #     gamma=2.0,
                    # #     alpha=0.25,
                    # #     reduction='mean'
                    # # ) * 20.0  # scale up to match magnitude of other losses
                    # loss_cls = self.finetune_head.ade_soft_anchor_ce_loss(
                    #     pred_logits=pred_anchor_scores,
                    #     gt_waypoints_xy=data["waypoints"][:, :, :2].to(pred_anchor_scores.device),
                    #     sigma=0.5,
                    #     topk=None,   # or an int
                    # )
                    # pred_anchor_idx = torch.argmax(pred_anchor_scores, dim=1)  # [B]
                    # path_pred = self.finetune_head.plan_anchor[pred_anchor_idx][0] # 6, 2
                    # pred_xy = path_pred[-1]  # final waypoint as prediction
                    # print(f"[Eval] Finetune predicted anchor classification loss: {loss_cls.item()}")
                    # return [{"p_pred": pred_xy, "loss_cls": loss_cls, "loss": loss_cls, 'path_pred': path_pred }]

                    # # Method 3: score on anchors plus offsets
                    # pred_anchor_scores, pred_offsets = self.finetune_head(
                    #     finetune_starting_query_feat,
                    #     finetune_starting_skip_result,
                    #     depth,
                    #     images.permute(0,1,4,2,3),  # to B,4,3,H,W
                    #     intrinsics,
                    #     cam2egos,
                    # )   # [B, num_anchors]; [B, num_anchors, 6, 2]
                    # target_anchor_idx, target_anchor = self.finetune_head.find_best_waypoint_anchor(
                    #     data['waypoints'][:, :, :2].to(pred_anchor_scores.device)  # [B, 6, 2]
                    # ) # [B]
                    # target_onehot = torch.zeros_like(pred_anchor_scores)
                    # target_onehot.scatter_(1, target_anchor_idx.unsqueeze(1), 1.0)
                    # # loss_cls = self.finetune_head.py_sigmoid_focal_loss(
                    # #     pred_anchor_scores,
                    # #     target_onehot,
                    # #     gamma=2.0,
                    # #     alpha=0.25,
                    # #     reduction='mean'
                    # # ) * 20.0  # scale up to match magnitude of other losses
                    # loss_cls = self.finetune_head.ade_soft_anchor_ce_loss(
                    #     pred_logits=pred_anchor_scores,
                    #     gt_waypoints_xy=data["waypoints"][:, :, :2].to(pred_anchor_scores.device),
                    #     sigma=0.5,
                    #     topk=None,   # or an int
                    # )
                    # pred_anchor_idx = torch.argmax(pred_anchor_scores, dim=1)  # [B]
                    # model_selected_anchor = self.finetune_head.plan_anchor[pred_anchor_idx][0] # 6, 2
                    # model_selected_offsets = torch.gather(
                    #     pred_offsets,
                    #     1,
                    #     pred_anchor_idx.view(-1, 1, 1).unsqueeze(-1).repeat(1, 1, pred_offsets.shape[2], 2)
                    # ).squeeze(1)  # [B, 6, 2]
                    # path_pred = model_selected_anchor + model_selected_offsets  # [B, 6, 2]
                    # path_pred = path_pred[0]
                    # pred_xy = path_pred[-1]  # final waypoint as prediction
                    # print(f"[Eval] Finetune predicted anchor classification loss: {loss_cls.item()}")
                    # loss_l1 = torch.nn.functional.l1_loss(path_pred, data['waypoints'][0, :, :2].to(path_pred.device))
                    # print(f"[Eval] Finetune predicted waypoint L1 loss: {loss_l1.item()}")
                    # loss = 2 * loss_cls + loss_l1 # scale to mimic the original 100:1 when the cls is focal loss
                    # return [{"p_pred": pred_xy, "loss_cls": loss_cls, "loss": loss, 'path_pred': path_pred, 'loss_l1': loss_l1 }]

                    # Method 3: direct bev affordance
                    use_occ_loss = False
                    use_diff_loss = False
                    use_multiple_samples = False
                    affordance_label = data['affordance_mask'].to(input_ids.device).float()
                    visible_mask = data['visible_mask'].to(input_ids.device)
                    traversable_mask = data['traversable_mask'].to(input_ids.device)
                    affordance_logits = self.finetune_head(
                        finetune_starting_query_feat,
                        finetune_starting_skip_result,
                        depth,
                        images.permute(0,1,4,2,3),  # to B,4,3,H,W
                        intrinsics,
                        cam2egos,
                        context_tokens=finetune_starting_ctx_feat,
                        sample_k=10 if use_multiple_samples else 1,
                        affordance_mask=affordance_label, # only for debug
                        traversable_mask=traversable_mask, # only for debug
                        visible_mask=visible_mask, # only for debug
                        instruction=data['instruction'] if 'instruction' in data else None, # only for debug
                        gt_xy=data['p_goal'][:, :2].to(input_ids.device) if 'p_goal' in data else None, # only for debug
                    )   # [B, H, W] or [B, K, H, W]
                    if affordance_logits.dim() == 3 and affordance_logits.shape[-1]==128: # B, H, W
                        # affordance_label = affordance_label * traversable_mask  # only consider traversable area
                        # ablation version: changeable geodesic radius
                        affordance_label = data['affordance_gt_bin'].to(input_ids.device).float()
                        visible_1d = visible_mask.view(visible_mask.shape[0], -1)  # [B, H*W]
                        
                        use_pseudo_label = False
                        if not use_pseudo_label:
                            affordance_label = affordance_label * traversable_mask * visible_mask
                        else:
                            raise NotImplementedError

                        affordance_1d = affordance_label.view(affordance_label.shape[0], -1)  # [B, H*W]
                        if use_multiple_samples:
                            B, K, H, W = affordance_logits.shape
                            # repeat GT for each sample: [B,H*W] -> [B*K,H*W]
                            affordance_1d = affordance_1d.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
                            visible_1d    = visible_1d.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)

                        if not use_multiple_samples:
                            affordance_logits_1d = affordance_logits.view(affordance_logits.shape[0], -1)  # [B, H*W]
                        else:
                            affordance_logits_1d = affordance_logits.view(B*K, H*W)  # [B*K, H*W]
                        if not use_diff_loss:
                            loss_aff = torch.nn.functional.binary_cross_entropy_with_logits(
                                # affordance_logits_1d[visible_1d.bool()], affordance_1d[visible_1d.bool()], reduction="mean"
                                affordance_logits_1d, affordance_1d, reduction="mean" # fix here to match training
                            ) 
                        else:
                            loss_aff_mse, loss_aff_bce = self.finetune_head.compute_diffusion_loss(
                                # affordance_gt_prob=data['affordance_gt_prob'].to(affordance_logits.device), # [B, H, W]
                                affordance_label=affordance_label, # [B, H, W]
                            )
                            loss_aff = loss_aff_mse + 0.0 * loss_aff_bce
                        loss = loss_aff

                        free_ground_obs, free_cam_obs = self.finetune_head.prepare_free_obs(
                            depth,            # (B, N, H, W) meters
                            cam2egos,      # (B, N, 4, 4)
                            intrinsics,       # (B, N, 3, 3)
                            grid_config=dict(
                                x=[-6.4, 6.4, 0.1],       # 128
                                y=[-6.4, 6.4, 0.1],       # 128
                            ),
                            Dx=128, Dy=128,           # BEV grid size
                            ground_z_range=(-1.6, -1.4),
                            cam_z_range=(-0.1, 0.1),
                            blocker_z_range=(-1.4, -0.1),
                            old_behavior=False,
                            # blocker_z_range=(-1.6, 0.2),
                            # old_behavior=True,
                        )
                        visible_online = free_cam_obs[0, 0].bool() # [H,W], True=valid

                        pred_xy = self.finetune_head.extract_xy_from_affordance(
                            affordance_logits[0], 
                            # visible_mask=visible_mask[0], 
                            # tranversable_mask=traversable_mask[0]
                            # visible_mask=visible_online,
                            visible_mask=None, # BUG: if we do this snap, we must make it taller than human otherwise any pred behind human will vanish
                            traversable_mask=None,
                        )  # shape [2] or K, 2
                        if not use_diff_loss:
                            print(f"[Eval] Finetune predicted affordance loss: {loss_aff.item()}")
                        else:
                            print(f"[Eval] Finetune predicted affordance MSE loss: {loss_aff_mse.item()}, BCE loss: {loss_aff_bce.item()}")
                        affordance_binary = torch.sigmoid(affordance_logits)
                        affordance_binary = (affordance_binary > 0.5).float()
                        # debug
                        # self.finetune_head.debug_vis(data, affordance_logits, pred_xy, exit=False)

                        return [{"loss": loss, "loss_aff": loss_aff, 'pred_xy': pred_xy, 'pred_aff': affordance_binary[0], \
                                'free_ground_obs': free_ground_obs[0, 0].bool(), 'free_cam_obs': free_cam_obs[0, 0].bool(), \
                                    'loss_aff_mse': loss_aff_mse if use_diff_loss else torch.tensor(0.0, device=loss_aff.device), \
                                'loss_aff_bce': loss_aff_bce if use_diff_loss else torch.tensor(0.0, device=loss_aff.device)}  ]

                    elif affordance_logits.shape[-1] == 12 or affordance_logits.shape[-1] == 2: # point regression output
                        pred_xy = affordance_logits
                        # if pred_xy.dim() == 3:
                        #     assert pred_xy.shape[1] == 1, "K>1 not implemented"
                        #     pred_xy = pred_xy.squeeze(1) # [B, 2]
                        if pred_xy.shape[1] == 12:
                            pred_waypoints_xy = affordance_logits.view(-1, 6, 2)  # [B, 6, 2]
                            target_waypoint_xy = data['waypoints'][:, :, :2].to(pred_waypoints_xy.device)  # [B, 6, 2]
                            loss_l1 = torch.nn.functional.l1_loss(pred_waypoints_xy, target_waypoint_xy)
                            placeholder = torch.zeros_like(data['affordance_mask'][0])
                            print(f"[Eval] Finetune predicted waypoint L1 loss: {loss_l1.item()}")
                            forward_prediction_reults.append({"loss_l1": loss_l1, "loss": loss_l1, 'pred_waypoints': pred_waypoints_xy[0], 'pred_xy': pred_waypoints_xy[0, -1], \
                                                              'pred_aff': placeholder, \
                                                            'free_ground_obs': placeholder.bool(), 'free_cam_obs': placeholder.bool(), })
                            return forward_prediction_reults
                        elif pred_xy.shape[1] == 2:
                            from xtuner.perception_modules.affordance_head import AffordanceHead
                            free_ground_obs, free_cam_obs = AffordanceHead.prepare_free_obs(
                                depth,            # (B, N, H, W) meters
                                cam2egos,      # (B, N, 4, 4)
                                intrinsics,       # (B, N, 3, 3)
                                grid_config=dict(
                                    x=[-6.4, 6.4, 0.1],       # 128
                                    y=[-6.4, 6.4, 0.1],       # 128
                                ),
                                Dx=128, Dy=128,           # BEV grid size
                                ground_z_range=(-1.6, -1.4),
                                cam_z_range=(-0.1, 0.1),
                                blocker_z_range=(-1.4, -0.1),
                                old_behavior=False,
                                # blocker_z_range=(-1.6, 0.2),
                                # old_behavior=True,
                            )
                            target_xy = data["p_goal"][:, :2].to(pred_xy.device)  # [B, 2]
                            loss_reg = torch.nn.functional.mse_loss(pred_xy, target_xy)
                            loss = loss_reg
                            print(f"[Eval] Finetune predicted point regression loss: {loss_reg.item()}")

                            valid=free_ground_obs[0, 0].bool() # [H,W], True=valid
                            snap_to_depth_free = False
                            if snap_to_depth_free:
                                wp=pred_xy # [1,2], x=fwd,y=left in meters
                                hf=(wp[:,0]+6.4)/0.1;wf=(wp[:,1]+6.4)/0.1 # float grid coords (keep sub-cell info)
                                hi=torch.clamp(torch.floor(hf).long(),0,127);wi=torch.clamp(torch.floor(wf).long(),0,127) # base cell
                                for i in range(wp.shape[0]):
                                    if valid[hi[i],wi[i]]:continue # already valid
                                    h0=hf[i];w0=wf[i]
                                    for r in range(1,128):
                                        hmin=max(0,hi[i].item()-r);hmax=min(127,hi[i].item()+r);wmin=max(0,wi[i].item()-r);wmax=min(127,wi[i].item()+r)
                                        sub=valid[hmin:hmax+1,wmin:wmax+1]
                                        if not sub.any():continue
                                        ys,xs=torch.nonzero(sub,as_tuple=True);hs=ys.to(h0.dtype)+hmin+0.5;ws=xs.to(w0.dtype)+wmin+0.5 # candidate cell centers in grid coords
                                        j=torch.argmin((hs-h0)**2+(ws-w0)**2);wp[i,0]=hs[j]*0.1-6.4;wp[i,1]=ws[j]*0.1-6.4 # snap to center in meters
                                        break
                                pred_xy_snap=wp[0]
                            else:
                                pred_xy_snap = pred_xy[0]
                            pred_aff = torch.zeros_like(data['affordance_mask'][0])
                            forward_prediction_reults.append({"loss_reg": loss_reg, "loss": loss, \
                                                            'pred_aff': pred_aff, "pred_xy": pred_xy_snap, \
                                                                'free_ground_obs': free_ground_obs[0, 0].bool(), 'free_cam_obs': free_cam_obs[0, 0].bool(), })
                            return forward_prediction_reults

                    elif affordance_logits.dim() == 4: # B, num_wp, H, W
                        # predict per-waypoint xy via per-channel argmax
                        pred_waypoints = self.finetune_head.extract_xy_from_affordance(
                            affordance_logits[0], visible_mask=None, traversable_mask=None
                        )  # (num_wp, 2)
                        pred_xy = pred_waypoints[-1]
                        placeholder = torch.zeros_like(data['affordance_mask'][0])
                        logits_wp = affordance_logits[0] # [num_wp, H, W]
                        wp_mask = data['wp_mask'][0].to(logits_wp.device).float()  # [num_wp, H, W]
                        # # debug
                        # save_path = "debug_outputs/2_wp_mask.npz"
                        # import numpy as np
                        # np.savez_compressed(
                        #     save_path,
                        #     wp_mask=wp_mask.cpu().numpy() if wp_mask is not None else None,
                        #     traversable_mask=traversable_mask.cpu().numpy(),
                        #     visible_mask=visible_mask.cpu().numpy(),
                        #     affordance_mask=affordance_label.cpu().numpy(),
                        #     pred_wp_mask=torch.sigmoid(logits_wp).cpu().numpy(),
                        #     pred_waypoints=pred_waypoints.cpu().numpy(),
                        # )
                        # exit(0)
                        # # end of debug
                        loss_wp = torch.nn.functional.binary_cross_entropy_with_logits(
                            logits_wp, wp_mask, reduction="mean"
                        )
                        print(f"[Eval] Finetune predicted waypoint heatmap loss: {loss_wp.item()}")
                        forward_prediction_reults.append(
                            {
                                "loss": loss_wp,
                                "pred_waypoints": pred_waypoints,
                                "pred_xy": pred_xy,
                                "pred_aff": placeholder,
                                "free_ground_obs": placeholder.bool(),
                                "free_cam_obs": placeholder.bool(),
                                "loss_wp": loss_wp,
                            }
                        )
                        return forward_prediction_reults

                    # pred_r_theta = self.finetune_head(
                    #     finetune_starting_query_feat, 
                    #     finetune_starting_ctx_feat, 
                    #     finetune_starting_skip_result,
                    #     depth, intrinsics
                    # )

                    # a, b, c = pred_r_theta[:, 0], pred_r_theta[:, 1], pred_r_theta[:, 2] # shape [B]
                    # pred_r = torch.sigmoid(c) * (6.4 - 1.0) + 1.0  # scale to [1, 6.4]
                    # u = torch.stack([a, b], dim=-1)
                    # u = torch.nn.functional.normalize(u, dim=-1, eps=1e-8)  # [B,2]
                    # pred_xy = pred_r.unsqueeze(-1) * u  # [B,2]
                    # pred_xy = self.finetune_head(
                    #     finetune_starting_query_feat, 
                    #     finetune_starting_ctx_feat, 
                    #     finetune_starting_skip_result,
                    #     depth, intrinsics
                    # )   # [B,2]
                    # target_xy = data["p_goal"][:, :2].to(pred_xy.device)  # [B, 2]
                    # loss_reg = torch.nn.functional.mse_loss(pred_xy, target_xy)
                    # # occ_field = torch.sqrt(data["occ_field"].to(pred_xy.device))  # [B, H, W], in agent local frame
                    # # loss_free_space = torch.nn.functional.grid_sample(
                    # #     occ_field.unsqueeze(1).float(),
                    # #     torch.stack([(((pred_xy[:,1]-(-6.4))/0.2)/(occ_field.shape[2]-1)*2-1),
                    # #                 (((pred_xy[:,0]-(-6.4))/0.2)/(occ_field.shape[1]-1)*2-1)], dim=-1).view(-1,1,1,2),
                    # #     mode="bilinear", padding_mode="border", align_corners=True
                    # # ).view(-1) # [B]
                    # loss_free_space = torch.tensor(0.0).to(input_ids.device)
                    # loss_free_space = torch.clamp(loss_free_space, max=5.0).mean()  # cap at 5.0 to avoid extreme outliers
                    # loss = 1*loss_reg + 1*loss_free_space
                    # pred_xy = pred_xy[0]
                    # target_xy = target_xy[0]
                    # pred_dir_id = int((torch.atan2(pred_xy[1], pred_xy[0]) + math.pi/8) / (2 * math.pi) * 8) % 8
                    # target_dir_id = int((torch.atan2(target_xy[1], target_xy[0]) + math.pi/8) / (2 * math.pi) * 8) % 8
                    # dstep, succ0_dir, succ1_dir = self._success_at_0_and_1_dir_only(
                    #     pred_dir_id, target_dir_id
                    # )
                    # return [{"p_pred": pred_xy, 'loss_reg': loss_reg, 'loss': loss, 'loss_free_space': loss_free_space, \
                    #                                     "dstep": dstep, "succ0_dir": succ0_dir, \
                    #                                     "succ1_dir": succ1_dir, }]
                    # pred_occ = self.finetune_head(
                    #     depth, 
                    #     images.permute(0,1,4,2,3),  # to B,4,3,H,W
                    #     intrinsics,
                    #     cam2egos,
                    # )
                    # occ_label = data['occ_label'].to(pred_occ.device).to(pred_occ.dtype)
                    # loss_occ = torch.nn.functional.binary_cross_entropy_with_logits(
                    #     pred_occ, occ_label, reduction="mean"
                    # )
                    # pred_occ_binary = torch.sigmoid(pred_occ)
                    # pred_occ_binary = (pred_occ > 0.5).float()
                    # loss = loss_occ
                    # print(f"[Eval] Finetune predicted occupancy loss: {loss_occ.item()}")
                    # return [{"pred_occ": pred_occ_binary[0], "loss_occ": loss_occ, "loss": loss, "p_pred": torch.zeros(2, device=input_ids.device), }]
                amp_ctx = autocast(dtype=torch.bfloat16)
                forward_prediction_reults = [] # mmengine requires this to be a list
                with amp_ctx:
                    out = self._llm_forward(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        image_flags=image_flags,
                        pixel_values=concat_images,
                        labels=None,  # no labels in eval
                        # labels=labels, # BUG
                        use_cache=use_cache,
                        output_hidden_states=True,   # we need hidden states for tgt token
                        return_dict=True, # we need dict output
                    )
                loss_ce = out.loss 
                assert loss_ce is None, "No CE loss in eval mode."
                loss_ce = torch.tensor(0.0).to(input_ids.device)
                hidden = out.hidden_states[-1] # B x T x H
                reg_losses = []
                assert len(variants) == 1, "Eval batch size should be 1."
                for i in range(len(variants)):
                    tgt_mask = (input_ids[i] == self.tgt_token_idx)
                    assert tgt_mask.sum() == 1, \
                        ValueError("Expected exactly one [TGT] token per sample in eval.")
                    tgt_state = hidden[i][tgt_mask].squeeze(0)  # [H]
                    if self.fuse_visual_tokens:
                        img_mask = (input_ids[i] == self.model.img_context_token_id)
                        assert img_mask.sum() == 16*16*4, "Expected 16x16x4 image context tokens."
                        img_pool = hidden[i][img_mask].mean(dim=0)  # global pool, [H]
                        fused = torch.cat([tgt_state, img_pool], dim=-1)
                    else:
                        fused = tgt_state
                    if predict_mode == 'region':
                        target_region = data["region_label"][i].to(input_ids.device)  # scalar
                        # - Method 2.1.1
                        # pred_region_logits = self.tgt_head_2(fused)  # [16]
                        # - Method 2.1.2
                        pred_region_logits = self.tgt_head_3(fused)  # [8]

                        pred_region = pred_region_logits.argmax(dim=-1)  # scalar
                        pred_region = pred_region * 2  # map back to 16-region

                        print(f"[Eval] Predicted region: {pred_region.item()}")
                        # loss_reg = torch.nn.functional.cross_entropy(
                        #     pred_region_logits.unsqueeze(0), target_region.unsqueeze(0)
                        # )
                        loss_reg = torch.tensor(0.0).to(input_ids.device)  # no reg loss in eval; keep as 0 for logging
                        loss = self.token_ce_weight * loss_ce + self.reg_weight * loss_reg
                        text, text_label = self.parse_pred_to_meta_action(
                            pred_region, target_region
                        )
                        move_dir, move_dir_label, range_tag, range_tag_label, dstep, succ1_dir, succ1_dir_and_size, succ0_dir, succ0_dir_and_size = self.parse_and_predict_meta_action(
                            text, text_label, data)
                        forward_prediction_reults.append(
                            {"p_pred": torch.zeros(2, device=input_ids.device), "text_pred": text, \
                                "dstep": dstep, "succ1_dir": succ1_dir, "succ1_dir_and_size": succ1_dir_and_size, \
                                    "succ0_dir": succ0_dir, "succ0_dir_and_size": succ0_dir_and_size, \
                            }
                        )
                    elif predict_mode == 'xy':
                        # - Method 1: direct xy regression
                        # pred_xy = self.tgt_head(fused)
                        # target_xy = data["p_goal"][i, :2].to(pred_xy.device)
                        # loss_reg = torch.nn.functional.mse_loss(pred_xy, target_xy)
                        # - Method 2: polar coordinate regression
                        pred_r_theta = self.tgt_head_polar(fused)  # [3]
                        a, b, c = pred_r_theta[0], pred_r_theta[1], pred_r_theta[2]
                        pred_r = torch.sigmoid(c) * (6.4 - 1.0) + 1.0  # scale to [1, 6.4]
                        u = torch.stack([a, b], dim=-1)
                        u = torch.nn.functional.normalize(u, dim=-1, eps=1e-8)  # [2]
                        pred_xy = pred_r.unsqueeze(-1) * u  # [2]
                        target_xy = data["p_goal"][i, :2].to(pred_xy.device)
                        loss_reg = torch.nn.functional.mse_loss(pred_xy, target_xy)

                        loss = self.token_ce_weight * loss_ce + self.reg_weight * loss_reg
                        # forward_prediction_reults.append({"p_pred": pred_xy, 'loss_reg': loss_reg, 'loss': loss})
                        # debug plot
                        # current_t = data["ep_idx"][i].item()
                        # s = str(data['conversations'][0])
                        # m = re.search(r"Instruction:\s*(.*?)(?:\n\s*\[TGT\]|\Z)", s, flags=re.S)
                        # if m:
                        #     instruction = m.group(1).strip()
                        # self.oracle_bev_generator.debug_plot_pred_and_goal(
                        #     yaw_deg=math.degrees(data['agent_yaw_hab_rad'][0].item()),
                        #     target_token=data['viewpoint_hash'][0],
                        #     pred_xy=pred_xy.detach().cpu().numpy(),
                        #     debug_path=f'debug_outputs/affordance_eval_result_{current_t}.jpg',
                        #     goal_xy=data['p_goal'][0, :2].cpu().numpy(),
                        #     instruction=instruction,
                        # )
                        pred_dir_id = int((torch.atan2(pred_xy[1], pred_xy[0]) + math.pi/8) / (2 * math.pi) * 8) % 8
                        target_dir_id = int((torch.atan2(target_xy[1], target_xy[0]) + math.pi/8) / (2 * math.pi) * 8) % 8
                        dstep, succ0_dir, succ1_dir = self._success_at_0_and_1_dir_only(
                            pred_dir_id, target_dir_id
                        )
                        snap_result_using_oracle_bev_label = True
                        if snap_result_using_oracle_bev_label:
                            trav=data['traversable_mask'][0].to(pred_xy.device).bool();vis=data['visible_mask'][0].to(pred_xy.device).bool();
                            valid=vis # only use visible area
                            # valid = vis & trav  # only use traversable area
                            p=pred_xy # [2], x=fwd,y=left in meters
                            hf=(p[0]+6.4)/0.1;wf=(p[1]+6.4)/0.1 # float grid coords
                            hi=int(torch.clamp(torch.floor(hf).long(),0,127).item());wi=int(torch.clamp(torch.floor(wf).long(),0,127).item())
                            if not valid[hi,wi]:
                                h0=hf;w0=wf
                                for r in range(1,128):
                                    hmin=max(0,hi-r);hmax=min(127,hi+r);wmin=max(0,wi-r);wmax=min(127,wi+r)
                                    sub=valid[hmin:hmax+1,wmin:wmax+1]
                                    if not sub.any():continue
                                    ys,xs=torch.nonzero(sub,as_tuple=True);hs=ys.to(h0.dtype)+hmin+0.5;ws=xs.to(w0.dtype)+wmin+0.5
                                    j=torch.argmin((hs-h0)**2+(ws-w0)**2);pred_xy=torch.stack([hs[j]*0.1-6.4,ws[j]*0.1-6.4]).to(p.dtype)
                                    break

                        pred_aff = torch.zeros_like(data['affordance_mask'][0])
                        forward_prediction_reults.append({"p_pred": pred_xy, 'loss_reg': loss_reg, 'loss': loss, \
                                                          "dstep": dstep, "succ0_dir": succ0_dir, \
                                                          "succ1_dir": succ1_dir, 'pred_aff': pred_aff, "pred_xy": pred_xy, })
                    elif predict_mode == 'waypoints':
                        pred_waypoints_xy = self.tgt_head_waypoints(tgt_state)  # [B, 12]
                        pred_waypoints_xy = pred_waypoints_xy.view(-1, 6, 2)  # [B, 6, 2]
                        target_waypoint_xy = data['waypoints'][:, :, :2].to(pred_waypoints_xy.device)  # [B, 6, 2]
                        loss_l1 = torch.nn.functional.l1_loss(pred_waypoints_xy, target_waypoint_xy)
                        pred_xy = pred_waypoints_xy[0, -1] # shape [2]

                        # # debug
                        # if not hasattr(self, 'debug_cnt'):
                        #     self.debug_cnt = 0
                        # save_path = f'debug_outputs/3_pred_xy_{self.debug_cnt}.npy'
                        # import numpy as np
                        # np.save(save_path, pred_waypoints_xy[0].detach().cpu().numpy())
                        # self.debug_cnt += 1
                        # # end of debug

                        use_free_space_loss = False
                        snap_result_using_oracle_bev_label = True
                        save_debug_img = False and snap_result_using_oracle_bev_label
                        if use_free_space_loss:
                            occ_field = torch.sqrt(data["occ_field"].to(pred_waypoints_xy.device))  # [B, H, W], in agent local frame
                            B, num_wp, _ = pred_waypoints_xy.shape
                            pred_wp_xy_reshaped = pred_waypoints_xy.view(B * num_wp, 2)
                            loss_free_space = torch.nn.functional.grid_sample(
                                occ_field.unsqueeze(1).float().repeat_interleave(num_wp, dim=0),
                                torch.stack([(((pred_wp_xy_reshaped[:,1]-(-6.4))/0.2)/(occ_field.shape[2]-1)*2-1),
                                            (((pred_wp_xy_reshaped[:,0]-(-6.4))/0.2)/(occ_field.shape[1]-1)*2-1)], dim=-1).view(-1,1,1,2),
                                mode="bilinear", padding_mode="border", align_corners=True
                            ).view(-1) # [B * num_wp]
                            loss_free_space = torch.clamp(loss_free_space, max=5.0).mean()  # cap at 5.0 to avoid extreme outliers
                            loss = loss_l1 + 0.5 * loss_free_space
                            print(f"[Eval] Predicted waypoint L1 loss: {loss_l1.item()}")
                            print(f"[Eval] Predicted waypoint free space loss: {loss_free_space.item()}")
                        else:
                            loss_free_space = torch.tensor(0.0, device=loss_l1.device)
                            loss = loss_l1
                            print(f"[Eval] Predicted waypoint L1 loss: {loss_l1.item()}")

                        if save_debug_img:
                            pred_waypoints_xy_raw=pred_waypoints_xy[0].detach().clone()
                            gt_waypoints_xy=target_waypoint_xy[0].detach().clone()
                            occ_label_dbg=data['occ_label'][0].detach().clone()
                            imgs_dbg=data['raw_img'][0].permute(0,3,1,2).detach().cpu().numpy() # (4,3,H,W) uint8 BGR
                            imgs_dbg = imgs_dbg[::-1] # from front left back right to right back left front
                            import numpy as np
                            imgs_dbg = np.array([imgs_dbg[2], imgs_dbg[3], imgs_dbg[0], imgs_dbg[1]]) # to front left back right


                        if snap_result_using_oracle_bev_label:
                            # occ_label=data['occ_label'][0].to(pred_waypoints_xy.device);free=occ_label.bool() # [H,W], True=free
                            # wp=pred_waypoints_xy[0] # [6,2], x=fwd,y=left in meters
                            # hf=(wp[:,0]+6.4)/0.2;wf=(wp[:,1]+6.4)/0.2 # float grid coords (keep sub-cell info)
                            # hi=torch.clamp(torch.floor(hf).long(),0,63);wi=torch.clamp(torch.floor(wf).long(),0,63) # base cell
                            # for i in range(wp.shape[0]):
                            #     if free[hi[i],wi[i]]:continue # already free
                            #     h0=hf[i];w0=wf[i]
                            #     for r in range(1,64):
                            #         hmin=max(0,hi[i].item()-r);hmax=min(63,hi[i].item()+r);wmin=max(0,wi[i].item()-r);wmax=min(63,wi[i].item()+r)
                            #         sub=free[hmin:hmax+1,wmin:wmax+1]
                            #         if not sub.any():continue
                            #         ys,xs=torch.nonzero(sub,as_tuple=True);hs=ys.to(h0.dtype)+hmin+0.5;ws=xs.to(w0.dtype)+wmin+0.5 # candidate cell centers in grid coords
                            #         j=torch.argmin((hs-h0)**2+(ws-w0)**2);wp[i,0]=hs[j]*0.2-6.4;wp[i,1]=ws[j]*0.2-6.4 # snap to center in meters
                            #         break
                            # pred_xy=wp[-1] # update endpoint after snapping
                            trav=data['traversable_mask'][0].to(pred_waypoints_xy.device).bool();
                            vis=data['visible_mask'][0].to(pred_waypoints_xy.device).bool();
                            # valid=trav&vis # [H,W], True=valid
                            # valid=vis # debug: only use visible area
                            from xtuner.perception_modules.affordance_head import AffordanceHead
                            depth = data["depth"].to(input_ids.device) # [B, 4, H, W]
                            cam2imgs = data["cam2imgs"]
                            cam2egos = data["cam2egos"]
                            if isinstance(cam2imgs, dict):
                                order = ["front", "left", "back", "right"]
                                cam2imgs = torch.stack([cam2imgs[k] for k in order], dim=1)
                                cam2egos = torch.stack([cam2egos[k] for k in order], dim=1)
                            cam2imgs = cam2imgs.to(input_ids.device)
                            cam2egos = cam2egos.to(input_ids.device)
                            intrinsics = cam2imgs[:, :, :3, :3]
                            free_ground_obs, free_cam_obs = AffordanceHead.prepare_free_obs(
                                depth,            # (B, N, H, W) meters
                                cam2egos,      # (B, N, 4, 4)
                                intrinsics,       # (B, N, 3, 3)
                                grid_config=dict(
                                    x=[-6.4, 6.4, 0.1],       # 128
                                    y=[-6.4, 6.4, 0.1],       # 128
                                ),
                                Dx=128, Dy=128,           # BEV grid size
                                ground_z_range=(-1.6, -1.4),
                                cam_z_range=(-0.1, 0.1),
                                blocker_z_range=(-1.4, -0.1),
                                old_behavior=False,
                                # blocker_z_range=(-1.6, 0.2),
                                # old_behavior=True,
                            )
                            valid=free_ground_obs[0, 0].bool() # [H,W], True=valid

                            # # debug
                            # dbg_vis_path = 'debug_outputs/0_depth_free_debug.npz'
                            # import numpy as np
                            # np.savez(
                            #     dbg_vis_path,
                            #     trav=trav.cpu().numpy(),
                            #     vis=vis.cpu().numpy(),
                            #     free_ground_obs=free_ground_obs[0, 0].cpu().numpy(),
                            #     free_cam_obs=free_cam_obs[0, 0].cpu().numpy(),
                            # )
                            # exit(0)
                            # # end of debug

                            wp=pred_waypoints_xy[0] # [6,2], x=fwd,y=left in meters
                            hf=(wp[:,0]+6.4)/0.1;wf=(wp[:,1]+6.4)/0.1 # float grid coords (keep sub-cell info)
                            hi=torch.clamp(torch.floor(hf).long(),0,127);wi=torch.clamp(torch.floor(wf).long(),0,127) # base cell
                            for i in range(wp.shape[0]):
                                if valid[hi[i],wi[i]]:continue # already valid
                                h0=hf[i];w0=wf[i]
                                for r in range(1,128):
                                    hmin=max(0,hi[i].item()-r);hmax=min(127,hi[i].item()+r);wmin=max(0,wi[i].item()-r);wmax=min(127,wi[i].item()+r)
                                    sub=valid[hmin:hmax+1,wmin:wmax+1]
                                    if not sub.any():continue
                                    ys,xs=torch.nonzero(sub,as_tuple=True);hs=ys.to(h0.dtype)+hmin+0.5;ws=xs.to(w0.dtype)+wmin+0.5 # candidate cell centers in grid coords
                                    j=torch.argmin((hs-h0)**2+(ws-w0)**2);wp[i,0]=hs[j]*0.1-6.4;wp[i,1]=ws[j]*0.1-6.4 # snap to center in meters
                                    break
                            pred_xy=wp[-1] # update endpoint after snapping

                        if save_debug_img:
                            import os,matplotlib.pyplot as plt
                            instr=data['instruction'][0];token=data['token'][0];out=f"debug_outputs/waypoint_{token}.jpg";os.makedirs(os.path.dirname(out),exist_ok=True)
                            raw=pred_waypoints_xy_raw.to(pred_waypoints_xy.device);gt=gt_waypoints_xy.to(pred_waypoints_xy.device);sn=pred_waypoints_xy[0];occ=occ_label_dbg
                            gth=(gt[:,0]+6.4)/0.2;gtw=(gt[:,1]+6.4)/0.2;rh=(raw[:,0]+6.4)/0.2;rw=(raw[:,1]+6.4)/0.2;sh=(sn[:,0]+6.4)/0.2;sw=(sn[:,1]+6.4)/0.2
                            fig=plt.figure(figsize=(10,9));gs=fig.add_gridspec(2,4,height_ratios=[0.75,3.25],wspace=0.0,hspace=0.02);fig.suptitle(instr,fontsize=8,y=0.98)
                            for k in range(4):
                                ax=fig.add_subplot(gs[0,k]);im=imgs_dbg[k].transpose(1,2,0)[:,:,::-1];ax.imshow(im,aspect='auto');ax.set_aspect('auto');ax.axis('off')
                            ax=fig.add_subplot(gs[1,:]);ax.imshow(occ.float().cpu(),interpolation='nearest');ax.plot([31.5],[31.5],'r.',markersize=10,label='ego')
                            ax.plot(gtw.cpu(),gth.cpu(),'-',label='gt path');ax.plot([gtw[-1].cpu()],[gth[-1].cpu()],'*',markersize=10,label='gt target')
                            ax.plot(rw.cpu(),rh.cpu(),'--',label='pred path');ax.plot([rw[-1].cpu()],[rh[-1].cpu()],'x',markersize=8,label='pred target')
                            ax.plot(sw.cpu(),sh.cpu(),'-',label='snapped pred path');ax.plot([sw[-1].cpu()],[sh[-1].cpu()],'D',markersize=6,label='snapped pred target')
                            ax.set_xlabel("Dy");ax.set_ylabel("Dx");ax.legend(fontsize=7,loc='upper right')
                            plt.subplots_adjust(left=0.005,right=0.995,top=0.95,bottom=0.04,wspace=0.0,hspace=0.02);plt.savefig(out,dpi=150);plt.close();print(f"[Eval] saved to {out}")

                        pred_aff = torch.zeros_like(data['affordance_mask'][0])
                        forward_prediction_reults.append({"p_pred": pred_xy, "loss_l1": loss_l1, "loss": loss, 'path_pred': pred_waypoints_xy[0], \
                                                          "loss_free_space": loss_free_space, 'pred_aff': pred_aff, "pred_xy": pred_xy, \
                                                             'free_ground_obs': free_ground_obs[0, 0].bool(), 'free_cam_obs': free_cam_obs[0, 0].bool(), })
                    else:
                        raise NotImplementedError(f"Unknown predict_mode: {predict_mode}")

                # forward_prediction_reults.append({"loss_ce": loss_ce, "loss": loss_ce, "p_pred": torch.zeros(2, device=input_ids.device)})
                return forward_prediction_reults
        elif self.output_mode == "RoboPoint":
            if self.training:
                out = self._llm_forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    image_flags=image_flags,
                    pixel_values=concat_images,
                    labels=labels,
                    use_cache=use_cache,
                    output_hidden_states=True,   # we need hidden states for tgt token
                    return_dict=True, # we need dict output
                )
                loss_ce = out.loss # effectively averaged over #per_batch_chat_samples*#average_answer
                return {"loss": loss_ce, "loss_ce": loss_ce}
            else:
                assert len(input_ids) == 1, "Eval batch size should be 1."
                amp_ctx = autocast(dtype=torch.bfloat16)
                forward_prediction_reults = [] # mmengine requires this to be a list
                # truncate prompt using labels
                prompt_input_ids, prompt_attention_mask, prompt_position_ids, prompt_len = (
                    self._truncate_to_prompt(input_ids, attention_mask, position_ids, labels)
                )
                with amp_ctx:
                    gen_ids = self._llm_forward_autoregressive(
                        input_ids=prompt_input_ids,
                        attention_mask=prompt_attention_mask,
                        image_flags=image_flags,
                        pixel_values=concat_images,
                    )
                gen_new_ids = gen_ids[:, prompt_len:]  # 1 x T'
                text = self.tokenizer.decode(gen_new_ids[0], skip_special_tokens=False)
                loss_ce = torch.tensor(0.0, device=input_ids.device) # no CE loss in AR eval; keep as 0 for ExternalMetric logging
                # Keep only valid vocab IDs
                label_ids = labels[0]
                valid = (label_ids >= 0) & (label_ids < self.tokenizer.vocab_size)
                text_label = self.tokenizer.decode(label_ids[valid], skip_special_tokens=False)

                move_dir, move_dir_label, range_tag, range_tag_label, dstep, succ1_dir, succ1_dir_and_size, succ0_dir, succ0_dir_and_size = self.parse_and_predict_meta_action(
                    text, text_label, data)

                pat = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")
                m = pat.search(text)
                if m:
                    xy = torch.tensor([float(m.group(1)), float(m.group(2))],
                                    device=input_ids.device, dtype=torch.float32)
                else:
                    xy = torch.zeros(2, device=input_ids.device, dtype=torch.float32)

                pred_xy = xy
                snap_result_using_oracle_bev_label = True
                if snap_result_using_oracle_bev_label:
                    trav=data['traversable_mask'][0].to(pred_xy.device).bool();vis=data['visible_mask'][0].to(pred_xy.device).bool();
                    valid=vis # only use visible area
                    # valid = vis & trav  # only use traversable area
                    p=pred_xy # [2], x=fwd,y=left in meters
                    hf=(p[0]+6.4)/0.1;wf=(p[1]+6.4)/0.1 # float grid coords
                    hi=int(torch.clamp(torch.floor(hf).long(),0,127).item());wi=int(torch.clamp(torch.floor(wf).long(),0,127).item())
                    if not valid[hi,wi]:
                        h0=hf;w0=wf
                        for r in range(1,128):
                            hmin=max(0,hi-r);hmax=min(127,hi+r);wmin=max(0,wi-r);wmax=min(127,wi+r)
                            sub=valid[hmin:hmax+1,wmin:wmax+1]
                            if not sub.any():continue
                            ys,xs=torch.nonzero(sub,as_tuple=True);hs=ys.to(h0.dtype)+hmin+0.5;ws=xs.to(w0.dtype)+wmin+0.5
                            j=torch.argmin((hs-h0)**2+(ws-w0)**2);pred_xy=torch.stack([hs[j]*0.1-6.4,ws[j]*0.1-6.4]).to(p.dtype)
                            break
            
                loss_reg = torch.nn.functional.mse_loss(xy, data["p_goal"][0, :2].to(xy.device)) # just for log
                pred_aff = torch.zeros_like(data['affordance_mask'][0])
                forward_prediction_reults.append({"p_pred": xy, 'loss_reg': loss_reg, 'loss': loss_ce, 'loss_ce': loss_ce, 'text_pred': text, \
                                                  "dstep": dstep, "succ1_dir": succ1_dir, "succ1_dir_and_size": succ1_dir_and_size, \
                                                    'pred_xy': pred_xy, 'pred_aff': pred_aff})
                return forward_prediction_reults
        elif self.output_mode == "LSSAttn":  
            predict_mode = "region"  # "xy" or "region" or "node"
            # TODO: check whether have vqa trained together will help the target regression
            if self.training:
                assert torch.get_autocast_gpu_dtype() == torch.bfloat16, "training LSSAttn mode requires bfloat16 autocast in optim_wrapper, grad will overflow fp16 inside llm backward."
                out = self._llm_forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    image_flags=image_flags,
                    pixel_values=concat_images,
                    labels=None,
                    use_cache=use_cache,
                    output_hidden_states=True,   # we need hidden states for tgt token
                    return_dict=True, # we need dict output
                )
                hidden = out.hidden_states[-1] # B x T x H
                tgt_mask = (input_ids == self.tgt_token_idx) # B x T
                # img_mask = (input_ids[0] == self.model.img_context_token_id)
                img_mask = (input_ids == self.model.img_context_token_id) # B x T
                assert img_mask.sum(-1).eq(16*16*4).all(), "Expected 16x16x4 image context tokens."

                assert tgt_mask.sum(-1).eq(1).all(), "Expected exactly one [TGT] per sample"
                # assert img_mask.sum() == 16*16*4, "Expected 16x16x4 image context tokens."
                tgt_state = torch.masked_select(hidden, tgt_mask.unsqueeze(-1)).view(hidden.size(0), -1)  # B,T,H and B,T,1 -> B*H -> B,H
                # img_states = hidden[:, img_mask]  # B x (16*16*4) x H
                B, N, fH, fW, H = hidden.shape[0], 4, 16, 16, hidden.shape[2]
                img_states = hidden[img_mask].contiguous().view(B, 4*16*16, H) # B x (16*16*4) x H
                img_states = img_states.view(B, N, fH, fW, H)  # B x 4 x fH=16 x fW=16 x H # N=view first, row major
                region_labels = data["region_label"].to(input_ids.device).long() # region labels (explicit field from dataset)

                # depth + camera matrices
                depth = data["depth"].to(input_ids.device)
                cam2imgs = data["cam2imgs"]
                cam2egos = data["cam2egos"]
                if isinstance(cam2imgs, dict):
                    order = ["front", "left", "back", "right"]
                    cam2imgs = torch.stack([cam2imgs[k] for k in order], dim=1)
                    cam2egos = torch.stack([cam2egos[k] for k in order], dim=1)
                cam2imgs = cam2imgs.to(input_ids.device)
                cam2egos = cam2egos.to(input_ids.device)
                intrinsics = cam2imgs[:, :, :3, :3]
                if predict_mode == "region":
                    # print(f"[Forward] img_states shape: {img_states.shape}, tgt_state shape: {tgt_state.shape}, depth shape: {depth.shape}, cam2egos shape: {cam2egos.shape}, intrinsics shape: {intrinsics.shape}")
                    # - Method 1
                    # region_preds = self.output_head(
                    #     depth,
                    #     img_states,
                    #     cam2egos,
                    #     intrinsics,
                    #     tgt_state,
                    # ) # B x num_regions
                    # # print(f"finite region preds? {torch.isfinite(region_preds).all()}")
                    # # print(f"finite region labels? {torch.isfinite(region_labels).all()}")
                    # if self.class_weights is not None:
                    #     class_weights_tensor = torch.tensor(self.class_weights).to(region_labels.device)
                    #     loss_cls = torch.nn.functional.cross_entropy(region_preds, region_labels, weight=class_weights_tensor)
                    # else:
                    #     loss_cls = torch.nn.functional.cross_entropy(region_preds, region_labels)
                    # return {"loss": loss_cls, "loss_cls": loss_cls}

                    # - Method 2: predict separately
                    dir_preds, range_preds = self.output_head(
                        depth,
                        img_states,
                        cam2egos,
                        intrinsics,
                        tgt_state,
                    ) # B x num_directions, B x num_ranges
                    direction_labels = (region_labels // 2).long()  # 0-7
                    range_labels = (region_labels % 2).long()       # 0-1
                    loss_cls = self.output_head.region_loss(dir_preds, direction_labels, range_preds, range_labels)
                    return {"loss": loss_cls, "loss_cls": loss_cls}

                    # # - Method 3: use the point to get fine direction classification
                    # direction_labels = self.output_head.get_fine_dir_id_from_targets(
                    #     data["p_goal"][:, :2].to(input_ids.device) # B x 2
                    # )  # 0-47
                    # dir_preds, _ = self.output_head(
                    #     depth,
                    #     img_states,
                    #     cam2egos,
                    #     intrinsics,
                    #     tgt_state,
                    # ) # B x num_fine_directions
                    # loss_cls = self.output_head.direction_loss(dir_preds, direction_labels)
                    # return {"loss": loss_cls, "loss_cls": loss_cls}
                elif predict_mode == "xy":
                    # # - Method 1: no cotrain
                    # pred_xy = self.output_head(
                    #     depth,
                    #     img_states,
                    #     cam2egos,
                    #     intrinsics,
                    #     tgt_state,
                    #     pred_regions=False,
                    # ) # B x 2
                    # target_xy = data["p_goal"][:, :2].to(pred_xy.device)  # B x 2
                    # loss_reg = self.output_head.target_loss(pred_xy, target_xy)
                    # return {"loss": loss_reg, "loss_reg": loss_reg}

                    # # - Method 2: cotrain with dir classification
                    # pred_xy, dir_preds = self.output_head(
                    #     depth,
                    #     img_states,
                    #     cam2egos,
                    #     intrinsics,
                    #     tgt_state,
                    #     pred_regions=False,
                    # ) # B x 2, B x num_directions
                    # target_xy = data["p_goal"][:, :2].to(pred_xy.device)  # B x 2
                    # direction_labels = data["region_label"].to(input_ids.device).long() // 2  # 0-7
                    # loss_reg = self.output_head.target_loss(pred_xy, target_xy)
                    # loss_cls = self.output_head.direction_loss(dir_preds, direction_labels)
                    # loss = loss_reg + 10*loss_cls
                    # return {"loss": loss, "loss_reg": loss_reg, "loss_cls": loss_cls}

                    # # - Method 3: cotrain, and oracle selection in training + winner takes all in eval
                    # cand_xy, dir_preds = self.output_head(
                    #     depth,
                    #     img_states,
                    #     cam2egos,
                    #     intrinsics,
                    #     tgt_state,
                    #     pred_regions=False,
                    # ) # B, 8, 2; B, 8
                    # direction_labels = data["region_label"].to(input_ids.device).long() // 2  # 0-7 # shape: [B]
                    # loss_cls = self.output_head.direction_loss(dir_preds, direction_labels)
                    # target_xy = data["p_goal"][:, :2].to(cand_xy.device)  # B x 2
                    # # oracle selection
                    # pred_xy = cand_xy.gather(dim=1, index=direction_labels.view(B, 1, 1).expand(-1, 1, 2)).squeeze(1)  # B x 2
                    # loss_reg = self.output_head.target_loss(pred_xy, target_xy)
                    # loss = loss_reg + 10*loss_cls
                    # return {"loss": loss, "loss_reg": loss_reg, "loss_cls": loss_cls}

                    # - Method 4: bev heatmap with dense feature
                    heatmap_preds = self.output_head(
                        depth,
                        img_states,
                        cam2egos,
                        intrinsics,
                        tgt_state,
                        pred_regions=False,
                    ) # B, Dx, Dy
                    # target_xy = data["p_goal"][:, :2].to(heatmap_preds.device)  # B x 2
                    # loss_focal = self.output_head.heatmap_loss(heatmap_preds, target_xy)
                    target_heatmap = data['target_heatmap'].to(heatmap_preds.device)  # B x Dx x Dy
                    return {"loss": loss_focal, "loss_focal": loss_focal}
                elif predict_mode == "node":
                    # prepare the boxes as nodes
                    p_goal_xy = data["p_goal"][:, :2].to(input_ids.device)  # B x 2
                    with autocast(enabled=False):
                        detected_boxes, selected_boxes = self.box_pipeline.run_batched(
                            data['raw_img'].float(), data['depth'].float(), data['conversations'], p_goal=p_goal_xy.float()
                        )
                        # self.output_head.forward_node(
                        #     depth,
                        #     img_states,
                        #     cam2egos,
                        #     intrinsics,
                        #     tgt_state,
                        #     detected_boxes,
                        #     selected_boxes,
                        #     p_goal_xy,
                        # )
                        # node_scores, node_xyz, node_mask = self.output_head.forward_node_graph(
                        #     img_feats=img_states,
                        #     query_embeddings=tgt_state,
                        #     detected_boxes=detected_boxes,
                        # )
                        # selected_boxes now contains idx/dist_xy/is_ego, so labeling is easy
                        # targets, selected_mask = self.output_head.label_nodes(
                        #     selected_boxes=selected_boxes,
                        #     node_mask=node_mask,
                        # )
                        # print(f"[DEBUG] targets: {targets}")
                        # loss_node = self.output_head.loss_soft_ce(
                        #     node_scores=node_scores,
                        #     targets=targets,
                        #     node_mask=node_mask,
                        # )
                        pred_xy = self.output_head.forward_node_graph_v2(
                            img_feats=img_states,
                            query_embeddings=tgt_state,
                            detected_boxes=detected_boxes,
                        )
                        loss_reg = self.output_head.target_loss(
                            p_pred=pred_xy,
                            p_target=p_goal_xy,
                        )
                    return {"loss": loss_reg, "loss_reg": loss_reg}

                else:
                    raise NotImplementedError(f"Unknown predict_mode: {predict_mode}")
            
            else:
                assert len(input_ids) == 1, "Eval batch size should be 1."
                # ensure bfloat16, cuz the training forward can be wrapper by the config optim wrapper but test forward not
                amp_ctx = autocast(dtype=torch.bfloat16) 
                forward_prediction_reults = [] # mmengine requires this to be a list
                with amp_ctx:
                    out = self._llm_forward(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        image_flags=image_flags,
                        pixel_values=concat_images,
                        labels=None,
                        use_cache=use_cache,
                        output_hidden_states=True,   # we need hidden states for tgt token
                        return_dict=True, # we need dict output
                    )
                hidden = out.hidden_states[-1] # B x T x H
                tgt_mask = (input_ids[0] == self.tgt_token_idx)  # all samples in batch these position shoule be the same
                img_mask = (input_ids[0] == self.model.img_context_token_id)
                assert tgt_mask.sum() == 1, \
                    ("Expected exactly one [TGT] token per sample in SingleTokenEmbedding variant.")
                assert img_mask.sum() == 16*16*4, "Expected 16x16x4 image context tokens."
                tgt_state = hidden[:, tgt_mask].squeeze(1)  # B x H
                img_states = hidden[:, img_mask]  # B x (16*16*4) x H
                region_labels = data["region_label"].to(input_ids.device).long() # region labels (explicit field from dataset)

                # depth + camera matrices
                depth = data["depth"].to(input_ids.device)
                cam2imgs = data["cam2imgs"]
                cam2egos = data["cam2egos"]
                if isinstance(cam2imgs, dict):
                    order = ["front", "left", "back", "right"]
                    cam2imgs = torch.stack([cam2imgs[k] for k in order], dim=1)
                    cam2egos = torch.stack([cam2egos[k] for k in order], dim=1)
                cam2imgs = cam2imgs.to(input_ids.device)
                cam2egos = cam2egos.to(input_ids.device)
                intrinsics = cam2imgs[:, :, :3, :3]

                # reshape image hidden states to per-view grids
                B = input_ids.size(0)
                img_states = img_states.view(B, 4, 16, 16, -1)
                if predict_mode == "region":
                    # # - Method 1
                    # region_preds = self.output_head(
                    #     depth,
                    #     img_states,
                    #     cam2egos,
                    #     intrinsics,
                    #     tgt_state,
                    # ) # B x num_regions
                    # pred_region = region_preds.argmax(dim=-1)  # B

                    # - Method 2: predict separately
                    dir_preds, range_preds = self.output_head(
                        depth,
                        img_states,
                        cam2egos,
                        intrinsics,
                        tgt_state,
                    ) # B x num_directions, B x num_ranges
                    pred_dir = dir_preds.argmax(dim=-1)  # B
                    pred_range = range_preds.argmax(dim=-1)  # B
                    pred_region = pred_dir * 2 + pred_range  # B
                    print(f"[Eval] Raw Region Preds: {pred_region}")
                    text, text_label = self.parse_pred_to_meta_action(
                        pred_region[0], region_labels[0])
                    print(f"[Eval] Predicted Meta Action: {text}; GT Meta Action: {text_label}")
                    move_dir, move_dir_label, range_tag, range_tag_label, dstep, succ1_dir, succ1_dir_and_size, succ0_dir, succ0_dir_and_size = self.parse_and_predict_meta_action(
                        text, text_label, data)
                    # # debug
                    # if move_dir != move_dir_label:
                    #     with torch.no_grad():
                    #         debug_file = 'debug_outputs/similarity_dir_debug.npz'
                    #         import numpy as np
                    #         np.savez_compressed(
                    #             debug_file,
                    #             per_dir_scores=dir_preds.cpu().numpy(),
                    #         )
                    #         print(f"Saved debug bev_grid_ids to {debug_file}")
                    #         exit(0)
                    # # end of debug
                    try:
                        best_xy = self.oracle_bev_predict_target(
                            move_dir=move_dir,
                            range_tag=range_tag,
                            data=data,
                        ) # BUG: sometimes this will crash cuz library version or bugs
                    except Exception as e:
                        import numpy as np
                        print(f"[Eval][Oracle BEV] Skipping sample due to voxelizer error: {e}")
                        best_xy = np.array([0.0, 0.0], dtype=np.float32)  # or None and return dummy
                    forward_prediction_reults.append(
                        {"p_pred": torch.from_numpy(best_xy), "text_pred": text, \
                            "dstep": dstep, "succ1_dir": succ1_dir, "succ1_dir_and_size": succ1_dir_and_size, \
                                "succ0_dir": succ0_dir, "succ0_dir_and_size": succ0_dir_and_size
                        }
                    )

                    # # - Method 3: use the point to get fine direction classification
                    # dir_preds, _ = self.output_head(
                    #     depth,
                    #     img_states,
                    #     cam2egos,
                    #     intrinsics,
                    #     tgt_state,
                    # ) # B x num_fine_directions
                    # pred_dir = dir_preds.argmax(dim=-1)  # B
                    # direction_labels = self.output_head.get_fine_dir_id_from_targets(
                    #     data["p_goal"][:, :2].to(input_ids.device)
                    # )  # 0-47
                    # dstep, succ0, succ1, succ2 = self._success_at_0_1_2_bins(
                    #     pred_dir[0], direction_labels[0]
                    # )
                    # pred_xy = self.oracle_bev_predict_target_from_dir48(
                    #     pred_dir[0], data
                    # )  # use oracle distance to get predicted point
                    # forward_prediction_reults.append(
                    #     {"p_pred": torch.from_numpy(pred_xy), "succ1_dir": succ1, \
                    #             "succ0_dir": succ0,  "succ2_dir": succ2,
                    #     }
                    # )
                    return forward_prediction_reults
                elif predict_mode == "xy":
                    # # - Method 1: no cotrain
                    # pred_xy = self.output_head(
                    #     depth,
                    #     img_states,
                    #     cam2egos,
                    #     intrinsics,
                    #     tgt_state,
                    #     pred_regions=False,
                    # ) # B x 2

                    # # - Method 2: cotrain with dir classification
                    # pred_xy, dir_preds = self.output_head(
                    #     depth,
                    #     img_states,
                    #     cam2egos,
                    #     intrinsics,
                    #     tgt_state,
                    #     pred_regions=False,
                    # ) # B x 2, B x num_directions

                    # # - Method 3: cotrain, and winner takes all
                    # cand_xy, dir_preds = self.output_head(
                    #     depth,
                    #     img_states,
                    #     cam2egos,
                    #     intrinsics,
                    #     tgt_state,
                    #     pred_regions=False,
                    # ) # B, 8, 2; B, 8
                    # # winner takes all
                    # pred_xy = cand_xy.gather(dim=1, index=dir_preds.argmax(dim=-1).view(B, 1, 1).expand(-1, 1, 2)).squeeze(1)  # B x 2

                    # - Method 4: bev heatmap with argmax
                    heatmap_preds = self.output_head(
                        depth,
                        img_states,
                        cam2egos,
                        intrinsics,
                        tgt_state,
                        pred_regions=False,
                    ) # B, Dx, Dy
                    pred_xy, dir_preds = self.output_head.convert_heatmap_to_coords_and_dirs(heatmap_preds)  # B x 2, B x num_directions

                    target_xy = data["p_goal"][:, :2].to(pred_xy.device)  # B x 2
                    loss_reg = self.output_head.target_loss(pred_xy, target_xy)
                    loss_cls = self.output_head.direction_loss(
                        dir_preds, data["region_label"].to(input_ids.device).long() // 2)
                    loss = loss_reg + loss_cls
                    pred_dir = dir_preds.argmax(dim=-1)  # B
                    pred_range = torch.zeros_like(pred_dir)  # dummy range prediction
                    pred_region = pred_dir * 2 + pred_range  # B
                    print(f"[Eval] Raw Region Preds: {pred_region}")
                    text, text_label = self.parse_pred_to_meta_action(
                        pred_region[0], region_labels[0])
                    print(f"[Eval] Predicted Meta Action: {text}; GT Meta Action: {text_label}")
                    move_dir, move_dir_label, range_tag, range_tag_label, dstep, succ1_dir, succ1_dir_and_size, succ0_dir, succ0_dir_and_size = self.parse_and_predict_meta_action(
                        text, text_label, data)
                    forward_prediction_reults.append({"p_pred": pred_xy[0], 'loss_reg': loss_reg, 'loss_cls': loss_cls, 'loss': loss, \
                                "succ1_dir": succ1_dir, "succ1_dir_and_size": succ1_dir_and_size, \
                                "succ0_dir": succ0_dir, "succ0_dir_and_size": succ0_dir_and_size})
                    return forward_prediction_reults
                elif predict_mode == "node":
                    # prepare the boxes as nodes
                    p_goal_xy = data["p_goal"][:, :2].to(input_ids.device)  # B x 2
                    with autocast(enabled=False):
                        detected_boxes, selected_boxes = self.box_pipeline.run_batched(
                            data['raw_img'].float(), data['depth'].float(), data['conversations'], p_goal=p_goal_xy.float()
                        )
                        node_scores, node_xyz, node_mask = self.output_head.forward_node_graph(
                            img_feats=img_states,
                            query_embeddings=tgt_state,
                            detected_boxes=detected_boxes,
                        )
                        # selected_boxes now contains idx/dist_xy/is_ego, so labeling is easy
                        targets, selected_mask = self.output_head.label_nodes(
                            selected_boxes=selected_boxes,
                            node_mask=node_mask,
                        )
                        loss_node = self.output_head.loss_soft_ce(
                            node_scores=node_scores,
                            targets=targets,
                            node_mask=node_mask,
                        )
                        sel_idx = torch.nonzero(selected_mask[0], as_tuple=False).squeeze(-1)  # (N_sel,)
                        sel_xy = node_xyz[0, sel_idx, :2]          # (N_sel, 2)
                        sel_tgt = targets[0, sel_idx]              # (N_sel,)
                        order = torch.argsort(sel_tgt, descending=True)
                        selected_node_xy = sel_xy[order].detach().cpu().numpy().tolist() # N_sel, 2
                        valid_scores = node_scores[0][node_mask[0]]  # N_valid
                        sorted_node_idx = torch.argsort(valid_scores, descending=True)
                        sorted_node_xy = node_xyz[0][node_mask[0]][sorted_node_idx, :2].cpu().numpy().tolist()  # list of [x,y]
                        from time import time
                        current_t = time()
                        s = str(data['conversations'][0])
                        m = re.search(r"Instruction:\s*(.*?)(?:\n\s*\[TGT\]|\Z)", s, flags=re.S)
                        if m:
                            instruction = m.group(1).strip()
                        pred_xy = self.oracle_bev_generator.oracle_free_point_from_nodes(
                            yaw_deg=math.degrees(data['agent_yaw_hab_rad'][0].item()),
                            target_token=data['viewpoint_hash'][0],
                            node_xy_list=sorted_node_xy,
                            debug=True,
                            debug_path=f'debug_outputs/node_eval_result_{current_t}.jpg',
                            selected_node_xy=selected_node_xy,
                            goal_xy=data['p_goal'][0, :2].cpu().numpy(),
                            instruction=instruction,
                        )
                        print(f"[Eval] Predicted Point from Nodes: {pred_xy}")
                        print(f"[Eval] Ground Truth Point: {data['p_goal'][0, :2].cpu().numpy()}")
                        print(f"[Eval] Loss Node: {loss_node.item()}")
                    return [{"loss": loss_node, "loss_node": loss_node, 'p_pred': torch.from_numpy(pred_xy)}]
                        
                        # pred_xy = self.output_head.forward_node_graph_v2(
                        #         img_feats=img_states,
                        #         query_embeddings=tgt_state,
                        #         detected_boxes=detected_boxes,
                        #     )
                        # loss_reg = self.output_head.target_loss(
                        #     p_pred=pred_xy,
                        #     p_target=p_goal_xy,
                        # )
                    return [{"loss": loss_reg, "loss_reg": loss_reg, 'p_pred': pred_xy[0]}]

                else:
                    raise NotImplementedError(f"Unknown predict_mode: {predict_mode}")
        
        else:
            raise NotImplementedError(f"Unknown output_mode: {self.output_mode}")
