# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import AutoTokenizer

from xtuner.dataset import InternVL_V1_5_Dataset, InternVL_V1_5_Dataset_Multiview, InternVL_V1_5_Dataset_Multiview_PKL
from xtuner.dataset.collate_fns import default_collate_fn, custom_collate_fn
from xtuner.dataset.samplers import LengthGroupedSampler, BalancedLengthGroupedSamplerSingleGPU
from xtuner.engine.hooks import DatasetInfoHook, DatasetInfoHookSpecialTokens
from xtuner.engine.runner import TrainLoop
from xtuner.model import InternVL_V1_5, InternVL_V1_5_NavTarget
from xtuner.perception_modules import AffordanceHead
from xtuner.utils import PROMPT_TEMPLATE

from mmengine.dataset import DefaultSampler
from xtuner.evaluation.metrics.external_metric import ExternalMetric

import os
# Paths below default to the original HPC container layout (data bind-mounted at /data,
# model weights at /model). Override with env vars for a local machine, e.g.
#   export BEACON_DATA_ROOT=/your/data ; export BEACON_MODEL_ROOT=/your/models
BEACON_DATA_ROOT = os.environ.get("BEACON_DATA_ROOT", "/data")
BEACON_MODEL_ROOT = os.environ.get("BEACON_MODEL_ROOT", "/model")
BEACON_WORK_ROOT = os.environ.get("BEACON_WORK_ROOT", "/workspace")  # code/work_dir bind-mount; Stage-2 load_from base

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
# path = 'OpenGVLab/InternVL2-2B'
path = f'{BEACON_MODEL_ROOT}/OpenGVLab/InternVL2-2B'

output_mode = "Affordance"  # "RoboPoint" or "Affordance" or "VQA"

custom_imports = dict(
    imports=['xtuner.perception_modules'],
    allow_failed_imports=False,
)

import sys as _sys
_IS_TEST = ('--checkpoint' in _sys.argv)  # this is the reliable signal
# _IS_TEST = True # debug zero shot
del _sys  # IMPORTANT: remove module object from config globals

# Data
# data_path = data_root + 'LLaVA-Instruct-150K/llava_v1_5_mix665k.json'
# data_path = data_root + 'mp3d_nav_ep0028.jsonl'
# image_folder = data_root + 'llava_images'
# image_folder = data_root + 'debug_images_S9hNv5qa7GM_0028'
training_scan_id = 'S9hNv5qa7GM' # '2azQ1b91cZZ' # 'S9hNv5qa7GM'
val_scan_id = '2azQ1b91cZZ'
if _IS_TEST:
    pkl_path = f'{BEACON_DATA_ROOT}/val_unseen.pkl'
else:
    pkl_path = f'{BEACON_DATA_ROOT}/train.pkl'

pkl_dataset_root = f"{BEACON_DATA_ROOT}/captures"
pkl_loader_kwargs = {
    'fov_deg': 90.0,
    'out_hw': (448, 448),
    # prefer this switch over `use_dynamic`
    'use_version': 'dynamic',  # 'dynamic' | 'static' | 'both'
    'use_dynamic': True, # only used if use_version is not provided, as backward compatibility
    'verbose': False,
    'max_distance_horizontal': 6.4,
    'max_distance_vertical': 0.5,
}

prompt_template = PROMPT_TEMPLATE.internlm2_chat
# max_length = 8192
max_length = 1536 # usually max length is just ~1200 in my data

# Scheduler & Optimizer
# batch_size = 8  # per_device
# batch_size = 2 # 4090
batch_size = 4 # A40
# batch_size = 1 # debug
# batch_size = 3 # debug bev
accumulative_counts = 2
# accumulative_counts = 1
# dataloader_num_workers = 2 # 4090
dataloader_num_workers = 4 # A40
# dataloader_num_workers = 0 # debug

# max_epochs = 1
# max_iters = 10000  # set your desired total iterations here
num_epoch_budget = 1 
# num_epoch_budget = 4.5
# num_epoch_budget = 10 # stage 2
# training_set = 1904
if 'mini' in pkl_path:
    training_set = 1000
elif 'overfit' in pkl_path:
    training_set = 1000
else:
    training_set = 75116  # target visible is 76419, all_navigable is 92802, has npz is 133531, total is 240329
max_iters = num_epoch_budget * (training_set // batch_size)  # IterBasedTrainLoop counts dataloader micro-iters; accumulation is handled by optim_wrapper single epoch hardcode
optim_type = AdamW
# official 1024 -> 4e-5
# lr = 1e-6
lr = 2e-05
betas = (0.9, 0.999)
weight_decay = 0.05
# weight_decay = 0.0
max_norm = 1  # grad clip
warmup_ratio = 0.03
lr_mult_ratio = 1.0

# # v2 (1e-4 not decreasing loss with 0.03 warmup); this one also not decreasing loss
# lr = 1e-4
# warmup_ratio = 0.10

# # v4  , not working worse than before
# lr_mult_ratio = 10.0

# # v5 still not working
# lr = 1e-6
# lr_mult_ratio = 1.0

# NOTE: 1e-5 lr, 4.5e total and stop at 3900 iters gives best result for the affordance stage
# is_stage2 = False
# log_interval = 10

is_stage2 = False
log_interval = 10 # 10

# finetune_head_lr = lr if not is_stage2 else 1e-4
# finetune_head_lr_mult = finetune_head_lr / lr
finetune_head_lr_mult = 5.0 if not is_stage2 else 5.0

# Save
save_steps = 10000 # 200 # 500 # 300 # 100 # 300 # debug
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)
use_pe3d = True
method_variant = 'catbev' # 'mlpxy' or 'catbev'
log_name = "QA" if output_mode == "VQA" else method_variant
mini_tag = ''
if 'mini' in pkl_path:
    mini_tag = 'mini'
elif 'overfit' in pkl_path:
    mini_tag = 'overfit'
work_dir = f'work_dir/{log_name}_PE3D_{use_pe3d}_batch{batch_size}x{int(accumulative_counts)}_lr{lr}x{int(finetune_head_lr_mult)}_epoch{num_epoch_budget}_fromnone_{mini_tag}'

# class balance optional
region_cls_freq = None
# region_cls_freq = {
#     "FRONT_SMALL": 277,
#     "FRONT_BIG": 244,
#     "FRONTLEFT_SMALL": 197,
#     "FRONTLEFT_BIG": 185,
#     "LEFT_SMALL": 90,
#     "LEFT_BIG": 139,
#     "BACKLEFT_SMALL": 30,
#     "BACKLEFT_BIG": 48,
#     "BACK_SMALL": 15,
#     "BACK_BIG": 25,
#     "BACKRIGHT_SMALL": 16,
#     "BACKRIGHT_BIG": 33,
#     "RIGHT_SMALL": 107,
#     "RIGHT_BIG": 74,
#     "FRONTRIGHT_SMALL": 217,
#     "FRONTRIGHT_BIG": 207
# }


#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
# model = dict(
#     type=InternVL_V1_5,
#     model_path=path,
#     freeze_llm=True,
#     freeze_visual_encoder=True,
#     # comment the following lines if you don't want to use Lora in llm
#     llm_lora=dict(
#         type=LoraConfig,
#         # r=128,
#         r=16,
#         lora_alpha=256,
#         lora_dropout=0.05,
#         target_modules=None,
#         task_type='CAUSAL_LM'),
#     # uncomment the following lines if you don't want to use Lora in visual encoder # noqa
#     # visual_encoder_lora=dict(
#     #     type=LoraConfig, r=64, lora_alpha=16, lora_dropout=0.05,
#     #     target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'])
# )

model = dict(
    type=InternVL_V1_5_NavTarget,
    model_path=path,
    freeze_llm=True,  # keep base frozen; embeddings are unfrozen in class
    freeze_visual_encoder=True,
    llm_lora=dict(
        type=LoraConfig, 
        r=16, 
        lora_alpha=256, 
        lora_dropout=0.05, 
        target_modules=None, 
        task_type='CAUSAL_LM'),
    reg_weight=0.5, # not used in RoboPoint mode
    token_ce_weight=1.0,          # keep some CE to prevent forgetting
    use_meta_action=False,      # whether to use meta action onehot in regression head
    use_pred_meta_action_onehot=False,  # flip to True to use predicted one-hot (detached)
    fuse_visual_tokens=False,  # adding it makes things worse
    output_mode=output_mode,
    # use_pe3d=True, 
    use_pe3d=use_pe3d,
    num_views=4, # not used
    region_cls_freq=region_cls_freq,
    finetune_head=dict(
        type=AffordanceHead,
        variant=method_variant,
    ) if output_mode == "Affordance" else None
)


#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
llava_dataset = dict(
    # type=InternVL_V1_5_Dataset,
    # type=InternVL_V1_5_Dataset_Multiview,
    # model_path=path,
    # data_paths=data_path,
    # image_folders=image_folder,
    # template=prompt_template,
    # max_length=max_length)
    type=InternVL_V1_5_Dataset_Multiview_PKL,
    model_path=path,
    pkl_path=pkl_path,
    dataset_root=pkl_dataset_root,
    template=prompt_template,
    max_length=max_length,
    loader_kwargs=pkl_loader_kwargs,
    output_mode=output_mode,
)
# train_sampler = BalancedLengthGroupedSamplerSingleGPU if output_mode == "Affordance" else LengthGroupedSampler
train_sampler = LengthGroupedSampler # debug ablate
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=llava_dataset,
    sampler=dict(
        type=train_sampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    # collate_fn=dict(type=default_collate_fn))
    collate_fn=dict(type=custom_collate_fn))

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    paramwise_cfg=dict(
        custom_keys={
            # x LR for tgt_head_polar only
            'tgt_head_polar': dict(lr_mult=lr_mult_ratio),
            'finetune_head': dict(lr_mult=finetune_head_lr_mult),
        }
    ),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=True),
    accumulative_counts=accumulative_counts,
    # loss_scale='dynamic',
    loss_scale=dict(enabled=False),   # key: disables GradScaler, it does not support bfloat16
    # dtype='float16')
    dtype='bfloat16') 


# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        # by_epoch=True,
        by_epoch=False,
        begin=0,
        # end=warmup_ratio * max_epochs,
        end=warmup_ratio * max_iters,
        # convert_to_iter_based=True),
    ),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        # by_epoch=True,
        by_epoch=False,
        # begin=warmup_ratio * max_epochs,
        begin=warmup_ratio * max_iters,
        # end=max_epochs,
        end=max_iters,
        # convert_to_iter_based=True)
    ),
]

# train, val, test setting
# train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters)


#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=path,
    trust_remote_code=True)

custom_hooks = [
    # dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(type=DatasetInfoHookSpecialTokens, tokenizer=tokenizer),
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=log_interval),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=False,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit,
        save_last=True,
    ),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(project='thesis_hpc', name=f'{log_name}_PE3D_{use_pe3d}_batch{batch_size}x{accumulative_counts}_lr{lr}x{int(finetune_head_lr_mult)}_epoch{num_epoch_budget}_fromnone_{mini_tag}')),
]
visualizer = dict(type='Visualizer', vis_backends=vis_backends)


# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None # BUG: not adding this will cause many random syntax errors in the test mode
# whether to resume training from the loaded checkpoint
resume = False
# resume = True # debug

if output_mode == "Affordance":
    if _IS_TEST:
        load_from = None # set in command line, but this one has higher priority so set to None here
        resume = False # BUG: if don't do this it will throw error even though resume is clearly not a testing config
    else:
        if not resume:
            if is_stage2:
                load_from = f'{BEACON_WORK_ROOT}/work_dir/QA_PE3D_True_batch4x2_lr3e-05_epoch2_/iter_37558.pth'
            else:
                load_from = None # use PE3D and isolate
        else:
            load_from = None

del _IS_TEST  # clean up

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)

#######################################################################
#                           PART 6  Evaluation                        #
#######################################################################
val_pkl = pkl_path # TODO: set val pkl path
val_root = pkl_dataset_root  # TODO: set val dataset root
val_llava_dataset = dict(
    type=InternVL_V1_5_Dataset_Multiview_PKL,
    # point to your val pkl/split, root, same template/model_path as train
    model_path=path,
    template=prompt_template,
    pkl_path=val_pkl,
    dataset_root=val_root,
    max_length=max_length * 10, # val can have longer max length
    loader_kwargs=pkl_loader_kwargs,
    output_mode=f'{output_mode}_eval',  # e.g., "RoboPoint_eval" or "Affordance_eval"
    )

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=val_llava_dataset,
    sampler=dict(type=DefaultSampler, shuffle=False),
    collate_fn=dict(type=custom_collate_fn),
)

test_evaluator = dict(
    type=ExternalMetric, 
    save_path=f"{work_dir}/val_seen_{output_mode}_overfit.npz"
)
test_cfg = dict(type="TestLoop")
