import math
import torch
import torch.nn as nn
from mmengine.model import BaseModule


        
        
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor
from xtuner.perception_modules.utils import lss_bev_pool, BEVResNetFPN, ResNetBEV, FPNBEV, LN2d, MiniSECONDEncoderDense, LN3d, BasicBlock3D

import numpy as np

# only seems working verion using bev cross attention, the deformable + anchor verison is not good, just the version B is good. 


        # # version B: bev regress
        # # ---- 6) BEV Feats ----
        # skip_result_xy = self.xy_from_polar_logits(skip_result)  # (B, 2)
        # skip_result_bev_coords = self.bev_from_xy(skip_result_xy)  # (B, 2)
        # target_anchor = self.heatmap_from_bev_coords(bev_coords=skip_result_bev_coords) # B, Dx, Dy
        # ego_anchor = self.heatmap_from_bev_coords(bev_coords=ego_bev_coords) # B, Dx, Dy
        # bev_feats_with_anchor = torch.cat([bev_feats, target_anchor.unsqueeze(1), ego_anchor.unsqueeze(1)], dim=1)  # (B, C+2, Dx, Dy)

        # bev_feats_out = self.bev_backbone(bev_feats_with_anchor)    # (B, C+2, Dx, Dy)

        # # --- 7) Motion Refinement Head ---
        # waypoint_logits = self.waypoint_head(bev_feats_out).permute(0, 2, 3, 1).contiguous()  # (B, Dx, Dy, 6)
        # return None

        # # ---- 7) Occ Head ----
        # occ_logits = self.occ_head(bev_feats_out).squeeze(1)  # (B, Dx, Dy)


        # # # debug
        # # import numpy as np
        # # debug_save_path = f'debug_outputs/debug_lss_result_{self.debug_cnt}.jpg'
        # # assert bev_feats.shape[0] == 1, "Debug only supports batch size 1."
        # # bev_np = bev_feats[0, :, :, :].detach().cpu().numpy()  # (C, Dx, Dy)
        # # bev_non_zero = np.abs(bev_np).sum(axis=0)                  # (Dx, Dy)
        # # import cv2
        # # bev_img = cv2.normalize(bev_non_zero, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # # cv2.imwrite(debug_save_path, bev_img)
        # # occ_result = occ_logits[0, :, :].detach().cpu().numpy()
        # # occ_result = np.where(occ_result > 0, 1.0, 0.0).astype(np.uint8) * 255
        # # debug_occ_save_path = f'debug_outputs/debug_occ_result_{self.debug_cnt}.jpg'
        # # cv2.imwrite(debug_occ_save_path, occ_result)
        # # print(f"[DEBUG] Saved LSS BEV features to {debug_save_path} and occ result to {debug_occ_save_path}")
        # # self.debug_cnt += 1
        # # # exit(0)

        # return occ_logits # (B, Dx, Dy)


import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp
import numpy as np
import math


# ------------------------------------------------------------
# Non-trivial blocks to match CogACT style:
# - PreNorm Transformer block:
#   norm1: LayerNorm(D, elementwise_affine=False, eps=1e-6)
#   attn : timm.Attention(D, num_heads=H, qkv_bias=True)
#   norm2: LayerNorm(D, elementwise_affine=False, eps=1e-6)
#   mlp  : timm.Mlp(D -> 4D -> D, GELU(approx="tanh"), drop=0)
#
# Init (to match CogACT-ish):
# - All Linear: Xavier uniform, bias=0
# - Then override some key layers with Normal(0,0.02) (optional)
# - Final projection(s) often set to 0 for stability (optional)
# ------------------------------------------------------------


# no diffusion anchor head
        # return score.squeeze(-1) # (B, K)


# =========================
# 0) IMPORT CHANGES (top of file)
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F  # <<< ADD (you already use F in ade_soft_anchor_ce_loss)
from timm.models.vision_transformer import Attention, Mlp
import numpy as np
import math
from diffusers.schedulers import DDIMScheduler  # <<< ADD


# =========================
# 1) ADD THIS CLASS (place it ABOVE AffordanceHead)
#    (exactly the CogACT timestep embedder you pasted)
# =========================

# diffusion anchor head


# bev affordance
from torch.cuda.amp import autocast
    


# special version for ablation
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler
from diffusers.models import UNet2DModel
from mmcv.ops import MultiScaleDeformableAttention
from torch.utils.checkpoint import checkpoint
class AffordanceHead(BaseModule):
    def __init__(
        self,
        data_preprocessor=dict(
            type='Det3DDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_size_divisor=1,
            pad_value=0,
        ),
        backbone=dict(
            type='TorchHubModel',
            repo_or_dir='facebookresearch/dinov2',
            model_name='dinov2_vitb14_reg'
        ),
        grid_config=dict(
            x=[-6.4, 6.4, 0.1],       # 128
            y=[-6.4, 6.4, 0.1],       # 128
            # z=[-1.6, 0.2, 1.8],       # 1
            z=[-1.6, 0.3, 1.9],       # 1
            depth=[0.0, 7.2, 0.1],    # 72
        ),
        input_size=(448, 448),
        collapse_z=True,
        proj_out_channels=256,
        bev_head_channels=256,
        variant=None,
    ):
        super().__init__()
        self.grid_config = grid_config
        self.input_size = input_size
        self.collapse_z = collapse_z

        # 1) Det3DDataPreprocessor in the loop (your raw is uint8 BGR; DINOv2 expects RGB float normalized)
        self.data_preprocessor = Det3DDataPreprocessor(
            mean=data_preprocessor.get("mean", [123.675, 116.28, 103.53]),
            std=data_preprocessor.get("std", [58.395, 57.12, 57.375]),
            bgr_to_rgb=data_preprocessor.get("bgr_to_rgb", True),
            pad_size_divisor=data_preprocessor.get("pad_size_divisor", 1),
            pad_value=data_preprocessor.get("pad_value", 0),
            pad_mask=False,
            pad_seg=False,
            batch_augments=None,
        )

        # 2) Backbone (torch.hub DINOv2, loaded from a LOCAL clone)
        if backbone.get("type", None) == "TorchHubModel":
            # self.backbone = torch.hub.load(backbone["repo_or_dir"], backbone["model_name"])
            import os
            # DINOv2 is loaded with source="local" from a clone of
            # facebookresearch/dinov2 named `facebookresearch_dinov2_main` under
            # BEACON_DINOV2_ROOT (defaults to the original container layout).
            dinov2_root = os.environ.get("BEACON_DINOV2_ROOT", "/model/DINOv2")
            os.environ["TORCH_HOME"] = dinov2_root  # so hub looks for checkpoints here
            self.backbone = torch.hub.load(
                os.path.join(dinov2_root, "facebookresearch_dinov2_main"),
                backbone["model_name"],
                source="local",
            )
            self.backbone.requires_grad_(False)
            self.backbone.is_init = True  # mmengine safety
            self.backbone.requires_grad_(False)
            self.backbone.eval()

            ps = getattr(self.backbone, "patch_size", 14)
            if isinstance(ps, (tuple, list)):
                ps = int(ps[0])
            self.patch_size = int(ps)
        else:
            raise NotImplementedError("Only TorchHubModel backbone is supported.")

        # You fixed to P(0): token grid resolution is H/patch_size, W/patch_size
        self.downsample = self.patch_size  # == 14 in your setup
        
        # for version A:
        # 3) Minimal replacement for "use only P(0)" from ViTDetFPN:
        #    a simple 1x1 projection from DINOv2 dim (768) to desired context dim (default 256).
        self.proj_out_channels = int(proj_out_channels)
        # self.proj_out_channels = 1 # debug and reduce vram
        self.img_proj = nn.Conv2d(768, self.proj_out_channels, kernel_size=1, bias=True)

        # self.bev_backbone = BEVResNetFPN(cin=self.proj_out_channels + 2, depth=10, base=64, fpn_dim=bev_head_channels + 2, out_dim=bev_head_channels + 2)
        # self.bev_encoder = ResNetBEV(cin=256, base=128, depth=10, norm='ln') # keep it same as v1 which can be reused
        # self.bev_fpn = FPNBEV(in_channels_list=[128, 256, 512, 1024], fpn_dim=bev_head_channels)
        self.bevfpn_fuse = nn.Sequential(
            nn.Conv2d(bev_head_channels, bev_head_channels, 3, padding=1, bias=False),
            # nn.BatchNorm2d(bev_head_channels),
            LN2d(bev_head_channels),
            nn.ReLU(inplace=True),
        )

        def _num_bins(vmin, vmax, step):
            return int(round((float(vmax) - float(vmin)) / float(step)))

        self.Dx = _num_bins(*grid_config['x'])
        self.Dy = _num_bins(*grid_config['y'])
        self.Dz = _num_bins(*grid_config['z'])
        assert self.Dx > 0 and self.Dy > 0 and self.Dz > 0, (self.Dx, self.Dy, self.Dz, self.grid_config)

        self.affordance_head = nn.Sequential(
            nn.Conv2d(bev_head_channels, bev_head_channels//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_head_channels//2, 1, kernel_size=1),
        )
        # # Method A: cat before resnet
        # self.query_to_prior = nn.Sequential(
        #     nn.Linear(2048, 256 * 4 * 4),
        #     nn.GELU(),
        # )
        # self.prior_upsampler = nn.Sequential(
        #     self.up2_block(256),
        #     self.up2_block(256),
        #     self.up2_block(256),
        # )
        # # self.prior_proj = nn.Conv2d(256, 256, kernel_size=1, bias=True)
        # self.prior_proj = nn.Identity()
        # # self.fuse_prior = nn.Sequential(
        # #     nn.Conv2d(bev_head_channels + self.prior_channels, bev_head_channels, kernel_size=3, padding=1, bias=False),
        # #     LN2d(bev_head_channels),
        # #     nn.ReLU(inplace=True),
        # # )

        # # Method B: standard film before resnet
        # self.bev_film_norm = LN2d(self.proj_out_channels)  # or nn.GroupNorm(32, C)
        # self.film_mlp = nn.Sequential(
        #     nn.Linear(2048, 512),
        #     nn.GELU(),
        #     nn.Linear(512, 2 * self.proj_out_channels),
        # )
        # self.obs_proj = nn.Conv2d(self.proj_out_channels + 2,
        #                   self.proj_out_channels,
        #                   kernel_size=1,
        #                   bias=True)

        # # Method C: film in FPN
        # # self.obs_proj = nn.Conv2d(self.proj_out_channels + 2, # for method C.1
        # self.obs_proj = nn.Conv2d(self.proj_out_channels + 3, # for method C.2
        #     self.proj_out_channels,
        #     kernel_size=1,
        #     bias=True)
        # self.bev_encoder = ResNetBEV(cin=256, base=128, depth=10, norm='ln') # keep it same as v1 which can be reused
        # self.bev_fpn = FPNBEV(in_channels_list=[128, 256, 512, 1024], fpn_dim=bev_head_channels)
        # self.query_to_prior16 = nn.Sequential(
        #     nn.Linear(2048, 32 * 16 * 16),
        #     nn.GELU(),
        # )
        # self.prior_up_32 = self.up2_block(32)  # 16->32
        # self.prior_up_64 = self.up2_block(32)  # 32->64
        # self.prior_up_128 = self.up2_block(32)  # 64->128
        # # For each level, turn prior feature into gamma/beta maps matching that level's channel count.
        # # (B, prior_ch, H, W) -> (B, 2*C_level, H, W) -> split into gamma/beta.
        # self.film2 = nn.Conv2d(32, 2 * 128, kernel_size=1, bias=True)   # for c2
        # self.film3 = nn.Conv2d(32, 2 * 256, kernel_size=1, bias=True)   # for c3
        # self.film4 = nn.Conv2d(32, 2 * 512, kernel_size=1, bias=True)   # for c4
        # self.film5 = nn.Conv2d(32, 2 * 1024, kernel_size=1, bias=True)  # for c5
        # # Norms before FiLM (diffusion standard: norm then scale/shift)
        # self.norm2 = LN2d(128)
        # self.norm3 = LN2d(256)
        # self.norm4 = LN2d(512)
        # self.norm5 = LN2d(1024)

        # # for Method C.3
        # self.point_encoder = MiniSECONDEncoderDense(in_channels=3,)
        # self.obs_proj_bevfusion = nn.Conv2d(self.proj_out_channels * 2 + 3, # for method C.3, must cat point feature then lss feature, cannot reverse
        #     self.proj_out_channels * 2,
        #     kernel_size=1,
        #     bias=True)
        # self.bev_encoder = ResNetBEV(cin=512, base=256, depth=10, norm='ln') # keep it same as v1 which can be reused
        # self.bev_fpn = FPNBEV(in_channels_list=[256, 512, 1024, 2048], fpn_dim=bev_head_channels)
        # self.query_to_prior16 = nn.Sequential(
        #     nn.Linear(2048, 32 * 16 * 16),
        #     nn.GELU(),
        # )
        # self.prior_up_32 = self.up2_block(32)  # 16->32
        # self.prior_up_64 = self.up2_block(32)  # 32->64
        # self.prior_up_128 = self.up2_block(32)  # 64->128
        # # For each level, turn prior feature into gamma/beta maps matching that level's channel count.
        # # (B, prior_ch, H, W) -> (B, 2*C_level, H, W) -> split into gamma/beta.
        # self.film2 = nn.Conv2d(32, 2 * 256, kernel_size=1, bias=True)   # for c2
        # self.film3 = nn.Conv2d(32, 2 * 512, kernel_size=1, bias=True)   # for c3
        # self.film4 = nn.Conv2d(32, 2 * 1024, kernel_size=1, bias=True)   # for c4
        # self.film5 = nn.Conv2d(32, 2 * 2048, kernel_size=1, bias=True)  # for c5
        # # Norms before FiLM (diffusion standard: norm then scale/shift)
        # self.norm2 = LN2d(256)
        # self.norm3 = LN2d(512)
        # self.norm4 = LN2d(1024)
        # self.norm5 = LN2d(2048)

        # self.pc_gate = nn.Sequential(
        #     nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=True),
        #     nn.GELU(),
        #     nn.Conv2d(3, 1, kernel_size=1, bias=True),
        # )
        # self.use_bev_feats = True

        ################### begin of ablation options ###################

        if variant is not None:
            self.variant = variant.lower()

        if self.variant == 'filmbev':
            # Method ablation 0: copied Method C. FiLM in FPN
            self.use_bev_feats = True
            self.obs_proj = nn.Conv2d(self.proj_out_channels + 3, # for method C.2
                self.proj_out_channels,
                kernel_size=1,
                bias=True)
            self.bev_encoder = ResNetBEV(cin=256, base=128, depth=10, norm='ln') # keep it same as v1 which can be reused
            self.bev_fpn = FPNBEV(in_channels_list=[128, 256, 512, 1024], fpn_dim=bev_head_channels)
            self.query_to_prior16 = nn.Sequential(
                nn.Linear(2048, 32 * 16 * 16),
                nn.GELU(),
            )
            self.prior_up_32 = self.up2_block(32)  # 16->32
            self.prior_up_64 = self.up2_block(32)  # 32->64
            self.prior_up_128 = self.up2_block(32)  # 64->128
            # For each level, turn prior feature into gamma/beta maps matching that level's channel count.
            # (B, prior_ch, H, W) -> (B, 2*C_level, H, W) -> split into gamma/beta.
            self.film2 = nn.Conv2d(32, 2 * 128, kernel_size=1, bias=True)   # for c2
            self.film3 = nn.Conv2d(32, 2 * 256, kernel_size=1, bias=True)   # for c3
            self.film4 = nn.Conv2d(32, 2 * 512, kernel_size=1, bias=True)   # for c4
            self.film5 = nn.Conv2d(32, 2 * 1024, kernel_size=1, bias=True)  # for c5
            # Norms before FiLM (diffusion standard: norm then scale/shift)
            self.norm2 = LN2d(128)
            self.norm3 = LN2d(256)
            self.norm4 = LN2d(512)
            self.norm5 = LN2d(1024)

            # for Method C.3
            self.point_encoder = MiniSECONDEncoderDense(in_channels=3,)
            self.obs_proj_bevfusion = nn.Conv2d(self.proj_out_channels * 2 + 3, # for method C.3, must cat point feature then lss feature, cannot reverse
                self.proj_out_channels * 2,
                kernel_size=1,
                bias=True)
            self.bev_encoder = ResNetBEV(cin=512, base=256, depth=10, norm='ln') # keep it same as v1 which can be reused
            self.bev_fpn = FPNBEV(in_channels_list=[256, 512, 1024, 2048], fpn_dim=bev_head_channels)
            self.query_to_prior16 = nn.Sequential(
                nn.Linear(2048, 32 * 16 * 16),
                nn.GELU(),
            )
            self.prior_up_32 = self.up2_block(32)  # 16->32
            self.prior_up_64 = self.up2_block(32)  # 32->64
            self.prior_up_128 = self.up2_block(32)  # 64->128
            # For each level, turn prior feature into gamma/beta maps matching that level's channel count.
            # (B, prior_ch, H, W) -> (B, 2*C_level, H, W) -> split into gamma/beta.
            self.film2 = nn.Conv2d(32, 2 * 256, kernel_size=1, bias=True)   # for c2
            self.film3 = nn.Conv2d(32, 2 * 512, kernel_size=1, bias=True)   # for c3
            self.film4 = nn.Conv2d(32, 2 * 1024, kernel_size=1, bias=True)   # for c4
            self.film5 = nn.Conv2d(32, 2 * 2048, kernel_size=1, bias=True)  # for c5
            # Norms before FiLM (diffusion standard: norm then scale/shift)
            self.norm2 = LN2d(256)
            self.norm3 = LN2d(512)
            self.norm4 = LN2d(1024)
            self.norm5 = LN2d(2048)

            self.pc_gate = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=True),
                nn.GELU(),
                nn.Conv2d(3, 1, kernel_size=1, bias=True),
            )


        if self.variant == "plainbev":
            # Method ablation 1: bev output but no fusion
            self.use_bev_feats = False
            
            # really used parts
            self.query_to_prior16 = nn.Sequential(
                nn.Linear(2048, 32 * 16 * 16),
                nn.GELU(),
            )
            self.prior_up_32 = self.up2_block(32)  # 16->32
            self.prior_up_64 = self.up2_block(32)  # 32->64
            self.prior_up_128 = self.up2_block(32)  # 64->128

            self.bev_encoder = ResNetBEV(cin=32, base=256, depth=10, norm='ln') # keep it same as v1 which can be reused
            self.bev_fpn = FPNBEV(in_channels_list=[256, 512, 1024, 2048], fpn_dim=bev_head_channels)

        # Method ablation 2: cat before resnet
        if self.variant == "catbev":
            self.use_bev_feats = True
            self.point_encoder = MiniSECONDEncoderDense(in_channels=3,)
            self.obs_proj_bevfusion = nn.Conv2d(self.proj_out_channels * 2 + 3, # for method C.3, must cat point feature then lss feature, cannot reverse
                self.proj_out_channels * 2,
                kernel_size=1,
                bias=True)
            self.bev_encoder = ResNetBEV(cin=512+32, base=256, depth=10, norm='ln') # keep it same as v1 which can be reused
            self.bev_fpn = FPNBEV(in_channels_list=[256, 512, 1024, 2048], fpn_dim=bev_head_channels)
            self.query_to_prior16 = nn.Sequential(
                nn.Linear(2048, 32 * 16 * 16),
                nn.GELU(),
            )
            self.prior_up_32 = self.up2_block(32)  # 16->32
            self.prior_up_64 = self.up2_block(32)  # 32->64
            self.prior_up_128 = self.up2_block(32)  # 64->128

            self.pc_gate = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=True),
                nn.GELU(),
                nn.Conv2d(3, 1, kernel_size=1, bias=True),
            )

        if self.variant == "attnxy":
            # Method ablation 3: cross attn to bev and output point
            self.use_bev_feats = True
            self.point_encoder = MiniSECONDEncoderDense(in_channels=3,)
            self.obs_proj_bevfusion = nn.Conv2d(self.proj_out_channels * 2 + 3, # for method C.3, must cat point feature then lss feature, cannot reverse
                self.proj_out_channels * 2,
                kernel_size=1,
                bias=True)
            self.pc_gate = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=True),
                nn.GELU(),
                nn.Conv2d(3, 1, kernel_size=1, bias=True),
            )
            self.bev_encoder = ResNetBEV(cin=512, base=256, depth=10, norm='ln') # keep it same as v1 which can be reused
            self.bev_fpn = FPNBEV(in_channels_list=[256, 512, 1024, 2048], fpn_dim=bev_head_channels)
            self.attn_dim = 512
            self.img_dim = proj_out_channels
            self.query_dim = 2048
            self.num_cross_layers = 3
            pos = self.build_ego_sincos_pos(Dx=128, Dy=128, dim=self.attn_dim, ego=63, sx=0.1, sy=0.1)
            self.register_buffer("positional_encoding", pos, persistent=False)
            self.query_in_proj  = nn.Linear(self.query_dim, self.attn_dim)
            self.query_out_proj = nn.Linear(self.attn_dim, self.query_dim)
            self.cross_ln_q  = nn.ModuleList([nn.LayerNorm(self.attn_dim) for _ in range(self.num_cross_layers)])
            self.cross_ln_kv = nn.ModuleList([nn.LayerNorm(self.attn_dim) for _ in range(self.num_cross_layers)])
            self.cross_mha = nn.ModuleList([
                nn.MultiheadAttention(embed_dim=self.attn_dim, num_heads=8, batch_first=True)
                for _ in range(self.num_cross_layers)
            ])
            self.cross_ln_ffn = nn.ModuleList([nn.LayerNorm(self.attn_dim) for _ in range(self.num_cross_layers)])
            self.cross_ffn = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.attn_dim, self.attn_dim * 4),  # 512 -> 2048
                    nn.GELU(),
                    nn.Linear(self.attn_dim * 4, self.attn_dim),  # 2048 -> 512
                )
                for _ in range(self.num_cross_layers)
            ])
            self.bev_xy_head = nn.Sequential(
                nn.Linear(self.query_dim, self.query_dim//2),
                nn.ReLU(),
                nn.Linear(self.query_dim//2, 2),  # predict 1 waypoint (x,y)
            )

        if self.variant == "catattnbev":
            # Method ablation 4: cat before attn
            self.use_bev_feats = True
            self.point_encoder = MiniSECONDEncoderDense(in_channels=3,)
            self.pc_gate = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=True),
                nn.GELU(),
                nn.Conv2d(3, 1, kernel_size=1, bias=True),
            )
            self.obs_proj_bevfusion = nn.Conv2d(self.proj_out_channels * 2 + 3, # for method C.3, must cat point feature then lss feature, cannot reverse
                self.proj_out_channels * 2,
                kernel_size=1,
                bias=True)
            self.query_to_prior16 = nn.Sequential(
                nn.Linear(2048, 32 * 16 * 16),
                nn.GELU(),
            )
            self.prior_up_32 = self.up2_block(32)  # 16->32
            self.prior_up_64 = self.up2_block(32)  # 32->64
            self.attn_dim = proj_out_channels * 2 + 32
            self.num_cross_layers = 3
            pos = self.build_ego_sincos_pos(Dx=64, Dy=64, dim=self.attn_dim, ego=31.5, sx=0.2, sy=0.2)
            self.register_buffer("positional_encoding", pos, persistent=False)
            self.cross_ln_qkv  = nn.ModuleList([nn.LayerNorm(self.attn_dim) for _ in range(self.num_cross_layers)])
            self.cross_mha = nn.ModuleList([
                nn.MultiheadAttention(embed_dim=self.attn_dim, num_heads=8, batch_first=True)
                for _ in range(self.num_cross_layers)
            ])
            self.cross_ln_ffn = nn.ModuleList([nn.LayerNorm(self.attn_dim) for _ in range(self.num_cross_layers)])
            self.cross_ffn = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.attn_dim, self.attn_dim * 4),  # 256 -> 1024
                    nn.GELU(),
                    nn.Linear(self.attn_dim * 4, self.attn_dim),  # 1024 -> 256
                )
                for _ in range(self.num_cross_layers)
            ])
            self.bev_encoder = ResNetBEV(cin=512+32, base=256, depth=10, norm='ln') # keep it same as v1 which can be reused
            self.bev_fpn = FPNBEV(in_channels_list=[256, 512, 1024, 2048], fpn_dim=bev_head_channels)
            # self.affordance_head_2 = nn.Sequential(
            #     nn.Conv2d(self.attn_dim, self.attn_dim//2, kernel_size=3, padding=1),
            #     nn.ReLU(inplace=True),
            #     nn.Conv2d(self.attn_dim//2, 1, kernel_size=1),
            # )

        # # Method ablation 5: cat before resnet and diffusion
        # self.use_bev_feats = True
        # self.point_encoder = MiniSECONDEncoderDense(in_channels=3,)
        # self.obs_proj_bevfusion = nn.Conv2d(self.proj_out_channels * 2 + 3, # for method C.3, must cat point feature then lss feature, cannot reverse
        #     self.proj_out_channels * 2,
        #     kernel_size=1,
        #     bias=True)
        # self.bev_encoder = ResNetBEV(cin=512+32, base=256, depth=10, norm='ln') # keep it same as v1 which can be reused
        # self.bev_fpn = FPNBEV(in_channels_list=[256, 512, 1024, 2048], fpn_dim=bev_head_channels)
        # self.query_to_prior16 = nn.Sequential(
        #     nn.Linear(2048, 32 * 16 * 16),
        #     nn.GELU(),
        # )
        # self.prior_up_32 = self.up2_block(32)  # 16->32
        # self.prior_up_64 = self.up2_block(32)  # 32->64
        # self.prior_up_128 = self.up2_block(32)  # 64->128

        # self.pc_gate = nn.Sequential(
        #     nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=True),
        #     nn.GELU(),
        #     nn.Conv2d(3, 1, kernel_size=1, bias=True),
        # )

        # # first try to add unconditional diffusion 
        # self.diff_T = 100
        # self.diff_infer_steps = 20
        # self.diff_train_scheduler = DDPMScheduler(
        #     num_train_timesteps=self.diff_T,
        #     beta_schedule="squaredcos_cap_v2",
        #     prediction_type="epsilon",  # predict noise
        # )
        # self.diff_infer_scheduler = DPMSolverMultistepScheduler.from_config(self.diff_train_scheduler.config)
        # # Fast inference sampler (much fewer steps than T)
        # self.diff_unet = UNet2DModel(
        #     sample_size=None,          # accept arbitrary H,W (128x128 ok)
        #     in_channels=1,             # x_t only (unconditional)
        #     out_channels=1,            # predict epsilon with same channels as x_t
        #     layers_per_block=2,
        #     block_out_channels=(64, 128, 256, 512),   # closest to your ResNet+FPN stage widths
        #     down_block_types=("DownBlock2D","DownBlock2D","DownBlock2D","DownBlock2D"),
        #     up_block_types=("UpBlock2D","UpBlock2D","UpBlock2D","UpBlock2D"),
        #     add_attention=False,
        # )

        if self.variant == "catlss":
            # Method ablation 6: cat before resnet, only lss
            self.use_bev_feats = False
            self.point_encoder = MiniSECONDEncoderDense(in_channels=3,)
            self.obs_proj_bevfusion = nn.Conv2d(self.proj_out_channels * 2 + 3, # for method C.3, must cat point feature then lss feature, cannot reverse
                self.proj_out_channels * 2,
                kernel_size=1,
                bias=True)
            self.bev_encoder = ResNetBEV(cin=256+32, base=256, depth=10, norm='ln') # keep it same as v1 which can be reused
            self.bev_fpn = FPNBEV(in_channels_list=[256, 512, 1024, 2048], fpn_dim=bev_head_channels)
            self.query_to_prior16 = nn.Sequential(
                nn.Linear(2048, 32 * 16 * 16),
                nn.GELU(),
            )
            self.prior_up_32 = self.up2_block(32)  # 16->32
            self.prior_up_64 = self.up2_block(32)  # 32->64
            self.prior_up_128 = self.up2_block(32)  # 64->128

        if self.variant == "catpoint":
            # Method ablation 7: cat before resnet, only pointcloud
            self.use_bev_feats = False
            self.point_encoder = MiniSECONDEncoderDense(in_channels=3,)
            self.obs_proj_bevfusion = nn.Conv2d(self.proj_out_channels * 2 + 3, # for method C.3, must cat point feature then lss feature, cannot reverse
                self.proj_out_channels * 2,
                kernel_size=1,
                bias=True)
            self.bev_encoder = ResNetBEV(cin=256+32, base=256, depth=10, norm='ln') # keep it same as v1 which can be reused
            self.bev_fpn = FPNBEV(in_channels_list=[256, 512, 1024, 2048], fpn_dim=bev_head_channels)
            self.query_to_prior16 = nn.Sequential(
                nn.Linear(2048, 32 * 16 * 16),
                nn.GELU(),
            )
            self.prior_up_32 = self.up2_block(32)  # 16->32
            self.prior_up_64 = self.up2_block(32)  # 32->64
            self.prior_up_128 = self.up2_block(32)  # 64->128

        if self.variant == "catnogate":
            # Method ablation 8: cat before resnet, no gate
            self.use_bev_feats = False
            self.point_encoder = MiniSECONDEncoderDense(in_channels=3,)
            self.obs_proj_bevfusion = nn.Conv2d(self.proj_out_channels * 2 + 3, # for method C.3, must cat point feature then lss feature, cannot reverse
                self.proj_out_channels * 2,
                kernel_size=1,
                bias=True)
            self.bev_encoder = ResNetBEV(cin=512+32, base=256, depth=10, norm='ln') # keep it same as v1 which can be reused
            self.bev_fpn = FPNBEV(in_channels_list=[256, 512, 1024, 2048], fpn_dim=bev_head_channels)
            self.query_to_prior16 = nn.Sequential(
                nn.Linear(2048, 32 * 16 * 16),
                nn.GELU(),
            )
            self.prior_up_32 = self.up2_block(32)  # 16->32
            self.prior_up_64 = self.up2_block(32)  # 32->64
            self.prior_up_128 = self.up2_block(32)  # 64->128

        # Method ablation 9: directly output point
        if self.variant == "mlpxy":
            self.use_bev_feats = False
            self.query_dim = 2048
            self.bev_xy_head = nn.Sequential(
                nn.Linear(self.query_dim, self.query_dim//2),
                nn.ReLU(),
                nn.Linear(self.query_dim//2, 2),  # predict 1 waypoint (x,y)
            )

        # # Method ablation 10: directly output point with diffusion
        # from xtuner.perception_modules.octo_action_head import SimpleDiffusionXYActionHeadTorch, cosine_beta_schedule
        # self.use_bev_feats = False
        # self.query_dim = 2048
        # self.bev_xy_head = SimpleDiffusionXYActionHeadTorch(
        #     embed_dim=self.query_dim,
        #     action_dim=2,
        #     diffusion_steps=20,
        #     n_diffusion_samples=1,   # training samples per datapoint
        # )
        # self.eval_K = 1
        # self.eval_seed = 42
        # self.data_mean = (1.0, 0.0)
        # self.data_std = (1.5, 2.0)

        # # Method ablation 11: deformable attn
        # self.use_bev_feats = True
        # self.num_layers = 3
        # self.point_encoder = MiniSECONDEncoderDense(in_channels=3,)
        # self.obs_proj_bevfusion = nn.Conv2d(self.proj_out_channels * 2 + 3, # for method C.3, must cat point feature then lss feature, cannot reverse
        #     self.proj_out_channels * 2,
        #     kernel_size=1,
        #     bias=True)
        # self.pc_gate = nn.Sequential(
        #     nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=True),
        #     nn.GELU(),
        #     nn.Conv2d(3, 1, kernel_size=1, bias=True),
        # )
        # self.bev_encoder = ResNetBEV(cin=512, base=256, depth=10, norm='ln') # keep it same as v1 which can be reused
        # self.bev_fpn = FPNBEV(in_channels_list=[256, 512, 1024, 2048], fpn_dim=512)
        # self.attn_dim = 512
        # self.img_dim = proj_out_channels
        # self.query_dim = 2048
        # self.num_anchors = 32
        # self.num_tokens = 1 + self.num_anchors
        # d = self.attn_dim  # 512
        # def _make_anchors_disk_xy01(K=32, device="cpu"):
        #     assert K == 32, "K must be 32 (8 rays * 4 radii)"
        #     R = 6.4
        #     radii_by_ray = [
        #         [1.736, 2.580, 3.548, 4.407],  # ray 0
        #         [1.390, 2.607, 3.813, 4.499],  # ray 1 (avg of 1&7)
        #         [1.226, 1.989, 2.990, 4.373],  # ray 2 (avg of 2&6)
        #         [1.199, 2.147, 2.715, 4.089],  # ray 3 (avg of 3&5)
        #         [0.971, 1.826, 2.727, 4.158],  # ray 4
        #         [1.199, 2.147, 2.715, 4.089],  # ray 5 (mirror of ray 3)
        #         [1.226, 1.989, 2.990, 4.373],  # ray 6 (mirror of ray 2)
        #         [1.390, 2.607, 3.813, 4.499],  # ray 7 (mirror of ray 1)
        #     ]
        #     anchors = []
        #     for ray in range(8):
        #         theta = 2.0 * math.pi * (ray / 8)  # ray center angle
        #         ct, st = math.cos(theta), math.sin(theta)
        #         for r in radii_by_ray[ray]:
        #             x = r * ct
        #             y = r * st
        #             x01 = x / (2.0 * R) + 0.5
        #             y01 = y / (2.0 * R) + 0.5
        #             anchors.append([x01, y01])
        #     anchors = torch.tensor(anchors, dtype=torch.float32, device=device)  # (32, 2)
        #     anchors = anchors.clamp(0.0, 1.0)
        #     return anchors
        # anchors_xy01 = _make_anchors_disk_xy01(K=self.num_anchors)
        # self.register_buffer("anchor_xy01", anchors_xy01, persistent=False)
        # self.anchor_content = nn.Embedding(self.num_anchors, d)
        # self.token_pos = nn.Embedding(self.num_anchors+1, d)
        # # self.query_in_proj  = nn.Linear(self.query_dim, self.attn_dim)
        # # self.self_ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(self.num_layers)])
        # # self.self_mha = nn.ModuleList([
        # #     nn.MultiheadAttention(embed_dim=d, num_heads=4, batch_first=True)
        # #     for _ in range(self.num_layers)
        # # ])
        # self.lang_film = nn.Sequential(
        #     nn.Linear(self.query_dim, 2*d),
        # )
        # self.cross_ln_q = nn.ModuleList([nn.LayerNorm(d) for _ in range(self.num_layers)])
        # self.cross_ln_v = nn.ModuleList([nn.LayerNorm(d) for _ in range(self.num_layers)])
        # self.lvl_embed = nn.Embedding(3, d)
        # # bev_pos = self.build_ego_sincos_pos(Dx=128, Dy=128, dim=self.attn_dim, ego=63, sx=0.1, sy=0.1)
        # # self.register_buffer("bev_pos", bev_pos, persistent=False)
        # self.msdeform = nn.ModuleList([
        #     MultiScaleDeformableAttention(
        #         embed_dims=d, num_heads=4, num_levels=3, num_points=4, batch_first=True, 
        #     )
        #     for _ in range(self.num_layers)
        # ])
        # self.msdeform_ffn_ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(self.num_layers)])
        # self.msdeform_ffn = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(d, d * 4),
        #         nn.GELU(),
        #         nn.Linear(d * 4, d),
        #     )
        #     for _ in range(self.num_layers)
        # ])
        # self.anchor_score_head = nn.Linear(d, 1)
        # self.anchor_offset_head = nn.Sequential(
        #     nn.Linear(d, d),
        #     nn.ReLU(),
        #     nn.Linear(d, 2),
        # )
        # self.offset_bound_m = (1.5, 1.0, 0.5) # 3 layers coarse to fine

        # # Method ablation 12: directly waypoint
        # self.use_bev_feats = False
        # self.query_dim = 2048
        # self.bev_xy_head = nn.Sequential(
        #     nn.Linear(self.query_dim, self.query_dim//2),
        #     nn.ReLU(),
        #     nn.Linear(self.query_dim//2, 12),  # predict 1 waypoint (x,y)
        # )

        # # Method ablate 13: mask based waypoint
        # self.use_bev_feats = True
        # self.point_encoder = MiniSECONDEncoderDense(in_channels=3,)
        # self.obs_proj_bevfusion = nn.Conv2d(self.proj_out_channels * 2 + 3, # for method C.3, must cat point feature then lss feature, cannot reverse
        #     self.proj_out_channels * 2,
        #     kernel_size=1,
        #     bias=True)
        # self.bev_encoder = ResNetBEV(cin=512+32, base=256, depth=10, norm='ln') # keep it same as v1 which can be reused
        # self.bev_fpn = FPNBEV(in_channels_list=[256, 512, 1024, 2048], fpn_dim=bev_head_channels)
        # self.query_to_prior16 = nn.Sequential(
        #     nn.Linear(2048, 32 * 16 * 16),
        #     nn.GELU(),
        # )
        # self.prior_up_32 = self.up2_block(32)  # 16->32
        # self.prior_up_64 = self.up2_block(32)  # 32->64
        # self.prior_up_128 = self.up2_block(32)  # 64->128

        # self.pc_gate = nn.Sequential(
        #     nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=True),
        #     nn.GELU(),
        #     nn.Conv2d(3, 1, kernel_size=1, bias=True),
        # )
        # self.wp_mask_head = nn.Sequential(
        #     nn.Conv2d(bev_head_channels, bev_head_channels//2, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(bev_head_channels//2, 6, kernel_size=1),
        # )

        if self.variant == "catvisbev":
            # Method ablation 14: cat before resnet
            self.use_bev_feats = True
            self.point_encoder = MiniSECONDEncoderDense(in_channels=3,)
            self.obs_proj_bevfusion = nn.Conv2d(self.proj_out_channels * 2 + 3, # for method C.3, must cat point feature then lss feature, cannot reverse
                self.proj_out_channels * 2,
                kernel_size=1,
                bias=True)
            self.bev_encoder = ResNetBEV(cin=512+32+32, base=256, depth=10, norm='ln') # keep it same as v1 which can be reused
            self.bev_fpn = FPNBEV(in_channels_list=[256, 512, 1024, 2048], fpn_dim=bev_head_channels)
            self.query_to_prior16 = nn.Sequential(
                nn.Linear(2048, 32 * 16 * 16),
                nn.GELU(),
            )
            self.prior_up_32 = self.up2_block(32)  # 16->32
            self.prior_up_64 = self.up2_block(32)  # 32->64
            self.prior_up_128 = self.up2_block(32)  # 64->128

            self.pc_gate = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=True),
                nn.GELU(),
                nn.Conv2d(3, 1, kernel_size=1, bias=True),
            )

            self.context_pixshuffle = nn.Sequential(
                nn.Conv2d(2048, 32 * 4 * 4, kernel_size=1, bias=True),  # 2048 -> 512
                nn.PixelShuffle(upscale_factor=4),                      # (512,32,32) -> (32,128,128)
            )

        
        ################### end of ablation options ###################

        # self.pc_gate = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True),
        #     nn.GELU(),
        #     nn.Conv2d(16, 256, kernel_size=1, bias=True),
        # )


        # # Method D: try direct overfit for debug, not good, but query only is already happening in the A which the network relies on query only
        # self.query_to_prior16 = nn.Sequential(
        #     nn.Linear(2048, 512),
        #     nn.GELU(),
        #     nn.Linear(512, 32*16*16),
        # )
        # self.prior_up_32 = self.up2_block(32)  # 16->32
        # self.prior_up_64 = self.up2_block(32)  # 32->64
        # self.prior_up_128 = self.up2_block(32)  # 64->128
        # self.affordance_head_dbg = nn.Sequential(
        #     nn.Conv2d(32, 16, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 1, kernel_size=1),
        # )

        # # Method E: mask2former
        # self.obs_proj = nn.Conv2d(self.proj_out_channels + 2,
        #     self.proj_out_channels,
        #     kernel_size=1,
        #     bias=True)
        # self.mask_query = nn.Parameter(torch.randn(1, 1, 256))
        # pos = self.build_ego_sincos_pos() # (Dx*Dy, 256)
        # self.register_buffer("positional_encoding", pos, persistent=False)
        # self.query_to_attn = nn.Linear(2048, 256)
        # self.token_self_attn = Attention(dim=256, num_heads=4, qkv_bias=True)
        # self.layer_norm_1 = nn.LayerNorm(256)
        # self.bev_cross_attn = torch.nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        # self.layer_norm_2 = nn.LayerNorm(256)
        # self.token_mlp = Mlp(in_features=256, hidden_features=512, out_features=256)
        # self.layer_norm_3 = nn.LayerNorm(256)

        # # Method F: cat before resnet but using geom only & G: first train without bev then refine
        # self.obs_proj = nn.Conv2d(self.proj_out_channels + 2,
        #     self.proj_out_channels,
        #     kernel_size=1,
        #     bias=True)
        # self.query_to_prior = nn.Sequential(
        #     nn.Linear(2048, 128 * 4 * 4),
        #     nn.GELU(),
        # )
        # self.prior_upsampler = nn.Sequential(
        #     self.up2_block(128),
        #     self.up2_block(128),
        #     self.up2_block(128),
        # )
        # # self.prior_proj = nn.Conv2d(256, 256, kernel_size=1, bias=True)
        # self.prior_proj = nn.Identity()
        # # self.fuse_prior = nn.Sequential(
        # #     nn.Conv2d(bev_head_channels + self.prior_channels, bev_head_channels, kernel_size=3, padding=1, bias=False),
        # #     LN2d(bev_head_channels),
        # #     nn.ReLU(inplace=True),
        # # # )
        # self.bev_encoder_2 = ResNetBEV(cin=4, base=4, depth=10, norm='ln')
        # self.bev_fpn_2 = FPNBEV(in_channels_list=[4, 8, 16, 32], fpn_dim=32)
        # self.bevfpn_fuse_2 = nn.Sequential(
        #     nn.Conv2d(32, 16, 3, padding=1, bias=False),
        #     # nn.BatchNorm2d(bev_head_channels),
        #     LN2d(16),
        #     nn.ReLU(inplace=True),
        # )
        # self.affordance_head_2 = nn.Sequential(
        #     nn.Conv2d(16, 8, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True), 
        #     nn.Conv2d(8, 1, kernel_size=1),
        # )

        # # Method I: geom only + film
        # self.bev_encoder_2 = ResNetBEV(cin=3, base=4, depth=10, norm='ln')
        # self.bev_fpn_2 = FPNBEV(in_channels_list=[4, 8, 16, 32], fpn_dim=32)
        # self.bevfpn_fuse_2 = nn.Sequential(
        #     nn.Conv2d(32, 16, 3, padding=1, bias=False),
        #     # nn.BatchNorm2d(bev_head_channels),
        #     LN2d(16),
        #     nn.ReLU(inplace=True),
        # )
        # self.affordance_head_2 = nn.Sequential(
        #     nn.Conv2d(16, 8, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True), 
        #     nn.Conv2d(8, 1, kernel_size=1),
        # )
        # self.obs_proj = nn.Conv2d(self.proj_out_channels + 2,
        #     self.proj_out_channels,
        #     kernel_size=1,
        #     bias=True)
        # self.query_to_prior16 = nn.Sequential(
        #     nn.Linear(2048, 32 * 16 * 16),
        #     nn.GELU(),
        # )
        # self.prior_up_32 = self.up2_block(32)  # 16->32
        # self.prior_up_64 = self.up2_block(32)  # 32->64
        # self.prior_up_128 = self.up2_block(32)  # 64->128
        # # For each level, turn prior feature into gamma/beta maps matching that level's channel count.
        # # (B, prior_ch, H, W) -> (B, 2*C_level, H, W) -> split into gamma/beta.
        # self.film2 = nn.Conv2d(32, 2 * 4, kernel_size=1, bias=True)   # for c2
        # self.film3 = nn.Conv2d(32, 2 * 8, kernel_size=1, bias=True)   # for c3
        # self.film4 = nn.Conv2d(32, 2 * 16, kernel_size=1, bias=True)   # for c4
        # self.film5 = nn.Conv2d(32, 2 * 32, kernel_size=1, bias=True)  # for c5
        # # Norms before FiLM (diffusion standard: norm then scale/shift)
        # self.norm2 = LN2d(4)
        # self.norm3 = LN2d(8)
        # self.norm4 = LN2d(16)
        # self.norm5 = LN2d(32)

        # # Method X: geom only + mask2former
        # self.obs_proj = nn.Conv2d(self.proj_out_channels + 2,
        #     self.proj_out_channels,
        #     kernel_size=1,
        #     bias=True)
        # self.mask_query = nn.Parameter(torch.randn(1, 1, 256))
        # pos = self.build_ego_sincos_pos() # (Dx*Dy, 256)
        # self.register_buffer("positional_encoding", pos, persistent=False)
        # self.query_to_attn = nn.Linear(2048, 256)
        # self.token_self_attn = Attention(dim=256, num_heads=4, qkv_bias=True)
        # self.layer_norm_1 = nn.LayerNorm(256)
        # self.bev_cross_attn = torch.nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        # self.layer_norm_2 = nn.LayerNorm(256)
        # self.token_mlp = Mlp(in_features=256, hidden_features=512, out_features=256)
        # self.layer_norm_3 = nn.LayerNorm(256)

        # placeholder for weight init
        self.init_weights()

        self.debug_cnt = 0
        self.latest_lss_for_debug = None

    def geom_forward(self, x):  # (B,3,128,128)
        m = self.geom
        e1 = m["e1"](x)
        z = m["mid"](m["d1"](e1))
        z = m["u1"](z)
        return m["out"](z)  # (B,1,128,128)
        
    def init_weights(self):
        # FiLM: start as identity (gamma=beta=0)
        if hasattr(self, "film_mlp"):
            last = None
            if isinstance(self.film_mlp, nn.Sequential) and len(self.film_mlp) > 0:
                last = self.film_mlp[-1]
            elif isinstance(self.film_mlp, nn.Linear):
                last = self.film_mlp

            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                if last.bias is not None:
                    nn.init.zeros_(last.bias)
        if hasattr(self, "film2"):
            for m in [self.film2, self.film3, self.film4, self.film5]:
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # ---- obs_proj: start as "copy original C, ignore +2 masks" ----
        if hasattr(self, "obs_proj") and self.obs_proj is not None:
            # obs_proj is Conv2d(C+2 -> C, k=1)
            with torch.no_grad():
                self.obs_proj.weight.zero_()
                if self.obs_proj.bias is not None:
                    self.obs_proj.bias.zero_()

                # identity mapping for the first C input channels
                out_c = self.obs_proj.out_channels
                in_c = self.obs_proj.in_channels
                C = out_c  # we want output C == original feature C

                # Only if shapes match what we expect: in_c == C+2 and out_c == C
                if in_c >= C:
                    for i in range(C):
                        self.obs_proj.weight[i, i, 0, 0] = 1.0

        if hasattr(self, "obs_proj_bevfusion") and self.obs_proj_bevfusion is not None:
            with torch.no_grad():
                self.obs_proj_bevfusion.weight.zero_()
                if self.obs_proj_bevfusion.bias is not None:
                    self.obs_proj_bevfusion.bias.zero_()

                out_c = self.obs_proj_bevfusion.out_channels
                in_c = self.obs_proj_bevfusion.in_channels

                # keep "old behavior": map first out_c input channels as identity
                for i in range(min(out_c, in_c)):
                    self.obs_proj_bevfusion.weight[i, i, 0, 0] = 1.0

        if hasattr(self, "cross_ffn"): # for ablation 3
            # Stable residual start: make FFN residual branch initially ~0
            for block in self.cross_ffn:
                if isinstance(block, torch.nn.Sequential) and len(block) > 0:
                    last = block[-1]
                    if isinstance(last, torch.nn.Linear):
                        torch.nn.init.zeros_(last.weight)
                        if last.bias is not None:
                            torch.nn.init.zeros_(last.bias)
        
        if hasattr(self, "msdeform_ffn"):
            for module in self.msdeform_ffn:
                if isinstance(module, nn.Sequential) and len(module) > 0:
                    last = module[-1]
                    if isinstance(last, nn.Linear):
                        nn.init.zeros_(last.weight)
                        if last.bias is not None:
                            nn.init.zeros_(last.bias)
        
        if hasattr(self, "anchor_offset_head"):
            last = self.anchor_offset_head[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                if last.bias is not None:
                    nn.init.zeros_(last.bias)
        
        if hasattr(self, 'lang_film'):
            nn.init.zeros_(self.lang_film[0].weight)
            nn.init.zeros_(self.lang_film[0].bias)

        if hasattr(self, 'context_pixshuffle'):
            last = self.context_pixshuffle[0]
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)


    def py_sigmoid_focal_loss(self,
                              pred,
                              target,
                              gamma=2.0,
                              alpha=0.25,
                              reduction='mean'):
        """
        pred: (B,H,W) raw logits
        target: (B,H,W) binary {0,1}
        gamma=2.0, alpha=0.25 are common default values from the original focal loss paper
        reduction: 'none' | 'mean' | 'sum'
        """
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight

        if reduction == 'mean':
            return loss.mean()
        if reduction == 'sum':
            return loss.sum()
        return loss

    def up2_block(self, ch):
        # nearest upsample + 3x3 conv + LN + ReLU
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
            LN2d(ch),
            nn.ReLU(inplace=True),
        )
    
    def build_ego_sincos_pos(self, Dx=128, Dy=128, dim=256, ego=63.0, sx=0.1, sy=0.1, device="cpu"):
        assert dim % 4 == 0, "dim must be divisible by 4 for 2D sincos (x and y)."
        half = dim // 2
        quarter = half // 2  # per axis (sin+cos)

        # grid indices
        i = torch.arange(Dx, device=device).float()  # forward
        j = torch.arange(Dy, device=device).float()  # left
        ii, jj = torch.meshgrid(i, j, indexing="ij")  # (Dx,Dy)

        x = (ii - ego) * sx   # forward meters
        y = (jj - ego) * sy   # left meters

        # frequencies
        omega = torch.arange(quarter, device=device).float()
        omega = 1.0 / (10000 ** (omega / quarter))  # (quarter,)

        # (Dx,Dy,quarter)
        x_out = x[..., None] * omega
        y_out = y[..., None] * omega

        pos_x = torch.cat([torch.sin(x_out), torch.cos(x_out)], dim=-1)  # (Dx,Dy,half)
        pos_y = torch.cat([torch.sin(y_out), torch.cos(y_out)], dim=-1)  # (Dx,Dy,half)

        pos = torch.cat([pos_x, pos_y], dim=-1)  # (Dx,Dy,dim)
        pos = pos.view(Dx * Dy, dim)             # (Dx*Dy, dim)
        return pos
    
    def extract_xy_from_affordance(
        self,
        affordance_logits: torch.Tensor,
        visible_mask=None,
        traversable_mask=None,
    ) -> torch.Tensor:
        if affordance_logits.dim() == 2:
            logits = affordance_logits.unsqueeze(0)  # (1,H,W)
        elif affordance_logits.dim() == 3:
            logits = affordance_logits  # (K,H,W)
        else:
            raise ValueError(f"Unsupported affordance_logits dim={affordance_logits.dim()}, expected 2 or 3.")

        K, H, W = logits.shape
        logits = logits.clone()

        if visible_mask is not None:
            vm = visible_mask
            if isinstance(vm, torch.Tensor):
                vm = vm.to(device=logits.device)
            else:
                vm = torch.as_tensor(vm, device=logits.device)
            if vm.dim() == 2:
                vm = vm.unsqueeze(0).expand(K, -1, -1)
            logits[vm == 0] = -float("inf")

        if traversable_mask is not None:
            tm = traversable_mask
            if isinstance(tm, torch.Tensor):
                tm = tm.to(device=logits.device)
            else:
                tm = torch.as_tensor(tm, device=logits.device)
            if tm.dim() == 2:
                tm = tm.unsqueeze(0).expand(K, -1, -1)
            logits[tm == 0] = -float("inf")

        idx = torch.argmax(logits.view(K, -1), dim=1).long()  # (K,)
        gx = idx // W
        gy = idx % W

        x_min, x_max, x_step = self.grid_config["x"]
        y_min, y_max, y_step = self.grid_config["y"]

        x = x_min + (gx.to(logits.dtype) + 0.5) * x_step
        y = y_min + (gy.to(logits.dtype) + 0.5) * y_step
        return torch.stack([x, y], dim=1)
    
    def debug_vis(self, data, afford_logits, pred_xy, exit=False,
                out_dir="./debug_outputs"):
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        import torch

        os.makedirs(out_dir, exist_ok=True)

        # --- masks ---
        A = (data["traversable_mask"][0].detach().cpu().numpy() > 0.5)
        B = (data["visible_mask"][0].detach().cpu().numpy() > 0.5)
        C = (data["affordance_mask"][0].detach().cpu().numpy() > 0.5)

        # --- pred binary map ---
        prob = torch.sigmoid(afford_logits[0].detach())
        pred_bin = (prob > 0.5).float().cpu().numpy()

        # --- your base grey system (uint8) ---
        base = np.zeros_like(A, dtype=np.uint8)
        base[B & A] = 255
        base[B & (~A)] = 0
        base[(~B) & A] = 180
        base[(~B) & (~A)] = 90
        bev_rgb = np.repeat(base[..., None], 3, axis=2)

        green_known_free = np.array([0, 255, 0], dtype=np.uint8)
        green_unknown_free = np.array([80, 160, 80], dtype=np.uint8)
        green_known_occ = np.array([0, 48, 0], dtype=np.uint8)
        green_unknown_occ = np.array([40, 80, 40], dtype=np.uint8)

        bev_rgb[C & B & A] = green_known_free
        bev_rgb[C & B & (~A)] = green_known_occ
        bev_rgb[C & (~B) & A] = green_unknown_free
        bev_rgb[C & (~B) & (~A)] = green_unknown_occ

        # --- pred point meters -> grid coords (edge=-6.4, step=0.1, center offset=0.05) ---
        x = float(pred_xy[0].detach().cpu().item())
        y = float(pred_xy[1].detach().cpu().item())
        gx = (x - (-6.4 + 0.05)) / 0.1
        gy = (y - (-6.4 + 0.05)) / 0.1

        # --- top 4 images ---
        imgs = data["raw_img"][0].detach().cpu().numpy()        # (4,H,W,3) uint8 BGR
        imgs = imgs.transpose(0, 3, 1, 2)                       # (4,3,H,W)
        imgs = imgs[::-1]
        imgs = np.array([imgs[2], imgs[3], imgs[0], imgs[1]])   # front left back right
        imgs = imgs.transpose(0, 2, 3, 1)[:, :, :, ::-1]         # RGB

        # --- LSS BEV feature visualization (nonzero mask) ---
        lss = getattr(self, "latest_lss_for_debug", None)
        lss_img = None
        ab_overlay = None  # new: A/B overlay for pred panel
        if lss is not None:
            bf = lss[0].detach().cpu().numpy()                  # (C_or_Cplus2,Dx,Dy)
            bf_feat = bf[:-2]                           # drop (ground_free, cam_free)

            # existing nonzero viz (same)
            nonzero = (np.abs(bf_feat).sum(axis=0) > 0).astype(np.float32)  # (Dx,Dy) 0/1
            lss_img = nonzero

            # new: if last two channels are (ground_free, cam_free), make overlay
            if bf.shape[0] >= (getattr(self, "proj_out_channels", 0) + 2):
                A_free = (bf[-2] > 0.5)  # ground free, (Dx,Dy)
                B_free = (bf[-1] > 0.5)  # cam free, (Dx,Dy)

                # Colors:
                # A & B -> light grey
                # ~A & B -> dark grey
                # ~A & ~B -> black
                # A & ~B -> yellow
                ab = np.zeros((A_free.shape[0], A_free.shape[1], 3), dtype=np.uint8)
                light_grey = np.array([200, 200, 200], dtype=np.uint8)
                dark_grey  = np.array([80, 80, 80], dtype=np.uint8)
                black      = np.array([0, 0, 0], dtype=np.uint8)
                yellow     = np.array([255, 255, 0], dtype=np.uint8)

                ab[(A_free) & (B_free)]  = light_grey
                ab[(~A_free) & (B_free)] = dark_grey
                ab[(~A_free) & (~B_free)] = black
                ab[(A_free) & (~B_free)] = yellow

                ab_overlay = ab

        token = data.get("token", "na")
        token = token[0] if isinstance(token, (list, tuple)) else token
        save_path = os.path.join(out_dir, f"aff_debug_{token}.jpg")

        # --- layout: 4 imgs on top, 3 panels bottom ---
        fig = plt.figure(figsize=(12, 7))
        fig.suptitle(data["instruction"][0], fontsize=9, y=0.98)
        gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 3.0], wspace=0.02, hspace=0.06)

        for k in range(4):
            ax = fig.add_subplot(gs[0, k:k+1])
            ax.imshow(imgs[k], aspect="auto")
            ax.set_aspect("auto")
            ax.axis("off")

        # bottom-left (cols 0-1): pred (+ optional A/B overlay)
        ax1 = fig.add_subplot(gs[1, 0:2])
        ax1.set_title(f"Pred (>0.5)  @ {x:.2f}, {y:.2f}", fontsize=9)

        if ab_overlay is not None:
            ax1.imshow(ab_overlay)  # RGB underlay
            ax1.imshow(pred_bin, cmap="gray", vmin=0, vmax=1, alpha=0.5)  # overlay 0.5 alpha
        else:
            ax1.imshow(pred_bin, cmap="gray", vmin=0, vmax=1)

        ax1.scatter(gy, gx, c="red", s=50, marker="x")

        # bottom-middle (cols 2-3): LSS
        ax2 = fig.add_subplot(gs[1, 2:4])
        ax2.set_title("LSS BEV (nonzero feats)", fontsize=9)
        if lss_img is None:
            ax2.text(0.5, 0.5, "latest_lss_for_debug is None", ha="center", va="center")
            ax2.set_axis_off()
        else:
            ax2.imshow(lss_img, cmap="gray", vmin=0, vmax=1)
            ax2.scatter(gy, gx, c="red", s=50, marker="x")

        # bottom-right (cols 4-5): colored GT overlay
        ax3 = fig.add_subplot(gs[1, 4:6])
        ax3.set_title("BEV (grey base + GT afford green)", fontsize=9)
        ax3.imshow(bev_rgb)
        ax3.scatter(gy, gx, c="red", s=50, marker="x")

        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[Debug] Saved affordance visualization to {save_path}")

        if exit:
            raise SystemExit(0)

    def xy_to_bev_float(self, xy):  # xy: (B,2) metric
        x_min, x_max, x_step = self.grid_config["x"]
        y_min, y_max, y_step = self.grid_config["y"]
        gx = (xy[:, 0] - x_min) / x_step  # float grid coord
        gy = (xy[:, 1] - y_min) / y_step
        return torch.stack([gx, gy], dim=-1)  # (B,2), float

    def prepare_bev_feats(
        self,
        depth,            # (B, N, H, W) depth in meters
        img_feats,        # (B, N, fH, fW, C)
        sensor2egos,      # (B, N, 4, 4)
        intrinsics,       # (B, N, 3, 3)
    ):
        """
        Kept very close to your draft. Only differences:
        - img_feats is assumed already at fH=fW=H/ps
        - enforce output BEV layout: (B, C, Dx, Dy) (never Dy then Dx downstream)
        """
        B, N, H, W = depth.shape
        # ps = self.downsample
        # fH, fW = H // ps, W // ps

        # assert img_feats.shape[2] == fH and img_feats.shape[3] == fW, \
        #     f"img_feats shape {img_feats.shape} not compatible with depth shape {depth.shape} and downsample {ps}"

        # # patch-mean depth (B, N, fH, fW)
        # depth_tiles = F.unfold(
        #     depth.flatten(0, 1).unsqueeze(1),
        #     kernel_size=ps,
        #     stride=ps
        # )
        # depth_tiles = depth_tiles.view(B * N, ps * ps, fH, fW)
        # depth_coarse = depth_tiles.mean(dim=1).view(B, N, fH, fW)

        ps = 1
        fH, fW = H, W
        assert img_feats.shape[2] == fH and img_feats.shape[3] == fW, \
            f"img_feats shape {img_feats.shape} not compatible with depth shape {depth.shape} and per-pixel LSS"
        depth_coarse = depth  # per-pixel depth

        # one-hot depth bins
        dmin, dmax, dstep = self.grid_config["depth"]
        num_bins = int(round((dmax - dmin) / dstep))
        depth_clamped = depth_coarse.clamp(min=dmin, max=dmax - 1e-4)
        bin_idx = torch.div(depth_clamped - dmin, dstep, rounding_mode="floor").long()
        bin_idx = bin_idx.clamp(0, num_bins - 1)
        depth_probs = F.one_hot(bin_idx, num_classes=num_bins)
        depth_probs = depth_probs.permute(0, 1, 4, 2, 3).float().contiguous()  # (B, N, D, fH, fW)

        depth_probs_bn = depth_probs.view(B * N, num_bins, fH, fW)
        C = img_feats.shape[4]
        img_feats_bn = img_feats.permute(0, 1, 4, 2, 3).contiguous().view(B * N, C, fH, fW)

        bev = lss_bev_pool(
            depth_probs_bn,
            img_feats_bn,
            sensor2egos,
            intrinsics,
            grid_config=self.grid_config,
            input_size=self.input_size,
            downsample=ps,
            collapse_z=self.collapse_z,
            pooling='avg',
        )  # (B, C*Dz, Dy, Dx) when collapse_z=True and Dz==1 -> (B, C, Dy, Dx)

        # Enforce downstream convention: always (Dx, Dy) spatial ordering.
        # lss_bev_pool returns (B, C, Dy, Dx) -> convert to (B, C, Dx, Dy)
        bev = bev.permute(0, 1, 3, 2).contiguous()
        return bev  # (B, C, Dx, Dy)

    def prepare_bev_feats_deterministic(
        self,
        depth,            # (B, N, H, W) meters
        img_feats,        # (B, N, Hf, Wf, C)
        sensor2egos,      # (B, N, 4, 4)
        intrinsics,       # (B, N, 3, 3)
    ):
        B, N, Hd, Wd = depth.shape
        Hf, Wf = img_feats.shape[2], img_feats.shape[3]
        device = depth.device
        feat_dtype = img_feats.dtype

        if (Hd != Hf) or (Wd != Wf):
            raise NotImplementedError(
                f"Depth/img_feats resolution mismatch: depth {Hd}x{Wd}, feats {Hf}x{Wf}."
            )

        # Grid config
        x_min, x_max, x_step = self.grid_config["x"]
        y_min, y_max, y_step = self.grid_config["y"]
        z_min, z_max, z_step = self.grid_config["z"]
        Dx, Dy, Dz = int(self.Dx), int(self.Dy), int(self.Dz)

        if Dz <= 0 or Dx <= 0 or Dy <= 0:
            ratio_z = (float(z_max) - float(z_min)) / float(z_step)
            raise RuntimeError(
                f"Invalid grid sizes Dx/Dy/Dz=({Dx},{Dy},{Dz}); z_cfg={self.grid_config['z']}, ratio_z={ratio_z}"
            )

        # Geometry in float32
        with autocast(enabled=False):
            u = torch.linspace(0, Wf - 1, Wf, device=device, dtype=torch.float32)
            v = torch.linspace(0, Hf - 1, Hf, device=device, dtype=torch.float32)
            v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")  # (Hf, Wf)
            u_grid = u_grid[None, None]
            v_grid = v_grid[None, None]

            fx = intrinsics[:, :, 0, 0].float()[..., None, None]
            fy = intrinsics[:, :, 1, 1].float()[..., None, None]
            cx = intrinsics[:, :, 0, 2].float()[..., None, None]
            cy = intrinsics[:, :, 1, 2].float()[..., None, None]

            z = depth.float()
            x = (u_grid - cx) / fx * z
            y = (v_grid - cy) / fy * z

            R = sensor2egos[:, :, :3, :3].float()
            t = sensor2egos[:, :, :3, 3].float()[..., None, None]

            X = R[:, :, 0, 0][..., None, None] * x + R[:, :, 0, 1][..., None, None] * y + R[:, :, 0, 2][..., None, None] * z + t[:, :, 0]
            Y = R[:, :, 1, 0][..., None, None] * x + R[:, :, 1, 1][..., None, None] * y + R[:, :, 1, 2][..., None, None] * z + t[:, :, 1]
            Z = R[:, :, 2, 0][..., None, None] * x + R[:, :, 2, 1][..., None, None] * y + R[:, :, 2, 2][..., None, None] * z + t[:, :, 2]

            ix = torch.floor((X - float(x_min)) / float(x_step)).to(torch.int64)
            iy = torch.floor((Y - float(y_min)) / float(y_step)).to(torch.int64)
            iz = torch.floor((Z - float(z_min)) / float(z_step)).to(torch.int64)

            valid = (
                (ix >= 0) & (ix < Dx) &
                (iy >= 0) & (iy < Dy) &
                (iz >= 0) & (iz < Dz) &
                torch.isfinite(z) &
                torch.isfinite(X) & torch.isfinite(Y) & torch.isfinite(Z)
            )

        # Flatten
        feat_flat = img_feats.reshape(-1, img_feats.shape[-1])  # (B*N*Hf*Wf, C)
        ix = ix.reshape(-1)
        iy = iy.reshape(-1)
        iz = iz.reshape(-1)
        valid = valid.reshape(-1)

        batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, N, Hf, Wf).reshape(-1)
        lin = batch_idx * (Dz * Dy * Dx) + iz * (Dy * Dx) + iy * Dx + ix

        lin = lin[valid]
        feat = feat_flat[valid]

        # Scatter accumulate in fp32 (stable), then cast back
        C = feat.shape[1]
        bev_sum = torch.zeros((B * Dz * Dy * Dx, C), device=device, dtype=torch.float32)
        cnt = torch.zeros((B * Dz * Dy * Dx, 1), device=device, dtype=torch.float32)

        bev_sum.index_add_(0, lin, feat.float())
        cnt.index_add_(0, lin, torch.ones((feat.shape[0], 1), device=device, dtype=torch.float32))

        # Reshape to (B, C, Dz, Dy, Dx)
        bev_sum = bev_sum.view(B, Dz, Dy, Dx, C).permute(0, 4, 1, 2, 3).contiguous()
        cnt = cnt.view(B, Dz, Dy, Dx, 1).permute(0, 4, 1, 2, 3).contiguous().clamp_min(1.0)

        bev = (bev_sum / cnt).to(feat_dtype)  # (B, C, Dz, Dy, Dx)

        # Collapse Z
        if self.collapse_z:
            if Dz == 1:
                bev = bev.squeeze(2)  # (B, C, Dy, Dx)
            else:
                bev = torch.cat(bev.unbind(dim=2), dim=1)  # (B, C*Dz, Dy, Dx)

            # RETURN MUST BE (B, C, Dx, Dy)
            bev = bev.permute(0, 1, 3, 2).contiguous()  # (B, C*, Dx, Dy)
            return bev

        # If not collapse_z: RETURN (B, C, Dz, Dx, Dy)
        bev = bev.permute(0, 1, 2, 4, 3).contiguous()
        return bev

    def prepare_bev_feats_deterministic_with_obs(
        self,
        depth,            # (B, N, H, W) meters
        img_feats,        # (B, N, H, W, C)  (must match depth resolution)
        sensor2egos,      # (B, N, 4, 4)
        intrinsics,       # (B, N, 3, 3)
        ground_z_range=(-1.6, -1.4),
        cam_z_range=(-0.1, 0.1),
        blocker_z_range=None,
    ):
        """
        Returns:
        bev_cat: (B, C+2, Dx, Dy)
            - first C channels are EXACTLY the same as your previous prepare_bev_feats_deterministic() output
            - +1 channel: free_ground_obs
            - +1 channel: free_cam_obs
        """

        # ---------------------------
        # helpers: supercover + raycast (CPU numpy, no grad)
        # ---------------------------
        def supercover_line(x0, y0, x1, y1):
            x0 = int(x0); y0 = int(y0); x1 = int(x1); y1 = int(y1)
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x1 >= x0 else -1
            sy = 1 if y1 >= y0 else -1

            x, y = x0, y0
            err = dx - dy
            yield (x, y)

            while x != x1 or y != y1:
                e2 = 2 * err
                step_x = False
                step_y = False
                px, py = x, y

                if e2 > -dy:
                    err -= dy
                    x += sx
                    step_x = True
                if e2 < dx:
                    err += dx
                    y += sy
                    step_y = True

                if step_x and step_y:
                    yield (px + sx, py)
                    yield (px, py + sy)

                yield (x, y)

        def compute_visible_mask_from_occ2d(occ2d_hw, origin_yx):
            # occ2d_hw: (H,W) bool
            H, W = occ2d_hw.shape
            oy, ox = int(origin_yx[0]), int(origin_yx[1])
            visible = np.zeros((H, W), dtype=bool)

            targets = []
            for ix in range(W):
                targets.append((ix, 0))
                targets.append((ix, H - 1))
            for iy in range(1, H - 1):
                targets.append((0, iy))
                targets.append((W - 1, iy))

            def cast_to(tx, ty):
                for (x, y) in supercover_line(ox, oy, tx, ty):
                    if not (0 <= x < W and 0 <= y < H):
                        break
                    visible[y, x] = True
                    if occ2d_hw[y, x]:
                        break

            for (tx, ty) in targets:
                dx = tx - ox
                dy = ty - oy
                cast_to(tx, ty)

                # diagonal robustness (same as your code)
                if dx != 0 and dy != 0 and abs(dx) == abs(dy):
                    candidates = []
                    if tx == 0 or tx == W - 1:
                        if 0 <= ty - 1 < H: candidates.append((tx, ty - 1))
                        if 0 <= ty + 1 < H: candidates.append((tx, ty + 1))
                    if ty == 0 or ty == H - 1:
                        if 0 <= tx - 1 < W: candidates.append((tx - 1, ty))
                        if 0 <= tx + 1 < W: candidates.append((tx + 1, ty))
                    for (nx, ny) in candidates[:2]:
                        cast_to(nx, ny)

            return visible

        # ---------------------------
        # sanity / shapes
        # ---------------------------
        B, N, Hd, Wd = depth.shape
        Hf, Wf = img_feats.shape[2], img_feats.shape[3]
        device = depth.device
        feat_dtype = img_feats.dtype

        if (Hd != Hf) or (Wd != Wf):
            raise NotImplementedError(
                f"Depth/img_feats resolution mismatch: depth {Hd}x{Wd}, feats {Hf}x{Wf}."
            )

        x_min, x_max, x_step = self.grid_config["x"]
        y_min, y_max, y_step = self.grid_config["y"]
        z_min, z_max, z_step = self.grid_config["z"]
        Dx, Dy, Dz = int(self.Dx), int(self.Dy), int(self.Dz)
        assert Dx > 0 and Dy > 0 and Dz > 0

        # ---------------------------
        # same geometry as your deterministic BEV
        # ---------------------------
        with autocast(enabled=False):
            u = torch.linspace(0, Wf - 1, Wf, device=device, dtype=torch.float32)
            v = torch.linspace(0, Hf - 1, Hf, device=device, dtype=torch.float32)
            v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")  # (H, W)
            u_grid = u_grid[None, None]
            v_grid = v_grid[None, None]

            fx = intrinsics[:, :, 0, 0].float()[..., None, None]
            fy = intrinsics[:, :, 1, 1].float()[..., None, None]
            cx = intrinsics[:, :, 0, 2].float()[..., None, None]
            cy = intrinsics[:, :, 1, 2].float()[..., None, None]

            z = depth.float()
            x = (u_grid - cx) / fx * z
            y = (v_grid - cy) / fy * z

            R = sensor2egos[:, :, :3, :3].float()
            t = sensor2egos[:, :, :3, 3].float()[..., None, None]

            X = R[:, :, 0, 0][..., None, None] * x + R[:, :, 0, 1][..., None, None] * y + R[:, :, 0, 2][..., None, None] * z + t[:, :, 0]
            Y = R[:, :, 1, 0][..., None, None] * x + R[:, :, 1, 1][..., None, None] * y + R[:, :, 1, 2][..., None, None] * z + t[:, :, 1]
            Z = R[:, :, 2, 0][..., None, None] * x + R[:, :, 2, 1][..., None, None] * y + R[:, :, 2, 2][..., None, None] * z + t[:, :, 2]

            ix = torch.floor((X - float(x_min)) / float(x_step)).to(torch.int64)
            iy = torch.floor((Y - float(y_min)) / float(y_step)).to(torch.int64)
            iz = torch.floor((Z - float(z_min)) / float(z_step)).to(torch.int64)

            valid_main = (
                (ix >= 0) & (ix < Dx) &
                (iy >= 0) & (iy < Dy) &
                (iz >= 0) & (iz < Dz) &
                torch.isfinite(z) &
                torch.isfinite(X) & torch.isfinite(Y) & torch.isfinite(Z)
            )

            # band filters (metric Z), still require in-bounds ix/iy and finite
            valid_xy = (
                (ix >= 0) & (ix < Dx) &
                (iy >= 0) & (iy < Dy) &
                torch.isfinite(z) &
                torch.isfinite(X) & torch.isfinite(Y) & torch.isfinite(Z)
            )
            zg0, zg1 = float(ground_z_range[0]), float(ground_z_range[1])
            zc0, zc1 = float(cam_z_range[0]), float(cam_z_range[1])
            valid_ground = valid_xy & (Z >= zg0) & (Z < zg1)
            valid_cam    = valid_xy & (Z >= zc0) & (Z < zc1)

            if blocker_z_range is not None:
                zb0, zb1 = float(blocker_z_range[0]), float(blocker_z_range[1])
                valid_block_any = valid_xy & (Z >= zb0) & (Z < zb1)


        # ---------------------------
        # flatten indices once
        # ---------------------------
        feat_flat = img_feats.reshape(-1, img_feats.shape[-1])  # (B*N*H*W, C)
        ix_f = ix.reshape(-1)
        iy_f = iy.reshape(-1)
        iz_f = iz.reshape(-1)

        valid_main_f   = valid_main.reshape(-1)
        valid_ground_f = valid_ground.reshape(-1)
        valid_cam_f    = valid_cam.reshape(-1)
        if blocker_z_range is not None:
            valid_block_any_f = valid_block_any.reshape(-1)

        batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, N, Hf, Wf).reshape(-1)

        # ---------------------------
        # (1) MAIN BEV FEATURES: identical to your previous function
        # ---------------------------
        lin_main = batch_idx * (Dz * Dy * Dx) + iz_f * (Dy * Dx) + iy_f * Dx + ix_f
        lin_main = lin_main[valid_main_f]
        feat_main = feat_flat[valid_main_f]

        C = feat_main.shape[1]
        bev_sum = torch.zeros((B * Dz * Dy * Dx, C), device=device, dtype=torch.float32)
        cnt = torch.zeros((B * Dz * Dy * Dx, 1), device=device, dtype=torch.float32)

        bev_sum.index_add_(0, lin_main, feat_main.float())
        cnt.index_add_(0, lin_main, torch.ones((feat_main.shape[0], 1), device=device, dtype=torch.float32))

        bev_sum = bev_sum.view(B, Dz, Dy, Dx, C).permute(0, 4, 1, 2, 3).contiguous()  # (B,C,Dz,Dy,Dx)
        cnt = cnt.view(B, Dz, Dy, Dx, 1).permute(0, 4, 1, 2, 3).contiguous().clamp_min(1.0)

        bev = (bev_sum / cnt).to(feat_dtype)  # (B,C,Dz,Dy,Dx)

        if self.collapse_z:
            if Dz == 1:
                bev_main = bev.squeeze(2)  # (B,C,Dy,Dx)
            else:
                bev_main = torch.cat(bev.unbind(dim=2), dim=1)  # (B,C*Dz,Dy,Dx)
            bev_main = bev_main.permute(0, 1, 3, 2).contiguous()  # (B,C*,Dx,Dy)
        else:
            bev_main = bev.permute(0, 1, 2, 4, 3).contiguous()  # (B,C,Dz,Dx,Dy) (not used by you)

        # ---------------------------
        # blocker grids from BEV space (your rule: nonzero => blocker)
        # ---------------------------
        # occ_any from main BEV (exactly your previous result)
        occ_any = (bev_main.detach().abs().sum(dim=1) > 0)   # (B,Dx,Dy) bool

        # ground evidence: scatter hits from Z in ground band into (ix,iy)
        # (we only need a mask, not features)
        lin2d = batch_idx * (Dy * Dx) + iy_f * Dx + ix_f

        def scatter2d_hit(valid_f):
            lin_v = lin2d[valid_f]
            tmp = torch.zeros((B * Dy * Dx,), device=device, dtype=torch.float32)
            tmp.index_add_(0, lin_v, torch.ones((lin_v.shape[0],), device=device, dtype=torch.float32))
            hit = (tmp.view(B, Dy, Dx) > 0)  # (B,Dy,Dx)
            return hit.permute(0, 2, 1).contiguous()  # (B,Dx,Dy)

        # ground_evidence = scatter2d_hit(valid_ground_f)  # (B,Dx,Dy)
        occ_cam = scatter2d_hit(valid_cam_f)             # (B,Dx,Dy)  (cam-height blockers)

        ground_hit = scatter2d_hit(valid_ground_f)

        if blocker_z_range is None:
            ground_evidence = ground_hit
        else:
            block_any = scatter2d_hit(valid_block_any_f) # (B,Dx,Dy)
            ground_evidence = ground_hit & (~block_any)

        # ---------------------------
        # raycast in BEV space (CPU numpy) using occ_any and occ_cam
        # ---------------------------
        # origin at ego (0,0) in grid indices
        ox = int(round((0.0 - float(x_min)) / float(x_step)))
        oy = int(round((0.0 - float(y_min)) / float(y_step)))
        ox = max(0, min(Dx - 1, ox))
        oy = max(0, min(Dy - 1, oy))
        origin_yx = (oy, ox)  # numpy expects (y,x) in (H,W)=(Dy,Dx)

        free_any_list = []
        free_cam_list = []

        occ_any_hw = occ_any.detach().permute(0, 2, 1).cpu().numpy()  # (B,Dy,Dx)
        occ_cam_hw = occ_cam.detach().permute(0, 2, 1).cpu().numpy()  # (B,Dy,Dx)

        for b in range(B):
            vis_any = compute_visible_mask_from_occ2d(occ_any_hw[b], origin_yx)  # (Dy,Dx)
            free_any = (vis_any & (~occ_any_hw[b]))                              # (Dy,Dx)

            vis_cam = compute_visible_mask_from_occ2d(occ_cam_hw[b], origin_yx)  # (Dy,Dx)
            free_cam = (vis_cam & (~occ_cam_hw[b]))                              # (Dy,Dx)

            free_any_list.append(free_any)
            free_cam_list.append(free_cam)

        free_any = torch.from_numpy(np.stack(free_any_list, axis=0)).to(device=device)  # (B,Dy,Dx) bool
        free_cam = torch.from_numpy(np.stack(free_cam_list, axis=0)).to(device=device)  # (B,Dy,Dx) bool
        free_any = free_any.permute(0, 2, 1).contiguous()  # (B,Dx,Dy)
        free_cam = free_cam.permute(0, 2, 1).contiguous()  # (B,Dx,Dy)

        # ---------------------------
        # final channels (no hand-crafted merge beyond what you specified)
        # ---------------------------
        free_ground_obs = (ground_evidence | free_any)  # (B,Dx,Dy) bool
        free_cam_obs = free_cam                          # (B,Dx,Dy) bool

        free_ground_obs = free_ground_obs.to(dtype=bev_main.dtype).unsqueeze(1)  # (B,1,Dx,Dy)
        free_cam_obs    = free_cam_obs.to(dtype=bev_main.dtype).unsqueeze(1)     # (B,1,Dx,Dy)

        bev_cat = torch.cat([bev_main, free_ground_obs, free_cam_obs], dim=1)    # (B,C+2,Dx,Dy)
        return bev_cat

    # BUG: prepare_bev_feats_deterministic_with_obs does not have a separate block z range so the ground evidence can include points under a table
    # should have a separate z range for blockers from ground points, if they are any, don't make it ground evidence


    def prepare_bev_feats_deterministic_with_obs_v2(
        self,
        depth,            # (B, N, H, W) meters
        img_feats,        # (B, N, H, W, C)  (must match depth resolution)
        sensor2egos,      # (B, N, 4, 4)
        intrinsics,       # (B, N, 3, 3)
        ground_z_range=(-1.6, -1.4),
        mid_z_range=(-0.1, 0.1),
        tall_z_range=(0.25, 0.35),
        blocker_z_range=(-1.4, -0.1),  # you said grid z is (-1.6, 0.25)
    ):
        """
        Returns:
        bev_cat: (B, C+4, Dx, Dy)
            - first C channels: EXACTLY the same as your previous deterministic BEV output
            - +1 channel: free_ground_obs
            - +1 channel: cnt2d_log1p   (per-cell projected point count, log1p compressed)
            - +1 channel: maskA_free    (raycast free w.r.t wall-like blockers: tall_hit & ~mid_cast_free)
            - +1 channel: maskB_free    (raycast free w.r.t mid-only blockers: mid_hit & ~tall_hit)
        """


        # ---------------------------
        # helpers: supercover + raycast (CPU numpy, no grad)
        # ---------------------------
        def supercover_line(x0, y0, x1, y1):
            x0 = int(x0); y0 = int(y0); x1 = int(x1); y1 = int(y1)
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x1 >= x0 else -1
            sy = 1 if y1 >= y0 else -1

            x, y = x0, y0
            err = dx - dy
            yield (x, y)

            while x != x1 or y != y1:
                e2 = 2 * err
                step_x = False
                step_y = False
                px, py = x, y

                if e2 > -dy:
                    err -= dy
                    x += sx
                    step_x = True
                if e2 < dx:
                    err += dx
                    y += sy
                    step_y = True

                if step_x and step_y:
                    yield (px + sx, py)
                    yield (px, py + sy)

                yield (x, y)

        def compute_visible_mask_from_occ2d(occ2d_hw, origin_yx):
            # occ2d_hw: (H,W) bool
            H, W = occ2d_hw.shape
            oy, ox = int(origin_yx[0]), int(origin_yx[1])
            visible = np.zeros((H, W), dtype=bool)

            targets = []
            for ix in range(W):
                targets.append((ix, 0))
                targets.append((ix, H - 1))
            for iy in range(1, H - 1):
                targets.append((0, iy))
                targets.append((W - 1, iy))

            def cast_to(tx, ty):
                for (x, y) in supercover_line(ox, oy, tx, ty):
                    if not (0 <= x < W and 0 <= y < H):
                        break
                    visible[y, x] = True
                    if occ2d_hw[y, x]:
                        break

            for (tx, ty) in targets:
                dx = tx - ox
                dy = ty - oy
                cast_to(tx, ty)

                # diagonal robustness (same as your code)
                if dx != 0 and dy != 0 and abs(dx) == abs(dy):
                    candidates = []
                    if tx == 0 or tx == W - 1:
                        if 0 <= ty - 1 < H: candidates.append((tx, ty - 1))
                        if 0 <= ty + 1 < H: candidates.append((tx, ty + 1))
                    if ty == 0 or ty == H - 1:
                        if 0 <= tx - 1 < W: candidates.append((tx - 1, ty))
                        if 0 <= tx + 1 < W: candidates.append((tx + 1, ty))
                    for (nx, ny) in candidates[:2]:
                        cast_to(nx, ny)

            return visible

        # ---------------------------
        # sanity / shapes
        # ---------------------------
        B, N, Hd, Wd = depth.shape
        Hf, Wf = img_feats.shape[2], img_feats.shape[3]
        device = depth.device
        feat_dtype = img_feats.dtype

        if (Hd != Hf) or (Wd != Wf):
            raise NotImplementedError(
                f"Depth/img_feats resolution mismatch: depth {Hd}x{Wd}, feats {Hf}x{Wf}."
            )

        x_min, x_max, x_step = self.grid_config["x"]
        y_min, y_max, y_step = self.grid_config["y"]
        z_min, z_max, z_step = self.grid_config["z"]
        Dx, Dy, Dz = int(self.Dx), int(self.Dy), int(self.Dz)
        assert Dx > 0 and Dy > 0 and Dz > 0

        # ---------------------------
        # same geometry as your deterministic BEV
        # ---------------------------
        with autocast(enabled=False):
            u = torch.linspace(0, Wf - 1, Wf, device=device, dtype=torch.float32)
            v = torch.linspace(0, Hf - 1, Hf, device=device, dtype=torch.float32)
            v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")  # (H, W)
            u_grid = u_grid[None, None]
            v_grid = v_grid[None, None]

            fx = intrinsics[:, :, 0, 0].float()[..., None, None]
            fy = intrinsics[:, :, 1, 1].float()[..., None, None]
            cx = intrinsics[:, :, 0, 2].float()[..., None, None]
            cy = intrinsics[:, :, 1, 2].float()[..., None, None]

            z = depth.float()
            x = (u_grid - cx) / fx * z
            y = (v_grid - cy) / fy * z

            R = sensor2egos[:, :, :3, :3].float()
            t = sensor2egos[:, :, :3, 3].float()[..., None, None]

            X = R[:, :, 0, 0][..., None, None] * x + R[:, :, 0, 1][..., None, None] * y + R[:, :, 0, 2][..., None, None] * z + t[:, :, 0]
            Y = R[:, :, 1, 0][..., None, None] * x + R[:, :, 1, 1][..., None, None] * y + R[:, :, 1, 2][..., None, None] * z + t[:, :, 1]
            Z = R[:, :, 2, 0][..., None, None] * x + R[:, :, 2, 1][..., None, None] * y + R[:, :, 2, 2][..., None, None] * z + t[:, :, 2]

            ix = torch.floor((X - float(x_min)) / float(x_step)).to(torch.int64)
            iy = torch.floor((Y - float(y_min)) / float(y_step)).to(torch.int64)
            iz = torch.floor((Z - float(z_min)) / float(z_step)).to(torch.int64)

            valid_main = (
                (ix >= 0) & (ix < Dx) &
                (iy >= 0) & (iy < Dy) &
                (iz >= 0) & (iz < Dz) &
                torch.isfinite(z) &
                torch.isfinite(X) & torch.isfinite(Y) & torch.isfinite(Z)
            )

            # band filters (metric Z), still require in-bounds ix/iy and finite
            valid_xy = (
                (ix >= 0) & (ix < Dx) &
                (iy >= 0) & (iy < Dy) &
                torch.isfinite(z) &
                torch.isfinite(X) & torch.isfinite(Y) & torch.isfinite(Z)
            )
            zg0, zg1 = float(ground_z_range[0]), float(ground_z_range[1])
            zm0, zm1 = float(mid_z_range[0]), float(mid_z_range[1])
            zt0, zt1 = float(tall_z_range[0]), float(tall_z_range[1])

            valid_ground = valid_xy & (Z >= zg0) & (Z < zg1)
            valid_mid    = valid_xy & (Z >= zm0) & (Z < zm1)
            valid_tall   = valid_xy & (Z >= zt0) & (Z < zt1)

            if blocker_z_range is not None:
                zb0, zb1 = float(blocker_z_range[0]), float(blocker_z_range[1])
                valid_block_any = valid_xy & (Z >= zb0) & (Z < zb1)


        # ---------------------------
        # flatten indices once
        # ---------------------------
        feat_flat = img_feats.reshape(-1, img_feats.shape[-1])  # (B*N*H*W, C)
        ix_f = ix.reshape(-1)
        iy_f = iy.reshape(-1)
        iz_f = iz.reshape(-1)

        valid_main_f   = valid_main.reshape(-1)
        valid_xy_f     = valid_xy.reshape(-1)       # <-- for cnt2d
        valid_ground_f = valid_ground.reshape(-1)
        valid_mid_f    = valid_mid.reshape(-1)
        valid_tall_f   = valid_tall.reshape(-1)
        if blocker_z_range is not None:
            valid_block_any_f = valid_block_any.reshape(-1)


        batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, N, Hf, Wf).reshape(-1)

        # ---------------------------
        # (1) MAIN BEV FEATURES: identical to your previous function
        # ---------------------------
        lin_main = batch_idx * (Dz * Dy * Dx) + iz_f * (Dy * Dx) + iy_f * Dx + ix_f
        lin_main = lin_main[valid_main_f]
        feat_main = feat_flat[valid_main_f]

        C = feat_main.shape[1]
        bev_sum = torch.zeros((B * Dz * Dy * Dx, C), device=device, dtype=torch.float32)
        cnt = torch.zeros((B * Dz * Dy * Dx, 1), device=device, dtype=torch.float32)

        bev_sum.index_add_(0, lin_main, feat_main.float())
        cnt.index_add_(0, lin_main, torch.ones((feat_main.shape[0], 1), device=device, dtype=torch.float32))

        bev_sum = bev_sum.view(B, Dz, Dy, Dx, C).permute(0, 4, 1, 2, 3).contiguous()  # (B,C,Dz,Dy,Dx)
        cnt = cnt.view(B, Dz, Dy, Dx, 1).permute(0, 4, 1, 2, 3).contiguous().clamp_min(1.0)

        bev = (bev_sum / cnt).to(feat_dtype)  # (B,C,Dz,Dy,Dx)

        if self.collapse_z:
            if Dz == 1:
                bev_main = bev.squeeze(2)  # (B,C,Dy,Dx)
            else:
                bev_main = torch.cat(bev.unbind(dim=2), dim=1)  # (B,C*Dz,Dy,Dx)
            bev_main = bev_main.permute(0, 1, 3, 2).contiguous()  # (B,C*,Dx,Dy)
        else:
            bev_main = bev.permute(0, 1, 2, 4, 3).contiguous()  # (B,C,Dz,Dx,Dy) (not used by you)

        # ---------------------------
        # blocker grids from BEV space (your rule: nonzero => blocker)
        # ---------------------------
        # occ_any from main BEV (exactly your previous result)
        occ_any = (bev_main.detach().abs().sum(dim=1) > 0)   # (B,Dx,Dy) bool

        # ground evidence: scatter hits from Z in ground band into (ix,iy)
        # (we only need a mask, not features)
        lin2d = batch_idx * (Dy * Dx) + iy_f * Dx + ix_f

        def scatter2d_hit(valid_f):
            lin_v = lin2d[valid_f]
            tmp = torch.zeros((B * Dy * Dx,), device=device, dtype=torch.float32)
            tmp.index_add_(0, lin_v, torch.ones((lin_v.shape[0],), device=device, dtype=torch.float32))
            hit = (tmp.view(B, Dy, Dx) > 0)  # (B,Dy,Dx)
            return hit.permute(0, 2, 1).contiguous()  # (B,Dx,Dy)

        # hits in bands
        mid_hit  = scatter2d_hit(valid_mid_f)    # (B,Dx,Dy)
        tall_hit = scatter2d_hit(valid_tall_f)   # (B,Dx,Dy)

        ground_hit = scatter2d_hit(valid_ground_f)

        if blocker_z_range is None:
            ground_evidence = ground_hit
        else:
            block_any = scatter2d_hit(valid_block_any_f)  # (B,Dx,Dy)
            ground_evidence = ground_hit & (~block_any)

        # cnt2d: per-cell projected point count (2D, all valid_xy)
        lin_v = lin2d[valid_xy_f]
        tmp_cnt = torch.zeros((B * Dy * Dx,), device=device, dtype=torch.float32)
        tmp_cnt.index_add_(0, lin_v, torch.ones((lin_v.shape[0],), device=device, dtype=torch.float32))
        cnt2d = tmp_cnt.view(B, Dy, Dx).permute(0, 2, 1).contiguous()  # (B,Dx,Dy)
        cnt2d = torch.log1p(cnt2d).to(dtype=bev_main.dtype)            # (B,Dx,Dy)


        # ---------------------------
        # raycast in BEV space (CPU numpy) using occ_any and occ_cam
        # ---------------------------
        # origin at ego (0,0) in grid indices
        ox = int(round((0.0 - float(x_min)) / float(x_step)))
        oy = int(round((0.0 - float(y_min)) / float(y_step)))
        ox = max(0, min(Dx - 1, ox))
        oy = max(0, min(Dy - 1, oy))
        origin_yx = (oy, ox)  # numpy expects (y,x) in (H,W)=(Dy,Dx)

        free_any_list = []
        mid_free_list = []
        maskA_list = []
        maskB_list = []

        occ_any_hw  = occ_any.detach().permute(0, 2, 1).cpu().numpy()   # (B,Dy,Dx)

        mid_hit_hw  = mid_hit.detach().permute(0, 2, 1).cpu().numpy()   # (B,Dy,Dx) as blocker
        tall_hit_hw = tall_hit.detach().permute(0, 2, 1).cpu().numpy()  # (B,Dy,Dx)

        for b in range(B):
            # free_any: visible w.r.t ANY points as blockers (same as before)
            vis_any = compute_visible_mask_from_occ2d(occ_any_hw[b], origin_yx)      # (Dy,Dx)
            free_any = (vis_any & (~occ_any_hw[b]))                                  # (Dy,Dx)

            # mid_cast_free: raycast using mid_hit as blockers
            vis_mid = compute_visible_mask_from_occ2d(mid_hit_hw[b], origin_yx)      # (Dy,Dx)
            mid_free = (vis_mid & (~mid_hit_hw[b]))                                  # (Dy,Dx)  == "mid cast free"

            # wall-like blockers for A: tall_hit & ~mid_cast_free
            wall_like = (tall_hit_hw[b] & (~mid_free))                               # (Dy,Dx)
            vis_A = compute_visible_mask_from_occ2d(wall_like, origin_yx)            # (Dy,Dx)
            maskA_free = (vis_A & (~wall_like))                                      # (Dy,Dx)

            # mid-only blockers for B: mid_hit & ~tall_hit
            mid_only = (mid_hit_hw[b] & (~tall_hit_hw[b]))                           # (Dy,Dx)
            vis_B = compute_visible_mask_from_occ2d(mid_only, origin_yx)             # (Dy,Dx)
            maskB_free = (vis_B & (~mid_only))                                       # (Dy,Dx)

            free_any_list.append(free_any)
            mid_free_list.append(mid_free)
            maskA_list.append(maskA_free)
            maskB_list.append(maskB_free)

        free_any = torch.from_numpy(np.stack(free_any_list, axis=0)).to(device=device)   # (B,Dy,Dx) bool
        maskA_free = torch.from_numpy(np.stack(maskA_list, axis=0)).to(device=device)    # (B,Dy,Dx) bool
        maskB_free = torch.from_numpy(np.stack(maskB_list, axis=0)).to(device=device)    # (B,Dy,Dx) bool

        free_any   = free_any.permute(0, 2, 1).contiguous()       # (B,Dx,Dy)
        maskA_free = maskA_free.permute(0, 2, 1).contiguous()     # (B,Dx,Dy)
        maskB_free = maskB_free.permute(0, 2, 1).contiguous()     # (B,Dx,Dy)


        # ---------------------------
        # final channels (no hand-crafted merge beyond what you specified)
        # ---------------------------
        free_ground_obs = (ground_evidence | free_any)  # (B,Dx,Dy) bool

        free_ground_obs = free_ground_obs.to(dtype=bev_main.dtype).unsqueeze(1)  # (B,1,Dx,Dy)
        cnt2d_ch        = cnt2d.unsqueeze(1)                                      # (B,1,Dx,Dy)
        maskA_ch        = maskA_free.to(dtype=bev_main.dtype).unsqueeze(1)        # (B,1,Dx,Dy)
        maskB_ch        = maskB_free.to(dtype=bev_main.dtype).unsqueeze(1)        # (B,1,Dx,Dy)

        # bev_cat = torch.cat([bev_main, free_ground_obs, cnt2d_ch, maskA_ch, maskB_ch], dim=1)  # (B,C+4,Dx,Dy)
        bev_cat = torch.cat([bev_main, free_ground_obs, cnt2d_ch, maskA_ch], dim=1)  # (B,C+3,Dx,Dy)
        return bev_cat

    def prepare_bev_feats_second(
        self,
        depth,            # (B, N, H, W) meters
        sensor2egos,      # (B, N, 4, 4)
        intrinsics,       # (B, N, 3, 3)
        pointcloud_config_z=(-1.6, 0.32, 0.12)
    ):
        # ---------------------------
        # sanity / shapes
        # ---------------------------
        B, N, Hd, Wd = depth.shape
        device = depth.device

        x_min, x_max, x_step = self.grid_config["x"]
        y_min, y_max, y_step = self.grid_config["y"]
        z_min, z_max, z_step = pointcloud_config_z
        Dz = int(round((float(z_max) - float(z_min)) / float(z_step)))
        Dx, Dy = int(self.Dx), int(self.Dy)
        assert Dx > 0 and Dy > 0 and Dz > 0

        # ---------------------------
        # same geometry as your deterministic BEV
        # ---------------------------
        with autocast(enabled=False):
            u = torch.linspace(0, Wd - 1, Wd, device=device, dtype=torch.float32)
            v = torch.linspace(0, Hd - 1, Hd, device=device, dtype=torch.float32)
            v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")  # (H, W)
            u_grid = u_grid[None, None]
            v_grid = v_grid[None, None]

            fx = intrinsics[:, :, 0, 0].float()[..., None, None]
            fy = intrinsics[:, :, 1, 1].float()[..., None, None]
            cx = intrinsics[:, :, 0, 2].float()[..., None, None]
            cy = intrinsics[:, :, 1, 2].float()[..., None, None]

            z = depth.float()
            x = (u_grid - cx) / fx * z
            y = (v_grid - cy) / fy * z

            R = sensor2egos[:, :, :3, :3].float()
            t = sensor2egos[:, :, :3, 3].float()[..., None, None]

            X = R[:, :, 0, 0][..., None, None] * x + R[:, :, 0, 1][..., None, None] * y + R[:, :, 0, 2][..., None, None] * z + t[:, :, 0]
            Y = R[:, :, 1, 0][..., None, None] * x + R[:, :, 1, 1][..., None, None] * y + R[:, :, 1, 2][..., None, None] * z + t[:, :, 1]
            Z = R[:, :, 2, 0][..., None, None] * x + R[:, :, 2, 1][..., None, None] * y + R[:, :, 2, 2][..., None, None] * z + t[:, :, 2]

            ix = torch.floor((X - float(x_min)) / float(x_step)).to(torch.int64)
            iy = torch.floor((Y - float(y_min)) / float(y_step)).to(torch.int64)
            iz = torch.floor((Z - float(z_min)) / float(z_step)).to(torch.int64)

            valid_main = (
                (ix >= 0) & (ix < Dx) &
                (iy >= 0) & (iy < Dy) &
                (iz >= 0) & (iz < Dz) &
                torch.isfinite(z) &
                torch.isfinite(X) & torch.isfinite(Y) & torch.isfinite(Z)
            )

        # ---------------------------
        # flatten indices once
        # ---------------------------
        feat_flat = torch.stack([X,Y,Z], dim=-1).reshape(-1,3)
        ix_f = ix.reshape(-1)
        iy_f = iy.reshape(-1)
        iz_f = iz.reshape(-1)

        valid_main_f   = valid_main.reshape(-1)

        batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, N, Hd, Wd).reshape(-1)

        lin_main = batch_idx * (Dz * Dy * Dx) + iz_f * (Dy * Dx) + iy_f * Dx + ix_f
        lin_main = lin_main[valid_main_f]
        feat_main = feat_flat[valid_main_f]

        C = 3
        vol_sum = torch.zeros((B * Dz * Dy * Dx, C), device=device, dtype=torch.float32)
        cnt = torch.zeros((B * Dz * Dy * Dx, 1), device=device, dtype=torch.float32)

        vol_sum.index_add_(0, lin_main, feat_main.float())
        cnt.index_add_(0, lin_main, torch.ones((feat_main.shape[0], 1), device=device, dtype=torch.float32))

        vol_sum = vol_sum.view(B, Dz, Dy, Dx, C).permute(0, 4, 1, 2, 3).contiguous()  # (B,3,Dz,Dy,Dx)
        cnt = cnt.view(B, Dz, Dy, Dx, 1).permute(0, 4, 1, 2, 3).contiguous().clamp_min(1.0)

        vol_zyx = vol_sum / cnt  # (B,3,Dz,Dy,Dx)
        vol_xyz = vol_zyx.permute(0, 1, 4, 3, 2).contiguous()  # (B,3,Dx,Dy,Dz)

        # # debug
        # save_path = 'debug_outputs/second_encoder.npz'
        # np.savez_compressed(
        #     save_path,
        #     vol_xyz=vol_xyz.cpu().numpy(),
        #     depth=depth.cpu().numpy(),
        # )
        # exit(0)
        # # end of debug

        bev = self.point_encoder(vol_xyz)  # (B,256,Dx,Dy)
        return bev

    @staticmethod
    def prepare_free_obs(
        depth,            # (B, N, H, W) meters
        sensor2egos,      # (B, N, 4, 4)
        intrinsics,       # (B, N, 3, 3)
        grid_config,      # dict with keys "x","y" (and optionally "z"), same format as before
        Dx, Dy,           # BEV grid size
        ground_z_range=(-1.6, -1.4),
        cam_z_range=(-0.1, 0.1),
        blocker_z_range=(-1.4, -0.1),
        old_behavior=False,
    ):
        """
        Returns:
        free_ground_obs: (B, 1, Dx, Dy)
        free_cam_obs:    (B, 1, Dx, Dy)

        Notes (as discussed):
        - occ_any is NOT from BEV features anymore; it's a depth-geometry blocker mask:
              occ_any := hit(Z in blocker_z_range)
        - ground evidence is "hit ground band AND not blocked":
              ground_evidence := hit(Z in ground_z_range) & ~occ_any
        - everything else is kept the same (2D scatter + 2D raycast).
        """

        # ---------------------------
        # helpers: supercover + raycast (CPU numpy, no grad)
        # ---------------------------
        def supercover_line(x0, y0, x1, y1):
            x0 = int(x0); y0 = int(y0); x1 = int(x1); y1 = int(y1)
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x1 >= x0 else -1
            sy = 1 if y1 >= y0 else -1

            x, y = x0, y0
            err = dx - dy
            yield (x, y)

            while x != x1 or y != y1:
                e2 = 2 * err
                step_x = False
                step_y = False
                px, py = x, y

                if e2 > -dy:
                    err -= dy
                    x += sx
                    step_x = True
                if e2 < dx:
                    err += dx
                    y += sy
                    step_y = True

                if step_x and step_y:
                    yield (px + sx, py)
                    yield (px, py + sy)

                yield (x, y)

        def compute_visible_mask_from_occ2d(occ2d_hw, origin_yx):
            # occ2d_hw: (H,W) bool
            H, W = occ2d_hw.shape
            oy, ox = int(origin_yx[0]), int(origin_yx[1])
            visible = np.zeros((H, W), dtype=bool)

            targets = []
            for ix in range(W):
                targets.append((ix, 0))
                targets.append((ix, H - 1))
            for iy in range(1, H - 1):
                targets.append((0, iy))
                targets.append((W - 1, iy))

            def cast_to(tx, ty):
                for (x, y) in supercover_line(ox, oy, tx, ty):
                    if not (0 <= x < W and 0 <= y < H):
                        break
                    visible[y, x] = True
                    if occ2d_hw[y, x]:
                        break

            for (tx, ty) in targets:
                dx = tx - ox
                dy = ty - oy
                cast_to(tx, ty)

                # diagonal robustness (same as your code)
                if dx != 0 and dy != 0 and abs(dx) == abs(dy):
                    candidates = []
                    if tx == 0 or tx == W - 1:
                        if 0 <= ty - 1 < H: candidates.append((tx, ty - 1))
                        if 0 <= ty + 1 < H: candidates.append((tx, ty + 1))
                    if ty == 0 or ty == H - 1:
                        if 0 <= tx - 1 < W: candidates.append((tx - 1, ty))
                        if 0 <= tx + 1 < W: candidates.append((tx + 1, ty))
                    for (nx, ny) in candidates[:2]:
                        cast_to(nx, ny)

            return visible

        # ---------------------------
        # sanity / shapes
        # ---------------------------
        B, N, Hf, Wf = depth.shape
        device = depth.device
        out_dtype = depth.dtype

        x_min, x_max, x_step = grid_config["x"]
        y_min, y_max, y_step = grid_config["y"]

        Dx = int(Dx); Dy = int(Dy)
        assert Dx > 0 and Dy > 0

        # ---------------------------
        # same geometry (backproject + ego transform)
        # ---------------------------
        with autocast(enabled=False):
            u = torch.linspace(0, Wf - 1, Wf, device=device, dtype=torch.float32)
            v = torch.linspace(0, Hf - 1, Hf, device=device, dtype=torch.float32)
            v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")  # (H, W)
            u_grid = u_grid[None, None]
            v_grid = v_grid[None, None]

            fx = intrinsics[:, :, 0, 0].float()[..., None, None]
            fy = intrinsics[:, :, 1, 1].float()[..., None, None]
            cx = intrinsics[:, :, 0, 2].float()[..., None, None]
            cy = intrinsics[:, :, 1, 2].float()[..., None, None]

            z = depth.float()
            x = (u_grid - cx) / fx * z
            y = (v_grid - cy) / fy * z

            R = sensor2egos[:, :, :3, :3].float()
            t = sensor2egos[:, :, :3, 3].float()[..., None, None]

            X = R[:, :, 0, 0][..., None, None] * x + R[:, :, 0, 1][..., None, None] * y + R[:, :, 0, 2][..., None, None] * z + t[:, :, 0]
            Y = R[:, :, 1, 0][..., None, None] * x + R[:, :, 1, 1][..., None, None] * y + R[:, :, 1, 2][..., None, None] * z + t[:, :, 1]
            Z = R[:, :, 2, 0][..., None, None] * x + R[:, :, 2, 1][..., None, None] * y + R[:, :, 2, 2][..., None, None] * z + t[:, :, 2]

            ix = torch.floor((X - float(x_min)) / float(x_step)).to(torch.int64)
            iy = torch.floor((Y - float(y_min)) / float(y_step)).to(torch.int64)

            valid_xy = (
                (ix >= 0) & (ix < Dx) &
                (iy >= 0) & (iy < Dy) &
                torch.isfinite(z) &
                torch.isfinite(X) & torch.isfinite(Y) & torch.isfinite(Z)
            )

            zg0, zg1 = float(ground_z_range[0]),  float(ground_z_range[1])
            zc0, zc1 = float(cam_z_range[0]),     float(cam_z_range[1])
            zb0, zb1 = float(blocker_z_range[0]), float(blocker_z_range[1])

            valid_ground  = valid_xy & (Z >= zg0) & (Z < zg1)
            valid_cam     = valid_xy & (Z >= zc0) & (Z < zc1)
            valid_blocker = valid_xy & (Z >= zb0) & (Z < zb1)

        # ---------------------------
        # flatten indices once
        # ---------------------------
        ix_f = ix.reshape(-1)
        iy_f = iy.reshape(-1)

        valid_xy_f      = valid_xy.reshape(-1)
        valid_ground_f  = valid_ground.reshape(-1)
        valid_cam_f     = valid_cam.reshape(-1)
        valid_blocker_f = valid_blocker.reshape(-1)

        batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, N, Hf, Wf).reshape(-1)

        # lin2d in (Dy,Dx) then permute to (Dx,Dy) to match your BEV convention
        lin2d = batch_idx * (Dy * Dx) + iy_f * Dx + ix_f

        def scatter2d_hit(valid_f):
            lin_v = lin2d[valid_f]
            tmp = torch.zeros((B * Dy * Dx,), device=device, dtype=torch.float32)
            tmp.index_add_(0, lin_v, torch.ones((lin_v.shape[0],), device=device, dtype=torch.float32))
            hit = (tmp.view(B, Dy, Dx) > 0)          # (B,Dy,Dx)
            return hit.permute(0, 2, 1).contiguous() # (B,Dx,Dy)

        # ---------------------------
        # blocker grids (depth geometry only)
        # ---------------------------
        # discussed replacement for occ_any:
        occ_any = scatter2d_hit(valid_blocker_f)     # (B,Dx,Dy) bool
        occ_cam = scatter2d_hit(valid_cam_f)         # (B,Dx,Dy) bool

        # discussed bug-fix for ground evidence:
        ground_hit = scatter2d_hit(valid_ground_f)   # (B,Dx,Dy) bool
        if old_behavior:
            ground_evidence = ground_hit               # (B,Dx,Dy) bool
        else:
            ground_evidence = ground_hit & (~occ_any)    # (B,Dx,Dy) bool

        # ---------------------------
        # raycast in BEV space (CPU numpy) using occ_any and occ_cam
        # ---------------------------
        ox = int(round((0.0 - float(x_min)) / float(x_step)))
        oy = int(round((0.0 - float(y_min)) / float(y_step)))
        ox = max(0, min(Dx - 1, ox))
        oy = max(0, min(Dy - 1, oy))
        origin_yx = (oy, ox)  # numpy expects (y,x) in (H,W)=(Dy,Dx)

        free_any_list = []
        free_cam_list = []

        occ_any_hw = occ_any.detach().permute(0, 2, 1).cpu().numpy()  # (B,Dy,Dx)
        occ_cam_hw = occ_cam.detach().permute(0, 2, 1).cpu().numpy()  # (B,Dy,Dx)

        for b in range(B):
            vis_any = compute_visible_mask_from_occ2d(occ_any_hw[b], origin_yx)  # (Dy,Dx)
            free_any = (vis_any & (~occ_any_hw[b]))                              # (Dy,Dx)

            vis_cam = compute_visible_mask_from_occ2d(occ_cam_hw[b], origin_yx)  # (Dy,Dx)
            free_cam = (vis_cam & (~occ_cam_hw[b]))                              # (Dy,Dx)

            free_any_list.append(free_any)
            free_cam_list.append(free_cam)

        free_any = torch.from_numpy(np.stack(free_any_list, axis=0)).to(device=device)  # (B,Dy,Dx) bool
        free_cam = torch.from_numpy(np.stack(free_cam_list, axis=0)).to(device=device)  # (B,Dy,Dx) bool
        free_any = free_any.permute(0, 2, 1).contiguous()  # (B,Dx,Dy)
        free_cam = free_cam.permute(0, 2, 1).contiguous()  # (B,Dx,Dy)

        # ---------------------------
        # final channels (same merge as before, with discussed ground_evidence fix)
        # ---------------------------
        free_ground_obs = (ground_evidence | free_any)  # (B,Dx,Dy) bool
        free_cam_obs    = free_cam                      # (B,Dx,Dy) bool

        free_ground_obs = free_ground_obs.to(dtype=out_dtype).unsqueeze(1)  # (B,1,Dx,Dy)
        free_cam_obs    = free_cam_obs.to(dtype=out_dtype).unsqueeze(1)     # (B,1,Dx,Dy)

        return free_ground_obs, free_cam_obs

    @staticmethod
    def prepare_free_obs_new(
        depth,            # (B, N, H, W) meters
        sensor2egos,      # (B, N, 4, 4)
        intrinsics,       # (B, N, 3, 3)
        grid_config,      # dict with keys "x","y": [min,max,step]
        Dx, Dy,           # BEV grid size
        ground_z_range=(-1.6, -1.4),
        blocker_z_range=(-1.4, -0.1),   # same meaning as old prepare_free_obs
        mid_z_range=(-0.1, 0.1),        # for Mask A
        tall_z_range=(0.25, 0.35),      # for Mask A
        old_behavior=False,             # keep same semantics as old prepare_free_obs
        return_bool=False,              # if False: return float (0/1) like old
    ):
        """
        Returns:
            free_ground_obs: (B, 1, Dx, Dy)  # EXACTLY old prepare_free_obs free_ground_obs logic (geometry-only)
            blocker_mask:    (B, 1, Dx, Dy)  # 2D hit mask for Z in blocker_z_range (no raycast)
            maskA_free:      (B, 1, Dx, Dy)  # Mask A free-space (v2-style): raycast w.r.t wall_like blockers

        Notes:
          - No cam-free output.
          - All masks are computed from depth geometry (no BEV features, no self.grid_config["z"]).
        """

        # ---------------------------
        # helpers: supercover + raycast (CPU numpy, no grad)
        # ---------------------------
        def supercover_line(x0, y0, x1, y1):
            x0 = int(x0); y0 = int(y0); x1 = int(x1); y1 = int(y1)
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x1 >= x0 else -1
            sy = 1 if y1 >= y0 else -1

            x, y = x0, y0
            err = dx - dy
            yield (x, y)

            while x != x1 or y != y1:
                e2 = 2 * err
                step_x = False
                step_y = False
                px, py = x, y

                if e2 > -dy:
                    err -= dy
                    x += sx
                    step_x = True
                if e2 < dx:
                    err += dx
                    y += sy
                    step_y = True

                if step_x and step_y:
                    yield (px + sx, py)
                    yield (px, py + sy)

                yield (x, y)

        def compute_visible_mask_from_occ2d(occ2d_hw, origin_yx):
            # occ2d_hw: (H,W) bool
            H, W = occ2d_hw.shape
            oy, ox = int(origin_yx[0]), int(origin_yx[1])
            visible = np.zeros((H, W), dtype=bool)

            targets = []
            for ix in range(W):
                targets.append((ix, 0))
                targets.append((ix, H - 1))
            for iy in range(1, H - 1):
                targets.append((0, iy))
                targets.append((W - 1, iy))

            def cast_to(tx, ty):
                for (x, y) in supercover_line(ox, oy, tx, ty):
                    if not (0 <= x < W and 0 <= y < H):
                        break
                    visible[y, x] = True
                    if occ2d_hw[y, x]:
                        break

            for (tx, ty) in targets:
                dx = tx - ox
                dy = ty - oy
                cast_to(tx, ty)

                # diagonal robustness (same as your prior code)
                if dx != 0 and dy != 0 and abs(dx) == abs(dy):
                    candidates = []
                    if tx == 0 or tx == W - 1:
                        if 0 <= ty - 1 < H: candidates.append((tx, ty - 1))
                        if 0 <= ty + 1 < H: candidates.append((tx, ty + 1))
                    if ty == 0 or ty == H - 1:
                        if 0 <= tx - 1 < W: candidates.append((tx - 1, ty))
                        if 0 <= tx + 1 < W: candidates.append((tx + 1, ty))
                    for (nx, ny) in candidates[:2]:
                        cast_to(nx, ny)

            return visible

        # ---------------------------
        # sanity / shapes
        # ---------------------------
        B, N, Hf, Wf = depth.shape
        device = depth.device
        out_dtype = torch.bool if return_bool else depth.dtype

        x_min, x_max, x_step = grid_config["x"]
        y_min, y_max, y_step = grid_config["y"]
        Dx = int(Dx); Dy = int(Dy)
        assert Dx > 0 and Dy > 0

        # ---------------------------
        # geometry: backproject + ego transform
        # ---------------------------
        with autocast(enabled=False):
            u = torch.linspace(0, Wf - 1, Wf, device=device, dtype=torch.float32)
            v = torch.linspace(0, Hf - 1, Hf, device=device, dtype=torch.float32)
            v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")  # (H, W)
            u_grid = u_grid[None, None]
            v_grid = v_grid[None, None]

            fx = intrinsics[:, :, 0, 0].float()[..., None, None]
            fy = intrinsics[:, :, 1, 1].float()[..., None, None]
            cx = intrinsics[:, :, 0, 2].float()[..., None, None]
            cy = intrinsics[:, :, 1, 2].float()[..., None, None]

            z = depth.float()
            x = (u_grid - cx) / fx * z
            y = (v_grid - cy) / fy * z

            R = sensor2egos[:, :, :3, :3].float()
            t = sensor2egos[:, :, :3, 3].float()[..., None, None]

            X = R[:, :, 0, 0][..., None, None] * x + R[:, :, 0, 1][..., None, None] * y + R[:, :, 0, 2][..., None, None] * z + t[:, :, 0]
            Y = R[:, :, 1, 0][..., None, None] * x + R[:, :, 1, 1][..., None, None] * y + R[:, :, 1, 2][..., None, None] * z + t[:, :, 1]
            Z = R[:, :, 2, 0][..., None, None] * x + R[:, :, 2, 1][..., None, None] * y + R[:, :, 2, 2][..., None, None] * z + t[:, :, 2]

            ix = torch.floor((X - float(x_min)) / float(x_step)).to(torch.int64)
            iy = torch.floor((Y - float(y_min)) / float(y_step)).to(torch.int64)

            valid_xy = (
                (ix >= 0) & (ix < Dx) &
                (iy >= 0) & (iy < Dy) &
                torch.isfinite(z) &
                torch.isfinite(X) & torch.isfinite(Y) & torch.isfinite(Z)
            )

            zg0, zg1 = float(ground_z_range[0]),  float(ground_z_range[1])
            zb0, zb1 = float(blocker_z_range[0]), float(blocker_z_range[1])
            zm0, zm1 = float(mid_z_range[0]),     float(mid_z_range[1])
            zt0, zt1 = float(tall_z_range[0]),    float(tall_z_range[1])

            valid_ground  = valid_xy & (Z >= zg0) & (Z < zg1)
            valid_blocker = valid_xy & (Z >= zb0) & (Z < zb1)
            valid_mid     = valid_xy & (Z >= zm0) & (Z < zm1)
            valid_tall    = valid_xy & (Z >= zt0) & (Z < zt1)

        # ---------------------------
        # flatten indices once
        # ---------------------------
        ix_f = ix.reshape(-1)
        iy_f = iy.reshape(-1)

        valid_ground_f  = valid_ground.reshape(-1)
        valid_blocker_f = valid_blocker.reshape(-1)
        valid_mid_f     = valid_mid.reshape(-1)
        valid_tall_f    = valid_tall.reshape(-1)

        batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, N, Hf, Wf).reshape(-1)

        # lin2d in (Dy,Dx) then permute to (Dx,Dy) to match your BEV convention
        lin2d = batch_idx * (Dy * Dx) + iy_f * Dx + ix_f

        def scatter2d_hit(valid_f):
            lin_v = lin2d[valid_f]
            tmp = torch.zeros((B * Dy * Dx,), device=device, dtype=torch.float32)
            if lin_v.numel() > 0:
                tmp.index_add_(0, lin_v, torch.ones((lin_v.shape[0],), device=device, dtype=torch.float32))
            hit = (tmp.view(B, Dy, Dx) > 0)          # (B,Dy,Dx)
            return hit.permute(0, 2, 1).contiguous() # (B,Dx,Dy)
    
        def scatter2d_count(valid_f):
            lin_v = lin2d[valid_f]
            tmp = torch.zeros((B * Dy * Dx,), device=device, dtype=torch.int32)
            if lin_v.numel() > 0:
                tmp.index_add_(0, lin_v, torch.ones((lin_v.shape[0],), device=device, dtype=torch.int32))
            cnt = tmp.view(B, Dy, Dx)                    # (B,Dy,Dx)
            return cnt.permute(0, 2, 1).contiguous()     # (B,Dx,Dy)

        # ---------------------------
        # depth-only hit masks
        # ---------------------------
        blocker_cnt = scatter2d_count(valid_blocker_f)  # (B,Dx,Dy) int
        blocker_mask = scatter2d_hit(valid_blocker_f)  # (B,Dx,Dy) bool  == old occ_any
        ground_hit   = scatter2d_hit(valid_ground_f)   # (B,Dx,Dy) bool
        mid_hit      = scatter2d_hit(valid_mid_f)      # (B,Dx,Dy) bool
        tall_hit     = scatter2d_hit(valid_tall_f)     # (B,Dx,Dy) bool

        # EXACT old prepare_free_obs ground_evidence logic
        if old_behavior:
            ground_evidence = ground_hit
        else:
            ground_evidence = ground_hit & (~blocker_mask)

        # ---------------------------
        # raycast (CPU numpy)
        # ---------------------------
        ox = int(math.floor((0.0 - float(x_min)) / float(x_step)))
        oy = int(math.floor((0.0 - float(y_min)) / float(y_step)))
        ox = max(0, min(Dx - 1, ox))
        oy = max(0, min(Dy - 1, oy))
        origin_yx = (oy, ox)  # numpy expects (y,x) in (H,W)=(Dy,Dx)

        # Convert to (B,Dy,Dx) for numpy, since compute_visible_mask_from_occ2d uses (H,W)
        blocker_hw = blocker_mask.detach().permute(0, 2, 1).cpu().numpy()  # (B,Dy,Dx)
        mid_hw     = mid_hit.detach().permute(0, 2, 1).cpu().numpy()       # (B,Dy,Dx)
        tall_hw    = tall_hit.detach().permute(0, 2, 1).cpu().numpy()      # (B,Dy,Dx)

        free_any_list = []
        mid_free_list = []
        maskA_list = []

        for b in range(B):
            # old prepare_free_obs "free_any": visible wrt blocker_mask
            vis_any = compute_visible_mask_from_occ2d(blocker_hw[b], origin_yx)
            free_any = (vis_any & (~blocker_hw[b]))

            # Mask A: mid_free then wall_like then free wrt wall_like
            vis_mid = compute_visible_mask_from_occ2d(mid_hw[b], origin_yx)
            mid_free = (vis_mid & (~mid_hw[b]))

            wall_like = (tall_hw[b] & (~mid_free))
            vis_A = compute_visible_mask_from_occ2d(wall_like, origin_yx)
            maskA_free = (vis_A & (~wall_like))

            free_any_list.append(free_any)
            mid_free_list.append(mid_free)  # (kept if you want debugging later)
            maskA_list.append(maskA_free)

        free_any = torch.from_numpy(np.stack(free_any_list, axis=0)).to(device=device)   # (B,Dy,Dx) bool
        maskA_free = torch.from_numpy(np.stack(maskA_list, axis=0)).to(device=device)    # (B,Dy,Dx) bool

        free_any = free_any.permute(0, 2, 1).contiguous()         # (B,Dx,Dy)
        maskA_free = maskA_free.permute(0, 2, 1).contiguous()     # (B,Dx,Dy)

        # ---------------------------
        # final outputs
        # ---------------------------
        free_ground_obs = (ground_evidence | free_any)  # (B,Dx,Dy) bool
        if return_bool:
            raise NotImplementedError
            free_ground_obs = free_ground_obs            # (B, Dx, Dy) bool
            blocker_mask_out = blocker_mask              # (B, Dx, Dy) bool
            maskA_out = maskA_free                       # (B, Dx, Dy) bool
        else:
            free_ground_obs = free_ground_obs.to(dtype=out_dtype)      # (B, Dx, Dy) float
            blocker_mask_out = blocker_mask.to(dtype=out_dtype)        # (B, Dx, Dy) float
            maskA_out = maskA_free.to(dtype=out_dtype)                 # (B, Dx, Dy) float
            blocker_cnt_out = blocker_cnt.to(dtype=out_dtype)                 # (B, Dx, Dy) float, optional extra output if you want count-based analysis later

        return free_ground_obs, blocker_mask_out, blocker_cnt_out, maskA_out

    def prepare_bev_feats_context(
        self,
        depth,          # (B, N, H, W), invalid == 10
        img_feats,      # (B, N, fH, fW, C)  e.g. (B,4,16,16,2048)
        sensor2egos,    # (B, N, 4, 4)
        intrinsics,     # (B, N, 3, 3)
        ds_ratio=4,     # 128/4 -> 32
    ):
        B, N, H, W = depth.shape
        fH, fW, C = img_feats.shape[2], img_feats.shape[3], img_feats.shape[-1]
        device = depth.device
        feat_dtype = img_feats.dtype

        x_min, x_max, x_step0 = self.grid_config["x"]
        y_min, y_max, y_step0 = self.grid_config["y"]
        z_min, z_max, _ = self.grid_config["z"]

        x_step = float(x_step0) * float(ds_ratio)
        y_step = float(y_step0) * float(ds_ratio)
        Dx = int(round((float(x_max) - float(x_min)) / x_step))
        Dy = int(round((float(y_max) - float(y_min)) / y_step))

        # ---- depth -> patch depth (mean, excluding invalid==10) ----
        ph, pw = H // fH, W // fW
        d = depth.float().reshape(B, N, fH, ph, fW, pw)
        valid = (d != 10.0)
        d_sum = (d * valid).sum(dim=(3, 5))
        d_cnt = valid.sum(dim=(3, 5))
        d_patch = d_sum / d_cnt.clamp_min(1)                 # (B,N,fH,fW)
        d_valid = (d_cnt > 0)                                # (B,N,fH,fW)

        # ---- patch center pixel coords (u,v) ----
        u0 = (torch.arange(fW, device=device, dtype=torch.float32) + 0.5) * pw - 0.5
        v0 = (torch.arange(fH, device=device, dtype=torch.float32) + 0.5) * ph - 0.5
        v_grid, u_grid = torch.meshgrid(v0, u0, indexing="ij")     # (fH,fW)
        u_grid = u_grid[None, None]                                # (1,1,fH,fW)
        v_grid = v_grid[None, None]                                # (1,1,fH,fW)

        with autocast(enabled=False):
            fx = intrinsics[:, :, 0, 0].float()[..., None, None]
            fy = intrinsics[:, :, 1, 1].float()[..., None, None]
            cx = intrinsics[:, :, 0, 2].float()[..., None, None]
            cy = intrinsics[:, :, 1, 2].float()[..., None, None]

            z = d_patch  # (B,N,fH,fW)
            x = (u_grid - cx) / fx * z
            y = (v_grid - cy) / fy * z

            R = sensor2egos[:, :, :3, :3].float()
            t = sensor2egos[:, :, :3, 3].float()[..., None, None]

            X = R[:, :, 0, 0][..., None, None] * x + R[:, :, 0, 1][..., None, None] * y + R[:, :, 0, 2][..., None, None] * z + t[:, :, 0]
            Y = R[:, :, 1, 0][..., None, None] * x + R[:, :, 1, 1][..., None, None] * y + R[:, :, 1, 2][..., None, None] * z + t[:, :, 1]
            Z = R[:, :, 2, 0][..., None, None] * x + R[:, :, 2, 1][..., None, None] * y + R[:, :, 2, 2][..., None, None] * z + t[:, :, 2]

            ix = torch.floor((X - float(x_min)) / x_step).to(torch.int64)
            iy = torch.floor((Y - float(y_min)) / y_step).to(torch.int64)

            valid_main = (
                d_valid &
                (ix >= 0) & (ix < Dx) &
                (iy >= 0) & (iy < Dy) &
                (Z >= float(z_min)) & (Z < float(z_max))
            )

        # ---- scatter mean into (B, C, Dx, Dy) ----
        feat_flat = img_feats.reshape(-1, C)
        ix_f = ix.reshape(-1)
        iy_f = iy.reshape(-1)
        v_f  = valid_main.reshape(-1)

        batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, N, fH, fW).reshape(-1)
        lin2d = batch_idx * (Dy * Dx) + iy_f * Dx + ix_f
        lin2d = lin2d[v_f]
        feat_v = feat_flat[v_f]

        out_sum = torch.zeros((B * Dy * Dx, C), device=device, dtype=torch.float32)
        out_cnt = torch.zeros((B * Dy * Dx, 1), device=device, dtype=torch.float32)
        out_sum.index_add_(0, lin2d, feat_v.float())
        out_cnt.index_add_(0, lin2d, torch.ones((feat_v.shape[0], 1), device=device, dtype=torch.float32))

        out = (out_sum / out_cnt.clamp_min(1.0)).view(B, Dy, Dx, C).permute(0, 3, 2, 1).contiguous()
        return out.to(dtype=feat_dtype)  # (B, C, Dx, Dy) == (B,2048,32,32) when ds_ratio=4

    @staticmethod
    def prob_to_logits(p, eps=1e-6):
        # p: torch.Tensor
        p = p.clamp(eps, 1.0 - eps)
        return torch.log(p) - torch.log1p(-p)

    def forward(
        self,
        query,            # (B, C)
        skip_result,      # (B, 3) logits for sinθ, cosθ, r, should not be used in this version
        depth,            # (B, N, H, W) depth in meters
        raw_imgs,         # (B, N, 3, H, W) uint8 BGR
        intrinsics,       # (B, N, 3, 3)
        sensor2egos,      # (B, N, 4, 4)
        context_tokens=None,
        **kwargs,
    ):
        # ---- 1) preprocess images (BGR->RGB, uint8->float, normalize) ----
        assert raw_imgs.dim() == 5, f"raw_imgs must be (B,N,3,H,W), got {raw_imgs.shape}"
        B, N, _, H, W = raw_imgs.shape
        assert (H, W) == self.input_size, f"Expected raw imgs == input_size {self.input_size}, got {(H, W)}"
        assert depth.shape[:2] == (B, N), f"depth shape {depth.shape} must match (B,N)=({B},{N})"

        imgs_bn = raw_imgs.flatten(0, 1)  # (B*N, 3, H, W)

        # Det3DDataPreprocessor expects 'img' and outputs 'imgs'
        data = {"inputs": {"img": imgs_bn}, "data_samples": None}
        data = self.data_preprocessor(data, training=self.training)
        imgs_bn = data["inputs"]["imgs"]  # (B*N, 3, H, W) float normalized RGB

        # ---- 2) backbone -> patch tokens -> (B*N, 768, fH, fW) ----
        # DINOv2 expects RGB; forward_features gives patch tokens.
        if self.backbone.training:
            self.backbone.eval()
        with torch.no_grad():
            feats_dict = self.backbone.forward_features(imgs_bn)

        tokens = feats_dict.get("x_norm_patchtokens", None)
        if tokens is None:
            tokens = feats_dict.get("x_patchtokens", None)
        if tokens is None:
            raise KeyError(
                "Could not find patch tokens in backbone.forward_features output. "
                "Expected key 'x_norm_patchtokens' (or 'x_patchtokens')."
            )

        # tokens: (B*N, T, 768), where T == fH*fW (+1 if cls token exists)
        ps = self.patch_size
        fH, fW = H // ps, W // ps
        expected_T = fH * fW
        assert expected_T == tokens.shape[1]
        x = tokens.transpose(1, 2).contiguous().view(B * N, 768, fH, fW)  # (B*N, 768, 32, 32)

        x = self.img_proj(x)  # (B*N, C, fH, fW)
        # newly added, we upsample back to get dense points with rgb feature
        x = F.interpolate(x, size=(H, W), mode="nearest")  # upsample to full res

        C = x.shape[1]
        if self.use_bev_feats:
            # img_feats = x.view(B, N, C, fH, fW).permute(0, 1, 3, 4, 2).contiguous()  # (B, N, fH, fW, C)
            img_feats = x.view(B, N, C, H, W).permute(0, 1, 3, 4, 2).contiguous() # (B, N, H, W, C)
            # bev_feats = self.prepare_bev_feats_deterministic(
            # bev_feats = self.prepare_bev_feats_deterministic_with_obs( # for Method C.1
            bev_feats = self.prepare_bev_feats_deterministic_with_obs_v2( # for Method C.2
                depth=depth,
                img_feats=img_feats,
                sensor2egos=sensor2egos,
                intrinsics=intrinsics,
                blocker_z_range=(-1.4, -0.1),
            )  # (B, C+m, Dx, Dy) = B, proj_c+m, 128, 128

            # for method C.3
            pointcloud_bev_feats = self.prepare_bev_feats_second(
                depth=depth,
                sensor2egos=sensor2egos,
                intrinsics=intrinsics,
                pointcloud_config_z=(-1.6, 0.32, 0.12)
            )  # (B, 256, Dx, Dy) = B, 256, 128, 128
            prior_masks = bev_feats[:, -3:, :, :]  # (B, 3, Dx, Dy)
            pc_gate = torch.sigmoid(self.pc_gate(prior_masks))  # (B,1 or 256,Dx,Dy) in [0,1]
            # if self.training:
            #     drop_p = 0.2
            #     if torch.rand((), device=pointcloud_bev_feats.device) < drop_p:
            #         # pointcloud_bev_feats = 0.0 * pointcloud_bev_feats   # hard
            #         pointcloud_bev_feats = 0.2 * pointcloud_bev_feats  # soft

            # bev_feats = torch.cat([bev_feats, pc_gate * pointcloud_bev_feats], dim=1)  # (B, 256 + 256 +3, Dx, Dy) # NOTE: this cat must first lss then point to be consistent with init weights
            lss_feats = bev_feats[:, :-3, :, :]  # (B, C, Dx, Dy)
            lss_gated = (1-pc_gate) * lss_feats  # (B, C, Dx, Dy)
            pc_gated  = pc_gate * pointcloud_bev_feats  # (B, 256, Dx, Dy)
            bev_feats = torch.cat([lss_gated, prior_masks, pc_gated], dim=1)  # (B, C + 256 +3, Dx, Dy)

        if context_tokens is not None and self.variant == 'catvisbev':
            assert self.bev_encoder.cin == 576
            context_bev_feats = self.prepare_bev_feats_context(
                depth=depth,
                img_feats=context_tokens, # B, N, fH, fW, 2048
                sensor2egos=sensor2egos,
                intrinsics=intrinsics,
            ) # B, 2048, 32, 32
            # # debug
            # debug_save_path = f'debug_outputs/0_context.npz'
            # import numpy as np
            # np.savez_compressed(
            #     debug_save_path,
            #     context_bev_feats = context_bev_feats.detach().cpu().numpy(),  # (B, 2048, 32, 32)
            #     traversable_mask = kwargs['traversable_mask'].detach().cpu().numpy(),  # (B,Dx,Dy)
            #     visible_mask     = kwargs['visible_mask'].detach().cpu().numpy(),
            #     affordance_mask  = kwargs['affordance_mask'].detach().cpu().numpy(),
            # )
            # print(f"[DEBUG] Saved context_bev_feats to {debug_save_path}")
            # exit(0)
            # # end of debug
            context_bev_feats = self.context_pixshuffle(context_bev_feats) # B, 32, 128, 128

        # # debug see the new 4 masks does expected or not
        # if not hasattr(self, 'masks_for_debug'):
        #     self.masks_for_debug = []
        #     self.imgs_for_debug = []
        #     self.debug_cnt = 0

        # online_masks = bev_feats[:, -4:, :, :].detach().float().cpu().numpy() # (B,4,Dx,Dy)

        # traversable_mask = kwargs['traversable_mask'].detach().cpu().numpy()  # (B,Dx,Dy)
        # visible_mask     = kwargs['visible_mask'].detach().cpu().numpy()
        # affordance_mask  = kwargs['affordance_mask'].detach().cpu().numpy()

        # all_masks = np.stack([
        #     traversable_mask,
        #     visible_mask,
        #     affordance_mask,
        #     online_masks[:, 0],
        #     online_masks[:, 1],
        #     online_masks[:, 2],
        #     online_masks[:, 3],
        # ], axis=1)  # (B,7,Dx,Dy)

        # self.masks_for_debug.append(all_masks)
        # self.imgs_for_debug.append(raw_imgs.detach().cpu().numpy())  # (B,N,3,H,W)
        # self.debug_cnt += 1

        # if self.debug_cnt >= 9:
        #     save_path = 'debug_outputs/aff_masks_dbg.npz'
        #     np.savez_compressed(
        #         save_path,
        #         imgs=np.stack(self.imgs_for_debug, axis=0),   # (T,B,N,3,H,W)
        #         masks=np.stack(self.masks_for_debug, axis=0)  # (T,B,7,Dx,Dy)
        #     )
        #     print(f"[DEBUG] Saved affordance masks and imgs to {save_path}")
        #     exit(0)
        # # end of debug


        # # debug, see how much the model rely on the extra two channels
        # rand_idx = torch.randperm(128*128, device=bev_feats.device)
        # to_rand = bev_feats[:, -2:, :, :].view(B, 2, 128*128)
        # randed = to_rand[:, :, rand_idx].view(B, 2, 128, 128)
        # bev_feats = torch.cat([bev_feats[:, :-2, :, :], randed], dim=1)
        # # end of debug

        # # debug
        # import numpy as np
        # import cv2
        # debug_save_path = f'debug_outputs/aff_lss_{self.debug_cnt}.jpg'
        # nonzero_bev_feats = (np.abs(bev_feats[0].detach().cpu().numpy()).sum(axis=0) > 0)  # (Dx, Dy) bool
        # bev_img = (nonzero_bev_feats.astype(np.uint8) * 255)
        # cv2.imwrite(debug_save_path, bev_img)
        # print(f"[DEBUG] Saved LSS BEV features to {debug_save_path}")
        # self.debug_cnt += 1
        # # end of debug

        if self.use_bev_feats:
            self.latest_lss_for_debug = bev_feats # (B, C or C+2, Dx, Dy)
            # obs_masks = bev_feats[:, -2:, :, :]  # (B, 2, Dx, Dy)
            # raw_bev_feats = bev_feats[:, :-2, :, :]  # (B, C, Dx, Dy)
            obs_masks = bev_feats[:, -3:, :, :]  # (B, 3, Dx, Dy)
            raw_bev_feats = bev_feats[:, :-3, :, :]  # (B, C, Dx, Dy)
            if bev_feats.shape[1] != C:
                # bev_feats = self.obs_proj(bev_feats)  # (B, C, Dx, Dy)
                # Method C.3
                bev_feats = self.obs_proj_bevfusion(bev_feats)  # (B, C, Dx, Dy)

        # # Method I: geom only + film, even 1e-4 not learning in overfitting, film not working for small ch numbers
        # # ---- build multi-scale prior maps from query ----
        # occ = (raw_bev_feats.abs().sum(dim=1) > 0).float()   # (B,Dx,Dy)
        # geom_bev_feats = torch.stack([occ, obs_masks[:,0], obs_masks[:,1]], dim=1)
        # stem, c2, c3, c4, c5 = self.bev_encoder_2(geom_bev_feats)
        # p16 = self.query_to_prior16(query).view(B, 32, 16, 16)   # (B, prior_ch, 16, 16)
        # p32 = self.prior_up_32(p16)                                      # (B, prior_ch, 32, 32)
        # p64 = self.prior_up_64(p32)                                      # (B, prior_ch, 64, 64)
        # p128 = self.prior_up_128(p64)                                      # (B, prior_ch, 128, 128)
        # # ---- spatial FiLM on C2..C5 ----
        # g2, b2 = self.film2(p128).chunk(2, dim=1)  # (B, c2_ch, 128, 128)
        # g3, b3 = self.film3(p64).chunk(2, dim=1)  # (B, c3_ch, 64, 64)
        # g4, b4 = self.film4(p32).chunk(2, dim=1)  # (B, c4_ch, 32, 32)
        # g5, b5 = self.film5(p16).chunk(2, dim=1) # (B, c5_ch, 16, 16)
        # c2 = self.norm2(c2) * (1.0 + g2) + b2
        # c3 = self.norm3(c3) * (1.0 + g3) + b3
        # c4 = self.norm4(c4) * (1.0 + g4) + b4
        # c5 = self.norm5(c5) * (1.0 + g5) + b5
        # p2, p3, p4, p5 = self.bev_fpn_2(c2, c3, c4, c5)
        # p3u = F.interpolate(p3, size=p2.shape[-2:], mode="nearest")
        # p4u = F.interpolate(p4, size=p2.shape[-2:], mode="nearest")
        # p5u = F.interpolate(p5, size=p2.shape[-2:], mode="nearest")
        # fpn_out = p2 + p3u + p4u + p5u
        # fpn_out = self.bevfpn_fuse_2(fpn_out) # B, bev_head_channels, 128, 128
        # aff_logits = self.affordance_head_2(fpn_out).squeeze(1) # (B, Dx, Dy)
        # return aff_logits


        # # Method A: cat, test found that, made bev feats to 0, iou from 0.8 becomes 0.7, so condition becomes content
        # q = self.query_to_prior(query)                      # (B, 256*4*4)
        # q = q.view(B, 256, 4, 4)                # (B, 256, 4, 4)
        # q = self.prior_upsampler(q)                       # (B, 256, 32, 32)
        # prior_low_res = self.prior_proj(q)                     # (B, 256, 32, 32)
        # prior = F.interpolate(prior_low_res, size=bev_feats.shape[-2:], mode="nearest")  # (B, 256, Dx, Dy)
        # bev_feats = torch.cat([bev_feats, prior], dim=1)   # (B, C+256=512, Dx, Dy)

        # # Method B: film, can use bev but it is too sparse so hard to learn
        # # query = torch.zeros_like(query) # debug to ablate
        # film = self.film_mlp(query)                          # (B, 2*C)
        # gamma, beta = film.chunk(2, dim=1)                   # each (B, C)
        # gamma = gamma.unsqueeze(-1).unsqueeze(-1)            # (B, C, 1, 1)
        # beta  = beta.unsqueeze(-1).unsqueeze(-1)             # (B, C, 1, 1)
        # bev_feats = self.bev_film_norm(bev_feats)
        # bev_feats = bev_feats * (1.0 + gamma) + beta         # (B, C, Dx, Dy)
        # # with torch.no_grad():
        # #     g = gamma.squeeze(-1).squeeze(-1)  # (B,C)
        # #     b = beta.squeeze(-1).squeeze(-1)
        # #     print("gamma mean/std/min/max:", g.mean().item(), g.std().item(), g.min().item(), g.max().item())
        # #     print("beta  mean/std/min/max:", b.mean().item(), b.std().item(), b.min().item(), b.max().item())

        # # Method F: only gemeotry obs
        # q = self.query_to_prior(query)                      # (B, 128*4*4)
        # q = q.view(B, 128, 4, 4)                # (B, 256, 4, 4)
        # q = self.prior_upsampler(q)                       # (B, 128, 32, 32)
        # prior_low_res = self.prior_proj(q)                     # (B, 128, 32, 32)
        # prior = F.interpolate(prior_low_res, size=bev_feats.shape[-2:], mode="nearest")  # (B, 128, Dx, Dy)
        # occ = (raw_bev_feats.abs().sum(dim=1) > 0).float()   # (B,Dx,Dy)
        # geom_bev_feats = torch.stack([occ, obs_masks[:,0], obs_masks[:,1]], dim=1)
        # # rand_idx = torch.randperm(self.Dx * self.Dy, device=bev_feats.device)
        # # geom_bev_feats = geom_bev_feats.view(B, 3, self.Dx * self.Dy)[:, :, rand_idx].view(B, 3, self.Dx, self.Dy) # debug shuffle
        # bev_feats = torch.cat([geom_bev_feats, prior], dim=1)   # (B, 128+3, Dx, Dy)

        # # Method G: first train the no bev feats part, then refine with topdown depth
        # stage = 0 # 0 or 1
        # q = self.query_to_prior(query)                      # (B, 128*4*4)
        # q = q.view(B, 128, 4, 4)                # (B, 256, 4, 4)
        # q = self.prior_upsampler(q)                       # (B, 128, 32, 32)
        # prior_low_res = self.prior_proj(q)                     # (B, 128, 32, 32)
        # prior = F.interpolate(prior_low_res, size=bev_feats.shape[-2:], mode="nearest")  # (B, 128, Dx, Dy)
        # occ = (raw_bev_feats.abs().sum(dim=1) > 0).float()   # (B,Dx,Dy)
        # geom_bev_feats = torch.stack([occ, obs_masks[:,0], obs_masks[:,1]], dim=1)
        # # rand_idx = torch.randperm(self.Dx * self.Dy, device=bev_feats.device)
        # # geom_bev_feats = geom_bev_feats.view(B, 3, self.Dx * self.Dy)[:, :, rand_idx].view(B, 3, self.Dx, self.Dy) # debug shuffle
        # # bev_feats = torch.cat([geom_bev_feats, prior], dim=1)   # (B, 128+3, Dx, Dy)
        # bev_feats = prior

        # rand_idx = torch.randperm(self.Dx * self.Dy, device=bev_feats.device)
        # bev_feats = bev_feats.view(B, bev_feats.shape[1], self.Dx * self.Dy)[:, :, rand_idx].view(B, bev_feats.shape[1], self.Dx, self.Dy) # debug shuffle
        # stem, c2, c3, c4, c5 = self.bev_encoder(bev_feats)  # stem & c2: B, 128, 64, 64; c5: B, 1024, 8, 8

        # # Method C: spatial film inside fpn
        # stem, c2, c3, c4, c5 = self.bev_encoder(bev_feats)  # stem & c2: B, 128, 64, 64; c5: B, 1024, 8, 8
        # # ---- build multi-scale prior maps from query ----
        # p16 = self.query_to_prior16(query).view(B, 32, 16, 16)   # (B, prior_ch, 16, 16)
        # p32 = self.prior_up_32(p16)                                      # (B, prior_ch, 32, 32)
        # p64 = self.prior_up_64(p32)                                      # (B, prior_ch, 64, 64)
        # p128 = self.prior_up_128(p64)                                      # (B, prior_ch, 128, 128)
        # # ---- spatial FiLM on C2..C5 ----
        # g2, b2 = self.film2(p128).chunk(2, dim=1)  # (B, c2_ch, 128, 128)
        # g3, b3 = self.film3(p64).chunk(2, dim=1)  # (B, c3_ch, 64, 64)
        # g4, b4 = self.film4(p32).chunk(2, dim=1)  # (B, c4_ch, 32, 32)
        # g5, b5 = self.film5(p16).chunk(2, dim=1) # (B, c5_ch, 16, 16)
        # c2 = self.norm2(c2) * (1.0 + g2) + b2
        # c3 = self.norm3(c3) * (1.0 + g3) + b3
        # c4 = self.norm4(c4) * (1.0 + g4) + b4
        # c5 = self.norm5(c5) * (1.0 + g5) + b5


        # p2, p3, p4, p5 = self.bev_fpn(c2, c3, c4, c5)
        # p3u = F.interpolate(p3, size=p2.shape[-2:], mode="nearest")
        # p4u = F.interpolate(p4, size=p2.shape[-2:], mode="nearest")
        # p5u = F.interpolate(p5, size=p2.shape[-2:], mode="nearest")
        # fpn_out = p2 + p3u + p4u + p5u
        # fpn_out = self.bevfpn_fuse(fpn_out) # B, bev_head_channels, 128, 128

        # aff_logits = self.affordance_head(fpn_out).squeeze(1)  # (B, Dx, Dy)

        ################### begin of ablation options ###################

        if self.variant == 'filmbev':
            # Ablation Method 0: Method C: spatial film inside fpn
            stem, c2, c3, c4, c5 = self.bev_encoder(bev_feats)  # stem & c2: B, 128, 64, 64; c5: B, 1024, 8, 8
            # ---- build multi-scale prior maps from query ----
            p16 = self.query_to_prior16(query).view(B, 32, 16, 16)   # (B, prior_ch, 16, 16)
            p32 = self.prior_up_32(p16)                                      # (B, prior_ch, 32, 32)
            p64 = self.prior_up_64(p32)                                      # (B, prior_ch, 64, 64)
            p128 = self.prior_up_128(p64)                                      # (B, prior_ch, 128, 128)
            # ---- spatial FiLM on C2..C5 ----
            g2, b2 = self.film2(p128).chunk(2, dim=1)  # (B, c2_ch, 128, 128)
            g3, b3 = self.film3(p64).chunk(2, dim=1)  # (B, c3_ch, 64, 64)
            g4, b4 = self.film4(p32).chunk(2, dim=1)  # (B, c4_ch, 32, 32)
            g5, b5 = self.film5(p16).chunk(2, dim=1) # (B, c5_ch, 16, 16)
            c2 = self.norm2(c2) * (1.0 + g2) + b2
            c3 = self.norm3(c3) * (1.0 + g3) + b3
            c4 = self.norm4(c4) * (1.0 + g4) + b4
            c5 = self.norm5(c5) * (1.0 + g5) + b5


            p2, p3, p4, p5 = self.bev_fpn(c2, c3, c4, c5)
            p3u = F.interpolate(p3, size=p2.shape[-2:], mode="nearest")
            p4u = F.interpolate(p4, size=p2.shape[-2:], mode="nearest")
            p5u = F.interpolate(p5, size=p2.shape[-2:], mode="nearest")
            fpn_out = p2 + p3u + p4u + p5u
            fpn_out = self.bevfpn_fuse(fpn_out) # B, bev_head_channels, 128, 128

            aff_logits = self.affordance_head(fpn_out).squeeze(1)  # (B, Dx, Dy)
            return aff_logits

        if self.variant == 'plainbev':
            # Method ablation 1: bev output but no fusion
            p16 = self.query_to_prior16(query).view(B, 32, 16, 16)   # (B, prior_ch, 16, 16)
            p32 = self.prior_up_32(p16)                                      # (B, prior_ch, 32, 32)
            p64 = self.prior_up_64(p32)                                      # (B, prior_ch, 64, 64)
            p128 = self.prior_up_128(p64)                                      # (B, prior_ch, 128, 128)
            bev_feats = p128
            stem, c2, c3, c4, c5 = self.bev_encoder(bev_feats) # stem & c2: B, 128, 64, 64; c5: B, 1024, 8, 8
            p2, p3, p4, p5 = self.bev_fpn(c2, c3, c4, c5)
            p3u = F.interpolate(p3, size=p2.shape[-2:], mode="nearest")
            p4u = F.interpolate(p4, size=p2.shape[-2:], mode="nearest")
            p5u = F.interpolate(p5, size=p2.shape[-2:], mode="nearest")
            fpn_out = p2 + p3u + p4u + p5u
            fpn_out = self.bevfpn_fuse(fpn_out) # B, bev_head_channels, 128, 128
            aff_logits = self.affordance_head(fpn_out).squeeze(1)  # (B, Dx, Dy)
            return aff_logits

        # Method ablation 2: cat before resnet
        # ---- build multi-scale prior maps from query ----
        if self.variant == 'catbev':
            p16 = self.query_to_prior16(query).view(B, 32, 16, 16)   # (B, prior_ch, 16, 16)
            p32 = self.prior_up_32(p16)                                      # (B, prior_ch, 32, 32)
            p64 = self.prior_up_64(p32)                                      # (B, prior_ch, 64, 64)
            p128 = self.prior_up_128(p64)                                      # (B, prior_ch, 128, 128)

            bev_feats = torch.cat([bev_feats, p128], dim=1)  # (B, C + prior_ch, Dx, Dy)
            stem, c2, c3, c4, c5 = self.bev_encoder(bev_feats)  # stem & c2: B, 128, 64, 64; c5: B, 1024, 8, 8
            p2, p3, p4, p5 = self.bev_fpn(c2, c3, c4, c5)
            p3u = F.interpolate(p3, size=p2.shape[-2:], mode="nearest")
            p4u = F.interpolate(p4, size=p2.shape[-2:], mode="nearest")
            p5u = F.interpolate(p5, size=p2.shape[-2:], mode="nearest")
            fpn_out = p2 + p3u + p4u + p5u
            fpn_out = self.bevfpn_fuse(fpn_out) # B, bev_head_channels, 128, 128
            aff_logits = self.affordance_head(fpn_out).squeeze(1)  # (B, Dx, Dy)
            return aff_logits

        # Method ablation 3: cross attn to bev
        if self.variant == 'attnxy':
            B = query.shape[0]
            bev_feats_1d = bev_feats.flatten(2).permute(0, 2, 1).contiguous()
            prior_masks_1d = prior_masks.flatten(2).permute(0, 2, 1).contiguous()
            empty_mask_1d = (prior_masks_1d.abs().sum(dim=2) < 1e-6)  # (B, Dx*Dy) bool, True means ignore in attn
            bev_pos = self.positional_encoding.unsqueeze(0)
            q_state = self.query_in_proj(query).unsqueeze(1)
            for i in range(self.num_cross_layers):
                qn = self.cross_ln_q[i](q_state)
                kvn = self.cross_ln_kv[i](bev_feats_1d)
                k = kvn + bev_pos
                v = kvn + bev_pos  # set v = kvn for "pos only on K"
                attn_out, _ = self.cross_mha[i](query=qn, key=k, value=v, key_padding_mask=empty_mask_1d, need_weights=False)
                q_state = q_state + attn_out
                q_state = q_state + self.cross_ffn[i](self.cross_ln_ffn[i](q_state))
            q_state = q_state.squeeze(1)
            refined_2048 = query + self.query_out_proj(q_state)
            pred_xy = self.bev_xy_head(refined_2048)
            return pred_xy

        if self.variant == 'catattnbev':
            # Method ablation 4: cat before attn
            p16 = self.query_to_prior16(query).view(B, 32, 16, 16)   # (B, prior_ch, 16, 16)
            p32 = self.prior_up_32(p16)                                      # (B, prior_ch, 32, 32)
            p64 = self.prior_up_64(p32)                                      # (B, prior_ch, 64, 64)
            bev_feats_ds = F.interpolate(bev_feats, size=(64,64), mode="bilinear", align_corners=False)  # (B, C, 64, 64)
            valid_128 = (prior_masks.abs().sum(dim=1, keepdim=True) > 0).to(prior_masks.dtype)          # (B, 1, 128, 128)
            # valid_128 = torch.ones((B, 1, 128, 128), device=bev_feats.device, dtype=bev_feats.dtype) # debug
            valid_64 = F.max_pool2d(valid_128, kernel_size=2, stride=2)     # if any valid then valid, (B, 1, 64, 64)

            # # debug
            # save_path = f'debug_outputs/aff_attn_bev.npz'
            # import numpy as np
            # traversable_mask = kwargs['traversable_mask'].detach().cpu().numpy()  # (B,Dx,Dy)
            # visible_mask     = kwargs['visible_mask'].detach().cpu().numpy()
            # affordance_mask  = kwargs['affordance_mask'].detach().cpu().numpy()
            # np.savez_compressed(
            #     save_path,
            #     valid64 = valid_64.float().detach().cpu().numpy(),  # (B,1,64,64)
            #     valid128 = valid_128.float().detach().cpu().numpy(), # (B,1,128,128)
            #     vis = visible_mask,  # (B,Dx,Dy)
            #     trav = traversable_mask,  # (B,Dx,Dy)
            #     aff = affordance_mask,  # (B,Dx,Dy)
            # )
            # exit(0)
            # # end of debug
            
            attn_bev_feats = torch.cat([bev_feats_ds, p64], dim=1)  # (B, 512+32, 64, 64)
            attn_bev_feats_1d = attn_bev_feats.flatten(2).permute(0, 2, 1).contiguous()
            # Hard feature mask + focus attention on valid area only:
            # gather valid tokens -> self-attn only on them -> scatter back; invalid stays 0 in feature space.
            valid_64_1d = (valid_64.squeeze(1) >= 0.5).flatten(1)   # (B, 4096) bool
            lengths = valid_64_1d.sum(dim=1).to(torch.int64)        # (B,)
            assert lengths.min().item() > 0, "No valid BEV tokens at 64x64; check mask generation."
            Lmax = int(lengths.max().item())

            C_attn = attn_bev_feats_1d.shape[-1]
            x_pack = attn_bev_feats_1d.new_zeros((B, Lmax, C_attn))
            pos_pack = attn_bev_feats_1d.new_zeros((B, Lmax, C_attn))
            pad_mask = torch.ones((B, Lmax), device=attn_bev_feats_1d.device, dtype=torch.bool)  # True means padded

            pos_1d = self.positional_encoding  # (4096, C_attn)
            for b in range(B):
                idx = torch.nonzero(valid_64_1d[b], as_tuple=False).squeeze(1)
                Lb = int(idx.numel())
                x_pack[b, :Lb] = attn_bev_feats_1d[b, idx]
                pos_pack[b, :Lb] = pos_1d[idx].to(dtype=attn_bev_feats_1d.dtype, device=attn_bev_feats_1d.device)
                pad_mask[b, :Lb] = False

            for i in range(self.num_cross_layers):
                x = x_pack + pos_pack
                residual = x_pack
                x_normed = self.cross_ln_qkv[i](x)
                attn_out, _ = self.cross_mha[i](
                    query=x_normed,
                    key=x_normed,
                    value=x_normed,
                    key_padding_mask=pad_mask,
                    need_weights=False,
                )
                x = residual + attn_out
                residual = x
                x_normed = self.cross_ln_ffn[i](x)
                ffn_out = self.cross_ffn[i](x_normed)
                x = residual + ffn_out
                x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
                x_pack = x   # for the next layer

            attn_bev_feats_1d_out = attn_bev_feats_1d.new_zeros(attn_bev_feats_1d.shape)
            for b in range(B):
                idx = torch.nonzero(valid_64_1d[b], as_tuple=False).squeeze(1)
                Lb = int(idx.numel())
                attn_bev_feats_1d_out[b, idx] = x_pack[b, :Lb]

            attn_bev_feats = attn_bev_feats_1d_out.permute(0, 2, 1).contiguous().view(B, self.attn_dim, 64, 64)
            attn_bev_feats_up = F.interpolate(attn_bev_feats, size=(128, 128), mode="nearest")  # (B, C_attn, 128, 128)
            attn_bev_feats_up = attn_bev_feats_up * (valid_128 >= 0.5).to(dtype=attn_bev_feats_up.dtype)
            # note: make fpn away makes it worse
            stem, c2, c3, c4, c5 = self.bev_encoder(attn_bev_feats_up)  # stem & c2: B, 128, 64, 64; c5: B, 1024, 8, 8
            p2, p3, p4, p5 = self.bev_fpn(c2, c3, c4, c5)
            p3u = F.interpolate(p3, size=p2.shape[-2:], mode="nearest")
            p4u = F.interpolate(p4, size=p2.shape[-2:], mode="nearest")
            p5u = F.interpolate(p5, size=p2.shape[-2:], mode="nearest")
            fpn_out = p2 + p3u + p4u + p5u
            fpn_out = self.bevfpn_fuse(fpn_out) # B, bev_head_channels, 128, 128
            aff_logits = self.affordance_head(fpn_out).squeeze(1)  # (B, Dx, Dy)
            # aff_logits = self.affordance_head_2(attn_bev_feats_up).squeeze(1)  # (B, Dx, Dy)

            return aff_logits

        # # Method ablation 5: cat before resnet and diffusion
        # # first quick check if setup works
        # uncond_diffusion = True
        # if uncond_diffusion:   # however you gate it (flag/config)
        #     if self.training:
        #         x0 = kwargs['affordance_mask'][:, None, :, :]  # (B,1,Dx,Dy)
        #         B = x0.shape[0]
        #         device = x0.device

        #         t = torch.randint(0, self.diff_T, (B,), device=device).long()
        #         noise = torch.randn_like(x0)
        #         x_t = self.diff_train_scheduler.add_noise(x0, noise, t)
        #         eps_pred = self.diff_unet(x_t, t).sample   # (B,1,H,W)

        #         loss = F.mse_loss(eps_pred, noise)
        #         return loss
        #     else:
        #         # sample x0 from noise (optionally start from a noised GT at t0)
        #         aff_gt = kwargs["affordance_mask"][:, None, :, :].to(torch.float32)  # (B,1,Dx,Dy)
        #         B, _, Dx, Dy = aff_gt.shape
        #         device = aff_gt.device

        #         start_from = None  # set to int t0 in [0, diff_T-1] if you want to start from x_t0

        #         # prepare inference timesteps
        #         self.diff_infer_scheduler.set_timesteps(self.diff_infer_steps, device=device)
        #         timesteps = self.diff_infer_scheduler.timesteps  # typically descending

        #         if start_from is None:
        #             x = torch.randn((B, 1, Dx, Dy), device=device, dtype=torch.float32)
        #         else:
        #             assert 0 <= start_from < self.diff_T
        #             t0 = torch.full((B,), start_from, device=device, dtype=torch.long)
        #             noise = torch.randn_like(aff_gt)
        #             x = self.diff_train_scheduler.add_noise(aff_gt, noise, t0).to(torch.float32)
        #             # only run steps for timesteps <= start_from
        #             timesteps = timesteps[timesteps <= start_from]

        #         all_x = []
        #         for t in timesteps:
        #             t_batch = torch.full((B,), int(t), device=device, dtype=torch.long)
        #             with torch.cuda.amp.autocast(enabled=False):
        #                 eps_pred = self.diff_unet(x, t_batch).sample  # uses x float32 here
        #             x = self.diff_infer_scheduler.step(eps_pred.float(), t, x).prev_sample.float()
        #             all_x.append(x.detach().cpu())
                
        #         # debug
        #         import numpy as np
        #         all_x = torch.stack(all_x, dim=0).permute(1,0,2,3,4).squeeze(2).numpy()  # (B, steps, Dx, Dy)
        #         path = f'debug_outputs/diffusion_infer.npz'
        #         np.savez_compressed(path, all_x=all_x)
        #         print(f"[DEBUG] Saved diffusion inference trajectory to {path}")
        #         exit(0)
        #         # end of debug

        #         # return logits for downstream; make probabilities valid first
        #         x_logits = self.prob_to_logits(x)
        #         loss = F.binary_cross_entropy_with_logits(x_logits, aff_gt)
        #         print(f"[Diffusion] Inference loss: {loss.item():.4f}")

        #         return x_logits.squeeze(1)  # (B,Dx,Dy)


        # # ---- build multi-scale prior maps from query ----
        # p16 = self.query_to_prior16(query).view(B, 32, 16, 16)   # (B, prior_ch, 16, 16)
        # p32 = self.prior_up_32(p16)                                      # (B, prior_ch, 32, 32)
        # p64 = self.prior_up_64(p32)                                      # (B, prior_ch, 64, 64)
        # p128 = self.prior_up_128(p64)                                      # (B, prior_ch, 128, 128)

        # bev_feats = torch.cat([bev_feats, p128], dim=1)  # (B, C + prior_ch, Dx, Dy)
        # stem, c2, c3, c4, c5 = self.bev_encoder(bev_feats)  # stem & c2: B, 128, 64, 64; c5: B, 1024, 8, 8
        # p2, p3, p4, p5 = self.bev_fpn(c2, c3, c4, c5)
        # p3u = F.interpolate(p3, size=p2.shape[-2:], mode="nearest")
        # p4u = F.interpolate(p4, size=p2.shape[-2:], mode="nearest")
        # p5u = F.interpolate(p5, size=p2.shape[-2:], mode="nearest")
        # fpn_out = p2 + p3u + p4u + p5u
        # fpn_out = self.bevfpn_fuse(fpn_out) # B, bev_head_channels, 128, 128
        # aff_logits = self.affordance_head(fpn_out).squeeze(1)  # (B, Dx, Dy)

        if self.variant == 'catlss':
            # Method ablation 6: cat before resnet, only lss
            # ---- build multi-scale prior maps from query ----
            img_feats = x.view(B, N, C, H, W).permute(0, 1, 3, 4, 2).contiguous() # (B, N, H, W, C)
            bev_feats = self.prepare_bev_feats_deterministic_with_obs_v2( 
                depth=depth,
                img_feats=img_feats,
                sensor2egos=sensor2egos,
                intrinsics=intrinsics,
                blocker_z_range=(-1.4, -0.1),
            )  # (B, C+m, Dx, Dy) = B, proj_c+m, 128, 128
            bev_feats = bev_feats[:, :-3, :, :]  # remove obs masks to ablate, (B, C, Dx, Dy)
            p16 = self.query_to_prior16(query).view(B, 32, 16, 16)   # (B, prior_ch, 16, 16)
            p32 = self.prior_up_32(p16)                                      # (B, prior_ch, 32, 32)
            p64 = self.prior_up_64(p32)                                      # (B, prior_ch, 64, 64)
            p128 = self.prior_up_128(p64)                                      # (B, prior_ch, 128, 128)
            bev_feats = torch.cat([bev_feats, p128], dim=1)  # (B, C + prior_ch, Dx, Dy)
            stem, c2, c3, c4, c5 = self.bev_encoder(bev_feats)  # stem & c2: B, 128, 64, 64; c5: B, 1024, 8, 8
            p2, p3, p4, p5 = self.bev_fpn(c2, c3, c4, c5)
            p3u = F.interpolate(p3, size=p2.shape[-2:], mode="nearest")
            p4u = F.interpolate(p4, size=p2.shape[-2:], mode="nearest")
            p5u = F.interpolate(p5, size=p2.shape[-2:], mode="nearest")
            fpn_out = p2 + p3u + p4u + p5u
            fpn_out = self.bevfpn_fuse(fpn_out) # B, bev_head_channels, 128, 128
            aff_logits = self.affordance_head(fpn_out).squeeze(1)  # (B, Dx, Dy)
            return aff_logits

        if self.variant == 'catpoint':
            # Method ablation 7: cat before resnet, only pointcloud
            # ---- build multi-scale prior maps from query ----
            pointcloud_bev_feats = self.prepare_bev_feats_second(
                    depth=depth,
                    sensor2egos=sensor2egos,
                    intrinsics=intrinsics,
                    pointcloud_config_z=(-1.6, 0.32, 0.12)
            )  # (B, 256, Dx, Dy) = B, 256, 128, 128
            bev_feats = pointcloud_bev_feats
            p16 = self.query_to_prior16(query).view(B, 32, 16, 16)   # (B, prior_ch, 16, 16)
            p32 = self.prior_up_32(p16)                                      # (B, prior_ch, 32, 32)
            p64 = self.prior_up_64(p32)                                      # (B, prior_ch, 64, 64)
            p128 = self.prior_up_128(p64)                                      # (B, prior_ch, 128, 128)
            bev_feats = torch.cat([bev_feats, p128], dim=1)  # (B, C + prior_ch, Dx, Dy)
            stem, c2, c3, c4, c5 = self.bev_encoder(bev_feats)  # stem & c2: B, 128, 64, 64; c5: B, 1024, 8, 8
            p2, p3, p4, p5 = self.bev_fpn(c2, c3, c4, c5)
            p3u = F.interpolate(p3, size=p2.shape[-2:], mode="nearest")
            p4u = F.interpolate(p4, size=p2.shape[-2:], mode="nearest")
            p5u = F.interpolate(p5, size=p2.shape[-2:], mode="nearest")
            fpn_out = p2 + p3u + p4u + p5u
            fpn_out = self.bevfpn_fuse(fpn_out) # B, bev_head_channels, 128, 128
            aff_logits = self.affordance_head(fpn_out).squeeze(1)  # (B, Dx, Dy)
            return aff_logits

        if self.variant == 'catnogate':
            # Method ablation 8: cat before resnet, no gate
            img_feats = x.view(B, N, C, H, W).permute(0, 1, 3, 4, 2).contiguous() # (B, N, H, W, C)
            bev_feats = self.prepare_bev_feats_deterministic_with_obs_v2( # for Method C.2
                depth=depth,
                img_feats=img_feats,
                sensor2egos=sensor2egos,
                intrinsics=intrinsics,
                blocker_z_range=(-1.4, -0.1),
            )  # (B, C+m, Dx, Dy) = B, proj_c+m, 128, 128
            pointcloud_bev_feats = self.prepare_bev_feats_second(
                    depth=depth,
                    sensor2egos=sensor2egos,
                    intrinsics=intrinsics,
                    pointcloud_config_z=(-1.6, 0.32, 0.12)
            )  # (B, 256, Dx, Dy) = B, 256, 128, 128
            bev_feats = torch.cat([bev_feats, pointcloud_bev_feats], dim=1)  # (B, 256+3+256, Dx, Dy)
            bev_feats = self.obs_proj_bevfusion(bev_feats) # (B, 512, Dx, Dy)
            p16 = self.query_to_prior16(query).view(B, 32, 16, 16)   # (B, prior_ch, 16, 16)
            p32 = self.prior_up_32(p16)                                      # (B, prior_ch, 32, 32)
            p64 = self.prior_up_64(p32)                                      # (B, prior_ch, 64, 64)
            p128 = self.prior_up_128(p64)                                      # (B, prior_ch, 128, 128)
            bev_feats = torch.cat([bev_feats, p128], dim=1)  # (B, C + prior_ch, Dx, Dy)
            stem, c2, c3, c4, c5 = self.bev_encoder(bev_feats)  # stem & c2: B, 128, 64, 64; c5: B, 1024, 8, 8
            p2, p3, p4, p5 = self.bev_fpn(c2, c3, c4, c5)
            p3u = F.interpolate(p3, size=p2.shape[-2:], mode="nearest")
            p4u = F.interpolate(p4, size=p2.shape[-2:], mode="nearest")
            p5u = F.interpolate(p5, size=p2.shape[-2:], mode="nearest")
            fpn_out = p2 + p3u + p4u + p5u
            fpn_out = self.bevfpn_fuse(fpn_out) # B, bev_head_channels, 128, 128
            aff_logits = self.affordance_head(fpn_out).squeeze(1)  # (B, Dx, Dy)
            return aff_logits

        # Method ablation 9: directly output point
        if self.variant == 'mlpxy':
            pred_xy = self.bev_xy_head(query)
            return pred_xy

        # # Method ablation 10: directly output point with diffusion
        # mean = query.new_tensor(self.data_mean)  # (2,)
        # std  = query.new_tensor(self.data_std)   # (2,)

        # # ---- disable autocast for diffusion head ----
        # with torch.autocast(device_type="cuda", enabled=False):
        #     query_fp32 = query.float()
        #     mean_fp32 = mean.float()
        #     std_fp32  = std.float()

        #     if self.training:
        #         gt_xy = kwargs["gt_xy"].to(query.device)   # (B,2)
        #         gt_xy = ((gt_xy - mean_fp32) / std_fp32).float()
        #         loss, _ = self.bev_xy_head.loss(obs_embed=query_fp32, actions_xy=gt_xy)
        #         return loss

        #     g = torch.Generator(device=query.device).manual_seed(self.eval_seed)
        #     gt_xy = kwargs["gt_xy"].to(query.device)   # (B,2)
        #     gt_xy = ((gt_xy - mean_fp32) / std_fp32).float()
        #     # xy, dbg = self.bev_xy_head.predict_action(
        #     xy, dbg = self.bev_xy_head.predict_action_from_gt_debug(
        #         # obs_embed=query_fp32, sample_shape=(self.eval_K,), generator=g
        #         obs_embed=query_fp32, actions_xy_gt=gt_xy, sample_shape=(self.eval_K,), generator=g
        #     )  # (K, B, 2) in fp32

        # # back outside autocast-disabled block (rest of model can stay bf16)
        # xy = xy.permute(1, 0, 2).contiguous()   # (B, K, 2)
        # xy = xy * std.float() + mean.float()    # de-normalize in fp32
        # xy = xy.to(query.dtype)                 # optional: match rest of pipeline dtype

        # # # debug
        # # save_path = f'debug_outputs/octo_diff.npz'
        # # print(f"[DEBUG] the scores: {dbg}")
        # # import numpy as np
        # # np.savez_compressed(
        # #     save_path,
        # #     mean=mean.cpu().numpy(),
        # #     std=std.cpu().numpy(),
        # #     pred_xy=xy.squeeze(1).cpu().numpy(),  # (B, 2)
        # #     gt_xy=kwargs["gt_xy"].cpu().numpy(),  # (B, 2)
        # #     scores=dbg
        # # )
        # # exit(0)
        # # # end of debug

        # return xy

        # # Method ablation 11: deformable attn
        # B = query.shape[0]
        # device = query.device
        # R = 6.4
        # span_m = 2.0 * R  # 12.8
        # # stem, c2, c3, c4, c5 = self.bev_encoder(bev_feats)
        # # p2, p3, p4, p5 = self.bev_fpn(c2, c3, c4, c5)
        # stem, c2, c3, c4, c5 = checkpoint(
        #     lambda x: self.bev_encoder(x),
        #     bev_feats,
        #     use_reentrant=False
        # )
        # p2, p3, p4, p5 = checkpoint(
        #     lambda a, b, c, d: self.bev_fpn(a, b, c, d),
        #     c2, c3, c4, c5,
        #     use_reentrant=False
        # )
        # levels = [p2, p3, p4]
        # lvl_vecs = self.lvl_embed(torch.arange(len(levels), device=device))  # (L,d)
        # levels = [levels[i] + lvl_vecs[i].view(1,-1,1,1) for i in range(len(levels))]
        # # flatten 
        # spatial_shapes = []
        # flats = []
        # for feat in levels:
        #     assert feat.dim() == 4
        #     b, c, h, w = feat.shape
        #     assert b == B and c == self.attn_dim
        #     spatial_shapes.append([h, w])
        #     flats.append(feat.flatten(2).transpose(1, 2).contiguous())  # (B, HW, C)
        # spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=levels[0].device)  # (L,2)
        # lvl_lens = spatial_shapes[:, 0] * spatial_shapes[:, 1]
        # level_start_index = torch.cat([lvl_lens.new_zeros((1,)), lvl_lens.cumsum(0)[:-1]], dim=0)  # (L,)
        # value = torch.cat(flats, dim=1)  # (B, sum(HW), C)
        # # vlm_tok = self.query_in_proj(query)  # (B,d)
        # anchor_emb = self.anchor_content(torch.arange(self.num_anchors, device=device))  # (K,d)
        # vlm_tok = torch.zeros_like(anchor_emb[0:1, :]).expand(B, -1)  # (B,d), fake placeholder
        # all_tok = torch.cat([vlm_tok.unsqueeze(1), anchor_emb.unsqueeze(0).repeat(B,1,1)], dim=1)
        # pos = self.token_pos(torch.arange(self.num_tokens, device=device))  # (1+K, d)
        # all_tok = all_tok + pos.unsqueeze(0)  # (B, 1+K, d)
        # anchor_xy01 = self.anchor_xy01.to(device=device)  # (K,2)
        # # ref = anchor_xy01.unsqueeze(0).unsqueeze(2).repeat(B, 1, spatial_shapes.shape[0], 1)  # (B,K,3,2)
        
        # scores_list = []
        # pred_xy01_list = []
        # ref_xy01 = anchor_xy01.unsqueeze(0).repeat(B, 1, 1) # differ per layer
        # gamma_beta = self.lang_film(query)         # (B,2d)
        # gamma, beta = gamma_beta.chunk(2, dim=-1)    # (B,d),(B,d)
        # gamma = gamma.unsqueeze(1)                   # (B,1,d)
        # beta  = beta.unsqueeze(1)
        # for i in range(self.num_layers):
        #     # # self attn
        #     # res = all_tok
        #     # xn = self.self_ln[i](all_tok)
        #     # x = res + self.self_mha[i](xn, xn, xn, need_weights=False)[0]
        #     # film
        #     t_anchor = all_tok[:, 1:, :]  # (B,K,d)
        #     vlm_tok = all_tok[:, 0:1, :]  # (B,1,d)
        #     t_anchor = t_anchor * (1.0 + gamma) + beta   # (B,K,d)
        #     x = torch.cat([vlm_tok, t_anchor], dim=1)  # (B, 1+K, d)
            
        #     # cross attn
        #     t_vlm = x[:, 0:1]  # (B,1,d)
        #     t_anchor = x[:, 1:]  # (B,K,d)
        #     res = t_anchor
        #     ref = ref_xy01.unsqueeze(2).repeat(1, 1, spatial_shapes.shape[0], 1)  # (B,K,3,2)
        #     with torch.autocast(device_type="cuda", enabled=False):
        #         cross_attn_out = self.msdeform[i](
        #             query=self.cross_ln_q[i](t_anchor).float(),
        #             value=self.cross_ln_v[i](value).float(),
        #             reference_points=ref.float(),
        #             spatial_shapes=spatial_shapes,
        #             level_start_index=level_start_index,
        #         )
        #     cross_attn_out = cross_attn_out.to(res.dtype)
        #     t_anchor = res + cross_attn_out
        #     x = torch.cat([t_vlm, t_anchor], dim=1)
        #     # ffn
        #     res = x
        #     x = res + self.msdeform_ffn[i](self.msdeform_ffn_ln[i](x))
        #     # heads
        #     t_anchor = x[:, 1:, :]  # (B,K,d)
        #     scores = self.anchor_score_head(t_anchor).squeeze(-1)  # (B,K) logits
        #     offset = self.anchor_offset_head(t_anchor.float())  # fp32
        #     delta_m = torch.tanh(offset) * float(self.offset_bound_m[i])
        #     delta_01 = delta_m / float(span_m)
        #     pred_xy01 = (ref_xy01 + delta_01).clamp(0.0, 1.0)
        #     ref_xy01 = pred_xy01.detach()

        #     scores_list.append(scores)
        #     pred_xy01_list.append(pred_xy01)
        #     all_tok = x  # for the next layer
        
        # if self.training:
        #     gt_xy = kwargs["gt_xy"].to(device=device)  # (B,2) meters
        #     # choose k* by final layer distance (meters)
        #     # pred_final01 = pred_xy01_list[-1] # shape: (B,K,2) in 01 space
        #     # hack: we use the oracle selection following previous works
        #     pred_final01 = self.anchor_xy01.unsqueeze(0).repeat(B, 1, 1)  # (B,K,2)
        #     pred_final_m = (pred_final01 - 0.5) * span_m
        #     dist2 = ((pred_final_m - gt_xy[:, None, :]) ** 2).sum(dim=-1)  # (B,K)
        #     k_star = dist2.argmin(dim=1)  # (B,)
        #     loss_cls_pos = 0.0
        #     loss_cls_neg = 0.0
        #     loss_reg = 0.0
        #     L = len(scores_list)

        #     for t in range(L):
        #         logits = scores_list[t]       # (B,K)
        #         pred01 = pred_xy01_list[t]    # (B,K,2)
        #         pred_m = (pred01 - 0.5) * float(span_m)
        #         y = torch.zeros_like(logits)
        #         y[torch.arange(B, device=device), k_star] = 1.0
        #         # bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")  # (B,K)
        #         bce = self.py_sigmoid_focal_loss(logits, y, reduction="none")  # (B,K)
        #         pos_mask = y > 0.5
        #         neg_mask = ~pos_mask
        #         loss_cls_pos = loss_cls_pos + bce[pos_mask].mean()
        #         loss_cls_neg = loss_cls_neg + bce[neg_mask].mean()
        #         pred_pos = pred_m[torch.arange(B, device=device), k_star]  # (B,2)
        #         loss_reg = loss_reg + F.smooth_l1_loss(pred_pos, gt_xy, reduction="mean")

        #     loss_cls_pos = loss_cls_pos / L
        #     loss_cls_neg = loss_cls_neg / L
        #     loss_reg = loss_reg / L

        #     return torch.stack([loss_cls_pos, loss_cls_neg, loss_reg], dim=0)  # (3,)
        # else:
        #     scores_final = scores_list[-1]
        #     pred_final01 = pred_xy01_list[-1]
        #     best = scores_final.argmax(dim=1)  # (B,)
        #     pred_best01 = pred_final01[torch.arange(B, device=device), best]  # (B,2)
        #     pred_best_m = (pred_best01 - 0.5) * float(span_m)  # meters, ego-centered
        #     return pred_best_m


        # # Method ablation 12: direct waypoint
        # pred_xy = self.bev_xy_head(query)
        # return pred_xy

        # # Method ablate 13: mask based waypoint
        # # ---- build multi-scale prior maps from query ----
        # with torch.autocast(device_type="cuda", enabled=False):
        #     query = query.float() # (B, query_dim)
        #     bev_feats = bev_feats.float() # (B, C, Dx, Dy)
        #     p16 = self.query_to_prior16(query).view(B, 32, 16, 16)   # (B, prior_ch, 16, 16)
        #     p32 = self.prior_up_32(p16)                                      # (B, prior_ch, 32, 32)
        #     p64 = self.prior_up_64(p32)                                      # (B, prior_ch, 64, 64)
        #     p128 = self.prior_up_128(p64)                                      # (B, prior_ch, 128, 128)

        #     bev_feats = torch.cat([bev_feats, p128], dim=1)  # (B, C + prior_ch, Dx, Dy)
        #     stem, c2, c3, c4, c5 = self.bev_encoder(bev_feats)  # stem & c2: B, 128, 64, 64; c5: B, 1024, 8, 8
        #     p2, p3, p4, p5 = self.bev_fpn(c2, c3, c4, c5)
        #     p3u = F.interpolate(p3, size=p2.shape[-2:], mode="nearest")
        #     p4u = F.interpolate(p4, size=p2.shape[-2:], mode="nearest")
        #     p5u = F.interpolate(p5, size=p2.shape[-2:], mode="nearest")
        #     fpn_out = p2 + p3u + p4u + p5u
        #     fpn_out = self.bevfpn_fuse(fpn_out) # B, bev_head_channels, 128, 128
        #     aff_logits = self.wp_mask_head(fpn_out) # (B, num_wp, Dx, Dy)

        if self.variant == 'catvisbev':
            # Method ablation 14: cat before resnet
            # ---- build multi-scale prior maps from query ----
            assert context_bev_feats is not None, 'need visual patches for this variant'
            p16 = self.query_to_prior16(query).view(B, 32, 16, 16)   # (B, prior_ch, 16, 16)
            p32 = self.prior_up_32(p16)                                      # (B, prior_ch, 32, 32)
            p64 = self.prior_up_64(p32)                                      # (B, prior_ch, 64, 64)
            p128 = self.prior_up_128(p64)                                      # (B, prior_ch, 128, 128)

            bev_feats = torch.cat([bev_feats, p128, context_bev_feats], dim=1)  # (B, C + prior_ch+32, Dx, Dy)
            stem, c2, c3, c4, c5 = self.bev_encoder(bev_feats)  # stem & c2: B, 128, 64, 64; c5: B, 1024, 8, 8
            p2, p3, p4, p5 = self.bev_fpn(c2, c3, c4, c5)
            p3u = F.interpolate(p3, size=p2.shape[-2:], mode="nearest")
            p4u = F.interpolate(p4, size=p2.shape[-2:], mode="nearest")
            p5u = F.interpolate(p5, size=p2.shape[-2:], mode="nearest")
            fpn_out = p2 + p3u + p4u + p5u
            fpn_out = self.bevfpn_fuse(fpn_out) # B, bev_head_channels, 128, 128
            aff_logits = self.affordance_head(fpn_out).squeeze(1)  # (B, Dx, Dy)


        ################### end of ablation options ###################

        # # Method D: query only debug
        # x = self.query_to_prior16(query)  # (B, 32*16*16)
        # x = x.view(B, 32, 16, 16)         # (B, 32, 16, 16)
        # x = self.prior_up_32(x)               # (B, 32, 32, 32)
        # x = self.prior_up_64(x)               # (B, 32, 64, 64)
        # x = self.prior_up_128(x)               # (B, 32, 128, 128)
        # aff_logits = self.affordance_head_dbg(x).squeeze(1)  # (B, Dx, Dy)

        # # Method E: mask2former
        # vlm_query = self.query_to_attn(query)           # (B, 256)
        # mq = self.mask_query.expand(B, 1, 256)      # if mask_query stored as (1,1,256)
        # vq = vlm_query.unsqueeze(1)                 # (B,1,256)
        # attn_q = torch.cat([mq, vq], dim=1)         # (B,2,256)
        # # TODO: do we need positional embedding here?

        # obs_any = obs_masks.sum(dim=1, keepdim=False).flatten(-2) > 0.5
        # key_padding_mask = ~obs_any  # (B, Dx*Dy), True means ignored

        # num_layers = 1
        # kv = p2.flatten(-2).permute(0, 2, 1) # (B, HW, 256)

        # # debug
        # idx = torch.randperm(kv.size(1), device=kv.device)
        # kv = kv[:, idx, :]               # (B, HW, 256)

        # for _ in range(num_layers):
        #     res1 = attn_q
        #     attn_q = self.token_self_attn(x=attn_q, attn_mask=None)  # (B, 2, 256)
        #     attn_q = self.layer_norm_1(attn_q + res1) # B, 2, 256
        #     res2 = attn_q
        #     attn_q, _ = self.bev_cross_attn(
        #         query=attn_q,
        #         key=kv + self.positional_encoding.unsqueeze(0).to(p2.device),
        #         value=kv,
        #         attn_mask=None,  
        #         key_padding_mask=key_padding_mask, # ablate this
        #     )
        #     attn_q = self.layer_norm_2(attn_q + res2) # B, 2, 256
        #     attn_q = self.token_mlp(attn_q) + attn_q  # B, 2, 256
        #     attn_q = self.layer_norm_3(attn_q)  # B, 2, 256
        #     pix = p2.flatten(-2).permute(0,2,1)  # (B,HW,256)
        #     mask_logits = torch.einsum("bqc,bkc->bqk", attn_q, pix)  # (B,2,HW)
        #     mask_logits = mask_logits[:,0].view(B, self.Dx, self.Dy)
        # aff_logits = mask_logits

        # # Method G: refined 
        # if stage == 1:
        #     stage1_bev_feats = torch.cat([geom_bev_feats, aff_logits.unsqueeze(1).detach()], dim=1)  # (B, 4, Dx, Dy)
        #     refined_stem, r_c2, r_c3, r_c4, r_c5 = self.bev_encoder_2(stage1_bev_feats)
        #     r_p2, r_p3, r_p4, r_p5 = self.bev_fpn_2(r_c2, r_c3, r_c4, r_c5)
        #     r_p3u = F.interpolate(r_p3, size=r_p2.shape[-2:], mode="nearest")
        #     r_p4u = F.interpolate(r_p4, size=r_p2.shape[-2:], mode="nearest")
        #     r_p5u = F.interpolate(r_p5, size=r_p2.shape[-2:], mode="nearest")
        #     r_fpn_out = r_p2 + r_p3u + r_p4u + r_p5u
        #     r_fpn_out = self.bevfpn_fuse_2(r_fpn_out) # B, bev_head_channels, 128, 128
        #     aff_logits = self.affordance_head_2(r_fpn_out).squeeze(1) + aff_logits.detach()  # (B, Dx, Dy)


        # # # debug
        # # import numpy as np
        # # debug_save_path = f'debug_outputs/debug_lss_result_{self.debug_cnt}.jpg'
        # # assert bev_feats.shape[0] == 1, "Debug only supports batch size 1."
        # # bev_np = bev_feats[0, :, :, :].detach().cpu().numpy()  # (C, Dx, Dy)
        # # bev_non_zero = np.abs(bev_np).sum(axis=0)                  # (Dx, Dy)
        # # import cv2
        # # bev_img = cv2.normalize(bev_non_zero, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # # cv2.imwrite(debug_save_path, bev_img)
        # # occ_result = aff_logits[0, :, :].detach().cpu().numpy()
        # # occ_result = np.where(occ_result > 0, 1.0, 0.0).astype(np.uint8) * 255
        # # debug_occ_save_path = f'debug_outputs/debug_occ_result_{self.debug_cnt}.jpg'
        # # cv2.imwrite(debug_occ_save_path, occ_result)
        # # print(f"[DEBUG] Saved LSS BEV features to {debug_save_path} and occ result to {debug_occ_save_path}")
        # # self.debug_cnt += 1
        # # # exit(0)

        return aff_logits # (B, Dx, Dy)
    

# bev affordance with diffusion refinement
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler
from diffusers.models import UNet2DModel
from torch.cuda.amp import autocast


# bev affordance with diffusion from scratch
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler
from diffusers.models import UNet2DModel
from torch.cuda.amp import autocast


    
from mmdet3d.registry import MODELS
from mmengine.model import BaseModel


from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS

    
