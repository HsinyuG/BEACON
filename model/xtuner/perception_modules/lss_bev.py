import torch
import torch.nn as nn
from mmengine.model import BaseModule, BaseModel
from mmdet3d.registry import MODELS
import numpy as np
from collections.abc import Iterable
from xtuner.perception_modules.utils import (
    lss_bev_pool,
)
from einops import rearrange


@MODELS.register_module()
class BEVHeadLSS(BaseModel):
    def __init__(self,
                 backbone,
                 neck,
                 embed_dims=256,
                 head_type='bev',  # 'bev' or 'qformer_img' or 'qformer_bev'
                 **kwargs):
        super().__init__(**kwargs)
        self.head_type = head_type
        self.embed_dims = embed_dims

        if backbone is not None:
            if backbone.type == 'TorchHubModel':
                self.backbone = torch.hub.load(backbone.repo_or_dir,
                                               backbone.model_name)
                self.backbone.requires_grad_(False)
                self.backbone.is_init = True  # otherwise it will be re-inited by mmengine
                self.patch_size = self.backbone.patch_size
            else:
                self.backbone = MODELS.build(backbone)
            self.frozen_backbone = all(not param.requires_grad
                                       for param in self.backbone.parameters())
        self.neck = MODELS.build(neck)

        in_channels_fpn = None
        if self.head_type == 'bev':
            self.bev_head = nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 2, kernel_size=1)
            )
            in_channels_fpn = 256

        if in_channels_fpn is not None:
            self.fpn_fine = nn.Sequential(
                nn.Conv2d(in_channels_fpn, 128, 3, 1, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            self.fpn_coarse = nn.Sequential(
                nn.Conv2d(128, 128, 3, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, 3, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )

    def prepare_inputs(self, inputs_dict, data_samples):
        num_views = data_samples[0].num_views
        inputs = inputs_dict['imgs']

        cam2img = []
        cam2ego = []
        ego2global = []
        img_aug_mat = []
        depth = []
        feats = []
        traversable_mask = []
        for i in range(len(data_samples)):
            data_samples[i].set_metainfo(
                {'cam2img': data_samples[i].cam2img[:num_views]})
            cam2img.append(data_samples[i].cam2img)
            data_samples[i].set_metainfo(
                {'cam2ego': data_samples[i].cam2ego[:num_views]})
            cam2ego.append(data_samples[i].cam2ego)
            ego2global.append(data_samples[i].ego2global)
            if hasattr(data_samples[i], 'img_aug_mat'):
                data_samples[i].set_metainfo(
                    {'img_aug_mat': data_samples[i].img_aug_mat[:num_views]})
                img_aug_mat.append(data_samples[i].img_aug_mat)
            depth.append(data_samples[i].depth)
            if hasattr(data_samples[i], 'feats'):
                feats.append(data_samples[i].feats)
            trav = data_samples[i].get('traversable_mask', None)
            if trav is None:
                raise KeyError('traversable_mask missing in data_samples metainfo')
            traversable_mask.append(trav)

        data_samples = dict(
            depth=depth,
            cam2img=cam2img,
            cam2ego=cam2ego,
            num_views=num_views,
            ego2global=ego2global,
            img_aug_mat=img_aug_mat if img_aug_mat else None,
            traversable_mask=traversable_mask
        )
        if feats:
            data_samples['feats'] = feats

        for k, v in data_samples.items():
            if isinstance(v, torch.Tensor) or not isinstance(v, Iterable):
                continue
            if isinstance(v[0], torch.Tensor):
                data_samples[k] = torch.stack(v).to(inputs)
            else:
                data_samples[k] = torch.from_numpy(np.stack(v)).to(inputs)
        return inputs, data_samples
    
    def prepare_view_transform(self, x, data_samples, depth_cfg=(0.0, 4.8, 0.1)):
        # x is an FPN feature list; pick the level aligned to the ViT patch grid.
        depth = data_samples['depth']  # B, n, H, W
        bs, n, H, W = depth.shape
        patch_H, patch_W = H // self.patch_size, W // self.patch_size

        if isinstance(x, (list, tuple)):
            feat_map = None
            for feat in x:
                if feat.shape[-2] == patch_H and feat.shape[-1] == patch_W:
                    feat_map = feat
                    break
            if feat_map is None:
                feat_map = x[0]
        else:
            feat_map = x

        # reshape features back to (B, n, C, H', W')
        feat_map = feat_map.reshape(bs, n, *feat_map.shape[1:]).contiguous()

        # Average depth within each patch (H' = H/ps, W' = W/ps).
        depth_tiles = torch.nn.functional.unfold(
            depth.flatten(0, 1).unsqueeze(1),
            kernel_size=self.patch_size,
            stride=self.patch_size)
        depth_tiles = depth_tiles.reshape(
            bs * n, self.patch_size * self.patch_size, patch_H, patch_W)
        depth_coarse = depth_tiles.mean(dim=1).reshape(bs, n, patch_H, patch_W)

        depth_min, depth_max, bin_size = depth_cfg
        num_bins = int(round((depth_max - depth_min) / bin_size))
        depth_clamped = depth_coarse.clamp(min=depth_min, max=depth_max - 1e-4)
        bin_idx = torch.div(depth_clamped - depth_min, bin_size, rounding_mode='floor').long()
        bin_idx = bin_idx.clamp(0, num_bins - 1)
        depth_bins = torch.nn.functional.one_hot(bin_idx, num_classes=num_bins) # B, n, H', W', D
        depth_bins = depth_bins.permute(0, 1, 4, 2, 3).float().contiguous()  # B, n, D, H', W'

        K_3x3 = data_samples['cam2img'][:, :, :3, :3]  # B, n, 3, 3

        depth_probs_bn=depth_bins.view(bs * n, num_bins, patch_H, patch_W) # B*n, D, H', W'
        img_feats_bn=feat_map.view(bs * n, feat_map.shape[2], patch_H, patch_W)
        assert depth_probs_bn.is_contiguous(), "depth_bins to lss_bev_pool must be contiguous"
        assert img_feats_bn.is_contiguous(), "feat_map to lss_bev_pool must be contiguous" # --> error here

        return dict(
            depth_probs_bn=depth_probs_bn,
            img_feats_bn=img_feats_bn,
            sensor2egos=data_samples['cam2ego'],
            intrinsics=K_3x3,
        )

    def bev_fpn(self, x):
        fine_feat = self.fpn_fine(x)
        coarse_feat = self.fpn_coarse(fine_feat)
        coarse_feat_upsampled = torch.nn.functional.interpolate(
            coarse_feat, size=fine_feat.shape[-2:], mode='bilinear', align_corners=True)
        fused_feat = torch.cat([fine_feat, coarse_feat_upsampled], dim=1)
        return fused_feat # B, 256, H, W
    
    def forward(self, inputs, data_samples, mode='loss'):
        inputs, data_samples = self.prepare_inputs(inputs, data_samples)

        # # DEBUG
        # import cv2
        # import os
        # debug_dir = "debug_outputs"
        # os.makedirs(debug_dir, exist_ok=True)

        # b = 0
        # imgs = inputs[b]  # (N, C, H, W)
        # depths = data_samples["depth"][b]  # (N, H, W)
        # mask = data_samples["traversable_mask"][b]  # (3, H, W)

        # mean = getattr(self.data_preprocessor, "mean", None)
        # std = getattr(self.data_preprocessor, "std", None)
        # if mean is not None and std is not None:
        #     mean = torch.as_tensor(mean, device=imgs.device).view(-1, 1, 1)
        #     std = torch.as_tensor(std, device=imgs.device).view(-1, 1, 1)

        # img_names = {0: 'front', 1: 'left', 2: 'back', 3: 'right'}
        # for i in range(min(4, imgs.shape[0])):
        #     img = imgs[i].detach().float()
        #     if mean is not None and std is not None:
        #         img = img * std + mean
        #     else:
        #         img = (img - img.min()) / (img.max() - img.min() + 1e-6) * 255.0
        #     img = img.cpu()
        #     img = img.clamp(0, 255).byte().permute(1, 2, 0).numpy()[:, :, ::-1]
        #     cv2.imwrite(os.path.join(debug_dir, f"img_{img_names[i]}.png"), img)

        #     d = depths[i].detach().float().cpu().numpy()
        #     d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        #     d_max = np.percentile(d, 95) if np.any(d > 0) else 1.0
        #     d_vis = (np.clip(d, 0, d_max) / max(d_max, 1e-6) * 255).astype(np.uint8)
        #     cv2.imwrite(os.path.join(debug_dir, f"depth_{img_names[i]}.png"), d_vis)

        # mask_np = mask.detach().cpu().numpy().astype(np.uint8)  # (3,H,W): [trav, sure]
        # trav_np = mask_np[0]                                    # 0/1
        # unk_np  = (mask_np[1] == 0)                             # bool unknown
        # valid_mask = mask_np[2] == 1                                   # bool valid
        # if not valid_mask.all():
        #     print(f"We met skipped sample in debug now, run again to get the debug images.")
        #     exit(0)

        # from xtuner.perception_modules.oracle_bev import BEVMaskGenerator
        # BEVMaskGenerator.save_vis(
        #     traversable_mask=trav_np,
        #     save_path=os.path.join(debug_dir, "debug_bev_mask_raw.png"),
        #     title="Traversable Mask Visualization",
        # )
        # BEVMaskGenerator.save_vis(
        #     traversable_mask=trav_np,
        #     unknown_mask=unk_np, 
        #     save_path=os.path.join(debug_dir, "debug_bev_mask_foggy.png"),
        #     title="Traversable + Unknown Fog",
        # )
        # combined_mask = np.logical_and(trav_np, ~unk_np)
        # BEVMaskGenerator.save_vis(
        #     traversable_mask=combined_mask,
        #     save_path=os.path.join(debug_dir, "debug_bev_mask_limited.png"),
        #     title="Traversable + Unknown Fog",
        # )

        # print(f"Debug images saved to {debug_dir}")
        # exit(0)
        # # END DEBUG


        bs, n = inputs.shape[:2]

        # ---------- Backbone + neck ----------
        if hasattr(self, 'backbone'):
            inputs_flat = inputs.flatten(0, 1)
            if self.frozen_backbone:
                if self.backbone.training:
                    self.backbone.eval()
                with torch.no_grad():
                    if isinstance(self.backbone, BaseModule):
                        x = self.backbone(inputs_flat)[0]
                    else:  # TorchHub ViT
                        x = self.backbone.forward_features(
                            inputs_flat)['x_norm_patchtokens']  # (B*n, num_patches, dim)
                        x = x.mT.reshape(
                            bs * n, -1,
                            inputs.shape[-2] // self.patch_size,
                            inputs.shape[-1] // self.patch_size
                        )  # (B*n, dim, H', W')
            else:
                x = self.backbone(inputs_flat)[0]
        else:
            x = data_samples['feats'].flatten(0, 1)

        if hasattr(self, 'projection'):
            x = self.projection(x.permute(0, 2, 3, 1))[0]
            x = x.permute(0, 3, 1, 2)
        if hasattr(self, 'backbone') or hasattr(self, 'projection'):
            data_samples['feats'] = x.reshape(bs, n, *x.shape[1:])
        if n > data_samples['num_views']:
            x = x.reshape(bs, n, *x.shape[1:])
            x = x[:, :data_samples['num_views']].flatten(0, 1)

        img_feats = self.neck(x)  # [B*num_views, C, H', W']

        # ================= LSS → BEV =================
        grid_config = dict(
            x=[-6.4, 6.4, 0.2],
            y=[-6.4, 6.4, 0.2],
            z=[-1.8, 0.2, 2.0],
            depth=[0.0, 7.2, 0.2],
        ) # dict with 'x','y','z','depth' tuples (min, max, interval)
        lss_inputs = self.prepare_view_transform(img_feats, data_samples, depth_cfg=grid_config['depth'])
        bev_feats = lss_bev_pool(**lss_inputs, collapse_z=True, grid_config=grid_config) # (B, C, Dy, Dx)
        bev_feats = bev_feats.permute(0, 1, 3, 2).contiguous()  # (B, C, Dx, Dy)

        # ================= BEV HEAD =================
        if self.head_type == 'bev':
            bev_fpn_input = torch.cat(
                [
                    bev_feats,
                    # meta_embed.unsqueeze(-1).unsqueeze(-1).expand(
                    #     -1, -1, bev_feats.shape[2], bev_feats.shape[3]
                    # ),
                ],
                dim=1,
            )  # [B, 256, H, W]

            bev_fpn_feats = self.bev_fpn(bev_fpn_input)          # [B, 256, H, W]
            bev_outputs = self.bev_head(bev_fpn_feats)           # [B, 2, H, W]

            all_masks = data_samples['traversable_mask'] # [B, 3, H, W]
            traversable_masks = all_masks[:, 0, :, :]  # [B, H, W], 1 for traversable
            sure_masks = all_masks[:, 1, :, :]          # [B, H, W], 1 for sure
            valid_masks = all_masks[:, 2, :, :]         # [B, H, W], 1 for valid

            if mode == 'predict':
                return bev_outputs, valid_masks # TODO: mask invalid out in evaluation
            elif mode == 'loss':
                # closer_range_mask = torch.zeros_like(bev_outputs[:, 0:1, :, :]) # [B, 1, H, W]
                # closer_range_mask[:, :, 14:50, 14:50] = 1.0  # only care about close range
                B, C, H, W = bev_outputs.shape
                bev_outputs_flat = bev_outputs.permute(0,2,3,1).reshape(B*H*W, C)
                target_mask = torch.logical_and(traversable_masks.bool(), sure_masks.bool()) # [B, H, W], only sure traversable & traversable areas
                target_flat = target_mask.long().reshape(-1)  # [B*H*W] # BUG: if batch is small like 1 or 2, and we happen to have all samples invalid, then loss will be nan
                # closer_range_mask_flat = closer_range_mask.view(-1).bool()  # [B*H*W]
                valid_mask_flat = valid_masks.reshape(-1).bool()  # [B*H*W]
                # loss_bev_close = nn.functional.cross_entropy(
                #     bev_outputs_flat[valid_mask_flat],   # [N, C]
                #     target_flat[valid_mask_flat],        # [N]
                #     reduction='mean'
                # )
                loss_bev = nn.functional.cross_entropy(
                    bev_outputs_flat[valid_mask_flat],   # [M, C]
                    target_flat[valid_mask_flat],        # [M]
                    reduction='mean'
                )
                # loss_bev = 1 * loss_bev_close + 1 * loss_bev_far
                if torch.isnan(loss_bev):
                    # try inspect which part causes nan
                    print(f"pred is nan or not? :{torch.isnan(bev_outputs).any()}")
                    print(f"target_flat is nan? : {torch.isnan(target_flat).any()}")
                    print(f"mask is nan? : {torch.isnan(valid_mask_flat).any()}")

                    for i in range(B):
                        loss_bev_i = nn.functional.cross_entropy(
                            bev_outputs_flat[i*H*W:(i+1)*H*W][valid_mask_flat[i*H*W:(i+1)*H*W]],   # [N_i, C]
                            target_flat[i*H*W:(i+1)*H*W][valid_mask_flat[i*H*W:(i+1)*H*W]],        # [N_i]
                            reduction='mean'
                        )
                        if torch.isnan(loss_bev_i):
                            print(f"Sample {i} in batch causes NaN loss in close range part.")
                        else:
                            print(f"Sample {i} in batch is OK in close range part.")
                assert not torch.isnan(loss_bev), "loss_bev is NaN, the batch is too small and all samples in batch are invalid likely ending on staircases."
                losses = dict(loss_bev=loss_bev)
                return losses
        # if we get here, head_type is invalid
        raise RuntimeError(f'Unknown head_type {self.head_type}')
