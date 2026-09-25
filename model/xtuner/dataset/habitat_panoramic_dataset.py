import os
import pickle

import numpy as np
from mmengine.dataset import BaseDataset

from mmdet3d.registry import DATASETS
from xtuner.perception_modules.oracle_bev import BEVMaskGenerator

import math
from tqdm import tqdm

@DATASETS.register_module()
class HabitatPanoramicDataset(BaseDataset):
    """Habitat panoramic dataset that crops 4 perspective views online.

    The PKL provides a single panoramic RGB/depth per entry. This dataset
    only wires the metadata and absolute paths; actual cropping happens in
    the transform to keep random yaw augmentation per epoch.
    """

    METAINFO = dict(classes=tuple())

    def __init__(self,
                 ann_file,
                 data_root,
                 pipeline=None,
                 max_frames=1,
                 range_xy=3.6,
                 range_z=(-2.0, 0.4),
                 sampling_sigma=3.0,
                 version='dynamic',
                 random_base_yaw=False,
                 num_yaw_bins=1,
                 min_traversable_ratio=0.04, # skip on-stairs samples
                 working_mode='dataloader', # 'dataloader' or 'dataengine'
                 **kwargs):
        self.num_views = 4
        self.max_frames = max_frames
        self.range_xy = range_xy
        self.range_z = range_z
        self.working_mode = working_mode
        self.random_base_yaw = random_base_yaw
        self.num_yaw_bins = num_yaw_bins
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            pipeline=pipeline,
            serialize_data=False,
            lazy_init=False,
            **kwargs)
        assert version in ['static', 'dynamic'], f"Unknown version {version}, should be 'static' or 'dynamic'."
        self.version = version
        self.sampling_sigma = sampling_sigma
        self.min_traversable_ratio = min_traversable_ratio

        self.bev_mask_generator = BEVMaskGenerator(
            ann_file=self.ann_file,
            out_dir=None, # will not be used
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
        if self.num_yaw_bins > 1:
            assert working_mode == 'dataengine', "Multiple yaw bins per sample is only supported in 'dataengine' mode."
            print(f'[DataEngine] Using {self.num_yaw_bins} yaw bins per sample.')
        if self.working_mode == 'dataengine':
            self.processed_cnt = 0
        print(f"HabitatPanoramicDataset using {self.version} data from pkl file {self.ann_file}.")

    def load_data_list(self):
        with open(self.ann_file, 'rb') as f:
            info = pickle.load(f)
        if self.working_mode == 'dataengine':
            # return info['data_list']
            data_list = []
            if self.num_yaw_bins > 1:
                for data in info['data_list']:
                    waypoint_token = data['waypoint_token']
                    dir = os.path.join(
                        self.data_root,
                        'multi_yaw_traversable_masks',
                        f"{waypoint_token}.npz"
                    )
                    if waypoint_token is not None and not os.path.exists(dir):
                        data_list.append(data)
                    elif waypoint_token is not None and os.path.exists(dir):
                        print(f"Skipping sample {waypoint_token} since traversable mask npz already exists.")
            else:
                for data in info['data_list']:
                    token = data['token']
                    dir =  os.path.join(
                        self.data_root,
                        'traversable_masks',
                        f"{token}.npz"
                    )
                    if token is not None and not os.path.exists(dir):
                        data_list.append(data)
                    elif token is not None and os.path.exists(dir):
                        print(f"Skipping sample {token} since traversable mask npz already exists.")
            return data_list
        else:
            # skip samples with missing labels
            data_list = []
            for data in info['data_list']:
                token = data['token']
                if os.path.exists(
                        os.path.join(self.data_root, 'traversable_masks', f"{token}.npz")):
                    data_list.append(data)
                else:
                    print(f"Skipping sample {token} due to missing traversable mask npz.")
            return data_list

    def _absolute_path(self, rel_path):
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.join(self.data_root, rel_path)

    def get_data_info(self, index):
        anchor = self.data_list[index]
        scene_token = anchor.get('scene_token', '')

        # collect candidates in same scene for multi-frame sampling
        frames = []
        if self.max_frames > 1:
            candidates = [f for f in self.data_list if f.get('scene_token', '') == scene_token]
            anchor_pose = np.array(anchor['ego2global'])
            anchor_ts = float(anchor.get('timestamp', 0.0))
            in_range = []
            for f in candidates:
                pose = np.array(f['ego2global'])
                z_rel = pose[2, 3] - anchor_pose[2, 3]
                if (np.linalg.norm(anchor_pose[:2, 3] - pose[:2, 3]) <= self.range_xy
                        and self.range_z[0] <= z_rel <= self.range_z[1]):
                    in_range.append(f)
            # in_range = sorted(in_range, key=lambda x: x.get('timestamp', 0))
            extra = [f for f in in_range if f is not anchor]
            k = self.max_frames - 1
            # print(f"Index {index}: Selected first {self.max_frames - 1} out of {len(extra)} extra frames.")
            # frames_sel = [anchor] + extra[:max(0, self.max_frames - 1)]
            if len(extra) > 0:
                # --- compute distances to anchor ---
                d_xy = np.array([
                    np.linalg.norm(np.array(f['ego2global'])[:2, 3] - anchor_pose[:2, 3])
                    for f in extra
                ], dtype=np.float32)

                d_z = np.array([
                    (np.array(f['ego2global'])[2, 3] - anchor_pose[2, 3])
                    for f in extra
                ], dtype=np.float32)

                d_t = np.array([
                    abs(float(f.get('timestamp', 0.0)) - anchor_ts)
                    for f in extra
                ], dtype=np.float32)

                # --- gaussian weighting (tune these sigmas) ---
                sigma_xy = getattr(self, "sigma_xy", self.sampling_sigma)   # meters
                sigma_z  = getattr(self, "sigma_z",  None)   # meters
                sigma_t  = getattr(self, "sigma_t",  None)  # seconds (set None to ignore time)

                r2 = (d_xy / sigma_xy) ** 2
                if sigma_z is not None:
                    r2 += (d_z / sigma_z) ** 2
                if sigma_t is not None:
                    r2 += (d_t / sigma_t) ** 2

                w = np.exp(-0.5 * r2)

                # --- ensure far frames still have a chance ---
                uniform_mix = getattr(self, "uniform_mix", 0.05)  # 0.0..1.0
                prob_floor  = getattr(self, "prob_floor", 1e-12)

                w = (1.0 - uniform_mix) * w + uniform_mix * np.ones_like(w)
                w = w + prob_floor
                p = w / w.sum()

                # --- sample without replacement ---
                rng = getattr(self, "rng", None)
                if rng is None:
                    rng = np.random.default_rng()  # or set self.rng in __init__ for reproducibility

                pick_n = min(k, len(extra))
                picked_idx = rng.choice(len(extra), size=pick_n, replace=False, p=p)
                picked = [extra[i] for i in picked_idx]

                # keep temporal order if your model expects it
                picked.sort(key=lambda x: x.get('timestamp', 0.0))
                d_xy_picked = d_xy[picked_idx]
                d_z_picked = d_z[picked_idx]
                d_t_picked = d_t[picked_idx]
                print(f"Index {index}: Ave d_xy={d_xy_picked.mean():.2f}m, \
                      d_z={d_z_picked.mean():.2f}m, d_t={d_t_picked.mean():.2f}s for {pick_n} frames out of {len(extra)}.")
                frames_sel = [anchor] + picked
            else:
                frames_sel = [anchor]
        else:
            frames_sel = [anchor]

        for frame in frames_sel:
            pano = frame['images']['PANORAMIC']
            if self.version == 'static':
                img_rel_path = pano['img_path'].replace('dynamic', 'static')
                depth_rel_path = pano['depth_path'].replace('dynamic', 'static')
            else:
                img_rel_path = pano['img_path']
                depth_rel_path = pano['depth_path']
            frames.append(
                dict(
                    pano_img_path=self._absolute_path(img_rel_path),
                    pano_depth_path=self._absolute_path(depth_rel_path),
                    ego2global=np.array(frame['ego2global']).astype(np.float32),
                    human_poses=frame.get('human_poses', {}),
                    timestamp=frame.get('timestamp', 0.0),
                    scene_token=frame.get('scene_token', scene_token)))

        num_frames = len(frames)

        if self.working_mode == 'dataengine':
            if self.num_yaw_bins == 1:
                # old feature: 1 yaw per sample
                if self.random_base_yaw:
                    # set a deterministic local seed
                    rng = np.random.default_rng(index)
                    base_yaw = float(rng.uniform(-180.0, 180.0))
                else:
                    base_yaw = 0.0
                anchor['habitat_base_yaw'] = math.radians(base_yaw) # the oracle bev generator expects radians
                traversable_mask = self.bev_mask_generator.oracle_bev_from_dataloader(
                    anchor, 
                    return_limited_mask=True,
                    origin_radius_m=0.0,
                ) # 2, H, W
                traversable_ration = (traversable_mask[0] == 1).sum() / np.prod(traversable_mask.shape[1:]) 
                if traversable_ration < self.min_traversable_ratio:
                    print(f"[{self.processed_cnt} / {len(self.data_list)}] Skipping sample {index} due to low traversable ratio {traversable_ration:.3f} < {self.min_traversable_ratio:.3f}.")
                    self.processed_cnt += 1
                    traversable_mask_and_yaw_path = os.path.join(
                        self.data_root,
                        'traversable_masks_low_ratio_debug',
                        f"{anchor['token']}.npz"
                    )
                    np.savez_compressed(
                        traversable_mask_and_yaw_path,
                        traversable_mask=traversable_mask,
                        base_yaw=base_yaw,
                    )
                    return # skip this sample in dataengine mode
                else:
                    # save traversable mask with 2 channels
                    traversable_mask_and_yaw_path = os.path.join(
                        self.data_root,
                        'traversable_masks',
                        f"{anchor['token']}.npz"
                    )
                    np.savez_compressed(
                        traversable_mask_and_yaw_path,
                        traversable_mask=traversable_mask,
                        base_yaw=base_yaw,
                    )
                    print(f"[{self.processed_cnt} / {len(self.data_list)}] Saved traversable mask and with yaw {base_yaw} for sample {index} to {traversable_mask_and_yaw_path}.")
                    self.processed_cnt += 1
                    return
                
            else: # new feature: multiple yaws per sample
                degree_per_bin = 360.0 / self.num_yaw_bins
                all_traversable_masks = []
                # all_base_yaws = []
                # all_traversable_ratios = []
                # for bin_idx in tqdm(range(self.num_yaw_bins), desc=f"Processing sample {index} for {self.num_yaw_bins} yaw bins"):
                #     curr_base_yaw = 0 + bin_idx * degree_per_bin
                #     anchor['habitat_base_yaw'] = math.radians(curr_base_yaw) # the oracle bev generator expects radians
                #     traversable_mask = self.bev_mask_generator.oracle_bev_from_dataloader(
                #         anchor, 
                #         return_limited_mask=True,
                #     ) # 2, H, W
                #     all_traversable_masks.append(traversable_mask)
                #     traversable_ration = (traversable_mask[0] == 1).sum() / np.prod(traversable_mask.shape[1:]) 
                #     all_base_yaws.append(curr_base_yaw)
                #     all_traversable_ratios.append(traversable_ration)
                degree_per_bin = 360.0 / self.num_yaw_bins
                assert self.num_yaw_bins % 4 == 0, "quadrant-rotation optimization assumes num_yaw_bins divisible by 4"
                bins_per_quad = self.num_yaw_bins // 4  # 64 if num_yaw_bins=256

                # +1 if yaw increases CCW, -1 if yaw increases CW (we auto-correct below if wrong)
                dir_sign = -1

                # ---- compute only the first quadrant [0, 90) ----
                first_quad_masks = []
                first_quad_ratios = []
                for bin_idx in tqdm(range(bins_per_quad), desc=f"Processing sample {index} for {bins_per_quad} yaw bins (0..90)"):
                    curr_base_yaw = bin_idx * degree_per_bin
                    anchor['habitat_base_yaw'] = math.radians(curr_base_yaw)
                    traversable_mask = self.bev_mask_generator.oracle_bev_from_dataloader(
                        anchor,
                        return_limited_mask=True,
                    )  # (2, H, W)
                    first_quad_masks.append(traversable_mask)
                    r = (traversable_mask[0] == 1).sum() / np.prod(traversable_mask.shape[1:])
                    first_quad_ratios.append(r)

                first_quad_masks = np.stack(first_quad_masks, axis=0)  # (64, 2, H, W)

                # ---- build full 256 by 90° rotations (vectorized) ----
                def build_full_from_first_quad(masks_0_90, dir_sign):
                    return np.concatenate(
                        [np.rot90(masks_0_90, k=dir_sign * q, axes=(2, 3)) for q in range(4)],
                        axis=0
                    )  # (256, 2, H, W)

                all_traversable_mask_np = build_full_from_first_quad(first_quad_masks, dir_sign)

                # ratios & yaws: rotation preserves counts, so just tile
                all_traversable_ratios = list(np.tile(np.array(first_quad_ratios, dtype=np.float32), 4))
                all_base_yaws_np = (np.arange(self.num_yaw_bins, dtype=np.float32) * degree_per_bin)

                # ---- one sanity check at +90° (bin = bins_per_quad) with 5% threshold ----
                # (bin 64 is exactly 90° when num_yaw_bins divisible by 4)
                yaw_90 = bins_per_quad * degree_per_bin
                anchor['habitat_base_yaw'] = math.radians(yaw_90)
                mask_90_true = self.bev_mask_generator.oracle_bev_from_dataloader(
                    anchor,
                    return_limited_mask=True,
                )  # (2, H, W)
                combined_mask_90_true = np.logical_and(
                    mask_90_true[0].astype(bool),
                    mask_90_true[1].astype(bool)
                )

                mask_90_rot = all_traversable_mask_np[bins_per_quad]  # derived 90° bin
                combined_mask_90_rot = np.logical_and(
                    mask_90_rot[0].astype(bool),
                    mask_90_rot[1].astype(bool)
                )
                diff_ratio = float(np.mean(combined_mask_90_true != combined_mask_90_rot))

                if diff_ratio > 0.1:
                    # raise RuntimeError(f"[Yaw sanity] sample {index}: diff@90 = {diff_ratio:.3f} (>0.1), possible yaw direction convention mismatch. Please check.")
                    # auto-flip direction once if convention is opposite
                    # dir_sign = -dir_sign
                    # all_traversable_mask_np = build_full_from_first_quad(first_quad_masks, dir_sign)
                    # diff_ratio2 = float(np.mean(mask_90_true != all_traversable_mask_np[bins_per_quad]))
                    # print(f"[Yaw sanity] sample {index}: diff@90 was {diff_ratio:.3f} (>0.05), flipped dir_sign. new diff={diff_ratio2:.3f}")
                    print(f"[Yaw sanity] sample {index}: diff@90 = {diff_ratio:.3f} (>0.1), possible yaw direction convention mismatch. Please check.")
                else:
                    print(f"[Yaw sanity] sample {index}: diff@90 = {diff_ratio:.3f}")

                min_traversable_ratio = min(all_traversable_ratios)
                if min_traversable_ratio < self.min_traversable_ratio:
                    print(f"[{self.processed_cnt} / {len(self.data_list)}] Skipping sample {index} due to low traversable ratio {min_traversable_ratio:.3f} < {self.min_traversable_ratio:.3f} among {self.num_yaw_bins} yaw bins.")
                    self.processed_cnt += 1
                    traversable_mask_and_yaw_path = os.path.join(
                        self.data_root,
                        'multi_yaw_traversable_masks_low_ratio_debug',
                        f"{anchor['waypoint_token']}.npz"
                    )
                    np.savez_compressed(
                        traversable_mask_and_yaw_path,
                        traversable_mask=all_traversable_mask_np,
                        base_yaw=all_base_yaws_np,
                    )
                    return # skip this sample in dataengine mode
                else:
                    # save traversable mask with 2 channels
                    traversable_mask_and_yaw_path = os.path.join(
                        self.data_root,
                        'multi_yaw_traversable_masks',
                        f"{anchor['waypoint_token']}.npz"
                    )
                    np.savez_compressed(
                        traversable_mask_and_yaw_path,
                        traversable_mask=all_traversable_mask_np,
                        base_yaw=all_base_yaws_np,
                    )
                    print(f"[{self.processed_cnt} / {len(self.data_list)}] Saved traversable mask and with {self.num_yaw_bins} yaws for sample {index} to {traversable_mask_and_yaw_path}.")
                    self.processed_cnt += 1
                    return

        elif self.working_mode == 'dataloader':
            assert self.random_base_yaw is False, 'base yaw is loaded from npz'
            traversable_mask_and_yaw_path = os.path.join(
                self.data_root,
                'traversable_masks',
                f"{anchor['token']}.npz"
            )
            loaded = np.load(traversable_mask_and_yaw_path)
            traversable_mask = loaded['traversable_mask'] # 2, H, W
            base_yaw = loaded['base_yaw'].item()
            traversable_ration = (traversable_mask[0] == 1).sum() / np.prod(traversable_mask.shape[1:]) 
            if traversable_ration < self.min_traversable_ratio:
                print(f"Warning: Sample {index} has low traversable ratio {traversable_ration:.3f} < {self.min_traversable_ratio:.3f}.")
                traversable_mask = np.stack([
                    traversable_mask[0],
                    traversable_mask[1],
                    np.zeros_like(traversable_mask[0])
                ], axis=0) # 3, H, W
            else:
                traversable_mask = np.stack([
                    traversable_mask[0],
                    traversable_mask[1],
                    np.ones_like(traversable_mask[0])
                ], axis=0) # 3, H, W
        else:
            raise ValueError(f"Unknown working_mode {self.working_mode}, should be 'dataloader' or 'dataengine'.")

        data_info = dict(
            sample_idx=anchor.get('sample_idx', index),
            token=anchor.get('token', f"pano_{scene_token}_{index}"),
            scene_idx=anchor.get('scene_idx', scene_token),
            scene_token=scene_token,
            num_views=self.num_views * num_frames,
            pano_img_path=frames[0]['pano_img_path'],
            pano_depth_path=frames[0]['pano_depth_path'],
            ego2global=np.stack([f['ego2global'] for f in frames], axis=0),
            frames=frames,
            occ_path=self._absolute_path(anchor.get('occ_path', '')),
            human_poses=anchor.get('human_poses', {}),
            human_poses_all=[f['human_poses'] for f in frames],
            traversable_mask=traversable_mask, # 3, H, W
            base_yaw=base_yaw,
        )
        return data_info

if __name__ == "__main__":
    # split = 'train'
    split = 'val'
    # scan_id = 'S9hNv5qa7GM'
    scan_id = '2azQ1b91cZZ'

    old_behavior = True
    if old_behavior:
        random_base_yaw = False
        num_yaw_bins = 1
    else:
        random_base_yaw = True
        num_yaw_bins = 256

    dataset = HabitatPanoramicDataset(
        ann_file = f'scan{scan_id}_Landmark-RxR-dynamic.pkl',
        data_root = os.environ.get(
            "BEACON_FALCON_DATA",
            "/path/to/Falcon/data",
        ) + f'/captures_v3_{split}_{scan_id}',
        max_frames=1,
        version='dynamic',
        random_base_yaw=random_base_yaw,
        working_mode='dataengine',
        num_yaw_bins=num_yaw_bins,
    )
    for i in range(len(dataset.data_list)):
        dataset.get_data_info(i)   # <-- IMPORTANT: not dataset[i]
    # # debug
    # i = 354 #407 # 354 # 358
    # dataset.get_data_info(i) 
