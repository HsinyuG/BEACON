import copy
import os
import math
from typing import Optional

import mmcv
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.transforms import BaseTransform
from mmengine.fileio import get
from PIL import Image
import trimesh

from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class BEVLoadMultiViewImageFromFiles(LoadMultiViewImageFromFiles):
    """Load multi channel images from a list of separate channel files.

    ``BEVLoadMultiViewImageFromFiles`` adds the following keys for the
    convenience of view transforms in the forward:
        - 'cam2lidar'
        - 'lidar2img'

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        num_views (int): Number of view in a frame. Defaults to 5.
        test_mode (bool): Whether is test mode in loading. Defaults to False.
        set_default_scale (bool): Whether to set default scale.
            Defaults to True.
    """

    def transform(self, results: dict) -> Optional[dict]:
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
            Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        # Support multi-view images with different shapes
        filename, cam2img, lidar2cam, cam2ego = [], [], [], []
        for _, cam_item in results['images'].items():
            filename.append(cam_item['img_path'])
            lidar2cam.append(cam_item['lidar2cam'])

            cam2img_array = np.eye(4).astype(np.float32)
            cam2img_array[:3, :3] = np.array(cam_item['cam2img']).astype(
                np.float32)
            cam2img.append(cam2img_array)

            cam2ego_array = np.array(cam_item['cam2ego']).astype(np.float32)
            cam2ego.append(cam2ego_array)

        results['img_path'] = filename
        results['cam2img'] = np.stack(cam2img, axis=0)
        results['lidar2cam'] = np.stack(lidar2cam, axis=0)
        results['cam2ego'] = np.stack(cam2ego, axis=0)

        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])

        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_bytes = [
            get(name, backend_args=self.backend_args) for name in filename
        ]
        imgs = [
            mmcv.imfrombytes(
                img_byte,
                flag=self.color_type,
                backend='pillow',
                channel_order='rgb') for img_byte in img_bytes
        ]
        # handle the image with different shape
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
            ]
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape[:2]
        if self.set_default_scale:
            results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['num_views'] = self.num_views
        return results


@TRANSFORMS.register_module()
class PointToMultiViewDepth(BaseTransform):

    def __init__(self, depth_cfg, downsample=1):
        self.downsample = downsample
        self.depth_cfg = depth_cfg

    def points2depth(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width))
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]

        kept1 = ((coor[:, 0] >= 0) & (coor[:, 0] < width) & (coor[:, 1] >= 0) &
                 (coor[:, 1] < height) & (depth < self.depth_cfg[1]) &
                 (depth >= self.depth_cfg[0]))
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth.to(depth_map)
        return depth_map

    def transform(self, results):
        pts_lidar = results['points']
        imgs = results['img']
        cam2imgs = results['cam2img']
        img_aug_mats = results['img_aug_mat']
        depth = []

        for i, cam_name in enumerate(results['images']):
            cam2img = cam2imgs[i]
            lidar2cam = results['images'][cam_name]['lidar2cam']
            lidar2img = cam2img @ lidar2cam

            post_rot = img_aug_mats[i][:3, :3]
            post_tran = img_aug_mats[i][:3, 3]

            pts_img = (
                pts_lidar.tensor[:, :3] @ lidar2img[:3, :3].T +
                lidar2img[:3, 3])
            pts_img = torch.cat(
                [pts_img[:, :2] / pts_img[:, 2:3], pts_img[:, 2:3]], 1)
            pts_img = pts_img @ post_rot.T + post_tran

            depth_map = self.points2depth(pts_img, imgs[i].shape[0],
                                          imgs[i].shape[1])
            depth.append(depth_map)
        results['gt_depth'] = torch.stack(depth)
        return results


@TRANSFORMS.register_module()
class LoadOccFromFile(BaseTransform):

    def transform(self, results):
        occ_path = os.path.join(results['occ_path'], 'labels.npz')
        occ_labels = np.load(occ_path)

        results['gt_semantic_seg'] = occ_labels['semantics']
        results['mask_lidar'] = occ_labels['mask_lidar']
        results['mask_camera'] = occ_labels['mask_camera']
        return results


@TRANSFORMS.register_module()
class ImageAug3D(BaseTransform):

    def __init__(self,
                 final_dim,
                 resize_lim,
                 bot_pct_lim=[0.0, 0.0],
                 rot_lim=[0.0, 0.0],
                 rand_flip=False,
                 is_train=False):
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train

    def sample_augmentation(self, results):
        H, W = results['ori_shape']
        fH, fW = self.final_dim
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(self, img, rotation, translation, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = Image.fromarray(img.astype('uint8'), mode='RGB')
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ])
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b

        return img, rotation, translation

    def transform(self, data):
        imgs = data['img']
        new_imgs = []
        transforms = []
        for img in imgs:
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(
                data)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            new_img, rotation, translation = self.img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            new_imgs.append(np.array(new_img).astype(np.float32))
            transforms.append(transform.numpy())
        data['img'] = new_imgs
        # update the calibration matrices
        data['img_aug_mat'] = transforms
        return data


@TRANSFORMS.register_module()
class BEVDataAug(BaseTransform):

    def __init__(self,
                 rot_lim=[0.0, 0.0],
                 scale_lim=[1.0, 1.0],
                 rand_flip=False):
        self.rot_lim = rot_lim
        self.scale_lim = scale_lim
        self.rand_flip = rand_flip

    def sample_augmentation(self):
        rotate = np.random.uniform(*self.rot_lim)
        scale = np.random.uniform(*self.scale_lim)
        flip_x = False
        flip_y = False
        if self.rand_flip:
            flip_x = np.random.choice([0, 1])
            flip_y = np.random.choice([0, 1])
        return rotate, scale, flip_x, flip_y

    def bev_transform(self, rotate, scale, flip_x, flip_y):
        theta = rotate / 180 * np.pi
        rotation = torch.Tensor([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ])
        scale_mat = torch.Tensor([[scale, 0, 0], [0, scale, 0], [0, 0, scale]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        if flip_x:
            flip_mat[0, 0] *= -1
        if flip_y:
            flip_mat[1, 1] *= -1
        rotation = flip_mat @ scale_mat @ rotation
        return rotation

    def transform(self, data):
        rotate, scale, flip_x, flip_y = self.sample_augmentation()
        assert rotate == 0 and scale == 1
        rotation = self.bev_transform(rotate, scale, flip_x, flip_y)

        if 'gt_semantic_seg' in data and (flip_x or flip_y):
            for key in ('gt_semantic_seg', 'mask_lidar', 'mask_camera'):
                if flip_x:
                    data[key] = data[key][::-1].copy()
                if flip_y:
                    data[key] = data[key][:, ::-1].copy()
        data['bev_aug_mat'] = rotation.numpy()
        return data


@TRANSFORMS.register_module()
class LoadSingleDepthMaps(BaseTransform):

    def __init__(self, key='depth', apply_aug=True):
        self.key = key
        self.apply_aug = apply_aug

    def transform(self, results):
        depth_maps = []
        img_aug_mats = results.get('img_aug_mat')
        images = list(results['images'].values())
        for i, cam_item in enumerate(images):
            depth_path = cam_item.get('depth_path')
            if depth_path is None:
                raise KeyError('Missing depth_path in dataset annotations.')
            # Allow absolute and relative paths (relative to working dir)
            depth = torch.from_numpy(np.load(depth_path)).float()
            if depth.ndim > 2:
                depth = depth.squeeze()

            if self.apply_aug and img_aug_mats is not None:
                post_rot = img_aug_mats[i][:3, :3]
                post_tran = img_aug_mats[i][:3, 3]
                assert post_rot[0, 1] == post_rot[1, 0] == 0  # noqa

                h, w = depth.shape
                depth = depth[None, None]
                depth = F.interpolate(
                    depth, (int(h * post_rot[1, 1] + 0.5),
                            int(w * post_rot[0, 0] + 0.5)),
                    mode='bilinear',
                    align_corners=False).squeeze(0).squeeze(0)
                depth = depth[int(post_tran[1]):, int(-post_tran[0]):]
            depth_maps.append(depth)

        results[self.key] = torch.stack(depth_maps)
        return results

@TRANSFORMS.register_module()
class LoadFeatMaps(BaseTransform):

    def __init__(self, data_root, key, apply_aug=False):
        self.data_root = data_root
        self.key = key
        self.apply_aug = apply_aug

    def transform(self, results):
        feats = []
        img_aug_mats = results.get('img_aug_mat')
        for i, filename in enumerate(results['filename']):
            feat = np.load(
                os.path.join(self.data_root,
                             filename.split('/')[-1].split('.')[0] + '.npy'))
            feat = torch.from_numpy(feat)

            if self.apply_aug and img_aug_mats is not None:
                post_rot = img_aug_mats[i][:3, :3]
                post_tran = img_aug_mats[i][:3, 3]
                assert post_rot[0, 1] == post_rot[1, 0] == 0  # noqa

                h, w = feat.shape
                mode = 'nearest' if torch.all(feat == feat.floor()) else 'bilinear'
                feat = F.interpolate(
                    feat[None, None], (int(h * post_rot[1, 1] + 0.5),
                                       int(w * post_rot[0, 0] + 0.5)),
                    mode=mode).squeeze()
                feat = feat[int(post_tran[1]):, int(-post_tran[0]):]
            feats.append(feat)

        results[self.key] = torch.stack(feats)
        return results

import re
@TRANSFORMS.register_module()
class LoadActions(BaseTransform):
    def __init__(self):
        # action_template = "Move to the {move_dir} with a {range_tag} step, and end facing {relative_dir}."
        self.pattern = re.compile(
            r"Move to the (?P<move_dir>.+?) with a (?P<range_tag>.+?) step, and end facing (?P<relative_dir>.+?)\.?"
        )        
        self.direction_label_map = {
            'Front': 0,
            'Front Right': 1,
            'Right': 2,
            'Back Right': 3,
            'Back': 4,
            'Back Left': 5,
            'Left': 6,
            'Front Left': 7,
        }
        self.vel_label_map = {
            'Small': 0,
            'Big': 1,
        }

    def transform(self, results):
        waypoints = torch.tensor(results['action_waypoints']).float()
        results['action_waypoints'] = waypoints
        m = self.pattern.match(results['meta_action'])
        if m is None:
            raise ValueError(f"Action string does not match expected format: {results['meta_action']}")
        move_dir = m.group('move_dir')
        range_tag = m.group('range_tag')
        # relative_dir = m.group('relative_dir') # not used currently
        results['dir_label'] = torch.tensor(self.direction_label_map[move_dir], dtype=torch.long)
        results['vel_label'] = torch.tensor(self.vel_label_map[range_tag], dtype=torch.long)
        return results


MAX_DEPTH = 10.0  # hardcode max
@TRANSFORMS.register_module()
class PanoToMultiView(BaseTransform):
    """Load a panoramic RGB/depth and crop 4 perspective views with random base yaw."""

    def __init__(self,
                 out_hw=(448, 448),
                 fov_deg=90.0,
                 yaw_offsets=(0.0, 90.0, 180.0, -90.0),
                 base_yaw=0.0,
                 random_base_yaw=False,
                 to_float32=True):
        self.out_hw = out_hw
        self.fov_deg = fov_deg
        self.yaw_offsets = yaw_offsets
        self.base_yaw = base_yaw
        self.random_base_yaw = random_base_yaw
        self.to_float32 = to_float32

    def _pano_maps(self, fov_deg, out_hw, yaw_deg, pitch_deg, pano_shape):
        H_p, W_p = pano_shape[:2]
        H_out, W_out = out_hw
        j, i = np.meshgrid(np.arange(W_out), np.arange(H_out))
        x = (j + 0.5) / W_out * 2.0 - 1.0
        y = (i + 0.5) / H_out * 2.0 - 1.0
        fov = np.deg2rad(fov_deg)
        s = np.tan(fov / 2.0)
        x_cam, y_cam, z_cam = x * s, -y * s, -np.ones_like(x)
        dirs_cam = np.stack([x_cam, y_cam, z_cam], -1)
        dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True)

        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        R_yaw = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0],
                          [-np.sin(yaw), 0, np.cos(yaw)]])
        R_pitch = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)],
                            [0, np.sin(pitch), np.cos(pitch)]])
        dirs_world = dirs_cam @ (R_pitch @ R_yaw).T
        X, Y, Z = dirs_world[..., 0], dirs_world[..., 1], dirs_world[..., 2]
        theta = np.arctan2(X, -Z)
        phi = np.arcsin(np.clip(Y, -1.0, 1.0))
        u = (theta + np.pi) / (2.0 * np.pi) * W_p
        v = (np.pi / 2.0 - phi) / np.pi * H_p
        return u.astype(np.float32), v.astype(np.float32)

    def _cam2img(self):
        H_out, W_out = self.out_hw
        fx = fy = W_out / (2.0 * math.tan(math.radians(self.fov_deg) / 2.0))
        cx = W_out / 2.0
        cy = H_out / 2.0
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                     dtype=np.float32)
        K4 = np.eye(4, dtype=np.float32)
        K4[:3, :3] = K
        return K4

    def _cam2ego(self, yaw_deg):
        yaw = math.radians(yaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        T = np.eye(4, dtype=np.float32)
        R_front = np.array([[0, 0, 1],
                    [-1, 0, 0],
                    [0, -1, 0]], dtype=np.float32)
        Rz = np.array([[c, -s, 0],
                    [s,  c, 0],
                    [0,  0, 1]], dtype=np.float32)
        T[:3, :3] = Rz @ R_front
        return T

    def transform(self, results):
        pano_img_path = results.get('pano_img_path')
        pano_depth_path = results.get('pano_depth_path')
        if pano_img_path is None or pano_depth_path is None:
            raise KeyError('pano_img_path and pano_depth_path are required.')

        pano_rgb = cv2.imread(pano_img_path, cv2.IMREAD_COLOR)
        if pano_rgb is None:
            raise FileNotFoundError(f'Cannot read pano image at {pano_img_path}')
        pano_rgb = pano_rgb[:, :, ::-1]  # BGR -> RGB

        pano_depth = np.load(pano_depth_path)
        if pano_depth.ndim == 3 and pano_depth.shape[2] == 1:
            pano_depth = pano_depth[:, :, 0]
        pano_depth = pano_depth.astype(np.float32)
        bad = (~np.isfinite(pano_depth)) | (pano_depth < 1e-3) | (pano_depth > MAX_DEPTH)
        pano_depth[bad] = MAX_DEPTH

        # if results.get('base_yaw', None) is not None:
        #     base_yaw = results['base_yaw']
        #     assert not self.random_base_yaw, 'Cannot set base_yaw in both transform class and dataset class.'
        # else:
        base_yaw = np.random.uniform(-180.0, 180.0) if self.random_base_yaw else self.base_yaw
        yaws = [base_yaw + off for off in self.yaw_offsets]

        H_out, W_out = self.out_hw
        cam2img = self._cam2img()
        img_list, depth_list, cam2img_list, cam2ego_list, lidar2cam_list, filename = [], [], [], [], [], []

        for i, yaw in enumerate(yaws):
            map_x, map_y = self._pano_maps(self.fov_deg, self.out_hw, yaw, 0.0,
                                           pano_rgb.shape)
            rgb_view = cv2.remap(
                pano_rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            depth_view = cv2.remap(
                pano_depth, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
            bad = (~np.isfinite(depth_view)) | (depth_view < 1e-3) | (depth_view > MAX_DEPTH)
            depth_view[bad] = MAX_DEPTH
            if self.to_float32:
                rgb_view = rgb_view.astype(np.float32)

            img_list.append(rgb_view)
            depth_list.append(torch.from_numpy(depth_view).float())
            cam2img_list.append(cam2img.copy())
            cam2ego_list.append(self._cam2ego(yaw))
            lidar2cam_list.append(np.eye(4, dtype=np.float32))
            filename.append(f'{os.path.basename(pano_img_path)}#view{i}')

        img_shape = (H_out, W_out)
        results['img'] = img_list
        results['depth'] = torch.stack(depth_list)
        results['cam2img'] = np.stack(cam2img_list, axis=0)
        results['cam2ego'] = np.stack(cam2ego_list, axis=0)
        results['lidar2cam'] = np.stack(lidar2cam_list, axis=0)
        results['filename'] = filename
        results['img_path'] = filename
        results['img_shape'] = img_shape
        results['ori_shape'] = img_shape
        results['pad_shape'] = img_shape
        results['scale_factor'] = 1.0
        num_channels = img_list[0].shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['num_views'] = len(yaws)
        results['base_yaw'] = base_yaw
        return results


@TRANSFORMS.register_module()
class PanoToMultiViewMultiFrame(PanoToMultiView):
    """Multi-frame version that shares a base yaw across frames."""

    def transform(self, results):
        frames = results.get('frames')
        if not frames:
            return super().transform(results)

        anchor_ego = frames[0]['ego2global']
        base_yaw = np.random.uniform(-180.0, 180.0) if self.random_base_yaw else self.base_yaw
        yaws = [base_yaw + off for off in self.yaw_offsets]

        cam2img = self._cam2img()
        img_list, depth_list, cam2img_list, cam2ego_list, lidar2cam_list, filename = [], [], [], [], [], []

        for fi, frame in enumerate(frames):
            pano_rgb = cv2.imread(frame['pano_img_path'], cv2.IMREAD_COLOR)
            if pano_rgb is None:
                raise FileNotFoundError(f'Cannot read pano image at {frame["pano_img_path"]}')
            pano_rgb = pano_rgb[:, :, ::-1]

            pano_depth = np.load(frame['pano_depth_path'])
            if pano_depth.ndim == 3 and pano_depth.shape[2] == 1:
                pano_depth = pano_depth[:, :, 0]
            pano_depth = pano_depth.astype(np.float32)
            bad = (~np.isfinite(pano_depth)) | (pano_depth < 1e-3) | (pano_depth > MAX_DEPTH)
            pano_depth[bad] = MAX_DEPTH

            for vi, yaw in enumerate(yaws):
                map_x, map_y = self._pano_maps(self.fov_deg, self.out_hw, yaw, 0.0,
                                               pano_rgb.shape)
                rgb_view = cv2.remap(
                    pano_rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
                depth_view = cv2.remap(
                    pano_depth, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
                bad = (~np.isfinite(depth_view)) | (depth_view < 1e-3) | (depth_view > MAX_DEPTH)
                depth_view[bad] = MAX_DEPTH
                if self.to_float32:
                    rgb_view = rgb_view.astype(np.float32)

                cam2ego_yaw = self._cam2ego(yaw)
                if fi == 0:
                    rel_cam2ego = cam2ego_yaw
                else:
                    rel_cam2ego = np.linalg.inv(anchor_ego) @ frame['ego2global'] @ cam2ego_yaw

                img_list.append(rgb_view)
                depth_list.append(torch.from_numpy(depth_view).float())
                cam2img_list.append(cam2img.copy())
                cam2ego_list.append(rel_cam2ego.astype(np.float32))
                lidar2cam_list.append(np.eye(4, dtype=np.float32))
                filename.append(f'{os.path.basename(frame["pano_img_path"])}#f{fi}_v{vi}')

        H_out, W_out = self.out_hw
        img_shape = (H_out, W_out)
        results['img'] = img_list
        results['depth'] = torch.stack(depth_list)
        results['cam2img'] = np.stack(cam2img_list, axis=0)
        results['cam2ego'] = np.stack(cam2ego_list, axis=0)
        results['lidar2cam'] = np.stack(lidar2cam_list, axis=0)
        results['filename'] = filename
        results['img_path'] = filename
        results['img_shape'] = img_shape
        results['ori_shape'] = img_shape
        results['pad_shape'] = img_shape
        results['scale_factor'] = 1.0
        num_channels = img_list[0].shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['num_views'] = len(filename)
        results['base_yaw'] = base_yaw
        return results


@TRANSFORMS.register_module()
class ApplyImgAugToDepth(BaseTransform):
    """Apply the recorded img_aug_mat (from ImageAug3D) to in-memory depth maps."""

    def __init__(self, key='depth'):
        self.key = key

    def transform(self, results):
        if 'img_aug_mat' not in results or self.key not in results:
            return results

        depth_maps = results[self.key]
        img_aug_mats = results['img_aug_mat']
        out = []
        for i in range(depth_maps.shape[0]):
            depth = depth_maps[i]
            post_rot = torch.as_tensor(img_aug_mats[i][:3, :3])
            post_tran = torch.as_tensor(img_aug_mats[i][:3, 3])
            assert post_rot[0, 1] == post_rot[1, 0] == 0

            h, w = depth.shape
            depth = depth[None, None]
            depth = F.interpolate(
                depth, (int(h * post_rot[1, 1] + 0.5),
                        int(w * post_rot[0, 0] + 0.5)),
                mode='bilinear',
                align_corners=False).squeeze(0).squeeze(0)
            depth = depth[int(post_tran[1]):, int(-post_tran[0]):]
            out.append(depth)

        results[self.key] = torch.stack(out)
        return results

@TRANSFORMS.register_module()
class LoadOccFromGLB(BaseTransform):
    """Crop local occupancy from MP3D glb and voxelize humans as cylinders."""

    def __init__(self,
                 half_xy=3.6,
                 z_up=0.4,
                 z_down=2.0,
                 pitch=0.1,
                 human_radius=0.2,
                 human_height=1.65):
        self.half_xy = half_xy
        self.z_up = z_up
        self.z_down = z_down
        self.pitch = pitch
        self.human_radius = human_radius
        self.human_height = human_height

    def _yaw_from_T_mp(self, T):
        return math.atan2(T[0, 2], T[2, 2])

    def _load_mesh(self, glb_path):
        mesh = trimesh.load(glb_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            try:
                geoms = mesh.to_geometry()
                mesh = (trimesh.util.concatenate(tuple(geoms.values()))
                        if isinstance(geoms, dict) else trimesh.util.concatenate(tuple(geoms)))
            except Exception:
                mesh = mesh.dump(concatenate=True)
        return mesh


    def _crop_occ_from_mp3d(self, glb_path, center_xyz, yaw_anchor_rad, half_xy=3.6, z_up=0.4, z_down=2.0, pitch=0.1):
        mesh = self._load_mesh(glb_path)
        if mesh is None or mesh.vertices.size == 0:
            return np.zeros((1, 1, 1), np.uint8)
        V = mesh.vertices.view(np.ndarray)
        F = mesh.faces.view(np.ndarray)
        cz, sz = math.cos(yaw_anchor_rad), math.sin(yaw_anchor_rad)
        R_wl = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float64)
        V_local = (V - center_xyz) @ R_wl.T
        inside = ((np.abs(V_local[:, 0]) <= half_xy) & (np.abs(V_local[:, 1]) <= half_xy) &
                (V_local[:, 2] <= z_up) & (V_local[:, 2] >= -z_down))
        keep_faces = inside[F].any(axis=1)
        F_keep = F[keep_faces]
        xy_dim = int(round((2 * half_xy) / pitch))
        z_dim = int(round((z_up + z_down) / pitch))
        occ = np.zeros((xy_dim, xy_dim, z_dim), np.uint8)
        if F_keep.size == 0:
            return occ
        uidx = np.unique(F_keep)
        remap = -np.ones(V_local.shape[0], int)
        remap[uidx] = np.arange(uidx.shape[0])
        F_local = remap[F_keep]
        V_local_crop = V_local[uidx]
        mesh_local = trimesh.Trimesh(vertices=V_local_crop, faces=F_local, process=False)
        vg = mesh_local.voxelized(pitch=pitch)
        pts = None if vg is None else vg.points
        if pts is None or len(pts) == 0:
            return occ
        y_min = -half_xy
        x_min = -half_xy
        z_min = -z_down
        for px, py, pz in pts:
            iy = int(np.round((py - y_min) / pitch))
            ix = int(np.round((px - x_min) / pitch))
            iz = int(np.round((pz - z_min) / pitch))
            if 0 <= iy < xy_dim and 0 <= ix < xy_dim and 0 <= iz < z_dim:
                occ[ix, iy, iz] = 1
        return occ


    def _voxelize_cylinder_in_ego(self, 
                                T_ego,
                                radius=0.2,
                                height=1.65,
                                half_xy=3.6,
                                z_up=0.4,
                                z_down=2.0,
                                pitch=0.1):
        cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
        cyl.apply_translation([0, 0, height / 2.0])

        V = cyl.vertices.view(np.ndarray)
        V_h = np.concatenate([V, np.ones((V.shape[0], 1))], axis=1)
        V_ego = (T_ego @ V_h.T).T[:, :3]
        cyl_ego = trimesh.Trimesh(vertices=V_ego, faces=cyl.faces.view(np.ndarray), process=False)

        vg = cyl_ego.voxelized(pitch=pitch)
        pts = None if vg is None else vg.points
        xy_dim = int(round((2 * half_xy) / pitch))
        z_dim = int(round((z_up + z_down) / pitch))
        grid = np.zeros((xy_dim, xy_dim, z_dim), np.uint8)
        if pts is None or len(pts) == 0:
            return grid

        x_min = -half_xy
        y_min = -half_xy
        z_min = -z_down
        for px, py, pz in pts:
            ix = int(np.round((px - x_min) / pitch))
            iy = int(np.round((py - y_min) / pitch))
            iz = int(np.round((pz - z_min) / pitch))
            if 0 <= ix < xy_dim and 0 <= iy < xy_dim and 0 <= iz < z_dim:
                grid[ix, iy, iz] = 1
        return grid


    def transform(self, results):
        occ_path = results.get('occ_path')
        if not occ_path:
            return results
        if not os.path.exists(occ_path):
            return results
        ego2global = results.get('ego2global')
        if ego2global is None:
            return results
        if ego2global.ndim == 3:
            ego_T = ego2global[0]
        else:
            ego_T = ego2global

        center_xyz = ego_T[:3, 3]
        yaw_anchor = self._yaw_from_T_mp(ego_T)
        occ_grid = self._crop_occ_from_mp3d(
            occ_path,
            center_xyz,
            yaw_anchor,
            half_xy=self.half_xy,
            z_up=self.z_up,
            z_down=self.z_down,
            pitch=self.pitch)

        human_grid = np.zeros_like(occ_grid)
        global2ego = np.linalg.inv(ego_T)
        human_poses = results.get('human_poses', {})
        for hp in human_poses.values():
            human2global = np.array(hp['human2global'], dtype=np.float32)
            human2ego = global2ego @ human2global
            z_e = human2ego[2, 3]
            if z_e < -self.z_down or z_e > self.z_up:
                continue
            human_grid |= self._voxelize_cylinder_in_ego(
                human2ego,
                radius=self.human_radius,
                height=self.human_height,
                half_xy=self.half_xy,
                z_up=self.z_up,
                z_down=self.z_down,
                pitch=self.pitch)

        results['occ_grid'] = occ_grid
        results['human_grid'] = human_grid

        # debug
        results_save_path = 'debug_outputs/loaded_data.npz'
        with open(results_save_path, "wb") as f:
            np.savez_compressed(
                f,
                img=np.stack(results['img']),
                depth=results['depth'].cpu().numpy(),
                cam2img=results['cam2img'],
                cam2ego=results['cam2ego'],
                ego2global=results['ego2global'],
                lidar2cam=results['lidar2cam'],
                occ_grid=results['occ_grid'],
                human_grid=results['human_grid'],
                num_views=results['num_views'],
                base_yaw=results['base_yaw'],
                depth_box_mask=results.get('inside_box_mask', None).cpu().numpy() if 'inside_box_mask' in results else None,
            )
        print(f'Saved loaded data to {results_save_path}')
        exit(0)
        return results


@TRANSFORMS.register_module()
class DepthBoxMask(BaseTransform):
    """Mask pixels whose 3D backprojection falls inside the anchor-oriented box."""

    def __init__(self,
                 half_xy=3.6,
                 z_up=0.4,
                 z_down=2.0,
                 use_mask=True,
                 allow_aug=False):
        self.half_xy = half_xy
        self.z_up = z_up
        self.z_down = z_down
        self.use_mask = use_mask
        self.allow_aug = allow_aug

    def transform(self, results):
        if not self.use_mask:
            return results
        if 'img_aug_mat' in results and results['img_aug_mat'] is not None and not self.allow_aug:
            raise RuntimeError('DepthBoxMask does not support img_aug_mat when allow_aug=False.')

        depth = results.get('depth')
        cam2ego = results.get('cam2ego')
        cam2img = results.get('cam2img')
        base_yaw = results.get('base_yaw', 0.0)
        if depth is None or cam2ego is None or cam2img is None:
            return results

        masks = []
        c = math.cos(math.radians(base_yaw))
        s = math.sin(math.radians(base_yaw))
        Rz = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                          device=depth.device, dtype=depth.dtype)

        for i in range(depth.shape[0]):
            d = depth[i]
            h, w = d.shape
            if d.dim() != 2:
                raise ValueError('DepthBoxMask expects per-view depth of shape (H, W).')
            ys = torch.arange(h, device=d.device, dtype=d.dtype)
            xs = torch.arange(w, device=d.device, dtype=d.dtype)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

            K = torch.as_tensor(cam2img[i][:3, :3], device=d.device, dtype=d.dtype)
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]

            z = d
            valid = z > 0
            x = (grid_x - cx) / fx * z
            y = (grid_y - cy) / fy * z
            pts_cam = torch.stack([x, y, z], dim=-1)

            c2e = torch.as_tensor(cam2ego[i], device=d.device, dtype=d.dtype)
            pts_ego = pts_cam.reshape(-1, 3) @ c2e[:3, :3].T + c2e[:3, 3]
            pts_ego = pts_ego @ Rz.T
            pts_ego = pts_ego.reshape(h, w, 3)

            inside = valid
            inside = inside & (pts_ego[..., 0].abs() <= self.half_xy)
            inside = inside & (pts_ego[..., 1].abs() <= self.half_xy)
            inside = inside & (pts_ego[..., 2] <= self.z_up) & (pts_ego[..., 2] >= -self.z_down)
            masks.append(inside)

        results['inside_box_mask'] = torch.stack(masks)
        return results


@TRANSFORMS.register_module()
class CollectHumanPoses(BaseTransform):
    """Project human poses into anchor ego frame and collect IDs/positions."""

    def transform(self, results):
        ego2global = results.get('ego2global')
        human_poses_all = results.get('human_poses_all', [])
        if ego2global is None or not human_poses_all:
            return results

        anchor_ego = torch.as_tensor(ego2global[0], dtype=torch.float32)
        global2anchor = torch.linalg.inv(anchor_ego)

        ids = []
        positions = []
        for frame_idx, hp_dict in enumerate(human_poses_all):
            frame_positions = []
            frame_ids = []
            for hid, hp in hp_dict.items():
                human2global = torch.as_tensor(hp['human2global'], dtype=torch.float32)
                human2anchor = global2anchor @ human2global
                frame_ids.append(int(hid))
                frame_positions.append(human2anchor[:3, 3])
            if frame_positions:
                ids.append(torch.tensor(frame_ids, dtype=torch.long))
                positions.append(torch.stack(frame_positions))
            else:
                ids.append(torch.empty(0, dtype=torch.long))
                positions.append(torch.empty(0, 3))

        results['human_ids'] = ids
        results['human_positions'] = positions
        return results


@TRANSFORMS.register_module()
class CustomAugmentation(BaseTransform):
    def __init__(self,
                 rand_flip=False,
                 rand_rotate=False,
                 force_happen=False, # debug feature
                ):
        self.rand_flip = rand_flip
        self.rand_rotate = rand_rotate
        self.force_happen = force_happen

    def transform(self, results):
        if self.force_happen:
            assert self.rand_flip == True or self.rand_rotate == True, \
                'You forced nothing to happen...'
        if not self.rand_flip and not self.rand_rotate:
            return results
        
        if self.rand_flip:
            do_flip = int(np.random.randint(0, 2)) if not self.force_happen else 1
            if do_flip == 0:
                return results

            # 1) flip all view images (H,W,C) along W
            if 'img' in results:
                results['img'] = [np.flip(im, axis=1).copy() for im in results['img']]

            # 2) flip depth (V,H,W) along W
            if 'depth' in results:
                depth = results['depth']
                if torch.is_tensor(depth):
                    results['depth'] = torch.flip(depth, dims=[-1])
                else:
                    results['depth'] = np.flip(depth, axis=-1).copy()

            # 3) swap left/right view content (fixed order: front,left,back,right)
            #    handle multi-frame by swapping within each block of 4
            if 'img' in results:
                imgs = results['img']
                for base in range(0, len(imgs), 4):
                    if base + 3 < len(imgs):
                        imgs[base + 1], imgs[base + 3] = imgs[base + 3], imgs[base + 1]
                results['img'] = imgs

            if 'depth' in results:
                depth = results['depth']
                if torch.is_tensor(depth):
                    depth = depth.clone()
                    for base in range(0, depth.shape[0], 4):
                        if base + 3 < depth.shape[0]:
                            tmp = depth[base + 1].clone()
                            depth[base + 1] = depth[base + 3]
                            depth[base + 3] = tmp
                    results['depth'] = depth
                else:
                    depth = depth.copy()
                    for base in range(0, depth.shape[0], 4):
                        if base + 3 < depth.shape[0]:
                            depth[base + 1], depth[base + 3] = depth[base + 3], depth[base + 1]
                    results['depth'] = depth

            # 4) flip traversable mask along W (C,H,W)
            if 'traversable_mask' in results:
                tm = results['traversable_mask']
                if torch.is_tensor(tm):
                    results['traversable_mask'] = torch.flip(tm, dims=[-1])
                else:
                    results['traversable_mask'] = np.flip(tm, axis=-1).copy()

        # --- random rotate (yaw) ---
        if self.rand_rotate:
            # k = 0,1,2,3 corresponds to 0,90,180,270 deg CCW
            k = int(np.random.choice([0, 1, 2, 3])) if not self.force_happen else 1
            if k != 0:
                # view permutation per 4-view block [F, L, B, R]
                # CCW 90: [L, B, R, F], 180: [B, R, F, L], 270: [R, F, L, B]
                perm_map = {
                    1: [1, 2, 3, 0],
                    2: [2, 3, 0, 1],
                    3: [3, 0, 1, 2],
                }

                if 'img' in results:
                    imgs = results['img']
                    out = imgs.copy()
                    for base in range(0, len(imgs), 4):
                        if base + 3 < len(imgs):
                            idx = perm_map[k]
                            out[base + 0] = imgs[base + idx[0]]
                            out[base + 1] = imgs[base + idx[1]]
                            out[base + 2] = imgs[base + idx[2]]
                            out[base + 3] = imgs[base + idx[3]]
                    results['img'] = out

                if 'depth' in results:
                    depth = results['depth']
                    if torch.is_tensor(depth):
                        depth = depth.clone()
                        for base in range(0, depth.shape[0], 4):
                            if base + 3 < depth.shape[0]:
                                idx = perm_map[k]
                                tmp = depth[base:base+4].clone()
                                depth[base + 0] = tmp[idx[0]]
                                depth[base + 1] = tmp[idx[1]]
                                depth[base + 2] = tmp[idx[2]]
                                depth[base + 3] = tmp[idx[3]]
                        results['depth'] = depth
                    else:
                        depth = depth.copy()
                        for base in range(0, depth.shape[0], 4):
                            if base + 3 < depth.shape[0]:
                                idx = perm_map[k]
                                tmp = depth[base:base+4].copy()
                                depth[base + 0] = tmp[idx[0]]
                                depth[base + 1] = tmp[idx[1]]
                                depth[base + 2] = tmp[idx[2]]
                                depth[base + 3] = tmp[idx[3]]
                        results['depth'] = depth

                # BEV label rotation: +H=front, +W=left
                # CCW robot yaw => clockwise array rotation
                rot_k = (-k) % 4
                if 'traversable_mask' in results:
                    tm = results['traversable_mask']
                    if torch.is_tensor(tm):
                        results['traversable_mask'] = torch.rot90(tm, k=rot_k, dims=(-2, -1))
                    else:
                        results['traversable_mask'] = np.rot90(tm, k=rot_k, axes=(-2, -1)).copy()


        return results