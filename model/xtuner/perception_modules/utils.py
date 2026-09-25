import torch
try:
    from xtuner.perception_modules.ops.bev_pool_v2 import bev_pool_v2
except:
    bev_pool_v2 = None

import torch.nn as nn
import torch.nn.functional as F

# helper function for LSS BEV pooling # TODO: remove hardcoded values
def lss_bev_pool(
    depth_probs_bn,      # (B*N, D, fH, fW), softmax over D
    img_feats_bn,        # (B*N, C, fH, fW), context feats
    sensor2egos,         # (B, N, 4, 4)
    intrinsics,          # (B, N, 3, 3)
    post_rots=None,           # (B, N, 3, 3)
    post_trans=None,          # (B, N, 3)
    bda=None,                 # (B, 3, 3)
    grid_config=dict(
        x=[-3.6, 3.6, 0.2],   # 36
        y=[-3.6, 3.6, 0.2],   # 36
        z=[-1.8, 0.2, 2.0],   # 1
        depth=[0.0, 4.8, 0.1],    # 48
    ),         # dict with 'x','y','z','depth' tuples (min, max, interval)
    input_size=(448, 448),          # (H_in, W_in)
    downsample=14, # == patch size basically
    collapse_z=True,
    pooling="sum",  # "sum" or "avg"
):
    assert depth_probs_bn.is_contiguous(), "depth to lss_bev_pool must be contiguous"
    assert img_feats_bn.is_contiguous(), "feats to lss_bev_pool must be contiguous"
    assert bev_pool_v2 is not None, "bev_pool_v2 is required for lss_bev_pool but failed to import"


    B, N = sensor2egos.shape[:2]
    if post_rots is None:
        post_rots = torch.eye(3, device=depth_probs_bn.device).view(1,1,3,3).expand(B,N,-1,-1)  # (B,N,3,3)
    if post_trans is None:
        post_trans = torch.zeros(B, N, 3, device=depth_probs_bn.device)  # (B,N,3)
    if bda is None:
        bda = torch.eye(3, device=depth_probs_bn.device).view(1,3,3).expand(B,-1,-1) # (B,3,3)
    # Precompute grid info
    grid_lower = torch.tensor([grid_config[a][0] for a in ('x','y','z')], device=depth_probs_bn.device)
    grid_interval = torch.tensor([grid_config[a][2] for a in ('x','y','z')], device=depth_probs_bn.device)
    grid_size = torch.tensor([(grid_config[a][1]-grid_config[a][0])/grid_config[a][2] for a in ('x','y','z')], device=depth_probs_bn.device)
    Dx, Dy, Dz = grid_size.long()

    # Helper: frustum template (u,v,d) at feature resolution
    def create_frustum(depth_cfg, input_size, downsample):
        H_in, W_in = input_size
        Hf, Wf = H_in // downsample, W_in // downsample
        d = torch.arange(*depth_cfg, device=depth_probs_bn.device).view(-1,1,1).expand(-1,Hf,Wf)
        u = torch.linspace(0, W_in-1, Wf, device=depth_probs_bn.device).view(1,1,Wf).expand_as(d)
        v = torch.linspace(0, H_in-1, Hf, device=depth_probs_bn.device).view(1,Hf,1).expand_as(d)
        return torch.stack((u,v,d), -1)  # (D, fH, fW, 3)
    frustum = create_frustum(grid_config['depth'], input_size, downsample)  # (D,fH,fW,3)
    D, fH, fW, _ = frustum.shape

    # Helper: project frustum to ego
    def get_ego_coor():
        B, N, _, _ = sensor2egos.shape
        pts = frustum - post_trans.view(B,N,1,1,1,3)           # (B,N,D,fH,fW,3)
        pts = torch.inverse(post_rots).view(B,N,1,1,1,3,3).matmul(pts.unsqueeze(-1))
        pts = torch.cat((pts[..., :2, :] * pts[..., 2:3, :], pts[..., 2:3, :]), 5)
        combine = sensor2egos[:, :, :3, :3].matmul(torch.inverse(intrinsics))
        pts = combine.view(B,N,1,1,1,3,3).matmul(pts).squeeze(-1)
        pts += sensor2egos[:, :, :3, 3].view(B,N,1,1,1,3)
        pts = bda.view(B,1,1,1,1,3,3).matmul(pts.unsqueeze(-1)).squeeze(-1)
        return pts  # (B,N,D,fH,fW,3)

    # Helper: prepare ranks
    def voxel_pooling_prepare(coor):
        B, N, Dp, Hp, Wp, _ = coor.shape
        num_pts = B * N * Dp * Hp * Wp
        ranks_depth = torch.arange(num_pts, device=coor.device, dtype=torch.int)
        ranks_feat = torch.arange(num_pts // Dp, device=coor.device, dtype=torch.int)
        ranks_feat = ranks_feat.view(B,N,1,Hp,Wp).expand(B,N,Dp,Hp,Wp).flatten()

        # coor_scaled = ((coor - grid_lower) / grid_interval).long().view(num_pts, 3)
        coor_scaled = torch.floor((coor - grid_lower) / grid_interval).long().view(num_pts, 3)  # (num_pts, 3)
        batch_idx = torch.arange(B, device=coor.device).view(B,1).expand(B, num_pts // B).reshape(num_pts,1)
        coor_full = torch.cat((coor_scaled, batch_idx), 1)  # (num_pts,4)

        kept = (coor_full[:,0] >= 0) & (coor_full[:,0] < grid_size[0]) & \
               (coor_full[:,1] >= 0) & (coor_full[:,1] < grid_size[1]) & \
               (coor_full[:,2] >= 0) & (coor_full[:,2] < grid_size[2])
        if kept.sum() == 0:
            return None, None, None, None, None

        coor_full, ranks_depth, ranks_feat = coor_full[kept], ranks_depth[kept], ranks_feat[kept]
        ranks_bev = coor_full[:,3] * (Dz*Dy*Dx) + coor_full[:,2] * (Dy*Dx) + coor_full[:,1] * Dx + coor_full[:,0]
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept_flag = torch.ones(ranks_bev.shape[0], device=coor.device, dtype=torch.bool)
        kept_flag[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept_flag)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int(), ranks_depth.int(), ranks_feat.int(), interval_starts.int(), interval_lengths.int()

    # Main flow
    coor = get_ego_coor()  # (B,N,D,fH,fW,3)
    ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths = voxel_pooling_prepare(coor)
    if ranks_feat is None:
        # no points fall inside grid
        return torch.zeros(
            (sensor2egos.shape[0], img_feats_bn.shape[1],
             1 if not collapse_z else grid_size[2].long().item(),
             Dy, Dx),
            device=img_feats_bn.device, dtype=img_feats_bn.dtype
        )
    feat = img_feats_bn.view(-1, fH, fW, img_feats_bn.shape[1])         # (B*N, fH, fW, C)
    depth = depth_probs_bn.view(-1, D, fH, fW)                          # (B*N, D, fH, fW)
    feat = feat.view(sensor2egos.shape[0], -1, fH, fW, img_feats_bn.shape[1])  # (B,N,fH,fW,C)
    depth = depth.view(sensor2egos.shape[0], -1, D, fH, fW)             # (B,N,D,fH,fW)

    bev_shape = (depth.shape[0], int(grid_size[2]), int(grid_size[1]), int(grid_size[0]), feat.shape[-1])
    bev = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_shape, interval_starts, interval_lengths)  # (B,C,Dz,Dy,Dx)

    # new feature, average pooling
    if pooling not in ("sum", "avg"):
        raise ValueError(f"Unsupported pooling={pooling}")

    if pooling == "avg":
        # count contributions per voxel using depth one-hot
        ones_feat = torch.ones_like(img_feats_bn[:, :1, :, :]).contiguous()  # (B*N,1,fH,fW)

        bev_shape_cnt = (depth.shape[0], int(grid_size[2]), int(grid_size[1]), int(grid_size[0]), 1)
        cnt = bev_pool_v2(
            depth_probs_bn,   # use real depth one-hot
            ones_feat,
            ranks_depth, ranks_feat, ranks_bev,
            bev_shape_cnt, interval_starts, interval_lengths
        )  # (B,1,Dz,Dy,Dx)

        if collapse_z:
            cnt = torch.cat(cnt.unbind(dim=2), dim=1)  # (B, Dz, Dy, Dx)
            C = img_feats_bn.shape[1]
            cnt = cnt.repeat_interleave(C, dim=1)      # (B, C*Dz, Dy, Dx)

        cnt = cnt.clamp_min(1.0)
        bev = bev / cnt



    if collapse_z:
        bev = torch.cat(bev.unbind(dim=2), dim=1)  # (B, C*Dz, Dy, Dx)
    return bev

def bev_to_ego(waypoint_heatmaps, pitch=0.2, H=36, W=36):
    """
    waypoint_heatmaps: (B, 6, H, W)
    """
    xs = torch.linspace(-3.6, 3.6-pitch, H, device=waypoint_heatmaps.device) + pitch / 2
    ys = torch.linspace(-3.6, 3.6-pitch, W, device=waypoint_heatmaps.device) + pitch / 2
    waypoints_bev_coords = torch.argmax(
        waypoint_heatmaps.view(waypoint_heatmaps.shape[0], waypoint_heatmaps.shape[1], -1), 
        dim=-1
    )  # B, 6
    waypoints_x_idx = waypoints_bev_coords // W
    waypoints_y_idx = waypoints_bev_coords % W
    waypoint_preds = torch.stack((
        xs[waypoints_x_idx],
        ys[waypoints_y_idx]
    ), dim=-1) # B, 6, 2

    return waypoint_preds # B, 6, 2


def gaussian_2d(radius, sigma=None, device=None):
    """Create a 2D Gaussian kernel with the given radius."""
    diameter = 2 * radius + 1
    if sigma is None:
        sigma = diameter / 6.0  # ≈ radius / 3
    coords = torch.arange(0, diameter, device=device, dtype=torch.float32)
    x, y = torch.meshgrid(coords, coords, indexing='ij')
    center = radius
    g = torch.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
    return g


def draw_gaussian_on_heatmap(heatmap, center, radius, sigma=None):
    """Draw a Gaussian on a heatmap in-place at integer center (x_idx, y_idx)."""
    x_idx, y_idx = int(center[0]), int(center[1])
    H, W = heatmap.shape[-2:]
    diameter = 2 * radius + 1
    gaussian = gaussian_2d(radius, sigma=sigma, device=heatmap.device)

    x0, y0 = x_idx - radius, y_idx - radius
    x1, y1 = x_idx + radius + 1, y_idx + radius + 1

    g_x0, g_y0 = 0, 0
    g_x1, g_y1 = diameter, diameter

    if x0 < 0:
        g_x0 = -x0
        x0 = 0
    if y0 < 0:
        g_y0 = -y0
        y0 = 0
    if x1 > H:
        g_x1 = diameter - (x1 - H)
        x1 = H
    if y1 > W:
        g_y1 = diameter - (y1 - W)
        y1 = W

    if (x0 >= x1) or (y0 >= y1):
        return heatmap

    heatmap[..., x0:x1, y0:y1] = torch.maximum(
        heatmap[..., x0:x1, y0:y1],
        gaussian[g_x0:g_x1, g_y0:g_y1]
    )
    return heatmap


def build_waypoint_heatmaps(waypoints, H=36, W=36, pitch=0.2, radius=2):
    """
    Build Gaussian heatmaps for waypoints.

    Args:
        waypoints (Tensor): (B, 6, 2) in meters (x, y) in ego frame.
    Returns:
        Tensor: (B, 6, H, W) heatmaps.
    """
    device = waypoints.device
    heatmaps = torch.zeros((waypoints.shape[0], waypoints.shape[1], H, W), device=device)
    x_offset = 3.6 - pitch / 2
    y_offset = 3.6 - pitch / 2
    x_idx = torch.round((waypoints[..., 0] + x_offset) / pitch).long()
    y_idx = torch.round((waypoints[..., 1] + y_offset) / pitch).long()

    for b in range(waypoints.shape[0]):
        for k in range(waypoints.shape[1]):
            xi, yi = x_idx[b, k].item(), y_idx[b, k].item()
            if 0 <= xi < H and 0 <= yi < W:
                draw_gaussian_on_heatmap(heatmaps[b, k], (xi, yi), radius=radius)
    return heatmaps

class LN2d(nn.Module):
    """A LayerNorm variant, popularized by Transformers, that performs
    pointwise mean and variance normalization over the channel dimension for
    inputs that have shape (batch_size, channels, height, width)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class _BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(x)
        return self.act(out + identity)


class ResNetBEV(nn.Module):
    """
    Returns C2..C5 like a standard ResNet backbone (no maxpool stem).
    Input/Output spatial order is whatever you use; just be consistent.
    """
    def __init__(self, cin, base=64, depth=10, norm='bn'):
        super().__init__()
        if depth == 10:
            layers = [1, 1, 1, 1]
        elif depth == 18:
            layers = [2, 2, 2, 2]
        else:
            raise ValueError("depth must be 10 or 18")
        
        self.cin = cin

        # Stem: keep resolution
        self.stem = nn.Sequential(
            nn.Conv2d(cin, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

        # Standard stage widths for BasicBlock ResNet
        c2, c3, c4, c5 = base, base * 2, base * 4, base * 8
        self.layer1 = self._make_stage(base,  c2, layers[0], stride=1)  # C2
        self.layer2 = self._make_stage(c2,    c3, layers[1], stride=2)  # C3
        self.layer3 = self._make_stage(c3,    c4, layers[2], stride=2)  # C4
        self.layer4 = self._make_stage(c4,    c5, layers[3], stride=2)  # C5

        self.out_channels = (c2, c3, c4, c5)
        
        assert norm in ('bn', 'gn', 'ln'), "norm must be 'bn', 'gn', or 'ln'"
        if norm == 'gn':
            gn = 32  # default group number
            for parent in self.modules():
                for name, child in parent.named_children():
                    if isinstance(child, nn.BatchNorm2d):
                        setattr(parent, name, nn.GroupNorm(gn, child.num_features))

        elif norm == 'ln':
            for parent in self.modules():
                for name, child in parent.named_children():
                    if isinstance(child, nn.BatchNorm2d):
                        setattr(parent, name, LN2d(child.num_features))


    @staticmethod
    def _make_stage(in_ch, out_ch, blocks, stride):
        layers = [_BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(_BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return x, c2, c3, c4, c5


class FPNBEV(nn.Module):
    """
    Standard top-down FPN producing P2..P5, all with fpn_dim channels.
    """
    def __init__(self, in_channels_list, fpn_dim=256):
        super().__init__()
        c2, c3, c4, c5 = in_channels_list
        self.lat2 = nn.Conv2d(c2, fpn_dim, 1)
        self.lat3 = nn.Conv2d(c3, fpn_dim, 1)
        self.lat4 = nn.Conv2d(c4, fpn_dim, 1)
        self.lat5 = nn.Conv2d(c5, fpn_dim, 1)

        self.out2 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.out3 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.out4 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.out5 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)

    def forward(self, c2, c3, c4, c5):
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")

        p5 = self.out5(p5)
        p4 = self.out4(p4)
        p3 = self.out3(p3)
        p2 = self.out2(p2)
        return p2, p3, p4, p5


class BEVResNetFPN(nn.Module):
    """
    What you want:
    - internal: ResNet backbone + FPN
    - external: one final local BEV feature map (B, C_out, Dx, Dy)
    """
    def __init__(self, cin, depth=10, base=64, fpn_dim=256, out_dim=256):
        super().__init__()
        self.backbone = ResNetBEV(cin=cin, base=base, depth=depth)
        self.fpn = FPNBEV(self.backbone.out_channels, fpn_dim=fpn_dim)

        # Fuse multi-scale into a single P2-resolution feature
        self.fuse = nn.Sequential(
            nn.Conv2d(fpn_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, bev_feats):
        # bev_feats: (B, Cin, Dx, Dy)
        c2, c3, c4, c5 = self.backbone(bev_feats)
        p2, p3, p4, p5 = self.fpn(c2, c3, c4, c5)

        # upsample all to p2 resolution and sum (simple + standard enough)
        p3u = F.interpolate(p3, size=p2.shape[-2:], mode="nearest")
        p4u = F.interpolate(p4, size=p2.shape[-2:], mode="nearest")
        p5u = F.interpolate(p5, size=p2.shape[-2:], mode="nearest")
        out = p2 + p3u + p4u + p5u

        return self.fuse(out)  # (B, out_dim, Dx, Dy)


class LN3d(nn.Module):
    """
    LayerNorm over channel dimension for 5D tensors shaped (B, C, D, H, W).
    Matches the LN2d behavior just extended to 3D volumes.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x

class BasicBlock3D(nn.Module):
    """ResNet BasicBlock using Conv3d + LN3d + ReLU, preserving shape."""
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = LN3d(channels, eps=eps)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = LN3d(channels, eps=eps)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + identity)


class MiniSECONDEncoderDense(nn.Module):
    """
    Input:  (B, C_in, H, W, Z) with H=W=128, Z=16 by your convention
    Output: (B, 256, H, W) via Z:16->8->4 and C:3->16->32->64 then flatten (64*4)
    """
    def __init__(self, in_channels: int = 3, eps: float = 1e-6):
        super().__init__()

        # stem: 3 -> 16
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            LN3d(16, eps=eps),
            nn.ReLU(inplace=True),
        )

        # stage1 refine @16 then Z-downsample -> 32 (Z:16->8)
        self.s1 = nn.Sequential(BasicBlock3D(16, eps=eps), BasicBlock3D(16, eps=eps))
        self.zdown1 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=(2, 1, 1), padding=1, bias=False),
            LN3d(32, eps=eps),
            nn.ReLU(inplace=True),
        )

        # stage2 refine @32 then Z-downsample -> 64 (Z:8->4)
        self.s2 = nn.Sequential(BasicBlock3D(32, eps=eps), BasicBlock3D(32, eps=eps))
        self.zdown2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=(2, 1, 1), padding=1, bias=False),
            LN3d(64, eps=eps),
            nn.ReLU(inplace=True),
        )

        # stage3 refine @64 (no downsample)
        self.s3 = nn.Sequential(BasicBlock3D(64, eps=eps), BasicBlock3D(64, eps=eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,C,H,W,Z) -> (B,C,Z,H,W) for Conv3d
        x = x.permute(0, 1, 4, 2, 3).contiguous()

        x = self.stem(x)     # (B,16,16,H,W)
        x = self.s1(x)
        x = self.zdown1(x)   # (B,32, 8,H,W)
        x = self.s2(x)
        x = self.zdown2(x)   # (B,64, 4,H,W)
        x = self.s3(x)

        # flatten Z into channels: (B,64,4,H,W) -> (B,256,H,W)
        b, c, z, h, w = x.shape
        return x.permute(0, 1, 3, 4, 2).contiguous().view(b, c * z, h, w)