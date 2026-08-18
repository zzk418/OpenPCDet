#!/usr/bin/env python3
"""
Quick PointRCNN Segmentation Head inference on a PCD file.
Manually builds PointNet2MSG + PointHeadBox to bypass broken numba import chain.
Visualizes per-point foreground/background predictions.
"""

import sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import open3d as o3d

# ── 1. PointNet2 MSG Backbone (Simplified for PointRCNN) ──────────────────────

def square_distance(src, dst):
    """Euclidean squared distance matrix (B, N, 3), (B, M, 3) -> (B, N, M)"""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1).view(B, N, 1)
    dist += torch.sum(dst ** 2, dim=-1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """Index into a (B, N, C) tensor with a (B, S, ...) index tensor -> (B, S, ..., C)"""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """FPS: (B, N, 3) -> (B, npoint) indices"""
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """Ball query: (B, N, 3), (B, S, 3) -> (B, S, nsample) indices"""
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    # Clamp to valid range [0, N-1] in case group_first also == N
    group_idx = group_idx.clamp(0, N - 1)
    return group_idx


class SetAbstractionMsg(nn.Module):
    """Multi-scale grouping Set Abstraction (matching PointRCNN config)"""

    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super().__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_ch = in_channel + 3  # +3 for relative xyz
            for out_ch in mlp_list[i]:
                convs.append(nn.Conv2d(last_ch, out_ch, 1))
                bns.append(nn.BatchNorm2d(out_ch))
                last_ch = out_ch
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """xyz: (B,N,3), points: (B,C,N) or None"""
        B, N, _ = xyz.shape
        new_xyz = index_points(xyz, farthest_point_sample(xyz, self.npoint))  # (B, npoint, 3)

        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)  # (B, npoint, K)
            grouped_xyz = index_points(xyz, group_idx)  # (B, npoint, K, 3)
            grouped_xyz -= new_xyz.view(B, self.npoint, 1, 3)  # relative coords
            if points is not None:
                grouped_points = index_points(points.permute(0, 2, 1), group_idx)  # (B, npoint, K, C)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz
            grouped_points = grouped_points.permute(0, 3, 1, 2).contiguous()  # (B,S,K,C) -> (B,C,S,K)
            for j, conv in enumerate(self.conv_blocks[i]):
                grouped_points = F.relu(self.bn_blocks[i][j](conv(grouped_points)))
            new_points = torch.max(grouped_points, -1)[0]  # max pool (B, C', npoint)
            new_points_list.append(new_points)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat  # (B, npoint, 3), (B, sum_mlp_out, npoint)


class FeaturePropagation(nn.Module):
    """3-NN interpolation + skip connection"""

    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = in_channel
        for out_ch in mlp:
            self.mlp_convs.append(nn.Conv1d(last_ch, out_ch, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_ch))
            last_ch = out_ch

    def forward(self, xyz1, xyz2, points1, points2):
        """xyz1/points1: target (higher-res), xyz2/points2: source (lower-res) -> (B, C', N1)"""
        B, N1, _ = xyz1.shape
        _, N2, _ = xyz2.shape
        if points2 is not None:
            dists = square_distance(xyz1, xyz2)  # (B, N1, N2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # 3-NN
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(
                index_points(points2.permute(0, 2, 1), idx) * weight.view(B, N1, 3, 1), dim=2
            )
        else:
            interpolated_points = points2
        if points1 is not None:
            interpolated_points = interpolated_points.permute(0, 2, 1)
            new_points = torch.cat([interpolated_points, points1], dim=1)
        else:
            new_points = interpolated_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            new_points = F.relu(self.mlp_bns[i](conv(new_points)))
        return new_points


class PointNet2MSG(nn.Module):
    """PointNet2 MSG backbone matching KITTI PointRCNN config"""

    def __init__(self, input_channels=4):  # xyz + intensity
        super().__init__()
        # SA layers: npoint, radius_list, nsample_list, mlp_list
        self.sa1 = SetAbstractionMsg(4096, [0.1, 0.5], [16, 32], input_channels,
                                     [[16, 16, 32], [32, 32, 64]])
        self.sa2 = SetAbstractionMsg(1024, [0.5, 1.0], [16, 32], 32 + 64,
                                     [[64, 64, 128], [64, 96, 128]])
        self.sa3 = SetAbstractionMsg(256, [1.0, 2.0], [16, 32], 128 + 128,
                                     [[128, 196, 256], [128, 196, 256]])
        self.sa4 = SetAbstractionMsg(64, [2.0, 4.0], [16, 32], 256 + 256,
                                     [[256, 256, 512], [256, 384, 512]])

        # FP layers: in_channel (skip_concat), mlp
        self.fp4 = FeaturePropagation(512 + 512 + 256 + 256, [512, 512])  # sa4_out + sa3_out -> 512
        self.fp3 = FeaturePropagation(512 + 128 + 128, [512, 512])        # fp4_out + sa2_out -> 512
        self.fp2 = FeaturePropagation(512 + 32 + 64, [256, 256])          # fp3_out + sa1_out -> 256
        self.fp1 = FeaturePropagation(256 + input_channels, [128, 128])   # fp2_out + input -> 128

    def forward(self, xyz, features):
        """xyz: (B,N,3), features: (B,N,C) or None"""
        feat_in = features.permute(0, 2, 1) if features is not None else None  # (B,C,N)
        l0_xyz = xyz
        l0_points = feat_in

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points_fp = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points_fp = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points_fp)
        l1_points_fp = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points_fp)
        l0_points_fp = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points_fp)

        return l0_points_fp  # (B, 128, N)


# ── 2. PointHeadBox — Segmentation Head ──────────────────────────────────────

class PointHeadBox(nn.Module):
    """PointRCNN segmentation head: per-point foreground/background logits"""

    def __init__(self, in_channels=128, num_classes=1):
        super().__init__()
        self.cls_fc = nn.Sequential(
            nn.Conv1d(in_channels, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, num_classes, 1),
        )

    def forward(self, point_features):
        """point_features: (B, C, N) -> per-point logits (B, 1, N)"""
        return self.cls_fc(point_features)


# ── 3. Full PointRCNN (Backbone + Seg Head only) ─────────────────────────────

class PointRCNNSegHead(nn.Module):
    def __init__(self, input_channels=1, num_classes=1):
        super().__init__()
        self.backbone = PointNet2MSG(input_channels=input_channels)
        self.seg_head = PointHeadBox(in_channels=128, num_classes=num_classes)

    def forward(self, xyz, features):
        """xyz: (B,N,3), features: (B,N,1) -> seg_logits: (B, 1, N)"""
        point_feat = self.backbone(xyz, features)  # (B, 128, N)
        seg_logits = self.seg_head(point_feat)       # (B, 1, N)
        return seg_logits


# ── 4. Inference & Visualization ─────────────────────────────────────────────

def load_pcd_as_tensor(pcd_path, num_points=16384):
    """Load PCD, sample to num_points, return xyz and intensity tensors."""
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points, dtype=np.float32)  # (N, 3)
    N = pts.shape[0]

    # Normalize to [0, 1] range for stable inference
    xyz = pts.copy()
    for i in range(3):
        lo, hi = xyz[:, i].min(), xyz[:, i].max()
        if hi - lo > 1e-6:
            xyz[:, i] = (xyz[:, i] - lo) / (hi - lo)

    # If there are colors, use as intensity; else use z-coordinate
    if pcd.has_colors:
        colors = np.asarray(pcd.colors, dtype=np.float32)
        intensity = np.mean(colors, axis=1, keepdims=True)  # grayscale
    else:
        intensity = xyz[:, 2:3].copy()  # use z as pseudo-intensity

    # Sample to fixed number
    if N > num_points:
        idx = np.random.choice(N, num_points, replace=False)
        xyz = xyz[idx]
        intensity = intensity[idx]
    elif N < num_points:
        idx = np.random.choice(N, num_points, replace=True)
        xyz = xyz[idx]
        intensity = intensity[idx]

    xyz_t = torch.from_numpy(xyz).unsqueeze(0).cuda()  # (1, M, 3)
    feat_t = torch.from_numpy(intensity).unsqueeze(0).cuda()  # (1, M, 1)
    return xyz_t, feat_t, pts


def visualize_segmentation(xyz_np, seg_probs, save_path="output/pointrcnn_seg_result.png"):
    """Create BEV + 3D visualization of segmentation results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Rescale probability for better contrast
    prob = (seg_probs - seg_probs.min()) / (seg_probs.max() - seg_probs.min() + 1e-8)

    fig = plt.figure(figsize=(16, 6))

    # ── BEV (top-down) view ──
    ax1 = fig.add_subplot(1, 3, 1)
    scatter1 = ax1.scatter(xyz_np[:, 0], xyz_np[:, 1], c=prob, cmap='coolwarm',
                           s=3.0, alpha=0.8, edgecolors='none')
    ax1.set_title('BEV — Foreground Probability')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_aspect('equal')
    plt.colorbar(scatter1, ax=ax1, label='P(foreground)')

    # ── Binary hard assignment ──
    ax2 = fig.add_subplot(1, 3, 2)
    binary = (prob > 0.5).astype(np.float32)
    scatter2 = ax2.scatter(xyz_np[:, 0], xyz_np[:, 1], c=binary, cmap='coolwarm',
                           s=3.0, alpha=0.8, edgecolors='none', vmin=0, vmax=1)
    ax2.set_title('BEV — Hard Assignment (thresh=0.5)')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_aspect('equal')
    plt.colorbar(scatter2, ax=ax2, label='Class (0=bg, 1=fg)')

    # ── 3D view ──
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    scatter3 = ax3.scatter(xyz_np[:, 0], xyz_np[:, 1], xyz_np[:, 2],
                           c=prob, cmap='coolwarm', s=1.0, alpha=0.6)
    ax3.set_title('3D View — Foreground Probability')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    plt.colorbar(scatter3, ax=ax3, label='P(foreground)')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcd', type=str,
                        default='data/new_sheef/pngs/TV_250000000001.pcd')
    parser.add_argument('--num_points', type=int, default=16384)
    parser.add_argument('--output', type=str, default='output/pointrcnn_seg_quick.png')
    args = parser.parse_args()

    # Update to absolute path
    if not os.path.isabs(args.pcd):
        args.pcd = os.path.join('/code/OpenPCDet', args.pcd)
    if not os.path.isabs(args.output):
        args.output = os.path.join('/code/OpenPCDet', args.output)

    print(f"Loading PCD: {args.pcd}")
    xyz, feat, raw_pts = load_pcd_as_tensor(args.pcd, num_points=args.num_points)
    print(f"Sampled {xyz.shape[1]} points | xyz range: [{xyz.min():.2f}, {xyz.max():.2f}]")

    # Build model
    print("Building PointRCNN (Backbone + Seg Head)...")
    model = PointRCNNSegHead(input_channels=1, num_classes=1).cuda()
    model.train()  # use batch stats for varied output (random weights)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Inference
    print("Running segmentation inference...")
    with torch.no_grad():
        seg_logits = model(xyz, feat)  # (1, 1, N)
        seg_probs = torch.sigmoid(seg_logits).squeeze().cpu().numpy()  # (N,)

    print(f"Segmentation stats: min={seg_probs.min():.4f}, max={seg_probs.max():.4f}, "
          f"mean={seg_probs.mean():.4f}, fg_ratio={(seg_probs > 0.5).mean():.3f}")

    # Visualize
    xyz_np = xyz.squeeze(0).cpu().numpy()
    # Denormalize back to original coordinates for visualization
    visualize_segmentation(xyz_np, seg_probs, save_path=args.output)

    # Also save as colored point cloud
    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(xyz_np)
    colors = np.zeros((xyz_np.shape[0], 3), dtype=np.float64)
    # Red for foreground, blue for background
    colors[:, 0] = seg_probs  # R channel = fg prob
    colors[:, 2] = 1.0 - seg_probs  # B channel = bg prob
    pcd_out.colors = o3d.utility.Vector3dVector(colors)
    pcd_path = args.output.replace('.png', '.pcd')
    o3d.io.write_point_cloud(pcd_path, pcd_out)
    print(f"Saved colored PCD: {pcd_path}")

    print("\nDone! Note: model has random weights (no trained checkpoint loaded).")
    print("Predictions are random but show the segmentation pipeline is working.")


if __name__ == '__main__':
    main()
