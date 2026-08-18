#!/usr/bin/env python
"""
多帧点云拼接 — 将 N 帧点云拼成正方形/矩形大仓库场景.

用法:
    # 9 帧拼成 3×3 紧密大仓库 (推荐)
    python tools/stitch_frames.py --crop --tight

    # 6 帧拼成 2×3
    python tools/stitch_frames.py --num_frames 6 --crop --tight

    # 零间距 + 紧密
    python tools/stitch_frames.py --crop --tight --gap 0

    # 帧间有 20% 重叠 (更像连续场景)
    python tools/stitch_frames.py --crop --tight --overlap 0.2

    # 统一 cell 模式 (整齐但可能有空白)
    python tools/stitch_frames.py --crop --cell_mode median --gap 2

    # 可视化
    python tools/stitch_frames.py --crop --tight --visualize --save_viz output/stitched.png

输出:
    data/warehouse_stitched/points/000000.npy
    data/warehouse_stitched/labels/000000.txt
"""

import argparse
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "warehouse")
POINTS_DIR = os.path.join(DATA_DIR, "points")
LABELS_DIR = os.path.join(DATA_DIR, "labels")
IMAGESETS_DIR = os.path.join(DATA_DIR, "ImageSets")

OUT_DIR = os.path.join(ROOT, "data", "warehouse_stitched")

CLASS_NAMES = ['箱子', '电动运输车', '货运自行车', '无人搬运车', '叉车']

# 预设裁剪范围 (仓库场景)
CROP_X = (-10, 30)
CROP_Y = (-15, 25)
CROP_Z = (-4, 2)


# ============================================================================
# 工具函数
# ============================================================================

def load_frame(index: int):
    """加载单帧点云和标签."""
    pts_path = os.path.join(POINTS_DIR, f"{index:06d}.npy")
    lbl_path = os.path.join(LABELS_DIR, f"{index:06d}.txt")

    if not os.path.exists(pts_path):
        raise FileNotFoundError(f"点云文件不存在: {pts_path}")

    pts = np.load(pts_path)  # (N, 5): x, y, z, intensity, timestamp

    labels = []
    if os.path.exists(lbl_path):
        with open(lbl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 7:
                    labels.append({
                        'x': float(parts[0]), 'y': float(parts[1]),
                        'z': float(parts[2]),
                        'dx': float(parts[3]), 'dy': float(parts[4]),
                        'dz': float(parts[5]),
                        'heading': float(parts[6]),
                        'class': parts[7] if len(parts) > 7 else 'unknown',
                    })
    return pts, labels


def compute_frame_bbox(pts):
    """单帧点云 XY 边界 (1%-99% 分位, 抑制离群点)."""
    x_low, x_high = np.percentile(pts[:, 0], [1, 99])
    y_low, y_high = np.percentile(pts[:, 1], [1, 99])
    return x_low, x_high, y_low, y_high


def determine_grid(num_frames, grid=None):
    """根据帧数确定行列布局 (尽量正方形)."""
    if grid is not None:
        return grid[0], grid[1]
    cols = int(np.ceil(np.sqrt(num_frames)))
    rows = int(np.ceil(num_frames / cols))
    return rows, cols


def get_frame_indices(num_frames, split="train", start_index=None, seed=42):
    """获取要拼接的帧索引列表."""
    split_file = os.path.join(IMAGESETS_DIR, f"{split}.txt")
    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            all_indices = [int(line.strip()) for line in f if line.strip()]
    else:
        all_indices = sorted([
            int(f.replace(".npy", ""))
            for f in os.listdir(POINTS_DIR) if f.endswith(".npy")
        ])

    if start_index is not None:
        idx = all_indices.index(start_index) if start_index in all_indices else 0
        selected = all_indices[idx:idx + num_frames]
    else:
        rng = np.random.RandomState(seed)
        rng.shuffle(all_indices)
        selected = sorted(all_indices[:num_frames])

    if len(selected) < num_frames:
        print(f"[warn] 请求 {num_frames} 帧, 但只有 {len(selected)} 帧可用")
    return selected


def crop_frame(pts, labels, x_range, y_range, z_range):
    """裁剪单帧点云和标签到指定范围."""
    mask = (
        (pts[:, 0] >= x_range[0]) & (pts[:, 0] <= x_range[1]) &
        (pts[:, 1] >= y_range[0]) & (pts[:, 1] <= y_range[1]) &
        (pts[:, 2] >= z_range[0]) & (pts[:, 2] <= z_range[1])
    )
    pts_cropped = pts[mask]
    labels_cropped = []
    for lbl in labels:
        if (x_range[0] <= lbl['x'] <= x_range[1] and
            y_range[0] <= lbl['y'] <= y_range[1] and
            z_range[0] <= lbl['z'] <= z_range[1]):
            labels_cropped.append(lbl)
    return pts_cropped, labels_cropped


# ============================================================================
# 拼接核心
# ============================================================================

def stitch_frames(frame_indices, rows, cols, gap=3.0,
                  crop=False, crop_x=CROP_X, crop_y=CROP_Y, crop_z=CROP_Z,
                  cell_mode="max", cell_size=None, overlap=0.0,
                  tight=False, jitter=0.0, rotate=0.0):
    """
    拼接多帧.

    Parameters
    ----------
    tight : bool
        True = 紧密模式: 每帧用自己的 bbox, 行列累积偏移.
    jitter : float
        每帧随机位移幅度 (米). 打破网格边界, 消除"墙".
    rotate : float
        每帧随机旋转幅度 (度). 让扫描线不平行, 隐藏接缝.
    """
    # ---- Step 1: 加载 & 裁剪 ----
    frames = {}
    for idx in frame_indices:
        pts, labels = load_frame(idx)
        n_orig = pts.shape[0]

        if crop:
            pts, labels = crop_frame(pts, labels, crop_x, crop_y, crop_z)

        # 始终用实际点的紧密 bbox (裁剪后或原始), 不用 crop 窗口当 bbox
        bbox = compute_frame_bbox(pts)
        span_x, span_y = bbox[1] - bbox[0], bbox[3] - bbox[2]

        tag = f"(crop {n_orig}→{pts.shape[0]})" if crop else ""
        print(f"  [{idx:06d}] {pts.shape[0]:6d} 点 {tag}, {len(labels):3d} 标签  "
              f"X:[{bbox[0]:.1f}, {bbox[1]:.1f}] span={span_x:.1f}  "
              f"Y:[{bbox[2]:.1f}, {bbox[3]:.1f}] span={span_y:.1f}")

        frames[idx] = {'pts': pts, 'labels': labels, 'bbox': bbox,
                       'span_x': span_x, 'span_y': span_y}

    frame_list = list(frame_indices)

    # ---- Step 2: 计算每个 frame 的偏移 ----
    if cell_size is not None:
        # 手动指定统一 cell
        cell_w, cell_h = cell_size
        mode_str = "manual"
        offsets = {}
        for i, idx in enumerate(frame_list):
            r, c = i // cols, i % cols
            ox = c * cell_w * (1 - overlap)
            oy = r * cell_h * (1 - overlap)
            offsets[idx] = (ox, oy)

    elif tight:
        # ============ 紧密模式: 行列累积偏移, 无墙 ============
        # 计算每列最大宽度 和 每行最大高度
        col_widths = [0.0] * cols
        row_heights = [0.0] * rows
        for i, idx in enumerate(frame_list):
            r, c = i // cols, i % cols
            f = frames[idx]
            col_widths[c] = max(col_widths[c], f['span_x'] + gap)
            row_heights[r] = max(row_heights[r], f['span_y'] + gap)

        # 累积偏移: 列/行起始位置
        col_offsets = [0.0]
        for cw in col_widths[:-1]:
            col_offsets.append(col_offsets[-1] + cw * (1 - overlap))
        row_offsets = [0.0]
        for rh in row_heights[:-1]:
            row_offsets.append(row_offsets[-1] + rh * (1 - overlap))

        offsets = {}
        for i, idx in enumerate(frame_list):
            r, c = i // cols, i % cols
            offsets[idx] = (col_offsets[c], row_offsets[r])

        mode_str = "tight"
        # 计算总场景尺寸
        total_w = col_offsets[-1] + max(
            frames[frame_list[i]]['span_x']
            for i in range(len(frame_list)) if i % cols == cols - 1 or i == len(frame_list) - 1
        )
        total_h = row_offsets[-1] + max(
            frames[frame_list[i]]['span_y']
            for i in range(len(frame_list)) if i // cols == rows - 1 or i == len(frame_list) - 1
        )
        cell_w = total_w / cols  # 平均, 仅用于可视化网格
        cell_h = total_h / rows
        print(f"\n  tight 模式: 列宽={[f'{w:.1f}' for w in col_widths]}  "
              f"列偏={[f'{o:.1f}' for o in col_offsets]}")
        print(f"             行高={[f'{h:.1f}' for h in row_heights]}  "
              f"行偏={[f'{o:.1f}' for o in row_offsets]}")
        print(f"  总场景: {total_w:.1f}m × {total_h:.1f}m  [gap={gap}m, overlap={overlap:.0%}]")

    else:
        # ============ 统一 cell 模式 (max / median) ============
        x_spans = [f['span_x'] for f in frames.values()]
        y_spans = [f['span_y'] for f in frames.values()]

        if cell_mode == "max":
            raw_w, raw_h = max(x_spans), max(y_spans)
        elif cell_mode == "median":
            raw_w, raw_h = float(np.median(x_spans)), float(np.median(y_spans))
        else:
            raise ValueError(f"未知 cell_mode: {cell_mode}")

        cell_w, cell_h = raw_w + gap, raw_h + gap
        mode_str = cell_mode

        offsets = {}
        for i, idx in enumerate(frame_list):
            r, c = i // cols, i % cols
            ox = c * cell_w * (1 - overlap)
            oy = r * cell_h * (1 - overlap)
            offsets[idx] = (ox, oy)

        print(f"\n  cell: {cell_w:.1f}m × {cell_h:.1f}m  "
              f"[mode={mode_str}, gap={gap}m, overlap={overlap:.0%}]")
        print(f"  网格: {rows}×{cols}, 总场景: {cell_w * cols:.1f}m × {cell_h * rows:.1f}m")

    # ---- Step 3: 逐帧偏移/变换并合并 ----
    all_pts = []
    all_frame_ids = []  # 追踪每个点来自哪帧
    all_labels = []
    frame_info = {}
    rng = np.random.RandomState(42)  # 固定种子, 可复现

    for i, idx in enumerate(frame_list):
        if i >= rows * cols:
            break

        r, c = i // cols, i % cols
        offset_x, offset_y = offsets[idx]
        f = frames[idx]
        origin_x, origin_y = f['bbox'][0], f['bbox'][2]

        # 随机 jitter: 打破网格边界
        jx = rng.uniform(-jitter, jitter) if jitter > 0 else 0.0
        jy = rng.uniform(-jitter, jitter) if jitter > 0 else 0.0

        # 随机旋转: 让扫描线不平行, 隐藏接缝
        rot_deg = rng.uniform(-rotate, rotate) if rotate > 0 else 0.0
        rot_rad = np.deg2rad(rot_deg)

        pts = f['pts'].copy()

        # Step A: 归一化到 bbox 原点
        pts[:, 0] -= origin_x
        pts[:, 1] -= origin_y

        # Step B: 绕 bbox 中心旋转 (绕 Z 轴)
        if abs(rot_rad) > 1e-6:
            cx = f['span_x'] / 2.0
            cy = f['span_y'] / 2.0
            cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
            pts[:, 0] -= cx
            pts[:, 1] -= cy
            x_new = pts[:, 0] * cos_r - pts[:, 1] * sin_r
            y_new = pts[:, 0] * sin_r + pts[:, 1] * cos_r
            pts[:, 0] = x_new + cx
            pts[:, 1] = y_new + cy

        # Step C: 偏移到网格位置 + jitter
        pts[:, 0] += offset_x + jx
        pts[:, 1] += offset_y + jy
        all_pts.append(pts)
        all_frame_ids.append(np.full(pts.shape[0], idx, dtype=np.int32))

        # 标签同样变换
        for lbl in f['labels']:
            new_lbl = lbl.copy()
            # 归一化
            lx = lbl['x'] - origin_x
            ly = lbl['y'] - origin_y
            # 旋转
            if abs(rot_rad) > 1e-6:
                cx = f['span_x'] / 2.0
                cy = f['span_y'] / 2.0
                lx -= cx; ly -= cy
                lx_new = lx * cos_r - ly * sin_r
                ly_new = lx * sin_r + ly * cos_r
                lx = lx_new + cx; ly = ly_new + cy
                new_lbl['heading'] = lbl['heading'] + rot_rad
            # 偏移
            new_lbl['x'] = lx + offset_x + jx
            new_lbl['y'] = ly + offset_y + jy
            new_lbl['frame_idx'] = idx
            new_lbl['grid_row'] = r
            new_lbl['grid_col'] = c
            all_labels.append(new_lbl)

        frame_info[idx] = {
            'grid_pos': (r, c),
            'n_points': pts.shape[0],
            'bbox_origin': (origin_x, origin_y),
            'cell_offset': (offset_x + jx, offset_y + jy),
            'bbox_span_x': f['span_x'],
            'bbox_span_y': f['span_y'],
            'cell_w': cell_w,
            'cell_h': cell_h,
            'jitter': (jx, jy),
            'rotation_deg': rot_deg,
        }
        extra = f" jitter=({jx:.1f},{jy:.1f}) rot={rot_deg:.1f}°" if (jitter > 0 or rotate > 0) else ""
        print(f"  [{idx:06d}] → grid({r},{c})  "
              f"offset=({offset_x:.1f},{offset_y:.1f})  "
              f"bbox=({f['span_x']:.1f}×{f['span_y']:.1f})  "
              f"{pts.shape[0]:,}pts{extra}")

    merged_pts = np.concatenate(all_pts, axis=0)
    merged_frame_ids = np.concatenate(all_frame_ids, axis=0)
    print(f"\n  合并: {merged_pts.shape[0]:,} 点, {len(all_labels)} 标签")
    return merged_pts, all_labels, frame_info, merged_frame_ids


# ============================================================================
# 保存 & 可视化
# ============================================================================

def save_merged(pts, labels, output_dir, index=0):
    """保存合并后的点云和标签."""
    pts_dir = os.path.join(output_dir, "points")
    lbl_dir = os.path.join(output_dir, "labels")
    os.makedirs(pts_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    pts_path = os.path.join(pts_dir, f"{index:06d}.npy")
    np.save(pts_path, pts.astype(np.float32))
    print(f"[save] 点云 → {pts_path}  ({pts.shape[0]:,} 点)")

    lbl_path = os.path.join(lbl_dir, f"{index:06d}.txt")
    with open(lbl_path, "w") as f:
        for lbl in labels:
            f.write(f"{lbl['x']:.6f} {lbl['y']:.6f} {lbl['z']:.6f} "
                    f"{lbl['dx']:.6f} {lbl['dy']:.6f} {lbl['dz']:.6f} "
                    f"{lbl['heading']:.6f} {lbl['class']}\n")
    print(f"[save] 标签 → {lbl_path}  ({len(labels)} 个框)")


def save_ply(pts, labels, frame_ids, output_path):
    """
    保存为 PLY 格式, 按帧着色.
    frame_ids: (N,) 数组, 每个点属于哪一帧.
    """
    import matplotlib.pyplot as plt
    cmap = plt.cm.tab10

    unique_frames = sorted(set(frame_ids))
    frame_to_color = {}
    for i, fid in enumerate(unique_frames):
        rgba = cmap(i % 10)
        frame_to_color[fid] = (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

    colors = np.array([frame_to_color[fid] for fid in frame_ids], dtype=np.uint8)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {pts.shape[0]}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property float intensity\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for i in range(pts.shape[0]):
            f.write(f'{pts[i,0]:.4f} {pts[i,1]:.4f} {pts[i,2]:.4f} '
                    f'{pts[i,3]:.4f} {colors[i,0]} {colors[i,1]} {colors[i,2]}\n')

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"[save] PLY → {output_path}  ({pts.shape[0]:,} pts, {len(unique_frames)} frames, {size_mb:.1f}MB)")


def save_ply_objects(pts, labels, output_path, radius=3.0, max_z=-0.5):
    """
    保存仅物体附近的点云 PLY (去除墙体).
    保留每个标注框周围 radius 米内 + Z < max_z 的点.
    按类别着色.
    """
    import matplotlib.pyplot as plt

    CLASS_COLORS = {
        '箱子': (255, 180, 0), '电动运输车': (255, 50, 50),
        '货运自行车': (50, 180, 255), '无人搬运车': (50, 255, 100),
        '叉车': (200, 100, 255),
    }
    DEFAULT_COLOR = (120, 120, 120)

    keep = np.zeros(pts.shape[0], dtype=bool)
    point_colors = np.full((pts.shape[0], 3), DEFAULT_COLOR[0], dtype=np.uint8)

    for lbl in labels:
        d = np.sqrt((pts[:, 0] - lbl['x']) ** 2 + (pts[:, 1] - lbl['y']) ** 2)
        mask = (d < radius) & (pts[:, 2] < max_z)
        color = CLASS_COLORS.get(lbl['class'], DEFAULT_COLOR)
        keep |= mask
        point_colors[mask] = color

    out_pts = pts[keep]
    out_colors = point_colors[keep]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {out_pts.shape[0]}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property float intensity\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for i in range(out_pts.shape[0]):
            f.write(f'{out_pts[i,0]:.4f} {out_pts[i,1]:.4f} {out_pts[i,2]:.4f} '
                    f'{out_pts[i,3]:.4f} {out_colors[i,0]} {out_colors[i,1]} {out_colors[i,2]}\n')

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    kept_pct = 100 * out_pts.shape[0] / pts.shape[0] if pts.shape[0] > 0 else 0
    print(f"[save] PLY-objects → {output_path}  "
          f"({out_pts.shape[0]:,} pts, {kept_pct:.0f}%, {size_mb:.1f}MB)  "
          f"radius={radius}m Z<{max_z}")


def visualize_merged(pts, labels, frame_info, rows, cols, cell_w, cell_h,
                     save_path=None, title="Stitched Warehouse"):
    """可视化: BEV 颜色分帧 + 网格线 + 标注框 + 3D."""
    import matplotlib
    backend = os.environ.get("MPLBACKEND", "")
    if backend:
        matplotlib.use(backend)
    elif "DISPLAY" not in os.environ:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.cm.tab10
    frame_color_map = {}
    for i, fidx in enumerate(sorted(frame_info.keys())):
        frame_color_map[fidx] = cmap(i % 10)

    fig = plt.figure(figsize=(22, 9))

    # ---- BEV ----
    ax_bev = fig.add_subplot(1, 2, 1)

    for fidx, info in sorted(frame_info.items()):
        r, c_pos = info['grid_pos']
        ox, oy = info['cell_offset']
        span_x = info['bbox_span_x']
        span_y = info['bbox_span_y']

        # 按空间范围筛选属于该帧的点
        margin = 2.0
        mask = (
            (pts[:, 0] >= ox - margin) & (pts[:, 0] <= ox + span_x + margin) &
            (pts[:, 1] >= oy - margin) & (pts[:, 1] <= oy + span_y + margin)
        )
        if mask.sum() > 0:
            color = frame_color_map[fidx]
            ax_bev.scatter(pts[mask, 0], pts[mask, 1],
                          c=[color], s=1.5, alpha=0.5,
                          edgecolors="none", linewidth=0,
                          label=f"{fidx:06d} ({r},{c_pos})")

    # 网格线 (累积偏移位置)
    col_edges = sorted(set(info['cell_offset'][0] for info in frame_info.values()))
    row_edges = sorted(set(info['cell_offset'][1] for info in frame_info.values()))
    for x in col_edges:
        ax_bev.axvline(x=x, color="#888888", linewidth=0.6, linestyle="--", alpha=0.4)
    for y in row_edges:
        ax_bev.axhline(y=y, color="#888888", linewidth=0.6, linestyle="--", alpha=0.4)

    # 标注框
    for lbl in labels:
        cx, cy = lbl['x'], lbl['y']
        dx, dy, hd = lbl['dx'], lbl['dy'], lbl['heading']
        corners_x, corners_y = [], []
        for sx, sy in [(-dx/2, -dy/2), (dx/2, -dy/2), (dx/2, dy/2), (-dx/2, dy/2)]:
            rx = sx * np.cos(hd) - sy * np.sin(hd) + cx
            ry = sx * np.sin(hd) + sy * np.cos(hd) + cy
            corners_x.append(rx); corners_y.append(ry)
        corners_x.append(corners_x[0]); corners_y.append(corners_y[0])
        ax_bev.plot(corners_x, corners_y, color="#CC3333", linewidth=1.0, alpha=0.6)

    ax_bev.set_xlabel("X (m)"); ax_bev.set_ylabel("Y (m)")
    ax_bev.set_title(f"BEV — {pts.shape[0]:,} pts, {len(labels)} boxes")
    ax_bev.set_aspect("equal")
    ax_bev.grid(True, alpha=0.15, linestyle="--")
    ax_bev.legend(loc="upper right", fontsize=5, ncol=3, markerscale=3)

    # ---- 3D ----
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
    n_disp = min(80000, pts.shape[0])
    if pts.shape[0] > n_disp:
        idx_s = np.random.RandomState(42).choice(pts.shape[0], n_disp, replace=False)
        pts_disp = pts[idx_s]
    else:
        pts_disp = pts
    ax_3d.scatter(pts_disp[:, 0], pts_disp[:, 1], pts_disp[:, 2],
                  c="#111111", s=0.3, alpha=0.5, edgecolors="none", linewidth=0)
    ax_3d.set_xlabel("X (m)"); ax_3d.set_ylabel("Y (m)"); ax_3d.set_zlabel("Z (m)")
    ax_3d.set_title(f"3D — {n_disp:,} / {pts.shape[0]:,} pts")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"[save] 可视化 → {save_path}")

    if matplotlib.get_backend().lower() != "agg":
        try:
            plt.show()
        except Exception:
            print("[warn] plt.show() 失败")
    plt.close(fig)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="多帧点云拼接 — 合成大仓库场景",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 紧密拼接 (推荐)
  python tools/stitch_frames.py --crop --tight

  # 6 帧 + 紧密 + 可视化
  python tools/stitch_frames.py --num_frames 6 --crop --tight --visualize --save_viz out.png

  # 带 20% 重叠 (更像连续场景)
  python tools/stitch_frames.py --crop --tight --overlap 0.2

  # 统一 cell 模式
  python tools/stitch_frames.py --crop --cell_mode median --gap 2
        """)
    parser.add_argument("--num_frames", type=int, default=9,
                        help="拼接帧数 (默认 9)")
    parser.add_argument("--grid", type=int, nargs=2, default=None,
                        help="网格行列, 如 --grid 3 3")
    parser.add_argument("--split", choices=["train", "val", "all"], default="train")
    parser.add_argument("--start_index", type=int, default=None,
                        help="起始帧索引 (顺序取, 不随机)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gap", type=float, default=2.0,
                        help="帧间距 米 (默认 2.0, tight 模式建议 0~2)")
    parser.add_argument("--overlap", type=float, default=0.0,
                        help="帧间重叠比例 0~1 (默认 0)")
    parser.add_argument("--tight", action="store_true",
                        help="紧密模式: 每帧用自身 bbox, 行列累积偏移, 消除点云墙")
    parser.add_argument("--jitter", type=float, default=0.0,
                        help="每帧随机位移幅度 米 (如 2.0). 打破网格边界, 消除接缝")
    parser.add_argument("--rotate", type=float, default=0.0,
                        help="每帧随机旋转幅度 度 (如 5.0). 让扫描线不平行, 隐藏墙体")
    parser.add_argument("--crop", action="store_true",
                        help="裁剪到统一仓库范围 [-10,30]×[-15,25]×[-4,2]")
    parser.add_argument("--crop_x", type=float, nargs=2, default=[-10, 30])
    parser.add_argument("--crop_y", type=float, nargs=2, default=[-15, 25])
    parser.add_argument("--crop_z", type=float, nargs=2, default=[-4, 2])
    parser.add_argument("--cell_mode", choices=["max", "median"], default="max",
                        help="统一 cell 模式 (仅非 tight 时生效)")
    parser.add_argument("--cell_size", type=float, nargs=2, default=None,
                        help="手动 cell 宽高, 如 --cell_size 40 35")
    parser.add_argument("--output_index", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="自定义输出目录")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--save_viz", type=str, default=None,
                        help="可视化保存路径")
    parser.add_argument("--save_ply", type=str, default=None,
                        help="PLY 输出路径 (按帧着色)")
    parser.add_argument("--save_ply_objects", type=str, default=None,
                        help="仅物体附近点云的 PLY 路径 (去墙体, 按类别着色)")
    parser.add_argument("--ply_radius", type=float, default=3.0,
                        help="物体附近保留半径 米 (默认 3.0)")
    parser.add_argument("--ply_max_z", type=float, default=-0.5,
                        help="保留点的最大 Z 高度 米 (默认 -0.5, 裁掉墙体)")
    args = parser.parse_args()

    # ---- 网格 ----
    rows, cols = determine_grid(args.num_frames, args.grid)
    actual_frames = min(args.num_frames, rows * cols)
    print(f"网格布局: {rows}×{cols}, 实际使用 {actual_frames} 帧")

    out_dir = args.output_dir or OUT_DIR

    # ---- 选取帧 ----
    indices = get_frame_indices(args.num_frames, args.split,
                                args.start_index, args.seed)
    print(f"选取帧: {[f'{i:06d}' for i in indices]}")

    # ---- 拼接 ----
    print("\n加载并拼接...")
    merged_pts, merged_labels, frame_info, merged_frame_ids = stitch_frames(
        indices, rows, cols,
        gap=args.gap,
        crop=args.crop,
        crop_x=tuple(args.crop_x), crop_y=tuple(args.crop_y),
        crop_z=tuple(args.crop_z),
        cell_mode=args.cell_mode,
        cell_size=tuple(args.cell_size) if args.cell_size else None,
        overlap=args.overlap,
        tight=args.tight,
        jitter=args.jitter,
        rotate=args.rotate,
    )

    sample_info = next(iter(frame_info.values()))
    cell_w = sample_info['cell_w']
    cell_h = sample_info['cell_h']

    # ---- 统计 ----
    print(f"\n{'='*50}")
    print(f"拼接结果统计:")
    print(f"  总点数:     {merged_pts.shape[0]:,}")
    print(f"  总标签数:   {len(merged_labels)}")
    print(f"  X: [{merged_pts[:,0].min():.1f}, {merged_pts[:,0].max():.1f}]")
    print(f"  Y: [{merged_pts[:,1].min():.1f}, {merged_pts[:,1].max():.1f}]")
    print(f"  Z: [{merged_pts[:,2].min():.2f}, {merged_pts[:,2].max():.2f}]")
    class_counts = {}
    for lbl in merged_labels:
        cls = lbl['class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    print(f"  各类标签:   {class_counts}")

    # ---- 保存 ----
    save_merged(merged_pts, merged_labels, out_dir, index=args.output_index)

    # ---- PLY ----
    if args.save_ply:
        save_ply(merged_pts, merged_labels, merged_frame_ids, args.save_ply)

    if args.save_ply_objects:
        save_ply_objects(merged_pts, merged_labels, args.save_ply_objects,
                         radius=args.ply_radius, max_z=args.ply_max_z)

    # ---- 可视化 ----
    if args.visualize or args.save_viz:
        print("\n生成可视化...")
        visualize_merged(
            merged_pts, merged_labels, frame_info,
            rows, cols, cell_w, cell_h,
            save_path=args.save_viz,
            title=f"Stitched Warehouse — {actual_frames}f ({rows}×{cols}), "
                  f"{merged_pts.shape[0]:,} pts, {len(merged_labels)} boxes"
        )


if __name__ == "__main__":
    main()
