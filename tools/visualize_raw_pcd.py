#!/usr/bin/env python
"""
原始点云数据可视化 (无标注框 / Raw point cloud visualization without bounding boxes).

用法:
    # 默认第一张 + BEV + 3D 左右并排
    python tools/visualize_raw_pcd.py

    # 指定帧索引
    python tools/visualize_raw_pcd.py --index 42

    # 只看 BEV 或只看 3D
    python tools/visualize_raw_pcd.py --index 0 --view bev
    python tools/visualize_raw_pcd.py --index 0 --view 3d

    # 按强度上色
    python tools/visualize_raw_pcd.py --index 0 --color_by intensity

    # 保存图像
    python tools/visualize_raw_pcd.py --index 0 --save output/raw_pcd.png
"""

import argparse
import os
import sys

import matplotlib
# headless 环境自动切 Agg, 避免 plt.show() 阻塞.
# MPLBACKEND 环境变量优先; 否则检测 DISPLAY.
_backend = os.environ.get("MPLBACKEND", "")
if _backend:
    matplotlib.use(_backend)
elif "DISPLAY" not in os.environ:
    matplotlib.use("Agg")
else:
    # WSL2 场景: DISPLAY 可能被设置但无实际 X server, 对 show() 做保护
    pass
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# 路径 & 常量
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "warehouse", "points")

# 参考推理脚本的裁剪范围 (仓库场景)
CROP_X = (-10, 30)
CROP_Y = (-15, 25)
CROP_Z = (-4, 2)


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------
def load_point_cloud(index: int = 0) -> np.ndarray:
    """加载第 index 帧原始点云, shape (N, 5): [x, y, z, intensity, timestamp]."""
    fname = os.path.join(DATA_DIR, f"{index:06d}.npy")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"找不到文件: {fname}")
    pts = np.load(fname)
    print(f"[OK] 加载 {fname} → {pts.shape[0]} 个点")
    print(f"     X: [{pts[:, 0].min():.2f}, {pts[:, 0].max():.2f}]")
    print(f"     Y: [{pts[:, 1].min():.2f}, {pts[:, 1].max():.2f}]")
    print(f"     Z: [{pts[:, 2].min():.2f}, {pts[:, 2].max():.2f}]")
    print(f"     intensity: [{pts[:, 3].min():.2f}, {pts[:, 3].max():.2f}]")
    return pts


# ---------------------------------------------------------------------------
# 裁剪
# ---------------------------------------------------------------------------
def crop_points(pts: np.ndarray, x_range, y_range, z_range):
    """按范围裁剪点云, 返回 (cropped, mask)."""
    mask = (
        (pts[:, 0] >= x_range[0])
        & (pts[:, 0] <= x_range[1])
        & (pts[:, 1] >= y_range[0])
        & (pts[:, 1] <= y_range[1])
        & (pts[:, 2] >= z_range[0])
        & (pts[:, 2] <= z_range[1])
    )
    return pts[mask], mask


# ---------------------------------------------------------------------------
# 绘制
# ---------------------------------------------------------------------------
def draw_bev(ax, pts: np.ndarray, colors=None, title="BEV (Top-Down)", s=3.0):
    """
    BEV 俯视图: X 为横轴, Y 为纵轴.
    点云黑色, scatter `s` 控制点大小.
    如果提供 colors, 按该数组映射颜色 (如 intensity).
    """
    if colors is not None:
        ax.scatter(pts[:, 0], pts[:, 1], c=colors, s=s, alpha=1.0, cmap="gnuplot",
                   edgecolors="none", linewidth=0)
    else:
        ax.scatter(pts[:, 0], pts[:, 1], c="#000000", s=s, alpha=1.0,
                   edgecolors="none", linewidth=0)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--")


def draw_3d(ax, pts: np.ndarray, colors=None, title="3D View", s=1.0):
    """
    3D 视图: X-Y-Z.
    点云黑色, scatter `s` 控制点大小.
    """
    if colors is not None:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=s, alpha=1.0,
                   cmap="gnuplot", edgecolors="none", linewidth=0)
    else:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="#000000", s=s, alpha=1.0,
                   edgecolors="none", linewidth=0)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title, fontsize=12, fontweight="bold")

    # 让 Z 轴方向看起来自然 (不反)
    # 默认 matplotlib 3D Z 向上, 保持即可


def visualize(pts: np.ndarray, view: str = "both", color_by: str = "none",
              save_path: str = None):
    """
    主绘制函数.

    Parameters
    ----------
    pts : np.ndarray, (N, 5)
    view : "bev" | "3d" | "both"
    color_by : "none" | "intensity"
    save_path : str or None
    """
    # ---- 确定颜色 ----
    if color_by == "intensity":
        colors = pts[:, 3]
        cbar_label = "Intensity"
    else:
        colors = None
        cbar_label = None

    # ---- 布局 ----
    if view == "both":
        fig = plt.figure(figsize=(18, 7))
        # 背景白色 (论文风格常用白色, 若需黑色可在 scatter 中调)
        ax_bev = fig.add_subplot(1, 2, 1)
        ax_3d = fig.add_subplot(1, 2, 2, projection="3d")

        draw_bev(ax_bev, pts, colors=colors, title="BEV (Top-Down View)")
        if colors is not None:
            sc = ax_bev.collections[0]
            cbar = fig.colorbar(sc, ax=ax_bev, fraction=0.046, pad=0.04)
            cbar.set_label(cbar_label)

        draw_3d(ax_3d, pts, colors=colors, title="3D View")

        fig.suptitle(f"Raw Point Cloud  |  {pts.shape[0]:,} points",
                     fontsize=14, fontweight="bold", y=1.02)
    elif view == "bev":
        fig, ax = plt.subplots(figsize=(9, 7))
        draw_bev(ax, pts, colors=colors, title="BEV (Top-Down View)")
        if colors is not None:
            cbar = fig.colorbar(ax.collections[0], ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(cbar_label)
        fig.suptitle(f"Raw Point Cloud  |  {pts.shape[0]:,} points",
                     fontsize=14, fontweight="bold", y=1.02)
    else:  # 3d
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        draw_3d(ax, pts, colors=colors, title="3D View")
        fig.suptitle(f"Raw Point Cloud  |  {pts.shape[0]:,} points",
                     fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"[OK] 保存到 {save_path}")

    # headless 或无可用 display 时安全退出
    backend = matplotlib.get_backend()
    if backend.lower() == "agg":
        print("[info] headless 模式, 跳过 plt.show()")
        plt.close(fig)
    else:
        try:
            plt.show()
        except Exception:
            print("[warn] plt.show() 失败, 可能是无 GUI 环境. 请用 --save 导出图片.")
            plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="原始点云可视化 (无标注框)")
    parser.add_argument("--index", type=int, default=0,
                        help="帧索引 (默认 0, 即 000000.npy)")
    parser.add_argument("--view", choices=["bev", "3d", "both"], default="both",
                        help="视图 (默认 both)")
    parser.add_argument("--color_by", choices=["none", "intensity"], default="none",
                        help="按什么上色 (默认 none = 纯黑)")
    parser.add_argument("--save", type=str, default=None,
                        help="保存路径, 不指定则仅显示")
    parser.add_argument("--crop", action="store_true",
                        help="启用预设裁剪范围 (聚焦仓库区域)")
    args = parser.parse_args()

    pts = load_point_cloud(args.index)

    if args.crop:
        pts, mask = crop_points(pts, CROP_X, CROP_Y, CROP_Z)
        print(f"[crop] 裁剪后: {pts.shape[0]} 个点 (保留 {mask.sum()})")

    visualize(pts, view=args.view, color_by=args.color_by, save_path=args.save)


if __name__ == "__main__":
    main()
