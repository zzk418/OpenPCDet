#!/usr/bin/env python3
"""快速可视化 Point-SAM proposals — 一个 PCD 的所有候选 mask 叠加显示"""

import sys, os, json
import numpy as np
import argparse

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (128, 0, 255), (0, 128, 255), (255, 128, 128), (128, 255, 128),
    (128, 128, 255), (192, 192, 0), (192, 0, 192), (0, 192, 192),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True, help="proposals.npz 文件")
    parser.add_argument("--info", type=str, default=None, help="info.json (可选)")
    parser.add_argument("--output", type=str, default=None, help="输出图片路径")
    parser.add_argument("--top", type=int, default=8, help="显示前 N 个 proposals")
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    masks = data["masks"]           # (M, N)
    xyz = data["xyz"]               # (N, 3)
    rgb = data.get("rgb", None)     # (N, 3)

    M, N = masks.shape
    print(f"Proposals: {M}, Points: {N}")

    # 3D bbox 统计
    for i in range(min(M, args.top)):
        m = masks[i]
        pts = xyz[m]
        if len(pts) == 0:
            continue
        c = pts.mean(0)
        d = pts.max(0) - pts.min(0)
        print(f"  [{i}] {m.sum():>6} pts | center=({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}) | size=({d[0]:.1f}, {d[1]:.1f}, {d[2]:.1f})")

    # 生成 BEV 叠加图
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available, skip visualization")
        return

    fig, axes = plt.subplots(1, min(2, args.top // 4 + 1), figsize=(16, 8))
    if not hasattr(axes, '__len__'):
        axes = [axes]
    axes = list(axes)

    # --- 左图: BEV 多色叠加 ---
    ax = axes[0]
    # 背景点云 (灰色)
    ax.scatter(xyz[:, 0], xyz[:, 2], c='lightgray', s=0.5, alpha=0.3)
    for i in range(min(M, args.top)):
        m = masks[i]
        if m.sum() == 0:
            continue
        pts = xyz[m]
        color = np.array(COLORS[i % len(COLORS)]) / 255.0
        ax.scatter(pts[:, 0], pts[:, 2], c=[color], s=2, alpha=0.7,
                   label=f'P{i} ({m.sum()}pts)')
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Z (mm)')
    ax.set_title('BEV: All Proposals')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_aspect('equal')

    # --- 右图: 3D 散点 ---
    if len(axes) > 1:
        ax = axes[1]
        from mpl_toolkits.mplot3d import Axes3D
        ax.remove()
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='lightgray', s=0.3, alpha=0.3)
        for i in range(min(M, args.top)):
            m = masks[i]
            if m.sum() == 0:
                continue
            pts = xyz[m]
            color = np.array(COLORS[i % len(COLORS)]) / 255.0
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=[color], s=3, alpha=0.7)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title('3D View')

    plt.tight_layout()
    out_path = args.output or args.npz.replace('.npz', '_viz.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
