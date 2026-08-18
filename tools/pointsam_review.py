#!/usr/bin/env python3
"""
Point-SAM 标注审核工具 — 为自动生成的 proposals 分配类别
==========================================================
审核工作流:
  1. 加载 _proposals.npz (Point-SAM 自动生成的候选 masks)
  2. 每个 proposal 显示 BEV 鸟瞰图 + 3D 统计信息
  3. 用户分配类别: beam / pillar / pallet / goods / discard

用法:
  # 交互审核单个文件
  python pointsam_review.py --annotation_dir output/shelf_annotations --stem TV_250000001828

  # 批量审核 (遍历所有 proposals)
  python pointsam_review.py --annotation_dir output/shelf_annotations --batch
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# ── 类别定义 ──
CLASSES = {
    0: "background",
    1: "beam",       # 横梁
    2: "pillar",     # 立柱
    3: "pallet",     # 卡板
    4: "goods",      # 货物
}

CLASS_COLORS = {
    0: (128, 128, 128),   # gray
    1: (255, 0, 0),       # red: beam
    2: (0, 0, 255),       # blue: pillar
    3: (0, 255, 0),       # green: pallet
    4: (255, 255, 0),     # yellow: goods
}


def load_proposals(npz_path: str) -> dict:
    """加载 proposals 文件。"""
    data = np.load(npz_path, allow_pickle=True)
    return {
        "masks": data["masks"],           # (M, N)
        "iou_scores": data["iou_scores"], # (M,)
        "n_points": data["n_points"],     # (M,)
        "xyz": data["xyz"],               # (N, 3)
        "rgb": data["rgb"],               # (N, 3)
    }


def render_bev_overview(xyz: np.ndarray, masks: np.ndarray, labels: dict,
                        output_path: str = None) -> np.ndarray:
    """生成所有 proposals 的 BEV 总览图。

    将整个点云和所有已标注的 mask 投影到 XZ 平面，不同类别用不同颜色。
    可用于快速预览标注效果。
    """
    # 确定 BEV 范围
    x = xyz[:, 0]
    z = xyz[:, 2]
    x_min, x_max = x.min() - 0.2, x.max() + 0.2
    z_min, z_max = z.min() - 0.2, z.max() + 0.2

    resolution = 0.01  # 1cm per pixel
    W = int((x_max - x_min) / resolution) + 1
    H = int((z_max - z_min) / resolution) + 1
    W = min(W, 4000)
    H = min(H, 4000)

    # 背景点云 (灰色)
    bev = np.zeros((H, W, 3), dtype=np.uint8)
    xi = np.clip(((x - x_min) / resolution).astype(int), 0, W - 1)
    zi = np.clip(((z - z_min) / resolution).astype(int), 0, H - 1)
    bev[zi, xi] = (80, 80, 80)

    # 叠加已分类的 masks
    for mask_id, class_id in labels.items():
        if class_id == 0:
            continue  # skip discarded
        mask = masks[mask_id]
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        mx = x[mask]
        mz = z[mask]
        mxi = np.clip(((mx - x_min) / resolution).astype(int), 0, W - 1)
        mzi = np.clip(((mz - z_min) / resolution).astype(int), 0, H - 1)
        bev[mzi, mxi] = color

    if output_path:
        from PIL import Image
        Image.fromarray(bev).save(output_path)
        print(f"  Saved BEV overview: {output_path}")

    return bev


def print_proposal_info(pid: int, proposal: dict, xyz: np.ndarray, mask: np.ndarray):
    """打印单个 proposal 的统计信息。"""
    pts = xyz[mask]
    x_min, y_min, z_min = pts.min(axis=0)
    x_max, y_max, z_max = pts.max(axis=0)
    dx, dy, dz = x_max - x_min, y_max - y_min, z_max - z_min

    print(f"\n  Proposal #{pid}:")
    print(f"    Points: {mask.sum()}")
    print(f"    IoU score: {proposal.get('iou_score', 0):.3f}")
    print(f"    BBox (m): dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}")
    print(f"    Center (m): x={pts[:,0].mean():.2f}, y={pts[:,1].mean():.2f}, z={pts[:,2].mean():.2f}")
    print(f"    Range: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}], z=[{z_min:.2f}, {z_max:.2f}]")


def interactive_review(npz_path: str, info_path: str):
    """交互式审核单个文件的 proposals。"""
    data = load_proposals(npz_path)
    masks = data["masks"]
    xyz = data["xyz"]

    with open(info_path) as f:
        info = json.load(f)

    M = masks.shape[0]
    print(f"\n{'='*60}")
    print(f"Review: {Path(npz_path).stem}")
    print(f"  Points: {len(xyz)}")
    print(f"  Proposals: {M}")
    print(f"\nClasses: 1=beam(横梁) 2=pillar(立柱) 3=pallet(卡板) 4=goods(货物) 0=discard")
    print(f"Commands: <id>=<class> | 'show <id>' | 'list' | 'overview' | 'done'")

    labels = {}
    for i, p_info in enumerate(info["proposals"]):
        if p_info.get("class_label") is not None:
            class_name = p_info["class_label"]
            class_id = {v: k for k, v in CLASSES.items()}.get(class_name, 0)
            labels[i] = class_id
            print(f"  [{i}] pre-labeled: {CLASSES[class_id]}")

    while True:
        try:
            cmd = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd:
            continue
        if cmd == "done":
            break
        if cmd == "list":
            for i in range(M):
                pts = xyz[masks[i]]
                if pts.sum() == 0:
                    continue
                label_str = CLASSES.get(labels.get(i, -1), "?")
                cx, cy, cz = pts.mean(axis=0)
                print(f"  [{i}] {mask.sum():>6} pts | label={label_str} | center=({cx:.2f}, {cy:.2f}, {cz:.2f})")
            continue
        if cmd == "overview":
            save_path = npz_path.replace("_proposals.npz", "_overview.png")
            render_bev_overview(xyz, masks, labels, save_path)
            continue
        if cmd.startswith("show "):
            try:
                pid = int(cmd.split()[1])
                if 0 <= pid < M:
                    print_proposal_info(pid, info["proposals"][pid], xyz, masks[pid])
                else:
                    print(f"  Invalid ID: {pid}")
            except ValueError:
                print(f"  Usage: show <id>")
            continue

        # Parse: <id>=<class>
        if "=" in cmd:
            try:
                pid_str, cls_str = cmd.split("=")
                pid = int(pid_str.strip())
                cls_id = int(cls_str.strip())
                if 0 <= pid < M and cls_id in CLASSES:
                    labels[pid] = cls_id
                    # 更新 info
                    info["proposals"][pid]["class_label"] = CLASSES.get(cls_id, "background")
                    print(f"  [{pid}] → {CLASSES[cls_id]}")
                    # 保存
                    with open(info_path, "w") as f:
                        json.dump(info, f, indent=2, ensure_ascii=False)
                else:
                    print(f"  Invalid: id={pid}, class={cls_id}")
            except ValueError:
                print(f"  Usage: <proposal_id>=<class_id>  e.g. 0=1")

    # 最终保存
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # 统计
    class_counts = {}
    for pid, cid in labels.items():
        name = CLASSES[cid]
        class_counts[name] = class_counts.get(name, 0) + 1
    print(f"\nReview done. Labels: {class_counts}")

    # 生成 BEV 总览
    overview_path = npz_path.replace("_proposals.npz", "_overview.png")
    render_bev_overview(xyz, masks, labels, overview_path)


def batch_review(annotation_dir: str):
    """批量审核模式：打印每个文件的 proposals 摘要。"""
    npz_files = sorted(Path(annotation_dir).glob("*_proposals.npz"))
    print(f"Found {len(npz_files)} annotation files\n")

    total_proposals = 0
    labeled = 0
    unlabeled = 0

    for npz_path in npz_files:
        stem = npz_path.stem.replace("_proposals", "")
        info_path = npz_path.parent / f"{stem}_info.json"

        data = np.load(npz_path)
        M = data["masks"].shape[0]
        total_proposals += M

        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            n_labeled = sum(1 for p in info["proposals"] if p.get("class_label"))
            n_unlabeled = M - n_labeled
            labeled += n_labeled
            unlabeled += n_unlabeled
            status = f"{n_labeled}/{M} labeled"
        else:
            status = "no info"

        print(f"  {stem:40s}  {M:3d} proposals  {status}")

    print(f"\nTotal: {total_proposals} proposals, {labeled} labeled, {unlabeled} unlabeled")


# ============================================================================
# 从审核结果生成语义分割标签
# ============================================================================

def generate_semantic_labels(annotation_dir: str, output_path: str = None):
    """将所有已审核的 proposals 合并为语义分割标签。

    生成格式: 每个 PCD 对应一个 (N,) 的类别标签数组。
    如果一个点属于多个 mask，取点数最小的 mask 的标签 (精细化优先)。

    Args:
        annotation_dir: 标注文件目录
        output_path: 汇总输出路径 (默认 annotation_dir/semantic_labels/)
    """
    if output_path is None:
        output_path = os.path.join(annotation_dir, "semantic_labels")
    os.makedirs(output_path, exist_ok=True)

    npz_files = sorted(Path(annotation_dir).glob("*_proposals.npz"))
    summary = []

    for npz_path in npz_files:
        stem = npz_path.stem.replace("_proposals", "")
        info_path = npz_path.parent / f"{stem}_info.json"

        if not info_path.exists():
            continue

        with open(info_path) as f:
            info = json.load(f)

        data = np.load(npz_path)
        masks = data["masks"]    # (M, N)
        xyz = data["xyz"]        # (N, 3)
        N = len(xyz)

        # 初始化所有点为 background (0)
        semantic = np.zeros(N, dtype=np.uint8)

        # 按 mask 点数升序排列 (小 mask 优先，更精细)
        mask_orders = sorted(
            [(i, p) for i, p in enumerate(info["proposals"]) if p.get("class_label")],
            key=lambda x: x[1].get("n_points", N),
        )

        class_name_to_id = {v: k for k, v in CLASSES.items()}

        for mask_id, p_info in mask_orders:
            class_name = p_info["class_label"]
            class_id = class_name_to_id.get(class_name, 0)
            if class_id == 0:
                continue
            semantic[masks[mask_id]] = class_id

        # 保存
        out_file = os.path.join(output_path, f"{stem}_labels.npy")
        np.save(out_file, semantic)
        summary.append({
            "stem": stem,
            "num_points": N,
            "class_distribution": {
                CLASSES[c]: int((semantic == c).sum())
                for c in range(len(CLASSES)) if (semantic == c).sum() > 0
            },
        })

        print(f"  {stem}: saved {out_file}")

    # 保存汇总
    summary_path = os.path.join(output_path, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary: {summary_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Point-SAM 标注审核与分类")
    parser.add_argument("--annotation_dir", type=str,
                        default="output/shelf_annotations",
                        help="标注文件目录 (含 *_proposals.npz)")
    parser.add_argument("--stem", type=str, default=None,
                        help="审核单个文件 (不含 _proposals 后缀)")
    parser.add_argument("--batch", action="store_true",
                        help="批量审核模式 (列出所有文件状态)")
    parser.add_argument("--generate_labels", action="store_true",
                        help="从审核结果生成语义分割标签")
    args = parser.parse_args()

    if args.generate_labels:
        generate_semantic_labels(args.annotation_dir)
        return

    if args.stem:
        npz_path = os.path.join(args.annotation_dir, f"{args.stem}_proposals.npz")
        info_path = os.path.join(args.annotation_dir, f"{args.stem}_info.json")
        if not os.path.exists(npz_path):
            print(f"Error: {npz_path} not found")
            sys.exit(1)
        interactive_review(npz_path, info_path)
    elif args.batch:
        batch_review(args.annotation_dir)
    else:
        # 默认交互式审核第一个未完成的
        npz_files = sorted(Path(args.annotation_dir).glob("*_proposals.npz"))
        if not npz_files:
            print(f"No proposals found in {args.annotation_dir}")
            print("Run pointsam_annotator.py first to generate proposals.")
            sys.exit(1)

        # 找第一个未完全标注的
        for npz_path in npz_files:
            stem = npz_path.stem.replace("_proposals", "")
            info_path = npz_path.parent / f"{stem}_info.json"
            if info_path.exists():
                with open(info_path) as f:
                    info = json.load(f)
                n_labeled = sum(1 for p in info["proposals"] if p.get("class_label"))
                n_total = len(info["proposals"])
                if n_labeled < n_total:
                    print(f"Picking first uncompleted: {stem} ({n_labeled}/{n_total})")
                    interactive_review(str(npz_path), str(info_path))
                    return

        # 全部完成，从第一个开始
        first = npz_files[0]
        stem = first.stem.replace("_proposals", "")
        info_path = first.parent / f"{stem}_info.json"
        interactive_review(str(first), str(info_path))


if __name__ == "__main__":
    main()
