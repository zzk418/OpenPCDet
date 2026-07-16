#!/usr/bin/env python3
"""
仓库场景 CenterPoint 推理 + 多视图可视化（通用版）
- BEV 俯视图 + 3D 视图，自动紧凑裁剪
- 点云纯黑色、高对比度
- 3D 框中心点标记 (● BEV / ★ 3D)，输出坐标 JSON
- 支持 CenterPoint / CenterPoint++ 两种模型
"""
import argparse, os, sys, json
from pathlib import Path

import numpy as np
import torch

_tools_dir = str(Path(__file__).resolve().parent / 'tools')
if os.path.exists(_tools_dir):
    os.chdir(_tools_dir)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.custom.custom_dataset import CustomDataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

# ============================================================
# 英文标签 / 颜色
# ============================================================
LABEL_MAP = {
    '箱子': 'Box',
    '电动运输车': 'ELF',
    '货运自行车': 'CargoBike',
    '无人搬运车': 'FTS',
    '叉车': 'ForkLift',
}
GT_COLOR = '#444444'  # 深灰，不和任何类别颜色混淆

CLASS_COLORS = {
    'Box': '#2196F3',
    'ELF': '#FF9800',
    'CargoBike': '#4CAF50',
    'FTS': '#9C27B0',
    'ForkLift': '#F44336',
}


def translate_label(name):
    for cn, en in LABEL_MAP.items():
        if cn in str(name):
            return en
    return str(name)


# ============================================================
# 几何工具
# ============================================================
def boxes_to_corners(boxes):
    """[x,y,z,dx,dy,dz,heading] -> (N,8,3)"""
    if len(boxes) == 0:
        return np.zeros((0, 8, 3))
    boxes = np.array(boxes)
    cl = []
    for box in boxes:
        x, y, z, dx, dy, dz, heading = box[:7]
        ch, sh = np.cos(heading), np.sin(heading)
        rot = np.array([[ch, -sh], [sh, ch]])
        corners_local = np.array([
            [-dx/2, -dy/2], [dx/2, -dy/2], [dx/2, dy/2], [-dx/2, dy/2]
        ])
        cxy = corners_local @ rot.T
        cxy[:, 0] += x
        cxy[:, 1] += y
        bottom = np.hstack([cxy, np.ones((4, 1)) * (z - dz/2)])
        top = np.hstack([cxy, np.ones((4, 1)) * (z + dz/2)])
        cl.append(np.vstack([bottom, top]))
    return np.array(cl)


def compact_crop_range(boxes_list, padding=4.0):
    """根据所有框的角点计算紧凑显示范围"""
    all_corners = []
    for bx in boxes_list:
        if len(bx) > 0:
            corners = boxes_to_corners(bx)  # (N,8,3)
            for c in corners:
                all_corners.append(c)
    if not all_corners:
        return None
    all_c = np.concatenate(all_corners, axis=0)
    x_min, x_max = all_c[:, 0].min(), all_c[:, 0].max()
    y_min, y_max = all_c[:, 1].min(), all_c[:, 1].max()
    dx = x_max - x_min
    dy = y_max - y_min
    max_dim = max(dx, dy) + padding * 2
    x_c = (x_min + x_max) / 2
    y_c = (y_min + y_max) / 2
    return [x_c - max_dim/2, x_c + max_dim/2, y_c - max_dim/2, y_c + max_dim/2]


def draw_3d_box(ax, corners, color, alpha, lw, label=None):
    b = corners
    edges = [
        (0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]
    for i, j in edges:
        ax.plot([b[i,0],b[j,0]], [b[i,1],b[j,1]], [b[i,2],b[j,2]],
                color=color, alpha=alpha, linewidth=lw)
    if label:
        ax.text(b[4,0], b[4,1], b[4,2], label, fontsize=8, color=color,
                weight='bold', ha='center', va='bottom')


# ============================================================
# 可视化主函数
# ============================================================
def create_visualization(points, pred_boxes, pred_scores, pred_labels,
                         gt_boxes, gt_names, class_names, output_file,
                         pc_range, sample_id="", show_labels=False):
    fig = plt.figure(figsize=(24, 11), dpi=150)

    en_gt_names = [translate_label(n) for n in gt_names] if gt_names is not None else []
    en_pred_labels = []
    for l in pred_labels:
        idx = int(l) - 1
        if 0 <= idx < len(class_names):
            en_pred_labels.append(translate_label(class_names[idx]))
        else:
            en_pred_labels.append(f'cls_{l}')

    # 紧凑裁剪
    all_boxes_for_crop = []
    if gt_boxes is not None and len(gt_boxes) > 0:
        all_boxes_for_crop.append(gt_boxes)
    if len(pred_boxes) > 0:
        keep = pred_scores >= 0.1
        all_boxes_for_crop.append(pred_boxes[keep])
    crop = compact_crop_range(all_boxes_for_crop, padding=4.0)

    # ===== 左图：BEV =====
    ax1 = fig.add_subplot(1, 2, 1)
    # 纯黑点云，大点，不透明
    ax1.scatter(points[:, 0], points[:, 1], c='black', s=3.0, alpha=1.0,
                rasterized=True, edgecolors='none', linewidths=0)

    # GT (深灰虚线)
    if gt_boxes is not None and len(gt_boxes) > 0:
        for box, name in zip(gt_boxes, en_gt_names):
            corners = boxes_to_corners([box])[0]
            bx, by = corners[[0,1,2,3,0], 0], corners[[0,1,2,3,0], 1]
            ax1.plot(bx, by, '--', color=GT_COLOR, linewidth=2.5, alpha=0.85)
            ax1.plot(box[0], box[1], 'x', color=GT_COLOR, markersize=10,
                     markeredgewidth=2)

    # Pred (实线) + 中心点
    if len(pred_boxes) > 0:
        for i, box in enumerate(pred_boxes):
            if pred_scores[i] < 0.1:
                continue
            name = en_pred_labels[i] if i < len(en_pred_labels) else '?'
            color = CLASS_COLORS.get(name, 'red')
            corners = boxes_to_corners([box])[0]
            bx, by = corners[[0,1,2,3,0], 0], corners[[0,1,2,3,0], 1]
            ax1.plot(bx, by, '-', color=color, linewidth=2.8, alpha=0.95)
            ax1.fill(bx, by, alpha=0.15, color=color)
            # 中心点大圆
            ax1.plot(box[0], box[1], 'o', color=color, markersize=14,
                     markeredgecolor='black', markeredgewidth=2, zorder=10)
            if show_labels:
                ax1.text(box[0], box[1] + 1.0,
                         f'{name}\n({box[0]:.1f},{box[1]:.1f})\n{pred_scores[i]:.2f}',
                         fontsize=8, ha='center', va='bottom', color='white',
                         weight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.9))

    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title(f'BEV | {sample_id} | Dashed=GT  Solid=Pred  ●=Center',
                  fontsize=13, weight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axis('equal')
    ax1.set_facecolor('#FAFAFA')

    if crop:
        ax1.set_xlim(crop[0], crop[1])
        ax1.set_ylim(crop[2], crop[3])
    else:
        ax1.set_xlim(pc_range[0], pc_range[3])
        ax1.set_ylim(pc_range[1], pc_range[4])

    # ===== 右图：3D =====
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_facecolor('#FAFAFA')

    n_pts = min(len(points), 8000)
    idx_r = np.random.choice(len(points), n_pts, replace=False)
    ax2.scatter(points[idx_r, 0], points[idx_r, 1], points[idx_r, 2],
                c='black', s=1.0, alpha=1.0, edgecolors='none', linewidths=0, rasterized=True)

    # GT 3D (深灰)
    if gt_boxes is not None and len(gt_boxes) > 0:
        gt_corners = boxes_to_corners(gt_boxes)
        for i, corners in enumerate(gt_corners):
            name = en_gt_names[i] if i < len(en_gt_names) else '?'
            draw_3d_box(ax2, corners, GT_COLOR, alpha=0.65, lw=1.5,
                        label=f'GT:{name}' if show_labels else None)

    # Pred 3D + 中心星号
    if len(pred_boxes) > 0:
        keep_idx = pred_scores >= 0.1
        pred_corners_all = boxes_to_corners(pred_boxes[keep_idx])
        pred_scores_f = pred_scores[keep_idx]
        en_labels_f = [en_pred_labels[i] for i in range(len(en_pred_labels)) if keep_idx[i]]
        for i, corners in enumerate(pred_corners_all):
            name = en_labels_f[i] if i < len(en_labels_f) else '?'
            color = CLASS_COLORS.get(name, 'red')
            draw_3d_box(ax2, corners, color, alpha=0.9, lw=2.0,
                        label=f'{name} {pred_scores_f[i]:.2f}' if show_labels else None)
            center = pred_boxes[keep_idx][i]
            ax2.scatter([center[0]], [center[1]], [center[2]],
                        c=color, s=100, marker='*', edgecolors='black',
                        linewidths=1.5, zorder=20)

    ax2.set_xlabel('X (m)', fontsize=11)
    ax2.set_ylabel('Y (m)', fontsize=11)
    ax2.set_zlabel('Z (m)', fontsize=11)
    ax2.set_title('3D View | Dashed=GT  Solid=Pred  ★=Center',
                  fontsize=13, weight='bold')

    if crop:
        ax2.set_xlim(crop[0], crop[1])
        ax2.set_ylim(crop[2], crop[3])
    else:
        ax2.set_xlim(pc_range[0], pc_range[3])
        ax2.set_ylim(pc_range[1], pc_range[4])
    ax2.set_zlim(-3, 5)

    # 图例
    legend_elements = [Patch(facecolor=c, alpha=0.6, label=n) for n, c in CLASS_COLORS.items()]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=1,
               framealpha=0.9, edgecolor='gray')

    plt.tight_layout()
    plt.savefig(str(output_file), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  saved: {output_file}")


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str,
                        default='cfgs/custom_models/centerpoint_pp_warehouse.yaml')
    parser.add_argument('--ckpt', type=str,
                        default='../output/best_ckpts/centerpoint_pp_epoch30_mAP46.98.pth')
    parser.add_argument('--root_path', type=str, default='../data/warehouse')
    parser.add_argument('--output_dir', type=str, default='../output/warehouse_inference_viz')
    parser.add_argument('--score_thresh', type=float, default=0.1)
    parser.add_argument('--max_samples', type=int, default=999,
                        help='最大样本数，默认跑全部验证集')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val'])

    args = parser.parse_args()

    print("=" * 60)
    print("  Warehouse CenterPoint Inference Visualizer")
    print("  Pure black points | Compact crop | Center markers")
    print("=" * 60)

    logger = common_utils.create_logger()

    # 1. Config
    print("\n[1] Loading config...")
    cfg_from_yaml_file(args.cfg_file, cfg)
    root_path = Path(args.root_path).resolve()
    en_class_names = [translate_label(cn) for cn in cfg.CLASS_NAMES]
    print(f"  Model: {cfg.MODEL.NAME}  |  Classes: {en_class_names}")

    # 2. Dataset
    print(f"\n[2] Loading {args.split} set...")
    dataset = CustomDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        training=False, root_path=root_path, logger=logger,
    )
    dataset.custom_infos = []
    dataset.include_data('test' if args.split == 'val' else 'train')
    split_file = root_path / 'ImageSets' / f'{args.split}.txt'
    if split_file.exists():
        dataset.sample_id_list = [x.strip() for x in open(split_file).readlines()]
    print(f"  Samples: {len(dataset)}")

    total = min(len(dataset), args.max_samples)
    print(f"  Running inference on {total} samples...")

    # 3. Model
    print("\n[3] Loading model...")
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_params_from_file(filename=str(ckpt_path), logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    print(f"  Checkpoint: {ckpt_path}")

    # 4. Inference
    print(f"\n[4] Inferencing...")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pc_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
    all_centers = {}
    total_det = 0

    with torch.no_grad():
        for idx in range(total):
            data_dict = dataset[idx]
            sample_id = dataset.sample_id_list[idx]
            points = data_dict['points']
            gt_boxes = data_dict.get('gt_boxes', None)
            gt_names = data_dict.get('gt_names', None)
            gt_count = len(gt_boxes) if gt_boxes is not None else 0

            batch_dict = dataset.collate_batch([data_dict])
            load_data_to_gpu(batch_dict)
            pred_dicts, _ = model.forward(batch_dict)

            pd = pred_dicts[0]
            pb = pd['pred_boxes'].cpu().numpy()
            ps = pd['pred_scores'].cpu().numpy()
            pl = pd['pred_labels'].cpu().numpy()

            keep = ps >= args.score_thresh
            pb, ps, pl = pb[keep], ps[keep], pl[keep]
            si = np.argsort(ps)[::-1]
            pb, ps, pl = pb[si], ps[si], pl[si]

            pred_count = len(pb)
            total_det += pred_count

            # 中心点坐标
            centers_list = []
            for i, (box, score, label) in enumerate(zip(pb, ps, pl)):
                idx_l = int(label) - 1
                cls_name = translate_label(cfg.CLASS_NAMES[idx_l]) if 0 <= idx_l < len(cfg.CLASS_NAMES) else f'cls_{label}'
                centers_list.append({
                    "class": cls_name,
                    "score": round(float(score), 4),
                    "center_x": round(float(box[0]), 3),
                    "center_y": round(float(box[1]), 3),
                    "center_z": round(float(box[2]), 3),
                    "dx": round(float(box[3]), 3),
                    "dy": round(float(box[4]), 3),
                    "dz": round(float(box[5]), 3),
                    "heading": round(float(box[6]), 4),
                })

            all_centers[str(sample_id)] = {
                "gt_count": gt_count,
                "pred_count": pred_count,
                "detections": centers_list
            }

            # 画图
            output_file = output_dir / f'{sample_id}.png'
            create_visualization(
                points=points, pred_boxes=pb, pred_scores=ps, pred_labels=pl,
                gt_boxes=gt_boxes, gt_names=gt_names, class_names=cfg.CLASS_NAMES,
                output_file=output_file, pc_range=pc_range, sample_id=sample_id,
            )

            if (idx + 1) % 50 == 0 or idx == total - 1:
                print(f"  [{idx+1}/{total}] done, total detections: {total_det}")

    # 保存 JSON
    json_path = output_dir / 'prediction_centers.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_centers, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"  Done! Total detections: {total_det}")
    print(f"  Images: {output_dir}")
    print(f"  Centers JSON: {json_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()