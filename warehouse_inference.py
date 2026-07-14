#!/usr/bin/env python3
"""
仓库场景 PV-RCNN 推理 + 多视图可视化
- 俯视图(BEV) + 3D 视角
- 自动裁剪到物体周围区域
- 英文标签（避免中文乱码）
- 使用训练集样本
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_tools_dir = str(Path(__file__).resolve().parent / 'tools')
if os.path.exists(_tools_dir):
    os.chdir(_tools_dir)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.custom.custom_dataset import CustomDataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

# 英文标签映射（避免中文乱码）
LABEL_MAP = {
    '箱子': 'Box',
    '电动运输车': 'ELFplusplus',
    '货运自行车': 'CargoBike',
    '无人搬运车': 'FTS',
    '叉车': 'ForkLift',
}

# 类别颜色
CLASS_COLORS = {
    'Box': '#2196F3',      # 蓝
    'ELFplusplus': '#FF9800',  # 橙
    'CargoBike': '#4CAF50',    # 绿
    'FTS': '#9C27B0',         # 紫
    'ForkLift': '#F44336',    # 红
}


def translate_label(name):
    """中文 → 英文"""
    for cn, en in LABEL_MAP.items():
        if cn in str(name):
            return en
    return str(name)


def boxes_to_corners(boxes):
    """将 3D 框 [x,y,z,dx,dy,dz,heading] 转为 8 个角点 (N,8,3)"""
    if len(boxes) == 0:
        return np.zeros((0, 8, 3))
    boxes = np.array(boxes)
    corners_list = []
    for box in boxes:
        x, y, z, dx, dy, dz, heading = box[:7]
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        corners_local = np.array([
            [-dx/2, -dy/2], [dx/2, -dy/2], [dx/2, dy/2], [-dx/2, dy/2]
        ])
        corners_xy = corners_local @ rot.T
        corners_xy[:, 0] += x
        corners_xy[:, 1] += y
        bottom = np.hstack([corners_xy, np.ones((4,1)) * (z - dz/2)])
        top = np.hstack([corners_xy, np.ones((4,1)) * (z + dz/2)])
        corners = np.vstack([bottom, top])
        corners_list.append(corners)
    return np.array(corners_list)


def auto_crop_range(points, boxes_list, padding=15.0):
    """根据物体位置自动裁剪显示范围"""
    all_pts = []
    for p in boxes_list:
        if len(p) > 0:
            all_pts.append(p[:, :2])
    if not all_pts:
        return None
    all_pts = np.vstack(all_pts)
    if len(all_pts) == 0:
        return None
    x_min, y_min = all_pts.min(axis=0) - padding
    x_max, y_max = all_pts.max(axis=0) + padding
    # 保持等比例
    dx = x_max - x_min
    dy = y_max - y_min
    max_dim = max(dx, dy)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    x_min = x_center - max_dim / 2
    x_max = x_center + max_dim / 2
    y_min = y_center - max_dim / 2
    y_max = y_center + max_dim / 2
    return [x_min, x_max, y_min, y_max]


def draw_3d_box(ax, corners, color, alpha, linewidth, label=None):
    """在 3D axes 上画单个框"""
    b = corners
    edges = [
        (0,1),(1,2),(2,3),(3,0),  # bottom
        (4,5),(5,6),(6,7),(7,4),  # top
        (0,4),(1,5),(2,6),(3,7),  # vertical
    ]
    for i, j in edges:
        ax.plot([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]],
                color=color, alpha=alpha, linewidth=linewidth)
    if label:
        ax.text(b[4,0], b[4,1], b[4,2], label, fontsize=8, color=color, weight='bold',
                ha='center', va='bottom')


def create_visualization(points, pred_boxes, pred_scores, pred_labels,
                         gt_boxes, gt_names, class_names, output_file,
                         pc_range, sample_id=""):
    """创建多视图可视化：俯视图 + 3D 视图"""
    fig = plt.figure(figsize=(20, 9), dpi=120)

    # 英文标签
    en_gt_names = [translate_label(n) for n in gt_names] if gt_names is not None else []
    en_pred_labels = []
    for l in pred_labels:
        idx = int(l) - 1
        if 0 <= idx < len(class_names):
            en_pred_labels.append(translate_label(class_names[idx]))
        else:
            en_pred_labels.append(f'cls_{l}')

    # 自动裁剪范围
    all_boxes = []
    if gt_boxes is not None and len(gt_boxes) > 0:
        all_boxes.append(gt_boxes)
    if len(pred_boxes) > 0:
        keep = pred_scores >= 0.1
        all_boxes.append(pred_boxes[keep])
    crop = auto_crop_range(points, all_boxes, padding=12.0)

    # ---------- 左图：俯视图 (BEV) ----------
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(points[:, 0], points[:, 1], c='black', s=0.3, alpha=0.5, rasterized=True)

    # GT 框 (虚线)
    if gt_boxes is not None and len(gt_boxes) > 0:
        for box, name in zip(gt_boxes, en_gt_names):
            color = CLASS_COLORS.get(name, '#00CC00')
            corners = boxes_to_corners([box])[0]
            bx, by = corners[[0,1,2,3,0], 0], corners[[0,1,2,3,0], 1]
            ax1.plot(bx, by, '--', color=color, linewidth=2.0, alpha=0.8)
            ax1.text(box[0], box[1], f'GT:{name}', fontsize=8, ha='center', va='bottom',
                     color='white', weight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))

    # 预测框 (实线)
    if len(pred_boxes) > 0:
        for i, box in enumerate(pred_boxes):
            if pred_scores[i] < 0.1:
                continue
            name = en_pred_labels[i] if i < len(en_pred_labels) else '?'
            color = CLASS_COLORS.get(name, 'red')
            corners = boxes_to_corners([box])[0]
            bx, by = corners[[0,1,2,3,0], 0], corners[[0,1,2,3,0], 1]
            ax1.plot(bx, by, '-', color=color, linewidth=2.5, alpha=0.9)
            ax1.fill(bx, by, alpha=0.1, color=color)
            ax1.text(box[0], box[1], f'{name}\n{pred_scores[i]:.2f}',
                     fontsize=8, ha='center', va='bottom', color='white', weight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))

    ax1.set_xlabel('X (m)', fontsize=11)
    ax1.set_ylabel('Y (m)', fontsize=11)
    ax1.set_title(f'BEV Top-Down: {sample_id}  |  Dashed=GT  Solid=Pred', fontsize=12, weight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axis('equal')
    ax1.set_facecolor('#F8F8F8')

    if crop:
        ax1.set_xlim(crop[0], crop[1])
        ax1.set_ylim(crop[2], crop[3])
    else:
        ax1.set_xlim(pc_range[0], pc_range[3])
        ax1.set_ylim(pc_range[1], pc_range[4])

    # ---------- 右图：3D 视图 ----------
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_facecolor('#F8F8F8')

    # 抽样显示点云（太多点影响性能）
    n_pts = min(len(points), 3000)
    idx = np.random.choice(len(points), n_pts, replace=False)
    ax2.scatter(points[idx, 0], points[idx, 1], points[idx, 2],
                c='black', s=0.3, alpha=0.5, rasterized=True)

    # GT 3D 框
    if gt_boxes is not None and len(gt_boxes) > 0:
        gt_corners = boxes_to_corners(gt_boxes)
        for i, corners in enumerate(gt_corners):
            name = en_gt_names[i] if i < len(en_gt_names) else '?'
            color = CLASS_COLORS.get(name, '#00CC00')
            draw_3d_box(ax2, corners, color, alpha=0.7, linewidth=1.5, label=f'GT:{name}')

    # 预测 3D 框
    if len(pred_boxes) > 0:
        keep_idx = pred_scores >= 0.1
        pred_corners_all = boxes_to_corners(pred_boxes[keep_idx])
        pred_scores_filtered = pred_scores[keep_idx]
        en_labels_filtered = [en_pred_labels[i] for i in range(len(en_pred_labels)) if keep_idx[i]]
        for i, corners in enumerate(pred_corners_all):
            name = en_labels_filtered[i] if i < len(en_labels_filtered) else '?'
            color = CLASS_COLORS.get(name, 'red')
            label = f'{name} {pred_scores_filtered[i]:.2f}'
            draw_3d_box(ax2, corners, color, alpha=0.9, linewidth=2.0, label=label)

    ax2.set_xlabel('X (m)', fontsize=10)
    ax2.set_ylabel('Y (m)', fontsize=10)
    ax2.set_zlabel('Z (m)', fontsize=10)
    ax2.set_title('3D View  |  Dashed=GT  Solid=Pred', fontsize=12, weight='bold')

    # 裁剪 3D 视图范围
    if crop:
        ax2.set_xlim(crop[0], crop[1])
        ax2.set_ylim(crop[2], crop[3])
    else:
        ax2.set_xlim(pc_range[0], pc_range[3])
        ax2.set_ylim(pc_range[1], pc_range[4])
    ax2.set_zlim(-3, 5)

    # 图例
    legend_elements = []
    for name, color in CLASS_COLORS.items():
        from matplotlib.patches import Patch
        legend_elements.append(Patch(facecolor=color, alpha=0.6, label=name))
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=1,
               framealpha=0.8, edgecolor='gray')

    plt.tight_layout()
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='仓库 PV-RCNN 推理可视化 (改进版)')
    parser.add_argument('--cfg_file', type=str,
                        default='cfgs/custom_models/pv_rcnn_warehouse.yaml')
    parser.add_argument('--ckpt', type=str,
                        default='../output/custom_models/pv_rcnn_warehouse/default/ckpt/checkpoint_epoch_30.pth')
    parser.add_argument('--root_path', type=str, default='../data/warehouse')
    parser.add_argument('--output_dir', type=str, default='../output/warehouse_train_inference')
    parser.add_argument('--score_thresh', type=float, default=0.1)
    parser.add_argument('--max_samples', type=int, default=30)
    parser.add_argument('--use_training', action='store_true', default=True,
                        help='使用训练集 (默认 True)')

    args = parser.parse_args()

    print("=" * 60)
    print("  仓库 PV-RCNN 推理可视化 (改进版)")
    print("  Multi-view: BEV + 3D  |  Auto-crop  |  English labels")
    print("=" * 60)

    logger = common_utils.create_logger()

    # ------ 1. 加载配置 ------
    print("\n[1] 加载配置...")
    cfg_from_yaml_file(args.cfg_file, cfg)
    root_path = Path(args.root_path).resolve()
    en_class_names = [translate_label(cn) for cn in cfg.CLASS_NAMES]
    print(f"  模型: {cfg.MODEL.NAME}")
    print(f"  类别: {en_class_names}")
    print(f"  数据路径: {root_path}")

    # ------ 2. 加载训练集 ------
    print("\n[2] 加载训练集...")
    cfg.DATA_CONFIG.DATA_SPLIT['train'] = 'train'
    dataset = CustomDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=root_path,
        logger=logger,
    )

    # 切换到训练集
    from pcdet.datasets.custom.custom_dataset import CustomDataset as CD
    dataset.split = 'train'
    split_file = root_path / 'ImageSets' / 'train.txt'
    if split_file.exists():
        dataset.sample_id_list = [x.strip() for x in open(split_file).readlines()]
    dataset.custom_infos = []
    dataset.include_data('train')

    print(f"  训练集样本数: {len(dataset)}")

    total = min(len(dataset), args.max_samples)
    import random
    random.seed(42)
    indices = sorted(random.sample(range(len(dataset)), total))
    print(f"  随机选择 {total} 个样本")

    # ------ 3. 加载模型 ------
    print("\n[3] 加载模型...")
    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset,
    )
    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到权重: {ckpt_path}")
    model.load_params_from_file(filename=str(ckpt_path), logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    print(f"  权重: {ckpt_path}")

    # ------ 4. 推理 ------
    print(f"\n[4] 开始推理 ({total} 个样本)...")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pc_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
    total_detections = 0

    with torch.no_grad():
        for t_idx, idx in enumerate(indices):
            print(f"\n  [{t_idx+1}/{total}] Sample #{idx}", end=' ')
            data_dict = dataset[idx]

            sample_id = dataset.sample_id_list[idx]
            points = data_dict['points']
            gt_boxes = data_dict.get('gt_boxes', None)
            gt_names = data_dict.get('gt_names', None)
            gt_count = len(gt_boxes) if gt_boxes is not None else 0

            # 前向推理
            batch_dict = dataset.collate_batch([data_dict])
            load_data_to_gpu(batch_dict)
            pred_dicts, _ = model.forward(batch_dict)

            pred_dict = pred_dicts[0]
            pred_boxes = pred_dict['pred_boxes'].cpu().numpy()
            pred_scores = pred_dict['pred_scores'].cpu().numpy()
            pred_labels = pred_dict['pred_labels'].cpu().numpy()

            # 过滤排序
            keep = pred_scores >= args.score_thresh
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]
            sort_idx = np.argsort(pred_scores)[::-1]
            pred_boxes = pred_boxes[sort_idx]
            pred_scores = pred_scores[sort_idx]
            pred_labels = pred_labels[sort_idx]

            pred_count = len(pred_boxes)
            total_detections += pred_count

            print(f"GT={gt_count}  Pred={pred_count}", end='')
            for i, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
                if i >= 3:
                    print(f"  ... +{pred_count-3}", end='')
                    break
                idx_l = int(label) - 1
                name = translate_label(cfg.CLASS_NAMES[idx_l]) if 0 <= idx_l < len(cfg.CLASS_NAMES) else f'cls_{label}'
                print(f"  {name} {score:.2f}", end='')

            # 画图
            output_file = output_dir / f'{sample_id}.png'
            create_visualization(
                points=points,
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                pred_labels=pred_labels,
                gt_boxes=gt_boxes,
                gt_names=gt_names,
                class_names=cfg.CLASS_NAMES,
                output_file=output_file,
                pc_range=pc_range,
                sample_id=sample_id,
            )

    print("\n" + "=" * 60)
    print(f"  Done! 总检测数: {total_detections}")
    print(f"  输出目录: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()