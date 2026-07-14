#!/usr/bin/env python3
"""
PV-RCNN 推理脚本 - 使用真实的 PV-RCNN 模型进行 3D 点云检测
"""
import argparse
import glob
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 确保从 tools/ 目录加载配置（yaml 中 _BASE_CONFIG_ 使用相对路径 cfgs/...）
_tools_dir = str(Path(__file__).resolve().parent / 'tools')
if os.path.exists(_tools_dir):
    os.chdir(_tools_dir)

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=False, root_path=None, logger=None, ext='.bin'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def visualize_and_save(points, pred_boxes, pred_scores, pred_labels, class_names, output_dir, frame_id):
    """
    可视化点云和检测结果，保存俯视图
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建俯视图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    
    # 绘制点云
    scatter = ax.scatter(points[:, 0], points[:, 1], c=points[:, 3], cmap='viridis', s=2, alpha=0.6)
    plt.colorbar(scatter, ax=ax, label='Intensity')

    # 绘制检测框（俯视图）
    if len(pred_boxes) > 0:
        for i, box in enumerate(pred_boxes):
            if pred_scores[i] > 0.3:  # 只显示置信度 > 0.3 的框
                x, y, z, dx, dy, dz, heading = box[0], box[1], box[2], box[3], box[4], box[5], box[6]
                # 计算框的四个角（考虑旋转）
                cos_h = np.cos(heading)
                sin_h = np.sin(heading)
                
                # 未旋转的框的四个角
                corners_local = np.array([
                    [-dx/2, -dy/2],
                    [dx/2, -dy/2],
                    [dx/2, dy/2],
                    [-dx/2, dy/2],
                    [-dx/2, -dy/2]
                ])
                
                # 应用旋转
                rotation_matrix = np.array([
                    [cos_h, -sin_h],
                    [sin_h, cos_h]
                ])
                
                corners_rotated = corners_local @ rotation_matrix.T
                corners_x = corners_rotated[:, 0] + x
                corners_y = corners_rotated[:, 1] + y
                
                ax.plot(corners_x, corners_y, 'r-', linewidth=2)
                label_int = int(pred_labels[i])
                if label_int < len(class_names):
                    label_name = class_names[label_int]
                else:
                    label_name = f"class_{label_int}"
                ax.text(x, y, f"{label_name}\n{pred_scores[i]:.2f}",
                        fontsize=9, ha='center', color='red', weight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Top View - PV-RCNN Object Detection Results', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # 保存图像
    output_file = output_dir / f'top_view_{frame_id:06d}.png'
    plt.tight_layout()
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    print(f"  ✓ Top view saved: {output_file}")
    plt.close()


def save_segmented_ply(points, pred_boxes, pred_scores, output_dir, frame_id):
    """
    保存分割出的点云为 PLY 文件
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 为每个检测框内的点着色
    point_colors = np.zeros((len(points), 3), dtype=np.uint8)
    point_colors[:] = [128, 128, 128]  # 默认灰色
    
    # 颜色列表
    colors = [
        [255, 0, 0],      # 红色
        [0, 255, 0],      # 绿色
        [0, 0, 255],      # 蓝色
        [255, 255, 0],    # 黄色
        [255, 0, 255],    # 品红
        [0, 255, 255],    # 青色
    ]
    
    # 为检测框内的点着色
    if len(pred_boxes) > 0:
        for box_idx, box in enumerate(pred_boxes):
            if pred_scores[box_idx] > 0.3:
                x, y, z, dx, dy, dz, heading = box[0], box[1], box[2], box[3], box[4], box[5], box[6]
                
                # 找出在该框内的点（考虑旋转）
                cos_h = np.cos(heading)
                sin_h = np.sin(heading)
                
                # 将点转换到框的局部坐标系
                dx_local = (points[:, 0] - x) * cos_h + (points[:, 1] - y) * sin_h
                dy_local = -(points[:, 0] - x) * sin_h + (points[:, 1] - y) * cos_h
                
                in_box = (
                    (np.abs(dx_local) <= dx/2) &
                    (np.abs(dy_local) <= dy/2) &
                    (points[:, 2] >= z - dz/2) & (points[:, 2] <= z + dz/2)
                )
                
                # 为这些点着色
                color = colors[box_idx % len(colors)]
                point_colors[in_box] = color
    
    # 保存为 PLY 文件
    output_file = output_dir / f'segmented_points_{frame_id:06d}.ply'
    
    # 创建 PLY 文件头
    with open(str(output_file), 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        
        # 写入点数据
        for i, point in enumerate(points):
            f.write(f'{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} ')
            f.write(f'{int(point_colors[i, 0])} {int(point_colors[i, 1])} {int(point_colors[i, 2])}\n')
    
    print(f"  ✓ Segmented PLY saved: {output_file}")


def parse_config():
    parser = argparse.ArgumentParser(description='PV-RCNN Inference')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pv_rcnn.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='../demo_data/demo_kitti',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='../wights/pv_rcnn_8369.pth', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--output_dir', type=str, default='output', help='output directory')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------PV-RCNN Inference-------------------------')
    
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    
    if args.ckpt is None:
        logger.warning('No checkpoint specified, using model without pretrained weights')
    else:
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    
    model.cuda()
    model.eval()
    
    print("=" * 60)
    print("PV-RCNN Inference and Visualization - KITTI Data")
    print("=" * 60)
    
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Processing sample index: \t{idx + 1}')
            print(f"\nProcessing file {idx + 1}/{len(demo_dataset)}")
            
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            pred_dict = pred_dicts[0]
            
            # 提取预测结果
            pred_boxes = pred_dict['pred_boxes'].cpu().numpy()
            pred_scores = pred_dict['pred_scores'].cpu().numpy()
            pred_labels = pred_dict['pred_labels'].cpu().numpy()
            
            # 获取原始点云
            points = data_dict['points'][:, 1:].cpu().numpy()
            
            print(f"  ✓ Point cloud: {len(points)} points")
            print(f"  ✓ Detected {len(pred_boxes)} objects")
            
            # 打印检测结果
            for i, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
                if score > 0.3:
                    label_int = int(label)
                    if label_int < len(cfg.CLASS_NAMES):
                        label_name = cfg.CLASS_NAMES[label_int]
                    else:
                        label_name = f"class_{label_int}"
                    print(f"    - Object {i}: {label_name} (confidence: {score:.4f})")
            
            # 可视化并保存结果
            print(f"  Generating visualization...")
            visualize_and_save(
                points=points,
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                pred_labels=pred_labels,
                class_names=cfg.CLASS_NAMES,
                output_dir=args.output_dir,
                frame_id=idx
            )
            
            # 保存分割的点云
            print(f"  Saving segmented point cloud...")
            save_segmented_ply(
                points=points,
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                output_dir=args.output_dir,
                frame_id=idx
            )

    print("\n" + "=" * 60)
    print(f"✓ Inference and visualization completed!")
    print(f"✓ Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()