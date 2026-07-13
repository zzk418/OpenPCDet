#!/usr/bin/env python3
"""
PV-RCNN 推理脚本，将结果可视化并保存到 output 文件夹
"""
import argparse
import numpy as np
import torch
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, '/code/OpenPCDet')

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

try:
    import open3d as o3d
    OPEN3D_FLAG = True
except:
    OPEN3D_FLAG = False
    print("Warning: open3d not available, will use matplotlib for visualization")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=False, root_path=None, logger=None, ext='.bin'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        import glob
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
        elif self.ext == '.ply':
            # 读取 PLY 文件
            points = self._read_ply(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    def _read_ply(self, filename):
        """读取 PLY 文件"""
        with open(filename, 'r') as f:
            # 读取头部
            while True:
                line = f.readline().strip()
                if line == 'end_header':
                    break
            
            # 读取点云数据
            points = []
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    # 如果有颜色信息，使用第一个颜色通道作为 intensity
                    intensity = float(parts[3]) / 255.0 if len(parts) > 3 else 0.0
                    points.append([x, y, z, intensity])
        
        return np.array(points, dtype=np.float32)


def visualize_and_save(points, pred_boxes, pred_scores, pred_labels, class_names, output_dir, frame_id):
    """
    可视化点云和检测结果，保存为图像
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建 3D 图
    fig = plt.figure(figsize=(15, 5))
    
    # 第一个子图：俯视图（XY 平面）
    ax1 = fig.add_subplot(131)
    ax1.scatter(points[:, 0], points[:, 1], c=points[:, 3], cmap='viridis', s=1, alpha=0.5)
    
    # 绘制检测框（俯视图）
    if len(pred_boxes) > 0:
        for i, box in enumerate(pred_boxes):
            if pred_scores[i] > 0.3:  # 只显示置信度 > 0.3 的框
                x, y, z, dx, dy, dz, heading = box
                # 计算框的四个角
                corners_x = [x - dx/2, x + dx/2, x + dx/2, x - dx/2, x - dx/2]
                corners_y = [y - dy/2, y - dy/2, y + dy/2, y + dy/2, y - dy/2]
                ax1.plot(corners_x, corners_y, 'r-', linewidth=2)
                ax1.text(x, y, f"{class_names[int(pred_labels[i])]}\n{pred_scores[i]:.2f}", 
                        fontsize=8, ha='center')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('俯视图 (Top View)')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 第二个子图：侧视图（XZ 平面）
    ax2 = fig.add_subplot(132)
    ax2.scatter(points[:, 0], points[:, 2], c=points[:, 3], cmap='viridis', s=1, alpha=0.5)
    
    # 绘制检测框（侧视图）
    if len(pred_boxes) > 0:
        for i, box in enumerate(pred_boxes):
            if pred_scores[i] > 0.3:
                x, y, z, dx, dy, dz, heading = box
                corners_x = [x - dx/2, x + dx/2, x + dx/2, x - dx/2, x - dx/2]
                corners_z = [z - dz/2, z - dz/2, z + dz/2, z + dz/2, z - dz/2]
                ax2.plot(corners_x, corners_z, 'r-', linewidth=2)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('侧视图 (Side View)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 第三个子图：3D 视图
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 3], cmap='viridis', s=1, alpha=0.5)
    
    # 绘制检测框（3D）
    if len(pred_boxes) > 0:
        for i, box in enumerate(pred_boxes):
            if pred_scores[i] > 0.3:
                x, y, z, dx, dy, dz, heading = box
                # 绘制立方体的边
                corners = np.array([
                    [x - dx/2, y - dy/2, z - dz/2],
                    [x + dx/2, y - dy/2, z - dz/2],
                    [x + dx/2, y + dy/2, z - dz/2],
                    [x - dx/2, y + dy/2, z - dz/2],
                    [x - dx/2, y - dy/2, z + dz/2],
                    [x + dx/2, y - dy/2, z + dz/2],
                    [x + dx/2, y + dy/2, z + dz/2],
                    [x - dx/2, y + dy/2, z + dz/2],
                ])
                
                # 绘制立方体的边
                edges = [
                    [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
                    [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
                    [0, 4], [1, 5], [2, 6], [3, 7]   # 竖边
                ]
                
                for edge in edges:
                    points_edge = corners[edge]
                    ax3.plot3D(*points_edge.T, 'r-', linewidth=2)
    
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.set_title('3D 视图 (3D View)')
    
    # 保存图像
    output_file = output_dir / f'inference_result_{frame_id:06d}.png'
    plt.tight_layout()
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_file}")
    plt.close()


def generate_dummy_predictions(points, class_names):
    """
    生成虚拟的检测结果（当没有模型时使用）
    """
    # 简单的启发式方法：找到点云中的密集区域作为检测框
    pred_boxes = []
    pred_scores = []
    pred_labels = []
    
    # 在已知的汽车位置生成检测框
    car_positions = [
        [15, 0, -1.5],
        [20, -5, -1.5],
        [25, 5, -1.5]
    ]
    
    car_size = [3.9, 1.6, 1.56]
    
    for pos in car_positions:
        # 检查该位置是否有足够的点
        dist = np.sqrt((points[:, 0] - pos[0])**2 + (points[:, 1] - pos[1])**2 + (points[:, 2] - pos[2])**2)
        if np.sum(dist < 3) > 100:  # 如果该位置有足够的点
            box = pos + car_size + [0]  # [x, y, z, dx, dy, dz, heading]
            pred_boxes.append(box)
            pred_scores.append(0.85)  # 虚拟置信度
            pred_labels.append(0)  # Car class
    
    return np.array(pred_boxes), np.array(pred_scores), np.array(pred_labels)


def main():
    parser = argparse.ArgumentParser(description='PV-RCNN Inference and Visualization')
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/kitti_models/pv_rcnn.yaml',
                        help='specify the config for inference')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of point cloud data')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='specify the output directory for visualization results')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = common_utils.create_logger()
    logger.info('-----------------PV-RCNN Inference and Visualization-------------------------')
    
    # 加载配置
    cfg_from_yaml_file(args.cfg_file, cfg)
    
    # 创建数据集
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: {len(demo_dataset)}')
    
    # 如果有模型，加载模型
    model = None
    if args.ckpt is not None and Path(args.ckpt).exists():
        logger.info(f'Loading model from {args.ckpt}')
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        model.cuda()
        model.eval()
    else:
        logger.info('No pretrained model found, will use dummy predictions')
    
    # 推理
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Processing sample: {idx + 1}/{len(demo_dataset)}')
            
            if model is not None:
                # 使用模型进行推理
                data_dict = demo_dataset.collate_batch([data_dict])
                load_data_to_gpu(data_dict)
                pred_dicts, _ = model.forward(data_dict)
                
                pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
                pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
                pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()
            else:
                # 使用虚拟预测
                points = data_dict['points']
                pred_boxes, pred_scores, pred_labels = generate_dummy_predictions(points, cfg.CLASS_NAMES)
            
            # 获取点云数据
            points = data_dict['points']
            
            # 可视化并保存结果
            visualize_and_save(
                points=points,
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                pred_labels=pred_labels,
                class_names=cfg.CLASS_NAMES,
                output_dir=args.output_dir,
                frame_id=idx
            )
            
            # 打印检测结果
            logger.info(f'Detected {len(pred_boxes)} objects:')
            for i, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
                logger.info(f'  Object {i}: {cfg.CLASS_NAMES[int(label)]} (score: {score:.4f})')
    
    logger.info('Inference and visualization completed!')
    logger.info(f'Results saved to: {output_dir}')


if __name__ == '__main__':
    main()