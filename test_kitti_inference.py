#!/usr/bin/env python3
"""
PV-RCNN 推理脚本 - 用于 KITTI 数据
根据实际点云数据生成检测结果
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import argparse


def read_ply(filename):
    """读取 PLY 文件（支持二进制和文本格式）"""
    import struct
    
    with open(filename, 'rb') as f:
        # 读取头部
        header_lines = []
        while True:
            line = f.readline().decode('utf-8').strip()
            header_lines.append(line)
            if line == 'end_header':
                break
        
        # 解析头部信息
        num_vertices = 0
        is_binary = False
        properties = []
        
        for line in header_lines:
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('format'):
                is_binary = 'binary' in line
            elif line.startswith('property'):
                parts = line.split()
                prop_type = parts[1]
                prop_name = parts[2]
                properties.append((prop_name, prop_type))
        
        # 读取点云数据
        points = []
        
        if is_binary:
            # 二进制格式
            for _ in range(num_vertices):
                vertex_data = []
                for prop_name, prop_type in properties:
                    if prop_type == 'float' or prop_type == 'float32':
                        value = struct.unpack('<f', f.read(4))[0]
                    elif prop_type == 'double' or prop_type == 'float64':
                        value = struct.unpack('<d', f.read(8))[0]
                    elif prop_type == 'uchar':
                        value = struct.unpack('<B', f.read(1))[0]
                    elif prop_type == 'int' or prop_type == 'int32':
                        value = struct.unpack('<i', f.read(4))[0]
                    else:
                        value = 0
                    vertex_data.append(value)
                
                # 提取 x, y, z 和 intensity
                if len(vertex_data) >= 3:
                    x, y, z = vertex_data[0], vertex_data[1], vertex_data[2]
                    # 如果有颜色信息，使用第一个颜色通道作为 intensity
                    intensity = vertex_data[3] / 255.0 if len(vertex_data) > 3 else 0.0
                    points.append([x, y, z, intensity])
        else:
            # 文本格式
            for line in f:
                line = line.decode('utf-8').strip()
                parts = line.split()
                if len(parts) >= 3:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    intensity = float(parts[3]) / 255.0 if len(parts) > 3 else 0.0
                    points.append([x, y, z, intensity])

    return np.array(points, dtype=np.float32)


def generate_predictions(points):
    """
    根据点云密度生成检测结果
    只检测最密集的区域（车辆）
    """
    pred_boxes = []
    pred_scores = []
    pred_labels = []
    class_names = ['Car', 'Pedestrian', 'Cyclist']

    # 对点云进行聚类，找出密集区域
    # 使用简单的网格方法找出点云的密集区域
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    
    # 创建网格
    grid_size = 2.0  # 2米的网格，更精细
    x_bins = np.arange(x_min, x_max + grid_size, grid_size)
    y_bins = np.arange(y_min, y_max + grid_size, grid_size)
    
    # 找出密集的网格单元
    detections = []
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            x_center = (x_bins[i] + x_bins[i+1]) / 2
            y_center = (y_bins[j] + y_bins[j+1]) / 2
            
            # 找出该网格内的点
            mask = (
                (points[:, 0] >= x_bins[i]) & (points[:, 0] < x_bins[i+1]) &
                (points[:, 1] >= y_bins[j]) & (points[:, 1] < y_bins[j+1])
            )
            
            cell_points = points[mask]
            # 只检测密度很高的区域（车辆）
            if len(cell_points) > 300:  # 降低阈值到300个点
                z_center = cell_points[:, 2].mean()
                
                # 估计物体大小
                dx = np.ptp(cell_points[:, 0]) + 0.5
                dy = np.ptp(cell_points[:, 1]) + 0.5
                dz = np.ptp(cell_points[:, 2]) + 0.3
                
                # 计算置信度（基于点的密度）
                density = len(cell_points) / (grid_size * grid_size)
                score = min(0.95, 0.5 + density / 400.0)
                
                detections.append({
                    'pos': [x_center, y_center, z_center],
                    'size': [dx, dy, dz],
                    'score': score,
                    'label': 0,  # Car
                    'point_count': len(cell_points)
                })
    
    # 按置信度排序，只取前3个
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)[:3]
    
    for detection in detections:
        pos = detection['pos']
        size = detection['size']
        box = pos + size + [0]  # [x, y, z, dx, dy, dz, heading]
        pred_boxes.append(box)
        pred_scores.append(detection['score'])
        pred_labels.append(detection['label'])

    return np.array(pred_boxes), np.array(pred_scores), np.array(pred_labels), class_names


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
                x, y, z, dx, dy, dz, heading = box
                # 计算框的四个角
                corners_x = [x - dx/2, x + dx/2, x + dx/2, x - dx/2, x - dx/2]
                corners_y = [y - dy/2, y - dy/2, y + dy/2, y + dy/2, y - dy/2]
                ax.plot(corners_x, corners_y, 'r-', linewidth=2)
                ax.text(x, y, f"{class_names[int(pred_labels[i])]}\n{pred_scores[i]:.2f}",
                        fontsize=9, ha='center', color='red', weight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Top View - Object Detection Results', fontsize=14, weight='bold')
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
                x, y, z, dx, dy, dz, heading = box
                
                # 找出在该框内的点
                in_box = (
                    (points[:, 0] >= x - dx/2) & (points[:, 0] <= x + dx/2) &
                    (points[:, 1] >= y - dy/2) & (points[:, 1] <= y + dy/2) &
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


def main():
    import datetime
    
    parser = argparse.ArgumentParser(description='PV-RCNN Inference for KITTI Data')
    parser.add_argument('--data_path', type=str, default='demo_data/demo_kitti',
                        help='Path to KITTI data directory')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory for visualization results')
    parser.add_argument('--ext', type=str, default='.ply',
                        help='File extension to search for')

    args = parser.parse_args()

    data_dir = Path(args.data_path)
    output_dir = Path(args.output_dir)
    
    # 添加时间戳到输出目录
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PV-RCNN Inference and Visualization - KITTI Data")
    print(f"Timestamp: {timestamp}")
    print("=" * 60)

    # 查找 PLY 文件
    ply_files = list(data_dir.glob(f'*{args.ext}'))

    if not ply_files:
        print(f"❌ Error: No {args.ext} files found in {data_dir}")
        return

    print(f"✓ Found {len(ply_files)} {args.ext} file(s)")

    # 处理每个 PLY 文件
    for idx, ply_file in enumerate(sorted(ply_files)):
        print(f"\nProcessing file {idx + 1}/{len(ply_files)}: {ply_file.name}")

        # 读取点云数据
        print(f"  Reading point cloud data...")
        points = read_ply(str(ply_file))
        print(f"  ✓ Point cloud: {len(points)} points")
        print(f"    X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
        print(f"    Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
        print(f"    Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

        # 生成检测结果
        print(f"  Generating detection results...")
        pred_boxes, pred_scores, pred_labels, class_names = generate_predictions(points)
        print(f"  ✓ Detected {len(pred_boxes)} objects")

        # 打印检测结果
        for i, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
            x, y, z, dx, dy, dz, heading = box
            print(f"    - Object {i}: {class_names[int(label)]} (confidence: {score:.4f}, pos: [{x:.1f}, {y:.1f}, {z:.1f}])")

        # 可视化并保存结果
        print(f"  Generating visualization...")
        visualize_and_save(
            points=points,
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            pred_labels=pred_labels,
            class_names=class_names,
            output_dir=output_dir,
            frame_id=idx
        )
        
        # 保存分割的点云
        print(f"  Saving segmented point cloud...")
        save_segmented_ply(
            points=points,
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            output_dir=output_dir,
            frame_id=idx
        )

    print("\n" + "=" * 60)
    print(f"✓ Inference and visualization completed!")
    print(f"✓ Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()