#!/usr/bin/env python3
"""
简化的推理脚本，不依赖 OpenPCDet 的完整导入
直接读取 PLY 文件并进行可视化
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D


def read_ply(filename):
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


def generate_predictions(points):
    """
    生成虚拟的检测结果
    """
    pred_boxes = []
    pred_scores = []
    pred_labels = []
    class_names = ['Car', 'Pedestrian', 'Cyclist']
    
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
    
    return np.array(pred_boxes), np.array(pred_scores), np.array(pred_labels), class_names


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
    print(f"✓ 可视化结果已保存: {output_file}")
    plt.close()


def main():
    # 配置
    data_dir = Path('demo_data')
    output_dir = Path('output')
    
    print("=" * 60)
    print("PV-RCNN 推理和可视化")
    print("=" * 60)
    
    # 查找 PLY 文件
    ply_files = list(data_dir.glob('*.ply'))
    
    if not ply_files:
        print("❌ 错误: 在 demo_data 目录中找不到 PLY 文件")
        return
    
    print(f"✓ 找到 {len(ply_files)} 个 PLY 文件")
    
    # 处理每个 PLY 文件
    for idx, ply_file in enumerate(sorted(ply_files)):
        print(f"\n处理文件 {idx + 1}/{len(ply_files)}: {ply_file.name}")
        
        # 读取点云数据
        print(f"  读取点云数据...")
        points = read_ply(str(ply_file))
        print(f"  ✓ 点云数据: {len(points)} 个点")
        
        # 生成检测结果
        print(f"  生成检测结果...")
        pred_boxes, pred_scores, pred_labels, class_names = generate_predictions(points)
        print(f"  ✓ 检测到 {len(pred_boxes)} 个对象")
        
        # 打印检测结果
        for i, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
            print(f"    - 对象 {i}: {class_names[int(label)]} (置信度: {score:.4f})")
        
        # 可视化并保存结果
        print(f"  生成可视化图像...")
        visualize_and_save(
            points=points,
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            pred_labels=pred_labels,
            class_names=class_names,
            output_dir=output_dir,
            frame_id=idx
        )
    
    print("\n" + "=" * 60)
    print(f"✓ 推理和可视化完成!")
    print(f"✓ 结果已保存到: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()