#!/usr/bin/env python3
"""
创建简单的测试数据 - 几个明显的长方形点云
"""
import numpy as np
from pathlib import Path

def create_box_points(center, size, num_points=500):
    """
    创建一个长方形点云
    center: [x, y, z]
    size: [dx, dy, dz]
    """
    x_min, x_max = center[0] - size[0]/2, center[0] + size[0]/2
    y_min, y_max = center[1] - size[1]/2, center[1] + size[1]/2
    z_min, z_max = center[2] - size[2]/2, center[2] + size[2]/2
    
    # 随机生成点
    points = np.random.uniform(
        [x_min, y_min, z_min],
        [x_max, y_max, z_max],
        (num_points, 3)
    )
    
    # 添加强度值（随机）
    intensity = np.random.uniform(0.3, 1.0, (num_points, 1))
    points = np.hstack([points, intensity])
    
    return points

def create_test_data():
    """
    创建测试数据：
    - 两个明显的车辆（长方形）
    - 一些背景点
    """
    print("Creating simple test data...")
    
    # 创建输出目录
    output_dir = Path('demo_data/simple_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建两个车辆点云
    # 车辆 1：位置 [10, 0, -1.5]，大小 [4, 2, 1.5]
    car1_points = create_box_points([10, 0, -1.5], [4, 2, 1.5], num_points=800)
    
    # 车辆 2：位置 [20, 5, -1.5]，大小 [4, 2, 1.5]
    car2_points = create_box_points([20, 5, -1.5], [4, 2, 1.5], num_points=800)
    
    # 创建背景点（稀疏的地面点）
    background_x = np.random.uniform(-5, 35, 2000)
    background_y = np.random.uniform(-10, 15, 2000)
    background_z = np.random.uniform(-3.5, -3.0, 2000)
    background_intensity = np.random.uniform(0.1, 0.3, 2000)
    background_points = np.column_stack([background_x, background_y, background_z, background_intensity])
    
    # 合并所有点
    all_points = np.vstack([car1_points, car2_points, background_points])
    
    # 随机打乱顺序
    np.random.shuffle(all_points)
    
    print(f"Total points: {len(all_points)}")
    print(f"  Car 1: 800 points at [10, 0, -1.5]")
    print(f"  Car 2: 800 points at [20, 5, -1.5]")
    print(f"  Background: 2000 points")
    
    # 保存为 .bin 文件
    bin_file = output_dir / 'simple_test_000000.bin'
    all_points.astype(np.float32).tofile(str(bin_file))
    print(f"✓ Saved: {bin_file}")
    
    # 保存为 .npy 文件
    npy_file = output_dir / 'simple_test_000000.npy'
    np.save(str(npy_file), all_points.astype(np.float32))
    print(f"✓ Saved: {npy_file}")
    
    # 保存为 PLY 文件
    ply_file = output_dir / 'simple_test_000000.ply'
    with open(str(ply_file), 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(all_points)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        
        for point in all_points:
            x, y, z, intensity = point
            # 将强度转换为 RGB
            intensity_val = int(np.clip(intensity * 255, 0, 255))
            f.write(f'{x:.6f} {y:.6f} {z:.6f} {intensity_val} {intensity_val} {intensity_val}\n')
    
    print(f"✓ Saved: {ply_file}")
    
    print("\n✓ Test data created successfully!")
    print(f"  Location: {output_dir}")
    print(f"  Files: simple_test_000000.bin, .npy, .ply")

if __name__ == '__main__':
    create_test_data()