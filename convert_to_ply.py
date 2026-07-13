#!/usr/bin/env python3
"""
将 KITTI 格式的点云数据转换为 PLY 格式
支持 .bin 和 .npy 文件
"""
import numpy as np
from pathlib import Path
import os

def read_kitti_bin(bin_file):
    """
    读取 KITTI 格式的 .bin 文件
    
    Args:
        bin_file: .bin 文件路径
        
    Returns:
        点云数据，形状为 (N, 4)，格式为 [x, y, z, intensity]
    """
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    return points

def read_kitti_npy(npy_file):
    """
    读取 NumPy 格式的 .npy 文件
    
    Args:
        npy_file: .npy 文件路径
        
    Returns:
        点云数据，形状为 (N, 4)，格式为 [x, y, z, intensity]
    """
    points = np.load(npy_file)
    return points

def write_ply(ply_file, points):
    """
    将点云数据写入 PLY 文件
    
    Args:
        ply_file: 输出 PLY 文件路径
        points: 点云数据，形状为 (N, 4)，格式为 [x, y, z, intensity]
    """
    num_points = len(points)
    
    with open(ply_file, 'w') as f:
        # 写入 PLY 头
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # 写入点云数据
        for point in points:
            x, y, z, intensity = point
            # 将 intensity 转换为 RGB 值 (0-255)
            intensity_val = int(np.clip(intensity * 255, 0, 255))
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {intensity_val} {intensity_val} {intensity_val}\n")
    
    print(f"✓ 已转换: {ply_file} ({num_points} 个点)")

def convert_file(input_file, output_file=None):
    """
    转换单个文件
    
    Args:
        input_file: 输入文件路径 (.bin 或 .npy)
        output_file: 输出文件路径 (可选，默认替换扩展名为 .ply)
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"✗ 文件不存在: {input_file}")
        return False
    
    # 确定输出文件路径
    if output_file is None:
        output_path = input_path.with_suffix('.ply')
    else:
        output_path = Path(output_file)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 读取输入文件
        if input_path.suffix == '.bin':
            print(f"读取 KITTI .bin 文件: {input_file}")
            points = read_kitti_bin(str(input_path))
        elif input_path.suffix == '.npy':
            print(f"读取 NumPy .npy 文件: {input_file}")
            points = read_kitti_npy(str(input_path))
        else:
            print(f"✗ 不支持的文件格式: {input_path.suffix}")
            return False
        
        # 确保点云数据格式正确
        if points.shape[1] < 4:
            print(f"✗ 点云数据格式错误，需要至少 4 列 (x, y, z, intensity)")
            return False
        
        # 只取前 4 列 (x, y, z, intensity)
        points = points[:, :4]
        
        # 写入 PLY 文件
        write_ply(str(output_path), points)
        return True
        
    except Exception as e:
        print(f"✗ 转换失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("KITTI 点云数据转换为 PLY 格式")
    print("=" * 60)
    
    # 转换 demo_data 目录下的所有 .bin 和 .npy 文件
    demo_dir = Path('demo_data')
    if demo_dir.exists():
        print(f"\n扫描目录: {demo_dir}")
        
        # 转换 .bin 文件
        bin_files = list(demo_dir.glob('*.bin'))
        if bin_files:
            print(f"\n找到 {len(bin_files)} 个 .bin 文件:")
            for bin_file in bin_files:
                convert_file(str(bin_file))
        
        # 转换 .npy 文件
        npy_files = list(demo_dir.glob('*.npy'))
        if npy_files:
            print(f"\n找到 {len(npy_files)} 个 .npy 文件:")
            for npy_file in npy_files:
                convert_file(str(npy_file))
        
        if not bin_files and not npy_files:
            print("未找到 .bin 或 .npy 文件")
    else:
        print(f"✗ 目录不存在: {demo_dir}")
    
    # 转换 data/kitti 目录下的所有 .bin 和 .npy 文件
    kitti_dir = Path('data/kitti')
    if kitti_dir.exists():
        print(f"\n扫描目录: {kitti_dir}")
        
        # 递归查找所有 .bin 和 .npy 文件
        bin_files = list(kitti_dir.rglob('*.bin'))
        npy_files = list(kitti_dir.rglob('*.npy'))
        
        if bin_files:
            print(f"\n找到 {len(bin_files)} 个 .bin 文件:")
            for bin_file in bin_files:
                convert_file(str(bin_file))
        
        if npy_files:
            print(f"\n找到 {len(npy_files)} 个 .npy 文件:")
            for npy_file in npy_files:
                convert_file(str(npy_file))
        
        if not bin_files and not npy_files:
            print("未找到 .bin 或 .npy 文件")
    
    print("\n" + "=" * 60)
    print("转换完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()