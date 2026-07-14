#!/usr/bin/env python3
"""
简单的KITTI数据测试 - 不依赖所有模块
"""

import os
import numpy as np
from pathlib import Path

def test_kitti_data():
    """测试KITTI示例数据"""
    
    print("="*60)
    print("KITTI示例数据验证")
    print("="*60)
    
    data_path = Path('data/kitti_sample/training')
    
    if not data_path.exists():
        print(f"✗ 数据路径不存在: {data_path}")
        return False
    
    print(f"✓ 数据路径存在: {data_path}\n")
    
    # 检查各个子目录
    subdirs = ['calib', 'image_2', 'label_2', 'velodyne']
    results = {}
    
    for subdir in subdirs:
        subdir_path = data_path / subdir
        if subdir_path.exists():
            files = list(subdir_path.glob('*'))
            results[subdir] = len(files)
            print(f"✓ {subdir}/: {len(files)} 文件")
        else:
            results[subdir] = 0
            print(f"✗ {subdir}/: 不存在")
    
    print("\n" + "="*60)
    print("数据内容检查")
    print("="*60)
    
    # 检查标定文件
    calib_files = list((data_path / 'calib').glob('*.txt'))
    if calib_files:
        print(f"\n✓ 标定文件 ({len(calib_files)} 个):")
        with open(calib_files[0], 'r') as f:
            lines = f.readlines()[:3]
            for line in lines:
                print(f"  {line.strip()}")
    
    # 检查标签文件
    label_files = list((data_path / 'label_2').glob('*.txt'))
    if label_files:
        print(f"\n✓ 标签文件 ({len(label_files)} 个):")
        with open(label_files[0], 'r') as f:
            lines = f.readlines()
            for line in lines:
                print(f"  {line.strip()}")
    
    # 检查点云文件
    velodyne_files = list((data_path / 'velodyne').glob('*.bin'))
    if velodyne_files:
        print(f"\n✓ 点云文件 ({len(velodyne_files)} 个):")
        points = np.fromfile(velodyne_files[0], dtype=np.float32).reshape(-1, 4)
        print(f"  - 点数: {len(points)}")
        print(f"  - X范围: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
        print(f"  - Y范围: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
        print(f"  - Z范围: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
        print(f"  - 强度范围: [{points[:, 3].min():.2f}, {points[:, 3].max():.2f}]")
    
    # 检查图像文件
    image_files = list((data_path / 'image_2').glob('*.png'))
    if image_files:
        print(f"\n✓ 图像文件 ({len(image_files)} 个):")
        try:
            from PIL import Image
            img = Image.open(image_files[0])
            print(f"  - 分辨率: {img.size}")
            print(f"  - 格式: {img.format}")
        except:
            print(f"  - 无法读取图像信息")
    
    print("\n" + "="*60)
    print("数据统计")
    print("="*60)
    
    total_files = sum(results.values())
    print(f"\n总文件数: {total_files}")
    print(f"样本数: {results['calib']}")
    
    if all(results.values()):
        print("\n✓ KITTI示例数据完整！")
        print("\n数据特点:")
        print("  - 5个样本")
        print("  - 每个样本包含:")
        print("    * 标定文件 (calib)")
        print("    * 标签文件 (label_2)")
        print("    * 点云文件 (velodyne)")
        print("    * 图像文件 (image_2)")
        print("\n可用于测试模型的数据加载和推理")
        return True
    else:
        print("\n✗ KITTI示例数据不完整")
        return False

if __name__ == '__main__':
    success = test_kitti_data()
    
    print("\n" + "="*60)
    if success:
        print("✓ 验证完成！数据可用于测试")
    else:
        print("✗ 验证失败")
    print("="*60)#!/usr/bin/env python3
"""
简单的KITTI数据测试 - 不依赖所有模块
"""

import os
import numpy as np
from pathlib import Path

def test_kitti_data():
    """测试KITTI示例数据"""
    
    print("="*60)
    print("KITTI示例数据验证")
    print("="*60)
    
    data_path = Path('data/kitti_sample/training')
    
    if not data_path.exists():
        print(f"✗ 数据路径不存在: {data_path}")
        return False
    
    print(f"✓ 数据路径存在: {data_path}\n")
    
    # 检查各个子目录
    subdirs = ['calib', 'image_2', 'label_2', 'velodyne']
    results = {}
    
    for subdir in subdirs:
        subdir_path = data_path / subdir
        if subdir_path.exists():
            files = list(subdir_path.glob('*'))
            results[subdir] = len(files)
            print(f"✓ {subdir}/: {len(files)} 文件")
        else:
            results[subdir] = 0
            print(f"✗ {subdir}/: 不存在")
    
    print("\n" + "="*60)
    print("数据内容检查")
    print("="*60)
    
    # 检查标定文件
    calib_files = list((data_path / 'calib').glob('*.txt'))
    if calib_files:
        print(f"\n✓ 标定文件 ({len(calib_files)} 个):")
        with open(calib_files[0], 'r') as f:
            lines = f.readlines()[:3]
            for line in lines:
                print(f"  {line.strip()}")
    
    # 检查标签文件
    label_files = list((data_path / 'label_2').glob('*.txt'))
    if label_files:
        print(f"\n✓ 标签文件 ({len(label_files)} 个):")
        with open(label_files[0], 'r') as f:
            lines = f.readlines()
            for line in lines:
                print(f"  {line.strip()}")
    
    # 检查点云文件
    velodyne_files = list((data_path / 'velodyne').glob('*.bin'))
    if velodyne_files:
        print(f"\n✓ 点云文件 ({len(velodyne_files)} 个):")
        points = np.fromfile(velodyne_files[0], dtype=np.float32).reshape(-1, 4)
        print(f"  - 点数: {len(points)}")
        print(f"  - X范围: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
        print(f"  - Y范围: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
        print(f"  - Z范围: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
        print(f"  - 强度范围: [{points[:, 3].min():.2f}, {points[:, 3].max():.2f}]")
    
    # 检查图像文件
    image_files = list((data_path / 'image_2').glob('*.png'))
    if image_files:
        print(f"\n✓ 图像文件 ({len(image_files)} 个):")
        try:
            from PIL import Image
            img = Image.open(image_files[0])
            print(f"  - 分辨率: {img.size}")
            print(f"  - 格式: {img.format}")
        except:
            print(f"  - 无法读取图像信息")
    
    print("\n" + "="*60)
    print("数据统计")
    print("="*60)
    
    total_files = sum(results.values())
    print(f"\n总文件数: {total_files}")
    print(f"样本数: {results['calib']}")
    
    if all(results.values()):
        print("\n✓ KITTI示例数据完整！")
        print("\n数据特点:")
        print("  - 5个样本")
        print("  - 每个样本包含:")
        print("    * 标定文件 (calib)")
        print("    * 标签文件 (label_2)")
        print("    * 点云文件 (velodyne)")
        print("    * 图像文件 (image_2)")
        print("\n可用于测试模型的数据加载和推理")
        return True
    else:
        print("\n✗ KITTI示例数据不完整")
        return False

if __name__ == '__main__':
    success = test_kitti_data()
    
    print("\n" + "="*60)
    if success:
        print("✓ 验证完成！数据可用于测试")
    else:
        print("✗ 验证失败")
