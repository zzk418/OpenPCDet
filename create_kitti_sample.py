#!/usr/bin/env python3
"""
创建KITTI格式的示例数据用于测试
"""

import os
import numpy as np
from pathlib import Path

def create_kitti_sample_data():
    """创建KITTI格式的示例数据"""
    
    # 创建目录结构
    base_dir = Path('data/kitti_sample')
    for split in ['training', 'testing']:
        for subdir in ['calib', 'image_2', 'label_2', 'velodyne']:
            (base_dir / split / subdir).mkdir(parents=True, exist_ok=True)
    
    print("创建KITTI示例数据...")
    
    # 创建5个样本
    for idx in range(5):
        sample_id = f'{idx:06d}'
        
        # 1. 创建标定文件 (calib)
        calib_content = f"""P0: 7.070493e+02 0.000000e+00 6.040814e+02 0.000000e+00 0.000000e+00 7.070493e+02 1.805066e+02 -3.797839e+01 0.000000e+00 0.000000e+00 1.000000e+00 -2.717806e-03
P1: 7.070493e+02 0.000000e+00 6.040814e+02 4.575831e+01 0.000000e+00 7.070493e+02 1.805066e+02 -3.885028e+01 0.000000e+00 0.000000e+00 1.000000e+00 4.745801e-03
P2: 7.070493e+02 0.000000e+00 6.040814e+02 4.687143e+01 0.000000e+00 7.070493e+02 1.805066e+02 -3.884477e+01 0.000000e+00 0.000000e+00 1.000000e+00 1.578479e-02
P3: 7.070493e+02 0.000000e+00 6.040814e+02 -3.289347e+01 0.000000e+00 7.070493e+02 1.805066e+02 2.423857e+00 0.000000e+00 0.000000e+00 1.000000e+00 9.750408e-03
R0_rect: 9.999239e-01 9.837760e-03 -7.445717e-03 -9.869795e-03 9.999421e-01 -4.278459e-03 7.402527e-03 4.351614e-03 9.999631e-01
Tr_velo_to_cam: 7.533745e-03 -9.999714e-01 -6.166020e-04 -4.069766e-03 1.480249e-02 7.280733e-04 -9.998902e-01 -7.631618e-02 9.998621e-01 7.523790e-03 1.480755e-02 -2.717806e-01
Tr_imu_to_velo: 9.999976e-01 7.553071e-04 -2.035826e-03 -8.086759e-01 -7.854027e-04 9.998898e-01 -1.482298e-02 3.195559e-01 2.024406e-03 1.482454e-02 9.998881e-01 -7.437787e-01
"""
        with open(base_dir / 'training' / 'calib' / f'{sample_id}.txt', 'w') as f:
            f.write(calib_content)
        
        # 2. 创建标签文件 (label_2)
        label_content = f"""Car 0.00 0 -1.57 100 50 200 150 1.5 2.0 4.0 5.0 1.0 2.0 0.5 -1.57 1.0
Pedestrian 0.00 0 -1.57 250 100 300 200 1.7 0.6 0.8 10.0 1.0 2.0 0.5 -1.57 1.0
"""
        with open(base_dir / 'training' / 'label_2' / f'{sample_id}.txt', 'w') as f:
            f.write(label_content)
        
        # 3. 创建点云文件 (velodyne)
        num_points = 100000
        points = np.random.randn(num_points, 4).astype(np.float32)
        points[:, 0] = points[:, 0] * 10 + 5
        points[:, 1] = points[:, 1] * 10
        points[:, 2] = points[:, 2] * 5 + 1
        points[:, 3] = np.random.rand(num_points) * 255
        
        points.tofile(base_dir / 'training' / 'velodyne' / f'{sample_id}.bin')
        
        # 4. 创建图像文件 (image_2)
        try:
            from PIL import Image
            img = Image.new('RGB', (1242, 375), color=(73, 109, 137))
            img.save(base_dir / 'training' / 'image_2' / f'{sample_id}.png')
        except:
            pass
        
        print(f"✓ 创建样本 {sample_id}")
    
    print(f"\n✓ KITTI示例数据已创建在: {base_dir}")
    return str(base_dir)

if __name__ == '__main__':
    create_kitti_sample_data()