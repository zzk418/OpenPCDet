#!/usr/bin/env python3
"""
仓库 LiDAR 数据集 → OpenPCDet CustomDataset 格式 (含坐标归一化)
从 samples.json 提取标注 → points/*.npy + labels/*.txt + infos
归一化策略：将坐标分布对齐到 KITTI 典型范围，以匹配预训练参数
"""
import os, sys, json, pickle
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).parent
WAREHOUSE_DIR = ROOT / 'warehouse_data'
OUTPUT_DIR = ROOT / 'data' / 'warehouse'
LABEL_TRANSLATION = {
    'Box': '箱子', 'ELFplusplus': '电动运输车',
    'CargoBike': '货运自行车', 'FTS': '无人搬运车', 'ForkLift': '叉车',
}
CLASS_NAMES = ['箱子', '电动运输车', '货运自行车', '无人搬运车', '叉车']
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'points').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'labels').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'ImageSets').mkdir(parents=True, exist_ok=True)


def compute_normalization_params(pcd_files):
    """统计所有点云, 计算归一化参数"""
    all_pts = []
    for f in tqdm(pcd_files, desc='Computing stats'):
        pcd = o3d.io.read_point_cloud(str(f))
        pts = np.asarray(pcd.points, dtype=np.float32)
        cols = np.asarray(pcd.colors, dtype=np.float32)
        intensity = cols[:, 0].reshape(-1, 1)
        all_pts.append(np.hstack([pts, intensity]))
    all_pts = np.vstack(all_pts)
    
    params = {
        'x_min': float(all_pts[:,0].min()),
        'x_max': float(all_pts[:,0].max()),
        'y_mean': float(all_pts[:,1].mean()),
        'z_min': float(all_pts[:,2].min()),
        'z_max': float(all_pts[:,2].max()),
        'intensity_mean': float(all_pts[:,3].mean()),
    }
    
    # KITTI 目标范围
    target_x_range = 65.0       # X 映射到 [0, 65]
    target_z_range = [-3.0, 1.0]  # Z 映射到 KITTI Z range
    target_intensity_mean = 0.3
    
    scale_x = target_x_range / (params['x_max'] - params['x_min'])
    scale_z = (target_z_range[1] - target_z_range[0]) / (params['z_max'] - params['z_min'])
    scale_i = target_intensity_mean / max(params['intensity_mean'], 1e-6)
    
    params['scale_x'] = scale_x
    params['shift_x'] = -params['x_min'] * scale_x
    params['shift_y'] = -params['y_mean']
    params['scale_z'] = scale_z
    params['shift_z'] = target_z_range[0] - params['z_min'] * scale_z
    params['scale_i'] = scale_i
    
    print(f'\n归一化参数 (使坐标对齐KITTI分布):')
    print(f'  X: ({params["x_min"]:.1f}~{params["x_max"]:.1f}) → [0, {target_x_range}]  scale={scale_x:.4f} shift={params["shift_x"]:.1f}')
    print(f'  Y: mean {params["y_mean"]:.2f} → 0  shift={params["shift_y"]:.2f}')
    print(f'  Z: ({params["z_min"]:.2f}~{params["z_max"]:.2f}) → {target_z_range}  scale={scale_z:.4f} shift={params["shift_z"]:.2f}')
    print(f'  I: mean {params["intensity_mean"]:.3f} → {target_intensity_mean}  scale={scale_i:.4f}')
    return params


def normalize_points(pts, params):
    """对点云坐标和 intensity 做归一化, 并标注框也做归一化"""
    x = pts[:, 0] * params['scale_x'] + params['shift_x']
    y = pts[:, 1] + params['shift_y']
    z = pts[:, 2] * params['scale_z'] + params['shift_z']
    i = pts[:, 3] * params['scale_i']
    return np.column_stack([x, y, z, i])


def normalize_boxes(boxes, params):
    """标注框: [x, y, z, l, w, h, yaw], 归一化后返回"""
    if len(boxes) == 0:
        return boxes
    nb = boxes.copy()
    nb[:, 0] = nb[:, 0] * params['scale_x'] + params['shift_x']
    nb[:, 1] = nb[:, 1] + params['shift_y']
    nb[:, 2] = nb[:, 2] * params['scale_z'] + params['shift_z']
    # 尺寸也需要缩放
    nb[:, 3] *= params['scale_x']  # l (X方向尺寸)
    # 宽度按Y缩放?... Y只做了平移没缩放
    # 高度缩放
    nb[:, 5] *= params['scale_z']  # h
    # yaw 不变
    return nb


def load_annotations():
    """加载 samples.json 中的标注"""
    with open(WAREHOUSE_DIR / 'samples.json') as f:
        data = json.load(f)
    annos = {}
    for sample in data['samples']:
        fp = sample.get('filepath', '')
        if not fp.endswith('.pcd'):
            continue
        sid = sample.get('scan_id', Path(fp).stem)
        detections = []
        gt = sample.get('ground_truth', {})
        for det in gt.get('detections', []):
            label_en = det.get('label', '')
            label = LABEL_TRANSLATION.get(label_en, '')
            if not label or label not in CLASS_MAP:
                continue
            loc = det.get('location', [0,0,0])
            dims = det.get('dimensions', [1,1,1])
            rot = det.get('rotation', [0,0,0])
            detection = {
                'name': label,
                'x': loc[0], 'y': loc[1], 'z': loc[2],
                'l': dims[1], 'w': dims[0], 'h': dims[2],
                'yaw': rot[2],
            }
            detections.append(detection)
        annos[sid] = detections
    return annos


def convert(limit=None):
    annos = load_annotations()
    pcd_files = sorted((WAREHOUSE_DIR / 'data').glob('*.pcd'))
    if limit:
        pcd_files = pcd_files[:limit]
    
    # 先计算归一化参数
    params = compute_normalization_params(pcd_files)
    
    all_infos = []
    for pcd_file in tqdm(pcd_files, desc='Converting'):
        sid = pcd_file.stem
        pcd = o3d.io.read_point_cloud(str(pcd_file))
        pts = np.asarray(pcd.points, dtype=np.float32)
        cols = np.asarray(pcd.colors, dtype=np.float32)
        intensity = cols[:, 0].reshape(-1, 1)
        points = np.hstack([pts, intensity])
        
        # 归一化
        points = normalize_points(points, params)
        
        # 保存
        np.save(OUTPUT_DIR / 'points' / f'{sid}.npy', points)
        
        # 获取标注
        dets = annos.get(sid, [])
        gt_boxes_list = []
        gt_names_list = []
        with open(OUTPUT_DIR / 'labels' / f'{sid}.txt', 'w') as f:
            for det in dets:
                # 先归一化坐标
                x = det['x'] * params['scale_x'] + params['shift_x']
                y = det['y'] + params['shift_y']
                z = det['z'] * params['scale_z'] + params['shift_z']
                l = det['l'] * params['scale_x']
                w = det['w']  # Y方向未缩放
                h = det['h'] * params['scale_z']
                yaw = det['yaw']
                f.write(f'{x:.6f} {y:.6f} {z:.6f} {l:.6f} {w:.6f} {h:.6f} {yaw:.6f} {det["name"]}\n')
                gt_boxes_list.append([x, y, z, l, w, h, yaw])
                gt_names_list.append(det['name'])
        
        gt_boxes = np.array(gt_boxes_list, dtype=np.float32)
        gt_names = np.array(gt_names_list)
        
        info = {
            'point_cloud': {'num_features': 4, 'lidar_idx': sid},
            'annos': {
                'name': gt_names,
                'gt_boxes_lidar': gt_boxes[:, :7] if len(gt_boxes) > 0 else np.zeros((0, 7), dtype=np.float32),
                'location': gt_boxes[:, :3] if len(gt_boxes) > 0 else np.zeros((0, 3)),
                'dimensions': gt_boxes[:, 3:6] if len(gt_boxes) > 0 else np.zeros((0, 3)),
                'rotation_y': gt_boxes[:, 6] if len(gt_boxes) > 0 else np.zeros(0),
                'difficulty': np.zeros(len(gt_boxes), dtype=np.int32),
                'truncated': np.zeros(len(gt_boxes), dtype=np.float32),
                'occluded': np.zeros(len(gt_boxes), dtype=np.int32),
                'alpha': np.zeros(len(gt_boxes), dtype=np.float32),
                'bbox': gt_boxes[:, :7] if len(gt_boxes) > 0 else np.zeros((0, 7)),
            },
            'num_points': len(points),
        }
        all_infos.append(info)
    
    split_idx = int(len(all_infos) * 0.8)
    train_infos = all_infos[:split_idx]
    val_infos = all_infos[split_idx:]
    
    # 统计
    total_boxes = sum(len(info['annos']['name']) for info in all_infos)
    class_counts = {}
    for info in all_infos:
        for name in info['annos']['name']:
            class_counts[name] = class_counts.get(name, 0) + 1
    
    print(f"\n{'='*60}")
    print(f"Statistics:")
    print(f"  Total frames: {len(all_infos)}")
    print(f"  Train: {len(train_infos)}, Val: {len(val_infos)}")
    print(f"  Total boxes: {total_boxes}")
    print(f"  Classes: {dict(sorted(class_counts.items(), key=lambda x:-x[1]))}")
    
    # 验证归一化后范围
    sample_pts = np.load(OUTPUT_DIR / 'points' / f'{pcd_files[0].stem}.npy')
    print(f"\n归一化后点云范围:")
    for i, name in enumerate(['X','Y','Z']):
        print(f"  {name}: [{sample_pts[:,i].min():.2f}, {sample_pts[:,i].max():.2f}]"
              f" mean={sample_pts[:,i].mean():.2f} std={sample_pts[:,i].std():.2f}")
    print(f"  Intensity: mean={sample_pts[:,3].mean():.3f} std={sample_pts[:,3].std():.3f}")
    
    # 保存
    with open(OUTPUT_DIR / 'warehouse_infos_train.pkl', 'wb') as f:
        pickle.dump(train_infos, f)
    with open(OUTPUT_DIR / 'warehouse_infos_val.pkl', 'wb') as f:
        pickle.dump(val_infos, f)
    with open(OUTPUT_DIR / 'ImageSets' / 'train.txt', 'w') as f:
        for info in train_infos:
            f.write(info['point_cloud']['lidar_idx'] + '\n')
    with open(OUTPUT_DIR / 'ImageSets' / 'val.txt', 'w') as f:
        for info in val_infos:
            f.write(info['point_cloud']['lidar_idx'] + '\n')
    
    print(f"\n输出: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    convert(limit=args.limit)