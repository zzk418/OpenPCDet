#!/usr/bin/env python3
"""
仓库 LiDAR → OpenPCDet 格式 (NuScenes 分布对齐)
归一化到 NuScenes 范围 [-51.2, 51.2, -5.0, 51.2, 51.2, 3.0]
5维特征: x, y, z, intensity, timestamp (timestamp=0)
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

# 清理旧数据
import shutil
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True)
(OUTPUT_DIR / 'points').mkdir()
(OUTPUT_DIR / 'labels').mkdir()
(OUTPUT_DIR / 'ImageSets').mkdir()


def compute_nuscenes_params(pcd_files):
    """统计原始数据, 计算映射到 NuScenes 范围的参数"""
    all_pts = []
    for f in tqdm(pcd_files, desc='Computing stats'):
        pcd = o3d.io.read_point_cloud(str(f))
        pts = np.asarray(pcd.points, dtype=np.float64)
        cols = np.asarray(pcd.colors, dtype=np.float64)
        intensity = cols[:, 0].reshape(-1, 1)
        all_pts.append(np.hstack([pts, intensity]))
    all_pts = np.vstack(all_pts)

    # 原始统计
    raw = {
        'x_mean': float(all_pts[:, 0].mean()),
        'y_mean': float(all_pts[:, 1].mean()),
        'x_absmax': float(max(abs(all_pts[:, 0].min()), abs(all_pts[:, 0].max()))),
        'y_absmax': float(max(abs(all_pts[:, 1].min()), abs(all_pts[:, 1].max()))),
        'z_min': float(all_pts[:, 2].min()),
        'z_max': float(all_pts[:, 2].max()),
    }

    # NuScenes 目标:
    #   X: [-51.2, 51.2]  →  center at 0
    #   Y: [-51.2, 51.2]  →  center at 0
    #   Z: [-5.0, 3.0]
    target_x_range = 51.2
    target_y_range = 51.2
    target_z_range = [-5.0, 3.0]

    # X: 去中心化 + 缩放到 [-51.2, 51.2]
    scale_x = target_x_range / raw['x_absmax'] if raw['x_absmax'] > target_x_range else 1.0
    shift_x = -raw['x_mean'] * scale_x

    # Y: 同样的处理
    scale_y = target_y_range / raw['y_absmax'] if raw['y_absmax'] > target_y_range else 1.0
    shift_y = -raw['y_mean'] * scale_y

    # Z: 线性映射
    z_range_orig = raw['z_max'] - raw['z_min']
    z_range_target = target_z_range[1] - target_z_range[0]  # 8.0
    scale_z = z_range_target / z_range_orig if z_range_orig > 0 else 1.0
    shift_z = target_z_range[0] - raw['z_min'] * scale_z

    # Intensity: 归一化到 [0, 1]
    i_mean = float(all_pts[:, 3].mean())
    scale_i = 0.5 / max(i_mean, 1e-6)

    params = {
        'shift_x': shift_x, 'scale_x': scale_x,
        'shift_y': shift_y, 'scale_y': scale_y,
        'shift_z': shift_z, 'scale_z': scale_z,
        'scale_i': scale_i,
    }

    print(f'\nNuScenes 对齐参数:')
    print(f'  X: mean {raw["x_mean"]:.1f} → 0, absmax {raw["x_absmax"]:.1f} → {target_x_range}')
    print(f'     scale={scale_x:.4f} shift={shift_x:.1f}')
    print(f'  Y: mean {raw["y_mean"]:.1f} → 0, absmax {raw["y_absmax"]:.1f} → {target_y_range}')
    print(f'     scale={scale_y:.4f} shift={shift_y:.1f}')
    print(f'  Z: [{raw["z_min"]:.1f}, {raw["z_max"]:.1f}] → {target_z_range}')
    print(f'     scale={scale_z:.4f} shift={shift_z:.1f}')
    print(f'  I: mean {i_mean:.3f} scale={scale_i:.4f}')
    return params


def transform_points(pts, p):
    """对齐到 NuScenes 分布, 返回 5 维 (x, y, z, intensity, timestamp=0)"""
    x = pts[:, 0] * p['scale_x'] + p['shift_x']
    y = pts[:, 1] * p['scale_y'] + p['shift_y']
    z = pts[:, 2] * p['scale_z'] + p['shift_z']
    i = pts[:, 3] * p['scale_i']
    ts = np.zeros_like(x)  # timestamp = 0
    return np.column_stack([x, y, z, i, ts]).astype(np.float32)


def transform_box(box, p):
    """标注框同样变换"""
    x, y, z, l, w, h, yaw = box
    x = x * p['scale_x'] + p['shift_x']
    y = y * p['scale_y'] + p['shift_y']
    z = z * p['scale_z'] + p['shift_z']
    l = l * p['scale_x']   # 长度按X缩放
    w = w * p['scale_y']   # 宽度按Y缩放
    h = h * p['scale_z']   # 高度按Z缩放
    return [x, y, z, l, w, h, yaw]


def load_annotations():
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
            loc = det.get('location', [0, 0, 0])
            dims = det.get('dimensions', [1, 1, 1])
            rot = det.get('rotation', [0, 0, 0])
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

    params = compute_nuscenes_params(pcd_files)

    all_infos = []
    for pcd_file in tqdm(pcd_files, desc='Converting'):
        sid = pcd_file.stem
        pcd = o3d.io.read_point_cloud(str(pcd_file))
        pts = np.asarray(pcd.points, dtype=np.float32)
        cols = np.asarray(pcd.colors, dtype=np.float32)
        intensity = cols[:, 0].reshape(-1, 1)
        raw_points = np.hstack([pts, intensity])

        points = transform_points(raw_points, params)
        np.save(OUTPUT_DIR / 'points' / f'{sid}.npy', points)

        dets = annos.get(sid, [])
        gt_boxes_list = []
        gt_names_list = []
        with open(OUTPUT_DIR / 'labels' / f'{sid}.txt', 'w') as f:
            for det in dets:
                raw_box = [det['x'], det['y'], det['z'],
                           det['l'], det['w'], det['h'], det['yaw']]
                x, y, z, l, w, h, yaw = transform_box(raw_box, params)
                f.write(f'{x:.6f} {y:.6f} {z:.6f} {l:.6f} {w:.6f} {h:.6f} {yaw:.6f} {det["name"]}\n')
                gt_boxes_list.append([x, y, z, l, w, h, yaw])
                gt_names_list.append(det['name'])

        gt_boxes = np.array(gt_boxes_list, dtype=np.float32)
        gt_names = np.array(gt_names_list)

        info = {
            'point_cloud': {'num_features': 5, 'lidar_idx': sid},
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

    total_boxes = sum(len(info['annos']['name']) for info in all_infos)
    class_counts = {}
    for info in all_infos:
        for name in info['annos']['name']:
            class_counts[name] = class_counts.get(name, 0) + 1

    print(f"\n{'='*60}")
    print(f"NuScenes 对齐统计:")
    print(f"  Total frames: {len(all_infos)}")
    print(f"  Train: {len(train_infos)}, Val: {len(val_infos)}")
    print(f"  Total boxes: {total_boxes}")
    print(f"  Classes: {dict(sorted(class_counts.items(), key=lambda x: -x[1]))}")

    # 验证
    sample_pts = np.load(OUTPUT_DIR / 'points' / f'{pcd_files[0].stem}.npy')
    print(f"\n对齐后点云范围:")
    for i, name in enumerate(['X', 'Y', 'Z', 'I', 'TS']):
        print(f"  {name}: [{sample_pts[:, i].min():.2f}, {sample_pts[:, i].max():.2f}]"
              f" mean={sample_pts[:, i].mean():.2f}")

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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    convert(limit=args.limit)