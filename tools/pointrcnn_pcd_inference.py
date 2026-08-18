"""
PointRCNN (Seg Head) 推理脚本 — 读取 PCD 点云文件并运行 PointRCNN 检测

用法:
    cd /code/OpenPCDet/tools
    python pointrcnn_pcd_inference.py \
        --pcd_path ../data/new_sheef/pngs/TV_250000000001.pcd \
        --cfg_file cfgs/kitti_models/pointrcnn.yaml \
        --ckpt ../checkpoints/pointrcnn_kitti.pth

PointRCNN Seg Head (PointHeadBox):
    - 第一阶段: 对每个点做 foreground/background 分割 + 3D box proposal
    - 输出 point_cls_scores: 每个点的前景置信度 (0~1)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import yaml

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def read_pcd_raw(pcd_path: str) -> np.ndarray:
    """
    读取 PCD 文件，返回 (N, 4) 数组 [x, y, z, intensity].

    自动检测单位：若坐标量级很大（如 >100），视为 mm 并转为米。
    支持 FIELDS: x y z rgb (packed uint) 或 x y z intensity。
    """
    fields, sizes, types, counts = [], [], [], []
    width, height, num_points = 1, 1, 0
    data_format = 'binary'

    with open(pcd_path, 'rb') as f:
        for _ in range(50):
            line = f.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            parts = line.split()
            if parts[0] == 'FIELDS':
                fields = parts[1:]
            elif parts[0] == 'SIZE':
                sizes = [int(x) for x in parts[1:]]
            elif parts[0] == 'TYPE':
                types = parts[1:]
            elif parts[0] == 'COUNT':
                counts = [int(x) for x in parts[1:]]
            elif parts[0] == 'WIDTH':
                width = int(parts[1])
            elif parts[0] == 'HEIGHT':
                height = int(parts[1])
            elif parts[0] == 'POINTS':
                num_points = int(parts[1])
            elif parts[0] == 'DATA':
                data_format = parts[1]
                break

    if not counts:
        counts = [1] * len(fields)
    if num_points == 0:
        num_points = width * height

    with open(pcd_path, 'rb') as f:
        content = f.read()
        marker = f'DATA {data_format}'.encode()
        idx = content.find(marker)
        raw = content[idx + len(marker) + 1:]

    TYPE_MAP = {'F': (4, np.float32), 'U': (4, np.uint32), 'I': (4, np.int32),
                'f': (4, np.float32), 'u': (4, np.uint32), 'i': (4, np.int32)}
    field_info = []
    offset = 0
    for f, s, t, c in zip(fields, sizes, types, counts):
        elem_size = TYPE_MAP.get(t, (4, np.float32))[0] * c
        field_info.append({'name': f, 'offset': offset, 'size': elem_size,
                          'dtype': TYPE_MAP.get(t, (4, np.float32))[1], 'count': c})
        offset += elem_size
    stride = offset

    # 找 xyz 索引
    x_idx = next((i for i, fi in enumerate(field_info) if fi['name'] == 'x'), 0)
    y_idx = next((i for i, fi in enumerate(field_info) if fi['name'] == 'y'), 1)
    z_idx = next((i for i, fi in enumerate(field_info) if fi['name'] == 'z'), 2)

    result = np.zeros((num_points, 4), dtype=np.float32)

    for i in range(num_points):
        row = raw[i * stride:(i + 1) * stride]
        for axis, fidx, col in [('x', x_idx, 0), ('y', y_idx, 1), ('z', z_idx, 2)]:
            fi = field_info[fidx]
            val = np.frombuffer(row[fi['offset']:fi['offset'] + fi['size']],
                               dtype=fi['dtype'])[0]
            result[i, col] = float(val)

        # intensity 或 rgb
        if 'intensity' in fields:
            ii = fields.index('intensity')
            fi = field_info[ii]
            val = np.frombuffer(row[fi['offset']:fi['offset'] + fi['size']],
                               dtype=fi['dtype'])[0]
            result[i, 3] = float(val)
        elif 'rgb' in fields:
            ri = fields.index('rgb')
            fi = field_info[ri]
            rgb_val = np.frombuffer(row[fi['offset']:fi['offset'] + fi['size']],
                                    dtype=np.uint32)[0]
            r = (rgb_val >> 16) & 0xFF
            result[i, 3] = float(r) / 255.0
        elif 'scalar' in fields:
            si = fields.index('scalar')
            fi = field_info[si]
            val = np.frombuffer(row[fi['offset']:fi['offset'] + fi['size']],
                               dtype=fi['dtype'])[0]
            result[i, 3] = float(val)

    # 自动检测并转换 mm → m
    xyz_mag = np.abs(result[:, :3]).max()
    if xyz_mag > 100:  # 毫米级
        result[:, :3] /= 1000.0
        print(f'  [PCD] Auto-detected mm units (max|xyz|={xyz_mag:.0f}), '
              f'converted to meters')

    return result


class PCDDemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True,
                 root_path=None, logger=None, pcd_path=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names,
            training=training, root_path=root_path, logger=logger
        )
        self.pcd_path = Path(pcd_path)
        self.sample_file_list = [str(self.pcd_path)]

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        points = read_pcd_raw(self.sample_file_list[index])  # (N, 4)
        input_dict = {'points': points, 'frame_id': index}
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def save_seg_ply(xyz, scores, save_path, threshold=0.3):
    """前景(红)/背景(灰) PLY"""
    fg = scores > threshold
    colors = np.zeros((xyz.shape[0], 3), dtype=np.float32)
    colors[fg] = [1.0, 0.0, 0.0]
    colors[~fg] = [0.5, 0.5, 0.5]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(save_path, pcd)
    return int(fg.sum())


def save_boxes_ply(boxes, scores, labels, class_names, save_path):
    """3D 检测框 PLY"""
    verts, edges_list, ecolors = [], [], []
    cls_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]
    box_edges = [[0, 1], [1, 2], [2, 3], [3, 0],
                 [4, 5], [5, 6], [6, 7], [7, 4],
                 [0, 4], [1, 5], [2, 6], [3, 7]]

    for bi, box in enumerate(boxes):
        x, y, z, dx, dy, dz, heading = box
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        c = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * [dx / 2, dy / 2]
        cr = c @ R.T
        bottom = np.hstack([cr, np.full((4, 1), z - dz / 2)])
        top = np.hstack([cr, np.full((4, 1), z + dz / 2)])
        all8 = np.vstack([bottom, top]) + np.array([x, y, 0])
        color = cls_colors[int(labels[bi]) % len(cls_colors)]
        base = len(verts)
        verts.extend(all8.tolist())
        for e in box_edges:
            edges_list.append([base + e[0], base + e[1]])
            ecolors.append(color)

    with open(save_path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element edge {len(edges_list)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for v in verts:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for li, l in enumerate(edges_list):
            r, g, b = ecolors[li]
            f.write(f"{l[0]} {l[1]} {int(r * 255)} {int(g * 255)} {int(b * 255)}\n")


def build_adapted_config(base_cfg_path: str, output_path: str,
                         point_cloud_range: list, num_points: int = -1):
    """
    基于 KITTI PointRCNN 配置，创建适配仓库数据的 config。
    主要修改: point_cloud_range 和 num_points。
    """
    with open(base_cfg_path, 'r') as f:
        base = yaml.safe_load(f)

    # 读取 kitti_dataset.yaml 并修改
    kitti_ds_path = Path(base_cfg_path).parent.parent / 'dataset_configs' / 'kitti_dataset.yaml'
    with open(kitti_ds_path, 'r') as f:
        ds_cfg = yaml.safe_load(f)

    ds_cfg['POINT_CLOUD_RANGE'] = point_cloud_range
    ds_cfg['FOV_POINTS_ONLY'] = False  # 仓库场景不需要 FOV 过滤

    # 写入临时 dataset config
    tmp_ds_path = output_path.replace('.yaml', '_dataset.yaml')
    with open(tmp_ds_path, 'w') as f:
        yaml.dump(ds_cfg, f)

    # 修改模型 config 中的 _BASE_CONFIG_ 和数据处理器
    base['DATA_CONFIG']['_BASE_CONFIG_'] = tmp_ds_path

    # 修改 sample_points 的 NUM_POINTS
    for proc in base['DATA_CONFIG'].get('DATA_PROCESSOR', []):
        if proc.get('NAME') == 'sample_points':
            proc['NUM_POINTS'] = {'train': num_points, 'test': num_points}

    with open(output_path, 'w') as f:
        yaml.dump(base, f)

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description='PointRCNN PCD Inference')
    parser.add_argument('--pcd_path', type=str, required=True,
                        help='PCD 点云文件路径')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pointrcnn.yaml',
                        help='PointRCNN 配置文件')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='预训练权重 .pth')
    parser.add_argument('--output_dir', type=str,
                        default='../output/pointrcnn_pcd_inference')
    parser.add_argument('--score_thresh', type=float, default=0.3,
                        help='Seg Head 前景分数阈值')
    parser.add_argument('--mm_input', action='store_true', default=False,
                        help='强制输入单位为毫米 (默认自动检测)')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger = common_utils.create_logger()
    logger.info('=' * 70)
    logger.info('PointRCNN (Seg Head) PCD Inference')
    logger.info('=' * 70)

    # 1. 先读 PCD，计算合适的 point_cloud_range
    points = read_pcd_raw(args.pcd_path)
    xyz = points[:, :3]
    margin = 1.0  # 边界扩展 1m
    pc_range = [
        float(np.floor(xyz[:, 0].min() - margin)),
        float(np.floor(xyz[:, 1].min() - margin)),
        float(np.floor(xyz[:, 2].min() - margin)),
        float(np.ceil(xyz[:, 0].max() + margin)),
        float(np.ceil(xyz[:, 1].max() + margin)),
        float(np.ceil(xyz[:, 2].max() + margin)),
    ]
    logger.info(f'PCD: {points.shape[0]} points')
    logger.info(f'XYZ range (m): X[{xyz[:, 0].min():.2f}, {xyz[:, 0].max():.2f}] '
                f'Y[{xyz[:, 1].min():.2f}, {xyz[:, 1].max():.2f}] '
                f'Z[{xyz[:, 2].min():.2f}, {xyz[:, 2].max():.2f}]')
    logger.info(f'Point cloud range (with margin): {pc_range}')

    # 2. 创建适配的 config
    adapted_cfg = os.path.join(args.output_dir, 'adapted_pointrcnn.yaml')
    build_adapted_config(args.cfg_file, adapted_cfg, pc_range, num_points=-1)
    logger.info(f'Adapted config: {adapted_cfg}')

    # 3. 加载适配后的配置
    cfg_from_yaml_file(adapted_cfg, cfg)
    logger.info(f'Classes: {cfg.CLASS_NAMES}')

    # 4. 构建数据集
    demo_dataset = PCDDemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        training=False, root_path=Path(args.pcd_path).parent,
        logger=logger, pcd_path=args.pcd_path,
    )
    logger.info(f'Loaded {len(demo_dataset)} sample(s)')

    # 5. 构建模型
    logger.info('Building PointRCNN model...')
    model = build_network(
        model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset
    )
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    logger.info('Model loaded.')

    # 6. 推理
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            fname = Path(args.pcd_path).stem
            logger.info(f'Running on: {Path(args.pcd_path).name}')

            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)

            pred_dicts, recall_dicts = model.forward(data_dict)

            # 拿到处理后的点云坐标
            proc_points = data_dict['points'].cpu().numpy()
            proc_xyz = proc_points[:, 1:4] if proc_points.shape[1] >= 4 else proc_points[:, :3]

            # Seg Head 输出 (point_cls_scores)
            seg_scores = None
            if 'point_cls_scores' in data_dict:
                seg_scores = data_dict['point_cls_scores'].cpu().numpy()
                logger.info(f'Seg Head scores: [{seg_scores.min():.4f}, {seg_scores.max():.4f}] '
                            f'mean={seg_scores.mean():.4f}')

            # 最终检测框
            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            pred_scores_np = pred_dicts[0]['pred_scores'].cpu().numpy()
            pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy().astype(int)

            logger.info(f'Final detections (after NMS): {len(pred_boxes)}')
            for bi in range(min(len(pred_boxes), 20)):
                cls_name = cfg.CLASS_NAMES[pred_labels[bi]]
                logger.info(f'  [{bi}] {cls_name}: score={pred_scores_np[bi]:.4f} '
                            f'box=({pred_boxes[bi][0]:.2f}, {pred_boxes[bi][1]:.2f}, '
                            f'{pred_boxes[bi][2]:.2f}, {pred_boxes[bi][3]:.2f}x'
                            f'{pred_boxes[bi][4]:.2f}x{pred_boxes[bi][5]:.2f})')

            # 保存
            if seg_scores is not None:
                seg_path = os.path.join(args.output_dir, f'{fname}_seg.ply')
                n_fg = save_seg_ply(proc_xyz, seg_scores, seg_path, args.score_thresh)
                logger.info(f'  -> {seg_path} (fg={n_fg}/{len(proc_xyz)})')

            if len(pred_boxes) > 0:
                boxes_path = os.path.join(args.output_dir, f'{fname}_boxes.ply')
                save_boxes_ply(pred_boxes, pred_scores_np, pred_labels,
                              cfg.CLASS_NAMES, boxes_path)
                logger.info(f'  -> {boxes_path}')

            raw_pcd = o3d.geometry.PointCloud()
            raw_pcd.points = o3d.utility.Vector3dVector(proc_xyz)
            raw_path = os.path.join(args.output_dir, f'{fname}_raw.ply')
            o3d.io.write_point_cloud(raw_path, raw_pcd)
            logger.info(f'  -> {raw_path}')

            logger.info('=' * 70)
            logger.info(f'Summary: {len(proc_xyz)} pts | '
                        f'Seg fg: {n_fg if seg_scores is not None else "N/A"} | '
                        f'Boxes: {len(pred_boxes)}')
            logger.info('=' * 70)

    logger.info('Done!')


if __name__ == '__main__':
    main()
