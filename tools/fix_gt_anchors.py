#!/usr/bin/env python3
"""用修正后的 Eagle-M4 Mega 内参重算标注 JSON 中的 anchor_3d。

背景: 旧标注工具的 _get_anchor_3d 默认参数是 (cx=320, cy=240) —— 错误。
正确的内参是 fx=fy=410.9, cx=307.0, cy=264.3 (fy/cy 由 PCD 射线网格实测,
fx 取方形像素先验; 权威出厂值可用 tools/query_camera_intrinsics.py 从相机读取)。

像素点击 (pixel_uv) 保持不变, 只重算 3D 反投影值。

用法:
    python tools/fix_gt_anchors.py --data_dir data/new_sheef/prototypes \
        --annot_dir data/new_sheef/prototype_annotations

原文件备份为 *.json.bak (首次运行时)。
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

FX, FY, CX, CY = 410.9, 410.9, 307.0, 264.3
IMG_W, IMG_H = 640, 480


def read_pcd(path):
    with open(path, 'rb') as f:
        while True:
            if f.readline().decode().strip() == 'DATA binary':
                break
        dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('rgb', 'u4')])
        data = np.frombuffer(f.read(), dtype=dtype)
    return np.column_stack([data['x'], data['y'], data['z']]).astype(np.float32)


def build_depth_map(xyz):
    valid = xyz[:, 2] > 1.0
    u = np.round(FX * xyz[valid, 0] / xyz[valid, 2] + CX).astype(np.int32)
    v = np.round(FY * xyz[valid, 1] / xyz[valid, 2] + CY).astype(np.int32)
    m = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    dm = np.full((IMG_H, IMG_W), np.nan, dtype=np.float32)
    u, v, z = u[m], v[m], xyz[valid, 2][m]
    for i in range(len(u)):
        if np.isnan(dm[v[i], u[i]]) or z[i] < dm[v[i], u[i]]:
            dm[v[i], u[i]] = z[i]
    return dm


def get_anchor_3d(u0, v0, depth_map, window=5):
    half = window // 2
    wd = depth_map[max(0, v0 - half):v0 + half + 1, max(0, u0 - half):u0 + half + 1]
    vd = wd[~np.isnan(wd)]
    if len(vd) == 0:
        # 扩大窗口重试
        for expand in range(half + 1, min(IMG_W, IMG_H) // 2, 5):
            wd2 = depth_map[max(0, v0 - expand):v0 + expand + 1,
                             max(0, u0 - expand):u0 + expand + 1]
            vd2 = wd2[~np.isnan(wd2)]
            if len(vd2) > 0:
                vd = vd2
                break
    if len(vd) == 0:
        return None, 0
    z0 = float(np.median(vd))
    x0 = (u0 - CX) * z0 / FX
    y0 = (v0 - CY) * z0 / FY
    return [round(x0, 1), round(y0, 1), round(z0, 1)], len(vd)


def main():
    global FX, FY, CX, CY
    parser = argparse.ArgumentParser(description='重算货架锚点标注的 3D 值 (修正内参)')
    parser.add_argument('--data_dir', action='append', default=[],
                        help='PCD 目录, 可多次指定 (默认 pngs + prototypes)')
    parser.add_argument('--annot_dir', default='data/new_sheef/prototype_annotations')
    parser.add_argument('--fx', type=float, default=FX)
    parser.add_argument('--fy', type=float, default=FY)
    parser.add_argument('--cx', type=float, default=CX)
    parser.add_argument('--cy', type=float, default=CY)
    args = parser.parse_args()

    FX, FY, CX, CY = args.fx, args.fy, args.cx, args.cy

    data_dirs = args.data_dir or ['data/new_sheef/pngs', 'data/new_sheef/prototypes']
    pcd_map = {}
    for d in data_dirs:
        for p in glob.glob(os.path.join(d, '*.pcd')):
            pcd_map[os.path.basename(p)[:-4]] = p
    print(f'PCD: {len(pcd_map)} 帧, 内参 fx={FX} fy={FY} cx={CX} cy={CY}')

    def find_pcd(stem):
        if stem in pcd_map:
            return pcd_map[stem]
        # 文件名带 clusterN_ 前缀的模糊匹配
        for name, path in pcd_map.items():
            if name.endswith('_' + stem) or name.endswith(stem):
                return path
        return None

    jsons = sorted(glob.glob(os.path.join(args.annot_dir, '*_anchor_v2.json')))
    n_ok = n_pcd_miss = n_depth_miss = 0
    for jp in jsons:
        with open(jp, encoding='utf-8') as f:
            ann = json.load(f)
        stem = ann.get('stem') or os.path.basename(jp).replace('_anchor_v2.json', '')
        pcd = find_pcd(stem)
        if pcd is None:
            print(f'  {stem}: 无对应 PCD, 跳过')
            n_pcd_miss += 1
            continue
        xyz = read_pcd(pcd)
        dm = build_depth_map(xyz)
        changed = False
        for k in ann.get('keypoints', []):
            u0, v0 = k['pixel_uv']
            new3d, n = get_anchor_3d(int(u0), int(v0), dm)
            if new3d is None:
                print(f'  {stem} ({u0},{v0}): 无深度, 保留原值')
                n_depth_miss += 1
                continue
            old3d = k.get('anchor_3d')
            if old3d != new3d:
                k['anchor_3d'] = new3d
                k['median_depth'] = new3d[2]
                k['valid_count'] = n
                k['intrinsics'] = {'fx': FX, 'fy': FY, 'cx': CX, 'cy': CY}
                changed = True
        if changed:
            bak = jp + '.bak'
            if not os.path.exists(bak):
                shutil.copy2(jp, bak)
            with open(jp, 'w', encoding='utf-8') as f:
                json.dump(ann, f, ensure_ascii=False, indent=2)
            print(f'  {stem}: 已更新 (备份 {os.path.basename(bak)})')
            n_ok += 1
        else:
            print(f'  {stem}: 无变化')
    print(f'\n完成: 更新 {n_ok}, 无 PCD {n_pcd_miss}, 无深度 {n_depth_miss}, 共 {len(jsons)} 个标注')


if __name__ == '__main__':
    main()
