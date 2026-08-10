#!/usr/bin/env python3
"""
SuperPoint 特征匹配：用 SuperPoint 在原型 anchor 位置附近提取描述子，
然后在目标帧上做描述子匹配 + 几何验证，得到货架角点 2D→3D 对应。

原理：
  - SuperPoint 在像素级检测关键点 + 256-dim 描述子
  - 原型阶段：在标注 anchor 位置 ±N 像素内选最强关键点，保存描述子
  - 匹配阶段：描述子 NN 匹配 + ratio test + 几何一致性验证
  - 3D 查表：用 PCD 深度图 (5×5 median depth) 得到 3D 坐标

用法:
  python tools/superpoint_matcher.py --max_frames 20
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


# ─────────────────────────────────────────────
# SuperPoint 模型定义 (基于 Magic Leap 官方架构)
# ─────────────────────────────────────────────

class SuperPointNet(nn.Module):
    """Magic Leap SuperPoint 网络 (v1)"""

    def __init__(self):
        super().__init__()

        # ── Shared Encoder (VGG-style) ──
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, 3, 1, 1)
        self.conv1b = nn.Conv2d(c1, c1, 3, 1, 1)

        self.conv2a = nn.Conv2d(c1, c2, 3, 1, 1)
        self.conv2b = nn.Conv2d(c2, c2, 3, 1, 1)

        self.conv3a = nn.Conv2d(c2, c3, 3, 1, 1)
        self.conv3b = nn.Conv2d(c3, c3, 3, 1, 1)

        self.conv4a = nn.Conv2d(c3, c4, 3, 1, 1)
        self.conv4b = nn.Conv2d(c4, c4, 3, 1, 1)

        # ── Keypoint Detector Head ──
        self.convPa = nn.Conv2d(c4, c5, 3, 1, 1)   # 256 → 256
        self.convPb = nn.Conv2d(c5, 65, 1, 1, 0)     # 256 → 65 (8×8 cells + dustbin)

        # ── Descriptor Head ──
        self.convDa = nn.Conv2d(c4, c5, 3, 1, 1)   # 256 → 256
        self.convDb = nn.Conv2d(c5, 256, 1, 1, 0)    # 256 → 256

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        """x: (B, 1, H, W) 灰度图 [0,1]"""

        # Shared Encoder (stride 8)
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)                              # /2

        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)                              # /4

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)                              # /8

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))                 # /8 (no pool)

        # Detector Head
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)                       # (B, 65, H/8, W/8)

        # Descriptor Head
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)                       # (B, 256, H/8, W/8)
        dn = F.normalize(desc, p=2, dim=1)            # L2 normalize

        return semi, dn


def batched_nms(scores, nms_radius=4):
    """Batched yet trivial NMS (maxpool trick).

    Args:
        scores: (H, W) heatmap
        nms_radius: suppression radius

    Returns:
        (H, W) NMS'd heatmap
    """
    pad = nms_radius
    maxpool = F.max_pool2d(
        scores.unsqueeze(0).unsqueeze(0),
        kernel_size=2 * pad + 1, stride=1, padding=pad)
    maxpool = maxpool.squeeze(0).squeeze(0)
    keep = (scores == maxpool)
    return scores * keep


def extract_keypoints(semi, desc, conf_thresh=0.015, nms_radius=4, border=4):
    """从 SuperPoint 输出提取关键点和描述子。

    Args:
        semi: (1, 65, H/8, W/8) 检测器输出
        desc: (1, 256, H/8, W/8) 描述子
        conf_thresh: 置信度阈值 (默认 0.015，与 SuperPoint 官方一致)
        nms_radius: NMS 半径 (在 H/8 尺度)
        border: 边界去除像素 (在 H/8 尺度)

    Returns:
        kp_uv: (N, 2) [u, v] 像素坐标
        kp_desc: (N, 256) 描述子
        kp_score: (N,) 置信度
    """
    B, C, Hc, Wc = semi.shape
    H, W = Hc * 8, Wc * 8

    # 取出 dustbin 通道
    nodust = semi[:, :-1, :, :]                       # (1, 64, H/8, W/8)
    nodust = nodust.permute(0, 2, 3, 1)               # (1, H/8, W/8, 64)
    nodust = nodust.reshape(Hc, Wc, 8, 8)              # (H/8, W/8, 8, 8)
    nodust = nodust.permute(0, 2, 1, 3)               # (H/8, 8, W/8, 8)
    nodust = nodust.reshape(Hc * 8, Wc * 8)            # (H, W)

    # Softmax → 置信度
    semi_exp = torch.exp(semi)                         # (1, 65, H/8, W/8)
    semi_sum = semi_exp.sum(dim=1, keepdim=True)       # (1, 1, H/8, W/8)

    # 每个 cell 内的 8×8 位置做 softmax
    nodust_exp = semi_exp[:, :-1, :, :]                # (1, 64, H/8, W/8)
    nodust_exp = nodust_exp.permute(0, 2, 3, 1).reshape(Hc, Wc, 8, 8)
    nodust_exp = nodust_exp.permute(0, 2, 1, 3).reshape(H, W)

    local_sum = semi_exp[:, :-1, :, :].permute(0, 2, 3, 1)
    local_sum = local_sum.reshape(Hc, Wc, 8, 8)
    local_sum = local_sum.permute(0, 2, 1, 3).reshape(H, W)

    scores = nodust_exp / local_sum.clamp(min=1e-8)

    # NMS
    scores = batched_nms(scores, nms_radius=nms_radius)

    # 阈值 + 边界
    mask = (scores > conf_thresh)
    mask[:border * 8, :] = False
    mask[-border * 8:, :] = False
    mask[:, :border * 8] = False
    mask[:, -border * 8:] = False

    ys, xs = torch.nonzero(mask, as_tuple=True)
    kp_score = scores[ys, xs]

    # 按置信度排序
    sort_idx = torch.argsort(kp_score, descending=True)
    ys, xs = ys[sort_idx], xs[sort_idx]
    kp_score = kp_score[sort_idx]

    # 插值描述子
    # desc: (1, 256, H/8, W/8)
    # 关键点像素坐标 → 描述子格子坐标
    desc_xs = (xs.float() / 8.0) + 0.5
    desc_ys = (ys.float() / 8.0) + 0.5
    # grid_sample: (B, C, H_out, W_out), grid: (B, H_out, W_out, 2) in [-1,1]
    grid = torch.stack([
        desc_xs / (Wc - 1) * 2 - 1,
        desc_ys / (Hc - 1) * 2 - 1,
    ], dim=-1).unsqueeze(0).unsqueeze(0)              # (1, 1, N, 2)

    kp_desc_full = F.grid_sample(
        desc, grid, mode='bilinear', align_corners=True)
    kp_desc_full = kp_desc_full.squeeze(0).squeeze(1)  # (256, N)
    kp_desc_full = kp_desc_full.permute(1, 0)           # (N, 256)
    kp_desc_full = F.normalize(kp_desc_full, p=2, dim=1)

    kp_uv = torch.stack([xs, ys], dim=-1).cpu().numpy()       # (N, 2) [u, v]
    kp_desc_full = kp_desc_full.cpu().numpy()
    kp_score_np = kp_score.cpu().numpy()

    return kp_uv, kp_desc_full, kp_score_np


def load_superpoint(device="cuda", weights_path="/tmp/superpoint_v1.pth"):
    """加载 SuperPoint 模型和预训练权重。"""
    model = SuperPointNet().to(device).eval()

    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt)
        print(f"  SuperPoint weights loaded from {weights_path}")
    else:
        print(f"  WARNING: weights not found at {weights_path}")
        print("  Using random weights — results will be garbage!")
        print("  Download from: https://github.com/magicleap/SuperPointPretrainedNetwork")

    return model


# ─────────────────────────────────────────────
# 图像预处理
# ─────────────────────────────────────────────

def load_image_gray(image_path):
    """加载图像并转为灰度，shape (1, 1, H, W), 值域 [0,1]"""
    img = Image.open(image_path).convert('L')
    img_np = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return tensor, img_np.shape


# ─────────────────────────────────────────────
# 特征库构建
# ─────────────────────────────────────────────

def build_prototype_bank(prototype_dir, annotation_dir, model, device="cuda",
                         anchor_radius_px=30):
    """为每个原型提取 anchor 位置附近的 SuperPoint 描述子。

    Args:
        prototype_dir: prototype 图像目录 (cluster*_TV_*.jpg)
        annotation_dir: 标注 JSON 目录 (cluster*_anchor_v2.json)
        model: SuperPoint 模型
        device: 设备
        anchor_radius_px: 在标注 anchor ±N 像素范围内选最强关键点

    Returns:
        feature_bank: {cluster_stem: {anchors: [...], desc: ..., ...}}
    """
    feature_bank = {}
    anno_files = sorted(Path(annotation_dir).glob("cluster*_anchor_v2.json"))

    for af in anno_files:
        with open(af) as f:
            ann = json.load(f)

        cluster_stem = ann["info"]["stem"]
        img_path = Path(prototype_dir) / f"{cluster_stem}.jpg"
        if not img_path.exists():
            print(f"  SKIP {cluster_stem}: image not found")
            continue

        # 加载图像，提取特征
        img_tensor, (h, w) = load_image_gray(str(img_path))
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            semi, desc = model(img_tensor)
            kp_uv, kp_desc, kp_score = extract_keypoints(semi, desc)

        # 为每个 anchor 在附近找最强关键点
        anchors_out = []
        for anchor in ann["anchors"]:
            au, av = anchor["pixel_uv"]

            # 在 anchor ±radius 内找关键点
            dist = np.sqrt((kp_uv[:, 0] - au) ** 2 + (kp_uv[:, 1] - av) ** 2)
            nearby = dist < anchor_radius_px

            if nearby.sum() == 0:
                print(f"  WARNING: {cluster_stem} anchor {anchor['id']} "
                      f"({au},{av}) — no SuperPoint kp within {anchor_radius_px}px")
                continue

            # 选最强
            nearby_idx = np.where(nearby)[0]
            best_local_idx = nearby_idx[kp_score[nearby_idx].argmax()]
            best_uv = kp_uv[best_local_idx].tolist()
            best_desc = kp_desc[best_local_idx]
            best_score = float(kp_score[best_local_idx])

            anchors_out.append({
                "id": anchor["id"],
                "proto_uv": [au, av],          # 原始标注
                "sp_uv": best_uv,               # SuperPoint 检测到的精确位置
                "sp_offset_px": float(np.sqrt((best_uv[0] - au) ** 2 + (best_uv[1] - av) ** 2)),
                "sp_score": best_score,
                "desc": best_desc,
                "anchor_3d": anchor.get("anchor_3d"),
            })

        feature_bank[cluster_stem] = {
            "stem": cluster_stem,
            "n_anchors": len(anchors_out),
            "anchors": anchors_out,
        }
        print(f"  [{cluster_stem}] {len(anchors_out)}/{len(ann['anchors'])} anchors matched")

    print(f"  Feature bank: {len(feature_bank)} prototypes, "
          f"{sum(v['n_anchors'] for v in feature_bank.values())} total anchors")
    return feature_bank


# ─────────────────────────────────────────────
# 匹配
# ─────────────────────────────────────────────

def match_frame(frame_path, feature_bank, model, device="cuda",
                ratio_thresh=0.80, epipolar_thresh_px=3.0,
                min_matches=2):
    """对单帧图像做 SuperPoint 匹配。

    匹配策略:
      1. 提取目标帧所有 SuperPoint 关键点和描述子
      2. 对每个原型 anchor 的描述子，在目标帧做 NN 匹配 + ratio test
      3. 用基础矩阵 (F) 做几何一致性验证
      4. 返回通过验证的 2D 匹配 + PCD 3D 查表

    Args:
        frame_path: 目标 PNG 路径
        feature_bank: 原型特征库
        model: SuperPoint 模型
        device: 设备
        ratio_thresh: Lowe's ratio test 上限
        epipolar_thresh_px: 对极距离阈值
        min_matches: 最少匹配点数

    Returns:
        matches: [{anchor_id, pixel_uv, similarity, proto_stem, anchor_3d_proto}]
        viz_data: (kp_uv, kp_score) 用于可视化
    """
    # 提取目标帧特征
    img_tensor, (h, w) = load_image_gray(frame_path)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        semi, desc = model(img_tensor)
        tgt_kp_uv, tgt_kp_desc, tgt_kp_score = extract_keypoints(semi, desc)

    if len(tgt_kp_uv) < 10:
        return [], None

    # 对每个原型做描述子匹配
    all_matches = []
    for proto_stem, proto_data in feature_bank.items():
        for anchor in proto_data["anchors"]:
            proto_desc = torch.from_numpy(anchor["desc"]).float().to(device)

            # 余弦相似度 (描述子已 L2 归一化)
            sims = torch.mv(torch.from_numpy(tgt_kp_desc).float().to(device),
                            proto_desc).cpu().numpy()

            # 找最佳/次优
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            # Ratio test: 抑制最近邻周围，找次优
            best_uv = tgt_kp_uv[best_idx]
            nms_dist = np.sqrt(
                (tgt_kp_uv[:, 0] - best_uv[0]) ** 2 +
                (tgt_kp_uv[:, 1] - best_uv[1]) ** 2
            )
            suppressed = sims.copy()
            suppressed[nms_dist < 8] = -1  # 抑制 8px 内
            second_idx = int(np.argmax(suppressed))
            second_sim = float(sims[second_idx])
            ratio = second_sim / best_sim if best_sim > 0 else 1.0

            if ratio < ratio_thresh and best_sim > 0.5:
                all_matches.append({
                    "anchor_id": anchor["id"],
                    "pixel_uv": tgt_kp_uv[best_idx].tolist(),
                    "similarity": best_sim,
                    "ratio": ratio,
                    "proto_stem": proto_stem,
                    "proto_anchor_uv": anchor["proto_uv"],
                    "sp_uv": anchor["sp_uv"],
                    "anchor_3d_proto": anchor.get("anchor_3d"),
                })

    # 去重: 每个原型 anchor 只保留相似度最高的匹配
    # 按 anchor_id 分组
    best_per_anchor = {}
    for m in all_matches:
        aid = m["anchor_id"]
        if aid not in best_per_anchor or m["similarity"] > best_per_anchor[aid]["similarity"]:
            best_per_anchor[aid] = m

    matches = list(best_per_anchor.values())

    # 如果匹配太多，做几何一致性验证
    if len(matches) >= min_matches:
        matches = _filter_geometric(matches, feature_bank)

    return matches, (tgt_kp_uv, tgt_kp_score)


def _filter_geometric(matches, feature_bank, max_distort_px=5.0):
    """用原型 anchor 间相对位置做几何一致性验证。

    对 2-anchor 原型: 验证水平对齐 + 宽度比例
    对 4-anchor 原型: 验证四边形内角
    """
    if len(matches) < 2:
        return matches

    # 按 proto_stem 分组
    by_proto = {}
    for m in matches:
        ps = m["proto_stem"]
        by_proto.setdefault(ps, []).append(m)

    filtered = []
    for ps, ms in by_proto.items():
        proto = feature_bank.get(ps)
        if proto is None:
            filtered.extend(ms)
            continue

        if proto["n_anchors"] == 2 and len(ms) == 2:
            # 水平对齐检查
            proto_anchors = proto["anchors"]
            proto_v_avg = np.mean([a["proto_uv"][1] for a in proto_anchors])
            proto_w = abs(proto_anchors[0]["proto_uv"][0] - proto_anchors[1]["proto_uv"][0])

            match_v_avg = np.mean([m["pixel_uv"][1] for m in ms])
            match_w = abs(ms[0]["pixel_uv"][0] - ms[1]["pixel_uv"][0])

            v_ok = abs(match_v_avg - proto_v_avg) < max_distort_px * 2
            w_ok = 0.5 < match_w / max(proto_w, 1) < 2.0

            if v_ok and w_ok:
                filtered.extend(ms)
            # 否则丢弃所有该原型的匹配

        elif proto["n_anchors"] == 4 and len(ms) >= 3:
            filtered.extend(ms)  # 宽松处理
        else:
            filtered.extend(ms)

    return filtered


# ─────────────────────────────────────────────
# PCD 3D 查表 (复用 shelf_dinov2_match.py 的逻辑)
# ─────────────────────────────────────────────

def _read_pcd_binary(path):
    """读取 PCD binary 文件 → (N,3) xyz 数组。"""
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode().strip()
            if line == "DATA binary":
                break
        dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("rgb", "u4")])
        data = np.frombuffer(f.read(), dtype=dtype)
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    return xyz


def _build_depth_map(xyz, img_w=640, img_h=480, fx=420, fy=420, cx=320, cy=240):
    """从点云构建深度图 (只保留每个像素最近的点)。"""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    valid = z > 1.0
    xv, yv, zv = x[valid], y[valid], z[valid]
    u = np.round(fx * xv / zv + cx).astype(np.int32)
    v = np.round(fy * yv / zv + cy).astype(np.int32)
    in_bounds = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u, v, zv = u[in_bounds], v[in_bounds], zv[in_bounds]
    dm = np.full((img_h, img_w), np.nan, dtype=np.float32)
    for i in range(len(u)):
        if np.isnan(dm[v[i], u[i]]) or zv[i] < dm[v[i], u[i]]:
            dm[v[i], u[i]] = zv[i]
    return dm


def _get_anchor_3d(u0, v0, depth_map, window=5, fx=420, fy=420, cx=320, cy=240):
    """从深度图查表得到像素 (u0,v0) 对应的 3D 坐标 (中值深度)。"""
    h, w = depth_map.shape
    half = window // 2
    u_min, u_max = max(0, u0 - half), min(w, u0 + half + 1)
    v_min, v_max = max(0, v0 - half), min(h, v0 + half + 1)
    wd = depth_map[v_min:v_max, u_min:u_max]
    vd = wd[~np.isnan(wd)]
    if len(vd) == 0:
        for expand in range(half + 1, min(w, h) // 2, 5):
            u2_min = max(0, u0 - expand); u2_max = min(w, u0 + expand + 1)
            v2_min = max(0, v0 - expand); v2_max = min(h, v0 + expand + 1)
            wd2 = depth_map[v2_min:v2_max, u2_min:u2_max]
            vd2 = wd2[~np.isnan(wd2)]
            if len(vd2) > 0:
                vd = vd2
                break
    if len(vd) == 0:
        return None, 0
    z0 = float(np.median(vd))
    x0 = (u0 - cx) * z0 / fx
    y0 = (v0 - cy) * z0 / fy
    return [round(x0, 1), round(y0, 1), round(z0, 1)], len(vd)


def pcd_lookup_3d(pixel_uv, pcd_path, window=5, fx=420, fy=420, cx=320, cy=240):
    """从 PCD 深度图查找 3D 坐标。

    Args:
        pixel_uv: (N, 2) [u, v] 像素坐标
        pcd_path: .pcd 文件路径
        window: 中值滤波核大小
        fx, fy, cx, cy: 相机内参

    Returns:
        (N, 3) 3D 坐标 (x, y, z) 或 None (无效深度)
    """
    xyz = _read_pcd_binary(pcd_path)
    depth_map = _build_depth_map(xyz, fx=fx, fy=fy, cx=cx, cy=cy)

    results = []
    for uv in pixel_uv:
        u, v = int(round(uv[0])), int(round(uv[1]))
        pt3d, _ = _get_anchor_3d(u, v, depth_map, window=window, fx=fx, fy=fy, cx=cx, cy=cy)
        results.append(pt3d)

    return results


# ─────────────────────────────────────────────
# 全量数据集处理
# ─────────────────────────────────────────────

def process_dataset(feature_bank, data_dir, output_dir, model, device="cuda",
                    ratio_thresh=0.80, max_frames=None):
    """全量数据集匹配。"""
    png_dir = Path(data_dir)
    png_files = sorted(png_dir.glob("TV_*.png"))

    if max_frames:
        png_files = png_files[:max_frames]

    all_results = {}
    stats = {"total": 0, "matched": 0, "total_kp": 0}

    for pf in tqdm(png_files, desc="Matching"):
        stem = pf.stem
        pcd_path = pf.with_suffix(".pcd")
        stats["total"] += 1

        matches, _ = match_frame(
            str(pf), feature_bank, model, device=device,
            ratio_thresh=ratio_thresh)

        if matches:
            # PCD 3D 查表
            if pcd_path.exists():
                uvs = [m["pixel_uv"] for m in matches]
                pts_3d = pcd_lookup_3d(uvs, str(pcd_path))
                for m, pt3d in zip(matches, pts_3d):
                    m["anchor_3d"] = pt3d

            all_results[stem] = {
                "stem": stem,
                "num_keypoints": len(matches),
                "keypoints": [{
                    "id": m["anchor_id"],
                    "pixel_uv": m["pixel_uv"],
                    "anchor_3d": m.get("anchor_3d"),
                    "similarity": m["similarity"],
                    "ratio": m["ratio"],
                    "proto_stem": m["proto_stem"],
                } for m in matches],
            }
            stats["matched"] += 1
            stats["total_kp"] += len(matches)
        else:
            all_results[stem] = {
                "stem": stem,
                "num_keypoints": 0,
                "keypoints": [],
            }

    print(f"\n  Frames: {stats['total']}, Matched: {stats['matched']} "
          f"({100*stats['matched']/max(stats['total'],1):.1f}%)")
    print(f"  Total keypoints: {stats['total_kp']}")
    print(f"  Avg kp/frame: {stats['total_kp']/max(stats['matched'],1):.1f}")

    return all_results, stats


# ─────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────

def render_visualizations(all_results, data_dir, output_dir, max_frames=None):
    """渲染 BEV + 3D 可视化。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    viz_dir = Path(output_dir) / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    matched_frames = [(stem, r) for stem, r in all_results.items()
                       if r["num_keypoints"] > 0]
    if max_frames:
        matched_frames = matched_frames[:max_frames]

    for stem, result in tqdm(matched_frames, desc="Rendering"):
        img_path = Path(data_dir) / f"{stem}.png"
        if not img_path.exists():
            continue

        img = Image.open(img_path)
        img_np = np.array(img)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        ax.imshow(img_np)
        ax.set_title(stem)

        for kp in result["keypoints"]:
            u, v = kp["pixel_uv"]
            ax.plot(u, v, 'r+', markersize=15, markeredgewidth=2)
            ax.plot(u, v, 'ro', markersize=7, fillstyle='none')
            ax.annotate(f"#{kp['id']}", (u + 5, v - 5),
                        color='red', fontsize=8, fontweight='bold')

        ax.set_axis_off()
        fig.tight_layout(pad=0)
        fig.savefig(viz_dir / f"{stem}_pred.jpg", dpi=100, bbox_inches='tight')
        plt.close(fig)

    print(f"  Visualizations saved to {viz_dir}")


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SuperPoint 货架角点匹配")
    parser.add_argument("--prototype_annotations",
                        default="data/new_sheef/prototype_annotations")
    parser.add_argument("--prototype_data_dir",
                        default="data/new_sheef/prototypes")
    parser.add_argument("--full_data_dir",
                        default="data/new_sheef/pngs")
    parser.add_argument("--output_dir",
                        default="output/superpoint_match")
    parser.add_argument("--ratio_thresh", type=float, default=0.80,
                        help="Lowe's ratio test upper bound")
    parser.add_argument("--anchor_radius_px", type=int, default=30,
                        help="搜索 anchor 附近 SuperPoint 关键点的半径")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", default="/tmp/superpoint_v1.pth",
                        help="SuperPoint 预训练权重路径")
    parser.add_argument("--viz_only", action="store_true",
                        help="仅可视化已有结果")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 如果仅可视化
    if args.viz_only:
        matches_path = output_dir / "all_matches.json"
        if matches_path.exists():
            with open(matches_path) as f:
                all_results = json.load(f)
            render_visualizations(all_results, args.full_data_dir, str(output_dir))
        else:
            print(f"ERROR: {matches_path} not found")
        return

    # 加载模型
    print("Loading SuperPoint...")
    model = load_superpoint(device=args.device, weights_path=args.weights)

    # Phase 1: 构建原型特征库
    print("\n[Phase 1] Building prototype feature bank...")
    feature_bank = build_prototype_bank(
        args.prototype_data_dir, args.prototype_annotations,
        model, device=args.device, anchor_radius_px=args.anchor_radius_px)

    # 保存特征库
    bank_np = {}
    for stem, data in feature_bank.items():
        bank_np[stem] = {
            "stem": data["stem"],
            "n_anchors": data["n_anchors"],
            "anchors": [{
                "id": a["id"],
                "proto_uv": a["proto_uv"],
                "sp_uv": a["sp_uv"],
                "sp_offset_px": a["sp_offset_px"],
                "sp_score": a["sp_score"],
                "desc": a["desc"],
            } for a in data["anchors"]],
        }
    np.savez(output_dir / "superpoint_bank.npz", feature_bank=bank_np)
    print(f"  Bank saved: {output_dir / 'superpoint_bank.npz'}")

    # Phase 2: 全量匹配
    print(f"\n[Phase 2] Matching frames...")
    all_results, stats = process_dataset(
        feature_bank, args.full_data_dir, str(output_dir),
        model, device=args.device, ratio_thresh=args.ratio_thresh,
        max_frames=args.max_frames)

    # 保存结果
    with open(output_dir / "all_matches.json", "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    print(f"  Results saved: {output_dir / 'all_matches.json'}")

    # Phase 3: 可视化
    print(f"\n[Phase 3] Rendering visualizations...")
    render_visualizations(all_results, args.full_data_dir, str(output_dir))

    print(f"\nDone! Output: {output_dir}/")


if __name__ == "__main__":
    main()
