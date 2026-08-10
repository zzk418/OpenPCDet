#!/usr/bin/env python3
"""
DINOv2 Patch 特征提取 → 全量跨帧匹配 → PCD 3D 查表 → 可视化
============================================================

Pipeline:
  1. 加载 12 个 prototype 标注, 提取 DINOv2 patch 特征 → 特征库
  2. 全量帧匹配: CLS 分配簇 → patch 余弦相似度匹配 → ratio test 去歧义
     → 几何一致性验证 (水平对齐 + 宽度比例) → PCD 3D 查表
  3. 导出 JSON + 可视化 + 评估指标

匹配质量控制 (相比旧版关键改进):
  - Ratio test (SIFT 风格): 最优/次优相似度比 < 0.9 才保留，过滤歧义匹配
  - Spatial NMS: ratio test 前抑制最优 patch 周围区域，防止相邻 patch 干扰
  - Per-anchor top-1: 每个原型 anchor 只保留 1 个最优匹配 (不再 top-3)
  - 几何一致性: 验证匹配点对水平对齐 + 宽度比例与原型一致
  - 默认相似度阈值 0.7 (原 0.6)

用法:
  python tools/shelf_dinov2_match.py                                # 完整流程
  python tools/shelf_dinov2_match.py --skip_extract                 # 重用已有特征库
  python tools/shelf_dinov2_match.py --viz_only                     # 仅重新可视化
  python tools/shelf_dinov2_match.py --max_frames 50                # 限制处理帧数
  python tools/shelf_dinov2_match.py --conf_thresh 0.8              # 收紧阈值
  python tools/shelf_dinov2_match.py --ratio_thresh 0.85            # 更严格的 ratio test
  python tools/shelf_dinov2_match.py --no_geometry_filter           # 关闭几何过滤
"""

import argparse, json, os, sys, time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

# ── PCD / Depth helpers (复用自 shelf_anchor_v3_web) ──

def _read_pcd_binary(path):
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


def _to_native(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    return obj


# ── DINOv2 ──

DINOV2_INPUT_SIZE = 518  # divisible by 14, larger res for finer patch grid
PATCH_SIZE = 14
GRID_H = None  # set dynamically after first inference
GRID_W = None
_GRID_PARAMS = None  # (patch_h, patch_w, actual_h, actual_w) cached from first call


def load_dinov2(device="cuda"):
    """加载 DINOv2 ViT-S/14。"""
    print("Loading DINOv2 ViT-S/14...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small", local_files_only=True)
    model = AutoModel.from_pretrained("facebook/dinov2-small", local_files_only=True).to(device).eval()
    print(f"  DINOv2 loaded on {device}, embed_dim=384")
    return processor, model


@torch.no_grad()
def extract_features(processor, model, img_bgr, device="cuda"):
    """提取 DINOv2 CLS + patch features。

    对 640×480 图像，禁用 center_crop，resize shortest_edge=518，
    得到约 691×518 的实际输入 (patch grid ≈ 49×37)。

    Args:
        img_bgr: (H, W, 3) BGR image at 640×480

    Returns:
        cls_feat: (384,) normalized
        patch_feats: (Hp, Wp, 384) normalized patch features
        patch_h, patch_w: number of patches
        actual_h, actual_w: pixel size after resize
    """
    global GRID_H, GRID_W, _GRID_PARAMS
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Disable center_crop, use larger shortest_edge for finer grid
    inputs = processor(images=img_rgb, return_tensors="pt",
                       do_center_crop=False,
                       size={"shortest_edge": DINOV2_INPUT_SIZE})
    inputs = {k: v.to(device) for k, v in inputs.items()}

    pixel_values = inputs["pixel_values"]
    _, _, H, W = pixel_values.shape
    patch_h, patch_w = H // PATCH_SIZE, W // PATCH_SIZE

    if GRID_H is None:
        GRID_H, GRID_W = patch_h, patch_w
        _GRID_PARAMS = (patch_h, patch_w, H, W)
        print(f"  DINOv2 input: {H}×{W} → patch grid: {patch_h}×{patch_w}")

    outputs = model(**inputs)
    hidden = outputs.last_hidden_state  # (1, 1+N_patches, 384)

    cls_feat = F.normalize(hidden[:, 0, :], dim=-1)  # (1, 384)
    patch_feats_t = F.normalize(hidden[:, 1:, :], dim=-1)  # (1, N_patches, 384)
    patch_feats_t = patch_feats_t.reshape(1, patch_h, patch_w, 384)

    return (cls_feat.cpu().numpy()[0],
            patch_feats_t.cpu().numpy()[0],
            patch_h, patch_w, H, W)


def pixel_to_patch(u, v, img_w=640, img_h=480, patch_h=None, patch_w=None,
                   actual_h=None, actual_w=None):
    """将 640×480 像素坐标映射到 DINOv2 patch 坐标。

    图像先 resize 到 actual_w×actual_h (DINOv2 实际输入尺寸)，
    然后按 PATCH_SIZE=14 划分 patch grid。
    """
    if patch_h is None: patch_h = GRID_H
    if patch_w is None: patch_w = GRID_W
    if actual_h is None or actual_w is None:
        actual_h = patch_h * PATCH_SIZE
        actual_w = patch_w * PATCH_SIZE

    u_dino = u * actual_w / img_w
    v_dino = v * actual_h / img_h
    pu = max(0, min(patch_w - 1, int(u_dino / PATCH_SIZE)))
    pv = max(0, min(patch_h - 1, int(v_dino / PATCH_SIZE)))
    return pu, pv


def patch_to_pixel(pu, pv, img_w=640, img_h=480, patch_h=None, patch_w=None,
                   actual_h=None, actual_w=None):
    """将 DINOv2 patch 坐标映射回 640×480 像素坐标 (patch 中心)。"""
    if patch_h is None: patch_h = GRID_H
    if patch_w is None: patch_w = GRID_W
    if actual_h is None or actual_w is None:
        actual_h = patch_h * PATCH_SIZE
        actual_w = patch_w * PATCH_SIZE

    u_dino = (pu + 0.5) * PATCH_SIZE
    v_dino = (pv + 0.5) * PATCH_SIZE
    u = u_dino * img_w / actual_w
    v = v_dino * img_h / actual_h
    return int(round(u)), int(round(v))


# ── Phase 1: 特征库构建 ──

def build_feature_bank(annotation_dir, data_dir, processor, model, output_dir, device="cuda"):
    """从 prototype 标注中提取 DINOv2 patch 特征库。

    Returns:
        feature_bank: dict mapping cluster_name → {
            'cls': (384,) array,
            'anchors': [{id, pixel_uv, anchor_3d, feature: (384,), patch_uv}]
        }
    """
    ann_dir = Path(annotation_dir)
    data_dir = Path(data_dir)

    feature_bank = {}

    ann_files = sorted(ann_dir.glob("cluster*_anchor_v2.json"))
    print(f"\n[Phase 1] Building feature bank from {len(ann_files)} prototypes...")

    for ann_file in ann_files:
        cluster_name = ann_file.stem.replace("_anchor_v2", "")  # e.g. "cluster0_TV_250000011092"
        with open(ann_file) as f:
            ann = json.load(f)

        stem = ann["stem"]
        num_id = stem  # cluster format uses full stem

        # Load image
        img_path = None
        for ext in [".jpg", ".png"]:
            cand = data_dir / f"{stem}{ext}"
            if cand.exists():
                img_path = str(cand)
                break
            cand = data_dir / f"{cluster_name}{ext}"  # fallback
            if cand.exists():
                img_path = str(cand)
                break
        if img_path is None:
            print(f"  [{cluster_name}] Image not found, skipping")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"  [{cluster_name}] Cannot read image, skipping")
            continue
        if img.shape[0] != 480 or img.shape[1] != 640:
            img = cv2.resize(img, (640, 480))

        # Extract DINOv2 features
        cls_feat, patch_feats, patch_h, patch_w, actual_h, actual_w = \
            extract_features(processor, model, img, device)

        anchors = []
        for kp in ann.get("keypoints", []):
            u, v = kp["pixel_uv"]
            pu, pv = pixel_to_patch(u, v, patch_h=patch_h, patch_w=patch_w,
                                    actual_h=actual_h, actual_w=actual_w)
            feat = patch_feats[pv, pu].copy()  # (384,)

            anchors.append({
                "id": kp.get("id", 0),
                "pixel_uv": [u, v],
                "patch_uv": [int(pu), int(pv)],
                "anchor_3d": kp.get("anchor_3d"),
                "feature": feat,
                "label": kp.get("label", ""),
            })

        feature_bank[cluster_name] = {
            "cls": cls_feat,
            "anchors": anchors,
            "stem": stem,
        }
        print(f"  [{cluster_name}] {len(anchors)} anchor features extracted")

    # Save feature bank
    os.makedirs(output_dir, exist_ok=True)
    feature_path = os.path.join(output_dir, "prototype_features.npz")
    save_data = {}
    for name, data in feature_bank.items():
        save_data[f"{name}_cls"] = data["cls"]
        save_data[f"{name}_stem"] = np.array(data["stem"])
        for i, anchor in enumerate(data["anchors"]):
            save_data[f"{name}_anchor{i}_feat"] = anchor["feature"]
            save_data[f"{name}_anchor{i}_uv"] = np.array(anchor["pixel_uv"])
            save_data[f"{name}_anchor{i}_3d"] = np.array(anchor["anchor_3d"]) if anchor["anchor_3d"] else np.zeros(3)

    np.savez_compressed(feature_path, **save_data)
    print(f"  Feature bank saved: {feature_path}")

    # Also save as JSON-compatible for inspection
    info = {}
    for name, data in feature_bank.items():
        info[name] = {
            "stem": data["stem"],
            "n_anchors": len(data["anchors"]),
            "anchors": [{"id": a["id"], "pixel_uv": a["pixel_uv"],
                         "patch_uv": a["patch_uv"], "label": a["label"]}
                       for a in data["anchors"]],
        }
    with open(os.path.join(output_dir, "feature_bank_info.json"), "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    return feature_bank


# ── Phase 2: 全量匹配 ──

def assign_cluster(cls_feat, feature_bank):
    """用 CLS 余弦相似度分配目标帧到最近 prototype 簇。"""
    best_cluster = None
    best_sim = -1
    for name, data in feature_bank.items():
        sim = float(np.dot(cls_feat, data["cls"]))
        if sim > best_sim:
            best_sim = sim
            best_cluster = name
    return best_cluster, best_sim


def match_keypoints(patch_feats, cluster_data, conf_thresh=0.7, ratio_thresh=0.90,
                    nms_patch_radius=3):
    """用 anchor patch 特征在目标帧特征图上做余弦相似度匹配。

    每个 prototype anchor 只返回最优的 1 个匹配。
    通过 ratio test (SIFT 风格) + spatial NMS 过滤模糊匹配:
      - 找到最佳 patch 后，抑制其周围 nms_patch_radius 范围内的 patch
      - 在抑制后的图中找次优匹配
      - 如果次优/最优 > ratio_thresh，说明匹配不唯一，丢弃

    Args:
        patch_feats: (Hp, Wp, 384) target frame patch features
        cluster_data: prototype cluster data with anchors
        conf_thresh: minimum cosine similarity for best match
        ratio_thresh: best/second-best ratio upper bound (lower=stricter)
        nms_patch_radius: spatial suppression radius in patch units

    Returns:
        matches: [{anchor_id, pixel_uv, patch_uv, similarity, ratio, anchor_3d_proto}]
    """
    matches = []
    p_h, p_w = patch_feats.shape[:2]
    flat_feats = patch_feats.reshape(-1, 384)

    for anchor in cluster_data["anchors"]:
        anchor_feat = anchor["feature"]
        sim_map = np.dot(flat_feats, anchor_feat)
        sim_map = sim_map.reshape(p_h, p_w)

        # Find best patch
        best_idx = int(np.argmax(sim_map.ravel()))
        best_pv, best_pu = divmod(best_idx, p_w)
        best_sim = float(sim_map[best_pv, best_pu])

        if best_sim < conf_thresh:
            continue

        # Ratio test with spatial NMS: suppress nearby patches, find second-best
        sim_suppressed = sim_map.copy()
        pv0 = max(0, best_pv - nms_patch_radius)
        pv1 = min(p_h, best_pv + nms_patch_radius + 1)
        pu0 = max(0, best_pu - nms_patch_radius)
        pu1 = min(p_w, best_pu + nms_patch_radius + 1)
        sim_suppressed[pv0:pv1, pu0:pu1] = -2.0  # below any possible cosine sim

        second_sim = float(sim_suppressed.max())
        if second_sim > 0 and best_sim > 0:
            ratio = second_sim / best_sim
        else:
            ratio = 0.0  # no second-best candidate → unique match

        if ratio > ratio_thresh:
            continue  # ambiguous match, discard

        u, v = patch_to_pixel(best_pu, best_pv, patch_h=p_h, patch_w=p_w)

        matches.append({
            "anchor_id": anchor["id"],
            "pixel_uv": [u, v],
            "patch_uv": [int(best_pu), int(best_pv)],
            "similarity": round(best_sim, 4),
            "ratio": round(ratio, 4),
            "anchor_3d_proto": anchor.get("anchor_3d"),
        })

    matches.sort(key=lambda x: x["anchor_id"])
    return matches


def filter_by_geometry(matches, cluster_data, max_vertical_px=80,
                       min_width_ratio=0.3, max_width_ratio=3.0):
    """几何一致性过滤：验证匹配点对的空间关系与原型一致。

    规则（基于原型 anchor 的 pixel_uv）:
      - 2-anchor 原型: 两关键点应大致水平，x 间距比例合理
      - 4-anchor 原型: 4 点应形成合理四边形，上下边大致水平

    Args:
        matches: match_keypoints 的输出
        cluster_data: prototype cluster data
        max_vertical_px: 两点允许的最大垂直偏差 (像素)
        min_width_ratio, max_width_ratio: 匹配宽度 / 原型宽度的允许范围

    Returns:
        filtered matches (可能为空)
    """
    n_matches = len(matches)
    if n_matches < 2:
        return matches  # 单个点无法做几何检查，保留

    proto_anchors = cluster_data["anchors"]
    n_proto = len(proto_anchors)

    if n_proto == 2:
        return _filter_pair_geometry(matches, proto_anchors,
                                     max_vertical_px, min_width_ratio, max_width_ratio)
    elif n_proto == 4:
        return _filter_quad_geometry(matches, proto_anchors,
                                     max_vertical_px, min_width_ratio, max_width_ratio)
    else:
        return matches


def _proto_horizontal_width(anchors):
    """计算原型 anchor 的水平宽度 (像素)。"""
    us = [a["pixel_uv"][0] for a in anchors]
    return max(us) - min(us)


def _filter_pair_geometry(matches, proto_anchors, max_vertical_px,
                          min_width_ratio, max_width_ratio):
    """2-anchor 几何检查：水平对齐 + 宽度比例。"""
    if len(matches) < 2:
        return matches

    proto_width = _proto_horizontal_width(proto_anchors)
    if proto_width <= 0:
        return matches

    # 按 anchor_id 分组取最佳匹配
    best_per_id = {}
    for m in matches:
        aid = m["anchor_id"]
        if aid not in best_per_id or m["similarity"] > best_per_id[aid]["similarity"]:
            best_per_id[aid] = m

    if len(best_per_id) < 2:
        return list(best_per_id.values())

    # 取前两个不同 anchor_id 的匹配
    ids_sorted = sorted(best_per_id.keys())
    m_a = best_per_id[ids_sorted[0]]
    m_b = best_per_id[ids_sorted[1]]

    u_a, v_a = m_a["pixel_uv"]
    u_b, v_b = m_b["pixel_uv"]

    # 1. 水平检查
    vertical_diff = abs(v_a - v_b)
    if vertical_diff > max_vertical_px:
        return []

    # 2. 宽度比例检查
    match_width = abs(u_a - u_b)
    width_ratio = match_width / proto_width
    if width_ratio < min_width_ratio or width_ratio > max_width_ratio:
        return []

    return [m_a, m_b]


def _filter_quad_geometry(matches, proto_anchors, max_vertical_px,
                          min_width_ratio, max_width_ratio):
    """4-anchor 几何检查：4 点应形成合理四边形。

    检查:
      1. 存在 4 个不同 anchor_id 的匹配
      2. 左右分界清晰 (按 x 坐标可分左右两组)
      3. 上下分界清晰 (按 y 坐标可分上下两组)
      4. 上边和下边大致水平
    """
    if len(matches) < 4:
        return matches

    proto_width = _proto_horizontal_width(proto_anchors)
    if proto_width <= 0:
        return matches

    # 按 anchor_id 分组取最佳匹配
    best_per_id = {}
    for m in matches:
        aid = m["anchor_id"]
        if aid not in best_per_id or m["similarity"] > best_per_id[aid]["similarity"]:
            best_per_id[aid] = m

    if len(best_per_id) < 4:
        return list(best_per_id.values())

    kps = list(best_per_id.values())

    # 1. 按 x 分成左右两组 (以中位数为界)
    xs = [m["pixel_uv"][0] for m in kps]
    median_x = float(np.median(xs))
    left = [m for m in kps if m["pixel_uv"][0] < median_x]
    right = [m for m in kps if m["pixel_uv"][0] >= median_x]

    if len(left) < 2 or len(right) < 2:
        return []

    # 2. 左右组内按 y 分上下
    def split_up_down(group):
        ys = [m["pixel_uv"][1] for m in group]
        median_y = float(np.median(ys))
        up = [m for m in group if m["pixel_uv"][1] < median_y]
        down = [m for m in group if m["pixel_uv"][1] >= median_y]
        return up, down

    left_up, left_down = split_up_down(left)
    right_up, right_down = split_up_down(right)

    if not (left_up and left_down and right_up and right_down):
        return []

    # 3. 取最佳匹配: 每组取 sim 最高的
    def best_in_group(group):
        return max(group, key=lambda m: m["similarity"])

    lu = best_in_group(left_up)
    ld = best_in_group(left_down)
    ru = best_in_group(right_up)
    rd = best_in_group(right_down)

    # 4. 宽度检查 (左右组之间)
    left_xs = [m["pixel_uv"][0] for m in [lu, ld]]
    right_xs = [m["pixel_uv"][0] for m in [ru, rd]]
    match_width = np.mean(right_xs) - np.mean(left_xs)
    width_ratio = match_width / proto_width
    if width_ratio < min_width_ratio or width_ratio > max_width_ratio:
        return []

    # 5. 水平检查 (上边 + 下边)
    if abs(lu["pixel_uv"][1] - ru["pixel_uv"][1]) > max_vertical_px:
        return []
    if abs(ld["pixel_uv"][1] - rd["pixel_uv"][1]) > max_vertical_px:
        return []

    quad = [lu, ld, ru, rd]
    quad.sort(key=lambda m: m["anchor_id"])
    return quad


def process_full_dataset(feature_bank, data_dir, output_dir, processor, model,
                         device="cuda", conf_thresh=0.7, ratio_thresh=0.90,
                         enable_geometry_filter=True, max_frames=None):
    """Phase 2: 全量数据集特征匹配 + PCD 3D 查表。

    Args:
        conf_thresh: 最低余弦相似度 (default 0.7)
        ratio_thresh: ratio test 上限 (default 0.90, 越低越严格)
        enable_geometry_filter: 是否启用几何一致性过滤
    """
    data_dir = Path(data_dir)

    # Scan all frames
    pcd_paths = sorted(data_dir.glob("TV_*.pcd"))
    pcd_paths = [p for p in pcd_paths if "000000001" not in p.stem]  # exclude outlier

    if max_frames:
        pcd_paths = pcd_paths[:max_frames]

    print(f"\n[Phase 2] Matching {len(pcd_paths)} frames...")

    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    all_results = {}
    match_stats = defaultdict(lambda: {"total": 0, "matched": 0, "keypoints": 0})

    for i, pcd_path in enumerate(pcd_paths):
        stem = pcd_path.stem
        num_id = stem.replace("TV_", "")

        # Find image
        img_path = None
        for ext in [".jpg", ".png"]:
            for cand_name in [stem, num_id]:
                cand = data_dir / f"{cand_name}{ext}"
                if cand.exists():
                    img_path = str(cand)
                    break
            if img_path:
                break
        if img_path is None:
            continue

        # Load & resize
        img = cv2.imread(img_path)
        if img is None:
            continue
        if img.shape[0] != 480 or img.shape[1] != 640:
            img = cv2.resize(img, (640, 480))

        # DINOv2 features
        cls_feat, patch_feats, _, _, _, _ = extract_features(processor, model, img, device)

        # Assign cluster
        cluster_name, cluster_sim = assign_cluster(cls_feat, feature_bank)
        cluster_data = feature_bank[cluster_name]

        # Match keypoints (ratio test + per-anchor top-1)
        raw_matches = match_keypoints(patch_feats, cluster_data,
                                      conf_thresh=conf_thresh,
                                      ratio_thresh=ratio_thresh)

        # Geometric consistency filter
        if enable_geometry_filter and raw_matches:
            raw_matches = filter_by_geometry(raw_matches, cluster_data)

        # PCD 3D lookup
        xyz = _read_pcd_binary(str(pcd_path))
        dm = _build_depth_map(xyz)

        keypoints = []
        for m in raw_matches:
            u, v = m["pixel_uv"]
            anchor_3d, valid_count = _get_anchor_3d(u, v, dm)
            if anchor_3d is None:
                continue
            keypoints.append({
                "id": m["anchor_id"],
                "pixel_uv": [u, v],
                "anchor_3d": anchor_3d,
                "similarity": m["similarity"],
                "ratio": m.get("ratio", 0),
                "valid_depth_count": valid_count,
                "cluster": cluster_name,
            })

        match_stats[cluster_name]["total"] += 1
        if keypoints:
            match_stats[cluster_name]["matched"] += 1
            match_stats[cluster_name]["keypoints"] += len(keypoints)

        result = {
            "stem": stem,
            "cluster": cluster_name,
            "cluster_similarity": round(cluster_sim, 4),
            "num_keypoints": len(keypoints),
            "keypoints": keypoints,
            "prototype_stem": feature_bank[cluster_name]["stem"],
        }
        all_results[stem] = result

        # Save individual result
        out_json = os.path.join(results_dir, f"{stem}_match.json")
        with open(out_json, "w") as f:
            json.dump(_to_native(result), f, indent=2, ensure_ascii=False)

        if (i + 1) % 50 == 0 or i == len(pcd_paths) - 1:
            total_kp = sum(r["num_keypoints"] for r in all_results.values())
            n_matched = sum(1 for r in all_results.values() if r["num_keypoints"] > 0)
            print(f"  [{i+1}/{len(pcd_paths)}] {n_matched} frames matched, {total_kp} keypoints total")

    # Save full results
    full_json = os.path.join(output_dir, "all_matches.json")
    with open(full_json, "w") as f:
        json.dump(_to_native(all_results), f, indent=2, ensure_ascii=False)

    return all_results, match_stats


# ── Phase 3: 全量帧可视化 ──

# Visual style (consistent with infer_shelf_anchor.py draw_keypoints)
DOT_COLOR = (0, 0, 255)        # BGR red
LEGEND_BG = (40, 40, 40)        # dark panel
LEGEND_FG = (240, 240, 240)     # light text
LEGEND_ACCENT = (80, 200, 255)  # gold accent for title


def draw_matches(img_bgr, result, output_path):
    """可视化 DINOv2 匹配结果: 图像 + 关键点 + legend。

    Format: consistent with shelf_anchor_v2_pred/viz (*_pred.jpg)
    """
    viz = img_bgr.copy()
    h, w = viz.shape[:2]
    keypoints = result.get("keypoints", [])

    # ── 1. Draw keypoints on the image ──
    for i, kp in enumerate(keypoints):
        u, v = kp["pixel_uv"]

        # thin crosshair
        cv2.line(viz, (u - 6, v), (u + 6, v), DOT_COLOR, 1)
        cv2.line(viz, (u, v - 6), (u, v + 6), DOT_COLOR, 1)

        # small filled circle + dark outline
        cv2.circle(viz, (u, v), 4, DOT_COLOR, -1)
        cv2.circle(viz, (u, v), 4, (30, 30, 30), 1)

        # tiny number label next to the dot
        cv2.putText(viz, str(i + 1), (u + 7, v - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, DOT_COLOR, 1, cv2.LINE_AA)

    # ── 2. Legend panel (top-left) ──
    if keypoints:
        n_kp = len(keypoints)
        line_h = 16
        panel_w = 220
        panel_h = 32 + n_kp * line_h + 6

        # semi-transparent background
        overlay = viz.copy()
        cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), LEGEND_BG, -1)
        cv2.addWeighted(overlay, 0.75, viz, 0.25, 0, viz)

        # border
        cv2.rectangle(viz, (8, 8), (8 + panel_w, 8 + panel_h), (100, 100, 100), 1)

        # title
        cv2.putText(viz, f"Shelf Corners  ({n_kp} pts)", (16, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, LEGEND_ACCENT, 1, cv2.LINE_AA)

        # separator line
        cv2.line(viz, (16, 33), (8 + panel_w - 8, 33), (80, 80, 80), 1)

        # each keypoint: "P1  X  1234  Y  -567  Z  2100  sim=0.92"
        for i, kp in enumerate(keypoints):
            a3d = kp.get("anchor_3d", [0, 0, 0])
            x, y, z = int(round(a3d[0])), int(round(a3d[1])), int(round(a3d[2]))
            sim = kp.get("similarity", 0)

            y0 = 50 + i * line_h
            cv2.putText(viz, f"P{i+1}", (16, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, DOT_COLOR, 1, cv2.LINE_AA)
            cv2.putText(viz, f"X {x:>6}  Y {y:>6}  Z {z:>6}  sim={sim:.2f}",
                        (42, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, LEGEND_FG, 1, cv2.LINE_AA)

    # ── 3. Bottom-right: frame info ──
    stem = result.get("stem", "")
    cluster = result.get("cluster", "")
    n_kp = len(keypoints)
    cv2.putText(viz, f"DINOv2 | {cluster} | {stem}  ({n_kp} pts)",
                (w - 310, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, viz)
    return output_path


def render_all_visualizations(all_results, data_dir, output_dir, max_frames=None):
    """为所有匹配帧生成可视化，支持限制最大帧数。"""
    data_dir = Path(data_dir)
    viz_dir = os.path.join(output_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    # Collect all matched frames with keypoints
    matched = [(stem, r) for stem, r in all_results.items() if r["num_keypoints"] > 0]
    n_matched = len(matched)

    if n_matched == 0:
        print("  No matched frames to visualize!")
        return viz_dir

    # Sort by stem for consistent ordering
    matched.sort(key=lambda x: x[0])

    if max_frames and max_frames < n_matched:
        # If limited, sample proportionally across clusters
        by_cluster = defaultdict(list)
        for stem, r in matched:
            by_cluster[r["cluster"]].append((stem, r))
        n_clusters = len(by_cluster)
        per_cluster = max(1, max_frames // n_clusters)
        sampled = []
        for cluster_name, items in sorted(by_cluster.items()):
            items.sort(key=lambda x: -max((kp.get("similarity", 0) for kp in x[1]["keypoints"]), default=0))
            sampled.extend(items[:per_cluster])
        matched = sampled[:max_frames]
        print(f"\n[Phase 3] Rendering {len(matched)}/{n_matched} visualizations "
              f"(sampled across {n_clusters} clusters, limit={max_frames})...")
    else:
        print(f"\n[Phase 3] Rendering all {n_matched} matched frames...")

    for stem, result in matched:
        img_path = None
        for ext in [".jpg", ".png"]:
            for cand_name in [stem, stem.replace("TV_", "")]:
                cand = data_dir / f"{cand_name}{ext}"
                if cand.exists():
                    img_path = str(cand)
                    break
            if img_path:
                break
        if img_path is None:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        if img.shape[0] != 480 or img.shape[1] != 640:
            img = cv2.resize(img, (640, 480))

        out_path = os.path.join(viz_dir, f"{stem}_pred.jpg")
        draw_matches(img, result, out_path)

    print(f"  Visualizations saved to {viz_dir}/")
    return viz_dir


# ── Phase 4: 评估指标 ──

def compute_metrics(all_results, match_stats, output_dir):
    """计算评估指标 + 导出 JSON。"""
    print(f"\n[Phase 4] Computing evaluation metrics...")

    results_list = list(all_results.values())
    n_total = len(results_list)
    n_matched = sum(1 for r in results_list if r["num_keypoints"] > 0)
    n_keypoints = sum(r["num_keypoints"] for r in results_list)

    # Similarity + ratio stats
    all_sims = []
    all_ratios = []
    for r in results_list:
        for kp in r.get("keypoints", []):
            all_sims.append(kp.get("similarity", 0))
            all_ratios.append(kp.get("ratio", 0))

    # Keypoint count distribution
    kp_dist = {}
    for r in results_list:
        n = r["num_keypoints"]
        kp_dist[str(n)] = kp_dist.get(str(n), 0) + 1

    # Per-cluster stats
    cluster_stats = {}
    for cluster_name, stats in match_stats.items():
        items = [r for r in results_list if r["cluster"] == cluster_name]
        n_items = len(items)
        n_matched_c = sum(1 for r in items if r["num_keypoints"] > 0)
        n_kp_c = sum(r["num_keypoints"] for r in items)

        sims_c = []
        for r in items:
            for kp in r.get("keypoints", []):
                sims_c.append(kp.get("similarity", 0))

        cluster_stats[cluster_name] = {
            "total_frames": n_items,
            "matched_frames": n_matched_c,
            "match_rate": round(n_matched_c / n_items, 4) if n_items > 0 else 0,
            "total_keypoints": n_kp_c,
            "avg_keypoints_per_frame": round(n_kp_c / n_items, 2) if n_items > 0 else 0,
            "avg_similarity": round(float(np.mean(sims_c)), 4) if sims_c else 0,
            "median_similarity": round(float(np.median(sims_c)), 4) if sims_c else 0,
        }

    metrics = {
        "model": "DINOv2 ViT-S/14",
        "input_size": DINOV2_INPUT_SIZE,
        "confidence_threshold": 0.7,
        "ratio_threshold": 0.90,
        "total_frames": n_total,
        "matched_frames": n_matched,
        "match_rate": round(n_matched / n_total, 4) if n_total > 0 else 0,
        "total_keypoints": n_keypoints,
        "avg_keypoints_per_frame": round(n_keypoints / n_total, 2) if n_total > 0 else 0,
        "keypoint_distribution": kp_dist,
        "avg_similarity": round(float(np.mean(all_sims)), 4) if all_sims else 0,
        "median_similarity": round(float(np.median(all_sims)), 4) if all_sims else 0,
        "similarity_p25": round(float(np.percentile(all_sims, 25)), 4) if all_sims else 0,
        "similarity_p75": round(float(np.percentile(all_sims, 75)), 4) if all_sims else 0,
        "avg_ratio": round(float(np.mean(all_ratios)), 4) if all_ratios else 0,
        "per_cluster": cluster_stats,
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"  Metrics saved: {metrics_path}")

    if n_total == 0:
        print("  No results to summarize.")
        return metrics

    # Print summary
    print(f"\n  {'='*55}")
    print(f"  Evaluation Summary")
    print(f"  {'='*55}")
    print(f"  Total frames:        {n_total}")
    print(f"  Matched frames:      {n_matched} ({100*n_matched/n_total:.1f}%)")
    print(f"  Total keypoints:     {n_keypoints}")
    print(f"  Avg keypoints/frame: {n_keypoints/n_total:.2f}")
    print(f"  Keypoint dist:       {dict(sorted(kp_dist.items(), key=lambda x: int(x[0])))}")
    print(f"  Avg similarity:      {metrics['avg_similarity']:.4f}")
    print(f"  Median similarity:   {metrics['median_similarity']:.4f}")
    print(f"  Avg ratio:           {metrics['avg_ratio']:.4f}")
    print(f"  {'='*55}")
    print(f"  Per-Cluster Breakdown:")
    for name, stats in sorted(cluster_stats.items()):
        print(f"    {name}: match_rate={stats['match_rate']:.2%}, "
              f"kp/frame={stats['avg_keypoints_per_frame']}, "
              f"avg_sim={stats['avg_similarity']:.4f}")

    return metrics


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="DINOv2 Patch 特征匹配 + 全量推理")
    parser.add_argument("--prototype_annotations", default="data/new_sheef/prototype_annotations")
    parser.add_argument("--prototype_data_dir", default="data/new_sheef/prototypes")
    parser.add_argument("--full_data_dir", default="data/new_sheef/pngs")
    parser.add_argument("--output_dir", default="output/shelf_dinov2_match")
    parser.add_argument("--conf_thresh", type=float, default=0.7,
                        help="Minimum cosine similarity for DINOv2 patch matching")
    parser.add_argument("--ratio_thresh", type=float, default=0.90,
                        help="Ratio test upper bound (lower=stricter)")
    parser.add_argument("--no_geometry_filter", action="store_true",
                        help="Disable geometric consistency filter")
    parser.add_argument("--viz_frames", type=int, default=None,
                        help="Max viz frames (default: all matched frames)")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_extract", action="store_true")
    parser.add_argument("--viz_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load DINOv2
    processor, model = load_dinov2(args.device)

    if args.viz_only:
        # Re-load existing results and re-visualize
        full_json = os.path.join(args.output_dir, "all_matches.json")
        if not os.path.exists(full_json):
            print(f"Error: {full_json} not found. Run without --viz_only first.")
            sys.exit(1)
        with open(full_json) as f:
            all_results = json.load(f)
        render_all_visualizations(all_results, args.full_data_dir, args.output_dir, args.viz_frames)
        return

    # Phase 1: Build feature bank from prototype annotations
    feature_npz = os.path.join(args.output_dir, "prototype_features.npz")
    if args.skip_extract and os.path.exists(feature_npz):
        print(f"[Phase 1] --skip_extract: reusing existing feature bank from {feature_npz}")
        print("  NOTE: annotations may have changed; for fresh features, omit --skip_extract")
    feature_bank = build_feature_bank(
        args.prototype_annotations, args.prototype_data_dir,
        processor, model, args.output_dir, args.device,
    )

    # Phase 2: Full dataset matching
    all_results, match_stats = process_full_dataset(
        feature_bank, args.full_data_dir, args.output_dir,
        processor, model, args.device,
        conf_thresh=args.conf_thresh,
        ratio_thresh=args.ratio_thresh,
        enable_geometry_filter=not args.no_geometry_filter,
        max_frames=args.max_frames,
    )

    # Phase 3: Render all visualizations
    render_all_visualizations(all_results, args.full_data_dir, args.output_dir, args.viz_frames)

    # Phase 4: Metrics
    compute_metrics(all_results, match_stats, args.output_dir)

    print(f"\nDone! Output: {args.output_dir}/")
    print(f"  all_matches.json       — full matching results")
    print(f"  prototype_features.npz — DINOv2 feature bank")
    print(f"  viz/*_pred.jpg         — visualizations (all matched frames)")
    print(f"  metrics.json            — evaluation metrics")


if __name__ == "__main__":
    main()
