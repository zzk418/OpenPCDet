#!/usr/bin/env python3
"""
用最优 YOLO-pose 模型重新生成伪标签（伪标签生成唯一入口）。

直接调用 YOLO-pose 推理 + PCD 3D 查表，输出锚点同时含 pixel_uv 和 anchor_3d。

合并来源:
  - regenerate_pseudo_labels.py: 本脚本主体 (最优模型推理 + PCD 3D 查表)
  - model_pseudo_label.py:       v4 旧模型版, 选帧逻辑已被本脚本覆盖, 已删除
  - select_pseudo_labels.py:     DINOv2 聚类选帧, 匹配方案弃用, 已删除

用法:
  python tools/pseudo_labels.py
  python tools/pseudo_labels.py --per_cluster 15 --conf 0.25
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/code/OpenPCDet")
INFERENCE_JSON = REPO / "output/shelf_pose_inference_v4/keypoints/all_keypoints.json"
ASSIGNMENTS = REPO / "data/new_sheef/cluster_assignments.json"
UNIFIED_DIR = REPO / "data/new_sheef/unified"
REVIEWED_DIR = REPO / "datasets/shelf_pose_pseudo/reviewed"
BEST_MODEL = REPO / "output/shelf_pose_train/shelf_reviewed_v4/weights/best.pt"

TARGET_CLUSTERS = {1, 3, 4, 5, 7, 9}

# Camera (640x480 — same as training resolution)
FX, FY = 420.0, 420.0
CX, CY = 307.0, 264.0
IMG_W, IMG_H = 640, 480


# ── PCD / depth utilities (same as infer_shelf_anchor.py) ──

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


def _build_depth_map(xyz):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    valid = z > 1.0
    xv, yv, zv = x[valid], y[valid], z[valid]
    u = np.round(FX * xv / zv + CX).astype(np.int32)
    v = np.round(FY * yv / zv + CY).astype(np.int32)
    in_bounds = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    u, v, zv = u[in_bounds], v[in_bounds], zv[in_bounds]
    dm = np.full((IMG_H, IMG_W), np.nan, dtype=np.float32)
    for i in range(len(u)):
        if np.isnan(dm[v[i], u[i]]) or zv[i] < dm[v[i], u[i]]:
            dm[v[i], u[i]] = zv[i]
    return dm


def _get_anchor_3d(u0, v0, depth_map, window=5):
    h, w = depth_map.shape
    half = window // 2
    u_min, u_max = max(0, u0 - half), min(w, u0 + half + 1)
    v_min, v_max = max(0, v0 - half), min(h, v0 + half + 1)
    wd = depth_map[v_min:v_max, u_min:u_max]
    vd = wd[~np.isnan(wd)]
    if len(vd) == 0:
        for expand in range(half + 1, min(w, h) // 2, 5):
            u2_min, u2_max = max(0, u0 - expand), min(w, u0 + expand + 1)
            v2_min, v2_max = max(0, v0 - expand), min(h, v0 + expand + 1)
            wd2 = depth_map[v2_min:v2_max, u2_min:u2_max]
            vd2 = wd2[~np.isnan(wd2)]
            if len(vd2) > 0:
                vd = vd2
                break
    if len(vd) == 0:
        return None
    z0 = float(np.median(vd))
    x0 = (u0 - CX) * z0 / FX
    y0 = (v0 - CY) * z0 / FY
    return [round(x0, 1), round(y0, 1), round(z0, 1)]


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


# ── Main logic ──

def load_data():
    with open(INFERENCE_JSON) as f:
        inference = json.load(f)
    with open(ASSIGNMENTS) as f:
        assignments = json.load(f)
    return inference, assignments


def get_target_stems(inference, assignments, per_cluster=15, min_kp_conf=0.5, min_box_conf=0.4):
    """Select target stems using v4 inference scores (same as model_pseudo_label.py)."""
    cluster_candidates = {}
    for stem, dets in inference.items():
        cid = int(assignments.get(stem, -1))
        if cid not in TARGET_CLUSTERS:
            continue
        if not dets:
            continue
        best = max(dets, key=lambda d: d["box_conf"])
        if best["box_conf"] < min_box_conf:
            continue
        kps = best["keypoints"]
        valid_kps = [(x, y, c) for x, y, c in kps if c > min_kp_conf]
        if len(valid_kps) < 1:
            continue
        score = best["box_conf"] * sum(c for _, _, c in valid_kps) / len(valid_kps)
        cluster_candidates.setdefault(cid, []).append((stem, score))

    selected_stems = set()
    for cid in sorted(cluster_candidates.keys()):
        candidates = sorted(cluster_candidates[cid], key=lambda x: -x[1])
        n = min(per_cluster, len(candidates))
        for stem, score in candidates[:n]:
            selected_stems.add(stem)
        print(f"  cluster{cid}: {n}/{len(candidates)} frames")

    print(f"\nTotal target stems: {len(selected_stems)}")
    return selected_stems


def infer_frame(model, img_bgr, dm, conf=0.25):
    """Run YOLO-pose on one frame, do PCD lookup for visible keypoints."""
    results = model(img_bgr, conf=conf, verbose=False)
    all_keypoints = []

    for r in results:
        if r.keypoints is None:
            continue
        kp_tensor = r.keypoints.data
        if kp_tensor is None or kp_tensor.shape[0] == 0:
            continue
        for inst_idx in range(kp_tensor.shape[0]):
            instance_kps = []
            for kp_idx in range(kp_tensor.shape[1]):
                x, y, kp_conf = kp_tensor[inst_idx, kp_idx].tolist()
                if kp_conf < conf:
                    continue
                u = int(round(x))
                v = int(round(y))
                if u < 0 or u >= IMG_W or v < 0 or v >= IMG_H:
                    continue
                anchor_3d = _get_anchor_3d(u, v, dm)
                if anchor_3d is None:
                    continue
                instance_kps.append({
                    "id": kp_idx,
                    "pixel_uv": [u, v],
                    "anchor_3d": anchor_3d,
                    "confidence": round(kp_conf, 4),
                })
            if instance_kps:
                all_keypoints.append(instance_kps)

    return all_keypoints[0] if all_keypoints else []


def process_stems(stems, model, conf=0.25):
    """Run inference + PCD lookup on selected stems."""
    os.makedirs(REVIEWED_DIR, exist_ok=True)

    succeed = 0
    fail = 0
    t_start = time.time()

    for stem in sorted(stems):
        pcd_path = UNIFIED_DIR / f"{stem}.pcd"
        img_path = None
        for ext in (".jpg", ".png"):
            cand = UNIFIED_DIR / f"{stem}{ext}"
            if cand.exists():
                img_path = cand
                break
        if img_path is None or not pcd_path.exists():
            fail += 1
            print(f"  [{stem}] MISSING FILES")
            continue

        t0 = time.time()

        img = cv2.imread(str(img_path))
        if img is None:
            fail += 1
            continue
        if img.shape[:2] != (IMG_H, IMG_W):
            img = cv2.resize(img, (IMG_W, IMG_H))

        xyz = _read_pcd_binary(str(pcd_path))
        dm = _build_depth_map(xyz)

        keypoints = infer_frame(model, img, dm, conf)

        elapsed = time.time() - t0

        result = {
            "stem": stem,
            "num_keypoints": len(keypoints),
            "keypoints": keypoints,
            "method": "yolo_pose_best_model",
            "model": str(BEST_MODEL.name),
            "confidence_threshold": conf,
            "inference_ms": round(elapsed * 1000),
            "reviewed": False,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        out_json = REVIEWED_DIR / f"{stem}_anchor_v2.json"
        with open(out_json, "w") as f:
            json.dump(_to_native(result), f, indent=2, ensure_ascii=False)

        n_kp = len(keypoints)
        kp_str = ", ".join(
            f"[{k['anchor_3d'][0]:.0f},{k['anchor_3d'][1]:.0f},{k['anchor_3d'][2]:.0f}]"
            for k in keypoints
        )
        print(f"  [{stem}] {n_kp} kp: {kp_str} ({elapsed*1000:.0f}ms)")
        succeed += 1

    total_elapsed = time.time() - t_start
    print(f"\nDone: {succeed} success, {fail} failed ({total_elapsed:.1f}s)")
    print(f"Output: {REVIEWED_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Re-generate pseudo-labels with best YOLO-pose model")
    parser.add_argument("--per_cluster", type=int, default=15)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not BEST_MODEL.exists():
        print(f"Best model not found: {BEST_MODEL}")
        sys.exit(1)

    print("Loading v4 inference + cluster assignments...")
    inference, assignments = load_data()

    print(f"\nSelecting stems (top {args.per_cluster} per cluster)...")
    stems = get_target_stems(inference, assignments, per_cluster=args.per_cluster)

    print(f"\nLoading YOLO-pose model: {BEST_MODEL}")
    from ultralytics import YOLO
    model = YOLO(str(BEST_MODEL))
    if args.device.isdigit():
        model.to(int(args.device))
    else:
        model.to(args.device)

    print(f"\nRunning inference on {len(stems)} frames (conf={args.conf})...")
    process_stems(stems, model, conf=args.conf)

    print(f"\nDone! Start review:")
    print(f"  python tools/shelf_anchor_web.py \\")
    print(f"    --data_dir data/new_sheef/pseudo_review_90 \\")
    print(f"    --output_dir datasets/shelf_pose_pseudo/reviewed")


if __name__ == "__main__":
    main()
