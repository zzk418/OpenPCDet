#!/usr/bin/env python3
"""
福州现场数据全量推理: YOLO-pose → PCD 点云深度查表 → 3D 锚点。

福州数据的 TV_*.pcd 是相机坐标系稀疏深度点云 (x y z mm, ~20k 点, 覆盖 ~6% 像素),
与 jpg 同帧同内参 (fx=fy=410.9, cx=307, cy=264.3)。pgm 深度图带 ~-1.09m 系统偏差,
不可直接用于锚点查表, 故深度一律取自 PCD。

查表规则: 关键点像素附近自适应窗口 (6px 起, 逐倍扩大至 64px), 取窗口内
PCD 点的 (x,y,z) 中位数, 要求至少 3 个有效点 (z ∈ [0.3, 8] m)。

用法:
  python tools/infer_shelf_anchor_fuzhou.py
"""
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/code/OpenPCDet")
BASE = REPO / "data/new_sheef/福州现场数据"
MODEL_PT = REPO / "output/shelf_pose_train/shelf_v6_s_direct_aug_c2/weights/last.pt"
OUTPUT_DIR = REPO / "output/shelf_pose_inference_v6s_fuzhou_conf50_pcd"

FX, FY, CX, CY = 410.9, 410.9, 307.0, 264.3  # Eagle-M4 Mega 实测内参
IMG_W, IMG_H = 640, 480
CONF = 0.5
Z_MIN_MM, Z_MAX_MM = 300.0, 8000.0
MIN_PTS = 3


def read_pcd(path):
    """PCD v0.7 binary (x y z rgb), 返回 (N,3) float32 mm。"""
    raw = path.read_bytes()
    hdr_end = raw.find(b"DATA binary\n") + len(b"DATA binary\n")
    n = int([l for l in raw[:hdr_end].split(b"\n") if b"POINTS" in l][0].split()[-1])
    arr = np.frombuffer(raw[hdr_end:], dtype=np.uint8, count=n * 16).reshape(n, 16)
    return arr[:, :12].copy().view(np.float32).reshape(n, 3)


def build_projection(xyz):
    """返回 (u, v, xyz_mm) 投影入 640x480 视野且 z 合法 的点。"""
    z = xyz[:, 2]
    ok = (z > Z_MIN_MM) & (z < Z_MAX_MM)
    u = xyz[ok, 0] / z[ok] * FX + CX
    v = xyz[ok, 1] / z[ok] * FY + CY
    in_f = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    return u[in_f], v[in_f], xyz[ok][in_f]


def pcd_lookup_3d(u0, v0, u, v, pts, r0=6, r_max=64):
    """关键点像素附近自适应窗口内 PCD 点的 (x,y,z) 中位数 (mm)。"""
    r = r0
    while r <= r_max:
        m = (np.abs(u - u0) <= r) & (np.abs(v - v0) <= r)
        if m.sum() >= MIN_PTS:
            med = np.median(pts[m], axis=0)
            return [round(float(x), 1) for x in med]
        r *= 2
    return None


def process_frame(model, jpg_path, pcd_path, out_dir, viz_dir, stem):
    img = cv2.imread(str(jpg_path))
    if img is None:
        return None
    if img.shape[:2] != (IMG_H, IMG_W):
        img = cv2.resize(img, (IMG_W, IMG_H))
    try:
        xyz = read_pcd(pcd_path)
    except Exception:
        return None
    u_all, v_all, pts = build_projection(xyz)

    t0 = time.time()
    results = model(img, conf=CONF, verbose=False)
    keypoints = []
    for r in results:
        if r.keypoints is None:
            continue
        t = r.keypoints.data
        if t is None or t.shape[0] == 0:
            continue
        for kp_idx in range(t.shape[1]):
            x, y, c = t[0, kp_idx].tolist()  # 第一实例
            if c < CONF:
                continue
            uk, vk = int(round(x)), int(round(y))
            if not (0 <= uk < IMG_W and 0 <= vk < IMG_H):
                continue
            a3d = pcd_lookup_3d(uk, vk, u_all, v_all, pts)
            if a3d is None:
                continue
            keypoints.append({"pixel_uv": [uk, vk], "anchor_3d": a3d,
                              "confidence": round(c, 3)})
        break

    elapsed = time.time() - t0
    result = {
        "stem": stem,
        "num_keypoints": len(keypoints),
        "keypoints": keypoints,
        "method": "yolo_pose_inference",
        "data_source": "pcd_camera_mm",
        "confidence_threshold": CONF,
        "inference_ms": round(elapsed * 1000),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_dir / f"{stem}_anchor_v2.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # viz: 沿用 infer_shelf_anchor 的绘图
    from infer_shelf_anchor import draw_keypoints
    draw_keypoints(img, keypoints, str(viz_dir / f"{stem}_pred.jpg"))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float, default=CONF)
    parser.add_argument("--model", default=str(MODEL_PT))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    from ultralytics import YOLO

    model_pt = Path(args.model)
    if not model_pt.exists():
        raise SystemExit(f"Model not found: {model_pt}")
    model = YOLO(str(model_pt))
    model.to("cuda:0")

    out_dir = Path(args.output_dir)
    viz_dir = out_dir / "viz"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    all_centers = []
    total_kp = 0
    n_frames = 0
    n_fail = 0
    z_values = []
    t_start = time.time()

    for sub in sorted(d.name for d in BASE.iterdir() if d.is_dir()):
        sub_dir = BASE / sub
        pcds = sorted(sub_dir.glob("TV_*.pcd"))
        n_sub_kp = 0
        for pf in pcds:
            sid = pf.stem.replace("TV_", "")
            stem = f"{sub}_{sid}"
            jpg = sub_dir / f"{sid}.jpg"
            if not jpg.exists():
                n_fail += 1
                continue
            result = process_frame(model, jpg, pf, out_dir, viz_dir, stem)
            if result is None:
                n_fail += 1
                continue
            n_frames += 1
            nk = result["num_keypoints"]
            n_sub_kp += nk
            total_kp += nk
            for kp in result["keypoints"]:
                z_values.append(kp["anchor_3d"][2])

            center_xyz = None
            if nk >= 2:
                a3d = [k["anchor_3d"] for k in result["keypoints"]]
                center_xyz = [round(float(np.mean([a[i] for a in a3d])), 1) for i in range(3)]
            all_centers.append({
                "stem": stem,
                "n_pred": nk,
                "keypoints": result["keypoints"],
                "center_xyz": center_xyz,
            })
        print(f"[{sub}] {len(pcds)} pcd, {n_sub_kp} 关键点")

    elapsed = time.time() - t_start
    print(f"\nDone: {n_frames} success, {n_fail} failed, {total_kp} keypoints ({elapsed:.0f}s)")

    if z_values:
        zs = np.array(z_values)
        print(f"深度分布 (mm): 中位 {np.median(zs):.0f}, p10 {np.percentile(zs,10):.0f}, "
              f"p90 {np.percentile(zs,90):.0f}, min {zs.min():.0f}, max {zs.max():.0f}")

    centers_path = viz_dir / "prediction_centers.json"
    with open(centers_path, "w") as f:
        json.dump(all_centers, f, indent=2, ensure_ascii=False)
    print(f"Centers: {centers_path} ({len(all_centers)} frames)")


if __name__ == "__main__":
    main()
