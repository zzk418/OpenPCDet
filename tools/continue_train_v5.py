#!/usr/bin/env python3
"""
用新审核数据继续训练最优 YOLO-pose 模型 → 推理 → 可视化。

流程:
  1. 仅加载 reviewed=true 的帧
  2. 构建 YOLO 数据集 + 离线增强
  3. 从 v4 checkpoint 微调 (低 LR, 少 epoch)
  4. 验证集推理 + 深度图例可视化 (两点中心 XYZ)

用法:
  python tools/continue_train_v5.py
  python tools/continue_train_v5.py --epochs 30 --aug_per_img 30
"""

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/code/OpenPCDet")
REVIEWED_DIR = REPO / "datasets/shelf_pose_pseudo/reviewed"
FULL_DATA = REPO / "data/new_sheef/pngs"
UNIFIED_DIR = REPO / "data/new_sheef/unified"
BEST_MODEL = REPO / "output/shelf_pose_train/shelf_reviewed_v4/weights/best.pt"
OUTPUT_NAME = "shelf_reviewed_v5"

# Camera
FX, FY = 420.0, 420.0
CX, CY = 307.0, 264.0
IMG_W, IMG_H = 640, 480


# ═══════════════════════════════════════════════════════════
# Step 1: 加载审核数据 (仅 reviewed=true)
# ═══════════════════════════════════════════════════════════

def load_reviewed(reviewed_dir):
    """仅加载 human-reviewed 且 confirmed 的帧。"""
    samples = []
    skipped = 0
    pending = 0
    for f in sorted(Path(reviewed_dir).glob("*_anchor_v2.json")):
        d = json.load(open(f))
        if d.get("skipped"):
            skipped += 1
            continue
        if not d.get("reviewed"):
            pending += 1
            continue
        kps_raw = d.get("keypoints", [])
        kps = [(kp["pixel_uv"][0], kp["pixel_uv"][1]) for kp in kps_raw]
        samples.append({"stem": d["stem"], "keypoints": kps})
    print(f"Loaded {len(samples)} reviewed frames (skipped={skipped}, pending={pending})")
    return samples


# ═══════════════════════════════════════════════════════════
# Step 2: YOLO 数据集
# ═══════════════════════════════════════════════════════════

def keypoints_to_yolo(kps, img_w=640, img_h=480, K=2):
    n = len(kps)
    if n < 1:
        return None
    kps_sorted = sorted(kps, key=lambda k: k[0])
    us = [k[0] / img_w for k in kps_sorted]
    vs = [k[1] / img_h for k in kps_sorted]
    cx = (min(us) + max(us)) / 2
    cy = (min(vs) + max(vs)) / 2
    bw = min(max(max(us) - min(us), 0.05) * 1.2, 1.0)
    bh = min(max(max(vs) - min(vs), 0.05) * 1.2, 1.0)

    slots = [0] if n == 1 else np.round(np.linspace(0, K - 1, n)).astype(int).tolist()
    slot_set = set(slots)
    slot_map = {s: i for i, s in enumerate(slots)}

    norm = []
    for i in range(K):
        if i in slot_set:
            u, v = kps_sorted[slot_map[i]]
            norm.extend([u / img_w, v / img_h, 2.0])
        else:
            norm.extend([0.0, 0.0, 0.0])

    return (f"0.000000 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " +
            " ".join(f"{x:.6f}" for x in norm))


def build_dataset(samples, output_dir, val_ratio=0.15, seed=42):
    random.seed(seed)
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (output_dir / sub).mkdir(parents=True)

    valid = []
    for s in samples:
        stem = s["stem"]
        img_path = None
        # Try unified first (new data), then pngs
        for data_dir in [UNIFIED_DIR, FULL_DATA]:
            for ext in [".jpg", ".png"]:
                cand = data_dir / f"{stem}{ext}"
                if cand.exists():
                    img_path = cand
                    break
            if img_path:
                break
        if img_path is None:
            print(f"  SKIP {stem}: no image")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if img.shape[:2] != (IMG_H, IMG_W):
            img = cv2.resize(img, (IMG_W, IMG_H))

        label = keypoints_to_yolo(s["keypoints"])
        if label is None:
            continue
        valid.append({"stem": stem, "img": img, "label": label})

    random.shuffle(valid)
    n_val = max(1, int(len(valid) * val_ratio))

    for split, items in [("train", valid[n_val:]), ("val", valid[:n_val])]:
        kp_total = 0
        for item in items:
            cv2.imwrite(str(output_dir / "images" / split / f"{item['stem']}.jpg"),
                        item["img"], [cv2.IMWRITE_JPEG_QUALITY, 90])
            with open(output_dir / "labels" / split / f"{item['stem']}.txt", "w") as f:
                f.write(item["label"] + "\n")
            kp_total += 1
        print(f"  [{split}] {len(items)} samples")

    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(f"""# V5 continued training dataset
path: {output_dir.resolve()}
train: images/train
val: images/val
kpt_shape: [2, 3]
flip_idx: [0, 1]
names:
  0: shelf
nc: 1
""")
    print(f"Dataset: {output_dir} (train={len(valid)-n_val}, val={n_val})")
    return yaml_path


# ═══════════════════════════════════════════════════════════
# Step 3: 增强
# ═══════════════════════════════════════════════════════════

def augment_single(img, kps):
    h, w = img.shape[:2]
    scale = np.random.uniform(0.8, 1.2)
    angle = np.random.uniform(-15, 15)
    tx = np.random.uniform(-0.1, 0.1) * w
    ty = np.random.uniform(-0.1, 0.1) * h
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    aug_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    aug_kps = []
    for (u, v) in kps:
        pt = np.array([u, v, 1.0])
        new_pt = M @ pt
        aug_kps.append((new_pt[0], new_pt[1]))

    aug_img = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    aug_img[:, :, 0] = np.clip(aug_img[:, :, 0] + np.random.uniform(-10, 10), 0, 179)
    aug_img[:, :, 1] = np.clip(aug_img[:, :, 1] * np.random.uniform(0.7, 1.3), 0, 255)
    aug_img[:, :, 2] = np.clip(aug_img[:, :, 2] * np.random.uniform(0.7, 1.3), 0, 255)
    aug_img = cv2.cvtColor(aug_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

    alpha = np.random.uniform(0.85, 1.15)
    beta = np.random.randint(-15, 15)
    aug_img = np.clip(alpha * aug_img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    if np.random.random() < 0.2:
        ksize = np.random.choice([3, 5])
        angle_blur = np.random.randint(0, 360)
        kernel = np.zeros((ksize, ksize))
        kernel[int((ksize - 1) / 2), :] = np.ones(ksize)
        kernel = cv2.warpAffine(kernel,
                                cv2.getRotationMatrix2D((ksize / 2, ksize / 2), angle_blur, 1),
                                (ksize, ksize))
        kernel = kernel / kernel.sum()
        aug_img = cv2.filter2D(aug_img, -1, kernel)

    if np.random.random() < 0.2:
        noise = np.random.normal(0, np.random.uniform(5, 20), aug_img.shape)
        aug_img = np.clip(aug_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return aug_img, aug_kps


def augment_dataset(input_dir, output_dir, aug_per_img=40, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (output_dir / sub).mkdir(parents=True)

    for split in ["train", "val"]:
        img_dir = input_dir / "images" / split
        lbl_dir = input_dir / "labels" / split
        if not img_dir.exists():
            continue

        all_samples = []
        for img_path in sorted(img_dir.glob("*.jpg")):
            stem = img_path.stem
            lbl_path = lbl_dir / f"{stem}.txt"
            if not lbl_path.exists():
                continue
            img = cv2.imread(str(img_path))
            label_line = lbl_path.read_text().strip()
            parts = label_line.split()
            kps = []
            for i in range(2):
                x = float(parts[5 + i * 3])
                y = float(parts[5 + i * 3 + 1])
                v = float(parts[5 + i * 3 + 2])
                if v > 0:
                    kps.append((x * IMG_W, y * IMG_H))
            all_samples.append((img.copy(), kps.copy(), f"{stem}_orig"))
            for i in range(aug_per_img):
                aug_img, aug_kps = augment_single(img, kps)
                valid_kps = [(u, v) for (u, v) in aug_kps if 0 <= u < IMG_W and 0 <= v < IMG_H]
                if len(valid_kps) >= 1:
                    all_samples.append((aug_img, valid_kps, f"{stem}_aug{i:03d}"))

        print(f"  [{split}] {len(all_samples)} samples (orig + {aug_per_img}x aug)")
        for a_img, a_kps, a_name in all_samples:
            label = keypoints_to_yolo(a_kps)
            if label is None:
                continue
            cv2.imwrite(str(output_dir / "images" / split / f"{a_name}.jpg"),
                        a_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with open(output_dir / "labels" / split / f"{a_name}.txt", "w") as f:
                f.write(label + "\n")

    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(f"""# V5 augmented dataset
path: {output_dir.resolve()}
train: images/train
val: images/val
kpt_shape: [2, 3]
flip_idx: [0, 1]
names:
  0: shelf
nc: 1
""")
    print(f"Augmented: {output_dir}")
    return yaml_path


# ═══════════════════════════════════════════════════════════
# Step 4: 微调
# ═══════════════════════════════════════════════════════════

def train_yolo(data_yaml, resume_ckpt, output_name, epochs=30):
    from ultralytics import YOLO

    print(f"\nFine-tuning from: {resume_ckpt}")
    print(f"  Data: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  LR: 0.0005 (lower for fine-tuning)")

    model = YOLO(str(resume_ckpt))
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=8,
        imgsz=640,
        lr0=0.0005,       # lower LR for fine-tuning
        lrf=0.01,
        freeze=0,
        device="0",
        name=output_name,
        project=str(REPO / "output/shelf_pose_train"),
        patience=15,
        workers=2,
        kobj=5.0,
        cls=1.0,
        box=5.0,
        pose=15.0,
        hsv_h=0.01,
        hsv_s=0.2,
        hsv_v=0.15,
        degrees=3.0,
        translate=0.03,
        scale=0.15,
        shear=0.5,
        fliplr=0.0,
        mosaic=0.3,
        nbs=64,
        warmup_epochs=2,
        cos_lr=True,
        close_mosaic=10,
        plots=True,
        save=True,
        exist_ok=True,
    )

    best = REPO / "output/shelf_pose_train" / output_name / "weights/best.pt"
    print(f"\nDone! Best model: {best}")
    return best


# ═══════════════════════════════════════════════════════════
# Step 5: 推理 + 可视化 (深度图例)
# ═══════════════════════════════════════════════════════════

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
    return [x0, y0, z0]


def infer_and_visualize(model_path, stems, output_dir):
    """Run inference on selected stems and save visualized results."""
    from ultralytics import YOLO

    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(str(model_path))
    model.to("cuda:0")

    DOT_COLOR = (0, 0, 255)        # red
    GT_COLOR = (80, 80, 80)        # dark gray for GT
    PANEL_BG = (35, 35, 45)
    PANEL_FG = (220, 220, 220)
    PANEL_ACCENT = (80, 210, 255)

    results_summary = []
    t_start = time.time()

    for stem in sorted(stems):
        # Find files
        pcd_path = UNIFIED_DIR / f"{stem}.pcd"
        img_path = None
        for data_dir in [UNIFIED_DIR, FULL_DATA]:
            for ext in [".jpg", ".png"]:
                cand = data_dir / f"{stem}{ext}"
                if cand.exists():
                    img_path = cand
                    break
            if img_path:
                break
        if img_path is None or not pcd_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if img.shape[:2] != (IMG_H, IMG_W):
            img = cv2.resize(img, (IMG_W, IMG_H))
        viz = img.copy()

        xyz = _read_pcd_binary(str(pcd_path))
        dm = _build_depth_map(xyz)

        # YOLO inference
        results = model(img, conf=0.25, verbose=False)
        pred_kps_3d = []

        for r in results:
            if r.keypoints is None:
                continue
            kp_tensor = r.keypoints.data
            if kp_tensor is None or kp_tensor.shape[0] == 0:
                continue
            for inst_idx in range(kp_tensor.shape[0]):
                for kp_idx in range(kp_tensor.shape[1]):
                    x, y, kp_conf = kp_tensor[inst_idx, kp_idx].tolist()
                    if kp_conf < 0.25:
                        continue
                    u, v = int(round(x)), int(round(y))
                    if u < 0 or u >= IMG_W or v < 0 or v >= IMG_H:
                        continue
                    a3d = _get_anchor_3d(u, v, dm)
                    if a3d is None:
                        continue
                    pred_kps_3d.append({"pixel_uv": [u, v], "anchor_3d": a3d, "confidence": kp_conf})

        # Load GT keypoints
        gt_kps = []
        gt_json = REVIEWED_DIR / f"{stem}_anchor_v2.json"
        if gt_json.exists():
            d = json.load(open(gt_json))
            if d.get("reviewed") and not d.get("skipped"):
                for kp in d.get("keypoints", []):
                    uv = kp.get("pixel_uv", [0, 0])
                    a3d = kp.get("anchor_3d")
                    gt_kps.append({"pixel_uv": [int(uv[0]), int(uv[1])], "anchor_3d": a3d})

        # ── Draw ──
        # GT: gray dashed cross
        for kp in gt_kps:
            u, v = kp["pixel_uv"]
            cv2.line(viz, (u - 8, v), (u + 8, v), GT_COLOR, 1, cv2.LINE_AA)
            cv2.line(viz, (u, v - 8), (u, v + 8), GT_COLOR, 1, cv2.LINE_AA)
            cv2.circle(viz, (u, v), 6, GT_COLOR, 1, cv2.LINE_AA)
            cv2.putText(viz, "GT", (u + 10, v - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, GT_COLOR, 1, cv2.LINE_AA)

        # Pred: red cross + dot
        for i, kp in enumerate(pred_kps_3d):
            u, v = kp["pixel_uv"]
            cv2.line(viz, (u - 7, v), (u + 7, v), DOT_COLOR, 1, cv2.LINE_AA)
            cv2.line(viz, (u, v - 7), (u, v + 7), DOT_COLOR, 1, cv2.LINE_AA)
            cv2.circle(viz, (u, v), 5, DOT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(viz, (u, v), 5, (30, 30, 30), 1, cv2.LINE_AA)

        # ── Depth legend (top-left panel) ──
        all_3d = []
        for kp in pred_kps_3d:
            all_3d.append(kp["anchor_3d"])

        if all_3d:
            n_kp = len(all_3d)
            line_h = 16
            panel_w = 250
            panel_h = 32 + n_kp * line_h + 6 + line_h + 4  # +1 line for center

            # semi-transparent bg
            overlay = viz.copy()
            cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), PANEL_BG, -1)
            cv2.addWeighted(overlay, 0.78, viz, 0.22, 0, viz)
            cv2.rectangle(viz, (8, 8), (8 + panel_w, 8 + panel_h), (100, 100, 100), 1)

            # title
            stem_short = stem.replace("TV_", "")
            cv2.putText(viz, f"V5 Pred  |  {stem_short}  |  {n_kp} pts", (16, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, PANEL_ACCENT, 1, cv2.LINE_AA)
            cv2.line(viz, (16, 33), (8 + panel_w - 8, 33), (80, 80, 80), 1)

            # each keypoint
            for i, a3d in enumerate(all_3d):
                y0 = 50 + i * line_h
                cv2.putText(viz, f"P{i+1}", (16, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, DOT_COLOR, 1, cv2.LINE_AA)
                cv2.putText(viz, f"X {a3d[0]:7.0f}  Y {a3d[1]:7.0f}  Z {a3d[2]:7.0f}",
                            (42, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, PANEL_FG, 1, cv2.LINE_AA)

            # Center point of all keypoints
            if n_kp >= 2:
                cx = np.mean([a[0] for a in all_3d])
                cy = np.mean([a[1] for a in all_3d])
                cz = np.mean([a[2] for a in all_3d])
                y0 = 50 + n_kp * line_h + 4
                cv2.line(viz, (16, y0 - 2), (8 + panel_w - 8, y0 - 2), (60, 60, 70), 1)
                cv2.putText(viz, "CENTER", (16, y0 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, PANEL_ACCENT, 1, cv2.LINE_AA)
                cv2.putText(viz, f"X {cx:7.0f}  Y {cy:7.0f}  Z {cz:7.0f}",
                            (72, y0 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 100), 1, cv2.LINE_AA)

                # Also draw center on image as yellow diamond
                # Project center back to image
                if cz > 0:
                    u_center = int(CX + FX * cx / cz)
                    v_center = int(CY + FY * cy / cz)
                    if 0 <= u_center < IMG_W and 0 <= v_center < IMG_H:
                        sz = 5
                        pts = np.array([
                            [u_center, v_center - sz],
                            [u_center + sz, v_center],
                            [u_center, v_center + sz],
                            [u_center - sz, v_center],
                        ], np.int32)
                        cv2.polylines(viz, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)

        # Save
        out_path = os.path.join(output_dir, f"{stem}_pred.jpg")
        cv2.imwrite(out_path, viz, [cv2.IMWRITE_JPEG_QUALITY, 92])

        summary = {
            "stem": stem,
            "n_pred_keypoints": len(pred_kps_3d),
            "n_gt_keypoints": len(gt_kps),
            "pred_kps": pred_kps_3d,
        }
        if len(pred_kps_3d) >= 2:
            summary["center_xyz"] = [round(float(np.mean([a["anchor_3d"][i] for a in pred_kps_3d])), 1) for i in range(3)]
        results_summary.append(summary)

    total_elapsed = time.time() - t_start
    print(f"\nVisualized {len(results_summary)} frames ({total_elapsed:.1f}s)")
    print(f"Output: {output_dir}/")

    # Save summary JSON
    summary_path = os.path.join(output_dir, "prediction_centers.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"Centers: {summary_path}")

    return results_summary


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="V5: Continue train + infer + visualize")
    parser.add_argument("--epochs", type=int, default=30, help="Fine-tuning epochs")
    parser.add_argument("--aug_per_img", type=int, default=40)
    parser.add_argument("--output_dir", default=str(REPO / "datasets/shelf_pose_v5_base"))
    parser.add_argument("--aug_dir", default=str(REPO / "datasets/shelf_pose_v5_aug"))
    parser.add_argument("--viz_dir", default=str(REPO / "output/shelf_pose_train/shelf_reviewed_v5/viz"))
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_viz", action="store_true")
    args = parser.parse_args()

    if not BEST_MODEL.exists():
        print(f"Best model not found: {BEST_MODEL}")
        sys.exit(1)

    # Step 1: Load reviewed
    print("=" * 60)
    print("Step 1: Load reviewed (only confirmed)")
    print("=" * 60)
    samples = load_reviewed(REVIEWED_DIR)
    if len(samples) == 0:
        print("ERROR: No reviewed samples!")
        sys.exit(1)

    # Step 2: Build dataset
    print("\n" + "=" * 60)
    print("Step 2: Build YOLO dataset")
    print("=" * 60)
    yaml_path = build_dataset(samples, args.output_dir)

    # Step 3: Augment
    print("\n" + "=" * 60)
    print(f"Step 3: Augment ({args.aug_per_img}x per image)")
    print("=" * 60)
    aug_yaml = augment_dataset(args.output_dir, args.aug_dir, args.aug_per_img)

    if args.skip_train:
        print(f"\nDone. Augmented dataset: {args.aug_dir}")
        return

    # Step 4: Train
    print("\n" + "=" * 60)
    print(f"Step 4: Fine-tune from V4 checkpoint ({args.epochs} epochs)")
    print("=" * 60)
    best_ckpt = train_yolo(aug_yaml, BEST_MODEL, OUTPUT_NAME, args.epochs)

    if args.skip_viz:
        print(f"\nDone! Model: {best_ckpt}")
        return

    # Step 5: Visualize
    print("\n" + "=" * 60)
    print("Step 5: Inference + Visualization (with depth legend)")
    print("=" * 60)
    val_stems = [s["stem"] for s in samples]
    infer_and_visualize(best_ckpt, val_stems, args.viz_dir)

    print(f"\n{'=' * 60}")
    print(f"All done!")
    print(f"  Model: {best_ckpt}")
    print(f"  Viz:   {args.viz_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
