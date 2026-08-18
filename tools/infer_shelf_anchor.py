#!/usr/bin/env python3
"""
YOLO-Pose inference → PCD 3D lookup → save results.

Pipeline:
  1. YOLO-pose predicts K keypoints on RGB image (normalized 0-1)
  2. Denormalize to 640×480 pixel coords
  3. For each visible keypoint: 5×5 median PCD depth lookup → (x, y, z)
  4. Save results as _anchor_v2.json (compatible with web review tool)

Usage:
  # Single frame
  python tools/infer_shelf_anchor.py --stem TV_250000036488

  # Batch all frames
  python tools/infer_shelf_anchor.py --batch

  # Different model
  python tools/infer_shelf_anchor.py --model output/shelf_pose_train/shelf_pose/weights/best.pt --batch
"""

import argparse, json, os, sys, time
from pathlib import Path

import cv2
import numpy as np

# ── Reuse V2 core functions (inline to avoid import issues) ──

def _read_pcd_binary(path):
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode().strip()
            if line == "DATA binary":
                break
        dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("rgb", "u4")])
        data = np.frombuffer(f.read(), dtype=dtype)
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    return xyz, None


def _build_depth_map(xyz, img_w=640, img_h=480, fx=410.9, fy=410.9, cx=307.0, cy=264.3):
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


def _get_anchor_3d(u0, v0, depth_map, window=5, fx=410.9, fy=410.9, cx=307.0, cy=264.3):
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
    x0 = (u0 - cx) * z0 / fx
    y0 = (v0 - cy) * z0 / fy
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


# ── Inference ──

def infer_frame(model, img_bgr, xyz, dm, img_w=640, img_h=480,
                 fx=410.9, fy=410.9, cx=307.0, cy=264.3, conf=0.3):
    """Run YOLO-pose on one frame, do PCD lookup for visible keypoints."""
    results = model(img_bgr, conf=conf, verbose=False)

    all_keypoints = []

    for r in results:
        if r.keypoints is None:
            continue

        kp_tensor = r.keypoints.data  # (N, K, 3) — [x, y, conf]
        if kp_tensor is None or kp_tensor.shape[0] == 0:
            continue

        for inst_idx in range(kp_tensor.shape[0]):
            instance_kps = []
            for kp_idx in range(kp_tensor.shape[1]):
                x, y, kp_conf = kp_tensor[inst_idx, kp_idx].tolist()
                if kp_conf < conf:
                    continue  # skip low-confidence keypoints

                u = int(round(x))
                v = int(round(y))
                if u < 0 or u >= img_w or v < 0 or v >= img_h:
                    continue

                anchor_3d = _get_anchor_3d(u, v, dm, fx=fx, fy=fy, cx=cx, cy=cy)
                if anchor_3d is None:
                    continue

                instance_kps.append({
                    "pixel_uv": [u, v],
                    "anchor_3d": anchor_3d,
                    "confidence": round(kp_conf, 3),
                })

            if instance_kps:
                all_keypoints.append(instance_kps)

    # Flatten: take the first instance (should be the only one for single-shelf)
    if all_keypoints:
        return all_keypoints[0]
    return []


def process_frame(stem, model, data_dir, output_dir, img_w, img_h,
                  fx, fy, cx, cy, conf):
    """Process a single frame: load → infer → save."""
    num_id = stem.replace("TV_", "")
    data_dir = Path(data_dir)

    # Find files
    pcd_path = data_dir / f"{stem}.pcd"
    img_path = None
    for ext in [".jpg", ".png"]:
        for cand_name in [stem, num_id, f"TV_{num_id}"]:
            cand = data_dir / f"{cand_name}{ext}"
            if cand.exists():
                img_path = cand
                break
        if img_path:
            break

    if img_path is None or not pcd_path.exists():
        return None

    t0 = time.time()

    # Load
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img_display = cv2.resize(img, (img_w, img_h)) if img.shape[:2] != (img_h, img_w) else img

    xyz, _ = _read_pcd_binary(str(pcd_path))
    dm = _build_depth_map(xyz, img_w, img_h, fx, fy, cx, cy)

    # Infer
    keypoints = infer_frame(model, img_display, xyz, dm, img_w, img_h,
                            fx, fy, cx, cy, conf)

    elapsed = time.time() - t0

    # Save
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "stem": stem,
        "num_keypoints": len(keypoints),
        "keypoints": keypoints,
        "method": "yolo_pose_inference",
        "confidence_threshold": conf,
        "inference_ms": round(elapsed * 1000),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_json = os.path.join(output_dir, f"{stem}_anchor_v2.json")
    with open(out_json, "w") as f:
        json.dump(_to_native(result), f, indent=2, ensure_ascii=False)

    # Viz
    viz_dir = os.path.join(output_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)
    viz_path = os.path.join(viz_dir, f"{stem}_pred.jpg")

    draw_keypoints(img_display, keypoints, viz_path)

    return result


# ── Visualization ──

DOT_COLOR = (0, 0, 255)       # BGR red
LEGEND_BG = (40, 40, 40)       # dark panel
LEGEND_FG = (240, 240, 240)    # light text
LEGEND_ACCENT = (80, 200, 255) # gold accent for title


def draw_keypoints(img, keypoints, output_path, gt_keypoints=None):
    """Draw predicted keypoints with a clean legend panel (v4 style).

    - Small uniform red dots + thin crosshair on the image
    - Center X marker at midpoint of all keypoints
    - Dashed line connecting keypoints
    - Top-left legend panel: numbered list with XYZ + center XYZ
    """
    viz = img.copy()
    h, w = viz.shape[:2]

    # ── 1. Draw keypoints on the image ──
    kp_points = []  # (u, v, anchor_3d) for center calculation
    for i, kp in enumerate(keypoints):
        u, v = kp["pixel_uv"]
        a3d = kp.get("anchor_3d", [0, 0, 0])
        kp_points.append((u, v, a3d))

        # thin crosshair
        cv2.line(viz, (u - 6, v), (u + 6, v), DOT_COLOR, 1)
        cv2.line(viz, (u, v - 6), (u, v + 6), DOT_COLOR, 1)

        # small filled circle + dark outline
        cv2.circle(viz, (u, v), 4, DOT_COLOR, -1)
        cv2.circle(viz, (u, v), 4, (30, 30, 30), 1)

        # tiny number label next to the dot
        cv2.putText(viz, str(i + 1), (u + 7, v - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, DOT_COLOR, 1, cv2.LINE_AA)

    # ── 1b. Dashed line connecting keypoints + center X ──
    has_center = False
    center_u = center_v = 0
    center_xyz = [0, 0, 0]
    if len(kp_points) >= 2:
        # Dashed line between consecutive keypoints
        for i in range(len(kp_points) - 1):
            u1, v1, _ = kp_points[i]
            u2, v2, _ = kp_points[i + 1]
            # Draw dashed line: series of short segments
            dist = np.sqrt((u2 - u1)**2 + (v2 - v1)**2)
            n_segs = max(2, int(dist / 8))
            for s in range(0, n_segs, 2):
                t0 = s / n_segs
                t1 = min((s + 1) / n_segs, 1.0)
                su0, sv0 = int(u1 + (u2 - u1) * t0), int(v1 + (v2 - v1) * t0)
                su1, sv1 = int(u1 + (u2 - u1) * t1), int(v1 + (v2 - v1) * t1)
                cv2.line(viz, (su0, sv0), (su1, sv1), DOT_COLOR, 1, cv2.LINE_AA)

        # Center point
        all_3d = [p[2] for p in kp_points]
        center_xyz = [np.mean([a[i] for a in all_3d]) for i in range(3)]
        # Project center back to image (average UV)
        center_u = int(np.mean([p[0] for p in kp_points]))
        center_v = int(np.mean([p[1] for p in kp_points]))
        has_center = True

        # X marker at center (yellow)
        x_sz = 5
        cv2.line(viz, (center_u - x_sz, center_v - x_sz),
                 (center_u + x_sz, center_v + x_sz), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(viz, (center_u + x_sz, center_v - x_sz),
                 (center_u - x_sz, center_v + x_sz), (0, 255, 255), 2, cv2.LINE_AA)

    # ── 2. Legend panel (top-left) ──
    if keypoints:
        n_kp = len(keypoints)
        line_h = 16
        panel_w = 220
        # Extra row for center if available
        extra_rows = 1 if has_center else 0
        panel_h = 32 + (n_kp + extra_rows) * line_h + 6

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

        # each keypoint: "P1  X  1234  Y  -567  Z  2100"
        for i, kp in enumerate(keypoints):
            a3d = kp.get("anchor_3d", [0, 0, 0])
            x, y, z = int(round(a3d[0])), int(round(a3d[1])), int(round(a3d[2]))

            y0 = 50 + i * line_h
            cv2.putText(viz, f"P{i+1}", (16, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, DOT_COLOR, 1, cv2.LINE_AA)
            cv2.putText(viz, f"X {x:>6}  Y {y:>6}  Z {z:>6}",
                        (42, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, LEGEND_FG, 1, cv2.LINE_AA)

        # Center point row
        if has_center:
            cx, cy, cz = int(round(center_xyz[0])), int(round(center_xyz[1])), int(round(center_xyz[2]))
            y0 = 50 + n_kp * line_h + 2
            cv2.line(viz, (16, y0 - 2), (8 + panel_w - 8, y0 - 2), (60, 60, 70), 1)
            cv2.putText(viz, "C", (16, y0 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(viz, f"X {cx:>6}  Y {cy:>6}  Z {cz:>6}",
                        (42, y0 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 150), 1, cv2.LINE_AA)

    # ── 3. Bottom-right: frame info ──
    cv2.putText(viz, f"YOLO-Pose | {len(keypoints)} corners",
                (w - 240, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, viz)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="YOLO-Pose inference → PCD 3D lookup")
    parser.add_argument("--model", default="output/shelf_pose_train/shelf_pose/weights/best.pt")
    parser.add_argument("--data_dir", default="data/new_sheef/pngs")
    parser.add_argument("--output_dir", default="output/shelf_anchor_v2_pred")
    parser.add_argument("--stem", default=None)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--device", default="cuda:0")
    # Camera
    parser.add_argument("--fx", type=float, default=410.9)
    parser.add_argument("--fy", type=float, default=410.9)
    parser.add_argument("--cx", type=float, default=307.0)
    parser.add_argument("--cy", type=float, default=264.3)
    parser.add_argument("--img_w", type=int, default=640)
    parser.add_argument("--img_h", type=int, default=480)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        print("Train first: python tools/train_shelf_pose.py")
        sys.exit(1)

    from ultralytics import YOLO
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    if args.device.isdigit():
        model.to(int(args.device))
    else:
        model.to(args.device)

    data_dir = Path(args.data_dir)

    if args.stem:
        stems = [args.stem]
    elif args.batch:
        # 全量处理 (2026-08-18 起不再排除 TV_250000000001 outlier 帧; 兼容无 TV_ 前缀的
        # 0807/0811/0812 现场批次, 如 250000010477.pcd)
        stems = sorted(p.stem for p in data_dir.glob("*.pcd"))
        print(f"Batch processing {len(stems)} frames...")
    else:
        print("Specify --stem or --batch")
        sys.exit(1)

    success = 0
    fail = 0
    total_kp = 0
    t_start = time.time()
    all_centers = []  # accumulate for prediction_centers.json

    for stem in stems:
        result = process_frame(
            stem, model, args.data_dir, args.output_dir,
            args.img_w, args.img_h, args.fx, args.fy, args.cx, args.cy, args.conf,
        )
        if result is None:
            fail += 1
            print(f"  [{stem}] FAILED")
        else:
            n_kp = result["num_keypoints"]
            total_kp += n_kp
            success += 1
            kp_str = ", ".join(
                f"[{k['anchor_3d'][0]:.0f},{k['anchor_3d'][1]:.0f},{k['anchor_3d'][2]:.0f}]"
                for k in result["keypoints"]
            )
            print(f"  [{stem}] {n_kp} keypoints: {kp_str} ({result['inference_ms']}ms)")

            # Collect center data
            kps = result["keypoints"]
            center_xyz = None
            if len(kps) >= 2:
                all_3d = [k["anchor_3d"] for k in kps if k.get("anchor_3d")]
                if all_3d:
                    center_xyz = [round(float(np.mean([a[i] for a in all_3d])), 1) for i in range(3)]
            all_centers.append({
                "stem": stem,
                "n_pred": n_kp,
                "keypoints": [{"pixel_uv": k["pixel_uv"], "anchor_3d": k.get("anchor_3d"),
                               "confidence": k.get("confidence")} for k in kps],
                "center_xyz": center_xyz,
            })

    elapsed = time.time() - t_start
    print(f"\nDone: {success} success, {fail} failed, {total_kp} keypoints total")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(stems)*1000:.0f}ms/frame)")
    print(f"Output: {args.output_dir}/")

    # Write prediction_centers.json
    if all_centers:
        viz_dir = os.path.join(args.output_dir, "viz")
        os.makedirs(viz_dir, exist_ok=True)
        centers_path = os.path.join(viz_dir, "prediction_centers.json")
        with open(centers_path, "w") as f:
            json.dump(all_centers, f, indent=2, ensure_ascii=False)
        print(f"Centers: {centers_path} ({len(all_centers)} frames)")


if __name__ == "__main__":
    main()
