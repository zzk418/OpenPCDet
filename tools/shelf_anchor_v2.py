#!/usr/bin/env python3
"""
Single-Anchor 3D Prompting (单主角点 3D 极速标注) — V2 MVP
==========================================================
货架识别 v2 方案: 仅需定位 1 个核心主角点 (如左前下角)，2D RGB 快速找点 + PCD 邻域中位数深度滤波。

管线:
  1. 加载对齐的 RGB + PCD
  2. 2D Prompt: 手动点击 / YOLO 检测框顶点 → (u0, v0)
  3. get_anchor_3d(u0, v0): 5×5 窗口深度中位数 → (x0, y0, z0)
  4. 输出 anchor_3d + 可视化

用法:
  # 交互式点击获取 anchor
  python shelf_anchor_v2.py \
      --pcd data/new_sheef/pngs/TV_250000036488.pcd \
      --img data/new_sheef/pngs/250000036488.png \
      --mode click

  # 自动 YOLO 检测框顶点 (需要 ultralytics)
  python shelf_anchor_v2.py \
      --pcd data/new_sheef/pngs/TV_250000036488.pcd \
      --img data/new_sheef/pngs/250000036488.png \
      --mode yolo

  # 批量处理
  python shelf_anchor_v2.py --data_dir data/new_sheef/pngs --mode yolo
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# PCD I/O
# ══════════════════════════════════════════════════════════════════════════════

def read_pcd_binary(path: str):
    """Read PCD v0.7 binary (FIELDS x y z rgb). Returns xyz (N,3), rgb (N,3) [0,1]."""
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline().decode().strip()
            if line == "DATA binary":
                break
            if line and " " in line:
                parts = line.split(" ", 1)
                header[parts[0]] = parts[1]

        dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("rgb", "u4")])
        num_points = int(header["WIDTH"]) * int(header["HEIGHT"])
        data = np.frombuffer(f.read(), dtype=dtype, count=num_points)

    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    rgb_raw = data["rgb"]
    r = ((rgb_raw >> 16) & 0xFF).astype(np.float32) / 255.0
    g = ((rgb_raw >> 8) & 0xFF).astype(np.float32) / 255.0
    b = (rgb_raw & 0xFF).astype(np.float32) / 255.0
    rgb = np.column_stack([r, g, b]).astype(np.float32)
    return xyz, rgb


# ══════════════════════════════════════════════════════════════════════════════
# 2D ↔ 3D Mapping
# ══════════════════════════════════════════════════════════════════════════════

def build_depth_map(xyz: np.ndarray, img_w: int = 640, img_h: int = 480,
                    fx: float = 410.9, fy: float = 410.9,
                    cx: float = 307.0, cy: float = 264.3) -> np.ndarray:
    """Build dense depth map from unstructured PCD using camera intrinsics.

    Projects each 3D point to pixel (u,v) via pinhole model.
    For pixels with multiple points, keeps the CLOSEST depth (front-most).
    Missing pixels remain NaN.

    Args:
        xyz: (N, 3) float32, 3D points in camera frame (z-forward)
        img_w, img_h: image dimensions
        fx, fy, cx, cy: camera intrinsics
    Returns:
        depth_map: (img_h, img_w) float32 depth in mm, NaN where no data
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # Pinhole projection: u = fx * x/z + cx, v = fy * y/z + cy
    valid = (z > 1.0)  # valid depth (>1mm)
    xv, yv, zv = x[valid], y[valid], z[valid]

    u = np.round(fx * xv / zv + cx).astype(np.int32)
    v = np.round(fy * yv / zv + cy).astype(np.int32)

    # Filter out-of-bounds pixels
    in_bounds = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u, v, zv = u[in_bounds], v[in_bounds], zv[in_bounds]

    # Build depth map (keep closest = smallest z per pixel)
    depth_map = np.full((img_h, img_w), np.nan, dtype=np.float32)
    for i in range(len(u)):
        if np.isnan(depth_map[v[i], u[i]]) or zv[i] < depth_map[v[i], u[i]]:
            depth_map[v[i], u[i]] = zv[i]

    return depth_map


def get_anchor_3d(u0: int, v0: int, depth_map: np.ndarray,
                  window: int = 5,
                  fx: float = 410.9, fy: float = 410.9,
                  cx: float = 307.0, cy: float = 264.3) -> dict:
    """Core V2 algorithm: map 2D anchor pixel to 3D with median depth filter.

    1. Take window×window neighborhood around (u0, v0)
    2. Compute MEDIAN depth in that window (rejects edge-penetrating outliers)
    3. Unproject to 3D: x = (u - cx)*z/fx, y = (v - cy)*z/fy, z = median_depth

    Args:
        u0, v0: anchor pixel coordinates
        depth_map: (H, W) depth image (NaN where no data)
        window: neighborhood size (default 5 → 5×5)
        fx, fy, cx, cy: camera intrinsics
    Returns:
        dict with anchor_3d, median_depth, window_depths, valid_count
    """
    h, w = depth_map.shape
    half = window // 2

    # Extract window
    u_min = max(0, u0 - half)
    u_max = min(w, u0 + half + 1)
    v_min = max(0, v0 - half)
    v_max = min(h, v0 + half + 1)

    window_depths = depth_map[v_min:v_max, u_min:u_max]
    valid_depths = window_depths[~np.isnan(window_depths)]

    if len(valid_depths) == 0:
        # Fallback: search in expanding window
        for expand in range(half + 1, min(w, h) // 2, 5):
            u_min2 = max(0, u0 - expand)
            u_max2 = min(w, u0 + expand + 1)
            v_min2 = max(0, v0 - expand)
            v_max2 = min(h, v0 + expand + 1)
            wd2 = depth_map[v_min2:v_max2, u_min2:u_max2]
            vd2 = wd2[~np.isnan(wd2)]
            if len(vd2) > 0:
                valid_depths = vd2
                break

    if len(valid_depths) == 0:
        return {"anchor_3d": None, "median_depth": None,
                "window_depths": [], "valid_count": 0,
                "error": "No valid depth in neighborhood"}

    median_depth = float(np.median(valid_depths))

    # Unproject to 3D
    x0 = (u0 - cx) * median_depth / fx
    y0 = (v0 - cy) * median_depth / fy
    z0 = median_depth

    return {
        "anchor_3d": [round(x0, 1), round(y0, 1), round(z0, 1)],
        "median_depth": round(median_depth, 1),
        "window_depths": [round(float(d), 1) for d in valid_depths[:20]],
        "valid_count": int(len(valid_depths)),
        "window_total": int(window * window),
        "center_uv": [u0, v0],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2D Prompt Acquisition
# ══════════════════════════════════════════════════════════════════════════════

def review_anchor(img_path: str, depth_map: np.ndarray,
                  fx: float, fy: float, cx: float, cy: float,
                  auto_u: int = None, auto_v: int = None) -> dict:
    """Human-in-the-loop review: model proposes → confirm or correct → PCD lookup.

    Workflow:
      1. Model auto-infers anchor → RED dot displayed on image
      2a. [猜对了] Press ENTER/SPACE to confirm
      2b. [猜偏了] Click the correct corner → GREEN dot replaces it
      3a. Press ENTER/SPACE to confirm the corrected point
      3b. ESC to skip this frame
      4. PCD 5×5 median depth lookup → precise 3D anchor saved

    Args:
        auto_u, auto_v: auto-detected anchor pixel (from depth heuristic or YOLO)
    """
    import cv2 as _cv2
    img = _cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    h, w = img.shape[:2]

    if auto_u is None or auto_v is None:
        auto_u, auto_v = _depth_based_anchor(depth_map)

    anchor_uv = [auto_u, auto_v]  # mutable: can be updated by click
    result = {}
    confirmed = False

    def redraw():
        """Draw the current anchor + 3D preview."""
        nonlocal result
        disp = img.copy()
        u, v = anchor_uv

        # 5×5 window
        half = 2
        _cv2.rectangle(disp, (u - half, v - half), (u + half, v + half),
                       (0, 255, 255), 1)

        # Live PCD lookup preview
        r = get_anchor_3d(u, v, depth_map, window=5, fx=fx, fy=fy, cx=cx, cy=cy)

        if r.get("anchor_3d"):
            a3 = r["anchor_3d"]
            status = "✓ CONFIRMED" if confirmed else "□ Review"
            color = (0, 255, 0) if confirmed else (0, 0, 255)  # green=done, red=proposed
            _cv2.circle(disp, (u, v), 6, color, -1)
            _cv2.circle(disp, (u, v), 10, color, 2)
            info = (f"[{status}] 3D=({a3[0]:.0f},{a3[1]:.0f},{a3[2]:.0f})mm "
                    f"| depth={r['median_depth']:.0f}mm "
                    f"| valid={r['valid_count']}/{r['window_total']}")
            _cv2.putText(disp, info, (10, h - 10),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        else:
            _cv2.circle(disp, (u, v), 6, (0, 0, 255), -1)
            _cv2.putText(disp, "NO DEPTH — click another point",
                         (10, h - 10), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Status bar
        bar = np.zeros((40, w, 3), dtype=np.uint8)
        hints = "ENTER=Confirm | Click=Correct | ESC=Skip | S=Save JSON"
        _cv2.putText(bar, hints, (10, 25), _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        disp = np.vstack([disp, bar])
        _cv2.imshow("Shelf Anchor V2 — Review", disp)

    def on_mouse(event, x, y, flags, param):
        nonlocal confirmed
        if confirmed:
            return  # locked after confirm
        if event == _cv2.EVENT_LBUTTONDOWN and y < h:
            anchor_uv[0] = x
            anchor_uv[1] = y
            redraw()

    _cv2.namedWindow("Shelf Anchor V2 — Review", _cv2.WINDOW_NORMAL)
    _cv2.setMouseCallback("Shelf Anchor V2 — Review", on_mouse)
    redraw()

    while True:
        key = _cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC → skip
            break
        elif key in (13, 32):  # ENTER or SPACE → confirm
            if confirmed:
                break  # second ENTER = done
            confirmed = True
            # Final PCD lookup
            u, v = anchor_uv
            result = get_anchor_3d(u, v, depth_map, window=5,
                                   fx=fx, fy=fy, cx=cx, cy=cy)
            result["method"] = "review"
            result["pixel_uv"] = [u, v]
            result["confirmed"] = True
            redraw()
            print(f"  ✓ Confirmed: ({u},{v}) → "
                  f"3D=[{result['anchor_3d'][0]:.0f},{result['anchor_3d'][1]:.0f},{result['anchor_3d'][2]:.0f}]mm")
        elif key == ord('s') and confirmed and result:
            out_path = img_path.replace('.png', '_anchor.json').replace('.jpg', '_anchor.json')
            with open(out_path, 'w') as f:
                json.dump({k: v for k, v in result.items()
                           if not isinstance(v, (np.ndarray,))}, f,
                          indent=2, ensure_ascii=False)
            print(f"  Saved: {out_path}")

    _cv2.destroyAllWindows()
    return result if confirmed else {}


def yolo_anchor(img_path: str, depth_map: np.ndarray,
                fx: float, fy: float, cx: float, cy: float,
                model_name: str = "yolo11n.pt",
                conf: float = 0.25) -> dict:
    """Auto: YOLO detection box bottom-left corner as anchor.

    Falls back to largest depth-connected-component bottom-left if no detections.
    """
    from ultralytics import YOLO

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    h, w = img.shape[:2]

    u0, v0 = None, None  # will be determined below

    # Try YOLO detection
    try:
        model = YOLO(model_name)
        results = model(img, conf=conf, verbose=False)
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            areas = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                areas.append((x2 - x1) * (y2 - y1))
            best_idx = np.argmax(areas)
            x1, y1, x2, y2 = boxes[best_idx].xyxy[0].tolist()
            cls_id = int(boxes[best_idx].cls[0])
            cls_name = model.names.get(cls_id, f"cls_{cls_id}")
            u0, v0 = int(x1), int(y2)
            print(f"  YOLO: {cls_name} bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]"
                  f" → anchor=({u0},{v0})")
    except Exception as e:
        print(f"  YOLO skipped: {e}")

    # Fallback: use depth structure to find shelf anchor
    if u0 is None:
        u0, v0 = _depth_based_anchor(depth_map)
        print(f"  Depth-heuristic anchor: ({u0}, {v0})")

    result = get_anchor_3d(u0, v0, depth_map, window=5, fx=fx, fy=fy, cx=cx, cy=cy)
    result["method"] = "yolo+depth"
    result["pixel_uv"] = [u0, v0]
    return result


def _depth_based_anchor(depth_map: np.ndarray) -> tuple:
    """Find anchor point from depth map structure.

    Strategy: find the largest contiguous region of valid depth,
    then pick its front-bottom-left pixel (closest depth = shelf front edge).
    """
    h, w = depth_map.shape

    # Binary mask of valid depth
    valid_mask = ~np.isnan(depth_map)

    if valid_mask.sum() == 0:
        return w // 2, h - 10

    # Find connected components (8-connected)
    valid_u8 = valid_mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        valid_u8, connectivity=8)

    if num_labels <= 1:
        return w // 2, h - 10

    # Skip background (label 0), find the largest component
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_mask = (labels == largest_label)

    # Within largest component, find the front-most region
    # (smallest Z/depth values = closest to camera)
    depths_in_largest = depth_map.copy()
    depths_in_largest[~largest_mask] = np.nan

    # Take the ~10% closest points as the front face
    valid_depths = depths_in_largest[~np.isnan(depths_in_largest)]
    if len(valid_depths) == 0:
        ys, xs = np.where(largest_mask)
        return int(np.median(xs)), int(np.max(ys))

    depth_threshold = np.percentile(valid_depths, 15)
    front_mask = largest_mask & (depth_map <= depth_threshold)

    if front_mask.sum() < 10:
        front_mask = largest_mask

    # Bottom-left of front face → anchor
    # Prefer pixels that HAVE depth data (not NaN)
    ys, xs = np.where(front_mask & ~np.isnan(depth_map))

    if len(ys) == 0:
        # Fallback: use any pixel in front mask
        ys, xs = np.where(front_mask)
        if len(ys) == 0:
            ys, xs = np.where(largest_mask)

    # bottom = max v (image coords: v increases downward)
    bottom_idx = np.argmax(ys)
    v0 = ys[bottom_idx]
    # Among bottom row, take the leftmost (min u)
    bottom_mask = ys >= v0 - 5
    bottom_ys = ys[bottom_mask]
    bottom_xs = xs[bottom_mask]
    u0 = int(np.min(bottom_xs))

    return u0, v0


# ══════════════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════════════

def visualize_result(img_path: str, depth_map: np.ndarray, result: dict, output_dir: str):
    """Save annotated image + depth visualization."""
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(img_path).stem

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    u0, v0 = result.get("pixel_uv", result.get("center_uv", [w // 2, h // 2]))

    # Annotated image
    vis = img.copy()
    cv2.circle(vis, (u0, v0), 5, (0, 255, 0), -1)
    cv2.circle(vis, (u0, v0), 8, (0, 255, 0), 2)
    cv2.rectangle(vis, (u0 - 2, v0 - 2), (u0 + 2, v0 + 2), (0, 255, 255), 1)

    if result.get("anchor_3d"):
        a3 = result["anchor_3d"]
        info = (f"Anchor3D: [{a3[0]:.0f}, {a3[1]:.0f}, {a3[2]:.0f}] mm | "
                f"depth={result['median_depth']:.0f}mm | "
                f"valid={result['valid_count']}/{result['window_total']}")
        cv2.putText(vis, info, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    out_img = os.path.join(output_dir, f"{stem}_anchor_v2.png")
    cv2.imwrite(out_img, vis)
    print(f"  Viz saved: {out_img}")

    # Depth map viz
    depth_vis = depth_map.copy()
    depth_finite = depth_vis[~np.isnan(depth_vis)]
    if len(depth_finite) > 0:
        vmin, vmax = np.percentile(depth_finite, [5, 95])
        depth_vis = np.clip(depth_vis, vmin, vmax)
        depth_vis = (depth_vis - vmin) / (vmax - vmin + 1e-8)
        depth_vis[np.isnan(depth_map)] = 0
        depth_vis = (depth_vis * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        cv2.circle(depth_vis, (u0, v0), 3, (0, 255, 0), -1)
        out_depth = os.path.join(output_dir, f"{stem}_depth_v2.png")
        cv2.imwrite(out_depth, depth_vis)
        print(f"  Depth viz: {out_depth}")


# ══════════════════════════════════════════════════════════════════════════════
# Processing
# ══════════════════════════════════════════════════════════════════════════════

def process_frame(pcd_path: str, img_path: str, args) -> dict:
    """Process single frame: PCD + RGB → 3D anchor."""
    stem = Path(pcd_path).stem
    t0 = time.time()

    xyz, rgb = read_pcd_binary(pcd_path)
    print(f"\n[{stem}] PCD: {len(xyz)} pts | img: {Path(img_path).name}")

    # Build depth map from PCD
    dm = build_depth_map(xyz, img_w=args.img_w, img_h=args.img_h,
                         fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)
    valid_pct = 100 * np.sum(~np.isnan(dm)) / (args.img_w * args.img_h)
    print(f"  Depth map: {args.img_w}×{args.img_h}, {valid_pct:.1f}% valid")

    # 2D Prompt → 3D Anchor
    if args.mode == "click":
        # Old click-only mode (no auto-proposal)
        result = review_anchor(img_path, dm, args.fx, args.fy, args.cx, args.cy)
    elif args.mode == "review":
        # New review mode: auto-propose + human confirm/correct
        auto_u, auto_v = _depth_based_anchor(dm)
        result = review_anchor(img_path, dm, args.fx, args.fy, args.cx, args.cy,
                               auto_u=auto_u, auto_v=auto_v)
    elif args.mode == "yolo":
        result = yolo_anchor(img_path, dm, args.fx, args.fy, args.cx, args.cy,
                             model_name=args.yolo_model, conf=args.conf)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    elapsed = time.time() - t0
    result["stem"] = stem
    result["elapsed_s"] = round(elapsed, 2)

    if result.get("anchor_3d"):
        a3 = result["anchor_3d"]
        print(f"  ✓ Anchor3D: [{a3[0]:.0f}, {a3[1]:.0f}, {a3[2]:.0f}] mm | t={elapsed:.1f}s")
    else:
        print(f"  ✗ Failed: {result.get('error', 'unknown')}")

    # Save JSON
    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, f"{stem}_anchor_v2.json")
    clean = {}
    for k, v in result.items():
        if isinstance(v, (np.integer, np.int64, np.int32)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating, np.float64, np.float32)):
            clean[k] = float(v)
        elif isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        elif isinstance(v, (list, tuple)):
            clean[k] = [int(x) if isinstance(x, (np.integer, np.int64, np.int32))
                        else float(x) if isinstance(x, (np.floating, np.float64, np.float32))
                        else x for x in v]
        else:
            clean[k] = v
    with open(out_json, "w") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    print(f"  JSON: {out_json}")

    if not args.no_viz:
        visualize_result(img_path, dm, clean, args.output_dir)

    return clean


def main():
    parser = argparse.ArgumentParser(description="Single-Anchor 3D Prompting V2 MVP")
    parser.add_argument("--pcd", "--pcd_path", dest="pcd_path", default=None)
    parser.add_argument("--img", "--img_path", dest="img_path", default=None)
    parser.add_argument("--data_dir", default=None,
                        help="Batch: directory with TV_*.pcd + matching *.png")
    parser.add_argument("--mode", default="review", choices=["review", "yolo", "click"],
                        help="review=auto-propose+human confirm | yolo=fully auto | click=manual")
    parser.add_argument("--output_dir", default="output/shelf_anchor_v2")
    parser.add_argument("--no_viz", action="store_true")
    # Camera intrinsics (estimated — tune for your camera setup)
    parser.add_argument("--fx", type=float, default=410.9)
    parser.add_argument("--fy", type=float, default=410.9)
    parser.add_argument("--cx", type=float, default=307.0)
    parser.add_argument("--cy", type=float, default=264.3)
    parser.add_argument("--img_w", type=int, default=640)
    parser.add_argument("--img_h", type=int, default=480)
    # YOLO
    parser.add_argument("--yolo_model", default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    if args.pcd_path and args.img_path:
        process_frame(args.pcd_path, args.img_path, args)
    elif args.data_dir:
        data_dir = Path(args.data_dir)
        pcds = sorted(data_dir.glob("TV_*.pcd"))
        print(f"Found {len(pcds)} PCDs in {args.data_dir}")
        for pcd_path in pcds:
            stem = pcd_path.stem
            num_id = stem.replace("TV_", "")
            candidates = list(data_dir.glob(f"{num_id}.jpg")) + \
                         list(data_dir.glob(f"{num_id}.png"))  # JPG first: real RGB
            if not candidates:
                print(f"  ✗ No matching image for {stem}")
                continue
            img_path = str(candidates[0])
            try:
                process_frame(str(pcd_path), img_path, args)
            except Exception as e:
                print(f"  ✗ Error: {e}")
    else:
        parser.error("Specify --pcd/--img or --data_dir")


if __name__ == "__main__":
    main()
