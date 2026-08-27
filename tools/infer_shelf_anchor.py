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


def _build_depth_map(xyz, img_w=640, img_h=480, fx=392.67, fy=411.42, cx=321.34, cy=236.55):
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


def _get_anchor_3d(u0, v0, depth_map, window=5, fx=392.67, fy=411.42, cx=321.34, cy=236.55):
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


def _fit_plane_depths(box_xyxy, kp_uvs, xyz, fx, fy, cx, cy):
    """Fit the shelf-face plane from box-interior PCD points, then place each
    keypoint on that plane (ray-plane intersection).

    货架前脸是竖直平面, 两个底角同在该面上 → 深度自然一致。比逐点 5×5 窗口
    中位数稳: 窗口在角点/缝隙处常抓到背景, 面拟合用框内全部点, 鲁棒得多。
    返回 (depths, info), depths[i] 与 kp_uvs[i] 对应; 点太少返回 None。
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    valid = z > 1.0
    X, Y, Z = x[valid], y[valid], z[valid]
    u = np.round(fx * X / Z + cx).astype(np.int32)
    v = np.round(fy * Y / Z + cy).astype(np.int32)
    x1, y1, x2, y2 = box_xyxy
    m = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    X, Y, Z = X[m], Y[m], Z[m]
    if len(Z) < 20:
        return None
    # 稳健过滤: 只留主导表面 (中位数 ± 1.5 IQR)
    q1, q3 = np.percentile(Z, [25, 75]); iqr = q3 - q1
    m2 = (Z >= q1 - 1.5 * iqr) & (Z <= q3 + 1.5 * iqr)
    X, Y, Z = X[m2], Y[m2], Z[m2]
    if len(Z) < 20:
        return None
    # 平面 z = a·x + b·y + c (面朝相机, 用最小二乘)
    A = np.stack([X, Y, np.ones_like(X)], axis=1)
    a, b, c = np.linalg.lstsq(A, Z, rcond=None)[0]
    zlo, zhi = np.percentile(Z, [1, 99])
    z_ref = float(np.percentile(Z, 10))   # 近端主导面深度, 供质量判定
    zmed = float(np.median(Z))
    depths = []
    for u0, v0 in kp_uvs:
        dx, dy = (u0 - cx) / fx, (v0 - cy) / fy
        denom = 1.0 - a * dx - b * dy
        if abs(denom) < 1e-3:          # 射线近平行于面 → 回退框中位深度
            t = zmed
        else:
            t = float(np.clip(c / denom, zlo - 300, zhi + 300))
        t = float(np.clip(t, 600, 6000))
        depths.append([round(t * dx, 1), round(t * dy, 1), round(t, 1)])
    return depths, {"z_ref": z_ref, "n_points": len(Z)}


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

def _build_instance_kps(inst_raw, box, xyz, dm, fx, fy, cx, cy):
    """一个实例的角点深度: 检测框内 PCD 面拟合优先, 回退逐点 5×5 窗口查表。

    返回 (instance_kps, method)。inst_raw = [(u, v, kp_conf), ...], box = xyxy。
    """
    instance_kps = []
    method = "none"
    fit = None
    if box is not None:
        fit = _fit_plane_depths(box, [(u, v) for u, v, _ in inst_raw], xyz, fx, fy, cx, cy)
    if fit is not None:
        depths, _info = fit
        method = "plane_fit"
        for (u, v, kp_conf), a3 in zip(inst_raw, depths):
            instance_kps.append({
                "pixel_uv": [u, v],
                "anchor_3d": a3,
                "confidence": round(kp_conf, 3),
            })
    else:
        method = "window_median"
        for (u, v, kp_conf) in inst_raw:
            anchor_3d = _get_anchor_3d(u, v, dm, fx=fx, fy=fy, cx=cx, cy=cy)
            if anchor_3d is None:
                continue
            instance_kps.append({
                "pixel_uv": [u, v],
                "anchor_3d": anchor_3d,
                "confidence": round(kp_conf, 3),
            })
    return instance_kps, method


def infer_frame(model, img_bgr, xyz, dm, img_w=640, img_h=480,
                 fx=392.67, fy=411.42, cx=321.34, cy=236.55, conf=0.3):
    """Run YOLO-pose on one frame, do PCD lookup for visible keypoints.

    深度取法: 优先用检测框内 PCD 拟合货架面平面, 角点射线与面求交 (两角深度
    自然一致); 框内点太少则回退逐点 5×5 窗口查表。返回 (keypoints, depth_method)。
    """
    results = model(img_bgr, conf=conf, verbose=False)

    all_keypoints = []
    method = "none"

    for r in results:
        if r.keypoints is None:
            continue

        kp_tensor = r.keypoints.data  # (N, K, 3) — [x, y, conf]
        if kp_tensor is None or kp_tensor.shape[0] == 0:
            continue

        for inst_idx in range(kp_tensor.shape[0]):
            # 收集该实例的合格角点像素
            inst_raw = []
            for kp_idx in range(kp_tensor.shape[1]):
                x, y, kp_conf = kp_tensor[inst_idx, kp_idx].tolist()
                if kp_conf < conf:
                    continue
                u = int(round(x))
                v = int(round(y))
                if u < 0 or u >= img_w or v < 0 or v >= img_h:
                    continue
                inst_raw.append((u, v, kp_conf))
            if not inst_raw:
                continue

            box = None
            if r.boxes is not None and len(r.boxes) > inst_idx:
                box = r.boxes.xyxy[inst_idx].cpu().numpy()
            instance_kps, inst_method = _build_instance_kps(inst_raw, box, xyz, dm, fx, fy, cx, cy)
            if instance_kps:
                method = inst_method
                all_keypoints.append((instance_kps, method))

    # Flatten: take the first instance (should be the only one for single-shelf)
    if all_keypoints:
        return all_keypoints[0]
    return [], "none"


# ── int8 sim 后端 (PC 仿真 ≈ 板子 NPU int8) ──

def _setup_int8sim(onnx, calib_dataset):
    """in-session build int8 + 初始化模拟器 (与生产转换同参: normal/layer, RGB2BGR)。

    返回 (rknn_session, deploy_module)。deploy_module 提供 letterbox/merge_outputs/
    decode_yolopose (与板端 infer_image.py 同款解码, 坐标反 letterbox 回原图)。
    """
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dep_dir = os.path.join(repo, 'output/rk3588_deploy')
    sys.path.insert(0, dep_dir)  # infer_image 内 import shelf_viz 需要
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'dep_infer_image', os.path.join(dep_dir, 'infer_image.py'))
    dep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dep)

    from rknn.api import RKNN
    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                target_platform='rk3588', quantized_algorithm='normal',
                quantized_method='layer', quant_img_RGB2BGR=True)
    if rknn.load_onnx(model=onnx) != 0:
        raise SystemExit(f'load_onnx failed: {onnx}')
    if rknn.build(do_quantization=True, dataset=calib_dataset) != 0:
        raise SystemExit(f'build int8 failed: {calib_dataset}')
    if rknn.init_runtime() != 0:
        raise SystemExit('init_runtime failed')
    return rknn, dep


def infer_frame_int8(rknn, dep, img_bgr, xyz, dm, img_w=640, img_h=480,
                     fx=392.67, fy=411.42, cx=321.34, cy=236.55, conf=0.3, iou=0.45):
    """int8 sim 后端: rknn 仿真(≈板子 NPU) 检测 → 同款面拟合深度。"""
    lb, _, _ = dep.letterbox(img_bgr)
    blob = np.ascontiguousarray(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB))[None]
    out = dep.merge_outputs(rknn.inference(inputs=[blob]))
    dets = dep.decode_yolopose(out, img_bgr.shape[:2], conf, iou)

    all_keypoints = []
    for det in dets:
        inst_raw = []
        for k in det['kpts']:
            u, v = int(round(k[0])), int(round(k[1]))
            if 0 <= u < img_w and 0 <= v < img_h:
                inst_raw.append((u, v, 1.0))
        if not inst_raw:
            continue
        box = det['box']
        instance_kps, inst_method = _build_instance_kps(inst_raw, box, xyz, dm, fx, fy, cx, cy)
        if instance_kps:
            all_keypoints.append((instance_kps, inst_method))

    if all_keypoints:
        return all_keypoints[0]
    return [], "none"


def process_frame(stem, model, data_dir, output_dir, img_w, img_h,
                  fx, fy, cx, cy, conf, backend="yolov8", iou=0.45):
    """Process a single frame: load → infer → save. backend: yolov8 | int8sim."""
    num_id = stem.replace("TV_", "")
    data_dir = Path(data_dir)

    # Find files
    pcd_path = data_dir / f"{stem}.pcd"
    img_path = None
    for ext in [".jpg", ".png"]:
        # TR_ 前缀 = 夜间/可见光图; 必须优先于同名深度图 (uint16 .png 读出来是黑的)
        for cand_name in [stem, num_id, f"TV_{num_id}", f"TR_{num_id}"]:
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
    if backend == "int8sim":
        rknn, dep = model
        keypoints, depth_method = infer_frame_int8(
            rknn, dep, img_display, xyz, dm, img_w, img_h, fx, fy, cx, cy, conf, iou)
    else:
        keypoints, depth_method = infer_frame(model, img_display, xyz, dm, img_w, img_h,
                                              fx, fy, cx, cy, conf)

    elapsed = time.time() - t0

    # 两角深度一致性: 货架前脸为平面, 两底角深度应接近 (阈值 200mm)
    depth_consistent = None
    if len(keypoints) >= 2:
        zs = [k["anchor_3d"][2] for k in keypoints if k.get("anchor_3d")]
        if len(zs) >= 2:
            depth_consistent = abs(zs[0] - zs[1]) <= 200.0

    # Save
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "stem": stem,
        "num_keypoints": len(keypoints),
        "keypoints": keypoints,
        "method": "yolo_pose_inference" if backend == "yolov8" else "yolo_pose_inference_int8sim",
        "depth_method": depth_method,
        "depth_consistent": depth_consistent,
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
    parser.add_argument("--backend", default="yolov8", choices=["yolov8", "int8sim"],
                        help="yolov8=ultralytics fp32; int8sim=rknn x86 仿真(≈板子 int8)")
    parser.add_argument("--onnx", default=None,
                        help="int8sim 用: 三输出拆分 ONNX (如 best_split.onnx)")
    parser.add_argument("--calib-dataset", default="output/rk3588_deploy_pre/calib_night/dataset.txt",
                        help="int8sim 用: rknn int8 标定集 (与生产转换同参)")
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--data_dir", default="data/new_sheef/pngs")
    parser.add_argument("--output_dir", default="output/shelf_anchor_v2_pred")
    parser.add_argument("--stem", default=None)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--device", default="cuda:0")
    # Camera
    parser.add_argument("--fx", type=float, default=392.67)
    parser.add_argument("--fy", type=float, default=411.42)
    parser.add_argument("--cx", type=float, default=321.34)
    parser.add_argument("--cy", type=float, default=236.55)
    parser.add_argument("--img_w", type=int, default=640)
    parser.add_argument("--img_h", type=int, default=480)
    args = parser.parse_args()

    if args.backend == "int8sim":
        if not args.onnx:
            print("--backend int8sim 需要 --onnx (如 weights/best_split.onnx)")
            sys.exit(1)
        print(f"int8 sim in-session build: {args.onnx}\n  标定: {args.calib_dataset}")
        model = _setup_int8sim(args.onnx, args.calib_dataset)
    else:
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
        # 全量处理 (2026-08-18 起不再排除 TV_250000000001 outlier 帧, 该帧 2026-08-26 已删除;
        # 兼容无 TV_ 前缀的 0807/0811/0812 现场批次, 如 250000010477.pcd)
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
            backend=args.backend, iou=args.iou,
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
