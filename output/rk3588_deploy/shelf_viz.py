#!/usr/bin/env python3
"""板端可视化 + PCD 深度查表 (与 PC 端 infer_shelf_anchor.py 同款, 略简)。

绘制样式与 `tools/infer_shelf_anchor.py::draw_keypoints` (v4) 一致:
  红点 + 十字准线 + P1/P2 编号 + 虚线连接 + 黄色 X 中心 + 左上图例面板 (PCD 3D XYZ)。

深度查表: PCD 点云按相机内参投影到图像平面 → 逐像素最小深度图 (z>1.0 过滤),
          关键点 5×5 窗口取中值深度 → 反投影 (x,y,z)。
内参: 深度 640×480, 自动读同目录 camera_intrinsics_<sn>.json (出厂实测),
      缺失回退 fx=392.67, fy=411.42, cx=321.34, cy=236.55。
"""
import glob
import json
import os

import cv2
import numpy as np


def _load_depth_intrinsics(fallback=(392.67, 411.42, 321.34, 236.55)):
    """读同目录 camera_intrinsics_*.json 的 depth_intrinsic (query_camera_intrinsics.py 输出)。
    缺失/解析失败回退到上次实测常量。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for p in sorted(glob.glob(os.path.join(script_dir, "camera_intrinsics_*.json"))):
        try:
            with open(p, encoding="utf-8") as f:
                di = json.load(f).get("depth_intrinsic", {})
            if di:
                return float(di["fx"]), float(di["fy"]), float(di["cx"]), float(di["cy"])
        except (OSError, ValueError, KeyError):
            continue
    return fallback


# 相机内参 (深度 640x480, 读 JSON 优先, 缺失回退实测值)
CAM_FX, CAM_FY, CAM_CX, CAM_CY = _load_depth_intrinsics()

# v4 绘制配色
DOT_COLOR = (0, 0, 255)        # 红
CENTER_COLOR = (0, 255, 255)   # 黄 X
LEGEND_BG = (40, 40, 40)
LEGEND_FG = (240, 240, 240)
LEGEND_ACCENT = (80, 200, 255)
OUTLINE = (30, 30, 30)


def read_pcd_binary(path):
    """读取 binary PCD → (N,3) float32 xyz。"""
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode().strip()
            if line == "DATA binary":
                break
        dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("rgb", "u4")])
        data = np.frombuffer(f.read(), dtype=dtype)
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    return xyz


def build_depth_map(xyz, img_w=640, img_h=480,
                    fx=CAM_FX, fy=CAM_FY, cx=CAM_CX, cy=CAM_CY):
    """PCD 投影 → 逐像素最小深度图 (NaN 表无点), 尺寸 (img_h, img_w)。"""
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


def get_anchor_3d(u0, v0, depth_map, window=5,
                  fx=CAM_FX, fy=CAM_FY, cx=CAM_CX, cy=CAM_CY):
    """深度图查表 → 3D 锚点 [x,y,z]。5×5 中值, 无点则逐级扩大; 仍无 → None。"""
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


def _is_valid_xyz(p):
    """单点是否有效: 有限且 z>0 (SDK 缺失像素为 0,0,0 或 NaN)。"""
    return np.isfinite(p).all() and p[2] > 0


def _valid_xyz_mask(pts):
    """(N,3) → bool mask (有限且 z>0)。"""
    return np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0)


def lookup_xyz(points, u, v, window=5):
    """RGB 对齐点云 (H,W,3) 直接查 (u,v) 像素的 3D 点 → [x,y,z]。

    相机开启 DEPTH_TO_RGB 对齐后 getPointCloud 输出与 RGB 同分辨率、同像素索引,
    故关键点像素 (u,v) 直接取 points[v,u] (不用内参投影, 不按比例缩放)。
    该像素无有效深度时, 局部窗口邻域采样 (5×5 起逐级扩大) 取有效点中位数,
    保证深度稳健; 仍无有效点 → None。
    """
    h, w = points.shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        return None
    p = points[v, u]
    if _is_valid_xyz(p):
        return [round(float(p[0]), 1), round(float(p[1]), 1), round(float(p[2]), 1)]
    for expand in range(window // 2, min(w, h) // 2, 5):
        u0, u1 = max(0, u - expand), min(w, u + expand + 1)
        v0, v1 = max(0, v - expand), min(h, v + expand + 1)
        blk = points[v0:v1, u0:u1].reshape(-1, 3)
        valid = blk[_valid_xyz_mask(blk)]
        if valid.size:
            med = np.median(valid, axis=0)
            return [round(float(med[0]), 1), round(float(med[1]), 1), round(float(med[2]), 1)]
    return None


def draw_keypoints(img, keypoints, output_path=None):
    """v4 风格绘制 (同 PC 端 infer_shelf_anchor 样式)。

    keypoints: list[{pixel_uv:[u,v], anchor_3d:[x,y,z]|None, confidence}]
    anchor_3d 为 None 时图例 XYZ 显示 "--" (无 PCD 深度)。
    返回绘制后的 BGR 图。
    """
    viz = img.copy()
    h, w = viz.shape[:2]

    kp_points = []  # (u, v, anchor_3d)
    for i, kp in enumerate(keypoints):
        u, v = kp["pixel_uv"]
        a3d = kp.get("anchor_3d")
        kp_points.append((u, v, a3d))

        # 细十字准线
        cv2.line(viz, (u - 6, v), (u + 6, v), DOT_COLOR, 1)
        cv2.line(viz, (u, v - 6), (u, v + 6), DOT_COLOR, 1)
        # 红点 + 深色轮廓
        cv2.circle(viz, (u, v), 4, DOT_COLOR, -1)
        cv2.circle(viz, (u, v), 4, OUTLINE, 1)
        # P 编号
        cv2.putText(viz, str(i + 1), (u + 7, v - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, DOT_COLOR, 1, cv2.LINE_AA)

    # 虚线连接 + 黄色 X 中心
    has_center = False
    center_u = center_v = 0
    center_xyz = None
    if len(kp_points) >= 2:
        for i in range(len(kp_points) - 1):
            u1, v1, _ = kp_points[i]
            u2, v2, _ = kp_points[i + 1]
            dist = np.sqrt((u2 - u1) ** 2 + (v2 - v1) ** 2)
            n_segs = max(2, int(dist / 8))
            for s in range(0, n_segs, 2):
                t0 = s / n_segs
                t1 = min((s + 1) / n_segs, 1.0)
                su0, sv0 = int(u1 + (u2 - u1) * t0), int(v1 + (v2 - v1) * t0)
                su1, sv1 = int(u1 + (u2 - u1) * t1), int(v1 + (v2 - v1) * t1)
                cv2.line(viz, (su0, sv0), (su1, sv1), DOT_COLOR, 1, cv2.LINE_AA)

        has_xyz = all(p[2] is not None for p in kp_points)
        if has_xyz:
            all_3d = [p[2] for p in kp_points]
            center_xyz = [np.mean([a[i] for a in all_3d]) for i in range(3)]
        center_u = int(np.mean([p[0] for p in kp_points]))
        center_v = int(np.mean([p[1] for p in kp_points]))
        has_center = True

        x_sz = 5
        cv2.line(viz, (center_u - x_sz, center_v - x_sz),
                 (center_u + x_sz, center_v + x_sz), CENTER_COLOR, 2, cv2.LINE_AA)
        cv2.line(viz, (center_u + x_sz, center_v - x_sz),
                 (center_u - x_sz, center_v + x_sz), CENTER_COLOR, 2, cv2.LINE_AA)

    # 图例面板 (左上)
    if keypoints:
        n_kp = len(keypoints)
        line_h = 16
        panel_w = 220
        extra_rows = 1 if has_center else 0
        panel_h = 32 + (n_kp + extra_rows) * line_h + 6

        overlay = viz.copy()
        cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), LEGEND_BG, -1)
        cv2.addWeighted(overlay, 0.75, viz, 0.25, 0, viz)
        cv2.rectangle(viz, (8, 8), (8 + panel_w, 8 + panel_h), (100, 100, 100), 1)

        cv2.putText(viz, f"Shelf Corners  ({n_kp} pts)", (16, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, LEGEND_ACCENT, 1, cv2.LINE_AA)
        cv2.line(viz, (16, 33), (8 + panel_w - 8, 33), (80, 80, 80), 1)

        def fmt(a3d):
            if a3d is None:
                return "--", "--", "--"
            return f"{int(round(a3d[0])):>6}", f"{int(round(a3d[1])):>6}", f"{int(round(a3d[2])):>6}"

        for i, kp in enumerate(keypoints):
            x, y, z = fmt(kp.get("anchor_3d"))
            y0 = 50 + i * line_h
            cv2.putText(viz, f"P{i+1}", (16, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, DOT_COLOR, 1, cv2.LINE_AA)
            cv2.putText(viz, f"X {x}  Y {y}  Z {z}", (42, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, LEGEND_FG, 1, cv2.LINE_AA)

        if has_center:
            x, y, z = fmt(center_xyz)
            y0 = 50 + n_kp * line_h + 2
            cv2.line(viz, (16, y0 - 2), (8 + panel_w - 8, y0 - 2), (60, 60, 70), 1)
            cv2.putText(viz, "C", (16, y0 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, CENTER_COLOR, 1, cv2.LINE_AA)
            cv2.putText(viz, f"X {x}  Y {y}  Z {z}", (42, y0 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 150), 1, cv2.LINE_AA)

    # 底部信息
    cv2.putText(viz, f"YOLO-Pose | {len(keypoints)} corners",
                (w - 240, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, viz)
    return viz
