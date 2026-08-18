#!/usr/bin/env python3
"""
Shelf Anchor V2 — Web 批量交互审核
====================================
人机协作 Web 界面: 自动提案 → 确认/修正 → PCD 查表 → 保存。

用法:
  python tools/shelf_anchor_v3_web.py
  python tools/shelf_anchor_v3_web.py --port 8080 --data_dir data/new_sheef/pngs   # 全量数据
  python tools/shelf_anchor_v3_web.py --data_dir data/new_sheef/prototypes         # 原型 (默认)

打开浏览器 http://localhost:5000，快速审核:
  左键点击  — 修正 anchor 位置
  ENTER     — 确认当前 anchor → PCD 查表 → 保存 → 下一帧
  ESC       — 跳过当前帧
  ←/→       — 上一帧/下一帧
  G         — 跳转到指定帧号
  S         — 保存当前帧 (不跳到下一帧)
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

# ══════════════════════════════════════════════════════════════════════════════
# 复用 V2 核心函数
# ══════════════════════════════════════════════════════════════════════════════

def _read_pcd_binary(path: str):
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode().strip()
            if line == "DATA binary":
                break
        dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("rgb", "u4")])
        data = np.frombuffer(f.read(), dtype=dtype)
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    rgb_raw = data["rgb"]
    r = ((rgb_raw >> 16) & 0xFF).astype(np.float32) / 255.0
    g = ((rgb_raw >> 8) & 0xFF).astype(np.float32) / 255.0
    b = (rgb_raw & 0xFF).astype(np.float32) / 255.0
    rgb = np.column_stack([r, g, b])
    return xyz, rgb


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


def _depth_based_anchor(depth_map):
    h, w = depth_map.shape
    valid_mask = ~np.isnan(depth_map)
    if valid_mask.sum() == 0:
        return w // 2, h - 10
    valid_u8 = valid_mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(valid_u8, connectivity=8)
    if num_labels <= 1:
        return w // 2, h - 10
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_mask = (labels == largest_label)
    depths = depth_map.copy()
    depths[~largest_mask] = np.nan
    valid_d = depths[~np.isnan(depths)]
    if len(valid_d) == 0:
        ys, xs = np.where(largest_mask)
        return int(np.median(xs)), int(np.max(ys))
    threshold = np.percentile(valid_d, 15)
    front_mask = largest_mask & (depth_map <= threshold)
    if front_mask.sum() < 10:
        front_mask = largest_mask
    ys, xs = np.where(front_mask & ~np.isnan(depth_map))
    if len(ys) == 0:
        ys, xs = np.where(front_mask)
    bottom_idx = int(np.argmax(ys))
    v0 = int(ys[bottom_idx])
    bottom_mask = ys >= v0 - 5
    u0 = int(np.min(xs[bottom_mask]))
    return u0, v0


def _to_native(obj):
    """递归转换 numpy 标量为 Python 原生类型。"""
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


# ══════════════════════════════════════════════════════════════════════════════
# 全局状态
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR = None
OUTPUT_DIR = None
YOLO_LABELS_DIR = None   # YOLO-format labels dir (for loading pseudo-labels)
STEMS = []          # list of pcd stems like "TV_250000036488"
FRAMES = []         # list of {stem, pcd_path, img_path, has_result}
IMG_W, IMG_H = 640, 480
# Eagle-M4 Mega 实测: fy=410.88, cy≈264.3 (PCD 网格); fx 方形像素先验; 出厂值见 tools/query_camera_intrinsics.py
FX, FY, CX, CY = 410.9, 410.9, 307.0, 264.3

# 深度图缓存: 避免每次点击都重读 PCD
_dm_cache = {}  # stem → (xyz, depth_map)
_DM_CACHE_MAX = 3

app = Flask(__name__)


def _cached_depth_map(stem):
    """获取缓存的深度图，避免重复读取 PCD。"""
    if stem in _dm_cache:
        return _dm_cache[stem]
    frame = next((f for f in FRAMES if f["stem"] == stem), None)
    if frame is None:
        return None, None
    xyz, _ = _read_pcd_binary(frame["pcd_path"])
    dm = _build_depth_map(xyz, IMG_W, IMG_H, FX, FY, CX, CY)
    # LRU-ish: 超过上限清最早的
    if len(_dm_cache) >= _DM_CACHE_MAX:
        oldest = next(iter(_dm_cache))
        del _dm_cache[oldest]
    _dm_cache[stem] = (xyz, dm)
    return xyz, dm


def _load_yolo_labels(stem, img_w=640, img_h=480, K=4):
    """从 YOLO labels 目录加载伪标注关键点 → [(u, v), ...]。

    扫描 train/ 和 val/ 子目录。
    YOLO pose 格式: class_id cx cy bw bh x1 y1 v1 ... xK yK vK
    K 由标签格式自动检测。
    """
    if not YOLO_LABELS_DIR:
        return None
    for split in ("train", "val"):
        label_path = os.path.join(YOLO_LABELS_DIR, split, f"{stem}.txt")
        if os.path.exists(label_path):
            try:
                with open(label_path) as f:
                    line = f.readline().strip()
                parts = line.split()
                if len(parts) < 8:  # at least class + bbox + 1 kp
                    continue
                # Auto-detect K from number of parts
                K_actual = (len(parts) - 5) // 3
                kps = []
                for i in range(K_actual):
                    x = float(parts[5 + i * 3])
                    y = float(parts[5 + i * 3 + 1])
                    v = float(parts[5 + i * 3 + 2])
                    if v > 0:
                        kps.append((x * img_w, y * img_h))
                if kps:
                    return kps
            except Exception:
                pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# API
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return HTML


@app.route("/api/info")
def api_info():
    """返回全局配置和帧列表摘要。"""
    done = 0
    skipped = 0
    for f in FRAMES:
        out_json = os.path.join(OUTPUT_DIR, f"{f['stem']}_anchor_v2.json")
        if os.path.exists(out_json):
            try:
                with open(out_json) as fh:
                    d = json.load(fh)
                if d.get("skipped"):
                    skipped += 1
                elif d.get("reviewed", True):
                    done += 1
            except Exception:
                done += 1  # unreadable → count as done
    return jsonify({
        "total": len(FRAMES),
        "done": done,
        "pending": len(FRAMES) - done - skipped,
        "img_w": IMG_W,
        "img_h": IMG_H,
    })


@app.route("/api/frames")
def api_frames():
    """返回所有帧的摘要列表。"""
    result = []
    for i, f in enumerate(FRAMES):
        out_json = os.path.join(OUTPUT_DIR, f"{f['stem']}_anchor_v2.json")
        has = os.path.exists(out_json)
        n_kp = 0
        skipped = False
        if has:
            try:
                with open(out_json) as fh:
                    d = json.load(fh)
                    if d.get("skipped"):
                        skipped = True
                    n_kp = d.get("num_keypoints", len(d.get("keypoints", [])))
                    if not n_kp and "anchor_3d" in d:
                        n_kp = 1
                    # 模型伪标签 (reviewed=false) 不算已完成
                    is_reviewed = d.get("reviewed", True)
            except Exception:
                pass
        num_id = f["stem"].replace("TV_", "")
        result.append({
            "index": i,
            "stem": f["stem"],
            "num_id": num_id,
            "done": has and not skipped and is_reviewed,
            "skipped": skipped,
            "n_keypoints": n_kp,
        })
    return jsonify(result)


@app.route("/api/load/<stem>")
def api_load(stem):
    """加载一帧: RGB 图片 (base64) + auto anchor + 已保存的关键点。"""
    frame = next((f for f in FRAMES if f["stem"] == stem), None)
    if frame is None:
        return jsonify({"error": f"Unknown stem: {stem}"}), 404

    t0 = time.time()

    # 读 RGB → resize → base64
    img = cv2.imread(frame["img_path"])
    if img is None:
        return jsonify({"error": "Cannot read image"}), 500
    orig_h, orig_w = img.shape[:2]
    scale_u = IMG_W / orig_w if orig_w != IMG_W else 1.0
    scale_v = IMG_H / orig_h if orig_h != IMG_H else 1.0
    if img.shape[0] != IMG_H or img.shape[1] != IMG_W:
        img = cv2.resize(img, (IMG_W, IMG_H))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buf).decode()

    # 读 PCD → 构建深度图 → 自动 anchor + 3D 查表
    xyz, dm = _cached_depth_map(stem)
    auto_u, auto_v = _depth_based_anchor(dm)
    auto_u, auto_v = int(auto_u), int(auto_v)
    auto_3d = _get_anchor_3d(auto_u, auto_v, dm)

    # 检查是否有已保存的关键点
    saved_keypoints = None
    out_json = os.path.join(OUTPUT_DIR, f"{stem}_anchor_v2.json")
    if os.path.exists(out_json):
        try:
            with open(out_json) as fh:
                d = json.load(fh)
                if "keypoints" in d:
                    saved_keypoints = d["keypoints"]
                elif "anchor_3d" in d:
                    # 兼容旧格式 (单关键点)
                    saved_keypoints = [{
                        "id": 0, "pixel_uv": d.get("pixel_uv"), "anchor_3d": d.get("anchor_3d")
                    }]
        except Exception:
            pass

    # 缩放已保存关键点的 pixel_uv 到 640x480 画布，并回填缺失的 anchor_3d
    # 只对在原始分辨率 (如 1280x960) 中的旧伪标签做缩放；
    # 新伪标签已在 640x480 中，检测到任何坐标超出画布范围才缩放
    if saved_keypoints:
        # 判断是否需要缩放：是否有坐标超出 640x480 范围
        needs_scale = False
        if scale_u != 1.0 or scale_v != 1.0:
            for kp in saved_keypoints:
                if kp.get("pixel_uv"):
                    u, v = kp["pixel_uv"][0], kp["pixel_uv"][1]
                    if u >= IMG_W or v >= IMG_H:
                        needs_scale = True
                        break
        for kp in saved_keypoints:
            if kp.get("pixel_uv"):
                if needs_scale:
                    kp["pixel_uv"] = [int(kp["pixel_uv"][0] * scale_u), int(kp["pixel_uv"][1] * scale_v)]
                if not kp.get("anchor_3d") and dm is not None:
                    u, v = int(kp["pixel_uv"][0]), int(kp["pixel_uv"][1])
                    kp["anchor_3d"] = _get_anchor_3d(u, v, dm)

    # 如果没有已保存的关键点，尝试加载 YOLO 伪标注
    if saved_keypoints is None:
        yolo_kps = _load_yolo_labels(stem, IMG_W, IMG_H)
        if yolo_kps and dm is not None:
            saved_keypoints = []
            for i, (u, v) in enumerate(yolo_kps):
                anchor_3d = _get_anchor_3d(int(u), int(v), dm)
                saved_keypoints.append({
                    "id": i,
                    "pixel_uv": [int(u), int(v)],
                    "anchor_3d": anchor_3d,
                    "label": f"YOLO-pseudo-{i}",
                })

    elapsed = time.time() - t0
    return jsonify(_to_native({
        "stem": stem,
        "img_b64": img_b64,
        "img_w": IMG_W,
        "img_h": IMG_H,
        "auto_uv": [auto_u, auto_v],
        "auto_3d": auto_3d,
        "saved_keypoints": saved_keypoints,
        "elapsed_ms": round(elapsed * 1000),
    }))


@app.route("/api/lookup", methods=["POST"])
def api_lookup():
    """PCD 查表: 前端传来 (u, v) → 返回 3D 坐标 (不保存)。"""
    data = request.get_json()
    stem = data.get("stem")
    u = int(data.get("u", 0))
    v = int(data.get("v", 0))

    _, dm = _cached_depth_map(stem)
    if dm is None:
        return jsonify({"error": f"Unknown stem: {stem}"}), 404

    anchor_3d = _get_anchor_3d(u, v, dm)
    if anchor_3d is None:
        return jsonify({"error": "No valid depth at this pixel"}), 400

    # 窗口统计
    half = 2
    h, w = dm.shape
    u_min, u_max = max(0, u - half), min(w, u + half + 1)
    v_min, v_max = max(0, v - half), min(h, v + half + 1)
    wd = dm[v_min:v_max, u_min:u_max]
    vd = wd[~np.isnan(wd)]

    return jsonify(_to_native({
        "pixel_uv": [u, v],
        "anchor_3d": anchor_3d,
        "median_depth": round(float(np.median(vd)), 1) if len(vd) > 0 else None,
        "valid_count": int(len(vd)),
        "window_total": int((u_max - u_min) * (v_max - v_min)),
    }))


@app.route("/api/save_frame", methods=["POST"])
def api_save_frame():
    """保存整帧所有关键点。"""
    data = request.get_json()
    stem = data.get("stem")
    keypoints = data.get("keypoints", [])  # [{id, pixel_uv, anchor_3d, label?}, ...]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result = {
        "stem": stem,
        "num_keypoints": len(keypoints),
        "keypoints": keypoints,
        "method": "web_review",
        "reviewed": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_json = os.path.join(OUTPUT_DIR, f"{stem}_anchor_v2.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return jsonify({"success": True, "num_keypoints": len(keypoints), "saved": out_json})


@app.route("/api/skip", methods=["POST"])
def api_skip():
    """跳过当前帧。"""
    data = request.get_json()
    stem = data.get("stem")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_json = os.path.join(OUTPUT_DIR, f"{stem}_anchor_v2.json")
    with open(out_json, "w") as f:
        json.dump({"stem": stem, "skipped": True,
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, f,
                  indent=2, ensure_ascii=False)
    return jsonify({"success": True, "skipped": True})


# ══════════════════════════════════════════════════════════════════════════════
# HTML 前端
# ══════════════════════════════════════════════════════════════════════════════

HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Shelf Anchor V2 — Multi-Keypoint</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif;
       background: #1a1a2e; color: #e0e0e0; overflow: hidden; height: 100vh; }
#app { display: flex; height: 100vh; }
#left { flex: 2; display: flex; flex-direction: column; background: #16213e; position: relative; }
#right { flex: 1; display: flex; flex-direction: column; background: #0f3460; min-width: 340px; max-width: 420px; }
#toolbar { padding: 10px 14px; background: #0a0a1a; display: flex; align-items: center;
           gap: 10px; font-size: 14px; flex-shrink: 0; flex-wrap: wrap; }
#toolbar .title { font-weight: bold; font-size: 16px; color: #e94560; margin-right: 8px; }
#toolbar button { padding: 6px 16px; border: none; border-radius: 4px; cursor: pointer;
                  font-size: 13px; font-weight: 600; transition: 0.15s; }
#toolbar button:hover { opacity: 0.8; transform: translateY(-1px); }
.btn-confirm { background: #00c853; color: #000; }
.btn-skip { background: #ff6d00; color: #000; }
.btn-nav { background: #3a3a5c; color: #ccc; }
.btn-save { background: #2979ff; color: #fff; }
.btn-clear { background: #c62828; color: #fff; }
#status { margin-left: auto; font-size: 13px; color: #888; }
#canvas-wrap { flex: 1; position: relative; display: flex; align-items: center;
               justify-content: center; overflow: hidden; background: #0d1117; }
#img-canvas { cursor: crosshair; image-rendering: auto; }
#info-bar { padding: 8px 14px; background: #0a0a1a; font-size: 13px; display: flex;
            gap: 20px; flex-shrink: 0; flex-wrap: wrap; }
#info-bar .label { color: #888; }
#info-bar .value { color: #e0e0e0; font-weight: 600; }
.ok { color: #00c853 !important; }
.warn { color: #ff6d00 !important; }
#right-header { padding: 12px 14px; background: #0a0a1a; font-size: 14px;
                font-weight: bold; color: #e94560; flex-shrink: 0; }
#frame-list { flex: 1 1 40%; overflow-y: auto; max-height: 50%; }
#frame-list .item { padding: 7px 12px; font-size: 12px; cursor: pointer;
                    border-bottom: 1px solid #1a1a3e; display: flex;
                    justify-content: space-between; align-items: center; transition: 0.1s; }
#frame-list .item:hover { background: #1a1a4e; }
#frame-list .item.current { background: #0f3460; font-weight: bold; border-left: 3px solid #e94560; }
#frame-list .item.done { color: #00c853; }
#frame-list .item.skipped { color: #888; text-decoration: line-through; }
#frame-list .item.pending { color: #ff6d00; }
#frame-list .kp-badge { background: #3a3a5c; padding: 2px 6px; border-radius: 8px;
                        font-size: 10px; color: #aaa; }
#kp-panel { flex: 1 1 auto; display: flex; flex-direction: column;
            border-top: 2px solid #1a1a3e; min-height: 0; }
#kp-panel-header { padding: 10px 14px; background: #0a0a1a; font-size: 13px;
                   font-weight: bold; display: flex; justify-content: space-between;
                   align-items: center; flex-shrink: 0; }
#kp-list { flex: 1; overflow-y: auto; padding: 6px 0; }
#kp-list .kp-item { padding: 8px 14px; border-bottom: 1px solid #1a1a3e;
                    display: flex; flex-direction: column; gap: 4px; transition: 0.1s; }
#kp-list .kp-item:hover { background: #1a1a4e; }
#kp-list .kp-item .kp-top { display: flex; align-items: center; gap: 8px; }
#kp-list .kp-dot { width: 14px; height: 14px; border-radius: 50%; border: 2px solid #fff;
                   flex-shrink: 0; }
#kp-list .kp-num { font-weight: bold; font-size: 14px; min-width: 20px; }
#kp-list .kp-uv { font-size: 11px; color: #888; }
#kp-list .kp-3d { font-family: monospace; font-size: 13px; font-weight: bold; color: #00c853; }
#kp-list .kp-meta { font-size: 11px; color: #666; }
#kp-list .kp-del { margin-left: auto; cursor: pointer; color: #ff5252; font-size: 18px;
                   padding: 0 4px; border-radius: 3px; }
#kp-list .kp-del:hover { background: #ff5252; color: #fff; }
#kp-list .kp-label { font-size: 11px; color: #aaa; margin-top: 2px; }
#kp-list .kp-label select { background: #2a2a4e; color: #ccc; border: 1px solid #444;
                            padding: 2px 4px; border-radius: 3px; font-size: 11px; }
#right-footer { padding: 10px 14px; background: #0a0a1a; font-size: 12px;
                color: #666; flex-shrink: 0; line-height: 1.8; }
#right-footer kbd { background: #333; padding: 1px 5px; border-radius: 2px; color: #aaa; }
</style>
</head>
<body>
<div id="app">
<div id="left">
  <div id="toolbar">
    <span class="title">&#9873; Shelf Anchor V2</span>
    <button class="btn-confirm" onclick="doConfirm()">&#10003; 确认并下一帧 (ENTER)</button>
    <button class="btn-save" onclick="doSave()">&#128190; 保存 (S)</button>
    <button class="btn-skip" onclick="doSkip()">&#8617; 跳过 (ESC)</button>
    <button class="btn-clear" onclick="doClear()">&#10007; 清空</button>
    <button class="btn-nav" onclick="goPrev()">&#9664;</button>
    <span id="frame-label" style="font-weight:bold;min-width:80px;text-align:center">-</span>
    <button class="btn-nav" onclick="goNext()">&#9654;</button>
    <span id="status">就绪</span>
  </div>
  <div id="canvas-wrap">
    <canvas id="img-canvas"></canvas>
  </div>
  <div id="info-bar">
    <span><span class="label">自动提案: </span><span class="value warn" id="info-auto">-</span></span>
    <span><span class="label">关键点: </span><span class="value" id="info-kp-count">0</span></span>
    <span><span class="label">辅助线角度: </span><span class="value" id="info-guide">-</span></span>
    <span><span class="label">耗时: </span><span class="value" id="info-time">-</span></span>
  </div>
</div>
<div id="right">
  <div id="right-header">&#9776; 帧列表 (<span id="list-stats">-</span>)</div>
  <div id="frame-list"></div>
  <div id="kp-panel">
    <div id="kp-panel-header">
      <span>&#9679; 当前帧关键点</span>
      <span style="font-size:11px;color:#888" id="kp-panel-count">0 个</span>
    </div>
    <div id="kp-list"></div>
  </div>
  <div id="right-footer">
    <kbd>左键</kbd>添加关键点 &nbsp; <kbd>右键</kbd>删除 &nbsp; <kbd>ENTER</kbd>保存并下一帧<br>
    <kbd>S</kbd>保存 &nbsp; <kbd>ESC</kbd>跳过 &nbsp; <kbd>&#8592;&#8594;</kbd>导航 &nbsp; <kbd>G</kbd>跳转<br>
    两个关键点间自动显示<span style="color:#ffeb3b">黄色辅助延长线</span>, 用于检查与平台平行
  </div>
</div>
</div>

<script>
// ═══════════════════════════════════════════════════════
// 关键点颜色
// ═══════════════════════════════════════════════════════
const KP_COLORS = [
  '#ff3333','#33ff33','#3388ff','#ffaa00','#ff00ff',
  '#00ffff','#ff8800','#88ff00','#aa44ff','#ff4488',
  '#ff6666','#66ff66','#66aaff','#ffcc44','#ff66ff',
];

// ═══════════════════════════════════════════════════════
// 状态
// ═══════════════════════════════════════════════════════
let frames = [];
let currentIdx = 0;
let currentStem = '';
let currentImg = null;
let autoUV = [320, 240];
let auto3D = null;
let keypoints = [];  // [{id, uv:[u,v], anchor3d:[x,y,z], median_depth, valid_count, label}]
let nextId = 0;
let loadTimeMs = 0;
let lookingUp = false;

// ═══════════════════════════════════════════════════════
// Canvas
// ═══════════════════════════════════════════════════════
const imgCanvas = document.getElementById('img-canvas');
const ctx = imgCanvas.getContext('2d');

function canvasUV(e) {
  const rect = imgCanvas.getBoundingClientRect();
  const scaleX = imgCanvas.width / rect.width;
  const scaleY = imgCanvas.height / rect.height;
  return [
    Math.round((e.clientX - rect.left) * scaleX),
    Math.round((e.clientY - rect.top) * scaleY),
  ];
}

function findNearbyKP(u, v, threshold) {
  // 返回距离 (u,v) 最近的 keypoint index, 或 -1
  let best = -1, bestDist = threshold + 1;
  const s = imgCanvas.width / 640;  // 缩放因子
  const threshPx = threshold * s;
  for (let i = 0; i < keypoints.length; i++) {
    const [ku, kv] = keypoints[i].uv;
    const d = Math.sqrt((ku - u)**2 + (kv - v)**2);
    if (d < threshPx && d < bestDist) { best = i; bestDist = d; }
  }
  return best;
}

imgCanvas.addEventListener('click', async function(e) {
  if (lookingUp) return;
  e.preventDefault();
  const [u, v] = canvasUV(e);
  if (u < 0 || u >= 640 || v < 0 || v >= 480) return;

  lookingUp = true;
  document.getElementById('status').textContent = 'PCD 查表中...';
  try {
    const resp = await fetch('/api/lookup', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({stem: currentStem, u, v}),
    });
    const data = await resp.json();
    if (data.error) {
      document.getElementById('status').textContent = '该位置无深度数据';
      lookingUp = false;
      return;
    }
    const kp = {
      id: nextId++,
      uv: [u, v],
      anchor3d: data.anchor_3d,
      median_depth: data.median_depth,
      valid_count: data.valid_count,
      window_total: data.window_total,
      label: '',
    };
    keypoints.push(kp);
    renderKeypoints();
    draw2D();
    document.getElementById('status').textContent =
      `已添加 #${kp.id+1}  [${kp.anchor3d.map(x=>Math.round(x)).join(', ')}] mm`;
  } catch(e) {
    document.getElementById('status').textContent = '查表失败: ' + e.message;
  }
  lookingUp = false;
});

imgCanvas.addEventListener('contextmenu', function(e) {
  e.preventDefault();
  const [u, v] = canvasUV(e);
  const idx = findNearbyKP(u, v, 15);
  if (idx >= 0) {
    const removed = keypoints.splice(idx, 1)[0];
    renderKeypoints();
    draw2D();
    document.getElementById('status').textContent = `已删除 #${removed.id+1}`;
  }
});

function draw2D() {
  if (!currentImg) return;
  const w = imgCanvas.width, h = imgCanvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(currentImg, 0, 0, w, h);

  const scale = w / 640;

  // 画自动提案 (虚线圆圈 — 如果没有手动点覆盖它)
  const autoCovered = keypoints.some(kp =>
    Math.abs(kp.uv[0] - autoUV[0]) < 8 && Math.abs(kp.uv[1] - autoUV[1]) < 8
  );
  if (!autoCovered) {
    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.arc(autoUV[0] * scale, autoUV[1] * scale, 10, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    // 小十字
    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.lineWidth = 1;
    const acx = autoUV[0] * scale, acy = autoUV[1] * scale;
    ctx.beginPath(); ctx.moveTo(acx-6, acy); ctx.lineTo(acx+6, acy);
    ctx.moveTo(acx, acy-6); ctx.lineTo(acx, acy+6); ctx.stroke();
  }

  // 辅助延长线: 两个关键点的连线延长至全图, 便于检查与平台是否平行
  if (keypoints.length >= 2) {
    const p1 = keypoints[0].uv, p2 = keypoints[1].uv;
    const dx = p2[0] - p1[0], dy = p2[1] - p1[1];
    if (dx !== 0 || dy !== 0) {
      // 求直线与 640x480 画布四边的参数 t, 延长到整图范围
      const ts = [];
      if (dx !== 0) { ts.push((0 - p1[0]) / dx, (640 - p1[0]) / dx); }
      if (dy !== 0) { ts.push((0 - p1[1]) / dy, (480 - p1[1]) / dy); }
      const tmin = Math.min(...ts), tmax = Math.max(...ts);
      const x1 = (p1[0] + tmin * dx) * scale, y1 = (p1[1] + tmin * dy) * scale;
      const x2 = (p1[0] + tmax * dx) * scale, y2 = (p1[1] + tmax * dy) * scale;
      // 黑色衬底 + 黄色虚线, 任何背景下都可见
      ctx.strokeStyle = 'rgba(0,0,0,0.65)';
      ctx.lineWidth = 3;
      ctx.setLineDash([10, 8]);
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      ctx.strokeStyle = 'rgba(255,235,59,0.9)';
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // 画所有关键点 — 精确十字星
  keypoints.forEach((kp, i) => {
    const cx = kp.uv[0] * scale, cy = kp.uv[1] * scale;
    const color = KP_COLORS[i % KP_COLORS.length];
    const arm = 5;  // 十字臂长

    // 外黑边 (让十字在任何背景下都可见)
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(cx - arm, cy); ctx.lineTo(cx + arm, cy);
    ctx.moveTo(cx, cy - arm); ctx.lineTo(cx, cy + arm);
    ctx.stroke();

    // 内彩色十字
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(cx - arm, cy); ctx.lineTo(cx + arm, cy);
    ctx.moveTo(cx, cy - arm); ctx.lineTo(cx, cy + arm);
    ctx.stroke();

    // YOLO 伪标签用虚线十字区分
    if (kp.label && kp.label.startsWith('YOLO')) {
      ctx.strokeStyle = 'rgba(255,165,0,0.6)';
      ctx.lineWidth = 3;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(cx - arm - 2, cy); ctx.lineTo(cx + arm + 2, cy);
      ctx.moveTo(cx, cy - arm - 2); ctx.lineTo(cx, cy + arm + 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 编号 (右上角小字)
    ctx.fillStyle = '#fff';
    ctx.font = `bold 10px monospace`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText(i + 1, cx + arm + 2, cy - arm);
  });
}

// ═══════════════════════════════════════════════════════
// API
// ═══════════════════════════════════════════════════════
async function loadFrame(stem) {
  document.getElementById('status').textContent = '加载中...';
  keypoints = [];
  nextId = 0;
  auto3D = null;
  renderKeypoints();

  try {
    const resp = await fetch('/api/load/' + stem);
    const data = await resp.json();
    if (data.error) { alert(data.error); return; }

    currentStem = stem;
    autoUV = data.auto_uv;
    auto3D = data.auto_3d;
    loadTimeMs = data.elapsed_ms;

    // 恢复已保存的关键点
    if (data.saved_keypoints && data.saved_keypoints.length > 0) {
      data.saved_keypoints.forEach((skp, i) => {
        keypoints.push({
          id: nextId++,
          uv: skp.pixel_uv || [0, 0],
          anchor3d: skp.anchor_3d,
          median_depth: skp.median_depth,
          valid_count: skp.valid_count,
          label: skp.label || '',
        });
      });
      renderKeypoints();
    }

    currentImg = new Image();
    currentImg.onload = function() {
      const wrap = document.getElementById('canvas-wrap');
      const maxW = wrap.clientWidth * 0.95;
      const maxH = wrap.clientHeight * 0.95;
      const s = Math.min(maxW / currentImg.width, maxH / currentImg.height, 1.0);
      imgCanvas.width = Math.round(currentImg.width * s);
      imgCanvas.height = Math.round(currentImg.height * s);
      draw2D();
    };
    currentImg.src = 'data:image/jpeg;base64,' + data.img_b64;

    updateInfoPanel();
    updateFrameListHighlight();
    document.getElementById('status').textContent =
      `OK (${loadTimeMs}ms) — ${keypoints.length} 关键点`;
  } catch(e) {
    document.getElementById('status').textContent = '加载失败: ' + e.message;
  }
}

async function doConfirm() {
  // 无手动关键点 → 自动确认 auto anchor
  if (keypoints.length === 0 && auto3D) {
    keypoints.push({
      id: nextId++,
      uv: [...autoUV],
      anchor3d: auto3D,
      median_depth: null,
      valid_count: null,
      label: 'auto',
    });
    renderKeypoints();
    draw2D();
  }
  if (keypoints.length === 0) {
    alert('没有关键点可保存。请先左键点击图像添加关键点。');
    return;
  }
  await saveAndNext();
}

async function doSave() {
  if (keypoints.length === 0) {
    alert('没有关键点可保存。');
    return;
  }
  await saveFrame(false);
  document.getElementById('status').textContent =
    `已保存 ${keypoints.length} 个关键点 ✓`;
}

async function saveAndNext() {
  const ok = await saveFrame(true);
  if (ok) {
    setTimeout(() => {
      if (currentIdx < frames.length - 1) { currentIdx++; loadFrame(frames[currentIdx].stem); }
    }, 300);
  }
}

async function saveFrame(advance) {
  document.getElementById('status').textContent = '保存中...';
  try {
    const payload = {
      stem: currentStem,
      keypoints: keypoints.map(kp => ({
        id: kp.id,
        pixel_uv: kp.uv,
        anchor_3d: kp.anchor3d,
        median_depth: kp.median_depth,
        valid_count: kp.valid_count,
        label: kp.label || '',
      })),
    };
    const resp = await fetch('/api/save_frame', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!data.success) { alert('Save failed'); return false; }

    // 更新帧列表
    const frame = frames.find(f => f.stem === currentStem);
    if (frame) { frame.done = true; frame.skipped = false; frame.n_keypoints = keypoints.length; }
    renderFrameList();
    return true;
  } catch(e) {
    document.getElementById('status').textContent = '保存失败: ' + e.message;
    return false;
  }
}

async function doSkip() {
  try {
    await fetch('/api/skip', { method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({stem: currentStem}),
    });
  } catch(e) {}
  const frame = frames.find(f => f.stem === currentStem);
  if (frame) { frame.done = true; frame.skipped = true; frame.n_keypoints = 0; }
  renderFrameList();
  setTimeout(() => { if (currentIdx < frames.length - 1) { currentIdx++; loadFrame(frames[currentIdx].stem); } }, 200);
}

function doClear() {
  if (keypoints.length === 0) return;
  keypoints = [];
  renderKeypoints();
  draw2D();
  document.getElementById('status').textContent = '已清空所有关键点';
}

// ═══════════════════════════════════════════════════════
// 导航
// ═══════════════════════════════════════════════════════
function goNext() {
  if (currentIdx < frames.length - 1) { currentIdx++; loadFrame(frames[currentIdx].stem); }
}
function goPrev() {
  if (currentIdx > 0) { currentIdx--; loadFrame(frames[currentIdx].stem); }
}
function goTo(idx) {
  if (idx >= 0 && idx < frames.length) { currentIdx = idx; loadFrame(frames[currentIdx].stem); }
}

// ═══════════════════════════════════════════════════════
// UI 更新
// ═══════════════════════════════════════════════════════
function updateInfoPanel() {
  document.getElementById('info-auto').textContent =
    `(${autoUV[0]}, ${autoUV[1]})` + (auto3D ? ' → ' + auto3D.map(Math.round).join(',') : '');
  document.getElementById('info-kp-count').textContent = keypoints.length;
  // 辅助线角度: 前两个关键点连线相对水平方向的角度 (图像坐标, y 向下)
  const guideEl = document.getElementById('info-guide');
  if (keypoints.length >= 2) {
    const [u1, v1] = keypoints[0].uv;
    const [u2, v2] = keypoints[1].uv;
    const deg = Math.atan2(v2 - v1, u2 - u1) * 180 / Math.PI;
    guideEl.textContent = (deg >= 0 ? '+' : '') + deg.toFixed(1) + '°';
  } else {
    guideEl.textContent = '-';
  }
  document.getElementById('frame-label').textContent = `${currentIdx + 1} / ${frames.length}`;
  document.getElementById('info-time').textContent = loadTimeMs + ' ms';
}

function renderKeypoints() {
  const panel = document.getElementById('kp-list');
  document.getElementById('kp-panel-count').textContent = keypoints.length + ' 个';

  if (keypoints.length === 0) {
    panel.innerHTML = '<div style="padding:20px;text-align:center;color:#555;font-size:13px">'
      + '左键点击图像添加关键点<br>'
      + (auto3D ? '<span style="color:#888">ENTER 可快速确认自动提案</span>' : '')
      + '</div>';
    updateInfoPanel();
    return;
  }

  let html = '';
  keypoints.forEach((kp, i) => {
    const color = KP_COLORS[i % KP_COLORS.length];
    const a = kp.anchor3d;
    html += `<div class="kp-item">`;
    html += `<div class="kp-top">`;
    html += `<span class="kp-dot" style="background:${color}"></span>`;
    html += `<span class="kp-num" style="color:${color}">#${i+1}</span>`;
    html += `<span class="kp-uv">(${kp.uv[0]}, ${kp.uv[1]})</span>`;
    html += `<span class="kp-del" title="删除" onclick="delKP(${i})">&times;</span>`;
    html += `</div>`;
    html += `<div class="kp-3d">[${a[0].toFixed(0)}, ${a[1].toFixed(0)}, ${a[2].toFixed(0)}]</div>`;
    html += `<div class="kp-meta">`;
    html += `深度 ${kp.median_depth != null ? kp.median_depth.toFixed(0) : '?'} mm`;
    if (kp.valid_count != null) html += ` | 窗口 ${kp.valid_count}/${kp.window_total}`;
    html += `</div>`;
    html += `<div class="kp-label">`;
    html += `<select onchange="setLabel(${i}, this.value)" value="${kp.label || ''}">`;
    html += `<option value="" ${!kp.label?'selected':''}>-- 无标签 --</option>`;
    html += `<option value="corner" ${kp.label==='corner'?'selected':''}>角点 corner</option>`;
    html += `<option value="center" ${kp.label==='center'?'selected':''}>中心 center</option>`;
    html += `<option value="beam-end" ${kp.label==='beam-end'?'selected':''}>横梁端 beam-end</option>`;
    html += `<option value="pillar" ${kp.label==='pillar'?'selected':''}>立柱 pillar</option>`;
    html += `<option value="other" ${kp.label==='other'?'selected':''}>其他 other</option>`;
    html += `</select>`;
    html += `</div>`;
    html += `</div>`;
  });
  panel.innerHTML = html;
  updateInfoPanel();
}

function delKP(i) {
  keypoints.splice(i, 1);
  renderKeypoints();
  draw2D();
  document.getElementById('status').textContent = `已删除关键点`;
}

function setLabel(i, val) {
  keypoints[i].label = val;
}

function renderFrameList() {
  const panel = document.getElementById('frame-list');
  const done = frames.filter(f => f.done && !f.skipped).length;
  const skipped = frames.filter(f => f.skipped).length;
  document.getElementById('list-stats').textContent =
    `总${frames.length} | ✓${done} | ↷${skipped} | ○${frames.length - done - skipped}`;

  let html = '';
  frames.forEach((f, i) => {
    let cls = 'item';
    if (i === currentIdx) cls += ' current';
    if (f.skipped) cls += ' skipped';
    else if (f.done) cls += ' done';
    else cls += ' pending';
    const icon = f.skipped ? '↷' : (f.done ? '✓' : '○');
    const nkp = f.n_keypoints || 0;
    html += `<div class="${cls}" onclick="goTo(${i})">`;
    html += `<span>${icon} #${i+1} ${f.num_id}</span>`;
    html += nkp > 0 ? `<span class="kp-badge">${nkp} pts</span>` : '';
    html += `</div>`;
  });
  panel.innerHTML = html;
}

function updateFrameListHighlight() {
  const items = document.querySelectorAll('#frame-list .item');
  items.forEach((el, i) => el.classList.toggle('current', i === currentIdx));
}

// ═══════════════════════════════════════════════════════
// 键盘
// ═══════════════════════════════════════════════════════
document.addEventListener('keydown', function(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  switch(e.key) {
    case 'Enter': e.preventDefault(); doConfirm(); break;
    case 'Escape': e.preventDefault(); doSkip(); break;
    case 'ArrowLeft': e.preventDefault(); goPrev(); break;
    case 'ArrowRight': e.preventDefault(); goNext(); break;
    case 'g': case 'G':
      e.preventDefault();
      const n = prompt('跳转到帧号 (1-' + frames.length + '):');
      if (n) { const idx = parseInt(n) - 1; if (idx >= 0 && idx < frames.length) goTo(idx); }
      break;
    case 's': case 'S': e.preventDefault(); doSave(); break;
    case 'c': case 'C': e.preventDefault(); doClear(); break;
  }
});

// ═══════════════════════════════════════════════════════
// 启动
// ═══════════════════════════════════════════════════════
window.addEventListener('resize', draw2D);

async function init() {
  try {
    const resp = await fetch('/api/frames');
    frames = await resp.json();
    const firstPending = frames.findIndex(f => !f.done && !f.skipped);
    currentIdx = firstPending >= 0 ? firstPending : 0;
    renderFrameList();
    if (frames.length > 0) loadFrame(frames[currentIdx].stem);
  } catch(e) {
    document.getElementById('status').textContent = '连接服务器失败: ' + e.message;
  }
}

init();
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# 启动
# ══════════════════════════════════════════════════════════════════════════════

def _scan_frames(data_dir: str):
    """扫描 data_dir 下所有 PCD + 对应图片。支持 TV_*.pcd 和 cluster*_TV_*.pcd。"""
    global STEMS, FRAMES
    p = Path(data_dir)
    pcds = sorted(set(p.glob("TV_*.pcd")) | set(p.glob("cluster*_TV_*.pcd")))
    STEMS = []
    FRAMES = []
    for pcd_path in pcds:
        stem = pcd_path.stem
        # Image lookup: try stem first, then num_id as fallback
        img_path = None
        for ext in [".jpg", ".png"]:
            for cand_name in [stem, stem.replace("TV_", "")]:
                cand = p / f"{cand_name}{ext}"
                if cand.exists():
                    img_path = str(cand)
                    break
            if img_path:
                break
        if img_path:
            STEMS.append(stem)
            FRAMES.append({
                "stem": stem,
                "num_id": stem.replace("TV_", ""),
                "pcd_path": str(pcd_path),
                "img_path": img_path,
                "has_result": False,
            })
    return FRAMES


def main():
    parser = argparse.ArgumentParser(description="Shelf Anchor V2 — Web 批量审核")
    parser.add_argument("--data_dir", default="data/new_sheef/prototypes")
    parser.add_argument("--output_dir", default="data/new_sheef/prototype_annotations")
    parser.add_argument("--yolo_labels_dir", default=None,
                        help="YOLO labels directory (e.g. datasets/shelf_pose_pseudo/labels) — load pseudo-labels as initial keypoints")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    # Camera
    parser.add_argument("--fx", type=float, default=410.9)
    parser.add_argument("--fy", type=float, default=410.9)
    parser.add_argument("--cx", type=float, default=307.0)
    parser.add_argument("--cy", type=float, default=264.3)
    parser.add_argument("--img_w", type=int, default=640)
    parser.add_argument("--img_h", type=int, default=480)
    args = parser.parse_args()

    global DATA_DIR, OUTPUT_DIR, YOLO_LABELS_DIR, FX, FY, CX, CY, IMG_W, IMG_H
    DATA_DIR = args.data_dir
    OUTPUT_DIR = args.output_dir
    YOLO_LABELS_DIR = args.yolo_labels_dir
    FX, FY, CX, CY = args.fx, args.fy, args.cx, args.cy
    IMG_W, IMG_H = args.img_w, args.img_h

    frames = _scan_frames(DATA_DIR)
    print(f"Found {len(frames)} frames in {DATA_DIR}")

    # 如果指定了 YOLO labels dir，只显示有 YOLO 标签的帧
    if YOLO_LABELS_DIR:
        global FRAMES, STEMS
        yolo_stems = set()
        for split in ("train", "val"):
            split_dir = os.path.join(YOLO_LABELS_DIR, split)
            if os.path.isdir(split_dir):
                for f in os.listdir(split_dir):
                    if f.endswith(".txt"):
                        yolo_stems.add(f.replace(".txt", ""))
        STEMS = [s for s in STEMS if s in yolo_stems]
        FRAMES = [f for f in FRAMES if f["stem"] in yolo_stems]
        print(f"Filtered to {len(FRAMES)} frames with YOLO labels")

    print(f"Output: {OUTPUT_DIR}")
    if YOLO_LABELS_DIR:
        print(f"YOLO labels: {YOLO_LABELS_DIR} (pseudo-labels as initial keypoints)")
    print(f"\nOpen http://localhost:{args.port} in your browser")
    print(f"\nMulti-keypoint annotation:")
    print(f"  Left-click  — Add keypoint (auto PCD lookup)")
    print(f"  Right-click — Delete nearest keypoint")
    print(f"  ENTER      — Save all & next frame (ENTER on empty = auto-confirm auto-anchor)")
    print(f"  S          — Save without advancing")
    print(f"  ESC        — Skip current frame")
    print(f"  ←/→        — Prev/next frame")
    print(f"  G          — Jump to frame by number")
    print(f"  C          — Clear all keypoints on current frame")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
