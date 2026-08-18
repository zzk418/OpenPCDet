#!/usr/bin/env python3
"""
Point-SAM Web Review — 3D 可视化审核工具
==========================================
轻量 Web 前端: 查看 Point-SAM 自动生成的 proposals, 点击分配类别,
自动保存到 _info.json。

用法:
  python pointsam_web_review.py                          # 默认
  python pointsam_web_review.py --port 8080               # 自定义端口
  python pointsam_web_review.py --annotation_dir output/shelf_annotations

快捷键:
  1/2/3/4/0   - 分配类别 (beam/pillar/pallet/goods/discard)
  N/P         - 下一帧/上一帧
  Space       - 取消选中
  G           - 跳转到指定帧号
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request

# Point-SAM (lazy import)
_SAM_ANNOTATOR = None
_SAM_AVAILABLE = False

def _init_sam(ckpt_path: str):
    """延迟加载 Point-SAM 模型 (仅当 --sam_ckpt 指定时)。"""
    global _SAM_ANNOTATOR, _SAM_AVAILABLE
    import torch
    import hydra
    from omegaconf import OmegaConf

    # Stubs 必须先注入，再导入 Point-SAM
    from pc_sam.torkit3d_stub import inject as inject_stubs
    inject_stubs()

    config_dir = "/code/Point-SAM/configs"
    try:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    except Exception:
        pass
    hydra.initialize_config_dir(config_dir, version_base=None)
    cfg = hydra.compose(config_name="large")
    OmegaConf.resolve(cfg)

    from safetensors.torch import load_model

    model = hydra.utils.instantiate(cfg.model)
    # fused layernorm is optional (needs apex), model works without it
    try:
        from pc_sam.utils.torch_utils import replace_with_fused_layernorm
        model.apply(replace_with_fused_layernorm)
    except Exception:
        pass
    load_model(model, ckpt_path)
    model.eval().cuda()
    _SAM_ANNOTATOR = model
    _SAM_AVAILABLE = True
    print(f"[SAM] Loaded: {ckpt_path}")

# ── 全局状态 ──
ANNOTATION_DIR = None
PCD_DIR = None
STEMS = []

# ── 类别定义 ──
CLASSES = {
    0: "background",
    1: "beam",       # 横梁
    2: "pillar",     # 立柱
    3: "pallet",     # 卡板
    4: "goods",      # 货物
}

CLASS_COLORS_HEX = {
    0: "#888888",
    1: "#ff4444",    # red: beam
    2: "#4488ff",    # blue: pillar
    3: "#44cc44",    # green: pallet
    4: "#ffcc00",    # yellow: goods
}

CLASS_COLORS_RGB = {
    0:  [0.55, 0.55, 0.55],
    1:  [1.0,  0.27, 0.27],
    2:  [0.27, 0.53, 1.0],
    3:  [0.27, 0.80, 0.27],
    4:  [1.0,  0.80, 0.0],
}

app = Flask(__name__)


# ═══════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════

@app.route("/api/stems")
def api_stems():
    """返回所有 PCD stems 及其标注状态。"""
    items = []
    for stem in STEMS:
        info_path = os.path.join(ANNOTATION_DIR, f"{stem}_info.json")
        npz_path = os.path.join(ANNOTATION_DIR, f"{stem}_proposals.npz")
        status = {"stem": stem}

        if os.path.exists(npz_path):
            try:
                data = np.load(npz_path, allow_pickle=True)
                status["n_points"] = int(data["xyz"].shape[0])
                status["n_proposals"] = int(data["masks"].shape[0])
            except Exception:
                status["n_points"] = 0
                status["n_proposals"] = 0
        else:
            status["n_points"] = 0
            status["n_proposals"] = 0

        if os.path.exists(info_path):
            try:
                with open(info_path) as f:
                    info = json.load(f)
                labeled = sum(1 for p in info.get("proposals", [])
                              if p.get("class_label") and p["class_label"] != "background")
                status["n_labeled"] = labeled
            except Exception:
                status["n_labeled"] = 0
        else:
            status["n_labeled"] = 0

        items.append(status)

    return jsonify(items)


@app.route("/api/load/<stem>")
def api_load(stem):
    """加载一帧的 proposals 数据。"""
    npz_path = os.path.join(ANNOTATION_DIR, f"{stem}_proposals.npz")
    info_path = os.path.join(ANNOTATION_DIR, f"{stem}_info.json")

    if not os.path.exists(npz_path):
        return jsonify({"error": f"Not found: {npz_path}"}), 404

    data = np.load(npz_path, allow_pickle=True)
    xyz = data["xyz"].astype(np.float32)      # (N, 3) mm
    rgb = data.get("rgb", None)               # (N, 3) float [0,1]
    masks = data["masks"]                     # (M, N) bool
    M, N = masks.shape

    # RGB → uint8 [0,255]
    if rgb is not None:
        rgb_uint8 = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    else:
        # 灰色默认
        rgb_uint8 = np.full((N, 3), 128, dtype=np.uint8)

    # 加载已有 labels
    existing_labels = {}
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        for i, p in enumerate(info.get("proposals", [])):
            lbl = p.get("class_label")
            if lbl and lbl != "background":
                existing_labels[str(i)] = lbl

    # 构建 proposals 列表 (只传点索引，压缩传输)
    proposals = []
    for i in range(M):
        indices = np.where(masks[i])[0]
        if len(indices) > 0:
            proposals.append({
                "id": i,
                "indices": indices.astype(np.int32).tolist(),
                "label": existing_labels.get(str(i)),
                "iou_score": float(data["iou_scores"][i]) if "iou_scores" in data else 0.0,
                "n_points": int(len(indices)),
            })

    # 按点数升序排列 (小 proposal 优先显示)
    proposals.sort(key=lambda p: p["n_points"])

    return jsonify({
        "stem": stem,
        "xyz": xyz.ravel().tolist(),            # flat [x0,y0,z0, x1,y1,z1, ...]
        "rgb": rgb_uint8.ravel().tolist(),       # flat [r0,g0,b0, r1,g1,b1, ...]
        "n_points": N,
        "proposals": proposals,
        "class_colors": CLASS_COLORS_HEX,
        "classes": {str(k): v for k, v in CLASSES.items()},
    })


@app.route("/api/label", methods=["POST"])
def api_label():
    """保存一个 proposal 的类别标签。"""
    data = request.get_json()
    stem = data["stem"]
    proposal_id = int(data["proposal_id"])
    class_label = data["class_label"]  # "beam" | "pillar" | "pallet" | "goods" | "background"

    info_path = os.path.join(ANNOTATION_DIR, f"{stem}_info.json")

    # 读取现有 info
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
    else:
        info = {"stem": stem, "proposals": []}

    # 确保 proposals 列表足够长
    while len(info["proposals"]) <= proposal_id:
        info["proposals"].append({})

    info["proposals"][proposal_id]["class_label"] = class_label

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    return jsonify({"ok": True, "proposal_id": proposal_id, "class_label": class_label})


@app.route("/api/stats")
def api_stats():
    """汇总统计。"""
    total = len(STEMS)
    labeled_count = 0
    proposal_counts = {c: 0 for c in CLASSES.values()}

    for stem in STEMS:
        info_path = os.path.join(ANNOTATION_DIR, f"{stem}_info.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
            has_label = False
            for p in info.get("proposals", []):
                lbl = p.get("class_label") or "background"
                if lbl and lbl != "background":
                    has_label = True
                    proposal_counts[lbl] = proposal_counts.get(lbl, 0) + 1
            if has_label:
                labeled_count += 1

    return jsonify({
        "total_frames": total,
        "labeled_frames": labeled_count,
        "proposal_counts": proposal_counts,
    })


# ── 关键点 API ──

@app.route("/api/keypoints/<stem>", methods=["GET", "POST"])
def api_keypoints(stem):
    """加载或保存关键点。"""
    kp_path = os.path.join(ANNOTATION_DIR, f"{stem}_keypoints.json")

    if request.method == "GET":
        if os.path.exists(kp_path):
            with open(kp_path) as f:
                return jsonify(json.load(f))
        return jsonify({"keypoints": []})

    # POST: 保存
    data = request.get_json()
    os.makedirs(ANNOTATION_DIR, exist_ok=True)
    with open(kp_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return jsonify({"ok": True, "n_keypoints": len(data.get("keypoints", []))})


# ── SAM 辅助角点检测 ──

@app.route("/api/sam_corners", methods=["POST"])
def api_sam_corners():
    """Point-SAM 辅助角点检测 — 一键分割货架结构 + 角点检测。

    输入: {stem, prompts: [[x,y,z],...]}  — 点击货架结构上的任意点
    输出: {keypoints: [{id, type, position},...]}
    """
    if not _SAM_AVAILABLE:
        return jsonify({"error": "SAM not loaded. Start with --sam_ckpt"}), 400

    data = request.get_json()
    stem = data["stem"]
    prompts = data.get("prompts", [])

    if not prompts:
        return jsonify({"error": "Need at least one shelf prompt"}), 400

    # 加载 PCD
    pcd_path = os.path.join(PCD_DIR, f"{stem}.pcd")
    if not os.path.exists(pcd_path):
        return jsonify({"error": f"PCD not found: {pcd_path}"}), 404

    xyz, rgb = _read_pcd_simple(pcd_path)

    import torch

    # 编码点云
    N = xyz.shape[0]
    min_pts = 2048
    if N < min_pts:
        rep = (min_pts + N - 1) // N
        xyz_pad = np.tile(xyz, (rep, 1))[:min_pts]
        rgb_pad = np.tile(rgb, (rep, 1))[:min_pts]
    else:
        xyz_pad = xyz
        rgb_pad = rgb

    coords_t = torch.from_numpy(xyz_pad).float().cuda().unsqueeze(0)
    colors_t = torch.from_numpy(rgb_pad).float().cuda().unsqueeze(0)

    # 归一化
    center = coords_t.mean(dim=1, keepdim=True)
    coords_t = coords_t - center
    scale = coords_t.norm(dim=2, keepdim=True).max().clamp(min=1e-6)
    coords_t = coords_t / scale

    with torch.no_grad():
        embeddings, patches = _SAM_ANNOTATOR.pc_encoder(coords_t, colors_t)

    # 所有提示点 = 货架前景 (label=1)
    prompt_xyz = np.array(prompts, dtype=np.float32)
    prompt_labels = np.ones(len(prompts), dtype=np.int64)

    prompt_t = torch.from_numpy(prompt_xyz).float().cuda().unsqueeze(0)
    # 用相同的归一化参数
    prompt_t = prompt_t - center
    prompt_t = prompt_t / scale
    label_t = torch.from_numpy(prompt_labels).long().cuda().unsqueeze(0)

    # Point-SAM 推理
    with torch.no_grad():
        masks, iou_preds = _SAM_ANNOTATOR.predict_masks(
            coords=coords_t,
            features=colors_t,
            prompt_coords=prompt_t,
            prompt_labels=label_t,
            prompt_masks=None,
            multimask_output=True,
        )

    # masks: (1, num_outputs, N_pad)
    # 取 IoU 最高的 mask
    best_idx = iou_preds[0].argmax().item()
    structure_mask = masks[0, best_idx, :N].cpu().numpy() > 0.5
    n_structure = structure_mask.sum()

    if n_structure < 100:
        return jsonify({"keypoints": [], "warning": f"Only {n_structure} pts in mask"})

    # 在纯净的结构 mask 上做网格角点检测
    xyz_struct = xyz[structure_mask]
    corners = _detect_grid_corners_simple(xyz_struct)

    # Build output
    keypoints = []
    kid = 0
    for c in corners.get("outer", []):
        keypoints.append({"id": kid, "type": "outer", "position": c["position"]})
        kid += 1
    for c in corners.get("inner", []):
        keypoints.append({"id": kid, "type": "inner", "position": c["position"]})
        kid += 1

    return jsonify({
        "keypoints": keypoints,
        "n_structure_points": int(n_structure),
    })


def _read_pcd_simple(path: str):
    """简化 PCD 读取，返回 xyz(N,3) float32, rgb(N,3) float32[0,1]."""
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


def _detect_grid_corners_simple(xyz: np.ndarray) -> dict:
    """网格角点检测的简化版（复用 shelf_grid_corners 的逻辑）。"""
    from scipy.signal import find_peaks

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if len(xyz) < 100:
        return {"outer": [], "inner": []}

    # X 峰值 (立柱)
    x_hist, x_edges = np.histogram(x, bins=max(5, int((x.max()-x.min())/100)))
    x_peaks, _ = find_peaks(x_hist, prominence=x_hist.max()*0.05, distance=2)
    x_pos = [(x_edges[p]+x_edges[p+1])/2 for p in x_peaks] if len(x_peaks) >= 2 else [x.min(), x.max()]

    # Z 峰值 (横梁)
    z_hist, z_edges = np.histogram(z, bins=max(5, int((z.max()-z.min())/120)))
    z_peaks, _ = find_peaks(z_hist, prominence=z_hist.max()*0.04, distance=2)
    z_pos = [(z_edges[p]+z_edges[p+1])/2 for p in z_peaks] if len(z_peaks) >= 2 else [z.min(), z.max()]

    corners = []
    radius = 120
    for xp in x_pos:
        for zp in z_pos:
            nearby = (abs(x - xp) < radius) & (abs(z - zp) < radius)
            y_val = float(y[nearby].mean()) if nearby.sum() > 0 else float(np.median(y))
            corners.append({
                "position": [float(xp), y_val, float(zp)],
                "n_nearby": int(nearby.sum()),
            })

    if len(corners) < 4:
        return {"outer": corners, "inner": []}

    all_x = [c["position"][0] for c in corners]
    all_z = [c["position"][2] for c in corners]
    x_min, x_max = min(all_x), max(all_x)
    z_min, z_max = min(all_z), max(all_z)
    tol = 200

    outer, inner = [], []
    for c in corners:
        px, _, pz = c["position"]
        on_x = (abs(px-x_min)<tol) or (abs(px-x_max)<tol)
        on_z = (abs(pz-z_min)<tol) or (abs(pz-z_max)<tol)
        c["type"] = "outer" if (on_x and on_z) else "inner"
        (outer if c["type"] == "outer" else inner).append(c)

    # 补全缺失外角点
    if len(outer) < 4 and len(all_x) >= 2 and len(all_z) >= 2:
        outer_set = set()
        for o in outer:
            ox = min(all_x, key=lambda g: abs(g-o["position"][0]))
            oz = min(all_z, key=lambda g: abs(g-o["position"][2]))
            outer_set.add((ox, oz))
        for ex, ez in [(x_min,z_min),(x_min,z_max),(x_max,z_min),(x_max,z_max)]:
            if not any(abs(ex-ox)<tol and abs(ez-oz)<tol for ox,oz in outer_set):
                nearby = (abs(x-ex)<radius) & (abs(z-ez)<radius)
                y_fill = float(y[nearby].mean()) if nearby.sum()>0 else float(np.median(y))
                outer.append({"position":[float(ex),y_fill,float(ez)],"n_nearby":int(nearby.sum()),"type":"outer"})

    return {"outer": outer, "inner": inner}


# ═══════════════════════════════════════════════════════════
# HTML 前端 (内联)
# ═══════════════════════════════════════════════════════════

HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Point-SAM Review — 货架标注审核</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       overflow: hidden; background: #1a1a2e; color: #eee; }

#canvas { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: 1; }

/* ── Top Bar ── */
#topbar { position: fixed; top: 0; left: 0; right: 0; z-index: 10;
          display: flex; align-items: center; gap: 12px; padding: 8px 14px;
          background: rgba(0,0,0,0.75); backdrop-filter: blur(8px);
          border-bottom: 1px solid rgba(255,255,255,0.1); }
#topbar .stem { font-weight: 600; font-size: 14px; min-width: 180px; }
#topbar .status { font-size: 12px; color: #aaa; }
#topbar button { padding: 6px 14px; border: 1px solid rgba(255,255,255,0.2);
                 background: rgba(255,255,255,0.08); color: #ddd; border-radius: 5px;
                 cursor: pointer; font-size: 12px; }
#topbar button:hover { background: rgba(255,255,255,0.18); }
#topbar button.active { background: #3b82f6; border-color: #3b82f6; }
#topbar .sep { width: 1px; height: 20px; background: rgba(255,255,255,0.15); }
#topbar input[type=number] { width: 50px; padding: 4px 6px; background: rgba(255,255,255,0.1);
                             border: 1px solid rgba(255,255,255,0.2); color: #fff;
                             border-radius: 4px; text-align: center; font-size: 12px; }

/* ── Sidebar: Info Panel ── */
#sidebar { position: fixed; top: 52px; right: 12px; z-index: 10;
           width: 260px; background: rgba(0,0,0,0.8); backdrop-filter: blur(8px);
           border-radius: 10px; border: 1px solid rgba(255,255,255,0.12);
           padding: 14px; font-size: 12px; max-height: calc(100vh - 120px);
           overflow-y: auto; }
#sidebar h3 { font-size: 13px; margin-bottom: 8px; color: #aaa; text-transform: uppercase;
              letter-spacing: 1px; }
#sidebar .prop { padding: 6px 8px; margin: 3px 0; border-radius: 5px; cursor: pointer;
                 border: 1px solid transparent; transition: all 0.15s; }
#sidebar .prop:hover { border-color: rgba(255,255,255,0.3); }
#sidebar .prop.selected { border-color: #ffaa00; background: rgba(255,170,0,0.15); }
#sidebar .prop .id { font-weight: 600; }
#sidebar .prop .pts { color: #888; }
#sidebar .prop .lbl { float: right; font-size: 10px; padding: 1px 6px; border-radius: 3px; }
#sidebar .prop .lbl.beam { background: #ff4444; color: #fff; }
#sidebar .prop .lbl.pillar { background: #4488ff; color: #fff; }
#sidebar .prop .lbl.pallet { background: #44cc44; color: #000; }
#sidebar .prop .lbl.goods { background: #ffcc00; color: #000; }

/* ── Bottom Class Buttons ── */
#classbar { position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%); z-index: 10;
            display: flex; gap: 10px; padding: 10px 20px;
            background: rgba(0,0,0,0.8); backdrop-filter: blur(8px);
            border-radius: 12px; border: 1px solid rgba(255,255,255,0.15); }
#classbar button { padding: 10px 20px; border: 2px solid transparent; border-radius: 8px;
                   font-size: 13px; font-weight: 600; cursor: pointer; color: #fff;
                   background: rgba(255,255,255,0.08); transition: all 0.15s; }
#classbar button:hover { transform: translateY(-2px); }
#classbar button .key { font-size: 10px; opacity: 0.7; display: block; }
#classbar .btn-beam { border-color: #ff4444; }
#classbar .btn-beam:hover, .btn-beam.pressed { background: #ff4444; }
#classbar .btn-pillar { border-color: #4488ff; }
#classbar .btn-pillar:hover, .btn-pillar.pressed { background: #4488ff; }
#classbar .btn-pallet { border-color: #44cc44; }
#classbar .btn-pallet:hover, .btn-pallet.pressed { background: #44cc44; }
#classbar .btn-goods { border-color: #ffcc00; color: #000; }
#classbar .btn-goods:hover, .btn-goods.pressed { background: #ffcc00; }
#classbar .btn-discard { border-color: #666; }
#classbar .btn-discard:hover, .btn-discard.pressed { background: #666; }

/* ── Hover tooltip ── */
#tooltip { position: fixed; z-index: 11; pointer-events: none; display: none;
           background: rgba(0,0,0,0.85); padding: 6px 10px; border-radius: 5px;
           font-size: 11px; color: #fff; }

/* ── Toast ── */
#toast { position: fixed; bottom: 100px; left: 50%; transform: translateX(-50%); z-index: 20;
         padding: 8px 20px; border-radius: 8px; font-size: 13px; font-weight: 600;
         background: #22c55e; color: #000; display: none; }

/* ── Help Panel ── */
#help-overlay { position: fixed; top: 0; left: 0; right: 0; bottom: 0; z-index: 100;
                background: rgba(0,0,0,0.6); display: flex; align-items: center; justify-content: center; }
#help-panel { background: #1e293b; border: 1px solid rgba(255,255,255,0.2); border-radius: 14px;
              padding: 28px 32px; max-width: 520px; font-size: 13px; line-height: 1.8;
              box-shadow: 0 12px 40px rgba(0,0,0,0.5); }
#help-panel h2 { font-size: 18px; margin-bottom: 14px; }
#help-panel .step { display: flex; gap: 10px; margin: 8px 0; align-items: flex-start; }
#help-panel .step .num { background: #3b82f6; color: #fff; min-width: 22px; height: 22px;
                         border-radius: 11px; text-align: center; line-height: 22px;
                         font-size: 11px; font-weight: 700; flex-shrink: 0; margin-top: 1px; }
#help-panel .key-tag { display: inline-block; background: #334155; color: #f1f5f9;
                       padding: 2px 7px; border-radius: 4px; font-weight: 700; font-size: 11px;
                       margin: 0 2px; border: 1px solid #475569; }
#help-panel .cls-beam { color: #ff6666; }
#help-panel .cls-pillar { color: #66aaff; }
#help-panel .cls-pallet { color: #55dd55; }
#help-panel .cls-goods { color: #ffdd44; }
#help-panel button { margin-top: 16px; padding: 8px 24px; background: #3b82f6; color: #fff;
                     border: none; border-radius: 8px; font-size: 14px; cursor: pointer; }
#help-btn { position: fixed; bottom: 18px; right: 20px; z-index: 99;
            width: 32px; height: 32px; border-radius: 16px; background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.25); color: #ccc; font-size: 16px;
            font-weight: 700; cursor: pointer; }
</style>
</head>
<body>

<canvas id="canvas"></canvas>

<!-- Top Bar -->
<div id="topbar">
  <span class="stem" id="stemLabel">—</span>
  <span class="sep"></span>
  <button id="btnPrev" title="上一帧 (P)">◀ Prev</button>
  <button id="btnNext" title="下一帧 (N)">Next ▶</button>
  <span class="sep"></span>
  <input type="number" id="gotoIdx" min="0" placeholder="帧号" title="跳转到帧号 (G)">
  <button id="btnGoto">Go</button>
  <span class="sep"></span>
  <span class="status" id="statusLabel">加载中...</span>
  <span style="flex:1"></span>
  <button id="btnMode" onclick="toggleMode()" style="background:#3b82f6;font-weight:700">📋 Proposals</button>
  <span class="status" id="statsLabel"></span>
</div>

<!-- Sidebar -->
<div id="sidebar">
  <div id="sidebarProposals">
    <h3>📋 Proposals</h3>
    <div id="propList">—</div>
  </div>
  <div id="sidebarKeypoints" style="display:none">
    <h3>📍 角点 <span style="font-weight:400;font-size:10px;color:#888" id="kpCount">0</span></h3>
    <div id="kpList"></div>
    <button onclick="clearAllKeypoints()" style="margin-top:8px;padding:4px 10px;font-size:10px;
      background:rgba(255,255,255,0.08);color:#f87171;border:1px solid #f87171;border-radius:4px;cursor:pointer">
      清除全部</button>
  </div>
</div>

<!-- Bottom Bar: mode-dependent -->
<div id="classbar" class="mode-proposals">
  <button class="btn-beam" onclick="assignClass('beam')">
    🔴 横梁<span class="key">Key 1</span></button>
  <button class="btn-pillar" onclick="assignClass('pillar')">
    🔵 立柱<span class="key">Key 2</span></button>
  <button class="btn-pallet" onclick="assignClass('pallet')">
    🟢 卡板<span class="key">Key 3</span></button>
  <button class="btn-goods" onclick="assignClass('goods')">
    🟡 货物<span class="key">Key 4</span></button>
  <button class="btn-discard" onclick="assignClass('background')">
    ⚫ 丢弃<span class="key">Key 0</span></button>
</div>

<div id="kpclassbar" style="position:fixed;bottom:16px;left:50%;transform:translateX(-50%);z-index:10;
     display:none; gap:10px; padding:10px 20px; background:rgba(0,0,0,0.8); backdrop-filter:blur(8px);
     border-radius:12px; border:1px solid rgba(255,255,255,0.15);">
  <button onclick="setKeypointType('outer')" id="btnOuter"
    style="padding:10px 20px;border:2px solid #ff9800;border-radius:8px;font-size:13px;font-weight:600;
    background:#ff9800;color:#000;cursor:pointer">
    📐 外角点<span class="key">Key 1</span></button>
  <button onclick="setKeypointType('inner')" id="btnInner"
    style="padding:10px 20px;border:2px solid #00bcd4;border-radius:8px;font-size:13px;font-weight:600;
    background:rgba(255,255,255,0.08);color:#fff;cursor:pointer">
    📏 内角点<span class="key">Key 2</span></button>
  <button onclick="deleteSelectedKeypoint()" id="btnDelKp"
    style="padding:10px 14px;border:2px solid #f87171;border-radius:8px;font-size:13px;
    background:rgba(255,255,255,0.08);color:#f87171;cursor:pointer">
    🗑 删除<span class="key">Del</span></button>
  <span class="sep" style="width:1px;height:28px;background:rgba(255,255,255,0.15);align-self:center"></span>
  <button onclick="runSAMAssist()" id="btnSAM"
    style="padding:10px 18px;border:2px solid #a855f7;border-radius:8px;font-size:13px;font-weight:600;
    background:#a855f7;color:#fff;cursor:pointer">
    🧠 SAM检测<span class="key">Key S</span></button>
  <span style="color:#888;font-size:11px;padding:8px;align-self:center">Shift+点击 加提示点 → SAM</span>
</div>

<div id="tooltip"></div>
<div id="toast"></div>

<!-- Help Overlay -->
<div id="help-overlay">
  <div id="help-panel">
    <h2>🖱️ 使用说明</h2>
    <p style="margin-bottom:10px;color:#ff9800">🔹 <b>关键点模式</b>（推荐先做）</p>
    <div class="step"><span class="num">1</span><span><b>旋转/缩放</b>：鼠标左键拖拽旋转，滚轮缩放，右键平移</span></div>
    <div class="step"><span class="num">2</span><span><b>Shift+点击</b>点云添加角点（自动吸附最近点）</span></div>
    <div class="step"><span class="num">3</span><span>按 <span class="key-tag">1</span> 外角点(📐) / <span class="key-tag">2</span> 内角点(📏) 切换类型</span></div>
    <div class="step"><span class="num">4</span><span><b>点击已有角点</b>可选中，按 <span class="key-tag">Del</span> 删除</span></div>
    <p style="margin-top:10px;margin-bottom:10px;color:#3b82f6">🔹 <b>Proposal 模式</b></p>
    <div class="step"><span class="num">1</span><span>点击点云选中 proposal，按 <span class="key-tag">1-4</span> 分配类别</span></div>
    <p style="margin-top:10px"><b>通用</b>：<span class="key-tag">N</span>/<span class="key-tag">P</span> 翻帧 | 顶栏切换模式</p>
    <button onclick="document.getElementById('help-overlay').style.display='none'">知道了，开始标注</button>
  </div>
</div>
<button id="help-btn" onclick="document.getElementById('help-overlay').style.display='flex'" title="使用帮助">?</button>

<!-- Three.js Import Map -->
<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.157.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.157.0/examples/jsm/"
  }
}
</script>

<script type="module">
import * as THREE from 'three';
import { TrackballControls } from 'three/addons/controls/TrackballControls.js';

// ══════════════════════════════════════════
// Globals
// ══════════════════════════════════════════
let scene, camera, renderer, controls;
let baseCloud = null;           // 基础点云 (BufferGeometry Points)
let overlayGroup = null;        // 标注覆盖层组
let stems = [];
let currentIdx = -1;
let currentData = null;         // {stem, xyz_flat, rgb_flat, n_points, proposals, ...}
let selectedPid = null;         // 当前选中的 proposal ID
let proposalsMap = {};          // pid → proposal data
let labelMap = {};              // pid → class_label
let classColors = {};

const SELECT_HIGHLIGHT = [1.0, 0.65, 0.0];  // orange

// ══════════════════════════════════════════
// Three.js Setup
// ══════════════════════════════════════════
function initScene() {
  const canvas = document.getElementById('canvas');
  const w = window.innerWidth, h = window.innerHeight;

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setSize(w, h);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a2e);

  camera = new THREE.PerspectiveCamera(55, w / h, 10, 50000);
  camera.position.set(1500, -2000, 3000);
  camera.up.set(0, -1, 0);  // Y up in PCD coords → neg Y for camera up

  controls = new TrackballControls(camera, renderer.domElement);
  controls.rotateSpeed = 2.5;
  controls.zoomSpeed = 1.5;
  controls.panSpeed = 1.0;
  controls.staticMoving = false;

  // Lights (for any mesh, though we mostly use Points)
  scene.add(new THREE.AmbientLight(0x404040, 2));
  const dir = new THREE.DirectionalLight(0xffffff, 1);
  dir.position.set(1, -1, 1);
  scene.add(dir);

  overlayGroup = new THREE.Group();
  scene.add(overlayGroup);

  // Click handler
  canvas.addEventListener('click', onCanvasClick);
  canvas.addEventListener('mousemove', onCanvasMove);
  window.addEventListener('resize', onResize);
  window.addEventListener('keydown', onKeyDown);

  // Axes helper
  scene.add(new THREE.AxesHelper(1000));

  animate();
}

function onResize() {
  const w = window.innerWidth, h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  if (mode === 'keypoints') updateKeypointRings();
  renderer.render(scene, camera);
}

// ══════════════════════════════════════════
// Point Cloud Rendering
// ══════════════════════════════════════════
function buildPointCloud(xyzFlat, rgbFlat, N) {
  // Remove old base cloud
  if (baseCloud) {
    scene.remove(baseCloud);
    baseCloud.geometry?.dispose();
    baseCloud.material?.dispose();
  }

  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(xyzFlat);   // [x0,y0,z0, x1,y0,z0, ...]
  const col = new Uint8Array(rgbFlat);      // [r0,g0,b0, r1,g0,b0, ...]

  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(
    new Float32Array(N * 3), 3));  // we'll set colors separately

  // Convert uint8 rgb to float colors
  const colorsFloat = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    colorsFloat[i * 3]     = col[i * 3] / 255;
    colorsFloat[i * 3 + 1] = col[i * 3 + 1] / 255;
    colorsFloat[i * 3 + 2] = col[i * 3 + 2] / 255;
  }
  geo.setAttribute('color', new THREE.BufferAttribute(colorsFloat, 3));

  const mat = new THREE.PointsMaterial({
    size: 6.0,
    vertexColors: true,
    sizeAttenuation: true,
    depthTest: true,
    blending: THREE.NormalBlending,
  });

  baseCloud = new THREE.Points(geo, mat);
  scene.add(baseCloud);
}

function clearOverlays() {
  while (overlayGroup.children.length > 0) {
    const child = overlayGroup.children[0];
    child.geometry?.dispose();
    child.material?.dispose();
    overlayGroup.remove(child);
  }
}

function rgbForClass(label) {
  switch (label) {
    case 'beam':  return [1.0, 0.27, 0.27];
    case 'pillar': return [0.27, 0.53, 1.0];
    case 'pallet': return [0.27, 0.80, 0.27];
    case 'goods':  return [1.0, 0.80, 0.0];
    default:       return [0.55, 0.55, 0.55];
  }
}

function buildProposalOverlay(indices, xyzFlat, colorRGB) {
  // indices: list of point indices into the full cloud
  // xyzFlat: flat [x0,y0,z0, ...] array
  const n = indices.length;
  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);

  for (let i = 0; i < n; i++) {
    const pi = indices[i];
    pos[i * 3]     = xyzFlat[pi * 3];
    pos[i * 3 + 1] = xyzFlat[pi * 3 + 1];
    pos[i * 3 + 2] = xyzFlat[pi * 3 + 2];
    col[i * 3]     = colorRGB[0];
    col[i * 3 + 1] = colorRGB[1];
    col[i * 3 + 2] = colorRGB[2];
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));

  const mat = new THREE.PointsMaterial({
    size: 10.0,
    vertexColors: true,
    sizeAttenuation: true,
    depthTest: true,
    depthWrite: false,
    blending: THREE.NormalBlending,
    transparent: true,
    opacity: 0.75,
  });

  const points = new THREE.Points(geo, mat);
  overlayGroup.add(points);
  return points;
}

function refreshOverlays() {
  clearOverlays();
  if (!currentData) return;

  const xyzFlat = currentData.xyz;

  // Draw labeled proposals in class colors
  for (const [pidStr, label] of Object.entries(labelMap)) {
    const pid = parseInt(pidStr);
    if (pid === selectedPid) continue;  // selected drawn last (on top)
    const prop = proposalsMap[pid];
    if (!prop) continue;
    const color = rgbForClass(label);
    prop._overlay = buildProposalOverlay(prop.indices, xyzFlat, color);
  }

  // Draw selected proposal in highlight color
  if (selectedPid !== null && proposalsMap[selectedPid]) {
    const prop = proposalsMap[selectedPid];
    const color = labelMap[selectedPid]
      ? rgbForClass(labelMap[selectedPid]).map(c => Math.min(1, c + 0.3))
      : SELECT_HIGHLIGHT;
    prop._overlay = buildProposalOverlay(prop.indices, xyzFlat, color);
  }
}

// ══════════════════════════════════════════
// Data Loading
// ══════════════════════════════════════════
async function loadStems() {
  const resp = await fetch('/api/stems');
  stems = await resp.json();
  document.getElementById('statsLabel').textContent =
    `${stems.length} frames`;
}

async function loadFrame(idx) {
  if (idx < 0 || idx >= stems.length) return;
  currentIdx = idx;
  selectedPid = null;
  proposalsMap = {};
  labelMap = {};

  const stem = stems[idx].stem;
  document.getElementById('stemLabel').textContent = `⏳ ${stem}`;
  document.getElementById('gotoIdx').value = idx;

  const resp = await fetch(`/api/load/${stem}`);
  if (!resp.ok) {
    document.getElementById('statusLabel').textContent = '❌ 加载失败';
    return;
  }

  currentData = await resp.json();
  classColors = currentData.class_colors || {};

  // Build proposals map
  for (const prop of currentData.proposals) {
    proposalsMap[prop.id] = prop;
    if (prop.label) {
      labelMap[prop.id] = prop.label;
    }
  }

  // Build point cloud
  buildPointCloud(currentData.xyz, currentData.rgb, currentData.n_points);

  // Build overlays
  refreshOverlays();

  // Update UI
  document.getElementById('stemLabel').textContent = stem;
  const nLabeled = Object.keys(labelMap).length;
  document.getElementById('statusLabel').textContent =
    `${currentData.n_points.toLocaleString()} pts · ${currentData.proposals.length} proposals · ${nLabeled} labeled`;

  // Auto-fit camera to point cloud
  autoFitCamera();

  // Update sidebar
  renderProposalList();

  // Load keypoints (always, for mode switch)
  loadKeypoints();
  if (mode === 'keypoints') {
    document.getElementById('statusLabel').textContent =
      `${currentData.n_points.toLocaleString()} pts · ${keypoints.length} 角点`;
  }
}

function autoFitCamera() {
  if (!currentData) return;
  const xyz = currentData.xyz;
  let xMin = Infinity, xMax = -Infinity;
  let yMin = Infinity, yMax = -Infinity;
  let zMin = Infinity, zMax = -Infinity;

  for (let i = 0; i < xyz.length; i += 3) {
    const x = xyz[i], y = xyz[i + 1], z = xyz[i + 2];
    if (x < xMin) xMin = x; if (x > xMax) xMax = x;
    if (y < yMin) yMin = y; if (y > yMax) yMax = y;
    if (z < zMin) zMin = z; if (z > zMax) zMax = z;
  }

  const cx = (xMin + xMax) / 2, cy = (yMin + yMax) / 2, cz = (zMin + zMax) / 2;
  const dx = xMax - xMin, dy = yMax - yMin, dz = zMax - zMin;
  const dist = Math.max(dx, dy, dz) * 1.5;

  camera.position.set(cx + dist * 0.4, cy - dist * 0.7, cz + dist * 0.5);
  controls.target.set(cx, cy, cz);
  controls.update();
}

// ══════════════════════════════════════════
// Proposal Selection & Labeling
// ══════════════════════════════════════════
function selectProposal(pid) {
  if (pid === selectedPid) {
    // Deselect
    selectedPid = null;
  } else {
    selectedPid = pid;
  }
  refreshOverlays();
  renderProposalList();
}

function assignClass(classLabel) {
  if (selectedPid === null) {
    showToast('⚠️ 请先点击选中一个 proposal');
    return;
  }

  const stem = stems[currentIdx].stem;

  // Update locally
  if (classLabel === 'background') {
    delete labelMap[selectedPid];
  } else {
    labelMap[selectedPid] = classLabel;
  }

  // Update proposal data
  if (proposalsMap[selectedPid]) {
    proposalsMap[selectedPid].label = classLabel === 'background' ? null : classLabel;
  }

  // Save to server
  fetch('/api/label', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      stem: stem,
      proposal_id: selectedPid,
      class_label: classLabel,
    }),
  }).catch(err => console.error('Save failed:', err));

  refreshOverlays();
  renderProposalList();

  const nLabeled = Object.keys(labelMap).length;
  document.getElementById('statusLabel').textContent =
    `${currentData.n_points.toLocaleString()} pts · ${currentData.proposals.length} proposals · ${nLabeled} labeled`;

  const className = classLabel === 'background' ? '丢弃' : classLabel;
  showToast(`✅ Proposal #${selectedPid} → ${className}`);
}

// ══════════════════════════════════════════
// Sidebar Proposal List
// ══════════════════════════════════════════
function renderProposalList() {
  const container = document.getElementById('propList');
  if (!currentData) {
    container.innerHTML = '<p style="color:#888">—</p>';
    return;
  }

  // Sort: labeled first, then by size
  const props = [...currentData.proposals];
  props.sort((a, b) => {
    const la = labelMap[a.id] ? 1 : 0;
    const lb = labelMap[b.id] ? 1 : 0;
    if (la !== lb) return lb - la;
    return a.n_points - b.n_points;
  });

  container.innerHTML = props.map(p => {
    const label = labelMap[p.id];
    const lblHtml = label
      ? `<span class="lbl ${label}">${label}</span>`
      : '';
    const selClass = p.id === selectedPid ? ' selected' : '';
    const ptsStr = p.n_points > 1000
      ? `${(p.n_points / 1000).toFixed(1)}k`
      : p.n_points;
    return `<div class="prop${selClass}" onclick="selectProposal(${p.id})">
      <span class="id">#${p.id}</span>
      <span class="pts">${ptsStr} pts</span>
      ${lblHtml}
    </div>`;
  }).join('');
}

// ══════════════════════════════════════════
// Mouse Interaction
// ══════════════════════════════════════════
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 12;

function getClickedProposal(event) {
  if (!baseCloud) return null;

  const rect = renderer.domElement.getBoundingClientRect();
  const mouse = new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1,
  );

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObject(baseCloud);

  if (intersects.length === 0) return null;

  // Get point index from intersection
  const idx = intersects[0].index;
  if (idx === undefined || !currentData) return null;

  // Find which proposal contains this point
  // Check smaller proposals first (they're more specific)
  const sortedPids = Object.keys(proposalsMap)
    .map(Number)
    .sort((a, b) => proposalsMap[a].n_points - proposalsMap[b].n_points);

  for (const pid of sortedPids) {
    const prop = proposalsMap[pid];
    // Binary search would be better, but linear is fine for 24 proposals
    if (prop.indices.includes(idx)) {
      return pid;
    }
  }
  return null;  // point not in any proposal
}

function onCanvasClick(event) {
  if (mode === 'keypoints') {
    addKeypointFromClick(event);
    return;
  }
  const pid = getClickedProposal(event);
  selectProposal(pid);
}

function onCanvasMove(event) {
  if (mode === 'keypoints') {
    document.body.style.cursor = event.shiftKey ? 'crosshair' : 'default';
    return;
  }
  const pid = getClickedProposal(event);
  const tooltip = document.getElementById('tooltip');

  if (pid !== null) {
    const prop = proposalsMap[pid];
    const ptsStr = prop.n_points > 1000
      ? `${(prop.n_points / 1000).toFixed(1)}k`
      : prop.n_points;
    const label = labelMap[pid] || '未标注';
    tooltip.style.display = 'block';
    tooltip.style.left = (event.clientX + 15) + 'px';
    tooltip.style.top = (event.clientY - 15) + 'px';
    tooltip.textContent = `Proposal #${pid} · ${ptsStr} pts · ${label}`;
    document.body.style.cursor = 'pointer';
  } else {
    tooltip.style.display = 'none';
    document.body.style.cursor = 'default';
  }
}

// ══════════════════════════════════════════
// Keyboard Shortcuts
// ══════════════════════════════════════════
function onKeyDown(event) {
  if (event.target.tagName === 'INPUT') return;

  if (mode === 'keypoints') {
    switch (event.key.toLowerCase()) {
      case '1': setKeypointType('outer'); break;
      case '2': setKeypointType('inner'); break;
      case 'delete':
      case 'backspace':
        event.preventDefault();
        deleteSelectedKeypoint();
        break;
      case 's':
        event.preventDefault();
        runSAMAssist();
        break;
      case 'escape':
        selectedKpId = null;
        refreshKeypointDisplay();
        break;
      case 'n':
      case 'arrowright':
        event.preventDefault();
        saveKeypoints(); navigateTo(currentIdx + 1); break;
      case 'p':
      case 'arrowleft':
        event.preventDefault();
        saveKeypoints(); navigateTo(currentIdx - 1); break;
      case 'g':
        event.preventDefault();
        document.getElementById('gotoIdx').focus();
        document.getElementById('gotoIdx').select();
        break;
    }
    return;
  }

  // Proposal mode
  switch (event.key.toLowerCase()) {
    case '1': assignClass('beam'); break;
    case '2': assignClass('pillar'); break;
    case '3': assignClass('pallet'); break;
    case '4': assignClass('goods'); break;
    case '0': assignClass('background'); break;
    case 'n':
    case 'arrowright':
      event.preventDefault();
      navigateTo(currentIdx + 1);
      break;
    case 'p':
    case 'arrowleft':
      event.preventDefault();
      navigateTo(currentIdx - 1);
      break;
    case ' ':
      event.preventDefault();
      selectProposal(null);
      break;
    case 'g':
      event.preventDefault();
      document.getElementById('gotoIdx').focus();
      document.getElementById('gotoIdx').select();
      break;
  }
}

// ══════════════════════════════════════════
// Navigation
// ══════════════════════════════════════════
function navigateTo(idx) {
  if (idx < 0) idx = 0;
  if (idx >= stems.length) idx = stems.length - 1;
  if (mode === 'keypoints') saveKeypoints();
  loadFrame(idx);
}

// ══════════════════════════════════════════
// Toast
// ══════════════════════════════════════════
function showToast(msg) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.style.display = 'block';
  clearTimeout(el._timeout);
  el._timeout = setTimeout(() => { el.style.display = 'none'; }, 1500);
}

// ══════════════════════════════════════════
// Keypoint Mode
// ══════════════════════════════════════════
let mode = 'proposals';  // 'proposals' | 'keypoints'
let keypoints = [];       // [{id, type:'inner'|'outer', position:[x,y,z]}]
let selectedKpId = null;
let kpType = 'outer';     // default type for new keypoints
let kpIdCounter = 0;
let kpSphereGroup = null;  // THREE.Group for keypoint spheres

const KP_COLORS = {
  outer: 0xff9800,  // orange
  inner: 0x00bcd4,  // cyan
  selected: 0xffff00, // yellow highlight
};

function toggleMode() {
  const btn = document.getElementById('btnMode');
  if (mode === 'proposals') {
    mode = 'keypoints';
    btn.textContent = '📍 角点';
    btn.style.background = '#ff9800';
    document.getElementById('classbar').style.display = 'none';
    document.getElementById('kpclassbar').style.display = 'flex';
    document.getElementById('sidebarProposals').style.display = 'none';
    document.getElementById('sidebarKeypoints').style.display = 'block';
    selectedPid = null;
    clearOverlays();
    loadKeypoints();
  } else {
    mode = 'proposals';
    btn.textContent = '📋 Proposals';
    btn.style.background = '#3b82f6';
    document.getElementById('classbar').style.display = 'flex';
    document.getElementById('kpclassbar').style.display = 'none';
    document.getElementById('sidebarProposals').style.display = 'block';
    document.getElementById('sidebarKeypoints').style.display = 'none';
    selectedKpId = null;
    refreshKeypointDisplay();
    refreshOverlays();
    renderProposalList();
  }
}
window.toggleMode = toggleMode;

function setKeypointType(type) {
  kpType = type;
  document.getElementById('btnOuter').style.background = type === 'outer' ? '#ff9800' : 'rgba(255,255,255,0.08)';
  document.getElementById('btnOuter').style.color = type === 'outer' ? '#000' : '#fff';
  document.getElementById('btnInner').style.background = type === 'inner' ? '#00bcd4' : 'rgba(255,255,255,0.08)';
  document.getElementById('btnInner').style.color = type === 'inner' ? '#000' : '#fff';
}
window.setKeypointType = setKeypointType;

function getClickedPoint3D(event) {
  if (!baseCloud) return null;

  const rect = renderer.domElement.getBoundingClientRect();
  const mouse = new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1,
  );

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObject(baseCloud);
  if (intersects.length === 0) return null;

  const idx = intersects[0].index;
  if (idx === undefined || !currentData) return null;

  return {
    index: idx,
    position: [
      currentData.xyz[idx * 3],
      currentData.xyz[idx * 3 + 1],
      currentData.xyz[idx * 3 + 2]
    ],
    point: intersects[0].point.clone(),
  };
}

function addKeypointFromClick(event) {
  if (!event.shiftKey) {
    // Without Shift: try to select an existing keypoint
    const hit = getClickedPoint3D(event);
    if (!hit) { selectedKpId = null; refreshKeypointDisplay(); return; }

    // Find nearest keypoint within threshold
    let nearestKp = null, nearestDist = 100; // 100mm threshold
    for (const kp of keypoints) {
      const dx = kp.position[0] - hit.position[0];
      const dy = kp.position[1] - hit.position[1];
      const dz = kp.position[2] - hit.position[2];
      const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
      if (dist < nearestDist) { nearestDist = dist; nearestKp = kp; }
    }
    selectedKpId = nearestKp ? nearestKp.id : null;
    refreshKeypointDisplay();
    renderKeypointList();
    return;
  }

  // Shift+click: add keypoint at nearest point
  const hit = getClickedPoint3D(event);
  if (!hit) return;

  const kp = {
    id: kpIdCounter++,
    type: kpType,
    position: hit.position,
    pointIndex: hit.index,
  };
  keypoints.push(kp);
  selectedKpId = kp.id;
  refreshKeypointDisplay();
  renderKeypointList();
  saveKeypoints();
  showToast(`✅ 添加${kpType === 'outer' ? '外' : '内'}角点 #${kp.id}`);
}

function refreshKeypointDisplay() {
  // Clear old spheres
  if (kpSphereGroup) {
    overlayGroup.remove(kpSphereGroup);
    kpSphereGroup.traverse(c => { if (c.geometry) c.geometry.dispose(); if (c.material) c.material.dispose(); });
  }
  kpSphereGroup = new THREE.Group();

  for (const kp of keypoints) {
    const color = kp.id === selectedKpId ? KP_COLORS.selected : KP_COLORS[kp.type] || KP_COLORS.outer;
    const mat = new THREE.MeshBasicMaterial({ color, depthTest: false });
    const geo = new THREE.SphereGeometry(12, 16, 16);
    const sphere = new THREE.Mesh(geo, mat);
    sphere.position.set(kp.position[0], kp.position[1], kp.position[2]);
    sphere.userData = { kpId: kp.id };
    kpSphereGroup.add(sphere);

    // Small label ring
    const ringGeo = new THREE.RingGeometry(16, 20, 32);
    const ringMat = new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide, depthTest: false, transparent: true, opacity: 0.5 });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.position.copy(sphere.position);
    ring.lookAt(camera.position);
    kpSphereGroup.add(ring);
  }

  overlayGroup.add(kpSphereGroup);
  updateKeypointRings();
}

function updateKeypointRings() {
  // Make rings face camera
  if (!kpSphereGroup) return;
  kpSphereGroup.children.forEach(c => {
    if (c.geometry && c.geometry.type === 'RingGeometry') {
      c.lookAt(camera.position);
    }
  });
}

function renderKeypointList() {
  const container = document.getElementById('kpList');
  document.getElementById('kpCount').textContent = keypoints.length;

  if (keypoints.length === 0) {
    container.innerHTML = '<p style="color:#666;font-size:11px">Shift+点击点云添加角点</p>';
    return;
  }

  container.innerHTML = keypoints.map(kp => {
    const typeLabel = kp.type === 'outer' ? '📐外' : '📏内';
    const pos = kp.position.map(v => (v / 1000).toFixed(2)).join(', ');
    const selClass = kp.id === selectedKpId ? ' style="border:1px solid #ffaa00;background:rgba(255,170,0,0.15)"' : '';
    return `<div class="prop"${selClass} onclick="selectKeypointById(${kp.id})">
      <span class="id">${typeLabel}#${kp.id}</span>
      <span class="pts" style="display:block;font-size:10px">(${pos})m</span>
      <span class="lbl" style="cursor:pointer;color:#f87171" onclick="event.stopPropagation();deleteKeypoint(${kp.id})">✕</span>
    </div>`;
  }).join('');
}

function selectKeypointById(id) {
  selectedKpId = id;
  refreshKeypointDisplay();
  renderKeypointList();
}
window.selectKeypointById = selectKeypointById;

function deleteKeypoint(id) {
  keypoints = keypoints.filter(k => k.id !== id);
  if (selectedKpId === id) selectedKpId = null;
  refreshKeypointDisplay();
  renderKeypointList();
  saveKeypoints();
  showToast('🗑 角点已删除');
}
window.deleteKeypoint = deleteKeypoint;

function deleteSelectedKeypoint() {
  if (selectedKpId !== null) deleteKeypoint(selectedKpId);
}
window.deleteSelectedKeypoint = deleteSelectedKeypoint;

function clearAllKeypoints() {
  if (!confirm('确定清除本帧所有角点？')) return;
  keypoints = [];
  selectedKpId = null;
  kpIdCounter = 0;
  refreshKeypointDisplay();
  renderKeypointList();
  saveKeypoints();
}
window.clearAllKeypoints = clearAllKeypoints;

async function runSAMAssist() {
  // 用当前标记的关键点作为 SAM 提示 → 分割货架 → 检测角点
  if (!currentData) return;
  const stem = stems[currentIdx].stem;

  // 收集提示点 (当前已标记的 keypoints 作为正提示)
  const prompts = keypoints.map(k => k.position);
  if (prompts.length === 0) {
    showToast('⚠️ 请先 Shift+点击货架结构上的几个点作为 SAM 提示');
    return;
  }

  document.getElementById('btnSAM').textContent = '⏳ SAM推理中...';
  document.getElementById('btnSAM').disabled = true;
  showToast(`🧠 SAM 处理 ${prompts.length} 个提示点...`);

  try {
    const resp = await fetch('/api/sam_corners', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stem, prompts }),
    });
    const data = await resp.json();

    if (data.error) {
      showToast('❌ ' + data.error);
      return;
    }

    // 用 SAM 结果替换当前关键点
    keypoints = data.keypoints || [];
    kpIdCounter = keypoints.length > 0 ? Math.max(...keypoints.map(k => k.id)) + 1 : 0;
    selectedKpId = null;
    refreshKeypointDisplay();
    renderKeypointList();
    saveKeypoints();

    const nOuter = data.keypoints.filter(k => k.type === 'outer').length;
    const nInner = data.keypoints.filter(k => k.type === 'inner').length;
    document.getElementById('statusLabel').textContent =
      `${currentData.n_points.toLocaleString()} pts · SAM: ${nOuter}外 + ${nInner}内`;
    showToast(`✅ SAM检测完成: ${nOuter} 外角点 + ${nInner} 内角点`);
  } catch (err) {
    console.error('SAM assist failed:', err);
    showToast('❌ SAM 推理失败');
  } finally {
    document.getElementById('btnSAM').textContent = '🧠 SAM检测';
    document.getElementById('btnSAM').disabled = false;
  }
}
window.runSAMAssist = runSAMAssist;

async function loadKeypoints() {
  if (!currentData) return;
  const stem = stems[currentIdx].stem;
  try {
    const resp = await fetch(`/api/keypoints/${stem}`);
    const data = await resp.json();
    keypoints = data.keypoints || [];
    kpIdCounter = keypoints.length > 0 ? Math.max(...keypoints.map(k => k.id)) + 1 : 0;
    selectedKpId = null;
    refreshKeypointDisplay();
    renderKeypointList();
    const n = keypoints.length;
    document.getElementById('statusLabel').textContent =
      `${currentData.n_points.toLocaleString()} pts · ${n} 角点`;
  } catch (err) {
    console.error('Load keypoints failed:', err);
  }
}

async function saveKeypoints() {
  if (!currentData) return;
  const stem = stems[currentIdx].stem;
  try {
    await fetch(`/api/keypoints/${stem}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ keypoints }),
    });
  } catch (err) {
    console.error('Save keypoints failed:', err);
  }
}

// ══════════════════════════════════════════
// Init
// ══════════════════════════════════════════
async function main() {
  initScene();
  await loadStems();

  // Bind nav buttons
  document.getElementById('btnPrev').addEventListener('click', () => {
    if (mode === 'keypoints') saveKeypoints();
    navigateTo(currentIdx - 1);
  });
  document.getElementById('btnNext').addEventListener('click', () => {
    if (mode === 'keypoints') saveKeypoints();
    navigateTo(currentIdx + 1);
  });
  document.getElementById('btnGoto').addEventListener('click', () => {
    const idx = parseInt(document.getElementById('gotoIdx').value);
    if (!isNaN(idx)) navigateTo(idx);
  });
  document.getElementById('gotoIdx').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      const idx = parseInt(e.target.value);
      if (!isNaN(idx)) navigateTo(idx);
    }
  });

  // Expose to global scope for onclick handlers
  window.selectProposal = selectProposal;
  window.assignClass = assignClass;
  window.deleteKeypoint = deleteKeypoint;
  window.selectKeypointById = selectKeypointById;
  window.deleteSelectedKeypoint = deleteSelectedKeypoint;
  window.clearAllKeypoints = clearAllKeypoints;
  window.setKeypointType = setKeypointType;
  window.runSAMAssist = runSAMAssist;

  // Load first substantial frame (skip tiny ones)
  let startIdx = 0;
  for (let i = 0; i < stems.length; i++) {
    if (stems[i].n_points > 5000) { startIdx = i; break; }
  }
  if (stems.length > 0) {
    loadFrame(startIdx);
  } else {
    document.getElementById('statusLabel').textContent = '❌ 无标注文件';
  }
}

main();
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    global ANNOTATION_DIR, PCD_DIR, STEMS, _SAM_AVAILABLE

    parser = argparse.ArgumentParser(description="Point-SAM Web Review")
    parser.add_argument("--annotation_dir", type=str,
                        default="output/shelf_annotations",
                        help="标注文件目录")
    parser.add_argument("--pcd_dir", type=str,
                        default="data/new_sheef/pngs",
                        help="PCD 文件目录 (SAM 推理时需要)")
    parser.add_argument("--sam_ckpt", type=str, default=None,
                        help="Point-SAM 权重路径 (启用 SAM 辅助角点检测)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Web 服务端口 (默认 5000)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="监听地址 (默认 127.0.0.1)")
    args = parser.parse_args()

    ANNOTATION_DIR = args.annotation_dir
    PCD_DIR = args.pcd_dir
    if not os.path.isdir(ANNOTATION_DIR):
        print(f"Error: annotation_dir not found: {ANNOTATION_DIR}")
        sys.exit(1)

    # 加载 Point-SAM (如有)
    if args.sam_ckpt:
        _init_sam(args.sam_ckpt)
        print(f"[SAM] Ready for assisted corner detection")

    # 收集所有 stems
    npz_files = sorted(Path(ANNOTATION_DIR).glob("*_proposals.npz"))
    STEMS = [p.stem.replace("_proposals", "") for p in npz_files]

    if not STEMS:
        print(f"Error: no *_proposals.npz found in {ANNOTATION_DIR}")
        print("Run pointsam_annotator.py first to generate proposals.")
        sys.exit(1)

    print(f"Found {len(STEMS)} frames with proposals")
    print(f"\n  Open: http://{args.host}:{args.port}")
    print(f"  Press Ctrl+C to stop\n")

    # Register routes
    app.add_url_rule("/", "index", lambda: HTML)

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
