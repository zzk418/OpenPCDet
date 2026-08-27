#!/usr/bin/env python3
"""
PCD-2D 对齐验证工具
==================
验证 2D RGB 图像和 PCD 点云是否通过当前内参正确对齐。

方法:
  1. 读取 PCD → 得到 (x,y,z) + RGB
  2. 用 pinhole 模型投影到 2D
  3. 将 PCD RGB 渲染到图像平面，与真实 JPG 叠图对比
  4. 结构对齐 → 内参正确；偏移 → 需要标定

用法:
  python tools/verify_pcd_alignment.py                          # 默认帧
  python tools/verify_pcd_alignment.py --stem TV_250000036488  # 指定帧
  python tools/verify_pcd_alignment.py --grid_search           # 搜索最优内参
"""

import argparse, os, sys, time
import cv2
import numpy as np
from pathlib import Path

# ── 复用 PCD 读取 ──

def read_pcd_binary(path):
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode().strip()
            if line == "DATA binary":
                break
        dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("rgb", "u4")])
        data = np.frombuffer(f.read(), dtype=dtype)
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    rgb_raw = data["rgb"]
    r = ((rgb_raw >> 16) & 0xFF).astype(np.float32)
    g = ((rgb_raw >> 8) & 0xFF).astype(np.float32)
    b = (rgb_raw & 0xFF).astype(np.float32)
    rgb = np.column_stack([r, g, b]).astype(np.float32)  # 0-255
    return xyz, rgb


def project_pcd_to_image(xyz, img_w, img_h, fx, fy, cx, cy):
    """Pinhole 投影: PCD (x,y,z) → 2D (u,v) + depth。
    返回 valid_mask, u, v, z (仅图像范围内的点)。
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    valid = z > 1.0
    xv, yv, zv = x[valid], y[valid], z[valid]
    u = np.round(fx * xv / zv + cx).astype(np.int32)
    v = np.round(fy * yv / zv + cy).astype(np.int32)
    in_bounds = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    return in_bounds, u, v, zv, np.where(valid)[0]


def render_pcd_rgb(rgb_pcd, u, v, in_bounds, img_w, img_h):
    """把 PCD 点云颜色渲染到 2D 图像平面上。
    每个像素取最近的点（最小深度）。
    """
    img = np.zeros((img_h, img_w, 3), dtype=np.float32)
    depth_buf = np.full((img_h, img_w), np.inf, dtype=np.float32)

    for i in range(len(u)):
        if not in_bounds[i]:
            continue
        ui, vi = u[i], v[i]
        # 原始索引
        idx = i  # u,v are already filtered by in_bounds
        # 但我们迭代所有点... 这样太慢。让我优化。

    # 向量化版本
    ub = u[in_bounds]
    vb = v[in_bounds]
    rgb_sub = rgb_pcd[in_bounds]

    for i in range(len(ub)):
        ui, vi = ub[i], vb[i]
        img[vi, ui] = rgb_sub[i]

    return (img / 255.0).astype(np.float32)


def render_pcd_rgb_vectorized(rgb_pcd, u, v, in_bounds, img_w, img_h):
    """向量化渲染（不处理遮挡，够用）。"""
    ub = u[in_bounds]
    vb = v[in_bounds]
    rgb_sub = rgb_pcd[in_bounds]

    img = np.zeros((img_h, img_w, 3), dtype=np.float32)
    img[vb, ub] = rgb_sub
    return img / 255.0


def compute_alignment_score(rgb_img, pcd_rgb_render, depth_map=None):
    """用边缘一致性评估对齐质量（不依赖颜色匹配）。

    思路: PCD 深度边缘 和 RGB 图像边缘 应该在相同位置。
    """
    h, w = rgb_img.shape[:2]
    gray_rgb = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

    # RGB 边缘
    edges_rgb = cv2.Canny(gray_rgb, 60, 180)

    # PCD 深度边缘
    if depth_map is not None:
        dm_valid = np.nan_to_num(depth_map, nan=0)
        dm_u8 = ((dm_valid - dm_valid.min()) / max(dm_valid.max() - dm_valid.min(), 1) * 255).astype(np.uint8)
        edges_pcd = cv2.Canny(dm_u8, 40, 120)
    else:
        gray_pcd = (pcd_rgb_render.mean(axis=2) * 255).astype(np.uint8)
        edges_pcd = cv2.Canny(gray_pcd, 40, 120)

    # Dilate edges for tolerance (±3 px)
    kernel = np.ones((5, 5), np.uint8)
    edges_rgb_d = cv2.dilate(edges_rgb, kernel)
    edges_pcd_d = cv2.dilate(edges_pcd, kernel)

    rgb_total = edges_rgb.sum() / 255
    pcd_total = edges_pcd.sum() / 255
    if rgb_total < 10 or pcd_total < 10:
        mask = (pcd_rgb_render.sum(axis=2) > 0.01).astype(np.uint8)
        return 0.0, mask

    # PCD edge 落在 RGB edge 附近的百分比
    overlap = (edges_pcd & edges_rgb_d).sum() / 255
    pcd_recall = overlap / max(pcd_total, 1)

    # RGB edge 落在 PCD edge 附近的百分比
    overlap2 = (edges_rgb & edges_pcd_d).sum() / 255
    rgb_recall = overlap2 / max(rgb_total, 1)

    # F1-like score
    score = 2 * pcd_recall * rgb_recall / max(pcd_recall + rgb_recall, 0.001)

    mask = (pcd_rgb_render.sum(axis=2) > 0.01).astype(np.uint8)
    return score, mask


# ── Visualization ──

def create_comparison_viz(rgb_img, pcd_render, mask, stem, fx, fy, cx, cy, output_dir):
    """创建对齐验证可视化: 真实图 | PCD投影 | 叠图 | 边缘对比"""
    h, w = rgb_img.shape[:2]
    rgb_f = rgb_img.astype(np.float32) / 255.0

    # 1. PCD 投影渲染图
    pcd_viz = (pcd_render * 255).astype(np.uint8)

    # 2. 叠图: 50/50 blend
    overlay = (rgb_f * 0.5 + pcd_render * 0.5)
    overlay = (overlay * 255).astype(np.uint8)

    # 3. 边缘对比: Canny on both, draw in different colors
    gray_rgb = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    gray_pcd = cv2.cvtColor(pcd_viz, cv2.COLOR_BGR2GRAY)

    edges_rgb = cv2.Canny(gray_rgb, 80, 200)
    edges_pcd = cv2.Canny(gray_pcd, 80, 200)

    edge_viz = np.zeros((h, w, 3), dtype=np.uint8)
    edge_viz[edges_rgb > 0] = [0, 255, 0]    # 真实图边缘 = 绿色
    edge_viz[edges_pcd > 0] = [255, 0, 255]   # PCD 边缘 = 品红
    # 重叠处 = 白色

    # 4. 差值热力图
    diff = np.abs(rgb_f - pcd_render).mean(axis=2)
    diff[mask == 0] = 0
    diff_viz = (diff * 255).astype(np.uint8)
    diff_viz = cv2.applyColorMap(diff_viz, cv2.COLORMAP_HOT)

    # 拼成大图
    row1 = np.hstack([rgb_img, pcd_viz])
    row2 = np.hstack([overlay, edge_viz])

    # 补标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    for row_img, label in [(rgb_img, "Real RGB"), (pcd_viz, "PCD Projected")]:
        cv2.putText(row_img, label, (10, 25), font, 0.6, (0, 255, 0), 2)

    big = np.vstack([row1, row2])

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{stem}_alignment_check.png")
    cv2.imwrite(out_path, big)

    # 同时存单独的叠图
    overlay_path = os.path.join(output_dir, f"{stem}_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    print(f"Saved: {out_path}")
    print(f"Saved: {overlay_path}")
    return out_path


# ── 网格搜索内参 ──

def grid_search_intrinsics(xyz, rgb_pcd, rgb_img, img_w, img_h):
    """在 fx/fy/cx/cy 上网格搜索，找到使 PCD 投影和真实图最匹配的内参。"""
    print(f"\n{'='*60}")
    print("Grid Search for Camera Intrinsics")
    print(f"{'='*60}")

    # 搜索范围
    base_f = (img_w + img_h) / 4  # ~280 for 640x480
    f_guesses = [base_f * r for r in [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]]
    cx_guesses = [img_w * r for r in [0.45, 0.48, 0.50, 0.52, 0.55]]
    cy_guesses = [img_h * r for r in [0.45, 0.48, 0.50, 0.52, 0.55]]

    best_score = -1
    best_params = None

    total = len(f_guesses) * len(cx_guesses) * len(cy_guesses)
    n = 0
    for fx in f_guesses:
        for cx in cx_guesses:
            for cy in cy_guesses:
                fy = fx  # assume square pixels
                in_bounds, u, v, z, orig_idx = project_pcd_to_image(
                    xyz, img_w, img_h, fx, fy, cx, cy)
                pcd_render = render_pcd_rgb_vectorized(
                    rgb_pcd[orig_idx], u, v, in_bounds, img_w, img_h)
                score, mask = compute_alignment_score(rgb_img, pcd_render)
                n += 1
                if score > best_score:
                    best_score = score
                    best_params = (fx, fy, cx, cy)
                if n % 50 == 0:
                    print(f"  [{n}/{total}] best={best_score:.4f} @ fx={best_params[0]:.0f} cx={best_params[2]:.0f} cy={best_params[3]:.0f}")

    print(f"\nBest: score={best_score:.4f}")
    print(f"  fx={best_params[0]:.1f} fy={best_params[1]:.1f} cx={best_params[2]:.1f} cy={best_params[3]:.1f}")
    return best_params


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Verify PCD-2D alignment")
    parser.add_argument("--data_dir", default="data/new_sheef/pngs")
    parser.add_argument("--stem", default=None, help="Frame to check (default: first)")
    parser.add_argument("--output_dir", default="output/alignment_check")
    parser.add_argument("--fx", type=float, default=392.67)
    parser.add_argument("--fy", type=float, default=411.42)
    parser.add_argument("--cx", type=float, default=321.34)
    parser.add_argument("--cy", type=float, default=236.55)
    parser.add_argument("--img_w", type=int, default=640)
    parser.add_argument("--img_h", type=int, default=480)
    parser.add_argument("--grid_search", action="store_true",
                        help="Search for optimal intrinsics")
    parser.add_argument("--batch", type=int, default=0,
                        help="Run on first N frames (0 = single frame)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    pcd_paths = sorted(data_dir.glob("TV_*.pcd"))

    if args.stem:
        pcd_path = data_dir / f"{args.stem}.pcd"
        if not pcd_path.exists():
            print(f"Not found: {pcd_path}")
            sys.exit(1)
        pcd_paths = [pcd_path]
    elif args.batch > 0:
        pcd_paths = pcd_paths[:args.batch]
    else:
        pcd_paths = pcd_paths[:1]

    img_w, img_h = args.img_w, args.img_h
    fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy

    for pcd_path in pcd_paths:
        stem = pcd_path.stem
        num_id = stem.replace("TV_", "")

        # 找对应 RGB 图
        img_path = None
        for ext in [".jpg", ".png"]:
            cand = data_dir / f"{num_id}{ext}"
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            print(f"[{stem}] No image found, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"[{stem}]")
        print(f"  PCD: {pcd_path}")
        print(f"  Img: {img_path}")
        print(f"  Intrinsics: fx={fx} fy={fy} cx={cx} cy={cy}")
        print(f"{'='*60}")

        t0 = time.time()

        # 加载
        xyz, rgb_pcd = read_pcd_binary(str(pcd_path))
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Cannot read image!")
            continue
        if img.shape[0] != img_h or img.shape[1] != img_w:
            img = cv2.resize(img, (img_w, img_h))

        print(f"  PCD: {len(xyz)} points")
        print(f"  PCD RGB range: [{rgb_pcd.min(axis=0)}] - [{rgb_pcd.max(axis=0)}]")
        print(f"  Image: {img.shape}")

        # 网格搜索
        if args.grid_search:
            best = grid_search_intrinsics(xyz, rgb_pcd, img, img_w, img_h)
            fx, fy, cx, cy = best

        # 投影
        in_bounds, u, v, z, orig_idx = project_pcd_to_image(
            xyz, img_w, img_h, fx, fy, cx, cy)
        n_proj = in_bounds.sum()
        print(f"  Projected: {n_proj}/{len(xyz)} points in image ({100*n_proj/len(xyz):.1f}%)")

        # 渲染 PCD RGB
        pcd_render = render_pcd_rgb_vectorized(
            rgb_pcd[orig_idx], u, v, in_bounds, img_w, img_h)

        # 构建深度图用于边缘检测
        dm = np.full((img_h, img_w), np.nan, dtype=np.float32)
        ub, vb = u[in_bounds], v[in_bounds]
        zv = z[in_bounds]
        for i in range(len(ub)):
            if np.isnan(dm[vb[i], ub[i]]) or zv[i] < dm[vb[i], ub[i]]:
                dm[vb[i], ub[i]] = zv[i]

        # 对齐评分 (edge-based)
        score, mask = compute_alignment_score(img, pcd_render, dm)
        coverage = mask.sum() / (img_w * img_h) * 100
        print(f"  Alignment score: {score:.4f}  (0=bad, 1=perfect)")
        print(f"  PCD coverage: {coverage:.1f}% of image")
        print(f"  Time: {time.time()-t0:.2f}s")

        # 生成可视化
        viz_path = create_comparison_viz(img, pcd_render, mask, stem,
                                         fx, fy, cx, cy, args.output_dir)

    print(f"\nDone. Open {args.output_dir}/ to view results.")


if __name__ == "__main__":
    main()
