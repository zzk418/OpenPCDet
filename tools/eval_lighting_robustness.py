#!/usr/bin/env python3
"""光照鲁棒性评估: 在带标签的 val 集上, 对每张干净图施加确定性光度扰动,
对比 r2(弱增强) vs r3(强增强) 的关键点定位误差。光度增强不改几何, 所以
"扰动图关键点 ≈ 干净图 GT 关键点" 是模型光照不变性的直接判据。

用法:
  conda run -n pc python tools/eval_lighting_robustness.py \
      --val /code/OpenPCDet/datasets/shelf_pose_mnv3corr \
      --weights /code/OpenPCDet/output/best_ckpts/shelf_mobilenet_r2_best.pt \
                /code/OpenPCDet/output/best_ckpts/shelf_mobilenet_r3_best.pt
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

W, H = 640, 480


# ── 确定性光度扰动 (与 augment_single_light 的光度部分同族, 但固定参数便于对照) ──
def _hsv_scale(img, v_mult, s_mult=1.0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_mult, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_mult, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _gamma(img, g):
    lut = np.clip((np.arange(256) / 255.0) ** (1.0 / g) * 255, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def _color_cast(img, r_scale, b_scale):
    b, g, r = cv2.split(img)
    r = np.clip(r.astype(np.float32) * r_scale, 0, 255)
    b = np.clip(b.astype(np.float32) * b_scale, 0, 255)
    return cv2.merge([b.astype(np.uint8), g, r.astype(np.uint8)])


def _clahe(img, clip=3.0):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def _gray_blend(img, t=0.8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.addWeighted(img, 1 - t, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), t, 0)


def _low_contrast(img, alpha=0.55, beta=25):
    return np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)


PERTURB = {
    "dark_V0.5":       lambda im: _hsv_scale(im, 0.5),
    "bright_V1.45":    lambda im: _hsv_scale(im, 1.45),
    "gamma0.5":        lambda im: _gamma(im, 0.5),
    "gamma2.0":        lambda im: _gamma(im, 2.0),
    "warm":            lambda im: _color_cast(im, 1.18, 0.82),
    "cold":            lambda im: _color_cast(im, 0.82, 1.18),
    "clahe3":          lambda im: _clahe(im, 3.0),
    "gray0.8":         lambda im: _gray_blend(im, 0.8),
    "low_contrast":    lambda im: _low_contrast(im),
    "sat0.5":          lambda im: _hsv_scale(im, 1.0, 0.5),
}


def read_gt_kps(lbl_path):
    """读 val 标签 → [(x, y), ...] 像素坐标 (已按 x 升序)。"""
    parts = Path(lbl_path).read_text().split()
    K = (len(parts) - 5) // 3
    kps = []
    for i in range(K):
        if float(parts[5 + i * 3 + 2]) > 0:
            kps.append((float(parts[5 + i * 3]) * W, float(parts[5 + i * 3 + 1]) * H))
    return sorted(kps, key=lambda p: p[0])


def predict_kps(model, img):
    """推理最高置信度实例的关键点 → [(x, y), ...] 像素坐标 (按 x 升序)。"""
    res = model.predict(img, imgsz=640, verbose=False, conf=0.25)[0]
    k = res.keypoints
    if k is None or k.xy is None or len(k.xy) == 0:
        return None
    if res.boxes is not None and len(res.boxes) > 0:
        i = int(np.argmax(res.boxes.conf.cpu().numpy()))
    else:
        i = 0
    pts = k.xy[i].cpu().numpy()[:, :2]
    return sorted([(float(p[0]), float(p[1])) for p in pts], key=lambda p: p[0])


def kpt_err(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    if a.shape != b.shape:
        return float("nan")
    return float(np.mean(np.hypot(a[:, 0] - b[:, 0], a[:, 1] - b[:, 1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--weights", nargs="+", required=True)
    args = ap.parse_args()

    models = {Path(w).name: YOLO(w) for w in args.weights}
    val = Path(args.val)
    imgs = sorted((val / "images" / "val").glob("*.jpg"))
    labels = sorted((val / "labels" / "val").glob("*.txt"))
    assert len(imgs) == len(labels), "images/labels 数量不一致"

    print(f"val 图像 {len(imgs)} 张, 扰动 {len(PERTURB)} 种, 模型 {list(models)}")

    # 每图: 干净 GT + 各扰动下各模型关键点
    # errs[model][perturb] = [err_per_img]
    errs = {m: {p: [] for p in PERTURB} for m in models}
    clean_gt_errs = {m: [] for m in models}

    for img_path, lbl_path in zip(imgs, labels):
        img = cv2.imread(str(img_path))
        if img is None or img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H))
        gt = read_gt_kps(lbl_path)
        for mname, model in models.items():
            pk = predict_kps(model, img)
            if pk is not None:
                clean_gt_errs[mname].append(kpt_err(pk, gt))
            for pname, fn in PERTURB.items():
                paug = fn(img)
                ppk = predict_kps(model, paug)
                if ppk is not None:
                    errs[mname][pname].append(kpt_err(ppk, gt))

    # 汇总
    def mean(x):
        x = [v for v in x if v == v]  # 去 nan
        return float(np.mean(x)) if x else float("nan")

    header = f"{'扰动':<14}" + "".join(f"{m:>14}" for m in models) + f"{'Δ(r3-r2)':>12}"
    print("\n关键点定位误差 (px, vs GT, 越小越好) — 干净图:")
    print(f"{'clean(干净)':<14}" + "".join(f"{mean(clean_gt_errs[m]):>14.2f}" for m in models))
    print("\n关键点定位误差 (px, vs GT) — 各光度扰动下:")
    print(header)
    for p in PERTURB:
        row = f"{p:<14}"
        for m in models:
            row += f"{mean(errs[m][p]):>14.2f}"
        # Δ = r3 - r2 (若只有两个模型; 取后一个减前一个)
        if len(models) == 2:
            a = mean(errs[list(models)[0]][p])
            b = mean(errs[list(models)[1]][p])
            row += f"{b - a:>+12.2f}"
        print(row)

    # 总体: 扰动下平均误差
    print("\n=== 扰动下平均误差 (px, 越小越鲁棒) ===")
    for m in models:
        allv = [e for p in PERTURB for e in errs[m][p] if e == e]
        print(f"  {m:<40} {mean(allv):>6.2f}  (干净 {mean(clean_gt_errs[m]):.2f})")


if __name__ == "__main__":
    main()
