#!/usr/bin/env python3
"""对 base 数据集做"物理合理"轻度增强(train only), val 原样透传。

设计依据: 相机固定安装、货架横梁水平、车靠近只有小幅姿态变化 →
  只增强现实中会出现的: 光照/HSV、轻缩放(远近)、小旋转/平移(车位姿)、
  低概率轻模糊/噪声; 关掉会失真的: 大角度旋转、强缩放、shear、fliplr、mosaic。

用法:
  python tools/augment_pose_dataset.py \
      --input datasets/shelf_pose_teacher_v2 \
      --output datasets/shelf_pose_teacher_v2_light8 \
      --aug_per_img 8 --mixup_prob 0.1 --seed 42
"""
import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

IMG_W, IMG_H = 640, 480


# ────────────────────────────── 轻增强单图 ──────────────────────────────
def augment_single_light(img, kps):
    """轻度几何(单次仿射合并缩放/旋转/平移) + 强光度增强 + 轻模糊/噪声。

    光度增强针对"光照变化"(只改颜色/明暗, 不动几何、不混标签):
      HSV(V 大范围模拟暗仓库~过曝) + 亮对比 + gamma(曝光) + CLAHE(强阴影) +
      色温/色偏 + 通道dropout(对单色/色偏鲁棒)。

    kps: [(u, v), ...]  (像素坐标)
    返回: (aug_img, aug_kps)
    """
    h, w = img.shape[:2]
    angle = np.random.uniform(-5, 5)
    s = np.random.uniform(0.8, 1.2)
    tx = np.random.uniform(-0.03, 0.03) * w
    ty = np.random.uniform(-0.03, 0.03) * h
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, s)
    M[0, 2] += tx
    M[1, 2] += ty

    aug_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT101)
    aug_kps = []
    for (u, v) in kps:
        new_pt = M @ np.array([u, v, 1.0])
        aug_kps.append((float(new_pt[0]), float(new_pt[1])))

    # ── 光度增强 (光照鲁棒) ──
    # 1) HSV: V 大范围(暗~过曝), S 中, H 轻(货架颜色中性)
    hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + np.random.uniform(-12, 12), 0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.55, 1.5), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.random.uniform(0.45, 1.6), 0, 255)
    aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 2) 亮度/对比 (线性)
    alpha = np.random.uniform(0.65, 1.35)
    beta = np.random.uniform(-45, 45)
    aug_img = np.clip(alpha * aug_img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # 3) gamma (幂律, 模拟曝光/背光): 亮度非线性重映射
    if np.random.random() < 0.6:
        gamma = np.random.uniform(0.45, 2.2)
        lut = np.clip((np.arange(256) / 255.0) ** (1.0 / gamma) * 255, 0, 255).astype(np.uint8)
        aug_img = cv2.LUT(aug_img, lut)

    # 4) CLAHE (局部对比均衡, 模拟强光+阴影并存): 只对亮度通道做
    if np.random.random() < 0.5:
        ycrcb = cv2.cvtColor(aug_img, cv2.COLOR_BGR2YCrCb)
        clahe = cv2.createCLAHE(clipLimit=np.random.uniform(1.5, 4.0), tileGridSize=(8, 8))
        ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
        aug_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # 5) 色偏/色温 (暖/冷 LED 等): R/B 通道独立缩放
    if np.random.random() < 0.5:
        scale_r = np.random.uniform(0.85, 1.2)
        scale_b = np.random.uniform(0.85, 1.2)
        b, g, r = cv2.split(aug_img)
        r = np.clip(r.astype(np.float32) * scale_r, 0, 255)
        b = np.clip(b.astype(np.float32) * scale_b, 0, 255)
        aug_img = cv2.merge([b.astype(np.uint8), g, r.astype(np.uint8)])

    # 6) 通道丢失/灰度混合 (对色偏/单色鲁棒)
    if np.random.random() < 0.2:
        gray = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY)
        t = np.random.uniform(0.5, 1.0)
        aug_img = cv2.addWeighted(aug_img, t, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 1.0 - t, 0)

    # 轻模糊 / 轻噪声 (各 10%)
    if np.random.random() < 0.1:
        aug_img = cv2.GaussianBlur(aug_img, (3, 3), 0)
    if np.random.random() < 0.1:
        noise = np.random.normal(0, np.random.uniform(3, 8), aug_img.shape)
        aug_img = np.clip(aug_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return aug_img, aug_kps


def read_label(lbl_path, img_w=IMG_W, img_h=IMG_H):
    """读 YOLO pose label → [[(u,v), ...], ...](每对象一组, 像素坐标)。"""
    objs = []
    for line in Path(lbl_path).read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        K = (len(parts) - 5) // 3
        kps = []
        for i in range(K):
            vis = float(parts[5 + i * 3 + 2])
            if vis > 0:
                u = float(parts[5 + i * 3]) * img_w
                v = float(parts[5 + i * 3 + 1]) * img_h
                kps.append((u, v))
        if kps:
            objs.append(kps)
    return objs


def write_label(kp_objs, lbl_path, img_w=IMG_W, img_h=IMG_H, K=2):
    """kp_objs → YOLO pose txt (单类; 每对象一行, K 个槽位, 缺失 vis=0)。"""
    lines = []
    for kps in kp_objs:
        if not kps:
            continue
        kps_sorted = sorted(kps, key=lambda k: k[0])
        us = [k[0] / img_w for k in kps_sorted]
        vs = [k[1] / img_h for k in kps_sorted]
        cx = (min(us) + max(us)) / 2
        cy = (min(vs) + max(vs)) / 2
        bw = min(max(max(us) - min(us), 0.05) * 1.2, 1.0)
        bh = min(max(max(vs) - min(vs), 0.05) * 1.2, 1.0)

        n = len(kps_sorted)
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
        lines.append(f"0.000000 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " +
                     " ".join(f"{x:.6f}" for x in norm))
    Path(lbl_path).write_text("\n".join(lines) + "\n" if lines else "")


def build_yaml(input_dir, output_dir):
    base = Path(input_dir) / "dataset.yaml"
    out = Path(output_dir) / "dataset.yaml"
    if base.exists():
        text = base.read_text()
    else:
        text = ("path: {out}\ntrain: images/train\nval: images/val\n"
                "kpt_shape: [2, 3]\nflip_idx: [0, 1]\nnames:\n  0: shelf\nnc: 1\n")
    text = text.replace(str(Path(input_dir).resolve()), str(Path(output_dir).resolve()))
    out.write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="datasets/shelf_pose_teacher_v2")
    ap.add_argument("--output", default="datasets/shelf_pose_teacher_v2_light8")
    ap.add_argument("--aug_per_img", type=int, default=8)
    ap.add_argument("--mixup_prob", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (out_dir / sub).mkdir(parents=True)

    # val 原样透传
    n_val = 0
    for img_path in sorted((in_dir / "images" / "val").glob("*.*")):
        if img_path.suffix.lower() not in (".jpg", ".png"):
            continue
        stem = img_path.stem
        shutil.copy(img_path, out_dir / "images" / "val" / img_path.name)
        lbl = in_dir / "labels" / "val" / f"{stem}.txt"
        if lbl.exists():
            shutil.copy(lbl, out_dir / "labels" / "val" / lbl.name)
            n_val += 1

    # train 轻度增强
    n_train = n_total = 0
    for img_path in sorted((in_dir / "images" / "train").glob("*.*")):
        if img_path.suffix.lower() not in (".jpg", ".png"):
            continue
        stem = img_path.stem
        lbl_path = in_dir / "labels" / "train" / f"{stem}.txt"
        if not lbl_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None or img.shape[:2] != (IMG_H, IMG_W):
            img = cv2.resize(img, (IMG_W, IMG_H))
        base_objs = read_label(lbl_path)
        if not base_objs:
            continue

        # 原图 + 增强副本
        out_img = out_dir / "images" / "train"
        out_lbl = out_dir / "labels" / "train"
        cv2.imwrite(str(out_img / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        write_label([k[:] for k in base_objs], out_lbl / f"{stem}.txt")
        n_train += 1

        for i in range(args.aug_per_img):
            aug_objs = []
            for obj in base_objs:
                a_img, a_kps = augment_single_light(img, obj)
                a_kps = [(u, v) for (u, v) in a_kps
                         if 0 <= u < IMG_W and 0 <= v < IMG_H]
                if len(a_kps) >= 1:
                    aug_objs.append(a_kps)
            if not aug_objs:
                continue
            # mixup (10%): 与随机基样混合, 标签合并
            if random.random() < args.mixup_prob:
                p_lbl = random.choice(list((in_dir / "labels" / "train").glob("*.txt")))
                p_img = cv2.imread(str(in_dir / "images" / "train" / f"{p_lbl.stem}.jpg"))
                if p_img is not None:
                    if p_img.shape[:2] != (IMG_H, IMG_W):
                        p_img = cv2.resize(p_img, (IMG_W, IMG_H))
                    p_objs = read_label(p_lbl)
                    if p_objs:
                        pa_img, pa_kps = augment_single_light(p_img, p_objs[0])
                        pa_kps = [(u, v) for (u, v) in pa_kps
                                  if 0 <= u < IMG_W and 0 <= v < IMG_H]
                        lam = np.random.uniform(0.4, 0.6)
                        m_img = np.clip(lam * a_img.astype(np.float32) +
                                        (1 - lam) * pa_img.astype(np.float32),
                                        0, 255).astype(np.uint8)
                        m_objs = [k[:] for k in aug_objs]
                        if pa_kps:
                            m_objs.append(pa_kps)
                        name = f"{stem}_mix{i:03d}"
                        cv2.imwrite(str(out_img / f"{name}.jpg"), m_img,
                                    [cv2.IMWRITE_JPEG_QUALITY, 90])
                        write_label(m_objs, out_lbl / f"{name}.txt")
                        n_train += 1
            cv2.imwrite(str(out_img / f"{stem}_a{i:03d}.jpg"), a_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            write_label(aug_objs, out_lbl / f"{stem}_a{i:03d}.txt")
            n_train += 1
        n_total += 1

    build_yaml(in_dir, out_dir)
    print(f"base 帧: {n_total} (train), val 透传: {n_val}")
    print(f"增强后 train 总样本: {n_train} (≈{n_train/max(n_total,1):.1f}x/底图)")
    print(f"输出: {out_dir}")


if __name__ == "__main__":
    main()
