#!/usr/bin/env python3
"""从推理/审核 json(教师伪标签)构建 YOLO-pose 训练数据集。

标签源: output/shelf_pose_inference_v6x_conf50_act_0903 教师全量推理 json
        (每个 *_anchor_v2.json 一帧; num_keypoints==0 或 skipped 的帧剔除)
图像源: data/new_sheef/pngs (先找 TV_<stem>, 回退到数字名)

用法:
  # 只建 base(未增强) — 用于先看真实帧量与划分
  python tools/build_teacher_dataset.py --output_dir datasets/shelf_pose_teacher_v2

  # base + 轻度增强(物理合理档: rot±5 / scale0.8-1.2 / HSV / 10x)
  python tools/build_teacher_dataset.py --aug light --aug_per_img 10 \
      --output_dir datasets/shelf_pose_teacher_v2_light10

  # 重度增强(旧管线同款 rot±15/scale0.5-1.5/40x) — 仅供对比, 不推荐直接训
  python tools/build_teacher_dataset.py --aug heavy --aug_per_img 40 \
      --output_dir datasets/shelf_pose_teacher_v2_heavy40
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/code/OpenPCDet")
IMG_W, IMG_H, K = 640, 480, 2


# ────────────────────────────── 标签 → YOLO pose ──────────────────────────────
def keypoints_to_yolo(kps, img_w=IMG_W, img_h=IMG_H, K=K):
    """与 reviewed_to_train 一致: 单对象, 关键点按 x 排序进槽位。"""
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


def find_image(data_dir, stem):
    for cand_name in [stem, stem.replace("TV_", "")]:
        for ext in [".jpg", ".png"]:
            cand = Path(data_dir) / f"{cand_name}{ext}"
            if cand.exists():
                return cand
    return None


def load_samples(src, data_dir, only_2kp=False):
    """读 json 目录 → [{stem, keypoints:[(u,v),...]}] (空锚点/skipped 剔除)。"""
    samples, dropped = [], {"empty": 0, "skipped": 0, "no_img": 0, "bad_img": 0,
                            "not2kp": 0}
    for p in sorted(Path(src).glob("*_anchor_v2.json")):
        d = json.loads(p.read_text())
        stem = d.get("stem", p.name[:-len("_anchor_v2.json")])
        if d.get("skipped"):
            dropped["skipped"] += 1
            continue
        kps = [tuple(kp["pixel_uv"]) for kp in d.get("keypoints", [])]
        if len(kps) < 1:
            dropped["empty"] += 1
            continue
        if only_2kp and len(kps) != 2:
            dropped["not2kp"] += 1
            continue
        img_path = find_image(data_dir, stem)
        if img_path is None:
            dropped["no_img"] += 1
            continue
        if cv2.imread(str(img_path)) is None:
            dropped["bad_img"] += 1
            continue
        samples.append({"stem": stem, "keypoints": kps})
    return samples, dropped


# ────────────────────────────── base 划分 ──────────────────────────────
def build_base(samples, out_dir, val_ratio, seed):
    random.seed(seed)
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (out_dir / sub).mkdir(parents=True)

    random.shuffle(samples)
    n_val = max(1, int(len(samples) * val_ratio))

    counts = {}
    written = {"train": 0, "val": 0}
    for split, items in [("train", samples[n_val:]), ("val", samples[:n_val])]:
        kp_total = 0
        n2 = n1 = 0
        for s in items:
            img_path = find_image(s["_data_dir"], s["stem"])
            img = cv2.imread(str(img_path))
            if img.shape[:2] != (IMG_H, IMG_W):
                img = cv2.resize(img, (IMG_W, IMG_H))
            label = keypoints_to_yolo(s["keypoints"])
            if label is None:
                continue
            cv2.imwrite(str(out_dir / "images" / split / f"{s['stem']}.jpg"),
                        img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            (out_dir / "labels" / split / f"{s['stem']}.txt").write_text(label + "\n")
            n = len(s["keypoints"])
            n2 += n == 2
            n1 += n == 1
            kp_total += n
            written[split] += 1
        counts[split] = (written[split], n2, n1, kp_total)
    return counts, n_val


def write_yaml(out_dir):
    out_dir = Path(out_dir)
    (out_dir / "dataset.yaml").write_text(
        f"""# 货架关键点检测 — 教师伪标签数据集
path: {out_dir.resolve()}
train: images/train
val: images/val
kpt_shape: [{K}, 3]
flip_idx: [{', '.join(str(i) for i in range(K))}]
names:
  0: shelf
nc: 1
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "output/shelf_pose_inference_v6x_conf50_act_0903"))
    ap.add_argument("--data_dir", default=str(REPO / "data/new_sheef/pngs"))
    ap.add_argument("--output_dir", default=str(REPO / "datasets/shelf_pose_teacher_v2"))
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aug", default="none", choices=["none", "light", "heavy"])
    ap.add_argument("--aug_per_img", type=int, default=10)
    ap.add_argument("--only_2kp", action="store_true",
                    help="只收 2 关键点帧 (剔除单锚修正)")
    args = ap.parse_args()

    samples, dropped = load_samples(args.src, args.data_dir, args.only_2kp)
    for s in samples:
        s["_data_dir"] = args.data_dir
    print(f"src: {args.src}")
    print(f"json 剔除: {dropped}, 可用帧: {len(samples)}")

    counts, n_val = build_base(samples, args.output_dir, args.val_ratio, args.seed)
    for split, (n, n2, n1, kp) in counts.items():
        print(f"  base[{split}] {n} 帧 (双锚点{n2}/单锚点{n1}, {kp} kp)")
    write_yaml(args.output_dir)
    print(f"base 数据集: {args.output_dir}  (val={n_val})")
    if args.aug != "none":
        from reviewed_to_train import augment_dataset  # heavy 档沿用旧实现
        print(f"[警告] --aug 增强逻辑尚未接入 build_teacher_dataset, 请在下一步实现/复用 reviewed_to_train.augment_dataset")
    # NOTE: 轻度增强档尚未在本脚本实现; 增强在架构/训练定稿后再以独立脚本叠加,
    #       base 始终保留, 保证随时可回归。


if __name__ == "__main__":
    main()
