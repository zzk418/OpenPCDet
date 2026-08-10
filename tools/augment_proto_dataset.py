#!/usr/bin/env python3
"""对 12 张原型图像做离线数据增强，生成 YOLO-Pose 训练集。

原理: 12 张图太少，YOLO 学不到。但我们可以离线生成大量增强样本
(旋转、平移、缩放、HSV 抖动)，同时精确计算变换后的关键点坐标。

用法:
  python tools/augment_proto_dataset.py                 # 生成数据集
  python tools/augment_proto_dataset.py --aug_per_img 50  # 每张图 50 个增强版
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import albumentations as A
import cv2
import numpy as np


def load_prototypes(ann_dir, img_dir):
    """加载原型标注 → [{stem, img_path, keypoints: [(u,v), ...]}]"""
    ann_dir = Path(ann_dir)
    img_dir = Path(img_dir)
    protos = []

    for af in sorted(ann_dir.glob("cluster*_anchor_v2.json")):
        with open(af) as f:
            d = json.load(f)
        stem = d["stem"]
        img_path = img_dir / f"{stem}.jpg"
        if not img_path.exists():
            continue
        kps = [(kp["pixel_uv"][0], kp["pixel_uv"][1]) for kp in d["keypoints"]]
        protos.append({"stem": stem, "img_path": img_path, "keypoints": kps})

    return protos


def build_augmentation_pipeline():
    """构建增强 pipeline。关键点坐标会随几何变换自动更新。"""
    return A.Compose([
        # 几何变换
        A.Affine(
            scale=(0.8, 1.2),           # 缩放 0.8~1.2
            translate_percent=(-0.1, 0.1),  # 平移 ±10%
            rotate=(-15, 15),            # 旋转 ±15°
            shear=(-5, 5),               # 剪切 ±5°
            p=0.8,
        ),
        # 透视变换 (模拟不同相机角度)
        A.Perspective(scale=(0.02, 0.05), p=0.3),
        # HSV 颜色抖动 (模拟不同光照)
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=30,
            val_shift_limit=30,
            p=0.8,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5,
        ),
        # 模糊 (模拟运动模糊)
        A.MotionBlur(blur_limit=5, p=0.2),
        # 噪声
        A.GaussNoise(var_limit=(5, 20), p=0.2),
    ], keypoint_params=A.KeypointParams(
        format='xy',        # (x, y) = (u, v)
        remove_invisible=False,
    ))


def keypoints_to_yolo(kps, img_w=640, img_h=480, K=4):
    """将关键点列表转换为 YOLO pose 格式字符串。"""
    n = len(kps)
    if n == 0:
        return None

    # 排序: 按 u 从左到右
    kps = sorted(kps, key=lambda k: k[0])

    # 将 n 个关键点分配到 K 个 slot
    if n == 1:
        slots = [0]
    else:
        slots = np.round(np.linspace(0, K - 1, n)).astype(int).tolist()

    us = [k[0] / img_w for k in kps]
    vs = [k[1] / img_h for k in kps]

    cx = (min(us) + max(us)) / 2
    cy = (min(vs) + max(vs)) / 2
    bw = max(max(us) - min(us), 0.05) * 1.2
    bh = max(max(vs) - min(vs), 0.05) * 1.2
    bw, bh = min(bw, 1.0), min(bh, 1.0)

    slot_set = set(slots)
    slot_map = {s: i for i, s in enumerate(slots)}

    norm = []
    for i in range(K):
        if i in slot_set:
            u, v = kps[slot_map[i]]
            norm.extend([u / img_w, v / img_h, 2.0])
        else:
            norm.extend([0.0, 0.0, 0.0])

    return f"0.000000 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " + \
           " ".join(f"{x:.6f}" for x in norm)


def generate_dataset(protos, output_dir, aug_per_img=30, val_ratio=0.15, seed=42):
    """生成增强数据集。

    Args:
        protos: load_prototypes() 返回的原型列表
        output_dir: 输出目录
        aug_per_img: 每张原型图生成的增强样本数
        val_ratio: 验证集比例 (从增强样本中随机选)
    """
    random.seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (output_dir / sub).mkdir(parents=True)

    pipeline = build_augmentation_pipeline()
    img_w, img_h = 640, 480
    K = 4

    all_samples = []  # [(img, kps, name), ...]

    print(f"Generating {aug_per_img} augmentations per prototype × {len(protos)} prototypes...")

    for proto in protos:
        img = cv2.imread(str(proto["img_path"]))
        if img is None:
            continue
        if img.shape[:2] != (img_h, img_w):
            img = cv2.resize(img, (img_w, img_h))

        kps_xy = proto["keypoints"]  # [(u, v), ...]

        # 原始图 (无增强)
        all_samples.append((img.copy(), kps_xy.copy(),
                            f"{proto['stem']}_orig"))

        # 生成增强样本
        for i in range(aug_per_img):
            augmented = pipeline(image=img, keypoints=kps_xy)
            aug_img = augmented["image"]
            aug_kps = augmented["keypoints"]

            # 过滤: 只保留在图像范围内的关键点
            valid_kps = []
            for (u, v) in aug_kps:
                if 0 <= u < img_w and 0 <= v < img_h:
                    valid_kps.append((u, v))

            if len(valid_kps) >= 1:  # 至少 1 个可见关键点
                name = f"{proto['stem']}_aug{i:03d}"
                all_samples.append((aug_img, valid_kps, name))

    print(f"Total samples: {len(all_samples)}")

    # 分割 train/val
    random.shuffle(all_samples)
    n_val = max(1, int(len(all_samples) * val_ratio))
    train_samples = all_samples[n_val:]
    val_samples = all_samples[:n_val]

    # 写文件
    for split, samples in [("train", train_samples), ("val", val_samples)]:
        kp_total = 0
        for img, kps, name in samples:
            label = keypoints_to_yolo(kps, img_w, img_h, K=K)
            if label is None:
                continue

            cv2.imwrite(
                str(output_dir / "images" / split / f"{name}.jpg"),
                img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            with open(output_dir / "labels" / split / f"{name}.txt", "w") as f:
                f.write(label + "\n")
            kp_total += len(kps)
        print(f"  [{split}] {len(samples)} samples, {kp_total} kp")

    # 写 dataset.yaml
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(f"""# 货架关键点检测 — 离线增强原型数据集
path: {output_dir.resolve()}
train: images/train
val: images/val
kpt_shape: [{K}, 3]
flip_idx: [{', '.join(str(i) for i in range(K))}]
names:
  0: shelf
nc: 1
""")

    print(f"\nDataset: {output_dir}")
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")
    print(f"  YAML: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_dir",
                        default="/code/OpenPCDet/data/new_sheef/prototype_annotations")
    parser.add_argument("--img_dir",
                        default="/code/OpenPCDet/data/new_sheef/prototypes")
    parser.add_argument("--output_dir",
                        default="datasets/shelf_pose_aug")
    parser.add_argument("--aug_per_img", type=int, default=30)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading prototypes...")
    protos = load_prototypes(args.ann_dir, args.img_dir)
    print(f"Loaded {len(protos)} prototypes")
    for p in protos:
        print(f"  {p['stem']}: {len(p['keypoints'])} kp")

    generate_dataset(protos, args.output_dir,
                     aug_per_img=args.aug_per_img,
                     val_ratio=args.val_ratio,
                     seed=args.seed)


if __name__ == "__main__":
    main()
