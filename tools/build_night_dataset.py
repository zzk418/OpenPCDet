#!/usr/bin/env python3
"""
生成夜间合成增强数据集 shelf_pose_reviewed_aug_night:
  - 复制原 train 集 + 每张 train 图生成一张夜间版 (1:1 混入)
  - 夜间变换: 亮度 ×U(0.3,0.6) + 对比度 U(0.7,1.0) + 灰度混合 U(0.4,1.0) + 高斯噪声
  - val 集保持原样 (评估白天性能不回归)
"""
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

SRC = Path("/code/OpenPCDet/datasets/shelf_pose_reviewed_aug")
DST = Path("/code/OpenPCDet/datasets/shelf_pose_reviewed_aug_night")
SEED = 0


def nightify(img):
    rng = np.random.default_rng()
    img = img.astype(np.float32)
    v = rng.uniform(0.3, 0.6)          # 整体压暗
    img *= v
    c = rng.uniform(0.7, 1.0)          # 对比度
    mean = img.mean(axis=(0, 1), keepdims=True)
    img = (img - mean) * c + mean
    g = rng.uniform(0.4, 1.0)          # 灰度混合
    gray = img.mean(axis=2, keepdims=True)
    img = img * g + gray * (1 - g)
    noise = rng.normal(0, rng.uniform(2, 7), img.shape)  # 暗部噪声
    img += noise
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    random.seed(SEED)
    if DST.exists():
        raise SystemExit(f"{DST} 已存在, 先手动处理")

    # val: 原样复制
    shutil.copytree(SRC / "images" / "val", DST / "images" / "val")
    shutil.copytree(SRC / "labels" / "val", DST / "labels" / "val")

    # train: 复制 + 夜间版
    (DST / "images" / "train").mkdir(parents=True)
    (DST / "labels" / "train").mkdir(parents=True)
    train_imgs = sorted((SRC / "images" / "train").glob("*.jpg"))
    for i, jpg in enumerate(train_imgs):
        stem = jpg.stem
        lab = SRC / "labels" / "train" / f"{stem}.txt"
        # 原图
        shutil.copy2(jpg, DST / "images" / "train" / jpg.name)
        shutil.copy2(lab, DST / "labels" / "train" / f"{stem}.txt")
        # 夜间版
        img = cv2.imread(str(jpg))
        night = nightify(img)
        cv2.imwrite(str(DST / "images" / "train" / f"{stem}_night.jpg"), night,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        shutil.copy2(lab, DST / "labels" / "train" / f"{stem}_night.txt")
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(train_imgs)}")

    n_train = len(list((DST / "images" / "train").glob("*.jpg")))
    n_val = len(list((DST / "images" / "val").glob("*.jpg")))
    print(f"Done: train={n_train} (原 {len(train_imgs)} + 夜间 {len(train_imgs)}), val={n_val}")

    yaml_text = f"""# 货架关键点检测 — 审核+增强+夜间合成数据集
path: /code/OpenPCDet/datasets/shelf_pose_reviewed_aug_night
train: images/train
val: images/val
kpt_shape: [2, 3]
flip_idx: [0, 1]
names:
  0: shelf
nc: 1
"""
    (DST / "dataset.yaml").write_text(yaml_text)
    print("dataset.yaml 已写入")


if __name__ == "__main__":
    main()
