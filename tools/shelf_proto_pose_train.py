#!/usr/bin/env python3
"""
将标注好的原型图像转换为 YOLO-Pose 训练数据并训练。

工作流:
  1. 读取 data/new_sheef/prototype_annotations/cluster*_anchor_v2.json (12个)
  2. 找到对应的原型图像 data/new_sheef/prototypes/cluster*_TV_*.jpg
  3. 转换为 YOLO keypoint 格式 (class_id bbox x1y1v1 ... xKyKvK)
  4. 用 YOLOv8-pose 训练 (12 张原图 + 数据增强)

12 张图很少，关键策略:
  - 强数据增强: mosaic, 旋转变换, HSV 抖动, 缩放
  - 小模型: yolov8n-pose (3M 参数)
  - 冻结 backbone, 只训练 keypoint head
  - 低学习率, 多 epoch

用法:
  python tools/shelf_proto_pose_train.py                     # 快速训练
  python tools/shelf_proto_pose_train.py --epochs 200         # 更长时间训练
  python tools/shelf_proto_pose_train.py --model yolov8s-pose.pt  # 大模型
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# 项目根目录 (主仓库)
REPO_ROOT = Path("/code/OpenPCDet")


def load_prototype_annotations(ann_dir, proto_img_dir):
    """加载原型标注, 返回 {stem: {img_path, keypoints: [{pixel_uv, anchor_3d, label}]}}

    Args:
        ann_dir: prototype_annotations 目录
        proto_img_dir: prototypes 图像目录

    Returns:
        dict[stem] = {img_path, keypoints}
    """
    ann_dir = Path(ann_dir)
    proto_img_dir = Path(proto_img_dir)
    annotations = {}

    for af in sorted(ann_dir.glob("cluster*_anchor_v2.json")):
        with open(af) as f:
            d = json.load(f)

        stem = d.get("stem", af.stem.replace("_anchor_v2", ""))
        kps = d.get("keypoints", [])

        if not kps:
            print(f"  SKIP {stem}: no keypoints")
            continue

        # 找图像
        img_path = proto_img_dir / f"{stem}.jpg"
        if not img_path.exists():
            # 尝试其他扩展名
            for ext in [".png", ".jpeg"]:
                cand = proto_img_dir / f"{stem}{ext}"
                if cand.exists():
                    img_path = cand
                    break

        if not img_path.exists():
            print(f"  SKIP {stem}: image not found ({img_path})")
            continue

        annotations[stem] = {
            "img_path": img_path,
            "keypoints": [
                {
                    "pixel_uv": kp["pixel_uv"],
                    "anchor_3d": kp.get("anchor_3d"),
                    "label": kp.get("label", ""),
                }
                for kp in kps
            ],
        }

    return annotations


def keypoints_to_yolo_pose(kps, img_w, img_h, K=4):
    """将关键点转换为 YOLO pose 格式字符串。

    Args:
        kps: [{pixel_uv, ...}] 关键点列表
        img_w, img_h: 图像尺寸
        K: 最大关键点数 (固定输出 K 个 slot)

    Returns:
        YOLO 格式字符串: "class_id cx cy w h x1 y1 v1 ... xK yK vK"
        或 None (无有效关键点)
    """
    if not kps:
        return None

    n = len(kps)

    # 将 n 个关键点均匀分配到 K 个 slot
    if n == 1:
        assignments = [(0, 0)]  # (kp_idx, slot_idx)
    else:
        slot_indices = np.round(np.linspace(0, K - 1, n)).astype(int)
        assignments = list(zip(range(n), slot_indices))

    # 构建归一化关键点序列
    norm_kps = []
    slot_set = set(s for _, s in assignments)
    kp_by_slot = {s: i for i, s in assignments}

    for slot in range(K):
        if slot in slot_set:
            kp = kps[kp_by_slot[slot]]
            u, v = kp["pixel_uv"]
            norm_kps.extend([u / img_w, v / img_h, 2.0])  # v=2: 可见且已标注
        else:
            norm_kps.extend([0.0, 0.0, 0.0])  # v=0: 不可见

    # Bounding box: 所有可见关键点 + 20% padding
    us = [k["pixel_uv"][0] / img_w for k in kps]
    vs = [k["pixel_uv"][1] / img_h for k in kps]

    cx = (min(us) + max(us)) / 2
    cy = (min(vs) + max(vs)) / 2
    bw = max(max(us) - min(us), 0.05) * 1.2
    bh = max(max(vs) - min(vs), 0.05) * 1.2

    # Clamp
    bw = min(bw, 1.0)
    bh = min(bh, 1.0)

    parts = [0, cx, cy, bw, bh] + norm_kps  # class 0 = shelf
    return " ".join(f"{x:.6f}" for x in parts)


def prepare_proto_dataset(annotations, output_dir, K=4, val_ratio=0.1, seed=42):
    """从原型标注创建 YOLO-Pose 数据集。

    Args:
        annotations: load_prototype_annotations() 返回值
        output_dir: 输出目录 (如 datasets/shelf_pose_proto)
        K: 最大关键点数
        val_ratio: 验证集比例
        seed: 随机种子
    """
    random.seed(seed)

    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # 创建目录结构
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (output_dir / sub).mkdir(parents=True)

    stems = list(annotations.keys())
    random.shuffle(stems)

    # 至少留 1 张做验证
    n_val = max(1, int(len(stems) * val_ratio))
    n_train = len(stems) - n_val

    train_stems = sorted(stems[:n_train])
    val_stems = sorted(stems[n_train:])

    print(f"\n  Train: {n_train} images, Val: {n_val} images")
    print(f"  K={K}, Total prototypes: {len(stems)}")

    img_w, img_h = 640, 480  # 采集分辨率

    kp_dist = {}  # keypoint count distribution
    train_kp_total, val_kp_total = 0, 0

    for split_name, split_stems in [("train", train_stems), ("val", val_stems)]:
        for stem in split_stems:
            ann = annotations[stem]

            # 读图
            img = cv2.imread(str(ann["img_path"]))
            if img is None:
                print(f"  WARNING: cannot read {ann['img_path']}")
                continue

            # 统一尺寸
            if img.shape[0] != img_h or img.shape[1] != img_w:
                img = cv2.resize(img, (img_w, img_h))

            # 保存图像
            out_img = output_dir / "images" / split_name / f"{stem}.jpg"
            cv2.imwrite(str(out_img), img, [cv2.IMWRITE_JPEG_QUALITY, 92])

            # 写标注
            kps = ann["keypoints"]
            n_kp = len(kps)
            kp_dist[n_kp] = kp_dist.get(n_kp, 0) + 1

            label = keypoints_to_yolo_pose(kps, img_w, img_h, K=K)
            if label is None:
                continue

            out_lbl = output_dir / "labels" / split_name / f"{stem}.txt"
            with open(out_lbl, "w") as f:
                f.write(label + "\n")

            if split_name == "train":
                train_kp_total += n_kp
            else:
                val_kp_total += n_kp

    # 写 dataset.yaml
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(f"""# 货架关键点检测 — 原型数据集 (12 张标注图)
path: {output_dir.resolve()}
train: images/train
val: images/val

# 关键点配置
kpt_shape: [{K}, 3]  # K 个关键点, (x, y, visibility)
flip_idx: [{', '.join(str(i) for i in range(K))}]

# 类别
names:
  0: shelf
nc: 1
""")

    print(f"\n  Keypoint distribution: {dict(sorted(kp_dist.items()))}")
    print(f"  Total keypoints: train={train_kp_total}, val={val_kp_total}")
    print(f"  Dataset saved: {output_dir}")
    print(f"  YAML: {yaml_path}")

    return yaml_path


def train(args):
    """训练 YOLOv8-pose。"""
    from ultralytics import YOLO

    # 1. 准备数据集
    print("=" * 60)
    print("[Step 1] Loading prototype annotations...")
    annotations = load_prototype_annotations(
        args.prototype_annotations,
        args.prototype_data_dir,
    )
    print(f"  Loaded {len(annotations)} annotated prototypes")

    if len(annotations) < 2:
        print("ERROR: Need at least 2 annotated prototypes")
        sys.exit(1)

    print(f"\n[Step 2] Preparing YOLO dataset...")
    yaml_path = prepare_proto_dataset(
        annotations,
        args.output_dataset,
        K=args.k,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # 2. 训练
    print(f"\n{'=' * 60}")
    print(f"[Step 3] Training YOLOv8-pose...")
    print(f"  Model:    {args.model}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Batch:    {args.batch}")
    print(f"  LR:       {args.lr0}")
    print(f"  Freeze:   {args.freeze} layers")
    print(f"  K:        {args.k}")
    print(f"  Mosaic:   {args.mosaic}")

    model = YOLO(args.model)

    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        lr0=args.lr0,
        lrf=args.lrf,
        freeze=args.freeze,
        device=args.device,
        name=args.name,
        project=args.project,
        patience=args.patience,
        workers=args.workers,
        # 小数据集优化
        nbs=64,
        warmup_epochs=max(1, args.epochs // 20),
        cos_lr=True,
        close_mosaic=0 if args.mosaic else 0,
        # 强数据增强
        hsv_h=0.015 if args.mosaic else 0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=10.0,       # ±10° 旋转
        translate=0.1,       # 10% 平移
        scale=0.5,           # 0.5~1.5 缩放
        shear=2.0,
        perspective=0.0005,
        fliplr=0.0,          # 货架不对称, 不翻转
        mosaic=1.0 if args.mosaic else 0.0,
        mixup=0.0,
        copy_paste=0.0,
        # 关键点 loss 权重
        kobj=1.0,
        # 数据检查
        plots=True,
        save=True,
        exist_ok=True,
    )

    # 3. 输出结果
    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"Best model: {best_pt}")
    print(f"\nTo run inference:")
    print(f"  python tools/infer_shelf_anchor.py --model {best_pt}")
    print(f"\nTo expand dataset (pseudo-labeling):")
    print(f"  python tools/shelf_proto_pseudo_label.py --model {best_pt}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="YOLO-Pose 训练 (基于 12 张标注原型 + 数据增强)")

    # 数据路径
    parser.add_argument("--prototype_annotations",
                        default=str(REPO_ROOT / "data/new_sheef/prototype_annotations"))
    parser.add_argument("--prototype_data_dir",
                        default=str(REPO_ROOT / "data/new_sheef/prototypes"))
    parser.add_argument("--output_dataset",
                        default="datasets/shelf_pose_proto")
    parser.add_argument("--k", type=int, default=4,
                        help="最多关键点数 (2 或 4)")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="验证集比例 (12 张图建议 1-2 张做验证)")
    parser.add_argument("--seed", type=int, default=42)

    # 模型
    parser.add_argument("--model", default="yolov8n-pose.pt",
                        help="预训练 pose 模型 (n/s/m/l)")
    parser.add_argument("--device", default="0",
                        help="GPU 设备 (或 'cpu')")

    # 训练超参
    parser.add_argument("--epochs", type=int, default=200,
                        help="训练轮数 (小数据集需要多轮)")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size (12 张图建议 2-4)")
    parser.add_argument("--lr0", type=float, default=0.0005,
                        help="初始学习率 (小数据集建议低 LR)")
    parser.add_argument("--lrf", type=float, default=0.01,
                        help="最终 LR 因子 (lr0 * lrf = min_lr)")
    parser.add_argument("--freeze", type=int, default=10,
                        help="冻结前 N 层 (0=全训练, 10=锁 backbone)")
    parser.add_argument("--patience", type=int, default=30,
                        help="早停耐心")
    parser.add_argument("--workers", type=int, default=2)

    # 增强
    parser.add_argument("--mosaic", type=float, default=0.5,
                        help="Mosaic 增强概率 (0=关闭)")

    # 输出
    parser.add_argument("--name", default="shelf_pose_proto",
                        help="实验名称")
    parser.add_argument("--project", default="output/shelf_pose_train",
                        help="输出项目目录")

    # 跳过训练 (仅准备数据)
    parser.add_argument("--prepare_only", action="store_true",
                        help="仅准备数据集, 不训练")

    args = parser.parse_args()

    # 加载 + 准备
    print("Loading prototype annotations...")
    annotations = load_prototype_annotations(
        args.prototype_annotations,
        args.prototype_data_dir,
    )
    print(f"Loaded {len(annotations)} annotated prototypes")

    for stem, ann in annotations.items():
        n_kp = len(ann["keypoints"])
        labels = [kp.get("label", "") for kp in ann["keypoints"]]
        print(f"  {stem}: {n_kp} kp — {labels}")

    if args.prepare_only:
        yaml_path = prepare_proto_dataset(
            annotations, args.output_dataset,
            K=args.k, val_ratio=args.val_ratio, seed=args.seed)
        print(f"\nDataset ready. To train:")
        print(f"  python tools/shelf_proto_pose_train.py")
        return

    # 训练
    train(args)


if __name__ == "__main__":
    main()
