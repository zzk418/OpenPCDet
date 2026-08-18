#!/usr/bin/env python3
"""
审核完成的伪标注 → YOLO-pose 训练集 → 离线增强 → 训练

用法:
  python tools/reviewed_to_train.py                     # 全流程
  python tools/reviewed_to_train.py --skip_augment      # 跳过增强(直接训120帧)
  python tools/reviewed_to_train.py --skip_train         # 只生成数据集不训练
  python tools/reviewed_to_train.py --skip_build --model yolov8x-pose.pt \
      --train_name shelf_v6_x_teacher_mixup             # 直接训已有增强数据集
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

REPO = Path("/code/OpenPCDet")
REVIEWED_DIR = REPO / "datasets/shelf_pose_pseudo/reviewed"
FULL_DATA = REPO / "data/new_sheef/pngs"


def load_reviewed(reviewed_dir):
    """加载审核结果 → [{stem, keypoints: [(u,v),...]}]"""
    samples = []
    for f in sorted(Path(reviewed_dir).glob("*_anchor_v2.json")):
        d = json.load(open(f))
        if d.get("skipped"):
            continue
        kps_raw = d.get("keypoints", [])
        kps = [(kp["pixel_uv"][0], kp["pixel_uv"][1]) for kp in kps_raw]
        samples.append({"stem": d["stem"], "keypoints": kps})
    return samples


def keypoints_to_yolo(kps, img_w=640, img_h=480, K=2):
    """关键点 → YOLO pose 格式字符串。"""
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


def build_yolo_dataset(samples, output_dir, val_ratio=0.15, seed=42):
    """审核结果 → YOLO 格式数据集。"""
    random.seed(seed)
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (output_dir / sub).mkdir(parents=True)

    img_w, img_h, K = 640, 480, 2

    # 复制图像 + 写标签
    valid = []
    for s in tqdm(samples, desc="Building dataset"):
        stem = s["stem"]
        # 找图像
        img_path = None
        for ext in [".jpg", ".png"]:
            cand = FULL_DATA / f"{stem}{ext}"
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            print(f"  SKIP {stem}: no image")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  SKIP {stem}: cannot read")
            continue
        if img.shape[:2] != (img_h, img_w):
            img = cv2.resize(img, (img_w, img_h))

        label = keypoints_to_yolo(s["keypoints"], img_w, img_h, K)
        if label is None:
            print(f"  SKIP {stem}: no valid kp")
            continue

        valid.append({"stem": stem, "img": img, "label": label})

    random.shuffle(valid)
    n_val = max(1, int(len(valid) * val_ratio))

    for split, items in [("train", valid[n_val:]), ("val", valid[:n_val])]:
        kp_total = 0
        for item in items:
            cv2.imwrite(str(output_dir / "images" / split / f"{item['stem']}.jpg"),
                        item["img"], [cv2.IMWRITE_JPEG_QUALITY, 90])
            with open(output_dir / "labels" / split / f"{item['stem']}.txt", "w") as f:
                f.write(item["label"] + "\n")
            kp_total += item["label"].count("2.000000")
        print(f"  [{split}] {len(items)} samples, {kp_total} kp")

    # dataset.yaml
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(f"""# 货架关键点检测 — 审核后数据集
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
    print(f"  Train: {len(valid) - n_val}, Val: {n_val}")
    return yaml_path


def augment_dataset(input_dir, output_dir, aug_per_img=40, mixup_prob=0.3, seed=42):
    """离线数据增强：强尺度 + 仿射 + HSV + 模糊 + mixup (仅 train)。"""
    random.seed(seed)
    np.random.seed(seed)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (output_dir / sub).mkdir(parents=True)

    img_w, img_h, K = 640, 480, 2

    for split in ["train", "val"]:
        img_dir = input_dir / "images" / split
        lbl_dir = input_dir / "labels" / split
        if not img_dir.exists():
            continue

        # 基础样本: (img, kp_objs, stem), kp_objs 为对象列表 (mixup 会产生多对象)
        base_samples = []
        for img_path in sorted(img_dir.glob("*.jpg")):
            stem = img_path.stem
            lbl_path = lbl_dir / f"{stem}.txt"
            if not lbl_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            kp_objs = []
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.split()
                kps = [(float(parts[5 + i * 3]) * img_w, float(parts[5 + i * 3 + 1]) * img_h)
                       for i in range(K) if float(parts[5 + i * 3 + 2]) > 0]
                if kps:
                    kp_objs.append(kps)
            if not kp_objs:
                continue
            base_samples.append((img, kp_objs, stem))

        all_samples = []  # (img, kp_objs, name)
        for img, kp_objs, stem in base_samples:
            # 原图
            all_samples.append((img.copy(), [k[:] for k in kp_objs], f"{stem}_orig"))

            # 几何/颜色增强
            for i in range(aug_per_img):
                aug_img, aug_kps = augment_single(img, kp_objs[0], img_w, img_h)
                valid_kps = [(u, v) for (u, v) in aug_kps if 0 <= u < img_w and 0 <= v < img_h]
                if len(valid_kps) >= 1:
                    all_samples.append((aug_img, [valid_kps], f"{stem}_aug{i:03d}"))

                # mixup (仅 train): 与随机另一张样本的增强结果混合, 标签双对象
                if split == "train" and random.random() < mixup_prob:
                    p_img, p_kp_objs, _ = random.choice(base_samples)
                    p_aug, p_aug_kps = augment_single(p_img, p_kp_objs[0], img_w, img_h)
                    lam = np.random.uniform(0.35, 0.65)
                    m_img = np.clip(lam * aug_img.astype(np.float32) +
                                    (1 - lam) * p_aug.astype(np.float32),
                                    0, 255).astype(np.uint8)
                    m_objs = []
                    if len(valid_kps) >= 1:
                        m_objs.append(valid_kps)
                    p_valid = [(u, v) for (u, v) in p_aug_kps
                               if 0 <= u < img_w and 0 <= v < img_h]
                    if len(p_valid) >= 1:
                        m_objs.append(p_valid)
                    if m_objs:
                        all_samples.append((m_img, m_objs, f"{stem}_mix{i:03d}"))

        print(f"  [{split}] {len(all_samples)} samples (orig + aug + mixup)")

        for a_img, kp_objs, a_name in all_samples:
            lines = [keypoints_to_yolo(obj, img_w, img_h, K) for obj in kp_objs]
            lines = [l for l in lines if l]
            if not lines:
                continue
            cv2.imwrite(str(output_dir / "images" / split / f"{a_name}.jpg"),
                        a_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with open(output_dir / "labels" / split / f"{a_name}.txt", "w") as f:
                f.write("\n".join(lines) + "\n")

    # dataset.yaml
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(f"""# 货架关键点检测 — 审核+增强数据集
path: {output_dir.resolve()}
train: images/train
val: images/val
kpt_shape: [{K}, 3]
flip_idx: [{', '.join(str(i) for i in range(K))}]
names:
  0: shelf
nc: 1
""")

    print(f"\nAugmented dataset: {output_dir}")
    return yaml_path


def augment_single(img, kps, img_w, img_h):
    """对单张图 + 关键点做数据增强 (强尺度 + 仿射 + 颜色/模糊/噪声)。"""
    h, w = img.shape[:2]

    # 随机尺度缩放 (0.5~1.5): 缩小→反射填充, 放大→随机裁剪
    s = np.random.uniform(0.5, 1.5)
    new_w, new_h = int(round(w * s)), int(round(h * s))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    kps = [(u * s, v * s) for (u, v) in kps]
    if new_w <= w and new_h <= h:
        pad_x = (w - new_w) // 2
        pad_y = (h - new_h) // 2
        img = cv2.copyMakeBorder(img, pad_y, h - new_h - pad_y,
                                 pad_x, w - new_w - pad_x, cv2.BORDER_REFLECT101)
        kps = [(u + pad_x, v + pad_y) for (u, v) in kps]
    else:
        crop_x = np.random.randint(0, max(1, new_w - w + 1))
        crop_y = np.random.randint(0, max(1, new_h - h + 1))
        img = img[crop_y:crop_y + h, crop_x:crop_x + w]
        kps = [(u - crop_x, v - crop_y) for (u, v) in kps]

    # 仿射变换 (小幅, 主尺度已由上面覆盖)
    scale = np.random.uniform(0.9, 1.1)
    angle = np.random.uniform(-15, 15)
    tx = np.random.uniform(-0.05, 0.05) * w
    ty = np.random.uniform(-0.05, 0.05) * h
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    aug_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 变换关键点
    aug_kps = []
    for (u, v) in kps:
        pt = np.array([u, v, 1.0])
        new_pt = M @ pt
        aug_kps.append((new_pt[0], new_pt[1]))

    # HSV 抖动
    aug_img = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    aug_img[:, :, 0] = np.clip(aug_img[:, :, 0] + np.random.uniform(-10, 10), 0, 179)
    aug_img[:, :, 1] = np.clip(aug_img[:, :, 1] * np.random.uniform(0.7, 1.3), 0, 255)
    aug_img[:, :, 2] = np.clip(aug_img[:, :, 2] * np.random.uniform(0.7, 1.3), 0, 255)
    aug_img = cv2.cvtColor(aug_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 亮度/对比度
    alpha = np.random.uniform(0.85, 1.15)
    beta = np.random.randint(-15, 15)
    aug_img = np.clip(alpha * aug_img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # 运动模糊 (20% 概率)
    if np.random.random() < 0.2:
        ksize = np.random.choice([3, 5])
        angle_blur = np.random.randint(0, 360)
        kernel = np.zeros((ksize, ksize))
        kernel[int((ksize - 1) / 2), :] = np.ones(ksize)
        kernel = cv2.warpAffine(kernel,
                                cv2.getRotationMatrix2D((ksize / 2, ksize / 2), angle_blur, 1),
                                (ksize, ksize))
        kernel = kernel / kernel.sum()
        aug_img = cv2.filter2D(aug_img, -1, kernel)

    # 高斯噪声 (20% 概率)
    if np.random.random() < 0.2:
        noise = np.random.normal(0, np.random.uniform(5, 20), aug_img.shape)
        aug_img = np.clip(aug_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return aug_img, aug_kps


def train_yolo(data_yaml, output_name="shelf_reviewed_v2", epochs=100, model_weights=None):
    """训练 YOLOv8-pose，查准优先 (K=2)。"""
    # 内存级启用 wandb 记录 (不改全局 settings.json)
    from ultralytics.utils import SETTINGS
    SETTINGS["wandb"] = True
    from ultralytics import YOLO

    if model_weights:
        pretrained = model_weights
    else:
        candidates = [
            "/code/OpenPCDet/.claude/worktrees/shelf-anchor-v2-mvp/yolov8n-pose.pt",
            "/code/OpenPCDet/yolo26n.pt",
            "yolov8n-pose.pt",
        ]
        pretrained = None
        for c in candidates:
            if Path(c).exists():
                pretrained = c
                break
        if pretrained is None:
            pretrained = "yolov8n-pose.pt"

    print(f"\nTraining YOLO-Pose on {data_yaml}")
    print(f"  Pretrained: {pretrained}")
    print(f"  Epochs: {epochs}")
    print(f"  Strategy: precision-first (kobj=5.0, cls=1.0)")

    model = YOLO(pretrained)

    # wandb 平滑记录: 每 25 个 batch 记一次训练损失到 train_smooth/*,
    # x 轴为全局迭代数, 便于观察 loss 下降趋势 (epoch 级 train/* 图表不变)
    try:
        import wandb

        def _log_train_batch_smooth(trainer):
            # 8.4 的 trainer 没有 batch_idx 属性, 用闭包计数器当全局步数;
            # tloss 是 dict {loss名: 滑动均值}, 直接 items()
            _log_train_batch_smooth.step = getattr(_log_train_batch_smooth, "step", 0) + 1
            if _log_train_batch_smooth.step % 25 != 0 or wandb.run is None:
                return
            try:
                vals = {n: float(v) for n, v in getattr(trainer, "tloss", {}).items()}
                if vals:
                    wandb.log({f"train_smooth/{n}": v for n, v in vals.items()},
                              step=_log_train_batch_smooth.step)
            except Exception:
                pass

        model.add_callback("on_train_batch_end", _log_train_batch_smooth)
    except ImportError:
        pass

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=8,
        imgsz=640,
        lr0=0.001,
        lrf=0.01,
        freeze=0,
        device="0",
        name=output_name,
        project=str(REPO / "output/shelf_pose_train"),
        patience=15,
        workers=2,
        kobj=5.0,
        cls=1.0,
        box=5.0,
        pose=15.0,
        hsv_h=0.01,
        hsv_s=0.2,
        hsv_v=0.15,
        degrees=3.0,
        translate=0.03,
        scale=0.15,
        shear=0.5,
        fliplr=0.0,
        mosaic=0.3,
        nbs=64,
        warmup_epochs=3,
        cos_lr=True,
        close_mosaic=10,
        plots=True,
        save=True,
        exist_ok=True,
    )

    best = REPO / "output/shelf_pose_train" / output_name / "weights/best.pt"
    print(f"\nDone! Best model: {best}")
    return best


def main():
    parser = argparse.ArgumentParser(description="审核结果 → YOLO → 增强 → 训练")
    parser.add_argument("--reviewed_dir", default=str(REVIEWED_DIR))
    parser.add_argument("--output_dir", default=str(REPO / "datasets/shelf_pose_reviewed"))
    parser.add_argument("--aug_dir", default=str(REPO / "datasets/shelf_pose_reviewed_aug"))
    parser.add_argument("--aug_per_img", type=int, default=40)
    parser.add_argument("--mixup_prob", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_augment", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_build", action="store_true",
                        help="跳过数据集构建, 直接训已有增强数据集")
    parser.add_argument("--model", default=None,
                        help="预训练权重, 如 yolov8x-pose.pt (默认自动选 yolov8n-pose)")
    parser.add_argument("--train_name", default="shelf_reviewed_v1",
                        help="训练输出文件夹名")
    args = parser.parse_args()

    # 跳过 Step 1/2: 直接训已有增强数据集
    if args.skip_build:
        aug_yaml = Path(args.aug_dir) / "dataset.yaml"
        if not aug_yaml.exists():
            raise SystemExit(f"--skip_build: 找不到 {aug_yaml}")
        print("=" * 60)
        print("Step 3: Train YOLOv8-pose (skip build)")
        print("=" * 60)
        train_yolo(aug_yaml, args.train_name, args.epochs, args.model)
        return

    # Step 1: 审核结果 → YOLO 格式
    print("=" * 60)
    print("Step 1: Convert reviewed → YOLO format")
    print("=" * 60)
    samples = load_reviewed(args.reviewed_dir)
    print(f"Loaded {len(samples)} reviewed frames")

    yolo_dir = args.output_dir
    yaml_path = build_yolo_dataset(samples, yolo_dir, args.val_ratio, args.seed)

    if args.skip_augment and not args.skip_train:
        # 直接训 120 帧
        train_yolo(yaml_path, args.train_name, args.epochs, args.model)
        return

    if args.skip_augment:
        print(f"\nDone. Dataset: {yolo_dir}")
        return

    # Step 2: 离线增强
    print("\n" + "=" * 60)
    print(f"Step 2: Offline augmentation ({args.aug_per_img} per image)")
    print("=" * 60)
    aug_yaml = augment_dataset(yolo_dir, args.aug_dir, args.aug_per_img,
                               args.mixup_prob, args.seed)

    if args.skip_train:
        print(f"\nDone. Augmented dataset: {args.aug_dir}")
        return

    # Step 3: 训练
    print("\n" + "=" * 60)
    print("Step 3: Train YOLOv8-pose")
    print("=" * 60)
    train_yolo(aug_yaml, args.train_name, args.epochs, args.model)


if __name__ == "__main__":
    main()
