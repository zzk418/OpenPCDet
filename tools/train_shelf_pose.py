#!/usr/bin/env python3
"""
Train YOLOv8-pose on shelf keypoint annotations.

Quick fine-tune: freezes backbone, trains only keypoint head.
Suitable for small datasets (~100 frames).

Usage:
  # Quick train (MVP — 50 epochs, small model)
  python tools/train_shelf_pose.py

  # Full train
  python tools/train_shelf_pose.py --epochs 100 --model yolov8s-pose.pt

  # Continue from a checkpoint (new run dir, explicit hyperparams)
  python tools/train_shelf_pose.py --ckpt output/shelf_pose_train/xxx/weights/last.pt \
      --data datasets/shelf_pose_reviewed_aug_night/dataset.yaml --name xxx_c2 \
      --epochs 40 --lr0 0.0002 --seed 0 --deterministic --exist_ok \
      --set kobj=5.0 cls=1.0 box=5.0 pose=15.0 dfl=1.5 auto_augment=randaugment

  # Standard ultralytics resume (all config from the run dir's args.yaml)
  python tools/train_shelf_pose.py --resume --ckpt output/shelf_pose_train/xxx/weights/last.pt

  # Named continue-training preset (原 resume_shelf_pose.py 配置表, 已合并到本脚本)
  python tools/train_shelf_pose.py --experiment night_c2
  python tools/train_shelf_pose.py --experiment list
"""

import argparse
import sys
from pathlib import Path

from ultralytics.utils import SETTINGS

# 内存级启用 wandb, 不改全局 settings.json (与历史续训脚本一致)
SETTINGS["wandb"] = True

from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO / "output/shelf_pose_train"

# 续训预设实验表 (合并自 resume_shelf_pose.py): 每轮差异仅在 起点权重/数据集/输出目录/epochs/lr0/patience/plots。
# 用 `--experiment <名称>` 复现, `--experiment list` 打印全部。
EXPERIMENTS = {
    "v6_s_direct_aug_c2": dict(
        ckpt="output/shelf_pose_train/shelf_v6_s_direct_aug/weights/best.pt",
        data="datasets/shelf_pose_reviewed_aug/dataset.yaml",
        name="shelf_v6_s_direct_aug_c2",
        epochs=76, lr0=0.001, lrf=0.01, warmup_epochs=3, close_mosaic=10,
        patience=15, plots=True,
    ),
    "night_c1": dict(
        ckpt="output/shelf_pose_train/shelf_v6_s_direct_aug_c2/weights/last.pt",
        data="datasets/shelf_pose_reviewed_aug_night/dataset.yaml",
        name="shelf_v6_s_night_c1",
        epochs=40, lr0=0.0005, lrf=0.01, warmup_epochs=2, close_mosaic=8,
        patience=15, plots=True,
    ),
    "night_c2": dict(
        ckpt="output/shelf_pose_train/shelf_v6_s_night_c1/weights/last.pt",
        data="datasets/shelf_pose_reviewed_aug_night/dataset.yaml",
        name="shelf_v6_s_night_c2",
        epochs=40, lr0=0.0002, lrf=0.01, warmup_epochs=2, close_mosaic=8,
        patience=15, plots=True,
    ),
    "e21_replay": dict(
        ckpt="output/shelf_pose_train/shelf_v6_s_direct_aug/weights/best.pt",
        data="datasets/shelf_pose_reviewed_aug/dataset.yaml",
        name="shelf_v6_s_direct_aug_e21_replay",
        epochs=21, lr0=0.001, lrf=0.01, warmup_epochs=3, close_mosaic=10,
        patience=100, plots=False,  # 不早停, 必须跑满 21 复现 e21
    ),
}

# 续训实验共享超参 (全部实验一致)
COMMON = dict(
    batch=8, imgsz=640, cos_lr=True, momentum=0.937, weight_decay=0.0005,
    workers=2, device="0", exist_ok=True, seed=0, deterministic=True,
    # 损失权重
    kobj=5.0, cls=1.0, box=5.0, pose=15.0, dfl=1.5,
    # 增强
    hsv_h=0.01, hsv_s=0.2, hsv_v=0.15, degrees=3.0, translate=0.03,
    scale=0.15, shear=0.5, fliplr=0.0, mosaic=0.3,
    auto_augment="randaugment", erasing=0.4, nbs=64,
)


def run_experiment(name):
    """按预设实验名续训: 加载起点权重, 用 COMMON + 该轮差异超参 train。"""
    cfg = EXPERIMENTS[name]
    ckpt = REPO / cfg["ckpt"]
    data = REPO / cfg["data"]
    if not ckpt.exists():
        print(f"Start checkpoint not found: {ckpt}")
        sys.exit(1)
    if not data.exists():
        print(f"Dataset YAML not found: {data}")
        sys.exit(1)

    print(f"Experiment '{name}': {cfg['ckpt']} -> {cfg['name']}")
    model = YOLO(str(ckpt))
    kwargs = dict(COMMON)
    kwargs.update({k: v for k, v in cfg.items() if k not in ("ckpt", "data", "name")})
    model.train(data=str(data), name=cfg["name"], project=str(OUTPUT_ROOT), **kwargs)
    print("Done.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8-pose on shelf keypoints")
    parser.add_argument("--data", default="datasets/shelf_pose/dataset.yaml")
    parser.add_argument("--model", default="yolov8n-pose.pt",
                        help="Pretrained pose model (n/s/m/l)")
    parser.add_argument("--ckpt", default=None,
                        help="Continue training from a checkpoint (overrides --model)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.001,
                        help="Initial LR (lower for fine-tune)")
    parser.add_argument("--lrf", type=float, default=0.01,
                        help="Final LR factor")
    parser.add_argument("--freeze", type=int, default=10,
                        help="Freeze first N layers (0=none, 10=backbone)")
    parser.add_argument("--device", default=0, type=int, help="GPU device (or 'cpu')")
    parser.add_argument("--name", default="shelf_pose", help="Run name")
    parser.add_argument("--project", default="output/shelf_pose_train")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--close_mosaic", type=int, default=5,
                        help="Disable mosaic augmentation last N epochs")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (None = ultralytics default)")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--exist_ok", action="store_true",
                        help="Overwrite existing output dir")
    parser.add_argument("--no_plots", action="store_true",
                        help="Disable training plots")
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Standard ultralytics resume; requires --ckpt, "
                             "all other config comes from the run dir's args.yaml")
    parser.add_argument("--experiment", default=None,
                        choices=list(EXPERIMENTS) + ["list"],
                        help="Named continue-training preset (merged resume_shelf_pose.py); "
                             "'list' prints all")
    parser.add_argument("--set", dest="set_cfgs", default=None, nargs=argparse.REMAINDER,
                        help="Override hyperparams, e.g. --set kobj=5.0 cls=1.0 "
                             "auto_augment=randaugment")
    return parser.parse_args()


def _coerce_value(v):
    """int -> float -> bool -> str."""
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def parse_set(items):
    """Parse 'key=value' items from --set into a typed dict."""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid --set item (expected key=value): {item!r}")
        key, val = item.split("=", 1)
        out[key.strip()] = _coerce_value(val.strip())
    return out


def build_train_kwargs(args):
    """Assemble model.train() kwargs. Unset optional args are omitted so the
    default run reproduces the original quick-train behavior exactly."""
    kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "freeze": args.freeze,
        "device": args.device,
        "name": args.name,
        "project": args.project,
        "patience": args.patience,
        "workers": args.workers,
        "resume": args.resume,
        # Pose-specific
        "nbs": 64,          # nominal batch size
        "warmup_epochs": args.warmup_epochs,
        "cos_lr": True,
        "close_mosaic": args.close_mosaic,
        # Augmentations (light — small dataset)
        "hsv_h": 0.01,      # minimal HSV aug (shelf colors matter less)
        "hsv_s": 0.3,
        "hsv_v": 0.2,
        "degrees": 5.0,     # small rotation
        "translate": 0.05,
        "scale": 0.3,
        "fliplr": 0.0,      # no horizontal flip (asymmetric shelf)
        # Keypoint specific
        "kobj": 1.0,        # keypoint objectness loss weight
    }
    if args.seed is not None:
        kwargs["seed"] = args.seed
    if args.deterministic:
        kwargs["deterministic"] = True
    if args.exist_ok:
        kwargs["exist_ok"] = True
    if args.momentum is not None:
        kwargs["momentum"] = args.momentum
    if args.weight_decay is not None:
        kwargs["weight_decay"] = args.weight_decay
    if args.no_plots:
        kwargs["plots"] = False
    kwargs.update(parse_set(args.set_cfgs))
    return kwargs


def main():
    args = parse_args()

    # Named continue-training preset (merged resume_shelf_pose.py)
    if args.experiment == "list":
        print("可用实验:")
        for name, cfg in EXPERIMENTS.items():
            print(f"  {name:22s} epochs={cfg['epochs']:3d} lr0={cfg['lr0']} -> {cfg['name']}")
        print("用法: python tools/train_shelf_pose.py --experiment <实验名>")
        return
    if args.experiment:
        run_experiment(args.experiment)
        return

    # Standard resume: ultralytics reads everything from the run dir's args.yaml
    if args.resume:
        if not args.ckpt:
            print("Error: --resume requires --ckpt <run_dir>/weights/last.pt")
            sys.exit(1)
        if not Path(args.ckpt).exists():
            print(f"Checkpoint not found: {args.ckpt}")
            sys.exit(1)
        print(f"Resuming from {args.ckpt}")
        model = YOLO(str(args.ckpt))
        model.train(resume=True)
        print("Done.")
        return

    # Check dataset
    if not Path(args.data).exists():
        print(f"Dataset YAML not found: {args.data}")
        print("Run: python tools/reviewed_to_train.py first")
        sys.exit(1)

    # Load model: continue from checkpoint if given, else pretrained weights
    if args.ckpt:
        if not Path(args.ckpt).exists():
            print(f"Checkpoint not found: {args.ckpt}")
            sys.exit(1)
        print(f"Continuing from {args.ckpt}")
        model = YOLO(str(args.ckpt))
    else:
        print(f"Loading model: {args.model}")
        model = YOLO(args.model)

    kwargs = build_train_kwargs(args)

    print(f"\nTraining config:")
    print(f"  Data:     {args.data}")
    print(f"  Start:    {args.ckpt if args.ckpt else args.model}")
    print(f"  Output:   {args.project}/{args.name}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Batch:    {args.batch}")
    print(f"  Img size: {args.imgsz}")
    print(f"  LR:       {args.lr0} → {args.lr0 * args.lrf:.6f}")
    print(f"  Freeze:   {args.freeze} layers")
    print(f"  Device:   {args.device}")
    print(f"  Patience: {args.patience}")

    results = model.train(**kwargs)

    # Save best model path
    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nTraining complete!")
    print(f"Best model: {best_pt}")
    print(f"\nTo run inference:")
    print(f"  python tools/infer_shelf_anchor.py --model {best_pt}")

    return results


if __name__ == "__main__":
    main()
