#!/usr/bin/env python3
"""
货架识别 yolov8s-pose 续训统一入口（合并自 continue_train_night/_c2/v6_s、resume_train_v6_s、replay_train_v6_s_e21）。

所有续训轮次的差异仅在: 起点权重 / 数据集 / 输出目录 / epochs / lr0 / patience / plots。
用配置表描述每一轮, 命令行 `python tools/resume_shelf_pose.py <实验名>` 即可复现。

实验记录:
  v6_s_direct_aug_c2  从 shelf_v6_s_direct_aug/best.pt (epoch24) 出发, 补齐 100 epochs (再训 76), 原超参
  night_c1            从 c2 last.pt (epoch25, precision 最优) 出发, 夜间 1:1 合成数据, lr0=0.0005 防白天回退
  night_c2            从 night_c1 last.pt (epoch4, 上次 40ep 计划被中断) 出发, lr0=0.0002, 再训 40ep → **最终部署模型**
  e21_replay          重放 c2 前 21 个 epoch 复现 epoch21 权重 (best.pt 按 fitness 落在 e20, e21 未落盘)

注: 原 resume_train_v6_s.py (标准 ultralytics resume=True) 已废弃 —— 原 run 的 last.pt
    被 strip_optimizer 清掉优化器状态, 无法标准 resume, 改用 v6_s_direct_aug_c2 方案。
"""
import argparse
import sys
from pathlib import Path

from ultralytics.utils import SETTINGS

REPO = Path("/code/OpenPCDet")
OUTPUT_ROOT = REPO / "output/shelf_pose_train"

# 每轮实验配置 (名称 -> 差异参数)
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

# 共享超参 (全部实验一致)
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


def main():
    parser = argparse.ArgumentParser(description="货架识别 yolov8s-pose 续训统一入口")
    parser.add_argument("experiment", nargs="?", choices=list(EXPERIMENTS) + ["list"],
                        default="list", help="实验名, 默认 list 打印全部")
    args = parser.parse_args()

    if args.experiment == "list":
        print("可用实验:")
        for name, cfg in EXPERIMENTS.items():
            print(f"  {name:22s} epochs={cfg['epochs']:3d} lr0={cfg['lr0']} -> {cfg['name']}")
        print("用法: python tools/resume_shelf_pose.py <实验名>")
        return

    cfg = EXPERIMENTS[args.experiment]
    ckpt = REPO / cfg["ckpt"]
    data = REPO / cfg["data"]
    if not ckpt.exists():
        print(f"start checkpoint not found: {ckpt}")
        sys.exit(1)
    if not data.exists():
        print(f"dataset yaml not found: {data}")
        sys.exit(1)

    # 内存级启用 wandb (不改全局 settings.json)
    SETTINGS["wandb"] = True
    from ultralytics import YOLO

    model = YOLO(str(ckpt))
    kwargs = dict(COMMON)
    kwargs.update({k: v for k, v in cfg.items() if k not in ("ckpt", "data", "name")})
    model.train(
        data=str(data),
        name=cfg["name"],
        project=str(OUTPUT_ROOT),
        **kwargs,
    )
    print("Done.")


if __name__ == "__main__":
    main()
