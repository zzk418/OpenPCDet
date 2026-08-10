#!/usr/bin/env python3
"""用已有 YOLO-Pose 模型在合并数据集上微调。
用法: python tools/finetune_merged.py
"""
import sys
from pathlib import Path
from ultralytics import YOLO

# 已有模型 + 合并数据集
BEST_PT = "/code/ultralytics/runs/pose/output/shelf_pose_train/shelf_corners_v1/weights/best.pt"
DATA = "datasets/shelf_pose_merged/dataset.yaml"

print(f"Loading model from {BEST_PT}")
model = YOLO(BEST_PT)

print(f"Fine-tuning on {DATA}")
results = model.train(
    data=DATA,
    epochs=80,
    batch=4,
    imgsz=640,
    lr0=0.0003,          # 较低 LR (fine-tune)
    lrf=0.01,
    freeze=0,             # 全模型微调 (数据集变了 K=3→K=4)
    device="0",
    name="shelf_merged_v1",
    project="output/shelf_pose_train",
    patience=20,
    workers=2,
    # 关键点
    kobj=1.0,
    # 增强
    hsv_h=0.01,
    hsv_s=0.4,
    hsv_v=0.3,
    degrees=10.0,
    translate=0.1,
    scale=0.4,
    shear=2.0,
    fliplr=0.0,
    mosaic=0.8,
    nbs=64,
    warmup_epochs=2,
    cos_lr=True,
    close_mosaic=5,
    plots=True,
    save=True,
    exist_ok=True,
)

best = Path("output/shelf_pose_train/shelf_merged_v1/weights/best.pt")
print(f"\nDone! Model: {best}")
print(f"python tools/infer_shelf_anchor.py --model {best}")
