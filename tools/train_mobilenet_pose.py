#!/usr/bin/env python3
"""训练「MobileNetV3-Large 骨干 + 轻量 YOLO-pose 头」货架双锚点检测器。

与现网 YOLOv8-pose 的差异只在 backbone: 完整 mobilenet_v3_large(ImageNet
预训练, torchvision) 替换 YOLO 骨干; neck 按 mobilenet 宽度瘦身; 检测/关键点
解码仍是标准 ultralytics Pose (nc=1, kpt_shape=[2,3]) → onnx 输出保持
[1,11,8400], 板端 fp16 部署链与之前完全一致 (只换 .rknn 文件)。

数据: 教师伪标签 base (shelf_pose_teacher_v2, 690/121) 的离线轻增档
      (_light3 ≈4x, mixup=0), val 原样透传。
训练: 直接 fp16 (amp=True), 超参与 teacher 配方一致 (reviewed_to_train.train_yolo),
      整模型重训 (freeze=0), 在线增强从轻 (离线已覆盖几何, 在线只留轻 HSV/翻转 0)。

用法:
  python tools/train_mobilenet_pose.py                       # 默认 (yaml + light3)
  python tools/train_mobilenet_pose.py --epochs 120 --name shelf_mobilenet_light3
  python tools/train_mobilenet_pose.py --yaml tools/yolo_pose_mobilenetv3.yaml
"""
import argparse
from pathlib import Path

from ultralytics.utils import SETTINGS

SETTINGS["wandb"] = True  # 内存级启用, 不改全局 settings.json

from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent
PROJECT = REPO / "output/shelf_pose_train"

# teacher 配方 (reviewed_to_train.train_yolo) 原样保留: 只换 data/model, 其余不动
HP = dict(
    batch=8, imgsz=640,
    lr0=0.001, lrf=0.01, freeze=0, nbs=64,
    warmup_epochs=3, cos_lr=True, close_mosaic=10,
    # 损失权重 (查准优先, K=2)
    kobj=5.0, cls=1.0, box=5.0, pose=15.0,
    # 在线增强: 离线 light3 已覆盖几何/HSV, 在线保留轻档 (fliplr=0 货架不对称)
    hsv_h=0.01, hsv_s=0.2, hsv_v=0.15,
    degrees=3.0, translate=0.03, scale=0.15, shear=0.5,
    fliplr=0.0, mosaic=0.3,
    amp=True,  # fp16 混合精度直接训
    plots=True, save=True, exist_ok=True,
    workers=2, device="0",
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="tools/yolo_pose_mobilenetv3.yaml",
                    help="模型架构 yaml (TorchVision mobilenet_v3_large + Pose 头)")
    ap.add_argument("--data", default="datasets/shelf_pose_teacher_v2_light3/dataset.yaml",
                    help="训练数据集 yaml")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--name", default="shelf_mobilenet_light3")
    ap.add_argument("--weights", default="",
                    help="续训起点 (如 best_ep70.pt); 给定时从该权重微调, 否则从 yaml+ImageNet 训")
    ap.add_argument("--lr0", type=float, default=None,
                    help="覆盖默认 lr0 (微调建议 3e-4 量级)")
    return ap.parse_args()


def main():
    args = parse_args()
    yaml_path = REPO / args.yaml
    data_path = REPO / args.data
    for p in (yaml_path, data_path):
        if not p.exists():
            raise SystemExit(f"找不到: {p}")

    model = YOLO(str(yaml_path) if not args.weights else args.weights)
    mode = f"微调 <- {args.weights}" if args.weights else f"从头(ImageNet 预训练骨干) <- {yaml_path}"

    # 训练前体检: 参数量 / stride / 是否 mobilenet 骨干已带 ImageNet 权重
    pm = model.model
    print(f"\n架构: {yaml_path}")
    print(f"参数量: {sum(p.numel() for p in pm.parameters()) / 1e6:.2f}M")
    print(f"stride: {pm.stride.tolist()}")
    tv_layers = [m for m in pm.modules() if type(m).__name__ == "TorchVision"]
    print(f"TorchVision(骨干) 层: {len(tv_layers)}; "
          f"预训练权重要求: {getattr(tv_layers[0], 'weights', None) if tv_layers else 'N/A'}")

    hp = dict(HP)
    hp.update(epochs=args.epochs, patience=args.patience,
              data=str(data_path), name=args.name, project=str(PROJECT))
    if args.lr0 is not None:
        hp["lr0"] = args.lr0

    print(f"\n训练 MobileNetV3-Large pose @ fp16(amp) — {hp['epochs']}ep, {hp['batch']}b")
    print(f"  模式:   {mode}")
    print(f"  Data:   {data_path}")
    print(f"  lr0:    {hp['lr0']}")
    print(f"  Output: {PROJECT / args.name}")
    model.train(**hp)

    best = PROJECT / args.name / "weights/best.pt"
    print(f"\nDone! Best: {best}")
    return best


if __name__ == "__main__":
    main()
