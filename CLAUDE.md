# OpenPCDet — 仓储 3D 目标检测

> 笔记: `F:\kie_note\项目\仓库运输语义理解规划` (WSL: `/kie_note/项目/仓库运输语义理解规划`)  
> 图片: 统一放 `assets/` 子目录

## 环境

```bash
conda activate pc
export PYTHONPATH=/code/OpenPCDet:$PYTHONPATH
```

## 数据集

`data/warehouse/` — 5 类仓储实体: `['箱子', '电动运输车', '货运自行车', '无人搬运车', '叉车']`

```
points/*.npy       # (N,5) [x,y,z,intensity,timestamp]
labels/*.txt       # x y z dx dy dz heading class_name
ImageSets/train.txt, val.txt   # 2098/525
```

数据增强: X/Y 随机翻转、±45° 全局旋转、缩放 0.95~1.05

## 模型

| 模型 | 配置文件 | 最优 Epoch | 3D mAP | 权重 |
|------|---------|:---------:|:------:|------|
| CenterPoint++ | `centerpoint_pp_warehouse.yaml` | 30 | **46.98%** | `output/best_ckpts/centerpoint_pp_epoch30_mAP46.98.pth` |
| CenterPoint | `centerpoint_warehouse.yaml` | 40 | 35.30% | `output/best_ckpts/centerpoint_epoch40_mAP35.30.pth` |

Epoch 30 之后 CenterPoint++ 叉车 AP 从 61.9% → 0%，明显过拟合。

## 常用命令

```bash
# 训练
cd tools
python train.py --cfg_file cfgs/custom_models/centerpoint_pp_warehouse.yaml \
    --batch_size 4 --epochs 50 --extra_tag default

# 推理 (全量验证集, 默认无文字标签)
python centerpoint_warehouse_inference.py

# 推理 (自定义权重)
python centerpoint_warehouse_inference.py \
    --cfg_file tools/cfgs/custom_models/centerpoint_pp_warehouse.yaml \
    --ckpt output/best_ckpts/centerpoint_pp_epoch30_mAP46.98.pth \
    --output_dir output/warehouse_inference_viz

# 带标签版本 (论文图示)
python centerpoint_warehouse_inference.py --show_labels --max_samples 1 --output_dir output/best_sample
```

## 可视化规格

- 左图 BEV 俯视图 + 右图 3D 视图
- 点云: 纯黑色, 高 alpha=1.0, BEV `s=3.0`, 3D `s=1.0`
- GT: 深灰 #444444 虚线框, X 标记中心
- Pred: 实线框 + 填充, 中心点 ● (BEV) / ★ (3D), 边缘黑色描边
- 裁剪: GT+Pred 框紧凑裁剪, padding=4m
- 标签: 英文 (Box / ELF / CargoBike / FTS / ForkLift), 默认关闭, `--show_labels` 开启
- 输出 `prediction_centers.json` 记录所有检测框中心点坐标

### YOLO-Pose 货架关键点可视化 (v4 风格)

用于 `infer_shelf_anchor.py` / `continue_train_v5.py` 推理输出。

- 关键点: 红色圆点 (BGR=0,0,255), 半径 4px, 细十字准线 ±6px, 黑色轮廓线宽 1px
- 编号: P1/P2 标在关键点右上角, 红色小字
- 连线: 两关键点之间红色虚线连接 (虚线段长 ~8px)
- 中心点: 黄色 **X** 标记 (BGR=0,255,255), 粗 2px, 大小 ±5px
- 图例面板: 左上角深灰半透明 (BGR=40,40,40, alpha=0.75), 外边框灰 (100,100,100)
  - 标题: "Shelf Corners (N pts)" 金色 (BGR=80,200,255)
  - 逐点: `P1  X 1234  Y -567  Z 2100` — 红色标签 + 白色坐标
  - 中心: `C X 1234  Y -567  Z 2100` — 黄色标签 + 浅黄坐标 (BGR=255,255,150)
  - 分隔线隔开点列表和中心行
- 底部信息: 右下角灰色 "YOLO-Pose | N corners"
- 输出 `prediction_centers.json` 记录所有检测关键点 3D 坐标 + 中心点 XYZ

## 货架关键点标注

### 数据

- **原型点云与图片**: `data/new_sheef/prototypes/` (12 帧, `clusterN_TV_*.pcd` + `.jpg`)
- **标注输出**: `data/new_sheef/prototype_annotations/` (每帧 `{stem}_anchor_v2.json`)

### Web 标注工具

```bash
python tools/shelf_anchor_v3_web.py
# 打开 http://localhost:5000
# 左键点击 → PCD 查表添加关键点
# ENTER → 保存并下一帧
```

JSON 格式: `{stem, num_keypoints, keypoints: [{id, pixel_uv, anchor_3d, median_depth, label}]}`

## 评估

KITTI 官方 eval 协议: 3D IoU → TP/FP/FN 匹配 → 41 点插值 PR 曲线 → 11-point AP

## 文档

- `work/基于无锚框 3D 目标检测的仓储机器人实时语义感知1.md`

## Git 规范

- **提交规范**: conventional commit (`feat`/`fix`/`chore`/`refactor`), 按功能逻辑分组, 禁止一次性大备份提交
- **分支**: WIP/重构走 feature 分支, master 保持干净; 重写历史/force-push 需用户明确确认
- **提交内容**: 提交前 `git status` + `git diff --cached` 核对实际内容; 改 .gitignore 规则后检查是否误伤同名文件 (曾因 `**build**` 误忽略 `build_night_dataset.py`)
