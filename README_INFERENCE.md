# PV-RCNN 推理和可视化指南

## 项目概述

本项目演示了如何使用 PV-RCNN 模型进行 3D 目标检测推理，并将结果可视化。

## 快速开始

### 1. 生成演示数据

```bash
# 生成 PLY 格式的点云数据
python download_sample_ply.py
```

这将在 `demo_data/` 目录中生成：
- `sample_000.ply` - PLY 格式的点云数据
- `sample_000.bin` - KITTI 格式的二进制点云数据
- `sample_000.npy` - NumPy 格式的点云数据

### 2. 运行推理和可视化

```bash
# 使用简化的推理脚本
python simple_inference.py
```

或者使用完整的 OpenPCDet 推理脚本（需要预训练模型）：

```bash
python inference_and_visualize.py \
    --cfg_file tools/cfgs/kitti_models/pv_rcnn.yaml \
    --data_path demo_data \
    --ext .ply \
    --output_dir output
```

### 3. 查看结果

推理结果将保存在 `output/` 目录中：

```bash
ls -lh output/
```

## 输出文件说明

### 可视化图像

生成的 PNG 图像包含三个子图：

1. **俯视图 (Top View)** - XY 平面
   - 显示点云的俯视投影
   - 红色矩形表示检测到的 3D 边界框
   - 标签显示类别和置信度

2. **侧视图 (Side View)** - XZ 平面
   - 显示点云的侧视投影
   - 红色矩形表示检测框的侧视投影

3. **3D 视图 (3D View)**
   - 显示点云的 3D 视图
   - 红色立方体表示检测到的 3D 边界框

## 数据格式

### PLY 文件格式

```
ply
format ascii 1.0
element vertex <num_points>
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
<点云数据>
```

### 点云坐标系

- **X 轴**: 指向前方
- **Y 轴**: 指向左方
- **Z 轴**: 指向上方
- **Z 轴原点**: 距离地面约 1.6m（KITTI 数据集标准）

## 检测结果格式

### 边界框表示

每个检测框由 7 个参数表示：

```
[x, y, z, dx, dy, dz, heading]
```

其中：
- `(x, y, z)` - 边界框中心坐标
- `(dx, dy, dz)` - 边界框尺寸（长、宽、高）
- `heading` - 旋转角度（弧度）

### 类别标签

- `0` - Car（汽车）
- `1` - Pedestrian（行人）
- `2` - Cyclist（骑行者）

## 脚本说明

### `download_sample_ply.py`

生成合成的点云数据，包含：
- 5000 个背景点（地面和周围环境）
- 3000 个汽车点（分布在 3 个位置）

### `simple_inference.py`

简化的推理脚本，不依赖完整的 OpenPCDet 导入：
- 读取 PLY 文件
- 生成虚拟检测结果
- 可视化并保存为 PNG 图像

### `inference_and_visualize.py`

完整的推理脚本，支持：
- 加载预训练的 PV-RCNN 模型
- 进行真实的 3D 目标检测推理
- 可视化检测结果

## 环境要求

### 必需的包

```bash
pip install numpy matplotlib
```

### 可选的包

```bash
# 用于完整的 OpenPCDet 推理
pip install torch torchvision

# 用于 3D 可视化
pip install open3d
```

## 常见问题

### Q: 如何使用自己的点云数据？

A: 将点云数据转换为 PLY 格式，放在 `demo_data/` 目录中，然后运行推理脚本。

PLY 文件格式示例：
```
ply
format ascii 1.0
element vertex 100
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
0.0 0.0 0.0 128 128 128
1.0 1.0 1.0 255 255 255
...
```

### Q: 如何使用预训练模型？

A: 下载预训练模型（例如 `pv_rcnn_8369.pth`），然后运行：

```bash
python inference_and_visualize.py \
    --cfg_file tools/cfgs/kitti_models/pv_rcnn.yaml \
    --data_path demo_data \
    --ckpt pv_rcnn_8369.pth \
    --output_dir output
```

### Q: 如何修改可视化参数？

A: 编辑 `simple_inference.py` 中的以下参数：

```python
# 置信度阈值
if pred_scores[i] > 0.3:  # 修改这个值

# 图像大小
fig = plt.figure(figsize=(15, 5))  # 修改这个值

# 输出分辨率
plt.savefig(str(output_file), dpi=150)  # 修改 dpi 值
```

## 参考资源

- [OpenPCDet GitHub](https://github.com/open-mmlab/OpenPCDet)
- [PV-RCNN 论文](https://arxiv.org/abs/1912.13192)
- [KITTI 数据集](http://www.cvlibs.net/datasets/kitti/)
- [PLY 文件格式](http://paulbourke.net/dataformats/ply/)

## 许可证

本项目遵循 OpenPCDet 的许可证。

## 更新日期

- 创建日期: 2026-07-13
- 最后更新: 2026-07-13