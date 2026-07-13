# Claude 环境配置记录

## 环境信息

- **操作系统**: Linux 6.18
- **IDE**: Visual Studio Code
- **默认 Shell**: /bin/bash
- **主目录**: /home/kie
- **当前工作目录**: /code/OpenPCDet

## 环境管理工具

### uv 包管理器

在 PC 环境下使用 `uv` 管理 Python 环境和依赖：

```bash
# 安装 uv
pip install uv

# 创建虚拟环境
uv venv .venv

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows

# 安装依赖
uv pip install -r requirements.txt

# 添加新的依赖
uv pip install package_name

# 冻结依赖
uv pip freeze > requirements.txt
```

## OpenPCDet 项目配置

### 项目结构

```
/code/OpenPCDet/
├── pcdet/              # 主要代码库
├── tools/              # 训练、测试、演示脚本
├── data/               # 数据目录
├── docs/               # 文档
├── requirements.txt    # 依赖列表
└── setup.py           # 项目配置
```

### 依赖安装

```bash
# 安装项目依赖
uv pip install -r requirements.txt

# 安装可视化工具
uv pip install open3d
# 或
uv pip install mayavi
```

### 常用命令

#### 1. 演示推理 (Demo)

```bash
# 使用 PV-RCNN 模型进行推理
python tools/demo.py \
    --cfg_file tools/cfgs/kitti_models/pv_rcnn.yaml \
    --ckpt pv_rcnn_8369.pth \
    --data_path demo_data
```

#### 2. 训练模型

```bash
# 训练 PV-RCNN 模型
python tools/train.py \
    --cfg_file tools/cfgs/kitti_models/pv_rcnn.yaml \
    --batch_size 2 \
    --epochs 80
```

#### 3. 测试模型

```bash
# 测试模型性能
python tools/test.py \
    --cfg_file tools/cfgs/kitti_models/pv_rcnn.yaml \
    --ckpt pv_rcnn_8369.pth
```

## 数据准备

### 创建演示数据

```bash
# 生成合成点云数据
python create_demo_data.py
```

生成的数据格式：
- `.bin` 文件：KITTI 格式的二进制点云数据
- `.npy` 文件：NumPy 格式的点云数据
- 点云格式：(num_points, 4) - [x, y, z, intensity]

### 数据坐标系

- **X 轴**: 指向前方
- **Y 轴**: 指向左方
- **Z 轴**: 指向上方
- **Z 轴原点**: 距离地面约 1.6m（KITTI 数据集标准）

## 模型下载

### PV-RCNN 预训练模型

| 模型 | 大小 | 性能 | 下载链接 |
|------|------|------|--------|
| PV-RCNN (KITTI) | 50M | Car@83.61 | [Google Drive](https://drive.google.com/file/d/1lIOq4Hxr0W3qsX83ilQv0nk1Cls6KAr-/view?usp=sharing) |

## 可视化工具

### Open3D 可视化

```python
from tools.visual_utils import open3d_vis_utils as V

# 可视化点云和检测结果
V.draw_scenes(
    points=points,
    ref_boxes=pred_boxes,
    ref_scores=pred_scores,
    ref_labels=pred_labels
)
```

## 常见问题

### 1. 模块导入错误

```bash
# 确保项目路径正确
export PYTHONPATH=/code/OpenPCDet:$PYTHONPATH
```

### 2. CUDA 相关问题

```bash
# 检查 CUDA 可用性
python -c "import torch; print(torch.cuda.is_available())"

# 如果需要 CPU 模式
# 在配置文件中设置 GPU 相关参数
```

### 3. 依赖版本冲突

```bash
# 使用 uv 创建干净的虚拟环境
uv venv --clear .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 开发工作流

### 1. 环境设置

```bash
# 克隆项目
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet

# 创建虚拟环境
uv venv .venv
source .venv/bin/activate

# 安装依赖
uv pip install -r requirements.txt
uv pip install open3d
```

### 2. 快速测试

```bash
# 生成演示数据
python create_demo_data.py

# 运行推理演示
python tools/demo.py \
    --cfg_file tools/cfgs/kitti_models/pv_rcnn.yaml \
    --data_path demo_data
```

### 3. 自定义开发

```bash
# 修改配置文件
# 编辑 tools/cfgs/kitti_models/pv_rcnn.yaml

# 修改模型代码
# 编辑 pcdet/models/detectors/pv_rcnn.py

# 运行测试
python tools/test.py --cfg_file tools/cfgs/kitti_models/pv_rcnn.yaml
```

## 参考资源

- [OpenPCDet GitHub](https://github.com/open-mmlab/OpenPCDet)
- [PV-RCNN 论文](https://arxiv.org/abs/1912.13192)
- [KITTI 数据集](http://www.cvlibs.net/datasets/kitti/)
- [uv 文档](https://docs.astral.sh/uv/)

## 更新日期

- 创建日期: 2026-07-13
- 最后更新: 2026-07-13