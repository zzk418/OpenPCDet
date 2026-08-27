# RK3588 货架关键点部署包 (最小推理版)

YOLOv8s-pose 货架关键点模型 **FGD 特征蒸馏版** (`shelf_v6_s_fgd_qat_int8.rknn`) 的 RK3588 NPU 端部署包。
模型已转 int8 `.rknn`, **拷贝到板子装上 rknn-toolkit-lite2 即可直接运行**。

> 当前模型: 2026-08-21 FGD 特征蒸馏版 (yolov8x-pose 教师做特征 KD + 夜间域 100 张标定 int8)。
> 强目标 conf **0.91~0.97** (int8 与 fp32 差 ≤0.02, 全场最高), test_neg 无误检, 关键点像素差 ≤6px。

## 目录结构

```
rk3588_deploy/                        # 最小推理包 (2026-08-21)
├── shelf_v6_s_fgd_qat_int8.rknn    # NPU 模型 (13.9MB, int8, RK3588)  ★当前生产模型 (md5 bbf33174)
├── shelf_viz.py                    # 可视化 + PCD 深度查表共用模块 (与 PC 端 infer_shelf_anchor 同款)
├── infer_image.py                  # 离线推理: 图片/目录 → 关键点 + PCD 深度 3D 锚点 + 可视化
├── infer_camera.py                 # 实时推理: 摄像头 → 关键点 + FPS (无 PCD, 2D)
├── imgs/                           # 10 张真实常规测试帧 + 同名 .pcd (深度查表用)
├── librknnrt.so                    # 配套 NPU 运行时 (7.4MB, 对应 rknn-toolkit-lite2 2.3.2)
├── requirements-board.txt          # 板端依赖清单 (rknn-toolkit-lite2 + numpy + opencv)
├── wheels_aarch64/                 # 板端离线 wheelhouse (aarch64, 无网安装用)
├── install_offline.sh              # 无网一键安装脚本 (headless opencv)
└── README.md                       # 本文档
```

> 📦 精简说明: 其余产物 (PTQ 备份、fp16、kl/split/qat_v3 实验模型, 标定集 `calib*/`,
> 转换日志 `convert*.log`, dev 脚本 `_verify_x86.py`, 5 张边界用例测试图) 统一放
> `output/rk3588_deploy_pre/`。

## 模型输出说明 (三输出拆分)

- 输入: 640x640 RGB, letterbox 等比缩放 + 114 填充, uint8 (mean/std 由 RKNN runtime 内部归一化)
- 输出: **3 个张量** (onnx_split_output.py 手术, conf 独占张量后不再需要 ×256):
  `output_box[1,4,8400]` + `output_conf[1,1,8400]` + `output_kpts[1,6,8400]`
- 解码端 (`infer_image.py` / `infer_camera.py` 的 `merge_outputs`) **按 shape 识别**合并回
  `[1,11,8400]` = `[cx,cy,w,h(letterbox像素) | cls_conf(0~1) | kpt1_x,y,c | kpt2_x,y,c] × 8400`
- conf 为原生 0~1, 不再有 ×256 手术; 关键点置信度通道是无信息量 logit, 解码统一置 1.0
- ⚠️ 推理时把整个 `outputs` 列表传给 `decode_yolopose`, **不要只传 `outputs[0]`**
  (只传 box 会 `IndexError: index 4 out of bounds for axis 1 with size 4`)

## 自带测试图 (10 张真实常规)

`imgs/` 内 10 张白天单货架常规帧 (来自 `datasets/shelf_pose_reviewed/images/val`, **非训练集**),
每张预期 dets=1, int8 全部正确检出且无误检:

| 图 (源 TV 帧) | fp32 | int8 (FGD) |
|----------------|:---:|:---:|
| `TV_250000012137.jpg` | 0.930 | 0.960 |
| `TV_250000017001.jpg` | 0.911 | 0.950 |
| `TV_250000020068.jpg` | 0.896 | 0.920 |
| `TV_250000027387.jpg` | 0.950 | 0.950 |
| `TV_250000034919.jpg` | 0.899 | 0.920 |
| `TV_250000043933.jpg` | 0.935 | 0.960 |
| `TV_250000053956.jpg` | 0.921 | 0.920 |
| `TV_250000057567.jpg` | 0.890 | 0.910 |
| `TV_250000105882.jpg` | 0.940 | 0.950 |
| `TV_250000106104.jpg` | 0.940 | 0.950 |

> 5 张边界用例 (test_day/test_night/test_multi/test_neg/test.jpg, 含 0/1/2 目标、夜间、负样本)
> 在 `output/rk3588_deploy_pre/imgs/`, 需要板端回归时拷入: `cp ../rk3588_deploy_pre/imgs/test*.jpg imgs/`

## 已知限制

- **test_multi 弱目标 (右侧弱货架) 丢失**: 特征 KD 在 fp32 就把弱目标压到阈值下
  (与 QAT v2 相同取舍, 用户接受)。弱目标检出是硬需求时回退
  `output/rk3588_deploy_pre/shelf_v6_s_night_c2_fp32_nightcalib_split.rknn` (弱目标 conf 0.39, 慢一倍)。
- conf 数值敏感 (用于置信度门控) → 用 fp16: `output/rk3588_deploy_pre/shelf_v6_s_night_c2_fp16.rknn`。

## 板端 NPU 运行时

`librknnrt.so` (NPU C 运行时) 必须和 python 绑定 `rknn-toolkit-lite2` **同版本** (本包 2.3.2)。
板子系统自带的老/残缺版会报错, 典型:

```
AttributeError: /usr/lib/librknnrt.so: undefined symbol: rknn_set_core_mask
```

修复 (板子上, 先备份可回滚):

```bash
cp /usr/lib/librknnrt.so /usr/lib/librknnrt.so.bak
cp ~/rk3588_deploy/librknnrt.so /usr/lib/librknnrt.so
bash infer_int8.sh
```

## 板端部署步骤 (RK3588, Ubuntu, python3.10)

```bash
# 1. 拷贝部署包到板子
scp -r rk3588_deploy/ user@<板子IP>:~/

# 2. 装依赖
ssh user@<板子IP>
cd ~/rk3588_deploy
pip install -r requirements-board.txt
bash install_offline.sh              # 无网环境用离线 wheelhouse

# 3. 离线推理 (无参数 = 推理 imgs/ 全部 10 张, 带 PCD 深度查表)
python infer_image.py
python infer_image.py --input imgs/TV_250000012137.jpg   # 单张
python infer_image.py --input ./imgs/                    # 结果在 results/ + detections.json

# 4. 摄像头实时推理
python infer_camera.py --camera 0
python infer_camera.py --no-display --json               # 无显示, 每帧一行 JSON 到 stdout
```

`detections.json` 示例 (图旁有同名 `.pcd` 时 `anchor_3d` 为深度查表结果):

```json
{"TV_250000012137.jpg": [{
  "box": [169.9, 162.0, 416.9, 198.0], "conf": 0.96,
  "keypoints": [[191.5, 164.9, 1.0], [397.4, 194.0, 1.0]],
  "anchor_3d": [[-414.6, -340.0, 1481.3], [342.5, -237.9, 1497.2]]
}]}
```

## 蓝芯 MRDVS 相机 SDK (货架识别用 RGB 流)

相机不接 `/dev/videoN` 而是走 MRDVS SDK 时用下面两个脚本 (`MRDVS_linux/` 是官网 CameraSDK 包):

```bash
# 0. 装 SDK (板子 aarch64): 装到 /opt/MRDVS/lib + 写 LD_LIBRARY_PATH, 再装 python 绑定
cd ~/rk3588_deploy/MRDVS_linux
sudo bash install.sh
pip install Sample/python/lx_camera_py-1.3.3-py3-none-any.whl

# 1. 读出厂内参 + 验证相机连通 (输出 camera_intrinsics_<sn>.json)
python query_camera_intrinsics.py --ip 192.168.100.86
python query_camera_intrinsics.py            # 自动发现第一台

# 2. SDK 实时推理: 与离线推理同一管线 (letterbox+decode_yolopose), 结果一致
python infer_camera_sdk.py --ip 192.168.100.86
python infer_camera_sdk.py --ip 192.168.100.86 --json --no-display   # 逐帧 JSON 给上位机
python infer_camera_sdk.py --ip 192.168.100.86 --depth3d   # 开深度流, 内参按 SN 自动读 camera_intrinsics_<sn>.json (同离线)
```

- `infer_camera_sdk.py` 复用 `infer_camera.py` 的 `letterbox/decode_yolopose`, 保证实时帧与离线结果一致;
  默认只开 2D RGB 流, `--depth3d` 才开 3D 深度流。`--depth3d` 自动开启 SDK 内置 RGBD 对齐
  (`LX_INT_RGBD_ALIGN_MODE=DEPTH_TO_RGB` + 强制帧同步): 点云与 RGB 同分辨率同索引,
  关键点像素直接查 `points[v,u]`, 缺失像素局部窗口邻域采样取中值; 对齐未生效
  (点云分辨率 ≠ RGB 分辨率) 时自动回退内参投影。
- `query_camera_intrinsics.py` 的 `find_sdk_lib` 会找 `/opt/MRDVS/lib` (install.sh 位置)、
  `linux_aarch64/` 等, 板子上无需手动 `--dll`。产物 `camera_intrinsics_<sn>.json` 放回
  部署包同目录即可: `shelf_viz.py` 导入时自动读它的 `depth_intrinsic`, `infer_camera_sdk.py`
  按 SN 自动匹配, 无需手工改代码。
- SDK 的 RGB 帧是 OpenCV BGR 顺序 (官方示例直接 imshow), 若画面偏蓝/偏橙加 `--color rgb`。

## 常用参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `--model` | .rknn 路径 | 同目录 `shelf_v6_s_fgd_qat_int8.rknn` |
| `--conf` | 置信度阈值 | 0.25 |
| `--iou` | NMS IoU 阈值 | 0.45 |
| `--cores` | NPU 核 bitmask: 1=单核 3=双核(0+1) 7=三核(0+1+2) 0=自动。**注意 3 是双核不是三核** | image 7 / camera 3 |
| `--pcd` | (image) PCD 目录, 默认图所在目录按同名 `{stem}.pcd` 匹配 | 图同目录 |
| `--camera` | 摄像头编号 `/dev/videoN` | 0 |
| `--json` | (camera) 每帧打印检测 JSON | 关 |
| `--no-display` | (camera) 无弹窗纯推理 | 关 |
| `--save` | (camera) 录制 mp4 | 关 |

## 换模型重转 (PC 端)

```bash
conda activate rknn
python tools/export_rknn.py \
    --onnx output/shelf_pose_train/shelf_v6_s_fgd_qat/weights/best_split.onnx \
    --out output/rk3588_deploy/shelf_v6_s_fgd_qat_int8.rknn \
    --quant int8 --algo normal --method layer --n-calib 100 \
    --calib-img-dir datasets/shelf_pose_reviewed_aug_night
```

标定图预处理必须和部署 letterbox 完全一致 (export_rknn.py 内部处理)。历史转换日志/实验记录在
`output/rk3588_deploy_pre/` (README 含 int8 conf 精度上限、QAT/FGD 蒸馏说明)。

## FAQ

**Q: x86 电脑上能跑这个 .rknn 吗?**
A: 不能。`.rknn` 只能在板端 RKNNLite 运行; PC 端 rknn-toolkit2 只用于转换/仿真。
x86 模拟验证用 `output/rk3588_deploy_pre/_verify_x86.py` (需 rknn env, 现场 build)。

**Q: 摄像头没画面?**
A: 检查 `--camera` 编号 (`ls /dev/video*`), USB 相机权限 `sudo chmod 666 /dev/video0`。

**Q: `--cores 3` 是三核吗? 双核/三核差别多大?**
A: 不是 —— **`--cores 3` 是双核** (bitmask 0b011 = 核 0+1), 三核是 `--cores 7` (0b111)。
`infer_image.py` 已默认三核 (7); `infer_camera.py` 默认双核 (3)。实测 int8 单模型小负载时
核间同步开销抵消并行收益, `--cores 3` 和 `--cores 7` 差别很小; 摄像头持续运行默认双核还能
省功耗/发热。若想三核, `infer_camera.py --cores 7`。

**Q: 装错 wheel 了?**
A: 板子装 **rknn-toolkit-lite2** (aarch64); PC 端 rknn-toolkit2 (x86_64) 是转换工具,
装到板子报 `not a supported wheel on this platform`。版本必须与 `librknnrt.so` 配套 (2.3.2)。
