# RK3588 货架关键点部署包 (最小推理版)

YOLOv8s-pose 货架关键点模型 **fp16 版** (`shelf_mobilenet_r3_fp16.rknn`) 的 RK3588 NPU 端部署包。
模型已转 fp16 `.rknn`, **拷贝到板子装上 rknn-toolkit-lite2 即可直接运行**。

> 当前生产模型: `shelf_mobilenet_r3_fp16.rknn` (fp16, MobileNetV3-Large-pose, 光照鲁棒 r3 权重)。
> fp16 无 int8 量化损失, conf 与 fp32 基本一致 (适合置信度门控), test_neg 无误检。
> 切换原因: int8 (FGD 蒸馏版) 强目标 conf 虽高但弱目标丢失、且 conf 数值不可靠 → 弃用 (用户决定)。

## 两行命令部署 (推荐)

```bash
# 行1 (PC): 把最小部署包拷到板子
scp -r output/rk3588_deploy/ root@<板子IP>:/root/

# 行2 (PC): 一键部署 (装依赖 → 装相机SDK → 注册自启服务 → 冒烟验证)
ssh root@<板子IP> "cd /root/rk3588_deploy && sudo bash deploy.sh"
```

- `deploy.sh` 依次: ① 离线装 python 依赖 (wheelhouse, 无网) ② 装 MRDVS 相机 SDK (已装跳过)
  ③ `install_shelf_service.sh` 注册开机自启 ④ 跑一张自带测试图冒烟验证。
- 换相机 IP: `CAMERA_IP=192.168.2.x` (默认 `192.168.2.151`): `ssh root@<板IP> "cd /root/rk3588_deploy && CAMERA_IP=192.168.2.x sudo bash deploy.sh"`
- 板子 python3 需与 `wheels_aarch64/` 匹配 (cp38); 版本不符时 [1/4] 步会直接报错。
- 旧包残留想彻底清掉, 先 `ssh root@<板IP> "rm -rf /root/rk3588_deploy"` 再执行上面两行。

## 通信架构: 车控/PLC 直连 (取代 Program_3)

原链路是两层转发: 车控/PLC(Modbus TCP **客户端**) ⇄ Program_3(Modbus TCP **Server** :30000)
⇄ JSON(:5511) ⇄ 本服务。现在去掉 Program_3, `shelf_pos_service.py` **自己就是
Modbus TCP Server**, 车控/PLC 按原协议直连读写 —— 连接角色 / 端口 / 寄存器
0~19 布局 / 语义与 Program_3 的 NEWMODBUS 完全一致:

- **0~9 车控可写**: `reg1` 拍照命令 (0=无 3=货架单次 4=货架连续; 1/2/5/6 托盘/平台
  不支持 → 异常), `reg2` 写 1 清故障码, `reg0` 叉车心跳
- **10~19 视觉回 (车控只读)**: `reg10` 视觉心跳(1s 翻转) `reg11` 拍照回令
  `reg12` 工作状态 (0空闲/1拍照中/2完成/3异常) `reg13` 故障码
  (照 Program_3 = `40 - error_code`: **41**=无货架/推理失败,
  **42**=不支持的拍照命令; 车写 reg2=1 清 0)
  `reg14~17` 位姿 X·Y·偏航·Z (×10, 0.1mm/0.1°, 可负=int16) `reg18` 板温
- **闭环**: 车控写 reg1=3 → 服务拍一帧推理中心点 → 换算补偿位姿写 reg14~17 +
  reg12=2(完成) → 车控读到位姿后写 reg1=0 → 回空闲并清 reg11, 才可触发下一次
- 位姿换算参考/钳位在 `plc_pose_ref.json`, **现场标定只改它, 不动代码**。
  默认 **ref = {x:-1567, z:-286} = 以『完美取货位』为基准的偏移 (delta)**: 完美中心
  在相机帧 (0,-286,1567)mm 时 reg14~17 全 0, 偏离多少就发多少 (0.1mm)。
  Program_3 那套角点 ref y1=635/y2=765/z=120 会给中心点硬加几百 mm 假偏移, 已弃。
  换算公式与现场标定/安全步骤见下面「位姿换算 · 现场标定」节

### 运行 / PLC 通信联测

```bash
# 板子常驻 (systemd 自启): 车控连 <板子IP>:30000 (照 Program_3 连法)
python3 shelf_pos_service.py

# 本机全链路回归 (不连相机/模型/RKNN): 起 --from-file 服务
python3 shelf_pos_service.py --from-file \
    --center-file center_xyz.json --plc-port 31000

# 另开终端当"车控"测 (心跳/单次拍照+位姿换算/不支持命令故障/应答复位):
python3 sim_plc_client.py --host 127.0.0.1 --port 31000 \
    --expect-xyz 0,0,1500.0          # 已知中心点 → 精确校验 reg14~17
# 一键: 自动拉 --from-file 服务 + 跑全部用例
python3 sim_plc_client.py --local-e2e
# 对板子/现场
python3 sim_plc_client.py --host 192.168.2.102 --port 30000
```

### 位姿换算 · 现场标定与安全

寄存器 = 以**完美取货位**为基准的偏移 ×10 (0.1mm / 0.1°, 可负=int16)。换算照
Program_3 的 `reg = ref + 测量` (refs 是**加性基线**, Program_3 里它填的是角点
`posReference_*`), 我们把 ref 填成"完美点各轴测量取反", 使回归到完美位姿时
reg14~17 全 0 —— 传出去的就是**相对完美位置的 delta**:

```
完美中心 (相机帧): x 横向=0, y 竖直=-286, z 前向=1567 (mm)
   ↳ ref = {x:-1567, y1:0, y2:0, z:-286, xita:0}

reg14 X前向 =  refX×10      + round(z×10)   = (z-1567)×10   # 前向偏移
reg15 Y横向 = (x<0:+refY1 | x>0:-refY2)×10 + round(x×10)
                                          = x×10            # 横向偏移 (refY1=Y2=0)
reg17 Z竖直 = (-refZ)×10 - round(-y×10)     = 2860 + y×10   # 在 y=-286 归 0
reg16 偏航  =  refXita×10 + 0                              # 中心点无偏航, 恒 0
```

- **residual 标定 (现场必做)**: 车停准"取货位"、货架摆正, 触发看 reg14~17 解码剩的
  **固定残差** (应为 0; 不为 0 = 相机/叉齿安装偏置), 把残差填回对应轴 ref (如
  前向恒差 -40mm → reg14 读出 +400 → ref.x 调 -40), 重触发应读出 0。ref 只需测一次,
  thr 一般不用动。
- **静态对拍 (上实测前)**: 完美位姿触发 → reg14~17 应全 0; 再手动把货架左/右/升/降
  已知量 → 对应轴应读出该量 (0.1mm)。若 reg14 前向实际应发**绝对距离**而非偏移,
  把 `ref.x` 改 0 即可 (会回到 z≈1.5m → reg14≈15670, 别和 thr 打架)。
- **符号**: reg14/reg15/reg17 正负和实际移动方向相反时, 改 `shelf_pos_service.py`
  顶部 `SIGN_X`/`SIGN_Z` (或对应轴 ref 正负), 别乱猜, 用静态对拍定。
- **安全阈值 thr**: 饱和钳位, 防指令离谱。日志 `reg14~17=[...]` 若**顶到 ±thr**
  说明读数超量程 → 别上实测, 先核对量程/寄存器语义。

**上实测前的安全流程 (静态→低速)**:
1. `sim_plc_client.py --local-e2e` 全绿 (协议层已稳)。
2. 板端 `shelf_pos_service.py --self-test`, 相机前有货架 → 打印 XYZ, 与卷尺量
   的方向/量级一致。
3. 车停准, 货架**人为偏一侧已知距离** (如左移 50mm), 触发 → reg15 应朝对应方向
   变 ~500 (0.1mm), 且没顶 thr。
4. 观察日志: `[plc] 拍照完成 → xyz=[...]mm reg14~17=[...]` 数值合理、reg13=0。
5. 货架/车身**错开量减到 ~0** 复测 (回归, 确认残差已清)。
6. 以上都对了, 才做**低速小偏移**的整机验证; 头几次留人在急停旁。

## 目录结构

```
rk3588_deploy/                        # 最小生产部署包 (2026-08-31)
├── shelf_mobilenet_r3_fp16.rknn   # NPU 模型 (15.5MB, fp16, RK3588)  ★当前生产模型
├── shelf_pos_service.py            # 生产服务 = Modbus TCP Server, 车控/PLC 直连 (systemd 自启)
├── plc_modbus.py                   # 纯 socket Modbus TCP Server (服务 import, 零第三方依赖)
├── plc_pose_ref.json               # 位姿换算参考/钳位 (照 Program_3 HuoJia, 现场只改它)
├── sim_plc_client.py              # 联测: 扮演车控测 PLC 通信 (心跳/单次/故障/连续, 本地可全链路)
├── infer_camera_sdk.py             # SDK 相机实时推理 (生产服务用)
├── infer_camera.py                 # letterbox/decode 核心 (被服务 import)
├── infer_image.py                  # 离线推理: 图片→关键点 + PCD 深度 3D 锚点
├── shelf_viz.py                    # 可视化 + PCD 深度查表共用模块
├── query_camera_intrinsics.py      # 相机内参读取 (被服务 import)
├── camera_intrinsics_E0BB6585B5893591.json  # 出厂内参 (按 SN 自动读)
├── imgs/                           # 10 张自带测试帧 + 同名 .pcd (冒烟/自检)
├── librknnrt.so                    # 配套 NPU 运行时 (7.4MB, rknn-toolkit-lite2 2.3.2)
├── MRDVS_linux/                    # 蓝芯相机 SDK 安装包 (install.sh → /opt/MRDVS)
├── wheels_aarch64/                 # 板端离线 wheelhouse (aarch64, 无网安装用)
├── requirements-board.txt          # 板端依赖清单 (rknn-toolkit-lite2 + numpy + opencv)
├── install_offline.sh              # 无网一键安装脚本 (headless opencv)
├── install_shelf_service.sh        # 注册开机自启服务
├── deploy.sh                       # ★ 一键部署 (PC 两行命令的第 2 行)
└── README.md                       # 本文档
```

> 📦 退役清单 (统一在 `output/rk3588_deploy.bak/retired_20260831/`): int8 模型
> (`shelf_v6_s_fgd_qat_int8.rknn` / `int8_hybrid.rknn`)、`infer_int8.sh`、
> dev 测试脚本 (`test_rgbd_align.py`)、静态 `shelf_pos.service`
> (服务文件由 `install_shelf_service.sh` 按实际目录动态生成)、实验报告
> (`int8_calib_hybrid_report.md`)、完整 imgs 测试集 (`imgs_all/`)、推理产物
> (`results*`)、`log/`。
> 更早的实验产物 (PTQ 备份、kl/split/qat_v3 模型、标定集、转换日志、`_verify_x86.py`、
> 5 张边界用例测试图) 在 `output/rk3588_deploy_pre/`。

## 模型输出说明 (解码自动兼容)

- 输入: 640x640 RGB, letterbox 等比缩放 + 114 填充, uint8 (mean/std 由 RKNN runtime 内部归一化)
- 解码端 (`infer_image.py` / `infer_camera.py`) 自动兼容两种输出格式:
  - 单输出旧模型: 直接透传 `[1,11,8400]`
  - 多输出拆分版 (box[1,4,8400]+conf[1,1,8400]+kpts[1,6,8400]): `merge_outputs` 按 shape 识别
    合并回 `[1,11,8400]` = `[cx,cy,w,h(letterbox像素) | cls_conf(0~1) | kpt1_x,y,c | kpt2_x,y,c] × 8400`
- conf 自动识别: 生产 fp16 / 拆分版 conf 原生 0~1; 旧手术版 ×256 (max>1.5 时自动 /256)
- 关键点置信度通道是无信息量 logit, 解码统一置 1.0
- ⚠️ 推理时把整个 `outputs` 列表传给 `decode_yolopose`, **不要只传 `outputs[0]`**
  (只传 box 会 `IndexError: index 4 out of bounds for axis 1 with size 4`)

> 5 张边界用例 (test_day/test_night/test_multi/test_neg/test.jpg, 含 0/1/2 目标、夜间、负样本)
> 在 `output/rk3588_deploy_pre/imgs/`, 需要板端回归时拷入: `cp ../rk3588_deploy_pre/imgs/test*.jpg imgs/`

## 已知限制

- **fp16 慢于 int8** (RK3588 约 3TOPS vs 6TOPS)。生产选 fp16 换 conf 可靠 (无量化损失,
  适合置信度门控); 若板端负载吃紧需切回 int8, 可用
  `output/rk3588_deploy.bak/retired_20260831/shelf_v6_s_fgd_qat_int8.rknn`, 但弱目标会丢。
- test_multi 弱目标 (右侧弱货架): int8 (FGD) 会压掉该目标, fp16 保持 fp32 精度可正常检出。

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
bash infer_fp16.sh
```

## 板端部署步骤 (RK3588, Ubuntu, python3.10)

> 最快: 用上面的**两行命令** (scp + deploy.sh)。下面手动一步步的步骤保留作参考/排查。

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
#    python 绑定离线装: --no-deps 不拉它声明的 opencv-python (cv2 已由 headless 提供, 装真 opencv-python 会冲突)
cd ~/rk3588_deploy/MRDVS_linux
sudo bash install.sh
pip install --no-index --no-deps Sample/python/lx_camera_py-1.3.3-py3-none-any.whl

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
