# int8 校准优化 + 混合精度调度实验报告 (2026-08-21)

**模型**: `shelf_v6_s_night_c2` YOLOv8s-pose (2 关键点) → RK3588 NPU
**工具**: rknn-toolkit2 2.3.2, PC 模拟器评估 (val 21 张带 GT, 标定用 train 集, 严格分离)

---

## TL;DR

| 问题 | 结论 |
|------|------|
| 真实 vs 增强数据校准? | **真实数据** (shelf_pose_reviewed/images/train)。增强数据 (aug/aug_night) 反会轻微变差 |
| 用多少张? | **16~64 张足够**; 16 张关键点误差最优, 32~64 张稳健性更足。**不要**用全量/上百张 |
| 关键一步 | `rknn.build(..., auto_hybrid=True)` 必须显式传参, 光设 config 的 `auto_hybrid_cos_thresh` 无效 |
| 混合精度阈值 | `auto_hybrid_cos_thresh=0.96~0.98` (保护 cos<阈值 的敏感层回 fp16), 越低越差 |
| 交付 | `shelf_v6_s_night_c2_int8_hybrid.rknn` (n=16 真实标定 + auto_hybrid 0.98) |

**最终指标 (val 21 张 / 边界用例):**

| 指标 | fp32 | 旧 PTQ int8 | QAT-KD v2 | **本方案 int8 hybrid** |
|------|------|------------|-----------|----------------------|
| conf 中位 | 0.918 | 0.82~0.84 | ~0.91 | **0.918** (= fp32) |
| 关键点误差 (px) | 1.5 | ~4 | ~4 | **2.7** |
| val 检出率 | 21/21 | 21/21 | 21/21 | **21/21** |
| test_multi 弱目标 | 检出 | 弱丢失 | 弱丢失 (0.178) | **检出 (0.446)** |
| test_neg 误检 | 0 | 0 | 0 | **0** |

---

## 1. "崩溃到 0" 的根因与复现

历史上 int8 崩到 0 有两层原因:

1. **conf 通道被 per-tensor int8 压死** (结构问题): `output0 = [box(0~800) | conf(0~1) | kpt]` 混在
   一个张量, int8 按整体 max 定 scale, 0~1 的 conf 被舍入成 0 → dets=0。
   已有两种手术: `onnx_scale_conf.py` (conf×256) 和 `onnx_split_output.py` (三输出拆分, 更干净)。
2. **标定预处理与部署不一致** (本次实验核心修正): 旧标定直接传原始图给 rknn 会被拉伸 resize,
   与部署 letterbox 不一致 → 激活范围标定偏。修正后纯 PTQ 已恢复到 conf 0.88~0.91。

本次所有 int8 都用 **split 结构** (三输出各走自己的 per-tensor scale) + **letterbox 标定**,
在 PC 模拟器上已无"崩溃到 0"。若真机仍偶发崩溃, 多为 NPU 溢出/数值差异, auto_hybrid
(修精度也修溢出) 正是针对性解法。

## 2. 标定实验: 真实 vs 增强, 样本数

标定源对比 (split 结构, normal/layer, n=100):

| 标定源 | conf 中位 | 关键点误差 |
|--------|:---------:|:----------:|
| 真实 (reviewed/train) | **0.885** | **3.3px** |
| 增强 (reviewed_aug_night) | 0.874 | 3.8px |
| 增强 (reviewed_aug) | 0.867 | 3.8px |

样本数扫描 (真实数据, split, normal/layer):

| n | conf 中位 | 关键点误差 |
|:-:|:---------:|:----------:|
| 16 | 0.903 | **2.9px** |
| 32 | 0.904 | 4.1px |
| 64 | **0.911** | 3.6px |
| 100 | 0.885 | 3.3px |
| 125 | 0.885 | 3.5px |

**结论**: 场景同质 (同一相机同一角度) 时, 少量代表性真实图比全量大样本更准 —— 样本越多
激活范围被撑得越宽, 量化 scale 越粗。真实数据 > 增强数据; 增强虽覆盖广, 但引入了与部署
分布不符的统计 (合成夜间、翻转等), 把范围撑大反而有害。

## 3. 逐层误差分析 (调度依据)

`accuracy_analysis` (5 张 val 图) 显示最差层集中在:

- **检测头 `model.22`**: `cv3.*/cv3.*.2/Conv` (box 回归) single_cos 0.945~0.984, `dfl/Softmax` 0.989,
  `cv4.1` (kpt 回归) 0.992
- **backbone CSP 层**: `model.4/5/6/8/9/15` 的 conv/act

这些层正是 int8 损失的主要来源 → 应调度回 fp16。

## 4. 混合精度: 误差大的层调度回 fp16

**关键 API 坑**: 光在 `rknn.config(auto_hybrid_cos_thresh=...)` 设阈值**无效**, 必须同时给
`rknn.build(..., auto_hybrid=True)` 传参才启用混合精度。历史 `convert_int8_split_ch_hyb90.log`
正是只设了阈值没开 build 开关 → 当时的 hybrid 实验全部静默失效。

阈值扫描 (split, 真实 n=64, `auto_hybrid=True`):

| cos_thresh | conf 中位 | 关键点误差 |
|:----------:|:---------:|:----------:|
| 0.98 | **0.918** | 3.4px |
| 0.96 | **0.918** | 3.4px |
| 0.94 | 0.907 | 3.6px |
| 0.92 | 0.911 | 3.8px |
| 0.90 | 0.911 | 3.7px |

**语义**: cos 低于阈值的层保留 fp16。阈值越高保护的层越多越准; 0.96~0.98 最优
(把 cos 0.945~0.99 的敏感层全保护住), 降到 0.94 以下头层重新被 int8 → 变差。

调度后仅 **14/243 层 (5.8%)** 为 fp16 (backbone CSP + head cv3.0), 全网络最差 single_cos
从 0.945 提升到 0.988, 几乎所有层 ≥0.998 ≈ fp32 水平。代价: 文件 13.9→23MB (fp16 权重
2 字节), NPU 上 fp16 层算力减半, 建议板端实测 FPS。

## 5. 最终交付

`tools/exp_rknn.py` 一键复现 (标定 → build → 同会话评估 → 导出):

```bash
conda run -n rknn python tools/exp_rknn.py \
    --onnx output/shelf_pose_train/shelf_v6_s_night_c2/weights/best_split.onnx \
    --calib-dir datasets/shelf_pose_reviewed/images/train --n-calib 16 \
    --out output/rk3588_deploy/shelf_v6_s_night_c2_int8_hybrid.rknn \
    --algo normal --method layer --auto-hybrid --hybrid 0.98 \
    --cases output/rk3588_deploy_pre/imgs --analyze
```

产物:
- **`output/rk3588_deploy/shelf_v6_s_night_c2_int8_hybrid.rknn`** (n=16 真实 + auto_hybrid 0.98)
- 边界用例: 10 张真实 val 全检出 conf 0.90~0.918; `test.jpg/day/night` 各 1 个 conf 0.918;
  `test_multi` **2 个 [0.864, 0.471]** (弱目标恢复, 历史 QAT 只有 0.178 丢失); `test_neg` 0 误检
- 备注: `TV_250000034919` 上 n=16 版多出一个 0.777 的框 —— 经核对是**同一货架的重复框**
  (fp32 本身在该帧就输出 6+ 个几乎重叠的框, int8 NMS 未完全合并), 非独立目标误检, 部署取
  top-1 无影响。若需消除可调低 NMS iou 或改用 n=32 标定。
- 逐层分析: `output/rk3588_deploy/analysis/shelf_v6_s_night_c2_int8_hybrid.rknn/error_analysis.txt`

## 6. 建议下一步

1. **板端验证**: 拷 rknn 到 RK3588 实测 FPS 与数值 (PC 模拟器偏乐观, 尤其溢出类问题)。
   若 FPS 不达标, 阈值降到 0.96 (同指标, fp16 层更少)。
2. **弱目标硬需求**: 本方案 test_multi 弱目标 0.446 已过 0.25 阈值, 若需更强可微调 conf 阈值。
3. **标定源维护**: 采集新场景后, 保持用该场景真实图 16~32 张标定; 避免用合成增强图。
4. **保留的探索方向**: `--algo mmse --method channel` 与 `quantized_dtype=w8a16` 未在本轮
   充分测试, 若真机 hybrid 仍不够可作后续对比。

## 复现工具 (本轮新增)

| 脚本 | 作用 |
|------|------|
| `tools/exp_rknn.py` | 标定→build→同会话 val 评估→边界用例→逐层分析→导出 一体化 |
| `tools/eval_rknn_gt.py` | 对已导出 rknn/onnx 做 val GT 评估 (注: load_rknn 不能 PC sim, 主流程用 exp_rknn) |
