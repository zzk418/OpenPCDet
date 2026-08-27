#!/usr/bin/env python3
"""ONNX 手术: 把 output0 里 0~1 的 cls 置信度通道放大 256 倍, 使 int8 量化不压死它。

背景: output0 [1,11,8400] = [box(0~800) | cls_conf(0~1) | kpt(0~800 + conf 0~1)]
      int8 输出量化按整体 max(~800) 定 scale, 0~1 的 cls conf 全被舍入成 0 (dets=0)。
修复: 在最终 Concat 前只给 cls conf 通道乘 256 (0~256 与 box 同级, 且不超过坐标范围,
      不触发 RKNN 编译器 REGTASK 字段溢出); 解码时再 /256 还原。坐标通道不动。

为什么不缩放 kpt conf (曾 ×256, 现已去掉):
  - kpt 的 c1/c2 通道实为未饱和 logits (可达 ~74), ×256 会把输出范围撑到 ~19000,
    触发 RKNN 编译器 REGTASK "bit width exceeds limit" (8192 > 8191) → 板端
    Concat_5 NPU "failed to submit" → 推理变垃圾 + 极慢 (0.2 FPS)。
  - 且 kpt conf 在 fp32 中本就饱和≈1.0, 无信息量; 解码端检测到全 0 会自动置 1.0。

用法: conda run -n pc python tools/onnx_scale_conf.py [in.onnx] [out.onnx]
"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper

SRC = sys.argv[1] if len(sys.argv) > 1 else "output/shelf_pose_train/shelf_v6_s_night_c2/weights/best.onnx"
DST = sys.argv[2] if len(sys.argv) > 2 else "output/shelf_pose_train/shelf_v6_s_night_c2/weights/best_conf_scaled.onnx"
SCALE = 256.0

m = onnx.load(SRC)
g = m.graph

# 找到产生 output0 的 Concat
concat = None
for n in g.node:
    if n.op_type == "Concat" and "output0" in n.output:
        concat = n
        break
assert concat is not None, "找不到 output0 Concat"
print(f"final Concat: {concat.name}  inputs={list(concat.input)}")

concat_idx = list(g.node).index(concat)
new_inputs = []
def add_node_before(n):
    g.node.insert(concat_idx, n)
for inp in concat.input:
    # 找这个输入的 producer, 判断是什么通道
    prod = None
    for n in g.node:
        if inp in n.output:
            prod = n
            break
    tag = prod.op_type if prod else "?"
    if inp == "/model.22/Sigmoid_output_0":
        # cls 置信度 [1,1,8400] → ×256 (唯一需要缩放的通道)
        cname = "cls_scale_const"
        mout = "/model.22/cls_scaled"
        g.initializer.append(numpy_helper.from_array(
            np.array(SCALE, dtype=np.float32), name=cname))
        add_node_before(helper.make_node("Mul", [inp, cname], [mout],
                                         name="/model.22/cls_scale_mul"))
        print(f"  cls {inp} ({tag}) -> Mul x{SCALE} -> {mout}")
        new_inputs.append(mout)
    elif inp == "/model.22/Reshape_10_output_0":
        # kpt [1,6,8400] = [x1,y1,c1,x2,y2,c2] — 不缩放 (见文件头说明)
        print(f"  kpt {inp} ({tag}) -> 不动 (不再 ×256, 避免 REGTASK 溢出; kpt conf 解码自动置 1.0)")
        new_inputs.append(inp)
    else:
        print(f"  box {inp} ({tag}) -> 不动")
        new_inputs.append(inp)

del concat.input[:]
concat.input.extend(new_inputs)
onnx.checker.check_model(m)
onnx.save(m, DST)
print(f"OK -> {DST}")
