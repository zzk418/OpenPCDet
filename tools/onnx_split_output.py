#!/usr/bin/env python3
"""ONNX 手术: 把 output0 [1,11,8400] 拆成 3 个独立输出, 各走自己的 int8 per-tensor scale。

背景: output0 = [box xywh(0~800) | cls_conf(0~1) | kpt(0~800 + conf logit~74)] 混在一个
      张量里 per-tensor int8 量化, scale 被坐标量纲主导, conf 通道精度被压死
      (即 onnx_scale_conf.py 的 ×256 手术要解决的同一结构问题)。

拆分后 (CONF_SCALE 手术整个不需要了):
    output_box   [1,4,8400]   xywh, 自己一个 scale (~800/127)
    output_conf  [1,1,8400]   cls conf 0~1, 自己一个 scale (1/127≈0.008, 原来混合张量里 ≈0.025)
    output_kpts  [1,6,8400]   x1,y1,c1,x2,y2,c2, 自己一个 scale (c1/c2 是 logit 无信息量, 解码置 1.0)

三输出 per-tensor 量化互不干扰。解码端按 shape 识别合并回 [1,11,8400]
(export_rknn.py / infer_image.py / infer_camera.py 的 merge_outputs)。

用法: conda run -n pc python tools/onnx_split_output.py [in.onnx] [out.onnx]
"""
import sys
import onnx
from onnx import helper, TensorProto

SRC = sys.argv[1] if len(sys.argv) > 1 else "output/shelf_pose_train/shelf_v6_s_night_c2/weights/best.onnx"
DST = sys.argv[2] if len(sys.argv) > 2 else "output/shelf_pose_train/shelf_v6_s_night_c2/weights/best_split.onnx"

m = onnx.load(SRC)
g = m.graph

# 找产生 output0 的最终 Concat
concat = None
for n in g.node:
    if n.op_type == "Concat" and "output0" in n.output:
        concat = n
        break
assert concat is not None, "找不到 output0 Concat"
print(f"final Concat: {concat.name}  inputs={list(concat.input)}")

# 按 producer 名分类三个通道 (与 onnx_scale_conf.py 一致的口径)
# 注意: graph output 的 ValueInfo name 必须 = 张量真名, 不能自造别名, 否则 checker 报错;
# 解码端按 shape 识别 (4/1/6 通道唯一), 不依赖名字。
new_outputs = []  # (tensor_name, shape)
for inp in concat.input:
    if inp == "/model.22/Sigmoid_output_0":       # cls conf [1,1,8400], 不缩放
        new_outputs.append((inp, [1, 1, 8400]))
        print(f"  output_conf  <- {inp}  [1,1,8400]")
    elif inp == "/model.22/Reshape_10_output_0":  # kpt [1,6,8400], 不缩放
        new_outputs.append((inp, [1, 6, 8400]))
        print(f"  output_kpts  <- {inp}  [1,6,8400]")
    else:                                          # box [1,4,8400]
        new_outputs.append((inp, [1, 4, 8400]))
        print(f"  output_box   <- {inp}  [1,4,8400]")

# 换成三个 graph output, 删掉原单输出与 Concat (避免计算后丢弃)
del g.output[:]
for tensor, shape in new_outputs:
    g.output.append(helper.make_tensor_value_info(tensor, TensorProto.FLOAT, shape))
g.node.remove(concat)
print("  removed output0 + Concat_5")

onnx.checker.check_model(m)
onnx.save(m, DST)
print(f"OK -> {DST}")
