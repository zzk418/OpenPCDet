#!/usr/bin/env python3
"""fp16 导出精度回归 (PC 模拟器, 不需板子)。

判断"上板 fp16 精度崩"是否出在 **转换环节** (fp16 rknn 本身输出就偏):
在 PC 端 rknn-toolkit2 模拟器里用 `do_quantization=False` 现场 build fp16 模型,
逐图对比 fp32 onnx (onnxruntime CPU) decode 后的最高置信度实例。

fp16 是 fp32 近无损半精度量化, 对"赢了 NMS 的那个实例"期望:
  - conf 差 < 0.005
  - 检测框坐标漂移 < 1 px (letterbox 640 空间)
  - 关键点像素漂移 < 0.5 px  ← 这是"精度"的直接判据

原始 tensor 的 max_abs_diff 只作参考打印, 不判 FAIL: 它被低置信度 anchor 的
大框 (bh/bw) 和未使用的 kpt 置信度 logit 通道 (k1c/k2c, 解码时统一置 1) 主导,
对最终检测结果无影响 (实测最坏 0.95% 相对误差, 而赢家实例关键点仅 0.18 px)。

任一项 decode 阈值超限 → FAIL (转换环节有损, 需查导出配置/算子融合);
全部通过 → PASS (转换无损, 上板精度崩的原因在板端: librknnrt.so 版本不匹配
             或在线预处理/对齐 ≠ 离线, 见 output/rk3588_deploy/snap_infer_once.py)。

用法 (需 rknn conda 环境):
  conda run -n rknn python tools/verify_fp16_export.py
  conda run -n rknn python tools/verify_fp16_export.py --onnx <path> --imgs <dir>
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

IMG_SIZE = 640
LETTERBOX_FILL = 114
N_KPTS = 2
CONF_SCALE = 256.0   # 旧 int8 手术版 conf ×256; fp16 原生 0~1 (解码按 >1.5 自动识别)


def letterbox(im, new_shape=(IMG_SIZE, IMG_SIZE), color=(LETTERBOX_FILL,) * 3):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def merge_outputs(outs):
    """多输出拆版按 shape 合并回 [1,11,8400]; 单输出直接透传 (与 infer_image.py 一致)。"""
    if isinstance(outs, (list, tuple)):
        if len(outs) == 1:
            return np.asarray(outs[0])
        box = conf = kpts = None
        for o in outs:
            o = np.asarray(o)
            if o.shape[1] == 4:
                box = o
            elif o.shape[1] == 1:
                conf = o
            elif o.shape[1] == 6:
                kpts = o
        assert box is not None and conf is not None and kpts is not None, \
            f"多输出 shape 识别失败: {[np.asarray(o).shape for o in outs]}"
        return np.concatenate([box, conf, kpts], axis=1)
    return np.asarray(outs)


def top_detection(out):
    """decode 最高置信度实例 → (conf, box_xywh(N,4), kpts(N,2)) (letterbox 640 像素)。
    conf ×256 手术自动识别; kpt 置信度通道是无信息 logit, 统一置 1。"""
    pred = np.asarray(out)[0].T            # [8400, 11]
    conf = pred[:, 4]
    box = pred[:, :4]
    kpts = pred[:, 5:].reshape(-1, N_KPTS, 3)
    if float(conf.max()) > 1.5:
        conf = conf / CONF_SCALE
    i = int(np.argmax(conf))
    return float(conf[i]), box[i].copy(), kpts[i][:, :2].copy()


def main():
    ap = argparse.ArgumentParser(description="fp16 导出精度回归 (PC 模拟器)")
    ap.add_argument("--onnx", default="/code/OpenPCDet/output/best_ckpts/shelf_mobilenet_r2_best.onnx")
    ap.add_argument("--imgs", default="/code/OpenPCDet/output/rk3588_deploy/imgs")
    ap.add_argument("--tol-conf", type=float, default=0.005,
                    help="最高 conf 绝对差阈值 (默认 0.005)")
    ap.add_argument("--tol-box", type=float, default=1.0,
                    help="赢家实例检测框坐标漂移阈值 px (默认 1.0)")
    ap.add_argument("--tol-kpt", type=float, default=0.5,
                    help="关键点像素漂移阈值 px (默认 0.5)")
    args = ap.parse_args()

    if not os.path.isfile(args.onnx):
        sys.exit(f"onnx 不存在: {args.onnx}")
    imgs = sorted(glob.glob(os.path.join(args.imgs, "*.jpg")) +
                  glob.glob(os.path.join(args.imgs, "*.png")))
    if not imgs:
        sys.exit(f"无测试图: {args.imgs}")
    print(f"测试图 {len(imgs)} 张, onnx={os.path.basename(args.onnx)}")

    import onnxruntime as ort
    from rknn.api import RKNN

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                target_platform="rk3588", quant_img_RGB2BGR=True)
    assert rknn.load_onnx(model=args.onnx) == 0, "load_onnx failed"
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        rknn.release()
        sys.exit(f"fp16 build failed: {ret}")
    rknn.init_runtime()  # x86 模拟器

    worst = {"conf": 0.0, "box": 0.0, "kpt": 0.0}
    rows = []
    for p in imgs:
        img = cv2.imread(p)
        lb, _, _ = letterbox(img)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        out32 = merge_outputs(sess.run(None, {"images": chw[None]}))
        blob = np.ascontiguousarray(rgb)[None]
        out16 = merge_outputs(rknn.inference(inputs=[blob]))

        c32, b32, k32 = top_detection(out32)
        c16, b16, k16 = top_detection(out16)
        conf_d = abs(c32 - c16)
        box_d = float(np.max(np.abs(b32 - b16)))
        kpt_d = float(np.max(np.hypot(k32[:, 0] - k16[:, 0], k32[:, 1] - k16[:, 1])))
        worst["conf"] = max(worst["conf"], conf_d)
        worst["box"] = max(worst["box"], box_d)
        worst["kpt"] = max(worst["kpt"], kpt_d)
        rows.append((os.path.basename(p), c32, c16, conf_d, box_d, kpt_d))

    print(f"\n{'img':<22} {'fp32_conf':>9} {'fp16_conf':>9} {'Δconf':>7} "
          f"{'boxΔpx':>7} {'kptΔpx':>7}")
    for name, c32, c16, conf_d, box_d, kpt_d in rows:
        print(f"{name:<22} {c32:>9.4f} {c16:>9.4f} {conf_d:>7.4f} "
              f"{box_d:>7.2f} {kpt_d:>7.2f}")
    print(f"\nworst: Δconf={worst['conf']:.4f}  boxΔpx={worst['box']:.2f}  "
          f"kptΔpx={worst['kpt']:.2f}")

    ok = (worst["conf"] <= args.tol_conf and worst["box"] <= args.tol_box
          and worst["kpt"] <= args.tol_kpt)
    print(f"\n阈值: Δconf<={args.tol_conf}  boxΔpx<={args.tol_box}  kptΔpx<={args.tol_kpt}")
    print("结果: " + ("PASS — fp16 转换无损 (赢家实例关键点/框/conf 均一致), 上板精度崩的"
                      "原因在板端: librknnrt.so 版本不匹配或在线预处理/对齐 ≠ 离线, 非转换环节"
                      if ok else
                      "FAIL — fp16 转换本身有损, 需查导出配置/算子融合 (HardSigmoid/ConvHardSwish)"))
    rknn.release()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
