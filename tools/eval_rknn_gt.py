#!/usr/bin/env python3
"""在 reviewed val 集上评估 RKNN / ONNX 的关键点检测精度 (GT-based)。

标定/评估分离: 标定用 train 集, 评估固定用 datasets/shelf_pose_reviewed/images/val (21 张, 带 GT)。

评估管线与部署 infer_image.py 完全一致: letterbox 640 + fill114 + BGR->RGB。
输出是 YOLO pose 归一化坐标 (box cxcywh + 2 keypoints, kpt vis=2)。

指标 (每张图取最高 conf 检测):
    det_rate@c  : conf>=c 且框与 GT 匹配 的图像占比 (c 默认 0.25, 部署阈值)
    det_rate_any: conf>=c 检出任意框的图像占比 (不看框匹配)
    kpt_err     : 检出图像的 P1/P2 与 GT 关键点距离 (原始图像像素) 的中位数
    conf_max    : 每图最高 conf 的中位数

用法:
    # 评估 rknn (PC 模拟器)
    conda run -n rknn python tools/eval_rknn_gt.py --model out.rknn
    # 评估 fp32 onnx 参考 (--model 传 .onnx)
    conda run -n rknn python tools/eval_rknn_gt.py --model best.onnx
    # 拆分输出模型用 --split 自动按 shape 合并
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_rknn import letterbox, merge_outputs, decode_lb

IMG_W, IMG_H = 640, 480  # 原图 640x480 (w x h)
CONF = 0.25


def load_gt(labels_dir):
    """{stem: (box_xyxy_norm, [(x,y,vis), (x,y,vis)])}  YOLO pose 标签"""
    gt = {}
    for p in glob.glob(os.path.join(labels_dir, '*.txt')):
        stem = os.path.splitext(os.path.basename(p))[0]
        with open(p) as f:
            toks = f.read().split()
        v = np.asarray(toks, dtype=np.float32)
        cx, cy, w, h = v[1:5]
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        kpts = [(v[5 + i * 3], v[6 + i * 3], v[7 + i * 3]) for i in range(2)]
        gt[stem] = ((x1, y1, x2, y2), kpts)
    return gt


def run_model(model_path, imgs):
    """加载模型并逐图推理, 返回 {stem: decoded_dets} (letterbox 640 像素坐标)。"""
    if model_path.endswith('.onnx'):
        import onnxruntime as ort
        sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        inp = sess.get_inputs()[0].name
        out = {}
        for p in imgs:
            img = cv2.imread(p)
            lb, _, _ = letterbox(img)
            blob = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
            y = merge_outputs(sess.run(None, {inp: blob[None]}))
            out[os.path.splitext(os.path.basename(p))[0]] = decode_lb(y)
    else:
        from rknn.api import RKNN
        rknn = RKNN(verbose=False)
        assert rknn.load_rknn(model_path) == 0, f'load_rknn failed: {model_path}'
        rknn.init_runtime()
        out = {}
        for p in imgs:
            img = cv2.imread(p)
            lb, _, _ = letterbox(img)
            blob = np.ascontiguousarray(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB))[None]
            y = merge_outputs(rknn.inference(inputs=[blob]))
            out[os.path.splitext(os.path.basename(p))[0]] = decode_lb(y)
        rknn.release()
    return out


def kpt_to_orig(kpts_lb, r, dw, dh):
    """letterbox 关键点 -> 原图像素 (640x480)。kpts_lb: (n,3) 或 (2,3)。"""
    k = np.asarray(kpts_lb, dtype=np.float32)
    ox = (k[..., 0] - dw) / r
    oy = (k[..., 1] - dh) / r
    return np.stack([ox, oy], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='rknn 或 onnx 路径')
    ap.add_argument('--imgs', default='datasets/shelf_pose_reviewed/images/val')
    ap.add_argument('--labels', default='datasets/shelf_pose_reviewed/labels/val')
    ap.add_argument('--conf', type=float, default=CONF)
    ap.add_argument('--kpt-err-px', type=float, default=30,
                    help='关键点误差像素阈值, 用于判定 "检出且正确" (部署 640x480 原图尺度)')
    args = ap.parse_args()

    imgs = sorted(glob.glob(os.path.join(args.imgs, '*.jpg')) +
                  glob.glob(os.path.join(args.imgs, '*.png')))
    gt = load_gt(args.labels)
    preds = run_model(args.model, imgs)

    det_any, det_match, kpt_errs = 0, 0, []
    conf_maxs = []
    per_img = []
    for p in imgs:
        stem = os.path.splitext(os.path.basename(p))[0]
        dets = preds.get(stem, [])
        if not dets:
            per_img.append((stem, 0.0, None, None, None))
            conf_maxs.append(0.0)
            continue
        conf_maxs.append(max(d[1] for d in dets))
        # 取最高 conf 检测
        best = max(dets, key=lambda d: d[1])
        box_lb, conf, kpts_lb = best
        lb, r, (dw, dh) = letterbox(cv2.imread(p))
        k_orig = kpt_to_orig(kpts_lb, r, dw, dh)
        det_any += 1
        # 框匹配: GT box 中心在 pred 框内 (或 IoU)
        (gx1, gy1, gx2, gy2), gkpts = gt[stem]
        gc = ((gx1 + gx2) / 2 * IMG_W, (gy1 + gy2) / 2 * IMG_H)  # GT 中心原图像素
        bx1, by1, bx2, by2 = box_lb
        bx1o, by1o = (bx1 - dw) / r, (by1 - dh) / r
        bx2o, by2o = (bx2 - dw) / r, (by2 - dh) / r
        match = (bx1o <= gc[0] <= bx2o) and (by1o <= gc[1] <= by2o)
        if match:
            det_match += 1
        # 关键点误差 (只对可见 GT kpt, 直接按序匹配)
        errs = []
        for (gx, gy, gv), (px, py) in zip(gkpts, k_orig):
            if gv >= 1.0:
                errs.append(np.hypot(px - gx * IMG_W, py - gy * IMG_H))
        if errs:
            kpt_errs.extend(errs)
        per_img.append((stem, conf, bool(match), k_orig, errs))

    n = len(imgs)
    print(f'== {os.path.basename(args.model)} | {n} 张 val (conf>={args.conf}) ==')
    print(f'  检出任意框          : {det_any}/{n}  ({det_any / n:.1%})')
    print(f'  框匹配 GT 检出率    : {det_match}/{n}  ({det_match / n:.1%})')
    if kpt_errs:
        print(f'  关键点误差(原图像素) : 中位 {np.median(kpt_errs):.1f}  均值 {np.mean(kpt_errs):.1f}  '
              f'(<= {args.kpt_err_px}px: {np.mean(np.asarray(kpt_errs) <= args.kpt_err_px):.1%})')
    if conf_maxs:
        print(f'  conf 分布           : 中位 {np.median(conf_maxs):.3f}  '
              f'max {max(conf_maxs):.3f}  最小 {min(conf_maxs):.3f}')
    # 逐图简表
    print('  逐图 (stem conf match err_px):')
    for stem, conf, match, k_orig, errs in per_img:
        e = f'{np.mean(errs):.1f}' if errs else '-'
        print(f'    {stem}  conf={conf:.3f}  match={"Y" if match else "N"}  kpt_err={e}')
    return per_img


if __name__ == '__main__':
    main()
