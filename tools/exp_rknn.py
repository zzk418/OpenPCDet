#!/usr/bin/env python3
"""int8 校准/量化实验工作台: 标定 → build → 同会话 val 评估 → 导出 rknn。

rknn-toolkit2 2.3.2 的 load_rknn 不支持 PC 模拟器推理, 评估必须在 build 会话内
(export_rknn.py 的 verify 就是这么做的)。因此本脚本把整条链串成一步, 便于批量对比:

    1. 标定集来源/数量     --calib-dir --n-calib  (真实/增强/混合 任意目录)
    2. 量化算法与方式     --algo --method       (normal/mmse/kl_divergence; layer/channel)
    3. 混合精度           --hybrid              (auto_hybrid_cos_thresh 值, 越低越多层留 fp16)
    4. 可选逐层误差分析   --analyze             (accuracy_analysis 每层 cos 相似度)

评估固定用 datasets/shelf_pose_reviewed/images/val (21 张, 带 GT), 指标同 eval_rknn_gt.py:
检出率 / 框匹配 GT 检出率 / 关键点误差(原图像素) / conf 中位。标定与评估严格分离。

用法:
    conda run -n rknn python tools/exp_rknn.py --onnx best_split.onnx \
        --calib-dir datasets/shelf_pose_reviewed/images/train --n-calib 125 \
        --out output/rk3588_deploy/exp/exp_name.rknn [--algo mmse --method channel] \
        [--hybrid 0.95] [--analyze]
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_rknn import prep_calib, letterbox, merge_outputs, decode_lb
from eval_rknn_gt import load_gt, kpt_to_orig

VAL_IMGS = 'datasets/shelf_pose_reviewed/images/val'
VAL_LABELS = 'datasets/shelf_pose_reviewed/labels/val'
IMG_W, IMG_H = 640, 480
CONF = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--calib-dir', required=True, help='标定图源目录 (真实/增强)')
    ap.add_argument('--n-calib', type=int, default=100)
    ap.add_argument('--algo', default='normal', choices=['normal', 'mmse', 'kl_divergence'])
    ap.add_argument('--method', default='layer',
                    choices=['layer', 'channel', 'group32', 'group64', 'group128', 'group256'])
    ap.add_argument('--hybrid', type=float, default=None, help='auto_hybrid_cos_thresh (须配 --auto-hybrid)')
    ap.add_argument('--auto-hybrid', action='store_true', help='build 时启用 auto_hybrid (混合精度, 敏感层留 fp16)')
    ap.add_argument('--quantized-dtype', default=None)
    ap.add_argument('--analyze', action='store_true', help='额外跑 accuracy_analysis 逐层误差')
    ap.add_argument('--n-analyze', type=int, default=5, help='accuracy_analysis 用几张 val 图')
    ap.add_argument('--cases', default=None, help='边界用例图目录 (test_day/night/multi/neg)')
    args = ap.parse_args()

    # 1. 标定集
    calib_paths = prep_calib(args.calib_dir, os.path.join(os.path.dirname(args.out), 'calib'), args.n_calib)
    dataset = os.path.join(os.path.dirname(args.out), 'calib', 'dataset.txt')
    with open(dataset, 'w') as f:
        f.write('\n'.join(calib_paths) + '\n')

    from rknn.api import RKNN
    rknn = RKNN(verbose=False)
    cfg = dict(
        mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
        target_platform='rk3588',
        quantized_algorithm=args.algo, quantized_method=args.method,
        quant_img_RGB2BGR=True,
    )
    if args.hybrid is not None:
        cfg['auto_hybrid_cos_thresh'] = args.hybrid
    if args.quantized_dtype is not None:
        cfg['quantized_dtype'] = args.quantized_dtype
    rknn.config(**cfg)
    assert rknn.load_onnx(model=args.onnx) == 0
    # auto_hybrid 必须显式传给 build (config 里的 cos_thresh 只是阈值); True 才启用混合精度
    ret = rknn.build(do_quantization=True, dataset=dataset, auto_hybrid=args.auto_hybrid)
    if ret != 0:
        print(f'BUILD_FAILED {args.out}')
        return 1
    rknn.init_runtime()  # PC 模拟器 (必须在 build 会话内)

    # 2. val 集评估
    imgs = sorted(glob.glob(os.path.join(VAL_IMGS, '*.jpg')))
    gt = load_gt(VAL_LABELS)
    det_any, det_match, kpt_errs, conf_maxs = 0, 0, [], []
    for p in imgs:
        stem = os.path.splitext(os.path.basename(p))[0]
        img = cv2.imread(p)
        lb, r, (dw, dh) = letterbox(img)
        blob = np.ascontiguousarray(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB))[None]
        y = merge_outputs(rknn.inference(inputs=[blob]))
        dets = decode_lb(y)
        if not dets:
            conf_maxs.append(0.0)
            continue
        conf_maxs.append(max(d[1] for d in dets))
        best = max(dets, key=lambda d: d[1])
        box_lb, conf, kpts_lb = best
        k_orig = kpt_to_orig(kpts_lb, r, dw, dh)
        det_any += 1
        (gx1, gy1, gx2, gy2), gkpts = gt[stem]
        gc = ((gx1 + gx2) / 2 * IMG_W, (gy1 + gy2) / 2 * IMG_H)
        bx1o, by1o = (box_lb[0] - dw) / r, (box_lb[1] - dh) / r
        bx2o, by2o = (box_lb[2] - dw) / r, (box_lb[3] - dh) / r
        if (bx1o <= gc[0] <= bx2o) and (by1o <= gc[1] <= by2o):
            det_match += 1
        for (gx, gy, gv), (px, py) in zip(gkpts, k_orig):
            if gv >= 1.0:
                kpt_errs.append(np.hypot(px - gx * IMG_W, py - gy * IMG_H))
    n = len(imgs)
    kpt_med = float(np.median(kpt_errs)) if kpt_errs else -1.0
    kpt_le30 = float(np.mean(np.asarray(kpt_errs) <= 30)) if kpt_errs else 0.0
    conf_med = float(np.median(conf_maxs)) if conf_maxs else 0.0
    res = (f'RESULT {os.path.basename(args.out)} | algo={args.algo} method={args.method} '
           f'n_calib={args.n_calib} hybrid={args.hybrid} | '
           f'det_any={det_any}/{n} det_match={det_match}/{n} '
           f'kpt_med={kpt_med:.1f}px kpt<=30px={kpt_le30:.1%} conf_med={conf_med:.3f}')
    print(res)

    # 2.5 边界用例回归 (可选): deploy_pre 的 test_* 图 (白天/夜间/多目标/负样本)
    if args.cases:
        import glob as _g
        case_imgs = sorted(_g.glob(os.path.join(args.cases, '*.jpg')))
        print(f'== cases ({len(case_imgs)} 张) ==')
        for p in case_imgs:
            img = cv2.imread(p)
            if img is None:
                continue
            lb, _, _ = letterbox(img)
            blob = np.ascontiguousarray(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB))[None]
            dets = decode_lb(merge_outputs(rknn.inference(inputs=[blob])))
            confs = ' '.join(f'{d[1]:.3f}' for d in dets) if dets else '-'
            print(f'  {os.path.basename(p)}: dets={len(dets)}  confs=[{confs}]')

    # 3. 逐层误差 (可选)
    if args.analyze:
        analyze_imgs = imgs[:args.n_analyze]
        adir = os.path.join(os.path.dirname(args.out), 'analysis', os.path.basename(args.out))
        os.makedirs(adir, exist_ok=True)
        try:
            rknn.accuracy_analysis(inputs=analyze_imgs, output_dir=adir)
            print(f'ANALYZE_DONE {adir}')
        except Exception as e:
            print(f'ANALYZE_FAILED {e}')

    # 4. 导出
    rknn.export_rknn(args.out)
    print(f'EXPORTED {args.out}')
    rknn.release()
    return 0


if __name__ == '__main__':
    sys.exit(main())
