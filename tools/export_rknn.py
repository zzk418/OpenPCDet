#!/usr/bin/env python3
"""shelf YOLOv8s-pose (2 关键点) ONNX -> RK3588 NPU RKNN 转换。

模型输出结构 (ultralytics 8.4 导出, DFL 已解码):
    output0: [1, 11, 8400]  = [4 box(xywh) + 1 cls + 6 pose(2点×3)] × 8400 锚

量化策略:
    int8: 需标定集 (每行一张图路径)。标定图必须和部署预处理完全一致
          (letterbox 640x640 + fill114 + RGB), 否则 int8 会偏。
          quantized_algorithm='normal' | 'mmse' | 'kl_divergence'
    fp16: 无需标定, 精度损失小但 RK3588 上慢一倍 (3TOPS vs 6TOPS)

标定图预处理 (--calib-img-dir): 读取后按 infer_image.py 的 letterbox 缩放到
640x640 (fill 114, INTER_LINEAR), 存到 --calib-out, 再写 dataset.txt。
(直接传原始图片给 rknn 只会被拉伸 resize, 与部署 letterbox 不一致 —— 这是
之前 int8 偏的主要原因之一。)

用法:
    conda run -n rknn python tools/export_rknn.py \
        [--onnx ...best_conf_scaled.onnx] [--out ...int8.rknn] \
        [--target rk3588] [--quant int8] [--algo mmse --method channel] \
        [--calib-img-dir datasets/shelf_pose_reviewed/images/train] \
        [--n-calib 100] [--calib-out output/rk3588_deploy/calib] \
        [--verify-dir output/rk3588_deploy/imgs]
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

IMG_SIZE = 640
LETTERBOX_FILL = 114
CONF_SCALE = 256.0
N_KPTS = 2


def letterbox(im, new_shape=(IMG_SIZE, IMG_SIZE), color=(LETTERBOX_FILL,) * 3):
    """等比缩放 + 114 填充。与 output/rk3588_deploy/infer_image.py 完全一致。"""
    shape = im.shape[:2]  # H, W
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # W, H
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def prep_calib(img_dir, out_dir, n=100):
    """从 img_dir 取 n 张真实图, letterbox 到 640x640, 存 out_dir, 返回绝对路径列表。

    标定图必须 = 部署预处理后的输入, rknn 加载时即为 640x640, 不会再做任何 resize。
    """
    img_dir = os.path.abspath(img_dir)
    out_dir = os.path.abspath(out_dir)
    imgs = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) +
                  glob.glob(os.path.join(img_dir, '*.png')))
    if not imgs:
        raise SystemExit(f'{img_dir} 下没有图片')
    step = max(1, len(imgs) // n)
    imgs = imgs[::step][:n]
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, p in enumerate(imgs):
        im = cv2.imread(p)
        if im is None:
            continue
        lb, _, _ = letterbox(im)
        out = os.path.join(out_dir, f'{i:04d}.jpg')
        cv2.imwrite(out, lb, [cv2.IMWRITE_JPEG_QUALITY, 95])
        paths.append(out)
    print(f'标定集: {len(paths)} 张 (letterbox {IMG_SIZE}x{IMG_SIZE}) <- {img_dir}')
    print(f'  输出: {out_dir}')
    return paths


def xywh2xyxy(xywh):
    x, y, w, h = xywh[..., 0], xywh[..., 1], xywh[..., 2], xywh[..., 3]
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=-1)


def nms(boxes, scores, iou_thres=0.45):
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_b = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_a + area_b - inter + 1e-9)
        order = rest[np.where(iou <= iou_thres)[0]]
    return np.array(keep, dtype=np.int64)


def merge_outputs(outs):
    """多输出模型 (onnx_split_output.py: box[1,4,8400]+conf[1,1,8400]+kpts[1,6,8400])
    按 shape 识别合并回 [1,11,8400]; 单输出模型 (旧手术版) 直接透传。

    按 shape 识别而非顺序/名字: rknn 与 onnxruntime 的输出顺序/命名可能不一致,
    但 4/1/6 三个通道数唯一。
    """
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
            f'多输出 shape 识别失败: {[np.asarray(o).shape for o in outs]}'
        return np.concatenate([box, conf, kpts], axis=1)
    return np.asarray(outs)


def decode_lb(out0, conf_thres=0.25, iou_thres=0.45):
    """[1,11,8400] -> [(box xyxy 640px, conf, kpts(n,3))], letterbox 坐标系。"""
    pred = merge_outputs(out0)[0].T  # [8400, 11]
    boxes_xywh, conf = pred[:, :4], pred[:, 4]
    kpts = pred[:, 5:].reshape(-1, N_KPTS, 3)
    if float(conf.max()) > 1.5:      # int8 手术版模型 cls conf 已 ×256, /256 还原
        conf = conf / CONF_SCALE
    # kpt 置信度通道是未饱和 logit (可达 ~74), 非 [0,1] 置信度, 无信息量 → 统一置 1.0
    # (手术只缩放 cls conf, kpt 不缩放; 旧的 kpts[...,2]/256 已无意义, 删除)
    kpts[..., 2] = 1.0
    m = conf >= conf_thres
    if not m.any():
        return []
    bx, c, k = xywh2xyxy(boxes_xywh[m]), conf[m], kpts[m]
    keep = nms(bx, c, iou_thres)
    return [(bx[i], c[i], k[i]) for i in keep]


def fmt_dets(dets):
    if not dets:
        return 'none'
    return '; '.join(
        f'c={c:.2f} bx={np.round(b, 1).tolist()} '
        f'k1={np.round(k[0, :2], 1).tolist()} k2={np.round(k[1, :2], 1).tolist()}'
        for b, c, k in dets)


def main():
    parser = argparse.ArgumentParser(description='ONNX -> RKNN (RK3588 NPU)')
    parser.add_argument('--onnx',
                        default='output/shelf_pose_train/shelf_v6_s_night_c2/weights/best_conf_scaled.onnx',
                        help='int8 用 onnx_scale_conf.py 手术后的 best_conf_scaled.onnx; fp16 用 best.onnx')
    parser.add_argument('--out', default='output/rk3588_deploy/shelf_v6_s_night_c2_int8.rknn')
    parser.add_argument('--target', default='rk3588')
    parser.add_argument('--quant', default='int8', choices=['int8', 'fp16'])
    parser.add_argument('--algo', default='normal',
                        choices=['normal', 'mmse', 'kl_divergence'],
                        help='量化算法 (int8), mmse 更准但更慢')
    parser.add_argument('--method', default='layer',
                        choices=['layer', 'channel', 'group32', 'group64',
                                 'group128', 'group256'],
                        help='量化方式 (int8): layer=逐层(默认), channel=逐通道(权重, 通常更准), '
                             'group{32..256}=分组量化 (仅配 quantized_dtype=w4a16, 4bit 权重省内存用, 精度不占优)')
    parser.add_argument('--quantized-dtype', default=None,
                        choices=['w8a8', 'w8a16', 'w16a16i', 'w16a16i_dfp', 'w4a16'],
                        help='权重x激活位宽 (默认 w8a8); w8a16=激活16bit 精度更高但算力减半, '
                             'w4a16=4bit 权重省内存')
    parser.add_argument('--auto-hybrid-cos-thresh', type=float, default=None,
                        help='auto-hybrid 量化 cos 阈值 (越低=越多敏感层保持 fp16, '
                             '默认不传=用 rknn 默认 0.98)')
    # 注: rknn-toolkit2 2.3.2 无标定 batch 参数 (build 的 rknn_batch_size 是推理 batch,
    #     会改输入形状, 不能用); 标定图数是 --n-calib
    parser.add_argument('--dataset', default=None,
                        help='现成 dataset.txt (已预处理好, 覆盖自动生成)')
    parser.add_argument('--calib-img-dir', default=None,
                        help='真实图源目录 (int8 标定), 自动 letterbox 后进 dataset.txt')
    parser.add_argument('--n-calib', type=int, default=100, help='标定图数量')
    parser.add_argument('--calib-out', default='output/rk3588_deploy/calib',
                        help='letterbox 后标定图输出目录')
    parser.add_argument('--verify-dir', default=None,
                        help='转换后 PC 模拟器验证目录 (逐图对比 fp32 onnx vs int8)')
    args = parser.parse_args()

    from rknn.api import RKNN

    dataset = None  # int8 标定集路径 (构建时确定, 精度分析时复用)

    rknn = RKNN(verbose=False)

    print('config...')
    cfg = dict(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=args.target,
        quant_img_RGB2BGR=True,   # 标定加载 BGR->RGB, 与部署 infer_image.py 一致
    )
    if args.quant == 'int8':
        # fp16 不量化: 不传 quantized_algorithm/method
        # (rknn-toolkit2 2.3.2 config() 拒绝 quantized_algorithm=None)
        cfg['quantized_algorithm'] = args.algo
        cfg['quantized_method'] = args.method
    if args.auto_hybrid_cos_thresh is not None:
        cfg['auto_hybrid_cos_thresh'] = args.auto_hybrid_cos_thresh
        print(f'  auto_hybrid_cos_thresh={args.auto_hybrid_cos_thresh}')
    if args.quantized_dtype is not None:
        cfg['quantized_dtype'] = args.quantized_dtype
        print(f'  quantized_dtype={args.quantized_dtype}')
    rknn.config(**cfg)

    print(f'load onnx ({args.quant})...')
    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        print(f'load_onnx failed: {ret}')
        sys.exit(1)

    print('build...')
    if args.quant == 'int8':
        dataset = args.dataset
        if dataset is None:
            if not args.calib_img_dir:
                print('int8 需要 --dataset 或 --calib-img-dir')
                sys.exit(1)
            calib_paths = prep_calib(args.calib_img_dir, args.calib_out, args.n_calib)
            dataset = os.path.join(args.calib_out, 'dataset.txt')
            with open(dataset, 'w') as f:
                f.write('\n'.join(calib_paths) + '\n')
            print(f'标定列表: {dataset}')
        ret = rknn.build(do_quantization=True, dataset=dataset)
    else:
        ret = rknn.build(do_quantization=False)

    if ret != 0:
        print(f'build failed: {ret}')
        sys.exit(1)

    print('export rknn...')
    ret = rknn.export_rknn(args.out)
    if ret != 0:
        print(f'export_rknn failed: {ret}')
        sys.exit(1)
    print(f'OK -> {args.out}')

    # 精度分析 (对 onnx 模拟器输出): 取标定集第一张做参考输入
    if args.quant == 'int8':
        print('accuracy analysis...')
        try:
            ref_img = None
            if dataset:
                with open(dataset) as f:
                    ref_img = f.readline().strip()
            ret = rknn.accuracy_analysis(inputs=[ref_img] if ref_img else None)
            print(f'accuracy_analysis: {ret}')
        except Exception as e:
            print(f'accuracy_analysis skipped: {e}')

    # PC 模拟器逐图验证 (build 会话内): int8 vs fp32 onnx 同一 letterbox 输入
    if args.verify_dir and args.quant == 'int8':
        import onnxruntime as ort
        print(f'\nverify (PC sim) {args.verify_dir} ...')
        rknn.init_runtime()  # 模拟器模式 (load_rknn 不能 sim, 必须在 build 会话内)
        ref = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
        for p in sorted(glob.glob(os.path.join(args.verify_dir, '*.jpg')) +
                        glob.glob(os.path.join(args.verify_dir, '*.png'))):
            img = cv2.imread(p)
            lb, _, _ = letterbox(img)
            blob = np.ascontiguousarray(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB))[None]
            out8 = merge_outputs(rknn.inference(inputs=[blob]))
            chw = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
            out32 = merge_outputs(ref.run(None, {'images': chw[None]}))
            print(f'  {os.path.basename(p)}:')
            print(f'    fp32: {fmt_dets(decode_lb(out32))}')
            print(f'    int8: {fmt_dets(decode_lb(out8))}')
    print('==== 转换+验证全部完成 ====')
    rknn.release()


if __name__ == '__main__':
    main()
