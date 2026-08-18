#!/usr/bin/env python3
"""night_c2 YOLOv8s-pose (2 关键点) ONNX -> RK3588 NPU RKNN 转换。

模型输出结构 (ultralytics 8.4 导出, DFL 已解码):
    output0: [1, 11, 8400]  = [4 box(xywh) + 1 cls + 6 pose(2点×3)] × 8400 锚

量化策略:
    int8: 需标定集 dataset.txt (每行一张图路径), quantized_algorithm='normal'
    fp16: 无需标定, 精度损失小但 RK3588 上慢一倍 (3TOPS vs 6TOPS)

用法:
    conda run -n rknn python tools/export_rknn.py \
        [--onnx weights/best.onnx] [--out output/rk3588_deploy/shelf_v6_s_night_c2.rknn] \
        [--target rk3588] [--quant int8] [--dataset dataset.txt]
"""
import argparse
import sys


def build_dataset_txt(img_dir, out_path, max_imgs=300):
    import glob
    import os
    # 必须写绝对路径: rknn 量化会把 dataset.txt 里的相对路径解析到该文件所在目录
    img_dir = os.path.abspath(img_dir)
    imgs = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) +
                  glob.glob(os.path.join(img_dir, '*.png')))
    step = max(1, len(imgs) // max_imgs)
    imgs = imgs[::step][:max_imgs]
    with open(out_path, 'w') as f:
        f.write('\n'.join(imgs) + '\n')
    print(f'标定集: {len(imgs)} 张 -> {out_path}')
    return out_path


def main():
    parser = argparse.ArgumentParser(description='ONNX -> RKNN (RK3588 NPU)')
    parser.add_argument('--onnx', default='output/shelf_pose_train/shelf_v6_s_night_c2/weights/best.onnx')
    parser.add_argument('--out', default='output/rk3588_deploy/shelf_v6_s_night_c2.rknn')
    parser.add_argument('--target', default='rk3588')
    parser.add_argument('--quant', default='int8', choices=['int8', 'fp16'])
    parser.add_argument('--dataset', default=None, help='dataset.txt (int8 标定用), 可用 --img_dir 自动生成')
    parser.add_argument('--img_dir', default=None, help='标定图目录, 如 datasets/shelf_pose_reviewed_aug_night/images/train')
    args = parser.parse_args()

    from rknn.api import RKNN

    rknn = RKNN(verbose=False)

    # 预处理: RGB, 640x640 等比例缩放 + 114 填充 (与 ultralytics letterbox 一致)
    print('config...')
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=args.target,
        quantized_algorithm='normal' if args.quant == 'int8' else None,
        # RK3588 核心分配: 3 核 NPU
        # (core mask 0b111 = 3 cores)
    )

    print(f'load onnx ({args.quant})...')
    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        print(f'load_onnx failed: {ret}')
        sys.exit(1)

    print('build...')
    if args.quant == 'int8':
        dataset = args.dataset or (build_dataset_txt(args.img_dir, 'output/rk3588_deploy/dataset.txt') if args.img_dir else None)
        if dataset is None:
            print('int8 需要 --dataset 或 --img_dir 指定标定集')
            sys.exit(1)
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

    # 精度分析 (对 onnx 模拟器输出): 取 dataset.txt 第一行图片做参考输入
    print('accuracy analysis...')
    try:
        ref_img = None
        if args.quant == 'int8' and (args.dataset or dataset):
            with open(args.dataset or dataset) as f:
                ref_img = f.readline().strip()
        ret = rknn.accuracy_analysis(inputs=[ref_img] if ref_img else None)
        print(f'accuracy_analysis: {ret}')
    except Exception as e:
        print(f'accuracy_analysis skipped: {e}')
    rknn.release()


if __name__ == '__main__':
    main()
