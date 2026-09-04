#!/usr/bin/env python3
"""RK3588 板端离线推理: 图片/目录 → 货架关键点检测 + 可视化。

用法:
    python infer_image.py                          # 默认模型/单图占位, 见参数
    python infer_image.py --input test.jpg
    python infer_image.py --input ./imgs/          # 目录
    python infer_image.py --model shelf_mobilenet_r2_fp16.rknn --conf 0.7 --cores 3

模型: shelf_mobilenet_r2_fp16.rknn (MobileNetV3-Large-pose, 640x640, fp16)
输出: 单输出或三输出拆分版 (box[1,4,8400] + conf[1,1,8400] + kpts[1,6,8400]) 自动兼容;
      decode_yolopose 内 merge_outputs 按 shape 合并回 [1,11,8400] = [box xywh(letterbox像素) |
      cls_conf | kpt1_x,y,c | kpt2_x,y,c] × 8400。

依赖: rknn-toolkit-lite2 (aarch64), numpy, opencv-python
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import shelf_viz  # PCD 深度查表 + v4 风格绘制 (同 PC 端 infer_shelf_anchor)

# ── 模型超参 (与 export_rknn.py 导出配置一致) ──
IMG_SIZE = 640          # 模型输入 640x640
LETTERBOX_FILL = 114    # 与 ultralytics letterbox 一致
N_KPTS = 2              # P1/P2 两个关键点
CONF_SCALE = 256.0      # 旧手术版模型 conf 已 ×256; 生产 fp16 conf 原生 0~1 (解码按 >1.5 自动识别)
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "shelf_mobilenet_r2_fp16.rknn")


# ═══════════════════════ 纯 numpy 预处理 ═══════════════════════

def letterbox(im, new_shape=(IMG_SIZE, IMG_SIZE), color=(LETTERBOX_FILL,) * 3):
    """等比缩放 + 114 填充 (与 ultralytics 一致)。返回 (填充图, ratio, (dw,dh))。"""
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


def xywh2xyxy(xywh):
    x, y, w, h = xywh[..., 0], xywh[..., 1], xywh[..., 2], xywh[..., 3]
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=-1)


def nms(boxes, scores, iou_thres=0.45):
    """boxes: Nx4 xyxy, scores: N → 保留下标 (score 降序)。"""
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
    """多输出模型 (输出拆分版: box[1,4,8400]+conf[1,1,8400]+kpts[1,6,8400])
    按 shape 识别合并回 [1,11,8400]; 单输出旧模型直接透传。

    按 shape 而非顺序/名字识别: rknn 输出顺序可能和导出时不一致, 但 4/1/6 通道数唯一。
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


def decode_yolopose(output0, orig_shape, conf_thres=0.7, iou_thres=0.45,
                    n_kpts=N_KPTS):
    """[1,11,8400] → 检测列表 (坐标反 letterbox 回原图)。

    Returns: list[{box: xyxy(原图像素), conf: float, kpts: (n_kpts,3) [x,y,conf]}]
    """
    H, W = orig_shape
    pred = merge_outputs(output0)[0].T  # [8400, 11]
    boxes_xywh = pred[:, :4]
    conf = pred[:, 4]
    kpts = pred[:, 5:].reshape(-1, n_kpts, 3)
    # int8 修复版模型输出 conf 已 ×256 (量化手术), 检测后 /256 还原; fp16 模型 conf 原生 0~1
    if float(conf.max()) > 1.5:
        conf = conf / CONF_SCALE
    # kpt 置信度通道是未饱和 logit (可达 ~74), 非 [0,1] 置信度, 无信息量 →
    # 统一置 1.0 (手术只缩放 cls conf, kpt 不缩放; 旧的 /256 已无意义)。
    # 框已通过 cls 置信度检测 → 关键点视为存在。
    kpts[..., 2] = 1.0

    mask = conf >= conf_thres
    if not mask.any():
        return []
    boxes_xywh, conf, kpts = boxes_xywh[mask], conf[mask], kpts[mask]
    boxes_xyxy = xywh2xyxy(boxes_xywh)

    keep = nms(boxes_xyxy, conf, iou_thres)

    _, r, (dw, dh) = letterbox(np.zeros((H, W, 3), dtype=np.uint8))
    dets = []
    for i in keep:
        box = boxes_xyxy[i].copy()
        box[0::2] = (box[0::2] - dw) / r
        box[1::2] = (box[1::2] - dh) / r
        kp = kpts[i].copy()
        kp[..., 0] = (kp[..., 0] - dw) / r
        kp[..., 1] = (kp[..., 1] - dh) / r
        dets.append({
            "box": box.astype(np.float32),
            "conf": float(conf[i]),
            "kpts": kp.astype(np.float32),
        })
    return dets


def attach_depth(dets, depth_map):
    """给每个检测的关键点附加 anchor_3d (PCD 深度查表)。无 depth_map → anchor_3d=None。"""
    for det in dets:
        anchored = []
        for p in det["kpts"]:
            u, v, _ = p
            a3d = None
            if depth_map is not None:
                a3d = shelf_viz.get_anchor_3d(int(round(u)), int(round(v)), depth_map)
            anchored.append(a3d)
        det["anchor_3d"] = anchored
    return dets


def draw_detections(img_bgr, dets, depth_map=None):
    """v4 风格绘制: 红点+准线+P 编号+虚线+黄 X 中心+图例面板 (带 PCD 3D XYZ)。

    与 PC 端 `output/shelf_pose_inference_v6s_night_c2_pngs0812_conf60/viz` 同款,
    板端略简: 只取 dets 里置信度最高的实例, 不画检测框。
    """
    img = img_bgr.copy()
    h, w = img.shape[:2]
    dets = attach_depth(dets, depth_map)
    if not dets:
        return img
    # 取最高置信度实例 (单货架场景)
    best = max(dets, key=lambda d: d["conf"])
    keypoints = []
    for idx, p in enumerate(best["kpts"]):
        u, v, c = p
        if not (0 <= u < w and 0 <= v < h):
            continue
        keypoints.append({
            "pixel_uv": [int(round(u)), int(round(v))],
            "anchor_3d": best["anchor_3d"][idx] if idx < len(best["anchor_3d"]) else None,
            "confidence": float(c),
        })
    return shelf_viz.draw_keypoints(img, keypoints)


def main():
    parser = argparse.ArgumentParser(description="RK3588 货架关键点离线推理")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input", default="./imgs",
                        help="单张图片路径或图片目录 (默认 ./imgs 测试图文件夹)")
    parser.add_argument("--conf", type=float, default=0.7)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--cores", type=int, default=7,
                        help="NPU 核: 1=单核 3=双核 7=三核(默认) 0=自动")
    parser.add_argument("--out", default="./results",
                        help="结果保存目录 (可视化 jpg + detections.json)")
    parser.add_argument("--pcd", default=None,
                        help="PCD 点云目录 (默认: 图片所在目录, 与上位机一样按同名 stem 匹配 "
                             "`{stem}.pcd` 做深度查表; 无对应 pcd 时仅画 2D)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"模型不存在: {args.model}")

    os.makedirs(args.out, exist_ok=True)

    # 收集图片
    if os.path.isdir(args.input):
        image_list = sorted(
            glob.glob(os.path.join(args.input, "*.[jJ][pP][gG]")) +
            glob.glob(os.path.join(args.input, "*.[pP][nN][gG]")))
    else:
        image_list = [args.input]
    if not image_list:
        sys.exit(f"未找到图片: {args.input}")
    print(f"共 {len(image_list)} 张图片")

    # 初始化 RKNNLite
    from rknnlite.api import RKNNLite
    rknn = RKNNLite()
    if rknn.load_rknn(args.model) != 0:
        sys.exit(f"load_rknn 失败: {args.model}")
    if rknn.init_runtime(core_mask=args.cores) != 0:
        sys.exit(f"init_runtime 失败 (core_mask={args.cores})")

    all_dets = {}
    total_ms = 0.0
    for img_path in image_list:
        stem = Path(img_path).stem
        img = cv2.imread(img_path)
        if img is None:
            print(f"[{stem}] 读图失败, 跳过")
            continue

        # letterbox 640 + RGB uint8 NHWC (mean/std 由 runtime 内部归一化)
        lb, _, _ = letterbox(img)
        blob = np.ascontiguousarray(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB))[None]

        t0 = time.perf_counter()
        outputs = rknn.inference(inputs=[blob])
        infer_ms = (time.perf_counter() - t0) * 1000

        dets = decode_yolopose(outputs, img.shape[:2], args.conf, args.iou)

        # PCD 深度查表 (与图片同目录按同名 stem 匹配, 同上位机; 缺失时 2D-only)
        depth_map = None
        pcd_dir = args.pcd if args.pcd else os.path.dirname(img_path)
        pcd_path = os.path.join(pcd_dir, f"{stem}.pcd")
        if os.path.exists(pcd_path):
            xyz = shelf_viz.read_pcd_binary(pcd_path)
            depth_map = shelf_viz.build_depth_map(xyz, *img.shape[:2][::-1])
        else:
            print(f"    [warn] 无 PCD: {pcd_path} (跳过 3D 查表)")

        vis = draw_detections(img, dets, depth_map)
        save_path = os.path.join(args.out, f"{stem}_out.jpg")
        cv2.imwrite(save_path, vis)

        all_dets[stem] = [{
            "box": d["box"].round(1).tolist(),
            "conf": round(d["conf"], 4),
            "keypoints": d["kpts"].round(1).tolist(),  # [[x,y,c],[x,y,c]]
            "anchor_3d": d.get("anchor_3d"),           # [[x,y,z]|None, ...] (深度查表)
        } for d in dets]
        total_ms += infer_ms

        print(f"[{stem}] dets={len(dets)}  infer={infer_ms:.1f}ms  pcd={'OK' if depth_map is not None else 'N/A'}  -> {save_path}")
        for d in all_dets[stem]:
            a3d = [a if a is None else [round(v, 1) for v in a] for a in (d["anchor_3d"] or [])]
            print(f"    conf={d['conf']:.3f} box={d['box']} kpts={d['keypoints']} anchor3d={a3d}")

    rknn.release()

    json_path = os.path.join(args.out, "detections.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_dets, f, ensure_ascii=False, indent=2)
    n = len(all_dets)
    print(f"\n完成: {n} 张, 平均推理 {total_ms / n:.1f} ms/frame"
          f" ({1000 * n / total_ms:.1f} FPS)")
    print(f"结果: {os.path.abspath(args.out)}/")


if __name__ == "__main__":
    main()
