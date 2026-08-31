#!/usr/bin/env python3
"""RK3588 板端实时推理: 摄像头 → 货架关键点检测 + 可视化。

用法:
    python infer_camera.py                     # 默认摄像头 /dev/video0
    python infer_camera.py --camera 1
    python infer_camera.py --cores 7 --json    # 三核 + 逐帧打印检测 JSON
    python infer_camera.py --save out.mp4      # 同时录制视频

模型: shelf_v6_s_night_c2_fp16.rknn (YOLOv8s-pose 货架关键点, 640x640, fp16)
输出: 单输出或三输出拆分版 (box[1,4,8400] + conf[1,1,8400] + kpts[1,6,8400]) 自动兼容,
      decode_yolopose 内 merge_outputs 按 shape 合并回 [1,11,8400]。

--json 模式逐帧向 stdout 打印一行 JSON, 便于上位机解析关键点像素坐标:
    {"t": 123.456, "fps": 28.3, "dets": [{"box": [...], "conf": 0.9,
                                          "keypoints": [[x,y,c],[x,y,c]]}]}

依赖: rknn-toolkit-lite2 (aarch64), numpy, opencv-python
"""
import argparse
import json
import os
import sys
import time
import cv2
import numpy as np

import shelf_viz  # v4 风格绘制 (同 PC 端 infer_shelf_anchor)

# ── 模型超参 (与 export_rknn.py 导出配置一致) ──
IMG_SIZE = 640          # 模型输入 640x640
LETTERBOX_FILL = 114    # 与 ultralytics letterbox 一致
N_KPTS = 2              # P1/P2 两个关键点
CONF_SCALE = 256.0      # 旧手术版模型 conf 已 ×256; 生产 fp16 conf 原生 0~1 (解码按 >1.5 自动识别)
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "shelf_v6_s_night_c2_fp16.rknn")


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
    按 shape 识别合并回 [1,11,8400]; 单输出旧模型直接透传。"""
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


def decode_yolopose(output0, orig_shape, conf_thres=0.25, iou_thres=0.45,
                    n_kpts=N_KPTS):
    """[1,11,8400] → 检测列表 (坐标反 letterbox 回原图)。

    Returns: list[{box: xyxy(原图像素), conf: float, kpts: (n_kpts,3) [x,y,conf]}]
    """
    H, W = orig_shape
    pred = merge_outputs(output0)[0].T  # [8400, 11]
    boxes_xywh = pred[:, :4]
    conf = pred[:, 4]
    kpts = pred[:, 5:].reshape(-1, n_kpts, 3)
    # 手术版模型 cls conf 已 ×256 (量化手术), 检测后 /256 还原; 拆分版/ fp16 conf 原生 0~1
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


def draw_detections(img_bgr, dets):
    """v4 风格绘制 (同 PC 端 viz; 摄像头无 PCD, 2D 关键点, 图例 XYZ 显示 --)。

    只取最高置信度实例, 不画检测框。
    """
    img = img_bgr.copy()
    h, w = img.shape[:2]
    if not dets:
        return img
    best = max(dets, key=lambda d: d["conf"])
    keypoints = []
    for p in best["kpts"]:
        u, v, c = p
        if not (0 <= u < w and 0 <= v < h):
            continue
        keypoints.append({
            "pixel_uv": [int(round(u)), int(round(v))],
            "anchor_3d": None,  # 摄像头流无同步 PCD
            "confidence": float(c),
        })
    return shelf_viz.draw_keypoints(img, keypoints)


def main():
    parser = argparse.ArgumentParser(description="RK3588 货架关键点实时推理")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号 /dev/videoN")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--cores", type=int, default=3,
                        help="NPU 核: 1=单核 3=双核(默认) 7=三核 0=自动")
    parser.add_argument("--json", action="store_true",
                        help="逐帧向 stdout 打印检测 JSON (供上位机)")
    parser.add_argument("--no-display", action="store_true",
                        help="无显示环境 (仅推理, 不弹窗)")
    parser.add_argument("--save", default=None, help="录制视频到 <path> (如 out.mp4)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"模型不存在: {args.model}")

    from rknnlite.api import RKNNLite
    rknn = RKNNLite()
    if rknn.load_rknn(args.model) != 0:
        sys.exit(f"load_rknn 失败: {args.model}")
    if rknn.init_runtime(core_mask=args.cores) != 0:
        sys.exit(f"init_runtime 失败 (core_mask={args.cores})")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit(f"无法打开摄像头 /dev/video{args.camera}")

    writer = None
    if args.save:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                 20, (w, h))

    print(f"模型: {args.model}")
    print(f"摄像头: /dev/video{args.camera}  ({args.width}x{args.height})  "
          f"cores={args.cores}")
    print("运行中, Ctrl+C / q 退出。")

    fps_count = 0
    fps_time = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("读取帧失败")
                break

            # letterbox 640 + RGB uint8 NHWC (mean/std 由 runtime 内部归一化)
            lb, _, _ = letterbox(frame)
            blob = np.ascontiguousarray(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB))[None]

            t0 = time.perf_counter()
            outputs = rknn.inference(inputs=[blob])
            infer_ms = (time.perf_counter() - t0) * 1000

            dets = decode_yolopose(outputs, frame.shape[:2], args.conf, args.iou)

            # FPS
            fps_count += 1
            now = time.time()
            if now - fps_time >= 1.0:
                fps = fps_count / (now - fps_time)
                fps_count = 0
                fps_time = now

            # 上位机 JSON 输出 (即使有多个检测也全量打印)
            if args.json:
                payload = {
                    "t": round(time.time(), 3),
                    "fps": round(fps, 2),
                    "infer_ms": round(infer_ms, 1),
                    "dets": [{
                        "box": d["box"].round(1).tolist(),
                        "conf": round(d["conf"], 4),
                        "keypoints": d["kpts"].round(1).tolist(),
                    } for d in dets],
                }
                print(json.dumps(payload, ensure_ascii=False), flush=True)

            if not args.no_display:
                vis = draw_detections(frame, dets)
                cv2.putText(vis, f"FPS:{fps:.1f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if writer is not None:
                    writer.write(vis)
                cv2.imshow("RKNN Shelf Pose", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
            elif writer is not None:
                writer.write(draw_detections(frame, dets))

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        rknn.release()
        print("已退出。")


if __name__ == "__main__":
    main()
