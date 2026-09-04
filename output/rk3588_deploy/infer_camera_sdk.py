#!/usr/bin/env python3
"""RK3588 板端实时推理 — 蓝芯 MRDVS LxCamera SDK 视频流 (RGB)。

与 infer_camera.py 走同一条推理管线 (letterbox 640 + RGB uint8 + rknn.inference
+ decode_yolopose), 保证实时帧结果和离线推理 (infer_image.py) 完全一致。
数据源换成 LxCamera SDK 的 getFrame -> getRGBImage (不再用 OpenCV VideoCapture)。

用法:
    # 自动发现相机, 打开第一台
    python infer_camera_sdk.py

    # 指定相机
    python infer_camera_sdk.py --ip 192.168.2.150
    python infer_camera_sdk.py --sn 519889C9A2A6468E
    python infer_camera_sdk.py --index 0

    # 逐帧 JSON 给上位机 / 无显示纯推理 / 录制
    python infer_camera_sdk.py --json --no-display
    python infer_camera_sdk.py --save out.mp4

    # 打开 3D 深度流, 关键点做深度查表出 anchor_3d (同离线 infer_image, 需相机支持)
    # 自动开启 SDK RGBD 对齐 (DEPTH_TO_RGB): 点云与 RGB 同分辨率同索引, 关键点像素直接查,
    # 缺失像素局部窗口邻域采样; 对齐不可用时内参投影回退 (内参按 SN 自动读 camera_intrinsics_<sn>.json)
    python infer_camera_sdk.py --depth3d

依赖: rknn-toolkit-lite2 (aarch64), numpy, opencv-python,
      lx_camera_py-1.3.3 wheel + SDK 动态库 (install.sh 装到 /opt/MRDVS/lib)

--json 模式逐帧向 stdout 打印一行 JSON:
    {"t": 123.456, "fps": 28.3, "infer_ms": 8.1,
     "dets": [{"box": [...], "conf": 0.9, "keypoints": [[x,y,c],[x,y,c]],
               "anchor_3d": [[x,y,z],null]}]}
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

import shelf_viz
# 与 infer_camera.py 共用同一推理管线 (保证与离线推理一致)
from infer_camera import letterbox, decode_yolopose, DEFAULT_MODEL
# SDK 动态库查找 / c_char32 字段清洗, 与内参脚本共用
from query_camera_intrinsics import find_sdk_lib, _bstr

# LxCameraSDK 是纯 Python wheel, 未安装时 --help 也应可用 → 装不上就绑 None,
# 在 open_camera 里再给友好报错。
try:
    from LxCameraSDK.lx_camera_api import LxCamera
    from LxCameraSDK.lx_camera_define import (
        LX_STATE, LX_OPEN_MODE, LX_CAMERA_FEATURE, LX_RGBD_ALIGN_MODE, LX_ALGORITHM_MODE)
except ImportError:
    LxCamera = LX_STATE = LX_OPEN_MODE = LX_CAMERA_FEATURE = LX_RGBD_ALIGN_MODE = None
    LX_ALGORITHM_MODE = None


class CameraOpenError(RuntimeError):
    """open_camera 失败并携带 SDK 返回码.

    继承 RuntimeError → 所有原来 except RuntimeError 的调用方不受影响;
    需要区分错误类型的调用方(如重连逻辑识别独占锁 -9)读 .ret。
    """
    def __init__(self, msg, ret=None):
        super().__init__(msg)
        self.ret = ret


def disable_builtin_algorithm(camera, handle):
    """关掉相机内置应用算法 (托盘/避障/货架). 内置算法会占一路算法流 (algor:1),
    和我们的 rgb+depth raw 流一起开 → 千兆带宽/流调度顶不住 → DcStartStream
    收不到流自动停 (LX_E_NOT_RECEIVE_STREAM), 还曾引发 FRAME_ID_NOT_MATCH.
    本项目用 Python 推理替代内置算法, 开流前必须关. 失败只警告不阻断 (万一要保留内置算法)."""
    try:
        ret = camera.DcSetIntValue(handle, LX_CAMERA_FEATURE.LX_INT_ALGORITHM_MODE,
                                   LX_ALGORITHM_MODE.MODE_ALL_OFF.value)
        if ret == LX_STATE.LX_SUCCESS:
            print("[algo] 内置算法已关 (LX_INT_ALGORITHM_MODE=0), 腾出流带宽给 rgb+depth", flush=True)
            return True
        print(f"[warn] 关闭内置算法失败({ret}), 若仍收不到流再查相机算法配置", flush=True)
    except Exception as e:
        print(f"[warn] 关闭内置算法异常 ({e})", flush=True)
    return False


def open_camera(dll_path, ip, sn, index):
    """打开相机 (按 --ip/--sn/--index, 默认第一台). 返回 (camera, handle, dev_info)。"""
    if LxCamera is None:
        sys.exit(
            "缺少 LxCameraSDK 包。请安装:\n"
            "  pip install <SDK 仓库>/Sample/python/lx_camera_py-1.3.3-py3-none-any.whl")
    camera = LxCamera(dll_path)
    camera.DcSetInfoOutput(0, True, "")  # 抑制 SDK 刷屏日志
    print(f"SDK: {camera.DcGetApiVersion()}")

    ret, dev_list, dev_num = camera.DcGetDeviceList()
    if ret != LX_STATE.LX_SUCCESS:
        raise RuntimeError(f"DcGetDeviceList 失败: {ret}")
    print(f"发现 {dev_num} 台相机")

    if ip:
        mode, param = LX_OPEN_MODE.OPEN_BY_IP, ip
    elif sn:
        mode, param = LX_OPEN_MODE.OPEN_BY_SN, sn
    else:
        if dev_num == 0:
            raise RuntimeError("未发现相机 (检查网络/供电)")
        idx = index if index is not None else 0
        if idx >= dev_num:
            raise RuntimeError(f"--index {idx} 超出范围 (共 {dev_num} 台)")
        dev = dev_list[idx]
        print(f"设备 {idx}: {_bstr(dev.name)}  IP={_bstr(dev.ip)}")
        s = _bstr(dev.sn)
        mode, param = (LX_OPEN_MODE.OPEN_BY_SN, s) if s else (LX_OPEN_MODE.OPEN_BY_INDEX, str(idx))

    ret, handle, dev_info = camera.DcOpenDevice(mode, param)
    if ret != LX_STATE.LX_SUCCESS:
        raise CameraOpenError(f"DcOpenDevice({mode}, {param}) 失败: {ret}", ret=ret)
    print(f"已打开: {_bstr(dev_info.name, '?')}  SN={_bstr(dev_info.sn, '?')}  "
          f"IP={_bstr(dev_info.ip)}")
    disable_builtin_algorithm(camera, handle)   # 关内置算法流, 否则和 rgb+depth 抢带宽
    return camera, handle, dev_info


def get_depth_intrinsics(camera, handle, intrinsics_path, sn=None):
    """深度相机内参 (fx,fy,cx,cy): 优先 --intrinsics JSON; 其次按相机 SN 自动找
    脚本同目录 camera_intrinsics_<sn>.json (query_camera_intrinsics.py 输出);
    再在线查 get3DIntricParam。"""
    candidates = []
    if intrinsics_path:
        candidates.append(intrinsics_path)
    if sn:
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       f"camera_intrinsics_{sn}.json"))
    for p in candidates:
        if p and os.path.isfile(p):
            with open(p, encoding='utf-8') as f:
                di = json.load(f).get("depth_intrinsic", {})
            if di:
                return di["fx"], di["fy"], di["cx"], di["cy"]
            print(f"[warn] {p} 无 depth_intrinsic, 改在线查询")
    ret, intr3d, _ = camera.get3DIntricParam(handle)
    if ret == LX_STATE.LX_SUCCESS and intr3d is not None:
        return intr3d[0], intr3d[1], intr3d[2], intr3d[3]
    return None


def setup_rgbd_align(camera, handle):
    """开启相机内置 RGBD 对齐 (DEPTH_TO_RGB): 深度/点云对齐到 RGB 分辨率,
    使 getPointCloud 输出与 RGB 同分辨率、同像素索引 → 关键点像素直接查点云,
    不再需要手动内参投影 + 比例缩放。同时开强制帧同步保证 RGB/D 同帧。
    失败返回 False (调用方回退内参投影)。"""
    try:
        if camera.DcSetIntValue(handle, LX_CAMERA_FEATURE.LX_INT_RGBD_ALIGN_MODE,
                                LX_RGBD_ALIGN_MODE.DEPTH_TO_RGB) != LX_STATE.LX_SUCCESS:
            print("[warn] 设置 RGBD 对齐失败, 回退内参投影")
            return False
        camera.DcSetBoolValue(handle, LX_CAMERA_FEATURE.LX_BOOL_ENABLE_SYNC_FRAME, True)
        print("[depth3d] RGBD 对齐已开启 (DEPTH_TO_RGB) + 强制帧同步")
        return True
    except Exception as e:
        print(f"[warn] RGBD 对齐设置异常 ({e}), 回退内参投影")
        return False


def build_sdk_depth_map(points, intrinsics):
    """SDK 点云 (H,W,3) → 逐像素深度图 (深度内参反投影). 对齐不可用时回退。"""
    dh, dw = points.shape[:2]                # points[depth_height, depth_width, 3]
    xyz = points.reshape(-1, 3).astype(np.float32)
    fx, fy, cx, cy = intrinsics
    return shelf_viz.build_depth_map(xyz, img_w=dw, img_h=dh, fx=fx, fy=fy, cx=cx, cy=cy)


def attach_anchor_3d(dets, points, depth_map, depth_intr, w, h, aligned):
    """给每个检测的关键点附加 anchor_3d。

    对齐模式 (aligned=True, 相机 DEPTH_TO_RGB): 点云与 RGB 同分辨率同索引,
    关键点像素 (u,v) 直接查 points[v,u], 该像素无有效深度时局部窗口邻域采样。
    内参回退 (aligned=False): RGB 坐标按分辨率比例映射到深度图, 5×5 中值查表。
    """
    for d in dets:
        a3d = []
        for u, v, c in d["kpts"]:
            if not (0 <= u < w and 0 <= v < h):
                a3d.append(None)
            elif aligned and points is not None:
                a3d.append(shelf_viz.lookup_xyz(points, int(round(u)), int(round(v))))
            elif depth_map is not None and depth_intr is not None:
                dh, dw = depth_map.shape
                fx, fy, cx, cy = depth_intr
                du = int(round(float(u) * dw / w))
                dv = int(round(float(v) * dh / h))
                a3d.append(shelf_viz.get_anchor_3d(du, dv, depth_map, fx=fx, fy=fy, cx=cx, cy=cy))
            else:
                a3d.append(None)
        d["anchor_3d"] = a3d
    return dets


def draw_detections(img_bgr, dets, points=None, depth_map=None, depth_intr=None, aligned=True):
    """v4 风格绘制 (同 PC 端). anchor_3d 优先取 det 已附的值 (attach_anchor_3d 算出),
    否则按对齐点云 / 内参投影现场算。全缺 → 2D 仅画关键点, XYZ 显示 '--'。"""
    img = img_bgr.copy()
    h, w = img.shape[:2]
    if not dets:
        return img
    best = max(dets, key=lambda d: d["conf"])
    keypoints = []
    a3ds = best.get("anchor_3d") or []
    for i, p in enumerate(best["kpts"]):
        u, v, c = p
        if not (0 <= u < w and 0 <= v < h):
            continue
        a3d = a3ds[i] if i < len(a3ds) else None
        if a3d is None:
            if aligned and points is not None:
                a3d = shelf_viz.lookup_xyz(points, int(round(u)), int(round(v)))
            elif depth_map is not None and depth_intr is not None:
                dh, dw = depth_map.shape
                fx, fy, cx, cy = depth_intr
                du = int(round(float(u) * dw / w))
                dv = int(round(float(v) * dh / h))
                a3d = shelf_viz.get_anchor_3d(du, dv, depth_map, fx=fx, fy=fy, cx=cx, cy=cy)
        keypoints.append({
            "pixel_uv": [int(round(u)), int(round(v))],
            "anchor_3d": a3d,
            "confidence": float(c),
        })
    return shelf_viz.draw_keypoints(img, keypoints)


def main():
    parser = argparse.ArgumentParser(description="蓝芯 MRDVS 相机实时视频流推理")
    parser.add_argument("--dll", default=None, help="LxCameraApi 动态库路径 (默认自动查找)")
    parser.add_argument("--ip", default=None, help="相机 IP")
    parser.add_argument("--sn", default=None, help="相机序列号")
    parser.add_argument("--index", type=int, default=None, help="按序号打开 (0 起)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=".rknn 模型路径")
    parser.add_argument("--conf", type=float, default=0.7)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--cores", type=int, default=3,
                        help="NPU 核: 1=单核 3=双核(默认) 7=三核 0=自动")
    parser.add_argument("--color", choices=["bgr", "rgb"], default="bgr",
                        help="SDK RGB 流通道顺序 (示例程序直接 imshow 不作转换, 默认 bgr; "
                             "若画面偏蓝/偏橙改 --color rgb)")
    parser.add_argument("--depth3d", action="store_true",
                        help="同时开 3D 深度流, 关键点深度查表出 anchor_3d "
                             "(SDK RGBD 对齐点云优先, 内参投影回退)")
    parser.add_argument("--intrinsics", default=None,
                        help="深度内参 JSON (query_camera_intrinsics.py 输出); "
                             "缺省按相机 SN 自动找 camera_intrinsics_<sn>.json, 再在线查询")
    parser.add_argument("--json", action="store_true", help="逐帧向 stdout 打印检测 JSON")
    parser.add_argument("--no-display", action="store_true", help="无显示环境纯推理")
    parser.add_argument("--save", default=None, help="录制视频到 <path> (如 out.mp4)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"模型不存在: {args.model}")
    dll_path = args.dll or find_sdk_lib()
    if dll_path is None:
        sys.exit("找不到 LxCameraApi 动态库, 请用 --dll 指定路径")

    # ── NPU 模型 (与 infer_camera.py 相同) ──
    from rknnlite.api import RKNNLite
    rknn = RKNNLite()
    if rknn.load_rknn(args.model) != 0:
        sys.exit(f"load_rknn 失败: {args.model}")
    if rknn.init_runtime(core_mask=args.cores) != 0:
        sys.exit(f"init_runtime 失败 (core_mask={args.cores})")

    # ── 相机 ──
    camera, handle, dev_info = open_camera(dll_path, args.ip, args.sn, args.index)
    cam_sn = _bstr(getattr(dev_info, "sn", None), None)

    # 开 2D (必) / 3D 深度 (可选) 流
    if camera.DcSetBoolValue(handle, LX_CAMERA_FEATURE.LX_BOOL_ENABLE_2D_STREAM, True) \
            != LX_STATE.LX_SUCCESS:
        sys.exit("开启 2D 流失败 (LX_BOOL_ENABLE_2D_STREAM)")
    if args.depth3d:
        if camera.DcSetBoolValue(handle, LX_CAMERA_FEATURE.LX_BOOL_ENABLE_3D_DEPTH_STREAM, True) \
                != LX_STATE.LX_SUCCESS:
            sys.exit("开启 3D 深度流失败 (LX_BOOL_ENABLE_3D_DEPTH_STREAM)")
        setup_rgbd_align(camera, handle)   # 对齐失败会自动回退内参投影
        # RGB/深度帧同步必须无条件开, 否则多路流帧不同步 → getFrame 频繁 FRAME_ID_NOT_MATCH
        if camera.DcSetBoolValue(handle, LX_CAMERA_FEATURE.LX_BOOL_ENABLE_SYNC_FRAME, True) \
                == LX_STATE.LX_SUCCESS:
            print("[sync] 已强制 RGB/深度帧同步 (ENABLE_SYNC_FRAME)")
        else:
            print("[warn] 强制帧同步设置失败, 靠取帧重试兜底")
    depth_intr = get_depth_intrinsics(camera, handle, args.intrinsics, cam_sn) \
        if args.depth3d else None
    if args.depth3d and depth_intr is None:
        print("[warn] 拿不到深度内参, 对齐不可用时无法内参投影回退")
    elif depth_intr is not None:
        print(f"深度内参: fx={depth_intr[0]:.3f} fy={depth_intr[1]:.3f} "
              f"cx={depth_intr[2]:.3f} cy={depth_intr[3]:.3f}")

    if camera.DcStartStream(handle) != LX_STATE.LX_SUCCESS:
        sys.exit("DcStartStream 失败")
    print(f"模型: {args.model}  cores={args.cores}  流已启动, Ctrl+C / q 退出。")

    writer = None  # 有 --save 时在拿到首帧后按实际帧尺寸创建

    fps_count, fps_time, fps = 0, time.time(), 0.0
    frame_fail = 0
    pc_mode = None   # getPointCloud 输出: 'aligned'(对齐到RGB) | 'depth'(内参投影), 首帧按形状判定
    try:
        while True:
            # ── 取一帧 (SDK: getFrame 触发 LX_CMD_GET_NEW_FRAME 并取数据指针) ──
            ret, data_ptr = camera.getFrame(handle)
            if ret != LX_STATE.LX_SUCCESS:
                # FRAME_ID_NOT_MATCH(帧不同步)/MULTI_MACHINE 是临时错误, 官方示例直接重取, 不计失败
                transient = ret in (
                    getattr(LX_STATE, "LX_E_FRAME_ID_NOT_MATCH", -1),
                    getattr(LX_STATE, "LX_E_FRAME_MULTI_MACHINE", -1))
                if transient:
                    time.sleep(0.03)
                    continue
                frame_fail += 1
                if frame_fail > 30:
                    print(f"[warn] 连续 30 次取帧失败, 停止")
                    break
                time.sleep(0.05)
                continue
            frame_fail = 0

            ret, rgb = camera.getRGBImage(data_ptr)
            if ret != LX_STATE.LX_SUCCESS or rgb is None:
                continue
            rgb = np.ascontiguousarray(rgb.copy())   # 脱离 SDK 缓冲 (下次 getFrame 会被复用)
            if rgb.ndim == 3 and rgb.shape[2] == 1:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
            if args.color == "rgb":                  # SDK 若是 RGB 顺序, 转回 BGR 对齐推理管线
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            frame_h, frame_w = rgb.shape[:2]
            if writer is None and args.save:
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                         20, (frame_w, frame_h))

            # ── 与离线推理完全相同的预处理/推理/解码 ──
            lb, _, _ = letterbox(rgb)
            blob = np.ascontiguousarray(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB))[None]
            t0 = time.perf_counter()
            outputs = rknn.inference(inputs=[blob])
            infer_ms = (time.perf_counter() - t0) * 1000
            dets = decode_yolopose(outputs, rgb.shape[:2], args.conf, args.iou)

            # ── 可选深度查表: SDK RGBD 对齐点云优先, 内参投影回退 ──
            points = None
            depth_map = None
            if args.depth3d:
                state, points = camera.getPointCloud(handle)
                if state != LX_STATE.LX_SUCCESS or points is None or points.size == 0:
                    points = None
                else:
                    if pc_mode is None:
                        ph, pw = points.shape[:2]
                        pc_mode = 'aligned' if (ph == frame_h and pw == frame_w) else 'depth'
                        if pc_mode == 'depth':
                            print(f"[warn] RGBD 对齐未生效: 点云 {pw}x{ph} != RGB "
                                  f"{frame_w}x{frame_h}, 用内参投影回退")
                    if pc_mode == 'depth' and depth_intr is not None:
                        depth_map = build_sdk_depth_map(points, depth_intr)
                dets = attach_anchor_3d(dets, points, depth_map, depth_intr,
                                        frame_w, frame_h, aligned=(pc_mode == 'aligned'))

            # ── FPS ──
            fps_count += 1
            now = time.time()
            if now - fps_time >= 1.0:
                fps = fps_count / (now - fps_time)
                fps_count, fps_time = 0, now

            # ── 上位机 JSON (含 anchor_3d, 与离线 detections.json 对齐) ──
            if args.json:
                payload = {
                    "t": round(time.time(), 3),
                    "fps": round(fps, 2),
                    "infer_ms": round(infer_ms, 1),
                    "dets": [{
                        "box": d["box"].round(1).tolist(),
                        "conf": round(d["conf"], 4),
                        "keypoints": d["kpts"].round(1).tolist(),
                        "anchor_3d": [None if a is None else [round(v, 1) for v in a]
                                      for a in d.get("anchor_3d", [None, None])],
                    } for d in dets],
                }
                print(json.dumps(payload, ensure_ascii=False), flush=True)

            # ── 显示 / 录制 ──
            vis = draw_detections(rgb, dets, points, depth_map, depth_intr,
                                  aligned=(pc_mode == 'aligned'))
            cv2.putText(vis, f"FPS:{fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if writer is not None:
                writer.write(vis)
            if not args.no_display:
                cv2.imshow("MRDVS Shelf Pose", vis)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        camera.DcStopStream(handle)
        camera.DcCloseDevice(handle)
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        rknn.release()
        print("已退出。")


if __name__ == '__main__':
    main()
