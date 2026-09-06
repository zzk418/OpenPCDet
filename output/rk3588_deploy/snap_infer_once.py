#!/usr/bin/env python3
"""在线抓 1 帧 → 存对齐 png+pcd → 同离线管线重推理 → 可视化 + 一致性核对。

针对"上板 fp16 精度崩"排查的第二步 (在线输入核对):
第一步 tools/verify_fp16_export.py 已证 fp16 转换无损; 精度崩若不在转换, 最大嫌疑
是 **在线预处理/对齐 ≠ 离线**。本脚本把在线抓到的帧存成「对齐的 png + pcd」, 再用
与离线完全相同的一段代码 (letterbox 640 → rknn → decode_yolopose → 深度查表) 对
①内存里的在线帧 和 ②存盘后重读的文件 各跑一遍, 断言两者结果一致。

一致 → 存盘的 png+pcd 忠实代表在线帧, 拿到 PC 用 offline 管线 (对齐直查路径) 复现,
      即"在线 == 离线"成立; 关键点 3D (anchor_3d) 以相机内参对齐直查, 不靠近似投影。
不一致 → 存盘/重读有损 (不该发生, 说明写 pcd 或读回有 bug)。

输出 (默认 output/rk3588_deploy/snap/):
  <stem>.png         RGB 原图 (无损, BGR 存)
  <stem>.pcd         对齐点云 (有序, FIELDS x y z rgb, HEIGHT=H — 与生产 dump 同格式)
  <stem>_depth.png   深度图 uint16 mm (点云 z, 与 RGB 同分辨率同索引)
  <stem>_viz.png     可视化: 关键点 + P1/P2 3D XYZ 图例 (v4 风格)
  <stem>.json        dets / center_xyz / 在线 vs 文件一致性结论

用法 (板子上执行):
  python snap_infer_once.py                     # 自动发现第一台相机
  python snap_infer_once.py --ip 192.168.2.151
  python snap_infer_once.py --sn 519889C9A2A6468E
  python snap_infer_once.py --out /root/snap --stem 20260906_A

依赖: rknn-toolkit-lite2, numpy, opencv-python, lx_camera_py + SDK 动态库。
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

import shelf_viz                          # lookup_xyz / draw_keypoints
from infer_camera import letterbox, decode_yolopose, DEFAULT_MODEL
from infer_camera_sdk import (            # 与生产 shelf_pos_service 同一套 SDK 封装
    LX_STATE, LX_CAMERA_FEATURE,
    open_camera, setup_rgbd_align, get_depth_intrinsics,
    attach_anchor_3d, build_sdk_depth_map)
from query_camera_intrinsics import find_sdk_lib, _bstr
from shelf_pos_service import write_pcd, load_pcd   # 存/读 有序 pcd (与生产 dump 一致)


def grab_aligned_frame(camera, handle, max_try=40, color="bgr"):
    """鲁棒抓 1 帧「对齐的」RGB + 点云。返回 (rgb, points, aligned)。

    反复取帧直到拿到 RGB 且点云形状 == RGB (即相机 DEPTH_TO_RGB 对齐已生效),
    FRAME_ID_NOT_MATCH / MULTI_MACHINE 等临时错重取不判失败 (与 _grab_frame 一致)。
    max_try 内拿不到对齐帧 → 返回最后一帧 (aligned=False), 调用方走内参投影回退。"""
    transient = set()
    for name in ("LX_E_FRAME_ID_NOT_MATCH", "LX_E_FRAME_MULTI_MACHINE"):
        v = getattr(LX_STATE, name, None)
        if v is not None:
            transient.add(v)
    last_rgb = last_points = None
    for i in range(max_try):
        ret, data_ptr = camera.getFrame(handle)
        if ret != LX_STATE.LX_SUCCESS:
            if ret in transient:
                time.sleep(0.03)
                continue
            if getattr(LX_STATE, "LX_E_RECONNECTING", None) == ret:
                time.sleep(0.5)
                continue
            time.sleep(0.05)
            continue
        ret, rgb = camera.getRGBImage(data_ptr)
        if ret != LX_STATE.LX_SUCCESS or rgb is None:
            time.sleep(0.03)
            continue
        rgb = np.ascontiguousarray(rgb.copy())   # 脱离 SDK 缓冲
        if rgb.ndim == 3 and rgb.shape[2] == 1:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
        if color == "rgb":
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        points = None
        try:
            state, points = camera.getPointCloud(handle)
            if state != LX_STATE.LX_SUCCESS or points is None or points.size == 0:
                points = None
        except Exception:
            points = None
        last_rgb, last_points = rgb, points
        if points is not None and points.ndim == 3 \
                and points.shape[:2] == rgb.shape[:2]:
            return rgb, points, True
    return last_rgb, last_points, False


def infer_dets(rknn, rgb, points, depth_intr, conf, iou):
    """对 (RGB, 点云) 跑完整推理 → dets (带 anchor_3d)。

    与 shelf_pos_service.ShelfEngine._infer_on 同一条链 (对齐直查优先, 内参投影回退):
      aligned  → attach_anchor_3d(aligned=True) 用 lookup_xyz 直接取 points[v,u]
      非对齐   → build_sdk_depth_map + attach_anchor_3d(aligned=False) 内参投影
    """
    fh, fw = rgb.shape[:2]
    lb, _, _ = letterbox(rgb)
    blob = np.ascontiguousarray(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB))[None]
    outputs = rknn.inference(inputs=[blob])
    dets = decode_yolopose(outputs, (fh, fw), conf, iou)

    depth_map = None
    aligned = False
    if points is not None and points.ndim == 3 and points.shape[2] == 3:
        ph, pw = points.shape[:2]
        aligned = (ph == fh and pw == fw)
        if not aligned and depth_intr is not None:
            depth_map = build_sdk_depth_map(points, depth_intr)
    return attach_anchor_3d(dets, points, depth_map, depth_intr, fw, fh, aligned)


def center_from_dets(dets):
    """最高置信度实例两角点 3D 取均值 → 货架中心点 [x,y,z] mm; 失败返回 None。"""
    if not dets:
        return None
    best = max(dets, key=lambda d: d["conf"])
    a3d = [a for a in (best.get("anchor_3d") or []) if a is not None]
    if len(a3d) < 2:
        return None
    return [round((a3d[0][i] + a3d[1][i]) / 2.0, 1) for i in range(3)]


def to_json_dets(dets):
    return [{
        "box": d["box"].round(1).tolist(),
        "conf": round(d["conf"], 4),
        "keypoints": d["kpts"].round(1).tolist(),
        "anchor_3d": [None if a is None else [round(v, 1) for v in a]
                      for a in d.get("anchor_3d", [None, None])],
    } for d in dets]


def draw(bgr, dets):
    """v4 风格绘制最高置信度实例关键点 + 3D XYZ (同 infer_image.py)。"""
    img = bgr.copy()
    if not dets:
        return img
    h, w = img.shape[:2]
    best = max(dets, key=lambda d: d["conf"])
    kps = []
    a3ds = best.get("anchor_3d") or []
    for i, p in enumerate(best["kpts"]):
        u, v, c = p
        if not (0 <= u < w and 0 <= v < h):
            continue
        kps.append({"pixel_uv": [int(round(u)), int(round(v))],
                    "anchor_3d": a3ds[i] if i < len(a3ds) else None,
                    "confidence": float(c)})
    return shelf_viz.draw_keypoints(img, kps)


def main():
    ap = argparse.ArgumentParser(description="在线抓 1 帧对齐 png+pcd 并同离线管线核对")
    ap.add_argument("--dll", default=None, help="LxCameraApi 动态库路径 (默认自动查找)")
    ap.add_argument("--ip", default=None)
    ap.add_argument("--sn", default=None)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--conf", type=float, default=0.7)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--cores", type=int, default=3)
    ap.add_argument("--color", choices=["bgr", "rgb"], default="bgr")
    ap.add_argument("--intrinsics", default=None,
                    help="深度内参 JSON; 缺省按 SN 自动找 camera_intrinsics_<sn>.json, 再在线查询")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "snap"))
    ap.add_argument("--stem", default=None, help="输出文件名 stem (默认 snap_YYYYMMDD_HHMMSS)")
    ap.add_argument("--max-try", type=int, default=40, help="抓对齐帧最大尝试次数")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"模型不存在: {args.model}")
    dll_path = args.dll or find_sdk_lib()
    if dll_path is None:
        sys.exit("找不到 LxCameraApi 动态库, 请用 --dll 指定路径")
    stem = args.stem or time.strftime("snap_%Y%m%d_%H%M%S")
    os.makedirs(args.out, exist_ok=True)

    # ── NPU ──
    from rknnlite.api import RKNNLite
    rknn = RKNNLite()
    if rknn.load_rknn(args.model) != 0:
        sys.exit(f"load_rknn 失败: {args.model}")
    if rknn.init_runtime(core_mask=args.cores) != 0:
        sys.exit(f"init_runtime 失败 (cores={args.cores})")

    camera = handle = None
    try:
        # ── 相机 + 2D/3D 流 + RGBD 对齐 + 帧同步 ──
        camera, handle, dev_info = open_camera(dll_path, args.ip, args.sn, args.index)
        cam_sn = _bstr(getattr(dev_info, "sn", None), None)
        if camera.DcSetBoolValue(handle, LX_CAMERA_FEATURE.LX_BOOL_ENABLE_2D_STREAM, True) \
                != LX_STATE.LX_SUCCESS:
            sys.exit("开启 2D 流失败")
        if camera.DcSetBoolValue(handle, LX_CAMERA_FEATURE.LX_BOOL_ENABLE_3D_DEPTH_STREAM, True) \
                != LX_STATE.LX_SUCCESS:
            sys.exit("开启 3D 深度流失败")
        setup_rgbd_align(camera, handle)   # DEPTH_TO_RGB; 失败回退内参投影
        if camera.DcSetBoolValue(handle, LX_CAMERA_FEATURE.LX_BOOL_ENABLE_SYNC_FRAME, True) \
                == LX_STATE.LX_SUCCESS:
            print("[sync] 已强制 RGB/深度帧同步")
        else:
            print("[warn] 强制帧同步设置失败, 靠取帧重试兜底")
        depth_intr = get_depth_intrinsics(camera, handle, args.intrinsics, cam_sn)
        if depth_intr is None:
            print("[warn] 拿不到深度内参, 对齐不可用时无法内参投影回退")
        else:
            print(f"深度内参: fx={depth_intr[0]:.3f} fy={depth_intr[1]:.3f} "
                  f"cx={depth_intr[2]:.3f} cy={depth_intr[3]:.3f}")

        if camera.DcStartStream(handle) != LX_STATE.LX_SUCCESS:
            sys.exit("DcStartStream 失败")
        print("流已启动, 抓取 1 帧对齐输入…")

        rgb, points, aligned = grab_aligned_frame(camera, handle, args.max_try, args.color)
        if rgb is None:
            sys.exit("抓不到 RGB 帧 (检查相机流)")
        if points is None:
            sys.exit("抓不到点云 (检查 3D 深度流是否开启 / 相机是否支持)")
        print(f"抓到帧: RGB {rgb.shape[:2][::-1]}  点云 {points.shape[:2][::-1]}  "
              f"{'对齐✓' if aligned else '未对齐(内参投影回退)'}")

        # ── 存盘: 对齐 png + pcd + 深度图 ──
        png_path = os.path.join(args.out, f"{stem}.png")
        pcd_path = os.path.join(args.out, f"{stem}.pcd")
        depth_path = os.path.join(args.out, f"{stem}_depth.png")
        cv2.imwrite(png_path, rgb)
        write_pcd(pcd_path, points, rgb)
        z = points[:, :, 2]
        valid = np.isfinite(z) & (z > 0)
        depth = np.zeros(z.shape, np.uint16)
        depth[valid] = np.clip(z[valid], 0, 65535).astype(np.uint16)
        cv2.imwrite(depth_path, depth)
        print(f"已存: {png_path}\n      {pcd_path}\n      {depth_path}")

        # ── ① 在线内存帧推理 ──
        dets_online = infer_dets(rknn, rgb, points, depth_intr, args.conf, args.iou)
        center_online = center_from_dets(dets_online)

        # ── ② 存盘文件重读, 同离线管线重推理 ──
        rgb2 = cv2.imread(png_path)
        xyz, _, org = load_pcd(pcd_path)
        points2 = xyz.reshape(org[0], org[1], 3) if org else None
        dets_file = infer_dets(rknn, rgb2, points2, depth_intr, args.conf, args.iou)
        center_file = center_from_dets(dets_file)

        # ── 一致性核对 (在线 == 离线) ──
        same = True
        if center_online is None and center_file is None:
            center_d = None
        elif center_online is None or center_file is None:
            same = False
            center_d = None
        else:
            center_d = [round(abs(center_online[i] - center_file[i]), 3) for i in range(3)]
            same = max(center_d) <= 0.5   # 点云浮点/PNG 重读不会引入 >0.5mm 差异

        # ── 可视化 ──
        viz_path = os.path.join(args.out, f"{stem}_viz.png")
        cv2.imwrite(viz_path, draw(rgb, dets_online))

        summary = {
            "stem": stem,
            "aligned": aligned,
            "rgb_shape": rgb.shape[:2][::-1],
            "points_shape": points.shape[:2][::-1],
            "online": {"center_xyz": center_online, "dets": to_json_dets(dets_online)},
            "from_file": {"center_xyz": center_file, "dets": to_json_dets(dets_file)},
            "center_diff_mm": center_d,
            "online_equals_offline": bool(same),
            "viz": os.path.basename(viz_path),
        }
        json_path = os.path.join(args.out, f"{stem}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 60)
        print(f"在线中心 XYZ (mm): {center_online}")
        print(f"文件重读中心 XYZ  (mm): {center_file}")
        print(f"差异 (mm): {center_d}")
        print(f"在线 == 离线: {'✓ PASS' if same else '✗ FAIL'}")
        if not same:
            print("  !!! 存盘/重读引入差异, 检查 write_pcd/load_pcd 或点云 dtype")
        print(f"可视化: {viz_path}")
        print(f"摘要: {json_path}")
        print("=" * 60)
        sys.exit(0 if same else 1)
    finally:
        if camera is not None and handle is not None:
            try:
                camera.DcStopStream(handle)
            except Exception:
                pass
            try:
                camera.DcCloseDevice(handle)
            except Exception:
                pass
        try:
            rknn.release()
        except Exception:
            pass


if __name__ == "__main__":
    main()
