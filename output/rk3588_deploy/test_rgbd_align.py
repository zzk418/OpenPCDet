#!/usr/bin/env python3
"""MRDVS(LxCamera) RGB-D 对齐数据专项测试 — headless, 不依赖显示器。

目标: 验证"对齐数据"的获取。所谓对齐 = 相机内置 DEPTH_TO_RGB 对齐开启后,
深度图 / 强度图 / 点云全部重采样到 RGB 分辨率, 与 RGB **同分辨率、同像素索引**,
同时强制帧同步保证 RGB 和深度属于同一帧。

用法 (RK3588 板, root):
    python3 test_rgbd_align.py [IP] [帧数]
    # 默认 IP=192.168.2.150, 帧数=10

输出 (默认 /root/rk3588_deploy/input_check/, 与服务 --dump-frames 一致):
    rgb_000.jpg / depth_000.png / pc_000.pcd     # 逐帧对齐三件套 (点云 pcd, 与 imgs/ 同格式)
    report.json                                   # 对齐判定 + 每帧分辨率/时间戳

判定标准 (与 infer_camera_sdk.py 的 pc_mode 检测一致):
    [OK]  点云尺寸 == RGB 尺寸  → 对齐生效, 关键点像素 (u,v) 可直接查 points[v,u]
    [FAIL] 点云尺寸 != RGB 尺寸 → 对齐未生效, 只能走内参投影回退
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

from LxCameraSDK.lx_camera_api import LxCamera
from LxCameraSDK.lx_camera_define import (
    LX_STATE, LX_OPEN_MODE, LX_CAMERA_FEATURE, LX_RGBD_ALIGN_MODE, LX_CAMERA_WORK_MODE,
    LX_ALGORITHM_MODE)
from shelf_pos_service import write_pcd   # 点云存 pcd (与 imgs/ 同格式, 不存 npy)

LIB = "/opt/MRDVS/lib/libLxCameraApi.so"   # install.sh 安装位置


def _bstr(raw, fallback=""):
    """c_char32 字段 → 干净字符串 (去掉尾部 \x00 再 strip)。"""
    try:
        return raw.decode("utf-8").rstrip("\x00").strip()
    except Exception:
        return fallback


def setup_rgbd_align(camera, handle):
    """开启 DEPTH_TO_RGB 对齐 + 强制帧同步。

    必须在 DcStartStream 之前调用 (对齐"停流后可修改"; 常开 WORK_FOREVER 模式下
    设置会返回 -24, 需先切 KEEP_HEARTBEAT)。返回 True 表示对齐设置成功。
    """
    # 相机处于常开模式时, 对齐/ROI/多机等参数一律不允许设置 → 先切到心跳模式
    state, wm = camera.DcGetIntValue(handle, LX_CAMERA_FEATURE.LX_INT_WORK_MODE)
    if state == LX_STATE.LX_SUCCESS and wm.cur_value == LX_CAMERA_WORK_MODE.WORK_FOREVER.value:
        print("  [info] 相机处于常开(WORK_FOREVER)模式, 先切回 KEEP_HEARTBEAT 以便设置对齐")
        r = camera.DcSetIntValue(handle, LX_CAMERA_FEATURE.LX_INT_WORK_MODE,
                                 LX_CAMERA_WORK_MODE.KEEP_HEARTBEAT)
        if r != LX_STATE.LX_SUCCESS:
            print(f"  [warn] 切换工作模式失败 ret={r}, 继续尝试")
    r = camera.DcSetIntValue(handle, LX_CAMERA_FEATURE.LX_INT_RGBD_ALIGN_MODE,
                             LX_RGBD_ALIGN_MODE.DEPTH_TO_RGB)
    if r != LX_STATE.LX_SUCCESS:
        print(f"  [FAIL] 设置 DEPTH_TO_RGB 对齐失败 ret={r} "
              f"({camera.DcGetErrorString(r)}) — 对齐数据拿不到")
        return False
    camera.DcSetBoolValue(handle, LX_CAMERA_FEATURE.LX_BOOL_ENABLE_SYNC_FRAME, True)
    return True


def main():
    ap = argparse.ArgumentParser(description="MRDVS RGB-D 对齐数据测试")
    ap.add_argument("ip", nargs="?", default="192.168.2.150", help="相机 IP")
    ap.add_argument("frames", nargs="?", type=int, default=10, help="拉取帧数")
    ap.add_argument("--out", default="/root/rk3588_deploy/input_check", help="输出目录 (默认和服务 --dump-frames 一致)")
    ap.add_argument("--dll", default=LIB)
    args = ap.parse_args()

    if not os.path.isfile(args.dll):
        sys.exit(f"[FAIL] {args.dll} 不存在 —— 先跑 install.sh 装 SDK")

    camera = LxCamera(args.dll)
    camera.DcSetInfoOutput(0, True, "")          # 抑制 SDK 刷屏日志
    print(f"[OK] SDK 加载成功: {camera.DcGetApiVersion()}")

    # ── 按 IP 直开 (绕开枚举 decode 问题, 也适合相机被占用不回应广播的场景) ──
    state, handle, info = camera.DcOpenDevice(LX_OPEN_MODE.OPEN_BY_IP, args.ip)
    if state != LX_STATE.LX_SUCCESS:
        sys.exit(f"[FAIL] DcOpenDevice({args.ip}) 失败: {state} ({camera.DcGetErrorString(state)})")
    print(f"[OK] 已打开: {_bstr(info.name, '?')}  SN={_bstr(info.sn, '?')}  IP={_bstr(info.ip)}")

    # ── 关内置算法流 (algor 会占一路流, 和 rgb+depth 抢带宽 → 收不到流) ──
    r = camera.DcSetIntValue(handle, LX_CAMERA_FEATURE.LX_INT_ALGORITHM_MODE,
                             LX_ALGORITHM_MODE.MODE_ALL_OFF.value)
    print(f"[algo] 内置算法 {'已关' if r == LX_STATE.LX_SUCCESS else f'关闭失败({r})'}")

    # ── 开 2D + 3D 深度流 (对齐数据两路必需) ──
    if camera.DcSetBoolValue(handle, LX_CAMERA_FEATURE.LX_BOOL_ENABLE_2D_STREAM, True) \
            != LX_STATE.LX_SUCCESS:
        sys.exit("[FAIL] 开启 2D 流失败")
    if camera.DcSetBoolValue(handle, LX_CAMERA_FEATURE.LX_BOOL_ENABLE_3D_DEPTH_STREAM, True) \
            != LX_STATE.LX_SUCCESS:
        sys.exit("[FAIL] 开启 3D 深度流失败")

    aligned = setup_rgbd_align(camera, handle)   # ★ 对齐开关 (停流时设置)

    # ── 取内参作参考信息 (对齐成功与否的旁证) ──
    ret, intr3d, _ = camera.get3DIntricParam(handle)
    if ret == LX_STATE.LX_SUCCESS and intr3d is not None:
        print(f"[info] 深度内参: fx={intr3d[0]:.3f} fy={intr3d[1]:.3f} "
              f"cx={intr3d[2]:.3f} cy={intr3d[3]:.3f}")

    if camera.DcStartStream(handle) != LX_STATE.LX_SUCCESS:
        sys.exit("[FAIL] DcStartStream 失败")
    print(f"[OK] 三路流已开启 (对齐={'ON' if aligned else 'OFF'}), 开始拉帧…")

    os.makedirs(args.out, exist_ok=True)
    report = {"ip": args.ip, "aligned_set": aligned, "frames": []}
    rgb_n = depth_n = pc_n = 0
    for i in range(args.frames):
        s, ptr = camera.getFrame(handle)
        if s != LX_STATE.LX_SUCCESS:
            print(f"  frame {i}: getFrame ret={s}, skip")
            time.sleep(0.2)
            continue

        s2, rgb = camera.getRGBImage(ptr)
        s3, depth = camera.getDepthImage(ptr)
        s4, pc = camera.getPointCloud(handle)    # 需深度流开启

        entry = {"idx": i, "rgb": None, "depth": None, "pc": None, "aligned": None}
        if s2 == LX_STATE.LX_SUCCESS and rgb is not None:
            rgb = np.ascontiguousarray(rgb.copy())   # 脱离 SDK 缓冲 (下次 getFrame 会复用)
            if rgb.ndim == 3 and rgb.shape[2] == 1:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
            rgb_n += 1
            entry["rgb"] = {"shape": list(rgb.shape)}
            cv2.imwrite(f"{args.out}/rgb_{i:03d}.jpg", rgb)
            print(f"  frame {i}: RGB {rgb.shape} saved")
        if s3 == LX_STATE.LX_SUCCESS and depth is not None:
            depth = np.ascontiguousarray(depth.copy())
            depth_n += 1
            entry["depth"] = {"shape": list(depth.shape), "dtype": str(depth.dtype)}
            cv2.imwrite(f"{args.out}/depth_{i:03d}.png", depth)
            print(f"  frame {i}: Depth {depth.shape} {depth.dtype} saved")
        if s4 == LX_STATE.LX_SUCCESS and pc is not None and pc.size > 0:
            pc_n += 1
            entry["pc"] = {"shape": list(pc.shape)}
            write_pcd(f"{args.out}/pc_{i:03d}.pcd", pc, rgb if rgb is not None else None)

            # ── 对齐判定: 点云尺寸 == RGB 尺寸 → 同像素索引, 可 (u,v) 直接查 ──
            if entry["rgb"]:
                ph, pw = pc.shape[:2]
                rh, rw = entry["rgb"]["shape"][:2]
                ok = (ph == rh and pw == rw)
                entry["aligned"] = bool(ok)
                if not ok and aligned:
                    print(f"  [warn] 对齐未生效: 点云 {pw}x{ph} != RGB {rw}x{rh} "
                          f"(常见: 固件/SDK 版本旧, 见 FAQ #14)")
                # 图像中心像素的 3D 坐标样例 (对齐时中心像素 ↔ 同一物点)
                if ok:
                    cu, cv = rw // 2, rh // 2
                    p = pc[cv, cu]
                    if abs(p[2]) > 0:
                        print(f"  [OK]  对齐: 点云与 RGB 同分辨率, 中心像素({cu},{cv}) "
                              f"XYZ=({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})mm")
        report["frames"].append(entry)
        time.sleep(0.1)

    # 帧同步旁证: 比较首帧 RGB/D 的传感器时间戳 (新 SDK 直接返回 FrameInfo, 旧版是指针)
    if ptr is not None:
        try:
            fr = ptr.contents if hasattr(ptr, "contents") else ptr
            rts, dts = fr.rgb_data.sensor_timestamp, fr.depth_data.sensor_timestamp
            print(f"[info] 末帧时间戳: RGB={rts} depth={dts} "
                  f"差={abs(int(rts) - int(dts))} (SYNC_FRAME=ON 应同帧)")
            report["last_frame_ts"] = {"rgb": int(rts), "depth": int(dts)}
        except Exception as e:
            print(f"[info] 读帧时间戳跳过 ({e}), 不影响判定")

    aligned_n = sum(1 for f in report["frames"] if f.get("aligned"))
    report["result"] = ("ALIGNED" if aligned and aligned_n == rgb_n and rgb_n > 0
                        else "NOT_ALIGNED")
    with open(f"{args.out}/report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    camera.DcStopStream(handle)
    camera.DcCloseDevice(handle)
    print(f"[DONE] RGB {rgb_n} 帧 / Depth {depth_n} 帧 / 点云 {pc_n} 帧 → {args.out}")
    print(f"      对齐判定: {report['result']}"
          + (" — 像素级对齐, 关键点可直查点云" if report["result"] == "ALIGNED"
             else " — 走内参投影回退 (参考 infer_camera_sdk.py --depth3d 的 depth 分支)"))
    sys.exit(0 if report["result"] == "ALIGNED" else 1)


if __name__ == '__main__':
    main()
