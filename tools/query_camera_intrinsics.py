#!/usr/bin/env python3
"""从 Eagle-M4 Mega 相机直接读取出厂标定内参 (权威值)。

使用蓝芯 MRDVS LxCamera SDK 的 get3DIntricParam / get2DIntricParam:
    intrinsic[fx, fy, cx, cy], distort[d1, d2, d3, d4, d5]
同时输出 3D 外参矩阵 (get3DTransMatrix)。

用法:
    # 自动发现相机, 按序号打开第一台
    python tools/query_camera_intrinsics.py

    # 指定 IP / 序列号
    python tools/query_camera_intrinsics.py --ip 192.168.100.86
    python tools/query_camera_intrinsics.py --sn <序列号>

    # SDK 动态库路径 (默认自动查找)
    python tools/query_camera_intrinsics.py --dll "/path/to/LxCameraApi.dll"

输出:
    camera_intrinsics_<sn>.json  +  控制台表格

依赖:
    pip install lx_camera_py-1.3.3-py3-none-any.whl
    (仓库: https://github.com/Lanxin-MRDVS/CameraSDK, linux/Sample/python/)
"""
import argparse
import json
import os
import sys
from datetime import datetime


def find_sdk_lib():
    """自动查找 SDK 动态库 (Windows .dll / Linux .so)。"""
    if sys.platform.startswith('win'):
        candidates = [
            r"D:\Program Files\Lanxin-MRDVS\SDK\lib\win_x64\LxCameraApi.dll",
            r"C:\Program Files\Lanxin-MRDVS\SDK\lib\win_x64\LxCameraApi.dll",
        ]
    else:
        candidates = [
            "/opt/Lanxin-MRDVS/SDK/lib/linux_x64/libLxCameraApi.so",
            "/usr/lib/libLxCameraApi.so",
        ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def main():
    parser = argparse.ArgumentParser(description="读取 Eagle-M4 Mega 出厂内参")
    parser.add_argument("--dll", default=None, help="LxCameraApi 动态库路径 (默认自动查找)")
    parser.add_argument("--ip", default=None, help="相机 IP, 如 192.168.100.86")
    parser.add_argument("--sn", default=None, help="相机序列号")
    parser.add_argument("--index", type=int, default=None, help="按序号打开 (0 起)")
    parser.add_argument("--out", default=None, help="JSON 输出路径 (默认 camera_intrinsics_<sn>.json)")
    args = parser.parse_args()

    dll_path = args.dll or find_sdk_lib()
    if dll_path is None:
        sys.exit("找不到 LxCameraApi 动态库, 请用 --dll 指定路径")

    try:
        from LxCameraSDK.lx_camera_define import LX_STATE, LX_OPEN_MODE
        from LxCameraSDK.lx_camera_api import LxCamera
    except ImportError:
        sys.exit(
            "缺少 LxCameraSDK 包。请安装:\n"
            "  pip install <CameraSDK 仓库>/linux/Sample/python/lx_camera_py-1.3.3-py3-none-any.whl\n"
            "  (https://github.com/Lanxin-MRDVS/CameraSDK)")

    camera = LxCamera(dll_path)
    print(f"SDK: {camera.DcGetApiVersion()}, 库: {dll_path}")

    ret, dev_list, dev_num = camera.DcGetDeviceList()
    if ret != LX_STATE.LX_SUCCESS:
        sys.exit(f"DcGetDeviceList 失败: {ret}")
    print(f"发现 {dev_num} 台相机")

    # 确定打开方式和参数
    if args.ip:
        mode, param = LX_OPEN_MODE.OPEN_BY_IP, args.ip
    elif args.sn:
        mode, param = LX_OPEN_MODE.OPEN_BY_SN, args.sn
    else:
        if dev_num == 0:
            sys.exit("未发现相机 (检查网络/供电)")
        idx = args.index if args.index is not None else 0
        dev = dev_list[idx]
        print(f"设备 {idx}: {dev.name.decode('utf-8').strip()}  IP={dev.ip.decode('utf-8').strip()}")
        # 优先按 SN 打开 (INDEX 模式 param 语义不明)
        sn = dev.sn.decode('utf-8').strip()
        if sn:
            mode, param = LX_OPEN_MODE.OPEN_BY_SN, sn
        else:
            mode, param = LX_OPEN_MODE.OPEN_BY_INDEX, str(idx)

    ret, handle, dev_info = camera.DcOpenDevice(mode, param)
    if ret != LX_STATE.LX_SUCCESS:
        sys.exit(f"DcOpenDevice({mode}, {param}) 失败: {ret}")
    try:
        name = dev_info.name.decode('utf-8').strip()
    except Exception:
        name = "?"
    try:
        sn = dev_info.sn.decode('utf-8').strip()
    except Exception:
        sn = args.sn or "unknown"
    print(f"已打开: {name}  SN={sn}")

    out = {"camera": name, "sn": sn, "queried_at": datetime.now().isoformat(timespec='seconds')}

    ret, intr3d, dist3d = camera.get3DIntricParam(handle)
    if ret == LX_STATE.LX_SUCCESS and intr3d is not None:
        out["depth_intrinsic"] = {
            "fx": intr3d[0], "fy": intr3d[1], "cx": intr3d[2], "cy": intr3d[3],
            "distortion": dist3d,
        }
        fx, fy, cx, cy = intr3d
        print(f"\n深度相机内参 (640x480):")
        print(f"  fx = {fx:.3f}   fy = {fy:.3f}")
        print(f"  cx = {cx:.3f}   cy = {cy:.3f}")
        print(f"  畸变 d1..d5 = {[round(d, 6) for d in dist3d]}")
        # FOV 换算
        import math
        print(f"  对应 FOV: H={2*math.degrees(math.atan(320/fx)):.2f}°  V={2*math.degrees(math.atan(240/fy)):.2f}°")
    else:
        print(f"get3DIntricParam 失败: {ret}")

    ret, intr2d, dist2d = camera.get2DIntricParam(handle)
    if ret == LX_STATE.LX_SUCCESS and intr2d is not None:
        out["rgb_intrinsic"] = {
            "fx": intr2d[0], "fy": intr2d[1], "cx": intr2d[2], "cy": intr2d[3],
            "distortion": dist2d,
        }
        print(f"\nRGB 相机内参 (1280x960):")
        print(f"  fx = {intr2d[0]:.3f}   fy = {intr2d[1]:.3f}")
        print(f"  cx = {intr2d[2]:.3f}   cy = {intr2d[3]:.3f}")
        print(f"  畸变 d1..d5 = {[round(d, 6) for d in dist2d]}")
    else:
        print(f"get2DIntricParam 失败: {ret}")

    ret, trans = camera.get3DTransMatrix(handle)
    if ret == LX_STATE.LX_SUCCESS and trans is not None:
        out["depth_extrinsic_3x4"] = trans.tolist()
        print(f"\n3D 外参 (3x4, R|t):")
        for row in trans:
            print(f"  [{row[0]:+.4f} {row[1]:+.4f} {row[2]:+.4f} | {row[3]:+.3f}]")
    else:
        print(f"get3DTransMatrix 失败: {ret}")

    camera.DcCloseDevice(handle)

    out_path = args.out or f"camera_intrinsics_{sn}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {out_path}")
    print("\n提示: 用此出厂值更新 shelf_anchor_web.py 的 FX/FY/CX/CY 和推理脚本内参。")


if __name__ == '__main__':
    main()
