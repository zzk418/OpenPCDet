#!/usr/bin/env python3
"""从蓝芯 MRDVS (Eagle-M4 Mega) 相机直接读取出厂标定内参 (权威值)。

用 LxCamera SDK (lx_camera_py wheel 1.3.3) 的 get3DIntricParam / get2DIntricParam:
    intrinsic[fx, fy, cx, cy], distort[d1, d2, d3, d4, d5]
同时输出 3D 外参矩阵 (get3DTransMatrix) 和当前分辨率 (LX_INT_*_IMAGE_*)。

API 已对照 wheel 内绑定核对 (LxCameraSDK/lx_camera_api.py):
    DcGetDeviceList() -> (ret, dev_list, dev_num)   # dev_list[i].{name,ip,sn,handle}
    DcOpenDevice(mode, param) -> (ret, handle, dev_info)   # 3 元组
    get2DIntricParam(handle)  -> (ret, [fx,fy,cx,cy], [d1..d5])
    get3DIntricParam(handle)  -> (ret, [fx,fy,cx,cy], [d1..d5])
    get3DTransMatrix(handle)  -> (ret, np.float32[3,4])     # R|t, 单位米/毫米见相机型号
    DcGetIntValue(handle, LX_INT_*_IMAGE_WIDTH/HEIGHT) -> (ret, LxIntValueInfo)

用法:
    # 自动发现相机, 按序号打开第一台
    python query_camera_intrinsics.py

    # 指定 IP / 序列号 / 序号
    python query_camera_intrinsics.py --ip 192.168.100.86
    python query_camera_intrinsics.py --sn 519889C9A2A6468E
    python query_camera_intrinsics.py --index 0

    # SDK 动态库路径 (默认自动查找, 优先 /opt/MRDVS/lib — install.sh 的安装位置)
    python query_camera_intrinsics.py --dll "/opt/MRDVS/lib/libLxCameraApi.so"

输出:
    camera_intrinsics_<sn>.json  +  控制台表格

依赖:
    pip install <SDK>/Sample/python/lx_camera_py-1.3.3-py3-none-any.whl
    (RK3588 板子: pip install MRDVS_linux/Sample/python/lx_camera_py-1.3.3-py3-none-any.whl)
"""
import argparse
import json
import math
import os
import sys
from datetime import datetime

# 相机分辨率回退值 (仅当 SDK 查不到当前分辨率时用, 见 main 内动态查询)
DEFAULT_DEPTH_RES = (640, 480)   # W, H
DEFAULT_RGB_RES = (1280, 960)    # W, H


def _bstr(raw, fallback=""):
    """c_char32 字段 → 干净字符串 (去掉尾部 \x00 再 strip)。"""
    try:
        return raw.decode("utf-8").rstrip("\x00").strip()
    except Exception:
        return fallback


def find_sdk_lib():
    """自动查找 SDK 动态库 (Windows .dll / Linux .so)。

    优先 install.sh 的安装位置 /opt/MRDVS/lib; 其次仓库内 SDK/lib/linux_{arch};
    再按 LD_LIBRARY_PATH 里带 libLxCameraApi.so 的目录。返回 None 表示没找到。
    """
    if sys.platform.startswith('win'):
        candidates = [
            r"D:\Program Files\Lanxin-MRDVS\SDK\lib\win_x64\LxCameraApi.dll",
            r"C:\Program Files\Lanxin-MRDVS\SDK\lib\win_x64\LxCameraApi.dll",
        ]
    else:
        arch = {"x86_64": "linux_x64", "aarch64": "linux_aarch64",
                "armv7l": "linux_arm32", "arm64": "linux_aarch64"}.get(
                    os.uname().machine, "linux_x64")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            "/opt/MRDVS/lib/libLxCameraApi.so",                       # install.sh 安装位置
            f"/opt/Lanxin-MRDVS/SDK/lib/{arch}/libLxCameraApi.so",
            os.path.join(script_dir, "MRDVS_linux", "SDK", "lib", arch, "libLxCameraApi.so"),
            os.path.join(script_dir, "..", "..", "CameraSDK", "MRDVS_linux", "SDK", "lib", arch, "libLxCameraApi.so"),
        ]
        # LD_LIBRARY_PATH 里能找到的 (install 时被写进 ~/.bashrc)
        for d in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
            if d.strip():
                candidates.append(os.path.join(d.strip(), "libLxCameraApi.so"))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def query_image_size(camera, handle, feature_w, feature_h, fallback):
    """读相机当前图像分辨率 (W,H)。失败/为 0 时用回退值。"""
    try:
        ret_w, info_w = camera.DcGetIntValue(handle, feature_w)
        ret_h, info_h = camera.DcGetIntValue(handle, feature_h)
        if ret_w == LX_STATE.LX_SUCCESS and ret_h == LX_STATE.LX_SUCCESS \
                and info_w.cur_value > 0 and info_h.cur_value > 0:
            return int(info_w.cur_value), int(info_h.cur_value)
    except Exception:
        pass
    return fallback


def print_intrinsics(tag, intr, dist, res_w, res_h):
    """打印一组内参 + 畸变 + FOV (按实际分辨率). 返回 None 处理放调用侧。"""
    fx, fy, cx, cy = intr
    print(f"\n{tag} 内参 ({res_w}x{res_h}):")
    print(f"  fx = {fx:.3f}   fy = {fy:.3f}")
    print(f"  cx = {cx:.3f}   cy = {cy:.3f}")
    print(f"  畸变 d1..d5 = {[round(d, 6) for d in dist]}")
    print(f"  对应 FOV: H={2*math.degrees(math.atan((res_w/2)/fx)):.2f}°  "
          f"V={2*math.degrees(math.atan((res_h/2)/fy)):.2f}°")
    return fx, fy, cx, cy


def main():
    parser = argparse.ArgumentParser(description="读取蓝芯 MRDVS (Eagle-M4 Mega) 出厂内参")
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
        from LxCameraSDK.lx_camera_define import LX_STATE, LX_OPEN_MODE, LX_CAMERA_FEATURE
        from LxCameraSDK.lx_camera_api import LxCamera
    except ImportError:
        sys.exit(
            "缺少 LxCameraSDK 包。请安装:\n"
            "  pip install <CameraSDK 仓库>/linux/Sample/python/lx_camera_py-1.3.3-py3-none-any.whl\n"
            "  (https://github.com/Lanxin-MRDVS/CameraSDK)")

    camera = LxCamera(dll_path)
    camera.DcSetInfoOutput(0, True, "")  # 抑制 SDK 刷屏日志
    print(f"SDK: {camera.DcGetApiVersion()}, 库: {dll_path}")

    ret, dev_list, dev_num = camera.DcGetDeviceList()
    if ret != LX_STATE.LX_SUCCESS:
        sys.exit(f"DcGetDeviceList 失败: {ret}")
    print(f"发现 {dev_num} 台相机")

    # ── 确定打开方式和参数 ──
    if args.ip:
        mode, param = LX_OPEN_MODE.OPEN_BY_IP, args.ip
    elif args.sn:
        mode, param = LX_OPEN_MODE.OPEN_BY_SN, args.sn
    else:
        if dev_num == 0:
            sys.exit("未发现相机 (检查网络/供电)")
        idx = args.index if args.index is not None else 0
        if idx >= dev_num:
            sys.exit(f"--index {idx} 超出范围 (共 {dev_num} 台)")
        dev = dev_list[idx]
        print(f"设备 {idx}: {_bstr(dev.name)}  IP={_bstr(dev.ip)}")
        sn = _bstr(dev.sn)          # 去 \x00 再比对, 否则 OPEN_BY_SN 匹配不上
        if sn:
            mode, param = LX_OPEN_MODE.OPEN_BY_SN, sn
        else:
            mode, param = LX_OPEN_MODE.OPEN_BY_INDEX, str(idx)

    ret, handle, dev_info = camera.DcOpenDevice(mode, param)
    if ret != LX_STATE.LX_SUCCESS:
        sys.exit(f"DcOpenDevice({mode}, {param}) 失败: {ret}")
    try:
        name = _bstr(dev_info.name, "?")
        sn = _bstr(dev_info.sn, args.sn or "unknown")
        print(f"已打开: {name}  SN={sn}  IP={_bstr(dev_info.ip)}")

        out = {"camera": name, "sn": sn, "queried_at": datetime.now().isoformat(timespec='seconds')}

        # ── 深度/2D 分辨率 (FOV 换算用真实值, 不再写死 640x480 / 1280x960) ──
        depth_w, depth_h = query_image_size(
            camera, handle,
            LX_CAMERA_FEATURE.LX_INT_3D_IMAGE_WIDTH,
            LX_CAMERA_FEATURE.LX_INT_3D_IMAGE_HEIGHT,
            DEFAULT_DEPTH_RES)
        rgb_w, rgb_h = query_image_size(
            camera, handle,
            LX_CAMERA_FEATURE.LX_INT_2D_IMAGE_WIDTH,
            LX_CAMERA_FEATURE.LX_INT_2D_IMAGE_HEIGHT,
            DEFAULT_RGB_RES)

        # ── 3D 内参 (深度相机) ──
        ret, intr3d, dist3d = camera.get3DIntricParam(handle)
        if ret == LX_STATE.LX_SUCCESS and intr3d is not None:
            fx3, fy3, cx3, cy3 = print_intrinsics("深度相机", intr3d, dist3d, depth_w, depth_h)
            out["depth_intrinsic"] = {
                "fx": fx3, "fy": fy3, "cx": cx3, "cy": cy3,
                "distortion": dist3d, "resolution": [depth_w, depth_h],
            }
        else:
            print(f"get3DIntricParam 失败: {ret}")

        # ── 2D 内参 (RGB 相机) ──
        ret, intr2d, dist2d = camera.get2DIntricParam(handle)
        if ret == LX_STATE.LX_SUCCESS and intr2d is not None:
            fx2, fy2, cx2, cy2 = print_intrinsics("RGB 相机", intr2d, dist2d, rgb_w, rgb_h)
            out["rgb_intrinsic"] = {
                "fx": fx2, "fy": fy2, "cx": cx2, "cy": cy2,
                "distortion": dist2d, "resolution": [rgb_w, rgb_h],
            }
        else:
            print(f"get2DIntricParam 失败: {ret}")

        # ── 3D 外参 (深度↔世界, 4 列 R|t) ──
        ret, trans = camera.get3DTransMatrix(handle)
        if ret == LX_STATE.LX_SUCCESS and trans is not None:
            out["depth_extrinsic_3x4"] = trans.tolist()
            print(f"\n3D 外参 (3x4, R|t):")
            for row in trans:
                print(f"  [{row[0]:+.4f} {row[1]:+.4f} {row[2]:+.4f} | {row[3]:+.3f}]")
        else:
            print(f"get3DTransMatrix 失败: {ret}")

        out_path = args.out or f"camera_intrinsics_{sn}.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n已保存: {out_path}")
        print("\n提示: shelf_viz.py / infer_camera_sdk.py 会自动读本 JSON 的 depth_intrinsic,"
              "无需手工改代码。")
    finally:
        camera.DcCloseDevice(handle)


if __name__ == '__main__':
    main()
