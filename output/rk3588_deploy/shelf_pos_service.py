#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""货架识别自动服务 — 自动监听 + 触发即推理 → 回 pos → Program_3 转 PLC.

一个进程串起整条链 (替代"推理写文件 + 服务读文件"的两进程方案):

    1. 启动    : 加载 RKNN 模型 + 打开相机 (LxCamera SDK, RGBD 深度对齐) + 监听 TCP :5511
    2. 收到触发 : Program_3 发 {"action":"3D","id":1,"random_number":1}
                  → 现场抓一帧 → 推理 → 深度查表 → 两角点 XYZ 取均值 = 货架中心点
    3. 回 pos  : {"error_code":0,"pos":[Y,Z,X,angle],"random_number":1}
    4. 发给 PLC: Program_3 收到 pos 后照旧写 Modbus 14~17 → 运控/叉车
                  (脚本不碰 Modbus, 只负责把货架中心算出来喂给 Program_3)

相机系 → pos 映射: pos = [x, -y, z, 0]  (x=右+, y=下+, z=前方; 符号不对翻 SIGN_X/SIGN_Z)

用法:
    python3 shelf_pos_service.py                     # 常驻: 自动监听, 触发即推理
    python3 shelf_pos_service.py --ip 192.168.2.150  # 指定相机 IP (默认自动发现第一台)
    python3 shelf_pos_service.py --port 5512         # 换监听端口
    python3 shelf_pos_service.py --self-test         # 现场推理一次, 打印 XYZ+JSON (验证整条链)
    python3 shelf_pos_service.py --from-file         # 测试模式: 只读 center_xyz.json, 不连相机
    # 输入核对: 保存前 N 帧两模态 (RGB jpg + PCD, 与 imgs/ 训练数据同格式), 供肉眼确认输入对不对
    python3 shelf_pos_service.py --self-test --dump-frames 5
    python3 shelf_pos_service.py --ip 192.168.2.150 --dump-frames 10 --dump-dir /root/rk3588_deploy/input_check
    # 离线自检: 不连相机, 直接喂 imgs/ 训练数据 (jpg + pcd) 或 dump 的帧, 没真货架也能测链
    python3 shelf_pos_service.py --self-test --image imgs/TV_xxx.jpg --pcd imgs/TV_xxx.pcd

依赖: rknnlite / LxCameraSDK / numpy / opencv (与 infer_camera_sdk.py 相同)
"""

import argparse
import glob
import json
import os
import fcntl
import signal
import socket
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# ── 复用板端推理管线 (与 infer_camera_sdk.py / infer_image.py 完全一致) ──
from infer_camera import letterbox, decode_yolopose, DEFAULT_MODEL
from infer_camera_sdk import (
    LX_STATE, LX_CAMERA_FEATURE, CameraOpenError,
    open_camera, get_depth_intrinsics, setup_rgbd_align,
    build_sdk_depth_map, attach_anchor_3d, _bstr,
)
import shelf_viz                          # build_depth_map: 无序点云→深度图 (离线 imgs)
from query_camera_intrinsics import find_sdk_lib

# ── 相机系 → 内置算法 pos 的符号配置 ─────────────────────────────
# 现场内置算法回 Y=760, Z=-87, X=1517 → 默认映射 pos=[x, -y, z, 0]
# 若同帧对拍差符号, 翻下面两个值即可
SIGN_X = 1.0     # pos[0]=Y 横向:  Y = SIGN_X * x
SIGN_Z = -1.0    # pos[1]=Z 竖直:  Z = SIGN_Z * y   (相机 y 下为正 → 取负)
# 货架中心点不含偏航 → angle 恒 0

# --from-file 测试模式的数据文件
CENTER_FILE = Path("/root/rk3588_deploy/center_xyz.json")

# ── 相机鲁棒性参数 (断流自愈 / 干净释放) ─────────────────────────
# 上电/重启连不上相机、或旧进程被杀后相机独占锁(-9)未释放 → 直接崩.
# 口径: 启动带退避重试等锁释放; 运行中断流自动 释放→退避→重开; 退出务必 close
# 干净释放 DcCloseDevice, 否则下次启动必撞 LX_E_CTRL_PERMISS_ERROR (-9).
STARTUP_CONNECT_TIMEOUT_S = 60.0    # 启动最多等多久连上相机, 超时退出交 systemd 重启
RECONNECT_BACKOFF0 = 1.0            # 重连退避初值 (秒)
RECONNECT_BACKOFF_MAX = 15.0        # 重连退避上限 (秒)
FRAME_STALE_SEC = 3.0               # 最新帧超过这么久 = 画面过期, 拒绝吐旧位姿
STREAM_HARD_FAIL = 20               # 连续 N 次 RGB 拿不到 → 判流死触发重连
LOCK_FILE = Path("/root/rk3588_deploy/shelf_pos_service.lock")

# 持有已获取的 flock fd, 防止文件对象被 GC 提前关闭 → 进程存活期间锁才有效
_LOCK_HOLDERS = []


def acquire_single_instance(lock_file=LOCK_FILE):
    # 单实例守护: 相机独占控制权同一时刻只给一个进程. flock 由内核在进程退出/被杀时
    # 自动释放, 不会因残留锁文件卡死下次启动. 返回 fd 需持有到进程结束.
    fd = open(lock_file, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys_exit(f"已有实例在运行 ({lock_file}) — 先停掉旧进程再启动, 否则抢相机独占锁(-9)")
    fd.write(str(os.getpid()))
    fd.flush()
    _LOCK_HOLDERS.append(fd)      # 持有引用到进程结束 → flock 才一直生效
    return fd



def read_center_file(path):
    """从文件读中心点 (仅测试用): center_xyz.json → {"xyz":[x,y,z]}"""
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"未找到 {p}")
    d = json.loads(p.read_text())
    xyz = d["xyz"]
    return [float(xyz[0]), float(xyz[1]), float(xyz[2])]


def save_aligned_frame(out_dir, idx, rgb, points):
    """保存一帧「两模态输入」供肉眼核对 (--dump-frames N 时后台线程自动调):
    rgb_%03d.jpg      RGB 原图 (BGR 存)
    pc_%03d.npy       原始点云 (H,W,3) XYZ mm — 可再喂 --self-test --image/--pcd 离线测
    depth_%03d.png    深度图 uint16 mm (点云 z, 对齐时与 RGB 同分辨率同索引)
    depth_%03d.jpg    深度伪彩 (jet), 直接看
    overlay_%03d.jpg  RGB | 深度 并排 — 一眼看两模态是否同视野、同分辨率对齐
    """
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"rgb_{idx:03d}.jpg"), rgb)
    if points is None:
        print(f"[dump] #{idx}: RGB {rgb.shape[:2]}  无点云", flush=True)
        return
    write_pcd(os.path.join(out_dir, f"pc_{idx:03d}.pcd"), points, rgb)

    ph, pw = points.shape[:2]
    rh, rw = rgb.shape[:2]
    if ph != rh or pw != rw:
        print(f"[dump] #{idx}: RGB {rgb.shape[:2]} vs 点云 {points.shape[:2]} — 不对齐!"
              f" 只存 rgb/pc, 跳过深度图 (真要对齐先查相机对齐开关)", flush=True)
        return
    # 深度图: 点云 z 轴 (mm), 无效点置 0
    z = points[:, :, 2]
    valid = np.isfinite(z) & (z > 0)
    depth = np.zeros(z.shape, np.uint16)
    depth[valid] = np.clip(z[valid], 0, 65535).astype(np.uint16)
    cv2.imwrite(os.path.join(out_dir, f"depth_{idx:03d}.png"), depth)
    # 伪彩色深度 (jet, 无效点黑色)
    norm = cv2.normalize(depth.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    color[depth == 0] = (0, 0, 0)
    cv2.imwrite(os.path.join(out_dir, f"depth_{idx:03d}.jpg"), color)
    # 并排: RGB | 深度, 一眼看两模态是否对齐
    side = np.hstack([rgb, color])
    cv2.putText(side, "RGB", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(side, "DEPTH(jet)", (rw + 10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(out_dir, f"overlay_{idx:03d}.jpg"), side)
    print(f"[dump] #{idx}: RGB {rgb.shape[:2]} 点云 {points.shape[:2]} 对齐✓ "
          f"→ {out_dir}/rgb_{idx:03d}.jpg …", flush=True)


def load_pcd(path):
    """读 PCD (binary/ascii, 有序/无序) → (xyz, rgb, org).
    xyz: (N,3) float32 mm; rgb: (N,) uint32 或 None; org: 有序时的 (H,W) 或 None.
    兼容 imgs/ 训练数据 (FIELDS x y z rgb, HEIGHT 1, binary) 与 dump 的有序 pcd."""
    np_types = {"F": np.float32, "U": np.uint32, "I": np.int32, "f": np.float32,
                "u": np.uint32, "i": np.int32}
    with open(path, "rb") as f:
        header, data_type = [], None
        f.readline()                       # # .PCD v0.7
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: PCD header 不完整")
            h = line.decode("ascii", "ignore").strip()
            if h.startswith("DATA"):
                data_type = h.split()[1]
                break
            header.append(h)

        def kv(k):
            for h in header:
                if h.startswith(k):
                    return h.split()[1:]
            return []

        fields = kv("FIELDS")
        sizes = [int(x) for x in kv("SIZE")]
        types = kv("TYPE")
        counts = [int(x) for x in kv("COUNT")]
        width, height = int(kv("WIDTH")[0]), int(kv("HEIGHT")[0])
        n = int(kv("POINTS")[0]) or (width * height)
        dtype = np.dtype([(fields[i], np_types[types[i]]) if counts[i] == 1
                          else (fields[i], np_types[types[i]], counts[i])
                          for i in range(len(fields))])

        if data_type == "binary":
            raw = f.read(n * dtype.itemsize)
            if len(raw) < n * dtype.itemsize:
                raise ValueError(f"{path}: 数据不完整 ({len(raw)} < {n * dtype.itemsize})")
            arr = np.frombuffer(raw, dtype=dtype).copy()
        elif data_type == "ascii":
            rows = []
            for line in f:
                if not line.strip():
                    continue
                rows.append(line.decode().split())
            arr = np.array(rows, dtype=object)
            out = np.zeros(n, dtype=dtype)
            for i, name in enumerate(fields):
                out[name] = arr[:, i].astype(dtype[name].base if dtype[name].ndim else dtype[name])
            arr = out
        else:
            raise ValueError(f"{path}: 不支持 DATA {data_type}")

    xyz = np.stack([arr["x"], arr["y"], arr["z"]], -1).astype(np.float32)
    rgb = arr["rgb"].astype(np.uint32) if "rgb" in fields else None
    org = (height, width) if height > 1 else None
    return xyz, rgb, org


def write_pcd(path, points, rgb_bgr=None):
    """有序点云 (H,W,3) XYZ mm + 对齐 RGB → PCD (FIELDS x y z rgb, binary).
    字段与 imgs/ 训练数据一致; 保留有序结构 (HEIGHT=H) 供离线加载器走对齐直查.
    点云与 RGB 同尺寸时才打包每点颜色, 否则 rgb 全 0."""
    ph, pw, _ = points.shape
    xyz = points.reshape(-1, 3).astype(np.float32)
    n = xyz.shape[0]
    col = np.zeros(n, np.uint32)
    if rgb_bgr is not None and rgb_bgr.shape[:2] == (ph, pw):
        r = rgb_bgr[..., 2].astype(np.uint32).reshape(-1)   # BGR→RGB
        g = rgb_bgr[..., 1].astype(np.uint32).reshape(-1)
        b = rgb_bgr[..., 0].astype(np.uint32).reshape(-1)
        col = (r << 16) | (g << 8) | b
    rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])
    rec["x"], rec["y"], rec["z"], rec["rgb"] = xyz[:, 0], xyz[:, 1], xyz[:, 2], col
    with open(path, "wb") as f:
        f.write(("# .PCD v0.7 - Point Cloud Data file format\n"
                 "VERSION 0.7\n"
                 "FIELDS x y z rgb\n"
                 "SIZE 4 4 4 4\n"
                 "TYPE F F F U\n"
                 "COUNT 1 1 1 1\n"
                 f"WIDTH {pw}\n"
                 f"HEIGHT {ph}\n"
                 "VIEWPOINT 0 0 0 1 0 0 0\n"
                 f"POINTS {n}\n"
                 "DATA binary\n").encode())
        f.write(rec.tobytes())


def load_intrinsics(path):
    """读 camera_intrinsics json → (fx,fy,cx,cy), (res_w, res_h)."""
    with open(path, encoding="utf-8") as f:
        di = json.load(f).get("depth_intrinsic", {})
    if not di:
        raise RuntimeError(f"{path} 无 depth_intrinsic")
    res = di.get("resolution") or [640, 480]
    return (float(di["fx"]), float(di["fy"]), float(di["cx"]), float(di["cy"])), \
           (int(res[0]), int(res[1]))


def find_intrinsics(path):
    """离线自检的深度内参: 优先 --intrinsics; 否则脚本同目录唯一的 camera_intrinsics_*.json."""
    if path and os.path.isfile(path):
        return path
    d = os.path.dirname(os.path.abspath(__file__))
    cands = sorted(glob.glob(os.path.join(d, "camera_intrinsics_*.json")))
    return cands[0] if len(cands) == 1 else None


def make_videos(out_dir, n):
    """把 dump 的 n 帧拼成视频流 (RGB / 深度伪彩), 便于逐帧核对同步性."""
    for tag, name in (("rgb", "rgb_stream.avi"), ("depth", "depth_stream.avi")):
        first = os.path.join(out_dir, f"{tag}_000.jpg")
        if not os.path.isfile(first):
            continue
        img = cv2.imread(first)
        h, w = img.shape[:2]
        vw = cv2.VideoWriter(os.path.join(out_dir, name),
                             cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (w, h))
        for i in range(n):
            p = os.path.join(out_dir, f"{tag}_{i:03d}.jpg")
            if os.path.isfile(p):
                vw.write(cv2.imread(p))
        vw.release()
        print(f"[dump] 视频流已生成: {os.path.join(out_dir, name)}", flush=True)


class ShelfEngine:
    """启动时一次性初始化 (模型+相机+深度), 每次触发现场推理出中心点."""

    # ── 相机生命周期 (open / teardown / reconnect) ──
    # 取帧线程是相机的唯一主人; 主线程只读缓存帧, 不直接碰相机句柄.

    def _permiss_err(self, ret):
        # 该返回码 = 相机独占控制权被占 (LX_E_CTRL_PERMISS_ERROR, -9)
        return ret is not None and ret == getattr(LX_STATE, "LX_E_CTRL_PERMISS_ERROR", None)

    def _teardown(self):
        # 幂等释放流+设备句柄. 退出/重连前必调 → 设备独占锁(-9)不留到下次启动
        cam, h = self.camera, self.handle
        self.camera, self.handle, self._connected = None, None, False
        if cam is not None and h is not None:
            try:
                cam.DcStopStream(h)
            except Exception:
                pass
            try:
                cam.DcCloseDevice(h)
            except Exception:
                pass
            print("[cam] 相机流+设备句柄已释放", flush=True)

    def _interrupted(self):
        return self._stop or (self._abort is not None and self._abort())

    def _open_once(self):
        # 一次完整连接: 开相机→配 2D/3D 流→RGBD 对齐→帧同步→启流.
        # 任一失败回滚已开的句柄再抛, 绝不把独占锁留在半开句柄上
        dll = self.cfg.dll or find_sdk_lib()
        if dll is None:
            raise RuntimeError("找不到 LxCameraApi 动态库, 用 --dll 指定路径")
        camera, handle, dev_info = open_camera(dll, self.cfg.ip, self.cfg.sn, self.cfg.index)
        self._sn = self._sn or _bstr(getattr(dev_info, "sn", None), None)
        try:
            if camera.DcSetBoolValue(
                    handle, LX_CAMERA_FEATURE.LX_BOOL_ENABLE_2D_STREAM, True) \
                    != LX_STATE.LX_SUCCESS:
                raise RuntimeError("开启 2D 流失败")
            if camera.DcSetBoolValue(
                    handle, LX_CAMERA_FEATURE.LX_BOOL_ENABLE_3D_DEPTH_STREAM, True) \
                    != LX_STATE.LX_SUCCESS:
                raise RuntimeError("开启 3D 深度流失败")
            # RGBD 对齐 (DEPTH_TO_RGB) 失败会自动回退内参投影
            self._aligned = setup_rgbd_align(camera, handle)
            # RGB/深度帧同步必须无条件开, 否则多路流不同步 → 频繁 FRAME_ID_NOT_MATCH
            if camera.DcSetBoolValue(handle, LX_CAMERA_FEATURE.LX_BOOL_ENABLE_SYNC_FRAME, True) \
                    == LX_STATE.LX_SUCCESS:
                print("[sync] 已强制 RGB/深度帧同步 (ENABLE_SYNC_FRAME)", flush=True)
            else:
                print("[warn] 强制帧同步设置失败 → 靠取帧重试兜底", flush=True)
            # 深度内参只在第一次连时在线查, 重连(同 SN)直接用缓存
            if self.depth_intr is None:
                self.depth_intr = get_depth_intrinsics(camera, handle, self.cfg.intrinsics,
                                                       self._sn)
                if self.depth_intr is not None:
                    fx, fy, cx, cy = self.depth_intr
                    print(f"深度内参: fx={fx:.3f} fy={fy:.3f} "
                          f"cx={cx:.3f} cy={cy:.3f}", flush=True)
                else:
                    print("[warn] 拿不到深度内参, 对齐不可用时无法内参投影回退", flush=True)
            self._start_stream(camera, handle)
            self.camera, self.handle, self._connected = camera, handle, True
            self._hard_fail = 0
            print(f"[cam] 相机已连接 SN={self._sn or '?'}", flush=True)
        except Exception:
            # 半开状态回滚, 避免句柄泄漏占着相机独占锁
            try:
                camera.DcStopStream(handle)
            except Exception:
                pass
            try:
                camera.DcCloseDevice(handle)
            except Exception:
                pass
            raise

    def _start_stream(self, camera, handle):
        # 常开(WORK_FOREVER)模式下相机自己已在流中, 再 DcStartStream 会失败 → 跳过
        wm_mode = None
        try:
            from LxCameraSDK.lx_camera_define import LX_CAMERA_WORK_MODE
            wm_mode = LX_CAMERA_WORK_MODE
        except ImportError:
            pass
        already_streaming = False
        if wm_mode is not None:
            st_wm, wm = camera.DcGetIntValue(handle, LX_CAMERA_FEATURE.LX_INT_WORK_MODE)
            if st_wm == LX_STATE.LX_SUCCESS and \
                    wm.cur_value == wm_mode.WORK_FOREVER.value:
                already_streaming = True
        if already_streaming:
            print("[stream] 相机常开(WORK_FOREVER), 已在流中, 跳过 DcStartStream", flush=True)
            return
        # NOT_RECEIVE_STREAM: 相机上一会话没收干净/带宽不足, 相机要一两秒恢复 → 重试
        ret_start = camera.DcStartStream(handle)
        if ret_start != LX_STATE.LX_SUCCESS:
            time.sleep(1.0)
            ret_start = camera.DcStartStream(handle)
        if ret_start != LX_STATE.LX_SUCCESS:
            err = getattr(camera, "DcGetErrorString", lambda r: "?")(ret_start)
            raise RuntimeError(f"DcStartStream 失败: {ret_start} ({err})")

    def _connect(self, initial):
        # 连相机带退避重试: -9 独占锁/网络未就绪都要等设备自己恢复, 不能死循环硬抢.
        # 启动(initial=True) 限时 STARTUP_CONNECT_TIMEOUT_S, 超时抛给上层退出(交 systemd 重启);
        # 运行中重连(initial=False) 无限重试直到 _stop/abort.
        backoff = RECONNECT_BACKOFF0
        t0 = time.time()
        attempt = 0
        while True:
            attempt += 1
            try:
                self._open_once()
                if self._interrupted():      # 刚连上就收到退出 → 别占着相机
                    self._teardown()
                    raise RuntimeError("连接期间收到退出信号")
                return
            except CameraOpenError as e:
                if self._permiss_err(e.ret):
                    why = "相机被占用(独占锁未释放), 等待设备自动恢复"
                else:
                    why = f"打开失败 ret={e.ret}" if e.ret is not None else str(e)
                print(f"[cam] 第 {attempt} 次连相机失败: {why}", flush=True)
            except RuntimeError as e:
                print(f"[cam] 第 {attempt} 次连相机失败: {e}", flush=True)
            if self._interrupted():
                raise RuntimeError("收到退出信号, 中断连接重试")
            if initial and time.time() - t0 > STARTUP_CONNECT_TIMEOUT_S:
                raise RuntimeError(
                    f"启动 {int(STARTUP_CONNECT_TIMEOUT_S)}s 内连不上相机, 退出待重启")
            # 0.25s 分片睡, 让 Ctrl+C / SIGTERM 能及时打断退避
            for _ in range(int(backoff * 4)):
                if self._interrupted():
                    raise RuntimeError("收到退出信号, 中断连接重试")
                time.sleep(0.25)
            backoff = min(backoff * 2.0, RECONNECT_BACKOFF_MAX)

    def __init__(self, cfg, abort=None):
        # abort: 无参可调用, 返回真时中断连接重试 (SIGTERM / 手动中断)
        self.cfg = cfg
        self._abort = abort
        self._stop = False
        self._lock = threading.Lock()   # 触发连发时串行化推理
        self.camera = self.handle = None
        self._connected = False
        self._sn = None
        self.depth_intr = None
        self._aligned = False
        self._grab_err = ""
        self._latest = None             # (rgb, points) 最新帧缓存
        self._latest_t = 0.0            # 缓存帧时间戳 (新鲜度/过期判断)
        self._hard_fail = 0             # 连续取帧失败计数 (>= STREAM_HARD_FAIL 触发重连)
        self._dumped = 0
        self.rknn = None

        # ── RKNN 模型 ──
        from rknnlite.api import RKNNLite
        rknn = RKNNLite()
        if rknn.load_rknn(cfg.model) != 0:
            raise RuntimeError(f"load_rknn 失败: {cfg.model}")
        if rknn.init_runtime(core_mask=cfg.cores) != 0:
            raise RuntimeError(f"init_runtime 失败 (cores={cfg.cores})")
        self.rknn = rknn

        try:
            # 相机: 启动即带退避重试 (等上电网络就绪 / 设备独占锁 -9 释放)
            self._connect(initial=True)
            # 后台常驻取帧线程: 连续消费 RGB+深度保持同步; 断流自动 释放→退避→重开
            self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
            self._grab_thread.start()
            t0 = time.time()
            while self._latest is None and time.time() - t0 < 10 \
                    and not self._interrupted():
                time.sleep(0.05)
            if self._latest is None:
                if self._interrupted():
                    raise RuntimeError("启动中断: 收到退出信号, 还没等到首帧")
                raise RuntimeError(
                    f"连上相机但 10 秒内没取到帧: {self._grab_err or '取帧线程未就绪'}")
        except (KeyboardInterrupt, RuntimeError):
            # 半途失败务必清场, 否则相机独占锁/RKNN 会留给下一个实例 → 下次启动撞 -9
            self._stop = True
            self._teardown()
            try:
                if self.rknn is not None:
                    self.rknn.release()
                    self.rknn = None
            except Exception:
                pass
            raise

    def _grab_loop(self):
        # 后台取帧线程 = 相机的唯一主人. 三态流转:
        #   connected: 持续取最新 RGB+点云
        #   _connected=False: 进 _connect 退避重连; 重连期间主线程 infer_once 因
        #                    画面过期报错(宁错不给旧位姿), 重连成功自动恢复
        while not self._stop:
            if not self._connected:
                try:
                    self._connect(initial=False)
                except RuntimeError:
                    continue                       # 退出信号/中断 → 外层 while 判 _stop
                if self._stop:
                    break
            try:
                self._read_one_frame()
            except Exception as e:
                # 取帧失败/句柄竞态等一律视为流异常 → 释放后重连, 不让取帧线程死掉
                self._grab_err = str(e)
                print(f"[cam] 取帧异常, 触发重连: {e}", flush=True)
                self._teardown()
            time.sleep(0.01)

    def _read_one_frame(self):
        # 取一帧 RGB+点云 → 更新 _latest (带时间戳). True=有新帧; False=瞬时丢帧.
        # 连续 STREAM_HARD_FAIL 次 RGB 失败, 或 _grab_frame 抛真错误 → 抛错触发重连
        data_ptr = self._grab_frame(max_try=6)   # 帧同步类临时错内部已重试
        ret, rgb = self.camera.getRGBImage(data_ptr)
        if ret != LX_STATE.LX_SUCCESS or rgb is None:
            self._hard_fail += 1
            if self._hard_fail >= STREAM_HARD_FAIL:
                raise RuntimeError(f"连续 {self._hard_fail} 次 RGB 解码失败")
            time.sleep(0.03)
            return False
        rgb = np.ascontiguousarray(rgb.copy())   # 脱离 SDK 缓冲 (下次 getFrame 会被复用)
        if rgb.ndim == 3 and rgb.shape[2] == 1:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
        if self.cfg.color == "rgb":              # SDK 若是 RGB 顺序, 转回 BGR 对齐推理管线
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # 3D 点云短时拿不到只置 None (RGB 才是必保流), 不算致命
        points = None
        try:
            state, points = self.camera.getPointCloud(self.handle)
            if state != LX_STATE.LX_SUCCESS or points is None or points.size == 0:
                points = None
        except Exception:
            points = None
        self._hard_fail = 0
        self._latest = (rgb, points)
        self._latest_t = time.time()
        # --dump-frames: 前 N 帧存两模态 (RGB+点云) 供核对输入对不对
        if self.cfg.dump_frames and self._dumped < self.cfg.dump_frames:
            try:
                save_aligned_frame(self.cfg.dump_dir, self._dumped, rgb, points)
                self._dumped += 1
                if self._dumped == self.cfg.dump_frames and self.cfg.dump_frames >= 2:
                    make_videos(self.cfg.dump_dir, self.cfg.dump_frames)
            except Exception as e:
                print(f"[dump] 保存失败: {e}", flush=True)
        return True

    def _grab_frame(self, max_try=8):
        """取一帧 RGB. FRAME_ID_NOT_MATCH(帧不同步)/MULTI_MACHINE 是临时错误,
        官方示例(application_pallet.cpp)的处理就是重取, 不判失败. RECONNECTING 时等一会再试."""
        retryable = set()
        for name in ("LX_E_FRAME_ID_NOT_MATCH", "LX_E_FRAME_MULTI_MACHINE"):
            v = getattr(LX_STATE, name, None)
            if v is not None:
                retryable.add(v)
        last = None
        for i in range(max_try):
            ret, data_ptr = self.camera.getFrame(self.handle)
            if ret == LX_STATE.LX_SUCCESS:
                return data_ptr
            last = ret
            if ret in retryable:
                if i:
                    print(f"[warn] 帧不同步({ret}), 重取 {i + 1}/{max_try}", flush=True)
                time.sleep(0.03)
                continue
            if getattr(LX_STATE, "LX_E_RECONNECTING", None) == ret:
                time.sleep(1.0)          # 设备重连, 慢一点等
                continue
            break                        # 其他错误是真错误
        raise RuntimeError(f"取帧失败: {last}")

    @staticmethod
    def _infer_on(rknn, rgb, points, depth_intr, cfg, depth_res=None):
        """对给定 (RGB, 点云) 跑完整推理: letterbox→RKNN→decode→深度查表→中心点.
        实时(缓存帧)与离线(imgs/保存帧)共用这条链. 返回相机系 [x,y,z] mm.
        points 三种形态:
          (H,W,3) 且 == RGB 尺寸 → 对齐直查 (aligned=True)
          (H,W,3) 但 != RGB 尺寸 → 内参投影 (同实时回退)
          (N,3) 无序 (imgs 的 .pcd) → 按 depth_res 投影成深度图再查
        depth_res: 无序点云投影目标分辨率 (深度内参分辨率, 缺省 640x480)."""
        fh, fw = rgb.shape[:2]
        lb, _, _ = letterbox(rgb)
        blob = np.ascontiguousarray(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB))[None]
        outputs = rknn.inference(inputs=[blob])
        dets = decode_yolopose(outputs, (fh, fw), cfg.conf, cfg.iou)

        # 深度查表 (RGBD 对齐点云优先, 内参投影回退)
        depth_map = None
        aligned = False
        if points is not None:
            if points.ndim == 3 and points.shape[2] == 3:
                ph, pw = points.shape[:2]
                aligned = (ph == fh and pw == fw)      # 对齐到 RGB 同分辨率同索引
                if not aligned and depth_intr is not None:
                    depth_map = build_sdk_depth_map(points, depth_intr)
            elif points.ndim == 2 and depth_intr is not None:
                # 无序点云 (imgs .pcd): 内参投影成深度图
                dw, dh = depth_res or (640, 480)
                fx, fy, cx, cy = depth_intr
                depth_map = shelf_viz.build_depth_map(points, img_w=dw, img_h=dh,
                                                      fx=fx, fy=fy, cx=cx, cy=cy)
        dets = attach_anchor_3d(dets, points, depth_map, depth_intr, fw, fh,
                                aligned=aligned)

        # 最高置信度实例, 两角点 3D 取均值 = 货架中心点
        if not dets:
            raise RuntimeError("未检测到货架")
        best = max(dets, key=lambda d: d["conf"])
        a3d = [a for a in (best.get("anchor_3d") or []) if a is not None]
        if len(a3d) < 2:
            raise RuntimeError(f"角点 3D 不足 ({len(a3d)}/2), 深度缺失")
        center = [(a3d[0][i] + a3d[1][i]) / 2.0 for i in range(3)]
        return [float(v) for v in center]

    def infer_once(self):
        # 用后台线程缓存的最新帧 → 推理 → 货架中心点 XYZ (相机系, mm).
        # 画面过期(断流/重连中) → 抛错, 上层 handle_client 回 error_code=-1, 宁错不给旧位姿
        with self._lock:
            latest = self._latest
            if latest is None:
                raise RuntimeError(f"取帧线程还没就绪: {self._grab_err or '未知'}")
            rgb, points = latest
            age = time.time() - self._latest_t
        if age > FRAME_STALE_SEC:
            raise RuntimeError(
                f"相机画面已过期 {age:.0f}s (取帧中断/重连中), 拒绝用旧帧推理")
        return self._infer_on(self.rknn, rgb, points, self.depth_intr, self.cfg)
    
    def get_shelf_center_xyz(self):
        """★核心插桩点★: 返回货架中心点相机系 3D [x, y, z] (mm).
        默认=现场推理; --from-file 时读文件 (测试)."""
        if self.cfg.from_file:
            return read_center_file(self.cfg.center_file)
        return self.infer_once()

    def close(self):
        # 停止取帧线程并释放相机/RKNN. 保证 DcCloseDevice → 下次启动不撞独占锁 -9.
        # systemd 停服务若不干净 close, 设备侧锁要几十秒才自动释放 → 下一个实例直接 -9
        self._stop = True
        self._teardown()
        th = getattr(self, "_grab_thread", None)
        if th is not None and th.is_alive():
            th.join(timeout=2.0)       # 别让取帧线程在 close 后还碰相机
        try:
            if self.rknn is not None:
                self.rknn.release()
                self.rknn = None
        except Exception:
            pass
def to_pos(xyz):
    """相机系 [x,y,z] → 内置算法 pos[Y,Z,X,angle] JSON 字符串."""
    x, y, z = xyz
    return json.dumps({
        "error_code": 0,
        "pos": [round(SIGN_X * x, 1),    # Y 横向
                round(SIGN_Z * y, 1),    # Z 竖直
                round(z, 1),             # X 前方深度
                0.0],                    # angle 偏航(中心点不含, 恒 0)
        "random_number": 1,
    }, separators=(",", ":"))


def handle_client(conn, get_xyz, cfg):
    """一个连接: 收到触发 → 现场推理 → 回一条 JSON. 连接保持, 可多次触发."""
    try:
        conn.settimeout(60)
        while True:
            data = conn.recv(4096)
            if not data:
                break                          # Program_3 断开
            msg = data.decode("utf-8", "ignore").strip()
            print(f"[触发] {msg}", flush=True)
            # 平台识别(id:2)不属于本服务职责, 回失败避免误给位姿
            if '"id":2' in msg:
                resp = json.dumps({"error_code": -1, "pos": [0, 0, 0, 0],
                                   "random_number": 1})
            else:
                try:
                    xyz = get_xyz()
                    resp = to_pos(xyz)
                    print(f"[OK]  XYZ={xyz} → {resp}", flush=True)
                except Exception as e:
                    resp = json.dumps({"error_code": -1, "pos": [0, 0, 0, 0],
                                       "random_number": 1})
                    print(f"[ERR] {e}", flush=True)
            conn.sendall(resp.encode("utf-8"))
    except Exception as e:
        print(f"[conn] {e}", flush=True)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="货架识别自动服务: 监听触发→推理→回 pos→PLC")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5511)
    ap.add_argument("--from-file", action="store_true",
                    help="测试模式: 中心点读 center_xyz.json, 不连相机不推理")
    ap.add_argument("--center-file", default=str(CENTER_FILE),
                    help="--from-file 时的中心点文件路径")
    ap.add_argument("--self-test", action="store_true",
                    help="现场推理一次, 打印 XYZ+pos JSON 后退出 (验证整条链)")
    # 相机/模型参数 (默认与 infer_camera_sdk.py 一致)
    ap.add_argument("--ip", default=None, help="相机 IP (默认自动发现第一台)")
    ap.add_argument("--sn", default=None, help="相机序列号")
    ap.add_argument("--index", type=int, default=None, help="按序号打开 (0 起)")
    ap.add_argument("--dll", default=None, help="LxCameraApi 动态库路径 (默认自动查找)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=".rknn 模型路径")
    ap.add_argument("--conf", type=float, default=0.7)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--cores", type=int, default=3,
                    help="NPU 核: 1=单核 3=双核(默认) 7=三核 0=自动")
    ap.add_argument("--color", choices=["bgr", "rgb"], default="bgr",
                    help="SDK RGB 流通道顺序 (画面偏蓝/偏橙改 --color rgb)")
    ap.add_argument("--intrinsics", default=None,
                    help="深度内参 JSON (缺省按 SN 自动找 camera_intrinsics_<sn>.json)")
    # 输入核对: 实时 dump 两模态帧 (供肉眼/离线复测)
    ap.add_argument("--dump-frames", type=int, default=0,
                    help="启动后保存前 N 帧对齐 RGB+点云 (自检/常驻都行, 默认 0=不存)")
    ap.add_argument("--dump-dir", default="/root/rk3588_deploy/input_check",
                    help="--dump-frames 的输出目录")
    # 离线自检: 不连相机, 用 dump 的帧跑推理
    ap.add_argument("--image", default=None,
                    help="离线自检: RGB 图 (dump 的 rgb_%%03d.jpg)")
    ap.add_argument("--pcd", default=None,
                    help="离线自检: 点云 .pcd (imgs/TV_xxx.pcd 无序 或 dump 有序) 或 .npy")
    args = ap.parse_args()

    if args.self_test and args.from_file:
        sys_exit("--self-test 与 --from-file 不能同时用")
    if bool(args.image) != bool(args.pcd):
        sys_exit("--image 和 --pcd 必须成对给 (jpg + pcd, 如 imgs/TV_xxx.jpg + imgs/TV_xxx.pcd)")

    # 停止信号: SIGTERM → 置位, 主循环干净退出 → finally 里 close() 释放相机独占锁.
    # (SIGINT 不接管, Ctrl+C 仍走 KeyboardInterrupt 原路径)
    _STOP = threading.Event()

    def _on_sigterm(sig, frm):
        print("\n[SIGTERM] 收到停止信号, 正在释放相机退出…", flush=True)
        _STOP.set()

    signal.signal(signal.SIGTERM, _on_sigterm)

    engine = None
    if not args.from_file:
        if not os.path.exists(args.model):
            sys_exit(f"模型不存在: {args.model}")
        if args.self_test and args.image and args.pcd:
            # ── 离线自检: 不连相机, 用 imgs/ 或 dump 的 (jpg + pcd) 测整条推理链 ──
            for p in (args.image, args.pcd):
                if not os.path.exists(p):
                    sys_exit(f"离线数据不存在: {p}")
            from rknnlite.api import RKNNLite
            rknn = RKNNLite()
            if rknn.load_rknn(args.model) != 0:
                sys_exit(f"load_rknn 失败: {args.model}")
            if rknn.init_runtime(core_mask=args.cores) != 0:
                sys_exit(f"init_runtime 失败 (cores={args.cores})")
            try:
                rgb = cv2.imread(args.image)          # BGR, 与实时路径同序
                if rgb is None:
                    raise RuntimeError(f"读图失败: {args.image}")
                if args.pcd.lower().endswith(".pcd"):
                    xyz, _, org = load_pcd(args.pcd)
                    # 有序 pcd → 还原 (H,W,3) 走对齐直查; 无序 (imgs) → (N,3) 内参投影
                    points = xyz.reshape(org[0], org[1], 3) if org else xyz
                else:
                    points = np.load(args.pcd)        # dump 旧 npy (H,W,3)
                # 深度内参: 对齐直查用不到; 无序/不对齐时投影必需 (自动找脚本同目录的 json)
                ic = find_intrinsics(args.intrinsics)
                depth_intr, depth_res = (load_intrinsics(ic) if ic else (None, None))
                xyz = ShelfEngine._infer_on(rknn, rgb, points, depth_intr, args,
                                            depth_res=depth_res)
                print(f"[self-test] 离线推理 ({os.path.basename(args.image)}) "
                      f"→ XYZ = {[round(v, 1) for v in xyz]}")
                print(f"[self-test] pos JSON = {to_pos(xyz)}")
                print("[self-test] ✅ 离线链路 OK: 模型加载 + 对齐点云深度查表 + 检测 全通")
            except RuntimeError as e:
                print(f"[self-test] {e}")
                if "未检测到货架" in str(e):
                    print("[self-test] ⚠️ 离线数据里没有货架 (图里没货架, 正常). "
                          "模型/推理/深度查表都正常.")
                else:
                    print("[self-test] ❌ 离线链路有问题, 按上面的错误排查")
            finally:
                rknn.release()
            return
        # 单实例守护 + 启动引擎 (相机连接自带退避重试)
        acquire_single_instance()   # 重复拉起会抢相机独占锁(-9)
        try:
            engine = ShelfEngine(args, abort=_STOP.is_set)
        except KeyboardInterrupt:
            if engine is not None:
                engine.close()
            print("已取消。", flush=True)
            sys.exit(130)
        except RuntimeError as e:
            print(f"[fatal] {e}", flush=True)
            sys.exit(1)
        if args.self_test:
            try:
                xyz = engine.infer_once()
                print(f"[self-test] 现场推理 → XYZ = {[round(v, 1) for v in xyz]}")
                print(f"[self-test] pos JSON  = {to_pos(xyz)}")
                print("[self-test] ✅ 整条链 OK: 模型+相机取帧+深度查表+检测 全通")
            except RuntimeError as e:
                print(f"[self-test] {e}")
                if "未检测到货架" in str(e):
                    print("[self-test] ⚠️ 相机画面里没有货架 (正常). 链路已通: 模型加载/取帧/推理都正常.")
                    print("[self-test]    放个货架或纸箱到画面里再跑一次, 就会输出 XYZ 和 pos JSON.")
                else:
                    print("[self-test] ❌ 链路有问题, 按上面的错误排查")
            finally:
                engine.close()
            return
        get_xyz = engine.get_shelf_center_xyz
        print(f"模型: {args.model}  cores={args.cores}", flush=True)
    else:
        # 测试模式: 不连相机, 直接验证协议
        if args.self_test:
            xyz = read_center_file(args.center_file)
            print(f"[self-test] XYZ = {xyz}")
            print(f"[self-test] JSON = {to_pos(xyz)}")
            return
        get_xyz = lambda: read_center_file(args.center_file)
        print("测试模式 --from-file: 中心点读 center_xyz.json, 不连相机", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(8)
    srv.settimeout(1.0)              # 每秒醒来查一次退出信号 (SIGTERM 干净退出)
    print(f"货架识别服务监听 {args.host}:{args.port} (协议: 内置算法同款)", flush=True)
    print("触发 → 现场推理 → 回 pos → Program_3 写 Modbus 14~17 → 运控/PLC", flush=True)
    print("相机: 断流自动重连; Ctrl+C / SIGTERM 干净退出并释放相机独占锁", flush=True)
    try:
        while not _STOP.is_set():
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue             # accept 超时 → 回 while 判 _STOP
            except OSError:
                break
            print(f"[连接] {addr}", flush=True)
            threading.Thread(target=handle_client, args=(conn, get_xyz, args),
                             daemon=True).start()
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()
        if engine is not None:
            engine.close()
        print("已退出。")


def sys_exit(msg):
    import sys
    print(msg)
    sys.exit(1)


if __name__ == "__main__":
    main()
