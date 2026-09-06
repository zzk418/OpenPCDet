#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测车控/PLC 直连 Modbus 通信 (取代 Program_3 后, 本脚本扮演"车控"客户端).

服务端是 shelf_pos_service.py 自带的 Modbus TCP Server (HoldingRegisters 0~19):
    0~9   车控→视觉: 0 心跳  1 拍照命令(0无/3货架单次/4货架连续/…其余=不支持)
                     2 故障复位(写1清 reg13)
    10~19 视觉→车控: 10 心跳  11 拍照回令  12 工作状态(0空闲/1拍照中/2完成/3异常)
                     13 故障码  14/15/16/17 = 位姿 X·Y·偏航·Z(0.1mm/0.1° 可负)
                     18 温度  19 预留

本脚本 = 纯 socket 实现的最小 Modbus TCP 客户端 (零依赖, 故意不 import 服务端模块,
按协议独立实现, 才能真校验服务端字节/语义). 用例:
    A  连接 + 读全部寄存器 + 心跳翻转
    B  货架单次 (reg1=3): 状态 →2(完成), reg11 回令=1, 位姿 reg14~17 换算打印
       → 车控应答 reg1=0 → 回空闲
    C  不支持的拍照命令 (reg1=5): → 状态3(异常) + reg13=42; reg2=1 复位故障
       → reg1=0 → 空闲 (命令 5 不触发推理, 确定性, 现场也稳)
    D  [可选 --continuous] 货架连续 (reg1=4): 连续吐完成+位姿 → reg1=0 停

用本地 --from-file 服务可做全链路回归 (不连相机/模型/RKNN):
    # 终端1: python shelf_pos_service.py --from-file \
    #            --center-file center_xyz.json --plc-port 31000
    # 终端2: python sim_plc_client.py --host 127.0.0.1 --port 31000 --expect-xyz X,Y,Z
一键本地全链路 (本脚本自己拉一个 --from-file 服务并跑上面全部用例):
    python sim_plc_client.py --local-e2e

对板子/现场 (车控本来就连 .102:30000, 照 Program_3 连法):
    python sim_plc_client.py --host 192.168.2.102 --port 30000
    # 相机前有货架 → 单次/连续会真实推理; C 用例照样确定
"""
import argparse
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time

# ── 寄存器布局 (与协议/服务端一致; 独立重写, 不依赖服务端模块) ──────────
CAR_HB, PHOTO_CMD, FAULT_RESET = 0, 1, 2          # 0~9 车控可写
VIS_HB, VIS_REPLY, VIS_STATE, VIS_FAULT = 10, 11, 12, 13
VIS_X, VIS_Y, VIS_YAW, VIS_Z = 14, 15, 16, 17      # 0.1mm / 0.1°
VIS_TEMP = 18
ST_IDLE, ST_SHOOT, ST_DONE, ST_ERR = 0, 1, 2, 3
CMD_NONE, CMD_PLT, CMD_PLT_CT, CMD_SHELF, CMD_SHELF_CT, CMD_PFM = range(6)
ST_NAMES = {0: "空闲", 1: "拍照中", 2: "完成", 3: "异常"}

# 换算: 位姿 Y=SIGN_X*x  Z=SIGN_Z*y  X=z (相机 mm, 与 to_pos 一致)
SIGN_X, SIGN_Z = 1.0, -1.0


# ── 最小 Modbus TCP 客户端 ─────────────────────────────────────────
class ModbusClient:
    def __init__(self, host, port, unit=1, timeout=8.0):
        self.host, self.port, self.unit = host, port, unit
        self._tid = 0
        self.sock = socket.create_connection((host, port), timeout=timeout)

    def _tx(self, pdu):
        self._tid = (self._tid + 1) & 0xFFFF
        tid = self._tid
        head = struct.pack(">HHHB", tid, 0, 1 + len(pdu), self.unit) + pdu
        self.sock.sendall(head)
        hdr = self._recv_exact(7)
        r_tid, pid, length, unit = struct.unpack(">HHHB", hdr)
        if r_tid != tid or pid != 0:
            raise IOError(f"MBAP 不符: tid {r_tid}≠{tid}")
        body = self._recv_exact(length - 1)        # 去掉 unit
        func = body[0]
        if func & 0x80:
            names = {1: "不支持的功能码", 2: "寄存器地址越界", 3: "非法值"}
            raise IOError(f"Modbus 异常 0x{func & 0x7F:02X} / "
                          f"code 0x{body[1]:02X} = {names.get(body[1], '未知')}")
        return body

    def _recv_exact(self, n):
        buf = b""
        while len(buf) < n:
            c = self.sock.recv(n - len(buf))
            if not c:
                raise IOError("连接被服务端关闭")
            buf += c
        return buf

    def read_holding(self, addr, cnt):
        body = self._tx(struct.pack(">BHH", 0x03, addr, cnt))
        nbytes = body[1]
        if len(body) != 2 + nbytes:
            raise IOError("读保持寄存器响应长度不符")
        return [struct.unpack(">H", body[i:i + 2])[0]
                for i in range(2, 2 + nbytes, 2)]

    def write_single(self, addr, val):
        return self._tx(struct.pack(">BHH", 0x06, addr, val & 0xFFFF))

    def write_multi(self, addr, vals):
        vals = [v & 0xFFFF for v in vals]
        pdu = struct.pack(">BHHB", 0x10, addr, len(vals), 2 * len(vals))
        pdu += struct.pack(">" + "H" * len(vals), *vals)
        self._tx(pdu)

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


def s16(v):
    return v - 65536 if v >= 32768 else v


def decode_pose(r):
    """reg14~17 → (x, y, yaw, z) 单位 mm/mm/°/mm (协议: ×0.1)."""
    return tuple(round(s16(v) / 10.0, 2) for v in r)


def expected_regs(xyz_mm, ref=None, thr=None):
    """按协议公式独立推 服务端应写出的 reg14~17 (整数×0.1, 可负).
    pos=[Y,Z,X,0]=[SIGN_X*x, SIGN_Z*y, z, 0]; X/Y/Z/yaw 换算见下; ±thr 饱和.
    ref/thr 与 plc_pose_ref.json 同步 (main 会读它传入); 兜底 ref 全 0 = 直出测量."""
    ref = ref or {"x": 0, "y1": 0, "y2": 0, "z": 0, "xita": 0}
    thr = thr or {"x": 20000, "y": 3000, "z": 2000, "xita": 50}
    x, y, z = xyz_mm
    pY, pZ, pX = SIGN_X * x, SIGN_Z * y, z          # 与 to_pos/plc_pose_to_regs 同口径
    rx = ref["x"] * 10 + round(pX * 10)
    ry = (ref["y1"] * 10 + round(pY * 10)) if pY < 0 else \
        (-ref["y2"] * 10 + round(pY * 10)) if pY > 0 else 0
    rw = ref["xita"] * 10
    rh = -ref["z"] * 10 - round(pZ * 10)

    def _cl(v, t):
        t = float(t)
        return -t if v < -t else (t if v > t else v)

    return [int(_cl(v, thr[k])) for v, k in
            ((rx, "x"), (ry, "y"), (rw, "xita"), (rh, "z"))]


# ── 小工具 ─────────────────────────────────────────────────────────
def rd(client, addr, cnt=1):
    return client.read_holding(addr, cnt)


def wr(client, addr, val):
    client.write_single(addr, val)
    time.sleep(0.05)


class Checker:
    """逐条打 PASS/FAIL, 记到错就置非零退出码."""

    def __init__(self, quiet=False):
        self.fail = 0
        self.quiet = quiet

    def check(self, name, cond, detail=""):
        tag = "✅ PASS" if cond else "❌ FAIL"
        if not cond:
            self.fail += 1
        print(f"  {tag}  {name}" + (f"   ({detail})" if detail else ""), flush=True)
        return cond

    def eq(self, name, got, exp, sname=""):
        ok = got == exp
        self.check(f"{name} = {got}" + (f" {sname}" if sname else "") +
                   ("" if ok else f"  ← 期望 {exp}"), ok)
        return ok


def wait_state(cli, target, timeout, on_change=None):
    """轮询 reg12 直到 == target; 每次变化回调 on_change(state, regs_dict).
    返回 True 命中, False 超时."""
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        try:
            st = rd(cli, VIS_STATE)[0]
        except Exception:
            time.sleep(0.05)
            continue
        if st != last:
            regs = rd(cli, VIS_X, 4) + rd(cli, VIS_FAULT, 1)
            if on_change:
                on_change(st, regs)
            last = st
        if st == target:
            return True
        time.sleep(0.05)
    return False


def sync_idle(cli):
    """确保工作状态=空闲(0). 否则车控先回 reg1=0 让服务回到空闲."""
    for _ in range(40):
        st = rd(cli, VIS_STATE)[0]
        if st == ST_IDLE:
            return True
        wr(cli, PHOTO_CMD, CMD_NONE)      # reg1=0: 清命令回空闲
        time.sleep(0.05)
    return False


# ── 场景 ───────────────────────────────────────────────────────────
def sc_heartbeat(cli, ck):
    v1 = rd(cli, VIS_HB)[0]
    time.sleep(1.3)
    v2 = rd(cli, VIS_HB)[0]
    ck.check(f"视觉心跳 reg10 翻转 ({v1}→{v2})", v1 != v2, "每 ~1s 0/1 切换")
    return ck


def sc_banner(cli, ck):
    regs = rd(cli, 0, 20)
    ck.check("连接成功 & 读到 20 个保持寄存器", len(regs) == 20,
             f"reg12状态={ST_NAMES.get(regs[VIS_STATE], regs[VIS_STATE])} "
             f"reg13故障={regs[VIS_FAULT]} reg18温度={regs[VIS_TEMP]}°C")
    return ck


def sc_shelf_once(cli, ck, expect_xyz=None, pose_cfg=None):
    print("\n── B 货架单次拍照 (reg1=3) ──", flush=True)
    if not sync_idle(cli):
        return ck.check("先复位到空闲(reg1=0)", False, "状态没回到 0")
    got_shot = {"v": False}
    wr(cli, PHOTO_CMD, CMD_SHELF)
    ok = wait_state(cli, ST_DONE, timeout=20.0, on_change=lambda st, r: None)
    ck.check("状态 → 2 拍照完成", ok, "20s 内未到 完成; 现场没货架?")
    time.sleep(0.1)
    reply = rd(cli, VIS_REPLY)[0]
    ck.check("拍照回令 reg11 = 1 (车控应答前保持)", reply == 1, f"reg11={reply}")
    pose = rd(cli, VIS_X, 4)
    x, y, yaw, z = decode_pose(pose)
    fault = rd(cli, VIS_FAULT)[0]
    print(f"  · 位姿 reg14~17={pose} → X={x}mm Y={y}mm "
          f"yaw={yaw}° Z={z}mm  (reg13故障={fault})", flush=True)
    ck.check("故障码 reg13 = 0 (成功不置故障)", fault == 0, f"reg13={fault}")
    if expect_xyz:
        exp = expected_regs(expect_xyz, **(pose_cfg or {}))
        for i, nm in zip(range(4), ("X", "Y", "yaw", "Z")):
            ck.eq(f"reg{VIS_X + i} {nm} 精确", s16(pose[i]), exp[i], "0.1mm/0.1°")
    else:
        # 无期望值时只做量程/合理域检查; 全 0 可能=恰在完美取货位, 不算错
        sane = all(abs(s16(p)) <= (20000, 3000, 500, 2000)[i] for i, p in enumerate(pose))
        ck.check("位姿在合理量程内 (未顶异常钳位)", sane,
                 f"{pose} 解码 X={x} Y={y} Z={z} (全0=完美位姿正常)")
    # 车控收位姿后应答: reg1=0 → 回空闲, 清回令
    wr(cli, PHOTO_CMD, CMD_NONE)
    ok = wait_state(cli, ST_IDLE, timeout=3.0)
    time.sleep(0.1)
    reply2 = rd(cli, VIS_REPLY)[0]
    ck.check("应答后回空闲 & 回令清 0", ok and reply2 == 0,
             f"state={rd(cli, VIS_STATE)[0]} reg11={reply2}")
    return ck


def sc_unsupported(cli, ck):
    print("\n── C 不支持命令 (reg1=5 平台单次) → 异常+故障码 ──", flush=True)
    if not sync_idle(cli):
        return ck.check("先复位到空闲", False)
    wr(cli, PHOTO_CMD, CMD_PFM)               # 5
    ok = wait_state(cli, ST_ERR, timeout=5.0)
    ck.check("状态 → 3 拍照异常", ok, "不支持命令应回异常")
    time.sleep(0.1)
    fault = rd(cli, VIS_FAULT)[0]
    ck.eq("故障码 reg13 = 42 (40-(-2))", fault, 42, f"reg13={fault}")
    pose = rd(cli, VIS_X, 4)
    ck.check("异常时位姿清 0", all(s16(p) == 0 for p in pose), f"{pose}")
    wr(cli, FAULT_RESET, 1)                   # reg2=1 故障复位
    time.sleep(0.1)
    ck.check("故障复位后 reg13 = 0", rd(cli, VIS_FAULT)[0] == 0,
             f"reg13={rd(cli, VIS_FAULT)[0]}")
    wr(cli, PHOTO_CMD, CMD_NONE)              # 回空闲
    ck.check("reg1=0 → 回空闲", wait_state(cli, ST_IDLE, timeout=3.0))
    return ck


def sc_shelf_continuous(cli, ck):
    print("\n── D 货架连续拍照 (reg1=4) ──", flush=True)
    if not sync_idle(cli):
        return ck.check("先复位到空闲", False)
    wr(cli, PHOTO_CMD, CMD_SHELF_CT)          # 4
    done_cnt, start = 0, time.time()
    while time.time() - start < 3.0:          # 观察 ~3s 连续吐位姿
        st = rd(cli, VIS_STATE)[0]
        if st == ST_DONE:
            done_cnt += 1                     # state=2 即成功 (完美位姿会合法地全 0)
        time.sleep(0.1)
    ck.check("连续模式多次拍照完成 (reg1=4 保持时)", done_cnt >= 2,
             f"3s 内见 {done_cnt} 次 done")
    wr(cli, PHOTO_CMD, CMD_NONE)              # 停止连续
    ok = wait_state(cli, ST_IDLE, timeout=3.0)
    ck.check("reg1=0 → 连续停, 回空闲", ok)
    return ck


# ── 一键本地全链路: 拉一个 --from-file 服务当被测对象 ─────────────────
def local_e2e(port, expect_xyz):
    """起子进程跑 shelf_pos_service.py --from-file, 完事杀掉."""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                      encoding="utf-8")
    json.dump({"xyz": list(expect_xyz)}, tmp)      # read_center_file 期望 {"xyz":[x,y,z]}
    tmp.close()
    log = os.path.join(tempfile.gettempdir(), "shelf_plc_e2e.log")
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "shelf_pos_service.py"),
           "--from-file", "--center-file", tmp.name, "--plc-port", str(port)]
    with open(log, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    try:
        for _ in range(100):                  # 等服务监听就绪
            try:
                s = socket.create_connection(("127.0.0.1", port), timeout=0.5)
                s.close()
                return proc
            except OSError:
                if proc.poll() is not None:
                    raise SystemExit(f"[local-e2e] 服务起不来, 看 {log}\n"
                                     + open(log).read()[-2000:])
                time.sleep(0.1)
        raise SystemExit(f"[local-e2e] 服务 10s 内没监听 {port}, 看 {log}")
    except BaseException:
        proc.terminate()
        raise


# ── 入口 ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="车控/PLC 直连 Modbus 通信测试")
    ap.add_argument("--host", default="192.168.2.102",
                    help="服务端 IP (板子默认 .102; 本地回归用 127.0.0.1)")
    ap.add_argument("--port", type=int, default=30000,
                    help="服务端 Modbus 端口 (默认 30000)")
    ap.add_argument("--expect-xyz", default=None, metavar="X,Y,Z",
                    help="已知货架中心(mm) → 精确校验位姿寄存器换算")
    ap.add_argument("--continuous", action="store_true",
                    help="加跑 D: 货架连续拍照")
    ap.add_argument("--local-e2e", action="store_true",
                    help="本地一键全链路: 起 --from-file 服务再跑全部用例")
    args = ap.parse_args()

    if args.local_e2e:
        args.host = "127.0.0.1"
        if args.port == 30000:                # 别撞本地已在跑的 30000 服务
            args.port = 31000
    xyz = tuple(float(v) for v in args.expect_xyz.split(",")) \
        if args.expect_xyz else None
    if xyz and len(xyz) != 3:
        sys.exit("--expect-xyz 要 3 个数: X,Y,Z (相机坐标 mm)")

    # 精确校验的换算参数跟 plc_pose_ref.json 走 (现场改 json 后测试自动跟着对)
    pose_cfg = None
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "plc_pose_ref.json")
    if os.path.isfile(cfg_path):
        try:
            d = json.load(open(cfg_path, encoding="utf-8"))
            if "ref" in d and "thr" in d:
                pose_cfg = {"ref": d["ref"], "thr": d["thr"]}
        except Exception:
            pass

    print(f"== 车控/PLC 直连测试 → {args.host}:{args.port} ==", flush=True)
    ck = Checker()
    proc = None
    cli = None
    try:
        if args.local_e2e:
            proc = local_e2e(args.port, xyz or (123.4, -56.7, 2100.5))
            print(f"[local-e2e] 已起 --from-file 服务 (pid {proc.pid}), "
                  f"中心点 xyz={xyz or (123.4, -56.7, 2100.5)}", flush=True)
        print("连接 Modbus…", flush=True)
        cli = ModbusClient(args.host, args.port)
        sc_banner(cli, ck)
        sc_heartbeat(cli, ck)
        sc_shelf_once(cli, ck, expect_xyz=xyz, pose_cfg=pose_cfg)
        sc_unsupported(cli, ck)
        if args.continuous or args.local_e2e:
            sc_shelf_continuous(cli, ck)
    except ConnectionRefusedError:
        print(f"❌ 连不上 {args.host}:{args.port} — 服务端没起? "
              f"(板子: systemctl status shelf-pos; "
              f"本地: python shelf_pos_service.py --from-file …)", flush=True)
        ck.fail = 1
    except OSError as e:
        print(f"❌ 网络/协议错误: {e}", flush=True)
        ck.fail = 1
    finally:
        if cli is not None:
            cli.close()
        if proc is not None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    print("\n" + ("✅ 全部通过" if ck.fail == 0 else f"❌ {ck.fail} 项失败"),
          flush=True)
    sys.exit(1 if ck.fail else 0)


if __name__ == "__main__":
    main()
