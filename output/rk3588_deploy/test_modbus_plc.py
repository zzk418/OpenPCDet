#!/usr/bin/env python3
"""上位机扮演 PLC/车控, 验证 Program_3→服务→Modbus 整条链路.

不需要真 PLC, 不需要看 Qt 界面. 原理: PLC 就是 Modbus 客户端, 上位机开一个
客户端连 Program_3 的 Modbus 服务端(.102:30000), 做车控干的事:
    1. 写寄存器1 = 任务  → 触发 Program_3 拍照 (3=货架识别单次)
    2. Program_3 发触发 JSON 给服务(:5511) → 服务推理回 pos → 写寄存器14~17
    3. 轮询读寄存器 10~19 → 工作状态=2(完成) 时打印位姿

纯 socket 实现, 不依赖 pymodbus. 协议: Modbus TCP (读 0x03 / 写 0x06).

用法:
    python3 test_modbus_plc.py                          # 默认连 .102:30000, 触发货架单次
    python3 test_modbus_plc.py --ip 192.168.2.102 --port 30000 --task 3
"""
import argparse
import socket
import struct
import time


# ── Modbus TCP 报文 ──────────────────────────────────────────────
def _mbap(tid, body):
    # MBAP: 事务id(2) 协议id(2)=0 长度(2) 单元id(1)=1
    return struct.pack(">HHHB", tid, 0, len(body) + 1, 1) + body


def _recv_exact(s, n):
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("连接断开")
        buf += chunk
    return buf


def _read_holding(s, tid, start, count):
    body = struct.pack(">BHH", 0x03, start, count)
    s.sendall(_mbap(tid, body))
    hdr = _recv_exact(s, 7)                 # MBAP 7 字节
    ln = struct.unpack(">H", hdr[4:6])[0]   # 长度字段
    body = _recv_exact(s, ln - 1)           # 去掉单元id
    if body[0] & 0x80:
        raise RuntimeError(f"Modbus 异常码 {body[1]}")
    nregs = body[1] // 2      # body[1] 是字节数, 除 2 = 寄存器个数
    return list(struct.unpack(f">{nregs}H", body[2:]))


def _write_holding(s, tid, addr, value):
    body = struct.pack(">BHH", 0x06, addr, value)
    s.sendall(_mbap(tid, body))
    _recv_exact(s, 12)                      # 回显 8 字节 + MBAP 7 字节
    return tid + 1


# 视觉→车控 寄存器 10~19
REG_NAMES = {
    10: "视觉心跳", 11: "拍照回令", 12: "工作状态", 13: "故障码",
    14: "位姿x(0.1mm)", 15: "位姿y(0.1mm)", 16: "角度yaw(0.1°)", 17: "位姿z(0.1mm)",
    18: "工控机温度",
}
STATE_STR = {0: "空闲", 1: "拍照中", 2: "完成", 3: "异常"}


def main():
    ap = argparse.ArgumentParser(description="上位机扮演 PLC: 触发货架识别并读回位姿")
    ap.add_argument("--ip", default="192.168.2.102", help="Program_3 所在板端 IP")
    ap.add_argument("--port", type=int, default=30000, help="Modbus 端口 (mainwindow.cpp: MODBUS_CLIENT_PORT)")
    ap.add_argument("--task", type=int, default=3, choices=[1, 2, 3, 4],
                    help="任务: 1=托盘单次 2=托盘连续 3=货架单次(默认) 4=货架连续")
    ap.add_argument("--timeout", type=float, default=20, help="等拍照完成的最大秒数")
    args = ap.parse_args()

    tid = 1
    s = socket.create_connection((args.ip, args.port), timeout=5)
    print(f"[OK] 已连 Program_3 Modbus: {args.ip}:{args.port}")

    # 叉车心跳(寄存器0)置 1, 表示车控在线
    tid = _write_holding(s, tid, 0, 1)
    # 触发任务 (3=货架识别单次)
    tid = _write_holding(s, tid, 1, args.task)
    task = {1: "托盘单次", 2: "托盘连续", 3: "货架单次", 4: "货架连续"}[args.task]
    print(f"[触发] 写 寄存器1 = {args.task} ({task}) → Program_3 应发触发给服务, 等拍照完成...")

    last = -1
    t0 = time.time()
    while time.time() - t0 < args.timeout:
        try:
            regs = _read_holding(s, tid, 10, 10)
        except ConnectionError:
            print("[FAIL] Modbus 连接断开 (Program_3 没在跑? 端口对不对?)")
            return
        tid += 1
        state = regs[2]
        if state != last:
            print(f"  工作状态: {state} {STATE_STR.get(state, '?')}"
                  + (f"  故障码: {regs[3]}" if state == 3 else ""))
            last = state
        if state == 2:                       # 完成
            # valueSend 是 quint16, 负的位姿差值存成 u16 二补码 → 按有符号解读
            s16 = lambda v: v - 65536 if v >= 32768 else v
            print("=" * 56)
            print("✅ 拍照完成 — Program_3 已把位姿写进 Modbus (PLC 会读到的就是这份):")
            for i in range(10, 19):
                if i in REG_NAMES:
                    raw = regs[i - 10]
                    show = s16(raw) if 14 <= i <= 17 else raw
                    print(f"  寄存器 {i:>2}  {REG_NAMES[i]:<12} = {show}  "
                          f"({raw})" if 14 <= i <= 17 else
                          f"  寄存器 {i:>2}  {REG_NAMES[i]:<12} = {raw}")
            print("-" * 56)
            print(f"→ 位姿: x={s16(regs[4]) / 10:.1f}mm  y={s16(regs[5]) / 10:.1f}mm  "
                  f"yaw={s16(regs[6]) / 10:.1f}°  z={s16(regs[7]) / 10:.1f}mm")
            print("链路 OK: 服务 → Program_3 → Modbus 已闭环")
            return
        if state == 3:                       # 异常
            print(f"[FAIL] 工作状态=异常 (故障码 {regs[3]} = 40 - {regs[3]})")
            print("      检查: 服务是否在跑? CameraInfo.xml 相机1 是否指向服务:5511?")
            return
        time.sleep(0.3)

    print(f"[FAIL] {args.timeout:.0f} 秒内没等到拍照完成。")
    print("      检查: 板端 Program_3 是否在运行? 服务日志有没有 [触发]/[OK]? 相机/模型是否正常?")


if __name__ == "__main__":
    main()
