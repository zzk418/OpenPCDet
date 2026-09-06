#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""货架识别服务 → 车控/PLC 的 Modbus TCP Server (直连, 取代 Program_3).

背景: 原链路是
    车控/PLC(Modbus客户端)  ⇄  Program_3(Modbus TCP Server :30000 + JSON转发)
                          ⇄  本货架服务(JSON :5511)
去掉 Program_3 后, 本服务直接充当 Modbus TCP Server, 车控/PLC 按原协议直连
读写位姿 —— 连接角色 / 端口 / 寄存器布局 / 语义与 Program_3 的 NEWMODBUS 一致:

    HoldingRegisters 0~19
    ─ 0~9   车控(客户端)→视觉   (车控可写)
        0   叉车心跳(0-1切换)    1   拍照命令
        2   故障复位(写1清 reg13)  3~9 预留
    ─ 10~19 视觉→车控 (只读, 车控写会被忽略/恢复原值)
        10  视觉心跳(1s 0-1切换) 11  拍照回令(1=有命令待复位, 0=无)
        12  工作状态 0空闲/1拍照中/2拍照完成/3拍照异常
        13  故障码              14  位姿x(0.1mm, int16)
        15  位姿y(0.1mm, int16) 16  角度yaw(0.1°, int16)
        17  位姿z(0.1mm, int16) 18  工控机温度(°C)  19  预留

拍照命令值(reg1): 0=无动作  1=托盘单次  2=托盘连续  3=货架单次  4=货架连续
                  5=平台单次  6=平台连续    (本服务只实现 货架 3/4, 其余回"不支持")
命令是电平触发: 车控置 reg1 非 0 → 视觉 拍照中→拍照完成/异常; 车控写回 0
→ 视觉回到空闲并清拍照回令, 才能触发下一次。

纯 socket 实现, 零第三方依赖 (板端 wheelhouse 无 pymodbus)。用法:
    from plc_modbus import PlcModbusServer, RegBank, PHOTO_CMD, ...
    regs = RegBank(on_car_reg=cb)      # cb(addr,value): 车控写了 0~9
    srv = PlcModbusServer(regs, host="0.0.0.0", port=30000)
    srv.start()
    regs.write(12, 2)                  # 服务侧写 reg12=拍照完成
    ...
    srv.stop()
"""
import socket
import struct
import threading
import time

N_REGS = 20            # HoldingRegisters 0~19 (与 Program_3 一致)
CAR_BASE, VIS_BASE = 0, 10

# ── 车控→视觉 (reg 0~9) ────────────────────────────────
CAR_HEARTBEAT, PHOTO_CMD, FAULT_RESET = 0, 1, 2
# ── 视觉→车控 (reg 10~19) ──────────────────────────────
VIS_HEARTBEAT, VIS_REPLY, VIS_STATE, VIS_FAULT = 10, 11, 12, 13
VIS_X, VIS_Y, VIS_YAW, VIS_Z, VIS_TEMP, VIS_RSVD = 14, 15, 16, 17, 18, 19

# 工作状态
ST_IDLE, ST_SHOOTING, ST_DONE, ST_ERR = 0, 1, 2, 3
# 拍照命令值
CMD_NONE, CMD_PALLET, CMD_PALLET_CONT, CMD_SHELF, \
    CMD_SHELF_CONT, CMD_PLATFORM, CMD_PLATFORM_CONT = range(7)

MODBUS_EXC_ILLEGAL_FUNC = 0x01
MODBUS_EXC_ILLEGAL_ADDR = 0x02
MODBUS_EXC_ILLEGAL_VALUE = 0x03


def u16(v):
    """int → u16 (负数按 16bit 二补码)."""
    return int(v) & 0xFFFF


class RegBank:
    """20 个保持寄存器, 线程安全. 车控写 0~9 → 存下并回调; 写 10~19 → 忽略(只读保护)."""

    def __init__(self, on_car_reg=None):
        self._v = [0] * N_REGS
        self._lock = threading.Lock()
        self.on_car_reg = on_car_reg        # on_car_reg(addr, value): 车控写了 0~9

    # -- 读 (车控 0x03 用 / 服务自用) --
    def read(self, addr):
        with self._lock:
            return self._v[addr]

    def read_many(self, addr, cnt):
        with self._lock:
            return list(self._v[addr:addr + cnt])

    # -- 视觉侧写: 心跳/回令/状态/故障/位姿 等 10~19 (也允许写 0~9 内部用) --
    def write(self, addr, value):
        with self._lock:
            self._v[addr] = u16(value)

    # -- 车控侧写: 0~9 生效并触发回调; 10~19 只读忽略 --
    def _car_write(self, addr, value):
        if not (CAR_BASE <= addr < CAR_BASE + 10):
            return False
        with self._lock:
            self._v[addr] = u16(value)
        if self.on_car_reg is not None:
            try:
                self.on_car_reg(addr, self._v[addr])
            except Exception:
                pass
        return True


def _mbap(tid, unit, pdu):
    # MBAP: 事务id(2) 协议id(2)=0 长度(2)=unit+pdu 单元id(1) + pdu
    return struct.pack(">HHHB", tid, 0, 1 + len(pdu), unit) + pdu


def _exc_pdu(func, code):
    return bytes([func | 0x80, code])


class PlcModbusServer:
    """Modbus TCP Server. 与 Program_3 对齐: HoldingRegisters 0~19, 任意单元号都回.

    支持功能码: 0x03 读保持 / 0x06 写单寄存器 / 0x10 写多寄存器;
    其余(线圈/输入寄存器等)回 Modbus 异常 0x01, 寄存器越界回 0x02.
    """

    def __init__(self, regs, host="0.0.0.0", port=30000, log=print):
        self.regs = regs
        self.host = host
        self.port = int(port)
        self._log = log
        self._srv = None
        self._thread = None
        self._clients = []
        self._clients_lock = threading.Lock()
        self._stop = threading.Event()

    # ── 生命周期 ────────────────────────────────────────
    def start(self):
        if self._thread is not None:
            return
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(4)
        srv.settimeout(0.5)          # 每秒醒来查停止标志
        self._srv = srv
        self._stop.clear()
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()
        self._log(f"[plc] Modbus Server 监听 {self.host}:{self.port} "
                  f"(HoldingRegisters 0~{N_REGS - 1}, 车控直连)", flush=True)

    def stop(self):
        self._stop.set()
        srv, self._srv = self._srv, None
        if srv is not None:
            try:
                srv.close()
            except Exception:
                pass
        with self._clients_lock:
            for c in self._clients:
                try:
                    c.close()
                except Exception:
                    pass
            self._clients.clear()
        th, self._thread = self._thread, None
        if th is not None and th.is_alive():
            th.join(timeout=2.0)

    # ── accept / 每连接一个处理线程 ─────────────────────
    def _accept_loop(self):
        while not self._stop.is_set():
            srv = self._srv
            if srv is None:
                break
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn.settimeout(15.0)
            with self._clients_lock:
                self._clients.append(conn)
            self._log(f"[plc] 车控已连接 {addr}", flush=True)
            threading.Thread(target=self._client_loop,
                             args=(conn, addr), daemon=True).start()

    def _client_loop(self, conn, addr):
        try:
            while not self._stop.is_set():
                try:
                    hdr = self._recv_exact(conn, 7)
                except socket.timeout:
                    continue                    # 空闲超时只是保活检查
                if hdr is None:
                    break
                tid, pid, length = struct.unpack(">HHH", hdr[:6])
                unit = hdr[6]
                if pid != 0 or length < 2 or length > 256:
                    break
                pdu = self._recv_exact(conn, length - 1)   # 去掉 unit 后是 PDU
                if pdu is None:
                    break
                resp_pdu = self._handle_pdu(pdu)
                try:
                    conn.sendall(_mbap(tid, unit, resp_pdu))
                except OSError:
                    break
        except (ConnectionError, OSError):
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass
            with self._clients_lock:
                if conn in self._clients:
                    self._clients.remove(conn)
            self._log(f"[plc] 车控断开 {addr}", flush=True)

    @staticmethod
    def _recv_exact(s, n):
        buf = b""
        while len(buf) < n:
            try:
                chunk = s.recv(n - len(buf))
            except socket.timeout:
                return None                 # 调用方按超时处理
            if not chunk:
                return b"" if buf else None
            buf += chunk
        return buf

    # ── 功能码处理 ──────────────────────────────────────
    def _handle_pdu(self, pdu):
        func = pdu[0]
        if func == 0x03:                     # 读保持寄存器
            if len(pdu) != 5:
                return _exc_pdu(func, MODBUS_EXC_ILLEGAL_VALUE)
            addr, cnt = struct.unpack(">HH", pdu[1:5])
            if cnt == 0 or addr + cnt > N_REGS:
                return _exc_pdu(func, MODBUS_EXC_ILLEGAL_ADDR)
            vals = self.regs.read_many(addr, cnt)
            return bytes([func, 2 * cnt]) + struct.pack(">" + "H" * cnt, *vals)
        if func == 0x06:                     # 写单寄存器
            if len(pdu) != 5:
                return _exc_pdu(func, MODBUS_EXC_ILLEGAL_VALUE)
            addr, val = struct.unpack(">HH", pdu[1:5])
            if addr >= N_REGS:
                return _exc_pdu(func, MODBUS_EXC_ILLEGAL_ADDR)
            self.regs._car_write(addr, val)
            return pdu                        # 回显 0x06+addr+val
        if func == 0x10:                     # 写多寄存器
            if len(pdu) < 6:
                return _exc_pdu(func, MODBUS_EXC_ILLEGAL_VALUE)
            addr, cnt, bytecnt = struct.unpack(">HHB", pdu[1:6])
            if cnt == 0 or bytecnt != 2 * cnt or addr + cnt > N_REGS \
                    or len(pdu) != 6 + bytecnt:
                return _exc_pdu(func, MODBUS_EXC_ILLEGAL_VALUE
                                if bytecnt != 2 * cnt else MODBUS_EXC_ILLEGAL_ADDR)
            vals = struct.unpack(">" + "H" * cnt, pdu[6:6 + bytecnt])
            for i, v in enumerate(vals):
                self.regs._car_write(addr + i, v)
            return pdu[:5]                    # 0x10+addr+cnt
        return _exc_pdu(func, MODBUS_EXC_ILLEGAL_FUNC)
