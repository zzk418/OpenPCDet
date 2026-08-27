#!/usr/bin/env bash
# 一行配置货架识别开机自启 (在 rk3588_deploy 目录下执行):
#
#     sudo bash install_shelf_service.sh
#
# 做的事:
#   1. 检查 python3 / shelf_pos_service.py / .rknn 模型
#   2. 按【实际目录】生成 /etc/systemd/system/shelf_pos.service
#   3. 启用 + 立即启动 (开机自启 + 现在就跑)
#   4. 打印状态和最近日志
#
# 可选环境变量: CAMERA_IP 默认 192.168.2.150
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT="/etc/systemd/system/shelf_pos.service"
PY="$(command -v python3 || true)"
CAMERA_IP="${CAMERA_IP:-192.168.2.150}"

# ── 0. 前置检查 ──
[ "$(id -u)" = "0" ] || { echo "需要 root: 用 sudo bash $0"; exit 1; }
[ -n "$PY" ] || { echo "找不到 python3"; exit 1; }
[ -f "$DIR/shelf_pos_service.py" ] || { echo "缺 $DIR/shelf_pos_service.py"; exit 1; }
MODEL="$(find "$DIR" -maxdepth 1 -name '*.rknn' | head -1 || true)"
[ -n "$MODEL" ] || echo "[warn] 目录里没有 .rknn 模型, 确认推理前模型已拷到位"

# ── 1. 生成单元 (路径按实际目录, 换目录/换机器也能装) ──
cat > "$UNIT" <<EOF
[Unit]
Description=Shelf recognition service (货架识别: 监听触发 -> 推理 -> 回 pos -> PLC)
After=network-online.target

[Service]
Type=simple
WorkingDirectory=$DIR
ExecStart=$PY $DIR/shelf_pos_service.py --ip $CAMERA_IP
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
echo "[1/4] 生成 $UNIT  OK  (相机 $CAMERA_IP, 脚本 $DIR)"

# ── 2. 重载 systemd ──
systemctl daemon-reload
echo "[2/4] daemon-reload OK"

# ── 3. 启用开机自启 + 立即启动 ──
systemctl enable --now shelf_pos
echo "[3/4] enable --now shelf_pos OK"

# ── 4. 状态 ──
echo "[4/4] 状态:"
systemctl --no-pager --full status shelf_pos | head -n 8 || true
sleep 1
echo "最近日志 (journalctl -u shelf_pos):"
journalctl -u shelf_pos -n 8 --no-pager 2>/dev/null || true
