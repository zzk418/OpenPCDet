#!/usr/bin/env bash
# RK3588 无网离线安装依赖 (可重复执行: 已装齐则跳过安装, 只自检)
#
# 用法:
#   bash install_offline.sh          # headless opencv (无显示/上位机, 推荐)
#
# 行为:
#   - 已装齐 (rknn-toolkit-lite2==2.3.2 + numpy + opencv) → 跳过 pip install, 只做自检
#   - 未装齐 / 版本不配套 (≠2.3.2) → pip --no-index 仅从本地 wheels_aarch64/ 安装, 全程无网
set -e
cd "$(dirname "$0")"

echo "==> Python 版本: $(python3 --version 2>&1)"

# ── 已装检查: 三者齐且 rknn-toolkit-lite2 为 2.3.2 (与 librknnrt.so 配套) 才算装好 ──
if python3 -c '
import importlib.metadata as md, numpy, cv2
from rknnlite.api import RKNNLite
ver = md.version("rknn-toolkit-lite2")
assert ver == "2.3.2", f"版本 {ver} 与 librknnrt.so 不配套, 需重装"
' 2>/dev/null; then
    echo "==> 依赖已装齐 (rknn-toolkit-lite2 2.3.2 + numpy + opencv), 跳过安装"
else
    echo "==> 未装齐, 离线安装依赖 (本地 wheelhouse)"
    echo "    模式: headless opencv (无显示/上位机, 推荐)"
    pip install --no-index --find-links wheels_aarch64 \
        rknn-toolkit-lite2==2.3.2 numpy opencv-python-headless
fi

echo "==> 验证导入"
python3 - <<'PY'
import numpy, cv2
from rknnlite.api import RKNNLite
print("OK  numpy", numpy.__version__, "| opencv", cv2.__version__)
print("OK  rknnlite 导入正常")
PY

echo "==> 完成。可运行: python infer_image.py (无参数 = 推理 imgs/ 全部测试图)"
