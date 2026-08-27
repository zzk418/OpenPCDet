#!/usr/bin/env bash
# RK3588 无网离线安装依赖
#
# 用法:
#   bash install_offline.sh          # headless opencv (无显示/上位机, 推荐)
#
# 原理: pip --no-index 仅从本地 wheels_aarch64/ 安装, 全程不访问网络
set -e
cd "$(dirname "$0")"

echo "==> Python 版本 (需 3.10)"
python3 --version

echo "==> 离线安装依赖 (本地 wheelhouse)"
echo "    模式: headless opencv (无显示/上位机, 推荐)"
pip install --no-index --find-links wheels_aarch64 \
    rknn-toolkit-lite2==2.3.2 numpy opencv-python-headless

echo "==> 验证导入"
python3 - <<'PY'
import numpy, cv2
from rknnlite.api import RKNNLite
print("OK  numpy", numpy.__version__, "| opencv", cv2.__version__)
print("OK  rknnlite 导入正常")
PY

echo "==> 完成。可运行: python infer_image.py (无参数 = 推理 imgs/ 全部测试图)"
