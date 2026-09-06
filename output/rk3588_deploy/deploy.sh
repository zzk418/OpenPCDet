#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════
# RK3588 货架识别 — 一键部署 (板子上执行, root)
#
# PC 端只需要两行命令:
#
#   scp -r output/rk3588_deploy/ root@<板子IP>:/root/
#   ssh  root@<板子IP> "cd /root/rk3588_deploy && sudo bash deploy.sh"
#
# 本脚本依次完成:
#   1. 离线安装 python 依赖   (本地 wheelhouse, 全程无网)
#   2. 安装 MRDVS 相机 SDK     (已装则跳过)
#   3. 注册货架识别自启服务    (systemd, 开机自启)
#   4. 冒烟验证                (跑一张自带测试图)
#
# 换相机 IP: 执行前设环境变量  CAMERA_IP=192.168.2.x  (默认 192.168.2.151)
# 板子 python3 需与 wheels_aarch64 匹配 (cp38); 版本不符时依赖安装会在此报错
# ═════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")"

echo "==> [1/4] 离线安装 python 依赖 (wheelhouse, 无网)"
bash install_offline.sh

echo "==> [2/4] 检查 MRDVS 相机 SDK"
if python3 -c "import LxCameraSDK" 2>/dev/null; then
    echo "    已安装, 跳过"
else
    echo "    未安装, 执行 MRDVS_linux/install.sh (SDK → /opt/MRDVS) + python 绑定"
    ( cd MRDVS_linux && bash install.sh )
    # 离线装 python 绑定: --no-index 不联网; --no-deps 不拉它声明的 opencv-python
    # (依赖 numpy 已由 [1/4] 装好, cv2 由 opencv-python-headless 提供, 名字不同但模块同名, 会冲突故不装)
    pip install --no-index --no-deps \
        MRDVS_linux/Sample/python/lx_camera_py-1.3.3-py3-none-any.whl
fi

echo "==> [3/4] 注册货架识别自启服务 (systemd)"
bash install_shelf_service.sh

echo "==> [4/4] 冒烟验证 (一张自带测试图)"
python3 infer_image.py --input imgs/TV_250000012137.jpg

echo
echo "✔ 部署完成"
echo "  服务状态:  systemctl status shelf_pos"
echo "  运行日志:  journalctl -u shelf_pos -f"
echo "  手动全量测: bash infer_fp16.sh"
