#!/usr/bin/env bash
# RK3588 板端推理脚本
#
# 用法:
#   bash infer_int8.sh                            # 离线推理 imgs/ 全部测试图 (10张)
#   bash infer_int8.sh --single <图路径> [参数]   # 单张图, 如 --single imgs/test_night.jpg --out /root/results
#   bash infer_int8.sh --camera [编号] [参数]     # 摄像头实时推理 (默认 /dev/video0), 需显示器弹窗
#   bash infer_int8.sh --camera-json [编号]       # 摄像头实时推理, JSON 输出 (无显示/上位机)
#
# 追加的原生参数会透传给底层脚本, 例如:
#   bash infer_int8.sh --camera 0 --conf 0.3 --cores 3
#   bash infer_int8.sh --single imgs/test.jpg --out /root/results
set -e
cd "$(dirname "$0")"

MODE="${1:-}"
shift 2>/dev/null || true

case "$MODE" in
  "")
    echo "模式: 离线推理 imgs/ 全部测试图"
    exec python3 infer_image.py
    ;;
  --single)
    [ -n "$1" ] || { echo "错误: --single 需要图片路径, 如 --single imgs/test.jpg"; exit 1; }
    exec python3 infer_image.py --input "$@"
    ;;
  --camera)
    if [ -n "$1" ]; then
      exec python3 infer_camera.py --camera "$@"
    else
      exec python3 infer_camera.py "$@"
    fi
    ;;
  --camera-json)
    exec python3 infer_camera.py --no-display --json --camera "$@"
    ;;
  -h|--help)
    sed -n '2,12p' "$0"
    ;;
  *)
    echo "未知模式: $MODE"
    echo "用法: bash infer.sh [--single <图>|--camera [编号]|--camera-json [编号]]"
    exit 1
    ;;
esac
