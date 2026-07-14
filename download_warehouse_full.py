#!/usr/bin/env python3
"""多线程补全 warehouse_data PNG 文件 —— wget 带超时+重试, PCD 跳过"""
import os, sys, subprocess, time
os.environ.setdefault('HTTPS_PROXY', 'http://127.0.0.1:7897')
os.environ.setdefault('HTTP_PROXY', 'http://127.0.0.1:7897')

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

DATA_DIR = Path('/code/OpenPCDet/warehouse_data/data')
TOTAL = 3287
NUM_WORKERS = 16
WGET_TIMEOUT = 60
WGET_RETRIES = 3
lock = threading.Lock()
BASE_URL = 'https://huggingface.co/datasets/Voxel51/lidar-warehouse-dataset/resolve/main/data'

print("=" * 60)
print("warehouse_data PNG 下载工具 (wget + 超时)")
print("=" * 60)

# ==================== PCD 检查 ====================
pcd_count = len(list(DATA_DIR.glob('*.pcd')))
print(f"\nPCD: {pcd_count}/{TOTAL} ({pcd_count/TOTAL*100:.1f}%)")
if pcd_count < TOTAL:
    print(f"  缺 {TOTAL - pcd_count} 个 PCD，跳过")

# ==================== PNG 下载 ====================
missing_png = []
for i in range(TOTAL):
    fname = f'{i:06d}.png'
    if not (DATA_DIR / fname).exists():
        missing_png.append(fname)

png_existing = TOTAL - len(missing_png)
print(f"PNG: {png_existing}/{TOTAL} ({png_existing/TOTAL*100:.1f}%), 需下载 {len(missing_png)} 个")

if not missing_png:
    print("✅ PNG 已完整!")
else:
    ok_png = [0]
    fail_png = [0]
    n = len(missing_png)
    print(f"并发={NUM_WORKERS}, 超时={WGET_TIMEOUT}s, 重试={WGET_RETRIES}")
    print(f"预计 ~{n / NUM_WORKERS / 3:.0f} 分钟\n")

    def download_png(fname):
        dest = DATA_DIR / fname
        url = f'{BASE_URL}/{fname}'
        for attempt in range(1, WGET_RETRIES + 1):
            try:
                r = subprocess.run(
                    ['wget', '-q', f'--timeout={WGET_TIMEOUT}',
                     '--tries=1', '-O', str(dest), url],
                    timeout=WGET_TIMEOUT + 15,
                    capture_output=True,
                )
                if r.returncode == 0 and dest.stat().st_size > 0:
                    with lock:
                        ok_png[0] += 1
                        if ok_png[0] % 500 == 0 or ok_png[0] == n:
                            png_now = len(list(DATA_DIR.glob('*.png')))
                            print(f'  PNG: {png_now}/{TOTAL} ({png_now/TOTAL*100:.1f}%)')
                    return True
                else:
                    dest.unlink(missing_ok=True)
            except Exception:
                dest.unlink(missing_ok=True)
            if attempt < WGET_RETRIES:
                time.sleep(1.5 * attempt)
        with lock:
            fail_png[0] += 1
            if fail_png[0] <= 5:
                print(f'  ✗ {fname} (重试{WGET_RETRIES}次失败)')
        return False

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(download_png, f): f for f in missing_png}
        for future in as_completed(futures):
            pass

    png_final = len(list(DATA_DIR.glob('*.png')))
    print(f'\nPNG 最终: {png_final}/{TOTAL}  [成功={ok_png[0]}, 失败={fail_png[0]}]')

# ==================== 汇总 ====================
pcd_final = len(list(DATA_DIR.glob('*.pcd')))
png_final = len(list(DATA_DIR.glob('*.png')))
print(f'\n{"=" * 60}')
print(f'最终: PCD={pcd_final}/{TOTAL}, PNG={png_final}/{TOTAL}')
if png_final >= TOTAL:
    print("✅ PNG 已完整!")
else:
    print(f"⚠ 还缺 {TOTAL - png_final} 个 PNG，可重新运行本脚本")