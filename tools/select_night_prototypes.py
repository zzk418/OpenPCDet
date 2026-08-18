#!/usr/bin/env python3
"""
从 福州现场数据 挑选最暗、货架面可见的帧作为夜间原型, 复制到 data/new_sheef/prototypes。

挑选规则:
  - jpg 平均亮度 <= MAX_BRIGHT (默认 95)
  - PCD 投影点中 1-3m 占比 >= MIN_FRAC (默认 8%, 保证货架面在视野内)
  - 最暗优先, 同子目录 id 间隔 >= 40 (去重连续帧), 每子目录 <= 12, 共 <= 40 帧

命名: TV_{子目录}_{id}.pcd/.jpg (与原型目录既有 clusterN_TV_*.pcd 共存)
清单: data/new_sheef/prototypes/night_manifest.json
"""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

BASE = Path("/code/OpenPCDet/data/new_sheef/福州现场数据")
OUT = Path("/code/OpenPCDet/data/new_sheef/prototypes")
MAX_BRIGHT = 95.0
MIN_FRAC = 0.08
MAX_PER_SUB = 12
ID_GAP = 40
MAX_TOTAL = 40


def read_pcd(path):
    raw = path.read_bytes()
    hdr_end = raw.find(b"DATA binary\n") + len(b"DATA binary\n")
    n = int([l for l in raw[:hdr_end].split(b"\n") if b"POINTS" in l][0].split()[-1])
    arr = np.frombuffer(raw[hdr_end:], dtype=np.uint8, count=n * 16).reshape(n, 16)
    return arr[:, :12].copy().view(np.float32).reshape(n, 3)


def scene_stats(xyz):
    z = xyz[:, 2]
    ok = (z > 300) & (z < 8000)  # mm
    u = xyz[ok, 0] / z[ok] * 420 + 307
    v = xyz[ok, 1] / z[ok] * 420 + 264
    in_f = (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
    zf = z[ok][in_f]
    if len(zf) < 100:
        return None
    return dict(median_m=round(float(np.median(zf)) / 1000, 2),
                frac_1_3m=round(float(((zf >= 1000) & (zf < 3000)).mean()), 3),
                frac_lt_1m=round(float((zf < 1000).mean()), 3),
                n=len(zf))


def main():
    cands = []
    for sub in sorted(d.name for d in BASE.iterdir() if d.is_dir()):
        for pf in sorted((BASE / sub).glob("TV_*.pcd")):
            sid = pf.stem.replace("TV_", "")
            jpg = BASE / sub / f"{sid}.jpg"
            if not jpg.exists():
                continue
            img = cv2.imread(str(jpg), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            b = img.mean()
            if b > MAX_BRIGHT:
                continue
            s = scene_stats(read_pcd(pf))
            if s is None or s["frac_1_3m"] < MIN_FRAC:
                continue
            cands.append(dict(sub=sub, sid=sid, id=int(sid), bright=round(float(b), 1), **s))

    cands.sort(key=lambda c: c["bright"])
    print(f"暗帧候选池 (亮度<={MAX_BRIGHT}, 1-3m占比>={MIN_FRAC:.0%}): {len(cands)}")

    sel = []
    for c in cands:
        if len(sel) >= MAX_TOTAL:
            break
        if sum(1 for s in sel if s["sub"] == c["sub"]) >= MAX_PER_SUB:
            continue
        if any(s["sub"] == c["sub"] and abs(s["id"] - c["id"]) < ID_GAP for s in sel):
            continue
        sel.append(c)

    print(f"选中 {len(sel)} 帧:")
    existing = {p.stem for p in OUT.iterdir()}
    manifest = []
    for c in sel:
        stem = f"TV_{c['sub']}_{c['sid']}"
        if stem in existing:
            print(f"  !! {stem} 已存在, 跳过")
            continue
        shutil.copy2(BASE / c["sub"] / f"TV_{c['sid']}.pcd", OUT / f"{stem}.pcd")
        shutil.copy2(BASE / c["sub"] / f"{c['sid']}.jpg", OUT / f"{stem}.jpg")
        manifest.append({**c, "stem": stem})
        print(f"  [{c['sub']}] {c['sid']} 亮度={c['bright']} 中位深={c['median_m']}m")

    man_path = OUT / "night_manifest.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\n已复制 {len(manifest)} 帧到 {OUT} (jpg+pcd), 清单: {man_path.name}")


if __name__ == "__main__":
    main()
