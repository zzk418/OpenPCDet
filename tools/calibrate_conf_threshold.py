#!/usr/bin/env python3
"""
在 119 帧人工审核 GT 上扫描 conf 阈值,找 P=100% 且 R 尽量高的工作点。

用法:
  python tools/calibrate_conf_threshold.py --models \
      output/shelf_pose_train/shelf_v6_s_direct_aug_c2/weights/best.pt \
      output/shelf_pose_train/shelf_v6_s_direct_aug_c2/weights/last.pt \
      output/shelf_pose_train/shelf_v6_s_direct_aug_e21_replay/weights/last.pt

匹配规则: 预测关键点与 GT 关键点像素距离 <= MATCH_PX (贪心一对一) 算 TP,
每帧只取第一个实例 (与 infer_shelf_anchor.py 一致)。
"""
import argparse
import glob
import json
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/code/OpenPCDet")
REVIEWED_DIR = REPO / "datasets/shelf_pose_pseudo/reviewed"
FULL_DATA = REPO / "data/new_sheef/pngs"
MATCH_PX = 12.0
IMG_W, IMG_H = 640, 480


def load_gt():
    """{stem: [(u, v), ...]} 仅 reviewed 且未 skipped。"""
    gt = {}
    for f in sorted(REVIEWED_DIR.glob("*_anchor_v2.json")):
        d = json.load(open(f))
        if d.get("skipped") or not d.get("reviewed"):
            continue
        kps = [(kp["pixel_uv"][0], kp["pixel_uv"][1]) for kp in d.get("keypoints", [])]
        gt[d["stem"]] = kps
    return gt


def find_image(stem):
    for ext in [".jpg", ".png"]:
        for cand_name in [stem, stem.replace("TV_", ""), f"TV_{stem.replace('TV_', '')}"]:
            cand = FULL_DATA / f"{cand_name}{ext}"
            if cand.exists():
                return cand
    return None


def run_inference(model, stems):
    """每个 stem → [{u, v, conf}...] (第一实例)。"""
    preds = {}
    for stem in sorted(stems):
        img_path = find_image(stem)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if img.shape[:2] != (IMG_H, IMG_W):
            img = cv2.resize(img, (IMG_W, IMG_H))
        results = model(img, conf=0.01, verbose=False)
        kps = []
        for r in results:
            if r.keypoints is None:
                continue
            t = r.keypoints.data
            if t is None or t.shape[0] == 0:
                continue
            for kp_idx in range(t.shape[1]):
                x, y, c = t[0, kp_idx].tolist()  # 第一实例
                if c >= 0.01:
                    kps.append({"u": x, "v": y, "conf": c})
            break  # 只取第一实例
        preds[stem] = kps
    return preds


def match_once(gt_kps, pred_kps):
    """贪心一对一匹配,返回 (n_tp, n_pred)。"""
    used = set()
    tp = 0
    for (pu, pv, _c) in pred_kps:
        best_i, best_d = None, MATCH_PX
        for i, (gu, gv) in enumerate(gt_kps):
            if i in used:
                continue
            d = ((pu - gu) ** 2 + (pv - gv) ** 2) ** 0.5
            if d < best_d:
                best_d, best_i = d, i
        if best_i is not None:
            used.add(best_i)
            tp += 1
    return tp, len(pred_kps)


def sweep(gt, preds):
    """扫描 conf ∈ [0.1, 0.95], 返回每档 (P, R) 和 P=100% 的最优档。"""
    confs = np.round(np.arange(0.10, 0.96, 0.05), 2)
    n_gt = sum(len(v) for v in gt.values())
    rows = []
    best = None  # (conf, R) at P == 1.0
    for c in confs:
        tp = fp = 0
        for stem, gk in gt.items():
            pk = [(p["u"], p["v"], p["conf"]) for p in preds.get(stem, []) if p["conf"] >= c]
            t, n_pred = match_once(gk, pk)
            tp += t
            fp += n_pred - t
        prec = tp / max(tp + fp, 1)
        rec = tp / max(n_gt, 1)
        rows.append((c, prec, rec))
        if prec >= 1.0 and (best is None or rec > best[1]):
            best = (c, rec)
    return rows, best, n_gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()

    from ultralytics import YOLO

    gt = load_gt()
    print(f"GT: {len(gt)} 帧, {sum(len(v) for v in gt.values())} 关键点")

    for mp in args.models:
        mp = Path(mp)
        if not mp.exists():
            print(f"  SKIP {mp}: not found")
            continue
        t0 = time.time()
        model = YOLO(str(mp))
        model.to("cuda:0")
        preds = run_inference(model, list(gt.keys()))
        rows, best, n_gt = sweep(gt, preds)
        print(f"\n=== {mp.parent.parent.name}/{mp.name} ({time.time()-t0:.0f}s) ===")
        print(f"  conf |    P    |    R   ")
        for c, p, r in rows:
            mark = "  <-- P=100%" if p >= 1.0 else ""
            print(f"  {c:.2f} | {p:.4f} | {r:.4f}{mark}")
        if best:
            print(f"  >>> P=100% 最优: conf={best[0]:.2f}, R={best[1]:.4f} "
                  f"({int(best[1]*n_gt)}/{n_gt} 关键点)")
        else:
            print("  >>> 无 P=100% 档位")


if __name__ == "__main__":
    main()
