#!/usr/bin/env python3
"""
用 v4 模型预测为增量 cluster 生成伪标签。
相比直接复制原型关键点，模型预测已学会视角变化，质量更高。

用法:
  python tools/model_pseudo_label.py                      # 全流程
  python tools/model_pseudo_label.py --per_cluster 15     # 每类 15 帧
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/code/OpenPCDet")
INFERENCE_JSON = REPO / "output/shelf_pose_inference_v4/keypoints/all_keypoints.json"
ASSIGNMENTS = REPO / "data/new_sheef/cluster_assignments.json"
FULL_DATA = REPO / "data/new_sheef/pngs"
REVIEWED_DIR = REPO / "datasets/shelf_pose_pseudo/reviewed"

# 只处理新增原型的 cluster
TARGET_CLUSTERS = {1, 3, 4, 5, 7, 9}


def load_data():
    with open(INFERENCE_JSON) as f:
        inference = json.load(f)
    with open(ASSIGNMENTS) as f:
        assignments = json.load(f)
    return inference, assignments


def select_frames(inference, assignments, per_cluster=15, min_kp_conf=0.5, min_box_conf=0.4, seed=42):
    """从目标 cluster 中选帧，要求模型预测置信度高。"""
    random.seed(seed)

    # 按 cluster 分组，记录最高置信度的帧
    cluster_candidates = {}
    for stem, dets in inference.items():
        cid = int(assignments.get(stem, -1))
        if cid not in TARGET_CLUSTERS:
            continue
        if not dets:
            continue
        # 取最佳检测
        best = max(dets, key=lambda d: d["box_conf"])
        if best["box_conf"] < min_box_conf:
            continue
        kps = best["keypoints"]
        valid_kps = [(x, y, c) for x, y, c in kps if c > min_kp_conf]
        if len(valid_kps) < 1:
            continue
        score = best["box_conf"] * sum(c for _, _, c in valid_kps) / len(valid_kps)
        cluster_candidates.setdefault(cid, []).append((stem, score, best))

    # 每类选 top-N
    selected = {}
    for cid in sorted(cluster_candidates.keys()):
        candidates = cluster_candidates[cid]
        candidates.sort(key=lambda x: -x[1])  # 按综合置信度降序
        n = min(per_cluster, len(candidates))
        chosen = candidates[:n]
        selected[cid] = chosen
        scores = [c[1] for c in chosen]
        print(f"  cluster{cid}: {n}/{len(candidates)} frames "
              f"(score range {min(scores):.3f}~{max(scores):.3f})")

    total = sum(len(v) for v in selected.values())
    print(f"\nTotal: {total} frames across {len(selected)} clusters")
    return selected


def save_as_reviewed(selected, reviewed_dir):
    """将模型预测保存为 anchor_v2.json 格式（可直接在 web 审核）。"""
    reviewed_dir = Path(reviewed_dir)
    reviewed_dir.mkdir(parents=True, exist_ok=True)

    # 先清理旧的伪标签（仅目标 cluster）
    for old in reviewed_dir.glob("*_anchor_v2.json"):
        old.unlink()

    saved = 0
    all_annotations = {}

    for cid, frames in selected.items():
        for stem, score, det in frames:
            kps = det["keypoints"]
            kp_list = []
            for i, (x, y, c) in enumerate(kps):
                if c > 0.5:
                    kp_list.append({
                        "id": i,
                        "pixel_uv": [round(x, 2), round(y, 2)],
                    })

            ann = {
                "stem": stem,
                "cluster_id": cid,
                "keypoints": kp_list,
                "source": f"model_v4_score_{score:.3f}",
                "reviewed": False,
            }
            all_annotations[stem] = ann
            saved += 1

    # 写入 reviewed 目录
    for stem, ann in all_annotations.items():
        path = reviewed_dir / f"{stem}_anchor_v2.json"
        with open(path, "w") as f:
            json.dump(ann, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {saved} pseudo-labels to {reviewed_dir}")
    return all_annotations


def main():
    parser = argparse.ArgumentParser(description="v4 模型预测 → 增量伪标签")
    parser.add_argument("--per_cluster", type=int, default=15)
    parser.add_argument("--min_kp_conf", type=float, default=0.5)
    parser.add_argument("--min_box_conf", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading v4 inference results + cluster assignments...")
    inference, assignments = load_data()
    print(f"  Inference frames: {len(inference)}")
    print(f"  Target clusters: {sorted(TARGET_CLUSTERS)}")

    print(f"\nSelecting top-{args.per_cluster} per cluster (min box_conf={args.min_box_conf}, kp_conf={args.min_kp_conf})...")
    selected = select_frames(inference, assignments,
                             per_cluster=args.per_cluster,
                             min_kp_conf=args.min_kp_conf,
                             min_box_conf=args.min_box_conf,
                             seed=args.seed)

    print(f"\nSaving as reviewed format...")
    save_as_reviewed(selected, REVIEWED_DIR)

    print(f"\nDone! Next steps:")
    print(f"  1. Review in web: python tools/shelf_anchor_v3_web.py")
    print(f"  2. Train: python tools/reviewed_to_train.py")


if __name__ == "__main__":
    main()
