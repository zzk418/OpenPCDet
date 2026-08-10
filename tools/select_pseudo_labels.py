#!/usr/bin/env python3
"""
从已有 DINOv2 聚类结果中，为每类选代表性帧，生成 YOLO-Pose 伪标注数据集。

原理:
  1. 读取 cluster_assignments.json (已有映射: frame_stem → cluster_id)
  2. 读取 cluster_info.json (已有原型 stem → cluster_id)
  3. 读取原型标注 (cluster*_anchor_v2.json → keypoints)
  4. 每类随机选 N 帧，用原型 keypoints 作为伪标注
  5. 生成 YOLO-Pose 数据集

用法:
  python tools/select_pseudo_labels.py --per_cluster 10
  python tools/select_pseudo_labels.py --per_cluster 10 --seed 123
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path("/code/OpenPCDet")
DEFAULT_ASSIGNMENTS = REPO_ROOT / "data/new_sheef/cluster_assignments.json"
DEFAULT_CLUSTER_INFO = REPO_ROOT / "data/new_sheef/cluster_info.json"
DEFAULT_PROTO_ANN = REPO_ROOT / "data/new_sheef/prototype_annotations"
DEFAULT_FULL_DATA = REPO_ROOT / "data/new_sheef/pngs"
DEFAULT_OUTPUT = REPO_ROOT / "datasets/shelf_pose_pseudo"


def load_data(assignments_path, cluster_info_path, proto_ann_dir):
    """加载已有聚类结果和原型标注。

    Returns:
        frame_clusters: {frame_stem: cluster_id}
        cluster_prototypes: {cluster_id: proto_stem}
        proto_annotations: {proto_stem: {keypoints: [(u,v), ...]}}
    """
    with open(assignments_path) as f:
        frame_clusters = json.load(f)

    with open(cluster_info_path) as f:
        cluster_info = json.load(f)

    # cluster_info['prototypes'] maps proto_stem → cluster_id
    # We need cluster_id → proto_stem
    cluster_prototypes = {}
    for entry in cluster_info['prototypes']:
        cluster_prototypes[int(entry['cluster_id'])] = entry['stem']

    proto_ann_dir = Path(proto_ann_dir)
    proto_annotations = {}
    for af in sorted(proto_ann_dir.glob("cluster*_anchor_v2.json")):
        with open(af) as f:
            d = json.load(f)
        stem_with_prefix = d['stem']  # e.g. "cluster0_TV_250000011092"
        # Also store by bare stem for compatibility with cluster_prototypes
        # cluster_info uses bare stems like "TV_250000011092"
        kps = [(kp['pixel_uv'][0], kp['pixel_uv'][1])
                for kp in d['keypoints']]
        entry = {"keypoints": kps}
        proto_annotations[stem_with_prefix] = entry
        # Add bare stem variant
        parts = stem_with_prefix.split('_', 1)
        if len(parts) == 2:
            proto_annotations[parts[1]] = entry

    return frame_clusters, cluster_prototypes, proto_annotations


def select_frames(frame_clusters, per_cluster=10, seed=42):
    """为每类随机选帧。

    Returns:
        selected: {cluster_id: [frame_stem, ...]}
    """
    random.seed(seed)

    # Group frames by cluster
    cluster_frames = {}
    for frame_stem, cid in frame_clusters.items():
        cid = int(cid)
        cluster_frames.setdefault(cid, []).append(frame_stem)

    selected = {}
    for cid in sorted(cluster_frames.keys()):
        frames = cluster_frames[cid]
        n = min(per_cluster, len(frames))
        chosen = random.sample(frames, n)
        selected[cid] = chosen
        print(f"  cluster{cid}: {n}/{len(frames)} frames selected")

    total = sum(len(v) for v in selected.values())
    print(f"\nTotal: {total} frames across {len(selected)} clusters")
    return selected


def keypoints_to_yolo(kps, img_w=640, img_h=480, K=4):
    """关键点列表 → YOLO pose 格式字符串。"""
    n = len(kps)
    if n == 0:
        return None

    kps_sorted = sorted(kps, key=lambda k: k[0])
    us = [k[0] / img_w for k in kps_sorted]
    vs = [k[1] / img_h for k in kps_sorted]

    cx = (min(us) + max(us)) / 2
    cy = (min(vs) + max(vs)) / 2
    bw = min(max(max(us) - min(us), 0.05) * 1.2, 1.0)
    bh = min(max(max(vs) - min(vs), 0.05) * 1.2, 1.0)

    # 将 n 个关键点均匀分配到 K 个 slot
    slots = [0] if n == 1 else np.round(np.linspace(0, K - 1, n)).astype(int).tolist()
    slot_set = set(slots)
    slot_map = {s: i for i, s in enumerate(slots)}

    norm = []
    for i in range(K):
        if i in slot_set:
            u, v = kps_sorted[slot_map[i]]
            norm.extend([u / img_w, v / img_h, 2.0])
        else:
            norm.extend([0.0, 0.0, 0.0])

    return (f"0.000000 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " +
            " ".join(f"{x:.6f}" for x in norm))


def generate_dataset(selected, cluster_prototypes, proto_annotations,
                      full_data_dir, output_dir, K=4, val_ratio=0.15, seed=42):
    """生成 YOLO-Pose 数据集。"""
    random.seed(seed)

    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (output_dir / sub).mkdir(parents=True)

    full_data_dir = Path(full_data_dir)
    img_w, img_h = 640, 480

    # 收集所有样本
    all_samples = []
    skipped_no_img = 0
    skipped_no_proto = 0

    for cid, frame_stems in selected.items():
        if cid not in cluster_prototypes:
            skipped_no_proto += len(frame_stems)
            continue
        proto_stem = cluster_prototypes[cid]
        if proto_stem not in proto_annotations:
            skipped_no_proto += len(frame_stems)
            continue
        kps = proto_annotations[proto_stem]["keypoints"]

        for frame_stem in frame_stems:
            # 找图像文件 (优先 jpg)
            img_path = None
            for ext in [".jpg", ".png"]:
                cand = full_data_dir / f"{frame_stem}{ext}"
                if cand.exists():
                    img_path = cand
                    break
            if img_path is None:
                skipped_no_img += 1
                continue

            all_samples.append({
                "stem": frame_stem,
                "img_path": str(img_path),
                "keypoints": kps[:],
                "proto_stem": proto_stem,
                "cluster_id": cid,
            })

    print(f"\nSamples: {len(all_samples)} total"
          f" (no_img={skipped_no_img}, no_proto={skipped_no_proto})")

    # 分割 train/val (stratified: 每类都有 val)
    random.shuffle(all_samples)
    n_val = max(1, int(len(all_samples) * val_ratio))

    for split, samples in [("train", all_samples[n_val:]),
                            ("val", all_samples[:n_val])]:
        kp_total = 0
        for s in samples:
            img = cv2.imread(s["img_path"])
            if img is None:
                continue
            if img.shape[:2] != (img_h, img_w):
                img = cv2.resize(img, (img_w, img_h))

            label = keypoints_to_yolo(s["keypoints"], img_w, img_h, K=K)
            if label is None:
                continue

            cv2.imwrite(
                str(output_dir / "images" / split / f"{s['stem']}.jpg"),
                img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            with open(output_dir / "labels" / split / f"{s['stem']}.txt", "w") as f:
                f.write(label + "\n")
            kp_total += len(s["keypoints"])

        print(f"  [{split}] {len(samples)} samples, {kp_total} kp")

    # dataset.yaml
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(f"""# 货架关键点检测 — 伪标注数据集
# 12 clusters × {args.per_cluster} frames, 原型 keypoints 作为伪标注
path: {output_dir.resolve()}
train: images/train
val: images/val
kpt_shape: [{K}, 3]
flip_idx: [{', '.join(str(i) for i in range(K))}]
names:
  0: shelf
nc: 1
""")

    print(f"\nDataset: {output_dir}")
    print(f"  Train: {len(all_samples) - n_val}, Val: {n_val}")
    return yaml_path


def main():
    global args
    parser = argparse.ArgumentParser(
        description="选代表性帧 + 伪标注 → YOLO-Pose 数据集")
    parser.add_argument("--assignments", default=str(DEFAULT_ASSIGNMENTS))
    parser.add_argument("--cluster_info", default=str(DEFAULT_CLUSTER_INFO))
    parser.add_argument("--proto_annotations", default=str(DEFAULT_PROTO_ANN))
    parser.add_argument("--full_data_dir", default=str(DEFAULT_FULL_DATA))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--per_cluster", type=int, default=10)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading data...")
    frame_clusters, cluster_prototypes, proto_annotations = load_data(
        args.assignments, args.cluster_info, args.proto_annotations)
    print(f"  Frames: {len(frame_clusters)}")
    print(f"  Clusters: {len(cluster_prototypes)}")
    print(f"  Prototypes annotated: {len(proto_annotations)}")

    print(f"\nSelecting {args.per_cluster} frames per cluster...")
    selected = select_frames(frame_clusters, args.per_cluster, args.seed)

    # Save selection log now (before generate_dataset deletes output_dir)
    output_dir = Path(args.output_dir)
    selection_log = {
        "per_cluster": args.per_cluster,
        "seed": args.seed,
        "selected": {f"cluster{cid}": frames for cid, frames in selected.items()},
    }

    print(f"\nGenerating YOLO-Pose dataset...")
    yaml_path = generate_dataset(
        selected, cluster_prototypes, proto_annotations,
        args.full_data_dir, args.output_dir,
        val_ratio=args.val_ratio, seed=args.seed)

    # Save selection log alongside generated dataset
    with open(output_dir / "frame_selection.json", "w") as f:
        json.dump(selection_log, f, indent=2, ensure_ascii=False)
    print(f"  Selection log: {output_dir / 'frame_selection.json'}")

    print(f"\nDone! To train:")
    print(f"  python tools/augment_proto_dataset.py "
          f"--output_dataset {args.output_dir} --k 4 --epochs 100")


if __name__ == "__main__":
    main()
