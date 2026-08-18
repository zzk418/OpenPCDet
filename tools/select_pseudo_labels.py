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

    原型以 prototype_annotations 目录里的最新标注为准:
    每簇可有多份原型 (多位姿/失败样例补充), 生成伪标签时逐帧选择最合适的。

    Returns:
        frame_clusters: {frame_stem: cluster_id}
        cluster_prototypes: {cluster_id: [proto_stem, ...]}
        proto_annotations: {proto_stem: {keypoints: [(u,v), ...]}}
    """
    with open(assignments_path) as f:
        frame_clusters = json.load(f)

    # 原型来源: prototype_annotations 目录 (最新标注为准)
    proto_ann_dir = Path(proto_ann_dir)
    proto_annotations = {}
    cluster_prototypes = {}
    for af in sorted(proto_ann_dir.glob("cluster*_anchor_v2.json")):
        with open(af) as f:
            d = json.load(f)
        if d.get("skipped"):
            continue
        stem_with_prefix = d['stem']  # e.g. "cluster0_TV_250000011092"
        kps = [(kp['pixel_uv'][0], kp['pixel_uv'][1])
                for kp in d.get('keypoints', [])]
        if not kps:
            continue
        entry = {"keypoints": kps}
        proto_annotations[stem_with_prefix] = entry
        # 裸 stem 变体 (e.g. "TV_250000011092"), 便于帧本身命中自己的标注
        parts = stem_with_prefix.split('_', 1)
        if len(parts) == 2:
            proto_annotations[parts[1]] = entry
            cid = int(parts[0].replace('cluster', ''))
            cluster_prototypes.setdefault(cid, []).append(stem_with_prefix)

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


def keypoints_to_yolo(kps, img_w=640, img_h=480, K=2):
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


# ── PCD 几何校验 (逐帧在候选原型里选最合适的) ──

FX, FY, CX, CY = 420.0, 420.0, 307.0, 264.0


def _read_pcd_xyz(path):
    """读 PCD binary, 返回 (N,3) float32 xyz。"""
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode().strip()
            if line == "DATA binary":
                break
        dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("rgb", "u4")])
        data = np.frombuffer(f.read(), dtype=dtype)
    return np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)


def _build_depth_map(xyz, img_w=640, img_h=480):
    """投影 xyz → 深度图 (z, 最近优先), 无效像素 NaN。"""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    valid = z > 1.0
    xv, yv, zv = x[valid], y[valid], z[valid]
    u = np.round(FX * xv / zv + CX).astype(np.int32)
    v = np.round(FY * yv / zv + CY).astype(np.int32)
    in_bounds = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u, v, zv = u[in_bounds], v[in_bounds], zv[in_bounds]
    dm = np.full((img_h, img_w), np.nan, dtype=np.float32)
    for i in range(len(u)):
        if np.isnan(dm[v[i], u[i]]) or zv[i] < dm[v[i], u[i]]:
            dm[v[i], u[i]] = zv[i]
    return dm


def _lookup_3d(uv, dm, window=5):
    """像素点 3D 查表: 窗口内最近深度点的 xyz, 无有效深度返回 None。"""
    u0, v0 = int(uv[0]), int(uv[1])
    h, w = dm.shape
    half = window // 2
    u_min, u_max = max(0, u0 - half), min(w, u0 + half + 1)
    v_min, v_max = max(0, v0 - half), min(h, v0 + half + 1)
    wd = dm[v_min:v_max, u_min:u_max]
    vd = wd[~np.isnan(wd)]
    if len(vd) == 0:
        return None
    d = float(np.min(vd))
    x = (u0 - CX) * d / FX
    y = (v0 - CY) * d / FY
    return (x, y, d)


def select_best_proto(proto_stems, proto_annotations, dm):
    """在候选原型里选与当前帧 PCD 几何最一致的一个。

    校验: 两 anchor 3D 宽度合理 (400~5000mm), 深度差/高度差小 (近水平横梁)。
    得分 = |Δz| + 0.2*|Δy|, 最小者胜; 全部不通过则回退第一个原型。
    """
    best = None
    for stem in proto_stems:
        kps = proto_annotations[stem]["keypoints"]
        if len(kps) < 2:
            continue
        p1 = _lookup_3d(kps[0], dm)
        p2 = _lookup_3d(kps[1], dm)
        if p1 is None or p2 is None:
            continue
        w = float(np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2))
        dz = abs(p2[2] - p1[2])
        dy = abs(p2[1] - p1[1])
        if not (400.0 <= w <= 5000.0 and dz <= 800.0 and dz <= 0.8 * w and dy <= 700.0):
            continue
        score = dz + 0.2 * dy
        if best is None or score < best[0]:
            best = (score, stem)
    return best[1] if best else proto_stems[0]


def generate_dataset(selected, cluster_prototypes, proto_annotations,
                      full_data_dir, output_dir, K=2, val_ratio=0.15, seed=42):
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
    dm_cache = {}  # frame_stem → depth_map (逐帧只建一次)

    for cid, frame_stems in selected.items():
        proto_stems = cluster_prototypes.get(cid, [])
        if not proto_stems:
            skipped_no_proto += len(frame_stems)
            continue

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

            # 帧本身就是原型 → 直接用自己的人工标注
            if frame_stem in proto_annotations:
                kps = proto_annotations[frame_stem]["keypoints"]
                proto_stem = frame_stem
            else:
                # 逐帧几何校验: 在候选原型中选与 PCD 最一致的一个
                dm = dm_cache.get(frame_stem)
                if dm is None:
                    pcd_path = full_data_dir / f"{frame_stem}.pcd"
                    if pcd_path.exists():
                        dm = _build_depth_map(_read_pcd_xyz(str(pcd_path)))
                        dm_cache[frame_stem] = dm
                proto_stem = (select_best_proto(proto_stems, proto_annotations, dm)
                              if dm is not None else proto_stems[0])
                kps = proto_annotations[proto_stem]["keypoints"]

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

    # 每帧实际使用的原型 (追溯用)
    with open(output_dir / "frame_proto_map.json", "w") as f:
        json.dump({s["stem"]: {"proto": s["proto_stem"], "cluster": s["cluster_id"]}
                   for s in all_samples}, f, indent=2, ensure_ascii=False)

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
    parser.add_argument("--K", type=int, default=2)
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
        args.full_data_dir, args.output_dir, K=args.K,
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
