#!/usr/bin/env python3
"""
Prepare shelf keypoint annotations for YOLOv8-pose training.

Converts output/shelf_anchor_v2/*.json → YOLO keypoint format with train/val split.

YOLO pose format (per image .txt):
  class_id bbox_cx bbox_cy bbox_w bbox_h x1 y1 v1 x2 y2 v2 ... xk yk vk

- Coordinates normalized to 0-1
- v=2 (visible+labeled), v=0 (not present)
- K=4 keypoints, ordered left-to-right by u-coordinate
- Bounding box = min/max of visible keypoints + 10% padding

Output:
  datasets/shelf_pose/
    images/train/, images/val/
    labels/train/, labels/val/
    dataset.yaml

Usage:
  python tools/prepare_shelf_pose.py
  python tools/prepare_shelf_pose.py --split 0.7 --k 4
"""

import argparse, json, os, random, sys, shutil
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np


def load_annotations(ann_dir: str, exclude_stems: set = None):
    """Load all annotations, handling both old and new JSON formats.

    Returns: dict[stem] = [{pixel_uv: [u,v], anchor_3d: [x,y,z], label: str}, ...]
    """
    if exclude_stems is None:
        exclude_stems = set()

    ann_dir = Path(ann_dir)
    annotations = {}

    for jf in sorted(ann_dir.glob("*.json")):
        stem = jf.stem.replace("_anchor_v2", "")
        if stem in exclude_stems:
            continue

        with open(jf) as f:
            d = json.load(f)

        if d.get("skipped"):
            continue

        kps = []
        if "keypoints" in d:
            for kp in d["keypoints"]:
                kps.append({
                    "pixel_uv": kp.get("pixel_uv", [0, 0]),
                    "anchor_3d": kp.get("anchor_3d"),
                    "label": kp.get("label", ""),
                })
        elif "anchor_3d" in d:
            # Old single-anchor format
            kps.append({
                "pixel_uv": d.get("pixel_uv", [0, 0]),
                "anchor_3d": d.get("anchor_3d"),
                "label": d.get("label", ""),
            })

        if kps:
            annotations[stem] = kps

    return annotations


def keypoints_to_yolo(kps, img_w, img_h, K=4):
    """Convert keypoints to YOLO pose format string.

    Keypoints are sorted left-to-right (by u), padded to K slots.
    Bounding box = min/max of visible keypoints + 10% padding.
    """
    if not kps:
        return None

    # Sort by u (left to right)
    sorted_kps = sorted(kps, key=lambda k: k["pixel_uv"][0])
    n = len(sorted_kps)

    # Spread keypoints across K slots: first → Slot0, last → Slot(K-1), middle evenly
    # 1 kp → Slot0; 2 kp → Slot0 + Slot(K-1); 3 kp → Slot0 + Slot1 + Slot2
    if n == 1:
        slot_positions = [0]
    else:
        slot_positions = np.round(np.linspace(0, K - 1, n)).astype(int).tolist()

    norm_kps = []
    slot_set = set(slot_positions)
    for i in range(K):
        if i in slot_set:
            idx = slot_positions.index(i)
            u, v = sorted_kps[idx]["pixel_uv"]
            norm_kps.extend([u / img_w, v / img_h, 2.0])  # v=2 visible
        else:
            norm_kps.extend([0.0, 0.0, 0.0])  # v=0 not present

    # Bounding box from visible keypoints
    visible = sorted_kps
    us = [k["pixel_uv"][0] / img_w for k in visible]
    vs = [k["pixel_uv"][1] / img_h for k in visible]

    cx = (min(us) + max(us)) / 2
    cy = (min(vs) + max(vs)) / 2
    bw = max(max(us) - min(us), 0.02) * 1.2  # 20% padding
    bh = max(max(vs) - min(vs), 0.02) * 1.2

    # Clamp to [0,1]
    bw = min(bw, 1.0)
    bh = min(bh, 1.0)

    parts = [0, cx, cy, bw, bh] + norm_kps  # class 0 = shelf
    return " ".join(f"{x:.6f}" for x in parts)


def prepare_dataset(annotations, data_dir, output_dir, split=0.7, K=4, seed=42):
    """Create YOLO dataset directory structure."""
    random.seed(seed)

    output_dir = Path(output_dir)
    data_dir = Path(data_dir)

    # Clean
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Create dirs
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (output_dir / sub).mkdir(parents=True)

    stems = list(annotations.keys())
    random.shuffle(stems)
    n_train = int(len(stems) * split)
    train_stems = sorted(stems[:n_train])
    val_stems = sorted(stems[n_train:])

    print(f"Train: {len(train_stems)}, Val: {len(val_stems)}")

    def process_split(stems, split_name):
        img_w, img_h = 640, 480
        kp_total = 0
        skipped = 0

        for stem in stems:
            num_id = stem.replace("TV_", "")

            # Find image
            img_path = None
            for ext in [".jpg", ".png"]:
                cand = data_dir / f"{num_id}{ext}"
                if cand.exists():
                    img_path = cand
                    break

            if img_path is None:
                skipped += 1
                continue

            # Read + resize image to 640×480
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue
            if img.shape[0] != img_h or img.shape[1] != img_w:
                img = cv2.resize(img, (img_w, img_h))

            # Save image
            out_img = output_dir / "images" / split_name / f"{stem}.jpg"
            cv2.imwrite(str(out_img), img, [cv2.IMWRITE_JPEG_QUALITY, 92])

            # Write label
            kps = annotations[stem]
            label_str = keypoints_to_yolo(kps, img_w, img_h, K=K)
            if label_str is None:
                skipped += 1
                continue

            out_label = output_dir / "labels" / split_name / f"{stem}.txt"
            with open(out_label, "w") as f:
                f.write(label_str + "\n")

            kp_total += min(len(kps), K)

        print(f"  [{split_name}] {len(stems)} frames, {kp_total} keypoints, {skipped} skipped")
        return kp_total

    t_kp = process_split(train_stems, "train")
    v_kp = process_split(val_stems, "val")

    # Write dataset.yaml
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(f"""# Shelf Keypoint Detection Dataset
path: {output_dir.resolve()}
train: images/train
val: images/val

# Keypoint config
kpt_shape: [{K}, 3]  # K keypoints, (x, y, visibility)
flip_idx: [{", ".join(str(i) for i in range(K))}]  # no symmetric flip for shelves

# Classes
names:
  0: shelf
nc: 1
""")

    print(f"\nDataset ready: {output_dir}")
    print(f"  YAML: {yaml_path}")
    print(f"  Total keypoints: train={t_kp}, val={v_kp}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Prepare shelf YOLO-pose dataset")
    parser.add_argument("--ann_dir", default="output/shelf_anchor_v2")
    parser.add_argument("--data_dir", default="data/new_sheef/pngs")
    parser.add_argument("--output_dir", default="datasets/shelf_pose")
    parser.add_argument("--split", type=float, default=0.7)
    parser.add_argument("--k", type=int, default=3, help="Max keypoints per instance")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude", nargs="*", default=["TV_250000000001"],
                        help="Stems to exclude")
    args = parser.parse_args()

    exclude = set(args.exclude)

    print(f"Loading annotations from {args.ann_dir}...")
    annotations = load_annotations(args.ann_dir, exclude_stems=exclude)
    print(f"Loaded {len(annotations)} annotated frames")

    # Stats
    kp_counts = {}
    for stem, kps in annotations.items():
        n = len(kps)
        kp_counts[n] = kp_counts.get(n, 0) + 1
    print(f"KP distribution: {dict(sorted(kp_counts.items()))}")
    print(f"Total keypoints: {sum(len(v) for v in annotations.values())}")

    yaml_path = prepare_dataset(
        annotations,
        args.data_dir,
        args.output_dir,
        split=args.split,
        K=args.k,
        seed=args.seed,
    )

    # Verify
    print(f"\nVerifying...")
    for split in ["train", "val"]:
        img_dir = Path(args.output_dir) / "images" / split
        lbl_dir = Path(args.output_dir) / "labels" / split
        n_imgs = len(list(img_dir.glob("*.jpg")))
        n_lbls = len(list(lbl_dir.glob("*.txt")))
        print(f"  {split}: {n_imgs} images, {n_lbls} labels")
        assert n_imgs == n_lbls, f"Mismatch: {n_imgs} vs {n_lbls}"


if __name__ == "__main__":
    main()
