#!/usr/bin/env python3
"""
Proper Precision-Recall curve generator for CenterPoint++ evaluation.
Computes PR directly from all detections (not via 41-point threshold sampling),
then interpolates to evenly-spaced recall levels for standard KITTI-style curves.

Usage:
    cd /code/OpenPCDet
    conda activate pc
    python tools/plot_pr_curves.py
"""

import copy
import pickle
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
from pcdet.datasets.kitti import kitti_utils
from pcdet.datasets.kitti.kitti_object_eval_python.rotate_iou import rotate_iou_gpu_eval

# ── Config ──────────────────────────────────────────────────────────
RESULT_PKL = "output/custom_models/centerpoint_pp_warehouse/default/eval/eval_mid_train/epoch_30/val/result.pkl"
GT_INFO_PKL = "data/warehouse/warehouse_infos_val.pkl"
OUTPUT_DIR = Path("output/pr_curves")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['箱子', '电动运输车', '货运自行车', '无人搬运车', '叉车']
CLASS_NAMES_EN = ['Box', 'ELF', 'CargoBike', 'FTS', 'ForkLift']

# CargoBike (货运自行车) excluded: only 1 GT instance in val set → AP not meaningful
PLOT_CLASSES = ['箱子', '电动运输车', '无人搬运车', '叉车']
PLOT_CLASSES_EN = ['Box', 'ELF', 'FTS', 'ForkLift']
PLOT_PALETTE = ['#1B9E77', '#D95F02', '#E7298A', '#66A61E']
MAP_NAME_TO_KITTI = {c: c for c in CLASS_NAMES}

# Colorblind-safe palette
PALETTE = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#66A61E']
METRIC_COLORS = {'3d': '#1B9E77', 'bev': '#7570B3'}

IOU_THRESH = {'3d': 0.5, 'bev': 0.25}


def load_data():
    print(f"Loading predictions from {RESULT_PKL}...")
    with open(RESULT_PKL, 'rb') as f:
        det_annos = pickle.load(f)
    print(f"  {len(det_annos)} frames")

    print(f"Loading GT from {GT_INFO_PKL}...")
    with open(GT_INFO_PKL, 'rb') as f:
        gt_infos = pickle.load(f)
    print(f"  {len(gt_infos)} frames")

    return det_annos, gt_infos


def compute_pr_raw(gt_annos, dt_annos, class_idx, metric):
    """
    Compute raw precision-recall by treating EVERY detection as a threshold.
    Returns (recall_array, precision_array) sorted by descending score.

    metric: 1=bev, 2=3d
    """
    metric_name = 'bev' if metric == 1 else '3d'
    iou_thresh = IOU_THRESH[metric_name]

    # Collect all detections globally
    all_entries = []
    total_gt = 0

    for frame_idx, (gt, dt) in enumerate(zip(gt_annos, dt_annos)):
        # Build class masks — handle both numpy arrays and lists
        dt_names = np.asarray(dt['name'])
        gt_names = np.asarray(gt['name'])

        cls_mask = dt_names == CLASS_NAMES[class_idx]
        dt_scores = np.asarray(dt['score'])[cls_mask]
        dt_locs = np.asarray(dt['location'])[cls_mask]
        dt_dims = np.asarray(dt['dimensions'])[cls_mask]
        dt_rots = np.asarray(dt['rotation_y'])[cls_mask]

        gt_mask = gt_names == CLASS_NAMES[class_idx]
        gt_locs = np.asarray(gt['location'])[gt_mask]
        gt_dims = np.asarray(gt['dimensions'])[gt_mask]
        gt_rots = np.asarray(gt['rotation_y'])[gt_mask]

        n_dt = len(dt_scores)
        n_gt = len(gt_locs)  # actual GT count for this class in this frame
        total_gt += n_gt

        if n_dt == 0:
            continue

        # Skip if no GT (all detections for this class are FP, recorded below)
        if n_gt == 0:
            for d in range(n_dt):
                all_entries.append({
                    'score': float(dt_scores[d]),
                    'frame_idx': frame_idx,
                    'best_iou': 0.0,
                    'best_gt_idx': -1,
                })
            continue

        # Compute IoU (n_gt > 0 guaranteed here)
        if metric == 1:  # bev
            gt_bev = np.concatenate([gt_locs[:, [0, 2]], gt_dims[:, [0, 2]], gt_rots[:, None]], axis=1)
            dt_bev = np.concatenate([dt_locs[:, [0, 2]], dt_dims[:, [0, 2]], dt_rots[:, None]], axis=1)
            ious = rotate_iou_gpu_eval(gt_bev, dt_bev, criterion=-1).T
        else:  # 3d
            gt_3d = np.concatenate([gt_locs, gt_dims, gt_rots[:, None]], axis=1)
            dt_3d = np.concatenate([dt_locs, dt_dims, dt_rots[:, None]], axis=1)
            ious = kitti_eval.d3_box_overlap(gt_3d, dt_3d, criterion=-1).T

        for d in range(n_dt):
            best_gt = np.argmax(ious[d])
            best_iou = ious[d, best_gt]
            all_entries.append({
                'score': float(dt_scores[d]),
                'frame_idx': frame_idx,
                'best_iou': float(best_iou),
                'best_gt_idx': int(best_gt),
            })

    if len(all_entries) == 0:
        return np.array([0.0]), np.array([0.0]), total_gt

    # Sort by score descending
    all_entries.sort(key=lambda x: x['score'], reverse=True)

    # Global greedy matching: for each detection (sorted by score), assign to best unmatched GT
    frame_gt_matched = [set() for _ in range(len(gt_annos))]
    tp_cumsum = 0
    fp_cumsum = 0
    recall_arr = np.zeros(len(all_entries))
    precision_arr = np.zeros(len(all_entries))

    for i, entry in enumerate(all_entries):
        fidx = entry['frame_idx']
        iou = entry['best_iou']
        gt_idx = entry['best_gt_idx']

        if iou >= iou_thresh and gt_idx >= 0 and gt_idx not in frame_gt_matched[fidx]:
            tp_cumsum += 1
            frame_gt_matched[fidx].add(gt_idx)
        else:
            fp_cumsum += 1

        recall_arr[i] = tp_cumsum / max(total_gt, 1)
        precision_arr[i] = tp_cumsum / max(tp_cumsum + fp_cumsum, 1)

    return recall_arr, precision_arr, total_gt


def interpolate_pr_at_recall_levels(recall_raw, precision_raw, num_points=101):
    """
    Standard KITTI PR curve interpolation:
    For each recall level r in [0, 1/(N-1), ..., 1.0]:
        precision[r] = max{precision[t] | recall[t] >= r}
    """
    mask = recall_raw > 0.001
    if not mask.any():
        return np.zeros(num_points), np.zeros(num_points)

    rec = recall_raw[mask]
    prec = precision_raw[mask]

    # Sort by recall for clean interpolation
    order = np.argsort(rec)
    rec = rec[order]
    prec = prec[order]

    recall_levels = np.linspace(0, 1.0, num_points)
    interp_prec = np.zeros(num_points)

    for i, r in enumerate(recall_levels):
        valid = rec >= r
        if valid.any():
            interp_prec[i] = np.max(prec[valid])
        else:
            interp_prec[i] = 0.0

    return recall_levels, interp_prec


def compute_all_pr_curves(gt_annos, dt_annos):
    """Compute PR curves for all classes and metrics."""
    results = {}

    for metric_val, metric_name in [(2, '3d'), (1, 'bev')]:
        print(f"\nComputing PR curves for {metric_name.upper()} (IoU={IOU_THRESH[metric_name]})...")
        results[metric_name] = {}

        for c in range(len(CLASS_NAMES)):
            recall_raw, precision_raw, total_gt = compute_pr_raw(
                gt_annos, dt_annos, c, metric_val)

            # Interpolate to 101 recall levels for smooth plotting
            recall_interp, precision_interp = interpolate_pr_at_recall_levels(
                recall_raw, precision_raw, num_points=101)

            # Interpolate to 41 points for AP calculation (KITTI standard)
            _, precision_41 = interpolate_pr_at_recall_levels(
                recall_raw, precision_raw, num_points=41)

            ap_11 = np.mean(precision_41[::4]) * 100  # 11-point
            ap_r40 = np.mean(precision_41[1:]) * 100  # 40-point

            results[metric_name][CLASS_NAMES[c]] = {
                'recall_raw': recall_raw.tolist(),
                'precision_raw': precision_raw.tolist(),
                'recall_interp': recall_interp.tolist(),
                'precision_interp': precision_interp.tolist(),
                'total_gt': int(total_gt),
                'AP_11': round(ap_11, 2),
                'AP_R40': round(ap_r40, 2),
            }
            print(f"  {CLASS_NAMES[c]:6s}: GT={total_gt:3d}, AP_11={ap_11:.2f}%, AP_R40={ap_r40:.2f}%")

    # mAP (excluding CargoBike: only 1 GT → AP not meaningful)
    for metric_name in ['3d', 'bev']:
        aps = [results[metric_name][c]['AP_R40'] for c in PLOT_CLASSES]
        results[metric_name]['mAP'] = round(np.mean(aps), 2)
        print(f"\n  {metric_name} mAP (R40, excl. CargoBike): {results[metric_name]['mAP']:.2f}%")

    return results


def plot_pr_curves(results):
    """Generate proper PR curves with raw points + interpolated curve."""

    # ── Figure 1: Per-class 3D PR ──────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle('CenterPoint++ Epoch 30 — 3D Precision-Recall (IoU=0.5)',
                 fontsize=14, fontweight='bold', y=1.02)

    for c, (ax, cls_cn, cls_en) in enumerate(zip(axes, PLOT_CLASSES, PLOT_CLASSES_EN)):
        data = results['3d'][cls_cn]
        recall_raw = np.array(data['recall_raw'])
        precision_raw = np.array(data['precision_raw'])
        recall_interp = np.array(data['recall_interp'])
        precision_interp = np.array(data['precision_interp'])
        ap = data['AP_R40']
        total_gt = data['total_gt']
        color = PLOT_PALETTE[c]

        # Raw points (subsampled for clarity)
        mask = recall_raw > 0.001
        if mask.sum() > 3:
            step = max(1, mask.sum() // 80)
            ax.scatter(recall_raw[mask][::step], precision_raw[mask][::step],
                       color=color, alpha=0.25, s=10, edgecolors='none', zorder=2)

        # Interpolated curve — step plot (standard for PR curves)
        ax.step(recall_interp, precision_interp, color=color, linewidth=2.0,
                where='post', zorder=3)
        ax.fill_between(recall_interp, precision_interp, alpha=0.10,
                        color=color, step='post')

        # AP badge
        ax.text(0.95, 0.08, f'AP$_{{R40}}$={ap:.1f}%',
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                color=color, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=color, alpha=0.9, linewidth=0.8))

        ax.set_xlabel('Recall', fontsize=9)
        if c == 0:
            ax.set_ylabel('Precision', fontsize=9)
        ax.set_title(f'{cls_en}  (GT={total_gt})', fontsize=11,
                     fontweight='bold', color=color)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'pr_curves_3d_per_class.png', dpi=200,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'pr_curves_3d_per_class.png'}")

    # ── Figure 2: BEV PR ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle('CenterPoint++ Epoch 30 — BEV Precision-Recall (IoU=0.25)',
                 fontsize=14, fontweight='bold', y=1.02)

    for c, (ax, cls_cn, cls_en) in enumerate(zip(axes, PLOT_CLASSES, PLOT_CLASSES_EN)):
        data = results['bev'][cls_cn]
        recall_raw = np.array(data['recall_raw'])
        precision_raw = np.array(data['precision_raw'])
        recall_interp = np.array(data['recall_interp'])
        precision_interp = np.array(data['precision_interp'])
        ap = data['AP_R40']
        total_gt = data['total_gt']
        color = PLOT_PALETTE[c]

        mask = recall_raw > 0.001
        if mask.sum() > 3:
            step = max(1, mask.sum() // 80)
            ax.scatter(recall_raw[mask][::step], precision_raw[mask][::step],
                       color=color, alpha=0.25, s=10, edgecolors='none', zorder=2)

        ax.step(recall_interp, precision_interp, color=color, linewidth=2.0,
                where='post', zorder=3)
        ax.fill_between(recall_interp, precision_interp, alpha=0.10,
                        color=color, step='post')

        ax.text(0.95, 0.08, f'AP$_{{R40}}$={ap:.1f}%',
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                color=color, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=color, alpha=0.9, linewidth=0.8))

        ax.set_xlabel('Recall', fontsize=9)
        if c == 0:
            ax.set_ylabel('Precision', fontsize=9)
        ax.set_title(f'{cls_en}  (GT={total_gt})', fontsize=11,
                     fontweight='bold', color=color)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'pr_curves_bev_per_class.png', dpi=200,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'pr_curves_bev_per_class.png'}")

    # ── Figure 3: 3D vs BEV comparison ─────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle('CenterPoint++ Epoch 30 — 3D vs BEV PR Curves',
                 fontsize=14, fontweight='bold', y=1.02)

    for c, (ax, cls_cn, cls_en) in enumerate(zip(axes, PLOT_CLASSES, PLOT_CLASSES_EN)):
        for mn, mc in [('3d', METRIC_COLORS['3d']), ('bev', METRIC_COLORS['bev'])]:
            data = results[mn][cls_cn]
            ri = np.array(data['recall_interp'])
            pi = np.array(data['precision_interp'])
            ap = data['AP_R40']
            ls = '-' if mn == '3d' else '--'
            lw = 2.0 if mn == '3d' else 1.5
            ax.step(ri, pi, color=mc, linewidth=lw, linestyle=ls,
                    where='post', label=f'{mn} (IoU={IOU_THRESH[mn]})')
            ax.fill_between(ri, pi, alpha=0.06, color=mc, step='post')

        ax.text(0.95, 0.15,
                f'3D AP={results["3d"][cls_cn]["AP_R40"]:.1f}%\nBEV AP={results["bev"][cls_cn]["AP_R40"]:.1f}%',
                transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
                family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#cccccc', alpha=0.9, linewidth=0.5))

        ax.set_xlabel('Recall', fontsize=9)
        if c == 0:
            ax.set_ylabel('Precision', fontsize=9)
        ax.set_title(f'{cls_en}', fontsize=11, fontweight='bold')
        ax.set_xlim(0, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc='lower left', framealpha=0.9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'pr_curves_3d_vs_bev.png', dpi=200,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'pr_curves_3d_vs_bev.png'}")

    # ── Figure 4: All classes combined ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('CenterPoint++ Epoch 30 — 3D PR Curves (IoU=0.5)',
                 fontsize=13, fontweight='bold')

    legend_elements = []
    for c, (cls_cn, cls_en) in enumerate(zip(PLOT_CLASSES, PLOT_CLASSES_EN)):
        data = results['3d'][cls_cn]
        ri = np.array(data['recall_interp'])
        pi = np.array(data['precision_interp'])
        ap = data['AP_R40']
        color = PLOT_PALETTE[c]

        ax.step(ri, pi, color=color, linewidth=2.0, where='post')
        ax.fill_between(ri, pi, alpha=0.08, color=color, step='post')
        legend_elements.append(
            Patch(facecolor=color, alpha=0.7,
                  label=f'{cls_en}  AP={ap:.1f}%'))

    mAP_val = results['3d']['mAP']
    legend_elements.append(
        Patch(facecolor='#333333', alpha=0.6,
              label=f'mAP = {mAP_val:.1f}%'))

    ax.legend(handles=legend_elements, fontsize=9, loc='lower left',
              framealpha=0.95, ncol=2)
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'pr_curves_3d_all_classes.png', dpi=200,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'pr_curves_3d_all_classes.png'}")

    # ── Figure 5: Raw PR points (scatter, no interpolation) ─────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle('CenterPoint++ Epoch 30 — 3D Raw PR Points (each dot = 1 detection threshold)',
                 fontsize=14, fontweight='bold', y=1.02)

    for c, (ax, cls_cn, cls_en) in enumerate(zip(axes, PLOT_CLASSES, PLOT_CLASSES_EN)):
        data = results['3d'][cls_cn]
        recall_raw = np.array(data['recall_raw'])
        precision_raw = np.array(data['precision_raw'])
        recall_interp = np.array(data['recall_interp'])
        precision_interp = np.array(data['precision_interp'])
        ap = data['AP_R40']
        total_gt = data['total_gt']
        color = PLOT_PALETTE[c]

        mask = recall_raw > 0.001
        if mask.any():
            ax.scatter(recall_raw[mask], precision_raw[mask],
                       c=[color], alpha=0.4, s=14, edgecolors='none', zorder=2)
        ax.step(recall_interp, precision_interp, color='#333333', linewidth=2.0,
                where='post', zorder=3)

        ax.text(0.95, 0.08, f'AP={ap:.1f}%',
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                color=color, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=color, alpha=0.9, linewidth=0.8))

        ax.set_xlabel('Recall', fontsize=9)
        if c == 0:
            ax.set_ylabel('Precision', fontsize=9)
        ax.set_title(f'{cls_en}  (GT={total_gt})', fontsize=11,
                     fontweight='bold', color=color)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'pr_raw_points_3d.png', dpi=200,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'pr_raw_points_3d.png'}")


def print_pr_analysis(results):
    """Print precision-recall tradeoff analysis."""
    print("\n" + "=" * 80)
    print("PRECISION-RECALL TRADEOFF ANALYSIS — 3D Detection (IoU=0.5)")
    print("=" * 80)

    for cls_cn, cls_en in zip(PLOT_CLASSES, PLOT_CLASSES_EN):
        data = results['3d'][cls_cn]
        rec = np.array(data['recall_raw'])
        prec = np.array(data['precision_raw'])
        ap = data['AP_R40']
        gt = data['total_gt']

        if gt == 0:
            print(f"\n  {cls_en}: No GT, skipping")
            continue

        mask = rec > 0.001
        if not mask.any():
            print(f"\n  {cls_en}: No detections, AP=0%")
            continue

        rec_v = rec[mask]
        prec_v = prec[mask]

        max_prec = np.max(prec_v)
        max_prec_idx = np.argmax(prec_v)
        max_prec_rec = rec_v[max_prec_idx]

        max_rec = np.max(rec_v)
        max_rec_idx = np.argmax(rec_v)
        max_rec_prec = prec_v[max_rec_idx]

        best_f1 = best_f1_p = best_f1_r = 0
        valid = (rec_v > 0.001) & (prec_v > 0.001)
        if valid.any():
            f1 = 2 * prec_v[valid] * rec_v[valid] / (prec_v[valid] + rec_v[valid])
            best_idx = np.argmax(f1)
            best_f1 = f1[best_idx]
            best_f1_p = prec_v[valid][best_idx]
            best_f1_r = rec_v[valid][best_idx]

        # P @ R=0.5
        mid_mask = rec_v >= 0.5
        p_at_r50 = np.max(prec_v[mid_mask]) if mid_mask.any() else 0

        print(f"\n  {cls_en} ({cls_cn}): GT={gt}, AP_R40={ap:.1f}%")
        print(f"    {'Max Precision:':<22s} {max_prec*100:.1f}%  @ R={max_prec_rec*100:.1f}%")
        print(f"    {'Max Recall:':<22s} {max_rec*100:.1f}%  @ P={max_rec_prec*100:.1f}%")
        print(f"    {'Best F1:':<22s} P={best_f1_p*100:.1f}%  R={best_f1_r*100:.1f}%  F1={best_f1:.3f}")
        print(f"    {'P @ R=0.5:':<22s} {p_at_r50*100:.1f}%")
        print(f"    {'P-R tradeoff:':<22s} R {max_prec_rec*100:.0f}→{max_rec*100:.0f}%  |  P {max_prec*100:.0f}→{max_rec_prec*100:.0f}%")

    print(f"\n  3D mAP (R40): {results['3d']['mAP']:.1f}%")
    print(f"  BEV mAP (R40): {results['bev']['mAP']:.1f}%")


def main():
    print("=" * 60)
    print("CenterPoint++ Precision-Recall Curve Generator (v2 — proper PR)")
    print("=" * 60)

    det_annos, gt_infos = load_data()

    # Transform to KITTI format
    eval_det_annos = copy.deepcopy(det_annos)
    eval_gt_annos = [copy.deepcopy(info['annos']) for info in gt_infos]
    print("Transforming to KITTI format...")
    kitti_utils.transform_annotations_to_kitti_format(
        eval_det_annos, map_name_to_kitti=MAP_NAME_TO_KITTI)
    kitti_utils.transform_annotations_to_kitti_format(
        eval_gt_annos, map_name_to_kitti=MAP_NAME_TO_KITTI, info_with_fakelidar=False)

    results = compute_all_pr_curves(eval_gt_annos, eval_det_annos)

    # Save JSON (trim raw arrays for readability)
    json_path = OUTPUT_DIR / 'pr_data.json'
    json_results = {}
    for metric in ['3d', 'bev']:
        json_results[metric] = {}
        for cls in CLASS_NAMES:
            d = results[metric][cls]
            json_results[metric][cls] = {
                'recall_interp': d['recall_interp'],
                'precision_interp': d['precision_interp'],
                'total_gt': d['total_gt'],
                'AP_11': d['AP_11'],
                'AP_R40': d['AP_R40'],
            }
        json_results[metric]['mAP'] = results[metric]['mAP']
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    print(f"\nPR data saved to: {json_path}")

    plot_pr_curves(results)
    print_pr_analysis(results)

    print(f"\nDone. Outputs in: {OUTPUT_DIR.resolve()}")


if __name__ == '__main__':
    main()
