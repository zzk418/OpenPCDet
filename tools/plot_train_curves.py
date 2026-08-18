"""Plot per-epoch train/val curves from a shelf-pose YOLO run's results.csv.

Usage:
    python tools/plot_train_curves.py [run_dir]          # defaults to newest run
    python tools/plot_train_curves.py output/shelf_pose_train/shelf_reviewed_v5

Outputs `train_curves.png` into the run dir and prints the best-epoch summary.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TRAIN_ROOT = REPO / "output/shelf_pose_train"

# light-surface tokens (dataviz default palette)
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
AXIS = "#c3c2b7"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4

# loss types -> hue; solid = train, dashed = val (same hue keeps the pair linked)
LOSSES = [("box_loss", C[0]), ("pose_loss", C[1]), ("dfl_loss", C[2])]

METRIC_SETS = {
    "Pose keypoints (P)": ["precision(P)", "recall(P)", "mAP50(P)", "mAP50-95(P)"],
    "Box (B)": ["precision(B)", "recall(B)", "mAP50(B)", "mAP50-95(B)"],
}


def style_ax(ax, ylim=None):
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlabel("Epoch", color=INK, fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)


def best_epoch(series):
    """Return (epoch, value) of the max of a metric series (0-indexed epoch + 1)."""
    s = series.dropna()
    return int(s.idxmax()) + 1, float(s.max())


def plot_run(run_dir: Path) -> Path:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        sys.exit(f"No results.csv in {run_dir}")
    df = pd.read_csv(csv_path)
    epochs = df["epoch"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 14), sharex=True)
    fig.patch.set_facecolor(SURFACE)

    # 1) losses: solid = train, dashed = val
    ax = axes[0]
    for loss_name, color in LOSSES:
        train_col, val_col = f"train/{loss_name}", f"val/{loss_name}"
        if train_col in df:
            ax.plot(epochs, df[train_col], color=color, lw=2, label=f"{loss_name} train")
        if val_col in df:
            ax.plot(epochs, df[val_col], color=color, lw=2, ls="--", label=f"{loss_name} val")
    style_ax(ax)
    ax.set_title("Losses", color=INK, fontsize=12, loc="left", pad=10)
    ax.legend(ncol=3, frameon=False, fontsize=8, labelcolor=INK, loc="upper right")

    # 2) val metrics per task
    for ax, (title, metrics) in zip(axes[1:], METRIC_SETS.items()):
        for metric, color in zip(metrics, C):
            col = f"metrics/{metric}"
            if col not in df:
                continue
            ax.plot(epochs, df[col], color=color, lw=2, label=metric.split("(")[0])
        # direct label at each task's best mAP50-95
        col = f"metrics/mAP50-95({title[-2]})"
        if col in df:
            ep, val = best_epoch(df[col])
            ax.plot([ep], [val], marker="*", ms=14, color=INK,
                    markeredgecolor=SURFACE, markeredgewidth=1, ls="none")
            ax.annotate(f"ep {ep}: {val:.3f}", (ep, val), textcoords="offset points",
                        xytext=(8, -12), color=INK, fontsize=9)
        style_ax(ax, ylim=(0.0, 1.0))
        ax.set_title(f"Val metrics — {title}", color=INK, fontsize=12, loc="left", pad=10)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")

    fig.tight_layout()
    out = run_dir / "train_curves.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    print(f"[{run_dir.name}]")
    for task, metrics in METRIC_SETS.items():
        m = metrics[3]  # mAP50-95
        col = f"metrics/{m}"
        if col in df:
            ep, val = best_epoch(df[col])
            print(f"  {task}: best {m} = {val:.4f} @ epoch {ep}")
    print(f"  chart -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", help="run dir containing results.csv "
                                               "(default: newest under output/shelf_pose_train)")
    args = ap.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        runs = sorted([p for p in TRAIN_ROOT.iterdir()
                       if p.is_dir() and (p / "results.csv").exists()],
                      key=lambda p: p.stat().st_mtime)
        if not runs:
            sys.exit(f"No run with results.csv under {TRAIN_ROOT}")
        run_dir = runs[-1]
    plot_run(run_dir)


if __name__ == "__main__":
    main()
