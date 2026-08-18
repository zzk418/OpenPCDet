#!/usr/bin/env python3
"""
Point-SAM 交互式标注工具 — 语义分割 + 关键点标注
==================================================
用于货架 PCD 点云的快速标注。

工作流:
  1. 加载 PCD → 3D 点云 (XYZ + RGB)
  2. Point-SAM 网格自动提示 → 候选分割 masks
  3. 保存 proposals (后续人工审核加类别标签)
  4. 关键点标注接口 (预留)

用法:
  # 单文件推理
  python pointsam_annotator.py \
      --pcd_path data/new_sheef/pngs/TV_250000001828.pcd \
      --ckpt_path /code/Point-SAM/pretrained/model.safetensors \
      --output_dir output/shelf_annotations

  # 批量处理整个目录
  python pointsam_annotator.py \
      --data_dir data/new_sheef \
      --ckpt_path /code/Point-SAM/pretrained/model.safetensors \
      --output_dir output/shelf_annotations

依赖:
  conda activate pc
  export PYTHONPATH=/code/OpenPCDet:/code/Point-SAM:$PYTHONPATH
"""

import argparse
import json
import os
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# ── Inject torkit3d & apex stubs BEFORE importing point-sam ──
sys.path.insert(0, "/code/Point-SAM")
import pc_sam.torkit3d_stub as _stub
_stub.inject()
_stub.inject_apex()

import hydra
from omegaconf import OmegaConf
from safetensors.torch import load_model

from pc_sam.model.pc_sam import PointCloudSAM
from pc_sam.utils.torch_utils import replace_with_fused_layernorm


# ============================================================================
# PCD Reader
# ============================================================================

def read_pcd_binary(filepath: str) -> dict:
    """读取 PCD v0.7 二进制文件 (FIELDS: x y z rgb).

    Returns:
        dict with keys:
            xyz:   (N, 3) float32  — 点云坐标
            rgb:   (N, 3) float32  — RGB 颜色 [0, 1]
            num_points: int
    """
    with open(filepath, "rb") as f:
        # 读取头部
        header = {}
        while True:
            line = f.readline().decode().strip()
            if not line:
                continue
            if "DATA" in line:
                header["data_type"] = line.split()[1]
                break
            parts = line.split()
            if len(parts) >= 2:
                header[parts[0]] = parts[1:]

        fields = header.get("FIELDS", [])
        sizes = [int(s) for s in header.get("SIZE", [])]
        types = header.get("TYPE", [])
        counts = [int(c) for c in header.get("COUNT", [])]
        width = int(header["WIDTH"][0])
        height = int(header["HEIGHT"][0])
        num_points = width * height

        # 计算每条记录的字节数
        bytes_per_point = sum(s * c for s, c in zip(sizes, counts))
        expected_bytes = num_points * bytes_per_point

        # 读取二进制数据
        rest = f.read()

        if header["data_type"] == "binary":
            data = rest[:expected_bytes]
        else:
            raise ValueError(f"Unsupported data type: {header['data_type']}")

    # 构建 numpy dtype 来解析交错存储的二进制数据
    # PCD binary 格式: 每个点的所有字段连续存储 (point by point)
    pcd_dtype_map = {"F": "f4", "U": "u4", "I": "i4"}
    dtype_spec = [(f, pcd_dtype_map[t]) for f, t in zip(fields, types)]
    point_dtype = np.dtype(dtype_spec)

    # 读取所有点
    points = np.frombuffer(data, dtype=point_dtype, count=num_points)

    # 提取 XYZ
    xyz = np.zeros((num_points, 3), dtype=np.float32)
    if "x" in points.dtype.names:
        xyz[:, 0] = points["x"].astype(np.float32)
    if "y" in points.dtype.names:
        xyz[:, 1] = points["y"].astype(np.float32)
    if "z" in points.dtype.names:
        xyz[:, 2] = points["z"].astype(np.float32)

    # 解包 RGB (uint32 packed → float [0,1])
    if "rgb" in points.dtype.names:
        rgb_packed = points["rgb"]
        r = ((rgb_packed >> 16) & 0xFF).astype(np.float32) / 255.0
        g = ((rgb_packed >> 8) & 0xFF).astype(np.float32) / 255.0
        b = (rgb_packed & 0xFF).astype(np.float32) / 255.0
        rgb = np.stack([r, g, b], axis=1)
    else:
        rgb = np.ones((num_points, 3), dtype=np.float32) * 0.5  # gray

    return {"xyz": xyz, "rgb": rgb, "num_points": num_points}


# ============================================================================
# Point-SAM Wrapper
# ============================================================================

class PointSAMAnnotator:
    """Point-SAM 标注器：加载模型 + 生成候选分割 masks."""

    def __init__(
        self,
        ckpt_path: str,
        config_dir: str = "/code/Point-SAM/configs",
        config_name: str = "large",
        device: str = "cuda",
        group_number: int = 512,
        group_size: int = 64,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.group_number = group_number
        self.group_size = group_size

        print(f"[PointSAM] Loading config: {config_name}")
        # hydra.initialize 需要相对于 CWD 的路径
        import hydra.core.global_hydra
        try:
            hydra.core.global_hydra.GlobalHydra.instance().clear()
        except Exception:
            pass
        # 使用绝对路径的 initialize_config_dir
        try:
            # hydra >= 1.3
            hydra.initialize_config_dir(config_dir, version_base=None)
            cfg = hydra.compose(config_name=config_name)
        except AttributeError:
            # hydra < 1.3, fallback to relative path
            cwd = os.getcwd()
            if os.path.isabs(config_dir):
                os.chdir(os.path.dirname(config_dir))
                rel_dir = os.path.basename(config_dir)
            else:
                rel_dir = config_dir
            with hydra.initialize(rel_dir, version_base=None):
                cfg = hydra.compose(config_name=config_name)
            os.chdir(cwd)
        OmegaConf.resolve(cfg)

        print(f"[PointSAM] Building model...")
        self.model: PointCloudSAM = hydra.utils.instantiate(cfg.model)
        self.model.apply(replace_with_fused_layernorm)

        print(f"[PointSAM] Loading weights: {ckpt_path}")
        load_model(self.model, ckpt_path)

        self.model.eval()
        self.model.to(self.device)
        print(f"[PointSAM] Ready. Device: {self.device}")

        # 缓存当前点云的编码结果
        self._pc_embeddings = None
        self._patches = None
        self._coords = None
        self._features = None

    @torch.no_grad()
    def set_pointcloud(self, xyz: np.ndarray, rgb: np.ndarray):
        """设置当前要标注的点云，并编码。

        Args:
            xyz: (N, 3) float32 — 原始坐标 (单位 mm 或 m，会自动归一化)
            rgb: (N, 3) float32 — RGB [0, 1]
        """
        N = xyz.shape[0]
        min_points = 2048  # Point-SAM 最少需要这么多点 (num_groups=1024 + safe margin)

        if N < min_points:
            # 点数太少，通过随机重复来padding
            repeat_times = (min_points + N - 1) // N
            xyz_pad = np.tile(xyz, (repeat_times, 1))[:min_points]
            rgb_pad = np.tile(rgb, (repeat_times, 1))[:min_points]
            print(f"  [Padding] {N} → {min_points} points")
        else:
            xyz_pad = xyz
            rgb_pad = rgb

        coords = torch.from_numpy(xyz_pad).float().to(self.device).unsqueeze(0)
        colors = torch.from_numpy(rgb_pad).float().to(self.device).unsqueeze(0)

        # 归一化坐标 (Point-SAM 期望 [-1, 1] 范围)
        center = coords.mean(dim=1, keepdim=True)
        coords = coords - center
        scale = coords.norm(dim=2, keepdim=True).max()
        if scale > 0:
            coords = coords / scale

        self._n_original = N  # 记录原始点数
        self._coords = coords
        self._features = colors
        self._scale = scale.item()
        self._center = center.squeeze(0).cpu().numpy()

        # 编码点云
        self._pc_embeddings, self._patches = self.model.pc_encoder(coords, colors)
        print(f"  [Encode] {coords.shape[1]} points → {self._pc_embeddings.shape[1]} patches")

    @torch.no_grad()
    def segment_with_prompts(
        self,
        prompt_coords: torch.Tensor,
        prompt_labels: torch.Tensor,
        multimask_output: bool = True,
    ) -> tuple:
        """用提示点做分割。

        Args:
            prompt_coords: (1, M, 3) — 归一化后的提示点坐标
            prompt_labels: (1, M) — 1=前景, 0=背景
            multimask_output: 是否输出多个候选 mask

        Returns:
            masks:      (1, num_outputs, N) — 二值 mask
            iou_preds:  (1, num_outputs) — 预测 IoU
        """
        masks, iou_preds = self.model.predict_masks(
            coords=self._coords,
            features=self._features,
            prompt_coords=prompt_coords,
            prompt_labels=prompt_labels,
            prompt_masks=None,
            multimask_output=multimask_output,
        )
        return masks, iou_preds

    @torch.no_grad()
    def auto_proposals(
        self,
        grid_resolution: int = 5,
        iou_threshold: float = 0.5,
        min_points: int = 100,
        max_proposals: int = 32,
    ) -> list[dict]:
        """自动生成候选分割 proposals (网格提示点方式)。

        在点云包围盒内均匀采样 grid_resolution^3 个提示点，
        每个点用 Point-SAM 生成 3 个候选 mask，然后按 IoU 去重。

        Args:
            grid_resolution: 空间网格分辨率
            iou_threshold: NMS 去重的 IoU 阈值
            min_points:  最小点数阈值，过滤太小的 mask
            max_proposals: 最多保留的 proposals 数

        Returns:
            list[dict]: 每个 proposal 包含:
                - mask: (N,) bool — 二值 mask
                - iou_score: float — 预测 IoU
                - prompt_xyz: (3,) — 提示点在原始坐标系的坐标
        """
        if self._coords is None:
            raise RuntimeError("请先调用 set_pointcloud()")

        coords_np = self._coords[0].cpu().numpy()  # (N_pad, 3) normalized
        N_full = coords_np.shape[0]
        N = getattr(self, '_n_original', N_full)  # 原始点数，用于 slice masks

        # 在归一化空间中生成网格提示点
        bbox_min = coords_np.min(axis=0)
        bbox_max = coords_np.max(axis=0)

        grid_axes = [
            np.linspace(bbox_min[i] + 0.1 * (bbox_max[i] - bbox_min[i]),
                        bbox_max[i] - 0.1 * (bbox_max[i] - bbox_min[i]),
                        grid_resolution)
            for i in range(3)
        ]
        grid_points = np.stack(np.meshgrid(*grid_axes, indexing="ij"), axis=-1).reshape(-1, 3)

        proposals = []
        seen_masks = []

        for gp in grid_points:
            prompt_coords = torch.from_numpy(gp).float().to(self.device).view(1, 1, 3)
            prompt_labels = torch.ones(1, 1, dtype=torch.long, device=self.device)

            masks, iou_preds = self.segment_with_prompts(
                prompt_coords, prompt_labels, multimask_output=True
            )
            # masks: (1, 3, N), iou_preds: (1, 3)

            for k in range(masks.shape[1]):
                # masks 是 logit/probability 值，需要二值化
                mask_raw = masks[0, k].cpu().numpy()  # (N_full,) float
                mask = (mask_raw > 0.0)[:N]  # 二值化 + 切回原始点数
                iou = iou_preds[0, k].item()

                n_points = int(mask.sum())
                if n_points < min_points:
                    continue

                # IoU 去重
                duplicate = False
                for prev_mask in seen_masks:
                    intersection = int((mask & prev_mask).sum())
                    union = int((mask | prev_mask).sum())
                    if union > 0 and intersection / union > iou_threshold:
                        duplicate = True
                        break

                if duplicate:
                    continue

                seen_masks.append(mask)

                # 把提示点坐标转回原始坐标系
                prompt_original = gp * self._scale + self._center

                proposals.append({
                    "mask": mask.astype(bool),
                    "iou_score": iou,
                    "prompt_xyz": prompt_original,
                    "n_points": int(n_points),
                })

                if len(proposals) >= max_proposals:
                    break

            if len(proposals) >= max_proposals:
                break

        # 按点数降序排列
        proposals.sort(key=lambda p: p["n_points"], reverse=True)
        print(f"  [AutoProposals] {len(grid_points)} prompts → {len(proposals)} unique masks")
        return proposals


# ============================================================================
# 标注保存
# ============================================================================

def save_annotations(
    output_dir: str,
    pcd_stem: str,
    xyz: np.ndarray,
    rgb: np.ndarray,
    proposals: list[dict],
    keypoints: dict = None,
):
    """保存标注结果。

    目录结构:
        output_dir/
        ├── {stem}_proposals.npz     # 候选 masks（待审核分类）
        ├── {stem}_info.json         # 元信息
        └── {stem}_keypoints.json    # 关键点标注（如有）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存 proposals
    masks = np.stack([p["mask"] for p in proposals], axis=0)  # (M, N)
    iou_scores = np.array([p["iou_score"] for p in proposals])
    n_points = np.array([p["n_points"] for p in proposals])
    prompt_xyzs = np.stack([p["prompt_xyz"] for p in proposals], axis=0)

    np.savez_compressed(
        os.path.join(output_dir, f"{pcd_stem}_proposals.npz"),
        masks=masks,
        iou_scores=iou_scores,
        n_points=n_points,
        prompt_xyzs=prompt_xyzs,
        # 保留原始点云数据以便后处理
        xyz=xyz,
        rgb=rgb,
    )

    # 保存元信息
    info = {
        "pcd_stem": pcd_stem,
        "num_points": len(xyz),
        "num_proposals": len(proposals),
        "proposals": [
            {
                "id": i,
                "n_points": p["n_points"],
                "iou_score": float(p["iou_score"]),
                "prompt_xyz": p["prompt_xyz"].tolist(),
                "class_label": None,  # 待审核: 'beam'/'pillar'/'pallet'/'goods'
            }
            for i, p in enumerate(proposals)
        ],
        "xyz_stats": {
            "min": xyz.min(axis=0).tolist(),
            "max": xyz.max(axis=0).tolist(),
            "mean": xyz.mean(axis=0).tolist(),
        },
    }
    with open(os.path.join(output_dir, f"{pcd_stem}_info.json"), "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # 关键点（如有）
    if keypoints:
        with open(os.path.join(output_dir, f"{pcd_stem}_keypoints.json"), "w") as f:
            json.dump(keypoints, f, indent=2, ensure_ascii=False)

    print(f"  [Save] {len(proposals)} proposals → {output_dir}/{pcd_stem}_*")


# ============================================================================
# 辅助: 可视化 BEV 投影
# ============================================================================

def render_bev_mask(xyz: np.ndarray, mask: np.ndarray, resolution: float = 0.01):
    """将 3D mask 投影到 BEV (XZ 平面)，生成 2D 鸟瞰图用于快速审核。

    Args:
        xyz: (N, 3) 原始坐标
        mask: (N,) bool
        resolution: BEV 分辨率 (m/pixel)

    Returns:
        bev: (H, W) uint8 — 0=背景, 1=mask 区域
        extent: [x_min, x_max, z_min, z_max]
    """
    masked_pts = xyz[mask]
    if len(masked_pts) == 0:
        return np.zeros((1, 1), dtype=np.uint8), [0, 1, 0, 1]

    x, z = masked_pts[:, 0], masked_pts[:, 2]
    x_min, x_max = x.min() - 0.1, x.max() + 0.1
    z_min, z_max = z.min() - 0.1, z.max() + 0.1

    W = max(1, int((x_max - x_min) / resolution))
    H = max(1, int((z_max - z_min) / resolution))

    bev = np.zeros((H, W), dtype=np.uint8)
    xi = np.clip(((x - x_min) / resolution).astype(int), 0, W - 1)
    zi = np.clip(((z - z_min) / resolution).astype(int), 0, H - 1)
    bev[zi, xi] = 1

    return bev, [x_min, x_max, z_min, z_max]


# ============================================================================
# 批量处理
# ============================================================================

def find_all_pcds(data_dir: str) -> list[str]:
    """递归查找所有 .pcd 文件。"""
    pcds = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".pcd"):
                pcds.append(os.path.join(root, f))
    pcds.sort()
    return pcds


def process_single(
    annotator: PointSAMAnnotator,
    pcd_path: str,
    output_dir: str,
    args,
) -> dict:
    """处理单个 PCD 文件。"""
    stem = Path(pcd_path).stem
    print(f"\n{'='*60}")
    print(f"[{stem}] Loading: {pcd_path}")

    try:
        pcd_data = read_pcd_binary(pcd_path)
    except Exception as e:
        print(f"  [ERROR] Failed to read PCD: {e}")
        return {"success": False, "error": str(e)}

    xyz = pcd_data["xyz"]
    rgb = pcd_data["rgb"]
    print(f"  Points: {len(xyz)}")

    # 降采样（可选，加速处理）
    if args.max_points and len(xyz) > args.max_points:
        idx = np.random.choice(len(xyz), args.max_points, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]
        print(f"  Downsampled to: {args.max_points}")

    # 编码点云
    annotator.set_pointcloud(xyz, rgb)

    # 自动生成 proposals
    proposals = annotator.auto_proposals(
        grid_resolution=args.grid_resolution,
        iou_threshold=args.iou_threshold,
        min_points=args.min_points,
        max_proposals=args.max_proposals,
    )

    # 保存
    save_annotations(output_dir, stem, xyz, rgb, proposals)

    return {
        "success": True,
        "stem": stem,
        "num_points": len(xyz),
        "num_proposals": len(proposals),
    }


def batch_process(annotator: PointSAMAnnotator, pcd_paths: list[str], output_dir: str, args):
    """批量处理所有 PCD。"""
    total = len(pcd_paths)
    results = []
    t_start = time.time()

    for i, pcd_path in enumerate(pcd_paths):
        print(f"\n[{i+1}/{total}]")
        result = process_single(annotator, pcd_path, output_dir, args)
        results.append(result)

        elapsed = time.time() - t_start
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (total - i - 1)
        print(f"  Progress: {i+1}/{total} | ETA: {remaining/60:.1f} min")

    # 汇总
    ok = sum(1 for r in results if r["success"])
    fail = total - ok
    print(f"\n{'='*60}")
    print(f"Done! {ok} success, {fail} failed. Total time: {(time.time()-t_start)/60:.1f} min")

    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Point-SAM 3D 点云标注工具")
    parser.add_argument("--pcd_path", type=str, default=None,
                        help="单个 PCD 文件路径")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="批量处理: PCD 所在根目录 (递归查找 *.pcd)")
    parser.add_argument("--ckpt_path", type=str,
                        default="/code/Point-SAM/pretrained/model.safetensors",
                        help="Point-SAM 预训练权重路径")
    parser.add_argument("--config_dir", type=str,
                        default="/code/Point-SAM/configs")
    parser.add_argument("--config_name", type=str, default="large",
                        help="模型配置名 (large / giant / base)")
    parser.add_argument("--output_dir", type=str,
                        default="output/shelf_annotations",
                        help="标注输出目录")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_points", type=int, default=50000,
                        help="最大点数 (超采样，0=不限制)")
    parser.add_argument("--grid_resolution", type=int, default=5,
                        help="网格提示点分辨率")
    parser.add_argument("--iou_threshold", type=float, default=0.6,
                        help="Proposal 去重 IoU 阈值")
    parser.add_argument("--min_points", type=int, default=100,
                        help="最小 mask 点数")
    parser.add_argument("--max_proposals", type=int, default=32,
                        help="每帧最多保留的 proposals")
    parser.add_argument("--interactive", action="store_true",
                        help="交互模式 (预留)")

    args = parser.parse_args()

    if not args.pcd_path and not args.data_dir:
        parser.error("必须指定 --pcd_path 或 --data_dir")

    # 初始化 Point-SAM
    annotator = PointSAMAnnotator(
        ckpt_path=args.ckpt_path,
        config_dir=args.config_dir,
        config_name=args.config_name,
        device=args.device,
    )

    # 处理
    if args.pcd_path:
        process_single(annotator, args.pcd_path, args.output_dir, args)
    elif args.data_dir:
        pcd_paths = find_all_pcds(args.data_dir)
        print(f"Found {len(pcd_paths)} PCD files in {args.data_dir}")
        batch_process(annotator, pcd_paths, args.output_dir, args)


if __name__ == "__main__":
    main()
