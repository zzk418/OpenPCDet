"""
AGV 全局路网自动生成工具
========================
算法: 检测框裁剪物体点云 → 合成平面仓库 → PRM + NetworkX MST

流程:
  1. 从 8 帧检测框中裁剪物体点云 (旋转bbox精确提取)
  2. 去重后按网格排列到平面地板上 (无墙体/货架干扰)
  3. BEV 占据栅格 + AGV 安全半径膨胀 (占据率 <2%)
  4. PRM: 自由空间采样 → KD-tree 近邻连边 → 道路骨架图
  5. 配对路径: PRM 最短路径 + A* 桥接
  6. NetworkX MST → 最小路网 (100% 连通)

用法:
  conda activate pc
  export PYTHONPATH=/code/OpenPCDet:$PYTHONPATH
  python tools/road_network_generator.py
"""

import numpy as np
import json
import os
import heapq
import time
from collections import deque
from typing import List, Dict, Tuple, Optional

import networkx as nx
from scipy.spatial import KDTree
from skimage import morphology

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ============================================================
# 配置
# ============================================================
GRID_RES = 0.25            # 栅格分辨率 (m)
ROBOT_RADIUS = 0.4         # AGV 安全半径 (m)
N_ROAD_SAMPLES = 600       # PRM 自由空间采样数
K_NEIGHBORS = 8            # KD-tree 近邻连接数
MAX_EDGE_LEN = 15.0        # PRM 最大连边距离 (m)
GRID_SPACING = 10.0        # 合成场景物体间距 (m, grid模式)
GRID_COLS = 5              # 合成场景网格列数 (grid模式)
AREA_SIZE = 60.0           # 随机布局区域边长 (m, random模式)
MIN_SEPARATION = 2.0       # 随机布局最小物体间距 (m)
LAYOUT_MODE = 'random'     # 'grid' = 规整网格, 'random' = 随机散落
BEV_PAD = 5.0              # BEV 自适应裁剪边距 (m)
DEDUP_RADIUS = 1.2         # 任务点去重半径 (m)
RANDOM_SEED = 42           # 随机种子 (可复现)
OUTPUT_DIR = 'output/road_network'

# 8 帧提取物体点云
FRAMES = ['002311', '002960', '003056', '003168', '003135', '002405', '002361', '002171']

# 类别配色
CLASS_COLORS = {
    'Box': '#E74C3C', 'ELF': '#2ECC71', 'CargoBike': '#3498DB',
    'FTS': '#F39C12', 'ForkLift': '#9B59B6',
}
CLASS_MARKERS = {
    'Box': 's', 'ELF': 'D', 'CargoBike': '^', 'FTS': 'o', 'ForkLift': 'P',
}


class RoadNetworkGenerator:
    """AGV 全局路网生成器 — 合成平面仓库 + PRM + MST"""

    def __init__(self, grid_res: float = GRID_RES, robot_radius: float = ROBOT_RADIUS):
        self.grid_res = grid_res
        self.robot_radius = robot_radius
        self.r_px = max(1, int(robot_radius / grid_res))

        # 内部状态
        self.occ_raw = None
        self.occ_dilated = None
        self.free = None
        self.x_min = self.x_max = self.y_min = self.y_max = 0.0
        self.gw = self.gh = 0

    # ---- 坐标转换 ----
    def to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        return int((wx - self.x_min) / self.grid_res), int((wy - self.y_min) / self.grid_res)

    def to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        return gx * self.grid_res + self.x_min, gy * self.grid_res + self.y_min

    def is_free(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.gw and 0 <= gy < self.gh and self.free[gy, gx] == 1

    # ---- 占据栅格 ----
    def build_grid(self, points: np.ndarray, obj_mask: np.ndarray):
        """构建 BEV 占据栅格: 物体点→障碍, 地板点→自由"""
        self.x_min = points[:, 0].min() - BEV_PAD
        self.x_max = points[:, 0].max() + BEV_PAD
        self.y_min = points[:, 1].min() - BEV_PAD
        self.y_max = points[:, 1].max() + BEV_PAD
        self.gw = int((self.x_max - self.x_min) / self.grid_res) + 1
        self.gh = int((self.y_max - self.y_min) / self.grid_res) + 1

        self.occ_raw = np.zeros((self.gh, self.gw), dtype=np.uint8)
        xs = ((points[obj_mask, 0] - self.x_min) / self.grid_res).astype(int)
        ys = ((points[obj_mask, 1] - self.y_min) / self.grid_res).astype(int)
        m = (xs >= 0) & (xs < self.gw) & (ys >= 0) & (ys < self.gh)
        self.occ_raw[ys[m], xs[m]] = 1

        kernel = np.ones((2 * self.r_px + 1, 2 * self.r_px + 1), dtype=np.uint8)
        self.occ_dilated = morphology.binary_dilation(self.occ_raw, kernel).astype(np.uint8)
        self.free = 1 - self.occ_dilated

        raw_pct = self.occ_raw.sum() / self.occ_raw.size * 100
        dil_pct = self.occ_dilated.sum() / self.occ_dilated.size * 100
        print(f"  Grid: {self.gw}×{self.gh}, Occ: {raw_pct:.1f}% → dil: {dil_pct:.1f}%")

    # ---- 线段碰撞检测 ----
    def line_free(self, p1, p2, step: float = 0.2) -> bool:
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        dist = np.hypot(dx, dy)
        n = max(int(dist / step), 1)
        for k in range(n + 1):
            t = k / n
            gx, gy = self.to_grid(p1[0] + t * dx, p1[1] + t * dy)
            if not self.is_free(gx, gy):
                return False
        return True

    # ---- Task Point 投影 ----
    def project_task_point(self, wx: float, wy: float) -> Tuple[float, float]:
        gx, gy = self.to_grid(wx, wy)
        if self.is_free(gx, gy):
            return wx, wy
        visited = {(gx, gy)}
        q = deque([(gx, gy)])
        while q:
            cx, cy = q.popleft()
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (-1, 1), (1, -1), (-1, -1)]:
                nx_g, ny_g = cx + dx, cy + dy
                if (nx_g, ny_g) in visited:
                    continue
                if not (0 <= nx_g < self.gw and 0 <= ny_g < self.gh):
                    continue
                visited.add((nx_g, ny_g))
                if self.is_free(nx_g, ny_g):
                    return self.to_world(nx_g, ny_g)
                q.append((nx_g, ny_g))
        return wx, wy

    # ---- A* 网格搜索 ----
    def astar(self, sx: float, sy: float, gx: float, gy: float) -> Optional[List[Tuple[float, float]]]:
        sgx, sgy = self.to_grid(sx, sy)
        ggx, ggy = self.to_grid(gx, gy)
        if not self.is_free(sgx, sgy) or not self.is_free(ggx, ggy):
            return None
        open_set = [(abs(sgx - ggx) + abs(sgy - ggy), 0, sgx, sgy)]
        came_from = {}
        g_score = {(sgx, sgy): 0}
        while open_set:
            _, cost, cx, cy = heapq.heappop(open_set)
            if (cx, cy) == (ggx, ggy):
                path = [(ggx, ggy)]
                while (cx, cy) in came_from:
                    cx, cy = came_from[(cx, cy)]
                    path.append((cx, cy))
                return [self.to_world(px, py) for px, py in reversed(path)]
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (-1, 1), (1, -1), (-1, -1)]:
                nx_g, ny_g = cx + dx, cy + dy
                if not self.is_free(nx_g, ny_g):
                    continue
                mc = 1.414 if dx and dy else 1.0
                ng = cost + mc
                if (nx_g, ny_g) not in g_score or ng < g_score[(nx_g, ny_g)]:
                    g_score[(nx_g, ny_g)] = ng
                    heapq.heappush(open_set, (ng + abs(nx_g - ggx) + abs(ny_g - ggy),
                                              ng, nx_g, ny_g))
                    came_from[(nx_g, ny_g)] = (cx, cy)
        return None

    # ---- 提取物体点云 ----
    @staticmethod
    def extract_objects(frame_ids: List[str], data: dict,
                        dedup_radius: float = DEDUP_RADIUS) -> Tuple[List[dict], List[dict]]:
        """从检测框裁剪物体点云, 返回 (objects, task_points)"""
        all_objs = []
        tp_raw = []

        for fid in frame_ids:
            pc = np.load(f'data/warehouse/points/{fid}.npy')
            for det in data.get(fid, {}).get('detections', []):
                cx, cy, cz = det['center_x'], det['center_y'], det['center_z']
                dx, dy, dz = det['dx'], det['dy'], det['dz']
                heading = det['heading']

                # 旋转 bbox 精确裁剪
                cos_h, sin_h = np.cos(-heading), np.sin(-heading)
                tx = pc[:, 0] - cx
                ty = pc[:, 1] - cy
                tz = pc[:, 2] - cz
                rx = cos_h * tx - sin_h * ty
                ry = sin_h * tx + cos_h * ty
                in_box = (np.abs(rx) <= dx / 2) & (np.abs(ry) <= dy / 2) & (np.abs(tz) <= dz / 2)

                if in_box.sum() >= 5:
                    all_objs.append({
                        'pts': pc[in_box, :3].copy(),  # xyz only
                        'cls': det['class'],
                        'score': det['score'],
                        'center': (cx, cy, cz),
                        'dims': (dx, dy, dz),
                        'heading': heading,
                        'frame': fid,
                    })
                    tp_raw.append({
                        'x': cx, 'y': cy,
                        'cls': det['class'],
                        'score': det['score'],
                        'frame': fid,
                    })

        print(f"  Extracted {len(all_objs)} objects from {len(frame_ids)} frames")

        # 去重: 同类别 + 距离 < dedup_radius → 保留最高分
        tp_unique, used = [], set()
        for i, t in enumerate(tp_raw):
            if i in used:
                continue
            cluster = [(i, t)]
            for j in range(i + 1, len(tp_raw)):
                if j not in used and tp_raw[j]['cls'] == t['cls'] and \
                   np.hypot(t['x'] - tp_raw[j]['x'], t['y'] - tp_raw[j]['y']) < dedup_radius:
                    cluster.append((j, tp_raw[j]))
                    used.add(j)
            best_idx = max(cluster, key=lambda x: x[1]['score'])[0]
            tp_unique.append({
                'x': tp_raw[best_idx]['x'], 'y': tp_raw[best_idx]['y'],
                'cls': tp_raw[best_idx]['cls'],
                'score': tp_raw[best_idx]['score'],
                'obj_idx': best_idx,
                'dims': all_objs[best_idx]['dims'],
                'heading': all_objs[best_idx]['heading'],
            })

        print(f"  After dedup: {len(tp_unique)} unique objects")
        return all_objs, tp_unique

    # ---- 合成场景布局 (grid) ----
    @staticmethod
    def layout_grid(objects: List[dict], task_points: List[dict],
                    spacing: float = GRID_SPACING,
                    n_cols: int = GRID_COLS) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """物体按规整网格排列"""
        syn_pts, syn_tp = [], []
        for i, tp in enumerate(task_points):
            obj = objects[tp['obj_idx']]
            pts = obj['pts'].copy()
            row, col = i // n_cols, i % n_cols
            nx_cx, ny_cy = col * spacing, -row * spacing
            ocx, ocy, ocz = obj['center']
            pts[:, 0] += (nx_cx - ocx)
            pts[:, 1] += (ny_cy - ocy)
            pts[:, 2] += (-2.0 - ocz)
            syn_pts.append(pts)
            syn_tp.append({'x': nx_cx, 'y': ny_cy, 'cls': tp['cls'],
                           'score': tp['score'], 'dims': tp['dims'], 'heading': tp['heading']})
        return RoadNetworkGenerator._add_floor(syn_pts, syn_tp, n_cols, spacing)

    # ---- 合成场景布局 (random) ----
    @staticmethod
    def layout_random(objects: List[dict], task_points: List[dict],
                      area_size: float = AREA_SIZE,
                      min_sep: float = MIN_SEPARATION,
                      seed: int = RANDOM_SEED) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """物体随机散落, 避免重叠, 模拟真实仓库"""
        rng = np.random.RandomState(seed)
        n = len(task_points)
        radii = [np.hypot(tp['dims'][0], tp['dims'][1]) / 2 + min_sep for tp in task_points]
        positions = []

        for i in range(n):
            r = radii[i]
            best = None
            for _ in range(200):
                cx = rng.uniform(r, area_size - r)
                cy = rng.uniform(-area_size + r, -r)
                ok = True
                for px, py, pr in positions:
                    if np.hypot(cx - px, cy - py) < pr + r:
                        ok = False
                        break
                if ok:
                    # 偏好不太孤立的位姿
                    if not positions or min(np.hypot(cx - px, cy - py) for px, py, _ in positions) < 15.0:
                        positions.append((cx, cy, r))
                        best = (cx, cy)
                        break
                    elif best is None:
                        best = (cx, cy)
            if best is None:
                cx = rng.uniform(r, area_size - r)
                cy = rng.uniform(-area_size + r, -r)
                positions.append((cx, cy, r))
                best = (cx, cy)

        syn_pts, syn_tp = [], []
        for i, tp in enumerate(task_points):
            obj = objects[tp['obj_idx']]
            pts = obj['pts'].copy()
            nx_cx, ny_cy = positions[i][0], positions[i][1]
            ocx, ocy, ocz = obj['center']
            pts[:, 0] += (nx_cx - ocx)
            pts[:, 1] += (ny_cy - ocy)
            pts[:, 2] += (-2.0 - ocz)
            syn_pts.append(pts)
            syn_tp.append({'x': nx_cx, 'y': ny_cy, 'cls': tp['cls'],
                           'score': tp['score'], 'dims': tp['dims'], 'heading': tp['heading']})

        return RoadNetworkGenerator._add_floor(syn_pts, syn_tp, 1, area_size,
                                                x_max_override=area_size,
                                                y_min_override=-area_size)

    @staticmethod
    def _add_floor(syn_pts, syn_tp, n_cols, spacing,
                   x_max_override=None, y_min_override=None):
        """添加地板点云 + 组装"""
        if x_max_override is not None:
            x_range = x_max_override
        else:
            x_range = n_cols * spacing
        if y_min_override is not None:
            y_min = y_min_override
        else:
            y_min = -((len(syn_tp) - 1) // n_cols + 1) * spacing

        fx = np.arange(-BEV_PAD, x_range + BEV_PAD, 0.5)
        fy = np.arange(y_min - BEV_PAD, BEV_PAD, 0.5)
        fxx, fyy = np.meshgrid(fx, fy)
        floor = np.column_stack([fxx.ravel(), fyy.ravel(),
                                  np.full(fxx.size, -2.2)])
        syn_pts.append(floor)
        combined = np.vstack(syn_pts)
        obj_mask = combined[:, 2] > -2.15
        return combined, obj_mask, syn_tp

    # ---- 合成场景布局 (统一入口) ----
    @staticmethod
    def layout_synthetic(objects: List[dict], task_points: List[dict],
                         mode: str = LAYOUT_MODE) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """统一入口: 根据 mode 选择 grid 或 random 布局"""
        if mode == 'random':
            return RoadNetworkGenerator.layout_random(objects, task_points)
        else:
            return RoadNetworkGenerator.layout_grid(objects, task_points)

    # ---- 主流程 ----
    def generate(self, frame_ids: List[str], data: dict) -> Tuple[nx.Graph, nx.Graph, List[dict]]:
        """
        主流程: 提取物体 → 合成布局 → PRM → MST
        返回: (PRM图, MST, task_points)
        """
        t0 = time.time()

        # 1. 提取物体点云
        objects, tp_all = self.extract_objects(frame_ids, data)
        n_tp = len(tp_all)

        # 2. 合成布局
        combined_pc, obj_mask, syn_tp = self.layout_synthetic(objects, tp_all)
        print(f"  Synthetic scene: {combined_pc.shape[0]} pts, {n_tp} objects "
              f"({time.time() - t0:.1f}s)")

        # 3. 占据栅格
        self.build_grid(combined_pc, obj_mask)

        # 4. 投影 task points
        tp_arr = np.zeros((n_tp, 2))
        for i, tp in enumerate(syn_tp):
            wx, wy = self.project_task_point(tp['x'], tp['y'])
            tp_arr[i] = [wx, wy]
        print(f"  Task points projected ({time.time() - t0:.1f}s)")

        # 5. PRM 采样
        road_nodes = []
        attempts = 0
        while len(road_nodes) < N_ROAD_SAMPLES and attempts < N_ROAD_SAMPLES * 30:
            attempts += 1
            gx = np.random.randint(3, self.gw - 3)
            gy = np.random.randint(3, self.gh - 3)
            if self.is_free(gx, gy):
                road_nodes.append(self.to_world(gx, gy))
        print(f"  PRM: {len(road_nodes)} samples ({time.time() - t0:.1f}s)")

        # 6. 构建 PRM 图
        all_nodes = np.array(list(tp_arr) + road_nodes)
        n_all = len(all_nodes)
        is_tp = [True] * n_tp + [False] * len(road_nodes)

        G_prm = nx.Graph()
        for i in range(n_all):
            G_prm.add_node(i, pos=(all_nodes[i, 0], all_nodes[i, 1]),
                           is_task=is_tp[i],
                           label=syn_tp[i]['cls'] if is_tp[i] else 'road')

        kdt = KDTree(all_nodes)
        for i in range(n_all):
            dists, idxs = kdt.query(all_nodes[i], k=min(K_NEIGHBORS + 2, n_all))
            for idx, d in zip(idxs, dists):
                if idx <= i or d > MAX_EDGE_LEN:
                    continue
                if self.line_free(all_nodes[i], all_nodes[idx]):
                    G_prm.add_edge(i, idx, weight=float(d), etype='prm')

        print(f"  PRM graph: {G_prm.number_of_nodes()}n/{G_prm.number_of_edges()}e "
              f"({time.time() - t0:.1f}s)")

        # 7. 配对路径
        tp_paths = {}
        n_pairs = n_tp * (n_tp - 1) // 2
        for i in range(n_tp):
            for j in range(i + 1, n_tp):
                try:
                    pl = nx.shortest_path_length(G_prm, i, j, weight='weight')
                    pn = nx.shortest_path(G_prm, i, j, weight='weight')
                    tp_paths[(i, j)] = ([all_nodes[n] for n in pn], pl)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pw = self.astar(tp_arr[i, 0], tp_arr[i, 1],
                                    tp_arr[j, 0], tp_arr[j, 1])
                    if pw is not None:
                        pl = sum(np.hypot(pw[k][0] - pw[k + 1][0],
                                          pw[k][1] - pw[k + 1][1])
                                 for k in range(len(pw) - 1))
                        tp_paths[(i, j)] = (pw, pl)

        print(f"  TP pairs: {len(tp_paths)}/{n_pairs} ({time.time() - t0:.1f}s)")

        # 8. MST
        G_tp = nx.Graph()
        for i in range(n_tp):
            G_tp.add_node(i, pos=(tp_arr[i, 0], tp_arr[i, 1]),
                          label=syn_tp[i]['cls'])
        for (i, j), (pw, pl) in tp_paths.items():
            G_tp.add_edge(i, j, weight=pl, path=pw)

        mst = nx.minimum_spanning_tree(G_tp, weight='weight')
        print(f"  MST: {mst.number_of_edges()} edges, "
              f"{mst.size(weight='weight'):.1f}m total")

        # 保存内部状态供可视化
        self._G_prm = G_prm
        self._all_nodes = all_nodes
        self._is_tp = is_tp
        self._syn_tp = syn_tp

        return G_prm, mst, syn_tp, combined_pc

    # ---- 路径简化 ----
    @staticmethod
    def simplify_path(path, angle_thresh: float = 0.86):
        if len(path) <= 2:
            return path
        result = [path[0]]
        for k in range(1, len(path) - 1):
            dx1 = path[k][0] - path[k - 1][0]
            dy1 = path[k][1] - path[k - 1][1]
            dx2 = path[k + 1][0] - path[k][0]
            dy2 = path[k + 1][1] - path[k][1]
            d1, d2 = np.hypot(dx1, dy1), np.hypot(dx2, dy2)
            if d1 < 1e-6 or d2 < 1e-6:
                continue
            dot = (dx1 * dx2 + dy1 * dy2) / (d1 * d2)
            if abs(dot) < angle_thresh:
                result.append(path[k])
        result.append(path[-1])
        return result

    # ---- 可视化 ----
    def visualize(self, G_prm: nx.Graph, mst: nx.Graph,
                  syn_tp: List[dict], output_path: str):
        ext = [self.x_min, self.x_max, self.y_min, self.y_max]

        bg = np.ones((self.gh, self.gw, 3))
        bg[self.occ_dilated > 0] = [0.88, 0.86, 0.83]
        bg[self.occ_raw > 0] = [0.72, 0.72, 0.72]

        n_tp = len(syn_tp)
        all_nodes = self._all_nodes
        is_tp = self._is_tp

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 13),
                                          facecolor='black')

        # 左: PRM 骨架
        for u, v, d in G_prm.edges(data=True):
            p1, p2 = all_nodes[u], all_nodes[v]
            tp_edge = (u < n_tp or v < n_tp)
            ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], '-',
                     color='#4A90D9',
                     linewidth=0.6 if tp_edge else 0.2,
                     alpha=0.5 if tp_edge else 0.15, zorder=2)

        # 右: 背景 + MST
        for ax in [ax1, ax2]:
            ax.imshow(bg, extent=ext, origin='lower', aspect='equal')

        for _, _, d in mst.edges(data=True):
            sp = self.simplify_path(d['path'])
            px, py = zip(*sp)
            ax2.plot(px, py, '-', color='#27AE60', linewidth=2.8, alpha=0.9, zorder=4)

        # PRM 采样节点
        prm_mask = np.array([not is_tp[i] for i in range(len(all_nodes))])
        ax2.scatter(all_nodes[prm_mask, 0], all_nodes[prm_mask, 1],
                    c='#6BB5FF', s=3, alpha=0.25, zorder=3)

        # Task points (物体)
        for tp in syn_tp:
            cls = tp['cls']
            c = CLASS_COLORS.get(cls, '#333333')
            m = CLASS_MARKERS.get(cls, 'o')
            dims = tp.get('dims', (0, 0))
            for ax in [ax1, ax2]:
                ax.scatter(tp['x'], tp['y'], c=c, s=150, marker=m,
                           edgecolors='black', linewidth=1.2, zorder=5)
                ax.annotate(f"{cls}\n{dims[0]:.1f}×{dims[1]:.1f}m",
                            (tp['x'], tp['y']), fontsize=5.5,
                            ha='center', va='bottom', xytext=(0, 9),
                            textcoords='offset points', fontweight='bold')

        # 图例
        leg = [mpatches.Patch(color=c, label=l) for l, c in CLASS_COLORS.items()]
        leg += [
            Line2D([0], [0], color='#4A90D9', lw=1, alpha=0.5, label='PRM Graph'),
            Line2D([0], [0], color='#27AE60', lw=2.5, alpha=0.9, label='MST Road Network'),
        ]
        for ax in [ax1, ax2]:
            ax.legend(handles=leg, loc='upper right', framealpha=0.92,
                      fontsize=8, ncol=2)

        n_pairs = n_tp * (n_tp - 1) // 2
        # Count connected pairs from MST
        mst_g = nx.Graph()
        mst_g.add_edges_from([(u, v) for u, v, _ in mst.edges(data=True)])
        comps = list(nx.connected_components(mst_g))
        n_conn = sum(len(c) * (len(c) - 1) // 2 for c in comps)

        raw_pct = self.occ_raw.sum() / self.occ_raw.size * 100
        dil_pct = self.occ_dilated.sum() / self.occ_dilated.size * 100

        ax1.set_title(
            f'Synthetic Warehouse — PRM Roadmap ({G_prm.number_of_nodes()}n/{G_prm.number_of_edges()}e)\n'
            f'{n_tp} objects from {len(FRAMES)} frames | '
            f'Occ={raw_pct:.1f}%→{dil_pct:.1f}% | '
            f'{n_conn}/{n_pairs} pairs connected',
            fontsize=11, fontweight='bold')
        ax2.set_title(
            f'MST Road Network — {mst.number_of_edges()} edges, '
            f'{mst.size(weight="weight"):.1f}m total\n'
            f'Objects cropped from detection boxes → flat floor layout | '
            f'Grid={self.grid_res}m | R={self.robot_radius}m',
            fontsize=11, fontweight='bold')

        for ax in [ax1, ax2]:
            ax.set_xlim(self.x_min, self.x_max)
            ax.set_ylim(self.y_min, self.y_max)
            ax.set_xlabel('X (m)', color='#CCCCCC')
            ax.set_ylabel('Y (m)', color='#CCCCCC')
            ax.set_aspect('equal')
            ax.set_facecolor('#1a1a1a')
            ax.tick_params(colors='#CCCCCC')
            ax.grid(True, alpha=0.12, color='#444444')
            for spine in ax.spines.values():
                spine.set_color('#555555')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='black', edgecolor='none')
        plt.close()
        print(f"  Saved: {output_path}")

    # ---- 导出 JSON ----
    def export_json(self, mst: nx.Graph, syn_tp: List[dict],
                    n_pairs: int, output_path: str):
        mst_g = nx.Graph()
        mst_g.add_edges_from([(u, v) for u, v, _ in mst.edges(data=True)])
        comps = list(nx.connected_components(mst_g))
        n_conn = sum(len(c) * (len(c) - 1) // 2 for c in comps)

        export = {
            'task_points': syn_tp,
            'n_task_points': len(syn_tp),
            'mst_edges': [
                {'u': int(u), 'v': int(v),
                 'weight': float(d['weight']),
                 'path': [[float(x), float(y)] for x, y in d['path']]}
                for u, v, d in mst.edges(data=True)
            ],
            'n_mst_edges': mst.number_of_edges(),
            'total_length_m': float(mst.size(weight='weight')),
            'n_pair_paths': n_conn,
            'n_pairs': n_pairs,
            'connectivity': f"{n_conn}/{n_pairs} ({100*n_conn/max(n_pairs,1):.0f}%)",
            'params': {
                'grid_resolution': self.grid_res,
                'robot_radius': self.robot_radius,
                'n_road_samples': N_ROAD_SAMPLES,
                'k_neighbors': K_NEIGHBORS,
                'max_edge_len': MAX_EDGE_LEN,
                'grid_spacing': GRID_SPACING,
                'grid_cols': GRID_COLS,
                'bounds': {
                    'x_min': float(self.x_min), 'x_max': float(self.x_max),
                    'y_min': float(self.y_min), 'y_max': float(self.y_max),
                },
                'frames': FRAMES,
            },
        }
        with open(output_path, 'w') as f:
            json.dump(export, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {output_path}")


# ============================================================
# main
# ============================================================
def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"{'='*60}")
    print(f"AGV Global Road Network Generator")
    print(f"{'='*60}")

    # 加载检测结果
    with open('output/warehouse_inference_viz/prediction_centers.json') as f:
        data = json.load(f)

    # 生成路网
    gen = RoadNetworkGenerator(grid_res=GRID_RES, robot_radius=ROBOT_RADIUS)

    print(f"\n[1] Extracting objects & building synthetic scene...")
    G_prm, mst, syn_tp, _ = gen.generate(FRAMES, data)

    print(f"\n[2] Visualizing...")
    gen.visualize(G_prm, mst, syn_tp,
                  f'{OUTPUT_DIR}/agv_road_network_synthetic.png')

    print(f"\n[3] Exporting...")
    n_pairs = len(syn_tp) * (len(syn_tp) - 1) // 2
    gen.export_json(mst, syn_tp, n_pairs,
                    f'{OUTPUT_DIR}/road_network_synthetic.json')

    # 摘要
    n_conn = sum(len(c) * (len(c) - 1) // 2
                 for c in nx.connected_components(
                     nx.Graph([(u, v) for u, v, _ in mst.edges(data=True)])))

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"  Frames:            {len(FRAMES)}")
    print(f"  Objects:           {len(syn_tp)}")
    print(f"  Connectivity:      {n_conn}/{n_pairs} ({100*n_conn/max(n_pairs,1):.0f}%)")
    print(f"  MST edges:         {mst.number_of_edges()}")
    print(f"  Total road length: {mst.size(weight='weight'):.1f} m")
    print(f"  Occupancy:         {gen.occ_raw.sum()/gen.occ_raw.size*100:.1f}% → "
          f"{gen.occ_dilated.sum()/gen.occ_dilated.size*100:.1f}% (dilated)")
    print(f"  Time:              {time.time() - t0:.1f}s")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
