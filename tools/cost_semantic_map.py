#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
代价逻辑语义地图生成 — 多帧拼接点云 → 业务点/路网 → 代价语义地图
=================================================================

图(实验图)三阶段管线:
  (a) 感知层    — 多帧拼接点云 BEV + 检测框 (原始输入)
  (b) 语义层    — 业务点(3 语义层) + 避障路网骨架 (MST of A* 路径)
  (c) 代价语义层 — 边颜色=拥挤度, 边宽度=车流量, 标注拥挤/高车流通道,
                  输出可直接用于动态规划 (Dijkstra/A*) 的带权路网

代价模型 (输出到 JSON):
    Cost(e) = L * (1 + α·Cong + β·Flow)
    Cong(e) : 沿边净空(距离变换) 反比 + 存储区邻近因子 → 拥挤度 [0,1]
    Flow(e) : 业务点对最短路径边介数 → 车流量 [0,1]
    α = 1.2, β = 0.8

用法:
    conda activate pc
    export PYTHONPATH=/code/OpenPCDet:$PYTHONPATH
    python tools/cost_semantic_map.py
"""

import os
import sys
import json
import time
import heapq
from collections import deque

import numpy as np
import networkx as nx
from scipy import ndimage
from skimage import morphology

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# ---------------------------------------------------------------
# 中文字体
# ---------------------------------------------------------------
for _p in ['/home/kie/.fonts/simhei.ttf', '/home/kie/.fonts/simsun.ttc']:
    if os.path.exists(_p):
        fm.fontManager.addfont(_p)
plt.rcParams['font.sans-serif'] = ['SimHei', 'SimSun', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------------
# 配置
# ---------------------------------------------------------------
GRID_RES = 0.25          # 栅格分辨率 (m)
ROBOT_R = 0.4            # AGV 安全半径 (m)
OCC_Z = -0.5             # 障碍高度阈值: z > OCC_Z 为障碍
CLEAR_FREE = 2.5         # 完全通畅通道净空 (m)
STORAGE_R = 5.0          # 存储区拥挤影响半径 (m)
DP_TOL = 1.4             # 路径简化容差 (m)
SNAP_R = 0.9             # 路网节点合并半径 (m)
ALPHA_CONG = 1.2         # 代价模型: 拥挤权重
BETA_FLOW = 0.8          # 代价模型: 车流权重
SEED = 42

DATA_PTS = 'data/warehouse_stitched/points/000000.npy'
DATA_LBL = 'data/warehouse_stitched/labels/000000.txt'
OUT_FIG = 'output/fig_cost_semantic_map.png'
OUT_JSON = 'output/cost_semantic_map.json'
OUT_DIR = os.path.dirname(OUT_FIG)

# 3 语义层 (已验证调色板, light 下全部通过)
LAYERS = [
    dict(key='storage',   zh='存储层', en='Storage',  color='#eda100', marker='s',
         classes=['箱子'],       abbr='Box', label_zh='箱子 Box'),
    dict(key='transport', zh='转运层', en='Transport', color='#2a78d6', marker='^',
         classes=['电动运输车', '货运自行车'], abbr='ELF/CB', label_zh='电动运输车 ELF · 货运自行车 CB'),
    dict(key='delivery',  zh='投递层', en='Delivery', color='#1baf7a', marker='D',
         classes=['无人搬运车'],   abbr='FTS', label_zh='无人搬运车 FTS'),
]
LAYER_BY_CLS = {c: L for L in LAYERS for c in L['classes']}

# 原始检测 4 类 (感知层面板)
CLASS_COLORS = {'箱子': '#FFB400', '电动运输车': '#E74C3C',
                '货运自行车': '#3498DB', '无人搬运车': '#32FF64'}

# 拥挤度连续色带 (单红相, 浅→深)
CONG_COLORS = ['#f8d8cf', '#f2b3a0', '#e67e52', '#d34e2f', '#a82b18', '#6b130a']
CONG_CMAP = LinearSegmentedColormap.from_list('cong', CONG_COLORS, N=256)


# ---------------------------------------------------------------
# 占据栅格 + A*
# ---------------------------------------------------------------
class OccupancyGrid:
    def __init__(self, pts, grid_res=GRID_RES, robot_r=ROBOT_R):
        self.res = grid_res
        self.r_px = max(1, int(robot_r / grid_res))
        self.x_min = pts[:, 0].min() - 2
        self.x_max = pts[:, 0].max() + 2
        self.y_min = pts[:, 1].min() - 2
        self.y_max = pts[:, 1].max() + 2
        self.gw = int((self.x_max - self.x_min) / grid_res) + 1
        self.gh = int((self.y_max - self.y_min) / grid_res) + 1

        occ = np.zeros((self.gh, self.gw), dtype=np.uint8)
        m = pts[:, 2] > OCC_Z
        ox = ((pts[m, 0] - self.x_min) / grid_res).astype(int)
        oy = ((pts[m, 1] - self.y_min) / grid_res).astype(int)
        ok = (ox >= 0) & (ox < self.gw) & (oy >= 0) & (oy < self.gh)
        occ[oy[ok], ox[ok]] = 1

        k = np.ones((2 * self.r_px + 1, 2 * self.r_px + 1), dtype=np.uint8)
        self.occ = morphology.binary_dilation(occ, k).astype(np.uint8)
        self.free = 1 - self.occ
        # 净空 (m): 自由空间到最近障碍的距离
        self.clearance = ndimage.distance_transform_edt(self.free) * grid_res

    def to_grid(self, wx, wy):
        return int((wx - self.x_min) / self.res), int((wy - self.y_min) / self.res)

    def to_world(self, gx, gy):
        return gx * self.res + self.x_min, gy * self.res + self.y_min

    def is_free(self, gx, gy):
        return 0 <= gx < self.gw and 0 <= gy < self.gh and self.free[gy, gx] == 1

    def project_free(self, wx, wy):
        """投影到最近自由格 (BFS)"""
        gx, gy = self.to_grid(wx, wy)
        if self.is_free(gx, gy):
            return wx, wy
        q = deque([(gx, gy)])
        seen = {(gx, gy)}
        while q:
            cx, cy = q.popleft()
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (-1, 1), (1, -1), (-1, -1)]:
                nxg, nyg = cx + dx, cy + dy
                if not (0 <= nxg < self.gw and 0 <= nyg < self.gh) or (nxg, nyg) in seen:
                    continue
                seen.add((nxg, nyg))
                if self.is_free(nxg, nyg):
                    return self.to_world(nxg, nyg)
                q.append((nxg, nyg))
        return wx, wy

    def astar(self, sx, sy, gx, gy):
        """A* 最短避障路径, 返回世界坐标 [(x,y),...] 或 None"""
        sg = self.to_grid(sx, sy)
        gg = self.to_grid(gx, gy)
        if not self.is_free(*sg) or not self.is_free(*gg):
            return None
        start = (sg[0], sg[1], 0.0)
        open_set = [(abs(sg[0] - gg[0]) + abs(sg[1] - gg[1]), 0, sg[0], sg[1])]
        came = {}
        gs = {start[:2]: 0.0}
        while open_set:
            _, cost, cx, cy = heapq.heappop(open_set)
            if (cx, cy) == gg:
                path = [gg]
                while (cx, cy) != sg:
                    cx, cy = came[(cx, cy)]
                    path.append((cx, cy))
                return [self.to_world(px, py) for px, py in reversed(path)]
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (-1, 1), (1, -1), (-1, -1)]:
                nxg, nyg = cx + dx, cy + dy
                if not self.is_free(nxg, nyg):
                    continue
                mc = 1.414 if dx and dy else 1.0
                ng = cost + mc
                if (nxg, nyg) not in gs or ng < gs[(nxg, nyg)]:
                    gs[(nxg, nyg)] = ng
                    heapq.heappush(open_set,
                                   (ng + abs(nxg - gg[0]) + abs(nyg - gg[1]), ng, nxg, nyg))
                    came[(nxg, nyg)] = (cx, cy)
        return None


# ---------------------------------------------------------------
# 路径简化 (Douglas-Peucker)
# ---------------------------------------------------------------
def douglas_peucker(pts, tol):
    if len(pts) <= 2:
        return pts
    p0, p1 = np.array(pts[0]), np.array(pts[-1])
    seg = p1 - p0
    seg_len = np.hypot(*seg)
    dmax, idx = 0.0, 0
    for i in range(1, len(pts) - 1):
        p = np.array(pts[i])
        if seg_len < 1e-9:
            d = np.hypot(*(p - p0))
        else:
            d = abs(np.cross(seg, p - p0)) / seg_len
        if d > dmax:
            dmax, idx = d, i
    if dmax > tol:
        left = douglas_peucker(pts[:idx + 1], tol)
        right = douglas_peucker(pts[idx:], tol)
        return left[:-1] + right
    return [pts[0], pts[-1]]


# ---------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------
def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- 1. 加载 ----
    pts = np.load(DATA_PTS)
    labels = []
    with open(DATA_LBL) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 8:
                labels.append(dict(x=float(p[0]), y=float(p[1]), z=float(p[2]),
                                   dx=float(p[3]), dy=float(p[4]), dz=float(p[5]),
                                   heading=float(p[6]), cls=p[7]))
    n_bp = len(labels)
    print(f"[1] 点云 {pts.shape[0]:,} | 业务对象 {n_bp}")

    # ---- 2. 占据栅格 + 净空 ----
    grid = OccupancyGrid(pts)
    occ_pct = grid.occ.sum() / grid.occ.size * 100
    print(f"[2] 栅格 {grid.gw}×{grid.gh}, 占据率 {occ_pct:.1f}%")

    # ---- 3. 业务点投影 + 语义层 ----
    bp = []  # dict(x, y, layer_idx, cls, id)
    for i, lbl in enumerate(labels):
        wx, wy = grid.project_free(lbl['x'], lbl['y'])
        bp.append(dict(id=i, x=wx, y=wy, cls=lbl['cls'],
                       layer=LAYER_BY_CLS[lbl['cls']]['key']))
    bp_xy = np.array([[b['x'], b['y']] for b in bp])

    # ---- 4. 路网: 业务点全连接 A* 距离 → MST → 简化路径 ----
    print("[3] 计算业务点对 A* 路径 (全连接)...")
    t1 = time.time()
    Gc = nx.Graph()
    for i in range(n_bp):
        Gc.add_node(i)
    paths = {}
    for i in range(n_bp):
        for j in range(i + 1, n_bp):
            pl = grid.astar(bp_xy[i, 0], bp_xy[i, 1], bp_xy[j, 0], bp_xy[j, 1])
            if pl is None:
                d = float(np.hypot(bp_xy[i, 0] - bp_xy[j, 0], bp_xy[i, 1] - bp_xy[j, 1]))
            else:
                d = float(sum(np.hypot(pl[k][0] - pl[k + 1][0], pl[k][1] - pl[k + 1][1])
                              for k in range(len(pl) - 1)))
            Gc.add_edge(i, j, weight=d)
            paths[(i, j)] = pl
    mst = nx.minimum_spanning_tree(Gc, weight='weight')
    print(f"    MST {mst.number_of_edges()} 边, "
          f"总长 {mst.size(weight='weight'):.0f}m ({time.time()-t1:.1f}s)")

    # 简化路径 + 路网节点合并 (形成交叉口)
    road_nodes = []      # [(x, y)]
    road_edges = []      # [(u, v)]
    bp_node = {}         # 业务点 id -> road node id
    for i in range(n_bp):
        bp_node[i] = None

    def _snap(x, y):
        for k, (nx_, ny_) in enumerate(road_nodes):
            if np.hypot(x - nx_, y - ny_) < SNAP_R:
                return k
        road_nodes.append((x, y))
        return len(road_nodes) - 1

    mst_edges = list(mst.edges())
    for (u, v) in mst_edges:
        pl = paths.get((u, v)) or paths.get((v, u))
        if pl is None:
            pl = [tuple(bp_xy[u]), tuple(bp_xy[v])]
        simp = douglas_peucker(pl, DP_TOL)
        ids = [_snap(x, y) for x, y in simp]
        for k in range(len(ids) - 1):
            if ids[k] != ids[k + 1]:
                road_edges.append((ids[k], ids[k + 1]))
        if bp_node[u] is None:
            bp_node[u] = _snap(bp_xy[u, 0], bp_xy[u, 1])
        if bp_node[v] is None:
            bp_node[v] = _snap(bp_xy[v, 0], bp_xy[v, 1])

    road_nodes_arr = np.array(road_nodes)
    # 去重边
    road_edges = list(dict.fromkeys(tuple(sorted(e)) for e in road_edges))
    # 计算边长
    edge_len = {e: float(np.hypot(*(road_nodes_arr[e[1]] - road_nodes_arr[e[0]])))
                for e in road_edges}
    print(f"[4] 路网 {len(road_nodes)} 节点 / {len(road_edges)} 边")

    # ---- 5. 代价模型 ----
    # Cong: 沿边净空 + 存储区邻近
    bp_box = [b for b in bp if b['layer'] == 'storage']
    box_xy = np.array([[b['x'], b['y']] for b in bp_box]) if bp_box else np.zeros((0, 2))

    def _edge_congestion(e):
        (x0, y0), (x1, y1) = road_nodes_arr[e[0]], road_nodes_arr[e[1]]
        L = edge_len[e]
        n = max(int(L / 0.8), 1)
        clears = []
        for k in range(n + 1):
            t = k / n
            wx, wy = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            gx, gy = grid.to_grid(wx, wy)
            gx = min(max(gx, 0), grid.gw - 1)
            gy = min(max(gy, 0), grid.gh - 1)
            clears.append(grid.clearance[gy, gx])
        clear = min(clears)
        geom = np.clip((CLEAR_FREE - clear) / CLEAR_FREE, 0, 1) ** 1.5
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        if len(box_xy):
            dbox = float(np.min(np.hypot(box_xy[:, 0] - mx, box_xy[:, 1] - my)))
        else:
            dbox = 1e9
        store = np.clip((STORAGE_R - dbox) / STORAGE_R, 0, 1) if dbox < STORAGE_R else 0.0
        return 0.7 * geom + 0.3 * store

    cong = {e: _edge_congestion(e) for e in road_edges}
    p95_c = np.percentile(list(cong.values()), 95)
    cong = {e: min(v / max(p95_c, 1e-9), 1.0) for e, v in cong.items()}

    # Flow: 业务点对最短路径边介数 (在路网图上)
    Gr = nx.Graph()
    Gr.add_nodes_from(range(len(road_nodes)))
    for e in road_edges:
        Gr.add_edge(e[0], e[1], weight=edge_len[e])
    flow = {e: 0 for e in road_edges}
    for i in range(n_bp):
        for j in range(i + 1, n_bp):
            try:
                sp = nx.shortest_path(Gr, bp_node[i], bp_node[j], weight='weight')
            except nx.NetworkXNoPath:
                continue
            for a, b in zip(sp[:-1], sp[1:]):
                key = tuple(sorted((a, b)))
                if key in flow:
                    flow[key] += 1
    p95_f = np.percentile(list(flow.values()), 95)
    flow = {e: min(v / max(p95_f, 1e-9), 1.0) for e, v in flow.items()}

    # 总代价: Cost = L * (1 + α·Cong + β·Flow)
    cost = {e: edge_len[e] * (1 + ALPHA_CONG * cong[e] + BETA_FLOW * flow[e])
            for e in road_edges}

    print(f"[5] 拥挤度 {min(cong.values()):.2f}~{max(cong.values()):.2f} | "
          f"车流 {min(flow.values()):.2f}~{max(flow.values()):.2f} | "
          f"总长 {sum(edge_len.values()):.0f}m")

    # ---- 6. 拥挤区热场 (仅用于可视化底图) ----
    xx = grid.x_min + (np.arange(grid.gw) + 0.5) * grid.res
    yy = grid.y_min + (np.arange(grid.gh) + 0.5) * grid.res
    X, Y = np.meshgrid(xx, yy)
    field = np.exp(-grid.clearance / 1.2)
    for bx, by in box_xy:
        field += 1.5 * np.exp(-((X - bx) ** 2 + (Y - by) ** 2) / (2 * 6.0 ** 2))
    field = ndimage.gaussian_filter(field, 1.5)
    field = field / max(field.max(), 1e-9)

    # ---- 7. 可视化 ----
    ext = [grid.x_min, grid.x_max, grid.y_min, grid.y_max]
    t2 = time.time()

    def _faded_bg(ax, alpha, color='#b9b9b9'):
        # 降采样后散点绘制, 点云背景淡化
        sub = pts[::2]
        ax.scatter(sub[:, 0], sub[:, 1], s=0.5, c=color, alpha=alpha,
                   edgecolors='none', linewidths=0, zorder=0)
        ax.set_xlim(grid.x_min, grid.x_max)
        ax.set_ylim(grid.y_min, grid.y_max)

    def _plot_nodes(ax, with_label=True, fs=6.2, zorder=6):
        for b in bp:
            L = LAYER_BY_CLS[b['cls']]
            ax.scatter(b['x'], b['y'], c=L['color'], s=72, marker=L['marker'],
                       edgecolors='white', linewidths=0.8, zorder=zorder)
            if with_label:
                abbr = {'箱子': 'Box', '电动运输车': 'ELF', '货运自行车': 'CB',
                        '无人搬运车': 'FTS'}[b['cls']]
                ax.annotate(f"{abbr}·{b['id']}", (b['x'], b['y']),
                            fontsize=fs, ha='left', va='center',
                            xytext=(6, 0), textcoords='offset points', zorder=zorder + 1)

    def _style_ax(ax, title):
        ax.set_title(title, fontsize=11.5, fontweight='bold', pad=6)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=8, colors='#898781')
        ax.grid(True, alpha=0.10, color='#c9c8c2', linewidth=0.5)
        ax.set_facecolor('#ffffff')
        for sp in ax.spines.values():
            sp.set_color('#c3c2b7')

    fig, axes = plt.subplots(1, 3, figsize=(27.5, 7.2), facecolor='white',
                             constrained_layout=True)
    axA, axB, axC = axes

    # ========== (a) 感知层 ==========
    _faded_bg(axA, alpha=0.32, color='#a6a6a6')
    for lbl in labels:
        cx, cy, dx, dy, hd = lbl['x'], lbl['y'], lbl['dx'], lbl['dy'], lbl['heading']
        c = CLASS_COLORS.get(lbl['cls'], '#888888')
        corners = []
        for sx, sy in [(-dx / 2, -dy / 2), (dx / 2, -dy / 2), (dx / 2, dy / 2), (-dx / 2, dy / 2)]:
            rx = sx * np.cos(hd) - sy * np.sin(hd) + cx
            ry = sx * np.sin(hd) + sy * np.cos(hd) + cy
            corners.append((rx, ry))
        corners.append(corners[0])
        px_, py_ = zip(*corners)
        axA.plot(px_, py_, color=c, lw=1.2, alpha=0.85, zorder=4)
        axA.fill(px_, py_, color=c, alpha=0.12, zorder=3)
        axA.scatter([cx], [cy], c=c, s=9, zorder=5)
    axA.text(0.985, 0.97, f"{pts.shape[0]:,} pts · {n_bp} detections",
             transform=axA.transAxes, ha='right', va='top', fontsize=8, color='#52514e')
    # 4 类图例
    legA = [mpatches.Patch(color=c, label=zh, alpha=0.55)
            for zh, c in [('箱子 Box', '#FFB400'), ('电动运输车 ELF', '#E74C3C'),
                          ('货运自行车 CB', '#3498DB'), ('无人搬运车 FTS', '#32FF64')]]
    axA.legend(handles=legA, loc='upper left', framealpha=0.92, fontsize=7.5, ncol=2,
               title='检测框 Detection', title_fontsize=8)
    _style_ax(axA, "(a) 感知层 · 多帧拼接点云\nPerception · stitched point cloud")

    # ========== (b) 语义层 ==========
    _faded_bg(axB, alpha=0.09)
    # 路网骨架 (细线)
    segs = [[road_nodes_arr[u], road_nodes_arr[v]] for u, v in road_edges]
    lcB = LineCollection(segs, colors='#6f92a8', linewidths=1.4, alpha=0.9, zorder=2)
    axB.add_collection(lcB)
    _plot_nodes(axB, with_label=True)
    legB = [Line2D([0], [0], marker=L['marker'], color='w', markerfacecolor=L['color'],
                   markeredgecolor='white', markersize=8,
                   label=f"{L['zh']} · {L['en']}") for L in LAYERS]
    axB.legend(handles=legB, loc='upper left', framealpha=0.92, fontsize=7.5,
               title='业务点语义层 Semantic layer', title_fontsize=8)
    axB.text(0.985, 0.97, f"{len(road_nodes)} nodes · {len(road_edges)} edges (MST/A*)",
             transform=axB.transAxes, ha='right', va='top', fontsize=8, color='#52514e')
    _style_ax(axB, "(b) 语义层 · 业务点 + 避障路网\nSemantics · business points + road network")

    # ========== (c) 代价语义层 ==========
    _faded_bg(axC, alpha=0.055)
    # 拥挤区热场
    ov = np.zeros((grid.gh, grid.gw, 4))
    ov[..., 3] = np.clip((field - 0.35) / 0.45, 0, 1) * 0.17
    rgb = CONG_CMAP(field)[..., :3]
    ov[..., :3] = rgb
    axC.imshow(ov, extent=ext, origin='lower', aspect='equal', zorder=1)
    # 代价边: 颜色=拥挤度, 宽度=车流量
    order = sorted(road_edges, key=lambda e: flow[e])
    segsC = [[road_nodes_arr[u], road_nodes_arr[v]] for u, v in order]
    lws = [0.6 + 3.6 * flow[e] for e in order]
    lcC = LineCollection(segsC, cmap=CONG_CMAP, norm=Normalize(0, 1),
                         linewidths=lws, zorder=4)
    lcC.set_array(np.array([cong[e] for e in order]))
    axC.add_collection(lcC)
    _plot_nodes(axC, with_label=True)

    # 拥挤/高车流通道标注
    e_cong = max(road_edges, key=lambda e: cong[e])
    e_flow = max(road_edges, key=lambda e: flow[e])
    if e_cong == e_flow:
        e_flow = sorted(road_edges, key=lambda e: flow[e])[-2]
    (cx0, cy0) = road_nodes_arr[e_cong[0]]
    (cx1, cy1) = road_nodes_arr[e_cong[1]]
    mx, my = (cx0 + cx1) / 2, (cy0 + cy1) / 2
    axC.annotate('① 拥挤通道\nCongested corridor', (mx, my), xytext=(mx + 5, my + 7),
                 fontsize=8, fontweight='bold', color='#7a1a0e',
                 arrowprops=dict(arrowstyle='->', color='#a82b18', lw=1.4, alpha=0.9),
                 bbox=dict(boxstyle='round,pad=0.3', fc='#fdf3f0', ec='#c46a52', alpha=0.95))
    (fx0, fy0) = road_nodes_arr[e_flow[0]]
    (fx1, fy1) = road_nodes_arr[e_flow[1]]
    fmx, fmy = (fx0 + fx1) / 2, (fy0 + fy1) / 2
    axC.annotate('② 高车流通道\nHigh-traffic passage', (fmx, fmy),
                 xytext=(fmx - 14, fmy + 8), fontsize=8, fontweight='bold', color='#1c5cab',
                 arrowprops=dict(arrowstyle='->', color='#2a78d6', lw=1.4, alpha=0.9),
                 bbox=dict(boxstyle='round,pad=0.3', fc='#eef4fc', ec='#5b8fd4', alpha=0.95))

    # 代价公式框
    cost_box = (f"代价 Cost = L·(1 + {ALPHA_CONG}·Cong + {BETA_FLOW}·Flow)\n"
                f"Cong : 通道净空→拥挤度 [0,1]\n"
                f"Flow : 业务最短路径边介数→车流量 [0,1]\n"
                f"路网: {len(road_edges)} 边 · {sum(edge_len.values()):.0f}m")
    axC.text(0.015, 0.985, cost_box, transform=axC.transAxes, ha='left', va='top',
             fontsize=7.5, fontname='SimHei', color='#3a3a3a',
             bbox=dict(boxstyle='round,pad=0.45', fc='#fafaf7', ec='#c3c2b7', alpha=0.95))
    # 车流量图例 (宽度)
    lw_leg = [Line2D([0], [0], color='#2a78d6', lw=1.0, label='低车流 low'),
              Line2D([0], [0], color='#2a78d6', lw=3.2, label='中车流 mid'),
              Line2D([0], [0], color='#2a78d6', lw=5.4, label='高车流 high')]
    legC = axC.legend(handles=lw_leg, loc='upper left', framealpha=0.92, fontsize=7.5,
                      title='边宽度 = 车流量 Flow', title_fontsize=8)
    _style_ax(axC, "(c) 代价语义地图 · 拥挤/车流成本\nCost-semantic map for dynamic planning")

    # 拥挤度 colorbar
    cb = fig.colorbar(lcC, ax=[axC], fraction=0.030, pad=0.015)
    cb.set_label('拥挤度 Congestion', fontsize=9)
    cb.ax.tick_params(labelsize=7.5)
    cb.outline.set_edgecolor('#c3c2b7')

    # 管线箭头 (置于面板间隙中部)
    for ax_, pos in [(axA, 0.345), (axB, 0.655)]:
        fig.text(pos, 0.5, '→', fontsize=30, color='#a0a09a', ha='center', va='center')

    fig.suptitle('代价逻辑语义地图构建 — 感知 → 语义 → 代价 (Cost-Semantic Map Construction)',
                 fontsize=15, fontweight='bold', y=0.98)
    fig.savefig(OUT_FIG, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[6] 图 → {OUT_FIG} ({time.time()-t2:.1f}s)")

    # ---- 8. 导出 JSON (供动态规划) ----
    export = {
        'meta': {
            'name': 'cost_semantic_map',
            'scene': 'warehouse_stitched/000000',
            'bounds': {'x_min': grid.x_min, 'x_max': grid.x_max,
                       'y_min': grid.y_min, 'y_max': grid.y_max},
            'cost_model': {'formula': 'Cost = L * (1 + alpha*Cong + beta*Flow)',
                           'alpha': ALPHA_CONG, 'beta': BETA_FLOW,
                           'clear_free_m': CLEAR_FREE, 'storage_r_m': STORAGE_R,
                           'robot_radius_m': ROBOT_R, 'grid_res_m': GRID_RES},
            'layers': [{'key': L['key'], 'zh': L['zh'], 'en': L['en'],
                        'classes': L['classes'], 'color': L['color']} for L in LAYERS],
        },
        'business_nodes': [
            {'id': b['id'], 'x': b['x'], 'y': b['y'], 'class': b['cls'],
             'layer': b['layer'], 'road_node': bp_node[b['id']]} for b in bp
        ],
        'road_nodes': [{'id': k, 'x': float(x), 'y': float(y)}
                       for k, (x, y) in enumerate(road_nodes)],
        'road_edges': [
            {'u': int(e[0]), 'v': int(e[1]), 'length_m': round(edge_len[e], 3),
             'congestion': round(cong[e], 4), 'flow': round(flow[e], 4),
             'cost': round(cost[e], 4)}
            for e in road_edges
        ],
        'stats': {
            'n_business_points': n_bp, 'n_road_nodes': len(road_nodes),
            'n_road_edges': len(road_edges), 'total_road_m': round(sum(edge_len.values()), 1),
            'occupancy_pct': round(occ_pct, 2),
        },
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
    print(f"[7] 路网代价图 → {OUT_JSON}")

    print(f"\n完成 总耗时 {time.time()-t0:.1f}s")
    print(f"  JSON 已含边代价, 可直接 Dijkstra/A* 动态规划")


if __name__ == '__main__':
    main()
