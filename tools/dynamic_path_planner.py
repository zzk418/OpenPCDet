#!/usr/bin/env python
"""
动态路径规划 — 无需预建路网, 占据栅格 + A* 即时规划任意两点间路径.

用法:
    python tools/dynamic_path_planner.py
"""

import numpy as np, heapq, json, os, time
from skimage import morphology
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

GRID_RES = 0.25
ROBOT_R = 0.4

CLASS_COLORS = {
    "箱子": "#FFB400", "电动运输车": "#E74C3C",
    "货运自行车": "#3498DB", "无人搬运车": "#32FF64",
}
CLASS_EN = {"箱子": "Box", "电动运输车": "ELF",
            "货运自行车": "CargoBike", "无人搬运车": "FTS"}

# ============================================================
# 占据栅格
# ============================================================
class OccupancyGrid:
    def __init__(self, pts, grid_res=GRID_RES, robot_r=ROBOT_R):
        self.res = grid_res
        self.r_px = max(1, int(robot_r / grid_res))

        self.x_min = pts[:,0].min() - 2
        self.x_max = pts[:,0].max() + 2
        self.y_min = pts[:,1].min() - 2
        self.y_max = pts[:,1].max() + 2
        self.gw = int((self.x_max - self.x_min) / grid_res) + 1
        self.gh = int((self.y_max - self.y_min) / grid_res) + 1

        # 障碍: Z > -0.5
        grid = np.zeros((self.gh, self.gw), dtype=np.uint8)
        occ_mask = pts[:, 2] > -0.5
        ox = ((pts[occ_mask, 0] - self.x_min) / grid_res).astype(int)
        oy = ((pts[occ_mask, 1] - self.y_min) / grid_res).astype(int)
        m = (ox >= 0) & (ox < self.gw) & (oy >= 0) & (oy < self.gh)
        grid[oy[m], ox[m]] = 1

        # 膨胀
        k = np.ones((2 * self.r_px + 1, 2 * self.r_px + 1), dtype=np.uint8)
        self.occ = morphology.binary_dilation(grid, k).astype(np.uint8)
        self.free = 1 - self.occ

    def to_grid(self, wx, wy):
        return int((wx - self.x_min) / self.res), int((wy - self.y_min) / self.res)

    def to_world(self, gx, gy):
        return gx * self.res + self.x_min, gy * self.res + self.y_min

    def is_free(self, gx, gy):
        return 0 <= gx < self.gw and 0 <= gy < self.gh and self.free[gy, gx] == 1

    # ---- A* ----
    def plan(self, sx, sy, gx, gy):
        """返回路径 [(wx,wy), ...] 或 None"""
        sgx, sgy = self.to_grid(sx, sy)
        ggx, ggy = self.to_grid(gx, gy)
        if not self.is_free(sgx, sgy) or not self.is_free(ggx, ggy):
            # 投影到最近自由格
            sgx, sgy = self._project(sgx, sgy)
            ggx, ggy = self._project(ggx, ggy)
            if sgx is None or ggx is None:
                return None

        open_set = [(abs(sgx-ggx)+abs(sgy-ggy), 0, sgx, sgy)]
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

            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]:
                nx_g, ny_g = cx + dx, cy + dy
                if not self.is_free(nx_g, ny_g):
                    continue
                mc = 1.414 if dx and dy else 1.0
                ng = cost + mc
                if (nx_g, ny_g) not in g_score or ng < g_score[(nx_g, ny_g)]:
                    g_score[(nx_g, ny_g)] = ng
                    heapq.heappush(open_set,
                        (ng + abs(nx_g-ggx) + abs(ny_g-ggy), ng, nx_g, ny_g))
                    came_from[(nx_g, ny_g)] = (cx, cy)
        return None

    def _project(self, gx, gy):
        """BFS 投影到最近自由格"""
        from collections import deque
        if self.is_free(gx, gy):
            return gx, gy
        visited = {(gx, gy)}
        q = deque([(gx, gy)])
        while q:
            cx, cy = q.popleft()
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]:
                nx_g, ny_g = cx+dx, cy+dy
                if (nx_g, ny_g) in visited:
                    continue
                if not (0 <= nx_g < self.gw and 0 <= ny_g < self.gh):
                    continue
                visited.add((nx_g, ny_g))
                if self.is_free(nx_g, ny_g):
                    return nx_g, ny_g
                q.append((nx_g, ny_g))
        return None, None


# ============================================================
# main
# ============================================================
def main():
    t0 = time.time()

    # 加载
    pts = np.load("data/warehouse_stitched/points/000000.npy")
    labels = []
    with open("data/warehouse_stitched/labels/000000.txt") as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 8:
                labels.append({"x": float(p[0]), "y": float(p[1]), "class": p[7]})
    n_tp = len(labels)

    # 建栅格
    grid = OccupancyGrid(pts)
    occ_pct = grid.occ.sum() / grid.occ.size * 100
    print("Grid: {}x{}, occ={:.1f}%, free={:.1f}%".format(
        grid.gw, grid.gh, occ_pct, 100-occ_pct))

    # 测试: 规划 Box(6) → FTS(14) (业务场景: 箱子到搬运车)
    src_idx, dst_idx = 6, 14
    path = grid.plan(labels[src_idx]["x"], labels[src_idx]["y"],
                     labels[dst_idx]["x"], labels[dst_idx]["y"])

    if path:
        path_len = sum(np.hypot(path[k][0]-path[k+1][0],
                                path[k][1]-path[k+1][1])
                       for k in range(len(path)-1))
        straight = np.hypot(labels[src_idx]["x"] - labels[dst_idx]["x"],
                            labels[src_idx]["y"] - labels[dst_idx]["y"])
        print("Path #{}->#{}: {} pts, {:.1f}m (straight={:.1f}m, detour={:.0f}%)".format(
            src_idx, dst_idx, len(path), path_len, straight,
            (path_len/straight-1)*100))
    else:
        print("Path #{}->#{}: FAILED".format(src_idx, dst_idx))

    # ---- 可视化 ----
    fig, ax = plt.subplots(figsize=(14, 13), facecolor="white")

    # 背景
    bg = np.ones((grid.gh, grid.gw, 3))
    bg[grid.occ > 0] = [0.88, 0.86, 0.83]
    ext = [grid.x_min, grid.x_max, grid.y_min, grid.y_max]
    ax.imshow(bg, extent=ext, origin="lower", aspect="equal")

    # 物体
    for i, lbl in enumerate(labels):
        c = CLASS_COLORS.get(lbl["class"], "#333")
        ax.scatter(lbl["x"], lbl["y"], c=c, s=180, edgecolors="black",
                   lw=1.2, zorder=5)
        name = CLASS_EN.get(lbl["class"], lbl["class"])
        ax.annotate("{}:{}".format(i, name), (lbl["x"], lbl["y"]),
                    fontsize=7, ha="center", va="bottom",
                    xytext=(0, 10), textcoords="offset points",
                    fontweight="bold")

    # 示例路径 (绿色粗线)
    if path:
        px, py = zip(*path)
        ax.plot(px, py, "-", color="#27AE60", lw=3.0, alpha=0.9, zorder=4)
        ax.scatter(px[0], py[0], c="#27AE60", s=120, marker="s",
                   edgecolors="black", lw=1.5, zorder=6, label="Start")
        ax.scatter(px[-1], py[-1], c="#E74C3C", s=120, marker="X",
                   edgecolors="black", lw=1.5, zorder=6, label="Goal")

    # 图例
    leg = [
        Line2D([0],[0],color="#27AE60",lw=3.0,label="A* Path"),
        Line2D([0],[0],marker="s",color="w",markerfacecolor="#27AE60",
               markersize=10,label="Start (Box #6)"),
        Line2D([0],[0],marker="X",color="w",markerfacecolor="#E74C3C",
               markersize=10,label="Goal (FTS #14)"),
    ]
    for cls, color in CLASS_COLORS.items():
        leg.append(Line2D([0],[0],marker="o",color="w",markerfacecolor=color,
                          markersize=10,label=CLASS_EN.get(cls, cls)))
    ax.legend(handles=leg, loc="upper right", framealpha=0.9, fontsize=8)

    ax.set_title("Dynamic Path Planning (no road network) | A* on occupancy grid | Box({}) -> FTS({})".format(
        src_idx, dst_idx), fontsize=14, fontweight="bold")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.12)
    plt.tight_layout()
    fig.savefig("output/dynamic_path_planning.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    print("[OK] output/dynamic_path_planning.png")

    # 统计: 所有136对的规划时间
    print("\nBenchmark: all-pairs planning...")
    times = []
    ok = 0
    for i in range(min(n_tp, 10)):
        for j in range(i+1, min(n_tp, 10)):
            t = time.time()
            p = grid.plan(labels[i]["x"], labels[i]["y"],
                          labels[j]["x"], labels[j]["y"])
            elapsed = time.time() - t
            times.append(elapsed)
            if p: ok += 1
    print("  {} paths, avg {:.0f}ms each".format(ok, np.mean(times)*1000))
    print("  Total time: {:.1f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
