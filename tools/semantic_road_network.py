#!/usr/bin/env python
"""语义路网生成 — 按语义层级连接物体: Box -> Transport -> FTS"""
import numpy as np, networkx as nx
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import combinations

SEMANTIC_LAYERS = [
    {"name": "1. Storage (Box)",        "classes": ["箱子"],                "color": "#FFB400"},
    {"name": "2. Transport (ELF/CB)",   "classes": ["电动运输车","货运自行车"], "color": "#32B4FF"},
    {"name": "3. Delivery (FTS)",       "classes": ["无人搬运车"],           "color": "#32FF64"},
]
CLASS_COLORS = {"箱子":"#FFB400","电动运输车":"#E74C3C","货运自行车":"#3498DB","无人搬运车":"#32FF64"}

# ---- 加载 ----
labels = []
with open("data/warehouse_stitched/labels/000000.txt") as f:
    for line in f:
        p = line.strip().split()
        if len(p)>=8:
            labels.append({"x":float(p[0]),"y":float(p[1]),"class":p[7]})
n_tp = len(labels)
pts_arr = np.array([[l["x"],l["y"]] for l in labels])

# ---- 分组 ----
layers = {}
for li, ld in enumerate(SEMANTIC_LAYERS):
    idxs = [i for i,l in enumerate(labels) if l["class"] in ld["classes"]]
    layers[li] = idxs
    names = ", ".join(["{}:{}".format(i, labels[i]["class"]) for i in idxs])
    print("  {}: {}obj [{}]".format(ld["name"], len(idxs), names))

# ---- 构建语义路网 ----
G_sem = nx.Graph()
for i in range(n_tp):
    G_sem.add_node(i, pos=(pts_arr[i,0], pts_arr[i,1]), cls=labels[i]["class"])
for li, idxs in layers.items():
    for i in idxs:
        G_sem.nodes[i]["layer"] = li

edge_info = []  # (i, j, type, dist)

# Rule 1: 同层同类 intra-class
for li, idxs in layers.items():
    cgroups = {}
    for i in idxs:
        cls = labels[i]["class"]
        cgroups.setdefault(cls, []).append(i)
    for cls, g in cgroups.items():
        if len(g) <= 4:
            for a,b in combinations(g, 2):
                d = np.hypot(pts_arr[a,0]-pts_arr[b,0], pts_arr[a,1]-pts_arr[b,1])
                G_sem.add_edge(a,b,weight=d,etype="intra")
                edge_info.append((a,b,"intra",d))
        else:
            # Box has 8 -> use MST
            Gc = nx.Graph()
            for a,b in combinations(g,2):
                d = np.hypot(pts_arr[a,0]-pts_arr[b,0], pts_arr[a,1]-pts_arr[b,1])
                Gc.add_edge(a,b,weight=d)
            mst_c = nx.minimum_spanning_tree(Gc)
            for u,v in mst_c.edges():
                d = mst_c[u][v]["weight"]
                G_sem.add_edge(u,v,weight=d,etype="intra")
                edge_info.append((u,v,"intra",d))

# Rule 2: 跨层 nearest-neighbor (每个节点连下层最近节点)
for li in range(len(SEMANTIC_LAYERS)-1):
    src = layers[li]; dst = layers[li+1]
    if not src or not dst: continue
    for si in src:
        best = min(dst, key=lambda j: np.hypot(pts_arr[si,0]-pts_arr[j,0], pts_arr[si,1]-pts_arr[j,1]))
        d = np.hypot(pts_arr[si,0]-pts_arr[best,0], pts_arr[si,1]-pts_arr[best,1])
        G_sem.add_edge(si, best, weight=d, etype="down")
        edge_info.append((si, best, "down", d))
    for di in dst:
        best = min(src, key=lambda j: np.hypot(pts_arr[di,0]-pts_arr[j,0], pts_arr[di,1]-pts_arr[j,1]))
        d = np.hypot(pts_arr[di,0]-pts_arr[best,0], pts_arr[di,1]-pts_arr[best,1])
        if not G_sem.has_edge(di, best):
            G_sem.add_edge(di, best, weight=d, etype="up")
            edge_info.append((di, best, "up", d))

# ---- 统计 ----
intra_n = sum(1 for _,_,t,_ in edge_info if t=="intra")
cross_dn = sum(1 for _,_,t,_ in edge_info if t=="down")
cross_up = sum(1 for _,_,t,_ in edge_info if t=="up")
total_len = sum(d for _,_,_,d in edge_info)

print("\nNetwork: {} edges, {:.0f}m".format(G_sem.number_of_edges(), total_len))
print("  Intra-class: {} edges".format(intra_n))
print("  Downstream:  {} edges (Box->Transport->FTS)".format(cross_dn))
print("  Upstream:    {} edges (feedback)".format(cross_up))

# ---- 可视化 ----
print("\nGenerating visualization...")
fig, ax = plt.subplots(figsize=(16, 13), facecolor="white")

# 物体散点
for i, lbl in enumerate(labels):
    c = CLASS_COLORS.get(lbl["class"], "#333")
    ax.scatter(pts_arr[i,0], pts_arr[i,1], c=c, s=220, edgecolors="black", lw=1.5, zorder=5)

# 边: intra 灰色细线, down 绿色粗线, up 橙色虚线
for i,j,t,d in edge_info:
    if t == "intra":
        ax.plot([pts_arr[i,0],pts_arr[j,0]], [pts_arr[i,1],pts_arr[j,1]],
                "-", color="#AAAAAA", lw=0.8, alpha=0.45, zorder=2)
    elif t == "down":
        ax.plot([pts_arr[i,0],pts_arr[j,0]], [pts_arr[i,1],pts_arr[j,1]],
                "-", color="#27AE60", lw=2.5, alpha=0.85, zorder=3)
    else:
        ax.plot([pts_arr[i,0],pts_arr[j,0]], [pts_arr[i,1],pts_arr[j,1]],
                "--", color="#E67E22", lw=1.5, alpha=0.55, zorder=2)

# 节点编号
for i, lbl in enumerate(labels):
    c = CLASS_COLORS.get(lbl["class"], "#333")
    ax.annotate(str(i), (pts_arr[i,0], pts_arr[i,1]),
                fontsize=8, ha="center", va="center", fontweight="bold", color="white",
                bbox=dict(boxstyle="circle,pad=0.3", facecolor=c, edgecolor="black", lw=1.2))

# 层级色带 + 标注
for li, ld in enumerate(SEMANTIC_LAYERS):
    idxs = layers[li]
    if not idxs: continue
    yc = np.mean([pts_arr[i,1] for i in idxs])
    ax.axhline(y=yc, color=ld["color"], lw=6, alpha=0.12, zorder=0)
    ax.text(pts_arr[:,0].min()-3, yc, ld["name"], fontsize=12, fontweight="bold",
            color=ld["color"], ha="right", va="center",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

# 图例
leg = [
    Line2D([0],[0],color="#AAAAAA",lw=0.8,label="Intra-class (same type mesh)"),
    Line2D([0],[0],color="#27AE60",lw=2.5,label="Downstream (Box->Transport->FTS)"),
    Line2D([0],[0],color="#E67E22",lw=1.5,ls="--",label="Upstream (feedback)"),
]
for cls,color in CLASS_COLORS.items():
    leg.append(Line2D([0],[0],marker="o",color="w",markerfacecolor=color,markersize=10,label=cls))
ax.legend(handles=leg, loc="upper right", framealpha=0.9, fontsize=9, ncol=2)

# 标签设置
ax.set_xlabel("X (m)", fontsize=12)
ax.set_ylabel("Y (m)", fontsize=12)
ax.set_aspect("equal")
ax.grid(True, alpha=0.12)

# 统计表
stats = (
    "Semantic Road Network\n"
    + "="*42 + "\n"
    + "Layer 1  Box (storage):       {} nodes\n".format(len(layers[0]))
    + "Layer 2  Transport (ELF+CB):  {} nodes\n".format(len(layers[1]))
    + "Layer 3  FTS (delivery):      {} nodes\n".format(len(layers[2]))
    + "="*42 + "\n"
    + "Intra-class edges:     {:3d}  ({:.0f}m)\n".format(intra_n, sum(d for _,_,t,d in edge_info if t=="intra"))
    + "Downstream (green):    {:3d}  ({:.0f}m)\n".format(cross_dn, sum(d for _,_,t,d in edge_info if t=="down"))
    + "Upstream (orange):     {:3d}  ({:.0f}m)\n".format(cross_up, sum(d for _,_,t,d in edge_info if t=="up"))
    + "="*42 + "\n"
    + "Total:                 {:3d} edges, {:.0f}m\n".format(G_sem.number_of_edges(), total_len)
    + "="*42 + "\n"
    + "Flow: Box --[pickup]--> Transport\n"
    + "      Transport --[deliver]--> FTS"
)
ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=9.5, va="top",
        family="monospace", bbox=dict(boxstyle="round", facecolor="#FAFAFA", alpha=0.9))

ax.set_title("Semantic Road Network | Box -> Transport(ELF/CargoBike) -> FTS | Hierarchical + Nearest-Neighbor",
             fontsize=15, fontweight="bold")
plt.tight_layout()
fig.savefig("output/road_network_semantic.png", dpi=200, bbox_inches="tight", facecolor="white")
print("[OK] output/road_network_semantic.png")

# 导出 JSON
import json
export = {
    "nodes": [{"id": i, "x": float(pts_arr[i,0]), "y": float(pts_arr[i,1]),
               "class": labels[i]["class"], "layer": G_sem.nodes[i]["layer"]}
              for i in range(n_tp)],
    "edges": [{"from": int(i), "to": int(j), "type": t, "distance": float(d)}
              for i,j,t,d in edge_info],
    "layers": [{"name": ld["name"], "classes": ld["classes"]} for ld in SEMANTIC_LAYERS],
}
with open("output/road_network_semantic.json", "w") as f:
    json.dump(export, f, indent=2, ensure_ascii=False)
print("[OK] output/road_network_semantic.json")
