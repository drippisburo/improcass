"""Figures for the report from results.csv (run pooledpreproc.exe first).
Usage:  python plot_results.py            (reads ./results.csv, writes fig*.png next to it)
"""
import csv
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

CSV = sys.argv[1] if len(sys.argv) > 1 else "results.csv"

# ---- read ----
rows = []
with open(CSV, newline="") as f:
    for r in csv.DictReader(f):
        rows.append({k: float(v) for k, v in r.items()})

def pick(lda, cls, d=None, k=None):
    out = [r for r in rows if r["LDAPCA"] == lda and r["classifier"] == cls]
    if d is not None: out = [r for r in out if r["d"] == d]
    if k is not None: out = [r for r in out if r["k"] == k]
    return out

# ---- style ----
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed categorical order
MARKERS = ["o", "s", "^", "D"]
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

def new_axes(title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, axis="y"); ax.set_axisbelow(True)
    return fig, ax

def line(ax, xs, ys, i, label, best=True):
    ax.plot(xs, ys, color=COLORS[i], marker=MARKERS[i], markersize=5, linewidth=2,
            markeredgecolor="white", markeredgewidth=0.8, label=label)
    # direct label at the right end, in ink not series colour
    ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8, color=INK2)
    if best:   # ring the best point of the series
        j = max(range(len(ys)), key=lambda t: ys[t])
        ax.plot(xs[j], ys[j], marker="o", markersize=11, markerfacecolor="none",
                markeredgecolor=COLORS[i], markeredgewidth=1.2)

def log_d_axis(ax, ds):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ds)); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(ds); ax.set_xticklabels([str(int(d)) for d in ds])
    ax.set_xlim(ds[0] * 0.85, ds[-1] * 2.2)       # room for the direct labels

K0 = 5          # k used in the d sweep
D0 = 40         # d used in the k sweep
METHODS = [(0, 0, "PCA + Mahalanobis"), (0, 1, "PCA + k-NN"),
           (1, 0, "PCA->LDA + Mahalanobis"), (1, 1, "PCA->LDA + k-NN")]

# ---- fig 1: BA vs d, four methods ----
fig, ax = new_axes("Balanced accuracy vs. PCA dimensions (5-fold, k = %d)" % K0,
                   "PCA dimensions kept, d (log scale)", "Balanced accuracy (%)")
all_d = sorted({r["d"] for r in pick(0, 0, k=K0)})
for i, (lda, cls, name) in enumerate(METHODS):
    rs = sorted(pick(lda, cls, k=K0), key=lambda r: r["d"])
    line(ax, [r["d"] for r in rs], [r["BA"] for r in rs], i, name)
log_d_axis(ax, all_d)
ax.axhline(100 / 3, color=MUTED, linewidth=0.8, linestyle=":")
ax.annotate("chance (33.3%)", (all_d[0], 100 / 3), xytext=(0, 4), textcoords="offset points", fontsize=7, color=MUTED)
ax.set_ylim(30, 85); ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.09), fontsize=8)
fig.tight_layout(); fig.savefig("fig1_ba_vs_d.png"); plt.close(fig)

# ---- fig 2: per-class recall vs d, best method ----
fig, ax = new_axes("Per-class recall vs. d  —  PCA + Mahalanobis",
                   "PCA dimensions kept, d (log scale)", "Recall (%)")
rs = sorted(pick(0, 0, k=K0), key=lambda r: r["d"])
ds = [r["d"] for r in rs]
for i, (col, name) in enumerate([("recall_clutter", "clutter"), ("recall_person", "person"), ("recall_cluster", "cluster")]):
    line(ax, ds, [r[col] for r in rs], i, name, best=False)
log_d_axis(ax, ds); ax.set_ylim(0, 100); ax.legend(loc="lower left", fontsize=8)
fig.tight_layout(); fig.savefig("fig2_recall_vs_d.png"); plt.close(fig)

# ---- fig 3: cluster recall, Mahalanobis vs k-NN (PCA only) ----
fig, ax = new_axes("Cluster recall vs. d  —  PCA, Mahalanobis vs. k-NN",
                   "PCA dimensions kept, d (log scale)", "Cluster recall (%)")
for i, (cls, name) in enumerate([(0, "Mahalanobis"), (1, "k-NN (k = %d)" % K0)]):
    rs = sorted(pick(0, cls, k=K0), key=lambda r: r["d"])
    line(ax, [r["d"] for r in rs], [r["recall_cluster"] for r in rs], i, name, best=False)
log_d_axis(ax, all_d); ax.set_ylim(0, 100); ax.legend(loc="lower left", fontsize=8)
fig.tight_layout(); fig.savefig("fig3_cluster_recall.png"); plt.close(fig)

# ---- fig 4: BA vs k ----
fig, ax = new_axes("Balanced accuracy vs. k  —  k-NN, d = %d" % D0, "k (neighbours)", "Balanced accuracy (%)")
for i, (lda, name) in enumerate([(0, "PCA + k-NN"), (1, "PCA->LDA + k-NN")]):
    rs = sorted(pick(lda, 1, d=D0), key=lambda r: r["k"])
    ks = [r["k"] for r in rs]
    line(ax, ks, [r["BA"] for r in rs], i, name, best=False)
ax.set_xticks(ks); ax.set_xlim(0, ks[-1] * 1.35); ax.set_ylim(50, 80); ax.legend(loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig("fig4_ba_vs_k.png"); plt.close(fig)

# ---- table: best setting per method ----
print("%-26s %4s %3s %6s   %7s %6s %7s" % ("method", "d", "k", "BA", "clutter", "person", "cluster"))
for lda, cls, name in METHODS:
    b = max(pick(lda, cls), key=lambda r: r["BA"])
    print("%-26s %4d %3d %6.1f   %7.1f %6.1f %7.1f" % (name, b["d"], b["k"], b["BA"], b["recall_clutter"], b["recall_person"], b["recall_cluster"]))
print("wrote fig1_ba_vs_d.png fig2_recall_vs_d.png fig3_cluster_recall.png fig4_ba_vs_k.png")
