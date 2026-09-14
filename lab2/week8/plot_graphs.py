#!/usr/bin/env python3
"""
plot_graphs.py -- presentation graphs 1-7 from the CAAS results.csv.

Value rule: every plotted time is the MINIMUM over that configuration's
repeats, and every speedup is (fastest serial run at that n) / (fastest run).
CAAS nodes are virtual machines on shared hosts; interference from other
tenants can only ADD time, so the fastest repeat is the best estimate of what
the code itself costs. Re-runs on quiet nodes reproduced the minimum, never the
slow repeats. Error bars show the full range down to the slowest repeat.

Schemes plotted: cyclic for MPI and hybrid (the best-balanced scheme),
dynamic for OpenMP, cyclic for pthreads.

Usage: python plot_graphs.py [results.csv] [output_dir]
Writes graph1.png ... graph7.png and graph_data.csv (the table view of every
plotted point).
"""

import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---- Palette: validated categorical slots 1-4, fixed per implementation ----
SURFACE = "#fcfcfb"
INK     = "#0b0b0b"
INK2    = "#52514e"
MUTED   = "#898781"
GRID    = "#e1e0d9"
AXIS    = "#c3c2b7"

STYLE = {
    "hybrid":   dict(color="#2a78d6", marker="o", label="Hybrid MPI+OpenMP"),
    "mpi":      dict(color="#eb6834", marker="s", label="Open MPI"),
    "openmp":   dict(color="#1baf7a", marker="^", label="OpenMP"),
    "pthreads": dict(color="#eda100", marker="D", label="POSIX Threads"),
    "serial":   dict(color=MUTED,     marker="o", label="Serial"),
}

SCHEME = {"hybrid": "cyclic", "mpi": "cyclic", "openmp": "dynamic",
          "pthreads": "cyclic", "serial": "none"}

N_BIG = 100_000_000

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 11,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "grid.linestyle": "-",
    "legend.frameon": False,
    "legend.labelcolor": INK2,
    "legend.handlelength": 3.0,
})

TABLE = []   # rows for graph_data.csv


# ---- Data ------------------------------------------------------------------

def load(path):
    runs = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r["impl"], r["scheme"], int(r["n"]), int(r["procs"]),
                   int(r["threads"]), int(r["nodes"]))
            runs[key].append((float(r["total_s"]), float(r["serial_s"])))
    return runs


RUNS = load(sys.argv[1] if len(sys.argv) > 1 else "results.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else "graphs"
os.makedirs(OUT, exist_ok=True)


def fastest(impl, n, p, t, nodes=1):
    """(min total, max total, serial-part seconds of the fastest run)."""
    rs = RUNS.get((impl, SCHEME[impl], n, p, t, nodes), [])
    if not rs:
        return None
    best = min(rs, key=lambda x: x[0])
    return best[0], max(x[0] for x in rs), best[1]


def base(n):
    s = fastest("serial", n, 1, 1)
    return s[0] if s else None


def speedup(impl, n, p, t, nodes=1):
    """(speedup from fastest run, speedup of slowest run) or None."""
    b, s = base(n), fastest(impl, n, p, t, nodes)
    if not b or not s:
        return None
    return b / s[0], b / s[1]


def record(graph, series, x, value, impl, n, p, t, nodes=1):
    s = fastest(impl, n, p, t, nodes)
    TABLE.append([graph, series, x, f"{value:.4f}", n, p, t, nodes,
                  f"{s[0]:.4f}" if s else "", f"{s[1]:.4f}" if s else ""])


def amdahl(f, p):
    return 1.0 / (f + (1.0 - f) / p)


def log2_grid(lo, hi, steps_per_octave=8):
    """Dense points between lo and hi, evenly spaced on a log2 axis, so that
    curves such as y = x are drawn as curves rather than straight chords."""
    pts, x = [], float(lo)
    while x < hi:
        pts.append(x)
        x *= 2 ** (1.0 / steps_per_octave)
    return pts + [float(hi)]


# ---- Drawing helpers -------------------------------------------------------

def new_fig(title, subtitle, ncols=1, width=9.0, height=5.2, sharey=False):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), sharey=sharey,
                             squeeze=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=15, color=INK,
                 fontweight="semibold", y=0.985)
    fig.text(0.01, 0.915, subtitle, ha="left", fontsize=10, color=INK2)
    return fig, axes[0]


def line(ax, xs, ys, impl, label=None, lows=None, dashed=False, hollow=False):
    st = STYLE[impl]
    kw = dict(color=st["color"], marker=st["marker"], markersize=7,
              linewidth=2, label=label or st["label"], zorder=3)
    if dashed:
        kw.update(linestyle=(0, (5, 3)), marker=None)
    if hollow:
        kw.update(markerfacecolor=SURFACE, markeredgecolor=st["color"],
                  linestyle="none", markersize=9, markeredgewidth=2)
    ax.plot(xs, ys, **kw)
    if lows is not None:
        err = [[y - lo for y, lo in zip(ys, lows)], [0] * len(ys)]
        ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor=st["color"],
                    elinewidth=1, capsize=3, alpha=0.6, zorder=2)


def ideal(ax, lo, hi, scale=1, label="Ideal (linear)"):
    xs = log2_grid(lo, hi)
    ax.plot(xs, [scale * x for x in xs], color=AXIS, linewidth=1.5, zorder=1,
            label=label)


def log2_x(ax, ticks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()


def corner_note(ax, text):
    """A label in the empty lower-right of the plot, clear of every line."""
    ax.text(0.98, 0.06, text, transform=ax.transAxes, ha="right", va="bottom",
            color=INK2, fontsize=9.5)


def save(fig, name, note):
    fig.text(0.01, 0.012, note, ha="left", va="bottom", fontsize=8.5,
             color=MUTED, linespacing=1.4)
    fig.tight_layout(rect=(0, 0.075 if "\n" in note else 0.045, 1, 0.9))
    fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


RULE = ("Each point: fastest repeat; bar down to slowest repeat. Speedup = fastest "
        "serial run / fastest run. CAAS, AMD EPYC 7763, 16 vCPU per node.")


# ---- Graph 1: runtime vs n --------------------------------------------------

ns = sorted({k[2] for k in RUNS if k[0] == "openmp" and k[4] == 16}
            & {k[2] for k in RUNS if k[0] == "mpi" and k[3] == 16 and k[5] == 1})
xs = [n / 1e6 for n in ns]

fig, (ax,) = new_fig("Graph 1 - Runtime vs problem size",
                     f"Serial vs Open MPI (16 processes) vs OpenMP (16 threads), "
                     f"one node, {len(ns)} values of n")
for impl, p, t in (("serial", 1, 1), ("openmp", 1, 16), ("mpi", 16, 1)):
    ys = [fastest(impl, n, p, t)[0] for n in ns]
    line(ax, xs, ys, impl)
    for n, x, y in zip(ns, xs, ys):
        record(1, impl, x, y, impl, n, p, t)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlabel("n (millions)")
ax.set_ylabel("Wall-clock time incl. file write (s, log scale)")
ax.legend(loc="upper left", fontsize=9.5)
save(fig, "graph1.png", RULE)


# ---- Graph 2: speedup vs n --------------------------------------------------

fig, (ax,) = new_fig("Graph 2 - Speedup vs problem size",
                     "Open MPI (16 processes) vs OpenMP (16 threads), one node, "
                     "against the serial version")
ax.plot([xs[0], xs[-1]], [16, 16], color=AXIS, linewidth=1.5, zorder=1,
        label="Ideal (16x)")
for impl, p, t in (("openmp", 1, 16), ("mpi", 16, 1)):
    sp = [speedup(impl, n, p, t) for n in ns]
    line(ax, xs, [s[0] for s in sp], impl, lows=[s[1] for s in sp])
    for n, x, s in zip(ns, xs, sp):
        record(2, impl, x, s[0], impl, n, p, t)
ax.set_ylim(0, 17.5)
ax.set_xlabel("n (millions)")
ax.set_ylabel("Speedup vs serial")
ax.legend(loc="lower right", fontsize=9.5)
save(fig, "graph2.png", RULE)


# ---- Speedup + efficiency pair, used by graphs 3 and 5 ------------------------

def speedup_and_efficiency(graph, title, subtitle, series, counts, extra, xlabel,
                           note):
    """series: [(impl, count -> (procs, threads))]; extra: optional
    (impl, procs, threads, nodes, label) point on two nodes."""
    fig, (ax_s, ax_e) = new_fig(title, subtitle, ncols=2, width=12.5, height=5.2)
    top = 32 if extra else counts[-1]
    ideal(ax_s, counts[0], top)
    ax_e.axhline(100, color=AXIS, linewidth=1.5, zorder=1, label="Ideal (100%)")

    for impl, pt in series:
        pts = [(c, speedup(impl, N_BIG, *pt(c))) for c in counts]
        pts = [(c, s) for c, s in pts if s]
        cs = [c for c, _ in pts]
        line(ax_s, cs, [s[0] for _, s in pts], impl, lows=[s[1] for _, s in pts])
        line(ax_e, cs, [100 * s[0] / c for c, s in pts], impl)
        for c, s in pts:
            record(graph, impl, c, s[0], impl, N_BIG, *pt(c))

    if extra:
        impl, p, t, nodes, label = extra
        s = speedup(impl, N_BIG, p, t, nodes)
        if s:
            line(ax_s, [32], [s[0]], impl, label=label, hollow=True)
            line(ax_e, [32], [100 * s[0] / 32], impl, label=label, hollow=True)
            corner_note(ax_s, f"{label}: {s[0]:.1f}x")
            record(graph, f"{impl}-2node", 32, s[0], impl, N_BIG, p, t, nodes)

    ticks = counts + ([32] if extra else [])
    for ax in (ax_s, ax_e):
        log2_x(ax, ticks)
        ax.set_xlabel(xlabel)
    ax_s.set_title("Speedup", color=INK, fontsize=11, loc="left")
    ax_e.set_title("Efficiency = speedup / cores (differences are easier to see here)",
                   color=INK, fontsize=11, loc="left")
    ax_s.set_ylabel("Speedup vs serial")
    ax_e.set_ylabel("Efficiency (%)")
    ax_e.set_ylim(60, 105)
    ax_s.legend(loc="upper left", fontsize=9)
    ax_e.legend(loc="lower left", fontsize=9)
    save(fig, f"graph{graph}.png", note)


# ---- Graph 3: speedup vs processes / threads at n = 1e8 ----------------------

speedup_and_efficiency(
    3, "Graph 3 - Speedup vs number of processes / threads",
    "n = 100,000,000. Open MPI processes vs POSIX Threads and OpenMP threads "
    "(same count)",
    [("pthreads", lambda c: (1, c)), ("openmp", lambda c: (1, c)),
     ("mpi", lambda c: (c, 1))],
    [1, 2, 4, 8, 16],
    ("mpi", 32, 1, 2, "Open MPI, 32 processes on 2 nodes"),
    "Processes (MPI) or threads (POSIX / OpenMP)",
    RULE + "\nThreads cannot cross a node, so only MPI extends past 16 cores.")


# ---- Graph 4: hybrid vs MPI, increasing threads at fixed processes ------------

fig, axes = new_fig("Graph 4 - Hybrid vs Open MPI: adding threads to a fixed "
                    "number of processes",
                    "n = 100,000,000, one node. Threads per process = 1 is the "
                    "pure Open MPI run with the same processes",
                    ncols=3, width=12.5, height=5.0, sharey=True)
for ax, P in zip(axes, (1, 2, 4)):
    T = [t for t in (1, 2, 4, 8, 16) if P * t <= 16]
    mpi = speedup("mpi", N_BIG, P, 1)
    pts = [(1, mpi)] + [(t, speedup("hybrid", N_BIG, P, t)) for t in T[1:]]
    ideal(ax, T[0], T[-1], scale=P)
    ax.axhline(mpi[0], color=STYLE["mpi"]["color"], linewidth=1.5, zorder=2,
               label=f"Open MPI, {P} process{'es' if P > 1 else ''}")
    line(ax, [t for t, _ in pts], [s[0] for _, s in pts], "hybrid",
         label="Hybrid", lows=[s[1] for _, s in pts])
    for t, s in pts:
        record(4, f"hybrid-P{P}", t, s[0],
               "mpi" if t == 1 else "hybrid", N_BIG, P, t)
    log2_x(ax, T)
    ax.set_title(f"{P} MPI process{'es' if P > 1 else ''}", color=INK,
                 fontsize=11, loc="left")
    ax.set_xlabel("OpenMP threads per process")
    ax.legend(loc="upper left", fontsize=9)
axes[0].set_ylabel("Speedup vs serial")
save(fig, "graph4.png", RULE)


# ---- Graph 5: hybrid vs OpenMP / pthreads at matched total threads -------------

speedup_and_efficiency(
    5, "Graph 5 - Hybrid vs POSIX Threads and OpenMP at the same total threads",
    "n = 100,000,000. Hybrid = (total / 2) MPI processes x 2 OpenMP threads, "
    "e.g. 8 = 4 processes x 2 threads",
    [("pthreads", lambda c: (1, c)), ("openmp", lambda c: (1, c)),
     ("hybrid", lambda c: (c // 2, 2))],
    [2, 4, 8, 16],
    ("hybrid", 16, 2, 2, "Hybrid, 16 x 2 on 2 nodes"),
    "Total threads",
    RULE + "\nA single shared-memory process cannot exceed one node's 16 cores.")


# ---- Graph 6: MPI empirical vs Amdahl -----------------------------------------

p1 = fastest("mpi", N_BIG, 1, 1)
f_mpi = p1[2] / base(N_BIG)
fig, (ax,) = new_fig("Graph 6 - Open MPI: empirical vs theoretical (Amdahl) speedup",
                     f"n = 100,000,000. Serial fraction f = {f_mpi:.4f} "
                     "(broadcast + prefix sum + merge + file write, at 1 process)")
P = [1, 2, 4, 8, 16]
ideal(ax, 1, 32)
grid_p = log2_grid(1, 32)
line(ax, grid_p, [amdahl(f_mpi, p) for p in grid_p], "mpi",
     label="Amdahl's Law (theoretical)", dashed=True)
sp = [speedup("mpi", N_BIG, p, 1) for p in P]
line(ax, P, [s[0] for s in sp], "mpi", label="Open MPI (empirical, 1 node)",
     lows=[s[1] for s in sp])
for p, s in zip(P, sp):
    record(6, "mpi-empirical", p, s[0], "mpi", N_BIG, p, 1)
    TABLE.append([6, "mpi-amdahl", p, f"{amdahl(f_mpi, p):.4f}", N_BIG, p, 1, 1,
                  "", ""])
s32 = speedup("mpi", N_BIG, 32, 1, nodes=2)
if s32:
    line(ax, [32], [s32[0]], "mpi", label="Open MPI (empirical, 32 on 2 nodes)",
         hollow=True)
    corner_note(ax, f"32 processes on 2 nodes: {s32[0]:.1f}x empirical, "
                    f"{amdahl(f_mpi, 32):.1f}x Amdahl")
    record(6, "mpi-empirical-2node", 32, s32[0], "mpi", N_BIG, 32, 1, 2)
    TABLE.append([6, "mpi-amdahl", 32, f"{amdahl(f_mpi, 32):.4f}", N_BIG, 32, 1,
                  2, "", ""])
log2_x(ax, [1, 2, 4, 8, 16, 32])
ax.set_xlabel("MPI processes")
ax.set_ylabel("Speedup vs serial")
ax.legend(loc="upper left", fontsize=9.5)
save(fig, "graph6.png", RULE)


# ---- Graph 7: hybrid empirical vs Amdahl, both directions ----------------------

fig, axes = new_fig("Graph 7 - Hybrid: empirical vs theoretical (Amdahl) speedup",
                    "n = 100,000,000. Left: processes increase at 2 threads each. "
                    "Right: threads increase in 1 process. x = total threads",
                    ncols=2, width=12.5, height=5.2, sharey=True)
for ax, (title, pairs, two_node, side) in zip(axes, (
        ("Increasing MPI processes (2 threads each)",
         [(1, 2), (2, 2), (4, 2), (8, 2)], (16, 2), "procs"),
        ("Increasing OpenMP threads (1 process)",
         [(1, 2), (1, 4), (1, 8), (1, 16)], None, "threads"))):
    f = fastest("hybrid", N_BIG, *pairs[0])[2] / base(N_BIG)
    top = 32 if two_node else 16
    wx = [p * t for p, t in pairs]
    ideal(ax, 2, top)
    grid_w = log2_grid(2, top)
    line(ax, grid_w, [amdahl(f, w) for w in grid_w], "hybrid",
         label=f"Amdahl's Law, f = {f:.4f}", dashed=True)
    sp = [speedup("hybrid", N_BIG, p, t) for p, t in pairs]
    line(ax, wx, [s[0] for s in sp], "hybrid", label="Hybrid (empirical, 1 node)",
         lows=[s[1] for s in sp])
    for (p, t), w, s in zip(pairs, wx, sp):
        record(7, f"hybrid-{side}-empirical", w, s[0], "hybrid", N_BIG, p, t)
        TABLE.append([7, f"hybrid-{side}-amdahl", w, f"{amdahl(f, w):.4f}",
                      N_BIG, p, t, 1, "", ""])
    if two_node:
        s2 = speedup("hybrid", N_BIG, *two_node, nodes=2)
        if s2:
            line(ax, [32], [s2[0]], "hybrid",
                 label="Hybrid (empirical, 16 x 2 on 2 nodes)", hollow=True)
            corner_note(ax, f"16 x 2 on 2 nodes: {s2[0]:.1f}x empirical, "
                            f"{amdahl(f, 32):.1f}x Amdahl")
            record(7, "hybrid-procs-empirical-2node", 32, s2[0], "hybrid",
                   N_BIG, 16, 2, 2)
            TABLE.append([7, "hybrid-procs-amdahl", 32, f"{amdahl(f, 32):.4f}",
                          N_BIG, 16, 2, 2, "", ""])
    log2_x(ax, [2, 4, 8, 16, 32] if two_node else [2, 4, 8, 16])
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel("Total threads (processes x threads)")
    ax.legend(loc="upper left", fontsize=9)
axes[0].set_ylabel("Speedup vs serial")
save(fig, "graph7.png", RULE)


# ---- Table view ------------------------------------------------------------------

with open(os.path.join(OUT, "graph_data.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["graph", "series", "x", "value", "n", "procs", "threads",
                "nodes", "fastest_s", "slowest_s"])
    w.writerows(TABLE)
print("wrote", os.path.join(OUT, "graph_data.csv"))
