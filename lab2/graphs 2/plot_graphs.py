#!/usr/bin/env python3
"""
plot_graphs.py -- presentation graphs from the CAAS results.csv.

Required graphs (lab spec, Task 4):
  graph1  runtime vs n            Open MPI vs OpenMP (16 workers), serial for reference
  graph2  speedup vs n            Open MPI vs OpenMP, plus hybrid 4x4 (Task 2 vs n)
  graph3  speedup vs workers      Open MPI processes vs POSIX / OpenMP threads
  graph4  hybrid vs Open MPI      adding threads to a fixed number of processes
  graph5  hybrid vs threads       same total number of threads
  graph6  Open MPI                empirical vs theoretical (Amdahl, Amdahl + comm)
  graph7  hybrid                  empirical vs theoretical, processes and threads

Supplementary graphs (appendix / discussion):
  graphS1 workload distribution   block vs weighted vs cyclic: speedup, imbalance
  graphS2 theoretical vs n        Amdahl with measured fractions at every n
  graphS3 oversubscription        16 vs 32 workers on one 16-core node

Value rule
  Every plotted time is the FASTEST repeat of that configuration, and every
  speedup is (fastest serial run at that n) / (fastest run). CAAS nodes are
  virtual machines on shared hosts: interference from other users can only
  ADD time, so the fastest repeat is the best estimate of the code's own cost.
  Error bars run down to the slowest repeat, so the noise stays visible.

Theoretical speedup (Amdahl's Law with a communication term, as in the
Week 7 extra class). All fractions are of the SERIAL program's runtime T_s:
  r_s = output (file write) time / T_s          (never parallelised; median
                                                over every run at that n)
  r_p = 1 - r_s                                (the prime search)
  k(p) = communication overhead measured in the parallel run at p / T_s
  S(p) = 1 / (r_s + r_p / p + k(p))
k(p) is measured, not assumed, so it grows with p and with the network.

Usage: python plot_graphs.py [results.csv] [output_dir]
Writes the PNGs and graph_data.csv (every plotted value, as a table).
"""

import csv
import logging
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

# ---- Palette: fixed colour per implementation --------------------------------
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
SCHEME_STYLE = {
    "cyclic":   dict(color="#eb6834", marker="s", label="Cyclic"),
    "weighted": dict(color="#8e5bd0", marker="P", label="Weighted"),
    "block":    dict(color="#6b6a65", marker="X", label="Block"),
}

# Scheme plotted for each implementation in the main graphs.
SCHEME = {"hybrid": "cyclic", "mpi": "cyclic", "openmp": "dynamic",
          "pthreads": "cyclic", "serial": "none"}

N_BIG = 100_000_000
CORES_PER_NODE = 16

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
    "legend.frameon": False,
    "legend.labelcolor": INK2,
    "legend.handlelength": 3.0,
})

TABLE = []   # rows for graph_data.csv


# ---- Data -------------------------------------------------------------------

def load(path):
    runs = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r["impl"], r["scheme"], int(r["n"]), int(r["procs"]),
                   int(r["threads"]), int(r["nodes"]))
            runs[key].append({k: float(r[k]) for k in
                              ("total_s", "serial_s", "parallel_s",
                               "overhead_s", "imbalance_pct", "write_s")})
    return runs


RUNS = load(sys.argv[1] if len(sys.argv) > 1 else "results.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else "graphs"
os.makedirs(OUT, exist_ok=True)


def runs_of(impl, n, p, t, nodes=1, scheme=None):
    return RUNS.get((impl, scheme or SCHEME[impl], n, p, t, nodes), [])


def fastest(impl, n, p, t, nodes=1, scheme=None):
    """The fastest repeat (a dict of its fields), plus the slowest total."""
    rs = runs_of(impl, n, p, t, nodes, scheme)
    if not rs:
        return None, None
    best = min(rs, key=lambda x: x["total_s"])
    return best, max(x["total_s"] for x in rs)


def base(n):
    best, _ = fastest("serial", n, 1, 1)
    return best["total_s"] if best else None


def speedup(impl, n, p, t, nodes=1, scheme=None):
    """(speedup of fastest run, speedup of slowest run) or None."""
    b = base(n)
    best, slow = fastest(impl, n, p, t, nodes, scheme)
    if not b or not best:
        return None
    return b / best["total_s"], b / slow


def serial_fraction(n):
    """r_s: time to output the result, as a fraction of the serial runtime.

    Every version writes the identical file with the identical function, so
    the output cost is measured in ALL runs at this n, and the median is used.
    A single run's write time is too noisy on CAAS's network file system
    (about 0.1-0.5 s), and would make the theory depend on one lucky or
    unlucky write."""
    writes = sorted(r["write_s"] for key, rs in RUNS.items() if key[2] == n
                    for r in rs)
    mid = len(writes) // 2
    median = writes[mid] if len(writes) % 2 else 0.5 * (writes[mid - 1] + writes[mid])
    return median / base(n)


def kappa(impl, n, p, t, nodes=1, scheme=None):
    """Communication / coordination overhead measured in the fastest parallel
    run, as a fraction of the serial runtime."""
    best, _ = fastest(impl, n, p, t, nodes, scheme)
    return best["overhead_s"] / base(n) if best else None


def amdahl(rs, workers, k=0.0):
    return 1.0 / (rs + (1.0 - rs) / workers + k)


def record(graph, series, x, value, n="", p="", t="", nodes="", note=""):
    TABLE.append([graph, series, x, f"{value:.4f}", n, p, t, nodes, note])


def pts_grid(lo, hi, steps=16):
    xs, x = [], float(lo)
    ratio = 2 ** (1.0 / steps)
    while x < hi:
        xs.append(x)
        x *= ratio
    return xs + [float(hi)]


# ---- Drawing helpers ----------------------------------------------------------

def new_fig(title, subtitle, ncols=1, width=9.0, height=5.2, sharey=False,
            nrows=1):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), sharey=sharey,
                             squeeze=False)
    top = 1 - 0.015 * 5.2 / height
    fig.suptitle(title, x=0.01, ha="left", fontsize=15, color=INK,
                 fontweight="semibold", y=top)
    fig.text(0.01, top - 0.07 * 5.2 / height, subtitle, ha="left", fontsize=10,
             color=INK2)
    fig._head = 1 - 0.1 * 5.2 / height
    return fig, (axes[0] if nrows == 1 else axes)


def line(ax, xs, ys, style, label=None, lows=None, highs=None, dashed=False,
         hollow=False, no_marker=False):
    kw = dict(color=style["color"], marker=style["marker"], markersize=7,
              linewidth=2, label=label or style["label"], zorder=3)
    if dashed:
        kw.update(linestyle=(0, (5, 3)), linewidth=1.8)
    if no_marker:
        kw.update(marker=None)
    if hollow:
        kw.update(markerfacecolor=SURFACE, markeredgecolor=style["color"],
                  linestyle="none", markersize=9, markeredgewidth=2)
    ax.plot(xs, ys, **kw)
    if lows is not None or highs is not None:
        lo = [y - l for y, l in zip(ys, lows)] if lows is not None else [0] * len(ys)
        hi = [h - y for y, h in zip(ys, highs)] if highs is not None else [0] * len(ys)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=style["color"],
                    elinewidth=1, capsize=3, alpha=0.55, zorder=2)


def ideal(ax, lo, hi, label="Ideal (linear)"):
    xs = [lo, hi]
    ax.plot(xs, xs, color=AXIS, linewidth=1.5, zorder=1, label=label)


def loglog(ax, ticks, yticks=None):
    """Log2 on both axes: ideal linear speedup is a straight 45-degree line."""
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    for axis, tk in ((ax.xaxis, ticks), (ax.yaxis, yticks or ticks)):
        axis.set_major_locator(FixedLocator(tk))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_locator(NullLocator())


def corner_note(ax, text, y=0.05):
    ax.text(0.98, y, text, transform=ax.transAxes, ha="right", va="bottom",
            color=INK2, fontsize=9.5)


def save(fig, name, note):
    fig.text(0.01, 0.012, note, ha="left", va="bottom", fontsize=8.5,
             color=MUTED, linespacing=1.4)
    lines = note.count("\n") + 1
    h = fig.get_size_inches()[1]
    fig.tight_layout(rect=(0, (0.1 + 0.16 * lines) / h, 1, fig._head))
    fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


RULE = ("Each point: fastest repeat; bars span to the slowest repeat. Speedup = "
        "fastest serial run / fastest run.\nCAAS: AMD EPYC 7763, 16 cores per node "
        "(2 sockets x 8 cores, 1 thread per core). Times include writing the output file.")


# =============================================================================
# Graph 1: runtime vs n
# =============================================================================

NS = sorted({k[2] for k in RUNS if k[0] == "serial"}
            & {k[2] for k in RUNS if k[0] == "mpi" and k[3] == 16 and k[5] == 1
               and k[1] == "cyclic"}
            & {k[2] for k in RUNS if k[0] == "openmp" and k[4] == 16})
XS = [n / 1e6 for n in NS]

fig, (ax,) = new_fig("Graph 1 - Runtime vs problem size",
                     f"Serial vs Open MPI (16 processes) vs OpenMP (16 threads), "
                     f"one node, {len(NS)} values of n")
for impl, p, t in (("serial", 1, 1), ("openmp", 1, 16), ("mpi", 16, 1)):
    data = [fastest(impl, n, p, t) for n in NS]
    ys = [d[0]["total_s"] for d in data]
    line(ax, XS, ys, STYLE[impl], highs=[d[1] for d in data])
    for n, x, y in zip(NS, XS, ys):
        record(1, impl, x, y, n, p, t, 1, "runtime_s")
ax.set_yscale("log")
ax.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 100]))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.yaxis.set_minor_locator(NullLocator())
ax.set_xlabel("n (millions)")
ax.set_ylabel("Wall-clock time (s, log scale)")
ax.legend(loc="upper left", fontsize=9.5)
corner_note(ax, "Serial grows roughly as n^1.4: trial division costs ~sqrt(k) "
                "per prime found", y=0.03)
save(fig, "graph1.png", RULE)


# =============================================================================
# Graph 2: speedup vs n
# =============================================================================

fig, (ax,) = new_fig("Graph 2 - Speedup vs problem size",
                     "16 workers on one node: Open MPI (16 processes), OpenMP "
                     "(16 threads), hybrid (4 processes x 4 threads)")
ax.axhline(16, color=AXIS, linewidth=1.5, zorder=1, label="Ideal (16x)")
for impl, p, t in (("openmp", 1, 16), ("mpi", 16, 1), ("hybrid", 4, 4)):
    sp = [speedup(impl, n, p, t) for n in NS]
    keep = [(x, s, n) for x, s, n in zip(XS, sp, NS) if s]
    line(ax, [x for x, _, _ in keep], [s[0] for _, s, _ in keep], STYLE[impl])
    for x, s, n in keep:
        record(2, impl, x, s[0], n, p, t, 1, f"slowest repeat {s[1]:.3f}")
ax.set_ylim(10, 17)
ax.set_xlabel("n (millions)")
corner_note(ax, "Note: the speedup axis starts at 10", y=0.93)
ax.set_ylabel("Speedup vs serial (axis starts at 10)")
ax.legend(loc="lower right", fontsize=9.5, ncol=2)
save(fig, "graph2.png",
     RULE.replace("; bars span to the slowest repeat", "") +
     "\nError bars omitted here for readability: 30 values of n x 3 series; "
     "the slowest repeats are listed in graph_data.csv.")


# =============================================================================
# Speedup + efficiency pair, used by graphs 3 and 5
# =============================================================================

def speedup_and_efficiency(graph, title, subtitle, series, counts, extras,
                           xlabel, note):
    """series: [(impl, count -> (procs, threads))]
    extras: [(impl, procs, threads, nodes, label)] points on two nodes."""
    fig, (ax_s, ax_e) = new_fig(title, subtitle, ncols=2, width=13, height=5.4)
    top = max([counts[-1]] + [p * t for _, p, t, _, _ in extras])
    ideal(ax_s, counts[0], top)
    ax_e.axhline(100, color=AXIS, linewidth=1.5, zorder=1, label="Ideal (100%)")

    for impl, pt in series:
        pts = [(c, speedup(impl, N_BIG, *pt(c))) for c in counts]
        pts = [(c, s) for c, s in pts if s]
        cs = [c for c, _ in pts]
        line(ax_s, cs, [s[0] for _, s in pts], STYLE[impl],
             lows=[s[1] for _, s in pts])
        line(ax_e, cs, [100 * s[0] / c for c, s in pts], STYLE[impl])
        for c, s in pts:
            record(graph, impl, c, s[0], N_BIG, *pt(c), 1, "speedup")

    notes = []
    for impl, p, t, nodes, label in extras:
        s = speedup(impl, N_BIG, p, t, nodes)
        if s:
            w = p * t
            line(ax_s, [w], [s[0]], STYLE[impl], label=label, hollow=True)
            line(ax_e, [w], [100 * s[0] / w], STYLE[impl], label=label, hollow=True)
            notes.append(f"{label}: {s[0]:.1f}x")
            record(graph, f"{impl}-2node", w, s[0], N_BIG, p, t, nodes, "speedup")
    if notes:
        corner_note(ax_s, "\n".join(notes))

    ticks = sorted(set(counts + [p * t for _, p, t, _, _ in extras]))
    loglog(ax_s, ticks)
    ax_e.set_xscale("log", base=2)
    ax_e.xaxis.set_major_locator(FixedLocator(ticks))
    ax_e.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_e.xaxis.set_minor_locator(NullLocator())
    for ax in (ax_s, ax_e):
        ax.set_xlabel(xlabel)
    ax_s.set_title("Speedup (log-log: ideal is the straight line)", color=INK,
                   fontsize=11, loc="left")
    ax_e.set_title("Efficiency = speedup / workers", color=INK, fontsize=11,
                   loc="left")
    ax_s.set_ylabel("Speedup vs serial")
    ax_e.set_ylabel("Efficiency (%)")
    ax_e.set_ylim(60, 105)
    ax_s.legend(loc="upper left", fontsize=9)
    ax_e.legend(loc="lower left", fontsize=9)
    save(fig, f"graph{graph}.png", note)


# =============================================================================
# Graph 3: speedup vs processes / threads at n = 1e8
# =============================================================================

speedup_and_efficiency(
    3, "Graph 3 - Speedup vs number of processes / threads",
    "n = 100,000,000. Open MPI processes vs POSIX Threads and OpenMP threads "
    "(same count)",
    [("pthreads", lambda c: (1, c)), ("openmp", lambda c: (1, c)),
     ("mpi", lambda c: (c, 1))],
    [1, 2, 4, 8, 16],
    [("mpi", 32, 1, 2, "Open MPI, 32 processes on 2 nodes")],
    "Processes (MPI) or threads (POSIX / OpenMP)",
    RULE + "\nThreads cannot cross a node, so only MPI extends past 16 cores.")


# =============================================================================
# Graph 4: hybrid vs MPI, increasing threads at fixed processes
# =============================================================================

PANELS = [P for P in (1, 2, 4, 8) if speedup("mpi", N_BIG, P, 1)]
fig, axes = new_fig("Graph 4 - Hybrid vs Open MPI: adding threads to a fixed "
                    "number of processes",
                    "n = 100,000,000, one node, log-log axes. 1 thread per process "
                    "is the pure Open MPI run with the same processes",
                    ncols=len(PANELS), width=3.4 * len(PANELS) + 1, height=5.0,
                    sharey=True)
for ax, P in zip(axes, PANELS):
    T = [t for t in (1, 2, 4, 8, 16) if P * t <= CORES_PER_NODE]
    mpi = speedup("mpi", N_BIG, P, 1)
    pts = [(1, mpi)] + [(t, speedup("hybrid", N_BIG, P, t)) for t in T[1:]]
    pts = [(t, s) for t, s in pts if s]
    ax.plot(T, [P * t for t in T], color=AXIS, linewidth=1.5, zorder=1,
            label="Ideal (linear)")
    ax.axhline(mpi[0], color=STYLE["mpi"]["color"], linewidth=1.2, zorder=2,
               linestyle=(0, (2, 2)),
               label=f"Open MPI, {P} process{'es' if P > 1 else ''}")
    line(ax, [t for t, _ in pts], [s[0] for _, s in pts], STYLE["hybrid"],
         label="Hybrid", lows=[s[1] for _, s in pts])
    for t, s in pts:
        record(4, f"hybrid-P{P}", t, s[0], N_BIG, P, t, 1,
               "mpi (t=1)" if t == 1 else "speedup")
    loglog(ax, T, [1, 2, 4, 8, 16])
    ax.set_title(f"{P} MPI process{'es' if P > 1 else ''}", color=INK,
                 fontsize=11, loc="left")
    ax.set_xlabel("OpenMP threads per process")
    ax.legend(loc="upper left", fontsize=8.5)
axes[0].set_ylabel("Speedup vs serial")
save(fig, "graph4.png", RULE)


# =============================================================================
# Graph 5: hybrid vs OpenMP / pthreads at matched total threads
# =============================================================================

speedup_and_efficiency(
    5, "Graph 5 - Hybrid vs POSIX Threads and OpenMP at the same total threads",
    "n = 100,000,000. Hybrid = (total / 2) MPI processes x 2 OpenMP threads, "
    "e.g. 8 = 4 processes x 2 threads",
    [("pthreads", lambda c: (1, c)), ("openmp", lambda c: (1, c)),
     ("hybrid", lambda c: (c // 2, 2))],
    [2, 4, 8, 16],
    [("hybrid", 16, 2, 2, "Hybrid, 16 x 2 on 2 nodes")],
    "Total threads",
    RULE + "\nA single shared-memory process cannot exceed one node's 16 cores.")


# =============================================================================
# Graphs 6 and 7: empirical vs theoretical
# =============================================================================

RS = serial_fraction(N_BIG)
THEORY_NOTE = (RULE + "\nTheory: S = 1 / (r_s + r_p/p + k).  r_s = file-write time "
               f"/ serial runtime = {RS:.4f} (median write over all runs at this n);  "
               "r_p = 1 - r_s;\nk = communication overhead measured in the "
               "parallel run at each point / serial runtime.  p = total workers.")


def theory_panels(ax_s, ax_e, graph, series_name, impl, configs, ticks):
    """configs: [(workers, procs, threads, nodes)] in increasing workers.
    Top: speedup, log-log.  Bottom: efficiency, where the gaps are visible."""
    top = ticks[-1]
    grid = pts_grid(ticks[0], top)
    ideal(ax_s, ticks[0], top)
    ax_e.axhline(100, color=AXIS, linewidth=1.5, zorder=1, label="Ideal (100%)")
    for ax, fn in ((ax_s, lambda w: amdahl(RS, w)),
                   (ax_e, lambda w: 100 * amdahl(RS, w) / w)):
        line(ax, grid, [fn(w) for w in grid], STYLE[impl],
             label="Amdahl (serial fraction only)", dashed=True, no_marker=True)

    emp, th, two = [], [], []
    for w, p, t, nodes in configs:
        s = speedup(impl, N_BIG, p, t, nodes)
        k = kappa(impl, N_BIG, p, t, nodes)
        if not s:
            continue
        theory = amdahl(RS, w, k)
        record(graph, f"{series_name}-empirical", w, s[0], N_BIG, p, t, nodes,
               f"slowest repeat {s[1]:.3f}")
        record(graph, f"{series_name}-amdahl", w, amdahl(RS, w), N_BIG, p, t,
               nodes, f"r_s={RS:.5f}")
        record(graph, f"{series_name}-amdahl+comm", w, theory, N_BIG, p, t,
               nodes, f"k={k:.5f}")
        th.append((w, theory))
        (emp if nodes == 1 else two).append((w, s))

    for ax, scale in ((ax_s, lambda w, v: v), (ax_e, lambda w, v: 100 * v / w)):
        ax.plot([w for w, _ in th], [scale(w, v) for w, v in th],
                color=STYLE[impl]["color"], linestyle="none", marker="x",
                markersize=9, markeredgewidth=2, zorder=4,
                label="Amdahl + measured communication")
        line(ax, [w for w, _ in emp], [scale(w, s[0]) for w, s in emp],
             STYLE[impl], label="Empirical, 1 node",
             lows=[scale(w, s[1]) for w, s in emp] if ax is ax_s else None)
        for w, s in two:
            line(ax, [w], [scale(w, s[0])], STYLE[impl], label="Empirical, 2 nodes",
                 hollow=True)

    for w, s in two:
        theory = dict(th)[w]
        corner_note(ax_e, f"{w} on 2 nodes: empirical {s[0]:.1f}x, "
                          f"theoretical {theory:.1f}x", y=0.05)
    loglog(ax_s, ticks)
    ax_e.set_xscale("log", base=2)
    ax_e.xaxis.set_major_locator(FixedLocator(ticks))
    ax_e.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_e.xaxis.set_minor_locator(NullLocator())
    ax_e.set_ylim(75, 102)
    ax_s.set_ylabel("Speedup vs serial")
    ax_e.set_ylabel("Efficiency (%)")
    ax_s.legend(loc="upper left", fontsize=8.5)


fig, (ax_s, ax_e) = new_fig("Graph 6 - Open MPI: empirical vs theoretical speedup",
                            "n = 100,000,000, cyclic scheme. Amdahl's Law with and "
                            "without the measured communication term",
                            ncols=2, width=13, height=5.6)
theory_panels(ax_s, ax_e, 6, "mpi", "mpi",
              [(p, p, 1, 1) for p in (1, 2, 4, 8, 16)] + [(32, 32, 1, 2)],
              [1, 2, 4, 8, 16, 32])
ax_s.set_title("Speedup (log-log)", color=INK, fontsize=11, loc="left")
ax_e.set_title("Efficiency: where the differences show", color=INK, fontsize=11,
               loc="left")
for ax in (ax_s, ax_e):
    ax.set_xlabel("MPI processes")
save(fig, "graph6.png", THEORY_NOTE)

fig, axes = new_fig("Graph 7 - Hybrid: empirical vs theoretical speedup",
                    "n = 100,000,000. Left: processes increase at 2 threads each. "
                    "Right: threads increase in 1 process. x = total threads",
                    ncols=2, nrows=2, width=13, height=9.5)
theory_panels(axes[0][0], axes[1][0], 7, "hybrid-procs", "hybrid",
              [(2 * p, p, 2, 1) for p in (1, 2, 4, 8)] + [(32, 16, 2, 2)],
              [2, 4, 8, 16, 32])
theory_panels(axes[0][1], axes[1][1], 7, "hybrid-threads", "hybrid",
              [(t, 1, t, 1) for t in (2, 4, 8, 16)],
              [2, 4, 8, 16])
axes[0][0].set_title("Increasing MPI processes (2 threads each): speedup",
                     color=INK, fontsize=11, loc="left")
axes[0][1].set_title("Increasing OpenMP threads (1 process): speedup",
                     color=INK, fontsize=11, loc="left")
axes[1][0].set_title("Efficiency", color=INK, fontsize=11, loc="left")
axes[1][1].set_title("Efficiency", color=INK, fontsize=11, loc="left")
for ax in axes[1]:
    ax.set_xlabel("Total threads (processes x threads)")
save(fig, "graph7.png", THEORY_NOTE)


# =============================================================================
# Supplementary S1: workload distribution schemes
# =============================================================================

fig, (ax_s, ax_i) = new_fig("Graph S1 - Workload distribution: block vs weighted "
                            "vs cyclic",
                            "Open MPI, n = 100,000,000. Imbalance = (slowest - "
                            "fastest rank search time) / slowest",
                            ncols=2, width=13, height=5.4)
P = [2, 4, 8, 16, 32]
ideal(ax_s, 2, 32)
for scheme in ("cyclic", "weighted", "block"):
    xs, ys, lows, imb = [], [], [], []
    for p in P:
        nodes = 2 if p > CORES_PER_NODE else 1
        s = speedup("mpi", N_BIG, p, 1, nodes, scheme)
        best, _ = fastest("mpi", N_BIG, p, 1, nodes, scheme)
        if not s:
            continue
        xs.append(p)
        ys.append(s[0])
        lows.append(s[1])
        imb.append(best["imbalance_pct"])
        record("S1", f"{scheme}-speedup", p, s[0], N_BIG, p, 1, nodes, "speedup")
        record("S1", f"{scheme}-imbalance", p, best["imbalance_pct"], N_BIG, p, 1,
               nodes, "imbalance_pct")
    line(ax_s, xs, ys, SCHEME_STYLE[scheme], lows=lows)
    line(ax_i, xs, imb, SCHEME_STYLE[scheme])
loglog(ax_s, P)
ax_i.set_xscale("log", base=2)
ax_i.xaxis.set_major_locator(FixedLocator(P))
ax_i.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax_i.xaxis.set_minor_locator(NullLocator())
ax_i.set_ylim(0, 100)
for ax in (ax_s, ax_i):
    ax.set_xlabel("MPI processes (32 = 2 nodes)")
ax_s.set_title("Speedup", color=INK, fontsize=11, loc="left")
ax_i.set_title("Load imbalance between ranks", color=INK, fontsize=11, loc="left")
ax_s.set_ylabel("Speedup vs serial")
ax_i.set_ylabel("Imbalance (%)")
ax_s.legend(loc="upper left", fontsize=9)
ax_i.legend(loc="center right", fontsize=9)
save(fig, "graphS1.png", RULE)


# =============================================================================
# Supplementary S2: theoretical vs empirical speedup as n increases
# =============================================================================

fig, axes = new_fig("Graph S2 - Theoretical vs empirical speedup as n increases",
                    "16 workers, one node. Theory uses r_s and k measured at every "
                    "n; the serial fraction shrinks as the search grows",
                    ncols=2, width=13, height=5.4, sharey=True)
for ax, (impl, p, t, title) in zip(axes, (("mpi", 16, 1, "Open MPI, 16 processes"),
                                          ("hybrid", 4, 4, "Hybrid, 4 x 4"))):
    xs, emp, lows, th_plain, th_comm = [], [], [], [], []
    for n in NS:
        s = speedup(impl, n, p, t)
        k = kappa(impl, n, p, t)
        if not s:
            continue
        rs = serial_fraction(n)
        xs.append(n / 1e6)
        emp.append(s[0])
        lows.append(s[1])
        th_plain.append(amdahl(rs, 16))
        th_comm.append(amdahl(rs, 16, k))
        record("S2", f"{impl}-empirical", n / 1e6, s[0], n, p, t, 1, "speedup")
        record("S2", f"{impl}-amdahl", n / 1e6, th_plain[-1], n, p, t, 1,
               f"r_s={rs:.5f}")
        record("S2", f"{impl}-amdahl+comm", n / 1e6, th_comm[-1], n, p, t, 1,
               f"k={k:.5f}")
    ax.axhline(16, color=AXIS, linewidth=1.5, zorder=1, label="Ideal (16x)")
    line(ax, xs, th_plain, STYLE[impl], label="Amdahl (serial fraction only)",
         dashed=True, no_marker=True)
    ax.plot(xs, th_comm, color=STYLE[impl]["color"], linestyle="none", marker="x",
            markersize=6, markeredgewidth=1.5, zorder=4,
            label="Amdahl + measured communication")
    line(ax, xs, emp, STYLE[impl], label="Empirical")
    ax.set_ylim(12, 16.5)
    ax.grid(axis="x", visible=False)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel("n (millions)")
    ax.legend(loc="lower right", fontsize=9)
axes[0].set_ylabel("Speedup vs serial")
save(fig, "graphS2.png",
     RULE.replace("; bars span to the slowest repeat", "") +
     "\nr_s = median file-write time at that n / serial runtime. On one node the "
     "measured communication k is tiny (~0.01 s), so the two theory series nearly "
     "coincide.\nThe speedup axis starts at 12.")


# =============================================================================
# Supplementary S3: oversubscription
# =============================================================================

GROUPS = [
    ("POSIX\nThreads", [("pthreads", "cyclic", 1, 16), ("pthreads", "cyclic", 1, 32)]),
    ("OpenMP",         [("openmp", "dynamic", 1, 16), ("openmp", "dynamic", 1, 32)]),
    ("Open MPI",       [("mpi", "cyclic-oversub", 16, 1), ("mpi", "cyclic-oversub", 32, 1)]),
    ("Hybrid\n4 procs x 4 / x 8", [("hybrid", "cyclic-oversub", 4, 4), ("hybrid", "cyclic-oversub", 4, 8)]),
    ("Hybrid\n8 / 16 procs x 2", [("hybrid", "cyclic-oversub", 8, 2), ("hybrid", "cyclic-oversub", 16, 2)]),
]
fig, (ax,) = new_fig("Graph S3 - Oversubscription: more workers than cores",
                     "n = 100,000,000 on one 16-core node: 16 workers vs 32 "
                     "workers (processes x threads)")
width = 0.36
b = base(N_BIG)
for gi, (label, pair) in enumerate(GROUPS):
    for j, (impl, scheme, p, t) in enumerate(pair):
        best, _ = fastest(impl, N_BIG, p, t, 1, scheme)
        if not best:
            continue
        s = b / best["total_s"]
        x = gi + (j - 0.5) * width
        colour = STYLE[impl]["color"]
        ax.bar(x, s, width * 0.92, color=colour if j == 0 else SURFACE,
               edgecolor=colour, linewidth=2, zorder=3)
        ax.text(x, s + 0.15, f"{s:.1f}x", ha="center", va="bottom",
                fontsize=9, color=INK2)
        ax.text(x, 0.5, f"{p * t}", ha="center", va="bottom", fontsize=9,
                color=SURFACE if j == 0 else colour, fontweight="semibold")
        record("S3", f"{impl}-{p}x{t}", p * t, s, N_BIG, p, t, 1, "speedup")
ax.axhline(16, color=AXIS, linewidth=1.5, zorder=1, linestyle=(0, (4, 3)))
ax.text(-0.45, 16.15, "16x = one worker per core", ha="left", va="bottom",
        color=MUTED, fontsize=9)
ax.set_xticks(range(len(GROUPS)))
ax.set_xticklabels([g for g, _ in GROUPS])
ax.set_ylim(0, 20)
ax.grid(axis="x", visible=False)
ax.set_ylabel("Speedup vs serial")
corner_note(ax, "Filled: 16 workers.  Outlined: 32 workers on the same 16 cores.  "
                "Number in bar = workers.", y=0.9)
save(fig, "graphS3.png",
     RULE.replace("; bars span to the slowest repeat", "") + "\nMPI and hybrid runs here are launched with mpirun --oversubscribe, "
            "so both bars in each pair use the same launcher.")


# =============================================================================
# Table view
# =============================================================================

with open(os.path.join(OUT, "graph_data.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["graph", "series", "x", "value", "n", "procs", "threads",
                "nodes", "note"])
    w.writerows(TABLE)
print("wrote", os.path.join(OUT, "graph_data.csv"))
