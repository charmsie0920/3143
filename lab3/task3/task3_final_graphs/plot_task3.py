#!/usr/bin/env python3
"""
plot_task3.py -- Task 3 graphs and tables for the EV charging WSN (wsn_ver3.c).
Charmaine Chee Hing Yi 33520399 | Nazara Glaukofani Lopulisa 35135271

Usage:
  python3 plot_task3.py INPUT [OUTPUT_DIR] [--compare ONE_MACHINE_INPUT]

  INPUT    the runs/ folder from wsn_sweep.job (reads runs/grid_*/ and runs/line_*/ with a DONE file)
           or a single results.csv (from task3_sweep.sh)
  --compare  a second input measured on ONE machine (e.g. task3_results/results.csv);
           adds graph 4, one machine vs two machines
  A folder input is treated as "2 machines" (wsn_sweep.job spreads every run), a csv as "1 machine".

Graphs (OUTPUT_DIR, default graphs/):
  1_task3a_grid.png          Task 3a: round-trip delays vs 2D grid size
  2_task3b_line.png          Task 3b: round-trip delays vs line size
  3_ondemand_vs_periodic.png waiting time and message count, our design vs periodic exchange
  4_one_vs_two_machines.png  (with --compare) what crossing the network adds
  5_every_message_type.png   delay of EVERY message type, grid and line (spec Task 4c)
  6a_node_versions.png       node versions: on-demand vs 3 periodic   (needs runs/variants_*)
  6b_base_versions.png       base station listener versions           (needs runs/variants_*)
  7_period.png               base station reply vs predefined period  (needs runs/period_*)
Tables (OUTPUT_DIR/tables/): every measured message type per size, plus summary.md

Needs: matplotlib   (pip install matplotlib)
"""
import csv
import glob
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------------- arguments
args = sys.argv[1:]
COMPARE = None
if "--compare" in args:
    i = args.index("--compare")
    COMPARE = args[i + 1]
    del args[i:i + 2]
BASE = 1          # base station listener used for the grid/line/period graphs (1 = Irecv+Test, our default)
if "--base" in args:
    i = args.index("--base")
    BASE = int(args[i + 1])
    del args[i:i + 2]
SRC = args[0] if args else ("runs" if os.path.isdir("runs") else "task3_results/results.csv")
OUT = args[1] if len(args) > 1 else "graphs"
TABLES = os.path.join(OUT, "tables")
os.makedirs(TABLES, exist_ok=True)

# ---------------------------------------------------------------- style
# validated categorical slots (fixed order); every series also gets a marker + direct label
SURFACE, INK, INK2, MUTED, GRIDC, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family": ["DejaVu Sans", "sans-serif"], "font.size": 11,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRIDC, "grid.linewidth": 0.7,
    "legend.frameon": False, "legend.labelcolor": INK2,
})

# ---------------------------------------------------------------- loading
def csv_files(src):
    if os.path.isdir(src):
        return sorted(os.path.join(d, "results.csv")
                      for d in glob.glob(os.path.join(src, "grid_*")) + glob.glob(os.path.join(src, "line_*"))
                      if os.path.isfile(os.path.join(d, "DONE")) and os.path.isfile(os.path.join(d, "results.csv")))
    return [src]

def num(s):
    return float(s) if s not in ("", None) else None

def load(src):
    """-> list of runs: dict(topology, m, n, nodes, variant, sim, nbr_msgs, alerts, msgs={type: stats})"""
    runs = {}
    for path in csv_files(src):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("topology") in (None, "topology") or int(r["dead_node"]) != -1:
                    continue
                key = (r["topology"], int(r["m"]), int(r["n"]), int(r["variant"]), int(r["base_variant"]))
                run = runs.setdefault(key, dict(topology=r["topology"], m=int(r["m"]), n=int(r["n"]),
                                               nodes=int(r["nodes"]), variant=int(r["variant"]),
                                               k=int(r["k"]), busy=int(r["busy_pct"]), cycle=float(r["cycle_s"]),
                                               threshold=int(r["threshold"]), sim=float(r["sim_time_s"]),
                                               nbr_msgs=float(r["nbr_msgs"]), alerts=int(r["alerts"]), msgs={}))
                run["msgs"][r["msg_type"]] = dict(count=float(r["count"]), avg=num(r["avg_ms"]),
                                                  min=num(r["min_ms"]), max=num(r["max_ms"]))
    # the same size may have been measured with several base listeners: keep the one chosen
    # with --base, fall back to another listener only if that size was never run with it
    best = {}
    for (topo, m, n, var, bv), run in runs.items():
        k = (topo, m, n, var)
        if k not in best or (bv == BASE and best[k][0] != BASE):
            best[k] = (bv, run)
    for bv, run in best.values():
        run["base_variant"] = bv
    return [run for _, run in best.values()]

def by_size(runs, topology, variant):
    """Combine runs with the same node count (e.g. 2x3 and 3x2): count-weighted means."""
    groups = defaultdict(list)
    for r in runs:
        if r["topology"] == topology and r["variant"] == variant:
            groups[r["nodes"]].append(r)
    out = []
    for n in sorted(groups):
        rs = groups[n]
        point = dict(nodes=n, runs=rs, shapes="/".join(f"{r['m']}x{r['n']}" for r in rs),
                     nbr_msgs=sum(r["nbr_msgs"] for r in rs), alerts=sum(r["alerts"] for r in rs),
                     node_minutes=sum(r["nodes"] * r["sim"] / 60 for r in rs), msgs={})
        for msg in {m for r in rs for m in r["msgs"]}:
            es = [r["msgs"][msg] for r in rs if msg in r["msgs"] and r["msgs"][msg]["count"]
                  and r["msgs"][msg]["avg"] is not None]
            if es:
                c = sum(e["count"] for e in es)
                point["msgs"][msg] = dict(count=c, avg=sum(e["avg"] * e["count"] for e in es) / c,
                                          min=min(e["min"] for e in es), max=max(e["max"] for e in es))
        out.append(point)
    return out

def v(p, msg, field="avg"):
    e = p["msgs"].get(msg)
    return e[field] if e else None

def overall(points, msg):
    """count-weighted mean over all sizes"""
    es = [p["msgs"][msg] for p in points if msg in p["msgs"]]
    c = sum(e["count"] for e in es)
    return sum(e["avg"] * e["count"] for e in es) / c if c else None

RUNS = load(SRC)
MACHINES = 2 if os.path.isdir(SRC) else 1
print(f"loaded {len(RUNS)} runs from {SRC} ({MACHINES} machine{'s' if MACHINES > 1 else ''} per run)")
from collections import Counter as _C
print("  base listener per grid/line run:", dict(_C(f"b{r['base_variant']}" for r in RUNS)), f"(wanted b{BASE})")
SUITE_DIRS = (glob.glob(os.path.join(SRC, "variants_*")) + glob.glob(os.path.join(SRC, "period_*"))
              if os.path.isdir(SRC) else [])
if not RUNS and not SUITE_DIRS:
    sys.exit("no finished runs found")
if RUNS:
    first = RUNS[0]
else:   # only variants/period runs in this folder: take the settings from one of them
    first = load(sorted(d for d in SUITE_DIRS if os.path.isfile(d + "/results.csv"))[0] + "/results.csv")[0]
SETUP = (f"{first['k']} ports per node, {first['busy']}% of ports busy, threshold {first['threshold']}, "
         f"{first['cycle']:g} s cycle, ~{first['sim']:.0f} s per run")
WHERE = ("every run split over 2 CAAS compute nodes" if MACHINES == 2
         else "every run on 1 CAAS compute node")

# ---------------------------------------------------------------- drawing helpers
def figure(title, subtitle, ncols=1, width=10, height=5.6):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), squeeze=False)
    fig.suptitle(title, x=0.012, y=0.975, ha="left", fontsize=16, fontweight="bold", color=INK)
    fig.text(0.012, 0.895, subtitle, ha="left", va="top", fontsize=10.5, color=INK2, linespacing=1.45)
    return fig, list(axes.ravel())

def finish(fig, name, note, top=0.87, right=1.0):
    fig.text(0.012, 0.015, note, ha="left", va="bottom", fontsize=8.8, color=MUTED, linespacing=1.4)
    fig.tight_layout(rect=(0, 0.06 + 0.03 * note.count("\n"), right, top))

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)

def series(ax, xs, ys, color, marker, label, dashed=False):
    ax.plot(xs, ys, color=color, marker=marker, markersize=7, linewidth=2.2, label=label,
            linestyle="--" if dashed else "-", markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3)

def end_label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(10, 0), textcoords="offset points", va="center",
                fontsize=10.5, color=INK2, annotation_clip=False)

def shape_points(ax, points, msg, color, marker):
    xs, ys = [], []
    for p in points:
        for r in p.get("runs", []):
            e = r["msgs"].get(msg)
            if e and e["count"] and e["avg"] is not None and len(p["runs"]) > 1:
                xs.append(r["nodes"]); ys.append(e["avg"])
    if xs:
        ax.scatter(xs, ys, s=26, facecolors="none", edgecolors=color, linewidths=1, alpha=0.75, zorder=2)

def xaxis_nodes(ax):
    ax.set_xlabel("EV charging nodes (one MPI process each)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.grid(axis="x", visible=False)

def write_table(name, header, rows):
    with open(os.path.join(TABLES, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote", os.path.join(TABLES, name))

ALL_TYPES = ["nbr_rtt", "nbr_round", "alert_rtt", "report", "reply", "termination", "base_processing"]

def full_table(points, name):
    write_table(name, ["nodes", "shapes", "alerts", "nbr_msgs"] + [f"{t}_avg_ms" for t in ALL_TYPES]
                + [f"{t}_max_ms" for t in ALL_TYPES],
                [[p["nodes"], p["shapes"], p["alerts"], int(p["nbr_msgs"])]
                 + [f"{v(p, t):.4f}" if v(p, t) is not None else "" for t in ALL_TYPES]
                 + [f"{v(p, t, 'max'):.4f}" if v(p, t, 'max') is not None else "" for t in ALL_TYPES]
                 for p in points])

# ---------------------------------------------------------------- graphs 1 & 2
def delay_graph(points, fname, title, kind):
    if not points:
        print(f"skip {fname}: no data")
        return
    xs = [p["nodes"] for p in points]
    alert = [v(p, "alert_rtt") for p in points]
    nbr = [v(p, "nbr_rtt") for p in points]
    n_shapes = sum(len(p["shapes"].split("/")) for p in points)
    fig, (ax,) = figure(title,
                        f"On-demand design, {n_shapes} {kind} from {xs[0]} to {xs[-1]} nodes, {WHERE}.\n{SETUP}.")
    series(ax, xs, alert, ORANGE, "s", "Alert -> base station reply")
    series(ax, xs, nbr, BLUE, "o", "Neighbour request -> response")
    shape_points(ax, points, "alert_rtt", ORANGE, "s")
    shape_points(ax, points, "nbr_rtt", BLUE, "o")
    ax.set_ylim(0, 4.5)
    ax.set_ylabel("Mean round-trip time (ms)")
    xaxis_nodes(ax)
    ax.legend(loc="lower left", ncol=2)
    finish(fig, fname,
           "Round trips are timed on the sending node, so no clock synchronisation is needed. Line = mean "
           "per node count; hollow markers = each individual shape.\n"
           "Every other message type (one-way report, reply, terminate, base processing): graph 5 and tables/.",
           right=0.84)
    end_label(ax, xs[-1], alert[-1], f"Alert round trip\n~{overall(points, 'alert_rtt'):.1f} ms", ORANGE)
    end_label(ax, xs[-1], nbr[-1], f"Neighbour round trip\n~{overall(points, 'nbr_rtt'):.1f} ms", BLUE)
    save(fig, fname)

grid = by_size(RUNS, "grid", 0)
line = by_size(RUNS, "line", 0)
delay_graph(grid, "1_task3a_grid.png", "Task 3a: communication delay vs grid size", "grid shapes")
delay_graph(line, "2_task3b_line.png", "Task 3b: communication delay vs line length", "line lengths")
def shape_table(runs, topology, name):
    rs = sorted([r for r in runs if r["topology"] == topology and r["variant"] == 0],
                key=lambda r: (r["nodes"], r["m"]))
    def g(r, t, f="avg"):
        e = r["msgs"].get(t)
        return f"{e[f]:.4f}" if e and e["count"] and e[f] is not None else ""
    write_table(name, ["shape", "nodes", "alerts", "nbr_msgs"] + [f"{t}_avg_ms" for t in ALL_TYPES]
                + [f"{t}_max_ms" for t in ALL_TYPES],
                [[f"{r['m']}x{r['n']}", r["nodes"], r["alerts"], int(r["nbr_msgs"])]
                 + [g(r, t) for t in ALL_TYPES] + [g(r, t, "max") for t in ALL_TYPES] for r in rs])
    return len(rs)

N_GRID_SHAPES = shape_table(RUNS, "grid", "task3a_grid_every_shape.csv")
N_LINE_SHAPES = shape_table(RUNS, "line", "task3b_line_every_length.csv")

# ---------------------------------------------------------------- graph 3: on-demand vs periodic
per = by_size(RUNS, "grid", 1)
common = sorted({p["nodes"] for p in grid} & {p["nodes"] for p in per})
if common:
    od = {p["nodes"]: p for p in grid}
    pe = {p["nodes"]: p for p in per}
    rate = lambda p: p["nbr_msgs"] / p["node_minutes"]
    fig, (a1, a2) = figure("Our on-demand design vs a periodic exchange",
                           "Periodic = every node swaps availability with its neighbours every cycle and joins a "
                           "whole-grid MPI_Allreduce\neach cycle to agree when to stop. "
                           f"2D grids, {WHERE}.", ncols=2, width=13.5, height=5.8)
    y_od = [v(od[n], "nbr_round") for n in common]
    y_pe = [v(pe[n], "term_allreduce") for n in common]
    series(a1, common, y_pe, ORANGE, "s", "Periodic: whole-grid Allreduce, every cycle")
    series(a1, common, y_od, BLUE, "o", "On-demand: ask own neighbours, only when busy")
    a1.set_title("How long a node waits on others (mean per wait)", loc="left", fontsize=12, color=INK)
    a1.set_ylabel("Milliseconds")
    a1.set_ylim(0, max(y_pe) * 1.45)          # headroom so the legend never sits on the data
    xaxis_nodes(a1)
    a1.legend(loc="upper left", fontsize=10)

    series(a2, common, [rate(pe[n]) for n in common], ORANGE, "s", "Periodic")
    series(a2, common, [rate(od[n]) for n in common], BLUE, "o", "On-demand")
    a2.set_title("Neighbour messages sent per node per minute", loc="left", fontsize=12, color=INK)
    a2.set_ylabel("Messages")
    a2.set_ylim(bottom=0)
    xaxis_nodes(a2)
    a2.legend(loc="lower right", fontsize=10)
    finish(fig, "3", f"{SETUP}. At 85% busy most nodes are congested every cycle, so on-demand asks often "
                     "(request + reply per neighbour);\nat normal load it sends far fewer messages than periodic. "
                     "Each size was run once, so the periodic line is noisy: one slow node delays the whole grid.",
           top=0.86)
    save(fig, "3_ondemand_vs_periodic.png")
    write_table("ondemand_vs_periodic.csv",
                ["nodes", "ondemand_nbr_round_ms", "periodic_allreduce_avg_ms", "periodic_allreduce_max_ms",
                 "ondemand_msgs_per_node_min", "periodic_msgs_per_node_min"],
                [[n, f"{v(od[n], 'nbr_round'):.4f}", f"{v(pe[n], 'term_allreduce'):.4f}",
                  f"{v(pe[n], 'term_allreduce', 'max'):.4f}", f"{rate(od[n]):.1f}", f"{rate(pe[n]):.1f}"]
                 for n in common])

# ---------------------------------------------------------------- graph 4: one vs two machines
if COMPARE:
    one = load(COMPARE)
    one_grid, one_line = by_size(one, "grid", 0), by_size(one, "line", 0)
    labels = ["Neighbour\nround trip", "Alert -> reply\nround trip"]
    msgs = ["nbr_rtt", "alert_rtt"]
    v1 = [overall(one_grid + one_line, m) for m in msgs]
    v2 = [overall(grid + line, m) for m in msgs]
    fig, (ax,) = figure("What crossing the network adds",
                        "Same code and settings (on-demand design). Mean over every message of all grid and line "
                        "sizes.", width=10, height=5.4)
    x = range(len(msgs))
    w = 0.34
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], v1, w, color=BLUE, label="1 compute node (shared memory)", zorder=3)
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], v2, w, color=ORANGE, label="2 compute nodes (network)", zorder=3)
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.2f} ms", (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10.5, color=INK2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=11, color=INK2)
    ax.set_ylabel("Mean round-trip time (ms)")
    ax.set_ylim(0, max(v2 + v1) * 1.25)
    ax.legend(loc="upper left")
    finish(fig, "4", "Most of each round trip is the 1 ms polling interval: a message waits until the receiver's "
                     "next check.\nSplitting each run over 2 machines added "
                     f"{(v2[0] - v1[0]):.2f} ms to a neighbour round trip and {(v2[1] - v1[1]):.2f} ms to an alert "
                     "round trip.", top=0.88)
    save(fig, "4_one_vs_two_machines.png")


# ---------------------------------------------------------------- graph 5: every message type
MSG_PANELS = [("nbr_rtt", "Neighbour request -> response\n(TAG_NBR_REQ / TAG_NBR_RESP, round trip)", 1),
              ("report", "Alert report, node -> base\n(TAG_REPORT, one-way)", 1),
              ("reply", "Nearest-node reply, base -> node\n(TAG_REPLY, one-way)", 1),
              ("termination", "Terminate, base -> node\n(TAG_TERM, one-way)", 1),
              ("alert_rtt", "Alert -> reply\n(round trip, timed on the node)", 1),
              ("base_processing", "Base station processing per alert\n(computation, not a message) - microseconds", 1000)]
if grid or line:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    fig.suptitle("Task 3: delay of every message type vs number of nodes", x=0.012, y=0.985, ha="left",
                 fontsize=16, fontweight="bold", color=INK)
    fig.text(0.012, 0.935, f"On-demand design, {WHERE}. Solid = 2D grid ({N_GRID_SHAPES} shapes), dashed = line "
             f"({N_LINE_SHAPES} lengths). Each panel has its own y-axis.", ha="left", va="top", fontsize=10.5, color=INK2)
    for ax, (msg, title, scale) in zip(axes.ravel(), MSG_PANELS):
        for pts, color, marker, dashed, lbl in ((grid, BLUE, "o", False, "2D grid"),
                                                (line, ORANGE, "s", True, "Line (1 x N)")):
            xs = [p["nodes"] for p in pts if v(p, msg) is not None]
            ys = [v(p, msg) * scale for p in pts if v(p, msg) is not None]
            if xs:
                ax.plot(xs, ys, color=color, marker=marker, markersize=5, linewidth=1.8, label=lbl,
                        linestyle="--" if dashed else "-", markeredgecolor=SURFACE, markeredgewidth=1, zorder=3)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.set_ylim(bottom=0, top=max(ax.get_ylim()[1], 0.1) * 1.15)
        ax.set_ylabel("microseconds" if scale == 1000 else "ms")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("EV charging nodes", fontsize=10)
    axes[0][0].legend(loc="lower right", fontsize=9.5)
    fig.text(0.012, 0.012, "One-way delays use Cristian clock synchronisation with the base station (error <= half the "
             "smallest sync round trip). Round trips need no synchronisation.\nTAG_DONE (shutdown statistics) and "
             "TAG_SYNC (start-up) are sent once per node and are not timed. Every value per size: see tables/.",
             ha="left", va="bottom", fontsize=8.8, color=MUTED, linespacing=1.4)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    save(fig, "5_every_message_type.png")

# ---------------------------------------------------------------- suites from wsn_sweep.job
def suite_runs(prefix):
    """runs/<prefix>_*/ with DONE -> list of (run dict, base log text)"""
    if not os.path.isdir(SRC):
        return []
    out = []
    for d in sorted(glob.glob(os.path.join(SRC, prefix + "_*"))):
        if not os.path.isfile(os.path.join(d, "DONE")):
            continue
        rs = load(d + "/results.csv") if os.path.isfile(d + "/results.csv") else []
        logs = glob.glob(os.path.join(d, "run_*", "base_station.log"))
        if rs and logs:
            r = rs[0]
            r["base_variant"] = int(os.path.basename(d).split("_b")[1].split("_")[0])
            r["period"] = float(os.path.basename(d).split("_p")[1].split("_")[0])
            out.append((r, open(logs[0], errors="replace").read()))
    return out

import re
NODE_V = {0: "v0 on-demand\n(Isend/Iprobe, only when busy)", 1: "v1 periodic\nNeighbor_allgather (blocking)",
          2: "v2 periodic\nIneighbor_allgather", 3: "v3 periodic\nIsend/Irecv + Waitall"}
BASE_V = {0: "b0 Iprobe + Recv,\nblocking Send reply", 1: "b1 Irecv + Test,\nIsend reply",
          2: "b2 persistent Recv_init\n+ Testsome, Isend reply"}
SLOTS4 = [BLUE, ORANGE, "#1baf7a", "#eda100"]

var = suite_runs("variants")
if var:
    sizes = sorted({(r["m"], r["n"]) for r, _ in var}, key=lambda s: (s[0] == 1, s[0] * s[1]))
    size_lbl = [f"{m}x{n}" + (" line" if m == 1 else " grid") for m, n in sizes]
    def mean_over(sel, msg):
        es = [r["msgs"][msg] for r in sel if msg in r["msgs"] and r["msgs"][msg]["count"]]
        c = sum(e["count"] for e in es)
        return sum(e["avg"] * e["count"] for e in es) / c if c else 0
    def wait_per_cycle(sel):
        """ms a node spends per cycle waiting on neighbours/others: exchange rounds + Allreduce"""
        tot = cyc = 0
        for r in sel:
            for t in ("nbr_round", "term_allreduce"):
                e = r["msgs"].get(t)
                if e and e["count"] and e["avg"] is not None:
                    tot += e["avg"] * e["count"]
            e = r["msgs"].get("term_allreduce")
            cyc += r["nodes"] * r["sim"]          # node-seconds ~ node-cycles at a 1 s cycle
        return tot / cyc if cyc else 0
    # ---- 6a: node versions
    fig, (a1,) = figure("Which way should nodes talk to their neighbours?",
                        "4 node versions, each run with all 3 base station versions at 3 sizes "
                        f"({WHERE}).\nBar = milliseconds per second a node spends waiting on other nodes "
                        "(lower = more time for useful work).", width=12, height=6.2)
    w = 0.2
    for i, (vv, name) in enumerate(NODE_V.items()):
        ys = [wait_per_cycle([r for r, _ in var if (r["m"], r["n"]) == sz and r["variant"] == vv]) for sz in sizes]
        bars = a1.bar([j + (i - 1.5) * w for j in range(len(sizes))], ys, w * 0.92, color=SLOTS4[i], label=name, zorder=3)
        for b in bars:
            a1.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=9, color=INK2)
    a1.set_ylabel("ms waiting per node per second")
    a1.set_xticks(range(len(sizes)))
    a1.set_xticklabels(size_lbl)
    a1.grid(axis="x", visible=False)
    a1.set_ylim(0, a1.get_ylim()[1] * 1.55)
    a1.legend(loc="upper left", fontsize=9, ncol=2)
    finish(fig, "6a", "Takeaway: asking neighbours only when busy (v0) wastes the least time. The periodic versions exchange "
                      "every cycle AND join a whole-grid\nMPI_Allreduce every cycle to agree when to stop, so every node "
                      "waits for the slowest one; the gap grows with network size.", top=0.82)
    save(fig, "6a_node_versions.png")

    # ---- 6b: base station versions
    fig, (a2,) = figure("Which way should the base station listen for reports?",
                        "3 base station versions, each run with all 4 node versions at 3 sizes "
                        f"({WHERE}).\nBar = alert -> reply round trip, timed on the reporting node "
                        "(lower = EVs redirected sooner).", width=12, height=6.2)
    w = 0.26
    for i, (bv, name) in enumerate(BASE_V.items()):
        ys = [mean_over([r for r, _ in var if (r["m"], r["n"]) == sz and r["base_variant"] == bv], "alert_rtt")
              for sz in sizes]
        bars = a2.bar([j + (i - 1) * w for j in range(len(sizes))], ys, w * 0.92, color=SLOTS4[i], label=name, zorder=3)
        for b in bars:
            a2.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=9, color=INK2)
    a2.set_ylabel("Alert round trip (ms)")
    a2.set_xticks(range(len(sizes)))
    a2.set_xticklabels(size_lbl)
    a2.grid(axis="x", visible=False)
    a2.set_ylim(0, a2.get_ylim()[1] * 1.5)
    a2.legend(loc="upper left", fontsize=9, ncol=3)
    finish(fig, "6b", "Takeaway: all three versions work correctly; the pre-posted non-blocking receive (b1) gave the "
                      "shortest round trip in every size,\nso b1 is our default listener (Iprobe -> Recv kept as option b0). "
                      "Full numbers: tables/versions.csv.", top=0.82)
    save(fig, "6b_base_versions.png")
    write_table("versions.csv", ["size", "node_version", "base_version", "msg_type", "count", "avg_ms", "max_ms"],
                [[f"{r['m']}x{r['n']}", r["variant"], r["base_variant"], t, int(e["count"]),
                  f"{e['avg']:.4f}", f"{e['max']:.4f}"]
                 for r, _ in sorted(var, key=lambda x: (x[0]["m"], x[0]["n"], x[0]["variant"], x[0]["base_variant"]))
                 for t, e in sorted(r["msgs"].items()) if e["count"] and e["avg"] is not None])

per_runs = sorted(suite_runs("period"), key=lambda x: x[0]["period"])
if any(r["base_variant"] == BASE for r, _ in per_runs):
    per_runs = [x for x in per_runs if x[0]["base_variant"] == BASE]
if per_runs:
    found, none = [], []
    for r, text in per_runs:
        m = re.search(r"replies\s*:\s*(\d+) with nearest nodes, (\d+) with none", text)
        found.append(int(m.group(1)) if m else 0)
        none.append(int(m.group(2)) if m else 0)
    r0 = per_runs[0][0]
    fig, (ax,) = figure("Task 2: effect of the base station's \"recent report\" period",
                        f"{r0['m']}x{r0['n']} grid, {WHERE}. A longer period counts more neighbour reports as "
                        "recent,\nso the base station more often answers \"no nodes available nearby\".",
                        width=11, height=5.6)
    xs = range(len(per_runs))
    ax.bar(xs, found, 0.6, color=BLUE, label="Reply lists the nearest available node(s)", zorder=3)
    ax.bar(xs, none, 0.6, bottom=found, color=ORANGE, label="Reply: no nodes available nearby", zorder=3)
    for x, f_, n_ in zip(xs, found, none):
        if f_ + n_:
            ax.annotate(f"{100 * f_ / (f_ + n_):.0f}% found", (x, f_ + n_), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=10, color=INK2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{r['period']:g} s" for r, _ in per_runs])
    ax.set_xlabel("Predefined period")
    ax.set_ylabel("Replies sent by the base station")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(f_ + n_ for f_, n_ in zip(found, none)) * 1.45)   # room for the legend
    ax.legend(loc="upper left", ncol=2)
    psetup = (f"{r0['k']} ports per node, {r0['busy']}% of ports busy, threshold {r0['threshold']}, "
              f"{r0['cycle']:g} s cycle, ~{r0['sim']:.0f} s per run")
    finish(fig, "7", f"{psetup}. Period 0 s = no report ever counts as recent, so the base always searches for "
                     "candidates.", top=0.84)
    save(fig, "7_period.png")
    write_table("period.csv", ["period_s", "alerts", "replies_found", "replies_none"],
                [[r["period"], r["alerts"], f_, n_] for (r, _), f_, n_ in zip(per_runs, found, none)])

# ---------------------------------------------------------------- summary for slides
L = ["# Task 3 summary", "", f"Source: {SRC} ({WHERE}); {len(RUNS)} runs", ""]
for name, pts in (("2D grid", grid), ("Line", line)):
    if pts:
        L += [f"## {name}: {pts[0]['nodes']}-{pts[-1]['nodes']} nodes (on-demand)", "",
              "| measurement | mean over all sizes (ms) | lowest size mean | highest size mean |", "|---|---|---|---|"]
        for t, lbl in (("nbr_rtt", "Neighbour request -> response"), ("alert_rtt", "Alert -> base reply"),
                       ("report", "Report one-way"), ("reply", "Reply one-way"),
                       ("termination", "Terminate one-way"), ("base_processing", "Base processing per alert")):
            vals = [v(p, t) for p in pts if v(p, t) is not None]
            if vals:
                L.append(f"| {lbl} | {overall(pts, t):.4f} | {min(vals):.4f} | {max(vals):.4f} |")
        L.append("")
if common:
    L += ["## On-demand vs periodic (grid)", "",
          "| nodes | on-demand wait (ms) | periodic Allreduce wait avg / max (ms) | msgs/node/min on-demand | periodic |",
          "|---|---|---|---|---|"]
    for n in common:
        L.append(f"| {n} | {v(od[n], 'nbr_round'):.2f} | {v(pe[n], 'term_allreduce'):.2f} / "
                 f"{v(pe[n], 'term_allreduce', 'max'):.1f} | {rate(od[n]):.0f} | {rate(pe[n]):.0f} |")
    L.append("")
if COMPARE:
    L += ["## One vs two compute nodes", "", f"Neighbour round trip: {v1[0]:.3f} -> {v2[0]:.3f} ms",
          f"Alert round trip: {v1[1]:.3f} -> {v2[1]:.3f} ms", ""]
with open(os.path.join(TABLES, "summary.md"), "w") as f:
    f.write("\n".join(L))
print("wrote", os.path.join(TABLES, "summary.md"))
