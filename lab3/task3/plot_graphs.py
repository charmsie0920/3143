#!/usr/bin/env python3
"""
plot_graphs.py -- Task 3 graphs and tables from the CAAS runs of wsn_sweep.job.
Charmaine Chee Hing Yi 33520399 | Nazara Glaukofani Lopulisa 35135271

Reads every finished run folder runs/<suite>_..._r<rep>/ (the ones with a DONE
file): its results.csv (one row per message type, written by the base station)
and its base_*.log (reply outcomes). Unfinished or failed runs are ignored.

Message types (as measured in wsn.c):
  nbr_exchange     node <-> node availability exchange, per cycle (MPI_Wtime around it)
  report           node -> base alert, one-way (clock-synchronised send/receive times)
  reply            base -> node nearest-available reply, one-way
  termination      base -> node termination message, one-way
  term_allreduce   termination-check MPI_Allreduce on the cart communicator, per cycle
  base_processing  base station receive -> reply send (computation, not a message)

Graphs written (to graphs/):
  graph_grid_all_types.png    Task 3a: delay vs 2D grid size, one panel per message type
  graph_line_all_types.png    Task 3b: delay vs line size, one panel per message type
  graph_type_<type>.png       grid vs line on the same axes, one graph per message type
  graph_variants.png          blocking vs non-blocking versions (node exchange, base listener)
  graph_period.png            base station reply outcome vs the "recent report" period
Tables written (to graphs/tables/):
  grid_delay.csv, line_delay.csv, scaling_fit.csv, variants.csv, period.csv, summary.md

Usage: python plot_graphs.py [runs_dir] [output_dir]      (defaults: runs graphs)
"""

import csv
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS_DIR = sys.argv[1] if len(sys.argv) > 1 else "runs"
OUT = sys.argv[2] if len(sys.argv) > 2 else "graphs"
TABLES = os.path.join(OUT, "tables")
os.makedirs(TABLES, exist_ok=True)

# ---- Palette (same validated slots as Lab 2; green < 3:1 contrast -> tables provided) ----
SURFACE = "#fcfcfb"
INK     = "#0b0b0b"
INK2    = "#52514e"
MUTED   = "#898781"
GRID    = "#e1e0d9"
AXIS    = "#c3c2b7"
SLOTS   = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical order, never cycled
MARKERS = ["o", "s", "^"]

TOPO = {"grid": dict(color=SLOTS[0], marker="o", label="2D grid (m x n)"),
        "line": dict(color=SLOTS[1], marker="s", label="Line (1 x N)")}

MSG = [
    ("nbr_exchange",    "Neighbour exchange (node <-> node)"),
    ("report",          "Alert report (node -> base)"),
    ("reply",           "Nearest-node reply (base -> node)"),
    ("termination",     "Termination message (base -> node)"),
    ("term_allreduce",  "Termination-check Allreduce (cart comm)"),
    ("base_processing", "Base station processing (not a message)"),
]
MSG_LABEL = dict(MSG)

NODE_VARIANTS = {0: "Neighbor_allgather (blocking)", 1: "Ineighbor_allgather (non-blocking)",
                 2: "Isend/Irecv (non-blocking p2p)"}
BASE_VARIANTS = {0: "Irecv + Test, Isend reply", 1: "Iprobe + Recv, blocking Send reply",
                 2: "Recv_init + Testsome, Isend reply"}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 10.5,
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
})


# ---- Data ------------------------------------------------------------------------

def num(s):
    return float(s) if s not in ("", None) else None


def load_runs(root):
    """List of runs: dict(suite, tag, cfg..., msgs={type: row}, replies_found, ...)."""
    runs = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isfile(os.path.join(d, "DONE")):
            continue
        res = os.path.join(d, "results.csv")
        if not os.path.isfile(res):
            continue
        tag = os.path.basename(d)
        run = dict(tag=tag, suite=tag.split("_")[0], msgs={})
        with open(res, newline="") as f:
            for r in csv.DictReader(f):
                for key in ("m", "n", "nodes", "k", "threshold", "busy_pct", "variant",
                            "base_variant", "alerts"):
                    run[key] = int(r[key])
                for key in ("cycle_s", "period_s", "poll_ms", "sim_time_s"):
                    run[key] = float(r[key])
                run["topology"] = r["topology"]
                run["msgs"][r["msg_type"]] = dict(
                    count=int(float(r["count"])), avg=num(r["avg_ms"]),
                    min=num(r["min_ms"]), max=num(r["max_ms"]))
        logs = glob.glob(os.path.join(d, "base_*.log"))
        run.update(replies_found=0, replies_none=0, late=0)
        if logs:
            text = open(logs[0], errors="replace").read()
            m = re.search(r"replies\s*:\s*(\d+) with nearest nodes, (\d+) with none", text)
            if m:
                run["replies_found"], run["replies_none"] = int(m.group(1)), int(m.group(2))
            m = re.search(r"(\d+) late \(after listening", text)
            if m:
                run["late"] = int(m.group(1))
        runs.append(run)
    return runs


RUNS = load_runs(RUNS_DIR)
print(f"loaded {len(RUNS)} finished runs from {RUNS_DIR}/")
if not RUNS:
    sys.exit("no finished runs (runs/*/DONE) found - run wsn_sweep.job first")


def by_suite(name):
    return [r for r in RUNS if r["suite"] == name]


def combine(entries):
    """Merge per-run (count, avg, min, max) into one: count-weighted mean."""
    entries = [e for e in entries if e and e["count"] > 0 and e["avg"] is not None]
    if not entries:
        return None
    n = sum(e["count"] for e in entries)
    return dict(count=n, avg=sum(e["count"] * e["avg"] for e in entries) / n,
                min=min(e["min"] for e in entries), max=max(e["max"] for e in entries))


def scaling(topology):
    """{msg: [(nodes, combined, [(shape, run avg), ...]), ...]} sorted by nodes."""
    runs = by_suite(topology)
    out = {}
    for msg, _ in MSG:
        groups = defaultdict(list)
        for r in runs:
            groups[r["nodes"]].append(r)
        series = []
        for nodes in sorted(groups):
            c = combine([r["msgs"].get(msg) for r in groups[nodes]])
            if c:
                pts = [(f'{r["m"]}x{r["n"]}', r["msgs"][msg]["avg"]) for r in groups[nodes]
                       if r["msgs"].get(msg) and r["msgs"][msg]["count"] > 0]
                series.append((nodes, c, pts))
        out[msg] = series
    return out


SCALE = {t: scaling(t) for t in TOPO}


def fit(series):
    """Least-squares line through the per-size means: (slope ms/node, r^2)."""
    if len(series) < 3:
        return None
    x = np.array([s[0] for s in series], float)
    y = np.array([s[1]["avg"] for s in series], float)
    slope, icpt = np.polyfit(x, y, 1)
    ss_res = float(np.sum((y - (slope * x + icpt)) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return slope, (1 - ss_res / ss_tot) if ss_tot > 0 else 1.0


def config_note(runs, varied=()):
    """Fixed parameters of a set of runs; `varied` ones are left out."""
    if not runs:
        return ""
    r = runs[0]
    parts = [f"k={r['k']} ports", f"cycle {r['cycle_s']:g} s", f"threshold {r['threshold']}",
             f"busy {r['busy_pct']}%"]
    if "variant" not in varied:
        parts.append(f"node exchange v{r['variant']}")
    if "base_variant" not in varied:
        parts.append(f"base listener b{r['base_variant']}")
    if "period" not in varied:
        parts.append(f"period {r['period_s']:g} s")
    return ", ".join(parts)


def fit_text(f):
    return f"linear fit {f[0] * 1e3:+.1f} us/node, R^2 {f[1]:.2f}"


# ---- Drawing helpers -------------------------------------------------------------

def new_fig(title, subtitle, nrows=1, ncols=1, width=9.0, height=5.2):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=15, color=INK,
                 fontweight="semibold", y=0.985)
    fig.text(0.01, 1 - 0.47 / height, subtitle, ha="left", va="top", fontsize=10, color=INK2)
    return fig, axes.ravel()


def save(fig, name, note, top=0.9):
    fig.text(0.01, 0.012, note, ha="left", va="bottom", fontsize=8.5, color=MUTED,
             linespacing=1.4)
    fig.tight_layout(rect=(0, 0.075 if "\n" in note else 0.045, 1, top))
    fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


def plot_series(ax, series, topo, label=None, points=True):
    st = TOPO[topo]
    xs = [s[0] for s in series]
    ax.plot(xs, [s[1]["avg"] for s in series], color=st["color"], marker=st["marker"],
            markersize=7, linewidth=2, label=label or st["label"], zorder=3,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    if points:   # individual runs (different shapes with the same node count)
        px = [s[0] for s in series for _ in s[2]]
        py = [p[1] for s in series for p in s[2]]
        if len(px) > len(xs):
            ax.scatter(px, py, s=22, facecolors="none", edgecolors=st["color"],
                       linewidths=1, alpha=0.7, zorder=2)


def write_csv(name, header, rows):
    with open(os.path.join(TABLES, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote", os.path.join(TABLES, name))


RULE = ("Each line point: count-weighted mean delay over every message of that type at that "
        "size; hollow points: individual runs (different m x n shapes).\n"
        "Measured on CAAS, every run spread over 2 compute nodes (Gigabit Ethernet). "
        "One-way delays use Cristian clock sync against the base station.")


# ---- Graphs 1 & 2: delay vs size, one panel per message type ---------------------

def all_types_figure(topo, title, subtitle, fname):
    data = SCALE[topo]
    if not any(data.values()):
        print(f"skip {fname}: no '{topo}' runs yet")
        return
    fig, axes = new_fig(title, subtitle, nrows=2, ncols=3, width=14, height=8.4)
    for ax, (msg, label) in zip(axes, MSG):
        series = data[msg]
        f = fit(series)
        ax.set_title(label, color=INK, fontsize=10.5, loc="left", pad=16)
        if f:   # fit sits between the title and the plot, never on the data
            ax.text(0, 1.02, fit_text(f), transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=8.5, color=INK2)
        if not series:
            ax.text(0.5, 0.5, "no messages of this type", transform=ax.transAxes,
                    ha="center", color=MUTED)
            continue
        plot_series(ax, series, topo)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("EV charging nodes")
        ax.set_ylabel("Mean delay (ms)")
    save(fig, fname, RULE, top=0.92)


grid_runs, line_runs = by_suite("grid"), by_suite("line")
all_types_figure(
    "grid", "Task 3a - Communication delay vs EV grid size (2D Cartesian grid)",
    f"{len(grid_runs)} grid sizes from 2x2 to {max((r['nodes'] for r in grid_runs), default=0)} "
    f"nodes. {config_note(grid_runs)}", "graph_grid_all_types.png")
all_types_figure(
    "line", "Task 3b - Communication delay vs number of nodes (line topology 1 x N)",
    f"{len(line_runs)} line instances. {config_note(line_runs)}", "graph_line_all_types.png")


# ---- Graphs 3..8: one per message type, grid vs line ------------------------------

for msg, label in MSG:
    series = {t: SCALE[t][msg] for t in TOPO if SCALE[t][msg]}
    if not series:
        continue
    notes = [f"{TOPO[t]['label']}: {fit_text(fit(s))}" for t, s in series.items() if fit(s)]
    fig, (ax,) = new_fig(f"{label}: delay vs number of nodes",
                         "2D grid vs line topology, same simulation parameters\n"
                         + "\n".join(notes), height=5.6)
    for topo, s in series.items():
        plot_series(ax, s, topo)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("EV charging nodes")
    ax.set_ylabel("Mean delay (ms)")
    if len(series) > 1:
        ax.legend(loc="upper left")
    save(fig, f"graph_type_{msg}.png", RULE, top=0.82)


# ---- Tables for the scaling graphs ---------------------------------------------------

fit_rows = []
for topo in TOPO:
    rows = []
    for msg, _ in MSG:
        for nodes, c, pts in SCALE[topo][msg]:
            rows.append([msg, nodes, " ".join(p[0] for p in pts), c["count"],
                         f"{c['avg']:.4f}", f"{c['min']:.4f}", f"{c['max']:.4f}"])
        f = fit(SCALE[topo][msg])
        s = SCALE[topo][msg]
        if f:
            fit_rows.append([topo, msg, len(s), s[0][0], s[-1][0], f"{s[0][1]['avg']:.4f}",
                             f"{s[-1][1]['avg']:.4f}", f"{f[0] * 1e3:.3f}", f"{f[1]:.3f}"])
    if rows:
        write_csv(f"{topo}_delay.csv",
                  ["msg_type", "nodes", "shapes", "messages", "avg_ms", "min_ms", "max_ms"], rows)
if fit_rows:
    write_csv("scaling_fit.csv", ["topology", "msg_type", "sizes", "min_nodes", "max_nodes",
                                  "avg_ms_smallest", "avg_ms_largest", "slope_us_per_node",
                                  "r_squared"], fit_rows)


# ---- Graph 9: blocking vs non-blocking versions -----------------------------------

var_runs = by_suite("variants")
var_rows = []
if var_runs:
    sizes = sorted({(r["m"], r["n"]) for r in var_runs}, key=lambda s: (s[0] == 1, s[0] * s[1]))
    size_lbl = [f"{m}x{n}" + (" line" if m == 1 else "") for m, n in sizes]

    def variant_value(msg, key, value, size):
        """Combined delay of msg over runs at size with cfg[key] == value."""
        return combine([r["msgs"].get(msg) for r in var_runs
                        if (r["m"], r["n"]) == size and r[key] == value])

    panels = [("nbr_exchange", "variant", NODE_VARIANTS, "Neighbour exchange, by node version"),
              ("term_allreduce", "variant", NODE_VARIANTS, "Termination-check Allreduce, by node version"),
              ("report", "base_variant", BASE_VARIANTS, "Alert report delay, by base listener"),
              ("reply", "base_variant", BASE_VARIANTS, "Reply delay, by base listener")]
    fig, axes = new_fig("Blocking vs non-blocking MPI versions",
                        "Node exchange versions (v0-v2) x base station listener versions (b0-b2), "
                        "every combination run at each size; each bar averages over the other version",
                        nrows=2, ncols=2, width=13, height=8.6)
    width = 0.26
    for ax, (msg, key, names, title) in zip(axes, panels):
        for i, (val, name) in enumerate(names.items()):
            xs, ys = [], []
            for j, size in enumerate(sizes):
                c = variant_value(msg, key, val, size)
                if c:
                    xs.append(j + (i - 1) * width)
                    ys.append(c["avg"])
            ax.bar(xs, ys, width=width, color=SLOTS[i], edgecolor=SURFACE, linewidth=2,
                   label=f"{key[0] if key == 'variant' else 'b'}{val}: {name}", zorder=3)
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels(size_lbl)
        ax.set_title(title, color=INK, fontsize=10.5, loc="left")
        ax.set_ylabel("Mean delay (ms)")
        ax.legend(loc="upper left", fontsize=8.5)
        ax.margins(y=0.25)
    for r in var_runs:
        for msg, _ in MSG:
            e = r["msgs"].get(msg)
            if e and e["count"] > 0:
                var_rows.append([f'{r["m"]}x{r["n"]}', r["variant"], r["base_variant"], msg,
                                 e["count"], f"{e['avg']:.4f}", f"{e['min']:.4f}", f"{e['max']:.4f}"])
    save(fig, "graph_variants.png",
         "Labels: v = node neighbour-exchange version, b = base station listener version. "
         "Full per-combination numbers in tables/variants.csv.\n"
         + config_note(var_runs, varied=("variant", "base_variant")),
         top=0.91)
    write_csv("variants.csv", ["size", "variant", "base_variant", "msg_type", "messages",
                               "avg_ms", "min_ms", "max_ms"], var_rows)
else:
    print("skip graph_variants.png: no 'variants' runs yet")


# ---- Graph 10: effect of the base station period ----------------------------------

per_runs = sorted(by_suite("period"), key=lambda r: r["period_s"])
if per_runs:
    fig, (ax,) = new_fig("Base station reply outcome vs \"recent report\" period",
                         f"{per_runs[0]['m']}x{per_runs[0]['n']} grid. A longer period means more "
                         "neighbour reports count as recent, so more replies say \"none nearby\"")
    xs = np.arange(len(per_runs))
    found = [r["replies_found"] for r in per_runs]
    none = [r["replies_none"] for r in per_runs]
    ax.bar(xs, found, width=0.6, color=SLOTS[0], edgecolor=SURFACE, linewidth=2,
           label="Reply lists nearest available node(s)", zorder=3)
    ax.bar(xs, none, width=0.6, bottom=found, color=SLOTS[1], edgecolor=SURFACE, linewidth=2,
           label="Reply: no nodes available nearby", zorder=3)
    for x, f_, n_ in zip(xs, found, none):
        tot = f_ + n_
        if tot:
            ax.text(x, tot, f"{100 * n_ / tot:.0f}% none", ha="center", va="bottom",
                    fontsize=9, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['period_s']:g} s" for r in per_runs])
    ax.set_xlabel("Predefined period (s)")
    ax.set_ylabel("Replies sent by the base station")
    ax.legend(loc="upper left")
    ax.margins(y=0.15)
    save(fig, "graph_period.png", config_note(per_runs, varied=("period",)))
    write_csv("period.csv", ["period_s", "alerts", "replies_found", "replies_none_nearby",
                             "late_reports", "report_avg_ms", "reply_avg_ms"],
              [[r["period_s"], r["alerts"], r["replies_found"], r["replies_none"], r["late"],
                f"{r['msgs']['report']['avg']:.4f}" if r["msgs"].get("report", {}).get("count") else "",
                f"{r['msgs']['reply']['avg']:.4f}" if r["msgs"].get("reply", {}).get("count") else ""]
               for r in per_runs])
else:
    print("skip graph_period.png: no 'period' runs yet")


# ---- Markdown summary (paste into slides / speaker notes) --------------------------

lines = ["# Task 3 summary", "", f"Finished runs: {len(RUNS)}", ""]
if fit_rows:
    lines += ["## Scaling (linear fit of mean delay vs number of nodes)", "",
              "| topology | message | sizes | nodes | mean delay smallest -> largest (ms) | slope (us/node) | R^2 |",
              "|---|---|---|---|---|---|---|"]
    for t, msg, k, lo, hi, a, b, s, r2 in fit_rows:
        lines.append(f"| {t} | {MSG_LABEL[msg]} | {k} | {lo}-{hi} | {a} -> {b} | {s} | {r2} |")
    lines.append("")
if var_runs:
    lines += ["## Versions (mean delay, ms)", "",
              "| size | v | b | neighbour exchange | report | reply | termination |",
              "|---|---|---|---|---|---|---|"]
    for r in sorted(var_runs, key=lambda r: (r["m"] == 1, r["nodes"], r["variant"], r["base_variant"])):
        def g(msg):
            e = r["msgs"].get(msg)
            return f"{e['avg']:.3f}" if e and e["count"] else "-"
        lines.append(f"| {r['m']}x{r['n']} | {r['variant']} | {r['base_variant']} | "
                     f"{g('nbr_exchange')} | {g('report')} | {g('reply')} | {g('termination')} |")
    lines.append("")
with open(os.path.join(TABLES, "summary.md"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote", os.path.join(TABLES, "summary.md"))
