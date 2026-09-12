#!/usr/bin/env python3
"""
analyse.py -- turn results.csv into the tables and numbers the report needs.

Produces:
  1. Correctness cross-check (every run at a given n found the same primes)
  2. Median runtimes with spread, empirical speedup and efficiency
  3. Amdahl's Law: measured serial fraction f, theoretical vs actual speedup
  4. Karp-Flatt metric: is the shortfall serial work, or parallel overhead?
  5. Gustafson: scaled speedup from the weak-scaling runs
  6. Implementation comparison (MPI vs pthreads vs OpenMP) for the graphs
  7. Network cost: same process count, one node vs two
  8. summary.csv, ready for plotting

Speedup is always against the median SERIAL time at the same n.

Usage:  python3 analyse.py [results.csv]
"""

import csv
import statistics
import sys
from collections import defaultdict

KNOWN_PI = {1000000: 78498, 5000000: 348513, 10000000: 664579,
            50000000: 3001134, 100000000: 5761455}


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "impl": r["impl"], "scheme": r["scheme"],
                "n": int(r["n"]), "procs": int(r["procs"]),
                "threads": int(r["threads"]), "nodes": int(r["nodes"]),
                "primes": int(r["primes"]),
                "total": float(r["total_s"]),
                "serial": float(r["serial_s"]),
                "parallel": float(r["parallel_s"]),
                "overhead": float(r["overhead_s"]),
                "imbalance": float(r["imbalance_pct"]),
            })
    return rows


def parallelism(r):
    """Degree of parallelism: MPI ranks for mpi, threads for the rest."""
    return r["procs"] if r["impl"] == "mpi" else r["threads"]


def check(rows):
    by_n = defaultdict(set)
    for r in rows:
        by_n[r["n"]].add(r["primes"])
    bad = []
    for n in sorted(by_n):
        c = by_n[n]
        if len(c) > 1:
            bad.append(f"  n={n}: runs disagree {sorted(c)}")
        elif n in KNOWN_PI and next(iter(c)) != KNOWN_PI[n]:
            bad.append(f"  n={n}: got {next(iter(c))}, expected {KNOWN_PI[n]}")
    if bad:
        print("CORRECTNESS PROBLEMS\n" + "\n".join(bad) + "\n")
    else:
        print("Correctness: all runs agree on prime counts.\n")


def summarise(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["impl"], r["scheme"], r["n"], parallelism(r), r["nodes"])].append(r)
    out = {}
    for k, rs in g.items():
        med = lambda f: statistics.median(x[f] for x in rs)
        out[k] = {"reps": len(rs), "total": med("total"),
                  "lo": min(x["total"] for x in rs),
                  "hi": max(x["total"] for x in rs),
                  "serial": med("serial"), "parallel": med("parallel"),
                  "overhead": med("overhead"), "imbalance": med("imbalance")}
    return out


def amdahl(f, p):
    return 1.0 / (f + (1.0 - f) / p) if (f + (1.0 - f) / p) > 0 else float("nan")


def karp_flatt(speedup, p):
    """Experimentally determined serial fraction. Flat as p grows = genuine
    serial work. Rising = parallel overhead."""
    if p <= 1 or speedup <= 0:
        return None
    return (1.0 / speedup - 1.0 / p) / (1.0 - 1.0 / p)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    try:
        rows = load(path)
    except FileNotFoundError:
        print(f"{path} not found. Run collect_results.sh first.")
        return
    if not rows:
        print("No data rows found.")
        return

    print(f"Loaded {len(rows)} runs from {path}\n")
    check(rows)
    s = summarise(rows)

    serial_base = {k[2]: v["total"] for k, v in s.items() if k[0] == "serial"}
    if not serial_base:
        print("WARNING: no serial baseline. Speedups cannot be computed.")
        print("Run baseline.job / nsize_all.job, then collect again.\n")

    # ---- 1. Main results table, per problem size ------------------------
    strong = {k: v for k, v in s.items() if "weak" not in k[1]}

    for n in sorted({k[2] for k in strong}):
        base = serial_base.get(n)
        print("=" * 92)
        hdr = f"n = {n:,}"
        if base:
            hdr += f"    serial baseline = {base:.3f} s"
        print(hdr)
        print("=" * 92)
        print(f"{'impl':<9}{'scheme':<10}{'par':>4}{'nd':>3}{'reps':>5}"
              f"{'median':>9}{'spread':>8}{'speedup':>9}{'effic':>7}"
              f"{'serial':>9}{'ovhd':>9}{'imbal':>7}")
        print("-" * 92)

        for k in sorted([k for k in strong if k[2] == n],
                        key=lambda k: (k[3], k[0], k[1], k[4])):
            impl, scheme, _, par, nodes = k
            v = s[k]
            spread = 100.0 * (v["hi"] - v["lo"]) / v["total"] if v["total"] else 0
            if base and impl != "serial" and v["total"] > 0:
                sp = base / v["total"]
                sp_s, ef_s = f"{sp:.2f}x", f"{100*sp/par:.0f}%"
            else:
                sp_s, ef_s = "-", "-"
            print(f"{impl:<9}{scheme:<10}{par:>4}{nodes:>3}{v['reps']:>5}"
                  f"{v['total']:>9.3f}{spread:>7.1f}%{sp_s:>9}{ef_s:>7}"
                  f"{v['serial']:>9.4f}{v['overhead']:>9.4f}"
                  f"{v['imbalance']:>6.1f}%")
        print()

    # ---- 2. Amdahl and Karp-Flatt ---------------------------------------
    print("=" * 92)
    print("Amdahl's Law: measured serial fraction vs actual speedup")
    print("=" * 92)
    print("f is measured from the phases that do NOT scale with processor")
    print("count (broadcast, prefix sum, merge), taken at the smallest")
    print("parallelism available for that implementation and scheme.\n")

    for (impl, scheme) in sorted({(k[0], k[1]) for k in s
                                  if k[0] != "serial" and "weak" not in k[1]}):
        for n in sorted({k[2] for k in s if k[0] == impl and k[1] == scheme}):
            keys = sorted([k for k in s if k[0] == impl and k[1] == scheme
                           and k[2] == n and k[4] == 1], key=lambda k: k[3])
            if len(keys) < 2:
                continue
            base = serial_base.get(n)
            v0 = s[keys[0]]
            f = v0["serial"] / v0["total"] if v0["total"] > 0 else 0.0

            print(f"{impl}/{scheme}  n={n:,}   measured f = {f:.6f}")
            print(f"  {'p':>4}{'actual':>10}{'amdahl':>10}{'gap':>9}"
                  f"{'karp-flatt e':>15}")
            for k in keys:
                p, v = k[3], s[k]
                if not base or v["total"] <= 0:
                    continue
                act = base / v["total"]
                th = amdahl(f, p)
                e = karp_flatt(act, p)
                e_s = f"{e:.4f}" if e is not None else "-"
                print(f"  {p:>4}{act:>9.2f}x{th:>9.2f}x"
                      f"{100*(th-act)/th:>8.0f}%{e_s:>15}")
            print()

    print("Reading the table: if actual tracks Amdahl, the serial fraction")
    print("explains the loss. If actual falls short AND Karp-Flatt e rises")
    print("with p, the loss is parallel overhead (communication, imbalance),")
    print("which Amdahl does not model.\n")

    # ---- 3. Gustafson (weak scaling) ------------------------------------
    weak = sorted([(k[3], k[2], v) for k, v in s.items() if "weak" in k[1]])

    if len(weak) >= 3:
        print("=" * 92)
        print("Gustafson's Law: weak scaling (work per process held constant)")
        print("=" * 92)
        print("Perfect weak scaling means identical runtime at every p, since")
        print("the problem grows in step with the processor count. Scaled")
        print("speedup S = p - f*(p-1); the shortfall is what overhead costs.\n")

        f_weak = weak[0][2]["serial"] / weak[0][2]["total"] \
            if weak[0][2]["total"] > 0 else 0.0
        one = weak[0][2]["total"]

        print(f"  measured f = {f_weak:.6f}\n")
        print(f"  {'p':>4}{'n':>14}{'runtime':>10}{'vs p=1':>9}"
              f"{'scaled S':>10}{'gustafson':>11}")
        for p, n, v in weak:
            gust = p - f_weak * (p - 1)
            scaled = p * one / v["total"] if v["total"] > 0 else 0.0
            print(f"  {p:>4}{n:>14,}{v['total']:>10.3f}"
                  f"{v['total']/one:>8.2f}x{scaled:>9.2f}x{gust:>10.2f}x")
        print()

    # ---- 4. Implementation comparison (presentation graphs) --------------
    print("=" * 92)
    print("Implementation comparison at matched parallelism (for graphs 1-3)")
    print("=" * 92)
    impls = ["serial", "pthreads", "openmp", "mpi"]
    pars = sorted({k[3] for k in strong if k[4] == 1})
    for n in sorted({k[2] for k in strong}):
        base = serial_base.get(n)
        shown = False
        for par in pars:
            row = {}
            for impl in impls:
                cand = [k for k in strong if k[0] == impl and k[2] == n
                        and k[3] == par and k[4] == 1]
                if cand:
                    row[impl] = min(s[k]["total"] for k in cand)
            if len([i for i in row if i != "serial"]) >= 2:
                if not shown:
                    print(f"\nn = {n:,}"
                          + (f"  (serial {base:.3f} s)" if base else ""))
                    print(f"  {'par':>4}" + "".join(f"{i:>22}" for i in impls[1:]))
                    shown = True
                cells = ""
                for impl in impls[1:]:
                    if impl in row:
                        t = row[impl]
                        sp = f" ({base/t:.2f}x)" if base else ""
                        cells += f"{t:.3f}s{sp}".rjust(22)
                    else:
                        cells += "-".rjust(22)
                print(f"  {par:>4}" + cells)
    print()

    # ---- 5. Network cost -------------------------------------------------
    print("=" * 92)
    print("Network cost: identical process count, one node vs two")
    print("=" * 92)
    found = False
    for k, v in sorted(s.items()):
        impl, scheme, n, par, nodes = k
        if nodes != 1 or impl != "mpi":
            continue
        two = s.get((impl, scheme, n, par, 2))
        if two:
            found = True
            print(f"  {scheme:<9} n={n:,} p={par}: "
                  f"{v['total']:.3f} -> {two['total']:.3f} s "
                  f"({two['total']-v['total']:+.3f}), "
                  f"overhead {v['overhead']:.4f} -> {two['overhead']:.4f} s "
                  f"({two['overhead']-v['overhead']:+.4f})")
    if not found:
        print("  No matched pairs yet (needs sweep_1node p=16 and sweep_2node16).")
    print()

    # ---- 6. Tidy output --------------------------------------------------
    with open("summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["impl", "scheme", "n", "parallelism", "nodes", "reps",
                    "median_s", "min_s", "max_s", "speedup", "efficiency_pct",
                    "serial_s", "overhead_s", "imbalance_pct", "karp_flatt_e"])
        for k, v in sorted(s.items()):
            impl, scheme, n, par, nodes = k
            base = serial_base.get(n)
            sp = base / v["total"] if base and impl != "serial" else None
            e = karp_flatt(sp, par) if sp else None
            w.writerow([impl, scheme, n, par, nodes, v["reps"],
                        f"{v['total']:.6f}", f"{v['lo']:.6f}", f"{v['hi']:.6f}",
                        f"{sp:.4f}" if sp else "",
                        f"{100*sp/par:.2f}" if sp else "",
                        f"{v['serial']:.6f}", f"{v['overhead']:.6f}",
                        f"{v['imbalance']:.2f}",
                        f"{e:.4f}" if e is not None else ""])
    print("Wrote summary.csv for plotting.")


if __name__ == "__main__":
    main()