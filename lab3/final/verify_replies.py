#!/usr/bin/env python3
"""
verify_replies.py -- independently re-check every base station decision in the logs.
Charmaine Chee Hing Yi 33520399 | Nazara Glaukofani Lopulisa 35135271

Replays each base_station.log alert by alert, rebuilding the base station's knowledge
(last report / last known full / last silent per node) from the log alone, and recomputes:
  - how many neighbours reported within the predefined period
  - the reply: "no nodes available nearby", "none in the whole grid", or ALL nodes at the
    smallest grid (Manhattan) distance that are not known full or silent within the period
then compares with the decision the base station logged.

Times in the log have 1 ms resolution, so a node whose age is within 1 ms of the period
can legitimately go either way; such cases are counted as "boundary", not as errors.

Usage: python3 verify_replies.py [runs_dir]      (default: runs)
"""
import glob
import os
import re
import sys

NEVER = float("-inf")
TOL = 0.0015                                    # log time resolution (1 ms) + rounding

HDR = re.compile(r"grid=(\d+)x(\d+) .* threshold=(\d+)")
PER = re.compile(r"period=([\d.]+)s")
ALERT = re.compile(r"\[t=\s*([\d.]+)s\] ALERT #\d+ from node (\d+) ")
NBR = re.compile(r"(up|down|left|right)=node(\d+)\(\d+,\d+\) (avail=(-?\d+)|NO-RESPONSE)")
DEC = re.compile(r"decision: (\d+)/(\d+) neighbours reported within [\d.]+s -> (.*?) \| processing")
FOUND = re.compile(r"(\d+) nearest available node\(s\) at distance (\d+):((?: node\d+\(\d+,\d+\))+)")


def check_log(path):
    text = open(path, errors="replace").read().splitlines()
    m, n, threshold = map(int, HDR.search(text[1]).groups())
    period = float(PER.search(text[2]).group(1))
    N = m * n
    last_report = [NEVER] * N
    last_full = [NEVER] * N
    last_silent = [NEVER] * N
    res = dict(alerts=0, found=0, ok=0, boundary=0, wrong=[])

    i = 0
    while i < len(text):
        a = ALERT.match(text[i])
        if not a:
            i += 1
            continue
        t, me = float(a.group(1)), int(a.group(2))
        nbrs = [(int(j), None if av is None else int(av)) for _, j, _, av in NBR.findall(text[i + 2])]
        d = DEC.search(text[i + 4])
        i += 5
        res["alerts"] += 1

        # neighbours' report ages are taken BEFORE this report updates the tables (as in handle_report)
        ages = [t - last_report[j] for j, _ in nbrs]
        recent = sum(age <= period for age in ages)
        near_edge = any(abs(age - period) <= TOL for age in ages)

        last_report[me] = last_full[me] = t
        last_silent[me] = NEVER
        for j, av in nbrs:
            if av is not None and 0 <= av < threshold:
                last_full[j] = t
            if av is None:
                last_silent[j] = t

        r0, c0 = divmod(me, n)
        if recent == len(nbrs):
            expect = "none nearby"
        else:
            def ok(j):
                return t - last_full[j] > period and t - last_silent[j] > period
            dist = {j: abs(j // n - r0) + abs(j % n - c0) for j in range(N) if j != me and ok(j)}
            if not dist:
                expect = "none in grid"
            else:
                best = min(dist.values())
                expect = (best, sorted(j for j, dd in dist.items() if dd == best))
            near_edge = near_edge or any(
                abs((t - last_full[j]) - period) <= TOL or abs((t - last_silent[j]) - period) <= TOL
                for j in range(N))

        logged_recent = int(d.group(1))
        dec = d.group(3)
        if dec.startswith("no nodes available nearby"):
            got = "none nearby"
        elif dec.startswith("no available node"):
            got = "none in grid"
        else:
            f = FOUND.match(dec)
            got = (int(f.group(2)), sorted(int(x) for x in re.findall(r"node(\d+)", f.group(3))))
            res["found"] += 1

        if got == expect and logged_recent == recent:
            res["ok"] += 1
        elif near_edge:
            res["boundary"] += 1
        else:
            res["wrong"].append(f"t={t:.3f}s node {me}: logged {got} (recent {logged_recent}), "
                                f"replay {expect} (recent {recent})")
    return res


def main():
    runs = sys.argv[1] if len(sys.argv) > 1 else "runs"
    logs = sorted(glob.glob(os.path.join(runs, "*", "run_*", "base_station.log")))
    if not logs:
        sys.exit(f"no base_station.log under {runs}/")
    tot = dict(runs=0, alerts=0, found=0, ok=0, boundary=0, wrong=0)
    for path in logs:
        r = check_log(path)
        tot["runs"] += 1
        for k in ("alerts", "found", "ok", "boundary"):
            tot[k] += r[k]
        tot["wrong"] += len(r["wrong"])
        for w in r["wrong"]:
            print(f"WRONG {os.path.relpath(path, runs)}: {w}")
    print(f"{tot['runs']} runs, {tot['alerts']} decisions replayed "
          f"({tot['found']} listed nearest nodes): {tot['ok']} match, "
          f"{tot['boundary']} on the 1 ms period boundary, {tot['wrong']} wrong")


if __name__ == "__main__":
    main()
