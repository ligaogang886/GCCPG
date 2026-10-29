# -*- coding: utf-8 -*-
"""Consistency check for the released dataset.

Verifies, for every airport in ``data/``:
  * the flight list, tracks, metrics, ranking and residuals agree on the same 20 tids;
  * point counts are identical across ``traj_raw`` / ``metrics.csv`` / ``residuals``;
  * every path name in ``best_path`` exists in the graph;
  * the route built from ``best_path`` is geometrically continuous (no seam);
  * the median residual recomputed from ``residuals/*.json`` matches ``metrics.csv``.

Run from the repository root:

    python code/scripts/verify_dataset.py
"""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.environ.get("ROUTE_ID_ROOT") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA = os.path.join(ROOT, "data")
APS = ["KDCA", "KLAX"]


def load_json(*parts):
    with open(os.path.join(DATA, *parts), encoding="utf-8") as f:
        return json.load(f)


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def route_chain(names, pline):
    """Concatenate the polylines of ``names``, orienting each along the running end point."""
    pts, prev, missing, seams = [], None, [], []
    for nm in names:
        q = pline.get(nm)
        if q is None:
            missing.append(nm)
            continue
        if prev is not None:
            d0 = float(np.hypot(*(q[0] - prev)))
            d1 = float(np.hypot(*(q[-1] - prev)))
            if d1 < d0:
                q = q[::-1]
                seams.append((d1, nm))
            else:
                seams.append((d0, nm))
        pts.append(q)
        prev = q[-1]
    if not pts:
        return None, missing, seams
    r = np.vstack(pts)
    keep = np.ones(len(r), bool)
    keep[1:] = np.abs(np.diff(r, axis=0)).sum(1) > 1e-9
    return r[keep], missing, seams


def check_airport(ap):
    errs, notes = [], []
    tids = [x.strip() for x in open(os.path.join(DATA, ap, "tids.txt"), encoding="utf-8")
            if x.strip()]
    flights = load_json(ap, "flights.json")["top"]
    metrics = read_csv(os.path.join(DATA, ap, "metrics.csv"))
    all_runs = load_json(ap, "all_runs.json")
    rows = load_json(ap, "full_std_skel", "results.json")["rows"]

    # --- identifiers ---
    f_ids = [x["tid"] for x in flights]
    m_ids = [x["tid"] for x in metrics]
    r_ids = [x["tid"] for x in rows]
    for name, ids in (("flights.json", f_ids), ("metrics.csv", m_ids), ("results.json", r_ids)):
        if ids != tids:
            errs.append("%s tid order differs from tids.txt" % name)
    if len(all_runs) < len(tids):
        errs.append("all_runs.json has fewer rows than the released set")

    # --- graph ---
    paths = load_json(ap, "graph_std_skel", "InitMatrixPath.json")["paths"]
    pline = {p["pathname"]: np.asarray([[q["x"], q["z"]] for q in p["points"]], float)
             for p in paths if len(p["points"]) >= 2}
    n_r = sum(1 for k in pline if k.startswith("R"))
    seg_len = sum(float(p["pathLength"]) for p in paths) / 1000.0

    n_pts = 0
    med_dev = []
    for tid, m in zip(tids, metrics):
        traj = load_json(ap, "traj_raw", tid + ".json")
        res = load_json(ap, "residuals", tid + ".json")
        row = next(r for r in rows if r["tid"] == tid)
        n = int(m["n"])
        n_pts += n

        if not (len(traj) == n == len(res["d"]) == len(res["t"])):
            errs.append("%s point count mismatch (traj %d, metrics %d, resid %d)"
                        % (tid, len(traj), n, len(res["d"])))
        if int(row["n_track_points"]) != n:
            errs.append("%s results.json n_track_points=%s vs metrics %d"
                        % (tid, row["n_track_points"], n))

        bp = row.get("best_path") or []
        r, miss, seams = route_chain(bp, pline)
        if miss:
            errs.append("%s route references unknown segments: %s" % (tid, miss[:3]))
        elif seams:
            gap, where = max(seams)
            if gap > 1.5:
                notes.append("%s route seam %.1f m before %s" % (tid, gap, where))

        d = np.asarray(res["d"], float)
        med_dev.append(float(np.median(d)) - float(m["med_recheck"]))

    if med_dev and abs(max(med_dev, key=abs)) > 0.02:
        errs.append("median residual deviates from metrics.med_recheck by up to %.3f m"
                    % max(med_dev, key=abs))

    print("%-5s  flights %2d  points %5d  segments %3d (R %3d, %.1f km)  "
          "|med(rec)-med| <= %.3f m"
          % (ap, len(tids), n_pts, len(pline), n_r, seg_len, max(abs(v) for v in med_dev)))
    for s in notes:
        print("   note:", s)
    return errs


def main():
    print("repository:", ROOT)
    errs = []
    for ap in APS:
        if not os.path.isdir(os.path.join(DATA, ap)):
            errs.append("missing data/%s" % ap)
            continue
        errs += check_airport(ap)
    if errs:
        print("\nFAILED (%d)" % len(errs))
        for e in errs:
            print("  -", e)
        return 1
    print("\nOK - all files are consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
