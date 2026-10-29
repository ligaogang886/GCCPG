# -*- coding: utf-8 -*-
"""从全量运行结果里挑出几何贴合最好的 20 条航班（发布样本的生成脚本）。

评价口径（都只针对算法输出，不含二次匹配）：
  med   : 轨迹 1 m 点 -> 识别路线 的距离中位数（越小越好）
  p95   : 同上 P95（惩罚局部偏离）
  cov15 : 距离 <=15 m 的轨迹点占比（越高越好）
  excess: med - floor，floor 为该轨迹到图的 p50；扣除图本身的下限，只留选路损失
  排除兜底：best_path 里一个匹配到的 R 都没有 ⇒ 实质是起终点最短路，不算识别成功
"""
import glob
import io
import json
import os

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.environ.get("ROUTE_ID_ROOT") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
AP = "KDCA"
SUB = "full_std_skel"
GRAPH = os.path.join(ROOT, AP, "graph_std_skel", "InitMatrixPath.json")
TRAJ = os.path.join(ROOT, AP, "traj_raw")
OUT_JSON = os.path.join(ROOT, AP, "flights.json")
OUT_TXT = os.path.join(ROOT, AP, "tids.txt")


def densify(Q, step=1.0):
    out = [Q[0]]
    for k in range(len(Q) - 1):
        a, b = Q[k], Q[k + 1]
        L = float(np.hypot(*(b - a)))
        n = max(int(L / step), 1)
        for i in range(1, n + 1):
            out.append(a + (b - a) * (i / float(n)))
    return np.asarray(out, float)


def chain(bp, pline):
    seq, prev = [], None
    for nm in bp:
        Q = pline.get(nm)
        if Q is None or len(Q) < 2:
            continue
        if prev is not None:
            d0 = float(np.hypot(*(Q[0] - prev)))
            d1 = float(np.hypot(*(Q[-1] - prev)))
            if d1 < d0:
                Q = Q[::-1]
        seq.append(Q)
        prev = Q[-1]
    if not seq:
        return None
    R = np.vstack(seq)
    keep = np.ones(len(R), bool)
    keep[1:] = np.abs(np.diff(R, axis=0)).sum(1) > 1e-9
    return R[keep]


def matched_R():
    out = {}
    for d in glob.glob(os.path.join(ROOT, AP, SUB, "_stage", "p*")):
        fs = glob.glob(os.path.join(d, "1-潜在R识别", "*.json"))
        if not fs:
            continue
        bn = os.path.basename(fs[0])
        tid = bn.split("_converted")[0].split("-潜在R识别")[0]
        try:
            out[tid] = list(json.load(open(fs[0], encoding="utf-8")).keys())
        except Exception:
            out[tid] = []
    return out


def main():
    pline = {}
    for p in json.load(open(GRAPH, encoding="utf-8"))["paths"]:
        q = np.asarray([[x["x"], x["z"]] for x in p["points"]], float)
        if len(q) >= 2:
            pline[p["pathname"]] = q

    rows = json.load(open(os.path.join(ROOT, AP, SUB, "results.json"),
                          encoding="utf-8"))["rows"]
    floor = {r["tid"]: r.get("traj_graph_p50") for r in
             json.load(open(os.path.join(ROOT, AP, "route_on_graph_skel.json"),
                            encoding="utf-8"))}
    mR = matched_R()

    # 同口径 floor：1 m 图点云对同一批轨迹点求最近距离
    g1 = json.load(open(os.path.join(ROOT, AP, "graph_std_skel",
                                     "InitMatrixPath-1m.json"), encoding="utf-8"))["paths"]
    G = np.asarray([[q["x"], q["z"]] for p in g1 for q in p["points"]], float)
    gtree = cKDTree(G)

    out = []
    for r in rows:
        bp = r.get("best_path") or []
        if not bp:
            continue
        tid = r["tid"]
        f = os.path.join(TRAJ, tid + ".json")
        if not os.path.exists(f):
            continue
        d = json.load(open(f, encoding="utf-8"))
        T = np.asarray([[x["properties"]["unity_x"], x["properties"]["unity_z"]]
                        for x in d if x["properties"].get("unity_x") is not None],
                       float)
        if len(T) < 10:
            continue
        R = chain(bp, pline)
        if R is None or len(R) < 2:
            continue
        dist = cKDTree(densify(R, 1.0)).query(T)[0]
        fl = float(np.median(gtree.query(T)[0]))
        rs, rl = set(mR.get(tid, [])), set(bp)
        out.append({
            "tid": tid,
            "n": int(len(T)),
            "npaths": len(bp),
            "ncand": r.get("n_candidates"),
            "med": round(float(np.median(dist)), 3),
            "p95": round(float(np.percentile(dist, 95)), 3),
            "max": round(float(dist.max()), 2),
            "cov15": round(float((dist <= 15).mean() * 100), 1),
            "floor": round(fl, 3),
            "excess": round(float(np.median(dist)) - fl, 3),
            "fallback": not bool(rs & rl),
            "nR": len(rs),
        })

    print("成功 %d 条，可打分 %d 条" % (len(rows) - sum(1 for r in rows if not r.get("best_path")), len(out)))
    fb = [x for x in out if x["fallback"]]
    print("其中兜底（best_path 无匹配 R）%d 条" % len(fb))

    good = [x for x in out if not x["fallback"]]
    print("非兜底 %d 条" % len(good))
    for c, m, p in ((98, 2.5, 8), (95, 2.5, 15), (90, 2.5, 30)):
        print("  cov15>=%d%% & med<=%.1f & p95<=%d : %d 条"
              % (c, m, p, sum(1 for x in good if x["cov15"] >= c and x["med"] <= m and x["p95"] <= p)))

    # 排序：整体贴合（med + 0.02*P95），只保留「整条都被路线覆盖」的
    strict = [x for x in good if x["cov15"] >= 98 and x["med"] <= 2.5 and x["p95"] <= 15]
    if len(strict) < 20:
        strict = [x for x in good if x["cov15"] >= 95 and x["med"] <= 2.5 and x["p95"] <= 30]
    strict.sort(key=lambda x: (x["med"] + 0.02 * x["p95"], x["med"]))
    top = strict[:20]

    print("\n%-3s %-26s %5s %4s %4s %6s %6s %6s %5s" %
          ("#", "tid", "med", "p95", "cov", "excess", "paths", "cand", "R"))
    for i, x in enumerate(top, 1):
        print("%-3d %-26s %5.2f %5.1f %4.0f %6.2f %5d %6s %5d" %
              (i, x["tid"], x["med"], x["p95"], x["cov15"], x["excess"],
               x["npaths"], x["ncand"], x["nR"]))

    allr = sorted(out, key=lambda x: x["med"])
    json.dump(allr, open(os.path.join(ROOT, AP, "all_runs.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    meds = np.array([x["med"] for x in good])
    print("\n全部 %d 条非兜底：med 中位 %.2f / P25 %.2f / P75 %.2f / P90 %.2f"
          % (len(good), np.median(meds), np.percentile(meds, 25),
             np.percentile(meds, 75), np.percentile(meds, 90)))
    print("med<=1.1 的 %d 条；med>15 的 %d 条" % ((meds <= 1.1).sum(), (meds > 15).sum()))
    json.dump(top, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    io.open(OUT_TXT, "w", encoding="utf-8", newline="\n").write(
        "\n".join(x["tid"] for x in top) + "\n")
    print("\nwrote", OUT_JSON, OUT_TXT)


if __name__ == "__main__":
    main()
