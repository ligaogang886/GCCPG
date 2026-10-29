# -*- coding: utf-8 -*-
"""
轨迹投射查看器（KLAX_cn20 绘画标准）
====================================
视觉与交互完全对齐 SHOWCASE/viewer_KLAX_cn20.html：
  · 衬线字体（Times New Roman / SimSun）、纯白底、1px 黑细边画框
  · 左侧 266px 侧栏：Airport / Flight / Layers / Statistics / View
  · 调色板 net #79706E · runway #4D4A49 · R #1F77B4 · track #2CA02C
              · route #FF7F0E · deviation #D62728
  · 画布内左下角比例尺、右上角图例框、鼠标悬停显示路径名
  · 旋转滑杆 / Fit to flight / Whole airport / Export PNG / ←→ 切换航班

内容
  · 灰线   滑行网络（全部路径）
  · 粗线   跑道
  · 蓝线   转弯段 R
  · 绿线   轨迹连线（淡）
  · 绿点   轨迹原始报点
  · 橙线   投射到图上的路径（逐点最近点去重）
  · 红圈   偏差 > 15 m 的轨迹点
  · 红虚线 轨迹点 → 投射垂足
"""
import os
import json
import glob
import random
import argparse
import statistics
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.environ.get("ROUTE_ID_ROOT") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

LAYOUT = {
    "KLAX": "Four parallel runways + perimeter loop",
    "KDCA": "Potomac riverfront, two crossing runways",
    "KSEA": "Three parallel runways",
    "KJFK": "Four runways, bay shoreline",
    "KEWR": "Three runways, harbor reclamation",
}


def load_graph(ap, sub):
    gd = os.path.join(ROOT, ap, sub)
    with open(os.path.join(gd, "InitMatrixPath.json"), encoding="utf-8") as f:
        paths = json.load(f)["paths"]
    meta = {}
    mp = os.path.join(gd, "path_meta.json")
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as f:
            meta = json.load(f)

    names, kinds, lens, isrwy, lines, rwy = [], [], [], [], [], []
    for p in paths:
        pts = p["points"]
        if len(pts) < 2:
            continue
        nm = p["pathname"]
        m = meta.get(nm, {})
        flat = []
        for q in pts:
            flat.append(round(q["x"], 1))
            flat.append(round(q["z"], 1))
        names.append(nm)
        kinds.append("R" if m.get("kind") == "R" else "N")
        lens.append(int(round(m.get("length", p.get("pathLength", 0)) or 0)))
        isr = bool(m.get("is_runway", False))
        isrwy.append(1 if isr else 0)
        lines.append(flat)
        if isr:
            rwy.append(flat)

    # 1 m 采样点（吸附用）
    with open(os.path.join(gd, "InitMatrixPath-1m.json"), encoding="utf-8") as f:
        p1 = json.load(f)["paths"]
    P, own = [], []
    for p in p1:
        for q in p["points"]:
            P.append((q["x"], q["z"]))
            own.append(p["pathname"])
    A = np.asarray(P, float)
    own = np.asarray(own)
    return dict(names=names, kinds=kinds, lens=lens, isrwy=isrwy,
                lines=lines, rwy=rwy), A, own, len(paths)


def runs_of(names, min_pts=2):
    out, i = [], 0
    while i < len(names):
        j = i
        while j + 1 < len(names) and names[j + 1] == names[i]:
            j += 1
        if j - i + 1 >= min_pts:
            out.append((names[i], j - i + 1))
        i = j + 1
    return out


def load_affine(ap):
    """OSM 经纬度 -> Amelia 局部坐标的仿射（与 build_paths_osm 同一套标定）。"""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build_paths_osm import fit_transform, make_proj
    cache = os.path.join(ROOT, ap, "osm", "_affine.json")
    if os.path.exists(cache):
        tf = json.load(open(cache, encoding="utf-8"))
    else:
        tf = fit_transform(ap)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        json.dump(tf, open(cache, "w", encoding="utf-8"), indent=1)
    return make_proj(tf), tf


BG_KIND = {"terminal": 3, "hangar": 2, "building": 1, "apron": 0}


def load_bg(osm_path, to_xy):
    """从 OSM 取机坪 / 航站楼 / 建筑轮廓，投到与路网同一坐标系。"""
    import xml.etree.ElementTree as ET
    root = ET.parse(osm_path).getroot()
    ll = {n.get("id"): (float(n.get("lat")), float(n.get("lon")))
          for n in root.findall("node")}
    polys = []
    for w in root.findall("way"):
        ids = [nd.get("ref") for nd in w.findall("nd")]
        if len(ids) < 4 or ids[0] != ids[-1] or any(i not in ll for i in ids):
            continue
        tg = {x.get("k"): x.get("v") for x in w.findall("tag")}
        aw, bd = tg.get("aeroway"), tg.get("building")
        if aw == "terminal":
            kind = "terminal"
        elif aw == "hangar":
            kind = "hangar"
        elif aw == "apron":
            kind = "apron"
        elif bd in ("yes", "terminal", "hangar", "commercial", "government",
                    "parking", "transportation"):
            kind = "terminal" if bd in ("terminal", "transportation") else "building"
        else:
            continue
        lat = np.array([ll[i][0] for i in ids], float)
        lon = np.array([ll[i][1] for i in ids], float)
        P = np.c_[*to_xy(lat, lon)]
        area = abs(np.dot(P[:-1, 0], P[1:, 1]) - np.dot(P[1:, 0], P[:-1, 1])) / 2.0
        if area < 60:            # 去掉喷淋小方块噪声
            continue
        polys.append({"k": kind, "a": round(float(area), 0),
                      "p": [[round(float(u), 1), round(float(v), 1)] for u, v in P]})
    polys.sort(key=lambda d: (BG_KIND[d["k"]], -d["a"]))
    return polys


POT_STEP = "1-潜在R识别"


def index_potential_r(stage_dir):
    """扫 _stage/<pNNN>/1-潜在R识别/*.json，返回 tid -> {R 路径名: 阶段1命中点数}。

    这是算法阶段 1 的输出（潜在 R 集合），与最终 best_path 里实际用到的 R 是两回事：
    前者是候选，后者是选定。
    """
    out = {}
    if not stage_dir or not os.path.isdir(stage_dir):
        return out
    for g in sorted(glob.glob(os.path.join(stage_dir, "*", POT_STEP, "*.json"))):
        tid = os.path.basename(g).split("_converted")[0]
        try:
            d = json.load(open(g, encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, dict):
            out[tid] = {k: int(v) for k, v in d.items()
                        if isinstance(v, (int, float)) and int(v) > 0}
    return out


def build_path_chain(bp, pline):
    """把 best_path 里的路径串成一条连续折线。

    相邻两条路径必须共享端点（同一条图边序列），否则记为接缝；返回 (折线, 接缝距离列表, 缺失路径)。
    """
    seq, gaps, missing = [], [], []
    prev = None
    for nm in bp:
        Q = pline.get(nm)
        if Q is None or len(Q) < 2:
            missing.append(nm)
            continue
        if prev is not None:
            d0 = float(np.hypot(*(Q[0] - prev)))
            d1 = float(np.hypot(*(Q[-1] - prev)))
            if d1 < d0:
                Q = Q[::-1]
            gaps.append(min(d0, d1))
        seq.append(Q)
        prev = Q[-1]
    if not seq:
        return None, [], missing
    R = np.vstack(seq)
    keep = np.ones(len(R), bool)
    keep[1:] = np.abs(np.diff(R, axis=0)).sum(1) > 1e-9
    return R[keep], gaps, missing


def densify(Q, step=1.0):
    """按固定步长重采样折线，供「点到路线」距离统计用"""
    out = [Q[0]]
    for k in range(len(Q) - 1):
        a, b = Q[k], Q[k + 1]
        L = float(np.hypot(*(b - a)))
        n = max(int(L / step), 1)
        for i in range(1, n + 1):
            out.append(a + (b - a) * (i / float(n)))
    return np.asarray(out, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ap", default="KLAX")
    ap.add_argument("--graph", default=None)
    ap.add_argument("--traj", default=None)
    ap.add_argument("--results", default=None, help="实验 results.json（取成功率与平均误差）")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--tids", default=None, help="只画这些 tid（每行一个）")
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--decim", type=int, default=1, help="轨迹抽稀步长")
    ap.add_argument("--out", default=None)
    ap.add_argument("--bg-osm", default=None,
                    help="底图 OSM（默认取 <AP>/osm/<AP>_std_healed.osm 或 <AP>.osm）")
    ap.add_argument("--no-bg", action="store_true", help="不画机坪/航站楼底图")
    ap.add_argument("--route", choices=["graph", "proj"], default="graph",
                    help="橙色线：graph=识别路线（best_path 串联，逐点在图）；proj=最近图点投射")
    ap.add_argument("--stage", default=None,
                    help="阶段目录（读 1-潜在R识别）；默认 <AP>/full_std_skel/_stage")
    a = ap.parse_args()

    if a.graph is None:
        a.graph = "graph_cn2" if a.ap == "KLAX" else "graph_std_skel"
    if a.traj is None:
        a.traj = "traj_org" if a.ap == "KLAX" else "traj_raw"
    if a.out is None:
        a.out = os.path.join(ROOT, "SHOWCASE", "viewer_%s_cn_proj%d.html" % (a.ap, a.n))

    net, A, own, npaths = load_graph(a.ap, a.graph)
    tree = cKDTree(A)
    nR = sum(1 for k in net["kinds"] if k == "R")
    print("[graph] %-16s 路径 %4d  1m点 %6d  R %d" % (a.graph, npaths, len(A), nR))

    # 路径名 -> 折线（用于把 best_path 串成识别路线）
    pline = {}
    for _n, _flat in zip(net["names"], net["lines"]):
        if len(_flat) >= 4:
            pline[_n] = np.asarray(_flat, float).reshape(-1, 2)

    # 阶段 1 的潜在 R 集合（tid -> {R 名: 命中点数}）
    if a.stage is None:
        a.stage = os.path.join(ROOT, a.ap, "full_std_skel", "_stage")
    potmap = index_potential_r(a.stage)
    print("[potR] %s  航班 %d" % (a.stage, len(potmap)))

    files = sorted(glob.glob(os.path.join(ROOT, a.ap, a.traj, "*.json")))
    if not files:
        raise SystemExit("没有轨迹：%s" % os.path.join(ROOT, a.ap, a.traj))
    if a.tids:
        want = [x.strip() for x in open(a.tids, encoding="utf-8") if x.strip()]
        idx = {os.path.basename(f)[:-5]: f for f in files}
        pick = [idx[t] for t in want if t in idx]
    else:
        rng = random.Random(a.seed)
        pick = rng.sample(files, min(a.n, len(files)))
    pick.sort()

    # 成功率 / 平均误差（可选）
    n_ok = n_all = None
    mean_err = None
    bytid = {}
    if a.results and os.path.exists(a.results):
        d = json.load(open(a.results, encoding="utf-8"))
        rows = d.get("rows", [])
        n_all = len(rows)
        n_ok = sum(1 for r in rows if r.get("best_path"))
        for r in rows:
            bytid[r["tid"]] = {"flio": r.get("flio"),
                               "ncand": r.get("n_candidates"),
                               "ok": bool(r.get("best_path")),
                               "bp": r.get("best_path") or []}

    flights = []
    for f in pick:
        tid = os.path.basename(f)[:-5]
        d = json.load(open(f, encoding="utf-8"))
        P = np.asarray([[x["properties"]["unity_x"], x["properties"]["unity_z"]] for x in d
                        if x["properties"].get("unity_x") is not None], float)
        if len(P) < 10:
            continue
        T = P[::a.decim] if a.decim > 1 else P
        flio = None
        for x in d[:3]:
            flio = x.get("properties", {}).get("flio")
            if flio:
                break
        dist, idx = tree.query(T)
        foot = A[idx]
        # 投射折线：相邻投射点相同则合并
        keep = [0]
        for i in range(1, len(foot)):
            if foot[i][0] != foot[keep[-1]][0] or foot[i][1] != foot[keep[-1]][1]:
                keep.append(i)
        route = foot[keep]
        # 橙色线：优先用识别路线（best_path 串成连通折线，必在图）；无结果时退回投射折线
        bp = (bytid.get(tid) or {}).get("bp") or []
        rkind, gaps, rdist = "proj", [], dist
        if a.route == "graph":
            # 未识别（无 best_path）就不画橙色线：宁可空，也不拿投射折线冒充路线
            route = None
            rkind, rdist = "none", np.full(len(T), np.nan)
            if bp:
                Rt, gaps, _miss = build_path_chain(bp, pline)
                if Rt is not None and len(Rt) > 1:
                    route = Rt
                    rkind = "graph"
                    rdist = cKDTree(densify(Rt, 0.5)).query(T)[0]
        names = own[idx]
        segs = runs_of(names)

        def plen(Q):
            if len(Q) < 2:
                return 0.0
            return float(np.hypot(np.diff(Q[:, 0]), np.diff(Q[:, 1])).sum())

        tlen = plen(T)
        rlen = plen(route) if route is not None else 0.0
        nN = sum(1 for n, _ in segs if not n.startswith("R"))
        nsegR = sum(1 for n, _ in segs if n.startswith("R"))
        # 潜在 R（阶段 1 输出）：名称 + 命中点数 + 图上折线
        pot = potmap.get(tid) or {}
        potr = []
        for nm, cnt in sorted(pot.items(), key=lambda x: -x[1]):
            q = pline.get(nm)
            if q is None:
                continue
            potr.append([nm, int(cnt),
                         [[round(float(x), 1), round(float(z), 1)] for x, z in q]])
        flights.append({
            "id": tid,
            "trk": [[round(float(x), 1), round(float(z), 1)] for x, z in T],
            "proj": [[round(float(x), 1), round(float(z), 1)] for x, z in foot],
            "route": ([[round(float(x), 1), round(float(z), 1)] for x, z in route]
                      if route is not None else None),
            "err": [round(float(v) * 10, 1) for v in dist],   # 0.1 m，与标准 view 一致
            "rerr": [round(float(v) * 10, 1) for v in rdist],
            "segs": [[n, c] for n, c in segs[:40]],
            "potr": potr,
            "exp": dict(bytid.get(tid, {}), **({"flio": flio} if flio and
                                               not bytid.get(tid, {}).get("flio") else {})),
            "m": {
                "npts": int(len(T)),
                "nseg": len(segs),
                "nN": nN,
                "nR": nsegR,
                "mean": round(float(dist.mean()), 2),
                "med": round(float(np.median(dist)), 2),
                "p95": round(float(np.percentile(dist, 95)), 2),
                "cov15": round(float((dist <= 15).mean() * 100.0), 1),
                "rlen": round(rlen, 0),
                "tlen": round(tlen, 0),
                "ratio": round(rlen / max(tlen, 1e-6), 2),
                "nover": int((dist > 15).sum()),
                "rkind": rkind,
                "npaths": len(bp),
                "gap": round(float(max(gaps) if gaps else 0.0), 3),
                "medr": (None if rkind != "graph" else round(float(np.median(rdist)), 2)),
                "p95r": (None if rkind != "graph" else round(float(np.percentile(rdist, 95)), 2)),
                "cov15r": (None if rkind != "graph" else round(float((rdist <= 15).mean() * 100.0), 1)),
                "noverr": 0 if rkind != "graph" else int((rdist > 15).sum()),
                "npotr": len(potr),
                "npotrpts": int(sum(c for _, c, _ in potr)),
            },
        })
        print("  %-26s n=%4d  p50=%5.2f p95=%6.2f cov15=%5.1f%%  段=%2d/%2d"
              % (tid, len(T), np.median(dist), np.percentile(dist, 95),
                 (dist <= 15).mean() * 100, nN, nsegR))

    bg, bg_src = [], ""
    if not a.no_bg:
        # 原始全量 OSM 面要素更全（含 building），优先；清洗版只保留 aeroway
        cands = [os.path.join(ROOT, a.ap + ".osm"),
                 os.path.join(ROOT, a.ap, "osm", "%s_std_healed.osm" % a.ap)]
        osm = a.bg_osm or next((c for c in cands if os.path.exists(c)), None)
        if osm and os.path.exists(osm):
            to_xy, tf = load_affine(a.ap)
            bg = load_bg(osm, to_xy)
            bg_src = os.path.basename(osm)
            kinds = {}
            for q in bg:
                kinds[q["k"]] = kinds.get(q["k"], 0) + 1
            print("[bg] %s  面 %d  %s  标定残差 mean %.2f m"
                  % (bg_src, len(bg), kinds, tf["res_mean"]))
        else:
            print("[bg] 未找到 OSM，跳过底图")

    data = {
        "bg": bg,
        "meta": [{"code": a.ap, "layout": LAYOUT.get(a.ap, ""),
                  "n_ok": n_ok, "n_all": n_all, "mean": mean_err,
                  "nnet": npaths, "nR": nR}],
        "nets": {a.ap: net},
        "flights": {a.ap: flights},
    }

    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False,
                                                   separators=(",", ":")))
    html = html.replace("__APT__", a.ap).replace("__GRAPH__", a.graph)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)
    print("[out] %s  %d 航班  %.0f KB" % (a.out, len(flights), len(html) / 1024))


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Taxi Path Projection · __APT__</title>
<style>
  :root{
    --net:#79706E;
    --runway:#4D4A49;
    --track:#2CA02C;
    --route:#FF7F0E;
    --pot:#1F77B4;
    --potr:#6A1B9A;
    --over:#D62728;
    --ink:#1A1A1A;
    --muted:#5A5A5A;
    --line:#111111;
  }
  *{box-sizing:border-box;}
  html,body{margin:0;height:100%;background:#fff;color:var(--ink);
    font-family:"Times New Roman","SimSun","Songti SC",serif;}
  #app{display:flex;height:100%;}

  #side{width:266px;flex:0 0 266px;background:#fff;border-right:1px solid var(--line);
    padding:18px 18px 14px;overflow-y:auto;}
  h1{font-size:15px;margin:0 0 2px;font-weight:700;letter-spacing:.2px;}
  .sub{font-size:11px;color:var(--muted);margin-bottom:16px;font-style:italic;line-height:1.5;}

  .cap{font-size:10px;letter-spacing:1.2px;color:var(--muted);text-transform:uppercase;
    margin:14px 0 6px;border-bottom:1px solid #DDD;padding-bottom:4px;}

  select,button{width:100%;padding:6px 8px;font-size:13px;border:1px solid var(--line);
    border-radius:2px;background:#fff;color:var(--ink);font-family:inherit;}
  select{font-size:12.5px;}
  button{cursor:pointer;}
  button:hover{background:#F4F4F4;}

  .layer{display:flex;align-items:center;gap:8px;padding:4px 0;font-size:12.5px;cursor:pointer;}
  .layer input{accent-color:#1A1A1A;margin:0;}
  .swatch{width:16px;height:3px;border-radius:0;display:inline-block;flex:0 0 16px;}
  .swatch.dot{width:8px;height:8px;border-radius:50%;flex-basis:8px;}

  .stats{margin-top:6px;font-size:12px;line-height:1.85;color:var(--muted);}
  .stats b{color:var(--ink);font-weight:700;}
  .stats .ref{border-top:1px dotted #CCC;margin-top:4px;padding-top:4px;}
  .stats .warn{color:#B3261E;}

  .rot{display:flex;align-items:center;gap:8px;margin:2px 0 10px;}
  .rot input[type=range]{flex:1 1 auto;width:auto;padding:0;border:0;background:none;
    accent-color:#1A1A1A;height:14px;cursor:pointer;}
  .rot span{font-size:11.5px;color:var(--muted);min-width:36px;text-align:right;
    font-variant-numeric:tabular-nums;}

  .hint{margin-top:14px;font-size:10.5px;color:#8A8A8A;line-height:1.6;font-style:italic;}

  .sty-l{font-size:11px;color:var(--muted);flex:0 0 74px;min-width:74px;text-align:left;}
  .sty-v{font-size:11px;color:var(--muted);flex:0 0 34px;min-width:34px;text-align:right;
    font-variant-numeric:tabular-nums;}
  /* range 的 min-content 约 129px，必须显式放开，否则把数值挤出侧栏 */
  .rot input.sty-r{flex:1 1 0;min-width:0;}

  #wrap{position:relative;flex:1;min-width:0;padding:14px;background:#fff;}
  #frame{position:absolute;inset:14px;border:1px solid var(--line);pointer-events:none;}
  canvas{display:block;width:100%;height:100%;cursor:grab;}
  canvas.drag{cursor:grabbing;}

  #tooltip{position:absolute;pointer-events:none;padding:2px 6px;font-size:11.5px;
    background:#fff;color:var(--ink);border:1px solid var(--line);border-radius:2px;
    transform:translate(-50%,-135%);display:none;white-space:nowrap;
    font-family:"Times New Roman","SimSun",serif;}

</style>
</head>
<body>
<div id="app">
  <div id="side">
    <h1>Taxi Path Projection</h1>
    <div class="sub">Amelia-10 &middot; __APT__ &middot; __GRAPH__</div>

    <div class="cap">Airport</div>
    <select id="apt"></select>

    <div class="cap">Flight</div>
    <select id="flight"></select>

    <div class="cap">Layers</div>
    <label class="layer"><input type="checkbox" id="ly-bg" checked><span class="swatch" style="background:#DCD9D7;border:1px solid #A9A3A1"></span>Apron / Terminal</label>
    <label class="layer"><input type="checkbox" id="ly-net" checked><span class="swatch" style="background:var(--net)"></span>Taxiway network</label>
    <label class="layer"><input type="checkbox" id="ly-rwy" checked><span class="swatch" style="background:var(--runway)"></span>Runways</label>
    <label class="layer"><input type="checkbox" id="ly-node" checked><span class="swatch dot" style="background:#fff;border:1px solid #111"></span>Path endpoints</label>
    <label class="layer"><input type="checkbox" id="ly-r"><span class="swatch" style="background:var(--pot)"></span>Curved segments (R)</label>
    <label class="layer"><input type="checkbox" id="ly-potr"><span class="swatch" style="background:var(--potr);height:4px"></span>Potential R (stage 1)</label>
    <label class="layer"><input type="checkbox" id="ly-line"><span class="swatch" style="background:var(--track);opacity:.55"></span>Trajectory line</label>
    <label class="layer"><input type="checkbox" id="ly-track" checked><span class="swatch dot" style="background:var(--track)"></span>Trajectory points</label>
    <label class="layer"><input type="checkbox" id="ly-best" checked><span class="swatch" style="background:var(--route)"></span>Identified route</label>
    <label class="layer"><input type="checkbox" id="ly-proj"><span class="swatch" style="background:#B08A78"></span>Projection (snap)</label>
    <label class="layer"><input type="checkbox" id="ly-over"><span class="swatch dot" style="background:#fff;border:1.5px solid var(--over)"></span>Deviation &gt; 15 m</label>
    <label class="layer"><input type="checkbox" id="ly-link"><span class="swatch" style="background:repeating-linear-gradient(90deg,#D62728 0 3px,transparent 3px 6px)"></span>Deviation links</label>

    <div class="cap">Style</div>
    <div class="rot"><span class="sty-l">Points</span>
      <input class="sty-r" type="range" id="sty-pt" min="0.6" max="9" step="0.2" value="3.0">
      <span class="sty-v" id="sty-pt-v">3.0</span></div>
    <div class="rot"><span class="sty-l">Points step</span>
      <input class="sty-r" type="range" id="sty-stp" min="1" max="25" step="1" value="1">
      <span class="sty-v" id="sty-stp-v">1</span></div>
    <div class="rot"><span class="sty-l">Route line</span>
      <input class="sty-r" type="range" id="sty-bst" min="0.6" max="12" step="0.2" value="4.0">
      <span class="sty-v" id="sty-bst-v">4.0</span></div>
    <div class="rot"><span class="sty-l">Pot. R text</span>
      <input class="sty-r" type="range" id="sty-pot" min="6" max="44" step="1" value="19">
      <span class="sty-v" id="sty-pot-v">19</span></div>
    <div class="rot"><span class="sty-l">Legend</span>
      <input class="sty-r" type="range" id="sty-lg" min="0.8" max="4" step="0.1" value="2.0">
      <span class="sty-v" id="sty-lg-v">2.00</span></div>

    <div class="cap">Statistics</div>
    <div class="stats" id="stats"></div>

    <div class="cap">View</div>
    <div class="rot">
      <input type="range" id="rot" min="0" max="360" step="1" value="0">
      <span id="rotVal">0&deg;</span>
    </div>
    <button id="reset">Fit to flight</button>
    <button id="fitall" style="margin-top:4px">Whole airport</button>
    <div style="height:6px"></div>
    <button id="png">Export PNG</button>
    <button id="png2" style="margin-top:4px">Export PNG (2&times;)</button>
    <button id="svg" style="margin-top:4px">Export SVG (vector)</button>

    <div class="hint">Scroll to zoom &middot; drag to pan &middot; double-click to reset<br>
      Hover network to see path name &middot; &larr;/&rarr; step through flights</div>
  </div>

  <div id="wrap">
    <canvas id="cv"></canvas>
    <div id="frame"></div>
    <div id="tooltip"></div>
  </div>
</div>

<script>
const DATA = __DATA__;

const cv  = document.getElementById("cv");
let   ctx = cv.getContext("2d");   // 导出 SVG 时会临时换成录制上下文
const wrap= document.getElementById("wrap");
const tip = document.getElementById("tooltip");

let dpr = window.devicePixelRatio || 1;

const ctr = {x:0, y:0};
let view = {scale:1, ox:0, oz:0, rot:0, c:1, s:0};

const ZOOM = Math.max(0.2, Math.min(40, parseFloat(new URLSearchParams(location.search).get("zoom") || "1")));

let current = null;
let curApt  = null;
let bbox = null;
const CACHE = {};
const BGC = {apron:["#E4E2E1","#ADA7A5"], building:["#D8D5D3","#ADA7A5"],
             hangar:["#57534F","#3E3B39"], terminal:["#2B2B2B","#2B2B2B"]};

function aptData(code){
  if(CACHE[code]) return CACHE[code];
  const n = DATA.nets[code];
  const nameIdx = new Map();
  n.lines.forEach((l, i) => nameIdx.set(n.names[i], i));
  // 几何直接以坐标数组保存（不再用 Path2D）：每帧重建路径，SVG 导出才能复用同一份绘制代码
  const o = {net:n, nameIdx, bg:(DATA.bg || []).map(q => ({k:q.k, p:q.p}))};
  CACHE[code] = o;
  return o;
}

function syncRot(){ view.c = Math.cos(view.rot); view.s = Math.sin(view.rot); }

function w2s(x,z){
  const dx=(x-view.ox)*view.scale, dz=(z-view.oz)*view.scale;
  return [ctr.x + dx*view.c - dz*view.s, ctr.y + dx*view.s + dz*view.c];
}
function s2w(X,Y){
  const u=(X-ctr.x)/view.scale, v=(Y-ctr.y)/view.scale;
  return [view.ox + u*view.c + v*view.s, view.oz - u*view.s + v*view.c];
}

// nolegend: 底图类图层（机坪/航站楼、滑行道、跑道、节点）不进入图例，只在左侧 Layers 里出现
const LEGEND_ITEMS = [
  {id:"ly-bg",    label:"Apron / Terminal",     nolegend:true, sw:{t:"rect", c:"#E4E2E1", ec:"#ADA7A5"}},
  {id:"ly-net",   label:"Taxiway",              nolegend:true, sw:{t:"line", c:"#79706E", h:1.8}},
  {id:"ly-rwy",   label:"Runway",               nolegend:true, sw:{t:"line", c:"#4D4A49", h:5.0}},
  {id:"ly-node",  label:"Path endpoint",        nolegend:true, sw:{t:"dot",  c:"#FFFFFF", ec:"#111111"}},
  {id:"ly-r",     label:"Curved segment (R)",   sw:{t:"line", c:"#1F77B4", h:2.4}},
  {id:"ly-potr",  label:"Potential R (stage 1)",sw:{t:"line", c:"#6A1B9A", h:4.0}},
  {id:"ly-line",  label:"Trajectory line",      sw:{t:"line", c:"#2CA02C", h:2.0, alpha:0.55}},
  {id:"ly-track", label:"Trajectory point",     sw:{t:"dot",  c:"#2CA02C"}},
  {id:"ly-best",  label:"Identified route",     sw:{t:"line", c:"#FF7F0E", h:3.4}},
  {id:"ly-proj",  label:"Projection (snap)",    sw:{t:"line", c:"#B08A78", h:1.6, dash:[5,4]}},
  {id:"ly-over",  label:"Deviation > 15 m",     sw:{t:"dot",  c:"none",    ec:"#D62728"}},
  {id:"ly-link",  label:"Deviation link",       sw:{t:"line", c:"#D62728", h:1.2, dash:[2,3], alpha:0.75}}
];

// ---- 可调外观（左侧 Style 面板实时生效，PNG / SVG 导出同步） ----------------
//   pt   轨迹点半径(CSS px)   best 识别路线线宽(px)
//   potr 潜在 R 标签字号(px)  lg   图例缩放系数（论文插图还会缩到 ~60%，默认 2x）
const STY = {pt:3.0, best:4.0, potr:19.0, lg:2.0, step:1};
let LGS = STY.lg;
function LGF(){ return (11*LGS) + 'px "Times New Roman","SimSun",serif'; }
function LGT(){ return (9*LGS)  + 'px "Times New Roman","SimSun",serif'; }

function legendItems(){
  const out = [];
  for(const it of LEGEND_ITEMS){
    if(it.nolegend) continue;
    const cb = document.getElementById(it.id);
    if(!cb || !cb.checked) continue;
    if(it.id === "ly-best" && !(current && current.route)) continue;
    if(it.id === "ly-proj" && !(current && current.proj)) continue;
    if(it.id === "ly-potr" && !(current && current.potr && current.potr.length)) continue;
    if((it.id === "ly-over" || it.id === "ly-link") && !(current && current.route)) continue;
    out.push(it);
  }
  return out;
}

function drawSwatch(sw, x, y, w){
  ctx.save();
  ctx.globalAlpha = sw.alpha === undefined ? 1 : sw.alpha;
  if(sw.t === "dot"){
    ctx.beginPath(); ctx.arc(x + w/2, y, 4.0*LGS, 0, Math.PI*2);
    if(sw.c !== "none"){ ctx.fillStyle = sw.c; ctx.fill(); }
    if(sw.ec){ ctx.strokeStyle = sw.ec; ctx.lineWidth = 1.4*LGS; ctx.stroke(); }
  } else if(sw.t === "rect"){
    ctx.beginPath();
    ctx.moveTo(x, y-5*LGS); ctx.lineTo(x+w, y-5*LGS);
    ctx.lineTo(x+w, y+5*LGS); ctx.lineTo(x, y+5*LGS);
    ctx.closePath();
    ctx.fillStyle = sw.c; ctx.fill();
    if(sw.ec){ ctx.strokeStyle = sw.ec; ctx.lineWidth = 1*LGS; ctx.stroke(); }
  } else {
    ctx.setLineDash(sw.dash ? sw.dash.map(v => v*LGS) : []);
    ctx.beginPath();
    ctx.moveTo(x, y); ctx.lineTo(x+w, y);
    ctx.strokeStyle = sw.c; ctx.lineWidth = (sw.h || 2)*LGS; ctx.lineCap = "butt";
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.restore();
}

// 图例画进 canvas（而不是 HTML 浮层），这样导出的 PNG / SVG 都自带图例
function drawLegend(){
  const items = legendItems();
  if(!items.length) return;
  ctx.save();
  ctx.font = LGF(); ctx.textAlign = "left"; ctx.textBaseline = "middle";
  let wmax = 0;
  for(const it of items) wmax = Math.max(wmax, ctx.measureText(it.label).width);
  const padX = 11*LGS, padY = 8*LGS, swW = 20*LGS, gap = 9*LGS;
  const rowH = 17*LGS, headH = 13*LGS;
  const bw = Math.ceil(padX*2 + swW + gap + wmax);
  const bh = Math.ceil(padY*2 + headH + items.length*rowH);
  const x1 = cv.clientWidth - 26, x0 = x1 - bw, y0 = 26;
  ctx.globalAlpha = 1;
  ctx.fillStyle = "rgba(255,255,255,.94)";
  ctx.strokeStyle = "#DCD8D6"; ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(x0, y0); ctx.lineTo(x1, y0);
  ctx.lineTo(x1, y0+bh); ctx.lineTo(x0, y0+bh);
  ctx.closePath();
  ctx.fill(); ctx.stroke();
  ctx.font = LGT(); ctx.fillStyle = "#8B8582";
  ctx.textAlign = "left"; ctx.textBaseline = "middle";
  ctx.fillText("LEGEND", x0+padX, y0+padY+headH*0.5);
  let y = y0 + padY + headH + rowH*0.5;
  for(const it of items){
    drawSwatch(it.sw, x0+padX, y, swW);
    ctx.font = LGF(); ctx.fillStyle = "#33302E";
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText(it.label, x0+padX+swW+gap, y);
    y += rowH;
  }
  ctx.restore();
}

function polyline(flat,color,widthPx,alpha){
  if(flat.length<4) return;
  ctx.globalAlpha = alpha===undefined?1:alpha;
  ctx.beginPath();
  let p = w2s(flat[0], flat[1]);
  ctx.moveTo(p[0], p[1]);
  for(let i=2;i<flat.length;i+=2){
    p = w2s(flat[i], flat[i+1]);
    ctx.lineTo(p[0], p[1]);
  }
  ctx.strokeStyle=color; ctx.lineWidth=widthPx;
  ctx.lineCap="round"; ctx.lineJoin="round"; ctx.stroke();
  ctx.globalAlpha=1;
}

// ---------------------------------------------------------------- 路径与矢量导出
function pathPoly(pts){                    // 闭合多边形：pts = [[x,z], ...]
  ctx.beginPath();
  if(!pts || pts.length < 3) return false;
  ctx.moveTo(pts[0][0], pts[0][1]);
  for(let i=1;i<pts.length;i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.closePath();
  return true;
}

function pathLines(lines, kinds, wantR){   // 多段折线：lines = [[x,z,x,z,...], ...]
  ctx.beginPath();
  if(!lines) return;
  for(let i=0;i<lines.length;i++){
    const l = lines[i];
    if(!l || l.length < 4) continue;
    if(kinds && wantR !== undefined && ((kinds[i] === "R") !== wantR)) continue;
    ctx.moveTo(l[0], l[1]);
    for(let k=2;k<l.length;k+=2) ctx.lineTo(l[k], l[k+1]);
  }
}

function svgEsc(s){
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

// 录制型 2D 上下文：只实现 draw() 用到的那些 API，把每次绘制转成 SVG 元素
function SvgRec(W,H){
  this.W = W; this.H = H; this.o = [];
  this.m = [1,0,0,1,0,0];
  this.st = [];
  this.fillStyle = "#000"; this.strokeStyle = "#000"; this.lineWidth = 1;
  this.globalAlpha = 1; this.lineCap = "butt"; this.lineJoin = "miter";
  this.dash = []; this.font = '10px serif';
  this.textAlign = "left"; this.textBaseline = "alphabetic";
  this.p = []; this.cur = null;
}
SvgRec.prototype._mul = function(A,B){
  return [A[0]*B[0]+A[2]*B[1], A[1]*B[0]+A[3]*B[1],
          A[0]*B[2]+A[2]*B[3], A[1]*B[2]+A[3]*B[3],
          A[0]*B[4]+A[2]*B[5]+A[4], A[1]*B[4]+A[3]*B[5]+A[5]];
};
SvgRec.prototype._pt = function(x,y){
  const m = this.m;
  return [m[0]*x + m[2]*y + m[4], m[1]*x + m[3]*y + m[5]];
};
SvgRec.prototype._k = function(){
  const m = this.m;
  return Math.sqrt(Math.abs(m[0]*m[3] - m[1]*m[2])) || 1;
};
SvgRec.prototype.save = function(){
  this.st.push({m:this.m.slice(), f:this.fillStyle, s:this.strokeStyle,
                w:this.lineWidth, a:this.globalAlpha, lc:this.lineCap,
                lj:this.lineJoin, d:this.dash.slice(), fn:this.font,
                ta:this.textAlign, tb:this.textBaseline});
};
SvgRec.prototype.restore = function(){
  const s = this.st.pop(); if(!s) return;
  this.m = s.m; this.fillStyle = s.f; this.strokeStyle = s.s;
  this.lineWidth = s.w; this.globalAlpha = s.a; this.lineCap = s.lc;
  this.lineJoin = s.lj; this.dash = s.d; this.font = s.fn;
  this.textAlign = s.ta; this.textBaseline = s.tb;
};
SvgRec.prototype.setTransform = function(a,b,c,d,e,f){ this.m = [a,b,c,d,e,f]; };
SvgRec.prototype.translate = function(x,y){ this.m = this._mul(this.m,[1,0,0,1,x,y]); };
SvgRec.prototype.scale = function(x,y){ this.m = this._mul(this.m,[x,0,0,y,0,0]); };
SvgRec.prototype.rotate = function(t){
  const c = Math.cos(t), s = Math.sin(t);
  this.m = this._mul(this.m,[c,s,-s,c,0,0]);
};
SvgRec.prototype.beginPath = function(){ this.p = []; this.cur = null; };
SvgRec.prototype.moveTo = function(x,y){
  this.cur = {pts:[this._pt(x,y)], closed:false};
  this.p.push(this.cur);
};
SvgRec.prototype.lineTo = function(x,y){
  if(!this.cur) return this.moveTo(x,y);
  this.cur.pts.push(this._pt(x,y));
};
SvgRec.prototype.closePath = function(){ if(this.cur) this.cur.closed = true; };
SvgRec.prototype.arc = function(cx,cy,r,a0,a1){
  const span = Math.abs(a1-a0), n = Math.max(16, Math.ceil(span*r*this._k()/2.2));
  for(let i=0;i<=n;i++){
    const a = a0 + (a1-a0)*i/n, x = cx + r*Math.cos(a), y = cy + r*Math.sin(a);
    const q = this._pt(x,y);
    if(!this.cur) { this.cur = {pts:[q], closed:false}; this.p.push(this.cur); }
    else this.cur.pts.push(q);
  }
};
SvgRec.prototype.setLineDash = function(d){ this.dash = (d || []).slice(); };
SvgRec.prototype._d = function(){
  const out = [];
  for(const sp of this.p){
    if(!sp.pts.length || sp.pts.length < 2) continue;
    out.push("M" + sp.pts.map(q => q[0].toFixed(2) + " " + q[1].toFixed(2)).join("L") +
             (sp.closed ? "Z" : ""));
  }
  return out.join(" ");
};
SvgRec.prototype.fill = function(){
  const d = this._d(); if(!d) return;
  this.o.push('<path d="' + d + '" fill="' + this.fillStyle + '"' +
    (this.globalAlpha < 1 ? ' fill-opacity="' + this.globalAlpha.toFixed(3) + '"' : '') + '/>');
};
SvgRec.prototype.stroke = function(){
  const d = this._d(); if(!d) return;
  const k = this._k();
  let a = '<path d="' + d + '" fill="none" stroke="' + this.strokeStyle +
          '" stroke-width="' + (this.lineWidth*k).toFixed(3) + '"';
  if(this.lineCap !== "butt") a += ' stroke-linecap="' + this.lineCap + '"';
  if(this.lineJoin !== "miter") a += ' stroke-linejoin="' + this.lineJoin + '"';
  if(this.dash.length) a += ' stroke-dasharray="' +
      this.dash.map(v => (v*k).toFixed(2)).join(" ") + '"';
  if(this.globalAlpha < 1) a += ' stroke-opacity="' + this.globalAlpha.toFixed(3) + '"';
  this.o.push(a + '/>');
};
SvgRec.prototype.fillRect = function(x,y,w,h){
  const p = [this._pt(x,y), this._pt(x+w,y), this._pt(x+w,y+h), this._pt(x,y+h)];
  const d = "M" + p.map(q => q[0].toFixed(2) + " " + q[1].toFixed(2)).join("L") + "Z";
  this.o.push('<path d="' + d + '" fill="' + this.fillStyle + '"/>');
};
SvgRec.prototype._text = function(txt,x,y,mode){
  const p = this._pt(x,y), k = this._k();
  const m = /^([\d.]+)px\s+(.+)$/.exec(this.font) || [0,"10","serif"];
  const fs = parseFloat(m[1] || 10) * k;
  const fam = String(m[2] || "serif").replace(/"/g, "'");
  const anch = this.textAlign === "right" ? "end"
             : (this.textAlign === "center" ? "middle" : "start");
  const bse = this.textBaseline === "middle" ? "central"
            : (this.textBaseline === "bottom" ? "auto"
            : (this.textBaseline === "top" ? "hanging" : "auto"));
  let a = '<text x="' + p[0].toFixed(2) + '" y="' + p[1].toFixed(2) + '"';
  if(mode === "stroke"){
    a += ' fill="none" stroke="' + this.strokeStyle +
         '" stroke-width="' + (this.lineWidth*k).toFixed(2) + '" stroke-linejoin="round"';
  } else {
    a += ' fill="' + this.fillStyle + '"';
  }
  a += ' font-family="' + fam + '" font-size="' + fs.toFixed(2) + '"';
  if(anch !== "start") a += ' text-anchor="' + anch + '"';
  if(bse !== "auto") a += ' dominant-baseline="' + bse + '"';
  if(this.globalAlpha < 1) a += (mode === "stroke" ? ' stroke-opacity="' : ' fill-opacity="') +
                                this.globalAlpha.toFixed(3) + '"';
  this.o.push(a + '>' + svgEsc(txt) + '</text>');
};
SvgRec.prototype.fillText = function(t,x,y){ this._text(t,x,y,"fill"); };
SvgRec.prototype.strokeText = function(t,x,y){ this._text(t,x,y,"stroke"); };
const _MCTX = document.createElement("canvas").getContext("2d");
SvgRec.prototype.measureText = function(t){ _MCTX.font = this.font; return _MCTX.measureText(t); };
SvgRec.prototype.serialize = function(){
  return '<?xml version="1.0" encoding="UTF-8"?>\n' +
    '<svg xmlns="http://www.w3.org/2000/svg" width="' + this.W + '" height="' + this.H +
    '" viewBox="0 0 ' + this.W + ' ' + this.H + '">\n' + this.o.join("\n") + '\n</svg>\n';
};

function scaleBar(){
  const H = cv.clientHeight;
  const target = cv.clientWidth*0.16/view.scale;
  const NICE = [1,2,5,10,20,50,100,200,250,500,1000,2000,2500,5000,10000];
  let L = NICE[0], bd = Infinity;
  for(const v of NICE){ const d=Math.abs(Math.log(v/target)); if(d<bd){bd=d;L=v;} }
  const x0 = 26, y0 = H - 30, px = L*view.scale;
  ctx.strokeStyle = "#4A4A4A"; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x0, y0); ctx.lineTo(x0+px, y0);
  ctx.moveTo(x0, y0-4); ctx.lineTo(x0, y0+4);
  ctx.moveTo(x0+px, y0-4); ctx.lineTo(x0+px, y0+4);
  ctx.stroke();
  ctx.fillStyle = "#4A4A4A";
  ctx.font = '10px "Times New Roman","SimSun",serif';
  ctx.textAlign = "left"; ctx.textBaseline = "bottom";
  ctx.fillText(L + " m", x0, y0 - 6);
}

function draw(){
  syncRot();
  const g = id => document.getElementById(id).checked;

  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.fillStyle="#FFFFFF";
  ctx.fillRect(0,0,cv.clientWidth,cv.clientHeight);
  if(!curApt) return;

  const applyWorld = () => {
    ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.translate(ctr.x, ctr.y);
    ctx.rotate(view.rot);
    ctx.scale(view.scale, view.scale);
    ctx.translate(-view.ox, -view.oz);
  };

  if(g("ly-bg") && curApt.bg.length){
    ctx.save(); applyWorld();
    for(const q of curApt.bg){
      const c = BGC[q.k] || BGC.building;
      if(!pathPoly(q.p)) continue;
      ctx.fillStyle = c[0];
      ctx.globalAlpha = (q.k === "apron") ? 0.9 : 0.95;
      ctx.fill();
      ctx.globalAlpha = 1;
      if(c[1] && q.k !== "terminal"){
        ctx.strokeStyle = c[1];
        ctx.lineWidth = 0.7/view.scale;
        ctx.stroke();
      }
    }
    ctx.restore();
    ctx.globalAlpha = 1;
  }

  if(g("ly-net")){
    ctx.save(); applyWorld();
    ctx.strokeStyle="#79706E";
    ctx.lineWidth = 1.6/view.scale;
    ctx.lineCap="round"; ctx.lineJoin="round";
    pathLines(curApt.net.lines); ctx.stroke();
    ctx.restore();
  }

  if(g("ly-rwy")){
    ctx.save(); applyWorld();
    ctx.strokeStyle="#4D4A49";
    ctx.lineWidth = 5.5/view.scale;
    ctx.lineCap="butt"; ctx.lineJoin="miter";
    pathLines(curApt.net.rwy || []); ctx.stroke();
    ctx.restore();
  }

  if(g("ly-node") && view.scale > 0.45){
    ctx.fillStyle="#FFFFFF"; ctx.strokeStyle="#111111"; ctx.lineWidth=1.1;
    const lines = curApt.net.lines;
    for(let i=0;i<lines.length;i++){
      const l = lines[i];
      const ends=[[l[0],l[1]],[l[l.length-2],l[l.length-1]]];
      for(const e of ends){
        const q=w2s(e[0], e[1]);
        if(q[0]<-10||q[1]<-10||q[0]>cv.clientWidth+10||q[1]>cv.clientHeight+10) continue;
        ctx.beginPath(); ctx.arc(q[0],q[1],2.2,0,Math.PI*2); ctx.fill(); ctx.stroke();
      }
    }
  }

  if(g("ly-r")){
    ctx.save(); applyWorld();
    ctx.strokeStyle="#1F77B4"; ctx.globalAlpha = 0.75;
    ctx.lineWidth = 2.2/view.scale;
    ctx.lineCap="round"; ctx.lineJoin="round";
    pathLines(curApt.net.lines, curApt.net.kinds, true); ctx.stroke();
    ctx.restore();
    ctx.globalAlpha = 1;
  }

  // 潜在 R：算法阶段 1 输出（不是最终 best_path 里选定的 R）
  if(g("ly-potr") && current && current.potr && current.potr.length){
    ctx.save(); applyWorld();
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    for(const q of current.potr){
      const P = q[2];
      if(P.length < 2) continue;
      ctx.beginPath();
      ctx.moveTo(P[0][0], P[0][1]);
      for(let i=1;i<P.length;i++) ctx.lineTo(P[i][0], P[i][1]);
      const wpx = 3.2 + 1.1*Math.min(3, Math.log(1 + q[1])/Math.LN2);
      ctx.strokeStyle = "rgba(255,255,255,.85)";     // 白色描边：压在跑道/机坪上也看得清
      ctx.globalAlpha = 0.6;
      ctx.lineWidth = (wpx + 2.2) / view.scale;
      ctx.stroke();
      ctx.strokeStyle = "#6A1B9A";
      ctx.globalAlpha = 0.45 + 0.5 * Math.min(1, q[1] / 60);
      ctx.lineWidth = wpx / view.scale;
      ctx.stroke();
    }
    ctx.restore();
    ctx.globalAlpha = 1;
    if(view.scale > 0.35){
      ctx.font = STY.potr + 'px "Times New Roman","SimSun",serif';
      ctx.textAlign = "left"; ctx.textBaseline = "middle";
      const placed = [];
      const hh = STY.potr*0.68;
      // 屏幕坐标预取：用于给候选位置打分，避免标签压在轨迹点或橙色路线上
      const CS = 32;                            // 打分用网格，避免每个候选都遍历全部点
      const mkGrid = pts => {
        const m = new Map();
        for(const p of pts){
          const k = Math.floor(p[0]/CS) + "," + Math.floor(p[1]/CS);
          m.set(k, (m.get(k)||0) + 1);
        }
        return m;
      };
      const TKG = mkGrid(current.trk.map(p => w2s(p[0], p[1])));
      const RTG = mkGrid((current.route || []).map(p => w2s(p[0], p[1])));
      const gHits = (G, b) => {
        let n = 0;
        const x0 = Math.floor(b[0]/CS), x1 = Math.floor(b[2]/CS);
        const y0 = Math.floor(b[1]/CS), y1 = Math.floor(b[3]/CS);
        for(let i=x0;i<=x1;i++) for(let j=y0;j<=y1;j++) n += G.get(i+","+j) || 0;
        return n;
      };
      for(const q of current.potr){
        const P = q[2], k0 = Math.floor(P.length/2);
        const m0 = w2s(P[k0][0], P[k0][1]);
        if(m0[0] < 0 || m0[1] < 0 || m0[0] > cv.clientWidth || m0[1] > cv.clientHeight) continue;
        // 沿路径取切线 -> 法向；标签沿法向甩到旁边，并用引线回指该 R
        const pa = w2s(P[Math.max(0, k0-1)][0], P[Math.max(0, k0-1)][1]);
        const pb = w2s(P[Math.min(P.length-1, k0+1)][0], P[Math.min(P.length-1, k0+1)][1]);
        let nx = -(pb[1]-pa[1]), ny = (pb[0]-pa[0]);
        const nn = Math.hypot(nx, ny) || 1; nx /= nn; ny /= nn;
        if(!isFinite(nx) || !isFinite(ny)){ nx = 0; ny = -1; }
        const t = q[0] + " (" + q[1] + ")";
        const w = ctx.measureText(t).width;
        const ux = (pb[0]-pa[0])/nn, uy = (pb[1]-pa[1])/nn;   // 切向
        let best = null;
        for(const side of [1, -1]){
          for(const d0 of [9, 15, 23, 33, 45, 60, 80, 105]){
            for(const tx of [-30, -15, 0, 15, 30]){            // 切向错开，避免同一走廊里的标签挤在一起
              const d = d0*STY.potr/19;
              const px = m0[0] + nx*d*side + ux*tx;
              const py = m0[1] + ny*d*side + uy*tx;
              const lx = Math.max(6, Math.min(px + (side>0 ? 5 : -5), cv.clientWidth - w - 6));
              const box = [lx - 2, py - hh, lx + w + 2, py + hh];
              if(box[0] < 2 || box[3] > cv.clientHeight - 2 || box[1] < 2 || box[2] > cv.clientWidth - 2) continue;
              let ov = false;                                  // 硬约束：不与已放置标签相交
              for(const b of placed){
                if(box[0] < b[2] && box[2] > b[0] && box[1] < b[3] && box[3] > b[1]){ ov = true; break; }
              }
              if(ov) continue;
              const pen = d0*0.05 + gHits(TKG, box)*6 + gHits(RTG, box)*20;
              if(!best || pen < best.pen) best = {pen:pen, lx:lx, py:py, box:box, side:side};
            }
          }
        }
        if(!best) continue;                                    // 实在放不下就不标，宁缺不叠
        // 引线：从 R 的中点拉到标签侧
        const ax = best.side > 0 ? best.box[0] : best.box[2];
        ctx.strokeStyle = "rgba(106,27,154,.55)";
        ctx.lineWidth = STY.potr*0.079;
        ctx.beginPath(); ctx.moveTo(m0[0], m0[1]); ctx.lineTo(ax, best.py); ctx.stroke();
        ctx.beginPath(); ctx.arc(m0[0], m0[1], STY.potr*0.158, 0, Math.PI*2);
        ctx.fillStyle = "#6A1B9A"; ctx.fill();
        placed.push(best.box);
        ctx.lineWidth = STY.potr*0.247; ctx.strokeStyle = "rgba(255,255,255,.9)";
        ctx.strokeText(t, best.lx, best.py);
        ctx.fillStyle = "#4A1364"; ctx.fillText(t, best.lx, best.py);
      }
    }
  }

  if(g("ly-line") && current){
    const flat=[]; current.trk.forEach(p=>flat.push(p[0],p[1]));
    polyline(flat,"#2CA02C",1.4,0.45);
  }

  if(g("ly-proj") && current && current.proj){
    const flat=[]; current.proj.forEach(p=>flat.push(p[0],p[1]));
    ctx.save(); ctx.setLineDash([5,4]);
    polyline(flat,"#B08A78",1.4,0.9);
    ctx.restore();
  }

  if(g("ly-link") && current && current.proj){
    ctx.save();
    ctx.setLineDash([2,3]);
    ctx.strokeStyle="rgba(214,39,40,.5)";
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    const T = current.trk, E = current.err, Q = current.proj;
    for(let i=0;i<T.length;i+=4){
      if((E[i]||0) <= 150) continue;
      const a = w2s(T[i][0], T[i][1]);
      const b = w2s(Q[i][0], Q[i][1]);
      ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]);
    }
    ctx.stroke(); ctx.restore();
  }

  if(g("ly-track") && current){
    ctx.fillStyle="#2CA02C";
    const T=current.trk;
    const st = Math.max(1, Math.round(STY.step));   // 取样间隔：1=全部点
    for(let i=0;i<T.length;i+=st){
      const q=w2s(T[i][0], T[i][1]);
      if(q[0]<-6||q[1]<-6||q[0]>cv.clientWidth+6||q[1]>cv.clientHeight+6) continue;
      ctx.beginPath(); ctx.arc(q[0],q[1],STY.pt,0,Math.PI*2); ctx.fill();
    }
  }

  if(g("ly-best") && current && current.route){
    const flat=[]; current.route.forEach(p=>flat.push(p[0],p[1]));
    polyline(flat,"#FF7F0E",STY.best,0.95);
  }

  if(g("ly-over") && current){
    const T = current.trk, E = current.err;
    ctx.strokeStyle = "#D62728";
    ctx.lineWidth = 1.2;
    let lx = null, ly = null;
    for(let i=0;i<T.length;i++){
      if((E[i]||0) <= 150) continue;
      const q = w2s(T[i][0], T[i][1]);
      if(q[0]<-8||q[1]<-8||q[0]>cv.clientWidth+8||q[1]>cv.clientHeight+8) continue;
      if(lx !== null && Math.hypot(q[0]-lx, q[1]-ly) < 11) continue;
      lx = q[0]; ly = q[1];
      ctx.beginPath(); ctx.arc(q[0],q[1],4.8,0,Math.PI*2); ctx.stroke();
    }
  }

  scaleBar();
  drawLegend();
}

let drag=null;
cv.addEventListener("mousedown",e=>{drag={x:e.clientX,y:e.clientY,ox:view.ox,oz:view.oz};cv.classList.add("drag");});
window.addEventListener("mouseup",()=>{drag=null;cv.classList.remove("drag");});
window.addEventListener("mousemove",e=>{
  if(!drag) return;
  syncRot();
  const du=(e.clientX-drag.x)/view.scale, dv=(e.clientY-drag.y)/view.scale;
  view.ox = drag.ox - (du*view.c + dv*view.s);
  view.oz = drag.oz - (-du*view.s + dv*view.c);
  draw();
});
cv.addEventListener("wheel",e=>{
  e.preventDefault();
  const r=cv.getBoundingClientRect();
  const mx=e.clientX-r.left, my=e.clientY-r.top;
  const w=s2w(mx,my);
  const k=Math.pow(1.0015, -e.deltaY);
  view.scale=Math.max(0.02, Math.min(400, view.scale*k));
  syncRot();
  const u=(mx-ctr.x)/view.scale, v=(my-ctr.y)/view.scale;
  view.ox = w[0] - u*view.c - v*view.s;
  view.oz = w[1] + u*view.s - v*view.c;
  draw();
},{passive:false});
cv.addEventListener("dblclick",()=>resetView());

let PSET = new Set();
cv.addEventListener("mousemove",e=>{
  if(drag || !curApt){ tip.style.display="none"; return; }
  const r=cv.getBoundingClientRect();
  const mx=e.clientX-r.left, my=e.clientY-r.top;
  let best=null, bestD=5;
  const lines = curApt.net.lines;
  for(let i=0;i<lines.length;i++){
    const l=lines[i];
    for(let j=0;j<l.length-2;j+=2){
      const a=w2s(l[j],l[j+1]), b=w2s(l[j+2],l[j+3]);
      const dx=b[0]-a[0], dy=b[1]-a[1];
      const L2=dx*dx+dy*dy||1e-9;
      let t=((mx-a[0])*dx+(my-a[1])*dy)/L2;
      t=Math.max(0,Math.min(1,t));
      const d=Math.hypot(mx-(a[0]+t*dx), my-(a[1]+t*dy));
      if(d<bestD){ bestD=d; best=i; }
    }
  }
  if(best!==null){
    const nm = curApt.net.names[best];
    const ln = curApt.net.lens[best];
    let txt = nm + (ln ? " &middot; "+ln+" m" : "") + (curApt.net.kinds[best]==="R" ? " &middot; R" : "");
    if(PSET.has(nm)){
      const k = current.segs.findIndex(x=>x[0]===nm) + 1;
      txt += " &middot; path #" + k;
    }
    const pq = current && (current.potr || []).find(x => x[0] === nm);
    if(pq) txt += " &middot; potential R (" + pq[1] + " pts)";
    tip.innerHTML = txt;
    tip.style.left=mx+"px"; tip.style.top=my+"px"; tip.style.display="block";
  }else{
    tip.style.display="none";
  }
});
cv.addEventListener("mouseleave",()=>{tip.style.display="none";});

// ---------- 数据 ----------
const META = {};
DATA.meta.forEach(m=>{ META[m.code]=m; });

const aptSel = document.getElementById("apt");
DATA.meta.forEach(m=>{
  const o=document.createElement("option");
  o.value=m.code;
  o.textContent = m.code;
  aptSel.appendChild(o);
});

const sel = document.getElementById("flight");
let FL = [];

function fmt(v,d){ return (v===null||v===undefined||isNaN(v)) ? "&mdash;" : Number(v).toFixed(d===undefined?1:d); }

function computeBBox(){
  let minX=Infinity,maxX=-Infinity,minZ=Infinity,maxZ=-Infinity;
  const acc=(x,z)=>{ if(x<minX)minX=x; if(x>maxX)maxX=x; if(z<minZ)minZ=z; if(z>maxZ)maxZ=z; };
  let used=false;
  if(current){
    current.trk.forEach(p=>{acc(p[0],p[1]); used=true;});
    (current.route||[]).forEach(p=>acc(p[0],p[1]));
  }
  if(!used && curApt) curApt.net.lines.forEach(l=>{for(let k=0;k<l.length;k+=2) acc(l[k],l[k+1]);});
  return {minX,maxX,minZ,maxZ};
}

function applyFit(){
  syncRot();
  let mnX=Infinity, mxX=-Infinity, mnY=Infinity, mxY=-Infinity;
  const corners=[[bbox.minX,bbox.minZ],[bbox.minX,bbox.maxZ],[bbox.maxX,bbox.minZ],[bbox.maxX,bbox.maxZ]];
  for(const c0 of corners){
    const rx=c0[0]*view.c - c0[1]*view.s;
    const ry=c0[0]*view.s + c0[1]*view.c;
    if(rx<mnX)mnX=rx; if(rx>mxX)mxX=rx;
    if(ry<mnY)mnY=ry; if(ry>mxY)mxY=ry;
  }
  view.ox = (bbox.minX + bbox.maxX)/2;
  view.oz = (bbox.minZ + bbox.maxZ)/2;
  const w = Math.max(cv.clientWidth - 70, 40);
  const h = Math.max(cv.clientHeight - 70, 40);
  view.scale = Math.min(w/Math.max(mxX-mnX,1), h/Math.max(mxY-mnY,1)) * ZOOM;
  draw();
}

function resetView(){ bbox = computeBBox(); applyFit(); }

function fitWholeAirport(){
  if(!curApt){ resetView(); return; }
  const N = curApt.net;
  const b = {minX:Infinity,maxX:-Infinity,minZ:Infinity,maxZ:-Infinity};
  const acc = (l)=>{ for(let k=0;k<l.length;k+=2){
      const x=l[k], z=l[k+1];
      if(x<b.minX)b.minX=x; if(x>b.maxX)b.maxX=x;
      if(z<b.minZ)b.minZ=z; if(z>b.maxZ)b.maxZ=z; } };
  N.lines.forEach((l,i)=>{ if(!N.isrwy || !N.isrwy[i]) acc(l); });
  if(!isFinite(b.minX)) N.lines.forEach(acc);
  if(!isFinite(b.minX)){ resetView(); return; }
  const bw = b.maxX - b.minX, bh = b.maxZ - b.minZ;
  if(bh > bw*1.15 && cv.clientWidth > cv.clientHeight){
    document.getElementById("rot").value = 90;
    document.getElementById("rotVal").textContent = "90\u00B0";
    view.rot = Math.PI/2; syncRot();
  }
  bbox = b; applyFit();
}

function setFlight(i){
  current = FL[i];
  curApt = aptData(aptSel.value);
  PSET = new Set();
  (current.segs||[]).forEach(x=>PSET.add(x[0]));
  const f = current, m = f.m;
  let h = "";
  h += "<div>Track points: <b>"+m.npts+"</b></div>";
  h += "<div>Projected segments: <b>"+m.nseg+"</b> <span style='font-size:11px'>(N "+m.nN+" / R "+m.nR+")</span></div>";
  if(m.npotr) h += "<div>Potential R: <b>"+m.npotr+"</b> paths &middot; <b>"+m.npotrpts+"</b> matched pts</div>";
  const RK = m.rkind==="graph" ? "graph path" : (m.rkind==="none" ? "not identified" : "projection");
  h += "<div>Route: <b>"+RK+"</b> &middot; <b>"+fmt(m.npaths,0)+"</b> paths</div>";
  h += "<div>Track &rarr; route: <b>"+fmt(m.medr,1)+"</b> med &middot; <b>"+fmt(m.p95r,1)+"</b> P95 m</div>";
  h += "<div>Track &rarr; graph: <b>"+fmt(m.med,1)+"</b> med &middot; <b>"+fmt(m.p95,1)+"</b> P95 m</div>";
  h += "<div>Cover &le;15 m: <b>"+fmt(m.cov15r,1)+"</b> %</div>";
  h += "<div>Points &gt;15 m off: <b"+(m.noverr?" class='warn'":"")+">"+m.noverr+"</b> pts</div>";
  h += "<div>Route / Track: <b>"+fmt(m.rlen,0)+"</b> / <b>"+fmt(m.tlen,0)+"</b> m</div>";
  h += "<div>Length ratio: <b>"+fmt(m.ratio,2)+"</b></div>";
  const M = META[aptSel.value];
  const X = f.exp || {};
  const typeTxt = (X.flio==="A" ? "Arrival" : X.flio==="D" ? "Departure" : "");
  let ref = typeTxt;
  if(X.ok===false) ref += (ref?" &middot; ":"") + "<b class='warn'>not identified</b>";
  else if(X.ncand!=null) ref += (ref?" &middot; ":"") + X.ncand + " cand.";
  if(ref) h += "<div class='ref'>"+ref+"</div>";
  if(M.n_ok!=null) h += "<div>"+M.code+" &middot; <b>"+M.n_ok+"/"+M.n_all+"</b> identified</div>";
  h += "<div>Graph: <b>"+M.nnet+"</b> paths &middot; <b>"+M.nR+"</b> R</div>";
  document.getElementById("stats").innerHTML = h;
  resetView();
}

function fillFlights(){
  FL = DATA.flights[aptSel.value];
  sel.innerHTML = "";
  FL.forEach((f,i)=>{
    const o=document.createElement("option");
    o.value=i;
    o.textContent = f.id;
    sel.appendChild(o);
  });
  if(FL.length){ sel.value=0; setFlight(0); } else { current=null; draw(); }
}

function resize(){
  dpr = window.devicePixelRatio || 1;
  const W = wrap.clientWidth - 28, H = wrap.clientHeight - 28;
  cv.width = Math.max(1, Math.floor(W*dpr));
  cv.height= Math.max(1, Math.floor(H*dpr));
  cv.style.width = W+"px"; cv.style.height = H+"px";
  ctr.x = cv.clientWidth/2;
  ctr.y = cv.clientHeight/2;
  ctx.setTransform(dpr,0,0,dpr,0,0);
  draw();
}

aptSel.addEventListener("change",()=>{ curApt = aptData(aptSel.value); fillFlights(); });
sel.addEventListener("change",()=>setFlight(+sel.value));
document.querySelectorAll(".layer input[type=checkbox]").forEach(cb=>cb.addEventListener("change",draw));

const rotEl=document.getElementById("rot"), rotVal=document.getElementById("rotVal");
rotEl.addEventListener("input",()=>{
  view.rot = (+rotEl.value) * Math.PI/180;
  rotVal.textContent = rotEl.value + "\u00B0";
  syncRot(); draw();
});

document.getElementById("reset").addEventListener("click",()=>resetView());
document.getElementById("fitall").addEventListener("click",fitWholeAirport);
function exportName(){
  const ap = aptSel.value, id = current ? current.id : "viewer";
  return (id.indexOf(ap + "_") === 0 ? id : ap + "_" + id);
}

// 以临时放大 dpr 重绘一帧再取图：图例在画布内，所以导出稿一定带图例
function exportPNG(scale){
  const od = dpr, ow = cv.width, oh = cv.height;
  dpr = scale;
  cv.width  = Math.round(cv.clientWidth  * scale);
  cv.height = Math.round(cv.clientHeight * scale);
  let url;
  try { draw(); url = cv.toDataURL("image/png"); }
  finally {
    cv.width = ow; cv.height = oh; dpr = od;
    draw();
  }
  const a = document.createElement("a");
  a.download = exportName() + (scale > 1 ? "@" + scale + "x" : "") + ".png";
  a.href = url;
  a.click();
}

document.getElementById("png").addEventListener("click",()=>exportPNG(1));
document.getElementById("png2").addEventListener("click",()=>exportPNG(2));

// SVG 导出：把 canvas 换成录制上下文重跑一遍 draw()，因此图例、比例尺、标注全部保留且为矢量
document.getElementById("svg").addEventListener("click",()=>{
  const R = new SvgRec(cv.width, cv.height), old = ctx;
  ctx = R;
  try { draw(); } finally { ctx = old; }
  const url = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(R.serialize());
  const a = document.createElement("a");
  a.download = exportName() + ".svg";
  a.href = url;
  a.click();
});
// ---- 外观滑杆：改 STY 后直接重绘；导出走同一套 draw()，所以 PNG / SVG 同步 ----
const STY_CTL = [["sty-pt", "pt", 1], ["sty-stp", "step", 0], ["sty-bst", "best", 1],
                 ["sty-pot", "potr", 0], ["sty-lg", "lg", 2]];
for(const c of STY_CTL){
  const el = document.getElementById(c[0]), lab = document.getElementById(c[0] + "-v");
  const sync = () => {
    const v = +el.value;
    STY[c[1]] = v;
    if(c[1] === "lg") LGS = v;
    lab.textContent = v.toFixed(c[2]);
    draw();
  };
  el.addEventListener("input", sync);
  sync();                       // 按 HTML 里的 value 初始化 STY（否则改初值不生效）
}

window.addEventListener("resize",resize);

window.addEventListener("keydown",e=>{
  if(e.key!=="ArrowLeft" && e.key!=="ArrowRight") return;
  const n = FL.length; if(!n) return;
  let i = (+sel.value + (e.key==="ArrowRight"?1:-1) + n) % n;
  sel.value = i; setFlight(i);
});

(function(){
  const q = new URLSearchParams(location.search);
  const a = q.get("apt");
  if(a && META[a]) aptSel.value = a;
  syncRot();
  resize();
  fillFlights();
  const tid = q.get("tid");
  if(tid){
    const i = FL.findIndex(f=>f.id===tid);
    if(i>=0){ sel.value=i; setFlight(i); }
  }
  if(q.get("view") === "all") fitWholeAirport();
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
