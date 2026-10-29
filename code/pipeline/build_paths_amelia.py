# -*- coding: utf-8 -*-
"""
Amelia semantic_graph -> 西安/重庆同 schema 的 N/R 路径集
============================================================
口径完全沿用 E:/科研2026/轨迹修正/tools/build_paths_json.py：
  1) junction 处拆分 -> 极大节点序列（一条 path）
  2) N/R 判定：path 内部最大三点转向角 < TURN_THR(5°) -> N，否则 R
  3) 端点命名 "Point (k)"（纬度降序、经度升序）；内部点全局连续编号
  4) 坐标 y=0、roa=0、float32 精度

Amelia 特有的两处适配（原脚本不需要、这里必须补）：
  A. way 边界已丢失 -> 用「度 != 2」的节点作为 junction（等价于"被>=2 way 引用 ∪ way 首尾"）
  B. 仅有 junction 拆分时，一条含弯的 degree-2 长链会整条被判 R（KDCA 有 3946m 的链），
     粒度粗到无法约束路线。因此额外在**直/弯过渡处**补切（曲率速率判据，尺度无关），
     使 N 始终是直段、R 始终是弯段，与原脚本"直线 N / 弯曲 R"的含义一致。

输出（out_dir/）：
  InitMatrixPath.json      路径图（pathLength/points 用于建图与打分）
  InitMatrixPath-1m.json   1m 重采样（潜在R网格匹配用）
  InitPath.json            副本（原算法仅在夹角计算用，schema 相同）
  connection_matrix.csv    邻接矩阵（对称 0/1，A_ij=1 <=> 两 path 共享端点）
  path_meta.json           pathname -> {kind, cls, u, v, length, is_runway}
  build_report.json        构建统计
"""
import os

_REPO = os.environ.get("ROUTE_ID_ROOT") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import sys
import json
import math
import pickle
import argparse
import statistics
from collections import Counter

import numpy as np
import networkx as nx

TURN_THR = 5.0        # N/R 判定阈值（度）—— 与 build_paths_json.py 一致
R_MIN = 0.45          # 曲率速率阈值（度/米）：0.45 deg/m <=> 半径 <= 127 m 视为弯
CURV_WIN = 40.0       # 曲率估计窗口（米）：40m 以抑制密集采样的航向抖动
BRIDGE_GAP = 3.0      # 弯区内的短直空洞桥接（米）
MIN_R_LEN = 8.0       # 转角最小弧长（米）：直角是单顶点急转，需撑开到西安 R 最短 7.96m 量级
MIN_SPAN = 2.0        # 最短 N 段（米），更短的并入邻段
STEP_M = 1.0          # 1m 重采样步长（米）
FILLET_MIN_TURN = 30.0  # 转角 >= 该值才插圆角弧（度）
FILLET_D = 8.0          # 切口点距转角点的距离（米）
FILLET_LEN = 40.0       # 圆角弧最大弧长（米）
CLS_NAME = {1: "hold_line", 3: "runway", 4: "taxiway"}


def f32(v):
    import struct
    return struct.unpack("<f", struct.pack("<f", v))[0]


# ============================================================
# 1) 读取
# ============================================================
def load_graph(pkl_path):
    g = pickle.load(open(pkl_path, "rb"))
    G = g["graph_networkx"]
    P = g["map_infos"]["all_polylines"]

    # 节点经纬度（graph 自带，x=lon, y=lat）
    ll = {n: (float(d["y"]), float(d["x"])) for n, d in G.nodes(data=True)}

    U = nx.Graph()
    xy = {}
    for u, v, k, d in G.edges(keys=True, data=True):
        row = P[int(d["osmid"])]
        x1, y1 = float(row[2]) * 1000.0, float(row[3]) * 1000.0
        x2, y2 = float(row[6]) * 1000.0, float(row[7]) * 1000.0
        xy[u] = (x1, y1)
        xy[v] = (x2, y2)
        if u == v:
            continue
        L = float(d.get("length", 0.0))
        c = int(row[8])
        if U.has_edge(u, v):
            if L > U[u][v]["length"]:
                U[u][v].update(length=L, cls=c)
        else:
            U.add_edge(u, v, length=L, cls=c)

    # 孤立节点（无边）不参与
    for n in list(U.nodes()):
        if n not in xy:
            U.remove_node(n)
    return U, xy, ll


# ============================================================
# 2) junction + 链
# ============================================================
def build_chains(U):
    junction = {n for n, d in U.degree() if d != 2}
    chains, visited = [], set()

    def walk(start, first):
        path = [start, first]
        visited.add((min(start, first), max(start, first)))
        prev, cur = start, first
        while cur not in junction:
            nxts = [w for w in U.neighbors(cur) if w != prev]
            if not nxts:
                break
            nxt = nxts[0]
            kk = (min(cur, nxt), max(cur, nxt))
            if kk in visited:
                break
            visited.add(kk)
            path.append(nxt)
            prev, cur = cur, nxt
        return path

    for n in sorted(junction):
        for nb in sorted(U.neighbors(n)):
            if (min(n, nb), max(n, nb)) in visited:
                continue
            chains.append(walk(n, nb))
    # 纯环兜底
    for u, v in list(U.edges()):
        if (min(u, v), max(u, v)) not in visited:
            chains.append(walk(u, v))
    return chains, junction


def chain_pts(U, xy, chain):
    pts, s = [], 0.0
    for i, n in enumerate(chain):
        if i > 0:
            s += U[chain[i - 1]][n]["length"]
        x, y = xy[n]
        pts.append((x, y, s))
    return pts


# ============================================================
# 3) 曲率速率 + 直/弯过渡切分
# ============================================================
def _interp(pts, s):
    if s <= pts[0][2]:
        return pts[0][0], pts[0][1]
    if s >= pts[-1][2]:
        return pts[-1][0], pts[-1][1]
    lo, hi = 0, len(pts) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if pts[mid][2] <= s:
            lo = mid
        else:
            hi = mid
    t = (s - pts[lo][2]) / max(1e-9, pts[lo + 1][2] - pts[lo][2])
    return (pts[lo][0] + t * (pts[lo + 1][0] - pts[lo][0]),
            pts[lo][1] + t * (pts[lo + 1][1] - pts[lo][1]))


def curvature_rates(pts, w=CURV_WIN):
    n = len(pts)
    r = [0.0] * n
    if n < 3:
        return r
    half = w / 2.0
    s0, sN = pts[0][2], pts[-1][2]
    for i in range(n):
        s = pts[i][2]
        sm, sp = max(s0, s - half), min(sN, s + half)
        if sp - sm < 1e-6:
            continue
        x0, y0 = _interp(pts, sm)
        xm, ym = _interp(pts, s)
        x1, y1 = _interp(pts, sp)
        h_in = math.atan2(ym - y0, xm - x0)
        h_out = math.atan2(y1 - ym, x1 - xm)
        d = math.atan2(math.sin(h_out - h_in), math.cos(h_out - h_in))
        r[i] = abs(math.degrees(d)) / (sp - sm)
    return r


def split_pieces(pts, r_min=None, win=None, min_r_len=None, min_span=None):
    """返回若干 (i0, i1) 顶点区间（闭区间），直段/弯段分开

    与原 build_paths_json.py 的差别仅在于「拆分位置」：原脚本有 OSM way 边界可用，
    Amelia 的图已丢失 way 边界，故改用曲率起止作为拆分点；N/R 的**分类判据完全相同**。
    """
    r_min = R_MIN if r_min is None else r_min
    win = CURV_WIN if win is None else win
    min_r_len = MIN_R_LEN if min_r_len is None else min_r_len
    min_span = MIN_SPAN if min_span is None else min_span

    n = len(pts)
    if n < 3:
        return [(0, n - 1)]
    rate = curvature_rates(pts, win)
    curved = [x >= r_min for x in rate]

    # ---- 1) 桥接短空洞 ----
    i = 0
    while i < n:
        if not curved[i]:
            j = i
            while j + 1 < n and not curved[j + 1]:
                j += 1
            if i > 0 and j < n - 1 and (pts[j][2] - pts[i][2]) <= BRIDGE_GAP:
                for t in range(i, j + 1):
                    curved[t] = True
            i = j + 1
        else:
            i += 1

    # ---- 2) 提取弯区 ----
    runs = []
    i = 0
    while i < n:
        if curved[i]:
            j = i
            while j + 1 < n and curved[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1

    # ---- 3) 弯区撑开到最小弧长（左右交替扩展，受链端点约束）----
    for r in runs:
        while pts[r[1]][2] - pts[r[0]][2] < min_r_len:
            can_l = r[0] > 0
            can_r = r[1] < n - 1
            if not can_l and not can_r:
                break
            # 交替：哪边先展，取决于当前哪侧还有余量（用长度平衡）
            if can_l and (not can_r or (r[0] <= n - 1 - r[1])):
                r[0] -= 1
            else:
                r[1] += 1

    # ---- 4) 合并重叠/相邻弯区 ----
    runs.sort()
    merged = []
    for r in runs:
        if merged and r[0] <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], r[1])
        else:
            merged.append(r)

    # ---- 5) 取补集得直段 ----
    segs = []
    cur = 0
    for a, b in merged:
        if a > cur:
            segs.append((cur, a))
        segs.append((a, b))
        cur = b
    if cur < n - 1:
        segs.append((cur, n - 1))
    if not segs:
        segs = [(0, n - 1)]

    # ---- 6) 过短段并入邻段（取较长邻段）----
    k = 0
    while k < len(segs) and len(segs) > 1:
        a, b = segs[k]
        if pts[b][2] - pts[a][2] < min_span:
            Lleft = (pts[segs[k - 1][1]][2] - pts[segs[k - 1][0]][2]) if k > 0 else -1
            Lright = (pts[segs[k + 1][1]][2] - pts[segs[k + 1][0]][2]) if k + 1 < len(segs) else -1
            if Lleft >= Lright and Lleft >= 0:
                segs[k - 1] = (segs[k - 1][0], b)
            elif Lright >= 0:
                segs[k + 1] = (a, segs[k + 1][1])
            else:
                break
            segs.pop(k)
            k = max(0, k - 1)
        else:
            k += 1
    return segs




# ============================================================
# 角点圆角化（Amelia 的 OSM 转角是无弧尖角，需补几何才能形成
# 与西安同量级的 R 锚点；物理上也更真实——航空器不可能在一点转向）
# ============================================================
def _leave_dir(pts, side, depth):
    """端点指向内部的单位方向与该方向有效深度"""
    if len(pts) < 2:
        return None, 0.0
    if side == 0:
        x0, y0 = pts[0][0], pts[0][1]
        s_t = min(pts[0][2] + depth, pts[-1][2])
        x1, y1 = _interp(pts, s_t)
    else:
        x0, y0 = pts[-1][0], pts[-1][1]
        s_t = max(pts[-1][2] - depth, pts[0][2])
        x1, y1 = _interp(pts, s_t)
    dx, dy = x1 - x0, y1 - y0
    m = math.hypot(dx, dy)
    if m < 1e-9:
        return None, 0.0
    return (dx / m, dy / m), m


def _point_at(pts, side, depth):
    if side == 0:
        s_t = min(pts[0][2] + depth, pts[-1][2])
    else:
        s_t = max(pts[-1][2] - depth, pts[0][2])
    return _interp(pts, s_t), s_t


def _norm(pts):
    """把弧长参数平移到从 0 开始"""
    if not pts:
        return pts
    s0 = pts[0][2]
    return [(x, y, s - s0) for (x, y, s) in pts]


def _cut(pts, side, s_cut):
    """在 s_cut 处切开：返回 (远离转角侧的保留段, 靠转角侧的被切段)，均从 A 侧归一"""
    cp = _interp(pts, s_cut)
    keep, rem = [], []
    for (x, y, s) in pts:
        if s <= s_cut:
            keep.append((x, y, s))
        if s >= s_cut:
            rem.append((x, y, s))
    if not keep or abs(keep[-1][2] - s_cut) > 1e-9:
        keep.append((cp[0], cp[1], s_cut))
    if not rem or abs(rem[0][2] - s_cut) > 1e-9:
        rem.insert(0, (cp[0], cp[1], s_cut))
    if side == 0:
        # 转角在起点：保留段 = [s_cut, L]，被切段 = [0, s_cut]（J->A），反转为 A->J
        keep = [p for p in keep if p[2] >= s_cut - 1e-9]
        rem = list(reversed([p for p in rem if p[2] <= s_cut + 1e-9]))
    else:
        keep = [p for p in keep if p[2] <= s_cut + 1e-9]
    return _norm(keep), _norm(rem)


def _bezier(A, C, B, step=2.0):
    """二次 Bezier：A 为起点、C 为控制点（转角点）、B 为终点。
    端点切线分别为 A->C、C->B，因此与两侧路径天然共线。"""
    approx = math.hypot(C[0] - A[0], C[1] - A[1]) + math.hypot(B[0] - C[0], B[1] - C[1])
    n = max(4, int(approx / step) + 2)
    out, s, prev = [], 0.0, None
    for k in range(n):
        tt = k / (n - 1.0)
        mt = 1.0 - tt
        x = mt * mt * A[0] + 2 * mt * tt * C[0] + tt * tt * B[0]
        y = mt * mt * A[1] + 2 * mt * tt * C[1] + tt * tt * B[1]
        if prev is not None:
            s += math.hypot(x - prev[0], y - prev[1])
        out.append((x, y, s))
        prev = (x, y)
    return out


def insert_fillets(recs, turn_thr, min_turn, d_max, max_len, depth=None):
    """在每个转角对插入圆角弧 R，并保留残段以维持连通性。
    返回 (recs, 统计 dict)"""
    import collections
    depth = d_max if depth is None else depth

    orig = set()
    for r in recs:
        orig.add(r["keys"][0])
        orig.add(r["keys"][1])

    ends_at = collections.defaultdict(list)
    for ri, r in enumerate(recs):
        for side in (0, 1):
            k = r["keys"][side]
            if k in orig:
                ends_at[k].append((ri, side))

    new_recs = []
    counter = [0]
    st = {"junctions": 0, "corners": 0, "fillets": 0, "stubs": 0, "skipped": 0}

    for J in sorted(ends_at, key=str):
        ends = ends_at[J]
        if len(ends) < 2:
            continue
        valid, jcoord = [], {}
        for ri, side in ends:
            r = recs[ri]
            if r["keys"][side] != J or r["length"] < 1.0:
                continue
            v, dep = _leave_dir(r["pts"], side, depth)
            if v is None or dep < 1e-6:
                continue
            valid.append((ri, side, v))
            jcoord[ri] = (r["pts"][0][0], r["pts"][0][1]) if side == 0 else (r["pts"][-1][0], r["pts"][-1][1])
        if len(valid) < 2:
            continue
        st["junctions"] += 1

        corners = []
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                ri, si, vi = valid[i]
                rj, sj, vj = valid[j]
                if ri == rj:
                    continue
                c = max(-1.0, min(1.0, vi[0] * vj[0] + vi[1] * vj[1]))
                turn = 180.0 - math.degrees(math.acos(c))
                if turn >= min_turn:
                    corners.append((i, j, turn))
        if not corners:
            continue

        part = sorted({i for (i, j, _) in corners} | {j for (i, j, _) in corners})
        Lmin = min(recs[valid[i][0]]["length"] for i in part)
        d = min(d_max, 0.30 * Lmin)
        if d < 1.0:
            st["skipped"] += 1
            continue

        cut = {}
        for i in part:
            ri, side, v = valid[i]
            r = recs[ri]
            pt, s_cut = _point_at(r["pts"], side, d)
            key = "S%d" % counter[0]
            counter[0] += 1
            keep, rem = _cut(r["pts"], side, s_cut)
            if len(rem) >= 2 and rem[-1][2] - rem[0][2] > 1e-6:
                stub = {"nids": r["nids"], "keys": (key, J), "pts": rem,
                        "length": rem[-1][2] - rem[0][2], "cls": r["cls"]}
                stub["maxturn"] = max_turn_deg(stub["pts"])
                stub["kind"] = "N" if stub["maxturn"] < turn_thr else "R"
                new_recs.append(stub)
                st["stubs"] += 1
            if len(keep) < 2 or keep[-1][2] - keep[0][2] < 1.0:
                continue
            r["pts"] = keep
            r["length"] = keep[-1][2] - keep[0][2]
            r["maxturn"] = max_turn_deg(keep)
            r["kind"] = "N" if r["maxturn"] < turn_thr else "R"
            r["keys"] = (key, r["keys"][1]) if side == 0 else (r["keys"][0], key)
            cut[i] = (key, pt, r["cls"])

        for (i, j, turn) in corners:
            if i not in cut or j not in cut:
                continue
            ka, pa, cls = cut[i]
            kb, pb, _ = cut[j]
            C = jcoord[valid[i][0]]
            fpts = _bezier(pa, C, pb)
            L = fpts[-1][2] - fpts[0][2]
            if L < 2.0 or L > max_len:
                continue
            f = {"nids": [], "keys": (ka, kb), "pts": fpts, "length": L, "cls": cls}
            f["maxturn"] = max_turn_deg(fpts)
            f["kind"] = "N" if f["maxturn"] < turn_thr else "R"
            new_recs.append(f)
            st["corners"] += 1
            if f["kind"] == "R":
                st["fillets"] += 1

    return recs + new_recs, st


# ============================================================
# 4) 最大三点转向角（原脚本判据）
# ============================================================
def max_turn_deg(pts):
    if len(pts) < 3:
        return 0.0
    mt = 0.0
    for i in range(1, len(pts) - 1):
        x1, y1, _ = pts[i - 1]
        x2, y2, _ = pts[i]
        x3, y3, _ = pts[i + 1]
        v1 = (x2 - x1, y2 - y1)
        v2 = (x3 - x2, y3 - y2)
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        c = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        mt = max(mt, math.degrees(math.acos(c)))
    return mt


# ============================================================
# 5) 主流程
# ============================================================
def build(ap, pkl_path, out_dir, r_min=None, win=None, min_r_len=None, turn_thr=None,
          fillet_min_turn=None, fillet_d=None, fillet_len=None):
    os.makedirs(out_dir, exist_ok=True)
    r_min = R_MIN if r_min is None else r_min
    win = CURV_WIN if win is None else win
    min_r_len = MIN_R_LEN if min_r_len is None else min_r_len
    turn_thr = TURN_THR if turn_thr is None else turn_thr
    fillet_min_turn = FILLET_MIN_TURN if fillet_min_turn is None else fillet_min_turn
    fillet_d = FILLET_D if fillet_d is None else fillet_d
    fillet_len = FILLET_LEN if fillet_len is None else fillet_len
    U, xy, ll = load_graph(pkl_path)
    chains, junction = build_chains(U)

    # --- 收集 piece ---
    pieces = []   # (nids, pts)
    for ch in chains:
        pts = chain_pts(U, xy, ch)
        for a, b in split_pieces(pts, r_min, win, min_r_len):
            nids = ch[a:b + 1]
            if len(nids) < 2:
                continue
            pieces.append((nids, pts[a:b + 1]))

    # --- N/R 判定（原脚本口径）---
    recs = []
    for nids, pts in pieces:
        mt = max_turn_deg(pts)
        recs.append({
            "nids": nids, "pts": pts, "maxturn": mt,
            "kind": "N" if mt < turn_thr else "R",
            "length": pts[-1][2] - pts[0][2],
            "cls": Counter(U[nids[i]][nids[i + 1]].get("cls", 4) for i in range(len(nids) - 1)).most_common(1)[0][0],
            "keys": (nids[0], nids[-1]),
        })

    # --- 角点圆角化 ---
    recs, fstat = insert_fillets(recs, turn_thr, fillet_min_turn, fillet_d, fillet_len)

    # --- 端点命名：原始节点按纬度降序/经度升序，合成切口点另编号 ---
    bnodes = set()
    for r in recs:
        bnodes.add(r["keys"][0])
        bnodes.add(r["keys"][1])
    order = sorted([n for n in bnodes if not isinstance(n, str)],
                   key=lambda n: (-ll[n][0], ll[n][1]))
    name_of = {n: "Point (%d)" % (k + 1) for k, n in enumerate(order)}
    snodes = sorted([n for n in bnodes if isinstance(n, str)])
    for k, n in enumerate(snodes, 1):
        name_of[n] = "Point (s%d)" % k

    # --- 命名 N-1.. / R-1..（与 build_paths_json.py 一致：先全部 N 再全部 R）---
    ns = [r for r in recs if r["kind"] == "N"]
    rs = [r for r in recs if r["kind"] == "R"]
    paths, meta = [], {}
    pname_counter = [0]

    def emit(group, prefix):
        for i, r in enumerate(group, 1):
            pname = "%s-%d" % (prefix, i)
            pts_out = []
            for (x, y, s) in r["pts"]:
                pts_out.append({
                    "pointname": pname_counter[0],
                    "x": round(f32(x), 6), "y": 0.0, "z": round(f32(y), 6), "roa": 0.0,
                })
                pname_counter[0] += 1
            if len(pts_out) < 2:
                continue
            paths.append({
                "pathname": pname,
                "startpoint": name_of[r["keys"][0]],
                "endpoint": name_of[r["keys"][1]],
                "pathLength": round(f32(r["length"]), 6),
                "points": pts_out,
            })
            meta[pname] = {
                "kind": prefix, "cls": int(r["cls"]),
                "cls_name": CLS_NAME.get(int(r["cls"]), "?"),
                "is_runway": int(r["cls"]) == 3,
                "u": name_of[r["keys"][0]], "v": name_of[r["keys"][1]],
                "length": round(r["length"], 3),
                "maxturn": round(r["maxturn"], 3),
                "n_points": len(pts_out),
            }

    emit(ns, "N")
    emit(rs, "R")

    # --- InitMatrixPath.json ---
    with open(os.path.join(out_dir, "InitMatrixPath.json"), "w", encoding="utf-8") as f:
        json.dump({"paths": paths}, f, ensure_ascii=False, indent=1)

    # --- InitPath.json（同 schema 副本）---
    with open(os.path.join(out_dir, "InitPath.json"), "w", encoding="utf-8") as f:
        json.dump({"paths": paths}, f, ensure_ascii=False, indent=1)

    # --- 1m 重采样 ---
    p1m = []
    n_1m = 0
    for p in paths:
        pts = [(q["x"], q["z"]) for q in p["points"]]
        out = []
        along = 0.0
        for i in range(len(pts) - 1):
            ax, az = pts[i]
            bx, bz = pts[i + 1]
            L = math.hypot(bx - ax, bz - az)
            if L < 1e-9:
                continue
            nstep = max(1, int(round(L / STEP_M)))
            for s in range(nstep):
                t = s / nstep
                out.append({"pointname": n_1m,
                            "x": round(ax + (bx - ax) * t, 3),
                            "y": 0.0,
                            "z": round(az + (bz - az) * t, 3),
                            "roa": 0.0})
                n_1m += 1
            along += L
        # 收尾点
        out.append({"pointname": n_1m, "x": round(pts[-1][0], 3), "y": 0.0,
                    "z": round(pts[-1][1], 3), "roa": 0.0})
        n_1m += 1
        p1m.append({"pathname": p["pathname"], "startpoint": p["startpoint"],
                    "endpoint": p["endpoint"], "pathLength": p["pathLength"],
                    "points": out})
    with open(os.path.join(out_dir, "InitMatrixPath-1m.json"), "w", encoding="utf-8") as f:
        json.dump({"paths": p1m}, f, ensure_ascii=False, indent=1)

    # --- 邻接矩阵（端点共享，对称 0/1）---
    node2paths = {}
    for p in paths:
        for nm in (p["startpoint"], p["endpoint"]):
            node2paths.setdefault(nm, []).append(p["pathname"])
    names = [p["pathname"] for p in paths]
    idx = {n: i for i, n in enumerate(names)}
    A = np.zeros((len(names), len(names)), dtype=np.int8)
    for nm, plist in node2paths.items():
        for i in range(len(plist)):
            for j in range(i + 1, len(plist)):
                a, b = plist[i], plist[j]
                if a == b:
                    continue
                A[idx[a], idx[b]] = 1
                A[idx[b], idx[a]] = 1
    import csv
    with open(os.path.join(out_dir, "connection_matrix.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + names)
        for i, nm in enumerate(names):
            w.writerow([nm] + A[i].tolist())

    with open(os.path.join(out_dir, "path_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    # --- 报告 ---
    nl = [m["length"] for m in meta.values() if m["kind"] == "N"]
    rl = [m["length"] for m in meta.values() if m["kind"] == "R"]
    mt_n = [m["maxturn"] for m in meta.values() if m["kind"] == "N"]
    mt_r = [m["maxturn"] for m in meta.values() if m["kind"] == "R"]
    rep = {
        "airport": ap,
        "source": os.path.basename(pkl_path),
        "n_paths": len(paths), "n_N": len(ns), "n_R": len(rs),
        "r_share_pct": round(100.0 * len(rs) / max(1, len(paths)), 2),
        "n_endpoint_nodes": len(order),
        "n_chains": len(chains), "n_junction_nodes": len(junction),
        "n_internal_points_1m": n_1m,
        "total_length_m": round(sum(m["length"] for m in meta.values()), 1),
        "adjacency_nnz": int(A.sum()),
        "N_len_p50": round(statistics.median(nl), 2) if nl else 0,
        "R_len_p50": round(statistics.median(rl), 2) if rl else 0,
        "N_len_p90": round(float(np.percentile(nl, 90)), 2) if nl else 0,
        "R_len_p90": round(float(np.percentile(rl, 90)), 2) if rl else 0,
        "N_maxturn_p90": round(float(np.percentile(mt_n, 90)), 2) if mt_n else 0,
        "R_maxturn_p50": round(float(statistics.median(mt_r)), 2) if mt_r else 0,
        "R_maxturn_p90": round(float(np.percentile(mt_r, 90)), 2) if mt_r else 0,
        "fillet": fstat,
        "params": {"TURN_THR": turn_thr, "R_MIN": r_min, "CURV_WIN": win,
                   "BRIDGE_GAP": BRIDGE_GAP, "MIN_R_LEN": min_r_len,
                   "MIN_SPAN": MIN_SPAN, "STEP_M": STEP_M,
                   "FILLET_MIN_TURN": fillet_min_turn, "FILLET_D": fillet_d,
                   "FILLET_LEN": fillet_len},
    }
    with open(os.path.join(out_dir, "build_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    return rep


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("airport")
    ap_.add_argument("--root", default=os.environ.get("AMELIA_GRAPH_ROOT", ""),
                     help="Amelia-10 graph_data_a10v01os directory (or AMELIA_GRAPH_ROOT)")
    ap_.add_argument("--out", default=_REPO)
    ap_.add_argument("--subdir", default="graph",
                     help="输出子目录名，默认 graph；无圆角对照用 graph_nofillet")
    ap_.add_argument("--r-min", type=float, default=None, help="曲率速率阈值 deg/m")
    ap_.add_argument("--turn-thr", type=float, default=None, help="N/R 判定阈值 度")
    ap_.add_argument("--fillet-min-turn", type=float, default=None,
                     help="转角 >= 该值才插圆角弧；设 180 即关闭圆角化")
    ap_.add_argument("--fillet-d", type=float, default=None)
    ap_.add_argument("--fillet-len", type=float, default=None)
    a = ap_.parse_args()
    if not a.root:
        ap_.error("--root is required: point it at the Amelia-10 graph directory or set AMELIA_GRAPH_ROOT")
    r = build(a.airport, os.path.join(a.root, a.airport, "semantic_graph.pkl"),
              os.path.join(a.out, a.airport.upper(), a.subdir),
              r_min=a.r_min, turn_thr=a.turn_thr,
              fillet_min_turn=a.fillet_min_turn, fillet_d=a.fillet_d,
              fillet_len=a.fillet_len)
    print(json.dumps(r, ensure_ascii=False, indent=1))
