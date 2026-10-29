# -*- coding: utf-8 -*-
"""
Amelia 轨迹清洗 + 转西安式轨迹 JSON（每场 500 条：进/离场各 250）
==================================================================
清洗链（每级留存数量全部记录，funnel 可追溯）：
  L0 逐小时 CSV -> 按 ID 流式聚成轨迹（跨文件拼接，绝对时间 = 文件时间戳 + Frame）
  L1 断点拆分：相邻点间隔 > GAP_SPLIT 即切段，取最长段
  L2 目标类型：全部点 Type == 0（航空器）
  L3 滑行段裁剪：保留最长一段 Speed <= SPD_MAX 连续段（去进近/起飞滑跑）
  L4 路网内裁剪：去掉首尾到路网 > D_NET 的点，要求 >= IN_NET_FRAC 点网内
  L5 门槛：时长 >= DUR_MIN、网内里程 >= LEN_MIN、平均速度 >= SPD_MIN
  L6 进/离场判定（跑道锚定）：比较首/末点到跑道类(class 3)路径的距离，
      贴跑道的一端即跑道端；两端同/异不可辨的用高度趋势兜底，仍不可辨则丢弃
  L7 固定种子分层抽样：进场 250 / 离场 250

坐标：Amelia 的 x/y 单位 km（机场局部笛卡尔）-> 算法 x_m = x*1000, z_m = y*1000。
为让核心算法零改动复用，转换文件把 (x_m, z_m) 占位写入 originLat/originLon，
运行器再把 convert_coordinates 打补丁成恒等映射（见 run_amelia.py）。
"""
import os
import sys
import json
import math
import glob
import random
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

GAP_SPLIT = 30        # s
SPD_MAX = 40.0        # kn
D_NET = 60.0          # m
IN_NET_FRAC = 0.90
DUR_MIN = 180         # s（进场轨迹被 40kn 裁掉进近+滑跑后普遍偏短，时长只作代理量）
LEN_MIN = 500.0       # m（网内里程才是路线覆盖度的有效度量，作主判据）
SPD_MIN = 3.0         # kn
RUNWAY_NEAR = 80.0    # m，判定"贴在跑道上"
ALT_SEP = 200.0       # ft，兜底用
SEED = 20261026


# ============================================================
# 路网索引
# ============================================================
def load_net_points(graph_dir):
    with open(os.path.join(graph_dir, "InitMatrixPath-1m.json"), encoding="utf-8") as f:
        paths = json.load(f)["paths"]
    with open(os.path.join(graph_dir, "path_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    pts, owner, is_rw = [], [], []
    for p in paths:
        rw = bool(meta.get(p["pathname"], {}).get("is_runway", False))
        for q in p["points"]:
            pts.append((q["x"], q["z"]))
            owner.append(p["pathname"])
            is_rw.append(rw)
    A = np.array(pts, dtype=float)
    is_rw = np.array(is_rw, dtype=bool)
    return {
        "tree": cKDTree(A),
        "pts": A,
        "owner": owner,
        "tree_rw": cKDTree(A[is_rw]) if is_rw.any() else None,
        "n": len(A),
        "n_rw": int(is_rw.sum()),
    }


# ============================================================
# 轨迹流式拼接
# ============================================================
def iter_tracks(traj_dir, stats, max_files=None, org_only=False):
    files = sorted(glob.glob(os.path.join(traj_dir, "*.csv")))
    if max_files:
        files = files[:max_files]
    stats["n_files"] = len(files)
    stats["n_rows"] = 0
    stats["n_raw_tracks"] = 0
    stats["n_int_dropped"] = 0

    open_tr = {}
    for fi, path in enumerate(files):
        if fi % 40 == 0:
            print("[scan] %d/%d 文件  已积累原始轨迹 %d 条"
                  % (fi, len(files), stats["n_raw_tracks"]), flush=True)
        base = os.path.basename(path)
        try:
            ts = int(base.split("_")[-1].split(".")[0])
        except Exception:
            continue
        cols = ["Frame", "ID", "Altitude", "Speed", "Lat", "Lon", "Type", "x", "y"]
        if org_only:
            cols.append("Interp")
        df = pd.read_csv(path, usecols=cols, dtype={"Interp": str} if org_only else None)
        if org_only:
            n0 = len(df)
            df = df[df["Interp"].astype(str) != "[INT]"]
            stats["n_int_dropped"] = stats.get("n_int_dropped", 0) + (n0 - len(df))
        stats["n_rows"] += len(df)
        df["t"] = ts + df["Frame"]

        for pid, sub in df.groupby("ID", sort=False):
            sub = sub.sort_values("t")
            rows = [tuple(r) for r in sub[["t", "Lat", "Lon", "x", "y",
                                           "Altitude", "Speed", "Type"]].to_numpy(dtype=float)]
            if not rows:
                continue
            pid = int(pid)
            prev = open_tr.get(pid)
            if prev is not None and rows[0][0] - prev[-1][0] <= GAP_SPLIT + 5:
                prev.extend(rows)
            else:
                if prev is not None:
                    stats["n_raw_tracks"] += 1
                    yield pid, prev
                open_tr[pid] = rows

        fend = ts + 3600
        for pid in list(open_tr.keys()):
            if open_tr[pid][-1][0] < fend - 120:
                stats["n_raw_tracks"] += 1
                yield pid, open_tr.pop(pid)

    for pid, rows in open_tr.items():
        stats["n_raw_tracks"] += 1
        yield pid, rows


# ============================================================
# 几何工具
# ============================================================
def split_gaps(rows):
    if not rows:
        return []
    segs, cur = [], [rows[0]]
    for r in rows[1:]:
        if r[0] - cur[-1][0] > GAP_SPLIT:
            segs.append(cur)
            cur = [r]
        else:
            cur.append(r)
    segs.append(cur)
    return max(segs, key=len)


def speed_trim(rows):
    best, cur = [], []
    for r in rows:
        if r[6] <= SPD_MAX:
            cur.append(r)
        else:
            if len(cur) > len(best):
                best = cur
            cur = []
    if len(cur) > len(best):
        best = cur
    return best


def net_trim(rows, net):
    if not rows:
        return [], 0.0
    P = np.array([[r[3] * 1000.0, r[4] * 1000.0] for r in rows], dtype=float)
    d, _ = net["tree"].query(P, k=1)
    idx = np.where(d <= D_NET)[0]
    if len(idx) == 0:
        return [], 0.0
    a, b = int(idx[0]), int(idx[-1]) + 1
    return rows[a:b], float(np.mean(d[a:b] <= D_NET))


def nearest_path(r, net, runway=False):
    p = np.array([[r[3] * 1000.0, r[4] * 1000.0]], dtype=float)
    if runway and net["tree_rw"] is not None:
        d, i = net["tree_rw"].query(p, k=1)
        # 换算成"最近跑道点"的距离；用跑道点云即可
        return float(d[0])
    d, i = net["tree"].query(p, k=1)
    return net["owner"][int(i[0])], float(d[0])


# ============================================================
# 主流程
# ============================================================
def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("airport")
    ap_.add_argument("--traj-root", default=r"E:\科研2026\数据\Amelia-10\data\traj_data_a10v08\raw_trajectories")
    ap_.add_argument("--out", default=r"E:\科研2026\轨迹路线识别\返修实验\R1-4")
    ap_.add_argument("--n-per-class", type=int, default=250)
    ap_.add_argument("--max-files", type=int, default=None)
    ap_.add_argument("--export", action="store_true", help="抽样并导出 JSON")
    ap_.add_argument("--org-only", action="store_true",
                     help="只保留 Interp=[ORG] 的原始报点（剔除 [INT] 插值点）")
    ap_.add_argument("--traj-subdir", default="traj_raw")
    ap_.add_argument("--graph-subdir", default="graph_nofillet",
                     help="用哪套路网做离网裁剪/跑道锚定（默认 graph_nofillet，"
                          "与 KDCA/KJFK 清洗口径一致）")
    a = ap_.parse_args()

    ap = a.airport.lower()
    AP = a.airport.upper()
    out_root = os.path.join(a.out, AP)
    graph_dir = os.path.join(out_root, a.graph_subdir)
    traj_dir = os.path.join(a.traj_root, ap)

    net = load_net_points(graph_dir)
    print("[net] 1m 点 %d（跑道类 %d）" % (net["n"], net["n_rw"]))

    stats = {}
    funnel = []
    cand = []
    c = defaultdict(int)

    for pid, rows in iter_tracks(traj_dir, stats, a.max_files, a.org_only):
        c["L0"] += 1
        if len(set(int(r[7]) for r in rows)) != 1 or rows[0][7] != 0.0:
            continue
        c["L1"] += 1
        seg = split_gaps(rows)
        if len(seg) < 30:
            continue
        c["L2"] += 1
        sp = speed_trim(seg)
        if len(sp) < 60:
            continue
        c["L3"] += 1
        tr, in_frac = net_trim(sp, net)
        if not tr or in_frac < IN_NET_FRAC:
            continue
        c["L4"] += 1

        Arr = np.array([[r[3] * 1000.0, r[4] * 1000.0] for r in tr], dtype=float)
        L = float(np.sum(np.hypot(*np.diff(Arr, axis=0).T)))
        dur = tr[-1][0] - tr[0][0]
        vmean = float(np.mean([r[6] for r in tr]))
        if dur < DUR_MIN or L < LEN_MIN or vmean < SPD_MIN:
            continue
        c["L5"] += 1

        # ---- 进/离场判定：跑道锚定 ----
        d0rw = nearest_path(tr[0], net, runway=True) if net["tree_rw"] is not None else 1e9
        d1rw = nearest_path(tr[-1], net, runway=True) if net["tree_rw"] is not None else 1e9
        if d0rw <= RUNWAY_NEAR < d1rw:
            flio, how = "A", "runway_start"
        elif d1rw <= RUNWAY_NEAR < d0rw:
            flio, how = "D", "runway_end"
        else:
            k = max(3, int(len(seg) * 0.15))
            alt0 = float(np.median([r[5] for r in seg[:k]]))
            alt1 = float(np.median([r[5] for r in seg[-k:]]))
            if alt0 - alt1 > ALT_SEP:
                flio, how = "A", "altitude"
            elif alt1 - alt0 > ALT_SEP:
                flio, how = "D", "altitude"
            else:
                c["L6_ambiguous"] += 1
                continue
        c["L6"] += 1

        sp0, dd0 = nearest_path(tr[0], net)
        sp1, dd1 = nearest_path(tr[-1], net)
        cand.append({
            "tid": "%s_%d_%d" % (AP, pid, int(rows[0][0])),
            "pid": pid, "flio": flio, "flio_rule": how, "n": len(tr),
            "t0": tr[0][0], "t1": tr[-1][0], "dur": dur, "len_m": L,
            "v_mean": vmean, "v_max": float(np.max([r[6] for r in tr])),
            "d_start_runway": d0rw, "d_end_runway": d1rw,
            "start_path_hint": sp0, "start_d": dd0,
            "end_path_hint": sp1, "end_d": dd1,
            "rows": np.asarray(tr, dtype=np.float64),
        })

    funnel = [
        {"stage": "L0_raw_tracks", "count": c["L0"], "note": "文件 %d，行 %d" % (stats["n_files"], stats["n_rows"])},
        {"stage": "L1_type_aircraft", "count": c["L1"]},
        {"stage": "L2_gap_split", "count": c["L2"]},
        {"stage": "L3_taxi_speed_trim", "count": c["L3"]},
        {"stage": "L4_in_network", "count": c["L4"]},
        {"stage": "L5_dur/len/speed_gate", "count": c["L5"]},
        {"stage": "L6_flio_resolved", "count": c["L6"], "note": "不可辨 %d" % c["L6_ambiguous"]},
    ]
    nA = sum(1 for x in cand if x["flio"] == "A")
    print("[cand] %d 条（进 %d / 离 %d）" % (len(cand), nA, len(cand) - nA))
    for f in funnel:
        print("   ", f)

    # ---- 分层抽样 ----
    rng = random.Random(SEED)
    picked = []
    for flio in ("A", "D"):
        pool = [x for x in cand if x["flio"] == flio]
        pool.sort(key=lambda x: x["tid"])
        if len(pool) > a.n_per_class:
            pool = rng.sample(pool, a.n_per_class)
        picked.extend(pool)
        funnel.append({"stage": "L7_sample_%s" % flio, "count": len(pool),
                       "note": "候选 %d" % sum(1 for x in cand if x["flio"] == flio)})
        print("[sample] %s: 候选 %d -> 抽 %d" % (flio, sum(1 for x in cand if x["flio"] == flio), len(pool)))

    with open(os.path.join(out_root, "cleaning_funnel.json"), "w", encoding="utf-8") as f:
        json.dump({"airport": AP, "funnel": funnel, "seed": SEED,
                   "n_candidates": len(cand),
                   "params": {"GAP_SPLIT": GAP_SPLIT, "SPD_MAX": SPD_MAX, "D_NET": D_NET,
                              "IN_NET_FRAC": IN_NET_FRAC, "DUR_MIN": DUR_MIN,
                              "LEN_MIN": LEN_MIN, "SPD_MIN": SPD_MIN,
                              "RUNWAY_NEAR": RUNWAY_NEAR, "ALT_SEP": ALT_SEP}},
                  f, ensure_ascii=False, indent=1)

    if not a.export:
        return

    # ---- 导出 ----
    raw_dir = os.path.join(out_root, a.traj_subdir)
    meta_dir = os.path.join(out_root, "traj_meta")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    metas = []
    for x in picked:
        rows = x["rows"]
        recs = []
        for r in rows:
            x_m = float(r[3]) * 1000.0
            z_m = float(r[4]) * 1000.0
            recs.append({
                "actp": "Track",
                "time": str(int(r[0])),
                "properties": {
                    # 保留真实经纬度（便于出图与回溯）
                    "originLat": float(r[1]), "originLon": float(r[2]),
                    "lat": float(r[1]), "lon": float(r[2]),
                    # 算法使用的米制平面坐标（Amelia 局部笛卡尔，km -> m）
                    "unity_x": x_m, "unity_z": z_m,
                    "alt_ft": float(r[5]), "spd_kn": float(r[6]),
                    "callsign": str(x["pid"]), "flio": x["flio"],
                    "rway": "", "agency": AP,
                },
            })
        with open(os.path.join(raw_dir, x["tid"] + ".json"), "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False)

        meta = {k: v for k, v in x.items() if k != "rows"}
        for k in ("t0", "t1", "dur", "len_m", "v_mean", "v_max",
                  "d_start_runway", "d_end_runway", "start_d", "end_d"):
            meta[k] = float(meta[k])
        metas.append(meta)
        with open(os.path.join(meta_dir, x["tid"] + ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)

    with open(os.path.join(out_root, "sample_list.json"), "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=1)
    print("[done] 导出 %d 条 -> %s" % (len(metas), out_root))


if __name__ == "__main__":
    main()
