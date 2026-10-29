# -*- coding: utf-8 -*-
"""
Amelia 跨机场泛化实验 —— 运行器
================================
设计要点：**不重写主流程**。直接复用 `算法核心代码/main.py` 的 process_flight
（含 10° -> 7° 折返兜底、候选生成、打分选优），只 patch 三个机场相关函数：

  main.transform_track_json          -> Amelia 已带 unity_x/unity_z，直接落盘
  main.get_start_end_from_trajectory -> amelia_adapter.get_start_end_xy（跑道锚定）
  main.extract_potential_r           -> amelia_adapter.extract_potential_r_xy

配置通过改写 config 模块属性注入（必须在 import graph_pathfinder 之前完成，
因为它在 import 时就用 connection_matrix.csv 建全局图）。

用法：
  python run_amelia.py kdca --limit 10        # 试点
  python run_amelia.py kdca                   # 全量 500
"""
import os
import sys
import json
import math
import time
import shutil
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ALGO = r"E:\科研2026\轨迹路线识别\算法部分\算法核心代码"
sys.path.insert(0, ALGO)
sys.path.insert(0, HERE)

OUT_ROOT_DEFAULT = r"E:\科研2026\轨迹路线识别\返修实验\R1-4"


# ============================================================
# 1) 注入机场配置（必须在导入核心模块之前）
# ============================================================
def inject_config(graph_dir):
    import config
    config.INIT_MATRIX_PATH = os.path.join(graph_dir, "InitMatrixPath.json")
    config.INIT_MATRIX_PATH_1M = os.path.join(graph_dir, "InitMatrixPath-1m.json")
    config.INIT_PATH = os.path.join(graph_dir, "InitPath.json")
    config.CONNECTION_MATRIX = os.path.join(graph_dir, "connection_matrix.csv")
    # 阈值保持与西安实验完全一致
    config.GRID_CELL_SIZE = 100.0
    config.GRID_SEARCH_RANGE = 2
    config.R_MATCH_DIST = 15.0
    config.R_TOP_K = 3
    config.ANGLE_THRESHOLD = 10
    config.ANGLE_THRESHOLD_ESCALATED = 7
    config.ESCALATE_ON_REVERSAL = True
    config.PATH_ANGLE_THRESHOLD = 110
    return config


# ============================================================
# 2) 指标
# ============================================================
def route_points(best_path, path_pts):
    out = []
    for name in best_path or []:
        out.extend(path_pts.get(name, []))
    return out


def compute_metrics(track_file, best_path, path_pts, net_tree=None):
    with open(track_file, encoding="utf-8") as f:
        data = json.load(f)
    tp = []
    for it in data:
        pr = it.get("properties", {})
        x, z = pr.get("unity_x"), pr.get("unity_z")
        if x is not None and z is not None:
            tp.append((float(x), float(z)))
    if not tp or not best_path:
        return None
    A = np.array(tp, dtype=float)
    B = np.array(route_points(best_path, path_pts), dtype=float)
    if len(B) == 0:
        return None
    # 逐点最近距离：用 KD 树（原分块欧氏矩阵在长路线上会 OOM）
    try:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(B).query(A, k=1)
        d = np.asarray(d, dtype=float)
    except Exception:
        d = np.empty(len(A))
        step = 500
        for i in range(0, len(A), step):
            blk = A[i:i + step]
            d[i:i + step] = np.sqrt(((blk[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min(1)
    out = {
        "n_track_points": int(len(A)),
        "n_route_points": int(len(B)),
        "mean_dist_m": float(d.mean()),
        "median_dist_m": float(np.median(d)),
        "p90_dist_m": float(np.percentile(d, 90)),
        "p95_dist_m": float(np.percentile(d, 95)),
        "max_dist_m": float(d.max()),
        "cov_10m": float((d <= 10).mean()),
        "cov_15m": float((d <= 15).mean()),
        "cov_30m": float((d <= 30).mean()),
    }
    # 下界参照：轨迹点到全路网（不限路线）的最近距离
    if net_tree is not None:
        lb, _ = net_tree.query(A, k=1)
        lb = np.asarray(lb, dtype=float)
        out["lb_mean_dist_m"] = float(lb.mean())
        out["lb_median_dist_m"] = float(np.median(lb))
        out["lb_p95_dist_m"] = float(np.percentile(lb, 95))
    return out


def count_dup_nodes(route):
    seen, dup = set(), 0
    for n in route or []:
        if n in seen:
            dup += 1
        seen.add(n)
    return dup


def load_path_lengths(init_matrix_path):
    with open(init_matrix_path, encoding="utf-8") as f:
        return {p["pathname"]: float(p["pathLength"]) for p in json.load(f)["paths"]}


# ============================================================
# 3) 主流程
# ============================================================
def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("airport")
    ap_.add_argument("--out-root", default=OUT_ROOT_DEFAULT)
    ap_.add_argument("--limit", type=int, default=None, help="只跑前 N 条（试点）")
    ap_.add_argument("--tag", default=None, help="输出子目录标签，默认 pilot/full")
    ap_.add_argument("--keep-stages", action="store_true", help="保留五阶段中间产物")
    ap_.add_argument("--path-angle", type=float, default=110.0,
                     help="相邻路径夹角下限（度），默认 110（与西安一致）")
    ap_.add_argument("--angle", type=float, default=10.0, help="潜在R转角阈值（度）")
    ap_.add_argument("--r-match", type=float, default=15.0, help="轨迹点-R路径距离阈值（米）")
    ap_.add_argument("--r-topk", type=int, default=3, help="每点保留的最近R个数")
    ap_.add_argument("--only", default=None,
                     help="只跑指定航班（逗号分隔的关键字，星号通配）")
    ap_.add_argument("--graph-subdir", default="graph",
                     help="路网子目录：graph（圆角）/ graph_nofillet / graph_raw（纯junction）")
    a = ap_.parse_args()

    AP = a.airport.upper()
    airport_root = os.path.join(a.out_root, AP)
    graph_dir = os.path.join(airport_root, a.graph_subdir)
    traj_dir = os.path.join(airport_root, "traj_raw")

    tag = a.tag or ("pilot" if a.limit else "full")
    out_dir = os.path.join(airport_root, tag)
    os.makedirs(out_dir, exist_ok=True)

    config = inject_config(graph_dir)
    config.PATH_ANGLE_THRESHOLD = a.path_angle
    config.ANGLE_THRESHOLD = a.angle
    config.R_MATCH_DIST = a.r_match
    config.R_TOP_K = a.r_topk
    config.R_MATCH_DIST = a.r_match
    config.R_TOP_K = a.r_topk

    # --- 导入核心模块（此时 config 已注入）---
    import main as core
    import amelia_adapter as AA
    from potential_r_matcher import GridPathMatcher

    # --- patch 三处机场相关函数 ---
    def _identity_convert(trajectory_file, output_dir=None):
        with open(trajectory_file, encoding="utf-8") as f:
            data = json.load(f)
        name = os.path.splitext(os.path.basename(trajectory_file))[0]
        if output_dir is None:
            output_dir = os.path.dirname(trajectory_file)
        os.makedirs(output_dir, exist_ok=True)
        out = os.path.join(output_dir, name + "_converted.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return out

    locator = AA.PathLocator(config.INIT_MATRIX_PATH)

    def _start_end(trajectory_file, stand_file=None, runway_file=None, num_points=None):
        s, e, ds, de = AA.get_start_end_xy(trajectory_file, locator)
        return s, e   # 与原版一致：返回单个 pathname，process_flight 内部再包 list

    core.transform_track_json = _identity_convert
    core.get_start_end_from_trajectory = _start_end
    core.extract_potential_r = AA.extract_potential_r_xy

    path_pts = AA._load_path_points(config.INIT_MATRIX_PATH)
    # 误差指标用 1 m 加密点（与西安口径一致；粗顶点会把误差放大 18–22%）
    path_pts_1m = AA._load_path_points(config.INIT_MATRIX_PATH_1M)
    path_len = load_path_lengths(config.INIT_MATRIX_PATH)
    # 下界参照用：全路网 1m 点云的 KD 树
    net_tree = None
    try:
        from scipy.spatial import cKDTree
        _d1m = json.load(open(config.INIT_MATRIX_PATH_1M, encoding="utf-8"))["paths"]
        _pts = np.array([[q["x"], q["z"]] for p in _d1m for q in p["points"]], dtype=float)
        net_tree = cKDTree(_pts)
        print("[lb] 路网 1m 点 %d，下界参照就绪" % len(_pts), flush=True)
    except Exception as ex:
        print("[lb] 下界参照不可用: %r" % (ex,), flush=True)

    files = sorted(f for f in os.listdir(traj_dir) if f.endswith(".json"))
    metas = {m["tid"]: m for m in json.load(open(os.path.join(airport_root, "sample_list.json"),
                                                 encoding="utf-8"))}
    if a.only:
        import fnmatch
        pats = [s.strip() for s in a.only.split(",") if s.strip()]
        files = [f for f in files if any(fnmatch.fnmatch(f, p) or p in f for p in pats)]
        print("[only] 命中 %d 条: %s" % (len(files), files))
    if a.limit:
        files = files[:a.limit]
    print("[run] %s: %d 条, 输出 -> %s" % (AP, len(files), out_dir))

    print("[run] 初始化网格匹配器 ...")
    t0 = time.time()
    matcher = GridPathMatcher(config.INIT_MATRIX_PATH_1M)
    print("[run] 匹配器就绪 %.1fs" % (time.time() - t0))

    rows = []
    t_start = time.time()
    for k, fn in enumerate(files, 1):
        tid = os.path.splitext(fn)[0]
        fpath = os.path.join(traj_dir, fn)
        # 复用同一暂存目录：不要在循环里 rmtree——批量删除守卫会在累计 50 条目时
        # 直接终止进程（本次踩坑：每次都在第 10 条航班静默中断）
        scratch = os.path.join(out_dir, "_stage")
        t1 = time.time()
        try:
            stat = core.process_flight(fpath, scratch, matcher)
        except Exception as ex:
            print("[ERR] %s: %r" % (tid, ex))
            rows.append({"tid": tid, "error": repr(ex)})
            continue
        dt = time.time() - t1

        try:
            best_file = os.path.join(scratch, "4-最优路径", "best-%s_converted.json" % tid)
            best = json.load(open(best_file, encoding="utf-8")) if os.path.exists(best_file) else {}
            bp = best.get("best_path")
            met = compute_metrics(fpath, bp, path_pts_1m, net_tree) or {}
        except Exception as ex:
            # 指标计算失败不应中断整批；单独记为 metric_err
            print("[METRIC-ERR] %s: %r" % (tid, ex))
            rows.append({"tid": tid, "error": "metric:" + repr(ex),
                         "n_candidates": stat.get("候选路径数")})
            continue

        m = metas.get(tid, {})
        rows.append({
            "tid": tid,
            "flio": m.get("flio"),
            "n_track_points": met.get("n_track_points", m.get("n")),
            "track_len_m": m.get("len_m"),
            "dur_s": m.get("dur"),
            "n_candidates": stat.get("候选路径数"),
            "best_score": stat.get("最优得分"),
            "mean_dist_m": met.get("mean_dist_m"),
            "median_dist_m": met.get("median_dist_m"),
            "p90_dist_m": met.get("p90_dist_m"),
            "p95_dist_m": met.get("p95_dist_m"),
            "max_dist_m": met.get("max_dist_m"),
            "cov_10m": met.get("cov_10m"),
            "cov_15m": met.get("cov_15m"),
            "cov_30m": met.get("cov_30m"),
            "lb_mean_dist_m": met.get("lb_mean_dist_m"),
            "lb_p95_dist_m": met.get("lb_p95_dist_m"),
            "n_route_paths": len(bp) if bp else 0,
            "route_len_m": round(sum(path_len.get(n, 0.0) for n in bp), 1) if bp else None,
            "n_reversal": stat.get("折返数"),
            "n_dup_node": count_dup_nodes(bp),
            "t_step2": stat.get("step2_潜在R识别时间(s)"),
            "t_step3": stat.get("step3_组合路径方案时间(s)"),
            "t_step4": stat.get("step4_最优路径搜索时间(s)"),
            "t_total": stat.get("total_time(s)"),
            "wall_s": round(dt, 3),
            "best_path": bp,
        })
        print("[%d/%d] %s %s 候选=%s 均值=%.1fm 折返=%s 用时=%.1fs"
              % (k, len(files), tid, m.get("flio", "?"), stat.get("候选路径数"),
                 met.get("mean_dist_m", float("nan")), stat.get("折返数"), dt),
              flush=True)

        # 增量落盘，防中途失败丢结果
        if k % 5 == 0 or k == len(files):
            with open(os.path.join(out_dir, "results_partial.json"), "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=1)

    el = time.time() - t_start
    summary = {
        "airport": AP, "n": len(rows), "elapsed_s": round(el, 1),
        "per_flight_s": round(el / max(1, len(rows)), 2),
        "config": {k: getattr(config, k) for k in
                   ("GRID_CELL_SIZE", "GRID_SEARCH_RANGE", "R_MATCH_DIST", "R_TOP_K",
                    "ANGLE_THRESHOLD", "ANGLE_THRESHOLD_ESCALATED", "ESCALATE_ON_REVERSAL",
                    "PATH_ANGLE_THRESHOLD")},
    }
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=1)

    ok = [r for r in rows if r.get("mean_dist_m") is not None]
    if ok:
        arr = lambda k: np.array([r[k] for r in ok if r.get(k) is not None], dtype=float)
        print("\n===== %s 汇总 (n=%d) =====" % (AP, len(ok)))
        for k in ("mean_dist_m", "median_dist_m", "p95_dist_m", "cov_15m", "n_candidates", "t_total"):
            v = arr(k)
            if len(v):
                print("  %-16s mean=%.2f  p50=%.2f  p90=%.2f" %
                      (k, v.mean(), np.median(v), np.percentile(v, 90)))
        print("  折返>0 的航班: %d / %d" % (sum(1 for r in ok if (r.get("n_reversal") or 0) > 0), len(ok)))
    print("[done] %s，总耗时 %.1f min" % (out_dir, el / 60))




if __name__ == "__main__":
    main()
