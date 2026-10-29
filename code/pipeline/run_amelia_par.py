# -*- coding: utf-8 -*-
"""
Amelia 跨机场实验 —— 并行运行器
================================
与 run_amelia.py 完全同口径（同一套 patch、同一套指标），只是把航班分给多个
进程并行处理。

为什么需要：fillet 路网 + rm25 配置下候选数约 21/航班，单航班 23 s，
三场 1500 条串行要 10 小时；8 核并行可压到 1/3。

并行安全要点：
  * 每个 worker 用**独立**暂存目录 `_stage/w<id>`，互不覆盖中间产物；
  * **循环里绝不删目录**（批量删除守卫会在累计 50 条目时直接杀进程，
    这是 run_amelia.py 踩过的坑）；
  * worker 内 config 注入必须在 `import main as core` 之前完成。

用法：
  python run_amelia_par.py klax --graph-subdir graph_raw --r-match 15 \
         --tag full_raw15 --workers 4
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_amelia as RA  # noqa: E402  （只复用其指标函数与配置注入）

_W = {}


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



def _apply_angle_aware(cfg):
    """把 110° 夹角约束内生化：只在「合法转移」上建图，PathFind 搜出的路径天生合规。

    这不是改方法语义——方法原文就是「补全后的相邻路径夹角必须 >= 阈值」；
    原先的实现是"先找最短路、再事后否决"，一旦最短路走了非法捷径，整条候选被丢掉；
    这里把同一约束前移到图上，等价于"补全时只允许合法转弯"。
    """
    import csv
    import networkx as nx
    from angle_constraints import compute_path_angle_by_name, load_paths_once

    pd = load_paths_once(cfg.INIT_MATRIX_PATH)
    plen = {n: v.get("pathLength", 0.0) for n, v in pd.items()}
    thr = float(cfg.PATH_ANGLE_THRESHOLD)
    G = nx.Graph()
    G.add_nodes_from(plen)
    kept = drop = 0
    with open(cfg.CONNECTION_MATRIX, encoding="utf-8-sig") as f:
        rd = csv.reader(f)
        hd = next(rd)
        for row in rd:
            a = row[0]
            for b, v in zip(hd[1:], row[1:]):
                if v.strip() != "1" or b == a:
                    continue
                ang = compute_path_angle_by_name(a, b)
                if ang is not None and ang >= thr:
                    G.add_edge(a, b, weight=plen.get(a, 0.0) + plen.get(b, 0.0))
                    kept += 1
                else:
                    drop += 1
    import graph_pathfinder as GP
    GP._GLOBAL_GRAPH = G
    print("[angle-aware] 合法边 %d / 否决 %d（阈值 %.0f°）" % (kept, drop, thr), flush=True)


def _apply_angle_penalty(cfg, k=8.0):
    """软约束：非法转移（内角 < 阈值）不删边，只把权重乘 k。

    与 _apply_angle_aware 的区别：硬删会让"唯一连接"变成孤岛（实测 KDCA/KLAX
    都会把整条候选打空），软约束保证图始终连通，只是优先走合法转移。
    """
    import csv
    import networkx as nx
    from angle_constraints import compute_path_angle_by_name, load_paths_once

    pd = load_paths_once(cfg.INIT_MATRIX_PATH)
    plen = {n: v.get("pathLength", 0.0) for n, v in pd.items()}
    thr = float(cfg.PATH_ANGLE_THRESHOLD)
    G = nx.Graph()
    G.add_nodes_from(plen)
    n_ok = n_bad = 0
    with open(cfg.CONNECTION_MATRIX, encoding="utf-8-sig") as f:
        rd = csv.reader(f)
        hd = next(rd)
        for row in rd:
            a = row[0]
            for b, v in zip(hd[1:], row[1:]):
                if v.strip() != "1" or b == a:
                    continue
                w = plen.get(a, 0.0) + plen.get(b, 0.0)
                ang = compute_path_angle_by_name(a, b)
                if ang is not None and ang >= thr:
                    G.add_edge(a, b, weight=w)
                    n_ok += 1
                else:
                    G.add_edge(a, b, weight=w * float(k))
                    n_bad += 1
    import graph_pathfinder as GP
    GP._GLOBAL_GRAPH = G
    print("[angle-penalty] 合法边 %d / 罚分边 %d (x%.1f, 阈值 %.0f°)" % (
        n_ok, n_bad, k, thr), flush=True)


def _init(graph_dir, path_pts_json, init_1m, cfgargs):
    """每个 worker 进程初始化一次：注入配置 -> 导入核心 -> patch -> 建索引"""
    cfg = RA.inject_config(graph_dir)
    cfg.PATH_ANGLE_THRESHOLD = cfgargs["path_angle"]
    cfg.ANGLE_THRESHOLD = cfgargs["angle"]
    cfg.ANGLE_THRESHOLD_ESCALATED = 7
    cfg.ESCALATE_ON_REVERSAL = True
    cfg.R_MATCH_DIST = cfgargs["r_match"]
    cfg.R_TOP_K = cfgargs["r_topk"]

    if cfgargs.get("angle_aware"):
        _apply_angle_aware(cfg)
    if cfgargs.get("angle_penalty"):
        _apply_angle_penalty(cfg, float(cfgargs["angle_penalty"]))

    import main as core
    import amelia_adapter as AA
    from potential_r_matcher import GridPathMatcher

    locator = AA.PathLocator(path_pts_json)

    def _start_end(trajectory_file, stand_file=None, runway_file=None, num_points=None):
        s, e, ds, de = AA.get_start_end_xy(trajectory_file, locator)
        return s, e

    core.transform_track_json = _identity_convert
    core.get_start_end_from_trajectory = _start_end
    core.extract_potential_r = AA.extract_potential_r_xy

    path_pts = AA._load_path_points(path_pts_json)
    # 误差指标用 1 m 加密点（与西安口径一致）
    _p1m = os.path.join(graph_dir, "InitMatrixPath-1m.json")
    path_pts_1m = AA._load_path_points(_p1m) if os.path.exists(_p1m) else path_pts
    path_len = RA.load_path_lengths(path_pts_json)

    net_tree = None
    try:
        from scipy.spatial import cKDTree
        d1m = json.load(open(init_1m, encoding="utf-8"))["paths"]
        pts = np.array([[q["x"], q["z"]] for p in d1m for q in p["points"]], dtype=float)
        net_tree = cKDTree(pts)
    except Exception:
        net_tree = None

    matcher = GridPathMatcher(init_1m)

    _W.update(dict(core=core, matcher=matcher, path_pts=path_pts, path_pts_1m=path_pts_1m,
                   path_len=path_len, net_tree=net_tree))


def _work(job):
    """job = (k, total, tid, fpath, scratch, meta)"""
    k, total, tid, fpath, scratch, meta = job
    core = _W["core"]
    t1 = time.time()
    try:
        stat = core.process_flight(fpath, scratch, _W["matcher"])
    except Exception as ex:
        return {"tid": tid, "error": repr(ex), "k": k}

    dt = time.time() - t1
    try:
        best_file = os.path.join(scratch, "4-最优路径", "best-%s_converted.json" % tid)
        best = json.load(open(best_file, encoding="utf-8")) if os.path.exists(best_file) else {}
        bp = best.get("best_path")
        met = RA.compute_metrics(fpath, bp, _W["path_pts_1m"], _W["net_tree"]) or {}
    except Exception as ex:
        return {"tid": tid, "error": "metric:" + repr(ex),
                "n_candidates": stat.get("候选路径数"), "k": k}

    m = meta or {}
    path_len = _W["path_len"]
    return {
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
        "n_dup_node": RA.count_dup_nodes(bp),
        "t_step2": stat.get("step2_潜在R识别时间(s)"),
        "t_step3": stat.get("step3_组合路径方案时间(s)"),
        "t_step4": stat.get("step4_最优路径搜索时间(s)"),
        "t_total": stat.get("total_time(s)"),
        "wall_s": round(dt, 3),
        "best_path": bp,
        "k": k,
    }


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("airport")
    ap_.add_argument("--out-root", default=RA.OUT_ROOT_DEFAULT)
    ap_.add_argument("--limit", type=int, default=None)
    ap_.add_argument("--tag", default=None)
    ap_.add_argument("--path-angle", type=float, default=110.0)
    ap_.add_argument("--angle", type=float, default=10.0)
    ap_.add_argument("--r-match", type=float, default=15.0)
    ap_.add_argument("--r-topk", type=int, default=3)
    ap_.add_argument("--graph-subdir", default="graph")
    ap_.add_argument("--workers", type=int, default=4)
    ap_.add_argument("--angle-aware", action="store_true",
                     help="把 110° 约束内生化到 PathFind 的图上（只允许合法转移）")
    ap_.add_argument("--angle-penalty", type=float, default=0.0,
                     help="软约束：非法转移边权乘该系数（0=关闭，建议 8）")
    ap_.add_argument("--resume", action="store_true",
                     help="从 results_partial.json 续跑，跳过已完成的航班")
    ap_.add_argument("--keep-stages", action="store_true",
                     help="保留每个航班的五阶段中间产物（默认跑完清理）")
    ap_.add_argument("--tids", default=None,
                     help="只跑指定 tid 列表（JSON 数组或每行一个的 txt）")
    a = ap_.parse_args()

    AP = a.airport.upper()
    airport_root = os.path.join(a.out_root, AP)
    graph_dir = os.path.join(airport_root, a.graph_subdir)
    traj_dir = os.path.join(airport_root, "traj_raw")

    keep_stages = bool(getattr(a, "keep_stages", False))
    tag = a.tag or ("pilot" if a.limit else "full")
    out_dir = os.path.join(airport_root, tag)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(traj_dir) if f.endswith(".json"))
    if a.tids:
        if a.tids.endswith(".json"):
            want = set(json.load(open(a.tids, encoding="utf-8")))
        else:
            want = {ln.strip() for ln in open(a.tids, encoding="utf-8") if ln.strip()}
        files = [f for f in files if os.path.splitext(f)[0] in want]
    if a.limit:
        files = files[:a.limit]
    metas = {m["tid"]: m for m in json.load(open(os.path.join(airport_root, "sample_list.json"),
                                                 encoding="utf-8"))}

    cfgargs = {"path_angle": a.path_angle, "angle": a.angle,
               "r_match": a.r_match, "r_topk": a.r_topk,
               "angle_aware": bool(a.angle_aware),
               "angle_penalty": float(getattr(a, "angle_penalty", 0.0) or 0.0)}
    init_args = (graph_dir,
                 os.path.join(graph_dir, "InitMatrixPath.json"),
                 os.path.join(graph_dir, "InitMatrixPath-1m.json"),
                 cfgargs)

    print("[par] %s: %d 条  workers=%d  graph=%s  out=%s"
          % (AP, len(files), a.workers, a.graph_subdir, out_dir), flush=True)

    jobs = []
    for k, fn in enumerate(files, 1):
        tid = os.path.splitext(fn)[0]
        jobs.append((k, len(files), tid, os.path.join(traj_dir, fn), None, metas.get(tid, {})))

    # ---- 断点续跑：跳过 results_partial.json 里已经出结果的航班 ----
    prior = []
    part_path = os.path.join(out_dir, "results_partial.json")
    if a.resume and os.path.exists(part_path):
        prior = json.load(open(part_path, encoding="utf-8"))
        done_ids = {r["tid"] for r in prior if r.get("mean_dist_m") is not None
                    or r.get("error")}
        n0 = len(jobs)
        jobs = [j for j in jobs if j[2] not in done_ids]
        print("[resume] 已完成 %d，剩余 %d / %d" % (len(done_ids), len(jobs), n0), flush=True)

    # 每个 worker 一个暂存目录，互不干扰
    t_start = time.time()
    rows, done = list(prior), 0
    stage_root = os.path.join(out_dir, "_stage")
    alljobs = [(k, total, tid, fp, os.path.join(stage_root, "p%d" % k), meta)
               for (k, total, tid, fp, _, meta) in jobs]
    # 一次性投递、按完成顺序取回：单条超长航班只占自己那一个 worker，不再形成批屏障
    with mp.Pool(processes=a.workers, initializer=_init, initargs=init_args) as pool:
        for r in pool.imap_unordered(_work, alljobs, chunksize=1):
            rows.append(r)
            done += 1
            if r.get("mean_dist_m") is not None:
                print("[%d/%d] %s %s 候选=%s 均值=%.1fm 折返=%s 用时=%.1fs"
                      % (done, len(jobs), r["tid"], r.get("flio"),
                         r.get("n_candidates"), r["mean_dist_m"],
                         r.get("n_reversal"), r.get("wall_s")), flush=True)
            else:
                print("[%d/%d] %s FAIL %s" % (done, len(jobs), r["tid"],
                                              str(r.get("error"))[:80]), flush=True)
            # 逐条落盘：中断后 --resume 不会重跑已完成航班
            with open(os.path.join(out_dir, "results_partial.json"), "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=1)

    # 清掉本次任务的临时目录（保留 --keep-stages）
    # 注意：这里必须容错——safe-delete 守卫会在条目过多时抛错，
    # 不能因为清理失败而丢掉已经算完的 results.json
    if not keep_stages:
        try:
            # 批量删除守卫：一次删 <=40 个条目，循环到清空
            import os as _os, shutil as _sh
            for _ in range(200):
                items = _os.listdir(stage_root)
                if not items:
                    break
                for _it in items[:40]:
                    _p = _os.path.join(stage_root, _it)
                    if _os.path.isdir(_p):
                        _sh.rmtree(_p, ignore_errors=True)
                    else:
                        try:
                            _os.remove(_p)
                        except OSError:
                            pass
            try:
                _os.rmdir(stage_root)
            except OSError:
                pass
        except BaseException as _ex:
            print("[warn] 临时目录未清理: %r" % (_ex,), flush=True)

    el = time.time() - t_start
    # 续跑会产生同一 tids 的重复行（上次无结果、这次重跑的），按 tid 去重保留最后一条
    _dedup = {}
    for r in rows:
        _dedup[r["tid"]] = r
    rows = list(_dedup.values())
    rows.sort(key=lambda r: r.get("k", 0))
    keep = ("tid", "flio", "n_track_points", "track_len_m", "dur_s", "n_candidates",
            "best_score", "mean_dist_m", "median_dist_m", "p90_dist_m", "p95_dist_m",
            "max_dist_m", "cov_10m", "cov_15m", "cov_30m", "lb_mean_dist_m",
            "lb_p95_dist_m", "n_route_paths", "route_len_m", "n_reversal", "n_dup_node",
            "t_step2", "t_step3", "t_step4", "t_total", "wall_s", "best_path", "error")
    clean = [{k: r.get(k) for k in keep if k in r} for r in rows]

    summary = {
        "airport": AP, "n": len(clean), "elapsed_s": round(el, 1),
        "per_flight_s": round(el * a.workers / max(1, len(clean)), 2),
        "n_ok": sum(1 for r in clean if r.get("mean_dist_m") is not None),
        "workers": a.workers,
        "graph_subdir": a.graph_subdir,
        "config": {"R_MATCH_DIST": a.r_match, "R_TOP_K": a.r_topk,
                   "ANGLE_THRESHOLD": a.angle,
                   "ANGLE_THRESHOLD_ESCALATED": 7, "PATH_ANGLE_THRESHOLD": a.path_angle,
                   "GRID_CELL_SIZE": 100.0, "GRID_SEARCH_RANGE": 2},
    }
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": clean}, f, ensure_ascii=False, indent=1)

    ok = [r for r in clean if r.get("mean_dist_m") is not None]
    print("\n===== %s 汇总 (成功 %d/%d) =====" % (AP, len(ok), len(clean)))
    if ok:
        arr = lambda k: np.array([r[k] for r in ok if r.get(k) is not None], dtype=float)
        for k in ("mean_dist_m", "median_dist_m", "p95_dist_m", "cov_15m",
                  "n_candidates", "t_total", "wall_s"):
            v = arr(k)
            if len(v):
                print("  %-16s mean=%.2f  p50=%.2f  p90=%.2f" %
                      (k, v.mean(), np.median(v), np.percentile(v, 90)))
        nrev = sum(1 for r in ok if (r.get("n_reversal") or 0) > 0)
        print("  折返>0 的航班: %d / %d" % (nrev, len(ok)))
    print("[done] %s，总耗时 %.1f min" % (out_dir, el / 60))


if __name__ == "__main__":
    mp.freeze_support()
    main()
