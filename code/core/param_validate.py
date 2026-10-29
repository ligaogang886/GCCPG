"""
参数验证：给定 (ANGLE_THRESHOLD, R_MATCH_DIST)，对全部航班重跑 Step2-4，
与基线结果逐航班对比，输出精度/可行性对照表。

用法：
    python param_validate.py 5 15              # 新参数
    python param_validate.py 5 15 --out out.json

判定指标：
    - 平均偏差 = best_score / 轨迹点数（越低越好）
    - 折返点 = route[i] == route[i+2] 的个数（0 最好）
    - 重复节点 = 路线中重复出现的节点数（0 最好）
"""

import os
import sys
import json
import time
import argparse

import config


def evaluate_route(route, path_dict, track):
    from best_path_search import extract_path_points, compute_total_distance

    pts = extract_path_points(route, path_dict)
    score = compute_total_distance(track, pts)
    rev = sum(1 for i in range(len(route) - 2) if route[i] == route[i + 2])
    dup = len(route) - len(set(route))
    return score, score / max(len(track), 1), rev, dup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("angle", type=float)
    ap.add_argument("dist", type=float)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    angle, dist = args.angle, args.dist
    tag = f"a{angle:g}_d{dist:g}"

    config.ANGLE_THRESHOLD = angle
    config.R_MATCH_DIST = dist

    from potential_r_matcher import GridPathMatcher, extract_potential_r
    from path_combinator import generate_path_schemes
    from best_path_search import run_best_path_search, build_path_dict

    net = json.load(open(config.INIT_MATRIX_PATH, encoding="utf-8"))
    path_dict = build_path_dict(net)

    best_dir = os.path.join(config.OUTPUT_DIR, "4-最优路径")
    conv_dir = os.path.join(config.OUTPUT_DIR, "flight_data_converted")
    r_dir = os.path.join(config.OUTPUT_DIR, "1-潜在R识别")
    full_dir = os.path.join(config.OUTPUT_DIR, "2-组合方案")

    files = sorted(
        (f for f in os.listdir(best_dir) if f.startswith("best-") and f.endswith(".json")),
        key=lambda f: json.load(open(os.path.join(best_dir, f), encoding="utf-8"))["total_candidates"]
    )  # 先跑候选少的航班，便于快速拿到对比证据

    matcher = GridPathMatcher(config.INIT_MATRIX_PATH_1M)

    rows = []
    for f in files:
        base = f.replace("best-", "").replace(".json", "")   # xxx_converted
        fid = base.replace("_converted", "")
        conv = os.path.join(conv_dir, base + ".json")

        track = [(it["properties"]["unity_x"], it["properties"]["unity_z"])
                 for it in json.load(open(conv, encoding="utf-8"))
                 if it["properties"].get("unity_x") is not None]

        base_best = json.load(open(os.path.join(best_dir, f), encoding="utf-8"))
        base_score = base_best["best_score"]
        base_dev = base_score / len(track)
        base_rev = sum(1 for i in range(len(base_best["best_path"]) - 2)
                       if base_best["best_path"][i] == base_best["best_path"][i + 2])

        out_dir = os.path.join(config.OUTPUT_DIR, "_sweep", tag, fid)
        os.makedirs(out_dir, exist_ok=True)

        t0 = time.time()
        r_new = extract_potential_r(conv, matcher, angle_threshold=angle, mode="step2",
                                    output_dir=os.path.join(out_dir, "r"))
        new_set = set(json.load(open(r_new, encoding="utf-8")).keys())
        base_set = set(json.load(open(os.path.join(r_dir, f"{base}-潜在R识别.json"),
                                      encoding="utf-8")).keys())

        if new_set == base_set:
            new_score, new_dev = base_score, base_dev
            new_rev, new_dup = base_rev, len(base_best["best_path"]) - len(set(base_best["best_path"]))
            n_cand = base_best["total_candidates"]
            reused = True
        else:
            full_file = generate_path_schemes(
                r_new, [json.load(open(os.path.join(full_dir, f"fullpath-{base}.json"),
                                       encoding="utf-8"))["start"]],
                json.load(open(os.path.join(full_dir, f"fullpath-{base}.json"),
                               encoding="utf-8"))["end"],
                output_dir=os.path.join(out_dir, "schemes"))
            route, new_score, res = run_best_path_search(
                conv, config.INIT_MATRIX_PATH, full_file,
                output_dir=os.path.join(out_dir, "best"))
            new_dev = new_score / len(track)
            new_rev = sum(1 for i in range(len(route) - 2) if route[i] == route[i + 2])
            new_dup = len(route) - len(set(route))
            n_cand = len(res)
            reused = False

        dt = time.time() - t0
        rows.append({
            "flight": fid,
            "base_dev": round(base_dev, 1),
            "new_dev": round(new_dev, 1),
            "delta_dev": round(new_dev - base_dev, 1),
            "base_rev": base_rev,
            "new_rev": new_rev,
            "new_dup": new_dup,
            "base_cand": base_best["total_candidates"],
            "new_cand": n_cand,
            "reused": reused,
            "sec": round(dt, 0)
        })
        print(f"{fid:<12} 偏差 {base_dev:>5.1f} -> {new_dev:>5.1f} ({new_dev-base_dev:+5.1f})  "
              f"折返 {base_rev} -> {new_rev}  候选 {base_best['total_candidates']:>5} -> {n_cand:<5} "
              f"{'(复用)' if reused else ''}  {dt:.0f}s", flush=True)

    print("\n===== 汇总 =====")
    worse = [r for r in rows if r["delta_dev"] > 0.5]
    better = [r for r in rows if r["delta_dev"] < -0.5]
    print(f"变差航班: {len(worse)} {[r['flight'] for r in worse]}")
    print(f"变好航班: {len(better)} {[r['flight'] for r in better]}")
    print(f"平均偏差 基线 {sum(r['base_dev'] for r in rows)/len(rows):.2f} -> "
          f"新 {sum(r['new_dev'] for r in rows)/len(rows):.2f}")

    if args.out:
        json.dump(rows, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("saved:", args.out)


if __name__ == "__main__":
    main()
