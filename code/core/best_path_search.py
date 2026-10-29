"""
最优路径搜索模块（清洗自原 fangantiqu.py）。

功能：对 Step3 生成的全部候选完整路径，用轨迹相似度打分——
将轨迹每个点候选路径点集（InitMatrixPath.json）的最近欧氏距离累加，
总距离最小者即为最优路线。
"""

import os
import json
import math

import config


# =========================
# 读取 JSON
# =========================
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# 保存 JSON
# =========================
def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# =========================
# 构建路径字典
# =========================
def build_path_dict(init_data):
    return {p["pathname"]: p for p in init_data["paths"]}


# =========================
# 欧氏距离
# =========================
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# =========================
# 提取路径点
# =========================
def extract_path_points(path_names, path_dict):
    all_points = []

    for name in path_names:
        path_obj = path_dict.get(name)
        if not path_obj:
            continue

        for pt in path_obj["points"]:
            all_points.append((pt["x"], pt["z"]))

    return all_points


# =========================
# 轨迹 -> 路径距离
# =========================
def compute_total_distance(track_points, path_points):
    total_dist = 0

    for tx, tz in track_points:
        min_dist = float("inf")

        for px, pz in path_points:
            d = distance((tx, tz), (px, pz))
            if d < min_dist:
                min_dist = d

        total_dist += min_dist

    return total_dist


# =========================
# 核心封装函数
# =========================
def run_best_path_search(track_file, init_file, filtered_file, output_dir=None):
    """
    最优路径搜索。

    参数：
    - track_file:    转换坐标后的轨迹 JSON（含 properties.unity_x/unity_z）
    - init_file:     路径图 JSON（提供路径点坐标）
    - filtered_file: Step3 组合方案 JSON（候选路径列表）
    - output_dir:    输出目录，None 则不保存

    返回：(best_path, best_score, results)
    """
    print("[最优路径] 开始最优路径搜索")

    # 读取数据
    track_data = load_json(track_file)
    init_data = load_json(init_file)
    filtered_data = load_json(filtered_file)

    path_dict = build_path_dict(init_data)

    # 提取轨迹点
    track_points = []

    for item in track_data:
        props = item.get("properties", {})

        x = props.get("unity_x")
        z = props.get("unity_z")

        if x is not None and z is not None:
            track_points.append((x, z))

    print(f"[最优路径] 轨迹点数量: {len(track_points)}")

    # 遍历所有候选路径
    best_path = None
    best_score = float("inf")

    results = []

    for idx, path_names in enumerate(filtered_data["paths"], 1):

        path_points = extract_path_points(path_names, path_dict)

        if not path_points:
            continue

        total_dist = compute_total_distance(track_points, path_points)

        results.append({
            "path": path_names,
            "score": total_dist
        })

        if total_dist < best_score:
            best_score = total_dist
            best_path = path_names

    # =========================
    # 保存结果
    # =========================
    output_file = None
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        file_name = os.path.splitext(os.path.basename(track_file))[0]

        output_file = os.path.join(
            output_dir,
            f"best-{file_name}.json"
        )

        output_data = {
            "best_path": best_path,
            "best_score": best_score,
            "total_candidates": len(results),
            "all_results": results
        }

        save_json(output_data, output_file)

    # =========================
    # 打印结果
    # =========================
    print("[最优路径] 最优路径:", best_path)
    print(f"[最优路径] 最小距离: {best_score}")
    print(f"[最优路径] 候选路径数: {len(results)}")
    if output_file:
        print(f"[最优路径] 已保存: {output_file}")

    return best_path, best_score, results
