"""
路径夹角约束模块（合并自原 Anglepanduan.py + GraphConstraints.py）。

功能：
  1. compute_path_angle_by_name：根据路径名计算两条路径在其公共端点处的夹角；
     路径几何来自 InitMatrixPath.json（带全局缓存，只加载一次）。
  2. check_three_paths_common_point：路径合法性约束——序列中任意相邻两条
     路径的夹角必须 >= threshold（默认 110 度），否则判定不可达 / 不合理。

原 GraphConstraints.py 中基于 InitPath.json 构建的 path_dict 与 is_same_coord
为死代码（未被调用），已移除；原 Anglepanduan.py 的默认数据文件
"InitMatrixPath.json"（运行时相对 cwd）统一改为 config.INIT_MATRIX_PATH
（与原运行时所读文件 md5 一致，行为不变）。
"""

import json
import numpy as np

import config


# ==============================
# 全局缓存（关键）
# ==============================
_PATH_DICT = None


# ==============================
# 读取路径数据（只加载一次）
# ==============================
def load_paths_once(json_file=None):
    global _PATH_DICT

    if json_file is None:
        json_file = config.INIT_MATRIX_PATH

    if _PATH_DICT is None:

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        _PATH_DICT = {
            item["pathname"]: item
            for item in data["paths"]
        }

    return _PATH_DICT


# ==============================
# 向量夹角计算
# ==============================
def angle_between(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)

    if norm == 0:
        return 0

    cos_angle = np.clip(dot / norm, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


# ==============================
# 核心：计算两条路径夹角
# ==============================
def compute_path_angle(path1, path2, threshold_dist=1.0):

    if len(path1["points"]) < 2 or len(path2["points"]) < 2:
        return None

    p1_start = np.array([path1["points"][0]["x"], path1["points"][0]["y"], path1["points"][0]["z"]])
    p1_end = np.array([path1["points"][-1]["x"], path1["points"][-1]["y"], path1["points"][-1]["z"]])

    p2_start = np.array([path2["points"][0]["x"], path2["points"][0]["y"], path2["points"][0]["z"]])
    p2_end = np.array([path2["points"][-1]["x"], path2["points"][-1]["y"], path2["points"][-1]["z"]])

    candidate_pairs = [
        ("start-start", p1_start, p2_start),
        ("start-end", p1_start, p2_end),
        ("end-start", p1_end, p2_start),
        ("end-end", p1_end, p2_end)
    ]

    for tag, p1_common, p2_common in candidate_pairs:
        if np.linalg.norm(p1_common - p2_common) < threshold_dist:

            if "start" in tag.split("-")[0]:
                p1_adj = np.array([path1["points"][1]["x"], path1["points"][1]["y"], path1["points"][1]["z"]])
            else:
                p1_adj = np.array([path1["points"][-2]["x"], path1["points"][-2]["y"], path1["points"][-2]["z"]])

            if "start" in tag.split("-")[1]:
                p2_adj = np.array([path2["points"][1]["x"], path2["points"][1]["y"], path2["points"][1]["z"]])
            else:
                p2_adj = np.array([path2["points"][-2]["x"], path2["points"][-2]["y"], path2["points"][-2]["z"]])

            v1 = p1_adj - p1_common
            v2 = p2_adj - p2_common

            return angle_between(v1, v2)

    return None


# ==============================
# 对外接口（自动加载缓存）
# ==============================
def compute_path_angle_by_name(name1, name2, json_file=None):
    path_dict = load_paths_once(json_file)

    path1 = path_dict.get(name1)
    path2 = path_dict.get(name2)

    if path1 is None or path2 is None:
        print(f"[夹角] 路径不存在: {name1} 或 {name2}")
        return None

    angle = compute_path_angle(path1, path2)

    if angle is None:
        print(f"[夹角] {name1} 和 {name2} 没有公共端点")

    return angle


# ==============================
# 路径序列夹角约束检查
# ==============================
def check_three_paths_common_point(path_list, threshold=None):
    """
    基于相邻路径夹角约束判断路径合法性。

    参数：
        path_list: 路径序列，如 ["R-100", "R-99", "R-98"]
        threshold: 角度阈值，默认取 config.PATH_ANGLE_THRESHOLD（110 度）

    返回：
        True  -> 所有相邻路径夹角 >= threshold
        False -> 存在夹角 < threshold（或无法计算）
    """
    if threshold is None:
        threshold = config.PATH_ANGLE_THRESHOLD

    # 至少两个路径才有角度
    if len(path_list) < 2:
        return True

    for i in range(len(path_list) - 1):
        p1 = path_list[i]
        p2 = path_list[i + 1]

        angle = compute_path_angle_by_name(p1, p2)

        # 防御：避免 None 或异常值
        if angle is None:
            return False

        if angle < threshold:
            return False

    return True
