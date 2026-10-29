"""
起终点路径提取模块（清洗自原 get_start_end_path.py）。

功能：从原始轨迹（经纬度）推断航班进/离场类型，提取起点路径与终点路径：
  - 进场（flio == "A"）：起点 = 跑道 Arrival 路径，终点 = 机位 standPath
  - 离场（flio == "D"）：起点 = 机位 pushPath，     终点 = 跑道 Departure 路径

机位通过轨迹首/尾若干点与 InitStand.json 的最近距离匹配确定。
"""

import json
import math

import config
from coordinate_transform import convert_coordinates


# ==============================
# 距离
# ==============================
def distance(x1, z1, x2, z2):
    return math.sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2)


# ==============================
# 主方法
# ==============================
def get_start_end_from_trajectory(
    trajectory_file,
    stand_file=None,
    runway_file=None,
    num_points=None
):
    """
    从轨迹文件提取 (start_path, end_path)。

    参数：
    - trajectory_file: 原始轨迹 JSON（list，元素含 properties）
    - stand_file:      机位信息 JSON，默认取 config.INIT_STAND
    - runway_file:     跑道信息 JSON，默认取 config.RUNWAY_INF
    - num_points:      取轨迹首/尾的点数，默认取 config.START_END_POINTS
    """
    if stand_file is None:
        stand_file = config.INIT_STAND
    if runway_file is None:
        runway_file = config.RUNWAY_INF
    if num_points is None:
        num_points = config.START_END_POINTS

    # ==============================
    # 1. 读取轨迹
    # ==============================
    with open(trajectory_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        return None, None

    props0 = data[0]["properties"]
    flio = props0.get("flio")
    rway = props0.get("rway")

    # ==============================
    # 2. 选取轨迹点（离场取头部，进场取尾部）
    # ==============================
    if flio == "D":
        selected = data[:min(num_points, len(data))]
    elif flio == "A":
        selected = data[-min(num_points, len(data)):]
    else:
        selected = data[:min(num_points, len(data))]

    # 转换为 Unity 坐标
    points = []
    for item in selected:
        p = item["properties"]
        lat = p.get("lat")
        lon = p.get("lon")

        if lat is None or lon is None:
            continue

        x, z = convert_coordinates(lat, lon)
        points.append((x, z))

    if not points:
        return None, None

    # ==============================
    # 3. 读取机位，匹配最近机位
    # ==============================
    with open(stand_file, "r", encoding="utf-8") as f:
        stands = json.load(f)["InitStand"]

    px, pz = points[0]

    min_dist = float("inf")
    best_stand = None

    for s in stands:
        d = distance(px, pz, s["x"], s["z"])
        if d < min_dist:
            min_dist = d
            best_stand = s

    # ==============================
    # 4. 机位路径（进场取 standPath，离场取 pushPath）
    # ==============================
    s_start, s_end = None, None

    if best_stand:
        if flio == "A":
            s_end = best_stand["standPath"]
        elif flio == "D":
            s_start = best_stand["pushPath"]

    # ==============================
    # 5. 跑道路径（进场取 Arrival，离场取 Departure）
    # ==============================
    with open(runway_file, "r", encoding="utf-8") as f:
        runway_data = json.load(f)["RunwayInf"][0]

    r_start, r_end = None, None

    if rway in runway_data:
        info = runway_data[rway]

        if flio == "A":
            r_start = info["Arrival"]
        elif flio == "D":
            r_end = info["Departure"]

    # ==============================
    # 6. 合并结果
    # ==============================
    start_path = r_start if r_start is not None else s_start
    end_path = r_end if r_end is not None else s_end

    return start_path, end_path
