"""
潜在 R 路径识别模块（清洗自原 R_canditate.py）。

功能：将转换坐标后的轨迹与西安机场路径图做网格距离匹配，
再通过转角过滤，提取轨迹经过的潜在 R（转弯）路径集合。

流程：
  Step1 距离匹配：每个轨迹点在其所在网格（及周围 5x5 网格）中寻找
       距离 <= R_MATCH_DIST 且 pathname 以 "R" 开头的路径，取最近 3 个；
       统计每个 R 的命中次数与首次出现位置。
  Step2 角度过滤：对每个轨迹点取前后共 5 点窗口，计算 3 个相邻转角，
       若最大转角超过 angle_threshold（判为转弯段），保留该点匹配到的 R。

输出：{R路径名: 命中次数}，按首次出现顺序排列的 JSON 集合。
"""

import os
import json
import math
from collections import defaultdict, Counter

import config
from coordinate_transform import convert_coordinates


# ==============================
# 保存函数
# ==============================
def save_r_statistics(output_path, r_counter, r_first_index):

    sorted_r = sorted(
        r_counter.items(),
        key=lambda x: r_first_index[x[0]]
    )

    result_dict = {r: count for r, count in sorted_r}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)

    print(f"[潜在R] R统计结果已保存到: {output_path}")


# ==============================
# 网格匹配类
# ==============================
class GridPathMatcher:
    """
    将路径图的点按 cell_size 网格建索引，支持快速查询
    某坐标附近距离阈值内的 R 路径。
    """

    def __init__(self, init_path_file: str, cell_size: float = None):
        if cell_size is None:
            cell_size = config.GRID_CELL_SIZE
        self.cell_size = cell_size
        self.grid = defaultdict(list)
        self._load_paths(init_path_file)

    def _load_paths(self, file):

        print(f"[潜在R] 加载路径图: {file}")

        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for path in data.get("paths", []):
            for pt in path.get("points", []):
                x = pt["x"]
                z = pt["z"]

                gx = int(math.floor(x / self.cell_size))
                gz = int(math.floor(z / self.cell_size))

                key = f"{gx}_{gz}"

                self.grid[key].append({
                    "pathname": path.get("pathname"),
                    "x": x,
                    "z": z
                })

        print("[潜在R] 路径点加载完成")

    # ==============================
    # 查询坐标附近最近的 K 个 R 路径
    # ==============================
    def find_top3_r_paths(self, x, z, dist_threshold=None):

        if dist_threshold is None:
            dist_threshold = config.R_MATCH_DIST

        qx = int(math.floor(x / self.cell_size))
        qz = int(math.floor(z / self.cell_size))

        candidates = []

        # 扩大搜索范围
        search_range = config.GRID_SEARCH_RANGE
        for dx in range(-search_range, search_range + 1):
            for dz in range(-search_range, search_range + 1):
                key = f"{qx + dx}_{qz + dz}"
                if key not in self.grid:
                    continue
                for pt in self.grid[key]:
                    dist = math.hypot(x - pt["x"], z - pt["z"])
                    # 先过滤 R，再加入候选
                    if dist <= dist_threshold and pt["pathname"].startswith("R"):
                        candidates.append((dist, pt["pathname"]))

        # 按距离排序
        candidates.sort(key=lambda x: x[0])

        # 取最近的 K 个
        top_k = [p for _, p in candidates[:config.R_TOP_K]]

        # 去重
        result = []
        seen = set()
        for p in top_k:
            if p not in seen:
                seen.add(p)
                result.append(p)

        return result


# ==============================
# 角度计算
# ==============================
def compute_angle(p1, p2, p3):
    v1 = (p2["x"] - p1["x"], p2["z"] - p1["z"])
    v2 = (p3["x"] - p2["x"], p3["z"] - p2["z"])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 == 0 or mag2 == 0:
        return 0
    cos_theta = max(-1, min(1, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_theta))


def is_curve(points5, threshold=10):
    angles = [
        compute_angle(points5[0], points5[1], points5[2]),
        compute_angle(points5[1], points5[2], points5[3]),
        compute_angle(points5[2], points5[3], points5[4])
    ]
    return max(angles) > threshold


# ==============================
# 核心接口（支持模式切换）
# ==============================
def extract_potential_r(track_file,
                        matcher: GridPathMatcher,
                        angle_threshold=None,
                        mode="step2",
                        step1_output_dir=None,
                        output_dir=None):
    """
    提取潜在 R 路径集合。

    参数：
    - track_file:       转换坐标后的轨迹 JSON（含 properties.unity_x/unity_z 或 originLat/originLon）
    - matcher:          GridPathMatcher 实例
    - angle_threshold:  转角阈值，默认取 config.ANGLE_THRESHOLD
    - mode:             "step1" 只做距离匹配；"step2" 追加角度过滤
    - step1_output_dir: Step1 原始R统计输出目录，None 则不保存
    - output_dir:       潜在R识别结果输出目录，None 则不保存

    返回：潜在R结果文件路径（未保存时返回 None）
    """
    if angle_threshold is None:
        angle_threshold = config.ANGLE_THRESHOLD

    print(f"[潜在R] 处理轨迹: {track_file}")
    print(f"[潜在R] 模式: {mode}")

    with open(track_file, "r", encoding="utf-8") as f:
        content = json.load(f)

    all_points = []
    all_matched_r = []

    r_counter = Counter()
    r_first_index = {}

    point_index = 0

    # ==============================
    # Step 1：距离匹配
    # ==============================
    for point in content:
        props = point.get("properties", {})
        lon = props.get("originLon") or props.get("lon")
        lat = props.get("originLat") or props.get("lat")
        if lon is None or lat is None:
            all_points.append(None)
            all_matched_r.append([])
            continue
        try:
            lon, lat = float(lon), float(lat)
        except:
            all_points.append(None)
            all_matched_r.append([])
            continue

        x, z = convert_coordinates(lat, lon)

        top3 = matcher.find_top3_r_paths(x, z)

        all_points.append({"x": x, "z": z})
        all_matched_r.append(top3)

        for r in top3:
            r_counter[r] += 1
            if r not in r_first_index:
                r_first_index[r] = point_index

        point_index += 1

    # ==============================
    # 保存 Step1 过程结果
    # ==============================
    if step1_output_dir is not None:
        os.makedirs(step1_output_dir, exist_ok=True)

        file_name = os.path.splitext(os.path.basename(track_file))[0]

        step1_output_file = os.path.join(
            step1_output_dir,
            f"{file_name}-step1.json"
        )

        save_r_statistics(
            step1_output_file,
            r_counter,
            r_first_index
        )

    # ==============================
    # Step2：角度过滤
    # ==============================
    if mode == "step1":
        result_counter = r_counter
        result_index = r_first_index
    else:
        new_counter = Counter()
        new_first_index = {}
        for i in range(2, len(all_points) - 2):
            window = all_points[i - 2:i + 3]
            if any(p is None for p in window):
                continue
            if not is_curve(window, threshold=angle_threshold):
                continue
            for r in all_matched_r[i]:
                new_counter[r] += 1
                if r not in new_first_index:
                    new_first_index[r] = i

        result_counter = new_counter
        result_index = r_first_index

    # ==============================
    # 保存
    # ==============================
    if output_dir is None:
        return None

    os.makedirs(output_dir, exist_ok=True)

    file_name = os.path.splitext(os.path.basename(track_file))[0]
    output_file = os.path.join(
        output_dir,
        f"{file_name}-潜在R识别.json"
    )

    save_r_statistics(output_file, result_counter, r_first_index)

    return output_file
