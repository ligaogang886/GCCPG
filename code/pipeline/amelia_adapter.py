# -*- coding: utf-8 -*-
"""
Amelia / 美国机场适配层 —— 让 `算法核心代码` 零改动复用
=========================================================
核心算法里只有两处与机场强相关：

1. 坐标口径：potential_r_matcher.extract_potential_r 里把
   properties.originLat/originLon 送进 Xi'an 标定的仿射变换。
   Amelia 已提供机场局部笛卡尔坐标（km），应直接用 unity_x/unity_z。
   -> 本模块 extract_potential_r_xy() 与原件逐行对应，仅坐标来源不同；
      网格匹配类 GridPathMatcher、转角逐过滤 is_curve、统计保存 save_r_statistics
      全部直接复用原件，保证口径一致。

2. 起终点：原 start_end_extractor 依赖 InitStand(机位)/RunwayInf(跑道)，
   Amelia 的 exit/ramp 图层为空，无机位信息。
   -> 本模块 get_start_end_xy() 采用「数据驱动 + 跑道锚定」：
      起点 = 轨迹首点最近的 network path；终点 = 轨迹末点最近的 network path。
      跑道端由清洗阶段已按 d_runway <= RUNWAY_NEAR 判定进/离场并裁剪对齐。

其余模块（angle_constraints / graph_pathfinder / path_combinator /
best_path_search）不涉及机场数据，原样复用。
"""

import os
import json
import math
from collections import Counter

import config
from potential_r_matcher import GridPathMatcher, is_curve, save_r_statistics


# ============================================================
# 坐标读取
# ============================================================
def _xy(props):
    x = props.get("unity_x")
    z = props.get("unity_z")
    if x is None or z is None:
        return None
    try:
        return float(x), float(z)
    except (TypeError, ValueError):
        return None


# ============================================================
# Step2：潜在 R 识别（Amelia 版，与原件逐行对应）
# ============================================================
def extract_potential_r_xy(track_file,
                           matcher: GridPathMatcher,
                           angle_threshold=None,
                           mode="step2",
                           step1_output_dir=None,
                           output_dir=None):
    if angle_threshold is None:
        angle_threshold = config.ANGLE_THRESHOLD

    with open(track_file, "r", encoding="utf-8") as f:
        content = json.load(f)

    all_points = []
    all_matched_r = []
    r_counter = Counter()
    r_first_index = {}
    point_index = 0

    # ---- Step1 距离匹配 ----
    for point in content:
        props = point.get("properties", {})
        xy = _xy(props)
        if xy is None:
            all_points.append(None)
            all_matched_r.append([])
            continue
        x, z = xy
        top3 = matcher.find_top3_r_paths(x, z)
        all_points.append({"x": x, "z": z})
        all_matched_r.append(top3)
        for r in top3:
            r_counter[r] += 1
            if r not in r_first_index:
                r_first_index[r] = point_index
        point_index += 1

    if step1_output_dir is not None:
        os.makedirs(step1_output_dir, exist_ok=True)
        name = os.path.splitext(os.path.basename(track_file))[0]
        save_r_statistics(os.path.join(step1_output_dir, f"{name}-step1.json"),
                          r_counter, r_first_index)

    # ---- Step2 角度过滤 ----
    if mode == "step1":
        result_counter, result_index = r_counter, r_first_index
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
        result_counter, result_index = new_counter, r_first_index

    if output_dir is None:
        return None
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(track_file))[0]
    out = os.path.join(output_dir, f"{name}-潜在R识别.json")
    save_r_statistics(out, result_counter, result_index)
    return out


# ============================================================
# 起终点：数据驱动 + 跑道锚定
# ============================================================
def _load_path_points(init_matrix_path):
    with open(init_matrix_path, "r", encoding="utf-8") as f:
        data = json.load(f)["paths"]
    return {p["pathname"]: [(q["x"], q["z"]) for q in p["points"]] for p in data}


def nearest_path(x, z, path_pts):
    """返回 (pathname, 距离)"""
    best, bd = None, float("inf")
    for name, pts in path_pts.items():
        for px, pz in pts:
            d = (px - x) ** 2 + (pz - z) ** 2
            if d < bd:
                bd, best = d, name
    return best, math.sqrt(bd)


class PathLocator:
    """带网格加速的最近 path 查询（100 m 网格，与算法网格口径一致）

    注意：必须用**点到线段**的距离，不能只用顶点。Amelia 的路网顶点间距差异极大
    （junction 拆分后有的 N 段单段就 1.6 km），只比顶点会把起点锚到几百米外的
    另一条 path 上——表现为整条路线整体平移（实测 KDCA_1001104 偏 155 m，
    而轨迹离网仅 2.1 m）。
    """

    def __init__(self, init_matrix_path, cell=100.0):
        self.cell = cell
        self.grid = {}
        self.segs = []      # (ax, az, bx, bz, name)
        self.pts = _load_path_points(init_matrix_path)
        for name, pts in self.pts.items():
            for i in range(len(pts) - 1):
                ax, az = pts[i]
                bx, bz = pts[i + 1]
                si = len(self.segs)
                self.segs.append((ax, az, bx, bz, name))
                x0, x1 = (ax, bx) if ax <= bx else (bx, ax)
                z0, z1 = (az, bz) if az <= bz else (bz, az)
                for cx in range(int(math.floor(x0 / cell)), int(math.floor(x1 / cell)) + 1):
                    for cz in range(int(math.floor(z0 / cell)), int(math.floor(z1 / cell)) + 1):
                        self.grid.setdefault((cx, cz), []).append(si)

    @staticmethod
    def _seg_dist(x, z, ax, az, bx, bz):
        vx, vz = bx - ax, bz - az
        L2 = vx * vx + vz * vz
        if L2 <= 1e-12:
            return math.hypot(x - ax, z - az)
        t = ((x - ax) * vx + (z - az) * vz) / L2
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        return math.hypot(x - (ax + t * vx), z - (az + t * vz))

    def nearest(self, x, z, max_ring=8):
        cx = int(math.floor(x / self.cell))
        cz = int(math.floor(z / self.cell))
        best, bd = None, float("inf")
        for r in range(0, max_ring + 1):
            for dx in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if max(abs(dx), abs(dz)) != r:
                        continue
                    for si in self.grid.get((cx + dx, cz + dz), ()):
                        ax, az, bx, bz, name = self.segs[si]
                        d = self._seg_dist(x, z, ax, az, bx, bz)
                        if d < bd:
                            bd, best = d, name
            # 环 r 之外的格子最近可能距离 >= r*cell，可安全提前退出
            if best is not None and bd <= r * self.cell:
                break
        if best is None:
            return nearest_path(x, z, self.pts)
        return best, bd


def get_start_end_xy(track_file, locator: PathLocator):
    """返回 (start_pathname, end_pathname, 首点距离, 末点距离)

    注意：返回的是**单个 pathname 字符串**，与原 get_start_end_from_trajectory 一致；
    原 process_flight 会在调用处再包一层 list。
    """
    with open(track_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    pts = []
    for it in data:
        xy = _xy(it.get("properties", {}))
        if xy is not None:
            pts.append(xy)
    if len(pts) < 2:
        return None, None, None, None
    s, ds = locator.nearest(*pts[0])
    e, de = locator.nearest(*pts[-1])
    return s, e, ds, de
