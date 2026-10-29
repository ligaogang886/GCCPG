"""
组合路径方案模块（清洗自原 R_zuhe.py，已删除文件头部注释掉的旧版本）。

功能：以起点路径为种子，按潜在R的先后顺序逐个尝试接入，通过
「PathFind 补全 + 相邻路径 110 度夹角约束」做分支筛选，
生成一批候选完整路径；最后接入终点路径并再次过滤。

分支规则（generate_combinations_branching）：
  对当前每个组合 combo 尝试 candidate = combo + [r]：
    - candidate 补全后通过夹角约束 -> 同时保留 combo 与 candidate（可走可不走该R）
    - 不通过 -> 保留 combo，以及 combo[:-1] + [r]（替换末尾节点）
  每一步过去重。
"""

import os
import json

from graph_pathfinder import PathFind
from angle_constraints import check_three_paths_common_point


# =========================
# 读取 JSON
# =========================
def load_r_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# 保存完整路径
# =========================
def save_full_paths(full_paths, start, end, output_file):

    result = {
        "start": start,
        "end": end,
        "total": len(full_paths),
        "paths": full_paths
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"[组合方案] 完整路径已保存到: {output_file}")


# =========================
# Path 构造
# =========================
def build_path(combo):
    return PathFind(combo)


# =========================
# 组合生成逻辑（无 end 约束）
# =========================
def generate_combinations_branching(r_dict, start_path):

    combinations = [start_path]

    for step_idx, r in enumerate(r_dict.keys(), 1):

        new_combinations = []

        for idx, combo in enumerate(combinations):

            # =========================
            # 尝试加入 R
            # =========================
            candidate = combo + [r]
            candidate_path = build_path(candidate)
            check_result = check_three_paths_common_point(candidate_path)

            if check_result:
                # 可达 -> 保留 原路径 + 增量路径
                new_combinations.append(combo)
                new_combinations.append(candidate)
            else:
                # 不可达 -> 末尾分支
                branch1 = combo

                if len(combo) > 1:
                    branch2 = combo[:-1] + [r]
                else:
                    branch2 = combo + [r]

                new_combinations.append(branch1)
                new_combinations.append(branch2)

        # =========================
        # 去重
        # =========================
        unique = []
        seen = set()

        for c in new_combinations:
            t = tuple(c)
            if t not in seen:
                seen.add(t)
                unique.append(c)

        combinations = unique

    return combinations


# =========================
# 最终筛选 + PathFind 展开
# =========================
def filter_and_expand(combinations, end):

    final_paths = []

    for idx, combo in enumerate(combinations):

        full_combo = combo + [end] if end else combo

        try:
            pf = PathFind(full_combo)

            try:
                full_path = list(pf)
            except:
                full_path = eval(str(pf))

            check_result = check_three_paths_common_point(pf)

            if check_result:
                final_paths.append(full_path)

        except Exception as e:
            pass

    return final_paths


# =========================
# 主入口
# =========================
def generate_path_schemes(r_json_path,
                          start_path=["N-789"],
                          end=None,
                          output_dir=None):
    """
    生成候选完整路径方案。

    参数：
    - r_json_path: 潜在R识别结果 JSON（{R: count}）
    - start_path:  起点路径，list（如 ["N-4"]）
    - end:         终点路径（如 "N-788"）
    - output_dir:  输出目录，None 则不保存

    返回：组合方案文件路径（未保存时返回 None）
    """
    r_dict = load_r_json(r_json_path)

    # ① 组合生成
    combinations = generate_combinations_branching(
        r_dict,
        start_path
    )

    # ② 最终筛选
    full_paths = filter_and_expand(combinations, end)

    print(f"[组合方案] {os.path.basename(r_json_path)}: 最终可达路径数 {len(full_paths)}")

    # 保存
    if output_dir is None:
        return None

    os.makedirs(output_dir, exist_ok=True)

    file_name = os.path.basename(r_json_path)
    file_name = file_name.replace("-潜在R识别", "")
    file_name = os.path.splitext(file_name)[0]

    output_file = os.path.join(
        output_dir,
        f"fullpath-{file_name}.json"
    )

    save_full_paths(
        full_paths,
        start_path[0],
        end,
        output_file
    )

    return output_file
