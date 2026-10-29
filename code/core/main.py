"""
西安机场滑行路线识别算法 —— 主入口。

 pipeline:
   Step0 起终点提取  start_end_extractor.get_start_end_from_trajectory
   Step1 坐标转换    coordinate_transform.transform_track_json
   Step2 潜在R识别   potential_r_matcher.extract_potential_r
   Step3 组合方案    path_combinator.generate_path_schemes
   Step4 最优路径    best_path_search.run_best_path_search

用法：
   python main.py                          # 批量处理默认输入目录的全部轨迹
   python main.py --input-dir <目录>       # 指定原始轨迹目录
   python main.py --single <文件.json>     # 只处理单个轨迹文件
   python main.py --output-dir <目录>      # 指定输出根目录（默认脚本下 output/）

输入：原始轨迹 JSON（list，元素含 properties.originLat/originLon/flio/rway）。
输出（output/ 下）：
   flight_data_converted/  转换坐标后的轨迹
   0-Step1原始R统计/       距离匹配的原始R统计（过程产物）
   1-潜在R识别/            角度过滤后的潜在R集合
   2-组合方案/             候选完整路径列表
   4-最优路径/             最优路线（含全部候选打分）
   runtime_statistics.json 各步骤耗时统计
"""

import os
import sys
import json
import time
import argparse

import config
from start_end_extractor import get_start_end_from_trajectory
from coordinate_transform import transform_track_json
from potential_r_matcher import GridPathMatcher, extract_potential_r
from path_combinator import generate_path_schemes
from best_path_search import run_best_path_search

# 默认输入目录：算法包内的轨迹样本（data 与代码同级）
DEFAULT_INPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "raw_flight_data"
)


# ==============================
# 折返检测
# ==============================
def count_reversal(route):
    """统计折返点：route[i] == route[i+2]，即同一路径连续进出（U 型折返）。"""
    if not route or len(route) < 3:
        return 0
    return sum(1 for i in range(len(route) - 2) if route[i] == route[i + 2])


# ==============================
# Step2-4 流水线（可切换转角阈值）
# ==============================
def run_route_pipeline(converted_file, file_name_no_ext, start_path, end_path,
                       matcher, output_root, angle_threshold):
    """
    执行 Step2 潜在R识别 -> Step3 组合方案 -> Step4 最优路径。
    结果写入 output_root 下固定子目录（同名文件覆盖）。
    返回 (best_path, best_score, 候选数, 各步骤耗时)
    """
    step1_dir = os.path.join(output_root, "0-Step1原始R统计")
    potential_r_dir = os.path.join(output_root, "1-潜在R识别")
    schemes_dir = os.path.join(output_root, "2-组合方案")
    best_dir = os.path.join(output_root, "4-最优路径")

    t0 = time.time()
    extract_potential_r(
        track_file=converted_file,
        matcher=matcher,
        angle_threshold=angle_threshold,
        mode="step2",
        step1_output_dir=step1_dir,
        output_dir=potential_r_dir
    )
    t1 = time.time()

    r_output = os.path.join(
        potential_r_dir,
        f"{file_name_no_ext}_converted-潜在R识别.json"
    )
    generate_path_schemes(
        r_json_path=r_output,
        start_path=start_path,
        end=end_path,
        output_dir=schemes_dir
    )
    t2 = time.time()

    filtered_file = os.path.join(
        schemes_dir,
        f"fullpath-{file_name_no_ext}_converted.json"
    )
    best_path, best_score, results = run_best_path_search(
        converted_file,
        config.INIT_MATRIX_PATH,
        filtered_file,
        output_dir=best_dir
    )
    t3 = time.time()

    return best_path, best_score, len(results), {
        "step2": t1 - t0,
        "step3": t2 - t1,
        "step4": t3 - t2
    }


# ==============================
# 单航班全流程
# ==============================
def process_flight(trajectory_file, output_root, matcher):
    """
    处理单个航班轨迹，返回耗时统计 dict。
    matcher: GridPathMatcher 实例（跨航班复用，避免重复加载路径图）。
    """
    file_name_no_ext = os.path.splitext(os.path.basename(trajectory_file))[0]

    converted_dir = os.path.join(output_root, "flight_data_converted")

    print(f"\n===== 正在处理: {os.path.basename(trajectory_file)} =====")

    # ==============================
    # 0. 获取起终点
    # ==============================
    start_path, end_path = get_start_end_from_trajectory(trajectory_file)

    start_path = [start_path]

    print(f"[起终点] start_path={start_path[0]}, end_path={end_path}")

    # ==============================
    # 1. 转为 Unity 坐标
    # ==============================
    converted_file = transform_track_json(
        trajectory_file,
        converted_dir
    )

    # ==============================
    # 2-4. 默认阈值流水线
    # ==============================
    print(f"[主流程] Step2-4（转角阈值 {config.ANGLE_THRESHOLD}°）")

    best_path, best_score, n_cand, times = run_route_pipeline(
        converted_file, file_name_no_ext, start_path, end_path,
        matcher, output_root, config.ANGLE_THRESHOLD
    )

    rev = count_reversal(best_path)

    # ==============================
    # 折返兜底：默认阈值漏检缓转弯时，最优路线会出现同一路径连续进出；
    # 此时用升级阈值仅在临时目录重跑该航班，结果更好才采用。
    # ==============================
    if (config.ESCALATE_ON_REVERSAL and rev > 0
            and config.ANGLE_THRESHOLD_ESCALATED < config.ANGLE_THRESHOLD):

        print(f"[主流程] 检测到折返 {rev} 处 -> 用 {config.ANGLE_THRESHOLD_ESCALATED}° 重跑（仅本航班）")

        esc_root = os.path.join(output_root, "_escalated", file_name_no_ext)

        best2, score2, cand2, times2 = run_route_pipeline(
            converted_file, file_name_no_ext, start_path, end_path,
            matcher, esc_root, config.ANGLE_THRESHOLD_ESCALATED
        )

        rev2 = count_reversal(best2)

        if rev2 < rev or (rev2 == rev and score2 < best_score):

            # 采用升级结果：把临时目录中的 4 个产物复制到标准位置
            import shutil
            pairs = [
                (os.path.join(esc_root, "0-Step1原始R统计", f"{file_name_no_ext}_converted-step1.json"),
                 os.path.join(output_root, "0-Step1原始R统计", f"{file_name_no_ext}_converted-step1.json")),
                (os.path.join(esc_root, "1-潜在R识别", f"{file_name_no_ext}_converted-潜在R识别.json"),
                 os.path.join(output_root, "1-潜在R识别", f"{file_name_no_ext}_converted-潜在R识别.json")),
                (os.path.join(esc_root, "2-组合方案", f"fullpath-{file_name_no_ext}_converted.json"),
                 os.path.join(output_root, "2-组合方案", f"fullpath-{file_name_no_ext}_converted.json")),
                (os.path.join(esc_root, "4-最优路径", f"best-{file_name_no_ext}_converted.json"),
                 os.path.join(output_root, "4-最优路径", f"best-{file_name_no_ext}_converted.json")),
            ]
            for src, dst in pairs:
                if os.path.exists(src):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)

            best_path, best_score, n_cand = best2, score2, cand2
            times = {k: times[k] + times2[k] for k in times}
            print(f"[主流程] 已采用升级结果：折返 {rev} -> {rev2}，得分 {best_score:.1f}")
        else:
            print(f"[主流程] 升级后无改善（折返 {rev2}），保留默认结果")

    total_time = times["step2"] + times["step3"] + times["step4"]

    print(f"[主流程] 完成: {os.path.basename(trajectory_file)}，"
          f"折返 {count_reversal(best_path)} 处，总耗时: {total_time:.6f} 秒")

    return {
        "flight": file_name_no_ext,
        "step2_潜在R识别时间(s)": round(times["step2"], 6),
        "step3_组合路径方案时间(s)": round(times["step3"], 6),
        "step4_最优路径搜索时间(s)": round(times["step4"], 6),
        "total_time(s)": round(total_time, 6),
        "候选路径数": n_cand,
        "最优得分": round(best_score, 3) if best_score is not None else None,
        "折返数": count_reversal(best_path)
    }


# ==============================
# 主程序
# ==============================
def main():

    parser = argparse.ArgumentParser(description="西安机场滑行路线识别算法")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR,
                        help="原始轨迹 JSON 目录（默认: ../raw_flight_data）")
    parser.add_argument("--single", default=None,
                        help="只处理单个轨迹 JSON 文件")
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR,
                        help="输出根目录（默认: 脚本目录下 output/）")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_root = os.path.abspath(args.output_dir)
    os.makedirs(output_root, exist_ok=True)

    # 待处理文件列表
    if args.single:
        track_files = [os.path.abspath(args.single)]
    else:
        track_files = sorted(
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.endswith(".json")
        )

    if not track_files:
        print(f"未找到轨迹文件: {input_dir}")
        sys.exit(1)

    print(f"共 {len(track_files)} 个轨迹文件待处理")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_root}")

    # 网格匹配器只初始化一次（内部缓存整个路径图）
    matcher = GridPathMatcher(config.INIT_MATRIX_PATH_1M)

    time_statistics = []

    for track_file in track_files:

        try:
            stat = process_flight(track_file, output_root, matcher)
            time_statistics.append(stat)
        except Exception as e:
            print(f"处理失败: {os.path.basename(track_file)}")
            print(e)

    # 保存总时间统计
    output_time_json = os.path.join(output_root, "runtime_statistics.json")

    with open(output_time_json, "w", encoding="utf-8") as f:
        json.dump(time_statistics, f, ensure_ascii=False, indent=4)

    print("\n全部轨迹处理完成！")
    print(f"时间统计已保存: {output_time_json}")


if __name__ == "__main__":
    main()
