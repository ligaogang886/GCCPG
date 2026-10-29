"""
坐标转换模块（合并自原 tounity.py / track_affine_transform.py）。

功能：将轨迹数据中的经纬度（originLat / originLon）通过仿射变换
转换为西安机场 Unity 坐标系（unity_x / unity_z）。
"""

import os
import json

import config


# ==============================
# 仿射变换
# ==============================
def affine_transform(lon, lat, a, b, c, d, tx, tz):
    unity_x = a * lon + b * lat + tx
    unity_z = c * lon + d * lat + tz
    return unity_x, unity_z


# ==============================
# 坐标转换（默认使用 config 中的机场标定参数）
# ==============================
def convert_coordinates(lat, lon):
    return affine_transform(
        lon, lat,
        config.AFFINE_A, config.AFFINE_B, config.AFFINE_C,
        config.AFFINE_D, config.AFFINE_TX, config.AFFINE_TZ
    )


# ==============================
# 单文件处理：轨迹 JSON 中写入 unity_x / unity_z
# ==============================
def transform_track_json(file_path, output_dir=None):
    """
    读取原始轨迹 JSON（list，每个元素含 properties.originLat/originLon），
    逐点做仿射变换，把 unity_x / unity_z 写回 properties，
    另存为 <文件名>_converted.json，返回新文件路径。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    count = 0

    for item in data:
        props = item.get("properties", {})

        lat = props.get("originLat")
        lon = props.get("originLon")

        if lat is None or lon is None:
            continue

        try:
            lat = float(lat)
            lon = float(lon)
        except:
            continue

        unity_x, unity_z = convert_coordinates(lat, lon)

        props["unity_x"] = unity_x
        props["unity_z"] = unity_z

        count += 1

    # 构造输出路径
    file_name = os.path.basename(file_path)
    file_name_no_ext = os.path.splitext(file_name)[0]

    # 如果没传 output_dir，就默认原目录
    if output_dir is None:
        output_dir = os.path.dirname(file_path)

    os.makedirs(output_dir, exist_ok=True)

    new_file_path = os.path.join(
        output_dir,
        file_name_no_ext + "_converted.json"
    )

    with open(new_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"[坐标转换] {file_name}: {count} 点 -> {new_file_path}")

    return new_file_path


# ==============================
# 批量处理：整个文件夹的轨迹 JSON
# ==============================
def process_flight_data(input_dir, output_dir, add_suffix=False):
    """
    批量处理航班轨迹数据（经纬度 -> Unity 坐标）。

    参数：
    - input_dir:  原始数据文件夹（如 raw_flight_data）
    - output_dir: 输出文件夹
    - add_suffix: 是否给输出文件添加 _converted 后缀
    """
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

    print(f"[坐标转换] 共发现 {len(files)} 个文件")

    total_count = 0

    for file_name in files:
        input_path = os.path.join(input_dir, file_name)

        if add_suffix:
            output_name = file_name.replace(".json", "_converted.json")
        else:
            output_name = file_name

        output_path = os.path.join(output_dir, output_name)

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        count = 0

        for item in data:
            props = item.get("properties", {})

            lat = props.get("originLat")
            lon = props.get("originLon")

            if lat is None or lon is None:
                continue

            try:
                lat = float(lat)
                lon = float(lon)
            except:
                continue

            unity_x, unity_z = convert_coordinates(lat, lon)

            props["unity_x"] = unity_x
            props["unity_z"] = unity_z

            count += 1

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"[坐标转换] {file_name} -> {count} 点")
        total_count += count

    print(f"[坐标转换] 全部完成，总转换点数: {total_count}")
