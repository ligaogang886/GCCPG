"""
路径补全模块（清洗自原 path_find.py，即被组合方案模块实际调用的版本）。

功能：在西安机场滑行道路网图上，把由关键节点（起点、潜在R、终点）组成的
custom_path 逐段补全为完整路径序列。

建图：connection_matrix.csv 邻接矩阵定义节点间连通性，
      InitMatrixPath.json 的 pathLength 作为边权重（两端路径长度之和）。
搜索：段间用 networkx 的 A*（heuristic=None，等价于 Dijkstra）求最短路，
      拼接时去掉段与段之间的重复节点；
      最后校验 custom_path 中所有节点都出现在完整路径中，否则视为失败。
"""

import json
import pandas as pd
import networkx as nx

import config


# ==============================
# 初始化图结构（只执行一次）
# ==============================
def _initialize_graph():

    def load_adjacency_matrix(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        return df.values, df.index.tolist()

    def load_path_lengths(json_path):
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        return {
            path["pathname"]: path["pathLength"]
            for path in data.get("paths", [])
        }

    def build_graph(adjacency_matrix, nodes, path_lengths):

        G = nx.Graph()

        for i, row in enumerate(adjacency_matrix):

            for j, value in enumerate(row):

                if value == 1:

                    weight = (
                        path_lengths.get(nodes[i], 0) +
                        path_lengths.get(nodes[j], 0)
                    )

                    G.add_edge(
                        nodes[i],
                        nodes[j],
                        weight=weight
                    )

        return G

    adjacency_matrix, nodes = load_adjacency_matrix(config.CONNECTION_MATRIX)

    path_lengths = load_path_lengths(config.INIT_MATRIX_PATH)

    G = build_graph(adjacency_matrix, nodes, path_lengths)

    return G


# ==============================
# 全局图（只构建一次）
# ==============================
_GLOBAL_GRAPH = _initialize_graph()


# ==============================
# PathFind 主函数
# ==============================
def PathFind(custom_path):
    """根据 custom_path 查找完整路径"""

    G = _GLOBAL_GRAPH

    full_path = []

    for i in range(len(custom_path) - 1):

        start = custom_path[i]
        end = custom_path[i + 1]

        if start not in G or end not in G:
            print("节点不存在:", start, end)
            return None

        try:

            path = nx.astar_path(
                G,
                start,
                end,
                heuristic=None,
                weight="weight"
            )

            if i < len(custom_path) - 2:
                full_path.extend(path[:-1])
            else:
                full_path.extend(path)

        except nx.NetworkXNoPath:

            print(f"无法从 {start} 到 {end} 找到路径")

            return None

    # 校验 custom_path 是否都在 full_path 中
    custom_path_set = set(custom_path)

    if not custom_path_set.issubset(set(full_path)):

        print(
            "路径中缺失节点:",
            custom_path_set - set(full_path)
        )

        return None

    return full_path
