"""
全局配置：路径、坐标变换参数、算法阈值。

所有路径均基于本文件所在目录解析，不依赖运行时工作目录（cwd），
整个「算法核心代码」文件夹拷走即可复现。
"""

import os

# ==============================
# 基础目录
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 输入数据目录（西安机场滑行道网 + 机位 + 跑道 + 邻接矩阵）
DATA_DIR = os.path.join(BASE_DIR, "data")

# 各数据文件
INIT_MATRIX_PATH_1M = os.path.join(DATA_DIR, "InitMatrixPath-1m.json")  # 1m 粒度路径图（潜在R网格匹配用）
INIT_MATRIX_PATH = os.path.join(DATA_DIR, "InitMatrixPath.json")        # 路径图（路径点 / 边长 / 建图用）
INIT_PATH = os.path.join(DATA_DIR, "InitPath.json")                    # 路径图（夹角计算用）
INIT_STAND = os.path.join(DATA_DIR, "InitStand.json")                  # 机位信息
RUNWAY_INF = os.path.join(DATA_DIR, "RunwayInf.json")                  # 跑道信息
CONNECTION_MATRIX = os.path.join(DATA_DIR, "connection_matrix.csv")    # 路径邻接矩阵

# ==============================
# 默认输出目录（main.py 使用，可被命令行参数覆盖）
# ==============================
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ==============================
# 坐标变换参数（经纬度 -> Unity 坐标的仿射变换系数）
# x = a * lon + b * lat + tx
# z = c * lon + d * lat + tz
# ==============================
AFFINE_A = 69219.734239
AFFINE_B = 73016.824547
AFFINE_C = -60472.614251
AFFINE_D = 83475.891532
AFFINE_TX = -10045470.936244
AFFINE_TZ = 3701948.837999

# ==============================
# 算法阈值
# ==============================
GRID_CELL_SIZE = 100.0        # 网格匹配的网格边长（米）
GRID_SEARCH_RANGE = 2         # 网格匹配时向四周扩展搜索的网格数
R_MATCH_DIST = 15.0           # 轨迹点与 R 路径的距离匹配阈值（米）
R_TOP_K = 3                   # 每个轨迹点保留的最近 R 路径个数
ANGLE_THRESHOLD = 10          # 潜在R识别的转角阈值（度）：5 点窗口内最大转角超过该值视为转弯段
# 兜底升级阈值：默认 10° 时，长而缓的出口转弯（如 CES2272_A 跑道出口）可能漏检，
# 导致最优路线出现"折返"（同一路径连续进出）。检测到折返时，用该阈值重跑该航班。
# 实测：10 -> 7 可补回 R-910 等出口转弯，候选数增至约 3 倍；
#       仅对出现折返的航班触发，其余航班结果不变。
ANGLE_THRESHOLD_ESCALATED = 7
ESCALATE_ON_REVERSAL = True   # 是否启用"折返检测 -> 自动升级阈值重跑"
PATH_ANGLE_THRESHOLD = 110    # 路径合法性约束：相邻路径夹角下限（度）
START_END_POINTS = 20         # 提取起终点时使用的轨迹首/尾点数
