"""
# use子包

子包路径：`sweeper400.use`

包含**协同调用其他子包功能**的模块和类，将功能封装为适用于特定任务的专用对象。
"""

# 将模块功能提升至包级别，可缩短外部import语句
from .caliber import CaliberFishNet, CaliberOctopus, CaliberSardine, PowerTester
from .sweeper_core import (
    Point2D,
    SweepData,
    SweeperCore,
    get_square_grid,
    load_sweep_data,
)

# 控制 import * 的行为
__all__ = [
    "CaliberSardine",
    "CaliberOctopus",
    "CaliberFishNet",
    "PowerTester",
    "Point2D",
    "SweepData",
    "load_sweep_data",
    "get_square_grid",
    "SweeperCore",
]
