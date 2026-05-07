"""
# use子包

子包路径：`sweeper400.use`

包含**协同调用其他子包功能**的模块和类，将功能封装为适用于特定任务的专用对象。
"""

# 将模块功能提升至包级别，可缩短外部import语句
from .evolver import Evolver, load_evolved_waveform
from .feedback_funcs import static_uniform_feedback, static_diff_feedback
from .sweeper import SweeperCore, get_square_grid

# 控制 import * 的行为
__all__ = [
    "Evolver",
    "load_evolved_waveform",
    "get_square_grid",
    "SweeperCore",
    "static_uniform_feedback",
    "static_diff_feedback",
]
