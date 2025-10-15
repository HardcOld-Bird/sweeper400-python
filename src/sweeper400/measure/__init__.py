"""
# measure子包

子包路径：`sweeper400.measure`

包含**NI数据采集**相关的模块和类。
"""

# 将模块功能提升至包级别，可缩短外部import语句
from .cont_sync_io import HiPerfCSSIO

# 控制 import * 的行为
__all__ = [
    "HiPerfCSSIO",
]
