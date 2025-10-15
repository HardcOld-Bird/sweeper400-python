"""
# move子包

子包路径：`sweeper400.move`

包含**步进电机控制**相关的模块和类。
"""

# 将模块功能提升至包级别，可缩短外部import语句
from .controller import MotorController

# 控制 import * 的行为
__all__ = [
    "MotorController",
]
