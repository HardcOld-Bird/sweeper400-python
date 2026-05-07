"""
# calib子包

子包路径：`sweeper400.calib`

包含校准相关的模块和类，用于各种情形下的测试和校准。
"""

# 将模块功能提升至包级别，可缩短外部import语句
from .caliber import (
    CaliberAnemone,
    CaliberFishNet,
    CaliberOctopus,
    CaliberSardine,
    PowerTester,
)

# 控制 import * 的行为
__all__ = [
    "CaliberSardine",
    "CaliberOctopus",
    "CaliberFishNet",
    "CaliberAnemone",
    "PowerTester",
]
