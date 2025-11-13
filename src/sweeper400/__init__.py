# pyright: standard
"""
# “sweeper400”
协同控制NI数据采集卡和步进电机，以进行自动化**扫场测量**的`python`包。

## 子包：
- **measure**：NI数据采集相关功能
- **move**：步进电机控制相关功能
- **analyze**：信号和数据处理相关功能
- **sweeper**：协同调用其他子包的专用对象

## 使用示例：
    ```python
    from sweeper400.logger import get_logger
    logger = get_logger(__name__)
    logger.info("开始使用 sweeper400")
    ```
"""

# 导出日志管理系统的主要接口
from .logger import (
    get_logger,
)

# 将模块功能提升至包级别，可缩短外部import语句
from .use import Sweeper

__version__ = "0.1.0"
__author__ = "402"

# 包级别的日志器
_package_logger = get_logger(__name__)
_package_logger.info("sweeper400 包已加载")

# 控制 import * 的行为
__all__ = [
    "get_logger",
    "Sweeper",
]
