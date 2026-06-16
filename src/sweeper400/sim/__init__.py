"""仿真模块

此模块提供基于 COMSOL Multiphysics 的声学仿真自动化接口，
封装四模型参数扫描仿真与增益系数计算功能，
以及单点深度验证功能。
"""

from .simulator import ScanResult, SimScanner

__all__ = ["SimScanner", "ScanResult"]
