"""matplotlib 中文配置模块"""

import matplotlib.pyplot as plt


def setup_chinese_fonts():
    """配置matplotlib支持中文显示"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Microsoft JhengHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
