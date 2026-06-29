# pyright: basic
"""
# 绘图函数调试脚本

该脚本用于调试绘图函数。
"""

import numpy as np

from sweeper400.analyze import (
    Point2D,
    calculate_amplitude_integral,
    combine_point_tf_data_list,
    load_compressed_data,
    pick_area,
    plot_comprehensive_experiment,
    plot_point_tf_data_list,
    prepare_comprehensive_experiment_data,
    sweep_data_to_point_tf_data_list,
)

# 总结果文件夹根路径
root_folder: str = "D:\\EveryoneDownloaded\\exp0617\\"

# 读取样例SweepData，并转换为绘图数据
raw_data_1 = load_compressed_data(
    file_path = root_folder + "5_M+_L_1st_sweep\\sweep_data.pkl"
)
plot_data_1 = sweep_data_to_point_tf_data_list(
    sweep_data=raw_data_1,
    lowcut=1700.0,
    highcut=6800.0,
)
raw_data_2 = load_compressed_data(
    file_path = root_folder + "15_X_L_bg_sweep\\sweep_data.pkl"
)
plot_data_2 = sweep_data_to_point_tf_data_list(
    sweep_data=raw_data_2,
    lowcut=1700.0,
    highcut=6800.0,
)
plot_data_3 = combine_point_tf_data_list(
    list_a=plot_data_1,
    list_b=plot_data_2,
    mode="minus"
)
# 绘制差值声场图
_ = plot_point_tf_data_list(
    plot_tf_results=plot_data_3,
    mode="interpolated",
    save_path=root_folder + "5_M+_L_1st_sweep\\result.png",
)

# %% 验证积分函数正确性
picked_area = pick_area(
    data_list=plot_data_3,
    picked_center=Point2D(x=155.5, y=155.5),
    picked_area_radius=100,
    area_shape="square",
)
_ = calculate_amplitude_integral(
    data_list=picked_area,
    mode="fourier",
    k_modulus=2 * np.pi / 0.1,
    k_angle_deg=0,
)
_ = calculate_amplitude_integral(
    data_list=picked_area,
    mode="fourier",
    k_modulus=2 * np.pi / 0.1,
    k_angle_deg=180,
)
