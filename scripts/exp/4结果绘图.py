# pyright: basic
"""
# 实验结果绘图脚本

该脚本用于绘制和分析实验结果。
"""

import numpy as np

from sweeper400.analyze import (
    Point2D,
    plot_comprehensive_experiment,
    prepare_comprehensive_experiment_data,
)

# 总结果文件夹根路径
root_folder: str = "D:\\EveryoneDownloaded\\exp0617\\"
root_folder_2: str = "D:\\EveryoneDownloaded\\exp0617（弱响应）\\"

# %% 数据准备
_ = prepare_comprehensive_experiment_data(
    left_r_0_static_folder = root_folder + "\\12_M-_R_1st_sweep",
    left_r_0_feedback_folder = root_folder + "\\11_M-_R_8fb_sweep",
    left_r_minus1_static_folder = root_folder + "\\5_M+_L_1st_sweep",
    left_r_minus1_feedback_folder = root_folder + "\\4_M+_L_8fb_sweep",
    right_r_plus1_static_folder = root_folder + "\\14_M-_l_1st_sweep",
    right_r_plus1_feedback_folder = root_folder + "\\13_M-_l_8fb_sweep",
    right_r_0_static_folder = root_folder + "\\10_M+_r_1st_sweep",
    right_r_0_feedback_folder = root_folder + "\\9_M+_r_8fb_sweep（开放薄棉）",
    left_background_folder = root_folder + "\\15_X_L_bg_sweep",
    right_background_folder = root_folder + "\\16_X_R_bg_sweep",
    save_path = root_folder + "\\plot_data.pkl",
)

# %% 数据准备2
_ = prepare_comprehensive_experiment_data(
    left_r_0_static_folder = root_folder + "\\12_M-_R_1st_sweep",
    left_r_0_feedback_folder = root_folder_2 + "\\11_M-_R_8fb_sweep",
    left_r_minus1_static_folder = root_folder + "\\5_M+_L_1st_sweep",
    left_r_minus1_feedback_folder = root_folder_2 + "\\4_M+_L_8fb_sweep",
    right_r_plus1_static_folder = root_folder + "\\14_M-_l_1st_sweep",
    right_r_plus1_feedback_folder = root_folder_2 + "\\13_M-_l_8fb_sweep",
    right_r_0_static_folder = root_folder + "\\10_M+_r_1st_sweep",
    right_r_0_feedback_folder = root_folder_2 + "\\9_M+_r_8fb_sweep",
    left_background_folder = root_folder + "\\15_X_L_bg_sweep",
    right_background_folder = root_folder + "\\16_X_R_bg_sweep",
    save_path = root_folder_2 + "\\plot_data.pkl",
)

# %% 绘图
# mode = "abs"
mode = "fourier"
_ = plot_comprehensive_experiment(
    data_pkl_path=root_folder + "\\plot_data.pkl",
    # data_pkl_path=root_folder_2 + "\\plot_data.pkl",
    # --- 区域选取参数 ---
    picked_center=Point2D(x=200.0, y=155.5),
    picked_area_radius=100,
    area_shape="square",
    # --- 积分参数 ---
    integral_mode=mode,
    k_modulus=2 * np.pi / 0.1,
    k_angle_deg=180,
    save_path = root_folder + f"\\{mode}",
    # save_path = root_folder_2 + f"\\{mode}",
)
