"""
# Caliber 硬件测试脚本

这是一个用于实际硬件测试的脚本，可以直接运行来测试校准功能。
"""

from sweeper400.analyze import load_compressed_data, plot_sweep_waveforms
from sweeper400.calib import CaliberFishNet
from sweeper400.config.exp_config import (
    ai_channels,
    ao_channels,
    ao_channels_feedback,
    ao_channels_static,
    best_frequency,
    grid,
    root_folder,
    sampling_info,
    sweep_ai_channel,
)

# %% 创建采样信息和正弦波参数（使用推荐的参数）

# 创建校准对象
caliber = CaliberFishNet(
    ai_channels=ai_channels,
    ao_channels=ao_channels,
    sampling_info=sampling_info,
    amplitude=0.01,
    # amplitude=0.1,
)

# %% 执行测试校准
result_path = "D:\\EveryoneDownloaded\\channel_test_1"
caliber.calibrate(
    starts_num=1,
    chunks_per_start=1,
    result_folder=result_path,
)

# 检查SweepData波形
# result_path = "D:\\EveryoneDownloaded\\channel_test"
# sd = load_compressed_data(result_path + "\\raw_sweep_data_1.pkl")
# plot_sweep_waveforms(
#     sd,
#     result_path,
#     zoom_factor=50,
# )

# %% 执行最终校准，结果存储在项目storage目录下
caliber.calibrate(
    starts_num=5,
    chunks_per_start=3,
)

# %% 检查SweepData波形
sd = load_compressed_data("D:\\XXXIIIGGG\\projects\\pySweep\\pySweepWS\\storage\\calib\\calib_result_fishnet\\raw_sweep_data_1.pkl")
plot_sweep_waveforms(
    sd,
    "D:\\XXXIIIGGG\\projects\\pySweep\\pySweepWS\\storage\\calib\\calib_result_fishnet",
    zoom_factor=1,
)
