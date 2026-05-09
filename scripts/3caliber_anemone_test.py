"""
# Caliber 硬件测试脚本

这是一个用于实际硬件测试的脚本，可以直接运行来测试校准功能。
"""

from sweeper400.analyze import init_sampling_info, plot_sweep_waveforms
from sweeper400.calib import CaliberAnemone
from sweeper400.analyze import load_sweep_data

# %% 创建采样信息和正弦波参数（使用推荐的参数）
sampling_info = init_sampling_info(171500.0, 34300)  # 采样率171.5kHz, 0.2秒

# 定义通道配置
ai_channels = (
    "PXI1Slot3/ai0",
    "PXI1Slot3/ai1",
    "PXI1Slot4/ai0",
    "PXI1Slot4/ai1",
    "PXI1Slot5/ai0",
    "PXI1Slot5/ai1",
    "PXI1Slot6/ai0",
    "PXI1Slot6/ai1",
    # "PXI1Slot2/ai0",
)

# 创建校准对象
caliber = CaliberAnemone(
    ai_channels=ai_channels,
    sampling_info=sampling_info,
)

# %% 执行校准
caliber.calibrate(
    chunks_num=3,
    result_folder="D:\\EveryoneDownloaded\\before_calib",
)

# %% 检查SweepData波形
sd = load_sweep_data("D:\\EveryoneDownloaded\\before_calib\\raw_sweep_data.pkl")
plot_sweep_waveforms(
    sd,
    "D:\\EveryoneDownloaded\\before_calib",
    zoom_factor=100,
)

# %% 创建验证对象并再次校准

caliber = CaliberAnemone(
    ai_channels=ai_channels,
    sampling_info=sampling_info,
    ai_comp_data="D:\\EveryoneDownloaded\\before_calib\\ai_comp_data.pkl",
)

caliber.calibrate(
    chunks_num=3,
    result_folder="D:\\EveryoneDownloaded\\after_calib",
)

# %% 检查SweepData波形
sd = load_sweep_data("D:\\EveryoneDownloaded\\after_calib\\raw_sweep_data.pkl")
plot_sweep_waveforms(
    sd,
    "D:\\EveryoneDownloaded\\after_calib",
    zoom_factor=100,
)

# %% 执行最终校准

# 创建校准对象
caliber = CaliberAnemone(
    ai_channels=ai_channels,
    sampling_info=sampling_info,
)

# 执行校准，结果存储在项目storage目录下
caliber.calibrate(
    chunks_num=20,
)
