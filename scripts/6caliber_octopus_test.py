"""
# Caliber 硬件测试脚本

这是一个用于实际硬件测试的脚本，可以直接运行来测试校准功能。
"""

from sweeper400.analyze import init_sampling_info, init_sine_args, plot_sweep_waveforms
from sweeper400.use import CaliberOctopus, load_sweep_data

# %% 创建采样信息和正弦波参数（使用推荐的参数）
sampling_info = init_sampling_info(171500.0, 85750)  # 采样率171.5kHz, 0.5秒
# sampling_info = init_sampling_info(171500.0, 17150)  # 采样率171.5kHz, 0.1秒
sine_args = init_sine_args(frequency=3430.0, amplitude=0.01, phase=0.0)  # 3430Hz正弦波，波长10cm

# 定义通道配置
ai_channel = "PXI1Slot2/ai0"
ao_channels = (
    "PXI1Slot3/ao0",
    "PXI1Slot3/ao1",
    "PXI1Slot4/ao0",
    "PXI1Slot4/ao1",
    "PXI1Slot5/ao0",
    "PXI1Slot5/ao1",
    "PXI1Slot6/ao0",
    "PXI1Slot6/ao1",
)

# 创建校准对象
caliber = CaliberOctopus(
    ai_channel=ai_channel,
    ao_channels=ao_channels,
    sampling_info=sampling_info,
    sine_args=sine_args,
)

# %% 执行校准
caliber.calibrate(
    starts_num=2,
    chunks_per_start=3,
    result_folder="D:\\EveryoneDownloaded\\before_calib",
)

# %% 检查SweepData波形
sd = load_sweep_data("D:\\EveryoneDownloaded\\before_calib\\raw_sweep_data.pkl")
plot_sweep_waveforms(
    sd,
    "D:\\EveryoneDownloaded\\before_calib",
    zoom_factor=1,
)

# %% 创建验证对象并再次校准

caliber = CaliberOctopus(
    ai_channel=ai_channel,
    ao_channels=ao_channels,
    sampling_info=sampling_info,
    sine_args=sine_args,
    comp_data="D:\\EveryoneDownloaded\\before_calib\\comp_data.pkl",
)

caliber.calibrate(
    starts_num=2,
    chunks_per_start=3,
    result_folder="D:\\EveryoneDownloaded\\after_calib",
)

# %% 执行最终校准

# 创建无补偿校准对象
caliber = CaliberOctopus(
    ai_channel=ai_channel,
    ao_channels=ao_channels,
    sampling_info=sampling_info,
    sine_args=sine_args,
)

# 执行校准，结果存储在项目storage目录下
caliber.calibrate(
    starts_num=10,
    chunks_per_start=3,
)
