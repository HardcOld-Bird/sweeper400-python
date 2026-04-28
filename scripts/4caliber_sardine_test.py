"""
# Caliber 硬件测试脚本

这是一个用于实际硬件测试的脚本，可以直接运行来测试校准功能。
"""

from sweeper400.analyze import init_sampling_info, init_sine_args, plot_sweep_waveforms
from sweeper400.use import CaliberSardine

# %% 创建采样信息和正弦波参数（使用推荐的参数）
sampling_info = init_sampling_info(171500.0, 85750)  # 采样率171.5kHz, 0.5秒
sine_args = init_sine_args(frequency=686.0, amplitude=0.02, phase=0.0)  # 3430Hz正弦波，波长10cm

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
    "PXI1Slot2/ai0",
)
# ao_channels = ("PXI1Slot3/ao0",)
ao_channels = ("PXI1Slot2/ao0",)

# 创建校准对象
caliber = CaliberSardine(
    ai_channels=ai_channels,
    ao_channels=ao_channels,
    sampling_info=sampling_info,
    sine_args=sine_args,
)

# %% 执行校准
caliber.calibrate(
    chunks_per_channel=3,
    result_folder="D:\\EveryoneDownloaded\\sardine_calib",
)

# %% 检查SweepData波形
sd = load_sweep_data("D:\\EveryoneDownloaded\\sardine_calib\\raw_sweep_data.pkl")
plot_sweep_waveforms(
    sd,
    "D:\\EveryoneDownloaded\\sardine",
    zoom_factor=1,
)

# %% 执行最终校准

# 执行校准，结果存储在项目storage目录下
caliber.calibrate(
    chunks_per_channel=10,
)
