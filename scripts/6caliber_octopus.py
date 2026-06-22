"""
# Caliber 硬件测试脚本

这是一个用于实际硬件测试的脚本，可以直接运行来测试校准功能。
"""

from sweeper400.analyze import init_sampling_info, load_compressed_data, plot_sweep_waveforms
from sweeper400.calib import CaliberOctopus

ai_channels = ("PXI1Slot5/ai0",)
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
temp_freq = 3407.5
temp_sampling_info = init_sampling_info(temp_freq*50,34300)

# 创建校准对象
caliber = CaliberOctopus(
    ai_channels=ai_channels,
    ao_channels=ao_channels,
    sampling_info=temp_sampling_info,
    frequency=temp_freq,
    amplitude=0.05,
)

# %% 执行校准
caliber.calibrate(
    starts_num=5,
    chunks_per_start=3,
    result_folder="D:\\EveryoneDownloaded\\before_calib",
)

# %% 检查SweepData波形
sd = load_compressed_data("D:\\EveryoneDownloaded\\before_calib\\raw_sweep_data_1.pkl")
plot_sweep_waveforms(
    sd,
    "D:\\EveryoneDownloaded\\before_calib",
    zoom_factor=100,
)

# %% 创建验证对象并再次校准

caliber = CaliberOctopus(
    ai_channels=ai_channels,
    ao_channels=ao_channels,
    sampling_info=temp_sampling_info,
    frequency=temp_freq,
    amplitude=0.05,
    ao_comp_data="D:\\EveryoneDownloaded\\before_calib\\ao_comp_data.pkl",
)

caliber.calibrate(
    starts_num=1,
    chunks_per_start=3,
    result_folder="D:\\EveryoneDownloaded\\after_calib",
)

# %% 执行最终校准

# 创建无补偿校准对象
caliber = CaliberOctopus(
    ai_channels=ai_channels,
    ao_channels=ao_channels,
    sampling_info=temp_sampling_info,
    frequency=temp_freq,
    amplitude=0.1,
)

# 执行校准，结果存储在项目storage目录下
caliber.calibrate(
    starts_num=10,
    chunks_per_start=3,
)

# %% 检查SweepData波形
sd = load_compressed_data("D:\\XXXIIIGGG\\projects\\pySweep\\pySweepWS\\storage\\calib\\calib_result_octopus\\raw_sweep_data_1.pkl")
plot_sweep_waveforms(
    sd,
    "D:\\XXXIIIGGG\\projects\\pySweep\\pySweepWS\\storage\\calib\\calib_result_octopus",
    zoom_factor=1,
)
