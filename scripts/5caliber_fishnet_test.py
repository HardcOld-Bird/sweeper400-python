"""
# Caliber 硬件测试脚本

这是一个用于实际硬件测试的脚本，可以直接运行来测试校准功能。
"""

from sweeper400.analyze import init_sampling_info, load_compressed_data, plot_sweep_waveforms
from sweeper400.calib import CaliberFishNet

# %% 创建采样信息和正弦波参数（使用推荐的参数）
# 经测试，chunk size 设置为0.2秒可能导致波形更新不及时，建议大于该时长
sampling_info = init_sampling_info(170437.5, 68600)  # 0.4秒

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
ao_channels = (
    "PXI1Slot3/ao0",
    "PXI1Slot3/ao1",
    "PXI1Slot4/ao0",
    "PXI1Slot4/ao1",
    "PXI1Slot5/ao0",
    "PXI1Slot5/ao1",
    "PXI1Slot6/ao0",
    "PXI1Slot6/ao1",
    "PXI1Slot2/ao0",
)

# 创建校准对象
caliber = CaliberFishNet(
    ai_channels=ai_channels,
    ao_channels=ao_channels,
    sampling_info=sampling_info,
    # amplitude=0.01,
    amplitude=0.1,
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
sd = load_compressed_data(result_path + "\\raw_sweep_data_1.pkl")
plot_sweep_waveforms(
    sd,
    result_path,
    zoom_factor=50,
)

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
