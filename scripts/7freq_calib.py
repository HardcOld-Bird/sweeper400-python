"""
# Caliber 硬件测试脚本

这是一个用于实际硬件测试的脚本，可以直接运行来测试校准功能。
"""

from sweeper400.calib import FrequencyOptimizer

ai_channels = (
    "PXI1Slot3/ai0",
    "PXI1Slot3/ai1",
    "PXI1Slot4/ai0",
    "PXI1Slot4/ai1",
    "PXI1Slot5/ai0",
    "PXI1Slot5/ai1",
    "PXI1Slot6/ai0",
    "PXI1Slot6/ai1",
)
ao_channel = "PXI1Slot2/ao0"

# 创建测试对象
fo = FrequencyOptimizer(
    ai_channels=ai_channels,
    ao_channel=ao_channel,
    amplitude=0.05,
)

# %% 执行幅值模式测试
fo.optimize(
    result_folder="D:\\EveryoneDownloaded\\freq_calib_test_amp",
)

# %% 执行相位模式测试
# fo.optimize(
#     mode="phase",
#     result_folder="D:\\EveryoneDownloaded\\freq_calib_test_phs",
# )

# %% 执行最终测试，结果存储在项目storage目录下
fo.optimize()
