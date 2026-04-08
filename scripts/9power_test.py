"""
# Caliber 硬件测试脚本

这是一个用于实际硬件测试的脚本，可以直接运行来测试校准功能。
"""

from sweeper400.use import PowerTester

# 创建测试对象
tester = PowerTester(
    ai_channel="PXI1Slot2/ai0",
    ao_channel="PXI1Slot3/ao0",
)

# %% 执行测试
tester.test(
    min_power=0.01,
    max_power=0.1,
    step_num=10,
    work_chunks_num=240,
    result_folder="D:\\EveryoneDownloaded\\power_test",
)

# %% 执行最终测试，结果存储在项目storage目录下
tester.test(
    min_power=0.01,
    max_power=0.1,
    step_num=10,
    work_chunks_num=240,
)
