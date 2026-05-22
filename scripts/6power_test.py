"""
# Caliber 硬件测试脚本

这是一个用于实际硬件测试的脚本，可以直接运行来测试校准功能。
"""

from sweeper400.calib import PowerTester

# 创建测试对象
tester = PowerTester(
    ai_channel="PXI1Slot2/ai0",
    ao_channel="PXI1Slot3/ao0",
)

# %% 执行近零测试
tester.test(
    start_power=0.01,
    end_power=0.02,
    step_num=2,
    work_chunks_num=50,  # 10秒
    result_folder="D:\\EveryoneDownloaded\\Ch1_T_0.01_0.02",
)

# %% 执行测试
tester.test(
    start_power=0.6,
    end_power=0.1,
    step_num=2,
    work_chunks_num=300,  # 1分钟
    result_folder="D:\\EveryoneDownloaded\\Ch1_C3_0.6_0.1",
)

# %% 执行最终测试，结果存储在项目storage目录下
tester.test(
    start_power=0.01,
    end_power=0.1,
    step_num=10,
    work_chunks_num=300,
)
