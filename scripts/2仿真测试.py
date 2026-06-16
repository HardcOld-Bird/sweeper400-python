"""
# Caliber 硬件测试脚本

这是一个用于实际硬件测试的脚本，可以直接运行来测试校准功能。
"""

import numpy as np

from sweeper400.analyze import get_sine, init_sampling_info
from sweeper400.sim import SimScanner
from sweeper400.use import Evolver

# 创建测试对象
simer = SimScanner()
# 连接Server
simer.connect()

# %% 执行仿真

# 定义方便参数
# cr_center = 1.006
# ci_center = -0.073
# half_scale = 0.016

cr_center = 1.01
ci_center = -0.06
half_scale = 1

# 执行仿真
sim_result = simer.run_scan(
    # f = 3430.0,
    # cr = 1.006,
    cr_min = cr_center - half_scale,
    cr_max = cr_center + half_scale,
    # ci = -0.073,
    ci_min = ci_center - half_scale,
    ci_max = ci_center + half_scale,
    res = 50,
)

# %% Evolver协同测试
my_cr = 0.990
# 经观察，Scanner对实验稳态的预测是正确的，只不过：
#   在接近实验稳态极点时，系统收敛过慢（增益系数*松弛因子过小）；
#   在接近增益系数极点（也即仿真极点）时，系统不收敛（增益因子过大）。
# 整体上，松弛因子分析对系统收敛的预测是准确的。
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
)
sweep_ai_channel = "PXI1Slot2/ai0"
ao_channels_feedback = (
    "PXI1Slot3/ao0",
    "PXI1Slot3/ao1",
    "PXI1Slot4/ao0",
    "PXI1Slot4/ao1",
    "PXI1Slot5/ao0",
    "PXI1Slot5/ao1",
    "PXI1Slot6/ao0",
    "PXI1Slot6/ao1",
)
ao_channels_static = ("PXI1Slot2/ao0",)
ao_channels = ao_channels_feedback + ao_channels_static

sampling_info = init_sampling_info(170437.5, 68600)

# 指定输出波形复振幅
cca = np.full(
    len(ao_channels_static),
    0.1 + 0j,
    dtype=np.complex128,
)
# 创建输出波形
static_output_waveform = get_sine(
    sampling_info=sampling_info,
    frequency=3408.75,
    channel_names=ao_channels_static,
    channel_complex_amplitudes=cca,
    full_cycle=True,
)
# 创建Evolver对象
evo = Evolver(
    ai_channels=ai_channels,
    ao_channels_static=ao_channels_static,
    ao_channels_feedback=ao_channels_feedback,
    static_output_waveform=static_output_waveform,
)
# 计算理论结果
_ = evo.simulate(
    cr=my_cr,
    ci=-0.07265,
    mode="floquet_probes",
    ao_amplitude_limit=100,
    result_folder="D:\\EveryoneDownloaded\\sim_test",
)

# %% 结束仿真
# 断开连接
simer.disconnect()
# 删除对象
del simer
