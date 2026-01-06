# pyright: basic
from sweeper400.analyze import (
    get_sine_cycles,
    init_sampling_info,
    init_sine_args,
)
from sweeper400.measure import (
    MultiChasCSIO,
)

# %% 创建输出波形
sampling_info = init_sampling_info(171500.0, 85750)  # 采样率171.5kHz, 0.5秒
sine_args = init_sine_args(
    frequency=3430.0, amplitude=0.01, phase=0.0
)  # 3430Hz正弦波，波长10cm
output_waveform = get_sine_cycles(sampling_info, sine_args)


# 定义数据导出函数
def export_data(ai_waveform, chunks_num):
    print(f"导出第 {chunks_num} 段数据")


# %%

# 创建Sweeper对象
csio = MultiChasCSIO(
    ai_channel="PXI2Slot2/ai0",
    ao_channels=(
        "PXI2Slot2/ao0",
        "PXI2Slot2/ao1",
        "PXI2Slot3/ao0",
        "PXI2Slot3/ao1",
        "PXI3Slot2/ao0",
        "PXI3Slot2/ao1",
        "PXI3Slot3/ao0",
        "PXI3Slot3/ao1",
    ),
    output_waveform=output_waveform,
    export_function=export_data,
)

# %% 启动任务
csio.start()

# %% 开始导出
csio.enable_export = True

# %% 关闭导出
csio.enable_export = False

# %% solo通道1
csio.set_ao_channels_status((True, False, False, False, False, False, False, False))
# %% solo通道2
csio.set_ao_channels_status((False, True, False, False, False, False, False, False))
# %% solo通道3
csio.set_ao_channels_status((False, False, True, False, False, False, False, False))
# %% solo通道4
csio.set_ao_channels_status((False, False, False, True, False, False, False, False))
# %% solo通道5
csio.set_ao_channels_status((False, False, False, False, True, False, False, False))
# %% solo通道6
csio.set_ao_channels_status((False, False, False, False, False, True, False, False))
# %% solo通道7
csio.set_ao_channels_status((False, False, False, False, False, False, True, False))
# %% solo通道8
csio.set_ao_channels_status((False, False, False, False, False, False, False, True))
# %% mute all
csio.set_ao_channels_status((False, False, False, False, False, False, False, False))

# %% 停止任务
csio.stop()

# %%
del csio
