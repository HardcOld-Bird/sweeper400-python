# pyright: basic
import numpy as np

from sweeper400.analyze import (
    get_sine,
    init_sampling_info,
)
from sweeper400.use import (
    SweeperCore,
    get_square_grid,
    load_evolved_waveform,
)

# 定义通道配置
ai_channels = (
    "PXI1Slot2/ai0",
    # "PXI1Slot3/ai0",
    # "PXI1Slot3/ai1",
    # "PXI1Slot4/ai0",
    # "PXI1Slot4/ai1",
    # "PXI1Slot5/ai0",
    # "PXI1Slot5/ai1",
    # "PXI1Slot6/ai0",
    # "PXI1Slot6/ai1",
)
sweep_ai_channel="PXI1Slot2/ai0"
ao_channels_static = (
    "PXI1Slot2/ao0",
    # "PXI1Slot3/ao0",
    # "PXI1Slot3/ao1",
    # "PXI1Slot4/ao0",
    # "PXI1Slot4/ao1",
    # "PXI1Slot5/ao0",
    # "PXI1Slot5/ao1",
    # "PXI1Slot6/ao0",
    # "PXI1Slot6/ao1",
)
# ao_channels_feedback = (
#     "PXI1Slot3/ao0",
#     "PXI1Slot3/ao1",
#     "PXI1Slot4/ao0",
#     "PXI1Slot4/ao1",
#     "PXI1Slot5/ao0",
#     "PXI1Slot5/ao1",
#     "PXI1Slot6/ao0",
#     "PXI1Slot6/ao1",
# )

# 读取（或创建）输出波形
# static_output_waveform = load_evolved_waveform(
#     file_path="D:\\EveryoneDownloaded\\L_10step_0d4s\\evolved_waveform.pkl",
#     segments=2,
# )

# 采样率建议为目标频率倍数，且尽可能接近又不大于200kHz
# 单段采样数≥8575（0.05s）以上时可正常运行
# 然而，总停顿时间过短时，步进电机将无法稳定工作
# 因此，建议至少停顿0.5s（例如0.2s×3）
sampling_info = init_sampling_info(171500.0, 34300)  # 采样率171.5kHz, 0.2秒
static_output_waveform = get_sine(
    sampling_info=sampling_info,
    frequency=3430.0,
    channel_names=ao_channels_static,
    channel_complex_amplitudes=np.asarray([0.01 + 0.0j]),
    full_cycle=True,
)

# 创建点阵
grid = get_square_grid(1.0, 311.0, 1.0, 311.0)

# 创建Sweeper对象
swp = SweeperCore(
    ai_channels=ai_channels,
    sweep_ai_channel=sweep_ai_channel,
    ao_channels_static=ao_channels_static,
    # ao_channels_feedback=ao_channels_feedback,
    static_output_waveform=static_output_waveform,
    point_list=grid,
)

# %% 开始扫场测量
swp.sweep(
    result_folder="D:\\EveryoneDownloaded\\L_INPUT_sweep",
    lowcut=1715.0,
    highcut=6860.0,
)

# %% 步进电机校准
swp.calib()

# %% 检查位置
swp.where()

# %% 移动位置1
swp.move_to(1.0, 1.0)

# %% 移动位置2
swp.move_to(1.0, 155.0)

# %% 移动位置3
swp.move_to(1.0, 311.0)

# %% 移动位置4
swp.move_to(311.0, 155.0)

# %% 移动位置5
swp.move_to(311.0, 311.0)

# %% 创建点阵
grid = get_square_grid(1.0, 311.0, 1.0, 311.0)
swp.new_point_list(grid)

# %% 查看进度
swp.get_progress()

# %% 中止扫场测量
swp.stop()

# %% 销毁扫场控制器
del swp
