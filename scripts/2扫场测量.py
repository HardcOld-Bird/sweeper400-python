# pyright: basic
from sweeper400.analyze import (
    get_sine_cycles,
    init_sampling_info,
    init_sine_args,
)
from sweeper400.use import (
    SweeperCore,
    get_square_grid,
    silent_feedback,
    static_uniform_feedback,
    static_diff_feedback,

)

# 定义通道配置
ai_channels = (
    "PXI1Slot2/ai0",
    "PXI1Slot3/ai0",
    "PXI1Slot3/ai1",
    "PXI1Slot4/ai0",
    "PXI1Slot4/ai1",
    "PXI1Slot5/ai0",
    "PXI1Slot5/ai1",
    "PXI1Slot6/ai0",
    "PXI1Slot6/ai1",
)
sweep_ai_channel="PXI1Slot2/ai0"
ao_channels_static = ("PXI1Slot2/ao0",)
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

# 创建输出波形
sampling_info = init_sampling_info(171500.0, 85750)  # 采样率171.5kHz, 0.5秒
sine_args = init_sine_args(
    frequency=3430.0, amplitude=0.000001, phase=0.0
)  # 3430Hz正弦波，波长10cm
static_output_waveform = get_sine_cycles(sampling_info, sine_args)

# 创建点阵
grid = get_square_grid(10.0, 20.0, 10.0, 20.0)

# 创建Sweeper对象
swp = SweeperCore(
    ai_channels=ai_channels,
    sweep_ai_channel=sweep_ai_channel,
    ao_channels_static=ao_channels_static,
    ao_channels_feedback=ao_channels_feedback,
    static_output_waveform=static_output_waveform,
    feedback_function=static_diff_feedback,
    point_list=grid,
)

# %% 步进电机校准
swp.calib()

# %% 检查位置
swp.where()

# %% 移动位置
swp.move_to(1.0, 1.0)

# %% 移动位置
swp.move_to(310.0, 310.0)

# %% 创建点阵
grid = get_square_grid(10.0, 310.0, 270.0, 310.0)
swp.new_point_list(grid)

# %% 开始扫场测量
swp.sweep(
    result_folder="D:\\EveryoneDownloaded\\sweep_data"
)

# %% 查看进度
swp.get_progress()

# %% 中止扫场测量
swp.stop()

# %% 销毁扫场控制器
del swp
