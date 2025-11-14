# pyright: basic
from sweeper400.analyze import (
    Point2D,
    get_sine_cycles,
    init_sampling_info,
    init_sine_args,
)
from sweeper400.use import (
    Sweeper,
)

# 创建输出波形
sampling_info = init_sampling_info(171500.0, 85750)  # 采样率171.5kHz, 0.5秒
sine_args = init_sine_args(
    frequency=3430.0, amplitude=0.03, phase=0.0
)  # 3430Hz正弦波，波长10cm
output_waveform = get_sine_cycles(sampling_info, sine_args)

# %%

grid = [Point2D(100.0, 260.0)]
# 创建Sweeper对象
swp = Sweeper(
    ai_channel="400Slot2/ai0",  # 传声器
    ao_channel="400Slot2/ao0",  # 扬声器
    output_waveform=output_waveform,
    point_list=grid,
)

# %%
# 确定点阵
swp.where()

# %%
# swp.move_to(220.0, 220.0)

# %%
grid = [Point2D(100.0, 260.0)]
swp.new_point_list(grid)

# %%
swp.sweep()

# %%
swp.get_progress()

# %%
swp.stop()

# %%
save_path = "D:\\EveryoneDownloaded\\mini_spks\\speaker_10.pkl"
swp.save_data(save_path)

# %%
del swp
