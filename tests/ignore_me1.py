# pyright: basic
from sweeper400.analyze import (
    get_sine_cycles,
    init_sampling_info,
    init_sine_args,
)
from sweeper400.use import (
    Sweeper,
    get_square_grid,
)

# 创建输出波形
sampling_info = init_sampling_info(34300.0, 17150)  # 采样率34.3kHz, 0.5秒
sine_args = init_sine_args(
    frequency=3430.0, amplitude=0.03, phase=0.0
)  # 3430Hz正弦波，波长10cm
output_waveform = get_sine_cycles(sampling_info, sine_args)

# 创建Sweeper对象
swp = Sweeper(
    ai_channel="400Slot2/ai0",  # 传声器
    ao_channel="400Slot2/ao0",  # 扬声器
    output_waveform=output_waveform,
)

# 确定点阵
swp.where()
# swp.move_to(220.0, 220.0)
grid = get_square_grid(20.0, 120.0, 260.0, 120.0)
swp.new_point_list(grid)


swp.sweep()

swp.get_progress()

swp.stop()

save_path = "D:\\EveryoneDownloaded\\test_data.pkl"
swp.save_data(save_path)

del swp
