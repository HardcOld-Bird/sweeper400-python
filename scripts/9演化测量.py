# pyright: basic
import numpy as np

from sweeper400.analyze import (
    get_sine,
    init_sampling_info,
    load_compressed_data,
    plot_waveform,
    plot_sweep_waveforms,
)
from sweeper400.use import (
    Evolver,
    load_evolved_waveform,
    SweeperCore,
    get_square_grid,
)

# %% 定义通道配置
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
# 采样率建议为目标频率倍数，且尽可能接近又不大于200kHz
# 0.2s有时会显示警告，建议0.4s
sampling_info = init_sampling_info(170437.5, 68600)  # 0.4秒
cca = np.full(
    len(ao_channels_static),
    0.05 + 0j,
    dtype=np.complex128,
)
static_output_waveform = get_sine(
    sampling_info=sampling_info,
    frequency=3408.75,
    channel_names=ao_channels_static,
    channel_complex_amplitudes=cca,
    full_cycle=False,
)

# 设定增益系数
gain_coefficients = (
    -0.308988-2.557868j,
    -1.196245-1.412161j,
    -1.541522-1.275200j,
    -1.518875-1.254205j,
    -1.534567-1.262961j,
    -1.532444-1.264957j,
    -1.499747-1.267534j,
    -1.703700-1.289199j,
)

# 创建Evolver对象
evo = Evolver(
    ai_channels=ai_channels,
    ao_channels_static=ao_channels_static,
    ao_channels_feedback=ao_channels_feedback,
    static_output_waveform=static_output_waveform,
    gain_coefficients=gain_coefficients,
    # fishnet_tf_data_path="D:\\EveryoneDownloaded\\fishnet_r\\tf_data.pkl",
    fishnet_tf_data_path="D:\\EveryoneDownloaded\\fishnet_L\\tf_data.pkl",
)

# %% 开始模拟
_ = evo.simulate(
    cycles_num=10,
    ao_amplitude_limit=100,
    result_folder="D:\\EveryoneDownloaded\\sim_L_0d4s\\",
)
# %% 进行更多模拟
_ = evo.simulate(
    cycles_num=20,
    ao_amplitude_limit=100,
    result_folder="D:\\EveryoneDownloaded\\sim_L_0d4s\\",
)
_ = evo.simulate(
    cycles_num=50,
    ao_amplitude_limit=100,
    result_folder="D:\\EveryoneDownloaded\\sim_L_0d4s\\",
)

# %% 开始演化
evo.evolve(
    cycles_num=10,
    ao_amplitude_limit=0.5,
    result_folder="D:\\EveryoneDownloaded\\evo_L_0d4s\\",
)

# %% 检查SweepData波形
sd = load_compressed_data("D:\\EveryoneDownloaded\\evo_L_0d4s\\raw_sweep_data.pkl")
plot_sweep_waveforms(
    sd,
    "D:\\EveryoneDownloaded\\evo_L_0d4s",
    zoom_factor=1,
)

# %% 读取演化后的波形
# final_waveform = load_evolved_waveform(
#     file_path="D:\\EveryoneDownloaded\\L_6step_0d2s\\evolved_waveform.pkl",
#     segments=None,
# )
# plot_waveform(
#     final_waveform,
#     save_path="D:\\EveryoneDownloaded\\L_6step_0d2s\\final_waveform.png",
#     zoom_factor=100,
# )
#
#
# # %% 创建Sweeper对象
#
# # 创建Sweeper对象
# swp = SweeperCore(
#     ai_channels=("PXI1Slot2/ai0",),
#     sweep_ai_channel="PXI1Slot2/ai0",
#     ao_channels_static=(
#     "PXI1Slot2/ao0",
#     "PXI1Slot3/ao0",
#     "PXI1Slot3/ao1",
#     "PXI1Slot4/ao0",
#     "PXI1Slot4/ao1",
#     "PXI1Slot5/ao0",
#     "PXI1Slot5/ao1",
#     "PXI1Slot6/ao0",
#     "PXI1Slot6/ao1",
# ),
#     static_output_waveform=final_waveform,
#     point_list=get_square_grid(1.0, 311.0, 1.0, 311.0),
# )
#
# # %% 步进电机校准
# swp.calib()
#
# # %% 检查位置
# swp.where()
#
# # %% 移动位置
# swp.move_to(160.0, 10.0)
#
# # %% 开始扫场测量
# swp.sweep(
#     result_folder="D:\\EveryoneDownloaded\\L_6step_0d2s_feedback_sweep\\",
#     lowcut=1715.0,
#     highcut=6860.0,
# )
#
# # %%
# swp.plot_data(
#     save_path="D:\\EveryoneDownloaded\\L_6step_0d2s_feedback_sweep\\",
#     lowcut=1715.0,
#     highcut=6860.0,
# )
#
# # %% 查看进度
# swp.get_progress()
#
# # %% 中止扫场测量
# swp.stop()
#
# # %% 销毁对象
# del evo, swp
