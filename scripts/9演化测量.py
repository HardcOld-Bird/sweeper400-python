# pyright: basic
"""
# 完整有源超表面实验脚本

该脚本用于按照标准流程执行完整的演化测量和扫场实验。
"""

import numpy as np

from sweeper400.analyze import (
    get_sine,
    init_sampling_info,
    load_compressed_data,
    plot_sweep_waveforms,
    plot_waveform,
)
from sweeper400.calib import CaliberFishNet, FrequencyOptimizer
from sweeper400.use import (
    Evolver,
    SweeperCore,
    get_square_grid,
    load_evolved_waveform,
)

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

# 定义总结果文件夹
root_folder = "D:\\EveryoneDownloaded\\exp0529"

# %%
# ============================================================================
# 1. 扫场传声器归位
# ============================================================================
# - 连接步进电机
# - 确保扫场传声器行程范围内无障碍物

# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    ao_channels_static=ao_channels_static,
)
# 传声器归位
swp.move_to(311.0, 311.0)
# 删除对象，避免干扰后续实验流程
del swp

# %%
# ============================================================================
# 2. 进行频率校准
# ============================================================================
# - 在此之前，可选完成Anemone校准（8传声器）
# - 在此之前，建议完成Octopus校准（8扬声器）
# - 使用正向样件M+
# - 声源放置在左入射（大响应）位置L
# - 封闭扫场窗口，充分布置吸声棉
# - 连接NI主机箱
# - 连接NI功放机箱，并开启声源通道

# 创建频率优化器
fo = FrequencyOptimizer(
    ai_channels=ai_channels,
    ao_channel=ao_channels_static[0],
    amplitude=0.1,
)
# 执行校准，结果存储在项目storage目录下
fo.optimize()

# %%
# ============================================================================
# 3. 测量正向样件M+左入射（大响应）时的 FishNet_TFData
# ============================================================================
# - 开启NI功放机箱的所有通道

# 创建采样信息（根据频率校准结果手动创建）
# 经测试，chunk size 设置为0.2秒可能导致波形更新不及时，建议大于该时长
sampling_info = init_sampling_info(170437.5, 68600)  # 0.4秒
# 记录最佳频率以作备用
best_frequency = 3408.75
# 创建渔网校准器
clb = CaliberFishNet(
    ai_channels=ai_channels,
    ao_channels=ao_channels,
    sampling_info=sampling_info,
    amplitude=0.1,
)
# 定义结果文件夹
result_path = root_folder + "\\1_FishNet_TFData_L"
# 执行校准（默认耗时较长）
clb.calibrate(
    # starts_num=1,
    # chunks_per_start=1,
    result_folder=result_path,
)

# %%
# ============================================================================
# 4. 对正向样件M+左入射（大响应）时的响应进行演化测量
# ============================================================================

# 指定输出波形复振幅
cca = np.full(
    len(ao_channels_static),
    0.1 + 0j,
    dtype=np.complex128,
)
# 创建输出波形
static_output_waveform = get_sine(
    sampling_info=sampling_info,
    frequency=best_frequency,
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
    gain_coefficients=gain_coefficients,
    fishnet_tf_data_path=root_folder + "\\1_FishNet_TFData_L\\tf_data.pkl",
)
# 定义结果文件夹
result_path = root_folder + "\\3_evo_L"
# 计算理论结果
_ = evo.simulate(
    cycles_num=10,
    ao_amplitude_limit=100,
    result_folder=result_path,
)
_ = evo.simulate(
    cycles_num=20,
    ao_amplitude_limit=100,
    result_folder=result_path,
)
_ = evo.simulate(
    cycles_num=50,
    ao_amplitude_limit=100,
    result_folder=result_path,
)
# 进行实际演化
_ = evo.evolve(
    cycles_num=20,
    ao_amplitude_limit=0.5,
    result_folder=result_path,
)
# 读取演化后的波形
evo_result_L = load_evolved_waveform(
    file_path=result_path + "\\evolved_waveform.pkl",
    segments=2,
)
# 绘制波形
_ = plot_waveform(
    evo_result_L,
    save_path=result_path + "\\evolved_waveform.png",
    zoom_factor=1,
)
_ = plot_waveform(
    evo_result_L,
    save_path=result_path + "\\evolved_waveform_detail.png",
    zoom_factor=50,
)

# %%
# ============================================================================
# 5. 测量正向样件M+右入射（小响应）时的 FishNet_TFData
# ============================================================================
# - 将声源更换至右入射（小响应）位置r

# 定义结果文件夹
result_path = root_folder + "\\2_FishNet_TFData_r"
# 执行校准（默认耗时较长）
clb.calibrate(
    # starts_num=1,
    # chunks_per_start=1,
    result_folder=result_path,
)

# %%
# ============================================================================
# 6. 对正向样件M+右入射（小响应）时的响应进行演化测量
# ============================================================================

# 创建Evolver对象
evo = Evolver(
    ai_channels=ai_channels,
    ao_channels_static=ao_channels_static,
    ao_channels_feedback=ao_channels_feedback,
    static_output_waveform=static_output_waveform,
    gain_coefficients=gain_coefficients,
    fishnet_tf_data_path=root_folder + "\\2_FishNet_TFData_r\\tf_data.pkl",
)
# 定义结果文件夹
result_path = root_folder + "\\4_evo_r"
# 计算理论结果
_ = evo.simulate(
    cycles_num=10,
    ao_amplitude_limit=100,
    result_folder=result_path,
)
_ = evo.simulate(
    cycles_num=20,
    ao_amplitude_limit=100,
    result_folder=result_path,
)
_ = evo.simulate(
    cycles_num=50,
    ao_amplitude_limit=100,
    result_folder=result_path,
)
# 进行实际演化
_ = evo.evolve(
    cycles_num=20,
    ao_amplitude_limit=0.5,
    result_folder=result_path,
)
# 读取演化后的波形
evo_result_r = load_evolved_waveform(
    file_path=result_path + "\\evolved_waveform.pkl",
    segments=2,
)
# 绘制波形
_ = plot_waveform(
    evo_result_r,
    save_path=result_path + "\\evolved_waveform.png",
    zoom_factor=1,
)
_ = plot_waveform(
    evo_result_r,
    save_path=result_path + "\\evolved_waveform_detail.png",
    zoom_factor=50,
)
# 对演化结果满意后，即可继续至扫场环节

# %%
# ============================================================================
# 7. 测量样件M+右入射（小响应）时的镜面反射场（仅反馈阵列）
# ============================================================================
# - 开启NI功放机箱的所有反馈通道，关闭声源通道
# - 打开扫场窗口（移除附近吸声棉），确保传声器行程范围内无障碍物

# 创建点阵
grid = get_square_grid(1.0, 311.0, 1.0, 311.0)
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    ao_channels_static=ao_channels,
    static_output_waveform=evo_result_r,
    point_list=grid,
)
# 扫场传声器
swp.move_to(1.0, 1.0)
# 步进电机校准
swp.calib()
# 开始扫场测量
swp.sweep(
    result_folder=root_folder + "\\5__sweep\\",
    lowcut=best_frequency / 2,
    highcut=best_frequency * 2,
)
