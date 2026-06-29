# pyright: basic
"""
# 完整有源超表面实验脚本

该脚本用于按照标准流程执行完整的演化测量和扫场实验。
"""

import time

import numpy as np

from sweeper400.analyze import (
    get_sine,
    plot_waveform,
)
from sweeper400.calib import CaliberFishNet, FrequencyOptimizer
from sweeper400.config.exp_config import (
    ai_channels,
    ao_channels,
    ao_channels_feedback,
    ao_channels_static,
    best_frequency,
    grid,
    root_folder,
    sampling_info,
    sweep_ai_channel,
)
from sweeper400.sim import SimScanner
from sweeper400.use import (
    Evolver,
    SweeperCore,
    load_evolved_waveform,
)

# %%
# ============================================================================
# 1. 扫场传声器归位
# ============================================================================
# - 连接步进电机
# - 确保扫场传声器行程范围内无障碍物

# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    ao_channels_static=("default_ao",),
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
# - 使用正向样件M+（通道数左小右大）
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

# 创建渔网校准器
clb = CaliberFishNet(
    ai_channels=ai_channels,
    ao_channels=ao_channels,
    sampling_info=sampling_info,
    amplitude=0.1,
)
# 定义结果文件夹
result_folder = root_folder + "\\1_FishNet_TFData_L"
# 执行校准
# （初次运行可取消注释 starts_num=1 和 chunks_per_start=1 行，验证所有通道正常工作）
# （正式运行则恢复注释，校准耗时将较长）
clb.calibrate(
    # starts_num=1,
    # chunks_per_start=1,
    result_folder=result_folder,
)

# %%
# ============================================================================
# 4. 基于大响应 FishNet_TFData 进行参数扫描仿真
# ============================================================================
# - 开启COMSOL

# Floquet 增益系数中心值（复平面）
# cr_center: float = 0.994
# ci_center: float = -0.068
# 8周期 增益系数中心值
cr_center: float = 1.009
ci_center: float = -0.05
# 参数扫描半范围
half_scale: float = 0.005

# 指定TFData路径
tf_data_path = root_folder + "\\1_FishNet_TFData_L\\tf_data.pkl"
# 定义结果文件夹
result_folder = root_folder + "\\2_Sim_Scan_L"

# 创建仿真器
sim = SimScanner()
# 连接Server
sim.connect()
# 执行仿真
_ = sim.run_scan(
    cr_min = cr_center - half_scale,
    cr_max = cr_center + half_scale,
    ci_min = ci_center - half_scale,
    ci_max = ci_center + half_scale,
    res = 50,
    input_amp_l="1[Pa]",
    input_amp_r="0[Pa]",
    fishnet_tf_data_path=tf_data_path,
    result_folder=result_folder,
)

# %%
# ============================================================================
# 5. 对正向样件M+左入射（大响应）时的响应进行演化模拟
# ============================================================================

# 声源输出波形复振幅数组（与 ao_channels_static 等长）
static_cca: np.ndarray = np.full(
    len(ao_channels_static),
    0.01 + 0j,
    dtype=np.complex128,
)
# 创建输出波形
static_output_waveform = get_sine(
    sampling_info=sampling_info,
    frequency=best_frequency,
    channel_names=ao_channels_static,
    channel_complex_amplitudes=static_cca,
    full_cycle=True,
)
# 指定TFData路径
tf_data_path = root_folder + "\\1_FishNet_TFData_L\\tf_data.pkl"
# 指定仿真结果路径
sim_result_path = root_folder + "\\2_Sim_Scan_L\\scan_result.npz"
# 创建演化器
evo = Evolver(
    ai_channels=ai_channels,
    ao_channels_static=ao_channels_static,
    ao_channels_feedback=ao_channels_feedback,
    static_output_waveform=static_output_waveform,
    fishnet_tf_data_path=tf_data_path,
    sim_result_scan_path=sim_result_path,
)
# 定义结果文件夹
result_folder = root_folder + "\\3_evo_L"
# 计算理论结果
_ = evo.simulate(
    # 大响应点
    # cr=1.011224,
    # ci=-0.049796,
    # 小响应点
    cr=1.005,
    ci=-0.0501,
    mode="eight_probes",
    ao_amplitude_limit=100,
    result_folder=result_folder,
)

# %%
# ============================================================================
# 6. 对正向样件M+左入射（大响应）时的响应进行演化测量
# ============================================================================

# 进行实际演化
_ = evo.evolve(
    cycles_num=50,
    ao_amplitude_limit=0.5,
    result_folder=result_folder,
)
# 读取演化后的波形
evo_result_L = load_evolved_waveform(
    file_path=result_folder + "\\evolved_waveform.pkl",
    segments=2,
)
# 绘制波形
_ = plot_waveform(
    evo_result_L,
    save_path=result_folder + "\\evolved_waveform.png",
    zoom_factor=1,
)
_ = plot_waveform(
    evo_result_L,
    save_path=result_folder + "\\evolved_waveform_detail.png",
    zoom_factor=200,
)

# %%
# ============================================================================
# 7. 测量样件M+左入射（大响应）时的逆反射场（仅反馈阵列+仅声源）
# ============================================================================
# - 打开扫场窗口（移除附近吸声棉），确保传声器行程范围内无障碍物
# - 开启步进电机
# - 开启NI功放机箱的所有通道

# ===【“仅反馈阵列”测量】===
# 定义结果文件夹
result_folder = root_folder + "\\4_M+_L_8fb_sweep"
# 创建仅包含反馈通道的波形
evo_result_L_8fb = load_evolved_waveform(
    file_path=root_folder + "\\3_evo_L\\evolved_waveform.pkl",
    segments=2,
    picked_channels=ao_channels_feedback,
)
# 绘制波形
_ = plot_waveform(
    evo_result_L_8fb,
    save_path=result_folder + "\\evo_result_L_8fb_waveform.png",
    zoom_factor=1,
)
_ = plot_waveform(
    evo_result_L_8fb,
    save_path=result_folder + "\\evo_result_L_8fb_detail.png",
    zoom_factor=200,
)
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    sweep_ai_channel=sweep_ai_channel,
    ao_channels_static=ao_channels_feedback,
    static_output_waveform=evo_result_L_8fb,
    point_list=grid,
)
# 扫场传声器归位
swp.move_to(1.0, 1.0)
# 步进电机校准
swp.calib()
# 开始扫场测量（阻塞式）
swp.sweep_blocking(
    result_folder=result_folder,
    lowcut=best_frequency / 2,
    highcut=best_frequency * 2,
)
# 清理资源
swp.cleanup()
del swp
# 等待60秒（1分钟）
time.sleep(60)

# ===【“仅声源”测量】===
# 定义结果文件夹
result_folder = root_folder + "\\5_M+_L_1st_sweep"
# 创建仅包含声源通道的波形
evo_result_L_1st = load_evolved_waveform(
    file_path=root_folder + "\\3_evo_L\\evolved_waveform.pkl",
    segments=2,
    picked_channels=ao_channels_static,
)
# 绘制波形
_ = plot_waveform(
    evo_result_L_1st,
    save_path=result_folder + "\\evo_result_L_1st_waveform.png",
    zoom_factor=1,
)
_ = plot_waveform(
    evo_result_L_1st,
    save_path=result_folder + "\\evo_result_L_1st_detail.png",
    zoom_factor=200,
)
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    sweep_ai_channel=sweep_ai_channel,
    ao_channels_static=ao_channels_static,
    static_output_waveform=evo_result_L_1st,
    point_list=grid,
)
# 开始扫场测量（阻塞式）
swp.sweep_blocking(
    result_folder=result_folder,
    lowcut=best_frequency / 2,
    highcut=best_frequency * 2,
)
# 传声器归位
swp.move_to(311.0, 311.0)
# 清理资源
swp.cleanup()
del swp

# %%
# ============================================================================
# 8. 进行频率校准（可选）
# ============================================================================
# - 将声源更换至右入射（小响应）位置r
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
fo.optimize(
    mode="min_amplitude",
    max_iterations=10,
    result_folder=root_folder + "\\0_calib_result_freq_min",
)

# %%
# ============================================================================
# 9. 测量正向样件M+右入射（小响应）时的 FishNet_TFData
# ============================================================================

# 创建渔网校准器
clb = CaliberFishNet(
    ai_channels=ai_channels,
    ao_channels=ao_channels,
    sampling_info=sampling_info,
    amplitude=0.1,
)
# 定义结果文件夹
result_folder = root_folder + "\\6_FishNet_TFData_r"
# 执行校准
# （初次运行可取消注释 starts_num=1 和 chunks_per_start=1 行，验证所有通道正常工作）
# （正式运行则恢复注释，校准耗时将较长）
clb.calibrate(
    # starts_num=1,
    # chunks_per_start=1,
    result_folder=result_folder,
)
# %%
# ============================================================================
# 10. 基于小响应 FishNet_TFData 进行参数扫描仿真
# ============================================================================
# - 开启COMSOL

# Floquet 增益系数中心值（复平面）
# cr_center: float = 0.994
# ci_center: float = -0.068
# 8周期 增益系数中心值
cr_center: float = 1.009
ci_center: float = -0.05
# 参数扫描半范围
half_scale: float = 0.005

# 指定TFData路径
tf_data_path = root_folder + "\\6_FishNet_TFData_r\\tf_data.pkl"
# 定义结果文件夹
result_folder = root_folder + "\\7_Sim_Scan_r"

# 创建仿真器
sim = SimScanner()
# 连接Server
sim.connect()
# 执行仿真
_ = sim.run_scan(
    cr_min = cr_center - half_scale,
    cr_max = cr_center + half_scale,
    ci_min = ci_center - half_scale,
    ci_max = ci_center + half_scale,
    res = 50,
    input_amp_l="0[Pa]",
    input_amp_r="1[Pa]",
    fishnet_tf_data_path=tf_data_path,
    result_folder=result_folder,
)

# %%
# ============================================================================
# 11. 对正向样件M+右入射（小响应）时的响应进行演化模拟
# ============================================================================

# 声源输出波形复振幅数组（与 ao_channels_static 等长）
static_cca: np.ndarray = np.full(
    len(ao_channels_static),
    0.01 + 0j,
    dtype=np.complex128,
)
# 创建输出波形
static_output_waveform = get_sine(
    sampling_info=sampling_info,
    frequency=best_frequency,
    channel_names=ao_channels_static,
    channel_complex_amplitudes=static_cca,
    full_cycle=True,
)
# 指定TFData路径
tf_data_path = root_folder + "\\6_FishNet_TFData_r\\tf_data.pkl"
# 指定仿真结果路径
sim_result_path = root_folder + "\\7_Sim_Scan_r\\scan_result.npz"
# 创建演化器
evo = Evolver(
    ai_channels=ai_channels,
    ao_channels_static=ao_channels_static,
    ao_channels_feedback=ao_channels_feedback,
    static_output_waveform=static_output_waveform,
    fishnet_tf_data_path=tf_data_path,
    sim_result_scan_path=sim_result_path,
)
# 定义结果文件夹
result_folder = root_folder + "\\8_evo_r"
# 计算理论结果
_ = evo.simulate(
    # 大响应点
    cr=1.011224,
    ci=-0.049796,
    # 小响应点
    # cr=1.005,
    # ci=-0.0501,
    mode="eight_probes",
    ao_amplitude_limit=100,
    result_folder=result_folder,
)

# %%
# ============================================================================
# 12. 对正向样件M+右入射（小响应）时的响应进行演化测量
# ============================================================================

# 进行实际演化
_ = evo.evolve(
    cycles_num=200,
    ao_amplitude_limit=0.5,
    result_folder=result_folder,
)
# 读取演化后的波形
evo_result_r = load_evolved_waveform(
    file_path=result_folder + "\\evolved_waveform.pkl",
    segments=2,
)
# 绘制波形
_ = plot_waveform(
    evo_result_r,
    save_path=result_folder + "\\evolved_waveform.png",
    zoom_factor=1,
)
_ = plot_waveform(
    evo_result_r,
    save_path=result_folder + "\\evolved_waveform_detail.png",
    zoom_factor=200,
)

# %%
# ============================================================================
# 13. 测量样件M+右入射（小响应）时的镜面反射场（仅反馈阵列+仅声源）
# ============================================================================
# - 打开扫场窗口（移除附近吸声棉），确保传声器行程范围内无障碍物
# - 开启步进电机
# - 开启NI功放机箱的所有通道

# ===【“仅反馈阵列”测量】===
# 定义结果文件夹
result_folder = root_folder + "\\9_M+_r_8fb_sweep"
# 创建仅包含反馈通道的波形
evo_result_r_8fb = load_evolved_waveform(
    file_path=root_folder + "\\8_evo_r\\evolved_waveform.pkl",
    segments=2,
    picked_channels=ao_channels_feedback,
)
# 绘制波形
_ = plot_waveform(
    evo_result_r_8fb,
    save_path=result_folder + "\\evo_result_r_8fb_waveform.png",
    zoom_factor=1,
)
_ = plot_waveform(
    evo_result_r_8fb,
    save_path=result_folder + "\\evo_result_r_8fb_detail.png",
    zoom_factor=200,
)
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    sweep_ai_channel=sweep_ai_channel,
    ao_channels_static=ao_channels_feedback,
    static_output_waveform=evo_result_r_8fb,
    point_list=grid,
)
# 扫场传声器归位
swp.move_to(1.0, 1.0)
# 步进电机校准
swp.calib()
# 开始扫场测量（阻塞式）
swp.sweep_blocking(
    result_folder=result_folder,
    lowcut=best_frequency / 2,
    highcut=best_frequency * 2,
)
# 清理资源
# swp.cleanup()
del swp
# 等待60秒（1分钟）
# time.sleep(60)

# ===【“仅声源”测量】===
# 定义结果文件夹
result_folder = root_folder + "\\10_M+_r_1st_sweep"
# 创建仅包含声源通道的波形
evo_result_r_1st = load_evolved_waveform(
    file_path=root_folder + "\\8_evo_r\\evolved_waveform.pkl",
    segments=2,
    picked_channels=ao_channels_static,
)
# 绘制波形
_ = plot_waveform(
    evo_result_r_1st,
    save_path=result_folder + "\\evo_result_r_1st_waveform.png",
    zoom_factor=1,
)
_ = plot_waveform(
    evo_result_r_1st,
    save_path=result_folder + "\\evo_result_r_1st_detail.png",
    zoom_factor=200,
)
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    sweep_ai_channel=sweep_ai_channel,
    ao_channels_static=ao_channels_static,
    static_output_waveform=evo_result_r_1st,
    point_list=grid,
)
# 开始扫场测量（阻塞式）
swp.sweep_blocking(
    result_folder=result_folder,
    lowcut=best_frequency / 2,
    highcut=best_frequency * 2,
)
# 传声器归位
swp.move_to(1.0, 1.0)
# 清理资源
# swp.cleanup()
# del swp

# %%
# ============================================================================
# 14. 测量样件M-右入射（大响应）时的镜面反射场（仅反馈阵列+仅声源）
# ============================================================================
# - 将样件更换为M-（通道数右小左大）
# - 建议使用CaliberFishnet验证所有通道正常工作

# ===【“仅反馈阵列”测量】===
# 定义结果文件夹
result_folder = root_folder + "\\11_M-_R_8fb_sweep"
# 创建仅包含反馈通道的波形
evo_result_L_8fb = load_evolved_waveform(
    file_path=root_folder + "\\3_evo_L\\evolved_waveform.pkl",
    segments=2,
    picked_channels=ao_channels_feedback,
)
# 绘制波形
_ = plot_waveform(
    evo_result_L_8fb,
    save_path=result_folder + "\\evo_result_L_8fb_waveform.png",
    zoom_factor=1,
)
_ = plot_waveform(
    evo_result_L_8fb,
    save_path=result_folder + "\\evo_result_L_8fb_detail.png",
    zoom_factor=200,
)
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    sweep_ai_channel=sweep_ai_channel,
    ao_channels_static=ao_channels_feedback,
    static_output_waveform=evo_result_L_8fb,
    point_list=grid,
)
# 扫场传声器归位
swp.move_to(1.0, 1.0)
# 步进电机校准
swp.calib()
# 开始扫场测量（阻塞式）
swp.sweep_blocking(
    result_folder=result_folder,
    lowcut=best_frequency / 2,
    highcut=best_frequency * 2,
)
# 清理资源
del swp

# ===【“仅声源”测量】===
# 定义结果文件夹
result_folder = root_folder + "\\12_M-_R_1st_sweep"
# 创建仅包含声源通道的波形
evo_result_L_1st = load_evolved_waveform(
    file_path=root_folder + "\\3_evo_L\\evolved_waveform.pkl",
    segments=2,
    picked_channels=ao_channels_static,
)
# 绘制波形
_ = plot_waveform(
    evo_result_L_1st,
    save_path=result_folder + "\\evo_result_L_1st_waveform.png",
    zoom_factor=1,
)
_ = plot_waveform(
    evo_result_L_1st,
    save_path=result_folder + "\\evo_result_L_1st_detail.png",
    zoom_factor=200,
)
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    sweep_ai_channel=sweep_ai_channel,
    ao_channels_static=ao_channels_static,
    static_output_waveform=evo_result_L_1st,
    point_list=grid,
)
# 开始扫场测量（阻塞式）
swp.sweep_blocking(
    result_folder=result_folder,
    lowcut=best_frequency / 2,
    highcut=best_frequency * 2,
)
# 传声器归位
swp.move_to(1.0, 1.0)

# %%
# ============================================================================
# 15. 测量样件M-左入射（小响应）时的逆反射场（仅反馈阵列+仅声源）
# ============================================================================
# - 将声源更换至左入射（小响应）位置r
# - 建议使用CaliberFishnet验证所有通道正常工作

# ===【“仅反馈阵列”测量】===
# 定义结果文件夹
result_folder = root_folder + "\\13_M-_l_8fb_sweep"
# 创建仅包含反馈通道的波形
evo_result_r_8fb = load_evolved_waveform(
    file_path=root_folder + "\\8_evo_r\\evolved_waveform.pkl",
    segments=2,
    picked_channels=ao_channels_feedback,
)
# 绘制波形
_ = plot_waveform(
    evo_result_r_8fb,
    save_path=result_folder + "\\evo_result_r_8fb_waveform.png",
    zoom_factor=1,
)
_ = plot_waveform(
    evo_result_r_8fb,
    save_path=result_folder + "\\evo_result_r_8fb_detail.png",
    zoom_factor=200,
)
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    sweep_ai_channel=sweep_ai_channel,
    ao_channels_static=ao_channels_feedback,
    static_output_waveform=evo_result_r_8fb,
    point_list=grid,
)
# 扫场传声器归位
swp.move_to(1.0, 1.0)
# 步进电机校准
swp.calib()
# 开始扫场测量（阻塞式）
swp.sweep_blocking(
    result_folder=result_folder,
    lowcut=best_frequency / 2,
    highcut=best_frequency * 2,
)
# 清理资源
del swp

# ===【“仅声源”测量】===
# 定义结果文件夹
result_folder = root_folder + "\\14_M-_l_1st_sweep"
# 创建仅包含声源通道的波形
evo_result_r_1st = load_evolved_waveform(
    file_path=root_folder + "\\8_evo_r\\evolved_waveform.pkl",
    segments=2,
    picked_channels=ao_channels_static,
)
# 绘制波形
_ = plot_waveform(
    evo_result_r_1st,
    save_path=result_folder + "\\evo_result_r_1st_waveform.png",
    zoom_factor=1,
)
_ = plot_waveform(
    evo_result_r_1st,
    save_path=result_folder + "\\evo_result_r_1st_detail.png",
    zoom_factor=200,
)
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    sweep_ai_channel=sweep_ai_channel,
    ao_channels_static=ao_channels_static,
    static_output_waveform=evo_result_r_1st,
    point_list=grid,
)
# 开始扫场测量（阻塞式）
swp.sweep_blocking(
    result_folder=result_folder,
    lowcut=best_frequency / 2,
    highcut=best_frequency * 2,
)
# 传声器归位
swp.move_to(1.0, 1.0)

# %%
# ============================================================================
# 16. 测量左入射时的背景场（仅声源）（可重复多次）
# ============================================================================
# - 移除超表面样件，并尽可能填充吸声棉，减少反射声
# - 开启NI功放机箱的声源通道（可关闭所有反馈通道）

# 定义结果文件夹
result_folder = root_folder + "\\15_X_L_bg_sweep"
# 创建仅包含声源通道的波形
evo_result_L_bg = load_evolved_waveform(
    file_path=root_folder + "\\3_evo_L\\evolved_waveform.pkl",
    segments=2,
    picked_channels=ao_channels_static,
)
# 绘制波形
_ = plot_waveform(
    evo_result_L_bg,
    save_path=result_folder + "\\evo_result_L_bg_waveform.png",
    zoom_factor=1,
)
_ = plot_waveform(
    evo_result_L_bg,
    save_path=result_folder + "\\evo_result_L_bg_detail.png",
    zoom_factor=200,
)
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    ao_channels_static=ao_channels_static,
    static_output_waveform=evo_result_L_bg,
    point_list=grid,
)
# 开始扫场测量
swp.sweep_blocking(
    result_folder=result_folder,
    lowcut=best_frequency / 2,
    highcut=best_frequency * 2,
)
# 传声器归位
swp.move_to(1.0, 1.0)

# %%
# ============================================================================
# 17. 测量右入射时的背景场（仅声源）（可重复多次）
# ============================================================================
# - 将声源更换至右入射（小响应）位置r

# 定义结果文件夹
result_folder = root_folder + "\\16_X_R_bg_sweep"
# 创建仅包含声源通道的波形
evo_result_R_bg = load_evolved_waveform(
    file_path=root_folder + "\\8_evo_r\\evolved_waveform.pkl",
    segments=2,
    picked_channels=ao_channels_static,
)
# 绘制波形
_ = plot_waveform(
    evo_result_R_bg,
    save_path=result_folder + "\\evo_result_R_bg_waveform.png",
    zoom_factor=1,
)
_ = plot_waveform(
    evo_result_R_bg,
    save_path=result_folder + "\\evo_result_R_bg_detail.png",
    zoom_factor=200,
)
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    ao_channels_static=ao_channels_static,
    static_output_waveform=evo_result_R_bg,
    point_list=grid,
)
# 开始扫场测量
swp.sweep_blocking(
    result_folder=result_folder,
    lowcut=best_frequency / 2,
    highcut=best_frequency * 2,
)
# 传声器归位
swp.move_to(1.0, 1.0)
