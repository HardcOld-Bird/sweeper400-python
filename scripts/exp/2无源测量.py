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
# 2. 测量样件M+左入射时的逆反射场
# ============================================================================
# - 正向摆放样件至M+状态
# - 打开扫场窗口（移除附近吸声棉），确保传声器行程范围内无障碍物
# - 开启步进电机
# - 开启NI功放机箱的声源通道

# 定义结果文件夹
result_folder = root_folder + "\\5_M+_L_1st_sweep"
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
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    sweep_ai_channel=sweep_ai_channel,
    ao_channels_static=ao_channels_static,
    static_output_waveform=static_output_waveform,
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

# %%
# ============================================================================
# 3. 测量样件M-左入射时的逆反射场
# ============================================================================
# - 将样件翻面至M-状态

# 定义结果文件夹
result_folder = root_folder + "\\14_M-_l_1st_sweep"
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
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    sweep_ai_channel=sweep_ai_channel,
    ao_channels_static=ao_channels_static,
    static_output_waveform=static_output_waveform,
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

# %%
# ============================================================================
# 4. 测量样件M-右入射时的镜面反射场
# ============================================================================
# - 将声源更换至右入射位置r

# 定义结果文件夹
result_folder = root_folder + "\\12_M-_R_1st_sweep"
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
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    sweep_ai_channel=sweep_ai_channel,
    ao_channels_static=ao_channels_static,
    static_output_waveform=static_output_waveform,
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

# %%
# ============================================================================
# 5. 测量样件M+右入射时的镜面反射场
# ============================================================================
# - 将样件翻面至M+状态
# - 打开扫场窗口（移除附近吸声棉），确保传声器行程范围内无障碍物
# - 开启步进电机
# - 开启NI功放机箱的所有通道

# 定义结果文件夹
result_folder = root_folder + "\\10_M+_r_1st_sweep"
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
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    sweep_ai_channel=sweep_ai_channel,
    ao_channels_static=ao_channels_static,
    static_output_waveform=static_output_waveform,
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

# %%
# ============================================================================
# 6. 测量左入射时的背景场（仅声源）（可重复多次）
# ============================================================================
# - 移除超表面样件，并尽可能填充吸声棉，减少反射声

# 定义结果文件夹
result_folder = root_folder + "\\15_X_L_bg_sweep"
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
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    sweep_ai_channel=sweep_ai_channel,
    ao_channels_static=ao_channels_static,
    static_output_waveform=static_output_waveform,
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

# %%
# ============================================================================
# 7. 测量右入射时的背景场（仅声源）（可重复多次）
# ============================================================================
# - 将声源更换至右入射（小响应）位置r

# 定义结果文件夹
result_folder = root_folder + "\\16_X_R_bg_sweep"
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
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    sweep_ai_channel=sweep_ai_channel,
    ao_channels_static=ao_channels_static,
    static_output_waveform=static_output_waveform,
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
