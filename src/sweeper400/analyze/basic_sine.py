"""
# 基础正弦波处理模块

模块路径：`sweeper400.analyze.basic_sine`

本模块包含与最简单的**单频正弦波**相关的函数和类。
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import periodogram

from .my_dtypes import (
    PositiveFloat,
    SamplingInfo,
    Waveform,
)
from ..logger import get_logger

# 获取模块日志器
logger = get_logger(__name__)


def get_sine(
    sampling_info: SamplingInfo,
    frequency: PositiveFloat,
    channel_names: tuple[str, ...],
    channel_complex_amplitudes: np.ndarray,
    timestamp: np.datetime64 | None = None,
    waveform_id: int | None = None,
    full_cycle: bool = False,
) -> Waveform:
    """
    生成包含单频正弦波的多（或单）通道Waveform对象

    该函数支持为每个通道应用不同的复振幅：
    - 复数的模长作为该通道的幅值
    - 复数的相角作为该通道的初始相位

    Args:
        sampling_info: 采样信息，包含采样率和采样点数
        frequency: 正弦波频率（Hz），必须为正实数
        channel_names: 通道名称元组，决定输出波形的通道数和channel_names属性
        channel_complex_amplitudes: 各通道复振幅数组（complex128），长度必须等于通道数。
            abs为该通道幅值，相角为该通道初始相位。
        timestamp: 采样开始时间戳，默认值为None（使用当前时间）
        waveform_id: 波形的唯一标识符，默认值为None
        full_cycle: 是否确保输出为整数个完整周期。默认为False。
            - False: 直接使用 sampling_info 中的 samples_num 生成波形
            - True: 自动调整采样点数为整数个周期（向上取整），
              要求采样率必须是频率的整数倍

    Returns:
        output_waveform: 多通道正弦波形Waveform对象，形状为 (n_channels, n_samples)

    Raises:
        ValueError: 当 channel_complex_amplitudes 长度与 channel_names 不匹配时
        ValueError: 当 full_cycle=True 且采样率不是频率的整数倍时

    Examples:
        ```python
        >>> import numpy as np
        >>> from sweeper400.analyze import get_sine, init_sampling_info
        >>> test_sampling_info = init_sampling_info(1000.0, 1024)
        >>> # 生成2通道正弦波，通道0: 幅值1.0相位0, 通道1: 幅值0.5相位π/4
        >>> test_cca = np.array([1.0 + 0j, 0.5 * np.exp(1j * np.pi / 4)])
        >>> wf = get_sine(sampling_info, 50.0, ("ch0", "ch1"), cca)
        >>> print(wf.shape)  # (2, 1024)
        ```
    """
    # 获取函数日志器
    f_logger = get_logger(f"{__name__}.get_sine")

    sampling_rate = sampling_info["sampling_rate"]
    requested_samples = sampling_info["samples_num"]
    channels_num = len(channel_names)

    f_logger.debug(
        f"生成正弦波: frequency={frequency}Hz, channels_num={channels_num}, "
        f"sampling_rate={sampling_rate}Hz, requested_samples={requested_samples}, "
        f"full_cycle={full_cycle}"
    )

    # 确定实际采样点数
    if full_cycle:
        # 整周期模式：检查采样率是否是频率的整数倍
        samples_per_cycle = sampling_rate / frequency
        if abs(samples_per_cycle - round(samples_per_cycle)) > 1e-10:
            f_logger.error(
                f"采样率 {sampling_rate}Hz 不是频率 {frequency}Hz 的整数倍。"
                f"一个周期需要 {samples_per_cycle:.6f} 个采样点，不是整数。",
                exc_info=True,
            )
            raise ValueError(
                f"采样率 {sampling_rate}Hz 不是频率 {frequency}Hz 的整数倍。"
                f"一个周期需要 {samples_per_cycle:.6f} 个采样点，不是整数。"
            )

        samples_per_cycle_int = int(round(samples_per_cycle))

        # 根据输入的samples_num计算总周期数（向上取整）
        total_cycles_int = int(np.ceil(requested_samples / samples_per_cycle_int))
        actual_samples = total_cycles_int * samples_per_cycle_int

        f_logger.debug(
            f"整周期模式: 每周期{samples_per_cycle_int}点, "
            f"周期数={total_cycles_int}, 实际采样点数={actual_samples}"
        )
    else:
        actual_samples = requested_samples

    # 生成时间序列
    duration = actual_samples / sampling_rate
    time_array = np.linspace(0, duration, actual_samples, endpoint=False)

    # 生成多通道波形数据
    # 利用向量化运算: 对每个通道使用其复振幅的模长和相角
    amplitudes = np.abs(channel_complex_amplitudes)  # (channels_num,)
    phases = np.angle(channel_complex_amplitudes)  # (channels_num,)

    # 向量化生成: (channels_num, 1) * sin(2π*f*t + (channels_num, 1))
    # time_array shape: (actual_samples,)
    # 广播: amplitudes[:, None] * sin(2π*f*time_array[None, :] + phases[:, None])
    omega_t = 2 * np.pi * frequency * time_array[np.newaxis, :]  # (1, actual_samples)
    multi_ch_data = amplitudes[:, np.newaxis] * np.sin(
        omega_t + phases[:, np.newaxis]
    )  # (channels_num, actual_samples)

    # 创建Waveform对象
    output_waveform = Waveform(
        input_array=multi_ch_data,
        sampling_rate=sampling_rate,
        channel_names=channel_names,
        timestamp=timestamp,
        waveform_id=waveform_id,
        frequency=frequency,
        channel_complex_amplitudes=channel_complex_amplitudes,
    )

    f_logger.debug(
        f"成功生成正弦波Waveform对象: shape={output_waveform.shape}, "
        f"duration={output_waveform.duration:.6f}s"
    )

    return output_waveform


def extract_single_tone_information_vvi(
    input_waveform: Waveform,
    approx_freq: PositiveFloat | None = None,
    error_percentage: PositiveFloat = 5.0,
    precise_mode: bool = False,
) -> Waveform:
    """
    对多通道Waveform进行单频信息提取，返回记录了估计结果的Waveform对象

    该函数对输入波形中的单频成分进行检测和参数估计，同时支持多通道和单通道波形。
    针对多通道情形进行了效率优化：频率估计仅使用第一个通道进行一次，
    然后将结果应用到所有通道。

    提取完成后，将 estimated_frequency 和 channel_complex_amplitudes 写入
    input_waveform 的相应属性，并作为 output_waveform 返回。若 input_waveform
    的 frequency 或 channel_complex_amplitudes 属性原本不为 None，则会输出
    警告并覆盖原有值。

    Args:
        input_waveform: 目标波形（多或单通道），形状为 (n_channels, n_samples)
        approx_freq: 搜索的中心频率（Hz）。默认为None，表示在全频率范围内搜索
        error_percentage: 允许的频率误差百分数，无单位。默认值为5.0
        precise_mode: 是否启用精确估计模式。默认为False。
            - False: 使用Periodogram+抛物线插值进行频率估计，线性最小二乘进行
              幅值和相位估计。速度快，适合大多数场景。
            - True: 在粗略估计基础上，使用curve_fit进行非线性最小二乘精确优化。
              精度更高但速度较慢。

    Returns:
        output_waveform: 记录了提取结果的Waveform对象（与input_waveform共享数据），
            其 frequency 属性为估计的正弦波频率（Hz），
            channel_complex_amplitudes 属性为各通道复振幅数组，形状为 (n_channels,)，
            dtype=complex128。abs为幅值，相角为相位（弧度制）。

    Examples:
        ```python
        >>> import numpy as np
        >>> from sweeper400.analyze import get_sine, init_sampling_info
        >>> sampling_info = init_sampling_info(1000.0, 1024)
        >>> cca = np.array([2.0 + 0j])  # 单通道，幅值2.0，相位0
        >>> wf = get_sine(sampling_info, 50.0, ("ch0",), cca)
        >>> result_wf = extract_single_tone_information_vvi(wf, approx_freq=50.0)
        >>> print(f"频率: {result_wf.frequency:.2f}Hz, 幅值: {np.abs(result_wf.channel_complex_amplitudes[0]):.4f}")
        ```
    """
    # 获取函数日志器
    f_logger = get_logger(f"{__name__}.extract_single_tone_information_vvi")

    channels_num = input_waveform.channels_num
    samples_num = input_waveform.samples_num
    sampling_rate = input_waveform.sampling_rate

    f_logger.debug(
        f"开始单频信息提取: channels={channels_num}, samples={samples_num}, "
        f"sampling_rate={sampling_rate}Hz, approx_freq={approx_freq}Hz, "
        f"precise_mode={precise_mode}"
    )

    # ==================== 第一阶段：共享频率估计 ====================
    # 使用第一个通道进行频率估计（假设所有通道频率相同）
    first_channel_data = input_waveform[0, :]
    nyquist_freq = sampling_rate / 2.0

    # 确定搜索频率范围
    if approx_freq is not None:
        freq_min_candidate = approx_freq * (1 - error_percentage / 100.0)
        freq_min = max(freq_min_candidate, 0.0)
        freq_max_candidate = approx_freq / (1 - error_percentage / 100.0)
        freq_max = min(freq_max_candidate, nyquist_freq)
        f_logger.debug(
            f"限定频率范围搜索: {freq_min:.2f}Hz - {freq_max:.2f}Hz "
            f"(中心频率: {approx_freq:.2f}Hz, 误差: ±{error_percentage:.1f}%)"
        )
    else:
        freq_min = 0.0
        freq_max = nyquist_freq
        f_logger.debug(f"全频率范围搜索: 0Hz - {nyquist_freq:.2f}Hz")

    # 使用Periodogram计算功率谱
    psd_freqs, psd_values = periodogram(
        first_channel_data,
        fs=sampling_rate,
        window="hann",
        detrend=False,
        return_onesided=True,
        scaling="spectrum",
    )

    # 转换为幅度谱
    psd_magnitude = np.sqrt(psd_values) * np.sqrt(2.0)

    # 在指定频率范围内搜索峰值
    freq_range_mask = (psd_freqs >= freq_min) & (psd_freqs <= freq_max)
    if not np.any(freq_range_mask):
        f_logger.error(
            f"指定的频率范围 [{freq_min:.2f}, {freq_max:.2f}] Hz 超出了有效范围",
            exc_info=True,
        )
        raise ValueError(
            f"指定的频率范围 [{freq_min:.2f}, {freq_max:.2f}] Hz 超出了有效范围"
        )

    psd_magnitude_in_range = psd_magnitude[freq_range_mask]
    psd_freqs_in_range = psd_freqs[freq_range_mask]

    # 找到幅度最大的频率点
    max_magnitude_idx = np.argmax(psd_magnitude_in_range)
    coarse_frequency = psd_freqs_in_range[max_magnitude_idx]

    # 抛物线拟合精细化频率估计
    if 0 < max_magnitude_idx < len(psd_magnitude_in_range) - 1:
        y1 = psd_magnitude_in_range[max_magnitude_idx - 1]
        y2 = psd_magnitude_in_range[max_magnitude_idx]
        y3 = psd_magnitude_in_range[max_magnitude_idx + 1]
        denominator = float(y1 - 2 * y2 + y3)
        if abs(denominator) > 1e-12:
            delta = 0.5 * float(y1 - y3) / denominator
            freq_resolution = float(psd_freqs[1] - psd_freqs[0])
            estimated_frequency = float(coarse_frequency) + delta * freq_resolution
        else:
            estimated_frequency = float(coarse_frequency)
    else:
        estimated_frequency = float(coarse_frequency)

    f_logger.debug(f"共享频率估计结果: {estimated_frequency:.6f}Hz")

    # ==================== 第二阶段：向量化幅值/相位估计（所有通道）====================
    # 使用线性最小二乘对所有通道同时进行相位和幅值估计
    # 截取前1~3个周期的数据用于相位估计（降低频率误差影响）
    cycles_for_phase_estimation = 3
    samples_per_cycle = float(sampling_rate) / estimated_frequency
    phase_estimation_samples = int(cycles_for_phase_estimation * samples_per_cycle)
    phase_estimation_samples = min(phase_estimation_samples, samples_num)

    time_array_for_phase = np.arange(phase_estimation_samples) / sampling_rate
    cos_term = np.cos(2 * np.pi * estimated_frequency * time_array_for_phase)
    sin_term = np.sin(2 * np.pi * estimated_frequency * time_array_for_phase)
    design_matrix = np.column_stack([cos_term, sin_term])

    # 截取所有通道的前phase_estimation_samples个样本
    waveform_data_for_phase = input_waveform[:, :phase_estimation_samples]

    # 向量化最小二乘求解：使用正规方程 (X^T X)^(-1) X^T y
    XtX = design_matrix.T @ design_matrix
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
    XtX_inv_Xt = XtX_inv @ design_matrix.T  # shape: (2, phase_estimation_samples)

    # 对所有通道同时计算系数: (channels_num, 2)
    coefficients = waveform_data_for_phase @ XtX_inv_Xt.T  # (channels_num, 2)

    # 提取a（cos系数）和b（sin系数）
    a_coeffs = coefficients[:, 0]
    b_coeffs = coefficients[:, 1]

    # 计算相位和幅值
    estimated_phases = np.arctan2(a_coeffs, b_coeffs)

    # 使用RMS方法计算幅值: 幅值 = sqrt(2) * RMS
    rms_values = np.sqrt(np.mean(input_waveform**2, axis=1))
    estimated_amplitudes = rms_values * np.sqrt(2)

    f_logger.debug(
        f"向量化幅值/相位估计完成: 幅值范围[{estimated_amplitudes.min():.6f}, "
        f"{estimated_amplitudes.max():.6f}], 相位范围[{estimated_phases.min():.6f}, "
        f"{estimated_phases.max():.6f}]"
    )

    # ==================== 第三阶段：可选的精确优化 ====================
    if precise_mode:
        f_logger.debug("启用精确模式，开始curve_fit优化...")
        time_array = np.arange(samples_num) / sampling_rate

        # 初始化优化结果数组
        optimized_amplitudes = np.zeros(channels_num)
        optimized_phases = np.zeros(channels_num)

        # 第一步：使用第一个通道优化频率（同时优化幅值和相位）
        def sine_model_full(
            t: np.ndarray, amplitude: float, frequency: float, phase: float
        ) -> np.ndarray:
            """完整正弦波模型: y = amplitude * sin(2π * f * t + phase)"""
            return amplitude * np.sin(2 * np.pi * frequency * t + phase)

        initial_params_full = [
            estimated_amplitudes[0],
            estimated_frequency,
            estimated_phases[0],
        ]

        try:
            optimal_params_full, _ = curve_fit(
                sine_model_full,
                time_array,
                input_waveform[0, :],
                p0=initial_params_full,
                bounds=([0.0, freq_min, -np.pi], [np.inf, freq_max, np.pi]),
                maxfev=5000,
            )
            optimized_frequency = float(optimal_params_full[1])
            optimized_amplitudes[0] = float(optimal_params_full[0])
            optimized_phases[0] = float(optimal_params_full[2])
            f_logger.debug(f"频率精确优化结果: {optimized_frequency:.6f}Hz")
        except Exception as e:
            f_logger.warning(f"频率优化失败: {e}，使用粗略估计的频率")
            optimized_frequency = estimated_frequency
            optimized_amplitudes[0] = estimated_amplitudes[0]
            optimized_phases[0] = estimated_phases[0]

        # 第二步：使用优化后的频率，对其余通道优化幅值和相位
        if channels_num > 1:
            def sine_model_fixed_freq(
                t: np.ndarray, amplitude: float, phase: float
            ) -> np.ndarray:
                """正弦波模型（固定频率）"""
                return amplitude * np.sin(
                    2 * np.pi * optimized_frequency * t + phase
                )

            for ch_idx in range(1, channels_num):
                channel_data = input_waveform[ch_idx, :]
                initial_params = [
                    estimated_amplitudes[ch_idx],
                    estimated_phases[ch_idx],
                ]

                try:
                    optimal_params, _ = curve_fit(
                        sine_model_fixed_freq,
                        time_array,
                        channel_data,
                        p0=initial_params,
                        bounds=([0.0, -np.pi], [np.inf, np.pi]),
                        maxfev=5000,
                    )
                    optimized_amplitudes[ch_idx] = float(optimal_params[0])
                    optimized_phases[ch_idx] = float(optimal_params[1])
                except Exception as e:
                    f_logger.warning(
                        f"通道{ch_idx}的curve_fit优化失败: {e}，使用粗略估计"
                    )
                    optimized_amplitudes[ch_idx] = estimated_amplitudes[ch_idx]
                    optimized_phases[ch_idx] = estimated_phases[ch_idx]

        # 使用优化后的值
        estimated_frequency = optimized_frequency
        estimated_amplitudes = optimized_amplitudes
        estimated_phases = optimized_phases
        f_logger.debug("curve_fit精确优化完成")

    # ==================== 构建复振幅结果 ====================
    # 复振幅 = 幅值 * exp(1j * 相位)
    channel_complex_amplitudes = estimated_amplitudes * np.exp(1j * estimated_phases)

    f_logger.debug(
        f"单频信息提取完成: frequency={estimated_frequency:.6f}Hz, "
        f"输出形状={channel_complex_amplitudes.shape}, "
        f"复振幅幅值范围=[{np.abs(channel_complex_amplitudes).min():.6f}, "
        f"{np.abs(channel_complex_amplitudes).max():.6f}]"
    )

    # ==================== 构建输出Waveform ====================
    # 检查input_waveform是否已有frequency或channel_complex_amplitudes
    if input_waveform.frequency is not None:
        f_logger.warning(
            f"input_waveform的frequency属性不为None（当前值={input_waveform.frequency}），"
            f"将被覆盖为新估计值 {estimated_frequency}"
        )
    if input_waveform.channel_complex_amplitudes is not None:
        f_logger.warning(
            "input_waveform的channel_complex_amplitudes属性不为None，"
            "将被覆盖为新估计值"
        )

    # 创建output_waveform：使用input_waveform的数据和元数据，更新频率和复振幅
    output_waveform = Waveform(
        input_array=np.asarray(input_waveform),
        sampling_rate=input_waveform.sampling_rate,
        channel_names=input_waveform.channel_names,
        timestamp=input_waveform.timestamp,
        waveform_id=input_waveform.waveform_id,
        frequency=estimated_frequency,
        channel_complex_amplitudes=channel_complex_amplitudes,
    )

    return output_waveform
