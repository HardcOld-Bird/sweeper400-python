"""
# 基础正弦波处理模块

模块路径：`sweeper400.analyze.basic_sine`

本模块包含与最简单的**单频正弦波**相关的函数和类。
"""

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import curve_fit
from scipy.signal import periodogram

from .my_dtypes import (
    PositiveFloat,
    SamplingInfo,
    SineArgs,
    Waveform,
    init_sine_args,
)
from ..logger import get_logger

# 获取模块日志器
logger = get_logger(__name__)


def get_sine(
    sampling_info: SamplingInfo,
    sine_args: SineArgs,
    channel_name: str | None = None,
    timestamp: np.datetime64 | None = None,
    waveform_id: int | None = None,
) -> Waveform:
    """
    使用几个简单的参数生成包含单频正弦波时域信号的Waveform对象

    Args:
        sampling_info: 采样信息，包含采样率和采样点数
        sine_args: 正弦波参数，包含频率、幅值和相位信息
        channel_name: 通道名称，默认值为None
        timestamp: 采样开始时间戳，默认值为None
        waveform_id: 波形的唯一标识符，默认值为None

    Returns:
        output_sine_wave: 包含单频正弦波的Waveform对象

    Examples:
        ```python
        >>> from analyze import init_sampling_info
        >>> test_sampling_info = init_sampling_info(1000, 1024)
        >>> test_sine_args = init_sine_args(50.0, 1.0, 0.0)
        >>> sine_wave = get_sine(sampling_info, sine_args)
        >>> print(sine_wave.shape)
        (1, 1024)
        >>> print(sine_wave.sampling_rate)
        1000.0
        ```
    """
    # 获取函数日志器
    f_logger = get_logger(f"{__name__}.get_sine")

    f_logger.debug(
        f"生成正弦波: frequency={sine_args['frequency']}Hz, "
        f"amplitude={sine_args['amplitude']}, phase={sine_args['phase']}rad, "
        f"sampling_rate={sampling_info['sampling_rate']}Hz, "
        f"samples_num={sampling_info['samples_num']}, "
        f"channel_name={channel_name}, "
        f"timestamp={timestamp}, waveform_id={waveform_id}"
    )

    # 生成时间序列
    # 使用 linspace 生成从 0 到 (samples_num-1)/sampling_rate 的时间点
    # endpoint=False 确保不包含最后一个时间点，避免周期性信号的重复
    duration = sampling_info["samples_num"] / sampling_info["sampling_rate"]
    time_array = np.linspace(0, duration, sampling_info["samples_num"], endpoint=False)

    # 生成正弦波数据
    # y(t) = amplitude * sin(2π * frequency * t + phase)
    sine_data = sine_args["amplitude"] * np.sin(
        2 * np.pi * sine_args["frequency"] * time_array + sine_args["phase"]
    )

    # 设置通道名称
    if channel_name is not None:
        channel_names = (channel_name,)
    else:
        channel_names = None

    # 创建Waveform对象
    output_sine_wave = Waveform(
        input_array=sine_data,
        sampling_rate=sampling_info["sampling_rate"],
        channel_names=channel_names,
        timestamp=timestamp,
        waveform_id=waveform_id,
        sine_args=sine_args,
    )

    f_logger.debug(f"成功生成正弦波Waveform对象: {output_sine_wave}")

    return output_sine_wave


def get_sine_cycles(
    sampling_info: SamplingInfo,
    sine_args: SineArgs,
    channel_name: str | None = None,
    timestamp: np.datetime64 | None = None,
    waveform_id: int | None = None,
) -> Waveform:
    """
    生成若干个完整周期的连续正弦波形Waveform

    该函数生成的波形始终从0相位开始并在0相位结束，确保输出波形可以无缝重复连接。
    输出波形的长度会根据输入参数自动调整为整数个周期。

    Args:
        sampling_info: 采样信息，包含采样率和采样点数
        sine_args: 正弦波参数，包含频率、幅值和相位信息（注意：相位参数会被忽略）
        channel_name: 通道名称，默认值为None
        timestamp: 采样开始时间戳，默认值为None
        waveform_id: 波形的唯一标识符，默认值为None

    Returns:
        output_sine_wave: 包含整数个周期正弦波的Waveform对象

    Raises:
        ValueError: 当采样率不是频率的整数倍时抛出

    Examples:
        ```python
        >>> # 生成1000Hz采样率下50Hz的正弦波，包含完整周期
        >>> from analyze import init_sampling_info, init_sine_args
        >>> test_sampling_info = init_sampling_info(1000, 1024)
        >>> test_sine_args = init_sine_args(50.0, 1.0, 0.0)  # 相位参数会被忽略
        >>> sine_wave = get_sine_cycles(test_sampling_info, test_sine_args)
        >>> print(f"实际采样点数: {sine_wave.shape[0]}")
        >>> print(f"周期数: {sine_wave.shape[0] * 50.0 / 1000}")
        ```
    """
    # 获取函数日志器
    f_logger = get_logger(f"{__name__}.get_sine_cycles")

    f_logger.debug(
        f"生成整周期正弦波: frequency={sine_args['frequency']}Hz, "
        f"amplitude={sine_args['amplitude']}, "
        f"sampling_rate={sampling_info['sampling_rate']}Hz, "
        f"requested_samples={sampling_info['samples_num']}, "
        f"channel_name={channel_name}, "
        f"timestamp={timestamp}, waveform_id={waveform_id}"
    )

    # 1. 检查采样率是否是频率的整数倍
    frequency = sine_args["frequency"]
    sampling_rate = sampling_info["sampling_rate"]

    # 计算一个周期的采样点数
    samples_per_cycle = sampling_rate / frequency

    # 检查是否为整数（允许小的浮点误差）
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
    f_logger.debug(f"每个周期采样点数: {samples_per_cycle_int}")

    # 2. 根据输入的samples_num计算总周期数（向上取整）
    requested_samples = sampling_info["samples_num"]
    total_cycles = np.ceil(requested_samples / samples_per_cycle_int)
    total_cycles_int = int(total_cycles)

    # 计算实际的采样点数（整数个周期）
    actual_samples = total_cycles_int * samples_per_cycle_int

    f_logger.debug(
        f"请求采样点数: {requested_samples}, "
        f"计算周期数: {total_cycles:.2f} -> {total_cycles_int}, "
        f"实际采样点数: {actual_samples}"
    )

    # 3. 生成时间序列（从0开始，确保整周期）
    # 使用 linspace 生成从 0 到 total_cycles_int 个周期的时间点
    # endpoint=False 确保不包含最后一个时间点，避免重复0相位点
    duration = actual_samples / sampling_rate
    time_array = np.linspace(0, duration, actual_samples, endpoint=False)

    # 4. 生成正弦波数据（忽略输入的相位，始终从0相位开始）
    # y(t) = amplitude * sin(2π * frequency * t + 0)
    sine_data = sine_args["amplitude"] * np.sin(2 * np.pi * frequency * time_array)

    # 创建修正的正弦波参数（相位设为0）
    actual_sine_args = init_sine_args(
        frequency=frequency,
        amplitude=sine_args["amplitude"],
        phase=0.0,  # 始终从0相位开始
    )

    # 设置通道名称
    if channel_name is not None:
        channel_names = (channel_name,)
    else:
        channel_names = None

    # 创建Waveform对象
    output_sine_wave = Waveform(
        input_array=sine_data,
        sampling_rate=sampling_rate,
        channel_names=channel_names,
        timestamp=timestamp,
        waveform_id=waveform_id,
        sine_args=actual_sine_args,
    )

    f_logger.debug(
        f"成功生成整周期正弦波Waveform对象: {output_sine_wave}, "
        f"周期数: {total_cycles_int}"
    )

    return output_sine_wave


def get_sine_multi_ch(
    sampling_info: SamplingInfo,
    sine_args: SineArgs,
    channel_names: tuple[str, ...],
    complex_amps: tuple[complex, ...] = (),
    timestamp: np.datetime64 | None = None,
    waveform_id: int | None = None,
) -> Waveform:
    """
    生成多通道正弦波形，支持为每个通道应用不同的复振幅调整

    该函数生成一个多通道波形。默认情况下，所有通道的数据相同（同步）。
    当提供 complex_amps 参数时，每个通道的波形将根据对应的复数进行调整：
    - 复数的模长作为幅值乘子
    - 复数的相位作为相位加子

    这是波形生成的第一步，生成的波形可以通过 comp_multi_ch_wf 函数进行补偿。

    Args:
        sampling_info: 采样信息，包含采样率和采样点数
        sine_args: 正弦波参数（频率、幅值和相位）
        channel_names: 通道名称元组，决定输出波形的通道数和channel_names属性
        complex_amps: 复振幅元组，每个复数对应一个通道的幅值乘子和相位加子。
                      如果为空元组或未提供，所有通道使用相同的原始波形。
                      如果元素数少于通道数，多出的通道使用原始波形。
                      如果元素数多于通道数，多出的复数被忽略。
        timestamp: 采样开始时间戳，默认值为None
        waveform_id: 波形的唯一标识符，默认值为None

    Returns:
        output_waveform: 多通道正弦波形（二维数组，每行对应一个通道）

    Examples:
        ```python
        >>> # 生成8通道同步正弦波形（所有通道相同）
        >>> from analyze import init_sampling_info, init_sine_args
        >>> test_sampling_info = init_sampling_info(171500.0, 85750)
        >>> test_sine_args = init_sine_args(3430.0, 0.01, 0.0)
        >>> ao_channels = ("PXI1Slot2/ao0", "PXI1Slot2/ao1", ...)
        >>> multi_ch_waveform = get_sine_multi_ch(sampling_info, test_sine_args, ao_channels)
        >>> print(multi_ch_waveform.shape)  # (8, 85750)

        >>> # 生成带复振幅调整的波形
        >>> import cmath
        >>> # 第一个通道幅值×1.5，相位+0.1rad；第二个通道幅值×0.8，相位-0.2rad
        >>> amps = (1.5 * cmath.exp(0.1j), 0.8 * cmath.exp(-0.2j))
        >>> multi_ch_waveform = get_sine_multi_ch(sampling_info, sine_args, ao_channels[:2], amps)
        ```
    """
    # 获取函数日志器
    f_logger = get_logger(f"{__name__}.get_sine_multi_ch")

    channels_num = len(channel_names)
    has_complex_amps = len(complex_amps) > 0

    f_logger.debug(
        f"生成多通道正弦波形: frequency={sine_args['frequency']}Hz, "
        f"amplitude={sine_args['amplitude']}, phase={sine_args['phase']}rad, "
        f"sampling_rate={sampling_info['sampling_rate']}Hz, "
        f"samples_num={sampling_info['samples_num']}, "
        f"channels_num={channels_num}, "
        f"complex_amps={'provided' if has_complex_amps else 'none'}"
    )

    # 创建多通道波形数组
    multi_ch_data = np.zeros((channels_num, sampling_info["samples_num"]))

    # 为每个通道生成波形
    for ch_idx in range(channels_num):
        # 检查是否有对应的复振幅调整
        if ch_idx < len(complex_amps):
            # 提取复振幅的模长和相位
            amp_multiplier = abs(complex_amps[ch_idx])
            phase_addition = np.angle(complex_amps[ch_idx])

            # 计算调整后的幅值和相位
            adjusted_amplitude = sine_args["amplitude"] * amp_multiplier
            adjusted_phase: float = sine_args["phase"] + phase_addition  # noqa

            f_logger.debug(
                f"通道 {ch_idx} ({channel_names[ch_idx]}): "
                f"幅值乘子={amp_multiplier:.6f}, 相位加子={phase_addition:.6f}rad, "
                f"调整后幅值={adjusted_amplitude:.6f}, 调整后相位={adjusted_phase:.6f}rad"
            )

            # 创建调整后的正弦波参数
            adjusted_sine_args = init_sine_args(
                frequency=sine_args["frequency"],
                amplitude=adjusted_amplitude,
                phase=adjusted_phase,
            )

            # 生成该通道的波形
            channel_waveform = get_sine(
                sampling_info=sampling_info,
                sine_args=adjusted_sine_args,
                timestamp=timestamp,
                waveform_id=waveform_id,
            )
        else:
            # 没有复振幅调整，使用原始参数
            channel_waveform = get_sine(
                sampling_info=sampling_info,
                sine_args=sine_args,
                timestamp=timestamp,
                waveform_id=waveform_id,
            )

        # 将波形数据存入多通道数组
        multi_ch_data[ch_idx, :] = channel_waveform

    # 创建多通道Waveform对象
    output_waveform = Waveform(
        input_array=multi_ch_data,
        sampling_rate=sampling_info["sampling_rate"],
        channel_names=channel_names,
        timestamp=timestamp,
        waveform_id=waveform_id,
        sine_args=sine_args,
    )

    f_logger.debug(
        f"成功生成多通道正弦波形: shape={output_waveform.shape}, "
        f"channels_num={output_waveform.channels_num}"
    )

    return output_waveform


def estimate_sine_args(
    input_waveform: Waveform,
    approx_freq: PositiveFloat | None = None,
    error_percentage: PositiveFloat = 5.0,
) -> SineArgs:
    """
    对Waveform对象进行粗略的正弦波参数估计

    采用混合策略进行参数估计：
    - 频率估计：Periodogram功率谱估计 + 抛物线插值（高精度，抗噪声）
    - 幅值估计：Periodogram功率谱估计 + 抛物线插值（高精度，0~7%误差）
    - 相位估计：短信号线性最小二乘法（使用原始信号，避免窗函数影响）

    这是一个快速的初始估计函数，为后续的精确拟合提供良好的初始值。

    Args:
        input_waveform: 目标波形，将在其中估计单频正弦波参数
        approx_freq: 搜索的中心频率（Hz）。默认为None，表示在全频率范围内搜索
        error_percentage: 允许的频率误差百分数，无单位。默认值为5.0

    Returns:
        estimated_sine_args: 包含估计的频率、幅值和相位信息的字典

    Examples:
        ```python
        >>> # 生成测试波形
        >>> from analyze import init_sampling_info, init_sine_args
        >>> sampling_info = init_sampling_info(1000, 1024)
        >>> sine_args = init_sine_args(50.0, 2.0, 0.5)
        >>> test_wave = get_sine(sampling_info, sine_args)
        >>> estimated_args = estimate_sine_args(test_wave, approx_freq=50.0)
        >>> print(f"估计频率: {estimated_args['frequency']:.2f}Hz")
        ```
    """
    # 获取函数日志器
    f_logger = get_logger(f"{__name__}.estimate_sine_args")

    f_logger.debug(
        f"开始粗略正弦波参数估计: approx_freq={approx_freq}Hz, "
        f"error_percentage={error_percentage}%, "
        f"waveform_shape={input_waveform.shape}, "
        f"sampling_rate={input_waveform.sampling_rate}Hz"
    )

    # 处理多通道数据，只使用第一个通道
    # Waveform 现在统一使用 2D 格式 (n_channels, n_samples)
    waveform_data = input_waveform[0, :]
    if not input_waveform.is_single_channel:
        f_logger.warning(f"检测到多通道数据({input_waveform.channels_num}通道)，使用第一个通道进行分析")

    # 获取基本参数
    sampling_rate = input_waveform.sampling_rate
    samples_num = input_waveform.samples_num
    nyquist_freq = sampling_rate / 2.0

    # 确定搜索频率范围
    if approx_freq is not None:
        # 根据误差百分比确定搜索范围
        freq_min_candidate = approx_freq * (1 - error_percentage / 100.0)
        freq_min = max(freq_min_candidate, 0.0)
        freq_max_candidate = approx_freq / (1 - error_percentage / 100.0)
        freq_max = min(freq_max_candidate, nyquist_freq)

        f_logger.debug(
            f"限定频率范围搜索: {freq_min:.2f}Hz - {freq_max:.2f}Hz "
            f"(中心频率: {approx_freq:.2f}Hz, 误差: ±{error_percentage:.1f}%)"
        )
    else:
        # 全频率范围搜索
        freq_min = 0.0
        freq_max = nyquist_freq
        f_logger.debug(f"全频率范围搜索: 0Hz - {nyquist_freq:.2f}Hz")

    # 使用Periodogram方法计算功率谱
    # window: 使用Hann窗以提高频率分辨率和降低频谱泄漏
    # scaling: 'spectrum' 返回功率谱（单位V^2），峰值的平方根是RMS幅度
    # return_onesided: 只返回正频率部分
    psd_freqs, psd_values = periodogram(
        waveform_data,
        fs=sampling_rate,
        window="hann",
        detrend=False,  # 不去趋势，相关操作交给filter完成
        return_onesided=True,  # 只返回正频率部分
        scaling="spectrum",  # 使用功率谱
    )

    f_logger.debug(
        f"Periodogram频率分辨率={psd_freqs[1] - psd_freqs[0]:.6f}Hz"
    )

    # 将功率谱转换为幅度谱
    # 对于功率谱（scaling='spectrum'），峰值 = (RMS幅度)^2
    # 对于正弦波，峰值幅度 = RMS幅度 * sqrt(2)
    # 因此，峰值幅度 = sqrt(功率谱峰值) * sqrt(2)
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
    coarse_magnitude = psd_magnitude_in_range[max_magnitude_idx]

    # 使用抛物线拟合进行频率和幅值的精细化估计
    if 0 < max_magnitude_idx < len(psd_magnitude_in_range) - 1:
        # 取峰值点及其相邻两点进行抛物线拟合
        y1 = psd_magnitude_in_range[max_magnitude_idx - 1]
        y2 = psd_magnitude_in_range[max_magnitude_idx]
        y3 = psd_magnitude_in_range[max_magnitude_idx + 1]

        # 抛物线插值公式，计算精确频率
        # 检查分母是否为零（这意味着三点共线，不太可能发生）
        denominator = float(y1 - 2 * y2 + y3)
        if abs(denominator) > 1e-12:  # 避免数值不稳定
            delta = 0.5 * float(y1 - y3) / denominator
            # 使用频率分辨率计算精确频率
            freq_resolution = float(psd_freqs[1] - psd_freqs[0])
            initial_frequency = float(coarse_frequency) + delta * freq_resolution
            # 计算精确幅值
            refined_magnitude = float(y2) - 0.25 * float(y1 - y3) * delta
            initial_amplitude = refined_magnitude  # 已经是峰值幅度

            f_logger.debug(
                f"抛物线拟合改进: 频率={initial_frequency:.6f}Hz, "
                f"幅值={initial_amplitude:.6f}, delta={delta:.6f}"
            )
        else:
            # 分母接近零，使用粗略估计
            initial_frequency = coarse_frequency
            initial_amplitude = coarse_magnitude  # 已经是峰值幅度

            f_logger.debug(
                f"抛物线拟合分母接近零，使用粗略估计: "
                f"频率={initial_frequency:.6f}Hz, 幅值={initial_amplitude:.6f}"
            )
    else:
        initial_frequency = coarse_frequency
        initial_amplitude = coarse_magnitude  # 已经是峰值幅度

        f_logger.debug(
            f"边界情况，使用粗略估计: "
            f"频率={initial_frequency:.6f}Hz, 幅值={initial_amplitude:.6f}"
        )

    f_logger.debug(
        f"Periodogram+抛物线拟合初始估计: "
        f"频率={initial_frequency:.6f}Hz, 幅值={initial_amplitude:.6f}"
    )

    # 计算用于相位估计的信号长度（前1-3个周期）
    # 这样可以降低对频率估计误差的敏感性
    cycles_for_phase_estimation = 3  # 使用3个周期
    samples_per_cycle = float(sampling_rate) / float(initial_frequency)
    phase_estimation_samples = int(cycles_for_phase_estimation * samples_per_cycle)

    # 确保不超过原始信号长度
    phase_estimation_samples = min(phase_estimation_samples, samples_num)

    f_logger.debug(
        f"相位估计参数: 使用前{cycles_for_phase_estimation}个周期, "
        f"每周期{samples_per_cycle:.1f}个采样点, "
        f"总计{phase_estimation_samples}个采样点 (原始信号{samples_num}个点)"
    )

    # 截取用于相位估计的信号和时间序列
    waveform_data_for_phase = waveform_data[:phase_estimation_samples]
    time_array_for_phase = np.arange(phase_estimation_samples) / sampling_rate

    # 使用截取的信号进行单次线性最小二乘拟合来估计初始相位
    cos_term_init = np.cos(2 * np.pi * initial_frequency * time_array_for_phase)
    sin_term_init = np.sin(2 * np.pi * initial_frequency * time_array_for_phase)
    design_matrix_init = np.column_stack([cos_term_init, sin_term_init])

    result_init = lstsq(
        design_matrix_init, waveform_data_for_phase, lapack_driver="gelsd"
    )
    coefficients_init = result_init[0]
    residuals_init = result_init[1] if len(result_init) > 1 else None
    rank_init = result_init[2] if len(result_init) > 2 else None
    s_init = result_init[3] if len(result_init) > 3 else None

    a_init, b_init = float(coefficients_init[0]), float(coefficients_init[1])
    initial_phase = float(np.arctan2(a_init, b_init))

    # 安全地处理可能为None的返回值
    condition_number_str = "N/A"
    if s_init is not None and len(s_init) > 0:  # noqa
        condition_number_str = f"{s_init[0] / s_init[-1]:.2e}"

    residuals_str = "N/A"
    if residuals_init is not None:
        # residuals_init 可能是标量或数组
        if np.isscalar(residuals_init):
            residuals_str = f"{residuals_init:.6e}"
        elif len(residuals_init) > 0:  # noqa
            residuals_str = f"{residuals_init[0]:.6e}"

    f_logger.debug(
        f"相位线性拟合结果: a={a_init:.6f}, b={b_init:.6f}, "
        f"矩阵秩={rank_init}, 条件数={condition_number_str}"
    )
    f_logger.debug(f"相位拟合残差平方和: {residuals_str}")

    # 验证初始估计的质量
    f_logger.debug(
        f"粗略估计完成: 频率={initial_frequency:.6f}Hz (Periodogram+抛物线插值), "
        f"幅值={initial_amplitude:.6f} (Periodogram+抛物线插值), "
        f"相位={initial_phase:.6f}rad ({initial_phase * 180 / np.pi:.1f}°) (线性拟合)"
    )

    # 返回粗略估计结果，确保类型正确
    estimated_sine_args = init_sine_args(
        float(initial_frequency), float(initial_amplitude), float(initial_phase)
    )

    return estimated_sine_args


def extract_single_tone_information_vvi(
    input_waveform: Waveform,
    approx_freq: PositiveFloat | None = None,
    error_percentage: PositiveFloat = 5.0,
) -> SineArgs:
    """
    在Waveform对象中搜索指定的频率成分，并返回该成分的详细信息

    Args:
        input_waveform: 目标波形，将在其中搜索单频正弦波
        approx_freq: 搜索的中心频率（Hz）。默认为None，表示在全频率范围内搜索
        error_percentage: 允许的频率误差百分数，无单位。默认值为5.0

    Returns:
        detected_sine_args: 包含搜索到的单频的精确频率、幅值和相位信息的字典

    Examples:
        ```python
        >>> # 生成测试波形
        >>> from analyze import init_sampling_info, init_sine_args
        >>> sampling_info = init_sampling_info(1000, 1024)
        >>> sine_args = init_sine_args(50.0, 2.0, 0.0)
        >>> test_wave = get_sine(sampling_info, sine_args)
        >>> test_sine_args = extract_single_tone_information_vvi(
        ...     test_wave, approx_freq=50.0
        ... )
        >>> print(
        ...     f"频率: {sine_args['frequency']:.2f}Hz, "
        ...     f"幅值: {sine_args['amplitude']:.2f}, "
        ...     f"相位: {sine_args['phase']:.4f}rad"
        ... )
        ```
    """
    # 获取函数日志器
    f_logger = get_logger(f"{__name__}.extract_single_tone_information_vvi")

    f_logger.debug(
        f"开始精确单频信息提取: approx_freq={approx_freq}Hz, "
        f"error_percentage={error_percentage}%, "
        f"waveform_shape={input_waveform.shape}, "
        f"sampling_rate={input_waveform.sampling_rate}Hz"
    )

    # 第一阶段：获取粗略估计
    estimated_args = estimate_sine_args(input_waveform, approx_freq, error_percentage)

    initial_frequency = estimated_args["frequency"]
    initial_amplitude = estimated_args["amplitude"]
    initial_phase = estimated_args["phase"]

    f_logger.debug(
        f"粗略估计结果: 频率={initial_frequency:.6f}Hz, "
        f"幅值={initial_amplitude:.6f}, 相位={initial_phase:.6f}rad"
    )

    # 处理多通道数据，只使用第一个通道
    # Waveform 现在统一使用 2D 格式 (n_channels, n_samples)
    waveform_data = input_waveform[0, :]
    if not input_waveform.is_single_channel:
        f_logger.warning(f"检测到多通道数据({input_waveform.channels_num}通道)，使用第一个通道进行分析")

    samples_num = input_waveform.samples_num
    sampling_rate = input_waveform.sampling_rate

    # 构建完整的时间序列用于curve_fit
    time_array = np.arange(samples_num) / sampling_rate

    # 计算频率搜索范围
    nyquist_freq = sampling_rate / 2
    if approx_freq is not None:
        # 根据误差百分比确定搜索范围
        freq_min_candidate = approx_freq * (1 - error_percentage / 100.0)
        freq_min = max(freq_min_candidate, 0.0)
        freq_max_candidate = approx_freq / (1 - error_percentage / 100.0)
        freq_max = min(freq_max_candidate, nyquist_freq)
    else:
        freq_min = 0.0
        freq_max = nyquist_freq

    # 定义正弦波模型函数
    def sine_model(
        t: np.ndarray, amplitude: float, frequency: float, phase: float
    ) -> np.ndarray:
        """正弦波模型: y = amplitude * sin(2π * frequency * t + phase)"""
        return amplitude * np.sin(2 * np.pi * frequency * t + phase)

    # 设置参数的初始值
    initial_params = [initial_amplitude, initial_frequency, initial_phase]

    # 设置参数边界
    # 幅值：必须为正
    # 频率：在搜索范围内
    # 相位：[-π, π]
    lower_bounds = [0.0, freq_min, -np.pi]
    upper_bounds = [np.inf, freq_max, np.pi]

    try:
        # 使用curve_fit进行非线性最小二乘拟合
        optimal_params, _ = curve_fit(
            sine_model,
            time_array,
            waveform_data,
            p0=initial_params,
            bounds=(lower_bounds, upper_bounds),
            maxfev=5000,  # 增加最大函数评估次数
        )

        detected_amplitude: float
        detected_frequency: float
        detected_phase: float
        detected_amplitude, detected_frequency, detected_phase = (
            float(optimal_params[0]),  # type: ignore
            float(optimal_params[1]),  # type: ignore
            float(optimal_params[2]),  # type: ignore
        )

        # 计算拟合质量
        fitted_signal = sine_model(
            time_array, detected_amplitude, detected_frequency, detected_phase  # noqa
        )
        residuals = waveform_data - fitted_signal
        rms_error = np.sqrt(np.mean(residuals**2))

        f_logger.debug(
            f"curve_fit优化结果: 幅值={detected_amplitude:.6f}, "
            f"频率={detected_frequency:.6f}, 相位={detected_phase:.6f}"
        )
        f_logger.debug(f"拟合RMS误差: {rms_error:.6f}")

    except Exception as e:
        f_logger.warning(f"curve_fit优化失败: {e}")
        f_logger.warning("回退到粗略估计")

        detected_frequency = initial_frequency
        detected_amplitude = initial_amplitude
        detected_phase = initial_phase

    f_logger.debug(
        f"精确估计环节结果: 频率={detected_frequency:.6f}Hz, "
        f"幅值={detected_amplitude:.6f}, 相位={detected_phase:.6f}rad"
    )

    detected_sine_args = init_sine_args(
        detected_frequency, detected_amplitude, detected_phase
    )

    return detected_sine_args


def esti_vvi_multi_ch(  # 待改进
    input_waveform: Waveform,
    approx_freq: PositiveFloat | None = None,
    error_percentage: PositiveFloat = 5.0,
    use_curve_fit: bool = False,
) -> np.ndarray:
    """
    对多通道Waveform进行单频信息提取，返回各通道的复振幅

    该函数专为多通道波形设计，注重效率优化。它对所有通道进行单频检测，
    并将结果转化为复振幅形式返回，便于后续进一步处理。

    优化策略：
    1. 频率估计在所有通道间共享（假设所有通道频率相同），只做一次Periodogram估计
    2. 相位估计使用向量化线性最小二乘，一次性处理所有通道
    3. 可选的curve_fit优化（默认关闭以提升速度）

    Args:
        input_waveform: 目标多通道波形，形状为 (n_channels, n_samples)
        approx_freq: 搜索的中心频率（Hz）。默认为None，表示在全频率范围内搜索
        error_percentage: 允许的频率误差百分数，无单位。默认值为5.0
        use_curve_fit: 是否使用curve_fit进行精确优化。默认为False，
                      使用粗略估计以获得更高效率。设为True可获得更高精度但速度较慢。

    Returns:
        complex_amplitudes: 一维复数ndarray，形状为 (n_channels,)
            每个元素对应一个通道的复振幅：amp * exp(1j * phase)

    Examples:
        ```python
        >>> # 生成8通道测试波形
        >>> from analyze import init_sampling_info, init_sine_args, get_sine_multi_ch
        >>> sampling_info = init_sampling_info(171500.0, 85750)
        >>> sine_args = init_sine_args(3430.0, 1.0, 0.0)
        >>> channels = tuple(f"ch{i}" for i in range(8))
        >>> test_wave = get_sine_multi_ch(sampling_info, sine_args, channels)
        >>> # 提取多通道复振幅
        >>> complex_amps = esti_vvi_multi_ch(test_wave, approx_freq=3430.0)
        >>> print(f"通道0复振幅: {complex_amps[0]:.4f}")
        >>> print(f"所有通道幅值: {np.abs(complex_amps)}")
        ```
    """
    # 获取函数日志器
    f_logger = get_logger(f"{__name__}.esti_vvi_multi_ch")

    n_channels = input_waveform.channels_num
    samples_num = input_waveform.samples_num
    sampling_rate = input_waveform.sampling_rate

    f_logger.debug(
        f"开始多通道单频信息提取: channels={n_channels}, samples={samples_num}, "
        f"sampling_rate={sampling_rate}Hz, approx_freq={approx_freq}Hz, "
        f"use_curve_fit={use_curve_fit}"
    )

    # 获取波形数据 (n_channels, n_samples)
    waveform_data = np.asarray(input_waveform)

    # ==================== 第一阶段：共享频率估计 ====================
    # 使用第一个通道进行频率估计（假设所有通道频率相同）
    # 这样可以避免对每个通道都做一次Periodogram
    first_channel_data = waveform_data[0, :]

    # 复用estimate_sine_args的逻辑进行频率和幅值估计
    # 但为了效率，我们手动实现核心逻辑
    nyquist_freq = sampling_rate / 2.0

    # 确定搜索频率范围
    if approx_freq is not None:
        freq_min_candidate = approx_freq * (1 - error_percentage / 100.0)
        freq_min = max(freq_min_candidate, 0.0)
        freq_max_candidate = approx_freq / (1 - error_percentage / 100.0)
        freq_max = min(freq_max_candidate, nyquist_freq)
    else:
        freq_min = 0.0
        freq_max = nyquist_freq

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

    # ==================== 第二阶段：向量化相位估计（所有通道）====================
    # 使用线性最小二乘对所有通道同时进行相位估计
    # 构建设计矩阵
    cycles_for_phase_estimation = 3
    samples_per_cycle = float(sampling_rate) / estimated_frequency
    phase_estimation_samples = int(cycles_for_phase_estimation * samples_per_cycle)
    phase_estimation_samples = min(phase_estimation_samples, samples_num)

    time_array_for_phase = np.arange(phase_estimation_samples) / sampling_rate
    cos_term = np.cos(2 * np.pi * estimated_frequency * time_array_for_phase)
    sin_term = np.sin(2 * np.pi * estimated_frequency * time_array_for_phase)
    design_matrix = np.column_stack([cos_term, sin_term])

    # 截取所有通道的前phase_estimation_samples个样本 (n_channels, phase_estimation_samples)
    waveform_data_for_phase = waveform_data[:, :phase_estimation_samples]

    # 向量化最小二乘求解：对每个通道求解 [a, b] 使得 y ≈ a*cos + b*sin
    # 使用numpy的lstsq，但需要对每个通道单独调用
    # 更高效的方式是手动求解正规方程: (X^T X)^(-1) X^T y

    # 计算 (X^T X)^(-1) X^T (2, phase_estimation_samples)
    XtX = design_matrix.T @ design_matrix
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        # 如果矩阵奇异，使用伪逆
        XtX_inv = np.linalg.pinv(XtX)
    XtX_inv_Xt = XtX_inv @ design_matrix.T  # shape: (2, phase_estimation_samples)

    # 对所有通道同时计算系数: (n_channels, 2) = (n_channels, n_samples) @ (n_samples, 2)
    coefficients = waveform_data_for_phase @ XtX_inv_Xt.T  # shape: (n_channels, 2)

    # 提取a和b系数
    a_coeffs = coefficients[:, 0]  # cos系数
    b_coeffs = coefficients[:, 1]  # sin系数

    # 计算相位
    estimated_phases = np.arctan2(a_coeffs, b_coeffs)

    # 计算幅值（使用RMS方法）
    # 对于正弦波，幅值 = sqrt(2) * RMS
    rms_values = np.sqrt(np.mean(waveform_data**2, axis=1))
    estimated_amplitudes = rms_values * np.sqrt(2)

    f_logger.debug(
        f"向量化相位/幅值估计完成: 幅值范围[{estimated_amplitudes.min():.6f}, "
        f"{estimated_amplitudes.max():.6f}], 相位范围[{estimated_phases.min():.6f}, "
        f"{estimated_phases.max():.6f}]"
    )

    # ==================== 第三阶段：可选的curve_fit优化 ====================
    if use_curve_fit:
        f_logger.debug("开始curve_fit优化...")
        time_array = np.arange(samples_num) / sampling_rate

        # 多通道情况下，所有通道共享相同频率
        # 策略：使用第一个通道优化频率、幅值、相位，然后将优化后的频率应用到所有通道
        # 再对每个通道单独优化幅值和相位（固定频率）

        # 初始化优化结果数组
        optimized_amplitudes = np.zeros(n_channels)
        optimized_phases = np.zeros(n_channels)

        # 第一步：使用第一个通道优化所有三个参数
        def sine_model_full(
            t: np.ndarray, amplitude: float, frequency: float, phase: float
        ) -> np.ndarray:
            """完整正弦波模型: y = amplitude * sin(2π * f * t + phase)"""
            return amplitude * np.sin(2 * np.pi * frequency * t + phase)

        first_channel_data = waveform_data[0, :]
        initial_params_full = [
            estimated_amplitudes[0],
            estimated_frequency,
            estimated_phases[0],
        ]

        try:
            optimal_params_full, _ = curve_fit(
                sine_model_full,
                time_array,
                first_channel_data,
                p0=initial_params_full,
                bounds=([0.0, freq_min, -np.pi], [np.inf, freq_max, np.pi]),
                maxfev=5000,
            )
            optimized_frequency = float(optimal_params_full[1])
            optimized_amplitudes[0] = float(optimal_params_full[0])
            optimized_phases[0] = float(optimal_params_full[2])
            f_logger.debug(f"频率优化结果: {optimized_frequency:.6f}Hz")
        except Exception as e:
            f_logger.warning(f"频率优化失败: {e}，使用粗略估计的频率")
            optimized_frequency = estimated_frequency
            optimized_amplitudes[0] = estimated_amplitudes[0]
            optimized_phases[0] = estimated_phases[0]

        # 第二步：使用优化后的频率，对每个通道优化幅值和相位
        def sine_model_fixed_freq(
            t: np.ndarray, amplitude: float, phase: float
        ) -> np.ndarray:
            """正弦波模型（固定频率）: y = amplitude * sin(2π * f * t + phase)"""
            return amplitude * np.sin(2 * np.pi * optimized_frequency * t + phase)

        # 优化剩余通道
        for ch_idx in range(1, n_channels):
            channel_data = waveform_data[ch_idx, :]
            initial_params = [estimated_amplitudes[ch_idx], estimated_phases[ch_idx]]

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
                f_logger.warning(f"通道{ch_idx}的curve_fit优化失败: {e}，使用粗略估计")
                optimized_amplitudes[ch_idx] = estimated_amplitudes[ch_idx]
                optimized_phases[ch_idx] = estimated_phases[ch_idx]

        # 使用优化后的值
        estimated_amplitudes = optimized_amplitudes
        estimated_phases = optimized_phases
        f_logger.debug("curve_fit优化完成")

    # ==================== 构建复振幅结果 ====================
    # 复振幅 = 幅值 * exp(1j * 相位)
    complex_amplitudes = estimated_amplitudes * np.exp(1j * estimated_phases)

    f_logger.debug(
        f"多通道单频信息提取完成: 输出形状={complex_amplitudes.shape}, "
        f"复振幅幅值范围=[{np.abs(complex_amplitudes).min():.6f}, "
        f"{np.abs(complex_amplitudes).max():.6f}]"
    )

    return complex_amplitudes
