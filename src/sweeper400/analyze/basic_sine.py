"""
# 基础正弦波处理模块

模块路径：`sweeper400.analyze.basic_sine`

本模块包含与最简单的**单频正弦波**相关的函数和类。
（本模块函数较为复杂，故均配备函数日志器）
"""

import numpy as np
from scipy.linalg import lstsq  # type: ignore
from scipy.optimize import curve_fit  # type: ignore
from scipy.signal import periodogram  # type: ignore

from sweeper400.logger import get_logger  # type: ignore

from .my_dtypes import (
    PositiveFloat,
    SamplingInfo,
    SineArgs,
    Waveform,
    init_sine_args,
)

# 获取模块日志器
logger = get_logger(__name__)


def get_sine(
    sampling_info: SamplingInfo,
    sine_args: SineArgs,
    timestamp: np.datetime64 | None = None,
    id: int | None = None,
) -> Waveform:
    """
    使用几个简单的参数生成包含单频正弦波时域信号的Waveform对象

    Args:
        sampling_info: 采样信息，包含采样率和采样点数
        sine_args: 正弦波参数，包含频率、幅值和相位信息
        timestamp: 采样开始时间戳，默认值为None
        id: 波形的唯一标识符，默认值为None

    Returns:
        output_sine_wave: 包含单频正弦波的Waveform对象

    Examples:
        ```python
        >>> sampling_info = init_sampling_info(1000, 1024)  # noqa
        >>> sine_args = init_sine_args(50.0, 1.0, 0.0)
        >>> sine_wave = get_sine(sampling_info, sine_args)
        >>> print(sine_wave.shape)
        (1024,)
        >>> print(sine_wave.sampling_rate)
        1000.0
        ```
    """
    # 获取函数日志器
    logger = get_logger(f"{__name__}.get_sine")

    logger.debug(
        f"生成正弦波: frequency={sine_args['frequency']}Hz, "
        f"amplitude={sine_args['amplitude']}, phase={sine_args['phase']}rad, "
        f"sampling_rate={sampling_info['sampling_rate']}Hz, "
        f"samples_num={sampling_info['samples_num']}, "
        f"timestamp={timestamp}, id={id}"
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

    # 创建Waveform对象
    output_sine_wave = Waveform(
        input_array=sine_data,
        sampling_rate=sampling_info["sampling_rate"],
        timestamp=timestamp,
        id=id,
        sine_args=sine_args,
    )

    logger.debug(f"成功生成正弦波Waveform对象: {output_sine_wave}")

    return output_sine_wave


def get_sine_cycles(
    sampling_info: SamplingInfo,
    sine_args: SineArgs,
    timestamp: np.datetime64 | None = None,
    id: int | None = None,
) -> Waveform:
    """
    生成若干个完整周期的连续正弦波形Waveform

    该函数生成的波形始终从0相位开始并在0相位结束，确保输出波形可以无缝重复连接。
    输出波形的长度会根据输入参数自动调整为整数个周期。

    Args:
        sampling_info: 采样信息，包含采样率和采样点数
        sine_args: 正弦波参数，包含频率、幅值和相位信息（注意：相位参数会被忽略）
        timestamp: 采样开始时间戳，默认值为None
        id: 波形的唯一标识符，默认值为None

    Returns:
        output_sine_wave: 包含整数个周期正弦波的Waveform对象

    Raises:
        ValueError: 当采样率不是频率的整数倍时抛出

    Examples:
        ```python
        >>> # 生成1000Hz采样率下50Hz的正弦波，包含完整周期
        >>> sampling_info = init_sampling_info(1000, 1024)  # type: ignore
        >>> sine_args = init_sine_args(50.0, 1.0, 0.0)  # 相位参数会被忽略
        >>> sine_wave = get_sine_cycles(sampling_info, sine_args)
        >>> print(f"实际采样点数: {sine_wave.shape[0]}")
        >>> print(f"周期数: {sine_wave.shape[0] * 50.0 / 1000}")
        ```
    """
    # 获取函数日志器
    logger = get_logger(f"{__name__}.get_sine_cycles")

    logger.debug(
        f"生成整周期正弦波: frequency={sine_args['frequency']}Hz, "
        f"amplitude={sine_args['amplitude']}, "
        f"sampling_rate={sampling_info['sampling_rate']}Hz, "
        f"requested_samples={sampling_info['samples_num']}, "
        f"timestamp={timestamp}, id={id}"
    )

    # 1. 检查采样率是否是频率的整数倍
    frequency = sine_args["frequency"]
    sampling_rate = sampling_info["sampling_rate"]

    # 计算一个周期的采样点数
    samples_per_cycle = sampling_rate / frequency

    # 检查是否为整数（允许小的浮点误差）
    if abs(samples_per_cycle - round(samples_per_cycle)) > 1e-10:
        logger.error(
            f"采样率 {sampling_rate}Hz 不是频率 {frequency}Hz 的整数倍。"
            f"一个周期需要 {samples_per_cycle:.6f} 个采样点，不是整数。",
            exc_info=True,
        )
        raise ValueError(
            f"采样率 {sampling_rate}Hz 不是频率 {frequency}Hz 的整数倍。"
            f"一个周期需要 {samples_per_cycle:.6f} 个采样点，不是整数。"
        )

    samples_per_cycle_int = int(round(samples_per_cycle))
    logger.debug(f"每个周期采样点数: {samples_per_cycle_int}")

    # 2. 根据输入的samples_num计算总周期数（向上取整）
    requested_samples = sampling_info["samples_num"]
    total_cycles = np.ceil(requested_samples / samples_per_cycle_int)
    total_cycles_int = int(total_cycles)

    # 计算实际的采样点数（整数个周期）
    actual_samples = total_cycles_int * samples_per_cycle_int

    logger.debug(
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

    # 创建Waveform对象
    output_sine_wave = Waveform(
        input_array=sine_data,
        sampling_rate=sampling_rate,
        timestamp=timestamp,
        id=id,
        sine_args=actual_sine_args,
    )

    logger.debug(
        f"成功生成整周期正弦波Waveform对象: {output_sine_wave}, "
        f"周期数: {total_cycles_int}"
    )

    return output_sine_wave


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
        >>> sampling_info = init_sampling_info(1000, 1024)  # type: ignore
        >>> sine_args = init_sine_args(50.0, 2.0, 0.5)
        >>> test_wave = get_sine(sampling_info, sine_args)
        >>> estimated_args = estimate_sine_args(test_wave, approx_freq=50.0)
        >>> print(f"估计频率: {estimated_args['frequency']:.2f}Hz")
        ```
    """
    # 获取函数日志器
    logger = get_logger(f"{__name__}.estimate_sine_args")

    logger.debug(
        f"开始粗略正弦波参数估计: approx_freq={approx_freq}Hz, "
        f"error_percentage={error_percentage}%, "
        f"waveform_shape={input_waveform.shape}, "
        f"sampling_rate={input_waveform.sampling_rate}Hz"
    )

    # 处理多通道数据，只使用第一个通道
    if input_waveform.ndim == 2:
        waveform_data = input_waveform[0, :]
        logger.debug("检测到多通道数据，使用第一个通道进行分析")
    else:
        waveform_data = input_waveform

    # 获取基本参数
    sampling_rate = float(input_waveform.sampling_rate)
    samples_num = int(input_waveform.samples_num)
    nyquist_freq = sampling_rate / 2.0

    # 确定搜索频率范围
    if approx_freq is None:
        # 全频率范围搜索
        freq_min = 0.0
        freq_max = nyquist_freq
        logger.debug(f"全频率范围搜索: 0Hz - {nyquist_freq:.2f}Hz")
    else:
        # 根据误差百分比确定搜索范围
        freq_min = float(approx_freq) * (1 - float(error_percentage) / 100.0)
        freq_max_candidate = float(approx_freq) / (1 - float(error_percentage) / 100.0)
        freq_max = min(freq_max_candidate, nyquist_freq)

        logger.debug(
            f"限定频率范围搜索: {freq_min:.2f}Hz - {freq_max:.2f}Hz "
            f"(中心频率: {approx_freq:.2f}Hz, 误差: ±{error_percentage:.1f}%)"
        )

    # 使用Periodogram方法计算功率谱
    # window: 使用Hann窗以提高频率分辨率和降低频谱泄漏
    # scaling: 'spectrum' 返回功率谱（单位V^2），峰值的平方根是RMS幅度
    # return_onesided: 只返回正频率部分
    psd_freqs, psd_values = periodogram(  # noqa
        waveform_data,
        fs=sampling_rate,
        window="hann",
        detrend=False,  # 不去趋势，相关操作交给filter完成
        return_onesided=True,  # 只返回正频率部分
        scaling="spectrum",  # 使用功率谱
    )

    logger.debug(
        f"Periodogram方法参数: window='hann', scaling='spectrum', "
        f"频率分辨率={psd_freqs[1] - psd_freqs[0]:.6f}Hz"
    )

    # 将功率谱转换为幅度谱
    # 对于功率谱（scaling='spectrum'），峰值 = (RMS幅度)^2
    # 对于正弦波，峰值幅度 = RMS幅度 * sqrt(2)
    # 因此，峰值幅度 = sqrt(功率谱峰值) * sqrt(2)
    psd_magnitude = np.sqrt(psd_values) * np.sqrt(2.0)

    # 在指定频率范围内搜索峰值
    freq_range_mask = (psd_freqs >= freq_min) & (psd_freqs <= freq_max)
    if not np.any(freq_range_mask):
        logger.error(
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
        # 检查分母是否为零，避免除零错误
        denominator = float(y1 - 2 * y2 + y3)
        if abs(denominator) > 1e-12:  # 避免数值不稳定
            delta = 0.5 * float(y1 - y3) / denominator
            # 使用频率分辨率计算精确频率
            freq_resolution = float(psd_freqs[1] - psd_freqs[0])
            initial_frequency = float(coarse_frequency) + delta * freq_resolution
            # 计算精确幅值
            refined_magnitude = float(y2) - 0.25 * float(y1 - y3) * delta
            initial_amplitude = refined_magnitude  # 已经是峰值幅度

            logger.debug(
                f"抛物线拟合改进: 频率={initial_frequency:.6f}Hz, "
                f"幅值={initial_amplitude:.6f}, delta={delta:.6f}"
            )
        else:
            # 分母接近零，使用粗略估计
            initial_frequency = coarse_frequency
            initial_amplitude = coarse_magnitude  # 已经是峰值幅度

            logger.debug(
                f"抛物线拟合分母接近零，使用粗略估计: "
                f"频率={initial_frequency:.6f}Hz, 幅值={initial_amplitude:.6f}"
            )
    else:
        initial_frequency = coarse_frequency
        initial_amplitude = coarse_magnitude  # 已经是峰值幅度

        logger.debug(
            f"边界情况，使用粗略估计: "
            f"频率={initial_frequency:.6f}Hz, 幅值={initial_amplitude:.6f}"
        )

    logger.debug(
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

    logger.debug(
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

    result_init = lstsq(  # type: ignore
        design_matrix_init, waveform_data_for_phase, lapack_driver="gelsd"
    )
    # 断言确保lstsq返回了有效结果，帮助类型检查器
    assert result_init is not None, "lstsq应该总是返回一个元组"

    coefficients_init = result_init[0]  # type: ignore
    residuals_init = result_init[1] if len(result_init) > 1 else None  # type: ignore
    rank_init = result_init[2] if len(result_init) > 2 else None  # type: ignore
    s_init = result_init[3] if len(result_init) > 3 else None  # type: ignore

    # 确保coefficients_init是一个包含两个元素的数组
    assert len(coefficients_init) >= 2, "coefficients_init应该至少包含两个系数"  # type: ignore
    a_init, b_init = float(coefficients_init[0]), float(coefficients_init[1])  # type: ignore
    initial_phase = float(np.arctan2(a_init, b_init))

    # 安全地处理可能为None的返回值
    condition_number_str = "N/A"
    if s_init is not None and len(s_init) > 0:  # type: ignore
        condition_number_str = f"{s_init[0] / s_init[-1]:.2e}"

    residuals_str = "N/A"
    if residuals_init is not None:
        # residuals_init 可能是标量或数组
        if np.isscalar(residuals_init):
            residuals_str = f"{residuals_init:.6e}"
        elif len(residuals_init) > 0:
            residuals_str = f"{residuals_init[0]:.6e}"

    logger.debug(
        f"相位线性拟合结果: a={a_init:.6f}, b={b_init:.6f}, "
        f"矩阵秩={rank_init}, 条件数={condition_number_str}"
    )
    logger.debug(f"相位拟合残差平方和: {residuals_str}")

    # 验证初始估计的质量
    logger.debug(
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
        >>> sampling_info = init_sampling_info(1000, 1024)  # type: ignore
        >>> sine_args = init_sine_args(50.0, 2.0, 0.0)
        >>> test_wave = get_sine(sampling_info, sine_args)
        >>> detected_sine_args = extract_single_tone_information_vvi(
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
    logger = get_logger(f"{__name__}.extract_single_tone_information_vvi")

    logger.debug(
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

    logger.debug(
        f"粗略估计结果: 频率={initial_frequency:.6f}Hz, "
        f"幅值={initial_amplitude:.6f}, 相位={initial_phase:.6f}rad"
    )

    # 处理多通道数据，只使用第一个通道
    if input_waveform.ndim == 2:
        waveform_data = input_waveform[0, :]
        logger.debug("检测到多通道数据，使用第一个通道进行分析")
    else:
        waveform_data = input_waveform

    samples_num = input_waveform.samples_num
    sampling_rate = input_waveform.sampling_rate

    # 构建完整的时间序列用于curve_fit
    time_array = np.arange(samples_num) / sampling_rate

    # 计算频率搜索范围
    if approx_freq is not None:
        freq_min = approx_freq * (1 - error_percentage / 100)
        freq_max = approx_freq * (1 + error_percentage / 100)
    else:
        freq_min = 0.0
        freq_max = sampling_rate / 2

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
        optimal_params, _ = curve_fit(  # type: ignore
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
            time_array, detected_amplitude, detected_frequency, detected_phase
        )
        residuals = waveform_data - fitted_signal
        rms_error = np.sqrt(np.mean(residuals**2))

        logger.debug(
            f"curve_fit优化结果: 幅值={detected_amplitude:.6f}, "
            f"频率={detected_frequency:.6f}, 相位={detected_phase:.6f}"
        )
        logger.debug(f"拟合RMS误差: {rms_error:.6f}")

    except Exception as e:
        logger.warning(f"curve_fit优化失败: {e}")
        logger.warning("回退到粗略估计")

        detected_frequency = initial_frequency
        detected_amplitude = initial_amplitude
        detected_phase = initial_phase

    logger.debug(
        f"精确估计环节结果: 频率={detected_frequency:.6f}Hz, "
        f"幅值={detected_amplitude:.6f}, 相位={detected_phase:.6f}rad"
    )

    detected_sine_args = init_sine_args(
        detected_frequency, detected_amplitude, detected_phase
    )

    return detected_sine_args
