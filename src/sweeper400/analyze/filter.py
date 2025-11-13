"""
# 数据滤波模块

模块路径：`sweeper400.analyze.filter`

包含对采集数据进行滤波处理的函数和类。
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, detrend, sosfilt

from sweeper400.logger import get_logger  # type: ignore

from .my_dtypes import PositiveFloat, Waveform

# 获取模块日志器
logger = get_logger(__name__)

# 定义滤波器类型（SOS格式的ndarray）
type FilterSOS = NDArray[np.float64]  # shape: (n_sections, 6)


def get_highpass_filter(
    sampling_rate: PositiveFloat,
    cutoff_freq: PositiveFloat = 10.0,
    order: int = 4,
) -> FilterSOS:
    """
    设计Butterworth高通滤波器

    该函数设计一个Butterworth高通滤波器，
    返回Second-Order Sections (SOS)格式的滤波器系数。
    设计好的滤波器可以被重复使用于具有相同采样率的多个信号，避免重复设计的计算开销。

    Args:
        sampling_rate: 信号采样率（Hz），必须为正实数
        cutoff_freq: 截止频率（Hz），必须为正实数，默认为10.0Hz。
            低于此频率的成分将被衰减
        order: 滤波器阶数，默认为4。较低的阶数（3-5）通常足够，且能避免数值不稳定

    Returns:
        sos: SOS格式的滤波器系数，可用于apply_filter函数

    Raises:
        ValueError: 当截止频率大于等于奈奎斯特频率时抛出

    Examples:
        ```python
        >>> # 设计一个10Hz截止频率的高通滤波器（采样率4000Hz）
        >>> sos = get_highpass_filter(sampling_rate=4000.0, cutoff_freq=10.0)
        >>> # 该滤波器可以重复用于多个信号
        >>> filtered1 = filter_waveform(signal1, sos)  # type: ignore
        >>> filtered2 = filter_waveform(signal2, sos)  # type: ignore
        ```

    Notes:
        - 使用Butterworth滤波器是因为其在通带内具有最大平坦的幅频响应
        - 使用SOS格式（output='sos'）避免高阶滤波器的数值不稳定性
        - 对于相同采样率和截止频率的信号，可以复用同一个滤波器
    """
    # 获取函数日志器
    func_logger = get_logger(f"{__name__}.get_highpass_filter")

    func_logger.debug(
        f"设计高通滤波器: sampling_rate={sampling_rate}Hz, "
        f"cutoff_freq={cutoff_freq}Hz, order={order}"
    )

    # 计算奈奎斯特频率
    nyquist_freq = sampling_rate / 2.0

    # 验证截止频率
    if cutoff_freq >= nyquist_freq:
        error_msg = (
            f"截止频率 ({cutoff_freq}Hz) 必须小于奈奎斯特频率 ({nyquist_freq}Hz)"
        )
        func_logger.error(error_msg)
        raise ValueError(error_msg)

    # 计算归一化截止频率（相对于奈奎斯特频率）
    normalized_cutoff = cutoff_freq / nyquist_freq

    func_logger.debug(
        f"归一化截止频率: {normalized_cutoff:.6f} "
        f"(cutoff={cutoff_freq}Hz / nyquist={nyquist_freq}Hz)"
    )

    # 设计Butterworth高通滤波器（使用SOS格式）
    try:
        sos: FilterSOS = butter(
            N=order,
            Wn=normalized_cutoff,
            btype="highpass",
            analog=False,
            output="sos",
        )
        func_logger.debug(f"成功设计{order}阶Butterworth高通滤波器（SOS格式）")
    except Exception as e:
        error_msg = f"滤波器设计失败: {e}"
        func_logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg) from e

    return sos


def filter_waveform(
    input_waveform: Waveform,
    sos: FilterSOS,
) -> Waveform:
    """
    对波形应用已设计好的滤波器

    使用零相位滤波（sosfiltfilt）对输入波形应用滤波器，确保输出信号无相位失真。
    该函数可以重复使用同一个滤波器处理多个波形，提高效率。

    Args:
        input_waveform: 输入的时域波形数据（Waveform对象）
        sos: SOS格式的滤波器系数（由design_highpass_filter等函数生成）

    Returns:
        filtered_waveform: 滤波后的波形数据（Waveform对象），保留原始元数据

    Raises:
        RuntimeError: 当滤波过程失败时抛出

    Examples:
        ```python
        >>> # 先设计滤波器
        >>> sos = get_highpass_filter(sampling_rate=4000.0, cutoff_freq=10.0)
        >>> # 对多个信号应用同一个滤波器
        >>> filtered1 = filter_waveform(signal1, sos)  # type: ignore
        >>> filtered2 = filter_waveform(signal2, sos)  # type: ignore
        >>> filtered3 = filter_waveform(signal3, sos)  # type: ignore
        ```

    Notes:
        - 使用sosfiltfilt进行零相位滤波，确保输出信号无相位失真
        - 对于多通道数据，会对每个通道分别进行滤波
        - 保留输入波形的所有元数据（采样率、时间戳、ID等）
    """
    # 获取函数日志器
    func_logger = get_logger(f"{__name__}.filter_waveform")

    func_logger.debug(
        f"应用滤波器: waveform_shape={input_waveform.shape}, "
        f"sampling_rate={input_waveform.sampling_rate}Hz"
    )

    # 应用零相位滤波
    try:
        # 处理单通道和多通道数据
        if input_waveform.ndim == 1:
            # 单通道数据
            filtered_data = sosfilt(sos, input_waveform)
            func_logger.debug("完成单通道数据滤波")
        elif input_waveform.ndim == 2:
            # 多通道数据，对每个通道分别滤波
            filtered_data = np.zeros_like(input_waveform)
            for i in range(input_waveform.shape[0]):
                filtered_data[i, :] = sosfilt(sos, input_waveform[i, :])
            func_logger.debug(f"完成{input_waveform.shape[0]}通道数据滤波")
        else:
            error_msg = f"不支持的数据维度: {input_waveform.ndim}（仅支持1D或2D）"
            func_logger.error(error_msg)
            raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"滤波过程失败: {e}"
        func_logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e

    # 创建输出Waveform对象，保留原始元数据
    filtered_waveform = Waveform(
        input_array=filtered_data,
        sampling_rate=input_waveform.sampling_rate,
        timestamp=input_waveform.timestamp,
        id=input_waveform.id,
        sine_args=input_waveform.sine_args,
    )

    func_logger.debug(
        f"成功创建滤波后的Waveform对象: {filtered_waveform}, "
        f"数据范围: [{np.min(filtered_data):.6f}, {np.max(filtered_data):.6f}]"
    )

    return filtered_waveform


def detrend_waveform(
    input_waveform: Waveform,
    detrend_type: str = "linear",
) -> Waveform:
    """
    去除波形的基线偏移

    使用scipy.signal.detrend函数去除波形数据的基线偏移（趋势项）。
    支持线性去趋势和常数去趋势两种模式。

    Args:
        input_waveform: 输入的时域波形数据（Waveform对象）
        detrend_type: 去趋势类型，可选值为 "linear"（线性去趋势）
            或 "constant"（常数去趋势，即去除均值），默认为 "linear"

    Returns:
        detrended_waveform: 去除基线偏移后的波形数据（Waveform对象），保留原始元数据

    Raises:
        ValueError: 当detrend_type参数不是 "linear" 或 "constant" 时抛出
        RuntimeError: 当去趋势过程失败时抛出

    Examples:
        ```python
        >>> # 去除线性趋势（默认）
        >>> detrended = detrend_waveform(waveform)  # type: ignore
        >>> # 仅去除直流偏移（均值）
        >>> detrended = detrend_waveform(  # type: ignore
        ...     waveform, detrend_type="constant"  # type: ignore
        ... )
        ```

    Notes:
        - "linear" 模式会拟合并去除线性趋势（斜率+偏移）
        - "constant" 模式仅去除信号的均值（直流偏移）
        - 对于多通道数据，会对每个通道分别进行去趋势处理
        - 保留输入波形的所有元数据（采样率、时间戳、ID等）
    """
    # 获取函数日志器
    func_logger = get_logger(f"{__name__}.detrend_waveform")

    func_logger.debug(
        f"去除波形基线偏移: waveform_shape={input_waveform.shape}, "
        f"sampling_rate={input_waveform.sampling_rate}Hz, "
        f"detrend_type={detrend_type}"
    )

    # 验证detrend_type参数
    if detrend_type not in ["linear", "constant"]:
        error_msg = (
            f"detrend_type参数必须为 'linear' 或 'constant'，得到: {detrend_type}"
        )
        func_logger.error(error_msg)
        raise ValueError(error_msg)

    # 应用去趋势处理
    try:
        # 处理单通道和多通道数据
        if input_waveform.ndim == 1:
            # 单通道数据
            detrended_data = detrend(input_waveform, type=detrend_type)
            func_logger.debug("完成单通道数据去趋势处理")
        elif input_waveform.ndim == 2:
            # 多通道数据，对每个通道分别去趋势
            detrended_data = np.zeros_like(input_waveform)
            for i in range(input_waveform.shape[0]):
                detrended_data[i, :] = detrend(input_waveform[i, :], type=detrend_type)
            func_logger.debug(f"完成{input_waveform.shape[0]}通道数据去趋势处理")
        else:
            error_msg = f"不支持的数据维度: {input_waveform.ndim}（仅支持1D或2D）"
            func_logger.error(error_msg)
            raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"去趋势处理失败: {e}"
        func_logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e

    # 创建输出Waveform对象，保留原始元数据
    detrended_waveform = Waveform(
        input_array=detrended_data,
        sampling_rate=input_waveform.sampling_rate,
        timestamp=input_waveform.timestamp,
        id=input_waveform.id,
        sine_args=input_waveform.sine_args,
    )

    func_logger.debug(
        f"成功创建去趋势后的Waveform对象: {detrended_waveform}, "
        f"数据范围: [{np.min(detrended_data):.6f}, {np.max(detrended_data):.6f}]"
    )

    return detrended_waveform
