"""
# 数据滤波模块

模块路径：`sweeper400.analyze.filter`

包含对采集数据进行滤波处理的函数和类。
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, detrend, sosfilt

from sweeper400.logger import get_logger  # noqa

from .my_dtypes import PointRawData, PositiveFloat, PositiveInt, SweepData, Waveform

# 获取模块日志器
logger = get_logger(__name__)

# 定义滤波器类型（SOS格式的ndarray）
type FilterSOS = NDArray[np.float64]  # shape: (n_sections, 6)


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
        >>> detrended = detrend_waveform(waveform)  # noqa
        >>> # 仅去除直流偏移（均值）
        >>> detrended = detrend_waveform(  # noqa
        ...     waveform, detrend_type="constant"  # noqa
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


def filter_waveform(
    input_waveform: Waveform,
    sos: FilterSOS,
) -> Waveform:
    """
    对波形应用已设计好的滤波器

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
        >>> # 对多个信号应用同一个滤波器
        >>> filtered1 = filter_waveform(signal1, sos)  # noqa
        >>> filtered2 = filter_waveform(signal2, sos)  # noqa
        >>> filtered3 = filter_waveform(signal3, sos)  # noqa
        ```

    Notes:
        - 对于多通道数据，会对每个通道分别进行滤波
        - 保留输入波形的所有元数据（采样率、时间戳、ID等）
    """
    # 获取函数日志器
    func_logger = get_logger(f"{__name__}.filter_waveform")

    func_logger.debug(
        f"应用滤波器: waveform_shape={input_waveform.shape}, "
        f"sampling_rate={input_waveform.sampling_rate}Hz"
    )

    # 应用滤波
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


def filter_sweep_data(
    sweep_data: SweepData,
    lowcut: PositiveFloat = 100.0,
    highcut: PositiveFloat = 20000.0,
    filter_order: PositiveInt = 4,
    trim_samples: int = 0,
) -> SweepData:
    """
    对SweepData中的所有波形应用带通滤波器

    该函数对SweepData中的所有AI波形和AO波形应用带通滤波器。
    滤波器只设计一次，然后复用到所有波形上，提高处理效率。

    Args:
        sweep_data: 原始的扫场测量数据
        lowcut: 低通截止频率（Hz），必须为正实数，默认为100.0Hz
        highcut: 高通截止频率（Hz），必须为正实数，默认为20000.0Hz
        filter_order: 高通滤波器阶数，默认为4
        trim_samples: 滤波后切除波形开头的采样点数量，用于消除边缘效应，默认为0

    Returns:
        滤波后的SweepData，结构与输入完全相同，但所有波形已应用高通滤波

    Raises:
        ValueError: 当输入数据为空时

    Examples:
        ```python
        >>> # 假设已有原始采集数据
        >>> raw_data = sweeper.get_data()  # noqa
        >>> # 应用高通滤波器（默认10Hz截止频率）
        >>> filtered_data = filter_sweep_data(raw_data)
        >>> # 或自定义滤波器参数并切除开头100个采样点
        >>> filtered_data = filter_sweep_data(  # noqa
        ...     raw_data,
        ...     lowcut=100.0,
        ...     highcut=20000.0,
        ...     filter_order=3,
        ...     trim_samples=100
        ... )
        >>> # 滤波后的数据可用于后续处理
        >>> tf_results = calculate_transfer_function(  # noqa
        ...     filtered_data, apply_highpass_filter=False)
        ```

    Notes:
        - 滤波器在所有波形间复用，避免重复设计的计算开销
        - 滤波后的数据结构与原始数据完全相同，可直接用于后续处理
        - trim_samples参数用于消除滤波器的边缘效应，当为0时不进行切除
    """
    # 获取函数日志器
    func_logger = get_logger(f"{__name__}.filter_sweep_data")

    ai_data_list = sweep_data["ai_data_list"]
    ao_data = sweep_data["ao_data"]

    # 验证输入数据
    if not ai_data_list:
        func_logger.error("输入数据列表为空")
        raise ValueError("输入数据列表不能为空")

    func_logger.info(
        f"开始对SweepData应用带通滤波器: "
        f"通带下限={lowcut}Hz, 通带上限={highcut}Hz, "
        f"阶数={filter_order}, "
        f"切除采样点数={trim_samples}, "
    )

    # 获取采样率（使用第一个有效波形的采样率）
    sampling_rate = ai_data_list[0]["ai_data"][0].sampling_rate
    func_logger.debug(f"采样率: {sampling_rate}Hz")

    sos: FilterSOS = butter(
        N=filter_order,
        Wn=[lowcut, highcut],
        btype="bandpass",
        analog=False,
        output="sos",
        fs=sampling_rate,
    )
    func_logger.debug("高通滤波器设计完成，将在所有波形复用")

    # 处理AI数据
    filtered_ai_data_list: list[PointRawData] = []
    processed_count = 0

    for point_idx, point_data in enumerate(ai_data_list):
        position = point_data["position"]
        ai_waveforms = point_data["ai_data"]

        func_logger.debug(
            f"处理点 {point_idx} @ ({position.x}, {position.y}), "
            f"共{len(ai_waveforms)}个波形"
        )

        # 对该点的所有AI波形应用滤波器
        filtered_ai_waveforms: list[Waveform] = []
        for _, waveform in enumerate(ai_waveforms):
            detrended_wf = detrend_waveform(waveform)
            filtered_wf = filter_waveform(detrended_wf, sos)

            # 如果需要切除开头的采样点
            if trim_samples > 0:
                # 切除开头的采样点
                trimmed_data = filtered_wf[..., trim_samples:]
                filtered_wf = Waveform(
                    input_array=trimmed_data,
                    sampling_rate=filtered_wf.sampling_rate,
                    timestamp=filtered_wf.timestamp,
                )

            filtered_ai_waveforms.append(filtered_wf)
            processed_count += 1

        # 创建滤波后的点数据
        filtered_point_data: PointRawData = {
            "position": position,
            "ai_data": filtered_ai_waveforms,
        }
        filtered_ai_data_list.append(filtered_point_data)

        func_logger.debug(f"点 {point_idx} 完成滤波，处理了{len(ai_waveforms)}个波形")

    func_logger.debug(f"所有AI波形滤波完成，共处理{processed_count}个波形")

    # 创建滤波后的SweepData
    filtered_sweep_data: SweepData = {
        "ai_data_list": filtered_ai_data_list,
        "ao_data": ao_data,
    }

    func_logger.info(
        f"SweepData滤波完成，共处理{processed_count}个波形 "
        f"({len(ai_data_list)}个测量点)"
    )

    return filtered_sweep_data
