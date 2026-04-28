"""
# 数据滤波模块

模块路径：`sweeper400.analyze.filter`

包含对采集数据进行滤波处理的函数和类。
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, detrend, sosfilt

from ..logger import get_logger
from .my_dtypes import PointSweepData, PositiveFloat, PositiveInt, SweepData, Waveform

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
        - 对于多通道数据，默认对每个通道独立进行去趋势处理
        - 使用scipy直接处理2D数据，性能优于手动循环
        - 保留输入波形的所有元数据（采样率、时间戳、波形ID等）
    """
    # 获取函数日志器
    f_logger = get_logger(f"{__name__}.detrend_waveform")

    f_logger.debug(
        f"去除波形基线偏移: waveform_shape={input_waveform.shape}, "
        f"sampling_rate={input_waveform.sampling_rate}Hz, "
        f"detrend_type={detrend_type}"
    )

    # 验证detrend_type参数
    if detrend_type not in ["linear", "constant"]:
        error_msg = (
            f"detrend_type参数必须为 'linear' 或 'constant'，得到: {detrend_type}"
        )
        f_logger.error(error_msg)
        raise ValueError(error_msg)

    # 应用去趋势处理
    try:
        # 直接使用scipy处理2D数据，自动沿指定轴操作
        # 2D数据形状 (n_channels, n_samples)，axis=-1 表示对最后一个轴，即时间轴去趋势
        detrended_data = detrend(input_waveform, axis=-1, type=detrend_type)
        f_logger.debug(f"完成{input_waveform.ndim}维波形去趋势处理")
    except Exception as e:
        error_msg = f"去趋势处理失败: {e}"
        f_logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e

    # 创建输出Waveform对象，保留原始元数据
    detrended_waveform = Waveform(
        input_array=detrended_data,
        sampling_rate=input_waveform.sampling_rate,
        channel_names=input_waveform.channel_names,
        timestamp=input_waveform.timestamp,
        waveform_id=input_waveform.waveform_id,
        sine_args=input_waveform.sine_args,
    )

    f_logger.debug(
        f"成功创建去趋势后的Waveform对象: {detrended_waveform}, "
        f"数据范围: [{np.min(detrended_data):.6f}, {np.max(detrended_data):.6f}]"
    )

    return detrended_waveform


def filter_waveform(
    input_waveform: Waveform,
    sos: FilterSOS,
    zi: NDArray[np.float64] | None = None,
) -> Waveform | tuple[Waveform, NDArray[np.float64]]:
    """
    对波形应用已设计好的滤波器（单向滤波）

    该函数可以重复使用同一个滤波器处理多个波形，提高效率。
    支持有状态滤波，通过zi参数传递滤波器状态，实现分段连续滤波。

    Args:
        input_waveform: 输入的时域波形数据（Waveform对象）
        sos: SOS格式的滤波器系数（由design_highpass_filter等函数生成）
        zi: 滤波器初始状态，可选。如果提供，则返回最终状态zf
            形状为 (n_sections, n_channels, 2)，其中n_sections是sos的行数

    Returns:
        如果zi为None: 仅返回滤波后的Waveform对象
        如果zi不为None: 返回元组 (filtered_waveform, zf)，其中zf是最终滤波器状态

    Raises:
        RuntimeError: 当滤波过程失败时抛出
        ValueError: 当zi形状不匹配时抛出

    Examples:
        ```python
        >>> # 无状态滤波（默认）
        >>> filtered = filter_waveform(signal, sos)  # noqa
        >>> # 有状态滤波（传递初始状态）
        >>> filtered, zf = filter_waveform(signal, sos, zi=zi_initial)  # noqa
        >>> # 使用最终状态继续滤波下一段数据
        >>> filtered_next, zf_next = filter_waveform(next_signal, sos, zi=zf)  # noqa
        ```

    Notes:
        - 对于多通道数据，默认对每个通道独立进行滤波
        - 使用scipy直接处理2D数据，性能优于手动循环
        - 有状态滤波可用于分段处理长数据流，保持滤波连续性
        - zi参数与scipy.signal.sosfilt的zi参数格式一致
        - 保留输入波形的所有元数据（采样率、时间戳、ID等）
    """
    # 获取函数日志器
    f_logger = get_logger(f"{__name__}.filter_waveform")

    f_logger.debug(
        f"应用滤波器: waveform_shape={input_waveform.shape}, "
        f"sampling_rate={input_waveform.sampling_rate}Hz, "
        f"zi={'provided' if zi is not None else 'None'}"
    )

    # 应用滤波
    try:
        # 直接使用scipy处理1D和2D数据，自动沿指定轴操作
        # 2D数据形状 (n_channels, n_samples)，axis=-1 表示对最后一个轴，即时间轴滤波
        # sosfilt 仅在 zi 不为 None 时才返回 (filtered_data, zf)，否则只返回 filtered_data
        if zi is not None:
            filtered_data, zf = sosfilt(sos, input_waveform, axis=-1, zi=zi)
        else:
            filtered_data = sosfilt(sos, input_waveform, axis=-1)
            zf = None
        f_logger.debug(f"完成{input_waveform.ndim}维波形滤波")

    except Exception as e:
        error_msg = f"滤波过程失败: {e}"
        f_logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e

    # 创建输出Waveform对象，保留原始元数据
    filtered_waveform = Waveform(
        input_array=filtered_data,
        sampling_rate=input_waveform.sampling_rate,
        channel_names=input_waveform.channel_names,
        timestamp=input_waveform.timestamp,
        waveform_id=input_waveform.waveform_id,
        sine_args=input_waveform.sine_args,
    )

    f_logger.debug(
        f"成功创建滤波后的Waveform对象: {filtered_waveform}, "
        f"数据范围: [{np.min(filtered_data):.6f}, {np.max(filtered_data):.6f}]"
    )

    if zi is not None:
        return filtered_waveform, zf  # noqa
    return filtered_waveform


def filter_sweep_data(
    sweep_data: SweepData,
    lowcut: PositiveFloat = 100.0,
    highcut: PositiveFloat = 20000.0,
    filter_order: PositiveInt = 4,
    trim_samples: int = 0,
    use_continuous_filtering: bool = True,
) -> SweepData:
    """
    对SweepData中的所有波形进行滤波

    该函数对SweepData中的所有AI波形进行去趋势、应用带通（单向）滤波、并按需裁剪信号头。
    滤波器只设计一次，然后复用到所有波形上，提高处理效率。
    支持连续滤波模式，通过传递滤波器状态(zi)在不同波形间保持滤波连续性，减少边缘效应。

    Args:
        sweep_data: 原始的扫场测量数据
        lowcut: 低通截止频率（Hz），必须为正实数，默认为100.0Hz
        highcut: 高通截止频率（Hz），必须为正实数，默认为20000.0Hz
        filter_order: 高通滤波器阶数，默认为4
        trim_samples: 滤波后切除波形开头的采样点数量，用于消除边缘效应，默认为0
            （一般来说，去趋势+单向滤波方案的边缘效应并不显著，无需切除）
        use_continuous_filtering: 是否使用连续滤波模式，默认为True
            - True: 在所有波形间传递滤波器状态(zi)，保持滤波连续性
            - False: 每个波形独立滤波（传统模式）

    Returns:
        滤波后的SweepData，结构与输入完全相同，但所有波形已被滤波

    Raises:
        ValueError: 当输入数据为空时

    Examples:
        ```python
        >>> # 假设已有原始采集数据
        >>> from use import SweeperCore
        >>> sweeper = SweeperCore()
        >>> raw_data = sweeper.export_data()
        >>> # 应用高通滤波器（默认10Hz截止频率）
        >>> filtered_data = filter_sweep_data(raw_data)
        >>> # 或自定义滤波器参数并切除开头100个采样点
        >>> filtered_data = filter_sweep_data(  # noqa
        ...     raw_data,
        ...     lowcut=1000.0,
        ...     highcut=10000.0,
        ...     filter_order=4,
        ...     trim_samples=500
        ... )
        >>> # 滤波后的数据可用于后续处理
        >>> plot_tf_results = calculate_transfer_function(filtered_data)  # noqa
        ```

    Notes:
        - 滤波器在所有波形间复用，避免重复设计的计算开销
        - 连续滤波模式通过传递zi状态，将多个波形视为一个连续信号进行滤波
        - 这可以显著减少波形间的边缘效应，提高滤波稳定性
        - 滤波后的数据结构与原始数据完全相同，可直接用于后续处理
        - trim_samples参数用于消除滤波器的边缘效应，当为0时不进行切除
    """
    # 获取函数日志器
    f_logger = get_logger(f"{__name__}.filter_sweep_data")

    ai_data_list = sweep_data["ai_data_list"]
    ao_data = sweep_data["ao_data"]

    f_logger.info(
        f"开始对SweepData应用带通滤波器: "
        f"通带下限={lowcut}Hz, 通带上限={highcut}Hz, "
        f"阶数={filter_order}, "
        f"切除采样点数={trim_samples}, "
        f"连续滤波模式={use_continuous_filtering}"
    )

    # 获取采样率（使用第一个有效波形的采样率）
    sampling_rate = ai_data_list[0]["ai_data"][0].sampling_rate
    f_logger.debug(f"采样率: {sampling_rate}Hz")

    sos: FilterSOS = butter(
        N=filter_order,
        Wn=[lowcut, highcut],
        btype="bandpass",
        analog=False,
        output="sos",
        fs=sampling_rate,
    )
    n_sections = sos.shape[0]
    f_logger.debug(f"带通滤波器设计完成，共{n_sections}个二阶节，将在所有波形复用")

    # 处理AI数据
    filtered_ai_data_list: list[PointSweepData] = []
    processed_count = 0

    # 初始化滤波器状态（用于连续滤波模式）
    # Waveform统一使用2D格式 (n_channels, n_samples)
    first_waveform = ai_data_list[0]["ai_data"][0]
    n_channels = first_waveform.channels_num  # 使用属性获取通道数
    zi: NDArray[np.float64] = np.zeros((n_sections, n_channels, 2), dtype=np.float64)

    f_logger.debug(f"初始化滤波器状态zi，形状: {zi.shape}")

    for point_idx, point_data in enumerate(ai_data_list):
        position = point_data["position"]
        ai_waveforms = point_data["ai_data"]

        f_logger.debug(
            f"处理点 {point_idx} @ ({position.x}, {position.y}), "
            f"共{len(ai_waveforms)}个波形"
        )

        # 对该点的所有AI波形应用滤波器
        filtered_ai_waveforms: list[Waveform] = []
        for waveform_idx, waveform in enumerate(ai_waveforms):
            # 去趋势处理
            detrended_wf = detrend_waveform(waveform)

            # 滤波处理（使用连续滤波模式或传统模式）
            if use_continuous_filtering:
                # 连续滤波模式：传递zi并接收新的zf
                filtered_wf, zf = filter_waveform(detrended_wf, sos, zi=zi)
                zi = zf  # 更新状态供下一个波形使用
                f_logger.debug(
                    f"  波形 {waveform_idx}: 使用连续滤波，传递zi状态"
                )
            else:
                # 传统模式：每个波形独立滤波
                filtered_wf= filter_waveform(detrended_wf, sos)
                f_logger.debug(f"  波形 {waveform_idx}: 独立滤波")

            # 如果需要切除开头的采样点
            if trim_samples > 0:
                # 切除开头的采样点
                filtered_wf = filtered_wf[..., trim_samples:]

            filtered_ai_waveforms.append(filtered_wf)  # noqa
            processed_count += 1

        # 创建滤波后的点数据
        filtered_point_data: PointSweepData = {
            "position": position,
            "ai_data": filtered_ai_waveforms,
        }
        filtered_ai_data_list.append(filtered_point_data)

        f_logger.debug(f"点 {point_idx} 完成滤波，处理了{len(ai_waveforms)}个波形")

    f_logger.debug(f"所有AI波形滤波完成，共处理{processed_count}个波形")

    # 创建滤波后的SweepData
    filtered_sweep_data: SweepData = {
        "ai_data_list": filtered_ai_data_list,
        "ao_data": ao_data,
    }

    f_logger.info(
        f"SweepData滤波完成，共处理{processed_count}个波形 "
        f"({len(ai_data_list)}个测量点)"
    )

    return filtered_sweep_data
