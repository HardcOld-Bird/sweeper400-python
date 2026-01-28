"""
# 后处理模块

模块路径：`sweeper400.analyze.post_process`

本模块提供对Sweeper类采集到的原始数据进行后处理的功能。
主要包含按位平均、传递函数计算等数据分析功能。
"""

import numpy as np

from ..logger import get_logger
from .basic_sine import extract_single_tone_information_vvi
from .my_dtypes import (
    CompData,
    PointCompData,
    PointSweepData,
    PointTFData,
    SweepData,
    TFData,
    Waveform,
)

# 获取模块日志器
logger = get_logger(__name__)


def average_sweep_data(
    sweep_data: SweepData,
) -> SweepData:
    """
    对SweepData中的所有波形进行按位相加并取平均

    该函数对SweepData中的所有AI波形进行按位相加并取平均，
    以减少随机噪声的影响。

    Args:
        sweep_data: 原始的扫场测量数据

    Returns:
        处理后的SweepData，结构与输入完全相同，但每个点只有一条AI波形

    Raises:
        ValueError: 当输入数据为空时
    """

    # 获取必要参数
    ai_data_list = sweep_data["ai_data_list"]
    sampling_rate = ai_data_list[0]["ai_data"][0].sampling_rate
    samples_num = ai_data_list[0]["ai_data"][0].samples_num

    averaged_ai_data_list = []
    # 遍历每个测量点
    for _, point_data in enumerate(ai_data_list):
        # 将所有波形数据按位相加
        ai_waveforms = point_data["ai_data"]
        summed_data = np.zeros(samples_num, dtype=np.float64)
        for wf in ai_waveforms:
            # 处理多通道数据，只使用第一个通道
            if wf.ndim == 2:
                summed_data += wf[0, :]
            else:
                summed_data += wf

        # 取平均
        averaged_data = summed_data / len(ai_waveforms)

        # 创建平均后的Waveform对象
        averaged_ai_waveform = Waveform(
            input_array=averaged_data,
            sampling_rate=sampling_rate,
            timestamp=ai_waveforms[0].timestamp,  # 使用第一个波形的时间戳
        )

        # 创建平均后的点数据
        averaged_point_data: PointSweepData = {
            "position": point_data["position"],
            "ai_data": [averaged_ai_waveform],
        }
        averaged_ai_data_list.append(averaged_point_data)

    # 创建平均后的SweepData
    averaged_sweep_data: SweepData = {
        "ai_data_list": averaged_ai_data_list,
        "ao_data": sweep_data["ao_data"],
    }

    return averaged_sweep_data


def calculate_transfer_function(
    sweep_data: SweepData,
) -> TFData:
    """
    计算Sweeper采集数据的传递函数

    对每个测量点的原始数据进行处理，计算输入输出信号的复数传递函数 H(ω) = A·e^(jφ)。
    具体步骤：
    1. 若存在多个AI波形chunks，进行按位相加并取平均
    2. 使用extract_single_tone_information_vvi提取AI信号的正弦波参数
    3. 使用共用的AO波形的正弦波参数
    4. 计算传递函数：幅值比 = AI幅值 / AO幅值，相位差 = AI相位 - AO相位
    5. 计算所有点的平均幅值比和平均相位差

    Args:
        sweep_data: Sweeper采集的完整测量数据，包含ai_data_list和ao_data

    Returns:
        TFData，包含以下字段：
            - tf_list: 传递函数数据列表，每个元素为PointTFData
            - frequency: 信号频率（Hz）
            - mean_amp_ratio: 所有点的平均幅值比
            - mean_phase_shift: 所有点的平均相位差（弧度制）

    Raises:
        ValueError: 当输入数据为空或格式不正确时
        RuntimeError: 当AO数据没有sine_args属性时

    Examples:
        ```python
        >>> # 假设已有采集的原始数据
        >>> sweep_data = sweeper.get_data()  # noqa
        >>> # 计算传递函数
        >>> tf_result = calculate_transfer_function(sweep_data)
        >>> for tf_data in tf_result["tf_list"]:
        ...     print(
        ...         f"位置: {tf_data['position']}, "
        ...         f"幅值比: {tf_data['amp_ratio']:.4f}, "
        ...         f"相位差: {tf_data['phase_shift']:.4f}rad"
        ...     )
        >>> print(f"频率: {tf_result['frequency']:.2f}Hz")
        >>> print(f"平均幅值比: {tf_result['mean_amp_ratio']:.6f}")
        ```
    """
    logger.info(f"开始计算传递函数，共 {len(sweep_data['ai_data_list'])} 个测量点。")

    # 处理AI数据：如果有多个波形，按位相加并取平均
    if len(sweep_data["ai_data_list"][0]["ai_data"]) > 1:
        logger.warning("检测到多个AI波形，将进行按位相加并取平均")
        sweep_data = average_sweep_data(sweep_data)

    # 验证AO数据的sine_args
    if sweep_data["ao_data"].sine_args is None:
        logger.error("AO波形没有sine_args属性")
        raise RuntimeError("AO波形必须包含sine_args属性")
    else:
        ao_sine_args = sweep_data["ao_data"].sine_args
        logger.debug(
            f"使用AO波形参数: 频率={ao_sine_args['frequency']:.2f}Hz, "
            f"幅值={ao_sine_args['amplitude']:.4f}"
        )

    # 获取原始数据列表
    ai_data_list = sweep_data["ai_data_list"]

    # 存储结果
    results: list[PointTFData] = []

    # 遍历每个测量点
    for point_idx, point_data in enumerate(ai_data_list):
        # 只在处理较少点数时或每10个点输出一次进度信息
        if len(ai_data_list) <= 20 or (point_idx + 1) % 10 == 0:
            logger.debug(
                f"处理第 {point_idx + 1}/{len(ai_data_list)} 个点: "
                f"{point_data['position']}"
            )

        try:
            ai_waveforms = point_data["ai_data"][0]

            # 1. 提取AI信号的正弦波参数
            ai_sine_args = extract_single_tone_information_vvi(ai_waveforms)

            # 2. 计算传递函数
            # 幅值比 = AI幅值 / AO幅值
            amp_ratio = ai_sine_args["amplitude"] / ao_sine_args["amplitude"]

            # 相位差 = AI相位 - AO相位（弧度制）
            phase_shift = ai_sine_args["phase"] - ao_sine_args["phase"]

            # 将相位差归一化到 [-π, π] 区间
            phase_shift = np.arctan2(np.sin(phase_shift), np.cos(phase_shift))

            # 3. 存储结果
            result: PointTFData = {
                "position": point_data["position"],
                "amp_ratio": float(amp_ratio),
                "phase_shift": float(phase_shift),
            }
            results.append(result)

        except Exception as e:
            logger.error(f"处理点 {point_idx} 时发生错误: {e}", exc_info=True)
            # 继续处理下一个点
            continue

    logger.info(f"传递函数计算完成，成功处理 {len(results)}/{len(ai_data_list)} 个点")

    # 计算平均幅值比和平均相位差
    amp_ratios = [tf["amp_ratio"] for tf in results]
    phase_shifts = [tf["phase_shift"] for tf in results]

    mean_amp_ratio = float(np.mean(amp_ratios))
    mean_phase_shift = float(np.mean(phase_shifts))

    logger.debug(
        f"平均幅值比: {mean_amp_ratio:.6f}, 平均相位差: {mean_phase_shift:.6f}rad"
    )

    # 返回包含结果列表和元数据的TFResult
    tf_result: TFData = {
        "tf_list": results,
        "frequency": ao_sine_args["frequency"],
        "mean_amp_ratio": mean_amp_ratio,
        "mean_phase_shift": mean_phase_shift,
    }

    return tf_result


def calculate_compensation_list(
    sweep_data: SweepData,
) -> CompData:
    """
    计算Sweeper采集数据的补偿参数列表

    对每个测量点的原始数据进行处理，计算相对于所有点平均值的补偿参数。
    具体步骤：
    1. 调用calculate_transfer_function获取所有点的传递函数及其元数据
    2. 对每个点计算相对于平均值的补偿参数：
       - 幅值补偿倍率 = 平均幅值比 / 该点幅值比
       - 时间延迟补偿 = (平均相位差 - 该点相位差) / (2π × 频率)

    Args:
        sweep_data: Sweeper采集的完整测量数据，包含ai_data_list和ao_data

    Returns:
        CompData，包含以下字段：
            - comp_list: 补偿参数数据列表，每个元素为PointCompData
            - frequency: 信号频率（Hz）
            - mean_amp_ratio: 所有点的平均幅值比
            - mean_phase_shift: 所有点的平均相位差（弧度制）

    Raises:
        ValueError: 当输入数据为空或格式不正确时
        RuntimeError: 当AO数据没有sine_args属性时

    Examples:
        ```python
        >>> # 假设已有采集的原始数据
        >>> sweep_data = sweeper.get_data()  # noqa
        >>> # 计算补偿参数
        >>> comp_result = calculate_compensation_list(sweep_data)
        >>> for comp_data in comp_result["comp_list"]:
        ...     print(
        ...         f"位置: {comp_data['position']}, "
        ...         f"幅值补偿倍率: {comp_data['amp_ratio']:.4f}, "
        ...         f"时间延迟补偿: {comp_data['time_delay']*1e6:.3f}μs"
        ...     )
        >>> print(f"频率: {comp_result['frequency']:.2f}Hz")
        ```
    """
    logger.info(f"开始计算补偿参数，共 {len(sweep_data['ai_data_list'])} 个测量点。")

    # 1. 首先计算所有点的传递函数
    tf_result = calculate_transfer_function(sweep_data)

    tf_list = tf_result["tf_list"]
    frequency = tf_result["frequency"]
    mean_amp_ratio = tf_result["mean_amp_ratio"]
    mean_phase_shift = tf_result["mean_phase_shift"]

    if not tf_list:
        logger.error("传递函数列表为空，无法计算补偿参数")
        raise ValueError("传递函数列表为空，无法计算补偿参数")

    logger.debug(f"使用频率: {frequency:.2f}Hz")

    # 2. 使用tf_to_comp工具函数计算每个点的补偿参数
    results: list[PointCompData] = []

    for tf_data in tf_list:
        try:
            # 使用工具函数进行转换
            comp_data = tf_to_comp(
                tf_data=tf_data,
                frequency=frequency,
                mean_amp_ratio=mean_amp_ratio,
                mean_phase_shift=mean_phase_shift,
            )
            results.append(comp_data)

        except Exception as e:
            logger.error(f"处理点 {tf_data['position']} 时发生错误: {e}", exc_info=True)
            # 继续处理下一个点
            continue

    logger.info(f"补偿参数计算完成，成功处理 {len(results)}/{len(tf_list)} 个点")

    # 返回包含结果列表和元数据的CompResult
    comp_result: CompData = {
        "comp_list": results,
        "frequency": frequency,
        "mean_amp_ratio": mean_amp_ratio,
        "mean_phase_shift": mean_phase_shift,
    }

    return comp_result


def tf_to_comp(
    tf_data: PointTFData,
    frequency: float,
    mean_amp_ratio: float,
    mean_phase_shift: float,
) -> PointCompData:
    """
    将传递函数数据转换为补偿数据

    将PointTFData（绝对传递函数）转换为PointCompData（相对于平均值的补偿参数）。
    转换公式：
    - 幅值补偿倍率 = 平均幅值比 / 该点幅值比
    - 时间延迟补偿 = (平均相位差 - 该点相位差) / (2π × 频率)

    Args:
        tf_data: 传递函数数据（包含绝对幅值比和相位差）
        frequency: 信号频率（Hz），用于将相位差转换为时间延迟
        mean_amp_ratio: 所有点的平均幅值比
        mean_phase_shift: 所有点的平均相位差（弧度制）

    Returns:
        补偿数据，包含相对于平均值的幅值补偿倍率和时间延迟补偿值

    Raises:
        ValueError: 当频率为0或负数时
        ZeroDivisionError: 当该点幅值比为0时

    Examples:
        ```python
        >>> tf_data = {
        ...     "position": Point2D(x=0.0, y=1.0),
        ...     "amp_ratio": 0.9,
        ...     "phase_shift": 0.1,
        ... }
        >>> comp_data = tf_to_comp(tf_data, frequency=1000.0,
        ...                         mean_amp_ratio=0.95, mean_phase_shift=0.05)
        >>> print(comp_data)
        ```
    """
    if frequency <= 0:
        logger.error(f"频率必须为正数，收到: {frequency}")
        raise ValueError(f"频率必须为正数，收到: {frequency}")

    if tf_data["amp_ratio"] == 0:
        logger.error(f"幅值比不能为0，位置: {tf_data['position']}")
        raise ZeroDivisionError(f"幅值比不能为0，位置: {tf_data['position']}")

    # 计算幅值补偿倍率
    amp_comp_ratio = mean_amp_ratio / tf_data["amp_ratio"]

    # 计算相位差补偿
    phase_shift_comp = mean_phase_shift - tf_data["phase_shift"]

    # 将相位差补偿转换为时间延迟补偿（秒）
    time_delay_comp = phase_shift_comp / (2.0 * np.pi * frequency)

    # 创建补偿数据
    comp_data: PointCompData = {
        "position": tf_data["position"],
        "amp_ratio": float(amp_comp_ratio),
        "time_delay": float(time_delay_comp),
    }

    return comp_data


def comp_to_tf(
    comp_data: PointCompData,
    frequency: float,
    mean_amp_ratio: float,
    mean_phase_shift: float,
) -> PointTFData:
    """
    将补偿数据转换为传递函数数据

    将PointCompData（相对于平均值的补偿参数）转换为PointTFData（绝对传递函数）。
    这是tf_to_comp函数的逆操作。
    转换公式：
    - 幅值比 = 平均幅值比 / 幅值补偿倍率
    - 相位差 = 平均相位差 - (时间延迟补偿 × 2π × 频率)

    Args:
        comp_data: 补偿数据（包含相对于平均值的补偿参数）
        frequency: 信号频率（Hz），用于将时间延迟转换为相位差
        mean_amp_ratio: 所有点的平均幅值比
        mean_phase_shift: 所有点的平均相位差（弧度制）

    Returns:
        传递函数数据，包含绝对幅值比和相位差

    Raises:
        ValueError: 当频率为0或负数时
        ZeroDivisionError: 当幅值补偿倍率为0时

    Examples:
        ```python
        >>> comp_data = {
        ...     "position": Point2D(x=0.0, y=1.0),
        ...     "amp_ratio": 1.05,
        ...     "time_delay": 1e-5,
        ... }
        >>> tf_data = comp_to_tf(comp_data, frequency=1000.0,
        ...                       mean_amp_ratio=0.95, mean_phase_shift=0.05)
        >>> print(tf_data)
        ```
    """
    if frequency <= 0:
        logger.error(f"频率必须为正数，收到: {frequency}")
        raise ValueError(f"频率必须为正数，收到: {frequency}")

    if comp_data["amp_ratio"] == 0:
        logger.error(f"幅值补偿倍率不能为0，位置: {comp_data['position']}")
        raise ZeroDivisionError(f"幅值补偿倍率不能为0，位置: {comp_data['position']}")

    # 计算幅值比（逆运算）
    amp_ratio = mean_amp_ratio / comp_data["amp_ratio"]

    # 将时间延迟补偿转换为相位差补偿
    phase_shift_comp = comp_data["time_delay"] * 2.0 * np.pi * frequency

    # 计算相位差（逆运算）
    phase_shift = mean_phase_shift - phase_shift_comp

    # 将相位差归一化到 [-π, π] 区间
    phase_shift = np.arctan2(np.sin(phase_shift), np.cos(phase_shift))

    # 创建传递函数数据
    tf_data: PointTFData = {
        "position": comp_data["position"],
        "amp_ratio": float(amp_ratio),
        "phase_shift": float(phase_shift),
    }

    return tf_data


def average_comp_data_list(comp_data_list: list[CompData]) -> CompData:
    """
    对多个CompData进行按位平均,返回平均后的CompData。

    该函数接收一个CompData列表,对每个通道位置的补偿参数(amp_ratio和time_delay)
    分别进行算术平均,同时对元数据字段(mean_amp_ratio和mean_phase_shift)也进行平均。

    Args:
        comp_data_list: CompData列表,要求:
            - 列表非空
            - 所有CompData的comp_list长度必须一致
            - 所有CompData的frequency应该相同(函数会使用第一个的frequency)

    Returns:
        CompData: 平均后的补偿数据,包含:
            - comp_list: 每个通道位置的平均补偿参数
            - frequency: 使用第一个CompData的频率
            - mean_amp_ratio: 所有CompData的mean_amp_ratio的平均值
            - mean_phase_shift: 所有CompData的mean_phase_shift的平均值

    Raises:
        ValueError: 如果输入列表为空
        ValueError: 如果各CompData的comp_list长度不一致

    Examples:
        >>> # 假设有3个CompData,每个包含8个通道的补偿数据
        >>> comp_data_1 = {...}  # 第1次测量
        >>> comp_data_2 = {...}  # 第2次测量
        >>> comp_data_3 = {...}  # 第3次测量
        >>> averaged = average_comp_data_list([comp_data_1, comp_data_2, comp_data_3])
        >>> # averaged包含8个通道的平均补偿参数
    """
    logger.info(f"开始平均 {len(comp_data_list)} 个CompData")

    # 验证输入列表非空
    if not comp_data_list:
        error_msg = "输入的CompData列表为空,无法进行平均"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # 验证所有CompData的comp_list长度一致
    first_length = len(comp_data_list[0]["comp_list"])
    for idx, comp_data in enumerate(comp_data_list):
        current_length = len(comp_data["comp_list"])
        if current_length != first_length:
            error_msg = (
                f"CompData列表中第 {idx} 个元素的comp_list长度({current_length}) "
                f"与第0个元素的长度({first_length})不一致"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    # 对每个通道位置进行平均
    channels_num = first_length
    averaged_comp_list: list[PointCompData] = []

    for channel_idx in range(channels_num):
        # 收集该通道在所有CompData中的amp_ratio和time_delay
        amp_ratios = [
            comp_data["comp_list"][channel_idx]["amp_ratio"]
            for comp_data in comp_data_list
        ]
        time_delays = [
            comp_data["comp_list"][channel_idx]["time_delay"]
            for comp_data in comp_data_list
        ]

        # 计算平均值
        avg_amp_ratio = float(np.mean(amp_ratios))
        avg_time_delay = float(np.mean(time_delays))

        # 创建平均后的PointCompData,position复用第一个CompData的position
        averaged_point: PointCompData = {
            "position": comp_data_list[0]["comp_list"][channel_idx]["position"],
            "amp_ratio": avg_amp_ratio,
            "time_delay": avg_time_delay,
        }
        averaged_comp_list.append(averaged_point)

        logger.debug(
            f"通道 {channel_idx}: 平均amp_ratio={avg_amp_ratio:.6f}, "
            f"平均time_delay={avg_time_delay * 1e6:.3f}μs"
        )

    # 计算平均的元数据
    avg_mean_amp_ratio = float(
        np.mean([comp_data["mean_amp_ratio"] for comp_data in comp_data_list])
    )
    avg_mean_phase_shift = float(
        np.mean([comp_data["mean_phase_shift"] for comp_data in comp_data_list])
    )
    avg_frequency = comp_data_list[0]["frequency"]  # 使用第一个的频率

    logger.info(
        f"平均完成: 频率={avg_frequency:.2f}Hz, "
        f"平均mean_amp_ratio={avg_mean_amp_ratio:.6f}, "
        f"平均mean_phase_shift={avg_mean_phase_shift:.6f}rad"
    )

    # 创建平均后的CompData
    averaged_comp_data: CompData = {
        "comp_list": averaged_comp_list,
        "frequency": avg_frequency,
        "mean_amp_ratio": avg_mean_amp_ratio,
        "mean_phase_shift": avg_mean_phase_shift,
    }

    return averaged_comp_data


def average_tf_data_list(tf_data_list: list[TFData]) -> TFData:
    """
    对多个TFData进行按位平均,返回平均后的TFData。

    该函数接收一个TFData列表,对每个通道位置的传递函数参数(amp_ratio和phase_shift)
    分别进行算术平均,同时对元数据字段(mean_amp_ratio和mean_phase_shift)也进行平均。

    Args:
        tf_data_list: TFData列表,要求:
            - 列表非空
            - 所有TFData的tf_list长度必须一致
            - 所有TFData的frequency应该相同(函数会使用第一个的frequency)

    Returns:
        TFData: 平均后的传递函数数据,包含:
            - tf_list: 每个通道位置的平均传递函数参数
            - frequency: 使用第一个TFData的频率
            - mean_amp_ratio: 所有TFData的mean_amp_ratio的平均值
            - mean_phase_shift: 所有TFData的mean_phase_shift的平均值

    Raises:
        ValueError: 如果输入列表为空
        ValueError: 如果各TFData的tf_list长度不一致

    Examples:
        >>> # 假设有3个TFData,每个包含8个通道的传递函数数据
        >>> tf_data_1 = {...}  # 第1次测量
        >>> tf_data_2 = {...}  # 第2次测量
        >>> tf_data_3 = {...}  # 第3次测量
        >>> averaged = average_tf_data_list([tf_data_1, tf_data_2, tf_data_3])
        >>> # averaged包含8个通道的平均传递函数参数
    """
    logger.info(f"开始平均 {len(tf_data_list)} 个TFData")

    # 验证输入列表非空
    if not tf_data_list:
        error_msg = "输入的TFData列表为空,无法进行平均"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # 验证所有TFData的tf_list长度一致
    first_length = len(tf_data_list[0]["tf_list"])
    for idx, tf_data in enumerate(tf_data_list):
        current_length = len(tf_data["tf_list"])
        if current_length != first_length:
            error_msg = (
                f"TFData列表中第 {idx} 个元素的tf_list长度({current_length}) "
                f"与第0个元素的长度({first_length})不一致"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    # 对每个通道位置进行平均
    channels_num = first_length
    averaged_tf_list: list[PointTFData] = []

    for channel_idx in range(channels_num):
        # 收集该通道在所有TFData中的amp_ratio和phase_shift
        amp_ratios = [
            tf_data["tf_list"][channel_idx]["amp_ratio"] for tf_data in tf_data_list
        ]
        phase_shifts = [
            tf_data["tf_list"][channel_idx]["phase_shift"] for tf_data in tf_data_list
        ]

        # 计算平均值
        avg_amp_ratio = float(np.mean(amp_ratios))
        avg_phase_shift = float(np.mean(phase_shifts))

        # 创建平均后的PointTFData,position复用第一个TFData的position
        averaged_point: PointTFData = {
            "position": tf_data_list[0]["tf_list"][channel_idx]["position"],
            "amp_ratio": avg_amp_ratio,
            "phase_shift": avg_phase_shift,
        }
        averaged_tf_list.append(averaged_point)

        logger.debug(
            f"通道 {channel_idx}: 平均amp_ratio={avg_amp_ratio:.6f}, "
            f"平均phase_shift={avg_phase_shift:.6f}rad"
        )

    # 计算平均的元数据
    avg_mean_amp_ratio = float(
        np.mean([tf_data["mean_amp_ratio"] for tf_data in tf_data_list])
    )
    avg_mean_phase_shift = float(
        np.mean([tf_data["mean_phase_shift"] for tf_data in tf_data_list])
    )
    avg_frequency = tf_data_list[0]["frequency"]  # 使用第一个的频率

    logger.info(
        f"平均完成: 频率={avg_frequency:.2f}Hz, "
        f"平均mean_amp_ratio={avg_mean_amp_ratio:.6f}, "
        f"平均mean_phase_shift={avg_mean_phase_shift:.6f}rad"
    )

    # 创建平均后的TFData
    averaged_tf_data: TFData = {
        "tf_list": averaged_tf_list,
        "frequency": avg_frequency,
        "mean_amp_ratio": avg_mean_amp_ratio,
        "mean_phase_shift": avg_mean_phase_shift,
    }

    return averaged_tf_data
