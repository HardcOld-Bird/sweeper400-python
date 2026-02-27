"""
# 后处理模块

模块路径：`sweeper400.analyze.post_process`

本模块提供对Sweeper类采集到的原始数据进行后处理的功能。
主要包含按位平均、传递函数计算等数据分析功能。
"""

import numpy as np

from ..logger import get_logger
from .my_dtypes import (
    ChannelCompData,
    ChannelTFData,
    CompData,
    PointSweepData,
    SineArgs,
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
    以减少随机噪声的影响。支持单通道和多通道波形。

    对于单通道波形（1D数组），返回1D平均波形。
    对于多通道波形（2D数组），对每个通道分别平均，返回2D平均波形。

    Args:
        sweep_data: 原始的扫场测量数据

    Returns:
        处理后的SweepData，结构与输入完全相同，但每个点只有一条AI波形

    Raises:
        ValueError: 当输入数据为空或波形维度不一致时
    """

    # 获取必要参数
    ai_data_list = sweep_data["ai_data_list"]

    if not ai_data_list or not ai_data_list[0]["ai_data"]:
        raise ValueError("输入的SweepData为空")

    first_waveform = ai_data_list[0]["ai_data"][0]
    sampling_rate = first_waveform.sampling_rate
    samples_num = first_waveform.samples_num
    is_multi_channel = first_waveform.ndim == 2

    if is_multi_channel:
        num_channels = first_waveform.shape[0]
        logger.debug(f"检测到多通道波形，通道数: {num_channels}")
    else:
        logger.debug("检测到单通道波形")

    averaged_ai_data_list = []
    # 遍历每个测量点
    for _, point_data in enumerate(ai_data_list):
        # 将所有波形数据按位相加
        ai_waveforms = point_data["ai_data"]

        if is_multi_channel:
            # 多通道情况：对每个通道分别平均
            # 初始化累加数组 (num_channels, samples_num)
            summed_data = np.zeros((num_channels, samples_num), dtype=np.float64)

            for wf in ai_waveforms:
                # 验证波形维度
                if wf.ndim != 2 or wf.shape[0] != num_channels:
                    raise ValueError(
                        f"波形维度不一致：期望 (num_channels={num_channels}, samples_num={samples_num})，"
                        f"实际 shape={wf.shape}"
                    )
                summed_data += wf

            # 取平均
            averaged_data = summed_data / len(ai_waveforms)

            # 创建平均后的Waveform对象（保留channel_names元数据）
            averaged_ai_waveform = Waveform(
                input_array=averaged_data,
                sampling_rate=sampling_rate,
                timestamp=ai_waveforms[0].timestamp,
                channel_names=ai_waveforms[0].channel_names
                if hasattr(ai_waveforms[0], "channel_names")
                else None,
            )
        else:
            # 单通道情况：直接平均
            summed_data = np.zeros(samples_num, dtype=np.float64)

            for wf in ai_waveforms:
                # 验证波形维度
                if wf.ndim == 2:
                    # 如果是2D但只有1个通道，提取第一个通道
                    if wf.shape[0] == 1:
                        summed_data += wf[0, :]
                    else:
                        raise ValueError(
                            f"波形维度不一致：期望单通道，实际 shape={wf.shape}"
                        )
                else:
                    summed_data += wf

            # 取平均
            averaged_data = summed_data / len(ai_waveforms)

            # 创建平均后的Waveform对象
            averaged_ai_waveform = Waveform(
                input_array=averaged_data,
                sampling_rate=sampling_rate,
                timestamp=ai_waveforms[0].timestamp,
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

    logger.info(
        f"SweepData平均完成，共处理{len(averaged_ai_data_list)}个测量点，"
        f"{'多通道' if is_multi_channel else '单通道'}模式"
    )

    return averaged_sweep_data


def tf_to_comp(
    tf_data: ChannelTFData,
    sine_args: SineArgs,
    mean_amp_ratio: float,
    mean_phase_shift: float,
) -> ChannelCompData:
    """将通道对传递函数数据转换为通道对补偿数据。

    将 :class:`ChannelTFData`（绝对传递函数）转换为
    :class:`ChannelCompData`（相对于平均值的补偿参数）。

    转换公式：

    - 幅值补偿倍率 = 平均幅值比 / 该通道对幅值比
    - 时间延迟补偿 = (平均相位差 - 该通道对相位差) / (2π × 频率)

    参数:
        tf_data: 单个通道对的传递函数数据（包含绝对幅值比和相位差）。
        sine_args: 正弦波参数（包含频率、幅值和相位信息），用于将相位差转换为时间延迟。
        mean_amp_ratio: 所有通道对的平均幅值比。
        mean_phase_shift: 所有通道对的平均相位差（弧度制）。

    返回:
        对应该通道对的补偿数据，包含相对于平均值的幅值补偿倍率和时间延迟补偿值。

    异常:
        ValueError: 当频率为 0 或负数时。
        ZeroDivisionError: 当该通道对幅值比为 0 时。
    """
    frequency = sine_args["frequency"]

    if frequency <= 0:
        logger.error(f"频率必须为正数，收到: {frequency}")
        raise ValueError(f"频率必须为正数，收到: {frequency}")

    if tf_data["amp_ratio"] == 0:
        logger.error(
            "幅值比不能为0，通道对: AO=%s, AI=%s",
            tf_data.get("ao_channel"),
            tf_data.get("ai_channel"),
        )
        raise ZeroDivisionError(
            f"幅值比不能为0，通道对: AO={tf_data.get('ao_channel')}, "
            f"AI={tf_data.get('ai_channel')}"
        )

    # 计算幅值补偿倍率
    amp_comp_ratio = mean_amp_ratio / tf_data["amp_ratio"]

    # 计算相位差补偿
    phase_shift_comp = mean_phase_shift - tf_data["phase_shift"]

    # 将相位差补偿转换为时间延迟补偿（秒）
    time_delay_comp = phase_shift_comp / (2.0 * np.pi * frequency)

    # 创建补偿数据（保持通道对标识不变）
    comp_data: ChannelCompData = {
        "ai_channel": tf_data["ai_channel"],
        "ao_channel": tf_data["ao_channel"],
        "amp_ratio": float(amp_comp_ratio),
        "time_delay": float(time_delay_comp),
    }

    return comp_data


def comp_to_tf(
    comp_data: ChannelCompData,
    sine_args: "SineArgs",
    mean_amp_ratio: float,
    mean_phase_shift: float,
) -> ChannelTFData:
    """将通道对补偿数据转换回通道对传递函数数据。

    这是 :func:`tf_to_comp` 的逆操作：

    - 幅值比 = 平均幅值比 / 幅值补偿倍率
    - 相位差 = 平均相位差 - (时间延迟补偿 × 2π × 频率)

    参数:
        comp_data: 单个通道对的补偿数据（包含相对于平均值的补偿参数）。
        sine_args: 正弦波参数（包含频率、幅值和相位信息），用于将时间延迟转换为相位差。
        mean_amp_ratio: 所有通道对的平均幅值比。
        mean_phase_shift: 所有通道对的平均相位差（弧度制）。

    返回:
        对应该通道对的传递函数数据，包含绝对幅值比和相位差。

    异常:
        ValueError: 当频率为 0 或负数时。
        ZeroDivisionError: 当幅值补偿倍率为 0 时。
    """
    frequency = sine_args["frequency"]

    if frequency <= 0:
        logger.error(f"频率必须为正数，收到: {frequency}")
        raise ValueError(f"频率必须为正数，收到: {frequency}")

    if comp_data["amp_ratio"] == 0:
        logger.error(
            "幅值补偿倍率不能为0，通道对: AO=%s, AI=%s",
            comp_data.get("ao_channel"),
            comp_data.get("ai_channel"),
        )
        raise ZeroDivisionError(
            f"幅值补偿倍率不能为0，通道对: AO={comp_data.get('ao_channel')}, "
            f"AI={comp_data.get('ai_channel')}"
        )

    # 计算幅值比（逆运算）
    amp_ratio = mean_amp_ratio / comp_data["amp_ratio"]

    # 将时间延迟补偿转换为相位差补偿
    phase_shift_comp = comp_data["time_delay"] * 2.0 * np.pi * frequency

    # 计算相位差（逆运算）
    phase_shift = mean_phase_shift - phase_shift_comp

    # 将相位差归一化到 [-π, π] 区间
    phase_shift = np.arctan2(np.sin(phase_shift), np.cos(phase_shift))

    # 创建传递函数数据（保持通道对标识不变）
    tf_data: ChannelTFData = {
        "ai_channel": comp_data["ai_channel"],
        "ao_channel": comp_data["ao_channel"],
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
            - 所有CompData的sine_args应该相同(函数会使用第一个的sine_args)

    Returns:
        CompData: 平均后的补偿数据,包含:
            - comp_list: 每个通道位置的平均补偿参数
            - sine_args: 使用第一个CompData的正弦波参数
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

    # 对每个通道对进行平均
    channels_num = first_length
    averaged_comp_list: list[ChannelCompData] = []

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

        # 创建平均后的 ChannelCompData, 通道信息复用第一个 CompData 中对应通道对
        averaged_point: ChannelCompData = {
            "ai_channel": comp_data_list[0]["comp_list"][channel_idx]["ai_channel"],
            "ao_channel": comp_data_list[0]["comp_list"][channel_idx]["ao_channel"],
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

    # 采样信息和正弦波参数: 假定所有 CompData 一致, 使用第一个
    if "sampling_info" not in comp_data_list[0]:
        error_msg = "平均 CompData 失败: 缺少 sampling_info 字段"
        logger.error(error_msg)
        raise ValueError(error_msg)

    avg_sampling_info = comp_data_list[0]["sampling_info"]
    avg_sine_args = comp_data_list[0]["sine_args"]

    logger.info(
        f"平均完成: 频率={avg_sine_args['frequency']:.2f}Hz, "
        f"平均mean_amp_ratio={avg_mean_amp_ratio:.6f}, "
        f"平均mean_phase_shift={avg_mean_phase_shift:.6f}rad"
    )

    # 创建平均后的CompData（匹配新的类型定义）
    averaged_comp_data: CompData = {
        "comp_list": averaged_comp_list,
        "sampling_info": avg_sampling_info,
        "sine_args": avg_sine_args,
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
            - 所有TFData的sine_args应该相同(函数会使用第一个的sine_args)

    Returns:
        TFData: 平均后的传递函数数据,包含:
            - tf_list: 每个通道位置的平均传递函数参数
            - sine_args: 使用第一个TFData的正弦波参数
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

    # 对每个通道对进行平均
    channels_num = first_length
    averaged_tf_list: list[ChannelTFData] = []

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

        # 创建平均后的 ChannelTFData, 通道信息复用第一个 TFData 中对应通道对
        averaged_point: ChannelTFData = {
            "ai_channel": tf_data_list[0]["tf_list"][channel_idx]["ai_channel"],
            "ao_channel": tf_data_list[0]["tf_list"][channel_idx]["ao_channel"],
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

    # 采样信息和正弦波参数: 假定所有 TFData 一致, 使用第一个
    if "sampling_info" not in tf_data_list[0]:
        error_msg = "平均 TFData 失败: 缺少 sampling_info 字段"
        logger.error(error_msg)
        raise ValueError(error_msg)

    avg_sampling_info = tf_data_list[0]["sampling_info"]
    avg_sine_args = tf_data_list[0]["sine_args"]

    logger.info(
        f"平均完成: 频率={avg_sine_args['frequency']:.2f}Hz, "
        f"平均mean_amp_ratio={avg_mean_amp_ratio:.6f}, "
        f"平均mean_phase_shift={avg_mean_phase_shift:.6f}rad"
    )

    # 创建平均后的TFData（匹配新的类型定义）
    averaged_tf_data: TFData = {
        "tf_list": averaged_tf_list,
        "sampling_info": avg_sampling_info,
        "sine_args": avg_sine_args,
        "mean_amp_ratio": avg_mean_amp_ratio,
        "mean_phase_shift": avg_mean_phase_shift,
    }

    return averaged_tf_data
