"""
# 后处理模块

模块路径：`sweeper400.analyze.post_process`

本模块提供对Sweeper类采集到的原始数据进行后处理的功能。
主要包含按位平均、传递函数计算等数据分析功能。
"""

import numpy as np

from ..logger import get_logger
from .basic_sine import extract_single_tone_information_vvi
from .my_dtypes import PointRawData, PointTFData, SweepData, Waveform

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
        averaged_point_data: PointRawData = {
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
) -> list[PointTFData]:
    """
    计算Sweeper采集数据的传递函数

    对每个测量点的原始数据进行处理，计算输入输出信号的复数传递函数 H(ω) = A·e^(jφ)。
    具体步骤：
    1. 若存在多个AI波形chunks，进行按位相加并取平均
    2. 使用extract_single_tone_information_vvi提取AI信号的正弦波参数
    3. 使用共用的AO波形的正弦波参数
    4. 计算传递函数：幅值比 = AI幅值 / AO幅值，相位差 = AI相位 - AO相位

    Args:
        sweep_data: Sweeper采集的完整测量数据，包含ai_data_list和ao_data

    Returns:
        传递函数结果列表，每个元素包含位置、绝对幅值比和绝对相位差

    Raises:
        ValueError: 当输入数据为空或格式不正确时
        RuntimeError: 当AO数据没有sine_args属性时

    Examples:
        ```python
        >>> # 假设已有采集的原始数据
        >>> sweep_data = sweeper.get_data()  # noqa
        >>> # 计算传递函数
        >>> tf_results = calculate_transfer_function(sweep_data)
        >>> for result in tf_results:
        ...     print(
        ...         f"位置: {result['position']}, "
        ...         f"幅值比: {result['amp_ratio']:.4f}, "
        ...         f"相位差: {result['phase_shift']:.4f}rad"
        ...     )
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

    return results
