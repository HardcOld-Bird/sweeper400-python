"""
# 通用信号处理模块

模块路径：`sweeper400.analyze.general_signal`

本模块包含适用于任何一般信号形式的工具函数和类。
这些函数不依赖于特定的信号形式（如正弦波），可以处理任意波形。
（本模块函数较为复杂，故均配备函数日志器）
"""

import pickle
from pathlib import Path

import numpy as np

from ..logger import get_logger
from .my_dtypes import CompData, Waveform

# 获取模块日志器
logger = get_logger(__name__)


def calib_multi_ch_wf(
    input_waveform: Waveform,
    comp_data_path: str | Path,
) -> Waveform:
    """
    基于补偿数据对多通道波形进行幅值和时间补偿

    该函数接收一个多通道Waveform和一个CompData文件路径，
    根据补偿数据对每个通道进行幅值和时间补偿。

    **重要假设**：
    - 输入波形的每个通道内容在时间上是首尾相接的（循环信号）
    - 输入波形的通道数必须与CompData的通道数一致
    - 时间补偿通过循环移位实现，遵循"只切割开头，不切割末尾"的原则

    **补偿原理**：
    - 幅值补偿：输出幅值 = 输入幅值 / 传递函数幅值比
    - 时间补偿：根据相位差计算时间延迟，对信号进行循环移位
      - 时间延迟 = 相位差 / (2π * 频率)
      - 采样点延迟 = 时间延迟 * 采样率
      - 通过np.roll进行循环移位（正延迟向右移，负延迟向左移）
      - 为确保信号开头不受影响，统一将信号开头部分移到末尾

    Args:
        input_waveform: 输入的多通道波形（二维数组，每行对应一个通道）
        comp_data_path: CompData文件的路径（.pkl文件）

    Returns:
        output_waveform: 补偿后的多通道波形

    Raises:
        ValueError: 当输入波形不是多通道时
        ValueError: 当通道数与CompData不匹配时
        FileNotFoundError: 当CompData文件不存在时
        RuntimeError: 当加载CompData失败时

    Examples:
        ```python
        >>> # 假设有一个多通道白噪声信号
        >>> noise_data = np.random.randn(8, 10000)  # 8通道，10000采样点
        >>> noise_waveform = Waveform(noise_data, sampling_rate=171500.0)
        >>> # 应用校准补偿
        >>> calibrated_waveform = calib_multi_ch_wf(
        ...     noise_waveform,
        ...     "comp_data.pkl"
        ... )
        >>> print(calibrated_waveform.shape)  # (8, 10000)
        ```
    """
    # 获取函数日志器
    logger = get_logger(f"{__name__}.calib_multi_ch_wf")

    logger.debug(
        f"开始多通道波形校准补偿: "
        f"waveform_shape={input_waveform.shape}, "
        f"sampling_rate={input_waveform.sampling_rate}Hz, "
        f"comp_data_path={comp_data_path}"
    )

    # 1. 验证输入波形是多通道
    if input_waveform.ndim != 2:
        logger.error(
            f"输入波形必须是多通道（二维数组），当前维度: {input_waveform.ndim}",
            exc_info=True,
        )
        raise ValueError(
            f"输入波形必须是多通道（二维数组），当前维度: {input_waveform.ndim}"
        )

    channels_num = input_waveform.channels_num
    samples_num = input_waveform.samples_num
    sampling_rate = input_waveform.sampling_rate

    logger.debug(f"输入波形: {channels_num}通道, {samples_num}采样点")

    # 2. 加载补偿数据
    comp_data_path = Path(comp_data_path)
    if not comp_data_path.exists():
        logger.error(f"补偿数据文件不存在: {comp_data_path}", exc_info=True)
        raise FileNotFoundError(f"补偿数据文件不存在: {comp_data_path}")

    try:
        with open(comp_data_path, "rb") as f:
            comp_data: CompData = pickle.load(f)
        logger.info(f"成功加载补偿数据文件: {comp_data_path}")
    except Exception as e:
        logger.error(f"加载补偿数据文件失败: {e}", exc_info=True)
        raise RuntimeError(f"加载补偿数据文件失败: {e}") from e

    # 3. 验证通道数一致性
    comp_channels_num = len(comp_data["comp_list"])
    if channels_num != comp_channels_num:
        logger.error(
            f"输入波形通道数({channels_num})与补偿数据通道数({comp_channels_num})不匹配",
            exc_info=True,
        )
        raise ValueError(
            f"输入波形通道数({channels_num})与补偿数据通道数({comp_channels_num})不匹配"
        )

    logger.debug(f"通道数验证通过: {channels_num}通道")

    # 4. 创建输出数组
    output_data = np.zeros_like(input_waveform)

    # 5. 对每个通道进行补偿
    for ch_idx in range(channels_num):
        # 提取补偿参数
        point_comp_data = comp_data["comp_list"][ch_idx]
        amp_ratio = point_comp_data["amp_ratio"]
        time_delay = point_comp_data["time_delay"]

        logger.debug(
            f"通道 {ch_idx}: 幅值比={amp_ratio:.6f}, 时间延迟={time_delay*1e6:.3f}μs"
        )

        # 5.1 幅值补偿
        # 输出幅值 = 输入幅值 * 幅值补偿比
        compensated_amplitude = input_waveform[ch_idx, :] * amp_ratio

        # 5.2 时间补偿（通过循环移位）
        # 从补偿数据中直接使用时间延迟
        # 时间延迟已经在补偿数据中计算好了
        # 正的time_delay表示需要让信号提前（向左移）
        frequency = comp_data["sine_args"]["frequency"]

        # 计算采样点延迟（四舍五入以获得整数采样点）
        # time_delay已经是需要补偿的时间延迟
        # np.roll的shift参数：正值向右移（延迟），负值向左移（提前）
        # 正的time_delay表示需要让信号提前，所以sample_delay应该为正（向右移）
        # 遵循"只切割开头，不切割末尾"原则
        sample_delay = int(np.round(time_delay * sampling_rate))

        logger.debug(
            f"通道 {ch_idx}: 时间延迟={time_delay*1e6:.3f}μs, "
            f"采样点延迟={sample_delay}"
        )

        # 使用np.roll进行循环移位
        # np.roll(array, shift): shift>0向右移，shift<0向左移
        compensated_signal = np.roll(compensated_amplitude, sample_delay)

        # 存储到输出数组
        output_data[ch_idx, :] = compensated_signal

    # 6. 创建输出Waveform对象
    output_waveform = Waveform(
        input_array=output_data,
        sampling_rate=sampling_rate,
        timestamp=input_waveform.timestamp,
        id=input_waveform.id,
        sine_args=input_waveform.sine_args,
    )

    logger.debug(
        f"多通道波形校准补偿完成: shape={output_waveform.shape}, "
        f"channels_num={output_waveform.channels_num}"
    )

    return output_waveform
