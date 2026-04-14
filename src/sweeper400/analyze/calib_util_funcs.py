"""
# 校准工具函数模块

模块路径：`sweeper400.analyze.calib_util_funcs`

包含原本属于caliber（或者说，随caliber产生）的工具函数。
"""

from pathlib import Path
from typing import Any

import numpy as np

from .my_dtypes import (
    CompData,
    SineArgs,
    Waveform,
)
from ..logger import get_logger

# 获取模块日志器
logger = get_logger(__name__)


def load_data_with_fallback(
    explicit_path: str | Path | None,
    default_path: str | Path,
    data_type: str,
) -> Any | None:
    """
    智能加载数据文件，支持显式路径、默认路径和回退到None的三级优先级。

    支持gzip压缩和非压缩文件格式。
    可用于加载CompData、TFData等各种数据类型。

    优先级：
    1. 用户提供的显式路径（如果提供）
    2. 默认路径下的数据文件（如果存在）
    3. 不使用数据（返回None）

    Args:
        explicit_path: 用户显式提供的数据文件路径（可选）
        default_path: 默认的全局数据文件路径
        data_type: 数据类型描述字符串，仅用于日志输出（如"AO补偿数据"、"TFData"等）

    Returns:
        Any: 成功加载的数据对象，如果都不存在则返回None

    Raises:
        FileNotFoundError: 当用户显式提供路径但文件不存在时
        RuntimeError: 当文件加载失败时
    """
    from ..analyze.post_process import load_compressed_data

    # 优先级1：用户显式提供的路径
    if explicit_path is not None:
        explicit_path_obj = Path(explicit_path)
        if not explicit_path_obj.exists():
            error_msg = f"{data_type}文件不存在: {explicit_path_obj}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            data: Any = load_compressed_data(
                explicit_path_obj, f"{data_type}"
            )
            logger.debug(
                f"成功加载{data_type}（用户显式路径）: {explicit_path_obj}"
            )
            return data
        except Exception as e:
            error_msg = f"加载{data_type}文件失败: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    # 优先级2：默认路径下的数据文件
    default_path_obj = Path(default_path)
    if default_path_obj.exists():
        try:
            data: Any = load_compressed_data(
                default_path_obj, f"{data_type}"
            )
            logger.debug(
                f"成功加载{data_type}（默认全局路径）: {default_path_obj}"
            )
            return data
        except Exception as e:
            logger.warning(
                f"默认{data_type}文件存在但加载失败: {default_path_obj}, "
                f"错误: {e}，将回退到无数据模式",
                exc_info=True,
            )
            # 继续回退到优先级3

    # 优先级3：不使用数据
    logger.debug(f"未找到{data_type}，将使用无数据模式")
    return None


def comp_ai_sine_args(
    sine_args: SineArgs,
    ai_comp_data: CompData | None,
    ai_channel_name: str,
) -> SineArgs:
    """
    应用AI通道补偿到正弦波参数

    该函数根据AI通道补偿数据（通常由CaliberSardine生成）对正弦波参数进行补偿，
    校正传声器之间的差异。如果未提供补偿数据或未找到指定通道的补偿数据，
    则返回原始的正弦波参数。

    补偿逻辑：
    - 幅值补偿：补偿后幅值 = 原始幅值 × AI幅值补偿倍率
    - 相位补偿：补偿后相位 = 原始相位 + AI时间延迟补偿对应的相位

    Args:
        sine_args: 原始正弦波参数，包含频率、幅值和相位信息
        ai_comp_data: AI通道补偿数据（CompData格式），如果为None则不进行补偿
        ai_channel_name: AI通道名称，用于查找对应的补偿参数

    Returns:
        SineArgs: 补偿后的正弦波参数。如果未找到补偿数据，返回原始参数的副本

    Examples:
        >>> # 应用补偿
        >>> original_args = {"frequency": 1000.0, "amplitude": 1.0, "phase": 0.0}
        >>> ai_comp = load_comp_data("ai_comp_data.pkl")
        >>> compensated_args = comp_ai_sine_args(original_args, ai_comp, "PXI1Slot2/ai0")
        >>> # compensated_args 包含补偿后的幅值和相位

        >>> # 未提供补偿数据
        >>> result = comp_ai_sine_args(original_args, None, "PXI1Slot2/ai0")
        >>> # result 等于 original_args
    """
    # 如果未提供补偿数据，返回原始参数的副本
    if ai_comp_data is None:
        return SineArgs(
            frequency=sine_args["frequency"],
            amplitude=sine_args["amplitude"],
            phase=sine_args["phase"],
        )

    # 查找指定AI通道的补偿数据
    comp_df = ai_comp_data["comp_dataframe"]

    # 检查AI通道是否在补偿数据中
    if ai_channel_name not in comp_df.index:
        logger.debug(f"AI通道 {ai_channel_name} 未找到补偿数据，使用原始参数")
        return SineArgs(
            frequency=sine_args["frequency"],
            amplitude=sine_args["amplitude"],
            phase=sine_args["phase"],
        )

    # 获取该通道的补偿参数
    amp_multiplier = comp_df.loc[ai_channel_name, "amp_multiplier"]
    time_increment = comp_df.loc[ai_channel_name, "time_increment"]

    # 应用AI通道补偿
    # 幅值补偿：补偿后幅值 = 原始幅值 × AI幅值补偿倍率
    compensated_amplitude = sine_args["amplitude"] * amp_multiplier

    # 相位补偿：补偿后相位 = 原始相位 + AI时间延迟补偿对应的相位
    time_delay_phase = time_increment * 2.0 * np.pi * sine_args["frequency"]
    compensated_phase = sine_args["phase"] + time_delay_phase

    # 归一化补偿后的相位到 [-π, π] 区间
    compensated_phase = float(
        np.arctan2(np.sin(compensated_phase), np.cos(compensated_phase))
    )

    logger.debug(
        f"AI通道 {ai_channel_name} 应用补偿: "
        f"amplitude {sine_args['amplitude']:.6f} -> {compensated_amplitude:.6f}, "
        f"phase {sine_args['phase']:.6f}rad -> {compensated_phase:.6f}rad"
    )

    # 返回补偿后的正弦波参数
    return SineArgs(
        frequency=sine_args["frequency"],
        amplitude=float(compensated_amplitude),
        phase=compensated_phase,
    )


def comp_multi_ch_wf(
    input_waveform: Waveform,
    comp_data: CompData | None,
) -> Waveform:
    """
    基于补偿数据对多通道波形进行幅值和时间补偿（支持部分补偿和单通道）

    该函数接收一个Waveform（可以是单通道1D或多通道2D）和补偿数据（CompData），
    根据补偿数据对每个通道进行幅值和时间补偿。

    **重要特性**：
    - 支持单通道波形（1D）：自动扩展为单通道2D进行处理
    - 支持部分补偿：输入波形的通道数可以多于CompData中的通道
    - 对于CompData中存在的通道，应用补偿
    - 对于CompData中不存在的通道，保持原样（不补偿）

    **重要假设**：
    - 输入波形的每个通道内容在时间上是首尾相接的（循环信号）
    - 输入波形的channel_names属性必须已设置，用于匹配补偿数据
    - 时间补偿通过循环移位实现，遵循"只切割开头，不切割末尾"的原则

    **补偿原理**：
    - 幅值补偿：输出幅值 = 输入幅值 × 幅值补偿倍率
    - 时间补偿：根据时间延迟计算采样点延迟，对信号进行循环移位
      - 采样点延迟 = 时间延迟 × 采样率
      - 通过np.roll进行循环移位（正延迟向右移，负延迟向左移）
      - 为确保信号开头不受影响，统一将信号开头部分移到末尾

    Args:
        input_waveform: 输入的波形（可以是1D单通道或2D多通道），
                       必须设置channel_names属性
        comp_data: 补偿数据（CompData格式），如果为None则不进行补偿

    Returns:
        output_waveform: 补偿后的波形（与输入维度相同）

    Raises:
        ValueError: 当输入波形维度不是1D或2D时
        ValueError: 当输入波形的channel_names为None时
        ValueError: 当通道数与channel_names长度不匹配时

    Examples:
        ```python
        >>> # 多通道白噪声信号示例
        >>> noise_data = np.random.randn(8, 10000)  # 8通道，10000采样点
        >>> ao_channels = ("PXI1Slot2/ao0", "PXI1Slot2/ao1", ...)
        >>> noise_waveform = Waveform(
        ...     noise_data,
        ...     sampling_rate=171500.0,
        ...     channel_names=ao_channels
        ... )
        >>> # 加载补偿数据（假设只包含部分通道）
        >>> ao_comp_data = load_data_with_fallback(...)
        >>> # 应用补偿（支持部分补偿）
        >>> calibrated_waveform = comp_multi_ch_wf(
        ...     noise_waveform,
        ...     ao_comp_data,
        ... )
        >>> print(calibrated_waveform.shape)  # (8, 10000)

        >>> # 单通道波形示例
        >>> single_ch_data = np.random.randn(10000)  # 1通道，10000采样点
        >>> single_ch_waveform = Waveform(
        ...     single_ch_data,
        ...     sampling_rate=171500.0,
        ...     channel_names=("PXI1Slot2/ao0",)
        ... )
        >>> # 应用补偿
        >>> calibrated_waveform = comp_multi_ch_wf(
        ...     single_ch_waveform,
        ...     ao_comp_data,
        ... )
        >>> print(calibrated_waveform.shape)  # (10000,) - 保持1D
        ```
    """
    # 获取函数日志器
    logger = get_logger(f"{__name__}.comp_multi_ch_wf")

    # 1. 如果未提供补偿数据，直接返回原始波形的副本（保持原维度）
    if comp_data is None:
        logger.debug("未提供AO补偿数据，返回原始波形")
        return input_waveform

    # 2. 验证输入波形维度并处理单通道情况
    original_ndim = input_waveform.ndim
    is_1d_input = original_ndim == 1

    if original_ndim == 1:
        # 单通道输入：扩展为2D以便统一处理
        logger.debug("检测到单通道输入(1D)，将扩展为2D进行处理")
        waveform_data = np.array(input_waveform).reshape(1, -1)
        channels_num = 1
    elif original_ndim == 2:
        # 多通道输入：直接使用
        waveform_data = np.array(input_waveform)
        channels_num = input_waveform.channels_num
    else:
        logger.error(
            f"输入波形维度必须是1D或2D，当前维度: {original_ndim}",
            exc_info=True,
        )
        raise ValueError(
            f"输入波形维度必须是1D或2D，当前维度: {original_ndim}"
        )

    samples_num = input_waveform.samples_num
    sampling_rate = input_waveform.sampling_rate

    # 3. 验证channel_names已设置
    channel_names = input_waveform.channel_names
    if channel_names is None:
        logger.error(
            "输入波形的channel_names属性为None，必须设置通道名称以匹配补偿数据",
            exc_info=True,
        )
        raise ValueError(
            "输入波形的channel_names属性为None，必须设置通道名称以匹配补偿数据"
        )

    logger.debug(
        f"开始补偿: "
        f"original_ndim={original_ndim}, "
        f"waveform_shape={input_waveform.shape}, "
        f"sampling_rate={input_waveform.sampling_rate}Hz, "
        f"channels={channel_names}"
    )

    logger.debug(f"输入波形: {channels_num}通道, {samples_num}采样点")

    # 4. 验证通道数一致性
    if channels_num != len(channel_names):
        logger.error(
            f"输入波形通道数({channels_num})与channel_names长度({len(channel_names)})不匹配",
            exc_info=True,
        )
        raise ValueError(
            f"输入波形通道数({channels_num})与channel_names长度({len(channel_names)})不匹配"
        )

    # 5. 获取补偿数据DataFrame
    comp_df = comp_data["comp_dataframe"]
    comp_channels = set(comp_df.index.tolist())

    # 6. 创建输出数组（初始复制输入数据）
    output_data = waveform_data.copy()

    # 7. 对每个通道进行补偿（仅对CompData中存在的通道）
    compensated_count = 0
    for ch_idx, channel_name in enumerate(channel_names):
        # 检查该通道是否在补偿数据中
        if channel_name not in comp_channels:
            logger.warning(f"通道 {channel_name}: 不在补偿数据中，保持原样")
            continue

        # 提取补偿参数
        amp_multiplier = comp_df.loc[channel_name, "amp_multiplier"]
        time_increment = comp_df.loc[channel_name, "time_increment"]

        logger.debug(
            f"通道 {channel_name}: 幅值倍率={amp_multiplier:.6f}, "
            f"时间增量={time_increment * 1e6:.3f}μs"
        )

        # 7.1 幅值补偿
        # 输出幅值 = 输入幅值 × 幅值补偿倍率
        compensated_amplitude = waveform_data[ch_idx, :] * amp_multiplier

        # 7.2 时间补偿（通过循环移位）
        # 计算采样点延迟（四舍五入以获得整数采样点）
        # time_increment是时间延迟补偿值（秒）
        # np.roll的shift参数：正值向右移（延迟），负值向左移（提前）
        sample_delay = int(np.round(time_increment * sampling_rate))

        logger.debug(
            f"通道 {channel_name}: 时间增量={time_increment * 1e6:.3f}μs, "
            f"采样点延迟={sample_delay}"
        )

        # 使用np.roll进行循环移位
        # np.roll(array, shift): shift>0向右移，shift<0向左移
        compensated_signal = np.roll(compensated_amplitude, sample_delay)

        # 存储到输出数组
        output_data[ch_idx, :] = compensated_signal
        compensated_count += 1

    logger.debug(f"补偿完成: {compensated_count}/{channels_num} 个通道已补偿")

    # 8. 创建输出Waveform对象（保持原维度）
    if is_1d_input:
        # 如果是1D输入，将结果squeeze回1D
        output_array = output_data.squeeze(axis=0)
    else:
        output_array = output_data

    output_waveform = Waveform(
        input_array=output_array,
        sampling_rate=sampling_rate,
        timestamp=input_waveform.timestamp,
        id=input_waveform.id,
        sine_args=input_waveform.sine_args,
    )

    logger.debug(
        f"补偿完成: original_ndim={original_ndim}, output_shape={output_waveform.shape}, "
        f"channels_num={output_waveform.channels_num}"
    )

    return output_waveform
