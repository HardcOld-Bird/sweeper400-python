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
    Waveform,
)
from .filter import detrend_waveform
from .post_process import load_compressed_data
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
    # 获取函数日志器
    f_logger = get_logger(f"{__name__}.load_data_with_fallback")

    # 优先级1：用户显式提供的路径
    if explicit_path is not None:
        explicit_path_obj = Path(explicit_path)
        if not explicit_path_obj.exists():
            error_msg = f"{data_type}文件不存在: {explicit_path_obj}"
            f_logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            data: Any = load_compressed_data(
                explicit_path_obj, f"{data_type}"
            )
            f_logger.debug(
                f"成功加载{data_type}（用户显式路径）: {explicit_path_obj}"
            )
            return data
        except Exception as e:
            error_msg = f"加载{data_type}文件失败: {e}"
            f_logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    # 优先级2：默认路径下的数据文件
    default_path_obj = Path(default_path)
    if default_path_obj.exists():
        try:
            data: Any = load_compressed_data(
                default_path_obj, f"{data_type}"
            )
            f_logger.debug(
                f"成功加载{data_type}（默认全局路径）: {default_path_obj}"
            )
            return data
        except Exception as e:
            f_logger.warning(
                f"默认{data_type}文件存在但加载失败: {default_path_obj}, "
                f"错误: {e}，将回退到无数据模式",
                exc_info=True,
            )
            # 继续回退到优先级3

    # 优先级3：不使用数据
    f_logger.debug(f"未找到{data_type}，将使用无数据模式")
    return None


def comp_waveform(
    input_waveform: Waveform,
    comp_data: CompData | None,
) -> Waveform:
    """
    基于补偿数据对多通道波形进行幅值和时间补偿（支持部分补偿）

    该函数接收一个Waveform（统一使用2D格式）和补偿数据（CompData），
    根据补偿数据对每个通道进行幅值和时间补偿。

    **重要特性**：
    - 支持部分补偿：输入波形的通道数可以多于CompData中的通道
    - 对于CompData中存在的通道，应用补偿
    - 对于CompData中不存在的通道，保持原样（不补偿）

    **重要假设**：
    - 输入波形的每个通道内容在时间上是首尾相接的（循环信号）
    - 输入波形的channel_names属性必须已设置，用于匹配补偿数据
    - 时间补偿通过循环移位实现，遵循"只切割开头，不切割末尾"的原则

    **补偿原理**：
    - 预处理：对每个通道进行去趋势处理（去除直流偏移），避免循环移位时产生基线跳变
    - 幅值补偿：输出幅值 = 输入幅值 × 幅值补偿倍率
    - 时间补偿：根据时间增量计算采样点偏移，对信号进行循环移位
      - time_increment 的物理含义：需要施加给信号的相位变化 = time_increment × 2πf
      - 由于 np.roll(signal, +n) 使信号相位减少 2πf×n/fs，
        为获得正确的相位增加方向，使用 np.roll(signal, -sample_delay)
      - 采样点偏移 = round(time_increment × 采样率)

    Args:
        input_waveform: 输入的波形（统一使用2D格式，单通道为(1, n_samples)），
                       必须设置channel_names属性
        comp_data: 补偿数据（CompData格式），如果为None则不进行补偿

    Returns:
        output_waveform: 补偿后的波形（2D格式）

    Raises:
        ValueError: 当输入波形的channel_names为None时
        ValueError: 当通道数与channel_names长度不匹配时

    Examples:
        ```python
        >>> # 多通道白噪声信号示例
        >>> noise_data = np.random.randn(8, 10000)  # 8通道，10000采样点
        >>> ao_channels = ("PXI1Slot2/ao0", "PXI1Slot2/ao1", ...)
        >>> noise_waveform = Waveform(
        ...     noise_data,  # noqa
        ...     sampling_rate=171500.0,
        ...     _channel_names=ao_channels
        ... )
        >>> # 加载补偿数据（假设只包含部分通道）
        >>> ao_comp_data = load_data_with_fallback(...)
        >>> # 应用补偿（支持部分补偿）
        >>> calibrated_waveform = comp_waveform(
        ...     noise_waveform,
        ...     ao_comp_data,
        ... )
        >>> print(calibrated_waveform.shape)  # (8, 10000)

        >>> # 单通道波形示例
        >>> single_ch_data = np.random.randn(10000)  # 1通道，10000采样点
        >>> single_ch_waveform = Waveform(
        ...     single_ch_data,  # noqa
        ...     sampling_rate=171500.0,
        ...     _channel_names=("PXI1Slot2/ao0",)
        ... )
        >>> # 应用补偿
        >>> calibrated_waveform = comp_waveform(
        ...     single_ch_waveform,
        ...     ao_comp_data,
        ... )
        >>> print(calibrated_waveform.shape)  # (1, 10000)
        ```
    """
    # 获取函数日志器
    f_logger = get_logger(f"{__name__}.comp_waveform")

    # 1. 如果未提供补偿数据，直接返回原始波形的副本
    if comp_data is None:
        f_logger.warning("未提供AO补偿数据，返回原始波形")
        return input_waveform

    # 2. 获取波形数据（Waveform统一使用2D格式）
    channels_num = input_waveform.channels_num
    sampling_rate = input_waveform.sampling_rate
    channel_names = input_waveform.channel_names

    f_logger.debug(
        f"开始补偿: "
        f"waveform_shape={input_waveform.shape}, "
        f"channels_num={channels_num}, "
        f"sampling_rate={input_waveform.sampling_rate}Hz, "
        f"channels={channel_names}"
    )

    # 3. 获取补偿数据DataFrame
    comp_df = comp_data["comp_dataframe"]
    comp_channels = set(comp_df.index.tolist())

    # 4. 对每个通道进行去趋势处理（避免循环移位时产生基线跳变）
    detrended_waveform = detrend_waveform(input_waveform)

    # 5. 创建输出数组（复制去趋势后的数据）
    output_data = detrended_waveform.copy()

    # 6. 对每个通道进行补偿（仅对CompData中存在的通道）
    # 并计算补偿后的channel_complex_amplitude
    input_cca = input_waveform.channel_complex_amplitudes
    input_freq = input_waveform.frequency
    if input_cca is not None and input_freq is not None:
        new_cca = input_cca.copy()
    else:
        new_cca = None

    compensated_count = 0
    for ch_idx, channel_name in enumerate(channel_names):
        # 检查该通道是否在补偿数据中
        if channel_name not in comp_channels:
            # 对缺失的非扫场麦克风通道输出警告（扫场麦克风缺失是预期行为）
            if channel_name != "PXI1Slot2/ai0":  # 扫场麦克风的通道名称
                f_logger.warning(f"通道 {channel_name}: 不在补偿数据中，保持原样")
            continue

        # 提取补偿参数
        amp_multiplier = comp_df.loc[channel_name, "amp_multiplier"]
        time_increment = comp_df.loc[channel_name, "time_increment"]

        f_logger.debug(
            f"通道 {channel_name}: 幅值倍率={amp_multiplier:.6f}, "
            f"时间增量={time_increment * 1e6:.3f}μs"
        )

        # 6.1 幅值补偿
        # 输出幅值 = 输入幅值 × 幅值补偿倍率
        compensated_amplitude = detrended_waveform[ch_idx, :] * amp_multiplier

        # 6.2 时间补偿（通过循环移位）
        # time_increment 的物理含义：需要施加的相位变化 = time_increment × 2πf
        # （与 comp_ai_sine_args 中 phase += time_increment * 2πf 保持一致）
        # 由于 np.roll(signal, +n) 使相位减少 2πf*n/fs（正移位=延迟=相位减少），
        # 而我们需要相位增加 time_increment*2πf，因此使用 -sample_delay
        sample_delay = int(np.round(time_increment * sampling_rate))

        # 使用np.roll进行循环移位（取反以匹配相位增加方向）
        compensated_signal = np.roll(compensated_amplitude, -sample_delay)

        f_logger.debug(
            f"通道 {channel_name}: 时间增量={time_increment * 1e6:.3f}μs, "
            f"采样点偏移={-sample_delay}（负号修正roll方向）"
        )

        # 存储到输出数组
        output_data[ch_idx, :] = compensated_signal
        compensated_count += 1

        # 记录补偿后的复振幅
        if new_cca is not None:
            assert input_freq is not None
            # 新复振幅 = 原复振幅 × amp_multiplier × exp(1j × time_increment × 2πf)
            phase_shift = time_increment * 2 * np.pi * input_freq
            new_cca[ch_idx] = (
                    new_cca[ch_idx] * amp_multiplier * np.exp(1j * phase_shift)
            )

    f_logger.debug(f"补偿完成: {compensated_count}/{channels_num} 个通道已补偿")

    # 7. 创建输出Waveform对象（2D格式）
    output_waveform = Waveform(
        input_array=output_data,
        sampling_rate=sampling_rate,
        channel_names=channel_names,
        timestamp=input_waveform.timestamp,
        waveform_id=input_waveform.waveform_id,
        frequency=input_freq,
        channel_complex_amplitude=new_cca,
    )

    f_logger.debug(
        f"补偿完成: output_shape={output_waveform.shape}, "
        f"channels_num={output_waveform.channels_num}"
    )

    return output_waveform
