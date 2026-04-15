"""
# 反馈函数模块

该模块包含用于 SingleChasCSIO 的反馈函数。
反馈函数接收 AI 波形数据，返回 AO 反馈波形数据。

反馈函数接口约定：
- 输入：补偿正弦参数字典、当前播放的反馈波形、静态输出波形、渔网 TF 数据
- 输出：8 通道 Waveform（AO 反馈数据）
"""

from ..analyze import Waveform, init_sine_args, get_sine_multi_ch, TFData

# 8 个 AO 反馈通道名称
_FEEDBACK_AO_CHANNELS = (
    "PXI1Slot3/ao0",
    "PXI1Slot3/ao1",
    "PXI1Slot4/ao0",
    "PXI1Slot4/ao1",
    "PXI1Slot5/ao0",
    "PXI1Slot5/ao1",
    "PXI1Slot6/ao0",
    "PXI1Slot6/ao1",
)

def static_uniform_feedback(
        ai_waveform: Waveform | None,
        static_output_waveform: Waveform | None,
        currently_playing_feedback_waveform: Waveform | None,
        fishnet_tf_data: TFData | None,
) -> Waveform:
    """
    固定 8 通道反馈函数（原型版本）

    该函数接收 9 通道 AI 波形，但**不使用** AI 数据。
    相反，它创建一个固定的 8 通道波形，应用 AO 补偿后返回。

    这是第一个反馈函数原型，用于测试和验证反馈机制。
    实际的反馈逻辑（基于 AI 数据生成 AO 输出）将在后续版本中实现。

    Args:
        - ai_waveform: Waveform | None,  # AI 波形
        - static_output_waveform: Waveform | None,  # 静态输出波形
        - currently_playing_feedback_waveform: Waveform | None,  # 当前播放的反馈波形
        - fishnet_tf_data: TFData | None,  # 渔网 TF 数据

    Returns:
        8 通道 AO 反馈波形

    Note:
        - 波形参数（频率、幅值、相位、采样率、采样点数）硬编码在函数体内
    """
    # 从输入波形获取采样信息（保持与 AI 相同的采样率）
    sampling_info = currently_playing_feedback_waveform.sampling_info

    # 硬编码的正弦波参数（可根据需要手动修改）
    sine_args = init_sine_args(
        frequency=3430.0,  # 频率：3430 Hz
        amplitude=0.02,  # 幅值：0.02 V
        phase=0.0,  # 初始相位：0 rad
    )

    # 创建 8 通道正弦波形
    feedback_waveform = get_sine_multi_ch(
        sampling_info=sampling_info,
        sine_args=sine_args,
        channel_names=_FEEDBACK_AO_CHANNELS,
    )

    return feedback_waveform

def static_diff_feedback(
        ai_waveform: Waveform | None,
        static_output_waveform: Waveform | None,
        currently_playing_feedback_waveform: Waveform | None,
        fishnet_tf_data: TFData | None,
) -> Waveform:
    """
    固定 8 通道反馈函数（原型版本）

    该函数接收 9 通道 AI 波形，但**不使用** AI 数据。
    相反，它创建一个固定的 8 通道波形，应用 AO 补偿后返回。

    这是第一个反馈函数原型，用于测试和验证反馈机制。
    实际的反馈逻辑（基于 AI 数据生成 AO 输出）将在后续版本中实现。

    Args:
        - ai_waveform: Waveform | None,  # AI 波形
        - static_output_waveform: Waveform | None,  # 静态输出波形
        - currently_playing_feedback_waveform: Waveform | None,  # 当前播放的反馈波形
        - fishnet_tf_data: TFData | None,  # 渔网 TF 数据

    Returns:
        8 通道 AO 反馈波形（已应用补偿）

    Note:
        - 波形参数（频率、幅值、相位、采样率、采样点数）硬编码在函数体内
        - 使用 storage/calib/calib_result_octopus/ao_comp_data.pkl 作为默认补偿数据
        - 如果补偿文件不存在，将使用无补偿模式
    """
    # 从输入波形获取采样信息（保持与 AI 相同的采样率）
    sampling_info = currently_playing_feedback_waveform.sampling_info

    # 硬编码的正弦波参数（可根据需要手动修改）
    sine_args = init_sine_args(
        frequency=3430.0,  # 频率：3430 Hz
        amplitude=1000,  # 幅值：0.5 V
        phase=0.0,  # 初始相位：0 rad
    )

    complex_amps = (
        -0.00000113165112806406773342531198-0.00000084717153092779020638261730j,
        0.00000178134465245466333314743306+0.00000224401729438072466407336214j,
        -0.00000227427826317513719677821694-0.00000353809190738851625347509942j,
        0.00000251696227766984811539474967+0.00000427926547585716013914107839j,
        -0.00000235085323869343136102424106-0.00000413643505902351037590560998j,
        0.00000169603179406841873904613228+0.00000327339273846739827687193566j,
        -0.00000085864382475058233050029281-0.00000223051571880521929408937136j,
        0.00000021192711737005039409534443+0.00000120688998049475036592576797j,
    )

    # 创建 8 通道正弦波形
    feedback_waveform = get_sine_multi_ch(
        sampling_info=sampling_info,
        sine_args=sine_args,
        channel_names=_FEEDBACK_AO_CHANNELS,
        complex_amps=complex_amps,
    )

    return feedback_waveform
