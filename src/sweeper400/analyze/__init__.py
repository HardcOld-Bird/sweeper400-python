"""
# analyze子包

子包路径：`sweeper400.analyze`

包含**信号和数据处理**相关的模块和类。
"""

# 将模块功能提升至包级别，可缩短外部import语句
from .basic_sine import (
    estimate_sine_args,
    extract_single_tone_information_vvi,
    get_sine,
    get_sine_cycles,
    get_sine_multi_ch,
)
from .filter import (
    detrend_waveform,
    filter_sweep_data,
    filter_waveform,
)
from .general_signal import calib_multi_ch_wf
from .my_dtypes import (
    ChannelCompData,
    ChannelTFData,
    CompData,
    Point2D,
    PointSweepData,
    PositiveFloat,
    PositiveInt,
    SamplingInfo,
    SineArgs,
    SweepData,
    TFData,
    Waveform,
    init_sampling_info,
    init_sine_args,
)
from .plot import (
    plot_sweep_waveforms,
    plot_transfer_function_discrete_distribution,
    plot_transfer_function_instantaneous_field,
    plot_transfer_function_interpolated_distribution,
    plot_waveform,
)
from .post_process import (
    average_comp_data_list,
    average_sweep_data,
    average_tf_data_list,
    comp_to_tf,
    tf_to_comp,
)
from .waveform_generator import SineGenerator, WaveformGenerator

# 控制 import * 的行为
__all__ = [
    "average_sweep_data",
    "average_comp_data_list",
    "average_tf_data_list",
    "ChannelCompData",
    "ChannelTFData",
    "CompData",
    "PositiveInt",
    "PositiveFloat",
    "SamplingInfo",
    "init_sampling_info",
    "SineArgs",
    "init_sine_args",
    "Waveform",
    "Point2D",
    "PointSweepData",
    "SweepData",
    "TFData",
    "get_sine",
    "get_sine_cycles",
    "get_sine_multi_ch",
    "estimate_sine_args",
    "extract_single_tone_information_vvi",
    "WaveformGenerator",
    "SineGenerator",
    "tf_to_comp",
    "comp_to_tf",
    "filter_sweep_data",
    "plot_transfer_function_discrete_distribution",
    "plot_transfer_function_interpolated_distribution",
    "plot_transfer_function_instantaneous_field",
    "plot_waveform",
    "plot_sweep_waveforms",
    "filter_waveform",
    "detrend_waveform",
    "calib_multi_ch_wf",
]
