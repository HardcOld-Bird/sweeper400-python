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
)
from .filter import detrend_waveform, filter_sweep_data, filter_waveform
from .my_dtypes import (
    Point2D,
    PointRawData,
    PositiveFloat,
    PositiveInt,
    SamplingInfo,
    SineArgs,
    SweepData,
    Waveform,
    init_sampling_info,
    init_sine_args,
)
from .post_process import (
    PointTFData,
    calculate_transfer_function,
    plot_sweep_waveforms,
    plot_transfer_function_discrete_distribution,
    plot_transfer_function_instantaneous_field,
    plot_transfer_function_interpolated_distribution,
    plot_waveform,
)
from .waveform_generator import SineGenerator, WaveformGenerator

# 控制 import * 的行为
__all__ = [
    "PositiveInt",
    "PositiveFloat",
    "SamplingInfo",
    "init_sampling_info",
    "SineArgs",
    "init_sine_args",
    "Waveform",
    "Point2D",
    "PointRawData",
    "SweepData",
    "get_sine",
    "get_sine_cycles",
    "estimate_sine_args",
    "extract_single_tone_information_vvi",
    "WaveformGenerator",
    "SineGenerator",
    "PointTFData",
    "calculate_transfer_function",
    "filter_sweep_data",
    "plot_transfer_function_discrete_distribution",
    "plot_transfer_function_interpolated_distribution",
    "plot_transfer_function_instantaneous_field",
    "plot_waveform",
    "plot_sweep_waveforms",
    "filter_waveform",
    "detrend_waveform",
]
