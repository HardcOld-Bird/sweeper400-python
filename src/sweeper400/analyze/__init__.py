"""
# analyze子包

子包路径：`sweeper400.analyze`

包含各种各样的底层工具模块和类，是本项目的底层子包，不应依赖任何其他子包。
"""

# 将模块功能提升至包级别，可缩短外部import语句
from .basic_sine import (
    estimate_sine_args,
    extract_single_tone_information_vvi,
    get_sine,
    get_sine_cycles,
    get_sine_multi_ch,
)
from .calib_util_funcs import (
    load_data_with_fallback,
    comp_ai_sine_args,
    comp_multi_ch_wf,
)
from .filter import (
    detrend_waveform,
    filter_sweep_data,
    filter_waveform,
)
from .my_dtypes import (
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
    plot_sweepdata_as_single_waveform,
    plot_transfer_function_discrete_distribution,
    plot_transfer_function_instantaneous_field,
    plot_transfer_function_interpolated_distribution,
    plot_waveform,
    sweep_data_to_point_tf_data,
)
from .post_process import (
    average_comp_data_list,
    average_sweep_data,
    average_tf_data_list,
    comp_to_tf,
    load_sweep_data,
    load_compressed_data,
    save_sweep_data,
    save_compressed_data,
    tf_to_comp,
)
# 控制 import * 的行为
__all__ = [
    "average_sweep_data",
    "average_comp_data_list",
    "average_tf_data_list",
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
    "tf_to_comp",
    "comp_to_tf",
    "load_sweep_data",
    "save_sweep_data",
    "filter_sweep_data",
    "plot_transfer_function_discrete_distribution",
    "plot_transfer_function_interpolated_distribution",
    "plot_transfer_function_instantaneous_field",
    "plot_waveform",
    "plot_sweep_waveforms",
    "plot_sweepdata_as_single_waveform",
    "filter_waveform",
    "detrend_waveform",
    "sweep_data_to_point_tf_data",
    "load_compressed_data",
    "save_compressed_data",
    "comp_ai_sine_args",
    "comp_multi_ch_wf",
    "load_data_with_fallback",
]
