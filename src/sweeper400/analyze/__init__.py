"""
# analyze子包

子包路径：`sweeper400.analyze`

包含各种各样的底层工具模块和类，是本项目的底层子包，不应依赖任何其他子包。
"""

# 将模块功能提升至包级别，可缩短外部import语句
from .basic_sine import (
    extract_single_tone_information_vvi,
    get_sine,
)
from .calib_util_funcs import (
    comp_waveform,
    load_data_with_fallback,
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
    SweepData,
    TFData,
    Waveform,
    init_comp_data,
    init_sampling_info,
)
from .plot import (
    PointTFData,
    plot_point_tf_data_list,
    plot_sweep_data_as_single_waveform,
    plot_sweep_waveforms,
    plot_waveform,
    subtract_point_tf_data_list,
    sweep_data_to_point_tf_data_list,
)
from .post_process import (
    average_comp_data_list,
    average_single_waveform,
    average_sweep_data,
    average_tf_data_list,
    comp_to_tf,
    load_compressed_data,
    save_compressed_data,
    tf_to_comp,
)

# 控制 import * 的行为
__all__ = [
    "average_sweep_data",
    "average_single_waveform",
    "average_comp_data_list",
    "average_tf_data_list",
    "CompData",
    "init_comp_data",
    "PositiveInt",
    "PositiveFloat",
    "SamplingInfo",
    "init_sampling_info",
    "Waveform",
    "Point2D",
    "PointSweepData",
    "SweepData",
    "TFData",
    "get_sine",
    "extract_single_tone_information_vvi",
    "tf_to_comp",
    "comp_to_tf",
    "filter_sweep_data",
    "plot_point_tf_data_list",
    "plot_waveform",
    "plot_sweep_waveforms",
    "plot_sweep_data_as_single_waveform",
    "filter_waveform",
    "detrend_waveform",
    "sweep_data_to_point_tf_data_list",
    "subtract_point_tf_data_list",
    "PointTFData",
    "load_compressed_data",
    "save_compressed_data",
    "comp_waveform",
    "load_data_with_fallback",
]
