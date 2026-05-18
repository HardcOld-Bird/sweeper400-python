# pyright: basic

from sweeper400.analyze import (
    load_compressed_data,
    sweep_data_to_point_tf_data_list,
    subtract_point_tf_data_list,
    plot_point_tf_data_list,
)

# 读取数据并绘图
total_field_sweep_data = load_compressed_data(
    "D:\\EveryoneDownloaded\\L_INPUT_t\\sweep_data.pkl",
)
total_field_list = sweep_data_to_point_tf_data_list(
    total_field_sweep_data,
    lowcut=1715.0,
    highcut=6860.0,
)
background_field_sweep_data = load_compressed_data(
    "D:\\EveryoneDownloaded\\L_INPUT_b\\sweep_data.pkl",
)
background_field_list = sweep_data_to_point_tf_data_list(
    background_field_sweep_data,
    lowcut=1715.0,
    highcut=6860.0,
)
scatter_field_list = subtract_point_tf_data_list(
    total_field_list,
    background_field_list,
)
plot_point_tf_data_list(
    scatter_field_list,
    mode="discrete",
    save_path="D:\\EveryoneDownloaded\\L_INPUT_s_discrete.png",
)
plot_point_tf_data_list(
    scatter_field_list,
    mode="interpolated",
    save_path="D:\\EveryoneDownloaded\\L_INPUT_s_interpolated.png",
)
plot_point_tf_data_list(
    scatter_field_list,
    mode="instantaneous",
    save_path="D:\\EveryoneDownloaded\\L_INPUT_s_instantaneous.png",
)
