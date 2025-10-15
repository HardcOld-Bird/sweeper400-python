# pyright: basic
from sweeper400.analyze import (
    calculate_transfer_function,
    plot_transfer_function_discrete_distribution,
    plot_transfer_function_instantaneous_field,
    plot_transfer_function_interpolated_distribution,
)
from sweeper400.use.sweeper import (
    load_sweep_data,
)

save_path = "D:\\EveryoneDownloaded\\test_data.pkl"
# save_path = "D:\\EveryoneDownloaded\\半场.pkl"
measurement_data = load_sweep_data(save_path)
tf = calculate_transfer_function(measurement_data)

_ = plot_transfer_function_discrete_distribution(
    tf, save_path="D:\\EveryoneDownloaded\\discrete_tf.png"
)
_ = plot_transfer_function_interpolated_distribution(
    tf,
    save_path="D:\\EveryoneDownloaded\\interpolated_tf.png",
)
_ = plot_transfer_function_instantaneous_field(
    tf, save_path="D:\\EveryoneDownloaded\\instantaneous_field.png"
)
