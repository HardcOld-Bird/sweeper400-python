# pyright: basic
from src.sweeper400.analyze import (
    average_sweep_data,
    calculate_transfer_function,
    filter_sweep_data,
    plot_sweep_waveforms,
    plot_transfer_function_discrete_distribution,
    plot_transfer_function_instantaneous_field,
    plot_transfer_function_interpolated_distribution,
)
from src.sweeper400.use.sweeper import (
    load_sweep_data,
)

save_path = "D:\\EveryoneDownloaded\\test_data.pkl"

measurement_data = load_sweep_data(save_path)

# %%

measurement_data = average_sweep_data(measurement_data)
clean_data = filter_sweep_data(
    measurement_data, lowcut=1000, highcut=10000, trim_samples=500
)

# %%

plot_sweep_waveforms(
    clean_data,
    output_dir="D:\\EveryoneDownloaded\\",
    zoom_factor=200,
)

# %%

tf = calculate_transfer_function(clean_data)

# %%
_ = plot_transfer_function_discrete_distribution(
    tf, save_path="D:\\EveryoneDownloaded\\discrete_tf.png"
)
# %%
_ = plot_transfer_function_interpolated_distribution(
    tf,
    save_path="D:\\EveryoneDownloaded\\interpolated_tf.png",
)
_ = plot_transfer_function_instantaneous_field(
    tf, save_path="D:\\EveryoneDownloaded\\instantaneous_field.png"
)
