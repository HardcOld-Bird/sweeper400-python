# pyright: basic
from sweeper400.analyze import (
    apply_highpass_filter_to_sweep_data,
    calculate_transfer_function,
    plot_sweep_waveforms,
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

# %%

clean_data = apply_highpass_filter_to_sweep_data(measurement_data)

# %%

plot_sweep_waveforms(
    clean_data,
    output_dir="D:\\EveryoneDownloaded\\",
    # zoom_factor=200,
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
