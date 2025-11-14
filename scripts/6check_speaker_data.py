# pyright: basic
from sweeper400.analyze import (
    calculate_transfer_function,
    filter_sweep_data,
    plot_waveform,
)
from sweeper400.use.sweeper import (
    load_sweep_data,
)

# %%
save_path = "D:\\EveryoneDownloaded\\speaker_c.pkl"
# save_path = "D:\\EveryoneDownloaded\\半场.pkl"
measurement_data = load_sweep_data(save_path)
clean_data = filter_sweep_data(measurement_data, cutoff_freq=343)

# %%
_ = plot_waveform(
    measurement_data["ai_data_list"][0]["ai_data"][0],
    save_path="D:\\EveryoneDownloaded\\waveform.png",
    zoom_factor=200,
)
_ = plot_waveform(
    clean_data["ai_data_list"][0]["ai_data"][0],
    save_path="D:\\EveryoneDownloaded\\waveform_clean.png",
    zoom_factor=200,
)

# %%
tf = calculate_transfer_function(measurement_data)
print(tf)

# 2.0440082250136458
# 2.0098370337272535
# 2.0090271621625755


# %%
tf_list = []
for i in range(1, 11):
    save_path = f"D:\\EveryoneDownloaded\\mini_spks\\speaker_{i}.pkl"
    measurement_data = load_sweep_data(save_path)
    clean_data = filter_sweep_data(
        measurement_data, lowcut=1000, highcut=10000, trim_samples=500
    )
    _ = plot_waveform(
        measurement_data["ai_data_list"][0]["ai_data"][0],
        save_path=f"D:\\EveryoneDownloaded\\mini_spks\\origin_{i}.png",
        zoom_factor=200,
    )
    _ = plot_waveform(
        clean_data["ai_data_list"][0]["ai_data"][0],
        save_path=f"D:\\EveryoneDownloaded\\mini_spks\\filtered_{i}.png",
        zoom_factor=200,
    )
    tf = calculate_transfer_function(measurement_data)
    tf_list.append(tf)
    print(tf)
