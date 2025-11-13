# pyright: basic
from sweeper400.analyze import (
    calculate_transfer_function,
)
from sweeper400.use.sweeper import (
    load_sweep_data,
)

for i in range(1, 11):
    save_path = f"D:\\EveryoneDownloaded\\speaker_{i}.pkl"
    measurement_data = load_sweep_data(save_path)
    tf = calculate_transfer_function(measurement_data)
    print(tf)
