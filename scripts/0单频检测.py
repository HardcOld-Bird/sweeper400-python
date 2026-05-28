# pyright: basic

import time

import numpy as np

from sweeper400.analyze import (
    extract_single_tone_information_vvi,
    get_sine,
    init_sampling_info,
)

sampling_info = init_sampling_info(171500.0, 34300)

# %%
test_wf = get_sine(
    sampling_info,
    frequency=3430,
    channel_names=("A", "B"),
    channel_complex_amplitudes=np.asarray([1+0j, 2+3j]),
)

start1 = time.perf_counter()
output1 = extract_single_tone_information_vvi(test_wf, approx_freq=3400.0, precise_mode=False)
end1 = time.perf_counter()

print(output1)
print(f"estimate_sine_args 耗时: {(end1 - start1) * 1000:.3f} ms")

start2 = time.perf_counter()
output2 = extract_single_tone_information_vvi(test_wf, approx_freq=3400.0, precise_mode=True)
end2 = time.perf_counter()

print(output2)
print(f"extract_single_tone_information_vvi 耗时: {(end2 - start2) * 1000:.3f} ms")
