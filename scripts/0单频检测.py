# pyright: basic

import time

from sweeper400.analyze import (
    estimate_sine_args,
    extract_single_tone_information_vvi,
    get_sine,
    init_sampling_info,
    init_sine_args,
)

sampling_info = init_sampling_info(171500.0, 34300)

# %%
sine_args = init_sine_args(3400, 2.0, 1.73)
test_wave = get_sine(
    sampling_info,
    sine_args,
)

start1 = time.perf_counter()
output1 = estimate_sine_args(test_wave, approx_freq=3430.0)
end1 = time.perf_counter()

print(output1)
print(f"estimate_sine_args 耗时: {(end1 - start1) * 1000:.3f} ms")

start2 = time.perf_counter()
output2 = extract_single_tone_information_vvi(test_wave, approx_freq=3430.0)
end2 = time.perf_counter()

print(output2)
print(f"extract_single_tone_information_vvi 耗时: {(end2 - start2) * 1000:.3f} ms")
