# pyright: basic
from sweeper400.analyze import (  # type: ignore
    estimate_sine_args,
    extract_single_tone_information_vvi,
    get_sine,
    init_sampling_info,
    init_sine_args,
)

sampling_info = init_sampling_info(68600, 34300)

# %%
sine_args = init_sine_args(3431, 1.0, 1.73)
testwave = get_sine(
    sampling_info,
    sine_args,
)

output1 = estimate_sine_args(testwave, approx_freq=3430.0)
output2 = extract_single_tone_information_vvi(testwave, approx_freq=3430.0)
print(output1)
print(output2)
