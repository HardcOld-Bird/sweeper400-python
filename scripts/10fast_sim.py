"""
# Caliber 硬件测试脚本

这是一个用于实际硬件测试的脚本，可以直接运行来测试校准功能。
"""

from sweeper400.sim import Simulator

# 设置增益系数
gain_coeffs = [
    -0.308988 - 2.557868j,
    -1.196245 - 1.412161j,
    -1.541522 - 1.275200j,
    -1.518875 - 1.254205j,
    -1.534567 - 1.262961j,
    -1.532444 - 1.264957j,
    -1.499747 - 1.267534j,
    -1.703700 - 1.289199j,
]

# 创建测试对象
simer = Simulator(gain_coefficients=gain_coeffs)

# %% 执行仿真
_ = simer.run_simulation(
    num_iterations=5,
)
# _ = simer.run_simulation(
#     num_iterations=100,
# )
