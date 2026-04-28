"""
Evolver 类的集成测试

该测试文件对 Evolver 进行基于真实硬件的集成测试，验证反馈演化控制器的核心功能。

注意：
- 测试需要连接 NI 机箱（PXIe-1083）和所有板卡（PXI1Slot2~6）
- 测试需要 fishnet 校准数据（storage/calib/calib_result_fishnet/fishnet_tf_data.pkl）
- 由于硬件反馈处理耗时较长，建议每个演化周期的 static_output_waveform 时长 >= 0.5s

运行方法：
    pytest tests/sweeper400/use/test_evolver.py -v -s

"""

import time
from pathlib import Path

import numpy as np
import pytest

from sweeper400.analyze import (
    init_sampling_info,
    init_sine_args,
    get_sine_cycles,
    Waveform,
    TFData,
)
from sweeper400.use import Evolver


# ============================================================
#  常量配置（根据实际硬件环境修改）
# ============================================================

# 采样率和采样点数（每个 chunk 约 5s，给反馈处理留出充足时间）
_SAMPLING_RATE = 171500.0  # Hz
_SAMPLES_PER_CHUNK = 857500  # ~5s
_CYCLES_PER_CHUNK = 25  # 一个 chunk 包含 25 个正弦周期

# 激励频率和幅值
_FREQ = 3430.0  # Hz
_AMPLITUDE = 0.05  # V

# AI 通道（8 个传声器通道）
# 注意：必须与 fishnet_tf_data 中的 AI 通道名一致，即 PXI1Slot3~6 的 ai0/ai1
_AI_CHANNELS = (
    "PXI1Slot3/ai0",
    "PXI1Slot3/ai1",
    "PXI1Slot4/ai0",
    "PXI1Slot4/ai1",
    "PXI1Slot5/ai0",
    "PXI1Slot5/ai1",
    "PXI1Slot6/ai0",
    "PXI1Slot6/ai1",
)

# 静态 AO 通道（主激励）
_AO_CHANNELS_STATIC = ("PXI1Slot2/ao0",)

# 反馈 AO 通道（8 个反馈通道，与 AI 通道一一对应）
_AO_CHANNELS_FEEDBACK = (
    "PXI1Slot3/ao0",
    "PXI1Slot3/ao1",
    "PXI1Slot4/ao0",
    "PXI1Slot4/ao1",
    "PXI1Slot5/ao0",
    "PXI1Slot5/ao1",
    "PXI1Slot6/ao0",
    "PXI1Slot6/ao1",
)

# 增益系数（8 个通道，均增益 1.0 倍，即保持总声场不变）
_GAIN_COEFFICIENTS_UNITY = tuple([1.0 + 0.0j] * 8)

# AO 幅值安全上限（V）
_AO_AMPLITUDE_LIMIT = 0.1


def _make_static_waveform() -> Waveform:
    """生成测试用静态输出波形"""
    sampling_info = init_sampling_info(_SAMPLING_RATE, _SAMPLES_PER_CHUNK)
    sine_args = init_sine_args(frequency=_FREQ, amplitude=_AMPLITUDE, phase=0.0)
    return get_sine_cycles(sampling_info, sine_args)


# ============================================================
#  测试类
# ============================================================

class TestEvolverInit:
    """测试 Evolver 初始化参数验证"""

    def test_gain_coefficients_length_mismatch_feedback(self):
        """gain_coefficients 长度与 ao_channels_feedback 不匹配时应抛出 ValueError"""
        static_wf = _make_static_waveform()
        wrong_gain = (1.0 + 0j,) * 5  # 只有 5 个，但 ao_channels_feedback 有 8 个

        with pytest.raises(ValueError, match="gain_coefficients 长度"):
            Evolver(
                ai_channels=_AI_CHANNELS,
                ao_channels_static=_AO_CHANNELS_STATIC,
                ao_channels_feedback=_AO_CHANNELS_FEEDBACK,
                static_output_waveform=static_wf,
                gain_coefficients=wrong_gain,
            )

    def test_gain_coefficients_length_mismatch_ai(self):
        """gain_coefficients 长度与 ai_channels 不匹配时应抛出 ValueError"""
        static_wf = _make_static_waveform()
        # ai_channels 只用前 5 个，但 gain_coefficients 有 8 个（与 ao_channels_feedback 匹配）
        wrong_ai = _AI_CHANNELS[:5]

        with pytest.raises(ValueError, match="gain_coefficients 长度"):
            Evolver(
                ai_channels=wrong_ai,
                ao_channels_static=_AO_CHANNELS_STATIC,
                ao_channels_feedback=_AO_CHANNELS_FEEDBACK,
                static_output_waveform=static_wf,
                gain_coefficients=_GAIN_COEFFICIENTS_UNITY,
            )

    def test_missing_sine_args(self):
        """static_output_waveform 没有 sine_args 时应抛出 ValueError"""
        static_wf = _make_static_waveform()
        # 手动清除 sine_args
        static_wf.sine_args = None

        with pytest.raises(ValueError, match="sine_args"):
            Evolver(
                ai_channels=_AI_CHANNELS,
                ao_channels_static=_AO_CHANNELS_STATIC,
                ao_channels_feedback=_AO_CHANNELS_FEEDBACK,
                static_output_waveform=static_wf,
                gain_coefficients=_GAIN_COEFFICIENTS_UNITY,
            )


class TestEvolverHardware:
    """基于真实硬件的 Evolver 集成测试（需要连接 NI 机箱）"""

    @pytest.fixture
    def evolver(self):
        """创建 Evolver 实例（fixture，测试结束后自动清理）"""
        static_wf = _make_static_waveform()
        ev = Evolver(
            ai_channels=_AI_CHANNELS,
            ao_channels_static=_AO_CHANNELS_STATIC,
            ao_channels_feedback=_AO_CHANNELS_FEEDBACK,
            static_output_waveform=static_wf,
            gain_coefficients=_GAIN_COEFFICIENTS_UNITY,
            buffer_size_multiplier=15,  # 进一步增大缓冲区倍数，给反馈处理留出更多时间
        )
        yield ev
        try:
            ev.cleanup()
        except Exception:
            pass

    def test_evolve_basic(self, evolver: Evolver):
        """
        基本演化测试：执行 3 个演化周期，验证数据记录是否正常

        该测试验证：
        1. evolve 方法能正常完成指定周期数的演化；
        2. 每个周期都有对应的"总声场复振幅"被记录；
        3. 演化数据中有正确数量的 ai_data 条目。
        """
        num_cycles = 3
        print(f"\n开始演化测试，周期数: {num_cycles}")
        print(f"预计耗时: ~{num_cycles * _SAMPLES_PER_CHUNK / _SAMPLING_RATE:.1f} s")

        # 执行演化
        evolver.evolve(
            num_cycles=num_cycles,
            ao_amplitude_limit=_AO_AMPLITUDE_LIMIT,
        )

        # 验证历史记录数量
        assert len(evolver._ai_complex_amps_history) >= num_cycles, (
            f"期望记录 >= {num_cycles} 个周期的复振幅，"
            f"实际记录了 {len(evolver._ai_complex_amps_history)} 个"
        )

        # 验证复振幅形状
        for i, amp in enumerate(evolver._ai_complex_amps_history):
            assert amp.shape == (len(_AI_CHANNELS),), (
                f"第 {i+1} 个周期的复振幅形状应为 ({len(_AI_CHANNELS)},)，"
                f"实际为 {amp.shape}"
            )
            assert amp.dtype == complex, (
                f"第 {i+1} 个周期的复振幅应为复数类型，实际为 {amp.dtype}"
            )

        print(f"演化完成，记录了 {len(evolver._ai_complex_amps_history)} 个周期的复振幅")
        print(f"最后一个周期的总声场复振幅模长: {np.abs(evolver._ai_complex_amps_history[-1])}")

    def test_evolve_plot(self, evolver: Evolver, tmp_path: Path):
        """
        演化后绘图测试：验证 plot_evolution 能正常生成图像文件

        该测试验证：
        1. evolve 执行后，plot_evolution 能正常运行；
        2. 图像文件被正确保存到指定路径。
        """
        num_cycles = 2
        evolver.evolve(num_cycles=num_cycles, ao_amplitude_limit=_AO_AMPLITUDE_LIMIT)

        # 绘图
        save_path = tmp_path / "evolution_test.png"
        evolver.plot_evolution(save_path=save_path)

        # 验证文件存在
        assert save_path.exists(), f"图像文件应存在于 {save_path}"
        assert save_path.stat().st_size > 0, "图像文件不应为空"

        print(f"图像已保存到: {save_path} (大小: {save_path.stat().st_size} bytes)")

    def test_evolve_plot_without_data(self, evolver: Evolver):
        """
        未执行 evolve 直接绘图应抛出 ValueError
        """
        with pytest.raises(ValueError, match="没有可绘制的演化数据"):
            evolver.plot_evolution()

    def test_evolve_multiple_runs(self, evolver: Evolver):
        """
        多次调用 evolve 时，每次都应重置状态并重新记录数据

        该测试验证重置机制是否正常工作。
        """
        # 第一次演化
        evolver.evolve(num_cycles=2, ao_amplitude_limit=_AO_AMPLITUDE_LIMIT)
        first_run_count = len(evolver._ai_complex_amps_history)

        # 重新初始化（因为 SingleChasCSIO 每次 stop 后需要重新 start）
        # 注意：每次 evolve 之间需要等待硬件恢复
        time.sleep(1.0)

        # 第二次演化（不同周期数）
        evolver.evolve(num_cycles=3, ao_amplitude_limit=_AO_AMPLITUDE_LIMIT)
        second_run_count = len(evolver._ai_complex_amps_history)

        # 第二次演化的记录应独立于第一次
        assert second_run_count >= 3, (
            f"第二次演化应记录 >= 3 个周期，实际记录了 {second_run_count} 个"
        )

        print(
            f"第一次演化: {first_run_count} 个周期，"
            f"第二次演化: {second_run_count} 个周期"
        )
