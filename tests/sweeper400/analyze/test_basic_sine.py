"""
测试 basic_sine 模块中的函数

模块路径: tests.sweeper400.analyze.test_basic_sine
"""

import numpy as np
import pytest

from sweeper400.analyze import (
    esti_vvi_multi_ch,
    extract_single_tone_information_vvi,
    get_sine,
    get_sine_multi_ch,
    init_sampling_info,
    init_sine_args,
)


class TestEstiVviMultiCh:
    """测试 esti_vvi_multi_ch 函数"""

    def test_single_channel_equivalent_to_extract_single_tone(self):
        """测试单通道情况下与extract_single_tone_information_vvi结果一致"""
        # 生成单通道测试波形
        sampling_info = init_sampling_info(10000, 2048)
        sine_args = init_sine_args(500.0, 2.0, 0.5)
        test_wave = get_sine(sampling_info, sine_args, channel_name="ch1")

        # 使用两个函数分别提取
        result_multi = esti_vvi_multi_ch(test_wave, approx_freq=500.0)
        result_single = extract_single_tone_information_vvi(test_wave, approx_freq=500.0)

        # 验证结果一致性（转换为复振幅比较）
        expected_complex = result_single["amplitude"] * np.exp(1j * result_single["phase"])

        assert result_multi.shape == (1,)
        assert np.abs(result_multi[0] - expected_complex) < 0.01  # 允许1%误差
        assert np.abs(np.abs(result_multi[0]) - result_single["amplitude"]) < 0.01
        assert np.abs(np.angle(result_multi[0]) - result_single["phase"]) < 0.01

    def test_multi_channel_same_signal(self):
        """测试多通道相同信号的情况"""
        # 生成8通道相同波形
        sampling_info = init_sampling_info(171500.0, 85750)
        sine_args = init_sine_args(3430.0, 1.0, 0.0)
        channels = tuple(f"ch{i}" for i in range(8))
        test_wave = get_sine_multi_ch(sampling_info, sine_args, channels)

        # 提取复振幅
        result = esti_vvi_multi_ch(test_wave, approx_freq=3430.0)

        # 验证结果
        assert result.shape == (8,)

        # 所有通道应该具有相近的幅值和相位
        amplitudes = np.abs(result)
        phases = np.angle(result)

        # 幅值差异应该很小（相同信号）
        assert np.std(amplitudes) < 0.01
        # 相位差异应该很小
        assert np.std(phases) < 0.01

        # 验证频率估计正确
        assert np.allclose(amplitudes, 1.0, atol=0.05)  # 幅值接近1

    def test_multi_channel_with_phase_shift(self):
        """测试多通道带相位差的情况"""
        sampling_info = init_sampling_info(171500.0, 85750)
        sine_args = init_sine_args(3430.0, 1.0, 0.0)
        channels = tuple(f"ch{i}" for i in range(4))

        # 为每个通道设置不同的相位偏移
        phase_shifts = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        complex_amps = [np.exp(1j * ph) for ph in phase_shifts]

        test_wave = get_sine_multi_ch(
            sampling_info, sine_args, channels, complex_amps=tuple(complex_amps)
        )

        # 提取复振幅
        result = esti_vvi_multi_ch(test_wave, approx_freq=3430.0)

        # 验证结果
        assert result.shape == (4,)

        # 验证相位差被正确检测
        detected_phases = np.angle(result)
        phase_diffs = np.diff(detected_phases)
        expected_diffs = [np.pi / 4, np.pi / 4, np.pi / 4]

        for i, (diff, expected) in enumerate(zip(phase_diffs, expected_diffs)):
            # 处理相位环绕
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            assert np.abs(diff - expected) < 0.1, f"通道{i}和{i+1}之间的相位差不正确"

    def test_multi_channel_with_amplitude_variation(self):
        """测试多通道带幅值变化的情况"""
        sampling_info = init_sampling_info(100000.0, 50000)
        sine_args = init_sine_args(5000.0, 1.0, 0.0)
        channels = tuple(f"ch{i}" for i in range(5))

        # 为每个通道设置不同的幅值
        amp_multipliers = [0.5, 0.8, 1.0, 1.2, 1.5]
        complex_amps = [amp * np.exp(0j) for amp in amp_multipliers]

        test_wave = get_sine_multi_ch(
            sampling_info, sine_args, channels, complex_amps=tuple(complex_amps)
        )

        # 提取复振幅
        result = esti_vvi_multi_ch(test_wave, approx_freq=5000.0)

        # 验证结果
        assert result.shape == (5,)

        # 验证幅值被正确检测
        detected_amplitudes = np.abs(result)

        for i, (detected, expected) in enumerate(zip(detected_amplitudes, amp_multipliers)):
            assert np.abs(detected - expected) < 0.05, f"通道{i}的幅值检测不正确"

    def test_without_approx_freq(self):
        """测试不提供approx_freq时的全频段搜索"""
        sampling_info = init_sampling_info(10000, 2048)
        sine_args = init_sine_args(1234.0, 1.5, 0.3)
        channels = tuple(f"ch{i}" for i in range(3))
        test_wave = get_sine_multi_ch(sampling_info, sine_args, channels)

        # 不提供approx_freq
        result = esti_vvi_multi_ch(test_wave)

        # 验证结果
        assert result.shape == (3,)

        # 验证检测到的频率接近1234Hz
        # 注意：由于我们不返回频率，只能通过复振幅间接验证
        # 如果频率估计错误，幅值和相位估计也会受影响
        amplitudes = np.abs(result)
        assert np.allclose(amplitudes, 1.5, atol=0.1)

    def test_use_curve_fit_option(self):
        """测试use_curve_fit选项"""
        sampling_info = init_sampling_info(10000, 2048)
        sine_args = init_sine_args(1000.0, 2.0, 0.5)
        channels = tuple(f"ch{i}" for i in range(4))
        test_wave = get_sine_multi_ch(sampling_info, sine_args, channels)

        # 不使用curve_fit
        result_fast = esti_vvi_multi_ch(test_wave, approx_freq=1000.0, use_curve_fit=False)
        # 使用curve_fit
        result_precise = esti_vvi_multi_ch(test_wave, approx_freq=1000.0, use_curve_fit=True)

        # 验证结果形状
        assert result_fast.shape == (4,)
        assert result_precise.shape == (4,)

        # 两种方法的结果应该相近
        np.testing.assert_allclose(result_fast, result_precise, rtol=0.05)

    def test_with_noisy_signal(self):
        """测试带噪声信号的情况"""
        np.random.seed(42)
        sampling_info = init_sampling_info(50000, 10000)
        sine_args = init_sine_args(5000.0, 1.0, 0.0)
        channels = tuple(f"ch{i}" for i in range(4))

        # 先生成干净信号
        clean_wave = get_sine_multi_ch(sampling_info, sine_args, channels)

        # 添加噪声
        noise = np.random.randn(4, 10000) * 0.05  # 5%噪声
        noisy_data = np.asarray(clean_wave) + noise

        from sweeper400.analyze.my_dtypes import Waveform

        noisy_wave = Waveform(
            noisy_data,
            sampling_rate=sampling_info["sampling_rate"],
            channel_names=channels,
        )

        # 提取复振幅
        result = esti_vvi_multi_ch(noisy_wave, approx_freq=5000.0)

        # 验证结果
        assert result.shape == (4,)

        # 在噪声存在下，幅值估计应该接近1.0
        amplitudes = np.abs(result)
        assert np.allclose(amplitudes, 1.0, atol=0.1)

    def test_return_type_is_complex_ndarray(self):
        """测试返回类型是否为复数ndarray"""
        sampling_info = init_sampling_info(10000, 1024)
        sine_args = init_sine_args(1000.0, 1.0, 0.0)
        test_wave = get_sine(sampling_info, sine_args)

        result = esti_vvi_multi_ch(test_wave, approx_freq=1000.0)

        # 验证返回类型
        assert isinstance(result, np.ndarray)
        assert np.iscomplexobj(result)
        assert result.ndim == 1

    def test_frequency_accuracy(self):
        """测试频率估计的准确性"""
        # 使用高精度参数
        sampling_info = init_sampling_info(100000.0, 50000)
        test_freq = 12345.67  # 使用非整数频率
        sine_args = init_sine_args(test_freq, 1.0, 0.0)
        channels = tuple(f"ch{i}" for i in range(3))
        test_wave = get_sine_multi_ch(sampling_info, sine_args, channels)

        # 提取复振幅（使用curve_fit以获得更高精度）
        result = esti_vvi_multi_ch(test_wave, approx_freq=test_freq, use_curve_fit=True)

        # 验证幅值准确性
        amplitudes = np.abs(result)
        assert np.allclose(amplitudes, 1.0, atol=0.01)

        # 验证相位一致性
        phases = np.angle(result)
        assert np.std(phases) < 0.01  # 相同信号，相位应该一致

    def test_performance_multi_channel(self):
        """测试多通道处理的性能（确保比单通道循环更快）"""
        import time

        sampling_info = init_sampling_info(171500.0, 85750)
        sine_args = init_sine_args(3430.0, 1.0, 0.0)
        channels = tuple(f"ch{i}" for i in range(10))
        test_wave = get_sine_multi_ch(sampling_info, sine_args, channels)

        # 测试esti_vvi_multi_ch性能
        start = time.perf_counter()
        for _ in range(10):
            result_multi = esti_vvi_multi_ch(test_wave, approx_freq=3430.0)
        time_multi = time.perf_counter() - start

        # 测试单通道循环性能（使用extract_single_tone_information_vvi）
        start = time.perf_counter()
        for _ in range(10):
            for ch_idx in range(10):
                ch_wave = test_wave[ch_idx : ch_idx + 1, :]
                extract_single_tone_information_vvi(ch_wave, approx_freq=3430.0)
        time_single = time.perf_counter() - start

        # 多通道方法应该比单通道循环快（至少快2倍）
        assert time_multi < time_single / 2, (
            f"多通道方法({time_multi:.3f}s)应该比单通道循环({time_single:.3f}s)更快"
        )
