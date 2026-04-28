"""
测试 post_process 模块中的函数

模块路径: tests.sweeper400.analyze.test_post_process
"""

import numpy as np
import pytest

from sweeper400.analyze.my_dtypes import Waveform, init_sine_args
from sweeper400.analyze.post_process import average_single_waveform


class TestAverageSingleWaveform:
    """测试 average_single_waveform 函数"""

    def test_basic_average_single_channel(self):
        """测试单通道波形的基本平均功能"""
        # 创建一个1000点的波形，值为1到1000
        data = np.arange(1, 1001).reshape(1, 1000).astype(np.float64)
        wf = Waveform(data, sampling_rate=10000, channel_names=("ch1",))

        # 10段平均，每段100点
        result = average_single_waveform(wf, segments=10)

        # 验证结果形状: (1, 100) - 每段100点，对应位置平均
        assert result.shape == (1, 100)
        assert result.channels_num == 1
        assert result.samples_num == 100

        # 验证平均结果：
        # 第1点 = (1 + 101 + 201 + ... + 901) / 10 = 451
        # 第2点 = (2 + 102 + 202 + ... + 902) / 10 = 452
        # 以此类推...
        expected = np.arange(451, 551).reshape(1, 100)
        np.testing.assert_array_almost_equal(result, expected)

    def test_basic_average_multi_channel(self):
        """测试多通道波形的基本平均功能"""
        # 创建2通道波形，每通道1000点
        ch1_data = np.arange(1, 1001)
        ch2_data = np.arange(1001, 2001)
        data = np.vstack([ch1_data, ch2_data]).astype(np.float64)
        wf = Waveform(data, sampling_rate=10000, channel_names=("ch1", "ch2"))

        # 10段平均
        result = average_single_waveform(wf, segments=10)

        # 验证结果形状: (2, 100)
        assert result.shape == (2, 100)
        assert result.channels_num == 2
        assert result.samples_num == 100

        # 验证每个通道的平均结果：
        # ch1第1点 = (1 + 101 + 201 + ... + 901) / 10 = 451
        # ch2第1点 = (1001 + 1101 + 1201 + ... + 1901) / 10 = 1451
        expected_ch1 = np.arange(451, 551)
        expected_ch2 = np.arange(1451, 1551)
        expected = np.vstack([expected_ch1, expected_ch2])
        np.testing.assert_array_almost_equal(result, expected)

    def test_default_segments(self):
        """测试默认segments值为10"""
        # 创建1000点的波形
        data = np.random.randn(1, 1000)
        wf = Waveform(data, sampling_rate=10000)

        # 使用默认segments
        result = average_single_waveform(wf)

        # 默认10段，每段100点
        assert result.samples_num == 100

    def test_indivisible_length_raises_error(self):
        """测试当长度不能被segments整除时抛出错误"""
        # 创建1000点的波形
        data = np.random.randn(1, 1000)
        wf = Waveform(data, sampling_rate=10000)

        # 1000不能被3整除，应该抛出ValueError
        with pytest.raises(ValueError) as exc_info:
            average_single_waveform(wf, segments=3)

        assert "不能被segments(3)整除" in str(exc_info.value)
        assert "余数为 1" in str(exc_info.value)
        assert "相位误差" in str(exc_info.value)

    def test_invalid_segments_zero(self):
        """测试segments为0时抛出错误"""
        data = np.random.randn(1, 100)
        wf = Waveform(data, sampling_rate=10000)

        with pytest.raises(ValueError) as exc_info:
            average_single_waveform(wf, segments=0)

        assert "必须是正整数" in str(exc_info.value)

    def test_invalid_segments_negative(self):
        """测试segments为负数时抛出错误"""
        data = np.random.randn(1, 100)
        wf = Waveform(data, sampling_rate=10000)

        with pytest.raises(ValueError) as exc_info:
            average_single_waveform(wf, segments=-5)

        assert "必须是正整数" in str(exc_info.value)

    def test_metadata_preserved(self):
        """测试元数据被正确保留"""
        data = np.random.randn(2, 1000)
        sine_args = init_sine_args(frequency=1000.0, amplitude=1.0, phase=0.0)
        timestamp = np.datetime64("2024-01-15T10:30:00", "ns")

        wf = Waveform(
            data,
            sampling_rate=50000,
            channel_names=("AI0", "AI1"),
            timestamp=timestamp,
            waveform_id=42,
            sine_args=sine_args,
        )

        result = average_single_waveform(wf, segments=10)

        # 验证元数据保留
        assert result.sampling_rate == 50000
        assert result.channel_names == ("AI0", "AI1")
        assert result.timestamp == timestamp
        assert result.waveform_id == 42
        assert result.sine_args == sine_args

    def test_noise_reduction(self):
        """测试噪声减少效果"""
        np.random.seed(42)
        # 创建带有噪声的信号
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 10 * t)  # 10Hz正弦波
        noise = np.random.randn(1000) * 0.1
        data = (signal + noise).reshape(1, 1000)

        wf = Waveform(data, sampling_rate=1000)

        # 10段平均
        result = average_single_waveform(wf, segments=10)

        # 验证噪声被减少（标准差应该减小）
        original_std = np.std(data)
        result_std = np.std(result)
        assert result_std < original_std

    def test_segments_equal_to_samples(self):
        """测试segments等于采样点数的情况（每段1点）"""
        data = np.arange(1, 101).reshape(1, 100).astype(np.float64)
        wf = Waveform(data, sampling_rate=10000)

        # 100段，每段1点
        result = average_single_waveform(wf, segments=100)

        # 结果应该与原数据相同
        assert result.shape == (1, 1)
        np.testing.assert_array_almost_equal(result, np.array([[50.5]]))

    def test_segments_one(self):
        """测试segments为1的情况（整段平均）"""
        data = np.arange(1, 101).reshape(1, 100).astype(np.float64)
        wf = Waveform(data, sampling_rate=10000)

        # 1段，整段平均
        result = average_single_waveform(wf, segments=1)

        # 结果为单个平均值
        assert result.shape == (1, 100)
        np.testing.assert_array_almost_equal(result, data)
