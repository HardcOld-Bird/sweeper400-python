"""
测试 general_signal 模块
"""

import pickle
from pathlib import Path

import numpy as np
import pytest

from sweeper400.analyze import (
    CalibData,
    Point2D,
    PointCompData,
    SamplingInfo,
    SineArgs,
    Waveform,
    calib_multi_ch_wf,
    init_sampling_info,
    init_sine_args,
)


class TestCalibMultiChWf:
    """测试 calib_multi_ch_wf 函数"""

    @pytest.fixture
    def sampling_info(self) -> SamplingInfo:
        """创建测试用的采样信息"""
        return init_sampling_info(sampling_rate=10000.0, samples_num=1000)

    @pytest.fixture
    def sine_args(self) -> SineArgs:
        """创建测试用的正弦波参数"""
        return init_sine_args(frequency=100.0, amplitude=1.0, phase=0.0)

    @pytest.fixture
    def calib_data(self, sine_args: SineArgs, sampling_info: SamplingInfo) -> CalibData:
        """创建测试用的校准数据"""
        # 创建4个通道的补偿数据
        comp_list: list[PointCompData] = []
        frequency = sine_args["frequency"]
        for ch_idx in range(4):
            phase_shift = ch_idx * 0.1  # 不同通道有不同的相位差
            # 根据相位差计算时间延迟
            time_delay = phase_shift / (2.0 * np.pi * frequency)

            comp_data: PointCompData = {
                "position": Point2D(x=0.0, y=float(ch_idx + 1)),
                "amp_ratio": 0.8 + ch_idx * 0.05,  # 不同通道有不同的幅值补偿倍率
                "time_delay": time_delay,  # 相对时间延迟
            }
            comp_list.append(comp_data)

        calib_data: CalibData = {
            "comp_list": comp_list,
            "ao_channels": ("ao0", "ao1", "ao2", "ao3"),
            "ai_channel": "ai0",
            "sampling_info": sampling_info,
            "sine_args": sine_args,
            "amp_ratio_mean": 0.825,  # 添加amp_ratio_mean字段（4个通道的平均值）
        }
        return calib_data

    @pytest.fixture
    def calib_data_file(
        self, calib_data: CalibData, tmp_path: Path
    ) -> Path:
        """创建临时的校准数据文件"""
        calib_file = tmp_path / "test_calib_data.pkl"
        with open(calib_file, "wb") as f:
            pickle.dump(calib_data, f)
        return calib_file

    @pytest.fixture
    def multi_ch_waveform(self, sampling_info: SamplingInfo) -> Waveform:
        """创建测试用的多通道波形（循环白噪声）"""
        # 生成4通道的随机信号
        np.random.seed(42)  # 固定随机种子以便复现
        data = np.random.randn(4, sampling_info["samples_num"])
        waveform = Waveform(
            input_array=data,
            sampling_rate=sampling_info["sampling_rate"],
        )
        return waveform

    def test_basic_functionality(
        self,
        multi_ch_waveform: Waveform,
        calib_data_file: Path,
    ) -> None:
        """测试基本功能"""
        print("\n=== 测试基本功能 ===")

        # 调用函数
        output_waveform = calib_multi_ch_wf(multi_ch_waveform, calib_data_file)

        # 验证输出形状
        assert output_waveform.shape == multi_ch_waveform.shape
        assert output_waveform.channels_num == multi_ch_waveform.channels_num
        assert output_waveform.samples_num == multi_ch_waveform.samples_num
        assert output_waveform.sampling_rate == multi_ch_waveform.sampling_rate

        print(f"输入波形形状: {multi_ch_waveform.shape}")
        print(f"输出波形形状: {output_waveform.shape}")
        print("基本功能测试通过")

    def test_amplitude_compensation(
        self,
        multi_ch_waveform: Waveform,
        calib_data_file: Path,
        calib_data: CalibData,
    ) -> None:
        """测试幅值补偿"""
        print("\n=== 测试幅值补偿 ===")

        # 调用函数
        output_waveform = calib_multi_ch_wf(multi_ch_waveform, calib_data_file)

        # 验证幅值补偿
        for ch_idx in range(multi_ch_waveform.channels_num):
            amp_ratio = calib_data["tf_list"][ch_idx]["amp_ratio"]
            # 输出应该是输入除以幅值比
            expected_amplitude = multi_ch_waveform[ch_idx, :] / amp_ratio
            # 由于还有时间补偿（循环移位），我们只检查RMS值
            input_rms = np.sqrt(np.mean(multi_ch_waveform[ch_idx, :] ** 2))
            output_rms = np.sqrt(np.mean(output_waveform[ch_idx, :] ** 2))
            expected_rms = input_rms / amp_ratio

            print(
                f"通道 {ch_idx}: 输入RMS={input_rms:.6f}, "
                f"输出RMS={output_rms:.6f}, 期望RMS={expected_rms:.6f}"
            )

            # 允许小的数值误差
            assert np.isclose(output_rms, expected_rms, rtol=1e-10)

        print("幅值补偿测试通过")

    def test_invalid_input_single_channel(
        self, sampling_info: SamplingInfo, calib_data_file: Path
    ) -> None:
        """测试单通道输入应该抛出错误"""
        print("\n=== 测试单通道输入错误处理 ===")

        # 创建单通道波形
        single_ch_data = np.random.randn(sampling_info["samples_num"])
        single_ch_waveform = Waveform(
            input_array=single_ch_data,
            sampling_rate=sampling_info["sampling_rate"],
        )

        # 应该抛出ValueError
        with pytest.raises(ValueError, match="输入波形必须是多通道"):
            calib_multi_ch_wf(single_ch_waveform, calib_data_file)

        print("单通道输入错误处理测试通过")

    def test_channel_mismatch(
        self, sampling_info: SamplingInfo, calib_data_file: Path
    ) -> None:
        """测试通道数不匹配应该抛出错误"""
        print("\n=== 测试通道数不匹配错误处理 ===")

        # 创建8通道波形（而校准数据只有4通道）
        wrong_ch_data = np.random.randn(8, sampling_info["samples_num"])
        wrong_ch_waveform = Waveform(
            input_array=wrong_ch_data,
            sampling_rate=sampling_info["sampling_rate"],
        )

        # 应该抛出ValueError
        with pytest.raises(ValueError, match="通道数.*不匹配"):
            calib_multi_ch_wf(wrong_ch_waveform, calib_data_file)

        print("通道数不匹配错误处理测试通过")
