"""
测试Caliber类的新功能：result_folder参数和多通道补偿波形生成

本测试文件测试以下功能：
1. Caliber.calibrate方法的result_folder参数
2. get_sine_multi_chs函数生成多通道补偿波形
3. 完整的校准-补偿工作流程
"""

import pickle
from pathlib import Path

import numpy as np
import pytest

from sweeper400.analyze import (
    CalibData,
    SamplingInfo,
    SineArgs,
    get_sine_multi_chs,
    init_sampling_info,
    init_sine_args,
)
from sweeper400.use.caliber import Caliber


class TestCaliberWithCompensation:
    """测试Caliber类的新功能"""

    @pytest.fixture
    def ai_channel(self):
        """AI通道"""
        return "PXI2Slot2/ai0"

    @pytest.fixture
    def ao_channels(self):
        """AO通道列表"""
        return (
            "PXI2Slot2/ao0",
            "PXI2Slot2/ao1",
            "PXI2Slot3/ao0",
            "PXI2Slot3/ao1",
            "PXI3Slot2/ao0",
            "PXI3Slot2/ao1",
            "PXI3Slot3/ao0",
            "PXI3Slot3/ao1",
        )

    @pytest.fixture
    def sampling_info(self):
        """采样信息"""
        return init_sampling_info(171500.0, 85750)

    @pytest.fixture
    def sine_args(self):
        """正弦波参数"""
        return init_sine_args(frequency=3430.0, amplitude=0.01, phase=0.0)

    @pytest.fixture
    def caliber(self, ai_channel, ao_channels, sampling_info, sine_args):
        """创建Caliber实例"""
        return Caliber(
            ai_channel=ai_channel,
            ao_channels=ao_channels,
            sampling_info=sampling_info,
            sine_args=sine_args,
        )

    @pytest.mark.hardware
    def test_calibrate_with_result_folder(self, caliber, tmp_path):
        """测试calibrate方法的result_folder参数"""
        print("\n=== 测试calibrate方法的result_folder参数 ===")

        # 创建结果文件夹路径
        result_folder = tmp_path / "calibration_results"

        # 执行校准并自动保存结果
        caliber.calibrate(starts_num=2, chunks_per_start=3, result_folder=result_folder)

        # 验证文件夹已创建
        assert result_folder.exists()
        assert result_folder.is_dir()

        # 验证所有文件都已保存
        assert (result_folder / "raw_sweep_data.pkl").exists()
        assert (result_folder / "calib_data.pkl").exists()
        assert (result_folder / "transfer_function_averaged.png").exists()
        assert (result_folder / "transfer_function_detailed.png").exists()

        print(f"\n所有文件已保存到: {result_folder}")
        print(f"  - raw_sweep_data.pkl: {(result_folder / 'raw_sweep_data.pkl').stat().st_size} bytes")
        print(f"  - calib_data.pkl: {(result_folder / 'calib_data.pkl').stat().st_size} bytes")
        print(f"  - transfer_function_averaged.png: {(result_folder / 'transfer_function_averaged.png').stat().st_size} bytes")
        print(f"  - transfer_function_detailed.png: {(result_folder / 'transfer_function_detailed.png').stat().st_size} bytes")

    @pytest.mark.hardware
    def test_get_sine_multi_chs_with_real_calibration(
        self, caliber, sampling_info, sine_args, tmp_path
    ):
        """测试使用真实校准数据生成多通道补偿波形"""
        print("\n=== 测试使用真实校准数据生成多通道补偿波形 ===")

        # 1. 执行校准
        print("\n步骤1: 执行校准...")
        caliber.calibrate(starts_num=2, chunks_per_start=3)

        # 2. 保存校准数据
        calib_file = tmp_path / "calib_data.pkl"
        caliber.save_calib_data(calib_file)

        # 3. 加载校准数据
        print("\n步骤2: 加载校准数据...")
        with open(calib_file, "rb") as f:
            calib_data: CalibData = pickle.load(f)

        # 4. 生成多通道补偿波形
        print("\n步骤3: 生成多通道补偿波形...")
        multi_ch_waveform = get_sine_multi_chs(
            sampling_info=sampling_info,
            sine_args=sine_args,
            calib_data=calib_data,
        )

        # 5. 验证波形
        print("\n步骤4: 验证波形...")
        assert multi_ch_waveform.ndim == 2
        assert multi_ch_waveform.channels_num == len(caliber.ao_channels)
        assert multi_ch_waveform.samples_num == sampling_info["samples_num"]
        assert multi_ch_waveform.sampling_rate == sampling_info["sampling_rate"]

        print(f"\n多通道补偿波形生成成功:")
        print(f"  - 形状: {multi_ch_waveform.shape}")
        print(f"  - 通道数: {multi_ch_waveform.channels_num}")
        print(f"  - 采样点数: {multi_ch_waveform.samples_num}")
        print(f"  - 采样率: {multi_ch_waveform.sampling_rate}Hz")

        # 6. 检查每个通道的幅值
        print("\n每个通道的补偿幅值:")
        for ch_idx in range(multi_ch_waveform.channels_num):
            channel_data = multi_ch_waveform[ch_idx, :]
            max_amplitude = np.max(np.abs(channel_data))
            tf_data = calib_data["final_tf_list"][ch_idx]
            expected_amplitude = sine_args["amplitude"] / tf_data["amp_ratio"]
            print(
                f"  通道 {ch_idx}: 最大幅值={max_amplitude:.6f}, "
                f"期望幅值={expected_amplitude:.6f}, "
                f"传递函数幅值比={tf_data['amp_ratio']:.6f}"
            )

    @pytest.mark.hardware
    def test_full_calibration_compensation_workflow(
        self, caliber, sampling_info, sine_args, tmp_path
    ):
        """测试完整的校准-补偿工作流程"""
        print("\n=== 测试完整的校准-补偿工作流程 ===")

        # 创建结果文件夹
        result_folder = tmp_path / "full_workflow"

        # 1. 执行校准并自动保存所有结果
        print("\n步骤1: 执行校准并自动保存...")
        caliber.calibrate(
            starts_num=2, chunks_per_start=3, result_folder=result_folder
        )

        # 2. 加载校准数据
        print("\n步骤2: 加载校准数据...")
        calib_file = result_folder / "calib_data.pkl"
        with open(calib_file, "rb") as f:
            calib_data: CalibData = pickle.load(f)

        # 3. 生成多通道补偿波形
        print("\n步骤3: 生成多通道补偿波形...")
        multi_ch_waveform = get_sine_multi_chs(
            sampling_info=sampling_info,
            sine_args=sine_args,
            calib_data=calib_data,
        )

        # 4. 验证结果
        print("\n步骤4: 验证结果...")
        assert multi_ch_waveform.channels_num == len(calib_data["ao_channels"])

        # 5. 打印校准结果摘要
        print("\n校准结果摘要:")
        for ch_idx, tf_data in enumerate(calib_data["final_tf_list"]):
            print(
                f"  通道 {ch_idx} ({calib_data['ao_channels'][ch_idx]}): "
                f"幅值比={tf_data['amp_ratio']:.6f}, "
                f"相位差={tf_data['phase_shift']:.6f}rad "
                f"({tf_data['phase_shift'] * 180 / np.pi:.2f}°)"
            )

        print("\n完整工作流程测试成功！")
