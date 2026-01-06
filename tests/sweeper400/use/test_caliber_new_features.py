"""
# Caliber 类新功能测试模块

测试路径：`tests/sweeper400/use/test_caliber_new_features.py`

本模块包含对 Caliber 类新功能的测试：
1. calib_data参数：使用校准数据文件初始化Caliber对象
2. ex_calibrate方法：多次独立校准并平均结果
"""

import pickle
from pathlib import Path

import pytest

from sweeper400.analyze import init_sampling_info, init_sine_args
from sweeper400.use import Caliber


class TestCaliberNewFeatures:
    """Caliber 类新功能的测试套件"""

    @pytest.fixture
    def sampling_info(self):
        """创建测试用的采样信息"""
        return init_sampling_info(171500.0, 85750)  # 采样率171.5kHz, 0.5秒

    @pytest.fixture
    def sine_args(self):
        """创建测试用的正弦波参数"""
        return init_sine_args(frequency=3430.0, amplitude=0.01, phase=0.0)

    @pytest.fixture
    def ai_channel(self):
        """AI通道配置"""
        return "PXI2Slot2/ai0"

    @pytest.fixture
    def ao_channels(self):
        """AO通道配置（8个通道）"""
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

    @pytest.mark.hardware
    def test_caliber_with_calib_data(
        self, ai_channel, ao_channels, sampling_info, sine_args, tmp_path
    ):
        """测试使用calib_data参数初始化Caliber对象"""
        print("\n=== 测试calib_data参数功能 ===")

        # 步骤1: 先执行一次校准并保存CalibData
        print("\n步骤1: 执行初始校准...")
        caliber1 = Caliber(
            ai_channel=ai_channel,
            ao_channels=ao_channels,
            sampling_info=sampling_info,
            sine_args=sine_args,
        )
        caliber1.calibrate(starts_num=2, chunks_per_start=3)

        # 保存CalibData
        calib_data_file = tmp_path / "test_calib_data.pkl"
        caliber1.save_calib_data(calib_data_file)
        print(f"CalibData已保存到: {calib_data_file}")

        # 步骤2: 使用calib_data参数创建新的Caliber对象
        print("\n步骤2: 使用calib_data参数创建新的Caliber对象...")
        caliber2 = Caliber(
            ai_channel=ai_channel,
            ao_channels=ao_channels,
            sampling_info=sampling_info,
            sine_args=sine_args,
            calib_data=calib_data_file,
        )

        # 验证输出波形是多通道的
        assert caliber2._output_waveform.ndim == 2
        assert caliber2._output_waveform.shape[0] == len(ao_channels)
        print(
            f"输出波形shape: {caliber2._output_waveform.shape} "
            f"(预期: ({len(ao_channels)}, {sampling_info['samples_num']}))"
        )

        print("\n=== calib_data参数功能测试完成 ===")

    @pytest.mark.hardware
    def test_ex_calibrate_method(
        self, ai_channel, ao_channels, sampling_info, sine_args, tmp_path
    ):
        """测试ex_calibrate方法"""
        print("\n=== 测试ex_calibrate方法 ===")

        # 创建Caliber对象
        caliber = Caliber(
            ai_channel=ai_channel,
            ao_channels=ao_channels,
            sampling_info=sampling_info,
            sine_args=sine_args,
        )

        # 执行扩展校准（3次独立校准，每次2次启动，每次启动3个chunk）
        result_folder = tmp_path / "ex_calibrate_results"
        print(f"\n执行扩展校准，结果将保存到: {result_folder}")

        caliber.ex_calibrate(
            ex_starts_num=3,
            starts_num=2,
            chunks_per_start=3,
            result_folder=result_folder,
        )

        # 验证结果文件
        calib_data_file = result_folder / "calib_data.pkl"
        averaged_plot_file = result_folder / "transfer_function_averaged.png"
        detailed_plot_file = result_folder / "transfer_function_detailed.png"

        assert calib_data_file.exists(), "CalibData文件不存在"
        assert averaged_plot_file.exists(), "averaged模式绘图不存在"
        assert detailed_plot_file.exists(), "detailed模式绘图不存在"

        # 验证临时文件夹已被删除
        temp_folder = result_folder / "temp_calib_data"
        assert not temp_folder.exists(), "临时文件夹未被删除"

        # 验证CalibData内容
        with open(calib_data_file, "rb") as f:
            calib_data = pickle.load(f)

        assert len(calib_data["tf_list"]) == len(ao_channels)
        assert calib_data["ao_channels"] == ao_channels
        assert calib_data["ai_channel"] == ai_channel

        print("\n扩展校准结果:")
        for idx, tf_data in enumerate(calib_data["tf_list"]):
            print(
                f"  通道 {idx} ({ao_channels[idx]}): "
                f"幅值比={tf_data['amp_ratio']:.6f}, "
                f"相位差={tf_data['phase_shift']:.6f}rad"
            )

        print(f"\n最终CalibData文件: {calib_data_file}")
        print(f"averaged模式绘图: {averaged_plot_file}")
        print(f"detailed模式绘图: {detailed_plot_file}")

        print("\n=== ex_calibrate方法测试完成 ===")
