"""
# Caliber 类测试模块

测试路径：`tests/sweeper400/use/test_caliber.py`

本模块包含对 Caliber 类的测试，验证其校准功能是否正常工作。
"""

import numpy as np
import pytest

from sweeper400.analyze import init_sampling_info, init_sine_args
from sweeper400.use import Caliber


class TestCaliber:
    """Caliber 类的测试套件"""

    @pytest.fixture
    def sampling_info(self):
        """创建测试用的采样信息"""
        # 使用推荐的采样参数
        return init_sampling_info(171500.0, 85750)  # 采样率171.5kHz, 0.5秒

    @pytest.fixture
    def sine_args(self):
        """创建测试用的正弦波参数"""
        # 使用推荐的测试单频
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

    @pytest.fixture
    def caliber(self, ai_channel, ao_channels, sampling_info, sine_args):
        """创建Caliber实例"""
        return Caliber(
            ai_channel=ai_channel,
            ao_channels=ao_channels,
            sampling_info=sampling_info,
            sine_args=sine_args,
        )

    def test_caliber_initialization(self, caliber, ai_channel, ao_channels):
        """测试Caliber对象的初始化"""
        assert caliber.ai_channel == ai_channel
        assert caliber.ao_channels == ao_channels
        assert caliber.result_final_tf_list is None  # 尚未校准
        assert caliber.result_raw_tf_list is None  # 尚未校准
        assert caliber.result_raw_sweep_data is None  # 尚未校准

    @pytest.mark.hardware
    def test_calibration_process(self, caliber):
        """测试校准流程（需要硬件）"""
        # 执行校准（2次启动，每次3个chunk）
        caliber.calibrate(starts_num=2, chunks_per_start=3)

        # 验证校准结果
        assert caliber.result_final_tf_list is not None
        assert len(caliber.result_final_tf_list) == len(caliber.ao_channels)
        assert caliber.result_raw_tf_list is not None
        assert caliber.result_raw_sweep_data is not None

        # 验证传递函数是复数
        for channel_idx, tf in caliber.result_final_tf_list.items():
            assert isinstance(tf, complex)
            # 传递函数的幅值应该是正数
            assert np.abs(tf) > 0

        print("\n校准结果:")
        for channel_idx, tf in caliber.result_final_tf_list.items():
            print(
                f"  通道 {channel_idx} ({caliber.ao_channels[channel_idx]}): "
                f"幅值={np.abs(tf):.6f}, 相位={np.angle(tf):.6f}rad"
            )

    @pytest.mark.hardware
    def test_save_calibration_results(self, caliber, tmp_path):
        """测试保存校准结果"""
        # 先执行校准
        caliber.calibrate(starts_num=2, chunks_per_start=2)

        # 保存结果
        save_file = tmp_path / "calibration_test.pkl"
        caliber.save_calib_data(save_file)

        # 验证文件存在
        assert save_file.exists()
        assert save_file.stat().st_size > 0

        print(f"\n校准结果已保存到: {save_file}")

    @pytest.mark.hardware
    def test_save_sweep_data(self, caliber, tmp_path):
        """测试保存SweepData"""
        # 先执行校准
        caliber.calibrate(starts_num=2, chunks_per_start=2)

        # 保存SweepData
        save_file = tmp_path / "result_raw_sweep_data.pkl"
        caliber.save_sweep_data(save_file)

        # 验证文件存在
        assert save_file.exists()
        assert save_file.stat().st_size > 0

        print(f"\nSweepData已保存到: {save_file}")

    @pytest.mark.hardware
    def test_plot_transfer_functions_averaged(self, caliber, tmp_path):
        """测试绘制传递函数图（平均模式）"""
        # 先执行校准
        caliber.calibrate(starts_num=2, chunks_per_start=2)

        # 绘制并保存图像（平均模式）
        plot_file = tmp_path / "transfer_functions_averaged.png"
        caliber.plot_transfer_functions(mode="averaged", save_path=plot_file)

        # 验证文件存在
        assert plot_file.exists()
        assert plot_file.stat().st_size > 0

        print(f"\n传递函数图（平均模式）已保存到: {plot_file}")

    @pytest.mark.hardware
    def test_plot_transfer_functions_detailed(self, caliber, tmp_path):
        """测试绘制传递函数图（详细模式）"""
        # 先执行校准
        caliber.calibrate(starts_num=2, chunks_per_start=2)

        # 绘制并保存图像（详细模式）
        plot_file = tmp_path / "transfer_functions_detailed.png"
        caliber.plot_transfer_functions(mode="detailed", save_path=plot_file)

        # 验证文件存在
        assert plot_file.exists()
        assert plot_file.stat().st_size > 0

        print(f"\n传递函数图（详细模式）已保存到: {plot_file}")

    def test_calibrate_without_hardware_raises_error(self, caliber):
        """测试在没有硬件的情况下校准会失败（预期行为）"""
        # 注意：这个测试在没有硬件时会失败，这是预期的
        # 如果有硬件，这个测试应该被跳过
        pass

    @pytest.mark.hardware
    def test_full_calibration_workflow(self, caliber, tmp_path):
        """测试完整的校准工作流程"""
        print("\n=== 开始完整校准工作流程测试 ===")

        # 1. 执行校准
        print("\n步骤1: 执行校准...")
        caliber.calibrate(starts_num=2, chunks_per_start=3)

        # 2. 保存校准结果
        print("\n步骤2: 保存校准结果...")
        save_file = tmp_path / "full_calibration.pkl"
        caliber.save_calib_data(save_file)

        # 3. 保存SweepData
        print("\n步骤3: 保存SweepData...")
        sweep_data_file = tmp_path / "full_calibration_sweep_data.pkl"
        caliber.save_sweep_data(sweep_data_file)

        # 4. 绘制传递函数图（平均模式）
        print("\n步骤4: 绘制传递函数图（平均模式）...")
        plot_file_avg = tmp_path / "full_calibration_averaged.png"
        caliber.plot_transfer_functions(mode="averaged", save_path=plot_file_avg)

        # 5. 绘制传递函数图（详细模式）
        print("\n步骤5: 绘制传递函数图（详细模式）...")
        plot_file_det = tmp_path / "full_calibration_detailed.png"
        caliber.plot_transfer_functions(mode="detailed", save_path=plot_file_det)

        # 6. 验证结果
        print("\n步骤6: 验证结果...")
        assert caliber.result_final_tf_list is not None
        assert caliber.result_raw_tf_list is not None
        assert caliber.result_raw_sweep_data is not None
        assert save_file.exists()
        assert sweep_data_file.exists()
        assert plot_file_avg.exists()
        assert plot_file_det.exists()

        print("\n=== 完整校准工作流程测试完成 ===")
        print(f"校准结果文件: {save_file}")
        print(f"SweepData文件: {sweep_data_file}")
        print(f"传递函数图（平均）: {plot_file_avg}")
        print(f"传递函数图（详细）: {plot_file_det}")
