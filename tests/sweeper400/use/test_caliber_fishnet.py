"""
# CaliberFishNet 类测试模块

测试路径：`tests/sweeper400/use/test_caliber_fishnet.py`

本模块包含对 CaliberFishNet 类的系统化测试，验证其多通道校准功能是否正常工作。
"""

import pytest

from sweeper400.analyze import (
    init_sampling_info,
    init_sine_args,
)
from sweeper400.use.caliber import CaliberFishNet


class TestCaliberFishNet:
    """CaliberFishNet 类的测试套件"""

    @pytest.fixture
    def sampling_info(self):
        """创建测试用的采样信息"""
        # 使用较短的采样时间以加快测试速度
        return init_sampling_info(48000.0, 4800)  # 0.1秒

    @pytest.fixture
    def sine_args(self):
        """创建测试用的正弦波参数"""
        # 使用1000Hz的正弦波，幅值0.01V
        return init_sine_args(frequency=1000.0, amplitude=0.01, phase=0.0)

    @pytest.fixture
    def ai_channels(self):
        """AI通道配置（实际硬件连接 - 9个麦克风）"""
        return (
            "PXI1Slot2/ai0",
            "PXI2Slot2/ai0",
            "PXI2Slot2/ai1",
            "PXI2Slot3/ai0",
            "PXI2Slot3/ai1",
            "PXI3Slot2/ai0",
            "PXI3Slot2/ai1",
            "PXI3Slot3/ai0",
            "PXI3Slot3/ai1",
        )

    @pytest.fixture
    def ao_channels(self):
        """AO通道配置（实际硬件连接 - 8个扬声器）"""
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
    def temp_result_folder(self, tmp_path):
        """创建临时结果文件夹"""
        result_folder = tmp_path / "calibration_results_fishnet"
        result_folder.mkdir(parents=True, exist_ok=True)
        return result_folder

    @pytest.mark.hardware
    def test_fishnet_initialization(
        self,
        ai_channels,
        ao_channels,
        sampling_info,
        sine_args,
    ):
        """测试CaliberFishNet初始化"""
        caliber = CaliberFishNet(
            ai_channels=ai_channels,
            ao_channels=ao_channels,
            sampling_info=sampling_info,
            sine_args=sine_args,
        )

        # 验证基本属性
        assert len(caliber.ai_channels) == 9
        assert len(caliber.ao_channels) == 8

        print(
            f"CaliberFishNet初始化成功，"
            f"AI通道数: {len(caliber.ai_channels)}, "
            f"AO通道数: {len(caliber.ao_channels)}"
        )

    @pytest.mark.hardware
    def test_calibrate_basic(
        self,
        ai_channels,
        ao_channels,
        sampling_info,
        sine_args,
        temp_result_folder,
    ):
        """测试基本校准流程"""
        caliber = CaliberFishNet(
            ai_channels=ai_channels,
            ao_channels=ao_channels,
            sampling_info=sampling_info,
            sine_args=sine_args,
        )

        # 执行校准（使用较少的启动次数和chunk数以加快测试）
        caliber.calibrate(
            starts_num=2,  # 使用2次独立校准
            chunks_per_start=2,  # 每次启动2个chunk
            apply_filter=True,
            lowcut=100.0,
            highcut=20000.0,
            result_folder=temp_result_folder,
        )

        # 验证结果
        assert caliber.result_averaged_tf_data is not None
        tf_data = caliber.result_averaged_tf_data

        # 验证TFData结构
        assert "tf_list" in tf_data
        assert "sine_args" in tf_data
        assert "mean_amp_ratio" in tf_data
        assert "mean_phase_shift" in tf_data
        assert "ao_channels" in tf_data
        assert "ai_channels" in tf_data
        assert "sampling_info" in tf_data

        # 验证传递函数数量 = 9 AI × 8 AO = 72
        assert len(tf_data["tf_list"]) == 72

        print(f"校准完成，共计算{len(tf_data['tf_list'])}个传递函数")
        print(f"平均幅值比: {tf_data['mean_amp_ratio']:.6f}")
        print(f"平均相位差: {tf_data['mean_phase_shift']:.6f}rad")

        # 验证文件已保存
        assert (temp_result_folder / "tf_data.pkl").exists()
        assert (temp_result_folder / "transfer_functions.png").exists()
        assert (temp_result_folder / "raw_sweep_data_1.pkl").exists()
        assert (temp_result_folder / "raw_sweep_data_2.pkl").exists()




    @pytest.mark.hardware
    def test_position_encoding(
        self,
        ai_channels,
        ao_channels,
        sampling_info,
        sine_args,
        temp_result_folder,
    ):
        """测试position坐标编码是否正确"""
        caliber = CaliberFishNet(
            ai_channels=ai_channels,
            ao_channels=ao_channels,
            sampling_info=sampling_info,
            sine_args=sine_args,
        )

        # 执行校准
        caliber.calibrate(
            starts_num=1,
            chunks_per_start=1,
            result_folder=temp_result_folder,
        )

        tf_data = caliber.result_averaged_tf_data
        assert tf_data is not None

        # 验证每个传递函数的position编码
        for tf_point in tf_data["tf_list"]:
            ao_idx = int(tf_point["position"].x)
            ai_idx = int(tf_point["position"].y)

            # 验证索引范围
            assert 0 <= ao_idx < 8, f"AO索引{ao_idx}超出范围"
            assert 0 <= ai_idx < 9, f"AI索引{ai_idx}超出范围"

        # 验证所有通道对都被测量
        measured_pairs = set()
        for tf_point in tf_data["tf_list"]:
            ao_idx = int(tf_point["position"].x)
            ai_idx = int(tf_point["position"].y)
            measured_pairs.add((ao_idx, ai_idx))

        # 应该有8×9=72个不同的通道对
        assert len(measured_pairs) == 72

        print(f"所有{len(measured_pairs)}个通道对的position编码验证通过")

    @pytest.mark.hardware
    def test_save_and_load_tf_data(
        self,
        ai_channels,
        ao_channels,
        sampling_info,
        sine_args,
        temp_result_folder,
    ):
        """测试TFData的保存和加载"""
        import pickle

        caliber = CaliberFishNet(
            ai_channels=ai_channels,
            ao_channels=ao_channels,
            sampling_info=sampling_info,
            sine_args=sine_args,
        )

        # 执行校准
        caliber.calibrate(
            starts_num=1,
            chunks_per_start=1,
            result_folder=temp_result_folder,
        )

        # 保存TFData到自定义路径
        custom_save_path = temp_result_folder / "custom_tf_data.pkl"
        caliber.save_tf_data(custom_save_path)

        # 验证文件存在
        assert custom_save_path.exists()

        # 加载并验证
        with open(custom_save_path, "rb") as f:
            loaded_tf_data = pickle.load(f)

        assert len(loaded_tf_data["tf_list"]) == 72
        assert loaded_tf_data["ao_channels"] == ao_channels
        assert loaded_tf_data["ai_channels"] == ai_channels

        print("TFData保存和加载测试通过")

    @pytest.mark.hardware
    def test_plot_transfer_functions(
        self,
        ai_channels,
        ao_channels,
        sampling_info,
        sine_args,
        temp_result_folder,
    ):
        """测试传递函数绘图功能"""
        caliber = CaliberFishNet(
            ai_channels=ai_channels,
            ao_channels=ao_channels,
            sampling_info=sampling_info,
            sine_args=sine_args,
        )

        # 执行校准
        caliber.calibrate(
            starts_num=1,
            chunks_per_start=1,
            result_folder=temp_result_folder,
        )

        # 绘制并保存图表
        plot_path = temp_result_folder / "test_plot.png"
        caliber.plot_transfer_functions(save_path=plot_path)

        # 验证图像文件存在
        assert plot_path.exists()

        print(f"传递函数图已保存到: {plot_path}")
