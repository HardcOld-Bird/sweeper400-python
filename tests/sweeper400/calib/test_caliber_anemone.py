"""
# CaliberAnemone 类测试模块

测试路径：`tests/sweeper400/calib/test_caliber_anemone.py`

本模块包含对 CaliberAnemone 类的系统化测试，验证其校准功能是否正常工作。
"""

import pytest

from sweeper400.analyze import init_sampling_info
from sweeper400.calib import CaliberAnemone


class TestCaliberAnemone:
    """CaliberAnemone 类的测试套件"""

    @pytest.fixture
    def sampling_info(self):
        """创建测试用的采样信息"""
        # 使用较短的采样时间以加快测试速度
        return init_sampling_info(48000.0, 4800)  # 0.1秒

    @pytest.fixture
    def ai_channels(self):
        """AI通道配置（实际硬件连接）"""
        # 使用Slot2上的AI通道，与虚拟AO通道对应
        return (
            "PXI1Slot2/ai0",
            "PXI1Slot2/ai1",
        )

    @pytest.fixture
    def temp_result_folder(self, tmp_path):
        """创建临时结果文件夹"""
        result_folder = tmp_path / "calibration_results"
        result_folder.mkdir(parents=True, exist_ok=True)
        return result_folder

    def test_infer_dummy_ao_channel(self):
        """测试虚拟AO通道推断逻辑"""
        assert CaliberAnemone._infer_dummy_ao_channel("PXI1Slot2/ai0") == "PXI1Slot2/ao0"
        assert CaliberAnemone._infer_dummy_ao_channel("PXI1Slot3/ai1") == "PXI1Slot3/ao1"
        assert CaliberAnemone._infer_dummy_ao_channel("Dev1/ai0") == "Dev1/ao0"

    def test_caliber_initialization(self, ai_channels, sampling_info):
        """测试CaliberAnemone初始化（无需硬件）"""
        caliber = CaliberAnemone(
            ai_channels=ai_channels,
            sampling_info=sampling_info,
            frequency=1000.0,
        )

        # 验证基本属性
        assert caliber.ai_channels == ai_channels
        assert caliber.frequency == 1000.0
        assert len(caliber.ai_channels) == 2
        assert caliber.result_comp_data is None
        assert caliber.result_tf_data is None

        print(f"CaliberAnemone初始化成功，AI通道数: {len(caliber.ai_channels)}")

    def test_caliber_initialization_with_custom_frequency(self, ai_channels, sampling_info):
        """测试使用自定义频率的初始化"""
        caliber = CaliberAnemone(
            ai_channels=ai_channels,
            sampling_info=sampling_info,
            frequency=2000.0,
        )

        assert caliber.frequency == 2000.0
        print("自定义频率初始化测试通过")

    def test_caliber_invalid_parameters(self, sampling_info):
        """测试无效参数的异常处理"""
        # 空AI通道列表
        with pytest.raises(ValueError, match="AI 通道列表不能为空"):
            CaliberAnemone(
                ai_channels=(),
                sampling_info=sampling_info,
            )

        # 无效频率
        with pytest.raises(ValueError, match="频率必须为正数"):
            CaliberAnemone(
                ai_channels=("PXI1Slot2/ai0",),
                sampling_info=sampling_info,
                frequency=0.0,
            )

        with pytest.raises(ValueError, match="频率必须为正数"):
            CaliberAnemone(
                ai_channels=("PXI1Slot2/ai0",),
                sampling_info=sampling_info,
                frequency=-100.0,
            )

        print("无效参数异常处理测试通过")

    @pytest.mark.hardware
    def test_calibrate_basic(
        self,
        ai_channels,
        sampling_info,
        temp_result_folder,
    ):
        """测试基本校准流程（需要硬件）"""
        caliber = CaliberAnemone(
            ai_channels=ai_channels,
            sampling_info=sampling_info,
            frequency=1000.0,
        )

        # 执行校准（使用较少的chunk数以加快测试）
        print("\n开始校准流程...")
        caliber.calibrate(
            chunks_num=2,
            apply_filter=True,
            result_folder=temp_result_folder,
        )

        # 验证校准结果
        assert caliber.result_comp_data is not None
        assert caliber.result_tf_data is not None

        comp_df = caliber.result_comp_data["comp_dataframe"]
        assert len(comp_df) == len(ai_channels)
        assert "amp_multiplier" in comp_df.columns
        assert "time_increment" in comp_df.columns

        print(f"校准完成，生成了 {len(comp_df)} 个通道的补偿数据")

        # 验证结果文件已保存
        assert (temp_result_folder / "ai_comp_data.pkl").exists()
        assert (temp_result_folder / "raw_sweep_data.pkl").exists()
        assert (temp_result_folder / "transfer_function_polar.png").exists()
        assert (temp_result_folder / "compensation_cartesian.png").exists()
        assert (temp_result_folder / "fusion_waveform.png").exists()

        print(f"所有结果文件已保存到: {temp_result_folder}")

    @pytest.mark.hardware
    def test_calibrate_without_filter(
        self,
        ai_channels,
        sampling_info,
        temp_result_folder,
    ):
        """测试不使用滤波的校准（需要硬件）"""
        caliber = CaliberAnemone(
            ai_channels=ai_channels,
            sampling_info=sampling_info,
            frequency=1000.0,
        )

        print("\n开始校准流程（不使用滤波）...")
        caliber.calibrate(
            chunks_num=2,
            apply_filter=False,
            result_folder=temp_result_folder,
        )

        assert caliber.result_comp_data is not None
        assert caliber.result_tf_data is not None

        print("不使用滤波的校准完成")

    @pytest.mark.hardware
    def test_save_and_load_comp_data(
        self,
        ai_channels,
        sampling_info,
        temp_result_folder,
    ):
        """测试保存和加载补偿数据（需要硬件）"""
        # 第一步：执行校准并保存
        caliber1 = CaliberAnemone(
            ai_channels=ai_channels,
            sampling_info=sampling_info,
            frequency=1000.0,
        )

        print("\n执行校准...")
        caliber1.calibrate(
            chunks_num=2,
            apply_filter=True,
            result_folder=temp_result_folder,
        )

        # 手动保存补偿数据
        comp_data_path = temp_result_folder / "test_comp_data.pkl"
        caliber1.save_comp_data(comp_data_path)
        print(f"补偿数据已保存到: {comp_data_path}")

        # 验证文件存在
        assert comp_data_path.exists()

        # 第二步：使用保存的补偿数据创建新的CaliberAnemone实例
        print("\n使用补偿数据创建新实例...")
        caliber2 = CaliberAnemone(
            ai_channels=ai_channels,
            sampling_info=sampling_info,
            frequency=1000.0,
            ai_comp_data=comp_data_path,
        )

        # 验证新实例已加载补偿数据
        assert caliber2._loaded_ai_comp_data is not None
        print("使用补偿数据创建实例成功")

    @pytest.mark.hardware
    def test_plot_comp_data(
        self,
        ai_channels,
        sampling_info,
        temp_result_folder,
    ):
        """测试绘图功能（需要硬件）"""
        caliber = CaliberAnemone(
            ai_channels=ai_channels,
            sampling_info=sampling_info,
            frequency=1000.0,
        )

        print("\n执行校准...")
        caliber.calibrate(
            chunks_num=2,
            apply_filter=True,
            result_folder=temp_result_folder,
        )

        # 测试极坐标模式绘图
        polar_path = temp_result_folder / "test_polar.png"
        caliber.plot_comp_data(mode="polar", save_path=polar_path)
        assert polar_path.exists()
        print(f"极坐标图已保存到: {polar_path}")

        # 测试直角坐标模式绘图
        cartesian_path = temp_result_folder / "test_cartesian.png"
        caliber.plot_comp_data(mode="cartesian", save_path=cartesian_path)
        assert cartesian_path.exists()
        print(f"直角坐标图已保存到: {cartesian_path}")

    @pytest.mark.hardware
    def test_calibrate_with_default_result_folder(
        self,
        ai_channels,
        sampling_info,
    ):
        """测试使用默认result_folder的校准（需要硬件）"""
        from pathlib import Path

        caliber = CaliberAnemone(
            ai_channels=ai_channels,
            sampling_info=sampling_info,
            frequency=1000.0,
        )

        # 执行校准，不指定result_folder（应使用默认路径）
        print("\n开始校准流程（使用默认result_folder）...")
        caliber.calibrate(
            chunks_num=2,
            apply_filter=True,
            result_folder=None,
        )

        # 验证校准结果
        assert caliber.result_comp_data is not None
        assert caliber.result_tf_data is not None

        # 验证默认路径下的文件已保存
        default_path = (
            Path(__file__).resolve().parents[3]
            / "storage"
            / "calib"
            / "calib_result_anemone"
        )

        assert default_path.exists(), f"默认路径不存在: {default_path}"
        assert (default_path / "ai_comp_data.pkl").exists(), "ai_comp_data.pkl未保存"
        assert (default_path / "raw_sweep_data.pkl").exists(), "raw_sweep_data.pkl未保存"
        assert (default_path / "transfer_function_polar.png").exists(), "polar图未保存"
        assert (default_path / "compensation_cartesian.png").exists(), "cartesian图未保存"

        print(f"默认路径校准完成，所有文件已保存到: {default_path}")
