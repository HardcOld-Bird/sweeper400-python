"""
# CaliberOctopus 类测试模块

测试路径：`tests/sweeper400/use/test_caliber.py`

本模块包含对 CaliberOctopus 类的系统化测试，验证其校准功能是否正常工作。
"""

import pytest

from sweeper400.analyze import (
    init_sampling_info,
    init_sine_args,
)
from sweeper400.use import CaliberOctopus


class TestCaliberOctopus:
    """CaliberOctopus 类的测试套件"""

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
        """AI通道配置（实际硬件连接）"""
        # 注意：必须使用Slot2上的AI通道，因为触发信号配置为使用PXI*Slot2/PFI0
        return ("PXI1Slot2/ai0",)

    @pytest.fixture
    def ao_channels(self):
        """AO通道配置（实际硬件连接）"""
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
        result_folder = tmp_path / "calibration_results"
        result_folder.mkdir(parents=True, exist_ok=True)
        return result_folder

    @pytest.mark.hardware
    def test_caliber_initialization(
        self,
        ai_channels,
        ao_channels,
        sampling_info,
        sine_args,
    ):
        """测试CaliberOctopus初始化"""
        caliber = CaliberOctopus(
            ai_channels=ai_channels,
            ao_channels=ao_channels,
            sampling_info=sampling_info,
            sine_args=sine_args,
        )

        # 验证基本属性
        assert caliber.ai_channels == ai_channels
        assert caliber.ao_channels == ao_channels
        assert len(caliber.ao_channels) == 8

        print(f"CaliberOctopus初始化成功，AO通道数: {len(caliber.ao_channels)}")

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
        caliber = CaliberOctopus(
            ai_channels=ai_channels,
            ao_channels=ao_channels,
            sampling_info=sampling_info,
            sine_args=sine_args,
        )

        # 执行校准（使用较少的启动次数和chunk数以加快测试）
        print("\n开始校准流程...")
        caliber.calibrate(
            starts_num=2,
            chunks_per_start=3,
            apply_filter=True,
            result_folder=temp_result_folder,
        )

        # 验证校准结果
        assert caliber.result_final_comp_list is not None
        assert len(caliber.result_final_comp_list) == len(ao_channels)

        print(f"校准完成，生成了 {len(caliber.result_final_comp_list)} 个通道的补偿数据")

        # 验证结果文件已保存
        assert (temp_result_folder / "calib_data.pkl").exists()
        assert (temp_result_folder / "raw_sweep_data.pkl").exists()
        assert (temp_result_folder / "transfer_function_polar.png").exists()
        assert (temp_result_folder / "transfer_function_cartesian.png").exists()

        print(f"所有结果文件已保存到: {temp_result_folder}")



    @pytest.mark.hardware
    def test_calibrate_with_custom_settle_time(
        self,
        ai_channels,
        ao_channels,
        sampling_info,
        sine_args,
        temp_result_folder,
    ):
        """测试使用自定义稳定时间的校准"""
        caliber = CaliberOctopus(
            ai_channels=ai_channels,
            ao_channels=ao_channels,
            sampling_info=sampling_info,
            sine_args=sine_args,
        )

        # 执行校准，使用自定义稳定时间
        custom_settle_time = 0.5  # 0.5秒
        print(f"\n开始校准流程（自定义稳定时间: {custom_settle_time}s）...")
        caliber.calibrate(
            starts_num=1,
            chunks_per_start=2,
            apply_filter=True,
            result_folder=temp_result_folder,
            settle_time=custom_settle_time,
        )

        # 验证校准结果
        assert caliber.result_final_comp_list is not None
        assert len(caliber.result_final_comp_list) == len(ao_channels)

        print("自定义稳定时间校准完成")

    @pytest.mark.hardware
    def test_calibrate_without_filter(
        self,
        ai_channels,
        ao_channels,
        sampling_info,
        sine_args,
        temp_result_folder,
    ):
        """测试不使用滤波的校准"""
        caliber = CaliberOctopus(
            ai_channels=ai_channels,
            ao_channels=ao_channels,
            sampling_info=sampling_info,
            sine_args=sine_args,
        )

        # 执行校准，不使用滤波
        print("\n开始校准流程（不使用滤波）...")
        caliber.calibrate(
            starts_num=1,
            chunks_per_start=2,
            apply_filter=False,
            result_folder=temp_result_folder,
        )

        # 验证校准结果
        assert caliber.result_final_comp_list is not None
        assert len(caliber.result_final_comp_list) == len(ao_channels)

        print("不使用滤波的校准完成")

    @pytest.mark.hardware
    def test_save_and_load_calib_data(
        self,
        ai_channels,
        ao_channels,
        sampling_info,
        sine_args,
        temp_result_folder,
    ):
        """测试保存和加载校准数据"""
        # 第一步：执行校准并保存
        caliber1 = CaliberOctopus(
            ai_channels=ai_channels,
            ao_channels=ao_channels,
            sampling_info=sampling_info,
            sine_args=sine_args,
        )

        print("\n执行校准...")
        caliber1.calibrate(
            starts_num=1,
            chunks_per_start=2,
            apply_filter=True,
            result_folder=None,  # 不自动保存
        )

        # 手动保存校准数据
        calib_data_path = temp_result_folder / "test_calib_data.pkl"
        caliber1.save_calib_data(calib_data_path)
        print(f"校准数据已保存到: {calib_data_path}")

        # 验证文件存在
        assert calib_data_path.exists()

        # 第二步：使用保存的校准数据创建新的CaliberOctopus实例
        print("\n使用校准数据创建新实例...")
        caliber2 = CaliberOctopus(
            ai_channels=ai_channels,
            ao_channels=ao_channels,
            sampling_info=sampling_info,
            sine_args=sine_args,
            calib_data=calib_data_path,
        )

        # 验证新实例的输出波形已经过补偿
        assert caliber2._output_waveform.channels_num == len(ao_channels)
        print(
            f"使用校准数据创建实例成功，输出波形通道数: {caliber2._output_waveform.channels_num}"
        )

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
        caliber = CaliberOctopus(
            ai_channels=ai_channels,
            ao_channels=ao_channels,
            sampling_info=sampling_info,
            sine_args=sine_args,
        )

        # 执行校准
        print("\n执行校准...")
        caliber.calibrate(
            starts_num=1,
            chunks_per_start=2,
            apply_filter=True,
            result_folder=None,
        )

        # 测试极坐标模式绘图
        polar_path = temp_result_folder / "test_polar.png"
        caliber.plot_transfer_functions(mode="polar", save_path=polar_path)
        assert polar_path.exists()
        print(f"极坐标图已保存到: {polar_path}")

        # 测试直角坐标模式绘图
        cartesian_path = temp_result_folder / "test_cartesian.png"
        caliber.plot_transfer_functions(mode="cartesian", save_path=cartesian_path)
        assert cartesian_path.exists()
        print(f"直角坐标图已保存到: {cartesian_path}")

    @pytest.mark.hardware
    def test_result_properties(
        self,
        ai_channels,
        ao_channels,
        sampling_info,
        sine_args,
    ):
        """测试结果属性访问"""
        caliber = CaliberOctopus(
            ai_channels=ai_channels,
            ao_channels=ao_channels,
            sampling_info=sampling_info,
            sine_args=sine_args,
        )

        # 校准前，结果应为None
        assert caliber.result_raw_sweep_data is None
        assert caliber.result_raw_tf_list is None
        assert caliber.result_final_comp_list is None

        # 执行校准（使用临时文件夹避免污染默认路径）
        print("\n执行校准...")
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            caliber.calibrate(
                starts_num=1,
                chunks_per_start=2,
                apply_filter=True,
                result_folder=tmpdir,
            )

        # 校准后，结果应不为None
        assert caliber.result_raw_sweep_data is not None
        assert caliber.result_final_comp_list is not None
        assert len(caliber.result_final_comp_list) == len(ao_channels)

        print("结果属性访问测试通过")

    @pytest.mark.hardware
    def test_calibrate_with_default_result_folder(
        self,
        ai_channels,
        ao_channels,
        sampling_info,
        sine_args,
    ):
        """测试使用默认result_folder的校准"""
        from pathlib import Path

        caliber = CaliberOctopus(
            ai_channels=ai_channels,
            ao_channels=ao_channels,
            sampling_info=sampling_info,
            sine_args=sine_args,
        )

        # 执行校准，不指定result_folder（应使用默认路径）
        print("\n开始校准流程（使用默认result_folder）...")
        caliber.calibrate(
            starts_num=1,
            chunks_per_start=2,
            apply_filter=True,
            result_folder=None,  # 不指定，使用默认路径
        )

        # 验证校准结果
        assert caliber.result_final_comp_list is not None
        assert len(caliber.result_final_comp_list) == len(ao_channels)

        # 验证默认路径下的文件已保存
        # 默认路径应该是项目根目录的 storage/calib/calib_result_octopus
        default_path = Path(__file__).resolve().parents[3] / "storage" / "calib" / "calib_result_octopus"

        assert default_path.exists(), f"默认路径不存在: {default_path}"
        assert (default_path / "calib_data.pkl").exists(), "calib_data.pkl未保存"
        assert (default_path / "raw_sweep_data.pkl").exists(), "raw_sweep_data.pkl未保存"
        assert (default_path / "transfer_function_polar.png").exists(), "polar图未保存"
        assert (default_path / "transfer_function_cartesian.png").exists(), "cartesian图未保存"

        print(f"默认路径校准完成，所有文件已保存到: {default_path}")
