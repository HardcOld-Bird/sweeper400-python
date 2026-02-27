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
        assert caliber.result_final_comp_data is not None
        assert len(caliber.result_final_comp_data["comp_list"]) == len(ao_channels)

        print(f"校准完成，生成了 {len(caliber.result_final_comp_data['comp_list'])} 个通道的补偿数据")

        # 验证结果文件已保存
        assert (temp_result_folder / "comp_data.pkl").exists()
        assert (temp_result_folder / "raw_sweep_data_1.pkl").exists()
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
        assert caliber.result_final_comp_data is not None
        assert len(caliber.result_final_comp_data["comp_list"]) == len(ao_channels)

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
        assert caliber.result_final_comp_data is not None
        assert len(caliber.result_final_comp_data["comp_list"]) == len(ao_channels)

        print("不使用滤波的校准完成")

    @pytest.mark.hardware
    def test_save_and_load_comp_data(
        self,
        ai_channels,
        ao_channels,
        sampling_info,
        sine_args,
        temp_result_folder,
    ):
        """测试保存和加载补偿数据"""
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

        # 手动保存补偿数据
        comp_data_path = temp_result_folder / "test_comp_data.pkl"
        caliber1.save_comp_data(comp_data_path)
        print(f"补偿数据已保存到: {comp_data_path}")

        # 验证文件存在
        assert comp_data_path.exists()

        # 第二步：使用保存的补偿数据创建新的CaliberOctopus实例
        print("\n使用补偿数据创建新实例...")
        caliber2 = CaliberOctopus(
            ai_channels=ai_channels,
            ao_channels=ao_channels,
            sampling_info=sampling_info,
            sine_args=sine_args,
            comp_data=comp_data_path,
        )

        # 验证新实例的输出波形已经过补偿
        assert caliber2._output_waveform.channels_num == len(ao_channels)
        print(
            f"使用补偿数据创建实例成功，输出波形通道数: {caliber2._output_waveform.channels_num}"
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
        assert caliber.result_final_comp_data is None

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
        assert caliber.result_final_comp_data is not None
        assert len(caliber.result_final_comp_data["comp_list"]) == len(ao_channels)

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
        assert caliber.result_final_comp_data is not None
        assert len(caliber.result_final_comp_data["comp_list"]) == len(ao_channels)

        # 验证默认路径下的文件已保存
        # 默认路径应该是项目根目录的 storage/calib/calib_result_octopus
        default_path = Path(__file__).resolve().parents[3] / "storage" / "calib" / "calib_result_octopus"

        assert default_path.exists(), f"默认路径不存在: {default_path}"
        assert (default_path / "comp_data.pkl").exists(), "comp_data.pkl未保存"
        assert (default_path / "raw_sweep_data_1.pkl").exists(), "raw_sweep_data_1.pkl未保存"
        assert (default_path / "transfer_function_polar.png").exists(), "polar图未保存"
        assert (default_path / "transfer_function_cartesian.png").exists(), "cartesian图未保存"

        print(f"默认路径校准完成，所有文件已保存到: {default_path}")

    @pytest.mark.hardware
    def test_calibration_improvement_workflow(
        self,
        ai_channels,
        ao_channels,
        sampling_info,
        sine_args,
        temp_result_folder,
    ):
        """
        测试完整的校准改善流程（参考scripts/6chs_calib_test.py）

        流程：
        1. 第一次校准（无补偿）- 观察各通道的不一致性
        2. 使用第一次的comp_data进行第二次校准 - 观察通道一致性的改善
        3. 验证补偿效果
        """
        # ============================================================
        # 第一阶段：初始校准（无补偿）
        # ============================================================
        print("\n" + "=" * 60)
        print("第一阶段：初始校准（无补偿）")
        print("=" * 60)

        before_calib_folder = temp_result_folder / "before_calib"
        before_calib_folder.mkdir(parents=True, exist_ok=True)

        caliber_before = CaliberOctopus(
            ai_channel=ai_channels[0],  # 使用单个AI通道
            ao_channels=ao_channels,
            sampling_info=sampling_info,
            sine_args=sine_args,
        )

        # 执行初始校准
        print("\n开始初始校准...")
        caliber_before.calibrate(
            starts_num=2,
            chunks_per_start=3,
            apply_filter=True,
            result_folder=before_calib_folder,
        )

        # 验证初始校准结果
        assert caliber_before.result_final_comp_data is not None
        assert len(caliber_before.result_final_comp_data["comp_list"]) == len(ao_channels)
        assert (before_calib_folder / "comp_data.pkl").exists()

        # 获取初始校准的传递函数数据（用于后续对比）
        tf_data_before = caliber_before.result_averaged_tf_data
        assert tf_data_before is not None

        # 计算初始校准的幅值比标准差（衡量通道不一致性）
        amp_ratios_before = [point["amp_ratio"] for point in tf_data_before["tf_list"]]
        import numpy as np
        amp_std_before = np.std(amp_ratios_before)
        amp_mean_before = np.mean(amp_ratios_before)

        print(f"\n初始校准结果:")
        print(f"  幅值比平均值: {amp_mean_before:.6f}")
        print(f"  幅值比标准差: {amp_std_before:.6f}")
        print(f"  幅值比范围: [{min(amp_ratios_before):.6f}, {max(amp_ratios_before):.6f}]")

        # ============================================================
        # 第二阶段：使用补偿数据再次校准
        # ============================================================
        print("\n" + "=" * 60)
        print("第二阶段：使用补偿数据再次校准")
        print("=" * 60)

        after_calib_folder = temp_result_folder / "after_calib"
        after_calib_folder.mkdir(parents=True, exist_ok=True)

        comp_data_path = before_calib_folder / "comp_data.pkl"

        caliber_after = CaliberOctopus(
            ai_channel=ai_channels[0],
            ao_channels=ao_channels,
            sampling_info=sampling_info,
            sine_args=sine_args,
            comp_data=comp_data_path,  # 使用第一次校准的补偿数据
        )

        # 执行补偿后的校准
        print(f"\n使用补偿数据: {comp_data_path}")
        print("开始补偿后的校准...")
        caliber_after.calibrate(
            starts_num=2,
            chunks_per_start=3,
            apply_filter=True,
            result_folder=after_calib_folder,
        )

        # 验证补偿后的校准结果
        assert caliber_after.result_final_comp_data is not None
        assert len(caliber_after.result_final_comp_data["comp_list"]) == len(ao_channels)

        # 获取补偿后的传递函数数据
        tf_data_after = caliber_after.result_averaged_tf_data
        assert tf_data_after is not None

        # 计算补偿后的幅值比标准差
        amp_ratios_after = [point["amp_ratio"] for point in tf_data_after["tf_list"]]
        amp_std_after = np.std(amp_ratios_after)
        amp_mean_after = np.mean(amp_ratios_after)

        print(f"\n补偿后校准结果:")
        print(f"  幅值比平均值: {amp_mean_after:.6f}")
        print(f"  幅值比标准差: {amp_std_after:.6f}")
        print(f"  幅值比范围: [{min(amp_ratios_after):.6f}, {max(amp_ratios_after):.6f}]")

        # ============================================================
        # 第三阶段：验证补偿效果
        # ============================================================
        print("\n" + "=" * 60)
        print("第三阶段：验证补偿效果")
        print("=" * 60)

        # 计算改善程度
        std_improvement = (amp_std_before - amp_std_after) / amp_std_before * 100

        print(f"\n通道一致性改善情况:")
        print(f"  初始标准差: {amp_std_before:.6f}")
        print(f"  补偿后标准差: {amp_std_after:.6f}")
        print(f"  改善程度: {std_improvement:.2f}%")

        # 验证补偿确实改善了通道一致性
        # 补偿后的标准差应该显著小于补偿前
        assert amp_std_after < amp_std_before, (
            f"补偿后的标准差({amp_std_after:.6f})应小于补偿前({amp_std_before:.6f})"
        )

        # 期望至少有30%的改善（这是一个合理的阈值）
        assert std_improvement > 30, (
            f"标准差改善程度({std_improvement:.2f}%)应大于30%"
        )

        print(f"\n✓ 补偿效果验证通过!")
        print(f"  标准差减少了 {std_improvement:.2f}%")
        print(f"  通道一致性显著改善")

        # 验证所有结果文件都已保存
        assert (before_calib_folder / "comp_data.pkl").exists()
        assert (before_calib_folder / "transfer_function_polar.png").exists()
        assert (after_calib_folder / "comp_data.pkl").exists()
        assert (after_calib_folder / "transfer_function_polar.png").exists()

        print(f"\n所有结果文件已保存:")
        print(f"  补偿前: {before_calib_folder}")
        print(f"  补偿后: {after_calib_folder}")
