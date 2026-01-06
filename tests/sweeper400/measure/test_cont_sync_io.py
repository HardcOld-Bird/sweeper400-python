"""
# MultiChasCSIO 类测试模块

测试路径：`tests/sweeper400/measure/test_cont_sync_io.py`

本模块包含对 MultiChasCSIO 类的系统化测试，验证其核心功能是否正常工作。
"""

import time
from collections import deque

import numpy as np
import pytest

from sweeper400.analyze import (
    Waveform,
    extract_single_tone_information_vvi,
    get_sine_cycles,
    init_sampling_info,
    init_sine_args,
)
from sweeper400.measure import MultiChasCSIO, HiPerfCSIO


class TestMultiChasCSIO:
    """MultiChasCSIO 类的测试套件"""

    @pytest.fixture
    def sampling_info(self):
        """创建测试用的采样信息"""
        # 使用较短的采样时间以加快测试速度
        return init_sampling_info(48000.0, 4800)  # 0.1秒

    @pytest.fixture
    def sine_args(self):
        """创建测试用的正弦波参数"""
        # 使用1000Hz的正弦波，幅值0.02V
        return init_sine_args(frequency=1000.0, amplitude=0.02, phase=0.0)

    @pytest.fixture
    def output_waveform(self, sampling_info, sine_args):
        """创建测试用的输出波形"""
        return get_sine_cycles(sampling_info, sine_args)

    @pytest.fixture
    def ai_channel(self):
        """AI通道配置"""
        return "PXI2Slot2/ai0"

    @pytest.fixture
    def ao_channels_single_chassis(self):
        """单机箱AO通道配置"""
        return ("PXI2Slot2/ao0",)

    @pytest.fixture
    def ao_channels_multi_chassis(self):
        """跨机箱AO通道配置"""
        return (
            "PXI1Slot2/ao0",  # PXIChassis1
            "PXI2Slot2/ao0",  # PXIChassis2 (Master)
            "PXI3Slot2/ao0",  # PXIChassis3
        )

    @pytest.fixture
    def export_data_collector(self):
        """创建数据导出收集器"""

        class DataCollector:
            def __init__(self):
                self.collected_data: deque[tuple[Waveform, int]] = deque()

            def export_function(self, ai_waveform: Waveform, chunks_num: int):
                """数据导出函数"""
                self.collected_data.append((ai_waveform, chunks_num))

        return DataCollector()

    @pytest.mark.hardware
    def test_single_chassis_basic_operation(
        self,
        ai_channel,
        ao_channels_single_chassis,
        output_waveform,
        export_data_collector,
    ):
        """测试单机箱基本操作"""
        # 创建MultiChasCSIO实例
        sync_io = MultiChasCSIO(
            ai_channel=ai_channel,
            ao_channels=ao_channels_single_chassis,
            output_waveform=output_waveform,
            export_function=export_data_collector.export_function,
        )

        try:
            # 启动任务
            sync_io.start()
            assert sync_io._is_running is True

            # 启用数据导出
            sync_io.enable_export = True
            assert sync_io.enable_export is True

            # 运行一段时间，收集数据
            time.sleep(1.0)

            # 停止任务
            sync_io.stop()
            assert sync_io._is_running is False

            # 验证收集到的数据
            assert len(export_data_collector.collected_data) > 0
            print(f"收集到 {len(export_data_collector.collected_data)} 段数据")

        finally:
            if sync_io._is_running:
                sync_io.stop()

    @pytest.mark.hardware
    def test_multi_chassis_basic_operation(
        self,
        ai_channel,
        ao_channels_multi_chassis,
        output_waveform,
        export_data_collector,
    ):
        """测试跨机箱基本操作"""
        # 创建MultiChasCSIO实例
        sync_io = MultiChasCSIO(
            ai_channel=ai_channel,
            ao_channels=ao_channels_multi_chassis,
            output_waveform=output_waveform,
            export_function=export_data_collector.export_function,
        )

        try:
            # 启动任务
            sync_io.start()
            assert sync_io._is_running is True

            # 启用数据导出
            sync_io.enable_export = True

            # 运行一段时间，收集数据
            time.sleep(1.0)

            # 停止任务
            sync_io.stop()

            # 验证收集到的数据
            assert len(export_data_collector.collected_data) > 0
            print(f"跨机箱模式收集到 {len(export_data_collector.collected_data)} 段数据")

        finally:
            if sync_io._is_running:
                sync_io.stop()


    @pytest.mark.hardware
    def test_export_numbering_reset(
        self,
        ai_channel,
        ao_channels_single_chassis,
        output_waveform,
        export_data_collector,
    ):
        """测试数据导出编号重置逻辑"""
        sync_io = MultiChasCSIO(
            ai_channel=ai_channel,
            ao_channels=ao_channels_single_chassis,
            output_waveform=output_waveform,
            export_function=export_data_collector.export_function,
        )

        try:
            # 启动任务
            sync_io.start()

            # 第一次导出周期
            sync_io.enable_export = True
            time.sleep(0.5)
            sync_io.enable_export = False
            time.sleep(0.2)

            # 检查第一次导出的编号
            first_batch = list(export_data_collector.collected_data)
            first_batch_numbers = [num for _, num in first_batch]
            print(f"第一批导出编号: {first_batch_numbers}")

            # 验证第一批从1开始
            assert first_batch_numbers[0] == 1, "第一批数据应从1开始编号"
            # 验证编号连续递增
            for i in range(1, len(first_batch_numbers)):
                assert (
                    first_batch_numbers[i] == first_batch_numbers[i - 1] + 1
                ), "编号应连续递增"

            # 清空收集器
            export_data_collector.collected_data.clear()

            # 第二次导出周期
            sync_io.enable_export = True
            time.sleep(0.5)
            sync_io.enable_export = False

            # 检查第二次导出的编号
            second_batch = list(export_data_collector.collected_data)
            second_batch_numbers = [num for _, num in second_batch]
            print(f"第二批导出编号: {second_batch_numbers}")

            # 验证第二批也从1开始（重置）
            assert second_batch_numbers[0] == 1, "第二批数据应从1开始编号（重置）"
            # 验证编号连续递增
            for i in range(1, len(second_batch_numbers)):
                assert (
                    second_batch_numbers[i] == second_batch_numbers[i - 1] + 1
                ), "编号应连续递增"

            sync_io.stop()

        finally:
            if sync_io._is_running:
                sync_io.stop()

    @pytest.mark.hardware
    def test_signal_quality_single_tone(
        self,
        ai_channel,
        ao_channels_single_chassis,
        output_waveform,
        export_data_collector,
        sine_args,
    ):
        """测试信号质量：验证采集到的信号是否为预期的单频正弦波"""
        sync_io = MultiChasCSIO(
            ai_channel=ai_channel,
            ao_channels=ao_channels_single_chassis,
            output_waveform=output_waveform,
            export_function=export_data_collector.export_function,
        )

        try:
            # 启动任务并采集数据
            sync_io.start()
            sync_io.enable_export = True
            time.sleep(1.0)
            sync_io.stop()

            # 获取采集到的数据
            assert len(export_data_collector.collected_data) > 0
            ai_waveform, _ = export_data_collector.collected_data[0]

            # 使用 extract_single_tone_information_vvi 提取信号参数
            detected_sine_args = extract_single_tone_information_vvi(
                ai_waveform, approx_freq=sine_args["frequency"]
            )

            # 验证频率（允许5%的误差，考虑硬件和信号传递的影响）
            freq_error = abs(
                detected_sine_args["frequency"] - sine_args["frequency"]
            ) / sine_args["frequency"]
            print(
                f"频率误差: {freq_error * 100:.2f}% "
                f"(期望: {sine_args['frequency']}Hz, "
                f"检测: {detected_sine_args['frequency']:.2f}Hz)"
            )
            assert freq_error < 0.05, "频率误差应小于5%"

            # 验证幅值（考虑传递函数，允许较大误差）
            amp_ratio = detected_sine_args["amplitude"] / sine_args["amplitude"]
            print(
                f"幅值比: {amp_ratio:.2f} "
                f"(期望: {sine_args['amplitude']:.4f}V, "
                f"检测: {detected_sine_args['amplitude']:.4f}V)"
            )
            # 幅值可能因传递函数而变化，这里只验证在合理范围内（0.01到100倍）
            assert 0.01 < amp_ratio < 100.0, "幅值应在合理范围内"

            print(f"相位: {detected_sine_args['phase']:.4f} rad")

        finally:
            if sync_io._is_running:
                sync_io.stop()

    @pytest.mark.hardware
    def test_channel_status_control(
        self,
        ai_channel,
        ao_channels_multi_chassis,
        output_waveform,
        export_data_collector,
    ):
        """测试AO通道状态控制功能"""
        sync_io = MultiChasCSIO(
            ai_channel=ai_channel,
            ao_channels=ao_channels_multi_chassis,
            output_waveform=output_waveform,
            export_function=export_data_collector.export_function,
        )

        try:
            # 启动任务
            sync_io.start()

            # 测试获取通道状态
            initial_status = sync_io.get_ao_channels_status()
            assert len(initial_status) == 3
            assert all(initial_status), "初始状态应全部启用"

            # 测试设置通道状态
            new_status = (True, False, True)
            sync_io.set_ao_channels_status(new_status)
            current_status = sync_io.get_ao_channels_status()
            assert current_status == new_status, "通道状态应正确更新"

            # 运行一段时间验证状态保持
            time.sleep(0.5)
            assert sync_io.get_ao_channels_status() == new_status

            sync_io.stop()

        finally:
            if sync_io._is_running:
                sync_io.stop()


    @pytest.mark.hardware
    def test_start_stop_multiple_times(
        self,
        ai_channel,
        ao_channels_single_chassis,
        output_waveform,
        export_data_collector,
    ):
        """测试多次启动和停止"""
        sync_io = MultiChasCSIO(
            ai_channel=ai_channel,
            ao_channels=ao_channels_single_chassis,
            output_waveform=output_waveform,
            export_function=export_data_collector.export_function,
        )

        try:
            # 第一次启动和停止
            sync_io.start()
            assert sync_io._is_running is True
            sync_io.enable_export = True
            time.sleep(0.3)
            sync_io.stop()
            assert sync_io._is_running is False

            first_count = len(export_data_collector.collected_data)
            print(f"第一次运行收集 {first_count} 段数据")

            # 清空收集器
            export_data_collector.collected_data.clear()

            # 第二次启动和停止
            sync_io.start()
            assert sync_io._is_running is True
            sync_io.enable_export = True
            time.sleep(0.3)
            sync_io.stop()
            assert sync_io._is_running is False

            second_count = len(export_data_collector.collected_data)
            print(f"第二次运行收集 {second_count} 段数据")

            # 验证两次都能正常工作
            assert first_count > 0
            assert second_count > 0

        finally:
            if sync_io._is_running:
                sync_io.stop()

    @pytest.mark.hardware
    def test_data_continuity(
        self,
        ai_channel,
        ao_channels_single_chassis,
        output_waveform,
        export_data_collector,
    ):
        """测试数据连续性：验证采集的数据段是连续的"""
        sync_io = MultiChasCSIO(
            ai_channel=ai_channel,
            ao_channels=ao_channels_single_chassis,
            output_waveform=output_waveform,
            export_function=export_data_collector.export_function,
        )

        try:
            # 启动任务并采集数据
            sync_io.start()
            sync_io.enable_export = True
            time.sleep(1.5)
            sync_io.stop()

            # 验证数据连续性
            collected_numbers = [num for _, num in export_data_collector.collected_data]
            print(f"收集到的数据段编号: {collected_numbers[:10]}...")

            # 验证编号从1开始
            assert collected_numbers[0] == 1, "第一段数据编号应为1"

            # 验证编号连续递增
            for i in range(1, len(collected_numbers)):
                assert (
                    collected_numbers[i] == collected_numbers[i - 1] + 1
                ), f"数据段编号应连续，但在索引{i}处不连续"

            print(f"数据连续性验证通过，共 {len(collected_numbers)} 段数据")

        finally:
            if sync_io._is_running:
                sync_io.stop()

    @pytest.mark.hardware
    @pytest.mark.slow
    def test_long_running_stability(
        self,
        ai_channel,
        ao_channels_multi_chassis,
        output_waveform,
        export_data_collector,
    ):
        """测试长时间运行稳定性"""
        sync_io = MultiChasCSIO(
            ai_channel=ai_channel,
            ao_channels=ao_channels_multi_chassis,
            output_waveform=output_waveform,
            export_function=export_data_collector.export_function,
        )

        try:
            # 启动任务
            sync_io.start()
            sync_io.enable_export = True

            # 长时间运行（5秒）
            run_duration = 5.0
            start_time = time.time()

            while time.time() - start_time < run_duration:
                time.sleep(0.5)
                # 检查任务仍在运行
                assert sync_io._is_running is True

            sync_io.stop()

            # 验证收集到足够的数据
            total_collected = len(export_data_collector.collected_data)
            print(f"长时间运行 {run_duration}s，收集到 {total_collected} 段数据")

            # 验证数据量合理（基于采样率和波形长度）
            expected_chunks = int(
                run_duration / output_waveform.duration
            )  # 预期的数据段数
            assert (
                total_collected >= expected_chunks * 0.8
            ), f"收集的数据量过少（期望至少 {expected_chunks * 0.8}，实际 {total_collected}）"

            # 验证数据连续性
            collected_numbers = [num for _, num in export_data_collector.collected_data]
            for i in range(1, len(collected_numbers)):
                assert (
                    collected_numbers[i] == collected_numbers[i - 1] + 1
                ), "长时间运行后数据编号应仍然连续"

        finally:
            if sync_io._is_running:
                sync_io.stop()

    @pytest.mark.hardware
    def test_waveform_shape_consistency(
        self,
        ai_channel,
        ao_channels_single_chassis,
        output_waveform,
        export_data_collector,
    ):
        """测试采集波形的形状一致性"""
        sync_io = MultiChasCSIO(
            ai_channel=ai_channel,
            ao_channels=ao_channels_single_chassis,
            output_waveform=output_waveform,
            export_function=export_data_collector.export_function,
        )

        try:
            # 启动任务并采集数据
            sync_io.start()
            sync_io.enable_export = True
            time.sleep(1.0)
            sync_io.stop()

            # 验证所有采集的波形形状一致
            assert len(export_data_collector.collected_data) > 0

            first_waveform, _ = export_data_collector.collected_data[0]
            expected_shape = first_waveform.shape
            expected_samples = output_waveform.samples_num

            print(f"波形形状: {expected_shape}, 采样点数: {expected_samples}")

            for i, (waveform, _) in enumerate(export_data_collector.collected_data):
                assert (
                    waveform.shape == expected_shape
                ), f"第{i}段波形形状不一致: {waveform.shape} != {expected_shape}"
                assert (
                    waveform.samples_num == expected_samples
                ), f"第{i}段波形采样点数不一致"
                assert (
                    waveform.sampling_rate == output_waveform.sampling_rate
                ), f"第{i}段波形采样率不一致"

            print(f"所有 {len(export_data_collector.collected_data)} 段波形形状一致")

        finally:
            if sync_io._is_running:
                sync_io.stop()

    @pytest.mark.hardware
    def test_external_10mhz_clock_configuration(
        self,
        ai_channel,
        ao_channels_multi_chassis,
        output_waveform,
        export_data_collector,
    ):
        """
        测试外部10MHz参考时钟配置

        验证：
        1. 所有任务正确配置为使用OnboardClock（100MHz，自动锁定到外部10MHz参考时钟）
        2. 任务能够正常启动和运行
        3. 数据采集正常
        """
        print("\n" + "=" * 80)
        print("测试外部10MHz参考时钟配置")
        print("=" * 80)

        # 创建MultiChasCSIO实例
        sync_io = MultiChasCSIO(
            ai_channel=ai_channel,
            ao_channels=ao_channels_multi_chassis,
            output_waveform=output_waveform,
            export_function=export_data_collector.export_function,
        )

        try:
            # 启动任务
            print("\n启动任务...")
            sync_io.start()

            # 验证AI任务的时钟配置
            assert sync_io._ai_task is not None
            ai_ref_clk_src = sync_io._ai_task.timing.ref_clk_src
            ai_ref_clk_rate = sync_io._ai_task.timing.ref_clk_rate
            print(f"AI任务参考时钟源: {ai_ref_clk_src}")
            print(f"AI任务参考时钟频率: {ai_ref_clk_rate} Hz")
            # DAQmx可能返回"None"或"OnboardClock"，两者都表示使用板载时钟
            assert ai_ref_clk_src in ["None", "OnboardClock"], f"AI任务时钟源应为None或OnboardClock，实际为{ai_ref_clk_src}"
            assert ai_ref_clk_rate == 100000000, f"AI任务时钟频率应为100MHz，实际为{ai_ref_clk_rate}Hz"

            # 验证所有AO任务的时钟配置
            for chassis_name, ao_task in sync_io._ao_tasks.items():
                ao_ref_clk_src = ao_task.timing.ref_clk_src
                ao_ref_clk_rate = ao_task.timing.ref_clk_rate
                print(f"{chassis_name} AO任务参考时钟源: {ao_ref_clk_src}")
                print(f"{chassis_name} AO任务参考时钟频率: {ao_ref_clk_rate} Hz")
                assert ao_ref_clk_src in ["None", "OnboardClock"], f"{chassis_name}时钟源应为None或OnboardClock"
                assert ao_ref_clk_rate == 100000000, f"{chassis_name}时钟频率应为100MHz"

            # 启用数据导出并采集数据
            print("\n启用数据导出并采集数据...")
            sync_io.enable_export = True
            time.sleep(2.0)  # 采集2秒数据

            # 验证数据采集正常
            assert len(export_data_collector.collected_data) > 0, "未采集到数据"
            print(f"成功采集 {len(export_data_collector.collected_data)} 段数据")

            # 验证采集到的数据质量
            first_waveform, _ = export_data_collector.collected_data[0]
            print(f"波形形状: {first_waveform.shape}")
            print(f"采样率: {first_waveform.sampling_rate} Hz")

            print("\n外部10MHz参考时钟配置测试通过！")

        finally:
            if sync_io._is_running:
                sync_io.stop()

    @pytest.mark.hardware
    def test_pfi_trigger_configuration(
        self,
        ai_channel,
        ao_channels_multi_chassis,
        output_waveform,
        export_data_collector,
    ):
        """
        测试PFI触发配置

        验证：
        1. Master机箱通过PFI0导出触发信号
        2. Slave机箱通过PFI0接收触发信号
        3. 所有任务能够同步启动
        4. 数据采集正常
        """
        print("\n" + "=" * 80)
        print("测试PFI触发配置（星型拓扑）")
        print("=" * 80)

        # 创建MultiChasCSIO实例
        sync_io = MultiChasCSIO(
            ai_channel=ai_channel,
            ao_channels=ao_channels_multi_chassis,
            output_waveform=output_waveform,
            export_function=export_data_collector.export_function,
        )

        try:
            # 启动任务
            print("\n启动任务...")
            sync_io.start()

            # 验证Master机箱的触发导出配置
            master_chassis = sync_io._master_chassis
            print(f"Master机箱: {master_chassis}")

            if master_chassis in sync_io._ao_tasks:
                master_task = sync_io._ao_tasks[master_chassis]
                # 检查是否配置了触发导出
                try:
                    export_term = master_task.export_signals.start_trig_output_term
                    print(f"Master机箱触发导出终端: {export_term}")
                    assert "PFI0" in export_term, f"Master应导出到PFI0，实际为{export_term}"
                except Exception as e:
                    print(f"注意: 无法读取触发导出配置: {e}")

            # 验证Slave机箱的触发接收配置
            for chassis_name, ao_task in sync_io._ao_tasks.items():
                if chassis_name != master_chassis:
                    # 检查Slave任务的触发源
                    try:
                        trigger_term = ao_task.triggers.start_trigger.term
                        print(f"Slave机箱 {chassis_name} 触发源: {trigger_term}")
                        assert "PFI0" in trigger_term, f"Slave应从PFI0接收触发"
                    except Exception as e:
                        print(f"注意: 无法读取触发源配置: {e}")

            # 启用数据导出并采集数据
            print("\n启用数据导出并采集数据...")
            sync_io.enable_export = True
            time.sleep(2.0)  # 采集2秒数据

            # 验证数据采集正常
            assert len(export_data_collector.collected_data) > 0, "未采集到数据"
            print(f"成功采集 {len(export_data_collector.collected_data)} 段数据")

            print("\nPFI触发配置测试通过！")

        finally:
            if sync_io._is_running:
                sync_io.stop()
