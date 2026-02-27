"""
# CaliberFishNet 多机箱数据采集调试脚本

用于排查CaliberFishNet在多AI通道(跨3个机箱)场景下的数据采集问题。

问题现象:
- CaliberOctopus (1个AI通道, 1个机箱) 工作正常,每次采集3个chunk
- CaliberFishNet (9个AI通道, 3个机箱) 采集失败,每次采集0个chunk
- 日志显示: "数据不足，未能收集到足够的AI数据包 (期望 3，实际 1)"

调试策略:
1. 测试不同数量的AI通道,找出临界点
2. 测试不同机箱组合,确认是否是特定机箱的问题
3. 增加详细日志,观察数据包收集过程
4. 测试不同的settle_time和chunks_per_start参数
"""

import pytest
from pathlib import Path

from sweeper400.analyze import init_sampling_info, init_sine_args
from sweeper400.measure import SingleChasCSIO
from sweeper400.analyze import Waveform
import numpy as np
import time


class TestMultiChassisAIDataCollection:
    """多机箱AI数据采集调试测试套件"""

    @pytest.fixture
    def sampling_info(self):
        """创建测试用的采样信息"""
        return init_sampling_info(48000.0, 4800)  # 0.1秒

    @pytest.fixture
    def sine_args(self):
        """创建测试用的正弦波参数"""
        return init_sine_args(frequency=1000.0, amplitude=0.01, phase=0.0)

    @pytest.fixture
    def ao_channels(self):
        """AO通道配置"""
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
    def test_single_ai_channel_single_chassis(self, sampling_info, sine_args, ao_channels):
        """
        测试1: 单个AI通道,单个机箱 (基线测试)

        预期: 应该成功采集3个chunk (与CaliberOctopus相同)
        """
        print("\n" + "=" * 60)
        print("测试1: 单个AI通道,单个机箱 (基线测试)")
        print("=" * 60)

        ai_channels = ("PXI1Slot2/ai0",)

        # 创建输出波形
        from sweeper400.analyze import get_sine_cycles
        single_waveform = get_sine_cycles(sampling_info, sine_args)
        multi_channel_data = np.tile(single_waveform, (len(ao_channels), 1))
        output_waveform = Waveform(
            input_array=multi_channel_data,
            sampling_rate=sampling_info["sampling_rate"],
            timestamp=single_waveform.timestamp,
            id=single_waveform.id,
            sine_args=sine_args,
        )

        # 数据收集变量
        collected_data = []
        target_chunks = 3
        collection_complete = False

        def export_function(ai_waveform, ao_static_waveform, ao_feedback_waveform, chunks_num):
            nonlocal collection_complete
            collected_data.append(ai_waveform)
            print(f"  导出回调被调用: 已收集 {len(collected_data)}/{target_chunks} 个chunk")
            if len(collected_data) >= target_chunks:
                collection_complete = True

        def feedback_function(ai_waveform):
            return np.zeros_like(ai_waveform)

        # 创建SingleChasCSIO
        sync_io = SingleChasCSIO(
            ai_channels=ai_channels,
            ao_channels_static=ao_channels,
            ao_channels_feedback=(),  # 不使用反馈通道
            static_output_waveform=output_waveform,
            export_function=export_function,
            feedback_function=feedback_function,
        )

        try:
            # 启动任务
            sync_io.start()
            print(f"SingleChasCSIO任务已启动")
            time.sleep(2.0)  # 等待稳定

            # 启用数据导出
            sync_io.enable_export = True
            print(f"数据导出已启用,开始采集...")

            # 等待采集完成
            max_wait_time = 2.0
            poll_interval = 0.05
            elapsed_time = 0.0

            while not collection_complete and elapsed_time < max_wait_time:
                time.sleep(poll_interval)
                elapsed_time += poll_interval

            # 禁用数据导出
            sync_io.enable_export = False
            print(f"数据导出已禁用")

            # 验证结果
            print(f"\n结果: 采集到 {len(collected_data)} 个chunk")
            assert len(collected_data) == target_chunks, (
                f"预期采集 {target_chunks} 个chunk,实际采集 {len(collected_data)} 个"
            )
            print("✓ 测试通过!")

        finally:
            sync_io.stop()
            print("SingleChasCSIO任务已停止")

    @pytest.mark.hardware
    def test_two_ai_channels_two_chassis(self, sampling_info, sine_args, ao_channels):
        """
        测试2: 两个AI通道,两个机箱

        预期: 应该成功采集3个chunk
        """
        print("\n" + "=" * 60)
        print("测试2: 两个AI通道,两个机箱")
        print("=" * 60)

        ai_channels = ("PXI1Slot2/ai0", "PXI2Slot2/ai0")

        self._run_data_collection_test(
            ai_channels, ao_channels, sampling_info, sine_args, expected_chunks=3
        )

    @pytest.mark.hardware
    def test_three_ai_channels_three_chassis(self, sampling_info, sine_args, ao_channels):
        """
        测试3: 三个AI通道,三个机箱 (每个机箱一个)

        这是关键测试 - 找出多机箱问题的临界点
        """
        print("\n" + "=" * 60)
        print("测试3: 三个AI通道,三个机箱 (每个机箱一个)")
        print("=" * 60)

        ai_channels = ("PXI1Slot2/ai0", "PXI2Slot2/ai0", "PXI3Slot2/ai0")

        self._run_data_collection_test(
            ai_channels, ao_channels, sampling_info, sine_args, expected_chunks=3
        )

    @pytest.mark.hardware
    def test_nine_ai_channels_three_chassis(self, sampling_info, sine_args, ao_channels):
        """
        测试4: 九个AI通道,三个机箱 (完整的CaliberFishNet配置)

        这是复现问题的测试
        """
        print("\n" + "=" * 60)
        print("测试4: 九个AI通道,三个机箱 (完整配置)")
        print("=" * 60)

        ai_channels = (
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

        self._run_data_collection_test(
            ai_channels, ao_channels, sampling_info, sine_args, expected_chunks=3
        )

    def _run_data_collection_test(
        self, ai_channels, ao_channels, sampling_info, sine_args, expected_chunks=3
    ):
        """
        通用的数据采集测试方法

        Args:
            ai_channels: AI通道元组
            ao_channels: AO通道元组
            sampling_info: 采样信息
            sine_args: 正弦波参数
            expected_chunks: 期望采集的chunk数量
        """
        print(f"\nAI通道配置: {len(ai_channels)} 个通道")
        for i, ch in enumerate(ai_channels):
            print(f"  AI[{i}]: {ch}")

        # 创建输出波形
        from sweeper400.analyze import get_sine_cycles
        single_waveform = get_sine_cycles(sampling_info, sine_args)
        multi_channel_data = np.tile(single_waveform, (len(ao_channels), 1))
        output_waveform = Waveform(
            input_array=multi_channel_data,
            sampling_rate=sampling_info["sampling_rate"],
            timestamp=single_waveform.timestamp,
            id=single_waveform.id,
            sine_args=sine_args,
        )

        # 数据收集变量
        collected_data = []
        target_chunks = expected_chunks
        collection_complete = False
        export_call_count = 0

        def export_function(ai_waveform, ao_static_waveform, ao_feedback_waveform, chunks_num):
            nonlocal collection_complete, export_call_count
            export_call_count += 1
            collected_data.append(ai_waveform)

            # 详细打印波形信息
            print(f"  [回调 #{export_call_count}] 导出函数被调用:")
            print(f"    AI波形shape={ai_waveform.shape}")
            print(f"    AI波形ndim={ai_waveform.ndim}")
            if hasattr(ai_waveform, 'channel_names'):
                print(f"    通道名称数量={len(ai_waveform.channel_names)}")
                print(f"    通道名称={ai_waveform.channel_names}")
            print(f"    已收集 {len(collected_data)}/{target_chunks} 个chunk")

            if len(collected_data) >= target_chunks:
                collection_complete = True

        def feedback_function(ai_waveform):
            return np.zeros_like(ai_waveform)

        # 创建SingleChasCSIO
        print(f"\n创建SingleChasCSIO...")
        sync_io = SingleChasCSIO(
            ai_channels=ai_channels,
            ao_channels_static=ao_channels,
            ao_channels_feedback=(),  # 不使用反馈通道
            static_output_waveform=output_waveform,
            export_function=export_function,
            feedback_function=feedback_function,
        )

        try:
            # 启动任务
            print(f"启动SingleChasCSIO任务...")
            sync_io.start()
            print(f"✓ SingleChasCSIO任务已启动")

            # 等待稳定
            settle_time = 2.0
            print(f"等待 {settle_time}s 以稳定...")
            time.sleep(settle_time)

            # 启用数据导出
            print(f"\n启用数据导出,开始采集 {expected_chunks} 个chunk...")
            sync_io.enable_export = True

            # 等待采集完成
            chunk_duration = output_waveform.duration
            # 增加等待时间倍数，给多机箱同步留出更多时间
            max_wait_time = chunk_duration * expected_chunks * 3.0
            poll_interval = 0.05
            elapsed_time = 0.0

            print(f"最大等待时间: {max_wait_time:.2f}s")

            while not collection_complete and elapsed_time < max_wait_time:
                time.sleep(poll_interval)
                elapsed_time += poll_interval

                # 每0.5秒打印一次进度
                if int(elapsed_time / 0.5) > int((elapsed_time - poll_interval) / 0.5):
                    print(f"  等待中... 已过 {elapsed_time:.1f}s, "
                          f"已收集 {len(collected_data)} 个chunk")

            # 禁用数据导出
            sync_io.enable_export = False
            print(f"\n✓ 数据导出已禁用")

            # 打印结果
            print(f"\n" + "=" * 60)
            print(f"测试结果:")
            print(f"  导出函数调用次数: {export_call_count}")
            print(f"  采集到的chunk数: {len(collected_data)}")
            print(f"  期望的chunk数: {expected_chunks}")
            print(f"=" * 60)

            # 验证结果
            if len(collected_data) == expected_chunks:
                print("✓ 测试通过!")
            else:
                print(f"✗ 测试失败!")
                print(f"  预期采集 {expected_chunks} 个chunk")
                print(f"  实际采集 {len(collected_data)} 个chunk")

                # 不立即断言失败,而是继续收集信息
                assert False, (
                    f"数据采集失败: 预期 {expected_chunks} 个chunk, "
                    f"实际 {len(collected_data)} 个chunk, "
                    f"导出函数被调用 {export_call_count} 次"
                )

        finally:
            sync_io.stop()
            print("\n✓ SingleChasCSIO任务已停止")
