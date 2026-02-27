"""
# SingleChasCSIO 类测试模块

测试路径：`tests/sweeper400/measure/test_single_chas_csio.py`

本模块包含对 SingleChasCSIO 类的系统化测试，验证其核心功能是否正常工作。
"""

import time
from collections import deque

import pytest

from sweeper400.analyze import (
    Waveform,
    extract_single_tone_information_vvi,
    get_sine_cycles,
    init_sampling_info,
    init_sine_args,
)
from sweeper400.measure.cont_sync_io import SingleChasCSIO


class TestSingleChasCSIO:
    """SingleChasCSIO 类的测试套件"""

    @staticmethod
    def _dummy_feedback_function(ai_waveform: Waveform) -> Waveform:
        """简单的反馈函数，返回零"""
        return ai_waveform * 0.0

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
    def ai_channels_single(self):
        """单AI通道配置"""
        return ("PXI1Slot2/ai0",)

    @pytest.fixture
    def ai_channels_multi(self):
        """多AI通道配置（同一机箱）"""
        return (
            "PXI1Slot2/ai0",
            "PXI1Slot3/ai0",
            "PXI1Slot3/ai1",
            "PXI1Slot4/ai0",
            "PXI1Slot4/ai1",
            "PXI1Slot5/ai0",
            "PXI1Slot5/ai1",
            "PXI1Slot6/ai0",
            "PXI1Slot6/ai1",
        )

    @pytest.fixture
    def ao_channels_static_single(self):
        """单个Static AO通道配置"""
        return ("PXI1Slot2/ao0",)

    @pytest.fixture
    def ao_channels_static_multi(self):
        """多个Static AO通道配置（同一机箱）"""
        return (
            "PXI1Slot2/ao0",
            "PXI1Slot3/ao0",
            "PXI1Slot3/ao1",
            "PXI1Slot4/ao0",
            "PXI1Slot4/ao1",
            "PXI1Slot5/ao0",
            "PXI1Slot5/ao1",
            "PXI1Slot6/ao0",
            "PXI1Slot6/ao1",
        )

    @pytest.fixture
    def ao_channels_feedback_single(self):
        """单个Feedback AO通道配置"""
        return ("PXI1Slot3/ao1",)

    @pytest.fixture
    def export_data_collector(self):
        """创建数据导出收集器"""

        class DataCollector:
            def __init__(self):
                self.collected_data: deque[
                    tuple[Waveform, Waveform, Waveform | None, int]
                ] = deque()

            def export_function(
                self,
                ai_waveform: Waveform,
                ao_static_waveform: Waveform,
                ao_feedback_waveform: Waveform | None,
                chunks_num: int,
            ):
                """数据导出函数"""
                self.collected_data.append(
                    (ai_waveform, ao_static_waveform, ao_feedback_waveform, chunks_num)
                )
                print(
                    f"  [导出] Chunk {chunks_num}: "
                    f"AI shape={ai_waveform.shape}, "
                    f"Static AO shape={ao_static_waveform.shape}, "
                    f"Feedback AO={'None' if ao_feedback_waveform is None else ao_feedback_waveform.shape}"
                )

            def clear(self):
                """清空收集的数据"""
                self.collected_data.clear()

        return DataCollector()

    def test_basic_initialization(
        self,
        ai_channels_single,
        ao_channels_static_single,
        output_waveform,
    ):
        """测试基本初始化"""
        print("\n=== 测试基本初始化 ===")

        # 创建实例
        sync_io = SingleChasCSIO(
            ai_channels=ai_channels_single,
            ao_channels_static=ao_channels_static_single,
            ao_channels_feedback=(),
            static_output_waveform=output_waveform,
            feedback_function=self._dummy_feedback_function,
            export_function=lambda *args: None,
        )

        # 验证属性
        assert sync_io.ai_channels_num == 1
        assert sync_io.ao_channels_num_static == 1
        assert sync_io.ao_channels_num_feedback == 0
        assert sync_io.enable_export is False

        print("✓ 基本初始化测试通过")

    def test_multi_channel_initialization(
        self,
        ai_channels_multi,
        ao_channels_static_multi,
        output_waveform,
    ):
        """测试多通道初始化"""
        print("\n=== 测试多通道初始化 ===")

        # 创建实例
        sync_io = SingleChasCSIO(
            ai_channels=ai_channels_multi,
            ao_channels_static=ao_channels_static_multi,
            ao_channels_feedback=(),
            static_output_waveform=output_waveform,
            feedback_function=self._dummy_feedback_function,
            export_function=lambda *args: None,
        )

        # 验证属性
        assert sync_io.ai_channels_num == 9
        assert sync_io.ao_channels_num_static == 9
        assert sync_io.ao_channels_num_feedback == 0

        print("✓ 多通道初始化测试通过")

    def test_cross_chassis_validation(
        self,
        output_waveform,
    ):
        """测试跨机箱验证（应该失败）"""
        print("\n=== 测试跨机箱验证 ===")

        # 尝试创建跨机箱实例（应该抛出异常）
        with pytest.raises(ValueError, match="所有通道必须位于同一机箱"):
            SingleChasCSIO(
                ai_channels=("PXI1Slot2/ai0",),
                ao_channels_static=("PXI2Slot2/ao0",),  # 不同机箱
                ao_channels_feedback=(),
                static_output_waveform=output_waveform,
                feedback_function=self._dummy_feedback_function,
                export_function=lambda *args: None,
            )

        print("✓ 跨机箱验证测试通过")

    def test_start_stop_basic(
        self,
        ai_channels_single,
        ao_channels_static_single,
        output_waveform,
        export_data_collector,
    ):
        """测试基本的启动和停止"""
        print("\n=== 测试基本的启动和停止 ===")

        # 创建实例
        sync_io = SingleChasCSIO(
            ai_channels=ai_channels_single,
            ao_channels_static=ao_channels_static_single,
            ao_channels_feedback=(),
            static_output_waveform=output_waveform,
            feedback_function=self._dummy_feedback_function,
            export_function=export_data_collector.export_function,
        )

        # 启动任务
        print("启动任务...")
        sync_io.start()
        assert sync_io._is_running is True

        # 等待一段时间
        time.sleep(1.0)

        # 停止任务
        print("停止任务...")
        sync_io.stop()
        assert sync_io._is_running is False

        print("✓ 基本启动停止测试通过")

    def test_data_collection_single_channel(
        self,
        ai_channels_single,
        ao_channels_static_single,
        output_waveform,
        export_data_collector,
    ):
        """测试单通道数据采集"""
        print("\n=== 测试单通道数据采集 ===")

        # 创建实例
        sync_io = SingleChasCSIO(
            ai_channels=ai_channels_single,
            ao_channels_static=ao_channels_static_single,
            ao_channels_feedback=(),
            static_output_waveform=output_waveform,
            feedback_function=self._dummy_feedback_function,
            export_function=export_data_collector.export_function,
        )

        # 启动任务
        print("启动任务...")
        sync_io.start()

        # 等待系统稳定
        time.sleep(1.0)

        # 启用数据导出
        print("启用数据导出...")
        export_data_collector.clear()
        sync_io.enable_export = True

        # 收集数据
        target_chunks = 5
        print(f"收集 {target_chunks} 个数据块...")
        timeout = 10.0
        start_time = time.time()

        while len(export_data_collector.collected_data) < target_chunks:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"数据收集超时，仅收集到 {len(export_data_collector.collected_data)} 个数据块")
            time.sleep(0.1)

        # 停止任务
        print("停止任务...")
        sync_io.stop()

        # 验证收集的数据
        print(f"成功收集 {len(export_data_collector.collected_data)} 个数据块")
        assert len(export_data_collector.collected_data) >= target_chunks

        # 验证第一个数据块
        ai_waveform, ao_static_waveform, ao_feedback_waveform, chunks_num = (
            export_data_collector.collected_data[0]
        )
        assert ai_waveform.channels_num == 1
        assert ai_waveform.samples_num == output_waveform.samples_num
        assert ao_static_waveform.channels_num == 1
        assert ao_feedback_waveform is None

        print("✓ 单通道数据采集测试通过")

    def test_data_collection_multi_channel(
        self,
        ai_channels_multi,
        ao_channels_static_multi,
        output_waveform,
        export_data_collector,
    ):
        """测试多通道数据采集"""
        print("\n=== 测试多通道数据采集 ===")

        # 创建实例
        sync_io = SingleChasCSIO(
            ai_channels=ai_channels_multi,
            ao_channels_static=ao_channels_static_multi,
            ao_channels_feedback=(),
            static_output_waveform=output_waveform,
            feedback_function=self._dummy_feedback_function,
            export_function=export_data_collector.export_function,
        )

        # 启动任务
        print("启动任务...")
        sync_io.start()

        # 等待系统稳定
        time.sleep(1.0)

        # 启用数据导出
        print("启用数据导出...")
        export_data_collector.clear()
        sync_io.enable_export = True

        # 收集数据
        target_chunks = 3
        print(f"收集 {target_chunks} 个数据块...")
        timeout = 10.0
        start_time = time.time()

        while len(export_data_collector.collected_data) < target_chunks:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"数据收集超时，仅收集到 {len(export_data_collector.collected_data)} 个数据块")
            time.sleep(0.1)

        # 停止任务
        print("停止任务...")
        sync_io.stop()

        # 验证收集的数据
        print(f"成功收集 {len(export_data_collector.collected_data)} 个数据块")
        assert len(export_data_collector.collected_data) >= target_chunks

        # 验证第一个数据块
        ai_waveform, ao_static_waveform, ao_feedback_waveform, chunks_num = (
            export_data_collector.collected_data[0]
        )
        assert ai_waveform.channels_num == 9
        assert ai_waveform.samples_num == output_waveform.samples_num
        assert ao_static_waveform.channels_num == 9
        assert ao_feedback_waveform is None

        print("✓ 多通道数据采集测试通过")

    def test_update_static_waveform(
        self,
        ai_channels_single,
        ao_channels_static_single,
        output_waveform,
        sampling_info,
        export_data_collector,
    ):
        """测试动态更换静态输出波形"""
        print("\n=== 测试动态更换静态输出波形 ===")

        # 创建实例
        sync_io = SingleChasCSIO(
            ai_channels=ai_channels_single,
            ao_channels_static=ao_channels_static_single,
            ao_channels_feedback=(),
            static_output_waveform=output_waveform,
            feedback_function=self._dummy_feedback_function,
            export_function=export_data_collector.export_function,
        )

        # 启动任务
        print("启动任务...")
        sync_io.start()

        # 等待系统稳定
        time.sleep(1.0)

        # 启用数据导出
        print("启用数据导出...")
        export_data_collector.clear()
        sync_io.enable_export = True

        # 收集一些数据
        time.sleep(0.5)

        # 更换波形
        print("更换输出波形...")
        new_sine_args = init_sine_args(frequency=2000.0, amplitude=0.03, phase=0.0)
        new_waveform = get_sine_cycles(sampling_info, new_sine_args)
        sync_io.update_static_output_waveform(new_waveform)

        # 继续收集数据
        time.sleep(0.5)

        # 停止任务
        print("停止任务...")
        sync_io.stop()

        # 验证收集的数据
        print(f"成功收集 {len(export_data_collector.collected_data)} 个数据块")
        assert len(export_data_collector.collected_data) > 0

        print("✓ 动态更换波形测试通过")

    def test_feedback_functionality(
        self,
        ai_channels_single,
        ao_channels_static_single,
        ao_channels_feedback_single,
        output_waveform,
        export_data_collector,
    ):
        """测试反馈功能"""
        print("\n=== 测试反馈功能 ===")

        # 定义反馈函数
        def feedback_function(ai_waveform: Waveform) -> Waveform:
            # 简单的反馈：将AI数据取反并缩放
            return ai_waveform * -0.1

        # 创建实例
        sync_io = SingleChasCSIO(
            ai_channels=ai_channels_single,
            ao_channels_static=ao_channels_static_single,
            ao_channels_feedback=ao_channels_feedback_single,
            static_output_waveform=output_waveform,
            feedback_function=feedback_function,
            export_function=export_data_collector.export_function,
        )

        # 启动任务
        print("启动任务...")
        sync_io.start()

        # 等待系统稳定
        time.sleep(1.0)

        # 启用数据导出
        print("启用数据导出...")
        export_data_collector.clear()
        sync_io.enable_export = True

        # 收集数据
        target_chunks = 3
        print(f"收集 {target_chunks} 个数据块...")
        timeout = 10.0
        start_time = time.time()

        while len(export_data_collector.collected_data) < target_chunks:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"数据收集超时")
            time.sleep(0.1)

        # 停止任务
        print("停止任务...")
        sync_io.stop()

        # 验证收集的数据
        print(f"成功收集 {len(export_data_collector.collected_data)} 个数据块")
        assert len(export_data_collector.collected_data) >= target_chunks

        # 验证反馈波形存在
        ai_waveform, ao_static_waveform, ao_feedback_waveform, chunks_num = (
            export_data_collector.collected_data[0]
        )
        assert ao_feedback_waveform is not None
        assert ao_feedback_waveform.channels_num == 1
        assert ao_feedback_waveform.samples_num == output_waveform.samples_num

        print("✓ 反馈功能测试通过")

    def test_context_manager(
        self,
        ai_channels_single,
        ao_channels_static_single,
        output_waveform,
    ):
        """测试上下文管理器"""
        print("\n=== 测试上下文管理器 ===")

        # 使用上下文管理器
        with SingleChasCSIO(
            ai_channels=ai_channels_single,
            ao_channels_static=ao_channels_static_single,
            ao_channels_feedback=(),
            static_output_waveform=output_waveform,
            feedback_function=self._dummy_feedback_function,
            export_function=lambda *args: None,
        ) as sync_io:
            # 启动任务
            print("启动任务...")
            sync_io.start()

            # 等待一段时间
            time.sleep(1.0)

            # 上下文管理器应该自动停止任务

        # 验证任务已停止
        assert sync_io._is_running is False

        print("✓ 上下文管理器测试通过")
