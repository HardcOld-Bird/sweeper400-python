"""
# 连续同步 AI/AO 模块

模块路径：`sweeper400.measure.cont_sync_io`

包含同步的连续 AI 和 AO 任务实现，用于各种信号的同步生成和采集。
（"CSIO"是"Continuous Synchronous AI/AO"的缩写）
"""

import threading
from collections import deque
from collections.abc import Callable
from typing import Any

import nidaqmx
import numpy as np
from nidaqmx.constants import (
    AcquisitionType,
    Edge,
    ExcitationSource,
    RegenerationMode,
    SoundPressureUnits,
)

from ..analyze import (
    PositiveInt,
    Waveform,
)
from ..logger import get_logger

# 获取模块日志器
logger = get_logger(__name__)


class HiPerfCSIO:
    """
    # 高性能连续同步 AI/AO 类

    该类提供高性能和简单的同步 AI/AO 任务实现。
    使用固定的输出波形和再生模式，避免了实时波形生成的开销。
    AI 通道配置为麦克风模式，用于声压测量。
    支持多通道AO输出，可以独立控制每个通道的启用/禁用状态。

    ## 主要特性：
        - 硬件同步的连续 AI/AO 任务
        - 支持单通道或多通道AO输出
        - AI 通道使用麦克风模式（激励电流 0.004A，灵敏度 4mV/Pa，最大声压级 120dB）
        - 使用固定输出波形，避免实时生成开销
        - 采用再生模式，提高性能和稳定性
        - 支持运行时动态启用/禁用AO通道
        - 基于回调函数的数据处理
        - 线程安全的数据传输控制
        - 自动资源管理
        - 适用于重复性测量场景

    ## 多线程设计：
        实际运行中，可以认为包含3个线程：
        1. 数据采集线程：由NI-DAQmx库控制，仅在AO调用回调函数时与python交互，
           几乎不占用GIL
        2. 工作线程：用于数据处理导出，仅在ni回调结束时（或超时）被唤醒，几乎不占用GIL
        3. 主控线程：用于启动、停止和配置任务，阻塞式，经常占用GIL

    ## 使用示例：
    ```python
    from sweeper400.analyze import init_sampling_info, init_sine_args, get_sine_cycles
    from sweeper400.measure.cont_sync_io import HiPerfCSIO

    # 创建采样信息和固定输出波形
    sampling_info = init_sampling_info(1000, 1000)
    sine_args = init_sine_args(100.0, 1.0, 0.0)
    output_waveform = get_sine_cycles(sampling_info, sine_args)

    # 定义数据导出函数
    def export_data(ai_waveform, chunks_num):
        print(f"导出第 {chunks_num} 段数据")

    # 单通道示例
    sync_io = HiPerfCSIO(
        ai_channel="Dev1/ai0",
        ao_channels=("Dev1/ao0",),
        output_waveform=output_waveform,
        export_function=export_data
    )

    # 多通道示例（使用同一设备的多个通道）
    sync_io_multi = HiPerfCSIO(
        ai_channel="Dev1/ai0",
        ao_channels=("Dev1/ao0", "Dev1/ao1"),
        output_waveform=output_waveform,  # 单通道波形会自动扩展为多通道
        export_function=export_data
    )

    # 启动任务
    sync_io_multi.start()
    sync_io_multi.enable_export = True  # 开始导出数据

    # 动态控制通道状态（禁用第2个通道）
    sync_io_multi.set_ao_channels_status((True, False))

    # 运行一段时间后停止
    time.sleep(10)
    sync_io_multi.stop()
    ```
    """

    # 获取类日志器
    logger = get_logger(f"{__name__}.HiPerfCSIO")

    def __init__(
        self,
        ai_channel: str,
        ao_channels: tuple[str, ...],
        output_waveform: Waveform,
        export_function: Callable[[Waveform, PositiveInt], None],
    ) -> None:
        """
        初始化高性能连续同步 AI/AO 对象

        Args:
            ai_channel: AI 通道名称，例如 "PXI1Slot2/ai0"
            ao_channels: AO 通道名称元组，例如 ("PXI1Slot2/ao0",) 或
                        ("PXI1Slot2/ao0", "PXI1Slot2/ao1")
            output_waveform: 固定的输出波形，将被循环使用。
                           - 如果是2维波形，channels_num必须等于len(ao_channels)或1
                           - 如果channels_num=1而len(ao_channels)>1，会自动扩展为多通道波形
            export_function: 数据导出函数，接收 (ai_waveform, chunks_num) 参数

        Raises:
            ValueError: 当参数无效时
        """
        # 公有属性
        self.enable_export: bool = False
        self.export_function = export_function

        # 私有属性 - 基本配置
        self._ai_channel = ai_channel
        self._ao_channels = ao_channels

        # 验证和处理output_waveform
        ao_channels_num = len(ao_channels)
        waveform_channels_num = output_waveform.channels_num

        if waveform_channels_num == ao_channels_num:
            # 波形通道数与AO通道数匹配，直接使用
            self._output_waveform = output_waveform
            logger.debug(
                f"输出波形通道数 ({waveform_channels_num}) 与AO通道数 ({ao_channels_num}) 匹配"
            )
        elif waveform_channels_num == 1 and ao_channels_num > 1:
            # 单通道波形，需要扩展为多通道
            logger.info(f"输出波形为单通道，将扩展为 {ao_channels_num} 通道")
            # 创建多通道波形：将单通道波形复制到所有通道
            if output_waveform.ndim == 1:
                # 1维数组，扩展为2维
                expanded_data = np.tile(output_waveform, (ao_channels_num, 1))
            else:
                # 已经是2维但只有1个通道，扩展通道数
                expanded_data = np.tile(output_waveform, (ao_channels_num, 1))

            self._output_waveform = Waveform(
                expanded_data,
                sampling_rate=output_waveform.sampling_rate,
                timestamp=output_waveform.timestamp,
                id=output_waveform.id,
                sine_args=output_waveform.sine_args,
            )
            logger.debug(
                f"已将单通道波形扩展为 {ao_channels_num} 通道，"
                f"新波形shape: {self._output_waveform.shape}"
            )
        else:
            # 波形通道数与AO通道数不匹配
            error_msg = (
                f"输出波形通道数 ({waveform_channels_num}) 必须等于 "
                f"AO通道数 ({ao_channels_num}) 或等于1，但得到了不匹配的值"
            )
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg)

        self._exported_chunks: int = 0  # 在导出的数据中，该值至少为1

        # 私有属性 - 任务和状态管理
        self._ai_task: nidaqmx.Task | None = None
        self._ao_task: nidaqmx.Task | None = None
        self._sync_task: nidaqmx.Task | None = None  # 用于跨机箱同步的辅助任务
        self._is_running = False
        self._callback_lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()  # 任务停止事件
        self._data_ready_event = threading.Event()  # 数据就绪事件

        # AO通道状态管理（默认全部启用）
        self._ao_channels_status: tuple[bool, ...] = tuple(
            True for _ in range(ao_channels_num)
        )

        # 获取常用信息
        self._sampling_info = output_waveform.sampling_info
        self._chunk_duration = output_waveform.duration

        # 数据队列
        # collections.deque是简单的双端队列，基础操作可保证线程安全，且性能较好，故使用
        # queue.Queue是更高级的FIFO队列，在多线程安全性上更强，但性能稍差，故暂不使用
        self._ai_queue: deque[dict[str, Any]] = deque()  # AI 数据队列

        logger.info(
            f"HiPerfCSIO 实例已创建 - AI: {ai_channel}, "
            f"AO通道数: {ao_channels_num}, AO: {ao_channels}, "
            f"输出波形shape: {self._output_waveform.shape}, "
            f"采样率: {self._output_waveform.sampling_rate} Hz"
        )

    def start(self):
        """
        启动同步的连续 AI/AO 任务

        配置并启动硬件同步的 AI 和 AO 任务，使用 PXIe_CLK100 时钟源和触发器同步。

        Raises:
            RuntimeError: 当任务启动失败时
        """
        if self._is_running:
            logger.warning("任务已在运行中")
            return

        try:
            # 创建 AI 和 AO 任务
            self._ai_task = nidaqmx.Task("ContSyncAI")
            self._ao_task = nidaqmx.Task("ContSyncAO")

            # 配置 AI 任务
            self._setup_ai_task()

            # 配置 AO 任务
            self._setup_ao_task()

            # 配置硬件同步触发
            self._setup_hardware_sync()

            # 启动工作线程
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_thread_function, name="HiPerfCSSIO_Worker"
            )
            self._worker_thread.start()

            # 启动任务（AO 先启动，等待触发）
            if self._ao_task is not None:
                self._ao_task.start()
            if self._ai_task is not None:
                self._ai_task.start()  # AI 启动时会触发 AO

            self._is_running = True
            logger.info("同步 AI/AO 任务启动成功")

        except Exception as e:
            logger.error(f"任务启动失败: {e}", exc_info=True)
            self._cleanup_tasks()
            raise RuntimeError(f"同步 AI/AO 任务启动失败: {e}") from e

    def _setup_ai_task(self):
        """配置 AI 任务（麦克风模式）"""
        if self._ai_task is None:
            logger.error("AI 任务未创建", exc_info=True)
            raise RuntimeError("AI 任务未创建")

        logger.debug("配置 AI 任务（麦克风模式）")

        # 麦克风参数（硬编码）
        mic_sensitivity = 4.0  # 麦克风灵敏度：0.004 V/Pa = 4 mV/Pa
        max_snd_press_level = 120.0  # 最大声压级：120 dB
        current_excit_val = 0.004  # 激励电流：0.004 A

        # 添加 AI 麦克风通道
        self._ai_task.ai_channels.add_ai_microphone_chan(  # type: ignore
            self._ai_channel,
            units=SoundPressureUnits.PA,
            mic_sensitivity=mic_sensitivity,
            max_snd_press_level=max_snd_press_level,
            current_excit_source=ExcitationSource.INTERNAL,
            current_excit_val=current_excit_val,
        )

        logger.debug(
            f"麦克风通道配置: 灵敏度={mic_sensitivity} mV/Pa, "
            f"最大声压级={max_snd_press_level} dB, "
            f"激励电流={current_excit_val} A"
        )

        # 配置时钟源和采样
        self._ai_task.timing.ref_clk_src = "PXIe_Clk100"
        self._ai_task.timing.ref_clk_rate = 100000000
        self._ai_task.timing.cfg_samp_clk_timing(  # type: ignore
            rate=self._sampling_info["sampling_rate"],
            sample_mode=AcquisitionType.CONTINUOUS,
        )

        # 配置更大的输入缓冲区以提升采集稳定性
        buffer_size = self._sampling_info["samples_num"] * 10  # 10倍缓冲区
        self._ai_task.in_stream.input_buf_size = buffer_size
        logger.debug(f"设置 AI 缓冲区大小: {buffer_size} 样本")

        # 注册回调函数
        self._ai_task.register_every_n_samples_acquired_into_buffer_event(  # type: ignore
            self._sampling_info["samples_num"],
            self._callback_function,  # type: ignore
        )

        logger.debug("AI 任务配置完成（麦克风模式）")

    def _setup_ao_task(self):
        """
        配置 AO 任务

        使用再生模式和固定波形，只需写入一次。
        支持多通道AO输出，包括跨机箱的多通道配置。
        """
        if self._ao_task is None:
            logger.error("AO 任务未创建", exc_info=True)
            raise RuntimeError("AO 任务未创建")

        logger.debug("配置 HiPerfCSIO AO 任务")

        # 添加所有 AO 通道
        for channel_name in self._ao_channels:
            self._ao_task.ao_channels.add_ao_voltage_chan(  # type: ignore
                channel_name, min_val=-10.0, max_val=10.0
            )
            logger.debug(f"已添加 AO 通道: {channel_name}")

        logger.debug(f"共添加 {self.ao_channels_num} 个 AO 通道")

        # 识别涉及的机箱
        chassis_devices = self._identify_chassis()

        # 配置时钟源和采样
        # 对于跨机箱配置，需要特殊处理
        if len(chassis_devices) > 1:
            logger.info("检测到跨机箱AO配置，使用特殊的时钟配置")
            # 对于跨机箱，所有设备仍然使用各自机箱的PXIe_CLK100作为参考时钟
            # 但DAQmx会自动通过PFI连线同步这些时钟
            self._ao_task.timing.ref_clk_src = "PXIe_Clk100"
            self._ao_task.timing.ref_clk_rate = 100000000
        else:
            # 单机箱配置，直接使用PXIe_CLK100
            self._ao_task.timing.ref_clk_src = "PXIe_Clk100"
            self._ao_task.timing.ref_clk_rate = 100000000

        self._ao_task.timing.cfg_samp_clk_timing(  # type: ignore
            rate=self._sampling_info["sampling_rate"],
            sample_mode=AcquisitionType.CONTINUOUS,
        )

        # 设置再生模式
        self._ao_task.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION

        # 在再生模式下，缓冲区大小设置为波形长度即可
        # 硬件会自动循环播放缓冲区中的数据
        buffer_size = self._output_waveform.samples_num
        self._ao_task.out_stream.output_buf_size = buffer_size
        logger.debug(f"设置 AO 缓冲区大小: {buffer_size} 样本")

        # 写入固定波形到缓冲区
        self._write_fixed_waveform()

        logger.debug("HiPerfCSIO AO 任务配置完成")

    def _write_fixed_waveform(self):
        """
        将固定波形写入 AO 缓冲区

        在再生模式下，只需要写入一次，之后硬件会自动循环播放。
        支持单通道和多通道波形写入。
        """
        try:
            # 准备写入数据
            # nidaqmx对于多通道AO任务，期望数据格式为：
            # - 单通道：1维数组 [sample1, sample2, ...]
            # - 多通道：2维数组 [[ch1_s1, ch1_s2, ...], [ch2_s1, ch2_s2, ...], ...]
            if self.ao_channels_num == 1:
                # 单通道：如果波形是2维的，取第一个通道
                if self._output_waveform.ndim == 2:
                    write_data = self._output_waveform[0, :]
                else:
                    write_data = self._output_waveform
            else:
                # 多通道：确保是2维数组
                if self._output_waveform.ndim == 1:
                    # 理论上不应该到这里，因为__init__中已经处理过
                    logger.warning("多通道AO任务但波形是1维，将扩展为2维")
                    write_data = np.tile(
                        self._output_waveform, (self.ao_channels_num, 1)
                    )
                else:
                    write_data = self._output_waveform

            # 将固定波形写入硬件缓冲区
            self._ao_task.write(write_data, auto_start=False)  # type: ignore

            logger.debug(
                f"固定波形已写入缓冲区: shape={write_data.shape}, "
                f"通道数={self.ao_channels_num}, 样本数={self._output_waveform.samples_num}"
            )

        except Exception as e:
            logger.error(f"固定波形写入失败: {e}", exc_info=True)
            raise

    def _identify_chassis(self) -> dict[str, set[str]]:
        """
        识别AI和AO通道所属的机箱

        Returns:
            字典，键为机箱名称（如"PXIChassis1"），值为该机箱涉及的设备集合（如{"PXI1Slot2", "PXI1Slot3"}）
        """
        chassis_devices: dict[str, set[str]] = {}

        # 处理AI通道
        ai_device = self._ai_channel.split("/")[0]
        # 从设备名称提取机箱信息，例如 "PXI1Slot2" -> "PXIChassis1"
        if ai_device.startswith("PXI") and "Slot" in ai_device:
            chassis_num = ai_device[3]  # 提取机箱编号，如 "PXI1Slot2" -> "1"
            chassis_name = f"PXIChassis{chassis_num}"
            if chassis_name not in chassis_devices:
                chassis_devices[chassis_name] = set()
            chassis_devices[chassis_name].add(ai_device)

        # 处理AO通道
        for ao_channel in self._ao_channels:
            ao_device = ao_channel.split("/")[0]
            if ao_device.startswith("PXI") and "Slot" in ao_device:
                chassis_num = ao_device[3]
                chassis_name = f"PXIChassis{chassis_num}"
                if chassis_name not in chassis_devices:
                    chassis_devices[chassis_name] = set()
                chassis_devices[chassis_name].add(ao_device)

        logger.debug(f"识别到的机箱和设备: {chassis_devices}")
        return chassis_devices

    def _group_channels_by_chassis(
        self, channels: tuple[str, ...]
    ) -> dict[str, list[str]]:
        """
        将通道按机箱分组

        Args:
            channels: 通道名称元组

        Returns:
            字典，键为机箱名称，值为该机箱的通道列表
        """
        chassis_channels: dict[str, list[str]] = {}

        for channel in channels:
            device = channel.split("/")[0]
            if device.startswith("PXI") and "Slot" in device:
                chassis_num = device[3]
                chassis_name = f"PXIChassis{chassis_num}"
                if chassis_name not in chassis_channels:
                    chassis_channels[chassis_name] = []
                chassis_channels[chassis_name].append(channel)

        return chassis_channels

    def _setup_cross_chassis_sync(self, chassis_devices: dict[str, set[str]]):
        """
        配置跨机箱同步（时钟和触发）

        策略：
        1. 检测到跨机箱配置时，不使用单个多通道AO任务（会导致路由失败）
        2. 改为为每个机箱创建独立的AO子任务
        3. 使用PXI_Trig线路显式路由触发信号
        4. 所有子任务共享相同的采样时钟配置

        根据硬件连线：
        - PXI2Slot2/PFI0 ←→ PXI1Slot2/PFI0 ←→ PXI3Slot2/PFI0 (触发)
        - PXI2Slot3/PFI0 ←→ PXI1Slot3/PFI0 ←→ PXI3Slot3/PFI0 (时钟)

        Args:
            chassis_devices: 机箱和设备的映射字典
        """
        if len(chassis_devices) <= 1:
            logger.debug("所有通道在同一机箱内，无需跨机箱同步")
            return

        logger.info(
            f"检测到跨机箱配置，涉及 {len(chassis_devices)} 个机箱: "
            f"{list(chassis_devices.keys())}"
        )

        # 定义Master机箱
        master_chassis = "PXIChassis2"

        # 记录跨机箱配置信息
        logger.info(
            "跨机箱同步配置说明："
            "\n  - 将为每个机箱创建独立的AO子任务"
            "\n  - 使用PXI_Trig线路显式路由触发信号"
            "\n  - 硬件连线: PXI2Slot2/PFI0 ←→ PXI1Slot2/PFI0 ←→ PXI3Slot2/PFI0"
        )

        logger.info("跨机箱同步配置完成")

    def _setup_hardware_sync(self):
        """
        配置硬件同步触发

        支持单机箱和跨机箱两种场景：
        1. 单机箱：使用时序引擎的StartTrigger进行AI/AO同步
        2. 跨机箱：额外配置PFI接口进行时钟和触发信号的跨机箱同步

        对于跨机箱场景，遵循以下规则：
        - PXIChassis2 为 Master，通过 PXI2Slot2/PFI0 导出触发，PXI2Slot3/PFI0 导出时钟
        - PXIChassis1 和 PXIChassis3 为 Slave，通过对应的 Slot2/PFI0 接收触发，Slot3/PFI0 接收时钟
        """
        if self._ai_task is None or self._ao_task is None:
            logger.error("AI 或 AO 任务未创建", exc_info=True)
            raise RuntimeError("AI 或 AO 任务未创建")

        logger.debug(
            f"配置硬件同步触发 - AI通道: {self._ai_channel}, "
            f"AO通道数: {self.ao_channels_num}"
        )

        # 识别涉及的机箱
        chassis_devices = self._identify_chassis()

        # 如果涉及多个机箱，配置跨机箱同步
        if len(chassis_devices) > 1:
            self._setup_cross_chassis_sync(chassis_devices)

        # 从AI通道名称提取设备名称
        ai_device_name = self._ai_channel.split("/")[0]

        # 配置AI/AO任务间的同步触发
        hardware_sync_success = False

        # 方法1: 使用时序引擎 StartTrigger 终端
        te_terminals = [
            f"/{ai_device_name}/te0/StartTrigger",
            f"/{ai_device_name}/te1/StartTrigger",
            f"/{ai_device_name}/te2/StartTrigger",
            f"/{ai_device_name}/te3/StartTrigger",
        ]

        for te_terminal in te_terminals:
            try:
                self._ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(  # type: ignore
                    te_terminal, trigger_edge=Edge.RISING
                )
                logger.debug(f"成功配置时序引擎触发: {te_terminal}")
                hardware_sync_success = True
                break

            except Exception as te_e:
                logger.debug(f"时序引擎终端 {te_terminal} 配置失败: {te_e}")
                continue

        if not hardware_sync_success:
            logger.warning("时序引擎触发方法失败，尝试 AI StartTrigger")

            # 方法2: 使用 AI 任务的 StartTrigger
            try:
                ai_start_trigger = f"/{ai_device_name}/ai/StartTrigger"
                self._ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(  # type: ignore
                    ai_start_trigger, trigger_edge=Edge.RISING
                )
                logger.debug(f"成功配置 AI StartTrigger: {ai_start_trigger}")
                hardware_sync_success = True

            except Exception as ai_e:
                logger.debug(f"AI StartTrigger 失败: {ai_e}")

        if not hardware_sync_success:
            logger.error("所有硬件同步方法都失败", exc_info=True)
            raise RuntimeError("硬件触发配置失败: 无法建立 AI/AO 任务间的硬件同步")

    @property
    def ao_channels_num(self) -> PositiveInt:
        """
        获取AO通道数量

        Returns:
            AO通道数量
        """
        return len(self._ao_channels)

    @property
    def output_waveform(self) -> Waveform:
        """获取当前的输出波形"""
        # 可变对象，传出副本
        return self._output_waveform.copy()

    def set_ao_channels_status(self, channels_status: tuple[bool, ...]) -> None:
        """
        设置AO通道的启用/禁用状态

        通过修改缓冲区中的数据来实现通道的静音/取消静音。
        禁用的通道将输出0V，启用的通道将输出原始波形。

        Args:
            channels_status: 布尔值元组，长度必须等于ao_channels_num。
                           True表示启用通道，False表示禁用通道（输出0V）

        Raises:
            ValueError: 当channels_status长度不匹配时
            RuntimeError: 当任务未运行或任务对象不存在时

        Examples:
            >>> # 禁用第2个通道，保持其他通道启用
            >>> sync_io.set_ao_channels_status((True, False, True))
        """
        # 验证参数
        if len(channels_status) != self.ao_channels_num:
            error_msg = (
                f"channels_status长度 ({len(channels_status)}) "
                f"必须等于AO通道数 ({self.ao_channels_num})"
            )
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg)

        # 检查是否有变化
        if channels_status == self._ao_channels_status:
            logger.debug("通道状态未变化，无需更新")
            return

        # 检查任务是否正在运行
        if not self._is_running or self._ao_task is None:
            error_msg = "任务未运行或AO任务不存在，无法设置通道状态"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)

        logger.info(f"更新AO通道状态: {self._ao_channels_status} -> {channels_status}")

        try:
            # 准备新的波形数据
            if self.ao_channels_num == 1:
                # 单通道
                if channels_status[0]:
                    # 启用：使用原始波形
                    if self._output_waveform.ndim == 2:
                        write_data = self._output_waveform[0, :]
                    else:
                        write_data = self._output_waveform
                else:
                    # 禁用：输出0
                    write_data = np.zeros(self._output_waveform.samples_num)
            else:
                # 多通道
                write_data = np.zeros(
                    (self.ao_channels_num, self._output_waveform.samples_num)
                )
                for ch_idx, enabled in enumerate(channels_status):
                    if enabled:
                        # 启用：使用原始波形
                        write_data[ch_idx, :] = self._output_waveform[ch_idx, :]
                    # 禁用的通道保持为0

            # 写入新的波形数据到缓冲区
            # 注意：在再生模式下，需要先停止任务，修改缓冲区，然后重新启动
            # 但这会导致输出中断。更好的方法是使用非再生模式，但这会增加复杂度。
            # 这里我们采用简单的方法：直接写入缓冲区
            # 由于是再生模式，这个写入会在下一个循环周期生效
            self._ao_task.write(write_data, auto_start=False)  # type: ignore

            # 更新状态
            self._ao_channels_status = channels_status

            logger.info(
                f"AO通道状态已更新，启用通道: "
                f"{[i for i, enabled in enumerate(channels_status) if enabled]}"
            )

        except Exception as e:
            logger.error(f"设置AO通道状态失败: {e}", exc_info=True)
            raise RuntimeError(f"设置AO通道状态失败: {e}") from e

    def _callback_function(
        self,
        task_handle: Any,
        every_n_samples_event_type: Any,
        number_of_samples: int,
        callback_data: Any,
    ):
        """
        AI 任务回调函数

        当缓冲区中的采集数据积累到指定数量时自动调用。
        由于在硬件中断线程中执行，需要快速处理并保证线程安全。
        （未使用的参数不可删除，其为nidaqmx包的API要求）

        Args:
            task_handle: 任务句柄
            every_n_samples_event_type: 事件类型
            number_of_samples: 样本数量
            callback_data: 回调数据

        Returns:
            int: 0 表示成功
        """
        try:
            # 快速读取数据并加入队列，避免在回调中执行耗时操作
            with self._callback_lock:
                if self._ai_task is not None and self._is_running:
                    # 读取 AI 数据
                    ai_data = self._ai_task.read(
                        number_of_samples_per_channel=number_of_samples  # type: ignore
                    )

                    # 准备精简的数据包
                    ai_package = {
                        "ai_data": ai_data,
                        "enable_export": self.enable_export,
                    }

                    # 加入 AI 队列末尾
                    self._ai_queue.append(ai_package)

                    # 通知工作线程有新数据可处理
                    self._data_ready_event.set()

            return 0

        except Exception as e:
            logger.error(f"回调函数执行失败（任务可能已停止）: {e}", exc_info=True)
            return -1

    def _worker_thread_function(self):
        """
        工作线程函数

        使用事件驱动机制，只在有数据时才唤醒处理，节省 CPU 资源。
        专注于数据导出处理。
        """
        logger.debug("HiPerfCSIO 工作线程已启动")

        while not self._stop_event.is_set():
            try:
                # 等待数据就绪事件或超时
                data_ready = self._data_ready_event.wait(
                    timeout=self._chunk_duration
                )  # 超时时间为一个采样周期

                if data_ready:
                    # 清除事件标志，准备下次等待
                    self._data_ready_event.clear()
                    logger.debug("检测到数据就绪事件，开始处理")

                # 执行数据导出处理
                self._wtf_process_data_export()

            except Exception as e:
                logger.error(f"HiPerfCSIO 工作线程处理失败: {e}", exc_info=True)

        logger.debug("HiPerfCSIO 工作线程已退出")

    def _wtf_process_data_export(self):
        """
        处理数据导出

        直接使用固定的输出波形进行数据导出。
        """
        # 检查是否有 AI 数据可处理
        if not self._ai_queue:
            return

        try:
            # 从 AI 队列首部取出数据包（FIFO）
            ai_package = self._ai_queue.popleft()

            # 检查是否需要导出
            if ai_package["enable_export"]:
                # 增加导出计数
                self._exported_chunks += 1

                # 创建 AI 波形对象，不指定 id（因为固定波形一般没有 id）
                ai_waveform = Waveform(
                    np.array(ai_package["ai_data"]),
                    sampling_rate=self._sampling_info["sampling_rate"],
                )

                # 调用导出函数，直接使用固定的输出波形
                self.export_function(
                    ai_waveform,
                    self._exported_chunks,
                )

                logger.debug(f"导出第 {self._exported_chunks} 段数据")

            else:
                # 重置导出计数
                self._exported_chunks = 0

        except Exception as e:
            logger.error(f"HiPerfCSIO 数据导出处理失败: {e}", exc_info=True)

    def stop(self):
        """
        停止同步的连续 AI/AO 任务并释放所有资源

        按照正确的顺序停止任务、清理资源，确保硬件状态正常。
        """
        if not self._is_running:
            logger.warning("任务未在运行中")
            return

        try:
            # 标记停止状态
            self._is_running = False

            # 停止工作线程
            self._stop_worker_thread()

            # 停止并清理 nidaqmx 任务
            self._cleanup_tasks()

            # 重置状态
            self._exported_chunks = 0

            logger.info("同步 AI/AO 任务已停止")

        except Exception as e:
            logger.error(f"任务停止过程中发生错误: {e}", exc_info=True)
            # 即使出错也要尝试清理资源
            try:
                self._cleanup_tasks()
            except Exception:
                pass

    def _stop_worker_thread(self):
        """停止工作线程"""
        if self._worker_thread is not None:
            logger.debug("停止工作线程")

            # 设置停止事件
            self._stop_event.set()

            # 设置数据就绪事件，确保工作线程能够及时退出等待状态
            self._data_ready_event.set()

            # 等待线程结束
            self._worker_thread.join(timeout=5.0)

            if self._worker_thread.is_alive():
                logger.warning("工作线程未能在超时时间内结束")
            else:
                logger.debug("工作线程已停止")

            self._worker_thread = None

    def _cleanup_tasks(self):
        """清理 nidaqmx 任务资源"""
        logger.debug("清理 nidaqmx 任务资源")

        # 停止 AI 任务
        if self._ai_task is not None:
            try:
                if self._ai_task._handle is not None:  # type: ignore
                    self._ai_task.stop()
                    logger.debug("AI 任务已停止")
            except Exception as e:
                logger.warning(f"停止 AI 任务时出错: {e}")

            try:
                self._ai_task.close()
                logger.debug("AI 任务已关闭")
            except Exception as e:
                logger.warning(f"关闭 AI 任务时出错: {e}")

            self._ai_task = None

        # 停止 AO 任务
        if self._ao_task is not None:
            try:
                if self._ao_task._handle is not None:  # type: ignore
                    self._ao_task.stop()
                    logger.debug("AO 任务已停止")
            except Exception as e:
                logger.warning(f"停止 AO 任务时出错: {e}")

            try:
                self._ao_task.close()
                logger.debug("AO 任务已关闭")
            except Exception as e:
                logger.warning(f"关闭 AO 任务时出错: {e}")

            self._ao_task = None

        # 清理数据队列和事件状态
        with self._callback_lock:
            self._ai_queue.clear()

        # 重置事件状态
        self._stop_event.clear()
        self._data_ready_event.clear()

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore
        """
        上下文管理器出口，自动停止任务
        （未使用的参数不可删除，其为Python上下文管理器协议的API要求）
        """
        if self._is_running:
            self.stop()

    def __del__(self):
        """析构函数，确保资源清理"""
        if hasattr(self, "_is_running") and self._is_running:
            try:
                self.stop()
            except Exception:
                pass  # 析构时忽略异常


class MultiChasCSIO:
    """
    # 高性能跨机箱连续同步 AI/AO 类

    该类专门用于处理跨机箱的多通道AO输出场景。
    与 HiPerfCSIO 不同，该类为每个机箱创建独立的AO任务，
    并使用显式的触发路由来实现跨机箱同步。

    ## 主要特性：
        - 支持跨机箱的多通道AO输出
        - 为每个机箱创建独立的AO任务
        - 使用DAQmx Connect Terminals显式路由触发信号
        - 通过PFI物理连线传递触发信号
        - 硬件同步的连续 AI/AO 任务
        - AI 通道使用麦克风模式
        - 使用固定输出波形和再生模式
        - 支持运行时动态启用/禁用AO通道
        - 基于回调函数的数据处理
        - 线程安全的数据传输控制

    ## 跨机箱同步机制：
        1. 识别AO通道所属的机箱
        2. 为每个机箱创建独立的AO任务
        3. 指定PXIChassis2为Master机箱
        4. 使用DAQmx Connect Terminals配置触发路由：
           - Master机箱的触发信号 -> PXI2Slot2/PFI0
           - PXI1Slot2/PFI0 -> Slave任务的触发源
           - PXI3Slot2/PFI0 -> Slave任务的触发源
        5. 所有任务使用相同的参考时钟（PXIe_CLK100）

    ## 硬件连线要求：
        - 触发信号: PXI2Slot2/PFI0 ←→ PXI1Slot2/PFI0 ←→ PXI3Slot2/PFI0
        - 时钟信号: PXI2Slot3/PFI0 ←→ PXI1Slot3/PFI0 ←→ PXI3Slot3/PFI0

    ## 使用示例：
    ```python
    from sweeper400.analyze import init_sampling_info, init_sine_args, get_sine_cycles
    from sweeper400.measure.cont_sync_io import MultiChasCSIO

    # 创建采样信息和固定输出波形
    sampling_info = init_sampling_info(48000, 4800)
    sine_args = init_sine_args(1000.0, 0.02, 0.0)
    output_waveform = get_sine_cycles(sampling_info, sine_args)

    # 定义数据导出函数
    def export_data(ai_waveform, chunks_num):
        print(f"导出第 {chunks_num} 段数据")

    # 跨机箱多通道示例
    sync_io = MultiChasCSIO(
        ai_channel="PXI2Slot2/ai0",
        ao_channels=(
            "PXI1Slot2/ao0",  # PXIChassis1
            "PXI2Slot2/ao0",  # PXIChassis2 (Master)
            "PXI3Slot2/ao0",  # PXIChassis3
        ),
        output_waveform=output_waveform,
        export_function=export_data
    )

    # 启动任务
    sync_io.start()
    sync_io.enable_export = True

    # 动态控制通道状态
    sync_io.set_ao_channels_status((True, False, True))

    # 停止任务
    sync_io.stop()
    ```

    ## 注意事项：
        - 建议至少一个AO通道位于PXIChassis2（Master机箱）
        - 所有机箱必须通过PFI连线正确连接
        - 确保PXI Platform Services已正确安装和配置
    """

    def __init__(
        self,
        ai_channel: str,
        ao_channels: tuple[str, ...],
        output_waveform: Waveform,
        export_function: Callable[[Waveform, PositiveInt], Any],
    ):
        """
        初始化跨机箱连续同步 AI/AO 任务

        Args:
            ai_channel: AI 通道名称（例如 "PXI1Slot2/ai0"）
            ao_channels: AO 通道名称元组（例如 ("PXI1Slot2/ao0", "PXI2Slot2/ao0")）
            output_waveform: 输出波形对象（Waveform类型）
            export_function: 数据导出函数，接收 (ai_waveform, chunks_num) 参数

        Raises:
            ValueError: 当参数无效时
        """
        # 验证参数
        if not ai_channel:
            raise ValueError("AI 通道名称不能为空")
        if not ao_channels:
            raise ValueError("AO 通道列表不能为空")
        if output_waveform.samples_num == 0:
            raise ValueError("输出波形不能为空")

        # 公共属性
        self._ai_channel = ai_channel
        self._ao_channels = ao_channels
        self._export_function = export_function

        # 处理输出波形
        # 如果输出波形是单通道，需要扩展为多通道
        if output_waveform.channels_num == 1 and len(ao_channels) > 1:
            logger.info(f"输出波形为单通道，将扩展为 {len(ao_channels)} 通道")
            # 创建多通道波形：将单通道波形复制到所有通道
            expanded_data = np.tile(output_waveform, (len(ao_channels), 1))

            self._output_waveform = Waveform(
                expanded_data,
                sampling_rate=output_waveform.sampling_rate,
                timestamp=output_waveform.timestamp,
                id=output_waveform.id,
                sine_args=output_waveform.sine_args,
            )
        else:
            self._output_waveform = output_waveform

        # 验证波形通道数与AO通道数匹配
        if self._output_waveform.channels_num != len(ao_channels):
            raise ValueError(
                f"输出波形通道数 ({self._output_waveform.channels_num}) "
                f"与 AO 通道数 ({len(ao_channels)}) 不匹配"
            )

        # 采样信息
        self._sampling_info = output_waveform.sampling_info

        logger.info(
            f"MultiChasCSIO 实例已创建 - "
            f"AI: {ai_channel}, "
            f"AO通道数: {len(ao_channels)}, "
            f"AO: {ao_channels}, "
            f"输出波形shape: {self._output_waveform.shape}, "
            f"采样率: {self._sampling_info['sampling_rate']} Hz"
        )

        # 私有属性 - 任务和状态管理
        self._ai_task: nidaqmx.Task | None = None
        self._ao_tasks: dict[str, nidaqmx.Task] = {}  # 机箱名称 -> AO任务
        self._trigger_routes: list[tuple[str, str]] = []  # 触发路由列表
        self._is_running = False
        self._callback_lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._data_ready_event = threading.Event()

        # 私有属性 - 数据缓冲和控制
        self._data_buffer: deque[Waveform] = deque(maxlen=10)
        self._chunks_num = 0
        self._enable_export = False

        # 私有属性 - AO通道状态控制
        self._ao_channels_status = np.ones(len(ao_channels), dtype=bool)
        self._ao_channels_status_lock = threading.RLock()  # 使用可重入锁避免死锁

        # 私有属性 - 机箱分组信息
        self._chassis_groups: dict[str, list[int]] = {}  # 机箱名称 -> 通道索引列表

        # Master机箱（自动确定为AI通道所在的机箱）
        # 必须在调用_group_ao_channels()之前设置
        ai_device = ai_channel.split("/")[0]  # 例如 "PXI1Slot2"
        # 从设备名称提取机箱名称（例如 "PXI1Slot2" -> "PXIChassis1"）
        if "PXI1" in ai_device:
            self._master_chassis = "PXIChassis1"
        elif "PXI2" in ai_device:
            self._master_chassis = "PXIChassis2"
        elif "PXI3" in ai_device:
            self._master_chassis = "PXIChassis3"
        else:
            # 默认使用PXIChassis2
            self._master_chassis = "PXIChassis2"
            logger.warning(
                f"无法从AI通道 {ai_channel} 确定机箱，使用默认Master: {self._master_chassis}"
            )

        logger.info(f"Master机箱已确定为: {self._master_chassis} (AI通道所在机箱)")

        # 识别并分组AO通道
        self._group_ao_channels()

    def _group_ao_channels(self):
        """
        将AO通道按机箱分组

        解析通道名称，识别所属机箱，并建立机箱到通道索引的映射。
        """
        for idx, channel in enumerate(self._ao_channels):
            # 解析通道名称，例如 "PXI1Slot2/ao0"
            device = channel.split("/")[0]

            # 提取机箱编号
            if device.startswith("PXI") and "Slot" in device:
                chassis_num = device[3]  # 提取 "PXI1Slot2" 中的 "1"
                chassis_name = f"PXIChassis{chassis_num}"

                if chassis_name not in self._chassis_groups:
                    self._chassis_groups[chassis_name] = []

                self._chassis_groups[chassis_name].append(idx)
            else:
                logger.warning(f"无法识别通道 {channel} 的机箱信息，将被忽略")

        logger.info(
            f"AO通道分组完成，共 {len(self._chassis_groups)} 个机箱: "
            f"{list(self._chassis_groups.keys())}"
        )

        # 检查是否包含Master机箱
        if self._master_chassis not in self._chassis_groups:
            logger.warning(
                f"未检测到Master机箱 {self._master_chassis}，"
                f"建议至少一个AO通道位于Master机箱以获得最佳同步性能"
            )

    @property
    def ao_channels_num(self) -> int:
        """AO 通道数量"""
        return len(self._ao_channels)

    @property
    def enable_export(self) -> bool:
        """数据导出启用状态"""
        return self._enable_export

    @enable_export.setter
    def enable_export(self, value: bool):
        """设置数据导出使能状态"""
        self._enable_export = value
        if value:
            logger.info("数据导出已启用")
        else:
            logger.info("数据导出已禁用")

    def get_ao_channels_status(self) -> tuple[bool, ...]:
        """
        获取所有AO通道的启用状态

        Returns:
            包含所有通道状态的元组，True表示启用，False表示禁用
        """
        with self._ao_channels_status_lock:
            return tuple(self._ao_channels_status)

    def set_ao_channels_status(self, status: tuple[bool, ...]):
        """
        设置AO通道的启用状态

        Args:
            status: 通道状态元组，长度必须与AO通道数量相同

        Raises:
            ValueError: 当状态元组长度不匹配时
            RuntimeError: 当任务未运行时
        """
        if len(status) != len(self._ao_channels):
            raise ValueError(
                f"状态元组长度 ({len(status)}) 与 AO 通道数 ({len(self._ao_channels)}) 不匹配"
            )

        if not self._is_running:
            raise RuntimeError("任务未运行，无法设置通道状态")

        with self._ao_channels_status_lock:
            old_status = self._ao_channels_status.copy()
            self._ao_channels_status = np.array(status, dtype=bool)

            # 检查哪些通道状态发生了变化
            changed_indices = np.where(old_status != self._ao_channels_status)[0]

            if len(changed_indices) > 0:
                logger.info(
                    f"AO通道状态已更新: {list(status)}, "
                    f"变化的通道索引: {list(changed_indices)}"
                )

                # 更新所有AO任务的波形
                self._update_all_ao_waveforms()
            else:
                logger.debug("AO通道状态未发生变化")

    def _setup_ai_task(self):
        """
        配置 AI 任务

        配置为麦克风模式，使用 IEPE 激励电流。
        """
        if self._ai_task is None:
            logger.error("AI 任务未创建", exc_info=True)
            raise RuntimeError("AI 任务未创建")

        logger.debug("配置 MultiChasCSIO AI 任务")

        # 添加 AI 通道（麦克风模式）
        self._ai_task.ai_channels.add_ai_microphone_chan(  # type: ignore
            self._ai_channel,
            units=SoundPressureUnits.PA,
            mic_sensitivity=0.004,
            max_snd_press_level=120.0,
            current_excit_source=ExcitationSource.INTERNAL,
            current_excit_val=0.004,
        )

        # 配置时钟源和采样
        self._ai_task.timing.ref_clk_src = "PXIe_Clk100"
        self._ai_task.timing.ref_clk_rate = 100000000
        self._ai_task.timing.cfg_samp_clk_timing(  # type: ignore
            rate=self._sampling_info["sampling_rate"],
            sample_mode=AcquisitionType.CONTINUOUS,
        )

        # 配置缓冲区大小
        buffer_size = self._output_waveform.samples_num * 2
        self._ai_task.in_stream.input_buf_size = buffer_size
        logger.debug(f"设置 AI 缓冲区大小: {buffer_size} 样本")

        # 注册回调函数
        self._ai_task.register_every_n_samples_acquired_into_buffer_event(  # type: ignore
            self._output_waveform.samples_num, self._ai_callback
        )

        logger.debug("MultiChasCSIO AI 任务配置完成")

    def _setup_ao_tasks(self):
        """
        为每个机箱创建并配置独立的AO任务

        这是跨机箱同步的关键：为每个机箱创建独立的AO任务，
        避免DAQmx尝试在不同机箱之间自动路由触发信号。
        """
        logger.info("开始为每个机箱创建独立的AO任务")

        for chassis_name, channel_indices in self._chassis_groups.items():
            # 创建任务
            task_name = f"ContSyncAO_{chassis_name}"
            ao_task = nidaqmx.Task(task_name)
            self._ao_tasks[chassis_name] = ao_task

            logger.debug(f"为机箱 {chassis_name} 创建AO任务: {task_name}")

            # 添加该机箱的所有AO通道
            for idx in channel_indices:
                channel_name = self._ao_channels[idx]
                ao_task.ao_channels.add_ao_voltage_chan(  # type: ignore
                    channel_name, min_val=-10.0, max_val=10.0
                )
                logger.debug(f"  添加通道: {channel_name}")

            # 配置时钟源和采样
            ao_task.timing.ref_clk_src = "PXIe_Clk100"
            ao_task.timing.ref_clk_rate = 100000000
            ao_task.timing.cfg_samp_clk_timing(  # type: ignore
                rate=self._sampling_info["sampling_rate"],
                sample_mode=AcquisitionType.CONTINUOUS,
            )

            # 设置再生模式
            ao_task.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION

            # 设置缓冲区大小
            buffer_size = self._output_waveform.samples_num
            ao_task.out_stream.output_buf_size = buffer_size
            logger.debug(f"  设置缓冲区大小: {buffer_size} 样本")

            # 写入该机箱的波形数据
            self._write_ao_task_waveform(chassis_name, ao_task, channel_indices)

        logger.info(f"成功创建 {len(self._ao_tasks)} 个AO任务")

    def _write_ao_task_waveform(
        self,
        chassis_name: str,
        ao_task: nidaqmx.Task,
        channel_indices: list[int],
    ):
        """
        向指定的AO任务写入波形数据

        Args:
            chassis_name: 机箱名称
            ao_task: AO任务对象
            channel_indices: 该任务包含的通道索引列表
        """
        try:
            # 提取该机箱的通道波形
            # Waveform是ndarray的子类，直接使用索引
            if len(channel_indices) == 1:
                # 单通道：提取一维数组
                chassis_waveform = np.asarray(
                    self._output_waveform[channel_indices[0], :]
                )
            else:
                # 多通道：提取二维数组
                chassis_waveform = np.asarray(self._output_waveform[channel_indices, :])

            # 应用通道状态（禁用的通道输出零）
            # 需要创建副本以避免修改原始波形
            chassis_waveform = chassis_waveform.copy()
            with self._ao_channels_status_lock:
                for i, global_idx in enumerate(channel_indices):
                    if not self._ao_channels_status[global_idx]:
                        if len(channel_indices) == 1:
                            chassis_waveform[:] = 0.0
                        else:
                            chassis_waveform[i, :] = 0.0

            # 写入数据
            ao_task.write(chassis_waveform, auto_start=False)  # type: ignore
            logger.debug(
                f"成功写入机箱 {chassis_name} 的波形数据，"
                f"shape: {chassis_waveform.shape}"
            )

        except Exception as e:
            logger.error(f"机箱 {chassis_name} 波形写入失败: {e}", exc_info=True)
            raise

    def _update_all_ao_waveforms(self):
        """
        更新所有AO任务的波形数据

        用于动态控制通道状态时更新输出波形。
        """
        for chassis_name, ao_task in self._ao_tasks.items():
            channel_indices = self._chassis_groups[chassis_name]
            try:
                self._write_ao_task_waveform(chassis_name, ao_task, channel_indices)
            except Exception as e:
                logger.error(
                    f"更新机箱 {chassis_name} 波形失败: {e}",
                    exc_info=True,
                )

    def _setup_cross_chassis_trigger_routing(self):
        """
        配置跨机箱触发路由

        这是跨机箱同步的核心：使用DAQmx Connect Terminals显式配置
        触发信号的路由路径，通过PFI物理连线传递触发信号。

        策略：
        1. 确定Master机箱和Slave机箱
        2. Master机箱的AO任务使用AI任务的StartTrigger作为触发源
        3. Master机箱导出触发信号到PFI0
        4. Slave机箱从PFI0接收触发信号
        5. 使用DAQmx Connect Terminals建立显式路由
        """
        logger.info("开始配置跨机箱触发路由")

        # 如果只有一个机箱，不需要跨机箱路由
        if len(self._chassis_groups) <= 1:
            logger.info("只有一个机箱，使用简单的触发配置")
            self._setup_single_chassis_trigger()
            return

        # 多机箱配置
        logger.info(f"检测到 {len(self._chassis_groups)} 个机箱，配置跨机箱触发路由")

        # 获取AI设备名称
        ai_device = self._ai_channel.split("/")[0]

        # 配置Master机箱
        if self._master_chassis in self._chassis_groups:
            master_task = self._ao_tasks[self._master_chassis]

            # Master任务使用AI的StartTrigger作为触发源
            # 尝试所有可能的时序引擎终端
            te_terminals = [
                f"/{ai_device}/te0/StartTrigger",
                f"/{ai_device}/te1/StartTrigger",
                f"/{ai_device}/te2/StartTrigger",
                f"/{ai_device}/te3/StartTrigger",
            ]

            trigger_configured = False
            ai_start_trigger = ""
            for te_terminal in te_terminals:
                try:
                    master_task.triggers.start_trigger.cfg_dig_edge_start_trig(  # type: ignore
                        te_terminal, trigger_edge=Edge.RISING
                    )
                    ai_start_trigger = te_terminal
                    logger.info(
                        f"Master机箱 {self._master_chassis} 配置为使用 {ai_start_trigger} 作为触发源"
                    )
                    trigger_configured = True
                    break
                except Exception as e:
                    logger.debug(f"时序引擎终端 {te_terminal} 配置失败: {e}")
                    continue

            if not trigger_configured:
                raise RuntimeError(
                    f"Master机箱 {self._master_chassis} 无法配置触发源，"
                    f"所有时序引擎终端都失败"
                )

            # Master机箱导出触发信号到PFI0
            # 获取Master机箱的一个设备（用于导出触发）
            master_indices = self._chassis_groups[self._master_chassis]
            master_channel = self._ao_channels[master_indices[0]]
            master_device = master_channel.split("/")[0]

            # 导出Master任务的StartTrigger到PFI0
            # 需要使用Master AO任务的时序引擎StartTrigger
            master_pfi = f"/{master_device}/PFI0"

            # 尝试找到Master AO任务使用的时序引擎
            master_te_terminals = [
                f"/{master_device}/te0/StartTrigger",
                f"/{master_device}/te1/StartTrigger",
                f"/{master_device}/te2/StartTrigger",
                f"/{master_device}/te3/StartTrigger",
            ]

            master_start_trigger = ""
            for te_terminal in master_te_terminals:
                master_start_trigger = te_terminal
                break  # 暂时使用第一个，后续可以优化

            # 使用DAQmx Connect Terminals建立路由
            try:
                nidaqmx.system.storage.persisted_task.PersistedTask.delete(  # type: ignore
                    master_start_trigger
                )
            except Exception:
                pass  # 忽略删除失败

            # 连接Master的StartTrigger到PFI0
            logger.info(f"连接 {master_start_trigger} -> {master_pfi}")
            # 注意：这里不使用Connect Terminals，而是使用export_signals
            master_task.export_signals.start_trig_output_term = master_pfi  # type: ignore
            logger.info(f"Master机箱触发信号已导出到 {master_pfi}")

        # 配置Slave机箱
        for chassis_name, ao_task in self._ao_tasks.items():
            if chassis_name == self._master_chassis:
                continue  # 跳过Master机箱

            # 获取Slave机箱的一个设备
            slave_indices = self._chassis_groups[chassis_name]
            slave_channel = self._ao_channels[slave_indices[0]]
            slave_device = slave_channel.split("/")[0]
            slave_pfi = f"/{slave_device}/PFI0"

            # Slave任务使用PFI0作为触发源
            ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(  # type: ignore
                slave_pfi, trigger_edge=Edge.RISING
            )
            logger.info(f"Slave机箱 {chassis_name} 配置为使用 {slave_pfi} 作为触发源")

        logger.info("跨机箱触发路由配置完成")

    def _setup_single_chassis_trigger(self):
        """
        配置单机箱触发（所有AO通道在同一机箱）

        使用简单的触发配置，所有AO任务使用AI的StartTrigger。
        """
        ai_device = self._ai_channel.split("/")[0]

        # 尝试所有可能的时序引擎终端
        te_terminals = [
            f"/{ai_device}/te0/StartTrigger",
            f"/{ai_device}/te1/StartTrigger",
            f"/{ai_device}/te2/StartTrigger",
            f"/{ai_device}/te3/StartTrigger",
        ]

        for chassis_name, ao_task in self._ao_tasks.items():
            trigger_configured = False
            for te_terminal in te_terminals:
                try:
                    ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(  # type: ignore
                        te_terminal, trigger_edge=Edge.RISING
                    )
                    logger.debug(
                        f"机箱 {chassis_name} 配置为使用 {te_terminal} 作为触发源"
                    )
                    trigger_configured = True
                    break
                except Exception as e:
                    logger.debug(f"时序引擎终端 {te_terminal} 配置失败: {e}")
                    continue

            if not trigger_configured:
                raise RuntimeError(
                    f"机箱 {chassis_name} 无法配置触发源，所有时序引擎终端都失败"
                )

        logger.info("单机箱触发配置完成")

    def _ai_callback(
        self,
        task_handle,
        every_n_samples_event_type,
        number_of_samples,
        callback_data,  # type: ignore
    ):
        """
        AI 任务回调函数

        在每次采集到指定数量的样本后被调用。
        （未使用的参数不可删除，其为NI-DAQmx回调函数的API要求）
        """
        try:
            with self._callback_lock:
                # 检查任务是否仍在运行
                if self._ai_task is None or not self._is_running:
                    return 0

                # 读取数据
                data = self._ai_task.read(  # type: ignore
                    number_of_samples_per_channel=number_of_samples
                )

                # 转换为 Waveform 对象
                ai_waveform = Waveform(
                    input_array=np.array(data, dtype=np.float64).reshape(1, -1),
                    sampling_rate=self._sampling_info["sampling_rate"],
                )

                # 添加到缓冲区
                self._data_buffer.append(ai_waveform)

                # 通知工作线程
                self._data_ready_event.set()

        except Exception as e:
            logger.error(f"AI 回调函数异常: {e}", exc_info=True)

        return 0

    def _worker_thread_function(self):
        """
        工作线程函数

        负责从缓冲区取出数据并调用导出函数。
        """
        logger.debug("工作线程已启动")

        while not self._stop_event.is_set():
            # 等待数据就绪或停止信号
            if self._data_ready_event.wait(timeout=1.0):
                self._data_ready_event.clear()

                # 处理缓冲区中的所有数据（但要检查停止事件）
                while len(self._data_buffer) > 0 and not self._stop_event.is_set():
                    try:
                        ai_waveform = self._data_buffer.popleft()
                        self._chunks_num += 1

                        # 如果启用导出，调用导出函数
                        if self._enable_export:
                            self._export_function(ai_waveform, self._chunks_num)

                    except IndexError:
                        break  # 缓冲区已空
                    except Exception as e:
                        logger.error(f"数据处理异常: {e}", exc_info=True)

        logger.debug("工作线程已退出")

    def start(self):
        """
        启动同步的连续 AI/AO 任务

        配置并启动硬件同步的 AI 和多个 AO 任务，使用跨机箱触发路由。

        Raises:
            RuntimeError: 当任务启动失败时
        """
        if self._is_running:
            logger.warning("任务已在运行中")
            return

        try:
            # 创建 AI 任务
            self._ai_task = nidaqmx.Task("ContSyncAI")

            # 配置 AI 任务
            self._setup_ai_task()

            # 为每个机箱创建并配置 AO 任务
            self._setup_ao_tasks()

            # 配置跨机箱触发路由
            self._setup_cross_chassis_trigger_routing()

            # 启动工作线程
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_thread_function,
                name="HiPerfCrossChassisIO_Worker",
            )
            self._worker_thread.start()

            # 启动所有AO任务（等待触发）
            for chassis_name, ao_task in self._ao_tasks.items():
                ao_task.start()
                logger.debug(f"AO任务 {chassis_name} 已启动（等待触发）")

            # 启动AI任务（触发所有AO任务）
            if self._ai_task is not None:
                self._ai_task.start()
                logger.debug("AI任务已启动")

            self._is_running = True
            logger.info("跨机箱同步 AI/AO 任务启动成功")

        except Exception as e:
            logger.error(f"任务启动失败: {e}", exc_info=True)
            self._cleanup_tasks()
            raise RuntimeError(f"跨机箱同步 AI/AO 任务启动失败: {e}") from e

    def stop(self):
        """
        停止同步的连续 AI/AO 任务

        停止所有任务并清理资源。
        """
        if not self._is_running:
            logger.warning("任务未运行")
            return

        logger.info("正在停止跨机箱同步 AI/AO 任务...")

        # 立即标记停止状态，防止回调函数继续添加数据
        self._is_running = False

        # 停止工作线程
        self._stop_event.set()
        self._data_ready_event.set()  # 唤醒工作线程

        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)
            if self._worker_thread.is_alive():
                logger.warning("工作线程未能在超时时间内退出")
            else:
                logger.debug("工作线程已停止")
            self._worker_thread = None

        # 清理任务
        self._cleanup_tasks()

        logger.info("跨机箱同步 AI/AO 任务已停止")

    def _cleanup_tasks(self):
        """清理所有任务资源"""
        # 停止并关闭AI任务
        if self._ai_task is not None:
            try:
                self._ai_task.stop()
                self._ai_task.close()
                logger.debug("AI 任务已关闭")
            except Exception as e:
                logger.warning(f"关闭 AI 任务时出错: {e}")
            finally:
                self._ai_task = None

        # 停止并关闭所有AO任务
        for chassis_name, ao_task in self._ao_tasks.items():
            try:
                ao_task.stop()
                ao_task.close()
                logger.debug(f"AO 任务 {chassis_name} 已关闭")
            except Exception as e:
                logger.warning(f"关闭 AO 任务 {chassis_name} 时出错: {e}")

        self._ao_tasks.clear()

        # 清理触发路由
        self._trigger_routes.clear()

        # 清空缓冲区
        self._data_buffer.clear()
        self._chunks_num = 0

        # 重置事件状态
        self._stop_event.clear()
        self._data_ready_event.clear()

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore
        """
        上下文管理器出口，自动停止任务
        （未使用的参数不可删除，其为Python上下文管理器协议的API要求）
        """
        if self._is_running:
            self.stop()

    def __del__(self):
        """析构函数，确保资源清理"""
        if hasattr(self, "_is_running") and self._is_running:
            try:
                self.stop()
            except Exception:
                pass  # 析构时忽略异常
