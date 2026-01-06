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


class MultiChasCSIO:
    """
    # 高性能跨机箱连续同步 AI/AO 类

    该类专门用于处理跨机箱的多通道AO输出场景。
    该类为每个机箱创建独立的AO任务，
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
        3. 指定AI通道所在机箱为Master机箱
        4. 使用DAQmx Connect Terminals配置触发路由：
           - Master机箱的触发信号通过PFI0导出
           - Slave机箱通过PFI0接收触发信号
        5. 所有任务使用外部10MHz参考时钟（通过机箱背板的"10 MHz REF IN"接口输入，
           自动锁定到各机箱的PXIe_Clk100或PXI_Clk10，实现跨机箱时钟同步）

    ## 硬件连线要求：
        - 触发信号: 所有6个PFI0全部相连（星型拓扑）
          PXI1Slot2/PFI0 ←→ PXI1Slot3/PFI0 ←→ PXI2Slot2/PFI0 ←→
          PXI2Slot3/PFI0 ←→ PXI3Slot2/PFI0 ←→ PXI3Slot3/PFI0
        - 外部参考时钟: 三个机箱的"10 MHz REF IN"接口均接收同步的10MHz时钟信号，
          自动锁定到各机箱的PXIe_Clk100或PXI_Clk10，实现跨机箱时钟同步

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

        # 私有属性 - AO通道状态控制（默认全部禁用）
        self._ao_channels_status = np.zeros(len(ao_channels), dtype=bool)
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
        # 不显式设置ref_clk_src，使用默认值（None）
        # 外部10MHz时钟通过机箱背板的"10 MHz REF IN"接口输入后，
        # 会自动锁定到PXIe_Clk100（对于PXIe设备）或PXI_Clk10（对于PXI设备）
        # 所有机箱的参考时钟都会锁定到外部10MHz参考时钟，实现跨机箱时钟同步
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
            # 不显式设置ref_clk_src，使用默认值（None）
            # 外部10MHz时钟通过机箱背板的"10 MHz REF IN"接口输入后，
            # 会自动锁定到PXIe_Clk100（对于PXIe设备）或PXI_Clk10（对于PXI设备）
            # 所有机箱的参考时钟都会锁定到外部10MHz参考时钟，实现跨机箱时钟同步
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
            # Waveform是ndarray的子类，需要根据波形维度和通道数量正确索引
            if self._output_waveform.ndim == 1:
                # 输出波形是一维的（单通道）
                if len(channel_indices) == 1:
                    # 单通道任务：直接使用整个波形
                    chassis_waveform = np.asarray(self._output_waveform)
                else:
                    # 理论上不应该到这里，因为__init__中已经处理过
                    logger.warning("一维波形但有多个通道索引，这不应该发生")
                    chassis_waveform = np.asarray(self._output_waveform)
            else:
                # 输出波形是二维的（多通道）
                if len(channel_indices) == 1:
                    # 单通道任务：提取一维数组
                    chassis_waveform = np.asarray(
                        self._output_waveform[channel_indices[0], :]
                    )
                else:
                    # 多通道任务：提取二维数组
                    chassis_waveform = np.asarray(
                        self._output_waveform[channel_indices, :]
                    )

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

                        # 如果启用导出，调用导出函数
                        if self._enable_export:
                            # 增加导出计数
                            self._chunks_num += 1
                            self._export_function(ai_waveform, self._chunks_num)
                        else:
                            # 重置导出计数
                            self._chunks_num = 0

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
