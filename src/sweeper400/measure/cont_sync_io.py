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
    Signal,
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

    该类专门用于处理跨机箱的多通道AI/AO场景。
    该类为每个机箱创建独立的AI和AO任务，
    并使用StartTrigger导出和接收机制实现跨机箱同步触发。

    ## 主要特性：
        - 支持跨机箱的多通道AI输入和AO输出
        - 为每个机箱创建独立的AI和AO任务
        - 使用StartTrigger导出/接收机制实现所有任务的同步触发
        - 硬件同步的连续 AI/AO 任务
        - AI 通道使用麦克风模式
        - 支持两种AO模式：
          * Static AO：使用再生模式，循环播放固定波形
          * Feedback AO：使用非再生模式，根据AI数据实时生成输出
        - 支持运行时动态更换静态输出波形
        - 基于回调函数的数据处理和反馈控制
        - 线程安全的数据传输控制

    ## 跨机箱同步机制：
        1. 识别AI和AO通道所属的机箱
        2. 为每个机箱创建独立的AI、Static AO和Feedback AO任务
        3. 按优先级选择Master任务
        （优先级：PXI1Slot2>PXI1Slot3>PXI2Slot2>PXI2Slot3>PXI3Slot2>PXI3Slot3）
        4. Master任务导出StartTrigger信号到其PFI0接口
        5. 其他Slave任务从各自机箱的PFI0接口接收触发信号
        6. 所有任务使用外部10MHz参考时钟（通过机箱背板的"10 MHz REF IN"接口输入，
           自动锁定到各机箱的PXIe_Clk100或PXI_Clk10，实现跨机箱时钟同步）

    ## 硬件连线要求：
        - 触发信号: 所有板卡的PFI0接口通过线缆相连，用于传递StartTrigger信号
        - 外部参考时钟: 三个机箱的"10 MHz REF IN"接口均接收同步的10MHz时钟信号，
          自动锁定到各机箱的PXIe_Clk100或PXI_Clk10，实现跨机箱时钟同步

    ## 使用示例：
    ```python
    from sweeper400.analyze import init_sampling_info, init_sine_args, get_sine_cycles
    from sweeper400.measure.cont_sync_io import MultiChasCSIO
    import numpy as np

    # 创建采样信息和静态输出波形
    sampling_info = init_sampling_info(48000, 4800)
    sine_args = init_sine_args(1000.0, 0.02, 0.0)
    static_output_waveform = get_sine_cycles(sampling_info, sine_args)

    # 定义反馈函数（根据AI数据生成AO输出）
    def feedback_function(ai_waveform):
        # 示例：将AI数据取反并缩放作为AO输出
        feedback_data = -0.5 * ai_waveform.view(np.ndarray)
        return Waveform(feedback_data, sampling_rate=ai_waveform.sampling_rate)

    # 定义数据导出函数
    def export_data(ai_waveform, ao_static_waveform, ao_feedback_waveform, chunks_num):
        print(f"导出第 {chunks_num} 段数据")
        print(f"AI波形shape: {ai_waveform.shape}")
        print(f"Static AO波形shape: {ao_static_waveform.shape}")
        if ao_feedback_waveform is not None:
            print(f"Feedback AO波形shape: {ao_feedback_waveform.shape}")

    # 跨机箱多通道示例
    sync_io = MultiChasCSIO(
        ai_channels=(
            "PXI1Slot2/ai0",  # PXIChassis1
            "PXI2Slot2/ai0",  # PXIChassis2
        ),
        ao_channels_static=(
            "PXI1Slot2/ao0",  # PXIChassis1 - 静态输出
        ),
        ao_channels_feedback=(
            "PXI2Slot2/ao0",  # PXIChassis2 - 反馈输出
            "PXI3Slot2/ao0",  # PXIChassis3 - 反馈输出
        ),
        static_output_waveform=static_output_waveform,
        feedback_function=feedback_function,
        export_function=export_data,
        # 可选：自定义缓冲区配置（如遇到缓冲区溢出警告时）
        # buffer_size_multiplier=5,  # AI和Feedback AO缓冲区大小倍数，默认5
    )

    # 启动任务（Master任务导出StartTrigger，Slave任务接收）
    sync_io.start()
    sync_io.enable_export = True

    # 动态更换静态输出波形
    new_waveform = get_sine_cycles(sampling_info, init_sine_args(2000.0, 0.02, 0.0))
    sync_io.update_static_output_waveform(new_waveform)

    # 停止任务
    sync_io.stop()
    ```

    ## 注意事项：
        - 所有板卡的PFI0接口必须通过线缆相连
        - 所有机箱必须接收同步的外部10MHz参考时钟
        - 确保PXI Platform Services已正确安装和配置
        - Master任务按优先级自动选择
            （PXI1Slot2>PXI1Slot3>PXI2Slot2>PXI2Slot3>PXI3Slot2>PXI3Slot3）
        - feedback_function必须返回与feedback_ao_channels数量匹配的Waveform对象
        - Static AO和Feedback AO通道至少要有一个非空
    """

    def __init__(
        self,
        ai_channels: tuple[str, ...],
        ao_channels_static: tuple[str, ...],
        ao_channels_feedback: tuple[str, ...],
        static_output_waveform: Waveform,
        feedback_function: Callable[[Waveform], Waveform],
        export_function: Callable[[Waveform, Waveform, Waveform, PositiveInt], Any],
        buffer_size_multiplier: PositiveInt = 5,
    ):
        """
        初始化跨机箱连续同步 AI/AO 任务

        Args:
            ai_channels: AI 通道名称元组（例如 ("PXI1Slot2/ai0", "PXI2Slot2/ai0")）
            ao_channels_static: 静态AO通道名称元组，使用再生模式
                （例如 ("PXI1Slot2/ao0",)）
            ao_channels_feedback: 反馈AO通道名称元组，使用非再生模式
                （例如 ("PXI2Slot2/ao0",)）
            static_output_waveform: 静态输出波形对象（Waveform类型），
                用于static_ao_channels
            feedback_function: 反馈函数，接收AI数据（多通道Waveform），
                返回feedback AO数据（多通道Waveform）
            export_function: 数据导出函数，接收 (ai_waveform, ao_static_waveform,
                ao_feedback_waveform, chunks_num) 参数。
                其中ao_static_waveform是当前的静态输出波形，
                ao_feedback_waveform是当前的反馈输出波形（如果没有反馈通道则为None）
            buffer_size_multiplier: AI和Feedback AO缓冲区大小倍数
                （相对于static_output_waveform.samples_num），
                默认为5。增大此值可减少缓冲区溢出风险，但会增加内存占用。

        Raises:
            ValueError: 当参数无效时
        """
        # 验证参数
        if not ai_channels:
            raise ValueError("AI 通道列表不能为空")
        if not ao_channels_static and not ao_channels_feedback:
            raise ValueError("Static AO 通道和 Feedback AO 通道不能同时为空")
        if static_output_waveform.samples_num == 0:
            raise ValueError("静态输出波形不能为空")

        # 公共属性 - 函数
        self._export_function = export_function
        self._feedback_function = feedback_function

        # 缓冲区配置参数
        self._buffer_size_multiplier = buffer_size_multiplier

        # 采样信息
        self._sampling_info = static_output_waveform.sampling_info

        # 私有属性 - 通道信息（统一的数据结构）
        # _channels_info: 包含 "ai", "ao_static", "ao_feedback" 三个键
        # 每个键的值是一个字典：机箱名 -> 该机箱的物理通道字符串列表
        self._channels_info: dict[str, dict[str, list[str]]] = {
            "ai": {},
            "ao_static": {},
            "ao_feedback": {},
        }

        # 私有属性 - 任务管理（统一的数据结构）
        # _tasks: 包含 "ai", "ao_static", "ao_feedback" 三个键
        # 每个键的值是一个字典：机箱名 -> nidaqmx.Task
        self._tasks: dict[str, dict[str, nidaqmx.Task]] = {
            "ai": {},
            "ao_static": {},
            "ao_feedback": {},
        }

        # 私有属性 - 状态管理
        self._is_running = False
        self._callback_lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None
        self._data_ready_event = threading.Event()
        self._stop_event = threading.Event()

        # 私有属性 - 数据缓冲和控制
        self._ai_queue: deque[dict[str, Any]] = deque()  # 存储AI数据包
        self._chunks_num = 0
        self._enable_export = False
        self._ai_tasks_num = 0  # AI任务总数，用于工作线程判断何时处理数据

        # 将输入通道信息预处理为统一的内置数据结构
        # 分组AI通道
        self._init_group_channels_by_type(ai_channels, self._channels_info["ai"], "AI")

        # 分组Static AO通道
        self._init_group_channels_by_type(
            ao_channels_static, self._channels_info["ao_static"], "Static AO"
        )

        # 分组Feedback AO通道
        self._init_group_channels_by_type(
            ao_channels_feedback, self._channels_info["ao_feedback"], "Feedback AO"
        )

        # 处理静态输出波形（在_channels_info初始化完成之后）
        if ao_channels_static:
            self._static_output_waveform = self._process_waveform_for_channels(
                static_output_waveform
            )
        else:
            self._static_output_waveform = static_output_waveform

        # 输出初始化完成日志
        logger.info(
            f"MultiChasCSIO 实例已创建 - "
            f"AI通道数: {len(ai_channels)}, "
            f"AI: {ai_channels}, "
            f"Static AO通道数: {len(ao_channels_static)}, "
            f"Static AO: {ao_channels_static}, "
            f"Feedback AO通道数: {len(ao_channels_feedback)}, "
            f"Feedback AO: {ao_channels_feedback}, "
            f"静态输出波形shape: {self._static_output_waveform.shape}, "
            f"采样率: {self._sampling_info['sampling_rate']} Hz, "
            f"缓冲区倍数: {buffer_size_multiplier}"
        )

    @staticmethod
    def _init_group_channels_by_type(
        channels: tuple[str, ...],
        chassis_groups: dict[str, list[str]],
        channel_type: str,
    ):
        """
        将指定类型的通道按机箱分组（静态方法）

        Args:
            channels: 通道名称元组
            chassis_groups: 机箱分组字典（将被修改），键为机箱名，值为物理通道字符串列表
            channel_type: 通道类型名称（用于日志）
        """
        for channel in channels:
            # 从通道名称中提取机箱名称
            device = channel.split("/")[0]
            chassis_name = None

            # 提取机箱编号
            if device.startswith("PXI") and "Slot" in device:
                chassis_num = device[3]  # 提取 "PXI1Slot2" 中的 "1"
                chassis_name = f"PXIChassis{chassis_num}"

            if chassis_name:
                if chassis_name not in chassis_groups:
                    chassis_groups[chassis_name] = []
                chassis_groups[chassis_name].append(channel)  # 存储物理通道字符串
            else:
                logger.warning(
                    f"无法识别{channel_type}通道 {channel} 的机箱信息，将被忽略"
                )

        logger.info(
            f"{channel_type}通道分组完成，共 {len(chassis_groups)} 个机箱: "
            f"{list(chassis_groups.keys())}"
        )

    def _process_waveform_for_channels(
        self,
        waveform: Waveform,
    ) -> Waveform:
        """
        处理波形以匹配通道要求（单通道扩展/补充通道信息/检查匹配）

        从self._channels_info["ao_static"]自动获取目标通道名称。

        Args:
            waveform: 输入波形对象

        Returns:
            处理后的波形对象

        Raises:
            ValueError: 当波形通道数与目标通道数不匹配时
        """
        # 从self._channels_info["ao_static"]获取所有静态AO通道列表
        all_ao_static_channels = []
        for channels in self._channels_info["ao_static"].values():
            all_ao_static_channels.extend(channels)

        channel_names = tuple(all_ao_static_channels)
        channels_num = len(channel_names)

        # 如果输出波形是单通道，需要扩展为多通道
        if waveform.channels_num == 1 and channels_num > 1:
            logger.info(f"静态输出波形为单通道，将扩展为 {channels_num} 通道")
            # 创建多通道波形：将单通道波形复制到所有通道
            expanded_data = np.tile(waveform, (channels_num, 1))

            return Waveform(
                expanded_data,
                sampling_rate=waveform.sampling_rate,
                channel_names=channel_names,  # 添加通道名称元数据
                timestamp=waveform.timestamp,
                id=waveform.id,
                sine_args=waveform.sine_args,
            )
        else:
            # 如果波形已经是多通道，检查是否需要添加channel_names
            if waveform.channels_num == channels_num and waveform.channel_names is None:
                # 添加通道名称元数据
                return Waveform(
                    waveform,
                    sampling_rate=waveform.sampling_rate,
                    channel_names=channel_names,
                    timestamp=waveform.timestamp,
                    id=waveform.id,
                    sine_args=waveform.sine_args,
                )
            elif waveform.channels_num == channels_num:
                # 通道数匹配且已有channel_names，直接返回
                return waveform
            else:
                # 通道数不匹配，抛出错误
                raise ValueError(
                    f"静态输出波形通道数 ({waveform.channels_num}) "
                    f"与目标通道数 ({channels_num}) 不匹配"
                )

    @property
    def ai_channels_num(self) -> int:
        """AI 通道数量"""
        return sum(len(channels) for channels in self._channels_info["ai"].values())

    @property
    def ao_channels_num_static(self) -> int:
        """Static AO 通道数量"""
        return sum(
            len(channels) for channels in self._channels_info["ao_static"].values()
        )

    @property
    def ao_channels_num_feedback(self) -> int:
        """Feedback AO 通道数量"""
        return sum(
            len(channels) for channels in self._channels_info["ao_feedback"].values()
        )

    @property
    def enable_export(self) -> bool:
        """数据导出启用状态"""
        return self._enable_export

    @enable_export.setter
    def enable_export(self, value: bool):
        """设置数据导出启用状态"""
        self._enable_export = value
        if value:
            logger.info("数据导出已启用")
        else:
            logger.info("数据导出已禁用")

    def update_static_output_waveform(self, new_waveform: Waveform):
        """
        更新静态输出波形

        在任务运行中动态更换static_output_waveform。
        该方法仅会更新所有Static AO任务的波形数据。

        Args:
            new_waveform: 新的静态输出波形对象（Waveform类型）

        Raises:
            ValueError: 当波形参数无效时
            RuntimeError: 当任务未运行时
        """
        if not self._is_running:
            raise RuntimeError("任务未运行，无法更新输出波形")

        if new_waveform.samples_num == 0:
            raise ValueError("输出波形不能为空")

        # 处理输出波形（单通道扩展/补充通道信息/检查匹配）
        processed_waveform = self._process_waveform_for_channels(new_waveform)

        # 验证采样率是否匹配
        if processed_waveform.sampling_rate != self._sampling_info["sampling_rate"]:
            logger.warning(
                f"新波形采样率 ({processed_waveform.sampling_rate} Hz) "
                f"与原采样率 ({self._sampling_info['sampling_rate']} Hz) 不匹配"
            )

        # 更新内部波形
        self._static_output_waveform = processed_waveform

        # 更新所有Static AO任务的波形数据
        for chassis_name in self._tasks["ao_static"].keys():
            try:
                self._write_ao_task_waveform_static(chassis_name)
                logger.info(f"机箱 {chassis_name} 的静态输出波形已更新")
            except Exception as e:
                logger.error(
                    f"更新机箱 {chassis_name} 静态波形失败: {e}",
                    exc_info=True,
                )
                raise

        logger.info(
            f"静态输出波形已更新 - shape: {self._static_output_waveform.shape}, "
            f"采样率: {self._static_output_waveform.sampling_rate} Hz"
        )

    def _setup_ai_tasks(self):
        """
        为每个机箱创建并配置独立的AI任务

        配置为麦克风模式，使用 IEPE 激励电流。
        """
        logger.info("开始为每个机箱创建独立的AI任务")

        # 记录AI任务总数
        self._ai_tasks_num = len(self._channels_info["ai"])

        for chassis_name, channel_names in self._channels_info["ai"].items():
            # 创建任务
            task_name = f"ContSyncAI_{chassis_name}"
            ai_task = nidaqmx.Task(task_name)
            self._tasks["ai"][chassis_name] = ai_task

            logger.debug(f"为机箱 {chassis_name} 创建AI任务: {task_name}")

            # 添加该机箱的所有AI通道（麦克风模式）
            for channel_name in channel_names:
                ai_task.ai_channels.add_ai_microphone_chan(  # type: ignore
                    channel_name,
                    units=SoundPressureUnits.PA,
                    mic_sensitivity=0.004,
                    max_snd_press_level=120.0,
                    current_excit_source=ExcitationSource.INTERNAL,
                    current_excit_val=0.004,
                )
                logger.debug(f"  添加通道: {channel_name}")

            # 配置时钟源和采样
            # 不显式设置ref_clk_src，使用默认值（None）
            # 外部10MHz时钟通过机箱背板的"10 MHz REF IN"接口输入后，
            # 会自动锁定到PXIe_Clk100（对于PXIe设备）或PXI_Clk10（对于PXI设备）
            # 所有机箱的参考时钟都会锁定到外部10MHz参考时钟，实现跨机箱时钟同步
            ai_task.timing.cfg_samp_clk_timing(  # type: ignore
                rate=self._sampling_info["sampling_rate"],
                sample_mode=AcquisitionType.CONTINUOUS,
            )

            # 配置缓冲区大小（使用可配置的倍数）
            buffer_size = (
                self._static_output_waveform.samples_num * self._buffer_size_multiplier
            )
            ai_task.in_stream.input_buf_size = buffer_size
            logger.debug(f"  设置AI缓冲区大小: {buffer_size} 样本")

            # 注册回调函数（使用闭包捕获chassis_name）
            callback_samples = self._static_output_waveform.samples_num

            def make_callback(chassis_name_captured: str):
                """创建回调函数的工厂函数"""

                def callback(task_handle, event_type, num_samples, callback_data):  # type: ignore
                    return self._ai_callback(
                        chassis_name_captured,
                        task_handle,
                        event_type,
                        num_samples,
                        callback_data,
                    )

                return callback

            ai_task.register_every_n_samples_acquired_into_buffer_event(  # type: ignore
                callback_samples, make_callback(chassis_name)
            )

        logger.info(f"成功创建 {len(self._tasks['ai'])} 个AI任务")

    def _setup_ao_tasks(self, task_type: str):
        """
        为每个机箱创建并配置独立的AO任务（通用方法）

        Args:
            task_type: 任务类型键名（"ao_static"或"ao_feedback"）
        """
        # 从 _channels_info 和 _tasks 中获取相应的数据结构
        chassis_groups = self._channels_info[task_type]
        tasks_dict = self._tasks[task_type]

        if not chassis_groups:
            logger.info(f"没有{task_type} AO通道，跳过{task_type} AO任务创建")
            return

        # 根据 task_type 确定配置参数
        if task_type == "ao_static":
            regeneration_mode = RegenerationMode.ALLOW_REGENERATION
            buffer_size_multiplier = 1
            task_name_prefix = "ContSyncStaticAO"
        elif task_type == "ao_feedback":
            regeneration_mode = RegenerationMode.DONT_ALLOW_REGENERATION
            buffer_size_multiplier = self._buffer_size_multiplier
            task_name_prefix = "ContSyncFeedbackAO"
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")

        logger.info(f"开始为每个机箱创建独立的{task_type} AO任务")

        for chassis_name, channel_names in chassis_groups.items():
            # 创建任务
            task_name = f"{task_name_prefix}_{chassis_name}"
            ao_task = nidaqmx.Task(task_name)
            tasks_dict[chassis_name] = ao_task

            logger.debug(f"为机箱 {chassis_name} 创建{task_type} AO任务: {task_name}")

            # 添加该机箱的所有AO通道
            for channel_name in channel_names:
                ao_task.ao_channels.add_ao_voltage_chan(  # type: ignore
                    channel_name, min_val=-10.0, max_val=10.0
                )
                logger.debug(f"  添加通道: {channel_name}")

            # 配置时钟源和采样
            ao_task.timing.cfg_samp_clk_timing(  # type: ignore
                rate=self._sampling_info["sampling_rate"],
                sample_mode=AcquisitionType.CONTINUOUS,
            )

            # 设置再生模式
            ao_task.out_stream.regen_mode = regeneration_mode

            # 设置缓冲区大小
            buffer_size = (
                self._static_output_waveform.samples_num * buffer_size_multiplier
            )
            ao_task.out_stream.output_buf_size = buffer_size
            logger.debug(f"  设置{task_type} AO缓冲区大小: {buffer_size} 样本")

            # 预写入波形数据
            if task_type == "ao_static":
                # Static AO: 写入静态输出波形
                self._write_ao_task_waveform_static(chassis_name)
            elif task_type == "ao_feedback":
                # Feedback AO: 写入全0静音波形
                self._write_ao_task_waveform_feedback_silence(chassis_name)

        logger.info(f"成功创建 {len(tasks_dict)} 个{task_type} AO任务")

    def _write_ao_task_waveform_static(self, chassis_name: str):
        """
        向指定的Static AO任务写入波形数据

        Args:
            chassis_name: 机箱名称
        """
        try:
            # 从内部数据结构获取任务和通道信息
            ao_task = self._tasks["ao_static"][chassis_name]
            channel_names = self._channels_info["ao_static"][chassis_name]

            # 获取所有 static AO 通道的完整列表（从 channel_names 元数据）
            all_channel_names = self._static_output_waveform.channel_names
            if all_channel_names is None:
                raise ValueError("静态输出波形缺少 channel_names 元数据")

            # 找到当前机箱通道在完整列表中的索引
            channel_indices = [all_channel_names.index(ch) for ch in channel_names]

            # 提取该机箱的通道波形
            # Waveform是ndarray的子类，需要根据波形维度和通道数量正确索引
            if self._static_output_waveform.ndim == 1:
                # 输出波形是一维的（单通道）
                if len(channel_indices) == 1:
                    # 单通道任务：直接使用整个波形
                    chassis_waveform = np.asarray(self._static_output_waveform)
                else:
                    # 理论上不应该到这里，因为__init__中已经处理过
                    logger.warning("一维波形但有多个通道索引，这不应该发生")
                    chassis_waveform = np.asarray(self._static_output_waveform)
            else:
                # 输出波形是二维的（多通道）
                if len(channel_indices) == 1:
                    # 单通道任务：提取一维数组
                    chassis_waveform = np.asarray(
                        self._static_output_waveform[channel_indices[0], :]
                    )
                else:
                    # 多通道任务：提取二维数组
                    chassis_waveform = np.asarray(
                        self._static_output_waveform[channel_indices, :]
                    )

            # 写入数据
            ao_task.write(chassis_waveform, auto_start=False)  # type: ignore
            logger.debug(
                f"成功写入机箱 {chassis_name} 的静态波形数据，"
                f"shape: {chassis_waveform.shape}"
            )

        except Exception as e:
            logger.error(f"机箱 {chassis_name} 静态波形写入失败: {e}", exc_info=True)
            raise

    def _write_ao_task_waveform_feedback_silence(self, chassis_name: str):
        """
        向指定的Feedback AO任务写入全0静音波形

        注意：此方法一次性写入2个chunk长度的静音波形。
        这是因为所有AI和AO任务同步开始，当AO输出一个chunk时，
        AI任务才刚刚返回第一段采集到的chunk，而此时反馈AO任务的缓冲区已经空了。
        我们必须先为反馈AO任务写入2个chunk，这样我们才有足够的时间处理反馈逻辑，
        并写入反馈AO缓冲区。

        Args:
            chassis_name: 机箱名称
        """
        try:
            # 从内部数据结构获取任务和通道信息
            ao_task = self._tasks["ao_feedback"][chassis_name]
            channel_names = self._channels_info["ao_feedback"][chassis_name]

            # 创建全0静音波形（2个chunk长度）
            samples_num = self._static_output_waveform.samples_num * 2  # 2个chunk
            channels_num = len(channel_names)

            if channels_num == 1:
                # 单通道：一维数组
                silence_waveform = np.zeros(samples_num, dtype=np.float64)
            else:
                # 多通道：二维数组
                silence_waveform = np.zeros(
                    (channels_num, samples_num), dtype=np.float64
                )

            # 写入数据
            ao_task.write(silence_waveform, auto_start=False)  # type: ignore
            logger.debug(
                f"成功写入机箱 {chassis_name} 的静音波形数据（2个chunk），"
                f"shape: {silence_waveform.shape}"
            )

        except Exception as e:
            logger.error(f"机箱 {chassis_name} 静音波形写入失败: {e}", exc_info=True)
            raise

    def _write_ao_task_waveform_feedback(
        self, chassis_name: str, feedback_waveform: Waveform
    ):
        """
        向指定的Feedback AO任务写入波形数据

        Args:
            chassis_name: 机箱名称
            feedback_waveform: 反馈波形数据（包含所有feedback通道）
        """
        try:
            # 从内部数据结构获取任务和通道信息
            ao_task = self._tasks["ao_feedback"][chassis_name]
            channel_names = self._channels_info["ao_feedback"][chassis_name]

            # 获取所有 feedback AO 通道的完整列表（从 channel_names 元数据）
            all_channel_names = feedback_waveform.channel_names
            if all_channel_names is None:
                raise ValueError("反馈波形缺少 channel_names 元数据")

            # 找到当前机箱通道在完整列表中的索引
            channel_indices = [all_channel_names.index(ch) for ch in channel_names]

            # 提取该机箱的通道波形
            if feedback_waveform.ndim == 1:
                # 输出波形是一维的（单通道）
                if len(channel_indices) == 1:
                    # 单通道任务：直接使用整个波形
                    chassis_waveform = np.asarray(feedback_waveform)
                else:
                    logger.warning("一维波形但有多个通道索引，这不应该发生")
                    chassis_waveform = np.asarray(feedback_waveform)
            else:
                # 输出波形是二维的（多通道）
                if len(channel_indices) == 1:
                    # 单通道任务：提取一维数组
                    chassis_waveform = np.asarray(
                        feedback_waveform[channel_indices[0], :]
                    )
                else:
                    # 多通道任务：提取二维数组
                    chassis_waveform = np.asarray(feedback_waveform[channel_indices, :])

            # 写入数据
            ao_task.write(chassis_waveform, auto_start=False)  # type: ignore
            logger.debug(
                f"成功写入机箱 {chassis_name} 的反馈波形数据，"
                f"shape: {chassis_waveform.shape}"
            )

        except Exception as e:
            logger.error(f"机箱 {chassis_name} 反馈波形写入失败: {e}", exc_info=True)
            raise

    def _collect_devices_from_tasks(self) -> set[str]:
        """
        从所有任务中收集设备信息，用于确认哪些机箱被真正使用

        Returns:
            all_devices: 所有设备集合
        """
        all_devices: set[str] = set()

        # 遍历所有三类任务
        for task_type in ["ai", "ao_static", "ao_feedback"]:
            chassis_groups = self._channels_info[task_type]

            for chassis_name in chassis_groups.keys():
                # 检查该机箱的通道，提取设备名
                channel_names = chassis_groups[chassis_name]
                if channel_names:
                    # 从第一个通道名提取设备名
                    channel_name = channel_names[0]
                    device = channel_name.split("/")[
                        0
                    ]  # 例如 "PXI1Slot2/ai0" -> "PXI1Slot2"
                    all_devices.add(device)

        return all_devices

    @staticmethod
    def _configure_slave_trigger_with_pfi(
        task: nidaqmx.Task, chassis_name: str, task_name: str
    ):
        """
        配置Slave任务的触发源（自动获取PFI终端）

        Args:
            task: 任务对象
            chassis_name: 机箱名称（例如 "PXIChassis1"）
            task_name: 任务名称（用于日志）
        """
        # 获取指定机箱的PFI0终端名称
        chassis_num = chassis_name[-1]  # 提取 "PXIChassis1" 中的 "1"
        pfi_device = f"PXI{chassis_num}Slot2"
        pfi_terminal = f"/{pfi_device}/PFI0"

        # 配置触发源
        task.triggers.start_trigger.cfg_dig_edge_start_trig(  # type: ignore
            pfi_terminal, trigger_edge=Edge.RISING
        )
        logger.info(f"Slave任务 {task_name} 配置为从 {pfi_terminal} 接收触发")

    def _setup_start_trigger_sync(self):
        """
        配置StartTrigger导出和接收同步

        使用Master/Slave模式实现跨机箱同步触发。
        Master任务导出StartTrigger到PFI0，Slave任务从PFI0接收触发信号。

        Master任务选择优先级（按板卡名称）：
        PXI1Slot2 > PXI1Slot3 > PXI2Slot2 > PXI2Slot3 > PXI3Slot2 > PXI3Slot3

        策略：
        1. 收集所有任务涉及的板卡设备
        2. 按优先级排序，选择第一个作为Master设备
        3. 找到Master设备对应的任务（AI或AO）
        4. Master任务导出StartTrigger到其PFI0接口
        5. 其他所有任务配置为从各自机箱的PFI0接收触发信号
        """
        logger.info("开始配置StartTrigger导出和接收同步")

        # 定义板卡优先级（从高到低）
        device_priority = [
            "PXI1Slot2",
            "PXI1Slot3",
            "PXI2Slot2",
            "PXI2Slot3",
            "PXI3Slot2",
            "PXI3Slot3",
        ]

        # 收集所有任务涉及的设备
        all_devices = self._collect_devices_from_tasks()

        logger.debug(f"收集到的所有设备: {sorted(all_devices)}")

        # 按优先级选择Master设备
        master_device = None
        for device in device_priority:
            if device in all_devices:
                master_device = device
                break

        if master_device is None:
            raise RuntimeError("无法找到合适的Master设备")

        logger.info(f"选择 {master_device} 作为Master设备")

        # 确定Master任务（优先选择AI任务，如果没有则选择Static AO或Feedback AO任务）
        # 从master_device提取机箱名称
        chassis_num = master_device[3]  # 提取 "PXI1Slot2" 中的 "1"
        master_chassis_name = f"PXIChassis{chassis_num}"

        master_task = None
        master_task_type = None

        # 优先选择AI任务
        if master_chassis_name in self._tasks["ai"]:
            master_task = self._tasks["ai"][master_chassis_name]
            master_task_type = "AI"
        # 其次选择Static AO任务
        elif master_chassis_name in self._tasks["ao_static"]:
            master_task = self._tasks["ao_static"][master_chassis_name]
            master_task_type = "Static AO"
        # 最后选择Feedback AO任务
        elif master_chassis_name in self._tasks["ao_feedback"]:
            master_task = self._tasks["ao_feedback"][master_chassis_name]
            master_task_type = "Feedback AO"

        if master_task is None:
            raise RuntimeError(f"Master设备 {master_device} 没有对应的任务")

        # Master任务导出StartTrigger到PFI0
        master_pfi_terminal = f"/{master_device}/PFI0"
        master_task.export_signals.export_signal(  # type: ignore
            Signal.START_TRIGGER, master_pfi_terminal
        )
        logger.info(
            f"Master任务 ({master_task_type}, {master_device}) "
            f"导出StartTrigger到 {master_pfi_terminal}"
        )

        # 配置所有Slave任务从各自机箱的PFI0接收触发信号
        # 配置所有AI任务的触发（除了Master AI任务）
        for chassis_name, ai_task in self._tasks["ai"].items():
            if ai_task is master_task:
                continue  # 跳过Master任务
            self._configure_slave_trigger_with_pfi(
                ai_task, chassis_name, f"AI {chassis_name}"
            )

        # 配置所有Static AO任务的触发（除了Master任务）
        for chassis_name, ao_task in self._tasks["ao_static"].items():
            if ao_task is master_task:
                continue  # 跳过Master任务
            self._configure_slave_trigger_with_pfi(
                ao_task, chassis_name, f"Static AO {chassis_name}"
            )

        # 配置所有Feedback AO任务的触发（除了Master任务）
        for chassis_name, ao_task in self._tasks["ao_feedback"].items():
            if ao_task is master_task:
                continue  # 跳过Master任务
            self._configure_slave_trigger_with_pfi(
                ao_task, chassis_name, f"Feedback AO {chassis_name}"
            )

        logger.info("StartTrigger导出和接收同步配置完成")

    def _ai_callback(
        self,
        chassis_name: str,
        task_handle,
        every_n_samples_event_type,
        number_of_samples,
        callback_data,  # type: ignore
    ):
        """
        AI 任务回调函数

        在每次采集到指定数量的样本后被调用。
        仅读取当前任务的数据并加入队列，不处理其他任务。
        保持逻辑简单和迅速，避免阻塞后台nidaqmx线程。
        （未使用的参数不可删除，其为NI-DAQmx回调函数的API要求）

        Args:
            chassis_name: 机箱名称（通过闭包传入）
            task_handle: 任务句柄
            every_n_samples_event_type: 事件类型
            number_of_samples: 样本数量
            callback_data: 回调数据
        """
        try:
            # 检查任务是否仍在运行
            if not self._is_running:
                return 0

            # 获取当前任务
            current_task = self._tasks["ai"].get(chassis_name)
            if current_task is None:
                logger.warning(f"无法找到机箱 {chassis_name} 的AI任务")
                return 0

            # 读取当前任务的数据
            try:
                data = current_task.read(  # type: ignore
                    number_of_samples_per_channel=number_of_samples
                )

                # 准备数据包
                ai_package = {
                    "chassis_name": chassis_name,
                    "data": data,
                    "samples_num": number_of_samples,
                }

                # 加入队列
                with self._callback_lock:
                    self._ai_queue.append(ai_package)

                # 通知工作线程
                self._data_ready_event.set()

            except Exception as e:
                logger.error(
                    f"从机箱 {chassis_name} 读取AI数据失败: {e}",
                    exc_info=True,
                )

        except Exception as e:
            logger.error(f"AI 回调函数异常: {e}", exc_info=True)

        return 0

    def _worker_thread_function(self):
        """
        工作线程函数

        负责从缓冲区取出数据并调用导出函数，同时处理feedback数据并写入Feedback AO任务。
        等待所有AI任务的数据包到齐后，才进行一次集中处理。

        触发机制：
        - 使用"剩余等待event次数"变量控制执行
        - 每当收到ai_tasks_num次event，或达到超时时间，尝试执行一次
        - 如果数据不足，输出警告并等待下次触发
        """
        logger.debug("工作线程已启动")

        # 计算超时时间：波形持续时间
        timeout = self._static_output_waveform.duration

        # 剩余等待event次数（用于控制执行触发）
        remaining_events = self._ai_tasks_num

        while not self._stop_event.is_set():
            # 等待数据就绪事件，超时时间为一个波形周期
            event_triggered = self._data_ready_event.wait(timeout=timeout)

            # 清除事件标志
            self._data_ready_event.clear()

            # 判断是否应该尝试执行
            should_execute = False

            if event_triggered:
                # 事件被触发，减少剩余等待次数
                remaining_events -= 1
                if remaining_events <= 0:
                    # 已收到足够的event，应该执行
                    should_execute = True
            else:
                # 超时，无论如何都尝试执行
                should_execute = True
                logger.debug("超时触发，尝试执行数据处理")

            if not should_execute:
                # 还需要等待更多event
                continue

            # 尝试收集所有AI任务的数据包
            collected_packages: list[dict[str, Any]] = []

            with self._callback_lock:
                # 一次性取出所有可用的数据包
                while len(self._ai_queue) > 0:
                    package = self._ai_queue.popleft()
                    collected_packages.append(package)

            # 检查是否收集到足够的数据包
            if len(collected_packages) < self._ai_tasks_num:
                logger.warning(
                    f"数据不足，未能收集到足够的AI数据包 "
                    f"(期望 {self._ai_tasks_num}，实际 {len(collected_packages)})，"
                    f"等待下次触发"
                )
                # 数据不足，保持remaining_events为0，等待下次event或超时触发
                remaining_events = 0
                continue

            # 数据充足，重置剩余等待次数
            remaining_events = self._ai_tasks_num

            # 处理收集到的数据包
            try:
                # 获取所有 AI 通道列表（按机箱顺序）
                all_ai_channels = []
                for channels in self._channels_info["ai"].values():
                    all_ai_channels.extend(channels)

                # 按机箱名称排序，确保通道顺序与 _channels_info["ai"] 一致
                collected_packages.sort(
                    key=lambda pkg: list(self._channels_info["ai"].keys()).index(
                        pkg["chassis_name"]
                    )
                )

                # 合并所有数据
                all_data = []
                for package in collected_packages:
                    data = package["data"]
                    # 将数据转换为numpy数组并确保是2D格式
                    data_array = np.array(data, dtype=np.float64)
                    if data_array.ndim == 1:
                        # 单通道数据，reshape为(1, samples)
                        data_array = data_array.reshape(1, -1)
                    all_data.append(data_array)

                # 沿通道轴（axis=0）合并
                combined_data = np.vstack(all_data)

                # 转换为 Waveform 对象，添加通道名称元数据
                ai_waveform = Waveform(
                    input_array=combined_data,
                    sampling_rate=self._sampling_info["sampling_rate"],
                    channel_names=tuple(all_ai_channels),
                )

                # 处理反馈
                feedback_waveform = None
                if self._channels_info["ao_feedback"]:
                    try:
                        feedback_waveform = self._feedback_function(ai_waveform)
                        # 验证反馈波形通道数
                        if (
                            feedback_waveform.channels_num
                            != self.ao_channels_num_feedback
                        ):
                            logger.error(
                                f"反馈函数返回的波形通道数 ({feedback_waveform.channels_num}) "
                                f"与 Feedback AO 通道数 ({self.ao_channels_num_feedback}) 不匹配"
                            )
                            feedback_waveform = None  # 重置为None，表示反馈失败
                        else:
                            # 写入所有Feedback AO任务
                            for chassis_name in self._tasks["ao_feedback"].keys():
                                try:
                                    self._write_ao_task_waveform_feedback(
                                        chassis_name, feedback_waveform
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"写入机箱 {chassis_name} 的反馈数据失败: {e}",
                                        exc_info=True,
                                    )
                    except Exception as e:
                        logger.error(f"调用反馈函数失败: {e}", exc_info=True)
                        feedback_waveform = None  # 重置为None，表示反馈失败

                # 处理导出
                if self._enable_export:
                    # 增加导出计数
                    self._chunks_num += 1
                    # 导出所有波形：AI、Static AO、Feedback AO
                    self._export_function(
                        ai_waveform,
                        self._static_output_waveform,
                        feedback_waveform,
                        self._chunks_num,
                    )
                else:
                    # 重置导出计数
                    self._chunks_num = 0

            except Exception as e:
                logger.error(f"数据处理异常: {e}", exc_info=True)

        logger.debug("工作线程已退出")

    def start(self):
        """
        启动同步的连续 AI/AO 任务

        配置并启动硬件同步的 AI 和 AO 任务，使用StartTrigger导出/接收同步。

        Raises:
            RuntimeError: 当任务启动失败时
        """
        if self._is_running:
            logger.warning("任务已在运行中")
            return

        try:
            # 为每个机箱创建并配置 AI 任务
            self._setup_ai_tasks()

            # 为每个机箱创建并配置 Static AO 任务
            self._setup_ao_tasks("ao_static")

            # 为每个机箱创建并配置 Feedback AO 任务
            self._setup_ao_tasks("ao_feedback")

            # 配置StartTrigger导出和接收同步
            self._setup_start_trigger_sync()

            # 启动工作线程
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_thread_function,
                name="MultiChasCSIO_Worker",
            )
            self._worker_thread.start()

            # 启动所有Feedback AO任务（等待StartTrigger触发）
            for chassis_name, ao_task in self._tasks["ao_feedback"].items():
                ao_task.start()
                logger.debug(
                    f"Feedback AO任务 {chassis_name} 已启动（等待StartTrigger触发）"
                )

            # 启动所有Static AO任务（等待StartTrigger触发）
            for chassis_name, ao_task in self._tasks["ao_static"].items():
                ao_task.start()
                logger.debug(
                    f"Static AO任务 {chassis_name} 已启动（等待StartTrigger触发）"
                )

            # 启动所有AI任务（等待StartTrigger触发）
            for chassis_name, ai_task in self._tasks["ai"].items():
                ai_task.start()
                logger.debug(f"AI任务 {chassis_name} 已启动（等待StartTrigger触发）")

            self._is_running = True
            logger.info("跨机箱同步 AI/AO 任务启动成功，使用StartTrigger同步")

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
        # 停止并关闭所有AI任务
        for chassis_name, ai_task in self._tasks["ai"].items():
            try:
                ai_task.stop()
                ai_task.close()
                logger.debug(f"AI 任务 {chassis_name} 已关闭")
            except Exception as e:
                logger.warning(f"关闭 AI 任务 {chassis_name} 时出错: {e}")

        self._tasks["ai"].clear()

        # 停止并关闭所有Static AO任务
        for chassis_name, ao_task in self._tasks["ao_static"].items():
            try:
                ao_task.stop()
                ao_task.close()
                logger.debug(f"Static AO 任务 {chassis_name} 已关闭")
            except Exception as e:
                logger.warning(f"关闭 Static AO 任务 {chassis_name} 时出错: {e}")

        self._tasks["ao_static"].clear()

        # 停止并关闭所有Feedback AO任务
        for chassis_name, ao_task in self._tasks["ao_feedback"].items():
            try:
                ao_task.stop()
                ao_task.close()
                logger.debug(f"Feedback AO 任务 {chassis_name} 已关闭")
            except Exception as e:
                logger.warning(f"关闭 Feedback AO 任务 {chassis_name} 时出错: {e}")

        self._tasks["ao_feedback"].clear()

        # 清空缓冲区
        self._ai_queue.clear()
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
