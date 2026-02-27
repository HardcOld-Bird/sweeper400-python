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


class SingleChasCSIO:
    """
    # 单机箱连续同步 AI/AO 类

    该类专门用于处理单机箱内的多通道AI/AO场景。
    所有通道位于同一机箱，因此可以使用单个AI任务和单个/两个AO任务，
    无需跨机箱同步触发机制。

    ## 主要特性：
        - 支持单机箱内的多通道AI输入和AO输出
        - 创建单个AI任务和最多两个AO任务（Static和Feedback）
        - 硬件自动同步的连续 AI/AO 任务（同一机箱内自动同步）
        - AI 通道使用麦克风模式
        - 支持两种AO模式：
          * Static AO：使用再生模式，循环播放固定波形
          * Feedback AO：使用非再生模式，根据AI数据实时生成输出
        - 支持运行时动态更换静态输出波形
        - 基于回调函数的数据处理和反馈控制
        - 线程安全的数据传输控制

    ## 同步机制：
        - 所有通道位于同一机箱，使用相同的采样时钟
        - 硬件自动同步，无需额外的触发配置
        - 简化的任务管理和数据处理流程

    ## 使用示例：
    ```python
    from sweeper400.analyze import init_sampling_info, init_sine_args, get_sine_cycles
    from sweeper400.measure.cont_sync_io import SingleChasCSIO
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

    # 单机箱多通道示例
    sync_io = SingleChasCSIO(
        ai_channels=(
            "PXI1Slot2/ai0",
            "PXI1Slot3/ai0",
            "PXI1Slot3/ai1",
        ),
        ao_channels_static=(
            "PXI1Slot2/ao0",
            "PXI1Slot3/ao0",
        ),
        ao_channels_feedback=(
            "PXI1Slot3/ao1",
        ),
        static_output_waveform=static_output_waveform,
        feedback_function=feedback_function,
        export_function=export_data,
        # 可选：自定义缓冲区配置（如遇到缓冲区溢出警告时）
        # buffer_size_multiplier=5,  # AI和Feedback AO缓冲区大小倍数，默认5
    )

    # 启动任务
    sync_io.start()
    sync_io.enable_export = True

    # 动态更换静态输出波形
    new_waveform = get_sine_cycles(sampling_info, init_sine_args(2000.0, 0.02, 0.0))
    sync_io.update_static_output_waveform(new_waveform)

    # 停止任务
    sync_io.stop()
    ```

    ## 注意事项：
        - 所有通道必须位于同一机箱
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
        初始化单机箱连续同步 AI/AO 任务

        Args:
            ai_channels: AI 通道名称元组（例如 ("PXI1Slot2/ai0", "PXI1Slot3/ai0")）
            ao_channels_static: 静态AO通道名称元组，使用再生模式
                （例如 ("PXI1Slot2/ao0",)）
            ao_channels_feedback: 反馈AO通道名称元组，使用非再生模式
                （例如 ("PXI1Slot3/ao0",)）
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

        # 私有属性 - 通道信息（简化的数据结构）
        self._ai_channels = ai_channels
        self._ao_channels_static = ao_channels_static
        self._ao_channels_feedback = ao_channels_feedback

        # 私有属性 - 任务管理（简化的数据结构）
        self._ai_task: nidaqmx.Task | None = None
        self._ao_task_static: nidaqmx.Task | None = None
        self._ao_task_feedback: nidaqmx.Task | None = None

        # 私有属性 - 状态管理
        self._is_running = False
        self._worker_thread: threading.Thread | None = None
        self._data_ready_event = threading.Event()  # 数据就绪事件
        self._stop_event = threading.Event()
        self._queue_lock = threading.Lock()  # 保护队列的锁

        # 私有属性 - 数据缓冲和控制
        self._ai_queue: deque[np.ndarray] = deque()  # 存储AI数据的双端队列
        self._chunks_num = 0
        self._enable_export = False

        # 处理静态输出波形
        if ao_channels_static:
            self._static_output_waveform = self._process_waveform_for_channels(
                static_output_waveform, ao_channels_static
            )
        else:
            self._static_output_waveform = static_output_waveform

        # 输出初始化完成日志
        logger.info(
            f"SingleChasCSIO 实例已创建 - "
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
    def _get_terminal_name_with_dev_prefix(
        task: nidaqmx.Task, terminal_name: str
    ) -> str:
        """
        获取带设备前缀的终端名称

        Args:
            task: 任务对象
            terminal_name: 终端名称（例如 "te0/StartTrigger"）

        Returns:
            带设备前缀的完整终端名称（例如 "/PXI1Slot2/te0/StartTrigger"）

        Raises:
            RuntimeError: 当任务中没有找到合适的设备时
        """
        from nidaqmx.constants import ProductCategory

        for device in task.devices:
            if device.product_category not in [
                ProductCategory.C_SERIES_MODULE,
                ProductCategory.SCXI_MODULE,
            ]:
                return f"/{device.name}/{terminal_name}"

        raise RuntimeError("在任务中未找到合适的设备")

    @staticmethod
    def _process_waveform_for_channels(
        waveform: Waveform,
        channel_names: tuple[str, ...],
    ) -> Waveform:
        """
        处理波形以匹配通道要求（单通道扩展/补充通道信息/检查匹配）

        Args:
            waveform: 输入波形对象
            channel_names: 目标通道名称元组

        Returns:
            处理后的波形对象

        Raises:
            ValueError: 当波形通道数与目标通道数不匹配时
        """
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
        return len(self._ai_channels)

    @property
    def ao_channels_num_static(self) -> int:
        """Static AO 通道数量"""
        return len(self._ao_channels_static)

    @property
    def ao_channels_num_feedback(self) -> int:
        """Feedback AO 通道数量"""
        return len(self._ao_channels_feedback)

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
        该方法仅会更新Static AO任务的波形数据。

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
        processed_waveform = self._process_waveform_for_channels(
            new_waveform, self._ao_channels_static
        )

        # 验证采样率是否匹配
        if processed_waveform.sampling_rate != self._sampling_info["sampling_rate"]:
            logger.warning(
                f"新波形采样率 ({processed_waveform.sampling_rate} Hz) "
                f"与原采样率 ({self._sampling_info['sampling_rate']} Hz) 不匹配"
            )

        # 更新内部波形
        self._static_output_waveform = processed_waveform

        # 更新Static AO任务的波形数据
        if self._ao_task_static is not None:
            try:
                self._write_ao_task_waveform_static()
                logger.info("静态输出波形已更新")
            except Exception as e:
                logger.error(f"更新静态波形失败: {e}", exc_info=True)
                raise

        logger.info(
            f"静态输出波形已更新 - shape: {self._static_output_waveform.shape}, "
            f"采样率: {self._static_output_waveform.sampling_rate} Hz"
        )

    def _setup_ai_task(self):
        """
        创建并配置AI任务

        配置为麦克风模式，使用 IEPE 激励电流。
        """
        logger.info("开始创建AI任务")

        # 创建任务
        task_name = "SingleChasCSIO_AI"
        self._ai_task = nidaqmx.Task(task_name)

        logger.debug(f"创建AI任务: {task_name}")

        # 添加所有AI通道（麦克风模式）
        for channel_name in self._ai_channels:
            self._ai_task.ai_channels.add_ai_microphone_chan(  # type: ignore
                channel_name,
                units=SoundPressureUnits.PA,
                mic_sensitivity=0.004,
                max_snd_press_level=120.0,
                current_excit_source=ExcitationSource.INTERNAL,
                current_excit_val=0.004,
            )
            logger.debug(f"  添加通道: {channel_name}")

        # 配置时钟源和采样
        self._ai_task.timing.cfg_samp_clk_timing(  # type: ignore
            rate=self._sampling_info["sampling_rate"],
            sample_mode=AcquisitionType.CONTINUOUS,
        )

        # 配置缓冲区大小（使用可配置的倍数）
        buffer_size = (
            self._static_output_waveform.samples_num * self._buffer_size_multiplier
        )
        self._ai_task.in_stream.input_buf_size = buffer_size
        logger.debug(f"  设置AI缓冲区大小: {buffer_size} 样本")

        # 注册回调函数
        callback_samples = self._static_output_waveform.samples_num
        self._ai_task.register_every_n_samples_acquired_into_buffer_event(  # type: ignore
            callback_samples, self._ai_callback
        )
        logger.debug(f"  已注册AI回调函数，每 {callback_samples} 样本触发一次")

        logger.info("AI任务创建成功")

    def _setup_ao_task(self, task_type: str, start_trigger_terminal: str | None = None):
        """
        创建并配置AO任务（Static或Feedback）

        Args:
            task_type: 任务类型，可选值为 "static" 或 "feedback"
            start_trigger_terminal: 开始触发器终端名称，用于同步触发。
                如果为None，则不配置触发器

        Raises:
            ValueError: 当task_type参数无效时
        """
        # 验证参数
        if task_type not in ("static", "feedback"):
            raise ValueError(
                f"无效的task_type参数: {task_type}，必须为'static'或'feedback'"
            )

        # 根据类型选择通道和任务对象
        if task_type == "static":
            channels = self._ao_channels_static
            task_name = "SingleChasCSIO_StaticAO"
            task_attr = "_ao_task_static"
            regen_mode = RegenerationMode.ALLOW_REGENERATION
            buffer_size = self._static_output_waveform.samples_num
        else:  # feedback
            channels = self._ao_channels_feedback
            task_name = "SingleChasCSIO_FeedbackAO"
            task_attr = "_ao_task_feedback"
            regen_mode = RegenerationMode.DONT_ALLOW_REGENERATION
            buffer_size = (
                self._static_output_waveform.samples_num * self._buffer_size_multiplier
            )

        # 检查是否有通道
        if not channels:
            logger.info(
                f"没有{task_type.capitalize()} AO通道，跳过{task_type.capitalize()} AO任务创建"
            )
            return

        logger.info(f"开始创建{task_type.capitalize()} AO任务")

        # 创建任务
        task = nidaqmx.Task(task_name)
        setattr(self, task_attr, task)

        logger.debug(f"创建{task_type.capitalize()} AO任务: {task_name}")

        # 添加所有AO通道
        for channel_name in channels:
            task.ao_channels.add_ao_voltage_chan(  # type: ignore
                channel_name, min_val=-10.0, max_val=10.0
            )
            logger.debug(f"  添加通道: {channel_name}")

        # 配置时钟源和采样
        task.timing.cfg_samp_clk_timing(  # type: ignore
            rate=self._sampling_info["sampling_rate"],
            sample_mode=AcquisitionType.CONTINUOUS,
        )

        # 配置同步触发器（如果提供了触发器终端）
        if start_trigger_terminal is not None:
            task.triggers.start_trigger.cfg_dig_edge_start_trig(start_trigger_terminal)  # type: ignore
            logger.debug(f"  配置开始触发器: {start_trigger_terminal}")

        # 设置再生模式
        task.out_stream.regen_mode = regen_mode

        # 设置缓冲区大小
        task.out_stream.output_buf_size = buffer_size
        logger.debug(f"  设置{task_type.capitalize()} AO缓冲区大小: {buffer_size} 样本")

        # 预写入波形数据
        if task_type == "static":
            self._write_ao_task_waveform_static()
        else:  # feedback
            self._write_ao_task_waveform_feedback_silence()

        logger.info(f"{task_type.capitalize()} AO任务创建成功")

    def _write_ao_task_waveform_static(self):
        """
        向Static AO任务写入波形数据
        """
        if self._ao_task_static is None:
            return

        try:
            # 提取波形数据
            if self._static_output_waveform.ndim == 1:
                # 单通道
                waveform_data = np.asarray(self._static_output_waveform)
            else:
                # 多通道
                waveform_data = np.asarray(self._static_output_waveform)

            # 写入数据
            self._ao_task_static.write(waveform_data, auto_start=False)  # type: ignore
            logger.debug(f"成功写入静态波形数据，shape: {waveform_data.shape}")

        except Exception as e:
            logger.error(f"静态波形写入失败: {e}", exc_info=True)
            raise

    def _write_ao_task_waveform_feedback_silence(self):
        """
        向Feedback AO任务写入全0静音波形

        注意：此方法一次性写入2个chunk长度的静音波形。
        这是因为所有AI和AO任务同步开始，当AO输出一个chunk时，
        AI任务才刚刚返回第一段采集到的chunk，而此时反馈AO任务的缓冲区已经空了。
        我们必须先为反馈AO任务写入2个chunk，这样我们才有足够的时间处理反馈逻辑，
        并写入反馈AO缓冲区。
        """
        if self._ao_task_feedback is None:
            return

        try:
            # 创建全0静音波形（2个chunk长度）
            samples_num = self._static_output_waveform.samples_num * 2  # 2个chunk
            channels_num = len(self._ao_channels_feedback)

            if channels_num == 1:
                # 单通道：一维数组
                silence_waveform = np.zeros(samples_num, dtype=np.float64)
            else:
                # 多通道：二维数组
                silence_waveform = np.zeros(
                    (channels_num, samples_num), dtype=np.float64
                )

            # 写入数据
            self._ao_task_feedback.write(silence_waveform, auto_start=False)  # type: ignore
            logger.debug(
                f"成功写入静音波形数据（2个chunk），shape: {silence_waveform.shape}"
            )

        except Exception as e:
            logger.error(f"静音波形写入失败: {e}", exc_info=True)
            raise

    def _write_ao_task_waveform_feedback(self, feedback_waveform: Waveform):
        """
        向Feedback AO任务写入波形数据

        Args:
            feedback_waveform: 反馈波形数据（包含所有feedback通道）
        """
        if self._ao_task_feedback is None:
            return

        try:
            # 提取波形数据
            channels_num = len(self._ao_channels_feedback)

            if feedback_waveform.ndim == 1:
                # 单通道：直接使用
                waveform_data = np.asarray(feedback_waveform)
            else:
                # 多通道
                if channels_num == 1:
                    # 单通道任务：提取一维数组
                    waveform_data = np.asarray(feedback_waveform[0, :])
                else:
                    # 多通道任务：使用二维数组
                    waveform_data = np.asarray(feedback_waveform)

            # 写入数据
            self._ao_task_feedback.write(waveform_data, auto_start=False)  # type: ignore
            logger.debug(f"成功写入反馈波形数据，shape: {waveform_data.shape}")

        except Exception as e:
            logger.error(f"反馈波形写入失败: {e}", exc_info=True)
            raise

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
        快速读取数据并加入队列，然后通知工作线程处理。
        （未使用的参数不可删除，其为NI-DAQmx回调函数的API要求）

        Args:
            task_handle: 任务句柄
            every_n_samples_event_type: 事件类型
            number_of_samples: 样本数量
            callback_data: 回调数据
        """
        try:
            # 检查任务是否仍在运行
            if not self._is_running:
                return 0

            # 读取原始数据
            data = self._ai_task.read(  # type: ignore
                number_of_samples_per_channel=number_of_samples
            )

            # 将数据加入队列并设置事件
            with self._queue_lock:
                self._ai_queue.append(data)

            # 通知工作线程有新数据
            self._data_ready_event.set()

        except Exception as e:
            logger.error(f"AI 回调函数异常: {e}", exc_info=True)

        return 0

    def _worker_thread_function(self):
        """
        工作线程函数

        负责从队列取出数据并调用导出函数，同时处理feedback数据并写入Feedback AO任务。

        触发机制：
        - 使用Event机制等待AI回调通知
        - 收到通知后立即处理队列中的所有数据
        """
        logger.debug("工作线程已启动")

        # 计算超时时间：波形持续时间
        timeout = self._static_output_waveform.duration

        while not self._stop_event.is_set():
            # 等待数据就绪事件
            data_ready = self._data_ready_event.wait(timeout=timeout)

            if not data_ready:
                # 超时，尝试处理数据
                continue

            # 清除事件标志
            self._data_ready_event.clear()

            # 处理队列中的所有数据
            while True:
                # 从队列中取出数据
                with self._queue_lock:
                    if not self._ai_queue:
                        # 队列为空，退出循环
                        break
                    raw_data = self._ai_queue.popleft()

                # 在锁外处理数据（避免长时间持有锁）
                try:
                    # 将原始数据转换为numpy数组并确保是2D格式
                    # 注意：nidaqmx的read()方法：
                    # - 单通道时返回1D列表 [sample1, sample2, ...]
                    # - 多通道时返回2D列表 [[ch1_sample1, ch1_sample2, ...], [ch2_sample1, ch2_sample2, ...]]
                    data_array = np.array(raw_data, dtype=np.float64)
                    if data_array.ndim == 1:
                        # 单通道数据，reshape为(1, samples)
                        data_array = data_array.reshape(1, -1)

                    # 转换为 Waveform 对象，添加通道名称元数据
                    ai_waveform = Waveform(
                        input_array=data_array,
                        sampling_rate=self._sampling_info["sampling_rate"],
                        channel_names=self._ai_channels,
                    )

                    # 处理反馈
                    feedback_waveform = None
                    if self._ao_channels_feedback:
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
                                # 写入Feedback AO任务
                                self._write_ao_task_waveform_feedback(feedback_waveform)
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

        配置并启动硬件同步的 AI 和 AO 任务。

        Raises:
            RuntimeError: 当任务启动失败时
        """
        if self._is_running:
            logger.warning("任务已在运行中")
            return

        try:
            # 创建并配置 AI 任务
            self._setup_ai_task()

            # 获取AI任务的开始触发器终端名称，用于同步AO任务
            ai_start_trigger_terminal = self._get_terminal_name_with_dev_prefix(
                self._ai_task,
                "te0/StartTrigger",  # type: ignore
            )
            logger.debug(f"AI开始触发器终端: {ai_start_trigger_terminal}")

            # 创建并配置 Static AO 任务（使用AI触发器同步）
            self._setup_ao_task(
                "static", start_trigger_terminal=ai_start_trigger_terminal
            )

            # 创建并配置 Feedback AO 任务（使用AI触发器同步）
            self._setup_ao_task(
                "feedback", start_trigger_terminal=ai_start_trigger_terminal
            )

            # 启动工作线程
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_thread_function,
                name="SingleChasCSIO_Worker",
            )
            self._worker_thread.start()

            # 启动所有任务
            # 注意：启动顺序很重要！
            # 1. 先启动AO任务（它们会等待AI的开始触发信号）
            # 2. 最后启动AI任务（触发所有任务同步开始）
            if self._ao_task_static is not None:
                self._ao_task_static.start()
                logger.debug("Static AO任务已启动（等待触发）")

            if self._ao_task_feedback is not None:
                self._ao_task_feedback.start()
                logger.debug("Feedback AO任务已启动（等待触发）")

            if self._ai_task is not None:
                self._ai_task.start()
                logger.debug("AI任务已启动（触发所有任务）")

            self._is_running = True
            logger.info("单机箱同步 AI/AO 任务启动成功")

        except Exception as e:
            logger.error(f"任务启动失败: {e}", exc_info=True)
            self._cleanup_tasks()
            raise RuntimeError(f"单机箱同步 AI/AO 任务启动失败: {e}") from e

    def stop(self):
        """
        停止同步的连续 AI/AO 任务

        停止所有任务并清理资源。
        """
        if not self._is_running:
            logger.warning("任务未运行")
            return

        logger.info("正在停止单机箱同步 AI/AO 任务...")

        # 立即标记停止状态，防止回调函数继续添加数据
        self._is_running = False

        # 停止工作线程
        self._stop_event.set()
        # 唤醒工作线程（设置事件确保线程能够退出）
        self._data_ready_event.set()

        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)
            if self._worker_thread.is_alive():
                logger.warning("工作线程未能在超时时间内退出")
            else:
                logger.debug("工作线程已停止")
            self._worker_thread = None

        # 清理任务
        self._cleanup_tasks()

        logger.info("单机箱同步 AI/AO 任务已停止")

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
            self._ai_task = None

        # 停止并关闭Static AO任务
        if self._ao_task_static is not None:
            try:
                self._ao_task_static.stop()
                self._ao_task_static.close()
                logger.debug("Static AO 任务已关闭")
            except Exception as e:
                logger.warning(f"关闭 Static AO 任务时出错: {e}")
            self._ao_task_static = None

        # 停止并关闭Feedback AO任务
        if self._ao_task_feedback is not None:
            try:
                self._ao_task_feedback.stop()
                self._ao_task_feedback.close()
                logger.debug("Feedback AO 任务已关闭")
            except Exception as e:
                logger.warning(f"关闭 Feedback AO 任务时出错: {e}")
            self._ao_task_feedback = None

        # 清空缓冲区和计数器
        with self._queue_lock:
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
