"""
# 连续同步 AI/AO 模块

模块路径：`sweeper400.measure.cont_sync_io`

包含同步的连续 AI 和 AO 任务实现，用于各种信号的同步生成和采集。
（"CSIO"是"Continuous Synchronous AI/AO"的缩写）
"""

import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import nidaqmx
import numpy as np
from nidaqmx.constants import (
    AcquisitionType,
    ExcitationSource,
    RegenerationMode,
    SoundPressureUnits,
    WriteRelativeTo,
    ProductCategory,
)

from ..analyze import (
    PositiveInt,
    TFData,
    Waveform,
    load_data_with_fallback,
    comp_waveform,
    pick_waveform_channels,
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
        - AI 任务启动时，通过 PXI_Trig7 背板线路向 AO 任务发送硬件触发信号
        - AO 任务预先配置为等待该触发信号，确保 AI/AO 精确同步启动
        - 使用 `export_signals` 显式路由触发信号，避免 DSA 多设备同步时的时序引擎资源冲突
          （技术背景详见：`参考资料/NI-DAQmx_DSA多设备同步时序引擎冲突问题分析.md`）

    ## 使用示例：
    ```python
    from sweeper400.analyze import init_sampling_info, get_sine
    from sweeper400.measure.cont_sync_io import SingleChasCSIO
    import numpy as np

    # 创建采样信息和静态输出波形
    sampling_info = init_sampling_info(48000, 4800)
    cca = np.array([0.02 + 0j])  # 单通道，幅值0.02，相位0
    static_output_waveform = get_sine(
        sampling_info, 1000.0, ("PXI1Slot2/ao0",), cca, full_cycle=True
    )

    # 定义反馈函数（根据补偿后的AI波形生成AO输出）
    # 注意：feedback_function现在接收AI波形、静态输出波形、当前播放的feedback波形和TFData
    def feedback_function(ai_waveform, static_wf, currently_playing, tf_data):
        # 示例：根据AI波形生成反馈波形
        feedback_cca = np.array([0.01 + 0j])  # feedback通道复振幅
        feedback_waveform = get_sine(
            sampling_info=sampling_info,
            frequency=1000.0,
            channel_names=("PXI1Slot3/ao1",),
            channel_complex_amplitudes=feedback_cca,
        )
        return feedback_waveform

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
        # 可选：指定补偿数据文件路径
        # ao_comp_data="path/to/ao_comp.pkl",
        # ai_comp_data="path/to/ai_comp.pkl",
    )

    # 启动任务
    sync_io.start()
    sync_io.enable_export = True

    # 动态更换静态输出波形
    new_cca = np.array([0.02 + 0j, 0.02 + 0j])  # 示例：2通道
    new_waveform = get_sine(
        sampling_info, 2000.0,
        ("PXI1Slot2/ao0", "PXI1Slot3/ao0"), new_cca, full_cycle=True
    )
    sync_io.update_static_output_waveform(new_waveform)

    # 停止任务
    sync_io.stop()
    ```

    ## 注意事项：
        - 所有通道必须位于同一机箱
        - feedback_function接收四个参数：ai_waveform, static_output_waveform,
          currently_playing_feedback_waveform, fishnet_tf_data
        - feedback_function必须返回与feedback_ao_channels数量匹配的Waveform对象
        - 如果提供了ao_comp_data或ai_comp_data，将自动应用补偿
        - Static AO和Feedback AO通道至少要有一个非空
    """

    # 获取类日志器（类属性，所有实例共享）
    logger = get_logger(f"{__name__}.SingleChasCSIO")

    def __init__(
        self,
        ai_channels: tuple[str, ...],
        ao_channels_static: tuple[str, ...],
        ao_channels_feedback: tuple[str, ...],
        static_output_waveform: Waveform,
        export_function: Callable[[Waveform, Waveform, Waveform | None, PositiveInt], Any],
        feedback_function: Callable[[Waveform, Waveform, Waveform, TFData], Waveform] | None = None,
        buffer_size_multiplier: PositiveInt = 5,
        ai_comp_data: str | Path | None = None,
        ao_comp_data: str | Path | None = None,
        fishnet_tf_data: str | Path | None = None,
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
            export_function: 数据导出函数，接收 (ai_waveform, ao_static_waveform,
                ao_feedback_waveform, chunks_num) 参数。
                其中ao_static_waveform是当前的静态输出波形，
                ao_feedback_waveform是当前的反馈输出波形（如果没有反馈通道则为None）
            feedback_function: 反馈函数，接收三个参数：
                1. ai_waveform: 刚刚采集到的AI数据
                2. static_output_waveform: 静态输出波形对象
                3. currently_playing_feedback_waveform: 当前正在播放的feedback波形（最旧的队列项）
                4. fishnet_tf_data: TFData对象，包含传递函数数据
                返回feedback AO数据（多通道Waveform）。如果为None，使用默认静音函数。
            buffer_size_multiplier: AI和Feedback AO缓冲区大小倍数
                （相对于static_output_waveform.samples_num），
                默认为5。增大此值可减少缓冲区溢出风险，但会增加内存占用。
            ai_comp_data: AI补偿数据文件路径（可选），用于补偿输入信号的正弦参数。
                如果提供，将使用load_data_with_fallback加载并存储为属性。
            ao_comp_data: AO补偿数据文件路径（可选），用于补偿输出波形。
                如果提供，将使用load_data_with_fallback加载并存储为属性。
            fishnet_tf_data: Fishnet传递函数数据文件路径（可选），用于反馈控制。
                如果提供，将使用load_data_with_fallback加载并存储为属性。

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
        # 如果feedback_function为None，使用默认的静音函数
        self._feedback_function = (
            feedback_function if feedback_function is not None else self._default_feedback_function
        )

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
        self._ai_queue_lock = threading.Lock()  # 保护AI数据队列的锁

        # 私有属性 - 数据缓冲和控制
        self._ai_queue: deque[Any] = deque()  # 存储AI数据的双端队列
        self._feedback_ao_queue: deque[Waveform] = deque()  # 存储已写入的feedback波形副本
        self._chunks_num = 0
        self._enable_export = False

        # 加载AI补偿数据
        self._ai_comp_data = load_data_with_fallback(
            explicit_path=ai_comp_data,
            default_path=Path("storage/calib/calib_result_anemone/ai_comp_data.pkl"),
            data_type="AI补偿数据",
        )
        # 加载AO补偿数据
        self._ao_comp_data = load_data_with_fallback(
            explicit_path=ao_comp_data,
            default_path=Path("storage/calib/calib_result_octopus/ao_comp_data.pkl"),
            data_type="AO补偿数据",
        )
        # 加载Fishnet传递函数数据
        self._fishnet_tf_data = load_data_with_fallback(
            explicit_path=fishnet_tf_data,
            default_path=Path("storage/calib/calib_result_fishnet/tf_data.pkl"),
            data_type="Fishnet传递函数数据",
        )

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
        for device in task.devices:
            if device.product_category not in [
                ProductCategory.C_SERIES_MODULE,
                ProductCategory.SCXI_MODULE,
            ]:
                return f"/{device.name}/{terminal_name}"

        raise RuntimeError("在任务中未找到合适的设备")

    @staticmethod
    def _default_feedback_function(
        ai_waveform: Waveform,
        static_output_waveform: Waveform | None,
        currently_playing_feedback_waveform: Waveform | None,
        fishnet_tf_data: TFData | None,
    ) -> Waveform:
        """
        默认反馈函数

        当用户未提供feedback_function时使用。返回与currently_playing_feedback_waveform
        形状、通道名称和采样信息完全相同的全零静音波形。

        Args:
            ai_waveform: 刚刚采集到的AI数据
            static_output_waveform: Waveform | None,
            currently_playing_feedback_waveform: Waveform | None,
            fishnet_tf_data: Fishnet传递函数数据

        Returns:
            全零静音波形，形状和元数据与currently_playing_feedback_waveform完全相同

        Raises:
            ValueError: 当currently_playing_feedback_waveform为None时
        """
        if currently_playing_feedback_waveform is None:
            raise ValueError(
                "默认反馈函数需要currently_playing_feedback_waveform参数，"
                "但传入值为None。请确保_feedback_ao_queue中有数据。"
            )

        # 基于currently_playing创建全零波形
        # 保持相同的形状、通道名称和采样信息
        silence_data = np.zeros_like(currently_playing_feedback_waveform)

        return Waveform(
            input_array=silence_data,
            sampling_rate=currently_playing_feedback_waveform.sampling_rate,
            channel_names=currently_playing_feedback_waveform.channel_names,
            timestamp=currently_playing_feedback_waveform.timestamp,
        )

    @staticmethod
    def _process_waveform_for_channels(
        waveform: Waveform,
        channel_names: tuple[str, ...],
    ) -> Waveform:
        """
        处理波形以匹配通道要求（单通道复制扩展/更新通道信息/检查匹配）

        该方法处理波形以匹配目标通道数量：
        1. 如果输入是单通道波形且需要多通道输出，将单通道数据复制到所有通道
        2. 如果波形已有正确通道数，更新channel_names

        Args:
            waveform: 输入波形对象
            channel_names: 目标通道名称元组

        Returns:
            处理后的波形对象（未应用AO补偿）

        Raises:
            ValueError: 当波形通道数与目标通道数不匹配时
        """
        channels_num = len(channel_names)

        # 情况1：单通道输入，需要多通道输出（直接复制单通道数据到所有通道）
        if waveform.is_single_channel and channels_num > 1:
            logger.info(f"静态输出波形为单通道，复制扩展为 {channels_num} 通道")

            # 将单通道数据复制到所有通道
            single_ch_data = np.asarray(waveform)[0, :]  # (n_samples,)
            multi_ch_data = np.tile(single_ch_data, (channels_num, 1))  # (channels_num, n_samples)

            # 扩展channel_complex_amplitude（如果存在）
            if waveform.channel_complex_amplitudes is not None:
                single_cca = waveform.channel_complex_amplitudes[0]
                multi_ch_cca = np.full(channels_num, single_cca, dtype=np.complex128)
            else:
                multi_ch_cca = None

            # 创建多通道Waveform对象
            multi_ch_waveform = Waveform(
                input_array=multi_ch_data,
                sampling_rate=waveform.sampling_rate,
                channel_names=channel_names,
                timestamp=waveform.timestamp,
                waveform_id=waveform.waveform_id,
                frequency=waveform.frequency,
                channel_complex_amplitude=multi_ch_cca,
            )

        # 情况2：多通道输入，检查通道数匹配
        elif waveform.channels_num == channels_num:
            # 使用 pick_waveform_channels 统一处理通道匹配/重排/报错逻辑
            multi_ch_waveform = pick_waveform_channels(waveform, channel_names)
            if waveform.channel_names != channel_names:
                logger.warning(
                    f"静态输出波形的channel_names顺序不匹配"
                    f"（原始: {waveform.channel_names}，目标: {channel_names}），"
                    f"已按目标顺序重新组装波形"
                )

        # 情况3：通道数不匹配
        else:
            raise ValueError(
                f"静态输出波形通道数 ({waveform.channels_num}) "
                f"与目标通道数 ({channels_num}) 不匹配"
            )

        return multi_ch_waveform

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
    def static_output_waveform(self) -> Waveform:
        """静态输出波形"""
        return self._static_output_waveform.copy()

    @property
    def enable_export(self) -> bool:
        """数据导出启用状态"""
        return self._enable_export

    @enable_export.setter
    def enable_export(self, value: bool):
        """设置数据导出启用状态"""
        self._enable_export = value
        if value:
            logger.info("数据导出：ON ✅")
        else:
            logger.info("数据导出：OFF ❌")

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
        raw_waveform = self._process_waveform_for_channels(
            new_waveform, self._ao_channels_static
        )

        # 验证采样率是否匹配
        if raw_waveform.sampling_rate != self._sampling_info["sampling_rate"]:
            logger.warning(
                f"新波形采样率 ({raw_waveform.sampling_rate} Hz) "
                f"与原采样率 ({self._sampling_info['sampling_rate']} Hz) 不匹配"
            )

        # 更新内部波形
        self._static_output_waveform = raw_waveform

        # 更新Static AO任务的波形数据
        if self._ao_task_static is not None:
            try:
                self._write_ao_task_waveform_static()
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
        self._ai_task.register_every_n_samples_acquired_into_buffer_event(
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

        # 创建任务
        task = nidaqmx.Task(task_name)
        setattr(self, task_attr, task)

        logger.debug(f"创建{task_type.capitalize()} AO任务: {task_name}")

        # 添加所有AO通道
        for channel_name in channels:
            task.ao_channels.add_ao_voltage_chan(
                channel_name, min_val=-10.0, max_val=10.0
            )
            logger.debug(f"  添加通道: {channel_name}")

        # 配置时钟源和采样
        task.timing.cfg_samp_clk_timing(
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
            self._write_ao_task_waveform_feedback_silent()

        logger.info(f"{task_type.capitalize()} AO任务创建成功")

    def _write_ao_task_waveform_static(self):
        """
        向Static AO任务写入波形数据

        使用 WriteRelativeTo.FIRST_SAMPLE 从缓冲区起始位置覆盖写入，
        这是在 ALLOW_REGENERATION 模式运行中更新波形的正确方法。
        若使用默认的 CURRENT_WRITE_POSITION，在缓冲区首次写满后，
        后续写入将因无可用空间而超时失败（错误码 -200292）。

        注意：nidaqmx对单通道和多通道任务的数据格式要求不同：
        - 单通道任务：需要1D数组，shape=(samples,)
        - 多通道任务：需要2D数组，shape=(channels, samples)
        """
        if self._ao_task_static is None:
            return

        try:
            # 写入之前，对静态输出波形应用AO补偿
            if self._ao_comp_data is not None:
                logger.debug("对静态输出波形应用AO补偿")
                comped_static_waveform: Waveform = comp_waveform(
                    self._static_output_waveform,
                    self._ao_comp_data,
                )
            else:
                comped_static_waveform = self._static_output_waveform

            # 根据通道数处理数据格式
            # nidaqmx要求：单通道任务使用1D数组，多通道任务使用2D数组，与项目约定有出入
            channels_num = self.ao_channels_num_static
            if channels_num == 1:
                # 单通道任务：将2D数组squeeze成1D数组
                write_data = comped_static_waveform.squeeze(axis=0)
                logger.debug(f"单通道任务，将波形数据squeeze为1D: {write_data.shape}")
            else:
                # 多通道任务：保持2D数组
                write_data = comped_static_waveform

            # 从缓冲区起始位置（position 0）覆盖写入新波形。
            # 这是在 ALLOW_REGENERATION 模式下更新波形的标准做法：
            # 由于再生模式不释放缓冲区空间，默认的 CURRENT_WRITE_POSITION
            # 会永远阻塞（CurrWritePos 一直停在缓冲区末尾），而 FIRST_SAMPLE
            # 则直接覆盖写入，下一个再生周期即生效。
            self._ao_task_static.out_stream.relative_to = WriteRelativeTo.FIRST_SAMPLE
            self._ao_task_static.out_stream.offset = 0

            # 写入数据
            self._ao_task_static.write(write_data, auto_start=False)
            logger.debug(f"成功写入静态波形数据，shape: {write_data.shape}")

        except Exception as e:
            logger.error(f"静态波形写入失败: {e}", exc_info=True)
            raise

    def _write_ao_task_waveform_feedback_silent(self):
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

            # 统一使用2D格式创建静音波形 (channels_num, samples_num)
            silent_waveform = np.zeros((channels_num, samples_num), dtype=np.float64)

            # 根据通道数决定写入格式
            if channels_num == 1:
                # 单通道任务：squeeze为1D数组
                write_data = silent_waveform.squeeze(axis=0)
            else:
                # 多通道任务：保持2D数组
                write_data = silent_waveform

            # 写入数据（全0波形无需补偿）
            self._ao_task_feedback.write(write_data, auto_start=False)

            # 缓存写入的波形副本（按chunk分割存储）
            chunk_samples = self._static_output_waveform.samples_num
            for i in range(2):  # 2个chunk
                start_idx = i * chunk_samples
                end_idx = (i + 1) * chunk_samples
                sliced_silent_waveform = Waveform(
                    input_array=silent_waveform[:, start_idx:end_idx].copy(),
                    sampling_rate=self._sampling_info["sampling_rate"],
                    channel_names=self._ao_channels_feedback,
                )
                self._feedback_ao_queue.append(sliced_silent_waveform)

            logger.debug(
                f"成功写入静音波形数据（2个chunk），shape: {sliced_silent_waveform.shape}"  # noqa
            )

        except Exception as e:
            logger.error(f"静音波形写入失败: {e}", exc_info=True)
            raise

    def _write_ao_task_waveform_feedback(self, raw_feedback_waveform: Waveform):
        """
        向Feedback AO任务写入波形数据

        Args:
            raw_feedback_waveform: 未补偿的反馈波形数据（包含所有feedback通道）
        """
        if self._ao_task_feedback is None:
            return

        try:
            # 写入之前，对feedback波形应用AO补偿
            if self._ao_comp_data is not None:
                logger.debug("对反馈波形应用AO补偿")
                comped_feedback_waveform: Waveform = comp_waveform(
                    raw_feedback_waveform,
                    self._ao_comp_data,
                )
            else:
                comped_feedback_waveform = raw_feedback_waveform

            # 写入数据
            self._ao_task_feedback.write(comped_feedback_waveform, auto_start=False)

            # 缓存写入的波形副本（保持2D格式）
            self._feedback_ao_queue.append(raw_feedback_waveform.copy())

            logger.debug(f"成功写入反馈波形数据，shape: {comped_feedback_waveform.shape}")

        except Exception as e:
            logger.error(f"反馈波形写入失败: {e}", exc_info=True)
            raise

    def _ai_callback(
        self,
        task_handle,
        every_n_samples_event_type,
        number_of_samples,
        callback_data,
    ):
        """
        AI 任务回调函数

        在每次采集到指定数量的样本后被调用。
        快速读取数据并加入队列，然后通知工作线程处理。
        （因此应保持高效，以避免阻塞数据采集）
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
            data = self._ai_task.read(  # noqa
                number_of_samples_per_channel=number_of_samples
            )

            # 将数据加入队列并设置事件
            with self._ai_queue_lock:
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
        包含AI补偿和AO补偿逻辑：
        1. 从raw_data提取每个AI通道的SineArgs并应用AI补偿
        2. 使用补偿后的SineArgs字典调用feedback_function
        3. 对feedback_function返回的波形应用AO补偿后写入任务

        触发机制：
        - 使用Event机制等待AI回调通知
        - 收到通知后立即处理队列中的所有数据
        """
        logger.debug("工作线程已启动")

        # 计算超时时间：波形持续时间
        timeout = self._static_output_waveform.duration

        while not self._stop_event.is_set():
            # 等待数据就绪事件
            self._data_ready_event.wait(timeout=timeout)

            # 清除事件标志
            self._data_ready_event.clear()

            # 处理队列中的所有数据
            while True:
                # 从队列中取出数据
                with self._ai_queue_lock:
                    if not self._ai_queue:
                        # 队列为空，退出循环
                        break
                    raw_data = self._ai_queue.popleft()

                # 在锁外处理数据（避免长时间持有锁）
                try:
                    # ===== 任务1：数据预处理 =====
                    # 注意：nidaqmx的read()方法：
                    # - 单通道时返回1D列表 [sample1, sample2, ...]
                    # - 多通道时返回2D列表 [[ch1_sample1, ch1_sample2, ...], [ch2_sample1, ch2_sample2, ...]]
                    # 将原始数据转换为 Waveform 对象，添加通道名称元数据
                    # Waveform.__new__ 会自动将1D数据reshape为(1, samples)的2D格式
                    ai_waveform = Waveform(
                        input_array=raw_data,
                        sampling_rate=self._sampling_info["sampling_rate"],
                        channel_names=self._ai_channels,
                    )
                    # 首先，对AI波形应用AI补偿
                    if self._ai_comp_data is not None:
                        logger.debug("对AI波形应用AI补偿")
                        ai_waveform = comp_waveform(
                            ai_waveform,
                            self._ai_comp_data,
                        )

                    # ===== 任务2：处理反馈 =====
                    feedback_waveform = None
                    if self._ao_channels_feedback:
                        try:
                            # ===== 获取对应feedback波形 =====
                            # logger.warning("开始获取对应feedback波形")
                            # 从队列中取出最旧的波形（采集时正在播放的）
                            currently_playing: Waveform | None = None
                            if self._feedback_ao_queue:
                                currently_playing = self._feedback_ao_queue.popleft()

                            # ===== 调用反馈函数 =====
                            # logger.warning("开始调用反馈函数")
                            # 使用补偿后的AI波形、当前播放的波形和fishnet_tf_data调用feedback_function
                            feedback_waveform = self._feedback_function(
                                ai_waveform,  # 使用已补偿的AI波形，消除硬件偏差
                                self._static_output_waveform,  # 使用未补偿的AO波形，方便处理
                                currently_playing,  # 未补偿的AO波形
                                self._fishnet_tf_data,
                            )

                            # ===== 波形写入阶段 =====
                            # logger.warning("开始写入Feedback AO任务")
                            # 写入Feedback AO任务
                            self._write_ao_task_waveform_feedback(feedback_waveform)

                        except Exception as e:
                            logger.error(f"调用反馈函数失败: {e}", exc_info=True)
                            feedback_waveform = None  # 重置为None，表示反馈失败

                    # ===== 任务3：处理导出 =====
                    if self._enable_export:
                        # 增加导出计数
                        self._chunks_num += 1
                        # 导出所有波形：AI、Static AO、Feedback AO
                        self._export_function(
                            ai_waveform,  # 已补偿的AI波形
                            self._static_output_waveform,  # 未补偿的AO波形
                            feedback_waveform,  # 未补偿的AO波形
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

            # 将 AI 的 StartTrigger 事件显式导出到 PXI_Trig7，供 AO 任务同步触发使用。
            #
            # 【背景】当 AI 任务跨多个 Slot 时，NI-DAQmx 会启动 DSA 多设备同步协议，
            # 在主设备（通常为最低编号 Slot）上自动分配时序引擎：
            #   te0 → AI 主采样时钟（Master Sample Clock）
            #   te2 → AI DSA 同步脉冲分发（Sync Pulse Distribution）
            #   te3 → AO 任务自身的主时序引擎（AO Master Timing Engine）
            # 因此，te0/te2 被 AI 占用（直接监听会造成运行时硬件冲突，报错 -200292），
            # te3 是 AO 自身引擎（若用作 AO 的触发源会构成循环自引用，立即报错 -89131）。
            # 于是无法通过任何 te 的 StartTrigger 终端安全地实现跨任务同步。
            #
            # 【方案】改用独立的 PXI 背板触发线 PXI_Trig7：
            #   1. 通过 export_signals.start_trig_output_term 将 AI 的 StartTrigger
            #      显式路由到 PXI_Trig7（一条与时序引擎资源完全隔离的独立信号线）
            #   2. AO 任务监听同一条 PXI_Trig7 线路，等待并接收 AI 的启动事件
            # DSA 内部同步通常仅占用 PXI_Trig0~Trig2，选择高编号 Trig7 以最大化
            # 与 DSA 内部分配冲突的安全裕量。
            # 技术细节详见：参考资料/多板卡同步时序引擎冲突问题分析.md
            ai_start_trigger_terminal = self._get_terminal_name_with_dev_prefix(
                self._ai_task,  # noqa
                "PXI_Trig7",
            )
            self._ai_task.export_signals.start_trig_output_term = (
                ai_start_trigger_terminal
            )
            logger.debug(f"AI StartTrigger 已显式导出至: {ai_start_trigger_terminal}")

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
        with self._ai_queue_lock:
            self._ai_queue.clear()
        self._feedback_ao_queue.clear()
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
