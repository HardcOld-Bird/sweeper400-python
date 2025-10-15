"""
# 连续同步 AI/AO 模块

模块路径：`sweeper400.measure.cont_sync_io`

包含同步的连续 AI 和 AO 任务实现，用于各种信号的同步生成和采集。
（"CSSIO"是"Continuous Synchronous Sine AI/AO"的缩写）
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

from sweeper400.analyze import (
    PositiveInt,
    Waveform,
)
from sweeper400.logger import get_logger

# 获取模块日志器
logger = get_logger(__name__)


class HiPerfCSSIO:
    """
    # 高性能连续同步 AI/AO 类

    该类提供高性能和简单的同步 AI/AO 任务实现。
    使用固定的输出波形和再生模式，避免了实时波形生成的开销。
    AI 通道配置为麦克风模式，用于声压测量。

    ## 主要特性：
        - 硬件同步的连续 AI/AO 任务
        - AI 通道使用麦克风模式（激励电流 0.004A，灵敏度 4mV/Pa，最大声压级 120dB）
        - 使用固定输出波形，避免实时生成开销
        - 采用再生模式，提高性能和稳定性
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
    from sweeper400.measure.cont_sync_io import HiPerfCSSIO

    # 创建采样信息和固定输出波形
    sampling_info = init_sampling_info(1000, 1000)
    sine_args = init_sine_args(100.0, 1.0, 0.0)
    output_waveform = get_sine_cycles(sampling_info, sine_args)

    # 定义数据导出函数
    def export_data(ai_waveform, chunks_num):
        print(f"导出第 {chunks_num} 段数据")

    # 创建高性能同步 AI/AO 对象
    sync_io = HiPerfCSSIO(
        ai_channel="PXI1Slot2/ai0",
        ao_channel="PXI1Slot2/ao0",
        output_waveform=output_waveform,
        export_function=export_data
    )

    # 启动任务
    sync_io.start()
    sync_io.enable_export = True  # 开始导出数据

    # 运行一段时间后停止
    time.sleep(10)
    sync_io.stop()
    ```
    """

    # 获取类日志器
    logger = get_logger(f"{__name__}.HiPerfCSSIO")

    def __init__(
        self,
        ai_channel: str,
        ao_channel: str,
        output_waveform: Waveform,
        export_function: Callable[[Waveform, PositiveInt], None],
    ) -> None:
        """
        初始化高性能连续同步 AI/AO 对象

        Args:
            ai_channel: AI 通道名称，例如 "PXI1Slot2/ai0"
            ao_channel: AO 通道名称，例如 "PXI1Slot2/ao0"
            output_waveform: 固定的输出波形，将被循环使用
            export_function: 数据导出函数，接收 (ai_waveform, chunks_num) 参数

        Raises:
            ValueError: 当参数无效时
        """
        # 公有属性
        self.enable_export: bool = False
        self.export_function = export_function

        # 私有属性 - 基本配置
        self._ai_channel = ai_channel
        self._ao_channel = ao_channel
        self._output_waveform = output_waveform
        self._exported_chunks: int = 0  # 在导出的数据中，该值至少为1

        # 私有属性 - 任务和状态管理
        self._ai_task: nidaqmx.Task | None = None
        self._ao_task: nidaqmx.Task | None = None
        self._is_running = False
        self._callback_lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()  # 任务停止事件
        self._data_ready_event = threading.Event()  # 数据就绪事件

        # 获取常用信息
        self._sampling_info = output_waveform.sampling_info
        self._chunk_duration = output_waveform.duration

        # 数据队列
        # collections.deque是简单的双端队列，基础操作可保证线程安全，且性能较好，故使用
        # queue.Queue是更高级的FIFO队列，在多线程安全性上更强，但性能稍差，故暂不使用
        self._ai_queue: deque[dict[str, Any]] = deque()  # AI 数据队列

        logger.info(
            f"HiPerfCSSIO 实例已创建 - AI: {ai_channel}, AO: {ao_channel}, "
            f"输出波形长度: {len(self._output_waveform)} 样本, "
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
        """
        if self._ao_task is None:
            logger.error("AO 任务未创建", exc_info=True)
            raise RuntimeError("AO 任务未创建")

        logger.debug("配置 HiPerfCSSIO AO 任务")

        # 添加 AO 通道
        self._ao_task.ao_channels.add_ao_voltage_chan(  # type: ignore
            self._ao_channel, min_val=-10.0, max_val=10.0
        )

        # 配置时钟源和采样
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

        logger.debug("HiPerfCSSIO AO 任务配置完成")

    def _write_fixed_waveform(self):
        """
        将固定波形写入 AO 缓冲区

        在再生模式下，只需要写入一次，之后硬件会自动循环播放
        """
        try:
            # 将固定波形写入硬件缓冲区
            self._ao_task.write(self._output_waveform, auto_start=False)  # type: ignore

            logger.debug(f"固定波形已写入缓冲区: {len(self._output_waveform)} 样本")

        except Exception as e:
            logger.error(f"固定波形写入失败: {e}", exc_info=True)
            raise

    def _setup_hardware_sync(self):
        """配置硬件同步触发"""
        if self._ai_task is None or self._ao_task is None:
            logger.error("AI 或 AO 任务未创建", exc_info=True)
            raise RuntimeError("AI 或 AO 任务未创建")

        logger.debug("配置硬件同步触发")

        # 从通道名称提取设备名称
        device_name = self._ai_channel.split("/")[0]

        # 尝试多种硬件同步方法
        hardware_sync_success = False

        # 方法1: 使用时序引擎 StartTrigger 终端
        te_terminals = [
            f"/{device_name}/te0/StartTrigger",
            f"/{device_name}/te1/StartTrigger",
            f"/{device_name}/te2/StartTrigger",
            f"/{device_name}/te3/StartTrigger",
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

            # 方法2: 使用 AI 任务的 StartTrigger（实际上无效，仅作为备用方法的示例）
            try:
                ai_start_trigger = f"/{device_name}/ai/StartTrigger"
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
    def output_waveform(self) -> Waveform:
        """获取当前的输出波形"""
        # 可变对象，传出副本
        return self._output_waveform.copy()

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
        logger.debug("HiPerfCSSIO 工作线程已启动")

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
                logger.error(f"HiPerfCSSIO 工作线程处理失败: {e}", exc_info=True)

        logger.debug("HiPerfCSSIO 工作线程已退出")

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
            logger.error(f"HiPerfCSSIO 数据导出处理失败: {e}", exc_info=True)

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
