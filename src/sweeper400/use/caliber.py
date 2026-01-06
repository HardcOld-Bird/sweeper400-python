"""
# 多通道校准模块

模块路径：`sweeper400.use.caliber`

包含用于多通道输出情形下各通道响应函数校准的类和函数。
"""

import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..analyze import (
    CalibData,
    Point2D,
    PointRawData,
    PointTFData,
    PositiveInt,
    SamplingInfo,
    SineArgs,
    SweepData,
    Waveform,
    average_sweep_data,
    calculate_transfer_function,
    filter_sweep_data,
    get_sine_cycles,
    get_sine_multi_chs,
)
from ..logger import get_logger
from ..measure import MultiChasCSIO

# 获取模块日志器
logger = get_logger(__name__)


class Caliber:
    """
    # 多通道校准类

    该类专门用于多通道输出情形下，各个通道响应函数的校准（calibration）。
    核心组件是MultiChasCSIO对象，用于控制NI数据采集卡的多通道同步数据输出和采集。

    ## 主要特性：
        - 使用单频正弦信号进行校准
        - 整个校准流程仅创建一次MultiChasCSIO对象，通过通道状态切换实现多通道测量
        - 支持两种重复测量方式：启动次数和连续chunk数
        - 使用SweepData格式存储数据，便于使用现有数据处理函数
        - 自动应用滤波和平均处理以提高准确度
        - 将校准结果序列化保存到本地磁盘
        - 在极坐标系中可视化校准结果

    ## 校准原理：
        1. 创建一个MultiChasCSIO对象，向所有通道发送相同的单频正弦信号
        2. 通过set_ao_channels_status切换通道，每次只启用一个通道
        3. 对每个通道进行多次启动，每次启动后采集多个连续chunk
        4. 将数据存储为SweepData格式（x坐标=启动次数，y坐标=通道序号）
        5. 使用average_sweep_data、filter_sweep_data和calculate_transfer_function处理数据
        6. 对同一通道的多次启动结果进行平均，得到最终传递函数

    ## 使用示例：
    ```python
    from sweeper400.use.caliber import Caliber
    from sweeper400.analyze import init_sampling_info, init_sine_args

    # 创建采样信息和正弦波参数
    sampling_info = init_sampling_info(171500.0, 85750)  # 采样率171.5kHz, 0.5秒
    sine_args = init_sine_args(frequency=3430.0, amplitude=0.01, phase=0.0)

    # 创建校准对象
    caliber = Caliber(
        ai_channel="PXI2Slot2/ai0",
        ao_channels=(
            "PXI2Slot2/ao0", "PXI2Slot2/ao1",
            "PXI2Slot3/ao0", "PXI2Slot3/ao1",
            "PXI3Slot2/ao0", "PXI3Slot2/ao1",
            "PXI3Slot3/ao0", "PXI3Slot3/ao1"
        ),
        sampling_info=sampling_info,
        sine_args=sine_args
    )

    # 执行校准（2次启动，每次4个chunk）
    # 可选：指定settle_time参数来覆盖默认值
    caliber.calibrate(starts_num=2, chunks_per_start=4)

    # 保存校准结果
    caliber.save_calib_data("calibration_results.pkl")

    # 绘制复平面图（概览模式）
    caliber.plot_transfer_functions(mode="overview")

    # 绘制复平面图（详细模式）
    caliber.plot_transfer_functions(mode="detailed")
    ```

    ## 注意事项：
        - 校准前确保硬件连接正确（传声器和扬声器已正确安装）
        - 建议在安静环境中进行校准以减少噪声干扰
        - 校准频率和幅值应根据实际应用场景选择
        - 该校准方法仅适用于单频单幅值工作点
    """

    # 获取类日志器
    logger = get_logger(f"{__name__}.Caliber")

    def __init__(
        self,
        ai_channel: str,
        ao_channels: tuple[str, ...],
        sampling_info: SamplingInfo,
        sine_args: SineArgs,
        calib_data: str | Path | None = None,
    ) -> None:
        """
        初始化校准对象

        Args:
            ai_channel: AI 通道名称（例如 "PXI2Slot2/ai0"）
            ao_channels: AO 通道名称元组（例如 ("PXI2Slot2/ao0", "PXI2Slot2/ao1", ...)）
            sampling_info: 采样信息，包含采样率和采样点数
            sine_args: 正弦波参数，包含频率、幅值和相位信息
            calib_data: 可选，校准数据文件路径。如果指定，将加载该文件并使用
                       get_sine_multi_chs生成已补偿的多通道波形作为输出波形

        Raises:
            ValueError: 当参数无效时
            FileNotFoundError: 当calib_data文件不存在时
            RuntimeError: 当calib_data文件加载失败时
        """
        # 验证参数
        if not ai_channel:
            raise ValueError("AI 通道名称不能为空")
        if not ao_channels:
            raise ValueError("AO 通道列表不能为空")

        # 保存配置参数
        self._ai_channel = ai_channel
        self._ao_channels = ao_channels
        self._sampling_info = sampling_info
        self._sine_args = sine_args

        # 生成输出波形
        if calib_data is not None:
            # 加载校准数据文件
            calib_data_path = Path(calib_data)
            if not calib_data_path.exists():
                raise FileNotFoundError(f"校准数据文件不存在: {calib_data_path}")

            try:
                with open(calib_data_path, "rb") as f:
                    loaded_calib_data: CalibData = pickle.load(f)
                logger.info(f"成功加载校准数据文件: {calib_data_path}")
            except Exception as e:
                logger.error(f"加载校准数据文件失败: {e}", exc_info=True)
                raise RuntimeError(f"加载校准数据文件失败: {e}") from e

            # 验证校准数据与当前配置的一致性
            if loaded_calib_data["ao_channels"] != ao_channels:
                logger.warning(
                    f"校准数据中的AO通道({loaded_calib_data['ao_channels']}) "
                    f"与当前配置({ao_channels})不一致"
                )

            # 使用get_sine_multi_chs生成多通道补偿波形
            self._output_waveform = get_sine_multi_chs(
                sampling_info=sampling_info,
                sine_args=sine_args,
                calib_data=loaded_calib_data,
            )
            logger.info(
                f"使用校准数据生成多通道补偿波形，shape={self._output_waveform.shape}"
            )
        else:
            # 使用get_sine_cycles生成单通道波形
            self._output_waveform = get_sine_cycles(sampling_info, sine_args)
            logger.info("使用get_sine_cycles生成单通道波形")

        # 计算默认的稳定等待时间（chunk时长 + 0.1秒）
        chunk_duration = self._output_waveform.duration
        self._default_settle_time = chunk_duration + 0.1

        # 校准数据存储（使用SweepData格式）
        self._result_raw_sweep_data: SweepData | None = None

        # 处理后的传递函数数据（未平均，保留所有启动次数的结果）
        self._result_raw_tf_list: list[PointTFData] | None = None

        # 最终传递函数结果（每个通道一个PointTFData，已平均）
        self._result_final_tf_list: list[PointTFData] | None = None

        logger.info(
            f"Caliber 实例已创建 - "
            f"AI: {ai_channel}, "
            f"AO通道数: {len(ao_channels)}, "
            f"频率: {sine_args['frequency']}Hz, "
            f"幅值: {sine_args['amplitude']}V, "
            f"采样率: {sampling_info['sampling_rate']}Hz, "
            f"默认稳定时间: {self._default_settle_time:.3f}s, "
            f"使用校准数据: {calib_data is not None}"
        )

    def _export_function(self, ai_waveform: Waveform, chunks_num: PositiveInt) -> None:
        """
        数据导出回调函数

        将采集到的AI波形添加到当前测量点的数据列表中。
        使用chunks_num参数精确控制采集的chunk数量。

        Args:
            ai_waveform: 采集到的AI波形
            chunks_num: 数据块编号（从1开始）
        """
        # 将数据添加到当前测量点
        if (
            self._result_raw_sweep_data is not None
            and len(self._result_raw_sweep_data["ai_data_list"]) > 0
        ):
            current_point = self._result_raw_sweep_data["ai_data_list"][-1]
            current_point["ai_data"].append(ai_waveform)
            logger.debug(
                f"采集到第 {chunks_num} 段数据，当前点: {current_point['position']}"
            )

            # 检查是否已采集到目标数量的chunk
            if (
                hasattr(self, "_target_chunks")
                and len(current_point["ai_data"]) >= self._target_chunks
            ):
                # 设置标志，通知主循环停止等待
                self._chunk_collection_complete = True

    def calibrate(
        self,
        starts_num: int = 2,
        chunks_per_start: int = 3,
        apply_filter: bool = True,
        lowcut: float = 100.0,
        highcut: float = 20000.0,
        result_folder: str | Path | None = None,
        settle_time: float | None = None,
    ) -> None:
        """
        执行校准流程

        使用新的重复测量机制：对每个通道进行多次启动，每次启动采集多个连续chunk。
        数据存储为SweepData格式，x坐标为启动次数，y坐标为通道序号。

        Args:
            starts_num: 启动次数，默认为2
            chunks_per_start: 每次启动采集的连续chunk数，默认为3
            apply_filter: 是否应用滤波，默认为True
            lowcut: 滤波器低频截止频率（Hz），默认为100.0
            highcut: 滤波器高频截止频率（Hz），默认为20000.0
            result_folder: 可选，结果保存文件夹路径。如果指定，将自动保存raw sweep data、
                          calib_data以及overview和detailed两种模式的绘图
            settle_time: 通道切换后的稳定等待时间（秒）。如果为None，则使用初始化时
                        计算的默认值（chunk时长 + 0.1秒）

        Raises:
            ValueError: 当参数无效时
        """
        if starts_num < 1:
            raise ValueError("启动次数必须至少为1")
        if chunks_per_start < 1:
            raise ValueError("每次启动的chunk数必须至少为1")

        # 确定使用的稳定等待时间
        actual_settle_time = settle_time if settle_time is not None else self._default_settle_time

        logger.info(
            f"开始校准流程 - "
            f"通道数: {len(self._ao_channels)}, "
            f"启动次数: {starts_num}, "
            f"每次启动chunk数: {chunks_per_start}, "
            f"稳定时间: {actual_settle_time:.3f}s"
        )

        # 初始化SweepData（显式类型标注）
        self._result_raw_sweep_data: SweepData = {
            "ai_data_list": [],
            "ao_data": self._output_waveform,
        }

        # 创建MultiChasCSIO对象（整个校准流程只创建一次）
        sync_io = MultiChasCSIO(
            ai_channel=self._ai_channel,
            ao_channels=self._ao_channels,
            output_waveform=self._output_waveform,
            export_function=self._export_function,
        )

        try:
            # 启动任务
            sync_io.start()
            logger.info("MultiChasCSIO任务已启动")

            # 等待系统初始化稳定
            time.sleep(2.0)

            # 执行测量循环
            for start_idx in range(starts_num):
                for channel_idx in range(len(self._ao_channels)):
                    # 创建虚拟点坐标（x=启动次数，y=通道序号）
                    virtual_position = Point2D(
                        x=float(start_idx + 1), y=float(channel_idx + 1)
                    )

                    logger.info(
                        f"测量点 ({start_idx + 1}, {channel_idx + 1}): "
                        f"启动 {start_idx + 1}/{starts_num}, "
                        f"通道 {channel_idx} ({self._ao_channels[channel_idx]})"
                    )

                    # 创建新的测量点数据
                    point_data: PointRawData = {
                        "position": virtual_position,
                        "ai_data": [],
                    }
                    self._result_raw_sweep_data["ai_data_list"].append(point_data)

                    # 设置通道状态（只启用当前通道）
                    channels_status = tuple(
                        i == channel_idx for i in range(len(self._ao_channels))
                    )
                    sync_io.set_ao_channels_status(channels_status)

                    # 等待通道切换稳定
                    logger.debug(f"等待通道切换稳定 {actual_settle_time:.3f}s")
                    time.sleep(actual_settle_time)

                    # 设置目标chunk数量和完成标志
                    self._target_chunks = chunks_per_start
                    self._chunk_collection_complete = False

                    # 启用数据导出
                    sync_io.enable_export = True

                    # 等待采集指定数量的chunk（使用轮询方式）
                    chunk_duration = self._output_waveform.duration
                    max_wait_time = (
                        chunk_duration * chunks_per_start * 2.0
                    )  # 最大等待时间
                    poll_interval = 0.05  # 轮询间隔50ms
                    elapsed_time = 0.0

                    logger.debug(f"开始采集 {chunks_per_start} 个chunk")
                    while (
                        not self._chunk_collection_complete
                        and elapsed_time < max_wait_time
                    ):
                        time.sleep(poll_interval)
                        elapsed_time += poll_interval

                    # 禁用数据导出
                    sync_io.enable_export = False

                    # 检查采集到的数据数量
                    collected_chunks = len(point_data["ai_data"])
                    logger.info(
                        f"点 ({start_idx + 1}, {channel_idx + 1}) "
                        f"采集完成，共 {collected_chunks} 个chunk"
                    )

                    # 验证采集到的chunk数量
                    if collected_chunks != chunks_per_start:
                        logger.warning(
                            f"预期采集 {chunks_per_start} 个chunk，"
                            f"实际采集 {collected_chunks} 个chunk"
                        )

                    # 短暂等待
                    time.sleep(0.1)

        finally:
            # 停止任务
            sync_io.stop()
            logger.info("MultiChasCSIO任务已停止")

        # 数据处理
        logger.info("开始处理校准数据...")

        # 1. 对每个点的多个chunk进行平均（先平均，减少数据量）
        logger.info("对每个点的多个chunk进行平均")
        averaged_data = average_sweep_data(self._result_raw_sweep_data)

        # 2. 应用滤波（如果需要）（后滤波，符合信号处理最佳实践）
        if apply_filter:
            logger.info(f"应用带通滤波器: {lowcut}Hz - {highcut}Hz")
            filtered_data = filter_sweep_data(
                averaged_data,
                lowcut=lowcut,
                highcut=highcut,
            )
        else:
            filtered_data = averaged_data

        # 3. 计算传递函数
        logger.info("计算传递函数")
        self._result_raw_tf_list = calculate_transfer_function(filtered_data)

        # 4. 对同一通道的多次启动结果进行平均
        logger.info("对同一通道的多次启动结果进行平均")
        self._result_final_tf_list = []

        for channel_idx in range(len(self._ao_channels)):
            # 找到所有属于该通道的测量点
            channel_tf_list = [
                tf_data
                for tf_data in self._result_raw_tf_list
                if tf_data["position"].y == channel_idx + 1
            ]

            if len(channel_tf_list) > 0:
                # 计算平均幅值比和平均相位差
                avg_amp_ratio = np.mean([tf["amp_ratio"] for tf in channel_tf_list])
                avg_phase_shift = np.mean([tf["phase_shift"] for tf in channel_tf_list])

                # 创建虚拟位置（x=0, y=通道序号）
                final_position = Point2D(x=0.0, y=float(channel_idx + 1))

                # 创建PointTFData
                final_tf_data: PointTFData = {
                    "position": final_position,
                    "amp_ratio": avg_amp_ratio,
                    "phase_shift": avg_phase_shift,
                }
                self._result_final_tf_list.append(final_tf_data)

                logger.info(
                    f"通道 {channel_idx} ({self._ao_channels[channel_idx]}) "
                    f"最终传递函数: 幅值比={avg_amp_ratio:.6f}, 相位差={avg_phase_shift:.6f}rad"
                )
            else:
                logger.warning(f"通道 {channel_idx} 没有有效测量数据")
                # 创建默认值
                final_position = Point2D(x=0.0, y=float(channel_idx + 1))
                final_tf_data: PointTFData = {
                    "position": final_position,
                    "amp_ratio": 1.0,
                    "phase_shift": 0.0,
                }
                self._result_final_tf_list.append(final_tf_data)

        logger.info("校准流程完成")

        # 如果指定了result_folder，自动保存结果
        if result_folder is not None:
            logger.info(f"开始保存校准结果到文件夹: {result_folder}")
            result_path = Path(result_folder)
            result_path.mkdir(parents=True, exist_ok=True)

            # 保存raw sweep data
            sweep_data_path = result_path / "raw_sweep_data.pkl"
            self.save_sweep_data(sweep_data_path)
            logger.info(f"已保存raw sweep data到: {sweep_data_path}")

            # 保存calib_data
            calib_data_path = result_path / "calib_data.pkl"
            self.save_calib_data(calib_data_path)
            logger.info(f"已保存calib_data到: {calib_data_path}")

            # 保存overview模式绘图
            overview_plot_path = result_path / "transfer_function_overview.png"
            self.plot_transfer_functions(mode="overview", save_path=overview_plot_path)
            logger.info(f"已保存overview模式绘图到: {overview_plot_path}")

            # 保存detailed模式绘图
            detailed_plot_path = result_path / "transfer_function_detailed.png"
            self.plot_transfer_functions(mode="detailed", save_path=detailed_plot_path)
            logger.info(f"已保存detailed模式绘图到: {detailed_plot_path}")

            logger.info(f"所有校准结果已保存到: {result_folder}")

    def ex_calibrate(
        self,
        ex_starts_num: int,
        starts_num: int = 2,
        chunks_per_start: int = 3,
        apply_filter: bool = True,
        lowcut: float = 100.0,
        highcut: float = 20000.0,
        result_folder: str | Path | None = None,
        settle_time: float | None = None,
    ) -> None:
        """
        执行扩展校准流程（多次独立校准并平均）

        该方法会多次调用calibrate方法进行独立校准，每次校准都经历完整的任务创建和删除流程。
        每次校准完成后保存CalibData文件，最后读取所有文件并对每个通道的数据进行平均，
        生成最终的CalibData文件和绘图。

        Args:
            ex_starts_num: 扩展启动次数，即独立校准的次数
            starts_num: 每次校准的启动次数，默认为2
            chunks_per_start: 每次启动采集的连续chunk数，默认为3
            apply_filter: 是否应用滤波，默认为True
            lowcut: 滤波器低频截止频率（Hz），默认为100.0
            highcut: 滤波器高频截止频率（Hz），默认为20000.0
            result_folder: 必需，结果保存文件夹路径。最终将保存一个平均后的CalibData文件、
                          overview模式绘图，以及每次校准的detailed模式绘图
            settle_time: 通道切换后的稳定等待时间（秒）。如果为None，则使用初始化时
                        计算的默认值（chunk时长 + 0.1秒）

        Raises:
            ValueError: 当参数无效时
            RuntimeError: 当result_folder未指定时
        """
        if ex_starts_num < 1:
            raise ValueError("扩展启动次数必须至少为1")
        if result_folder is None:
            raise RuntimeError("ex_calibrate方法必须指定result_folder参数")

        logger.info(
            f"开始扩展校准流程 - "
            f"扩展启动次数: {ex_starts_num}, "
            f"每次校准的启动次数: {starts_num}, "
            f"每次启动chunk数: {chunks_per_start}"
        )

        # 创建结果文件夹
        result_path = Path(result_folder)
        result_path.mkdir(parents=True, exist_ok=True)

        # 创建临时文件夹用于存储每次校准的结果
        temp_folder = result_path / "temp_calib_data"
        temp_folder.mkdir(parents=True, exist_ok=True)

        # 存储每次校准的CalibData文件路径
        calib_data_files: list[Path] = []

        # 执行多次独立校准
        for ex_idx in range(ex_starts_num):
            logger.info(f"开始第 {ex_idx + 1}/{ex_starts_num} 次独立校准")

            # 调用calibrate方法（不设置result_folder）
            self.calibrate(
                starts_num=starts_num,
                chunks_per_start=chunks_per_start,
                apply_filter=apply_filter,
                lowcut=lowcut,
                highcut=highcut,
                result_folder=None,  # 不保存中间结果
                settle_time=settle_time,
            )

            # 保存这次校准的CalibData
            calib_data_path = temp_folder / f"calib_data_{ex_idx + 1}.pkl"
            self.save_calib_data(calib_data_path)
            calib_data_files.append(calib_data_path)
            logger.info(f"第 {ex_idx + 1} 次校准完成，已保存到: {calib_data_path}")

            # 保存这次校准的 detailed 模式绘图
            detailed_plot_path = result_path / f"transfer_function_detailed_{ex_idx + 1}.png"
            self.plot_transfer_functions(mode="detailed", save_path=detailed_plot_path)
            logger.info(f"已保存第 {ex_idx + 1} 次校准的detailed模式绘图到: {detailed_plot_path}")

        # 读取所有CalibData文件并进行平均
        logger.info("开始平均所有校准结果...")

        # 加载所有CalibData
        all_calib_data: list[CalibData] = []
        for calib_file in calib_data_files:
            try:
                with open(calib_file, "rb") as f:
                    calib_data: CalibData = pickle.load(f)
                    all_calib_data.append(calib_data)
            except Exception as e:
                logger.error(f"加载校准数据文件失败: {calib_file}, 错误: {e}", exc_info=True)
                raise RuntimeError(f"加载校准数据文件失败: {calib_file}") from e

        # 验证所有CalibData的通道数一致
        channels_num = len(self._ao_channels)
        for idx, calib_data in enumerate(all_calib_data):
            if len(calib_data["tf_list"]) != channels_num:
                raise RuntimeError(
                    f"第 {idx + 1} 次校准的通道数({len(calib_data['tf_list'])}) "
                    f"与预期({channels_num})不一致"
                )

        # 对每个通道的数据进行平均
        averaged_tf_list: list[PointTFData] = []
        for channel_idx in range(channels_num):
            # 收集该通道在所有校准中的幅值比和相位差
            amp_ratios = [
                calib_data["tf_list"][channel_idx]["amp_ratio"]
                for calib_data in all_calib_data
            ]
            phase_shifts = [
                calib_data["tf_list"][channel_idx]["phase_shift"]
                for calib_data in all_calib_data
            ]

            # 计算平均值
            avg_amp_ratio = np.mean(amp_ratios)
            avg_phase_shift = np.mean(phase_shifts)

            # 创建平均后的PointTFData
            final_position = Point2D(x=0.0, y=float(channel_idx + 1))
            averaged_tf_data: PointTFData = {
                "position": final_position,
                "amp_ratio": avg_amp_ratio,
                "phase_shift": avg_phase_shift,
            }
            averaged_tf_list.append(averaged_tf_data)

            logger.info(
                f"通道 {channel_idx} ({self._ao_channels[channel_idx]}) "
                f"平均传递函数: 幅值比={avg_amp_ratio:.6f}, 相位差={avg_phase_shift:.6f}rad"
            )

        # 创建最终的CalibData
        final_calib_data: CalibData = {
            "tf_list": averaged_tf_list,
            "ao_channels": self._ao_channels,
            "ai_channel": self._ai_channel,
            "sampling_info": self._sampling_info,
            "sine_args": self._sine_args,
        }

        # 更新内部状态为平均后的结果（这样 plot_transfer_functions 的 overview 模式能使用真正的最终数据）
        self._result_final_tf_list = averaged_tf_list

        # 保存最终的CalibData
        final_calib_data_path = result_path / "calib_data.pkl"
        try:
            with open(final_calib_data_path, "wb") as f:
                pickle.dump(final_calib_data, f)
            logger.info(f"最终平均CalibData已保存到: {final_calib_data_path}")
        except Exception as e:
            logger.error(f"保存最终CalibData失败: {e}", exc_info=True)
            raise OSError(f"保存最终CalibData失败: {e}") from e

        # 绘制overview模式绘图（使用最终平均后的数据）
        overview_plot_path = result_path / "transfer_function_overview.png"
        self.plot_transfer_functions(mode="overview", save_path=overview_plot_path)
        logger.info(f"已保存overview模式绘图到: {overview_plot_path}")

        # 删除临时文件夹及其内容
        import shutil

        try:
            shutil.rmtree(temp_folder)
            logger.info(f"已删除临时文件夹: {temp_folder}")
        except Exception as e:
            logger.warning(f"删除临时文件夹失败: {e}")

        logger.info(f"扩展校准流程完成，所有结果已保存到: {result_folder}")

    def save_calib_data(self, file_path: str | Path) -> None:
        """
        保存校准结果到本地文件

        使用pickle序列化将最终传递函数保存到磁盘。

        Args:
            file_path: 保存文件的路径（支持字符串或Path对象）

        Raises:
            RuntimeError: 当尚未执行校准时
            IOError: 当文件保存失败时
        """
        if self._result_final_tf_list is None:
            raise RuntimeError("尚未执行校准，无法保存结果")

        # 转换为Path对象
        save_path = Path(file_path)

        # 确保父目录存在
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 准备保存的数据（只保存最终平均后的传递函数）
        calib_data: CalibData = {
            "tf_list": self._result_final_tf_list,
            "ao_channels": self._ao_channels,
            "ai_channel": self._ai_channel,
            "sampling_info": self._sampling_info,
            "sine_args": self._sine_args,
        }

        try:
            with open(save_path, "wb") as f:
                pickle.dump(calib_data, f)
            logger.info(f"校准结果已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存校准结果失败: {e}", exc_info=True)
            raise OSError(f"保存校准结果失败: {e}") from e

    def save_sweep_data(self, file_path: str | Path) -> None:
        """
        保存原始SweepData（未平均/滤波处理）到本地文件

        Args:
            file_path: 保存文件的路径（支持字符串或Path对象）

        Raises:
            RuntimeError: 当尚未执行校准时
            IOError: 当文件保存失败时
        """
        if self._result_raw_sweep_data is None:
            raise RuntimeError("尚未执行校准，无法保存数据")

        # 转换为Path对象
        save_path = Path(file_path)

        # 确保父目录存在
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(save_path, "wb") as f:
                pickle.dump(self._result_raw_sweep_data, f)
            logger.info(f"校准SweepData已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存SweepData失败: {e}", exc_info=True)
            raise OSError(f"保存SweepData失败: {e}") from e

    def plot_transfer_functions(
        self,
        mode: str = "overview",
        save_path: str | Path | None = None,
    ) -> None:
        """
        绘制传递函数

        支持两种模式：
        - "overview": 在极坐标系中每个通道绘制一个点（平均后的结果）
        - "detailed": 在直角坐标系中每次启动绘制一个点（显示所有原始测量）

        Args:
            mode: 绘图模式，"overview"或"detailed"，默认为"overview"
            save_path: 可选，图像保存路径。如果为None，则只显示不保存

        Raises:
            RuntimeError: 当尚未执行校准时
            ValueError: 当mode参数无效时
        """
        if self._result_raw_tf_list is None or self._result_final_tf_list is None:
            raise RuntimeError("尚未执行校准，无法绘制结果")

        if mode not in ("overview", "detailed"):
            raise ValueError("mode参数必须为'overview'或'detailed'")

        logger.info(f"开始绘制传递函数图（模式: {mode}）")

        if mode == "overview":
            # 概览模式：极坐标图
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="polar")

            # 每个通道一个点
            angles = []
            magnitudes = []
            channel_indices = []

            for tf_data in self._result_final_tf_list:
                # 从PointTFData中提取幅值比和相位差
                magnitude = tf_data["amp_ratio"]
                angle = tf_data["phase_shift"]
                channel_idx = int(tf_data["position"].y) - 1

                angles.append(angle)
                magnitudes.append(magnitude)
                channel_indices.append(channel_idx)

            # 绘制散点
            scatter = ax.scatter(
                angles,
                magnitudes,
                c=channel_indices,
                cmap="viridis",
                s=150,
                alpha=0.8,
                edgecolors="black",
                linewidths=2,
                zorder=3,
            )

            # 添加简洁的数字标注
            for angle, magnitude, channel_idx in zip(
                angles, magnitudes, channel_indices, strict=True
            ):
                ax.annotate(
                    f"{channel_idx}",
                    xy=(angle, magnitude),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                    alpha=0.8,
                    zorder=4,
                )

            # 设置极坐标图样式
            ax.set_theta_zero_location("E")  # 0度在右侧
            ax.set_theta_direction(1)  # 逆时针为正
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)

            # 设置标题
            ax.set_title(
                f"传递函数极坐标分布（概览模式）\n"
                f"频率: {self._sine_args['frequency']}Hz, "
                f"幅值: {self._sine_args['amplitude']}V",
                fontsize=14,
                pad=20,
            )

            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label("通道索引", fontsize=10)

            # 在 colorbar 上标注通道名称
            cbar.set_ticks(range(len(self._ao_channels)))
            tick_labels = [
                f"{idx}: {self._ao_channels[idx]}"
                for idx in range(len(self._ao_channels))
            ]
            cbar.set_ticklabels(tick_labels)
            cbar.ax.tick_params(labelsize=8)

        else:  # detailed mode
            # 详细模式：直角坐标图（复平面）
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111)

            # 每次启动一个点
            real_parts = []
            imag_parts = []
            channel_indices = []
            start_indices = []  # 记录启动次数

            for tf_data in self._result_raw_tf_list:
                # 从虚拟坐标中提取通道索引和启动次数
                channel_idx = int(tf_data["position"].y) - 1
                start_idx = int(tf_data["position"].x) - 1

                # 从PointTFData中提取幅值比和相位差，转换为复数
                magnitude = tf_data["amp_ratio"]
                angle = tf_data["phase_shift"]

                # 计算实部和虚部
                real_part = magnitude * np.cos(angle)
                imag_part = magnitude * np.sin(angle)

                real_parts.append(real_part)
                imag_parts.append(imag_part)
                channel_indices.append(channel_idx)
                start_indices.append(start_idx)

            # 绘制散点
            scatter = ax.scatter(
                real_parts,
                imag_parts,
                c=channel_indices,
                cmap="viridis",
                s=120,
                alpha=0.7,
                edgecolors="black",
                linewidths=1.5,
                zorder=3,
            )

            # 只对每个通道的第一次启动添加详细的黄色标签框
            labeled_channels = set()  # 记录已添加标签的通道
            for real_part, imag_part, channel_idx, start_idx in zip(
                real_parts, imag_parts, channel_indices, start_indices, strict=True
            ):
                # 只为每个通道的第一次启动（start_idx == 0）添加标签
                if start_idx == 0 and channel_idx not in labeled_channels:
                    ax.annotate(
                        f"Ch{channel_idx}\n{self._ao_channels[channel_idx]}",
                        xy=(real_part, imag_part),
                        xytext=(15, 15),
                        textcoords="offset points",
                        fontsize=8,
                        bbox={
                            "boxstyle": "round,pad=0.4",
                            "facecolor": "yellow",
                            "alpha": 0.6,
                        },
                        arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=0.2"},
                        zorder=4,
                    )
                    labeled_channels.add(channel_idx)

            # 设置直角坐标图样式
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
            ax.set_xlabel("实部 (Real)", fontsize=12)
            ax.set_ylabel("虚部 (Imaginary)", fontsize=12)
            ax.set_aspect("equal", adjustable="datalim")  # 保持纵横比

            # 设置标题
            ax.set_title(
                f"传递函数复平面分布（详细模式）\n"
                f"频率: {self._sine_args['frequency']}Hz, "
                f"幅值: {self._sine_args['amplitude']}V",
                fontsize=14,
                pad=20,
            )

            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
            cbar.set_label("通道索引", fontsize=10)

        plt.tight_layout()

        # 保存或显示
        if save_path is not None:
            save_path_obj = Path(save_path)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path_obj, dpi=300, bbox_inches="tight")
            logger.info(f"传递函数图已保存到: {save_path_obj}")

        plt.show()
        logger.info("传递函数极坐标图绘制完成")

    @property
    def result_raw_sweep_data(self) -> SweepData | None:
        """
        获取校准过程中采集的原始SweepData

        Returns:
            SweepData对象，如果尚未校准则返回None
        """
        if self._result_raw_sweep_data is None:
            return None
        # 深拷贝以避免外部修改
        import copy

        return copy.deepcopy(self._result_raw_sweep_data)

    @property
    def result_raw_tf_list(self) -> list[PointTFData] | None:
        """
        获取所有测量点的传递函数数据（未平均）

        Returns:
            传递函数数据列表，如果尚未校准则返回None
        """
        if self._result_raw_tf_list is None:
            return None
        return self._result_raw_tf_list.copy()

    @property
    def result_final_tf_list(self) -> list[PointTFData] | None:
        """
        获取最终的传递函数（每个通道平均后的结果）

        Returns:
            传递函数列表（每个通道一个PointTFData），如果尚未校准则返回None
        """
        if self._result_final_tf_list is None:
            return None
        return self._result_final_tf_list.copy()

    @property
    def ao_channels(self) -> tuple[str, ...]:
        """
        获取AO通道列表

        Returns:
            AO通道名称元组
        """
        return self._ao_channels

    @property
    def ai_channel(self) -> str:
        """
        获取AI通道名称

        Returns:
            AI通道名称
        """
        return self._ai_channel
