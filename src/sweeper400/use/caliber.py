"""
# 多通道校准模块

模块路径：`sweeper400.use.caliber`

包含用于多通道输出情形下各通道响应函数校准的类和函数。
"""

import pickle
import time
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from ..analyze import (
    CalibData,
    Point2D,
    PointCompData,
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
    get_sine_multi_ch,
)
from ..logger import get_logger
from ..measure import MultiChasCSIO

# 获取模块日志器
logger = get_logger(__name__)


class CaliberOctopus:
    """
    # 多通道校准类（章鱼模式）

    该类专门用于多通道输出情形下，各个通道响应函数的校准（calibration）。
    核心组件是MultiChasCSIO对象，用于控制NI数据采集卡的多通道同步数据输出和采集。
    "Octopus"（章鱼）命名表示该类使用章鱼型波导，测量单个AI通道与多个AO通道的传递函数。

    ## 主要特性：
        - 使用单频正弦信号进行校准
        - 通过多次独立校准并平均来提高精度
        - 每次独立校准创建一次MultiChasCSIO对象，通过动态更换波形实现多通道测量
        - 使用SweepData格式存储数据，便于使用现有数据处理函数
        - 自动应用滤波和平均处理以提高准确度
        - 将校准结果序列化保存到本地磁盘
        - 支持极坐标可视化模式
        - 支持基于已有校准数据进行校准验证（通过calib_data参数）

    ## 校准原理：
        1. 多次执行独立校准（每次创建和销毁MultiChasCSIO对象）
        2. 每次独立校准中：
           a. 创建一个MultiChasCSIO对象，向所有通道发送相同的单频正弦信号
           b. 通过动态更换波形，每次只向一个通道发送信号（其他通道输出零）
           c. 对每个通道采集多个连续chunk
           d. 将数据存储为SweepData格式（x坐标=1，y坐标=通道序号）
           e. 使用average_sweep_data、filter_sweep_data和calculate_transfer_function处理数据
           f. 提取每个通道的绝对幅值比和相位差
           g. 计算所有通道的平均值，然后计算每个通道相对于平均值的相对幅值比和时间延迟
        3. 对所有独立校准的结果进行平均，得到最终的相对幅值比和时间延迟
        4. 保存相对幅值比、时间延迟以及绝对幅值比的平均值

    ## 使用示例：
    ```python
    from sweeper400.use.caliber import CaliberOctopus
    from sweeper400.analyze import init_sampling_info, init_sine_args

    # 创建采样信息和正弦波参数
    sampling_info = init_sampling_info(171500.0, 85750)  # 采样率171.5kHz, 0.5秒
    sine_args = init_sine_args(frequency=3430.0, amplitude=0.01, phase=0.0)

    # 创建校准对象
    caliber = CaliberOctopus(
        ai_channels=("PXI1Slot2/ai0",),
        ao_channels=(
            "PXI2Slot2/ao0", "PXI2Slot2/ao1",
            "PXI2Slot3/ao0", "PXI2Slot3/ao1",
            "PXI3Slot2/ao0", "PXI3Slot2/ao1",
            "PXI3Slot3/ao0", "PXI3Slot3/ao1"
        ),
        sampling_info=sampling_info,
        sine_args=sine_args
    )

    # 执行校准（10次独立校准，每次4个chunk）
    # 可选：指定settle_time参数来覆盖默认值
    caliber.calibrate(starts_num=10, chunks_per_start=4)

    # 校准结果会自动保存到默认路径
    # 也可以手动保存到指定路径
    caliber.save_calib_data("calibration_results.pkl")

    # 绘制极坐标图
    caliber.plot_transfer_functions(mode="polar")

    # 校准验证：使用已有校准数据进行二次校准
    caliber_verify = CaliberOctopus(
        ai_channels=("PXI1Slot2/ai0",),
        ao_channels=(...),  # 与上面相同
        sampling_info=sampling_info,
        sine_args=sine_args,
        calib_data="calibration_results.pkl"  # 使用之前的校准结果
    )
    caliber_verify.calibrate(starts_num=10, chunks_per_start=4)
    # 如果校准有效，验证结果应显示所有通道响应一致
    ```

    ## 注意事项：
        - 校准前确保硬件连接正确（传声器和扬声器已正确安装）
        - 建议在安静环境中进行校准以减少噪声干扰
        - 校准频率和幅值应根据实际应用场景选择
        - 该校准方法仅适用于单频单幅值工作点
        - 使用calib_data参数进行校准验证时，应使用相同的通道配置和测量参数
    """

    # 获取类日志器
    logger = get_logger(f"{__name__}.CaliberOctopus")

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
            ai_channel: AI 通道名称字符串（例如 "PXI2Slot2/ai0"）
            ao_channels: AO 通道名称元组（例如 ("PXI2Slot2/ao0", "PXI2Slot2/ao1", ...)）
            sampling_info: 采样信息，包含采样率和采样点数
            sine_args: 正弦波参数，包含频率、幅值和相位信息
            calib_data: 可选，校准数据文件路径。如果指定，将加载该文件并使用
                       get_sine_multi_ch生成已补偿的多通道波形。在校准测量时，
                       每个通道将发送其对应的补偿信号（而非相同的未补偿信号），
                       从而验证校准补偿的有效性

        Raises:
            ValueError: 当参数无效时
            FileNotFoundError: 当calib_data文件不存在时
            RuntimeError: 当calib_data文件加载失败时
        """
        # 验证参数
        if not ai_channel:
            raise ValueError("AI 通道不能为空")
        if not ao_channels:
            raise ValueError("AO 通道列表不能为空")

        # 保存配置参数
        self._ai_channel = ai_channel  # 校准时通常只使用一个AI通道
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
            self._output_waveform = get_sine_multi_ch(
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

        # 处理后的传递函数数据（绝对传递函数，未平均，保留所有启动次数的结果）
        self._result_raw_tf_list: list[PointTFData] | None = None

        # 所有starts的补偿数据（未平均，保留所有启动次数的结果）
        self._result_raw_comp_list: list[PointCompData] | None = None

        # 最终补偿数据结果（每个通道一个PointCompData，已平均）
        self._result_final_comp_list: list[PointCompData] | None = None

        # 所有通道传递函数幅值比的平均值
        self._amp_ratio_mean: float | None = None

        logger.info(
            f"CaliberOctopus 实例已创建 - "
            f"AI: {ai_channel}, "
            f"AO通道数: {len(ao_channels)}, "
            f"频率: {sine_args['frequency']}Hz, "
            f"幅值: {sine_args['amplitude']}V, "
            f"采样率: {sampling_info['sampling_rate']}Hz, "
            f"默认稳定时间: {self._default_settle_time:.3f}s, "
            f"使用校准数据: {calib_data is not None}"
        )

    def _create_single_channel_waveform(self, active_channel_idx: int) -> Waveform:
        """
        创建只启用一个通道的多通道波形

        生成一个多通道波形，其中只有指定通道输出信号，其他通道输出零。
        如果初始化时提供了calib_data，则使用已补偿的波形数据；否则使用未补偿的波形。

        Args:
            active_channel_idx: 要启用的通道索引（0-based）

        Returns:
            多通道波形对象，只有指定通道有信号

        Raises:
            ValueError: 当通道索引无效时
        """
        if active_channel_idx < 0 or active_channel_idx >= len(self._ao_channels):
            raise ValueError(
                f"通道索引 {active_channel_idx} 超出范围 [0, {len(self._ao_channels) - 1}]"
            )

        # 创建多通道波形数组
        num_channels = len(self._ao_channels)

        # 检查self._output_waveform是否为多通道波形（已应用校准补偿）
        if self._output_waveform.ndim == 2:
            # 多通道情况：使用已补偿的波形数据
            num_samples = self._output_waveform.samples_num
            multi_channel_data = np.zeros((num_channels, num_samples), dtype=np.float64)
            # 只在指定通道填充已补偿的信号
            multi_channel_data[active_channel_idx, :] = self._output_waveform[
                active_channel_idx, :
            ]
            logger.debug(
                f"使用已补偿的多通道波形数据创建单通道激活波形（通道 {active_channel_idx}）"
            )
        else:
            # 单通道情况：使用未补偿的波形（所有通道使用相同的信号）
            num_samples = self._output_waveform.samples_num
            multi_channel_data = np.zeros((num_channels, num_samples), dtype=np.float64)
            # 只在指定通道填充信号
            multi_channel_data[active_channel_idx, :] = self._output_waveform
            logger.debug(
                f"使用未补偿的单通道波形数据创建单通道激活波形（通道 {active_channel_idx}）"
            )

        # 创建Waveform对象
        waveform = Waveform(
            input_array=multi_channel_data,
            sampling_rate=self._sampling_info["sampling_rate"],
            timestamp=self._output_waveform.timestamp,
            id=self._output_waveform.id,
            sine_args=self._sine_args,
        )

        return waveform

    def _export_function(
        self,
        ai_waveform: Waveform,
        ao_static_waveform: Waveform,
        ao_feedback_waveform: Waveform | None,
        chunks_num: PositiveInt,
    ) -> None:
        """
        数据导出回调函数

        将采集到的AI波形添加到当前测量点的数据列表中。
        使用chunks_num参数精确控制采集的chunk数量。

        Args:
            ai_waveform: 采集到的AI波形
            ao_static_waveform: 当前的静态输出波形
            ao_feedback_waveform: 当前的反馈输出波形（如果没有反馈通道则为None）
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

    def _feedback_function(self, ai_waveform: Waveform) -> Waveform:
        """
        反馈函数（CaliberOctopus不使用反馈功能）

        返回全0静音波形，因为校准过程不需要反馈控制。

        Args:
            ai_waveform: AI波形数据

        Returns:
            全0静音波形
        """
        # 返回全0静音波形（与ai_waveform相同shape）
        import numpy as np

        silence_data = np.zeros_like(ai_waveform)
        return Waveform(
            input_array=silence_data,
            sampling_rate=ai_waveform.sampling_rate,
        )

    def _single_calibrate(
        self,
        chunks_per_start: int = 3,
        apply_filter: bool = True,
        lowcut: float = 100.0,
        highcut: float = 20000.0,
        settle_time: float | None = None,
    ) -> None:
        """
        执行单次校准流程（内部方法）

        对每个通道采集多个连续chunk。
        数据存储为SweepData格式，x坐标固定为1，y坐标为通道序号。

        校准结果以相对值形式存储：
        - 相对幅值比：相对于所有通道平均值的补偿倍率
        - 时间延迟：相对于所有通道平均相位的时间差（秒）
        - 绝对幅值比平均值：所有通道传递函数幅值比的平均值

        Args:
            chunks_per_start: 每次启动采集的连续chunk数，默认为3
            apply_filter: 是否应用滤波，默认为True
            lowcut: 滤波器低频截止频率（Hz），默认为100.0
            highcut: 滤波器高频截止频率（Hz），默认为20000.0
            settle_time: 通道切换后的稳定等待时间（秒）。如果为None，则使用初始化时
                        计算的默认值（chunk时长 + 0.1秒）

        Raises:
            ValueError: 当参数无效时
        """
        if chunks_per_start < 1:
            raise ValueError("每次启动的chunk数必须至少为1")

        # 确定使用的稳定等待时间
        actual_settle_time = (
            settle_time if settle_time is not None else self._default_settle_time
        )

        logger.info(
            f"开始单次校准流程 - "
            f"通道数: {len(self._ao_channels)}, "
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
            ai_channels=(self._ai_channel,),
            ao_channels_static=self._ao_channels,
            ao_channels_feedback=(),  # 校准不使用反馈通道
            static_output_waveform=self._output_waveform,
            feedback_function=self._feedback_function,
            export_function=self._export_function,
        )

        try:
            # 启动任务
            sync_io.start()
            logger.info("MultiChasCSIO任务已启动")

            # 等待系统初始化稳定
            time.sleep(2.0)

            # 执行测量循环（遍历所有通道）
            for channel_idx in range(len(self._ao_channels)):
                # 创建虚拟点坐标（x=1，y=通道序号）
                virtual_position = Point2D(x=1.0, y=float(channel_idx + 1))

                logger.info(
                    f"测量通道 {channel_idx} ({self._ao_channels[channel_idx]})"
                )

                # 创建新的测量点数据
                point_data: PointRawData = {
                    "position": virtual_position,
                    "ai_data": [],
                }
                self._result_raw_sweep_data["ai_data_list"].append(point_data)

                # 创建只启用当前通道的波形
                single_channel_waveform = self._create_single_channel_waveform(
                    channel_idx
                )

                # 更新静态输出波形
                sync_io.update_static_output_waveform(single_channel_waveform)
                logger.debug(
                    f"已更新波形，启用通道 {channel_idx} ({self._ao_channels[channel_idx]})"
                )

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
                max_wait_time = chunk_duration * chunks_per_start * 2.0  # 最大等待时间
                poll_interval = 0.05  # 轮询间隔50ms
                elapsed_time = 0.0

                logger.debug(f"开始采集 {chunks_per_start} 个chunk")
                while (
                    not self._chunk_collection_complete and elapsed_time < max_wait_time
                ):
                    time.sleep(poll_interval)
                    elapsed_time += poll_interval

                # 禁用数据导出
                sync_io.enable_export = False

                # 检查采集到的数据数量
                collected_chunks = len(point_data["ai_data"])
                logger.info(
                    f"通道 {channel_idx} 采集完成，共 {collected_chunks} 个chunk"
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

        # 3. 计算传递函数（绝对传递函数：AI相对于AO）
        logger.info("计算传递函数")
        raw_tf_list = calculate_transfer_function(filtered_data)

        # 保存原始传递函数列表（用于后续分析）
        self._result_raw_tf_list = raw_tf_list

        # 4. 提取每个通道的绝对幅值比和绝对相位差
        logger.info("提取每个通道的传递函数数据")
        channel_abs_amp_ratios = []  # 存储每个通道的绝对幅值比
        channel_abs_phase_shifts = []  # 存储每个通道的绝对相位差

        for channel_idx in range(len(self._ao_channels)):
            # 找到该通道的测量点（应该只有一个，x=1, y=channel_idx+1）
            channel_tf_data = next(
                (
                    tf_data
                    for tf_data in raw_tf_list
                    if tf_data["position"].y == channel_idx + 1
                ),
                None,
            )

            if channel_tf_data is not None:
                # 提取幅值比和相位差（绝对值）
                amp_ratio = channel_tf_data["amp_ratio"]
                phase_shift = channel_tf_data["phase_shift"]

                channel_abs_amp_ratios.append(amp_ratio)
                channel_abs_phase_shifts.append(phase_shift)

                logger.debug(
                    f"通道 {channel_idx} ({self._ao_channels[channel_idx]}) "
                    f"绝对幅值比={amp_ratio:.6f}, 绝对相位差={phase_shift:.6f}rad"
                )
            else:
                logger.warning(f"通道 {channel_idx} 没有有效测量数据，使用默认值")
                channel_abs_amp_ratios.append(1.0)
                channel_abs_phase_shifts.append(0.0)

        # 5. 计算所有通道的平均值
        mean_amp_ratio = np.mean(channel_abs_amp_ratios)
        mean_phase_shift = np.mean(channel_abs_phase_shifts)

        logger.info(
            f"所有通道平均值: 幅值比={mean_amp_ratio:.6f}, 相位差={mean_phase_shift:.6f}rad"
        )

        # 6. 计算补偿数据（相对于所有通道平均值的补偿参数）
        logger.info("计算补偿数据")
        self._result_final_comp_list = []
        frequency = self._sine_args["frequency"]

        for channel_idx in range(len(self._ao_channels)):
            # 计算幅值补偿倍率（补偿到平均值需要乘以的倍率）
            amp_comp_ratio = mean_amp_ratio / channel_abs_amp_ratios[channel_idx]

            # 计算相对相位差（相对于平均值）
            phase_shift_relative = (
                mean_phase_shift - channel_abs_phase_shifts[channel_idx]
            )

            # 将相对相位差转换为时间延迟补偿值（秒）
            # time_delay_comp = phase_shift_relative / (2π * frequency)
            time_delay_comp = phase_shift_relative / (2.0 * np.pi * frequency)

            # 创建虚拟位置（x=0, y=通道序号）
            comp_position = Point2D(x=0.0, y=float(channel_idx + 1))

            # 创建PointCompData（存储补偿参数）
            comp_data: PointCompData = {
                "position": comp_position,
                "amp_ratio": amp_comp_ratio,
                "time_delay": time_delay_comp,
            }
            self._result_final_comp_list.append(comp_data)

            logger.info(
                f"通道 {channel_idx} ({self._ao_channels[channel_idx]}) "
                f"幅值补偿倍率={amp_comp_ratio:.6f}, "
                f"时间延迟补偿={time_delay_comp * 1e6:.3f}μs"
            )

        # 保存平均幅值比（用于后续保存到CalibData）
        self._amp_ratio_mean = mean_amp_ratio

        logger.info("单次校准流程完成")

    def calibrate(
        self,
        starts_num: int = 10,
        chunks_per_start: int = 3,
        apply_filter: bool = True,
        lowcut: float = 100.0,
        highcut: float = 20000.0,
        result_folder: str | Path | None = None,
        settle_time: float | None = None,
    ) -> None:
        """
        执行校准流程（多次独立校准并平均）

        该方法会多次调用_single_calibrate方法进行独立校准，每次校准都经历完整的任务创建和删除流程。
        每次校准完成后保存CalibData文件，最后读取所有文件并对每个通道的数据进行平均，
        生成最终的CalibData文件和绘图。

        Args:
            starts_num: 独立校准的次数，默认为10。
                这是提高校准精度的主要手段，建议设置更高的值。
            chunks_per_start: 每次启动采集的连续chunk数，默认为3
            apply_filter: 是否应用滤波，默认为True
            lowcut: 滤波器低频截止频率（Hz），默认为100.0
            highcut: 滤波器高频截止频率（Hz），默认为20000.0
            result_folder: 可选，结果保存文件夹路径。如果为None，将使用默认路径
                          'storage/calib/calib_result_octopus'（相对于项目根目录）。
                          最终将保存一个平均后的CalibData文件和polar模式绘图
            settle_time: 通道切换后的稳定等待时间（秒）。如果为None，则使用初始化时
                        计算的默认值（chunk时长 + 0.1秒）

        Raises:
            ValueError: 当参数无效时
        """
        if starts_num < 1:
            raise ValueError("启动次数必须至少为1")

        logger.info(
            f"开始校准流程 - "
            f"独立校准次数: {starts_num}, "
            f"每次启动chunk数: {chunks_per_start}"
        )

        # 确定结果保存路径（如果未指定，使用默认路径）
        if result_folder is None:
            # 使用默认路径：项目根目录的 storage/calib/calib_result_octopus
            result_path = (
                Path(__file__).resolve().parents[3]
                / "storage"
                / "calib"
                / "calib_result_octopus"
            )
            logger.info(f"未指定result_folder，使用默认路径: {result_path}")
        else:
            result_path = Path(result_folder)
            logger.info(f"使用指定的result_folder: {result_path}")

        # 创建结果文件夹
        result_path.mkdir(parents=True, exist_ok=True)

        # 创建临时文件夹用于存储每次校准的结果
        temp_folder = result_path / "temp_calib_data"
        temp_folder.mkdir(parents=True, exist_ok=True)

        # 存储每次校准的CalibData文件路径
        calib_data_files: list[Path] = []

        # 存储所有starts的raw_comp_list（用于cartesian模式绘图）
        all_raw_comp_list: list[PointCompData] = []

        # 执行多次独立校准
        for calib_idx in range(starts_num):
            logger.info(f"开始第 {calib_idx + 1}/{starts_num} 次独立校准")

            # 调用_single_calibrate方法
            self._single_calibrate(
                chunks_per_start=chunks_per_start,
                apply_filter=apply_filter,
                lowcut=lowcut,
                highcut=highcut,
                settle_time=settle_time,
            )

            # 收集这次校准的raw_comp_list（添加start_idx信息）
            if self._result_final_comp_list is not None:
                for comp_data in self._result_final_comp_list:
                    # 创建新的PointCompData，添加start_idx信息到position的x坐标
                    raw_comp_data: PointCompData = {
                        "position": Point2D(
                            x=float(calib_idx),  # x坐标存储start索引
                            y=comp_data["position"].y,  # y坐标存储通道索引
                        ),
                        "amp_ratio": comp_data["amp_ratio"],
                        "time_delay": comp_data["time_delay"],
                    }
                    all_raw_comp_list.append(raw_comp_data)

            # 保存这次校准的CalibData
            calib_data_path = temp_folder / f"calib_data_{calib_idx + 1}.pkl"
            self.save_calib_data(calib_data_path)
            calib_data_files.append(calib_data_path)
            logger.info(f"第 {calib_idx + 1} 次校准完成，已保存到: {calib_data_path}")

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
                logger.error(
                    f"加载校准数据文件失败: {calib_file}, 错误: {e}", exc_info=True
                )
                raise RuntimeError(f"加载校准数据文件失败: {calib_file}") from e

        # 验证所有CalibData的通道数一致
        channels_num = len(self._ao_channels)
        for idx, calib_data in enumerate(all_calib_data):
            if len(calib_data["comp_list"]) != channels_num:
                raise RuntimeError(
                    f"第 {idx + 1} 次校准的通道数({len(calib_data['comp_list'])}) "
                    f"与预期({channels_num})不一致"
                )

        # 对每个通道的数据进行平均
        averaged_comp_list: list[PointCompData] = []
        for channel_idx in range(channels_num):
            # 收集该通道在所有校准中的相对幅值补偿比和相对时间延迟
            amp_ratios_relative = [
                calib_data["comp_list"][channel_idx]["amp_ratio"]
                for calib_data in all_calib_data
            ]
            time_delays_relative = [
                calib_data["comp_list"][channel_idx]["time_delay"]
                for calib_data in all_calib_data
            ]

            # 计算平均值
            avg_amp_ratio_relative = np.mean(amp_ratios_relative)
            avg_time_delay_relative = np.mean(time_delays_relative)

            # 创建平均后的PointCompData
            final_position = Point2D(x=0.0, y=float(channel_idx + 1))
            averaged_comp_data: PointCompData = {
                "position": final_position,
                "amp_ratio": avg_amp_ratio_relative,
                "time_delay": avg_time_delay_relative,
            }
            averaged_comp_list.append(averaged_comp_data)

            logger.info(
                f"通道 {channel_idx} ({self._ao_channels[channel_idx]}) "
                f"平均相对幅值补偿比={avg_amp_ratio_relative:.6f}, "
                f"平均相对时间延迟={avg_time_delay_relative * 1e6:.3f}μs"
            )

        # 计算平均的amp_ratio_mean
        avg_amp_ratio_mean = np.mean(
            [calib_data["amp_ratio_mean"] for calib_data in all_calib_data]
        )
        logger.info(f"平均绝对幅值比典型值: {avg_amp_ratio_mean:.6f}")

        # 创建最终的CalibData
        final_calib_data: CalibData = {
            "comp_list": averaged_comp_list,
            "ao_channels": self._ao_channels,
            "ai_channel": self._ai_channel,
            "sampling_info": self._sampling_info,
            "sine_args": self._sine_args,
            "amp_ratio_mean": avg_amp_ratio_mean,
        }

        # 更新内部状态为平均后的结果（这样 plot_transfer_functions 的 overview 模式能使用真正的最终数据）
        self._result_final_comp_list = averaged_comp_list
        self._amp_ratio_mean = avg_amp_ratio_mean

        # 保存所有starts的raw_comp_list（用于cartesian模式绘图）
        self._result_raw_comp_list = all_raw_comp_list

        # 保存最终的CalibData
        final_calib_data_path = result_path / "calib_data.pkl"
        try:
            with open(final_calib_data_path, "wb") as f:
                pickle.dump(final_calib_data, f)
            logger.info(f"最终平均CalibData已保存到: {final_calib_data_path}")
        except Exception as e:
            logger.error(f"保存最终CalibData失败: {e}", exc_info=True)
            raise OSError(f"保存最终CalibData失败: {e}") from e

        # 保存最后一次校准的原始SweepData（用于后续分析）
        raw_sweep_data_path = result_path / "raw_sweep_data.pkl"
        self.save_sweep_data(raw_sweep_data_path)

        # 绘制polar模式绘图（使用最终平均后的数据）
        polar_plot_path = result_path / "transfer_function_polar.png"
        self.plot_transfer_functions(mode="polar", save_path=polar_plot_path)
        logger.info(f"已保存polar模式绘图到: {polar_plot_path}")

        # 绘制cartesian模式绘图(使用最终平均后的数据)
        cartesian_plot_path = result_path / "transfer_function_cartesian.png"
        self.plot_transfer_functions(mode="cartesian", save_path=cartesian_plot_path)
        logger.info(f"已保存cartesian模式绘图到: {cartesian_plot_path}")

        # 删除临时文件夹及其内容
        import shutil

        try:
            shutil.rmtree(temp_folder)
            logger.info(f"已删除临时文件夹: {temp_folder}")
        except Exception as e:
            logger.warning(f"删除临时文件夹失败: {e}")

        logger.info(f"校准流程完成，所有结果已保存到: {result_path}")

    def save_calib_data(self, file_path: str | Path) -> None:
        """
        保存校准结果到本地文件

        使用pickle序列化将最终补偿数据保存到磁盘。

        Args:
            file_path: 保存文件的路径（支持字符串或Path对象）

        Raises:
            RuntimeError: 当尚未执行校准时
            IOError: 当文件保存失败时
        """
        if self._result_final_comp_list is None or self._amp_ratio_mean is None:
            raise RuntimeError("尚未执行校准，无法保存结果")

        # 转换为Path对象
        save_path = Path(file_path)

        # 确保父目录存在
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 准备保存的数据（只保存最终平均后的补偿数据）
        calib_data: CalibData = {
            "comp_list": self._result_final_comp_list,
            "ao_channels": self._ao_channels,
            "ai_channel": self._ai_channel,
            "sampling_info": self._sampling_info,
            "sine_args": self._sine_args,
            "amp_ratio_mean": self._amp_ratio_mean,
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
        mode: Literal["polar", "cartesian"],
        save_path: str | Path | None = None,
    ) -> None:
        """
        绘制补偿数据

        支持两种绘图模式：
        - polar（极坐标）: 在极坐标系中显示所有补偿数据的分布情况，属于"概览图"
        - cartesian（直角坐标）: 使用自适应坐标轴缩放显示测量点周围的细节，属于"细节图"

        Args:
            mode: 绘图模式，"polar"为极坐标图，"cartesian"为直角坐标图
            save_path: 可选，图像保存路径。如果为None，则只显示不保存

        Raises:
            RuntimeError: 当尚未执行校准时
        """
        if self._result_final_comp_list is None:
            raise RuntimeError("尚未执行校准，无法绘制结果")

        # 获取频率用于时间延迟到相位的转换
        frequency = self._sine_args["frequency"]

        # 根据模式选择数据源
        if mode == "polar":
            # polar模式使用平均后的最终结果
            comp_data_list = self._result_final_comp_list
        elif mode == "cartesian":
            # cartesian模式使用所有starts的详细数据
            if self._result_raw_comp_list is None:
                raise RuntimeError("尚未执行校准，无法绘制详细结果")
            comp_data_list = self._result_raw_comp_list
        else:
            raise ValueError(f"不支持的绘图模式: {mode}")

        # 提取所有数据
        amp_ratios = []
        time_delays = []
        channel_indices = []
        start_indices = []  # 用于cartesian模式标注

        for comp_data in comp_data_list:
            # 从PointCompData中提取相对幅值补偿倍率和时间延迟
            amp_ratio_relative = comp_data["amp_ratio"]
            time_delay = comp_data["time_delay"]
            channel_idx = int(comp_data["position"].y) - 1
            start_idx = int(comp_data["position"].x)  # x坐标存储start索引

            amp_ratios.append(amp_ratio_relative)
            time_delays.append(time_delay)
            channel_indices.append(channel_idx)
            start_indices.append(start_idx)

        if mode == "polar":
            logger.info("开始绘制传递函数极坐标图")
            # 创建极坐标图
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="polar")

            # 将时间延迟转换为相位（弧度）
            angles = [2.0 * np.pi * frequency * td for td in time_delays]

            # 绘制散点
            scatter = ax.scatter(
                angles,
                amp_ratios,
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
                angles, amp_ratios, channel_indices, strict=True
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
                f"补偿数据极坐标分布\n"
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

        elif mode == "cartesian":
            logger.info("开始绘制补偿数据直角坐标图")
            # 创建直角坐标图
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111)

            # 绘制散点（使用与polar模式相同的颜色映射）
            scatter = ax.scatter(
                time_delays,
                amp_ratios,
                c=channel_indices,
                cmap="viridis",
                s=150,
                alpha=0.8,
                edgecolors="black",
                linewidths=2,
                zorder=3,
            )

            # 为每个数据点添加详细标签（通道名称 + start编号）
            for time_delay, amp_ratio, channel_idx, start_idx in zip(
                time_delays, amp_ratios, channel_indices, start_indices, strict=True
            ):
                # 获取通道名称
                channel_name = self._ao_channels[channel_idx]
                # 创建标签（通道名称 + start编号）
                label = f"{channel_name} #{start_idx + 1}"

                # 使用annotate添加带箭头的标签
                ax.annotate(
                    label,
                    xy=(time_delay, amp_ratio),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="arc3,rad=0.2",
                        color="gray",
                        lw=1.5,
                    ),
                    zorder=4,
                )

            # 设置坐标轴标签
            ax.set_xlabel("时间延迟 (s)", fontsize=12)
            ax.set_ylabel("相对幅值比", fontsize=12)

            # 设置网格
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)

            # 设置标题
            ax.set_title(
                f"补偿数据直角坐标分布（细节图）\n"
                f"频率: {self._sine_args['frequency']}Hz, "
                f"幅值: {self._sine_args['amplitude']}V",
                fontsize=14,
                pad=20,
            )

            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
            cbar.set_label("通道索引", fontsize=10)

            # 在 colorbar 上标注通道名称
            cbar.set_ticks(range(len(self._ao_channels)))
            tick_labels = [
                f"{idx}: {self._ao_channels[idx]}"
                for idx in range(len(self._ao_channels))
            ]
            cbar.set_ticklabels(tick_labels)
            cbar.ax.tick_params(labelsize=8)

            # 自适应坐标轴范围，显示测量点周围的细节
            time_delay_array = np.array(time_delays)
            amp_ratio_array = np.array(amp_ratios)

            # 计算数据范围
            td_min, td_max = time_delay_array.min(), time_delay_array.max()
            ar_min, ar_max = amp_ratio_array.min(), amp_ratio_array.max()

            # 添加边距（数据范围的10%）
            td_margin = (td_max - td_min) * 0.1 if td_max > td_min else 1e-6
            ar_margin = (ar_max - ar_min) * 0.1 if ar_max > ar_min else 0.01

            ax.set_xlim(td_min - td_margin, td_max + td_margin)
            ax.set_ylim(ar_min - ar_margin, ar_max + ar_margin)

        plt.tight_layout()

        # 保存或显示
        if save_path is not None:
            save_path_obj = Path(save_path)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path_obj, dpi=300, bbox_inches="tight")
            logger.info(f"补偿数据图已保存到: {save_path_obj}")

        plt.show()
        logger.info("补偿数据图绘制完成")

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
        获取所有starts的传递函数数据（未平均）

        包含所有独立校准运行的详细数据。例如，如果校准8个通道，运行3次starts，
        则此列表包含3×8=24个PointTFData对象。每个对象的position.x存储start索引，
        position.y存储通道索引+1。

        此数据用于cartesian模式绘图，以显示所有测量点的细节。

        Returns:
            传递函数数据列表（包含所有starts的数据），如果尚未校准则返回None
        """
        if self._result_raw_tf_list is None:
            return None
        return self._result_raw_tf_list.copy()

    @property
    def result_raw_comp_list(self) -> list[PointCompData] | None:
        """
        获取所有starts的补偿数据（未平均）

        包含所有独立校准运行的详细数据。例如，如果校准8个通道，运行3次starts，
        则此列表包含3×8=24个PointCompData对象。每个对象的position.x存储start索引，
        position.y存储通道索引+1。

        此数据用于cartesian模式绘图，以显示所有测量点的细节。

        Returns:
            补偿数据列表（包含所有starts的数据），如果尚未校准则返回None
        """
        if self._result_raw_comp_list is None:
            return None
        return self._result_raw_comp_list.copy()

    @property
    def result_final_comp_list(self) -> list[PointCompData] | None:
        """
        获取最终的补偿数据（每个通道平均后的结果）

        Returns:
            补偿数据列表（每个通道一个PointCompData），如果尚未校准则返回None
        """
        if self._result_final_comp_list is None:
            return None
        return self._result_final_comp_list.copy()

    @property
    def ao_channels(self) -> tuple[str, ...]:
        """
        获取AO通道列表

        Returns:
            AO通道名称元组
        """
        return self._ao_channels

    @property
    def ai_channels(self) -> tuple[str, ...]:
        """
        获取AI通道列表

        Returns:
            AI通道名称元组
        """
        return self._ai_channel
