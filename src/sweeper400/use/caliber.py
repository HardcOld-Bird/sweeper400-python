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
    ChannelCompData,
    ChannelTFData,
    CompData,
    Point2D,
    PointSweepData,
    PositiveInt,
    SamplingInfo,
    SineArgs,
    SweepData,
    TFData,
    Waveform,
    average_comp_data_list,
    average_sweep_data,
    average_tf_data_list,
    comp_to_tf,
    extract_single_tone_information_vvi,
    filter_sweep_data,
    get_sine_cycles,
    get_sine_multi_ch,
    tf_to_comp,
)
from ..logger import get_logger
from ..measure import SingleChasCSIO

# 获取模块日志器
logger = get_logger(__name__)


class CaliberOctopus:
    """
    # 多通道校准类（章鱼模式）

    该类专门用于多通道输出情形下，各个通道响应函数的校准（calibration）。
    核心组件是SingleChasCSIO对象，用于控制NI数据采集卡的多通道同步数据输出和采集。
    "Octopus"（章鱼）命名表示该类使用章鱼型波导，测量单个AI通道与多个AO通道的传递函数。

    ## 主要特性：
        - 使用单频正弦信号进行校准
        - 通过多次独立校准并平均来提高精度
        - 每次独立校准创建一次SingleChasCSIO对象，通过动态更换波形实现多通道测量
        - 使用SweepData格式存储数据，便于使用现有数据处理函数
        - 自动应用滤波和平均处理以提高准确度
        - 将校准结果序列化保存到本地磁盘
        - 支持极坐标可视化模式
        - 支持基于已有补偿数据进行校准验证（通过comp_data参数）

    ## 校准原理：
        1. 第一阶段：循环调用_single_calibrate，采集并保存原始数据
           - 多次执行独立校准（每次创建和销毁SingleChasCSIO对象）
           - 每次独立校准中：
             a. 创建SingleChasCSIO对象
             b. 通过动态更换波形，每次只向一个通道发送信号（其他通道输出零）
             c. 对每个通道采集多个连续chunk
             d. 将数据存储为SweepData格式（x坐标=0，y坐标=通道序号）
             e. 返回原始SweepData并立即保存到文件
        2. 第二阶段：处理SweepData，计算补偿数据
           - 对每个SweepData进行平均和滤波处理
           - 使用类内部方法 _calculate_comp_data 计算每次校准的CompData
           - 收集所有CompData的ChannelCompData用于cartesian模式绘图
           - 绘制补偿数据直角坐标图（细节图）
        3. 第三阶段：平均所有校准结果
           - 对多个CompData进行平均，得到平均后的单个CompData
           - 使用comp_to_tf将平均后的CompData转换为TFData
        4. 第四阶段：绘制极坐标图和保存最终数据
           - 使用平均后的TFData绘制传递函数极坐标图（概览图）
           - 保存最终的CompData文件

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
    caliber.save_comp_data("calibration_results.pkl")

    # 绘制极坐标图
    caliber.plot_comp_data(mode="polar")

    # 校准验证：使用已有补偿数据进行二次校准
    caliber_verify = CaliberOctopus(
        ai_channels=("PXI1Slot2/ai0",),
        ao_channels=(...),  # 与上面相同
        sampling_info=sampling_info,
        sine_args=sine_args,
        comp_data="calibration_results.pkl"  # 使用之前的校准结果
    )
    caliber_verify.calibrate(starts_num=10, chunks_per_start=4)
    # 如果校准有效，验证结果应显示所有通道响应一致
    ```

    ## 注意事项：
        - 校准前确保硬件连接正确（传声器和扬声器已正确安装）
        - 建议在安静环境中进行校准以减少噪声干扰
        - 校准频率和幅值应根据实际应用场景选择
        - 该校准方法仅适用于单频单幅值工作点
        - 使用comp_data参数进行校准验证时，应使用相同的通道配置和测量参数
    """

    # 获取类日志器
    logger = get_logger(f"{__name__}.CaliberOctopus")

    def __init__(
        self,
        ai_channels: tuple[str, ...],
        ao_channels: tuple[str, ...],
        sampling_info: SamplingInfo,
        sine_args: SineArgs,
        comp_data: str | Path | None = None,
    ) -> None:
        """
        初始化校准对象

        Args:
            ai_channels: AI 通道名称元组（例如 ("PXI2Slot2/ai0",)）。
                         章鱼模式下通常只使用第一个元素，其余元素将被忽略。
            ao_channels: AO 通道名称元组（例如 ("PXI2Slot2/ao0", "PXI2Slot2/ao1", ...)）
            sampling_info: 采样信息，包含采样率和采样点数
            sine_args: 正弦波参数，包含频率、幅值和相位信息
            comp_data: 可选，补偿数据文件路径。如果指定，将加载该文件并使用
                       补偿波形。支持部分补偿：comp_data中可以只包含部分AO通道，
                       未包含的通道将使用未补偿信号。在校准测量时，每个通道将
                       发送其对应的补偿信号（而非相同的未补偿信号），从而验证
                       校准补偿的有效性

        Raises:
            ValueError: 当参数无效时
            FileNotFoundError: 当comp_data文件不存在时
            RuntimeError: 当comp_data文件加载失败时
        """
        # 验证参数
        if not ai_channels:
            raise ValueError("AI 通道列表不能为空")
        if not ao_channels:
            raise ValueError("AO 通道列表不能为空")

        # 保存配置参数
        self._ai_channels = ai_channels  # 章鱼模式下默认使用第一个元素
        self._ao_channels = ao_channels
        self._sampling_info = sampling_info
        self._sine_args = sine_args

        # 生成输出波形
        if comp_data is not None:
            # 加载补偿数据文件
            comp_data_path = Path(comp_data)
            if not comp_data_path.exists():
                raise FileNotFoundError(f"补偿数据文件不存在: {comp_data_path}")

            try:
                with open(comp_data_path, "rb") as f:
                    loaded_comp_data: CompData = pickle.load(f)
                logger.info(f"成功加载补偿数据文件: {comp_data_path}")
            except Exception as e:
                logger.error(f"加载补偿数据文件失败: {e}", exc_info=True)
                raise RuntimeError(f"加载补偿数据文件失败: {e}") from e

            # 从 comp_list 中提取已有补偿数据的 AO 通道名称集合
            # 新的 CompData 不再有 ao_channels 字段，通道名存储在每个 ChannelCompData 中
            comp_channel_map: dict[str, int] = {
                comp_point["ao_channel"]: idx
                for idx, comp_point in enumerate(loaded_comp_data["comp_list"])
            }
            comp_ao_channels = set(comp_channel_map.keys())

            # 生成多通道波形（支持部分补偿）
            # 首先生成未补偿的单通道波形
            single_channel_waveform = get_sine_cycles(sampling_info, sine_args)

            # 创建多通道数组
            num_channels = len(ao_channels)
            multi_channel_data = np.zeros(
                (num_channels, single_channel_waveform.samples_num), dtype=np.float64
            )

            # 对每个通道，检查是否在comp_data中
            compensated_count = 0
            for idx, channel_name in enumerate(ao_channels):
                if channel_name in comp_ao_channels:
                    # 该通道在comp_data中，使用补偿波形
                    comp_idx = comp_channel_map[channel_name]
                    temp_comp_data: CompData = {
                        "comp_list": [loaded_comp_data["comp_list"][comp_idx]],
                        "sampling_info": loaded_comp_data["sampling_info"],
                        "sine_args": loaded_comp_data["sine_args"],
                        "mean_amp_ratio": loaded_comp_data["mean_amp_ratio"],
                        "mean_phase_shift": loaded_comp_data["mean_phase_shift"],
                    }
                    compensated_waveform = get_sine_multi_ch(
                        sampling_info=sampling_info,
                        sine_args=sine_args,
                        comp_data=temp_comp_data,
                    )
                    multi_channel_data[idx, :] = compensated_waveform[0, :]
                    compensated_count += 1
                    logger.debug(f"通道 {channel_name} 使用补偿波形")
                else:
                    # 该通道不在comp_data中，使用未补偿波形
                    multi_channel_data[idx, :] = single_channel_waveform
                    logger.debug(f"通道 {channel_name} 使用未补偿波形")

            # 创建多通道Waveform对象
            self._output_waveform = Waveform(
                input_array=multi_channel_data,
                sampling_rate=sampling_info["sampling_rate"],
                timestamp=single_channel_waveform.timestamp,
                id=single_channel_waveform.id,
                sine_args=sine_args,
            )
            logger.info(
                f"使用部分补偿数据生成多通道波形，shape={self._output_waveform.shape}，"
                f"补偿通道数={compensated_count}/{len(ao_channels)}"
            )
        else:
            # 生成多通道未补偿波形（每个通道都是相同的正弦波）
            # 首先生成单通道波形
            single_channel_waveform = get_sine_cycles(sampling_info, sine_args)

            # 创建多通道数组，每个通道都是相同的信号
            num_channels = len(ao_channels)
            multi_channel_data = np.tile(single_channel_waveform, (num_channels, 1))

            # 创建多通道Waveform对象
            self._output_waveform = Waveform(
                input_array=multi_channel_data,
                sampling_rate=sampling_info["sampling_rate"],
                timestamp=single_channel_waveform.timestamp,
                id=single_channel_waveform.id,
                sine_args=sine_args,
            )
            logger.info(f"生成多通道未补偿波形，shape={self._output_waveform.shape}")

        # 计算默认的稳定等待时间（chunk时长 + 0.1秒）
        chunk_duration = self._output_waveform.duration
        self._default_settle_time = chunk_duration + 0.1

        # 内部状态变量（用于校准过程）
        # 临时存储当前正在采集的SweepData（供_export_function使用）
        self._current_sweep_data: SweepData | None = None

        # 数据采集控制变量
        self._target_chunks: int = 0  # 目标采集chunk数量
        self._chunk_collection_complete: bool = False  # chunk采集完成标志

        # 校准结果存储
        # 所有starts的补偿数据（未平均，用于cartesian模式绘图）
        self._result_raw_comp_data: CompData | None = None

        # 平均后的传递函数数据（用于polar模式绘图）
        self._result_averaged_tf_data: TFData | None = None

        # 最终补偿数据结果（已平均的CompData）
        self._result_averaged_comp_data: CompData | None = None

        # 所有通道传递函数幅值比的平均值
        self._amp_ratio_mean: float | None = None

        logger.info(
            f"Caliber 实例已创建 - "
            f"AI通道数: {len(ai_channels)}, "
            f"AO通道数: {len(ao_channels)}, "
            f"频率: {sine_args['frequency']}Hz, "
            f"幅值: {sine_args['amplitude']}V, "
            f"采样率: {sampling_info['sampling_rate']}Hz, "
            f"默认稳定时间: {self._default_settle_time:.3f}s, "
            f"使用补偿数据: {comp_data is not None}"
        )

    def _create_single_channel_waveform(self, active_channel_idx: int) -> Waveform:
        """
        创建只启用一个通道的多通道波形

        生成一个多通道波形，其中只有指定通道输出信号，其他通道输出零。
        从self._output_waveform（多通道波形）中提取指定通道的数据。

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
        num_samples = self._output_waveform.samples_num
        multi_channel_data = np.zeros((num_channels, num_samples), dtype=np.float64)

        # 只在指定通道填充信号（从多通道波形中提取）
        multi_channel_data[active_channel_idx, :] = self._output_waveform[
            active_channel_idx, :
        ]

        logger.debug(f"创建单通道激活波形（通道 {active_channel_idx}）")

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
            hasattr(self, "_current_sweep_data")
            and self._current_sweep_data is not None
            and len(self._current_sweep_data["ai_data_list"]) > 0
        ):
            current_point = self._current_sweep_data["ai_data_list"][-1]
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

    @staticmethod
    def _feedback_function(ai_waveform: Waveform) -> Waveform:
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

    def _calculate_comp_data(
        self,
        filtered_sweep_data: SweepData,
    ) -> CompData:
        """
        从已滤波/平均的 SweepData 计算补偿数据（CompData）。

        该方法替代了原全局函数 calculate_compensation_list，将计算逻辑内聚到类内部，
        因为计算过程需要访问类内部的通道配置信息（self._ai_channels[0]、self._ao_channels）。
        章鱼模式下固定使用 self._ai_channels[0] 作为参考 AI 通道。

        计算流程：
        1. 对每个 AO 通道（SweepData 中的每个测量点），提取 AI 波形并计算传递函数
        2. 计算所有通道对的平均幅值比和平均相位差
        3. 将传递函数转换为补偿数据（相对于平均值的补偿参数）
        4. 构建并返回 CompData

        Args:
            filtered_sweep_data: 已经过平均和滤波处理的 SweepData，
                每个测量点的 ai_data 列表中只有一条波形（已平均）。
                SweepData 中的测量点顺序与 self._ao_channels 的顺序一致。

        Returns:
            CompData: 包含所有通道对补偿参数的数据容器

        Raises:
            RuntimeError: 当 AO 波形没有 sine_args 属性时
            ValueError: 当测量点数量与 AO 通道数不一致时
        """
        ao_waveform = filtered_sweep_data["ao_data"]
        if ao_waveform.sine_args is None:
            raise RuntimeError("AO波形没有sine_args属性，无法计算传递函数")

        ao_sine_args = ao_waveform.sine_args
        num_ao_channels = len(self._ao_channels)

        if len(filtered_sweep_data["ai_data_list"]) != num_ao_channels:
            raise ValueError(
                f"SweepData中的测量点数({len(filtered_sweep_data['ai_data_list'])}) "
                f"与AO通道数({num_ao_channels})不一致"
            )

        # 第一步：计算每个通道对的传递函数（ChannelTFData）
        tf_list: list[ChannelTFData] = []
        for channel_idx, point_data in enumerate(filtered_sweep_data["ai_data_list"]):
            ao_channel_name = self._ao_channels[channel_idx]
            ai_waveform = point_data["ai_data"][0]  # 已平均的单通道 AI 波形

            # 提取 AI 信号的正弦波参数
            ai_sine_args = extract_single_tone_information_vvi(
                ai_waveform,
                approx_freq=ao_sine_args["frequency"],
            )

            # 计算传递函数
            amp_ratio = ai_sine_args["amplitude"] / ao_sine_args["amplitude"]
            phase_shift = ai_sine_args["phase"] - ao_sine_args["phase"]
            # 将相位差归一化到 [-π, π] 区间
            phase_shift = float(np.arctan2(np.sin(phase_shift), np.cos(phase_shift)))

            tf_point: ChannelTFData = {
                "ai_channel": self._ai_channels[0],
                "ao_channel": ao_channel_name,
                "amp_ratio": float(amp_ratio),
                "phase_shift": phase_shift,
            }
            tf_list.append(tf_point)

            logger.debug(
                f"AO通道 {ao_channel_name} -> AI通道 {self._ai_channels[0]}: "
                f"amp_ratio={amp_ratio:.6f}, phase_shift={phase_shift:.6f}rad"
            )

        # 第二步：计算所有通道对的平均幅值比和平均相位差
        mean_amp_ratio = float(np.mean([tf["amp_ratio"] for tf in tf_list]))
        mean_phase_shift = float(np.mean([tf["phase_shift"] for tf in tf_list]))

        logger.debug(
            f"平均幅值比={mean_amp_ratio:.6f}, 平均相位差={mean_phase_shift:.6f}rad"
        )

        # 第三步：将传递函数转换为补偿数据
        comp_list: list[ChannelCompData] = []
        for tf_point in tf_list:
            comp_point = tf_to_comp(
                tf_data=tf_point,
                sine_args=ao_sine_args,
                mean_amp_ratio=mean_amp_ratio,
                mean_phase_shift=mean_phase_shift,
            )
            comp_list.append(comp_point)

        # 第四步：构建 CompData
        comp_data: CompData = {
            "comp_list": comp_list,
            "sampling_info": self._sampling_info,
            "sine_args": ao_sine_args,
            "mean_amp_ratio": mean_amp_ratio,
            "mean_phase_shift": mean_phase_shift,
        }

        logger.info(
            f"补偿数据计算完成，共 {len(comp_list)} 个通道对，"
            f"频率={ao_sine_args['frequency']:.2f}Hz, "
            f"平均幅值比={mean_amp_ratio:.6f}"
        )
        return comp_data

    def _single_calibrate(
        self,
        chunks_per_start: int = 3,
        settle_time: float | None = None,
    ) -> SweepData:
        """
        执行单次校准流程（内部方法）

        对每个通道采集多个连续chunk，返回原始的SweepData。
        数据存储为SweepData格式，x坐标为通道索引（0-based），y坐标固定为0固定为0。

        该方法只负责硬件控制和数据采集，不进行任何数据处理（平均、滤波、
        计算传递函数等）。所有数据处理工作由calibrate方法统一完成。

        Args:
            chunks_per_start: 每次启动采集的连续chunk数，默认为3
            settle_time: 通道切换后的稳定等待时间（秒）。如果为None，则使用初始化时
                        计算的默认值（chunk时长 + 0.1秒）

        Returns:
            原始的SweepData，包含所有通道的采集数据

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
            f"AO通道数: {len(self._ao_channels)}, "
            f"AI通道数: {len(self._ai_channels)}, "
            f"每次启动chunk数: {chunks_per_start}, "
            f"稳定时间: {actual_settle_time:.3f}s"
        )

        # 初始化SweepData（显式类型标注）
        sweep_data: SweepData = {
            "ai_data_list": [],
            "ao_data": self._output_waveform,
        }

        # 临时存储当前sweep_data的引用，供export_function使用
        self._current_sweep_data = sweep_data

        # 创建SingleChasCSIO对象（整个校准流程只创建一次）
        sync_io = SingleChasCSIO(
            ai_channels=self._ai_channels,
            ao_channels_static=self._ao_channels,
            ao_channels_feedback=(),  # 校准不使用反馈通道
            static_output_waveform=self._output_waveform,
            feedback_function=self._feedback_function,
            export_function=self._export_function,
        )

        try:
            # 启动任务
            sync_io.start()
            logger.info("SingleChasCSIO任务已启动")

            # 等待系统初始化稳定
            time.sleep(2.0)

            # 执行测量循环（遍历所有AO通道）
            for channel_idx in range(len(self._ao_channels)):
                ao_channel_name = self._ao_channels[channel_idx]

                logger.info(f"测量AO通道 {channel_idx} ({ao_channel_name})")

                # 创建新的测量点数据（使用 PointSweepData 格式存储原始采集数据）
                # position 字段用 x=channel_idx, y=0.0 作为临时索引，
                # 后续处理时通过 position.x 映射到真实 AO 通道名
                point_data: PointSweepData = {
                    "position": Point2D(x=float(channel_idx), y=0.0),
                    "ai_data": [],
                }
                sweep_data["ai_data_list"].append(point_data)

                # 创建只启用当前AO通道的波形
                single_channel_waveform = self._create_single_channel_waveform(
                    channel_idx
                )

                # 更新静态输出波形
                sync_io.update_static_output_waveform(single_channel_waveform)
                logger.debug(
                    f"已更新波形，启用AO通道 {channel_idx} ({self._ao_channels[channel_idx]})"
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
                    f"AO通道 {channel_idx} 采集完成，共 {collected_chunks} 个chunk"
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
            logger.info("SingleChasCSIO任务已停止")

        # 清理临时引用
        self._current_sweep_data = None

        logger.info("单次校准流程完成，返回原始SweepData")
        return sweep_data

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

        该方法实现四阶段校准工作流程：
        1. 第一阶段：循环调用_single_calibrate，采集并保存原始数据
           - 多次执行独立校准（每次创建和销毁SingleChasCSIO对象）
           - 每次校准返回原始SweepData，立即保存为raw_sweep_data_N.pkl文件
        2. 第二阶段：处理SweepData，计算补偿数据
           - 对每个SweepData进行平均和滤波处理
           - 使用类内部方法 _calculate_comp_data 计算每次校准的CompData
           - 收集所有CompData的ChannelCompData用于绘制补偿数据直角坐标图（细节图）
        3. 第三阶段：平均所有校准结果
           - 对多个CompData进行平均，得到平均后的单个CompData
           - 使用comp_to_tf将平均后的CompData转换为TFData
        4. 第四阶段：绘制极坐标图和保存最终数据
           - 使用平均后的TFData绘制传递函数极坐标图（概览图）
           - 保存最终的CalibData文件

        Args:
            starts_num: 独立校准的次数，默认为10。
                这是提高校准精度的主要手段，建议设置更高的值。
            chunks_per_start: 每次启动采集的连续chunk数，默认为3
            apply_filter: 是否应用滤波，默认为True
            lowcut: 滤波器低频截止频率（Hz），默认为100.0
            highcut: 滤波器高频截止频率（Hz），默认为20000.0
            result_folder: 可选，结果保存文件夹路径。如果为None，将使用默认路径
                          'storage/calib/calib_result_octopus'（相对于项目根目录）。
                          最终将保存多个raw_sweep_data_N.pkl文件、两幅绘图和一个平均后的CalibData文件
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

        # 存储所有starts的SweepData
        all_sweep_data_list: list[SweepData] = []

        # 执行多次独立校准，获取SweepData并立即保存
        logger.info("=" * 60)
        logger.info("第一阶段：循环调用_single_calibrate，采集并保存原始数据")
        logger.info("=" * 60)
        for calib_idx in range(starts_num):
            logger.info(f"开始第 {calib_idx + 1}/{starts_num} 次独立校准")

            # 调用_single_calibrate方法，获取SweepData
            sweep_data = self._single_calibrate(
                chunks_per_start=chunks_per_start,
                settle_time=settle_time,
            )

            # 保存这次校准的原始SweepData
            sweep_data_path = result_path / f"raw_sweep_data_{calib_idx + 1}.pkl"
            try:
                with open(sweep_data_path, "wb") as f:
                    pickle.dump(sweep_data, f)
                logger.info(
                    f"第 {calib_idx + 1} 次SweepData已保存到: {sweep_data_path}"
                )
            except Exception as e:
                logger.error(f"保存SweepData失败: {e}", exc_info=True)
                raise OSError(f"保存SweepData失败: {e}") from e

            # 将SweepData添加到列表中，供后续处理
            all_sweep_data_list.append(sweep_data)

        # 第二阶段：处理所有SweepData，计算CompData
        logger.info("=" * 60)
        logger.info("第二阶段：处理SweepData，计算补偿数据")
        logger.info("=" * 60)

        # 存储所有starts的CompData
        all_comp_data_list: list[CompData] = []

        for calib_idx, sweep_data in enumerate(all_sweep_data_list):
            logger.info(f"处理第 {calib_idx + 1}/{starts_num} 次校准的数据")

            # 1. 对每个点的多个chunk进行平均
            logger.info("对每个点的多个chunk进行平均")
            averaged_data = average_sweep_data(sweep_data)

            # 2. 应用滤波（如果需要）
            if apply_filter:
                logger.info(f"应用带通滤波器: {lowcut}Hz - {highcut}Hz")
                filtered_data = filter_sweep_data(
                    averaged_data,
                    lowcut=lowcut,
                    highcut=highcut,
                )
            else:
                filtered_data = averaged_data

            # 3. 使用类内部方法 _calculate_comp_data 计算补偿数据
            logger.info("计算补偿数据")
            comp_data = self._calculate_comp_data(filtered_data)
            all_comp_data_list.append(comp_data)

            logger.info(
                f"第 {calib_idx + 1} 次校准数据处理完成，"
                f"频率={comp_data['sine_args']['frequency']:.2f}Hz, "
                f"平均幅值比={comp_data['mean_amp_ratio']:.6f}"
            )

        # 从all_comp_data_list构建包含所有starts数据的CompData（用于cartesian模式绘图）
        # 收集所有starts的comp_list，为每个 ChannelCompData 附加 start_idx 信息
        # 注意：ChannelCompData 本身不含 start_idx，此处将所有 starts 的数据合并到一个列表，
        # 供 plot_comp_data(mode="cartesian") 使用时按 ao_channel 分组
        all_raw_comp_list: list[ChannelCompData] = []
        for comp_data in all_comp_data_list:
            all_raw_comp_list.extend(comp_data["comp_list"])

        # 使用第一个CompData的元数据创建包含所有starts数据的CompData
        self._result_raw_comp_data: CompData = {
            "comp_list": all_raw_comp_list,
            "sampling_info": all_comp_data_list[0]["sampling_info"],
            "sine_args": all_comp_data_list[0]["sine_args"],
            "mean_amp_ratio": all_comp_data_list[0]["mean_amp_ratio"],
            "mean_phase_shift": all_comp_data_list[0]["mean_phase_shift"],
        }

        # 绘制cartesian模式绘图（使用所有starts的详细数据）
        logger.info("绘制补偿数据直角坐标图（细节图）")
        cartesian_plot_path = result_path / "compensation_cartesian.png"
        self.plot_comp_data(mode="cartesian", save_path=cartesian_plot_path)
        logger.info(f"已保存cartesian模式绘图到: {cartesian_plot_path}")

        # 第三阶段：平均所有CompData
        logger.info("=" * 60)
        logger.info("第三阶段：平均所有校准结果")
        logger.info("=" * 60)

        # 验证所有CompData的通道数一致
        channels_num = len(self._ao_channels)
        for idx, comp_data in enumerate(all_comp_data_list):
            if len(comp_data["comp_list"]) != channels_num:
                raise RuntimeError(
                    f"第 {idx + 1} 次校准的通道数({len(comp_data['comp_list'])}) "
                    f"与预期({channels_num})不一致"
                )

        # 使用average_comp_data_list函数对所有CompData进行平均
        logger.info("对所有CompData进行平均")
        averaged_comp_data: CompData = average_comp_data_list(all_comp_data_list)

        # 输出每个通道的平均结果日志
        for channel_idx in range(channels_num):
            avg_amp_ratio_relative = averaged_comp_data["comp_list"][channel_idx][
                "amp_ratio"
            ]
            avg_time_delay_relative = averaged_comp_data["comp_list"][channel_idx][
                "time_delay"
            ]
            logger.info(
                f"通道 {channel_idx} ({self._ao_channels[channel_idx]}) "
                f"平均相对幅值补偿比={avg_amp_ratio_relative:.6f}, "
                f"平均相对时间延迟={avg_time_delay_relative * 1e6:.3f}μs"
            )

        # 输出平均元数据日志
        logger.info(
            f"平均元数据: 频率={averaged_comp_data['sine_args']['frequency']:.2f}Hz, "
            f"平均幅值比={averaged_comp_data['mean_amp_ratio']:.6f}, "
            f"平均相位差={averaged_comp_data['mean_phase_shift']:.6f}rad"
        )

        # 使用comp_to_tf将平均后的CompData转换为TFData
        logger.info("将平均后的CompData转换为TFData")
        averaged_tf_list: list[ChannelTFData] = []
        for comp_point in averaged_comp_data["comp_list"]:
            tf_point = comp_to_tf(
                comp_data=comp_point,
                sine_args=averaged_comp_data["sine_args"],
                mean_amp_ratio=averaged_comp_data["mean_amp_ratio"],
                mean_phase_shift=averaged_comp_data["mean_phase_shift"],
            )
            averaged_tf_list.append(tf_point)

        # 构建平均后的TFData（符合新类型定义，包含 sampling_info）
        averaged_tf_data: TFData = {
            "tf_list": averaged_tf_list,
            "sampling_info": averaged_comp_data["sampling_info"],
            "sine_args": averaged_comp_data["sine_args"],
            "mean_amp_ratio": averaged_comp_data["mean_amp_ratio"],
            "mean_phase_shift": averaged_comp_data["mean_phase_shift"],
        }

        # 保存平均后的TFData到内部状态（用于polar模式绘图）
        self._result_averaged_tf_data = averaged_tf_data

        # 第四阶段：绘制极坐标图和保存最终数据
        logger.info("=" * 60)
        logger.info("第四阶段：绘制极坐标图和保存最终数据")
        logger.info("=" * 60)

        # 绘制polar模式绘图（使用平均后的TFData）
        logger.info("绘制传递函数极坐标图（概览图）")
        polar_plot_path = result_path / "transfer_function_polar.png"
        self.plot_comp_data(mode="polar", save_path=polar_plot_path)
        logger.info(f"已保存polar模式绘图到: {polar_plot_path}")

        # 创建最终的CompData（符合新类型定义，不含 ao_channels/ai_channel 冗余字段）
        final_comp_data: CompData = {
            "comp_list": averaged_comp_data["comp_list"],
            "sampling_info": self._sampling_info,
            "sine_args": self._sine_args,
            "mean_amp_ratio": averaged_comp_data["mean_amp_ratio"],
            "mean_phase_shift": averaged_comp_data["mean_phase_shift"],
        }

        # 保存最终的CompData
        final_comp_data_path = result_path / "comp_data.pkl"
        try:
            with open(final_comp_data_path, "wb") as f:
                pickle.dump(final_comp_data, f)
            logger.info(f"最终平均CompData已保存到: {final_comp_data_path}")
        except Exception as e:
            logger.error(f"保存最终CompData失败: {e}", exc_info=True)
            raise OSError(f"保存最终CompData失败: {e}") from e

        # 更新内部状态为平均后的结果
        self._result_averaged_comp_data = averaged_comp_data
        self._amp_ratio_mean = averaged_comp_data["mean_amp_ratio"]

        logger.info("=" * 60)
        logger.info(f"校准流程完成，所有结果已保存到: {result_path}")
        logger.info("=" * 60)

    def save_comp_data(self, file_path: str | Path) -> None:
        """
        保存校准结果到本地文件

        使用pickle序列化将最终补偿数据保存到磁盘。

        Args:
            file_path: 保存文件的路径（支持字符串或Path对象）

        Raises:
            RuntimeError: 当尚未执行校准时
            IOError: 当文件保存失败时
        """
        if self._result_averaged_comp_data is None or self._amp_ratio_mean is None:
            raise RuntimeError("尚未执行校准，无法保存结果")

        # 转换为Path对象
        save_path = Path(file_path)

        # 确保父目录存在
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 准备保存的数据（只保存最终平均后的补偿数据，符合新类型定义）
        comp_data: CompData = {
            "comp_list": self._result_averaged_comp_data["comp_list"],
            "sampling_info": self._sampling_info,
            "sine_args": self._sine_args,
            "mean_amp_ratio": self._result_averaged_comp_data["mean_amp_ratio"],
            "mean_phase_shift": self._result_averaged_comp_data["mean_phase_shift"],
        }

        try:
            with open(save_path, "wb") as f:
                pickle.dump(comp_data, f)
            logger.info(f"校准结果已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存校准结果失败: {e}", exc_info=True)
            raise OSError(f"保存校准结果失败: {e}") from e

    def plot_comp_data(
        self,
        mode: Literal["polar", "cartesian"],
        save_path: str | Path | None = None,
    ) -> None:
        """
        绘制传递函数或补偿数据

        支持两种绘图模式：
        - polar（极坐标）: 使用平均后的TFData绘制传递函数极坐标图，属于"概览图"
        - cartesian（直角坐标）: 使用所有starts的CompData绘制补偿数据直角坐标图，属于"细节图"

        Args:
            mode: 绘图模式，"polar"为极坐标图，"cartesian"为直角坐标图
            save_path: 可选，图像保存路径。如果为None，则只显示不保存

        Raises:
            RuntimeError: 当尚未执行校准时
        """
        # 根据模式选择数据源和绘图方式
        if mode == "polar":
            # polar模式使用平均后的TFData
            if (
                not hasattr(self, "_result_averaged_tf_data")
                or self._result_averaged_tf_data is None
            ):
                raise RuntimeError("尚未执行校准，无法绘制极坐标图")

            tf_data = self._result_averaged_tf_data

            # 提取传递函数数据
            amp_ratios = []
            phase_shifts = []
            channel_indices = []  # AO 通道在 self._ao_channels 中的索引

            for tf_point in tf_data["tf_list"]:
                amp_ratios.append(tf_point["amp_ratio"])
                phase_shifts.append(tf_point["phase_shift"])
                # 通过 ao_channel 名称查找对应的通道索引
                ao_ch = tf_point["ao_channel"]
                ch_idx = (
                    self._ao_channels.index(ao_ch) if ao_ch in self._ao_channels else 0
                )
                channel_indices.append(ch_idx)

        elif mode == "cartesian":
            # cartesian模式使用所有starts的CompData
            if self._result_raw_comp_data is None:
                raise RuntimeError("尚未执行校准，无法绘制直角坐标图")

            comp_data = self._result_raw_comp_data

            # 提取补偿数据
            # 注意：_result_raw_comp_data 的 comp_list 包含所有 starts 的数据（已合并），
            # 每个 ChannelCompData 通过 ao_channel 字段标识通道
            amp_ratios = []
            time_delays = []
            channel_indices = []  # AO 通道在 self._ao_channels 中的索引

            for comp_point in comp_data["comp_list"]:
                amp_ratios.append(comp_point["amp_ratio"])
                time_delays.append(comp_point["time_delay"])
                ao_ch = comp_point["ao_channel"]
                ch_idx = (
                    self._ao_channels.index(ao_ch) if ao_ch in self._ao_channels else 0
                )
                channel_indices.append(ch_idx)

        else:
            raise ValueError(f"不支持的绘图模式: {mode}")

        if mode == "polar":
            logger.info("开始绘制传递函数极坐标图")
            # 创建极坐标图
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="polar")

            # 使用相位差作为角度（已经是弧度）
            angles = phase_shifts

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
                f"传递函数极坐标分布（平均后）\n"
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

            # 为每个数据点添加详细标签（AO 通道名称）
            for time_delay, amp_ratio, channel_idx in zip(
                time_delays, amp_ratios, channel_indices, strict=True
            ):
                # 获取通道名称
                channel_name = self._ao_channels[channel_idx]
                # 创建标签（AO 通道名称）
                label = channel_name

                # 使用annotate添加带箭头的标签
                ax.annotate(
                    label,
                    xy=(time_delay, amp_ratio),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.9,
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": "wheat",
                        "alpha": 0.7,
                    },
                    arrowprops={
                        "arrowstyle": "->",
                        "connectionstyle": "arc3,rad=0.2",
                        "color": "gray",
                        "lw": 1.5,
                    },
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
    def result_averaged_tf_data(self) -> TFData | None:
        """
        获取平均后的传递函数数据

        包含所有通道平均后的传递函数数据，用于polar模式绘图。

        Returns:
            TFData对象，如果尚未校准则返回None
        """
        if self._result_averaged_tf_data is None:
            return None
        # 深拷贝以避免外部修改
        import copy

        return copy.deepcopy(self._result_averaged_tf_data)

    @property
    def result_raw_comp_data(self) -> CompData | None:
        """
        获取所有starts的补偿数据（未平均）

        包含所有独立校准运行的详细数据。CompData中的comp_list包含所有starts×通道数的
        ChannelCompData对象，每个对象通过 ao_channel 字段标识对应的AO通道。

        此数据用于cartesian模式绘图，以显示所有测量点的细节。

        Returns:
            CompData对象（包含所有starts的数据），如果尚未校准则返回None
        """
        if self._result_raw_comp_data is None:
            return None
        # 深拷贝以避免外部修改
        import copy

        return copy.deepcopy(self._result_raw_comp_data)

    @property
    def result_final_comp_data(self) -> CompData | None:
        """
        获取最终的补偿数据（每个通道平均后的结果）

        Returns:
            CompData对象（每个通道一个ChannelCompData），如果尚未校准则返回None
        """
        if self._result_averaged_comp_data is None:
            return None
        # 深拷贝以避免外部修改
        import copy

        return copy.deepcopy(self._result_averaged_comp_data)

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

        章鱼模式下通常只有一个元素，渔网模式下可能有多个元素。

        Returns:
            AI通道名称元组
        """
        return self._ai_channels


class CaliberFishNet(CaliberOctopus):
    """
    # 多通道校准类（渔网模式）

    该类专门用于多AI通道与多AO通道之间传递函数的校准（calibration）。
    核心组件是SingleChasCSIO对象，用于控制NI数据采集卡的多通道同步数据输出和采集。
    "FishNet"（渔网）命名表示该类测量多个AI通道与多个AO通道之间的传递函数矩阵。

    ## 主要特性：
        - 使用单频正弦信号进行校准
        - 通过多次独立校准并平均来提高精度
        - 每次独立校准创建一次SingleChasCSIO对象，通过动态更换波形实现多通道测量
        - 同时采集所有AI通道的数据，测量每个AO通道与所有AI通道的传递函数
        - 使用SweepData格式存储数据（多通道Waveform）
        - 自动应用滤波和平均处理以提高准确度
        - 将校准结果序列化保存到本地磁盘
        - 支持基于已有补偿数据进行部分通道补偿（comp_data可包含少于ao_channels的通道）
        - 重点关注TFData而非CompData，因为结果用于分析而非直接补偿

    ## 校准原理：
        1. 第一阶段：循环调用_single_calibrate，采集并保存原始数据
           - 多次执行独立校准（每次创建和销毁SingleChasCSIO对象）
           - 每次独立校准中：
             a. 创建SingleChasCSIO对象（多个AI通道）
             b. 通过动态更换波形，每次只向一个AO通道发送信号（其他通道输出零）
             c. 同时采集所有AI通道的数据
             d. 对每个AO通道采集多个连续chunk
             e. 将数据存储为SweepData格式（position.x=AO索引，position.y=0）
             f. 返回原始SweepData并立即保存到文件
        2. 第二阶段：处理SweepData，计算传递函数数据
           - 对每个SweepData进行平均和滤波处理
           - 使用类内部方法 _calculate_tf_data 计算每次校准的TFData
           - 每个 ChannelTFData 通过 ao_channel/ai_channel 字段标识通道对
        3. 第三阶段：平均所有校准结果
           - 对多个TFData进行平均，得到平均后的单个TFData
        4. 第四阶段：绘制直角坐标图和保存最终数据
           - 使用平均后的TFData绘制传递函数直角坐标图
           - 保存最终的TFData文件

    ## 使用示例：
    ```python
    from sweeper400.use.caliber import CaliberFishNet
    from sweeper400.analyze import init_sampling_info, init_sine_args

    # 创建采样信息和正弦波参数
    sampling_info = init_sampling_info(171500.0, 85750)  # 采样率171.5kHz, 0.5秒
    sine_args = init_sine_args(frequency=3430.0, amplitude=0.01, phase=0.0)

    # 创建校准对象
    caliber = CaliberFishNet(
        ai_channels=(
            "PXI1Slot2/ai0", "PXI2Slot2/ai0", "PXI2Slot2/ai1",
            "PXI2Slot3/ai0", "PXI2Slot3/ai1", "PXI3Slot2/ai0",
            "PXI3Slot2/ai1", "PXI3Slot3/ai0", "PXI3Slot3/ai1"
        ),
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
    caliber.calibrate(starts_num=10, chunks_per_start=4)

    # 校准结果会自动保存到默认路径
    # 也可以手动保存到指定路径
    caliber.save_tf_data("calibration_results.pkl")

    # 绘制传递函数图
    caliber.plot_comp_data()
    ```

    ## 注意事项：
        - 校准前确保硬件连接正确（传声器和扬声器已正确安装）
        - 建议在安静环境中进行校准以减少噪声干扰
        - 校准频率和幅值应根据实际应用场景选择
        - 该校准方法仅适用于单频单幅值工作点
        - 使用comp_data参数时，可以只包含部分AO通道的补偿数据
    """

    # 获取类日志器
    logger = get_logger(f"{__name__}.CaliberFishNet")

    def __init__(
        self,
        ai_channels: tuple[str, ...],
        ao_channels: tuple[str, ...],
        sampling_info: SamplingInfo,
        sine_args: SineArgs,
        comp_data: str | Path | None = None,
    ) -> None:
        """
        初始化校准对象

        调用父类 CaliberOctopus.__init__ 完成通用初始化（参数验证、波形生成、
        状态变量初始化等），然后补充 FishNet 独有的结果存储变量。

        Args:
            ai_channels: AI 通道名称元组（例如 ("PXI1Slot2/ai0", "PXI2Slot2/ai0", ...)）。
                         渔网模式下通常包含多个 AI 通道，以测量完整的传递函数矩阵。
            ao_channels: AO 通道名称元组（例如 ("PXI2Slot2/ao0", "PXI2Slot2/ao1", ...)）
            sampling_info: 采样信息，包含采样率和采样点数
            sine_args: 正弦波参数，包含频率、幅值和相位信息
            comp_data: 可选，补偿数据文件路径。如果指定，将加载该文件并使用
                       补偿波形。支持部分通道补偿：comp_data中可以只包含部分AO通道，
                       未包含的通道将使用未补偿信号

        Raises:
            ValueError: 当参数无效时
            FileNotFoundError: 当comp_data文件不存在时
            RuntimeError: 当comp_data文件加载失败时
        """
        # 调用父类初始化（完成参数验证、波形生成、通用状态变量初始化）
        super().__init__(
            ai_channels=ai_channels,
            ao_channels=ao_channels,
            sampling_info=sampling_info,
            sine_args=sine_args,
            comp_data=comp_data,
        )

        # FishNet 独有的校准结果存储变量
        # 所有starts的传递函数数据（未平均，用于绘制细节图）
        self._result_raw_tf_data: TFData | None = None

        # 注意：_result_averaged_tf_data 已由父类初始化为 None，此处无需重复声明

    def _calculate_tf_data(
        self,
        filtered_sweep_data: SweepData,
    ) -> list[ChannelTFData]:
        """
        从已滤波/平均的 SweepData 计算传递函数数据（ChannelTFData 列表）。

        该方法将 SweepData 中的多通道 AI 波形分解为多个单通道传递函数，
        对应父类的 _calculate_comp_data 方法。该方法只负责计算，不进行
        平均或滤波处理——这些预处理工作由调用方（calibrate）负责完成。

        SweepData 中的测量点顺序与 self._ao_channels 的顺序一致，
        每个测量点的 position.x 存储 AO 通道索引（0-based），用于映射到真实通道名。

        Args:
            filtered_sweep_data: 已经过平均和滤波处理的 SweepData，
                每个测量点的 ai_data 列表中只有一条波形（已平均）。

        Returns:
            ChannelTFData 列表，每个元素对应一个 AI-AO 通道对的传递函数，
            包含 ai_channel 和 ao_channel 字段（真实通道名称字符串）

        Raises:
            RuntimeError: 当 AO 波形没有 sine_args 属性时
            ValueError: 当 AI 波形维度或通道数不符合预期时
        """
        logger.info("开始计算多通道AI数据的传递函数")

        # 1. 提取AO波形的正弦波参数
        ao_waveform = filtered_sweep_data["ao_data"]
        if ao_waveform.sine_args is None:
            raise RuntimeError("AO波形没有sine_args属性")
        ao_sine_args = ao_waveform.sine_args

        # 2. 对每个AO通道（每个测量点），提取所有AI通道的传递函数
        all_tf_points: list[ChannelTFData] = []

        for point_data in filtered_sweep_data["ai_data_list"]:
            ao_channel_idx = int(point_data["position"].x)  # AO通道索引（0-based）
            ao_channel_name = self._ao_channels[ao_channel_idx]  # 真实AO通道名
            ai_waveform = point_data["ai_data"][0]  # 已平均的多通道AI波形

            # 检查AI波形是否为多通道
            if ai_waveform.ndim != 2:
                raise ValueError(
                    f"期望多通道AI波形（2D数组），但得到{ai_waveform.ndim}维数组"
                )

            num_ai_channels = ai_waveform.shape[0]
            if num_ai_channels != len(self._ai_channels):
                raise ValueError(
                    f"AI波形通道数({num_ai_channels})与配置的AI通道数({len(self._ai_channels)})不一致"
                )

            # 对每个AI通道，计算传递函数
            for ai_channel_idx in range(num_ai_channels):
                ai_channel_name = self._ai_channels[ai_channel_idx]  # 真实AI通道名

                # 提取单通道AI波形
                single_ai_waveform = Waveform(
                    input_array=ai_waveform[ai_channel_idx, :],
                    sampling_rate=ai_waveform.sampling_rate,
                    timestamp=ai_waveform.timestamp,
                )

                # 使用extract_single_tone_information_vvi提取AI信号的正弦波参数
                ai_sine_args = extract_single_tone_information_vvi(
                    single_ai_waveform,
                    approx_freq=ao_sine_args["frequency"],
                )

                # 计算传递函数
                amp_ratio = ai_sine_args["amplitude"] / ao_sine_args["amplitude"]
                phase_shift = ai_sine_args["phase"] - ao_sine_args["phase"]

                # 将相位差归一化到 [-π, π] 区间
                phase_shift = float(
                    np.arctan2(np.sin(phase_shift), np.cos(phase_shift))
                )

                # 创建 ChannelTFData（使用真实通道名称）
                tf_point: ChannelTFData = {
                    "ai_channel": ai_channel_name,
                    "ao_channel": ao_channel_name,
                    "amp_ratio": float(amp_ratio),
                    "phase_shift": phase_shift,
                }
                all_tf_points.append(tf_point)

                logger.debug(
                    f"AO通道 {ao_channel_name} -> AI通道 {ai_channel_name}: "
                    f"amp_ratio={amp_ratio:.6f}, phase_shift={phase_shift:.6f}rad"
                )

        logger.info(f"传递函数计算完成，共计算 {len(all_tf_points)} 个通道对")
        return all_tf_points

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

        该方法实现四阶段校准工作流程：
        1. 第一阶段：循环调用_single_calibrate，采集并保存原始数据
           - 多次执行独立校准（每次创建和销毁SingleChasCSIO对象）
           - 每次校准返回原始SweepData，立即保存为raw_sweep_data_N.pkl文件
        2. 第二阶段：处理SweepData，计算传递函数数据
           - 对每个SweepData进行平均和滤波处理
           - 对每个SweepData进行平均和滤波，再调用_calculate_tf_data计算TFData
        3. 第三阶段：平均所有校准结果
           - 对多个TFData进行平均，得到平均后的单个TFData
        4. 第四阶段：绘制图表和保存最终数据
           - 使用平均后的TFData绘制传递函数图
           - 保存最终的TFData文件

        Args:
            starts_num: 独立校准的次数，默认为10。
                这是提高校准精度的主要手段，建议设置更高的值。
            chunks_per_start: 每次启动采集的连续chunk数，默认为3
            apply_filter: 是否应用滤波，默认为True
            lowcut: 滤波器低频截止频率（Hz），默认为100.0
            highcut: 滤波器高频截止频率（Hz），默认为20000.0
            result_folder: 可选，结果保存文件夹路径。如果为None，将使用默认路径
                          'storage/calib/calib_result_fishnet'（相对于项目根目录）。
                          最终将保存多个raw_sweep_data_N.pkl文件、绘图和一个平均后的TFData文件
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
            # 使用默认路径：项目根目录的 storage/calib/calib_result_fishnet
            result_path = (
                Path(__file__).resolve().parents[3]
                / "storage"
                / "calib"
                / "calib_result_fishnet"
            )
            logger.info(f"未指定result_folder，使用默认路径: {result_path}")
        else:
            result_path = Path(result_folder)
            logger.info(f"使用指定的result_folder: {result_path}")

        # 创建结果文件夹
        result_path.mkdir(parents=True, exist_ok=True)

        # 存储所有starts的SweepData
        all_sweep_data_list: list[SweepData] = []

        # 执行多次独立校准，获取SweepData并立即保存
        logger.info("=" * 60)
        logger.info("第一阶段：循环调用_single_calibrate，采集并保存原始数据")
        logger.info("=" * 60)
        for calib_idx in range(starts_num):
            logger.info(f"开始第 {calib_idx + 1}/{starts_num} 次独立校准")

            # 调用_single_calibrate方法，获取SweepData
            sweep_data = self._single_calibrate(
                chunks_per_start=chunks_per_start,
                settle_time=settle_time,
            )

            # 保存这次校准的原始SweepData
            sweep_data_path = result_path / f"raw_sweep_data_{calib_idx + 1}.pkl"
            try:
                with open(sweep_data_path, "wb") as f:
                    pickle.dump(sweep_data, f)
                logger.info(
                    f"第 {calib_idx + 1} 次SweepData已保存到: {sweep_data_path}"
                )
            except Exception as e:
                logger.error(f"保存SweepData失败: {e}", exc_info=True)
                raise OSError(f"保存SweepData失败: {e}") from e

            # 将SweepData添加到列表中，供后续处理
            all_sweep_data_list.append(sweep_data)

        # 第二阶段：处理所有SweepData，计算TFData
        logger.info("=" * 60)
        logger.info("第二阶段：处理SweepData，计算传递函数数据")
        logger.info("=" * 60)

        # 存储所有starts的TFData
        all_tf_data_list: list[TFData] = []

        for calib_idx, sweep_data in enumerate(all_sweep_data_list):
            logger.info(f"处理第 {calib_idx + 1}/{starts_num} 次校准的数据")

            # 1. 对每个点的多个chunk进行平均
            logger.info("对每个点的多个chunk进行平均")
            averaged_data = average_sweep_data(sweep_data)

            # 2. 应用滤波（如果需要）
            if apply_filter:
                logger.info(f"应用带通滤波器: {lowcut}Hz - {highcut}Hz")
                filtered_data = filter_sweep_data(
                    averaged_data,
                    lowcut=lowcut,
                    highcut=highcut,
                )
            else:
                filtered_data = averaged_data

            # 3. 使用 _calculate_tf_data 计算传递函数
            tf_points = self._calculate_tf_data(filtered_data)

            # 计算平均幅值比和平均相位差
            amp_ratios = [tf["amp_ratio"] for tf in tf_points]
            phase_shifts = [tf["phase_shift"] for tf in tf_points]
            mean_amp_ratio = float(np.mean(amp_ratios))
            mean_phase_shift = float(np.mean(phase_shifts))

            # 构建TFData（符合新类型定义，包含 sampling_info，不含冗余通道列表字段）
            tf_data: TFData = {
                "tf_list": tf_points,
                "sampling_info": self._sampling_info,
                "sine_args": self._sine_args,
                "mean_amp_ratio": mean_amp_ratio,
                "mean_phase_shift": mean_phase_shift,
            }
            all_tf_data_list.append(tf_data)

            logger.info(
                f"第 {calib_idx + 1} 次校准数据处理完成，"
                f"频率={self._sine_args['frequency']:.2f}Hz, "
                f"平均幅值比={mean_amp_ratio:.6f}, "
                f"平均相位差={mean_phase_shift:.6f}rad"
            )

        # 第三阶段：平均所有TFData
        logger.info("=" * 60)
        logger.info("第三阶段：平均所有校准结果")
        logger.info("=" * 60)

        # 使用average_tf_data_list函数对所有TFData进行平均
        logger.info("对所有TFData进行平均")
        averaged_tf_data: TFData = average_tf_data_list(all_tf_data_list)

        # 输出平均元数据日志
        logger.info(
            f"平均元数据: 频率={averaged_tf_data['sine_args']['frequency']:.2f}Hz, "
            f"平均幅值比={averaged_tf_data['mean_amp_ratio']:.6f}, "
            f"平均相位差={averaged_tf_data['mean_phase_shift']:.6f}rad"
        )

        # 保存平均后的TFData到内部状态
        self._result_averaged_tf_data = averaged_tf_data

        # 第四阶段：绘制图表和保存最终数据
        logger.info("=" * 60)
        logger.info("第四阶段：绘制图表和保存最终数据")
        logger.info("=" * 60)

        # 绘制传递函数图
        logger.info("绘制传递函数图")
        plot_path = result_path / "transfer_functions.png"
        self.plot_tf_data(save_path=plot_path)
        logger.info(f"已保存传递函数图到: {plot_path}")

        # 保存最终的TFData
        final_tf_data_path = result_path / "tf_data.pkl"
        try:
            with open(final_tf_data_path, "wb") as f:
                pickle.dump(averaged_tf_data, f)
            logger.info(f"最终平均TFData已保存到: {final_tf_data_path}")
        except Exception as e:
            logger.error(f"保存最终TFData失败: {e}", exc_info=True)
            raise OSError(f"保存最终TFData失败: {e}") from e

        logger.info("=" * 60)
        logger.info(f"校准流程完成，所有结果已保存到: {result_path}")
        logger.info("=" * 60)

    def save_tf_data(self, file_path: str | Path) -> None:
        """
        保存校准结果到本地文件

        使用pickle序列化将最终传递函数数据保存到磁盘。

        Args:
            file_path: 保存文件的路径（支持字符串或Path对象）

        Raises:
            RuntimeError: 当尚未执行校准时
            IOError: 当文件保存失败时
        """
        if self._result_averaged_tf_data is None:
            raise RuntimeError("尚未执行校准，无法保存结果")

        # 转换为Path对象
        save_path = Path(file_path)

        # 确保父目录存在
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(save_path, "wb") as f:
                pickle.dump(self._result_averaged_tf_data, f)
            logger.info(f"校准结果已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存校准结果失败: {e}", exc_info=True)
            raise OSError(f"保存校准结果失败: {e}") from e

    @property
    def result_averaged_tf_data(self) -> TFData | None:
        """
        获取平均后的传递函数数据

        包含所有通道对平均后的传递函数数据。

        Returns:
            TFData对象，如果尚未校准则返回None
        """
        if self._result_averaged_tf_data is None:
            return None
        # 深拷贝以避免外部修改
        import copy

        return copy.deepcopy(self._result_averaged_tf_data)

    def plot_tf_data(
        self,
        save_path: str | Path | None = None,
    ) -> None:
        """
        绘制传递函数图

        绘制两幅直角坐标系下的彩色折线图：
        - 第一幅：幅值比 vs 通道序数差
        - 第二幅：相位差 vs 通道序数差

        通道序数差 = AI通道索引 - AO通道索引

        每个AO通道使用同一颜色绘制，数据点附带"AO索引→AI索引"标签。
        图例中显示索引与通道名称的对应关系。

        Args:
            save_path: 可选，图像保存路径。如果为None，则只显示不保存

        Raises:
            RuntimeError: 当尚未执行校准时
        """
        if self._result_averaged_tf_data is None:
            raise RuntimeError("尚未执行校准，无法绘制图表")

        tf_data = self._result_averaged_tf_data

        logger.info("开始绘制传递函数图")

        # 准备数据
        # 按AO通道名称分组
        # 格式: {ao_channel_name: [(ai_channel_name, ao_idx, ai_idx, channel_diff, amp_ratio, phase_shift), ...]}
        ao_groups: dict[str, list[tuple[str, int, int, int, float, float]]] = {}

        for tf_point in tf_data["tf_list"]:
            ao_ch = tf_point["ao_channel"]
            ai_ch = tf_point["ai_channel"]
            # 通过通道名查找索引（用于计算通道序数差）
            ao_idx = self._ao_channels.index(ao_ch) if ao_ch in self._ao_channels else 0
            ai_idx = self._ai_channels.index(ai_ch) if ai_ch in self._ai_channels else 0
            channel_diff = ai_idx - ao_idx  # 通道序数差
            amp_ratio = tf_point["amp_ratio"]
            phase_shift = tf_point["phase_shift"]

            if ao_ch not in ao_groups:
                ao_groups[ao_ch] = []
            ao_groups[ao_ch].append(
                (ai_ch, ao_idx, ai_idx, channel_diff, amp_ratio, phase_shift)
            )

        # 对每个AO组内的数据按通道序数差排序
        for ao_ch in ao_groups:
            ao_groups[ao_ch].sort(key=lambda x: x[3])  # 按channel_diff排序

        # 创建图表（两个子图）
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

        # 使用颜色映射
        colors = plt.cm.tab10(np.linspace(0, 1, len(ao_groups)))

        # 绘制第一幅图：幅值比 vs 通道序数差
        for color_idx, (ao_ch, data_points) in enumerate(sorted(ao_groups.items())):
            channel_diffs = [p[3] for p in data_points]
            amp_ratios = [p[4] for p in data_points]
            ao_indices = [p[1] for p in data_points]
            ai_indices = [p[2] for p in data_points]

            # 绘制折线和数据点
            ax1.plot(
                channel_diffs,
                amp_ratios,
                marker="o",
                markersize=8,
                linewidth=2,
                color=colors[color_idx],
                label=f"AO{ao_indices[0]}: {ao_ch}",
                alpha=0.8,
            )

            # 添加数据点标签
            for channel_diff, amp_ratio, ao_idx, ai_idx in zip(
                channel_diffs, amp_ratios, ao_indices, ai_indices, strict=True
            ):
                ax1.annotate(
                    f"{ao_idx}→{ai_idx}",
                    xy=(channel_diff, amp_ratio),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=7,
                    alpha=0.7,
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": colors[color_idx],
                        "alpha": 0.3,
                    },
                )

        ax1.set_xlabel("通道序数差 (AI索引 - AO索引)", fontsize=12)
        ax1.set_ylabel("幅值比", fontsize=12)
        ax1.set_title(
            f"传递函数幅值比分布\n频率: {self._sine_args['frequency']}Hz",
            fontsize=14,
            pad=15,
        )
        ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
        ax1.legend(loc="best", fontsize=8, ncol=2)

        # 绘制第二幅图：相位差 vs 通道序数差
        for color_idx, (ao_ch, data_points) in enumerate(sorted(ao_groups.items())):
            channel_diffs = [p[3] for p in data_points]
            phase_shifts = [p[5] for p in data_points]
            ao_indices = [p[1] for p in data_points]
            ai_indices = [p[2] for p in data_points]

            # 绘制折线和数据点
            ax2.plot(
                channel_diffs,
                phase_shifts,
                marker="o",
                markersize=8,
                linewidth=2,
                color=colors[color_idx],
                label=f"AO{ao_indices[0]}: {ao_ch}",
                alpha=0.8,
            )

            # 添加数据点标签
            for channel_diff, phase_shift, ao_idx, ai_idx in zip(
                channel_diffs, phase_shifts, ao_indices, ai_indices, strict=True
            ):
                ax2.annotate(
                    f"{ao_idx}→{ai_idx}",
                    xy=(channel_diff, phase_shift),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=7,
                    alpha=0.7,
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": colors[color_idx],
                        "alpha": 0.3,
                    },
                )

        ax2.set_xlabel("通道序数差 (AI索引 - AO索引)", fontsize=12)
        ax2.set_ylabel("相位差 (rad)", fontsize=12)
        ax2.set_title(
            f"传递函数相位差分布\n频率: {self._sine_args['frequency']}Hz",
            fontsize=14,
            pad=15,
        )
        ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
        ax2.legend(loc="best", fontsize=8, ncol=2)

        # 添加通道索引与名称对应关系的文本框
        # 在第一幅图的右侧添加AI通道信息
        ai_channel_info = "AI通道索引:\n"
        for idx, channel_name in enumerate(self._ai_channels):
            ai_channel_info += f"{idx}: {channel_name}\n"

        ax1.text(
            1.02,
            0.5,
            ai_channel_info,
            transform=ax1.transAxes,
            fontsize=8,
            verticalalignment="center",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "wheat", "alpha": 0.5},
        )

        plt.tight_layout()

        # 保存或显示
        if save_path is not None:
            save_path_obj = Path(save_path)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path_obj, dpi=300, bbox_inches="tight")
            logger.info(f"传递函数图已保存到: {save_path_obj}")

        plt.show()
        logger.info("传递函数图绘制完成")
