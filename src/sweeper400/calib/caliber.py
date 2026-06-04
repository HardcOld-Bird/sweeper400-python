"""
# 多通道校准模块

模块路径：`sweeper400.calib.clb`

包含用于多通道输出情形下各通道响应函数校准的类和函数。
"""

import time
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..analyze import (
    CompData,
    Point2D,
    PointSweepData,
    PositiveFloat,
    PositiveInt,
    SamplingInfo,
    SweepData,
    TFData,
    Waveform,
    average_comp_data_list,
    average_sweep_data,
    average_tf_data_list,
    comp_to_tf,
    extract_single_tone_information_vvi,
    filter_sweep_data,
    get_sine,
    init_comp_data,
    init_sampling_info,
    load_data_with_fallback,
    load_freq_optimizer_result,
    save_compressed_data,
    tf_to_comp,
    plot_sweep_data_as_single_waveform,
)
from ..logger import get_logger
from ..measure import SingleChasCSIO

# 获取模块日志器
logger = get_logger(__name__)


class CaliberAnemone:
    """
    # 多通道校准类（海葵模式）

    该类专门用于多AI通道情形下，各个通道响应一致性的校准（calibration）。
    "Anemone"（海葵）命名表示该类使用海葵型波导，波导上提供若干个传声器插槽，
    每个插槽处的声压幅值和相位都相同。通过同时采集所有插槽（AI通道）的数据，
    自动计算各通道间的差异并生成补偿数据。

    ## 主要特性：
        - 外部声源提供稳定的1000Hz正弦信号（无需Python生成测试信号）
        - 同时采集所有AI通道的多通道波形
        - 自动应用滤波和平均处理以提高准确度
        - 使用extract_single_tone_information_vvi函数提取各通道单频信息
        - 以所有通道复振幅的平均值作为理想真值
        - 生成AI通道补偿数据（CompData）用于后续测量中补偿通道差异
        - 将校准结果序列化保存到本地磁盘
        - 支持极坐标和直角坐标可视化模式

    ## 校准原理：
        1. 第一阶段：数据采集
           - 创建SingleChasCSIO对象（只使用虚拟AO通道输出静音）
           - 同时采集所有AI通道的多个连续chunk
           - 数据存储为SweepData格式（单个测量点）
        2. 第二阶段：数据处理
           - 对多个chunk进行平均和滤波
           - 使用extract_single_tone_information_vvi提取每个通道的复振幅
        3. 第三阶段：补偿计算
           - 计算所有AI通道复振幅的平均值（理想真值）
           - 构造TFData（每个通道相对于平均值的传递函数）
           - 使用tf_to_comp转换为CompData
           - 补偿参数基于绝对时间延迟（而非固定频率的相位差），
             因此可适用于后续不同频率的测量场景
        4. 第四阶段：绘图和保存
           - 绘制传递函数极坐标图和补偿数据直角坐标图
           - 绘制融合波形图
           - 保存最终的CompData文件

    ## 使用示例：
    ```python
    from sweeper400.use.clb import CaliberAnemone
    from sweeper400.analyze import init_sampling_info

    # 创建采样信息
    sampling_info = init_sampling_info(171500.0, 85750)  # 采样率171.5kHz, 0.5秒

    # 创建校准对象
    clb = CaliberAnemone(
        ai_channels=(
            "PXI1Slot2/ai0", "PXI1Slot2/ai1",
            "PXI1Slot3/ai0", "PXI1Slot3/ai1",
            "PXI1Slot4/ai0", "PXI1Slot4/ai1",
            "PXI1Slot5/ai0", "PXI1Slot5/ai1"
        ),
        sampling_info=sampling_info,
        frequency=1000.0,
    )

    # 执行校准（采集3个chunk）
    clb.calibrate(chunks_num=3)

    # 校准结果会自动保存到默认路径
    # 也可以手动保存到指定路径
    clb.save_comp_data("calibration_results.pkl")

    # 绘制极坐标图
    clb.plot_comp_data(mode="polar")

    # 绘制直角坐标图
    clb.plot_comp_data(mode="cartesian")
    ```

    ## 注意事项：
        - 校准前确保外部声源已开启并稳定输出1000Hz正弦信号
        - 建议在安静环境中进行校准以减少噪声干扰
        - 所有传声器应安装在波导的对称位置，确保物理真值一致
        - 该校准方法仅适用于单频工作点，但补偿参数（时间延迟）可跨频率复用
        - 相位差在校准频率（1000Hz）下计算，但存储为时间延迟，因此适用于其他频率
    """

    # 获取类日志器（类属性，所有实例共享）
    logger = get_logger(f"{__name__}.CaliberAnemone")

    def __init__(
        self,
        ai_channels: tuple[str, ...],
        sampling_info: SamplingInfo | None = None,
        frequency: float = 1000.0,
        ai_comp_data: str | Path | None = None,
    ) -> None:
        """
        初始化校准对象

        Args:
            ai_channels: AI 通道名称元组（例如 ("PXI1Slot2/ai0", "PXI1Slot3/ai0", ...)）。
                         海葵模式下同时采集所有通道的数据。
            sampling_info: 采样信息，包含采样率和采样点数。如果为None，将自动从
                          FrequencyOptimizer的存储结果中加载。
            frequency: 外部声源频率（Hz），默认为1000.0。如果为None，将自动从
                      FrequencyOptimizer的存储结果中加载。
            ai_comp_data: 可选，AI补偿数据文件路径。支持三级优先级：
                       1. 用户显式提供的路径（如果提供）
                       2. 默认路径 "storage/calib/calib_result_anemone/ai_comp_data.pkl"
                       3. 不使用补偿（如果都不存在）

        Raises:
            ValueError: 当参数无效时
            RuntimeError: 当无法获取frequency/sampling_info时
        """
        # 验证参数
        if not ai_channels:
            raise ValueError("AI 通道列表不能为空")

        # 当 sampling_info 为 None 时，使用默认值
        if sampling_info is None:
            # 根据1000Hz的预期频率设置
            sampling_info = init_sampling_info(200000, 40000)
            logger.info(
                f"sampling_info 使用默认值: "
                f"sampling_rate={sampling_info['sampling_rate']:.1f}Hz, "
                f"samples_num={sampling_info['samples_num']}"
            )

        if frequency <= 0:
            raise ValueError("频率必须为正数")

        # 保存配置参数
        self._ai_channels = ai_channels
        self._sampling_info = sampling_info
        self._frequency = frequency
        self._ai_comp_data_path = ai_comp_data

        # 推断虚拟AO通道（从第一个AI通道推断，用于SingleChasCSIO）
        self._dummy_ao_channel = self._infer_dummy_ao_channel(ai_channels[0])
        logger.info(f"推断虚拟AO通道: {self._dummy_ao_channel}")

        # 生成静音波形作为虚拟AO输出
        # SingleChasCSIO要求至少一个AO通道，但海葵模式不需要Python输出信号
        samples_num = sampling_info["samples_num"]
        sampling_rate = sampling_info["sampling_rate"]
        silence_data = np.zeros((1, samples_num), dtype=np.float64)
        self._silence_waveform = Waveform(
            input_array=silence_data,
            sampling_rate=sampling_rate,
            channel_names=(self._dummy_ao_channel,),
            timestamp=np.datetime64("now", "ns"),
        )

        # 计算默认的稳定等待时间（chunk时长 + 0.1秒）
        chunk_duration = self._silence_waveform.duration
        self._default_settle_time = chunk_duration + 0.1

        # 内部状态变量（用于校准过程）
        self._current_channel_data: list[Waveform] = []
        self._target_chunks: int = 0
        self._chunk_collection_complete: bool = False

        # 校准结果存储
        self._result_tf_data: TFData | None = None
        self._result_comp_data: CompData | None = None

        logger.info(
            f"CaliberAnemone 实例已创建 - "
            f"AI通道数: {len(ai_channels)}, "
            f"频率: {frequency}Hz, "
            f"采样率: {sampling_info['sampling_rate']}Hz, "
            f"默认稳定时间: {self._default_settle_time:.3f}s, "
            f"使用补偿数据: {ai_comp_data is not None}"
        )

    @staticmethod
    def _infer_dummy_ao_channel(ai_channel: str) -> str:
        """
        从AI通道名推断对应的AO通道名

        例如："PXI1Slot2/ai0" -> "PXI1Slot2/ao0"

        Args:
            ai_channel: AI通道名称

        Returns:
            推断的AO通道名称
        """
        # 尝试将ai替换为ao
        ao_channel = ai_channel.replace("/ai", "/ao")
        if ao_channel == ai_channel:
            # 如果替换失败，尝试其他模式
            ao_channel = ai_channel.replace("ai", "ao", 1)
        return ao_channel

    def _export_function(
        self,
        ai_waveform: Waveform,
        ao_static_waveform: Waveform,
        ao_feedback_waveform: Waveform | None,
        chunks_num: PositiveInt,
    ) -> None:
        """
        数据导出回调函数

        将采集到的AI波形添加到当前通道的数据列表中。

        Args:
            ai_waveform: 采集到的AI波形（多通道）
            ao_static_waveform: 当前的静态输出波形（静音）
            ao_feedback_waveform: 当前的反馈输出波形（None）
            chunks_num: 数据块编号（从1开始）
        """
        if hasattr(self, "_current_channel_data"):
            self._current_channel_data.append(ai_waveform)
            logger.debug(f"采集到第 {chunks_num} 段数据")

            if (
                hasattr(self, "_target_chunks")
                and len(self._current_channel_data) >= self._target_chunks
            ):
                self._chunk_collection_complete = True

    def calibrate(
        self,
        chunks_num: int = 3,
        apply_filter: bool = True,
        lowcut: float = 100.0,
        highcut: float = 20000.0,
        result_folder: str | Path | None = None,
        settle_time: float | None = None,
    ) -> None:
        """
        执行校准流程

        该方法实现四阶段校准工作流程：
        1. 第一阶段：数据采集
           - 创建SingleChasCSIO对象（虚拟AO通道输出静音）
           - 同时采集所有AI通道的多个连续chunk
           - 保存原始SweepData到raw_sweep_data.pkl
        2. 第二阶段：数据处理
           - 对多个chunk进行平均和滤波
           - 使用extract_single_tone_information_vvi提取每个通道的复振幅
        3. 第三阶段：补偿计算
           - 计算所有AI通道复振幅的平均值（理想真值）
           - 构造TFData并转换为CompData
        4. 第四阶段：绘图和保存
           - 绘制polar和cartesian模式绘图
           - 绘制融合波形图
           - 保存最终的CompData文件

        Args:
            chunks_num: 采集的连续chunk数，默认为3
            apply_filter: 是否应用滤波，默认为True
            lowcut: 滤波器低频截止频率（Hz），默认为100.0
            highcut: 滤波器高频截止频率（Hz），默认为20000.0
            result_folder: 可选，结果保存文件夹路径。如果为None，将使用默认路径
                          'storage/calib/calib_result_anemone'（相对于项目根目录）。
                          最终将保存raw_sweep_data.pkl、三幅绘图和一个CompData文件
            settle_time: 采集开始前的稳定等待时间（秒）。如果为None，则使用初始化时
                        计算的默认值（chunk时长 + 0.1秒）

        Raises:
            ValueError: 当参数无效时
        """
        if chunks_num < 1:
            raise ValueError("chunk数必须至少为1")

        actual_settle_time = (
            settle_time if settle_time is not None else self._default_settle_time
        )

        logger.info(
            f"开始校准流程 - "
            f"AI通道数: {len(self._ai_channels)}, "
            f"chunk数: {chunks_num}, "
            f"稳定时间: {actual_settle_time:.3f}s"
        )

        # 确定结果保存路径
        if result_folder is None:
            result_path = (
                Path(__file__).resolve().parents[3]
                / "storage"
                / "calib"
                / "calib_result_anemone"
            )
            logger.info(f"未指定result_folder，使用默认路径: {result_path}")
        else:
            result_path = Path(result_folder)
            logger.info(f"使用指定的result_folder: {result_path}")

        result_path.mkdir(parents=True, exist_ok=True)

        # 第一阶段：数据采集
        logger.info("=" * 60)
        logger.info("第一阶段：数据采集")
        logger.info("=" * 60)

        sweep_data: SweepData = {
            "ai_data_list": [],
            "ao_data": self._silence_waveform,
        }

        # 创建测量点数据
        point_data: PointSweepData = {
            "position": Point2D(x=0.0, y=0.0),
            "ai_data": [],
        }
        sweep_data["ai_data_list"].append(point_data)

        # 创建SingleChasCSIO对象
        sync_io = SingleChasCSIO(
            ai_channels=self._ai_channels,
            ao_channels_static=(self._dummy_ao_channel,),
            ao_channels_feedback=(),
            static_output_waveform=self._silence_waveform,
            export_function=self._export_function,
            ai_comp_data=self._ai_comp_data_path,
        )

        try:
            sync_io.start()
            logger.info("SingleChasCSIO任务已启动")

            # 等待系统初始化稳定
            time.sleep(actual_settle_time)

            # 初始化采集控制变量
            self._current_channel_data = []
            self._target_chunks = chunks_num
            self._chunk_collection_complete = False

            # 启用数据导出
            sync_io.enable_export = True

            # 等待采集指定数量的chunk
            chunk_duration = self._silence_waveform.duration
            max_wait_time = chunk_duration * chunks_num * 2.0
            poll_interval = 0.05
            elapsed_time = 0.0

            logger.debug(f"开始采集 {chunks_num} 个chunk")
            while (
                not self._chunk_collection_complete and elapsed_time < max_wait_time
            ):
                time.sleep(poll_interval)
                elapsed_time += poll_interval

            # 禁用数据导出
            sync_io.enable_export = False

            collected_chunks = len(self._current_channel_data)
            logger.info(f"采集完成，共 {collected_chunks} 个chunk")

            if collected_chunks != chunks_num:
                logger.warning(
                    f"预期采集 {chunks_num} 个chunk，"
                    f"实际采集 {collected_chunks} 个chunk"
                )

            # 将采集到的数据添加到测量点
            point_data["ai_data"].extend(self._current_channel_data)

        finally:
            sync_io.stop()
            logger.info("SingleChasCSIO任务已停止")

        logger.info("数据采集阶段完成")

        # 保存原始SweepData
        raw_sweep_data_path = result_path / "raw_sweep_data.pkl"
        try:
            save_compressed_data(sweep_data, raw_sweep_data_path, 6, "SweepData")
            logger.info(f"原始SweepData已保存到: {raw_sweep_data_path}")
        except Exception as e:
            logger.error(f"保存原始SweepData失败: {e}", exc_info=True)
            raise OSError(f"保存原始SweepData失败: {e}") from e

        # 第二阶段：数据处理
        logger.info("=" * 60)
        logger.info("第二阶段：数据处理")
        logger.info("=" * 60)

        # 1. 对多个chunk进行平均
        logger.info("对多个chunk进行平均")
        averaged_sweep_data = average_sweep_data(sweep_data)

        # 2. 应用滤波（如果需要）
        if apply_filter:
            logger.info(f"应用带通滤波器: {lowcut}Hz - {highcut}Hz")
            filtered_sweep_data = filter_sweep_data(
                averaged_sweep_data,
                lowcut=lowcut,
                highcut=highcut,
            )
        else:
            filtered_sweep_data = averaged_sweep_data

        # 3. 绘制融合波形图
        logger.info("绘制融合波形图")
        fusion_plot_path = result_path / "fusion_waveform.png"
        try:
            from ..analyze.plot import plot_sweep_data_as_single_waveform

            fig, _ = plot_sweep_data_as_single_waveform(
                sweep_data=filtered_sweep_data,
                save_path=str(fusion_plot_path),
                zoom_factor=200,
            )
            plt.close(fig)
            logger.info(f"融合波形图已保存到: {fusion_plot_path}")
        except Exception as e:
            logger.error(f"绘制融合波形图时发生错误: {e}", exc_info=True)

        # 4. 提取多通道复振幅
        logger.info("提取多通道单频信息")
        ai_waveform = filtered_sweep_data["ai_data_list"][0]["ai_data"][0]
        result_wf = extract_single_tone_information_vvi(
            ai_waveform,
            approx_freq=self._frequency,
            precise_mode=True,
        )
        complex_amplitudes = result_wf.channel_complex_amplitudes

        # 记录各通道复振幅
        for idx, ai_channel in enumerate(self._ai_channels):
            amp = np.abs(complex_amplitudes[idx])
            phase = np.angle(complex_amplitudes[idx])
            logger.info(
                f"AI通道 {ai_channel}: "
                f"幅值={amp:.6f}, 相位={phase:.6f}rad"
            )

        # 第三阶段：补偿计算
        logger.info("=" * 60)
        logger.info("第三阶段：补偿计算")
        logger.info("=" * 60)

        # 计算所有通道复振幅的平均值（理想真值）
        mean_complex_amp = np.mean(complex_amplitudes)
        logger.info(
            f"平均复振幅: 幅值={np.abs(mean_complex_amp):.6f}, "
            f"相位={np.angle(mean_complex_amp):.6f}rad"
        )

        # 计算每个通道相对于平均值的传递函数
        # TF_i = A_i / A_mean
        tf_complex = complex_amplitudes / mean_complex_amp

        # 计算均值（用于TFData，理论上应接近1和0）
        mean_amp_ratio = float(np.mean(np.abs(tf_complex)))
        mean_phase_shift = float(np.mean(np.angle(tf_complex)))

        # 构建TFData（列矩阵：N行1列）
        tf_df = pd.DataFrame(
            tf_complex.reshape(-1, 1),
            index=list(self._ai_channels),
            columns=["mean_ref"],
        )

        tf_data: TFData = {
            "tf_dataframe": tf_df,
            "sampling_info": self._sampling_info,
            "frequency": self._frequency,
            "mean_amp_ratio": mean_amp_ratio,
            "mean_phase_shift": mean_phase_shift,
        }
        self._result_tf_data = tf_data

        # 将TFData转换为CompData
        comp_data = tf_to_comp(tf_data)

        # 记录补偿数据
        for ai_channel in self._ai_channels:
            amp_multiplier = comp_data["comp_dataframe"].loc[
                ai_channel, "amp_multiplier"
            ]
            time_increment = comp_data["comp_dataframe"].loc[
                ai_channel, "time_increment"
            ]
            logger.info(
                f"AI通道 {ai_channel}: "
                f"amp_multiplier={amp_multiplier:.6f}, "
                f"time_increment={time_increment * 1e6:.3f}μs"
            )

        self._result_comp_data = comp_data

        # 第四阶段：绘图和保存
        logger.info("=" * 60)
        logger.info("第四阶段：绘图和保存")
        logger.info("=" * 60)

        # 绘制polar模式绘图
        logger.info("绘制传递函数极坐标图")
        polar_plot_path = result_path / "transfer_function_polar.png"
        self.plot_comp_data(mode="polar", save_path=polar_plot_path)
        logger.info(f"已保存polar模式绘图到: {polar_plot_path}")

        # 绘制cartesian模式绘图
        logger.info("绘制补偿数据直角坐标图")
        cartesian_plot_path = result_path / "compensation_cartesian.png"
        self.plot_comp_data(mode="cartesian", save_path=cartesian_plot_path)
        logger.info(f"已保存cartesian模式绘图到: {cartesian_plot_path}")

        # 保存最终的CompData
        final_comp_data_path = result_path / "ai_comp_data.pkl"
        try:
            save_compressed_data(comp_data, final_comp_data_path, 6, "CompData")
            logger.info(f"最终CompData已保存到: {final_comp_data_path}")
        except Exception as e:
            logger.error(f"保存最终CompData失败: {e}", exc_info=True)
            raise OSError(f"保存最终CompData失败: {e}") from e

        logger.info("=" * 60)
        logger.info(f"校准流程完成，所有结果已保存到: {result_path}")
        logger.info("=" * 60)

    def save_comp_data(
        self,
        save_path: str | Path,
        compress_level: int = 6,
    ) -> None:
        """
        保存校准结果到本地文件（使用gzip压缩）

        Args:
            save_path: 保存文件的路径
            compress_level: gzip压缩级别（0-9），默认6

        Raises:
            RuntimeError: 当尚未执行校准时
            IOError: 当文件保存失败时
        """
        if self._result_comp_data is None:
            raise RuntimeError("尚未执行校准，无法保存结果")

        from ..analyze.post_process import save_compressed_data

        save_compressed_data(
            self._result_comp_data, save_path, compress_level, "CompData"
        )

    def plot_comp_data(
        self,
        mode: Literal["polar", "cartesian"],
        save_path: str | Path | None = None,
    ) -> None:
        """
        绘制传递函数或补偿数据

        支持两种绘图模式：
        - polar（极坐标）: 使用TFData绘制传递函数极坐标图
        - cartesian（直角坐标）: 使用CompData绘制补偿数据直角坐标图

        Args:
            mode: 绘图模式，"polar"为极坐标图，"cartesian"为直角坐标图
            save_path: 可选，图像保存路径。如果为None，则只显示不保存

        Raises:
            RuntimeError: 当尚未执行校准时
        """
        if mode == "polar":
            if self._result_tf_data is None:
                raise RuntimeError("尚未执行校准，无法绘制极坐标图")

            tf_data = self._result_tf_data
            tf_df = tf_data["tf_dataframe"]
            tf_complex = tf_df.iloc[:, 0].values

            amp_ratios = np.abs(tf_complex).tolist()
            phase_shifts = np.angle(tf_complex).tolist()

            ai_channel_names = tf_df.index.tolist()
            channel_indices = [
                self._ai_channels.index(ai_ch) if ai_ch in self._ai_channels else 0
                for ai_ch in ai_channel_names
            ]

            logger.info("开始绘制传递函数极坐标图")
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="polar")

            angles = phase_shifts

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

            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)

            ax.set_title(
                f"传递函数极坐标分布\n"
                f"频率: {self._frequency}Hz, "
                f"通道数: {len(self._ai_channels)}",
                fontsize=14,
                pad=20,
            )

            cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label("通道索引", fontsize=10)

            cbar.set_ticks(range(len(self._ai_channels)))
            tick_labels = [
                f"{idx}: {self._ai_channels[idx]}"
                for idx in range(len(self._ai_channels))
            ]
            cbar.set_ticklabels(tick_labels)
            cbar.ax.tick_params(labelsize=8)

        elif mode == "cartesian":
            if self._result_comp_data is None:
                raise RuntimeError("尚未执行校准，无法绘制直角坐标图")

            comp_data = self._result_comp_data
            comp_df = comp_data["comp_dataframe"]

            amp_multipliers = comp_df["amp_multiplier"].values.tolist()
            time_increments = comp_df["time_increment"].values.tolist()

            ai_channel_names = comp_df.index.tolist()
            channel_indices = [
                self._ai_channels.index(ai_ch) if ai_ch in self._ai_channels else 0
                for ai_ch in ai_channel_names
            ]

            logger.info("开始绘制补偿数据直角坐标图")
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111)

            scatter = ax.scatter(
                time_increments,
                amp_multipliers,
                c=channel_indices,
                cmap="viridis",
                s=150,
                alpha=0.8,
                edgecolors="black",
                linewidths=2,
                zorder=3,
            )

            for time_increment, amp_multiplier, channel_idx in zip(
                time_increments, amp_multipliers, channel_indices, strict=True
            ):
                channel_name = self._ai_channels[channel_idx]
                label = channel_name

                ax.annotate(
                    label,
                    xy=(time_increment, amp_multiplier),
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

            ax.set_xlabel("时间延迟补偿 (s)", fontsize=12)
            ax.set_ylabel("幅值补偿倍率", fontsize=12)
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)

            ax.set_title(
                f"补偿数据直角坐标分布\n"
                f"频率: {self._frequency}Hz, "
                f"通道数: {len(self._ai_channels)}",
                fontsize=14,
                pad=20,
            )

            cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
            cbar.set_label("通道索引", fontsize=10)

            cbar.set_ticks(range(len(self._ai_channels)))
            tick_labels = [
                f"{idx}: {self._ai_channels[idx]}"
                for idx in range(len(self._ai_channels))
            ]
            cbar.set_ticklabels(tick_labels)
            cbar.ax.tick_params(labelsize=8)

            # 自适应坐标轴范围
            time_increment_array = np.array(time_increments)
            amp_multiplier_array = np.array(amp_multipliers)

            ti_min, ti_max = time_increment_array.min(), time_increment_array.max()
            am_min, am_max = amp_multiplier_array.min(), amp_multiplier_array.max()

            ti_margin = (ti_max - ti_min) * 0.1 if ti_max > ti_min else 1e-6
            am_margin = (am_max - am_min) * 0.1 if am_max > am_min else 0.01

            ax.set_xlim(ti_min - ti_margin, ti_max + ti_margin)
            ax.set_ylim(am_min - am_margin, am_max + am_margin)

        else:
            raise ValueError(f"不支持的绘图模式: {mode}")

        plt.tight_layout()

        if save_path is not None:
            save_path_obj = Path(save_path)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path_obj, dpi=300, bbox_inches="tight")
            logger.info(f"图像已保存到: {save_path_obj}")

        plt.show()
        logger.info("绘图完成")

    @property
    def result_tf_data(self) -> TFData | None:
        """
        获取传递函数数据

        Returns:
            TFData对象，如果尚未校准则返回None
        """
        if self._result_tf_data is None:
            return None
        import copy

        return copy.deepcopy(self._result_tf_data)

    @property
    def result_comp_data(self) -> CompData | None:
        """
        获取补偿数据

        Returns:
            CompData对象，如果尚未校准则返回None
        """
        if self._result_comp_data is None:
            return None
        import copy

        return copy.deepcopy(self._result_comp_data)

    @property
    def ai_channels(self) -> tuple[str, ...]:
        """
        获取AI通道列表

        Returns:
            AI通道名称元组
        """
        return self._ai_channels

    @property
    def frequency(self) -> float:
        """
        获取校准频率

        Returns:
            校准频率（Hz）
        """
        return self._frequency


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
    from sweeper400.use.clb import CaliberOctopus
    from sweeper400.analyze import init_sampling_info

    # 创建采样信息
    sampling_info = init_sampling_info(171500.0, 85750)  # 采样率171.5kHz, 0.5秒

    # 创建校准对象
    clb = CaliberOctopus(
        ai_channels=("PXI1Slot2/ai0",),
        ao_channels=(
            "PXI2Slot2/ao0", "PXI2Slot2/ao1",
            "PXI2Slot3/ao0", "PXI2Slot3/ao1",
            "PXI3Slot2/ao0", "PXI3Slot2/ao1",
            "PXI3Slot3/ao0", "PXI3Slot3/ao1"
        ),
        sampling_info=sampling_info,
        frequency=3430.0,
        amplitude=0.01,
    )

    # 执行校准（10次独立校准，每次4个chunk）
    # 可选：指定settle_time参数来覆盖默认值
    clb.calibrate(starts_num=10, chunks_per_start=4)

    # 校准结果会自动保存到默认路径
    # 也可以手动保存到指定路径
    clb.save_comp_data("calibration_results.pkl")

    # 绘制极坐标图
    clb.plot_comp_data(mode="polar")

    # 校准验证：使用已有补偿数据进行二次校准
    caliber_verify = CaliberOctopus(
        ai_channels=("PXI1Slot2/ai0",),
        ao_channels=(...),  # 与上面相同
        sampling_info=sampling_info,
        frequency=3430.0,
        amplitude=0.01,
        ao_comp_data="calibration_results.pkl"  # 使用之前的校准结果
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

    # 获取类日志器（类属性，所有实例共享）
    logger = get_logger(f"{__name__}.CaliberOctopus")

    def __init__(
        self,
        ai_channels: tuple[str, ...],
        ao_channels: tuple[str, ...],
        sampling_info: SamplingInfo | None = None,
        frequency: PositiveFloat | None = None,
        amplitude: PositiveFloat = 0.01,
        ai_comp_data: str | Path | None = None,
        ao_comp_data: str | Path | None = None,
    ) -> None:
        """
        初始化校准对象

        Args:
            ai_channels: AI 通道名称元组（例如 ("PXI2Slot2/ai0",)）。
                         章鱼模式下通常只使用第一个元素，其余元素将被忽略。
            ao_channels: AO 通道名称元组（例如 ("PXI2Slot2/ao0", "PXI2Slot2/ao1", ...)）
            sampling_info: 采样信息，包含采样率和采样点数。如果为None，将自动从
                          FrequencyOptimizer的存储结果中加载。
            frequency: 正弦波频率（Hz）。如果为None，将自动从
                      FrequencyOptimizer的存储结果中加载。
            amplitude: 正弦波幅值（V），默认0.01
            ai_comp_data: 可选，AI补偿数据文件路径，将传递给CSIO。
            ao_comp_data: 可选，AO补偿数据文件路径，将传递给CSIO。

        Raises:
            ValueError: 当参数无效时
            FileNotFoundError: 当用户显式提供路径但文件不存在时
            RuntimeError: 当补偿数据文件加载失败时或无法获取frequency/sampling_info时
        """
        # 验证参数
        if not ai_channels:
            raise ValueError("AI 通道列表不能为空")
        if not ao_channels:
            raise ValueError("AO 通道列表不能为空")

        # 当 sampling_info 或 frequency 为 None 时，从 FrequencyOptimizer 结果加载
        if sampling_info is None or frequency is None:
            freq_result = load_freq_optimizer_result()
            if freq_result is None:
                raise RuntimeError(
                    "sampling_info 或 frequency 为 None，但无法从 "
                    "'storage/calib/calib_result_freq/freq_result.pkl' 加载默认值。"
                    "请先运行 FrequencyOptimizer 或显式提供这些参数。"
                )
            if sampling_info is None:
                sampling_info = freq_result["sampling_info"]
                logger.info(
                    f"sampling_info 从存储结果加载: "
                    f"sampling_rate={sampling_info['sampling_rate']:.1f}Hz, "
                    f"samples_num={sampling_info['samples_num']}"
                )
            if frequency is None:
                frequency = freq_result["frequency"]
                logger.info(f"frequency 从存储结果加载: {frequency:.2f}Hz")

        # 保存配置参数
        self._ai_channels = ai_channels  # 章鱼模式下默认使用第一个元素
        self._ao_channels = ao_channels
        self._sampling_info = sampling_info
        self._frequency = frequency
        self._amplitude = amplitude
        self._ai_comp_data_path = ai_comp_data
        self._ao_comp_data_path = ao_comp_data

        # 生成同步多通道波形（所有通道相同的复振幅）
        cca = np.full(
            len(ao_channels),
            amplitude,
            dtype=np.complex128,
        )
        self._output_waveform = get_sine(
            sampling_info=sampling_info,
            frequency=frequency,
            channel_names=ao_channels,
            channel_complex_amplitudes=cca,
            full_cycle=True,
        )

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
            f"频率: {frequency}Hz, "
            f"幅值: {amplitude}V, "
            f"采样率: {sampling_info['sampling_rate']}Hz, "
            f"默认稳定时间: {self._default_settle_time:.3f}s"
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
            channel_names=self._ao_channels,
            sampling_rate=self._sampling_info["sampling_rate"],
            timestamp=self._output_waveform.timestamp,
            waveform_id=self._output_waveform.waveform_id,
            frequency=self._frequency,
            channel_complex_amplitudes=self._output_waveform.channel_complex_amplitudes,
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
            RuntimeError: 当 self._frequency 为 None 时
            ValueError: 当测量点数量与 AO 通道数不一致时
        """
        ao_frequency = self._frequency
        ao_amplitude = self._amplitude
        ao_phase = 0.0
        num_ao_channels = len(self._ao_channels)

        if len(filtered_sweep_data["ai_data_list"]) != num_ao_channels:
            raise ValueError(
                f"SweepData中的测量点数({len(filtered_sweep_data['ai_data_list'])}) "
                f"与AO通道数({num_ao_channels})不一致"
            )

        # 第一步：计算每个通道对的传递函数
        # 准备存储传递函数数据的列表（临时用于计算）
        amp_ratios_list = []
        phase_shifts_list = []
        ao_channel_names_list = []

        for channel_idx, point_data in enumerate(filtered_sweep_data["ai_data_list"]):
            ao_channel_name = self._ao_channels[channel_idx]
            ai_waveform = point_data["ai_data"][0]  # 已平均的单通道 AI 波形

            # 提取 AI 信号的正弦波参数（单通道波形，返回记录了结果的Waveform）
            result_wf = extract_single_tone_information_vvi(
                ai_waveform,
                approx_freq=ao_frequency,
            )

            # 计算传递函数（使用复振幅的模长和相角）
            ai_complex_amps = result_wf.channel_complex_amplitudes
            ai_amplitude = float(np.abs(ai_complex_amps[0]))
            ai_phase = float(np.angle(ai_complex_amps[0]))
            amp_ratio = ai_amplitude / ao_amplitude
            phase_shift = ai_phase - ao_phase
            # 将相位差归一化到 [-π, π] 区间
            phase_shift = float(np.arctan2(np.sin(phase_shift), np.cos(phase_shift)))

            # 保存到列表
            amp_ratios_list.append(float(amp_ratio))
            phase_shifts_list.append(phase_shift)
            ao_channel_names_list.append(ao_channel_name)

            logger.debug(
                f"AO通道 {ao_channel_name} -> AI通道 {self._ai_channels[0]}: "
                f"amp_ratio={amp_ratio:.6f}, phase_shift={phase_shift:.6f}rad"
            )

        # 第二步：计算所有通道对的平均幅值比和平均相位差
        mean_amp_ratio = float(np.mean(amp_ratios_list))
        mean_phase_shift = float(np.mean(phase_shifts_list))

        logger.debug(
            f"平均幅值比={mean_amp_ratio:.6f}, 平均相位差={mean_phase_shift:.6f}rad"
        )

        # 第三步：构建复数传递函数
        tf_complex_list = [
            amp * np.exp(1j * phase)
            for amp, phase in zip(amp_ratios_list, phase_shifts_list, strict=True)
        ]

        # 构建TFData的DataFrame（行矩阵：1行N列）
        # 对于CaliberOctopus，AI通道恒定（只用第一个AI通道），AO通道变化
        # 因此DataFrame是行矩阵：index为固定的AI通道名，columns为AO通道名
        tf_df = pd.DataFrame(
            [tf_complex_list],  # 行矩阵：1行N列，需要用列表包装
            index=[self._ai_channels[0]],  # 行索引：AI通道名（只有一行）
            columns=ao_channel_names_list,  # 列索引：AO通道名
        )

        # 构建TFData
        tf_data: TFData = {
            "tf_dataframe": tf_df,
            "sampling_info": self._sampling_info,
            "frequency": ao_frequency,
            "mean_amp_ratio": mean_amp_ratio,
            "mean_phase_shift": mean_phase_shift,
        }

        # 第四步：将TFData转换为CompData（TFData是行矩阵，符合要求）
        comp_data = tf_to_comp(tf_data)

        # 记录补偿数据（用于调试）
        for ao_channel in ao_channel_names_list:
            amp_multiplier = comp_data["comp_dataframe"].loc[
                ao_channel, "amp_multiplier"
            ]
            time_increment = comp_data["comp_dataframe"].loc[
                ao_channel, "time_increment"
            ]
            logger.debug(
                f"AO通道 {ao_channel}: "
                f"amp_multiplier={amp_multiplier:.6f}, "
                f"time_increment={time_increment * 1e6:.3f}μs"
            )

        logger.info(
            f"补偿数据计算完成，共 {len(ao_channel_names_list)} 个通道对，"
            f"频率={ao_frequency:.2f}Hz, "
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
        # 使用所有AI通道（CaliberOctopus通常只有1个，CaliberFishNet有多个）
        sync_io = SingleChasCSIO(
            ai_channels=self._ai_channels,  # 使用所有AI通道
            ao_channels_static=self._ao_channels,
            ao_channels_feedback=(),  # 校准不使用反馈通道
            static_output_waveform=self._output_waveform,
            export_function=self._export_function,
            ai_comp_data=self._ai_comp_data_path,
            ao_comp_data=self._ao_comp_data_path,
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
                save_compressed_data(sweep_data, sweep_data_path, 6, "SweepData")
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
                lowcut = self._frequency * 0.5
                highcut = self._frequency * 2.0
                logger.info(f"应用带通滤波器: {lowcut}Hz - {highcut}Hz")
                filtered_data = filter_sweep_data(
                    averaged_data,
                    lowcut=lowcut,
                    highcut=highcut,
                    trim_samples=300,  # 约6个周期
                )
            else:
                filtered_data = averaged_data

            # 3. 绘制滤波后数据的融合画像
            logger.info("绘制滤波后数据的融合画像")
            fusion_plot_filename = f"fusion_waveform_{calib_idx + 1}.png"
            fusion_plot_path = result_path / fusion_plot_filename

            try:
                fig, _ = plot_sweep_data_as_single_waveform(
                    sweep_data=filtered_data,
                    save_path=str(fusion_plot_path),
                    zoom_factor=50,
                )
                # 关闭图形以释放内存
                import matplotlib.pyplot as plt
                plt.close(fig)

                logger.info(f"融合画像已保存到: {fusion_plot_path}")
            except Exception as e:
                logger.error(
                    f"绘制第 {calib_idx + 1} 次校准的融合画像时发生错误: {e}",
                    exc_info=True,
                )
                # 继续处理，不中断整个校准流程

            # 4. 使用类内部方法 _calculate_comp_data 计算补偿数据
            logger.info("计算补偿数据")
            comp_data = self._calculate_comp_data(filtered_data)
            all_comp_data_list.append(comp_data)

            logger.info(
                f"第 {calib_idx + 1} 次校准数据处理完成，"
                f"频率={comp_data['frequency']:.2f}Hz, "
                f"平均幅值比={comp_data['mean_amp_ratio']:.6f}"
            )

        # 从all_comp_data_list构建包含所有starts数据的CompData（用于cartesian模式绘图）
        # 将所有CompData的DataFrame纵向堆叠（所有starts的数据）
        all_raw_comp_dfs = [
            comp_data["comp_dataframe"] for comp_data in all_comp_data_list
        ]

        # 纵向拼接所有DataFrame（不需要去重，保留所有starts的数据）
        all_raw_comp_df = pd.concat(all_raw_comp_dfs, axis=0)

        # 使用第一个CompData的元数据创建包含所有starts数据的CompData
        self._result_raw_comp_data: CompData = init_comp_data(
            comp_dataframe=all_raw_comp_df,
            sampling_info=all_comp_data_list[0]["sampling_info"],
            frequency=all_comp_data_list[0]["frequency"],
            mean_amp_ratio=all_comp_data_list[0]["mean_amp_ratio"],
            mean_phase_shift=all_comp_data_list[0]["mean_phase_shift"],
        )

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
            if len(comp_data["comp_dataframe"]) != channels_num:
                raise RuntimeError(
                    f"第 {idx + 1} 次校准的通道数({len(comp_data['comp_dataframe'])}) "
                    f"与预期({channels_num})不一致"
                )

        # 使用average_comp_data_list函数对所有CompData进行平均
        logger.info("对所有CompData进行平均")
        averaged_comp_data: CompData = average_comp_data_list(all_comp_data_list)

        # 输出每个通道的平均结果日志
        for ao_channel in self._ao_channels:
            avg_amp_multiplier = averaged_comp_data["comp_dataframe"].loc[
                ao_channel, "amp_multiplier"
            ]
            avg_time_increment = averaged_comp_data["comp_dataframe"].loc[
                ao_channel, "time_increment"
            ]
            logger.info(
                f"AO通道 {ao_channel}: "
                f"平均幅值补偿倍率={avg_amp_multiplier:.6f}, "
                f"平均时间延迟补偿={avg_time_increment * 1e6:.3f}μs"
            )

        # 输出平均元数据日志
        logger.info(
            f"平均元数据: 频率={averaged_comp_data['frequency']:.2f}Hz, "
            f"平均幅值比={averaged_comp_data['mean_amp_ratio']:.6f}, "
            f"平均相位差={averaged_comp_data['mean_phase_shift']:.6f}rad"
        )

        # 使用comp_to_tf将平均后的CompData转换为TFData
        logger.info("将平均后的CompData转换为TFData")
        averaged_tf_data = comp_to_tf(averaged_comp_data)

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

        # 创建最终的CompData（使用DataFrame结构）
        final_comp_data: CompData = {
            "comp_dataframe": averaged_comp_data["comp_dataframe"],
            "sampling_info": self._sampling_info,
            "frequency": averaged_comp_data["frequency"],
            "mean_amp_ratio": averaged_comp_data["mean_amp_ratio"],
            "mean_phase_shift": averaged_comp_data["mean_phase_shift"],
        }

        # 保存最终的CompData
        final_comp_data_path = result_path / "ao_comp_data.pkl"
        try:
            save_compressed_data(
                final_comp_data, final_comp_data_path, 6, "CompData"
            )
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

    def save_comp_data(
        self,
        save_path: str | Path,
        compresslevel: int = 6,
    ) -> None:
        """
        保存校准结果到本地文件（使用gzip压缩）

        将最终补偿数据序列化并使用gzip压缩保存到磁盘。

        Args:
            save_path: 保存文件的路径（支持字符串或Path对象，建议使用.pkl或.pkl.gz扩展名）
            compresslevel: gzip压缩级别（0-9），默认6。
                          0表示不压缩，9表示最大压缩。

        Raises:
            RuntimeError: 当尚未执行校准时
            IOError: 当文件保存失败时
        """
        if self._result_averaged_comp_data is None or self._amp_ratio_mean is None:
            raise RuntimeError("尚未执行校准，无法保存结果")

        # 准备保存的数据（只保存最终平均后的补偿数据）
        comp_data: CompData = {
            "comp_dataframe": self._result_averaged_comp_data["comp_dataframe"],
            "sampling_info": self._sampling_info,
            "frequency": self._frequency,
            "mean_amp_ratio": self._result_averaged_comp_data["mean_amp_ratio"],
            "mean_phase_shift": self._result_averaged_comp_data["mean_phase_shift"],
        }

        # 使用post_process模块的通用压缩保存函数
        from ..analyze.post_process import save_compressed_data

        save_compressed_data(comp_data, save_path, compresslevel, "CompData")

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

            # 从DataFrame中提取传递函数数据
            # CaliberOctopus的tf_dataframe是行矩阵：index为AI通道名，columns为AO通道名
            tf_df = tf_data["tf_dataframe"]

            # 调试信息
            logger.debug(f"TFData DataFrame形状: {tf_df.shape}")
            logger.debug(f"TFData DataFrame索引: {tf_df.index.tolist()}")
            logger.debug(f"TFData DataFrame列名: {tf_df.columns.tolist()}")

            tf_complex = tf_df.iloc[0, :].values  # 取第一行（唯一的行）的所有列

            logger.debug(f"提取的复数传递函数数量: {len(tf_complex)}")
            logger.debug(f"复数传递函数值: {tf_complex}")

            # 提取幅值比和相位差
            amp_ratios = np.abs(tf_complex).tolist()
            phase_shifts = np.angle(tf_complex).tolist()

            logger.debug(f"幅值比: {amp_ratios}")
            logger.debug(f"相位差: {phase_shifts}")

            # 获取AO通道名称和对应索引
            ao_channel_names = tf_df.columns.tolist()
            channel_indices = [
                self._ao_channels.index(ao_ch) if ao_ch in self._ao_channels else 0
                for ao_ch in ao_channel_names
            ]

            logger.debug(f"AO通道名称: {ao_channel_names}")
            logger.debug(f"通道索引: {channel_indices}")

        elif mode == "cartesian":
            # cartesian模式使用所有starts的CompData
            if self._result_raw_comp_data is None:
                raise RuntimeError("尚未执行校准，无法绘制直角坐标图")

            comp_data = self._result_raw_comp_data

            # 从DataFrame中提取补偿数据
            # CaliberOctopus的comp_dataframe的index为AO通道名（可能重复多次，因为包含所有starts）
            comp_df = comp_data["comp_dataframe"]

            # 提取幅值补偿倍率和时间延迟补偿
            amp_multipliers = comp_df["amp_multiplier"].values.tolist()
            time_increments = comp_df["time_increment"].values.tolist()

            # 获取AO通道名称和对应索引（注意：index中可能有重复的通道名）
            ao_channel_names = comp_df.index.tolist()
            channel_indices = [
                self._ao_channels.index(ao_ch) if ao_ch in self._ao_channels else 0
                for ao_ch in ao_channel_names
            ]

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
                f"频率: {self._frequency}Hz, "
                f"幅值: {self._amplitude}V",
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
                time_increments,
                amp_multipliers,
                c=channel_indices,
                cmap="viridis",
                s=150,
                alpha=0.8,
                edgecolors="black",
                linewidths=2,
                zorder=3,
            )

            # 为每个数据点添加详细标签（AO 通道名称）
            for time_increment, amp_multiplier, channel_idx in zip(
                time_increments, amp_multipliers, channel_indices, strict=True
            ):
                # 获取通道名称
                channel_name = self._ao_channels[channel_idx]
                # 创建标签（AO 通道名称）
                label = channel_name

                # 使用annotate添加带箭头的标签
                ax.annotate(
                    label,
                    xy=(time_increment, amp_multiplier),
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
            ax.set_xlabel("时间延迟补偿 (s)", fontsize=12)
            ax.set_ylabel("幅值补偿倍率", fontsize=12)

            # 设置网格
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)

            # 设置标题
            ax.set_title(
                f"补偿数据直角坐标分布（细节图）\n"
                f"频率: {self._frequency}Hz, "
                f"幅值: {self._amplitude}V",
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
            time_increment_array = np.array(time_increments)
            amp_multiplier_array = np.array(amp_multipliers)

            # 计算数据范围
            ti_min, ti_max = time_increment_array.min(), time_increment_array.max()
            am_min, am_max = amp_multiplier_array.min(), amp_multiplier_array.max()

            # 添加边距（数据范围的10%）
            ti_margin = (ti_max - ti_min) * 0.1 if ti_max > ti_min else 1e-6
            am_margin = (am_max - am_min) * 0.1 if am_max > am_min else 0.01

            ax.set_xlim(ti_min - ti_margin, ti_max + ti_margin)
            ax.set_ylim(am_min - am_margin, am_max + am_margin)

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
    from sweeper400.use.clb import CaliberFishNet
    from sweeper400.analyze import init_sampling_info

    # 创建采样信息
    sampling_info = init_sampling_info(171500.0, 85750)  # 采样率171.5kHz, 0.5秒

    # 创建校准对象
    clb = CaliberFishNet(
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
        frequency=3430.0,
        amplitude=0.01,
    )

    # 执行校准（10次独立校准，每次4个chunk）
    clb.calibrate(starts_num=10, chunks_per_start=4)

    # 校准结果会自动保存到默认路径
    # 也可以手动保存到指定路径
    clb.save_tf_data("calibration_results.pkl")

    # 绘制传递函数图
    clb.plot_comp_data()
    ```

    ## 注意事项：
        - 校准前确保硬件连接正确（传声器和扬声器已正确安装）
        - 建议在安静环境中进行校准以减少噪声干扰
        - 校准频率和幅值应根据实际应用场景选择
        - 该校准方法仅适用于单频单幅值工作点
        - 使用ao_comp_data参数时，可以只包含部分AO通道的补偿数据
        - 使用ai_comp_data参数时，可以只包含部分AI通道的补偿数据
        - 建议先使用CaliberSardine校准AI通道，再使用CaliberOctopus校准AO通道，
          最后使用CaliberFishNet并同时提供两个补偿数据以获得最精确的传递函数矩阵
    """

    # 获取类日志器（类属性，所有实例共享）
    logger = get_logger(f"{__name__}.CaliberFishNet")

    def __init__(
        self,
        ai_channels: tuple[str, ...],
        ao_channels: tuple[str, ...],
        sampling_info: SamplingInfo | None = None,
        frequency: PositiveFloat | None = None,
        amplitude: PositiveFloat = 0.01,
        ao_comp_data: str | Path | None = None,
        ai_comp_data: str | Path | None = None,
    ) -> None:
        """
        初始化校准对象

        调用父类 CaliberOctopus.__init__ 完成通用初始化（参数验证、波形生成、
        状态变量初始化等），然后补充 FishNet 独有的AI补偿数据加载和结果存储变量。

        Args:
            ai_channels: AI 通道名称元组（例如 ("PXI1Slot2/ai0", "PXI2Slot2/ai0", ...)）。
                         渔网模式下通常包含多个 AI 通道，以测量完整的传递函数矩阵。
            ao_channels: AO 通道名称元组（例如 ("PXI2Slot2/ao0", "PXI2Slot2/ao1", ...)）
            sampling_info: 采样信息，包含采样率和采样点数。如果为None，将自动从
                          FrequencyOptimizer的存储结果中加载。
            frequency: 正弦波频率（Hz）。如果为None，将自动从
                      FrequencyOptimizer的存储结果中加载。
            amplitude: 正弦波幅值（V），默认0.01
            ao_comp_data: 可选，AO通道补偿数据文件路径（通常由CaliberOctopus生成）。
                       支持三级优先级（由父类CaliberOctopus处理）：
                       1. 用户显式提供的路径（如果提供）
                       2. 默认路径 "storage/calib/calib_result_octopus/ao_comp_data.pkl"
                       3. 不使用补偿（如果都不存在）

                       如果加载成功，将使用补偿波形。支持部分通道补偿：
                       ao_comp_data中可以只包含部分AO通道，未包含的通道将使用未补偿信号
            ai_comp_data: 可选，AI通道补偿数据文件路径（通常由CaliberSardine生成）。
                       支持三级优先级：
                       1. 用户显式提供的路径（如果提供）
                       2. 默认路径 "storage/calib/calib_result_sardine/ai_comp_data.pkl"
                       3. 不使用补偿（如果都不存在）

                       如果加载成功，将在计算传递函数时应用AI通道补偿，校正传声器之间的
                       差异，从而获得更加精确的结果。支持部分通道补偿：ai_comp_data中
                       可以只包含部分AI通道，未包含的通道将不进行补偿

        Raises:
            ValueError: 当参数无效时
            FileNotFoundError: 当用户显式提供路径但文件不存在时
            RuntimeError: 当补偿数据文件加载失败时
        """
        # 调用父类初始化（完成参数验证、波形生成、通用状态变量初始化）
        # 父类会智能加载AO补偿数据（优先级：显式路径 > 默认路径 > 无补偿）
        super().__init__(
            ai_channels=ai_channels,
            ao_channels=ao_channels,
            sampling_info=sampling_info,
            frequency=frequency,
            amplitude=amplitude,
            ai_comp_data=ai_comp_data,
            ao_comp_data=ao_comp_data,
        )

        # FishNet 独有的校准结果存储变量
        # 所有starts的传递函数数据（未平均，用于绘制细节图）
        self._result_raw_tf_data: TFData | None = None

        # 注意：_result_averaged_tf_data 已由父类初始化为 None，此处无需重复声明

    def _calculate_tf_data(
        self,
        filtered_sweep_data: SweepData,
    ) -> TFData:
        """
        从已滤波/平均的 SweepData 计算传递函数数据（TFData）。

        该方法将 SweepData 中的多通道 AI 波形分解为多个单通道传递函数，
        对应父类的 _calculate_comp_data 方法。该方法只负责计算，不进行
        平均或滤波处理——这些预处理工作由调用方（calibrate）负责完成。

        如果在初始化时提供了 ai_comp_data（通常由 CaliberSardine 生成），
        该方法会在计算传递函数后应用 AI 通道补偿，校正传声器之间的差异，
        从而获得更加精确的结果。补偿逻辑：
        - 幅值比 = 原始幅值比 × AI幅值补偿倍率
        - 相位差 = 原始相位差 + AI时间延迟补偿对应的相位

        SweepData 中的测量点顺序与 self._ao_channels 的顺序一致，
        每个测量点的 position.x 存储 AO 通道索引（0-based），用于映射到真实通道名。

        Args:
            filtered_sweep_data: 已经过平均和滤波处理的 SweepData，
                每个测量点的 ai_data 列表中只有一条波形（已平均）。

        Returns:
            TFData: 包含传递函数DataFrame的数据容器，DataFrame是方形矩阵
            （行索引为AO通道名，列索引为AI通道名）。
            如果提供了 ai_comp_data，返回的传递函数已应用 AI 补偿。

        Raises:
            RuntimeError: 当 AO 波形没有 frequency 或 channel_complex_amplitudes 属性时
            ValueError: 当 AI 波形维度或通道数不符合预期时
        """
        logger.info("开始计算多通道AI数据的传递函数")

        # 1. 提取AO波形的正弦波参数
        ao_waveform = filtered_sweep_data["ao_data"]
        if ao_waveform.frequency is None:
            raise RuntimeError("AO波形没有frequency属性")
        if ao_waveform.channel_complex_amplitudes is None:
            raise RuntimeError("AO波形没有channel_complex_amplitudes属性")
        ao_frequency = ao_waveform.frequency
        ao_cca = ao_waveform.channel_complex_amplitudes

        # 2. 初始化DataFrame（方形矩阵：行索引为AO通道，列索引为AI通道）
        # 直接创建DataFrame并在循环中填充，避免中间字典的开销
        tf_df = pd.DataFrame(
            index=list(self._ao_channels),
            columns=list(self._ai_channels),
            dtype=complex,
        )

        # 3. 遍历所有测量点，计算并填充传递函数
        for point_data in filtered_sweep_data["ai_data_list"]:
            ao_channel_idx = int(point_data["position"].x)  # AO通道索引（0-based）
            ao_channel_name = self._ao_channels[ao_channel_idx]  # 真实AO通道名
            ai_waveform = point_data["ai_data"][0]  # 已平均的多通道AI波形

            # 验证AI波形通道数
            num_ai_channels = ai_waveform.channels_num
            if num_ai_channels != len(self._ai_channels):
                raise ValueError(
                    f"AI波形通道数({num_ai_channels})与配置的AI通道数({len(self._ai_channels)})不一致"
                )

            # 对每个AI通道，计算传递函数并直接填充到DataFrame
            for ai_channel_idx in range(num_ai_channels):
                ai_channel_name = self._ai_channels[ai_channel_idx]  # 真实AI通道名

                # 提取单通道AI波形
                single_ai_waveform = Waveform(
                    input_array=ai_waveform[ai_channel_idx, :],
                    sampling_rate=ai_waveform.sampling_rate,
                    channel_names=(ai_channel_name,),
                    timestamp=ai_waveform.timestamp,
                )

                # 使用extract_single_tone_information_vvi提取AI信号的正弦波参数
                result_wf = extract_single_tone_information_vvi(
                    single_ai_waveform,
                    approx_freq=ao_frequency,
                )

                # 计算传递函数（使用补偿后的AI正弦波参数）
                ai_complex_amps = result_wf.channel_complex_amplitudes
                ai_amplitude = float(np.abs(ai_complex_amps[0]))
                ai_phase = float(np.angle(ai_complex_amps[0]))

                # 获取当前AO通道的复振幅
                ao_amplitude = float(np.abs(ao_cca[ao_channel_idx]))
                ao_phase = float(np.angle(ao_cca[ao_channel_idx]))

                amp_ratio = ai_amplitude / ao_amplitude
                phase_shift = ai_phase - ao_phase

                # 将相位差归一化到 [-π, π] 区间
                phase_shift = float(
                    np.arctan2(np.sin(phase_shift), np.cos(phase_shift))
                )

                # 计算复数传递函数并直接填充到DataFrame
                tf_complex = amp_ratio * np.exp(1j * phase_shift)
                tf_df.loc[ao_channel_name, ai_channel_name] = tf_complex

                logger.debug(
                    f"AO通道 {ao_channel_name} -> AI通道 {ai_channel_name}: "
                    f"amp_ratio={amp_ratio:.6f}, phase_shift={phase_shift:.6f}rad"
                )

        # 4. 计算平均幅值比和平均相位差
        tf_complex_values = tf_df.values.flatten()
        amp_ratios = np.abs(tf_complex_values)
        phase_shifts = np.angle(tf_complex_values)
        mean_amp_ratio = float(np.mean(amp_ratios))
        mean_phase_shift = float(np.mean(phase_shifts))

        # 5. 构建TFData
        tf_data: TFData = {
            "tf_dataframe": tf_df,
            "sampling_info": self._sampling_info,
            "frequency": ao_frequency,
            "mean_amp_ratio": mean_amp_ratio,
            "mean_phase_shift": mean_phase_shift,
        }

        logger.info(
            f"传递函数计算完成，共计算 {tf_df.size} 个通道对，"
            f"平均幅值比={mean_amp_ratio:.6f}, 平均相位差={mean_phase_shift:.6f}rad"
        )
        return tf_data

    def calibrate(
        self,
        starts_num: int = 10,
        chunks_per_start: int = 3,
        apply_filter: bool = True,
        result_folder: str | Path | None = None,
        settle_time: float | None = None,
    ) -> TFData:
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
            result_folder: 可选，结果保存文件夹路径。如果为None，将使用默认路径
                          'storage/calib/calib_result_fishnet'（相对于项目根目录）。
                          最终将保存多个raw_sweep_data_N.pkl文件、绘图和一个平均后的TFData文件
            settle_time: 通道切换后的稳定等待时间（秒）。如果为None，则使用初始化时
                        计算的默认值（chunk时长 + 0.1秒）

        Returns:
            TFData: 平均后的传递函数数据

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
                save_compressed_data(sweep_data, sweep_data_path, 6, "SweepData")
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
                lowcut = self._frequency * 0.5
                highcut = self._frequency * 2.0
                logger.info(f"应用带通滤波器: {lowcut}Hz - {highcut}Hz")
                filtered_data = filter_sweep_data(
                    averaged_data,
                    lowcut=lowcut,
                    highcut=highcut,
                    trim_samples=300,  # 约6个周期
                )
            else:
                filtered_data = averaged_data

            # 3. 绘制滤波后数据的融合画像
            logger.info("绘制滤波后数据的融合画像")
            fusion_plot_filename = f"fusion_waveform_{calib_idx + 1}.png"
            fusion_plot_path = result_path / fusion_plot_filename

            try:
                fig, _ = plot_sweep_data_as_single_waveform(
                    sweep_data=filtered_data,
                    save_path=str(fusion_plot_path),
                    zoom_factor=50,
                )
                # 关闭图形以释放内存
                import matplotlib.pyplot as plt
                plt.close(fig)

                logger.info(f"融合画像已保存到: {fusion_plot_path}")
            except Exception as e:
                logger.error(
                    f"绘制第 {calib_idx + 1} 次校准的融合画像时发生错误: {e}",
                    exc_info=True,
                )
                # 继续处理，不中断整个校准流程

            # 4. 使用 _calculate_tf_data 计算传递函数（已返回TFData）
            tf_data = self._calculate_tf_data(filtered_data)
            all_tf_data_list.append(tf_data)

            logger.info(
                f"第 {calib_idx + 1} 次校准数据处理完成，"
                f"频率={tf_data['frequency']:.2f}Hz, "
                f"平均幅值比={tf_data['mean_amp_ratio']:.6f}, "
                f"平均相位差={tf_data['mean_phase_shift']:.6f}rad"
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
            f"平均元数据: 频率={averaged_tf_data['frequency']:.2f}Hz, "
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
        self.save_tf_data(save_path=final_tf_data_path)

        # 导出TFData为可读的csv文件，方便用户查看具体数值
        self._export_tf_data_to_csv(averaged_tf_data, result_path)

        logger.info("=" * 60)
        logger.info(f"校准流程完成，所有结果已保存到: {result_path}")
        logger.info("=" * 60)

        return averaged_tf_data

    def save_tf_data(
        self,
        save_path: str | Path,
        compresslevel: int = 6,
    ) -> None:
        """
        保存校准结果到本地文件（使用gzip压缩）

        将最终传递函数数据序列化并使用gzip压缩保存到磁盘。

        Args:
            save_path: 保存文件的路径（支持字符串或Path对象，建议使用.pkl或.pkl.gz扩展名）
            compresslevel: gzip压缩级别（0-9），默认6。
                          0表示不压缩，9表示最大压缩。

        Raises:
            RuntimeError: 当尚未执行校准时
            IOError: 当文件保存失败时
        """
        if self._result_averaged_tf_data is None:
            raise RuntimeError("尚未执行校准，无法保存结果")

        # 使用post_process模块的通用压缩保存函数
        from ..analyze.post_process import save_compressed_data

        save_compressed_data(
            self._result_averaged_tf_data, save_path, compresslevel, "TFData"
        )

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

    @staticmethod
    def _export_tf_data_to_csv(
        tf_data: TFData,
        result_path: Path,
    ) -> None:
        """
        将TFData导出为csv文件，方便用户查看具体数值。

        导出两个csv文件到结果文件夹：
        - "tf_data_amp_ratio.csv": 幅值比矩阵（行=AO通道，列=AI通道）
        - "tf_data_phase_shift_rad.csv": 相位差矩阵（弧度制，行=AO通道，列=AI通道）

        Args:
            tf_data: 要导出的TFData对象
            result_path: 结果文件夹路径
        """
        tf_df = tf_data["tf_dataframe"]

        # 分解复数传递函数为幅值比和相位差
        amp_ratio_df = tf_df.apply(lambda col: col.map(lambda x: float(np.abs(x))))
        phase_shift_df = tf_df.apply(lambda col: col.map(lambda x: float(np.angle(x))))

        # 写入csv文件
        try:
            amp_ratio_path = result_path / "tf_data_amp_ratio.csv"
            phase_shift_path = result_path / "tf_data_phase_shift_rad.csv"
            amp_ratio_df.to_csv(amp_ratio_path)
            phase_shift_df.to_csv(phase_shift_path)
            logger.info(
                f"TFData已导出为csv文件: {amp_ratio_path}, {phase_shift_path}"
            )
        except Exception as e:
            logger.error(f"导出TFData为csv失败: {e}", exc_info=True)

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
        # 从DataFrame中提取传递函数数据
        tf_df = tf_data["tf_dataframe"]

        # 按AO通道名称分组
        # 格式: {ao_ch: [(ai_ch, ao_idx, ai_idx, ch_diff, amp, phase), ...]}
        ao_groups: dict[str, list[tuple[str, int, int, int, float, float]]] = {}

        # 遍历DataFrame的所有元素
        for ao_ch in tf_df.index:  # 行索引：AO通道名
            ao_idx = self._ao_channels.index(ao_ch) if ao_ch in self._ao_channels else 0

            for ai_ch in tf_df.columns:  # 列索引：AI通道名
                ai_idx = (
                    self._ai_channels.index(ai_ch) if ai_ch in self._ai_channels else 0
                )

                # 获取传递函数复数值
                tf_complex = tf_df.loc[ao_ch, ai_ch]
                amp_ratio = float(np.abs(tf_complex))
                phase_shift = float(np.angle(tf_complex))

                # 计算通道序数差
                channel_diff = ai_idx - ao_idx

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
            f"传递函数幅值比分布\n频率: {self._frequency}Hz",
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
            f"传递函数相位差分布\n频率: {self._frequency}Hz",
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


class FrequencyOptimizer:
    """
    # 频率优化器

    该类用于测定当天最佳实验频率。由于声速会随空气特性变化，固定尺寸的声学
    超构表面样件的最佳工作频率每天会有细微差异。本类通过现场测定确定当天最佳频率。

    ## 原理：
        超表面的8个周期性管槽结构中分别插入了传声器。设计目标是相邻管槽测得的
        声波恰好反相（相差π）。当频率偏离最佳值时，相邻通道的相位差会偏离π，
        且偏移量与频率偏差成线性关系。通过测量多个频率点的相位偏差，线性插值
        找到相位偏差为零的频率即为最佳频率。

    ## 算法流程：
        1. 在初始频率（默认3430Hz）处测量所有传声器的传递函数
        2. 计算相邻通道相位差与π的加权偏差（signed metric）
        3. 在附近频率（如3440Hz）再次测量
        4. 通过线性插值预测最佳频率（metric为零的点）
        5. 在预测频率处测量验证
        6. 如果偏差仍然显著，重复步骤4-5直至收敛

    ## 采样参数规则：
        对于频率f：
        - sampling_rate = f × 50（每周期50个采样点）
        - samples_num = 34300（固定，对应686个完整周期）

    ## 相位偏差指标：
        对每对相邻通道(i, i+1)，计算 angle(-z_{i+1}/z_i)，理想值为0。
        使用幅值平方作为权重进行加权平均，得到有方向的偏差指标。

    ## 使用示例：
    ```python
    from sweeper400.calib import FrequencyOptimizer

    optimizer = FrequencyOptimizer(
        ai_channels=(
            "PXI1Slot2/ai0", "PXI1Slot2/ai1",
            "PXI1Slot3/ai0", "PXI1Slot3/ai1",
            "PXI1Slot4/ai0", "PXI1Slot4/ai1",
            "PXI1Slot5/ai0", "PXI1Slot5/ai1",
        ),
        ao_channel="PXI1Slot2/ao0",
        amplitude=0.01,
    )

    # 执行频率优化
    optimal_freq = optimizer.optimize()
    print(f"最佳频率: {optimal_freq:.2f} Hz")

    # 绘制优化结果
    optimizer.plot_result()
    ```

    ## 注意事项：
        - 校准前确保超表面样件和传声器已正确安装
        - AI通道数不必恰好为8，类内部会自适应处理
        - 结果保存后，其他类可通过设置 frequency=None 和 sampling_info=None 自动加载
    """

    # 获取类日志器（类属性，所有实例共享）
    logger = get_logger(f"{__name__}.FrequencyOptimizer")

    # 固定参数
    SAMPLES_PER_CYCLE: int = 50  # 每周期采样点数
    SAMPLES_NUM: int = 34300  # 单段采样数（686个周期）

    def __init__(
        self,
        ai_channels: tuple[str, ...],
        ao_channel: str,
        amplitude: PositiveFloat = 0.01,
        ai_comp_data: str | Path | None = None,
        ao_comp_data: str | Path | None = None,
    ) -> None:
        """
        初始化频率优化器

        Args:
            ai_channels: AI 通道名称元组（超表面管槽中的传声器）。
                         典型为8个通道，但允许多于或少于8个。
            ao_channel: AO 通道名称（单个扬声器/喇叭）。
            amplitude: 输出正弦波幅值（V），默认0.01。
            ai_comp_data: 可选，AI补偿数据文件路径，传递给内部CaliberFishNet。
            ao_comp_data: 可选，AO补偿数据文件路径，传递给内部CaliberFishNet。

        Raises:
            ValueError: 当参数无效时
        """
        if not ai_channels:
            raise ValueError("AI 通道列表不能为空")
        if len(ai_channels) < 2:
            raise ValueError("AI 通道数至少为2（需要计算相邻通道相位差）")

        self._ai_channels = ai_channels
        self._ao_channel = ao_channel
        self._amplitude = amplitude
        self._ai_comp_data = ai_comp_data
        self._ao_comp_data = ao_comp_data

        # 测量历史记录: [(frequency, signed_metric, amplitudes, phases)]
        self._measurement_history: list[
            dict[str, float | np.ndarray]
        ] = []

        # 最终结果
        self._optimal_frequency: float | None = None
        self._optimal_sampling_info: SamplingInfo | None = None
        self._optimization_mode: Literal["phase", "amplitude"] | None = None
        self._metric_ref: float = 1.0  # 归一化基准（第一次测量的|metric|）

        logger.info(
            f"FrequencyOptimizer 实例已创建 - "
            f"AI通道数: {len(ai_channels)}, "
            f"AO通道: {ao_channel}, "
            f"幅值: {amplitude}V"
        )

    def _make_sampling_info(self, frequency: float) -> SamplingInfo:
        """
        根据频率生成对应的采样信息

        规则: sampling_rate = frequency × 50, samples_num = 34300

        Args:
            frequency: 目标频率（Hz）

        Returns:
            SamplingInfo 字典
        """
        sampling_rate = frequency * self.SAMPLES_PER_CYCLE
        return init_sampling_info(sampling_rate, self.SAMPLES_NUM)

    def _measure_at_frequency(
        self,
        frequency: float,
        measurement_index: int,
        starts_num: int = 3,
        chunks_per_start: int = 3,
        settle_time: float | None = None,
        result_path: Path | None = None,
    ) -> TFData:
        """
        在指定频率处进行一次完整的传递函数测量

        创建一个 CaliberFishNet 实例（1个AO × N个AI），调用其 calibrate 方法
        执行完整的多次校准流程（包含保存原始数据、平均、滤波等），
        并将校准数据保存为结果文件夹中的子文件夹以便追溯。

        Args:
            frequency: 测量频率（Hz）
            measurement_index: 测量序号（从1开始），用于子文件夹命名
            starts_num: 独立校准次数，默认3（calibrate的starts_num参数）
            chunks_per_start: 每次采集的chunk数，默认3
            settle_time: 稳定等待时间（秒）
            result_path: 可选，结果文件夹根路径。如果提供，将在其中创建
                        子文件夹保存本次测量的完整校准数据。

        Returns:
            TFData: 平均后的传递函数数据
        """
        sampling_info = self._make_sampling_info(frequency)

        logger.info(
            f"在 {frequency:.2f}Hz 处进行测量 (#{measurement_index}) "
            f"(采样率={sampling_info['sampling_rate']:.1f}Hz, "
            f"采样数={sampling_info['samples_num']}, "
            f"独立校准次数={starts_num})"
        )

        # 创建 CaliberFishNet 实例（1个AO通道 × N个AI通道）
        caliber = CaliberFishNet(
            ai_channels=self._ai_channels,
            ao_channels=(self._ao_channel,),
            sampling_info=sampling_info,
            frequency=frequency,
            amplitude=self._amplitude,
            ai_comp_data=self._ai_comp_data,
            ao_comp_data=self._ao_comp_data,
        )

        # 确定本次测量的子文件夹路径
        if result_path is not None:
            sub_folder = (
                result_path
                / f"measurement_{measurement_index}_{frequency:.2f}Hz"
            )
        else:
            sub_folder = (
                Path(__file__).resolve().parents[3]
                / "storage"
                / "calib"
                / "calib_result_freq"
                / f"measurement_{measurement_index}_{frequency:.2f}Hz"
            )

        # 使用完整的 calibrate 方法执行校准
        tf_data = caliber.calibrate(
            starts_num=starts_num,
            chunks_per_start=chunks_per_start,
            apply_filter=True,
            result_folder=sub_folder,
            settle_time=settle_time,
        )

        logger.info(f"频率 {frequency:.2f}Hz 处测量完成 (#{measurement_index})")
        return tf_data

    @staticmethod
    def _compute_metric(
        tf_data: TFData,
        mode: Literal["phase", "amplitude"] = "amplitude",
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        从TFData计算优化指标

        根据 mode 选择不同的计算逻辑：
        - phase mode: 计算相邻通道相位差与π的加权平均偏差（rad）
        - amplitude mode: 计算所有通道幅值之和

        Args:
            tf_data: 传递函数数据（1行×N列矩阵，行为AO通道，列为AI通道）
            mode: 优化模式，默认 "amplitude"

        Returns:
            tuple of:
            - metric (float): 指标值
              - phase mode: 加权平均相位偏差（正=频率偏高，负=频率偏低）
              - amplitude mode: 所有通道幅值之和（越大越好）
            - amplitudes (ndarray): 各通道幅值数组
            - phases (ndarray): 各通道相位数组
        """
        tf_df = tf_data["tf_dataframe"]

        # 提取复数传递函数值（1行×N列，取第一行的所有列）
        tf_complex = tf_df.iloc[0, :].values.astype(complex)

        # 提取幅值和相位
        amplitudes = np.abs(tf_complex)
        phases = np.angle(tf_complex)

        if mode == "amplitude":
            # 目标函数：所有通道幅值之和
            metric = float(np.sum(amplitudes))
            logger.debug(
                f"幅值之和指标: metric={metric:.2f}, "
                f"平均幅值={np.mean(amplitudes):.6f}, "
                f"通道数={len(amplitudes)}"
            )
        else:
            # 计算相邻通道的相位偏差
            n_channels = len(tf_complex)
            deviations = np.zeros(n_channels - 1)
            weights = np.zeros(n_channels - 1)

            for i in range(n_channels - 1):
                ratio = -tf_complex[i + 1] / tf_complex[i]
                deviations[i] = np.angle(ratio)
                weights[i] = min(amplitudes[i] ** 2, amplitudes[i + 1] ** 2)

            if np.sum(weights) > 0:
                metric = float(np.average(deviations, weights=weights))
            else:
                metric = float(np.mean(deviations))

            logger.debug(
                f"相位偏差指标: metric={metric:.6f}rad, "
                f"平均幅值={np.mean(amplitudes):.6f}"
            )

        return metric, amplitudes, phases

    def optimize(
        self,
        mode: Literal["amplitude", "phase"] = "amplitude",
        initial_freq: float = 3430.0,
        second_freq: float = 3420.0,
        max_iterations: int = 10,
        tolerance: float = 0.001,
        starts_num: int = 1,
        chunks_per_start: int = 3,
        settle_time: float | None = None,
        result_folder: str | Path | None = None,
    ) -> float:
        """
        执行频率优化流程

        通过迭代测量和分析，找到最佳工作频率。
        支持两种优化模式：

        - **phase mode**: 寻找使相邻通道相位差与π的偏差为零的频率。
          使用线性插值（割线法）迭代逼近零点。适用于理想化的无限周期模型。
        - **amplitude mode**: 寻找使所有通道幅值之和最大的频率。
          仅假设目标函数是单峰的（先增后减），使用区间收缩法逐步
          逼近极大值。适用于有限周期的实际超表面。

        Args:
            mode: 优化模式，"amplitude" 或 "phase"，默认 "amplitude"
            initial_freq: 初始测量频率（Hz），默认3430.0
            second_freq: 第二个测量频率（Hz），默认3440.0
            max_iterations: 最大迭代次数，默认10
            tolerance: 收敛容差（归一化比例），默认0.005。
                      所有metric均以第一次测量值的绝对值为基准进行归一化，
                      因此tolerance表示"相对于初始值的比例"。
                      - phase mode: 当 |normalized_metric| < tolerance 时停止
                      - amplitude mode: 收敛判据为存在3个相邻测量点(p1,p2,p3)，
                        其中p2的metric大于p1和p3，且两侧归一化metric之差均
                        < tolerance
            starts_num: 每次测量中的独立校准次数，默认3
            chunks_per_start: 每次测量采集的chunk数，默认3
            settle_time: 稳定等待时间（秒）
            result_folder: 可选，结果保存文件夹路径。如果为None，使用默认路径
                          'storage/calib/calib_result_freq'

        Returns:
            float: 最佳频率（Hz）

        Raises:
            ValueError: 当 mode 参数无效时
            RuntimeError: 当优化未能收敛时
        """
        # 验证 mode 参数
        if mode not in ("amplitude", "phase"):
            raise ValueError(
                f"mode 必须为 'amplitude' 或 'phase'，收到: '{mode}'"
            )

        self._optimization_mode = mode
        logger.info("=" * 60)
        logger.info(f"开始频率优化流程 (mode={mode})")
        logger.info(f"初始频率: {initial_freq:.2f}Hz, 第二频率: {second_freq:.2f}Hz")
        logger.info(f"最大迭代: {max_iterations}, 容差: {tolerance}")
        logger.info(f"每次测量: {starts_num}次独立校准, {chunks_per_start}个chunk/次")
        logger.info("=" * 60)

        # 提前解析结果保存路径，用于传递给 _measure_at_frequency 保存子文件夹
        if result_folder is None:
            resolved_result_path = (
                Path(__file__).resolve().parents[3]
                / "storage"
                / "calib"
                / "calib_result_freq"
            )
        else:
            resolved_result_path = Path(result_folder)
        resolved_result_path.mkdir(parents=True, exist_ok=True)

        # 清空测量历史
        self._measurement_history = []
        measurement_counter = 0

        # 第一次测量
        measurement_counter += 1
        tf_data_1 = self._measure_at_frequency(
            initial_freq,
            measurement_index=measurement_counter,
            starts_num=starts_num,
            chunks_per_start=chunks_per_start,
            settle_time=settle_time,
            result_path=resolved_result_path,
        )
        metric_1, amps_1, phases_1 = self._compute_metric(tf_data_1, mode)

        # 设定归一化基准：以第一次测量的|metric|为参考
        self._metric_ref = abs(metric_1) if abs(metric_1) > 1e-12 else 1.0
        logger.info(
            f"频率 {initial_freq:.2f}Hz: raw metric = {metric_1:.6f}, "
            f"归一化基准 = {self._metric_ref:.6f}"
        )

        # 归一化后存储
        metric_1_norm = metric_1 / self._metric_ref
        self._measurement_history.append({
            "frequency": initial_freq,
            "metric": metric_1_norm,
            "amplitudes": amps_1,
            "phases": phases_1,
        })

        # Phase mode: 检查是否已满足条件（使用归一化值）
        if mode == "phase" and abs(metric_1_norm) < tolerance:
            logger.info(f"初始频率已满足容差要求，最佳频率: {initial_freq:.2f}Hz")
            self._finalize(initial_freq, resolved_result_path)
            return initial_freq

        # 第二次测量
        measurement_counter += 1
        tf_data_2 = self._measure_at_frequency(
            second_freq,
            measurement_index=measurement_counter,
            starts_num=starts_num,
            chunks_per_start=chunks_per_start,
            settle_time=settle_time,
            result_path=resolved_result_path,
        )
        metric_2, amps_2, phases_2 = self._compute_metric(tf_data_2, mode)
        metric_2_norm = metric_2 / self._metric_ref
        self._measurement_history.append({
            "frequency": second_freq,
            "metric": metric_2_norm,
            "amplitudes": amps_2,
            "phases": phases_2,
        })
        logger.info(
            f"频率 {second_freq:.2f}Hz: raw metric = {metric_2:.6f}, "
            f"归一化 = {metric_2_norm:.6f}"
        )

        # Phase mode: 检查是否已满足条件（使用归一化值）
        if mode == "phase" and abs(metric_2_norm) < tolerance:
            logger.info(f"第二频率已满足容差要求，最佳频率: {second_freq:.2f}Hz")
            self._finalize(second_freq, resolved_result_path)
            return second_freq

        # 进入迭代优化
        freq_center = (initial_freq + second_freq) / 2.0

        if mode == "amplitude":
            iterate_result = self._iterate_amplitude(
                initial_freq, second_freq, metric_1_norm, metric_2_norm,
                freq_center, max_iterations, tolerance, starts_num,
                chunks_per_start, settle_time, resolved_result_path,
                measurement_counter,
            )
        else:
            iterate_result = self._iterate_phase(
                initial_freq, second_freq, metric_1_norm, metric_2_norm,
                freq_center, max_iterations, tolerance, starts_num,
                chunks_per_start, settle_time, resolved_result_path,
                measurement_counter,
            )

        return iterate_result


    def _iterate_phase(
        self,
        initial_freq: float,
        second_freq: float,
        metric_1: float,
        metric_2: float,
        freq_center: float,
        max_iterations: int,
        tolerance: float,
        starts_num: int,
        chunks_per_start: int,
        settle_time: float | None,
        resolved_result_path: Path,
        measurement_counter: int,
    ) -> float:
        """Phase mode 迭代：割线法逼近相位偏差零点"""
        freq_a, metric_a = initial_freq, metric_1
        freq_b, metric_b = second_freq, metric_2

        for iteration in range(max_iterations):
            logger.info(f"--- 迭代 {iteration + 1}/{max_iterations} ---")

            # 线性插值预测零点
            if abs(metric_b - metric_a) < 1e-12:
                logger.warning("两个测量点的metric几乎相同，无法线性插值")
                break

            predicted_freq = freq_a - metric_a * (freq_b - freq_a) / (
                metric_b - metric_a
            )

            # 安全检查：预测频率不应偏离初始值太远（±200Hz）
            if abs(predicted_freq - freq_center) > 200.0:
                logger.warning(
                    f"预测频率 {predicted_freq:.2f}Hz 偏离中心过远，"
                    f"限制在 [{freq_center - 200:.0f}, {freq_center + 200:.0f}]Hz"
                )
                predicted_freq = np.clip(
                    predicted_freq, freq_center - 200.0, freq_center + 200.0
                )

            logger.info(f"线性插值预测最佳频率: {predicted_freq:.2f}Hz")

            # 在预测频率处测量
            assert isinstance(predicted_freq, float)
            measurement_counter += 1
            tf_data_new = self._measure_at_frequency(
                predicted_freq,
                measurement_index=measurement_counter,
                starts_num=starts_num,
                chunks_per_start=chunks_per_start,
                settle_time=settle_time,
                result_path=resolved_result_path,
            )
            metric_new, amps_new, phases_new = self._compute_metric(
                tf_data_new, "phase"
            )
            metric_new_norm = metric_new / self._metric_ref
            self._measurement_history.append({
                "frequency": predicted_freq,
                "metric": metric_new_norm,
                "amplitudes": amps_new,
                "phases": phases_new,
            })
            logger.info(
                f"频率 {predicted_freq:.2f}Hz: "
                f"归一化metric = {metric_new_norm:.6f}"
            )

            # 检查收敛（使用归一化值）
            if abs(metric_new_norm) < tolerance:
                logger.info(
                    f"已收敛！最佳频率: {predicted_freq:.2f}Hz "
                    f"(归一化metric={metric_new_norm:.6f} < tolerance={tolerance})"
                )
                self._finalize(predicted_freq, resolved_result_path)
                return predicted_freq

            # 选择下一对插值点：使用新点替换同号的旧点
            if metric_new_norm * metric_a > 0:
                freq_a, metric_a = predicted_freq, metric_new_norm
            else:
                freq_b, metric_b = predicted_freq, metric_new_norm

        # 未能完全收敛，使用metric绝对值最小的频率点
        best_entry = min(
            self._measurement_history, key=lambda x: abs(x["metric"])
        )
        best_freq = float(best_entry["frequency"])

        logger.warning(
            f"达到最大迭代次数 {max_iterations}，"
            f"使用metric最小的频率: {best_freq:.2f}Hz "
            f"(metric={best_entry['metric']:.6f}rad)"
        )
        self._finalize(best_freq, resolved_result_path)
        return best_freq

    def _iterate_amplitude(
        self,
        initial_freq: float,
        second_freq: float,
        metric_1: float,
        metric_2: float,
        freq_center: float,
        max_iterations: int,
        tolerance: float,
        starts_num: int,
        chunks_per_start: int,
        settle_time: float | None,
        resolved_result_path: Path,
        measurement_counter: int,
    ) -> float:
        """
        Amplitude mode 迭代：区间收缩法寻找极大值

        仅假设目标函数是单峰的（先增后减，有唯一极大值）。
        通过逐步缩减包围峰值的区间来逼近极大值。

        算法步骤：
        1. 测量第三个点（根据前两点趋势方向选择），以建立初始区间
        2. 每次迭代：
           a. 将所有历史测量点按频率排序
           b. 检查收敛条件（是否存在满足条件的3相邻点）
           c. 找到当前metric最大的点（peak candidate）
           d. 若peak在边缘：向该方向延伸一步
           e. 若peak被包围：在较宽半区间的中点测量以缩减区间
        """
        step = second_freq - initial_freq  # 有符号步长，保留方向信息

        # --- 第三个测量点 ---
        # 根据前两点的趋势选择第三个点，以"包围"可能的峰值
        if metric_2 > metric_1:
            # 幅值朝 second_freq 方向增大，继续延伸
            third_freq = second_freq + step
        elif metric_1 > metric_2:
            # 幅值朝 initial_freq 方向增大，反向延伸
            third_freq = initial_freq - step
        else:
            # 相等（极罕见），取中点
            third_freq = freq_center

        # 安全限制
        third_freq = float(
            np.clip(third_freq, freq_center - 200.0, freq_center + 200.0)
        )

        measurement_counter += 1
        tf_data_3 = self._measure_at_frequency(
            third_freq,
            measurement_index=measurement_counter,
            starts_num=starts_num,
            chunks_per_start=chunks_per_start,
            settle_time=settle_time,
            result_path=resolved_result_path,
        )
        metric_3, amps_3, phases_3 = self._compute_metric(
            tf_data_3, "amplitude"
        )
        metric_3_norm = metric_3 / self._metric_ref
        self._measurement_history.append({
            "frequency": third_freq,
            "metric": metric_3_norm,
            "amplitudes": amps_3,
            "phases": phases_3,
        })
        logger.info(
            f"频率 {third_freq:.2f}Hz: 归一化metric = {metric_3_norm:.6f}"
        )

        # --- 区间收缩迭代 ---
        for iteration in range(max_iterations):
            logger.info(f"--- 迭代 {iteration + 1}/{max_iterations} ---")

            # 将所有历史测量点按频率排序
            sorted_hist = sorted(
                self._measurement_history, key=lambda x: x["frequency"]
            )
            n_points = len(sorted_hist)

            # 检查收敛条件：是否存在3个相邻点满足条件
            for i in range(n_points - 2):
                p1, p2, p3 = sorted_hist[i], sorted_hist[i + 1], sorted_hist[i + 2]
                if (
                    p2["metric"] > p1["metric"]
                    and p2["metric"] > p3["metric"]
                    and (p2["metric"] - p1["metric"]) < tolerance
                    and (p2["metric"] - p3["metric"]) < tolerance
                ):
                    best_freq = float(p2["frequency"])
                    logger.info(
                        f"已收敛！找到极大值点: {best_freq:.4f}Hz "
                        f"(metric={p2['metric']:.6f}, "
                        f"Δleft={p2['metric'] - p1['metric']:.6f}, "
                        f"Δright={p2['metric'] - p3['metric']:.6f} "
                        f"< tolerance={tolerance})"
                    )
                    self._finalize(best_freq, resolved_result_path)
                    return best_freq

            # 找到当前metric最大的点的索引（在sorted_hist中）
            best_idx = max(
                range(n_points), key=lambda i: sorted_hist[i]["metric"]
            )

            # 决定下一个测量点
            if best_idx == 0:
                # Peak在左边缘 — 向左延伸
                left_step = (
                    sorted_hist[1]["frequency"] - sorted_hist[0]["frequency"]
                )
                new_freq = sorted_hist[0]["frequency"] - left_step
                logger.info(
                    f"Peak在左边缘，向左延伸: {new_freq:.2f}Hz"
                )
            elif best_idx == n_points - 1:
                # Peak在右边缘 — 向右延伸
                right_step = (
                    sorted_hist[-1]["frequency"] - sorted_hist[-2]["frequency"]
                )
                new_freq = sorted_hist[-1]["frequency"] + right_step
                logger.info(
                    f"Peak在右边缘，向右延伸: {new_freq:.2f}Hz"
                )
            else:
                # Peak被包围 — 在较宽半区间的中点测量
                left_freq = sorted_hist[best_idx - 1]["frequency"]
                peak_freq = sorted_hist[best_idx]["frequency"]
                right_freq = sorted_hist[best_idx + 1]["frequency"]

                left_width = peak_freq - left_freq
                right_width = right_freq - peak_freq

                if left_width >= right_width:
                    new_freq = (left_freq + peak_freq) / 2.0
                    logger.info(
                        f"Peak被包围，缩减左半区间: "
                        f"[{left_freq:.2f}, {peak_freq:.2f}] → "
                        f"中点 {new_freq:.2f}Hz"
                    )
                else:
                    new_freq = (peak_freq + right_freq) / 2.0
                    logger.info(
                        f"Peak被包围，缩减右半区间: "
                        f"[{peak_freq:.2f}, {right_freq:.2f}] → "
                        f"中点 {new_freq:.2f}Hz"
                    )

            # 安全限制
            new_freq = float(
                np.clip(new_freq, freq_center - 200.0, freq_center + 200.0)
            )

            # 在新频率处测量
            measurement_counter += 1
            tf_data_new = self._measure_at_frequency(
                new_freq,
                measurement_index=measurement_counter,
                starts_num=starts_num,
                chunks_per_start=chunks_per_start,
                settle_time=settle_time,
                result_path=resolved_result_path,
            )
            metric_new, amps_new, phases_new = self._compute_metric(
                tf_data_new, "amplitude"
            )
            metric_new_norm = metric_new / self._metric_ref
            self._measurement_history.append({
                "frequency": new_freq,
                "metric": metric_new_norm,
                "amplitudes": amps_new,
                "phases": phases_new,
            })
            logger.info(
                f"频率 {new_freq:.2f}Hz: 归一化metric = {metric_new_norm:.6f}"
            )

        # 未能完全收敛，使用幅值最大的测量点
        best_entry = max(
            self._measurement_history, key=lambda x: x["metric"]
        )
        best_freq = float(best_entry["frequency"])

        logger.warning(
            f"达到最大迭代次数 {max_iterations}，"
            f"使用幅值之和最大的频率: {best_freq:.2f}Hz "
            f"(metric={best_entry['metric']:.6f})"
        )
        self._finalize(best_freq, resolved_result_path)
        return best_freq

    def _finalize(
        self,
        optimal_frequency: float,
        result_folder: str | Path | None = None,
    ) -> None:
        """
        完成优化：保存结果和绘图

        Args:
            optimal_frequency: 确定的最佳频率
            result_folder: 结果保存路径
        """
        self._optimal_frequency = optimal_frequency
        self._optimal_sampling_info = self._make_sampling_info(optimal_frequency)

        # 确定结果保存路径
        if result_folder is None:
            result_path = (
                Path(__file__).resolve().parents[3]
                / "storage"
                / "calib"
                / "calib_result_freq"
            )
        else:
            result_path = Path(result_folder)

        result_path.mkdir(parents=True, exist_ok=True)

        # 保存核心结果（供其他类自动加载）
        freq_result = {
            "frequency": optimal_frequency,
            "sampling_info": self._optimal_sampling_info,
            "mode": self._optimization_mode,
        }
        freq_result_path = result_path / "freq_result.pkl"
        try:
            save_compressed_data(freq_result, freq_result_path, 6, "频率优化结果")
            logger.info(f"频率优化结果已保存到: {freq_result_path}")
        except Exception as e:
            logger.error(f"保存频率优化结果失败: {e}", exc_info=True)

        # 保存完整的测量历史数据
        history_path = result_path / "measurement_history.pkl"
        try:
            save_compressed_data(
                self._measurement_history, history_path, 6, "测量历史数据"
            )
            logger.info(f"测量历史数据已保存到: {history_path}")
        except Exception as e:
            logger.error(f"保存测量历史数据失败: {e}", exc_info=True)

        # 绘制并保存结果图
        plot_path = result_path / "freq_optimization.png"
        self.plot_result(save_path=plot_path)

        logger.info(f"频率优化完成，所有结果已保存到: {result_path}")

    def plot_result(self, save_path: str | Path | None = None) -> None:
        """
        绘制频率优化结果图

        根据优化模式自动选择绘图方式：
        - **phase mode**: 横轴频率，纵轴相位偏差(rad)，线性趋势线+零点标记
        - **amplitude mode**: 横轴频率，纵轴幅值之和，测量点连线+峰值标记

        所有测量过的频率点绘制为散点，最终确定的最佳频率以竖线标记。

        Args:
            save_path: 可选，图像保存路径。如果为None则只显示不保存。

        Raises:
            RuntimeError: 当尚未执行优化时
        """
        if not self._measurement_history:
            raise RuntimeError("尚未执行频率优化，无法绘图")

        logger.info("开始绘制频率优化结果图")

        mode = self._optimization_mode or "amplitude"

        # 提取数据
        frequencies = np.array(
            [entry["frequency"] for entry in self._measurement_history]
        )
        metrics = np.array(
            [entry["metric"] for entry in self._measurement_history]
        )

        # 创建图表
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))

        # 绘制散点（所有测量点）
        ax.scatter(
            frequencies,
            metrics,
            s=120,
            c="steelblue",
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
            label="测量点",
        )

        # 为每个散点添加序号标注
        for idx, (freq, metric) in enumerate(
            zip(frequencies, metrics, strict=True)
        ):
            ax.annotate(
                f"#{idx + 1}",
                xy=(freq, metric),
                xytext=(5, 8),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )

        # 根据模式绘制趋势线
        freq_range = np.linspace(
            frequencies.min() - 5, frequencies.max() + 5, 200
        )

        if mode == "phase":
            self._plot_trend(ax, frequencies, metrics, freq_range, "phase")
        else:
            self._plot_trend(ax, frequencies, metrics, freq_range, "amplitude")

        # 标记最终确定的最佳频率
        if self._optimal_frequency is not None:
            ax.axvline(
                self._optimal_frequency,
                color="red",
                linestyle="-",
                linewidth=2,
                alpha=0.5,
                label=f"最佳频率: {self._optimal_frequency:.2f}Hz",
            )

        # 设置坐标轴和标题
        ax.set_xlabel("频率 (Hz)", fontsize=12)
        if mode == "phase":
            ax.set_ylabel("归一化相位偏差", fontsize=12)
            # 绘制零线
            ax.axhline(
                0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5
            )
        else:
            ax.set_ylabel("归一化幅值之和", fontsize=12)

        mode_label = "Phase" if mode == "phase" else "Amplitude"
        ax.set_title(
            f"频率优化结果 ({mode_label} mode)\n"
            f"AI通道数: {len(self._ai_channels)}, "
            f"AO通道: {self._ao_channel}",
            fontsize=14,
            pad=15,
        )
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
        ax.legend(loc="best", fontsize=10)

        plt.tight_layout()

        # 保存或显示
        if save_path is not None:
            save_path_obj = Path(save_path)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path_obj, dpi=300, bbox_inches="tight")
            logger.info(f"频率优化图已保存到: {save_path_obj}")

        plt.show()
        logger.info("频率优化图绘制完成")

    @staticmethod
    def _plot_trend(
        ax: plt.Axes,
        frequencies: np.ndarray,
        metrics: np.ndarray,
        freq_range: np.ndarray,
        mode: Literal["phase", "amplitude"] = "amplitude",
    ) -> None:
        """
        根据模式绘制趋势线

        Args:
            ax: matplotlib 坐标轴
            frequencies: 测量频率数组
            metrics: 指标数组
            freq_range: 用于绘制趋势线的频率范围
            mode: 优化模式，默认 "amplitude"
        """
        if mode == "phase":
            # Phase mode: 线性趋势线 + 零点标记
            if len(frequencies) < 2:
                return

            coeffs = np.polyfit(frequencies, metrics, 1)
            poly = np.poly1d(coeffs)

            ax.plot(
                freq_range,
                poly(freq_range),
                "r--",
                linewidth=2,
                alpha=0.7,
                label=f"线性趋势线 (斜率={coeffs[0]:.4f} rad/Hz)",
            )

            # 标记趋势线零点
            if abs(coeffs[0]) > 1e-12:
                zero_freq = -coeffs[1] / coeffs[0]
                ax.axvline(
                    zero_freq,
                    color="green",
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.7,
                )
                ax.scatter(
                    [zero_freq],
                    [0],
                    s=200,
                    marker="*",
                    c="green",
                    edgecolors="darkgreen",
                    linewidths=1.5,
                    zorder=6,
                    label=f"趋势线零点: {zero_freq:.2f}Hz",
                )
        else:
            # Amplitude mode: 连接线 + 峰值标记
            if len(frequencies) < 2:
                return

            # 按频率排序绘制连接线
            sort_idx = np.argsort(frequencies)
            sorted_freqs = frequencies[sort_idx]
            sorted_metrics = metrics[sort_idx]

            ax.plot(
                sorted_freqs,
                sorted_metrics,
                "r--",
                linewidth=1.5,
                alpha=0.6,
                label="测量点连线",
            )

            # 标记当前最大值点
            best_idx = np.argmax(metrics)
            ax.scatter(
                [frequencies[best_idx]],
                [metrics[best_idx]],
                s=200,
                marker="*",
                c="green",
                edgecolors="darkgreen",
                linewidths=1.5,
                zorder=6,
                label=f"最大幅值点: {frequencies[best_idx]:.2f}Hz",
            )

    @property
    def optimal_frequency(self) -> float | None:
        """
        获取优化后的最佳频率

        Returns:
            最佳频率（Hz），如果尚未执行优化则返回None
        """
        return self._optimal_frequency

    @property
    def optimal_sampling_info(self) -> SamplingInfo | None:
        """
        获取优化后的最佳采样信息

        Returns:
            SamplingInfo字典，如果尚未执行优化则返回None
        """
        return self._optimal_sampling_info

    @property
    def measurement_history(
        self,
    ) -> list[dict[str, float | np.ndarray]]:
        """
        获取所有测量历史记录

        Returns:
            测量历史列表，每条记录包含 frequency、metric、amplitudes、phases
        """
        return self._measurement_history


class PowerTester:
    """
    # 功率测试类

    该类专注于测试单个AO通道在特定工作功率下的工作状态。
    通过创建多个CaliberOctopus对象，以不同功率（Amplitude）进行测试，
    观察AO通道（的扬声器）在越来越高功率持续工作下传输特性的变化。

    ## 主要特性：
        - 测试单个AI通道与单个AO通道的组合
        - 从已有的ao_comp_data中读取校准信息
        - 通过mean_amp_ratio和mean_phase_shift还原原始传递函数复数值
        - 按功率从小到大逐步测试AO通道的工作状态
        - 每个功率点进行work（高强度工作模拟）和examine（检测）两阶段测试
        - 生成PowerTest概览图，展示传输特性随功率的变化趋势

    ## 测试流程：
        1. 初始化阶段：
           - 加载ao_comp_data文件（必须成功，否则报错）
           - 从comp_dataframe中获取指定AO通道的补偿信息
           - 还原原始传递函数复数值
           - 读取sampling_info和frequency
        2. start方法执行阶段：
           - 根据min_power、end_power、step_num生成功率序列
           - 对每个功率值创建CaliberOctopus对象进行测试
           - 每个测试包含work和examine两个阶段
        3. 结果汇总阶段：
           - 收集所有examine阶段的传递函数结果
           - 绘制PowerTest概览图（极坐标折线图）

    ## 使用示例：
    ```python
    from sweeper400.use.clb import PowerTester

    # 创建功率测试对象
    tester = PowerTester(
        ai_channel="PXI1Slot2/ai0",
        ao_channel="PXI2Slot2/ao0",
        ao_comp_data="storage/calib/calib_result_octopus/ao_comp_data.pkl"
    )

    # 执行功率测试
    tester.test(
        start_power=0.01,
        end_power=0.05,
        step_num=5,
        work_chunks_num=240,
        result_folder="storage/calib/power_test_result"
    )
    ```

    ## 注意事项：
        - 必须提供有效的ao_comp_data文件，否则初始化失败
        - ao_comp_data中必须包含指定的ao_channel信息
        - 测试前确保硬件连接正确
        - 测试过程可能较长，请耐心等待
    """

    # 获取类日志器（类属性，所有实例共享）
    logger = get_logger(f"{__name__}.PowerTester")

    def __init__(
        self,
        ai_channel: str,
        ao_channel: str,
        ao_comp_data: str | Path | None = None,
    ) -> None:
        """
        初始化功率测试对象

        Args:
            ai_channel: AI 通道名称（单个通道）
            ao_channel: AO 通道名称（单个通道）
            ao_comp_data: AO补偿数据文件路径（必须提供且有效）

        Raises:
            ValueError: 当ao_comp_data为None或文件加载失败时
            RuntimeError: 当指定的AO通道在补偿数据中不存在时
        """
        # 保存通道配置
        self._ai_channel = ai_channel
        self._ao_channel = ao_channel
        self._ao_comp_data_path = ao_comp_data

        # 尝试使用load_data_with_fallback加载ao_comp_data（必须成功）
        default_ao_comp_path = Path("storage/calib/calib_result_octopus/ao_comp_data.pkl")
        loaded_ao_comp_data = load_data_with_fallback(
            explicit_path=ao_comp_data,
            default_path=default_ao_comp_path,
            data_type="AO补偿数据",
        )

        if loaded_ao_comp_data is None:
            error_msg = "未找到AO补偿数据文件"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # 从comp_dataframe中获取指定AO通道的补偿信息
        comp_df = loaded_ao_comp_data["comp_dataframe"]

        if ao_channel not in comp_df.index:
            error_msg = (
                f"AO通道 '{ao_channel}' 在补偿数据中不存在。"
                f"可用通道: {comp_df.index.tolist()}"
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        # 获取该通道的补偿参数
        amp_multiplier = comp_df.loc[ao_channel, "amp_multiplier"]
        time_increment = comp_df.loc[ao_channel, "time_increment"]

        # 通过mean_amp_ratio和mean_phase_shift还原原始TF复数值
        # 补偿数据的定义：理想真值 = 测量值 × amp_multiplier
        # 因此：测量值 = 理想真值 / amp_multiplier
        # 但实际上，comp_data存储的是"如何从测量值补偿到理想真值"
        # 我们需要还原的是该通道相对于平均值的传递函数
        #
        # 根据tf_to_comp的实现：
        # amp_multiplier = mean_amp_ratio / amp_ratio
        # time_increment = (phase_shift - mean_phase_shift) / (2 * pi * frequency)
        #
        # 因此还原原始TF：
        # amp_ratio = mean_amp_ratio / amp_multiplier
        # phase_shift = mean_phase_shift + time_increment * 2 * pi * frequency

        mean_amp_ratio = loaded_ao_comp_data["mean_amp_ratio"]
        mean_phase_shift = loaded_ao_comp_data["mean_phase_shift"]
        frequency = loaded_ao_comp_data["frequency"]

        # 还原原始传递函数的幅值比和相位差
        original_amp_ratio = mean_amp_ratio / amp_multiplier
        original_phase_shift = mean_phase_shift + (
            time_increment * 2.0 * np.pi * frequency
        )
        # 将相位归一化到 [-π, π] 区间
        original_phase_shift = float(
            np.arctan2(np.sin(original_phase_shift), np.cos(original_phase_shift))
        )

        # 构建复数传递函数值
        self._original_tf_complex = original_amp_ratio * np.exp(
            1j * original_phase_shift
        )

        self.logger.info(
            f"还原原始传递函数 - "
            f"AO通道: {ao_channel}, "
            f"幅值比: {original_amp_ratio:.6f}, "
            f"相位差: {original_phase_shift:.6f}rad, "
            f"复数值: {self._original_tf_complex}"
        )

        # 读取并保存sampling_info和频率/相位信息（用于后续测试流程）
        self._sampling_info = loaded_ao_comp_data["sampling_info"]
        self._frequency = frequency
        self._phase = 0.0  # 默认相位，test()中可覆盖

        self.logger.info(
            f"PowerTester 实例已创建 - "
            f"AI通道: {ai_channel}, "
            f"AO通道: {ao_channel}, "
            f"频率: {self._frequency}Hz"
        )

    @property
    def original_tf_complex(self) -> complex:
        """
        获取还原的原始传递函数复数值

        Returns:
            原始传递函数的复数表示
        """
        return self._original_tf_complex

    @property
    def sampling_info(self) -> SamplingInfo:
        """
        获取采样信息

        Returns:
            SamplingInfo对象
        """
        return self._sampling_info

    @property
    def frequency(self) -> float:
        """
        获取正弦波频率

        Returns:
            频率（Hz）
        """
        return self._frequency

    # 近零功率点幅值常量，用于线性度验证
    NEAR_ZERO_POWER: float = 0.01
    # work阶段后的冷却等待时间（秒），用于消除温度对传递函数的影响
    COOLING_TIME: float = 420.0
    # 两次校准测量之间的间隔时间（秒），避免连续测量的热效应累积
    EXAMINE_INTERVAL: float = 90.0

    def test(
        self,
        start_power: PositiveFloat = 0.01,
        end_power: PositiveFloat = 0.02,
        step_num: PositiveInt = 2,
        work_chunks_num: PositiveInt = 120,
        result_folder: str = "storage/calib/power_test_result",
    ) -> None:
        """
        执行功率测试流程

        该方法创建多个CaliberOctopus对象，以不同功率（Amplitude）进行测试，
        观察AO通道在越来越高功率持续工作下传输特性的变化。

        每个功率点的测试包含三个阶段：
        1. work阶段：高强度工作模拟（使用_single_calibrate，不保存数据）
        2. examine阶段：在当前功率点进行校准检测
        3. near-zero examine阶段：在近零功率点(0.01V)进行校准检测，验证线性度

        通过对比两个功率点处的传递函数变化，可以追踪哪个功率点的工作负荷
        破坏了通道的线性度。

        测试完成后，绘制PowerTest概览图，展示传输特性随功率的变化趋势。

        Args:
            start_power: 起始功率（Amplitude），默认0.01
            end_power: 结束功率（Amplitude），默认0.02
            step_num: 功率步数，默认2
            work_chunks_num: work阶段的chunk数量，默认120
            result_folder: 结果保存文件夹路径

        Raises:
            ValueError: 当min_power >= max_power时
        """
        # 参数验证
        # if start_power >= end_power:
        #     error_msg = f"start_power({start_power})必须严格小于max_power({end_power})"
        #     self.logger.error(error_msg)
        #     raise ValueError(error_msg)

        self.logger.info(
            f"开始功率测试 - "
            f"功率范围: {start_power}V ~ {end_power}V, "
            f"步数: {step_num}, "
            f"work_chunks_num: {work_chunks_num}, "
            f"近零功率点: {self.NEAR_ZERO_POWER}V, "
            f"冷却时间: {self.COOLING_TIME}s, "
            f"校准间隔: {self.EXAMINE_INTERVAL}s"
        )

        # 生成功率序列
        power_values = np.linspace(start_power, end_power, step_num)
        self.logger.info(f"功率序列: {power_values}")

        # 确定结果保存路径
        result_path = Path(result_folder)
        result_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"结果保存路径: {result_path}")

        # 存储每个功率点的测试结果
        # 格式: [(power, tf_complex), ...]
        test_results: list[tuple[float, complex]] = []
        # 存储近零功率点的线性度测试结果
        # 格式: [(work_power, tf_complex_at_near_zero), ...]
        near_zero_results: list[tuple[float, complex]] = []

        # 遍历每个功率值进行测试
        for power_idx, power in enumerate(power_values):
            self.logger.info(
                "\n" + "=" * 60
                + f"\n开始第 {power_idx + 1}/{step_num} 个功率点测试: {power}V"
                + "\n" + "=" * 60
            )

            # 创建CaliberOctopus对象
            # 使用单个AI通道和单个AO通道
            caliber = CaliberOctopus(
                ai_channels=(self._ai_channel,),
                ao_channels=(self._ao_channel,),
                sampling_info=self._sampling_info,
                frequency=self._frequency,
                amplitude=float(power),
                ao_comp_data=self._ao_comp_data_path,
            )

            # 第一阶段：work（高强度工作模拟）
            self.logger.info(f"开始work阶段（高强度工作模拟），chunks_num={work_chunks_num}")
            try:
                # 调用_single_calibrate进行工作模拟（不保存数据）
                caliber._single_calibrate(
                    chunks_per_start=work_chunks_num,
                    settle_time=None,  # 使用默认值
                )
                self.logger.info("work阶段完成")
            except Exception as e:
                self.logger.error(f"work阶段发生错误: {e}", exc_info=True)
                # 继续执行examine阶段，即使work阶段失败

            # 冷却等待阶段：消除温度对传递函数的影响
            self.logger.info(
                f"开始冷却等待阶段，等待 {self.COOLING_TIME}s "
                f"以消除温度影响..."
            )
            time.sleep(self.COOLING_TIME)
            self.logger.info("冷却等待完成")

            # 第二阶段：examine（在当前功率点检测）
            self.logger.info("开始examine阶段（当前功率点检测）")

            # 创建当前功率点的结果子文件夹
            power_result_folder = result_path / f"power_{power_idx + 1}_{power:.4f}V"

            try:
                # 调用calibrate进行检测
                # starts_num=3, apply_filter=False（方便观察波形失真）
                caliber.calibrate(
                    starts_num=3,
                    chunks_per_start=3,  # 使用默认值
                    apply_filter=False,  # 不应用滤波，方便观察波形失真
                    result_folder=power_result_folder,
                    settle_time=None,  # 使用默认值
                )

                # 从caliber获取examine阶段的传递函数结果
                tf_data = caliber.result_averaged_tf_data
                if tf_data is not None:
                    tf_df = tf_data["tf_dataframe"]
                    # 提取单个通道的传递函数复数值
                    # tf_dataframe是行矩阵：index为"AI"（固定字符串），columns为AO通道名
                    # 使用iloc[0, 0]获取唯一元素，避免依赖具体的行列名
                    tf_complex = tf_df.iloc[0, 0]
                    test_results.append((float(power), complex(tf_complex)))
                    self.logger.info(
                        f"examine阶段完成 - "
                        f"功率: {power}V, "
                        f"幅值比: {np.abs(tf_complex):.6f}, "
                        f"相位差: {np.angle(tf_complex):.6f}rad"
                    )
                else:
                    self.logger.warning(
                        f"examine阶段未产生有效的传递函数数据，功率: {power}V"
                    )
                    # 使用NaN标记失败的数据点
                    test_results.append((float(power), complex(np.nan, np.nan)))

            except Exception as e:
                self.logger.error(
                    f"examine阶段发生错误: {e}", exc_info=True
                )
                # 使用NaN标记失败的数据点
                test_results.append((float(power), complex(np.nan, np.nan)))

            # 两次校准之间的间隔等待
            self.logger.info(
                f"等待校准间隔 {self.EXAMINE_INTERVAL}s..."
            )
            time.sleep(self.EXAMINE_INTERVAL)

            # 第三阶段：near-zero examine（在近零功率点检测线性度）
            self.logger.info(
                f"开始near-zero examine阶段（近零功率点线性度检测，"
                f"amplitude={self.NEAR_ZERO_POWER}V）"
            )

            # 创建近零功率点的CaliberOctopus对象
            caliber_near_zero = CaliberOctopus(
                ai_channels=(self._ai_channel,),
                ao_channels=(self._ao_channel,),
                sampling_info=self._sampling_info,
                frequency=self._frequency,
                amplitude=self.NEAR_ZERO_POWER,
                ao_comp_data=self._ao_comp_data_path,
            )

            # 创建近零功率点的结果子文件夹
            near_zero_result_folder = (
                result_path / f"power_{power_idx + 1}_{power:.4f}V_near_zero"
            )

            try:
                caliber_near_zero.calibrate(
                    starts_num=3,
                    chunks_per_start=3,
                    apply_filter=False,
                    result_folder=near_zero_result_folder,
                    settle_time=None,
                )

                # 从caliber_near_zero获取近零功率点的传递函数结果
                tf_data_nz = caliber_near_zero.result_averaged_tf_data
                if tf_data_nz is not None:
                    tf_df_nz = tf_data_nz["tf_dataframe"]
                    tf_complex_nz = tf_df_nz.iloc[0, 0]
                    near_zero_results.append(
                        (float(power), complex(tf_complex_nz))
                    )
                    self.logger.info(
                        f"near-zero examine阶段完成 - "
                        f"work功率: {power}V, "
                        f"近零测量幅值比: {np.abs(tf_complex_nz):.6f}, "
                        f"近零测量相位差: {np.angle(tf_complex_nz):.6f}rad"
                    )
                else:
                    self.logger.warning(
                        f"near-zero examine阶段未产生有效数据，work功率: {power}V"
                    )
                    near_zero_results.append(
                        (float(power), complex(np.nan, np.nan))
                    )

            except Exception as e:
                self.logger.error(
                    f"near-zero examine阶段发生错误: {e}", exc_info=True
                )
                near_zero_results.append(
                    (float(power), complex(np.nan, np.nan))
                )

        # 绘制PowerTest概览图
        self.logger.info("=" * 60)
        self.logger.info("绘制PowerTest概览图")
        self.logger.info("=" * 60)

        self._plot_power_test_overview(test_results, near_zero_results, result_path)

        self.logger.info("=" * 60)
        self.logger.info(f"功率测试完成，所有结果已保存到: {result_path}")
        self.logger.info("=" * 60)

    def _plot_power_test_overview(
        self,
        test_results: list[tuple[float, complex]],
        near_zero_results: list[tuple[float, complex]],
        save_path: Path,
    ) -> None:
        """
        绘制PowerTest概览图

        类似于CaliberOctopus的plot_comp_data所输出的polar图，
        但数据点是各个功率点的传递函数结果，并使用折线连接。
        包含两条折线：当前功率点线性度和近零功率点线性度。

        Args:
            test_results: 测试结果列表，格式为[(power, tf_complex), ...]
            near_zero_results: 近零功率点测试结果列表，
                              格式为[(work_power, tf_complex_at_near_zero), ...]
            save_path: 结果保存路径
        """
        # 过滤掉无效数据点（NaN）
        valid_results = [
            (power, tf) for power, tf in test_results if not np.isnan(tf.real)
        ]
        valid_near_zero_results = [
            (power, tf) for power, tf in near_zero_results if not np.isnan(tf.real)
        ]

        if not valid_results and not valid_near_zero_results:
            self.logger.warning("没有有效的测试结果，无法绘制概览图")
            return

        # 创建极坐标图
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="polar")

        # 绘制当前功率点折线
        if valid_results:
            powers = [r[0] for r in valid_results]
            tf_complexes = [r[1] for r in valid_results]
            amp_ratios = [np.abs(tf) for tf in tf_complexes]
            phase_shifts = [np.angle(tf) for tf in tf_complexes]

            ax.plot(
                phase_shifts,
                amp_ratios,
                marker="o",
                markersize=10,
                linewidth=2,
                color="blue",
                alpha=0.8,
                label="当前功率点线性度",
            )

            # 添加功率标注
            for angle, magnitude, power in zip(
                phase_shifts, amp_ratios, powers, strict=True
            ):
                ax.annotate(
                    f"{power:.4f}V",
                    xy=(angle, magnitude),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                    alpha=0.8,
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": "lightskyblue",
                        "alpha": 0.7,
                    },
                    zorder=4,
                )

            # 标记起始点（最小功率）
            ax.scatter(
                [phase_shifts[0]],
                [amp_ratios[0]],
                c="green",
                s=200,
                marker="s",
                label=f"起始 ({powers[0]:.4f}V)",
                zorder=5,
            )

            # 标记终点（最大功率）
            if len(phase_shifts) > 1:
                ax.scatter(
                    [phase_shifts[-1]],
                    [amp_ratios[-1]],
                    c="red",
                    s=200,
                    marker="^",
                    label=f"终点 ({powers[-1]:.4f}V)",
                    zorder=5,
                )

        # 绘制近零功率点折线
        if valid_near_zero_results:
            nz_powers = [r[0] for r in valid_near_zero_results]
            nz_tf_complexes = [r[1] for r in valid_near_zero_results]
            nz_amp_ratios = [np.abs(tf) for tf in nz_tf_complexes]
            nz_phase_shifts = [np.angle(tf) for tf in nz_tf_complexes]

            ax.plot(
                nz_phase_shifts,
                nz_amp_ratios,
                marker="D",
                markersize=8,
                linewidth=2,
                color="purple",
                alpha=0.8,
                linestyle="--",
                label=f"近零功率点线性度 ({self.NEAR_ZERO_POWER}V)",
            )

            # 添加近零功率点标注（标注对应的work功率）
            for angle, magnitude, work_power in zip(
                nz_phase_shifts, nz_amp_ratios, nz_powers, strict=True
            ):
                ax.annotate(
                    f"w:{work_power:.4f}V",
                    xy=(angle, magnitude),
                    xytext=(-5, -10),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.7,
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": "plum",
                        "alpha": 0.7,
                    },
                    zorder=4,
                )

        # 标记原始传递函数（从ao_comp_data还原的值）
        original_amp = np.abs(self._original_tf_complex)
        original_phase = np.angle(self._original_tf_complex)
        ax.scatter(
            [original_phase],
            [original_amp],
            c="orange",
            s=200,
            marker="*",
            label="原始TF (校准值)",
            zorder=5,
        )

        # 设置极坐标图样式
        ax.set_theta_zero_location("E")  # 0度在右侧
        ax.set_theta_direction(1)  # 逆时针为正
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)

        # 设置标题
        ax.set_title(
            f"PowerTest 传递函数极坐标分布\n"
            f"AI: {self._ai_channel}, AO: {self._ao_channel}\n"
            f"频率: {self._frequency}Hz",
            fontsize=14,
            pad=20,
        )

        # 添加图例
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=10)

        # 保存图像
        plot_path = save_path / "power_test_overview.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        self.logger.info(f"PowerTest概览图已保存到: {plot_path}")

        plt.show()
        self.logger.info("PowerTest概览图绘制完成")

        # 额外绘制一幅直角坐标图，展示幅值比和相位差随功率的变化
        self._plot_power_test_cartesian(
            valid_results, valid_near_zero_results, save_path
        )

    def _plot_power_test_cartesian(
        self,
        valid_results: list[tuple[float, complex]],
        valid_near_zero_results: list[tuple[float, complex]],
        save_path: Path,
    ) -> None:
        """
        绘制PowerTest直角坐标图

        展示幅值比和相位差随功率的变化趋势，包含当前功率点和近零功率点两条折线。

        Args:
            valid_results: 有效测试结果列表，格式为[(power, tf_complex), ...]
            valid_near_zero_results: 有效近零功率点测试结果列表，
                                    格式为[(work_power, tf_complex_at_near_zero), ...]
            save_path: 结果保存路径
        """
        # 创建直角坐标图（两个子图）
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 标记原始传递函数的幅值比和相位差
        original_amp = np.abs(self._original_tf_complex)
        original_phase = np.angle(self._original_tf_complex)

        # --- 当前功率点数据 ---
        if valid_results:
            powers = [r[0] for r in valid_results]
            tf_complexes = [r[1] for r in valid_results]
            amp_ratios = [np.abs(tf) for tf in tf_complexes]
            phase_shifts = [np.angle(tf) for tf in tf_complexes]

            # 第一幅图：幅值比 vs 功率
            ax1.plot(
                powers,
                amp_ratios,
                marker="o",
                markersize=8,
                linewidth=2,
                color="blue",
                alpha=0.8,
                label="当前功率点线性度",
            )

            # 第二幅图：相位差 vs 功率
            ax2.plot(
                powers,
                phase_shifts,
                marker="o",
                markersize=8,
                linewidth=2,
                color="blue",
                alpha=0.8,
                label="当前功率点线性度",
            )

        # --- 近零功率点数据 ---
        if valid_near_zero_results:
            nz_powers = [r[0] for r in valid_near_zero_results]
            nz_tf_complexes = [r[1] for r in valid_near_zero_results]
            nz_amp_ratios = [np.abs(tf) for tf in nz_tf_complexes]
            nz_phase_shifts = [np.angle(tf) for tf in nz_tf_complexes]

            # 第一幅图：近零功率点幅值比 vs work功率
            ax1.plot(
                nz_powers,
                nz_amp_ratios,
                marker="D",
                markersize=7,
                linewidth=2,
                color="purple",
                alpha=0.8,
                linestyle="--",
                label=f"近零功率点线性度 ({self.NEAR_ZERO_POWER}V)",
            )

            # 第二幅图：近零功率点相位差 vs work功率
            ax2.plot(
                nz_powers,
                nz_phase_shifts,
                marker="D",
                markersize=7,
                linewidth=2,
                color="purple",
                alpha=0.8,
                linestyle="--",
                label=f"近零功率点线性度 ({self.NEAR_ZERO_POWER}V)",
            )

        # 标记原始传递函数参考线
        ax1.axhline(
            y=original_amp,
            color="orange",
            linestyle="-.",
            linewidth=1.5,
            label=f"原始TF幅值比: {original_amp:.6f}",
        )

        ax1.set_xlabel("work功率 (V)", fontsize=12)
        ax1.set_ylabel("幅值比", fontsize=12)
        ax1.set_title(
            f"幅值比随功率变化\nAI: {self._ai_channel}, AO: {self._ao_channel}",
            fontsize=14,
            pad=15,
        )
        ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
        ax1.legend(loc="best", fontsize=10)

        ax2.axhline(
            y=original_phase,
            color="orange",
            linestyle="-.",
            linewidth=1.5,
            label=f"原始TF相位差: {original_phase:.6f} rad",
        )

        ax2.set_xlabel("work功率 (V)", fontsize=12)
        ax2.set_ylabel("相位差 (rad)", fontsize=12)
        ax2.set_title(
            f"相位差随功率变化\n频率: {self._frequency}Hz",
            fontsize=14,
            pad=15,
        )
        ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
        ax2.legend(loc="best", fontsize=10)

        plt.tight_layout()

        # 保存图像
        plot_path = save_path / "power_test_cartesian.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        self.logger.info(f"PowerTest直角坐标图已保存到: {plot_path}")

        plt.show()
        self.logger.info("PowerTest直角坐标图绘制完成")
