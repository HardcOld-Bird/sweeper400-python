"""
# 反馈演化器模块

模块路径：`sweeper400.use.evolver`

该模块包含 Evolver 类，提供基于反馈的声场演化控制功能。
与 SweeperCore 类似，Evolver 基于 SingleChasCSIO 构建，但其核心目标不是空间扫场测量，
而是通过反馈控制，让各 AI 通道的声场按照指定的"增益系数"进行演化。

每一个演化周期，Evolver 都会：
1. 对当前采集到的 AI 波形和当前播放的反馈波形进行滤波处理；
2. 利用 fishnet_tf_data（传递函数矩阵），实时计算所需的 AO 输出；
3. 以迭代的方式逐步调整每个 AO 通道的输出，使每个 AI 通道的
   "总声场复振幅" 与 "入射声场复振幅" 之比逼近用户指定的增益系数；
4. 将每一周期的"总声场复振幅"储存下来，以便事后绘图分析。
"""

import threading
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter

from ..analyze import (
    Point2D,
    PositiveFloat,
    PositiveInt,
    SineArgs,
    SweepData,
    Waveform,
    TFData,
    average_single_waveform,
    detrend_waveform,
    esti_vvi_multi_ch,
    filter_waveform,
    get_sine_multi_ch,
    save_compressed_data,
    load_compressed_data,
)
from ..measure import SingleChasCSIO
from ..logger import get_logger

# 获取模块日志器
logger = get_logger(__name__)


class Evolver:
    """
    # 反馈演化器

    基于 SingleChasCSIO 的反馈演化控制器。与扫场不同，Evolver 不控制步进电机进行空间移动，
    而是专注于通过多通道反馈控制，让声场按照指定的增益系数进行演化。

    ## 工作原理：

    用户指定一组"增益系数"（每个反馈 AO 通道对应一个复数），Evolver 每一演化周期都会：

    1. 对当前 AI 波形和正在播放的反馈 AO 波形进行去趋势和带通滤波；
    2. 用 `esti_vvi_multi_ch` 计算 AI 各通道的"总声场复振幅"和各 AO 通道
       的"旧电输出复振幅"；
    3. 利用 fishnet_tf_data 中的传递函数，从"总声场复振幅"中扣除各 AO 通道的
       贡献，得到"入射声场复振幅"；
    4. 根据目标增益系数，计算所需声场增量，再转换为所需 AO 输出增量，更新下一轮
       的反馈 AO 输出复振幅；
    5. 如果任意通道的新 AO 幅值超过安全上限，则立即报错并终止演化；
    6. 将每一周期的"总声场复振幅"记录在内部的 SweepData 中（x 坐标为周期序号），
       供事后调用 `plot_evolution` 绘图分析。

    ## 使用方式：

    ```python
    from sweeper400.analyze import init_sampling_info, init_sine_args, get_sine_cycles
    from sweeper400.use import Evolver, load_evolved_waveform

    # 创建采样信息和静态输出波形（建议时长较长，以给反馈处理留出充足时间）
    sampling_info = init_sampling_info(171500, 85750)  # 约 0.5s 每 chunk
    sine_args = init_sine_args(3430.0, 0.05, 0.0)
    static_waveform = get_sine_cycles(sampling_info, sine_args, cycles=25)

    # 增益系数（每个反馈 AO 通道一个复数）
    # 复数模长 > 1 表示放大，< 1 表示衰减，相位表示相移
    gain_coefficients = (1.5 + 0.0j,) * 8  # 8个通道，均增益1.5倍

    evolver = Evolver(
        ai_channels=("PXI1Slot2/ai0", ...),
        ao_channels_static=("PXI1Slot2/ao0",),
        ao_channels_feedback=("PXI1Slot3/ao0", ...),
        static_output_waveform=static_waveform,
        gain_coefficients=gain_coefficients,
    )

    # 执行10个演化周期，并自动保存结果
    final_wf = evolver.evolve(
        num_cycles=10,
        result_folder="output/evolution_result"
    )

    # 之后可以直接加载保存的波形，用于 SweeperCore 扫场（无需反馈）
    evolved_wf = load_evolved_waveform("output/evolution_result/evolved_waveform.pkl")
    # sweeper = SweeperCore(
    #     ai_channels=(...),
    #     ao_channels_static=evolved_wf.channel_names,
    #     static_output_waveform=evolved_wf,
    #     point_list=grid,
    # )
    ```

    ## 注意事项：

    - `static_output_waveform` 的时长即为每个演化周期的时长，建议较长（推荐 0.5s 以上），
      以给 feedback_function 中较重的计算留出足够时间。
    - `gain_coefficients` 的长度必须与 `ao_channels_feedback` 的长度相同，
      且与 AI 通道的数量相同（每个 AO 反馈通道对应一个 AI 通道）。
    - 演化过程中若任意通道的 AO 幅值超过 `ao_amplitude_limit`（默认 0.1 V），
      将立即报错并终止演化。
    - 演化数据储存在 `_evolution_data`（SweepData 格式）中，x 坐标即为周期序号（从 1 开始）。
    - `evolve()` 成功完成后会返回一个多通道合并 Waveform（static + feedback 通道），
      该波形可直接作为 `SweeperCore` 的 `static_output_waveform` 使用。
    - 使用 `load_evolved_waveform()` 函数可从文件加载之前保存的演化波形。
    """

    # 获取类日志器
    logger = get_logger(f"{__name__}.Evolver")

    def __init__(
        self,
        ai_channels: tuple[str, ...],
        ao_channels_static: tuple[str, ...],
        ao_channels_feedback: tuple[str, ...],
        static_output_waveform: Waveform,
        gain_coefficients: tuple[complex, ...],
    ) -> None:
        """
        初始化反馈演化器

        Args:
            ai_channels: AI 通道名称元组，例如 ("PXI1Slot2/ai0", "PXI1Slot3/ai0", ...)
                数量应与 ao_channels_feedback 相同（每个 AO 通道对应一个 AI 通道）。
            ao_channels_static: 静态 AO 通道名称元组，例如 ("PXI1Slot2/ao0",)，
                静态 AO 用于输出主激励信号（不参与反馈调整）。
            ao_channels_feedback: 反馈 AO 通道名称元组，例如 ("PXI1Slot3/ao0", ...)，
                这些通道的输出将被反馈函数实时调整。
            static_output_waveform: 静态主激励输出波形。该波形的 SineArgs（频率、幅值、相位）
                将被保存并在反馈函数中反复使用。**建议时长较长（0.5s 以上）**，
                以给每轮 callback 中的信号处理留出足够的执行时间。
            gain_coefficients: 增益系数元组（每个 feedback AO 通道对应一个复数），
                决定每个 AI 通道的"总声场复振幅" / "入射声场复振幅"的目标比值。
                - 长度必须与 `ao_channels_feedback`（及 `ai_channels`）相同。
                - 模长 > 1 表示增益，< 1 表示衰减，相位角表示声场相移目标。

        Raises:
            ValueError: 当 gain_coefficients 长度与 ao_channels_feedback 不同时；
                        或 gain_coefficients 长度与 ai_channels 不同时；
                        或 static_output_waveform 没有 sine_args 属性时。
            RuntimeError: 当 SingleChasCSIO 初始化失败时。
        """
        # 参数验证
        n_fb = len(ao_channels_feedback)
        n_ai = len(ai_channels)
        n_gain = len(gain_coefficients)

        if n_gain != n_fb:
            raise ValueError(
                f"gain_coefficients 长度 ({n_gain}) 必须与 "
                f"ao_channels_feedback 长度 ({n_fb}) 相同"
            )
        if n_gain != n_ai:
            raise ValueError(
                f"gain_coefficients 长度 ({n_gain}) 必须与 "
                f"ai_channels 长度 ({n_ai}) 相同"
            )
        if static_output_waveform.sine_args is None:
            raise ValueError(
                "static_output_waveform 必须包含 sine_args 属性，"
                "请使用 get_sine_cycles 或 get_sine_multi_ch 生成波形。"
            )

        # 保存通道配置
        self._ai_channels = ai_channels
        self._ao_channels_static = ao_channels_static
        self._ao_channels_feedback = ao_channels_feedback

        # 保存增益系数（转为复数 ndarray，便于向量化运算）
        self._gain_coefficients = np.array(gain_coefficients, dtype=complex)

        # 保存静态输出波形的 SineArgs（频率、幅值、相位，反馈函数中反复使用）
        self._static_sine_args: SineArgs = static_output_waveform.sine_args
        self._static_output_waveform = static_output_waveform

        # ---- 反馈函数状态 ----
        # 当前各 AO 反馈通道的输出复振幅（向量化）
        # 初始值为 0（静默状态，等待第一轮反馈后才开始调整）
        self._current_ao_complex_amps = np.zeros(n_fb, dtype=complex)

        # 每次反馈后储存的"总声场复振幅"（每轮一个 ndarray，形状 (n_ai,)）
        # 储存在 _ai_complex_amps_history 中，按周期序号索引（从 1 开始）
        self._ai_complex_amps_history: list[np.ndarray] = []

        # 线程同步
        self._cycle_done_event = threading.Event()
        self._stop_flag = False
        self._evolve_error: Exception | None = None

        # SweepData 格式的演化数据（x 坐标为周期序号，y 坐标置 0）
        self._evolution_data: SweepData = {
            "ai_data_list": [],
            "ao_data": static_output_waveform,
        }

        # 创建 SingleChasCSIO 实例
        try:
            self.logger.debug("正在初始化 SingleChasCSIO...")
            self._measure_controller = SingleChasCSIO(
                ai_channels=ai_channels,
                ao_channels_static=ao_channels_static,
                ao_channels_feedback=ao_channels_feedback,
                static_output_waveform=static_output_waveform,
                export_function=self._data_export_callback,
                feedback_function=self._feedback_method,
            )
            self.logger.debug("SingleChasCSIO 初始化成功")
        except Exception as e:
            error_msg = f"SingleChasCSIO 初始化失败: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

        self.logger.info(
            f"Evolver 初始化完成 - "
            f"AI通道: {ai_channels}, "
            f"Static AO通道: {ao_channels_static}, "
            f"Feedback AO通道: {ao_channels_feedback}, "
            f"增益系数: {gain_coefficients}"
        )

    # =========================================================================
    #  内部反馈函数（作为类方法，以便访问 self 中的状态）
    # =========================================================================

    def _feedback_method(
        self,
        ai_waveform: Waveform,
        static_output_waveform: Waveform | None,
        currently_playing_feedback_waveform: Waveform | None,
        fishnet_tf_data: TFData | None,
    ) -> Waveform:
        """
        反馈函数（由 SingleChasCSIO 在后台工作线程中每周期调用一次）

        该方法执行每个演化周期的核心反馈控制逻辑：
        1. 对 AI 波形和当前反馈 AO 波形进行去趋势和带通滤波，获得纯净信号；
        2. 用 esti_vvi_multi_ch 计算各通道的复振幅（总声场、旧电输出）；
        3. 利用 fishnet_tf_data，从总声场中扣除各 AO 通道的贡献，得到入射声场复振幅；
        4. 依据增益系数，计算所需 AO 输出增量，更新下一轮反馈波形；
        5. 幅值安全检查：若任意通道超限，设置 _stop_flag 并抛出异常。

        Args:
            ai_waveform: 当前采集到的多通道 AI 波形（n_ai_ch, n_samples）
            static_output_waveform: 当前的静态输出波形（由 SingleChasCSIO 传入，可能为 None）
            currently_playing_feedback_waveform: 当前正在播放的反馈 AO 波形（可能为 None）
            fishnet_tf_data: Fishnet 传递函数数据（TFData），包含各 AO→AI 通道对的复数传递函数

        Returns:
            新的反馈 AO 多通道波形（n_fb_ch, n_samples）

        Raises:
            RuntimeError: 若 fishnet_tf_data 为 None（必须提供），
                          或若任意通道的新 AO 幅值超过 ao_amplitude_limit。
        """
        # =====================================================================
        # Step 0: 前置检查
        # =====================================================================
        if fishnet_tf_data is None:
            msg = "Evolver._feedback_method 需要 fishnet_tf_data，但收到 None。"
            self.logger.error(msg)
            self._stop_flag = True
            raise RuntimeError(msg)

        if currently_playing_feedback_waveform is None:
            # 首次 callback 时队列可能还为空，返回全零波形
            self.logger.warning("currently_playing_feedback_waveform 为 None，跳过本轮反馈")
            return self._make_silence_waveform()

        # =====================================================================
        # Step 1: 去趋势 + 带通滤波
        # =====================================================================
        freq = self._static_sine_args["frequency"]
        lowcut = freq * 0.5
        highcut = freq * 2.0
        sampling_rate = ai_waveform.sampling_rate

        sos = butter(
            N=4,
            Wn=[lowcut, highcut],
            btype="bandpass",
            analog=False,
            output="sos",
            fs=sampling_rate,
        )

        # 非连续滤波（zi=None，不传递状态）
        ai_filtered: Waveform = filter_waveform(ai_waveform, sos)

        # =====================================================================
        # Step 2: 计算复振幅
        # =====================================================================
        logger.warning("开始单频检测")
        # 总声场复振幅（AI 各通道）
        total_ai_complex_amps = esti_vvi_multi_ch(
            ai_filtered,
            approx_freq=freq,
            use_curve_fit=False,
        )  # shape: (n_ai,)

        # 旧电输出复振幅（当前反馈 AO 各通道）
        old_ao_complex_amps = esti_vvi_multi_ch(
            currently_playing_feedback_waveform,
            approx_freq=freq,
            # use_curve_fit=True,
        )  # shape: (n_fb,)
        logger.warning("结束单频检测")

        # 储存本轮的总声场复振幅（用于事后绘图）
        self._ai_complex_amps_history.append(total_ai_complex_amps.copy())
        self.logger.debug(
            f"第 {len(self._ai_complex_amps_history)} 轮总声场复振幅: "
            f"{total_ai_complex_amps}"
        )

        # =====================================================================
        # Step 3: 从总声场中扣除各 AO 通道贡献，获得入射声场复振幅
        # =====================================================================
        # fishnet_tf_data["tf_dataframe"]: DataFrame
        #   - index: AO 通道名称
        #   - columns: AI 通道名称
        #   - 值: 复数传递函数（AO→AI）
        tf_df = fishnet_tf_data["tf_dataframe"]

        n_fb = len(self._ao_channels_feedback)

        # 构建传递函数向量：每个 AO 对应的 AI 通道的传递函数
        # tf_vec[i] = TF(ao_channels_feedback[i] -> ai_channels[i])
        tf_vec = np.zeros(n_fb, dtype=complex)
        for i in range(n_fb):
            ao_ch = self._ao_channels_feedback[i]
            ai_ch = self._ai_channels[i]
            try:
                tf_vec[i] = tf_df.loc[ao_ch, ai_ch]
            except KeyError as e:
                msg = (
                    f"fishnet_tf_data 中找不到通道对 ({ao_ch} -> {ai_ch})，"
                    f"请检查 fishnet_tf_data 的 AO/AI 通道名称是否正确。"
                )
                self.logger.error(msg)
                self._stop_flag = True
                raise RuntimeError(msg) from e

        # 入射声场复振幅 = 总声场 - AO 贡献（向量化）
        # incident[i] = total_ai[i] - old_ao[i] * tf_vec[i]
        incident_complex_amps = total_ai_complex_amps - old_ao_complex_amps * tf_vec

        self.logger.debug(
            f"入射声场复振幅: {incident_complex_amps}"
        )

        # =====================================================================
        # Step 4: 计算新的反馈 AO 复振幅
        # =====================================================================
        # 目标总声场 = 入射声场 * 增益系数
        target_total_ai = incident_complex_amps * self._gain_coefficients

        # 所需声场增量 = 目标总声场 - 当前总声场
        delta_ai = target_total_ai - total_ai_complex_amps

        # 所需 AO 增量 = 声场增量 / 传递函数（向量化复数除法）
        delta_ao = delta_ai / tf_vec  # 如果 tf_vec 某项为 0，会产生 inf，下方会检查

        # 新 AO 复振幅 = 旧 AO 复振幅 + AO 增量
        new_ao_complex_amps = old_ao_complex_amps + delta_ao

        # 更新内部缓存的 AO 复振幅
        self._current_ao_complex_amps = new_ao_complex_amps.copy()

        self.logger.debug(
            f"新 AO 复振幅: {new_ao_complex_amps}, "
            f"模长: {np.abs(new_ao_complex_amps)}"
        )

        # =====================================================================
        # Step 5: 幅值安全检查
        # =====================================================================
        max_amplitude = np.max(np.abs(new_ao_complex_amps))
        if max_amplitude > self._ao_amplitude_limit:
            exceeding_indices = np.where(
                np.abs(new_ao_complex_amps) > self._ao_amplitude_limit
            )[0]
            msg = (
                f"反馈 AO 幅值超过安全上限 {self._ao_amplitude_limit} V！"
                f"超限通道索引: {exceeding_indices.tolist()}, "
                f"超限幅值: {np.abs(new_ao_complex_amps)[exceeding_indices].tolist()}"
            )
            self.logger.error(msg)
            self._stop_flag = True
            self._evolve_error = RuntimeError(msg)
            # 通知 evolve 方法终止
            self._cycle_done_event.set()
            raise RuntimeError(msg)

        # =====================================================================
        # Step 6: 基于新复振幅生成新的反馈 AO 波形
        # =====================================================================
        new_feedback_waveform = get_sine_multi_ch(
            sampling_info=self._static_output_waveform.sampling_info,
            sine_args=self._static_sine_args,
            channel_names=self._ao_channels_feedback,
            complex_amps=tuple(new_ao_complex_amps.tolist()),
        )

        # 通知 evolve 方法一个周期已完成
        self._cycle_done_event.set()

        return new_feedback_waveform

    def _make_silence_waveform(self) -> Waveform:
        """
        生成全零静音反馈波形（与 static_output_waveform 采样信息一致）

        Returns:
            全零多通道 Waveform，通道数与 ao_channels_feedback 相同
        """
        n_fb = len(self._ao_channels_feedback)
        samples_num = self._static_output_waveform.samples_num
        sampling_rate = self._static_output_waveform.sampling_rate

        silence_data = np.zeros((n_fb, samples_num), dtype=np.float64)
        return Waveform(
            input_array=silence_data,
            sampling_rate=sampling_rate,
            channel_names=self._ao_channels_feedback,
        )

    def _build_final_waveform(self) -> Waveform:
        """
        构建 evolve 最终状态的多通道合并波形

        将静态 AO 通道信号和反馈 AO 通道信号拼接为一个多通道 Waveform，
        使得该波形可以直接作为 SweeperCore 的 `static_output_waveform` 使用，
        从而无需反馈逻辑即可重放出与 evolve 最终状态一致的声场。

        合并波形的 `sine_args` 属性将设置为原始主激励的 SineArgs。

        Returns:
            多通道合并 Waveform，通道顺序为：
            `ao_channels_static` + `ao_channels_feedback`

        Raises:
            ValueError: 当静态波形通道数与 `ao_channels_static` 不匹配时。
        """
        # ---- 处理 static 部分 ----
        # 如果原始波形是单通道但 ao_channels_static 有多个通道，
        # 需要像 SingleChasCSIO 一样扩展为多通道
        if (
            self._static_output_waveform.is_single_channel
            and len(self._ao_channels_static) > 1
        ):
            self.logger.debug(
                f"静态波形为单通道，扩展为 {len(self._ao_channels_static)} 通道"
            )
            static_wf = get_sine_multi_ch(
                sampling_info=self._static_output_waveform.sampling_info,
                sine_args=self._static_sine_args,
                channel_names=self._ao_channels_static,
            )
        elif self._static_output_waveform.channels_num != len(
            self._ao_channels_static
        ):
            raise ValueError(
                f"静态波形通道数 ({self._static_output_waveform.channels_num}) "
                f"与 ao_channels_static 长度 ({len(self._ao_channels_static)}) 不匹配"
            )
        else:
            static_wf = self._static_output_waveform.copy()

        # 确保 static 波形具有正确的 channel_names
        if static_wf.channel_names is None:
            static_wf.channel_names = self._ao_channels_static
        elif static_wf.channel_names != self._ao_channels_static:
            static_wf = Waveform(
                input_array=static_wf.copy(),
                sampling_rate=static_wf.sampling_rate,
                channel_names=self._ao_channels_static,
                timestamp=static_wf.timestamp,
                waveform_id=static_wf.waveform_id,
                sine_args=static_wf.sine_args,
            )

        # ---- 生成 feedback 部分的波形 ----
        if len(self._ao_channels_feedback) > 0:
            feedback_wf = get_sine_multi_ch(
                sampling_info=self._static_output_waveform.sampling_info,
                sine_args=self._static_sine_args,
                channel_names=self._ao_channels_feedback,
                complex_amps=tuple(self._current_ao_complex_amps.tolist()),
            )

            # 拼接 static 和 feedback 波形
            combined_data = np.vstack([static_wf, feedback_wf])
            combined_channel_names = (
                self._ao_channels_static + self._ao_channels_feedback
            )
        else:
            # 没有 feedback 通道，直接使用 static 波形
            combined_data = static_wf.copy()
            combined_channel_names = self._ao_channels_static

        # 创建合并波形，sine_args 使用主激励的 SineArgs
        final_waveform = Waveform(
            input_array=combined_data,
            sampling_rate=static_wf.sampling_rate,
            channel_names=combined_channel_names,
            sine_args=self._static_sine_args,
        )

        self.logger.debug(
            f"构建最终合并波形完成: shape={final_waveform.shape}, "
            f"channels={final_waveform.channel_names}"
        )

        return final_waveform

    # =========================================================================
    #  数据导出回调（由 SingleChasCSIO 每 chunk 调用）
    # =========================================================================

    def _data_export_callback(
        self,
        ai_waveform: Waveform,
        ao_static_waveform: Waveform,
        ao_feedback_waveform: Waveform | None,
        chunks_num: int,
    ) -> None:
        """
        数据导出回调函数（将每周期的 AI 波形记录到 _evolution_data 中）

        x 坐标使用周期序号（chunks_num），y 坐标置 0。
        这里采集的是所有 AI 通道的多通道波形，储存在 SweepData 的 ai_data_list 中。

        Args:
            ai_waveform: 采集到的多通道 AI 波形
            ao_static_waveform: 当前静态 AO 波形
            ao_feedback_waveform: 当前反馈 AO 波形（可能为 None）
            chunks_num: 当前周期序号（从 1 开始）
        """
        # 将数据以 PointSweepData 格式存入（x 坐标为周期序号，y=0）
        point_data = {
            "position": Point2D(float(chunks_num), 0.0),
            "ai_data": [ai_waveform],
        }
        self._evolution_data["ai_data_list"].append(point_data)

        self.logger.debug(f"已记录第 {chunks_num} 个周期的 AI 波形")

        # 达到目标周期数时设置事件
        if chunks_num >= self._target_num_cycles:
            self._cycle_done_event.set()

    # =========================================================================
    #  主控方法：evolve
    # =========================================================================

    def evolve(
        self,
        num_cycles: PositiveInt = 10,
        ao_amplitude_limit: PositiveFloat = 0.1,
        result_folder: str | Path | None = None,
    ) -> Waveform | None:
        """
        启动反馈演化过程（阻塞执行）

        该方法将启动 SingleChasCSIO 任务，执行指定数量的反馈演化周期，
        然后自动停止任务并返回。

        演化过程：
        1. 重置内部状态；
        2. 启动数据采集和 AO 输出任务；
        3. 等待反馈函数执行 num_cycles 次；
        4. 停止任务，构建并返回最终合并波形；
        5. 如果提供了 `result_folder`，自动保存最终波形和演化轨迹图。

        如果反馈过程中任意通道的 AO 幅值超过 `ao_amplitude_limit`，
        将立即停止并抛出 RuntimeError。

        Args:
            num_cycles: 演化周期数，默认 10。每个周期的时长等于
                static_output_waveform 的时长。
            ao_amplitude_limit: 反馈 AO 幅值安全上限（V），默认 0.1 V。
                若任意通道的新 AO 幅值超过此值，将立即终止演化并报错。
            result_folder: 结果保存文件夹路径，如果提供则自动保存：
                - 最终合并波形（`evolved_waveform.pkl`）
                - 演化轨迹图（`evolution.png`）
                该波形可直接作为 SweeperCore 的 `static_output_waveform` 使用，
                无需反馈逻辑即可重放出与 evolve 最终状态一致的声场。

        Returns:
            最终合并的多通道 Waveform（static + feedback 通道），
            如果演化未成功完成则返回 None。

        Raises:
            RuntimeError: 若演化过程中 AO 幅值超限，或任务启动/停止失败。

        Examples:
            ```python
            # 仅执行演化
            final_wf = evolver.evolve(num_cycles=20, ao_amplitude_limit=0.05)

            # 执行演化并自动保存结果
            evolver.evolve(
                num_cycles=20,
                result_folder="output/evolution_result"
            )
            ```
        """
        # ---- 重置内部状态 ----
        self._target_num_cycles = num_cycles
        self._ao_amplitude_limit = ao_amplitude_limit
        self._stop_flag = False
        self._evolve_error = None
        self._cycle_done_event.clear()
        self._ai_complex_amps_history.clear()
        self._current_ao_complex_amps = np.zeros(len(self._ao_channels_feedback), dtype=complex)
        self._evolution_data["ai_data_list"].clear()

        # 初始化返回值（若演化未成功完成则保持为 None）
        final_waveform: Waveform | None = None

        self.logger.info("=" * 60)
        self.logger.info("开始反馈演化")
        self.logger.info(f"演化周期数: {num_cycles}")
        self.logger.info(f"AO 幅值安全上限: {ao_amplitude_limit} V")
        self.logger.info(
            f"每周期时长: {self._static_output_waveform.duration:.3f} s "
            f"（预计总时长: {num_cycles * self._static_output_waveform.duration:.1f} s）"
        )
        self.logger.info("=" * 60)

        try:
            # 启动采集任务
            self._measure_controller.start()
            self.logger.info("SingleChasCSIO 任务已启动")

            # 启用数据导出
            self._measure_controller.enable_export = True

            # 等待足够的周期完成
            # 超时时间 = 每个周期时长 * (num_cycles + 3) + 10s（余量）
            chunk_duration = self._static_output_waveform.duration
            total_timeout = chunk_duration * (num_cycles + 3) + 10.0

            self.logger.debug(f"等待演化完成，超时时间: {total_timeout:.1f} s")

            while True:
                # 等待 cycle_done_event（由 feedback 或 export callback 触发）
                triggered = self._cycle_done_event.wait(timeout=chunk_duration + 5.0)
                self._cycle_done_event.clear()

                if not triggered:
                    self.logger.warning("演化周期等待超时，继续等待...")
                    continue

                # 检查是否有错误（幅值超限等）
                if self._evolve_error is not None:
                    raise self._evolve_error

                # 检查是否已完成足够的周期数
                completed_cycles = len(self._ai_complex_amps_history)
                self.logger.info(f"已完成 {completed_cycles}/{num_cycles} 个演化周期")

                if completed_cycles >= num_cycles:
                    self.logger.info("所有演化周期已完成，准备停止任务...")
                    break

                # 超时保护
                if len(self._evolution_data["ai_data_list"]) >= num_cycles + 2:
                    self.logger.warning("数据导出数量超出预期，强制停止...")
                    break

            # ---- 演化成功完成，构建最终合并波形 ----
            final_waveform = self._build_final_waveform()

            # 保存结果（如果提供了 result_folder）
            if result_folder is not None:
                try:
                    result_folder_path = Path(result_folder)
                    result_folder_path.mkdir(parents=True, exist_ok=True)

                    # 保存最终合并波形
                    waveform_save_path = result_folder_path / "evolved_waveform.pkl"
                    save_compressed_data(
                        final_waveform,
                        waveform_save_path,
                        data_type_name="演化最终波形",
                    )

                    # 绘制并保存演化轨迹图
                    plot_save_path = result_folder_path / "evolution.png"
                    self.plot_evolution(save_path=plot_save_path)

                    self.logger.info(f"结果已保存到: {result_folder_path}")
                except Exception as e:
                    self.logger.error(f"保存结果失败: {e}", exc_info=True)

        finally:
            # 确保任务始终被停止
            try:
                self._measure_controller.enable_export = False
                self._measure_controller.stop()
                self.logger.info("SingleChasCSIO 任务已停止")
            except Exception as e:
                self.logger.error(f"停止任务时出错: {e}", exc_info=True)

        self.logger.info("=" * 60)
        self.logger.info("反馈演化完成")
        self.logger.info(f"共完成 {len(self._ai_complex_amps_history)} 个演化周期")
        self.logger.info("=" * 60)

        return final_waveform

    # =========================================================================
    #  绘图方法：plot_evolution
    # =========================================================================

    def plot_evolution(
        self,
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> None:
        """
        在复平面上绘制各 AI 通道声场复振幅的演化轨迹

        将 `evolve` 方法执行后记录的每个周期的"总声场复振幅"绘制在同一复平面上。
        每个 AI 通道都有一条折线轨迹，起点（方形标记）和终点（星形标记）被特别标注。

        Args:
            save_path: 图像保存路径（包含扩展名，如 "output/evolution.png"）。
                如果为 None，则不保存文件。
            show: 是否调用 plt.show() 显示图像，默认 False。

        Raises:
            ValueError: 若没有演化数据（尚未执行 evolve 方法）。

        Examples:
            ```python
            evolver.evolve(num_cycles=10)
            evolver.plot_evolution(save_path="output/evolution.png")
            ```
        """
        if not self._ai_complex_amps_history:
            raise ValueError("没有可绘制的演化数据，请先调用 evolve 方法执行演化。")

        num_cycles = len(self._ai_complex_amps_history)
        n_ai = len(self._ai_channels)

        # 将历史记录转换为 (num_cycles, n_ai) 的 ndarray
        ai_array = np.array(self._ai_complex_amps_history)  # (num_cycles, n_ai)

        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # 颜色映射（最多支持 20 通道）
        colors = plt.cm.tab20(np.linspace(0, 1, max(n_ai, 2)))

        for i in range(n_ai):
            trajectory = ai_array[:, i]  # (num_cycles,)
            real_parts = trajectory.real
            imag_parts = trajectory.imag

            channel_label = (
                self._ai_channels[i] if i < len(self._ai_channels) else f"AI通道{i + 1}"
            )
            color = colors[i % len(colors)]

            # 绘制连接折线
            ax.plot(
                real_parts,
                imag_parts,
                color=color,
                linewidth=2,
                label=channel_label,
                alpha=0.7,
                zorder=1,
            )

            # 绘制中间点（小圆点）
            if num_cycles > 2:
                ax.scatter(
                    real_parts[1:-1],
                    imag_parts[1:-1],
                    color=color,
                    s=30,
                    alpha=0.5,
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=2,
                )

            # 起点（方形标记）
            ax.scatter(
                real_parts[0],
                imag_parts[0],
                marker="s",
                s=150,
                color=color,
                edgecolors="black",
                linewidths=2,
                alpha=0.9,
                zorder=3,
            )

            # 终点（星形标记）
            ax.scatter(
                real_parts[-1],
                imag_parts[-1],
                marker="*",
                s=300,
                color=color,
                edgecolors="black",
                linewidths=2,
                alpha=0.9,
                zorder=3,
            )

            # 仅对第一条线标注"起点"/"终点"文字，避免过多文字重叠
            if i == 0:
                ax.annotate(
                    "起点",
                    xy=(real_parts[0], imag_parts[0]),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=9,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7
                    ),
                )
                ax.annotate(
                    "终点",
                    xy=(real_parts[-1], imag_parts[-1]),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=9,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7
                    ),
                )

        # 原点参考线和标记
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.plot(
            0,
            0,
            marker="x",
            markersize=10,
            color="red",
            markeredgewidth=2,
            label="原点",
        )

        # 坐标轴和标题
        ax.set_xlabel("实部", fontsize=12)
        ax.set_ylabel("虚部", fontsize=12)
        ax.set_title(
            f"反馈演化 - AI 声场复振幅在复平面上的演化轨迹\n"
            f"（演化周期数: {num_cycles}，"
            f"频率: {self._static_sine_args['frequency']:.1f} Hz）",
            fontsize=14,
            fontweight="bold",
        )

        # 图例和网格
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()

        # 保存图像
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                self.logger.info(f"演化轨迹图已保存到: {save_path}")
            except Exception as e:
                self.logger.error(f"保存图像失败: {e}", exc_info=True)

        if show:
            plt.show()

        plt.close(fig)

    # =========================================================================
    #  资源清理
    # =========================================================================

    def cleanup(self) -> None:
        """
        清理资源，停止所有内部任务和线程。

        建议在不再使用 Evolver 时显式调用此方法。
        """
        if not hasattr(self, "_measure_controller"):
            return
        try:
            self._measure_controller.enable_export = False
            self._measure_controller.stop()
            self.logger.info("Evolver 资源清理完成")
        except Exception as e:
            self.logger.error(f"清理资源时出错: {e}", exc_info=True)

    def __del__(self) -> None:
        """析构函数，确保资源被正确释放"""
        self.cleanup()


def load_evolved_waveform(
    file_path: str | Path,
    segments: PositiveInt | None = None,
) -> Waveform:
    """
    从文件加载演化最终波形

    加载由 `Evolver.evolve()` 的 `result_folder` 参数保存的
    `evolved_waveform.pkl` 文件，返回内存中的 Waveform 对象。

    该波形可直接作为 `SweeperCore` 的 `static_output_waveform` 参数使用，
    无需任何反馈逻辑即可重放出与 evolve 最终状态一致的声场。

    Args:
        file_path: 波形文件路径（通常为由 `Evolver.evolve()` 生成的
            `evolved_waveform.pkl` 文件）。
        segments: 分段平均的段数，用于压缩波形时长。Evolver 保存的波形通常
            较长（例如 0.5s 以上），若直接用于 Sweeper 扫场会显著增加每点耗时。
            如果提供此参数，将对加载的波形调用 `average_single_waveform` 进行
            分段平均，使波形时长缩短为原来的 1/segments（采样点数同比例减少）。
            要求波形的采样点数能被 `segments` 整除。
            例如 `segments=10` 可将 0.5s 波形压缩为 0.05s。
            默认为 None（不压缩，保持原始时长）。

    Returns:
        多通道 Waveform 对象，包含 static 和 feedback 通道的合并信号。
        如果提供了 `segments`，返回的波形时长为原始波形的 1/segments。

    Raises:
        FileNotFoundError: 当文件不存在时。
        IOError: 当文件读取失败时。
        ValueError: 当加载的数据不是有效的 Waveform 时；
                    或当 `segments` 不能整除波形采样点数时。

    Examples:
        ```python
        from sweeper400.use.evolver import load_evolved_waveform
        from sweeper400.use import SweeperCore

        # 加载演化波形（保持原始时长）
        evolved_wf = load_evolved_waveform("output/evolution_result/evolved_waveform.pkl")

        # 加载并压缩为原始时长的 1/10（适合扫场，减少每点耗时）
        evolved_wf_fast = load_evolved_waveform(
            "output/evolution_result/evolved_waveform.pkl",
            segments=10
        )

        # 直接用于 SweeperCore 进行扫场测量（无需反馈）
        sweeper = SweeperCore(
            ai_channels=("PXI1Slot2/ai0", ...),
            ao_channels_static=evolved_wf_fast.channel_names,
            static_output_waveform=evolved_wf_fast,
            point_list=grid,
        )
        sweeper.sweep(result_folder="output/sweep_result")
        ```
    """
    # 获取函数日志器
    f_logger = get_logger(f"{__name__}.load_evolved_waveform")

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"演化波形文件不存在: {file_path}")

    loaded_data = load_compressed_data(file_path, data_type_name="演化波形")

    if not isinstance(loaded_data, Waveform):
        raise ValueError(
            f"加载的数据类型不正确，期望 Waveform，"
            f"实际为 {type(loaded_data).__name__}"
        )

    f_logger.info(
        f"演化波形加载成功: {loaded_data.shape}, "
        f"channels={loaded_data.channel_names}, "
        f"sine_args={loaded_data.sine_args}"
    )

    # ---- 可选：应用分段平均压缩时长 ----
    if segments is not None:
        try:
            loaded_data = average_single_waveform(loaded_data, segments=segments)
            f_logger.info(
                f"已应用分段平均压缩: segments={segments}, "
                f"压缩后形状={loaded_data.shape}, "
                f"压缩后时长={loaded_data.duration:.4f}s"
            )
        except Exception as e:
            f_logger.error(f"分段平均压缩失败: {e}", exc_info=True)
            raise

    return loaded_data
