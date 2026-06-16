"""
# 反馈演化器模块

模块路径：`sweeper400.use.evolver`

该模块包含 Evolver 类，提供基于反馈的声场演化控制功能。
"""

import threading
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import butter

from ..analyze import (
    Point2D,
    PointSweepData,
    PositiveFloat,
    PositiveInt,
    SweepData,
    TFData,
    Waveform,
    average_single_waveform,
    extract_single_tone_information_vvi,
    filter_waveform,
    get_sine,
    load_compressed_data,
    load_data_with_fallback,
    pick_waveform_channels,
    save_compressed_data,
)
from ..logger import get_logger
from ..measure import SingleChasCSIO

# 获取模块日志器
logger = get_logger(__name__)


class Evolver:
    """
    # 反馈演化器

    基于 SingleChasCSIO 的离散式反馈演化控制器。
    与扫场不同，Evolver 不控制步进电机进行空间移动，而是专注于通过多通道反馈控制，
    让各 AI 通道的总声场按照从 SimScanner 扫描结果中选取的"增益系数"演化收敛至理论稳态。

    ## 工作原理

    增益系数不再由用户手动指定，而是由 ``simulate()`` 方法从 SimScanner 生成的
    参数扫描结果（``scan_result.npz``）中，根据 (cr, ci) 和 mode 自动选取。
    演化器内部将 static + feedback 通道合并成一个统一的稳态输出，并以
    "稳态测量 → 离线计算 → 整体更换 → 再次等待稳态"的循环执行：

    1. 对最新一段 AI 数据进行去趋势 + 带通滤波；
    2. 用 ``extract_single_tone_information_vvi`` 估计 AI 各通道的"总声场复振幅"；
    3. 利用预先存储的传递矩阵对角元，从总声场中扣除"自身反馈通道贡献"，
       得到"入射声场复振幅"；
    4. 依据增益系数计算目标总声场，进而得到"全步长"反馈 AO 更新量 Δa；
       再按**自适应松弛因子 α*** 缩放（``a_new = a_old + α* · Δa``），避免
       谱半径过大时的发散；
    5. 幅值安全检查（默认 0.5 V 上限）；
    6. 整体重建 static + feedback 合并波形并通过
       ``SingleChasCSIO.update_static_output_waveform`` 更换；
    7. 阻塞等待 ``settle_time``，确保新稳态形成后再开始下一轮。

    ## 收敛性与自适应松弛因子

    一旦 TFData 与 ``gain_coefficients`` 给定，反馈迭代算子
    ``A = (I - G)(I - D^{-1} M^T)`` 就完全确定（M=tf_feedback, D=diag(tf_diag),
    G=diag(gain)）。``simulate()`` 阶段会：

    - 计算 A 的全部特征值和谱半径 ρ(A) → 存为 ``_iteration_eigenvalues``
      与 ``_spectral_radius_unrelaxed``；
    - 在 α∈(0, 1] 上扫描，挑选使 ``ρ((1-α)I + α A)`` 最小的 α*，并将其
      存为 ``_relaxation_factor``，对应谱半径存为 ``_spectral_radius_relaxed``；
    - 通过日志直观告知用户系统能否收敛、衰减率多少。

    后续 ``_feedback_method`` 与 ``evolve()`` 都会自动使用 α*。

    ## 使用方式

    ```python
    import numpy as np
    from sweeper400.analyze import init_sampling_info, get_sine
    from sweeper400.use import Evolver, load_evolved_waveform

    sampling_info = init_sampling_info(171500.0, 686000)  # 4s/段
    cca = np.array([0.05 + 0j])
    static_wf = get_sine(
        sampling_info, 3430.0, ("PXI1Slot2/ao0",), cca, full_cycle=True
    )

    evolver = Evolver(
        ai_channels=("PXI1Slot3/ai0", ...),                # 8 个
        ao_channels_static=("PXI1Slot2/ao0",),
        ao_channels_feedback=("PXI1Slot3/ao0", ...),       # 8 个
        static_output_waveform=static_wf,
        # sim_result_scan_path=...                              # 可选，默认走 storage/sim/sim_result_scan
    )

    # 先调用 simulate 选取增益系数并计算理论解
    evolver.simulate(
        cr=1.006, ci=-0.073,
        mode="eight_probes",     # 或 "floquet_probes"
        pick_max=False,          # 或 True
        result_folder="output/sim_result",
    )

    # 然后调用 evolve 执行硬件演化
    final_wf = evolver.evolve(
        cycles_num=10,
        result_folder="output/evolution_result",
    )
    ```

    ## 注意事项

    - ``static_output_waveform`` 时长建议较长（>= 0.5 s），以便 update_static_output_waveform
      生效后能尽快进入稳态。
    - ``evolve()`` 使用的增益系数来自最后一次 ``simulate()`` 调用所选取的结果。
    - 演化过程中若任意通道的 AO 幅值超过 ``ao_amplitude_limit``，将立即终止。
    - 演化数据储存在 ``_evolution_data``（``SweepData`` 格式）中，x 坐标为周期序号（从 1 开始）。
    - ``evolve()`` 返回的最终合并 Waveform 可直接作为 ``SweeperCore`` 的
      ``static_output_waveform`` 使用，从而无需反馈逻辑即可重放出最终稳态。
    """

    # 类日志器（类属性，所有实例共享）
    logger = get_logger(f"{__name__}.Evolver")

    # =========================================================================
    #  类常量
    # =========================================================================

    #: 解析迭代仿真收敛容差（相对误差，0.01%）
    _ITERATION_TOLERANCE: float = 0.0001

    #: 解析迭代仿真最大迭代次数（超过此值仍不收敛则视为发散）
    _MAX_ITERATION_CYCLES: int = 10000

    # =========================================================================
    #  初始化
    # =========================================================================

    def __init__(
        self,
        ai_channels: tuple[str, ...],
        ao_channels_static: tuple[str, ...],
        ao_channels_feedback: tuple[str, ...],
        static_output_waveform: Waveform,
        fishnet_tf_data_path: str | Path | None = None,
        sim_result_scan_path: str | Path | None = None,
        settle_time: PositiveFloat | None = None,
        convergence_threshold: PositiveFloat = 0.95,
    ) -> None:
        """
        初始化反馈演化器。

        Args:
            ai_channels: AI 通道名称元组。数量应与 `ao_channels_feedback` 相同。
            ao_channels_static: 静态主激励 AO 通道名称元组（不参与反馈调整）。
            ao_channels_feedback: 反馈 AO 通道名称元组，演化过程中其复振幅
                会被实时更新。
            static_output_waveform: 静态主激励输出波形。其 `frequency` 与
                `channel_complex_amplitudes` 将作为系统初始激励参数被反复使用。
            fishnet_tf_data_path: Fishnet 传递函数数据文件路径（可选）。
                直接转交给内部 `SingleChasCSIO`；同时 Evolver 自身也会
                通过 `load_data_with_fallback` 读取该文件，并将
                `tf_dataframe` 数据部分预先转为复数 ndarray 储存为属性，
                供反馈数据处理与理论解求解使用。
            sim_result_scan_path: SimScanner 参数扫描结果目录路径（可选）。
                默认从 `storage/sim/sim_result_scan` 读取。该目录中应包含
                `scan_result.npz` 文件，由 `SimScanner.run_scan()` 生成。
                `simulate()` 方法将从该文件中加载增益系数。
            settle_time: 每次 `update_static_output_waveform` 之后的
                等待时间（秒）。默认值为 `2 × static_output_waveform.duration + 0.1`，
                保证旧波形完全退出缓冲区、新波形在再生模式下完全填充。
            convergence_threshold: 收敛性安全阈值，默认 0.95。`simulate()` 阶段
                求出最优松弛因子 α* 及对应的迭代算子谱半径 ρ(A_α*)；若
                `ρ(A_α*) > convergence_threshold`，即"理论上虽收敛但每周期
                衰减率不足 (1 - threshold)"，将输出警告。

        Raises:
            ValueError: 当通道数不一致，或缺少必要的复振幅/频率属性时。
            RuntimeError: 当 fishnet_tf_data 加载失败、SingleChasCSIO 初始化失败、
                tf_dataframe 中缺少必要的 AO/AI 通道时。
        """
        # ---- 参数验证 ----
        n_fb = len(ao_channels_feedback)
        n_ai = len(ai_channels)

        if n_fb != n_ai:
            raise ValueError(
                f"ao_channels_feedback 长度 ({n_fb}) 必须与 "
                f"ai_channels 长度 ({n_ai}) 相同"
            )
        if static_output_waveform.frequency is None:
            raise ValueError(
                "static_output_waveform 必须设置 frequency 属性，"
                "请使用 get_sine 函数生成。"
            )
        if static_output_waveform.channel_complex_amplitudes is None:
            raise ValueError(
                "static_output_waveform 必须设置 channel_complex_amplitudes 属性，"
                "请使用 get_sine 函数生成。"
            )

        # ---- 保存通道与基本配置 ----
        self._ai_channels: tuple[str, ...] = ai_channels
        self._ao_channels_static: tuple[str, ...] = ao_channels_static
        self._ao_channels_feedback: tuple[str, ...] = ao_channels_feedback
        self._ao_channels_combined: tuple[str, ...] = (
            ao_channels_static + ao_channels_feedback
        )

        self._frequency: PositiveFloat = static_output_waveform.frequency
        self._user_static_output_waveform: Waveform = static_output_waveform

        # ---- 静态 AO 通道复振幅向量（处理单通道扩展） ----
        # static_output_waveform 可能是单通道，但用户的 ao_channels_static 可能多于 1。
        # 若是单通道扩展，则把单一复振幅复制到所有静态通道。
        self._static_ao_complex_amps: np.ndarray = self._expand_static_complex_amps(
            static_output_waveform, ao_channels_static
        )

        # ---- 加载 fishnet_tf_data_path ----
        # 与 SingleChasCSIO 行为保持一致：使用 load_data_with_fallback
        loaded_tf_data = load_data_with_fallback(
            explicit_path=fishnet_tf_data_path,
            default_path=Path("storage/calib/calib_result_fishnet/tf_data.pkl"),
            data_type="Fishnet传递函数数据",
        )
        if loaded_tf_data is None:
            raise RuntimeError(
                "Evolver 需要 fishnet_tf_data_path，但既未提供显式路径，也未在默认路径找到。"
                "请先完成 fishnet 校准，或显式指定 fishnet_tf_data_path 路径。"
            )
        self._fishnet_tf_data: TFData = loaded_tf_data  # noqa
        # 用户提供的原始路径，传给 CSIO 时直接使用
        self._fishnet_tf_data_path: str | Path | None = fishnet_tf_data_path

        # ---- 预处理传递矩阵 ----
        # 将 tf_dataframe 的数据部分提前转为 complex ndarray，并切出常用子矩阵
        (
            self._tf_full_matrix,
            self._tf_static_to_ai,
            self._tf_feedback_to_ai,
            self._tf_diag,
        ) = self._build_tf_matrices(self._fishnet_tf_data["tf_dataframe"])

        # ---- SimScanner 扫描结果路径 ----
        if sim_result_scan_path is None:
            self._sim_result_scan_path: Path = (
                Path("storage/sim/sim_result_scan")
            )
        else:
            self._sim_result_scan_path = Path(sim_result_scan_path)

        # ---- 增益系数与收敛分析（延迟到 simulate 阶段） ----
        # 增益系数由 simulate() 从扫描结果中选取后设置。
        self._gain_coefficients: np.ndarray | None = None
        # 松弛因子 α* 和谱半径也在 simulate() 中通过 _analyze_convergence 计算。
        self._convergence_threshold: float = float(convergence_threshold)
        self._relaxation_factor: float | None = None
        # 缓存已加载的扫描结果数据（npz 内容），避免重复读取磁盘。
        self._scan_result_data: np.lib.npyio.NpzFile | None = None
        # 最近一次 simulate 选取的增益系数（供 plot_gain_coefficients 使用）
        self._picked_gain_8: np.ndarray | None = None
        self._picked_floquet_gains_3: np.ndarray | None = None
        self._picked_cr: float | None = None
        self._picked_ci: float | None = None

        # ---- 反馈滤波器（按频率构建一次） ----
        self._sos = butter(
            N=4,
            Wn=[self._frequency * 0.5, self._frequency * 2.0],
            btype="bandpass",
            analog=False,
            output="sos",
            fs=static_output_waveform.sampling_rate,
        )

        # ---- 等待时间默认值 ----
        if settle_time is None:
            self._wait_time_after_update: PositiveFloat = float(
                2.0 * static_output_waveform.duration + 0.1
            )
        else:
            self._wait_time_after_update = float(settle_time)

        # ---- 初始化理论稳态解 ----
        self._theoretical_feedback_ao_complex_amps: np.ndarray | None = None
        self._theoretical_total_ai_complex_amps: np.ndarray | None = None

        # ---- 反馈状态（演化期间动态更新） ----
        self._current_ao_complex_amps: np.ndarray = np.zeros(n_fb, dtype=np.complex128)
        # 实测/模拟过程中各周期的轨迹：
        # - `_ai_complex_amps_history[k]` 为第 k 周期估计/模拟的 AI 总声场复振幅；
        # - `_ao_complex_amps_history[k]` 为第 k 周期由反馈律算出的"新"反馈 AO 复振幅。
        self._ai_complex_amps_history: list[np.ndarray] = []
        self._ao_complex_amps_history: list[np.ndarray] = []

        # 评估时使用的最新一段 AI 波形（由导出回调写入主线程取用）
        self._latest_ai_waveform: Waveform | None = None
        self._latest_ai_chunks_num: int = 0
        self._ai_data_lock = threading.Lock()
        self._ai_data_event = threading.Event()

        # SweepData 格式的演化数据
        self._evolution_data: SweepData = {
            "ai_data_list": [],
            "ao_data": static_output_waveform,
        }

        # ---- 构建初始合并波形（feedback 部分为 0） ----
        # 同时缓存为属性，以便后续 evolve() 在多次启动 CSIO 之间重置稳态输出。
        initial_combined = self._build_combined_waveform(self._current_ao_complex_amps)
        self._initial_combined_waveform: Waveform = initial_combined

        # ---- 创建 SingleChasCSIO ----
        # 关键：将所有 AO 通道作为 static 通道交给 CSIO，
        # CSIO 的 feedback 通道留空，feedback_function 也留空（不会被触发）。
        try:
            self.logger.debug("正在初始化 SingleChasCSIO（仅静态模式）...")
            self._measure_controller = SingleChasCSIO(
                ai_channels=ai_channels,
                ao_channels_static=self._ao_channels_combined,
                ao_channels_feedback=(),
                static_output_waveform=initial_combined,
                export_function=self._data_export_callback,
                feedback_function=None,
                fishnet_tf_data=fishnet_tf_data_path,
            )
            self.logger.debug("SingleChasCSIO 初始化成功")
        except Exception as e:
            error_msg = f"SingleChasCSIO 初始化失败: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

        # 演化运行期间的辅助状态
        self._target_num_cycles: PositiveInt = 1
        self._ao_amplitude_limit: PositiveFloat = 1.0
        self._stop_flag: bool = False
        self._evolve_error: Exception | None = None

        self.logger.info(
            f"Evolver 初始化完成 - "
            f"AI 通道: {ai_channels}, "
            f"Static AO 通道: {ao_channels_static}, "
            f"Feedback AO 通道: {ao_channels_feedback}, "
            f"频率: {self._frequency:.2f} Hz, "
            f"扫描结果路径: {self._sim_result_scan_path}, "
            f"稳态等待: {self._wait_time_after_update:.3f} s"
        )

    # =========================================================================
    #  初始化辅助函数
    # =========================================================================

    @staticmethod
    def _expand_static_complex_amps(
        static_waveform: Waveform,
        ao_channels_static: tuple[str, ...],
    ) -> np.ndarray:
        """
        将 `static_waveform` 的复振幅扩展/对齐到 `ao_channels_static` 长度。

        - 单通道波形 + 多通道目标：把单一复振幅复制到所有目标通道；
        - 通道数一致：直接返回；
        - 其它：抛错。
        """
        cca = np.asarray(
            static_waveform.channel_complex_amplitudes, dtype=np.complex128
        )
        n_target = len(ao_channels_static)
        if static_waveform.is_single_channel and n_target > 1:
            return np.full(n_target, cca[0], dtype=np.complex128)
        if static_waveform.channels_num == n_target:
            return cca
        raise ValueError(
            f"static_output_waveform 通道数 ({static_waveform.channels_num}) "
            f"与 ao_channels_static 长度 ({n_target}) 不匹配，且不是单通道扩展场景"
        )

    def _build_tf_matrices(
        self, tf_dataframe: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        从 `tf_dataframe` 中提取所需的传递矩阵（complex ndarray）。

        Args:
            tf_dataframe: TFData 中的 DataFrame，index 为 AO 通道名，columns 为 AI 通道名。

        Returns:
            tf_full: 整个 dataframe 的复数 ndarray（按 dataframe 原顺序）。
            tf_static_to_ai: 形状 (n_static, n_ai)，索引顺序与 ao_channels_static、
                ai_channels 一一对应。
            tf_feedback_to_ai: 形状 (n_feedback, n_ai)，索引顺序与 ao_channels_feedback、
                ai_channels 一一对应。
            tf_diag: 形状 (n_feedback,) 的"传递矩阵对角元向量"，
                tf_diag[i] = TF(ao_feedback[i] -> ai_channels[i])。
        """
        # ---- 完整矩阵（保持 dataframe 顺序，方便调试与扩展使用） ----
        try:
            tf_full = np.asarray(tf_dataframe.values, dtype=np.complex128)
        except Exception as e:
            raise RuntimeError(
                f"无法将 fishnet_tf_data_path['tf_dataframe'] 转为复数 ndarray: {e}"
            ) from e

        # ---- 校验所需通道是否齐全 ----
        ao_index = list(tf_dataframe.index)
        ai_columns = list(tf_dataframe.columns)
        missing_static = [c for c in self._ao_channels_static if c not in ao_index]
        missing_feedback = [c for c in self._ao_channels_feedback if c not in ao_index]
        missing_ai = [c for c in self._ai_channels if c not in ai_columns]
        if missing_static or missing_feedback or missing_ai:
            raise RuntimeError(
                "fishnet_tf_data_path 中缺少必要的通道：\n"
                f"  缺失静态 AO: {missing_static}\n"
                f"  缺失反馈 AO: {missing_feedback}\n"
                f"  缺失 AI: {missing_ai}\n"
                "请检查 fishnet 校准是否覆盖了所有需要的通道。"
            )

        # ---- 抽取静态 AO -> AI 的子矩阵，shape = (n_static, n_ai) ----
        static_block = tf_dataframe.loc[
            list(self._ao_channels_static), list(self._ai_channels)
        ].values
        tf_static_to_ai = np.asarray(static_block, dtype=np.complex128)

        # ---- 抽取反馈 AO -> AI 的子矩阵，shape = (n_feedback, n_ai) ----
        feedback_block = tf_dataframe.loc[
            list(self._ao_channels_feedback), list(self._ai_channels)
        ].values
        tf_feedback_to_ai = np.asarray(feedback_block, dtype=np.complex128)

        # ---- 对角元向量（成对配对：第 i 个反馈 AO 与第 i 个 AI） ----
        # tf_diag[i] = TF(ao_feedback[i] -> ai_channels[i]) = tf_feedback_to_ai[i, i]
        tf_diag = np.diag(tf_feedback_to_ai).astype(np.complex128).copy()

        self.logger.debug(
            f"传递矩阵已预处理: tf_full.shape={tf_full.shape}, "
            f"tf_static_to_ai.shape={tf_static_to_ai.shape}, "
            f"tf_feedback_to_ai.shape={tf_feedback_to_ai.shape}, "
            f"tf_diag.shape={tf_diag.shape}"
        )

        return tf_full, tf_static_to_ai, tf_feedback_to_ai, tf_diag

    def _analyze_convergence(self) -> tuple[np.ndarray, float, float, float]:
        """
        ## 反馈迭代收敛性分析与自适应松弛因子选取

        ### 背景

        `_feedback_method` 执行的是不动点迭代。**完整代入式**更新（α=1）下，
        把 analytical 模式的反馈律按矩阵形式展开后可得：

            a_new = A · a_old + b
            其中  A = (I - G)(I - D^{-1} M^T),
                  b = (G - I) D^{-1} S
            G = diag(gain_coefficients), D = diag(tf_diag), M = tf_feedback_to_ai。

        引入"松弛因子" α∈(0, 1]——即每周期只前进 α 倍的"全步长 Δa"——后，
        递推变为：

            a_new = ((1 - α)I + α A) · a_old + α b

        新迭代算子 A_α = (1-α)I + α A 的特征值满足：

            μ_i(α) = 1 - α (1 - λ_i)         (λ_i 为 A 的特征值)

        几何上，每个 λ_i 沿"由 1 指向 λ_i"的射线，按 α 比例从端点 1 滑向 λ_i。

        ### 收敛性判据

        当且仅当 ρ(A_α) = max_i |μ_i(α)| < 1 时，迭代必收敛，且每周期误差按
        ρ(A_α)^k 衰减。

        ### 自适应 α* 选取

        在 α∈(0, 1] 上做一维网格扫描，挑选使 ρ(A_α) 最小的 α*。
        - 若 α=1 已收敛（即 ρ(A) < 1），扫描通常就会回到 α≈1 附近；
        - 若 α=1 发散，扫描会自动找到合适的"减速因子"，往往能把
          ρ(A_α) 压回 1 以下；
        - 若所有 α∈(0, 1] 都给不出 ρ < 1，说明系统在该 TF + 增益组合下
          物理上无法用单步松弛迭代稳定（受迭代算子某些"贴近 1"的特征值
          所限——这是系统固有性质，与 α 选取无关）。

        ### 收敛速度安全闸

        即使 ρ(A_α*) < 1（理论收敛），过于贴近 1 的谱半径会带来漫长的
        振荡过渡过程（误差衰减率 ≈ ρ^k）。因此本方法在求出 α* 后会
        与 `self._convergence_threshold`（默认 0.9）比较：
        - `ρ(A_α*) ≤ threshold`：通过，记录日志后正常返回；
        - `ρ(A_α*) > threshold`：抛 `RuntimeError`，提示用户调整
          `gain_coefficients`（让 |g_i - 1| 更小）或更换 TF 数据。
          这是一个"安全闸"——而非"求解器"，因为受系统固有谱所限，
          不存在能强行把 ρ 压到任意值以下的 α。

        Returns:
            eigenvalues: 迭代算子 A 的特征值，shape (n_fb,)。
            spectral_radius_unrelaxed: α=1 时的谱半径，等于 max|λ_i|。
            relaxation_factor: 自适应选取的松弛因子 α*。
            spectral_radius_relaxed: 使用 α* 后迭代算子的谱半径
                ρ(A_{α*})；< 1 表示迭代将以该比例衰减收敛。

        Raises:
            RuntimeError: 当 ρ(A_α*) > self._convergence_threshold 时。
        """
        n = len(self._tf_diag)
        gain_diag = np.diag(self._gain_coefficients)
        # (D^{-1} M^T)_{ij} = M^T_{ij}/d_i = M_{ji}/d_i —— 即 M^T 按行除以 d_i。
        # 注：该矩阵的对角元恒为 1（M_{ii}/d_i = d_i/d_i），
        #     故 (I - D^{-1} M^T) 的对角元恒为 0，迭代算子完全由 TF 串扰驱动。
        d_inv_mt = self._tf_feedback_to_ai.T / self._tf_diag[:, None]
        iteration_operator = (np.eye(n) - gain_diag) @ (np.eye(n) - d_inv_mt)

        eigenvalues = np.linalg.eigvals(iteration_operator).astype(np.complex128)
        rho_unrelaxed = float(np.max(np.abs(eigenvalues)))

        # ---- 网格扫描最优松弛因子 α* ∈ (0, 1] ----
        # 1001 个采样点足够稠密（步长 ~1e-3），开销可忽略。
        alpha_grid = np.linspace(1e-3, 1.0, 1001)
        # rho_curve[k] = max_i |1 - α_k (1 - λ_i)|；用广播一次性算完。
        mu_matrix = 1.0 - alpha_grid[:, None] * (1.0 - eigenvalues[None, :])
        rho_curve = np.max(np.abs(mu_matrix), axis=1)
        best_idx = int(np.argmin(rho_curve))
        best_alpha = float(alpha_grid[best_idx])
        rho_relaxed = float(rho_curve[best_idx])

        # ---- 日志输出 ----
        threshold = self._convergence_threshold
        self.logger.info("=" * 60)
        self.logger.info("反馈迭代收敛性分析（基于谱半径）")
        self.logger.info(
            f"  迭代算子 A = (I-G)(I - D^-1 M^T) 的谱半径 ρ(A) = {rho_unrelaxed:.4f}"
        )
        if rho_unrelaxed < 1.0:
            self.logger.info(
                f"  α=1（无松弛）下迭代必收敛，每周期误差衰减率 ≈ {rho_unrelaxed:.4f}"
            )
        else:
            self.logger.warning(
                f"  α=1（无松弛）下迭代会发散，每周期误差放大率 ≈ {rho_unrelaxed:.4f}！"
                "—— 已自动启用松弛因子。"
            )

        self.logger.info(
            f"  自适应松弛因子 α* = {best_alpha:.4f}（在 (0, 1] 上网格扫描得到）"
        )
        if rho_relaxed < 1.0:
            self.logger.info(
                f"  应用 α* 后迭代算子谱半径 ρ(A_α*) = {rho_relaxed:.4f}（< 1，必收敛）"
            )
        else:
            self.logger.warning(
                f"  即使引入松弛仍无法将谱半径降到 1 以下："
                f"min ρ = {rho_relaxed:.4f}（≥ 1）。"
            )
        self.logger.info("=" * 60)

        # ---- 收敛速度安全闸 ----
        # 即使 ρ < 1 但贴近 1，迭代过渡会非常缓慢；超过阈值即抛错，
        # 让用户在硬件演化前就能察觉并调整参数。
        if rho_relaxed > threshold:
            wrn_msg = (
                f"反馈迭代收敛过慢：自适应松弛后谱半径 ρ(A_α*) = {rho_relaxed:.4f} "
                f"超过 convergence_threshold = {threshold:.4f}（每周期误差仅按 "
                f"{rho_relaxed:.3f}^k 衰减，过渡振荡会非常持久）。\n"
                "可能原因与建议：\n"
                "  1) gain_coefficients 中 |g_i - 1| 过大 —— 让增益系数更接近 1，"
                "降低反馈律的激进度；\n"
                "  2) TF 矩阵串扰过强（非对角元相对对角元过大）—— 检查 fishnet "
                "校准质量，或更换 TF 数据；\n"
                "  3) 若已确认无法改善但仍想强制运行，可在初始化时显式提高 "
                "convergence_threshold（如 0.95、1.0），但请理解这意味着"
                "可能需要数十个周期才能收敛。"
            )
            self.logger.warning(wrn_msg)

        return eigenvalues, rho_unrelaxed, best_alpha, rho_relaxed

    # =========================================================================
    #  参数扫描仿真结果加载与增益系数选取
    # =========================================================================

    def _load_scan_result(self) -> np.lib.npyio.NpzFile:
        """
        加载 SimScanner 的参数扫描结果 npz 文件。

        优先使用缓存（`_scan_result_data`），避免重复读取磁盘。

        Returns:
            npz 文件对象，包含 cr_values, ci_values, eight_gains, floquet_gains 等数组。

        Raises:
            FileNotFoundError: 当扫描结果文件不存在时。
        """
        if self._scan_result_data is not None:
            return self._scan_result_data

        npz_path = self._sim_result_scan_path / "scan_result.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"SimScanner 扫描结果文件不存在: {npz_path}\n"
                "请先运行 SimScanner.run_scan() 生成参数扫描结果，"
                "或在初始化时指定正确的 sim_result_scan_path。"
            )

        self._scan_result_data = np.load(npz_path, allow_pickle=False)
        self.logger.info(
            f"已加载 SimScanner 扫描结果: {npz_path}, "
            f"cr_values={self._scan_result_data['cr_values']}, "
            f"ci_values={self._scan_result_data['ci_values']}"
        )
        return self._scan_result_data

    def _pick_gains_from_scan(
        self,
        cr: float,
        ci: float,
        mode: Literal["eight_probes", "floquet_probes"] = "eight_probes",
        pick_max: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float, float]:
        """
        从扫描结果中选取一组 (cr, ci) 参数组合的增益系数。

        ## 选取逻辑

        - `pick_max=False`：选取距离输入 (cr, ci) 最近的参数组合。
        - `pick_max=True`：
            - `mode="eight_probes"`：选取"8 个增益系数模长的平均值"最大的一组。
            - `mode="floquet_probes"`：选取"基于传声器 (point1) 的增益系数模长"最大的一组。

        ## 返回值

        Returns:
            gain_8: 8 周期模式 8 个增益系数，shape (8,)。
            floquet_gains_3: Floquet 模式 3 个增益系数 [bnd1, bnd2, point1]，shape (3,)。
            gain_for_experiment: 用于实验的 8 个增益系数，shape (8,)。
                - mode="eight_probes" 时与 gain_8 相同；
                - mode="floquet_probes" 时将 point1 增益复制为 8 个。
            cr_idx: 选中的 cr 索引。
            ci_idx: 选中的 ci 索引。
            picked_cr: 选中的 cr 值。
            picked_ci: 选中的 ci 值。
        """
        data = self._load_scan_result()
        cr_values = data["cr_values"]
        ci_values = data["ci_values"]
        eight_gains = data["eight_gains"]       # (res, res, 8)
        floquet_gains = data["floquet_gains"]   # (res, res, 3)

        if pick_max:
            if mode == "eight_probes":
                # 选取"8 个增益系数模长的平均值"最大的一组
                metric = np.mean(np.abs(eight_gains), axis=2)  # (res, res)
            else:  # floquet_probes
                # 选取"基于传声器 (point1) 的增益系数模长"最大的一组
                metric = np.abs(floquet_gains[:, :, 2])  # (res, res)
            flat_idx = int(np.argmax(metric))
            cr_idx, ci_idx = np.unravel_index(flat_idx, metric.shape)
            self.logger.info(
                f"pick_max=True, mode={mode}: "
                f"选中 cr_idx={cr_idx}, ci_idx={ci_idx}, "
                f"cr={cr_values[cr_idx]:.6f}, ci={ci_values[ci_idx]:.6f}, "
                f"metric={metric[cr_idx, ci_idx]:.6f}"
            )
        else:
            # 选取距离输入 (cr, ci) 最近的参数组合
            cr_diffs = np.abs(cr_values - cr)
            ci_diffs = np.abs(ci_values - ci)
            cr_idx = int(np.argmin(cr_diffs))
            ci_idx = int(np.argmin(ci_diffs))
            self.logger.info(
                f"pick_max=False: 选取最近参数组合 "
                f"cr_idx={cr_idx} (cr={cr_values[cr_idx]:.6f}, 输入={cr:.6f}), "
                f"ci_idx={ci_idx} (ci={ci_values[ci_idx]:.6f}, 输入={ci:.6f})"
            )

        # 提取增益系数
        gain_8 = np.asarray(eight_gains[cr_idx, ci_idx, :], dtype=np.complex128)  # (8,)
        floquet_gains_3 = np.asarray(
            floquet_gains[cr_idx, ci_idx, :], dtype=np.complex128
        )  # (3,) [bnd1, bnd2, point1]

        # 构建用于实验的增益系数
        if mode == "eight_probes":
            gain_for_experiment = gain_8.copy()
        else:  # floquet_probes: 使用 point1 (idx=2) 的增益复制到全部 8 个通道
            gain_for_experiment = np.full(8, floquet_gains_3[2], dtype=np.complex128)

        picked_cr = float(cr_values[cr_idx])
        picked_ci = float(ci_values[ci_idx])

        self.logger.info(
            f"增益系数已选取 - "
            f"8周期模式: {gain_8}, "
            f"Floquet模式 [bnd1, bnd2, point1]: {floquet_gains_3}, "
            f"实验用增益: {gain_for_experiment}"
        )

        return (
            gain_8,
            floquet_gains_3,
            gain_for_experiment,
            cr_idx,
            ci_idx,
            picked_cr,
            picked_ci,
        )

    def _simulate_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """
        ## 物理建模（matrix 模式内核）

        系统由两个独立的物理关系联立而成：

        ① 叠加原理（物理定律）：
            T_i(总声场Total) = S_i(静态输入贡献Static) + sum_j(对j求和) tf_feedback[j, i] * amps_feedback[j]
            其中 S_i = sum_s tf_static[s, i] * amps_static[s]

        ② 增益约束（控制目标）：
            T_i = gf_i（增益系数） * (T_i - tf_feedback[i, i] * amps_feedback[i])

        从 ② 解出 amps_feedback[i]：
            amps_feedback[i] = β_i * T_i,    其中 β_i = (gf_i - 1) / (gf_i * tf_feedback[i, i])

        β_i 即"反馈增益因子"——给定 AI[i] 的总声场，反馈扬声器 i 应当输出的复振幅系数。

        将 amps_feedback[j] = β_j * T_j 代入 ①，得到关于 T 的闭环反馈方程：

            (I - tf_feedback^T @ diag(β)) @ T = S

        其中 tf_feedback (shape (n_fb, n_ai))。

        该方程具有清晰的控制论意义：
            - I:                   恒等映射；
            - tf_feedback^T @ diag(β):    一轮完整反馈回路（声场 → AO 输出 → 物理传播回声场）；
            - I - tf_feedback^T @ diag(β): 闭环算子，扣除反馈回路效应后的"净"映射；
            - S:                   仅静态喇叭驱动时各 AI 通道收到的声场（外部输入）。

        即：**总声场减去经过一轮反馈回路产生的声场，等于原始静态激励**。

        求解 T 之后，反馈 AO 复振幅由反馈律一步给出：
            amps_feedback[i] = β_i * T[i]

        Returns:
            theoretical_feedback_ao_complex_amps: shape (n_fb,)
            theoretical_total_ai_complex_amps: shape (n_ai,)

        Raises:
            ValueError: 当 gain_coefficients 中存在零值（β_i 发散）时。
            RuntimeError: 当闭环矩阵 (I - tf_feedback^T @ diag(β)) 奇异时。
        """
        n = len(self._ai_channels)
        gf = self._gain_coefficients          # (n,)
        amps_static = self._static_ao_complex_amps   # (n_static,)

        tf_static = self._tf_static_to_ai          # (n_static, n_ai)
        tf_feedback = self._tf_feedback_to_ai      # (n_feedback, n_ai)
        tf_diag = self._tf_diag                    # (n_feedback,)，即 tf_feedback[i, i]

        # ---- 静态激励在各 AI 通道的贡献 S[i] = sum_s amps_static[s] * tf_static[s, i] ----
        static_contribution = amps_static @ tf_static                       # (n_ai,)

        # ---- 反馈增益因子 β[i] = (gf[i] - 1) / (gf[i] * tf_diag[i]) ----
        # 物理意义：为满足增益约束，a_feedback[i] = β[i] * T[i]
        if np.any(gf == 0):
            raise ValueError(
                "gain_coefficients 中存在零值（要求总声场为零），"
                "无法使用闭环反馈方程求解（β_i 将发散）。"
            )
        beta = (gf - 1.0) / (gf * tf_diag)                            # (n,)

        # ---- 闭环反馈方程：(I - tf_feedback^T @ diag(β)) @ T = S ----
        # tf_feedback^T 形状 (n_ai, n_feedback)，其 [i, j] 元素 = tf_feedback[j, i]；
        # 右乘 diag(β) 等价于将 tf_feedback^T 的第 j 列乘以 β[j]，即按列广播。
        loop_gain_matrix = tf_feedback.T * beta                     # (n_ai, n_feedback) = (n, n)
        closed_loop_matrix = np.eye(n) - loop_gain_matrix           # (n, n)

        try:
            theoretical_total_ai = np.linalg.solve(
                closed_loop_matrix, static_contribution
            )
        except np.linalg.LinAlgError as e:
            self.logger.error(
                f"求解理论稳态解的闭环方程失败（闭环矩阵可能奇异）：{e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"求解理论稳态解失败：{e}。请检查 gain_coefficients 与传递矩阵。"
            ) from e

        # ---- 由反馈律求反馈 AO 复振幅：a_fb[i] = β[i] * T[i] ----
        theoretical_feedback = beta * theoretical_total_ai          # (n,)

        return (
            theoretical_feedback.astype(np.complex128),
            theoretical_total_ai.astype(np.complex128),
        )

    # =========================================================================
    #  理论解：simulate（公有）
    # =========================================================================

    def simulate(
        self,
        cr: float,
        ci: float,
        mode: Literal["eight_probes", "floquet_probes"] = "eight_probes",
        pick_max: bool = False,
        ao_amplitude_limit: PositiveFloat = 0.5,
        result_folder: str | Path | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        从扫描结果中选取增益系数，计算理论稳态解，并执行解析迭代仿真。

        ## 执行流程

        本方法分四个阶段执行：

        1. **增益系数选取阶段**：根据输入参数 (cr, ci) 和 mode，从
           SimScanner 生成的 ``scan_result.npz`` 中选取一组增益系数。
           选取后立即完成收敛性分析与自适应松弛因子计算，
           并将增益系数存入 ``_gain_coefficients``，松弛因子存入
           ``_relaxation_factor``。

        2. **矩阵求解阶段**：在已知 ``static_ao_complex_amps``、
           ``gain_coefficients`` 和传递矩阵的前提下，将"叠加原理 + 增益约束"
           联立为线性方程组 ``(I - tf_feedback^T @ diag(β)) @ T = S``，
           一次性解出稳态总声场 T 与反馈 AO 复振幅。该结果作为唯一的
           "理论稳态解"存入 ``_theoretical_feedback_ao_complex_amps`` 与
           ``_theoretical_total_ai_complex_amps``。

        3. **解析迭代阶段**：从 0 反馈出发，每周期调用
           ``_feedback_method(mode="analytical")``（内部用传递矩阵解析推算总声场，
           等价于"无噪声理想环境"），并将结果回灌到 ``_current_ao_complex_amps``。
           迭代将持续进行，直到连续两轮 8 个传声器复振幅的相对差小于
           收敛容差（1%），或者超过最大迭代次数（10000）仍未收敛（此时
           发出警告并认为系统发散）。

        4. **可选绘图阶段**：若提供 ``result_folder``，则自动绘制并保存
           AI 总声场轨迹、反馈 AO 轨迹以及增益系数极坐标复平面图。

        ## evolve 的前置条件

        调用一次成功的 ``simulate()`` 是后续 ``evolve()`` 运行的必要前提
        （evolve 需要拿到理论解和增益系数），否则 ``evolve()`` 会拒绝执行。
        ``evolve()`` 使用的增益系数来自最后一次 ``simulate()`` 调用。

        Args:
            cr: 管槽声速实部放缩因子，用于从扫描结果中选取增益系数。
            ci: 管槽声速虚部放缩因子，用于从扫描结果中选取增益系数。
            mode: 增益系数选取模式。
                - ``"eight_probes"``（默认）：使用 8 周期模式的 8 个增益系数。
                - ``"floquet_probes"``：使用 Floquet 模式基于传声器 (point1)
                  的增益系数，将其复制到全部 8 个通道。
            pick_max: 是否选取"最优"参数组合而非最近的。
                - ``False``（默认）：选取距离 (cr, ci) 最近的参数组合。
                - ``True``：当 mode="eight_probes" 时选取 8 个增益系数模长
                  平均值最大的一组；当 mode="floquet_probes" 时选取基于
                  传声器 (point1) 的增益系数模长最大的一组。
            ao_amplitude_limit: 反馈 AO 幅值安全上限（V），默认 0.5。
                与 ``evolve()`` 同名参数行为一致；仅在迭代阶段生效，超过即抛错。
            result_folder: 结果保存文件夹路径，可选。若提供则在末尾自动绘制
                并保存 AI/AO 轨迹图及增益系数图。

        Returns:
            (theoretical_feedback_ao_complex_amps, theoretical_total_ai_complex_amps)，
            即矩阵阶段求得的理论稳态解。

        Raises:
            FileNotFoundError: 当扫描结果文件不存在时。
            ValueError: 当 gain_coefficients 中存在零值时。
            RuntimeError: 当闭环矩阵奇异，或迭代阶段反馈 AO 超过安全上限时。
        """
        # ---- 阶段 0：清空旧状态 ----
        n_fb = len(self._ao_channels_feedback)
        self._ao_amplitude_limit = float(ao_amplitude_limit)
        self._current_ao_complex_amps = np.zeros(n_fb, dtype=np.complex128)
        self._ai_complex_amps_history.clear()
        self._ao_complex_amps_history.clear()
        self._theoretical_feedback_ao_complex_amps = None
        self._theoretical_total_ai_complex_amps = None

        # ---- 阶段 1：从扫描结果中选取增益系数 + 收敛性分析 ----
        self.logger.info(
            f"simulate 阶段 1/3: 选取增益系数 "
            f"(cr={cr}, ci={ci}, mode={mode}, pick_max={pick_max})..."
        )
        (
            gain_8,
            floquet_gains_3,
            gain_for_experiment,
            _cr_idx,
            _ci_idx,
            picked_cr,
            picked_ci,
        ) = self._pick_gains_from_scan(cr, ci, mode, pick_max)

        # 设置增益系数（供 _simulate_matrix 和 _feedback_method 使用）
        self._gain_coefficients = gain_for_experiment

        # 缓存选取的增益系数信息（供 plot_gain_coefficients 使用）
        self._picked_gain_8 = gain_8
        self._picked_floquet_gains_3 = floquet_gains_3
        self._picked_cr = picked_cr
        self._picked_ci = picked_ci

        # 收敛性分析 + 自适应松弛因子
        eigenvalues, rho_unrelaxed, best_alpha, rho_relaxed = (
            self._analyze_convergence()
        )
        self._relaxation_factor = best_alpha
        self._iteration_eigenvalues = eigenvalues
        self._spectral_radius_unrelaxed = rho_unrelaxed
        self._spectral_radius_relaxed = rho_relaxed

        # ---- 阶段 2：矩阵求解，写入理论解 ----
        self.logger.info("simulate 阶段 2/3: 通过线性方程组求解理论稳态解...")
        theoretical_feedback, theoretical_total_ai = self._simulate_matrix()
        self._theoretical_feedback_ao_complex_amps = theoretical_feedback
        self._theoretical_total_ai_complex_amps = theoretical_total_ai
        self.logger.info(
            f"理论解 - 反馈 AO 复振幅模长: {np.abs(theoretical_feedback)}"
        )
        self.logger.info(
            f"理论解 - 总声场复振幅模长: {np.abs(theoretical_total_ai)}"
        )

        # ---- 阶段 3：解析迭代仿真（基于收敛判据的自动终止） ----
        self.logger.info(
            f"simulate 阶段 3/3: 解析迭代仿真，"
            f"容差={self._ITERATION_TOLERANCE:.1%}, "
            f"最大迭代={self._MAX_ITERATION_CYCLES}, "
            f"ao_amplitude_limit={ao_amplitude_limit} V"
        )
        converged = False
        prev_ai_complex_amps: np.ndarray | None = None
        relative_change = float("inf")

        for cycle_idx in range(1, self._MAX_ITERATION_CYCLES + 1):
            new_ao_complex_amps = self._feedback_method(
                ai_waveform=None, mode="analytical"
            )
            self._current_ao_complex_amps = new_ao_complex_amps.copy()

            # 收敛性判断：当前轮 vs 上一轮的 AI 总声场复振幅
            current_ai = self._ai_complex_amps_history[-1]
            if prev_ai_complex_amps is not None:
                diff_norm = np.linalg.norm(current_ai - prev_ai_complex_amps)
                ref_norm = np.linalg.norm(prev_ai_complex_amps)
                relative_change = (
                    diff_norm / ref_norm if ref_norm > 0 else diff_norm
                )
                if relative_change < self._ITERATION_TOLERANCE:
                    self.logger.info(
                        f"迭代仿真于第 {cycle_idx} 轮收敛 "
                        f"(相对变化={relative_change:.6f} < "
                        f"容差={self._ITERATION_TOLERANCE:.1%})"
                    )
                    converged = True
                    break

            prev_ai_complex_amps = current_ai.copy()

            if cycle_idx % 100 == 0:
                self.logger.debug(
                    f"[simulate] 周期 {cycle_idx}/{self._MAX_ITERATION_CYCLES} "
                    f"相对变化={relative_change:.6f}"
                )

        if not converged:
            self.logger.warning(
                f"迭代仿真未在 {self._MAX_ITERATION_CYCLES} 轮内收敛，"
                f"系统可能发散！最终相对变化={relative_change:.6f}"
            )

        if self._ai_complex_amps_history and self._ao_complex_amps_history:
            self.logger.info(
                f"迭代仿真终点（{len(self._ai_complex_amps_history)} 轮） - "
                f"反馈 AO 复振幅模长: "
                f"{np.abs(self._ao_complex_amps_history[-1])}"
            )
            self.logger.info(
                f"迭代仿真终点 - 总声场复振幅模长: "
                f"{np.abs(self._ai_complex_amps_history[-1])}"
            )

        # ---- 阶段 4：可选绘图 ----
        if result_folder is not None:
            try:
                result_folder_path = Path(result_folder)
                result_folder_path.mkdir(parents=True, exist_ok=True)
                actual_cycles = len(self._ai_complex_amps_history)
                self.plot_evolution(
                    save_path=result_folder_path / f"sim_ai_{actual_cycles}steps.png",
                    target="ai",
                    mode="absolute",
                )
                self.plot_evolution(
                    save_path=result_folder_path / f"sim_ao_{actual_cycles}steps.png",
                    target="ao",
                    mode="absolute",
                )
                self.plot_gain_coefficients(
                    save_path=result_folder_path / "sim_gains_complex.png",
                )
            except Exception as e:
                self.logger.error(f"保存仿真结果失败: {e}", exc_info=True)

        return theoretical_feedback, theoretical_total_ai

    # =========================================================================
    #  波形构建辅助
    # =========================================================================

    def _build_combined_waveform(
        self,
        feedback_complex_amps: np.ndarray,
    ) -> Waveform:
        """
        基于既定的 static AO 复振幅和给定的 feedback AO 复振幅，
        生成一个覆盖所有 AO 通道（static + feedback）的合并波形。

        合并波形的通道顺序与 `self._ao_channels_combined` 保持一致，
        作为 `SingleChasCSIO.update_static_output_waveform` 的输入。

        Args:
            feedback_complex_amps: 反馈通道复振幅，shape (n_feedback,)

        Returns:
            合并的多通道 Waveform。
        """
        combined_cca = np.concatenate(
            [self._static_ao_complex_amps, np.asarray(feedback_complex_amps, dtype=np.complex128)]
        )
        combined_waveform = get_sine(
            sampling_info=self._user_static_output_waveform.sampling_info,
            frequency=self._frequency,
            channel_names=self._ao_channels_combined,
            channel_complex_amplitudes=combined_cca,
        )
        return combined_waveform

    # =========================================================================
    #  反馈数据处理（主线程内部使用，不再作为 CSIO 回调）
    # =========================================================================

    def _feedback_method(
        self,
        ai_waveform: Waveform | None,
        mode: Literal["acquisition", "analytical"] = "acquisition",
    ) -> np.ndarray:
        """
        基于一段（实测或解析推算的）"总声场复振幅"，按反馈律计算下一轮的反馈
        AO 复振幅。

        ## 模式

        - `"acquisition"`：**实测模式**（默认）。从传入的 `ai_waveform` 中通过
          带通滤波 + 单频信息提取得到"总声场复振幅"。该模式与硬件采集流程
          严格对接，反馈结果会受到外部噪声、串扰、设备畸变等因素影响。
        - `"analytical"`：**解析模式**。完全忽略 `ai_waveform`（甚至允许传入
          None），转而用预先存储的传递矩阵
          `T_i = sum_s tf_static[s, i] * amps_static[s]
                 + sum_j tf_feedback[j, i] * old_ao[j]`
          直接推算"总声场复振幅"。该模式排除一切外部噪声/串扰/采集畸变，
          仅保留反馈律本身的数值行为，方便调试程序逻辑与对比理论。

        ## 通用步骤（无论哪种模式都执行）

        1. 取得"总声场复振幅" T（acquisition 来自实测、analytical 来自传递矩阵）；
        2. 旧反馈 AO 复振幅取 `self._current_ao_complex_amps`；
        3. 用 `tf_diag` 扣除自身反馈通道的贡献得到入射声场复振幅；
        4. 由增益系数计算"全步长"反馈 AO 更新量 Δa，并按
           **自适应松弛因子** ``self._relaxation_factor`` (α*) 缩放：
           ``a_new = a_old + α* · Δa``。该 α* 在 ``simulate()`` 阶段由
           ``_analyze_convergence`` 计算，目的是让迭代算子
           ``(1-α)I + α A`` 的谱半径尽量小（< 1 即收敛）；
        5. 幅值安全检查（任意通道超限即抛错）；
        6. 同时把本轮 T 和新 AO 复振幅追加到内部历史轨迹中。

        Args:
            ai_waveform: 一整段稳态下的 AI 多通道波形。`mode="analytical"` 时
                可传入 None（被忽略）。
            mode: 反馈数据来源模式，详见上方说明。

        Returns:
            新的反馈 AO 复振幅向量，shape (n_feedback,)。

        Raises:
            ValueError: 当 mode 非法，或 acquisition 模式下 ai_waveform 为 None 时。
            RuntimeError: 当反馈 AO 幅值超过 `self._ao_amplitude_limit` 时。
        """
        # ---- Step 1: 取得"总声场复振幅" ----
        if mode == "acquisition":
            if ai_waveform is None:
                raise ValueError(
                    "_feedback_method(mode='acquisition') 需要传入有效的 ai_waveform"
                )
            # 带通滤波
            ai_filtered = filter_waveform(ai_waveform, self._sos)
            assert isinstance(ai_filtered, Waveform)
            # 单频信息提取
            ai_filtered = extract_single_tone_information_vvi(
                input_waveform=ai_filtered,
                approx_freq=self._frequency,
                precise_mode=True,
            )
            total_ai_complex_amps = np.asarray(
                ai_filtered.channel_complex_amplitudes, dtype=np.complex128
            )
        elif mode == "analytical":
            # T = static_ao @ tf_static_to_ai + current_ao @ tf_feedback_to_ai
            # 形状：static_ao:(n_static,), tf_static_to_ai:(n_static, n_ai)
            #       current_ao:(n_feedback,), tf_feedback_to_ai:(n_feedback, n_ai)
            total_ai_complex_amps = (
                self._static_ao_complex_amps @ self._tf_static_to_ai
                + self._current_ao_complex_amps @ self._tf_feedback_to_ai
            ).astype(np.complex128)
        else:
            raise ValueError(
                f"非法 mode: {mode!r}，仅支持 'acquisition' 或 'analytical'。"
            )

        # 储存本轮总声场复振幅
        self._ai_complex_amps_history.append(total_ai_complex_amps.copy())
        self.logger.debug(
            f"[{mode}] 第 {len(self._ai_complex_amps_history)} 轮总声场复振幅模长: "
            f"{np.abs(total_ai_complex_amps)}"
        )

        # ---- Step 2: 旧反馈 AO 复振幅（直接取自当前缓存） ----
        old_ao_complex_amps = self._current_ao_complex_amps.copy()

        # ---- Step 3: 入射声场复振幅 ----
        # incident[i] = total[i] - old_ao[i] * tf_diag[i]
        incident_complex_amps = (
            total_ai_complex_amps - old_ao_complex_amps * self._tf_diag
        )

        # ---- Step 4: 计算新反馈 AO 复振幅（含自适应松弛因子 α*） ----
        # 反馈律的"全步长"更新量：delta_ao = (g·incident - T) / d
        # 引入松弛因子 α∈(0, 1] 后，每周期只前进 α 倍，使迭代算子从 A 变为
        # (1-α)I + α A，从而把谱半径从 ρ(A) 压低到 ρ(A_α)（详见
        # `_analyze_convergence` 的推导）。α* 由初始化阶段一次性确定。
        target_total_ai = incident_complex_amps * self._gain_coefficients
        delta_ai = target_total_ai - total_ai_complex_amps
        delta_ao = delta_ai / self._tf_diag
        new_ao_complex_amps = (
            old_ao_complex_amps + self._relaxation_factor * delta_ao
        ).astype(np.complex128)

        self.logger.debug(
            f"[{mode}] 松弛因子 α={self._relaxation_factor:.4f}, "
            f"新反馈 AO 复振幅模长: {np.abs(new_ao_complex_amps)}"
        )

        # ---- Step 5: 幅值安全检查 ----
        magnitudes: np.ndarray = np.abs(new_ao_complex_amps)
        if np.max(magnitudes) > self._ao_amplitude_limit:
            exceeding_indices = np.where(magnitudes > self._ao_amplitude_limit)[0]
            msg = (
                f"反馈 AO 幅值超过安全上限 {self._ao_amplitude_limit} V! "
                f"超限通道索引: {exceeding_indices.tolist()}, "
                f"幅值: {magnitudes[exceeding_indices].tolist()}"
            )
            self.logger.error(msg)
            raise RuntimeError(msg)

        # ---- Step 6: 追加 AO 历史 ----
        self._ao_complex_amps_history.append(new_ao_complex_amps.copy())

        return new_ao_complex_amps

    # =========================================================================
    #  数据导出回调（仅用于把最新一段 AI 波形交给主线程）
    # =========================================================================

    def _data_export_callback(
        self,
        ai_waveform: Waveform,
        ao_static_waveform: Waveform,
        ao_feedback_waveform: Waveform | None,
        chunks_num: int,
    ) -> None:
        """
        SingleChasCSIO 的导出回调。

        在新架构下，CSIO 的 feedback 通道为空，`ao_feedback_waveform` 始终为 None；
        `ao_static_waveform` 即当前正在播放的"合并波形"。

        本回调只负责把最新一段 AI 波形和当前的 chunks 序号写入共享变量，
        并通过 `_ai_data_event` 通知主线程。所有耗时的反馈处理逻辑都在主线程
        `evolve()` 循环里完成，从而完全不受回调延迟限制。
        """
        with self._ai_data_lock:
            self._latest_ai_waveform = ai_waveform
            self._latest_ai_chunks_num = chunks_num
        self._ai_data_event.set()

    # =========================================================================
    #  主控方法：evolve
    # =========================================================================

    def evolve(
        self,
        cycles_num: PositiveInt = 10,
        ao_amplitude_limit: PositiveFloat = 0.5,
        result_folder: str | Path | None = None,
    ) -> Waveform | None:
        """
        启动反馈演化过程（阻塞执行）。

        演化过程使用最后一次 ``simulate()`` 调用所选取的增益系数和松弛因子，
        执行如下步骤：

        1. 重置内部状态；
        2. 启动 SingleChasCSIO（CSIO 的 feedback 通道为空，仅以稳态模式输出
           合并波形）；
        3. 循环执行 num_cycles 个演化周期：
           - 等待 ``settle_time`` 秒以确保稳态；
           - 取一段最新的 AI 波形；
           - 调用 ``_feedback_method`` 处理数据，得到新的反馈 AO 复振幅；
           - 构建新的合并波形并通过 ``update_static_output_waveform`` 更换；
        4. 停止 CSIO，构建并返回最终合并波形；
        5. 如果提供 ``result_folder``，自动保存最终波形和演化轨迹图。

        Args:
            cycles_num: 演化周期数，默认 10。
            ao_amplitude_limit: 反馈 AO 幅值安全上限（V），默认 0.5 V。
            result_folder: 结果保存文件夹路径，若提供则自动保存最终波形和演化图。

        Returns:
            最终合并的多通道 Waveform；若演化未成功则返回 None。

        Raises:
            RuntimeError: 当 AO 幅值超限或任务启动失败时；当尚未通过
                ``simulate(...)`` 取得理论解和增益系数时也会拒绝执行。
        """
        # ---- 前置校验：必须先有理论解和增益系数 ----
        if (
            self._theoretical_feedback_ao_complex_amps is None
            or self._theoretical_total_ai_complex_amps is None
        ):
            raise RuntimeError(
                "尚未获取理论解，请先调用 simulate(...) 方法，再执行 evolve。"
            )
        if self._gain_coefficients is None or self._relaxation_factor is None:
            raise RuntimeError(
                "尚未设置增益系数或松弛因子，请先调用 simulate(...) 方法。"
            )

        # ---- 重置状态 ----
        self._target_num_cycles = cycles_num
        self._ao_amplitude_limit = ao_amplitude_limit
        self._stop_flag = False
        self._evolve_error = None
        self._ai_complex_amps_history.clear()
        self._ao_complex_amps_history.clear()
        self._current_ao_complex_amps = np.zeros(
            len(self._ao_channels_feedback), dtype=np.complex128
        )
        self._evolution_data["ai_data_list"].clear()
        with self._ai_data_lock:
            self._latest_ai_waveform = None
            self._latest_ai_chunks_num = 0
        self._ai_data_event.clear()

        # 关键：重置 SingleChasCSIO 内部记录的"最近一次稳态输出波形"。
        # CSIO 的 `_static_output_waveform` 是实例属性，stop() 后并不会被清理；
        # 若直接进入下一次 start()，将以上一次演化的最终输出波形预填 AO 缓冲区，
        # 造成"残余状态污染"。这里直接将其重置为初始合并波形（feedback 全 0）。
        self._measure_controller._static_output_waveform = (  # noqa: SLF001
            self._initial_combined_waveform.copy()
        )

        final_waveform: Waveform | None = None

        chunk_duration = self._user_static_output_waveform.duration
        wait_time = self._wait_time_after_update

        self.logger.info("=" * 60)
        self.logger.info("开始反馈演化")
        self.logger.info(f"演化周期数: {cycles_num}")
        self.logger.info(f"增益系数: {self._gain_coefficients}")
        self.logger.info(f"松弛因子 α*: {self._relaxation_factor:.4f}")
        self.logger.info(f"AO 幅值安全上限: {ao_amplitude_limit} V")
        self.logger.info(f"每段时长: {chunk_duration:.3f} s")
        self.logger.info(f"每轮稳态等待: {wait_time:.3f} s")
        self.logger.info(
            f"预计总时长: {cycles_num * (wait_time + chunk_duration):.1f} s"
        )
        self.logger.info("=" * 60)

        try:
            # ---- 启动 CSIO ----
            self._measure_controller.start()
            self.logger.info("SingleChasCSIO 任务已启动（稳态模式）")

            # ---- 演化主循环 ----
            for cycle_idx in range(1, cycles_num + 1):
                # 1. 稳态等待：让上一次（或初始）波形充分形成稳态
                self.logger.debug(f"周期 {cycle_idx}: 等待 {wait_time:.3f} s 稳态...")
                time.sleep(wait_time)

                # 2. 清除之前的事件 / 缓存，等待一段全新的 AI 数据
                with self._ai_data_lock:
                    self._latest_ai_waveform = None
                self._ai_data_event.clear()
                self.logger.debug(f"周期 {cycle_idx}: 等待新一段 AI 数据...")
                # 启用导出，使主线程能从回调获取 AI 数据
                self._measure_controller.enable_export = True
                triggered = self._ai_data_event.wait(timeout=chunk_duration * 5 + 5.0)
                # 禁用导出
                self._measure_controller.enable_export = False
                if not triggered:
                    raise RuntimeError(
                        f"周期 {cycle_idx} 等待 AI 数据超时（>{chunk_duration * 5 + 5.0:.1f} s）"
                    )

                with self._ai_data_lock:
                    if self._latest_ai_waveform is None:
                        raise RuntimeError(
                            f"周期 {cycle_idx} 未取到 AI 波形，可能任务异常"
                        )
                    ai_waveform = self._latest_ai_waveform
                    chunks_num = self._latest_ai_chunks_num

                # 3. 记录本轮原始 AI 波形（x 坐标即周期序号）
                point_data: PointSweepData = {
                    "position": Point2D(float(cycle_idx), 0.0),
                    "ai_data": [ai_waveform],
                }
                self._evolution_data["ai_data_list"].append(point_data)
                self.logger.debug(
                    f"周期 {cycle_idx}: 已记录 AI 波形（CSIO chunks_num={chunks_num}）"
                )

                # 4. 处理数据，得到新反馈 AO 复振幅（实测模式）
                new_ao_complex_amps = self._feedback_method(
                    ai_waveform=ai_waveform,
                    mode="acquisition",
                )

                # 5. 更新当前缓存
                self._current_ao_complex_amps = new_ao_complex_amps.copy()

                # 6. 构建新的合并波形并切换稳态输出
                new_combined = self._build_combined_waveform(new_ao_complex_amps)
                self._measure_controller.update_static_output_waveform(new_combined)
                self.logger.info(
                    f"周期 {cycle_idx}/{cycles_num} 完成，"
                    f"已切换稳态输出"
                )

            # ---- 演化成功，构建最终合并波形 ----
            final_waveform = self._build_combined_waveform(self._current_ao_complex_amps)

            # ---- 可选：保存结果 ----
            if result_folder is not None:
                try:
                    result_folder_path = Path(result_folder)
                    result_folder_path.mkdir(parents=True, exist_ok=True)

                    # 演化原始数据
                    sweep_data_save_path = result_folder_path / "raw_sweep_data.pkl"
                    save_compressed_data(
                        self._evolution_data,
                        sweep_data_save_path,
                        data_type_name="演化SweepData",
                    )

                    # 最终合并波形
                    waveform_save_path = result_folder_path / "evolved_waveform.pkl"
                    save_compressed_data(
                        final_waveform,
                        waveform_save_path,
                        data_type_name="演化结果Waveform",
                    )

                    # 演化轨迹图（同时绘制两种模式下 AI 总声场轨迹与反馈 AO 轨迹）
                    self.plot_evolution(
                        save_path=result_folder_path / f"evo_ai_{cycles_num}steps.png",
                        target="ai",
                    )
                    self.plot_evolution(
                        save_path=result_folder_path / f"evo_ao_{cycles_num}steps.png",
                        target="ao",
                    )
                    self.plot_evolution(
                        save_path=result_folder_path / f"evo_ai(diff)_{cycles_num}steps.png",
                        target="ai",
                        mode="diff",
                    )
                    self.plot_evolution(
                        save_path=result_folder_path / f"evo_ao(diff)_{cycles_num}steps.png",
                        target="ao",
                        mode="diff",
                    )

                    self.logger.info(f"结果已保存到: {result_folder_path}")
                except Exception as e:
                    self.logger.error(f"保存结果失败: {e}", exc_info=True)

        finally:
            # ---- 始终停止任务 ----
            try:
                self._measure_controller.stop()
                self.logger.info("SingleChasCSIO 任务已停止")
            except Exception as e:
                self.logger.error(f"停止任务时出错: {e}", exc_info=True)

        self.logger.info("=" * 60)
        self.logger.info("反馈演化完成")
        self.logger.info(
            f"共完成 {len(self._ai_complex_amps_history)} 个演化周期"
        )
        self.logger.info("=" * 60)

        return final_waveform

    # =========================================================================
    #  绘图方法
    # =========================================================================

    def plot_evolution(
        self,
        save_path: str | Path | None = None,
        show: bool = False,
        target: Literal["ai", "ao"] = "ai",
        mode: Literal["absolute", "diff"] = "absolute",
    ) -> None:
        """
        在复平面上绘制各通道的演化轨迹。

        ## 模式

        - `"absolute"`（默认，simulate 使用）：直接绘制"实测/模拟值"本身的复平面轨迹。
          该模式不依赖理论解；若理论解已存在（即已调用过 `simulate(...)`），
          则会作为参考点（红色 ×）一并标注于图中。
        - `"diff"`（evolve 使用）：绘制"实测/模拟值 - 理论稳态解"的差值轨迹。
          理想情况下随着演化进行，所有折线应逐渐向坐标原点（理论稳态解所在位置）
          收敛。该模式要求理论解必须已通过 `simulate(...)` 求得。

        理论稳态解（即每个通道在理想反馈下应当达到的复振幅）由 `simulate(...)`
        方法的**矩阵阶段**事先求得并储存于 `_theoretical_total_ai_complex_amps`
        （针对 AI）与 `_theoretical_feedback_ao_complex_amps`（针对反馈 AO），
        与迭代仿真终点无关。

        Args:
            save_path: 图像保存路径（包含扩展名），可选。
            show: 是否调用 plt.show() 显示图像，默认 False。
            target: 绘图目标。
                - `"ai"`（默认）：绘制 AI 各通道的"总声场复振幅"轨迹；
                - `"ao"`：绘制反馈 AO 各通道的"反馈复振幅"轨迹。
            mode: 绘图模式，详见上方说明。

        Raises:
            ValueError: 若没有可绘制的演化数据；若 target / mode 取值非法；
                若选择 `mode="diff"` 但所选 target 对应的理论解尚未求得。
        """
        # ---- 选择数据源 ----
        if target == "ai":
            history = self._ai_complex_amps_history
            theoretical = self._theoretical_total_ai_complex_amps
            channel_names = self._ai_channels
            quantity_label = "AI 总声场复振幅"
        elif target == "ao":
            history = self._ao_complex_amps_history
            theoretical = self._theoretical_feedback_ao_complex_amps
            channel_names = self._ao_channels_feedback
            quantity_label = "反馈 AO 复振幅"
        else:
            raise ValueError(
                f"非法 target: {target!r}，仅支持 'ai' 或 'ao'。"
            )

        if mode not in ("diff", "absolute"):
            raise ValueError(
                f"非法 mode: {mode!r}，仅支持 'absolute' 或 'diff'。"
            )

        if not history:
            raise ValueError(
                "没有可绘制的演化数据，请先调用 evolve(...) 或 simulate(...) 方法。"
            )
        if mode == "diff" and theoretical is None:
            raise ValueError(
                f"target={target!r} 对应的理论解尚未求得，无法绘制 mode='diff' 模式。"
                "请先调用 simulate(...) 方法（建议使用 mode='iteration'），"
                "或改用 mode='absolute' 直接绘制实际值轨迹。"
            )

        num_cycles = len(history)
        n_ch = len(channel_names)

        # 数据源：history 形状 (cycles_num, n_ch)
        data_array = np.asarray(history)
        if mode == "absolute":
            plot_array = data_array
            xlabel = f"实部（{quantity_label}）"
            ylabel = f"虚部（{quantity_label}）"
            title = (
                f"反馈演化 - {quantity_label} 复平面轨迹\n"
                f"（周期数: {num_cycles}，频率: {self._frequency:.1f} Hz）"
            )
        else:  # mode == "diff"
            # 相对于理论解的差值轨迹
            plot_array = data_array - theoretical[None, :]
            xlabel = f"实部（{quantity_label} - 理论）"
            ylabel = f"虚部（{quantity_label} - 理论）"
            title = (
                f"反馈演化 - {quantity_label} 与 理论稳态解 之差\n"
                f"（周期数: {num_cycles}，频率: {self._frequency:.1f} Hz）"
            )

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        colors = plt.cm.tab20(np.linspace(0, 1, max(n_ch, 2)))

        for i in range(n_ch):
            trajectory = plot_array[:, i]
            real_parts = trajectory.real
            imag_parts = trajectory.imag

            channel_label = (
                channel_names[i] if i < len(channel_names) else f"通道{i + 1}"
            )
            color = colors[i % len(colors)]

            # 折线
            ax.plot(
                real_parts, imag_parts,
                color=color, linewidth=2, alpha=0.7,
                label=channel_label, zorder=1,
            )

            # 中间点
            if num_cycles > 2:
                ax.scatter(
                    real_parts[1:-1], imag_parts[1:-1],
                    color=color, s=30, alpha=0.5,
                    edgecolors="white", linewidths=0.5, zorder=2,
                )

            # 起点（方形）
            ax.scatter(
                real_parts[0], imag_parts[0],
                marker="s", s=150, color=color,
                edgecolors="black", linewidths=2, alpha=0.9, zorder=3,
            )

            # 终点（星形）
            ax.scatter(
                real_parts[-1], imag_parts[-1],
                marker="*", s=300, color=color,
                edgecolors="black", linewidths=2, alpha=0.9, zorder=3,
            )

            if i == 0:
                ax.annotate(
                    "起点",
                    xy=(real_parts[0], imag_parts[0]),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=9, fontweight="bold",
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "yellow", "alpha": 0.7},
                )
                ax.annotate(
                    "终点",
                    xy=(real_parts[-1], imag_parts[-1]),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=9, fontweight="bold",
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightgreen", "alpha": 0.7},
                )

        # 坐标轴参考线
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        # 理论解标注
        if mode == "diff":
            # diff 模式：理论解恒在原点
            ax.plot(
                0, 0,
                marker="x", markersize=12,
                color="red", markeredgewidth=2,
                label="理论稳态解（原点）",
            )
        elif theoretical is not None:
            # absolute 模式：若理论解已存在，将其逐通道作为红色 × 标注
            ax.scatter(
                theoretical.real, theoretical.imag,
                marker="x", s=120,
                color="red", linewidths=2,
                label="理论稳态解",
                zorder=4,
            )

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()

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

    def plot_gain_coefficients(
        self,
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> None:
        """
        在复平面上绘制所选 (cr, ci) 组合的增益系数图示。

        无论 simulate 时 mode 为何，均绘制 11 个增益系数点：

        - 8 周期模式的 8 个增益系数：使用圆形绘图点，并用折线依次连接
          （1→2→3→...→7→8，但 8 与 1 不相连）。
        - Floquet 周期模式的 3 个增益系数 [bnd1, bnd2, point1]：
          使用不同形状绘图点（point1 用菱形、bnd1 用上三角、bnd2 用下三角），
          彼此不连接。

        本方法需要先调用 ``simulate()`` 以选取增益系数，否则将拒绝执行。

        Args:
            save_path: 图像保存路径（包含扩展名），可选。
            show: 是否调用 plt.show() 显示图像，默认 False。

        Raises:
            RuntimeError: 当尚未调用 simulate() 选取增益系数时。
        """
        if (
            self._picked_gain_8 is None
            or self._picked_floquet_gains_3 is None
        ):
            raise RuntimeError(
                "尚未选取增益系数，请先调用 simulate(...) 方法。"
            )

        gain_8 = self._picked_gain_8          # shape (8,)
        floquet_3 = self._picked_floquet_gains_3  # shape (3,) [bnd1, bnd2, point1]
        picked_cr = self._picked_cr
        picked_ci = self._picked_ci

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # ---- 绘制 8 周期模式增益系数 ----
        # 圆形绘图点 + 折线连接（1→2→...→8，8与1不相连）
        colors_8 = plt.cm.tab10(np.linspace(0, 0.8, 8))
        for i in range(8):
            ax.scatter(
                gain_8[i].real, gain_8[i].imag,
                marker="o", s=120, color=colors_8[i],
                edgecolors="black", linewidths=1, zorder=3,
                label=f"8周期 g{i + 1} ({np.abs(gain_8[i]):.3f})",
            )
        # 折线连接 1→2→...→8（不闭合）
        ax.plot(
            gain_8.real, gain_8.imag,
            color="steelblue", linewidth=1.5, alpha=0.6,
            linestyle="-", zorder=2,
        )

        # ---- 绘制 Floquet 模式增益系数 ----
        # point1 (idx=2) → 菱形, bnd1 (idx=0) → 上三角, bnd2 (idx=1) → 下三角
        floquet_labels = ["bnd1", "bnd2", "point1"]
        floquet_markers = ["^", "v", "D"]  # 上三角, 下三角, 菱形
        floquet_colors = ["forestgreen", "darkorange", "crimson"]
        for i in range(3):
            ax.scatter(
                floquet_3[i].real, floquet_3[i].imag,
                marker=floquet_markers[i], s=180,
                color=floquet_colors[i],
                edgecolors="black", linewidths=1.5, zorder=4,
                label=(
                    f"Floquet {floquet_labels[i]} "
                    f"({np.abs(floquet_3[i]):.3f})"
                ),
            )

        # ---- 参考线和装饰 ----
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        # 绘制单位圆参考线
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta),
                color="lightgray", linewidth=1, linestyle=":", alpha=0.5)

        ax.set_xlabel("实部", fontsize=12)
        ax.set_ylabel("虚部", fontsize=12)
        ax.set_title(
            f"增益系数极坐标复平面\n"
            f"(cr={picked_cr:.6f}, ci={picked_ci:.6f}, "
            f"频率={self._frequency:.1f} Hz)",
            fontsize=14, fontweight="bold",
        )
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                self.logger.info(f"增益系数图已保存到: {save_path}")
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
        清理资源，停止内部 SingleChasCSIO 任务。

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
        """析构函数，确保资源被释放。"""
        try:
            self.cleanup()
        except Exception:
            # 析构期间忽略异常
            pass


# =============================================================================
#  独立辅助函数
# =============================================================================

def load_evolved_waveform(
    file_path: str | Path,
    segments: PositiveInt | None = None,
    picked_channels: tuple[str, ...] | None = None,
) -> Waveform:
    """
    从文件加载演化最终波形。

    加载由 `Evolver.evolve()` 在 `result_folder` 下保存的 `evolved_waveform.pkl`
    文件，返回内存中的 Waveform 对象。该波形可直接作为 `SweeperCore` 的
    `static_output_waveform` 参数使用，无需任何反馈逻辑即可重放出与 evolve
    最终状态一致的声场。

    Args:
        file_path: 波形文件路径（通常为 `evolved_waveform.pkl`）。
        segments: 分段平均的段数。Evolver 保存的波形通常较长（>= 0.5 s），
            若直接用于 Sweeper 扫场会显著增加每点耗时。提供此参数将对加载的
            波形调用 `average_single_waveform` 进行分段平均，使时长缩短为
            原来的 1 / segments。要求采样点数能被 segments 整除。
            默认 None（不压缩）。
        picked_channels: 需要保留的通道名称元组。若不为 None，则仅保留指定的
            通道（按 `picked_channels` 中的顺序排列），丢弃其余通道。若
            `picked_channels` 中包含原始波形不存在的通道名，则抛出 ValueError。
            默认 None（保留所有通道）。

    Returns:
        多通道 Waveform 对象，包含 static + feedback 通道的合并信号
        （若指定了 picked_channels 则仅包含所选通道）。

    Raises:
        FileNotFoundError: 当文件不存在时。
        IOError: 当文件读取失败时。
        ValueError: 当加载的数据不是有效的 Waveform，或 segments 不能整除采样点数时，
            或 picked_channels 中包含原始波形不存在的通道名时。
    """
    f_logger = get_logger(f"{__name__}.load_evolved_waveform")

    file_path = Path(file_path)
    assert isinstance(file_path, Path)
    if not file_path.exists():
        raise FileNotFoundError(f"演化波形文件不存在: {file_path}")

    loaded_data = load_compressed_data(file_path, data_type_name="演化波形")
    if not isinstance(loaded_data, Waveform):
        raise ValueError(
            f"加载的数据类型不正确，期望 Waveform，"
            f"实际为 {type(loaded_data).__name__}"
        )

    f_logger.info(
        f"演化波形加载成功: shape={loaded_data.shape}, "
        f"channels={loaded_data.channel_names}, "
        f"frequency={loaded_data.frequency}"
    )

    # ---- 分段平均 ----
    if segments is not None:
        try:
            loaded_data = average_single_waveform(loaded_data, segments=segments)
            f_logger.info(
                f"已应用分段平均压缩: segments={segments}, "
                f"压缩后 shape={loaded_data.shape}, "
                f"压缩后时长={loaded_data.duration:.4f} s"
            )
        except Exception as e:
            f_logger.error(f"分段平均压缩失败: {e}", exc_info=True)
            raise

    # ---- 通道筛选 ----
    if picked_channels is not None:
        loaded_data = pick_waveform_channels(loaded_data, picked_channels)
        f_logger.info(
            f"已筛选通道: picked_channels={picked_channels}, "
            f"筛选后 shape={loaded_data.shape}"
        )

    return loaded_data
