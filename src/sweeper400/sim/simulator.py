"""
# 仿真器模块

模块路径：`sweeper400.sim.simulator`

该模块包含 SimScanner 类，封装声学超表面 COMSOL 仿真自动化接口。
基于四个 COMSOL 模型文件，执行 (cr, ci) 参数空间的增益系数扫描仿真，
覆盖 8 周期模式和 Floquet 周期模式。

四个模型文件：
    - eight_probes_para_scan.mph: 8 周期参数扫描
    - eight_probes_single.mph: 8 周期单点仿真
    - floquet_probes_para_scan.mph: Floquet 周期参数扫描
    - floquet_probes_single.mph: Floquet 周期单点仿真
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import mph
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from ..analyze import load_compressed_data, load_data_with_fallback, save_compressed_data
from ..config import setup_chinese_fonts
from ..logger import get_logger

logger = get_logger(__name__)

#: 仿真模式字面量类型
SimulationMode = Literal["eight_scan", "eight_single", "floquet_scan", "floquet_single"]


@dataclass
class ScanResult:
    """参数扫描仿真结果数据类

    封装 8 周期模式和 Floquet 周期模式的完整扫描结果，包括
    增益系数、传递矩阵/函数、稳态和初态探针值等。

    Attributes:
        f: 仿真频率 (Hz)
        cr: 管槽声速实部放缩因子（扫描中心值）
        cr_min: cr 扫描下限
        cr_max: cr 扫描上限
        ci: 管槽声速虚部放缩因子（扫描中心值）
        ci_min: ci 扫描下限
        ci_max: ci 扫描上限
        res: cr/ci 共用扫描分辨率（每轴取值数）
        cr_values: cr 扫描值数组，shape (res,)
        ci_values: ci 扫描值数组，shape (res,)
        eight_target: 8 周期模式最终稳态 (point1~8)，shape (res, res, 8)
        eight_initial: 8 周期模式初态 (point1~8)，shape (8,)
        eight_tf_matrix: 8 周期模式仿真传递矩阵 (speaker→probe)，shape (8, 8)
        eight_gains: 8 周期模式增益系数，shape (res, res, 8)
        floquet_target: Floquet 模式最终稳态 (bnd1, bnd2, point1)，shape (res, res, 3)
        floquet_initial: Floquet 模式初态 (bnd1, bnd2, point1)，shape (3,)
        floquet_tf: Floquet 模式传递函数 (speaker→bnd1, bnd2, point1)，shape (3,)
        floquet_gains: Floquet 模式增益系数 (基于bnd1, bnd2, point1)，shape (res, res, 3)
        exp_eight_steady: 真实实验系统稳态（8 周期增益），shape (res, res, 8)
        exp_floquet_steady: 真实实验系统稳态（Floquet 增益），shape (res, res, 8)
        exp_eight_rho: 8 周期增益收敛性 - 未松弛谱半径 ρ(A)，shape (res, res)
        exp_eight_alpha: 8 周期增益收敛性 - 最优松弛因子 α*，shape (res, res)
        exp_eight_rho_relaxed: 8 周期增益收敛性 - 松弛后谱半径 ρ(A_α*)，shape (res, res)
        exp_floquet_rho: Floquet 增益收敛性 - 未松弛谱半径 ρ(A)，shape (res, res)
        exp_floquet_alpha: Floquet 增益收敛性 - 最优松弛因子 α*，shape (res, res)
        exp_floquet_rho_relaxed: Floquet 增益收敛性 - 松弛后谱半径 ρ(A_α*)，shape (res, res)
    """

    # 输入参数
    f: float
    cr: float
    cr_min: float
    cr_max: float
    ci: float
    ci_min: float
    ci_max: float
    res: int
    cr_values: np.ndarray  # shape (res,)
    ci_values: np.ndarray  # shape (res,)

    # 8 周期模式
    eight_target: np.ndarray  # shape (res, res, 8), complex128
    eight_initial: np.ndarray  # shape (8,), complex128
    eight_tf_matrix: np.ndarray  # shape (8, 8), complex128
    eight_gains: np.ndarray  # shape (res, res, 8), complex128

    # Floquet 周期模式
    floquet_target: np.ndarray  # shape (res, res, 3), complex128
    floquet_initial: np.ndarray  # shape (3,), complex128
    floquet_tf: np.ndarray  # shape (3,), complex128
    floquet_gains: np.ndarray  # shape (res, res, 3), complex128

    # 真实实验系统稳态
    exp_eight_steady: np.ndarray  # shape (res, res, 8), complex128
    exp_floquet_steady: np.ndarray  # shape (res, res, 8), complex128

    # 真实实验系统收敛性分析
    exp_eight_rho: np.ndarray  # shape (res, res)
    exp_eight_alpha: np.ndarray  # shape (res, res)
    exp_eight_rho_relaxed: np.ndarray  # shape (res, res)
    exp_floquet_rho: np.ndarray  # shape (res, res)
    exp_floquet_alpha: np.ndarray  # shape (res, res)
    exp_floquet_rho_relaxed: np.ndarray  # shape (res, res)

    def __repr__(self) -> str:
        return (
            f"ScanResult(\n"
            f"  f={self.f} Hz, cr={self.cr}, ci={self.ci}, res={self.res}\n"
            f"  cr_range=[{self.cr_min}, {self.cr_max}], "
            f"ci_range=[{self.ci_min}, {self.ci_max}]\n"
            f"  eight_gains shape={self.eight_gains.shape}, "
            f"mean |gain|={np.mean(np.abs(self.eight_gains)):.4f}\n"
            f"  floquet_gains shape={self.floquet_gains.shape}, "
            f"mean |gain|={np.mean(np.abs(self.floquet_gains)):.4f}\n"
            f")"
        )


@dataclass
class _SimCache:
    """仿真缓存数据结构

    缓存 run_scan 中仅依赖于 f、input_amp_l、input_amp_r 的计算结果，
    避免 f 和入射激励参数不变时重复执行耗时的 COMSOL 仿真。

    缓存范围（对应 run_scan 中的步骤）：
        - 步骤 2: 8 周期模式空管槽初态
        - 步骤 3: 8 周期模式 8×8 传递矩阵
        - 步骤 7: Floquet 模式空管槽初态
        - 步骤 8: Floquet 模式传递函数

    Attributes:
        f: 仿真频率 (Hz)
        input_amp_l: 左侧入射声源激励幅值 (COMSOL 参数字符串)
        input_amp_r: 右侧入射声源激励幅值 (COMSOL 参数字符串)
        eight_initial: 8 周期模式空管槽初态，shape (8,), complex128
        eight_tf_matrix: 8 周期模式传递矩阵，shape (8, 8), complex128
        floquet_initial: Floquet 模式空管槽初态，shape (3,), complex128
        floquet_tf: Floquet 模式传递函数，shape (3,), complex128
    """

    f: float
    input_amp_l: str
    input_amp_r: str
    eight_initial: np.ndarray
    eight_tf_matrix: np.ndarray
    floquet_initial: np.ndarray
    floquet_tf: np.ndarray

    def matches(self, f: float, input_amp_l: str, input_amp_r: str) -> bool:
        """检查缓存参数是否与给定参数匹配

        Args:
            f: 仿真频率 (Hz)
            input_amp_l: 左侧入射声源激励幅值
            input_amp_r: 右侧入射声源激励幅值

        Returns:
            参数匹配时返回 True，否则 False
        """
        return (
            self.f == f
            and self.input_amp_l == input_amp_l
            and self.input_amp_r == input_amp_r
        )


class SimScanner:
    """声学超表面 COMSOL 参数扫描仿真器

    基于四个 COMSOL 模型文件执行 (cr, ci) 参数空间的增益系数扫描仿真。
    管理与 COMSOL Server 的连接，提供参数化仿真运行和增益系数计算接口。

    run_scan() 主方法执行以下流程：
        1. 8 周期模式参数扫描 → 各参数组合的目标稳态 (point1~8)
        2. 8 周期模式空管槽仿真 → 初态（可缓存）
        3. 8 周期模式传递矩阵计算 (8 次单点仿真)（可缓存）
        4. 批量求解 8 周期模式增益系数
        5. 真实实验系统稳态计算 + 8 周期收敛性分析
        6. Floquet 模式参数扫描 → 各参数组合的目标稳态 (bnd1, bnd2, point1)
        7. Floquet 模式空管槽仿真 → 初态（可缓存）
        8. Floquet 模式传递函数计算 (1 次单点仿真)（可缓存）
        9. 求解 Floquet 模式增益系数 (3 种)
        10. 真实实验系统稳态计算 + Floquet 收敛性分析
        11. 保存数据和绘图

    步骤 2/3/7/8 的结果仅依赖于 f、input_amp_l、input_amp_r 参数，
    当这些参数不变时自动复用 ``storage/sim/sim_cache`` 中的缓存。

    Attributes:
        client: MPh Client 对象
        storage_dir: 仿真结果存储根目录
        last_result: 最近一次扫描结果
    """

    # =========================================================================
    # 类常量：四个 COMSOL 模型文件路径
    # =========================================================================

    _MPHS_DIR: Path = Path(__file__).parent / "mphs"

    #: 8 周期参数扫描模型（cr/ci 参数扫描，结果表含 cr, ci, point1~8）
    EIGHT_PARA_SCAN_FILE: Path = _MPHS_DIR / "eight_probes_para_scan.mph"

    #: 8 周期单点仿真模型（单个 cr/ci 点，结果表含 point1~8）
    EIGHT_SINGLE_FILE: Path = _MPHS_DIR / "eight_probes_single.mph"

    #: Floquet 周期参数扫描模型（cr/ci 参数扫描，结果表含 cr, ci, bnd1, bnd2, point1）
    FLOQUET_PARA_SCAN_FILE: Path = _MPHS_DIR / "floquet_probes_para_scan.mph"

    #: Floquet 周期单点仿真模型（单个 cr/ci 点，结果表含 bnd1, bnd2, point1）
    FLOQUET_SINGLE_FILE: Path = _MPHS_DIR / "floquet_probes_single.mph"

    # =========================================================================
    # 类常量：真实实验系统通道名称（硬编码，与 TFData 的 tf_dataframe 索引对应）
    # =========================================================================

    #: 8 个扬声器 AO 通道（speaker 1~8）
    _SPEAKER_CHANNELS: tuple[str, ...] = (
        "PXI1Slot3/ao0", "PXI1Slot3/ao1",
        "PXI1Slot4/ao0", "PXI1Slot4/ao1",
        "PXI1Slot5/ao0", "PXI1Slot5/ao1",
        "PXI1Slot6/ao0", "PXI1Slot6/ao1",
    )

    #: 8 个传声器 AI 通道（point 1~8）
    _POINT_CHANNELS: tuple[str, ...] = (
        "PXI1Slot3/ai0", "PXI1Slot3/ai1",
        "PXI1Slot4/ai0", "PXI1Slot4/ai1",
        "PXI1Slot5/ai0", "PXI1Slot5/ai1",
        "PXI1Slot6/ai0", "PXI1Slot6/ai1",
    )

    #: 静态激励 AO 通道
    _STATIC_AO_CHANNEL: str = "PXI1Slot2/ao0"

    #: 静态激励 AI 通道（用于计算初态）
    _STATIC_AI_CHANNEL: str = "PXI1Slot2/ai0"

    #: 默认 Fishnet TFData 存储路径（相对于 workspace 根目录）
    _DEFAULT_TF_DATA_PATH: str = "storage/calib/calib_result_fishnet/tf_data.pkl"

    def __init__(self) -> None:
        """初始化仿真器"""
        self.client: mph.Client | None = None
        self._models: dict[str, mph.Model] = {}

        # 存储路径：workspace_root / storage / sim
        self._workspace_root = Path(__file__).parent.parent.parent.parent
        self.storage_dir: Path = self._workspace_root / "storage" / "sim"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.last_result: ScanResult | None = None

        # 缓存目录：storage/sim/sim_cache
        self._sim_cache_dir: Path = self.storage_dir / "sim_cache"
        self._sim_cache_file: Path = self._sim_cache_dir / "sim_cache.pkl"

        logger.info("SimScanner 初始化完成")

    # =========================================================================
    # 连接管理
    # =========================================================================

    def connect(self) -> None:
        """连接到 COMSOL Server"""
        if self.client is not None:
            logger.debug("已处于连接状态，跳过重复连接")
            return
        try:
            self.client = mph.start(cores=1)
            logger.info("已连接到 COMSOL Server")
        except Exception as e:
            raise RuntimeError(f"无法连接到 COMSOL Server: {e}") from e

    def disconnect(self) -> None:
        """断开与 COMSOL Server 的连接"""
        if self.client is not None:
            try:
                self.client.disconnect()
                logger.info("已断开 COMSOL 连接")
            except Exception as e:
                logger.warning(f"断开连接时出现警告: {e}")
            finally:
                self.client = None
                self._models.clear()

    def __enter__(self) -> SimScanner:
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.disconnect()
        return False

    @property
    def is_connected(self) -> bool:
        return self.client is not None

    # =========================================================================
    # 缓存管理
    # =========================================================================

    def _load_sim_cache(
        self, f: float, input_amp_l: str, input_amp_r: str,
    ) -> _SimCache | None:
        """尝试加载仿真缓存

        从 ``storage/sim/sim_cache/sim_cache.pkl`` 加载缓存数据，
        并校验 f、input_amp_l、input_amp_r 参数是否匹配。

        Args:
            f: 仿真频率 (Hz)
            input_amp_l: 左侧入射声源激励幅值
            input_amp_r: 右侧入射声源激励幅值

        Returns:
            缓存命中且参数匹配时返回 _SimCache 对象，否则返回 None
        """
        if not self._sim_cache_file.exists():
            return None

        try:
            cache = load_compressed_data(
                self._sim_cache_file, data_type_name="仿真缓存",
            )
        except Exception as e:
            logger.warning(f"加载仿真缓存失败，将重新计算: {e}")
            return None

        if not isinstance(cache, _SimCache):
            logger.warning("仿真缓存格式不匹配，将重新计算")
            return None

        if not cache.matches(f, input_amp_l, input_amp_r):
            logger.info(
                f"仿真缓存参数不匹配 (缓存: f={cache.f}, "
                f"input_amp_l={cache.input_amp_l}, "
                f"input_amp_r={cache.input_amp_r}; "
                f"当前: f={f}, input_amp_l={input_amp_l}, "
                f"input_amp_r={input_amp_r})，将重新计算"
            )
            return None

        logger.info("仿真缓存命中，参数匹配")
        return cache

    def _save_sim_cache(self, cache: _SimCache) -> None:
        """保存仿真缓存到磁盘

        将缓存数据保存到 ``storage/sim/sim_cache/sim_cache.pkl``。

        Args:
            cache: 仿真缓存数据对象
        """
        try:
            save_compressed_data(
                cache, self._sim_cache_file,
                data_type_name="仿真缓存",
            )
        except Exception as e:
            logger.warning(f"保存仿真缓存失败: {e}")

    # =========================================================================
    # 主方法
    # =========================================================================

    def run_scan(
        self,
        f: float = 3430.0,
        cr: float = 1.006,
        cr_min: float = 1.004,
        cr_max: float = 1.008,
        ci: float = -0.073,
        ci_min: float = -0.075,
        ci_max: float = -0.071,
        res: int = 10,
        input_amp_l: str = "1[Pa]",
        input_amp_r: str = "0[Pa]",
        fishnet_tf_data_path: str | Path | None = None,
        result_folder: str | Path | None = None,
    ) -> ScanResult:
        """运行完整参数扫描仿真流程

        依次执行 8 周期模式和 Floquet 周期模式的参数扫描仿真，
        计算各参数组合下的增益系数，并保存数据和绘图。

        Args:
            f: 仿真频率 (Hz)，默认 3430
            cr: 管槽声速实部放缩因子（扫描中心值），默认 1.005
            cr_min: cr 扫描下限，默认 1.004
            cr_max: cr 扫描上限，默认 1.006
            ci: 管槽声速虚部放缩因子（扫描中心值），默认 -0.073
            ci_min: ci 扫描下限，默认 -0.074
            ci_max: ci 扫描上限，默认 -0.072
            res: cr/ci 共用扫描分辨率（每轴取值数），默认 10
            input_amp_l: 左侧入射声源激励幅值 (COMSOL 参数字符串)，默认 "1[Pa]"
            input_amp_r: 右侧入射声源激励幅值 (COMSOL 参数字符串)，默认 "0[Pa]"
            fishnet_tf_data_path: Fishnet 传递函数数据文件路径（可选）。
                若为 None 则从默认路径 (storage/calib/calib_result_fishnet) 读取。
                用于计算真实实验系统的预期稳态。
            result_folder: 结果保存文件夹路径（可选）。
                若为 None 则保存到默认路径 ``storage/sim/sim_result_scan``。
                若指定路径，则数据和绘图将保存到该路径下。

        Returns:
            ScanResult 对象

        Raises:
            RuntimeError: 如果未连接或仿真失败
        """
        if self.client is None:
            raise RuntimeError("未连接到 COMSOL Server，请先调用 connect() 方法")

        # ---- 加载真实实验系统 TFData ----
        tf_data = load_data_with_fallback(
            explicit_path=fishnet_tf_data_path,
            default_path=self._workspace_root / self._DEFAULT_TF_DATA_PATH,
            data_type="Fishnet传递函数数据",
        )
        exp_tf_matrix: np.ndarray | None = None
        exp_initial_state: np.ndarray | None = None
        if tf_data is not None:
            tf_df: pd.DataFrame = tf_data["tf_dataframe"]
            # 提取实验传递矩阵: speaker→probe, shape (8, 8)
            # tf_matrix[i, j] = speaker_i 到 point_j 的传递函数
            exp_tf_matrix = np.asarray(
                tf_df.loc[list(self._SPEAKER_CHANNELS), list(self._POINT_CHANNELS)].values,
                dtype=np.complex128,
            )
            # 初态: 静态激励 AO 通道在各 AI 通道的贡献 (激励=1)
            exp_initial_state = np.asarray(
                tf_df.loc[self._STATIC_AO_CHANNEL, list(self._POINT_CHANNELS)].values,
                dtype=np.complex128,
            )
            logger.info(
                f"已加载实验传递矩阵: shape={exp_tf_matrix.shape}, "
                f"初态模长={np.abs(exp_initial_state)}"
            )
        else:
            logger.warning("未找到 Fishnet TFData，将跳过真实实验系统稳态计算")

        logger.info(
            f"开始参数扫描仿真: f={f}Hz, cr=[{cr_min},{cr_max}], "
            f"ci=[{ci_min},{ci_max}], res={res}"
        )

        cr_values = np.linspace(cr_min, cr_max, res)
        ci_values = np.linspace(ci_min, ci_max, res)

        # ---- 加载仿真缓存 ----
        # 步骤 2/3/7/8 的结果仅依赖于 f、input_amp_l、input_amp_r，
        # 当这些参数不变时可复用缓存，避免重复执行耗时的 COMSOL 仿真。
        cache = self._load_sim_cache(f, input_amp_l, input_amp_r)
        cache_hit = cache is not None
        if cache_hit:
            logger.info("仿真缓存命中，跳过步骤 2/3/7/8 的 COMSOL 仿真")
        else:
            logger.info("仿真缓存未命中，将执行完整仿真并更新缓存")

        # =================================================================
        # 8 周期模式
        # =================================================================
        logger.info("=== 8 周期模式 ===")

        # 1. 参数扫描 → 目标稳态 (res*res 组, 每组 8 个探针)
        logger.info("1/10: eight_probes_para_scan 参数扫描...")
        eight_target_flat = self._run_single_simulation(
            "eight_scan", f, cr, cr_min, cr_max, ci, ci_min, ci_max, res,
            input_amp_l=input_amp_l, input_amp_r=input_amp_r,
            speaker_amps_8=np.zeros(8, dtype=complex),
        )
        eight_target = eight_target_flat.reshape(res, res, 8)
        logger.info(f"  目标稳态: shape={eight_target.shape}")

        # 2. 空管槽 (cr=1, ci=0) → 初态（可缓存）
        if cache_hit:
            logger.info("2/10: eight_probes_single 空管槽 (cr=1, ci=0) [缓存]...")
            eight_initial = cache.eight_initial
        else:
            logger.info("2/10: eight_probes_single 空管槽 (cr=1, ci=0)...")
            eight_initial = self._run_single_simulation(
                "eight_single", f, 1.0, ci_min, ci_max, 0.0, ci_min, ci_max, res,
                input_amp_l=input_amp_l, input_amp_r=input_amp_r,
                speaker_amps_8=np.zeros(8, dtype=complex),
            )
        logger.info(f"  初态: {np.abs(eight_initial)}")

        # 3. 传递矩阵 (8 次仿真)（可缓存）
        if cache_hit:
            logger.info("3/10: 计算 8x8 传递矩阵 [缓存]...")
            eight_tf_matrix = cache.eight_tf_matrix
        else:
            logger.info("3/10: 计算 8x8 传递矩阵...")
            eight_tf_matrix = self._compute_eight_transfer_matrix(f)

        # 4. 批量求解增益系数
        logger.info("4/10: 批量求解 8 周期模式增益系数...")
        eight_gains = self._solve_gains_batch(
            eight_target, eight_initial, eight_tf_matrix
        )
        logger.info(
            f"  增益系数: shape={eight_gains.shape}, "
            f"mean |gain|={np.mean(np.abs(eight_gains)):.4f}"
        )

        # 5. 使用仿真增益 + 实验传递矩阵计算真实实验系统稳态
        exp_eight_steady = np.zeros((res, res, 8), dtype=np.complex128)
        if exp_tf_matrix is not None and exp_initial_state is not None:
            logger.info("5/10: 计算真实实验系统稳态（8 周期增益）...")
            exp_eight_steady = self._solve_exp_steady_batch(
                eight_gains, exp_tf_matrix, exp_initial_state,
            )
            logger.info(
                f"  实验稳态: shape={exp_eight_steady.shape}, "
                f"mean |steady|={np.mean(np.abs(exp_eight_steady)):.4f}"
            )

        # 6. 收敛性分析
        exp_eight_rho = np.zeros((res, res))
        exp_eight_alpha = np.zeros((res, res))
        exp_eight_rho_relaxed = np.zeros((res, res))
        if exp_tf_matrix is not None:
            logger.info("  计算 8 周期模式收敛性...")
            exp_eight_rho, exp_eight_alpha, exp_eight_rho_relaxed = (
                self._analyze_convergence_batch(eight_gains, exp_tf_matrix)
            )
            safe_8 = int(np.sum(exp_eight_rho < 1.0))
            recoverable_8 = int(np.sum(
                (exp_eight_rho >= 1.0) & (exp_eight_rho_relaxed < 1.0)
            ))
            logger.info(
                f"  8周期: 安全区={safe_8}/{res*res}, "
                f"可恢复区={recoverable_8}/{res*res}"
            )

        # =================================================================
        # Floquet 周期模式
        # =================================================================
        logger.info("=== Floquet 周期模式 ===")

        # 7. 参数扫描 → 目标稳态 (res*res 组, 每组 bnd1+bnd2+point1)
        logger.info("6/10: floquet_probes_para_scan 参数扫描...")
        floquet_target_flat = self._run_single_simulation(
            "floquet_scan", f, cr, cr_min, cr_max, ci, ci_min, ci_max, res,
            positive_is_left="1", input_amp="1[Pa]",
            speaker_amp_floquet="0",
        )
        floquet_target = floquet_target_flat.reshape(res, res, 3)
        logger.info(f"  目标稳态: shape={floquet_target.shape}")

        # 8. 空管槽 → 初态（可缓存）
        if cache_hit:
            logger.info("7/10: floquet_probes_single 空管槽 (cr=1, ci=0) [缓存]...")
            floquet_initial = cache.floquet_initial
        else:
            logger.info("7/10: floquet_probes_single 空管槽 (cr=1, ci=0)...")
            floquet_initial = self._run_single_simulation(
                "floquet_single", f, 1.0, ci_min, ci_max, 0.0, ci_min, ci_max, res,
                positive_is_left="1", input_amp="1[Pa]",
                speaker_amp_floquet="0",
            )
        logger.info(f"  初态: {np.abs(floquet_initial)}")

        # 9. 传递函数 (1 次仿真, speaker_amp=1, input_amp=0)（可缓存）
        if cache_hit:
            logger.info("8/10: floquet_probes_single 传递函数 (speaker_amp=1) [缓存]...")
            floquet_tf = cache.floquet_tf
        else:
            logger.info("8/10: floquet_probes_single 传递函数 (speaker_amp=1)...")
            floquet_tf = self._run_single_simulation(
                "floquet_single", f, 1.0, ci_min, ci_max, 0.0, ci_min, ci_max, res,
                positive_is_left="1", input_amp="0[Pa]",
                speaker_amp_floquet="1",
            )
        logger.info(f"  传递函数: {np.abs(floquet_tf)}")

        # 10. 求解 Floquet 增益系数 (3 种目标探针)
        logger.info("9/10: 求解 Floquet 模式增益系数...")
        # 增益定义: 稳态传声器(point1)结果 / 初态传声器(point1)结果
        # 对于不同目标探针，通过传递函数反推传声器稳态值：
        #   1. 由目标探针的初态/稳态差和该探针的 TF，计算扬声器输出
        #   2. 由扬声器输出和 point1 的 TF，计算传声器测量值
        #   3. 传声器测量值 / 传声器初态 = 增益系数
        # 索引: 0=bnd1, 1=bnd2, 2=point1
        initial_point1 = floquet_initial[2]  # point1 初态
        tf_point1 = floquet_tf[2]  # speaker → point1 传递函数
        floquet_gains = np.zeros((res, res, 3), dtype=np.complex128)
        for probe_idx in range(3):
            target_probe = floquet_target[:, :, probe_idx]  # (res, res)
            initial_probe = floquet_initial[probe_idx]  # scalar
            tf_probe = floquet_tf[probe_idx]  # speaker → 该探针传递函数

            # 计算达到目标探针稳态所需的扬声器输出
            delta = target_probe - initial_probe
            speaker_amps = delta / tf_probe

            # 由扬声器输出推导传声器(point1)测量值
            mic_steady = initial_point1 + speaker_amps * tf_point1

            floquet_gains[:, :, probe_idx] = mic_steady / initial_point1

        logger.info(
            f"  Floquet 增益系数: shape={floquet_gains.shape}, "
            f"mean |gain|={np.mean(np.abs(floquet_gains)):.4f}"
        )

        # 取 point1 (idx=2) 对应的增益，复制为 8 个相同增益（供稳态和收敛性分析使用）
        floquet_point1_gains = np.repeat(
            floquet_gains[:, :, 2:3], 8, axis=2
        )  # (res, res, 8)

        # 11. 使用 Floquet 增益 + 实验传递矩阵计算真实实验系统稳态
        exp_floquet_steady = np.zeros((res, res, 8), dtype=np.complex128)
        if exp_tf_matrix is not None and exp_initial_state is not None:
            logger.info("10/10: 计算真实实验系统稳态（Floquet 增益）...")
            exp_floquet_steady = self._solve_exp_steady_batch(
                floquet_point1_gains, exp_tf_matrix, exp_initial_state,
            )
            logger.info(
                f"  实验稳态: shape={exp_floquet_steady.shape}, "
                f"mean |steady|={np.mean(np.abs(exp_floquet_steady)):.4f}"
            )

        # 12. 收敛性分析
        exp_floquet_rho = np.zeros((res, res))
        exp_floquet_alpha = np.zeros((res, res))
        exp_floquet_rho_relaxed = np.zeros((res, res))
        if exp_tf_matrix is not None:
            logger.info("  计算 Floquet 模式收敛性...")
            exp_floquet_rho, exp_floquet_alpha, exp_floquet_rho_relaxed = (
                self._analyze_convergence_batch(
                    floquet_point1_gains, exp_tf_matrix
                )
            )
            safe_f = int(np.sum(exp_floquet_rho < 1.0))
            recoverable_f = int(np.sum(
                (exp_floquet_rho >= 1.0) & (exp_floquet_rho_relaxed < 1.0)
            ))
            logger.info(
                f"  Floquet: 安全区={safe_f}/{res*res}, "
                f"可恢复区={recoverable_f}/{res*res}"
            )

        # ---- 保存仿真缓存（仅缓存未命中时） ----
        if not cache_hit:
            self._save_sim_cache(_SimCache(
                f=f, input_amp_l=input_amp_l, input_amp_r=input_amp_r,
                eight_initial=eight_initial,
                eight_tf_matrix=eight_tf_matrix,
                floquet_initial=floquet_initial,
                floquet_tf=floquet_tf,
            ))

        # =================================================================
        # 组装结果
        # =================================================================
        result = ScanResult(
            f=f, cr=cr, cr_min=cr_min, cr_max=cr_max,
            ci=ci, ci_min=ci_min, ci_max=ci_max, res=res,
            cr_values=cr_values, ci_values=ci_values,
            eight_target=eight_target,
            eight_initial=eight_initial,
            eight_tf_matrix=eight_tf_matrix,
            eight_gains=eight_gains,
            floquet_target=floquet_target,
            floquet_initial=floquet_initial,
            floquet_tf=floquet_tf,
            floquet_gains=floquet_gains,
            exp_eight_steady=exp_eight_steady,
            exp_floquet_steady=exp_floquet_steady,
            exp_eight_rho=exp_eight_rho,
            exp_eight_alpha=exp_eight_alpha,
            exp_eight_rho_relaxed=exp_eight_rho_relaxed,
            exp_floquet_rho=exp_floquet_rho,
            exp_floquet_alpha=exp_floquet_alpha,
            exp_floquet_rho_relaxed=exp_floquet_rho_relaxed,
        )

        self.last_result = result

        # 保存数据和绘图
        save_dir = self._save_scan_result(result, result_folder=result_folder)
        self._plot_scan_results(result, save_dir)

        logger.info(f"参数扫描仿真完成，结果已保存至: {save_dir}")
        return result

    # =========================================================================
    # 单次仿真执行
    # =========================================================================

    def _run_single_simulation(
        self,
        mode: SimulationMode,
        f: float,
        cr: float,
        cr_min: float,
        cr_max: float,
        ci: float,
        ci_min: float,
        ci_max: float,
        res: int,
        *,
        input_amp_l: str = "1[Pa]",
        input_amp_r: str = "0[Pa]",
        speaker_amps_8: np.ndarray | None = None,
        positive_is_left: str = "1",
        input_amp: str = "1[Pa]",
        speaker_amp_floquet: str = "0",
    ) -> np.ndarray:
        """运行单次仿真并返回探针数据

        根据 mode 参数选择不同的仿真模型和数据提取方式：
        - "eight_scan": 8 周期参数扫描，返回 shape (n_rows, 8) 的 point1~8
        - "eight_single": 8 周期单点，返回 shape (8,) 的 point1~8
        - "floquet_scan": Floquet 参数扫描，返回 shape (n_rows, 3) 的 bnd1+bnd2+point1
        - "floquet_single": Floquet 单点，返回 shape (3,) 或 (n_probe,) 的探针值

        Args:
            mode: 仿真模式，可选 "eight_scan", "eight_single",
                "floquet_scan", "floquet_single"
            f: 仿真频率 (Hz)
            cr: 管槽声速实部放缩因子
            cr_min: cr 扫描下限
            cr_max: cr 扫描上限
            ci: 管槽声速虚部放缩因子
            ci_min: ci 扫描下限
            ci_max: ci 扫描上限
            res: 扫描分辨率
            input_amp_l: 左侧入射激励幅值 (COMSOL 字符串)
            input_amp_r: 右侧入射激励幅值 (COMSOL 字符串)
            speaker_amps_8: 8 个扬声器幅值数组 (仅 eight 模式使用)
            positive_is_left: 入射方向 +1=左45° -1=右45° (仅 floquet 模式)
            input_amp: 入射激励幅值 (仅 floquet 模式)
            speaker_amp_floquet: 扬声器法向位移幅值 (仅 floquet 模式)

        Returns:
            探针复数数据 ndarray，形状取决于 mode
        """
        file_map: dict[SimulationMode, Path] = {
            "eight_scan": self.EIGHT_PARA_SCAN_FILE,
            "eight_single": self.EIGHT_SINGLE_FILE,
            "floquet_scan": self.FLOQUET_PARA_SCAN_FILE,
            "floquet_single": self.FLOQUET_SINGLE_FILE,
        }
        mph_file = file_map[mode]
        model = self._load_model(mph_file)
        java_model = model.java

        # 设置参数
        self._set_params(
            java_model, mode, f, cr, cr_min, cr_max, ci, ci_min, ci_max, res,
            input_amp_l=input_amp_l,
            input_amp_r=input_amp_r,
            speaker_amps=speaker_amps_8,
            positive_is_left=positive_is_left,
            input_amp=input_amp,
            speaker_amp=speaker_amp_floquet,
        )

        # 求解
        model.clear()
        model.solve()

        # 提取数据
        if mode == "eight_scan":
            return self._extract_scan_table(java_model, "point1", probe_start=3, n_probes=8)
        elif mode == "eight_single":
            return self._extract_single_row(java_model, "point1", probe_start=1, n_probes=8)
        elif mode == "floquet_scan":
            return self._extract_scan_table(java_model, "bnd1", probe_start=3, n_probes=3)
        else:  # floquet_single
            return self._extract_single_row(java_model, "bnd1", probe_start=1, n_probes=3)

    # =========================================================================
    # 参数设置
    # =========================================================================

    @staticmethod
    def _set_params(
        java_model,
        mode: SimulationMode,
        f: float,
        cr: float,
        cr_min: float,
        cr_max: float,
        ci: float,
        ci_min: float,
        ci_max: float,
        res: int,
        *,
        input_amp_l: str = "1[Pa]",
        input_amp_r: str = "0[Pa]",
        speaker_amps: np.ndarray | None = None,
        positive_is_left: str = "1",
        input_amp: str = "1[Pa]",
        speaker_amp: str = "0",
    ) -> None:
        """设置仿真模型参数（统一入口，根据 mode 设置模式特有参数）

        公共参数 (f, cr, cr_min, cr_max, ci, ci_min, ci_max, res) 对所有模式通用。
        模式特有参数通过关键字参数传入：
        - eight 模式: input_amp_l, input_amp_r, speaker_amps
        - floquet 模式: positive_is_left, input_amp, speaker_amp

        Args:
            java_model: COMSOL Java 模型对象
            mode: 仿真模式
            f: 仿真频率 (Hz)
            cr: 管槽声速实部放缩因子
            cr_min: cr 扫描下限
            cr_max: cr 扫描上限
            ci: 管槽声速虚部放缩因子
            ci_min: ci 扫描下限
            ci_max: ci 扫描上限
            res: 扫描分辨率
            input_amp_l: 左侧入射激励幅值 (仅 eight 模式)
            input_amp_r: 右侧入射激励幅值 (仅 eight 模式)
            speaker_amps: 8 个扬声器幅值数组 (仅 eight 模式)
            positive_is_left: 入射方向 (仅 floquet 模式)
            input_amp: 入射激励幅值 (仅 floquet 模式)
            speaker_amp: 扬声器法向位移幅值 (仅 floquet 模式)
        """
        p = java_model.param()

        # 公共参数
        p.set("f", f"{f}[1/s]")
        p.set("cr", str(cr))
        p.set("cr_min", str(cr_min))
        p.set("cr_max", str(cr_max))
        p.set("ci", str(ci))
        p.set("ci_min", str(ci_min))
        p.set("ci_max", str(ci_max))
        p.set("res", str(res))

        # 模式特有参数
        if mode.startswith("eight"):
            p.set("input_amp_l", input_amp_l)
            p.set("input_amp_r", input_amp_r)
            if speaker_amps is None:
                speaker_amps = np.zeros(8, dtype=complex)
            for i in range(8):
                amp = complex(speaker_amps[i])
                p.set(f"speaker_amp_{i + 1}", _format_complex(amp))
        else:
            p.set("positive_is_left", positive_is_left)
            p.set("input_amp", input_amp)
            p.set("speaker_amp", speaker_amp)

    # =========================================================================
    # 数据提取
    # =========================================================================

    @staticmethod
    def _get_table_data(java_model, probe_name: str) -> list[list[str]]:
        """从 COMSOL 探针获取原始表格数据（跳过表头行）"""
        probe = java_model.probe(probe_name)
        table_name = str(probe.getString("table"))
        result_table = java_model.result().table(table_name)
        raw_data = result_table.getTableData(True)

        if len(raw_data) < 1:
            raise RuntimeError(f"探针 '{probe_name}' 表数据为空")

        # 跳过表头行：尝试将首列转换为 float，失败则视为表头
        # 注意：getTableData 返回 java.lang.String 对象，需先转为 Python str
        data_rows: list[list[str]] = []
        for row in raw_data:
            try:
                float(str(row[0]))
                data_rows.append([str(c) for c in row])
            except (ValueError, IndexError, TypeError):
                continue

        if not data_rows:
            raise RuntimeError(f"探针 '{probe_name}' 表数据中无有效数据行")

        return data_rows

    def _extract_scan_table(
        self, java_model, probe_name: str,
        probe_start: int, n_probes: int,
    ) -> np.ndarray:
        """从参数扫描结果表提取探针数据

        参数扫描表结构: [freq, cr, ci, probe1, probe2, ...]
        数据行数为扫描总点数（如 res*res）。

        Args:
            java_model: Java 模型对象
            probe_name: 用于获取表名的探针名称
            probe_start: 探针数据起始列索引（0-based, freq/cr/ci 之后）
            n_probes: 探针数量

        Returns:
            shape (n_rows, n_probes) 的复数数组
        """
        data_rows = self._get_table_data(java_model, probe_name)
        n_rows = len(data_rows)
        result = np.zeros((n_rows, n_probes), dtype=np.complex128)

        for i, row in enumerate(data_rows):
            for j in range(n_probes):
                result[i, j] = _parse_complex_str(row[probe_start + j])

        logger.debug(
            f"参数扫描数据提取: {n_rows} 行 x {n_probes} 探针, "
            f"首行幅值={np.abs(result[0])}"
        )
        return result

    def _extract_single_row(
        self, java_model, probe_name: str,
        probe_start: int, n_probes: int,
    ) -> np.ndarray:
        """从单点仿真结果表提取探针数据

        单点仿真表结构: [freq, probe1, probe2, ...]
        仅一行数据。

        Args:
            java_model: Java 模型对象
            probe_name: 用于获取表名的探针名称
            probe_start: 探针数据起始列索引（0-based, freq 之后）
            n_probes: 探针数量

        Returns:
            shape (n_probes,) 的复数数组
        """
        data_rows = self._get_table_data(java_model, probe_name)
        row = data_rows[-1]  # 取最后一行（单点仿真仅一行）

        result = np.zeros(n_probes, dtype=np.complex128)
        for j in range(n_probes):
            result[j] = _parse_complex_str(row[probe_start + j])

        logger.debug(f"单点数据提取: {n_probes} 探针, 幅值={np.abs(result)}")
        return result

    # =========================================================================
    # 模型加载
    # =========================================================================

    def _load_model(self, mph_file: Path) -> mph.Model:
        """加载 COMSOL 模型（带缓存）

        根据文件路径缓存模型实例，避免重复加载。

        Args:
            mph_file: COMSOL 模型文件路径

        Returns:
            MPh Model 对象

        Raises:
            RuntimeError: 如果未连接或加载失败
        """
        key = str(mph_file)
        if key in self._models:
            return self._models[key]

        if self.client is None:
            raise RuntimeError("未连接到 COMSOL Server")

        logger.info(f"正在加载模型: {mph_file.name} ...")
        try:
            model = self.client.load(str(mph_file))
            self._models[key] = model
            logger.info(f"模型加载完成: {mph_file.name}")
            return model
        except Exception as e:
            raise RuntimeError(f"模型加载失败 ({mph_file.name}): {e}") from e

    # =========================================================================
    # 传递矩阵计算
    # =========================================================================

    def _compute_eight_transfer_matrix(
        self, f: float,
    ) -> np.ndarray:
        """计算 8 周期模式 8×8 传递矩阵 (speaker→probe)

        执行 8 次单点仿真，每次仅激活一个扬声器 (speaker_amp=1)，
        测量 8 个探针的响应，组装 8×8 传递矩阵。
        设置 cr=1, ci=0, input_amp_l=0, input_amp_r=0。

        Args:
            f: 仿真频率 (Hz)

        Returns:
            shape (8, 8) 的复数传递矩阵，tf[i,j] = probe_j response to speaker_i
        """
        tf_matrix = np.zeros((8, 8), dtype=np.complex128)

        for speaker_idx in range(8):
            speaker_amps = np.zeros(8, dtype=np.complex128)
            speaker_amps[speaker_idx] = 1.0 + 0j

            logger.info(f"  传递矩阵: speaker_{speaker_idx + 1}/8 ...")
            probes = self._run_single_simulation(
                "eight_single", f,
                cr=1.0, cr_min=1.0, cr_max=1.0,
                ci=0.0, ci_min=0.0, ci_max=0.0, res=1,
                input_amp_l="0[Pa]", input_amp_r="0[Pa]",
                speaker_amps_8=speaker_amps,
            )
            tf_matrix[speaker_idx, :] = probes

        logger.info(f"8x8 传递矩阵计算完成，对角元幅值: {np.abs(np.diag(tf_matrix))}")
        return tf_matrix

    # =========================================================================
    # 增益系数批量求解
    # =========================================================================

    @staticmethod
    def _solve_gains_batch(
        probes_target: np.ndarray,
        probes_initial: np.ndarray,
        tf_matrix: np.ndarray,
    ) -> np.ndarray:
        """批量求解增益系数（向量化，避免逐点循环）

        对所有 (cr, ci) 参数组合，使用同一传递矩阵高效求解增益系数。
        利用 LU 分解预计算，对所有右端项一次性求解线性方程组。

        物理模型:
            叠加原理: T = S + M^T @ a_fb
            增益定义: g_i = T_i / (T_i - M[i,i] * a_fb_i)

        Args:
            probes_target: 目标稳态，shape (res, res, 8)
            probes_initial: 初始态，shape (8,)
            tf_matrix: 传递矩阵 M，shape (8, 8)

        Returns:
            增益系数数组，shape (res, res, 8)
        """
        r1, r2, n = probes_target.shape  # (res, res, 8)
        flat_target = probes_target.reshape(-1, n)  # (res*res, 8)

        # 预计算 LU 分解（传递矩阵对所有参数组合相同）
        mt = tf_matrix.T  # (8, 8)
        from numpy.linalg import solve
        # 使用 numpy 的 broadcast solve: (8,8) 矩阵, (8, total) 右端项
        delta = flat_target - probes_initial[np.newaxis, :]  # (total, 8)
        # solve(M^T, delta^T) → (8, total) → 转置为 (total, 8)
        speaker_amps = solve(mt, delta.T).T  # (total, 8)

        # g_i = T_i / (T_i - M[i,i] * a_fb_i)
        tf_diag = np.diag(tf_matrix)  # (8,)
        incident = flat_target - speaker_amps * tf_diag[np.newaxis, :]
        gains_flat = flat_target / incident

        return gains_flat.reshape(r1, r2, n).astype(np.complex128)

    # =========================================================================
    # 真实实验系统稳态批量求解
    # =========================================================================

    @staticmethod
    def _solve_exp_steady_batch(
        gains: np.ndarray,
        exp_tf_matrix: np.ndarray,
        exp_initial_state: np.ndarray,
    ) -> np.ndarray:
        """批量求解真实实验系统的预期稳态（向量化）

        参考 Evolver._simulate_matrix 的物理模型：
            闭环方程: (I - M^T @ diag(β)) @ T = S
            其中 β_i = (g_i - 1) / (g_i * M[i,i])
            S 为静态激励贡献（初态），M 为实验传递矩阵。

        对所有 (cr, ci) 参数组合批量求解，利用 numpy 的 batched solve 一次完成。

        Args:
            gains: 增益系数，shape (res, res, 8)
            exp_tf_matrix: 实验传递矩阵 (speaker→probe)，shape (8, 8)
            exp_initial_state: 实验初态 (S)，shape (8,)

        Returns:
            实验系统稳态，shape (res, res, 8)
        """
        r1, r2, n = gains.shape  # (res, res, 8)
        flat_gains = gains.reshape(-1, n)  # (total, 8)

        tf_diag = np.diag(exp_tf_matrix)  # (8,)
        mt = exp_tf_matrix.T  # (8, 8)

        # β_i = (g_i - 1) / (g_i * M[i,i])，shape (total, 8)
        beta = (flat_gains - 1.0) / (flat_gains * tf_diag[np.newaxis, :])

        # 闭环矩阵: (I - M^T @ diag(β))，对所有参数组合不同
        # mt 按列乘以 beta: loop_gain[k, i, j] = mt[i, j] * beta[k, j]
        loop_gain = mt[np.newaxis, :, :] * beta[:, np.newaxis, :]  # (total, 8, 8)
        closed_loop = np.eye(n)[np.newaxis, :, :] - loop_gain  # (total, 8, 8)

        # 批量求解 (total, 8, 8) @ T = S
        # S 需要 (total, 8, 1) 以匹配 gufunc 签名 (m,m),(m,n)->(m,n)
        S = np.broadcast_to(
            exp_initial_state[np.newaxis, :, np.newaxis],
            (flat_gains.shape[0], n, 1),
        )  # (total, 8, 1)
        T_flat = np.linalg.solve(closed_loop, S).squeeze(-1)  # (total, 8)

        return T_flat.reshape(r1, r2, n).astype(np.complex128)

    # =========================================================================
    # 收敛性批量分析
    # =========================================================================

    @staticmethod
    def _analyze_convergence_batch(
        gains: np.ndarray,
        exp_tf_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """批量计算反馈系统收敛性指标（向量化）

        参考 Evolver._analyze_convergence 的物理模型：
            迭代算子 A = (I - G)(I - D^{-1} M^T)
            其中 G = diag(gain), D = diag(tf_diag), M = tf_feedback_to_ai

        对所有 (cr, ci) 参数组合批量计算：
        1. 未松弛谱半径 ρ(A) = max|λ_i|
        2. 最优松弛因子 α* ∈ (0, 1]，使 ρ((1-α)I + αA) 最小
        3. 松弛后谱半径 ρ(A_{α*})

        Args:
            gains: 增益系数，shape (res, res, 8)
            exp_tf_matrix: 实验传递矩阵，shape (8, 8)

        Returns:
            rho: 未松弛谱半径，shape (res, res)
            alpha_star: 最优松弛因子，shape (res, res)
            rho_relaxed: 松弛后谱半径，shape (res, res)
        """
        r1, r2, n = gains.shape  # (res, res, 8)
        flat_gains = gains.reshape(-1, n)  # (total, 8)
        total = flat_gains.shape[0]

        tf_diag = np.diag(exp_tf_matrix)  # (8,)
        mt = exp_tf_matrix.T  # (8, 8)

        # D^{-1} M^T: 按行除以 d_i
        d_inv_mt = mt / tf_diag[:, None]  # (8, 8)

        # 迭代算子 A_k = (I - diag(g_k)) @ (I - D^{-1} M^T)
        I_n = np.eye(n)
        fixed_part = I_n - d_inv_mt  # (8, 8)，所有参数组合共享
        # 向量化构建 diag(gain) 批量矩阵: (total, 8, 8)
        gain_matrices = I_n - (
            flat_gains[:, :, None] * np.eye(n)[None, :, :]
        )
        A_batch = gain_matrices @ fixed_part[np.newaxis, :, :]  # (total, 8, 8)

        # 批量特征值
        eigenvalues_batch = np.linalg.eigvals(A_batch)  # (total, 8)

        # 未松弛谱半径
        rho_flat = np.max(np.abs(eigenvalues_batch), axis=1)  # (total,)

        # 网格扫描最优松弛因子 α* ∈ (0, 1]
        alpha_grid = np.linspace(1e-3, 1.0, 1001)  # (1001,)
        # μ_i(α) = 1 - α(1 - λ_i)
        mu = (
            1.0
            - alpha_grid[:, None, None]
            * (1.0 - eigenvalues_batch[None, :, :])
        )  # (1001, total, 8)
        rho_curve = np.max(np.abs(mu), axis=2)  # (1001, total)
        best_idx = np.argmin(rho_curve, axis=0)  # (total,)

        alpha_flat = alpha_grid[best_idx]  # (total,)
        rho_relaxed_flat = rho_curve[best_idx, np.arange(total)]  # (total,)

        return (
            rho_flat.reshape(r1, r2),
            alpha_flat.reshape(r1, r2),
            rho_relaxed_flat.reshape(r1, r2),
        )

    # =========================================================================
    # 保存与可视化
    # =========================================================================

    def _save_scan_result(
        self, result: ScanResult, result_folder: str | Path | None = None,
    ) -> Path:
        """将扫描结果保存到磁盘

        Args:
            result: 扫描结果
            result_folder: 自定义保存路径。若为 None 则保存到默认路径
                ``storage/sim/sim_result_scan``。

        Returns:
            实际保存的目录路径
        """
        if result_folder is not None:
            save_dir = Path(result_folder)
        else:
            save_dir = self.storage_dir / "sim_result_scan"
        save_dir.mkdir(parents=True, exist_ok=True)

        # 使用 npz 压缩格式保存
        np.savez_compressed(
            save_dir / "scan_result.npz",
            # 输入参数
            f=result.f,
            cr=result.cr, cr_min=result.cr_min, cr_max=result.cr_max,
            ci=result.ci, ci_min=result.ci_min, ci_max=result.ci_max,
            res=result.res,
            cr_values=result.cr_values, ci_values=result.ci_values,
            # 8 周期模式
            eight_target=result.eight_target,
            eight_initial=result.eight_initial,
            eight_tf_matrix=result.eight_tf_matrix,
            eight_gains=result.eight_gains,
            # Floquet 模式
            floquet_target=result.floquet_target,
            floquet_initial=result.floquet_initial,
            floquet_tf=result.floquet_tf,
            floquet_gains=result.floquet_gains,
            # 真实实验系统稳态
            exp_eight_steady=result.exp_eight_steady,
            exp_floquet_steady=result.exp_floquet_steady,
            # 收敛性分析
            exp_eight_rho=result.exp_eight_rho,
            exp_eight_alpha=result.exp_eight_alpha,
            exp_eight_rho_relaxed=result.exp_eight_rho_relaxed,
            exp_floquet_rho=result.exp_floquet_rho,
            exp_floquet_alpha=result.exp_floquet_alpha,
            exp_floquet_rho_relaxed=result.exp_floquet_rho_relaxed,
        )
        logger.info(f"扫描数据已保存: {save_dir / 'scan_result.npz'}")
        return save_dir

    def _plot_scan_results(self, result: ScanResult, save_dir: Path) -> None:
        """绘制 6 幅参数空间热图

        第 1 幅: 8 周期模式 - 8 个增益系数模长的平均值
        第 2 幅: 8 周期模式 - 最终稳态模长平均值
        第 3 幅: 真实实验系统稳态模长平均值（8 周期增益）+ 收敛性叠加
        第 4 幅: Floquet 模式 - 基于传声器 (point1) 的增益系数模长
        第 5 幅: Floquet 模式 - 基于传声器 (point1) 的最终稳态模长
        第 6 幅: 真实实验系统稳态模长平均值（Floquet 增益）+ 收敛性叠加

        Args:
            result: 扫描结果
            save_dir: 保存目录
        """
        setup_chinese_fonts()
        cr = result.cr_values
        ci = result.ci_values

        # ------------------------------------------------------------------
        # 第 1 幅: 8 周期模式 - 8 个增益系数模长平均值
        # ------------------------------------------------------------------
        eight_gain_mean = np.mean(np.abs(result.eight_gains), axis=2)
        self._plot_discrete_heatmap(
            cr, ci, eight_gain_mean,
            title="8 周期模式 - 增益系数模长平均值",
            save_path=save_dir / "eight_gains_mean.png",
        )

        # ------------------------------------------------------------------
        # 第 2 幅: 8 周期模式 - 最终稳态模长平均值
        # ------------------------------------------------------------------
        eight_target_mean = np.mean(np.abs(result.eight_target), axis=2)
        self._plot_discrete_heatmap(
            cr, ci, eight_target_mean,
            title="8 周期模式 - 最终稳态模长平均值",
            save_path=save_dir / "eight_target_mean.png",
            colorbar_label="稳态模长",
        )

        # ------------------------------------------------------------------
        # 第 3 幅: 真实实验系统稳态模长平均值（8 周期增益）
        # ------------------------------------------------------------------
        exp_eight_mean = np.mean(np.abs(result.exp_eight_steady), axis=2)
        conv_eight = None
        if result.exp_eight_rho is not None:
            conv_eight = {
                "rho": result.exp_eight_rho,
                "rho_relaxed": result.exp_eight_rho_relaxed,
                "cr_values": cr,
                "ci_values": ci,
                "data_for_best": exp_eight_mean,
                "best_label_prefix": "8周期",
            }
        self._plot_discrete_heatmap(
            cr, ci, exp_eight_mean,
            title="真实实验系统 - 稳态模长平均值（8 周期增益）",
            save_path=save_dir / "exp_eight_steady_mean.png",
            colorbar_label="稳态模长",
            convergence_overlay=conv_eight,
        )

        # ------------------------------------------------------------------
        # 第 4 幅: Floquet 模式 - 基于传声器 (point1) 的增益系数模长
        # ------------------------------------------------------------------
        floquet_point1_gain = np.abs(result.floquet_gains[:, :, 2])
        self._plot_discrete_heatmap(
            cr, ci, floquet_point1_gain,
            title="Floquet 模式 - 基于传声器 (point1) 增益系数模长",
            save_path=save_dir / "floquet_gain_point1.png",
        )

        # ------------------------------------------------------------------
        # 第 5 幅: Floquet 模式 - 基于传声器 (point1) 的最终稳态模长
        # ------------------------------------------------------------------
        floquet_point1_target = np.abs(result.floquet_target[:, :, 2])
        self._plot_discrete_heatmap(
            cr, ci, floquet_point1_target,
            title="Floquet 模式 - 基于传声器 (point1) 最终稳态模长",
            save_path=save_dir / "floquet_target_point1.png",
            colorbar_label="稳态模长",
        )

        # ------------------------------------------------------------------
        # 第 6 幅: 真实实验系统稳态模长平均值（Floquet 增益）
        # ------------------------------------------------------------------
        exp_floquet_mean = np.mean(np.abs(result.exp_floquet_steady), axis=2)
        conv_floquet = None
        if result.exp_floquet_rho is not None:
            conv_floquet = {
                "rho": result.exp_floquet_rho,
                "rho_relaxed": result.exp_floquet_rho_relaxed,
                "cr_values": cr,
                "ci_values": ci,
                "data_for_best": exp_floquet_mean,
                "best_label_prefix": "Floquet",
            }
        self._plot_discrete_heatmap(
            cr, ci, exp_floquet_mean,
            title="真实实验系统 - 稳态模长平均值（Floquet 增益）",
            save_path=save_dir / "exp_floquet_steady_mean.png",
            colorbar_label="稳态模长",
            convergence_overlay=conv_floquet,
        )

        logger.info(f"参数空间热图已保存至: {save_dir}")

    @staticmethod
    def _plot_discrete_heatmap(
        cr_values: np.ndarray,
        ci_values: np.ndarray,
        data_2d: np.ndarray,
        title: str,
        save_path: Path,
        dpi: int = 150,
        colorbar_label: str = "增益系数模长",
        convergence_overlay: dict | None = None,
    ) -> Path:
        """绘制离散二维颜色图（不平滑，保持离散色块）

        使用 pcolormesh 绘制正方形网格热图，x 轴为 cr，y 轴为 ci，
        颜色代表数据值。保持离散色块风格，不进行插值平滑。

        可选收敛性叠加层：当 convergence_overlay 不为 None 时，在热图上绘制
        收敛性标记（橘黄色危险区、红色不可达区、最优参数点标记）。

        Args:
            cr_values: cr 轴坐标值，shape (n_cr,)
            ci_values: ci 轴坐标值，shape (n_ci,)
            data_2d: 数据矩阵，shape (n_cr, n_ci)
            title: 图表标题
            save_path: 保存路径
            dpi: 图像分辨率
            colorbar_label: 颜色条标签
            convergence_overlay: 收敛性叠加数据字典，包含:
                - rho: 未松弛谱半径，shape (res, res)
                - rho_relaxed: 松弛后谱半径，shape (res, res)
                - cr_values: cr 坐标数组
                - ci_values: ci 坐标数组
                - data_for_best: 用于选取最优点的 2D 数据
                - best_label_prefix: 最优标签前缀

        Returns:
            保存的文件路径
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))

        # pcolormesh 需要边界坐标 (n+1 个点)
        cr_edges = np.zeros(len(cr_values) + 1)
        ci_edges = np.zeros(len(ci_values) + 1)

        if len(cr_values) > 1:
            d_cr = cr_values[1] - cr_values[0]
            cr_edges[:-1] = cr_values - d_cr / 2
            cr_edges[-1] = cr_values[-1] + d_cr / 2
        else:
            cr_edges = np.array([cr_values[0] - 0.5, cr_values[0] + 0.5])

        if len(ci_values) > 1:
            d_ci = ci_values[1] - ci_values[0]
            ci_edges[:-1] = ci_values - d_ci / 2
            ci_edges[-1] = ci_values[-1] + d_ci / 2
        else:
            ci_edges = np.array([ci_values[0] - 0.5, ci_values[0] + 0.5])

        # data_2d shape (n_cr, n_ci), pcolormesh 需要 (n_ci, n_cr) 或使用 indexing
        im = ax.pcolormesh(
            cr_edges, ci_edges, data_2d.T,
            cmap="viridis", shading="flat",
        )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(colorbar_label)

        # ---- 收敛性叠加层 ----
        if convergence_overlay is not None:
            rho = convergence_overlay["rho"]
            rho_relaxed = convergence_overlay["rho_relaxed"]
            ov_cr = convergence_overlay["cr_values"]
            ov_ci = convergence_overlay["ci_values"]
            data_for_best = convergence_overlay["data_for_best"]
            label_prefix = convergence_overlay["best_label_prefix"]

            # 分类掩码
            safe_mask = rho < 1.0
            danger_mask = (rho >= 1.0) & (rho_relaxed < 1.0)
            unreachable_mask = (rho >= 1.0) & (rho_relaxed >= 1.0)

            # 单元格尺寸
            cell_w = float(cr_edges[1] - cr_edges[0]) if len(cr_edges) > 1 else 1.0
            cell_h = float(ci_edges[1] - ci_edges[0]) if len(ci_edges) > 1 else 1.0

            # 自适应缩放：以 res=10 为基准，线宽/标记随分辨率反比缩放
            n_cells = max(rho.shape[0], rho.shape[1], 1)
            s = max(0.15, min(1.0, 10.0 / n_cells))
            lw_red = max(0.4, 2.0 * s)
            lw_cross = max(0.3, 1.5 * s)
            lw_yellow = max(0.6, 4.0 * s)
            star_size = max(5, 18 * s)
            star_edge = max(0.3, 0.8 * s)
            anno_offset = max(8, int(15 * s))
            anno_font = max(5, int(8 * s))

            def _build_cell_rects(
                mask: np.ndarray,
            ) -> tuple[list[list[tuple]], list[list[tuple]]]:
                """为 mask 中每个 True 单元格生成矩形轮廓线段和对角线。"""
                rects: list[list[tuple]] = []
                crosses: list[list[tuple]] = []
                xs, ys = np.where(mask)
                for i, j in zip(xs, ys, strict=True):
                    x0, y0 = float(cr_edges[i]), float(ci_edges[j])
                    x1, y1 = x0 + cell_w, y0 + cell_h
                    rects.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])
                    crosses.append([(x0, y0), (x1, y1)])
                    crosses.append([(x1, y0), (x0, y1)])
                return rects, crosses

            # ---- 不可达区：红色矩形轮廓线 + 对角红叉（先画，在下层） ----
            unreach_rects, unreach_crosses = _build_cell_rects(unreachable_mask)
            if unreach_rects:
                ax.add_collection(LineCollection(
                    unreach_rects, colors="red",
                    linewidths=lw_red, zorder=3,
                ))
                ax.add_collection(LineCollection(
                    unreach_crosses, colors="red",
                    linewidths=lw_cross, zorder=3,
                ))

            # ---- 危险区：中黄色矩形轮廓线（后画，在上层，线宽加倍） ----
            danger_rects, _ = _build_cell_rects(danger_mask)
            if danger_rects:
                ax.add_collection(LineCollection(
                    danger_rects, colors="#FFD700",
                    linewidths=lw_yellow, zorder=4,
                ))

            # ---- 标记可达到区域中稳态模长均值最大的参数点 ----
            reachable_mask = safe_mask | danger_mask
            if np.any(reachable_mask):
                masked_data = np.where(reachable_mask, data_for_best, -np.inf)
                flat_idx = int(np.argmax(masked_data))
                best_i, best_j = np.unravel_index(flat_idx, masked_data.shape)
                best_cr = float(ov_cr[best_i])
                best_ci = float(ov_ci[best_j])
                best_val = float(data_for_best[best_i, best_j])

                # 绿色五角星标记
                ax.plot(
                    best_cr, best_ci, marker="*", markersize=star_size,
                    color="lime", markeredgecolor="black",
                    markeredgewidth=star_edge,
                    zorder=5,
                )
                # 标签
                label_text = (
                    f"{label_prefix} 最优点\n"
                    f"cr={best_cr:.5f}\nci={best_ci:.5f}\n"
                    f"稳态模长均值={best_val:.4f}"
                )
                ax.annotate(
                    label_text, xy=(best_cr, best_ci),
                    xytext=(anno_offset, anno_offset),
                    textcoords="offset points",
                    fontsize=anno_font, color="black",
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": "white",
                        "edgecolor": "black",
                        "alpha": 0.85,
                    },
                    arrowprops={"arrowstyle": "->", "color": "black"},
                    zorder=6,
                )

            # ---- 图例（纯线条风格，线宽与绘图区一致） ----
            legend_handles = []
            if np.any(safe_mask):
                legend_handles.append(
                    Line2D(
                        [0], [0], color="green", linewidth=2.0,
                        label=f"安全区 (\u03c1<1, {int(np.sum(safe_mask))}点)",
                    )
                )
            if np.any(danger_mask):
                legend_handles.append(
                    Line2D(
                        [0], [0], color="#FFD700",
                        linewidth=max(1.5, lw_yellow),
                        label=f"危险区 (\u03c1\u22651, \u03b1*可恢复, "
                              f"{int(np.sum(danger_mask))}点)",
                    )
                )
            if np.any(unreachable_mask):
                legend_handles.append(
                    Line2D(
                        [0], [0], color="red",
                        linewidth=max(1.5, lw_red),
                        label=f"不可达区 (\u03c1\u22651, \u03b1*不可恢复, "
                              f"{int(np.sum(unreachable_mask))}点)",
                    )
                )
            if legend_handles:
                ax.legend(
                    handles=legend_handles,
                    loc="upper right", fontsize=8,
                    framealpha=0.9,
                )

        ax.set_xlabel("cr", fontsize=12)
        ax.set_ylabel("ci", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        logger.debug(f"离散热图已保存: {save_path}")
        return save_path


# =============================================================================
# 模块级辅助函数
# =============================================================================

def _format_complex(value: complex, unit: str = "") -> str:
    """将复数值格式化为 COMSOL 参数字符串"""
    if value.imag == 0:
        return f"{value.real}{unit}"
    else:
        return f"({value.real}+({value.imag})*i){unit}"


def _parse_complex_str(s: str) -> complex:
    """解析 COMSOL 的复数字符串格式"""
    s = str(s).strip()
    s = s.replace("i", "j")
    return complex(s)
