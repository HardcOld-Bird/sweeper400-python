"""数值仿真模块

此模块提供基于传递函数矩阵的数值仿真功能，用于快速模拟反馈控制系统的行为。
"""

import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..logger import get_logger

# 获取模块日志器
logger = get_logger(__name__)


class Simulator:
    """基于传递函数矩阵的数值仿真器

    此类是GainEPSimulator的超级精简版和超级高效版。核心不再是COMSOL仿真计算，
    而是单纯的Python数值计算，运算速度将会高得多。

    ## 工作原理：
        1. 假定CaliberFishNet的校准工作已经完成，并且相关校准文件（传递函数矩阵）
           已经保存在storage目录中
        2. 传递函数矩阵应该是9×8的形状（9个AO通道 → 8个AI通道）
        3. 第一个AO通道作为主激励通道（原本的"pamp"），剩余8个AO通道作为反馈通道
        4. 8个AI通道对应传声器测得的声压信号
        5. 通过传递函数矩阵进行数值计算，模拟反馈控制系统的行为

    ## 主要特性：
        - 加载并验证校准文件中的传递函数矩阵
        - 执行指定次数的反馈迭代循环
        - 使用传递函数矩阵进行数值计算（替代COMSOL仿真）
        - 绘制AI信号在复平面上的演化轨迹
        - 自动保存仿真结果图像

    Attributes:
        tf_data_path: 传递函数数据文件路径
        tf_matrix: 传递函数矩阵，形状为(9, 8)的复数数组
        primary_ao_index: 主激励AO通道在矩阵中的索引（0-8）
        gain_coefficients: 8个管槽的增益系数（复数值）
        ao_channel_names: 9个AO通道的名称列表
        ai_channel_names: 8个AI通道的名称列表
        storage_dir: 仿真结果存储目录

    Examples:
        >>> # 使用默认路径和默认主激励通道初始化仿真器
        >>> gain_coeffs = [1.0+0j] * 8  # 8个管槽的增益系数
        >>> simulator = Simulator(gain_coefficients=gain_coeffs)
        >>>
        >>> # 或者指定自定义路径和通道
        >>> from pathlib import Path
        >>> tf_file = Path("storage/calib/calib_result_fishnet/tf_data.pkl")
        >>> simulator = Simulator(
        ...     gain_coefficients=gain_coeffs,
        ...     tf_data_path=tf_file,
        ...     primary_ao_channel="PXI1Slot3/ao0"
        ... )
        >>>
        >>> # 运行仿真
        >>> ao_history, ai_history = simulator.run_simulation(
        ...     num_iterations=10,
        ...     primary_ao_amplitude=1.0
        ... )
    """

    def __init__(
        self,
        gain_coefficients: list[complex],
        tf_data_path: str | Path | None = None,
        primary_ao_channel: str = "PXI1Slot2/ao0",
    ) -> None:
        """初始化仿真器

        Args:
            gain_coefficients: 8个管槽的增益系数（复数值列表）
            tf_data_path: 传递函数数据文件路径（CaliberFishNet的校准结果）。
                如果为None，则自动使用默认路径：
                "storage/calib/calib_result_fishnet/tf_data.pkl"
            primary_ao_channel: 主激励AO通道名称，默认为"PXI1Slot2/ao0"

        Raises:
            FileNotFoundError: 如果传递函数文件不存在
            ValueError: 如果传递函数矩阵形状不是9×8，或者增益系数数量不是8
            KeyError: 如果指定的主激励通道不在AO通道列表中
        """
        logger.info("=" * 70)
        logger.info("初始化Simulator仿真器")
        logger.info("=" * 70)

        # 处理tf_data_path参数
        if tf_data_path is None:
            # 使用默认路径
            # 获取当前文件所在的sim目录，向上3级到达工作区根目录
            current_file = Path(__file__)  # simulator.py
            workspace_root = current_file.parent.parent.parent.parent  # 到达工作区根目录
            default_path = (
                workspace_root / "storage" / "calib" / "calib_result_fishnet" / "tf_data.pkl"
            )
            self.tf_data_path = default_path
            logger.info(f"使用默认传递函数文件路径: {self.tf_data_path}")
        else:
            # 使用用户提供的路径
            self.tf_data_path = Path(tf_data_path)
            logger.info(f"使用用户指定的传递函数文件路径: {self.tf_data_path}")

        # 验证文件存在性
        if not self.tf_data_path.exists():
            logger.error(f"传递函数文件不存在: {self.tf_data_path}")
            raise FileNotFoundError(f"传递函数文件不存在: {self.tf_data_path}")

        # 验证增益系数数量
        if len(gain_coefficients) != 8:
            logger.error(f"增益系数数量必须是8，当前为{len(gain_coefficients)}")
            raise ValueError(f"增益系数数量必须是8，当前为{len(gain_coefficients)}")

        # 加载传递函数数据
        logger.info(f"加载传递函数数据: {self.tf_data_path}")
        try:
            with open(self.tf_data_path, "rb") as f:
                tf_data = pickle.load(f)
        except Exception as e:
            logger.error(f"加载传递函数文件失败: {e}", exc_info=True)
            raise RuntimeError(f"加载传递函数文件失败: {e}") from e

        # 提取DataFrame
        tf_df: pd.DataFrame = tf_data["tf_dataframe"]
        logger.info(f"传递函数DataFrame形状: {tf_df.shape}")
        logger.info(f"AO通道（行索引）: {list(tf_df.index)}")
        logger.info(f"AI通道（列索引）: {list(tf_df.columns)}")

        # 验证形状是否为9×8
        if tf_df.shape != (9, 8):
            logger.error(
                f"传递函数矩阵形状必须是(9, 8)，当前为{tf_df.shape}。"
                f"请确保使用CaliberFishNet校准了9个AO通道和8个AI通道。"
            )
            raise ValueError(
                f"传递函数矩阵形状必须是(9, 8)，当前为{tf_df.shape}"
            )

        # 提取通道名称
        self.ao_channel_names = list(tf_df.index)
        self.ai_channel_names = list(tf_df.columns)

        # 验证主激励通道是否存在
        if primary_ao_channel not in self.ao_channel_names:
            logger.error(
                f"指定的主激励通道'{primary_ao_channel}'不在AO通道列表中。"
                f"可用的AO通道: {self.ao_channel_names}"
            )
            raise KeyError(
                f"指定的主激励通道'{primary_ao_channel}'不在AO通道列表中"
            )

        # 记录主激励通道索引
        self.primary_ao_index = self.ao_channel_names.index(primary_ao_channel)
        logger.info(
            f"主激励通道: {primary_ao_channel} (索引: {self.primary_ao_index})"
        )

        # 转换DataFrame为NumPy数组
        self.tf_matrix = tf_df.values  # 形状: (9, 8)
        logger.info(f"传递函数矩阵已转换为NumPy数组，形状: {self.tf_matrix.shape}")

        # 存储增益系数
        self.gain_coefficients = np.array(gain_coefficients, dtype=complex)
        logger.info(f"增益系数: {self.gain_coefficients}")

        # 设置存储路径
        # 获取工作区根目录（tf_data_path在storage/calib/xxx/下）
        self.storage_dir = self.tf_data_path.parent.parent.parent.parent / "storage" / "sim"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"仿真结果存储目录: {self.storage_dir}")

        logger.info("✓ Simulator仿真器初始化完成")
        logger.info("=" * 70)

    def _simulate_ai_signals(self, ao_amplitudes: np.ndarray) -> np.ndarray:
        """根据AO幅值计算AI信号

        此方法使用传递函数矩阵进行数值计算，模拟物理系统的行为。

        工作原理：
            对于每个AI通道i，其信号等于所有AO通道对它的贡献之和：
            AI[i] = sum(AO[j] * TF[j, i] for j in range(9))

        这等价于矩阵乘法：
            AI_vector = TF_matrix^T @ AO_vector

        Args:
            ao_amplitudes: 9个AO通道的复数幅值，形状为(9,)

        Returns:
            8个AI通道的复数信号，形状为(8,)

        Raises:
            ValueError: 如果ao_amplitudes的形状不是(9,)
        """
        # 验证输入形状
        if ao_amplitudes.shape != (9,):
            raise ValueError(
                f"ao_amplitudes的形状必须是(9,)，当前为{ao_amplitudes.shape}"
            )

        # 使用传递函数矩阵计算AI信号
        # tf_matrix的形状是(9, 8)，表示9个AO → 8个AI
        # 因此 AI = tf_matrix^T @ AO，结果形状为(8,)
        ai_signals = self.tf_matrix.T @ ao_amplitudes

        return ai_signals

    def _compute_next_ao(
        self,
        current_ao_amplitudes: np.ndarray,
        current_ai_signals: np.ndarray,
    ) -> np.ndarray:
        """计算下一时刻的AO幅值

        此方法实现反馈控制逻辑，参考GainEPSimulator的_generate_feedback方法
        （logic_mode="p_and_d"）。

        工作原理：
            1. 主激励AO通道保持不变
            2. 对于每个反馈AO通道i（对应管槽i）：
               - 当前总声压 = current_ai_signals[i]
               - 当前反馈AO幅值 = current_ao_amplitudes[i+1]（注意索引偏移）
               - 传递函数 = tf_matrix[i+1, i]（该反馈AO对该AI的传递函数）
               - 入射声压 = 总声压 - 当前反馈AO幅值 × 传递函数
               - 目标总声压 = 入射声压 × 增益系数[i]
               - 声压差 = 目标总声压 - 当前总声压
               - AO增量 = 声压差 / 传递函数
               - 新反馈AO幅值 = 当前反馈AO幅值 + AO增量

        Args:
            current_ao_amplitudes: 当前9个AO通道的复数幅值，形状为(9,)
            current_ai_signals: 当前8个AI通道的复数信号，形状为(8,)

        Returns:
            下一时刻的9个AO通道的复数幅值，形状为(9,)

        Raises:
            ValueError: 如果输入形状不正确
        """
        # 验证输入形状
        if current_ao_amplitudes.shape != (9,):
            raise ValueError(
                f"current_ao_amplitudes的形状必须是(9,)，"
                f"当前为{current_ao_amplitudes.shape}"
            )
        if current_ai_signals.shape != (8,):
            raise ValueError(
                f"current_ai_signals的形状必须是(8,)，"
                f"当前为{current_ai_signals.shape}"
            )

        # 初始化新的AO幅值（复制当前值）
        new_ao_amplitudes = current_ao_amplitudes.copy()

        # 主激励AO通道保持不变（索引为self.primary_ao_index）
        # 只更新其余8个反馈AO通道

        # 获取除主激励通道外的8个反馈AO通道的索引
        feedback_ao_indices = [i for i in range(9) if i != self.primary_ao_index]

        # 对于每个反馈AO通道，计算新的幅值
        for local_idx, ao_idx in enumerate(feedback_ao_indices):
            # local_idx: 0-7（对应8个管槽）
            # ao_idx: 在9个AO中的实际索引（0-8，但跳过primary_ao_index）

            # 当前总声压（AI信号）
            total_pressure = current_ai_signals[local_idx]

            # 当前反馈AO幅值
            current_ao_value = current_ao_amplitudes[ao_idx]

            # 从传递函数矩阵中提取该反馈AO对该AI的传递函数
            # tf_matrix的行索引是AO索引，列索引是AI索引
            transfer_func = self.tf_matrix[ao_idx, local_idx]

            # 该管槽的增益系数
            gain_coeff = self.gain_coefficients[local_idx]

            # 计算入射声压（扣除反馈AO自身的贡献）
            incident_pressure = total_pressure - current_ao_value * transfer_func

            # 计算目标总声压（使用增益系数）
            target_total_pressure = incident_pressure * gain_coeff

            # 计算声压差
            pressure_diff = target_total_pressure - total_pressure

            # 计算AO增量
            ao_increment = pressure_diff / transfer_func

            # 计算新的AO幅值（增量调整）
            new_ao_amplitudes[ao_idx] = current_ao_value + ao_increment

        return new_ao_amplitudes

    def run_simulation(
        self,
        num_iterations: int,
        primary_ao_amplitude: complex | float = 1.0,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """运行反馈迭代仿真

        此方法执行指定次数的反馈迭代循环，模拟反馈控制系统的行为。

        工作流程：
            1. 初始化：主激励AO幅值设为primary_ao_amplitude，其余AO为0
            2. 第一次迭代：使用初始AO幅值计算AI信号
            3. 后续迭代：
               - 根据当前AI信号和增益系数计算新的AO幅值
               - 使用新的AO幅值计算新的AI信号
            4. 记录所有迭代的AO幅值和AI信号
            5. 绘制AI信号在复平面上的演化轨迹

        Args:
            num_iterations: 迭代次数（必须≥1）
            primary_ao_amplitude: 主激励AO通道的幅值（复数或实数），默认为1.0

        Returns:
            tuple: (ao_history, ai_history)
                - ao_history: 所有迭代的AO幅值列表，每个元素形状为(9,)
                - ai_history: 所有迭代的AI信号列表，每个元素形状为(8,)

        Raises:
            ValueError: 如果num_iterations < 1
        """
        # 验证参数
        if num_iterations < 1:
            raise ValueError(f"num_iterations必须≥1，当前值: {num_iterations}")

        logger.info("\n" + "=" * 70)
        logger.info("开始反馈迭代仿真")
        logger.info("=" * 70)
        logger.info(f"迭代次数: {num_iterations}")
        logger.info(f"主激励AO通道: {self.ao_channel_names[self.primary_ao_index]}")
        logger.info(f"主激励AO幅值: {primary_ao_amplitude}")
        logger.info(f"增益系数: {self.gain_coefficients}")

        # 初始化存储列表
        ao_history: list[np.ndarray] = []
        ai_history: list[np.ndarray] = []

        # 初始化AO幅值：主激励通道为primary_ao_amplitude，其余为0
        current_ao = np.zeros(9, dtype=complex)
        current_ao[self.primary_ao_index] = complex(primary_ao_amplitude)

        # 执行迭代循环
        for iteration in range(num_iterations):
            logger.info(f"\n--- 迭代 {iteration + 1}/{num_iterations} ---")

            # 计算当前AI信号
            current_ai = self._simulate_ai_signals(current_ao)
            logger.info(f"AI信号范数: {np.linalg.norm(current_ai):.6e}")

            # 保存当前迭代的数据
            ao_history.append(current_ao.copy())
            ai_history.append(current_ai.copy())

            # 如果不是最后一次迭代，计算下一次的AO幅值
            if iteration < num_iterations - 1:
                current_ao = self._compute_next_ao(current_ao, current_ai)
                logger.info(f"新AO幅值范数: {np.linalg.norm(current_ao):.6e}")

        logger.info("\n" + "=" * 70)
        logger.info("反馈迭代仿真完成")
        logger.info("=" * 70)

        # 绘制AI信号演化轨迹
        self._plot_ai_evolution(ai_history, num_iterations)

        return ao_history, ai_history

    def _plot_ai_evolution(
        self,
        ai_history: list[np.ndarray],
        num_iterations: int,
    ) -> None:
        """在复平面上绘制AI信号演化轨迹并保存

        在复平面上绘制8条折线，第n条线代表第n个AI通道（传声器）的声压信号
        在迭代过程中的演化轨迹。横轴是实部，纵轴是虚部。

        图像会自动保存到storage/sim/目录。

        Args:
            ai_history: 所有迭代的AI信号列表，每个元素形状为(8,)
            num_iterations: 迭代次数
        """
        logger.info("\n开始绘制AI信号演化轨迹...")

        # 转换为NumPy数组以便处理
        ai_array = np.array(ai_history)  # shape: (num_iterations, 8)

        # 创建图形 - 使用复平面绘图
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # 定义颜色映射
        colors = plt.cm.tab10(np.linspace(0, 1, 8))

        # 绘制8条轨迹线
        for i in range(8):
            # 提取该AI通道在所有迭代中的复数值
            trajectory = ai_array[:, i]
            real_parts = trajectory.real
            imag_parts = trajectory.imag

            # 先绘制连接线（不带标记）
            ax.plot(
                real_parts,
                imag_parts,
                color=colors[i],
                linewidth=2,
                label=f"AI通道{i + 1}",
                alpha=0.6,
                zorder=1,
            )

            # 绘制所有中间迭代点（小圆点）
            if num_iterations > 2:
                ax.scatter(
                    real_parts[1:-1],
                    imag_parts[1:-1],
                    color=colors[i],
                    s=30,  # 点的大小
                    alpha=0.5,
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=2,
                )

            # 标记起点（第一次迭代）- 方形
            ax.scatter(
                real_parts[0],
                imag_parts[0],
                marker="s",
                s=150,  # 更大的起点
                color=colors[i],
                edgecolors="black",
                linewidths=2,
                alpha=0.9,
                zorder=3,
            )

            # 标记终点（最后一次迭代）- 星形
            ax.scatter(
                real_parts[-1],
                imag_parts[-1],
                marker="*",
                s=300,  # 更大的终点
                color=colors[i],
                edgecolors="black",
                linewidths=2,
                alpha=0.9,
                zorder=3,
            )

            # 在起点和终点标注文字（仅对第四条线标注，避免重复）
            if i == 3:
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

        # 绘制原点参考
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

        # 设置坐标轴标签和标题
        ax.set_xlabel("实部", fontsize=12)
        ax.set_ylabel("虚部", fontsize=12)

        ax.set_title(
            f"数值仿真 - AI信号在复平面上的演化轨迹\n"
            f"(迭代次数: {num_iterations})",
            fontsize=14,
            fontweight="bold",
        )

        # 设置图例
        ax.legend(loc="best", fontsize=9, framealpha=0.9)

        # 设置网格
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)

        # 设置相等的纵横比，使复平面不失真
        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()

        # 生成文件名并保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ai_evolution_iter{num_iterations}_{timestamp}.png"
        save_path = self.storage_dir / filename

        try:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"✓ 图像已保存到: {save_path}")
        except Exception as e:
            logger.error(f"⚠ 保存图像失败: {e}", exc_info=True)

        plt.close(fig)

        logger.info("✓ 已完成AI信号演化轨迹绘制")
