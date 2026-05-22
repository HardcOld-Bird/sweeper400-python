"""
# Evolver 收敛性调试脚本

针对一个具体困惑：使用 fishnet_r/tf_data.pkl 时迭代和矩阵解吻合，
但更换为 fishnet_L/tf_data.pkl 后迭代发散，矩阵解则给出更大的稳态值。

本脚本**不依赖任何硬件**，只读取 TFData 文件并做纯数值分析：

1. 加载两组 TFData，重建与 Evolver 内部完全一致的子矩阵
   (tf_static, tf_feedback, tf_diag)。
2. 用矩阵法直接解出理论稳态 T*、a*；
3. 推导迭代算子 A = (I - G)(I - D^{-1} M^T)，计算其特征值和谱半径
   ρ(A)（>1 即必然发散）；
4. 用纯 numpy 重放 simulate 的解析迭代过程，打印每周期的复振幅模长
   并与稳态对比，直观看到 r 收敛 / L 发散；
5. 最后打印各组 TFData 的"对角占优度"指标，解释根因。

用法（在仓库根目录下）：

    python tests/sweeper400/use/debug_evolver_convergence.py

或在 PyCharm/IDEA 里直接运行此文件即可。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sweeper400.analyze import TFData, load_data_with_fallback


# =============================================================================
#  与 scripts/8演化测量.py 完全一致的配置
# =============================================================================

AI_CHANNELS = (
    "PXI1Slot3/ai0",
    "PXI1Slot3/ai1",
    "PXI1Slot4/ai0",
    "PXI1Slot4/ai1",
    "PXI1Slot5/ai0",
    "PXI1Slot5/ai1",
    "PXI1Slot6/ai0",
    "PXI1Slot6/ai1",
)
AO_CHANNELS_STATIC = ("PXI1Slot2/ao0",)
AO_CHANNELS_FEEDBACK = (
    "PXI1Slot3/ao0",
    "PXI1Slot3/ao1",
    "PXI1Slot4/ao0",
    "PXI1Slot4/ao1",
    "PXI1Slot5/ao0",
    "PXI1Slot5/ao1",
    "PXI1Slot6/ao0",
    "PXI1Slot6/ao1",
)

STATIC_AMP = 0.005 + 0j  # 与脚本里的 cca 一致
GAIN_COEFFICIENTS = np.array(
    [
        -0.308988 - 2.557868j,
        -1.196245 - 1.412161j,
        -1.541522 - 1.275200j,
        -1.518875 - 1.254205j,
        -1.534567 - 1.262961j,
        -1.532444 - 1.264957j,
        -1.499747 - 1.267534j,
        -1.703700 - 1.289199j,
    ],
    dtype=np.complex128,
)

CYCLES_NUM = 10

CASES = {
    "r_INPUT (收敛)": "D:\\EveryoneDownloaded\\fishnet_r\\tf_data.pkl",
    "L_INPUT (发散)": "D:\\EveryoneDownloaded\\fishnet_L\\tf_data.pkl",
}


# =============================================================================
#  核心分析函数
# =============================================================================

def build_matrices(tf_data: TFData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从 TFData 中按 Evolver 的索引顺序抽取子矩阵。

    Returns:
        tf_static: (n_static, n_ai)
        tf_feedback: (n_fb, n_ai)
        tf_diag: (n_fb,)，等于 tf_feedback[i, i]
    """
    df = tf_data["tf_dataframe"]
    tf_static = np.asarray(
        df.loc[list(AO_CHANNELS_STATIC), list(AI_CHANNELS)].values,
        dtype=np.complex128,
    )
    tf_feedback = np.asarray(
        df.loc[list(AO_CHANNELS_FEEDBACK), list(AI_CHANNELS)].values,
        dtype=np.complex128,
    )
    tf_diag = np.diag(tf_feedback).astype(np.complex128).copy()
    return tf_static, tf_feedback, tf_diag


def matrix_solution(
    tf_static: np.ndarray,
    tf_feedback: np.ndarray,
    tf_diag: np.ndarray,
    gain: np.ndarray,
    static_amps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """一次性解出理论稳态：(I - M^T diag(beta)) T = S。"""
    n = len(tf_diag)
    beta = (gain - 1.0) / (gain * tf_diag)             # (n,)
    static_contribution = static_amps @ tf_static      # (n,)
    closed_loop = np.eye(n) - tf_feedback.T * beta     # (n,n)
    T_star = np.linalg.solve(closed_loop, static_contribution)
    a_star = beta * T_star
    return T_star, a_star


def iteration_operator(
    tf_feedback: np.ndarray,
    tf_diag: np.ndarray,
    gain: np.ndarray,
) -> np.ndarray:
    """
    迭代律推导：a_new = (I-G)(I - D^{-1} M^T) a_old + (G-I) D^{-1} S

    返回不动点迭代算子 A = (I - G)(I - D^{-1} M^T)。
    """
    n = len(tf_diag)
    G = np.diag(gain)
    # (D^{-1} M^T)_{ij} = M^T_{ij} / d_i = M_{ji}/d_i —— 即 M^T 按行除以 d_i
    # 该矩阵的对角元恒为 1 (M_{ii}/d_i = d_i/d_i = 1)，因此 (I - D^{-1} M^T) 对角元为 0
    D_inv_MT = tf_feedback.T / tf_diag[:, None]
    A = (np.eye(n) - G) @ (np.eye(n) - D_inv_MT)
    return A


def analytical_iterate(
    tf_static: np.ndarray,
    tf_feedback: np.ndarray,
    tf_diag: np.ndarray,
    gain: np.ndarray,
    static_amps: np.ndarray,
    cycles_num: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    精确还原 Evolver._feedback_method(mode="analytical") 的迭代逻辑。
    返回每周期的 (T_history, a_history)。
    """
    n_fb = len(tf_diag)
    a_old = np.zeros(n_fb, dtype=np.complex128)
    T_history: list[np.ndarray] = []
    a_history: list[np.ndarray] = []

    for _ in range(cycles_num):
        # 解析模式下的 T
        T = static_amps @ tf_static + a_old @ tf_feedback   # (n_ai,)
        # 入射、目标、新 ao（与 _feedback_method 完全一致）
        incident = T - a_old * tf_diag
        target_total = incident * gain
        delta_ai = target_total - T
        delta_ao = delta_ai / tf_diag
        a_new = a_old + delta_ao

        T_history.append(T.copy())
        a_history.append(a_new.copy())

        a_old = a_new
    return T_history, a_history


# =============================================================================
#  打印辅助
# =============================================================================

def print_section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_subsection(title: str) -> None:
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


def diag_dominance_metric(M: np.ndarray) -> tuple[float, np.ndarray]:
    """计算每行 |M_ii| / sum_{j!=i} |M_ij| 的"对角占优度"。"""
    n = M.shape[0]
    abs_M = np.abs(M)
    diag = np.diag(abs_M)
    off_sum = abs_M.sum(axis=1) - diag
    # 避免除零
    ratios = diag / np.where(off_sum > 0, off_sum, np.nan)
    return float(np.nanmin(ratios)), ratios


# =============================================================================
#  主流程
# =============================================================================

def analyze_one_case(label: str, tf_path: str) -> None:
    print_section(f"CASE: {label}\n  path = {tf_path}")

    if not Path(tf_path).exists():
        print(f"!! 文件不存在，跳过：{tf_path}")
        return

    tf_data = load_data_with_fallback(
        explicit_path=tf_path,
        default_path="storage/calib/calib_result_fishnet/tf_data.pkl",
        data_type="Fishnet TFData",
    )
    if tf_data is None:
        print("!! TFData 加载失败")
        return

    tf_static, tf_feedback, tf_diag = build_matrices(tf_data)
    static_amps = np.full(len(AO_CHANNELS_STATIC), STATIC_AMP, dtype=np.complex128)

    # ---- 1. TF 子矩阵尺度 ----
    print_subsection("(1) tf_feedback 子矩阵概览（与 ao_channels_feedback × ai_channels 对齐）")
    np.set_printoptions(precision=3, suppress=False, linewidth=200)
    print("|tf_feedback| (8×8 模长)：")
    print(np.abs(tf_feedback))
    print("\n|tf_diag| =", np.abs(tf_diag))
    min_ratio, all_ratios = diag_dominance_metric(tf_feedback)
    print(
        "\n各行对角占优比 |M_ii| / sum_{j!=i}|M_ij|："
    )
    for i, r in enumerate(all_ratios):
        flag = "  <-- 不占优" if r < 1.0 else ""
        print(f"  ch{i}: {r:.4f}{flag}")
    print(f"最小占优比 = {min_ratio:.4f}（<1 表示该行非对角串扰超过自身对角）")

    # ---- 2. 矩阵法稳态 ----
    print_subsection("(2) 矩阵法稳态解（精确）")
    T_star, a_star = matrix_solution(
        tf_static, tf_feedback, tf_diag, GAIN_COEFFICIENTS, static_amps
    )
    print("|T*|       =", np.abs(T_star))
    print("|a_fb*|    =", np.abs(a_star))

    # ---- 3. 迭代算子谱半径 ----
    print_subsection("(3) 迭代算子 A = (I-G)(I - D^{-1} M^T) 的谱分析")
    A = iteration_operator(tf_feedback, tf_diag, GAIN_COEFFICIENTS)
    eigvals = np.linalg.eigvals(A)
    abs_eig = np.abs(eigvals)
    rho = float(abs_eig.max())
    print("特征值（按模长降序）：")
    for ev in sorted(eigvals, key=lambda z: -abs(z)):
        print(f"  {ev.real:+.4f}{ev.imag:+.4f}j   |·|={abs(ev):.4f}")
    print(f"\n谱半径 ρ(A) = {rho:.4f}")
    if rho < 1:
        print(f"  -> ρ<1，迭代必收敛（理论上误差按 ρ^k = {rho:.3f}^k 衰减）")
    else:
        print(f"  -> ρ≥1，迭代必发散（每周期最大模长按 {rho:.3f} 倍放大）")

    # ---- 4. 解析迭代轨迹 ----
    print_subsection("(4) 解析迭代轨迹（与 simulate(mode='analytical') 等价）")
    T_hist, a_hist = analytical_iterate(
        tf_static, tf_feedback, tf_diag,
        GAIN_COEFFICIENTS, static_amps, CYCLES_NUM,
    )
    print(f"{'周期':>4} | {'max|T - T*|':>14} | {'max|a - a*|':>14} | {'max|a|':>10}")
    print("-" * 64)
    for k in range(CYCLES_NUM):
        err_T = np.max(np.abs(T_hist[k] - T_star))
        err_a = np.max(np.abs(a_hist[k] - a_star))
        max_a = np.max(np.abs(a_hist[k]))
        print(f"{k+1:>4} | {err_T:>14.4e} | {err_a:>14.4e} | {max_a:>10.4f}")

    # ---- 5. 与谱半径预测对比 ----
    print_subsection("(5) 误差衰减比 vs 理论 ρ(A)")
    print(f"理论每步衰减/放大率 ρ(A) = {rho:.4f}")
    if len(a_hist) >= 3:
        ratios_step = []
        for k in range(1, CYCLES_NUM):
            prev = np.linalg.norm(a_hist[k - 1] - a_star)
            curr = np.linalg.norm(a_hist[k] - a_star)
            if prev > 0:
                ratios_step.append(curr / prev)
        print(
            "实测 ||a_k - a*|| / ||a_{k-1} - a*||（应稳定地接近 ρ(A)）："
        )
        for i, r in enumerate(ratios_step, start=2):
            print(f"  k={i}: {r:.4f}")


def main() -> None:
    print_section("Evolver 收敛性诊断脚本")
    print(
        "本脚本只做纯数值分析，不连接硬件。\n"
        "目的是验证：迭代发散与否完全由 TF 矩阵串扰特性 + 增益系数决定的"
        "迭代算子谱半径 ρ(A) 控制。"
    )

    for label, path in CASES.items():
        analyze_one_case(label, path)

    print_section("结论速览")
    print(
        "若你看到 r_INPUT 的 ρ(A) < 1 且迭代误差按 ρ^k 衰减，\n"
        "而 L_INPUT 的 ρ(A) > 1 且迭代误差按 ρ^k 放大，\n"
        "即可确认：两种 TF 数据下，矩阵法都给出了正确的稳态解；\n"
        "但 evolver 当前的迭代律是无阻尼的不动点迭代，\n"
        "其稳定域受 TF 串扰强度和增益激进程度的双重限制——\n"
        "L_INPUT 数据下迭代算子谱半径已经超过 1，所以发散。\n"
        "\n"
        "解决思路（不在本脚本内实现，仅供参考）：\n"
        "  a) 引入松弛因子 α∈(0,1]: a_new = a_old + α·delta_ao；\n"
        "  b) 减小 |g_i - 1|（即让增益系数更接近 1，降低迭代激进度）；\n"
        "  c) 在 evolve() 之前用矩阵解作为初始化（即在第 0 步直接把 a_fb 设为 a_star，\n"
        "     再做小幅迭代修正硬件偏差），可以避免大幅震荡。\n"
    )


if __name__ == "__main__":
    main()
