# pyright: basic
"""
# 数据可视化模块

模块路径：`sweeper400.analyze.plot`

包含对采集数据进行可视化处理的函数和类。
"""

import math
from pathlib import Path
from typing import Literal, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle
from scipy.interpolate import griddata

from ..config import setup_chinese_fonts
from ..logger import get_logger
from .basic_sine import extract_single_tone_information_vvi
from .filter import filter_sweep_data
from .my_dtypes import Point2D, PositiveFloat, SweepData, Waveform
from .post_process import average_sweep_data, load_compressed_data

# 配置matplotlib中文字体支持
setup_chinese_fonts()

# 获取模块日志器
logger = get_logger(__name__)


# 本地定义的空间点传递函数数据类型，专用于空间扫场绘图函数。
class PointTFData(TypedDict):
    """
    空间点传递函数数据格式（仅供 plot.py 内部绘图函数使用）。

    ## 内部组成:
        **position**: 测量点的二维空间坐标（mm）
        **complex_amplitude**: 复数传递函数，幅值代表幅值比，相位代表相位差（弧度制）
    """

    position: Point2D
    complex_amplitude: complex


def sweep_data_to_point_tf_data_list(
    sweep_data: SweepData,
    lowcut: float = 100.0,
    highcut: float = 20000.0,
    filter_order: int = 4,
    trim_samples: int = 0,
) -> list[PointTFData]:
    """
    将SweepData转换为PointTFData列表

    该函数对SweepData进行后处理，提取每个测量点的单频信息，
    转换为用于绘图的PointTFData格式。
    处理流程：
    1. 使用average_sweep_data对每个点的多个波形进行平均
    2. 使用filter_sweep_data对平均后的数据进行滤波
    3. 对每个点的平均波形使用extract_single_tone_information_vvi提取单频信息
    4. 计算相对于参考信号的复数传递函数

    参考信号信息直接从 sweep_data["ao_data"] 的 frequency 和
    channel_complex_amplitudes 属性获取。

    Args:
        sweep_data: 扫场测量数据，包含单通道AI波形
        lowcut: 带通滤波器低截止频率（Hz），默认100.0
        highcut: 带通滤波器高截止频率（Hz），默认20000.0
        filter_order: 滤波器阶数，默认4
        trim_samples: 滤波后切除波形开头的采样点数量，默认0

    Returns:
        list[PointTFData]: 每个测量点的传递函数数据列表

    Raises:
        ValueError: 当输入数据为空或无法获取参考信号参数时

    Examples:
        ```python
        >>> # 假设已有SweepData
        >>> from analyze import load_compressed_data
        >>> test_sweep_data = load_compressed_data("measurement.pkl")
        >>> plot_tf_results = sweep_data_to_point_tf_data_list(test_sweep_data)
        >>> # 现在可以使用绘图函数
        >>> fig, axes = plot_point_tf_data_list(plot_tf_results, mode="discrete")
        ```
    """
    # 获取函数日志器
    f_logger = get_logger(f"{__name__}.sweep_data_to_point_tf_data_list")

    f_logger.info("开始将SweepData转换为PointTFData列表")

    # 验证输入数据
    if not sweep_data["ai_data_list"]:
        raise ValueError("SweepData的ai_data_list为空")

    # 从ao_data获取参考信号参数
    ao_waveform = sweep_data["ao_data"]

    # 获取参考频率
    ref_frequency = getattr(ao_waveform, "frequency", None)
    if ref_frequency is None:
        raise ValueError("无法从ao_data获取frequency属性，请确保ao_data已设置frequency")

    # 获取参考复振幅（取第一个通道），用于计算传递函数 H = detected / ref
    cca = getattr(ao_waveform, "channel_complex_amplitudes", None)
    if cca is not None:
        ref_complex_amp: complex = complex(cca[0])
    else:
        ref_complex_amp = 1.0 + 0j
        f_logger.warning("ao_data未设置channel_complex_amplitudes，使用默认参考复振幅=1+0j")

    # 避免参考复振幅为零
    if ref_complex_amp == 0:
        ref_complex_amp = 1.0 + 0j
        f_logger.warning("参考信号复振幅为0，已替换为1+0j")

    f_logger.info(
        f"参考信号参数: 频率={ref_frequency}Hz, "
        f"复振幅={ref_complex_amp} (幅值={abs(ref_complex_amp):.4f})"
    )

    # 1. 对SweepData进行平均
    f_logger.debug("步骤1: 对SweepData进行平均")
    averaged_sweep_data = average_sweep_data(sweep_data)

    # 2. 对平均后的数据进行滤波
    f_logger.debug("步骤2: 对平均后的数据进行滤波")
    filtered_sweep_data = filter_sweep_data(
        averaged_sweep_data,
        lowcut=lowcut,
        highcut=highcut,
        filter_order=filter_order,
        trim_samples=trim_samples,
    )

    # 3. 对每个点的波形提取单频信息并计算传递函数
    f_logger.debug("步骤3: 提取单频信息并计算传递函数")
    tf_results: list[PointTFData] = []

    for point_data in filtered_sweep_data["ai_data_list"]:
        position = point_data["position"]
        # 每个点只有一个波形（已经平均过了）
        waveform = point_data["ai_data"][0]

        # 提取单频信息（返回记录了结果的Waveform对象）
        result_wf = extract_single_tone_information_vvi(
            waveform,
            approx_freq=ref_frequency,
        )

        # 复数传递函数 = 检测复振幅 / 参考复振幅
        complex_amplitude = complex(
            result_wf.channel_complex_amplitudes[0] / ref_complex_amp
        )

        tf_results.append({
            "position": position,
            "complex_amplitude": complex_amplitude,
        })

    f_logger.info(f"SweepData转换完成，共 {len(tf_results)} 个点的传递函数数据")

    return tf_results


def combine_point_tf_data_list(
    list_a: list[PointTFData] | None,
    list_b: list[PointTFData] | None,
    mode: Literal["add", "minus"] = "minus",
) -> list[PointTFData] | None:
    """
    将两个 PointTFData 列表逐点相加或相减。

    假定两个列表长度相同，且相同序号的点具有相同的 position。
    - mode="add":   result.complex_amplitude = a.complex_amplitude + b.complex_amplitude
    - mode="minus": result.complex_amplitude = a.complex_amplitude - b.complex_amplitude

    若任一输入为 None，直接返回 None（安全模式）。

    Args:
        list_a: 第一个传递函数数据列表（可为 None）
        list_b: 第二个传递函数数据列表（可为 None）
        mode: 运算模式，"add"为相加，"minus"为相减，默认"minus"

    Returns:
        list[PointTFData] | None: 运算后的传递函数数据列表，
        若任一输入为 None 则返回 None

    Raises:
        ValueError: 当两个列表长度不一致或mode不合法时
    """
    # None 安全模式
    if list_a is None or list_b is None:
        return None

    f_logger = get_logger(f"{__name__}.combine_point_tf_data_list")

    if mode not in ("add", "minus"):
        raise ValueError(f"mode必须为'add'或'minus'，当前值: {mode!r}")

    if len(list_a) != len(list_b):
        raise ValueError(
            f"两个列表长度不一致: list_a={len(list_a)}, list_b={len(list_b)}"
        )

    f_logger.info(
        f"开始{'相加' if mode == 'add' else '相减'}操作，共 {len(list_a)} 个点"
    )

    result_list: list[PointTFData] = []
    for point_a, point_b in zip(list_a, list_b, strict=True):
        if mode == "add":
            combined = point_a["complex_amplitude"] + point_b["complex_amplitude"]
        else:
            combined = point_a["complex_amplitude"] - point_b["complex_amplitude"]
        result_list.append({
            "position": point_a["position"],
            "complex_amplitude": combined,
        })

    f_logger.info("运算完成")
    return result_list


def pick_area(
    data_list: list[PointTFData],
    picked_center: Point2D = Point2D(x=0, y=0),
    picked_area_radius: PositiveFloat = 100,
    area_shape: Literal["square", "circle"] = "square",
) -> list[PointTFData]:
    """
    从 PointTFData 列表中筛选出位于指定区域内的点。

    根据 area_shape 选择不同的距离判定方式：
    - "square": x 和 y 到 center 的距离分别小于 radius（方形区域）
    - "circle": 欧氏距离小于 radius（圆形区域）

    Args:
        data_list: 输入的 PointTFData 列表
        picked_center: 区域中心点坐标（mm），默认 (0, 0)
        picked_area_radius: 区域半径（mm），默认 100
        area_shape: 区域形状，"square" 或 "circle"，默认 "square"

    Returns:
        list[PointTFData]: 筛选后的点列表
    """
    f_logger = get_logger(f"{__name__}.pick_area")

    result: list[PointTFData] = []
    cx, cy = picked_center.x, picked_center.y
    r = picked_area_radius

    for point in data_list:
        px, py = point["position"].x, point["position"].y
        if area_shape == "square":
            if abs(px - cx) <= r and abs(py - cy) <= r:
                result.append(point)
        else:  # circle
            if math.sqrt((px - cx) ** 2 + (py - cy) ** 2) <= r:
                result.append(point)

    f_logger.info(
        f"区域筛选完成: center=({cx}, {cy}), radius={r}, shape={area_shape}, "
        f"筛选出 {len(result)}/{len(data_list)} 个点"
    )
    return result


def calculate_amplitude_integral(
    data_list: list[PointTFData],
    mode: Literal["abs", "fourier"] = "abs",
    k_modulus: float = 2 * np.pi / 0.1,
    k_angle_deg: float = 0,
) -> float | complex:
    """
    计算 PointTFData 列表所代表区域的平均振幅或傅里叶分量。

    - mode="abs": 计算所有点 complex_amplitude 模长的平均值（返回 float）
    - mode="fourier": 计算向波矢 k 方向传播的分量的平均振幅（返回 complex）。
      数学公式为:
          (|k| / (2π)) * (1/N) * Σ p(r_j) * exp(-i k·r_j)
      其中 k_x = k_modulus * cos(k_angle_deg), k_y = k_modulus * sin(k_angle_deg)，
      r_j 为各点坐标（已从 mm 换算为 m）。

    Args:
        data_list: 输入的 PointTFData 列表
        mode: 计算模式，"abs" 或 "fourier"，默认 "abs"
        k_modulus: 波矢 k 的模值（1/m），默认 2π/0.1
        k_angle_deg: 波矢 k 的角度（度），0=x正向，90=y正向，默认 0

    Returns:
        float (mode="abs") 或 complex (mode="fourier")
    """
    f_logger = get_logger(f"{__name__}.calculate_amplitude_integral")

    if not data_list:
        f_logger.warning("输入数据为空，返回 0")
        return 0.0 if mode == "abs" else 0.0 + 0j

    if mode == "abs":
        amps = [abs(p["complex_amplitude"]) for p in data_list]
        result = float(np.mean(amps))
        f_logger.info(f"abs模式: 平均振幅 = {result:.6f}")
        return result

    # fourier 模式
    k_angle_rad = math.radians(k_angle_deg)
    kx = k_modulus * math.cos(k_angle_rad)
    ky = k_modulus * math.sin(k_angle_rad)

    total = 0.0 + 0j
    for p in data_list:
        x_m = p["position"].x * 1e-3  # mm → m
        y_m = p["position"].y * 1e-3
        phase = kx * x_m + ky * y_m
        total += p["complex_amplitude"] * np.exp(-1j * phase)

    result = (k_modulus / (2 * np.pi)) * total / len(data_list)
    f_logger.info(
        f"fourier模式: k=({kx:.4f}, {ky:.4f}), "
        f"结果={result:.6f} (|结果|={abs(result):.6f})"
    )
    return result


def plot_point_tf_data_list(
    plot_tf_results: list[PointTFData],
    mode: Literal["discrete", "interpolated", "instantaneous"] = "interpolated",
    save_path: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[Figure, Axes | tuple[Axes, Axes]]:
    """
    绘制传递函数的空间分布图（统一接口）

    该函数接收已计算好的传递函数结果，根据mode参数选择不同的绘图模式：
    - "discrete": 方形色块版本，使用等距网格色块表示，适用于等距网格数据
    - "interpolated": 插值版本，使用连续彩色区域（contourf）展现空间分布
    - "instantaneous": 瞬时声压场，计算复振幅的实部 Re(H) 模拟某一瞬间的声压场分布

    Args:
        plot_tf_results: 传递函数计算结果列表（PointTFData格式）
        mode: 绘图模式，可选 "discrete"、"interpolated"、"instantaneous"，
            默认为 "interpolated"
        save_path: 保存图片的路径，如果为None则不保存，默认为None
        vmin: 颜色映射最小值（仅 instantaneous 模式），默认自动计算
        vmax: 颜色映射最大值（仅 instantaneous 模式），默认自动计算

    Returns:
        fig: matplotlib Figure对象
        axes: 对于 "discrete" 和 "interpolated" 模式返回 (ax1, ax2) 元组；
              对于 "instantaneous" 模式返回单个 Axes 对象
    """
    # 获取函数日志器
    f_logger = get_logger(f"{__name__}.plot_point_tf_data_list")

    if not plot_tf_results:
        f_logger.error("传递函数结果为空，无法绘图")
        raise ValueError("传递函数结果不能为空")

    valid_modes = ("discrete", "interpolated", "instantaneous")
    if mode not in valid_modes:
        raise ValueError(f"mode必须为 {valid_modes} 之一，当前值: {mode!r}")

    f_logger.info(f"绘制 {len(plot_tf_results)} 个点的传递函数分布（模式: {mode}）")

    # 提取公共数据
    x_coords = np.array([result["position"].x for result in plot_tf_results])
    y_coords = np.array([result["position"].y for result in plot_tf_results])
    complex_amplitudes = np.array([result["complex_amplitude"] for result in plot_tf_results])
    amp_ratios = np.abs(complex_amplitudes)
    phase_shifts = np.angle(complex_amplitudes)

    f_logger.debug(
        f"数据范围: X=[{x_coords.min():.2f}, {x_coords.max():.2f}], "
        f"Y=[{y_coords.min():.2f}, {y_coords.max():.2f}], "
        f"幅值比=[{amp_ratios.min():.4f}, {amp_ratios.max():.4f}], "
        f"相位差=[{phase_shifts.min():.4f}, {phase_shifts.max():.4f}]"
    )

    # 根据模式分派绘图
    if mode == "discrete":
        fig, axes_pair = _plot_discrete(
            x_coords, y_coords, amp_ratios, phase_shifts, f_logger
        )
        result: tuple[Figure, Axes | tuple[Axes, Axes]] = (fig, axes_pair)
    elif mode == "interpolated":
        fig, axes_pair = _plot_interpolated(
            x_coords, y_coords, amp_ratios, phase_shifts, f_logger
        )
        result = (fig, axes_pair)
    else:  # instantaneous
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        _plot_instantaneous_on_ax(
            ax, fig, x_coords, y_coords, complex_amplitudes,
            vmin=vmin, vmax=vmax,
            show_colorbar=True, colorbar_label="瞬时声压场强度",
            title="瞬时声压场分布 Re(H)",
        )
        plt.tight_layout()
        f_logger.debug("瞬时声压场分布图绘制完成")
        result = (fig, ax)

    # 保存图片（如果指定了路径）
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        f_logger.info(f"图片已保存至: {save_path}")

    return result


def _plot_discrete(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    amp_ratios: np.ndarray,
    phase_shifts: np.ndarray,
    f_logger,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """绘制方形色块版本的传递函数空间分布图（内部函数）"""
    # 计算网格间距
    unique_x = np.unique(x_coords)
    unique_y = np.unique(y_coords)
    dx = float(np.min(np.diff(unique_x))) if len(unique_x) > 1 else 1.0
    dy = float(np.min(np.diff(unique_y))) if len(unique_y) > 1 else 1.0

    block_width = dx * 0.9
    block_height = dy * 0.9

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 幅值比分布图
    amp_cmap = "viridis"
    for x, y, amp in zip(x_coords, y_coords, amp_ratios, strict=False):
        rect = Rectangle(
            (x - block_width / 2, y - block_height / 2),
            block_width,
            block_height,
            facecolor=plt.cm.get_cmap(amp_cmap)(
                (amp - amp_ratios.min()) / (amp_ratios.max() - amp_ratios.min())
            ),
            edgecolor="black",
            linewidth=0.5,
        )
        ax1.add_patch(rect)

    ax1.set_xlim(x_coords.min() - dx, x_coords.max() + dx)
    ax1.set_ylim(y_coords.min() - dy, y_coords.max() + dy)
    ax1.set_xlabel("X 坐标 (mm)", fontsize=12)
    ax1.set_ylabel("Y 坐标 (mm)", fontsize=12)
    ax1.set_title("传递函数 - 幅值比空间分布", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_aspect("equal", adjustable="box")

    scatter1 = ax1.scatter(
        [], [], c=[], cmap=amp_cmap, vmin=amp_ratios.min(), vmax=amp_ratios.max()
    )
    cbar1 = fig.colorbar(scatter1, ax=ax1, label="幅值比")
    cbar1.ax.tick_params(labelsize=10)

    # 相位差分布图
    phase_cmap = "twilight"
    for x, y, phase in zip(x_coords, y_coords, phase_shifts, strict=False):
        rect = Rectangle(
            (x - block_width / 2, y - block_height / 2),
            block_width,
            block_height,
            facecolor=plt.cm.get_cmap(phase_cmap)(
                (phase - phase_shifts.min()) / (phase_shifts.max() - phase_shifts.min())
            ),
            edgecolor="black",
            linewidth=0.5,
        )
        ax2.add_patch(rect)

    ax2.set_xlim(x_coords.min() - dx, x_coords.max() + dx)
    ax2.set_ylim(y_coords.min() - dy, y_coords.max() + dy)
    ax2.set_xlabel("X 坐标 (mm)", fontsize=12)
    ax2.set_ylabel("Y 坐标 (mm)", fontsize=12)
    ax2.set_title("传递函数 - 相位差空间分布", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_aspect("equal", adjustable="box")

    scatter2 = ax2.scatter(
        [], [], c=[], cmap=phase_cmap, vmin=phase_shifts.min(), vmax=phase_shifts.max()
    )
    cbar2 = fig.colorbar(scatter2, ax=ax2, label="相位差 (rad)")
    cbar2.ax.tick_params(labelsize=10)

    plt.tight_layout()
    f_logger.debug("方形色块分布图绘制完成")
    return fig, (ax1, ax2)


def _plot_interpolated(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    amp_ratios: np.ndarray,
    phase_shifts: np.ndarray,
    f_logger,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """绘制插值版本的传递函数空间分布图（内部函数）"""
    grid_resolution = 100
    interpolation_method = "cubic"

    # 创建插值网格
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1

    xi = np.linspace(x_min - x_margin, x_max + x_margin, grid_resolution)
    yi = np.linspace(y_min - y_margin, y_max + y_margin, grid_resolution)
    Xi, Yi = np.meshgrid(xi, yi)

    # 处理相位周期性：转换为复数形式进行插值
    phase_complex = np.exp(1j * phase_shifts)
    points = np.column_stack((x_coords, y_coords))

    try:
        amp_interpolated = griddata(
            points, amp_ratios, (Xi, Yi), method=interpolation_method, fill_value=np.nan
        )
        phase_real_interp = griddata(
            points, phase_complex.real, (Xi, Yi), method=interpolation_method, fill_value=np.nan
        )
        phase_imag_interp = griddata(
            points, phase_complex.imag, (Xi, Yi), method=interpolation_method, fill_value=np.nan
        )
        phase_interpolated = np.angle(phase_real_interp + 1j * phase_imag_interp)
    except Exception as e:
        f_logger.warning(f"cubic插值失败({e})，回退到linear插值")
        amp_interpolated = griddata(
            points, amp_ratios, (Xi, Yi), method="linear", fill_value=np.nan
        )
        phase_real_interp = griddata(
            points, phase_complex.real, (Xi, Yi), method="linear", fill_value=np.nan
        )
        phase_imag_interp = griddata(
            points, phase_complex.imag, (Xi, Yi), method="linear", fill_value=np.nan
        )
        phase_interpolated = np.angle(phase_real_interp + 1j * phase_imag_interp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 幅值比插值图
    im1 = ax1.contourf(Xi, Yi, amp_interpolated, levels=50, cmap="viridis", extend="both")
    ax1.set_xlabel("X 坐标 (mm)", fontsize=12)
    ax1.set_ylabel("Y 坐标 (mm)", fontsize=12)
    ax1.set_title("传递函数 - 幅值比插值分布", fontsize=14, fontweight="bold")
    ax1.set_aspect("equal", adjustable="box")
    cbar1 = fig.colorbar(im1, ax=ax1, label="幅值比")
    cbar1.ax.tick_params(labelsize=10)

    # 相位差插值图
    im2 = ax2.contourf(Xi, Yi, phase_interpolated, levels=50, cmap="twilight", extend="both")
    ax2.set_xlabel("X 坐标 (mm)", fontsize=12)
    ax2.set_ylabel("Y 坐标 (mm)", fontsize=12)
    ax2.set_title("传递函数 - 相位差插值分布", fontsize=14, fontweight="bold")
    ax2.set_aspect("equal", adjustable="box")
    cbar2 = fig.colorbar(im2, ax=ax2, label="相位差 (rad)")
    cbar2.ax.tick_params(labelsize=10)

    plt.tight_layout()
    f_logger.debug("插值分布图绘制完成")
    return fig, (ax1, ax2)


def _plot_instantaneous_on_ax(
    ax: Axes,
    fig: Figure,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    complex_amplitudes: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    show_colorbar: bool = True,
    colorbar_label: str = "瞬时声压场强度",
    title: str = "",
) -> plt.cm.ScalarMappable:
    """在已有的 Axes 上绘制瞬时场插值图（内部辅助函数）。

    Returns:
        contourf 返回的 ScalarMappable 对象，可用于外部创建共享 ColorBar。
    """
    grid_resolution = 100
    instantaneous_field = complex_amplitudes.real

    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1

    xi = np.linspace(x_min - x_margin, x_max + x_margin, grid_resolution)
    yi = np.linspace(y_min - y_margin, y_max + y_margin, grid_resolution)
    Xi, Yi = np.meshgrid(xi, yi)

    points = np.column_stack((x_coords, y_coords))
    try:
        field_interp = griddata(
            points, instantaneous_field, (Xi, Yi),
            method="cubic", fill_value=np.nan,
        )
    except Exception:
        field_interp = griddata(
            points, instantaneous_field, (Xi, Yi),
            method="linear", fill_value=np.nan,
        )

    # 强制对称颜色范围，确保白色始终代表零声压
    if vmin is None or vmax is None:
        auto_vmax = float(max(
            np.max(np.abs(instantaneous_field)),
            np.nanmax(np.abs(field_interp)),
        ))
        if vmin is None:
            vmin = -auto_vmax
        if vmax is None:
            vmax = auto_vmax
    # 强制对称: vmin = -vmax
    abs_max = max(abs(vmin), abs(vmax))
    vmin, vmax = -abs_max, abs_max

    im = ax.contourf(
        Xi, Yi, field_interp,
        levels=50, cmap="RdBu_r", vmin=vmin, vmax=vmax,
    )
    ax.set_xlabel("X 坐标 (mm)", fontsize=10)
    ax.set_ylabel("Y 坐标 (mm)", fontsize=10)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")

    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, label=colorbar_label)
        cbar.ax.tick_params(labelsize=10)

    return im


# ---------------------------------------------------------------------------
#  综合实验绘图函数
# ---------------------------------------------------------------------------


def _render_subplot(
    ax: Axes, fig: Figure,
    data_list: list[PointTFData] | None,
    state: str,
    vmin: float, vmax: float,
    picked_center: Point2D,
    picked_area_radius: float,
    area_shape: str,
    title: str = "",
):
    """渲染单个子图（normal / partial / unavailable）。"""
    ax.set_title(title, fontsize=10, fontweight="bold")

    if state == "unavailable" or data_list is None:
        ax.set_facecolor("lightgray")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # 内联绘制瞬时场插值图
    _x = np.array([p["position"].x for p in data_list])
    _y = np.array([p["position"].y for p in data_list])
    _ca = np.array([p["complex_amplitude"] for p in data_list])
    _plot_instantaneous_on_ax(
        ax, fig, _x, _y, _ca,
        vmin=vmin, vmax=vmax,
        show_colorbar=False, title=title,
    )

    if state == "partial":
        for spine in ax.spines.values():
            spine.set_edgecolor("orange")
            spine.set_linewidth(3)
        ax.text(
            0.5, 0.5, "!",
            transform=ax.transAxes, fontsize=48,
            color="orange", alpha=0.3,
            ha="center", va="center", fontweight="bold",
        )
    else:  # normal — 绘制 picked_area 轮廓
        cx, cy = picked_center.x, picked_center.y
        r = picked_area_radius
        if area_shape == "circle":
            patch = Circle(
                (cx, cy), r, fill=False,
                edgecolor="orange", linewidth=2, linestyle="--",
            )
            ax.add_patch(patch)
        else:
            ax.plot(
                [cx - r, cx + r, cx + r, cx - r, cx - r],
                [cy - r, cy - r, cy + r, cy + r, cy - r],
                color="orange", linewidth=2, linestyle="--",
            )


def plot_comprehensive_experiment(
    # --- 10 组数据路径 ---
    left_r_0_static_folder: str | Path | None = None,
    left_r_0_feedback_folder: str | Path | None = None,
    left_r_minus1_static_folder: str | Path | None = None,
    left_r_minus1_feedback_folder: str | Path | None = None,
    right_r_plus1_static_folder: str | Path | None = None,
    right_r_plus1_feedback_folder: str | Path | None = None,
    right_r_0_static_folder: str | Path | None = None,
    right_r_0_feedback_folder: str | Path | None = None,
    left_background_folder: str | Path | None = None,
    right_background_folder: str | Path | None = None,
    # --- 区域选取参数 ---
    picked_center: Point2D = Point2D(x=155.5, y=155.5),
    picked_area_radius: PositiveFloat = 100,
    area_shape: Literal["square", "circle"] = "square",
    # --- 积分参数 ---
    integral_mode: Literal["abs", "fourier"] = "abs",
    k_modulus: float = 2 * np.pi / 0.1,
    k_angle_deg: float = 180,
    save_path: str | Path | None = None,
) -> tuple[Figure, tuple]:
    """
    绘制一次完整综合实验的结果图。

    基于 10 组 SweepData 进行处理、加减运算、区域选取、振幅积分，
    最终生成 passive 和 active 两张图，每张包含瞬时场分布、
    共享归一化 ColorBar 以及散射矩阵数值。

    子图状态:
    - **normal**: 正常绘制瞬时场 + picked_area 橘黄色线框
    - **partial**: 绘制瞬时场 + 橘黄色粗边框 + 感叹号（数据部分缺失）
    - **unavailable**: 灰色填充（数据完全不可用）

    Args:
        left_r_0_static_folder: 左侧 r=0 静态数据文件夹路径
        left_r_0_feedback_folder: 左侧 r=0 反馈数据文件夹路径
        left_r_minus1_static_folder: 左侧 r=-1 静态数据文件夹路径
        left_r_minus1_feedback_folder: 左侧 r=-1 反馈数据文件夹路径
        right_r_plus1_static_folder: 右侧 r=+1 静态数据文件夹路径
        right_r_plus1_feedback_folder: 右侧 r=+1 反馈数据文件夹路径
        right_r_0_static_folder: 右侧 r=0 静态数据文件夹路径
        right_r_0_feedback_folder: 右侧 r=0 反馈数据文件夹路径
        left_background_folder: 左侧背景数据文件夹路径
        right_background_folder: 右侧背景数据文件夹路径
        picked_center: 区域选取中心点 (mm)，默认 (155.5, 155.5)
        picked_area_radius: 区域选取半径 (mm)，默认 100
        area_shape: 区域形状 "square" 或 "circle"，默认 "square"
        integral_mode: 积分模式 "abs" 或 "fourier"，默认 "abs"
        k_modulus: 波矢模值 (1/m)，默认 2π/0.1
        k_angle_deg: 波矢角度 (度)，默认 180
        save_path: 图片保存路径，None 则不保存

    Returns:
        (fig_passive, fig_active): 两个 Figure 对象的元组
    """
    f_logger = get_logger(f"{__name__}.plot_comprehensive_experiment")
    f_logger.info("开始绘制综合实验结果")

    # -- 内部辅助: 从文件夹加载 SweepData --
    def _load_from_folder(folder: str | Path | None) -> SweepData | None:
        if folder is None:
            return None
        pkl_path = Path(folder) / "sweep_data.pkl"
        try:
            return load_compressed_data(pkl_path, data_type_name="SweepData")
        except (FileNotFoundError, OSError) as e:
            f_logger.warning(f"加载失败: {pkl_path}, {e}")
            return None

    # -- 内部辅助: 渲染 S 矩阵 --
    def _render_s_matrix_inner(
        ax: Axes,
        s_real: list[list[float]],
        s_complex: list[list[complex]] | None = None,
    ):
        ax.axis("off")
        ax.set_title("Scattering Matrix", fontsize=11, fontweight="bold")
        m = [[f"{abs(v):.4f}" for v in row] for row in s_real]
        mat_str = (
            f"$S = "
            f"\\left[\\begin{{matrix}}"
            f"{m[0][0]} & {m[0][1]} \\\\[4pt]"
            f"{m[1][0]} & {m[1][1]}"
            f"\\end{{matrix}}\\right]$"
        )
        y_pos = 0.65 if s_complex is not None else 0.5
        ax.text(0.5, y_pos, mat_str, fontsize=13, ha="center", va="center")
        if s_complex is not None:
            rows_strs = []
            for row in s_complex:
                parts = [f"{v.real:+.3f}{v.imag:+.3f}i" for v in row]
                rows_strs.append(" & ".join(parts))
            cmat = " \\\\[4pt]".join(rows_strs)
            cmat_str = f"$\\left[\\begin{{matrix}}{cmat}\\end{{matrix}}\\right]$"
            ax.text(0.5, 0.28, cmat_str, fontsize=9, ha="center", va="center")

    # ==================================================================
    # 1. 加载数据
    # ==================================================================
    folder_map = {
        "l_r0_s": left_r_0_static_folder,
        "l_r0_f": left_r_0_feedback_folder,
        "l_rm1_s": left_r_minus1_static_folder,
        "l_rm1_f": left_r_minus1_feedback_folder,
        "r_rp1_s": right_r_plus1_static_folder,
        "r_rp1_f": right_r_plus1_feedback_folder,
        "r_r0_s": right_r_0_static_folder,
        "r_r0_f": right_r_0_feedback_folder,
        "l_bg": left_background_folder,
        "r_bg": right_background_folder,
    }
    raw: dict[str, SweepData | None] = {}
    for key, folder in folder_map.items():
        raw[key] = _load_from_folder(folder)
        if raw[key] is None and folder is not None:
            f_logger.warning(f"数据加载失败: {key} → {folder}")

    # 获取实验频率（从第一个可用的 sweep_data 中提取）
    ref_freq = 1000.0
    for sd in raw.values():
        if sd is not None:
            ref_freq = float(getattr(sd["ao_data"], "frequency", 1000.0))
            break
    lowcut, highcut = ref_freq / 2, ref_freq * 2
    f_logger.info(f"实验频率: {ref_freq} Hz, 滤波范围: [{lowcut}, {highcut}] Hz")

    # ==================================================================
    # 2. 转换为 PointTFData
    # ==================================================================
    tf: dict[str, list[PointTFData] | None] = {}
    for key, sd in raw.items():
        if sd is not None:
            tf[key] = sweep_data_to_point_tf_data_list(
                sd, lowcut=lowcut, highcut=highcut
            )
        else:
            tf[key] = None

    # ==================================================================
    # 3. 加减运算 → passive / active
    #    combine_point_tf_data_list 已内置 None 安全特性
    # ==================================================================
    # passive = static - background
    p_l_r0 = combine_point_tf_data_list(tf["l_r0_s"], tf["r_bg"], mode="minus")
    p_l_rm1 = combine_point_tf_data_list(tf["l_rm1_s"], tf["l_bg"], mode="minus")
    p_r_rp1 = combine_point_tf_data_list(tf["r_rp1_s"], tf["l_bg"], mode="minus")
    p_r_r0 = combine_point_tf_data_list(tf["r_r0_s"], tf["r_bg"], mode="minus")

    # active = passive + feedback
    a_l_r0 = combine_point_tf_data_list(p_l_r0, tf["l_r0_f"], mode="add")
    a_l_rm1 = combine_point_tf_data_list(p_l_rm1, tf["l_rm1_f"], mode="add")
    a_r_rp1 = combine_point_tf_data_list(p_r_rp1, tf["r_rp1_f"], mode="add")
    a_r_r0 = combine_point_tf_data_list(p_r_r0, tf["r_r0_f"], mode="add")

    bg_data = tf["l_bg"]

    # ==================================================================
    # 4. 确定每个子图的状态
    # ==================================================================
    def _passive_state(static_key: str, bg_key: str) -> str:
        if raw[static_key] is None:
            return "unavailable"
        if raw[bg_key] is None:
            return "partial"
        return "normal"

    def _active_state(static_key: str, bg_key: str, fb_key: str) -> str:
        if raw[static_key] is None:
            return "unavailable"
        if raw[bg_key] is None or raw[fb_key] is None:
            return "partial"
        return "normal"

    passive_info = [
        (p_l_r0, _passive_state("l_r0_s", "r_bg"), "r\u2080 (left)"),
        (p_r_rp1, _passive_state("r_rp1_s", "l_bg"), "r\u208a\u2081 (right)"),
        (p_l_rm1, _passive_state("l_rm1_s", "l_bg"), "r\u208b\u2081 (left)"),
        (p_r_r0, _passive_state("r_r0_s", "r_bg"), "r\u2080 (right)"),
    ]
    active_info = [
        (a_l_r0, _active_state("l_r0_s", "r_bg", "l_r0_f"), "r\u2080 (left)"),
        (a_r_rp1, _active_state("r_rp1_s", "l_bg", "r_rp1_f"), "r\u208a\u2081 (right)"),
        (a_l_rm1, _active_state("l_rm1_s", "l_bg", "l_rm1_f"), "r\u208b\u2081 (left)"),
        (a_r_r0, _active_state("r_r0_s", "r_bg", "r_r0_f"), "r\u2080 (right)"),
    ]
    bg_state = "normal" if bg_data is not None else "unavailable"

    # ==================================================================
    # 5. 区域选取 + 振幅积分
    # ==================================================================
    def _pick_and_integrate(
        data: list[PointTFData] | None,
        k_angle: float | None = None,
    ) -> tuple[float, complex | None]:
        """返回 (amp_for_display, complex_integral_or_None)。"""
        if data is None:
            return 0.0, None
        picked = pick_area(data, picked_center, picked_area_radius, area_shape)
        amp = calculate_amplitude_integral(picked, mode=integral_mode,
                                           k_modulus=k_modulus,
                                           k_angle_deg=k_angle if k_angle is not None else k_angle_deg)
        if integral_mode == "fourier":
            return abs(amp), amp  # type: ignore[return-value]
        return float(amp), None

    p_amps = [_pick_and_integrate(d)[0] for d, _, _ in passive_info]
    a_amps = [_pick_and_integrate(d)[0] for d, _, _ in active_info]
    bg_amp_val, _ = _pick_and_integrate(bg_data, k_angle=0)

    # fourier 模式下额外计算复数积分值
    p_complex = (
        [_pick_and_integrate(d)[1] for d, _, _ in passive_info]
        if integral_mode == "fourier" else None
    )
    a_complex = (
        [_pick_and_integrate(d)[1] for d, _, _ in active_info]
        if integral_mode == "fourier" else None
    )

    # ==================================================================
    # 6. 计算全局颜色范围和归一化
    # ==================================================================
    all_datasets: list[list[PointTFData]] = []
    for d, s, _ in passive_info + active_info:
        if d is not None and s != "unavailable":
            all_datasets.append(d)
    if bg_data is not None:
        all_datasets.append(bg_data)

    global_max = 0.0
    for d in all_datasets:
        inst_vals = np.array([p["complex_amplitude"].real for p in d])
        if inst_vals.size:
            global_max = max(global_max, float(np.max(np.abs(inst_vals))))

    bg_amp_abs = abs(bg_amp_val) if bg_amp_val else 1.0
    norm_ref = max(global_max, bg_amp_abs)
    if norm_ref == 0:
        norm_ref = 1.0
    g_vmax = norm_ref
    g_vmin = -norm_ref

    # ==================================================================
    # 7. 构建单张图的通用绘制器
    # ==================================================================
    def _build_figure(
        info_list: list[tuple],
        bg_st: str,
        s_real: list[list[float]],
        s_cplx: list[list[complex]] | None,
        fig_title: str,
    ) -> Figure:
        fig = plt.figure(figsize=(24, 12))
        gs = GridSpec(
            2, 5, figure=fig,
            width_ratios=[1.2, 1, 1, 0.55, 0.35],
            hspace=0.32, wspace=0.28,
        )

        # 左侧背景子图（跨两行）
        ax_bg = fig.add_subplot(gs[:, 0])
        _render_subplot(
            ax_bg, fig, bg_data, bg_st, g_vmin, g_vmax,
            picked_center, picked_area_radius, area_shape,
            title="Background (left)",
        )

        # 2×2 子图: 左上=r₀(L), 右上=r₊₁(R), 左下=r₋₁(L), 右下=r₀(R)
        positions = [(0, 1), (0, 2), (1, 1), (1, 2)]
        for (data, state, title), (r, c) in zip(info_list, positions):
            ax = fig.add_subplot(gs[r, c])
            _render_subplot(
                ax, fig, data, state, g_vmin, g_vmax,
                picked_center, picked_area_radius, area_shape,
                title=title,
            )

        # S 矩阵（第4列，跨两行）
        s_ax = fig.add_subplot(gs[:, 3])
        _render_s_matrix_inner(s_ax, s_real, s_cplx)

        # 共享 ColorBar（第5列，归一化至背景振幅）
        sm = plt.cm.ScalarMappable(
            cmap="RdBu_r",
            norm=plt.Normalize(vmin=g_vmin, vmax=g_vmax),
        )
        cbar_ax = fig.add_subplot(gs[:, 4])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        ticks = np.linspace(g_vmin, g_vmax, 7)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t / bg_amp_abs:.2f}" for t in ticks])
        cbar.set_label("归一化振幅 (÷ 背景)", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

        fig.suptitle(fig_title, fontsize=14, fontweight="bold", y=0.98)
        return fig

    # ==================================================================
    # 8. 绘制 Passive 图
    # ==================================================================
    s_real_p = [
        [p_amps[0], p_amps[1]],   # [r₀(L), r₊₁(R)]
        [p_amps[2], p_amps[3]],   # [r₋₁(L), r₀(R)]
    ]
    s_cplx_p = None
    if integral_mode == "fourier" and p_complex is not None:
        s_cplx_p = [
            [p_complex[0], p_complex[1]],
            [p_complex[2], p_complex[3]],
        ]
    fig_passive = _build_figure(
        passive_info, bg_state, s_real_p, s_cplx_p,
        f"Passive  |  mode={integral_mode}  freq={ref_freq:.0f} Hz",
    )

    # ==================================================================
    # 9. 绘制 Active 图
    # ==================================================================
    s_real_a = [
        [a_amps[0], a_amps[1]],
        [a_amps[2], a_amps[3]],
    ]
    s_cplx_a = None
    if integral_mode == "fourier" and a_complex is not None:
        s_cplx_a = [
            [a_complex[0], a_complex[1]],
            [a_complex[2], a_complex[3]],
        ]
    fig_active = _build_figure(
        active_info, bg_state, s_real_a, s_cplx_a,
        f"Active  |  mode={integral_mode}  freq={ref_freq:.0f} Hz",
    )

    # ==================================================================
    # 10. 保存
    # ==================================================================
    if save_path is not None:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        fig_passive.savefig(sp.with_stem(sp.stem + "_passive"),
                            dpi=300, bbox_inches="tight")
        fig_active.savefig(sp.with_stem(sp.stem + "_active"),
                           dpi=300, bbox_inches="tight")
        f_logger.info(f"图片已保存至: {save_path}")

    f_logger.info("综合实验绘图完成")
    return fig_passive, fig_active


# ---------------------------------------------------------------------------
#  时域波形绘图函数
# ---------------------------------------------------------------------------


def plot_waveform(
    waveform: Waveform,
    figsize: tuple[float, float] = (12, 6),
    title: str | None = None,
    save_path: str | None = None,
    show_grid: bool = True,
    zoom_factor: float = 1.0,
    point_index: int | None = None,
    waveform_index: int | None = None,
) -> tuple[Figure, Axes]:
    """
    绘制Waveform对象的时域波形图

    该函数接收一个Waveform对象，绘制其时域波形图。
    支持单通道和多通道波形的可视化。

    Args:
        waveform: 要可视化的Waveform对象
        figsize: 图形尺寸 (宽, 高)，单位为英寸，默认为 (12, 6)
        title: 图形标题，如果为None则使用默认标题，默认为None
        save_path: 保存图片的路径，如果为None则不保存，默认为None
        show_grid: 是否显示网格，默认为True
        zoom_factor: 时间轴缩放因子，用于放大波形的时间轴。
            例如，zoom_factor=10时仅绘制前1/10的波形，zoom_factor=2时绘制前1/2的波形。
            默认为1.0（不缩放，绘制完整波形）
        point_index: 数据点序号（从0开始），用于在标题中显示，默认为None
        waveform_index: 波形序号（从0开始），用于在标题中显示，默认为None

    Returns:
        fig: matplotlib Figure对象
        ax: Axes对象

    Raises:
        ValueError: 当输入的waveform为空或zoom_factor小于等于0时

    Examples:
        ```python
        假设已有一个Waveform对象
        >>> test_waveform = Waveform(np.sin(2*np.pi*1000*np.linspace(0, 1, 10000)),
        ...                     sampling_rate=10000)
        >>> test_fig, test_ax = plot_waveform(waveform)
        >>> plt.show()
        或者保存图片
        >>> test_fig, test_ax = plot_waveform(waveform, save_path="waveform.png")
        放大10倍，仅绘制前1/10的波形
        >>> test_fig, test_ax = plot_waveform(waveform, zoom_factor=10)
        >>> plt.show()
        指定数据点和波形序号
        >>> test_fig, test_ax = plot_waveform(waveform, point_index=5, waveform_index=2)
        >>> plt.show()
        ```
    """
    # 获取函数日志器
    f_logger = get_logger(f"{__name__}.plot_waveform")

    f_logger.info("开始绘制Waveform时域波形图")

    # 验证输入
    if waveform.size == 0:
        f_logger.error("输入的Waveform对象为空，无法绘图")
        raise ValueError("Waveform对象不能为空")

    if zoom_factor <= 0:
        f_logger.error(f"zoom_factor必须大于0，当前值: {zoom_factor}")
        raise ValueError("zoom_factor必须大于0")

    f_logger.info(
        f"绘制Waveform: shape={waveform.shape}, "
        f"sampling_rate={waveform.sampling_rate}Hz, "
        f"duration={waveform.duration:.6f}s, "
        f"zoom_factor={zoom_factor}"
    )

    # 1. 根据zoom_factor计算需要绘制的样本数
    total_samples = waveform.samples_num
    samples_to_plot = int(total_samples / zoom_factor)

    # 确保至少绘制2个采样点
    if samples_to_plot < 2:
        f_logger.warning(
            f"zoom_factor={zoom_factor}过大，导致绘制样本数不足2个，"
            f"将调整为绘制2个采样点"
        )
        samples_to_plot = 2

    # 计算实际绘制的时长
    duration_to_plot = samples_to_plot / waveform.sampling_rate

    f_logger.debug(
        f"zoom_factor={zoom_factor}, 总样本数={total_samples}, "
        f"绘制样本数={samples_to_plot}, 绘制时长={duration_to_plot:.6f}s"
    )

    # 2. 生成时间轴（仅包含需要绘制的部分）
    time_array = np.linspace(0, duration_to_plot, samples_to_plot, endpoint=False)

    f_logger.debug(
        f"时间轴范围: [0, {duration_to_plot:.6f}]s, 采样点数: {samples_to_plot}"
    )

    # 3. 创建图形
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # 4. 绘制波形（仅绘制需要的部分）
    # Waveform统一使用2D格式 (n_channels, n_samples)
    colors = plt.cm.tab10(np.linspace(0, 1, waveform.channels_num))
    for ch_idx in range(waveform.channels_num):
        ax.plot(
            time_array,
            waveform[ch_idx, :samples_to_plot],
            linewidth=1.0,
            color=colors[ch_idx],
            label=f"通道 {ch_idx + 1}",
            alpha=0.8,
        )
    f_logger.debug(f"绘制 {waveform.channels_num} 个通道的波形")

    # 5. 设置坐标轴标签和标题
    ax.set_xlabel("时间 (s)", fontsize=12)
    ax.set_ylabel("幅值", fontsize=12)

    # 设置标题
    if title is None:
        # 使用默认标题
        title_parts = []

        # 添加数据点和波形序号信息
        if point_index is not None and waveform_index is not None:
            title_parts.append(f"数据点 {point_index} - 波形 {waveform_index}")
        elif point_index is not None:
            title_parts.append(f"数据点 {point_index}")
        elif waveform_index is not None:
            title_parts.append(f"波形 {waveform_index}")

        # 添加基本信息
        if zoom_factor == 1.0:
            title_parts.append(
                f"采样率: {waveform.sampling_rate:.0f} Hz, "
                f"持续时间: {waveform.duration:.6f} s"
            )
        else:
            title_parts.append(
                f"采样率: {waveform.sampling_rate:.0f} Hz, "
                f"显示时长: {duration_to_plot:.6f} s / {waveform.duration:.6f} s, "
                f"缩放: {zoom_factor}x"
            )

        default_title = (
            "时域波形图 - " + " | ".join(title_parts) if title_parts else "时域波形图"
        )
        ax.set_title(default_title, fontsize=14, fontweight="bold")
    else:
        ax.set_title(title, fontsize=14, fontweight="bold")

    # 6. 添加图例（如果有多个通道）
    if waveform.channels_num > 1:
        ax.legend(loc="upper right", fontsize=10)

    # 7. 可选：显示网格
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle="--")

    f_logger.debug("波形图绘制完成")

    # 8. 调整布局
    plt.tight_layout()

    # 9. 保存图片（如果指定了路径）
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        f_logger.info(f"波形图已保存至: {save_path}")

    return fig, ax


def plot_sweep_waveforms(
    sweep_data: SweepData,
    output_dir: str,
    zoom_factor: float = 1.0,
    figsize: tuple[float, float] = (12, 6),
    show_grid: bool = True,
) -> str:
    """
    批量绘制SweepData中所有波形的时域图

    该函数接收一个SweepData对象，在指定的输出目录中创建一个新文件夹，
    并对SweepData中每一个数据点的每一段waveform波形使用plot_waveform函数
    绘制时域波形图。所有图像将被智能地自动命名并保存。

    Args:
        sweep_data: 要可视化的SweepData对象
        output_dir: 输出目录路径
        zoom_factor: 时间轴缩放因子，传递给plot_waveform函数，默认为1.0
        figsize: 图形尺寸 (宽, 高)，单位为英寸，默认为 (12, 6)
        show_grid: 是否显示网格，默认为True

    Returns:
        创建的输出文件夹的完整路径

    Raises:
        ValueError: 当输入的sweep_data为空时
        OSError: 当无法创建输出目录时

    Examples:
        ```python
        假设已有一个SweepData对象
        >>> sweep_data = load_sweep_data("measurement.pkl")  # noqa
        >>> output_folder = plot_sweep_waveforms(
        ...     sweep_data,
        ...     output_dir="D:/plots",
        ...     zoom_factor=10
        ... )
        >>> print(f"所有波形图已保存至: {output_folder}")
        ```
    """
    # 获取函数日志器
    f_logger = get_logger(f"{__name__}.plot_sweep_waveforms")

    from datetime import datetime

    f_logger.info("开始批量绘制SweepData波形图")

    # 验证输入
    ai_data_list = sweep_data["ai_data_list"]
    if not ai_data_list:
        f_logger.error("输入的SweepData对象为空，无法绘图")
        raise ValueError("SweepData对象不能为空")

    # 创建输出目录
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        f_logger.info(f"输出目录不存在，创建目录: {output_dir_path}")
        output_dir_path.mkdir(parents=True, exist_ok=True)

    # 创建带时间戳的子文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"waveforms_{timestamp}"
    output_folder = output_dir_path / folder_name

    try:
        output_folder.mkdir(parents=True, exist_ok=True)
        f_logger.info(f"创建输出文件夹: {output_folder}")
    except OSError as e:
        f_logger.error(f"无法创建输出文件夹: {e}", exc_info=True)
        raise

    # 统计总波形数
    total_waveforms = sum(len(point_data["ai_data"]) for point_data in ai_data_list)
    f_logger.info(f"开始绘制 {len(ai_data_list)} 个数据点的共 {total_waveforms} 个波形")

    # 遍历每个数据点
    waveform_count = 0
    for point_idx, point_data in enumerate(ai_data_list):
        position = point_data["position"]
        ai_waveforms = point_data["ai_data"]

        f_logger.debug(
            f"处理数据点 {point_idx} (位置: {position}), 共 {len(ai_waveforms)} 个波形"
        )

        # 遍历该点的每个波形
        for waveform_idx, waveform in enumerate(ai_waveforms):
            # 生成文件名：point_{点序号}_waveform_{波形序号}.png
            filename = f"point_{point_idx:04d}_waveform_{waveform_idx:02d}.png"
            save_path = output_folder / filename

            # 绘制波形图
            try:
                fig, _ = plot_waveform(
                    waveform=waveform,
                    figsize=figsize,
                    title=None,  # 使用默认标题，会自动包含点序号和波形序号
                    save_path=str(save_path),
                    show_grid=show_grid,
                    zoom_factor=zoom_factor,
                    point_index=point_idx,
                    waveform_index=waveform_idx,
                )
                # 关闭图形以释放内存
                plt.close(fig)

                waveform_count += 1

                # 每10个波形输出一次进度
                if waveform_count % 10 == 0 or waveform_count == total_waveforms:
                    f_logger.info(
                        f"进度: {waveform_count}/{total_waveforms} "
                        f"({waveform_count / total_waveforms * 100:.1f}%)"
                    )

            except Exception as e:
                f_logger.error(
                    f"绘制数据点 {point_idx} 的波形 {waveform_idx} 时发生错误: {e}",
                    exc_info=True,
                )
                # 继续处理下一个波形
                continue

    f_logger.info(f"批量绘制完成，成功绘制 {waveform_count}/{total_waveforms} 个波形图")
    f_logger.info(f"所有波形图已保存至: {output_folder}")

    return str(output_folder)


def plot_sweep_data_as_single_waveform(
    sweep_data: SweepData,
    figsize: tuple[float, float] = (12, 6),
    title: str | None = None,
    save_path: str | None = None,
    show_grid: bool = True,
    zoom_factor: float = 1.0,
) -> tuple[Figure, Axes]:
    """
    绘制SweepData的融合波形图

    该函数接收一个SweepData对象，先使用average_sweep_data函数对其进行平均，
    然后将所有测量点的所有AI通道波形按位相加并取平均，融合为单一的Waveform，
    最后使用plot_waveform函数绘制这个融合波形。

    融合逻辑：
    1. 首先对SweepData调用average_sweep_data，将每个测量点的多个波形平均为一个波形
    2. 然后遍历所有测量点的ai_data，将所有波形按位相加
    3. 最后除以总波形数，得到全局融合的单一波形
    4. 使用plot_waveform函数绘制这个融合波形

    Args:
        sweep_data: 要可视化的SweepData对象
        figsize: 图形尺寸 (宽, 高)，单位为英寸，默认为 (12, 6)
        title: 图形标题，如果为None则使用默认标题，默认为None
        save_path: 保存图片的路径，如果为None则不保存，默认为None
        show_grid: 是否显示网格，默认为True
        zoom_factor: 时间轴缩放因子，传递给plot_waveform函数，默认为1.0

    Returns:
        fig: matplotlib Figure对象
        ax: Axes对象

    Raises:
        ValueError: 当输入的sweep_data为空时

    Examples:
        ```python
        # 假设已有一个SweepData对象
        >>> sweep_data = load_sweep_data("measurement.pkl")  # noqa
        >>> fig, ax = plot_sweep_data_as_single_waveform(sweep_data)
        >>> plt.show()
        # 或者保存图片并放大10倍
        >>> fig, ax = plot_sweep_data_as_single_waveform(
        ...     sweep_data,
        ...     save_path="fusion_waveform.png",
        ...     zoom_factor=10
        ... )
        ```
    """
    # 获取函数日志器
    f_logger = get_logger(f"{__name__}.plot_sweep_data_as_single_waveform")

    f_logger.info("开始绘制SweepData的融合波形图")

    # 验证输入
    ai_data_list = sweep_data["ai_data_list"]
    if not ai_data_list:
        f_logger.error("输入的SweepData对象为空，无法绘图")
        raise ValueError("SweepData对象不能为空")

    f_logger.info(f"输入SweepData包含 {len(ai_data_list)} 个测量点")

    # 第一步：对SweepData进行平均（每个测量点的多个波形平均为一个波形）
    f_logger.info("对SweepData进行平均处理")
    averaged_sweep_data = average_sweep_data(sweep_data)

    # 第二步：获取参考波形的元数据
    first_point_waveform = averaged_sweep_data["ai_data_list"][0]["ai_data"][0]
    sampling_rate = first_point_waveform.sampling_rate
    samples_num = first_point_waveform.samples_num
    num_channels = first_point_waveform.channels_num

    f_logger.debug(
        f"波形信息: 通道数={num_channels}，采样点数={samples_num}"
    )

    # 第三步：按位相加所有测量点的所有AI波形
    total_waveforms = 0  # 计算总波形数

    # 初始化累加数组 (num_channels, samples_num)
    summed_data = np.zeros((num_channels, samples_num), dtype=np.float64)

    for point_data in averaged_sweep_data["ai_data_list"]:
        for waveform in point_data["ai_data"]:
            # 验证波形维度
            if waveform.channels_num != num_channels or waveform.samples_num != samples_num:
                f_logger.warning(
                    f"波形维度不一致：期望 ({num_channels}, {samples_num})，"
                    f"实际 ({waveform.channels_num}, {waveform.samples_num})，跳过此波形"
                )
                continue
            summed_data += waveform
            total_waveforms += 1

    # 取平均
    if total_waveforms == 0:
        f_logger.error("没有有效的波形数据可供融合")
        raise ValueError("没有有效的波形数据可供融合")

    averaged_data = summed_data / total_waveforms

    # 创建融合后的Waveform对象（保留channel_names元数据）
    fusion_waveform = Waveform(
        input_array=averaged_data,
        sampling_rate=sampling_rate,
        timestamp=first_point_waveform.timestamp,
        channel_names=first_point_waveform.channel_names,
    )

    f_logger.info(
        f"融合完成: {num_channels}个通道，融合了{total_waveforms}个波形"
    )

    # 第四步：使用plot_waveform函数绘制融合波形
    f_logger.info("绘制融合波形")

    # 构建默认标题（如果未指定）
    if title is None:
        title = (
            f"融合波形图 - 融合了{total_waveforms}个波形 | "
            f"采样率: {sampling_rate:.0f} Hz"
        )

    fig, ax = plot_waveform(
        waveform=fusion_waveform,
        figsize=figsize,
        title=title,
        save_path=save_path,
        show_grid=show_grid,
        zoom_factor=zoom_factor,
    )

    f_logger.info("SweepData融合波形图绘制完成")

    return fig, ax
