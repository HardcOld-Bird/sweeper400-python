# pyright: basic
"""
# 数据可视化模块

模块路径：`sweeper400.analyze.plot`

包含对采集数据进行可视化处理的函数和类。
"""

from typing import Literal, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata

from ..logger import get_logger
from .basic_sine import extract_single_tone_information_vvi
from .filter import filter_sweep_data
from .my_dtypes import Point2D, SweepData, Waveform
from .post_process import average_sweep_data


# 获取模块日志器
logger = get_logger(__name__)

# 配置matplotlib中文字体支持
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Microsoft JhengHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


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


def subtract_point_tf_data_list(
    total_field_list: list[PointTFData],
    background_field_list: list[PointTFData],
) -> list[PointTFData]:
    """
    将两个 PointTFData 列表逼点相减，获取去除背景场后的复振幅。

    假定两个列表长度相同，且相同序号的点具有相同的 position。
    对每一对同位置点，计算:
        result.complex_amplitude = total.complex_amplitude - background.complex_amplitude

    返回的列表可以直接传入 plot_point_tf_data_list 进行绘图。

    Args:
        total_field_list: 总场（含背景）的传递函数数据列表
        background_field_list: 背景场的传递函数数据列表

    Returns:
        list[PointTFData]: 去除背景场后的传递函数数据列表

    Raises:
        ValueError: 当两个列表长度不一致时

    Examples:
        ```python
        >>> total = sweep_data_to_point_tf_data_list(total_sweep_data)
        >>> background = sweep_data_to_point_tf_data_list(background_sweep_data)
        >>> net_field = subtract_point_tf_data_list(total, background)
        >>> fig, axes = plot_point_tf_data_list(net_field, mode="interpolated")
        ```
    """
    f_logger = get_logger(f"{__name__}.subtract_point_tf_data_list")

    if len(total_field_list) != len(background_field_list):
        raise ValueError(
            f"两个列表长度不一致: "
            f"total_field_list={len(total_field_list)}, "
            f"background_field_list={len(background_field_list)}"
        )

    f_logger.info(
        f"开始计算差值场，共 {len(total_field_list)} 个点"
    )

    result_list: list[PointTFData] = []
    for total_point, bg_point in zip(total_field_list, background_field_list, strict=True):
        net_complex_amplitude = (
            total_point["complex_amplitude"] - bg_point["complex_amplitude"]
        )
        result_list.append({
            "position": total_point["position"],
            "complex_amplitude": net_complex_amplitude,
        })

    f_logger.info("差值场计算完成")
    return result_list


def plot_point_tf_data_list(
    plot_tf_results: list[PointTFData],
    mode: Literal["discrete", "interpolated", "instantaneous"] = "interpolated",
    save_path: str | None = None,
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

    Returns:
        fig: matplotlib Figure对象
        axes: 对于 "discrete" 和 "interpolated" 模式返回 (ax1, ax2) 元组；
              对于 "instantaneous" 模式返回单个 Axes 对象

    Raises:
        ValueError: 当输入数据为空或mode不合法时

    Examples:
        ```python
        >>> # 假设已有传递函数计算结果
        >>> test_plot_tf_results = sweep_data_to_point_tf_data_list(test_sweep_data)  # noqa
        >>> # 使用离散色块模式
        >>> fig, (ax1, ax2) = plot_point_tf_data_list(test_plot_tf_results, mode="discrete")
        >>> # 使用插值模式
        >>> fig, (ax1, ax2) = plot_point_tf_data_list(test_plot_tf_results, mode="interpolated")
        >>> # 使用瞬时声压场模式
        >>> fig, ax = plot_point_tf_data_list(test_plot_tf_results, mode="instantaneous")
        >>> plt.show()
        ```
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
        fig, ax = _plot_instantaneous(
            x_coords, y_coords, complex_amplitudes, f_logger
        )
        result = (fig, ax)

    # 保存图片（如果指定了路径）
    if save_path is not None:
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


def _plot_instantaneous(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    complex_amplitudes: np.ndarray,
    f_logger,
) -> tuple[Figure, Axes]:
    """绘制瞬时声压场分布图（内部函数）"""
    grid_resolution = 100
    interpolation_method = "cubic"

    # 瞬时声压场 = 复振幅的实部 Re(H)
    instantaneous_field = complex_amplitudes.real

    # 创建插值网格
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1

    xi = np.linspace(x_min - x_margin, x_max + x_margin, grid_resolution)
    yi = np.linspace(y_min - y_margin, y_max + y_margin, grid_resolution)
    Xi, Yi = np.meshgrid(xi, yi)

    points = np.column_stack((x_coords, y_coords))

    try:
        field_interpolated = griddata(
            points, instantaneous_field, (Xi, Yi),
            method=interpolation_method, fill_value=np.nan,
        )
    except Exception as e:
        f_logger.warning(f"cubic插值失败({e})，回退到linear插值")
        field_interpolated = griddata(
            points, instantaneous_field, (Xi, Yi),
            method="linear", fill_value=np.nan,
        )

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # 对称颜色范围，确保零值在中央
    vmax = max(np.max(np.abs(instantaneous_field)), np.nanmax(np.abs(field_interpolated)))
    vmin = -vmax

    im = ax.contourf(
        Xi, Yi, field_interpolated,
        levels=50, cmap="RdBu_r", vmin=vmin, vmax=vmax,
    )
    ax.set_xlabel("X 坐标 (mm)", fontsize=12)
    ax.set_ylabel("Y 坐标 (mm)", fontsize=12)
    ax.set_title("瞬时声压场分布 Re(H)", fontsize=14, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(im, ax=ax, label="瞬时声压场强度")
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    f_logger.debug("瞬时声压场分布图绘制完成")
    return fig, ax


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
    from pathlib import Path

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
