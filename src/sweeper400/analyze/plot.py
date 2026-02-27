# pyright: basic
"""
# 数据可视化模块

模块路径：`sweeper400.analyze.plot`

包含对采集数据进行可视化处理的函数和类。
"""

from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata

from ..logger import get_logger
from .my_dtypes import Point2D, SweepData, Waveform


# 本地定义的空间点传递函数数据类型，专用于空间扫场绘图函数。
# 注意：此类型与 ChannelTFData 不同，它包含空间坐标 position，
# 用于描述空间中某一测量点的传递函数结果。
class PointTFData(TypedDict):
    """
    空间点传递函数数据格式（仅供 plot.py 内部绘图函数使用）。

    ## 内部组成:
        **position**: 测量点的二维空间坐标（mm）
        **amp_ratio**: 幅值比
        **phase_shift**: 相位差（弧度制）
    """

    position: Point2D
    amp_ratio: float
    phase_shift: float

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


def plot_transfer_function_discrete_distribution(
    tf_results: list[PointTFData],
    figsize: tuple[float, float] = (14, 6),
    amp_cmap: str = "viridis",
    phase_cmap: str = "twilight",
    save_path: str | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """
    绘制传递函数的空间分布图（方形色块版本）

    该函数接收已计算好的传递函数结果，绘制幅值比和相位差的二维空间分布图。
    两张子图并排显示，使用方形色块表示幅值比/相位差的大小，形成"像素画"效果。
    适用于等距网格数据的可视化。

    Args:
        tf_results: 传递函数计算结果列表
        figsize: 图形尺寸 (宽, 高)，单位为英寸，默认为 (14, 6)
        amp_cmap: 幅值比图的colormap名称，默认为 "viridis"
        phase_cmap: 相位差图的colormap名称，默认为 "twilight"
        save_path: 保存图片的路径，如果为None则不保存，默认为None

    Returns:
        fig: matplotlib Figure对象
        (ax1, ax2): 包含两个Axes对象的元组，分别对应幅值比图和相位差图

    Raises:
        ValueError: 当输入数据为空时

    Examples:
        ```python
        >>> # 假设已有传递函数计算结果
        >>> tf_results = calculate_transfer_function(raw_data)  # noqa
        >>> fig, (ax1, ax2) = plot_transfer_function_discrete_distribution(tf_results)
        >>> plt.show()
        >>> # 或者保存图片
        >>> fig, axes = plot_transfer_function_discrete_distribution(  # noqa
        ...     tf_results, save_path="transfer_function.png"
        ... )
        ```
    """
    logger.info("开始绘制传递函数空间分布图（方形色块版本）")

    if not tf_results:
        logger.error("传递函数结果为空，无法绘图")
        raise ValueError("传递函数结果不能为空")

    logger.info(f"绘制 {len(tf_results)} 个点的传递函数分布")

    # 2. 提取数据
    x_coords = np.array([result["position"].x for result in tf_results])
    y_coords = np.array([result["position"].y for result in tf_results])
    amp_ratios = np.array([result["amp_ratio"] for result in tf_results])
    phase_shifts = np.array([result["phase_shift"] for result in tf_results])

    logger.debug(
        f"数据范围: X=[{x_coords.min():.2f}, {x_coords.max():.2f}], "
        f"Y=[{y_coords.min():.2f}, {y_coords.max():.2f}], "
        f"幅值比=[{amp_ratios.min():.4f}, {amp_ratios.max():.4f}], "
        f"相位差=[{phase_shifts.min():.4f}, {phase_shifts.max():.4f}]"
    )

    # 3. 计算网格参数（假设数据为等距网格）
    unique_x = np.unique(x_coords)
    unique_y = np.unique(y_coords)

    # 计算网格间距
    if len(unique_x) > 1:
        dx = np.min(np.diff(unique_x))
    else:
        dx = 1.0  # 默认间距

    if len(unique_y) > 1:
        dy = np.min(np.diff(unique_y))
    else:
        dy = 1.0  # 默认间距

    # 方形色块的尺寸（稍微小于网格间距以避免重叠）
    block_width = dx * 0.9
    block_height = dy * 0.9

    logger.debug(
        f"网格间距: dx={dx:.2f}, dy={dy:.2f}, "
        f"色块尺寸: {block_width:.2f}x{block_height:.2f}"
    )

    # 4. 创建图形和子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 5. 绘制幅值比分布图（方形色块）
    for _i, (x, y, amp) in enumerate(zip(x_coords, y_coords, amp_ratios, strict=False)):
        # 创建方形色块
        rect1 = Rectangle(
            (x - block_width / 2, y - block_height / 2),
            block_width,
            block_height,
            facecolor=plt.cm.get_cmap(amp_cmap)(
                (amp - amp_ratios.min()) / (amp_ratios.max() - amp_ratios.min())
            ),
            edgecolor="black",
            linewidth=0.5,
        )
        ax1.add_patch(rect1)

    # 设置坐标轴范围
    ax1.set_xlim(x_coords.min() - dx, x_coords.max() + dx)
    ax1.set_ylim(y_coords.min() - dy, y_coords.max() + dy)
    ax1.set_xlabel("X 坐标 (mm)", fontsize=12)
    ax1.set_ylabel("Y 坐标 (mm)", fontsize=12)
    ax1.set_title("传递函数 - 幅值比空间分布", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_aspect("equal", adjustable="box")

    # 添加颜色条（使用虚拟的scatter来创建colorbar）
    scatter1 = ax1.scatter(
        [], [], c=[], cmap=amp_cmap, vmin=amp_ratios.min(), vmax=amp_ratios.max()
    )
    cbar1 = fig.colorbar(scatter1, ax=ax1, label="幅值比")
    cbar1.ax.tick_params(labelsize=10)

    logger.debug("幅值比分布图绘制完成")

    # 6. 绘制相位差分布图（方形色块）
    for _i, (x, y, phase) in enumerate(
        zip(x_coords, y_coords, phase_shifts, strict=False)
    ):
        # 创建方形色块
        rect2 = Rectangle(
            (x - block_width / 2, y - block_height / 2),
            block_width,
            block_height,
            facecolor=plt.cm.get_cmap(phase_cmap)(
                (phase - phase_shifts.min()) / (phase_shifts.max() - phase_shifts.min())
            ),
            edgecolor="black",
            linewidth=0.5,
        )
        ax2.add_patch(rect2)

    # 设置坐标轴范围
    ax2.set_xlim(x_coords.min() - dx, x_coords.max() + dx)
    ax2.set_ylim(y_coords.min() - dy, y_coords.max() + dy)
    ax2.set_xlabel("X 坐标 (mm)", fontsize=12)
    ax2.set_ylabel("Y 坐标 (mm)", fontsize=12)
    ax2.set_title("传递函数 - 相位差空间分布", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_aspect("equal", adjustable="box")

    # 添加颜色条（使用虚拟的scatter来创建colorbar）
    scatter2 = ax2.scatter(
        [], [], c=[], cmap=phase_cmap, vmin=phase_shifts.min(), vmax=phase_shifts.max()
    )
    cbar2 = fig.colorbar(scatter2, ax=ax2, label="相位差 (rad)")
    cbar2.ax.tick_params(labelsize=10)

    logger.debug("相位差分布图绘制完成")

    # 6. 调整布局
    plt.tight_layout()

    # 7. 保存图片（如果指定了路径）
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"图片已保存至: {save_path}")

    logger.info("传递函数空间分布图绘制完成")

    return fig, (ax1, ax2)


def plot_transfer_function_interpolated_distribution(
    tf_results: list[PointTFData],
    figsize: tuple[float, float] = (14, 6),
    amp_cmap: str = "viridis",
    phase_cmap: str = "twilight",
    interpolation_method: str = "cubic",
    grid_resolution: int = 100,
    save_path: str | None = None,
    show_measurement_points: bool = True,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """
    绘制传递函数的插值空间分布图（解决相位周期性问题）

    该函数接收已计算好的传递函数结果，使用插值方法创建连续的彩色区域图像，
    更好地展现声场的空间分布特性。特别处理了相位的周期性问题，避免-π到π的跳跃影响插值效果。

    Args:
        tf_results: 传递函数计算结果列表
        figsize: 图形尺寸 (宽, 高)，单位为英寸，默认为 (14, 6)
        amp_cmap: 幅值比图的colormap名称，默认为 "viridis"
        phase_cmap: 相位差图的colormap名称，默认为 "twilight"
        interpolation_method: 插值方法，可选 "linear", "nearest", "cubic"，
            默认为 "cubic"
        grid_resolution: 插值网格分辨率，默认为 100
        save_path: 保存图片的路径，如果为None则不保存，默认为None
        show_measurement_points: 是否在插值图上显示原始测量点，默认为True

    Returns:
        fig: matplotlib Figure对象
        (ax1, ax2): 包含两个Axes对象的元组，分别对应幅值比图和相位差图

    Raises:
        ValueError: 当输入数据为空时

    Examples:
        ```python
        >>> # 假设已有传递函数计算结果
        >>> tf_results = calculate_transfer_function(raw_data)  # noqa
        >>> fig, (ax1, ax2) = plot_transfer_function_interpolated_distribution(
        ...     tf_results
        ... )
        >>> plt.show()
        >>> # 或者保存图片并使用线性插值
        >>> fig, axes = plot_transfer_function_interpolated_distribution(  # noqa
        ...     tf_results,
        ...     interpolation_method="linear",
        ...     save_path="transfer_function_interpolated.png"
        ... )
        ```
    """
    logger.info("开始绘制传递函数插值空间分布图")

    if not tf_results:
        logger.error("传递函数结果为空，无法绘图")
        raise ValueError("传递函数结果不能为空")

    logger.info(f"绘制 {len(tf_results)} 个点的传递函数插值分布")

    # 2. 提取数据
    x_coords = np.array([result["position"].x for result in tf_results])
    y_coords = np.array([result["position"].y for result in tf_results])
    amp_ratios = np.array([result["amp_ratio"] for result in tf_results])
    phase_shifts = np.array([result["phase_shift"] for result in tf_results])

    logger.debug(
        f"数据范围: X=[{x_coords.min():.2f}, {x_coords.max():.2f}], "
        f"Y=[{y_coords.min():.2f}, {y_coords.max():.2f}], "
        f"幅值比=[{amp_ratios.min():.4f}, {amp_ratios.max():.4f}], "
        f"相位差=[{phase_shifts.min():.4f}, {phase_shifts.max():.4f}]"
    )

    # 3. 创建插值网格
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # 扩展边界以获得更好的插值效果
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_margin = x_range * 0.1  # 10%的边界扩展
    y_margin = y_range * 0.1

    xi = np.linspace(x_min - x_margin, x_max + x_margin, grid_resolution)
    yi = np.linspace(y_min - y_margin, y_max + y_margin, grid_resolution)
    Xi, Yi = np.meshgrid(xi, yi)

    logger.debug(
        f"创建插值网格: {grid_resolution}x{grid_resolution}, "
        f"方法: {interpolation_method}"
    )

    # 4. 处理相位的周期性问题
    # 将相位转换为复数形式进行插值，避免-π到π的跳跃
    phase_complex = np.exp(1j * phase_shifts)  # 转换为单位圆上的复数

    logger.debug("处理相位周期性，转换为复数形式进行插值")

    # 5. 执行插值
    points = np.column_stack((x_coords, y_coords))

    try:
        # 插值幅值比
        amp_interpolated = griddata(
            points, amp_ratios, (Xi, Yi), method=interpolation_method, fill_value=np.nan
        )

        # 插值相位的实部和虚部
        phase_real_interpolated = griddata(
            points,
            phase_complex.real,
            (Xi, Yi),
            method=interpolation_method,
            fill_value=np.nan,
        )

        phase_imag_interpolated = griddata(
            points,
            phase_complex.imag,
            (Xi, Yi),
            method=interpolation_method,
            fill_value=np.nan,
        )

        # 从插值后的实部和虚部重构相位
        phase_interpolated_complex = (
            phase_real_interpolated + 1j * phase_imag_interpolated
        )
        phase_interpolated = np.angle(phase_interpolated_complex)  # 转换回相位角度

        logger.debug("插值计算完成（包含相位周期性处理）")

    except Exception as e:
        logger.error(f"插值计算失败: {e}")
        # 如果cubic插值失败，回退到linear插值
        if interpolation_method == "cubic":
            logger.warning("cubic插值失败，回退到linear插值")
            amp_interpolated = griddata(
                points, amp_ratios, (Xi, Yi), method="linear", fill_value=np.nan
            )

            # 对相位也使用linear插值的复数方法
            phase_real_interpolated = griddata(
                points, phase_complex.real, (Xi, Yi), method="linear", fill_value=np.nan
            )
            phase_imag_interpolated = griddata(
                points, phase_complex.imag, (Xi, Yi), method="linear", fill_value=np.nan
            )
            phase_interpolated_complex = (
                phase_real_interpolated + 1j * phase_imag_interpolated
            )
            phase_interpolated = np.angle(phase_interpolated_complex)
        else:
            raise

    # 5. 创建图形和子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 6. 绘制幅值比插值分布图
    im1 = ax1.contourf(
        Xi, Yi, amp_interpolated, levels=50, cmap=amp_cmap, extend="both"
    )
    ax1.set_xlabel("X 坐标 (mm)", fontsize=12)
    ax1.set_ylabel("Y 坐标 (mm)", fontsize=12)
    ax1.set_title("传递函数 - 幅值比插值分布", fontsize=14, fontweight="bold")
    ax1.set_aspect("equal", adjustable="box")

    # 添加颜色条
    cbar1 = fig.colorbar(im1, ax=ax1, label="幅值比")
    cbar1.ax.tick_params(labelsize=10)

    # 可选：显示原始测量点
    if show_measurement_points:
        ax1.scatter(
            x_coords,
            y_coords,
            c="white",
            s=30,
            edgecolors="black",
            linewidths=1,
            alpha=0.8,
            zorder=10,
        )

    logger.debug("幅值比插值分布图绘制完成")

    # 7. 绘制相位差插值分布图
    im2 = ax2.contourf(
        Xi, Yi, phase_interpolated, levels=50, cmap=phase_cmap, extend="both"
    )
    ax2.set_xlabel("X 坐标 (mm)", fontsize=12)
    ax2.set_ylabel("Y 坐标 (mm)", fontsize=12)
    ax2.set_title("传递函数 - 相位差插值分布", fontsize=14, fontweight="bold")
    ax2.set_aspect("equal", adjustable="box")

    # 添加颜色条
    cbar2 = fig.colorbar(im2, ax=ax2, label="相位差 (rad)")
    cbar2.ax.tick_params(labelsize=10)

    # 可选：显示原始测量点
    if show_measurement_points:
        ax2.scatter(
            x_coords,
            y_coords,
            c="white",
            s=30,
            edgecolors="black",
            linewidths=1,
            alpha=0.8,
            zorder=10,
        )

    logger.debug("相位差插值分布图绘制完成")

    # 8. 调整布局
    plt.tight_layout()

    # 9. 保存图片（如果指定了路径）
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"插值分布图已保存至: {save_path}")

    logger.info("传递函数插值空间分布图绘制完成")

    return fig, (ax1, ax2)


def plot_transfer_function_instantaneous_field(
    tf_results: list[PointTFData],
    figsize: tuple[float, float] = (10, 8),
    field_cmap: str = "RdBu_r",
    interpolation_method: str = "cubic",
    grid_resolution: int = 100,
    save_path: str | None = None,
    show_measurement_points: bool = True,
) -> tuple[Figure, Axes]:
    """
    绘制瞬时声压场分布图

    该函数接收已计算好的传递函数结果，计算 A·sin(φ) 来模拟某一瞬间的声压场分布。
    使用插值方法创建连续的彩色区域图像，展现瞬时声场的空间分布特性。

    Args:
        tf_results: 传递函数计算结果列表
        figsize: 图形尺寸 (宽, 高)，单位为英寸，默认为 (10, 8)
        field_cmap: 声压场图的colormap名称，默认为 "RdBu_r"（红蓝色图，适合表示正负值）
        interpolation_method: 插值方法，可选 "linear", "nearest", "cubic"，
            默认为 "cubic"
        grid_resolution: 插值网格分辨率，默认为 100
        save_path: 保存图片的路径，如果为None则不保存，默认为None
        show_measurement_points: 是否在插值图上显示原始测量点，默认为True

    Returns:
        fig: matplotlib Figure对象
        ax: Axes对象

    Raises:
        ValueError: 当输入数据为空时

    Examples:
        ```python
        >>> # 假设已有传递函数计算结果
        >>> tf_results = calculate_transfer_function(raw_data)  # noqa
        >>> fig, ax = plot_transfer_function_instantaneous_field(tf_results)
        >>> plt.show()
        >>> # 或者保存图片并使用线性插值
        >>> fig, ax = plot_transfer_function_instantaneous_field(  # noqa
        ...     tf_results,
        ...     interpolation_method="linear",
        ...     save_path="instantaneous_field.png"
        ... )
        ```
    """
    logger.info("开始绘制瞬时声压场分布图")

    if not tf_results:
        logger.error("传递函数结果为空，无法绘图")
        raise ValueError("传递函数结果不能为空")

    logger.info(f"绘制 {len(tf_results)} 个点的瞬时声压场分布")

    # 1. 提取数据
    x_coords = np.array([result["position"].x for result in tf_results])
    y_coords = np.array([result["position"].y for result in tf_results])
    amp_ratios = np.array([result["amp_ratio"] for result in tf_results])
    phase_shifts = np.array([result["phase_shift"] for result in tf_results])

    # 2. 计算瞬时声压场值 A·sin(φ)
    instantaneous_field = amp_ratios * np.sin(phase_shifts)

    logger.debug(
        f"数据范围: X=[{x_coords.min():.2f}, {x_coords.max():.2f}], "
        f"Y=[{y_coords.min():.2f}, {y_coords.max():.2f}], "
        f"瞬时声压场=[{instantaneous_field.min():.4f}, {instantaneous_field.max():.4f}]"
    )

    # 3. 创建插值网格
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # 扩展边界以获得更好的插值效果
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_margin = x_range * 0.1  # 10%的边界扩展
    y_margin = y_range * 0.1

    xi = np.linspace(x_min - x_margin, x_max + x_margin, grid_resolution)
    yi = np.linspace(y_min - y_margin, y_max + y_margin, grid_resolution)
    Xi, Yi = np.meshgrid(xi, yi)

    logger.debug(
        f"创建插值网格: {grid_resolution}x{grid_resolution}, "
        f"方法: {interpolation_method}"
    )

    # 4. 执行插值
    points = np.column_stack((x_coords, y_coords))

    try:
        # 插值瞬时声压场值（由于sin函数本身具有周期性，无需特殊处理）
        field_interpolated = griddata(
            points,
            instantaneous_field,
            (Xi, Yi),
            method=interpolation_method,
            fill_value=np.nan,
        )

        logger.debug("瞬时声压场插值计算完成")

    except Exception as e:
        logger.error(f"插值计算失败: {e}")
        # 如果cubic插值失败，回退到linear插值
        if interpolation_method == "cubic":
            logger.warning("cubic插值失败，回退到linear插值")
            field_interpolated = griddata(
                points,
                instantaneous_field,
                (Xi, Yi),
                method="linear",
                fill_value=np.nan,
            )
        else:
            raise

    # 5. 创建图形
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # 6. 绘制瞬时声压场分布图
    # 使用对称的颜色范围，确保零值在颜色图中央
    # 首先基于原始数据计算对称范围，然后考虑插值数据的范围
    original_field_max = np.max(np.abs(instantaneous_field))
    interpolated_field_max = np.nanmax(np.abs(field_interpolated))

    # 取两者中的较大值，确保所有数据都在颜色范围内
    vmax = max(original_field_max, interpolated_field_max)
    vmin = -vmax

    logger.debug(f"颜色范围: [{vmin:.4f}, {vmax:.4f}]")

    # 使用 contourf 但不使用 extend 参数，避免颜色突变
    im = ax.contourf(
        Xi,
        Yi,
        field_interpolated,
        levels=50,
        cmap=field_cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel("X 坐标 (mm)", fontsize=12)
    ax.set_ylabel("Y 坐标 (mm)", fontsize=12)
    ax.set_title("瞬时声压场分布 (A·sin(φ))", fontsize=14, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")

    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax, label="瞬时声压场强度")
    cbar.ax.tick_params(labelsize=10)

    # 可选：显示原始测量点
    if show_measurement_points:
        ax.scatter(
            x_coords,
            y_coords,
            c="white",
            s=30,
            edgecolors="black",
            linewidths=1,
            alpha=0.8,
            zorder=10,
        )

    logger.debug("瞬时声压场分布图绘制完成")

    # 7. 调整布局
    plt.tight_layout()

    # 8. 保存图片（如果指定了路径）
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"瞬时声压场分布图已保存至: {save_path}")

    logger.info("瞬时声压场分布图绘制完成")

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
        >>> waveform = Waveform(np.sin(2*np.pi*1000*np.linspace(0, 1, 10000)),
        ...                     sampling_rate=10000)
        >>> fig, ax = plot_waveform(waveform)
        >>> plt.show()
        或者保存图片
        >>> fig, ax = plot_waveform(waveform, save_path="waveform.png")
        放大10倍，仅绘制前1/10的波形
        >>> fig, ax = plot_waveform(waveform, zoom_factor=10)
        >>> plt.show()
        指定数据点和波形序号
        >>> fig, ax = plot_waveform(waveform, point_index=5, waveform_index=2)
        >>> plt.show()
        ```
    """
    logger.info("开始绘制Waveform时域波形图")

    # 验证输入
    if waveform.size == 0:
        logger.error("输入的Waveform对象为空，无法绘图")
        raise ValueError("Waveform对象不能为空")

    if zoom_factor <= 0:
        logger.error(f"zoom_factor必须大于0，当前值: {zoom_factor}")
        raise ValueError("zoom_factor必须大于0")

    logger.info(
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
        logger.warning(
            f"zoom_factor={zoom_factor}过大，导致绘制样本数不足2个，"
            f"将调整为绘制2个采样点"
        )
        samples_to_plot = 2

    # 计算实际绘制的时长
    duration_to_plot = samples_to_plot / waveform.sampling_rate

    logger.debug(
        f"zoom_factor={zoom_factor}, 总样本数={total_samples}, "
        f"绘制样本数={samples_to_plot}, 绘制时长={duration_to_plot:.6f}s"
    )

    # 2. 生成时间轴（仅包含需要绘制的部分）
    time_array = np.linspace(0, duration_to_plot, samples_to_plot, endpoint=False)

    logger.debug(
        f"时间轴范围: [0, {duration_to_plot:.6f}]s, 采样点数: {samples_to_plot}"
    )

    # 3. 创建图形
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # 4. 绘制波形（仅绘制需要的部分）
    if waveform.ndim == 1:
        # 单通道波形
        ax.plot(
            time_array,
            waveform[:samples_to_plot],
            linewidth=1.0,
            color="blue",
            label="通道 1",
        )
        logger.debug("绘制单通道波形")
    else:
        # 多通道波形
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
        logger.debug(f"绘制 {waveform.channels_num} 个通道的波形")

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

    logger.debug("波形图绘制完成")

    # 8. 调整布局
    plt.tight_layout()

    # 9. 保存图片（如果指定了路径）
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"波形图已保存至: {save_path}")

    logger.info("Waveform时域波形图绘制完成")

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
    from datetime import datetime
    from pathlib import Path

    logger.info("开始批量绘制SweepData波形图")

    # 验证输入
    ai_data_list = sweep_data["ai_data_list"]
    if not ai_data_list:
        logger.error("输入的SweepData对象为空，无法绘图")
        raise ValueError("SweepData对象不能为空")

    # 创建输出目录
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        logger.info(f"输出目录不存在，创建目录: {output_dir_path}")
        output_dir_path.mkdir(parents=True, exist_ok=True)

    # 创建带时间戳的子文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"waveforms_{timestamp}"
    output_folder = output_dir_path / folder_name

    try:
        output_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"创建输出文件夹: {output_folder}")
    except OSError as e:
        logger.error(f"无法创建输出文件夹: {e}", exc_info=True)
        raise

    # 统计总波形数
    total_waveforms = sum(len(point_data["ai_data"]) for point_data in ai_data_list)
    logger.info(f"开始绘制 {len(ai_data_list)} 个数据点的共 {total_waveforms} 个波形")

    # 遍历每个数据点
    waveform_count = 0
    for point_idx, point_data in enumerate(ai_data_list):
        position = point_data["position"]
        ai_waveforms = point_data["ai_data"]

        logger.debug(
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
                    logger.info(
                        f"进度: {waveform_count}/{total_waveforms} "
                        f"({waveform_count / total_waveforms * 100:.1f}%)"
                    )

            except Exception as e:
                logger.error(
                    f"绘制数据点 {point_idx} 的波形 {waveform_idx} 时发生错误: {e}",
                    exc_info=True,
                )
                # 继续处理下一个波形
                continue

    logger.info(f"批量绘制完成，成功绘制 {waveform_count}/{total_waveforms} 个波形图")
    logger.info(f"所有波形图已保存至: {output_folder}")

    return str(output_folder)
