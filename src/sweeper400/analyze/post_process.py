# pyright: basic, reportMissingImports=false
"""
# 后处理模块

模块路径：`sweeper400.analyze.post_process`

本模块提供对Sweeper类采集到的原始数据进行后处理的功能。
主要包含传递函数计算等数据分析功能。
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata

from sweeper400.logger import get_logger

from .basic_sine import extract_single_tone_information_vvi

# , TYPE_CHECKING
from .my_dtypes import PointTFData, SweepData, Waveform

# 配置matplotlib中文字体支持
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Microsoft JhengHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# 获取模块日志器
logger = get_logger(__name__)


def calculate_transfer_function(
    sweep_data: SweepData,
) -> list[PointTFData]:
    """
    计算Sweeper采集数据的传递函数

    对每个测量点的原始数据进行处理，计算输入输出信号的传递函数。
    具体步骤：
    1. 对每个点的多个AI波形chunks进行按位相加并取平均，减少随机噪声
    2. 使用extract_single_tone_information_vvi提取AI信号的正弦波参数
    3. 使用共用的AO波形的正弦波参数
    4. 计算传递函数：幅值比 = AI幅值 / AO幅值，相位差 = AI相位 - AO相位

    Args:
        sweep_data: Sweeper采集的完整测量数据，包含ai_data_list和ao_data

    Returns:
        传递函数结果列表，每个元素包含位置、幅值比和相位差信息

    Raises:
        ValueError: 当输入数据为空或格式不正确时
        RuntimeError: 当AO数据没有sine_args属性时

    Examples:
        ```python
        >>> # 假设已有采集的原始数据
        >>> sweep_data = sweeper.get_data()
        >>> tf_results = calculate_transfer_function(sweep_data)
        >>> for result in tf_results:
        ...     print(
        ...         f"位置: {result['position']}, 幅值比: {result['amp_ratio']:.4f}, "
        ...         f"相位差: {result['phase_shift']:.4f}"
        ...     )
        ```
    """
    ai_data_list = sweep_data["ai_data_list"]
    ao_data = sweep_data["ao_data"]

    logger.info(f"开始计算传递函数，共 {len(ai_data_list)} 个测量点")

    # 验证输入数据
    if not ai_data_list:
        logger.error("输入数据列表为空")
        raise ValueError("输入数据列表不能为空")

    # 验证AO数据的sine_args
    if ao_data.sine_args is None:
        logger.error("AO波形没有sine_args属性")
        raise RuntimeError("AO波形必须包含sine_args属性")

    ao_sine_args = ao_data.sine_args
    logger.debug(
        f"使用AO波形参数: 频率={ao_sine_args['frequency']:.2f}Hz, "
        f"幅值={ao_sine_args['amplitude']:.4f}"
    )

    # 存储结果
    results: list[PointTFData] = []

    # 遍历每个测量点
    for point_idx, point_data in enumerate(ai_data_list):
        # 只在处理较少点数时或每10个点输出一次进度信息
        if len(ai_data_list) <= 20 or (point_idx + 1) % 10 == 0:
            logger.debug(
                f"处理第 {point_idx + 1}/{len(ai_data_list)} 个点: "
                f"{point_data['position']}"
            )

        try:
            # 1. 处理AI数据：按位相加并取平均
            ai_waveforms = point_data["ai_data"]
            if not ai_waveforms:
                logger.warning(f"点 {point_idx} 的AI数据为空，跳过该点")
                continue

            # 获取采样率和波形长度（所有波形应该有相同的采样率和长度）
            sampling_rate = ai_waveforms[0].sampling_rate
            waveform_length = ai_waveforms[0].samples_num

            # 验证所有波形的长度是否一致
            for i, wf in enumerate(ai_waveforms):
                if wf.samples_num != waveform_length:
                    logger.warning(
                        f"点 {point_idx} 的第 {i} 个波形长度不一致 "
                        f"(期望: {waveform_length}, 实际: {wf.samples_num})，跳过该点"
                    )
                    break
            else:
                # 如果所有波形长度都一致，进行按位相加并取平均
                logger.debug(
                    f"点 {point_idx} 有 {len(ai_waveforms)} 个chunks，"
                    f"每个长度 {waveform_length}"
                )

                # 将所有波形数据按位相加
                summed_data = np.zeros(waveform_length, dtype=np.float64)
                for wf in ai_waveforms:
                    # 处理多通道数据，只使用第一个通道
                    if wf.ndim == 2:
                        summed_data += wf[0, :]
                    else:
                        summed_data += wf

                # 取平均
                averaged_data = summed_data / len(ai_waveforms)

                # 创建平均后的Waveform对象
                ai_averaged_waveform = Waveform(
                    input_array=averaged_data,
                    sampling_rate=sampling_rate,
                    timestamp=ai_waveforms[0].timestamp,  # 使用第一个波形的时间戳
                )

                logger.debug(
                    f"点 {point_idx} 完成 {len(ai_waveforms)} 个chunks的按位平均"
                )

                # 2. 提取AI信号的正弦波参数
                ai_sine_args = extract_single_tone_information_vvi(ai_averaged_waveform)

                # 3. 计算传递函数
                # 幅值比 = AI幅值 / AO幅值
                amp_ratio = ai_sine_args["amplitude"] / ao_sine_args["amplitude"]

                # 相位差 = AI相位 - AO相位（弧度制）
                phase_shift = ai_sine_args["phase"] - ao_sine_args["phase"]

                # 将相位差归一化到 [-π, π] 区间
                phase_shift = np.arctan2(np.sin(phase_shift), np.cos(phase_shift))

                # 4. 存储结果
                result: PointTFData = {
                    "position": point_data["position"],
                    "amp_ratio": float(amp_ratio),
                    "phase_shift": float(phase_shift),
                }
                results.append(result)

        except Exception as e:
            logger.error(f"处理点 {point_idx} 时发生错误: {e}", exc_info=True)
            # 继续处理下一个点
            continue

    logger.info(f"传递函数计算完成，成功处理 {len(results)}/{len(ai_data_list)} 个点")

    return results


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
        >>> tf_results = calculate_transfer_function(raw_data)
        >>> fig, (ax1, ax2) = plot_transfer_function_discrete_distribution(tf_results)
        >>> plt.show()
        >>> # 或者保存图片
        >>> fig, axes = plot_transfer_function_discrete_distribution(
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
        >>> tf_results = calculate_transfer_function(raw_data)
        >>> fig, (ax1, ax2) = plot_transfer_function_interpolated_distribution(
        ...     tf_results
        ... )
        >>> plt.show()
        >>> # 或者保存图片并使用线性插值
        >>> fig, axes = plot_transfer_function_interpolated_distribution(
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
        >>> tf_results = calculate_transfer_function(raw_data)
        >>> fig, ax = plot_transfer_function_instantaneous_field(tf_results)
        >>> plt.show()
        >>> # 或者保存图片并使用线性插值
        >>> fig, ax = plot_transfer_function_instantaneous_field(
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
