"""gainEP项目的COMSOL仿真函数

此模块提供用于gainEP项目的COMSOL仿真自动化函数，包括参数扫描设置、
仿真运行和探针数据提取。
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mph
import numpy as np
import pyvista as pv


@dataclass
class ScatteringMatrix:
    """散射矩阵数据容器

    封装四个探针的散射矩阵元数据，对应不同的入射和反射方向。

    Attributes:
        S11: 左入射的镜面反射系数 (bnd1探针数据)
        S21: 左入射的异常/逆反射系数 (bnd2探针数据)
        S12: 右入射的异常/逆反射系数 (bnd3探针数据)
        S22: 右入射的镜面反射系数 (bnd4探针数据)
        cr_values: cr参数的扫描值数组
        ci_values: ci参数的扫描值数组

    Notes:
        - 所有S矩阵元均为复数数组，形状为 (n_params, n_spatial_points)
        - n_params: 参数扫描点数量（cr和ci的组合）
        - n_spatial_points: 边界上的空间积分点数量
    """

    S11: np.ndarray  # bnd1: 左入射镜面反射
    S21: np.ndarray  # bnd2: 左入射异常反射
    S12: np.ndarray  # bnd3: 右入射异常反射
    S22: np.ndarray  # bnd4: 右入射镜面反射
    cr_values: np.ndarray  # cr参数扫描值
    ci_values: np.ndarray  # ci参数扫描值


def run_gain_ep_simulation(
    mph_file: Path,
    cr_range: tuple[float, float, float],
    ci_range: tuple[float, float, float],
    save_path: Path,
    freq: float = 3430.0,
    client: mph.Client | None = None,
) -> mph.Client:
    """运行gainEP参数扫描仿真并保存散射矩阵数据到硬盘

    此函数加载COMSOL模型，设置cr和ci参数的扫描范围，运行仿真，
    提取四个边界探针的数据作为散射矩阵元，并保存到指定路径。

    连接管理策略：
    - 如果传入client=None，函数会创建新的COMSOL连接，并在返回时保留该连接
    - 如果传入了client对象，函数会复用该连接，并在返回时保留该连接
    - 调用者负责在所有仿真完成后手动断开连接（client.disconnect()）

    Args:
        mph_file: COMSOL模型文件路径 (.mph文件)
        cr_range: cr参数范围，格式为 (start, step, end)
        ci_range: ci参数范围，格式为 (start, step, end)
        save_path: 保存散射矩阵数据的路径 (.npz文件)
        freq: 频率值（Hz），默认为3430.0
        client: MPh Client对象，如果为None则自动创建新连接

    Returns:
        mph.Client: MPh Client对象，可用于后续仿真复用

    Raises:
        FileNotFoundError: 如果mph_file不存在
        RuntimeError: 如果COMSOL连接或仿真失败

    Examples:
        单次仿真（自动创建连接）:
        >>> from pathlib import Path
        >>> mph_file = Path("mphs/gainEP/gainEP_basic.mph")
        >>> cr_range = (1.003, 0.001, 1.005)
        >>> ci_range = (-0.074, 0.001, -0.072)
        >>> save_path = Path("storage/gainEP/data/simulation_result.npz")
        >>> client = run_gain_ep_simulation(mph_file, cr_range, ci_range, save_path)
        >>> client.disconnect()  # 手动断开连接

        多次仿真（复用连接）:
        >>> client = run_gain_ep_simulation(mph_file, cr_range1, ci_range1, save_path1, client=None)
        >>> client = run_gain_ep_simulation(mph_file, cr_range2, ci_range2, save_path2, client=client)
        >>> client = run_gain_ep_simulation(mph_file, cr_range3, ci_range3, save_path3, client=client)
        >>> client.disconnect()  # 所有仿真完成后断开
    """
    # 验证文件存在
    if not mph_file.exists():
        raise FileNotFoundError(f"COMSOL模型文件不存在: {mph_file}")

    # 连接COMSOL Server（如果需要）
    if client is None:
        try:
            client = mph.start()
        except Exception as e:
            raise RuntimeError(f"无法连接到COMSOL Server: {e}")

    try:
        # 加载模型
        model = client.load(str(mph_file))

        # 设置参数扫描范围
        # COMSOL的range格式: 'range(start, step, end)'
        cr_str = f"range({cr_range[0]},{cr_range[1]},{cr_range[2]})"
        ci_str = f"range({ci_range[0]},{ci_range[1]},{ci_range[2]})"
        freq_str = str(freq)

        # 通过Java API设置参数扫描
        # 找到参数扫描求解器特征（通常是'p1'）
        java_model = model.java
        solver = java_model.sol("sol1")

        # 设置参数扫描列表
        # 根据MATLAB代码，参数扫描在's1'或's2'的'p1'特征中
        # 我们需要找到正确的求解器步骤
        try:
            # 尝试访问's1'的参数扫描
            param_sweep = solver.feature("s1").feature("p1")
            param_sweep.set("plistarr", [freq_str, cr_str, ci_str])
        except:
            # 如果's1'不存在，尝试's2'
            try:
                param_sweep = solver.feature("s2").feature("p1")
                param_sweep.set("plistarr", [freq_str, cr_str, ci_str])
            except Exception as e:
                raise RuntimeError(f"无法设置参数扫描: {e}")

        # 运行仿真
        model.solve()

        # 从探针结果表中提取数据
        # 所有探针的结果存储在同一个表中（通常是'tbl2'）
        # 获取第一个探针的表名称
        probe = java_model.probe("bnd1")
        table_name = probe.getString("table")

        # 获取表数据
        result_table = java_model.result().table(table_name)
        table_data = result_table.getTableData(True)  # True表示包含标题

        # 解析复数字符串
        def parse_complex(s):
            """解析COMSOL的复数字符串格式: 'real+imagi'"""
            s = str(s).strip()
            s = s.replace("i", "j")
            return complex(s)

        # 提取探针数据
        # 表结构: [freq, cr, ci, bnd1, bnd2, bnd3, bnd4]
        probe_data = {
            "bnd1": [],
            "bnd2": [],
            "bnd3": [],
            "bnd4": [],
        }

        for row in table_data:
            if len(row) >= 7:
                probe_data["bnd1"].append(parse_complex(row[3]))
                probe_data["bnd2"].append(parse_complex(row[4]))
                probe_data["bnd3"].append(parse_complex(row[5]))
                probe_data["bnd4"].append(parse_complex(row[6]))

        # 转换为numpy数组
        for probe_name in probe_data:
            probe_data[probe_name] = np.array(probe_data[probe_name])

        # 生成参数值数组
        cr_values = np.arange(cr_range[0], cr_range[2] + cr_range[1] / 2, cr_range[1])
        ci_values = np.arange(ci_range[0], ci_range[2] + ci_range[1] / 2, ci_range[1])

        # 保存数据到硬盘
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path,
            S11=probe_data["bnd1"],
            S21=probe_data["bnd2"],
            S12=probe_data["bnd3"],
            S22=probe_data["bnd4"],
            cr_values=cr_values,
            ci_values=ci_values,
        )

        print(f"仿真数据已保存到: {save_path}")

        return client

    except Exception:
        # 如果发生错误，重新抛出异常（不断开连接，让调用者决定）
        raise


def load_scattering_matrix(data_path: Path) -> ScatteringMatrix:
    """从硬盘加载散射矩阵数据

    Args:
        data_path: 散射矩阵数据文件路径 (.npz文件)

    Returns:
        ScatteringMatrix对象

    Raises:
        FileNotFoundError: 如果数据文件不存在

    Examples:
        >>> from pathlib import Path
        >>> data_path = Path("storage/gainEP/data/simulation_result.npz")
        >>> s_matrix = load_scattering_matrix(data_path)
    """
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    data = np.load(data_path)
    return ScatteringMatrix(
        S11=data["S11"],
        S21=data["S21"],
        S12=data["S12"],
        S22=data["S22"],
        cr_values=data["cr_values"],
        ci_values=data["ci_values"],
    )


def plot_scattering_matrix_2d(
    data_path: Path,
    save_path: Path | None = None,
    dpi: int = 150,
    cmap: str = "viridis",
) -> plt.Figure:
    """绘制散射矩阵元在二维参数空间(cr, ci)中的幅值分布

    创建2x2子图布局，按照矩阵形式排列四个散射矩阵元的幅值热图：
    - 左上: S11 (左入射镜面反射)
    - 右上: S12 (右入射异常反射)
    - 左下: S21 (左入射异常反射)
    - 右下: S22 (右入射镜面反射)

    每个子图使用独立的colorbar以适应不同数量级的数据。

    Args:
        data_path: 散射矩阵数据文件路径 (.npz文件)
        save_path: 保存图像的路径，如果为None则不保存
        dpi: 图像分辨率，默认150
        cmap: colormap名称，默认'viridis'

    Returns:
        matplotlib Figure对象

    Notes:
        - 横轴为cr参数，纵轴为ci参数
        - 颜色表示复数的绝对值（幅值）
        - 每个子图对空间点求平均后绘制

    Examples:
        >>> from pathlib import Path
        >>> data_path = Path("storage/gainEP/data/simulation_result.npz")
        >>> fig = plot_scattering_matrix_2d(data_path, save_path=Path("output.png"))
        >>> plt.show()
    """
    # 加载数据
    s_matrix = load_scattering_matrix(data_path)

    # 计算幅值
    # 数据形状: (n_params,) - 每个参数点一个值（探针已经积分）
    s11_abs = np.abs(s_matrix.S11)
    s12_abs = np.abs(s_matrix.S12)
    s21_abs = np.abs(s_matrix.S21)
    s22_abs = np.abs(s_matrix.S22)

    # 重塑为二维网格 (n_cr, n_ci)
    # COMSOL的filled扫描顺序：最后一个参数(ci)变化最快
    # 所以数据顺序是：对于每个cr，ci从小到大变化
    n_cr = len(s_matrix.cr_values)
    n_ci = len(s_matrix.ci_values)

    # reshape时，ci是列（变化快），cr是行（变化慢）
    s11_2d = s11_abs.reshape(n_cr, n_ci)
    s12_2d = s12_abs.reshape(n_cr, n_ci)
    s21_2d = s21_abs.reshape(n_cr, n_ci)
    s22_2d = s22_abs.reshape(n_cr, n_ci)

    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 定义子图数据和标题（按矩阵形式排列）
    plot_data = [
        (s11_2d, "S11 (左入射镜面反射)", axes[0, 0]),
        (s12_2d, "S12 (右入射异常反射)", axes[0, 1]),
        (s21_2d, "S21 (左入射异常反射)", axes[1, 0]),
        (s22_2d, "S22 (右入射镜面反射)", axes[1, 1]),
    ]

    # 绘制每个子图
    for data, title, ax in plot_data:
        # 使用pcolormesh绘制热图
        im = ax.pcolormesh(
            s_matrix.cr_values,
            s_matrix.ci_values,
            data.T,  # 转置以匹配坐标轴
            cmap=cmap,
            shading="auto",
        )

        ax.set_xlabel("cr")
        ax.set_ylabel("ci")
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")

        # 添加独立的colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("幅值")

    # 设置总标题
    fig.suptitle("散射矩阵元幅值分布 (参数空间)", fontsize=14, y=0.995)
    plt.tight_layout()

    # 保存图像
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def run_10in16out_simulation(
    mph_file: Path,
    pamp_L: float,
    pamp_R: float,
    vn_1: complex,
    vn_2: complex,
    vn_3: complex,
    vn_4: complex,
    vn_5: complex,
    vn_6: complex,
    vn_7: complex,
    vn_8: complex,
    client: mph.Client | None = None,
) -> tuple[
    complex,
    complex,
    complex,
    complex,
    complex,
    complex,
    complex,
    complex,
    complex,
    complex,
    complex,
    complex,
    complex,
    complex,
    complex,
    complex,
    mph.Client,
]:
    """运行gainEP_10in16out仿真并返回16个域探针结果

    此函数加载COMSOL模型，设置10个输入参数（2个背景压力场幅值 + 8个法向位移），
    运行仿真，提取16个域探针的复数结果。

    连接管理策略：
    - 如果传入client=None，函数会创建新的COMSOL连接，并在返回时保留该连接
    - 如果传入了client对象，函数会复用该连接，并在返回时保留该连接
    - 调用者负责在所有仿真完成后手动断开连接（client.disconnect()）

    Args:
        mph_file: COMSOL模型文件路径 (.mph文件)
        pamp_L: 背景压力场 L 的压力幅值（实数）
        pamp_R: 背景压力场 R 的压力幅值（实数）
        vn_1: 法向位移 1 的向内位移（复数）
        vn_2: 法向位移 2 的向内位移（复数）
        vn_3: 法向位移 3 的向内位移（复数）
        vn_4: 法向位移 4 的向内位移（复数）
        vn_5: 法向位移 5 的向内位移（复数）
        vn_6: 法向位移 6 的向内位移（复数）
        vn_7: 法向位移 7 的向内位移（复数）
        vn_8: 法向位移 8 的向内位移（复数）
        client: MPh Client对象，如果为None则自动创建新连接

    Returns:
        tuple: 包含17个元素的元组：
            - 前16个元素：域探针 dom1 到 dom16 的复数结果
            - 第17个元素：mph.Client对象，可用于后续仿真复用

    Raises:
        FileNotFoundError: 如果mph_file不存在
        RuntimeError: 如果COMSOL连接或仿真失败

    Examples:
        单次仿真（自动创建连接）:
        >>> from pathlib import Path
        >>> mph_file = Path("mphs/gainEP/gainEP_10in16out.mph")
        >>> results = run_10in16out_simulation(
        ...     mph_file, pamp_L=1.0, pamp_R=1.0,
        ...     vn_1=0j, vn_2=0j, vn_3=0j, vn_4=0j,
        ...     vn_5=0j, vn_6=0j, vn_7=0j, vn_8=0j
        ... )
        >>> dom1, dom2, ..., dom16, client = results
        >>> client.disconnect()  # 手动断开连接

        多次仿真（复用连接）:
        >>> results1 = run_10in16out_simulation(mph_file, 1.0, 1.0, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, client=None)
        >>> *probe_results1, client = results1
        >>> results2 = run_10in16out_simulation(mph_file, 2.0, 2.0, 1j, 1j, 1j, 1j, 1j, 1j, 1j, 1j, client=client)
        >>> *probe_results2, client = results2
        >>> client.disconnect()  # 所有仿真完成后断开
    """
    # 验证文件存在
    if not mph_file.exists():
        raise FileNotFoundError(f"COMSOL模型文件不存在: {mph_file}")

    # 连接COMSOL Server（如果需要）
    if client is None:
        try:
            client = mph.start()
        except Exception as e:
            raise RuntimeError(f"无法连接到COMSOL Server: {e}")

    try:
        # 加载模型
        model = client.load(str(mph_file))

        # 获取Java模型对象
        java_model = model.java

        # 设置背景压力场幅值
        # bpf1: 背景压力场 L
        # bpf2: 背景压力场 R
        java_model.physics("acpr").feature("bpf1").set("pamp", pamp_L)
        java_model.physics("acpr").feature("bpf2").set("pamp", pamp_R)

        # 设置法向位移（复数值）
        # ndisp1 到 ndisp8 对应 8 个法向位移边界条件
        vn_list = [vn_1, vn_2, vn_3, vn_4, vn_5, vn_6, vn_7, vn_8]
        for i, vn in enumerate(vn_list, start=1):
            # COMSOL中复数格式为 "real+imag*i"
            vn_str = f"{vn.real}+{vn.imag}*i"
            java_model.physics("acpr").feature(f"ndisp{i}").set("ndisp", vn_str)

        # 运行仿真
        model.solve()

        # 从探针结果表中提取数据
        # 所有探针的结果存储在同一个表中（tbl5）
        # 获取第一个探针的表名称
        probe = java_model.probe("dom1")
        table_name = probe.getString("table")

        # 获取表数据
        result_table = java_model.result().table(table_name)
        table_data = result_table.getTableData(True)  # True表示包含标题

        # 解析复数字符串
        def parse_complex(s):
            """解析COMSOL的复数字符串格式: 'real+imagi'"""
            s = str(s).strip()
            s = s.replace("i", "j")
            return complex(s)

        # 提取16个域探针的数据
        # 表结构: [freq, dom1, dom2, ..., dom16]
        # 注意：表数据可能没有标题行，直接就是数据
        probe_results = []

        if len(table_data) < 1:
            raise RuntimeError("探针表数据为空，无法提取结果")

        # 取第一行数据（单次仿真）
        row = table_data[0]

        # 检查列数（应该是17列：1个频率 + 16个探针）
        if len(row) < 17:
            raise RuntimeError(f"表数据列数不足，期望至少17列，实际{len(row)}列")

        # 提取16个探针的值（索引1到16，索引0是频率）
        for i in range(1, 17):
            probe_results.append(parse_complex(row[i]))

        if len(probe_results) != 16:
            raise RuntimeError(
                f"提取的探针数据数量不正确，期望16个，实际{len(probe_results)}个"
            )

        # 返回16个探针结果 + client
        return (*probe_results, client)

    except Exception:
        # 如果发生错误，重新抛出异常（不断开连接，让调用者决定）
        raise


def plot_eigenvalues_3d(
    data_path: Path,
    mode: str = "Re",
    opacity: float = 1.0,
    save_path: Path | None = None,
    screenshot_kwargs: dict | None = None,
) -> pv.Plotter:
    """绘制S矩阵特征值在参数空间中的三维曲面图

    在(cr, ci)参数空间中绘制S矩阵的两个复特征值的实部或虚部曲面。
    根据mode参数选择显示：
    - mode='Re': 显示两个特征值的实部曲面（蓝色和红色）
    - mode='Im': 显示两个特征值的虚部曲面（青色和橙色）

    Args:
        data_path: 散射矩阵数据文件路径 (.npz文件)
        mode: 显示模式，'Re'显示实部，'Im'显示虚部，默认为'Re'
        opacity: 曲面透明度，范围0-1，默认为1.0（完全不透明）
        save_path: 保存图像的路径，如果为None则不保存
        screenshot_kwargs: 传递给plotter.screenshot()的参数字典

    Returns:
        PyVista Plotter对象

    Raises:
        ValueError: 如果mode不是'Re'或'Im'，或opacity不在0-1范围内

    Notes:
        - 对每个参数点，计算2x2 S矩阵的特征值
        - S矩阵由空间平均的矩阵元构成
        - 使用PyVista绘制交互式三维曲面

    Examples:
        显示实部曲面:
        >>> from pathlib import Path
        >>> data_path = Path("storage/gainEP/data/simulation_result.npz")
        >>> plotter = plot_eigenvalues_3d(data_path, mode='Re', save_path=Path("eigenvals_re.png"))
        >>> plotter.show()

        显示半透明虚部曲面:
        >>> plotter = plot_eigenvalues_3d(data_path, mode='Im', opacity=0.5, save_path=Path("eigenvals_im.png"))
        >>> plotter.show()
    """
    # 验证mode参数
    if mode not in ["Re", "Im"]:
        raise ValueError(f"mode必须是'Re'或'Im'，当前值为'{mode}'")

    # 验证opacity参数
    if not 0 <= opacity <= 1:
        raise ValueError(f"opacity必须在0-1范围内，当前值为{opacity}")

    # 加载数据
    s_matrix = load_scattering_matrix(data_path)

    # 计算每个参数点的平均矩阵元（对空间点求平均）
    # 如果数据是一维的，直接使用；如果是二维的，对空间维度求平均
    if s_matrix.S11.ndim == 1:
        # 一维数据：每个参数点只有一个值
        s11_avg = s_matrix.S11
        s12_avg = s_matrix.S12
        s21_avg = s_matrix.S21
        s22_avg = s_matrix.S22
    else:
        # 二维数据：对空间维度（axis=1）求平均
        s11_avg = np.mean(s_matrix.S11, axis=1)
        s12_avg = np.mean(s_matrix.S12, axis=1)
        s21_avg = np.mean(s_matrix.S21, axis=1)
        s22_avg = np.mean(s_matrix.S22, axis=1)

    # 计算每个参数点的特征值
    n_params = len(s11_avg)
    eigenvals = np.zeros((n_params, 2), dtype=complex)

    for i in range(n_params):
        # 构建2x2 S矩阵
        S = np.array([[s11_avg[i], s12_avg[i]], [s21_avg[i], s22_avg[i]]])
        # 计算特征值
        eigs = np.linalg.eigvals(S)
        eigenvals[i] = eigs

    # 提取实部和虚部
    eig1_real = eigenvals[:, 0].real
    eig1_imag = eigenvals[:, 0].imag
    eig2_real = eigenvals[:, 1].real
    eig2_imag = eigenvals[:, 1].imag

    # 重塑为二维网格
    # COMSOL的filled扫描顺序：最后一个参数(ci)变化最快
    n_cr = len(s_matrix.cr_values)
    n_ci = len(s_matrix.ci_values)

    eig1_real_2d = eig1_real.reshape(n_cr, n_ci)
    eig1_imag_2d = eig1_imag.reshape(n_cr, n_ci)
    eig2_real_2d = eig2_real.reshape(n_cr, n_ci)
    eig2_imag_2d = eig2_imag.reshape(n_cr, n_ci)

    # 创建网格坐标
    CR, CI = np.meshgrid(s_matrix.cr_values, s_matrix.ci_values, indexing="ij")

    # 计算坐标轴范围，用于自适应缩放
    cr_range = s_matrix.cr_values.max() - s_matrix.cr_values.min()
    ci_range = s_matrix.ci_values.max() - s_matrix.ci_values.min()

    # 合并所有特征值数据以计算z轴范围
    all_eigs = np.concatenate(
        [
            eig1_real_2d.flatten(),
            eig1_imag_2d.flatten(),
            eig2_real_2d.flatten(),
            eig2_imag_2d.flatten(),
        ]
    )
    eig_range = all_eigs.max() - all_eigs.min()

    # 计算缩放因子，使三个轴的显示比例接近
    # 选择最大范围作为参考
    max_range = max(cr_range, ci_range, eig_range)

    # 如果某个轴的范围太小，使用适当的缩放
    cr_scale = max_range / cr_range if cr_range > 0 else 1.0
    ci_scale = max_range / ci_range if ci_range > 0 else 1.0
    eig_scale = max_range / eig_range if eig_range > 0 else 1.0

    # 应用缩放（只缩放坐标，不改变数据值）
    CR_scaled = CR * cr_scale
    CI_scaled = CI * ci_scale
    eig1_real_2d_scaled = eig1_real_2d * eig_scale
    eig1_imag_2d_scaled = eig1_imag_2d * eig_scale
    eig2_real_2d_scaled = eig2_real_2d * eig_scale
    eig2_imag_2d_scaled = eig2_imag_2d * eig_scale

    # 创建PyVista plotter（如果需要保存截图，使用off_screen模式）
    plotter = pv.Plotter(off_screen=(save_path is not None))

    # 根据mode选择要显示的曲面和颜色映射
    if mode == "Re":
        # 显示实部曲面，用虚部数据作为颜色映射
        surfaces = [
            (
                CR_scaled,
                CI_scaled,
                eig1_real_2d_scaled,
                eig1_imag_2d,
                "λ₁ 实部",
                "λ₁ 虚部",
            ),
            (
                CR_scaled,
                CI_scaled,
                eig2_real_2d_scaled,
                eig2_imag_2d,
                "λ₂ 实部",
                "λ₂ 虚部",
            ),
        ]
        cmap = "coolwarm"  # 适合表示正负值的colormap
        # 计算统一的颜色映射范围（虚部数据）
        all_scalars = np.concatenate([eig1_imag_2d.flatten(), eig2_imag_2d.flatten()])
    else:  # mode == 'Im'
        # 显示虚部曲面，用实部数据作为颜色映射
        surfaces = [
            (
                CR_scaled,
                CI_scaled,
                eig1_imag_2d_scaled,
                eig1_real_2d,
                "λ₁ 虚部",
                "λ₁ 实部",
            ),
            (
                CR_scaled,
                CI_scaled,
                eig2_imag_2d_scaled,
                eig2_real_2d,
                "λ₂ 虚部",
                "λ₂ 实部",
            ),
        ]
        cmap = "viridis"  # 适合表示单调变化的colormap
        # 计算统一的颜色映射范围（实部数据）
        all_scalars = np.concatenate([eig1_real_2d.flatten(), eig2_real_2d.flatten()])

    # 计算统一的颜色范围
    clim = [all_scalars.min(), all_scalars.max()]

    # 添加每个曲面
    for i, (x, y, z, scalars, label, scalar_label) in enumerate(surfaces):
        # 创建结构化网格
        grid = pv.StructuredGrid(x, y, z)

        # 将scalars数据添加到网格（需要展平为一维数组）
        grid[scalar_label] = scalars.T.ravel()  # 转置以匹配网格点的顺序

        # 添加到plotter，使用scalars进行颜色映射
        # 注意：不使用label参数，因为我们使用scalar bar而不是图例
        # 只在第一个曲面显示scalar bar，确保两个曲面使用统一的颜色范围
        plotter.add_mesh(
            grid,
            scalars=scalar_label,
            cmap=cmap,
            clim=clim,
            opacity=opacity,
            smooth_shading=True,
            show_scalar_bar=(i == 0),  # 只在第一个曲面显示scalar bar
        )

    # 设置坐标轴标签
    mode_text = "实部" if mode == "Re" else "虚部"
    axis_labels = {"xlabel": "cr", "ylabel": "ci", "zlabel": f"特征值{mode_text}"}

    plotter.add_axes(**axis_labels)
    plotter.show_grid(**axis_labels)

    # 设置相机视角
    plotter.camera_position = "iso"

    # 添加标题和说明
    plotter.add_text(
        f"S矩阵特征值{mode_text} (参数空间)\n注: 坐标已缩放以优化显示",
        position="upper_edge",
        font_size=10,
        color="black",
    )

    # 保存截图（在off_screen模式下）
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs = screenshot_kwargs or {"window_size": [1920, 1080]}
        # 在off_screen模式下，show()会渲染并返回图像
        plotter.show(screenshot=str(save_path), **kwargs)

    return plotter
