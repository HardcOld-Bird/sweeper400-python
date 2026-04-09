"""
# 后处理模块

模块路径：`sweeper400.analyze.post_process`

本模块提供对Sweeper类采集到的原始数据进行后处理的功能。
主要包含按位平均、传递函数计算等数据分析功能。
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from ..logger import get_logger
from .my_dtypes import (
    CompData,
    PointSweepData,
    SineArgs,
    SweepData,
    TFData,
    Waveform,
)

# 获取模块日志器
logger = get_logger(__name__)


# 加载测量数据的工具函数
def load_sweep_data(file_path: str | Path) -> SweepData:
    """
    从文件加载测量数据

    加载由Sweeper.save_data()保存的测量数据。

    Args:
        file_path: 数据文件的路径（.pkl文件）

    Returns:
        SweepData: 包含以下键的字典：
            - "ai_data_list": List[PointSweepData]，每个PointRawData包含：
                - "position": Point2D对象，表示该点的坐标
                - "ai_data": List[Waveform]，该点采集的所有AI波形
            - "ao_data": Waveform，扫场过程中使用的输出波形

    Raises:
        FileNotFoundError: 当文件不存在时
        IOError: 当文件读取失败时
        ValueError: 当数据格式不正确时

    Examples:
        >>> sweep_data = load_sweep_data("sweep_data.pkl")
        >>> ai_data_list = sweep_data["ai_data_list"]
        >>> ao_data = sweep_data["ao_data"]
        >>> print(f"加载了 {len(ai_data_list)} 个点的数据")
        >>> print(f"输出波形采样率: {ao_data.sampling_rate}Hz")
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    logger.info(f"开始加载测量数据: {file_path}")

    try:
        with open(file_path, "rb") as f:
            loaded_data = pickle.load(f)

        # 检查数据格式
        if (
            isinstance(loaded_data, dict)
            and "ai_data_list" in loaded_data
            and "ao_data" in loaded_data
        ):
            logger.info("检测到SweepData格式数据")
            ai_data_list = loaded_data["ai_data_list"]  # type: ignore
            logger.info(f"数据加载成功，共 {len(ai_data_list)} 个点")  # type: ignore
            return loaded_data  # type: ignore
        else:
            raise ValueError(
                "数据格式不正确，期望包含'ai_data_list'和'ao_data'键的字典"
            )

    except Exception as e:
        logger.error(f"数据加载失败: {e}", exc_info=True)
        raise OSError(f"无法从 {file_path} 加载数据: {e}") from e


def average_sweep_data(
    sweep_data: SweepData,
) -> SweepData:
    """
    对SweepData中的所有波形进行按位相加并取平均

    该函数对SweepData中的所有AI波形进行按位相加并取平均，
    以减少随机噪声的影响。支持单通道和多通道波形。

    对于单通道波形（1D数组），返回1D平均波形。
    对于多通道波形（2D数组），对每个通道分别平均，返回2D平均波形。

    Args:
        sweep_data: 原始的扫场测量数据

    Returns:
        处理后的SweepData，结构与输入完全相同，但每个点只有一条AI波形

    Raises:
        ValueError: 当输入数据为空或波形维度不一致时
    """

    # 获取必要参数
    ai_data_list = sweep_data["ai_data_list"]

    if not ai_data_list or not ai_data_list[0]["ai_data"]:
        raise ValueError("输入的SweepData为空")

    first_waveform = ai_data_list[0]["ai_data"][0]
    sampling_rate = first_waveform.sampling_rate
    samples_num = first_waveform.samples_num
    is_multi_channel = first_waveform.ndim == 2

    if is_multi_channel:
        num_channels = first_waveform.shape[0]
        logger.debug(f"检测到多通道波形，通道数: {num_channels}")
    else:
        logger.debug("检测到单通道波形")

    averaged_ai_data_list = []
    # 遍历每个测量点
    for _, point_data in enumerate(ai_data_list):
        # 将所有波形数据按位相加
        ai_waveforms = point_data["ai_data"]

        if is_multi_channel:
            # 多通道情况：对每个通道分别平均
            # 初始化累加数组 (num_channels, samples_num)
            summed_data = np.zeros((num_channels, samples_num), dtype=np.float64)

            for wf in ai_waveforms:
                # 验证波形维度
                if wf.ndim != 2 or wf.shape[0] != num_channels:
                    raise ValueError(
                        f"波形维度不一致：期望 (num_channels={num_channels}, samples_num={samples_num})，"
                        f"实际 shape={wf.shape}"
                    )
                summed_data += wf

            # 取平均
            averaged_data = summed_data / len(ai_waveforms)

            # 创建平均后的Waveform对象（保留channel_names元数据）
            averaged_ai_waveform = Waveform(
                input_array=averaged_data,
                sampling_rate=sampling_rate,
                timestamp=ai_waveforms[0].timestamp,
                channel_names=ai_waveforms[0].channel_names
                if hasattr(ai_waveforms[0], "channel_names")
                else None,
            )
        else:
            # 单通道情况：直接平均
            summed_data = np.zeros(samples_num, dtype=np.float64)

            for wf in ai_waveforms:
                # 验证波形维度
                if wf.ndim == 2:
                    # 如果是2D但只有1个通道，提取第一个通道
                    if wf.shape[0] == 1:
                        summed_data += wf[0, :]
                    else:
                        raise ValueError(
                            f"波形维度不一致：期望单通道，实际 shape={wf.shape}"
                        )
                else:
                    summed_data += wf

            # 取平均
            averaged_data = summed_data / len(ai_waveforms)

            # 创建平均后的Waveform对象
            averaged_ai_waveform = Waveform(
                input_array=averaged_data,
                sampling_rate=sampling_rate,
                timestamp=ai_waveforms[0].timestamp,
            )

        # 创建平均后的点数据
        averaged_point_data: PointSweepData = {
            "position": point_data["position"],
            "ai_data": [averaged_ai_waveform],
        }
        averaged_ai_data_list.append(averaged_point_data)

    # 创建平均后的SweepData
    averaged_sweep_data: SweepData = {
        "ai_data_list": averaged_ai_data_list,
        "ao_data": sweep_data["ao_data"],
    }

    logger.info(
        f"SweepData平均完成，共处理{len(averaged_ai_data_list)}个测量点，"
        f"{'多通道' if is_multi_channel else '单通道'}模式"
    )

    return averaged_sweep_data


def tf_to_comp(
    tf_data: TFData,
    sine_args: SineArgs | None = None,
    mean_amp_ratio: float | None = None,
    mean_phase_shift: float | None = None,
) -> CompData:
    """将传递函数数据转换为补偿数据。

    将 :class:`TFData`（绝对传递函数）转换为 :class:`CompData`（相对于平均值的补偿参数）。

    转换公式：

    - 幅值补偿倍率 = 平均幅值比 / 该通道对幅值比
    - 时间延迟补偿 = (平均相位差 - 该通道对相位差) / (2π × 频率)

    参数:
        tf_data: 传递函数数据（必须为行矩阵或列矩阵，即仅一行或一列）
        sine_args: 正弦波参数（可选，如果为None则使用tf_data中的sine_args）
        mean_amp_ratio: 所有通道对的平均幅值比（可选，如果为None则使用tf_data中的mean_amp_ratio）
        mean_phase_shift: 所有通道对的平均相位差（可选，如果为None则使用tf_data中的mean_phase_shift）

    返回:
        CompData: 补偿数据，包含相对于平均值的幅值补偿倍率和时间延迟补偿值

    异常:
        ValueError: 当频率为 0 或负数，或TFData不是行矩阵/列矩阵时
        ZeroDivisionError: 当某通道对幅值比为 0 时
    """
    # 使用默认值
    if sine_args is None:
        sine_args = tf_data["sine_args"]
    if mean_amp_ratio is None:
        mean_amp_ratio = tf_data["mean_amp_ratio"]
    if mean_phase_shift is None:
        mean_phase_shift = tf_data["mean_phase_shift"]

    frequency = sine_args["frequency"]

    if frequency <= 0:
        logger.error(f"频率必须为正数，收到: {frequency}")
        raise ValueError(f"频率必须为正数，收到: {frequency}")

    # 验证TFData的形状：必须为行矩阵（1行N列）或列矩阵（N行1列）
    tf_df = tf_data["tf_dataframe"]
    n_rows, n_cols = tf_df.shape

    if n_rows != 1 and n_cols != 1:
        logger.error(
            f"TFData必须为行矩阵（1行N列）或列矩阵（N行1列），实际形状: ({n_rows}, {n_cols})"
        )
        raise ValueError(
            f"TFData必须为行矩阵（1行N列）或列矩阵（N行1列），实际形状: ({n_rows}, {n_cols})"
        )

    # 从复数传递函数中提取幅值比和相位差
    tf_complex = tf_df.values.flatten()  # 展平为一维数组
    amp_ratios = np.abs(tf_complex)
    phase_shifts = np.angle(tf_complex)

    # 检查是否有幅值比为0的情况
    if np.any(amp_ratios == 0):
        logger.error("存在幅值比为0的通道对")
        raise ZeroDivisionError("存在幅值比为0的通道对")

    # 计算幅值补偿倍率
    amp_multipliers = mean_amp_ratio / amp_ratios

    # 计算相位差补偿
    phase_shift_comp = mean_phase_shift - phase_shifts

    # 将相位差补偿转换为时间延迟补偿（秒）
    time_increments = phase_shift_comp / (2.0 * np.pi * frequency)

    # 确定通道名称列表（行矩阵用columns，列矩阵用index）
    if n_rows == 1:
        # 行矩阵：1行N列，通道名称在columns中
        channel_names = tf_df.columns.tolist()
    else:
        # 列矩阵：N行1列，通道名称在index中
        channel_names = tf_df.index.tolist()

    # 构建CompData的DataFrame
    comp_df = pd.DataFrame(
        {
            "amp_multiplier": amp_multipliers,
            "time_increment": time_increments,
        },
        index=channel_names,
    )

    # 创建CompData
    comp_data: CompData = {
        "comp_dataframe": comp_df,
        "sampling_info": tf_data["sampling_info"],
        "sine_args": sine_args,
        "mean_amp_ratio": mean_amp_ratio,
        "mean_phase_shift": mean_phase_shift,
    }

    logger.debug(
        f"TFData转换为CompData完成，通道数: {len(channel_names)}, "
        f"平均幅值比: {mean_amp_ratio:.6f}, 平均相位差: {mean_phase_shift:.6f}rad"
    )

    return comp_data


def comp_to_tf(
    comp_data: CompData,
    sine_args: SineArgs | None = None,
    mean_amp_ratio: float | None = None,
    mean_phase_shift: float | None = None,
) -> TFData:
    """将补偿数据转换回传递函数数据。

    这是 :func:`tf_to_comp` 的逆操作：

    - 幅值比 = 平均幅值比 / 幅值补偿倍率
    - 相位差 = 平均相位差 - (时间延迟补偿 × 2π × 频率)

    参数:
        comp_data: 补偿数据（必须为行矩阵或列矩阵）
        sine_args: 正弦波参数（可选，如果为None则使用comp_data中的sine_args）
        mean_amp_ratio: 所有通道对的平均幅值比（可选，如果为None则使用comp_data中的mean_amp_ratio）
        mean_phase_shift: 所有通道对的平均相位差（可选，如果为None则使用comp_data中的mean_phase_shift）

    返回:
        TFData: 传递函数数据，包含绝对幅值比和相位差

    异常:
        ValueError: 当频率为 0 或负数，或CompData不是行矩阵/列矩阵时
        ZeroDivisionError: 当幅值补偿倍率为 0 时
    """
    # 使用默认值
    if sine_args is None:
        sine_args = comp_data["sine_args"]
    if mean_amp_ratio is None:
        mean_amp_ratio = comp_data["mean_amp_ratio"]
    if mean_phase_shift is None:
        mean_phase_shift = comp_data["mean_phase_shift"]

    frequency = sine_args["frequency"]

    if frequency <= 0:
        logger.error(f"频率必须为正数，收到: {frequency}")
        raise ValueError(f"频率必须为正数，收到: {frequency}")

    # 验证CompData的形状（DataFrame应该只有2列：amp_multiplier和time_increment）
    comp_df = comp_data["comp_dataframe"]
    if comp_df.shape[1] != 2:
        logger.error(
            f"CompData的DataFrame应该有2列（amp_multiplier和time_increment），实际列数: {comp_df.shape[1]}"
        )
        raise ValueError(
            f"CompData的DataFrame应该有2列，实际列数: {comp_df.shape[1]}"
        )

    # 提取补偿参数
    amp_multipliers = comp_df["amp_multiplier"].values
    time_increments = comp_df["time_increment"].values

    # 检查是否有幅值补偿倍率为0的情况
    if np.any(amp_multipliers == 0):
        logger.error("存在幅值补偿倍率为0的通道")
        raise ZeroDivisionError("存在幅值补偿倍率为0的通道")

    # 计算幅值比（逆运算）
    amp_ratios = mean_amp_ratio / amp_multipliers

    # 将时间延迟补偿转换为相位差补偿
    phase_shift_comp = time_increments * 2.0 * np.pi * frequency

    # 计算相位差（逆运算）
    phase_shifts = mean_phase_shift - phase_shift_comp

    # 将相位差归一化到 [-π, π] 区间
    phase_shifts = np.arctan2(np.sin(phase_shifts), np.cos(phase_shifts))

    # 构建复数传递函数（幅值比 * e^(j*相位差)）
    tf_complex = amp_ratios * np.exp(1j * phase_shifts)

    # 获取通道名称
    channel_names = comp_df.index.tolist()

    # 判断CompData是行矩阵还是列矩阵，并构建相同形状的TFData
    n_channels = len(channel_names)

    # CompData的DataFrame有N行2列（index为通道名，columns为amp_multiplier和time_increment）
    # 我们需要根据通道数量判断原始TFData是行矩阵还是列矩阵
    # 如果只有1个通道，无法判断，默认为列矩阵
    # 通常：CaliberSardine -> 列矩阵（多个AI通道），CaliberOctopus -> 行矩阵（多个AO通道）

    # 根据通道名判断是AI通道还是AO通道，决定构建行矩阵还是列矩阵
    # CaliberSardine: AI通道 -> 列矩阵（N行1列）
    # CaliberOctopus: AO通道 -> 行矩阵（1行N列）
    first_channel = channel_names[0].lower()

    if "ai" in first_channel and "ao" not in first_channel:
        # AI通道 -> CaliberSardine -> 列矩阵（N行1列）
        tf_df = pd.DataFrame(
            tf_complex.reshape(-1, 1),
            index=channel_names,
            columns=["TF"],  # 列名任意
        )
        logger.debug(f"检测到AI通道，构建列矩阵TFData: {n_channels}行1列")
    elif "ao" in first_channel and "ai" not in first_channel:
        # AO通道 -> CaliberOctopus -> 行矩阵（1行N列）
        tf_df = pd.DataFrame(
            [tf_complex],  # 行矩阵：1行N列
            index=["AI"],  # 行名任意（因为AI通道在CaliberOctopus中是固定的）
            columns=channel_names,
        )
        logger.debug(f"检测到AO通道，构建行矩阵TFData: 1行{n_channels}列")
    else:
        # 无法判断，默认为列矩阵
        logger.warning(
            f"无法从通道名判断矩阵类型（第一个通道: {channel_names[0]}），默认构建列矩阵"
        )
        tf_df = pd.DataFrame(
            tf_complex.reshape(-1, 1),
            index=channel_names,
            columns=["TF"],
        )

    # 创建TFData
    tf_data: TFData = {
        "tf_dataframe": tf_df,
        "sampling_info": comp_data["sampling_info"],
        "sine_args": sine_args,
        "mean_amp_ratio": mean_amp_ratio,
        "mean_phase_shift": mean_phase_shift,
    }

    logger.debug(
        f"CompData转换为TFData完成，通道数: {len(channel_names)}, "
        f"平均幅值比: {mean_amp_ratio:.6f}, 平均相位差: {mean_phase_shift:.6f}rad"
    )

    return tf_data


def average_comp_data_list(comp_data_list: list[CompData]) -> CompData:
    """
    对多个CompData进行按位平均,返回平均后的CompData。

    该函数接收一个CompData列表,对每个通道位置的补偿参数(amp_multiplier和time_increment)
    分别进行算术平均,同时对元数据字段(mean_amp_ratio和mean_phase_shift)也进行平均。

    Args:
        comp_data_list: CompData列表,要求:
            - 列表非空
            - 所有CompData的comp_dataframe形状必须一致
            - 所有CompData的sine_args应该相同(函数会使用第一个的sine_args)

    Returns:
        CompData: 平均后的补偿数据,包含:
            - comp_dataframe: 每个通道位置的平均补偿参数
            - sine_args: 使用第一个CompData的正弦波参数
            - mean_amp_ratio: 所有CompData的mean_amp_ratio的平均值
            - mean_phase_shift: 所有CompData的mean_phase_shift的平均值

    Raises:
        ValueError: 如果输入列表为空
        ValueError: 如果各CompData的comp_dataframe形状不一致

    Examples:
        >>> # 假设有3个CompData,每个包含8个通道的补偿数据
        >>> comp_data_1 = {...}  # 第1次测量
        >>> comp_data_2 = {...}  # 第2次测量
        >>> comp_data_3 = {...}  # 第3次测量
        >>> averaged = average_comp_data_list([comp_data_1, comp_data_2, comp_data_3])
        >>> # averaged包含8个通道的平均补偿参数
    """
    logger.info(f"开始平均 {len(comp_data_list)} 个CompData")

    # 验证输入列表非空
    if not comp_data_list:
        error_msg = "输入的CompData列表为空,无法进行平均"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # 验证所有CompData的comp_dataframe形状一致
    first_shape = comp_data_list[0]["comp_dataframe"].shape
    first_index = comp_data_list[0]["comp_dataframe"].index.tolist()

    for idx, comp_data in enumerate(comp_data_list):
        current_shape = comp_data["comp_dataframe"].shape
        current_index = comp_data["comp_dataframe"].index.tolist()

        if current_shape != first_shape:
            error_msg = (
                f"CompData列表中第 {idx} 个元素的comp_dataframe形状({current_shape}) "
                f"与第0个元素的形状({first_shape})不一致"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if current_index != first_index:
            error_msg = (
                f"CompData列表中第 {idx} 个元素的comp_dataframe索引与第0个元素不一致"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    # 将所有CompData的DataFrame按位平均
    # 收集所有DataFrame到列表中
    all_dfs = [comp_data["comp_dataframe"] for comp_data in comp_data_list]

    # 使用pandas的concat和mean进行平均
    # 注意：所有DataFrame的index和columns必须一致
    averaged_df = pd.concat(all_dfs).groupby(level=0).mean()

    logger.debug(
        f"DataFrame平均完成，通道数: {len(averaged_df)}, "
        f"平均amp_multiplier范围: "
        f"[{averaged_df['amp_multiplier'].min():.6f}, "
        f"{averaged_df['amp_multiplier'].max():.6f}], "
        f"平均time_increment范围: "
        f"[{averaged_df['time_increment'].min() * 1e6:.3f}μs, "
        f"{averaged_df['time_increment'].max() * 1e6:.3f}μs]"
    )

    # 计算平均的元数据
    avg_mean_amp_ratio = float(
        np.mean([comp_data["mean_amp_ratio"] for comp_data in comp_data_list])
    )
    avg_mean_phase_shift = float(
        np.mean([comp_data["mean_phase_shift"] for comp_data in comp_data_list])
    )

    # 采样信息和正弦波参数: 假定所有 CompData 一致, 使用第一个
    if "sampling_info" not in comp_data_list[0]:
        error_msg = "平均 CompData 失败: 缺少 sampling_info 字段"
        logger.error(error_msg)
        raise ValueError(error_msg)

    avg_sampling_info = comp_data_list[0]["sampling_info"]
    avg_sine_args = comp_data_list[0]["sine_args"]

    logger.info(
        f"平均完成: 频率={avg_sine_args['frequency']:.2f}Hz, "
        f"平均mean_amp_ratio={avg_mean_amp_ratio:.6f}, "
        f"平均mean_phase_shift={avg_mean_phase_shift:.6f}rad"
    )

    # 创建平均后的CompData
    averaged_comp_data: CompData = {
        "comp_dataframe": averaged_df,
        "sampling_info": avg_sampling_info,
        "sine_args": avg_sine_args,
        "mean_amp_ratio": avg_mean_amp_ratio,
        "mean_phase_shift": avg_mean_phase_shift,
    }

    return averaged_comp_data


def average_tf_data_list(tf_data_list: list[TFData]) -> TFData:
    """
    对多个TFData进行按位平均,返回平均后的TFData。

    该函数接收一个TFData列表,对每个通道位置的传递函数(复数形式)
    分别进行算术平均,同时对元数据字段(mean_amp_ratio和mean_phase_shift)也进行平均。

    Args:
        tf_data_list: TFData列表,要求:
            - 列表非空
            - 所有TFData的tf_dataframe形状必须一致
            - 所有TFData的sine_args应该相同(函数会使用第一个的sine_args)

    Returns:
        TFData: 平均后的传递函数数据,包含:
            - tf_dataframe: 每个通道位置的平均传递函数(复数形式)
            - sine_args: 使用第一个TFData的正弦波参数
            - mean_amp_ratio: 所有TFData的mean_amp_ratio的平均值
            - mean_phase_shift: 所有TFData的mean_phase_shift的平均值

    Raises:
        ValueError: 如果输入列表为空
        ValueError: 如果各TFData的tf_dataframe形状不一致

    Examples:
        >>> # 假设有3个TFData,每个包含8个通道的传递函数数据
        >>> tf_data_1 = {...}  # 第1次测量
        >>> tf_data_2 = {...}  # 第2次测量
        >>> tf_data_3 = {...}  # 第3次测量
        >>> averaged = average_tf_data_list([tf_data_1, tf_data_2, tf_data_3])
        >>> # averaged包含8个通道的平均传递函数参数
    """
    logger.info(f"开始平均 {len(tf_data_list)} 个TFData")

    # 验证输入列表非空
    if not tf_data_list:
        error_msg = "输入的TFData列表为空,无法进行平均"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # 验证所有TFData的tf_dataframe形状一致
    first_shape = tf_data_list[0]["tf_dataframe"].shape
    first_index = tf_data_list[0]["tf_dataframe"].index.tolist()
    first_columns = tf_data_list[0]["tf_dataframe"].columns.tolist()

    for idx, tf_data in enumerate(tf_data_list):
        current_shape = tf_data["tf_dataframe"].shape
        current_index = tf_data["tf_dataframe"].index.tolist()
        current_columns = tf_data["tf_dataframe"].columns.tolist()

        if current_shape != first_shape:
            error_msg = (
                f"TFData列表中第 {idx} 个元素的tf_dataframe形状({current_shape}) "
                f"与第0个元素的形状({first_shape})不一致"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if current_index != first_index or current_columns != first_columns:
            error_msg = (
                f"TFData列表中第 {idx} 个元素的tf_dataframe索引或列名与第0个元素不一致"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    # 将所有TFData的DataFrame按位平均
    # 收集所有DataFrame到列表中
    all_dfs = [tf_data["tf_dataframe"] for tf_data in tf_data_list]

    # 对复数DataFrame进行平均：分别平均实部和虚部，然后重新组合
    # 这样可以保证相位信息正确平均
    # 提取实部和虚部（直接使用numpy的real和imag属性）
    real_parts = [df.values.real for df in all_dfs]
    imag_parts = [df.values.imag for df in all_dfs]

    # 对所有数组求平均
    avg_real_array = np.mean(real_parts, axis=0)
    avg_imag_array = np.mean(imag_parts, axis=0)

    # 重新组合为复数并创建DataFrame（保持原有的索引和列名）
    avg_complex_array = avg_real_array + 1j * avg_imag_array
    averaged_df = pd.DataFrame(
        avg_complex_array,
        index=all_dfs[0].index,
        columns=all_dfs[0].columns,
    )

    # 从平均后的复数中提取幅值比和相位差用于日志
    avg_tf_complex = averaged_df.values.flatten()
    avg_amp_ratios = np.abs(avg_tf_complex)
    avg_phase_shifts = np.angle(avg_tf_complex)

    logger.debug(
        f"DataFrame平均完成，通道对数: {averaged_df.size}, "
        f"平均幅值比范围: [{avg_amp_ratios.min():.6f}, {avg_amp_ratios.max():.6f}], "
        f"平均相位差范围: [{avg_phase_shifts.min():.6f}rad, {avg_phase_shifts.max():.6f}rad]"
    )

    # 计算平均的元数据
    avg_mean_amp_ratio = float(
        np.mean([tf_data["mean_amp_ratio"] for tf_data in tf_data_list])
    )
    avg_mean_phase_shift = float(
        np.mean([tf_data["mean_phase_shift"] for tf_data in tf_data_list])
    )

    # 采样信息和正弦波参数: 假定所有 TFData 一致, 使用第一个
    if "sampling_info" not in tf_data_list[0]:
        error_msg = "平均 TFData 失败: 缺少 sampling_info 字段"
        logger.error(error_msg)
        raise ValueError(error_msg)

    avg_sampling_info = tf_data_list[0]["sampling_info"]
    avg_sine_args = tf_data_list[0]["sine_args"]

    logger.info(
        f"平均完成: 频率={avg_sine_args['frequency']:.2f}Hz, "
        f"平均mean_amp_ratio={avg_mean_amp_ratio:.6f}, "
        f"平均mean_phase_shift={avg_mean_phase_shift:.6f}rad"
    )

    # 创建平均后的TFData
    averaged_tf_data: TFData = {
        "tf_dataframe": averaged_df,
        "sampling_info": avg_sampling_info,
        "sine_args": avg_sine_args,
        "mean_amp_ratio": avg_mean_amp_ratio,
        "mean_phase_shift": avg_mean_phase_shift,
    }

    return averaged_tf_data
