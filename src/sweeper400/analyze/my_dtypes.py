"""
# 自定义数据类型模块

模块路径：`sweeper400.analyze.my_dtypes`

本模块定义了sweeper400项目中特有的自定义**数据类型和容器**。
主要包含用于管理时域波形数据的Waveform类。
"""
# 允许类型前向引用（也即“Waveform”可以写为Waveform）
from __future__ import annotations

from typing import Annotated, Any, NamedTuple, TypedDict, TypeGuard

import numpy as np
import pandas as pd
from pydantic import AfterValidator

from ..logger import get_logger

# 获取模块日志器
logger = get_logger(__name__)


# 定义 "PositiveInt" 类型（同时支持类型检查和运行时验证）
# 1.定义验证函数，可在运行时抛出异常
def validate_positive_int(num: int) -> int:
    """
    PositiveInt类型的验证函数。

    Args:
        num: 待验证的整数

    Returns:
        如果验证通过，返回原整数

    Raises:
        ValueError: 如果不是正整数
    """
    if positive_int_type_guard(num):
        return num
    else:
        logger.error(f"PositiveInt验证失败: {num}", exc_info=True)
        raise ValueError("PositiveInt必须是正整数")


# 2.使用Annotated定义类型（会运行验证函数，并使用返回值替换原始值。
# 如果初始值异常则抛出异常。）
PositiveInt = Annotated[int, AfterValidator(validate_positive_int)]


# 3.定义类型守卫，静态检查可获知PositiveInt类型
def positive_int_type_guard(num: int) -> TypeGuard[PositiveInt]:
    """
    PositiveInt的类型守卫函数。

    Args:
        num: 待验证的整数

    Returns:
        如果验证通过，返回True
    """
    return num > 0


# 类似地，定义 "PositiveFloat" 类型
def validate_positive_float(num: float) -> float:
    """
    PositiveFloat类型的验证函数。

    Args:
        num: 待验证的浮点数

    Returns:
        如果验证通过，返回原浮点数

    Raises:
        ValueError: 如果不是正数
    """
    if positive_float_type_guard(num):
        return num
    else:
        logger.error(f"PositiveFloat验证失败: {num}", exc_info=True)
        raise ValueError("PositiveFloat必须是正数")


PositiveFloat = Annotated[float, AfterValidator(validate_positive_float)]


def positive_float_type_guard(num: float) -> TypeGuard[PositiveFloat]:
    """
    PositiveFloat的类型守卫函数。

    Args:
        num: 待验证的浮点数

    Returns:
        如果验证通过，返回True
    """
    return num > 0


# 定义 "SamplingInfo" 类型（及其初始化工具函数），其为特定格式的dict
class SamplingInfo(TypedDict):
    """
    具有标准化格式的**采样信息**`dict`。

    ## 内部组成:
        **sampling_rate**: 采样率，正实数（Hz）
        **samples_num**: 单次采样数，正整数
    """

    sampling_rate: PositiveFloat  # 正实数
    samples_num: PositiveInt  # 正整数


def init_sampling_info(
    sampling_rate: PositiveFloat, samples_num: PositiveInt
) -> SamplingInfo:
    """
    标准化地生成采样信息字典

    Args:
        sampling_rate: 采样率，必须为正实数（Hz），且不建议超过200kHz
        samples_num: 总采样数，必须为正整数

    Returns:
        sampling_info: 包含采样率和采样数信息的字典

    Examples:
        ```python
        >>> test_sampling_info = init_sampling_info(PositiveInt(1000), PositiveInt(2048))
        >>> print(test_sampling_info)
        {'sampling_rate': 1000, 'samples_num': 2048}
        ```
    """
    logger.debug(
        f"创建采样信息dict: sampling_rate={sampling_rate}Hz, samples_num={samples_num}"
    )

    sampling_info: SamplingInfo = {
        "sampling_rate": sampling_rate,
        "samples_num": samples_num,
    }

    return sampling_info


# 定义 "SineArgs" 类型（及其初始化工具函数），其为特定格式的dict
class SineArgs(TypedDict):
    """
    具有标准化格式的**正弦波参数**`dict`。

    ## 内部组成:
        **frequency**: 正弦波频率，正实数（Hz）
        **amplitude**: 正弦波幅值，正实数（无单位）
        **phase**: 正弦波弧度制初始相位，实数
    """

    frequency: PositiveFloat  # 正实数
    amplitude: PositiveFloat  # 正实数
    phase: float  # 实数


def init_sine_args(
    frequency: PositiveFloat,
    amplitude: PositiveFloat = 1.0,
    phase: float = 0.0,
) -> SineArgs:
    """
    标准化地生成正弦波参数字典

    Args:
        frequency: 正弦波频率，必须为正实数（Hz）
        amplitude: 正弦波幅值，必须为正实数（无单位）
        phase: 正弦波弧度制初始相位，实数

    Returns:
        sine_args: 包含正弦波参数信息的字典

    Examples:
        ```python
        >>> test_sine_args = init_sine_args(1000, 1.0, 0.0)
        >>> print(test_sine_args)
        {'frequency': 1000, 'amplitude': 1.0, 'phase': 0.0}
        ```
    """
    logger.debug(
        f"创建正弦波参数dict: frequency={frequency}Hz, "
        f"amplitude={amplitude}, phase={phase}rad"
    )

    sine_args: SineArgs = {
        "frequency": frequency,
        "amplitude": amplitude,
        "phase": phase,
    }

    return sine_args


# 先定义Waveform的pickle重建函数
def _rebuild_waveform_from_pickle(
    array_data: np.ndarray,
    sampling_rate: PositiveFloat,
    channel_names: tuple[str, ...],
    timestamp: np.datetime64,
    waveform_id: int | None,
    frequency: PositiveFloat | None,
    channel_complex_amplitudes: np.ndarray | None,
) -> Waveform:
    """
    从pickle数据重建Waveform对象

    Args:
        array_data: 数组数据
        sampling_rate: 采样率
        channel_names: 通道名称元组
        timestamp: 时间戳
        waveform_id: 波形ID（整数）
        frequency: 正弦波频率（Hz）
        channel_complex_amplitudes: 各通道复振幅数组

    Returns:
        重建的Waveform对象
    """
    # 转换为Waveform类型
    waveform_obj: Waveform = array_data.view(Waveform)  # noqafrom __future__ import annotations

    # 设置所有自定义属性（包括None值，确保属性存在）
    waveform_obj._sampling_rate = sampling_rate
    waveform_obj._channel_names = channel_names
    waveform_obj.timestamp = timestamp
    waveform_obj.waveform_id = waveform_id
    waveform_obj.frequency = frequency
    waveform_obj._channel_complex_amplitudes = channel_complex_amplitudes

    return waveform_obj


# 定义 "Waveform" 类型，其为ndarray的子类
class Waveform(np.ndarray):
    """
    # 时域波形数据容器类

    继承自`numpy.ndarray`，用于管理时域波形数据及其元数据。
    **统一使用二维数组存储**，形状始终为 `(n_channels, n_samples)`。

    ## 设计原则：
        - 内部数据**始终为 2D**，形状为 `(n_channels, n_samples)`
        - 单通道数据自动转换为 `(1, n_samples)` 形状
        - 这消除了处理 1D/2D 混合数据时的复杂性

    ## 新增属性：
        - **sampling_rate**: 波形数据的**采样率**（Hz），只读属性
        - **channel_names**: 波形的**通道名称元组**，可修改属性
        - **timestamp**: 波形采样开始**时间戳**，可修改属性
        - **waveform_id**: 波形的**唯一标识符**，可修改属性，可选属性
        - **frequency**: 波形的**正弦波频率**（Hz），可修改属性，可选属性
        - **channel_complex_amplitudes**: 各通道的**复振幅**（模长为幅值，相角为相位），
          可修改属性，可选属性，长度必须等于通道数

    ## 使用示例：
        创建单通道波形（自动转换为 2D）：
        ```python
        >>> data = np.array([1.0, 2.0, 3.0, 4.0])  # 1D 输入
        >>> waveform = Waveform(data, sampling_rate=1000, _channel_names=("ch1",))
        >>> # 内部存储为 shape=(1, 4) 的 2D 数组
        ```

        创建多通道波形：
        ```python
        >>> data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3) 形状
        >>> waveform = Waveform(data, sampling_rate=1000, _channel_names=("ch1", "ch2"))
        >>> # 内部存储为 shape=(2, 3) 的 2D 数组
        ```

        检查是否为单通道：
        ```python
        >>> if waveform.is_single_channel:
        >>>     print("这是单通道波形")
        ```
    """

    # "__new__"方法中不支持属性的类型注解（而"__init__"中可以），
    # 因此需要首先显式声明类型
    _sampling_rate: PositiveFloat
    _channel_names: tuple[str, ...]
    timestamp: np.datetime64
    waveform_id: int | None
    frequency: PositiveFloat | None
    _channel_complex_amplitudes: np.ndarray | None  # complex128, shape=(n_channels,)

    # 获取类日志器（类属性，所有实例共享）
    logger = get_logger(f"{__name__}.Waveform")

    def __new__(
        cls,
        input_array: np.ndarray | list[float | int],  # python原生float也即numpy.float64
        sampling_rate: PositiveFloat,
        channel_names: tuple[str, ...],
        timestamp: np.datetime64 | None = None,
        waveform_id: int | None = None,
        frequency: PositiveFloat | None = None,
        channel_complex_amplitudes: np.ndarray | None = None,
        **kwargs: Any,
    ) -> Waveform:
        """
        创建Waveform对象

        输入数组会被**统一转换为 2D 形状** `(n_channels, n_samples)`：
        - 1D 输入 `(n_samples,)` → 自动 reshape 为 `(1, n_samples)`
        - 2D 输入 `(n_channels, n_samples)` → 保持原状

        Args:
            input_array: 输入的波形数据数组（1D 或 2D）
            sampling_rate: 采样率（Hz），必须为正实数
            channel_names: 通道名称元组，长度必须等于通道数
            timestamp: 采样开始时间戳，可选，默认为当前时间
            waveform_id: 波形的唯一标识符，可选，默认为None
            frequency: 正弦波频率（Hz），可选，默认为None
            channel_complex_amplitudes: 各通道复振幅数组（complex128），
                长度必须等于通道数，模长为幅值，相角为相位，可选，默认为None
            **kwargs: 传递给numpy.ndarray的其他参数

        Returns:
            Waveform对象实例，内部数据始终为 2D

        Raises:
            TypeError: 当输入数据无法转换为numpy数组时
            ValueError: 当输入数组维度不是1D或2D时
            ValueError: 当channel_names长度与通道数不匹配时
            ValueError: 当channel_complex_amplitudes长度与通道数不匹配时
        """
        # 转换输入数组为numpy数组
        try:
            arr = np.asarray(input_array, dtype=np.float64)
        except Exception as e:
            logger.error(f"输入数据无法转换为numpy数组: {e}", exc_info=True)
            raise TypeError(f"输入数据无法转换为numpy数组: {e}") from e

        # 统一转换为 2D 形状 (n_channels, n_samples)
        if arr.ndim == 1:
            # 1D 输入: (n_samples,) -> (1, n_samples)
            arr = arr.reshape(1, -1)
            logger.debug(f"1D 输入自动转换为 2D: shape={arr.shape}")
        elif arr.ndim == 2:
            # 2D 输入: 保持原状 (n_channels, n_samples)
            pass
        else:
            logger.error(f"只支持1维或2维数组，得到{arr.ndim}维数组", exc_info=True)
            raise ValueError(f"只支持1维或2维数组，得到{arr.ndim}维数组")

        # 创建 Waveform 视图
        obj: Waveform = arr.view(cls)  # noqa

        # 验证channel_names长度（现在 obj 始终为 2D）
        n_channels = obj.shape[0]
        if len(channel_names) != n_channels:
            logger.error(
                f"channel_names长度({len(channel_names)})与通道数({n_channels})不匹配",
                exc_info=True,
            )
            raise ValueError(
                f"channel_names长度({len(channel_names)})与通道数({n_channels})不匹配"
            )

        # 验证channel_complex_amplitudes长度
        if channel_complex_amplitudes is not None:
            n_channels = obj.shape[0]
            cca = np.asarray(channel_complex_amplitudes, dtype=np.complex128)
            if cca.ndim != 1 or len(cca) != n_channels:
                logger.error(
                    f"channel_complex_amplitude长度({len(cca)})与通道数({n_channels})不匹配",
                    exc_info=True,
                )
                raise ValueError(
                    f"channel_complex_amplitude长度({len(cca)})与通道数({n_channels})不匹配"
                )
            channel_complex_amplitudes = cca

        # 设置必要属性
        obj._sampling_rate = sampling_rate
        obj._channel_names = channel_names  # 存储为只读属性，方便设置setter

        # 设置时间戳
        if timestamp is None:
            obj.timestamp = np.datetime64("now", "ns")
            logger.debug(f"使用自动时间戳: {obj.timestamp}")
        else:
            obj.timestamp = timestamp
            logger.debug(f"使用指定时间戳: {obj.timestamp}")

        # 设置其他可选的元数据
        obj.waveform_id = waveform_id
        obj.frequency = frequency
        obj._channel_complex_amplitudes = channel_complex_amplitudes  # 存储为只读属性，方便设置setter

        logger.debug(
            f"创建Waveform对象: shape={obj.shape}, sampling_rate={sampling_rate}Hz"
        )

        return obj

    def __array_finalize__(self, obj: np.ndarray | None) -> None:
        """
        数组完成时的回调函数

        在numpy数组操作后保持自定义属性
        """
        if obj is None:
            return

        # 安全地获取属性
        if hasattr(obj, "_sampling_rate"):
            self._sampling_rate = obj._sampling_rate  # noqa
        else:  # 如果没有，说明是在__new__中，稍后会设置
            pass  # 保持现有值

        if hasattr(obj, "_channel_names"):
            self._channel_names = obj._channel_names  # noqa
        else:
            pass  # 同理

        if hasattr(obj, "timestamp"):
            self.timestamp = obj.timestamp
        else:
            pass  # 同理

        if hasattr(obj, "waveform_id"):
            self.waveform_id = obj.waveform_id
        else:
            pass  # 同理

        if hasattr(obj, "frequency"):
            self.frequency = obj.frequency
        else:
            pass  # 同理

        if hasattr(obj, "_channel_complex_amplitudes"):
            # 直接赋值私有属性，跳过setter验证。
            # 因为NumPy切片（如nidaqmx内部的data[0][0]）会触发__array_finalize__，
            # 此时新数组shape已变化，走setter的channels_num校验会误报错误。
            self._channel_complex_amplitudes = obj._channel_complex_amplitudes  # noqa
        else:
            pass  # 同理

    def __reduce__(self) -> tuple[Any, ...]:
        """
        自定义pickle序列化过程

        Returns:
            包含重建对象所需信息的元组
        """
        # 调用前述重建函数，传递数组数据和自定义属性
        return (
            _rebuild_waveform_from_pickle,
            (
                np.asarray(self),  # 数组数据
                self._sampling_rate,  # 不应为None
                self._channel_names,  # 不应为None
                self.timestamp,  # 不应为None
                getattr(self, "waveform_id", None),
                getattr(self, "frequency", None),
                getattr(self, "channel_complex_amplitudes", None),
            ),
        )

    @property
    def channels_num(self) -> PositiveInt:
        """
        通道数

        由于内部数据始终为 2D，直接返回 shape[0]。
        （ndarray的shape为元组，不会被意外更改，直接传出是安全的）

        Returns:
            通道数（始终 >= 1）
        """
        return self.shape[0]

    @property
    def is_single_channel(self) -> bool:
        """
        检查是否为单通道波形

        Returns:
            如果是单通道（channels_num == 1）返回 True，否则返回 False
        """
        return self.shape[0] == 1

    @property
    def samples_num(self) -> PositiveInt:
        """
        采样点数

        由于内部数据始终为 2D，直接返回 shape[1]。
        （ndarray的shape为元组，不会被意外更改，直接传出是安全的）

        Returns:
            每个通道的采样点数
        """
        return self.shape[1]

    @property
    def sampling_rate(self) -> PositiveFloat:
        """
        采样率（Hz）

        只读属性，只能在对象创建时指定
        （float是不可变对象，直接传出是安全的）

        Returns:
            采样率值（Hz）
        """
        return self._sampling_rate

    @property
    def channel_names(self) -> tuple[str, ...]:
        """
        通道名称元组

        Returns:
            通道名称元组，长度等于通道数
        """
        return self._channel_names

    @channel_names.setter
    def channel_names(self, new_value: tuple[str, ...]) -> None:
        """
        设置通道名称元组

        Args:
            new_value: 新的通道名称元组，长度必须等于通道数
        """
        if len(new_value) != self.channels_num:
            raise ValueError(
                f"channel_names长度({len(new_value)})与通道数({self.channels_num})不匹配"
            )
        self._channel_names = new_value

    @property
    def sampling_info(self) -> SamplingInfo:
        """
        采样信息

        （临时变量，直接传出是安全的）

        Returns:
            采样信息字典
        """
        sampling_info = init_sampling_info(self.sampling_rate, self.samples_num)
        return sampling_info

    @property
    def channel_complex_amplitudes(self) -> np.ndarray | None:
        """
        各通道的复振幅数组

        Returns:
            复振幅数组（complex128），长度等于通道数，或None
        """
        return self._channel_complex_amplitudes

    @channel_complex_amplitudes.setter
    def channel_complex_amplitudes(self, new_value: np.ndarray | None) -> None:
        """
        设置各通道的复振幅数组

        Args:
            new_value: 新的复振幅数组（complex128），长度必须等于通道数，或None

        Raises:
            ValueError: 当数组长度与通道数不匹配时
        """
        if new_value is not None:
            cca = np.asarray(new_value, dtype=np.complex128)
            if cca.ndim != 1 or len(cca) != self.channels_num:
                raise ValueError(
                    f"channel_complex_amplitude长度({len(cca)})与通道数({self.channels_num})不匹配"
                )
            self._channel_complex_amplitudes = cca
        else:
            self._channel_complex_amplitudes = None

    @property
    def duration(self) -> PositiveFloat:
        """
        波形持续时间（秒）

        （临时变量，直接传出是安全的）

        Returns:
            波形持续时间，单位为秒
        """
        return self.samples_num / self.sampling_rate

    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        return (
            f"Waveform(shape={self.shape}, "
            f"sampling_rate={self.sampling_rate}Hz, "
            f"duration={self.duration:.6f}s, "
            f"timestamp={self.timestamp}, "
            f"waveform_id={self.waveform_id})"
        )

    def __str__(self) -> str:
        """返回对象的简洁字符串表示"""
        return self.__repr__()


# Sweeper相关数据类型
# 定义二维空间点坐标的命名元组
class Point2D(NamedTuple):
    """
    二维空间点坐标

    Attributes:
        x: X轴坐标（mm）
        y: Y轴坐标（mm）
    """

    x: float
    y: float


# 定义Sweeper采集的单点原始数据格式
class PointSweepData(TypedDict):
    """
    Sweeper采集的单点原始数据格式

    ## 内部组成:
        **position**: 测量点的二维坐标
        **ai_data**: 该点采集的所有AI波形
    """

    position: Point2D
    ai_data: list[Waveform]


# 定义Sweeper采集的完整测量数据格式
class SweepData(TypedDict):
    """
    Sweeper采集的完整测量数据格式

    ## 内部组成:
        **ai_data_list**: 测量数据列表
        **ao_data**: 输出波形
    """

    ai_data_list: list[PointSweepData]
    ao_data: Waveform


# 数据后处理相关数据类型
# 定义传递函数计算结果容器格式
class TFData(TypedDict):
    """
    传递函数计算结果容器格式。

    该类型包含传递函数计算的完整结果，使用DataFrame存储所有通道对的传递函数数据，
    以及计算过程中的关键参数（采样信息、正弦波参数、平均幅值比、平均相位差）。

    ## 内部组成:
        **tf_dataframe**: 传递函数数据DataFrame，
            - index: AO通道名称（行索引）
            - columns: AI通道名称（列索引）
            - 内容: 复数形式的传递函数值（幅值比 * e^(j*相位差)）
        **sampling_info**: 采样信息（必选）
        **frequency**: 正弦波频率（Hz）
        **mean_amp_ratio**: 所有通道对的平均幅值比
        **mean_phase_shift**: 所有通道对的平均相位差（弧度制）

    说明：
        - TFData专注于描述「通道对」的频率响应本身
        - TFData通常应为"方形"矩阵（行数、列数>1）
        - 使用复数形式同时存储幅值比和相位差信息
    """

    tf_dataframe: pd.DataFrame  # 复数类型DataFrame
    sampling_info: SamplingInfo
    frequency: float
    mean_amp_ratio: float
    mean_phase_shift: float


# 定义补偿参数计算结果容器格式
class CompData(TypedDict):
    """
    补偿参数计算结果容器格式（也用作校准数据格式）。

    该类型包含补偿参数计算的完整结果，使用DataFrame存储所有通道的补偿数据，
    以及计算过程中的关键参数（采样信息、正弦波参数、平均幅值比、平均相位差）。

    ## 内部组成:
        **comp_dataframe**: 补偿参数DataFrame，
            - index: 通道名称（行索引）
                - 对于CaliberOctopus（AO通道校准）: 记录AO通道名称
                - 对于CaliberSardine（AI通道校准）: 记录AI通道名称
            - columns: ['amp_multiplier', 'time_increment']（列索引）
                - amp_multiplier: 幅值补偿倍率（补偿到平均值需要乘以的倍率）
                - time_increment: 时间延迟补偿值（补偿到平均值需要加上的时间差，单位：秒）
            - 内容: 浮点数
        **sampling_info**: 采样信息（必选）
        **frequency**: 正弦波频率（Hz）
        **mean_amp_ratio**: 所有通道对的平均幅值比
        **mean_phase_shift**: 所有通道对的平均相位差（弧度制）

    说明：
        - CompData专注于「通道」本身的补偿信息
        - CompData的DataFrame总是"长/宽为1的方形"，即行矩阵（1行多列）或列矩阵（多行1列）
        - 要么ao_channel恒定（AI通道校准），要么ai_channel恒定（AO通道校准）
    """

    comp_dataframe: pd.DataFrame  # 浮点数类型DataFrame
    sampling_info: SamplingInfo
    frequency: float
    mean_amp_ratio: float
    mean_phase_shift: float


def init_comp_data(
    comp_dataframe: pd.DataFrame,
    sampling_info: SamplingInfo,
    frequency: float,
    mean_amp_ratio: float,
    mean_phase_shift: float,
) -> CompData:
    """
    手动获取CompData的工具函数

    Args:
        comp_dataframe: 补偿参数DataFrame
        sampling_info: 采样信息（必选）
        frequency: 正弦波频率（Hz）
        mean_amp_ratio: 所有通道对的平均幅值比
        mean_phase_shift: 所有通道对的平均相位差（弧度制）

    Returns:
        CompData: 补偿参数计算结果容器格式
    """
    logger.debug(
        f"手动创建CompData: comp_dataframe={comp_dataframe}, "
        f"sampling_info={sampling_info}, "
        f"frequency={frequency}, "
        f"mean_amp_ratio={mean_amp_ratio}, "
        f"mean_phase_shift={mean_phase_shift}"
    )

    comp_data = CompData(
        comp_dataframe=comp_dataframe,
        sampling_info=sampling_info,
        frequency=frequency,
        mean_amp_ratio=mean_amp_ratio,
        mean_phase_shift=mean_phase_shift,
    )

    return comp_data
