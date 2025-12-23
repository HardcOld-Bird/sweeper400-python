"""
# 自定义数据类型模块

模块路径：`sweeper400.analyze.my_dtypes`

本模块定义了sweeper400项目中特有的自定义**数据类型和容器**。
主要包含用于管理时域波形数据的Waveform类。
"""

from typing import Annotated, Any, NamedTuple, TypedDict, TypeGuard

import numpy as np
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
        >>> sampling_info = init_sampling_info(PositiveInt(1000), PositiveInt(2048))
        >>> print(sampling_info)
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
        >>> sine_args = init_sine_args(1000, 1.0, 0.0)
        >>> print(sine_args)
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
    timestamp: np.datetime64,
    id: int | None,
    sine_args: SineArgs | None,
) -> "Waveform":
    """
    从pickle数据重建Waveform对象

    Args:
        array_data: 数组数据
        sampling_rate: 采样率
        timestamp: 时间戳
        id: ID
        sine_args: 正弦波参数

    Returns:
        重建的Waveform对象
    """
    # 转换为Waveform类型
    waveform_obj = array_data.view(Waveform)

    # 设置所有自定义属性（包括None值，确保属性存在）
    waveform_obj._sampling_rate = sampling_rate  # type: ignore
    waveform_obj.timestamp = timestamp
    waveform_obj.id = id
    waveform_obj.sine_args = sine_args

    return waveform_obj


# 定义 "Waveform" 类型，其为ndarray的子类
class Waveform(np.ndarray):
    """
    # 时域波形数据容器类

    继承自`numpy.ndarray`，用于管理时域波形数据及其元数据。
    支持单通道数据（一维数组）和多通道数据（二维数组）。

    ## 新增属性：
        - **sampling_rate**: 波形数据的**采样率**（Hz），只读属性
        - **timestamp**: 波形采样开始**时间戳**，可修改属性
        - **id**: 波形的**唯一标识符**，可修改属性，可选属性
        - **sine_args**: 波形的**正弦波参数**，可修改属性，可选属性

    ## 使用示例：
        创建单通道波形：
        ```python
        data = np.array([1.0, 2.0, 3.0, 4.0])
        waveform = Waveform(data, sampling_rate=1000)
        ```

        创建多通道波形：
        ```python
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        waveform = Waveform(data, sampling_rate=1000)
        ```

        指定时间戳：
        ```python
        waveform = Waveform(data, sampling_rate=1000,
                           timestamp=np.datetime64("2024-01-01T12:00:00", "ns"))
        ```
    """

    # "__new__"方法中不支持属性的类型注解（而"__init__"中可以），
    # 因此需要首先显式声明类型
    _sampling_rate: PositiveFloat
    timestamp: np.datetime64
    id: int | None
    sine_args: SineArgs | None

    # 获取类日志器（类属性，所有实例共享）
    logger = get_logger(f"{__name__}.Waveform")

    def __new__(
        cls,
        input_array: np.ndarray | list[float | int],  # python原生float也即numpy.float64
        sampling_rate: PositiveFloat,
        timestamp: np.datetime64 | None = None,
        id: int | None = None,
        sine_args: SineArgs | None = None,
        **kwargs: Any,
    ) -> "Waveform":
        """
        创建Waveform对象

        Args:
            input_array: 输入的波形数据数组
            sampling_rate: 采样率（Hz），必须为正实数
            timestamp: 采样开始时间戳，可选，默认为当前时间
            id: 波形的唯一标识符，可选，默认为None
            **kwargs: 传递给numpy.ndarray的其他参数

        Returns:
            Waveform对象实例
        """
        # 转换输入数组为numpy数组
        try:
            obj = np.asarray(input_array, dtype=np.float64).view(cls)
        except Exception as e:
            logger.error(f"输入数据无法转换为numpy数组: {e}", exc_info=True)
            raise TypeError(f"输入数据无法转换为numpy数组: {e}") from e

        # 验证数组维度（只支持1D和2D）
        if obj.ndim not in [1, 2]:
            logger.error(f"只支持1维或2维数组，得到{obj.ndim}维数组", exc_info=True)
            raise ValueError(f"只支持1维或2维数组，得到{obj.ndim}维数组")

        # 设置只读属性
        obj._sampling_rate = sampling_rate

        # 设置时间戳
        if timestamp is None:
            obj.timestamp = np.datetime64("now", "ns")
            logger.debug(f"使用自动时间戳: {obj.timestamp}")
        else:
            obj.timestamp = timestamp
            logger.debug(f"使用指定时间戳: {obj.timestamp}")

        # 设置其他可选的元数据
        obj.id = id
        obj.sine_args = sine_args

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
            self._sampling_rate = obj._sampling_rate  # type: ignore
        else:  # 如果没有，说明是在__new__中，稍后会设置
            pass  # 保持现有值

        if hasattr(obj, "timestamp"):
            self.timestamp = obj.timestamp  # type: ignore
        else:
            pass  # 同理

        if hasattr(obj, "id"):
            self.id = obj.id  # type: ignore
        else:
            pass  # 同理

        if hasattr(obj, "sine_args"):
            self.sine_args = obj.sine_args  # type: ignore
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
                self.timestamp,  # 不应为None
                getattr(self, "id", None),
                getattr(self, "sine_args", None),
            ),
        )

    @property
    def channels_num(self) -> PositiveInt:
        """
        通道数

        （ndarray的shape为元组，不会被意外更改，直接传出是安全的）

        Returns:
            通道数，单通道返回1，多通道返回通道数
        """
        if self.ndim == 1:
            return 1
        else:
            return self.shape[0]

    @property
    def samples_num(self) -> PositiveInt:
        """
        采样点数

        （ndarray的shape为元组，不会被意外更改，直接传出是安全的）

        Returns:
            每个通道的采样点数
        """
        if self.ndim == 1:
            return self.shape[0]
        else:
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
            f"id={self.id})"
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
class PointRawData(TypedDict):
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

    ai_data_list: list[PointRawData]
    ao_data: Waveform


# 数据后处理相关数据类型
# 定义传递函数结果数据格式
class PointTFData(TypedDict):
    """
    传递函数计算结果格式

    ## 内部组成:
        **position**: 测量点的二维坐标
        **amp_ratio**: 幅值比（输入/输出）
        **phase_shift**: 相位差（输入-输出，弧度制）
    """

    position: "Point2D"
    amp_ratio: float
    phase_shift: float
