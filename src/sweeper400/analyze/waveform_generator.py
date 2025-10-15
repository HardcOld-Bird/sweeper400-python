"""
# 波形生成器模块

模块路径：`sweeper400.analyze.waveform_generator`

包含连续波形生成器的抽象基类和具体实现。
"""

from abc import ABC, abstractmethod

import numpy as np

from sweeper400.logger import get_logger

from .basic_sine import get_sine
from .my_dtypes import PositiveFloat, SamplingInfo, SineArgs, Waveform

# 获取模块日志器
logger = get_logger(__name__)


class WaveformGenerator(ABC):
    """
    # 连续波形生成器抽象基类

    定义了所有连续波形生成器的通用接口。
    子类必须实现generate方法来生成连续的波形数据。

    ## 核心方法：
        - **__init__**: 初始化生成器，接受采样信息和时间戳
        - **generate**: 生成下一个连续波形（抽象方法）
        - **_update_next_parameters**: 更新下一次生成的参数

    Examples:
        ```python
        # 子类实现示例
        class MyGenerator(WaveformGenerator):
            def generate(self) -> Waveform:
                # 实现具体的波形生成逻辑
                pass
        ```
    """

    # 获取类日志器
    logger = get_logger(f"{__name__}.WaveformGenerator")

    def __init__(
        self,
        sampling_info: SamplingInfo,
        next_timestamp: np.datetime64 | None = None,
        next_id: int = 1,
    ) -> None:
        """
        初始化波形生成器基类

        Args:
            sampling_info: 采样信息，包含采样率和采样点数
            next_timestamp: 下一次生成波形的时间戳，默认为None
            next_id: 下一次生成波形的唯一标识符，默认为None
        """
        self._sampling_info = sampling_info
        self._next_timestamp = next_timestamp
        self._next_id = next_id

        logger.debug(
            f"初始化WaveformGenerator基类: "
            f"sampling_info={self._sampling_info}, "
            f"next_timestamp={self._next_timestamp}, "
            f"next_id={self._next_id}"
        )

    @abstractmethod
    def generate(self) -> Waveform:
        """
        生成下一个连续波形

        子类必须实现此方法来生成具体的波形数据。

        Returns:
            生成的波形对象
        """
        pass

    def _update_next_parameters(self, current_waveform: Waveform) -> None:
        """
        更新下一次生成使用的参数

        默认实现只更新时间戳和id，子类可以重写此方法来更新其他参数。

        Args:
            current_waveform: 当前生成的波形
        """
        # 计算时长对应的纳秒数
        duration_ns = int(current_waveform.duration * 1e9)
        # 使用numpy的timedelta64进行时间戳计算，保持纳秒精度
        self._next_timestamp = current_waveform.timestamp + np.timedelta64(
            duration_ns, "ns"
        )

        # 更新id
        self._next_id += 1

        logger.debug(
            f"更新next_timestamp: {self._next_timestamp}, next_id: {self._next_id}"
        )

    @property
    def sampling_info(self) -> SamplingInfo:
        """
        获取当前的采样信息

        Returns:
            当前采样信息的副本
        """
        return self._sampling_info.copy()

    @property
    def chunk_duration(self) -> PositiveFloat:
        """
        单段波形持续时间（秒）

        Returns:
            单段波形持续时间，单位为秒
        """
        return self._sampling_info["samples_num"] / self._sampling_info["sampling_rate"]


class SineGenerator(WaveformGenerator):
    """
    # 连续正弦波形生成器类

    基于WaveformGenerator抽象基类的正弦波生成器实现。
    使用简单参数多次**连续**生成单频正弦`Waveform`对象。
    初相位会智能更新，因此多次生成的Waveform首尾相接可合成一个无缝的连续波形。

    Examples:
        ```python
        >>> sampling_info = init_sampling_info(1000, 1024)
        >>> sine_args = init_sine_args(50.0, 1.0, 0.0)
        >>> generator = SineGenerator(sampling_info, sine_args)
        >>> wave1 = generator.generate()
        >>> wave2 = generator.generate()
        # wave1和wave2可以无缝连接
        ```
    """

    # 获取类日志器
    logger = get_logger(f"{__name__}.SineGenerator")

    def __init__(
        self,
        sampling_info: SamplingInfo,
        next_sine_args: SineArgs,
        next_timestamp: np.datetime64 | None = None,
        next_id: int = 1,
    ) -> None:
        """
        初始化连续正弦波形生成器

        Args:
            sampling_info: 采样信息，包含采样率和采样点数
            sine_args: 正弦波参数，包含频率、幅值和初相位信息
            next_timestamp: 下一次合成Waveform的时间戳，默认为None
            next_id: 下一次生成Waveform的唯一标识符，默认为1
        """
        # 调用父类初始化
        super().__init__(sampling_info, next_timestamp, next_id)

        # 设置正弦波特有参数
        self._next_sine_args: SineArgs = next_sine_args

        logger.debug(
            f"初始化SineGenerator: "
            f"frequency={self._next_sine_args['frequency']}Hz, "
            f"amplitude={self._next_sine_args['amplitude']}, "
            f"phase={self._next_sine_args['phase']}rad"
        )

    def generate(self) -> Waveform:
        """
        生成连续的正弦波形

        Returns:
            包含单频正弦波的Waveform对象
        """
        logger.debug(
            f"生成连续正弦波: frequency={self._next_sine_args['frequency']}Hz, "
            f"amplitude={self._next_sine_args['amplitude']}, "
            f"phase={self._next_sine_args['phase']}rad, "
            f"timestamp={self._next_timestamp}, id={self._next_id}"
        )

        # 调用get_sine函数生成波形
        output_sine_wave = get_sine(
            sampling_info=self._sampling_info,
            sine_args=self._next_sine_args,
            timestamp=self._next_timestamp,
            id=self._next_id,
        )

        # 更新下一次生成的参数
        self._update_next_parameters(output_sine_wave)

        logger.debug(
            f"成功生成连续正弦波，更新参数: "
            f"next_phase={self._next_sine_args['phase']}rad, "
            f"next_timestamp={self._next_timestamp}, next_id={self._next_id}"
        )

        return output_sine_wave

    def _update_next_parameters(self, current_waveform: Waveform) -> None:
        """
        更新下一次生成使用的相位、时间戳和id

        Args:
            current_waveform: 当前生成的波形
        """
        # 计算下一次的相位，确保波形连续
        # 相位增量 = 2π * frequency * duration
        phase_increment = (
            2 * np.pi * self._next_sine_args["frequency"] * current_waveform.duration
        )
        self._next_sine_args["phase"] = (
            self._next_sine_args["phase"] + phase_increment
        ) % (2 * np.pi)

        # 调用父类方法更新时间戳和id
        super()._update_next_parameters(current_waveform)

        logger.debug(f"更新next_phase: {self._next_sine_args['phase']}rad")

    @property
    def next_sine_args(self) -> SineArgs:
        """
        获取当前的正弦波参数

        Returns:
            当前正弦波参数的副本
        """
        return self._next_sine_args.copy()
