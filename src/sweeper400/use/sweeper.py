"""
# 声学扫场测量模块

模块路径：`sweeper400.use.sweeper`

该模块主要包括SweeperCore类，提供声学扫场测量的核心功能，协同控制步进电机和数据采集系统，
实现空间中多点位的自动化声场信号采集。
"""

import copy
import threading
import time
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import TypedDict

import numpy as np
from matplotlib import pyplot as plt

from ..analyze import (
    Point2D,
    PositiveFloat,
    PositiveInt,
    SineArgs,
    SweepData,
    Waveform,
    TFData,
    get_sine_cycles,
    init_sampling_info,
    init_sine_args,
    load_sweep_data,
    save_sweep_data,
    plot_transfer_function_discrete_distribution,
    plot_transfer_function_instantaneous_field,
    plot_transfer_function_interpolated_distribution,
    sweep_data_to_point_tf_data,
)
from ..measure import SingleChasCSIO
from ..move import MotorController
from ..logger import get_logger

# 获取模块日志器
logger = get_logger(__name__)


# 辅助函数：以常见模式获取点阵（grid/point_list）
# 1. 矩形网格
def get_square_grid(
    x_start: float,
    x_end: float,
    y_start: float,
    y_end: float,
    step_size: PositiveFloat = 10.0,
) -> list[Point2D]:
    """
    生成矩形网格点阵

    在指定的矩形区域内生成均匀分布的网格点阵。
    根据给定的点间距自动计算点数，支持任意方向的扫场。
    点的顺序为：先沿点数较少的轴扫描（相等则X轴优先），蛇形进行。

    Args:
        x_start: X轴起始坐标（mm）
        x_end: X轴结束坐标（mm）
        y_start: Y轴起始坐标（mm）
        y_end: Y轴结束坐标（mm）
        step_size: 点间距（mm），默认10.0mm

    Returns:
        List[Point2D]: 生成的点阵列表

    Raises:
        ValueError: 当参数无效时

    Examples:
        >>> # 生成从(0,0)到(20,20)的网格，点间距10mm
        >>> test_grid = get_square_grid(0, 20, 0, 20, 10.0)
        >>> len(test_grid)
        9

        >>> # 支持反向扫场：从(20,20)到(0,0)
        >>> test_grid = get_square_grid(20, 0, 20, 0, 10.0)
        >>> len(test_grid)
        9
    """
    # 计算X和Y方向的距离和点数
    x_distance = abs(x_end - x_start)
    y_distance = abs(y_end - y_start)

    # 根据点间距计算点数（至少包含起点，如果距离足够则包含更多点）
    x_points = int(x_distance / step_size) + 1
    y_points = int(y_distance / step_size) + 1

    # 生成X和Y坐标数组
    # 使用linspace确保起点和终点都被包含（如果点数>1）
    if x_points == 1:
        x_coords = np.array([x_start])
    else:
        # 计算实际能放置的点数，确保不超过终点
        actual_x_range = (x_points - 1) * step_size

        if x_start <= x_end:
            x_coords = np.linspace(
                x_start, min(x_start + actual_x_range, x_end), x_points
            )
        else:
            x_coords = np.linspace(
                x_start, max(x_start - actual_x_range, x_end), x_points
            )

    if y_points == 1:
        y_coords = np.array([y_start])
    else:
        # 计算实际能放置的点数，确保不超过终点
        actual_y_range = (y_points - 1) * step_size

        if y_start <= y_end:
            y_coords = np.linspace(
                y_start, min(y_start + actual_y_range, y_end), y_points
            )
        else:
            y_coords = np.linspace(
                y_start, max(y_start - actual_y_range, y_end), y_points
            )

    # 生成网格点阵（点数较少的轴优先）
    grid: list[Point2D] = []
    if x_points <= y_points:
        for index, y in enumerate(y_coords):
            if index % 2 == 0:
                for x in x_coords:
                    grid.append(Point2D(float(x), float(y)))
            else:
                for x in reversed(x_coords):
                    grid.append(Point2D(float(x), float(y)))
    else:
        for index, x in enumerate(x_coords):
            if index % 2 == 0:
                for y in y_coords:
                    grid.append(Point2D(float(x), float(y)))
            else:
                for y in reversed(y_coords):
                    grid.append(Point2D(float(x), float(y)))

    logger.info(f"网格点阵生成完成，共 {len(grid)} 个点")
    logger.debug(f"X方向: {x_points} 个点，范围 {x_start:.1f} 到 {x_coords[-1]:.1f} mm")
    logger.debug(f"Y方向: {y_points} 个点，范围 {y_start:.1f} 到 {y_coords[-1]:.1f} mm")
    logger.debug(f"点间距: {step_size:.1f} mm")

    return grid


# 2. 直线
def get_line_grid(  # 暂停维护
    x_start: float,
    y_start: float,
    x_end: float,
    y_end: float,
    num_points: PositiveInt,
) -> list[Point2D]:
    """
    生成直线点阵

    在两点之间生成均匀分布的直线点阵。

    Args:
        x_start: 起始点X坐标（mm）
        y_start: 起始点Y坐标（mm）
        x_end: 结束点X坐标（mm）
        y_end: 结束点Y坐标（mm）
        num_points: 点数

    Returns:
        List[Point2D]: 生成的点阵列表

    Raises:
        ValueError: 当参数无效时

    Examples:
        >>> # 生成从(0,0)到(100,0)的5个点
        >>> test_line_grid = get_line_grid(0, 0, 100, 0, 5)
        >>> len(test_line_grid)
        5
    """

    logger.info(
        f"生成直线点阵: 从({x_start}, {y_start})到({x_end}, {y_end}), {num_points}个点"
    )

    # 生成坐标数组
    x_coords = np.linspace(x_start, x_end, num_points)
    y_coords = np.linspace(y_start, y_end, num_points)

    # 生成点阵
    line: list[Point2D] = []
    for x, y in zip(x_coords, y_coords, strict=False):
        line.append(Point2D(float(x), float(y)))

    logger.info(f"直线点阵生成完成，共 {len(line)} 个点")

    return line


# 定义扫场状态枚举
class SweepState(Enum):
    """
    扫场测量状态枚举

    IDLE: 空闲状态，未开始扫场或已完成/中止
    RUNNING: 正在执行扫场测量
    STOPPING: 正在停止扫场测量
    COMPLETED: 扫场测量成功完成
    ERROR: 扫场测量出现错误
    """

    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    ERROR = "error"


# 定义进度信息类型
class SweepProgress(TypedDict):
    """
    扫场进度信息

    state: 当前扫场状态
    completed_points: 已完成的点数
    remaining_points: 剩余点数
    elapsed_time: 已用时间（秒）
    remaining_time: 预估剩余时间（秒），如果无法预估则为None
    average_time_per_point: 平均每点耗时（秒），如果无法计算则为None
    """

    state: SweepState
    completed_points: int
    remaining_points: int
    elapsed_time: float
    remaining_time: float | None
    average_time_per_point: float | None


class SweeperCore:
    """
    # 声学扫场测量核心控制器

    该类协同控制步进电机和数据采集系统，实现自动化的声学扫场测量。
    在预定义的点阵中，依次移动到每个点位，采集指定数量的声场信号chunk。

    作为"核心"类，SweeperCore提供复杂、通用、严谨的功能实现，支持多通道AI/AO配置。

    ## 线程化设计：
        - 扫场操作在后台线程中执行，主线程不被阻塞
        - 支持实时状态查询和进度监控
        - 可以随时中止正在进行的扫场操作

    ## 主要功能：
        1. 点阵管理：存储和验证测量点阵
        2. 运动控制：自动控制机械臂移动到各个点位
        3. 数据采集：在每个点位采集指定数量的chunk（支持多通道）
        4. 数据存储：规范化存储所有点位的测量数据
        5. 状态监控：实时监控测量进度和状态
        6. 线程管理：后台执行扫场，支持中断和异常处理

    ## 使用方式：
    ```python
    # 创建输出波形
    from sweeper400.analyze import init_sampling_info, init_sine_args, get_sine_cycles
    sampling_info = init_sampling_info(48000, 4800)
    sine_args = init_sine_args(1000.0, 1.0, 0.0)
    output_waveform = get_sine_cycles(sampling_info, sine_args, cycles=100)

    # 创建扫场测量器（自动管理硬件控制器）
    sweeper = SweeperCore(
        ai_channels=("PXI1Slot2/ai0", "PXI1Slot3/ai0"),
        ao_channels_static=("PXI1Slot2/ao0",),
        static_output_waveform=output_waveform,
        point_list=grid
    )

    # 非阻塞扫场
    sweeper.sweep()  # 立即返回
    while sweeper.is_running():
        progress = sweeper.get_progress()
        print(f"进度: {progress['completed_points']}/{sweeper.total_points_num}")
        time.sleep(1)

    # 中止扫场
    sweeper.stop()

    # 清理资源（自动清理内部控制器）
    sweeper.cleanup()
    ```
    """

    # 获取类日志器
    logger = get_logger(f"{__name__}.SweeperCore")

    def __init__(
        self,
        ai_channels: tuple[str, ...],
        sweep_ai_channel: str | None = None,
        ao_channels_static: tuple[str, ...] = (),
        ao_channels_feedback: tuple[str, ...] = (),
        static_output_waveform: Waveform | None = None,
        feedback_function: Callable[[Waveform, Waveform, Waveform, TFData], Waveform] | None = None,
        buffer_size_multiplier: PositiveInt = 5,
        point_list: list[Point2D] | None = None,
        chunks_per_point: PositiveInt = 3,
        settle_time: PositiveFloat = 0.5,
    ) -> None:
        """
        初始化声学扫场测量核心控制器

        Args:
            ai_channels: AI通道名称元组，例如 ("PXI1Slot2/ai0", "PXI1Slot3/ai0")
            sweep_ai_channel: 用于扫场的传声器AI通道名称，如果为None则默认使用ai_channels的第一个元素
            ao_channels_static: 静态AO通道名称元组，例如 ("PXI1Slot2/ao0",)，可选，默认为空
            ao_channels_feedback: 反馈AO通道名称元组，可选，默认为空
            static_output_waveform: 静态输出波形对象，如果未提供则创建默认波形
            feedback_function: 反馈函数，接收AI波形，返回反馈AO波形，可选
            buffer_size_multiplier: AI和Feedback AO缓冲区大小倍数，默认5
            point_list: 测量点阵，Point2D对象的列表，如果未提供则为空列表
            chunks_per_point: 每个点采集的chunk数量，默认3
            settle_time: 电机停止后的稳定等待时间（秒），默认0.5秒

        Raises:
            ValueError: 当参数无效时
            RuntimeError: 当硬件初始化失败时
        """
        # 验证并设置sweep_ai_channel
        if sweep_ai_channel is None:
            if not ai_channels:
                raise ValueError(
                    "ai_channels不能为空，必须至少包含一个元素"
                )
            sweep_ai_channel = ai_channels[0]
            logger.warning("sweep_ai_channel未提供，将使用ai_channels的第一个元素")
        elif sweep_ai_channel not in ai_channels:
            raise ValueError(f"sweep_ai_channel '{sweep_ai_channel}' 必须在ai_channels中")
        # 存储sweep_ai_channel参数
        assert sweep_ai_channel is not None
        self._sweep_ai_channel = sweep_ai_channel

        # 处理输出波形为None的情况
        if static_output_waveform is None:
            # 创建默认波形
            sampling_info = init_sampling_info(10000.0, 5000)
            sine_args = init_sine_args(frequency=1000.0, amplitude=0.0, phase=0.0)
            static_output_waveform = get_sine_cycles(sampling_info, sine_args)
            logger.debug(
                "未提供输出波形，创建默认波形：1000Hz正弦波，幅值0.0，采样率10kHz"
            )
        # 类型断言，静态输出波形不可能为None
        assert static_output_waveform is not None

        # 存储配置参数
        self._ai_channels = ai_channels
        self._ao_channels_static = ao_channels_static
        self._ao_channels_feedback = ao_channels_feedback
        self._feedback_function = feedback_function
        self._buffer_size_multiplier = buffer_size_multiplier

        # 创建核心组件
        try:
            logger.debug("正在初始化数据采集控制器...")
            self._measure_controller = SingleChasCSIO(
                ai_channels=ai_channels,
                ao_channels_static=ao_channels_static,
                ao_channels_feedback=ao_channels_feedback,
                static_output_waveform=static_output_waveform,
                export_function=self._data_export_callback,
                feedback_function=feedback_function,  # 如果为None，SingleChasCSIO使用默认函数
                buffer_size_multiplier=buffer_size_multiplier,
            )
            logger.debug("数据采集控制器初始化成功")
        except Exception as e:
            error_msg = f"数据采集控制器初始化失败: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

        try:
            logger.debug("正在初始化步进电机控制器...")
            self._move_controller = MotorController()
            logger.debug("步进电机控制器初始化成功")
        except Exception as e:
            error_msg = f"步进电机控制器初始化失败: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

        # 存储测量参数
        self._point_list = point_list if point_list is not None else []
        self._TOTAL_POINTS_NUM = len(self._point_list)
        self._CHUNKS_PER_POINT = chunks_per_point
        self._SETTLE_TIME = settle_time

        # 数据存储
        self._sweep_data: SweepData = {
            "ai_data_list": [],
            "ao_data": static_output_waveform,
        }
        # 数据结构:
        # {
        #     "ai_data_list": [（列表索引即为点序号）
        #         {
        #             "position": Point2D,
        #             "ai_data": List[Waveform],
        #         },
        #         ...
        #     ],
        #     "ao_data": Waveform,
        # }

        # 状态标志和线程管理
        self._state = SweepState.IDLE  # 扫场状态
        self._state_lock = threading.Lock()  # 保护状态的线程锁
        self._current_point_index = 0
        self._point_index_lock = threading.Lock()  # 保护点索引的线程锁
        self._enough_chunks_event = threading.Event()

        # 扫场工作线程
        self._sweep_thread: threading.Thread | None = None

        # 时间跟踪
        self._sweep_start_time: float | None = None  # 扫场开始时间
        self._first_point_start_time: float | None = None  # 第一个点开始测量时间
        self._time_lock = threading.Lock()  # 保护时间变量的线程锁

        logger.info(
            f"SweeperCore 初始化完成 - "
            f"AI通道: {ai_channels}, "
            f"扫场AI通道: {self._sweep_ai_channel}, "
            f"Static AO通道: {ao_channels_static if ao_channels_static else '无'}, "
            f"Feedback AO通道: {ao_channels_feedback if ao_channels_feedback else '无'}, "
            f"测量点数: {len(self._point_list)}, "
            f"每点chunk数: {self._CHUNKS_PER_POINT}, "
            f"稳定时间: {self._SETTLE_TIME}s"
        )
        logger.debug(f"测量点阵: {self._point_list}")

    def _data_export_callback(
        self,
        ai_waveform: Waveform,
        ao_static_waveform: Waveform,
        ao_feedback_waveform: Waveform | None,
        chunks_num: int,
    ) -> None:
        """
        数据导出回调函数

        该函数会被SingleChasCSIO在（后台工作线程中）每次数据导出时调用，用于收集测量数据。
        使用线程锁保证线程安全。
        只保存sweep_ai_channel指定的单通道数据。

        Args:
            ai_waveform: 采集到的AI波形（多通道）
            ao_static_waveform: 当前的静态AO输出波形
            ao_feedback_waveform: 当前的反馈AO输出波形（如果没有反馈通道则为None）
            chunks_num: 当前chunk编号（从1开始）
        """
        # 检查是否提前中止
        current_state = self._get_state()
        if current_state not in (SweepState.RUNNING,):
            return

        # 使用线程锁安全地获取当前点的索引
        with self._point_index_lock:
            point_idx = self._current_point_index

        # 初始化当前点的数据存储（如果尚未初始化）
        # 由于点是按顺序处理的，point_idx 应该等于当前列表长度
        if point_idx >= len(self._sweep_data["ai_data_list"]):
            current_position = self._point_list[point_idx]
            self._sweep_data["ai_data_list"].append(
                {
                    "position": current_position,
                    "ai_data": [],
                }
            )
            logger.debug(f"初始化点 {point_idx} 的数据存储，位置: {current_position}")

        # 获取sweep_ai_channel在ai_channels中的索引
        try:
            channel_idx = self._ai_channels.index(self._sweep_ai_channel)
        except ValueError:
            logger.error(f"sweep_ai_channel '{self._sweep_ai_channel}' 不在ai_channels中")
            return

        # 从多通道波形中提取单通道数据
        # Waveform统一使用2D格式 (n_channels, n_samples)
        channel_data = ai_waveform[channel_idx, :]
        single_channel_waveform = Waveform(
            input_array=channel_data,
            sampling_rate=ai_waveform.sampling_rate,
            channel_names=(self._sweep_ai_channel,),
            timestamp=ai_waveform.timestamp,
            waveform_id=ai_waveform.waveform_id,
            sine_args=ai_waveform.sine_args,
        )

        # 存储单通道AI数据
        self._sweep_data["ai_data_list"][point_idx]["ai_data"].append(single_channel_waveform)

        # 检查是否已采集足够chunk
        if chunks_num >= self._CHUNKS_PER_POINT:
            self._enough_chunks_event.set()

        logger.debug(
            f"点 {point_idx} 采集第 {chunks_num}/{self._CHUNKS_PER_POINT} 个chunk, "
            f"扫场AI通道: {self._sweep_ai_channel} (索引: {channel_idx})"
        )

    def where(self) -> tuple[float, float]:
        """
        获取当前步进电机的绝对位置

        这是MotorController.get_current_position_2D方法的封装，方便用户查询
        当前的电机位置。

        Returns:
            tuple[float, float]: (X轴位置, Y轴位置)，单位为毫米

        Raises:
            RuntimeError: 当步进电机控制器未初始化时

        Examples:
            >>> sweeper = SweeperCore(...)
            >>> x_pos, y_pos = sweeper.where()
            >>> print(f"当前位置: ({x_pos:.3f}, {y_pos:.3f}) mm")
        """
        if not hasattr(self, "_move_controller"):
            raise RuntimeError("步进电机控制器未初始化")

        position = self._move_controller.get_current_position_2D()

        logger.info(f"当前位置: ({position[0]:.3f}, {position[1]:.3f}) mm")
        return position

    def move_to(self, x_mm: float, y_mm: float) -> bool:
        """
        移动步进电机到指定的绝对坐标位置

        这是MotorController.move_absolute_2D方法的封装，方便用户在创建sweeper后
        移动步进电机来确定位置。

        Args:
            x_mm: X轴目标绝对位置（毫米），0mm代表X轴负限位（零位）
            y_mm: Y轴目标绝对位置（毫米），0mm代表Y轴负限位（零位）

        Returns:
            bool: 移动是否成功完成

        Raises:
            RuntimeError: 当步进电机控制器未初始化时

        Examples:
            >>> sweeper = SweeperCore(...)
            >>> success = sweeper.move_to(100.0, 50.0)
            >>> if success:
            ...     print("移动成功")
        """
        if not hasattr(self, "_move_controller"):
            raise RuntimeError("步进电机控制器未初始化")

        logger.info(f"移动到位置: ({x_mm:.3f}, {y_mm:.3f}) mm")
        return self._move_controller.move_absolute_2D(x_mm=x_mm, y_mm=y_mm)

    def calib(self) -> bool:
        """
        执行步进电机的自动零位校准

        这是MotorController.calibrate_all_axis方法的封装，方便用户在创建sweeper后
        执行零位校准。

        Returns:
            bool: 校准是否成功完成

        Raises:
            RuntimeError: 当步进电机控制器未初始化时

        Examples:
            >>> sweeper = SweeperCore(...)
            >>> success = sweeper.calib()
            >>> if success:
            ...     print("校准成功")
        """
        if not hasattr(self, "_move_controller"):
            raise RuntimeError("步进电机控制器未初始化")

        logger.info("执行零位校准...")
        return self._move_controller.calibrate_all_axis()

    def new_point_list(self, point_list: list[Point2D]) -> None:
        """
        更新测量点阵

        该方法允许用户在创建sweeper后修改测量点阵。

        Args:
            point_list: 新的测量点阵，Point2D对象的列表

        Examples:
            >>> sweeper = SweeperCore(...)
            >>> new_point_list = [Point2D(100.0, 50.0), Point2D(200.0, 100.0)]
            >>> sweeper.new_point_list(new_point_list)
        """
        self._point_list = point_list
        self._TOTAL_POINTS_NUM = len(self._point_list)

    def new_csio(
        self,
        ai_channels: tuple[str, ...],
        ao_channels_static: tuple[str, ...] = (),
        ao_channels_feedback: tuple[str, ...] = (),
        static_output_waveform: Waveform | None = None,
        feedback_function: Callable[[Waveform, Waveform, Waveform, TFData], Waveform] | None = None,
        buffer_size_multiplier: PositiveInt = 5,
    ) -> None:
        """
        创建新的SingleChasCSIO实例，替换当前的数据采集控制器

        该方法允许用户在创建sweeper后修改AI/AO通道配置和输出波形。
        如果当前有正在运行的扫场任务，会先停止该任务。

        Args:
            ai_channels: 新的AI通道名称元组，例如 ("PXI1Slot2/ai0", "PXI1Slot3/ai0")
            ao_channels_static: 新的静态AO通道名称元组，例如 ("PXI1Slot2/ao0",)
            static_output_waveform: 新的静态输出波形对象
            ao_channels_feedback: 反馈AO通道名称元组，可选，默认为空
            feedback_function: 反馈函数，接收AI波形，返回反馈AO波形，可选
            buffer_size_multiplier: AI和Feedback AO缓冲区大小倍数，默认5

        Raises:
            RuntimeError: 当创建新的SingleChasCSIO失败时

        Examples:
            >>> sweeper = SweeperCore(...)
            >>> # 创建新的输出波形
            >>> sampling_info = init_sampling_info(48000.0, 4800)
            >>> sine_args = init_sine_args(2000.0, 0.02, 0.0)
            >>> new_waveform = get_sine_cycles(sampling_info, sine_args)
            >>> # 更新配置
            >>> sweeper.new_csio(
            ...     ("PXI1Slot2/ai1", "PXI1Slot3/ai1"),
            ...     ("PXI1Slot2/ao1",),
            ...     ("PXI1Slot3/ao1",),
            ...     new_waveform
            ... )
        """
        # 如果正在运行扫场，先停止
        if self._is_running:
            logger.warning("检测到扫场正在运行，先停止扫场...")
            self.stop(timeout=15.0)

        # 清理旧的数据采集控制器
        if hasattr(self, "_measure_controller"):
            try:
                logger.info("正在清理旧的数据采集控制器...")
                self._measure_controller.stop()
            except Exception as e:
                logger.warning(f"清理旧的数据采集控制器时出错: {e}")

        # 处理输出波形为None的情况
        if static_output_waveform is None:
            # 创建默认波形
            sampling_info = init_sampling_info(10000.0, 5000)
            sine_args = init_sine_args(frequency=1000.0, amplitude=0.01, phase=0.0)
            static_output_waveform = get_sine_cycles(sampling_info, sine_args)
            logger.debug(
                "未提供输出波形，创建默认波形：1000Hz正弦波，幅值0.01，采样率10kHz"
            )
        # 类型断言，静态输出波形不可能为None
        assert static_output_waveform is not None

        # 更新配置参数
        self._ai_channels = ai_channels
        self._ao_channels_static = ao_channels_static
        self._ao_channels_feedback = ao_channels_feedback
        self._feedback_function = feedback_function
        self._buffer_size_multiplier = buffer_size_multiplier

        # 创建新的数据采集控制器
        try:
            logger.info("正在创建新的数据采集控制器...")
            self._measure_controller = SingleChasCSIO(
                ai_channels=ai_channels,
                ao_channels_static=ao_channels_static,
                ao_channels_feedback=ao_channels_feedback,
                static_output_waveform=static_output_waveform,
                export_function=self._data_export_callback,
                feedback_function=feedback_function,  # 如果为None，SingleChasCSIO使用默认函数
                buffer_size_multiplier=buffer_size_multiplier,
            )
            # 存储新的输出波形
            self._sweep_data["ao_data"] = static_output_waveform

            logger.info(
                f"新的数据采集控制器创建成功 - "
                f"AI: {ai_channels}, "
                f"Static AO: {ao_channels_static}, "
                f"Feedback AO: {ao_channels_feedback if ao_channels_feedback else '无'}"
            )
        except Exception as e:
            error_msg = f"创建新的数据采集控制器失败: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def reset(self) -> None:
        """
        重置测量器状态

        **清除所有已采集的数据**，重置状态标志，准备进行新的测量。
        如果扫场正在运行，会先停止扫场。
        """
        # 如果正在运行，先停止
        if self._is_running:
            logger.warning("检测到扫场正在运行，先停止扫场...")
            self.stop(timeout=15.0)

        # 初始化数据存储
        self._sweep_data["ai_data_list"].clear()

        # 重置状态标志（使用线程锁保护）
        with self._point_index_lock:
            self._current_point_index = 0

        with self._time_lock:
            self._sweep_start_time = None
            self._first_point_start_time = None

        self._enough_chunks_event.clear()
        self._set_state(SweepState.IDLE)

        logger.info("SweeperCore 状态已重置")

    def sweep(
        self,
        result_folder: str | Path | None = None,
        lowcut: float = 100.0,
        highcut: float = 20000.0,
    ) -> bool:
        """
        启动扫场测量（非阻塞）

        在后台线程中执行扫场测量，立即返回。使用_get_state()和get_progress()监控进度。
        如果提供了result_folder，扫场完成后会自动保存数据和绘图。

        Args:
            result_folder: 结果保存文件夹路径，如果提供则自动保存数据和绘图。
                数据将保存为"sweep_data.pkl"，三种模式的图像分别保存为：
                - "discrete_distribution.png"
                - "interpolated_distribution.png"
                - "instantaneous_field.png"
            lowcut: 带通滤波器低截止频率（Hz），用于最终绘图，默认100.0
            highcut: 带通滤波器高截止频率（Hz），用于最终绘图，默认20000.0

        Returns:
            bool: 是否成功启动扫场测量（不代表测量完成）

        Raises:
            RuntimeError: 当扫场已在运行或系统状态异常时
        """
        # 检查当前状态
        if self._is_running:
            logger.warning("扫场测量已在运行中，无法重复启动")
            return False

        # 检查是否有之前的线程需要清理
        if self._sweep_thread is not None and self._sweep_thread.is_alive():
            logger.warning("检测到之前的扫场线程仍在运行，等待其结束...")
            self._sweep_thread.join(timeout=5.0)
            if self._sweep_thread.is_alive():
                logger.error("无法停止之前的扫场线程")
                return False

        try:
            # 重置状态
            self.reset()

            # 存储结果文件夹路径
            # 这些属性仅用于传递给wtf，因此暂不在init中定义
            if result_folder is not None:
                self._result_folder = Path(result_folder)
            else:
                self._result_folder = None

            # 存储绘图参数
            self._plot_lowcut = lowcut
            self._plot_highcut = highcut

            # 创建并启动扫场工作线程
            self._sweep_thread = threading.Thread(
                target=self._worker_thread_function,
                name="SweeperWorker",
                daemon=False,  # 不设为守护线程，确保能正常完成
            )
            assert self._sweep_thread is not None  # 消除类型警告
            self._sweep_thread.start()

            logger.info("后台线程已启动")
            return True

        except Exception as e:
            logger.error(f"启动扫场测量失败: {e}", exc_info=True)
            self._set_state(SweepState.ERROR)
            return False

    def _worker_thread_function(self) -> None:
        """
        扫场工作线程的主函数

        在后台线程中执行扫场逻辑，支持中断和异常处理
        """
        try:
            logger.info("=" * 50)
            logger.info("开始扫场测量...")
            logger.info(f"总测量点数: {self._TOTAL_POINTS_NUM}")
            logger.info(f"每点chunk数: {self._CHUNKS_PER_POINT}")
            logger.info("=" * 50)

            # 设置运行状态
            self._set_state(SweepState.RUNNING)

            # 记录开始时间
            sweep_start_time = time.time()
            with self._time_lock:
                self._sweep_start_time = sweep_start_time

            # 启动数据采集任务
            self._measure_controller.start()

            # 遍历所有测量点
            for point_index, point in enumerate(self._point_list):
                # 检查是否请求停止
                current_state = self._get_state()
                if current_state not in (SweepState.RUNNING,):
                    logger.info("后台线程收到停止请求，退出...")
                    self._set_state(SweepState.IDLE)
                    return

                logger.info(
                    f"\n--- 处理点 {point_index + 1}/{len(self._point_list)} ---"
                )

                # 使用线程锁安全地更新当前点索引
                with self._point_index_lock:
                    self._current_point_index = point_index

                # 移动到目标点位
                if not self._wtf_move_to_point(point):
                    logger.error(f"移动到点 {point_index} 失败", exc_info=True)
                    self._set_state(SweepState.ERROR)
                    return

                # 若为第一个点，记录第一个点开始测量的时间（排除准备时间）
                if point_index == 0:
                    with self._time_lock:
                        self._first_point_start_time = time.time()
                # 否则，每隔5个点输出预估剩余时间（最初五个点除外）
                elif point_index <= 3 or (point_index + 1) % 5 == 0:
                    progress = self.get_progress()
                    if progress["remaining_time"] is not None:
                        logger.info(
                            f"（预估剩余时间: {progress['remaining_time']:.2f}s）"
                        )

                # 在当前点位采集数据
                if not self._wtf_collect_data_at_point(point_index):
                    logger.error(f"点 {point_index} 数据采集失败", exc_info=True)
                    self._set_state(SweepState.ERROR)
                    return

            # 计算总耗时
            total_time = time.time() - sweep_start_time

            # 停止数据采集任务
            self._measure_controller.stop()

            logger.info("=" * 50)
            logger.info("扫场测量完成！")
            logger.info(f"总测量点数: {self._TOTAL_POINTS_NUM}")
            logger.info(f"总耗时: {total_time:.2f}s ({total_time / 60:.2f}min)")
            logger.info(f"平均每点耗时: {total_time / len(self._point_list):.2f}s")
            logger.info("=" * 50)

            # 如果指定了结果文件夹，自动保存数据和绘图
            if hasattr(self, "_result_folder") and self._result_folder is not None:
                try:
                    logger.info(f"开始自动保存结果到: {self._result_folder}")

                    # 确保目录存在
                    self._result_folder.mkdir(parents=True, exist_ok=True)

                    # 保存数据
                    data_save_path = self._result_folder / "sweep_data.pkl"
                    self.save_data(data_save_path)

                    # 绘制三种模式的图像
                    plot_base_path = self._result_folder / "plot"
                    self.plot_data(
                        plot_base_path,
                        lowcut=self._plot_lowcut,
                        highcut=self._plot_highcut,
                    )

                    logger.info(f"结果自动保存完成: {self._result_folder}")

                except Exception as save_e:
                    logger.error(f"自动保存结果失败: {save_e}", exc_info=True)

            # 设置完成状态
            self._set_state(SweepState.COMPLETED)

        except Exception as e:
            logger.error(f"扫场测量过程中发生异常: {e}", exc_info=True)
            self._set_state(SweepState.ERROR)

        finally:
            # 只有在非正常完成时才执行清理
            current_state = self._get_state()
            if current_state not in (SweepState.COMPLETED,):
                try:
                    self._measure_controller.stop()
                except Exception as e:
                    logger.error(f"停止数据采集任务时出错: {e}", exc_info=True)

    def _wtf_move_to_point(self, point: Point2D) -> bool:
        """
        移动到指定点位（阻塞操作）

        Args:
            point: 目标点位坐标

        Returns:
            bool: 移动是否成功
        """
        # 检查是否请求停止
        current_state = self._get_state()
        if current_state not in (SweepState.RUNNING,):
            logger.debug("移动过程中检测到停止请求")
            return False

        logger.info(f"移动到点位: ({point.x:.3f}, {point.y:.3f}) mm")

        # 执行2D绝对运动
        success = self._move_controller.move_absolute_2D(x_mm=point.x, y_mm=point.y)

        if not success:
            logger.error(f"移动到点位 ({point.x:.3f}, {point.y:.3f}) 失败")
            return False

        # 等待电机稳定
        logger.debug(f"等待电机稳定 {self._SETTLE_TIME}s")
        time.sleep(self._SETTLE_TIME)

        return True

    def _wtf_collect_data_at_point(self, point_index: int) -> bool:
        """
        在当前点位采集数据。
        阻塞操作，会等待_enough_chunks_event事件到达。

        Args:
            point_index: 点位索引

        Returns:
            bool: 采集是否成功
        """
        logger.debug(f"开始在点 {point_index} 采集数据...")

        # 预估持续时间并设置超时
        chunk_duration = self._measure_controller.static_output_waveform.duration
        expected_duration = chunk_duration * self._CHUNKS_PER_POINT
        timeout = expected_duration + 5.0  # 设置超时为预期时间+5秒

        # 启用数据导出
        self._measure_controller.enable_export = True
        logger.debug("已启用数据导出")

        try:
            # 等待数据就绪事件或超时
            # 该事件在CSIO的_export_function中被设置
            chunks_ready = self._enough_chunks_event.wait(timeout=timeout)

            # 事件成功到达
            if chunks_ready:
                # 停止数据导出
                self._measure_controller.enable_export = False
                # 清除事件标志
                self._enough_chunks_event.clear()
                logger.info(f"点 {point_index + 1} 数据采集完成")
                return True

            # 事件超时
            else:
                self._measure_controller.enable_export = False
                logger.error(
                    f"点 {point_index} 数据采集超时（{timeout:.1f}s），强制停止"
                )
                return False

        except Exception as e:
            self._measure_controller.enable_export = False
            logger.error(f"点 {point_index} 数据采集过程中发生异常: {e}")
            return False

    def get_progress(self) -> SweepProgress:
        """
        获取扫场进度信息

        Returns:
            SweepProgress: 进度信息字典
        """
        with self._state_lock:
            state = self._state

        with self._point_index_lock:
            current_point = self._current_point_index

        # 计算剩余点数
        remaining_points = self._TOTAL_POINTS_NUM - current_point

        with self._time_lock:
            sweep_start_time = self._sweep_start_time
            first_point_start_time = self._first_point_start_time

        # 计算时间相关信息
        current_time = time.time()

        # 计算已用时间
        if sweep_start_time is not None:
            elapsed_time = current_time - sweep_start_time
        else:
            elapsed_time = 0.0

        # 计算平均每点耗时和剩余时间预估
        estimated_remaining_time: float | None = None
        average_time_per_point: float | None = None

        if (
            first_point_start_time is not None
            and current_point > 0  # 若第一个点未完成，无法估计剩余时间
            and state == SweepState.RUNNING
        ):
            # 计算实际测量时间（排除准备时间）
            measurement_elapsed_time = current_time - first_point_start_time
            average_time_per_point = measurement_elapsed_time / current_point

            # 计算剩余点数和预估时间
            if remaining_points > 0:
                estimated_remaining_time = average_time_per_point * remaining_points

        return SweepProgress(
            state=state,
            completed_points=current_point,  # 已完成的点数等于当前点索引
            remaining_points=remaining_points,
            elapsed_time=elapsed_time,
            remaining_time=estimated_remaining_time,
            average_time_per_point=average_time_per_point,
        )

    def _set_state(self, new_state: SweepState) -> None:
        """
        线程安全地设置扫场状态

        Args:
            new_state: 新的状态
        """
        with self._state_lock:
            old_state = self._state
            self._state = new_state

        logger.debug(f"状态变更: {old_state.value} -> {new_state.value}")

    def _get_state(self) -> SweepState:
        """
        获取当前扫场状态

        Returns:
            SweepState: 当前状态
        """
        with self._state_lock:
            return self._state

    @property
    def _is_running(self) -> bool:
        """
        检查扫场是否正在运行（内部方法）

        Returns:
            bool: 是否正在运行
        """
        state = self._get_state()
        return state in (SweepState.RUNNING, SweepState.STOPPING)

    def stop(self, timeout: float = 10.0) -> bool:
        """
        中止扫场测量

        设置停止标志并等待扫场线程结束。

        Args:
            timeout: 等待超时时间（秒），默认10秒

        Returns:
            bool: 是否成功停止
        """
        if not self._is_running:
            logger.warning("扫场测量未在运行，无需中止")
            return True

        logger.info("正在中止扫场测量...")

        # 设置停止状态
        self._set_state(SweepState.STOPPING)

        # 中断数据采集等待
        self._enough_chunks_event.set()

        # 等待扫场线程结束
        if self._sweep_thread is not None and self._sweep_thread.is_alive():
            self._sweep_thread.join(timeout=timeout)

            if self._sweep_thread.is_alive():
                logger.warning(f"扫场线程未能在{timeout}秒内结束")
                return False

        logger.info("扫场测量已成功中止")
        self._set_state(SweepState.IDLE)
        return True

    def import_data(self, file_path: str | Path) -> None:
        """
        从文件导入已有的SweepData

        从磁盘加载之前保存的测量数据，方便进行二次绘图或分析。
        导入的数据将替换当前实例中的_sweep_data。

        Args:
            file_path: 数据文件的路径（.pkl文件）

        Raises:
            FileNotFoundError: 当文件不存在时
            IOError: 当文件读取失败时
            ValueError: 当数据格式不正确时

        Examples:
            >>> sweeper = SweeperCore(...)
            >>> sweeper.import_data("path/to/sweep_data.pkl")
            >>> # 现在可以使用plot_data方法进行绘图
            >>> sweeper.plot_data("output.png")
        """
        file_path: Path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        logger.info(f"开始导入测量数据: {file_path}")

        try:
            # 使用load_sweep_data函数加载数据
            loaded_data = load_sweep_data(file_path)

            # 替换当前数据
            self._sweep_data = loaded_data

            # 更新点列表（从导入的数据中提取）
            ai_data_list = loaded_data["ai_data_list"]
            if ai_data_list:
                self._point_list = [point_data["position"] for point_data in ai_data_list]
                self._TOTAL_POINTS_NUM = len(self._point_list)

            logger.info(f"数据导入成功，共 {len(ai_data_list)} 个点")

        except Exception as e:
            logger.error(f"数据导入失败: {e}", exc_info=True)
            raise

    def export_data(self) -> SweepData:
        """
        导出测量数据（深拷贝）

        Returns:
            SweepData: 测量数据副本
        """
        data_copy: SweepData = copy.deepcopy(self._sweep_data)
        return data_copy

    def save_data(
        self,
        save_path: str | Path,
        compress_level: int = 9,
    ) -> None:
        """
        保存测量数据到文件（使用gzip压缩）

        将测量数据和输出波形一起打包保存为字典格式，使用gzip压缩以减小文件体积。

        数据结构：
        {
            "ai_data_list": List[PointSweepData]，每个PointRawData包含：
                - "position": Point2D对象，表示该点的坐标
                - "ai_data": List[Waveform]，该点采集的所有AI波形
            "ao_data": Waveform，扫场过程中使用的输出波形
        }

        Args:
            save_path: 保存文件的路径（建议使用.pkl或.pkl.gz扩展名）
            compress_level: gzip压缩级别（0-9），默认9。
                          0表示不压缩，9表示最大压缩。
                          级别越高压缩率越高但速度越慢。

        Raises:
            ValueError: 当没有数据可保存时
            IOError: 当文件保存失败时
            RuntimeError: 当数据采集控制器未初始化时
        """
        # python中，空容器视为False
        if not self._sweep_data:
            raise ValueError("没有可保存的数据，请先执行扫场测量")

        # 使用统一的save_sweep_data函数保存数据
        save_sweep_data(self._sweep_data, save_path, compresslevel=compress_level)

    def plot_data(
        self,
        save_path: str | Path,
        ref_sine_args: SineArgs | None = None,
        lowcut: float = 100.0,
        highcut: float = 20000.0,
        filter_order: int = 4,
        trim_samples: int = 0,
    ) -> None:
        """
        绘制扫场数据的空间分布图（三种模式）

        将self._sweep_data转换为PointTFData列表，然后一次性绘制三种模式的图像：
        - 离散分布图（方形色块）
        - 插值分布图
        - 瞬时声压场图

        Args:
            save_path: 保存图片的基础路径（不含扩展名），三张图将分别保存为：
                - {save_path}_discrete.png
                - {save_path}_interpolated.png
                - {save_path}_instantaneous.png
            ref_sine_args: 参考正弦波参数，如果为None则尝试从ao_data获取
            lowcut: 带通滤波器低截止频率（Hz），默认100.0
            highcut: 带通滤波器高截止频率（Hz），默认20000.0
            filter_order: 滤波器阶数，默认4
            trim_samples: 滤波后切除波形开头的采样点数量，默认0

        Raises:
            ValueError: 当没有数据可绘制时
            OSError: 当图片保存失败时

        Examples:
            ```python
            >>> # 绘制三种模式的图像
            >>> sweeper.plot_data("output/plot")  # noqa
            >>> # 生成：output/plot_discrete.png, output/plot_interpolated.png, output/plot_instantaneous.png
            ```
        """
        # 验证数据存在
        if not self._sweep_data or not self._sweep_data["ai_data_list"]:
            raise ValueError("没有可绘制的数据，请先执行扫场测量")

        # 转换为Path对象
        save_path_base = Path(save_path)

        # 确保目录存在
        save_path_base.parent.mkdir(parents=True, exist_ok=True)

        logger.info("开始绘制扫场数据分布图（三种模式）")

        # 将SweepData转换为PointTFData列表
        try:
            plot_tf_results = sweep_data_to_point_tf_data(
                self._sweep_data,
                ref_sine_args=ref_sine_args,
                lowcut=lowcut,
                highcut=highcut,
                filter_order=filter_order,
                trim_samples=trim_samples,
            )
            logger.debug("plot_tf_results计算完成")
        except Exception as e:
            logger.error(f"转换SweepData到PointTFData失败: {e}", exc_info=True)
            raise ValueError(f"数据转换失败: {e}") from e

        # 定义三种模式的绘图配置
        plot_configs = [
            ("discrete", plot_transfer_function_discrete_distribution),
            ("interpolated", plot_transfer_function_interpolated_distribution),
            ("instantaneous", plot_transfer_function_instantaneous_field),
        ]

        # 依次绘制三种模式的图像
        for mode_name, plot_func in plot_configs:
            try:
                save_path_with_mode = save_path_base.parent / f"{save_path_base.name}_{mode_name}.png"
                fig, _ = plot_func(
                    plot_tf_results,
                    save_path=str(save_path_with_mode),
                )
                plt.close(fig)
                logger.debug(f"{mode_name} 模式图像绘制完成")
            except Exception as e:
                logger.error(f"绘制 {mode_name} 模式图像失败: {e}", exc_info=True)
                raise OSError(f"无法保存 {mode_name} 模式图片: {e}") from e

        logger.info("三种模式的扫场数据分布图绘制完成")

    def cleanup(self) -> None:
        """
        清理资源

        清理状态，停止所有线程，销毁内部创建的控制器对象。
        """
        # 如果正在运行，先停止
        if self._is_running:
            logger.warning("检测到扫场正在运行，强制停止...")
            self.stop(timeout=15.0)

        # 清理数据采集控制器
        if hasattr(self, "_measure_controller"):
            try:
                logger.debug("正在清理数据采集控制器...")
                self._measure_controller.stop()  # 确保停止任务
                # SingleChasCSIO没有专门的cleanup方法，stop()已经处理了资源清理
                logger.debug("数据采集控制器清理完成")
            except Exception as e:
                logger.error(f"清理数据采集控制器时出错: {e}")

        # 清理步进电机控制器
        if hasattr(self, "_move_controller"):
            try:
                logger.debug("正在清理步进电机控制器...")
                self._move_controller.cleanup()
                logger.debug("步进电机控制器清理完成")
            except Exception as e:
                logger.error(f"清理步进电机控制器时出错: {e}")

        # 重置状态
        try:
            self.reset()
        except Exception as e:
            logger.error(f"重置状态时出错: {e}")

        logger.info("SweeperCore 资源清理完成")

    def __del__(self):
        """析构函数，确保资源被正确释放"""
        self.cleanup()
