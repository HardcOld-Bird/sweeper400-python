"""
# 步进电机控制模块

模块路径：`sweeper400.move.controller`

该模块主要包括`MotorController`类，提供MT-22E控制器的完整封装，支持X、Y双轴步进电机的精确控制。
通过调用MT_API.dll实现与硬件的通信，提供高级的运动控制接口。

- 出于兼容性考虑，本模块暂未使用PositiveInt和PositiveFloat等自定义类型，未来可考虑使用
- 出于兼容性考虑，本模块暂未使用threading进行异步处理，未来可考虑使用
"""

import os
import time
from ctypes import (
    POINTER,
    byref,
    c_char_p,
    c_double,
    c_int32,
    c_uint16,
    create_string_buffer,
    windll,
)

# 导入日志系统
from ..logger import get_logger

# 获取模块专用日志器
logger = get_logger(__name__)


class MotorController:
    """
    # 步进电机控制器类

    该类封装了MT-22E控制器的API调用，提供安全的电机控制功能。
    支持X、Y两轴步进电机的精确控制，包括零位校准、绝对/相对运动等。

    ## 主要功能：
        1. 硬件连接管理：自动初始化DLL、建立USB连接
        2. 电机参数管理：步进角度、细分数、螺距、传动比的设置和换算
        3. 零位校准：自动双轴零位校准，建立坐标系
        4. 运动控制：支持1D/2D绝对和相对运动，智能参数设置
        5. 状态监控：实时获取电机状态、位置信息
        6. 安全保护：超时保护、限位检测、异常处理

    Attributes:
        _step_angle: 步进角度（度）
        _subdivision: 驱动器细分数
        _pitch: 螺距（mm）
        _transmission_ratio: 传动比
        _is_initialized: API初始化状态
        _is_connected: 控制器连接状态
        _is_calibrated: 零位校准状态


    ## 使用示例：
    ```python
    controller = MotorController()
    controller.calibrate_all_axis()  # 零位校准
    controller.move_absolute_2D(x_mm=100.0, y_mm=50.0)  # 移动到指定位置
    controller.cleanup()  # 清理资源
    ```

    ## 注意事项：
        - 使用前建议执行零位校准
        - 坐标系以负限位为原点(0, 0)
        - 最大行程：MAX_POSITION mm（基于硬件限制）
        - 脱机模式下部分功能使用本地计算
    """

    # 获取类日志器
    logger = get_logger(f"{__name__}.MotorController")

    def __init__(
        self,
        step_angle: float = 1.8,
        subdivision: int = 20,
        pitch: float = 4.0,
        transmission_ratio: float = 1.0,
    ):
        """初始化电机控制器

        Args:
            step_angle: 步进角度（度），默认1.8°
            subdivision: 驱动器细分数，默认20
            pitch: 螺距（mm），默认4mm
            transmission_ratio: 传动比，默认1.0（无传动）
        """
        # 将电机参数存入私有属性
        self._step_angle: float = step_angle
        self._subdivision: int = subdivision
        self._pitch: float = pitch
        self._transmission_ratio: float = transmission_ratio

        logger.info(
            f"电机控制器实例已创建 - 参数: 步进角度={step_angle}°, "
            f"细分数={subdivision}, 螺距={pitch}mm, 传动比={transmission_ratio}"
        )
        # 定义类常量
        self.MAX_POSITION = 313.0  # 最大行程（毫米）

        # 初始化状态信息
        self._is_initialized = False
        self._is_connected = False
        self._is_calibrated = False
        self._current_position = [0.0, 0.0]

        # 初始化API和连接
        self._initialize_dll()
        self._setup_api_types()
        self._connect_usb()

        # 记录初始位置
        self.get_current_position_2D()

    def _initialize_dll(self) -> bool:
        """初始化DLL文件（私有方法）

        Returns:
            bool: 初始化是否成功
        """
        try:
            # 设置DLL路径并加载
            dll_path = os.path.join(os.path.dirname(__file__), "MT_API.dll")
            self._api = windll.LoadLibrary(dll_path)
            logger.debug(f"成功加载DLL: {dll_path}")

            # 初始化API
            result = self._api.MT_Init()
            if result == 0:
                self._is_initialized = True
                logger.debug("API初始化成功")
                return self._is_initialized
            else:
                self._is_initialized = False
                logger.error(f"API初始化失败，错误码: {result}", exc_info=True)
                return self._is_initialized

        except Exception as e:
            self._is_initialized = False
            logger.error(f"初始化过程中发生异常: {e}", exc_info=True)
            return self._is_initialized

    def _setup_api_types(self) -> bool:
        """设置API函数的参数类型

        Returns:
            bool: 设置是否成功
        """
        if not self._is_initialized:
            logger.error("API未初始化，无法设置函数参数类型", exc_info=True)
            return False

        # 硬件信息相关函数
        self._api.MT_Get_Product_Resource.argtypes = [POINTER(c_int32)]
        self._api.MT_Get_Product_ID.argtypes = [c_char_p]
        self._api.MT_Get_Product_SN.argtypes = [c_char_p]

        # 辅助计算函数
        self._api.MT_Help_Step_Line_Real_To_Steps.argtypes = [
            c_double,
            c_int32,
            c_double,
            c_double,
            c_double,
        ]
        self._api.MT_Help_Step_Line_Real_To_Steps.restype = c_int32

        self._api.MT_Help_Step_Line_Steps_To_Real.argtypes = [
            c_double,
            c_int32,
            c_double,
            c_double,
            c_int32,
        ]
        self._api.MT_Help_Step_Line_Steps_To_Real.restype = c_double

        # 通信相关函数
        self._api.MT_Open_UART.argtypes = [c_char_p]
        self._api.MT_Open_USB.argtypes = []

        # 零位模式相关函数
        self._api.MT_Set_Axis_Mode_Home_Home_Switch.argtypes = [c_uint16]
        self._api.MT_Set_Axis_Mode_Home_Encoder_Index.argtypes = [c_uint16]
        self._api.MT_Set_Axis_Mode_Home_Encoder_Home_Switch.argtypes = [c_uint16]
        self._api.MT_Set_Axis_Home_V.argtypes = [c_uint16, c_int32]
        self._api.MT_Set_Axis_Home_Acc.argtypes = [c_uint16, c_int32]
        self._api.MT_Set_Axis_Home_Dec.argtypes = [c_uint16, c_int32]
        self._api.MT_Set_Axis_Home_Stop.argtypes = [c_uint16]
        self._api.MT_Set_Axis_Halt.argtypes = [c_uint16]

        # 状态查询相关函数
        self._api.MT_Get_Axis_Status2.argtypes = [
            c_uint16,
            POINTER(c_int32),
            POINTER(c_int32),
            POINTER(c_int32),
            POINTER(c_int32),
            POINTER(c_int32),
            POINTER(c_int32),
        ]
        self._api.MT_Get_Axis_Status_Run.argtypes = [c_uint16, POINTER(c_int32)]
        self._api.MT_Get_Axis_Status_Zero.argtypes = [c_uint16, POINTER(c_int32)]
        self._api.MT_Get_Axis_Mode.argtypes = [c_uint16, POINTER(c_int32)]
        self._api.MT_Get_Encoder_Num.argtypes = [POINTER(c_int32)]

        # 位置相关函数
        self._api.MT_Get_Axis_Software_P.argtypes = [c_uint16, POINTER(c_int32)]
        self._api.MT_Set_Axis_Software_P.argtypes = [c_uint16, c_int32]

        # 位置模式相关函数
        self._api.MT_Set_Axis_Mode_Position_Open.argtypes = [c_uint16]
        self._api.MT_Set_Axis_Position_Acc.argtypes = [c_uint16, c_int32]
        self._api.MT_Set_Axis_Position_Dec.argtypes = [c_uint16, c_int32]
        self._api.MT_Set_Axis_Position_V_Max.argtypes = [c_uint16, c_int32]
        self._api.MT_Set_Axis_Position_P_Target_Rel.argtypes = [c_uint16, c_int32]
        self._api.MT_Set_Axis_Position_P_Target_Abs.argtypes = [c_uint16, c_int32]
        self._api.MT_Set_Axis_Position_Stop.argtypes = [c_uint16]

        logger.debug("API函数参数类型设置完成")

        return True

    def _connect_usb(self) -> bool:
        """通过USB连接控制器

        Returns:
            bool: 连接是否成功
        """
        if not self._is_initialized:
            logger.error("API未初始化，无法连接", exc_info=True)
            return self._is_connected

        try:
            result = self._api.MT_Open_USB()
            if result == 0:
                # 检查连接状态
                check_result = self._api.MT_Check()
                if check_result == 0:
                    self._is_connected = True
                    logger.debug("USB连接成功")
                    return self._is_connected
                else:
                    self._is_connected = False
                    logger.error(f"连接检查失败，错误码: {check_result}", exc_info=True)
                    return self._is_connected
            else:
                self._is_connected = False
                logger.error(f"USB连接失败，错误码: {result}", exc_info=True)
                return self._is_connected

        except Exception as e:
            self._is_connected = False
            logger.error(f"USB连接过程中发生异常: {e}", exc_info=True)
            return self._is_connected

    def get_hardware_info(self) -> dict[str, str | int]:
        """获取硬件信息

        Returns:
            dict: 包含硬件信息的字典
        """
        if not self._is_connected:
            logger.error("未连接到控制器，无法获取硬件信息", exc_info=True)
            return {}

        info: dict[str, str | int] = {}

        try:
            # 获取产品资源信息
            resource = c_int32(0)
            result = self._api.MT_Get_Product_Resource(byref(resource))
            if result == 0:
                info["resource"] = resource.value
                logger.info(f"产品资源信息: 0x{resource.value:08X}")
            else:
                logger.warning(f"获取产品资源失败，错误码: {result}")

            # 获取产品ID
            product_id = create_string_buffer(16)
            result = self._api.MT_Get_Product_ID(product_id)
            if result == 0:
                info["product_id"] = product_id.value.decode("gbk")
                logger.info(f"产品ID: {info['product_id']}")
            else:
                logger.warning(f"获取产品ID失败，错误码: {result}")

            # 获取产品序列号
            serial_number = create_string_buffer(12)
            result = self._api.MT_Get_Product_SN(serial_number)
            if result == 0:
                info["serial_number"] = serial_number.value.decode("gbk")
                logger.info(f"产品序列号: {info['serial_number']}")
            else:
                logger.warning(f"获取产品序列号失败，错误码: {result}")

        except Exception as e:
            logger.error(f"获取硬件信息时发生异常: {e}", exc_info=True)

        return info

    def get_axis_status(self, axis: int = 0) -> dict[str, int | bool | str]:
        """获取轴状态信息（使用MT_Get_Axis_Status2一次性获取完整信息）

        Args:
            axis: 轴序号

        Returns:
            dict: 包含轴状态信息的字典
        """
        if not self._is_connected:
            logger.error("未连接到控制器，无法获取轴状态", exc_info=True)
            return {}

        status: dict[str, int | bool | str] = {}

        try:
            # 使用MT_Get_Axis_Status2一次性获取所有状态信息
            run_status = c_int32(0)
            direction = c_int32(0)
            neg_limit = c_int32(0)
            pos_limit = c_int32(0)
            zero_status = c_int32(0)
            mode = c_int32(0)

            result = self._api.MT_Get_Axis_Status2(
                axis,
                byref(run_status),
                byref(direction),
                byref(neg_limit),
                byref(pos_limit),
                byref(zero_status),
                byref(mode),
            )

            if result == 0:
                # 运行状态
                status["is_running"] = bool(run_status.value)

                # 运动方向
                status["direction"] = "正向" if direction.value == 1 else "负向"
                status["direction_code"] = direction.value

                # 限位状态
                status["neg_limit_active"] = bool(neg_limit.value)
                status["pos_limit_active"] = bool(pos_limit.value)

                # 零位状态
                status["at_zero"] = bool(zero_status.value)

                # 工作模式
                mode_names = {
                    0: "零位模式",
                    1: "速度模式",
                    2: "位置模式",
                    3: "直线插补",
                    4: "圆弧插补",
                }
                status["mode"] = mode_names.get(mode.value, f"未知模式({mode.value})")
                status["mode_code"] = mode.value

                logger.debug(
                    f"轴{axis}状态: 运行={status['is_running']}, "
                    f"方向={status['direction']}, "
                    f"负限位={status['neg_limit_active']}, "
                    f"正限位={status['pos_limit_active']}, "
                    f"零位={status['at_zero']}, 模式={status['mode']}"
                )
            else:
                logger.error(f"获取轴{axis}状态失败，错误码: {result}", exc_info=True)

        except Exception as e:
            logger.error(f"获取轴{axis}状态时发生异常: {e}", exc_info=True)

        return status

    def get_motor_parameters(self) -> dict[str, float | int]:
        """获取当前电机参数

        Returns:
            dict: 包含电机参数的字典
        """
        return {
            "step_angle": self._step_angle,
            "subdivision": self._subdivision,
            "pitch": self._pitch,
            "transmission_ratio": self._transmission_ratio,
        }  # 参数均为不可变对象，直接传出是安全的

    def _check_motor_parameters(self) -> bool:
        """检查电机参数是否有效（必须为正值）

        Returns:
            bool: 参数是否有效
        """
        # 参数在初始化时已设置，不会为None，只需检查是否为有效值
        if self._step_angle <= 0 or self._subdivision <= 0 or self._pitch <= 0:
            logger.error("电机参数无效，无法进行换算", exc_info=True)
            return False
        return True

    def mm_to_steps(self, distance_mm: float) -> int:
        """将物理距离（mm）转换为脉冲数

        根据电机参数（步进角度、细分数、螺距、传动比）计算对应的脉冲数。
        优先使用API函数进行转换，连接失败时使用本地计算。

        计算公式：
        steps = (distance_mm / pitch) * (360° / step_angle) * subdivision *
                transmission_ratio

        Args:
            distance_mm: 物理距离（mm），可以为正值或负值

        Returns:
            int: 对应的脉冲数，正值表示正向运动，负值表示负向运动

        Raises:
            ValueError: 当电机参数无效时（步进角度、细分数或螺距≤0）
        """
        if not self._check_motor_parameters():
            raise ValueError("电机参数无效，无法进行换算")

        if not self._is_connected:
            logger.warning("未连接到控制器，使用本地计算")

        try:
            # 根据连接情况选择计算方法
            if self._is_connected:
                # 使用API函数进行转换
                steps = self._api.MT_Help_Step_Line_Real_To_Steps(
                    self._step_angle,
                    self._subdivision,
                    self._pitch,
                    self._transmission_ratio,
                    distance_mm,
                )
                logger.debug(f"API计算: {distance_mm}mm -> {steps}步")
            else:
                # 本地计算：steps = (distance_mm / pitch) * (360 / step_angle) *
                # subdivision * transmission_ratio
                steps_per_revolution = (360.0 / self._step_angle) * self._subdivision
                steps = int(
                    (distance_mm / self._pitch)
                    * steps_per_revolution
                    * self._transmission_ratio
                )
                logger.debug(f"本地计算: {distance_mm}mm -> {steps}步")

            return steps

        except Exception as e:
            logger.error(f"距离转换脉冲数时发生异常: {e}", exc_info=True)
            raise

    def steps_to_mm(self, steps: int) -> float:
        """将脉冲数转换为物理距离（mm）

        根据电机参数（步进角度、细分数、螺距、传动比）计算对应的物理距离。
        优先使用API函数进行转换，连接失败时使用本地计算。

        计算公式：
        distance = (steps / steps_per_revolution) * pitch / transmission_ratio
        其中 steps_per_revolution = (360° / step_angle) * subdivision

        Args:
            steps: 脉冲数，可以为正值或负值

        Returns:
            float: 对应的物理距离（mm），正值表示正向距离，负值表示负向距离

        Raises:
            ValueError: 当电机参数无效时（步进角度、细分数或螺距≤0）
        """
        if not self._check_motor_parameters():
            raise ValueError("电机参数未完整设置")

        if not self._is_connected:
            logger.warning("未连接到控制器，使用本地计算")

        try:
            # 根据连接情况选择计算方法
            if self._is_connected:
                # 使用API函数进行转换
                distance = self._api.MT_Help_Step_Line_Steps_To_Real(
                    self._step_angle,
                    self._subdivision,
                    self._pitch,
                    self._transmission_ratio,
                    steps,
                )
                logger.debug(f"API计算: {steps}步 -> {distance}mm")
            else:
                # 本地计算：distance = (steps / steps_per_revolution) * pitch /
                # transmission_ratio
                steps_per_revolution = (360.0 / self._step_angle) * self._subdivision
                distance = (
                    (steps / steps_per_revolution)
                    * self._pitch
                    / self._transmission_ratio
                )
                logger.debug(f"本地计算: {steps}步 -> {distance}mm")

            return distance

        except Exception as e:
            logger.error(f"脉冲数转换距离时发生异常: {e}", exc_info=True)
            raise

    def _calibrate_one_axis(
        self,
        axis: int = 0,
        home_speed: int = -1000,
        acceleration: int = 500,
        deceleration: int = 500,
        timeout_seconds: float = 360.0,
    ) -> bool:
        """执行单轴零位校准

        Args:
            axis: 轴序号，0为X轴，1为Y轴
            home_speed: 零位模式速度（Hz/s），负值表示负向查找。慢速查找的精度更高。
            acceleration: 加速度（Hz/s²），电机默认为500
            deceleration: 减速度（Hz/s²），电机默认为500
            timeout_seconds: 超时时间（秒）

        Returns:
            bool: 校准是否成功
        """
        if not self._is_connected:
            logger.error("未连接到控制器，无法执行零位校准", exc_info=True)
            return False

        try:
            # 停止当前运动
            self._api.MT_Set_Axis_Halt(axis)

            # 设置零位模式
            result = self._api.MT_Set_Axis_Mode_Home_Home_Switch(axis)

            if result != 0:
                logger.error(f"设置零位模式失败，错误码: {result}", exc_info=True)
                return False

            # 设置零位参数
            self._api.MT_Set_Axis_Home_V(axis, home_speed)
            self._api.MT_Set_Axis_Home_Acc(axis, acceleration)
            self._api.MT_Set_Axis_Home_Dec(axis, deceleration)

            logger.info(
                f"轴{axis}: 开始零位校准，速度={home_speed}Hz/s, "
                f"加速度={acceleration}Hz/s², 减速度={deceleration}Hz/s²"
            )

            # 等待校准完成
            start_time = time.time()
            logger.info(
                f"轴{axis}: 等待零位校准完成，最大等待时间{timeout_seconds}秒..."
            )

            while time.time() - start_time < timeout_seconds:
                # 检查运行状态
                run_status = c_int32(0)
                result = self._api.MT_Get_Axis_Status_Run(axis, byref(run_status))

                if result == 0 and run_status.value == 0:
                    # 电机已停止，等待系统处理零位设置
                    logger.info(f"轴{axis}: 电机已停止，等待系统处理零位设置...")
                    time.sleep(1.0)  # 等待1秒让系统处理零位设置

                    # 检查当前位置是否已设为0（零位校准成功的标志）
                    current_pos = c_int32(200)
                    pos_result = self._api.MT_Get_Axis_Software_P(
                        axis, byref(current_pos)
                    )

                    if pos_result == 0:
                        # 检查位置是否为0
                        if abs(current_pos.value) < 100:  # 允许小的误差
                            logger.info(
                                f"轴{axis}: 零位校准成功，"
                                f"当前位置已自动设为{current_pos.value}"
                            )
                            self._current_position[axis] = 0.0
                            return True
                        else:
                            # 如果位置不为0，可能是碰到限位但未找到零位开关
                            # 根据API文档，此时应该手动设置当前位置为0
                            logger.info(
                                f"轴{axis}: 当前位置 = {current_pos.value}，"
                                f"手动设置零位"
                            )
                            self._api.MT_Set_Axis_Software_P(axis, 0)
                            time.sleep(0.5)  # 等待设置生效

                            # 再次检查位置
                            self._api.MT_Get_Axis_Software_P(axis, byref(current_pos))
                            logger.info(
                                f"轴{axis}: 手动设置完成，"
                                f"新位置值 = {current_pos.value}"
                            )
                            self._current_position[axis] = 0.0
                            return True
                    else:
                        logger.warning(
                            f"轴{axis}: 无法读取当前位置，错误码: {pos_result}"
                        )
                        return False

                time.sleep(1.0)  # 1000ms检查间隔

            # 超时处理
            logger.warning(f"轴{axis}: 零位校准超时，强制停止")
            self._api.MT_Set_Axis_Home_Stop(axis)
            return False

        except Exception as e:
            logger.error(f"轴{axis}: 零位校准过程中发生异常: {e}", exc_info=True)
            try:
                self._api.MT_Set_Axis_Home_Stop(axis)
            except Exception:
                pass
            return False

    def calibrate_all_axis(self) -> bool:
        """自动双轴零位校准

        该方法执行完整的双轴零位校准流程：
        1. 快速移动到负限位附近（10mm, 10mm）
        2. 依次校准X轴和Y轴到零位
        3. 建立坐标系，零位为(0, 0)

        校准成功后，_is_calibrated状态将设为True，
        后续可以使用绝对坐标进行精确定位。

        Returns:
            bool: 校准是否成功完成

        Note:
            - 校准过程需要一定时间（一般小于1分钟）
            - 校准失败时会自动停止电机运动
            - 建议在系统启动后首次使用前执行校准
        """
        if not self._is_connected:
            logger.error("未连接到控制器，无法执行自动校准", exc_info=True)
            self._is_calibrated = False
            return self._is_calibrated

        logger.info("开始自动零位校准...")

        try:
            # 快速移动到负限位附近
            move_success = self.move_absolute_2D(x_mm=10.0, y_mm=10.0)

            if not move_success:
                logger.error("自动校准前快速移动到负限位附近失败", exc_info=True)
                self._is_calibrated = False
                return self._is_calibrated

            # 校准X轴（轴序号0）
            logger.info("开始校准X轴...")
            x_success = self._calibrate_one_axis(axis=0, timeout_seconds=20.0)

            if not x_success:
                logger.error("X轴校准失败", exc_info=True)
                self._is_calibrated = False
                return self._is_calibrated

            # 校准Y轴（轴序号1）
            logger.info("开始校准Y轴...")
            y_success = self._calibrate_one_axis(axis=1, timeout_seconds=20.0)

            if not y_success:
                logger.error("Y轴校准失败", exc_info=True)
                self._is_calibrated = False
                return self._is_calibrated

            # 所有轴校准成功
            self._is_calibrated = True
            logger.info("自动零位校准完成，所有轴已校准")
            return self._is_calibrated

        except Exception as e:
            logger.error(f"自动校准过程中发生异常: {e}", exc_info=True)
            self._is_calibrated = False
            return self._is_calibrated

    def _calculate_smart_motion_parameters(
        self, distance_abs: float
    ) -> tuple[int, int, int, float]:
        """根据运动距离智能计算运动参数（简化版）

        该方法根据运动距离的绝对值自动计算最优的运动参数（简化版）：
        - 加速度/减速度：固定5000
        - 最大速度：固定20000
        - 超时时间：30mm以下=10s，30-120mm线性插值到15s，120mm以上按比例延长到25s

        Args:
            distance_abs: 运动距离的绝对值（毫米）

        Returns:
            tuple: (max_speed, acceleration, deceleration, timeout_seconds)
        """
        # 确保输入距离为正
        distance_abs = abs(distance_abs)

        # 设置加速度/减速度
        acceleration = deceleration = 5000

        # 设置最大速度
        max_speed = 20000

        # 智能设置超时时间
        if distance_abs <= 30.0:
            timeout_seconds = 10.0
        elif distance_abs <= 120.0:
            # 30-120mm之间线性插值：从10s到15s
            ratio = (distance_abs - 30.0) / (120.0 - 30.0)
            timeout_seconds = 10.0 + ratio * (15.0 - 10.0)
        else:
            # 120mm以上：从15s线性延长到320mm的25s，并无限外延
            ratio = (distance_abs - 120.0) / (320.0 - 120.0)
            timeout_seconds = 15.0 + ratio * (25.0 - 15.0)
            # 确保超时时间不会过短
            timeout_seconds = max(timeout_seconds, 15.0)

        logger.debug(
            f"智能参数计算 - 距离={distance_abs:.3f}mm: 超时={timeout_seconds:.1f}s"
        )

        return max_speed, acceleration, deceleration, timeout_seconds

    def _calculate_smart_motion_parameters_backup(  # 暂停维护
        self, distance_abs: float
    ) -> tuple[int, int, int, float]:
        """根据运动距离智能计算运动参数

        该方法根据运动距离的绝对值自动计算最优的运动参数：
        - 加速度/减速度：3mm以下=500，30mm以上=5000，中间线性插值
        - 最大速度：3mm以下=1000，3-30mm线性插值到10000，
          30-120mm线性插值到20000，120mm以上=20000
        - 超时时间：30mm以下=10s，30-120mm线性插值到15s，120mm以上按比例延长到25s

        Args:
            distance_abs: 运动距离的绝对值（毫米）

        Returns:
            tuple: (max_speed, acceleration, deceleration, timeout_seconds)
        """
        # 确保输入距离为正
        distance_abs = abs(distance_abs)

        # 智能设置加速度/减速度
        if distance_abs <= 3.0:
            acceleration = deceleration = 500
        elif distance_abs >= 30.0:
            acceleration = deceleration = 5000
        else:
            # 3-30mm之间线性插值：从500到5000
            ratio = (distance_abs - 3.0) / (30.0 - 3.0)
            acceleration = deceleration = int(500 + ratio * (5000 - 500))

        # 智能设置最大速度
        if distance_abs <= 3.0:
            max_speed = 1000
        elif distance_abs <= 30.0:
            # 3-30mm之间线性插值：从1000到10000
            ratio = (distance_abs - 3.0) / (30.0 - 3.0)
            max_speed = int(1000 + ratio * (10000 - 1000))
        elif distance_abs <= 120.0:
            # 30-120mm之间线性插值：从10000到20000
            ratio = (distance_abs - 30.0) / (120.0 - 30.0)
            max_speed = int(10000 + ratio * (20000 - 10000))
        else:
            max_speed = 20000

        # 智能设置超时时间
        if distance_abs <= 30.0:
            timeout_seconds = 10.0
        elif distance_abs <= 120.0:
            # 30-120mm之间线性插值：从10s到15s
            ratio = (distance_abs - 30.0) / (120.0 - 30.0)
            timeout_seconds = 10.0 + ratio * (15.0 - 10.0)
        else:
            # 120mm以上：从15s线性延长到320mm的25s，并无限外延
            ratio = (distance_abs - 120.0) / (320.0 - 120.0)
            timeout_seconds = 15.0 + ratio * (25.0 - 15.0)
            # 确保超时时间不会过短
            timeout_seconds = max(timeout_seconds, 15.0)

        logger.debug(
            f"智能参数计算 - 距离={distance_abs:.3f}mm: "
            f"速度={max_speed}Hz/s, 加速度={acceleration}Hz/s², "
            f"减速度={deceleration}Hz/s², 超时={timeout_seconds:.1f}s"
        )

        return max_speed, acceleration, deceleration, timeout_seconds

    def _move_relative_1D(
        self,
        axis: int = 0,
        distance_mm: float = 0.0,
        max_speed: int = 1000,
        acceleration: int = 500,
        deceleration: int = 500,
        timeout_seconds: float = 360.0,
    ) -> bool:
        """相对运动指定距离（毫米）（默认速度较慢）

        Args:
            axis: 轴序号，0为X轴，1为Y轴
            distance_mm: 运动距离（毫米），正值为正向，负值为负向
            max_speed: 最大速度（Hz/s）
            acceleration: 加速度（Hz/s²），电机默认为500
            deceleration: 减速度（Hz/s²），电机默认为500
            timeout_seconds: 超时时间（秒）

        Returns:
            bool: 运动是否成功完成
        """
        # 检查连接状态
        if not self._is_connected:
            logger.error("未连接到控制器，无法执行运动", exc_info=True)
            return False

        # 当移动距离为0时，直接返回成功
        if distance_mm == 0.0:
            logger.debug(f"轴{axis}: 运动距离为0，无需移动")
            return True

        # 当移动距离不为0时，执行移动
        try:
            # 将距离转换为脉冲数
            steps = self.mm_to_steps(distance_mm)

            # 停止当前运动
            self._api.MT_Set_Axis_Halt(axis)
            time.sleep(0.1)

            # 切换到位置模式
            result = self._api.MT_Set_Axis_Mode_Position_Open(axis)
            if result != 0:
                logger.error(
                    f"轴{axis}: 设置位置模式失败，错误码: {result}", exc_info=True
                )
                return False

            # 设置运动参数
            self._api.MT_Set_Axis_Position_V_Max(axis, max_speed)
            self._api.MT_Set_Axis_Position_Acc(axis, acceleration)
            self._api.MT_Set_Axis_Position_Dec(axis, deceleration)

            # 静默获取运动前位置
            start_pos = c_int32(0)
            self._api.MT_Get_Axis_Software_P(axis, byref(start_pos))

            # 执行相对运动
            result = self._api.MT_Set_Axis_Position_P_Target_Rel(axis, steps)
            if result != 0:
                logger.error(
                    f"轴{axis}: 启动相对运动失败，错误码: {result}", exc_info=True
                )
                return False

            logger.debug(f"轴{axis}: 开始相对运动...")

            # 等待运动完成
            start_time = time.time()

            while time.time() - start_time < timeout_seconds:
                # 检查运行状态
                run_status = c_int32(0)
                result = self._api.MT_Get_Axis_Status_Run(axis, byref(run_status))

                if result == 0 and run_status.value == 0:
                    # 运动已完成，获取最终位置
                    end_pos = c_int32(0)
                    self._api.MT_Get_Axis_Software_P(axis, byref(end_pos))
                    # 更新当前位置
                    self._current_position[axis] = self.steps_to_mm(end_pos.value)

                    # 静默计算实际运动步数
                    actual_steps = end_pos.value - start_pos.value

                    # （基于脉冲步）检查运动精度
                    if actual_steps == steps:
                        logger.debug(f"轴{axis}: 运动完成，脉冲步数准确")
                        return True  # 一切正常
                    else:
                        error_ratio = (
                            abs(actual_steps - steps) / abs(steps) if steps != 0 else 0
                        )
                        logger.warning(
                            f"轴{axis}: 运动脉冲数异常，误差比例 = {error_ratio:.4f}"
                        )
                        return True  # 仍然认为运动成功，但记录警告

                time.sleep(0.2)  # 200ms检查间隔

            # 超时处理
            logger.warning(f"轴{axis}: 运动超时，强制停止")
            self._api.MT_Set_Axis_Position_Stop(axis)
            return False

        except Exception as e:
            logger.error(f"轴{axis}: 运动过程中发生异常: {e}", exc_info=True)
            try:
                self._api.MT_Set_Axis_Position_Stop(axis)
            except Exception:
                pass
            return False

    def move_absolute_1D(
        self,
        axis: int = 0,
        position_mm: float = 0.0,
    ) -> bool:
        """绝对运动到指定位置（毫米）（智能设置参数）

        该方法根据运动距离自动设置最优的速度、加速度和超时参数，运动速度较快。

        Args:
            axis: 轴序号，0为X轴，1为Y轴
            position_mm: 目标绝对位置（毫米），0mm代表负限位（零位）

        Returns:
            bool: 运动是否成功完成
        """
        # 检查连接状态
        if not self._is_connected:
            logger.error("未连接到控制器，无法执行运动", exc_info=True)
            return False

        # 检查目标位置是否合理（不允许明显超出行程范围）
        if position_mm < 0 or position_mm > self.MAX_POSITION:
            logger.error(
                f"轴{axis}: 目标位置明显超出合理范围，当前目标={position_mm:.3f}mm",
                exc_info=True,
            )
            return False

        try:
            # 获取当前位置（相对于零位的位置）
            current_pos = c_int32(0)
            self._api.MT_Get_Axis_Software_P(axis, byref(current_pos))
            current_mm = self.steps_to_mm(current_pos.value)

            # 计算相对距离
            distance_mm = position_mm - current_mm

            # 当移动距离为0时，直接返回成功
            if distance_mm == 0.0:
                logger.debug(f"轴{axis}: 运动距离为0，无需移动")
                return True

            # 当移动距离不为0时，继续执行移动
            # 计算距离绝对值
            distance_abs = abs(distance_mm)

            logger.debug(
                f"轴{axis}: 绝对运动: 当前位置={current_mm:.3f}mm, "
                f"需要移动距离={distance_mm:.3f}mm"
            )

            # 使用智能参数计算方法
            max_speed, acceleration, deceleration, timeout_seconds = (
                self._calculate_smart_motion_parameters(distance_abs)
            )

            logger.debug(
                f"轴{axis}: 智能参数设置: 最大速度={max_speed}Hz/s, "
                f"加速度={acceleration}Hz/s², 减速度={deceleration}Hz/s², "
                f"超时时间={timeout_seconds:.1f}s"
            )

            # 调用相对运动方法
            return self._move_relative_1D(
                axis=axis,
                distance_mm=distance_mm,
                max_speed=max_speed,
                acceleration=acceleration,
                deceleration=deceleration,
                timeout_seconds=timeout_seconds,
            )

        except Exception as e:
            logger.error(f"轴{axis}: 绝对运动过程中发生异常: {e}", exc_info=True)
            return False

    def get_current_position_1D(self, axis: int = 0) -> float:
        """获取当前位置（毫米）

        Args:
            axis: 轴序号，0为X轴，1为Y轴

        Returns:
            float: 当前位置（毫米）
        """
        if not self._is_connected:
            logger.error("未连接到控制器，无法获取位置", exc_info=True)
            return -1.0

        try:
            current_pos = c_int32(0)
            result = self._api.MT_Get_Axis_Software_P(axis, byref(current_pos))

            if result == 0:
                position_mm = self.steps_to_mm(current_pos.value)
                logger.debug(
                    f"轴{axis}: 当前位置 = {current_pos.value}步 ({position_mm:.3f}mm)"
                )
                self._current_position[axis] = position_mm
                return position_mm
            else:
                logger.error(f"轴{axis}: 获取位置失败，错误码: {result}", exc_info=True)
                return -1.0

        except Exception as e:
            logger.error(f"轴{axis}: 获取位置时发生异常: {e}", exc_info=True)
            return -1.0

    def _move_relative_2D(
        self,
        x_distance_mm: float = 0.0,
        x_max_speed: int = 1000,
        x_acceleration: int = 500,
        x_deceleration: int = 500,
        x_timeout_seconds: float = 360.0,
        y_distance_mm: float = 0.0,
        y_max_speed: int = 1000,
        y_acceleration: int = 500,
        y_deceleration: int = 500,
        y_timeout_seconds: float = 360.0,
    ) -> bool:
        """两轴并发相对运动（默认速度较慢）

        该方法可以让X轴和Y轴同时开始运动，提高运动效率。

        Args:
            x_distance_mm: X轴运动距离（毫米），正值为正向，负值为负向
            x_max_speed: X轴最大速度（Hz/s）
            x_acceleration: X轴加速度（Hz/s²）
            x_deceleration: X轴减速度（Hz/s²）
            x_timeout_seconds: X轴超时时间（秒）
            y_distance_mm: Y轴运动距离（毫米），正值为正向，负值为负向
            y_max_speed: Y轴最大速度（Hz/s）
            y_acceleration: Y轴加速度（Hz/s²）
            y_deceleration: Y轴减速度（Hz/s²）
            y_timeout_seconds: Y轴超时时间（秒）

        Returns:
            bool: 两轴运动是否都成功完成
        """
        # 检查连接状态
        if not self._is_connected:
            logger.error("未连接到控制器，无法执行2D运动", exc_info=True)
            return False

        # 当x_distance_mm为0时，退化为Y轴1D移动
        if x_distance_mm == 0.0:
            # 当y_distance_mm也为0时，直接返回成功
            if y_distance_mm == 0.0:
                logger.debug("两轴运动距离均为0，无需移动")
                return True
            else:
                logger.debug("X轴运动距离为0，退化为Y轴1D移动")
                return self._move_relative_1D(
                    axis=1,
                    distance_mm=y_distance_mm,
                    max_speed=y_max_speed,
                    acceleration=y_acceleration,
                    deceleration=y_deceleration,
                    timeout_seconds=y_timeout_seconds,
                )

        # 当x_distance_mm不为0，但y_distance_mm为0时，退化为X轴1D移动
        if y_distance_mm == 0.0:
            logger.debug("Y轴运动距离为0，退化为X轴1D移动")
            return self._move_relative_1D(
                axis=0,
                distance_mm=x_distance_mm,
                max_speed=x_max_speed,
                acceleration=x_acceleration,
                deceleration=x_deceleration,
                timeout_seconds=x_timeout_seconds,
            )

        # 当两轴距离都不为0时，继续执行2D运动
        try:
            # 将距离转换为脉冲数
            x_steps = self.mm_to_steps(x_distance_mm)
            y_steps = self.mm_to_steps(y_distance_mm)

            # 停止当前运动
            self._api.MT_Set_Axis_Halt(0)  # X轴
            self._api.MT_Set_Axis_Halt(1)  # Y轴
            time.sleep(0.1)

            # 设置X轴位置模式和参数
            result_x = self._api.MT_Set_Axis_Mode_Position_Open(0)
            if result_x != 0:
                logger.error(f"X轴设置位置模式失败，错误码: {result_x}", exc_info=True)
                return False

            self._api.MT_Set_Axis_Position_V_Max(0, x_max_speed)
            self._api.MT_Set_Axis_Position_Acc(0, x_acceleration)
            self._api.MT_Set_Axis_Position_Dec(0, x_deceleration)

            # 设置Y轴位置模式和参数
            result_y = self._api.MT_Set_Axis_Mode_Position_Open(1)
            if result_y != 0:
                logger.error(f"Y轴设置位置模式失败，错误码: {result_y}", exc_info=True)
                return False

            self._api.MT_Set_Axis_Position_V_Max(1, y_max_speed)
            self._api.MT_Set_Axis_Position_Acc(1, y_acceleration)
            self._api.MT_Set_Axis_Position_Dec(1, y_deceleration)

            # 静默获取运动前位置
            x_start_pos = c_int32(0)
            y_start_pos = c_int32(0)
            self._api.MT_Get_Axis_Software_P(0, byref(x_start_pos))
            self._api.MT_Get_Axis_Software_P(1, byref(y_start_pos))

            # 同时启动两轴相对运动
            result_x = self._api.MT_Set_Axis_Position_P_Target_Rel(0, x_steps)
            result_y = self._api.MT_Set_Axis_Position_P_Target_Rel(1, y_steps)

            if result_x != 0:
                logger.error(f"X轴启动相对运动失败，错误码: {result_x}", exc_info=True)
                return False
            if result_y != 0:
                logger.error(f"Y轴启动相对运动失败，错误码: {result_y}", exc_info=True)
                return False

            logger.debug("2D并发运动已启动...")

            # 等待两轴运动完成
            max_timeout = max(x_timeout_seconds, y_timeout_seconds)
            start_time = time.time()
            x_completed = False
            y_completed = False

            while time.time() - start_time < max_timeout:
                # 检查X轴运行状态
                if not x_completed:
                    x_run_status = c_int32(0)
                    result = self._api.MT_Get_Axis_Status_Run(0, byref(x_run_status))
                    if result == 0 and x_run_status.value == 0:
                        x_completed = True
                        logger.debug("X轴运动完成")

                # 检查Y轴运行状态
                if not y_completed:
                    y_run_status = c_int32(0)
                    result = self._api.MT_Get_Axis_Status_Run(1, byref(y_run_status))
                    if result == 0 and y_run_status.value == 0:
                        y_completed = True
                        logger.debug("Y轴运动完成")

                # 如果两轴都完成，退出循环
                if x_completed and y_completed:
                    break

                time.sleep(0.2)  # 200ms检查间隔

            # 超时处理
            if not (x_completed and y_completed):
                logger.warning("2D运动超时，强制停止")
                self._api.MT_Set_Axis_Position_Stop(0)
                self._api.MT_Set_Axis_Position_Stop(1)
                return False

            # 获取运动后位置
            x_end_pos = c_int32(0)
            y_end_pos = c_int32(0)
            self._api.MT_Get_Axis_Software_P(0, byref(x_end_pos))
            self._api.MT_Get_Axis_Software_P(1, byref(y_end_pos))

            # 更新当前位置
            self._current_position = [
                self.steps_to_mm(x_end_pos.value),
                self.steps_to_mm(y_end_pos.value),
            ]

            # 计算实际运动步数
            x_actual_steps = x_end_pos.value - x_start_pos.value
            y_actual_steps = y_end_pos.value - y_start_pos.value

            # （基于脉冲步）检查运动精度
            if x_actual_steps == x_steps and y_actual_steps == y_steps:
                logger.debug("2D运动完成，脉冲步数准确")
                return True  # 一切正常
            else:
                x_error_ratio = (
                    abs(x_actual_steps - x_steps) / abs(x_steps) if x_steps != 0 else 0
                )
                y_error_ratio = (
                    abs(y_actual_steps - y_steps) / abs(y_steps) if y_steps != 0 else 0
                )
                logger.warning(
                    f"2D运动脉冲数异常，X轴误差比例={x_error_ratio:.4f}, "
                    f"Y轴误差比例={y_error_ratio:.4f}"
                )
                return True  # 仍然认为运动成功，但记录警告

        except Exception as e:
            logger.error(f"2D运动过程中发生异常: {e}", exc_info=True)
            try:
                self._api.MT_Set_Axis_Position_Stop(0)
                self._api.MT_Set_Axis_Position_Stop(1)
            except Exception:
                pass
            return False

    def move_absolute_2D(
        self,
        x_mm: float = 0.0,
        y_mm: float = 0.0,
    ) -> bool:
        """并发移动电机到指定的绝对坐标位置（智能设置参数）

        该方法使用并发运动，让X轴和Y轴同时开始运动，提高运动效率。
        运动参数（速度、加速度、超时时间）根据各轴的运动距离智能设置。

        Args:
            x_mm: X轴目标绝对位置（毫米），0mm代表X轴负限位（零位）
            y_mm: Y轴目标绝对位置（毫米），0mm代表Y轴负限位（零位）

        Returns:
            bool: 移动是否成功完成

        Note:
            - 坐标范围：0-MAX_POSITION mm（基于电机行程限制）
            - 当某轴运动距离为0时，自动退化为单轴运动
            - 运动前会自动检查目标位置的合理性
            - 支持亚毫米级精度定位
        """
        # 检查连接状态
        if not self._is_connected:
            logger.error("未连接到控制器，无法执行运动", exc_info=True)
            return False

        # 检查目标位置是否在合理范围内
        if x_mm < 0 or x_mm > self.MAX_POSITION or y_mm < 0 or y_mm > self.MAX_POSITION:
            logger.error(
                f"目标位置超出合理范围: X={x_mm:.3f}mm, Y={y_mm:.3f}mm", exc_info=True
            )
            return False

        try:
            # 获取当前位置
            x_current_pos = c_int32(0)
            y_current_pos = c_int32(0)
            self._api.MT_Get_Axis_Software_P(0, byref(x_current_pos))
            self._api.MT_Get_Axis_Software_P(1, byref(y_current_pos))

            x_current_mm = self.steps_to_mm(x_current_pos.value)
            y_current_mm = self.steps_to_mm(y_current_pos.value)

            # 计算相对距离
            x_distance_mm = x_mm - x_current_mm
            y_distance_mm = y_mm - y_current_mm

            # 当x_distance_mm为0时，退化为Y轴1D移动
            if x_distance_mm == 0.0:
                # 当y_distance_mm也为0时，直接返回成功
                if y_distance_mm == 0.0:
                    logger.debug("两轴运动距离均为0，无需移动")
                    return True
                else:
                    logger.debug("X轴运动距离为0，退化为Y轴1D移动")
                    return self.move_absolute_1D(
                        axis=1,
                        position_mm=y_mm,
                    )

            # 当x_distance_mm不为0，但y_distance_mm为0时，退化为X轴1D移动
            if y_distance_mm == 0.0:
                logger.debug("Y轴运动距离为0，退化为X轴1D移动")
                return self.move_absolute_1D(
                    axis=0,
                    position_mm=x_mm,
                )

            # 当两轴距离都不为0时，继续执行2D运动
            # 计算距离绝对值
            x_distance_abs = abs(x_distance_mm)
            y_distance_abs = abs(y_distance_mm)

            logger.debug(f"当前位置: X={x_current_mm:.3f}mm, Y={y_current_mm:.3f}mm")
            logger.debug(
                f"需要运动距离: X={x_distance_mm:.3f}mm, Y={y_distance_mm:.3f}mm"
            )

            # 为X轴计算智能参数
            x_max_speed, x_acceleration, x_deceleration, x_timeout_seconds = (
                self._calculate_smart_motion_parameters(x_distance_abs)
            )

            # 为Y轴计算智能参数
            y_max_speed, y_acceleration, y_deceleration, y_timeout_seconds = (
                self._calculate_smart_motion_parameters(y_distance_abs)
            )

            logger.debug(
                f"X轴智能参数: 速度={x_max_speed}Hz/s, "
                f"加速度={x_acceleration}Hz/s², 超时={x_timeout_seconds:.1f}s"
            )
            logger.debug(
                f"Y轴智能参数: 速度={y_max_speed}Hz/s, "
                f"加速度={y_acceleration}Hz/s², 超时={y_timeout_seconds:.1f}s"
            )

            # 调用并发相对运动方法
            return self._move_relative_2D(
                x_distance_mm=x_distance_mm,
                x_max_speed=x_max_speed,
                x_acceleration=x_acceleration,
                x_deceleration=x_deceleration,
                x_timeout_seconds=x_timeout_seconds,
                y_distance_mm=y_distance_mm,
                y_max_speed=y_max_speed,
                y_acceleration=y_acceleration,
                y_deceleration=y_deceleration,
                y_timeout_seconds=y_timeout_seconds,
            )

        except Exception as e:
            logger.error(f"2D并发空间移动过程中发生异常: {e}", exc_info=True)
            return False

    def get_current_position_2D(self) -> tuple[float, float]:
        """获取当前X和Y轴的绝对位置（毫米）

        Returns:
            tuple[float, float]: (X轴位置, Y轴位置)，单位为毫米
        """
        x_pos = self.get_current_position_1D(0)  # X轴
        y_pos = self.get_current_position_1D(1)  # Y轴

        # 更新当前位置
        self._current_position = [
            x_pos,
            y_pos,
        ]

        return (x_pos, y_pos)

    def cleanup(self):
        """清理资源"""
        self._disconnect()
        if self._is_initialized:
            try:
                self._api.MT_DeInit()
                self._is_initialized = False
                logger.info("步进电机资源已释放")
            except Exception as e:
                logger.error(f"清理资源时发生异常: {e}", exc_info=True)

    def _disconnect(self):
        """断开连接"""
        if self._is_connected:
            try:
                self._api.MT_Close_USB()
                self._api.MT_Close_UART()
                self._api.MT_Close_Net()
                self._is_connected = False
                logger.debug("已断开控制器连接")
            except Exception as e:
                logger.error(f"断开连接时发生异常: {e}", exc_info=True)

    def __del__(self):
        """析构函数，确保资源被正确释放"""
        self.cleanup()
