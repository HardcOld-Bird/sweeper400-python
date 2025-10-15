"""
# sweeper400 日志管理系统

模块路径：`sweeper400.logger`

提供统一的日志管理功能，支持：
1. 仅Terminal输出，无本地文件保存
2. 清晰的模块/类/方法来源标识
3. 可调整的日志级别和可见性
4. 层级化的日志管理
5. 智能简要模式：根据调用层级自动过滤INFO日志

## 日志级别（从低到高）：
- DEBUG (10): 详细的调试信息
- INFO (20): 一般信息
- CRIT (25): 重要信息（介于INFO和WARN之间）
- WARN (30): 警告信息
- ERROR (40): 错误信息

## 便捷模式：
- set_debug_mode(): 显示所有日志
- set_verbose_mode(): 详细模式，显示所有INFO及以上级别（默认模式）
- set_brief_mode(): 简要模式，只显示顶层INFO + 所有CRIT/WARN/ERROR
- set_quiet_mode(): 安静模式，只显示WARN及以上

## 使用示例:
    ### 在模块中使用：
    ```python
    from sweeper400.logger import get_logger

    logger = get_logger(__name__)
    logger.info("这是一条信息日志")
    logger.crit("这是一条重要信息")
    logger.error("这是一条错误信息", exc_info=True)
    ```

    ### 在类中使用：
    ```python
    class MyClass:
        def __init__(self):
            self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

        def my_method(self):
            method_logger = get_logger(
                f"{__name__}.{self.__class__.__name__}.my_method"
            )
            method_logger.debug("方法执行开始")
    ```

    ### 设置简要模式：
    ```python
    from sweeper400.logger import set_brief_mode
    set_brief_mode()  # 只显示顶层模块的INFO日志
    ```
"""

import datetime
import logging
import sys
from enum import Enum
from typing import Any

# 添加自定义CRIT级别到logging模块
logging.addLevelName(25, "CRIT")
# 重新定义标准级别的显示名称为更短的版本
logging.addLevelName(30, "WARN")  # WARNING -> WARN


class LogLevel(Enum):
    """日志级别枚举"""

    DEBUG = logging.DEBUG  # 10 - 详细的调试信息
    INFO = logging.INFO  # 20 - 一般信息
    CRIT = 25  # 25 - 重要信息（介于INFO和WARNING之间）
    WARN = logging.WARNING  # 30 - 警告信息
    ERROR = logging.ERROR  # 40 - 错误信息


class SweeperFormatter(logging.Formatter):
    """自定义日志格式化器"""

    # 不同级别的颜色代码
    COLORS = {
        "DEBUG": "\033[36m",  # 青色
        "INFO": "\033[32m",  # 绿色
        "CRIT": "\033[38;5;154m",  # 黄绿色（256色模式，更明显的过渡）
        "WARN": "\033[33m",  # 黄色
        "ERROR": "\033[31m",  # 红色
        "RESET": "\033[0m",  # 重置颜色
    }

    def format(self, record: logging.LogRecord) -> str:
        try:
            # 获取颜色
            color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
            reset = self.COLORS["RESET"]

            # 格式化时间戳（Windows兼容）
            dt = datetime.datetime.fromtimestamp(record.created)
            timestamp = dt.strftime("%H:%M:%S.%f")[:-3]  # 精确到毫秒

            # 简化模块名显示（只显示最后两级）
            name_parts = record.name.split(".")
            if len(name_parts) > 2:
                display_name = "..." + ".".join(name_parts[-2:])
            else:
                display_name = record.name

            # 构建日志消息
            formatted_message = (
                f"{color}[{timestamp}] "
                f"{record.levelname:5s} "
                f"{display_name:24s} | "
                f"{record.getMessage()}{reset}"
            )

            # 如果有异常信息，添加异常堆栈
            if record.exc_info:
                formatted_message += "\n" + self.formatException(record.exc_info)

            return formatted_message
        except Exception:
            # 如果格式化失败，返回基本格式
            return f"[LOG FORMAT ERROR] {record.levelname}: {record.getMessage()}"


class SmartLevelFilter(logging.Filter):
    """智能层级过滤器

    在简要模式下，只允许顶层模块的INFO级别日志通过，
    同时允许所有CRIT及以上级别的日志通过。
    """

    def __init__(self, manager: "SweeperLoggerManager"):
        super().__init__()
        self._manager = manager

    def filter(self, record: logging.LogRecord) -> bool:
        """过滤日志记录"""
        # 如果不是简要模式，允许所有日志通过
        if not self._manager._brief_mode:  # type: ignore
            return True

        # 如果是简要模式
        # 如果是CRIT及以上级别，总是允许通过
        if record.levelno >= 25:  # CRIT及以上
            return True

        # 如果是INFO级别，需要检查是否为顶层模块
        if record.levelno == logging.INFO:
            return self._is_top_level_module(record.name)

        # DEBUG级别在简要模式下不显示
        return record.levelno > logging.DEBUG

    def _is_top_level_module(self, logger_name: str) -> bool:
        """判断是否为顶层模块

        顶层模块的判断逻辑：
        1. 检查是否为sweeper400的直接子包（如sweeper400.use, sweeper400.move等）
        2. 如果有更高层级的模块正在使用，则隐藏子模块的INFO日志
        """
        # 获取所有当前活跃的日志器名称
        active_loggers = list(self._manager._loggers.keys())  # type: ignore

        # 分析当前logger的层级
        logger_parts = logger_name.split(".")

        # 如果不是sweeper400包的模块，总是显示
        if not logger_name.startswith("sweeper400."):
            return True

        # 找出所有sweeper400包内的活跃logger，按层级深度分组
        sweeper_loggers = [
            name for name in active_loggers if name.startswith("sweeper400.")
        ]

        # 找出最短的路径长度（最顶层的调用）
        min_depth = (
            min(len(name.split(".")) for name in sweeper_loggers)
            if sweeper_loggers
            else len(logger_parts)
        )

        # 如果当前logger的层级深度等于最小深度，则认为是顶层
        current_depth = len(logger_parts)

        # 特殊处理：如果有use包的logger活跃，则只显示use包的日志
        use_loggers = [
            name for name in sweeper_loggers if name.startswith("sweeper400.use.")
        ]
        if use_loggers and not logger_name.startswith("sweeper400.use."):
            return False

        # 否则，只显示最浅层级的日志
        return current_depth <= min_depth


class SweeperLoggerManager:
    """sweeper400 日志管理器"""

    def __init__(self):
        self._loggers: dict[str, logging.Logger] = {}
        self._global_level = LogLevel.INFO  # 默认为详细模式的INFO级别
        self._module_levels: dict[str, LogLevel] = {}
        self._handler: logging.StreamHandler[Any] | None = None
        self._brief_mode = False  # 简要模式标志，默认为详细模式
        self._smart_filter: SmartLevelFilter | None = None
        self._setup_handler()

    def _setup_handler(self):
        """设置日志处理器"""
        if self._handler is None:
            self._handler = logging.StreamHandler(sys.stdout)
            self._handler.setFormatter(SweeperFormatter())
            # 创建并添加智能过滤器
            self._smart_filter = SmartLevelFilter(self)
            self._handler.addFilter(self._smart_filter)

    def get_logger(self, name: str) -> logging.Logger:
        """
        获取指定名称的日志器

        Args:
            name: 日志器名称，通常使用 __name__ 或 模块.类.方法 格式

        Returns:
            配置好的日志器实例
        """
        if name not in self._loggers:
            logger = logging.getLogger(name)
            logger.handlers.clear()  # 清除默认处理器
            if self._handler is not None:
                logger.addHandler(self._handler)
            logger.propagate = False  # 防止重复输出

            # 设置日志级别
            effective_level = self._get_effective_level(name)
            logger.setLevel(effective_level.value)

            # 为logger添加crit方法
            def crit(message: Any, *args: Any, **kwargs: Any) -> None:
                if logger.isEnabledFor(25):
                    logger._log(25, message, args, **kwargs)

            # 使用setattr避免类型检查问题
            logger.crit = crit  # type: ignore[attr-defined]

            self._loggers[name] = logger

        return self._loggers[name]

    def _get_effective_level(self, name: str) -> LogLevel:
        """获取有效的日志级别"""
        # 检查是否有特定模块的级别设置
        name_parts = name.split(".")
        for i in range(len(name_parts), 0, -1):
            partial_name = ".".join(name_parts[:i])
            if partial_name in self._module_levels:
                return self._module_levels[partial_name]

        # 返回全局级别
        return self._global_level

    def set_global_level(self, level: LogLevel):
        """
        设置全局日志级别

        Args:
            level: 日志级别
        """
        self._global_level = level
        # 更新所有已存在的日志器
        for name, logger in self._loggers.items():
            if name not in self._module_levels:
                logger.setLevel(level.value)

    def set_module_level(self, module_name: str, level: LogLevel):
        """
        设置特定模块的日志级别

        Args:
            module_name: 模块名称
            level: 日志级别
        """
        self._module_levels[module_name] = level

        # 更新匹配的日志器
        for name, logger in self._loggers.items():
            if name.startswith(module_name):
                logger.setLevel(level.value)

    def get_current_levels(self) -> dict[str, Any]:
        """获取当前的日志级别配置"""
        result = {
            "global_level": self._global_level.name,
            "module_levels": {k: v.name for k, v in self._module_levels.items()},
        }
        return result

    def reset_levels(self):
        """重置所有日志级别为默认值"""
        self._global_level = LogLevel.INFO
        self._module_levels.clear()

        # 更新所有日志器
        for logger in self._loggers.values():
            logger.setLevel(LogLevel.INFO.value)

    def set_brief_mode(self, enabled: bool):
        """设置简要模式

        Args:
            enabled: 是否启用简要模式
        """
        self._brief_mode = enabled

    def is_brief_mode(self) -> bool:
        """检查是否为简要模式"""
        return self._brief_mode


# 全局日志管理器实例
_logger_manager = SweeperLoggerManager()


def get_logger(name: str) -> logging.Logger:
    """
    获取日志器的便捷函数

    Args:
        name: 日志器名称，建议使用 __name__

    Returns:
        配置好的日志器实例

    Example:
        logger = get_logger(__name__)
        logger.info("这是一条信息")
    """
    return _logger_manager.get_logger(name)


# 日志级别控制函数
def set_global_log_level(level: LogLevel):
    """设置全局日志级别"""
    _logger_manager.set_global_level(level)


def set_module_log_level(module_name: str, level: LogLevel):
    """设置特定模块的日志级别"""
    _logger_manager.set_module_level(module_name, level)


def get_log_levels() -> dict[str, Any]:
    """获取当前日志级别配置"""
    return _logger_manager.get_current_levels()


def reset_log_levels():
    """重置所有日志级别"""
    _logger_manager.reset_levels()


# 便捷的日志级别调整函数
def set_debug_mode():
    """开启调试模式（显示所有日志）"""
    set_global_log_level(LogLevel.DEBUG)
    _logger_manager.set_brief_mode(False)


def set_verbose_mode():
    """详细模式（显示所有INFO及以上级别，默认模式）"""
    set_global_log_level(LogLevel.INFO)
    _logger_manager.set_brief_mode(False)


def set_brief_mode():
    """简要模式（只显示顶层INFO + 所有IMPORTANT/WARNING/ERROR）"""
    set_global_log_level(LogLevel.INFO)
    _logger_manager.set_brief_mode(True)


def set_quiet_mode():
    """安静模式（只显示警告和错误）"""
    set_global_log_level(LogLevel.WARN)
    _logger_manager.set_brief_mode(False)
