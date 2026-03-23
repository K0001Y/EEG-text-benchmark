"""通用工具模块。

提供日志记录等通用功能。
"""

from .logging_utils import setup_logging, get_logger

__all__ = [
    'setup_logging',
    'get_logger',
]
