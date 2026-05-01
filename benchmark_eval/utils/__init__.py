"""通用工具模块。

提供日志记录、噪声生成、参数解析等通用功能。
"""

from .logging_utils import setup_logging, get_logger
from .noise_utils import generate_gaussian_noise, generate_zero_signal, shuffle_signal, apply_noise
from .arg_utils import add_common_retrieval_args, add_diagnostic_args

__all__ = [
    'setup_logging',
    'get_logger',
    'generate_gaussian_noise',
    'generate_zero_signal',
    'shuffle_signal',
    'apply_noise',
    'add_common_retrieval_args',
    'add_diagnostic_args',
]
