import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logging(log_dir: str, log_name: str = "eval.log", level: int = logging.INFO) -> logging.Logger:
    """Configure root logger with both file and console handlers.

    - Creates log_dir if it does not exist
    - Writes logs to log_dir/log_name with rotation
    - Logs INFO and above by default
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)

    logger = logging.getLogger("benchmark_eval")
    # Avoid adding multiple handlers on repeated setup
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # File handler with rotation
    file_handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    file_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s", "%H:%M:%S")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logging initialized. Log file: %s", log_path)
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a child logger under the benchmark_eval root logger."""
    root = logging.getLogger("benchmark_eval")
    if not root.handlers:
        # Fallback: basicConfig if setup_logging was not called yet
        logging.basicConfig(level=logging.INFO)
        root = logging.getLogger("benchmark_eval")
    return root.getChild(name) if name else root
