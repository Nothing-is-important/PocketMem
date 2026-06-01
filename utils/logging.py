"""统一日志系统。

用法：
    from utils import get_logger
    logger = get_logger(__name__)
    logger.info("message")
    logger.warning("message")
    logger.error("message", exc_info=True)
"""

import logging
import sys
from pathlib import Path

_initialized = False


def setup_logging(
    level: int = logging.INFO,
    log_file: str = None,
) -> None:
    """全局日志初始化（在应用入口调用一次）。"""
    global _initialized
    if _initialized:
        return

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # 控制台输出
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)

    root = logging.getLogger("pocketmemory")
    root.setLevel(level)
    root.addHandler(console)

    # 文件输出（可选）
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """获取以 'pocketmemory' 为根的层级 logger。"""
    return logging.getLogger(f"pocketmemory.{name}")
