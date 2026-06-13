"""统一日志系统。

每次启动自动保存日志到 logs/server.log（仅保留最后一次）。

用法：
    from utils import get_logger
    logger = get_logger(__name__)
"""

import logging
import os
import sys
from pathlib import Path

_initialized = False
LOG_FILE = "logs/server.log"


def setup_logging(level: int = logging.INFO) -> None:
    """全局日志初始化——控制台 + 文件双输出。"""
    global _initialized
    if _initialized:
        return

    # 确保日志目录存在，清空上次日志
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)

    # 文件
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(fmt)

    # 根 logger
    root = logging.getLogger("teammind")
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(file_handler)

    # 同时捕获 print 输出到文件
    _setup_stdout_capture()

    _initialized = True


class _TeeWriter:
    """同时写入原始 stdout 和日志文件。"""
    def __init__(self, original, log_path):
        self.original = original
        self.log_path = log_path

    def write(self, s):
        self.original.write(s)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(s)
        except Exception:
            pass

    def flush(self):
        self.original.flush()

    def isatty(self):
        return self.original.isatty()


def _setup_stdout_capture():
    """捕获 print() 输出到日志文件。"""
    sys.stdout = _TeeWriter(sys.stdout, LOG_FILE)
    sys.stderr = _TeeWriter(sys.stderr, LOG_FILE)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"teammind.{name}")
