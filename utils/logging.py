"""统一日志系统。

控制台：WARNING 及以上 + 关键 INFO（启动/完成/错误）
日志文件：所有 DEBUG 及以上（每次启动覆盖 logs/server.log）

用法：
    from utils import get_logger, log_file
    logger = get_logger(__name__)
    log_file("这条消息只写文件，不显示在控制台")
"""

import logging
import os
import sys
from pathlib import Path

_initialized = False
LOG_FILE = "logs/server.log"


class _ConsoleFilter(logging.Filter):
    """控制台只显示 WARNING+ 和关键 INFO。"""
    KEY_INFOS = {"setup", "startup", "complete", "索引完成", "启动服务"}

    def filter(self, record):
        if record.levelno >= logging.WARNING:
            return True
        if record.levelno == logging.INFO:
            msg = record.getMessage().lower()
            return any(kw in msg for kw in self.KEY_INFOS)
        return False


def setup_logging(level: int = logging.DEBUG) -> None:
    """全局日志初始化。"""
    global _initialized
    if _initialized:
        return

    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 文件处理器：DEBUG 级别，记录所有
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    # 控制台处理器：WARNING+，仅关键信息
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.addFilter(_ConsoleFilter())
    console.setFormatter(fmt)

    root = logging.getLogger("teammind")
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(file_handler)

    # print() 重定向到文件（不刷屏）
    sys.stdout = _FileEchoWriter(sys.stdout, LOG_FILE)
    sys.stderr = _FileEchoWriter(sys.stderr, LOG_FILE)

    _initialized = True


class _FileEchoWriter:
    """print() → 文件 + 控制台（精简）。"""
    def __init__(self, original, log_path):
        self._orig = original
        self._log = log_path
        self._line_buf = ""

    def write(self, s):
        # 写入日志文件
        try:
            with open(self._log, "a", encoding="utf-8") as f:
                f.write(s)
        except Exception:
            pass
        # 控制台：只输出包含关键信息的行
        self._line_buf += s
        while "\n" in self._line_buf:
            line, self._line_buf = self._line_buf.split("\n", 1)
            stripped = line.strip()
            if stripped and _is_console_worthy(stripped):
                self._orig.write(line + "\n")

    def flush(self):
        if self._line_buf.strip():
            stripped = self._line_buf.strip()
            if _is_console_worthy(stripped):
                self._orig.write(self._line_buf + "\n")
            self._line_buf = ""
        self._orig.flush()

    def isatty(self):
        return True  # 保持 ANSI 颜色支持


def _is_console_worthy(line: str) -> bool:
    """判断一行 print 输出是否值得显示在控制台。"""
    # 进度条（含 | 和 %）→ 不显示
    if "|" in line and "%" in line:
        return False
    if "it/s" in line or "Loading weights" in line:
        return False
    # 关键信息 → 显示
    keywords = ["ERROR", "Traceback", "索引完成", "启动服务", "生成演示数据",
                "初始化", "设备:", "Embedding", "LLM", "[OK]", "[DEBUG", "✗"]
    return any(kw in line for kw in keywords)


def log_file(msg: str):
    """消息只写入日志文件，不显示在控制台。"""
    logging.getLogger("teammind").info(msg)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"teammind.{name}")
