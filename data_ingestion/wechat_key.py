"""微信数据库密钥提取器。

从微信进程内存中提取 SQLCipher 解密密钥。
需要：微信已登录、pymem、管理员权限（Windows）。

参考：opencode/skills/wechat-db-decrypt.skill.md
"""

import os
import sys
from pathlib import Path
from typing import Optional


def extract_key_from_memory() -> Optional[str]:
    """从 Weixin.exe 进程内存提取 SQLCipher 密钥。

    原理：密钥以 x'<64 hex chars><32 hex chars>' 格式存储在
    WeChatWin.dll 内存中，位于 "iphone\x00" 标记之前 ~0x70 字节处。

    Returns:
        64 字符 hex 密钥字符串，失败返回 None
    """
    # Windows 平台检查
    if sys.platform != "win32":
        return None

    # 检查管理员权限
    if not _is_admin():
        return None

    try:
        import pymem
        import pymem.process
    except ImportError:
        return None

    try:
        pm = pymem.Pymem("Weixin.exe")
    except pymem.exception.ProcessNotFound:
        try:
            pm = pymem.Pymem("WeChat.exe")
        except pymem.exception.ProcessNotFound:
            return None

    try:
        wechat_module = pymem.process.module_from_name(
            pm.process_handle, "WeChatWin.dll"
        )
    except pymem.exception.ModuleNotFoundError:
        return None

    if not wechat_module:
        return None

    module_base = wechat_module.lpBaseOfDll
    module_size = wechat_module.SizeOfImage
    chunk_size = 0x100000  # 1MB
    phone_marker = b"iphone\x00"

    for offset in range(0, module_size, chunk_size):
        try:
            chunk = pm.read_bytes(
                module_base + offset,
                min(chunk_size, module_size - offset),
            )
        except pymem.exception.MemoryReadError:
            continue

        idx = 0
        while True:
            idx = chunk.find(phone_marker, idx)
            if idx == -1:
                break

            # 密钥在 "iphone\x00" 前约 0x70 字节
            key_offset = idx - 0x70
            if key_offset >= 0:
                key_bytes = chunk[key_offset:key_offset + 64]
                # 验证：64 字节非零 hex 字符串
                if len(key_bytes) == 64 and key_bytes != b"\x00" * 64:
                    try:
                        key_hex = key_bytes.decode("ascii")
                        if all(c in "0123456789abcdefABCDEF" for c in key_hex):
                            return key_hex.lower()
                    except UnicodeDecodeError:
                        pass
            idx += 1

    return None


def extract_key_from_config() -> Optional[str]:
    """从微信配置文件尝试提取密钥（降级方案）。

    某些微信版本在配置文件中存储密钥派生数据。
    """
    home = Path.home()
    config_paths = [
        home / "Documents" / "WeChat Files" / "All Users" / "config",
        home / "Documents" / "xwechat_files" / "config",
    ]

    import re
    hex_pattern = re.compile(r"[0-9a-fA-F]{64}")

    for config_path in config_paths:
        if not config_path.exists():
            continue
        try:
            with open(config_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            matches = hex_pattern.findall(content)
            if matches:
                return matches[0].lower()
        except (OSError, UnicodeDecodeError):
            continue

    return None


def get_wechat_key() -> Optional[str]:
    """获取微信数据库解密密钥（尝试多种方法）。

    Returns:
        64 字符 hex 密钥，失败返回 None
    """
    # 方法 1：内存扫描（最可靠，需要管理员权限 + 微信登录）
    key = extract_key_from_memory()
    if key:
        return key

    # 方法 2：配置文件查找（降级方案）
    key = extract_key_from_config()
    if key:
        return key

    return None


def _is_admin() -> bool:
    """检查是否以管理员权限运行（Windows）。"""
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        return False
