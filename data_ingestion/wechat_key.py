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

    使用 psutil 获取 WeChatWin.dll 基址（无需管理员），
    然后用 pymem 或 ctypes 读取进程内存。

    Returns:
        64 字符 hex 密钥字符串，失败返回 None
    """
    if sys.platform != "win32":
        return None

    # 获取 WeChatWin.dll 基址
    module_base, module_size = _get_wechatwin_base()
    if not module_base:
        return None

    # 尝试读取进程内存
    chunk = _read_process_memory(module_base, module_size)
    if not chunk:
        return None

    # 搜索 iphone 标记
    phone_marker = b"iphone\x00"
    KEY_OFFSETS = [0x70, 0x68, 0x78, 0x60, 0x80, 0x50, 0x90, 0xA0, 0x40, 0xB0]

    idx = 0
    while True:
        idx = chunk.find(phone_marker, idx)
        if idx == -1:
            break

        for key_offset in KEY_OFFSETS:
            pos = idx - key_offset
            if pos >= 0:
                key_bytes = chunk[pos:pos + 64]
                if len(key_bytes) == 64 and key_bytes != b"\x00" * 64:
                    try:
                        key_hex = key_bytes.decode("ascii")
                        if all(c in "0123456789abcdefABCDEF" for c in key_hex):
                            return key_hex.lower()
                    except UnicodeDecodeError:
                        pass
        idx += 1

    return None


def _get_wechatwin_base() -> tuple:
    """获取 WeChatWin.dll 的基址和大小。

    使用 psapi.EnumProcessModules（不需要管理员）。

    Returns:
        (base_address, size) 或 (None, None)
    """
    import ctypes
    from ctypes import wintypes

    psapi = ctypes.windll.psapi
    kernel32 = ctypes.windll.kernel32

    # 找微信进程
    process_names = ["Weixin.exe", "WeChat.exe"]
    target_pid = None
    for name in process_names:
        try:
            import pymem
            pm = pymem.Pymem(name)
            target_pid = pm.process_id
            break
        except Exception:
            continue

    if not target_pid:
        return None, None

    # OpenProcess with minimal rights
    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010
    handle = kernel32.OpenProcess(
        PROCESS_QUERY_INFORMATION | PROCESS_VM_READ,
        False, target_pid
    )
    if not handle:
        return None, None

    try:
        # Enumerate modules
        hModules = (ctypes.c_void_p * 1024)()
        cbNeeded = wintypes.DWORD()
        if not psapi.EnumProcessModules(handle, ctypes.byref(hModules),
                                         ctypes.sizeof(hModules), ctypes.byref(cbNeeded)):
            return None, None

        num_modules = cbNeeded.value // ctypes.sizeof(ctypes.c_void_p)
        MODULE_NAME_LEN = 260
        name_buffer = ctypes.create_unicode_buffer(MODULE_NAME_LEN)

        for i in range(num_modules):
            mod_handle = hModules[i]
            if psapi.GetModuleFileNameExW(handle, mod_handle, name_buffer, MODULE_NAME_LEN):
                mod_name = name_buffer.value.lower()
                if "wechatwin.dll" in mod_name:
                    # Get module info
                    class MODULEINFO(ctypes.Structure):
                        _fields_ = [("lpBaseOfDll", ctypes.c_void_p),
                                    ("SizeOfImage", wintypes.DWORD),
                                    ("EntryPoint", ctypes.c_void_p)]
                    mod_info = MODULEINFO()
                    if psapi.GetModuleInformation(handle, mod_handle,
                                                   ctypes.byref(mod_info),
                                                   ctypes.sizeof(mod_info)):
                        return mod_info.lpBaseOfDll, mod_info.SizeOfImage
        return None, None
    finally:
        kernel32.CloseHandle(handle)


def _read_process_memory(base: int, size: int, max_read: int = 50 * 1024 * 1024) -> bytes:
    """读取进程内存。

    Args:
        base: 起始地址
        size: 模块大小
        max_read: 最大读取字节数（默认 50MB）

    Returns:
        读取到的字节，失败返回 None
    """
    read_size = min(size, max_read)

    # 使用 pymem（已有进程句柄）
    try:
        import pymem
        for name in ["Weixin.exe", "WeChat.exe"]:
            try:
                pm = pymem.Pymem(name)
                return pm.read_bytes(base, read_size)
            except Exception:
                continue
    except ImportError:
        pass

    return None


def extract_key_from_config() -> Optional[str]:
    """从微信配置文件尝试提取密钥（降级方案）。

    搜索已知的配置文件和目录。
    """
    import re
    home = Path.home()
    hex_pattern = re.compile(r"[0-9a-fA-F]{64}")

    # 配置文件路径（文件 + 目录中的 config.data）
    search_paths = [
        home / "Documents" / "WeChat Files" / "All Users" / "config",
        home / "Documents" / "WeChat Files" / "All Users" / "config" / "config.data",
        home / "Documents" / "xwechat_files" / "config",
        home / "Documents" / "xwechat_files" / "config" / "config.data",
        home / "AppData" / "Roaming" / "Tencent" / "WeChat" / "All Users" / "config",
    ]

    for config_path in search_paths:
        if not config_path.exists() or config_path.is_dir():
            continue
        try:
            # 尝试文本模式读取
            try:
                with open(config_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except UnicodeDecodeError:
                # 二进制文件，尝试 latin-1
                with open(config_path, "r", encoding="latin-1", errors="replace") as f:
                    content = f.read()

            matches = hex_pattern.findall(content)
            # 过滤明显不是密钥的模式（全0、全F等）
            for match in matches:
                key = match.lower()
                if key != "0" * 64 and key != "f" * 64:
                    return key
        except (OSError, UnicodeDecodeError):
            continue

    return None


def get_wechat_key() -> Optional[str]:
    """获取微信数据库解密密钥（尝试多种方法）。

    psutil 获取模块地址不需要管理员，
    ctypes ReadProcessMemory 在某些系统上也不需要。
    """
    # 方法 1：psutil + ctypes 读取（不要求管理员）
    key = extract_key_from_memory()
    if key:
        return key

    # 方法 2：通用 hex 密钥模式扫描
    key = _scan_hex_key_in_memory()
    if key:
        return key

    # 方法 3：UAC 提权辅助进程（弹窗确认后以管理员运行）
    try:
        from data_ingestion.wechat_key_elevated import extract_key_elevated
        key = extract_key_elevated()
        if key:
            return key
    except ImportError:
        pass

    # 方法 4：配置文件查找（降级方案）
    key = extract_key_from_config()
    if key:
        return key

    return None


def _scan_hex_key_in_memory() -> Optional[str]:
    """扫描 WeChatWin.dll 内存中的通用 hex 密钥模式。

    不依赖特定标记偏移，直接搜索 64 字符 hex 字符串。
    """
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

    import re
    hex_pattern = re.compile(rb"[0-9a-fA-F]{64}")

    module_base = wechat_module.lpBaseOfDll
    module_size = wechat_module.SizeOfImage
    chunk_size = 0x100000

    for offset in range(0, module_size, chunk_size):
        try:
            chunk = pm.read_bytes(
                module_base + offset,
                min(chunk_size, module_size - offset),
            )
        except pymem.exception.MemoryReadError:
            continue

        for match in hex_pattern.finditer(chunk):
            key = match.group().decode("ascii").lower()
            # 基本的密钥合理性检查：不是全零或全F
            if key != "0" * 64 and key != "f" * 64:
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
