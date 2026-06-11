"""微信数据库密钥提取器。

从微信进程内存中提取 SQLCipher 解密密钥。
支持 WeChat 3.x (WeChatWin.dll + iphone marker) 和 WeChat 4.x (x'<64hex><32hex>' 全进程扫描)。

需要：微信已登录、pymem、管理员权限（Windows，全内存扫描需要）。

参考：
- opencode/skills/wechat-db-decrypt.skill.md
- github.com/ylytdeng/wechat-decrypt (WeChat 4.x reference)
- github.com/L1en2407/wechat-decrypt
"""

import ctypes
import os
import re
import sys
from ctypes import wintypes
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Windows API constants ──────────────────────────────────────
MEM_COMMIT = 0x1000
MEM_READABLE_PROTECTS = {0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80}
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010
PROCESS_VM_OPERATION = 0x0008


class MEMORY_BASIC_INFORMATION(ctypes.Structure):
    """Windows MEMORY_BASIC_INFORMATION.

    Windows 10+ includes a PartitionId field (WORD, 2 bytes)
    between AllocationProtect and RegionSize.
    Using c_ulong (4 bytes) accounts for PartitionId + padding,
    keeping RegionSize aligned at offset 24.
    """
    _fields_ = [
        ("BaseAddress", ctypes.c_ulonglong),
        ("AllocationBase", ctypes.c_ulonglong),
        ("AllocationProtect", wintypes.DWORD),
        ("PartitionId", wintypes.DWORD),   # Windows 10+: WORD + alignment padding
        ("RegionSize", ctypes.c_ulonglong),
        ("State", wintypes.DWORD),
        ("Protect", wintypes.DWORD),
        ("Type", wintypes.DWORD),
    ]


# ── Process helpers ────────────────────────────────────────────

def _get_wechat_pids() -> List[Tuple[int, str]]:
    """Find all running WeChat processes via tasklist.
    
    Returns:
        List of (pid, process_name) tuples sorted by pid.
    """
    import subprocess
    pids = []
    for proc_name in ("Weixin.exe", "WeChat.exe"):
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"IMAGENAME eq {proc_name}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.strip('"').split('","')
                if len(parts) >= 2:
                    try:
                        pid = int(parts[1])
                        pids.append((pid, proc_name))
                    except ValueError:
                        continue
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    return pids


def _open_process(pid: int) -> Optional[int]:
    """Open a process handle with PROCESS_VM_READ access."""
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(
        PROCESS_QUERY_INFORMATION | PROCESS_VM_READ | PROCESS_VM_OPERATION,
        False, pid,
    )
    return handle if handle else None


def _close_handle(handle: int) -> None:
    """Safely close a Windows handle."""
    if handle:
        ctypes.windll.kernel32.CloseHandle(handle)


# ── Memory region enumeration ──────────────────────────────────

def _enum_readable_regions(handle: int, max_region_size: int = 500 * 1024 * 1024) -> List[Tuple[int, int]]:
    """Enumerate all committed, readable memory regions in a process.

    Uses VirtualQueryEx to walk the process address space.

    Args:
        handle: Open process handle with PROCESS_QUERY_INFORMATION.
        max_region_size: Skip regions larger than this (safety limit).

    Returns:
        List of (base_address, region_size) tuples.
    """
    kernel32 = ctypes.windll.kernel32
    regions = []
    addr = 0
    max_addr = 0x7FFFFFFFFFFF  # 64-bit user space limit

    while addr < max_addr:
        mbi = MEMORY_BASIC_INFORMATION()
        result = kernel32.VirtualQueryEx(
            ctypes.c_uint64(handle),
            ctypes.c_uint64(addr),
            ctypes.byref(mbi),
            ctypes.sizeof(mbi),
        )
        if result == 0:
            break

        if (mbi.State == MEM_COMMIT
                and mbi.Protect in MEM_READABLE_PROTECTS
                and 0 < mbi.RegionSize <= max_region_size):
            regions.append((mbi.BaseAddress, mbi.RegionSize))

        next_addr = mbi.BaseAddress + mbi.RegionSize
        if next_addr <= addr:
            break
        addr = next_addr

    return regions


def _read_process_memory_block(handle: int, base: int, size: int) -> Optional[bytes]:
    """Read a block of process memory via ReadProcessMemory.

    Args:
        handle: Open process handle with PROCESS_VM_READ.
        base: Starting address.
        size: Number of bytes to read.

    Returns:
        Raw bytes or None if read failed.
    """
    kernel32 = ctypes.windll.kernel32
    buf = ctypes.create_string_buffer(size)
    bytes_read = ctypes.c_size_t(0)

    if kernel32.ReadProcessMemory(
        ctypes.c_uint64(handle),
        ctypes.c_uint64(base),
        buf,
        ctypes.c_size_t(size),
        ctypes.byref(bytes_read),
    ):
        return buf.raw[:bytes_read.value]
    return None


# ── Key extraction: Full process scan (WeChat 4.x) ─────────────

# WeChat 4.x stores raw keys as: x'<64hex_enc_key><32hex_salt>'
# The enc_key is the post-PBKDF2 derived key, the salt is the DB salt.
WECHAT4_KEY_PATTERN = re.compile(rb"x'([0-9a-fA-F]{64})([0-9a-fA-F]{32})'")
# Generic 64-char hex fallback
GENERIC_HEX64 = re.compile(rb"[0-9a-fA-F]{64}")


def _scan_full_process_memory() -> Optional[str]:
    """Scan ALL WeChat process memory regions for raw encryption keys.

    For WeChat 4.x: searches for x'<64hex_key><32hex_salt>' pattern.
    Falls back to generic 64-char hex scanning.

    Requires admin privileges (PROCESS_VM_READ).

    Returns:
        64-char hex key string, or None if not found.
    """
    if sys.platform != "win32":
        return None

    pids = _get_wechat_pids()
    if not pids:
        return None

    # Sort by descending PID (main process usually launched first)
    for pid, proc_name in pids:
        handle = _open_process(pid)
        if not handle:
            continue

        try:
            regions = _enum_readable_regions(handle)
            if not regions:
                continue

            total_mb = sum(s for _, s in regions) / (1024 * 1024)
            # Scan regions, prioritizing large ones (more likely to contain key)
            regions_sorted = sorted(regions, key=lambda x: x[1], reverse=True)

            for base, size in regions_sorted:
                data = _read_process_memory_block(handle, base, size)
                if not data:
                    continue

                # Try WeChat 4.x x' pattern first
                for match in WECHAT4_KEY_PATTERN.finditer(data):
                    key_hex = match.group(1).decode("ascii").lower()
                    if key_hex != "0" * 64 and key_hex != "f" * 64:
                        return key_hex

                # Fallback: generic 64-char hex
                for match in GENERIC_HEX64.finditer(data):
                    key_hex = match.group().decode("ascii").lower()
                    if key_hex != "0" * 64 and key_hex != "f" * 64:
                        # Avoid false positives from repetitive patterns
                        unique_chars = len(set(key_hex))
                        if unique_chars >= 8:  # Real keys have decent entropy
                            return key_hex
        finally:
            _close_handle(handle)

    return None


def _scan_all_keys_full_process() -> Dict[str, str]:
    """Scan full process memory and return ALL key-salt pairs found.

    Returns:
        Dictionary mapping salt_hex → enc_key_hex.
        Multiple databases can share the same salt.
    """
    if sys.platform != "win32":
        return {}

    result = {}
    pids = _get_wechat_pids()

    for pid, proc_name in pids:
        handle = _open_process(pid)
        if not handle:
            continue

        try:
            regions = _enum_readable_regions(handle)
            regions_sorted = sorted(regions, key=lambda x: x[1], reverse=True)

            for base, size in regions_sorted:
                data = _read_process_memory_block(handle, base, size)
                if not data:
                    continue

                for match in WECHAT4_KEY_PATTERN.finditer(data):
                    key_hex = match.group(1).decode("ascii").lower()
                    salt_hex = match.group(2).decode("ascii").lower()
                    if key_hex != "0" * 64 and key_hex != "f" * 64:
                        result[salt_hex] = key_hex
        finally:
            _close_handle(handle)

    return result


# ── Key extraction: Module-based scan (pymem, WeChat 3.x compat) ──

def _scan_module_with_pymem(target_modules: List[str]) -> Optional[str]:
    """Use pymem to scan specific DLL modules for hex keys.

    Searches for both generic 64-char hex and WeChat 4.x x' pattern.
    Works as a fallback when full process scan is unavailable.

    Args:
        target_modules: List of DLL names to search (e.g. ["WeChatWin.dll", "Weixin.dll"])

    Returns:
        64-char hex key string, or None if not found.
    """
    try:
        import pymem
        import pymem.process
    except ImportError:
        return None

    # Find WeChat process
    pm = None
    for proc_name in ("Weixin.exe", "WeChat.exe"):
        try:
            pm = pymem.Pymem(proc_name)
            break
        except pymem.exception.ProcessNotFound:
            continue

    if pm is None:
        return None

    # Try each target module
    for module_name in target_modules:
        try:
            mod = pymem.process.module_from_name(pm.process_handle, module_name)
        except pymem.exception.ModuleNotFoundError:
            continue

        if not mod:
            continue

        module_base = mod.lpBaseOfDll
        module_size = mod.SizeOfImage
        chunk_size = 0x100000  # 1MB chunks

        # First pass: WeChat 4.x x' pattern (targeted)
        for offset in range(0, module_size, chunk_size):
            try:
                chunk = pm.read_bytes(
                    module_base + offset,
                    min(chunk_size, module_size - offset),
                )
            except pymem.exception.MemoryReadError:
                continue

            for match in WECHAT4_KEY_PATTERN.finditer(chunk):
                key_hex = match.group(1).decode("ascii").lower()
                if key_hex != "0" * 64 and key_hex != "f" * 64:
                    return key_hex

            for match in GENERIC_HEX64.finditer(chunk):
                key_hex = match.group().decode("ascii").lower()
                if key_hex != "0" * 64 and key_hex != "f" * 64:
                    unique_chars = len(set(key_hex))
                    if unique_chars >= 8:
                        return key_hex

        # If x' pattern found nothing, try iphone marker (WeChat 3.x compat)
        phone_marker = b"iphone\x00"
        KEY_OFFSETS = [0x70, 0x68, 0x78, 0x60, 0x80, 0x50, 0x90, 0xA0, 0x40, 0xB0]

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


# ── Key extraction: Config file fallback ───────────────────────

def extract_key_from_config() -> Optional[str]:
    """从微信配置文件尝试提取密钥（降级方案）。

    搜索已知的配置文件和目录。
    """
    home = Path.home()
    hex_pattern = re.compile(r"[0-9a-fA-F]{64}")

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
            try:
                with open(config_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(config_path, "r", encoding="latin-1", errors="replace") as f:
                    content = f.read()

            matches = hex_pattern.findall(content)
            for match in matches:
                key = match.lower()
                if key != "0" * 64 and key != "f" * 64:
                    return key
        except (OSError, UnicodeDecodeError):
            continue

    return None


# ── Main API ───────────────────────────────────────────────────

def get_wechat_key() -> Optional[str]:
    """获取微信数据库解密密钥（尝试多种方法，按成功率排序）。

    方法链：
    1. 全进程内存扫描（VirtualQueryEx + ReadProcessMemory）—— WeChat 4.x 主方案
    2. pymem 模块扫描（Weixin.dll + WeChatWin.dll）—— 备用方案
    3. UAC 提权辅助进程
    4. 配置文件查找

    Returns:
        64 字符 hex 密钥字符串，失败返回 None
    """
    # 方法 1：全进程内存扫描（WeChat 4.x - 需要管理员权限）
    key = _scan_full_process_memory()
    if key:
        return key

    # 方法 2：pymem 模块扫描（支持 WeChat 3.x + 4.x）
    # 按优先级尝试：Weixin.dll（4.x）、WeChatWin.dll（3.x）
    target_modules = ["Weixin.dll", "WeChatWin.dll"]
    key = _scan_module_with_pymem(target_modules)
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


def get_all_wechat_keys() -> Dict[str, str]:
    """获取所有微信数据库密钥（salt → key 映射）。

    扫描全进程内存，提取所有 x'<64hex_key><32hex_salt>' 密钥对。

    Returns:
        Dict[str, str]: salt_hex → enc_key_hex
    """
    keys = _scan_all_keys_full_process()
    if keys:
        return keys

    # Fallback: try pymem-based scan with the same pattern
    try:
        import pymem
        import pymem.process
        import pymem.exception as pm_exc

        pm = None
        for proc_name in ("Weixin.exe", "WeChat.exe"):
            try:
                pm = pymem.Pymem(proc_name)
                break
            except pm_exc.ProcessNotFound:
                continue

        if pm is None:
            return {}

        mod_names = ["Weixin.dll", "WeChatWin.dll"]
        for mod_name in mod_names:
            try:
                mod = pymem.process.module_from_name(pm.process_handle, mod_name)
            except pm_exc.ModuleNotFoundError:
                continue

            if not mod:
                continue

            chunk_size = 0x100000
            for offset in range(0, mod.SizeOfImage, chunk_size):
                try:
                    chunk = pm.read_bytes(
                        mod.lpBaseOfDll + offset,
                        min(chunk_size, mod.SizeOfImage - offset),
                    )
                except pm_exc.MemoryReadError:
                    continue

                for match in WECHAT4_KEY_PATTERN.finditer(chunk):
                    key_hex = match.group(1).decode("ascii").lower()
                    salt_hex = match.group(2).decode("ascii").lower()
                    if key_hex != "0" * 64 and key_hex != "f" * 64:
                        keys[salt_hex] = key_hex
    except ImportError:
        pass

    return keys


def _is_admin() -> bool:
    """检查是否以管理员权限运行（Windows）。"""
    if sys.platform != "win32":
        return True
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        return False
