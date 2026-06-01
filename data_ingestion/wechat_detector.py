"""微信状态检测器。

检测微信是否在运行，获取用户信息，为后续自动解密做准备。
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


def get_wechat_status() -> Dict:
    """检测微信运行状态和用户信息。

    Returns:
        {
            "running": True/False,
            "process_name": "Weixin.exe",
            "wxid": "wxid_xxx" 或 None,
            "data_dir": "C:/Users/.../WeChat Files/wxid_xxx" 或 None,
            "msg_dbs": ["MSG0.db", ...] 或 [],
            "can_import": True/False,
            "hint": "提示文字",
        }
    """
    result = {
        "running": False,
        "process_name": "",
        "wxid": None,
        "data_dir": None,
        "msg_dbs": [],
        "can_import": False,
        "hint": "",
    }

    # Step 1: 检测微信进程
    process_name = _find_wechat_process()
    if not process_name:
        result["hint"] = "微信未运行。启动并登录微信后可自动导入聊天记录。"
        return result

    result["running"] = True
    result["process_name"] = process_name

    # Step 2: 查找微信数据目录
    wxid, data_dir = _find_wechat_data_dir()
    if not wxid:
        result["hint"] = "微信正在运行，但未找到数据目录。"
        return result

    result["wxid"] = wxid
    result["data_dir"] = str(data_dir)

    # Step 3: 检查消息数据库
    msg_dir = data_dir / "Msg" / "Multi"
    if not msg_dir.exists():
        msg_dir = data_dir / "Msg"

    if msg_dir.exists():
        msg_dbs = sorted(
            [f.name for f in msg_dir.glob("MSG*.db")],
            key=lambda x: int("".join(c for c in x if c.isdigit()) or "0"),
        )
        result["msg_dbs"] = msg_dbs

    # Step 4: 判断是否可导入
    if result["msg_dbs"]:
        total_size = sum(
            (msg_dir / db).stat().st_size for db in result["msg_dbs"]
        )
        result["can_import"] = True
        db_count = len(result["msg_dbs"])
        size_mb = total_size / (1024 * 1024)
        result["hint"] = (
            f"微信运行中 · {wxid} · 发现 {db_count} 个消息数据库 ({size_mb:.0f}MB)"
        )
    else:
        result["hint"] = f"微信运行中 · {wxid} · 暂未发现消息数据库"

    return result


def _find_wechat_process() -> Optional[str]:
    """检测微信进程是否存在。"""
    wechat_names = ["Weixin.exe", "WeChat.exe", "weixin.exe", "wechat.exe"]
    try:
        # Windows: tasklist
        output = subprocess.check_output(
            ["tasklist", "/FI", "STATUS eq RUNNING"],
            text=True, timeout=5,
        )
        for name in wechat_names:
            if name.lower() in output.lower():
                return name
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # 备用方案：检查常见路径下是否有锁文件或进程
    try:
        import psutil
        for proc in psutil.process_iter(["name"]):
            try:
                if proc.info["name"] and proc.info["name"].lower() in [
                    n.lower() for n in wechat_names
                ]:
                    return proc.info["name"]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except ImportError:
        pass

    return None


def _find_wechat_data_dir() -> tuple:
    """查找微信数据目录和 wxid。

    Returns:
        (wxid, data_dir) 或 (None, None)
    """
    # Windows 常见路径
    home = Path.home()
    candidates = [
        home / "Documents" / "WeChat Files",
        home / "Documents" / "xwechat_files",
        Path("D:/WeChat Files"),
        Path("E:/WeChat Files"),
    ]

    for base in candidates:
        if not base.exists():
            continue
        for entry in base.iterdir():
            if entry.is_dir() and entry.name not in ("All Users", "Applet", "WMPF"):
                # 检查是否有 Msg 子目录（确认是微信数据目录）
                msg_dir = entry / "Msg"
                if msg_dir.exists():
                    return entry.name, entry
                # 微信 4.x 可能在 db_storage
                db_storage = entry / "db_storage"
                if db_storage.exists():
                    return entry.name, entry

    return None, None
