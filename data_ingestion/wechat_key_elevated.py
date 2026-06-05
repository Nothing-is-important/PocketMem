"""WeChat key extraction - UAC elevation via ShellExecuteW.

Uses Windows ShellExecute with 'runas' verb to trigger UAC popup
for one-time admin privilege escalation.
"""
import ctypes
import sys
import os
import tempfile
import time


def extract_key_elevated() -> str:
    """Extract key via UAC-elevated subprocess.

    Writes a temp Python script, launches it with ShellExecute(runas),
    waits for result file. Returns key or empty string.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    result_file = os.path.join(tempfile.gettempdir(), "pm_key_result.txt")
    script_file = os.path.join(tempfile.gettempdir(), "pm_extract_key.py")

    # Clean old results
    for f in (result_file, script_file):
        try:
            os.remove(f)
        except OSError:
            pass

    # Write extraction script
    script_code = f'''
import sys, os
sys.path.insert(0, r"{project_root}")
try:
    from data_ingestion.wechat_key import extract_key_from_memory, _scan_hex_key_in_memory
    key = extract_key_from_memory()
    if not key:
        key = _scan_hex_key_in_memory()
    with open(r"{result_file}", "w") as f:
        f.write(key if key else "NONE")
except Exception as e:
    with open(r"{result_file}", "w") as f:
        f.write(f"ERROR:{e}")
'''
    with open(script_file, "w", encoding="utf-8") as f:
        f.write(script_code)

    # Trigger UAC via ShellExecuteW
    try:
        ret = ctypes.windll.shell32.ShellExecuteW(
            None,                      # hwnd
            "runas",                   # verb - triggers UAC popup
            sys.executable,            # python.exe
            f'"{script_file}"',        # args
            None,                      # working dir
            0,                         # SW_HIDE
        )
        if ret <= 32:  # ShellExecute failed
            return ""

        # Wait for elevated process to finish (max 60s)
        for _ in range(120):
            if os.path.exists(result_file):
                time.sleep(0.3)
                with open(result_file, "r") as f:
                    content = f.read().strip()
                try:
                    os.remove(result_file)
                except OSError:
                    pass
                if content and not content.startswith("ERROR") and content != "NONE" and len(content) == 64:
                    return content
                return ""
            time.sleep(0.5)
    except Exception:
        pass
    finally:
        try:
            os.remove(script_file)
        except OSError:
            pass

    return ""
