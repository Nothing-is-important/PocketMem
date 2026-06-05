"""WeChat key extraction helper - UAC elevation.

When called as main, extracts key and prints to stdout.
When imported, provides extract_key_elevated() for subprocess call.
"""
import subprocess
import sys
import os
import json


def extract_key_elevated() -> str:
    """Run key extraction via UAC-elevated subprocess.

    Triggers Windows UAC popup. Returns key or empty string.
    """
    script = __file__
    try:
        # Write a temp PowerShell script that:
        # 1. Runs Python with the current script as admin
        # 2. Captures output
        ps_cmd = (
            f'$p = Start-Process -FilePath "{sys.executable}" '
            f'-ArgumentList "{script}", "--extract" '
            f'-Verb RunAs -Wait -PassThru -WindowStyle Hidden; '
            f'$p.ExitCode'
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=60,
        )
        exit_code = result.stdout.strip()
        # Read result file
        result_file = script + ".result"
        if os.path.exists(result_file):
            with open(result_file) as f:
                key = f.read().strip()
            os.remove(result_file)
            if len(key) == 64:
                return key
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return ""


def _extract_and_save():
    """Internal: extract key and save to temp file."""
    result_file = __file__ + ".result"
    try:
        from data_ingestion.wechat_key import extract_key_from_memory
        key = extract_key_from_memory()
        if key:
            with open(result_file, "w") as f:
                f.write(key)
    except Exception:
        # Try generic hex scan
        try:
            from data_ingestion.wechat_key import _scan_hex_key_in_memory
            key = _scan_hex_key_in_memory()
            if key:
                with open(result_file, "w") as f:
                    f.write(key)
        except Exception:
            pass


if __name__ == "__main__":
    _extract_and_save()
