"""WeChat key extraction helper - runs with UAC elevation.

This script runs as a separate process with admin privileges
to extract the SQLCipher key from WeChat process memory.
"""
import subprocess
import sys
import json


def extract_key_elevated() -> str:
    """Run key extraction in an elevated process.

    Uses PowerShell Start-Process -Verb RunAs to trigger UAC elevation.
    Returns the key string or empty string on failure.
    """
    helper_script = __file__
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy", "Bypass",
                "-Command",
                f"Start-Process -FilePath '{sys.executable}' "
                f"-ArgumentList '{helper_script} --extract' "
                f"-Verb RunAs -Wait -WindowStyle Hidden; "
                f"if (Test-Path '{helper_script}.result') {{ "
                f"Get-Content '{helper_script}.result' "
                f"}}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout.strip()
        if output and len(output) == 64:
            return output
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return ""


def _extract_and_save():
    """Internal: extract key and save to result file."""
    import os
    result_file = __file__ + ".result"

    try:
        from data_ingestion.wechat_key import extract_key_from_memory
        key = extract_key_from_memory()
        if key:
            with open(result_file, "w") as f:
                f.write(key)
            print(key)
            return
    except Exception:
        pass

    # Clean up on failure
    try:
        os.remove(result_file)
    except OSError:
        pass


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--extract":
        _extract_and_save()
    else:
        key = extract_key_elevated()
        if key:
            print(key)
        else:
            print("", end="")
