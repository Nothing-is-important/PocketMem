"""微信密钥提取一键诊断脚本。

用法（需要管理员权限 + 微信已登录）：
    uv run python scripts/diag_wechat_key.py

或者直接 Python：
    python scripts/diag_wechat_key.py
"""
import sys
import ctypes
import time

# Fix: Ensure UTF-8 output on Windows GBK terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=" * 60)
print("  微信密钥提取诊断")
print("=" * 60)

# 1. Admin check
try:
    is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
except Exception:
    is_admin = sys.platform != "win32"
print(f"\n[1] 管理员权限: {'[OK] 是' if is_admin else '[FAIL] 否（需要以管理员身份运行）'}")

# 2. WeChat process check
import subprocess
wechat_running = False
for name in ["Weixin.exe", "WeChat.exe"]:
    try:
        r = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {name}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=5,
        )
        if name.lower() in r.stdout.lower():
            wechat_running = True
            break
    except Exception:
        pass
print(f"[2] 微信进程: {'[OK] 运行中' if wechat_running else '[FAIL] 未运行'}")

if not wechat_running:
    print("\n[!] 请先启动微信并登录，然后重新运行本脚本。")
    sys.exit(0)

# 3. Import check
print("[3] 依赖检查:")
for mod_name in ["pymem", "Crypto", "psutil"]:
    try:
        __import__(mod_name)
        print(f"    [OK] {mod_name}")
    except ImportError:
        print(f"    [FAIL] {mod_name} 未安装")

# 4. Key extraction
print("\n[4] 提取密钥...")
sys.path.insert(0, ".")

try:
    from data_ingestion.wechat_key import get_wechat_key

    t0 = time.time()
    key = get_wechat_key()
    elapsed = time.time() - t0

    if key:
        print(f"    [OK] 成功！耗时 {elapsed:.1f}s")
        print(f"    密钥: {key}")
    else:
        print(f"    [FAIL] 失败（耗时 {elapsed:.1f}s）")
        if not is_admin:
            print("    -> 请以管理员身份运行本脚本")
        print("    -> 确认微信已登录")
except Exception as e:
    print(f"    [FAIL] 异常: {e}")

print("\n" + "=" * 60)
