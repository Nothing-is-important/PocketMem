"""最终修复：将 JSON 字符串内部的英文双引号替换为中文书名号。"""
import json, re, os

BATCHES = [
    "data/training_data_cn_batch2.json",
    "data/training_data_cn_batch3.json",
    "data/training_data_cn_batch4.json",
    "data/training_data_cn_batch5.json",
    "data/training_data_cn_batch6.json",
]

def fix_inner_quotes(text):
    """在 JSON 字符串值内部，将英文直双引号替换为中文书名号。
    只处理行内的 "xxx" 模式（中文语境中的英文引号）。
    """
    lines = text.split("\n")
    fixed_lines = []
    for line in lines:
        stripped = line.strip()
        # 只处理 JSON 值行：以 "input": " 或 "output": " 开头
        if stripped.startswith('"input": "') or stripped.startswith('"output": "'):
            # 提取 JSON key + 开头的引号
            prefix = '"input": "' if '"input"' in stripped[:20] else '"output": "'
            # 提取值内容（去除前缀和结尾的 "）
            body = stripped[len(prefix):]
            if body.endswith('",'):
                body = body[:-2]
            elif body.endswith('"'):
                body = body[:-1]
            # 替换值内部的英文双引号（中文语境中）
            body = body.replace('"', "《")
            body = body.replace('"', "》")
            # 重建行
            if stripped.endswith('",'):
                line = prefix + body + '",'
            elif stripped.endswith('"'):
                line = prefix + body + '"'
        fixed_lines.append(line)
    return "\n".join(fixed_lines)

total = 0
for fname in BATCHES:
    with open(fname, "r", encoding="utf-8") as f:
        content = f.read()

    # 修复中文语境中的英文双引号
    content = fix_inner_quotes(content)

    try:
        data = json.loads(content)
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"{fname}: {len(data)} entries FIXED")
        total += len(data)
    except json.JSONDecodeError as e:
        print(f"{fname}: STILL BROKEN at L{e.lineno} C{e.colno}")
        # Show hex of problem area
        lines = content.split("\n")
        line = lines[e.lineno - 1] if e.lineno <= len(lines) else ""
        start = max(0, e.colno - 10)
        snippet = line[start:e.colno + 10]
        print(f"  Around error: ...{snippet}...")

print(f"\nTotal fixed: {total} entries")
