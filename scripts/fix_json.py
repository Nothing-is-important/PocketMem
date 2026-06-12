"""修复 JSON 批次文件中的中文引号问题。"""
import json, os

BATCHES = [
    "data/training_data_cn_batch2.json",
    "data/training_data_cn_batch3.json",
    "data/training_data_cn_batch4.json",
    "data/training_data_cn_batch5.json",
    "data/training_data_cn_batch6.json",
]

for fname in BATCHES:
    with open(fname, "r", encoding="utf-8") as f:
        text = f.read()

    # Chinese curly double quotes -> escaped straight quotes
    text = text.replace("“", '\\"')  # "
    text = text.replace("”", '\\"')  # "

    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)

    try:
        with open(fname, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"{fname}: {len(data)} entries OK")
    except json.JSONDecodeError as e:
        print(f"{fname}: STILL BROKEN at line {e.lineno}: {e.msg}")
        # Show the problematic line
        lines = text.split("\n")
        if e.lineno > 0:
            start = max(0, e.lineno - 2)
            end = min(len(lines), e.lineno + 2)
            for i in range(start, end):
                marker = ">>>" if i + 1 == e.lineno else "   "
                print(f"  {marker} L{i+1}: {lines[i][:120]}")
