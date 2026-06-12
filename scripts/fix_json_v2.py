"""Re-encode all batch files as proper JSON using Python's json.dumps()."""
import json, os

BATCHES = [
    "data/training_data_cn_batch1.json",
    "data/training_data_cn_batch2.json",
    "data/training_data_cn_batch3.json",
    "data/training_data_cn_batch4.json",
    "data/training_data_cn_batch5.json",
    "data/training_data_cn_batch6.json",
]

total = 0
for fname in BATCHES:
    with open(fname, "r", encoding="utf-8") as f:
        content = f.read()

    # Strategy: Use Python's JSON parser leniently by using json.loads with strict=False
    # If that fails, try ast.literal_eval
    # If that fails, manually try to extract each dict

    # Clean up: replace Chinese curly quotes with escaped straight quotes inside JSON strings
    content = content.replace("“", '\\"')  # "
    content = content.replace("”", '\\"')  # "
    content = content.replace("‘", "'")    # '
    content = content.replace("’", "'")    # '

    try:
        data = json.loads(content)
        # Re-serialize to ensure clean JSON
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"{fname}: {len(data)} entries (re-encoded)")
        total += len(data)
    except json.JSONDecodeError as e:
        print(f"{fname}: FAILED at line {e.lineno} col {e.colno}: {e.msg}")
        # Show context
        lines = content.split("\n")
        for i in range(max(0, e.lineno-3), min(len(lines), e.lineno+2)):
            marker = ">>>" if i+1 == e.lineno else "   "
            snippet = lines[i][:100]
            # Replace non-ASCII with repr for debugging
            if any(ord(c) > 127 for c in snippet[:20]):
                snippet_hex = ' '.join(f'{ord(c):04X}' for c in snippet[:10])
                print(f"  {marker} L{i+1}: [{snippet_hex}...]")
            else:
                print(f"  {marker} L{i+1}: {snippet}")

print(f"\nTotal: {total} entries")
