"""合并所有中文翻译批次 + 导出项目数据文件。

1. 合并 batch1-batch6 → 完整中文训练数据
2. 随机打乱 + 8:1:1 划分
3. 同时导出为项目数据（data/demo/enron_cn/）
"""

import json, os, random, re
from pathlib import Path

random.seed(42)

BATCHES = [
    "data/training_data_cn_batch1.json",
    "data/training_data_cn_batch2.json",
    "data/training_data_cn_batch3.json",
    "data/training_data_cn_batch4.json",
    "data/training_data_cn_batch5.json",
    "data/training_data_cn_batch6.json",
]
PROJECT_DATA_DIR = "data/demo/enron_cn"


def merge_batches():
    all_data = []
    for path in BATCHES:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                batch = json.load(f)
            all_data.extend(batch)
            print(f"  {path}: {len(batch)} 条")
    return all_data


def proofread(data):
    """校对：检查每条数据的基本质量。"""
    issues = []
    clean = []
    for i, item in enumerate(data):
        inp = item.get("input", "")
        out = item.get("output", "")
        inst = item.get("instruction", "")

        # 检查必填字段
        if not inp or not out or not inst:
            issues.append(f"[{i}] 缺字段")
            continue
        # 检查引用
        if "[来源:" not in out and "来源" not in out:
            issues.append(f"[{i}] 缺少来源引用")
        # 检查最小长度
        if len(inp) < 50 or len(out) < 30:
            issues.append(f"[{i}] 长度过短")
            continue

        clean.append(item)

    if issues:
        print(f"\n校对发现 {len(issues)} 个问题:")
        for iss in issues[:10]:
            print(f"  {iss}")
    return clean


def export_project_data(data):
    """将训练数据中的邮件正文导出为可索引的项目数据文件。"""
    os.makedirs(PROJECT_DATA_DIR, exist_ok=True)
    exported = 0
    seen = set()

    for i, item in enumerate(data):
        inp = item.get("input", "")
        # 提取邮件主题
        m = re.search(r"主题[:：]\s*(.+?)(?:\n|$)", inp)
        subject = (m.group(1).strip()[:60] if m else f"企业文档_{i:03d}")[:60]

        # 提取正文（移除指令部分）
        body = inp.split("问题:")[0] if "问题:" in inp else inp

        if subject in seen:
            continue
        seen.add(subject)

        safe = re.sub(r'[/\\:*?"<>|\n\r]', '', subject)[:40]
        fname = f"enron_{i:03d}_{safe}.txt"
        fpath = os.path.join(PROJECT_DATA_DIR, fname)

        with open(fpath, "w", encoding="utf-8") as f:
            f.write(f"From: Enron企业邮件 <enron@enron.com>\n")
            f.write(f"Subject: {subject}\n")
            f.write(f"Level: internal\n")
            f.write(f"\n{body[:1200]}\n")
        exported += 1

    print(f"项目数据: {exported} 个文件 → {PROJECT_DATA_DIR}/")


def main():
    print("合并中文翻译批次")
    print("=" * 50)

    # 1. 合并
    all_data = merge_batches()
    print(f"\n合并总计: {len(all_data)} 条")

    # 2. 校对
    clean = proofread(all_data)
    print(f"校对后: {len(clean)} 条")

    # 3. 打乱 + 划分
    random.shuffle(clean)
    n = len(clean)
    splits = {
        "train": clean[:int(n * 0.8)],
        "val": clean[int(n * 0.8):int(n * 0.9)],
        "test": clean[int(n * 0.9):],
    }

    for name, subset in splits.items():
        path = f"data/training_data_{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)
        print(f"  {name}: {len(subset)} 条 → {path}")

    # 4. 导出项目数据
    export_project_data(clean)

    # 5. 全量保存
    with open("data/training_data_cn_full.json", "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)

    print(f"\n完成: {len(clean)} 条中文训练数据 + {PROJECT_DATA_DIR}/ 项目数据")
    print("下一步: python scripts/train_qlora.py --model Qwen3-4B")


if __name__ == "__main__":
    main()
