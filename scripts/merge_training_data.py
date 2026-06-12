"""合并训练数据：合成数据 + Enron/CUAD 真实数据集。

策略：
- 真实企业邮件（Enron, 120封）→ 作为企业场景核心
- 真实合同Q&A（CUAD, 50条）→ 训练合同条款抽取能力
- 中文合成数据（180条）→ 保持中文风格

合并后 ~400 条，8:1:1 划分。
"""

import json
import os
import random

random.seed(42)

SYNTHETIC = "data/training_data.json"
REAL_EN = "data/training_data_real_en.json"
OUTPUT_DIR = "data"
FINAL_FILES = {
    "train": "data/training_data_train.json",
    "val": "data/training_data_val.json",
    "test": "data/training_data_test.json",
}


def main():
    all_data = []

    # 1. 加载合成数据
    synth_count = 0
    if os.path.exists(SYNTHETIC):
        with open(SYNTHETIC, "r", encoding="utf-8") as f:
            synth = json.load(f)
        if isinstance(synth, dict) and "train" in synth:
            synth_data = synth["train"] + synth.get("val", []) + synth.get("test", [])
        else:
            synth_data = synth if isinstance(synth, list) else []
        all_data.extend(synth_data)
        synth_count = len(synth_data)
        print(f"合成数据: {synth_count} 条")
    else:
        print(f"合成数据不存在: {SYNTHETIC}（先运行 generate_training_data.py）")

    # 2. 加载真实数据
    real_count = 0
    if os.path.exists(REAL_EN):
        with open(REAL_EN, "r", encoding="utf-8") as f:
            real = json.load(f)
        all_data.extend(real)
        real_count = len(real)
        print(f"真实数据: {real_count} 条 (Enron+CUAD)")
    else:
        print(f"真实数据不存在: {REAL_EN}（先运行 extract_real_datasets.py）")

    # 3. 去重（基于 input 字段的前 100 字符）
    seen = set()
    dedup = []
    for item in all_data:
        key = item.get("input", "")[:100]
        if key not in seen:
            seen.add(key)
            dedup.append(item)
    print(f"去重后: {len(dedup)} 条（去掉了 {len(all_data) - len(dedup)} 条重复）")

    # 4. 打乱 + 划分
    random.shuffle(dedup)
    n = len(dedup)
    train = dedup[: int(n * 0.8)]
    val = dedup[int(n * 0.8) : int(n * 0.9)]
    test = dedup[int(n * 0.9) :]

    splits = {"train": train, "val": val, "test": test}

    for name, data in splits.items():
        path = FINAL_FILES[name]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  {name}: {len(data)} 条 → {path}")

    print(f"\n总计: {n} 条训练数据")
    print(f"  来源: 合成 {synth_count} + 真实企业 {real_count}")
    print(f"\n下一步: python scripts/train_qlora.py")


if __name__ == "__main__":
    main()
