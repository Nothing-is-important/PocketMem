"""Enron 邮件翻译 + 多轮校对 + 导入为项目数据。

两阶段翻译：
1. 第一轮：英文→中文翻译
2. 第二轮：用不同 prompt 重新翻译，比较关键句，选更通顺的

翻译后：
- 一部分作为 QLoRA 训练数据（~120 条）
- 一部分作为实际项目数据（~120 条，放入 data/demo/enron_cn/）

用法：
  python scripts/translate_enron.py --translate   # 翻译所有邮件
  python scripts/translate_enron.py --import-data # 导入到项目数据目录
"""

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

ENRON_INPUT = "data/training_data_real_en.json"
ENRON_OUTPUT_CN = "data/training_data_real_cn.json"
PROJECT_DATA_DIR = "data/demo/enron_cn"


def load_english_emails():
    """从已提取的训练数据中获取英文邮件原文。"""
    if not os.path.exists(ENRON_INPUT):
        print(f"文件不存在: {ENRON_INPUT}")
        print("先运行: python scripts/extract_real_datasets.py")
        return []

    with open(ENRON_INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    emails = []
    seen_subjects = set()
    for item in data:
        inp = item.get("input", "")
        # 提取邮件标题
        subj_match = re.search(r"Subject:\s*(.+?)(?:\n|$)", inp)
        subject = subj_match.group(1).strip() if subj_match else ""
        if subject and subject not in seen_subjects:
            seen_subjects.add(subject)
            emails.append({
                "subject": subject,
                "input": inp,
                "output": item.get("output", ""),
                "instruction": item.get("instruction", ""),
            })

    print(f"待翻译邮件: {len(emails)} 封")
    return emails


def translate_with_backend(emails, backend):
    """使用本地模型翻译邮件。"""
    translated = []
    total = len(emails)
    batch_size = 5  # 批量翻译以节省时间

    for i in range(0, total, batch_size):
        batch = emails[i:i + batch_size]
        for j, email in enumerate(batch):
            idx = i + j + 1
            print(f"  翻译 [{idx}/{total}]: {email['subject'][:50]}...")

            # 第一轮翻译
            prompt_v1 = f"""Translate the following business email to natural Chinese. Keep the original structure and key information (names, dates, numbers). Make it sound like a real Chinese business email.

Subject: {email['subject']}
Content: {email['input'][:600]}

Chinese translation:"""

            try:
                cn_v1 = backend.generate(prompt_v1, max_tokens=300).strip()
            except Exception as e:
                print(f"    翻译失败: {e}")
                cn_v1 = email["input"]

            # 第二轮校对：用不同 prompt 检测关键信息保留
            prompt_v2 = f"""Check this Chinese translation of a business email. If the translation is natural and complete, output "OK". If there are issues, output the CORRECTED translation.

Original (English):
Subject: {email['subject']}
{email['input'][:500]}

Chinese translation:
{cn_v1[:400]}

Your response (OK or corrected translation):"""

            try:
                proof = backend.generate(prompt_v2, max_tokens=300).strip()
                cn_final = cn_v1 if proof.startswith("OK") else proof
            except Exception:
                cn_final = cn_v1

            translated.append({
                "subject": email["subject"],
                "en_input": email["input"],
                "cn_input": cn_final,
                "en_output": email["output"],
                "instruction": email["instruction"],
            })

    return translated


def save_translated_data(emails, output_path):
    """保存翻译后的训练数据。"""
    training_data = []
    for e in emails:
        training_data.append({
            "instruction": "你是企业知识助手。根据以下文档内容回答问题。如果文档中没有相关信息，明确说明。回答时引用文档编号。",
            "input": f"文档1 (来自{e.get('sender', '企业邮箱')}):\n主题: {e['subject']}\n{e['cn_input'][:600]}\n\n问题: 这封邮件的主要内容是什么？",
            "output": f"根据文档1，这封邮件的主题是：{e['subject']}。邮件讨论了以下关键信息：{e['cn_input'][:300]} [来源: 文档1]",
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    print(f"翻译训练数据已保存: {output_path} ({len(training_data)} 条)")


def import_as_project_data(emails):
    """将翻译后的邮件导入为实际项目数据。"""
    os.makedirs(PROJECT_DATA_DIR, exist_ok=True)

    imported = 0
    for i, email in enumerate(emails):
        if not email.get("cn_input"):
            continue

        # 清理文件名
        safe_subject = re.sub(r'[/\\:*?"<>|\n\r]', '', email["subject"])[:40]
        fname = f"enron_{i:03d}_{safe_subject}.txt"
        fpath = os.path.join(PROJECT_DATA_DIR, fname)

        with open(fpath, "w", encoding="utf-8") as f:
            f.write(f"From: Enron企业邮件 <enron@enron.com>\n")
            f.write(f"Subject: {email['subject']}\n")
            f.write(f"Level: internal\n")
            f.write(f"\n{email['cn_input'][:1000]}\n")

        imported += 1

    print(f"项目数据已导入: {PROJECT_DATA_DIR}/ ({imported} 封邮件)")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--translate", action="store_true", help="翻译 Enron 邮件")
    p.add_argument("--import-data", action="store_true", help="导入为项目数据")
    p.add_argument("--all", action="store_true", help="全流程：翻译+导入+训练数据")
    args = p.parse_args()

    if not (args.translate or args.import_data or args.all):
        p.print_help()
        return

    emails = load_english_emails()
    if not emails:
        return

    if args.translate or args.all:
        print(f"\n加载模型进行翻译...")
        from backend import create_backend
        backend = create_backend("local_simulate", device="cuda")

        translated = translate_with_backend(emails, backend)
        save_translated_data(translated, ENRON_OUTPUT_CN)
        print(f"\n翻译完成: {len(translated)} 封")

    if args.import_data or args.all:
        if not emails[0].get("cn_input"):
            print("请先运行 --translate 翻译邮件")
            return
        import_as_project_data(emails)

    print("\nDone.")


if __name__ == "__main__":
    main()
