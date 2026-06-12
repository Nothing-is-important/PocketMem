"""从 Enron + CUAD 真实数据集提取训练数据。

Enron: 1.3GB 真实企业邮件 → 精选 150 封代表性邮件
CUAD:   40MB 商业合同Q&A → 精选 50 条相关问答对

输出: data/training_data_real.json（补充到合成数据中）
"""

import json
import os
import random
import re
from datetime import datetime
from pathlib import Path

random.seed(42)

ENRON_DIR = r"e:\PythonProject\enron_mail_20150507\maildir"
CUAD_FILE = r"e:\PythonProject\cuad-main\cuad-main\data\CUADv1.json"
OUTPUT = "data/training_data_real_en.json"


# ═══════════════════════════════════════════════════════════════
# Enron 邮件提取
# ═══════════════════════════════════════════════════════════════

def extract_enron_emails(max_per_person=20, max_total=150):
    """从 Enron 数据集精选代表性邮件。

    策略：
    - 选 5-6 个核心人物（CEO/高层/中层/律师）
    - 每人取 ~20 封不同主题的邮件
    - 过滤掉太短/纯转发/垃圾邮件
    - 保留邮件头 + 正文
    """
    key_people = ["skilling-j", "lay-k", "kaminski-v", "kean-s", "shapiro-r", "sanders-r"]
    available = [d for d in os.listdir(ENRON_DIR) if os.path.isdir(os.path.join(ENRON_DIR, d))]
    selected = [p for p in key_people if p in available]

    if len(selected) < 3:
        # fallback: take any
        selected = available[:6]

    print(f"Enron: 从 {len(available)} 个邮箱中选取 {len(selected)} 个核心人物")

    emails = []
    for person in selected[:6]:
        person_dir = os.path.join(ENRON_DIR, person)
        person_emails = []
        for category in os.listdir(person_dir):
            cat_dir = os.path.join(person_dir, category)
            if not os.path.isdir(cat_dir):
                continue
            for fname in os.listdir(cat_dir):
                fpath = os.path.join(cat_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    parsed = _parse_enron_mail(content)
                    if parsed and len(parsed["body"]) > 100:
                        person_emails.append(parsed)
                except Exception:
                    continue

        random.shuffle(person_emails)
        picked = person_emails[:max_per_person]
        emails.extend(picked)
        print(f"  {person}: {len(person_emails)} 封邮件 → 选取 {len(picked)} 封")

    print(f"Enron 总计: {len(emails)} 封邮件")
    return emails


def _parse_enron_mail(raw):
    """解析 Enron 邮件格式。"""
    headers_end = raw.find("\n\n")
    if headers_end == -1:
        return None

    headers_text = raw[:headers_end]
    body = raw[headers_end:].strip()

    def hdr(key):
        pattern = re.compile(rf"^{key}:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
        m = pattern.search(headers_text)
        return m.group(1).strip() if m else ""

    subject = hdr("Subject")
    sender = hdr("From")
    to = hdr("To")
    date_str = hdr("Date")

    # 过滤垃圾
    if not subject or not body:
        return None
    if len(body) < 100:
        return None
    if re.search(r"SPAM|advertisement|unsubscribe", subject, re.IGNORECASE):
        return None
    if re.search(r">[^>]*$", body[:200]):  # 纯引用
        return None

    # 清理正文：去除 > 引用行和签名
    clean_body = []
    for line in body.split("\n"):
        line = line.strip()
        if line.startswith(">"):
            continue
        if line in ("Best regards", "Regards", "Sincerely", "Thanks", "---"):
            break
        clean_body.append(line)

    return {
        "sender": _clean_email_addr(sender),
        "to": _clean_email_addr(to),
        "subject": subject[:150],
        "date": date_str[:50],
        "body": "\n".join(clean_body)[:800],
    }


def _clean_email_addr(s):
    """提取 email 地址中的名称或直接返回。"""
    s = s.strip()
    m = re.search(r'"([^"]+)"', s)
    if m:
        return m.group(1)
    m = re.search(r"([^<\s]+@[^>\s]+)", s)
    if m:
        return m.group(1).split("@")[0]
    return s[:50]


# ═══════════════════════════════════════════════════════════════
# CUAD 合同Q&A 提取
# ═══════════════════════════════════════════════════════════════

def extract_cuad_questions(max_questions=50):
    """从 CUAD 提取合同Q&A对。"""
    print(f"\nCUAD: 加载 {CUAD_FILE}...")

    with open(CUAD_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # CUAD 格式: {"data": [{"title": "...", "paragraphs": [{"context": "...", "qas": [...]}]}]}
    entries = []
    for item in data.get("data", []):
        title = item.get("title", "")
        for para in item.get("paragraphs", []):
            context = para.get("context", "")[:1000]
            for qa in para.get("qas", []):
                question = qa.get("question", "")
                answers = qa.get("answers", [])
                answer_text = answers[0]["text"] if answers else ""
                if question and answer_text and len(context) > 50:
                    entries.append({
                        "title": title,
                        "context": context,
                        "question": question,
                        "answer": answer_text,
                    })

    print(f"CUAD 总共: {len(entries)} 条 Q&A")
    random.shuffle(entries)

    # 选与项目知识库场景相关的类别
    relevant_types = [
        "Agreement Date", "Effective Date", "Expiration Date",
        "Parties", "Governing Law", "Renewal Term",
        "Termination", "Confidentiality", "Payment Terms",
    ]

    picked = []
    for e in entries:
        if any(t.lower() in e["title"].lower() for t in relevant_types):
            picked.append(e)
        if len(picked) >= max_questions:
            break

    if len(picked) < max_questions:
        picked = entries[:max_questions]

    print(f"CUAD 选取: {len(picked)} 条（{len(set(e['title'] for e in picked))} 种条款类型）")
    return picked


# ═══════════════════════════════════════════════════════════════
# 转为训练数据格式
# ═══════════════════════════════════════════════════════════════

def to_training_format(enron_emails, cuad_questions):
    """将 Enron 邮件和 CUAD Q&A 转为 Alpaca 训练格式。"""
    data = []

    # Enron 邮件 → 事实型和实体型问题
    for i, mail in enumerate(enron_emails):
        subject = mail["subject"]
        sender = mail["sender"]
        body = mail["body"][:500]

        # 每条邮件生成1-2个训练样本
        templates = [
            {
                "instruction": "You are an enterprise knowledge assistant. Answer based on the document provided. Cite the source.",
                "input": f"Document 1 (Email from {sender}):\nSubject: {subject}\n{body}\n\nQuestion: What is the main topic of this email?",
                "output": f"Based on the email from {sender}, the main topic is: {subject}. The key points mentioned are: {body[:200]} [Source: Email from {sender}]",
            },
            {
                "instruction": "You are an enterprise knowledge assistant. Extract key information from the document.",
                "input": f"Document 1:\n{body}\n\nQuestion: What key decisions or action items are mentioned in this document?",
                "output": f"Based on the provided document from {sender} regarding '{subject}', the key information extracted is: {body[:300]} [Source: Document 1]",
            },
        ]

        data.append(random.choice(templates))

    # CUAD 合同Q&A → 事实型问题
    for item in cuad_questions:
        data.append({
            "instruction": "You are an enterprise knowledge assistant. Answer based on the contract excerpt provided. Cite the specific clause.",
            "input": f"Contract Excerpt:\n{item['context'][:600]}\n\nQuestion: {item['question']}",
            "output": f"According to the contract, {item['answer']}. [Source: {item['title']} clause]",
        })

    return data


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    print("从 Enron + CUAD 提取真实企业训练数据")
    print("=" * 50)

    # 1. Enron
    emails = extract_enron_emails(max_per_person=20, max_total=150)

    # 2. CUAD
    cuad = extract_cuad_questions(max_questions=50)

    # 3. 转为训练格式
    train_data = to_training_format(emails, cuad)

    # 4. 保存
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    print(f"\n输出: {OUTPUT}")
    print(f"总计: {len(train_data)} 条训练数据")
    print(f"  Enron 邮件: {len(emails)} 封 → ~{len(emails)*2} 条")
    print(f"  CUAD 合同:  {len(cuad)} 条 Q&A")
    print()
    print("下一步: 将这些数据与合成数据合并")
    print("  python scripts/merge_training_data.py")


if __name__ == "__main__":
    main()
