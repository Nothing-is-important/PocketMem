"""从全部 150 个 Enron 邮箱 + 全部 CUAD 中全面提取数据。

策略：
- Enron: 150个邮箱，每人采样 5-8 封高质量邮件 → ~800-1000 封
- CUAD: 20,910条Q&A，过滤后取 200-300 条
- 翻译成中文 → 训练数据

为什么不全取？528K 封邮件翻译工作量太大，而且大量是自动回复/垃圾/纯转发。
800 封精选 + 300 条 CUAD = 1100 条，远超 QLoRA 行为适配所需的 500 条下限。
"""

import json
import os
import random
import re
from pathlib import Path

random.seed(42)

ENRON_DIR = r"e:\PythonProject\enron_mail_20150507\maildir"
CUAD_FILE = r"e:\PythonProject\cuad-main\cuad-main\data\CUADv1.json"
OUTPUT_EN = "data/enron_cuad_full_en.json"


def extract_enron_all(per_person=6, max_total=900):
    """从所有 Enron 邮箱中提取高质量邮件。"""
    people = sorted([d for d in os.listdir(ENRON_DIR)
                     if os.path.isdir(os.path.join(ENRON_DIR, d))])

    print(f"Enron: {len(people)} 个邮箱")

    all_emails = []
    skipped = {"short": 0, "spam": 0, "forward": 0, "autoreply": 0}

    for person in people:
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
                except Exception:
                    continue

                parsed = _parse_enron(content)
                if not parsed:
                    skipped["short"] += 1
                    continue
                if _is_spam(parsed):
                    skipped["spam"] += 1
                    continue
                if _is_forward(parsed):
                    skipped["forward"] += 1
                    continue
                if _is_autoreply(parsed):
                    skipped["autoreply"] += 1
                    continue

                person_emails.append(parsed)

        # 每人取 per_person 封，优先不同主题
        random.shuffle(person_emails)
        seen_subjects = set()
        picked = []
        for e in person_emails:
            subj_key = e["subject"][:40].lower()
            if subj_key not in seen_subjects:
                seen_subjects.add(subj_key)
                picked.append(e)
            if len(picked) >= per_person:
                break

        all_emails.extend(picked)

        if len(all_emails) >= max_total:
            break

    print(f"  选取 {len(all_emails)} 封（来自 {len(people)} 人）")
    print(f"  过滤: 短/空={skipped['short']} 垃圾={skipped['spam']} 转发={skipped['forward']} 自动回复={skipped['autoreply']}")
    return all_emails


def _parse_enron(raw):
    """解析 Enron 邮件。"""
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
    date_str = hdr("Date")

    if not subject or not body or len(body) < 80:
        return None

    # 清理正文
    clean_body = []
    for line in body.split("\n"):
        line = line.strip()
        if line.startswith(">"):
            continue
        if re.match(r"^-{3,}", line):
            break
        if line in ("Best regards", "Regards", "Sincerely", "Thanks", "---"):
            break
        clean_body.append(line)

    return {
        "sender": _clean_sender(sender),
        "subject": subject[:120],
        "date": date_str[:50],
        "body": "\n".join(clean_body)[:600],
    }


def _clean_sender(s):
    s = s.strip()
    m = re.search(r'"([^"]+)"', s)
    if m: return m.group(1)
    m = re.search(r"([^<\s]+@[^>\s]+)", s)
    if m: return m.group(1).split("@")[0]
    return s[:40]


def _is_spam(parsed):
    spam_words = ["unsubscribe", "click here", "free", "offer", "discount",
                  "SPAM", "advertisement", "limited time", "act now"]
    text = (parsed["subject"] + " " + parsed["body"][:200]).lower()
    return any(w in text for w in spam_words)


def _is_forward(parsed):
    return parsed["subject"].lower().startswith("fw:") or \
           parsed["subject"].lower().startswith("fwd:")


def _is_autoreply(parsed):
    return "out of office" in (parsed["subject"] + parsed["body"][:100]).lower() or \
           "auto:" in parsed["subject"].lower()


def extract_cuad_all(max_questions=300):
    """从 CUAD 提取全部相关 Q&A。"""
    print(f"\nCUAD: 加载...")
    with open(CUAD_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = []
    for item in data.get("data", []):
        title = item.get("title", "")
        for para in item.get("paragraphs", []):
            context = para.get("context", "")[:800]
            for qa in para.get("qas", []):
                question = qa.get("question", "")
                answers = qa.get("answers", [])
                answer_text = answers[0]["text"] if answers else ""
                if question and answer_text and len(context) > 40:
                    entries.append({
                        "title": title,
                        "context": context,
                        "question": question,
                        "answer": answer_text,
                    })

    print(f"  CUAD 全部: {len(entries)} 条")

    # 去重（按问题文本）
    seen_q = set()
    unique = []
    for e in entries:
        q = e["question"][:60]
        if q not in seen_q:
            seen_q.add(q)
            unique.append(e)

    random.shuffle(unique)
    result = unique[:max_questions]

    types = set(e["title"] for e in result)
    print(f"  去重后选取: {len(result)} 条 ({len(types)} 种合同条款)")
    return result


def to_training_format(emails, cuad_entries):
    """转为 Alpaca 格式（保持英文以待翻译）。"""
    data = []

    for e in emails:
        data.append({
            "instruction": "You are an enterprise knowledge assistant. Answer based on the document provided. Cite the source.",
            "input": f"Document (Email from {e['sender']}):\nSubject: {e['subject']}\n{e['body']}",
            "output": f"Based on the email from {e['sender']} regarding '{e['subject']}', the key information extracted is: {e['body'][:250]} [Source: Email from {e['sender']}]",
        })

    for item in cuad_entries:
        data.append({
            "instruction": "You are an enterprise knowledge assistant. Answer based on the contract clause provided. Cite the specific clause.",
            "input": f"Contract Clause ({item['title']}):\n{item['context']}\n\nQuestion: {item['question']}",
            "output": f"According to the {item['title']} clause, {item['answer']}. [Source: {item['title']}]",
        })

    return data


def main():
    print("从全部 Enron + CUAD 提取训练数据")
    print("=" * 50)

    emails = extract_enron_all(per_person=6, max_total=900)
    cuad_entries = extract_cuad_all(max_questions=300)

    data = to_training_format(emails, cuad_entries)
    random.shuffle(data)

    os.makedirs(os.path.dirname(OUTPUT_EN), exist_ok=True)
    with open(OUTPUT_EN, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n输出: {OUTPUT_EN}")
    print(f"总计: {len(data)} 条 (Enron {len(emails)} + CUAD {len(cuad_entries)})")
    print(f"下一步: 翻译成中文训练数据")


if __name__ == "__main__":
    main()
