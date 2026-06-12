"""企业邮件解析器。

支持格式：
- 结构化 TXT（我们生成的演示数据）
- EML/MIME（标准邮件格式，待扩展）
- MSG（Outlook 格式，待扩展）

解析为标准 ChatMessage，与现有文本解析器保持一致的接口，
便于统一索引。
"""

import os
import re
from datetime import datetime
from typing import List, Optional

from .chunker import ConversationChunker, DocumentChunk
from .txt_parser import ChatMessage


def parse_mail_file(filepath: str) -> List[ChatMessage]:
    """解析邮件文件。返回结构化的消息列表。

    邮件格式：
        From: 张伟 <zhangwei@yunfan.com>
        To: 李娜 <lina@yunfan.com>, 王磊 <wanglei@yunfan.com>
        Date: 2026-03-20 14:30:00
        Subject: 凤凰项目技术选型方案
        Level: internal

        邮件正文...
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    headers, _, body = content.partition("\n\n")

    from_name = _parse_header(headers, "From", "unknown")
    to_str = _parse_header(headers, "To", "")
    subject = _parse_header(headers, "Subject", os.path.basename(filepath))
    date_str = _parse_header(headers, "Date", "")
    level = _parse_header(headers, "Level", "internal")

    timestamp = _parse_date(date_str) if date_str else datetime.now()

    # 清理邮件正文：去掉签名等
    body = _clean_mail_body(body)

    return [ChatMessage(
        timestamp=timestamp,
        sender=from_name,
        content=f"Subject: {subject}\n\n{body}",
        msg_type="text",
        chat_name=f"{from_name} ({subject[:30]}...)",
        chat_type="private",
    )]


def _parse_header(text: str, key: str, default: str) -> str:
    """从邮件头中解析特定字段。"""
    pattern = re.compile(rf"^{key}:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
    match = pattern.search(text)
    return match.group(1).strip() if match else default


def _parse_date(date_str: str) -> datetime:
    """解析日期字符串。"""
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%a, %d %b %Y %H:%M:%S %z",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return datetime.now()


def _clean_mail_body(body: str) -> str:
    """清理邮件正文：去掉签名、引用等。"""
    lines = body.strip().split("\n")
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        # 跳过引用行
        if stripped.startswith(">"):
            continue
        # 跳过常见签名分隔符之后的内容
        if stripped in ("--", "---", "——", "Best regards", "Regards", "Thanks"):
            break
        clean_lines.append(line)
    return "\n".join(clean_lines).strip()


def load_mail(
    filepath: str,
    chunker: Optional[ConversationChunker] = None,
) -> List[DocumentChunk]:
    """加载邮件文件并分块。"""
    chkr = chunker or ConversationChunker()

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # 从邮件头提取元数据
    meta = _extract_metadata(filepath, content)

    return chkr.chunk_text(content, metadata=meta)


def _extract_metadata(filepath: str, content: str) -> dict:
    """从邮件内容提取元数据用于索引。"""
    headers = content.split("\n\n")[0] if "\n\n" in content else ""

    return {
        "source_file": filepath,
        "source_type": "email",
        "file_name": os.path.basename(filepath),
        "from": _parse_header(headers, "From", ""),
        "to": _parse_header(headers, "To", ""),
        "subject": _parse_header(headers, "Subject", ""),
        "level": _parse_header(headers, "Level", "internal"),
        "timestamp": _parse_header(headers, "Date", ""),
    }
