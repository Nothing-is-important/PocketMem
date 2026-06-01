"""Markdown 文档加载器。

按章节标题分块，保留代码块格式。
"""

import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .chunker import ConversationChunker, DocumentChunk


def load_markdown(
    filepath: str,
    chunker: Optional[ConversationChunker] = None,
) -> List[DocumentChunk]:
    """加载 Markdown 文件并分块。

    Args:
        filepath: Markdown 文件路径
        chunker: 分块器实例，不提供时使用默认配置

    Returns:
        DocumentChunk 列表
    """
    path = Path(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chkr = chunker or ConversationChunker()
    file_mtime = datetime.fromtimestamp(path.stat().st_mtime)

    # 提取标题作为文档级元数据
    title = _extract_title(text) or path.stem

    return chkr.chunk_text(text, metadata={
        "source_file": str(path),
        "source_type": "markdown",
        "file_name": path.name,
        "title": title,
        "file_mtime": file_mtime.isoformat(),
        "timestamp": file_mtime.isoformat(),
    })


def _extract_title(text: str) -> Optional[str]:
    """提取 Markdown 的一级标题。"""
    match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None
