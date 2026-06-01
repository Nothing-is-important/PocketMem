"""PDF 文档加载器。

使用 pdfplumber 提取文本，保留页码信息。
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .chunker import ConversationChunker, DocumentChunk


def load_pdf(
    filepath: str,
    chunker: Optional[ConversationChunker] = None,
) -> List[DocumentChunk]:
    """加载 PDF 文件并分块。

    Args:
        filepath: PDF 文件路径
        chunker: 分块器实例

    Returns:
        DocumentChunk 列表
    """
    import pdfplumber

    path = Path(filepath)
    chkr = chunker or ConversationChunker()
    file_mtime = datetime.fromtimestamp(path.stat().st_mtime)

    full_text_parts = []
    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                full_text_parts.append(f"[第{page_num}页]\n{text.strip()}")

        if not full_text_parts:
            print(f"[PDF Loader] 警告: {path.name} 未提取到文字（可能是扫描件）")
            return []

    full_text = "\n\n".join(full_text_parts)

    return chkr.chunk_text(full_text, metadata={
        "source_file": str(path),
        "source_type": "pdf",
        "file_name": path.name,
        "total_pages": len(full_text_parts),
        "file_mtime": file_mtime.isoformat(),
        "timestamp": file_mtime.isoformat(),
    })
