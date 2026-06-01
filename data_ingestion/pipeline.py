"""统一数据摄取编排器。

检测文件类型，路由到正确的加载器，输出统一的 DocumentChunk 列表。
"""

from pathlib import Path
from typing import List, Optional

from .chunker import ConversationChunker, DocumentChunk
from .markdown_loader import load_markdown
from .pdf_loader import load_pdf
from .wechat_parser import filter_text_messages, parse_wechat_export


SUPPORTED_EXTENSIONS = {
    ".txt": "wechat",
    ".md": "markdown",
    ".markdown": "markdown",
    ".pdf": "pdf",
}


class IngestionPipeline:
    """统一数据摄取管线。

    用法:
        pipeline = IngestionPipeline()
        chunks = pipeline.ingest("data/wechat_export.txt")
        # 或摄取整个目录
        chunks = pipeline.ingest_directory("data/raw/")
    """

    def __init__(self, chunker: Optional[ConversationChunker] = None):
        self.chunker = chunker or ConversationChunker()
        self._content_hashes: set = set()

    def ingest(self, filepath: str) -> List[DocumentChunk]:
        """摄取单个文件。

        Args:
            filepath: 文件路径

        Returns:
            DocumentChunk 列表
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"不支持的文件类型: {ext}。"
                f"支持: {list(SUPPORTED_EXTENSIONS.keys())}"
            )

        source_type = SUPPORTED_EXTENSIONS[ext]

        if source_type == "wechat":
            return self._ingest_wechat(filepath)
        elif source_type == "markdown":
            return load_markdown(filepath, self.chunker)
        elif source_type == "pdf":
            return load_pdf(filepath, self.chunker)
        else:
            raise ValueError(f"未知来源类型: {source_type}")

    def ingest_directory(self, dirpath: str) -> List[DocumentChunk]:
        """摄取目录中所有支持的文件。

        Args:
            dirpath: 目录路径

        Returns:
            DocumentChunk 列表
        """
        path = Path(dirpath)
        if not path.is_dir():
            raise NotADirectoryError(f"目录不存在: {dirpath}")

        all_chunks = []
        for ext in SUPPORTED_EXTENSIONS:
            for file_path in path.rglob(f"*{ext}"):
                try:
                    chunks = self.ingest(str(file_path))
                    all_chunks.extend(chunks)
                    print(f"  [OK] {file_path.name}: {len(chunks)} chunks")
                except Exception as e:
                    print(f"  ✗ {file_path.name}: {e}")

        return all_chunks

    def deduplicate(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """基于 chunk_id (MD5) 去重。"""
        seen = set()
        unique = []
        for chunk in chunks:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                unique.append(chunk)
        return unique

    def _ingest_wechat(self, filepath: str) -> List[DocumentChunk]:
        """摄取微信导出文件。"""
        chat_name = Path(filepath).stem
        messages = parse_wechat_export(filepath, chat_name=chat_name)
        text_messages = filter_text_messages(messages)
        print(f"  [{chat_name}] 解析 {len(messages)} 条消息，"
              f"其中 {len(text_messages)} 条文本消息")
        return self.chunker.chunk_messages(text_messages, source_file=filepath)
