"""数据源管理器 —— 自动发现、格式检测、增量索引。

支持的格式：
    - 企业邮件 TXT（From/To/Subject 格式）
    - Markdown 笔记文件
    - PDF 文档

用法:
    from data_ingestion.source_manager import SourceManager
    manager = SourceManager(data_dir="./data")
    sources = manager.scan()            # 扫描所有数据源
    result = manager.ingest_new()       # 索引新文件
    stats = manager.get_stats()         # 获取数据源统计
"""

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from utils import get_logger

logger = get_logger("sources")

# 支持的文件扩展名 → 类型映射
FORMAT_MAP = {
    ".txt": "mail",
    ".md": "markdown",
    ".markdown": "markdown",
    ".pdf": "pdf",
}


class SourceManager:
    """管理数据源的扫描、索引和统计。

    职责：
    - 扫描数据目录，发现支持的文件
    - 格式检测（邮件 TXT vs 普通 TXT）
    - 增量索引（只索引新文件或修改过的文件）
    - 数据源统计（哪个文件有多少条消息）
    """

    def __init__(
        self,
        data_dir: str = "./data",
        raw_subdir: str = "raw",
        demo_subdir: str = "demo",
    ):
        self._data_dir = Path(data_dir)
        self._raw_dir = self._data_dir / raw_subdir
        self._demo_dir = self._data_dir / demo_subdir
        self._indexed_files: Dict[str, dict] = {}  # content_hash → {filepath, indexed_at, chunk_count, ...}
        # 向后兼容：文件路径到哈希的映射（用于快速查找）
        self._path_to_hash: Dict[str, str] = {}
        # 确保目录存在
        self._raw_dir.mkdir(parents=True, exist_ok=True)

    def scan(self, directory: str = None) -> List[dict]:
        """扫描目录，返回所有支持的文件及其元数据。

        Returns:
            [{"path": "...", "name": "...", "type": "mail", "size": 12345, "preview": "..."}, ...]
        """
        target = Path(directory) if directory else self._raw_dir
        if not target.exists():
            return []

        results = []
        for ext, source_type in FORMAT_MAP.items():
            for file_path in target.rglob(f"*{ext}"):
                info = self._inspect_file(file_path, source_type)
                if info:
                    results.append(info)

        # 也扫描 demo 目录
        if self._demo_dir.exists() and (directory is None):
            for ext, source_type in FORMAT_MAP.items():
                for file_path in self._demo_dir.rglob(f"*{ext}"):
                    info = self._inspect_file(file_path, source_type)
                    if info:
                        info["is_demo"] = True
                        results.append(info)

        return sorted(results, key=lambda x: x["name"])

    def _inspect_file(self, file_path: Path, source_type: str) -> Optional[dict]:
        """检查单个文件的元数据。"""
        try:
            stat = file_path.stat()
            # 读前 500 字符作为预览
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                preview = f.read(500)

            # 格式验证
            if source_type == "mail":
                # 邮件格式由 pipeline 自动检测
                pass
            elif source_type == "markdown":
                if not preview.strip().startswith("#") and "##" not in preview[:200]:
                    source_type = "text"  # 可能是普通文本

            return {
                "path": str(file_path),
                "name": file_path.name,
                "type": source_type,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "preview": _truncate_preview(preview),
                "message_count": self._count_messages(preview, source_type),
                "indexed": str(file_path) in self._path_to_hash,
            }
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("Failed to inspect %s: %s", file_path, e)
            return None

    def _count_messages(self, preview: str, source_type: str) -> int:
        """估算消息条数。"""
        return 0

    def _content_hash(self, file_path: str) -> str:
        """计算文件内容的 SHA256 哈希（用于去重）。"""
        h = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
        except OSError:
            return ""
        return h.hexdigest()

    def ingest_new(
        self,
        pipeline=None,
        indexer=None,
        directory: str = None,
    ) -> dict:
        """增量索引——只处理新文件或未被索引的文件。

        Args:
            pipeline: IngestionPipeline 实例
            indexer: Indexer 实例
            directory: 要扫描的目录（默认 raw + demo）

        Returns:
            {"new_files": 3, "new_chunks": 45, "skipped": 2, "total_indexed": 195}
        """
        from data_ingestion.indexer import Indexer
        from data_ingestion.pipeline import IngestionPipeline

        if pipeline is None:
            pipeline = IngestionPipeline()
        if indexer is None:
            raise ValueError("indexer is required for ingestion")

        sources = self.scan(directory)
        new_files = 0
        new_chunks = 0
        skipped = 0

        for src in sources:
            # 内容哈希去重：相同内容的文件只索引一次
            file_hash = self._content_hash(src["path"])
            if not file_hash or file_hash in self._indexed_files:
                skipped += 1
                continue

            try:
                chunks = pipeline.ingest(src["path"])
                deduped = pipeline.deduplicate(chunks)
                indexed_count = indexer.index(deduped)

                if indexed_count > 0:
                    new_files += 1
                    new_chunks += indexed_count
                    self._indexed_files[file_hash] = {
                        "filepath": src["path"],
                        "filename": src["name"],
                        "indexed_at": datetime.now().isoformat(),
                        "chunk_count": indexed_count,
                        "message_count": src.get("message_count", 0),
                        "type": src["type"],
                    }
                    self._path_to_hash[src["path"]] = file_hash
                    print(f"  [OK] {src['name']}: {indexed_count} chunks")
                else:
                    skipped += 1
            except Exception as e:
                print(f"  [FAIL] {src['name']}: {e}")

        return {
            "new_files": new_files,
            "new_chunks": new_chunks,
            "skipped": skipped,
            "total_indexed": sum(
                info.get("chunk_count", 0)
                for info in self._indexed_files.values()
            ),
        }

    def get_stats(self) -> dict:
        """获取数据源统计信息。"""
        sources_info = []
        for file_hash, info in self._indexed_files.items():
            # 优先从 filepath 提取文件名（最可靠），其次用 filename 字段
            fp = info.get("filepath", "")
            name = Path(fp).name if fp else ""
            if not name:
                name = info.get("filename", "")
            if not name and self._path_to_hash:
                # 反向查找：通过 hash 找到原始路径
                for path, h in self._path_to_hash.items():
                    if h == file_hash:
                        name = Path(path).name
                        break
            if not name:
                # 安全回退：尝试从 key 本身提取文件名（mark_indexed 以文件路径为 key）
                name = Path(file_hash).name
            if not name:
                name = f"file_{file_hash[:8]}"
            sources_info.append({
                "name": name,
                "type": info.get("type", "unknown"),
                "message_count": info.get("message_count", 0),
                "chunk_count": info.get("chunk_count", 0),
                "indexed_at": info.get("indexed_at", ""),
            })

        return {
            "total_sources": len(sources_info),
            "total_chunks": sum(s["chunk_count"] for s in sources_info),
            "sources": sorted(sources_info, key=lambda x: x["name"]),
            "watch_dirs": {
                "raw": str(self._raw_dir),
                "demo": str(self._demo_dir) if self._demo_dir.exists() else None,
            },
        }

    def mark_indexed(self, filepath: str, chunk_count: int = 0, source_type: str = ""):
        """手动标记文件为已索引。"""
        self._indexed_files[filepath] = {
            "filepath": filepath,
            "filename": Path(filepath).name,
            "indexed_at": datetime.now().isoformat(),
            "chunk_count": chunk_count,
            "message_count": 0,
            "type": source_type,
        }

    def watch_dir(self) -> str:
        """返回用户应该放置数据文件的目录路径。"""
        return str(self._raw_dir)


def _truncate_preview(text: str, max_chars: int = 200) -> str:
    """截断预览文本。"""
    text = text.strip().replace("\n", " | ")
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text
