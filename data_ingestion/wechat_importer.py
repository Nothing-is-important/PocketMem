"""微信聊天记录导入器 —— v2.0。

将微信桌面版导出的 .txt 文件解析、分块并索引，
通过回调函数报告实时进度（供 SSE 流式传输）。

用法:
    from data_ingestion.wechat_importer import WechatImporter
    importer = WechatImporter(pipeline, indexer)
    for progress in importer.import_from_directory("./data/raw"):
        print(progress)  # {"file": "张三.txt", "stage": "parsing", "count": 45, ...}
"""

import time
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional

from data_ingestion.wechat_parser import parse_wechat_export, ChatMessage
from data_ingestion.chunker import DocumentChunk
from utils import get_logger

logger = get_logger("wechat_import")


class WechatImporter:
    """微信聊天记录导入器。

    封装 parse → chunk → index 全流程，
    通过 yield 报告每步进度。
    """

    def __init__(self, pipeline=None, indexer=None):
        """初始化导入器。

        Args:
            pipeline: IngestionPipeline 实例（可选，延迟加载）
            indexer: Indexer 实例（可选，延迟加载）
        """
        self._pipeline = pipeline
        self._indexer = indexer
        self._stats = {"files_found": 0, "files_imported": 0, "messages": 0, "chunks": 0}

    def discover_wechat_files(self, directory: str) -> List[Path]:
        """发现目录中的微信导出文件。

        通过检查文件内容（时间戳模式）而非仅依赖扩展名来识别微信格式。

        Args:
            directory: 要扫描的目录路径

        Returns:
            Path 对象列表（微信格式的 .txt 文件）
        """
        target = Path(directory)
        if not target.exists():
            return []

        wechat_files = []
        for txt_file in target.rglob("*.txt"):
            if self._is_wechat_format(txt_file):
                wechat_files.append(txt_file)

        self._stats["files_found"] = len(wechat_files)
        return sorted(wechat_files, key=lambda p: p.name)

    def _is_wechat_format(self, filepath: Path) -> bool:
        """检测文件是否为微信导出格式。

        前 10 行中至少 2 行匹配 "YYYY-MM-DD HH:MM:SS SenderName" 模式。
        """
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                lines = [f.readline() for _ in range(20)]
        except (OSError, UnicodeDecodeError):
            return False

        match_count = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 微信格式: "2026-03-15 14:30:22 SenderName" 或变体
            parts = line.split(" ", 3)
            if len(parts) >= 3:
                date_part = parts[0]
                time_part = parts[1]
                # 日期检查: 包含 -
                if "-" in date_part and len(date_part) >= 8:
                    # 时间检查: 包含 :
                    if ":" in time_part and len(time_part) >= 4:
                        match_count += 1
                        if match_count >= 2:
                            return True
        return False

    def import_from_directory(
        self,
        directory: str,
        progress_callback: Optional[Callable] = None,
    ) -> Generator[Dict, None, None]:
        """从目录导入所有微信导出文件。

        对每个文件执行 parse → chunk → index，
        通过 yield 报告每步进度。

        Yields:
            {"event": "discover", "files": [...]}
            {"event": "file_start", "file": "张三.txt", "size": 12345}
            {"event": "parse_done", "file": "张三.txt", "messages": 45}
            {"event": "index_done", "file": "张三.txt", "chunks": 12}
            {"event": "file_done", "file": "张三.txt", "status": "ok"}
            {"event": "file_error", "file": "张三.txt", "error": "..."}
            {"event": "complete", "stats": {...}}
        """
        # 发现文件
        wechat_files = self.discover_wechat_files(directory)
        yield {
            "event": "discover",
            "files": [f.name for f in wechat_files],
            "count": len(wechat_files),
        }

        if not wechat_files:
            yield {"event": "complete", "stats": dict(self._stats), "message": "未找到微信导出文件"}
            return

        # 逐个文件导入
        for filepath in wechat_files:
            yield {
                "event": "file_start",
                "file": filepath.name,
                "size": filepath.stat().st_size,
            }

            try:
                # Step 1: 解析
                messages = parse_wechat_export(str(filepath))
                if not messages:
                    yield {
                        "event": "file_done",
                        "file": filepath.name,
                        "status": "skipped",
                        "reason": "未解析到消息",
                    }
                    continue

                self._stats["messages"] += len(messages)
                yield {
                    "event": "parse_done",
                    "file": filepath.name,
                    "messages": len(messages),
                }

                # Step 2: 分块 + 索引
                from data_ingestion.pipeline import IngestionPipeline
                from data_ingestion.indexer import Indexer

                pipeline = self._pipeline or IngestionPipeline()
                indexer = self._indexer
                if indexer is None:
                    raise ValueError("indexer is required for import")

                # 使用管线处理
                chunks = pipeline.ingest(str(filepath))
                deduped = pipeline.deduplicate(chunks)
                indexed_count = indexer.index(deduped)

                self._stats["chunks"] += indexed_count
                self._stats["files_imported"] += 1

                yield {
                    "event": "index_done",
                    "file": filepath.name,
                    "chunks": indexed_count,
                }
                yield {
                    "event": "file_done",
                    "file": filepath.name,
                    "status": "ok",
                    "messages": len(messages),
                    "chunks": indexed_count,
                }

            except Exception as e:
                logger.error("Failed to import %s: %s", filepath.name, e)
                yield {
                    "event": "file_error",
                    "file": filepath.name,
                    "error": str(e),
                }

        yield {
            "event": "complete",
            "stats": dict(self._stats),
            "message": f"导入完成: {self._stats['files_imported']} 个文件, "
                       f"{self._stats['messages']} 条消息, {self._stats['chunks']} 个片段",
        }

    @property
    def stats(self) -> Dict:
        return dict(self._stats)
