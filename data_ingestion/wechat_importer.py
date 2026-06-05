# WeChat chat history importer v2.0 + database decryption.
#
# Supports two import modes:
# 1. TXT export files (.txt files in data/raw/)
# 2. Encrypted database (extract key from WeChat process, decrypt MSG*.db)

import os
import time
from pathlib import Path
from typing import Dict, Generator, List, Optional

from utils import get_logger

logger = get_logger("wechat_import")


class WechatImporter:
    """WeChat chat history importer.

    Orchestrates parse -> chunk -> index pipeline,
    reports progress via generator yield for SSE streaming.
    """

    def __init__(self, pipeline=None, indexer=None):
        self._pipeline = pipeline
        self._indexer = indexer
        self._stats = {"files_found": 0, "files_imported": 0, "messages": 0, "chunks": 0}

    def discover_wechat_files(self, directory: str) -> List[Path]:
        """Discover WeChat export .txt files in a directory.

        Identifies WeChat format by timestamp pattern in file content,
        not just by file extension.
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
        """Check if file is WeChat export format.

        At least 2 of first 20 lines must match 'YYYY-MM-DD HH:MM:SS Name' pattern.
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
            parts = line.split(" ", 3)
            if len(parts) >= 3:
                date_part = parts[0]
                time_part = parts[1]
                if "-" in date_part and len(date_part) >= 8:
                    if ":" in time_part and len(time_part) >= 4:
                        match_count += 1
                        if match_count >= 2:
                            return True
        return False

    def import_from_directory(
        self,
        directory: str,
        progress_callback: Optional[callable] = None,
    ) -> Generator[Dict, None, None]:
        """Import all WeChat export files from a directory.

        Yields progress events for SSE streaming:
        {"event": "discover", "files": [...], "count": N}
        {"event": "file_start", "file": "...", "size": N}
        {"event": "parse_done", "file": "...", "messages": N}
        {"event": "index_done", "file": "...", "chunks": N}
        {"event": "file_done", "file": "...", "status": "ok"}
        {"event": "file_error", "file": "...", "error": "..."}
        {"event": "complete", "stats": {...}, "message": "..."}
        """
        wechat_files = self.discover_wechat_files(directory)
        yield {
            "event": "discover",
            "files": [f.name for f in wechat_files],
            "count": len(wechat_files),
        }

        if not wechat_files:
            yield {"event": "complete", "stats": dict(self._stats), "message": "No WeChat export files found"}
            return

        for filepath in wechat_files:
            yield {
                "event": "file_start",
                "file": filepath.name,
                "size": filepath.stat().st_size,
            }

            try:
                from data_ingestion.wechat_parser import parse_wechat_export

                messages = parse_wechat_export(str(filepath))
                if not messages:
                    yield {"event": "file_done", "file": filepath.name, "status": "skipped", "reason": "No messages parsed"}
                    continue

                self._stats["messages"] += len(messages)
                yield {"event": "parse_done", "file": filepath.name, "messages": len(messages)}

                from data_ingestion.pipeline import IngestionPipeline
                from data_ingestion.indexer import Indexer

                pipeline = self._pipeline or IngestionPipeline()
                indexer = self._indexer
                if indexer is None:
                    yield {"event": "file_error", "file": filepath.name, "error": "Indexer not initialized"}
                    continue

                chunks = pipeline.ingest(str(filepath))
                deduped = pipeline.deduplicate(chunks)
                indexed_count = indexer.index(deduped)

                self._stats["chunks"] += indexed_count
                self._stats["files_imported"] += 1

                yield {"event": "index_done", "file": filepath.name, "chunks": indexed_count}
                yield {"event": "file_done", "file": filepath.name, "status": "ok",
                       "messages": len(messages), "chunks": indexed_count}

            except Exception as e:
                logger.error("Failed to import %s: %s", filepath.name, e)
                yield {"event": "file_error", "file": filepath.name, "error": str(e)}

        yield {
            "event": "complete",
            "stats": dict(self._stats),
            "message": f"Import done: {self._stats['files_imported']} files, "
                       f"{self._stats['messages']} messages, {self._stats['chunks']} chunks",
        }

    def import_from_database(
        self,
        output_dir: str,
    ) -> Generator[Dict, None, None]:
        """Import messages from encrypted WeChat database.

        Requires WeChat to be logged in and admin privileges.
        Flow: check env -> extract key -> decrypt DB -> export TXT -> index.
        """
        import sys

        # 诊断信息
        diagnostics = []

        # 检查微信是否运行
        from data_ingestion.wechat_detector import _find_wechat_process
        wechat_running = _find_wechat_process() is not None
        if not wechat_running:
            diagnostics.append("微信未运行，请先登录微信")
        else:
            diagnostics.append("微信已运行")

        # 检查管理员权限
        import ctypes
        try:
            is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
        except Exception:
            is_admin = sys.platform != "win32"
        if not is_admin:
            diagnostics.append("未以管理员权限运行（密钥提取需要）")
            diagnostics.append("请右键 -> 以管理员身份运行终端")
        else:
            diagnostics.append("管理员权限: OK")

        yield {
            "event": "db_diagnostics",
            "diagnostics": diagnostics,
            "wechat_running": wechat_running,
            "is_admin": is_admin,
        }

        # 必须微信运行
        if not wechat_running:
            yield {
                "event": "db_error",
                "error": "微信未运行。请先登录微信，然后重试。",
                "hint": "或者将微信导出的 .txt 文件放到 data/raw/ 后点击「扫描」",
                "diagnostics": diagnostics,
            }
            return

        yield {"event": "db_discover", "stage": "extracting_key"}

        # Step 1: Extract encryption key
        from data_ingestion.wechat_key import get_wechat_key
        key = get_wechat_key()

        if not key:
            # 根据情况给出具体建议
            if not is_admin:
                error_msg = "密钥提取需要管理员权限。请右键「以管理员身份运行」终端后重试。"
            else:
                error_msg = "无法提取微信密钥。请确保微信已登录且为最新版本。"
            yield {
                "event": "db_error",
                "error": error_msg,
                "hint": "或者将微信导出的 .txt 文件放到 data/raw/ 后点击「扫描」",
                "diagnostics": diagnostics,
            }
            return

        yield {"event": "db_key_ok", "stage": "decrypting"}

        # Step 2: Find database files
        from data_ingestion.wechat_detector import _find_wechat_data_dir
        wxid, data_dir = _find_wechat_data_dir()

        if not data_dir:
            yield {"event": "db_error", "error": "WeChat data directory not found"}
            return

        msg_dir = data_dir / "Msg" / "Multi"
        if not msg_dir.exists():
            msg_dir = data_dir / "Msg"

        db_files = []
        if msg_dir.exists():
            db_files = sorted(msg_dir.glob("MSG*.db"))

        if not db_files:
            yield {"event": "db_error", "error": f"No message databases found in {msg_dir}"}
            return

        yield {"event": "db_discover", "files": [f.name for f in db_files], "count": len(db_files)}

        # Step 3: Decrypt, export, index each database
        from data_ingestion.wechat_decryptor import decrypt_database, read_messages, export_to_text

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        db_count = 0

        for db_file in db_files:
            db_name = db_file.stem
            yield {"event": "file_start", "file": db_file.name, "stage": "decrypting"}

            decrypted_path = decrypt_database(str(db_file), key)
            if not decrypted_path:
                yield {"event": "file_error", "file": db_file.name, "error": "Decryption failed"}
                continue

            messages = read_messages(decrypted_path)
            if not messages:
                yield {"event": "file_done", "file": db_file.name, "status": "skipped", "reason": "No text messages"}
                try:
                    os.remove(decrypted_path)
                except OSError:
                    pass
                continue

            yield {"event": "parse_done", "file": db_file.name, "messages": len(messages)}

            txt_path = output_path / f"{db_name}.txt"
            export_to_text(messages, str(txt_path), chat_name=wxid or db_name)

            from data_ingestion.pipeline import IngestionPipeline
            from data_ingestion.indexer import Indexer

            pipeline = self._pipeline or IngestionPipeline()
            indexer = self._indexer
            if indexer is None:
                yield {"event": "file_error", "file": db_file.name, "error": "Indexer not initialized"}
                continue

            chunks = pipeline.ingest(str(txt_path))
            deduped = pipeline.deduplicate(chunks)
            indexed_count = indexer.index(deduped)

            self._stats["chunks"] += indexed_count
            self._stats["messages"] += len(messages)
            self._stats["files_imported"] += 1
            db_count += 1

            yield {"event": "index_done", "file": db_file.name, "chunks": indexed_count}
            yield {"event": "file_done", "file": db_file.name, "status": "ok",
                   "messages": len(messages), "chunks": indexed_count}

            try:
                os.remove(decrypted_path)
            except OSError:
                pass

        yield {
            "event": "complete",
            "stats": dict(self._stats),
            "message": f"DB import done: {db_count} databases, "
                       f"{self._stats['messages']} messages, {self._stats['chunks']} chunks",
        }

    @property
    def stats(self) -> Dict:
        return dict(self._stats)
