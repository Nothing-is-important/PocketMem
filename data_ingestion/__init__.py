"""数据摄取管线 —— 邮件/Markdown/PDF 解析、分块、索引。"""

from .chunker import ConversationChunker, DocumentChunk
from .indexer import Indexer
from .pipeline import IngestionPipeline
from .time_utils import (
    compute_temporal_decay,
    days_since,
    extract_date_from_query,
    parse_relative_time,
)
from .txt_parser import ChatMessage, filter_text_messages, parse_text_export

__all__ = [
    "ConversationChunker",
    "DocumentChunk",
    "Indexer",
    "IngestionPipeline",
    "ChatMessage",
    "filter_text_messages",
    "parse_text_export",
    "compute_temporal_decay",
    "days_since",
    "extract_date_from_query",
    "parse_relative_time",
]
