"""API 请求/响应 Pydantic 模型。"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="用户查询")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None, description="多轮对话历史 [{'query': '...', 'answer': '...'}, ...]"
    )


class AskResponse(BaseModel):
    query: str
    answer: str
    intent: str = ""
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0
    cache_hit: bool = False


class MemoryStats(BaseModel):
    total_documents: int = 0
    collection_name: str = ""
    date_range: Optional[str] = None
    top_contacts: List[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class IngestRequest(BaseModel):
    directory: str = Field(default="", description="要扫描的目录路径（留空则使用 data/raw）")


class IngestResponse(BaseModel):
    new_files: int = 0
    new_chunks: int = 0
    skipped: int = 0
    total_indexed: int = 0


class SourceInfo(BaseModel):
    name: str
    type: str = "unknown"
    message_count: int = 0
    chunk_count: int = 0
    indexed_at: str = ""


class SourceListResponse(BaseModel):
    total_sources: int = 0
    total_chunks: int = 0
    watch_dir: str = ""
    sources: List[SourceInfo] = Field(default_factory=list)
