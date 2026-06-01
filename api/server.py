"""FastAPI 服务 —— PocketMemory REST API。

端点：
- POST /ask           同步问答
- POST /ask/stream    SSE 流式问答
- GET  /memory/stats  索引统计
- GET  /health        健康检查
- POST /ingest        扫描并索引新数据源
- GET  /data/sources  列出已索引的数据源
"""

import json
import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from agent.state import create_initial_state
from .models import (
    AskRequest, AskResponse, ErrorResponse,
    IngestRequest, IngestResponse,
    MemoryStats, SourceInfo, SourceListResponse,
)


def _get_components(request: Request):
    """从 app.state 获取组件（替代全局变量）。"""
    return request.app.state


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期 —— 占位，实际初始化在 run_demo.py 中完成。"""
    yield


app = FastAPI(
    title="PocketMemory",
    description="端侧个人记忆助手 Agent API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 前端静态文件 ──
_frontend_dir = Path(__file__).parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")


@app.get("/")
async def index():
    """返回前端页面。"""
    index_path = _frontend_dir / "index.html"
    if index_path.exists():
        # 禁用缓存，确保前端修改立即生效
        response = FileResponse(index_path)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    return {"message": "PocketMemory API", "docs": "/docs"}


@app.get("/health")
async def health(request: Request):
    backend = getattr(request.app.state, "backend", None)
    backend_type = backend.backend_type if backend else "not_initialized"
    return {"status": "ok", "backend": backend_type}


@app.get("/memory/stats", response_model=MemoryStats)
async def memory_stats(request: Request):
    vector_store = getattr(request.app.state, "vector_store", None)
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    stats = vector_store.get_collection_stats()
    return MemoryStats(
        total_documents=stats["count"],
        collection_name=stats["name"],
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, request: Request):
    state = request.app.state
    agent_graph = getattr(state, "agent_graph", None)
    cache = getattr(state, "cache", None)

    if agent_graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    t0 = time.time()
    query = req.query.strip()

    # 缓存检查
    cache_hit = False
    if cache:
        cached = cache.lookup(query)
        if cached:
            return AskResponse(
                query=query,
                answer=cached["answer"],
                intent=cached.get("intent", ""),
                sources=cached.get("sources", []),
                latency_ms=(time.time() - t0) * 1000,
                cache_hit=True,
            )

    state_data = create_initial_state(query)
    result = agent_graph.invoke(state_data)

    answer = result.get("final_answer", "")
    intent = result.get("intent", "")
    sources = result.get("memory_context", [])

    latency = (time.time() - t0) * 1000

    if cache:
        cache.store(query, {
            "answer": answer,
            "intent": intent,
            "sources": [
                {"content": s.get("content", "")[:200], "score": s.get("composite_score", 0)}
                for s in sources[:5]
            ],
        })

    return AskResponse(
        query=query,
        answer=answer,
        intent=intent,
        sources=sources[:5],
        latency_ms=latency,
        cache_hit=False,
    )


@app.post("/ask/stream")
async def ask_stream(req: AskRequest, request: Request):
    agent_graph = getattr(request.app.state, "agent_graph", None)
    if agent_graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    async def event_generator() -> AsyncGenerator[dict, None]:
        query = req.query.strip()
        yield {"data": json.dumps({"event": "start", "data": query})}

        state_data = create_initial_state(
            query,
            conversation_history=req.conversation_history or [],
        )

        try:
            async for event in agent_graph.astream(state_data):
                node_name = list(event.keys())[0] if event else ""
                node_state = event.get(node_name, {})

                if node_name == "router":
                    intent = node_state.get("intent", "unknown")
                    latency_stats = node_state.get("latency_stats", {})
                    _log_event("router", f"意图={intent}", intent=intent)
                    yield {"data": json.dumps({"event": "router", "data": intent,
                        "latency_ms": latency_stats.get("router_ms", 0)})}
                    await asyncio.sleep(0.3)  # 让前端有时间渲染步骤动画
                elif node_name == "retrieve":
                    results = node_state.get("retrieval_results", [])
                    latency_stats = node_state.get("latency_stats", {})
                    _log_event("retrieve", f"检索到{len(results)}条", count=len(results))
                    yield {"data": json.dumps({"event": "retrieve", "data": {
                        "count": len(results),
                        "latency_ms": latency_stats.get("retrieval_ms", 0),
                    }})}
                    await asyncio.sleep(0.3)
                elif node_name == "judge":
                    ctx_count = len(node_state.get("memory_context", []))
                    sufficient = node_state.get("context_sufficient", False)
                    latency_stats = node_state.get("latency_stats", {})
                    _log_event("judge", f"相关{ctx_count}条, 充分={sufficient}",
                               relevant_count=ctx_count, sufficient=sufficient)
                    yield {"data": json.dumps({"event": "judge", "data": {
                        "relevant_count": ctx_count,
                        "sufficient": sufficient,
                        "latency_ms": latency_stats.get("judge_ms", 0),
                    }})}
                    await asyncio.sleep(0.3)
                elif node_name == "generate":
                    # 真正的 token-by-token 流式生成
                    from agent.generator import build_generator_prompt, GENERATOR_MAX_TOKENS
                    
                    prompt = build_generator_prompt(node_state)
                    latency_stats = node_state.get("latency_stats", {})
                    gen_t0 = time.time()
                    
                    # 使用 backend.generate_stream 逐 token 推送
                    backend = getattr(request.app.state, "backend", None)
                    if backend and hasattr(backend, 'generate_stream'):
                        for token_chunk in backend.generate_stream(prompt, max_tokens=GENERATOR_MAX_TOKENS):
                            if token_chunk:
                                yield {"data": json.dumps({
                                    "event": "generate_token",
                                    "data": token_chunk,
                                })}
                                await asyncio.sleep(0.01)
                    
                    gen_latency = (time.time() - gen_t0) * 1000
                    yield {"data": json.dumps({
                        "event": "generate",
                        "data": "",
                        "latency_ms": gen_latency,
                    })}
                    _log_event("generate", f"流式生成完成", latency=gen_latency)
        except Exception as e:
            yield {"data": json.dumps({"event": "error", "data": str(e)})}

        yield {"data": json.dumps({"event": "done", "data": ""})}

    return EventSourceResponse(event_generator())


# ═══════════════════════════════════════════════════════════
# 数据源管理端点
# ═══════════════════════════════════════════════════════════

@app.get("/data/sources", response_model=SourceListResponse)
async def list_sources(request: Request):
    """列出所有已索引的数据源。"""
    source_mgr = getattr(request.app.state, "source_manager", None)
    if source_mgr is None:
        raise HTTPException(status_code=503, detail="Source manager not initialized")

    stats = source_mgr.get_stats()
    return SourceListResponse(
        total_sources=stats["total_sources"],
        total_chunks=stats["total_chunks"],
        watch_dir=source_mgr.watch_dir(),
        sources=[SourceInfo(**s) for s in stats["sources"]],
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_data(req: IngestRequest, request: Request):
    """扫描数据目录并索引新发现的数据源。

    支持微信桌面版导出 TXT、Markdown 笔记、PDF 文档。
    已索引的文件会自动跳过（增量索引）。
    """
    source_mgr = getattr(request.app.state, "source_manager", None)
    pipeline = getattr(request.app.state, "pipeline", None)
    indexer = getattr(request.app.state, "indexer", None)

    if source_mgr is None or pipeline is None or indexer is None:
        raise HTTPException(
            status_code=503,
            detail="Data ingestion components not initialized. Run scripts/run_demo.py --serve first."
        )

    directory = req.directory if req.directory else None
    result = source_mgr.ingest_new(
        pipeline=pipeline,
        indexer=indexer,
        directory=directory,
    )
    return IngestResponse(**result)


@app.get("/data/watched")
async def watched_directory(request: Request):
    """返回监控目录路径——用户应该把微信导出文件放在这里。"""
    source_mgr = getattr(request.app.state, "source_manager", None)
    if source_mgr is None:
        raise HTTPException(status_code=503, detail="Source manager not initialized")
    return {
        "watch_dir": source_mgr.watch_dir(),
        "supported_formats": {
            ".txt": "微信桌面版导出 TXT（聊天记录右键→导出）",
            ".md": "Markdown 笔记文件",
            ".pdf": "PDF 文档",
        },
        "usage": "将文件放入 watch_dir 后，调用 POST /ingest 即可自动索引",
    }


# ═══════════════════════════════════════════════════════════
# 日志端点
# ═══════════════════════════════════════════════════════════

# 内存日志缓冲区（最近 200 条）
_log_buffer: list = []
_MAX_LOG_ENTRIES = 200


def _log_event(event_type: str, detail: str = "", **kwargs):
    """记录一条 Agent 执行事件到内存日志缓冲区。"""
    from datetime import datetime
    entry = {
        "time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
        "event": event_type,
        "detail": detail,
        **kwargs,
    }
    _log_buffer.append(entry)
    if len(_log_buffer) > _MAX_LOG_ENTRIES:
        _log_buffer.pop(0)


@app.get("/logs")
async def get_logs(limit: int = 50, event: str = ""):
    """获取最近的 Agent 执行日志。

    参数:
        limit: 返回条数（默认 50）
        event: 按事件类型过滤（router/retrieve/judge/generate/error）
    """
    logs = _log_buffer
    if event:
        logs = [l for l in logs if l["event"] == event]
    return {"total": len(_log_buffer), "logs": logs[-limit:]}


@app.delete("/logs")
async def clear_logs():
    """清空日志缓冲区。"""
    _log_buffer.clear()
    return {"status": "cleared"}


# ═══════════════════════════════════════════════════════════
# 微信状态端点
# ═══════════════════════════════════════════════════════════

@app.get("/wechat/status")
async def wechat_status():
    """检测微信运行状态和可导入数据。

    Returns:
        {
            "running": true,
            "wxid": "wxid_xxx",
            "data_dir": "C:/Users/.../WeChat Files/wxid_xxx",
            "msg_dbs": ["MSG0.db"],
            "can_import": true,
            "hint": "微信运行中 · wxid_xxx · 发现 1 个消息数据库 (52MB)"
        }
    """
    from data_ingestion.wechat_detector import get_wechat_status
    return get_wechat_status()


@app.post("/wechat/import")
async def wechat_import():
    """导入微信消息数据库（需要先解密）。

    当前状态：数据库解密功能正在开发中。
    微信 MSG0.db 使用 SQLCipher 4 加密，需要从进程内存提取密钥。
    """
    return {
        "status": "pending",
        "chunks": 0,
        "hint": "数据库解密功能开发中。需要 pycryptodome + pymem 从微信进程内存提取密钥。参考 .opencode/skills/wechat-db-decrypt.skill.md",
    }
