"""FastAPI 服务 —— TeamMind REST API。

端点：
- POST /ask           同步问答
- POST /ask/stream    SSE 流式问答
- GET  /memory/stats  索引统计
- GET  /health        健康检查
- POST /ingest        扫描并索引新数据源
- GET  /data/sources  列出已索引的数据源
"""

import json
import os
import re
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
    title="TeamMind",
    description="端侧企业知识助手 API",
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
    return {"message": "TeamMind API", "docs": "/docs"}


@app.get("/open-file")
async def open_file(path: str):
    """在文件管理器中打开文件（Windows / macOS / Linux）。"""
    import subprocess, sys
    try:
        if sys.platform == "win32":
            os.startfile(os.path.abspath(path))
        elif sys.platform == "darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/health")
async def health(request: Request):
    backend = getattr(request.app.state, "backend", None)
    backend_type = backend.backend_type if backend else "not_initialized"
    device = getattr(backend, "device", "unknown") if backend else "unknown"
    # 提取模型可读名称
    model_name = ""
    if backend:
        llm = getattr(backend, '_llm_model', None) or getattr(backend, '_local', None)
        if llm and hasattr(llm, 'config'):
            model_name = getattr(llm.config, '_name_or_path', '') or getattr(llm.config, 'model_type', '')
        elif llm and hasattr(getattr(llm, '_local', None), 'config'):
            model_name = getattr(llm._local.config, '_name_or_path', '')
    if not model_name and backend:
        model_name = backend_type
    # 友好名称映射
    friendly = {"local_simulate": "Qwen3", "vllm": "vLLM", "dual_mode": "Dual"}
    display = friendly.get(backend_type, backend_type)
    if "4B" in str(model_name): display = "Qwen3-4B"
    elif "8B" in str(model_name): display = "Qwen3-8B"
    elif "1.5B" in str(model_name) or "1.7B" in str(model_name): display = "Qwen3-1.7B"
    return {"status": "ok", "backend": backend_type, "model": display, "device": device}


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

        gen_state = None  # judge 后的状态，用于生成
        gen_t0 = 0.0

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
                    await asyncio.sleep(0.3)
                elif node_name == "retrieve":
                    results = node_state.get("retrieval_results", [])
                    latency_stats = node_state.get("latency_stats", {})
                    _log_event("retrieve", f"检索到{len(results)}条", count=len(results))
                    yield {"data": json.dumps({"event": "retrieve",
                        "latency_ms": latency_stats.get("retrieval_ms", 0),
                        "data": {"count": len(results)}})}
                    await asyncio.sleep(0.3)
                elif node_name == "judge":
                    ctx_count = len(node_state.get("memory_context", []))
                    sufficient = node_state.get("context_sufficient", False)
                    latency_stats = node_state.get("latency_stats", {})
                    _log_event("judge", f"相关{ctx_count}条, 充分={sufficient}",
                               relevant_count=ctx_count, sufficient=sufficient)
                    yield {"data": json.dumps({"event": "judge",
                        "latency_ms": latency_stats.get("judge_ms", 0),
                        "data": {"relevant_count": ctx_count, "sufficient": sufficient}})}

                    # 保存状态并退出 astream，后续由 SSE handler 直接生成
                    gen_state = node_state
                    break

            # ── astream 循环结束，进入生成阶段 ──
            if gen_state is not None:
                gen_t0 = time.time()
                backend = getattr(request.app.state, "backend", None)
                thinking_text = ""

                if backend and hasattr(backend, 'generate_with_thinking'):
                    from agent.generator import build_thinking_messages
                    import inspect
                    messages = build_thinking_messages(gen_state)

                    first_think = True
                    thinking_text = ""
                    answer_text = ""

                    gen_result = backend.generate_with_thinking(messages, max_tokens=2048)
                    # vLLM 后端返回 async generator，本地后端返回 generator
                    if inspect.isasyncgen(gen_result):
                        async for event_type, text in gen_result:
                            if event_type == "think":
                                if first_think:
                                    yield {"data": json.dumps({"event": "think_start", "data": ""})}
                                    first_think = False
                                thinking_text += text
                                yield {"data": json.dumps({"event": "think_token", "data": text})}
                            elif event_type == "answer":
                                answer_text += text
                                yield {"data": json.dumps({"event": "answer_token", "data": text})}
                            await asyncio.sleep(0.005)
                    else:
                        for event_type, text in gen_result:
                            if event_type == "think":
                                if first_think:
                                    yield {"data": json.dumps({"event": "think_start", "data": ""})}
                                    first_think = False
                                thinking_text += text
                                yield {"data": json.dumps({"event": "think_token", "data": text})}
                            elif event_type == "answer":
                                answer_text += text
                                yield {"data": json.dumps({"event": "answer_token", "data": text})}
                            await asyncio.sleep(0.005)

                    # 更新 state 用于 hook
                    gen_state["final_answer"] = answer_text.strip() or thinking_text.strip()
                    gen_state["_thinking"] = thinking_text.strip()
                else:
                    # 降级：标准生成
                    from agent.generator import build_generator_prompt
                    prompt = build_generator_prompt(gen_state)
                    gen_state["final_answer"] = backend.generate(prompt, max_tokens=2048)

                # 思考结束事件
                if thinking_text.strip():
                    yield {"data": json.dumps({"event": "think_end", "data": ""})}

                gen_latency = (time.time() - gen_t0) * 1000
                gen_state["latency_stats"]["generate_ms"] = gen_latency

                # 来源引用
                sources = gen_state.get("memory_context", [])
                if sources:
                    yield {"data": json.dumps({"event": "sources", "data": [
                        {"content": s.get("content", "")[:200],
                         "file": s.get("metadata", {}).get("source_file", ""),
                         "name": (s.get("metadata", {}).get("source_file", "") or "").replace("\\","/").split("/")[-1] or None}
                        for s in sources[:5]
                    ]})}

                yield {"data": json.dumps({"event": "generate", "data": "", "latency_ms": gen_latency})}

                # 手动触发 post_generate hook（用户画像记录）
                try:
                    from agent.hooks import hooks
                    hooks.run("post_generate", gen_state)
                except Exception:
                    pass

                _log_event("generate", f"生成完成", latency=gen_latency)
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

    支持企业邮件、Markdown 文档、PDF 文件。
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
    # 数据变更后清除 LLM 推荐缓存，下次请求退回模板算法
    invalidate_suggestion_cache()
    return IngestResponse(**result)


@app.get("/data/watched")
async def watched_directory(request: Request):
    """返回监控目录路径——企业数据文件放在这里。"""
    source_mgr = getattr(request.app.state, "source_manager", None)
    if source_mgr is None:
        raise HTTPException(status_code=503, detail="Source manager not initialized")
    return {
        "watch_dir": source_mgr.watch_dir(),
        "supported_formats": {
            ".txt": "企业邮件 TXT（From/To/Subject 格式）",
            ".md": "Markdown 笔记文件",
            ".pdf": "PDF 文档",
        },
        "usage": "将文件放入 watch_dir 后，调用 POST /ingest 即可自动索引",
    }


# ═══════════════════════════════════════════════════════════
# 推荐问题（混合方案：模板秒开 → LLM 后台升级）
# ═══════════════════════════════════════════════════════════

# 模块级缓存
_suggestion_cache: dict = {"suggestions": [], "source": "template"}
_refresh_in_progress: bool = False


@app.get("/data/suggestions")
async def data_suggestions(request: Request):
    """返回推荐问题（优先 LLM 缓存，降级模板算法）。

    混合策略：
    - 页面加载：模板算法（<5ms，秒开）
    - 首次查询后：异步触发 LLM 生成
    - 后续加载：直接读 LLM 缓存
    """
    # LLM 缓存命中 → 直接返回
    if _suggestion_cache["source"] == "llm" and _suggestion_cache["suggestions"]:
        return _suggestion_cache

    # 降级：模板算法生成
    vector_store = getattr(request.app.state, "vector_store", None)
    source_mgr = getattr(request.app.state, "source_manager", None)
    suggestions = _generate_template_suggestions(vector_store, source_mgr)

    # 更新模板缓存
    _suggestion_cache["suggestions"] = suggestions
    _suggestion_cache["source"] = "template"

    return {"suggestions": suggestions, "source": "template"}


@app.post("/data/suggestions/refresh")
async def refresh_suggestions(request: Request):
    """后台用 LLM 重新生成推荐问题（异步，立即返回）。

    前端应在用户首次查询后调用此端点（fire-and-forget），
    后续 GET /data/suggestions 将返回 LLM 生成的优质问题。
    """
    global _refresh_in_progress

    if _refresh_in_progress:
        return {"status": "already_refreshing"}

    backend = getattr(request.app.state, "backend", None)
    vector_store = getattr(request.app.state, "vector_store", None)

    if backend is None or vector_store is None or vector_store.count() == 0:
        return {"status": "no_data"}

    # 小模型（<3B）输出不可控，跳过 LLM 刷新，依赖增强后的模板算法
    if _is_small_model(backend):
        return {"status": "skipped", "reason": "model_too_small"}

    import threading

    def _run_llm_refresh():
        global _suggestion_cache, _refresh_in_progress
        try:
            suggestions = _generate_llm_suggestions(backend, vector_store)
            _suggestion_cache = {"suggestions": suggestions, "source": "llm"}
        except Exception as e:
            _log_event("suggestions", f"LLM refresh failed: {e}")
        finally:
            _refresh_in_progress = False

    _refresh_in_progress = True
    thread = threading.Thread(target=_run_llm_refresh, daemon=True)
    thread.start()
    return {"status": "refreshing"}


def invalidate_suggestion_cache():
    """数据变更时清除 LLM 缓存，下次请求退回模板算法。"""
    global _suggestion_cache
    _suggestion_cache = {"suggestions": [], "source": "template"}


def _is_small_model(backend) -> bool:
    """检测当前模型是否为小参数量模型（<3B），不适合做建议生成。"""
    model_name = getattr(backend, '_llm_model', None)
    if model_name is None:
        return True
    # 检查 HuggingFace config 中的参数量
    try:
        config = getattr(model_name, 'config', None) or getattr(backend._llm_model, 'config', None)
        if config:
            params = getattr(config, 'num_parameters', 0) or sum(
                p.numel() for p in backend._llm_model.parameters()
            )
            if params > 0 and params < 3_000_000_000:
                return True
    except Exception:
        pass
    # 回退：通过模型名判断
    name_lower = str(getattr(backend, '_llm_model', '')).lower()
    small_indicators = ['1.5b', '1b', '0.5b', '500m', 'tiny', 'small', 'mini']
    return any(ind in name_lower for ind in small_indicators)


def _generate_llm_suggestions(backend, vector_store) -> list:
    """用 LLM 生成自然语言推荐问题。

    Args:
        backend: InferenceBackend 实例（已加载 LLM）
        vector_store: VectorStore 实例

    Returns:
        ["问题1", "问题2", ...] — 最多 5 个
    """
    sample_docs = vector_store.get_sample_documents(n=20)
    if not sample_docs:
        return _generate_template_suggestions(vector_store, None)

    # 拼接样本文档（截断避免 prompt 过长）
    doc_text = "\n---\n".join(doc[:200] for doc in sample_docs[:15])
    if len(doc_text) > 3000:
        doc_text = doc_text[:3000] + "\n..."

    prompt = f"""你是一个智能记忆助手。根据以下用户的记忆片段，生成 5 个简短、自然的推荐搜索问题。
要求：
- 每个问题不超过 15 个字
- 问题要多样化，覆盖不同人物和话题
- 用口语化中文，像真人会问的问题
- 只输出问题本身，每行一个，不要编号、不要解释

记忆片段：
{doc_text}

推荐问题："""

    try:
        raw = backend.generate(prompt, max_tokens=150)
        lines = [line.strip() for line in raw.strip().split("\n") if line.strip()]
        # 清洗：去掉可能的编号前缀
        suggestions = []
        for line in lines:
            cleaned = line.lstrip("0123456789.、)） ").strip()
            if cleaned and len(cleaned) >= 4:
                suggestions.append(cleaned)
        if suggestions:
            return suggestions[:5]
    except Exception:
        pass

    # LLM 失败时降级到模板
    return _generate_template_suggestions(vector_store, None)


def _generate_template_suggestions(vector_store, source_mgr) -> list:
    """基于已索引数据用模板算法动态生成推荐问题。"""
    from collections import Counter
    from rag.entity_extractor import extract_people, extract_topics

    people_names: list = []
    topics: list = []

    # 1. 从向量库采样文档，提取实体
    if vector_store is not None and vector_store.count() > 0:
        sample_docs = vector_store.get_sample_documents(n=30)
        all_text = "\n".join(sample_docs)
        if all_text.strip():
            raw_people = extract_people(all_text)
            raw_topics = extract_topics(all_text, top_n=8)
            # 过滤噪声实体
            people_names = [_clean_entity(p, "person") for p in raw_people]
            people_names = [p for p in people_names if p]
            topics = [_clean_entity(t, "topic") for t in raw_topics]
            topics = [t for t in topics if t]

    # 2. 从用户画像获取高频词作为补充
    try:
        from agent.user_profile import profile
        freq_terms = profile._data.get("query_stats", {}).get("frequent_terms", {})
        if freq_terms:
            top_terms = sorted(freq_terms.items(), key=lambda x: x[1], reverse=True)[:5]
            for term, count in top_terms:
                cleaned = _clean_entity(term, "topic")
                if count >= 2 and cleaned and cleaned not in topics:
                    topics.append(cleaned)
    except Exception:
        pass

    # 3. 模板生成
    person_templates = [
        "{name}负责什么工作？",
        "{name}最近的发言是什么？",
        "{name}参与了哪些决策？",
    ]
    topic_templates = [
        "{topic}的方案是什么？",
        "{topic}的结论是什么？",
        "{topic}的技术选型有哪些？",
    ]
    generic_defaults = [
        "凤凰项目技术选型的结论是什么？",
        "W4A8量化的效果怎么样？",
        "最近有哪些重要的项目决策？",
        "AccessGuard是什么？怎么设计的？",
        "文档分级的安全方案是什么？",
    ]

    suggestions = []

    # 用人名生成问题
    name_counter = Counter(people_names)
    for name, _ in name_counter.most_common(3):
        tmpl = person_templates[len(suggestions) % len(person_templates)]
        suggestions.append(tmpl.format(name=name))

    # 用主题词生成问题
    for topic in topics:
        if len(suggestions) >= 5:
            break
        if topic not in {s.replace("{name}", "").replace("{topic}", "") for s in suggestions}:
            tmpl = topic_templates[len(suggestions) % len(topic_templates)]
            suggestions.append(tmpl.format(topic=topic))

    # 兜底：用通用默认问题填满
    while len(suggestions) < 5:
        fallback = generic_defaults[len(suggestions) % len(generic_defaults)]
        if fallback not in suggestions:
            suggestions.append(fallback)

    return suggestions[:5]


def _clean_entity(text: str, entity_type: str) -> str:
    """清理和验证提取的实体，过滤噪声。

    Args:
        text: 原始实体文本
        entity_type: "person" 或 "topic"

    Returns:
        清洗后的实体文本，无效实体返回空字符串
    """
    if not text or len(text) < 2:
        return ""

    # 纯数字/日期碎片 → 过滤
    if text.isdigit():
        return ""
    if len(text) <= 3 and all(c.isdigit() or c in "-/:." for c in text):
        return ""

    # 人名特定过滤
    if entity_type == "person":
        # 必须全部是中文常见字符（人名不应含数字、英文、标点）
        if not all('\u4e00' <= c <= '\u9fff' for c in text):
            return ""
        # 长度检查：中文人名通常 2-3 字
        if len(text) > 4:
            return ""
        # 去噪词
        _person_noise = {
            "上次", "这次", "下次", "今天", "明天", "昨天", "现在", "之前", "之后",
            "可以", "可能", "应该", "已经", "还是", "不过", "但是", "因为", "所以",
            "这个", "那个", "什么", "怎么", "哪里", "哪个",
            "一个", "一些", "一下", "一起", "一定",
            "方法", "方向", "方案", "方式", "地方",
            "我们", "他们", "你们", "不是", "不会", "没有",
        }
        if text in _person_noise:
            return ""

    # 主题词过滤
    if entity_type == "topic":
        # 纯数字/符号/英文短串 → 过滤
        if len(text) <= 2 and not all('\u4e00' <= c <= '\u9fff' for c in text):
            return ""
        # 过滤明显的时间碎片
        if re.match(r'^\d{2,4}[-/年]?\d{0,2}[-/月]?\d{0,2}[日号]?$', text):
            return ""

    return text.strip()


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


