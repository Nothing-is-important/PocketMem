"""PocketMemory 演示脚本。"""

import sys
from pathlib import Path

# Windows GBK 终端 → UTF-8
sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))


def setup():
    """初始化所有组件。"""
    from config import get_settings

    settings = get_settings()
    print(f"设备: {settings.device}")
    print(f"Embedding 模型: {settings.embedding_model}")
    print(f"LLM 模型: {settings.llm_model}")

    # 1. 创建推理后端
    from backend import create_backend

    backend = create_backend(
        "local_simulate",
        embedding_model_name=settings.embedding_model,
        llm_model_name=settings.llm_model,
        device=settings.device,
        chunk_size=settings.chunk_size,
    )

    # 2. 创建向量库和检索器
    from rag.bm25_retriever import BM25Retriever
    from rag.embeddings import OnDeviceEmbeddings
    from rag.entity_aware_retriever import EntityAwareRetriever
    from rag.hybrid_retriever import HybridRetriever
    from rag.vector_store import VectorStore

    embeddings = OnDeviceEmbeddings(backend)
    vector_store = VectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
    )
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(
        vector_store, bm25,
        rrf_k=settings.rrf_k,
        top_k=settings.retrieval_top_k,
    )
    entity_retriever = EntityAwareRetriever(
        hybrid,
        entity_boost_factor=settings.entity_boost_factor,
    )

    # 3. 缓存
    from api.cache import TwoTierCache

    cache = TwoTierCache(
        embedding_fn=backend.embed,
        l1_ttl=settings.cache_l1_ttl,
    )

    # 4. 生成演示数据（如果还没有）
    demo_dir = Path(settings.demo_data_dir)
    if not demo_dir.exists() or not any(demo_dir.iterdir()):
        print("\n生成演示数据...")
        from scripts.ingest_demo_data import generate_demo_data

        generate_demo_data(settings.demo_data_dir)

    # 5. 索引数据
    print("\n索引演示数据...")
    from data_ingestion.indexer import Indexer
    from data_ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    indexer = Indexer(vector_store, bm25)

    all_chunks = pipeline.ingest_directory(settings.demo_data_dir)
    all_chunks = pipeline.deduplicate(all_chunks)
    indexed = indexer.index(all_chunks)

    print(f"索引完成: {indexed} 个新文档块 (总计 {vector_store.count()})")

    # 5.5 初始化数据源管理器
    from data_ingestion.source_manager import SourceManager

    source_mgr = SourceManager(
        data_dir=settings.data_dir,
        raw_subdir="raw",
        demo_subdir="demo",
    )
    # 标记演示数据为已索引（避免后续重复索引），并统计每个文件的 chunk 数
    demo_dir_path = Path(settings.demo_data_dir)
    if demo_dir_path.exists():
        # 按源文件统计 chunk 数
        file_chunk_counts = {}
        for chunk in all_chunks:
            src = chunk.metadata.get("source_file", "")
            file_chunk_counts[src] = file_chunk_counts.get(src, 0) + 1

        for f in demo_dir_path.rglob("*"):
            if f.suffix in (".txt", ".md", ".pdf"):
                fpath = str(f)
                count = file_chunk_counts.get(fpath, 0)
                source_mgr.mark_indexed(fpath, chunk_count=count)
    # 同时扫描 data/raw 目录（用户放置真实微信导出的地方）
    raw_sources = source_mgr.scan()
    if raw_sources:
        new_result = source_mgr.ingest_new(pipeline=pipeline, indexer=indexer)
        if new_result["new_files"] > 0:
            print(f"数据源: 发现 {new_result['new_files']} 个新文件，"
                  f"新增 {new_result['new_chunks']} 个文档块")
    print(f"数据源监控目录: {source_mgr.watch_dir()}")

    # 6. 构建 Agent 图
    from agent.graph import build_agent_graph

    agent_graph = build_agent_graph(
        backend=backend,
        entity_aware_retriever=entity_retriever,
        hybrid_retriever=hybrid,
    )

    # 6.5 注册 Hook：用户画像（自动记录查询 + 个性化注入）
    from agent.hooks import hooks
    from agent.user_profile import profile

    def _hook_inject_profile(state):
        """pre_retrieve hook: 检索前注入用户画像上下文"""
        query = state.get("query", "")
        context = profile.inject_context(query)
        if context:
            state["user_context"] = context
        return state

    def _hook_record_query(state):
        """post_generate hook: 生成后记录查询行为"""
        query = state.get("query", "")
        intent = state.get("intent", "")
        # 从检索结果中提取涉及的联系人
        contacts = []
        for item in state.get("memory_context", []):
            participants = item.get("metadata", {}).get("participants", [])
            if isinstance(participants, list):
                contacts.extend(participants)
        profile.record_query(query, intent, contacts)
        profile.save()
        return state

    hooks.register("pre_retrieve", _hook_inject_profile)
    hooks.register("post_generate", _hook_record_query)
    print("  Hook 已注册: pre_retrieve(inject_profile), post_generate(record_query)")

    # 7. 挂载组件到 FastAPI app.state
    from api.server import app as fastapi_app
    fastapi_app.state.agent_graph = agent_graph
    fastapi_app.state.cache = cache
    fastapi_app.state.vector_store = vector_store
    fastapi_app.state.backend = backend
    fastapi_app.state.source_manager = source_mgr
    fastapi_app.state.pipeline = pipeline
    fastapi_app.state.indexer = indexer

    return {
        "backend": backend,
        "agent_graph": agent_graph,
        "vector_store": vector_store,
        "cache": cache,
        "source_manager": source_mgr,
        "pipeline": pipeline,
        "indexer": indexer,
    }


def run_preset_queries(agent_graph):
    """运行预设演示查询。"""
    queries = [
        "张三推荐的那家火锅店叫什么？在哪里？",
        "关于团建，大家都讨论了什么方案？",
        "我的笔记里关于Transformer架构写了什么？",
        "端侧推理优化有哪些方法？",
        "李四面试拿了哪家公司的offer？",
    ]

    for query in queries:
        print(f"\n{'=' * 60}")
        print(f"Q: {query}")
        print("-" * 60)

        from agent.state import create_initial_state

        state = create_initial_state(query)
        result = agent_graph.invoke(state)

        print(f"A: {result.get('final_answer', '(无回答)')}")
        print(f"   意图: {result.get('intent', 'N/A')}")
        print(f"   检索片段数: {len(result.get('memory_context', []))}")
        print(f"   延迟: {result.get('latency_stats', {})}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PocketMemory 演示")
    parser.add_argument("--once", action="store_true", help="运行预设查询并退出")
    parser.add_argument("--serve", action="store_true", help="启动 FastAPI 服务")
    args = parser.parse_args()

    print("初始化 PocketMemory...")
    components = setup()

    if args.once:
        run_preset_queries(components["agent_graph"])
    elif args.serve:
        import uvicorn
        from config import get_settings

        settings = get_settings()
        print(f"\n启动服务: http://{settings.server_host}:{settings.server_port}")
        uvicorn.run(
            "api.server:app",
            host=settings.server_host,
            port=settings.server_port,
            reload=False,
        )
    else:
        # 交互模式
        print("\n交互模式（输入 'quit' 退出，输入 'stats' 查看统计）")
        agent_graph = components["agent_graph"]

        while True:
            try:
                query = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                break
            if query.lower() == "stats":
                print(f"索引文档数: {components['vector_store'].count()}")
                continue

            from agent.state import create_initial_state

            state = create_initial_state(query)
            result = agent_graph.invoke(state)

            print(f"\n{result.get('final_answer', '(无回答)')}")


if __name__ == "__main__":
    main()
