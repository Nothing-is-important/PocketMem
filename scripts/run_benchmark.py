"""TeamMind Benchmark 运行脚本。

用法:
    uv run python scripts/run_benchmark.py
"""
import os
import sys
import time

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

# 离线模式（国内无法访问 HuggingFace）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from config import get_settings

settings = get_settings()

print("=" * 60)
print("TeamMind Benchmark")
print("=" * 60)
print(f"设备: {settings.device}")
print(f"Embedding: {settings.embedding_model}")
print(f"LLM: {settings.llm_model}")
print(f"Chunk Size: {settings.chunk_size}")

# ── 初始化 ──
t_total_start = time.time()

print("\n[1/5] 加载推理后端...")
from backend import create_backend

backend = create_backend(
    "local_simulate",
    embedding_model_name=settings.embedding_model,
    llm_model_name=settings.llm_model,
    device=settings.device,
    chunk_size=settings.chunk_size,
)

print("\n[2/5] 构建检索引擎...")
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
hybrid = HybridRetriever(vector_store, bm25, rrf_k=settings.rrf_k, top_k=settings.retrieval_top_k)
entity_retriever = EntityAwareRetriever(hybrid, entity_boost_factor=settings.entity_boost_factor)

print(f"  VectorStore: {vector_store.count()} documents")

# 如果向量库为空，先索引
if vector_store.count() == 0:
    print("\n  索引演示数据...")
    from data_ingestion.indexer import Indexer
    from data_ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    indexer = Indexer(vector_store, bm25)
    all_chunks = pipeline.ingest_directory(settings.demo_data_dir)
    all_chunks = pipeline.deduplicate(all_chunks)
    indexed = indexer.index(all_chunks)
    print(f"  索引完成: {indexed} 个新文档块 (总计 {vector_store.count()})")

print("\n[3/5] 构建 Agent 图...")
from agent.graph import build_agent_graph

agent_graph = build_agent_graph(
    backend=backend,
    entity_aware_retriever=entity_retriever,
    hybrid_retriever=hybrid,
)

print("\n[4/5] 运行 Benchmark 查询...")
from eval.benchmark import load_test_queries

queries = load_test_queries()
print(f"  测试查询数: {len(queries)}")

# ── 延迟基准测试 ──
print("\n  --- 延迟基准 ---")
embed_t0 = time.time()
_ = backend.embed(["测试延迟基准"])
embed_latency = (time.time() - embed_t0) * 1000
print(f"  Embedding 延迟: {embed_latency:.1f}ms")

gen_t0 = time.time()
_ = backend.generate("1+1=? 请用中文回答", max_tokens=64)
gen_latency = (time.time() - gen_t0) * 1000
print(f"  生成延迟 (64 tokens): {gen_latency:.1f}ms")

# ── 端到端查询测试 ──
print("\n  --- 端到端查询 ---")
from agent.state import create_initial_state

results = []
for item in queries:
    query = item["query"]
    expected_keywords = item.get("expected_keywords", [])
    query_type = item.get("type", "unknown")

    t0 = time.time()
    state = create_initial_state(query)
    result = agent_graph.invoke(state)
    e2e_latency = (time.time() - t0) * 1000

    answer = result.get("final_answer", "")
    intent = result.get("intent", "unknown")
    ctx_count = len(result.get("memory_context", []))
    latency_stats = result.get("latency_stats", {})

    # 关键词命中检查
    keyword_hits = sum(1 for kw in expected_keywords if kw in answer)
    keyword_rate = keyword_hits / len(expected_keywords) if expected_keywords else 1.0

    results.append({
        "query": query,
        "type": query_type,
        "intent": intent,
        "context_count": ctx_count,
        "keyword_hit_rate": keyword_rate,
        "e2e_latency_ms": e2e_latency,
        "latency_breakdown": latency_stats,
    })

    status = "✓" if ctx_count > 0 else "✗"
    print(f"  {status} [{query_type}] {query[:30]}...")
    print(f"     意图={intent}, 检索={ctx_count}条, 关键词命中={keyword_rate:.0%}, "
          f"延迟={e2e_latency:.0f}ms")

# ── 汇总统计 ──
print("\n[5/5] 汇总统计")
print("=" * 60)

successful_queries = [r for r in results if r["context_count"] > 0]
avg_e2e = sum(r["e2e_latency_ms"] for r in results) / len(results)
avg_context = sum(r["context_count"] for r in results) / len(results)
avg_keyword = sum(r["keyword_hit_rate"] for r in results) / len(results)

print(f"查询成功率 (检索>0): {len(successful_queries)}/{len(results)}")
print(f"平均检索片段数: {avg_context:.1f}")
print(f"平均关键词命中率: {avg_keyword:.1%}")
print(f"平均端到端延迟: {avg_e2e:.0f}ms")
print(f"Embedding 延迟: {embed_latency:.1f}ms")
print(f"生成延迟 (64 tokens): {gen_latency:.1f}ms")
print(f"模型: {settings.llm_model}")
print(f"设备: {settings.device}")

t_total = time.time() - t_total_start
print(f"\n总耗时: {t_total:.1f}s")
print("=" * 60)
