"""Mock 模式 —— 不加载模型，跑通全流程，找出所有 bug。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np

print("=" * 60)
print("Mock 模式全链路测试")
print("=" * 60)

# ── Step 1: Mock InferenceBackend ──
print("\n[1] 创建 Mock Backend...")

class MockBackend:
    backend_type = "mock"

    def embed(self, texts):
        return np.random.randn(len(texts), 512).astype(np.float32)

    def generate(self, prompt, max_tokens=512):
        return "这是模拟的回答。"

    def logits(self, prompt):
        return np.random.randn(1000).astype(np.float32)

    def get_last_latency(self):
        return {"embed_ms": 10, "generate_ms": 50}

backend = MockBackend()
print("  OK")

# ── Step 2: Embeddings + VectorStore + BM25 ──
print("\n[2] 创建 VectorStore + BM25...")

from rag.embeddings import OnDeviceEmbeddings
from rag.vector_store import VectorStore
from rag.bm25_retriever import BM25Retriever
from rag.hybrid_retriever import HybridRetriever
from rag.entity_aware_retriever import EntityAwareRetriever

import shutil, os
db_dir = "./data/chroma_db_test"
if os.path.exists(db_dir):
    shutil.rmtree(db_dir)

embeddings = OnDeviceEmbeddings(backend)
print("  OnDeviceEmbeddings OK, name:", embeddings.name())

vector_store = VectorStore(
    persist_dir=db_dir,
    collection_name="test_memory",
    embedding_function=embeddings,
)
print(f"  VectorStore OK, count: {vector_store.count()}")

bm25 = BM25Retriever()
print("  BM25 OK")

hybrid = HybridRetriever(vector_store, bm25, rrf_k=60, top_k=10)
print("  Hybrid OK")

entity_retriever = EntityAwareRetriever(hybrid, entity_boost_factor=1.2)
print("  EntityAware OK")

# ── Step 3: 缓存 ──
print("\n[3] 创建缓存...")

from api.cache import TwoTierCache
cache = TwoTierCache(embedding_fn=backend.embed)
print("  TwoTierCache OK")

# ── Step 4: 摄取数据 ──
print("\n[4] 摄取演示数据...")

from data_ingestion.pipeline import IngestionPipeline
from data_ingestion.indexer import Indexer

pipeline = IngestionPipeline()
indexer = Indexer(vector_store, bm25)

all_chunks = pipeline.ingest_directory("./data/demo")
print(f"  总 chunks: {len(all_chunks)}")

all_chunks = pipeline.deduplicate(all_chunks)
print(f"  去重后: {len(all_chunks)}")

indexed = indexer.index(all_chunks)
print(f"  索引完成: {indexed} new chunks")
print(f"  VectorStore count: {vector_store.count()}")

# ── Step 5: 检索测试 ──
print("\n[5] 检索测试...")

results = entity_retriever.search("凤凰项目技术选型的结论是什么？")
print(f"  实体感知检索: {len(results)} 条结果")
if results:
    r = results[0]
    print(f"    第一条: score={r['score']:.3f}, content={r['content'][:60]}...")

# ── Step 6: Agent 图 ──
print("\n[6] 构建 Agent 图...")

from agent.graph import build_agent_graph
from agent.state import create_initial_state

agent_graph = build_agent_graph(
    backend=backend,
    entity_aware_retriever=entity_retriever,
    hybrid_retriever=hybrid,
)
print("  Agent graph 编译成功")

# ── Step 7: Agent 推理测试 ──
print("\n[7] Agent 推理测试...")

for query in [
    "凤凰项目技术选型的结论是什么？",
    "我的笔记里关于Transformer写了什么？",
    "你好",
]:
    state = create_initial_state(query)
    result = agent_graph.invoke(state)
    answer = result.get("final_answer", "")
    intent = result.get("intent", "unknown")
    ctx_count = len(result.get("memory_context", []))
    print(f"  Q: {query[:30]}...")
    print(f"    意图: {intent}, 上下文: {ctx_count}条, 回答: {answer[:40]}...")

# ── Step 8: API 挂载 ──
print("\n[8] API 挂载...")

from api.server import app as fastapi_app
fastapi_app.state.agent_graph = agent_graph
fastapi_app.state.cache = cache
fastapi_app.state.vector_store = vector_store
fastapi_app.state.backend = backend
print("  app.state 挂载完成")

# ── Step 9: API 端点测试 ──
print("\n[9] API 端点测试...")

from fastapi.testclient import TestClient
client = TestClient(fastapi_app)

# Health
r = client.get("/health")
assert r.status_code == 200, f"Health failed: {r.status_code}"
print(f"  GET /health: {r.json()}")

# Stats
r = client.get("/memory/stats")
assert r.status_code == 200, f"Stats failed: {r.status_code}"
print(f"  GET /memory/stats: {r.json()}")

# Ask
r = client.post("/ask", json={"query": "凤凰项目技术选型的结论是什么？"})
assert r.status_code == 200, f"Ask failed: {r.status_code}"
data = r.json()
print(f"  POST /ask: intent={data['intent']}, answer={data['answer'][:40]}..., latency={data['latency_ms']:.0f}ms")

# Ask stream
r = client.post("/ask/stream", json={"query": "测试流式"})
assert r.status_code == 200, f"Stream failed: {r.status_code}"
events = 0
for line in r.text.split('\n'):
    if line.startswith('data:'):
        events += 1
print(f"  POST /ask/stream: {events} SSE events")

# Data sources - need SourceManager on app.state
from data_ingestion.source_manager import SourceManager
source_mgr = SourceManager(data_dir="./data")
fastapi_app.state.source_manager = source_mgr
fastapi_app.state.pipeline = pipeline
fastapi_app.state.indexer = indexer

# Data sources list
r = client.get("/data/sources")
assert r.status_code == 200, f"Sources failed: {r.status_code}"
data = r.json()
print(f"  GET /data/sources: {data['total_sources']} sources, {data['total_chunks']} chunks, watch={data['watch_dir']}")

# Watched dir
r = client.get("/data/watched")
assert r.status_code == 200, f"Watched failed: {r.status_code}"
print(f"  GET /data/watched: {r.json()['watch_dir']}")

# Ingest
r = client.post("/ingest", json={"directory": ""})
assert r.status_code == 200, f"Ingest failed: {r.status_code}"
data = r.json()
print(f"  POST /ingest: new_files={data['new_files']}, new_chunks={data['new_chunks']}, total={data['total_indexed']}")

# ── 清理 ──
print("\n[10] 清理...")
shutil.rmtree(db_dir, ignore_errors=True)
print("  Done")

print("\n" + "=" * 60)
print("全链路测试完成！所有 9 个步骤通过。")
print("=" * 60)
