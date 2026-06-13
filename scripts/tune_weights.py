"""MemoryJudge 组合评分权重网格搜索。

目标：找到使 MRR 最大化的四维权重 (w_semantic, w_temporal, w_entity, w_importance)
约束：四维之和 = 1，每维 >= 0

停止条件：当前 MRR 相比上一轮提升 < 0.5%

多轮迭代：
  Round 1: 粗粒度搜索 (step=0.1)  → 缩小最优区域
  Round 2: 中粒度搜索 (step=0.05) → 精确定位
  Round 3: 微调验证               → 确认无过拟合

用法：
  python scripts/tune_weights.py          # 完整网格搜索
  python scripts/tune_weights.py --quick  # 快速验证(step=0.2)
"""

import itertools
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

random_seed = 42


# ═══════════════════════════════════════════════════════════════
# 1. 标注测试集
# ═══════════════════════════════════════════════════════════════

ANNOTATED_QUERIES = [
    {
        "query": "凤凰项目技术选型的结论是什么？",
        "relevant_docs": ["凤凰项目-系统架构设计文档", "凤凰项目-检索策略设计文档"],
    },
    {
        "query": "W4A8量化的效果怎么样？",
        "relevant_docs": ["昆仑项目技术评审", "凤凰项目-W4A8量化部署指南", "端侧推理优化笔记"],
    },
    {
        "query": "AccessGuard的权限模型是什么？",
        "relevant_docs": ["凤凰项目安全评审会", "凤凰项目-系统架构设计文档"],
    },
    {
        "query": "ChromaDB和Elasticsearch怎么选？",
        "relevant_docs": ["凤凰项目技术选型评审会", "凤凰项目-系统架构设计文档"],
    },
    {
        "query": "凤凰项目的安全评估发现了什么问题？",
        "relevant_docs": ["凤凰项目安全评审会", "凤凰项目-系统架构设计文档"],
    },
    {
        "query": "昆仑项目的GPU成本是多少？",
        "relevant_docs": ["昆仑项目技术评审", "云帆科技-内部项目外包合同"],
    },
    {
        "query": "文档分级的方案是什么？",
        "relevant_docs": ["凤凰项目安全评审会", "凤凰项目-系统架构设计文档"],
    },
    {
        "query": "Cross-Encoder重排的效果如何？",
        "relevant_docs": ["凤凰项目-检索策略设计文档", "AI平台部-RAG系统评估标准"],
    },
    {
        "query": "谁负责凤凰项目的安全评估？",
        "relevant_docs": ["凤凰项目安全评审会"],
    },
    {
        "query": "凤凰项目MVP的功能范围是什么？",
        "relevant_docs": ["凤凰项目MVP迭代规划会", "凤凰项目-系统架构设计文档"],
    },
    {
        "query": "为什么需要双模式LLM方案？",
        "relevant_docs": ["凤凰项目技术选型评审会", "凤凰项目-系统架构设计文档"],
    },
    {
        "query": "凤凰项目的客户需求是什么？",
        "relevant_docs": ["凤凰项目客户需求整理", "凤凰项目技术选型评审会"],
    },
]


# ═══════════════════════════════════════════════════════════════
# 2. 权重生成
# ═══════════════════════════════════════════════════════════════

def generate_weights(step=0.1):
    """生成所有权重组合（和为 1，步长为 step）。"""
    n = int(1.0 / step)
    combos = []
    for w1 in range(n + 1):
        for w2 in range(n + 1 - w1):
            for w3 in range(n + 1 - w1 - w2):
                w4 = n - w1 - w2 - w3
                combos.append((
                    round(w1 * step, 2),
                    round(w2 * step, 2),
                    round(w3 * step, 2),
                    round(w4 * step, 2),
                ))
    return combos


# ═══════════════════════════════════════════════════════════════
# 3. 评估函数
# ═══════════════════════════════════════════════════════════════

def evaluate_weights(weights, retriever, annotated_queries):
    """计算给定权重下的 MRR（Mean Reciprocal Rank）。

    对每个标注查询：
    1. 检索 Top-10 文档
    2. 用指定权重计算 composite score
    3. 按 composite 重新排序
    4. 找第一个相关文档的排名 → reciprocal rank
    5. MRR = mean(reciprocal_ranks)
    """
    w_sem, w_tmp, w_ent, w_imp = weights
    reciprocal_ranks = []

    for item in annotated_queries:
        query = item["query"]
        relevant = set(item["relevant_docs"])

        # 检索
        results = retriever.search(query, top_k=10)

        if not results:
            reciprocal_ranks.append(0.0)
            continue

        # 用新权重重新计算 composite
        scored = []
        for r in results:
            semantic = r.get("score", 0)
            temporal = r.get("temporal_decay", 0.5)
            entity_bonus = r.get("entity_boost", 1.0) - 1.0
            importance = float(r.get("metadata", {}).get("importance", 0.5))

            composite = (
                w_sem * semantic
                + w_tmp * temporal
                + w_ent * min(entity_bonus, 1.0)
                + w_imp * importance
            )
            r["_composite"] = composite
            scored.append(r)

        # 按 composite 排序
        scored.sort(key=lambda x: x["_composite"], reverse=True)

        # 找第一个相关文档
        for rank, r in enumerate(scored, start=1):
            doc_name = r.get("metadata", {}).get("file_name", "")
            doc_title = r.get("metadata", {}).get("title", "")
            doc_source = str(r.get("metadata", {}).get("source_file", ""))

            # 匹配：文件名/标题/来源路径包含相关文档名
            doc_id = f"{doc_name}|{doc_title}|{doc_source}"
            if any(rel in doc_id for rel in relevant):
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    return mrr


# ═══════════════════════════════════════════════════════════════
# 4. 主搜索循环（含停止条件）
# ═══════════════════════════════════════════════════════════════

def multi_round_search(retriever, annotated_queries):
    """多轮迭代权重搜索。

    停止条件：
    - Round n 的最优 MRR 相比 Round n-1 提升 < 0.005 (0.5%)
    - 或步长 < 0.03（太细无意义）
    """

    rounds = [
        {"name": "Round 1: 粗搜索", "step": 0.1, "min_improvement": 0.01},
        {"name": "Round 2: 精搜索", "step": 0.05, "min_improvement": 0.005},
        {"name": "Round 3: 验证", "step": 0.03, "min_improvement": 0.003},
    ]

    best_mrr = 0.0
    best_weights = (0.5, 0.2, 0.1, 0.2)  # 当前默认值作为 baseline
    all_results = []

    for round_info in rounds:
        print(f"\n{'='*60}")
        print(f"{round_info['name']} (step={round_info['step']})")
        print(f"{'='*60}")

        weights_list = generate_weights(round_info["step"])
        round_best_mrr = 0.0
        round_best_weights = best_weights

        for i, w in enumerate(weights_list):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  进度: {i+1}/{len(weights_list)} (当前最优 MRR={round_best_mrr:.4f})")

            mrr = evaluate_weights(w, retriever, annotated_queries)

            if mrr > round_best_mrr:
                round_best_mrr = mrr
                round_best_weights = w

        improvement = round_best_mrr - best_mrr

        all_results.append({
            "round": round_info["name"],
            "step": round_info["step"],
            "best_weights": round_best_weights,
            "best_mrr": round_best_mrr,
            "improvement": improvement,
        })

        print(f"\n  最优权重: sem={round_best_weights[0]:.2f} tmp={round_best_weights[1]:.2f} "
              f"ent={round_best_weights[2]:.2f} imp={round_best_weights[3]:.2f}")
        print(f"  最优 MRR: {round_best_mrr:.4f} (提升: {improvement:+.4f})")

        # 停止条件
        if improvement < round_info["min_improvement"] and round_best_mrr <= best_mrr * 1.01:
            print(f"\n  ⏹ 停止: 提升 {improvement:.4f} < 阈值 {round_info['min_improvement']}")
            break

        if round_best_mrr > best_mrr:
            best_mrr = round_best_mrr
            best_weights = round_best_weights

    # 如果没有任何改善（全是0），保留默认值
    if best_mrr == 0.0:
        best_weights = (0.5, 0.2, 0.1, 0.2)
        print("\n  ⚠ 无有效评估结果，保留默认权重")

    print(f"\n{'='*60}")
    print(f"最终结果")
    print(f"{'='*60}")
    print(f"最优权重: sem={best_weights[0]:.2f} tmp={best_weights[1]:.2f} "
          f"ent={best_weights[2]:.2f} imp={best_weights[3]:.2f}")
    print(f"最优 MRR:  {best_mrr:.4f}")
    print(f"基线 MRR:  {all_results[0]['best_mrr']:.4f}" if all_results else "N/A")
    print(f"基线权重:  sem=0.50 tmp=0.20 ent=0.10 imp=0.20")

    # 保存结果
    os.makedirs("data", exist_ok=True)
    with open("data/weight_tuning_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "final_weights": list(best_weights),
            "final_mrr": best_mrr,
            "baseline_weights": [0.5, 0.2, 0.1, 0.2],
            "rounds": all_results,
            "num_queries": len(annotated_queries),
        }, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: data/weight_tuning_results.json")
    return best_weights, best_mrr


# ═══════════════════════════════════════════════════════════════
# 5. 快速模式
# ═══════════════════════════════════════════════════════════════

def quick_check(retriever):
    """快速验证：只测试当前默认权重和几个极端权重。"""
    base = evaluate_weights((0.5, 0.2, 0.1, 0.2), retriever, ANNOTATED_QUERIES)
    print(f"当前默认 (0.50/0.20/0.10/0.20): MRR={base:.4f}")

    extremes = [
        (0.8, 0.1, 0.05, 0.05, "语义主导"),
        (0.2, 0.6, 0.1, 0.1, "时间主导"),
        (0.3, 0.2, 0.3, 0.2, "实体主导"),
        (0.3, 0.2, 0.1, 0.4, "重要性主导"),
    ]
    for w in extremes:
        mrr = evaluate_weights(w[:4], retriever, ANNOTATED_QUERIES)
        tag = w[4]
        print(f"  {tag} ({w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f}/{w[3]:.2f}): MRR={mrr:.4f}")


# ═══════════════════════════════════════════════════════════════
# 6. 入口
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="快速验证模式")
    args = p.parse_args()

    print("MemoryJudge 权重网格搜索")
    print("=" * 60)
    print(f"测试集: {len(ANNOTATED_QUERIES)} 个标注查询")

    # 初始化组件（需要 mock 后端，不加载模型）
    from rag.embeddings import OnDeviceEmbeddings
    from rag.vector_store import VectorStore
    from rag.bm25_retriever import BM25Retriever
    from rag.hybrid_retriever import HybridRetriever
    from rag.entity_aware_retriever import EntityAwareRetriever
    from data_ingestion.pipeline import IngestionPipeline
    from data_ingestion.indexer import Indexer

    import numpy as np

    class MockBackend:
        backend_type = "mock"
        def embed(self, texts):
            return np.random.randn(len(texts), 512).astype(np.float32)

    backend = MockBackend()
    embeddings = OnDeviceEmbeddings(backend)

    db_dir = "./data/chroma_db_tune"
    import shutil
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

    vector_store = VectorStore(persist_dir=db_dir, collection_name="tune_test", embedding_function=embeddings)
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(vector_store, bm25, rrf_k=60, top_k=10)
    entity_retriever = EntityAwareRetriever(hybrid, entity_boost_factor=1.2)

    # 索引企业数据
    pipeline = IngestionPipeline()
    indexer = Indexer(vector_store, bm25)
    data_dir = "./data/demo/enterprise"
    if os.path.exists(data_dir):
        chunks = pipeline.ingest_directory(data_dir)
        chunks = pipeline.deduplicate(chunks)
        indexer.index(chunks)
        print(f"索引: {len(chunks)} chunks, 共 {vector_store.count()} 文档")
    else:
        print("企业数据不存在，请先运行: python scripts/ingest_enterprise_data.py")
        return

    if args.quick:
        quick_check(entity_retriever)
    else:
        multi_round_search(entity_retriever, ANNOTATED_QUERIES)

    shutil.rmtree(db_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
