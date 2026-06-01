"""PocketMemory 评估框架。

对比指标：
- retrieval recall@5 / MRR
- 答案相关性（LLM-as-Judge）
- 延迟（Embedding / Generate / 端到端）
- 有时间衰减 vs 无时间衰减
- 有实体加权 vs 无实体加权
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_test_queries(filepath: str = None) -> List[Dict]:
    """加载测试查询数据集。"""
    if filepath is None:
        filepath = Path(__file__).parent / "test_queries.json"

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(
    retriever,
    test_queries: List[Dict],
    top_k: int = 5,
) -> Dict:
    """评估检索性能。

    指标：
    - recall@k: 相关文档出现在 top-k 中的比例
    - MRR: Mean Reciprocal Rank
    """
    recalls = []
    reciprocal_ranks = []

    for item in test_queries:
        query = item["query"]
        relevant_ids = set(item.get("relevant_ids", []))

        if not relevant_ids:
            continue

        results = retriever.search(query, top_k=top_k)
        result_ids = set(r.get("id", r.get("content", "")[:16]) for r in results)

        # Recall@k
        hits = len(relevant_ids & result_ids)
        recall = hits / len(relevant_ids) if relevant_ids else 1.0
        recalls.append(recall)

        # MRR
        for rank, r in enumerate(results, start=1):
            rid = r.get("id", r.get("content", "")[:16])
            if rid in relevant_ids:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    return {
        "recall_at_k": sum(recalls) / len(recalls) if recalls else 0,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0,
        "num_queries": len(test_queries),
        "num_evaluated": len(recalls),
    }


def evaluate_answer_quality(
    backend,
    test_queries: List[Dict],
    agent_graph=None,
) -> List[Dict]:
    """使用 LLM-as-Judge 评估答案质量。

    对每个测试查询：
    1. 用 Agent 生成答案
    2. 用 LLM 对答案打分（1-5）
    """
    results = []

    for item in test_queries:
        query = item["query"]
        expected_keywords = item.get("expected_keywords", [])

        t0 = time.time()
        if agent_graph:
            from agent.state import create_initial_state
            state = create_initial_state(query)
            result = agent_graph.invoke(state)
            answer = result.get("final_answer", "")
            latency = (time.time() - t0) * 1000
        else:
            answer = backend.generate(f"回答以下问题：{query}", max_tokens=256)
            latency = (time.time() - t0) * 1000

        # LLM-as-Judge 打分
        score = _llm_judge_score(backend, query, answer, expected_keywords)

        results.append({
            "query": query,
            "answer": answer[:200],
            "expected_keywords": expected_keywords,
            "score": score,
            "latency_ms": latency,
        })

    return results


def _llm_judge_score(
    backend, query: str, answer: str, expected_keywords: List[str]
) -> int:
    """LLM 对答案质量打分（1-5 分制）。"""
    from config.prompts import prompts

    judge_prompt = prompts.get("benchmark_judge").format(
        query=query,
        answer=answer[:500],
        expected_keywords=", ".join(expected_keywords) if expected_keywords else "无特定要求",
    )

    response = backend.generate(judge_prompt, max_tokens=8).strip()
    try:
        return int(response[0])
    except (ValueError, IndexError):
        return 3


def run_benchmark(
    backend,
    agent_graph,
    retriever,
    test_queries_path: Optional[str] = None,
) -> Dict:
    """运行完整 Benchmark。

    Returns:
        {
            "retrieval": {...},
            "answer_quality": [...],
            "latency_breakdown": {...},
        }
    """
    queries = load_test_queries(test_queries_path)
    if not queries:
        return {"error": "No test queries loaded"}

    print(f"Running benchmark on {len(queries)} queries...")

    # 检索评估
    retrieval_metrics = evaluate_retrieval(retriever, queries)
    print(f"  Recall@{5}: {retrieval_metrics['recall_at_k']:.3f}")
    print(f"  MRR: {retrieval_metrics['mrr']:.3f}")

    # 答案质量评估
    quality_results = evaluate_answer_quality(backend, queries, agent_graph)
    avg_score = sum(r["score"] for r in quality_results) / len(quality_results)
    avg_latency = sum(r["latency_ms"] for r in quality_results) / len(quality_results)
    print(f"  Avg Answer Score: {avg_score:.1f}/5")
    print(f"  Avg Latency: {avg_latency:.0f}ms")

    return {
        "retrieval": retrieval_metrics,
        "answer_quality": quality_results,
        "avg_score": avg_score,
        "avg_latency_ms": avg_latency,
    }
