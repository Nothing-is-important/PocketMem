"""MemoryRelevance Judge —— 个人记忆相关性判断。

与 Evidence Judge 的区别：
- 所有个人数据受信任，不需要信源标注
- 基于组合评分公式而非多步 LLM 验证
- 减少 LLM 调用：~12 → ~2 per query

四步流程：
1. 组合评分 = 0.6×语义 + 0.3×时间 + 0.1×实体
2. 对话去重（内容哈希）
3. 充足性判断（≥2 唯一会话线程 或 ≥3 高置信度 chunk）
4. Reflect：不足时生成补充查询，最多 1 次
"""

import hashlib
import time
from typing import List

from .state import AgentState

# Prompt 从配置加载
from config.prompts import prompts


def create_memory_judge_node(
    backend,
    relevance_threshold: float = 0.3,
    max_reflect_iterations: int = 1,
):
    """创建 MemoryRelevance Judge 节点。

    Args:
        backend: InferenceBackend 实例
        relevance_threshold: 相关性阈值（组合评分低于此值将被丢弃）
        max_reflect_iterations: 最大补充搜索次数

    Returns:
        memory_judge_node 函数
    """

    def memory_judge_node(state: AgentState) -> AgentState:
        t0 = time.time()

        results = state["retrieval_results"]
        query = state["query"]

        if not results:
            state["context_sufficient"] = False
            state["memory_context"] = []
            return state

        # Step 1: 组合评分
        scored = _composite_scoring(results, state.get("extracted_entities", {}))

        # Step 2: 阈值过滤
        filtered = [r for r in scored if r["composite_score"] >= relevance_threshold]

        # Step 3: 去重
        deduped = _deduplicate_chunks(filtered)

        # Step 4: 充足性判断
        sufficient = _check_sufficiency(deduped)

        # Step 5: LLM 精排（仅在结果很多且模型足够快时）
        # 小模型（1.5B）做精排太慢（>20s），跳过以控制延迟
        # 组合评分（composite score）已经足够筛选
        # if len(deduped) > 3:
        #     deduped = _llm_rerank(backend, query, deduped)

        state["memory_context"] = deduped
        state["context_sufficient"] = sufficient
        state["latency_stats"]["judge_ms"] = (time.time() - t0) * 1000

        # Step 6: Reflect —— 不足时生成补充查询
        if not sufficient and state["reflect_count"] < max_reflect_iterations:
            reflect_query = _generate_reflect_query(
                backend, query, deduped
            )
            if reflect_query:
                state["reflect_queries"].append(reflect_query)
                state["reflect_count"] += 1
                # 组合查询用于补充检索
                state["query"] = f"{query} {reflect_query}"

        state["messages"].append({
            "role": "judge",
            "content": (
                f"MemoryJudge: {len(filtered)}/{len(results)} relevant, "
                f"{len(deduped)} unique, sufficient={sufficient}"
            ),
        })

        return state

    return memory_judge_node


def _composite_scoring(results: list, query_entities: dict) -> list:
    """Step 1: 计算组合评分。

    composite = 0.5×语义 + 0.2×时间 + 0.1×实体 + 0.2×重要性

    重要性权重 0.2 说明：
    - 包含日期/地点/电话的"信息型"消息比"好的收到"更有记忆价值
    - 多人讨论比私聊更有信息密度
    - 但语义相关性仍然是主导因素（0.5）
    """
    # 归一化 RRF 分数：RRF 原始值很小（max~0.033），需要归一化到 0-1
    if results:
        max_rrf = max(r.get("score", 0) for r in results)
    for item in results:
        rrf_score = item.get("score", 0)
        # 优先使用 vector_score（余弦相似度，天然 0-1），
        # 其次使用归一化后的 RRF 分数
        semantic = item.get("vector_score", 0)
        if semantic <= 0 and max_rrf > 0:
            semantic = rrf_score / max_rrf  # 归一化到 0-1
        temporal = item.get("temporal_decay", 0.5)
        entity_bonus = item.get("entity_boost", 1.0) - 1.0
        importance = item.get("metadata", {}).get("importance", 0.5)

        composite = (
            0.5 * semantic
            + 0.2 * temporal
            + 0.1 * min(entity_bonus, 1.0)
            + 0.2 * importance
        )
        item["composite_score"] = round(composite, 4)

    results.sort(key=lambda x: x["composite_score"], reverse=True)
    return results


def _deduplicate_chunks(results: list) -> list:
    """Step 2: 基于内容前 100 字符的 MD5 去重。"""
    seen = set()
    unique = []
    for item in results:
        content_hash = hashlib.md5(
            item.get("content", "")[:100].encode("utf-8")
        ).hexdigest()
        if content_hash not in seen:
            seen.add(content_hash)
            unique.append(item)
    return unique


def _check_sufficiency(results: list) -> bool:
    """Step 3: 充足性判断。

    至少 2 个唯一会话线程（session_idx）或至少 3 个高置信度 chunk。
    """
    if not results:
        return False

    # 检查唯一会话数
    sessions = set()
    for item in results:
        session = item.get("metadata", {}).get("session_idx", "")
        if session != "":
            sessions.add(session)

    if len(sessions) >= 2:
        return True

    # 检查高置信度 chunk 数（composite_score >= 0.5）
    high_conf = [r for r in results if r.get("composite_score", 0) >= 0.5]
    if len(high_conf) >= 3:
        return True

    return len(results) >= 3


def _llm_rerank(backend, query: str, results: list) -> list:
    """Step 4: 使用 LLM 对检索结果精排（当结果较多时）。"""
    if len(results) <= 3:
        return results

    # 构建精简上下文（只取前 8 个，避免 prompt 过长）
    top_results = results[:8]
    context_parts = []
    for i, item in enumerate(top_results):
        snippet = item.get("content", "")[:200]
        context_parts.append(f"[片段{i}] {snippet}...")

    prompt = prompts.get("judge_rerank").format(
        query=query,
        context="\n\n".join(context_parts),
    )

    response = backend.generate(prompt, max_tokens=128)

    # 解析最佳片段索引
    best_idx = _parse_best_index(response)
    if best_idx is not None and best_idx < len(top_results):
        # 将最佳片段排到第一
        best = top_results.pop(best_idx)
        top_results.insert(0, best)

    return top_results


def _generate_reflect_query(
    backend, original_query: str, results: list
) -> str:
    """生成补充搜索查询。"""
    if not results:
        return ""

    # 提取已有结果中的关键词
    existing_keywords = set()
    for item in results[:3]:
        content = item.get("content", "")
        # 简单提取：取内容中不常见的词
        words = content.split()
        existing_keywords.update(w for w in words if len(w) >= 2)

    prompt = prompts.get("judge_reflect").format(
        original_query=original_query,
        result_count=len(results),
    )

    reflect = backend.generate(prompt, max_tokens=32).strip()
    return reflect if len(reflect) > 0 else ""


def _parse_best_index(response: str) -> int:
    """从 LLM 响应中解析最佳片段索引。"""
    import re

    # 匹配 "最佳片段：X" 或 "最佳片段=X"
    match = re.search(r"最佳片段[：:=]\s*(\d+)", response)
    if match:
        return int(match.group(1))

    # 匹配最高分的片段
    scores = re.findall(r"片段(\d+)=\s*(\d+)分?", response)
    if scores:
        best = max(scores, key=lambda x: int(x[1]))
        return int(best[0])

    return None
