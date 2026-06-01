"""检索节点 —— 调用检索器获取文档块。

根据意图选择检索模式：
- memory_lookup → 时间感知 + 实体加权检索
- knowledge_lookup → 标准混合检索
"""

import time
from typing import Optional

from .state import AgentState


def create_retriever_node(
    entity_aware_retriever,
    hybrid_retriever,
):
    """创建检索节点工厂函数。

    Args:
        entity_aware_retriever: EntityAwareRetriever（用于记忆查询）
        hybrid_retriever: HybridRetriever（用于知识查询）

    Returns:
        retriever_node 函数
    """

    def retriever_node(state: AgentState) -> AgentState:
        query = state["query"]
        intent = state["intent"]

        t0 = time.time()

        if intent == "memory_lookup":
            results = entity_aware_retriever.search(query)
            # 记忆查询：应用时间衰减（30天半衰期）
            results = apply_temporal_decay(results)
            # 构建时间上下文（如 "2026-03-15 ~ 2026-04-20"）
            state["temporal_context"] = build_temporal_context(results)
        elif intent == "knowledge_lookup":
            results = hybrid_retriever.search(query)
        else:
            results = hybrid_retriever.search(query)

        state["retrieval_results"] = results
        state["latency_stats"]["retrieval_ms"] = (time.time() - t0) * 1000

        # 记录检索统计
        state["messages"].append({
            "role": "retriever",
            "content": f"Retrieved {len(results)} chunks",
        })

        return state

    return retriever_node


def apply_temporal_decay(
    results: list,
    half_life_days: float = 30.0,
    reference_time=None,
) -> list:
    """在检索结果上应用时间衰减。

    在 RRF 融合分数基础上乘以时间衰减权重。

    Args:
        results: 检索结果列表（已有 score 字段）
        half_life_days: 半衰期天数
        reference_time: 参考时间（默认当前时间）

    Returns:
        应用时间衰减后的结果列表（就地修改 score 和 score_raw 字段）
    """
    import math
    from datetime import datetime, timedelta

    ref = reference_time or datetime.now()
    lam = math.log(2) / half_life_days

    for item in results:
        timestamp_str = item.get("metadata", {}).get("timestamp", "")
        if not timestamp_str:
            continue

        try:
            ts = datetime.fromisoformat(timestamp_str)
            days = (ref - ts).total_seconds() / 86400.0
            decay = math.exp(-lam * max(days, 0))
        except (ValueError, TypeError):
            decay = 1.0

        # 保存原始分数，应用衰减
        item["score_raw"] = item.get("score", 0)
        item["score"] = item["score"] * decay
        item["temporal_decay"] = round(decay, 4)

    # 重新排序
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def build_temporal_context(results: list) -> str:
    """从检索结果构建时间上下文描述。"""
    if not results:
        return "无时间信息"

    timestamps = []
    for r in results:
        ts = r.get("metadata", {}).get("timestamp", "")
        if ts:
            try:
                from datetime import datetime
                timestamps.append(datetime.fromisoformat(ts))
            except ValueError:
                pass

    if not timestamps:
        return "无时间信息"

    from data_ingestion.time_utils import format_time_range
    return format_time_range(min(timestamps), max(timestamps))
