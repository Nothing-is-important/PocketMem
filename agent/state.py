"""Agent 状态定义 —— LangGraph 图的状态类型。"""

from typing import Any, Dict, List, TypedDict


class AgentState(TypedDict):
    """LangGraph Agent 全局状态。

    注意：messages 使用普通 list（非 add_messages reducer），
    因为我们的节点追加的是自定义消息格式 {"role": "router", ...}，
    不是 LangChain 的标准 Message 对象。
    """

    messages: list
    query: str
    intent: str
    retrieval_results: List[Dict[str, Any]]
    memory_context: List[Dict[str, Any]]
    context_sufficient: bool
    extracted_entities: Dict[str, List[str]]
    temporal_context: str
    reflect_queries: List[str]
    reflect_count: int
    final_answer: str
    latency_stats: Dict[str, float]
    # 多轮对话支持
    conversation_history: List[Dict[str, str]]  # [{"query": "...", "answer": "..."}, ...]
    user_context: str  # 用户画像上下文，注入 generator prompt


def create_initial_state(query: str, conversation_history: list = None) -> AgentState:
    """创建初始状态。"""
    return AgentState(
        messages=[],
        query=query,
        intent="",
        retrieval_results=[],
        memory_context=[],
        context_sufficient=False,
        extracted_entities={},
        temporal_context="",
        reflect_queries=[],
        reflect_count=0,
        final_answer="",
        latency_stats={},
        conversation_history=conversation_history or [],
        user_context="",
    )
