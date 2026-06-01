"""LangGraph Agent 图编排。

图结构：
    START → Router
             ├─ refuse → END
             └─ 其他所有意图 → Retrieve → MemoryJudge
                                      ├─ sufficient → Generate → END
                                      └─ insufficient → Reflect(1次)

Hook 挂载点：
    pre_route      - Router 执行前
    pre_retrieve   - Retriever 执行前
    post_retrieve  - Retriever 执行后
    post_generate  - Generator 执行后
"""

from langgraph.graph import END, StateGraph

from .generator import create_generator_node
from .hooks import hooks
from .memory_judge import create_memory_judge_node
from .retriever_node import create_retriever_node
from .router import create_router_node
from .state import AgentState


def _wrap_with_hooks(node_fn, pre_point=None, post_point=None):
    """用 hook 包装节点函数——在执行前后插入 hook 调用。

    Hook 失败不阻塞主流程。
    """

    def wrapped(state):
        if pre_point:
            state = hooks.run(pre_point, state)
        result = node_fn(state)
        if post_point:
            result = hooks.run(post_point, result)
        return result

    return wrapped


def build_agent_graph(
    backend,
    entity_aware_retriever,
    hybrid_retriever,
    max_reflect_iterations: int = 1,
) -> StateGraph:
    """构建 PocketMemory Agent 图。

    节点自动被 Hook 系统包装，支持在 4 个挂载点扩展功能。
    """
    # 创建节点（原始函数 + Hook 包装）
    router = _wrap_with_hooks(
        create_router_node(backend),
        pre_point="pre_route",
    )
    retriever = _wrap_with_hooks(
        create_retriever_node(entity_aware_retriever, hybrid_retriever),
        pre_point="pre_retrieve",
        post_point="post_retrieve",
    )
    memory_judge = create_memory_judge_node(
        backend,
        max_reflect_iterations=max_reflect_iterations,
    )
    generator = _wrap_with_hooks(
        create_generator_node(backend),
        post_point="post_generate",
    )

    # 构建图
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router)
    workflow.add_node("retrieve", retriever)
    workflow.add_node("judge", memory_judge)
    workflow.add_node("generate", generator)

    # 入口
    workflow.set_entry_point("router")

    # 路由条件边
    workflow.add_conditional_edges(
        "router",
        _route_by_intent,
        {
            "retrieve": "retrieve",
            "generate": "generate",
            "end": END,
        },
    )

    # 检索后进入判断
    workflow.add_edge("retrieve", "judge")

    # 判断条件边
    workflow.add_conditional_edges(
        "judge",
        _route_after_judge,
        {
            "generate": "generate",
            "reflect": "retrieve",  # 补充搜索 → 回到检索
            "end": END,
        },
    )

    # 生成后结束
    workflow.add_edge("generate", END)

    return workflow.compile()


def _route_by_intent(state: AgentState) -> str:
    """根据意图决定下一步。
    
    RAG 项目：所有查询都应走检索管线，只有明确拒绝的查询才跳过。
    """
    intent = state.get("intent", "general")
    
    if intent == "refuse":
        return "end"
    # 所有其他意图（memory_lookup, knowledge_lookup, general）→ 检索
    return "retrieve"


def _route_after_judge(state: AgentState) -> str:
    """根据 MemoryJudge 结果决定下一步。"""
    if state.get("context_sufficient", False):
        return "generate"

    if state.get("reflect_queries") and state.get("reflect_count", 0) <= 1:
        return "reflect"

    return "generate"
