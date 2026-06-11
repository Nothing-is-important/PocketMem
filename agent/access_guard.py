"""AccessGuard —— 文档权限拦截节点。

企业级必需要素：不同权限的用户看到不同的检索结果。

设计原则：默认拒绝（deny by default）
- 文档分级：public / internal / confidential
- 用户角色：admin / editor / viewer
- admin 可看所有，editor 可看 public+internal，viewer 只看 public
- 元数据中无 level 字段 → 视为 confidential（宁可多拦，不能漏拦）
"""

from .state import AgentState


# 权限矩阵：用户角色能看哪些文档
PERMISSION_MATRIX = {
    "admin": {"public", "internal", "confidential"},
    "editor": {"public", "internal"},
    "viewer": {"public"},
}


def create_access_guard_node(default_role: str = "editor"):
    """创建 AccessGuard 节点。

    Args:
        default_role: 演示用默认角色。生产环境从 SSO/JWT 获取。
    """

    def access_guard_node(state: AgentState) -> AgentState:
        results = state.get("retrieval_results", [])
        if not results:
            state["messages"].append({
                "role": "guard",
                "content": "AccessGuard: no results to filter",
            })
            return state

        allowed = _get_allowed_levels(default_role)
        before = len(results)

        filtered = []
        blocked_ids = []
        for item in results:
            doc_level = _extract_level(item)
            if doc_level in allowed:
                filtered.append(item)
            else:
                blocked_ids.append(item.get("id", "unknown"))

        state["retrieval_results"] = filtered

        blocked_count = before - len(filtered)
        state["messages"].append({
            "role": "guard",
            "content": (
                f"AccessGuard (role={default_role}): "
                f"{before} → {len(filtered)} results "
                f"(blocked {blocked_count} docs: {blocked_ids[:3]}{'...' if blocked_count > 3 else ''})"
            ),
        })

        return state

    return access_guard_node


def _get_allowed_levels(role: str) -> set:
    return PERMISSION_MATRIX.get(role, {"public"})


def _extract_level(item: dict) -> str:
    """从检索结果的元数据中提取文档分级。

    优先级：metadata.level > metadata.source_type推断 > 默认confidential
    """
    meta = item.get("metadata", {})

    level = meta.get("level", "")
    if level and level in ("public", "internal", "confidential"):
        return level

    # 从来源类型推断：contracts → confidential, meetings → internal
    source_file = str(meta.get("source_file", ""))
    if "contract" in source_file.lower() or "合同" in source_file:
        return "confidential"
    if "meeting" in source_file.lower() or "会议" in source_file:
        return "internal"

    # 默认：无标记 = 视为机密（deny by default）
    return "confidential"
