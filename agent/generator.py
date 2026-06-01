"""答案生成节点 —— 两阶段生成：提取事实 → 基于事实回答。

针对 1.5B 小模型优化：
- 阶段 1（提取）：从对话 chunk 中提取关键事实（任务简单——从原文复制相关句子）
- 阶段 2（回答）：基于干净的事实列表生成自然语言回答（上下文干净，任务明确）

相比"直接喂对话给模型"，信息密度提升 6-10 倍。
"""

import time

from .state import AgentState

# Prompt 从配置加载（支持版本管理和热加载）
from config.prompts import prompts

GENERATOR_MAX_TOKENS = 512


def create_generator_node(backend, enable_fact_extraction: bool = True):
    """创建答案生成节点。

    Args:
        backend: InferenceBackend 实例
        enable_fact_extraction: 是否启用两阶段生成（默认开启）
    """

    def generator_node(state: AgentState) -> AgentState:
        t0 = time.time()

        query = state["query"]
        memory_context = state.get("memory_context", [])
        temporal_context = state.get("temporal_context", "")
        user_context = state.get("user_context", "")

        if not memory_context:
            prompt = _build_no_context_prompt(query, user_context)
            answer = backend.generate(prompt, max_tokens=GENERATOR_MAX_TOKENS)
        elif enable_fact_extraction and len(memory_context) >= 2:
            # 两阶段生成：提取 → 回答
            facts = _extract_facts(backend, query, memory_context)
            prompt = _build_fact_prompt(
                query, facts, temporal_context, user_context
            )
            answer = backend.generate(prompt, max_tokens=GENERATOR_MAX_TOKENS)
            state["extracted_facts"] = facts
        else:
            # 降级：直接生成（1 条结果时不需要提取）
            prompt = _build_context_prompt(
                query, memory_context, temporal_context, user_context
            )
            answer = backend.generate(prompt, max_tokens=GENERATOR_MAX_TOKENS)

        state["final_answer"] = answer
        state["latency_stats"]["generate_ms"] = (time.time() - t0) * 1000

        state["messages"].append({
            "role": "generator",
            "content": answer,
        })

        return state

    return generator_node


def _extract_facts(backend, query: str, memory_context: list) -> str:
    """阶段 1：从对话 chunk 中提取与查询相关的关键事实。

    这是一个极简任务——模型只需要从原文中找出相关句子并复制出来，
    不需要总结、改写或生成。1.5B 模型完全能胜任。
    """
    # 构建精简的 chunk 文本（只保留关键信息，减少噪音）
    chunk_texts = []
    # 只取 Top-3 chunk，减少噪音（后两条开源项目 chunk 会让小模型分心）
    for i, item in enumerate(memory_context[:3]):
        content = item.get("content", "")
        meta = item.get("metadata", {})
        ts = meta.get("timestamp", "")[:10] if meta.get("timestamp") else ""
        participants = meta.get("participants", "")
        if isinstance(participants, list):
            participants = ", ".join(participants)

        # 精简格式：去掉不必要的前缀
        label = f"片段{i+1}"
        if ts:
            label += f" ({ts}"
            if participants:
                label += f", {participants}"
            label += ")"

        # 截取核心内容（去头去尾的对话格式噪音）
        if len(content) > 300:
            focused = content[:150] + "\n...\n" + content[-150:]
        else:
            focused = content
        chunk_texts.append(f"{label}:\n{focused}")

    chunks_text = "\n\n".join(chunk_texts)

    extraction_prompt = prompts.get("fact_extractor").format(
        query=query,
        chunks_text=chunks_text,
    )

    facts = backend.generate(extraction_prompt, max_tokens=256)
    return facts.strip()


def _build_fact_prompt(
    query: str,
    facts: str,
    temporal_context: str,
    user_context: str = "",
) -> str:
    """阶段 2：基于提取的事实生成自然语言回答。"""
    system_prompt = prompts.get("generator")
    instruction = "请根据以上关键信息回答用户的问题："

    parts = [system_prompt]
    if user_context:
        parts.append(user_context)
    parts.append(f"用户查询：{query}")
    if temporal_context and temporal_context != "无时间信息":
        parts.append(f"相关记忆的时间范围：{temporal_context}")
    parts.append(f"从记忆片段中提取的关键信息：\n{facts}")
    parts.append(instruction)

    return "\n\n".join(parts)


def _build_context_prompt(
    query: str,
    memory_context: list,
    temporal_context: str,
    user_context: str = "",
) -> str:
    """构建带上下文的生成提示词。"""
    context_parts = []
    for i, item in enumerate(memory_context[:5]):
        content = item.get("content", "")
        meta = item.get("metadata", {})

        ts = meta.get("timestamp", "")
        participants = meta.get("participants", "")
        if isinstance(participants, list):
            participants = ", ".join(participants)

        header = f"[记忆片段{i + 1}]"
        if ts:
            header += f" 时间: {ts[:10]}"
        if participants:
            header += f" 参与者: {participants}"

        # 展示上下文：前200字 + 后150字，1.5B模型聚焦关键信息
        if len(content) > 400:
            focused = content[:200] + "\n...\n" + content[-150:]
        else:
            focused = content

        context_parts.append(f"{header}\n{focused}")

    context_str = "\n\n".join(context_parts)
    system_prompt = prompts.get("generator")
    instruction = "请根据以上记忆回答用户的问题："

    return prompts.get("generator_context").format(
        system_prompt=system_prompt,
        user_context=user_context,
        query=query,
        temporal_context=temporal_context,
        context=context_str,
        instruction=instruction,
    )


def _build_no_context_prompt(query: str, user_context: str = "") -> str:
    """构建无上下文的提示词。"""
    return prompts.get("generator_no_context").format(
        system_prompt=prompts.get("generator"),
        user_context=user_context,
        query=query,
    )
