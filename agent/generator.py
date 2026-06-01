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

GENERATOR_MAX_TOKENS = 2048


def build_thinking_messages(state: AgentState) -> list:
    """构建 Qwen3 思考模式的 chat messages。

    Returns:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    """
    query = state["query"]
    memory_context = state.get("memory_context", [])
    temporal_context = state.get("temporal_context", "")
    conv_history = state.get("conversation_history", [])

    # System prompt：简洁，与思考模式配合
    system_prompt = (
        "你是一个个人记忆助手。先分析提供的记忆片段，再给出回答。"
        "只基于记忆片段回答，不要编造。找不到就说不知道。"
        "回答末尾注明来源日期和参与人。回答简洁。"
    )

    # 用户消息：拼接记忆上下文 + 查询
    user_parts = []
    if temporal_context and temporal_context != "无时间信息":
        user_parts.append(f"时间范围：{temporal_context}")

    if memory_context:
        user_parts.append("相关记忆片段：")
        for i, item in enumerate(memory_context[:5]):
            content = item.get("content", "")[:300]
            meta = item.get("metadata", {})
            ts = meta.get("timestamp", "")[:10] if meta.get("timestamp") else ""
            participants = meta.get("participants", "")
            if isinstance(participants, list):
                participants = ", ".join(participants)
            label = f"[{i+1}]"
            if ts:
                label += f" {ts}"
            if participants:
                label += f" ({participants})"
            user_parts.append(f"{label}\n{content}")
    else:
        user_parts.append("（未找到相关记忆片段）")

    # 多轮对话历史
    if conv_history:
        recent = conv_history[-2:]
        user_parts.append("\n之前的对话：")
        for turn in recent:
            user_parts.append(f"用户：{turn.get('query', '')}")
            user_parts.append(f"助手：{turn.get('answer', '')}")

    user_parts.append(f"\n用户查询：{query}")
    user_message = "\n".join(user_parts)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


def build_generator_prompt(state: AgentState) -> str:
    """从 AgentState 构建最终生成 prompt（流式和非流式共用）。"""
    query = state["query"]
    memory_context = state.get("memory_context", [])
    temporal_context = state.get("temporal_context", "")
    user_context = state.get("user_context", "")
    # 多轮对话：注入最近 3 轮历史
    conv_ctx = _build_conversation_context(state.get("conversation_history", []))

    if not memory_context:
        prompt = _build_no_context_prompt(query, user_context)
        return _inject_conv(prompt, conv_ctx)
    elif len(memory_context) >= 2:
        facts = _extract_facts_no_cache(state)
        state["extracted_facts"] = facts
        prompt = _build_fact_prompt(query, facts, temporal_context, user_context)
        return _inject_conv(prompt, conv_ctx)
    else:
        prompt = _build_context_prompt(query, memory_context, temporal_context, user_context)
        return _inject_conv(prompt, conv_ctx)


def _build_conversation_context(history: list) -> str:
    """构建多轮对话上下文。"""
    if not history:
        return ""
    recent = history[-3:]  # 最近 3 轮
    lines = ["\n之前的对话："]
    for i, turn in enumerate(recent):
        lines.append(f"用户：{turn.get('query', '')}")
        lines.append(f"助手：{turn.get('answer', '')}")
    return "\n".join(lines)


def _inject_conv(prompt: str, conv_ctx: str) -> str:
    """将对话上下文注入 prompt（插入到 system prompt 之后）。"""
    if not conv_ctx:
        return prompt
    # 在第一个空行后插入对话上下文
    parts = prompt.split("\n\n", 1)
    if len(parts) == 2:
        return parts[0] + "\n" + conv_ctx + "\n\n" + parts[1]
    return prompt + "\n" + conv_ctx


def _extract_facts_no_cache(state: AgentState) -> str:
    """从 memory_context 提取事实（无 backend 版本，用于 prompt 构建后流式调用）。"""
    memory_context = state.get("memory_context", [])
    chunk_texts = []
    for i, item in enumerate(memory_context[:3]):
        content = item.get("content", "")
        meta = item.get("metadata", {})
        ts = meta.get("timestamp", "")[:10] if meta.get("timestamp") else ""
        participants = meta.get("participants", "")
        if isinstance(participants, list):
            participants = ", ".join(participants)
        label = f"片段{i+1}"
        if ts:
            label += f" ({ts}"
            if participants:
                label += f", {participants}"
            label += ")"
        if len(content) > 300:
            focused = content[:150] + "\n...\n" + content[-150:]
        else:
            focused = content
        chunk_texts.append(f"{label}:\n{focused}")

    chunks_text = "\n\n".join(chunk_texts)
    return prompts.get("fact_extractor").format(query=state["query"], chunks_text=chunks_text)


def create_generator_node(backend, enable_fact_extraction: bool = True):
    """创建答案生成节点。

    Qwen3+ 思考模式：使用 generate_with_thinking() 生成，
    自动分离思考过程和最终回答。
    """

    def generator_node(state: AgentState) -> AgentState:
        t0 = time.time()
        query = state["query"]

        # 构建对话消息（支持 Qwen3 chat template）
        messages = build_thinking_messages(state)

        # 优先使用思考模式生成（流式 yield，节点内收集结果）
        if hasattr(backend, 'generate_with_thinking'):
            thinking_text = ""
            answer_text = ""
            for event_type, text in backend.generate_with_thinking(
                messages, max_tokens=GENERATOR_MAX_TOKENS
            ):
                if event_type == "think":
                    thinking_text += text
                elif event_type == "answer":
                    answer_text += text
            # 思考模式返回空 → 降级到标准生成
            if not answer_text.strip() and not thinking_text.strip():
                prompt = build_generator_prompt(state)
                answer_text = backend.generate(prompt, max_tokens=GENERATOR_MAX_TOKENS)
                thinking_text = ""
            state["final_answer"] = answer_text.strip() or thinking_text.strip()
            state["_thinking"] = thinking_text.strip()
        else:
            # 降级：标准生成
            prompt = build_generator_prompt(state)
            state["final_answer"] = backend.generate(prompt, max_tokens=GENERATOR_MAX_TOKENS)

        state["latency_stats"]["generate_ms"] = (time.time() - t0) * 1000

        state["messages"].append({
            "role": "generator",
            "content": state["final_answer"],
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
