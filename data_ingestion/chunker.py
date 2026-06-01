"""对话感知分块器。

将聊天消息分组成对话片段（基于时间临近性），然后切成适合检索的固定大小块。
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .wechat_parser import ChatMessage


@dataclass
class DocumentChunk:
    """用于索引和检索的统一文档块。"""
    chunk_id: str
    content: str
    metadata: Dict = field(default_factory=dict)

    def to_langchain_document(self):
        """转换为 LangChain Document 格式。"""
        from langchain.schema import Document
        return Document(page_content=self.content, metadata=self.metadata)


class ConversationChunker:
    """对话感知分块器。

    策略：
    1. 将时间间隔 <= gap_minutes 的消息归为同一会话
    2. 每个会话内部按 target_tokens 切块，带 overlap_tokens 重叠
    3. 每个块携带元数据：时间范围、参与者、重要性评分
    """

    def __init__(
        self,
        gap_minutes: int = 30,
        target_tokens: int = 512,
        overlap_tokens: int = 64,
    ):
        self.gap_minutes = gap_minutes
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_messages(
        self,
        messages: List[ChatMessage],
        source_file: str = "",
    ) -> List[DocumentChunk]:
        """将消息列表分块。

        Args:
            messages: 已排序的 ChatMessage 列表
            source_file: 来源文件路径

        Returns:
            DocumentChunk 列表
        """
        if not messages:
            return []

        # Step 1: 按时间间隔分组为会话
        sessions = self._group_into_sessions(messages)

        # Step 2: 每个会话内切块
        chunks = []
        for session_idx, session_msgs in enumerate(sessions):
            session_chunks = self._chunk_session(
                session_msgs, session_idx, source_file
            )
            chunks.extend(session_chunks)

        return chunks

    def _group_into_sessions(
        self, messages: List[ChatMessage]
    ) -> List[List[ChatMessage]]:
        """按时间间隔将消息分组成会话。"""
        if not messages:
            return []

        sessions = []
        current_session = [messages[0]]

        for i in range(1, len(messages)):
            gap = messages[i].timestamp - messages[i - 1].timestamp
            if gap <= timedelta(minutes=self.gap_minutes):
                current_session.append(messages[i])
            else:
                sessions.append(current_session)
                current_session = [messages[i]]

        sessions.append(current_session)
        return sessions

    def _chunk_session(
        self,
        messages: List[ChatMessage],
        session_idx: int,
        source_file: str,
    ) -> List[DocumentChunk]:
        """将一个会话内的消息切为多个块。"""
        # 转为格式化文本
        text = "\n".join(m.to_text() for m in messages)

        # 简单估算 token 数（中文 ~1.5 字符/token，英文 ~4 字符/token）
        estimated_tokens = self._estimate_tokens(text)

        if estimated_tokens <= self.target_tokens:
            return [self._make_chunk(
                text, session_idx, 0, messages, source_file
            )]

        # 需要切分：逐消息累积直到达到目标 token 数
        chunks = []
        current_lines = []
        current_msgs = []
        current_tokens = 0
        chunk_idx = 0

        for msg in messages:
            msg_text = msg.to_text()
            msg_tokens = self._estimate_tokens(msg_text)

            if current_tokens + msg_tokens > self.target_tokens and current_lines:
                chunks.append(self._make_chunk(
                    "\n".join(current_lines),
                    session_idx, chunk_idx, current_msgs, source_file,
                ))
                chunk_idx += 1

                # 保留最后一条消息作为重叠
                overlap = current_lines[-1] if current_lines else ""
                current_lines = [overlap] if overlap else []
                current_msgs = [current_msgs[-1]] if current_msgs else []
                current_tokens = self._estimate_tokens(overlap)

            current_lines.append(msg_text)
            current_msgs.append(msg)
            current_tokens += msg_tokens

        if current_lines:
            chunks.append(self._make_chunk(
                "\n".join(current_lines),
                session_idx, chunk_idx, current_msgs, source_file,
            ))

        return chunks

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict] = None,
    ) -> List[DocumentChunk]:
        """将纯文本（如 Markdown/PDF 内容）分块。

        Args:
            text: 待分块文本
            metadata: 附加元数据

        Returns:
            DocumentChunk 列表
        """
        base_meta = metadata or {}

        estimated_tokens = self._estimate_tokens(text)
        if estimated_tokens <= self.target_tokens:
            chunk_id = self._hash_content(text)
            return [DocumentChunk(
                chunk_id=chunk_id,
                content=text,
                metadata={**base_meta, "chunk_id": chunk_id, "chunk_index": 0},
            )]

        # 按段落切分
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_lines = []
        current_tokens = 0
        chunk_idx = 0

        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)

            if current_tokens + para_tokens > self.target_tokens and current_lines:
                content = "\n\n".join(current_lines)
                chunk_id = self._hash_content(content)
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    content=content,
                    metadata={
                        **base_meta,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_idx,
                    },
                ))
                chunk_idx += 1
                current_lines = [current_lines[-1]] if current_lines else []
                current_tokens = self._estimate_tokens(current_lines[0]) if current_lines else 0

            current_lines.append(para)
            current_tokens += para_tokens

        if current_lines:
            content = "\n\n".join(current_lines)
            chunk_id = self._hash_content(content)
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                content=content,
                metadata={**base_meta, "chunk_id": chunk_id, "chunk_index": chunk_idx},
            ))

        return chunks

    def _make_chunk(
        self,
        text: str,
        session_idx: int,
        chunk_idx: int,
        messages: List[ChatMessage],
        source_file: str,
    ) -> DocumentChunk:
        """创建一个带完整元数据的 DocumentChunk。"""
        chunk_id = self._hash_content(text)
        participants = list(set(m.sender for m in messages if m.sender))
        timestamps = [m.timestamp for m in messages]

        return DocumentChunk(
            chunk_id=chunk_id,
            content=text,
            metadata={
                "chunk_id": chunk_id,
                "session_idx": session_idx,
                "chunk_index": chunk_idx,
                "source_file": source_file,
                "source_type": "wechat",
                "participants": participants,
                "participant_count": len(participants),
                "timestamp": timestamps[0].isoformat() if timestamps else "",
                "timestamp_earliest": min(timestamps).isoformat() if timestamps else "",
                "timestamp_latest": max(timestamps).isoformat() if timestamps else "",
                "message_count": len(messages),
                "chunk_tokens_estimate": self._estimate_tokens(text),
                "importance": self._calculate_importance(messages, text),
            },
        )

    def _calculate_importance(
        self, messages: list, text: str
    ) -> float:
        """计算一个块的重要性评分 (0.0-1.0)。

        三个维度：
        1. 具体信息密度：包含日期、地点、电话号码？
        2. 互动热度：消息间隔 < 5 分钟说明有真实对话？
        3. 参与者数量：多人讨论比私聊更有信息量？

        基线 0.5，各维度加分，最高 1.0。
        """
        import re

        if not messages:
            # 纯文档（MD/PDF）→ 基于内容估算
            score = 0.5
            if re.search(r"\d{4}[-/年]", text):
                score += 0.1
            if len(text) > 500:
                score += 0.1  # 长文档 = 信息量大
            return min(score, 1.0)

        score = 0.5  # 基线：普通的聊天消息

        # 维度 1：具体信息密度（日期/地点/电话/数字）
        concrete_signals = 0
        if re.search(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}", text):
            concrete_signals += 1  # 具体日期
        if re.search(r"(在|去|地址|位置|店|餐厅|公司|医院|学校)", text):
            concrete_signals += 1  # 地点提及
        if re.search(r"\d{11}", text) or re.search(r"\d{3,4}-\d{7,8}", text):
            concrete_signals += 1  # 电话号码
        if re.search(r"\d+元|\d+块|¥\d+|\$\d+|[0-9]+万", text):
            concrete_signals += 1  # 金额/价格
        score += min(concrete_signals * 0.1, 0.3)

        # 维度 2：互动热度（5 分钟内回复 = 真实交流）
        close_replies = 0
        for i in range(1, len(messages)):
            gap = (messages[i].timestamp - messages[i - 1].timestamp).total_seconds()
            if gap < 300:  # 5分钟
                close_replies += 1
        if close_replies >= 3:
            score += 0.2
        elif close_replies >= 1:
            score += 0.1

        # 维度 3：参与人数
        participants = set(m.sender for m in messages if hasattr(m, "sender"))
        if len(participants) >= 3:
            score += 0.1

        return min(score, 1.0)

    def _estimate_tokens(self, text: str) -> int:
        """粗略估算 token 数。中文按 1.5 字符/token，英文按 4 字符/token。"""
        chinese_chars = sum(1 for c in text if "一" <= c <= "鿿")
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    @staticmethod
    def _hash_content(content: str) -> str:
        return hashlib.md5(content.encode("utf-8")).hexdigest()[:16]
