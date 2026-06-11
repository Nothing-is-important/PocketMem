"""双模式推理后端 —— 非机密走API（快），机密走本地模型（安全）。

核心逻辑：
- 检索后检查文档等级分布
- 如果所有文档都是 public → API 模式（DeepSeek/vLLM）
- 如果存在 internal/confidential → 本地模型模式（W4A8量化）
- 如果一个请求同时涉及两种文档 → 本地模式（安全优先）
"""

import numpy as np
from typing import List

from .base import InferenceBackend
from utils import get_logger

logger = get_logger("dual_backend")


class DualModeBackend(InferenceBackend):
    """双模式推理后端。

    自动根据文档等级切换推理模式：
    - API Mode: 速度快，适合公开文档（通过 OpenAI 兼容 API）
    - Local Mode: 数据不出设备，适合机密文档（本地 W4A8 量化模型）

    安全原则：一旦涉及 internal/confidential 文档，强制使用本地模式。
    """

    def __init__(
        self,
        api_backend: InferenceBackend,    # VLLMBackend 或任何 OpenAI 兼容后端
        local_backend: InferenceBackend,   # LocalSimulateBackend（本地量化模型）
        embedding_backend=None,            # Embedding 模型（两种模式共用）
        default_mode: str = "local",       # 默认模式：安全优先
    ):
        """
        Args:
            api_backend: API 推理后端（速度快，用于公开文档）
            local_backend: 本地推理后端（安全，用于机密文档）
            embedding_backend: Embedding 后端（通常两个是一样的，取第一个）
            default_mode: 默认模式。建议 "local"——宁慢不泄露。
        """
        self._api = api_backend
        self._local = local_backend
        self._embedding = embedding_backend or local_backend  # embedding 通用
        self._default_mode = default_mode
        self._current_mode = default_mode
        self._last_latency = {"embed_ms": 0, "generate_ms": 0}

    @property
    def backend_type(self) -> str:
        return f"dual_mode(api={self._api.backend_type}, local={self._local.backend_type})"

    # ── Embedding（两模式共用 BGE）──

    def embed(self, texts: List[str]) -> np.ndarray:
        result = self._embedding.embed(texts)
        self._last_latency["embed_ms"] = getattr(self._embedding, '_last_embed_latency_ms', 0)
        return result

    # ── 文本生成（自动路由）──

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """生成答案。默认走本地模式。要切换，先用 decide_mode()。"""
        backend = self._local if self._current_mode == "local" else self._api
        result = backend.generate(prompt, max_tokens)
        self._last_latency["generate_ms"] = getattr(backend, '_last_generate_latency_ms', 0)
        logger.info("Generate: mode=%s", self._current_mode)
        return result

    def logits(self, prompt: str) -> np.ndarray:
        return self._local.logits(prompt)

    # ── 模式切换 ──

    def decide_mode(self, documents: list) -> str:
        """根据文档等级决定使用哪个模式。

        Args:
            documents: 检索到的文档列表（每项含 metadata.level）

        Returns:
            "api" 或 "local"
        """
        has_confidential = False
        has_internal = False

        for doc in documents:
            meta = doc.get("metadata", {})
            level = meta.get("level", "").lower()

            if level == "confidential":
                has_confidential = True
            elif level == "internal":
                has_internal = True

        # 安全优先：有机密文档 → 强制本地
        if has_confidential:
            self._current_mode = "local"
        elif has_internal:
            self._current_mode = "local"  # 内部文档也走本地（安全优先策略）
        elif documents:
            self._current_mode = "api"    # 全是公开文档 → API 加速
        else:
            self._current_mode = self._default_mode

        logger.info(
            "Mode decision: %s (confidential=%s, internal=%s, docs=%d)",
            self._current_mode, has_confidential, has_internal, len(documents)
        )
        return self._current_mode

    @property
    def current_mode(self) -> str:
        return self._current_mode

    def get_last_latency(self) -> dict:
        return self._last_latency
