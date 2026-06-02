"""vLLM 推理后端 —— 通过 OpenAI 兼容 API 连接。

特性：
- 5-10x 推理加速（PagedAttention + CUDA Graph + 连续批处理）
- 原生支持 Qwen3 /think 思考模式（chat_template_kwargs）
- token 级流式 SSE，实时检测 </think> 分隔

部署要求：
- vLLM Server 需在 WSL2/Docker 中运行
- `vllm serve F:/Models/Qwen3-4B --port 8001`
"""

import time
import numpy as np
from typing import List

from .base import InferenceBackend
from utils import get_logger

logger = get_logger("vllm")


class VLLMBackend(InferenceBackend):
    """通过 OpenAI 兼容 API 连接 vLLM Server。

    Embedding 仍使用 sentence-transformers（100MB，本地推理已足够快）。
    LLM 生成全部走 vLLM 的 HTTP API。
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001/v1",
        model_name: str = "Qwen/Qwen3-4B",
        embedding_model_name: str = "BAAI/bge-small-zh-v1.5",
        device: str = "cuda",
    ):
        self._base_url = base_url
        self._model = model_name
        self.device = device if device == "cuda" else "cpu"
        self._client = None  # 延迟初始化

        # Embedding 用本地 sentence-transformers（轻量、够快）
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", embedding_model_name)
        self._embedding_model = SentenceTransformer(
            embedding_model_name,
            device=self.device,
            trust_remote_code=True,
        )

        self._last_embed_latency_ms = 0.0
        self._last_generate_latency_ms = 0.0

    @property
    def backend_type(self) -> str:
        return "vllm"

    def _ensure_client(self):
        """延迟初始化 OpenAI 客户端（避免 import 时连接）。"""
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                base_url=self._base_url,
                api_key="not-needed",
            )

    # ═══════════════════════════════════════════════════════════════
    # Embedding（本地 sentence-transformers）
    # ═══════════════════════════════════════════════════════════════

    def embed(self, texts: List[str]) -> np.ndarray:
        t0 = time.time()
        embeddings = self._embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._last_embed_latency_ms = (time.time() - t0) * 1000
        return embeddings.astype(np.float32)

    # ═══════════════════════════════════════════════════════════════
    # 文本生成（vLLM OpenAI API）
    # ═══════════════════════════════════════════════════════════════

    def generate(self, prompt: str, max_tokens: int = 2048) -> str:
        """同步生成（非流式，用于降级/Router）。"""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._generate_async(prompt, max_tokens))

    async def _generate_async(self, prompt: str, max_tokens: int = 2048) -> str:
        t0 = time.time()
        self._ensure_client()

        response = await self._client.completions.create(
            model=self._model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.8,
        )
        text = response.choices[0].text.strip()

        self._last_generate_latency_ms = (time.time() - t0) * 1000
        return text

    async def generate_with_thinking(self, messages: list, max_tokens: int = 2048):
        """异步流式思考模式生成。

        通过 vLLM 的 chat_template_kwargs 启用 /think 模式，
        在 SSE 流中实时检测 </think> 分隔思考与回答。

        Yields:
            ("think", str): 思考过程文本片段
            ("answer", str): 最终回答文本片段
        """
        t0 = time.time()
        self._ensure_client()

        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.95,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            stream=True,
        )

        in_thinking = True
        buffer = ""
        THINK_END = "</think>"

        async for chunk in stream:
            delta = chunk.choices[0].delta
            text = delta.content if delta.content else ""

            if not text:
                continue

            buffer += text

            if in_thinking and THINK_END in buffer:
                idx = buffer.index(THINK_END)
                think_part = buffer[:idx]
                if think_part.strip():
                    yield ("think", think_part)
                in_thinking = False
                buffer = buffer[idx + len(THINK_END):]
                if buffer.strip():
                    yield ("answer", buffer)
                buffer = ""
            elif in_thinking:
                if buffer.strip():
                    yield ("think", buffer)
                    buffer = ""
            else:
                if buffer.strip():
                    yield ("answer", buffer)
                    buffer = ""

        if buffer.strip():
            yield ("answer" if not in_thinking else "think", buffer)

        self._last_generate_latency_ms = (time.time() - t0) * 1000
        logger.info(
            "VLLM GenerateThinking: latency=%.1fms", self._last_generate_latency_ms
        )

    # ═══════════════════════════════════════════════════════════════
    # Logits（vLLM 不直接支持，用于评测时降级）
    # ═══════════════════════════════════════════════════════════════

    def logits(self, prompt: str) -> np.ndarray:
        raise NotImplementedError(
            "vLLM backend 不支持 logits 接口。评测请使用 LocalSimulateBackend。"
        )

    def get_last_latency(self) -> dict:
        return {
            "embed_ms": self._last_embed_latency_ms,
            "generate_ms": self._last_generate_latency_ms,
        }
