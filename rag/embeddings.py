"""端侧 Embedding 模型封装。

包装 InferenceBackend.embed()，提供与 LangChain 兼容的接口。
"""

from typing import List

import numpy as np


class OnDeviceEmbeddings:
    """端侧 Embedding 模型。

    包装 InferenceBackend.embed() 并提供：
    - LangChain 兼容接口 (embed_documents, embed_query)
    - 原始 numpy 输出 (embed_raw)
    - 批量处理以避免 OOM
    """

    def __init__(self, backend, batch_size: int = 32):
        """
        Args:
            backend: InferenceBackend 实例
            batch_size: 批量推理大小
        """
        self._backend = backend
        self._batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """LangChain 兼容接口：嵌入文档列表。"""
        all_embeddings = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            embeddings = self._backend.embed(batch)
            all_embeddings.extend(embeddings.tolist())
        return all_embeddings

    def embed_query(self, text: str = "", input: str = "") -> List[float]:
        """LangChain 兼容接口：嵌入单个查询，返回 1D 列表。

        支持两种调用方式：embed_query(text="...") 或 embed_query(input="...")
        """
        query = text or input
        return self._backend.embed([query])[0].tolist()

    def embed_raw(self, texts: List[str]) -> np.ndarray:
        """返回 numpy array，用于缓存和相似度计算。"""
        return self._backend.embed(texts)

    def __call__(self, input: List[str]) -> List[List[float]]:
        """ChromaDB EmbeddingFunction 兼容接口。参数名必须是 `input`。"""
        return self.embed_documents(input)

    def name(self) -> str:
        """ChromaDB 要求的方法。"""
        return "bge-small-zh-v1.5"

    @property
    def dim(self) -> int:
        """Embedding 维度（通过一次前向推理获取）。"""
        sample = self.embed_raw(["test"])
        return sample.shape[1]
