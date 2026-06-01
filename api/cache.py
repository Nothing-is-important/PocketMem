"""语义缓存。
L2 语义缓存: 基于向量相似度的缓存（余弦相似度 > 阈值），支持查询归一化提升命中率。
"""
import re
import time
from typing import Any, List, Optional

import numpy as np


def _normalize_query(query: str) -> str:
    """查询归一化：去标点、去多余空格、小写，提升缓存命中率。"""
    query = re.sub(r'[，。！？、；：""''（）【】《》\s]+', '', query)
    return query.strip().lower()


class SemanticCache:
    """语义缓存 —— 相似查询返回缓存结果。"""

    def __init__(
        self,
        embedding_fn,
        threshold: float = 0.95,
        ttl_seconds: int = 1800,
    ):
        self._embed = embedding_fn
        self._threshold = threshold
        self._ttl = ttl_seconds
        self._keys: List[str] = []
        self._embeddings: List = []
        self._values: List[Any] = []
        self._timestamps: List[float] = []

    def get(self, query: str) -> Optional[Any]:
        if not self._keys:
            return None

        import numpy as np
        query_emb = self._embed([query])[0]

        for i, stored_emb in enumerate(self._embeddings):
            # 检查 TTL
            if time.time() - self._timestamps[i] > self._ttl:
                continue

            # 余弦相似度
            similarity = np.dot(query_emb, stored_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(stored_emb)
            )
            if similarity >= self._threshold:
                return self._values[i]

        return None

    def set(self, query: str, value: Any):
        import numpy as np
        query_emb = self._embed([query])[0]

        self._keys.append(query)
        self._embeddings.append(query_emb)
        self._values.append(value)
        self._timestamps.append(time.time())

        # 限制缓存大小
        max_size = 256
        if len(self._keys) > max_size:
            self._keys.pop(0)
            self._embeddings.pop(0)
            self._values.pop(0)
            self._timestamps.pop(0)

    def clear(self):
        self._keys.clear()
        self._embeddings.clear()
        self._values.clear()
        self._timestamps.clear()


class TwoTierCache:
    """语义缓存协调器（仅保留 L2 语义层，L1 精确匹配命中率接近 0）。"""

    def __init__(
        self,
        embedding_fn,
        l1_capacity: int = 128,
        l1_ttl: int = 300,
        l2_threshold: float = 0.95,
        l2_ttl: int = 1800,
    ):
        # 保留初始化参数签名兼容性，但只使用 L2
        self._l2 = SemanticCache(
            embedding_fn=embedding_fn,
            threshold=l2_threshold,
            ttl_seconds=l2_ttl,
        )

    def lookup(self, query: str) -> Optional[dict]:
        """语义缓存查询（归一化后匹配）。"""
        normalized = _normalize_query(query)
        result = self._l2.get(normalized)
        if result is not None:
            result["cache_tier"] = "L2"
        return result

    def store(self, query: str, result: dict):
        """写入语义缓存。"""
        normalized = _normalize_query(query)
        self._l2.set(normalized, result)

    def clear(self):
        self._l2.clear()
