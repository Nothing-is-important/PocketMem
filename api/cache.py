"""双层语义缓存。

L1 精确缓存: 基于 MD5 查询键的 LRU 缓存，TTL 5 分钟
L2 语义缓存: 基于向量相似度的缓存（余弦相似度 > 0.95），TTL 30 分钟
"""

import hashlib
import time
from collections import OrderedDict
from typing import Any, List, Optional


class LRUCache:
    """L1 精确缓存 —— 相同查询直接返回缓存结果。"""

    def __init__(self, capacity: int = 128, ttl_seconds: int = 300):
        self._capacity = capacity
        self._ttl = ttl_seconds
        self._store: OrderedDict = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        if key not in self._store:
            return None

        entry = self._store[key]
        if time.time() - entry["ts"] > self._ttl:
            del self._store[key]
            return None

        # LRU: 移到最后
        self._store.move_to_end(key)
        return entry["value"]

    def set(self, key: str, value: Any):
        if key in self._store:
            self._store.move_to_end(key)
        else:
            if len(self._store) >= self._capacity:
                self._store.popitem(last=False)
        self._store[key] = {"value": value, "ts": time.time()}

    def clear(self):
        self._store.clear()


class SemanticCache:
    """L2 语义缓存 —— 相似查询返回缓存结果。"""

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
    """L1 + L2 双层缓存协调器。"""

    def __init__(
        self,
        embedding_fn,
        l1_capacity: int = 128,
        l1_ttl: int = 300,
        l2_threshold: float = 0.95,
        l2_ttl: int = 1800,
    ):
        self._l1 = LRUCache(capacity=l1_capacity, ttl_seconds=l1_ttl)
        self._l2 = SemanticCache(
            embedding_fn=embedding_fn,
            threshold=l2_threshold,
            ttl_seconds=l2_ttl,
        )

    def lookup(self, query: str) -> Optional[dict]:
        """双层查询：先 L1 精确匹配，再 L2 语义匹配。"""
        l1_key = hashlib.md5(query.encode("utf-8")).hexdigest()

        # L1: 精确缓存
        result = self._l1.get(l1_key)
        if result is not None:
            result["cache_tier"] = "L1"
            return result

        # L2: 语义缓存
        result = self._l2.get(query)
        if result is not None:
            result["cache_tier"] = "L2"
            # 写入 L1 以加速后续相同查询
            self._l1.set(l1_key, result)
            return result

        return None

    def store(self, query: str, result: dict):
        """同时写入两级缓存。"""
        l1_key = hashlib.md5(query.encode("utf-8")).hexdigest()
        self._l1.set(l1_key, result)
        self._l2.set(query, result)

    def clear(self):
        self._l1.clear()
        self._l2.clear()
