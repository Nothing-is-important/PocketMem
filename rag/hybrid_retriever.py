"""混合检索器 —— RRF 融合向量检索和 BM25 关键词检索。

RRF 公式: score = sum(1 / (k + rank_i))，k=60（TREC 实验推荐值）
两路检索并行执行（ThreadPoolExecutor），总延迟 = max(向量, BM25) 而非 sum。
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from .bm25_retriever import BM25Retriever
from .vector_store import VectorStore


class HybridRetriever:
    """混合检索器。

    将向量语义检索和 BM25 关键词检索的结果通过 RRF 融合，
    兼顾语义相关性和关键词精确匹配。
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25: BM25Retriever,
        rrf_k: int = 60,
        top_k: int = 10,
    ):
        self._vector_store = vector_store
        self._bm25 = bm25
        self._rrf_k = rrf_k
        self._top_k = top_k

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        vector_where: Optional[Dict] = None,
    ) -> List[Dict]:
        """混合检索。

        Args:
            query: 查询文本
            top_k: 返回结果数（覆盖默认值）
            vector_where: 向量检索的元数据过滤条件

        Returns:
            融合后的检索结果列表，每个结果包含:
            - content: 文本内容
            - score: RRF 融合分数
            - vector_score: 向量检索分数
            - bm25_score: BM25 检索分数
            - metadata: 元数据
        """
        k = top_k or self._top_k

        # 并行执行向量检索和 BM25 检索
        with ThreadPoolExecutor(max_workers=2) as executor:
            vector_future = executor.submit(
                self._vector_store.search, query, top_k=k * 2, where=vector_where
            )
            bm25_future = executor.submit(
                self._bm25.search, query, top_k=k * 2
            )
            vector_results = vector_future.result()
            bm25_results = bm25_future.result()

        # RRF 融合
        return self._rrf_fusion(vector_results, bm25_results, k)

    def _rrf_fusion(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict],
        top_k: int,
    ) -> List[Dict]:
        """RRF 融合两路检索结果。

        RRF score = Σ 1/(k + rank_i)
        其中 rank 从 1 开始。
        """
        rrf_scores: Dict[str, dict] = {}

        # 向量检索结果
        for rank, item in enumerate(vector_results, start=1):
            doc_id = item.get("id", item.get("content", ""))
            rrf = 1.0 / (self._rrf_k + rank)
            rrf_scores[doc_id] = {
                "content": item.get("content", ""),
                "metadata": item.get("metadata", {}),
                "score": rrf,
                "vector_score": item.get("score", 0),
                "bm25_score": 0,
            }

        # BM25 检索结果
        for rank, item in enumerate(bm25_results, start=1):
            doc_id = item.get("content", "")  # BM25 用 content 做 key
            rrf = 1.0 / (self._rrf_k + rank)

            if doc_id in rrf_scores:
                rrf_scores[doc_id]["score"] += rrf
                rrf_scores[doc_id]["bm25_score"] = item.get("score", 0)
            else:
                rrf_scores[doc_id] = {
                    "content": item.get("content", ""),
                    "metadata": {},
                    "score": rrf,
                    "vector_score": 0,
                    "bm25_score": item.get("score", 0),
                }

        # 按 RRF 分数排序
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x["score"],
            reverse=True,
        )

        return sorted_results[:top_k]
