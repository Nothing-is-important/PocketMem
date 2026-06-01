"""实体感知检索器 —— 在 RRF 融合后应用实体加权。

查询 → 提取实体 → 混合检索 → RRF 融合 → 实体加权 → 排序返回
"""

from typing import Dict, List, Optional

from .entity_extractor import boost_by_entity_overlap, extract_query_entities
from .hybrid_retriever import HybridRetriever


class EntityAwareRetriever:
    """实体感知检索器。

    包装 HybridRetriever，在 RRF 融合后根据实体重叠度加权。
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        entity_boost_factor: float = 1.2,
    ):
        self._hybrid = hybrid_retriever
        self._boost_factor = entity_boost_factor

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        vector_where: Optional[Dict] = None,
    ) -> List[Dict]:
        """实体感知检索。

        流程：
        1. 提取查询中的实体
        2. 混合检索（向量 + BM25 → RRF 融合）
        3. 对每个结果计算实体重叠加权
        4. 重新排序返回
        """
        query_entities = extract_query_entities(query)

        results = self._hybrid.search(
            query, top_k=top_k, vector_where=vector_where
        )

        # 实体加权
        for result in results:
            entity_boost = boost_by_entity_overlap(
                query_entities,
                result.get("metadata", {}),
                self._boost_factor,
            )
            result["score"] = result["score"] * entity_boost
            result["entity_boost"] = round(entity_boost, 3)

        # 重新排序
        results.sort(key=lambda x: x["score"], reverse=True)

        return results

    @property
    def hybrid(self) -> HybridRetriever:
        return self._hybrid
