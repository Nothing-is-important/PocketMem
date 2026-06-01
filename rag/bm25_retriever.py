"""BM25 关键词检索器。

使用 jieba 中文分词 + rank_bm25 库实现。
适合精确关键词匹配，与向量检索互补。
"""

from typing import List

import jieba
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """基于 BM25 的关键词检索器。

    特点：
    - jieba 中文分词
    - 自动增量索引
    - 与向量检索结果通过 RRF 融合
    """

    def __init__(self):
        self._corpus: List[List[str]] = []
        self._documents: List[str] = []
        self._model: BM25Okapi = None

    def index(self, documents: List[str]):
        """添加文档到 BM25 索引。

        Args:
            documents: 文档文本列表
        """
        tokenized = [self._tokenize(doc) for doc in documents]
        self._corpus.extend(tokenized)
        self._documents.extend(documents)
        self._rebuild_model()

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        """BM25 关键词检索。

        Args:
            query: 查询文本
            top_k: 返回结果数

        Returns:
            [{"content": str, "score": float, "rank": int}, ...]
        """
        if self._model is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._model.get_scores(tokenized_query)

        # 排序取 top_k
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (idx, score) in enumerate(indexed[:top_k]):
            if score > 0:
                results.append({
                    "content": self._documents[idx],
                    "score": float(score),
                    "rank": rank + 1,
                })

        return results

    def clear(self):
        """清空索引。"""
        self._corpus = []
        self._documents = []
        self._model = None

    def _tokenize(self, text: str) -> List[str]:
        """中文分词 + 去停用词。"""
        words = jieba.lcut(text)
        return [w.strip() for w in words if len(w.strip()) > 1]

    def _rebuild_model(self):
        """重建 BM25 模型。"""
        if self._corpus:
            self._model = BM25Okapi(self._corpus)
