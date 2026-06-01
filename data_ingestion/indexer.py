"""索引器 —— 将 DocumentChunk 写入向量数据库和 BM25 索引。"""

from typing import List

from .chunker import DocumentChunk


class Indexer:
    """将摄取管线的输出写入 ChromaDB 和 BM25。

    用法:
        vector_store = VectorStore(...)
        bm25 = BM25Retriever()
        indexer = Indexer(vector_store, bm25)
        indexer.index(chunks)
    """

    def __init__(self, vector_store, bm25_retriever):
        self._vector_store = vector_store
        self._bm25 = bm25_retriever
        self._indexed_hashes: set = set()

    def index(self, chunks: List[DocumentChunk]) -> int:
        """批量索引文档块。

        Args:
            chunks: DocumentChunk 列表

        Returns:
            新索引的文档数（跳过已索引的）
        """
        new_chunks = [
            c for c in chunks
            if c.chunk_id not in self._indexed_hashes
        ]
        if not new_chunks:
            return 0

        # 写入向量数据库
        self._vector_store.add_chunks(new_chunks)

        # 写入 BM25 索引
        texts = [c.content for c in new_chunks]
        self._bm25.index(texts)

        # 记录已索引
        for c in new_chunks:
            self._indexed_hashes.add(c.chunk_id)

        return len(new_chunks)

    def clear(self):
        """清除索引。"""
        self._indexed_hashes.clear()
        self._vector_store.clear()
        self._bm25.clear()

    @property
    def indexed_count(self) -> int:
        return len(self._indexed_hashes)
