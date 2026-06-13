"""ChromaDB 向量库封装。

本地持久化、HNSW 索引，支持元数据过滤。
"""

from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from data_ingestion.chunker import DocumentChunk


class VectorStore:
    """ChromaDB 本地向量库。

    特性：
    - 持久化到磁盘（断点续用）
    - HNSW 索引（高召回、低延迟）
    - 余弦距离度量
    - 元数据过滤（按来源、时间、参与者等）
    """

    def __init__(
        self,
        persist_dir: str = "./data/chroma_db",
        collection_name: str = "pocket_memory",
        embedding_function=None,
    ):
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # 如果提供了自定义 embedding 函数则使用，否则用 ChromaDB 默认
        self._embedding_fn = embedding_function

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[DocumentChunk]):
        """批量添加文档块到向量库。"""
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [_sanitize_metadata(c.metadata) for c in chunks]
        # DEBUG
        if chunks and metadatas:
            print(f"[add_chunks] first meta keys={list(chunks[0].metadata.keys())[:8] if hasattr(chunks[0], 'metadata') else 'NO_META'}")

        # 如果有自定义 embedding 函数，预计算 embedding
        # 避免 ChromaDB 尝试用默认模型下载
        if self._embedding_fn:
            embeddings = self._embedding_fn(documents)
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        else:
            self._collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

    def add_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
    ):
        """直接添加预计算的 embeddings（跳过 ChromaDB 的 embedding 步骤）。"""
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas or [{}] * len(ids),
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """语义检索。

        用自定义 embedding 函数预计算查询向量，
        避免 ChromaDB 内部 embed_query 接口兼容问题。
        """
        if self._embedding_fn:
            query_emb = self._embedding_fn([query])[0]
            return self.search_by_embedding(query_emb, top_k, where)

        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return _format_results(results)

    def search_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """使用预计算的 embedding 检索。"""
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        return _format_results(results)

    def count(self) -> int:
        """返回已索引文档数。"""
        return self._collection.count()

    def clear(self):
        """清空集合中所有文档。"""
        all_ids = self._collection.get()["ids"]
        if all_ids:
            self._collection.delete(ids=all_ids)

    def get_sample_documents(self, n: int = 20) -> List[str]:
        """获取样本文档内容（用于实体提取等离线分析）。

        Args:
            n: 返回的最大文档数

        Returns:
            文档内容字符串列表
        """
        if self._collection.count() == 0:
            return []
        try:
            result = self._collection.get(
                limit=min(n, self._collection.count()),
                include=["documents"],
            )
            docs = result.get("documents", [])
            return [d for d in docs if d and d.strip()] if docs else []
        except Exception:
            return []

    def get_collection_stats(self) -> Dict:
        """获取集合统计信息。"""
        return {
            "count": self._collection.count(),
            "name": self._collection.name,
        }


def _sanitize_metadata(metadata: Dict) -> Dict:
    """ChromaDB 要求元数据值为 str/int/float/bool。"""
    clean = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, list):
            clean[k] = ", ".join(str(x) for x in v)
        else:
            clean[k] = str(v)
    return clean


def _format_results(results) -> List[Dict]:
    """将 ChromaDB 返回格式转为统一的字典列表。"""
    formatted = []
    raw_ids = results.get("ids", [[]])
    raw_docs = results.get("documents", [[]])
    raw_metas = results.get("metadatas", [[]])
    ids = raw_ids[0] if raw_ids else []
    docs = raw_docs[0] if raw_docs else []
    metas = raw_metas[0] if raw_metas else []
    distances = results.get("distances", [[]])[0]

    # DEBUG
    from utils import log_file
    log_file(f"  [_format] ids#={len(ids)}, metas#={len(metas)}, first_meta_keys={list(metas[0].keys()) if metas else 'EMPTY'}")

    for i in range(len(ids)):
        # 余弦距离 → 相似度分数（距离 0 = 完全相似）
        distance = distances[i] if i < len(distances) else 0
        score = 1.0 - distance / 2.0  # 近似转换

        formatted.append({
            "id": ids[i] if i < len(ids) else "",
            "content": docs[i] if i < len(docs) else "",
            "metadata": metas[i] if i < len(metas) else {},
            "score": round(score, 4),
        })

    return formatted
