"""全局配置，通过环境变量驱动，无需改代码即可切换模型和后端。"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _find_model(model_dirs: list[str], hf_name: str) -> str:
    """自动查找模型路径：本地目录 > HF_HOME 缓存 > HuggingFace 在线。

    优先级：
    1. 环境变量 POCKET_EMBEDDING_MODEL / POCKET_LLM_MODEL
    2. 常见本地路径
    3. HuggingFace 模型名（在线下载）
    """
    for d in model_dirs:
        if os.path.isdir(d):
            return d
    return hf_name


def _model_search_paths(local_name: str) -> list[str]:
    """生成模型可能的本地路径列表。"""
    candidates = []

    # HF_HOME 环境变量
    hf_home = os.getenv("HF_HOME", "")
    if hf_home:
        candidates.append(os.path.join(hf_home, local_name))

    # 常见的本地缓存位置
    home = str(Path.home())
    candidates.extend([
        f"F:/Models/{local_name}",                              # Windows 自定义
        f"{home}/Models/{local_name}",                          # Linux/Mac 自定义
        f"{home}/.cache/huggingface/hub/models--{local_name}",  # HF 默认缓存
    ])

    return candidates


@dataclass
class Settings:
    # ── 推理后端 ──
    backend_type: str = os.getenv("POCKET_BACKEND", "local_simulate")
    device: str = os.getenv("POCKET_DEVICE", "cuda")

    # ── 模型（自动检测本地路径）──
    embedding_model: str = os.getenv(
        "POCKET_EMBEDDING_MODEL",
        _find_model(
            _model_search_paths("bge-small-zh-v1.5"),
            "BAAI/bge-small-zh-v1.5",
        ),
    )
    llm_model: str = os.getenv(
        "POCKET_LLM_MODEL",
        _find_model(
            _model_search_paths("Qwen2.5-1.5B-Instruct"),
            "Qwen/Qwen2.5-1.5B-Instruct",
        ),
    )
    chunk_size: int = int(os.getenv("POCKET_CHUNK_SIZE", "512"))
    max_tokens: int = int(os.getenv("POCKET_MAX_TOKENS", "512"))

    # ── 向量数据库 ──
    chroma_persist_dir: str = os.getenv(
        "POCKET_CHROMA_DIR", "./data/chroma_db"
    )
    chroma_collection: str = os.getenv(
        "POCKET_CHROMA_COLLECTION", "pocket_memory"
    )

    # ── 检索 ──
    rrf_k: int = int(os.getenv("POCKET_RRF_K", "60"))
    retrieval_top_k: int = int(os.getenv("POCKET_RETRIEVAL_TOP_K", "10"))
    temporal_half_life_days: float = float(
        os.getenv("POCKET_TEMPORAL_HALF_LIFE", "30")
    )
    entity_boost_factor: float = float(
        os.getenv("POCKET_ENTITY_BOOST", "1.2")
    )

    # ── 缓存 ──
    cache_l1_ttl: int = int(os.getenv("POCKET_CACHE_L1_TTL", "300"))
    cache_l2_ttl: int = int(os.getenv("POCKET_CACHE_L2_TTL", "1800"))
    cache_l2_threshold: float = float(
        os.getenv("POCKET_CACHE_L2_THRESHOLD", "0.95")
    )

    # ── 服务 ──
    server_host: str = os.getenv("POCKET_HOST", "0.0.0.0")
    server_port: int = int(os.getenv("POCKET_PORT", "8000"))

    # ── Agent ──
    max_reflect_iterations: int = 1
    relevance_threshold: float = 0.3
    sufficiency_min_threads: int = 2
    sufficiency_min_chunks: int = 3

    # ── 路径 ──
    data_dir: str = "./data"
    raw_data_dir: str = "./data/raw"
    demo_data_dir: str = "./data/demo"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
