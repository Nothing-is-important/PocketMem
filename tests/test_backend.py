"""验证 InferenceBackend 抽象层。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from backend import create_backend
from backend.local_simulate import LocalSimulateBackend
from config import get_settings


@pytest.fixture(scope="module")
def backend():
    """使用 settings 中的本地模型路径 + CPU 以加速测试。"""
    settings = get_settings()
    return LocalSimulateBackend(
        embedding_model_name=settings.embedding_model,
        llm_model_name=settings.llm_model,
        device="cpu",
        chunk_size=512,
    )


class TestInferenceBackend:
    def test_embed_shape(self, backend):
        embeddings = backend.embed(["你好世界", "Hello World"])
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.dtype == np.float32
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == 512  # bge-small-zh-v1.5

    def test_embed_single_text(self, backend):
        embeddings = backend.embed(["测试"])
        assert embeddings.shape == (1, 512)

    def test_generate_returns_string(self, backend):
        result = backend.generate("1+1=?", max_tokens=10)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_with_longer_prompt(self, backend):
        prompt = "请用中文回答：什么是机器学习？" * 5
        result = backend.generate(prompt, max_tokens=30)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_logits_shape(self, backend):
        logits = backend.logits("你好")
        assert isinstance(logits, np.ndarray)
        assert logits.ndim == 1
        assert logits.dtype == np.float32

    def test_backend_type(self, backend):
        assert backend.backend_type == "local_simulate"

    def test_latency_recording(self, backend):
        backend.embed(["测试延迟"])
        backend.generate("测试延迟", max_tokens=5)
        latency = backend.get_last_latency()
        assert "embed_ms" in latency
        assert "generate_ms" in latency
        assert latency["embed_ms"] > 0
        assert latency["generate_ms"] > 0


class TestBackendFactory:
    def test_create_local_simulate(self):
        settings = get_settings()
        backend = create_backend(
            "local_simulate",
            embedding_model_name=settings.embedding_model,
            llm_model_name=settings.llm_model,
            device="cpu",
        )
        assert backend.backend_type == "local_simulate"

    def test_create_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown backend type"):
            create_backend("nonexistent")

    def test_create_mobile_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            create_backend("mobile_android")
