"""InferenceBackend 抽象层 —— 统一端侧推理接口。

支持后端：
- local_simulate: transformers 本地推理（开发调试）
- vllm: vLLM OpenAI 兼容 API（生产，5-10x 加速）
- mobile_android/mobile_ios: 手机部署骨架
"""

from .base import InferenceBackend
from .local_simulate import LocalSimulateBackend
from .mobile_backend import MobileBackend
from .vllm_backend import VLLMBackend


def create_backend(
    backend_type: str = "local_simulate", **kwargs
) -> InferenceBackend:
    """工厂函数：根据类型字符串创建推理后端实例。"""
    backends = {
        "local_simulate": LocalSimulateBackend,
        "vllm": VLLMBackend,
        "mobile_android": MobileBackend,
        "mobile_ios": MobileBackend,
    }
    cls = backends.get(backend_type)
    if cls is None:
        raise ValueError(
            f"Unknown backend type: {backend_type}. "
            f"Available: {list(backends.keys())}"
        )
    return cls(**kwargs)


__all__ = [
    "InferenceBackend",
    "LocalSimulateBackend",
    "VLLMBackend",
    "MobileBackend",
    "create_backend",
]
