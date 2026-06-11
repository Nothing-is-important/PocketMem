"""InferenceBackend 抽象层 —— 统一端侧推理接口。

支持三种后端模式：
- local_simulate: 本地模型推理（PC开发/机密文档）
- vllm: vLLM OpenAI 兼容 API（生产，5-10x 加速）
- dual_mode: 自动切换（公开→API，机密→本地）
"""

from .base import InferenceBackend
from .local_simulate import LocalSimulateBackend
from .mobile_backend import MobileBackend
from .vllm_backend import VLLMBackend
from .dual_backend import DualModeBackend


def create_backend(
    backend_type: str = "local_simulate", **kwargs
) -> InferenceBackend:
    """工厂函数：根据类型字符串创建推理后端实例。"""
    backends = {
        "local_simulate": LocalSimulateBackend,
        "mobile_android": MobileBackend,
        "mobile_ios": MobileBackend,
        "vllm": VLLMBackend,
        "dual_mode": DualModeBackend,
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
    "MobileBackend",
    "VLLMBackend",
    "DualModeBackend",
    "create_backend",
]
