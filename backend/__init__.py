"""InferenceBackend 抽象层 —— 统一端侧推理接口。

设计理念：抽象层将 LLM/Embedding 推理与业务逻辑解耦，
PC 开发用 LocalSimulateBackend，部署时切换到 MobileBackend。
"""

from .base import InferenceBackend
from .local_simulate import LocalSimulateBackend
from .mobile_backend import MobileBackend


def create_backend(
    backend_type: str = "local_simulate", **kwargs
) -> InferenceBackend:
    """工厂函数：根据类型字符串创建推理后端实例。"""
    backends = {
        "local_simulate": LocalSimulateBackend,
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
    "MobileBackend",
    "create_backend",
]
