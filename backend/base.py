"""统一推理后端抽象接口。

设计理念：
- hbm_runtime 不区分文本/图像，统一通过 numpy array 输入输出
- 这个接口也不区分模态，上层业务只管传数据
- 三个实现：LocalSimulateBackend (PC开发) / MobileBackend (手机部署)
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class InferenceBackend(ABC):
    """统一推理后端抽象接口。"""

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """文本 Embedding 推理。

        Args:
            texts: 待编码的文本列表

        Returns:
            shape (len(texts), embedding_dim), dtype=float32
        """
        ...

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """LLM 文本生成推理。

        对应 hbm_runtime.run() 的 Decode 阶段。

        Args:
            prompt: 输入提示词
            max_tokens: 最大生成 token 数

        Returns:
            生成的文本
        """
        ...

    def generate_with_image(
        self, prompt: str, image_path: str, max_tokens: int = 512
    ) -> str:
        """多模态生成推理 —— 输入文本 + 图片，输出文本。

        仅多模态后端（如 Qwen2.5-VL）需要实现。
        纯文本后端此方法抛出 NotImplementedError。

        Args:
            prompt: 文本提示词
            image_path: 图片文件路径
            max_tokens: 最大生成 token 数

        Returns:
            生成的文本
        """
        raise NotImplementedError(
            f"{self.backend_type} 不支持多模态推理。"
        )

    @abstractmethod
    def logits(self, prompt: str) -> np.ndarray:
        """获取 Logits（用于评测，非生成场景）。

        对应 hbm_runtime.run() 的 Prefill 阶段。

        Args:
            prompt: 输入提示词

        Returns:
            shape (vocab_size,), dtype=float32
        """
        ...

    @property
    @abstractmethod
    def backend_type(self) -> str:
        """返回后端类型标识。"""
        ...

    @property
    def is_multimodal(self) -> bool:
        """是否支持多模态。"""
        return False
