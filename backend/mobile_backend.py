"""MobileBackend —— 手机端推理后端骨架。

Android 方案: ONNX Runtime / llama.cpp
iOS 方案: CoreML / MLC-LLM

与 hbm_runtime 的类比（体现经验迁移）：
- .hbm 编译 ⇄ .onnx / .mlmodelc 导出（离线编译步骤）
- hbm_runtime.run() ⇄ ORT.Run() / MLModel.prediction()（numpy feed 推理）
- Chunk Size=512 约束在手机端同样适用（内存/功耗优化）

PC 开发阶段请使用 LocalSimulateBackend。
"""

from typing import List

import numpy as np

from .base import InferenceBackend


class MobileBackend(InferenceBackend):
    """手机端推理后端。

    Android 推荐方案: ONNX Runtime
        1. 离线: PyTorch 模型 → torch.onnx.export() → .onnx
        2. 在线: ORTMobile 加载 .onnx → tokenizer 编码 → numpy feed → Run()

    iOS 推荐方案: CoreML
        1. 离线: PyTorch 模型 → coremltools.convert() → .mlmodelc
        2. 在线: MLModel 加载 → tokenizer 编码 → MLMultiArray feed → prediction
    """

    def __init__(
        self,
        embedding_model_path: str = "",
        llm_model_path: str = "",
        tokenizer_path: str = "",
        platform: str = "android",
        chunk_size: int = 512,
    ):
        self.chunk_size = chunk_size
        self.platform = platform
        self._embedding_model_path = embedding_model_path
        self._llm_model_path = llm_model_path

        raise NotImplementedError(
            f"MobileBackend 需要在 {platform} 设备上运行。"
            "PC 开发阶段请使用 LocalSimulateBackend。"
        )

    @property
    def backend_type(self) -> str:
        return f"mobile_{self.platform}"

    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError(f"需要在 {self.platform} 设备上运行")

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        raise NotImplementedError(f"需要在 {self.platform} 设备上运行")

    def logits(self, prompt: str) -> np.ndarray:
        raise NotImplementedError(f"需要在 {self.platform} 设备上运行")
