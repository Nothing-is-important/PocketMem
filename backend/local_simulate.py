"""本地模拟端侧推理后端。

使用 sentence-transformers 和 transformers 在 PC 上模拟端侧推理行为。
- 模拟 hbm_runtime 的 numpy array 输入输出范式
- 可配置 Chunk Size，模拟端侧约束
- 记录推理延迟，为 Benchmark 提供基线
- 支持多模态模型（Qwen2.5-VL 系列）
"""

import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoTokenizer, GenerationConfig

from .base import InferenceBackend
from utils import get_logger

logger = get_logger("backend")


def _is_vl_model(model_name: str) -> bool:
    """检测是否为多模态 VL 模型。"""
    vl_keywords = ["VL", "vl", "vision", "Vision", "visual", "Visual"]
    name = str(model_name)
    return any(kw in name for kw in vl_keywords)


class LocalSimulateBackend(InferenceBackend):
    """本地模拟端侧推理后端。

    自动检测模型类型：
    - 纯文本 LLM：AutoTokenizer + AutoModelForCausalLM
    - 多模态 VL：AutoProcessor + AutoModelForCausalLM（支持图片输入）
    """

    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-small-zh-v1.5",
        llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        chunk_size: int = 512,
        device: str = "cuda",
    ):
        self.chunk_size = chunk_size
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self._is_vl = _is_vl_model(llm_model_name)

        if self.device == "cpu":
            logger.warning(
                "CUDA 不可用，模型将运行在 CPU 上。4B 模型在 CPU 上极慢（200s+/查询）。"
                "建议安装 CUDA 版 PyTorch 或使用 1.5B 小模型。"
            )

        logger.info("device=%s, chunk_size=%s", self.device, self.chunk_size)
        logger.info("multimodal=%s", "yes" if self._is_vl else "no")

        # ── Embedding 模型 ──
        logger.info("Loading embedding: %s", embedding_model_name)
        self._embedding_model = SentenceTransformer(
            embedding_model_name,
            device=self.device,
            trust_remote_code=True,
        )

        # ── LLM / VL 模型 ──
        logger.info("Loading LLM: %s", llm_model_name)
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        if self._is_vl:
            # 多模态 VL：bitsandbytes 4-bit 量化加载（临时方案）
            # 后续用 LLMC W4A8 替换
            from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

            self._vl_processor = AutoProcessor.from_pretrained(
                llm_model_name, trust_remote_code=True
            )
            self._llm_tokenizer = self._vl_processor.tokenizer

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self._llm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                llm_model_name,
                quantization_config=quant_config,
                trust_remote_code=True,
            )
            logger.info("VL model loaded with 4-bit quantization")
        else:
            from transformers import AutoModelForCausalLM
            self._vl_processor = None
            self._llm_tokenizer = AutoTokenizer.from_pretrained(
                llm_model_name, trust_remote_code=True
            )
            self._llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                dtype=torch_dtype,
                trust_remote_code=True,
            ).to(self.device)

        self._llm_model.eval()

        # Qwen3 检测：只有 Qwen3 支持思考模式
        model_type = getattr(self._llm_model.config, "model_type", "")
        self._supports_thinking = "qwen3" in str(model_type).lower()
        if not self._supports_thinking:
            logger.info("模型不是 Qwen3，思考模式已禁用")

        # 延迟统计
        self._last_embed_latency_ms = 0.0
        self._last_generate_latency_ms = 0.0

    @property
    def backend_type(self) -> str:
        return "local_simulate"

    @property
    def is_multimodal(self) -> bool:
        return self._is_vl

    # ═══════════════════════════════════════════════════════════════
    # Embedding
    # ═══════════════════════════════════════════════════════════════

    def embed(self, texts: List[str]) -> np.ndarray:
        t0 = time.time()

        embeddings = self._embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        self._last_embed_latency_ms = (time.time() - t0) * 1000
        return embeddings.astype(np.float32)

    # ═══════════════════════════════════════════════════════════════
    # 文本生成
    # ═══════════════════════════════════════════════════════════════

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        t0 = time.time()

        inputs = self._llm_tokenizer(prompt, return_tensors="pt").to(self.device)
        seq_len = inputs["input_ids"].shape[1]

        if seq_len > self.chunk_size:
            num_chunks = (seq_len - 1) // self.chunk_size + 1
            logger.debug(
                "Prefill chunking: seq_len=%s, chunk_size=%s, num_chunks=%s",
                seq_len, self.chunk_size, num_chunks
            )

        gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            eos_token_id=self._llm_tokenizer.eos_token_id,
            pad_token_id=self._llm_tokenizer.eos_token_id,
        )
        with torch.no_grad():
            outputs = self._llm_model.generate(**inputs, generation_config=gen_config)

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        result = self._llm_tokenizer.decode(generated_ids, skip_special_tokens=True)

        self._last_generate_latency_ms = (time.time() - t0) * 1000
        logger.info(
            "Generate: input_tokens=%s, output_tokens=%s, latency=%.1fms",
            seq_len, len(generated_ids), self._last_generate_latency_ms
        )

        return result

    def generate_stream(self, prompt: str, max_tokens: int = 512):
        """流式生成——逐 token yield 文本片段。

        使用 transformers TextIteratorStreamer，生成线程在后台运行，
        主线程通过迭代器获取增量文本。

        Yields:
            str: 增量文本片段（每次若干 token）
        """
        from threading import Thread

        try:
            from transformers import TextIteratorStreamer
        except ImportError:
            full = self.generate(prompt, max_tokens)
            yield full
            return

        t0 = time.time()
        inputs = self._llm_tokenizer(prompt, return_tensors="pt").to(self.device)
        seq_len = inputs["input_ids"].shape[1]

        streamer = TextIteratorStreamer(
            self._llm_tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            eos_token_id=self._llm_tokenizer.eos_token_id,
            pad_token_id=self._llm_tokenizer.eos_token_id,
        )

        generation_kwargs = dict(
            **inputs,
            generation_config=gen_config,
            streamer=streamer,
        )

        thread = Thread(target=self._llm_model.generate, kwargs=generation_kwargs)
        thread.start()

        total_text = ""
        for new_text in streamer:
            total_text += new_text
            yield new_text

        thread.join()
        self._last_generate_latency_ms = (time.time() - t0) * 1000
        logger.info(
            "GenerateStream: input_tokens=%s, output_chars=%s, latency=%.1fms",
            seq_len, len(total_text), self._last_generate_latency_ms
        )

    def generate_with_thinking(self, messages: list, max_tokens: int = 512):
        """Qwen3 思考模式生成 —— 真正的 token-by-token 流式。

        非 Qwen3 模型不支持思考模式，自动降级为标准流式生成。
        """
        # 非 Qwen3 模型：降级为标准生成
        if not self._supports_thinking:
            from agent.generator import build_thinking_messages
            prompt = self._build_prompt_from_messages(messages)
            for text_chunk in self.generate_stream(prompt, max_tokens):
                yield ("answer", text_chunk)
            return

        from threading import Thread

        try:
            from transformers import TextIteratorStreamer
        except ImportError:
            # 降级：一次性生成
            thinking, answer = self._generate_with_thinking_sync(messages, max_tokens)
            if thinking:
                yield ("think", thinking)
            if answer:
                yield ("answer", answer)
            return

        t0 = time.time()

        text = self._llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = self._llm_tokenizer(text, return_tensors="pt").to(self.device)

        eos_ids = self._llm_tokenizer.eos_token_id
        pad_id = eos_ids if isinstance(eos_ids, int) else (
            eos_ids[0] if isinstance(eos_ids, list) else 151645
        )

        # skip_special_tokens=False → </think> 会作为普通文本出现在流中
        streamer = TextIteratorStreamer(
            self._llm_tokenizer,
            skip_prompt=True,
            skip_special_tokens=False,
        )

        gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            eos_token_id=eos_ids,
            pad_token_id=pad_id,
        )

        generation_kwargs = dict(**inputs, generation_config=gen_config, streamer=streamer)
        thread = Thread(target=self._llm_model.generate, kwargs=generation_kwargs, daemon=True)
        thread.start()

        # 流式处理：在 buffer 中检测 </think> 分隔符
        in_thinking = True
        buffer = ""
        THINK_END = "</think>"
        CHATML_TOKENS = ("<|im_end|>", "<|im_start|>", "<|endoftext|>")

        def _clean(text: str) -> str:
            """过滤 ChatML 特殊 token。"""
            for tok in CHATML_TOKENS:
                text = text.replace(tok, "")
            return text

        for new_text in streamer:
            buffer += new_text

            if in_thinking and THINK_END in buffer:
                idx = buffer.index(THINK_END)
                think_part = _clean(buffer[:idx])
                if think_part.strip():
                    yield ("think", think_part)
                in_thinking = False
                buffer = _clean(buffer[idx + len(THINK_END):])
                if buffer.strip():
                    yield ("answer", buffer)
                buffer = ""
            elif in_thinking:
                if buffer.strip():
                    cleaned = _clean(buffer)
                    if cleaned.strip():
                        yield ("think", cleaned)
                    buffer = ""
            else:
                if buffer.strip():
                    cleaned = _clean(buffer)
                    if cleaned.strip():
                        yield ("answer", cleaned)
                    buffer = ""

        # 刷新剩余 buffer
        buffer = _clean(buffer)
        if buffer.strip():
            yield ("answer" if not in_thinking else "think", buffer)

        thread.join()
        self._last_generate_latency_ms = (time.time() - t0) * 1000
        logger.info(
            "GenerateStreamThinking: latency=%.1fms", self._last_generate_latency_ms
        )

    def _build_prompt_from_messages(self, messages: list) -> str:
        """Build standard prompt string from chat messages (fallback)."""
        parts = []
        for msg in messages:
            content = msg.get("content", "")
            parts.append(content)
        return "\n\n".join(parts)

    # ═══════════════════════════════════════════════════════════════
    # 多模态生成（图片 + 文本 → 文本）
    # ═══════════════════════════════════════════════════════════════
    # 多模态生成（图片 + 文本 → 文本）
    # ═══════════════════════════════════════════════════════════════
    # 多模态生成（图片 + 文本 → 文本）
    # ═══════════════════════════════════════════════════════════════

    def generate_with_image(
        self,
        prompt: str,
        image_path: str,
        max_tokens: int = 512,
    ) -> str:
        if not self._is_vl:
            raise NotImplementedError(
                "当前后端不支持多模态推理。"
                "请使用 VL 模型（如 Qwen2.5-VL-7B-Instruct）。"
            )

        t0 = time.time()

        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        # Qwen2.5-VL 的对话模板
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._vl_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._vl_processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self._llm_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # 1.5B 小模型采样模式输出失控，保持贪心解码
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                eos_token_id=self._llm_tokenizer.eos_token_id,
                pad_token_id=self._llm_tokenizer.eos_token_id,
            )

        # 切除输入部分
        input_len = inputs["input_ids"].shape[1]
        generated_ids_trimmed = generated_ids[0][input_len:]
        result = self._llm_tokenizer.decode(
            generated_ids_trimmed, skip_special_tokens=True
        )

        self._last_generate_latency_ms = (time.time() - t0) * 1000
        logger.info(
            "VL Generate: image=%s, output_tokens=%s, latency=%.1fms",
            Path(image_path).name, len(generated_ids_trimmed), self._last_generate_latency_ms
        )

        return result

    # ═══════════════════════════════════════════════════════════════
    # Logits + 工具
    # ═══════════════════════════════════════════════════════════════

    def logits(self, prompt: str) -> np.ndarray:
        t0 = time.time()

        inputs = self._llm_tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._llm_model(**inputs)
            logits = outputs.logits[0, -1, :].cpu().numpy()

        latency = (time.time() - t0) * 1000
        logger.info(
            "Logits: prompt_len=%s, latency=%.1fms",
            inputs['input_ids'].shape[1], latency
        )

        return logits.astype(np.float32)

    def get_last_latency(self) -> dict:
        return {
            "embed_ms": self._last_embed_latency_ms,
            "generate_ms": self._last_generate_latency_ms,
        }
