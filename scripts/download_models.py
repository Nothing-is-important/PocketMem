"""模型预下载脚本 —— 在离线前一次性下载所有 HuggingFace 模型。

国内用户：设置 HF_ENDPOINT 环境变量使用镜像站。
    # Windows PowerShell
    $env:HF_ENDPOINT = "https://hf-mirror.com"
    python scripts/download_models.py

    # 或者在命令行直接：
    set HF_ENDPOINT=https://hf-mirror.com && python scripts/download_models.py

下载内容：
1. BAAI/bge-small-zh-v1.5  —— Embedding 模型 (~100MB)
2. Qwen/Qwen2.5-1.5B-Instruct —— LLM (~3GB)

下载后的模型缓存在 ~/.cache/huggingface/hub/，
之后开发不需要联网。
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_endpoint():
    """检查 HF_ENDPOINT 是否设置了镜像。"""
    endpoint = os.environ.get("HF_ENDPOINT", "")
    if endpoint:
        print(f"  使用镜像: {endpoint}")
    else:
        print("  ⚠ 未设置 HF_ENDPOINT，直连 HuggingFace")
        print("  如果下载失败，请先设置镜像：")
        print("    set HF_ENDPOINT=https://hf-mirror.com")
        print("    然后重新运行此脚本\n")


def download_models():
    print("=" * 60)
    print("预下载 HuggingFace 模型")
    print("=" * 60)
    check_endpoint()

    # 1. Embedding 模型
    print("\n[1/2] 下载 Embedding 模型: BAAI/bge-small-zh-v1.5")
    print("  大小: ~100MB")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        emb = model.encode(["测试"])
        print(f"  ✓ 下载完成，维度: {emb.shape[1]}")
    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        print("  手动方案 1: 设置镜像后重试")
        print("    set HF_ENDPOINT=https://hf-mirror.com")
        print("  手动方案 2: ModelScope 下载")
        print("    https://modelscope.cn/models/BAAI/bge-small-zh-v1.5")

    # 2. LLM
    print("\n[2/2] 下载 LLM: Qwen/Qwen2.5-1.5B-Instruct")
    print("  大小: ~3GB，请耐心等待...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print("  下载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            trust_remote_code=True,
        )

        print("  下载模型权重...")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        param_count = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"  ✓ 下载完成，参数量: {param_count:.1f}B")
    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        print("  手动方案: ModelScope 下载")
        print("    pip install modelscope")
        print("    python -c \"from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')\"")

    # 验证缓存
    print("\n" + "=" * 60)
    print("验证模型缓存")

    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(cache_dir):
        total_size = 0
        file_count = 0
        for dirpath, _, filenames in os.walk(cache_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                    file_count += 1
                except OSError:
                    pass
        print(f"  缓存目录: {cache_dir}")
        print(f"  文件数: {file_count}")
        print(f"  缓存大小: {total_size / 1024**3:.1f}GB")
        if total_size > 100_000_000:  # >100MB
            print("  ✓ 模型已缓存，可离线使用")
        else:
            print("  ⚠ 缓存很小，可能下载不完整")
    else:
        print("  ⚠ 缓存目录不存在")

    print("\n完成！")


if __name__ == "__main__":
    download_models()
