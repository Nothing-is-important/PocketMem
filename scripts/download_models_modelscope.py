"""从 ModelScope 下载模型到本地（国内用户推荐，速度快）。

ModelScope 不需要翻墙，阿里云节点，下载速度通常 10-50MB/s。

下载内容：
1. BAAI/bge-small-zh-v1.5        —— Embedding 模型 (~100MB)
2. Qwen/Qwen2.5-VL-7B-Instruct    —— 多模态 LLM (~15GB)

存储位置：F:/Models/
    F:/Models/bge-small-zh-v1.5/
    F:/Models/Qwen2.5-VL-7B-Instruct/

下载后代码直接用本地路径加载，不再依赖网络。
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

MODELS_DIR = "F:/Models"


def download_via_modelscope(model_id: str, local_name: str):
    """从 ModelScope 下载模型到本地目录。"""
    from modelscope import snapshot_download

    target = os.path.join(MODELS_DIR, local_name)

    if os.path.exists(target) and os.listdir(target):
        size = _dir_size(target)
        print(f"  ✓ 已存在: {target} ({size:.0f}MB)")
        return target

    print(f"  下载中: {model_id} → {target}")
    try:
        downloaded = snapshot_download(model_id, cache_dir=MODELS_DIR)
        if downloaded != target:
            import shutil
            if os.path.exists(target):
                shutil.rmtree(target)
            shutil.move(downloaded, target)
        size = _dir_size(target)
        print(f"  ✓ 完成: {target} ({size:.0f}MB)")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        print(f"  手动: https://modelscope.cn/models/{model_id}")
        return None

    return target


def _dir_size(path: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total / (1024 * 1024)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 60)
    print("从 ModelScope 下载模型")
    print(f"目标目录: {MODELS_DIR}")
    print("=" * 60)

    # 1. BGE Embedding（文本 → 向量）
    print("\n[1/2] Embedding 模型: bge-small-zh-v1.5 (~100MB)")
    download_via_modelscope("BAAI/bge-small-zh-v1.5", "bge-small-zh-v1.5")

    # 2. Qwen2.5-VL 多模态 LLM（文本生成 + 图片理解）
    print("\n[2/2] 多模态 LLM: Qwen2.5-VL-7B-Instruct (~15GB)")
    print("  这个比较大，请耐心等待...")
    print("  包含: ViT 图像编码器 + Qwen2.5-7B LLM backbone")
    download_via_modelscope(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen2.5-VL-7B-Instruct",
    )

    print("\n" + "=" * 60)
    total = _dir_size(MODELS_DIR)
    print(f"下载完成！总大小: {total:.0f}MB")
    print(f"位置: {MODELS_DIR}")
    print()
    print("模型清单:")
    print(f"  {MODELS_DIR}/bge-small-zh-v1.5/       ← Embedding")
    print(f"  {MODELS_DIR}/Qwen2.5-VL-7B-Instruct/  ← 多模态 LLM")
    print()
    print("下一步: LLMC 量化 LLM backbone (W4A8)，ViT 保持 FP16")
    print("量化后显存: ViT ~1.5GB + LLM ~3.5GB + KV Cache ~2GB = ~7GB")


if __name__ == "__main__":
    main()
