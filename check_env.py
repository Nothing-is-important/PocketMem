import torch
import sys

print("=" * 50)
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA (PyTorch):", torch.version.cuda)
    print("cuDNN:", torch.backends.cudnn.version())
    print("GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_capability(i)})")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️ 未检测到 CUDA GPU，将使用 CPU 模式运行")

# 验证关键库
for lib in ["transformers", "sentence_transformers", "langgraph", "chromadb"]:
    try:
        __import__(lib.replace("-", "_"))
        print(f"✓ {lib}: OK")
    except ImportError:
        print(f"✗ {lib}: NOT INSTALLED")
print("=" * 50)