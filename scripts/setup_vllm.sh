#!/bin/bash
# PocketMemory vLLM 环境安装脚本
# 在 WSL2 终端中运行: bash setup_vllm.sh

set -e

echo "=== Step 1: 安装 uv（快速 Python 包管理器）==="
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv $(uv --version)"

echo ""
echo "=== Step 2: 创建虚拟环境 ==="
python3 -m venv ~/vllm-env
echo "虚拟环境: ~/vllm-env"

echo ""
echo "=== Step 3: 安装 vLLM（约 3GB，需要 5-15 分钟）==="
uv pip install vllm --python ~/vllm-env/bin/python

echo ""
echo "=== Step 4: 验证安装 ==="
~/vllm-env/bin/python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

~/vllm-env/bin/python -c "import vllm; print(f'vLLM: {vllm.__version__}')"

echo ""
echo "=== 完成！==="
echo ""
echo "启动 vLLM Server（模型路径按实际调整）："
echo "  ~/vllm-env/bin/vllm serve /mnt/f/Models/Qwen3-4B --port 8001 --gpu-memory-utilization 0.9"
echo ""
echo "Windows 端启动 FastAPI："
echo "  uv run python scripts/run_demo.py --serve --backend vllm"
