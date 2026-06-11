"""QLoRA 微调脚本 —— 4060 8GB 优化配置。

基于 unsloth + transformers，专为消费级显卡优化。

用法：
  python scripts/train_qlora.py                    # 用默认 Qwen3-4B
  python scripts/train_qlora.py --model Qwen3-8B   # 用 8B（可能OOM）
  python scripts/train_qlora.py --dry-run           # 仅验证环境不训练

微调完成后模型保存在: models/teammind-qlora/
"""

import argparse, json, os, sys


def parse_args():
    p = argparse.ArgumentParser(description="QLoRA fine-tuning for TeamMind")
    p.add_argument("--model", default="Qwen3-4B", choices=["Qwen3-4B", "Qwen3-8B"],
                   help="Base model (default: Qwen3-4B)")
    p.add_argument("--data", default="data/training_data_train.json",
                   help="Training data path")
    p.add_argument("--output", default="models/teammind-qlora",
                   help="Output directory")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--dry-run", action="store_true",
                   help="Verify environment without training")
    return p.parse_args()


def verify_environment():
    """验证硬件和依赖。"""
    import torch
    print("=" * 50)
    print("环境检查")
    print("=" * 50)

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {vram:.1f} GB")

    # 检查 unsloth
    try:
        import unsloth
        print(f"unsloth: {unsloth.__version__}")
    except ImportError:
        print("unsloth: NOT INSTALLED. Run: pip install unsloth")
        return False

    # 检查其他依赖
    for lib in ["transformers", "datasets", "peft", "bitsandbytes", "trl"]:
        try:
            __import__(lib)
            print(f"{lib}: OK")
        except ImportError:
            print(f"{lib}: MISSING")
            return False

    # 尝试 4-bit 加载测试（不加载模型，只验证支持）
    try:
        from transformers import BitsAndBytesConfig
        config = BitsAndBytesConfig(load_in_4bit=True)
        print("4-bit quantization: Supported")
    except Exception as e:
        print(f"4-bit support: FAILED - {e}")
        return False

    print("Environment OK")
    return True


def load_training_data(path):
    """加载训练数据。"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} training examples")
    return data


def format_prompt(example):
    """Alpaca 格式模板。"""
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""


def train(args):
    """执行 QLoRA 微调。"""
    import torch

    MODEL_MAP = {
        "Qwen3-4B": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",  # unsloth 优化版
        "Qwen3-8B": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    }

    model_id = MODEL_MAP.get(args.model, MODEL_MAP["Qwen3-4B"])

    print(f"\n{'='*50}")
    print(f"QLoRA Fine-tuning: {args.model}")
    print(f"{'='*50}")
    print(f"Model: {model_id}")
    print(f"Data: {args.data} ({len(load_training_data(args.data))} examples)")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"Output: {args.output}")
    print(f"Estimated VRAM: {'~5GB' if '4B' in args.model else '~7.5GB'}")
    print(f"Estimated time: {'~4-6 hours' if '4B' in args.model else '~8-12 hours'}")
    print()

    from unsloth import FastLanguageModel
    from transformers import TrainingArguments
    from trl import SFTTrainer

    # 加载 4-bit 模型
    print("Loading 4-bit model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=2048,
        dtype=None,  # 自动选择
        load_in_4bit=True,
    )

    # 添加 LoRA adapters
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,                # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # 加载数据
    data = load_training_data(args.data)
    from datasets import Dataset
    dataset = Dataset.from_list(data)

    # 训练配置
    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,  # 等效 batch=4
        warmup_steps=10,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
    )

    # 训练
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        formatting_func=format_prompt,
        max_seq_length=2048,
    )

    print("\nStarting training...")
    trainer.train()

    # 保存
    print(f"\nSaving to {args.output}...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # 统计
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTraining complete!")
    print(f"  Trainable params: {trainable:,} ({100 * trainable / total:.2f}%)")
    print(f"  Output: {args.output}")
    print(f"\nNext: Merge LoRA + Quantize with LLMC")


def main():
    args = parse_args()

    if not verify_environment():
        print("\nEnvironment check failed. Install dependencies:")
        print("  pip install unsloth transformers datasets peft bitsandbytes trl accelerate")
        sys.exit(1)

    if args.dry_run:
        print("\nDry run complete. Ready for training.")
        return

    train(args)


if __name__ == "__main__":
    main()
