"""生成模拟微信聊天数据用于演示。

生成 3 个月跨度的 200+ 条消息，涵盖 5+ 个联系人，
包含多种自然场景：项目讨论、聚餐计划、技术问答、旅行规划等。
"""

import os
import random
from datetime import datetime, timedelta
from pathlib import Path

# ── 模拟联系人 ──
CONTACTS = {
    "张三": {
        "relation": "同事",
        "topics": ["项目进度", "技术讨论", "聚餐", "团建"],
    },
    "李四": {
        "relation": "前同事",
        "topics": ["跳槽经验", "行业动态", "投资理财"],
    },
    "王五": {
        "relation": "大学同学",
        "topics": ["游戏", "篮球", "怀旧", "聚会"],
    },
    "赵六": {
        "relation": "技术群友",
        "topics": ["大模型", "量化推理", "开源项目", "论文讨论"],
    },
    "陈阿姨": {
        "relation": "家人",
        "topics": ["家庭", "健康", "天气", "节日"],
    },
}

# ── 场景对话模板 ──
SCENARIOS = {
    "聚餐": [
        ("{sender}", "周末有空吗？一起吃个饭？"),
        ("{other}", "好啊，去哪？"),
        ("{sender}", "上次那家火锅店怎么样？渝味火锅，在朝阳区建国路88号"),
        ("{other}", "可以，几点？"),
        ("{sender}", "周六中午12点吧"),
        ("{other}", "好的，我叫上{third}一起"),
        ("{sender}", "行，那周六见"),
    ],
    "项目讨论": [
        ("{sender}", "新版本的需求文档你看完了吗？"),
        ("{other}", "看完了，有几个地方需要讨论"),
        ("{sender}", "主要是{feature}这部分，时间比较紧"),
        ("{other}", "我觉得可以先做MVP，{feature}放到下个迭代"),
        ("{sender}", "也行，我去跟产品沟通一下deadline"),
        ("{other}", "好的，我先把接口文档整理出来"),
    ],
    "技术交流": [
        ("{sender}", "最近在看{tech}，感觉很有前景"),
        ("{other}", "确实，我们也在调研这个方向"),
        ("{sender}", "你们打算落地到什么场景？"),
        ("{other}", "主要是{direction}，还在做POC"),
        ("{sender}", "回头可以交流一下经验"),
    ],
    "团建讨论": [
        ("{sender}", "下个月团建大家有什么想法？"),
        ("{other}", "上次去的密云水库太远了"),
        ("{third}", "我投古北水镇一票"),
        ("{sender}", "或者十渡？听说那边漂流不错"),
        ("{other}", "我都行，看预算"),
        ("{sender}", "那我建个投票吧，三个选项：密云水库、古北水镇、十渡"),
    ],
    "面试经验": [
        ("{sender}", "上周面试了{company}，面了三轮"),
        ("{other}", "结果怎么样？"),
        ("{sender}", "拿到offer了，总包比现在高30%"),
        ("{other}", "恭喜！什么方向？"),
        ("{sender}", "做大模型推理优化的，正好是我感兴趣的方向"),
        ("{other}", "那挺好的，准备去吗？"),
    ],
}

# ── 独立消息（非对话） ──
SOLO_MESSAGES = [
    "{sender}: 今天加班，晚点到",
    "{sender}: 项目deadline延期到下周了",
    "{sender}: 推荐一个开源项目：{repo}，值得一看",
    "{sender}: 感冒了，请假一天",
    "{sender}: 周末去了{place}，风景不错",
    "{sender}: 最近在学习{skill}，有一起的吗？",
    "{sender}: 分享一篇文章：{article_title}",
    "{sender}: 周末约了{activity}，要一起吗？",
    "{sender}: 下周三{company}来我们公司做技术交流",
    "{sender}: 团建时间定了，{date}去{place}",
]

COMPANIES = ["字节", "阿里", "腾讯", "百度", "华为", "小米", "蔚来", "快手", "小红书", "大疆"]
PLACES = ["密云水库", "古北水镇", "十渡", "香山", "颐和园", "798", "三里屯", "后海"]
TECHS = ["LangGraph", "RAG", "Agent", "量化推理", "ONNX Runtime", "ChromaDB", "vLLM", "LoRA"]
REPOS = ["vllm-project/vllm", "langchain-ai/langgraph", "huggingface/transformers", "chroma-core/chromadb"]
SKILLS = ["CUDA编程", "ONNX模型导出", "Prompt Engineering", "分布式训练", "K8s"]
ARTICLES = [
    "DeepSeek-V3技术报告解读",
    "端侧大模型部署的挑战与方案",
    "RAG系统设计最佳实践",
    "W4A8量化在移动端的应用",
]
ACTIVITIES = ["打篮球", "看电影", "爬山", "桌游", "唱K", "密室逃脱"]


def generate_demo_data(output_dir: str = "data/demo"):
    """生成演示数据集。"""
    os.makedirs(output_dir, exist_ok=True)

    random.seed(42)
    now = datetime.now()
    start_date = now - timedelta(days=90)

    # ── 生成私聊对话 ──
    for contact_name, info in CONTACTS.items():
        messages = []
        current_date = start_date

        # 随机分布对话
        msg_count = random.randint(25, 50)
        for _ in range(msg_count):
            current_date += timedelta(
                hours=random.randint(2, 72),
                minutes=random.randint(0, 59),
            )

            # 70% 场景对话，30% 独立消息
            if random.random() < 0.7:
                scenario_name = random.choice(list(SCENARIOS.keys()))
                scenario = SCENARIOS[scenario_name]

                other_contacts = [c for c in CONTACTS if c != contact_name]
                other = random.choice(other_contacts)
                third = random.choice([c for c in other_contacts if c != other])

                for sender_role, template in scenario:
                    ts = current_date + timedelta(minutes=random.randint(0, 5))

                    # sender_role 是 "{sender}" / "{other}" / "{third}" 角色标记
                    role_map = {"{sender}": contact_name, "{other}": other, "{third}": third}
                    actual_sender = role_map.get(sender_role, contact_name)

                    content = template.format(
                        sender=contact_name,
                        other=other,
                        third=third,
                        feature=random.choice(["用户权限管理", "支付流程优化", "搜索功能重构", "推送系统升级"]),
                        tech=random.choice(TECHS),
                        direction=random.choice(["智能客服", "代码助手", "数据分析", "内容推荐"]),
                        company=random.choice(COMPANIES),
                        date=current_date.strftime("%m月%d日"),
                    )
                    messages.append((current_date, actual_sender, content))
            else:
                template = random.choice(SOLO_MESSAGES)
                content = template.format(
                    sender=contact_name,
                    repo=random.choice(REPOS),
                    place=random.choice(PLACES),
                    skill=random.choice(SKILLS),
                    article_title=random.choice(ARTICLES),
                    activity=random.choice(ACTIVITIES),
                    company=random.choice(COMPANIES),
                    date=current_date.strftime("%m月%d日"),
                )
                messages.append((current_date, contact_name, content))

        # 按时间排序并写入文件
        messages.sort(key=lambda x: x[0])

        filepath = os.path.join(output_dir, f"{contact_name}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            for ts, sender, content in messages:
                f.write(f"{ts:%Y-%m-%d %H:%M:%S} {sender}\n")
                f.write(f"{content}\n\n")

        print(f"  [OK] {contact_name}.txt: {len(messages)} 条消息")

    # ── 生成 Markdown 笔记 ──
    notes_dir = os.path.join(output_dir, "notes")
    os.makedirs(notes_dir, exist_ok=True)

    notes = {
        "Transformer架构学习笔记.md": """# Transformer 架构学习笔记

## 注意力机制

自注意力（Self-Attention）是 Transformer 的核心。计算公式：

$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$

### 多头注意力

多头注意力通过多个注意力头并行计算，每个头关注不同的表示子空间。

- 头数通常设为 8 或 16
- 每个头的维度 = d_model / num_heads

## 位置编码

Transformer 本身不具序列感知能力，需要位置编码补充位置信息。

### 正弦位置编码

使用不同频率的正弦和余弦函数。

### 可学习位置编码

像 BERT 和 GPT 一样学习位置嵌入。

## 前馈网络

每个注意力层后接一个两层全连接网络，中间使用 ReLU 或 GELU 激活。

## Layer Normalization

- Post-LN：原始 Transformer 的做法（注意力 + 残差 + LN）
- Pre-LN：先 LN 再注意力/FFN，训练更稳定

## 实际应用

- BERT：仅使用 Encoder，适合理解任务
- GPT：仅使用 Decoder，适合生成任务
- T5：Encoder-Decoder 架构，适合 Seq2Seq 任务
""",
        "端侧推理优化笔记.md": """# 端侧推理优化笔记

## 模型量化

### W4A8 量化方案

使用 LLMC 框架对 Qwen3-4B 做后训练量化（PTQ）：

- 方法：QuRot + LWC + GPTQ
- 权重：4-bit，激活：8-bit
- 困惑度损失：2%-5%
- 模型压缩率：50%

### 量化感知训练 vs 后训练量化

- QAT：训练时模拟量化，精度高但需要重新训练
- PTQ：训练后直接量化，速度快但需要校准数据

## 推理优化

### Chunk Prefill

将长序列分块处理，Chunk Size=512：
- 减少峰值内存
- 提高吞吐量
- 适合端侧设备

### KV Cache 管理

- 连续内存分配
- 动态扩展策略
- 内存复用

## 框架对比

| 框架 | 平台 | 推理速度 | 量化支持 |
|------|------|----------|----------|
| ONNX Runtime | 跨平台 | 中等 | INT8/FP16 |
| llama.cpp | CPU/GPU | 快 | Q4/Q5/Q8 |
| MLC-LLM | 移动端 | 很快 | W4A16 |
| MediaPipe | Android | 快 | FP16 |

## 实践经验

在 S600 芯片上做推理评测的经验：
1. Chunk Size=512 在端侧是内存/功耗的较优平衡点
2. KV Cache 复用可以节省 30% 推理时间
3. Embedding 模型 Prefill 优化可将 Token 延迟从 56ms 降至 40ms
""",
        "Agent开发学习笔记.md": """# AI Agent 开发学习笔记

## Agent 架构模式

### ReAct 模式

Reasoning + Acting，交替进行推理和行动。

### Plan-Execute 模式

先制定计划，再逐步执行。

### Multi-Agent 协作

- 分层式：一个主 Agent 分配任务给子 Agent
- 对话式：多个 Agent 通过对话协作
- 辩论式：多个 Agent 从不同角度讨论后达成共识

## LangGraph 核心概念

### StateGraph

有向图，节点接收 state 并返回 state。

### 条件边

根据 state 内容动态路由到不同节点。

### Checkpointing

自动保存中间状态，支持人机交互和断点续传。

## RAG 检索增强

### 混合检索

- 向量检索：语义相似度
- BM25：关键词匹配
- RRF 融合：score = 1/(k+rank)，k=60

### 重排序

检索后使用 Cross-Encoder 重新排序，提高准确率。

### 时间感知

对个人记忆场景，添加时间衰减权重：
score_final = score_rrf * exp(-λ * days_since)

## PTRM 推理增强

基于 arXiv:2605.19943 的多轨迹推理：
- 在推理过程中注入高斯噪声（σ ≈ 0.01-0.05）
- 并行运行 K 条轨迹
- 投票/Q head 选择最优结果
- 意图分类准确率提升 ~7%
""",
    }

    for filename, content in notes.items():
        filepath = os.path.join(notes_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  [OK] notes/{filename}")

    print(f"\n演示数据生成完毕！输出目录: {output_dir}")
    print(f"运行: python scripts/ingest_demo_data.py")


if __name__ == "__main__":
    generate_demo_data()
