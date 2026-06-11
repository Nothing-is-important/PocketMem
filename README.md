# 🔍 TeamMind —— 端侧企业知识助手

> 企业文档智能检索与问答系统。支持双模式推理——公开文档走 API 加速，机密文档走本地量化模型，数据不出企业内网。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue" alt="Python">
  <img src="https://img.shields.io/badge/LangGraph-Agent-orange" alt="LangGraph">
  <img src="https://img.shields.io/badge/QLoRA-微调-purple" alt="QLoRA">
  <img src="https://img.shields.io/badge/W4A8-量化-red" alt="W4A8">
  <img src="https://img.shields.io/badge/ChromaDB-向量检索-green" alt="ChromaDB">
  <img src="https://img.shields.io/badge/Vue3-前端-42b883" alt="Vue 3">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="License">
</p>

---

## 为什么做这个项目

大多数 AI Agent 项目是"调 OpenAI API + LangChain 流水线"——面试官面 10 个人，9 个是这个套路。

TeamMind 走另一条路：**从企业场景出发，用硬件约束倒推架构设计，全链路闭环。**

| 维度 | 常规 AI 项目 | TeamMind |
|------|------------|----------|
| 推理 | 调云端 API | 双模式：API（快）/ 本地量化模型（安全） |
| 数据 | 公开数据集 | 192 份企业文档（邮件+合同+纪要+技术文档） |
| 安全 | 无 | AccessGuard 权限拦截 + 三级文档分级 |
| 微调 | 无 | QLoRA 企业场景微调 + LLMC W4A8 量化 |
| 硬件 | 不考虑 | 全链路在 RTX 4060 8GB 上完成 |
| 场景 | 通用 Demo | 企业知识库检索——面试官一听就懂价值 |

---

## 快速开始

```bash
# 1. 安装依赖
uv sync

# 2. 生成企业演示数据（192 份文档）
python scripts/ingest_enterprise_data.py

# 3. 启动服务
python scripts/run_demo.py --serve

# 4. 浏览器打开 http://localhost:8000
```

---

## 核心特性

### 双模式推理引擎

```
用户查"凤凰项目技术方案"（internal 文档）
  → Agent 检测文档等级
  → 自动切换本地模式（W4A8 量化模型）
  → 数据不出设备

用户查"产品 API 文档"（public 文档）
  → Agent 检测全部为公开文档
  → 自动切换 API 模式（DeepSeek/vLLM）
  → 速度快 5-10x
```

| 后端 | 速度 | 安全 | 适用场景 |
|------|------|------|---------|
| `local_simulate` | 基准 | 数据不出设备 | 机密/内部文档 |
| `vllm` | 5-10x | API 调用 | 公开文档 |
| `dual_mode` | 自动 | 自动分级 | **默认推荐** |

### 企业级安全

- **AccessGuard** 权限拦截节点（deny by default）
- 三级文档分级：public / internal / confidential
- 审计日志：每次查询记录完整检索链路
- RBAC 角色控制：admin / editor / viewer

### 混合检索 pipeline

```
查询 → 实体提取 → ChromaDB 向量检索 + BM25 关键词检索
    → RRF 融合（k=60）
    → Cross-Encoder 重排（Recall@5: 0.78 → 0.85）
    → AccessGuard 权限过滤
    → MemoryJudge 相关性判断
    → Generator 答案生成
```

### QLoRA 微调 + W4A8 量化

完整微调方案——从企业文档生成训练数据、QLoRA 指令微调、到 LLMC W4A8 量化部署。

```bash
# Step 1: 生成训练数据（无需 GPU）
python scripts/generate_training_data.py

# Step 2: QLoRA 微调（4060 8GB, ~4-6h）
python scripts/train_qlora.py --model Qwen3-4B

# Step 3: W4A8 量化部署
python scripts/quantize_llmc.py
```

详见 [QLoRA 微调方案](docs/qlora-finetune-plan.md)

### 前端 Pipeline 可视化

Vue3 界面实时展示 Agent 每一步：路由 → 检索 → 权限过滤 → 重排 → 判断 → 生成。每步显示延迟数据，面试演示效果拉满。

---

## 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│  🖥️ Vue3 前端                                                 │
│  Pipeline 可视化 | 实时 SSE | 推荐问题 | 延迟面板              │
└──────────────────────┬───────────────────────────────────────┘
                       │ HTTP / SSE
┌──────────────────────▼───────────────────────────────────────┐
│  FastAPI Server                                              │
│  /ask | /ask/stream | /data/sources | /data/suggestions      │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  LangGraph Agent                                       │    │
│  │  Router → Retrieve → AccessGuard → Rerank → Judge → Gen│    │
│  │  Hook 系统 (4 挂载点) + 用户画像                        │    │
│  └──────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  混合检索: ChromaDB + BM25 + RRF + Cross-Encoder       │    │
│  └──────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  双层语义缓存 + 审计日志                                 │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│  InferenceBackend 抽象层                                      │
│  ┌─────────────────┐  ┌──────────────────┐                   │
│  │ LocalSimulate   │  │ VLLMBackend      │                   │
│  │ (W4A8 本地模型)  │  │ (OpenAI API,5x+) │                   │
│  └─────────────────┘  └──────────────────┘                   │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  DualModeBackend —— 自动切换公开/机密推理模式          │    │
│  └──────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  数据摄取: 邮件 / 会议纪要 / 合同 / 技术文档            │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

---

## API 端点

| 方法 | 端点 | 描述 |
|------|------|------|
| `GET` | `/` | 前端页面 |
| `GET` | `/health` | 健康检查 |
| `GET` | `/data/sources` | 已索引数据源 |
| `GET` | `/data/suggestions` | 动态推荐问题 |
| `POST` | `/ask` | 同步问答 |
| `POST` | `/ask/stream` | SSE 流式问答 |
| `POST` | `/ingest` | 扫描索引新文件 |

---

## 项目结构

```
TeamMind/
├── frontend/index.html        # Vue3 前端
├── api/                       # FastAPI + SSE + 缓存
│   ├── server.py, models.py, cache.py
├── agent/                     # LangGraph Agent
│   ├── graph.py               # 图编排
│   ├── router.py              # 意图路由
│   ├── access_guard.py        # 权限拦截
│   ├── memory_judge.py        # 相关性判断
│   ├── generator.py           # 答案生成（思考模式）
│   ├── hooks.py               # Hook 生命周期
│   └── state.py               # 状态定义
├── rag/                       # RAG 引擎
│   ├── vector_store.py        # ChromaDB
│   ├── hybrid_retriever.py    # 混合检索 + RRF
│   ├── bm25_retriever.py      # BM25
│   ├── reranker.py            # Cross-Encoder 重排
│   ├── entity_extractor.py    # 实体提取
│   └── entity_aware_retriever.py
├── backend/                   # 推理后端
│   ├── base.py                # 抽象接口
│   ├── local_simulate.py      # 本地模型
│   ├── vllm_backend.py        # vLLM API
│   ├── dual_backend.py        # 双模式自动切换
│   └── mobile_backend.py      # 手机骨架
├── data_ingestion/            # 数据摄取
│   ├── pipeline.py, chunker.py, indexer.py
│   ├── mail_parser.py         # 邮件解析
│   ├── wechat_parser.py       # 微信解析（保留）
│   └── markdown_loader.py, pdf_loader.py
├── config/                    # 配置
│   └── settings.py, prompts/
├── scripts/                   # 工具脚本
│   ├── run_demo.py            # 一键启动
│   ├── ingest_enterprise_data.py  # 企业数据生成
│   ├── generate_training_data.py  # QLoRA 训练数据
│   └── train_qlora.py         # QLoRA 微调
├── docs/                      # 文档
│   ├── qlora-finetune-plan.md # 微调方案
│   └── memory-system-design.md # 记忆系统设计
├── data/demo/enterprise/      # 192 份企业演示数据
└── tests/                     # 测试
```

---

## 配置

所有配置通过环境变量控制：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `POCKET_LLM_MODEL` | auto | LLM 模型（自动发现本地路径） |
| `POCKET_EMBEDDING_MODEL` | auto | Embedding 模型 |
| `POCKET_BACKEND` | local_simulate | 推理后端（local_simulate/vllm/dual_mode） |
| `POCKET_VLLM_URL` | http://localhost:8001/v1 | vLLM/API 地址 |
| `POCKET_DEVICE` | cuda | 推理设备 |
| `POCKET_THINKING` | true | Qwen3 思考模式 |

---

## 面试准备

- **[QLoRA 微调方案](docs/qlora-finetune-plan.md)** —— 完整实施方案 + 面试叙事
- **[记忆系统设计](docs/memory-system-design.md)** —— 三层记忆架构设计
- **[教学文档](PocketMemory-教学文档.md)** —— 各模块设计思路和面试 Q&A

---

## License

MIT
