# 🧠 PocketMemory —— 你的第二大脑

> 微信聊天记录太多找不到关键信息？笔记散落各处难以检索？  
> PocketMemory 把你的聊天记录和笔记变成**可搜索的个人记忆库**——本地运行，数据不出设备。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue" alt="Python">
  <img src="https://img.shields.io/badge/LangGraph-Agent-orange" alt="LangGraph">
  <img src="https://img.shields.io/badge/Qwen3-思考模式-purple" alt="Qwen3">
  <img src="https://img.shields.io/badge/ChromaDB-向量检索-green" alt="ChromaDB">
  <img src="https://img.shields.io/badge/Vue3-前端-42b883" alt="Vue 3">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="License">
</p>

---

## ✨ 为什么选 PocketMemory？

| 你的痛点 | PocketMemory 的解法 |
|---------|-------------------|
| "上次张三推荐的那家火锅店叫什么来着？" | 语义搜索 + 人名识别，秒级定位 |
| "我的笔记里关于 Transformer 写了什么？" | 混合检索（向量 + 关键词），双路互补 |
| "数据上传到云端？不行，这有隐私风险" | **100% 本地运行**，模型和数据都在你的机器上 |
| "我要看到模型的推理过程，不是黑盒回答" | 💭 **思考面板**——Qwen3 实时展示推理过程 |
| "回答太慢了" | 可选 vLLM 后端，**5-10x 推理加速** |
| "同一个问题重复问" | 双层语义缓存，命中直接返回 |

---

## 🚀 5 分钟快速开始

```bash
# 1. 安装依赖
uv sync

# 2. 启动服务（模型自动从 ModelScope 下载，国内友好）
uv run python scripts/run_demo.py --serve

# 3. 浏览器打开 http://localhost:8000
```

**首次启动自动索引 demo 数据**（5 个人的微信聊天 + 3 篇技术笔记），可以直接体验。

**换你自己的数据：** 把微信导出的 `.txt` 文件放到 `data/raw/`，点左侧"🔍 扫描"。

---

## 🎯 核心特性

### 💭 Qwen3 思考模式——推理过程透明化

模型回答前会**实时展示推理过程**，不是黑盒。前端可折叠思考面板，闪烁光标流式显示。

```
用户：张三推荐了哪家火锅店？
  ↓
💭 思考过程 ▾
  │ 分析查询意图：查找张三的推荐
  │ 检查记忆片段[1]：张三提到渝味火锅
  │ 确认地址：朝阳区建国路88号
  │ 验证无其他火锅店推荐
  ↓
张三推荐了渝味火锅，位于朝阳区建国路88号。（来源：2026-05-05，张三、王五）
```

**支持的模型：** Qwen3-1.7B（默认）/ Qwen3-4B / Qwen3-8B  
**关闭思考：** `$env:POCKET_THINKING="false"`

### ⚡ 混合检索引擎

```
查询 → 实体提取（人名/日期/主题）
     → ChromaDB 向量检索（语义匹配）
     → BM25 关键词检索（精确匹配）
     → RRF 融合（k=60）
     → 实体加权（人名 ×1.2）+ 时间衰减（30天半衰期）
     → Top-K 结果
```

| 场景 | 纯向量 | 纯关键词 | **混合检索** |
|------|--------|---------|------------|
| "好吃的火锅" 找 "川菜推荐" | ✅ | ❌ | ✅ |
| 精确查找 "渝味火锅" | ⚠️ | ✅ | ✅ |
| "张三推荐了哪家餐厅" | ✅ | ✅ | ✅ |

### 🧠 Agent 编排（LangGraph）

```
Router（意图分类：关键词快速路由，0ms）
  → Retrieve（混合检索，~30ms）
    → Judge（组合评分公式，0ms，纯数学无 LLM）
      → Generate（思考模式流式生成）
```

**每次查询：最多 1 次 LLM 调用**（Router 关键词命中时为 0 次）。

### 📊 前端 Pipeline 可视化

实时显示每个阶段的耗时：路由 → 检索 → 判断 → 生成。绿色 = 完成，蓝色 = 进行中。

### 🔄 多轮对话

支持指代消解。"那家人均多少？"→ 自动关联上文"渝味火锅"。

### 💡 动态推荐问题

基于已索引数据自动生成推荐查询，随数据变化动态更新。

### 🗄️ 数据源管理

左侧面板显示所有已索引文件，支持 `扫描` 新文件和 `微信导入`（开发中）。

### 🔌 推理后端可切换

| 后端 | 速度 | 适用场景 |
|------|------|---------|
| `local_simulate`（默认） | 基准 | 开发调试，无需额外部署 |
| `vllm` | **5-10x** | 生产环境，需 WSL2 部署 vLLM Server |

```bash
# 切换到 vLLM
uv run python scripts/run_demo.py --serve --backend vllm
```

### 📦 模型下载——国内网络友好

模型自动发现优先级：**本地路径 → ModelScope（免科学上网）→ HuggingFace**

---

## 🏗️ 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│  🖥️ 前端（Vue 3 单文件，暗色主题）                              │
│  ┌──────────┬───────────────────────────────────────────────┐│
│  │ 数据源面板 │  对话窗口 + 💭 思考面板 + Pipeline 可视化       ││
│  │ 📂 已索引  │  推荐问题 | 搜索输入                           ││
│  └──────────┴───────────────────────────────────────────────┘│
└──────────────────────┬───────────────────────────────────────┘
                       │ HTTP / SSE
┌──────────────────────▼───────────────────────────────────────┐
│  FastAPI Server                                              │
│  /ask | /ask/stream | /data/sources | /data/suggestions      │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  LangGraph Agent: Router → Retrieve → Judge → Generate│    │
│  │  Hook 系统 (4 挂载点) + 用户画像 (自动记录查询行为)      │    │
│  └──────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  混合检索: ChromaDB (HNSW) + BM25 (jieba) + RRF       │    │
│  │  + 实体加权 + 时间衰减（30天半衰期）                    │    │
│  └──────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  双层缓存: L1 精确匹配 + L2 语义相似度                  │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│  InferenceBackend 抽象层                                      │
│  ┌─────────────────┐  ┌──────────────────┐                   │
│  │ LocalSimulate   │  │ VLLMBackend      │                   │
│  │ (transformers)  │  │ (OpenAI API,5x+) │                   │
│  └─────────────────┘  └──────────────────┘                   │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  数据摄取: WeChat TXT / Markdown / PDF                │    │
│  │  SourceManager → Chunker → Indexer                    │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

---

## 📡 API 端点

| 方法 | 端点 | 描述 |
|------|------|------|
| `GET` | `/` | 前端页面 |
| `GET` | `/health` | 健康检查（含设备信息 device:cuda/cpu） |
| `GET` | `/data/sources` | 已索引数据源列表 |
| `GET` | `/data/suggestions` | 动态推荐问题（模板+LLM混合方案） |
| `POST` | `/data/suggestions/refresh` | 后台 LLM 刷新推荐问题 |
| `POST` | `/ask` | 同步问答 |
| `POST` | `/ask/stream` | SSE token 级流式问答（含思考过程） |
| `POST` | `/ingest` | 扫描并索引新文件 |
| `GET` | `/data/watched` | 监控目录路径 |
| `GET` | `/wechat/status` | 微信运行状态 |
| `POST` | `/wechat/import` | 导入微信数据（开发中） |

---

## 📁 项目结构

```
PocketAgenticRAG/
├── frontend/index.html       # Vue 3 单文件前端
├── api/
│   ├── server.py             # FastAPI 服务（SSE 流式 + 思考分离）
│   ├── models.py             # Pydantic 数据模型
│   └── cache.py              # 双层语义缓存
├── agent/
│   ├── graph.py              # LangGraph 图编排
│   ├── state.py              # Agent 状态定义
│   ├── router.py             # 意图路由（关键词 + LLM）
│   ├── retriever_node.py     # 检索节点
│   ├── memory_judge.py       # 组合评分判断（零 LLM 调用）
│   ├── generator.py          # 答案生成（思考模式）
│   ├── hooks.py              # Hook 生命周期系统
│   └── user_profile.py       # 跨会话用户画像
├── rag/
│   ├── vector_store.py       # ChromaDB 向量库
│   ├── hybrid_retriever.py   # 混合检索 + RRF 融合
│   ├── bm25_retriever.py     # BM25 关键词检索
│   ├── entity_extractor.py   # 轻量级中文实体提取
│   └── entity_aware_retriever.py  # 实体加权检索
├── backend/
│   ├── base.py               # InferenceBackend 抽象接口
│   ├── local_simulate.py     # transformers 本地推理
│   ├── vllm_backend.py       # vLLM OpenAI API 后端
│   └── mobile_backend.py     # 手机端部署骨架
├── data_ingestion/
│   ├── source_manager.py     # 数据源管理（扫描+增量索引）
│   ├── pipeline.py           # 摄取管线
│   ├── chunker.py            # 对话感知分块
│   └── wechat_parser.py      # 微信导出解析
├── config/
│   ├── settings.py           # 全局配置（环境变量驱动）
│   └── prompts/              # Prompt YAML（版本化+热加载）
├── scripts/
│   ├── run_demo.py           # 一键启动（支持 --serve / --backend）
│   └── setup_vllm.sh         # WSL2 vLLM 一键安装
├── eval/
│   ├── benchmark.py          # Benchmark 框架
│   └── test_queries.json     # 测试查询集
└── PocketMemory-教学文档.md   # 完整教学文档（2300+ 行）
```

---

## 🧪 Benchmark

```bash
uv run python -m eval.benchmark
```

评测维度：检索 Recall@K / MRR、生成 LLM-Judge 评分、延迟分阶段统计、消融实验。

---

## 🔧 配置

所有配置通过**环境变量**控制，无需改代码：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `POCKET_LLM_MODEL` | Qwen3-1.7B | LLM 模型路径或 HF 名称 |
| `POCKET_EMBEDDING_MODEL` | bge-small-zh-v1.5 | Embedding 模型 |
| `POCKET_DEVICE` | cuda | 推理设备（cuda/cpu） |
| `POCKET_THINKING` | true | 是否启用思考模式 |
| `POCKET_BACKEND` | local_simulate | 推理后端（vllm 需要 WSL2） |
| `POCKET_VLLM_URL` | http://localhost:8001/v1 | vLLM Server 地址 |

---

## 🎓 面试/求职

项目配套 **[完整教学文档](PocketMemory-教学文档.md)**（2300+ 行），涵盖：
- 4 周学习路线图
- 每个技术决策的"为什么"
- 13 个面试必问问题的标准回答
- Prompt 工程的设计考量与迭代历史
- v1.0 → v1.3 架构演进

---

## 📄 许可证

MIT
