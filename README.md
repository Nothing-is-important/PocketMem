# 🧠 PocketMemory —— 端侧个人记忆助手

> 你的微信聊天记录和笔记，本地索引、离线查询、数据不出设备。

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector-purple.svg)](https://www.trychroma.com/)
[![Vue 3](https://img.shields.io/badge/Vue-3.x-42b883.svg)](https://vuejs.org/)

---

## 快速开始

```bash
# 1. 安装依赖
cd PocketAgenticRAG
uv sync

# 2. 启动 API 服务 + 前端界面
uv run python scripts/run_demo.py --serve

# 3. 浏览器打开
# http://localhost:8000
```

**系统要求：** Python 3.12+ / RTX 4060 8GB（或同等 GPU）/ Windows 10+

---

## 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│  🖥 前端界面（Vue 3 单文件，OpenCode 风格暗色主题）              │
│  ┌──────────┬───────────────────────────────────────────────┐│
│  │ 左侧栏    │  对话窗口（消息流 + 思考指示器）                  ││
│  │          │                                               ││
│  │ 📂 数据源 │                                               ││
│  │ [扫描]   │  ──────────────────────────────────────────── ││
│  │ [导入]   │  管道: 🎯路由 → 🔍检索 → ⚖️判断 → ✨生成        ││
│  │          │  建议标签 | 搜索输入                            ││
│  │ 本地文件  │                                               ││
│  │ 微信导入  │                                               ││
│  └──────────┴───────────────────────────────────────────────┘│
└──────────────────────┬───────────────────────────────────────┘
                       │ HTTP / SSE
┌──────────────────────▼───────────────────────────────────────┐
│  FastAPI Server (api/server.py)                               │
│  /ask | /ask/stream | /data/sources | /ingest | /health      │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  LangGraph Agent: Router → Retrieve → Judge → Generate│    │
│  │  (2 次 LLM 调用 / 查询，组合评分替代多次验证)           │    │
│  └──────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  混合检索: ChromaDB (HNSW) + BM25 (jieba) + RRF 融合  │    │
│  │  + 实体加权 + 时间衰减（30 天半衰期）                   │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│  InferenceBackend 抽象层 + 数据摄取管线                        │
│  Qwen2.5-1.5B / Qwen3-4B W4A8 | WeChat/MD/PDF → Chunk → Index│
└──────────────────────────────────────────────────────────────┘
```

---

## 功能特性

- **🔍 混合检索** — ChromaDB 向量检索 + BM25 关键词检索，ThreadPoolExecutor 并行执行 + RRF 融合
- **⏰ 时间感知** — 指数时间衰减（30 天半衰期），符合艾宾浩斯遗忘曲线
- **👤 实体加权** — 百家姓 + 正则轻量级实体提取，零额外显存
- **🧠 Agent 编排** — LangGraph 状态图：Router → Retrieve → Judge → Generate，Router 关键词优先（命中跳过 LLM）
- **💾 语义缓存** — L2 余弦相似度缓存 + 查询归一化，提升重复查询命中率
- **📡 SSE 真流式** — Token 级逐字推送（backend.generate_stream），非 5 字符伪造
- **💬 多轮对话** — conversation_history 注入近 3 轮对话，支持指代消解
- **📂 数据源管理** — 支持微信 TXT / Markdown / PDF，SHA256 内容哈希去重，左侧常驻面板
- **📱 Android APK** — Kotlin + Jetpack Compose，连接 PC 后端
- **🔌 推理抽象层** — 策略模式，PC（transformers）和手机（ONNX）一键切换

---

## 项目结构

```
PocketAgenticRAG/
├── frontend/
│   └── index.html          # Vue 3 单文件前端（948 行）
├── api/
│   ├── server.py           # FastAPI 服务（SSE 流式）
│   ├── models.py           # Pydantic 数据模型
│   └── cache.py            # 语义缓存（查询归一化 + L2 余弦相似度）
├── agent/
│   ├── graph.py            # LangGraph 图定义
│   ├── state.py            # Agent 状态
│   ├── router.py           # 意图路由
│   ├── retriever_node.py   # 检索节点
│   ├── memory_judge.py     # 组合评分判断
│   ├── generator.py        # 答案生成
│   ├── hooks.py            # Hook 系统
│   └── user_profile.py     # 用户画像
├── rag/
│   ├── vector_store.py     # ChromaDB 向量库
│   ├── hybrid_retriever.py # 混合检索 + RRF
│   ├── bm25_retriever.py   # BM25 关键词检索
│   ├── embeddings.py       # Embedding 模型
│   ├── entity_extractor.py # 实体提取
│   └── entity_aware_retriever.py  # 实体加权检索
├── data_ingestion/
│   ├── source_manager.py   # 数据源管理器
│   ├── pipeline.py         # 摄取管线
│   ├── chunker.py          # 文本分块
│   ├── indexer.py          # 索引器
│   ├── wechat_parser.py    # 微信解析器
│   ├── wechat_detector.py  # 微信进程检测
│   └── markdown_loader.py  # Markdown 加载器
├── backend/                # 推理后端抽象层
├── config/                 # 配置 + Prompt YAML
├── android/                # Android APK (Kotlin)
├── docs/                   # 设计文档
├── scripts/                # 运行脚本
└── tests/                  # 测试
```

---

## API 端点

| 方法 | 端点 | 描述 |
|------|------|------|
| `GET` | `/` | 前端页面 |
| `GET` | `/health` | 健康检查 |
| `GET` | `/data/sources` | 数据源列表 |
| `POST` | `/ask` | 同步问答（支持 conversation_history 参数） |
| `POST` | `/ask/stream` | SSE token 级流式问答（支持多轮对话） |
| `POST` | `/ingest` | 扫描索引新文件 |
| `GET` | `/wechat/status` | 微信运行状态 |
| `POST` | `/wechat/import` | 导入微信数据 |

---

## 技术栈

| 层级 | 技术 |
|------|------|
| **前端** | Vue 3 (CDN 单文件)、CSS Grid、暗色主题 |
| **后端** | FastAPI、SSE 流式、Python 3.12 |
| **Agent** | LangGraph、StateGraph、条件边 + 循环 |
| **检索** | ChromaDB (HNSW)、BM25 (jieba)、RRF 融合 |
| **模型** | Qwen2.5-1.5B / Qwen3-4B W4A8、BGE-small-zh-v1.5 |
| **推理** | Transformers (PyTorch)、ONNX Runtime、LLMC 量化 |
| **移动端** | Kotlin、Jetpack Compose、Retrofit |
| **包管理** | uv (Rust, 快 10-100x) |

---

## 许可证

MIT
