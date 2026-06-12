# TeamMind 项目完整修改方案

## 当前状态审计

| 模块 | 状态 | 致命问题 |
|------|------|---------|
| Agent 图 | 4 节点 | **AccessGuard 未接入**，写了但没挂 |
| 推理后端 | 3 种 | DualModeBackend 未与文档分级联动 |
| 训练数据 | 119 条 | **太少**，Enron 528K 封邮件只用了 68 封 |
| 微调 | 脚本就绪 | **一次没跑过**，无任何实验数据 |
| Benchmark | 文件存在 | **空壳**，无任何指标数据 |
| 前端 | Vue3 暗色主题 | 推荐问题列表用旧模板，企业数据未索引 |
| 演示数据 | 192 份企业文档 | 可索引但未被 Agent 用作实际数据源 |
| Enron 项目数据 | 119 个文件 | 翻译版，未索引 |

---

## Phase 1：数据基座（今天）

**做什么：** 让项目有真实数据支撑，不再依赖假数据。

### 1.1 全量提取 Enron + CUAD

```
输入: Enron 528K 封 + CUAD 20,910 条
输出: ~1100 条英文训练数据（Alpaca 格式）
操作: python scripts/extract_full_enron.py
```

**采样策略：** 150 人 × 每人 6 封 = ~900 封 + CUAD 200 条 = 1100 条。
**为什么不全取？** 528K 封中 70% 是自动回复/垃圾/纯转发，取了没用。
**为什么 1100 条够了？** LIMA 论文证明 1000 条高质量数据可以匹敌 50K 条。QLoRA 做行为适配，不是知识注入。

### 1.2 直接使用英文原版，不翻译

**理由：** Qwen3 是中英双语模型。LoRA 只改 1.2% 权重，中文能力保留在冻结的 98.8% 里。英文训练的引用行为会直接迁移到中文。

**instruction 用双语：** `"你是企业知识助手。Answer based on the document. Cite sources. 说'文档中未提及'当信息不存在。"`

### 1.3 企业数据实际索引

将 192 份企业文档 + Enron 119 份中文邮件索引进 ChromaDB，作为实际可检索的项目数据。

**删除：** 微信 demo 数据（张三李四火锅店）不再作为默认数据源。

### 1.4 文件变更

| 操作 | 文件 |
|------|------|
| 新建 | `scripts/extract_full_enron.py` |
| 修改 | `scripts/generate_training_data.py` → 删除，不需要合成数据 |
| 修改 | `scripts/run_demo.py` → 默认数据源改为企业数据 |
| 删除 | `scripts/ingest_demo_data.py` → 不再需要微信模拟数据 |

---

## Phase 2：Agent 图重构（今天）

**做什么：** 把写了没挂的模块接入 Agent 图。

### 2.1 接入 AccessGuard

当前 Agent 图：`Router → Retrieve → Judge → Generate`
修复后：`Router → Retrieve → **AccessGuard** → Judge → Generate`

AccessGuard 放在 Retrieve 之后、Judge 之前。检索到文档后先按用户权限过滤再判断相关性。

### 2.2 接入 DualModeBackend

Generator 节点在调用 `backend.generate()` 之前，检测已过滤文档的等级分布：
- 全部 public → API 模式（快）
- 含 internal/confidential → 本地模式（安全）

### 2.3 接入 Reranker

在 Judge 之前增加可选的重排节点。显存够时加载 Cross-Encoder，不够时降级为轻量规则重排。

### 2.4 文件变更

| 操作 | 文件 |
|------|------|
| 修改 | `agent/graph.py` → 增加 guard + rerank 节点 + 条件边 |

---

## Phase 3：QLoRA 微调（明天，需 GPU 4-6h）

**做什么：** 真正跑一次微调，拿到实验数据。

### 3.1 环境准备

```bash
pip install unsloth transformers datasets peft bitsandbytes trl accelerate
python scripts/train_qlora.py --dry-run
```

### 3.2 开始训练

```bash
python scripts/train_qlora.py --model Qwen3-4B --epochs 3
```

**期望结果：**
- Loss: 2.0 → 0.5（正常）
- 训练时间: 4-6 小时
- 输出: `models/teammind-qlora/` (~50MB LoRA adapter)

### 3.3 微调后验证

用 20 个企业查询测试微调前后对比：
- 引用准确率：微调前 ~40% → 微调后 ~80%+
- "不知道"正确率：微调前 ~20% → 微调后 ~70%+

---

## Phase 4：Benchmark（明晚，1-2h）

**做什么：** 拿到真实数据，填上简历里所有"提升 X%"的空。

### 4.1 检索质量

| 指标 | 目标 |
|------|------|
| Recall@5 | ≥0.80 |
| MRR | ≥0.65 |

### 4.2 答案质量（LLM-as-Judge, 1-5）

| 指标 | 目标 |
|------|------|
| Faithfulness | ≥0.85 |
| Answer Relevancy | ≥0.80 |

### 4.3 系统性能

| 指标 | 目标 |
|------|------|
| P50 延迟 | <2s |
| P95 延迟 | <5s |
| 显存峰值 | <7.5GB |

### 4.4 文件变更

| 操作 | 文件 |
|------|------|
| 修改 | `eval/benchmark.py` → 简化，只跑核心指标 |
| 新建 | `scripts/benchmark_compare.py` → 微调前后对比脚本 |

---

## Phase 5：前端 + 文档（后天）

### 5.1 修复推荐问题 Bug

`_generate_template_suggestions` 的模板从 "张三推荐了..." 改为企业场景模板。

### 5.2 默认数据源切换

前端启动时默认展示企业数据，搜索占位符改为 "搜索企业知识库..."

### 5.3 更新 README

填上 Benchmark 的真实数据，替换掉所有"预计"、"目标"。

### 5.4 更新教学文档

第 13 节（QLoRA）补充真实实验结果。

---

## 优先级排序

| 优先级 | 阶段 | 时间 | 面试提升 |
|--------|------|------|---------|
| **P0** | Phase 1：数据基座 | 今天 | 从"假数据"到"真实企业数据" |
| **P0** | Phase 2：Agent 图重构 | 今天 | AccessGuard + DualMode 终于能用 |
| **P1** | Phase 3：QLoRA 微调 | 明天（需GPU） | 项目核心竞争力 |
| **P1** | Phase 4：Benchmark | 明晚 | 简历里的数字不再是编的 |
| **P2** | Phase 5：前端 + 文档 | 后天 | 演示效果 + 面试话术 |

## 不改的部分

- InferenceBackend 抽象层 → 已经够好
- ChromaDB + BM25 + RRF 检索 → 已经够好
- 前端暗色主题 → 已经够好
- Vue3 架构 → 不动
- 教学文档结构 → 只补充数据，不改结构
