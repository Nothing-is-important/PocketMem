# AI Agent 开发学习笔记

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
