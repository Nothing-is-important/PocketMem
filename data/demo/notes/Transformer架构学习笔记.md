# Transformer 架构学习笔记

## 注意力机制

自注意力（Self-Attention）是 Transformer 的核心。计算公式：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

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
