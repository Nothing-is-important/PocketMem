"""QLoRA 微调训练数据生成器。

从 192 份企业文档自动生成 Q&A 训练对。

生成策略：
1. 结构化提取：从会议纪要/合同/技术文档中提取关键信息 → 生成事实型问题
2. 多文档综合：关联不同文档中提到同一主题的片段 → 生成综合型问题
3. 实体关联：基于人名/项目名/技术名词 → 生成关联型问题
4. 否定样本：故意问文档中没有的内容 → 训练模型"不编造"

输出格式（Alpaca 格式）：
{
  "instruction": "根据以下文档内容回答问题...",
  "input": "文档1: ...\n文档2: ...\n\n问题: ...",
  "output": "根据文档X，..."
}
"""

import json
import os
import random
import re
from pathlib import Path

random.seed(42)

# ├── Paths
ENTERPRISE_DIR = "data/demo/enterprise"
OUTPUT_FILE = "data/training_data.json"
random.seed(42)


def load_all_documents(base_dir):
    """加载所有企业文档。"""
    docs = []
    for root, _, files in os.walk(base_dir):
        for fname in files:
            if fname.endswith(('.txt', '.md')):
                path = os.path.join(root, fname)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    doc_type = _guess_type(root, fname)
                    docs.append({
                        'file': fname,
                        'path': path,
                        'type': doc_type,
                        'content': content,
                    })
                except Exception:
                    continue
    return docs


def _guess_type(root, fname):
    root_lower = root.lower()
    if 'email' in root_lower: return 'email'
    if 'meeting' in root_lower or '会议' in fname: return 'meeting'
    if 'contract' in root_lower or '合同' in fname: return 'contract'
    if 'doc' in root_lower: return 'doc'
    return 'other'


def extract_key_snippets(docs):
    """从文档中提取高质量文本片段。"""
    snippets = []
    for doc in docs:
        content = doc['content']

        # 去除邮件头，保留正文
        if doc['type'] == 'email':
            parts = content.split('\n\n', 2)
            body = parts[-1] if len(parts) > 1 else content
        else:
            body = content

        # 按段落分割
        paragraphs = [p.strip() for p in body.split('\n\n') if len(p.strip()) > 80]

        for p in paragraphs:
            # 过滤：太短、纯列表、纯代码
            if len(p) < 50: continue
            if p.count('|') > 5: continue  # 表格跳过
            if p.startswith('```'): continue

            snippets.append({
                'doc': doc,
                'text': p[:800],  # 截断
            })
    return snippets


def generate_factual_questions(snippets, count=80):
    """从文本片段生成事实型问题。"""
    # 匹配模式 → 问题模板
    patterns = [
        (r'(\S{2,6})选择了?(\S{2,10})[，,]\s*理由[是为：:]?\s*(.{10,100})',
         ["{company}为什么选择{choice}？", "选择{choice}的理由是什么？",
          "{company}在技术选型中选择了{choice}，原因是什么？"]),
        (r'(R\w+\S*)\s*[（(]([^)）]{5,30})[)）]',
         ["{metric}指标有什么要求？", "{metric}的达标标准是什么？",
          "系统性能指标{metric}的具体要求是什么？"]),
        (r'(\S+)负责人?[是为]\s*(\S{2,4})\s*[，,]?\s*负责\s*(\S{4,15})',
         ["谁负责{task}？", "哪个团队在做{task}？",
          "关于{task}，主要负责人是谁？"]),
        (r'(\S{3,10})延迟[约在]?(\d+[-~]\d+)\s*ms',
         ["{module}模块的延迟是多少？", "{module}的性能数据是什么？"]),
        (r'版本[是为]\s*v?([\d.]+).*?日期[是为]?\s*(\d{4}-\d{2}-\d{2})',
         ["v{version}的发布日期是什么时候？", "最新版本的版本号和日期是什么？"]),
    ]

    questions = []
    for snippet in random.sample(snippets, min(count * 2, len(snippets))):
        text = snippet['text']
        for pattern, templates in patterns:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                question = random.choice(templates)
                # 构建答案：基于匹配的文本片段
                answer_snippet = text[max(0, match.start() - 30): match.end() + 100]
                try:
                    q = question.format(
                        company='云帆科技',
                        choice=groups[0] if len(groups) > 0 else '',
                        task=groups[-1] if len(groups) > 0 else '',
                        module=groups[0] if len(groups) > 0 else '',
                        version=groups[0] if len(groups) > 0 else '',
                        metric=groups[0] if len(groups) > 0 else '',
                    )
                except (IndexError, KeyError):
                    q = None
                if q:
                    questions.append({
                        'question': q,
                        'answer': f"根据文档内容，{answer_snippet.strip()[:300]}",
                        'sources': [snippet['doc']['file']],
                        'type': 'factual',
                    })
                break
        if len(questions) >= count:
            break
    return questions[:count]


def generate_entity_questions(snippets, count=50):
    """基于实体生成关联型问题。使用硬编码的企业实体。"""
    known_people = ['张伟', '李娜', '王磊', '陈静', '赵明', '刘洋', '周婷', '吴强', '林芳', '马超', '孙悦', '黄涛']
    known_projects = ['凤凰', '昆仑', '泰山', '凌霄', '朱雀']
    known_tech = ['ChromaDB', 'BM25', 'Cross-Encoder', 'AccessGuard', 'W4A8', 'QLoRA', 'LangGraph',
                  'RRF', 'HNSW', 'BGE-large', 'FastAPI', 'SSE', 'Docker', 'K8s', 'Redis']

    questions = []
    question_templates = [
        ("{person}负责什么工作？", "person"),
        ("{person}参与了哪些决策？", "person"),
        ("{project}项目是什么？", "project"),
        ("{project}项目的技术栈是什么？", "project"),
        ("{tech}和其他方案相比有什么优势？", "tech"),
        ("{tech}在项目中的作用是什么？", "tech"),
        ("{person}和{project}项目有什么关系？", "cross"),
        ("{project}项目中{tech}是怎么用的？", "cross"),
    ]

    for _ in range(count):
        tmpl, qtype = random.choice(question_templates)
        person = random.choice(known_people)
        project = random.choice(known_projects)
        tech = random.choice(known_tech)

        q = tmpl.format(person=person, project=project, tech=tech)

        # 从 snippets 中找到相关文本作为答案上下文
        relevant_snippets = []
        keywords = q.replace('？', '').replace('什么', '').replace('怎么', '')
        for s in snippets[:200]:
            if any(kw in s['text'] for kw in [person, project, tech] if len(kw) > 1):
                relevant_snippets.append(s)
            if len(relevant_snippets) >= 3:
                break

        source_files = [s['doc']['file'] for s in relevant_snippets[:3]]
        answer_context = '; '.join(s['text'][:200] for s in relevant_snippets[:2])

        questions.append({
            'question': q,
            'answer': f"根据企业文档，{answer_context[:400] if answer_context else '请参考相关文档获取详细信息。'}",
            'sources': source_files,
            'type': 'entity',
        })

    return questions[:count]


def generate_comparison_questions(snippets, count=30):
    """生成对比/综合型问题。"""
    comparisons = [
        ("ChromaDB 和 Elasticsearch 对比，哪个更适合中小企业？",
         "ChromaDB纯Python实现，部署简单，10万文档内延迟<50ms；Elasticsearch功能强大但需要Java环境和更多资源。中小企业推荐ChromaDB。"),
        ("向量检索和BM25关键词检索各有什么优劣？",
         "向量检索擅长语义匹配（同义词、改写表达），BM25擅长精确关键词匹配。两者通过RRF融合，Recall@5从0.62提升到0.78。"),
        ("BGE-large 和 BGE-small 的对比结论是什么？",
         "BGE-large Recall@5=0.86，模型400MB；BGE-small Recall@5=0.78，模型100MB。移动端用small，服务端用large。"),
        ("FP16和W4A8量化模型在推理速度和精度上有何区别？",
         "Qwen3-8B的FP16推理25t/s显存14.8GB，W4A8推理42t/s显存3.8GB。PPL损失5.3%，RAG场景下影响可忽略。"),
        ("凤凰项目为什么选择双模式LLM方案？",
         "普通文档走DeepSeek API追求速度，机密文档走本地Qwen3-8B W4A8确保数据安全。这是安全合规与性能的最佳平衡。"),
        ("Cross-Encoder重排带来了多少提升？",
         "Recall@5从0.78提升到0.85(+9%)，MRR从0.63提升到0.72(+14%)。代价是额外300MB显存和85ms延迟。"),
    ]

    questions = []
    for _ in range(min(count, len(comparisons) * 5)):
        q, a = random.choice(comparisons)
        # 找到相关文档
        rel = []
        for s in snippets[:300]:
            if any(w in s['text'] for w in q[:20]):
                rel.append(s['doc']['file'])
        questions.append({
            'question': q,
            'answer': a,
            'sources': list(set(rel))[:3] if rel else ['internal reports'],
            'type': 'comparison',
        })
    return questions


def generate_negative_samples(count=20):
    """生成否定样本：问文档中不存在的内容，训练模型说'不知道'。"""
    fake_questions = [
        ("去年Q4的营收是多少？", "文档中未提及具体的营收数据。"),
        ("谁负责波士顿办事处的运营？", "文档中未提及波士顿办事处。波士顿不属于当前文档覆盖范围。"),
        ("公司的上市计划是什么？", "文档中未提及上市计划的相关信息。"),
        ("AI平台的DAU是多少？", "当前文档中未提供DAU数据。"),
        ("张三是什么时候加入公司的？", "现有文档中未找到张三的入职信息。"),
        ("为什么要收购X公司？", "文档中未涉及收购X公司的相关讨论。"),
        ("明年的预算规划是多少？", "文档中未包含明年的预算规划。"),
        ("美国和欧洲市场的拓展计划？", "文档中未提及海外市场拓展计划。"),
    ]

    questions = []
    for i in range(min(count, len(fake_questions) * 3)):
        q, a = fake_questions[i % len(fake_questions)]
        questions.append({
            'question': q,
            'answer': a,
            'sources': [],
            'type': 'negative',
        })
    return questions


def build_training_data(docs):
    """构建完整的训练数据集。"""
    snippets = extract_key_snippets(docs)

    print(f"  文档: {len(docs)} 份")
    print(f"  片段: {len(snippets)} 条")

    all_questions = []

    # 各类问题
    factual = generate_factual_questions(snippets, 80)
    all_questions.extend(factual)
    print(f"  事实型: {len(factual)}")

    entity = generate_entity_questions(snippets, 50)
    all_questions.extend(entity)
    print(f"  实体型: {len(entity)}")

    comparison = generate_comparison_questions(snippets, 30)
    all_questions.extend(comparison)
    print(f"  对比型: {len(comparison)}")

    negative = generate_negative_samples(20)
    all_questions.extend(negative)
    print(f"  否定型: {len(negative)}")

    random.shuffle(all_questions)

    # 转为 Alpaca 格式
    alpaca_data = []
    for item in all_questions:
        # 构建 input 上下文（模拟 RAG 检索返回的文档）
        sources_text = ""
        if item['sources']:
            for i, src in enumerate(item['sources'][:3], 1):
                # 找实际文档内容
                doc_content = ""
                for d in docs:
                    if d['file'] == src:
                        doc_content = d['content'][:300]
                        break
                if not doc_content:
                    doc_content = f"企业文档: {src}"
                sources_text += f"\n文档{i}: {src}\n{doc_content}\n"

        alpaca_entry = {
            "instruction": "你是企业知识助手。根据以下文档内容回答问题。如果文档中没有相关信息，明确说明。回答时引用文档编号。保持专业、简洁。",
            "input": f"{sources_text}\n问题: {item['question']}",
            "output": item['answer'],
        }
        alpaca_data.append(alpaca_entry)

    # 8:1:1 划分
    random.shuffle(alpaca_data)
    n = len(alpaca_data)
    train = alpaca_data[:int(n * 0.8)]
    val = alpaca_data[int(n * 0.8):int(n * 0.9)]
    test = alpaca_data[int(n * 0.9):]

    return {'train': train, 'val': val, 'test': test}


def main():
    print("生成 QLoRA 微调训练数据")
    print("=" * 50)

    docs = load_all_documents(ENTERPRISE_DIR)
    data = build_training_data(docs)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    for split_name in ['train', 'val', 'test']:
        split_path = f"data/training_data_{split_name}.json"
        with open(split_path, 'w', encoding='utf-8') as f:
            json.dump(data[split_name], f, ensure_ascii=False, indent=2)

    print(f"\n训练数据已保存:")
    print(f"  训练集: {len(data['train'])} 条")
    print(f"  验证集: {len(data['val'])} 条")
    print(f"  测试集: {len(data['test'])} 条")
    print(f"  总计:   {len(data['train']) + len(data['val']) + len(data['test'])} 条")
    print(f"\n下一步: python scripts/train_qlora.py")


if __name__ == "__main__":
    main()
