"""意图路由节点。

将用户查询分为四类：
- memory_lookup: 关于过去对话、人物、事件的查询（走时间感知检索）
- knowledge_lookup: 关于笔记、文档、知识点（走标准检索）
- general: 普通对话（直接生成）
- refuse: 拒绝回答的查询

两级路由策略：
1. LLM 分类（主力，大模型效果好）
2. 关键词兜底（1.5B 等小模型分类不准时的保底方案）
"""

import re
from typing import Dict

from .state import AgentState

# Prompt 从配置加载（支持版本管理和热加载）
from config.prompts import prompts

# 关键词兜底规则
_MEMORY_KEYWORDS = [
    "说过", "说了", "推荐", "介绍", "提到", "提了", "讲过", "聊过",
    "讨论", "商量", "决定", "去过", "吃了", "买了",
    "投票", "选了", "面试", "offer", "公司",
    "合同", "项目", "方案", "技术", "文档", "邮件", "会议",
    "上次", "之前", "以前", "上周", "上个月", "去年", "那天",
    "什么时候", "什么时间", "在哪里", "怎么去", "多少钱",
    "哪些", "哪家", "哪个", "谁", "什么地方",
]
_KNOWLEDGE_KEYWORDS = [
    "笔记", "文档", "论文", "文章", "资料", "学习", "教程",
    "原理", "架构", "算法", "方法", "优化", "代码", "实现",
    "定义", "概念", "公式", "总结", "归纳",
    "写了什么", "内容是什么", "讲什么",
]
# 常见人名触发词（百家姓 + 常见称呼后缀）
_NAME_PATTERN = re.compile(
    r"([赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张"
    r"孔曹严华金魏陶姜戚谢邹柏水窦章苏潘葛范彭郎鲁韦昌马苗花方俞任袁柳鲍"
    r"史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮下齐康伍余元卜顾孟平黄"
    r"和穆萧尹姚邵汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁杜阮蓝"
    r"闵席季麻强贾路娄危江童颜郭梅盛林钟徐邱骆高夏蔡田樊胡凌霍虞万支柯管卢"
    r"莫房]"
    r"[一-鿿]{1,2})"
)


def create_router_node(backend):
    """创建路由节点工厂函数。"""

    def router_node(state: AgentState) -> AgentState:
        import time
        t0 = time.time()
        query = state["query"]

        if not query or not query.strip():
            state["intent"] = "refuse"
            return state

        # Step 1: 关键词快速路由（命中则跳过 LLM，节省 2-3s）
        keyword_intent = _keyword_fallback(query)
        if keyword_intent:
            intent = keyword_intent
            raw_response = f"keyword:{keyword_intent}"
        else:
            # Step 2: LLM 分类（关键词未命中时）
            prompt = prompts.get("router").format(query=query)
            raw_response = backend.generate(prompt, max_tokens=16).strip().lower()
            intent = _parse_intent(raw_response)
            # Step 3: 关键词兜底（LLM 说 general 时再检查一次）
            if intent == "general":
                keyword_intent = _keyword_fallback(query)
                if keyword_intent:
                    intent = keyword_intent

        state["intent"] = intent
        state["messages"].append({
            "role": "router",
            "content": f"Intent: {intent} (LLM: {raw_response[:20]})",
        })
        state["latency_stats"]["router_ms"] = (time.time() - t0) * 1000
        return state

    return router_node


def _parse_intent(raw: str) -> str:
    """解析 LLM 输出的意图标签。"""
    raw = raw.strip().lower()
    valid_intents = ["memory_lookup", "knowledge_lookup", "general", "refuse"]
    for intent in valid_intents:
        if intent in raw:
            return intent
    return "general"


# 常见非人名词——匹配姓氏+1-2字但实际是普通词汇（用于 _keyword_fallback 的噪声过滤）
_NAME_NOISE = {
    "方法", "方向", "方便", "方案", "方式", "方针",
    "时间", "时候", "时钟",
    "然后", "自然",
    "黄金", "黄色",
    "阳光", "太阳",
    "可以", "可能",
    "我们", "他们", "你们",
    "什么", "怎么", "这么", "那么",
    "一个", "一些", "这个", "那个",
    "不是", "不会", "不能", "不要", "不过",
    "现在", "已经", "还是", "但是", "因为", "所以",
    "如果", "虽然", "而且", "或者",
    "开始", "出来", "起来", "下来", "回来",
    "一下", "一起", "一定", "一样",
    "地方", "地点",
    "笔记", "记录",
}


def _keyword_fallback(query: str) -> str:
    """关键词兜底：当 LLM 返回 general 时，检查是否应走检索。

    Returns:
        "memory_lookup", "knowledge_lookup", 或 ""（不覆盖）
    """
    # 包含人名 → 很可能在问某个人的事（需过滤"方法""方向"等常见非人名词）
    name_matches = _NAME_PATTERN.findall(query)
    real_names = [
        m for m in name_matches
        if m not in _NAME_NOISE and m[:2] not in _NAME_NOISE
    ]
    if real_names:
        return "memory_lookup"

    # 检查知识类关键词
    knowledge_hits = sum(1 for kw in _KNOWLEDGE_KEYWORDS if kw in query)
    if knowledge_hits >= 2:
        return "knowledge_lookup"

    # 检查记忆类关键词
    memory_hits = sum(1 for kw in _MEMORY_KEYWORDS if kw in query)
    if memory_hits >= 2:
        return "memory_lookup"

    return ""  # 保持 LLM 的分类结果


def intent_to_retrieval_mode(intent: str) -> str:
    """将意图映射到检索模式。"""
    mapping: Dict[str, str] = {
        "memory_lookup": "temporal",
        "knowledge_lookup": "standard",
        "general": "none",
        "refuse": "none",
    }
    return mapping.get(intent, "standard")
