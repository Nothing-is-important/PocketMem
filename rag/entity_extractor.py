"""轻量级中文实体提取。

不使用额外 NER 模型（省显存），基于规则 + TF-IDF。
"""

import re
from collections import Counter
from typing import Dict, List, Set

import jieba
import jieba.posseg as pseg


# 常见中文姓氏（百家姓前 100）
_COMMON_SURNAMES: Set[str] = {
    "赵", "钱", "孙", "李", "周", "吴", "郑", "王", "冯", "陈", "褚", "卫",
    "蒋", "沈", "韩", "杨", "朱", "秦", "尤", "许", "何", "吕", "施", "张",
    "孔", "曹", "严", "华", "金", "魏", "陶", "姜", "戚", "谢", "邹", "喻",
    "柏", "水", "窦", "章", "苏", "潘", "葛", "奚", "范", "彭", "郎", "鲁",
    "韦", "昌", "马", "苗", "凤", "花", "方", "俞", "任", "袁", "柳", "鲍",
    "史", "唐", "费", "廉", "岑", "薛", "雷", "贺", "倪", "汤", "滕", "殷",
    "罗", "毕", "郝", "邬", "安", "常", "乐", "于", "时", "傅", "皮", "卞",
    "齐", "康", "伍", "余", "元", "卜", "顾", "孟", "平", "黄", "和", "穆",
    "萧", "尹", "姚", "邵", "汪", "祁", "毛", "禹", "狄", "米", "贝", "明",
    "臧", "计", "伏", "成", "戴", "谈", "宋", "茅", "庞", "熊", "纪", "舒",
    "屈", "项", "祝", "董", "梁", "杜", "阮", "蓝", "闵", "席", "季", "麻",
    "强", "贾", "路", "娄", "危", "江", "童", "颜", "郭", "梅", "盛", "林",
    "钟", "徐", "邱", "骆", "高", "夏", "蔡", "田", "樊", "胡", "凌", "霍",
    "虞", "万", "支", "柯", "管", "卢", "莫", "房", "裘", "缪", "干", "解",
    "应", "宗", "丁", "宣", "邓", "郁", "单", "杭", "洪", "包", "诸", "左",
    "石", "崔", "吉", "龚", "程", "邢", "滑", "裴", "陆", "荣", "翁", "荀",
    "羊", "惠", "甄", "家", "封", "芮", "羿", "储", "靳", "井", "段", "富",
    "巫", "乌", "焦", "巴", "弓", "牧", "谷", "车", "侯", "蓬", "全", "郗",
    "班", "仰", "秋", "仲", "伊", "宫", "宁", "仇", "甘", "武", "刘", "景",
    "詹", "龙", "叶", "幸", "司", "韶", "郜", "黎", "蓟", "薄", "印", "白",
    "翟", "谭", "贡", "劳", "姬", "申", "扶", "堵", "冉", "宰", "雍", "桑",
    "桂", "牛", "寿", "通", "边", "燕", "冀", "尚", "农", "温", "别", "庄",
    "晏", "柴", "瞿", "阎", "充", "慕", "连", "茹", "习", "宦", "艾", "鱼",
    "容", "向", "古", "易", "慎", "戈", "廖", "居", "衡", "都", "耿", "满",
    "匡", "文", "寇", "广", "禄", "阙", "东", "师", "巩", "聂", "勾", "敖",
    "冷", "辛", "阚", "那", "简", "饶", "曾", "母", "沙", "养", "鞠", "丰",
    "关", "相", "查", "荆", "游", "竺", "权", "逯", "盖", "益", "桓", "公",
    # 复姓
    "司马", "上官", "欧阳", "夏侯", "诸葛", "闻人", "东方", "赫连", "皇甫",
    "尉迟", "公羊", "澹台", "公冶", "宗政", "濮阳", "淳于", "仲孙", "太叔",
    "申屠", "公孙", "轩辕", "令狐", "钟离", "宇文", "长孙", "慕容", "司徒",
}


def extract_entities(text: str) -> Dict[str, List[str]]:
    """从文本中提取实体。

    Args:
        text: 输入文本

    Returns:
        {"people": [...], "dates": [...], "topics": [...]}
    """
    return {
        "people": extract_people(text),
        "dates": extract_date_mentions(text),
        "topics": extract_topics(text),
    }


def extract_people(text: str) -> List[str]:
    """提取中文人名。

    策略：
    1. jieba 词性标注识别 nr（人名）标签
    2. 正则匹配 "姓氏 + 1-2个汉字"
    """
    people = set()

    # jieba 词性标注
    for word, flag in pseg.cut(text):
        if flag == "nr" and len(word) <= 3:
            people.add(word)

    # 正则补充：常见姓氏 + 1-2 个常见名字用字
    name_pattern = re.compile(
        r"([" + "".join(_COMMON_SURNAMES) + r"])([一-鿿]{1,2})"
    )
    for match in name_pattern.finditer(text):
        name = match.group(0)
        if 2 <= len(name) <= 3:
            people.add(name)

    # 去噪：排除非人名的高频词
    noise_words = {
        "今天", "明天", "昨天", "现在", "可以", "这个", "那个", "什么",
        "怎么", "为什么", "因为", "所以", "而且", "或者", "不过", "但是",
        "开始", "已经", "还是", "出来", "起来", "下来", "上来", "回来",
        "一些", "一下", "一起", "一定", "一个", "一样", "一边", "一点",
        "方法", "方向", "方便", "方案", "方式", "方针",
        "地方", "地点", "笔记", "记录", "结果",
        "然后", "自然", "当然", "可以", "可能", "可是", "可见",
        "我们", "他们", "你们", "不是", "不会", "不能", "不要", "不过",
    }
    people -= noise_words

    # 查询中的关键词（不应出现在人名中）
    query_keywords = {"项目", "方案", "技术", "合同", "邮件", "会议", "文档", "方法"}
    filtered = set()
    for name in people:
        ok = True
        for kw in query_keywords:
            if kw in name and len(name) > len(kw):
                # 实体中包含 query_keyword → 去噪
                clean = name.replace(kw, "")
                if len(clean) >= 2:
                    filtered.add(clean)
                ok = False
                break
        if ok:
            filtered.add(name)
    people = filtered

    return list(people)


def extract_date_mentions(text: str) -> List[str]:
    """提取日期提及。

    支持格式：
    - YYYY年MM月DD日
    - YYYY-MM-DD / YYYY/MM/DD
    - MM月DD日
    - 今天、昨天、前天、上周等
    """
    dates = []

    patterns = [
        r"\d{4}年\d{1,2}月\d{1,2}日",
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",
        r"\d{1,2}月\d{1,2}日",
    ]
    for pat in patterns:
        dates.extend(re.findall(pat, text))

    relative_words = ["今天", "昨天", "前天", "上周", "上个月", "明天", "下周"]
    dates.extend([w for w in relative_words if w in text])

    return dates


def extract_topics(text: str, top_n: int = 5) -> List[str]:
    """使用 TF-IDF 提取主题关键词。

    配合 jieba 分词，取 TF 最高的名词/动词。
    """
    words = jieba.lcut(text)
    # 过滤：只保留有意义的词（长度 >= 2）
    meaningful = [
        w for w in words
        if len(w) >= 2
        and not w.isspace()
        and not all(c in "，。！？；：""''（）【】《》…—·" for c in w)
    ]

    counter = Counter(meaningful)
    # 取 top_n
    return [word for word, _ in counter.most_common(top_n)]


def extract_query_entities(query: str) -> Dict[str, List[str]]:
    """从用户查询中提取实体（用于检索加权）。"""
    return extract_entities(query)


def boost_by_entity_overlap(
    query_entities: Dict[str, List[str]],
    doc_metadata: Dict,
    boost_factor: float = 1.2,
) -> float:
    """根据查询实体和文档实体重叠程度计算加权因子。

    Args:
        query_entities: 查询中提取的实体
        doc_metadata: 文档元数据（含 participants, topics 等字段）
        boost_factor: 每命中一个实体的加权倍数

    Returns:
        加权因子（≥ 1.0）
    """
    boost = 1.0

    # 人名匹配：查询中提到的人名 vs 文档参与者
    query_people = set(query_entities.get("people", []))
    doc_people = _get_metadata_set(doc_metadata, "participants")
    people_hits = len(query_people & doc_people)
    boost *= boost_factor ** people_hits

    # 主题匹配：查询主题词 vs 文档主题词
    query_topics = set(query_entities.get("topics", []))
    doc_topics = _get_metadata_set(doc_metadata, "topics")
    topic_hits = len(query_topics & doc_topics)
    boost *= (boost_factor * 0.5) ** topic_hits  # 主题权重较低

    return boost


def _get_metadata_set(metadata: Dict, key: str) -> set:
    """安全地从元数据中获取集合值。"""
    val = metadata.get(key, "")
    if isinstance(val, list):
        return set(val)
    if isinstance(val, str) and val:
        return set(val.split(", "))
    return set()
