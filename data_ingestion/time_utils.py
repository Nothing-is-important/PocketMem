"""时间工具函数。"""

import re
from datetime import datetime, timedelta
from typing import Optional


# 相对时间表达模式
_RELATIVE_TIME_PATTERNS = [
    (re.compile(r"(\d+)\s*天前"), lambda n: timedelta(days=int(n))),
    (re.compile(r"(\d+)\s*周前"), lambda n: timedelta(weeks=int(n))),
    (re.compile(r"(\d+)\s*个月(?:前|之前)"), lambda n: timedelta(days=int(n) * 30)),
    (re.compile(r"(\d+)\s*小时前"), lambda n: timedelta(hours=int(n))),
    (re.compile(r"(\d+)\s*分钟前"), lambda n: timedelta(minutes=int(n))),
    (re.compile(r"昨天"), lambda _: timedelta(days=1)),
    (re.compile(r"前天"), lambda _: timedelta(days=2)),
    (re.compile(r"上周"), lambda _: timedelta(weeks=1)),
    (re.compile(r"上(?:个)?月"), lambda _: timedelta(days=30)),
    (re.compile(r"今天"), lambda _: timedelta(days=0)),
    (re.compile(r"最近"), lambda _: timedelta(days=7)),
]


def parse_relative_time(text: str, reference: Optional[datetime] = None) -> Optional[datetime]:
    """从文本中解析相对时间表达式。

    Args:
        text: 包含时间表达的文本
        reference: 参考时间（默认为当前时间）

    Returns:
        解析后的绝对时间，无法解析时返回 None
    """
    ref = reference or datetime.now()

    for pattern, delta_fn in _RELATIVE_TIME_PATTERNS:
        match = pattern.search(text)
        if match:
            group = match.group(1) if match.lastindex else "1"
            delta = delta_fn(group)
            return ref - delta

    return None


def days_since(timestamp: datetime, reference: Optional[datetime] = None) -> float:
    """计算某个时间戳距今多少天。"""
    ref = reference or datetime.now()
    return (ref - timestamp).total_seconds() / 86400.0


def compute_temporal_decay(
    timestamp: datetime,
    half_life_days: float = 30.0,
    reference: Optional[datetime] = None,
) -> float:
    """计算时间衰减权重。

    公式: exp(-λ * days_since)，λ = ln(2) / half_life_days

    Args:
        timestamp: 文档时间戳
        half_life_days: 半衰期（天）
        reference: 参考时间

    Returns:
        衰减权重 (0.0 ~ 1.0)
    """
    import math
    lam = math.log(2) / half_life_days
    return math.exp(-lam * days_since(timestamp, reference))


def extract_date_from_query(query: str) -> Optional[datetime]:
    """从用户查询中提取日期信息。"""
    # 尝试匹配绝对日期: YYYY-MM-DD, YYYY/MM/DD, MM-DD
    abs_patterns = [
        re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})"),
        re.compile(r"(\d{1,2})月(\d{1,2})[日号]"),
    ]

    for pattern in abs_patterns:
        match = pattern.search(query)
        if match:
            try:
                groups = match.groups()
                if len(groups[0]) == 4:  # YYYY-MM-DD
                    return datetime(int(groups[0]), int(groups[1]), int(groups[2]))
                else:  # MM-DD
                    year = datetime.now().year
                    return datetime(year, int(groups[0]), int(groups[1]))
            except ValueError:
                continue

    # 尝试相对时间
    return parse_relative_time(query)


def format_time_range(earliest: datetime, latest: datetime) -> str:
    """格式化时间范围为可读字符串。"""
    if earliest.date() == latest.date():
        return earliest.strftime("%Y年%m月%d日")
    return f"{earliest:%Y年%m月%d日} 至 {latest:%Y年%m月%d日}"
