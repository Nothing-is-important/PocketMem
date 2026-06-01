"""微信聊天记录导出解析器。

支持微信桌面版导出的 TXT 格式：
    YYYY-MM-DD HH:MM:SS SenderName
    Message content (single or multi-line)

也兼容以下变体格式：
    - YYYY/MM/DD HH:MM:SS SenderName
    - MM-DD HH:MM:SS SenderName
    - 系统消息（无发送者）
    - [Image], [File], [Video] 等媒体标记
    - 引用回复消息

解析为结构化 ChatMessage 列表。
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class ChatMessage:
    timestamp: datetime
    sender: str
    content: str
    msg_type: str = "text"  # text | system | media | file
    chat_name: str = ""     # 聊天对象名称或群名
    chat_type: str = "private"  # private | group

    def to_text(self) -> str:
        """返回适合检索展示的文本表示。"""
        return f"[{self.timestamp:%Y-%m-%d %H:%M}] {self.sender}: {self.content}"


# 消息头部正则：匹配 "日期 时间 发送者" 格式
# 支持: YYYY-MM-DD, YYYY/MM/DD, MM-DD 等变体
_MSG_HEADER_PATTERN = re.compile(
    r"^(\d{2,4}[-/]\d{1,2}[-/]\d{1,2})\s+"   # 日期
    r"(\d{1,2}:\d{2}(?::\d{2})?)\s+"           # 时间
    r"(.+?)$"                                    # 发送者
)

# 系统消息关键词
_SYSTEM_KEYWORDS = [
    "开启了朋友验证", "已添加", "以上是打招呼的内容",
    "邀请", "加入了群聊", "退出了群聊", "修改群名为",
    "撤回了一条消息", "你无法邀请", "被移除",
]

# 媒体标记
_MEDIA_PATTERN = re.compile(
    r"^\[(图片|照片|视频|语音|文件|动画表情|链接|小程序|转账|红包|位置|名片|语音通话|视频通话)\]$"
)


def parse_wechat_export(filepath: str, chat_name: str = "") -> List[ChatMessage]:
    """解析微信导出的聊天记录 TXT 文件。

    Args:
        filepath: 微信导出 TXT 文件路径
        chat_name: 聊天对象名称（如"张三"或"技术交流群"），
                   不提供时从文件名推断

    Returns:
        ChatMessage 列表
    """
    if not chat_name:
        import os
        basename = os.path.splitext(os.path.basename(filepath))[0]
        chat_name = basename

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    messages: List[ChatMessage] = []
    current_msg: Optional[dict] = None

    for line in lines:
        line = line.rstrip("\n").rstrip("\r")

        # 跳过空行和纯分隔线
        if not line.strip():
            if current_msg is not None:
                _finalize_message(messages, current_msg)
                current_msg = None
            continue

        # 尝试匹配消息头
        header_match = _MSG_HEADER_PATTERN.match(line.strip())

        if header_match and _is_valid_header(line.strip()):
            # 保存前一条消息
            if current_msg is not None:
                _finalize_message(messages, current_msg)

            date_str, time_str, sender = header_match.groups()
            sender = sender.strip()

            timestamp = _parse_datetime(date_str, time_str)
            chat_type = _infer_chat_type(sender, chat_name)

            current_msg = {
                "timestamp": timestamp,
                "sender": sender,
                "lines": [],
                "chat_type": chat_type,
            }
        elif current_msg is not None:
            # 续行：多行消息内容
            current_msg["lines"].append(line.strip())

    # 保存最后一条消息
    if current_msg is not None:
        _finalize_message(messages, current_msg)

    return messages


def _is_valid_header(line: str) -> bool:
    """检查匹配到的头部是否像是一个真实的消息头（而非正文中的类似格式）。"""
    match = _MSG_HEADER_PATTERN.match(line)
    if not match:
        return False
    date_str = match.group(1)
    # 日期中应该有分隔符和合理的数字
    return bool(re.search(r"\d{2,4}[-/]\d{1,2}[-/]\d{1,2}", date_str))


def _parse_datetime(date_str: str, time_str: str) -> datetime:
    """解析日期时间字符串，支持多种格式。"""
    date_str = date_str.replace("/", "-")
    time_str = time_str.strip()

    # 统一为 YYYY-MM-DD
    parts = date_str.split("-")
    if len(parts[0]) == 2:  # YY-MM-DD → 20YY-MM-DD
        date_str = f"20{date_str}"

    datetime_str = f"{date_str} {time_str}"

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(datetime_str, fmt)
        except ValueError:
            continue

    # 最终回退：只用日期，时间默认为 00:00:00
    return datetime.strptime(date_str, "%Y-%m-%d")


def _infer_chat_type(sender: str, chat_name: str) -> str:
    """推断聊天类型（私聊/群聊）。"""
    # 如果发送者名称看起来像群名，或者是已知群特征
    group_indicators = ["群", "交流", "通知", "项目组", "团队", "讨论组"]
    if any(indicator in chat_name for indicator in group_indicators):
        return "group"
    if any(indicator in sender for indicator in group_indicators):
        return "group"
    # 如果发送者和聊天名不同，可能是群聊
    if sender != chat_name and chat_name:
        return "group"
    return "private"


def _finalize_message(messages: List[ChatMessage], msg_data: dict):
    """将收集的消息数据转为 ChatMessage 并加入列表。"""
    content = "\n".join(msg_data["lines"]).strip()
    if not content:
        return

    msg_type = _classify_message(content)

    messages.append(ChatMessage(
        timestamp=msg_data["timestamp"],
        sender=msg_data["sender"],
        content=content,
        msg_type=msg_type,
        chat_type=msg_data["chat_type"],
    ))


def _classify_message(content: str) -> str:
    """分类消息类型。"""
    content_stripped = content.strip()

    # 媒体消息
    if _MEDIA_PATTERN.match(content_stripped):
        if "图片" in content_stripped or "照片" in content_stripped:
            return "image"
        if "视频" in content_stripped:
            return "video"
        if "文件" in content_stripped:
            return "file"
        if "语音" in content_stripped:
            return "voice"
        return "media"

    # 系统消息
    for keyword in _SYSTEM_KEYWORDS:
        if keyword in content_stripped:
            return "system"

    return "text"


def filter_text_messages(messages: List[ChatMessage]) -> List[ChatMessage]:
    """过滤出纯文本消息（去除系统消息和媒体标记）。"""
    return [m for m in messages if m.msg_type == "text"]
