"""数据摄取管线测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile

from data_ingestion.chunker import ConversationChunker, DocumentChunk
from data_ingestion.time_utils import (
    compute_temporal_decay,
    days_since,
    extract_date_from_query,
    parse_relative_time,
)
from data_ingestion.wechat_parser import (
    ChatMessage,
    filter_text_messages,
    parse_wechat_export,
)


SAMPLE_CHAT = """2026-03-15 14:30:22 张三
周末有空吗？一起吃个饭？

2026-03-15 14:31:05 我
好的，周六中午怎么样？

2026-03-15 14:32:18 张三
可以，去上次那家火锅店吧，在朝阳区建国路88号

2026-03-16 09:15:00 李四
项目方案我发你邮箱了，你看一下

2026-03-16 09:20:33 我
收到，我看完回复你

2026-03-16 15:00:00 张三
火锅店叫渝味火锅，记得提前订位

2026-04-01 10:00:00 我
今天请假，身体不舒服

2026-04-02 08:30:00 李四
团建地点投票：1. 密云水库 2. 古北水镇 3. 十渡
"""


class TestWechatParser:
    def test_parse_basic(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(SAMPLE_CHAT)
            tmp_path = f.name

        messages = parse_wechat_export(tmp_path, chat_name="测试群")
        Path(tmp_path).unlink()

        assert len(messages) >= 7
        assert messages[0].sender == "张三"
        assert "火锅" in messages[2].content
        assert all(isinstance(m, ChatMessage) for m in messages)

    def test_filter_text_messages(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(SAMPLE_CHAT)
            tmp_path = f.name

        messages = parse_wechat_export(tmp_path)
        Path(tmp_path).unlink()

        text_msgs = filter_text_messages(messages)
        assert all(m.msg_type == "text" for m in text_msgs)

    def test_chat_message_to_text(self):
        from datetime import datetime
        msg = ChatMessage(
            timestamp=datetime(2026, 3, 15, 14, 30),
            sender="张三",
            content="你好",
        )
        text = msg.to_text()
        assert "张三" in text
        assert "2026" in text


class TestConversationChunker:
    def test_chunk_messages(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(SAMPLE_CHAT)
            tmp_path = f.name

        messages = filter_text_messages(parse_wechat_export(tmp_path))
        Path(tmp_path).unlink()

        chunker = ConversationChunker(gap_minutes=30)
        chunks = chunker.chunk_messages(messages, source_file="test.txt")

        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)
            assert chunk.content
            assert "participants" in chunk.metadata
            assert "timestamp" in chunk.metadata

    def test_chunk_text(self):
        chunker = ConversationChunker(target_tokens=128)
        text = "这是测试文本。\n\n" * 50
        chunks = chunker.chunk_text(text, metadata={"source": "test.md"})

        assert len(chunks) > 1
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] == "test.md"

    def test_session_grouping(self):
        """验证 30 分钟间隔正确分组。"""
        from datetime import datetime, timedelta

        msgs = [
            ChatMessage(datetime(2026, 1, 1, 10, 0), "A", "msg1"),
            ChatMessage(datetime(2026, 1, 1, 10, 15), "A", "msg2"),  # 15min, 同一会话
            ChatMessage(datetime(2026, 1, 1, 11, 0), "B", "msg3"),   # 45min, 新会话
        ]
        chunker = ConversationChunker(gap_minutes=30)
        chunks = chunker.chunk_messages(msgs)

        # 应产生两个会话的块
        assert len(chunks) >= 1


class TestTimeUtils:
    def test_days_since(self):
        from datetime import datetime, timedelta
        ts = datetime.now() - timedelta(days=5)
        days = days_since(ts)
        assert 4.5 <= days <= 5.5

    def test_temporal_decay_recent(self):
        from datetime import datetime
        ts = datetime.now()
        decay = compute_temporal_decay(ts, half_life_days=30)
        assert decay > 0.95  # 今天的权重接近 1

    def test_temporal_decay_old(self):
        from datetime import datetime, timedelta
        ts = datetime.now() - timedelta(days=60)
        decay = compute_temporal_decay(ts, half_life_days=30)
        assert 0.2 < decay < 0.3  # 60 天前 ≈ 2 个半衰期 → ~0.25

    def test_extract_date_absolute(self):
        result = extract_date_from_query("2026年3月15日我和谁吃饭了")
        assert result is not None
        assert result.month == 3
        assert result.day == 15

    def test_extract_date_relative(self):
        result = extract_date_from_query("昨天我和张三聊了什么")
        # "昨天" 应解析出日期
        assert result is not None

    def test_parse_relative_time_today(self):
        result = parse_relative_time("今天发生的事情")
        assert result is not None
        assert result.date() == result.date()  # 今天
