"""微信数据库解密器。

使用 SQLCipher 4 密钥解密微信 MSG 数据库，读取消息内容。

参考：opencode/skills/wechat-db-decrypt.skill.md
"""

import os
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

try:
    from Crypto.Hash import HMAC, SHA1
    from Crypto.Protocol.KDF import PBKDF2
    from Crypto.Cipher import AES
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False


PAGE_SIZE = 4096
SQLITE_HEADER = b"SQLite format 3\x00"


@dataclass
class WechatMessage:
    """解密后的微信消息。"""
    msg_id: int
    talker: str         # 对话对象（wxid 或群 ID）
    content: str        # 消息内容
    msg_type: int       # 1=文本, 3=图片, 34=语音, 43=视频, 47=表情, 49=链接
    create_time: int    # Unix 时间戳
    is_group: bool = False

    @property
    def timestamp(self) -> datetime:
        return datetime.fromtimestamp(self.create_time)

    def to_text(self) -> str:
        return f"[{self.timestamp:%Y-%m-%d %H:%M}] {self.talker}: {self.content}"


def decrypt_database(db_path: str, key_hex: str, output_path: str = None) -> Optional[str]:
    """解密微信 SQLCipher 4 数据库。

    Args:
        db_path: 加密的 .db 文件路径
        key_hex: 64 字符 hex 密钥
        output_path: 输出路径（默认：临时文件）

    Returns:
        解密后的数据库文件路径，失败返回 None
    """
    if not HAS_CRYPTO:
        return None

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".db", prefix="wechat_decrypted_")
        os.close(fd)

    key_bytes = bytes.fromhex(key_hex)

    try:
        with open(db_path, "rb") as f:
            raw = f.read()
    except OSError:
        return None

    if len(raw) < PAGE_SIZE:
        return None

    # 从原始密钥派生加密密钥
    salt = raw[:16]
    key = PBKDF2(
        key_bytes, salt, dkLen=32, count=4000,
        prf=lambda p, s: HMAC.new(p, s, SHA1).digest(),
    )

    output = bytearray()

    for page_num in range(len(raw) // PAGE_SIZE):
        page = raw[page_num * PAGE_SIZE:(page_num + 1) * PAGE_SIZE]

        if page_num == 0:
            iv = page[16:32]
            cipher = AES.new(key, AES.MODE_CBC, iv)
            decrypted = cipher.decrypt(page[32:PAGE_SIZE - 32])
            output.extend(SQLITE_HEADER)
            output.extend(decrypted[len(SQLITE_HEADER):])
            output.extend(b"\x00" * 32)
        else:
            iv = page[-48:-32]
            cipher = AES.new(key, AES.MODE_CBC, iv)
            decrypted = cipher.decrypt(page[:PAGE_SIZE - 48])
            output.extend(decrypted)
            output.extend(b"\x00" * 48)

    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(output)
    except OSError:
        return None

    # 验证解密成功
    try:
        conn = sqlite3.connect(output_path)
        conn.execute("SELECT name FROM sqlite_master LIMIT 1")
        conn.close()
        return output_path
    except sqlite3.DatabaseError:
        try:
            os.remove(output_path)
        except OSError:
            pass
        return None


def read_messages(decrypted_db_path: str) -> List[WechatMessage]:
    """从解密后的数据库读取文本消息。

    Args:
        decrypted_db_path: 解密后的 .db 文件路径

    Returns:
        WechatMessage 列表，按时间排序
    """
    messages = []

    try:
        conn = sqlite3.connect(decrypted_db_path)
    except sqlite3.DatabaseError:
        return messages

    # 检测表结构（WeChat 3.x vs 4.x）
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='message'"
    )
    has_message_table = cursor.fetchone() is not None

    if has_message_table:
        # WeChat 4.x: message 表
        columns = _get_columns(conn, "message")
        query = _build_message_query(columns)
    else:
        # WeChat 3.x: 可能使用 msg 表
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='msg'"
        )
        if cursor.fetchone():
            columns = _get_columns(conn, "msg")
            query = _build_message_query(columns, table="msg")
        else:
            conn.close()
            return messages

    try:
        for row in conn.execute(query):
            try:
                content = row[2] if row[2] else ""
                # 尝试 ZSTD 解压（WeChat 4.x 某些版本）
                if content and _looks_compressed(content):
                    content = _try_decompress(content)
            except (IndexError, TypeError):
                continue

            messages.append(WechatMessage(
                msg_id=row[0] if row[0] else 0,
                talker=row[1] if row[1] else "",
                content=str(content) if content else "",
                msg_type=row[3] if len(row) > 3 and row[3] else 1,
                create_time=row[4] if len(row) > 4 and row[4] else 0,
                is_group="@" in str(row[1]) if row[1] else False,
            ))
    except sqlite3.DatabaseError:
        pass
    finally:
        conn.close()

    return messages


def export_to_text(messages: List[WechatMessage], output_path: str,
                   chat_name: str = "") -> str:
    """将消息导出为微信 TXT 格式（兼容现有解析管线）。

    Args:
        messages: WechatMessage 列表
        output_path: 输出文件路径
        chat_name: 聊天对象名称

    Returns:
        输出文件路径
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for msg in messages:
            sender = msg.talker
            if chat_name:
                sender = chat_name
            f.write(f"{msg.timestamp:%Y-%m-%d %H:%M:%S} {sender}\n")
            f.write(f"{msg.content}\n\n")

    return output_path


def _get_columns(conn, table: str) -> List[str]:
    """获取表的列名列表。"""
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cursor.fetchall()]


def _build_message_query(columns: List[str], table: str = "message") -> str:
    """根据实际列名构建消息查询。"""
    # 常见列名映射
    col_map = {
        "msg_id": ["msg_id", "msv_order", "msgid", "id"],
        "talker": ["talker", "sender", "strtalker"],
        "content": ["content", "message", "strcontent"],
        "type": ["type", "msg_type", "messagetype"],
        "time": ["create_time", "createtime", "timestamp", "time"],
    }

    def _find_col(candidates):
        for c in candidates:
            if c in columns:
                return c
        return None

    col_id = _find_col(col_map["msg_id"]) or "rowid"
    col_talker = _find_col(col_map["talker"]) or "''"
    col_content = _find_col(col_map["content"]) or "''"
    col_type = _find_col(col_map["type"]) or "1"
    col_time = _find_col(col_map["time"]) or "0"

    return (
        f"SELECT {col_id}, {col_talker}, {col_content}, {col_type}, {col_time} "
        f"FROM {table} "
        f"WHERE {col_type} = 1 "  # 只要文本消息
        f"ORDER BY {col_time} ASC"
    )


def _looks_compressed(content: str) -> bool:
    """检测内容是否像是压缩过的（二进制数据）。"""
    if not content:
        return False
    # ZSTD 压缩的内容通常包含不可打印字符
    try:
        content.encode("ascii")
        return False
    except UnicodeEncodeError:
        return len(content) > 20


def _try_decompress(content: str) -> str:
    """尝试 ZSTD 解压。"""
    try:
        import zstd
        raw = content.encode("latin-1", errors="replace")
        return zstd.decompress(raw).decode("utf-8", errors="replace")
    except (ImportError, Exception):
        pass
    return content
