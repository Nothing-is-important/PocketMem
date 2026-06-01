# WeChat Database Decryption Skill

## Purpose
Guide the implementation of automatic WeChat database decryption for PocketMemory. This skill contains all known technical details, code templates, and reference implementations.

## Technical Background

### WeChat PC Database Storage
- **Location (Windows):** `Documents\WeChat Files\[wxid]\Msg\Multi\MSG0.db`
- **Location (WeChat 4.x):** `Documents\xwechat_files\[wxid]\db_storage\message\message_*.db`
- **Encryption:** SQLCipher 4 (AES-256-CBC, HMAC-SHA512, PBKDF2-HMAC-SHA1, 4000 iterations)
- **Key format:** 64-char hex string, cached in WeChat process memory as `x'<64hex_key><32hex_salt>'`
- **Page size:** 4096 bytes for SQLCipher 4

### Database Schema (After Decryption)
- `message/message_*.db`: Chat messages (msg_id, talker, content, type, create_time)
- `contact/contact.db`: Contacts (username, alias, nickname, remark)
- `session/session.db`: Session list with latest message summaries

### Message Types
- Type 1: Text
- Type 3: Image
- Type 34: Voice
- Type 43: Video
- Type 47: Sticker
- Type 49: Shared link / file
- Type 10000: System notification

## Key Extraction Methods

### Method A: Memory Scan (Windows, requires pymem)
```python
import pymem
import pymem.process

def extract_key_windows():
    """Extract SQLCipher key from Weixin.exe process memory."""
    pm = pymem.Pymem("Weixin.exe")
    wechat_module = pymem.process.module_from_name(
        pm.process_handle, "WeChatWin.dll"
    )
    if not wechat_module:
        print("WeChatWin.dll not found. Make sure WeChat is logged in.")
        return None
    
    module_base = wechat_module.lpBaseOfDll
    module_size = wechat_module.SizeOfImage
    
    # Key pattern: x'<64 hex chars><32 hex chars>'
    # Search near known markers in WeChatWin.dll
    chunk_size = 0x100000  # 1MB chunks
    phone_pattern = b"iphone\x00"
    
    for offset in range(0, module_size, chunk_size):
        try:
            chunk = pm.read_bytes(module_base + offset, min(chunk_size, module_size - offset))
        except:
            continue
        
        idx = 0
        while True:
            idx = chunk.find(phone_pattern, idx)
            if idx == -1:
                break
            # Key is approximately 0x70 bytes before "iphone\x00"
            key_offset = idx - 0x70
            if key_offset >= 0:
                key = chunk[key_offset:key_offset + 64]
                if len(key) == 64 and key != b"\x00" * 64:
                    return key.hex()
            idx += 1
    
    return None
```

### Method B: Static Analysis (if key is stored in config)
```python
def extract_key_from_config():
    """Try known key locations from WeChat config files."""
    import sqlite3
    from pathlib import Path
    import hashlib
    
    # Some versions store key derivation data
    # Key = MD5(IMEI + UIN)[:7] for older versions
    # For WeChat 4.x, key is in process memory only
    
    config_paths = [
        Path.home() / "Documents/WeChat Files/All Users/config",
        Path.home() / "Documents/xwechat_files/config",
    ]
    
    for path in config_paths:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                # Try to find hex key patterns
                import re
                hex_keys = re.findall(r"[0-9a-fA-F]{64}", content)
                if hex_keys:
                    return hex_keys[0]
    return None
```

## Database Decryption Code Template

```python
import os
from Crypto.Hash import HMAC, SHA1
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Cipher import AES

PAGE_SIZE = 4096
SQLITE_HEADER = b"SQLite format 3\x00"

def decrypt_wechat_db(db_path: str, key_hex: str, output_path: str) -> bool:
    """Decrypt a single WeChat SQLCipher 4 database."""
    key_bytes = bytes.fromhex(key_hex)
    
    with open(db_path, "rb") as f:
        raw = f.read()
    
    if len(raw) < PAGE_SIZE:
        return False
    
    # Derive encryption key from raw key
    salt = raw[:16]  # First 16 bytes are the salt
    key = PBKDF2(
        key_bytes, salt, dkLen=32, count=4000,
        prf=lambda p, s: HMAC.new(p, s, SHA1).digest()
    )
    
    output = bytearray()
    
    for page_num in range(len(raw) // PAGE_SIZE):
        page = raw[page_num * PAGE_SIZE : (page_num + 1) * PAGE_SIZE]
        
        if page_num == 0:
            # Page 1: header page
            iv = page[16:32]
            cipher = AES.new(key, AES.MODE_CBC, iv)
            decrypted = cipher.decrypt(page[32:PAGE_SIZE - 32])
            output.extend(SQLITE_HEADER)
            output.extend(decrypted[len(SQLITE_HEADER):])
            output.extend(b"\x00" * 32)
        else:
            # Subsequent pages: IV is at the end
            iv = page[-48:-32]
            cipher = AES.new(key, AES.MODE_CBC, iv)
            decrypted = cipher.decrypt(page[:PAGE_SIZE - 48])
            output.extend(decrypted)
            output.extend(b"\x00" * 48)
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(output)
    
    # Verify decryption succeeded
    import sqlite3
    try:
        conn = sqlite3.connect(output_path)
        conn.execute("SELECT name FROM sqlite_master LIMIT 1")
        conn.close()
        return True
    except sqlite3.DatabaseError:
        os.remove(output_path)
        return False


def read_messages(decrypted_db_path: str) -> list:
    """Read chat messages from a decrypted WeChat database."""
    import sqlite3
    messages = []
    conn = sqlite3.connect(decrypted_db_path)
    
    # WeChat 4.x schema (message/message_*.db)
    cursor = conn.execute("""
        SELECT 
            msv_order, talker, content, type, create_time
        FROM message
        WHERE type = 1  -- Text messages only
        ORDER BY create_time ASC
    """)
    
    for row in cursor.fetchall():
        messages.append({
            "id": row[0],
            "talker": row[1],
            "content": row[2],
            "type": row[3],
            "time": row[4],
        })
    
    conn.close()
    return messages
```

## Integration Into PocketMemory

### Architecture
```
pocketmemory/
├── data_ingestion/
│   ├── wechat_detector.py      ✅ Already implemented
│   ├── wechat_decryptor.py     🔜 NEW: Decrypt MSG0.db using key
│   └── source_manager.py       Update: Add "wechat_db" source type
```

### Implementation Plan

1. **Install dependency:** `uv add pycryptodome pymem`
2. **Create key extractor:** `data_ingestion/wechat_key.py`
   - Try memory scan first (requires admin)
   - Fall back to config file search
3. **Create decryptor:** `data_ingestion/wechat_decryptor.py`
   - Decrypt MSG0.db to temp directory
   - Read messages and format as TXT
   - Feed to existing pipeline
4. **Add API endpoint:** `POST /wechat/import`
   - Extract key → Decrypt → Parse → Index
   - Return progress/status
5. **Update frontend:** Add "导入微信数据" button when `can_import: true`

### Dependencies
```bash
uv add pycryptodome pymem
```

### Reference Projects
- github.com/L1en2407/wechat-decrypt — WeChat 4.x decryptor
- github.com/ylytdeng/wechat-decrypt — Most comprehensive, supports export
- github.com/titanwings/ex-skill — Windows key extraction example
- github.com/ZedeX/weixin-decrypte-script — API server approach

## Known Issues & Edge Cases

1. **WeChat version changes:** 3.x used SQLCipher 3 (different page layout), 4.x uses SQLCipher 4
2. **Admin privileges:** Memory scanning requires running as administrator on Windows
3. **WeChat must be logged in:** Key only exists in memory when WeChat is logged in
4. **Multiple WeChat accounts:** `Documents/WeChat Files/` may have multiple `wxid_*` directories
5. **Database locked:** MSG0.db may be locked while WeChat is writing to it → copy to temp first
6. **ZSTD compression:** WeChat 4.x content may be ZSTD-compressed in the `content` column
