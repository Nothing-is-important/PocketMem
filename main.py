"""PocketMemory —— 端侧个人记忆助手 Agent。

启动 FastAPI 服务：
    python main.py
    uv run python main.py

指定端口：
    POCKET_PORT=8080 python main.py
"""

import uvicorn

from config import get_settings


def main():
    settings = get_settings()
    print(f"Starting PocketMemory server on {settings.server_host}:{settings.server_port}")
    uvicorn.run(
        "api.server:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=True,
    )


if __name__ == "__main__":
    main()
