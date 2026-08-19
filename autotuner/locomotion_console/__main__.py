"""Entrypoint for the locomotion console backend.

Serves the FastAPI app on :8000. If config/ssh.json contains ssh_host the
backend starts in real mode by default. Set LOCOMOTION_CONSOLE_SOURCE=fake
to force demo data.
"""
from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("LOCOMOTION_CONSOLE_HOST", "127.0.0.1")
    port = int(os.environ.get("LOCOMOTION_CONSOLE_PORT", "8000"))
    # 默认关闭热重载，避免编辑过程中加载半成品模块；开发者可显式开启。
    reload = os.environ.get("LOCOMOTION_CONSOLE_RELOAD", "0") in ("1", "true", "True")
    uvicorn.run("autotuner.locomotion_console.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
