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
    # AUTO-RELOAD default OFF (0708): reload=True (uvicorn's file-watching reloader) destabilised the running
    # backend when source files were edited live (reload mid-edit → crash / "进不去系统"). Default OFF = the
    # known-good stable behavior; a restart loads new code. Opt in with LOCOMOTION_CONSOLE_RELOAD=1 for dev.
    reload = os.environ.get("LOCOMOTION_CONSOLE_RELOAD", "0") in ("1", "true", "True")
    uvicorn.run("autotuner.locomotion_console.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
