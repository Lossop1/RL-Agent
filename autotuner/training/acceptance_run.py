"""兼容入口；Taili 验收命令已迁至 :mod:`autotuner.taili_ops.acceptance_run`。"""

from autotuner.taili_ops import acceptance_run as _implementation
from autotuner.taili_ops.acceptance_run import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    raise SystemExit(_implementation.main())
