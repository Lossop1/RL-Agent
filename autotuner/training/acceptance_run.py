"""兼容入口；Taili 验收命令已迁至 :mod:`products.taili.ops.acceptance_run`。"""

from products.taili.ops import acceptance_run as _implementation
from products.taili.ops.acceptance_run import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    raise SystemExit(_implementation.main())
