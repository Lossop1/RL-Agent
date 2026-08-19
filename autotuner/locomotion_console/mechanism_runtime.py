"""兼容入口；机制运行时的权威实现已移出 Web 控制台。"""

from autotuner.mechanisms import mechanism_runtime as _implementation
from autotuner.mechanisms.mechanism_runtime import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_implementation, name)

