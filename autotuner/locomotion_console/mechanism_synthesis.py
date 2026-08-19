"""兼容入口；机制合成器的权威实现已移出 Web 控制台。"""

from autotuner.mechanisms import mechanism_synthesis as _implementation
from autotuner.mechanisms.mechanism_synthesis import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_implementation, name)

