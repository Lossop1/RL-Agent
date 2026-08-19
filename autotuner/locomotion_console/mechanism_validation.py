"""兼容入口；机制校验器的权威实现已移出 Web 控制台。"""

from autotuner.mechanisms import mechanism_validation as _implementation
from autotuner.mechanisms.mechanism_validation import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_implementation, name)

