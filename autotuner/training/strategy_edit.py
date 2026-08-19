"""兼容入口；权威实现位于 :mod:`autotuner.taili_ops.strategy_edit`。"""

from autotuner.taili_ops import strategy_edit as _implementation
from autotuner.taili_ops.strategy_edit import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_implementation, name)
