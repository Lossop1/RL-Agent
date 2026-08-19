"""兼容入口；权威实现位于 :mod:`autotuner.taili_ops.early_stop`。"""

from autotuner.taili_ops import early_stop as _implementation
from autotuner.taili_ops.early_stop import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_implementation, name)
