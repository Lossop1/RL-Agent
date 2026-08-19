"""兼容入口；权威实现位于 :mod:`products.taili.ops.early_stop`。"""

from products.taili.ops import early_stop as _implementation
from products.taili.ops.early_stop import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_implementation, name)
