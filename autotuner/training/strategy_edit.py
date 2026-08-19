"""兼容入口；权威实现位于 :mod:`products.taili.ops.strategy_edit`。"""

from products.taili.ops import strategy_edit as _implementation
from products.taili.ops.strategy_edit import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_implementation, name)
