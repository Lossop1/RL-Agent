"""兼容入口；Taili 核心实现已迁至 products.taili.core。"""
from __future__ import annotations

from products.taili import core as _implementation

__path__ = _implementation.__path__
__all__ = getattr(_implementation, "__all__", ())


def __getattr__(name: str):
    return getattr(_implementation, name)
