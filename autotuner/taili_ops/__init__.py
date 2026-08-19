"""兼容入口；Taili 专用运维已迁至 products.taili.ops。"""
from __future__ import annotations

from products.taili import ops as _implementation

__path__ = _implementation.__path__
__all__ = getattr(_implementation, "__all__", ())


def __getattr__(name: str):
    return getattr(_implementation, name)
