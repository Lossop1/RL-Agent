"""兼容入口；Taili payload 构建实现已迁至 products.taili.payload。"""
from __future__ import annotations

from products.taili import payload as _implementation

__path__ = _implementation.__path__
__all__ = getattr(_implementation, "__all__", ())


def __getattr__(name: str):
    return getattr(_implementation, name)
