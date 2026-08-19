"""兼容入口；Taili 任务实现已迁至 products.taili.blind_locomotion。"""
from __future__ import annotations

from products.taili import blind_locomotion as _implementation

__path__ = _implementation.__path__
__all__ = getattr(_implementation, "__all__", ())


def __getattr__(name: str):
    return getattr(_implementation, name)
