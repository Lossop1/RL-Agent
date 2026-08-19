"""兼容入口；研究周期编排的权威实现位于 :mod:`autotuner.research`。"""

from autotuner.research import research_cycle as _implementation
from autotuner.research.research_cycle import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_implementation, name)

