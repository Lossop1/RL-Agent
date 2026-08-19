"""兼容入口；研究状态模型的权威实现位于 :mod:`autotuner.research`。"""

from autotuner.research import research_state as _implementation
from autotuner.research.research_state import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_implementation, name)

