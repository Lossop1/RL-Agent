"""兼容入口；结果学习的权威实现位于 :mod:`autotuner.research`。"""

from autotuner.research import outcome_learning as _implementation
from autotuner.research.outcome_learning import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_implementation, name)

