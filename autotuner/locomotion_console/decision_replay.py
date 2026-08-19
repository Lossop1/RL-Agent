"""兼容入口；决策回放的权威实现位于 :mod:`autotuner.research`。"""

from autotuner.research import decision_replay as _implementation
from autotuner.research.decision_replay import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_implementation, name)

