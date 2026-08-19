"""兼容入口；研究执行监督器的权威实现位于 :mod:`autotuner.research`。"""

from autotuner.research import research_supervisor as _implementation
from autotuner.research.research_supervisor import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_implementation, name)

