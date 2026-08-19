"""兼容入口；持久研究记忆的权威实现位于 :mod:`autotuner.research`。"""

from autotuner.research import campaign_journal as _implementation
from autotuner.research.campaign_journal import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_implementation, name)
