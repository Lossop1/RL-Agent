"""兼容入口；研究台账的权威实现位于 :mod:`autotuner.research`。"""

from autotuner.research import research_ledger as _implementation
from autotuner.research.research_ledger import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    raise SystemExit(_implementation.main())

