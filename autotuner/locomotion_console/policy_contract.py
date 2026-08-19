"""兼容入口；策略交付契约的权威实现位于 :mod:`autotuner.research`。"""

from autotuner.research import policy_contract as _implementation
from autotuner.research.policy_contract import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    raise SystemExit(_implementation.main())

