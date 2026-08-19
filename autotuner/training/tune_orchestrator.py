"""兼容入口；权威实现位于 :mod:`autotuner.taili_ops.tune_orchestrator`。"""

from autotuner.taili_ops import tune_orchestrator as _implementation
from autotuner.taili_ops.tune_orchestrator import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    raise SystemExit(_implementation.main())
