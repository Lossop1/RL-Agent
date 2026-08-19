"""兼容入口；门控校准的权威实现位于 :mod:`autotuner.research`。"""

from autotuner.research import gate_calibration as _implementation
from autotuner.research.gate_calibration import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_implementation, name)

