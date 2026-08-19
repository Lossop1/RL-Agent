"""兼容入口；权威实现位于 :mod:`autotuner.infrastructure.remote`。"""

from autotuner.infrastructure import remote as _implementation
from autotuner.infrastructure.remote import *  # noqa: F401,F403

_AcceptNewHostKeyPolicy = _implementation._AcceptNewHostKeyPolicy
_DEFAULT_KNOWN_HOSTS = _implementation._DEFAULT_KNOWN_HOSTS


def __getattr__(name: str):
    return getattr(_implementation, name)
