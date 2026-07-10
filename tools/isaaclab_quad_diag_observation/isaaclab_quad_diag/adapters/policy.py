from __future__ import annotations

from typing import Any


class PolicyAdapter:
    """Minimal policy adapter interface."""

    def act_mean(self, obs: Any) -> Any:  # pragma: no cover - environment-specific
        raise NotImplementedError


class IdentityPolicyAdapter(PolicyAdapter):
    def act_mean(self, obs: Any) -> Any:
        return obs
