"""skrl Gaussian actor for the Taili blind TerrainPerceiver policy.

Runtime contract from docs/taili_strategy_decisions.md:
  actor input = body[53] + z_terrain[32] = 85
  TerrainPerceiver history = [25, 54]
  z_terrain = [u(history), u(mirror(history))], so mirroring swaps the halves

The critic may receive the full privileged observation from the env, but this
policy only reads the deployable blind slice:
  body53 | history(25*54) = 1403
"""
from __future__ import annotations

import torch
from skrl.models.torch import GaussianMixin, Model

try:  # packaged payload: taili_blind_runtime.taili_core
    from .taili_core import taili_models as M
    from .taili_core.taili_models import apply_grad_scale
except ImportError:  # local source tree: autotuner.taili_core
    if __package__ == "taili_blind_runtime":
        raise
    try:
        from autotuner.taili_core import taili_models as M
        from autotuner.taili_core.taili_models import apply_grad_scale
    except ImportError:
        from taili_core import taili_models as M
        from taili_core.taili_models import apply_grad_scale


BODY_DIM = 53
HIST_LEN = 25
TICK_DIM = 54
HIST_FLAT = HIST_LEN * TICK_DIM
ACTOR_OBS = BODY_DIM + HIST_FLAT
Z_DIM = 32


class TerrainPerceiverPolicy(GaussianMixin, Model):
    """Blind-deployable, structurally L/R-equivariant actor."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        reduction: str = "sum",
        initial_log_std: float = -1.0,
        dropout: float = 0.0,
        actor_hidden: list[int] | tuple[int, ...] | None = None,
        **_unused,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        self.perceiver = M.TerrainPerceiver(dropout=dropout)
        self.actor = M.EquivariantActor(
            hidden=tuple(actor_hidden or (1024, 512)),
            initial_log_std=initial_log_std,
        )
        # policy->perceiver gradient ramp; the aux patch updates this every AMP update.
        self.register_buffer("grad_scale", torch.tensor(0.0))
        self._last_z = None

    def compute(self, inputs, role: str = ""):
        states = inputs.get("states")
        body = states[:, :BODY_DIM]
        history = states[:, BODY_DIM:BODY_DIM + HIST_FLAT].reshape(-1, HIST_LEN, TICK_DIM)
        z = self.perceiver.encode(history)
        self._last_z = z
        z_actor = apply_grad_scale(z, float(self.grad_scale))
        mean = self.actor.mean(body, z_actor)
        return mean, self.actor.log_std(), {"z": z}

    def aux_predict(self):
        """Return geom/risk predictions for the latest z. None before first compute()."""
        return None if self._last_z is None else self.perceiver.aux(self._last_z)


def terrain_perceiver_model(observation_space, action_space, device, return_source: bool = False, **kwargs):
    """skrl Runner-compatible factory."""
    if return_source:
        return "TerrainPerceiverPolicy: body53 + TerrainPerceiver(history25x54)->z32 + equivariant actor."
    return TerrainPerceiverPolicy(observation_space, action_space, device, **kwargs)


def terrain_perceiver_gaussian_model(observation_space, action_space, device, return_source: bool = False, **kwargs):
    """Backward-compatible factory name used by older Runner patches."""
    return terrain_perceiver_model(observation_space, action_space, device, return_source=return_source, **kwargs)
