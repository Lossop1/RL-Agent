"""Patch skrl AMP to jointly train the Taili TerrainPerceiver auxiliary heads.

The env appends training-only labels to the observation tail:
  body53 | history(25*54) | privileged197 | geom9 | geom_mask9 | risk2 | risk_mask2

The policy still only acts from body53 + z_terrain32. This patch adds a separate
perceiver optimizer after the normal AMP update and ramps policy gradients into z,
matching docs/taili_strategy_decisions.md.
"""
from __future__ import annotations

import torch

try:  # packaged payload
    from .taili_core import taili_losses
except ImportError:
    if __package__ == "taili_blind_runtime":
        raise
    try:
        from autotuner.taili_core import taili_losses
    except ImportError:
        from taili_core import taili_losses


BODY = 53
HIST = 25 * 54
PRIV = 197
LABEL_START = BODY + HIST + PRIV
HLEN, TICK = 25, 54

_CFG = {"ramp": 50000, "epochs": 1, "batch": 4096, "lr_scale": 0.5}
_WARNED = set()


def _configured_aux_defaults():
    try:
        from .taili_blind_config import get_config_value, load_taili_blind_config
    except Exception:
        return {}
    try:
        cfg = load_taili_blind_config()
        return dict(get_config_value(cfg, "model.terrain_perceiver_aux", {}) or {})
    except Exception:
        return {}


def patch_amp_terrain_aux(ramp_steps=None, epochs=None, batch=None, lr_scale=None):
    from skrl.agents.torch.amp import AMP

    if getattr(AMP, "_taili_tp_aux_patched", False):
        return
    configured = _configured_aux_defaults()
    _CFG.update(
        ramp=int(ramp_steps if ramp_steps is not None else configured.get("ramp_steps", 50000)),
        epochs=int(epochs if epochs is not None else configured.get("epochs", 1)),
        batch=int(batch if batch is not None else configured.get("batch", 4096)),
        lr_scale=float(lr_scale if lr_scale is not None else configured.get("lr_scale", 0.5)),
    )
    orig = AMP._update

    def _update(self, timestep, timesteps):
        orig(self, timestep, timesteps)
        # style-reward observability: the env-side amp_style telemetry is structurally 0 (skrl
        # combines style OUTSIDE the env), so recompute the discriminator style reward on the
        # rollout amp_states here and surface it via TB + stdout. Mirrors skrl 1.4.3 AMP._update.
        try:
            with torch.no_grad():
                amp_states = self.memory.get_tensor_by_name("amp_states")
                flat = amp_states.reshape(-1, amp_states.shape[-1])
                logits, _, _ = self.discriminator.act(
                    {"states": self._amp_state_preprocessor(flat)}, role="discriminator")
                style = -torch.log(torch.clamp(1.0 - torch.sigmoid(logits), min=1e-4))
                mean_style = float(style.mean()) * float(self._discriminator_reward_scale)
            weighted = float(self._style_reward_weight) * mean_style
            if hasattr(self, "track_data"):
                self.track_data("Reward / Style reward (mean)", mean_style)
                self.track_data("Reward / Style reward (weighted)", weighted)
            self._tp_style_tick = getattr(self, "_tp_style_tick", 0) + 1
            if self._tp_style_tick % 25 == 1:
                print(f"[TPAMP] style_reward mean={mean_style:.3f} weighted={weighted:.3f}", flush=True)
        except Exception as e:
            if "style" not in _WARNED:
                print(f"[TP] style telemetry skipped ({type(e).__name__}: {e})", flush=True)
                _WARNED.add("style")
        try:
            policy = self.policy
            if not hasattr(policy, "perceiver"):
                return
            policy.grad_scale.fill_(taili_losses.grad_scale_schedule(int(timestep), _CFG["ramp"]))

            states = self.memory.get_tensor_by_name("states")
            states = states.reshape(-1, states.shape[-1])
            if states.shape[-1] < LABEL_START + 22:
                return

            if getattr(self, "_tp_aux_opt", None) is None:
                cfg = getattr(self, "cfg", {}) or {}
                lr = float(cfg.get("learning_rate", 5e-5)) * _CFG["lr_scale"]
                self._tp_aux_opt = torch.optim.Adam(list(policy.perceiver.parameters()), lr=lr)

            n = states.shape[0]
            last = 0.0
            for _ in range(_CFG["epochs"]):
                idx = torch.randint(0, n, (min(_CFG["batch"], n),), device=states.device)
                raw = states[idx]
                proc = self._state_preprocessor(raw)
                z = policy.perceiver.encode(proc[:, BODY:BODY + HIST].reshape(-1, HLEN, TICK))
                geom_pred, risk_pred = policy.perceiver.aux(z)
                loss = taili_losses.aux_loss(
                    geom_pred,
                    raw[:, LABEL_START:LABEL_START + 9],
                    raw[:, LABEL_START + 9:LABEL_START + 18],
                    risk_pred,
                    raw[:, LABEL_START + 18:LABEL_START + 20],
                    raw[:, LABEL_START + 20:LABEL_START + 22],
                )
                self._tp_aux_opt.zero_grad()
                loss.backward()
                self._tp_aux_opt.step()
                last = float(loss.detach())
            self._tp_aux_last = last
            if hasattr(self, "track_data"):
                self.track_data("Loss / TP aux", last)
        except Exception as e:
            if "tpaux" not in _WARNED:
                print(f"[TP] aux step skipped ({type(e).__name__}: {e}); perceiver trains via policy grad only", flush=True)
                _WARNED.add("tpaux")

    AMP._update = _update
    AMP._taili_tp_aux_patched = True
    print("[TP] AMP._update patched: TerrainPerceiver geom/risk aux + grad_scale ramp (blind, not teacher)", flush=True)


def patch_terrain_aux(scale=1.0, epochs=1, batch=4096, lr=1.0e-3):
    """Compatibility shim for older registration code."""
    del scale, lr
    return patch_amp_terrain_aux(epochs=epochs, batch=batch)
