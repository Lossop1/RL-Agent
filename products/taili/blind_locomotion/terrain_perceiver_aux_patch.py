"""Patch skrl AMP to jointly train the Taili TerrainPerceiver auxiliary heads.

The env appends training-only labels to the observation tail:
  body57 | history(25*54) | privileged197 | geom9 | geom_mask9 | risk8 | risk_mask8

The policy still only acts from body57 + z_terrain32. This patch adds a separate
perceiver optimizer after the normal AMP update and ramps policy gradients into z,
matching docs/taili_strategy_decisions.md.
"""
from __future__ import annotations

import torch

try:  # packaged payload
    from .taili_core import taili_losses, taili_models
except ImportError:
    if __package__ == "taili_blind_runtime":
        raise
    try:
        from products.taili.core import taili_losses, taili_models
    except ImportError:
        from taili_core import taili_losses, taili_models


BODY = 57
HIST = 25 * 54
PRIV = 197
LABEL_START = BODY + HIST + PRIV
HLEN, TICK = 25, 54
GEOM_DIM = taili_models.GEOM_DIM
RISK_DIM = taili_models.RISK_DIM
AUX_LABEL_DIM = 2 * (GEOM_DIM + RISK_DIM)

_CFG = {
    "ramp": 50000,
    "epochs": 1,
    "batch": 4096,
    "lr_scale": 0.5,
    "entropy_final": 0.005,
    "entropy_hold_fraction": 0.75,
    "log_std_floor_initial": -1.50,
    "log_std_floor_final": -2.50,
}
_WARNED = set()


def _exploration_schedule(
    maturity: float,
    *,
    entropy_initial: float,
    entropy_final: float,
    hold_fraction: float,
) -> float:
    """在基础步态形成前保持探索，之后按同一成熟度平滑降低熵。"""
    maturity = min(max(float(maturity), 0.0), 1.0)
    hold = min(max(float(hold_fraction), 0.0), 0.999999)
    alpha = min(max((maturity - hold) / (1.0 - hold), 0.0), 1.0)
    entropy = (1.0 - alpha) * float(entropy_initial) + alpha * float(entropy_final)
    return entropy


def _optimizer_grads_are_finite(optimizer) -> bool:
    """优化器执行前确认所有已生成的梯度均为有限值。"""
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                return False
    return True


def _terrain_amp_task_correction(
    style_reward: torch.Tensor,
    style_scale: torch.Tensor,
    *,
    style_weight: float,
    task_weight: float,
) -> torch.Tensor:
    """把 SKRL 的满额 AMP 回报精确校正为逐样本风格缩放回报。"""
    task_weight = float(task_weight)
    if abs(task_weight) < 1e-12:
        raise ValueError("terrain AMP scaling requires a non-zero task_reward_weight")
    scale = torch.clamp(style_scale.to(dtype=style_reward.dtype, device=style_reward.device), 0.0, 1.0)
    return (float(style_weight) / task_weight) * style_reward * (scale - 1.0)


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
        entropy_final=float(configured.get("exploration_entropy_final", 0.005)),
        entropy_hold_fraction=float(configured.get("exploration_entropy_hold_fraction", 0.75)),
        log_std_floor_initial=float(configured.get("exploration_log_std_floor_initial", -1.50)),
        log_std_floor_final=float(configured.get("exploration_log_std_floor_final", -2.50)),
    )
    orig = AMP._update
    orig_record_transition = AMP.record_transition

    def _record_transition(
        self,
        states,
        actions,
        rewards,
        next_states,
        terminated,
        truncated,
        infos,
        timestep,
        timesteps,
    ):
        scale = infos.get("amp_style_scale") if isinstance(infos, dict) else None
        if scale is None and isinstance(infos, dict):
            # 兼容旧环境；新环境会把地形退让和过渡退让合并为 amp_style_scale。
            scale = infos.get("terrain_pattern_scale")
        if scale is None:
            scale = torch.ones(rewards.shape[0], 1, dtype=rewards.dtype, device=rewards.device)
        else:
            scale = torch.as_tensor(scale, dtype=rewards.dtype, device=rewards.device).reshape(rewards.shape[0], -1)[:, :1]
        scales = getattr(self, "_taili_amp_style_scales", None)
        if scales is None:
            scales = []
            self._taili_amp_style_scales = scales
        scales.append(scale.detach().clone())
        memory_size = int(getattr(getattr(self, "memory", None), "memory_size", getattr(self, "_rollouts", 1)))
        if len(scales) > memory_size:
            del scales[:-memory_size]
        capability = infos.get("exploration_capability") if isinstance(infos, dict) else None
        if capability is not None:
            capability = torch.as_tensor(capability, dtype=torch.float32).mean()
            self._taili_exploration_capability = min(max(float(capability), 0.0), 1.0)
        return orig_record_transition(
            self,
            states,
            actions,
            rewards,
            next_states,
            terminated,
            truncated,
            infos,
            timestep,
            timesteps,
        )

    def _update(self, timestep, timesteps):
        policy = self.policy
        exploration_maturity = 1.0
        if hasattr(policy, "perceiver"):
            # grad_scale 是既有的持久化缓冲。只允许成熟度增加，避免 resume
            # 把感知器梯度和探索调度重新拉回训练早期。
            scheduled = taili_losses.grad_scale_schedule(int(timestep), _CFG["ramp"])
            maturity = max(float(policy.grad_scale), float(scheduled))
            policy.grad_scale.fill_(maturity)
            if not hasattr(self, "_taili_entropy_initial"):
                self._taili_entropy_initial = float(getattr(self, "_entropy_loss_scale", 0.0))
            capability = float(getattr(self, "_taili_exploration_capability", maturity))
            exploration_maturity = min(maturity, capability)
            entropy = _exploration_schedule(
                exploration_maturity,
                entropy_initial=self._taili_entropy_initial,
                entropy_final=_CFG["entropy_final"],
                hold_fraction=_CFG["entropy_hold_fraction"],
            )
            self._entropy_loss_scale = entropy
            self._taili_exploration_tick = getattr(self, "_taili_exploration_tick", 0) + 1
            if hasattr(self, "track_data"):
                self.track_data("Exploration / maturity", maturity)
                self.track_data("Exploration / capability maturity", exploration_maturity)
                self.track_data("Exploration / entropy scale", entropy)
            if self._taili_exploration_tick % 25 == 1:
                print(
                    f"[TPEXPL] perceiver={maturity:.3f} capability={exploration_maturity:.3f} "
                    f"entropy={entropy:.5f}",
                    flush=True,
                )
        # SKRL 在 _update 内才组合 task/style 回报。先按 rollout 顺序重建逐样本
        # terrain_pattern_scale，再给 task 回报加入代数校正：
        # wt*(task + ws/wt*style*(scale-1)) + ws*style = wt*task + ws*style*scale。
        try:
            with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                rewards = self.memory.get_tensor_by_name("rewards")
                amp_states = self.memory.get_tensor_by_name("amp_states")
                amp_logits, _, _ = self.discriminator.act(
                    {"states": self._amp_state_preprocessor(amp_states)}, role="discriminator"
                )
                style_reward = -torch.log(
                    torch.maximum(
                        1.0 - 1.0 / (1.0 + torch.exp(-amp_logits)),
                        torch.tensor(0.0001, device=self.device),
                    )
                )
                style_reward = (style_reward * float(self._discriminator_reward_scale)).view(rewards.shape)
                saved_scales = getattr(self, "_taili_amp_style_scales", [])
                if saved_scales:
                    style_scale = torch.stack(saved_scales[-rewards.shape[0]:], dim=0).reshape(rewards.shape)
                else:
                    style_scale = torch.ones_like(rewards)
                correction = _terrain_amp_task_correction(
                    style_reward,
                    style_scale,
                    style_weight=self._style_reward_weight,
                    task_weight=self._task_reward_weight,
                )
                rewards.add_(correction)
                self._taili_amp_scale_mean = float(style_scale.mean())
                self._taili_amp_style_effective_mean = float((style_reward * style_scale).mean())
        except Exception as e:
            if "terrain_amp" not in _WARNED:
                print(f"[TP] terrain AMP scaling failed ({type(e).__name__}: {e})", flush=True)
                _WARNED.add("terrain_amp")
            raise
        # skrl 会用同一个 Adam 更新 actor、critic 和 discriminator。任一分支产生
        # NaN 梯度都可能在一次 step 中污染全部模型，因此必须在 Adam 前拒绝更新。
        optimizer = getattr(self, "optimizer", None)
        original_step = getattr(optimizer, "step", None)
        if original_step is not None:
            def _finite_step(*args, **kwargs):
                if not _optimizer_grads_are_finite(optimizer):
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        "AMP optimizer rejected a non-finite gradient; last checkpoint remains usable"
                    )
                return original_step(*args, **kwargs)

            optimizer.step = _finite_step
        try:
            orig(self, timestep, timesteps)
        finally:
            getattr(self, "_taili_amp_style_scales", []).clear()
            if original_step is not None:
                optimizer.step = original_step
        # 部署使用确定性均值；该下限只约束训练采样。上下楼尚未形成时不能让全局
        # log_std 因平地先收敛而塌缩，能力成熟后再连续放开到最终下限。
        if hasattr(policy, "actor") and hasattr(policy.actor, "log_std_param"):
            floor_initial = float(_CFG["log_std_floor_initial"])
            floor_final = float(_CFG["log_std_floor_final"])
            log_std_floor = (
                (1.0 - exploration_maturity) * floor_initial
                + exploration_maturity * floor_final
            )
            with torch.no_grad():
                policy.actor.log_std_param.clamp_(min=log_std_floor)
            if hasattr(self, "track_data"):
                self.track_data("Exploration / log std floor", log_std_floor)
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
                self.track_data("Reward / AMP style scale", getattr(self, "_taili_amp_scale_mean", 1.0))
                self.track_data(
                    "Reward / Style reward (effective)",
                    getattr(self, "_taili_amp_style_effective_mean", mean_style),
                )
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
            states = self.memory.get_tensor_by_name("states")
            states = states.reshape(-1, states.shape[-1])
            if states.shape[-1] < LABEL_START + AUX_LABEL_DIM:
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
                    raw[:, LABEL_START:LABEL_START + GEOM_DIM],
                    raw[:, LABEL_START + GEOM_DIM:LABEL_START + 2 * GEOM_DIM],
                    risk_pred,
                    raw[:, LABEL_START + 2 * GEOM_DIM:LABEL_START + 2 * GEOM_DIM + RISK_DIM],
                    raw[:, LABEL_START + 2 * GEOM_DIM + RISK_DIM:LABEL_START + AUX_LABEL_DIM],
                )
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("TerrainPerceiver auxiliary loss is non-finite")
                self._tp_aux_opt.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.perceiver.parameters(), max_norm=1.0)
                if not bool(torch.isfinite(grad_norm)):
                    self._tp_aux_opt.zero_grad(set_to_none=True)
                    raise RuntimeError("TerrainPerceiver auxiliary gradient is non-finite")
                self._tp_aux_opt.step()
                if not all(bool(torch.isfinite(p).all()) for p in policy.perceiver.parameters()):
                    raise RuntimeError("TerrainPerceiver parameters became non-finite after auxiliary update")
                last = float(loss.detach())
            self._tp_aux_last = last
            if hasattr(self, "track_data"):
                self.track_data("Loss / TP aux", last)
        except Exception as e:
            if "tpaux" not in _WARNED:
                print(f"[TP] aux step skipped ({type(e).__name__}: {e}); perceiver trains via policy grad only", flush=True)
                _WARNED.add("tpaux")

    AMP.record_transition = _record_transition
    AMP._update = _update
    AMP._taili_tp_aux_patched = True
    print(
        "[TP] AMP patched: per-sample style scaling + TerrainPerceiver aux/grad ramp (blind actor)",
        flush=True,
    )


def patch_terrain_aux(scale=1.0, epochs=1, batch=4096, lr=1.0e-3):
    """Compatibility shim for older registration code."""
    del scale, lr
    return patch_amp_terrain_aux(epochs=epochs, batch=batch)
