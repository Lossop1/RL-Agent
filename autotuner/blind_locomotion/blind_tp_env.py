"""Blind-dog (TerrainPerceiver) env — strict per taili_strategy_decisions.md.

Subclasses the reference AMP env and OVERRIDES ONLY the observation assembly to the new
Runtime-IO recipe (tick54 history + body53) via the unit-verified taili_core. Reuses all the
inherited sim machinery (sensors, terrain, action/delay, push, DR, AMP buffer, rewards, reset).

Packaged as taili_blind_runtime; remote deployment only needs this package on PYTHONPATH.
"""
import math
import os
from dataclasses import asdict, is_dataclass
import types

import numpy as np
import torch

try:                                            # packaged payload
    from .taili_core import (taili_obs, taili_symmetry, taili_amp_reference,
                             taili_reward, taili_terrain_labels)
except ImportError:
    if __package__ == "taili_blind_runtime":
        raise
    try:                                        # local source tree
        from autotuner.taili_core import (taili_obs, taili_symmetry, taili_amp_reference,
                                          taili_reward, taili_terrain_labels)
    except ImportError:
        from taili_core import (taili_obs, taili_symmetry, taili_amp_reference,
                                taili_reward, taili_terrain_labels)

try:
    from .telemetry_emit import TrainingTelemetryEmitter
except Exception:  # pragma: no cover - remote deployment may copy files differently
    try:
        from telemetry_emit import TrainingTelemetryEmitter
    except Exception:
        try:
            if __package__ == "taili_blind_runtime":
                raise
            from autotuner.blind_locomotion.telemetry_emit import TrainingTelemetryEmitter
        except Exception:
            TrainingTelemetryEmitter = None

try:
    from .taili_blind_config import active_direction_progress, phase_command_spec
except Exception:  # pragma: no cover - local fallback for unusual import layouts
    try:
        from autotuner.blind_locomotion.taili_blind_config import active_direction_progress, phase_command_spec
    except Exception:
        active_direction_progress = None
        phase_command_spec = None

from .taili_amp_env import TailiAmpEnv, quaternion_to_tangent_and_normal
from .parametric_ref import flat_reference   # analytic reference joints for the LIVE imitation reward (RANK-1)

HIST_LEN = 25
TICK_DIM = 54


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _spec_float(spec, key: str, default: float) -> float:
    try:
        return float(spec.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _spec_range(spec, key: str, fallback) -> tuple[float, float]:
    value = spec.get(key, fallback)
    try:
        lo, hi = value
        return float(lo), float(hi)
    except (TypeError, ValueError):
        lo, hi = fallback
        return float(lo), float(hi)


def _sample_uniform(n: int, lo: float, hi: float, device) -> torch.Tensor:
    if hi < lo:
        lo, hi = hi, lo
    if abs(hi - lo) < 1e-9:
        return torch.full((n,), float(lo), device=device)
    return torch.rand(n, device=device) * (hi - lo) + lo


class TailiBlindTPEnv(TailiAmpEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # new perceiver history: FIFO of the last 25 tick54 frames
        self._tick_history = torch.zeros((self.num_envs, HIST_LEN, TICK_DIM), device=self.device)
        self._rcfg = taili_reward.reward_cfg_from_env()          # strategy-book reward; TAILI_RW_* env overrides for tuning
        self._reward_cfg_printed = False
        self._prev_in_contact = None                            # for touchdown (swing→stance) detection; anomaly #5 B1 fix
        self._td_impact = _env_flag("TAILI_TD_IMPACT", bool(getattr(self.cfg, "touchdown_impact_only", False)))
        self._telemetry = TrainingTelemetryEmitter() if TrainingTelemetryEmitter is not None else None
        self._hip_joint_ids, _ = self.robot.find_joints(".*_hip_joint")
        self._stagger_pending = True     # one-shot episode-phase randomization after the initial mass reset
        self._cmd_hold = None            # per-env random command hold (steps), drawn in _get_observations
        self._quality_duty_ema = torch.full((self.num_envs, 4), 0.5, device=self.device)
        self._quality_diag_pair_ema = torch.full((self.num_envs,), 0.5, device=self.device)
        self._quality_slip_speed_ema = torch.zeros(self.num_envs, device=self.device)
        self._quality_slip_excess_ema = torch.zeros(self.num_envs, device=self.device)
        self._quality_slip_high_ema = torch.zeros(self.num_envs, device=self.device)
        self._quality_height_low_ema = torch.zeros(self.num_envs, device=self.device)
        self._quality_tilt_high_ema = torch.zeros(self.num_envs, device=self.device)
        self._quality_window_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_commands(self, env_ids, *args, **kwargs):
        # CURRICULUM forcing (anomaly #4: policy ignores command → command-independent creep/stand).
        # TAILI_CUR_FIXED_VX: pin ALL commands to (vx,0,0) — no stand escape, no variety → the policy
        # MUST learn to walk at that speed (the Gaussian rewards EXACT tracking). Phase-0 bootstrap;
        # later phases widen the range to teach command-conditioning. Unset → normal parent sampling.
        super()._resample_commands(env_ids, *args, **kwargs)
        if len(env_ids) == 0:
            return
        snap = bool(kwargs.get("snap", args[0] if args else False))
        phase = int(getattr(self, "_phase", getattr(self.cfg, "init_phase", 0)))
        spec = phase_command_spec(self.cfg, phase) if phase_command_spec is not None else {}
        mode = str(spec.get("command_mode") or getattr(self.cfg, "training_command_mode", "normal") or "normal")
        self._last_command_mode = mode
        self._last_command_spec = spec
        try:
            target = None
            fv_override = os.environ.get("TAILI_CUR_FIXED_VX")
            lo_override, hi_override = os.environ.get("TAILI_CUR_FWD_LO"), os.environ.get("TAILI_CUR_FWD_HI")
            if fv_override is not None:
                target = torch.zeros((len(env_ids), 3), device=self.device)
                target[:, 0] = float(fv_override)
                self._last_command_mode = "env_fixed_forward"
            elif lo_override is not None and hi_override is not None:
                lo_v, hi_v = float(lo_override), float(hi_override)
                target = torch.zeros((len(env_ids), 3), device=self.device)
                target[:, 0] = torch.rand(len(env_ids), device=self.device) * (hi_v - lo_v) + lo_v
                self._last_command_mode = "env_forward_range"
            elif mode == "fixed_forward":
                target = torch.zeros((len(env_ids), 3), device=self.device)
                target[:, 0] = _spec_float(spec, "fixed_vx", float(getattr(self.cfg, "training_fixed_vx", 0.5)))
            elif mode == "forward_range":
                lo_v, hi_v = _spec_range(spec, "forward_range", getattr(self.cfg, "training_forward_range", (0.3, 0.7)))
                target = torch.zeros((len(env_ids), 3), device=self.device)
                target[:, 0] = torch.rand(len(env_ids), device=self.device) * (hi_v - lo_v) + lo_v
            elif mode == "stand_only":
                target = torch.zeros((len(env_ids), 3), device=self.device)
            elif mode == "single_axis":
                n, dev = len(env_ids), self.device
                target = torch.zeros((n, 3), device=dev)
                stand_prob = _spec_float(spec, "stand_prob", float(getattr(self.cfg, "stand_prob", 0.0)))
                active = torch.rand(n, device=dev) >= stand_prob
                weights = torch.tensor([
                    max(0.0, _spec_float(spec, "prob_fwd", float(getattr(self.cfg, "cmd_prob_fwd", 0.25)))),
                    max(0.0, _spec_float(spec, "prob_back", float(getattr(self.cfg, "cmd_prob_back", 0.25)))),
                    max(0.0, _spec_float(spec, "prob_lat", float(getattr(self.cfg, "cmd_prob_lat", 0.25)))),
                    max(0.0, _spec_float(spec, "prob_yaw", float(getattr(self.cfg, "cmd_prob_yaw", 0.25)))),
                ], device=dev)
                if float(weights.sum()) <= 1e-9:
                    weights[:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
                u = torch.rand(n, device=dev) * weights.sum()
                c0 = weights[0]
                c1 = c0 + weights[1]
                c2 = c1 + weights[2]
                fwd = active & (u < c0)
                back = active & (u >= c0) & (u < c1)
                lat = active & (u >= c1) & (u < c2)
                yaw = active & (u >= c2)
                f_lo, f_hi = _spec_range(spec, "fwd_range", getattr(self.cfg, "cmd_fwd_range", (0.3, 0.7)))
                b_lo, b_hi = _spec_range(spec, "back_range", getattr(self.cfg, "cmd_back_range", (0.2, 0.45)))
                l_lo, l_hi = _spec_range(spec, "lat_range", getattr(self.cfg, "cmd_lat_range", (0.15, 0.35)))
                y_lo, y_hi = _spec_range(spec, "yaw_range", getattr(self.cfg, "cmd_yaw_range", (0.25, 0.8)))
                f_hi = min(f_hi, float(getattr(self, "_vel_max_fwd", f_hi)))
                b_hi = min(b_hi, float(getattr(self, "_vel_max_back", b_hi)))
                l_hi = min(l_hi, float(getattr(self, "_vel_max_lat", l_hi)))
                y_hi = min(y_hi, float(getattr(self, "_vel_max_yaw", y_hi)))
                target[:, 0] = torch.where(fwd, _sample_uniform(n, f_lo, f_hi, dev), target[:, 0])
                target[:, 0] = torch.where(back, -_sample_uniform(n, b_lo, b_hi, dev), target[:, 0])
                lat_sign = torch.where(torch.rand(n, device=dev) < 0.5, torch.ones(n, device=dev), -torch.ones(n, device=dev))
                yaw_sign = torch.where(torch.rand(n, device=dev) < 0.5, torch.ones(n, device=dev), -torch.ones(n, device=dev))
                target[:, 1] = torch.where(lat, lat_sign * _sample_uniform(n, l_lo, l_hi, dev), target[:, 1])
                target[:, 2] = torch.where(yaw, yaw_sign * _sample_uniform(n, y_lo, y_hi, dev), target[:, 2])
            elif mode == "mixed":
                n, dev = len(env_ids), self.device
                target = torch.zeros((n, 3), device=dev)
                stand_prob = _spec_float(spec, "stand_prob", float(getattr(self.cfg, "stand_prob", 0.0)))
                near_zero_prob = _spec_float(spec, "near_zero_prob", 0.0)
                axis_prob = _spec_float(spec, "mixed_axis_prob", 1.0)
                active = torch.rand(n, device=dev) >= stand_prob
                near_zero = active & (torch.rand(n, device=dev) < near_zero_prob)
                moving = active & ~near_zero
                f_lo, f_hi = _spec_range(spec, "fwd_range", getattr(self.cfg, "cmd_fwd_range", (0.3, 0.7)))
                b_lo, b_hi = _spec_range(spec, "back_range", getattr(self.cfg, "cmd_back_range", (0.2, 0.45)))
                l_lo, l_hi = _spec_range(spec, "lat_range", getattr(self.cfg, "cmd_lat_range", (0.15, 0.35)))
                y_lo, y_hi = _spec_range(spec, "yaw_range", getattr(self.cfg, "cmd_yaw_range", (0.25, 0.8)))
                f_hi = min(f_hi, float(getattr(self, "_vel_max_fwd", f_hi)))
                b_hi = min(b_hi, float(getattr(self, "_vel_max_back", b_hi)))
                l_hi = min(l_hi, float(getattr(self, "_vel_max_lat", l_hi)))
                y_hi = min(y_hi, float(getattr(self, "_vel_max_yaw", y_hi)))
                p_fwd = max(0.0, _spec_float(spec, "prob_fwd", 0.5))
                p_back = max(0.0, _spec_float(spec, "prob_back", 0.5))
                p_sum = max(1e-9, p_fwd + p_back)
                x_is_fwd = torch.rand(n, device=dev) < (p_fwd / p_sum)
                # PURE-TURN PRACTICE (0706 D3s): the mixed sampler unconditionally set an x (fwd/back)
                # component on every moving command, so a PURE in-place turn (vx=vy=0, wz!=0) — exactly
                # what physeval A2 tests — was NEVER sampled in the mixed phases (the training bulk).
                # The policy therefore never learned to turn without forward motion → the "脚前勾"
                # (swing legs hook forward under yaw/lateral) and the A2 yaw p90 tail both trace here.
                # x_axis_prob<1 lets the x-component drop out; x_axis_prob=1.0 (default) = old behavior.
                x_axis_prob = _spec_float(spec, "x_axis_prob", 1.0)
                x_on = torch.rand(n, device=dev) < x_axis_prob
                lat_on = torch.rand(n, device=dev) < axis_prob
                yaw_on = torch.rand(n, device=dev) < axis_prob
                # guarantee an x-off command is a REAL turn (pure yaw/strafe), not an accidental stand
                yaw_on = yaw_on | (~x_on & ~lat_on)
                lat_sign = torch.where(torch.rand(n, device=dev) < 0.5, torch.ones(n, device=dev), -torch.ones(n, device=dev))
                yaw_sign = torch.where(torch.rand(n, device=dev) < 0.5, torch.ones(n, device=dev), -torch.ones(n, device=dev))
                target[:, 0] = torch.where(x_is_fwd, _sample_uniform(n, f_lo, f_hi, dev), -_sample_uniform(n, b_lo, b_hi, dev))
                target[:, 0] = torch.where(x_on, target[:, 0], torch.zeros(n, device=dev))
                target[:, 1] = torch.where(lat_on, lat_sign * _sample_uniform(n, l_lo, l_hi, dev), torch.zeros(n, device=dev))
                target[:, 2] = torch.where(yaw_on, yaw_sign * _sample_uniform(n, y_lo, y_hi, dev), torch.zeros(n, device=dev))
                target = torch.where(moving[:, None], target, torch.zeros_like(target))
                nz_scale = _spec_float(spec, "near_zero_scale", 0.05)
                near_noise = (torch.rand((n, 3), device=dev) * 2.0 - 1.0) * nz_scale
                target = torch.where(near_zero[:, None], near_noise, target)
            if target is not None:
                self._cmd_target[env_ids] = target
                if snap:
                    self.commands[env_ids] = target
                return
        except (ValueError, TypeError):
            pass
        # Malformed explicit override or unknown recipe mode: keep the parent sampler result.
        return

    def _draw_cmd_hold(self, n: int) -> torch.Tensor:
        """Per-env random command hold duration in control steps ([resample_s_min, resample_s_max])."""
        step_dt = float(self.cfg.dt) * float(self.cfg.decimation)
        lo_s = float(getattr(self.cfg, "cmd_resample_s_min", getattr(self.cfg, "cmd_resample_s", 5.0)))
        hi_s = float(getattr(self.cfg, "cmd_resample_s_max", lo_s))
        lo = max(1, int(min(lo_s, hi_s) / step_dt))
        hi = max(lo + 1, int(max(lo_s, hi_s) / step_dt) + 1)
        return torch.randint(lo, hi, (n,), device=self.device, dtype=torch.long)

    def _get_observations(self) -> dict:
        if not getattr(self, "use_external_commands", False):
            # DESYNCHRONIZE the fleet once, after the runner's initial mass reset: with a ~0 fall
            # rate all 4096 envs otherwise stay phase-locked forever (same timeout step, same
            # command-switch instants) — rollout batches alternate between "everything resets at
            # once" and "nothing happens", and the 500-step diagnostics alias onto fixed episode
            # phases, biasing the phase gates.
            if getattr(self, "_stagger_pending", True):
                self.episode_length_buf[:] = torch.randint(
                    0, max(1, int(self.max_episode_length) - 1),
                    (self.num_envs,), device=self.device, dtype=self.episode_length_buf.dtype)
                self._cmd_hold = self._draw_cmd_hold(self.num_envs)
                self._stagger_pending = False
            # per-env random command hold (replaces the synchronized modulo-resample)
            self._cmd_hold -= 1
            due = self._cmd_hold <= 0
            if due.any():
                ids = due.nonzero(as_tuple=False).flatten()
                self._resample_commands(ids)
                self._cmd_hold[ids] = self._draw_cmd_hold(len(ids))
        N, dev = self.num_envs, self.device
        lp = self._leg_phases()
        gait_obs = torch.cat([torch.sin(2 * math.pi * lp), torch.cos(2 * math.pi * lp)], dim=-1)   # (N,8)

        # noised raw quantities (deployment-faithful)
        jpos_n = self.robot.data.joint_pos + torch.randn(N, 12, device=dev) * self.cfg.obs_noise_jpos
        jvel_n = self.robot.data.joint_vel + torch.randn(N, 12, device=dev) * self.cfg.obs_noise_jvel
        angv_n = self.robot.data.root_ang_vel_b + torch.randn(N, 3, device=dev) * self.cfg.obs_noise_angvel
        grav_n = self.robot.data.projected_gravity_b + torch.randn(N, 3, device=dev) * self.cfg.obs_noise_gravity
        angv_n = angv_n + self._imu_bias[:, 0:3]
        grav_n = grav_n + self._imu_bias[:, 3:6]

        # NEW tick54 (q_rel/dq/q_des_rel/q_error/gyro/grav) via taili_obs; q_default = action_offset
        tick = taili_obs.assemble_tick54(jpos_n, jvel_n, self.last_actions, angv_n, grav_n,
                                         action_scale=self.action_scale, q_default=self.action_offset)
        self._tick_history = torch.cat([tick.unsqueeze(1), self._tick_history[:, :-1]], dim=1)   # FIFO

        # foot contact cache (rewards reuse it)
        forces_now = self._contact_sensor.data.net_forces_w[:, self._feet_contact_ids, :].norm(dim=-1)
        contact_threshold = float(getattr(self.cfg, "contact_force_threshold", 10.0))
        self._in_contact = (forces_now > contact_threshold).float()

        # body53 = angv | grav | cmd | jpos_rel | jvel | last_act | gait
        body = torch.cat([angv_n, grav_n, self.commands, jpos_n - self.action_offset,
                          jvel_n, self.last_actions, gait_obs], dim=-1)                            # (N,53)
        blind_obs = torch.cat([body, self._tick_history.reshape(N, -1)], dim=-1)                   # (N,1403)

        # privileged (critic/AMP only) — same as parent
        hits = self._height_scanner.data.ray_hits_w
        hscan = (self.robot.data.root_pos_w[:, 2:3] - hits[:, :, 2] - self.cfg.stand_height).clamp(-1.0, 1.0)
        hscan = torch.nan_to_num(hscan, nan=0.0, posinf=0.0, neginf=0.0)
        self._terrain_ctx = self._compute_terrain_ctx()
        priv = torch.cat([self.robot.data.root_lin_vel_b, hscan, self._terrain_ctx, self._in_contact], dim=-1)
        labels = self._compute_aux_labels()                                                       # (N,22) geom9+mask9+risk2+mask2
        obs = torch.cat([blind_obs, priv, labels], dim=-1)                                         # (N,1622)

        # AMP buffer (inherited _compute_amp_obs for now; frame51 swap is a later increment)
        amp = torch.nan_to_num(self._compute_amp_obs(), nan=0.0, posinf=0.0, neginf=0.0)
        if getattr(self, "_amp_frame_stride", 1) > 1:
            self._push_strided_amp(amp)                 # AMP STRIDE WINDOW: strided ~1-gait-period export
        else:
            for i in reversed(range(self.cfg.num_amp_observations - 1)):
                self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
            self.amp_observation_buffer[:, 0] = amp
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}
        # MULTI-CRITIC: surface the per-group reward vector stashed by _get_rewards (this step, run
        # just before this call) so the agent's record_transition can carry [N,K] into the rollout.
        # Gated: FLAG UNSET -> extras is byte-identical to stock.
        if os.environ.get("TAILI_MULTI_CRITIC") == "1" and getattr(self, "_reward_groups", None) is not None:
            self.extras["reward_groups"] = self._reward_groups
        self.last_actions = self.actions.clone()        # FIFO bookkeeping (parent does this in _get_observations)
        self._log_step = getattr(self, "_log_step", 0) + 1
        if self._log_step % self.cfg.log_every == 0 and not getattr(self, "use_external_commands", False):
            self._log_training_diag()
        return {"policy": obs}

    # ── geom/risk aux labels from privileged sim signals (taili_terrain_labels) ──
    def _compute_aux_labels(self) -> torch.Tensor:
        """Returns (N,22) = geom_label9 | geom_mask9 | risk_label2 | risk_mask2. Training-only;
        appended to the privileged obs so it rides the rollout for the aux trainer hook.
        v1: slope/roughness (height-scanner plane fit) + impact/support; foot_h/edge masked off."""
        rd = self.robot.data
        N, dev = self.num_envs, self.device
        hits = self._height_scanner.data.ray_hits_w                                  # (N,R,3) world
        rel_xy = hits[:, :, :2] - rd.root_pos_w[:, :2].unsqueeze(1)
        z = torch.nan_to_num(hits[:, :, 2], nan=0.0)
        a, b, _ = taili_terrain_labels.fit_local_plane(rel_xy, z)                     # world slope a,b
        geom3 = torch.stack([taili_terrain_labels.slope_norm(a), taili_terrain_labels.slope_norm(b),
                             taili_terrain_labels.log_roughness_norm(rel_xy, z)], dim=-1)  # (N,3)
        # foot_h4 (0706, STAIRS SENSING): per-foot terrain height relative to the base ground level, the
        # channel the perceiver NEEDED for stairs and that v1 left as a masked-off zero stub. Without it the
        # blind latent encoded only slope/roughness → the robot was literally blind to step height and could
        # never learn how high to lift on a riser (D[slope] passed, stairs failed). Supervise it from the
        # (training-privileged) height scanner so the deploy-time latent can estimate step structure.
        foot_pos = rd.body_pos_w[:, self.foot_indexes, :]                            # (N,4,3)
        _fdxy = foot_pos[:, :, None, :2] - hits[:, None, :, :2]                       # (N,4,R,2)
        _fnear = (_fdxy * _fdxy).sum(-1).argmin(dim=2)                                # (N,4) nearest ray per foot
        foot_ground = torch.gather(z, 1, _fnear)                                     # (N,4) terrain z under each foot
        base_ground = z.median(dim=1).values                                        # (N,) local ground level
        foot_h4 = ((foot_ground - base_ground.unsqueeze(1)) / 0.40).clamp(-1.0, 1.0)  # (N,4) per-foot elevation, norm
        geom = torch.cat([geom3, foot_h4, torch.zeros(N, 2, device=dev)], dim=-1)     # (N,9): slope3 + foot_h4 + edge2(0)
        geom_mask = torch.tensor([1., 1., 1., 1., 1., 1., 1., 0., 0.], device=dev).unsqueeze(0).expand(N, 9)

        forces = self._contact_sensor.data.net_forces_w[:, self._feet_contact_ids, :].norm(dim=-1)  # (N,4)
        f_max = forces.max(dim=1).values
        impact_n = taili_terrain_labels.impact_score_norm(f_max, torch.tensor(self.cfg.robot_mass_kg, device=dev)
                                                          if hasattr(self.cfg, "robot_mass_kg") else torch.tensor(39.0, device=dev))
        cc = self._in_contact.sum(dim=1)
        tilt = torch.arccos(torch.clamp(-rd.projected_gravity_b[:, 2], -1.0, 1.0))
        support = torch.clamp((3.0 - cc) / 3.0, 0.0, 1.0)
        support = torch.maximum(support, torch.clamp((tilt - math.radians(10)) / math.radians(20), 0.0, 1.0))
        risk = torch.stack([impact_n, support], dim=-1)                              # (N,2)
        risk_mask = torch.ones(N, 2, device=dev)
        return torch.cat([geom, geom_mask, risk, risk_mask], dim=-1)                 # (N,22)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self._tick_history[env_ids] = 0.0
        # fresh random command hold for the reset envs (keeps the fleet desynchronized)
        if getattr(self, "_cmd_hold", None) is not None and len(env_ids) > 0:
            self._cmd_hold[env_ids] = self._draw_cmd_hold(len(env_ids))
        # RSI places feet in/near contact; mark them "previously in contact" so the first post-reset
        # step is NOT mis-counted as a fresh touchdown (anomaly #5 B1 fix; only used when _td_impact).
        if self._prev_in_contact is not None:
            self._prev_in_contact[env_ids] = 1.0
        if hasattr(self, "_quality_window_ready"):
            self._quality_duty_ema[env_ids] = 0.5
            self._quality_diag_pair_ema[env_ids] = 0.5
            self._quality_slip_speed_ema[env_ids] = 0.0
            self._quality_slip_excess_ema[env_ids] = 0.0
            self._quality_slip_high_ema[env_ids] = 0.0
            self._quality_height_low_ema[env_ids] = 0.0
            self._quality_tilt_high_ema[env_ids] = 0.0
            self._quality_window_ready[env_ids] = False

    def _update_quality_windows(self, in_contact, settled_contact, foot_vel_xy, base_h, tilt_rel, moving):
        """Update segment-like gait/reality metrics used by reward, phase gates and telemetry.

        Single-frame proxies were too permissive: all-four-feet crawl could look diagonal,
        high-duty crawl could look balanced, and one slipping stance foot was diluted by a
        four-foot mean. These EMA metrics are closer to what diagnostics judge over a run.
        """
        f = in_contact.dtype
        if not hasattr(self, "_quality_window_ready"):
            self._quality_duty_ema = torch.full_like(in_contact, 0.5)
            self._quality_diag_pair_ema = torch.full_like(base_h, 0.5)
            self._quality_slip_speed_ema = torch.zeros_like(base_h)
            self._quality_slip_excess_ema = torch.zeros_like(base_h)
            self._quality_slip_high_ema = torch.zeros_like(base_h)
            self._quality_height_low_ema = torch.zeros_like(base_h)
            self._quality_tilt_high_ema = torch.zeros_like(base_h)
            self._quality_window_ready = torch.zeros(base_h.shape[0], dtype=torch.bool, device=base_h.device)

        beta = min(max(float(getattr(self._rcfg, "quality_window_ema_beta", 0.98)), 0.0), 0.999)
        stance = (in_contact > 0.5).to(f)
        settled = (settled_contact > 0.5).to(f)
        moving_mask = moving > 0.5
        moving_col = moving_mask[:, None]

        if stance.shape[-1] >= 4:
            fl, fr, rl, rr = stance[:, 0], stance[:, 1], stance[:, 2], stance[:, 3]
            diag_pair = torch.clamp(
                fl * rr * (1.0 - fr) * (1.0 - rl)
                + fr * rl * (1.0 - fl) * (1.0 - rr),
                0.0,
                1.0,
            )
        else:
            diag_pair = torch.zeros_like(base_h)

        duty_target = min(max(float(getattr(self._rcfg, "duty_target", getattr(self.cfg, "gait_duty", 0.5))), 0.05), 0.95)
        duty_sample = torch.where(moving_col, stance, torch.full_like(stance, duty_target))
        # DIAGONAL-TROT METRIC — measure it ONLY where a diagonal trot applies: a forward/back/lateral
        # command that DOMINATES yaw. During yaw-dominant turning the robot legitimately isn't in a
        # forward diagonal trot, so diag_pair reads low there; averaging that in dragged the EMA to a
        # ~0.42 ceiling (and DOWN further as yaw tracking improved — an anti-correlation that made the
        # phase-0 diag gate unreachable and stalled every run at phase 0). Feeding the EMA its own
        # current value on non-applicable frames FREEZES it, so diag_gate reflects straight-line trot
        # quality. This also stops the diagonal_contact reward (same EMA) from fighting turning gaits.
        _cmd = self.commands
        _lin_mag = torch.linalg.norm(_cmd[:, :2], dim=-1)
        _trot_applicable = moving_mask & (_lin_mag > 0.10) & (_lin_mag >= 0.6 * _cmd[:, 2].abs())
        diag_sample = torch.where(_trot_applicable, diag_pair, self._quality_diag_pair_ema)

        contact_denom = settled.sum(dim=-1).clamp(min=1.0)
        slip_speed_mean = (foot_vel_xy * settled).sum(dim=-1) / contact_denom
        slip_excess = torch.clamp(
            (foot_vel_xy - 0.05) / 0.05,
            min=0.0,
            max=float(getattr(self._rcfg, "slip_excess_cap", 1.5)),
        )
        slip_excess_mean = (slip_excess ** 2 * settled).sum(dim=-1) / contact_denom
        slip_high_threshold = float(getattr(self._rcfg, "slip_high_threshold", 0.20))
        slip_high_fraction = ((foot_vel_xy > slip_high_threshold).to(f) * settled).sum(dim=-1) / contact_denom
        slip_speed_sample = torch.where(moving_mask, slip_speed_mean, torch.zeros_like(slip_speed_mean))
        slip_excess_sample = torch.where(moving_mask, slip_excess_mean, torch.zeros_like(slip_excess_mean))
        slip_high_sample = torch.where(moving_mask, slip_high_fraction, torch.zeros_like(slip_high_fraction))

        h_ok = float(getattr(self._rcfg, "h_ok", 0.47))
        h_close = float(getattr(self._rcfg, "h_gate_close", 0.42))
        height_low = torch.clamp((h_ok - base_h) / max(h_ok - h_close, 1e-6), 0.0, 1.0)
        tilt_ok = float(getattr(self._rcfg, "tilt_ok_rad", math.radians(15)))
        tilt_close = float(getattr(self._rcfg, "tilt_gate_close_rad", math.radians(40)))
        tilt_high = torch.clamp((tilt_rel - tilt_ok) / max(tilt_close - tilt_ok, 1e-6), 0.0, 1.0)

        cold = ~self._quality_window_ready
        if bool(cold.any()):
            self._quality_duty_ema[cold] = duty_sample[cold]
            self._quality_diag_pair_ema[cold] = diag_sample[cold]
            self._quality_slip_speed_ema[cold] = slip_speed_sample[cold]
            self._quality_slip_excess_ema[cold] = slip_excess_sample[cold]
            self._quality_slip_high_ema[cold] = slip_high_sample[cold]
            self._quality_height_low_ema[cold] = height_low[cold]
            self._quality_tilt_high_ema[cold] = tilt_high[cold]
            self._quality_window_ready[cold] = True

        self._quality_duty_ema.mul_(beta).add_(duty_sample * (1.0 - beta))
        self._quality_diag_pair_ema.mul_(beta).add_(diag_sample * (1.0 - beta))
        self._quality_slip_speed_ema.mul_(beta).add_(slip_speed_sample * (1.0 - beta))
        self._quality_slip_excess_ema.mul_(beta).add_(slip_excess_sample * (1.0 - beta))
        self._quality_slip_high_ema.mul_(beta).add_(slip_high_sample * (1.0 - beta))
        self._quality_height_low_ema.mul_(beta).add_(height_low * (1.0 - beta))
        self._quality_tilt_high_ema.mul_(beta).add_(tilt_high * (1.0 - beta))

        duty_w = self._quality_duty_ema
        front = 0.5 * (duty_w[:, 0] + duty_w[:, 1])
        rear = 0.5 * (duty_w[:, 2] + duty_w[:, 3])
        # B4 symmetry uses the EVAL's PER-PAIR L/R axis |FL-FR|+|RL-RR| (dof [FL,FR,RL,RR]). The old
        # whole-body 0.5(FL+RL) vs 0.5(FR+RR) CANCELS opposite-sign front/rear skew (the limping
        # backward-trot: front L>R while rear R>L reads ~0), so the reward saw symmetry while eval saw
        # 2|x| — no w_duty_balance could close B4[back04]. Front-rear balance kept as a SEPARATE term. (D3m)
        lr_pair_skew = (duty_w[:, 0] - duty_w[:, 1]).abs() + (duty_w[:, 2] - duty_w[:, 3]).abs()
        duty_symmetry_score = 1.0 - torch.clamp(lr_pair_skew + (front - rear).abs(), 0.0, 1.0)
        duty_target_error = (duty_w - duty_target).abs().mean(dim=-1)
        duty_target_score = 1.0 - torch.clamp(duty_target_error / max(duty_target, 1.0 - duty_target, 1e-6), 0.0, 1.0)
        duty_quality = torch.clamp(duty_symmetry_score * duty_target_score, 0.0, 1.0)

        return {
            "duty_by_leg": duty_w,
            "duty_mean": duty_w.mean(dim=-1),
            "duty_spread": duty_w.max(dim=-1).values - duty_w.min(dim=-1).values,
            "duty_symmetry_score": duty_symmetry_score,
            "duty_target_score": duty_target_score,
            "duty_target_error": duty_target_error,
            "duty_quality": duty_quality,
            "diag_pair_score": self._quality_diag_pair_ema,
            "slip_speed": self._quality_slip_speed_ema,
            "slip_excess": self._quality_slip_excess_ema,
            "slip_high_fraction": self._quality_slip_high_ema,
            "height_low_risk": self._quality_height_low_ema,
            "tilt_high_risk": self._quality_tilt_high_ema,
            "slip_speed_inst": slip_speed_mean,
            "slip_high_fraction_inst": slip_high_fraction,
            "diag_pair_inst": diag_pair,
        }

    # ── AMP: frame51 = motion43 + command3 + mode_onehot5 (D), terrain-agnostic ──
    def _compute_amp_obs(self):
        """Policy-side AMP obs: ACHIEVED motion43 (robot) + command + mode. Item-for-item
        aligned with collect_reference_motions (the analytic q_default reference)."""
        jp = self.robot.data.joint_pos
        jv = self.robot.data.joint_vel
        bh = self.robot.data.root_pos_w[:, 2:3] - self._terrain_height_under_base().unsqueeze(1)
        tn = quaternion_to_tangent_and_normal(self.robot.data.root_quat_w)
        rel_b = self._feet_rel_base(self.robot.data.body_pos_w[:, self.foot_indexes],
                                    self.robot.data.root_pos_w, self.robot.data.root_quat_w, self.n_feet)
        mode = taili_amp_reference.mode_onehot(self.commands)
        return torch.cat([jp, jv, bh, tn, rel_b, self.commands, mode], dim=-1)   # 43 + 3 + 5 = 51

    def collect_reference_motions(self, num_samples, current_times=None):
        """Reference-side AMP frames via the q_default-aligned analytic generator (taili_core).

        Note: IsaacLab returns joint_names in the canonical [FL,FR,RL,RR] x [hip,thigh,calf] order
        (verified), so motion_dof_indexes is identity and the analytic reference (canonical) is
        already aligned with _compute_amp_obs (raw joint_pos, also canonical) — no remap needed.
        """
        K = self.cfg.num_amp_observations
        env_sel = torch.randint(0, self.num_envs, (num_samples,), device=self.device)
        cmd_s = self.commands[env_sel]
        if current_times is None:
            current_times = np.random.uniform(0.0, 2.0, num_samples).astype(np.float32)
        _stride = getattr(self, "_amp_frame_stride", 1)                       # AMP STRIDE WINDOW: match policy-side spacing
        times = (np.expand_dims(current_times, -1) - (1.0 / 50.0) * _stride * np.arange(K)).flatten()
        times_t = torch.as_tensor(times, dtype=torch.float32, device=self.device)
        cmd_rep = cmd_s.repeat_interleave(K, dim=0)
        frame = taili_amp_reference.frame51(
            cmd_rep,
            times_t,
            gait_period=self.cfg.gait_period,
            gait_period_slope=self.cfg.gait_period_slope,
            gait_period_min=self.cfg.gait_period_min,
            clearance_base=self.cfg.base_clearance,
            stance_dx=self.cfg.stance_dx,
        )                                                                         # (num_samples*K, 51)
        return frame.view(-1, self.amp_observation_size)

    # ── reward: strategy-book taili_reward (replaces the reference hand-tuned reward) ──
    def _terrain_height_under_base(self):
        """Robust local terrain height from the height scanner (median of ray hits, per env).
        Terrain-ready replacement for env_origins z: identical on flat, tracks the actual
        support surface on slopes/stairs/rough. Non-finite rays fall back to the env origin."""
        origins_z = self._terrain.env_origins[:, 2]
        try:
            hits_z = self._height_scanner.data.ray_hits_w[:, :, 2]
            hits_z = torch.where(torch.isfinite(hits_z), hits_z, origins_z.unsqueeze(1))
            return hits_z.median(dim=1).values
        except Exception:
            return origins_z

    def _get_rewards(self) -> torch.Tensor:
        rd = self.robot.data
        N, dev = self.num_envs, self.device
        cfg = self._rcfg

        terrain_h = self._terrain_height_under_base()                                 # (N,)
        base_h = rd.root_pos_w[:, 2] - terrain_h                                       # above LOCAL terrain
        grav = rd.projected_gravity_b
        tilt_rel = torch.arccos(torch.clamp(-grav[:, 2], -1.0, 1.0))                   # world tilt (slope-rel: TODO)
        roll = torch.arcsin(torch.clamp(grav[:, 1], -1.0, 1.0))
        pitch = torch.arcsin(torch.clamp(-grav[:, 0], -1.0, 1.0))
        in_contact = self._in_contact                                                 # (N,4) cached in _get_observations
        cc = in_contact.sum(dim=1)
        spd_xy = torch.norm(self.commands[:, :2], dim=1)
        # YAW REWARD-GATE FIX (0704): the yaw moving-threshold (0.1) must match taili_reward's
        # yaw_cmd_gate (0.05). A gentle yaw command in (0.05, 0.10] was YAW-commanded but not
        # "moving", so tracking_yaw (× moving_gate) was ZEROED and the env paid a STAND reward for
        # it — training the robot to stand still when told to turn gently. The velocity curriculum
        # drives the yaw ceiling down to the floor (0.10) whenever yaw_prog<=vel_cur_down, so most
        # yaw commands landed at ~0.10 → misclassified as standing → tracking_yaw≈0 (measured) →
        # no yaw gradient → yaw frozen ~0.55 (death spiral). lat works because its floor is 0.15>0.1.
        moving = ((spd_xy > 0.1) | (self.commands[:, 2].abs() > 0.05)).float()
        stand_gate = 1.0 - moving
        prev_contact = self._prev_in_contact
        if prev_contact is None or prev_contact.shape != in_contact.shape:
            prev_contact = torch.ones_like(in_contact)                                # first call: no false touchdown
        self._prev_in_contact = in_contact.detach().clone()
        td_mask = None
        if self._td_impact:                                                           # opt-in: B1 shapes the TOUCHDOWN event (anomaly #5)
            td_mask = ((in_contact > 0.5) & (prev_contact < 0.5)).float()             # swing→stance transition = landing
        # SETTLED stance = in contact this step AND the previous one. The touchdown frame carries
        # ~swing-speed horizontal velocity while the force first crosses the threshold; counting it
        # as "slip" double-books the B1 landing event into B2 and inflates the slip mean ~3x
        # (measured 0.27 linear mean vs ~0.09 typical settled foot) — a spec-clean trot could
        # never pass the 0.20 slip gate. B2 owns settled stance only; B1 owns the landing frame.
        settled_contact = in_contact * (prev_contact > 0.5).float()
        foot_pos = rd.body_pos_w[:, self.foot_indexes, :]                             # (N,4,3)
        foot_vel = rd.body_lin_vel_w[:, self.foot_indexes, :]                         # (N,4,3)
        # CONTACT-POINT planar velocity for ALL slip consumers (quality windows → reward → slip_gate).
        # The foot-LINK CoM velocity includes sphere ROLLING (ω·r, r=0.014m ≈ 0.05-0.15 m/s at walking
        # cadence) that is NOT sliding; training on it forced slip_free_speed=0.15 while the spec (and
        # the corrected physeval) demand contact-point slip ≤ 0.05 — the policy converged to ~0.17
        # because the reward literally never asked for less (D3n). v_cp = v + ω × r, r=(0,0,-0.014).
        # touchdown_vz (B1) stays foot-LINK z, matching the physeval's landing metric.
        _foot_ang = rd.body_ang_vel_w[:, self.foot_indexes, :]                        # (N,4,3)
        _r_cp = torch.tensor([0.0, 0.0, -0.014], device=dev).view(1, 1, 3)
        _foot_cp_vel = foot_vel + torch.cross(_foot_ang, _r_cp.expand_as(_foot_ang), dim=-1)
        foot_vel_xy = _foot_cp_vel[:, :, :2].norm(dim=-1)
        # #10 TERRAIN-AWARE FOOT CLEARANCE (0705 physeval B3 fix). The old code measured clearance vs
        # the base-MEDIAN terrain height and hardwired local_obstacle_h=0, so the reward's clearance
        # target stayed a flat 0.08 m EVEN on terrain — forcing the policy to over-lift everywhere
        # (~0.20 m) to clear obstacles while being penalized for it (B3 stuck 0.20 m across 25k/50k).
        # Mirror the proven parent (taili_amp_env.py:900-916): (a) per-foot clearance above the nearest
        # height-scanner ray under EACH foot, and (b) a roughness-derived local_obstacle_h so the reward
        # raises the target on real terrain (target -> obstacle+margin) and keeps it 0.08 on flat — so a
        # low flat swing is the reward-optimal gait on flat while high lift is still free on terrain.
        try:
            _hits = self._height_scanner.data.ray_hits_w                              # (N,P,3) world
            _hz = torch.nan_to_num(_hits[:, :, 2], nan=0.0, posinf=0.0, neginf=0.0)
            _dxy = foot_pos[:, :, None, :2] - _hits[:, None, :, :2]                    # (N,4,P,2)
            _nearest = torch.argmin(torch.nan_to_num((_dxy * _dxy).sum(dim=-1), nan=1e9), dim=-1)  # (N,4)
            _ground_z = torch.gather(_hz, 1, _nearest)                                # (N,4)
            foot_clearance_terr = torch.clamp(foot_pos[:, :, 2] - _ground_z, min=0.0)  # (N,4) above local ground
            _rough = self._terrain_ctx[:, 2] if self._terrain_ctx.shape[-1] > 2 else torch.zeros(N, device=dev)
            local_obstacle_h = (torch.clamp(
                (_rough - float(getattr(self.cfg, "clr_rough_flat", 0.01)))
                / max(float(getattr(self.cfg, "clr_rough_span", 0.12)), 1e-6), 0.0, 1.0)
                * float(getattr(self.cfg, "clr_rough_bonus_max", 0.26)))              # (N,) 0 on flat, ->~0.26 on rough
            # DISCRETE terrains (stairs/boxes) FORCE a full obstacle height (0706, terrain diag D3p):
            # on stair TREADS the roughness std reads low → the target relaxed toward flat 0.08 → the
            # swing foot clipped the next RISER (38° pitch spike, forward-ascent stumble). Same
            # catch-22 the parent's discrete_clearance solved: demand riser+margin lift everywhere on
            # discrete terrain, not just near edges. 0.22 = max step 0.18 + spec D margin 0.04.
            self._ensure_gate_mask()
            _disc = getattr(self, "_discrete_terrain_mask", None)
            if _disc is not None:
                # 方向感知强制抬脚 (0706 D3r): 只有"沿指令方向前方地面在升高"(上行)才强制 0.22m;
                # 方向盲的整环境强制曾在下行帧也逼高抬腿 → 更长滞空+更重落地+前倾栽头,
                # 把 D[stairs] 摔倒率从 18.8% 推到 40.6%(boxes 平顶无下行,故同修复反而受益)。
                # 奖励侧允许特权信息:用高度扫描的前方最高点 - 脚下地面 = 前方升量。
                from isaaclab.utils.math import quat_apply
                _hitsw = self._height_scanner.data.ray_hits_w                       # (N,P,3)
                _relxy = _hitsw[:, :, :2] - self.robot.data.root_pos_w[:, None, :2]
                _cmd3 = torch.nn.functional.pad(self.commands[:, :2], (0, 1))
                _cmdw = quat_apply(self.robot.data.root_quat_w, _cmd3)[:, :2]
                _cn = _cmdw / _cmdw.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                _ahead = (_relxy * _cn[:, None, :]).sum(-1) > 0.15                   # 指令方向前方的扫描点
                _hz2 = torch.nan_to_num(_hitsw[:, :, 2], nan=-1e3, posinf=-1e3, neginf=-1e3)
                _ahead_max = torch.where(_ahead, _hz2, torch.full_like(_hz2, -1e3)).max(dim=1).values
                _rise = _ahead_max - terrain_h                                       # >0 = 前方升高
                _ascending = _disc & (_rise > 0.04) & (self.commands[:, :2].norm(dim=-1) > 0.1)
                # ADAPTIVE clearance (0706): the old fixed clamp(min=0.22) capped the lift target at ~0.22-0.26m
                # regardless of the ACTUAL step height, so 25-30cm risers were unclimbable (foot maxed at 0.20m).
                # `_rise` already holds the true riser height ahead — target it + margin, up to 0.40m, so the
                # policy learns to lift as high as the step demands. Paired with the foot_h4 perceiver channel
                # (which lets the blind latent estimate the same height at deploy) this is the stairs unlock.
                _adaptive = (_rise + 0.06).clamp(0.10, 0.40)                          # riser + 6cm margin, ≤40cm
                local_obstacle_h = torch.where(_ascending, torch.maximum(local_obstacle_h, _adaptive), local_obstacle_h)
        except Exception:
            foot_clearance_terr = torch.clamp(foot_pos[:, :, 2] - terrain_h[:, None], min=0.0)
            local_obstacle_h = torch.zeros(N, device=dev)
        quality = self._update_quality_windows(in_contact, settled_contact, foot_vel_xy, base_h, tilt_rel, moving)
        self._base_h_min = float(base_h.min())
        self._tilt_deg_max = float(torch.rad2deg(tilt_rel).max())
        support_stand = torch.clamp((3.0 - cc) / 3.0, 0.0, 1.0)
        support_move = torch.clamp((2.0 - cc) / 2.0, 0.0, 1.0)
        support_instab = torch.where(moving > 0.5, support_move, support_stand)
        support_instab = torch.maximum(support_instab,
                                       torch.clamp((tilt_rel - math.radians(10)) / math.radians(20), 0.0, 1.0))
        support_instab = torch.maximum(support_instab, torch.clamp(quality["height_low_risk"], 0.0, 1.0))
        support_instab = torch.maximum(support_instab, torch.clamp(quality["tilt_high_risk"], 0.0, 1.0))

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = self.reset_terminated if hasattr(self, "reset_terminated") else torch.zeros(N, dtype=torch.bool, device=dev)
        terminal_window = (terminated & ~time_out).float()

        gate = taili_reward.stable_motion_gate(base_h, tilt_rel, support_instab,
                                               torch.zeros(N, device=dev), terminal_window, cfg)
        torque_limit = self.robot.actuators["legs"].effort_limit

        inp = types.SimpleNamespace(
            cmd=self.commands,
            base_lin_vel=rd.root_lin_vel_b,
            base_ang_vel=rd.root_ang_vel_b,
            base_h_above_terrain=base_h,
            tilt_rel=tilt_rel,
            stable_motion_gate=gate,
            stand_gate=stand_gate,
            moving_gate=moving,
            terminal_window=terminal_window,
            foot_contact=in_contact,
            stance_slip_contact=settled_contact,
            desired_foot_contact=(self._leg_phases() < self.cfg.gait_duty).float(),
            foot_vel_xy=foot_vel_xy,                                                  # (N,4)
            duty_by_leg_window=quality["duty_by_leg"],
            duty_quality_window=quality["duty_quality"],
            duty_symmetry_window=quality["duty_symmetry_score"],
            duty_target_window=quality["duty_target_score"],
            duty_spread_window=quality["duty_spread"],
            diagonal_pair_window=quality["diag_pair_score"],
            stance_slip_speed_window=quality["slip_speed"],
            stance_slip_high_fraction=quality["slip_high_fraction"],                  # RANK-2B: p90 TAIL proxy (frac settled feet slipping >thr) — B2 scores the tail, the mean window hides it
            touchdown_vz=foot_vel[:, :, 2].abs(),                                     # (N,4) proxy
            touchdown_mask=td_mask,                                                   # (N,4) 1@landing or None (legacy); anomaly #5 B1
            # last_air_time, NOT current_air_time: the sensor latches the completed stride's air
            # duration into last_air_time and ZEROES current_air_time on the touchdown step — with
            # current_air_time the (air − target)·touchdown term degenerates to a constant −target
            # per landing (stride-length blind, and a relative subsidy for not stepping at all).
            foot_air_time=self._contact_sensor.data.last_air_time[:, self._feet_contact_ids],
            # per-env air-time target = the gait clock's scheduled swing duration period*(1-duty),
            # recomputed from the same speed-adaptive period the clock uses, so the anti-shuffle
            # reward can never fight the clock (a fixed 0.30 s asked for longer swings than the
            # clock schedules -> gait_match capped ~0.79).
            air_time_target=(torch.clamp(
                # YAW-AWARE CADENCE (0705 D3m): effective speed includes the yaw foot-lever
                # (0.30*|wz| ~ |wz|*r_foot), so a high-yaw command shortens the period → the anti-shuffle
                # reward demands FASTER stepping for a fast turn. Yaw-blind cadence (‖v_xy‖ only) let
                # high-yaw commands step at the slow nominal cadence — too few steps to realize the turn,
                # worsening as yaw rises (A2 p90 got WORSE with training). Mirrored in the AMP reference.
                self.cfg.gait_period - self.cfg.gait_period_slope * (
                    torch.norm(self.commands[:, :2], dim=1) + 0.15 * self.commands[:, 2].abs()),
                min=self.cfg.gait_period_min, max=self.cfg.gait_period) * (1.0 - self.cfg.gait_duty)),
            foot_clearance=foot_clearance_terr,                                       # per-foot, above LOCAL ground (#10)
            local_obstacle_h=local_obstacle_h,                                        # roughness-derived; 0 on flat (#10)
            hip_deviation=(rd.joint_pos[:, self._hip_joint_ids]
                           - self.action_offset[:, self._hip_joint_ids]).abs().mean(dim=-1),
            torque=rd.applied_torque,
            torque_limit=torque_limit,
            torque_clamped=(rd.applied_torque.abs() >= torque_limit).float(),
            action=self.actions,
            last_action=self.last_actions,
            default_pose_error=(rd.joint_pos - self.action_offset).abs().mean(dim=1),
            amp_reward=torch.zeros(N, device=dev),                                    # skrl AMP adds style reward
            body_collision=torch.zeros(N, device=dev),                               # TODO non-foot contact sampling
            quality_gate=torch.full((N,), float(getattr(self, "_penalty_gate", 0.0)), device=dev),
            terminal_reason=None,                                                     # per-env terminal handled below
        )
        comp = taili_reward.compute_reward_components(inp, cfg)
        total = comp["total"] - cfg.w_terminal * terminal_window                      # per-env terminal penalty
        # 绊脚惩罚 (0706 D3q): 足端水平接触力 > 2x 竖直力 = 脚踹在台阶沿/竖直面上(绊脚),这是
        # D[stairs] 摔倒(40.6%)的直接机制 —— 经典 legged-gym feet_stumble 项。仅在真实接触(>5N)
        # 且水平主导时触发;量级为绊脚足占比(0~1),W=1.0 与其他步态罚同阶。
        _ff3 = self._contact_sensor.data.net_forces_w[:, self._feet_contact_ids, :]
        _stumble = ((_ff3[:, :, :2].norm(dim=-1) > 2.0 * _ff3[:, :, 2].abs().clamp(min=1.0))
                    & (_ff3.norm(dim=-1) > 5.0)).float().mean(dim=1)
        comp["stumble"] = -1.0 * _stumble                                             # 进 TPPEN 日志
        total = total + comp["stumble"]
        # 爬升激励 (0706, STAIRS): the reward field had NO term for GAINING height — terrain_progress
        # rewards horizontal command-tracking, so on stairs the robot maximised reward by walking the flat
        # pit floor and never committing to climb. Reward upward base velocity while moving on discrete
        # terrain so climbing a riser PAYS. Stateless (no prev-height buffer to reset); clamp(min=0) so a
        # downstroke is never penalised. Gated to real forward intent + discrete terrain only (flat/slope
        # unaffected). W matched to a mid gait term.
        _disc_climb = getattr(self, "_discrete_terrain_mask", None)
        if _disc_climb is not None:
            _moving_cmd = self.commands[:, :2].norm(dim=-1) > 0.1
            _climb = self.robot.data.root_lin_vel_w[:, 2].clamp(min=0.0) \
                * _disc_climb.float() * _moving_cmd.float()
            comp["climb"] = float(getattr(self.cfg, "w_climb", 1.0)) * _climb          # 进 TPREW 日志
            total = total + comp["climb"]
        # (A) LIVE JOINT-IMITATION reward (RANK-1, 0707) — THE positive attractor toward the clean reference
        # gait. The parent's r_imitate (taili_amp_env._get_rewards) is DEAD in this TP subclass: it OVERRIDES
        # _get_rewards and never calls super(), so the policy only ever saw the reference through the
        # contact-blind AMP discriminator → a crude skating gait (feet slip / slam on landing) was the reward
        # optimum, failing B1 (soft landing) / B2 (no slip) / B4 (L-R symmetry) / C (clean stand). Reward
        # MATCHING the analytic reference JOINTS each step: this reproduces the reference's soft sin^2 touchdown
        # (B1), velocity-matched no-slip retraction (B2) and L/R symmetry (B4/C) BY CONSTRUCTION. POSITIVE, so it
        # joins the positive-reward sum — immune to the penalty-budget homeostat that neuters the penalties. NEW
        # cfg knobs (w_imitate_live / imitate_sigma_live) leave the dead parent's rew_imitate/imitate_sigma alone.
        _w_imitate = float(getattr(self.cfg, "w_imitate_live", 0.0))
        if _w_imitate != 0.0:
            # PHASE ALIGNMENT (the subtle part): flat_reference maps time→per-leg phase via
            # p = ((t / T_ref) + TROT[leg]) % 1, with T_ref recomputed INSIDE flat_reference from the PLANAR
            # command speed (norm(cmd[:2])). Passing t = self._gait_phase * T_im with T_im == that SAME planar
            # period makes t / T_ref == self._gait_phase, so p == (self._gait_phase + TROT) == self._leg_phases()
            # exactly (flat_reference's TROT {FL,FR,RL,RR}=0,.5,.5,0 == self._trot_offsets). We feed the current
            # phase VALUE (not integrated time), so the env clock's yaw-inclusive ADVANCE rate is irrelevant —
            # only the phase→joints map matters, and it is phase-aligned to the gait clock the env actually runs.
            # T_im MUST stay PLANAR to match flat_reference's internal T_ref (same recipe the parent dead code used).
            _spd_im = torch.norm(self.commands[:, :2], dim=1)
            _T_im = torch.clamp(self.cfg.gait_period - self.cfg.gait_period_slope * _spd_im,
                                min=self.cfg.gait_period_min, max=self.cfg.gait_period)
            _rough_im = self._terrain_ctx[:, 2] if self.cfg.terrain_ctx_dim >= 3 else None
            _jp_ref = flat_reference(
                self.commands, self._gait_phase * _T_im,
                gait_period=self.cfg.gait_period, gait_period_slope=self.cfg.gait_period_slope,
                gait_period_min=self.cfg.gait_period_min, clearance_base=self.cfg.base_clearance,
                roughness=_rough_im, clearance_rough_gain=self.cfg.ref_clearance_rough_gain,
                stance_dx=self.cfg.stance_dx, jp_only=True, iters=12)                  # FAST PATH: jp only, 12 IK iters
            # flat_reference emits CLIP dof order [hip x4, thigh x4, calf x4]; motion_dof_indexes maps that to the
            # sim joint order of rd.joint_pos (identity in this env — see collect_reference_motions — but apply it
            # so the match stays correct even if the layouts ever diverge, exactly like the parent dead code).
            _jp_ref = _jp_ref[:, self.motion_dof_indexes]
            _jdiff = rd.joint_pos - _jp_ref                                            # (N,12) sim vs reference, aligned
            # FORWARD ANCHOR: unlike the parent (which turn-gated forward to ~0), FLOOR the turn fraction at
            # imitate_fwd_floor so pure-forward buckets — where B1/B2/B4 are scored — ALSO get anchored. The
            # reference is speed-COVARIANT (period/stride scale with |cmd|) so it constrains gait SHAPE, not net
            # speed → forward velocity tracking (A1) is protected.
            _turn_mag = self.commands[:, 1].abs() + 0.30 * self.commands[:, 2].abs()
            _fwd_mag = self.commands[:, 0].abs()
            _turn_frac = _turn_mag / (_turn_mag + _fwd_mag + 1e-3)
            _turn_frac_floored = torch.clamp(_turn_frac, min=float(getattr(self.cfg, "imitate_fwd_floor", 0.3)))
            _sigma_im = float(getattr(self.cfg, "imitate_sigma_live", 0.12))
            comp["imitate"] = (_w_imitate * _turn_frac_floored
                               * torch.exp(-_jdiff.pow(2).sum(dim=1) / _sigma_im) * moving)   # moving-only positive attractor
            total = total + comp["imitate"]
            # phi1->phi2 phase-gate style metric: mean per-joint error vs reference over moving envs.
            _mv = moving > 0.5
            self._style_err = (float(_jdiff.abs().mean(dim=1)[_mv].mean()) if bool(_mv.any())
                               else getattr(self, "_style_err", 1.0))
        # (C) SWING-DIRECTION attractor (0708, "脚在空中往滑" fix) — POSITIVE, command-directional, MID-SWING.
        # No reward constrained the swing foot's IN-AIR fore-aft velocity, so the forward swing primitive leaks
        # onto lateral/back/yaw commands (foot slides body-FORWARD when NOT commanded forward). A penalty is
        # doomed by the penalty-budget homeostat (why the touchdown penalty stayed flat ~-0.009); this is a
        # POSITIVE attractor keyed to the reference's OWN per-foot commanded stride rate vfx_ref=vx-wz*foot_y0,
        # so a CORRECT forward swing is a FIXED POINT (untouched, speed-covariant → A1 protected) and only
        # UNcommanded body-forward drift falls off the attractor. Gated to mid-air feet (in_contact<0.5) =
        # exactly the frame the failed touchdown penalty never read. Fore-aft only → terrain-clearance-safe.
        _w_swing = float(getattr(self.cfg, "w_swing_dir", 0.0))
        if _w_swing != 0.0:
            from isaaclab.utils.math import quat_apply_inverse
            _qsw = rd.root_quat_w[:, None, :].expand(-1, 4, -1).reshape(-1, 4)
            _v_rel_w = foot_vel - rd.root_lin_vel_w[:, None, :]                           # (N,4,3) foot vel REL base, world
            _v_fwd_b = quat_apply_inverse(_qsw, _v_rel_w.reshape(-1, 3)).reshape(self.num_envs, 4, 3)[:, :, 0]  # body fore-aft
            _foot_y0 = 0.2082 * torch.tensor([1., -1., 1., -1.], device=self.device)      # sy*(HIPY+THIGHY): FL,FR,RL,RR
            _vfx_ref = self.commands[:, 0:1] - self.commands[:, 2:3] * _foot_y0[None, :]   # (N,4) commanded fore-aft stride rate
            _margin_sw = float(getattr(self.cfg, "swing_dir_margin", 0.15))
            # allow up to the reference's mid-swing PEAK (2*vfx_ref, the foot advances at ~2x the average mid-swing)
            # + a fixed margin, so a CORRECT forward swing (vx>0) is fully free; only UNcommanded fwd drift is caught.
            _excess_sw = torch.clamp(_v_fwd_b - 2.0 * _vfx_ref.clamp(min=0.0) - _margin_sw, min=0.0)
            _swing_m = (in_contact < 0.5).float()                                         # (N,4) mid-air feet
            _sw_err = (_swing_m * _excess_sw.pow(2)).sum(dim=1) / _swing_m.sum(dim=1).clamp(min=1.0)
            _sig_sw = float(getattr(self.cfg, "swing_dir_sigma", 0.08))
            comp["swing_dir"] = _w_swing * moving * torch.exp(-_sw_err / _sig_sw)          # POSITIVE → joins pos_sum
            total = total + comp["swing_dir"]
            self._swing_excess = float((_swing_m * _excess_sw).sum() / _swing_m.sum().clamp(min=1.0))  # measurement EMA-able
        # (B) SETTLE-BRAKE reward (A3/C, 0708) — reward active DECELERATION when commanded to STOP but still
        # moving. taili_reward's stand term rewards the speed LEVEL (~0), NOTHING rewards the RATE of braking,
        # so the policy COASTS to a halt (physeval A3 fails ONLY on settle=1.54s > 1.0s spec; C fails only via
        # A3). Rewarding the per-step speed DROP while stopping breaks the coast. Gated to cmd~0 AND still-moving
        # (>0.08 m/s) so it can't be farmed during locomotion, can't fight tracking, and the post-reset one-step
        # staleness of _prev_settle_speed (env resets to ~0 speed) is masked by the >0.08 gate. w=0 => byte-noop.
        _w_brake = float(getattr(self.cfg, "w_settle_brake", 0.0))
        if _w_brake != 0.0:
            if not hasattr(self, "_prev_settle_speed"):
                self._prev_settle_speed = torch.zeros(self.num_envs, device=self.device)
            _spd_now = torch.linalg.norm(rd.root_lin_vel_b[:, :2], dim=-1)
            _cmd_still = ((torch.norm(self.commands[:, :2], dim=1) <= 0.1)
                          & (self.commands[:, 2].abs() <= 0.05))
            _decel = torch.clamp(self._prev_settle_speed - _spd_now, min=0.0)          # >0 when slowing
            comp["settle_brake"] = (_w_brake * _decel * _cmd_still.to(rd.joint_pos.dtype)
                                    * (_spd_now > 0.08).to(rd.joint_pos.dtype))
            total = total + comp["settle_brake"]
            self._prev_settle_speed = _spd_now.detach()
        with torch.no_grad():
            cmd_xy = self.commands[:, :2]
            cmd_xy_norm_sq = torch.sum(cmd_xy * cmd_xy, dim=1)
            cmd_wz_sq = self.commands[:, 2] * self.commands[:, 2]
            progress_thr = 0.0025
            lin_prog = torch.clamp(
                torch.sum(rd.root_lin_vel_b[:, :2] * cmd_xy, dim=1) / cmd_xy_norm_sq.clamp(min=progress_thr),
                0.0,
                1.0,
            )
            ang_prog = torch.clamp(
                rd.root_ang_vel_b[:, 2] * self.commands[:, 2] / cmd_wz_sq.clamp(min=progress_thr),
                0.0,
                1.0,
            )
            moving_lin = cmd_xy_norm_sq > progress_thr
            moving_ang = cmd_wz_sq > progress_thr
            # Gate the per-direction progress that drives phase-advance/regress, DR level-up and the
            # velocity ceiling on the SAME curriculum-gate mask the parent uses (exclude hard terrain
            # types slope_inv/boxes that stay stuck at low levels). fall_rate/upright in the shared
            # _log_training_diag are already gate-masked, so leaving progress unmasked made the
            # curriculum signal self-inconsistent and let ~25% hard-terrain envs drag advancement.
            self._ensure_gate_mask()
            gm_e = self._gate_mask
            fwd_mask = moving_lin & (self.commands[:, 0] > 0.1) & gm_e
            back_mask = moving_lin & (self.commands[:, 0] < -0.1) & gm_e
            lat_mask = moving_lin & (self.commands[:, 1].abs() > 0.1) & gm_e
            yaw_mask = moving_ang & gm_e
            if bool(fwd_mask.any()):
                self._fwd_prog = float(lin_prog[fwd_mask].mean())
            if bool(back_mask.any()):
                self._back_prog = float(lin_prog[back_mask].mean())
            if bool(lat_mask.any()):
                self._lat_prog = float(lin_prog[lat_mask].mean())
            if bool(yaw_mask.any()):
                self._yaw_prog = float(ang_prog[yaw_mask].mean())
            gait_match = torch.zeros(N, device=dev)
            moving_any = moving > 0.5
            try:
                leg_phase = self._leg_phases()
                desired_stance = (leg_phase < self.cfg.gait_duty).float()
                gait_match = (desired_stance * in_contact + (1.0 - desired_stance) * (1.0 - in_contact)).mean(dim=1)
                gait_mean = float(gait_match[moving_any].mean()) if bool(moving_any.any()) else 0.0
                fl, fr, rl, rr = in_contact[:, 0], in_contact[:, 1], in_contact[:, 2], in_contact[:, 3]
                diag = 1.0 - 0.5 * ((fl - rr).abs() + (fr - rl).abs())
                diag_pair_inst_mean = float(quality["diag_pair_inst"][moving_any].mean()) if bool(moving_any.any()) else 0.0
                front = 0.5 * (fl + fr)
                rear = 0.5 * (rl + rr)
                left = 0.5 * (fl + rl)
                right = 0.5 * (fr + rr)
                duty_balance = 1.0 - torch.clamp((rear - front).abs() + (left - right).abs(), 0.0, 1.0)
                diag_mean = float(quality["diag_pair_score"][moving_any].mean()) if bool(moving_any.any()) else 0.0
                duty_balance_mean = float(quality["duty_quality"][moving_any].mean()) if bool(moving_any.any()) else 0.0
                duty_balance_inst_mean = float(duty_balance[moving_any].mean()) if bool(moving_any.any()) else 0.0
            except Exception:
                gait_mean = 0.0
                diag_mean = 0.0
                diag_pair_inst_mean = 0.0
                duty_balance_mean = 0.0
                duty_balance_inst_mean = 0.0
            foot_slip_xy = foot_vel_xy
            self._slip_now = float(quality["slip_speed"][moving_any].mean()) if bool(moving_any.any()) else 0.0
            self._slip_inst = float(quality["slip_speed_inst"][moving_any].mean()) if bool(moving_any.any()) else 0.0
            self._slip_high_fraction = float(quality["slip_high_fraction"][moving_any].mean()) if bool(moving_any.any()) else 0.0
            self._diag_contact = diag_mean
            self._diag_pair_instant = diag_pair_inst_mean
            self._duty_balance = duty_balance_mean
            self._duty_balance_instant = duty_balance_inst_mean
            self._duty_spread = float(quality["duty_spread"][moving_any].mean()) if bool(moving_any.any()) else 0.0
            self._duty_target_score = float(quality["duty_target_score"][moving_any].mean()) if bool(moving_any.any()) else 0.0
            self._duty_symmetry_score = float(quality["duty_symmetry_score"][moving_any].mean()) if bool(moving_any.any()) else 0.0
            self._height_low_risk = float(quality["height_low_risk"].mean())
            self._tilt_high_risk = float(quality["tilt_high_risk"].mean())
            duty_by_leg_mean = quality["duty_by_leg"].mean(dim=0)
            self._duty_by_leg = {
                "FL": float(duty_by_leg_mean[0]),
                "FR": float(duty_by_leg_mean[1]),
                "RL": float(duty_by_leg_mean[2]),
                "RR": float(duty_by_leg_mean[3]),
            }
            self._style_err = getattr(self, "_style_err", 1.0)
            self._dbg = (
                float(lin_prog[moving_lin].mean()) if bool(moving_lin.any()) else 0.0,
                float(ang_prog[moving_ang].mean()) if bool(moving_ang.any()) else 0.0,
                float(comp["tracking_lin"].mean()),
                float(comp["tracking_yaw"].mean()),
                0.0,
                gait_mean,
            )
            self._last_gait_match = gait_mean
            # invariant-4a budget monitor: |regular penalties| / positive task reward. Tier-S
            # event penalties (collision / saturation / terminal) are excluded — 4b allows them
            # to exceed the task. The parent's penalty ramp steps DOWN when this exceeds
            # penalty_budget_ratio_max, so quality penalties can never make walking unprofitable.
            _tier_s = ("torque_saturation", "terminal_penalty")
            _skip = ("total", "stable_motion_gate", "stand_gate", "moving_gate", "quality_gate")
            pos_sum, neg_sum = 0.0, 0.0
            for _k, _v in comp.items():
                if _k in _skip or _k in _tier_s or not torch.is_tensor(_v):
                    continue
                _m = float(_v.mean())
                if _m >= 0.0:
                    pos_sum += _m
                else:
                    neg_sum += -_m
            _ratio = neg_sum / max(pos_sum, 1e-6)
            _prev = float(getattr(self, "_budget_ratio_ema", _ratio))
            self._budget_ratio_ema = 0.995 * _prev + 0.005 * _ratio
            self._rew_dbg = {
                "lin": float(comp["tracking_lin"].mean()),
                "ang": float(comp["tracking_yaw"].mean()),
                "gait": float(comp["gait_anchor"].mean()),
                "diag": float(comp["diagonal_contact"].mean()),
                "duty": float(comp["duty_balance"].mean()),
                "imit": 0.0,
                "stand": float(comp["stand"].mean()),
                "height": 0.0,
                "slip": float(comp["stance_slip"].mean()),
                "clear": float((comp["clearance_under"] + comp["clearance_over"]).mean()),
                "hip": 0.0,
                "offax": float(comp["off_axis"].mean()),
                "land": float(comp["landing_impact"].mean()),
                "torq": float((comp["torque_margin"] + comp["torque_saturation"]).mean()),
                "arate": float(comp["action_rate"].mean()),
                "vz": 0.0,
                "wxy": 0.0,
            }

        # periodic progress log (poll-able): reward / tracking / speed / gate / collapse
        self._rew_log_step = getattr(self, "_rew_log_step", 0) + 1
        telemetry_interval = max(1, int(os.environ.get("TAILI_TELEMETRY_INTERVAL", "10") or 10))
        total_steps = int(os.environ.get("TAILI_TOTAL_STEPS", "0") or 0)
        emit_telemetry = (
            self._rew_log_step == 1
            or self._rew_log_step % telemetry_interval == 0
            or (total_steps > 0 and self._rew_log_step >= total_steps)
        )
        if emit_telemetry and not getattr(self, "use_external_commands", False):
            with torch.no_grad():
                vb = rd.root_lin_vel_b
                cmd_xy = self.commands[:, :2]
                cmd_xy_norm = torch.linalg.norm(cmd_xy, dim=1)
                lin_err_vec = torch.linalg.norm(cmd_xy - vb[:, :2], dim=1)
                lin_err = lin_err_vec.mean().item()
                yaw_err = (rd.root_ang_vel_b[:, 2] - self.commands[:, 2]).abs().mean().item()
                speed = float(torch.norm(vb[:, :2], dim=1).mean())
                gait = float(gate.mean())
                v_along = torch.sum(vb[:, :2] * cmd_xy, dim=1) / torch.clamp(cmd_xy_norm, min=1e-6)
                progress_ratio = torch.clamp(v_along / torch.clamp(cmd_xy_norm, min=1e-6), 0.0, 1.0)
                torque_limit_t = torch.as_tensor(torque_limit, device=dev).float()
                if torque_limit_t.ndim == 1:
                    torque_limit_t = torque_limit_t.unsqueeze(0).expand_as(rd.applied_torque)
                torque_util = (rd.applied_torque.abs() / torque_limit_t.clamp_min(1e-6)).mean()
                terrain_mean = None
                terrain_max = None
                if hasattr(self, "_terrain") and hasattr(self._terrain, "terrain_levels"):
                    terrain_levels = self._terrain.terrain_levels.float()
                    terrain_mean = float(terrain_levels.mean())
                    terrain_max = int(terrain_levels.max().item())
                phase = getattr(self, "_phase", None)
                dr_level = getattr(self, "_dr_level", None)
                progress_by_dir = {
                    "fwd": float(getattr(self, "_fwd_prog", 0.0)),
                    "back": float(getattr(self, "_back_prog", 0.0)),
                    "lat": float(getattr(self, "_lat_prog", 0.0)),
                    "yaw": float(getattr(self, "_yaw_prog", 0.0)),
                }
                active_dirs = ()
                if active_direction_progress is not None:
                    progress_gate, active_dirs = active_direction_progress(progress_by_dir, self.cfg, getattr(self, "_phase", None))
                else:
                    progress_gate = min(progress_by_dir.values())
                gait_gate = float(getattr(self, "_last_gait_match", 0.0))
                fall_rate = None
                try:
                    # a real fall is LOW **and** TILTED — an UPRIGHT robot on low terrain (descended a
                    # slope/stair, abs z < termination_height while fine) is NOT a fall. The old abs-height-only
                    # check pinned fall_rate at ~0.10 (upright-but-low envs) and blocked the phase gate (0707).
                    _below = rd.root_pos_w[:, 2] < self.cfg.termination_height
                    _tilted = rd.projected_gravity_b[:, 2] >= -0.7   # NOT upright
                    fall_rate = float((_below & _tilted).float().mean())
                except Exception:
                    pass
                reward_payload = {
                    "total": float(total.mean()),
                    "lin_err": lin_err,
                    "speed": speed,
                    "gait": gait,
                    "base_h": float(base_h.mean()),
                    "upright": float((-rd.projected_gravity_b[:, 2]).clamp(0.0, 1.0).mean()),
                }
                for name, value in comp.items():
                    if torch.is_tensor(value) and value.numel() > 0:
                        reward_payload[name] = float(value.mean())
                if not self._reward_cfg_printed:
                    self._reward_cfg_printed = True
                    try:
                        reward_cfg_payload = (
                            asdict(cfg)
                            if is_dataclass(cfg)
                            else {
                                name: float(getattr(cfg, name))
                                for name in dir(cfg)
                                if not name.startswith("_") and isinstance(getattr(cfg, name), (int, float))
                            }
                        )
                    except Exception:
                        reward_cfg_payload = {}
                    for key, value in reward_cfg_payload.items():
                        if isinstance(value, (int, float)):
                            reward_payload[f"cfg_{key}"] = float(value)
                command_payload = {
                    "cmd_vx": float(self.commands[:, 0].mean()),
                    "cmd_vy": float(self.commands[:, 1].mean()),
                    "cmd_wz": float(self.commands[:, 2].mean()),
                    "actual_vx": float(vb[:, 0].mean()),
                    "actual_vy": float(vb[:, 1].mean()),
                    "actual_wz": float(rd.root_ang_vel_b[:, 2].mean()),
                    "v_along": float(v_along.mean()),
                    "speed_xy": speed,
                    "lin_err": lin_err,
                    "yaw_err": yaw_err,
                    "progress_ratio": float(progress_ratio.mean()),
                    # Directional progress is the operational blocker for phase gates.
                    # Keep it in JSONL instead of relying on human console.log parsing.
                    "progress_fwd": progress_by_dir["fwd"],
                    "progress_back": progress_by_dir["back"],
                    "progress_lat": progress_by_dir["lat"],
                    "progress_yaw": progress_by_dir["yaw"],
                    "progress_min_active": float(progress_gate),
                    "progress_lagging_dir": min(
                        (name for name in (active_dirs or progress_by_dir.keys())),
                        key=lambda name: progress_by_dir.get(str(name), 0.0),
                    ) if progress_by_dir else "",
                    "gait_match": float(getattr(self, "_last_gait_match", 0.0)),
                    "diagonal_contact": float(getattr(self, "_diag_contact", 0.0)),
                    "duty_balance": float(getattr(self, "_duty_balance", 0.0)),
                    "stance_slip": float(getattr(self, "_slip_now", 0.0)),
                    "diagonal_pair_instant": float(getattr(self, "_diag_pair_instant", 0.0)),
                    "duty_balance_instant": float(getattr(self, "_duty_balance_instant", 0.0)),
                    "duty_spread_window": float(getattr(self, "_duty_spread", 0.0)),
                    "duty_target_score": float(getattr(self, "_duty_target_score", 0.0)),
                    "duty_symmetry_score": float(getattr(self, "_duty_symmetry_score", 0.0)),
                    "stance_slip_instant": float(getattr(self, "_slip_inst", 0.0)),
                    "stance_slip_high_fraction": float(getattr(self, "_slip_high_fraction", 0.0)),
                }
                duty_by_leg = getattr(self, "_duty_by_leg", {})
                if isinstance(duty_by_leg, dict):
                    for leg, value in duty_by_leg.items():
                        command_payload[f"duty_{str(leg).lower()}"] = float(value)
                curriculum_payload = {
                    "phase": f"phi{phase}" if phase is not None else "",
                    "command_mode": getattr(self, "_last_command_mode", ""),
                    "terrain_mean": terrain_mean,
                    "terrain_max": terrain_max,
                    "dr_level": dr_level,
                    "phase_count": float(getattr(self, "_phase_count", 0)),
                    "penalty_gate": float(getattr(self, "_penalty_gate", 0.0)),
                    "budget_ratio": float(getattr(self, "_budget_ratio_ema", 0.0)),
                    "clearance_gate": float(getattr(self, "_clearance_gate", 0.0)),
                    "terrain_health_ok": 1.0 if getattr(self, "_terrain_health_ok", True) else 0.0,
                }
                if progress_gate is not None:
                    curriculum_payload["progress_gate"] = progress_gate
                    curriculum_payload["progress_fwd"] = progress_by_dir["fwd"]
                    curriculum_payload["progress_back"] = progress_by_dir["back"]
                    curriculum_payload["progress_lat"] = progress_by_dir["lat"]
                    curriculum_payload["progress_yaw"] = progress_by_dir["yaw"]
                    curriculum_payload["yaw_ceil"] = float(getattr(self, "_vel_max_yaw", 0.0))   # reachable-ramp diagnostic
                    curriculum_payload["progress_lagging_dir"] = command_payload["progress_lagging_dir"]
                    if active_dirs:
                        curriculum_payload["active_dirs"] = ",".join(active_dirs)
                    cfg_obj = getattr(self, "cfg", None)
                    progress_threshold = None
                    if cfg_obj is not None and phase is not None:
                        progress_threshold = getattr(cfg_obj, f"phase_gate_prog_{int(phase)}", None)
                    if progress_threshold is not None and progress_gate < float(progress_threshold):
                        curriculum_payload["blocked_by"] = "progress"
                        curriculum_payload["next_gate"] = "phase_progress"
                if gait_gate is not None:
                    curriculum_payload["gait_gate"] = gait_gate
                    curriculum_payload["diagonal_gate"] = float(getattr(self, "_diag_contact", 0.0))
                    curriculum_payload["duty_balance_gate"] = float(getattr(self, "_duty_balance", 0.0))
                    curriculum_payload["slip_gate"] = float(getattr(self, "_slip_now", 0.0))
                    curriculum_payload["duty_spread_gate"] = float(getattr(self, "_duty_spread", 0.0))
                if fall_rate is not None:
                    cfg_obj = getattr(self, "cfg", None)
                    fall_threshold = getattr(cfg_obj, "phase_gate_fall_2", None) if cfg_obj is not None else None
                    curriculum_payload["fall_gate"] = fall_rate
                    if fall_threshold is not None and fall_rate >= float(fall_threshold):
                        curriculum_payload["blocked_by"] = curriculum_payload.get("blocked_by", "fall")
                        curriculum_payload["next_gate"] = curriculum_payload.get("next_gate", "fall_rate")
                health_payload = {
                    "stable_motion_gate": float(gate.mean()),
                    "moving_gate": float(moving.mean()),
                    "stand_gate": float(stand_gate.mean()),
                    "base_h": float(base_h.mean()),
                    "upright": float((-rd.projected_gravity_b[:, 2]).clamp(0.0, 1.0).mean()),
                    "tilt_deg": float(torch.rad2deg(tilt_rel).mean()),
                    "support_instability": float(support_instab.mean()),
                    "height_low_risk_window": float(getattr(self, "_height_low_risk", 0.0)),
                    "tilt_high_risk_window": float(getattr(self, "_tilt_high_risk", 0.0)),
                    "base_h_min": float(base_h.min()),
                    "tilt_deg_max": float(torch.rad2deg(tilt_rel).max()),
                    "contact_count": float(cc.mean()),
                    "contacts_mean": float(in_contact.mean()),
                    "torque_util": float(torque_util),
                    "terminal_rate": float(terminal_window.mean()),
                }
                if fall_rate is not None:
                    health_payload["fall_rate"] = fall_rate
                if self._telemetry is not None:
                    self._telemetry.emit(
                        step=self._rew_log_step,
                        total_steps=total_steps or None,
                        reward=reward_payload,
                        curriculum=curriculum_payload,
                        health=health_payload,
                        command=command_payload,
                    )
                else:
                    print("[TPREW] step %d rew %.3f lin_err %.3f speed %.3f gate %.2f tracking_lin %.3f stand %.3f"
                          % (self._rew_log_step, float(total.mean()), lin_err,
                             speed, gait, float(comp["tracking_lin"].mean()), float(comp["stand"].mean())), flush=True)
        # MULTI-CRITIC (TAILI_MULTI_CRITIC): split the SAME reward components into K objective
        # groups and stash the [N,K] vector for the multi-critic agent. sum over K == the scalar
        # `total` (the env-level terminal penalty applied outside `comp` is routed to the "stab"
        # group via `extra_by_group`), so summing the vector reproduces the scalar reward exactly.
        # _get_observations surfaces this into `extras` AFTER it resets extras. Flag UNSET -> skipped.
        if os.environ.get("TAILI_MULTI_CRITIC") == "1":
            try:
                _mc_extra = {"stab": -cfg.w_terminal * terminal_window}
                self._reward_groups = torch.nan_to_num(
                    taili_reward.group_reward_vector(comp, extra_by_group=_mc_extra),
                    nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                self._reward_groups = None
        return torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
