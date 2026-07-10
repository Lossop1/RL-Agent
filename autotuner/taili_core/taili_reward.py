"""Production reward / gate -- strict per taili_strategy_decisions.md
(Reward/Gate  principles +  mapping + 3a gate/penalty + 3b weights/reduction +  impl details #1/#4/#8).

THE single reward source: the training env calls compute_reward_components; the
taili_reward_budget_backend wires the SAME function via its compute_components adapter
(no second formula). Reduction policy: ordinary penalties -> per-robot scalar (mean / fraction),
NOT raw sum (3b). exp-kernel tracking with sigma = acceptance band (#1). collapse penalties
give a recovery gradient in the gate-closed region. terminal != timeout (#4).

`inputs` / `cfg` follow the budget backend schema (RewardInputs / RewardBudgetConfig) so the
same function is testable by the budget suite. Pure torch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch


@dataclass
class RewardConfig:
    """Production reward config -- strategy-book starting values (3a thresholds + 3b weights).
    Same field names as the budget backend's RewardBudgetConfig so one function serves both."""
    # #1 tracking bandwidth = acceptance band
    sigma_lin_abs: float = 0.10
    sigma_lin_rel: float = 0.15
    sigma_yaw: float = 0.15
    sigma_stand_speed: float = 0.05
    sigma_stand_yaw: float = 0.05
    # stand-quality kernels. sigma_stand_action 0.15 made the action factor ~exp(-7)~0 at
    # walking-scale actions (||a||/sqrt(12) ~ 0.4) so the whole stand Gaussian was structurally 0 and
    # stand conditioning had no gradient beyond the far-kernel (telemetry: stand=0.000).
    sigma_stand_pose: float = 0.20
    sigma_stand_action: float = 0.30
    # Single absolute height source. h_ok/h_gate_close are derived by the
    # config loader and env override path from nominal_base_h:
    #   h_ok = nominal_base_h - 0.05, h_gate_close = nominal_base_h - 0.10.
    # Low/collapse height penalties are intentionally absent; height acts
    # only through stable_motion_gate and termination.
    nominal_base_h: float = 0.52
    h_ok: float = 0.47
    h_gate_close: float = 0.42
    tilt_ok_rad: float = 0.2617993878          # 15 deg
    tilt_gate_close_rad: float = 0.6981317008  # 40 deg
    # B1/B2/B3 / terrain
    flat_clearance_target: float = 0.08
    terrain_clearance_margin: float = 0.04
    clearance_band: float = 0.02
    terrain_v_floor: float = 0.10
    # 3b weight bands (starting; physeval-tuned)
    w_tracking_lin: float = 2.0
    w_tracking_yaw: float = 1.0
    w_stand: float = 1.5
    # A3/C four-foot planted: the `stand` term rewards body stillness only, so a policy can hold the
    # body still while shuffling feet in place (crawl) and still collect it — violating the spec's
    # four-foot duty >= 0.95. This term directly rewards all four feet in contact under a stand
    # command. stand-gated, so it never makes walking safer (no stand-still trap).
    w_stand_contact: float = 0.8
    w_terrain_progress: float = 0.5
    # Terrain-conditioned progress coupling: progress stays learnable through a floor, but full
    # terrain-progress credit requires low slip and, on easy ground, a regular gait. Rough/high
    # terrain relaxes the flat rhythm template and instead rewards progress with sufficient swing
    # clearance. This keeps one ability-level lever instead of a forest of posture/contact knobs.
    healthy_progress_floor: float = 0.35
    healthy_progress_slip_target: float = 0.12
    healthy_progress_slip_width: float = 0.18
    terrain_clearance_drive: float = 0.35
    # Tracking should mean usable locomotion, not just base velocity matching.
    # Keep a floor so early PPO still sees command-following gradients; the
    # remaining credit is paid only when existing contact/slip/clearance signals
    # say the motion is physically valid.
    validated_tracking_floor: float = 0.40
    tracking_validity_terrain_mix: float = 1.0
    w_torque_margin: float = 1.0
    w_torque_saturation: float = 5.0
    w_clearance_under: float = 0.4
    w_clearance_over: float = 0.05
    w_off_axis: float = 0.5
    w_action_rate: float = 0.02
    w_stance_slip: float = 0.05
    w_stance_slip_late: float = 0.20
    # RANK-2B (0707): penalise the p90 TAIL directly (fraction of settled feet slipping above the high
    # threshold), not just the mean window — B2 scores the p90, and an EMA mean is structurally blind to the
    # tail. Phased via quality_gate so it only bites once the base gait is up.
    w_stance_slip_high: float = 0.0
    w_stance_slip_high_late: float = 0.60
    # RANK-2B: landing-impact TAIL — the worst-foot touchdown vz (max over feet), which tracks B1's p95 far
    # better than the per-foot mean (a single hard-slamming foot fails B1 but barely moves the mean).
    w_landing_impact_tail: float = 0.0
    w_landing_impact_tail_late: float = 0.50
    # 脚前勾 forward-hook fix (0708): penalise HORIZONTAL foot velocity at the touchdown frame (the reward
    # blind spot that let the swing foot hook forward at plant in all directions). Mirrors the B1 vertical
    # landing penalty; quality-gated late. 0 = byte-identical no-op. late 0.60 = same band as slip_high_late.
    w_touchdown_slip: float = 0.0
    w_touchdown_slip_late: float = 0.60
    # GRADED slip shape: penalty is linear in settled-stance foot speed ABOVE slip_free_speed,
    # up to slip_speed_scale, then saturates. slip_free_speed is the measurement floor of this
    # proxy, NOT tolerated sliding: the foot-LINK world velocity of a physically slip-free gait
    # reads ~0.15 m/s (sphere rolling ~0.07 + load-transfer frames after the settled mask ~0.1 +
    # PhysX residual with velocity-iteration 0). Penalizing below the floor just taxed walking.
    # The old capped-quadratic (saturation at 0.125 m/s) additionally had ZERO gradient over the
    # skating regime — bounded value keeps invariant 4a; constant slope keeps a way out.
    slip_free_speed: float = 0.15
    slip_speed_scale: float = 1.0
    # B1 landing impact, same graded shape as slip: linear in touchdown vz above the spec
    # threshold (0.10) up to impact_speed_scale, then saturated. The old capped-quadratic
    # saturated at 0.40 m/s — real landings (0.5-1.5 m/s) sat in its zero-gradient region.
    impact_speed_scale: float = 1.0
    w_landing_impact: float = 0.15
    w_landing_impact_late: float = 0.85
    # flat-gait posture / rate quality (all graded & bounded):
    # orient  — tilt above ~5 deg costs linearly up to orient_scale (level body while walking)
    # base_vz — vertical bounce (quadratic, clamped)   base_wxy — roll/pitch rates (quadratic, clamped)
    # hip_deviation — hip abduction flare away from q_default (linear, scale 0.5 rad)
    w_orient: float = 0.30
    orient_soft_rad: float = 0.09
    orient_scale_rad: float = 0.35
    w_base_vz: float = 1.0
    w_base_wxy: float = 0.10
    w_hip_deviation: float = 0.40
    hip_deviation_scale: float = 0.50
    w_diagonal_contact: float = 0.20
    w_duty_balance: float = 0.20
    duty_target: float = 0.50
    quality_window_ema_beta: float = 0.98
    slip_high_threshold: float = 0.20
    # positive gait anchor: rewards matching the gait-clock contact schedule (shaping toward a
    # clean trot cannot make "not walking" safer, unlike the mismatch penalties it complements).
    w_gait_anchor: float = 1.0
    w_feet_air_time: float = 1.0
    air_time_target: float = 0.30
    # wrong-direction pressure: moving against the commanded linear direction / yaw sign. Missing
    # reward alone leaves a hardened forward attractor cost-free under backward/yaw commands.
    w_wrong_dir: float = 0.5
    rear_duty_tolerance: float = 0.05
    w_terminal: float = 10.0
    # anomaly #2 fix -- FAR-FIELD tracking/stand shaping (Laplacian exp(-err/scale), gradient
    # everywhere) complementing the sigma=acceptance-band Gaussians (which are gradient-starved beyond
    # ~2 sigma). Pulls the policy toward the target from far off; the tight Gaussian does final precision.
    w_track_far: float = 1.0
    w_yaw_far: float = 0.5
    w_stand_far: float = 0.8
    scale_track_far: float = 0.35
    scale_stand_far: float = 0.3


def default_reward_cfg() -> RewardConfig:
    try:
        try:
            from ..blind_locomotion.taili_blind_config import reward_config_mapping
        except Exception:
            from ..taili_blind_config import reward_config_mapping
        values = reward_config_mapping()
        valid = {field.name for field in fields(RewardConfig)}
        return RewardConfig(**{k: v for k, v in values.items() if k in valid})
    except Exception:
        return RewardConfig()


def reward_cfg_from_env(base: "RewardConfig | None" = None) -> RewardConfig:
    """RewardConfig with any numeric field overridable via env var TAILI_RW_<FIELD> (UPPER).
    Lets reward-weight tuning iterate per training run WITHOUT a code change + redeploy -- e.g.
    `TAILI_RW_W_TRACK_FAR=2.0 python tp_train.py`. Unset vars keep the dataclass default."""
    import dataclasses
    import os
    cfg = base if base is not None else default_reward_cfg()
    applied = {}
    for f in dataclasses.fields(cfg):
        env = os.environ.get("TAILI_RW_" + f.name.upper())
        if env is None:
            continue
        cur = getattr(cfg, f.name)
        try:
            setattr(cfg, f.name, type(cur)(env) if isinstance(cur, (int, float)) and not isinstance(cur, bool) else cur)
            applied[f.name] = env
        except (ValueError, TypeError):
            pass
    if applied:
        try:
            print(f"[taili_reward] env overrides: {applied}", flush=True)
        except Exception:
            pass
    # Keep height single-source even when TAILI_RW_* overrides are used.
    cfg.h_ok = float(cfg.nominal_base_h) - 0.05
    cfg.h_gate_close = float(cfg.nominal_base_h) - 0.10
    return cfg


def exp_kernel(err, sigma):
    sigma = torch.clamp(torch.as_tensor(sigma, dtype=err.dtype, device=err.device), min=1e-9)
    return torch.exp(-((err / sigma) ** 2))


def far_kernel(err, scale):
    """Laplacian shaping kernel exp(-err/scale): bounded [0,1], strictly decreasing, NONZERO gradient
    for all err (unlike the Gaussian exp_kernel, whose gradient vanishes beyond ~2 sigma). Used as the
    far-field companion so the policy can climb toward the target from a poor initialization."""
    scale = torch.clamp(torch.as_tensor(scale, dtype=err.dtype, device=err.device), min=1e-9)
    return torch.exp(-err.abs() / scale)


def _smoothstep(edge0, edge1, x):
    t = torch.clamp((x - edge0) / (edge1 - edge0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _metric_gate(value, target, *, higher=True, width=0.10):
    """Soft [0, 1] gate used to couple progress to gait health without killing gradients."""
    target_t = torch.as_tensor(target, dtype=value.dtype, device=value.device)
    width_t = torch.as_tensor(width, dtype=value.dtype, device=value.device).clamp_min(1e-6)
    if higher:
        return _smoothstep(target_t - width_t, target_t, value)
    return 1.0 - _smoothstep(target_t, target_t + width_t, value)


def stable_motion_gate(base_h_above_terrain, tilt_rel, support_instability,
                       severe_body_collision, terminal_window, cfg, alpha=(1.0, 1.0, 1.0)):
    """3a two-regime gate: soft product in the gray zone, hard 0 at collapse/terminal.
    Provided here for the ENV to compute the shared gate; the reward uses the provided value."""
    g_height = _smoothstep(cfg.h_gate_close, cfg.h_ok, base_h_above_terrain)
    g_tilt = 1.0 - _smoothstep(cfg.tilt_ok_rad, cfg.tilt_gate_close_rad, tilt_rel)
    g_support = torch.clamp(1.0 - support_instability, 0.0, 1.0)
    g_soft = (g_height ** alpha[0]) * (g_tilt ** alpha[1]) * (g_support ** alpha[2])
    hard = (base_h_above_terrain < cfg.h_gate_close) | (tilt_rel > cfg.tilt_gate_close_rad) \
        | (severe_body_collision > 0.5) | (terminal_window > 0.5)
    return torch.where(hard, torch.zeros_like(g_soft), torch.clamp(g_soft, 0.0, 1.0))


def compute_reward_components(inp, cfg):
    """Return dict of per-robot-scalar reward components (+ gate diagnostics + total)."""
    comp = {}
    f = torch.float64 if inp.cmd.dtype == torch.float64 else torch.float32

    gate = inp.stable_motion_gate
    stand_gate, moving_gate = inp.stand_gate, inp.moving_gate
    quality_gate = getattr(inp, "quality_gate", 1.0)
    quality_gate = torch.as_tensor(quality_gate, dtype=gate.dtype, device=gate.device)
    if quality_gate.ndim == 0:
        quality_gate = quality_gate.expand_as(gate)
    comp["stable_motion_gate"] = gate
    comp["stand_gate"] = stand_gate
    comp["moving_gate"] = moving_gate
    comp["quality_gate"] = quality_gate

    cmd = inp.cmd
    vel = inp.base_lin_vel
    cmd_xy = cmd[:, :2]
    cmd_xy_norm = torch.linalg.norm(cmd_xy, dim=-1)
    foot_contact = getattr(inp, "foot_contact", None)
    if foot_contact is not None:
        foot_contact = foot_contact.to(dtype=f, device=gate.device)
        contact_count = foot_contact.sum(dim=-1)
    else:
        contact_count = torch.full_like(gate, 2.0)

    # A1/A2 raw command tracking signals. They are paid below after terrain/contact
    # validity is known, so "tracking" means usable locomotion rather than mere base
    # velocity matching during slip, foot drag, or collision shortcuts.
    sigma_v = torch.maximum(torch.full_like(cmd_xy_norm, cfg.sigma_lin_abs), cfg.sigma_lin_rel * cmd_xy_norm)
    lin_err = torch.linalg.norm(vel[:, :2] - cmd_xy, dim=-1)
    tracking_lin_raw = exp_kernel(lin_err, sigma_v)
    yaw_err = (inp.base_ang_vel[:, 2] - cmd[:, 2]).abs()
    yaw_cmd_gate = (cmd[:, 2].abs() > 0.05).to(f)
    tracking_yaw_raw = exp_kernel(yaw_err, cfg.sigma_yaw)

    # A3/C stand -- stillness + default-pose + low-action, stand-gated
    speed = torch.linalg.norm(vel[:, :2], dim=-1)
    yaw_rate = inp.base_ang_vel[:, 2].abs()
    action_norm = torch.linalg.norm(inp.action, dim=-1) / (inp.action.shape[-1] ** 0.5)
    stand = (exp_kernel(speed, cfg.sigma_stand_speed) * exp_kernel(yaw_rate, cfg.sigma_stand_yaw)
             * exp_kernel(inp.default_pose_error.abs(), getattr(cfg, "sigma_stand_pose", 0.20))
             * exp_kernel(action_norm, getattr(cfg, "sigma_stand_action", 0.30)))
    comp["stand"] = cfg.w_stand * stand * stand_gate * gate

    # A3/C FOUR-FOOT PLANTED — reward all four feet in contact under a stand command. The `stand`
    # Gaussian above scores body stillness only; without this a "still body + feet shuffling in
    # place" gait scores well yet fails the spec four-foot duty >= 0.95. stand-gated (no walking cost).
    if foot_contact is not None:
        four_foot = foot_contact.mean(dim=-1)
        comp["stand_contact"] = getattr(cfg, "w_stand_contact", 0.0) * four_foot * stand_gate * gate
    else:
        comp["stand_contact"] = torch.zeros_like(gate)

    # anomaly #2 fix -- FAR-FIELD shaping (gradient everywhere) so the policy can climb toward the
    # command from far off, where the sigma=acceptance-band Gaussians above give ~0 gradient. Same gates
    # as their Gaussian partners -> off in collapse, no new degenerate optimum; bounded by the weights.
    s_far = getattr(cfg, "scale_track_far", 0.5)
    s_stand_far = getattr(cfg, "scale_stand_far", 0.3)
    tracking_lin_far_raw = far_kernel(lin_err, s_far)
    tracking_yaw_far_raw = far_kernel(yaw_err, s_far)
    comp["stand_far"] = (getattr(cfg, "w_stand_far", 0.6) * far_kernel(speed, s_stand_far)
                         * far_kernel(yaw_rate, s_stand_far) * stand_gate * gate)

    # D terrain stable_progress -- linear-command direction only.
    # Contract: raw progress is dot(v_xy, cmd_xy) / |cmd_xy|.  Pure yaw has no
    # linear terrain progress owner; yaw tracking owns A2.  The previous x-only
    # proxy rewarded forward drift during lateral/yaw commands, exactly the
    # behavior the direction diagnostics exposed.
    cmd_xy_norm_safe = torch.clamp(cmd_xy_norm, min=1e-6)
    v_along = torch.sum(vel[:, :2] * cmd_xy, dim=-1) / cmd_xy_norm_safe
    has_linear_cmd = cmd_xy_norm > 0.05
    ratio = torch.clamp(v_along / cmd_xy_norm_safe, 0.0, 1.0)
    floor_ok = ((v_along >= cfg.terrain_v_floor) & has_linear_cmd).to(f)
    swing_for_progress = (inp.foot_contact <= 0.5).to(f)
    target_for_progress = torch.where(
        inp.local_obstacle_h[:, None] > 0,
        inp.local_obstacle_h[:, None] + cfg.terrain_clearance_margin,
        torch.full_like(inp.foot_clearance, cfg.flat_clearance_target),
    )
    under_for_progress = torch.clamp(
        (target_for_progress - inp.foot_clearance - cfg.clearance_band) / 0.05,
        min=0.0,
    )
    clearance_success = torch.clamp(1.0 - under_for_progress, 0.0, 1.0)
    clearance_success = (
        clearance_success * swing_for_progress
    ).sum(dim=-1) / swing_for_progress.sum(dim=-1).clamp(min=1.0)
    terrain_difficulty = torch.clamp(inp.local_obstacle_h / 0.25, 0.0, 1.0)

    diag_window = getattr(inp, "diagonal_pair_window", None)
    duty_window = getattr(inp, "duty_quality_window", None)
    slip_high_window = getattr(inp, "stance_slip_high_fraction", None)
    diag_score = diag_window.to(dtype=f, device=gate.device) if diag_window is not None else torch.full_like(gate, 0.70)
    duty_score = duty_window.to(dtype=f, device=gate.device) if duty_window is not None else torch.full_like(gate, 0.70)
    slip_tail = slip_high_window.to(dtype=f, device=gate.device) if slip_high_window is not None else torch.zeros_like(gate)
    slip_gate = _metric_gate(
        slip_tail,
        float(getattr(cfg, "healthy_progress_slip_target", 0.12)),
        higher=False,
        width=float(getattr(cfg, "healthy_progress_slip_width", 0.18)),
    )
    rhythm_gate = 0.5 * (
        _metric_gate(diag_score, 0.52, higher=True, width=0.18)
        + _metric_gate(duty_score, 0.58, higher=True, width=0.18)
    )
    terrain_mix = torch.clamp(
        terrain_difficulty * float(getattr(cfg, "tracking_validity_terrain_mix", 1.0)),
        0.0,
        1.0,
    )
    valid_contact = slip_gate * (
        (1.0 - terrain_mix) * rhythm_gate + terrain_mix * clearance_success
    )
    healthy_floor = float(getattr(cfg, "healthy_progress_floor", 0.35))
    healthy_progress_gate = torch.clamp(healthy_floor + (1.0 - healthy_floor) * valid_contact, 0.0, 1.0)
    tracking_floor = float(getattr(cfg, "validated_tracking_floor", healthy_floor))
    validated_tracking_gate = torch.clamp(tracking_floor + (1.0 - tracking_floor) * valid_contact, 0.0, 1.0)
    terrain_drive = 1.0 + float(getattr(cfg, "terrain_clearance_drive", 0.35)) * terrain_difficulty * clearance_success
    comp["healthy_progress_gate"] = healthy_progress_gate
    comp["validated_tracking_gate"] = validated_tracking_gate
    comp["tracking_lin"] = cfg.w_tracking_lin * tracking_lin_raw * moving_gate * gate * validated_tracking_gate
    comp["tracking_yaw"] = cfg.w_tracking_yaw * tracking_yaw_raw * moving_gate * gate * yaw_cmd_gate * validated_tracking_gate
    comp["tracking_lin_far"] = getattr(cfg, "w_track_far", 0.6) * tracking_lin_far_raw * moving_gate * gate * validated_tracking_gate
    comp["tracking_yaw_far"] = getattr(cfg, "w_yaw_far", 0.3) * tracking_yaw_far_raw * moving_gate * gate * yaw_cmd_gate * validated_tracking_gate
    comp["terrain_progress"] = cfg.w_terrain_progress * ratio * floor_ok * gate * healthy_progress_gate * terrain_drive

    # wrong-direction pressure (#8 extension): moving AGAINST the commanded linear direction or
    # yaw sign is actively penalized. Tracking/progress only withhold reward for it, which leaves
    # a hardened forward attractor cost-free under backward/yaw commands.
    w_wd = getattr(cfg, "w_wrong_dir", 0.0)
    lin_wrong = torch.clamp(-v_along, min=0.0, max=1.5) * has_linear_cmd.to(f)
    has_yaw_cmd = cmd[:, 2].abs() > 0.05
    yaw_wrong = torch.clamp(-(inp.base_ang_vel[:, 2] * torch.sign(cmd[:, 2])), min=0.0, max=1.5) * has_yaw_cmd.to(f)
    comp["wrong_dir"] = -w_wd * (lin_wrong ** 2 + yaw_wrong ** 2) * moving_gate * gate

    # off-axis command-aware residual (#8): penalize only near-zero-command axes.
    # NOT moving-gated: under a stand command ALL axes are near-zero, so this same term is the
    # anti-drift pressure for the quiet stand (A3/C). With the old moving_gate factor it was OFF
    # exactly when standing — drift cost nothing beyond forgone stand reward, and the stand
    # Gaussian (sigma 0.05) is gradient-starved at the observed ~0.1 m/s drift.
    eps = 0.05
    off = torch.zeros_like(speed)
    off = off + torch.where(cmd[:, 0].abs() <= eps, vel[:, 0].abs(), torch.zeros_like(off))
    off = off + torch.where(cmd[:, 1].abs() <= eps, vel[:, 1].abs(), torch.zeros_like(off))
    off = off + torch.where(cmd[:, 2].abs() <= eps, inp.base_ang_vel[:, 2].abs(), torch.zeros_like(off))
    comp["off_axis"] = -cfg.w_off_axis * (off ** 2) * gate

    # B3 clearance band -- per-foot(swing) MEAN (reduction policy: not raw sum)
    swing = (inp.foot_contact <= 0.5).to(f)
    target = torch.where(inp.local_obstacle_h[:, None] > 0,
                         inp.local_obstacle_h[:, None] + cfg.terrain_clearance_margin,
                         torch.full_like(inp.foot_clearance, cfg.flat_clearance_target))
    under = torch.clamp((target - inp.foot_clearance - cfg.clearance_band) / 0.05, min=0.0)
    over = torch.clamp((inp.foot_clearance - target - cfg.clearance_band) / 0.10, min=0.0)
    comp["clearance_under"] = -cfg.w_clearance_under * (under ** 2 * swing).mean(dim=-1)
    comp["clearance_over"] = -cfg.w_clearance_over * (over ** 2 * swing).mean(dim=-1)

    # B4 gait structure -- moving-only quality terms.  Tracking alone can be
    # satisfied by crawl/drag shortcuts; these terms make the diagonal trot
    # contact schedule, front/rear duty balance and L/R balance visible to PPO.
    stance = (inp.foot_contact > 0.5).to(f)
    desired = getattr(inp, "desired_foot_contact", None)
    if desired is not None:
        desired = desired.to(dtype=f, device=stance.device)
        gait_mismatch = (stance - desired).abs().mean(dim=-1)
        gait_match = 1.0 - gait_mismatch
        # positive gait anchor -- NOT quality-gate ramped: rewarding the clock-consistent contact
        # schedule from the start shapes toward a clean trot without ever making "not walking"
        # the safer policy (the failure mode of pure mismatch penalties).
        comp["gait_anchor"] = (
            getattr(cfg, "w_gait_anchor", 0.0)
            * torch.clamp(2.0 * gait_match - 1.0, min=0.0)
            * moving_gate
            * gate
        )
    else:
        comp["gait_anchor"] = torch.zeros_like(gate)
    diagonal_pair_window = getattr(inp, "diagonal_pair_window", None)
    duty_quality_window = getattr(inp, "duty_quality_window", None)
    if diagonal_pair_window is not None and duty_quality_window is not None:
        diagonal_pair_window = diagonal_pair_window.to(dtype=f, device=gate.device)
        duty_quality_window = duty_quality_window.to(dtype=f, device=gate.device)
        comp["diagonal_contact"] = (
            -getattr(cfg, "w_diagonal_contact", 0.0)
            * torch.clamp(1.0 - diagonal_pair_window, 0.0, 1.0)
            * moving_gate
            * gate
            * quality_gate
        )
        comp["duty_balance"] = (
            -getattr(cfg, "w_duty_balance", 0.0)
            * (torch.clamp(1.0 - duty_quality_window, 0.0, 1.0) ** 2)
            * moving_gate
            * gate
            * quality_gate
        )
    elif stance.shape[-1] >= 4:
        fl, fr, rl, rr = stance[:, 0], stance[:, 1], stance[:, 2], stance[:, 3]
        diagonal_mismatch = 0.5 * ((fl - rr).abs() + (fr - rl).abs())
        front_duty = 0.5 * (fl + fr)
        rear_duty = 0.5 * (rl + rr)
        left_duty = 0.5 * (fl + rl)
        right_duty = 0.5 * (fr + rr)
        rear_bias = torch.clamp(rear_duty - front_duty - getattr(cfg, "rear_duty_tolerance", 0.05), min=0.0)
        lr_bias = (left_duty - right_duty).abs()
        comp["diagonal_contact"] = -getattr(cfg, "w_diagonal_contact", 0.0) * diagonal_mismatch * moving_gate * gate * quality_gate
        comp["duty_balance"] = -getattr(cfg, "w_duty_balance", 0.0) * (rear_bias ** 2 + lr_bias ** 2) * moving_gate * gate * quality_gate
    else:
        comp["diagonal_contact"] = torch.zeros_like(gate)
        comp["duty_balance"] = torch.zeros_like(gate)

    # B2 stance slip — GRADED, bounded. Linear in settled-stance foot speed up to
    # slip_speed_scale (1 m/s), then saturated: constant gradient over the whole skating
    # regime. The previous capped-quadratic saturated at 0.125 m/s, so a skating gait
    # (settled feet at 0.3-1.0 m/s) sat entirely in a zero-gradient region and PPO had no
    # local signal to slide less (observed: slip pinned at ~0.55 from 2k steps while every
    # other metric improved). The env-provided window is already settled- and moving-masked;
    # do not multiply by the current frame's gate or a brief collapse/stand frame hides
    # recent sliding.
    slip_w = getattr(cfg, "w_stance_slip", 0.05) + quality_gate * getattr(cfg, "w_stance_slip_late", 0.0)
    slip_scale = max(float(getattr(cfg, "slip_speed_scale", 1.0)), 1e-6)
    slip_free = float(getattr(cfg, "slip_free_speed", 0.15))
    speed_window = getattr(inp, "stance_slip_speed_window", None)
    if speed_window is not None:
        speed_window = speed_window.to(dtype=f, device=gate.device)
        comp["stance_slip"] = -slip_w * torch.clamp((speed_window - slip_free) / slip_scale, 0.0, 1.0)
    else:
        slip_stance = getattr(inp, "stance_slip_contact", stance)
        slip_stance = slip_stance.to(dtype=f, device=stance.device)
        denom = slip_stance.sum(dim=-1).clamp(min=1.0)
        speed_mean = (inp.foot_vel_xy * slip_stance).sum(dim=-1) / denom
        comp["stance_slip"] = -slip_w * torch.clamp((speed_mean - slip_free) / slip_scale, 0.0, 1.0)
    # RANK-2B (0707): p90-TAIL slip penalty on top of the mean window — the fraction of settled feet slipping
    # above the high threshold is what B2 (p90 ≤ 0.05) actually scores; the mean window is blind to that tail.
    # Config-gated (default 0 → byte-identical to before); quality_gate-phased so it bites once gait is up.
    slip_high = getattr(inp, "stance_slip_high_fraction", None)
    if slip_high is not None:
        slip_high = slip_high.to(dtype=f, device=gate.device)
        slip_high_w = getattr(cfg, "w_stance_slip_high", 0.0) + quality_gate * getattr(cfg, "w_stance_slip_high_late", 0.0)
        comp["stance_slip"] = comp["stance_slip"] - slip_high_w * slip_high

    # feet air time — ANTI-SHUFFLE, clock-consistent. The target is the gait clock's scheduled
    # swing duration (per-env: period*(1-duty)); a fixed 0.30 s fought the clock, whose swing is
    # only 0.20-0.264 s (the reward pulled for longer-than-scheduled swings, so the policy could
    # never lock the clock and gait_match capped ~0.79). Credit is clamped at 0: short swings
    # (shuffle) are penalized, over-long swings earn nothing (the clock, via gait_anchor, owns the
    # upper bound) — the two rhythm terms now AGREE, so a clock-locked trot is the joint optimum.
    td_mask_air = getattr(inp, "touchdown_mask", None)
    foot_air_time = getattr(inp, "foot_air_time", None)
    if td_mask_air is not None and foot_air_time is not None:
        td_mask_air = td_mask_air.to(dtype=f, device=gate.device)
        foot_air_time = foot_air_time.to(dtype=f, device=gate.device)
        air_target = getattr(inp, "air_time_target", None)
        if air_target is None:
            air_target = getattr(cfg, "air_time_target", 0.30)
        else:
            air_target = air_target.to(dtype=f, device=gate.device)
            if air_target.ndim == 1:
                air_target = air_target[:, None]
        air_short = torch.clamp(foot_air_time - air_target, max=0.0)   # <=0 (only shuffle penalized)
        comp["feet_air_time"] = getattr(cfg, "w_feet_air_time", 0.0) * (
            air_short * td_mask_air).sum(dim=-1) * moving_gate * gate
    else:
        comp["feet_air_time"] = torch.zeros_like(gate)

    # B1 landing impact -- per-foot MEAN. The book's B1 is a TOUCHDOWN event (spec: touchdown vz p95).
    # If the env supplies `touchdown_mask` (1 only on the swing->stance transition this step), gate the
    # penalty to ACTUAL landings -- so it shapes the landing velocity WITHOUT punishing the swing
    # vertical motion walking requires (the every-step |foot_vz| proxy did the latter -> stand-still trap, anomaly
    # #5). No mask supplied -> legacy every-step proxy (byte-identical, so in-flight runs are unaffected).
    impact_scale = max(float(getattr(cfg, "impact_speed_scale", 1.0)), 1e-6)
    impact_over = torch.clamp((inp.touchdown_vz.abs() - 0.10) / impact_scale, 0.0, 1.0)
    td_mask = getattr(inp, "touchdown_mask", None)
    impact_w = getattr(cfg, "w_landing_impact", 1.0) + quality_gate * getattr(cfg, "w_landing_impact_late", 0.0)
    if td_mask is not None:
        comp["landing_impact"] = -impact_w * (impact_over * td_mask).mean(dim=-1)
    else:
        comp["landing_impact"] = -impact_w * impact_over.mean(dim=-1)
    # RANK-2B (0707): worst-foot landing TAIL (max over feet) — B1 scores the p95 touchdown vz; one hard-slamming
    # foot fails B1 while barely moving the per-foot mean above. Config-gated (default 0 → byte-identical).
    if getattr(cfg, "w_landing_impact_tail", 0.0) != 0.0 or getattr(cfg, "w_landing_impact_tail_late", 0.0) != 0.0:
        tail_w = getattr(cfg, "w_landing_impact_tail", 0.0) + quality_gate * getattr(cfg, "w_landing_impact_tail_late", 0.0)
        worst = (impact_over * td_mask).max(dim=-1).values if td_mask is not None else impact_over.max(dim=-1).values
        comp["landing_impact"] = comp["landing_impact"] - tail_w * worst

    # TOUCHDOWN-SLIP (0708, "脚前勾" forward-hook fix) — MIRRORS the B1 vertical landing penalty but on the
    # HORIZONTAL foot velocity at the touchdown frame. The swing foot arrives at contact still carrying forward
    # (and lateral) swing velocity, and NO term scored it: B2 stance-slip reads only SETTLED contact (excludes the
    # touchdown frame by construction, blind_tp_env settled_contact), B1 penalizes only the VERTICAL touchdown_vz,
    # and the position-only imitation is turn-gated (weak for forward). So the foot HOOKS forward at plant in ALL
    # directions (td_mask fires on every landing). This penalises foot_vel_xy (planar contact-point speed) ABOVE a
    # 0.05 m/s free floor on the landing frame -> pulls the foot onto the reference's world-stationary plant.
    # Foot-world-velocity only, never base velocity -> propulsion/cruise speed untouched (no A1-collapse mechanism).
    ts_w = getattr(cfg, "w_touchdown_slip", 0.0) + quality_gate * getattr(cfg, "w_touchdown_slip_late", 0.0)
    if (getattr(cfg, "w_touchdown_slip", 0.0) != 0.0 or getattr(cfg, "w_touchdown_slip_late", 0.0) != 0.0):
        ts_over = torch.clamp((inp.foot_vel_xy - 0.05) / impact_scale, 0.0, 1.0)      # per-foot planar over-floor
        comp["touchdown_slip"] = -ts_w * ((ts_over * td_mask).mean(dim=-1) if td_mask is not None
                                          else ts_over.mean(dim=-1))

    # flat-gait posture / rate quality — level body, no bounce, no roll/pitch flutter, hips
    # tucked at q_default. All graded & bounded; gate keeps them from piling onto a collapse.
    tilt_over = torch.clamp(
        (inp.tilt_rel - getattr(cfg, "orient_soft_rad", 0.09)) / max(getattr(cfg, "orient_scale_rad", 0.35), 1e-6),
        0.0, 1.0)
    comp["orient"] = -getattr(cfg, "w_orient", 0.0) * tilt_over * gate
    # gate these like the sibling orient term (the block comment's stated invariant): on a collapse
    # the gate hard-zeros, so we neither pile a bounce/roll penalty onto the -terminal nor tax the
    # upward vz the robot needs to recover in the gate-closed band. gate~1 during normal gait, so
    # steady-state shaping is unchanged.
    comp["base_vz"] = -getattr(cfg, "w_base_vz", 0.0) * torch.clamp(vel[:, 2] ** 2, max=1.0) * gate
    wxy_sq = inp.base_ang_vel[:, 0] ** 2 + inp.base_ang_vel[:, 1] ** 2
    comp["base_wxy"] = -getattr(cfg, "w_base_wxy", 0.0) * torch.clamp(wxy_sq, max=4.0) * gate
    # hip deviation — COMMAND-GATED. The AMP reference keeps hip abduction at q_default (0) ONLY
    # for forward/back/stand; lateral & yaw legitimately need hip abduction (reference sdy != 0),
    # so penalizing |hip| there would fight the commanded motion. Apply it only when no lateral
    # and no yaw is commanded — exactly the case where any hip swing is off-style balance flailing.
    hip_dev = getattr(inp, "hip_deviation", None)
    if hip_dev is not None:
        hip_dev = hip_dev.to(dtype=f, device=gate.device)
        no_lat_yaw = ((cmd[:, 1].abs() <= 0.05) & (cmd[:, 2].abs() <= 0.05)).to(f)
        comp["hip_deviation"] = -getattr(cfg, "w_hip_deviation", 0.0) * torch.clamp(
            hip_dev / max(getattr(cfg, "hip_deviation_scale", 0.5), 1e-6), 0.0, 1.0) * no_lat_yaw * gate
    else:
        comp["hip_deviation"] = torch.zeros_like(gate)

    # F2 torque -- margin per-joint MEAN ; saturation = fraction-of-joints (event-class, 4b)
    limits = inp.torque_limit
    if limits.ndim == 1:
        limits = limits.unsqueeze(0).expand_as(inp.torque)
    util = inp.torque.abs() / torch.clamp(limits, min=1e-6)
    margin_excess = torch.clamp((util - 0.85) / 0.15, min=0.0)
    comp["torque_margin"] = -cfg.w_torque_margin * (margin_excess ** 2).mean(dim=-1)
    sat = ((util >= 1.0) | (inp.torque_clamped > 0.5)).to(f)
    comp["torque_saturation"] = -cfg.w_torque_saturation * sat.mean(dim=-1)

    # B5/F3 smoothness -- normalized action rate (per-robot)
    comp["action_rate"] = -cfg.w_action_rate * ((inp.action - inp.last_action) ** 2).mean(dim=-1)

    # #4 terminal (true terminal only; timeout is NOT penalized -- env passes terminal_reason)
    is_terminal = 1.0 if (inp.terminal_reason is not None and inp.terminal_reason != "timeout") else 0.0
    comp["terminal_penalty"] = -cfg.w_terminal * torch.full_like(gate, is_terminal)

    total = torch.zeros_like(gate)
    for k, v in comp.items():
        if k not in (
            "stable_motion_gate",
            "stand_gate",
            "moving_gate",
            "quality_gate",
            "healthy_progress_gate",
            "validated_tracking_gate",
        ):
            total = total + v
    comp["total"] = total
    return comp


# ---------------------------------------------------------------------------
# MULTI-CRITIC reward-group split (TAILI_MULTI_CRITIC).
# ---------------------------------------------------------------------------
# Objective grouping of the scalar reward components into K vector heads so a
# multi-critic PPO can run GAE per objective. The split is a partition: every
# non-gate, non-total component maps to exactly one group and any UNKNOWN key
# (e.g. an env-added component like "stumble"/"climb") falls into the fallback
# group, so `group_reward_vector(comp).sum(dim=-1)` is IDENTICALLY the scalar
# `comp["total"]` (plus whatever env-level extras the caller injects). That
# identity is what makes the flag-OFF path — summing the vector — byte-equal to
# stock scalar training.
# TURN is its OWN group (0707): the multi-critic z-scores each GROUP's advantage then sums, so a group
# with a large-magnitude/easy member drowns its smaller members. YAW (anti-symmetric, harder, weight 1.0)
# was grouped WITH linear tracking (weight 2.0, easier, larger) in "track" — so MC gave yaw NO independent
# voice and the policy let yaw regress (tracking_yaw peaked ~14k then declined) while optimizing linear+gait.
# Splitting yaw into its own "turn" group gives it an independently-normalized advantage — the whole point
# of multi-critic for the yaw-vs-linear conflict.
REWARD_GROUP_NAMES = ("track", "turn", "gait", "stab", "style")
REWARD_GROUP_GATES = ("stable_motion_gate", "stand_gate", "moving_gate", "quality_gate", "healthy_progress_gate", "total")
REWARD_GROUP_FALLBACK = "stab"
_REWARD_GROUP_OF = {
    # task / command tracking (positive task reward)
    "tracking_lin": "track",
    "tracking_yaw": "turn", "tracking_yaw_far": "turn",   # YAW gets its own normalized advantage
    "tracking_lin_far": "track",
    "terrain_progress": "track", "stand": "track", "stand_contact": "track",
    "stand_far": "track", "climb": "track", "terrain_up": "track",
    "terrain_down": "track", "feet_air_time": "track",
    # gait quality (rhythm / contact schedule / slip / clearance band)
    "gait_anchor": "gait", "diagonal_contact": "gait", "duty_balance": "gait",
    "stance_slip": "gait", "clearance_under": "gait", "clearance_over": "gait",
    # stability / posture / safety
    "orient": "stab", "base_vz": "stab", "base_wxy": "stab", "off_axis": "stab",
    "wrong_dir": "stab", "lateral_underspeed": "stab", "hip_deviation": "stab", "landing_impact": "stab",
    "stumble": "stab", "terminal_penalty": "stab",
    # style / effort / smoothness penalties
    "torque_margin": "style", "torque_saturation": "style", "action_rate": "style",
}


def num_reward_groups() -> int:
    return len(REWARD_GROUP_NAMES)


def group_reward_vector(comp, extra_by_group=None):
    """Split a reward-component dict into a [N, K] group vector (TAILI_MULTI_CRITIC).

    Gate diagnostics and the aggregate "total" are excluded; every remaining
    component is summed into its objective column (unknown keys -> fallback
    group). `extra_by_group` adds env-level per-group terms (each an [N] tensor),
    e.g. the terminal penalty the env applies outside `comp`. The invariant is
    `group_reward_vector(comp, extra).sum(dim=-1) == comp["total"] + sum(extra)`.
    """
    names = REWARD_GROUP_NAMES
    col = {g: i for i, g in enumerate(names)}
    ref = comp.get("total")
    if ref is None:
        for v in comp.values():
            if torch.is_tensor(v):
                ref = v
                break
    n = ref.shape[0]
    vec = torch.zeros(n, len(names), dtype=ref.dtype, device=ref.device)
    for k, v in comp.items():
        if k in REWARD_GROUP_GATES or not torch.is_tensor(v):
            continue
        g = _REWARD_GROUP_OF.get(k, REWARD_GROUP_FALLBACK)
        vec[:, col[g]] = vec[:, col[g]] + v.to(dtype=ref.dtype)
    if extra_by_group:
        for g, v in extra_by_group.items():
            if v is None:
                continue
            vec[:, col[g]] = vec[:, col[g]] + torch.as_tensor(v, dtype=ref.dtype, device=ref.device)
    return vec
