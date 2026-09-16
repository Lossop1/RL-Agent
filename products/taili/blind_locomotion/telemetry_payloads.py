"""训练遥测 payload 构造器。

这些函数只把 `_get_rewards` 已经算出的张量和环境状态整理成外部遥测契约。
它们不改变训练状态、不参与奖励计算；字段名需要保持稳定，供前端、智能体和人工调参读取。
"""
from __future__ import annotations

import time
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

import torch


def _float_attr(obj: Any, name: str, default: float = 0.0) -> float:
    return float(getattr(obj, name, default))


def _command_bucket_masks(commands: torch.Tensor, *, deadband: float = 0.05) -> dict[str, torch.Tensor]:
    """Return disjoint command buckets for the realized batch.

    Direction audit fields are axis-oriented and intentionally overlap for a
    mixed command. Coverage accounting needs a second, disjoint view so a
    batch cannot be mistaken for six command regimes from one mean vector.
    """
    active = commands.abs() > deadband
    active_count = active.sum(dim=-1)
    single = active_count == 1
    return {
        "stand": active_count == 0,
        "mixed": active_count > 1,
        "forward": single & active[:, 0] & (commands[:, 0] >= 0.0),
        "backward": single & active[:, 0] & (commands[:, 0] < 0.0),
        "lateral": single & active[:, 1],
        "yaw": single & active[:, 2],
    }


def _reward_cfg_payload(cfg: Any) -> dict[str, float]:
    try:
        payload = (
            asdict(cfg)
            if is_dataclass(cfg)
            else {
                name: float(getattr(cfg, name))
                for name in dir(cfg)
                if not name.startswith("_") and isinstance(getattr(cfg, name), (int, float))
            }
        )
    except Exception:
        payload = {}
    return {str(key): float(value) for key, value in payload.items() if isinstance(value, (int, float))}


def _lagging_progress(env: Any, progress: Mapping[str, float], active_dirs) -> tuple[str, float]:
    """按每个方向自己的阶段门槛查找真实阻塞方向。"""
    names = tuple(str(name) for name in (active_dirs or progress.keys()) if str(name) in progress)
    if not names:
        return "", 0.0
    cfg = getattr(env, "cfg", None)
    phase = int(getattr(env, "_phase", getattr(cfg, "init_phase", 0))) if cfg is not None else 0
    default = float(getattr(cfg, f"phase_gate_prog_{phase}", 1.0)) if cfg is not None else 1.0
    thresholds = getattr(cfg, f"phase_progress_thresholds_{phase}", {}) if cfg is not None else {}
    if not isinstance(thresholds, Mapping):
        thresholds = {}
    ratios = {
        name: float(progress[name]) / max(float(thresholds.get(name, default)), 1e-6)
        for name in names
    }
    lagging = min(names, key=lambda name: ratios[name])
    return lagging, ratios[lagging]


def build_reward_payload(
    *,
    total: torch.Tensor,
    lin_err: float,
    speed: float,
    gait: float,
    base_h: torch.Tensor,
    upright: float,
    comp: Mapping[str, Any],
    terrain_probe: Mapping[str, Any],
    reward_cfg: Any,
    include_reward_cfg: bool,
) -> dict[str, float]:
    reward_payload = {
        "total": float(total.mean()),
        "lin_err": lin_err,
        "speed": speed,
        "gait": gait,
        "base_h": float(base_h.mean()),
        "upright": upright,
    }
    for name, value in comp.items():
        if name == "total":
            continue
        if torch.is_tensor(value) and value.numel() > 0:
            reward_payload[name] = float(value.mean())
    for name, value in terrain_probe.items():
        if torch.is_tensor(value) and value.numel() > 0:
            reward_payload[f"terrain_probe_{name}"] = float(value.mean())
    if include_reward_cfg:
        for key, value in _reward_cfg_payload(reward_cfg).items():
            reward_payload[f"cfg_{key}"] = float(value)
    return reward_payload


def build_command_payload(
    *,
    env: Any,
    commands: torch.Tensor,
    vb: torch.Tensor,
    root_ang_vel_b: torch.Tensor,
    v_along: torch.Tensor,
    speed: float,
    lin_err: float,
    yaw_err: float,
    progress_ratio: torch.Tensor,
    progress_gate: float,
    progress_by_dir: Mapping[str, float],
    raw_progress_by_dir: Mapping[str, float],
    active_dirs: tuple[str, ...] | list[str] | set[str],
    n: int,
    device: torch.device,
) -> dict[str, float | str]:
    lagging_dir, lagging_ratio = _lagging_progress(env, progress_by_dir, active_dirs)
    transition_state = getattr(env, "_cmd_transition_state", torch.zeros(n, device=device))
    settle_mask = transition_state == 2
    settle_denom = settle_mask.float().sum().clamp(min=1.0)
    strict_ready = getattr(env, "_cmd_transition_strict_ready", torch.zeros(n, dtype=torch.bool, device=device))
    body_safe = getattr(env, "_cmd_transition_body_safe", torch.zeros(n, dtype=torch.bool, device=device))
    motion_ready = getattr(env, "_cmd_transition_motion_ready", torch.zeros(n, dtype=torch.bool, device=device))
    posture_ready = getattr(env, "_cmd_transition_posture_ready", torch.zeros(n, dtype=torch.bool, device=device))
    support_phase_ready = getattr(
        env, "_cmd_transition_support_phase_ready", torch.zeros(n, dtype=torch.bool, device=device)
    )
    motion_fault = getattr(env, "_cmd_transition_motion_fault", torch.zeros(n, device=device))
    motion_target = getattr(env, "_cmd_transition_motion_target", torch.zeros(n, device=device))
    motion_excess = getattr(env, "_cmd_transition_motion_excess", torch.zeros(n, device=device))
    target_commands = getattr(env, "_cmd_target", commands)
    command_contract_error = (commands - target_commands).abs().amax(dim=-1)
    command_payload: dict[str, float | str] = {
        "target_vx": float(target_commands[:, 0].mean()),
        "target_vy": float(target_commands[:, 1].mean()),
        "target_wz": float(target_commands[:, 2].mean()),
        "cmd_vx": float(commands[:, 0].mean()),
        "cmd_vy": float(commands[:, 1].mean()),
        "cmd_wz": float(commands[:, 2].mean()),
        "actual_vx": float(vb[:, 0].mean()),
        "actual_vy": float(vb[:, 1].mean()),
        "actual_wz": float(root_ang_vel_b[:, 2].mean()),
        "v_along": float(v_along.mean()),
        "speed_xy": speed,
        "lin_err": lin_err,
        "yaw_err": yaw_err,
        "command_contract_error_mean": float(command_contract_error.mean()),
        "command_contract_error_max": float(command_contract_error.max()),
        "transition_policy_managed": 1.0 if getattr(env.cfg, "cmd_transition_policy_managed", False) else 0.0,
        "progress_ratio": float(progress_ratio.mean()),
        "heading_error_abs": float(getattr(env, "_heading_error", torch.zeros(n, device=device)).abs().mean()),
        # 分方向进展是阶段门控的实际阻塞项，需要写入 JSONL，而不是只靠人工读 console。
        "progress_fwd": progress_by_dir["fwd"],
        "progress_back": progress_by_dir["back"],
        "progress_lat": progress_by_dir["lat"],
        "progress_yaw": progress_by_dir["yaw"],
        "raw_progress_fwd": raw_progress_by_dir["fwd"],
        "raw_progress_back": raw_progress_by_dir["back"],
        "raw_progress_lat": raw_progress_by_dir["lat"],
        "raw_progress_yaw": raw_progress_by_dir["yaw"],
        "instant_progress_fwd": _float_attr(env, "_fwd_progress_instant"),
        "instant_progress_back": _float_attr(env, "_back_progress_instant"),
        "instant_progress_lat": _float_attr(env, "_lat_progress_instant"),
        "instant_progress_yaw": _float_attr(env, "_yaw_progress_instant"),
        "progress_samples_fwd": _float_attr(env, "_fwd_progress_samples"),
        "progress_samples_back": _float_attr(env, "_back_progress_samples"),
        "progress_samples_lat": _float_attr(env, "_lat_progress_samples"),
        "progress_samples_yaw": _float_attr(env, "_yaw_progress_samples"),
        "progress_min_active": float(progress_gate),
        "progress_validity": _float_attr(env, "_progress_validity"),
        "transition_strength": float(getattr(env, "_cmd_transition_strength", torch.zeros(n, device=device)).mean()),
        "transition_waiting_frac": float((getattr(env, "_cmd_transition_wait_steps", torch.zeros(n, device=device)) > 0).float().mean()),
        "transition_stable_frac": float((getattr(env, "_cmd_transition_stable_steps", torch.zeros(n, device=device)) > 0).float().mean()),
        "transition_decel_frac": float((getattr(env, "_cmd_transition_state", torch.zeros(n, device=device)) == 1).float().mean()),
        "transition_settle_frac": float((getattr(env, "_cmd_transition_state", torch.zeros(n, device=device)) == 2).float().mean()),
        "transition_reanchor_frac": float((getattr(env, "_cmd_transition_state", torch.zeros(n, device=device)) == 3).float().mean()),
        "transition_accel_frac": float((getattr(env, "_cmd_transition_state", torch.zeros(n, device=device)) == 4).float().mean()),
        "transition_failed_frac": float(getattr(env, "_cmd_transition_failure_ema", 0.0)),
        "transition_active_frac": float((getattr(env, "_cmd_transition_state", torch.zeros(n, device=device)) > 0).float().mean()),
        "transition_event_count": float(getattr(env, "_cmd_transition_event_count", 0)),
        "transition_failure_count": float(getattr(env, "_cmd_transition_failure_count", 0)),
        "transition_stop_event_count": float(getattr(env, "_cmd_transition_stop_event_count", 0)),
        "transition_stop_failure_count": float(getattr(env, "_cmd_transition_stop_failure_count", 0)),
        "transition_switch_event_count": float(getattr(env, "_cmd_transition_switch_event_count", 0)),
        "transition_switch_failure_count": float(getattr(env, "_cmd_transition_switch_failure_count", 0)),
        "transition_reverse_event_count": float(getattr(env, "_cmd_transition_reverse_event_count", 0)),
        "transition_reverse_failure_count": float(getattr(env, "_cmd_transition_reverse_failure_count", 0)),
        "transition_axis_event_count": float(getattr(env, "_cmd_transition_axis_event_count", 0)),
        "transition_axis_failure_count": float(getattr(env, "_cmd_transition_axis_failure_count", 0)),
        "flat_transition_safety_success": (
            1.0 - float(getattr(env, "_cmd_transition_flat_failure_ema", 0.0))
            if int(getattr(env, "_cmd_transition_flat_event_count", 0)) > 0 else 0.0
        ),
        "flat_transition_handoff_success": (
            1.0 - float(getattr(env, "_cmd_handoff_failure_ema", 0.0))
            if int(getattr(env, "_cmd_handoff_event_count", 0)) > 0 else 0.0
        ),
        "flat_transition_event_count": float(getattr(env, "_cmd_transition_flat_event_count", 0)),
        "flat_transition_failure_count": float(getattr(env, "_cmd_transition_flat_failure_count", 0)),
        "flat_transition_handoff_event_count": float(getattr(env, "_cmd_handoff_event_count", 0)),
        "flat_transition_stop_failure_rate": float(
            getattr(env, "_cmd_transition_flat_kind_failure_ema", {}).get("stop", 0.0)
        ),
        "flat_transition_reverse_failure_rate": float(
            getattr(env, "_cmd_transition_flat_kind_failure_ema", {}).get("reverse", 0.0)
        ),
        "flat_transition_axis_failure_rate": float(
            getattr(env, "_cmd_transition_flat_kind_failure_ema", {}).get("axis", 0.0)
        ),
        "flat_transition_stop_event_count": float(
            getattr(env, "_cmd_transition_flat_kind_event_count", {}).get("stop", 0)
        ),
        "flat_transition_reverse_event_count": float(
            getattr(env, "_cmd_transition_flat_kind_event_count", {}).get("reverse", 0)
        ),
        "flat_transition_axis_event_count": float(
            getattr(env, "_cmd_transition_flat_kind_event_count", {}).get("axis", 0)
        ),
        "transition_action_rate": float(getattr(env, "_cmd_transition_action_rate", torch.zeros(n, device=device)).mean()),
        "transition_strict_ready_rate": float((strict_ready.float() * settle_mask.float()).sum() / settle_denom),
        "transition_body_safe_rate": float((body_safe.float() * settle_mask.float()).sum() / settle_denom),
        "transition_motion_ready_rate": float((motion_ready.float() * settle_mask.float()).sum() / settle_denom),
        "transition_posture_ready_rate": float((posture_ready.float() * settle_mask.float()).sum() / settle_denom),
        "transition_support_phase_ready_rate": float(
            (support_phase_ready.float() * settle_mask.float()).sum() / settle_denom
        ),
        "transition_motion_fault": float((motion_fault * settle_mask.float()).sum() / settle_denom),
        "transition_motion_target": float((motion_target * settle_mask.float()).sum() / settle_denom),
        "transition_motion_excess": float((motion_excess * settle_mask.float()).sum() / settle_denom),
        "transition_task_weight": _float_attr(env, "_transition_task_weight_mean", 1.0),
        "transition_gait_weight": _float_attr(env, "_transition_gait_weight_mean", 1.0),
        "transition_support_ema": float(
            getattr(env, "_cmd_transition_support_ema", torch.zeros(n, device=device)).mean()
        ),
        "transition_zero_frac": float(getattr(env, "_cmd_transition_zero_frac", torch.full((n,), 0.5, device=device)).mean()),
        "progress_lagging_dir": lagging_dir,
        "progress_lagging_ratio": lagging_ratio,
        "gait_match": _float_attr(env, "_last_gait_match"),
        "gait_match_zero_lag": _float_attr(env, "_gait_match_zero_lag"),
        "gait_match_best_lag": _float_attr(env, "_best_lag_gait_match"),
        "diagonal_contact": _float_attr(env, "_diag_contact"),
        "duty_balance": _float_attr(env, "_duty_balance"),
        "duty_balance_product": _float_attr(env, "_duty_balance_product"),
        "stance_slip": _float_attr(env, "_slip_now"),
        "diagonal_pair_instant": _float_attr(env, "_diag_pair_instant"),
        "duty_balance_instant": _float_attr(env, "_duty_balance_instant"),
        "duty_spread_window": _float_attr(env, "_duty_spread"),
        "duty_target_score": _float_attr(env, "_duty_target_score"),
        "duty_symmetry_score": _float_attr(env, "_duty_symmetry_score"),
        "duty_range_error": _float_attr(env, "_duty_range_error"),
        "duty_symmetry_error": _float_attr(env, "_duty_symmetry_error"),
        "duty_cycle_valid_frac": _float_attr(env, "_duty_cycle_valid_frac"),
        "gait_period_score": _float_attr(env, "_gait_period_score"),
        "gait_period_error": _float_attr(env, "_gait_period_error"),
        "actual_contact_period": _float_attr(env, "_actual_contact_period"),
        "contact_period_valid_frac": _float_attr(env, "_contact_period_valid_frac"),
        "yaw_gait_gate": _float_attr(env, "_yaw_gait_gate"),
        "stance_slip_instant": _float_attr(env, "_slip_inst"),
        "stance_slip_high_fraction": _float_attr(env, "_slip_high_fraction"),
    }
    flat_quality = getattr(env, "_flat_quality_metrics", {})
    if isinstance(flat_quality, Mapping):
        for name, value in flat_quality.items():
            command_payload[f"flat_{name}"] = float(value)
    command_payload["flat_core_gate_ok"] = 1.0 if getattr(env, "_flat_core_gate_ok", False) else 0.0
    command_payload["flat_gait_gate_ok"] = 1.0 if getattr(env, "_flat_gait_gate_ok", False) else 0.0
    duty_by_leg = getattr(env, "_duty_by_leg", {})
    if isinstance(duty_by_leg, dict):
        for leg, value in duty_by_leg.items():
            command_payload[f"duty_{str(leg).lower()}"] = float(value)
    gait_by_direction = getattr(env, "_gait_by_direction", {})
    if isinstance(gait_by_direction, dict):
        for direction, value in gait_by_direction.items():
            command_payload[f"gait_{direction}"] = float(value)
    duty_by_direction = getattr(env, "_duty_by_direction", {})
    if isinstance(duty_by_direction, dict):
        for direction, value in duty_by_direction.items():
            command_payload[f"duty_{direction}"] = float(value)
    execution = getattr(env, "_command_execution_coverage", {})
    if isinstance(execution, dict):
        for direction, value in execution.items():
            command_payload[f"execution_{direction}"] = float(value)
    direction_audit = getattr(env, "_direction_audit", {})
    if isinstance(direction_audit, Mapping):
        for direction, values in direction_audit.items():
            if not isinstance(values, Mapping):
                continue
            for name, value in values.items():
                if isinstance(value, (int, float)):
                    command_payload[f"direction_{direction}_{name}"] = float(value)
    command_payload["quality_capability_gate"] = _float_attr(env, "_quality_capability_gate")
    # Keep a disjoint per-batch command view alongside the existing
    # direction-oriented audit. The latter is useful for progress attribution
    # but overlaps mixed commands; this view is the source for coverage proof.
    steady_eval = transition_state == 0
    for prefix, batch_commands in (("target", target_commands), ("applied", commands)):
        for bucket, mask in _command_bucket_masks(batch_commands).items():
            command_payload[f"bucket_{bucket}_{prefix}_samples"] = float(mask.float().sum())
            if prefix == "applied":
                command_payload[f"bucket_{bucket}_eligible_samples"] = float(
                    (mask & steady_eval).float().sum()
                )
    return command_payload


def build_curriculum_payload(
    *,
    env: Any,
    phase: int | None,
    dr_level: Any,
    terrain_mean: float | None,
    terrain_max: int | None,
    terrain_stats: Mapping[str, float],
    terrain_type_payload: Mapping[str, float | int],
    progress_gate: float | None,
    progress_by_dir: Mapping[str, float],
    raw_progress_by_dir: Mapping[str, float],
    active_dirs: tuple[str, ...] | list[str] | set[str],
    command_payload: Mapping[str, float | str],
    gait_gate: float | None,
    fall_rate: float | None,
) -> dict[str, Any]:
    curriculum_payload: dict[str, Any] = {
        "phase": f"phi{phase}" if phase is not None else "",
        "command_mode": getattr(env, "_last_command_mode", ""),
        "terrain_mean": terrain_mean,
        "terrain_max": terrain_max,
        "terrain_flat_mean": float(terrain_stats.get("terrain_flat_mean", 0.0)),
        "terrain_flat_max": int(terrain_stats.get("terrain_flat_max", 0)),
        "terrain_real_mean": float(terrain_stats.get("terrain_real_mean", 0.0)),
        "terrain_real_max": int(terrain_stats.get("terrain_real_max", 0)),
        "terrain_discrete_mean": float(terrain_stats.get("terrain_discrete_mean", 0.0)),
        "terrain_discrete_max": int(terrain_stats.get("terrain_discrete_max", 0)),
        "dr_level": dr_level,
        "terrain_move_up_rate": _float_attr(env, "_terrain_curriculum_move_up_rate"),
        "terrain_move_down_rate": _float_attr(env, "_terrain_curriculum_move_down_rate"),
        "terrain_failure_down_rate": _float_attr(env, "_terrain_curriculum_failure_down_rate"),
        "terrain_stable_end_rate": _float_attr(env, "_terrain_curriculum_stable_end_rate"),
        "terrain_speed_ok_rate": _float_attr(env, "_terrain_curriculum_speed_ok_rate"),
        "terrain_eligible_frac": _float_attr(env, "_terrain_curriculum_eligible_frac"),
        "terrain_stair_height_ok_rate": _float_attr(env, "_terrain_curriculum_stair_height_ok_rate"),
        "terrain_boxes_success_rate": _float_attr(env, "_terrain_boxes_success_ema"),
        "terrain_stairs_down_success_rate": _float_attr(env, "_terrain_stairs_down_success_ema"),
        "terrain_stairs_up_success_rate": _float_attr(env, "_terrain_stairs_up_success_ema"),
        "terrain_boxes_collapse_rate": _float_attr(env, "_terrain_boxes_collapse_ema"),
        "terrain_stairs_down_collapse_rate": _float_attr(env, "_terrain_stairs_down_collapse_ema"),
        "terrain_stairs_up_collapse_rate": _float_attr(env, "_terrain_stairs_up_collapse_ema"),
        "terrain_discrete_move_up_rate": _float_attr(env, "_terrain_curriculum_discrete_move_up_rate"),
        "terrain_discrete_move_down_rate": _float_attr(env, "_terrain_curriculum_discrete_move_down_rate"),
        "terrain_discrete_failure_down_rate": _float_attr(env, "_terrain_curriculum_discrete_failure_down_rate"),
        "terrain_discrete_stable_end_rate": _float_attr(env, "_terrain_curriculum_discrete_stable_end_rate"),
        "phase_count": float(getattr(env, "_phase_count", 0)),
        "penalty_gate": _float_attr(env, "_penalty_gate"),
        "budget_ratio": _float_attr(env, "_budget_ratio_ema"),
        "clearance_gate": _float_attr(env, "_clearance_gate"),
        "terrain_health_ok": 1.0 if getattr(env, "_terrain_health_ok", True) else 0.0,
    }
    curriculum_payload.update(terrain_type_payload)
    if progress_gate is not None:
        curriculum_payload["progress_gate"] = progress_gate
        curriculum_payload["progress_fwd"] = progress_by_dir["fwd"]
        curriculum_payload["progress_back"] = progress_by_dir["back"]
        curriculum_payload["progress_lat"] = progress_by_dir["lat"]
        curriculum_payload["progress_yaw"] = progress_by_dir["yaw"]
        curriculum_payload["raw_progress_fwd"] = raw_progress_by_dir["fwd"]
        curriculum_payload["raw_progress_back"] = raw_progress_by_dir["back"]
        curriculum_payload["raw_progress_lat"] = raw_progress_by_dir["lat"]
        curriculum_payload["raw_progress_yaw"] = raw_progress_by_dir["yaw"]
        curriculum_payload["progress_validity"] = _float_attr(env, "_progress_validity")
        curriculum_payload["yaw_ceil"] = _float_attr(env, "_vel_max_yaw")   # yaw 速度上限诊断。
        curriculum_payload["progress_lagging_dir"] = command_payload["progress_lagging_dir"]
        if active_dirs:
            curriculum_payload["active_dirs"] = ",".join(active_dirs)
        cfg_obj = getattr(env, "cfg", None)
        progress_threshold = None
        if cfg_obj is not None and phase is not None:
            progress_threshold = getattr(cfg_obj, f"phase_gate_prog_{int(phase)}", None)
        if progress_threshold is not None and progress_gate < float(progress_threshold):
            curriculum_payload["blocked_by"] = "progress"
            curriculum_payload["next_gate"] = "phase_progress"
    if gait_gate is not None:
        curriculum_payload["gait_gate"] = gait_gate
        curriculum_payload["diagonal_gate"] = _float_attr(env, "_diag_contact")
        curriculum_payload["duty_balance_gate"] = _float_attr(env, "_duty_balance")
        curriculum_payload["slip_gate"] = _float_attr(env, "_slip_now")
        curriculum_payload["duty_spread_gate"] = _float_attr(env, "_duty_spread")
    curriculum_payload["execution_gate"] = _float_attr(env, "_execution_gate")
    curriculum_payload["transition_failure_gate"] = _float_attr(env, "_transition_failure_gate")
    curriculum_payload["phase_transition_gate_active"] = float(
        bool(getattr(env, "_phase_transition_gate_active", False))
    )
    curriculum_payload["dr_gate_count"] = _float_attr(env, "_dr_gate_count")
    curriculum_payload["dr_gate_eligible"] = float(bool(getattr(env, "_dr_gate_eligible", False)))
    curriculum_payload["dr_gate_terrain_level"] = _float_attr(env, "_dr_gate_terrain_level")
    curriculum_payload["dr_gate_terrain_required"] = _float_attr(env, "_dr_gate_terrain_required")
    curriculum_payload["dr_gate_progress_required"] = _float_attr(env, "_dr_gate_progress_required")
    curriculum_payload["dr_apply_fraction"] = _float_attr(env, "_dr_apply_fraction")
    curriculum_payload["dr_transition_kind_ok"] = float(bool(getattr(env, "_dr_transition_kind_ok", False)))
    curriculum_payload["dr_transition_handoff_ok"] = float(bool(getattr(env, "_dr_transition_handoff_ok", False)))
    curriculum_payload["flat_core_gate_ok"] = float(bool(getattr(env, "_flat_core_gate_ok", False)))
    curriculum_payload["flat_gait_gate_ok"] = float(bool(getattr(env, "_flat_gait_gate_ok", False)))
    curriculum_payload["terrain_health_slip_high_value"] = _float_attr(env, "_slip_high_fraction")
    curriculum_payload["phase_gate_ok"] = float(bool(getattr(env, "_phase_gate_ok", False)))
    curriculum_payload["phase_gate_eval_step"] = float(getattr(env, "_phase_gate_eval_step", 0))
    for group_name, attr_name in (
        ("phase", "_phase_gate_status"),
        ("flat_core", "_flat_core_gate_checks"),
        ("flat_gait", "_flat_gait_gate_checks"),
    ):
        checks = getattr(env, attr_name, {})
        if isinstance(checks, Mapping):
            for name, value in checks.items():
                curriculum_payload[f"{group_name}_gate_{name}_ok"] = float(bool(value))
    gate_values = getattr(env, "_phase_gate_values", {})
    if isinstance(gate_values, Mapping):
        for name, value in gate_values.items():
            curriculum_payload[f"phase_gate_{name}_value"] = float(value)

    # blocked_by 必须来自与 phase_count 同一次评估的最终条件，不能只看 progress，
    # 也不能把更高频的即时指标误写成当前门控结论。
    phase_checks = getattr(env, "_phase_gate_status", {})
    core_checks = getattr(env, "_flat_core_gate_checks", {})
    gait_checks = getattr(env, "_flat_gait_gate_checks", {})
    if isinstance(phase_checks, Mapping) and phase_checks:
        active_checks: list[tuple[str, bool]] = [
            ("progress", bool(phase_checks.get("progress", False))),
            ("execution", bool(phase_checks.get("execution", False))),
            ("terminal_rate", bool(phase_checks.get("terminal_rate", False))),
            ("quality", bool(phase_checks.get("quality", False))),
        ]
        terrain_phase_active = bool(phase_checks.get("terrain_phase_active", False))
        terrain_mixed_active = bool(phase_checks.get("terrain_mixed_active", False))
        if terrain_phase_active:
            if isinstance(core_checks, Mapping):
                active_checks.extend(
                    (f"flat_{name}", bool(value)) for name, value in core_checks.items()
                )
            if isinstance(gait_checks, Mapping):
                active_checks.extend(
                    (f"flat_{name}", bool(value)) for name, value in gait_checks.items()
                )
        else:
            active_checks.append(("transition", bool(phase_checks.get("transition", False))))
        if terrain_mixed_active:
            active_checks.extend((
                ("terrain_level", bool(phase_checks.get("terrain_level", False))),
                ("terrain_discrete", bool(phase_checks.get("terrain_discrete", False))),
                ("terrain_type", bool(phase_checks.get("terrain_type", False))),
                ("terrain_capability", bool(phase_checks.get("terrain_capability", False))),
                ("terrain_fall", bool(phase_checks.get("terrain_fall", False))),
            ))
        blockers = [name for name, passed in active_checks if not passed]
        curriculum_payload["phase_gate_blockers"] = ",".join(blockers)
        if blockers:
            curriculum_payload["blocked_by"] = ",".join(blockers)
            curriculum_payload["next_gate"] = "phase_gate"
        else:
            curriculum_payload.pop("blocked_by", None)
            curriculum_payload.pop("next_gate", None)
    cfg_obj = getattr(env, "cfg", None)
    if cfg_obj is not None and phase is not None:
        def phase_threshold(name: str, default: float) -> float:
            for current_phase in range(int(phase), -1, -1):
                value = getattr(cfg_obj, f"phase_gate_{name}_{current_phase}", None)
                if value is not None:
                    return float(value)
            return float(default)

        threshold_defaults = {
            "prog": 0.70,
            "slip": 0.20,
            "diag": 0.80,
            "duty": 0.80,
            "duty_target": 0.80,
            "duty_symmetry": 0.80,
            "period": 0.55,
            "yaw_gait": 0.35,
            "duty_valid": 0.70,
            "execution": 0.80,
            "terminal_rate": 0.01,
            "transition_fail": 0.12,
            "air": 0.0,
            "flat_tilt_p95": 0.16,
            "flat_wxy": 0.40,
            "flat_height_error_p95": 0.055,
            "flat_touchdown_vz_p95": 0.60,
            "flat_slip_high": 0.28,
            "flat_trajectory_p95": 3.0,
            "flat_false_terrain_response": 0.01,
        }
        duty_target_default = phase_threshold("duty", threshold_defaults["duty"])
        threshold_defaults["duty_target"] = duty_target_default
        threshold_defaults["duty_symmetry"] = duty_target_default
        for name, default in threshold_defaults.items():
            curriculum_payload[f"phase_gate_{name}_target"] = phase_threshold(name, default)
    if fall_rate is not None:
        cfg_obj = getattr(env, "cfg", None)
        fall_threshold = getattr(cfg_obj, "phase_gate_fall_2", None) if cfg_obj is not None else None
        curriculum_payload["fall_gate"] = fall_rate
        if fall_threshold is not None and fall_rate >= float(fall_threshold):
            curriculum_payload["blocked_by"] = curriculum_payload.get("blocked_by", "fall")
            curriculum_payload["next_gate"] = curriculum_payload.get("next_gate", "fall_rate")
    return curriculum_payload


def build_checkpoint_performance_snapshot(
    *,
    reward_payload: Mapping[str, Any],
    health_payload: Mapping[str, Any],
    phase: int | None,
    episode_length_mean: float | None = None,
) -> dict[str, Any]:
    """构造检查点登记用的性能快照（P4.2 的 CheckpointRegistry 消费它）。

    这里**刻意收数值 phase**，而不是从 ``build_curriculum_payload`` 的返回值里取。
    那个 payload 的 ``phase`` 是给人看的显示串（``"phi0"``），解析它就是个陷阱：
    2026-09-16 之前写的是 ``int(curriculum_payload.get("phase", 0))``，
    于是训练**第一步**就抛 ``ValueError: invalid literal for int() with base 10: 'phi0'``
    （``phase is None`` 时值是空串，同样抛）。两种表示各走各的路，显示串只用来显示。

    键名同样是个陷阱，2026-09-16 对着真遥测核过：

    - ``reward_payload`` 里没有 ``"mean"``，总数叫 **``"total"``**。
      原先读 ``.get("mean", 0.0)`` 于是恒为 0.0——而 curator 的评分一半权重压
      在 reward_mean 上、质量门也拿它比阈值，等于整条择优链路在看一个常数。
    - ``health_payload`` 里没有 ``episode_length_mean``。它由调用方从环境的
      ``episode_length_buf`` 现算传进来；拿不到时记 0.0 而不是 100.0——
      100.0 会让 ``episode_length > 100`` 这类门看起来通过，是编数据。
    """
    return {
        "reward_mean": float(reward_payload.get("total", 0.0)),
        "terminal_rate": float(health_payload.get("terminal_rate", 0.0)),
        "episode_length_mean": float(episode_length_mean) if episode_length_mean is not None else 0.0,
        "curriculum_phase": int(phase) if phase is not None else 0,
        "checkpoint_mtime": time.time(),
    }


def build_health_payload(
    *,
    gate: torch.Tensor,
    moving: torch.Tensor,
    stand_gate: torch.Tensor,
    base_h: torch.Tensor,
    upright: float,
    tilt_rel: torch.Tensor,
    support_instab: torch.Tensor,
    cc: torch.Tensor,
    in_contact: torch.Tensor,
    torque_util: torch.Tensor,
    terminal_window: torch.Tensor,
    height_low_risk: float,
    tilt_high_risk: float,
    fall_rate: float | None,
) -> dict[str, float]:
    health_payload = {
        "stable_motion_gate": float(gate.mean()),
        "moving_gate": float(moving.mean()),
        "stand_gate": float(stand_gate.mean()),
        "base_h": float(base_h.mean()),
        "upright": upright,
        "tilt_deg": float(torch.rad2deg(tilt_rel).mean()),
        "support_instability": float(support_instab.mean()),
        "height_low_risk_window": float(height_low_risk),
        "tilt_high_risk_window": float(tilt_high_risk),
        "base_h_min": float(base_h.min()),
        "tilt_deg_max": float(torch.rad2deg(tilt_rel).max()),
        "contact_count": float(cc.mean()),
        "contacts_mean": float(in_contact.mean()),
        "torque_util": float(torque_util),
        "terminal_rate": float(terminal_window.mean()),
    }
    if fall_rate is not None:
        health_payload["fall_rate"] = fall_rate
    return health_payload
