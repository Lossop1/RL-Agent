"""训练遥测字段契约测试。

该测试只解析源码，不运行仿真。目的不是验证字段数值，而是防止关键遥测字段
在整理 `_get_rewards` 或拆分 telemetry builder 时被无意改名或删除。
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

from products.taili.blind_locomotion.telemetry_payloads import _command_bucket_masks, _lagging_progress


ROOT = Path(__file__).resolve().parents[4]
BLIND_ENV = ROOT / "products" / "taili" / "blind_locomotion" / "blind_tp_env.py"
CORE_REWARD = ROOT / "products" / "taili" / "core" / "taili_reward.py"
TELEMETRY_PAYLOADS = ROOT / "products" / "taili" / "blind_locomotion" / "telemetry_payloads.py"


def _payload_module() -> ast.Module:
    return ast.parse(TELEMETRY_PAYLOADS.read_text(encoding="utf-8"))


def _dict_literal_keys(node: ast.AST) -> set[str]:
    if not isinstance(node, ast.Dict):
        return set()
    return {
        key.value
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


def _payload_keys(payload_name: str) -> set[str]:
    keys: set[str] = set()
    for node in ast.walk(_payload_module()):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == payload_name:
                    keys.update(_dict_literal_keys(node.value))
        if isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == payload_name and node.value is not None:
                keys.update(_dict_literal_keys(node.value))
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == payload_name:
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                keys.add(node.slice.value)
    return keys


def test_command_bucket_masks_are_disjoint_and_complete():
    commands = torch.tensor(
        [
            [0.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
            [0.0, 0.3, 0.0],
            [0.0, 0.0, 0.4],
            [0.0, 0.0, 0.0],
            [0.4, 0.2, 0.0],
        ]
    )
    masks = _command_bucket_masks(commands)
    assert set(masks) == {"forward", "backward", "lateral", "yaw", "stand", "mixed"}
    stacked = torch.stack(list(masks.values()), dim=0)
    assert torch.all(stacked.sum(dim=0) == 1)
    assert all(int(mask.sum()) == 1 for mask in masks.values())


def test_command_payload_keeps_core_fields():
    required = {
        "cmd_vx",
        "cmd_vy",
        "cmd_wz",
        "target_vx",
        "target_vy",
        "target_wz",
        "actual_vx",
        "actual_vy",
        "actual_wz",
        "v_along",
        "speed_xy",
        "lin_err",
        "yaw_err",
        "command_contract_error_mean",
        "command_contract_error_max",
        "transition_policy_managed",
        "progress_fwd",
        "progress_back",
        "progress_lat",
        "progress_yaw",
        "raw_progress_fwd",
        "raw_progress_back",
        "raw_progress_lat",
        "raw_progress_yaw",
        "instant_progress_fwd",
        "instant_progress_back",
        "instant_progress_lat",
        "instant_progress_yaw",
        "progress_samples_fwd",
        "progress_samples_back",
        "progress_samples_lat",
        "progress_samples_yaw",
        "progress_min_active",
        "progress_validity",
        "progress_lagging_dir",
        "progress_lagging_ratio",
        "transition_active_frac",
        "transition_strength",
        "transition_zero_frac",
        "transition_waiting_frac",
        "transition_stable_frac",
        "transition_decel_frac",
        "transition_settle_frac",
        "transition_reanchor_frac",
        "transition_accel_frac",
        "transition_failed_frac",
        "transition_event_count",
        "transition_failure_count",
        "transition_stop_event_count",
        "transition_stop_failure_count",
        "transition_switch_event_count",
        "transition_switch_failure_count",
        "transition_reverse_event_count",
        "transition_reverse_failure_count",
        "transition_axis_event_count",
        "transition_axis_failure_count",
        "transition_action_rate",
        "transition_strict_ready_rate",
        "transition_body_safe_rate",
        "transition_motion_ready_rate",
        "transition_posture_ready_rate",
        "transition_support_phase_ready_rate",
        "transition_motion_fault",
        "transition_motion_target",
        "transition_motion_excess",
        "transition_task_weight",
        "transition_gait_weight",
        "transition_support_ema",
        "flat_transition_safety_success",
        "flat_transition_handoff_success",
        "flat_transition_event_count",
        "flat_transition_failure_count",
        "flat_transition_handoff_event_count",
        "flat_transition_stop_failure_rate",
        "flat_transition_reverse_failure_rate",
        "flat_transition_axis_failure_rate",
        "flat_transition_stop_event_count",
        "flat_transition_reverse_event_count",
        "flat_transition_axis_event_count",
        "gait_match",
        "gait_match_zero_lag",
        "gait_match_best_lag",
        "gait_period_score",
        "gait_period_error",
        "actual_contact_period",
        "contact_period_valid_frac",
        "yaw_gait_gate",
        "duty_range_error",
        "duty_symmetry_error",
        "duty_cycle_valid_frac",
        "diagonal_contact",
        "duty_balance",
        "stance_slip",
        "stance_slip_high_fraction",
    }
    assert required <= _payload_keys("command_payload")
    source = TELEMETRY_PAYLOADS.read_text(encoding="utf-8")
    assert 'command_payload[f"duty_{direction}"]' in source
    assert 'command_payload[f"execution_{direction}"]' in source
    assert 'command_payload[f"direction_{direction}_{name}"]' in source
    assert 'command_payload["quality_capability_gate"]' in source


def test_duty_and_period_use_physical_time_windows_instead_of_pure_diagonal_switches():
    source = BLIND_ENV.read_text(encoding="utf-8")
    assert "required_duty_steps" in source
    assert "touchdown_period" in source
    assert "_last_touchdown_step" in source
    assert "_duty_cycle_switches" not in source
    assert "_contact_pattern" not in source


def test_touchdown_force_uses_positive_peak_over_the_short_contact_window():
    source = BLIND_ENV.read_text(encoding="utf-8")
    assert "_foot_force_norm - self._prev_foot_force_norm" in source
    assert "torch.maximum(self._touchdown_force_hold, _force_rate_now)" in source
    assert '"touchdown_force_rate_p95"' in source


def test_stair_state_machine_is_absent_from_the_runtime_source():
    source = BLIND_ENV.read_text(encoding="utf-8")
    assert "_stair_event_stage_memory" not in source
    assert "_stair_event_lead_foot" not in source
    assert "compute_stair_event_progress(" not in source
    assert "terrain_curriculum.loaded_support_height(" in source


def test_stair_recovery_uses_current_contact_with_only_a_bounded_reaction_trace():
    source = BLIND_ENV.read_text(encoding="utf-8")
    assert "terrain_curriculum.unexpected_body_contact_score(" in source
    assert "terrain_curriculum.map_body_collision_to_legs(" in source
    assert "terrain_collision_foot_mask=_current_collision_strength" in source
    assert "terrain_curriculum.update_terrain_collision_trace(" in source
    assert "self._terrain_collision_trace[env_ids] = 0.0" in source
    assert '"collision_trace_mean": _collision_trace.mean()' in source
    assert "terrain_clearance_response=_terrain_clearance_response" in source
    assert 'self._contact_sensor.find_bodies(f"{leg}_(hip|thigh|calf)")' in source
    assert 'self._contact_sensor.find_bodies("base_link")' in source
    assert '"limb_collision_response_mean"' in source
    assert '"base_collision_response_mean"' in source
    assert "terrain_recovery_termination_height" in source
    assert "terrain_recovery_termination_tilt_deg" in source


def test_stair_reward_uses_direction_drive_without_layer_or_potential_state():
    env_source = BLIND_ENV.read_text(encoding="utf-8")
    reward_source = CORE_REWARD.read_text(encoding="utf-8")
    assert "terrain_curriculum.continuous_terrain_height_drive(" not in env_source
    assert "terrain_curriculum.continuous_terrain_layer_hold(" not in env_source
    assert "terrain_curriculum.support_layer_split_quality(" not in env_source
    assert '* unified_progress_ratio' in reward_source
    assert '* unified_alignment_credit' in reward_source
    assert '* moving_gate\n        * hard_survival_gate' in reward_source
    assert '+ comp["terrain_contact_quality"]' not in env_source
    assert '+ comp["terrain_support_loss"]' not in env_source


def test_reset_cannot_inject_velocity_credit_or_escape_a_failed_command():
    source = BLIND_ENV.with_name("taili_amp_env.py").read_text(encoding="utf-8")
    resample_at = source.index("self._resample_commands(env_ids, snap=True)")
    preserve_at = source.index("taili_amp_reference.preserve_failed_command_targets(")
    neutralize_at = source.index("taili_amp_reference.neutralize_reset_velocities(root, jv)")
    write_velocity_at = source.index("self.robot.write_root_com_velocity_to_sim(root[:, 7:], env_ids)")

    assert resample_at < preserve_at
    assert neutralize_at < write_velocity_at


def test_stair_commands_do_not_consume_the_flat_forward_occupancy_quota():
    source = BLIND_ENV.read_text(encoding="utf-8")
    assert "non_stair_global = ~stair_mask" in source
    assert "single_axis_occupancy_deficits(\n                        non_stair_total," in source
    assert "deficits[1] = max(0, deficits[1] - stair_count)" not in source


def test_quality_schedule_keeps_balanced_maturity_and_phase_gate_minimum():
    blind_source = BLIND_ENV.read_text(encoding="utf-8")
    amp_source = BLIND_ENV.with_name("taili_amp_env.py").read_text(encoding="utf-8")
    assert "_quality_progress_balanced = taili_reward.balanced_progress_geometric" in blind_source
    assert "min_prog, active_dirs = active_direction_progress" in amp_source


def test_quality_schedule_is_not_weakened_by_penalty_budget_feedback():
    blind_source = BLIND_ENV.read_text(encoding="utf-8")
    amp_source = BLIND_ENV.with_name("taili_amp_env.py").read_text(encoding="utf-8")
    assert "_quality_gate_value = self._quality_gate_latched" in blind_source
    assert "_refinement_gate_value = self._refinement_gate_latched" in blind_source
    assert "budgeted_quality_gate" not in blind_source
    assert "if not self._penalty_budget_controls_quality:" in amp_source
    assert "self._penalty_gate = 1.0" in amp_source


def test_phase_gate_does_not_wait_for_penalty_budget_ramp():
    source = BLIND_ENV.with_name("taili_amp_env.py").read_text(encoding="utf-8")
    assert "and self._penalty_gate >= 1.0" not in source
    assert "(self._penalty_gate >= 1.0 or not self._penalty_budget_controls_quality)" in source


def test_flat_false_terrain_response_uses_configurable_noise_tolerance():
    source = BLIND_ENV.with_name("taili_amp_env.py").read_text(encoding="utf-8")
    blind_source = BLIND_ENV.read_text(encoding="utf-8")
    assert '_phase_thr("flat_false_terrain_response", 0.01)' in source
    assert 'flat_false_terrain_response", 1e-6' not in source
    assert "terrain_response_height_deadband" in blind_source
    assert "terrain_response_delta_deadband" in blind_source


def test_terminal_swing_deceleration_uses_the_best_lag_reference_phase():
    source = BLIND_ENV.read_text(encoding="utf-8")
    assert "_reference_phase = (self._gait_phase + _phase_lag) % 1.0" in source
    assert "_trajectory_phase = (self._leg_phases() + _phase_lag[:, None]) % 1.0" in source


def test_terrain_curriculum_is_not_serially_locked_by_flat_direction_progress():
    source = BLIND_ENV.with_name("taili_amp_env.py").read_text(encoding="utf-8")
    assert "terrain_unlocked = terrain_curriculum_active" in source
    assert "terrain_unlocked = terrain_curriculum_active and self._advance_ok" not in source


def test_lagging_direction_uses_each_directions_own_threshold():
    env = SimpleNamespace(
        _phase=0,
        cfg=SimpleNamespace(
            phase_gate_prog_0=0.65,
            phase_progress_thresholds_0={"fwd": 0.65, "back": 0.60, "lat": 0.60, "yaw": 0.35},
        ),
    )
    direction, ratio = _lagging_progress(
        env,
        {"fwd": 0.64, "back": 0.62, "lat": 0.61, "yaw": 0.50},
        ("fwd", "back", "lat", "yaw"),
    )
    assert direction == "fwd"
    assert ratio == 0.64 / 0.65


def test_transition_completion_uses_physical_safety_without_clock_reanchor():
    source = BLIND_ENV.with_name("taili_amp_env.py").read_text(encoding="utf-8")
    assert "motion_ready" in source
    assert "& leg_state_ready" not in source
    assert "self._gait_phase[ids[nearest_a]]" not in source
    assert "transition_support_phase_ready(" in source
    assert "ready = self._cmd_transition_stable_steps[ids] >= required_steps" in source
    assert "& support_phase_ready" not in source
    assert "_cmd_transition_flat_failure_ema" in source
    assert "_cmd_handoff_failure_ema" in source


def test_policy_managed_transition_never_rewrites_actor_command_or_target():
    source = BLIND_ENV.with_name("taili_amp_env.py").read_text(encoding="utf-8")
    reward_source = BLIND_ENV.read_text(encoding="utf-8")
    core_reward_source = CORE_REWARD.read_text(encoding="utf-8")
    assert "self.commands.copy_(self._cmd_target)" in source
    assert "cmd_transition_policy_managed" in source
    assert "self._cmd_target[failed_ids] = 0.0" not in source
    assert "old_command=self._cmd_transition_start[ids]" in source
    assert "self._cmd_transition_require_stop[ids] = through_stop[idx]" in source
    assert "_cmd_transition_support_ema" in source
    assert "& _flat_transition" in reward_source
    assert "(0.30 + 0.70 * gate) * _flat_transition_f" in reward_source
    assert "_transition_task_weight = torch.ones_like" in reward_source
    assert "taili_reward.transition_style_weight(" in reward_source
    assert '_amp_style_scale_for_agent' in reward_source
    assert 'self.extras["amp_style_scale"]' in reward_source
    assert "_policy_steady_f = torch.ones_like" not in reward_source
    assert "_new = _old * _policy_steady_f" not in reward_source
    assert "phase_command = self.commands" in source
    assert reward_source.count("from isaaclab.utils.math import quat_apply_inverse") == 1
    assert '"directional_support_balance"' in core_reward_source
    assert '"lateral_coordinated_progress"' in core_reward_source
    assert '"foot_trajectory"' in core_reward_source
    assert "hip_deviation_max=" in reward_source
    assert "_flush_transition_rate_stats" in source
    assert "easy_event = direct & transition_intent" in source
    assert "self._record_command_transition_result(easy_ids, failed=False)" in source
    assert "self._cmd_transition_kind[easy_ids] = 0" in source
    for key in ("tracking_lin", "tracking_yaw", "tracking_lin_far", "tracking_yaw_far", "terrain_progress"):
        assert f'"{key}"' in core_reward_source


def test_flat_quality_and_continuous_stair_contracts_are_explicit():
    source = BLIND_ENV.read_text(encoding="utf-8")
    parent_source = BLIND_ENV.with_name("taili_amp_env.py").read_text(encoding="utf-8")
    assert "scope_terrain_response" in source
    assert '"false_terrain_response"' in source
    assert '"support_delta_mean"' in source
    assert '"support_dispersion_mean"' in source
    assert '"direction_ratio_mean"' in source
    assert 'comp["direction_alignment"]' in source
    assert 'comp["terrain_tracking_scale"]' in source
    assert '"down_control_fault_mean"' in source
    assert 'f"terrain_{name}"' in parent_source
    assert "_flat_direction_masks" in source
    assert "_direction_extreme" in source
    assert '"duty_valid": float(flat_metrics.get("duty_valid"' in parent_source
    assert '"period": float(flat_metrics.get("period"' in parent_source
    assert '"yaw_gait": (not yaw_required or yaw_gait_score >= yaw_gait_thr)' in parent_source


def test_transition_is_diagnostic_only_for_phase_and_dr():
    source = BLIND_ENV.with_name("taili_amp_env.py").read_text(encoding="utf-8")
    assert "self._phase_transition_gate_active = False" in source
    assert "transition_safety_ok = True" in source
    assert 'for name in ("stop", "reverse", "axis")' in source
    assert "dr_transition_ok = True" in source


def test_curriculum_payload_keeps_gate_and_terrain_fields():
    required = {
        "phase",
        "command_mode",
        "dr_level",
        "phase_count",
        "phase_gate_ok",
        "penalty_gate",
        "budget_ratio",
        "clearance_gate",
        "terrain_health_ok",
        "progress_gate",
        "progress_fwd",
        "progress_back",
        "progress_lat",
        "progress_yaw",
        "raw_progress_fwd",
        "raw_progress_back",
        "raw_progress_lat",
        "raw_progress_yaw",
        "progress_validity",
        "progress_lagging_dir",
        "active_dirs",
        "blocked_by",
        "next_gate",
        "terrain_mean",
        "terrain_max",
        "terrain_real_mean",
        "terrain_real_max",
        "terrain_discrete_mean",
        "terrain_discrete_max",
        "terrain_move_up_rate",
        "terrain_move_down_rate",
        "terrain_failure_down_rate",
        "terrain_stable_end_rate",
        "terrain_speed_ok_rate",
        "terrain_eligible_frac",
        "phase_transition_gate_active",
        "dr_transition_kind_ok",
        "dr_transition_handoff_ok",
        "gait_gate",
        "diagonal_gate",
        "duty_balance_gate",
        "slip_gate",
        "fall_gate",
        "yaw_ceil",
    }
    assert required <= _payload_keys("curriculum_payload")


def test_runtime_phase_gate_emits_exact_subconditions_and_values():
    parent_source = BLIND_ENV.with_name("taili_amp_env.py").read_text(encoding="utf-8")
    payload_source = TELEMETRY_PAYLOADS.read_text(encoding="utf-8")
    assert "self._phase_gate_ok = bool(gate)" in parent_source
    assert "self._phase_gate_status =" in parent_source
    assert "self._phase_gate_values =" in parent_source
    assert '"_flat_core_gate_checks"' in payload_source
    assert '"_flat_gait_gate_checks"' in payload_source
    assert 'f"phase_gate_{name}_value"' in payload_source


def test_health_payload_keeps_stability_fields():
    required = {
        "stable_motion_gate",
        "moving_gate",
        "stand_gate",
        "base_h",
        "base_h_min",
        "upright",
        "tilt_deg",
        "tilt_deg_max",
        "support_instability",
        "height_low_risk_window",
        "tilt_high_risk_window",
        "contact_count",
        "contacts_mean",
        "torque_util",
        "terminal_rate",
        "fall_rate",
    }
    assert required <= _payload_keys("health_payload")


def test_reward_payload_keeps_base_fields_and_dynamic_passthroughs():
    required = {"total", "lin_err", "speed", "gait", "base_h", "upright"}
    assert required <= _payload_keys("reward_payload")

    source = TELEMETRY_PAYLOADS.read_text(encoding="utf-8")
    assert "for name, value in comp.items():" in source
    assert "reward_payload[name]" in source
    assert "for name, value in terrain_probe.items():" in source
    assert 'reward_payload[f"terrain_probe_{name}"]' in source
    assert 'reward_payload[f"cfg_{key}"]' in source


def test_emitter_receives_all_four_payloads():
    source = BLIND_ENV.read_text(encoding="utf-8")
    assert "build_reward_payload(" in source
    assert "build_command_payload(" in source
    assert "build_curriculum_payload(" in source
    assert "build_health_payload(" in source
    assert "self._telemetry.emit(" in source
    assert "reward=reward_payload" in source
    assert "curriculum=curriculum_payload" in source
    assert "health=health_payload" in source
    assert "command=command_payload" in source
