"""compute_reward_components 的奖励不变量回归测试。

这些测试把 `docs/taili_strategy_decisions.md` 中的设计意图变成可执行检查：
单机器人均值规约、不能站立骗前进、反向运动压力、terminal 与 timeout 区分、
collapse 门控清零等。测试只用 CPU torch，不依赖仿真。
"""
from dataclasses import fields
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from autotuner.taili_core.taili_reward import (
    REWARD_GROUP_NAMES,
    RewardConfig,
    adaptive_terrain_clearance_margin,
    compute_reward_components,
    duty_symmetry_score,
    group_reward_vector,
    huber_excess,
    normalized_underspeed_cost,
    settle_brake_potential_delta,
    settle_brake_signal,
    smooth_bounded_excess,
    sustained_task_credit_gate,
    support_structure_gate,
    tolerance_score,
    tracking_motion_quality,
    transition_failure_penalty,
    transition_core_fault,
    transition_motion_fault,
    transition_readiness_signal,
    transition_style_weight,
    transition_support_phase_ready,
    update_transition_failure_ema,
)


def test_transition_style_is_disabled_while_braking_and_smoothly_released():
    braking = torch.tensor([True, False, False, False])
    releasing = torch.tensor([False, True, True, False])
    progress = torch.tensor([0.8, 0.0, 0.5, 0.2])

    weight = transition_style_weight(braking, releasing, progress)

    assert torch.allclose(weight, torch.tensor([0.0, 0.0, 0.5, 1.0]))


def test_huber_excess_keeps_a_linear_tail_without_old_scale_explosion():
    values = torch.tensor([5.0, 35.0, 65.0])
    cost = huber_excess(values, free=5.0, scale=30.0)
    assert torch.allclose(cost, torch.tensor([0.0, 0.5, 1.5]))


def test_normalized_underspeed_has_equal_standing_pressure_at_every_speed():
    low = normalized_underspeed_cost(
        torch.tensor([0.0]), torch.tensor([0.2]), min_ratio=0.85
    )
    high = normalized_underspeed_cost(
        torch.tensor([0.0]), torch.tensor([0.8]), min_ratio=0.85
    )
    half = normalized_underspeed_cost(
        torch.tensor([0.34]), torch.tensor([0.8]), min_ratio=0.85
    )

    assert low.item() == pytest.approx(1.0)
    assert high.item() == pytest.approx(1.0)
    assert half.item() == pytest.approx(0.25)


def test_sustained_task_credit_requires_multiple_gait_cycles():
    gate = sustained_task_credit_gate(
        torch.tensor([0.0, 1.5, 2.25, 3.0]),
        floor=0.20,
        start_s=1.50,
        full_s=3.00,
    )
    assert torch.allclose(gate, torch.tensor([0.20, 0.20, 0.60, 1.00]))


def test_task_credit_is_diagnostic_and_does_not_scale_base_tracking():
    cfg = RewardConfig(
        w_tracking_lin=1.0,
        w_track_far=0.0,
        w_supported_progress=0.0,
        task_credit_floor=0.20,
        task_credit_start_s=1.50,
        task_credit_full_s=3.00,
    )
    early_fast = compute_reward_components(make_inp(
        1, task_credit_age_s=torch.zeros(1)
    ), cfg)
    mature_fast = compute_reward_components(make_inp(
        1, task_credit_age_s=torch.full((1,), 3.0)
    ), cfg)
    terrain_fast = compute_reward_components(make_inp(
        1,
        task_credit_age_s=torch.zeros(1),
        terrain_response=torch.ones(1),
    ), cfg)

    assert early_fast["task_credit_gate"].item() == pytest.approx(0.20)
    assert early_fast["tracking_lin"].item() == pytest.approx(
        mature_fast["tracking_lin"].item()
    )
    assert terrain_fast["task_credit_gate"].item() == pytest.approx(1.0)


def test_underspeed_cost_depends_on_signed_speed_not_posture():
    cfg = RewardConfig(
        w_linear_underspeed=1.0,
        linear_underspeed_min_ratio=0.85,
        w_yaw_underspeed=1.0,
        yaw_underspeed_min_ratio=0.75,
    )
    linear_clean = compute_reward_components(make_inp(
        1, base_lin_vel=torch.zeros(1, 3)
    ), cfg)
    linear_still_shaking = compute_reward_components(make_inp(
        1,
        base_lin_vel=torch.zeros(1, 3),
        base_ang_vel=torch.tensor([[1.5, 0.0, 0.0]]),
    ), cfg)
    linear_fast_shaking = compute_reward_components(make_inp(
        1,
        base_ang_vel=torch.tensor([[1.5, 0.0, 0.0]]),
    ), cfg)
    yaw_still = compute_reward_components(make_inp(
        1,
        cmd=torch.tensor([[0.0, 0.0, 0.6]]),
        base_lin_vel=torch.zeros(1, 3),
        base_ang_vel=torch.zeros(1, 3),
    ), cfg)
    yaw_fast_shaking = compute_reward_components(make_inp(
        1,
        cmd=torch.tensor([[0.0, 0.0, 0.6]]),
        base_lin_vel=torch.zeros(1, 3),
        base_ang_vel=torch.tensor([[1.5, 0.0, 0.6]]),
    ), cfg)

    assert linear_clean["linear_underspeed"].item() == pytest.approx(-1.0)
    assert linear_still_shaking["linear_underspeed"].item() == pytest.approx(-1.0)
    assert yaw_still["yaw_underspeed"].item() == pytest.approx(-1.0)
    assert linear_fast_shaking["linear_underspeed"].item() == pytest.approx(0.0)
    assert yaw_fast_shaking["yaw_underspeed"].item() == pytest.approx(0.0)


def test_adaptive_terrain_clearance_margin_is_small_and_height_aware():
    margin = adaptive_terrain_clearance_margin(
        torch.tensor([0.04, 0.18, 0.30]), minimum=0.02, gain=0.05, maximum=0.035
    )
    assert torch.allclose(margin, torch.tensor([0.022, 0.029, 0.035]))


def test_duty_symmetry_tolerance_is_the_half_score_error():
    error = torch.tensor([0.0, 0.125, 0.25, 0.50])
    score = duty_symmetry_score(error, tolerance=0.25)
    assert torch.allclose(score, torch.tensor([1.0, 0.75, 0.50, 0.0]))


def test_generic_tolerance_score_uses_the_same_semantics():
    assert torch.allclose(
        tolerance_score(torch.tensor([0.0, 0.22, 0.44]), tolerance=0.22),
        torch.tensor([1.0, 0.5, 0.0]),
    )


def test_settle_brake_potential_rewards_deceleration_and_penalizes_acceleration():
    previous = torch.tensor([0.6, 0.2, 0.05])
    current = torch.tensor([0.3, 0.4, 0.05])
    delta = settle_brake_potential_delta(previous, current)
    assert delta[0] > 0.0
    assert delta[1] < 0.0
    assert delta[2] == 0.0
    signal = settle_brake_signal(previous, current)
    assert signal[0] > 0.0
    assert signal[1] < 0.0
    assert signal[2] > 0.0


def test_transition_failure_penalty_is_a_one_step_event_cost():
    gate = torch.tensor([1.0, 0.5, 1.0])
    failed = torch.tensor([True, True, False])
    penalty = transition_failure_penalty(failed, 2.0, gate)
    assert torch.allclose(penalty, torch.tensor([-2.0, -1.0, 0.0]))


def test_transition_motion_fault_matches_stop_and_old_axis_semantics():
    fault = transition_motion_fault(
        torch.tensor([[0.3, 0.4], [0.3, 0.4], [0.3, 0.4]]),
        torch.tensor([0.24, 0.24, 0.24]),
        torch.tensor([0, 1, 2]),
        torch.tensor([True, False, False]),
        0.10,
        0.12,
    )
    assert torch.allclose(fault, torch.tensor([5.0, 4.0, 2.0]))


def test_transition_motion_fault_allows_a_smooth_reversal_zero_crossing():
    old = torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    fault = transition_motion_fault(
        torch.tensor([[0.3, 0.0], [-0.2, 0.0]]),
        torch.zeros(2),
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.bool),
        0.10,
        0.12,
        old_command=old,
    )
    assert torch.allclose(fault, torch.tensor([3.0, 0.0]))


def test_transition_readiness_signal_rewards_fault_reduction_and_near_ready_state():
    previous = torch.tensor([0.8, 0.2, 0.05])
    current = torch.tensor([0.5, 0.4, 0.05])
    signal = transition_readiness_signal(previous, current)
    assert signal[0] > 0.0
    assert signal[1] < 0.0
    assert signal[2] > 0.0
    steady_low = transition_readiness_signal(torch.tensor([0.1]), torch.tensor([0.1]))
    steady_high = transition_readiness_signal(torch.tensor([0.8]), torch.tensor([0.8]))
    assert steady_low > steady_high


def test_transition_core_fault_keeps_gradients_for_all_dimensions():
    terms = torch.tensor([[0.9, 0.5, 0.2]], requires_grad=True)
    fault = transition_core_fault(terms)
    fault.sum().backward()

    assert fault.item() == pytest.approx(0.5 * 0.9 + 0.5 * (0.9 + 0.5 + 0.2) / 3.0)
    assert torch.all(terms.grad > 0.0)


def test_transition_support_phase_requires_four_feet_at_exchange_phase():
    support = torch.tensor([4.0, 3.0, 4.0, 4.0, 4.0])
    phase = torch.tensor([0.02, 0.02, 0.48, 0.25, 1.02])
    ready = transition_support_phase_ready(support, phase, phase_window=0.05)
    assert ready.tolist() == [True, False, True, False, True]


def test_transition_failure_ema_uses_complete_event_batch():
    value, valid = update_transition_failure_ema(10, 2, 0.0, False, 0.95)
    assert valid is True
    assert value == 0.2

    value, valid = update_transition_failure_ema(10, 6, value, valid, 0.5)
    assert valid is True
    assert value == 0.4

    unchanged, unchanged_valid = update_transition_failure_ema(0, 0, value, valid, 0.5)
    assert unchanged == value
    assert unchanged_valid is True


def test_cycle_yaw_residual_only_constrains_linear_commands():
    cfg = RewardConfig(w_cycle_yaw_residual=1.0)
    energy = torch.full((1,), 0.25)
    linear = compute_reward_components(make_inp(1, cycle_yaw_residual_energy=energy), cfg)
    turning = compute_reward_components(make_inp(
        1,
        cmd=torch.tensor([[0.0, 0.0, 0.6]]),
        base_ang_vel=torch.tensor([[0.0, 0.0, 0.6]]),
        cycle_yaw_residual_energy=energy,
    ), cfg)
    assert linear["cycle_yaw_residual"].item() < 0.0
    assert turning["cycle_yaw_residual"].item() == 0.0


def test_touchdown_force_rate_only_penalizes_touchdown_feet():
    cfg = RewardConfig(w_touchdown_force_rate=1.0)
    force_rate = torch.ones(1, 4)
    no_touchdown = compute_reward_components(make_inp(
        1, touchdown_force_rate=force_rate, touchdown_mask=torch.zeros(1, 4)), cfg)
    touchdown = compute_reward_components(make_inp(
        1, touchdown_force_rate=force_rate,
        touchdown_mask=torch.tensor([[1.0, 0.0, 0.0, 0.0]])), cfg)
    assert no_touchdown["touchdown_force_rate"].item() == 0.0
    assert touchdown["touchdown_force_rate"].item() < 0.0


def test_touchdown_impact_uses_positive_downward_speed_magnitude():
    cfg = RewardConfig(w_landing_impact=1.0, w_landing_impact_late=0.0)
    mask = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    light = compute_reward_components(make_inp(
        1, touchdown_vz=torch.tensor([[0.05, 0.0, 0.0, 0.0]]), touchdown_mask=mask), cfg)
    heavy = compute_reward_components(make_inp(
        1, touchdown_vz=torch.tensor([[0.50, 0.0, 0.0, 0.0]]), touchdown_mask=mask), cfg)
    assert light["landing_impact"].item() == 0.0
    assert heavy["landing_impact"].item() < 0.0


def test_touchdown_impact_remains_monotonic_above_old_saturation_point():
    speed = torch.tensor([0.20, 0.50, 1.20, 1.80], requires_grad=True)
    cost = smooth_bounded_excess(speed, free=0.20, scale=0.75)
    assert cost[0].item() == 0.0
    assert cost[1] < cost[2] < cost[3] < 1.0
    cost[-1].backward()
    assert speed.grad[-1] > 0.0


def test_touchdown_worst_foot_tail_keeps_linear_pressure_on_heavy_impacts():
    cfg = RewardConfig(
        w_landing_impact=0.0,
        w_landing_impact_late=0.0,
        w_landing_impact_tail=1.0,
        w_landing_impact_tail_late=0.0,
        touchdown_vz_free=0.12,
        impact_speed_scale=0.30,
    )
    mask = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    costs = []
    for speed in (0.42, 0.72, 1.02):
        out = compute_reward_components(make_inp(
            1,
            touchdown_vz=torch.tensor([[speed, 0.0, 0.0, 0.0]]),
            touchdown_mask=mask,
            touchdown_persistence_scale=1.0,
        ), cfg)
        costs.append(-out["landing_impact"].item())
    assert costs[0] < costs[1] < costs[2]
    assert costs[2] - costs[1] == pytest.approx(costs[1] - costs[0], rel=1e-5)


def test_touchdown_persistence_scale_distributes_event_cost():
    cfg = RewardConfig(w_landing_impact=1.0, w_landing_impact_late=0.0)
    common = dict(
        touchdown_vz=torch.tensor([[0.60, 0.0, 0.0, 0.0]]),
        touchdown_mask=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
    instant = compute_reward_components(make_inp(1, touchdown_persistence_scale=1.0, **common), cfg)
    held = compute_reward_components(make_inp(1, touchdown_persistence_scale=0.30, **common), cfg)
    assert held["landing_impact"].item() == pytest.approx(0.30 * instant["landing_impact"].item())


def test_base_angular_acceleration_reports_platform_jerk():
    cfg = RewardConfig(w_base_ang_accel=1.0, base_ang_accel_free=1.0, base_ang_accel_scale=6.0)
    quiet = compute_reward_components(make_inp(1, base_ang_accel=torch.zeros(1, 3)), cfg)
    abrupt = compute_reward_components(make_inp(
        1, base_ang_accel=torch.tensor([[8.0, 0.0, 0.0]])
    ), cfg)
    assert quiet["base_ang_accel"].item() == 0.0
    assert abrupt["base_ang_accel"].item() < 0.0


def test_core_stability_is_not_weakened_by_quality_maturity_gate():
    cfg = RewardConfig(w_base_wxy=1.0, w_orient=1.0)
    common = dict(
        base_ang_vel=torch.tensor([[1.0, 0.0, 0.0]]),
        tilt_rel=torch.tensor([0.20]),
    )
    early = compute_reward_components(make_inp(1, quality_gate=torch.zeros(1), **common), cfg)
    mature = compute_reward_components(make_inp(1, quality_gate=torch.ones(1), **common), cfg)
    assert torch.allclose(early["base_wxy"], mature["base_wxy"])
    assert torch.allclose(early["orient"], mature["orient"])


def test_core_quality_rewards_progress_with_a_stable_platform():
    cfg = _reward_cfg_only(
        w_core_quality=1.0,
        core_progress_floor=0.20,
    )
    stopped = compute_reward_components(make_inp(
        1,
        base_lin_vel=torch.zeros(1, 3),
    ), cfg)
    tracked = compute_reward_components(make_inp(1), cfg)
    oscillating = compute_reward_components(make_inp(
        1,
        base_ang_vel=torch.tensor([[0.5, 0.4, 0.3]]),
        base_lin_vel=torch.tensor([[0.5, 0.0, 0.3]]),
        tilt_rel=torch.tensor([0.10]),
        base_h_above_terrain=torch.tensor([0.50]),
    ), cfg)

    assert 0.0 < stopped["core_quality"].item() < tracked["core_quality"].item()
    assert oscillating["core_quality"].item() < tracked["core_quality"].item()


def test_core_quality_does_not_treat_commanded_yaw_as_platform_shake():
    cfg = _reward_cfg_only(w_core_quality=1.0)
    common = dict(
        cmd=torch.tensor([[0.0, 0.0, 0.6]]),
        base_lin_vel=torch.zeros(1, 3),
    )
    commanded_yaw = compute_reward_components(make_inp(
        1,
        base_ang_vel=torch.tensor([[0.0, 0.0, 0.6]]),
        **common,
    ), cfg)
    yaw_with_roll = compute_reward_components(make_inp(
        1,
        base_ang_vel=torch.tensor([[0.4, 0.0, 0.6]]),
        **common,
    ), cfg)

    assert commanded_yaw["core_quality"].item() > yaw_with_roll["core_quality"].item()


def test_core_quality_relaxes_only_after_passive_terrain_response():
    cfg = _reward_cfg_only(w_core_quality=1.0, core_terrain_relief=0.70)
    flat = compute_reward_components(make_inp(1, terrain_response=torch.zeros(1)), cfg)
    terrain = compute_reward_components(make_inp(1, terrain_response=torch.ones(1)), cfg)

    assert terrain["core_quality"].item() == pytest.approx(0.30 * flat["core_quality"].item())


def test_core_quality_also_rewards_quiet_standing():
    cfg = _reward_cfg_only(
        w_core_quality=1.0,
        core_progress_floor=0.20,
    )
    quiet = compute_reward_components(make_inp(
        1,
        cmd=torch.zeros(1, 3),
        stand_gate=torch.ones(1),
        moving_gate=torch.zeros(1),
        base_lin_vel=torch.zeros(1, 3),
        base_ang_vel=torch.zeros(1, 3),
        tilt_rel=torch.zeros(1),
        base_h_above_terrain=torch.full((1,), cfg.nominal_base_h),
    ), cfg)
    wobble = compute_reward_components(make_inp(
        1,
        cmd=torch.zeros(1, 3),
        stand_gate=torch.ones(1),
        moving_gate=torch.zeros(1),
        base_lin_vel=torch.tensor([[0.0, 0.0, 0.0]]),
        base_ang_vel=torch.tensor([[0.6, 0.4, 0.0]]),
        tilt_rel=torch.tensor([0.10]),
        base_h_above_terrain=torch.full((1,), cfg.nominal_base_h - 0.05),
    ), cfg)

    assert quiet["core_quality"].item() > 0.0
    assert wobble["core_quality"].item() < quiet["core_quality"].item()


def test_core_quality_gives_standing_full_floor_credit():
    cfg = _reward_cfg_only(
        w_core_quality=1.0,
        core_progress_floor=0.20,
    )
    stand = compute_reward_components(make_inp(
        1,
        cmd=torch.zeros(1, 3),
        stand_gate=torch.ones(1),
        moving_gate=torch.zeros(1),
        base_lin_vel=torch.zeros(1, 3),
        base_ang_vel=torch.zeros(1, 3),
        tilt_rel=torch.zeros(1),
        base_h_above_terrain=torch.full((1,), cfg.nominal_base_h),
    ), cfg)
    move = compute_reward_components(make_inp(
        1,
        cmd=torch.zeros(1, 3),
        stand_gate=torch.zeros(1),
        moving_gate=torch.ones(1),
        base_lin_vel=torch.zeros(1, 3),
        base_ang_vel=torch.zeros(1, 3),
        tilt_rel=torch.zeros(1),
        base_h_above_terrain=torch.full((1,), cfg.nominal_base_h),
    ), cfg)

    assert stand["core_quality"].item() == pytest.approx(1.0, rel=1e-6)
    assert move["core_quality"].item() == pytest.approx(0.20, rel=1e-6)


def test_core_stability_costs_keep_gradient_beyond_old_saturation_points():
    cfg = RewardConfig(
        w_orient=1.0,
        orient_soft_rad=0.05,
        orient_scale_rad=0.20,
        w_base_wxy=1.0,
        w_flat_move_height=1.0,
        flat_move_height_target=0.52,
        flat_move_height_band=0.03,
    )
    moderate = compute_reward_components(make_inp(
        1,
        tilt_rel=torch.tensor([0.25]),
        base_ang_vel=torch.tensor([[1.0, 0.0, 0.0]]),
        base_h_above_terrain=torch.tensor([0.46]),
    ), cfg)
    severe = compute_reward_components(make_inp(
        1,
        tilt_rel=torch.tensor([0.55]),
        base_ang_vel=torch.tensor([[3.0, 0.0, 0.0]]),
        base_h_above_terrain=torch.tensor([0.40]),
    ), cfg)
    assert severe["orient"].item() < moderate["orient"].item()
    assert severe["base_wxy"].item() < moderate["base_wxy"].item()
    assert severe["flat_move_height"].item() < moderate["flat_move_height"].item()


def test_tracking_motion_quality_uses_the_worst_body_axis():
    cfg = RewardConfig(
        tracking_motion_floor=0.20,
        tracking_wxy_target=0.20,
        tracking_wxy_width=0.40,
        tracking_vz_target=0.10,
        tracking_vz_width=0.30,
    )
    healthy = tracking_motion_quality(
        torch.zeros(1, 3), torch.zeros(1, 3), cfg
    )
    falling = tracking_motion_quality(
        torch.zeros(1, 3), torch.tensor([[0.0, 0.0, -0.80]]), cfg
    )

    assert healthy.item() == pytest.approx(1.0)
    assert falling.item() == pytest.approx(0.20)


def test_base_vertical_velocity_keeps_a_tail_during_collapse():
    cfg = RewardConfig(
        w_base_vz=1.0,
        base_vz_free=0.05,
        base_vz_scale=0.75,
    )
    moderate = compute_reward_components(make_inp(
        1, base_lin_vel=torch.tensor([[0.5, 0.0, -0.80]])
    ), cfg)
    severe = compute_reward_components(make_inp(
        1, base_lin_vel=torch.tensor([[0.5, 0.0, -1.80]])
    ), cfg)

    assert severe["base_vz"].item() < moderate["base_vz"].item() < 0.0


def test_posture_gate_is_diagnostic_and_never_scales_direction_drive():
    cfg = RewardConfig(
        w_tracking_lin=1.0,
        w_track_far=1.0,
        w_supported_progress=1.0,
        w_terrain_progress=1.0,
        tracking_posture_floor=0.55,
    )
    healthy = compute_reward_components(make_inp(
        1,
        stable_motion_gate=torch.ones(1),
        terrain_response=torch.zeros(1),
    ), cfg)
    collapsed_flat = compute_reward_components(make_inp(
        1,
        stable_motion_gate=torch.zeros(1),
        terrain_response=torch.zeros(1),
    ), cfg)
    contacted_terrain = compute_reward_components(make_inp(
        1,
        stable_motion_gate=torch.zeros(1),
        terrain_response=torch.ones(1),
    ), cfg)

    assert collapsed_flat["tracking_posture_gate"].item() == pytest.approx(0.55)
    assert collapsed_flat["tracking_lin"].item() == pytest.approx(healthy["tracking_lin"].item())
    assert collapsed_flat["supported_progress"].item() == pytest.approx(healthy["supported_progress"].item())
    assert contacted_terrain["tracking_posture_gate"].item() == pytest.approx(1.0)
    assert contacted_terrain["supported_progress"].item() == pytest.approx(healthy["supported_progress"].item())
    assert contacted_terrain["terrain_progress"].item() > 0.0


def test_yaw_angular_acceleration_diagnostic_excludes_transition_acceleration():
    cfg = RewardConfig(w_base_ang_accel=1.0, base_ang_accel_free=1.0, base_ang_accel_scale=6.0)
    yaw_jerk = torch.tensor([[0.0, 0.0, 8.0]])
    steady = compute_reward_components(make_inp(
        1,
        base_ang_accel=yaw_jerk,
        foot_trajectory_scope=torch.ones(1),
    ), cfg)
    transition = compute_reward_components(make_inp(
        1,
        base_ang_accel=yaw_jerk,
        foot_trajectory_scope=torch.zeros(1),
    ), cfg)
    assert steady["base_ang_accel"].item() < 0.0
    assert transition["base_ang_accel"].item() == 0.0


def test_foot_trajectory_is_shared_but_backwards_receives_more_correction():
    cfg = RewardConfig(
        w_foot_trajectory=1.0,
        foot_trajectory_error_scale=0.35,
        foot_trajectory_back_scale=1.20,
    )
    common = dict(
        foot_trajectory_error=torch.tensor([0.35]),
        foot_trajectory_scope=torch.ones(1),
    )
    forward = compute_reward_components(make_inp(1, **common), cfg)
    backward = compute_reward_components(make_inp(
        1,
        cmd=torch.tensor([[-0.5, 0.0, 0.0]]),
        base_lin_vel=torch.tensor([[-0.5, 0.0, 0.0]]),
        **common,
    ), cfg)
    assert backward["foot_trajectory"].item() == pytest.approx(
        1.20 * forward["foot_trajectory"].item()
    )


def test_foot_trajectory_is_disabled_during_transition_and_relaxed_on_terrain():
    cfg = RewardConfig(w_foot_trajectory=1.0)
    common = dict(foot_trajectory_error=torch.ones(1))
    steady = compute_reward_components(make_inp(
        1, foot_trajectory_scope=torch.ones(1), terrain_response=torch.zeros(1), **common), cfg)
    transition = compute_reward_components(make_inp(
        1, foot_trajectory_scope=torch.zeros(1), terrain_response=torch.zeros(1), **common), cfg)
    terrain = compute_reward_components(make_inp(
        1, foot_trajectory_scope=torch.ones(1), terrain_response=torch.ones(1), **common), cfg)
    assert steady["foot_trajectory"].item() < terrain["foot_trajectory"].item() < 0.0
    assert transition["foot_trajectory"].item() == 0.0


def test_foot_trajectory_error_is_additive_not_a_direction_drive_gate():
    """轨迹质量负责纠偏，不能让尚未成型的方向同时失去任务驱动。"""
    cfg = RewardConfig(
        w_foot_trajectory=1.0,
        w_lateral_coordinated_progress=1.0,
        lateral_quality_floor=0.20,
    )
    trajectory = dict(foot_trajectory_scope=torch.ones(1))

    yaw_common = dict(
        n=1,
        cmd=torch.tensor([[0.0, 0.0, 0.6]]),
        base_ang_vel=torch.tensor([[0.0, 0.0, 0.6]]),
        **trajectory,
    )
    yaw_clean = compute_reward_components(
        make_inp(foot_trajectory_error=torch.zeros(1), **yaw_common), cfg
    )
    yaw_bad = compute_reward_components(
        make_inp(foot_trajectory_error=torch.full((1,), 4.0), **yaw_common), cfg
    )
    assert yaw_bad["tracking_yaw"].item() == pytest.approx(yaw_clean["tracking_yaw"].item())
    assert yaw_bad["yaw_progress"].item() == pytest.approx(yaw_clean["yaw_progress"].item())
    assert yaw_bad["foot_trajectory"].item() < yaw_clean["foot_trajectory"].item()

    lateral_common = dict(
        n=1,
        cmd=torch.tensor([[0.0, 0.3, 0.0]]),
        base_lin_vel=torch.tensor([[0.0, 0.3, 0.0]]),
        duty_by_leg_window=torch.full((1, 4), 0.5),
        support_force_by_leg_window=torch.full((1, 4), 100.0),
        **trajectory,
    )
    lateral_clean = compute_reward_components(
        make_inp(foot_trajectory_error=torch.zeros(1), **lateral_common), cfg
    )
    lateral_bad = compute_reward_components(
        make_inp(foot_trajectory_error=torch.full((1,), 4.0), **lateral_common), cfg
    )
    assert lateral_bad["lateral_coordinated_progress"].item() == pytest.approx(
        lateral_clean["lateral_coordinated_progress"].item()
    )
    assert lateral_bad["foot_trajectory"].item() < lateral_clean["foot_trajectory"].item()


def test_swing_action_accel_does_not_penalize_support_joints():
    cfg = RewardConfig(w_swing_action_accel=1.0)
    accel = torch.ones(1, 12)
    support = compute_reward_components(make_inp(
        1, swing_action_accel=accel, swing_joint_mask=torch.zeros(1, 12)), cfg)
    swing = compute_reward_components(make_inp(
        1, swing_action_accel=accel, swing_joint_mask=torch.ones(1, 12)), cfg)
    assert support["swing_action_accel"].item() == 0.0
    assert swing["swing_action_accel"].item() < 0.0


def test_swing_foot_velocity_delta_penalizes_only_swing_feet_and_relaxes_on_terrain():
    cfg = RewardConfig(w_swing_action_accel=1.0, swing_foot_velocity_delta_scale=0.8)
    common = dict(
        swing_action_accel=torch.zeros(1, 12),
        swing_joint_mask=torch.ones(1, 12),
        swing_foot_velocity_delta=torch.ones(1, 4),
    )
    support = compute_reward_components(make_inp(
        1, **common, swing_foot_mask=torch.zeros(1, 4)), cfg)
    flat = compute_reward_components(make_inp(
        1, **common, swing_foot_mask=torch.ones(1, 4)), cfg)
    terrain = compute_reward_components(make_inp(
        1, **common, swing_foot_mask=torch.ones(1, 4), terrain_response=torch.ones(1)), cfg)
    assert support["swing_action_accel"].item() == 0.0
    assert flat["swing_action_accel"].item() < terrain["swing_action_accel"].item() < 0.0


def test_swing_velocity_smoothing_keeps_a_non_saturated_tail():
    cfg = RewardConfig(w_swing_action_accel=1.0, swing_foot_velocity_delta_scale=0.6)
    common = dict(
        swing_action_accel=torch.zeros(1, 12),
        swing_joint_mask=torch.ones(1, 12),
        swing_foot_mask=torch.ones(1, 4),
    )
    moderate = compute_reward_components(make_inp(
        1, swing_foot_velocity_delta=torch.ones(1, 4), **common
    ), cfg)
    severe = compute_reward_components(make_inp(
        1, swing_foot_velocity_delta=torch.full((1, 4), 3.0), **common
    ), cfg)
    assert severe["swing_action_accel"].item() < moderate["swing_action_accel"].item() < 0.0


def test_terminal_swing_velocity_is_an_additive_contact_quality_cost():
    cfg = RewardConfig(w_terminal_swing_velocity=1.0)
    clean = compute_reward_components(make_inp(
        1, terminal_swing_velocity_cost=torch.zeros(1)
    ), cfg)
    heavy = compute_reward_components(make_inp(
        1, terminal_swing_velocity_cost=torch.ones(1)
    ), cfg)
    assert clean["terminal_swing_velocity"].item() == 0.0
    assert heavy["terminal_swing_velocity"].item() < 0.0


def make_inp(n=2, **over):
    """构造一个合理的前进行走样本：0.5 m/s 前进命令、直立、对角小跑。"""
    z = torch.zeros(n)
    o = torch.ones(n)
    d = dict(
        cmd=torch.tensor([[0.5, 0.0, 0.0]]).repeat(n, 1),
        base_lin_vel=torch.tensor([[0.5, 0.0, 0.0]]).repeat(n, 1),
        base_ang_vel=torch.zeros(n, 3),
        stable_motion_gate=o.clone(),
        stand_gate=z.clone(),          # 运动状态，不是站立状态。
        moving_gate=o.clone(),
        quality_gate=o.clone(),
        action=torch.zeros(n, 12),
        last_action=torch.zeros(n, 12),
        default_pose_error=z.clone(),
        foot_contact=torch.tensor([[1.0, 0.0, 0.0, 1.0]]).repeat(n, 1),  # 对角 FL+RR 触地。
        local_obstacle_h=z.clone(),
        terrain_response=z.clone(),
        foot_clearance=torch.tensor([[0.0, 0.08, 0.08, 0.0]]).repeat(n, 1),
        foot_vel_xy=torch.zeros(n, 4),   # 没有稳定窗口输入时的滑移兜底路径。
        desired_foot_contact=torch.tensor([[1.0, 0.0, 0.0, 1.0]]).repeat(n, 1),
        touchdown_vz=torch.zeros(n, 4),
        tilt_rel=z.clone(),
        torque=torch.zeros(n, 12),
        torque_limit=torch.full((12,), 100.0),
        torque_clamped=torch.zeros(n, 12),
        terminal_reason=None,
    )
    d.update(over)
    return SimpleNamespace(**d)


def _total(inp, cfg=None):
    cfg = cfg or RewardConfig()
    return compute_reward_components(inp, cfg)["total"]


def _reward_cfg_only(**overrides):
    """构造只启用测试显式列出权重的奖励配置。"""
    cfg = RewardConfig()
    for field in fields(RewardConfig):
        if field.name.startswith("w_"):
            setattr(cfg, field.name, 0.0)
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg


def test_compiled_dynamic_mechanism_enters_production_reward_and_diagnostics(tmp_path, monkeypatch):
    from autotuner.mechanisms.mechanism_runtime import clear_runtime_cache
    from autotuner.mechanisms.mechanism_specs import (
        Expression,
        GateSpec,
        MechanismBundle,
        MetricSpec,
        RewardTermSpec,
        SignalSpec,
    )

    cfg = _reward_cfg_only(w_tracking_lin=1.0)
    monkeypatch.delenv("TAILI_MECHANISM_BUNDLE", raising=False)
    clear_runtime_cache()
    baseline = compute_reward_components(make_inp(2), cfg)

    bundle = MechanismBundle(
        id="bundle:production-hook",
        status="approved",
        contract_ref="contract:test",
        signals=(SignalSpec(
            name="existing.tracking", description="built-in tracking component",
            source_ref="component.tracking_lin",
        ),),
        rewards=(RewardTermSpec(
            id="reward:tracking-bonus", name="tracking bonus", role="positive_drive",
            expression=Expression.signal("existing.tracking"), weight=0.25,
            reward_group="track", intended_effect="strengthen tracking",
            failure_region="tracking is weak", success_region="tracking is strong",
        ),),
        metrics=(MetricSpec(
            id="metric:tracking-live", name="tracking live",
            expression=Expression.signal("existing.tracking"), intended_reading="higher is better",
        ),),
        gates=(GateSpec(
            id="gate:tracking-live", name="tracking live gate", metric_ref="metric:tracking-live",
            comparator="ge", threshold=0.5, action="advance", scope="curriculum",
            rationale="require live tracking",
        ),),
        evaluator_refs=("evaluator:test",),
    )
    path = tmp_path / "mechanisms.json"
    path.write_text(bundle.model_dump_json(), encoding="utf-8")
    monkeypatch.setenv("TAILI_MECHANISM_BUNDLE", str(path))
    clear_runtime_cache()

    result = compute_reward_components(make_inp(2), cfg)
    dynamic = result["dynamic/reward:tracking-bonus"]
    assert torch.allclose(dynamic, 0.25 * baseline["tracking_lin"])
    assert torch.allclose(result["total"], baseline["total"] + dynamic)
    assert result["_dynamic_metrics"]["metric:tracking-live"] == pytest.approx(1.0)
    assert result["_dynamic_gates"]["gate:tracking-live"]["passed"] is True
    grouped = group_reward_vector(result)
    track_index = REWARD_GROUP_NAMES.index("track")
    assert grouped[:, track_index].sum() >= dynamic.sum()


def test_base_linear_drive_is_independent_of_posture_and_command_age():
    cfg = _reward_cfg_only(
        w_tracking_lin=2.0,
        w_track_far=1.0,
        w_supported_progress=0.75,
    )
    clean = compute_reward_components(make_inp(
        1,
        task_credit_age_s=torch.full((1,), 3.0),
    ), cfg)
    unstable_early = compute_reward_components(make_inp(
        1,
        stable_motion_gate=torch.zeros(1),
        task_credit_age_s=torch.zeros(1),
        base_lin_vel=torch.tensor([[0.5, 0.0, 0.35]]),
        base_ang_vel=torch.tensor([[1.5, 0.0, 0.0]]),
    ), cfg)

    for name in ("tracking_lin", "tracking_lin_far", "supported_progress"):
        assert unstable_early[name].item() == pytest.approx(clean[name].item())


def test_base_yaw_drive_is_independent_of_posture_and_command_age():
    cfg = _reward_cfg_only(
        w_tracking_yaw=2.0,
        w_yaw_far=1.0,
        w_yaw_progress=0.75,
    )
    common = dict(
        cmd=torch.tensor([[0.0, 0.0, 0.6]]),
        base_lin_vel=torch.zeros(1, 3),
    )
    clean = compute_reward_components(make_inp(
        1,
        base_ang_vel=torch.tensor([[0.0, 0.0, 0.6]]),
        task_credit_age_s=torch.full((1,), 3.0),
        **common,
    ), cfg)
    unstable_early = compute_reward_components(make_inp(
        1,
        stable_motion_gate=torch.zeros(1),
        base_ang_vel=torch.tensor([[1.5, 0.0, 0.6]]),
        task_credit_age_s=torch.zeros(1),
        **common,
    ), cfg)

    for name in ("tracking_yaw", "tracking_yaw_far", "yaw_progress"):
        assert unstable_early[name].item() == pytest.approx(clean[name].item())


def test_task_return_orders_capability_before_failure():
    cfg = _reward_cfg_only(
        w_tracking_lin=2.0,
        w_track_far=1.0,
        w_supported_progress=0.75,
        w_orient=0.8,
        w_base_vz=0.8,
        w_base_wxy=0.8,
        w_terminal=8.0,
    )
    stable = _total(make_inp(1), cfg).item()
    unstable = _total(make_inp(
        1,
        tilt_rel=torch.tensor([0.15]),
        base_lin_vel=torch.tensor([[0.5, 0.0, 0.20]]),
        base_ang_vel=torch.tensor([[0.8, 0.0, 0.0]]),
    ), cfg).item()
    slow = _total(make_inp(
        1, base_lin_vel=torch.tensor([[0.20, 0.0, 0.0]])
    ), cfg).item()
    stopped = _total(make_inp(
        1, base_lin_vel=torch.zeros(1, 3)
    ), cfg).item()
    wrong = _total(make_inp(
        1, base_lin_vel=torch.tensor([[-0.20, 0.0, 0.0]])
    ), cfg).item()
    crashed = _total(make_inp(
        1,
        hard_survival_gate=torch.zeros(1),
        terminal_reason="fall",
    ), cfg).item()

    assert stable > unstable > slow > stopped > wrong > crashed


def test_reduced_speed_terrain_progress_beats_stopping_without_terrain_bonus():
    cfg = _reward_cfg_only(
        w_tracking_lin=2.0,
        w_track_far=1.0,
        w_supported_progress=0.75,
        w_terrain_progress=0.0,
    )
    common = dict(
        terrain_response=torch.ones(1),
        cmd=torch.tensor([[0.5, 0.0, 0.0]]),
    )
    moving = _total(make_inp(
        1, base_lin_vel=torch.tensor([[0.15, 0.0, 0.0]]), **common
    ), cfg)
    stopped = _total(make_inp(
        1, base_lin_vel=torch.zeros(1, 3), **common
    ), cfg)

    assert moving.item() > stopped.item()


def test_base_angular_acceleration_is_diagnostic_only():
    cfg = _reward_cfg_only(
        w_base_ang_accel=4.0,
        w_base_ang_accel_raw_tail=4.0,
    )
    clean = compute_reward_components(make_inp(
        1,
        base_ang_accel=torch.zeros(1, 3),
        base_ang_accel_raw=torch.zeros(1, 3),
    ), cfg)
    jerky = compute_reward_components(make_inp(
        1,
        base_ang_accel=torch.full((1, 3), 30.0),
        base_ang_accel_raw=torch.full((1, 3), 60.0),
    ), cfg)

    assert jerky["base_ang_accel"].item() < clean["base_ang_accel"].item()
    assert jerky["total"].item() == pytest.approx(clean["total"].item())


# ── #1 跟踪：命中命令必须比偏离命令得分更高 ───────────────────────────────
def test_perfect_tracking_beats_miss():
    cfg = RewardConfig()
    good = compute_reward_components(make_inp(), cfg)
    miss = compute_reward_components(make_inp(base_lin_vel=torch.tensor([[0.0, 0.0, 0.0]]).repeat(2, 1)), cfg)
    assert good["tracking_lin"].mean() > miss["tracking_lin"].mean()


def test_direction_progress_has_gradient_below_legacy_speed_floor():
    cfg = RewardConfig(
        w_supported_progress=1.0,
        w_terrain_progress=0.0,
        healthy_progress_floor=1.0,
    )
    slow = compute_reward_components(make_inp(
        1, base_lin_vel=torch.tensor([[0.02, 0.0, 0.0]])
    ), cfg)
    stopped = compute_reward_components(make_inp(
        1, base_lin_vel=torch.zeros(1, 3)
    ), cfg)
    assert slow["supported_progress"].item() > stopped["supported_progress"].item()


def test_direction_progress_reaches_full_credit_at_sixty_percent_speed():
    cfg = RewardConfig(
        w_supported_progress=1.0,
        w_terrain_progress=0.0,
        direction_progress_full_ratio=0.60,
    )
    cmd = torch.tensor([[0.50, 0.0, 0.0]])
    half = compute_reward_components(make_inp(
        1, cmd=cmd, base_lin_vel=torch.tensor([[0.15, 0.0, 0.0]])
    ), cfg)
    full = compute_reward_components(make_inp(
        1, cmd=cmd, base_lin_vel=torch.tensor([[0.30, 0.0, 0.0]])
    ), cfg)
    off_axis = compute_reward_components(make_inp(
        1, cmd=cmd, base_lin_vel=torch.tensor([[0.30, 0.30, 0.0]])
    ), cfg)

    assert half["supported_progress"].item() == pytest.approx(0.5)
    assert full["supported_progress"].item() == pytest.approx(1.0)
    assert off_axis["supported_progress"].item() < full["supported_progress"].item()


def test_terrain_direction_progress_uses_its_own_low_speed_target():
    cfg = RewardConfig(
        w_supported_progress=1.0,
        w_terrain_progress=1.0,
        direction_progress_full_ratio=0.60,
        terrain_progress_full_ratio=0.35,
    )
    result = compute_reward_components(make_inp(
        1,
        cmd=torch.tensor([[0.50, 0.0, 0.0]]),
        base_lin_vel=torch.tensor([[0.175, 0.0, 0.0]]),
        terrain_response=torch.ones(1),
    ), cfg)

    assert result["direction_progress_ratio"].item() < 1.0
    assert result["terrain_direction_progress_ratio"].item() == pytest.approx(1.0)
    assert result["terrain_progress"].item() > 0.0


def test_terrain_direction_drive_keeps_a_gradient_at_collision_stop():
    cfg = RewardConfig(
        w_supported_progress=0.0,
        w_terrain_progress=1.0,
        terrain_direction_alignment_floor=0.35,
    )
    velocity = torch.zeros(1, 3, requires_grad=True)
    result = compute_reward_components(make_inp(
        1,
        base_lin_vel=velocity,
        terrain_response=torch.ones(1),
    ), cfg)
    result["terrain_progress"].sum().backward()

    assert velocity.grad is not None
    assert velocity.grad[0, 0].item() > 0.0


def test_passive_terrain_response_separates_general_and_terrain_progress():
    cfg = RewardConfig(
        w_supported_progress=1.0,
        w_terrain_progress=1.0,
        healthy_progress_floor=1.0,
    )
    flat = compute_reward_components(make_inp(1, terrain_response=torch.zeros(1)), cfg)
    terrain = compute_reward_components(make_inp(1, terrain_response=torch.ones(1)), cfg)
    assert flat["supported_progress"].item() > 0.0
    assert terrain["supported_progress"].item() > 0.0
    assert flat["terrain_progress"].item() == 0.0
    assert terrain["terrain_progress"].item() > 0.0


def test_flat_direction_drive_is_independent_from_additive_core_quality():
    cfg = RewardConfig(
        w_supported_progress=1.0,
        w_terrain_progress=1.0,
        direction_progress_support_floor=0.0,
        direction_progress_quality_floor=0.0,
    )
    common = dict(
        n=1,
        cmd=torch.tensor([[0.5, 0.0, 0.0]]),
        base_lin_vel=torch.tensor([[0.3, 0.0, 0.0]]),
    )
    controlled = compute_reward_components(make_inp(
        base_ang_vel=torch.zeros(1, 3),
        terrain_response=torch.zeros(1),
        **common,
    ), cfg)
    oscillating = compute_reward_components(make_inp(
        base_ang_vel=torch.tensor([[0.8, 0.0, 0.8]]),
        terrain_response=torch.zeros(1),
        **common,
    ), cfg)
    terrain = compute_reward_components(make_inp(
        base_ang_vel=torch.tensor([[0.8, 0.0, 0.8]]),
        terrain_response=torch.ones(1),
        **common,
    ), cfg)

    assert controlled["supported_progress"].item() > 0.0
    assert oscillating["direction_progress_quality_gate"].item() < controlled["direction_progress_quality_gate"].item()
    assert oscillating["supported_progress"].item() == pytest.approx(controlled["supported_progress"].item())
    assert terrain["supported_progress"].item() == pytest.approx(controlled["supported_progress"].item())
    assert controlled["total"].item() > oscillating["total"].item()
    assert terrain["terrain_progress"].item() > 0.0


def test_mixed_linear_progress_does_not_duplicate_yaw_tracking_quality():
    cfg = RewardConfig(
        w_supported_progress=1.0,
        direction_progress_support_floor=0.0,
        direction_progress_quality_floor=0.0,
    )
    common = dict(
        n=1,
        cmd=torch.tensor([[0.5, 0.0, 0.3]]),
        base_lin_vel=torch.tensor([[0.3, 0.0, 0.0]]),
    )
    matched = compute_reward_components(make_inp(
        base_ang_vel=torch.tensor([[0.0, 0.0, 0.3]]),
        **common,
    ), cfg)
    mismatched = compute_reward_components(make_inp(
        base_ang_vel=torch.tensor([[0.0, 0.0, 0.9]]),
        **common,
    ), cfg)
    assert matched["supported_progress"].item() > 0.0
    assert mismatched["supported_progress"].item() == pytest.approx(matched["supported_progress"].item())
    assert matched["tracking_yaw"].item() > mismatched["tracking_yaw"].item()


def test_terrain_direction_credit_does_not_wait_for_clearance_quality():
    cfg = RewardConfig(w_supported_progress=0.0, w_terrain_progress=1.0)
    common = dict(
        terrain_response=torch.ones(1),
        terrain_clearance_response=torch.ones(1),
        terrain_clearance_foot_mask=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        terrain_clearance_ramp=torch.ones(1),
        terrain_event_direction=torch.ones(1),
        terrain_probe_target=torch.tensor([0.20]),
    )
    blocked = compute_reward_components(make_inp(
        1, foot_clearance=torch.zeros(1, 4), **common
    ), cfg)
    cleared = compute_reward_components(make_inp(
        1, foot_clearance=torch.tensor([[0.20, 0.0, 0.0, 0.0]]), **common
    ), cfg)
    assert blocked["terrain_clearance_success"].item() < cleared["terrain_clearance_success"].item()
    assert torch.allclose(blocked["terrain_progress"], cleared["terrain_progress"])


def test_no_stand_still_trap_under_forward_command():
    # 前进命令下站着不动不能比真实前进得分更高。
    cfg = RewardConfig()
    walking = _total(make_inp())
    standing = _total(make_inp(
        base_lin_vel=torch.zeros(2, 3),
        foot_contact=torch.ones(2, 4),
        foot_clearance=torch.zeros(2, 4),
        desired_foot_contact=torch.ones(2, 4),
    ))
    assert walking.mean() > standing.mean()


# ── #8 反向运动压力：与命令相反的运动必须被主动惩罚 ─────────────────────
def test_wrong_direction_is_penalized():
    cfg = RewardConfig()
    backward_under_fwd_cmd = make_inp(base_lin_vel=torch.tensor([[-0.5, 0.0, 0.0]]).repeat(2, 1))
    comp = compute_reward_components(backward_under_fwd_cmd, cfg)
    assert comp["wrong_dir"].mean() < 0.0


def test_off_axis_penalizes_drift_under_stand_command():
    # off_axis 不受 moving_gate 控制，因此站立横漂也要被惩罚。
    cfg = RewardConfig()
    inp = make_inp(
        cmd=torch.zeros(2, 3),
        base_lin_vel=torch.tensor([[0.0, 0.3, 0.0]]).repeat(2, 1),  # 侧向漂移。
        stand_gate=torch.ones(2),
        moving_gate=torch.zeros(2),
    )
    comp = compute_reward_components(inp, cfg)
    assert comp["off_axis"].mean() < 0.0


def test_off_axis_keeps_a_linear_tail_for_sideways_collapse_velocity():
    cfg = RewardConfig(
        w_off_axis=1.0,
        off_axis_free=0.03,
        off_axis_scale=0.25,
    )
    costs = []
    for lateral_speed in (0.30, 0.80, 1.30):
        out = compute_reward_components(make_inp(
            1,
            base_lin_vel=torch.tensor([[0.5, lateral_speed, 0.0]]),
        ), cfg)
        costs.append(-out["off_axis"].item())

    assert costs[0] < costs[1] < costs[2]
    assert costs[2] - costs[1] == pytest.approx(
        costs[1] - costs[0], rel=1e-5
    )


# ── 3b 规约策略：按机器人均值规约，避免随关节数无界放大 ─────────────────
def test_torque_margin_is_mean_not_sum():
    cfg = RewardConfig()
    # 一个关节超过 0.85 margin 和十二个关节都超过相比，原始求和会放大约 12 倍。
    tq_one = torch.zeros(1, 12); tq_one[0, 0] = 95.0    # 使用率 0.95 > 0.85。
    tq_all = torch.full((1, 12), 95.0)
    lim = torch.full((12,), 100.0)
    c_one = compute_reward_components(make_inp(1, torque=tq_one, torque_limit=lim), cfg)
    c_all = compute_reward_components(make_inp(1, torque=tq_all, torque_limit=lim), cfg)
    # 均值规约后，全十二关节惩罚约为单关节的 12 倍，但不会无界放大。
    ratio = float(c_all["torque_margin"].mean() / c_one["torque_margin"].mean())
    assert 11.0 < ratio < 13.0


# ── #4 terminal 与 timeout 必须区分 ─────────────────────────────────────
def test_timeout_not_penalized_but_terminal_is():
    cfg = RewardConfig()
    timeout = compute_reward_components(make_inp(terminal_reason="timeout"), cfg)
    fell = compute_reward_components(make_inp(terminal_reason="base_contact"), cfg)
    assert float(timeout["terminal_penalty"].mean()) == 0.0
    assert float(fell["terminal_penalty"].mean()) == -cfg.w_terminal


# ── 3a 门控：软失稳保留恢复梯度，真实终止才清零任务项 ───────────────────
def test_hard_survival_gate_zeroes_tracking_but_soft_instability_keeps_a_floor():
    cfg = RewardConfig()
    recoverable = compute_reward_components(make_inp(
        stable_motion_gate=torch.zeros(2),
        hard_survival_gate=torch.ones(2),
    ), cfg)
    terminal = compute_reward_components(make_inp(
        stable_motion_gate=torch.zeros(2),
        hard_survival_gate=torch.zeros(2),
    ), cfg)
    assert float(recoverable["tracking_lin"].mean()) > 0.0
    assert float(terminal["tracking_lin"].mean()) == 0.0
    assert float(terminal["tracking_yaw"].mean()) == 0.0
    assert float(terminal["yaw_progress"].mean()) == 0.0
    assert float(terminal["stand"].mean()) == 0.0


def test_preterminal_instability_keeps_safety_costs_but_terminal_closes_them():
    # 软失稳阶段必须持续收到安全梯度；只有真实 terminal 才关闭这些代价。
    cfg = RewardConfig(
        w_supported_progress=2.0,
        w_flat_move_height=1.0,
        flat_move_height_target=0.52,
        flat_move_height_band=0.03,
    )
    common = dict(
        stable_motion_gate=torch.zeros(2),
        quality_gate=torch.full((2,), 0.25),
        refinement_gate=torch.zeros(2),
        base_lin_vel=torch.tensor([[0.5, 0.0, -2.0]]).repeat(2, 1),   # 保持前进，同时大幅下沉。
        base_ang_vel=torch.tensor([[3.0, 3.0, 0.0]]).repeat(2, 1),    # 大幅滚转/俯仰角速度。
        base_h_above_terrain=torch.full((2,), cfg.h_gate_close - 0.02),
        hip_deviation=torch.full((2,), 1.0),
    )
    recoverable = compute_reward_components(make_inp(
        **common,
        hard_survival_gate=torch.ones(2),
    ), cfg)
    terminal = compute_reward_components(make_inp(
        **common,
        hard_survival_gate=torch.zeros(2),
    ), cfg)

    assert float(recoverable["base_vz"].mean()) < 0.0
    assert float(recoverable["base_wxy"].mean()) < 0.0
    assert float(recoverable["flat_move_height"].mean()) < 0.0
    assert float(recoverable["supported_progress"].mean()) > 0.0
    assert float(recoverable["tracking_lin"].mean()) > 0.0
    assert float(terminal["base_vz"].mean()) == 0.0
    assert float(terminal["base_wxy"].mean()) == 0.0
    assert float(terminal["flat_move_height"].mean()) == 0.0
    assert float(terminal["supported_progress"].mean()) == 0.0
    # 髋关节形态仍是精修项，不负责倒地前的基础安全。
    assert float(recoverable["hip_deviation"].mean()) == 0.0


def test_healthy_tracking_strictly_beats_sinking_at_equal_direction_speed():
    # 方向驱动相同，但健康通过必须在数学上严格优于“短暂冲速后下沉”。
    cfg = RewardConfig(
        w_tracking_lin=2.25,
        w_track_far=1.0,
        w_supported_progress=2.0,
        w_support_integrity=2.4,
        w_flat_move_height=1.0,
        w_base_vz=1.6,
        w_base_wxy=1.8,
        flat_move_height_target=0.52,
        flat_move_height_band=0.03,
    )
    common = dict(
        quality_gate=torch.tensor([0.25]),
        refinement_gate=torch.tensor([0.08]),
        hard_survival_gate=torch.ones(1),
    )
    healthy = compute_reward_components(make_inp(
        1,
        base_h_above_terrain=torch.tensor([0.52]),
        **common,
    ), cfg)
    sinking = compute_reward_components(make_inp(
        1,
        stable_motion_gate=torch.zeros(1),
        base_h_above_terrain=torch.tensor([0.35]),
        base_lin_vel=torch.tensor([[0.5, 0.0, -0.8]]),
        base_ang_vel=torch.tensor([[0.0, 1.6, 0.0]]),
        **common,
    ), cfg)

    assert healthy["supported_progress"].item() > 0.0
    assert sinking["supported_progress"].item() == pytest.approx(healthy["supported_progress"].item())
    assert healthy["total"].item() > sinking["total"].item() + 2.0


def test_support_integrity_penalizes_low_body_under_motion():
    cfg = RewardConfig(w_support_integrity=3.0)
    low = compute_reward_components(make_inp(
        base_h_above_terrain=torch.full((2,), cfg.h_gate_close - 0.02),
        foot_contact=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
    ), cfg)
    ok = compute_reward_components(make_inp(base_h_above_terrain=torch.full((2,), cfg.nominal_base_h)), cfg)

    assert low["support_integrity"].mean() < 0.0
    assert ok["support_integrity"].mean() == 0.0
    assert low["total"].mean() < ok["total"].mean()


def test_support_structure_requires_front_and_rear_for_forward_motion():
    cfg = RewardConfig(w_support_integrity=2.0)
    good = compute_reward_components(make_inp(
        foot_contact=torch.tensor([[1.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
        desired_foot_contact=torch.tensor([[1.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
        diagonal_pair_window=torch.ones(2),
        duty_quality_window=torch.ones(2),
        stance_slip_high_fraction=torch.zeros(2),
    ), cfg)
    rear_only = compute_reward_components(make_inp(
        foot_contact=torch.tensor([[0.0, 0.0, 1.0, 1.0]]).repeat(2, 1),
        desired_foot_contact=torch.tensor([[0.0, 0.0, 1.0, 1.0]]).repeat(2, 1),
        diagonal_pair_window=torch.ones(2),
        duty_quality_window=torch.ones(2),
        stance_slip_high_fraction=torch.zeros(2),
    ), cfg)

    assert torch.allclose(good["support_structure_gate"], torch.ones(2))
    assert torch.allclose(rear_only["support_structure_gate"], torch.zeros(2))
    assert rear_only["support_integrity"].mean() < good["support_integrity"].mean()
    assert torch.allclose(rear_only["tracking_lin"], good["tracking_lin"])


def test_support_structure_does_not_reintroduce_yaw_linear_validator():
    cmd = torch.tensor([[0.0, 0.0, 0.7]])
    two_front_feet = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    gate = support_structure_gate(two_front_feet, cmd)
    assert torch.allclose(gate, torch.ones(1))


def test_gait_anchor_requires_support_quality():
    cfg = RewardConfig(w_gait_anchor=1.0)
    good = compute_reward_components(make_inp(
        base_h_above_terrain=torch.full((2,), cfg.nominal_base_h),
        foot_contact=torch.tensor([[1.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
        desired_foot_contact=torch.tensor([[1.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
    ), cfg)
    low_body = compute_reward_components(make_inp(
        base_h_above_terrain=torch.full((2,), cfg.h_gate_close - 0.02),
        foot_contact=torch.tensor([[1.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
        desired_foot_contact=torch.tensor([[1.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
    ), cfg)

    assert good["gait_anchor"].mean() > low_body["gait_anchor"].mean()
    assert float(low_body["gait_anchor"].mean()) == 0.0


def test_gait_phase_mismatch_pushes_wrong_contact_out_of_zero_gradient_region():
    cfg = RewardConfig(w_gait_anchor=0.0, w_gait_phase_mismatch=1.0)
    matched = compute_reward_components(make_inp(
        1,
        foot_contact=torch.tensor([[1.0, 0.0, 0.0, 1.0]]),
        desired_foot_contact=torch.tensor([[1.0, 0.0, 0.0, 1.0]]),
    ), cfg)
    mismatched = compute_reward_components(make_inp(
        1,
        foot_contact=torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
        desired_foot_contact=torch.tensor([[1.0, 0.0, 0.0, 1.0]]),
    ), cfg)

    assert matched["gait_phase_mismatch"].item() == 0.0
    assert mismatched["gait_phase_mismatch"].item() < 0.0


def test_contact_period_penalty_requires_a_valid_measured_cycle():
    cfg = RewardConfig(
        w_contact_period=1.0,
        contact_period_free=0.05,
        contact_period_scale=0.15,
    )
    matched = compute_reward_components(make_inp(
        1,
        contact_period_error_window=torch.zeros(1),
        contact_period_valid_window=torch.ones(1),
    ), cfg)
    too_fast = compute_reward_components(make_inp(
        1,
        contact_period_error_window=torch.full((1,), 0.25),
        contact_period_valid_window=torch.ones(1),
    ), cfg)
    not_measured = compute_reward_components(make_inp(
        1,
        contact_period_error_window=torch.full((1,), 0.25),
        contact_period_valid_window=torch.zeros(1),
    ), cfg)

    assert matched["contact_period"].item() == 0.0
    assert too_fast["contact_period"].item() < 0.0
    assert not_measured["contact_period"].item() == 0.0


def test_action_magnitude_is_a_flat_soft_limit_with_passive_terrain_relief():
    cfg = RewardConfig(
        w_action_magnitude=1.0,
        action_magnitude_free=1.0,
        action_magnitude_scale=1.0,
        action_magnitude_terrain_relief=0.75,
    )
    nominal = compute_reward_components(make_inp(
        1, action=torch.ones(1, 12)
    ), cfg)
    large_flat = compute_reward_components(make_inp(
        1, action=torch.full((1, 12), 3.0)
    ), cfg)
    large_after_contact = compute_reward_components(make_inp(
        1,
        action=torch.full((1, 12), 3.0),
        terrain_response=torch.ones(1),
    ), cfg)

    assert nominal["action_magnitude"].item() == 0.0
    assert large_flat["action_magnitude"].item() < 0.0
    assert large_after_contact["action_magnitude"].item() == pytest.approx(
        0.25 * large_flat["action_magnitude"].item()
    )


def test_terrain_type_specific_safety_is_diagnostic_only():
    comp = {
        "terrain_collapse": torch.tensor([-1.0, -2.0]),
        "total": torch.zeros(2),
    }
    grouped = group_reward_vector(comp)
    assert torch.count_nonzero(grouped) == 0


# ── A2 yaw 跟踪必须由真实 yaw 命令激活 ─────────────────────────────────
def test_yaw_tracking_requires_yaw_command():
    cfg = RewardConfig()
    # yaw 命令为零时 yaw_cmd_gate 关闭，即使 wz 匹配也不能拿 tracking_yaw。
    no_yaw_cmd = compute_reward_components(make_inp(), cfg)
    assert float(no_yaw_cmd["tracking_yaw"].mean()) == 0.0
    assert float(no_yaw_cmd["yaw_progress"].mean()) == 0.0
    yaw = make_inp(
        cmd=torch.tensor([[0.0, 0.0, 0.8]]).repeat(2, 1),
        base_ang_vel=torch.tensor([[0.0, 0.0, 0.8]]).repeat(2, 1),
    )
    assert float(compute_reward_components(yaw, cfg)["tracking_yaw"].mean()) > 0.0


def test_pure_yaw_does_not_collect_linear_tracking_reward():
    cfg = RewardConfig()
    yaw = make_inp(
        cmd=torch.tensor([[0.0, 0.0, 0.6]]).repeat(2, 1),
        base_ang_vel=torch.tensor([[0.0, 0.0, 0.6]]).repeat(2, 1),
    )
    comp = compute_reward_components(yaw, cfg)
    assert float(comp["tracking_lin"].mean()) == 0.0
    assert float(comp["tracking_lin_far"].mean()) == 0.0


def test_yaw_support_moment_rewards_command_sign_and_penalizes_opposite():
    cfg = RewardConfig(w_yaw_support_moment=0.35, w_yaw_wrong_moment=0.55)
    common = dict(
        cmd=torch.tensor([[0.0, 0.0, 0.6]]),
        base_ang_vel=torch.tensor([[0.0, 0.0, 0.3]]),
    )
    good = compute_reward_components(make_inp(n=1, yaw_support_moment_norm=torch.tensor([0.5]), **common), cfg)
    bad = compute_reward_components(make_inp(n=1, yaw_support_moment_norm=torch.tensor([-0.5]), **common), cfg)
    assert good["yaw_support_moment"].item() > 0.0
    assert good["yaw_wrong_moment"].item() == 0.0
    assert bad["yaw_support_moment"].item() == 0.0
    assert bad["yaw_wrong_moment"].item() < 0.0


def test_wrong_yaw_is_normalized_by_command_magnitude():
    cfg = RewardConfig(w_wrong_dir=1.2)
    small = compute_reward_components(make_inp(
        n=1,
        cmd=torch.tensor([[0.0, 0.0, 0.3]]),
        base_ang_vel=torch.tensor([[0.0, 0.0, -0.15]]),
    ), cfg)
    large = compute_reward_components(make_inp(
        n=1,
        cmd=torch.tensor([[0.0, 0.0, 0.6]]),
        base_ang_vel=torch.tensor([[0.0, 0.0, -0.3]]),
    ), cfg)
    assert torch.allclose(small["wrong_dir"], large["wrong_dir"])


def test_wrong_yaw_keeps_a_tail_gradient_for_severe_reversal():
    cfg = RewardConfig(w_wrong_dir=1.2)
    moderate = compute_reward_components(make_inp(
        n=1,
        cmd=torch.tensor([[0.0, 0.0, 0.6]]),
        base_ang_vel=torch.tensor([[0.0, 0.0, -0.6]]),
    ), cfg)
    severe = compute_reward_components(make_inp(
        n=1,
        cmd=torch.tensor([[0.0, 0.0, 0.6]]),
        base_ang_vel=torch.tensor([[0.0, 0.0, -1.2]]),
    ), cfg)
    assert severe["wrong_dir"].item() < moderate["wrong_dir"].item() < 0.0


def test_yaw_progress_rewards_true_same_direction_turn():
    cfg = RewardConfig(w_tracking_yaw=0.0, w_yaw_far=0.0, w_yaw_progress=2.0)
    good = compute_reward_components(make_inp(
        cmd=torch.tensor([[0.0, 0.0, 0.6]]).repeat(2, 1),
        base_ang_vel=torch.tensor([[0.0, 0.0, 0.6]]).repeat(2, 1),
    ), cfg)
    slow = compute_reward_components(make_inp(
        cmd=torch.tensor([[0.0, 0.0, 0.6]]).repeat(2, 1),
        base_ang_vel=torch.tensor([[0.0, 0.0, 0.15]]).repeat(2, 1),
    ), cfg)
    wrong = compute_reward_components(make_inp(
        cmd=torch.tensor([[0.0, 0.0, 0.6]]).repeat(2, 1),
        base_ang_vel=torch.tensor([[0.0, 0.0, -0.3]]).repeat(2, 1),
    ), cfg)

    assert good["yaw_progress"].mean() > slow["yaw_progress"].mean() > 0.0
    assert float(wrong["yaw_progress"].mean()) == 0.0


def test_yaw_progress_does_not_reward_overspeed_as_full_tracking():
    cfg = RewardConfig(
        w_tracking_yaw=0.0,
        w_yaw_far=0.0,
        w_yaw_progress=1.0,
        scale_yaw_far=0.60,
    )
    target = compute_reward_components(make_inp(
        n=1,
        cmd=torch.tensor([[0.0, 0.0, 0.6]]),
        base_ang_vel=torch.tensor([[0.0, 0.0, 0.6]]),
    ), cfg)
    overspeed = compute_reward_components(make_inp(
        n=1,
        cmd=torch.tensor([[0.0, 0.0, 0.6]]),
        base_ang_vel=torch.tensor([[0.0, 0.0, 1.2]]),
    ), cfg)
    assert target["yaw_progress"].item() > overspeed["yaw_progress"].item() > 0.0


def test_yaw_progress_is_in_turn_reward_group():
    comp = {"yaw_progress": torch.tensor([1.0, 2.0]), "total": torch.zeros(2)}
    grouped = group_reward_vector(comp)
    turn_col = REWARD_GROUP_NAMES.index("turn")
    assert torch.allclose(grouped[:, turn_col], comp["yaw_progress"])


def test_supported_and_terrain_progress_share_the_tracking_group():
    supported = torch.tensor([0.4, 0.8])
    terrain = torch.tensor([0.2, 0.6])
    comp = {
        "supported_progress": supported,
        "terrain_progress": terrain,
        "total": supported + terrain,
    }
    grouped = group_reward_vector(comp)
    track_col = REWARD_GROUP_NAMES.index("track")
    assert torch.allclose(grouped[:, track_col], supported + terrain)


def test_transition_motion_profile_is_in_stability_reward_group():
    profile = torch.tensor([-0.25, -1.0])
    comp = {"transition_motion_profile": profile, "total": profile.clone()}
    grouped = group_reward_vector(comp)
    stab_col = REWARD_GROUP_NAMES.index("stab")
    assert torch.allclose(grouped[:, stab_col], profile)
    assert torch.allclose(grouped.sum(dim=-1), comp["total"])


def test_yaw_tracking_keeps_base_drive_when_quality_is_poor():
    cfg = RewardConfig(w_tracking_yaw=1.0, yaw_tracking_base_fraction=0.60)
    inp = make_inp(
        cmd=torch.tensor([[0.0, 0.0, 0.6]]).repeat(2, 1),
        base_ang_vel=torch.tensor([[1.0, 1.0, 0.6]]).repeat(2, 1),
        foot_contact=torch.zeros(2, 4),
        duty_quality_window=torch.zeros(2),
        stance_slip_high_fraction=torch.ones(2),
    )
    comp = compute_reward_components(inp, cfg)
    assert torch.all(comp["validated_yaw_tracking_gate"] >= 0.60)
    assert torch.all(comp["tracking_yaw"] > 0.0)


def test_directional_support_balance_uses_command_axis():
    cfg = RewardConfig(w_directional_support_balance=1.0)
    # 前后 duty 不平衡、左右平均平衡：横移是主惩罚，直行仍保留较弱前后约束。
    duty = torch.tensor([[0.8, 0.8, 0.2, 0.2]])
    forward = compute_reward_components(make_inp(
        n=1,
        duty_by_leg_window=duty,
        duty_quality_window=torch.ones(1),
        diagonal_pair_window=torch.ones(1),
    ), cfg)
    lateral = compute_reward_components(make_inp(
        n=1,
        cmd=torch.tensor([[0.0, 0.3, 0.0]]),
        base_lin_vel=torch.tensor([[0.0, 0.3, 0.0]]),
        duty_by_leg_window=duty,
        duty_quality_window=torch.ones(1),
        diagonal_pair_window=torch.ones(1),
    ), cfg)
    assert forward["directional_support_balance"].item() < 0.0
    assert lateral["directional_support_balance"].item() < forward["directional_support_balance"].item()


def test_yaw_support_balance_detects_diagonal_duty_mismatch():
    cfg = RewardConfig(w_directional_support_balance=1.0)
    common = dict(
        n=1,
        cmd=torch.tensor([[0.0, 0.0, 0.8]]),
        base_lin_vel=torch.zeros((1, 3)),
        base_ang_vel=torch.tensor([[0.0, 0.0, 0.8]]),
        duty_quality_window=torch.ones(1),
        diagonal_pair_window=torch.ones(1),
    )
    balanced = compute_reward_components(make_inp(
        duty_by_leg_window=torch.full((1, 4), 0.5), **common
    ), cfg)
    diagonal_mismatch = compute_reward_components(make_inp(
        duty_by_leg_window=torch.tensor([[0.8, 0.2, 0.2, 0.8]]), **common
    ), cfg)
    assert diagonal_mismatch["directional_support_balance"].item() < balanced[
        "directional_support_balance"
    ].item()


def test_stand_contact_exposes_a_single_missing_foot():
    cfg = RewardConfig(w_stand_contact=1.0)
    common = dict(
        n=1,
        cmd=torch.zeros((1, 3)),
        base_lin_vel=torch.zeros((1, 3)),
        stand_gate=torch.ones(1),
        moving_gate=torch.zeros(1),
        desired_foot_contact=torch.ones((1, 4)),
    )
    all_feet = compute_reward_components(make_inp(
        foot_contact=torch.ones((1, 4)), **common
    ), cfg)
    missing_one = compute_reward_components(make_inp(
        foot_contact=torch.tensor([[1.0, 0.0, 1.0, 1.0]]), **common
    ), cfg)
    assert all_feet["stand_contact"].item() == 1.0
    assert missing_one["stand_contact"].item() < 0.25


def test_lateral_coordinated_progress_requires_motion_and_quality():
    cfg = RewardConfig(w_lateral_coordinated_progress=1.0, lateral_quality_floor=0.30)
    coordinated = compute_reward_components(make_inp(
        cmd=torch.tensor([[0.0, 0.3, 0.0]]),
        base_lin_vel=torch.tensor([[0.0, 0.3, 0.0]]),
        base_ang_vel=torch.zeros((1, 3)),
        duty_by_leg_window=torch.full((1, 4), 0.5),
        support_force_by_leg_window=torch.full((1, 4), 100.0),
        foot_relative_velocity_body=torch.tensor([[[0.0, 0.3, 0.0]] * 4]),
    ), cfg)
    uncoordinated = compute_reward_components(make_inp(
        cmd=torch.tensor([[0.0, 0.3, 0.0]]),
        base_lin_vel=torch.tensor([[0.0, 0.3, 0.0]]),
        base_ang_vel=torch.tensor([[0.8, 0.8, 0.8]]),
        duty_by_leg_window=torch.tensor([[0.8, 0.8, 0.2, 0.2]]),
        support_force_by_leg_window=torch.tensor([[180.0, 180.0, 20.0, 20.0]]),
        foot_relative_velocity_body=torch.tensor([[[0.0, 0.6, 0.0], [0.0, 0.6, 0.0], [0.0, 0.1, 0.0], [0.0, 0.1, 0.0]]]),
    ), cfg)
    forward = compute_reward_components(make_inp(), cfg)

    assert coordinated["lateral_coordinated_progress"].mean().item() > uncoordinated["lateral_coordinated_progress"].mean().item() > 0.0
    assert forward["lateral_coordinated_progress"].mean().item() == 0.0


def test_lateral_forward_cross_motion_reduces_coordinated_progress():
    cfg = RewardConfig(w_lateral_coordinated_progress=1.0, lateral_quality_floor=0.20)
    common = dict(
        n=1,
        cmd=torch.tensor([[0.0, 0.3, 0.0]]),
        base_ang_vel=torch.zeros((1, 3)),
        duty_by_leg_window=torch.full((1, 4), 0.5),
        support_force_by_leg_window=torch.full((1, 4), 100.0),
        foot_relative_velocity_body=torch.tensor([[[0.0, 0.3, 0.0]] * 4]),
    )
    clean = compute_reward_components(make_inp(
        base_lin_vel=torch.tensor([[0.0, 0.3, 0.0]]), **common
    ), cfg)
    crossing = compute_reward_components(make_inp(
        base_lin_vel=torch.tensor([[0.3, 0.3, 0.0]]), **common
    ), cfg)
    assert clean["lateral_coordinated_progress"].item() > crossing["lateral_coordinated_progress"].item()


def test_lateral_front_rear_foot_speed_mismatch_increases_directional_penalty():
    cfg = RewardConfig(w_directional_support_balance=1.0)
    common = dict(
        n=1,
        cmd=torch.tensor([[0.0, 0.3, 0.0]]),
        base_lin_vel=torch.tensor([[0.0, 0.3, 0.0]]),
        duty_by_leg_window=torch.full((1, 4), 0.5),
        support_force_by_leg_window=torch.full((1, 4), 100.0),
        duty_quality_window=torch.ones(1),
        diagonal_pair_window=torch.ones(1),
    )
    balanced_vel = torch.zeros(1, 4, 3)
    # 世界中近似静止的支撑足，相对机身速度应为 -v_base。
    balanced_vel[:, :, 1] = -0.3
    mismatched_vel = balanced_vel.clone()
    mismatched_vel[:, :2, 1] = -0.8
    mismatched_vel[:, 2:, 1] = -0.1
    balanced = compute_reward_components(make_inp(foot_relative_velocity_body=balanced_vel, **common), cfg)
    mismatched = compute_reward_components(make_inp(foot_relative_velocity_body=mismatched_vel, **common), cfg)
    assert mismatched["directional_support_balance"].item() < balanced["directional_support_balance"].item()


def test_lateral_diagonal_pair_velocity_penalizes_front_rear_mismatch():
    cfg = RewardConfig(
        w_lateral_pair_velocity=1.0,
        lateral_pair_velocity_free=0.12,
        lateral_pair_velocity_scale=0.60,
    )
    common = dict(
        n=1,
        cmd=torch.tensor([[0.0, 0.3, 0.0]]),
        base_lin_vel=torch.tensor([[0.0, 0.3, 0.0]]),
        foot_trajectory_scope=torch.ones(1),
    )
    coordinated_velocity = torch.zeros(1, 4, 3)
    coordinated_velocity[:, :, 1] = -0.3
    mismatched_velocity = coordinated_velocity.clone()
    mismatched_velocity[:, :2, 1] = -0.8
    mismatched_velocity[:, 2:, 1] = -0.1

    coordinated = compute_reward_components(make_inp(
        foot_relative_velocity_body=coordinated_velocity, **common
    ), cfg)
    mismatched = compute_reward_components(make_inp(
        foot_relative_velocity_body=mismatched_velocity, **common
    ), cfg)

    assert coordinated["lateral_pair_velocity"].item() == 0.0
    assert mismatched["lateral_pair_velocity"].item() < 0.0


def test_yaw_far_kernel_keeps_gradient_when_exact_kernel_is_saturated():
    cfg = RewardConfig(w_tracking_yaw=2.8, sigma_yaw=0.10, w_yaw_far=1.3, scale_yaw_far=0.60)
    slow = compute_reward_components(make_inp(
        n=1,
        cmd=torch.tensor([[0.0, 0.0, 0.6]]),
        base_ang_vel=torch.tensor([[0.0, 0.0, 0.2]]),
    ), cfg)
    better = compute_reward_components(make_inp(
        n=1,
        cmd=torch.tensor([[0.0, 0.0, 0.6]]),
        base_ang_vel=torch.tensor([[0.0, 0.0, 0.3]]),
    ), cfg)
    assert slow["tracking_yaw"].item() < 1e-5
    assert better["tracking_yaw_far"].item() > slow["tracking_yaw_far"].item() > 0.0


def test_cycle_wxy_energy_penalizes_zero_mean_oscillation():
    cfg = RewardConfig(w_cycle_wxy_bias=0.35, cycle_wxy_rms_free=0.18, cycle_wxy_rms_scale=0.45)
    quiet = compute_reward_components(make_inp(
        n=1,
        cycle_wxy_energy=torch.tensor([0.01]),
    ), cfg)
    oscillating = compute_reward_components(make_inp(
        n=1,
        # 代表窗口内 +0.5/-0.5 rad/s 往复，均值为零但均方不为零。
        cycle_wxy_energy=torch.tensor([0.25]),
    ), cfg)
    assert quiet["cycle_wxy_bias"].item() == 0.0
    assert oscillating["cycle_wxy_bias"].item() < 0.0


def test_heading_and_cycle_costs_keep_gradient_beyond_the_old_saturation_point():
    cfg = RewardConfig(
        w_heading_hold=1.0,
        heading_hold_scale=0.35,
        w_cycle_wxy_bias=1.0,
        cycle_wxy_rms_free=0.10,
        cycle_wxy_rms_scale=0.30,
    )
    moderate = compute_reward_components(make_inp(
        n=1,
        heading_error=torch.tensor([0.5]),
        cycle_wxy_energy=torch.tensor([0.25]),
    ), cfg)
    severe = compute_reward_components(make_inp(
        n=1,
        heading_error=torch.tensor([1.0]),
        cycle_wxy_energy=torch.tensor([1.0]),
    ), cfg)
    assert severe["heading_hold"].item() < moderate["heading_hold"].item() < 0.0
    assert severe["cycle_wxy_bias"].item() < moderate["cycle_wxy_bias"].item() < 0.0


def test_touchdown_horizontal_tail_distinguishes_severe_sweep():
    cfg = RewardConfig(
        w_touchdown_slip=0.0,
        w_touchdown_slip_late=0.0,
        w_touchdown_slip_tail=1.0,
        impact_speed_scale=0.35,
    )
    common = dict(
        n=1,
        touchdown_mask=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        touchdown_persistence_scale=1.0,
    )
    moderate = compute_reward_components(make_inp(
        touchdown_xy_speed=torch.tensor([[0.5, 0.0, 0.0, 0.0]]),
        **common,
    ), cfg)
    severe = compute_reward_components(make_inp(
        touchdown_xy_speed=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        **common,
    ), cfg)
    assert severe["touchdown_slip"].item() < moderate["touchdown_slip"].item() < 0.0


def test_cycle_wxy_penalty_is_relaxed_during_terrain_response():
    cfg = RewardConfig(w_cycle_wxy_bias=1.0, cycle_wxy_rms_free=0.10, cycle_wxy_rms_scale=0.45)
    common = dict(n=1, cycle_wxy_energy=torch.tensor([0.25]))
    flat = compute_reward_components(make_inp(terrain_response=torch.zeros(1), **common), cfg)
    terrain = compute_reward_components(make_inp(terrain_response=torch.ones(1), **common), cfg)
    assert torch.allclose(terrain["cycle_wxy_bias"], 0.35 * flat["cycle_wxy_bias"])


# ── B2 滑移惩罚要分级且有界，避免零梯度打滑区 ─────────────────────────
def test_slip_penalty_graded_and_bounded():
    cfg = RewardConfig()
    slow = compute_reward_components(make_inp(stance_slip_speed_window=torch.full((2,), 0.3)), cfg)
    fast = compute_reward_components(make_inp(stance_slip_speed_window=torch.full((2,), 0.8)), cfg)
    faster = compute_reward_components(make_inp(stance_slip_speed_window=torch.full((2,), 5.0)), cfg)
    # 滑移越大惩罚越强，但高速时会饱和。
    assert slow["stance_slip"].mean() > fast["stance_slip"].mean()   # 0.3 的惩罚弱于 0.8。
    assert float(faster["stance_slip"].mean()) >= -(cfg.w_stance_slip + cfg.w_stance_slip_late) - 1e-6


def test_clearance_terrain_aware_semantics():
    # 平地上高摆腿会被 over-clearance 惩罚；地形上目标高度随障碍升高。
    # 环境从高度扫描器提供 local_obstacle_h，这里只验证奖励的目标高度逻辑。
    cfg = RewardConfig()
    cfg.w_clearance_over = 1.0  # 固定权重，让符号判断明确。
    # 摆动脚是 FR、RL，给它们 0.20 m 的高抬脚。
    hi = torch.tensor([[0.0, 0.20, 0.20, 0.0]]).repeat(2, 1)
    flat = compute_reward_components(make_inp(local_obstacle_h=torch.zeros(2), foot_clearance=hi), cfg)
    assert flat["clearance_over"].mean() < -0.01, "平地过高抬脚必须被惩罚"
    # 地形：障碍 0.25 m，目标为 0.25+margin；0.28 m 抬脚仍在允许带内。
    terr = compute_reward_components(make_inp(
        local_obstacle_h=torch.full((2,), 0.25),
        terrain_response=torch.ones(2),
        foot_clearance=torch.tensor([[0.0, 0.28, 0.28, 0.0]]).repeat(2, 1)), cfg)
    assert float(terr["clearance_over"].mean()) == 0.0, "地形允许带内抬脚不应被惩罚"
    # 平地目标高度 0.08 m 也不应被惩罚。
    ok = compute_reward_components(make_inp(local_obstacle_h=torch.zeros(2)), cfg)
    assert float(ok["clearance_over"].mean()) == 0.0


def test_contact_quality_is_additive_and_does_not_gate_tracking():
    cfg = RewardConfig()
    cfg.validated_tracking_floor = 0.40
    cfg.healthy_progress_slip_target = 0.10
    cfg.healthy_progress_slip_width = 0.10

    good = compute_reward_components(make_inp(
        diagonal_pair_window=torch.ones(2),
        duty_quality_window=torch.ones(2),
        stance_slip_high_fraction=torch.zeros(2),
    ), cfg)
    bad = compute_reward_components(make_inp(
        diagonal_pair_window=torch.zeros(2),
        duty_quality_window=torch.zeros(2),
        stance_slip_high_fraction=torch.ones(2),
    ), cfg)

    assert good["validated_tracking_gate"].mean() > bad["validated_tracking_gate"].mean()
    assert torch.allclose(good["tracking_lin"], bad["tracking_lin"])
    assert bad["stance_slip"].mean() < good["stance_slip"].mean()


def test_pure_yaw_tracking_does_not_depend_on_linear_duty_or_diag():
    cfg = RewardConfig()
    cfg.validated_tracking_floor = 0.20
    yaw_inp = dict(
        cmd=torch.tensor([[0.0, 0.0, 0.7]]).repeat(2, 1),
        base_lin_vel=torch.zeros(2, 3),
        base_ang_vel=torch.tensor([[0.0, 0.0, 0.7]]).repeat(2, 1),
        foot_contact=torch.tensor([[1.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
    )
    clean = compute_reward_components(make_inp(
        **yaw_inp,
        diagonal_pair_window=torch.ones(2),
        duty_quality_window=torch.ones(2),
        stance_slip_high_fraction=torch.zeros(2),
    ), cfg)
    bad_linear_rhythm = compute_reward_components(make_inp(
        **yaw_inp,
        diagonal_pair_window=torch.zeros(2),
        duty_quality_window=torch.zeros(2),
        stance_slip_high_fraction=torch.zeros(2),
    ), cfg)

    assert torch.allclose(clean["tracking_yaw"], bad_linear_rhythm["tracking_yaw"])
    assert torch.allclose(clean["validated_yaw_tracking_gate"], bad_linear_rhythm["validated_yaw_tracking_gate"])
    assert float(bad_linear_rhythm["tracking_yaw"].mean()) > 0.0


def test_flat_motion_height_is_penalized_but_terrain_event_is_relaxed():
    cfg = RewardConfig(
        w_flat_move_height=1.0,
        flat_move_height_target=0.56,
        flat_move_height_band=0.04,
        flat_move_height_terrain_relief=0.80,
    )
    low_flat = compute_reward_components(make_inp(
        base_h_above_terrain=torch.full((2,), 0.50),
        local_obstacle_h=torch.zeros(2),
    ), cfg)
    low_terrain = compute_reward_components(make_inp(
        base_h_above_terrain=torch.full((2,), 0.50),
        local_obstacle_h=torch.full((2,), 0.12),
        terrain_response=torch.ones(2),
    ), cfg)
    assert low_flat["flat_move_height"].mean() < 0.0
    assert torch.allclose(low_terrain["flat_move_height"], 0.20 * low_flat["flat_move_height"])


def test_body_oscillation_is_additive_and_does_not_scale_tracking_drive():
    cfg = RewardConfig(
        tracking_posture_floor=0.45,
        tracking_motion_floor=0.45,
        tracking_wxy_target=0.20,
        tracking_wxy_width=0.30,
        tracking_vz_target=0.10,
        tracking_vz_width=0.20,
    )
    clean = compute_reward_components(make_inp(), cfg)
    shaking = compute_reward_components(make_inp(
        base_ang_vel=torch.tensor([[1.1, 0.9, 0.0]]).repeat(2, 1),
        base_lin_vel=torch.tensor([[0.5, 0.0, 0.45]]).repeat(2, 1),
    ), cfg)
    assert torch.all(shaking["tracking_lin"] > 0.0)
    assert torch.allclose(shaking["tracking_lin"], clean["tracking_lin"])
    assert clean["tracking_motion_gate"].mean() > shaking["tracking_motion_gate"].mean()
    assert shaking["base_wxy"].mean() < clean["base_wxy"].mean()
    assert shaking["base_vz"].mean() < clean["base_vz"].mean()
    assert shaking["total"].mean() < clean["total"].mean()


def test_hip_deviation_is_relaxed_but_not_removed_for_lateral_and_yaw():
    cfg = RewardConfig(w_hip_deviation=1.0, hip_deviation_scale=0.5,
                       hip_deviation_lat_scale=0.40, hip_deviation_yaw_scale=0.30)
    forward = compute_reward_components(make_inp(hip_deviation=torch.full((2,), 0.5)), cfg)
    lateral = compute_reward_components(make_inp(
        cmd=torch.tensor([[0.0, 0.4, 0.0]]).repeat(2, 1),
        base_lin_vel=torch.tensor([[0.0, 0.4, 0.0]]).repeat(2, 1),
        hip_deviation=torch.full((2,), 0.5),
    ), cfg)
    yaw = compute_reward_components(make_inp(
        cmd=torch.tensor([[0.0, 0.0, 0.6]]).repeat(2, 1),
        base_ang_vel=torch.tensor([[0.0, 0.0, 0.6]]).repeat(2, 1),
        hip_deviation=torch.full((2,), 0.5),
    ), cfg)
    assert forward["hip_deviation"].mean() < lateral["hip_deviation"].mean() < 0.0
    assert forward["hip_deviation"].mean() < yaw["hip_deviation"].mean() < 0.0


def test_pure_x_hip_constraint_uses_worst_leg_while_lateral_remains_relaxed():
    cfg = RewardConfig(w_hip_deviation=1.0, hip_deviation_scale=0.5,
                       hip_deviation_lat_scale=0.40, hip_deviation_yaw_scale=0.30)
    forward = compute_reward_components(make_inp(
        hip_deviation=torch.full((2,), 0.10),
        hip_deviation_max=torch.full((2,), 0.50),
    ), cfg)
    lateral = compute_reward_components(make_inp(
        cmd=torch.tensor([[0.0, 0.4, 0.0]]).repeat(2, 1),
        base_lin_vel=torch.tensor([[0.0, 0.4, 0.0]]).repeat(2, 1),
        hip_deviation=torch.full((2,), 0.10),
        hip_deviation_max=torch.full((2,), 0.50),
    ), cfg)

    assert forward["hip_deviation"].mean() < lateral["hip_deviation"].mean() < 0.0


def test_clearance_quality_does_not_gate_direction_task():
    cfg = RewardConfig(w_supported_progress=1.0, w_terrain_progress=0.0)
    low = compute_reward_components(make_inp(
        local_obstacle_h=torch.full((2,), 0.25),
        terrain_response=torch.ones(2),
        foot_clearance=torch.tensor([[0.0, 0.10, 0.10, 0.0]]).repeat(2, 1),
        diagonal_pair_window=torch.ones(2),
        duty_quality_window=torch.ones(2),
        stance_slip_high_fraction=torch.zeros(2),
    ), cfg)
    high = compute_reward_components(make_inp(
        local_obstacle_h=torch.full((2,), 0.25),
        terrain_response=torch.ones(2),
        foot_clearance=torch.tensor([[0.0, 0.30, 0.30, 0.0]]).repeat(2, 1),
        diagonal_pair_window=torch.ones(2),
        duty_quality_window=torch.ones(2),
        stance_slip_high_fraction=torch.zeros(2),
    ), cfg)

    assert high["validated_tracking_gate"].mean() > low["validated_tracking_gate"].mean()
    assert torch.allclose(high["supported_progress"], low["supported_progress"])
    assert high["clearance_under"].mean() > low["clearance_under"].mean()


def test_unstable_contact_keeps_terrain_drive_but_not_flat_progress_credit():
    cfg = RewardConfig(w_supported_progress=1.0, w_terrain_progress=1.0, w_base_wxy=1.0)
    common = dict(
        n=1,
        local_obstacle_h=torch.full((1,), 0.25),
        terrain_response=torch.ones(1),
        foot_clearance=torch.tensor([[0.0, 0.30, 0.30, 0.0]]),
        diagonal_pair_window=torch.ones(1),
        duty_quality_window=torch.ones(1),
        stance_slip_high_fraction=torch.zeros(1),
    )
    good = compute_reward_components(make_inp(**common), cfg)
    invalid = compute_reward_components(make_inp(
        base_ang_vel=torch.tensor([[3.0, 0.0, 0.0]]), **common
    ), cfg)
    assert good["supported_progress"].item() >= invalid["supported_progress"].item()
    assert invalid["terrain_progress"].item() > 0.0
    assert invalid["base_wxy"].item() < good["base_wxy"].item()


def test_flat_tracking_is_strict_and_terrain_tracking_keeps_a_nonzero_floor():
    cfg = RewardConfig(terrain_tracking_min_scale=0.45)
    flat = compute_reward_components(make_inp(local_obstacle_h=torch.zeros(2)), cfg)
    terrain = compute_reward_components(make_inp(
        local_obstacle_h=torch.full((2,), 0.30),
        terrain_response=torch.ones(2),
    ), cfg)

    assert torch.allclose(flat["terrain_tracking_scale"], torch.ones(2))
    assert torch.all(terrain["terrain_tracking_scale"] >= 0.45)
    assert terrain["tracking_lin"].mean() > 0.0
    assert terrain["tracking_lin"].mean() < flat["tracking_lin"].mean()
    assert torch.allclose(terrain["supported_progress"], flat["supported_progress"])


def test_height_scan_does_not_relax_rewards_before_contact():
    cfg = RewardConfig(
        w_flat_move_height=1.0,
        flat_move_height_target=0.56,
        flat_move_height_band=0.04,
        flat_move_height_terrain_relief=0.80,
    )
    scanned_only = compute_reward_components(make_inp(
        base_h_above_terrain=torch.full((2,), 0.50),
        local_obstacle_h=torch.full((2,), 0.20),
        terrain_response=torch.zeros(2),
    ), cfg)
    contacted = compute_reward_components(make_inp(
        base_h_above_terrain=torch.full((2,), 0.50),
        local_obstacle_h=torch.full((2,), 0.20),
        terrain_response=torch.ones(2),
    ), cfg)

    assert scanned_only["flat_move_height"].mean() < 0.0
    assert torch.allclose(contacted["flat_move_height"], 0.20 * scanned_only["flat_move_height"])
    assert torch.allclose(scanned_only["terrain_tracking_scale"], torch.ones(2))


def test_terrain_response_keeps_thirty_five_percent_of_linear_gait_structure():
    cfg = RewardConfig(w_gait_anchor=1.0, w_diagonal_contact=1.0, w_duty_balance=1.0)
    common = dict(
        diagonal_pair_window=torch.zeros(2),
        duty_quality_window=torch.zeros(2),
        desired_foot_contact=torch.tensor([[1.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
    )
    flat = compute_reward_components(make_inp(terrain_response=torch.zeros(2), **common), cfg)
    terrain = compute_reward_components(make_inp(terrain_response=torch.ones(2), **common), cfg)

    assert torch.allclose(terrain["diagonal_contact"], 0.35 * flat["diagonal_contact"])
    assert torch.allclose(terrain["duty_balance"], 0.35 * flat["duty_balance"])


def test_legacy_terrain_gate_cannot_block_contact_direction_credit():
    cfg = RewardConfig(w_supported_progress=1.0, w_terrain_progress=1.0, healthy_progress_floor=1.0)
    blocked = compute_reward_components(make_inp(
        1,
        terrain_response=torch.ones(1),
        terrain_progress_gate=torch.zeros(1),
    ), cfg)
    assert blocked["supported_progress"].item() > 0.0
    assert blocked["terrain_progress"].item() > 0.0


def test_two_foot_yaw_support_is_better_than_four_foot_dragging():
    cfg = RewardConfig(yaw_support_contact_target=2.25, yaw_support_contact_width=0.75)
    common = dict(
        n=1,
        cmd=torch.tensor([[0.0, 0.0, 0.6]]),
        base_ang_vel=torch.tensor([[0.0, 0.0, 0.6]]),
    )
    alternating = compute_reward_components(make_inp(
        foot_contact=torch.tensor([[1.0, 0.0, 0.0, 1.0]]),
        **common,
    ), cfg)
    dragging = compute_reward_components(make_inp(
        foot_contact=torch.ones(1, 4),
        **common,
    ), cfg)
    assert alternating["yaw_quality_gate"].item() > dragging["yaw_quality_gate"].item()


def test_yaw_progress_is_base_drive_and_support_quality_is_additive():
    cfg = RewardConfig(
        w_tracking_yaw=0.0,
        w_yaw_far=0.0,
        w_yaw_progress=1.0,
        yaw_progress_quality_floor=0.50,
    )
    common = dict(
        n=1,
        cmd=torch.tensor([[0.0, 0.0, 0.6]]),
        base_ang_vel=torch.tensor([[0.0, 0.0, 0.6]]),
    )
    clean = compute_reward_components(make_inp(
        foot_contact=torch.tensor([[1.0, 0.0, 0.0, 1.0]]), **common
    ), cfg)
    dragging = compute_reward_components(make_inp(
        foot_contact=torch.ones(1, 4), **common
    ), cfg)
    assert clean["yaw_progress"].item() == pytest.approx(dragging["yaw_progress"].item())
    assert clean["yaw_quality_gate"].item() > dragging["yaw_quality_gate"].item()
    assert clean["total"].item() > dragging["total"].item()


def test_extra_third_contact_cannot_collect_gait_anchor_reward():
    cfg = RewardConfig(w_gait_anchor=1.0, w_excess_support=1.0)
    exact = compute_reward_components(make_inp(
        1,
        foot_contact=torch.tensor([[1.0, 0.0, 0.0, 1.0]]),
        desired_foot_contact=torch.tensor([[1.0, 0.0, 0.0, 1.0]]),
        foot_trajectory_scope=torch.ones(1),
    ), cfg)
    extra = compute_reward_components(make_inp(
        1,
        foot_contact=torch.tensor([[1.0, 1.0, 0.0, 1.0]]),
        desired_foot_contact=torch.tensor([[1.0, 0.0, 0.0, 1.0]]),
        foot_trajectory_scope=torch.ones(1),
    ), cfg)
    assert exact["gait_anchor"].item() > 0.0
    assert extra["gait_anchor"].item() == 0.0
    assert extra["excess_support"].item() < exact["excess_support"].item()


def test_terrain_freefall_has_no_tracking_or_progress_reward():
    cfg = RewardConfig(w_supported_progress=1.0, w_terrain_progress=1.0)
    freefall = compute_reward_components(make_inp(
        1,
        terrain_response=torch.ones(1),
        foot_contact=torch.zeros(1, 4),
        base_h_above_terrain=torch.full((1,), 0.20),
        stable_motion_gate=torch.zeros(1),
        hard_survival_gate=torch.zeros(1),
    ), cfg)
    assert freefall["tracking_lin"].item() == 0.0
    assert freefall["tracking_lin_far"].item() == 0.0
    assert freefall["supported_progress"].item() == 0.0
    assert freefall["terrain_progress"].item() == 0.0


def test_flat_height_penalizes_both_crouching_and_excess_height():
    cfg = RewardConfig(
        w_flat_move_height=1.0,
        flat_move_height_target=0.52,
        flat_move_height_band=0.02,
    )
    centered = compute_reward_components(make_inp(1, base_h_above_terrain=torch.tensor([0.52])), cfg)
    low = compute_reward_components(make_inp(1, base_h_above_terrain=torch.tensor([0.47])), cfg)
    high = compute_reward_components(make_inp(1, base_h_above_terrain=torch.tensor([0.57])), cfg)
    assert centered["flat_move_height"].item() == 0.0
    assert low["flat_move_height"].item() < 0.0
    assert high["flat_move_height"].item() < 0.0


def test_clearance_worst_leg_is_not_hidden_by_other_swing_leg():
    cfg = RewardConfig(
        flat_clearance_target=0.06,
        clearance_band=0.01,
        w_clearance_under=1.0,
        clearance_worst_leg_mix=1.0,
    )
    out = compute_reward_components(make_inp(
        1,
        foot_contact=torch.tensor([[1.0, 0.0, 0.0, 1.0]]),
        foot_clearance=torch.tensor([[0.0, 0.06, 0.01, 0.0]]),
        swing_apex_weight=torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
    ), cfg)
    assert out["clearance_under"].item() < -0.5


def test_uphill_collision_clearance_is_not_suppressed_outside_gait_apex():
    cfg = RewardConfig(
        flat_clearance_target=0.06,
        clearance_band=0.02,
        w_clearance_under=1.0,
        clearance_worst_leg_mix=1.0,
    )
    common = dict(
        foot_contact=torch.tensor([[1.0, 0.0, 1.0, 1.0]]),
        foot_clearance=torch.tensor([[0.0, 0.04, 0.0, 0.0]]),
        swing_apex_weight=torch.zeros(1, 4),
        terrain_response=torch.tensor([0.4]),
        terrain_clearance_foot_mask=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        terrain_clearance_ramp=torch.tensor([0.4]),
        terrain_probe_target=torch.tensor([0.32]),
    )
    uphill = compute_reward_components(make_inp(
        1,
        **common,
        terrain_event_direction=torch.tensor([1.0]),
    ), cfg)
    downhill = compute_reward_components(make_inp(
        1,
        **common,
        terrain_event_direction=torch.tensor([-1.0]),
    ), cfg)

    assert uphill["clearance_under"].item() < -0.1
    assert downhill["clearance_under"].item() == 0.0


def test_uphill_collision_can_initiate_liftoff_from_a_supporting_leg():
    cfg = RewardConfig(
        flat_clearance_target=0.06,
        clearance_band=0.02,
        w_clearance_under=1.0,
        clearance_worst_leg_mix=1.0,
    )
    out = compute_reward_components(make_inp(
        1,
        foot_contact=torch.ones((1, 4)),
        foot_clearance=torch.zeros((1, 4)),
        swing_apex_weight=torch.zeros((1, 4)),
        terrain_response=torch.tensor([1.0]),
        terrain_clearance_foot_mask=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        terrain_collision_foot_mask=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        terrain_clearance_ramp=torch.tensor([1.0]),
        terrain_event_direction=torch.tensor([1.0]),
        terrain_probe_target=torch.tensor([0.12]),
    ), cfg)
    assert out["clearance_under"].item() < -0.1


def test_supporting_collision_leg_clearance_is_independent_from_direction_drive():
    cfg = RewardConfig(
        w_terrain_progress=0.0,
        w_supported_progress=1.0,
        flat_clearance_target=0.06,
        clearance_band=0.02,
    )
    common = dict(
        cmd=torch.tensor([[0.35, 0.0, 0.0]]),
        base_lin_vel=torch.tensor([[0.25, 0.0, 0.0]]),
        foot_contact=torch.ones((1, 4)),
        clearance_support_contact=torch.ones((1, 4)),
        terrain_response=torch.ones(1),
        terrain_clearance_foot_mask=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        terrain_collision_foot_mask=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        terrain_clearance_ramp=torch.ones(1),
        terrain_event_direction=torch.ones(1),
        terrain_probe_target=torch.tensor([0.10]),
        diagonal_pair_window=torch.ones(1),
        duty_quality_window=torch.ones(1),
        stance_slip_high_fraction=torch.zeros(1),
    )
    blocked = compute_reward_components(make_inp(
        1,
        foot_clearance=torch.zeros((1, 4)),
        **common,
    ), cfg)
    cleared = compute_reward_components(make_inp(
        1,
        foot_clearance=torch.tensor([[0.10, 0.0, 0.0, 0.0]]),
        **common,
    ), cfg)

    assert blocked["terrain_clearance_success"].item() == 0.0
    assert cleared["terrain_clearance_success"].item() == 1.0
    assert blocked["terrain_progress"].item() == 0.0
    assert cleared["terrain_progress"].item() == 0.0
    assert torch.allclose(blocked["supported_progress"], cleared["supported_progress"])
    assert torch.allclose(blocked["tracking_lin"], cleared["tracking_lin"])
    assert blocked["clearance_under"].item() < cleared["clearance_under"].item()


def test_recent_collision_trace_shapes_clearance_without_gating_direction():
    cfg = RewardConfig(
        w_terrain_progress=0.0,
        w_supported_progress=1.0,
        flat_clearance_target=0.06,
        clearance_band=0.02,
    )
    common = dict(
        cmd=torch.tensor([[0.35, 0.0, 0.0]]),
        base_lin_vel=torch.tensor([[0.20, 0.0, 0.0]]),
        foot_contact=torch.ones((1, 4)),
        clearance_support_contact=torch.ones((1, 4)),
        terrain_response=torch.tensor([0.5]),
        terrain_clearance_response=torch.ones(1),
        terrain_clearance_foot_mask=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        terrain_clearance_ramp=torch.ones(1),
        terrain_event_direction=torch.ones(1),
        terrain_probe_target=torch.tensor([0.10]),
        diagonal_pair_window=torch.ones(1),
        duty_quality_window=torch.ones(1),
        stance_slip_high_fraction=torch.zeros(1),
    )
    blocked = compute_reward_components(make_inp(
        1,
        foot_clearance=torch.zeros((1, 4)),
        **common,
    ), cfg)
    cleared = compute_reward_components(make_inp(
        1,
        foot_clearance=torch.tensor([[0.10, 0.0, 0.0, 0.0]]),
        **common,
    ), cfg)

    assert blocked["terrain_progress"].item() == 0.0
    assert cleared["terrain_progress"].item() == 0.0
    assert torch.allclose(blocked["supported_progress"], cleared["supported_progress"])
    assert blocked["terrain_clearance_success"].item() == 0.0
    assert cleared["terrain_clearance_success"].item() == 1.0


def test_reactive_clearance_does_not_inherit_the_flat_tolerance_band():
    cfg = RewardConfig(
        flat_clearance_target=0.06,
        clearance_band=0.02,
    )
    out = compute_reward_components(make_inp(
        1,
        foot_contact=torch.ones((1, 4)),
        clearance_support_contact=torch.ones((1, 4)),
        foot_clearance=torch.tensor([[0.08, 0.0, 0.0, 0.0]]),
        terrain_response=torch.ones(1),
        terrain_clearance_response=torch.ones(1),
        terrain_clearance_foot_mask=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        terrain_clearance_ramp=torch.ones(1),
        terrain_event_direction=torch.ones(1),
        terrain_probe_target=torch.tensor([0.10]),
    ), cfg)

    assert out["terrain_clearance_success"].item() < 1.0


def test_upper_layer_placement_completes_clearance_without_requiring_liftoff():
    cfg = RewardConfig(
        w_terrain_progress=1.0,
        flat_clearance_target=0.06,
        clearance_band=0.02,
    )
    out = compute_reward_components(make_inp(
        1,
        cmd=torch.tensor([[0.35, 0.0, 0.0]]),
        base_lin_vel=torch.tensor([[0.20, 0.0, 0.0]]),
        foot_contact=torch.ones((1, 4)),
        clearance_support_contact=torch.ones((1, 4)),
        foot_clearance=torch.zeros((1, 4)),
        terrain_response=torch.ones(1),
        terrain_clearance_foot_mask=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        terrain_collision_foot_mask=torch.zeros((1, 4)),
        terrain_placement_success_by_foot=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        terrain_clearance_ramp=torch.ones(1),
        terrain_event_direction=torch.ones(1),
        terrain_probe_target=torch.tensor([0.10]),
        diagonal_pair_window=torch.ones(1),
        duty_quality_window=torch.ones(1),
        stance_slip_high_fraction=torch.zeros(1),
    ), cfg)

    assert out["terrain_clearance_success"].item() == 1.0
    assert out["terrain_progress"].item() > 0.0


def test_reactive_clearance_uses_linear_tail_for_a_far_probe_target():
    cfg = RewardConfig(
        flat_clearance_target=0.06,
        clearance_band=0.02,
        w_clearance_under=1.0,
        clearance_worst_leg_mix=1.0,
    )
    common = dict(
        foot_contact=torch.tensor([[1.0, 0.0, 1.0, 1.0]]),
        foot_clearance=torch.tensor([[0.0, 0.04, 0.0, 0.0]]),
        swing_apex_weight=torch.zeros(1, 4),
        terrain_response=torch.tensor([1.0]),
        terrain_clearance_foot_mask=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        terrain_clearance_ramp=torch.tensor([1.0]),
        terrain_event_direction=torch.tensor([1.0]),
    )
    near = compute_reward_components(make_inp(
        1,
        **common,
        terrain_probe_target=torch.tensor([0.12]),
    ), cfg)["clearance_under"]
    far = compute_reward_components(make_inp(
        1,
        **common,
        terrain_probe_target=torch.tensor([0.32]),
    ), cfg)["clearance_under"]

    assert far.item() < near.item() < 0.0
    assert far.item() > -10.0


def test_touchdown_slip_tail_penalizes_single_bad_foot():
    cfg = RewardConfig(
        w_touchdown_slip=0.0,
        w_touchdown_slip_late=0.1,
        w_touchdown_slip_tail=1.0,
    )
    out = compute_reward_components(make_inp(
        1,
        touchdown_xy_speed=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        touchdown_mask=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
    ), cfg)
    assert out["touchdown_slip"].item() < -0.45


def test_heading_hold_owns_angle_while_off_axis_owns_yaw_rate():
    cfg = RewardConfig(w_heading_hold=1.0, w_off_axis=1.0)
    clean = compute_reward_components(make_inp(heading_error=torch.zeros(2)), cfg)
    rate_only = compute_reward_components(make_inp(
        heading_error=torch.zeros(2),
        base_ang_vel=torch.tensor([[0.0, 0.0, 0.2]]).repeat(2, 1),
    ), cfg)
    heading_drift = compute_reward_components(make_inp(
        heading_error=torch.full((2,), 0.2),
    ), cfg)

    assert torch.allclose(clean["heading_hold"], torch.zeros(2))
    assert torch.allclose(rate_only["heading_hold"], torch.zeros(2))
    assert rate_only["off_axis"].mean() < 0.0
    assert heading_drift["heading_hold"].mean() < 0.0


def test_stand_posture_keeps_gradient_when_multiplicative_stand_kernel_is_small():
    cfg = RewardConfig(w_stand=0.0, w_stand_posture=1.0)
    stand_common = dict(
        cmd=torch.zeros(1, 3),
        base_lin_vel=torch.zeros(1, 3),
        stand_gate=torch.ones(1),
        moving_gate=torch.zeros(1),
    )
    clean = compute_reward_components(make_inp(
        1,
        **stand_common,
        base_h_above_terrain=torch.tensor([cfg.nominal_base_h]),
        default_pose_error=torch.zeros(1),
        default_pose_error_tail=torch.zeros(1),
    ), cfg)
    crouched = compute_reward_components(make_inp(
        1,
        **stand_common,
        base_h_above_terrain=torch.tensor([cfg.nominal_base_h - 0.04]),
        default_pose_error=torch.tensor([0.15]),
        default_pose_error_tail=torch.tensor([0.30]),
    ), cfg)
    moving = compute_reward_components(make_inp(
        1,
        default_pose_error=torch.tensor([0.15]),
        default_pose_error_tail=torch.tensor([0.30]),
    ), cfg)

    assert clean["stand_posture"].item() == 0.0
    assert crouched["stand_posture"].item() < 0.0
    assert moving["stand_posture"].item() == 0.0


def test_cycle_vz_energy_penalizes_flat_bounce_and_relaxes_after_terrain_contact():
    cfg = RewardConfig(
        w_cycle_vz_bias=1.0,
        cycle_vz_rms_free=0.08,
        cycle_vz_rms_scale=0.22,
        terrain_style_max_relief=0.65,
    )
    clean = compute_reward_components(make_inp(
        1, cycle_vz_energy=torch.tensor([0.08 ** 2])
    ), cfg)
    bouncing = compute_reward_components(make_inp(
        1, cycle_vz_energy=torch.tensor([0.25 ** 2])
    ), cfg)
    terrain = compute_reward_components(make_inp(
        1,
        cycle_vz_energy=torch.tensor([0.25 ** 2]),
        terrain_response=torch.ones(1),
    ), cfg)

    assert clean["cycle_vz_bias"].item() == pytest.approx(0.0)
    assert bouncing["cycle_vz_bias"].item() < 0.0
    assert bouncing["cycle_vz_bias"].item() < terrain["cycle_vz_bias"].item() < 0.0


def test_pure_yaw_translation_is_a_direct_additive_cost():
    cfg = RewardConfig(w_yaw_translation=1.0)
    yaw_common = dict(
        cmd=torch.tensor([[0.0, 0.0, 0.6]]),
        base_ang_vel=torch.tensor([[0.0, 0.0, 0.6]]),
    )
    clean = compute_reward_components(make_inp(
        1, **yaw_common, base_lin_vel=torch.zeros(1, 3)
    ), cfg)
    drifting = compute_reward_components(make_inp(
        1, **yaw_common, base_lin_vel=torch.tensor([[0.20, 0.0, 0.0]])
    ), cfg)
    linear = compute_reward_components(make_inp(
        1, base_lin_vel=torch.tensor([[0.20, 0.0, 0.0]])
    ), cfg)

    assert clean["yaw_translation"].item() == 0.0
    assert drifting["yaw_translation"].item() < 0.0
    assert linear["yaw_translation"].item() == 0.0


def test_front_rear_extension_uses_worst_flat_geometry_without_locking_terrain():
    cfg = RewardConfig(w_front_rear_extension=1.0, terrain_style_max_relief=0.65)
    flat = compute_reward_components(make_inp(
        1, front_rear_extension_error=torch.tensor([0.08])
    ), cfg)
    terrain = compute_reward_components(make_inp(
        1,
        front_rear_extension_error=torch.tensor([0.08]),
        terrain_response=torch.ones(1),
    ), cfg)

    assert flat["front_rear_extension"].item() < 0.0
    assert flat["front_rear_extension"].item() < terrain["front_rear_extension"].item() < 0.0


def test_contact_chatter_penalizes_short_same_foot_cycles():
    cfg = RewardConfig(w_contact_chatter=1.0)
    clean = compute_reward_components(make_inp(
        1, contact_chatter_window=torch.zeros(1)
    ), cfg)
    chatter = compute_reward_components(make_inp(
        1, contact_chatter_window=torch.tensor([0.75])
    ), cfg)

    assert clean["contact_chatter"].item() == 0.0
    assert chatter["contact_chatter"].item() < 0.0


def test_raw_angular_acceleration_tail_matches_the_unfiltered_gate_quantity():
    cfg = RewardConfig(
        w_base_ang_accel=0.0,
        w_base_ang_accel_raw_tail=1.0,
        base_ang_accel_raw_free=8.0,
        base_ang_accel_raw_scale=20.0,
    )
    clean = compute_reward_components(make_inp(
        1,
        base_ang_accel=torch.zeros(1, 3),
        base_ang_accel_raw=torch.zeros(1, 3),
    ), cfg)
    shaking = compute_reward_components(make_inp(
        1,
        base_ang_accel=torch.zeros(1, 3),
        base_ang_accel_raw=torch.tensor([[30.0, 0.0, 0.0]]),
    ), cfg)

    assert clean["base_ang_accel"].item() == 0.0
    assert shaking["base_ang_accel"].item() < 0.0


def test_scalar_and_multi_critic_totals_exclude_all_diagnostic_gates():
    cfg = RewardConfig(
        w_supported_progress=1.0,
        w_terrain_progress=1.0,
        w_yaw_progress=1.0,
    )
    comp = compute_reward_components(make_inp(
        1,
        terrain_response=torch.tensor([0.5]),
        cycle_vz_energy=torch.tensor([0.04]),
        contact_chatter_window=torch.tensor([0.25]),
    ), cfg)
    grouped = group_reward_vector(comp)

    assert torch.allclose(grouped.sum(dim=-1), comp["total"])


def test_total_is_finite_everywhere():
    cfg = RewardConfig()
    for over in [{}, {"terminal_reason": "x"}, {"stable_motion_gate": torch.zeros(2)},
                 {"base_lin_vel": torch.full((2, 3), 5.0)}]:
        t = _total(make_inp(**over), cfg)
        assert torch.isfinite(t).all()
