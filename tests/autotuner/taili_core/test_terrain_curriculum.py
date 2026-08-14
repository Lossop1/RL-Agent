import torch

from autotuner.taili_core.terrain_curriculum import (
    classify_support_contact,
    command_leading_leg_weights,
    compute_terrain_curriculum_moves,
    continuous_terrain_contact_response,
    filter_loaded_support_height,
    loaded_support_height,
    map_body_collision_to_legs,
    scope_terrain_response,
    terrain_motion_credit,
    terrain_parameter_upper_bound,
    terrain_recovery_termination,
    unexpected_body_contact_score,
    update_contact_support_reference,
    update_terrain_collision_trace,
)


def test_support_contact_rejects_horizontal_wall_collision():
    force = torch.tensor([[[100.0, 0.0, 2.0], [0.0, 0.0, 120.0]]])
    normal = torch.tensor([[0.0, 0.0, 1.0]])
    out = classify_support_contact(force, normal, normal_force_min=10.0)
    assert out["contact"].tolist() == [[False, True]]


def test_loaded_support_height_uses_only_normal_load_bearing_feet():
    out = loaded_support_height(
        ground_z=torch.tensor([[0.20, 0.20, 0.00, 0.00]]),
        support_contact=torch.tensor([[True, True, True, True]]),
        support_force=torch.tensor([[150.0, 50.0, 100.0, 100.0]]),
        fallback_height=torch.tensor([9.0]),
    )
    assert torch.allclose(out["height"], torch.tensor([0.10]))
    assert out["valid"].tolist() == [True]
    assert out["loaded_count"].tolist() == [4]


def test_unloaded_high_foot_cannot_change_loaded_support_height():
    out = loaded_support_height(
        ground_z=torch.tensor([[0.30, 0.00, 0.00, 0.00]]),
        support_contact=torch.tensor([[False, True, True, True]]),
        support_force=torch.tensor([[0.0, 100.0, 100.0, 100.0]]),
        fallback_height=torch.tensor([0.0]),
    )
    assert torch.allclose(out["height"], torch.tensor([0.0]))


def test_no_support_uses_fallback_but_is_invalid():
    out = loaded_support_height(
        ground_z=torch.tensor([[0.30, 0.20, 0.10, 0.00]]),
        support_contact=torch.zeros((1, 4), dtype=torch.bool),
        support_force=torch.zeros((1, 4)),
        fallback_height=torch.tensor([0.07]),
    )
    assert torch.allclose(out["height"], torch.tensor([0.07]))
    assert out["valid"].tolist() == [False]


def test_support_height_filter_smooths_valid_load_and_holds_when_unloaded():
    filtered = filter_loaded_support_height(
        observed_height=torch.tensor([0.20, 0.20]),
        previous_height=torch.tensor([0.00, 0.05]),
        valid=torch.tensor([True, False]),
        alpha=0.20,
    )
    assert torch.allclose(filtered, torch.tensor([0.04, 0.05]))


def test_curriculum_parameter_target_grows_by_row_without_jumping_to_maximum():
    target = terrain_parameter_upper_bound(
        torch.tensor([0, 4, 9]),
        num_rows=10,
        value_min=0.04,
        value_max=0.30,
    )
    assert torch.allclose(target, torch.tensor([0.066, 0.17, 0.30]), atol=1e-6)
    fixed = terrain_parameter_upper_bound(
        torch.tensor([0]),
        num_rows=10,
        value_min=0.04,
        value_max=0.04,
    )
    assert torch.allclose(fixed, torch.tensor([0.04]))


def test_terrain_motion_credit_keeps_slow_progress_and_exposes_overspeed_gradient():
    out = terrain_motion_credit(
        v_along=torch.tensor([0.0, 0.14, 0.35, 0.525, -0.10]),
        command_speed=torch.full((5,), 0.35),
        progress_full_ratio=0.35,
        overspeed_start_ratio=1.15,
        overspeed_span_ratio=0.35,
    )
    assert out["progress"].tolist() == [0.0, 1.0, 1.0, 1.0, 0.0]
    assert out["overspeed"][:3].tolist() == [0.0, 0.0, 0.0]
    assert out["overspeed"][3] > 0.0
    assert out["overspeed"][4] == 0.0


def test_flat_loaded_support_has_zero_response_and_progress():
    ground = torch.zeros((1, 4))
    contact = torch.ones((1, 4), dtype=torch.bool)
    force = torch.full((1, 4), 100.0)
    state = loaded_support_height(
        ground_z=ground,
        support_contact=contact,
        support_force=force,
    )
    response = continuous_terrain_contact_response(
        ground_z=ground,
        support_contact=contact,
        support_force=force,
        current_height=state["height"],
        previous_height=torch.tensor([0.0]),
    )
    assert torch.allclose(response["response"], torch.tensor([0.0]))
    assert torch.allclose(response["support_delta"], torch.tensor([0.0]))


def test_contact_response_filters_flat_noise_but_keeps_four_centimeter_step():
    contact = torch.ones((1, 4), dtype=torch.bool)
    force = torch.full((1, 4), 100.0)
    flat_noise = continuous_terrain_contact_response(
        ground_z=torch.tensor([[0.006, 0.0, 0.0, 0.0]]),
        support_contact=contact,
        support_force=force,
        current_height=torch.tensor([0.0]),
        previous_height=torch.tensor([0.006]),
    )
    assert torch.allclose(flat_noise["response"], torch.tensor([0.0]))

    step = continuous_terrain_contact_response(
        ground_z=torch.tensor([[0.04, 0.0, 0.0, 0.0]]),
        support_contact=contact,
        support_force=force,
        current_height=torch.tensor([0.0]),
        previous_height=torch.tensor([0.0]),
    )
    assert torch.allclose(step["response"], torch.tensor([0.8]), atol=1e-6)


def test_horizontal_collision_creates_response_not_support_height():
    ground = torch.zeros((1, 4))
    response = continuous_terrain_contact_response(
        ground_z=ground,
        support_contact=torch.tensor([[False, True, True, True]]),
        support_force=torch.tensor([[0.0, 100.0, 100.0, 100.0]]),
        current_height=torch.tensor([0.0]),
        previous_height=torch.tensor([0.0]),
        collision_score_by_foot=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
    assert torch.allclose(response["support_delta"], torch.tensor([0.0]))
    assert response["response"].tolist() == [1.0]
    assert response["leg_response"].tolist() == [[1.0, 0.0, 0.0, 0.0]]


def test_collision_trace_is_reactive_smooth_and_scope_bounded():
    current = torch.tensor([[0.6, 0.0, 0.0, 0.0]])
    trace = update_terrain_collision_trace(
        current_response=current,
        previous_trace=torch.zeros_like(current),
        dt=0.02,
        decay_time=0.12,
        active=torch.tensor([True]),
    )
    assert torch.allclose(trace, current)

    decayed = update_terrain_collision_trace(
        current_response=torch.zeros_like(current),
        previous_trace=trace,
        dt=0.02,
        decay_time=0.12,
        active=torch.tensor([True]),
    )
    assert 0.0 < decayed[0, 0] < trace[0, 0]
    assert torch.count_nonzero(decayed[0, 1:]) == 0

    cleared = update_terrain_collision_trace(
        current_response=torch.ones_like(current),
        previous_trace=decayed,
        dt=0.02,
        decay_time=0.12,
        active=torch.tensor([False]),
    )
    assert torch.count_nonzero(cleared) == 0


def test_collision_trace_remains_actionable_for_half_a_gait_cycle():
    trace = torch.ones((1, 4))
    for _ in range(20):
        trace = update_terrain_collision_trace(
            current_response=torch.zeros_like(trace),
            previous_trace=trace,
            dt=0.02,
            decay_time=0.40,
            active=torch.tensor([True]),
        )

    assert 0.35 < trace[0, 0] < 0.40


def test_unexpected_body_contact_uses_continuous_force_score():
    score = unexpected_body_contact_score(
        torch.tensor([[[0.0, 0.0, 4.0], [27.5, 0.0, 0.0], [0.0, 50.0, 0.0]]]),
        force_start=5.0,
        force_span=45.0,
    )
    assert torch.allclose(score, torch.tensor([[0.0, 0.5, 1.0]]))


def test_base_collision_maps_to_command_leading_legs():
    command = torch.tensor([
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [0.0, 0.0],
    ])
    expected = torch.tensor([
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    assert torch.allclose(command_leading_leg_weights(command), expected)
    mapped = map_body_collision_to_legs(
        limb_collision_score=torch.zeros((5, 4)),
        base_collision_score=torch.ones(5),
        command_xy=command,
    )
    assert torch.allclose(mapped, expected)


def test_limb_collision_is_retained_without_a_command():
    mapped = map_body_collision_to_legs(
        limb_collision_score=torch.tensor([[0.0, 0.7, 0.0, 0.0]]),
        base_collision_score=torch.tensor([1.0]),
        command_xy=torch.zeros((1, 2)),
    )
    assert torch.allclose(mapped, torch.tensor([[0.0, 0.7, 0.0, 0.0]]))


def test_discrete_recovery_pose_survives_but_true_collapse_terminates():
    died = terrain_recovery_termination(
        base_height_local=torch.tensor([0.22, 0.17, 0.30, 0.30]),
        projected_gravity_z=torch.tensor([-0.87, -0.87, -0.40, -0.87]),
        finite=torch.tensor([True, True, True, False]),
        min_height=0.18,
        max_tilt_deg=60.0,
    )
    assert died.tolist() == [False, True, True, True]


def test_flat_scope_never_relaxes_style():
    response = scope_terrain_response(
        torch.tensor([1.0, 0.8]),
        torch.tensor([True, False]),
    )
    assert torch.allclose(response, torch.tensor([0.0, 0.8]))


def _curriculum_inputs():
    return dict(
        cmd_mag=torch.tensor([0.5]),
        forward_dist=torch.tensor([1.0]),
        support_height_delta=torch.tensor([0.10]),
        expected_height_direction=torch.tensor([1.0]),
        support_height_valid=torch.tensor([True]),
        valid_episode=torch.tensor([True]),
        terrain_curriculum_active=True,
        terrain_unlocked=True,
        terminal_now=torch.tensor([False]),
        base_h_local=torch.tensor([0.50]),
        upright_score=torch.tensor([0.98]),
        contact_count=torch.tensor([3.0]),
        body_wxy=torch.tensor([0.2]),
        v_along=torch.tensor([0.45]),
        max_episode_length_s=10.0,
        terrain_move_up_dist=2.0,
        stair_height_min=0.08,
        stair_forward_min=0.75,
    )


def test_curriculum_accepts_stable_loaded_height_traversal():
    out = compute_terrain_curriculum_moves(**_curriculum_inputs())
    assert out["height_ok"].tolist() == [True]
    assert out["controlled_up"].tolist() == [True]
    assert out["move_up"].tolist() == [True]
    assert out["move_down"].tolist() == [False]


def test_curriculum_rejects_root_jump_without_loaded_height_gain():
    args = _curriculum_inputs()
    args["support_height_delta"] = torch.tensor([0.0])
    out = compute_terrain_curriculum_moves(**args)
    assert out["height_ok"].tolist() == [False]
    assert out["move_up"].tolist() == [False]
    assert out["low_progress_down"].tolist() == [True]


def test_curriculum_rejects_collapse_even_after_height_gain():
    args = _curriculum_inputs()
    args["terminal_now"] = torch.tensor([True])
    out = compute_terrain_curriculum_moves(**args)
    assert out["move_up"].tolist() == [False]
    assert out["failure_down"].tolist() == [True]
    assert out["move_down"].tolist() == [True]


def test_curriculum_downstairs_uses_negative_height_delta():
    args = _curriculum_inputs()
    args["support_height_delta"] = torch.tensor([-0.10])
    args["expected_height_direction"] = torch.tensor([-1.0])
    out = compute_terrain_curriculum_moves(**args)
    assert out["controlled_down"].tolist() == [True]
    assert out["move_up"].tolist() == [True]


def test_stair_curriculum_ignores_episode_without_translation_command():
    args = _curriculum_inputs()
    args["cmd_mag"] = torch.tensor([0.0])
    args["forward_dist"] = torch.tensor([0.0])
    args["support_height_delta"] = torch.tensor([0.0])
    out = compute_terrain_curriculum_moves(**args)
    assert out["evaluation_eligible"].tolist() == [False]
    assert out["move_up"].tolist() == [False]
    assert out["move_down"].tolist() == [False]


def test_support_reference_ignores_unloaded_outlier():
    feet = torch.tensor([[[0.3, 0.2, 0.0], [0.3, -0.2, 0.0], [-0.3, 0.2, 0.0], [-0.3, -0.2, 1.0]]])
    out = update_contact_support_reference(
        foot_position_w=feet,
        contact=torch.tensor([[True, True, True, False]]),
        support_force=torch.tensor([[100.0, 100.0, 100.0, 0.0]]),
        previous_point_w=torch.zeros((1, 3)),
        previous_normal_w=torch.tensor([[0.0, 0.0, 1.0]]),
        alpha=1.0,
        foot_radius=0.0,
    )
    assert torch.allclose(out["point_w"][0, 2], torch.tensor(0.0), atol=1e-6)
    assert out["plane_valid"].tolist() == [True]
