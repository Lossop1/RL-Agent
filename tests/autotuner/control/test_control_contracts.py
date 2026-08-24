from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from autotuner.control import (
    BodyCommand,
    BodyTarget,
    ContactState,
    ContactTerrainEstimator,
    ContinuousTrotGait,
    FlatTerrain,
    FootPlan,
    GaitConfig,
    RobotState,
    StairTerrain,
    WholeBodyConfig,
)
from autotuner.control.manifest import build_manifest
from autotuner.control.mpc import CentroidalMpc, CentroidalMpcConfig


def _state() -> RobotState:
    return RobotState(
        time_s=0.0,
        q=np.zeros(12),
        dq=np.zeros(12),
        base_position=np.asarray([0.0, 0.0, 0.55]),
        base_velocity_body=np.zeros(3),
        base_rpy=np.zeros(3),
        base_angular_velocity_body=np.zeros(3),
        foot_positions_body=np.zeros((4, 3)),
        foot_velocities_body=np.zeros((4, 3)),
        contacts=ContactState(np.ones(4, dtype=bool), np.ones(4)),
    )


def test_contracts_copy_and_validate_shapes() -> None:
    state = _state()
    assert state.q.shape == (12,)
    assert BodyCommand(0.2, -0.1, 0.3).as_array().tolist() == [0.2, -0.1, 0.3]
    obstacle = ContactState(
        np.zeros(4, dtype=bool),
        np.zeros(4),
        np.asarray([True, False, False, False]),
        np.asarray([25.0, 0.0, 0.0, 0.0]),
    )
    assert obstacle.obstacle_contact.tolist() == [True, False, False, False]
    with pytest.raises(ValueError, match="shape"):
        RobotState(
            time_s=0.0,
            q=np.zeros(11),
            dq=state.dq,
            base_position=state.base_position,
            base_velocity_body=state.base_velocity_body,
            base_rpy=state.base_rpy,
            base_angular_velocity_body=state.base_angular_velocity_body,
            foot_positions_body=state.foot_positions_body,
            foot_velocities_body=state.foot_velocities_body,
            contacts=state.contacts,
        )


def test_gait_command_ramp_is_shared_by_preview_and_plan() -> None:
    gait = ContinuousTrotGait(
        np.zeros((4, 3)),
        GaitConfig(command_ramp_time=0.10),
    )
    command = BodyCommand(vx=0.4)
    initial = _state()
    assert gait.effective_command(initial, command).as_array().tolist() == [0.0, 0.0, 0.0]
    gait.plan(initial, command, BodyTarget(
        velocity_body=np.zeros(3),
        acceleration_body=np.zeros(3),
        height=0.55,
        rpy=np.zeros(3),
        angular_acceleration_body=np.zeros(3),
        centroidal_wrench=np.zeros(6),
    ), FlatTerrain())
    halfway = RobotState(
        time_s=0.05,
        q=initial.q,
        dq=initial.dq,
        base_position=initial.base_position,
        base_velocity_body=initial.base_velocity_body,
        base_rpy=initial.base_rpy,
        base_angular_velocity_body=initial.base_angular_velocity_body,
        foot_positions_body=initial.foot_positions_body,
        foot_velocities_body=initial.foot_velocities_body,
        contacts=initial.contacts,
    )
    assert gait.effective_command(halfway, command).vx == pytest.approx(0.2)


def test_manifest_rejects_files_outside_root(tmp_path) -> None:
    inside = tmp_path / "inside.txt"
    inside.write_text("ok", encoding="utf-8")
    manifest = build_manifest(tmp_path, [inside], artifact_type="test")
    assert manifest["files"][0]["path"] == "inside.txt"
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("no", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="outside root"):
            build_manifest(tmp_path, [outside], artifact_type="test")
    finally:
        outside.unlink()


def test_artifact_manifest_is_deterministic_and_has_no_runtime_fields(tmp_path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("stable\n", encoding="utf-8")

    first = build_manifest(tmp_path, [source], artifact_type="test", metadata={"z": 1})
    second = build_manifest(tmp_path, [source], artifact_type="test", metadata={"z": 1})

    assert first == second
    assert "generated_at" not in first
    assert "git_revision" not in first
    assert "python" not in first

    source.write_text("changed\n", encoding="utf-8")
    changed = build_manifest(tmp_path, [source], artifact_type="test", metadata={"z": 1})
    assert changed["artifact_digest"] != first["artifact_digest"]


def test_stair_height_matches_generated_riser_boundaries() -> None:
    terrain = StairTerrain(rise=0.15, run=0.30, start_x=0.20, max_steps=3)
    assert terrain.height_at(0.20 - 1.0e-9, 0.0) == 0.0
    assert terrain.height_at(0.20, 0.0) == 0.15
    assert terrain.height_at(0.50 - 1.0e-9, 0.0) == 0.15
    assert terrain.height_at(0.50, 0.0) == 0.30
    assert terrain.height_at(10.0, 0.0) == pytest.approx(0.45)


def test_stair_touchdown_is_projected_onto_the_next_support_surface() -> None:
    terrain = StairTerrain(rise=0.15, run=0.30, start_x=0.20, max_steps=8)
    upward = terrain.touchdown_target(
        np.asarray([0.16, -0.2]),
        np.asarray([0.176, -0.2]),
        np.asarray([0.25, 0.0]),
        0.042,
        current_support_height=0.0,
    )
    assert upward[0] > 0.20 + 0.042
    assert upward[2] == pytest.approx(0.15)

    downward = terrain.touchdown_target(
        np.asarray([2.34, 0.2]),
        np.asarray([2.29, 0.2]),
        np.asarray([-0.20, 0.0]),
        0.042,
        current_support_height=1.20,
    )
    assert downward[0] < 2.30 - 0.042
    assert downward[2] == pytest.approx(1.05)

    unchanged = terrain.touchdown_target(
        np.asarray([-0.20, 0.2]),
        np.asarray([-0.05, 0.2]),
        np.asarray([0.25, 0.0]),
        0.042,
        current_support_height=0.0,
    )
    assert unchanged.tolist() == pytest.approx([-0.05, 0.2, 0.0])


def test_stair_touchdown_cannot_skip_an_unconfirmed_tread() -> None:
    terrain = StairTerrain(rise=0.15, run=0.30, start_x=0.20, max_steps=8)
    target = terrain.touchdown_target(
        np.asarray([0.40, 0.0]),
        np.asarray([0.72, 0.0]),
        np.asarray([0.18, 0.0]),
        0.042,
        current_support_height=0.0,
    )

    assert 0.20 < target[0] < 0.50
    assert target[2] == pytest.approx(0.15)


def test_stair_climb_keeps_an_established_high_tread_under_a_slow_body() -> None:
    terrain = StairTerrain(rise=0.15, run=0.30, start_x=0.20, max_steps=8)
    target = terrain.touchdown_target(
        np.asarray([0.10, 0.0]),
        np.asarray([0.107, 0.0]),
        np.asarray([0.18, 0.0]),
        0.042,
        current_support_height=0.15,
    )

    assert target[0] > 0.20
    assert target[2] == pytest.approx(0.15)


def test_mpc_velocity_reference_has_bounded_position_and_yaw_lead() -> None:
    mpc = CentroidalMpc(
        CentroidalMpcConfig(max_position_error=0.15, max_yaw_error=0.35)
    )
    initial = _state()
    for index in range(101):
        state = RobotState(
            time_s=0.02 * index,
            q=initial.q,
            dq=initial.dq,
            base_position=initial.base_position,
            base_velocity_body=initial.base_velocity_body,
            base_rpy=initial.base_rpy,
            base_angular_velocity_body=initial.base_angular_velocity_body,
            foot_positions_body=initial.foot_positions_body,
            foot_velocities_body=initial.foot_velocities_body,
            contacts=initial.contacts,
        )
        mpc.step(state, BodyCommand(vx=1.0, wz=1.0), FlatTerrain(), nominal_height=0.55)

    assert np.linalg.norm(mpc._target_position_xy - initial.base_position[:2]) <= 0.15 + 1.0e-12
    assert abs(mpc._target_yaw - initial.base_rpy[2]) <= 0.35 + 1.0e-12


def test_mpc_forwards_support_pitch_to_the_body_target() -> None:
    mpc = CentroidalMpc(CentroidalMpcConfig())
    output = mpc.step(
        _state(),
        BodyCommand(),
        FlatTerrain(),
        nominal_height=0.55,
        support_pitch=-0.2,
    )

    assert output.target.rpy == pytest.approx([0.0, -0.2, 0.0])


def test_contact_height_cannot_replace_a_neighbouring_stair_tread() -> None:
    contacts = ContactState(np.ones(4, dtype=bool), np.full(4, 50.0))
    estimator = ContactTerrainEstimator(alpha=1.0, force_threshold=5.0, max_correction=0.025)
    estimator.update(contacts, np.full(4, 0.042 + 0.15), foot_radius=0.042)
    assert estimator.corrected_height(0, 0.0) == 0.0

    estimator.reset()
    estimator.update(contacts, np.full(4, 0.042 + 0.01), foot_radius=0.042)
    assert estimator.corrected_height(0, 0.0) == pytest.approx(0.01)


def test_stair_body_height_waits_for_established_support() -> None:
    terrain = StairTerrain(rise=0.15, run=0.30, start_x=0.20, max_steps=3)

    # The base is close enough that the old lookahead crossed the first riser,
    # but no high support has been established yet.
    assert terrain.body_height_target(0.15, 0.25, 0.55, support_height=0.0) == pytest.approx(0.55)
    assert terrain.body_height_target(0.15, 0.25, 0.55, support_height=0.15) == pytest.approx(0.70)


def test_contact_estimator_rejects_mixed_stair_levels_for_global_support() -> None:
    estimator = ContactTerrainEstimator(alpha=1.0, force_threshold=5.0)
    contacts = ContactState(
        np.ones(4, dtype=bool),
        np.asarray([50.0, 50.0, 50.0, 50.0]),
    )
    estimator.update(
        contacts,
        np.asarray([0.042, 0.192, 0.042, 0.192]),
        foot_radius=0.042,
    )
    assert estimator.support_height() is None
    assert estimator.support_pitch() == pytest.approx(0.0)

    no_support = ContactState(np.zeros(4, dtype=bool), np.zeros(4))
    estimator.update(no_support, np.zeros(4), foot_radius=0.042)
    assert estimator.support_height() is None


def test_contact_estimator_transitions_support_height_directionally() -> None:
    estimator = ContactTerrainEstimator(alpha=1.0, force_threshold=5.0)
    contacts = ContactState(
        np.ones(4, dtype=bool),
        np.asarray([50.0, 50.0, 50.0, 50.0]),
    )
    estimator.update(
        contacts,
        np.full(4, 0.042 + 0.15),
        foot_radius=0.042,
    )
    estimator.update(
        contacts,
        np.asarray([0.042 + 0.15, 0.042 + 0.15, 0.042, 0.042]),
        foot_radius=0.042,
        support_direction=-0.2,
    )
    assert estimator.support_height() == pytest.approx(0.0)

    estimator.reset()
    estimator.update(
        contacts,
        np.full(4, 0.042),
        foot_radius=0.042,
    )
    estimator.update(
        contacts,
        np.asarray([0.042, 0.042, 0.042 + 0.15, 0.042 + 0.15]),
        foot_radius=0.042,
        support_direction=0.2,
    )
    assert estimator.support_height() == pytest.approx(0.15)


def test_contact_estimator_does_not_switch_for_a_single_new_support_foot() -> None:
    estimator = ContactTerrainEstimator(alpha=1.0, force_threshold=5.0)
    contacts = ContactState(np.ones(4, dtype=bool), np.full(4, 50.0))
    estimator.update(
        contacts,
        np.full(4, 0.042),
        foot_radius=0.042,
    )
    estimator.update(
        contacts,
        np.asarray([0.042 + 0.15, 0.042, 0.042, 0.042]),
        foot_radius=0.042,
        support_direction=0.2,
    )
    assert estimator.support_height() == pytest.approx(0.0)


def test_contact_estimator_derives_pitch_from_staggered_support() -> None:
    estimator = ContactTerrainEstimator(alpha=1.0, force_threshold=5.0)
    contacts = ContactState(
        np.asarray([True, False, False, True]),
        np.asarray([50.0, 0.0, 0.0, 50.0]),
    )
    estimator.update(
        contacts,
        np.asarray([0.192, 0.0, 0.042, 0.042]),
        foot_radius=0.042,
        foot_world_xy=np.asarray(
            [[0.30, 0.2], [0.30, -0.2], [-0.30, 0.2], [-0.30, -0.2]]
        ),
        reference_x=0.0,
    )
    assert estimator.support_pitch() == pytest.approx(-np.arctan(0.25))


def test_swing_clearance_is_applied_after_terrain_height() -> None:
    feet = np.asarray(
        [[0.3, 0.2, -0.5], [0.3, -0.2, -0.5], [-0.3, 0.2, -0.5], [-0.3, -0.2, -0.5]]
    )
    gait = ContinuousTrotGait(
        feet,
        GaitConfig(command_ramp_time=0.1, swing_clearance=0.06, foot_radius=0.042),
    )
    target = BodyTarget(
        velocity_body=np.zeros(3),
        acceleration_body=np.zeros(3),
        height=0.55,
        rpy=np.zeros(3),
        angular_acceleration_body=np.zeros(3),
        centroidal_wrench=np.zeros(6),
    )
    initial = _state()
    first = RobotState(
        time_s=0.1,
        q=initial.q,
        dq=initial.dq,
        base_position=initial.base_position,
        base_velocity_body=initial.base_velocity_body,
        base_rpy=initial.base_rpy,
        base_angular_velocity_body=initial.base_angular_velocity_body,
        foot_positions_body=feet,
        foot_velocities_body=initial.foot_velocities_body,
        contacts=initial.contacts,
    )
    gait.plan(first, BodyCommand(vx=0.4), target, FlatTerrain())
    second = RobotState(
        time_s=0.2,
        q=first.q,
        dq=first.dq,
        base_position=first.base_position,
        base_velocity_body=first.base_velocity_body,
        base_rpy=first.base_rpy,
        base_angular_velocity_body=first.base_angular_velocity_body,
        foot_positions_body=first.foot_positions_body,
        foot_velocities_body=first.foot_velocities_body,
        contacts=first.contacts,
    )
    plan = gait.plan(second, BodyCommand(vx=0.4), target, FlatTerrain())
    baseline = 0.042 - target.height
    assert plan.swing_mask[[0, 3]].all()
    assert not plan.height_transition_mask.any()
    assert np.all(plan.positions_body[[0, 3], 2] > baseline + 0.03)
    assert np.isfinite(plan.velocities_body).all()


def test_stair_gait_uses_full_body_rotation_for_measured_terrain_height() -> None:
    feet = np.asarray(
        [[0.3, 0.2, -0.5], [0.3, -0.2, -0.5], [-0.3, 0.2, -0.5], [-0.3, -0.2, -0.5]]
    )
    gait = ContinuousTrotGait(
        feet,
        GaitConfig(command_ramp_time=0.1),
    )
    state = _state()
    state = RobotState(
        time_s=0.0,
        q=state.q,
        dq=state.dq,
        base_position=state.base_position,
        base_velocity_body=state.base_velocity_body,
        base_rpy=np.asarray([0.0, -0.2, 0.0]),
        base_angular_velocity_body=state.base_angular_velocity_body,
        foot_positions_body=feet,
        foot_velocities_body=state.foot_velocities_body,
        contacts=state.contacts,
    )
    target = BodyTarget(
        velocity_body=np.zeros(3),
        acceleration_body=np.zeros(3),
        height=0.55,
        rpy=np.zeros(3),
        angular_acceleration_body=np.zeros(3),
        centroidal_wrench=np.zeros(6),
    )

    gait._motion_scale = 1.0
    gait.plan(
        state,
        BodyCommand(vx=0.4),
        target,
        StairTerrain(rise=0.15, run=0.30, start_x=10.0),
    )

    expected_front_height = (
        state.base_position[2]
        - np.sin(-0.2) * feet[0, 0]
        + np.cos(-0.2) * feet[0, 2]
        - gait.config.foot_radius
    )
    assert gait._stance_terrain_height[0] == pytest.approx(expected_front_height)


def test_stair_height_transition_does_not_follow_a_low_body() -> None:
    feet = np.asarray(
        [[0.10, 0.2, -0.408], [0.10, -0.2, -0.408], [-0.3, 0.2, -0.408], [-0.3, -0.2, -0.408]]
    )
    gait = ContinuousTrotGait(
        feet,
        GaitConfig(
            period=0.72,
            period_min=0.72,
            duty_factor=0.68,
            command_ramp_time=0.1,
            swing_clearance=0.06,
            foot_radius=0.042,
        ),
    )
    state = RobotState(
        time_s=0.0,
        q=np.zeros(12),
        dq=np.zeros(12),
        base_position=np.asarray([0.0, 0.0, 0.45]),
        base_velocity_body=np.asarray([0.4, 0.0, 0.0]),
        base_rpy=np.zeros(3),
        base_angular_velocity_body=np.zeros(3),
        foot_positions_body=feet,
        foot_velocities_body=np.zeros((4, 3)),
        contacts=ContactState(np.ones(4, dtype=bool), np.full(4, 50.0)),
    )
    target = BodyTarget(
        velocity_body=np.zeros(3),
        acceleration_body=np.zeros(3),
        height=0.55,
        rpy=np.zeros(3),
        angular_acceleration_body=np.zeros(3),
        centroidal_wrench=np.zeros(6),
    )
    gait.plan(state, BodyCommand(), target, FlatTerrain())
    gait.phase = (1.0 - gait.config.duty_factor) - 1.0e-3
    gait._motion_scale = 1.0
    gait._was_moving = True
    plan = gait.plan(
        state,
        BodyCommand(vx=0.4),
        target,
        StairTerrain(rise=0.15, run=0.30, start_x=0.20),
    )

    assert plan.height_transition_mask[0]
    desired_world_center_z = state.base_position[2] + plan.positions_body[0, 2]
    assert desired_world_center_z == pytest.approx(0.15 + gait.config.foot_radius, abs=1.0e-4)


def test_stair_swing_lifts_before_horizontal_riser_crossing() -> None:
    feet = np.asarray(
        [[0.10, 0.2, -0.408], [0.10, -0.2, -0.408], [-0.3, 0.2, -0.408], [-0.3, -0.2, -0.408]]
    )
    gait = ContinuousTrotGait(
        feet,
        GaitConfig(
            period=1.0,
            period_min=1.0,
            duty_factor=0.68,
            command_ramp_time=0.1,
            swing_clearance=0.06,
            foot_radius=0.042,
            obstacle_lift_delay=0.20,
        ),
    )
    target = BodyTarget(
        velocity_body=np.zeros(3),
        acceleration_body=np.zeros(3),
        height=0.55,
        rpy=np.zeros(3),
        angular_acceleration_body=np.zeros(3),
        centroidal_wrench=np.zeros(6),
    )
    initial = RobotState(
        time_s=0.0,
        q=np.zeros(12),
        dq=np.zeros(12),
        base_position=np.asarray([0.0, 0.0, 0.45]),
        base_velocity_body=np.asarray([0.4, 0.0, 0.0]),
        base_rpy=np.zeros(3),
        base_angular_velocity_body=np.zeros(3),
        foot_positions_body=feet,
        foot_velocities_body=np.zeros((4, 3)),
        contacts=ContactState(np.ones(4, dtype=bool), np.full(4, 50.0)),
    )
    gait.plan(initial, BodyCommand(), target, FlatTerrain())
    gait._motion_scale = 1.0
    gait._was_moving = True
    gait.phase = 0.10
    plan = gait.plan(
        initial,
        BodyCommand(vx=0.4),
        target,
        StairTerrain(rise=0.15, run=0.30, start_x=0.20),
    )

    assert plan.height_transition_mask[0]
    assert plan.positions_body[0, 0] < feet[0, 0] + 0.02


def test_stair_height_change_keeps_the_other_diagonal_foot_in_support() -> None:
    feet = np.asarray(
        [
            [0.2939, 0.2082, -0.5055],
            [0.2941, -0.2082, -0.5057],
            [-0.3144, 0.2082, -0.5049],
            [-0.3141, -0.2082, -0.5051],
        ]
    )
    gait = ContinuousTrotGait(
        feet,
        GaitConfig(
            period=1.10,
            period_min=1.10,
            duty_factor=0.68,
            command_ramp_time=0.1,
            stair_max_swing_legs=1,
        ),
    )
    target = BodyTarget(
        velocity_body=np.zeros(3),
        acceleration_body=np.zeros(3),
        height=0.55,
        rpy=np.zeros(3),
        angular_acceleration_body=np.zeros(3),
        centroidal_wrench=np.zeros(6),
    )
    state = RobotState(
        time_s=0.0,
        q=np.zeros(12),
        dq=np.zeros(12),
        base_position=np.asarray([-0.25, 0.0, 0.547]),
        base_velocity_body=np.asarray([0.18, 0.0, 0.0]),
        base_rpy=np.zeros(3),
        base_angular_velocity_body=np.zeros(3),
        foot_positions_body=feet,
        foot_velocities_body=np.zeros((4, 3)),
        contacts=ContactState(np.ones(4, dtype=bool), np.full(4, 50.0)),
    )
    gait.plan(state, BodyCommand(), target, FlatTerrain())
    gait._motion_scale = 1.0
    gait._was_moving = True
    gait.phase = 0.55

    plan = gait.plan(
        state,
        BodyCommand(vx=0.18),
        target,
        StairTerrain(rise=0.15, run=0.30, start_x=0.20),
    )

    assert plan.swing_mask.tolist() == [False, True, False, False]
    assert plan.height_transition_mask[1]
    assert plan.world_anchored_mask[1]
    assert plan.contact_weights[2] == pytest.approx(1.0)


def test_height_transition_has_a_targeted_vertical_wbc_weight() -> None:
    config = WholeBodyConfig(height_transition_vertical_weight=50.0)
    assert config.height_transition_vertical_weight == 50.0


def test_contact_acquisition_has_an_explicit_three_axis_wbc_weight() -> None:
    config = WholeBodyConfig(
        contact_acquisition_horizontal_weight=240.0,
        contact_acquisition_vertical_weight=360.0,
    )

    assert config.contact_acquisition_horizontal_weight == 240.0
    assert config.contact_acquisition_vertical_weight == 360.0


def test_active_contact_has_configured_minimum_normal_force() -> None:
    config = WholeBodyConfig(minimum_contact_normal_force=10.0)
    assert config.minimum_contact_normal_force == pytest.approx(10.0)


def test_contact_weight_unloads_before_liftoff_and_loads_after_touchdown() -> None:
    feet = np.asarray(
        [[0.3, 0.2, -0.5], [0.3, -0.2, -0.5], [-0.3, 0.2, -0.5], [-0.3, -0.2, -0.5]]
    )
    gait = ContinuousTrotGait(
        feet,
        GaitConfig(duty_factor=0.68, contact_blend_phase=0.08),
    )
    swing_end = 1.0 - gait.config.duty_factor
    phases = np.asarray(
        [0.0, swing_end, swing_end + gait.config.contact_blend_phase, 1.0 - 1.0e-9]
    )
    weights = gait._contact_weights(phases, moving=True)

    assert weights[0] == pytest.approx(0.0)
    assert weights[1] == pytest.approx(0.0)
    assert weights[2] == pytest.approx(1.0)
    assert weights[3] == pytest.approx(0.0, abs=1.0e-12)


def test_diagonal_swing_legs_each_latch_their_own_touchdown_target() -> None:
    """A diagonal swing must use one shared previous-cycle snapshot."""

    feet = np.asarray(
        [[0.3, 0.2, -0.5], [0.3, -0.2, -0.5], [-0.3, 0.2, -0.5], [-0.3, -0.2, -0.5]]
    )
    gait = ContinuousTrotGait(
        feet,
        GaitConfig(
            period=1.0,
            period_min=1.0,
            duty_factor=0.68,
            command_ramp_time=0.1,
        ),
    )
    initial = _state()
    target = BodyTarget(
        velocity_body=np.zeros(3),
        acceleration_body=np.zeros(3),
        height=0.55,
        rpy=np.zeros(3),
        angular_acceleration_body=np.zeros(3),
        centroidal_wrench=np.zeros(6),
    )
    gait.plan(initial, BodyCommand(), target, FlatTerrain())
    gait._motion_scale = 1.0
    gait._was_moving = True
    gait.phase = 0.10

    plan = gait.plan(
        initial,
        BodyCommand(vx=0.4),
        target,
        # The proposed footholds are still well before the first riser.  The
        # staircase therefore must preserve the ordinary diagonal swing until
        # geometry presents a real height transition.
        StairTerrain(rise=0.15, run=0.30, start_x=10.0),
    )

    assert plan.swing_mask.tolist() == [True, False, False, True]
    assert np.allclose(
        plan.swing_start_world_xy[[0, 3]],
        np.zeros((2, 2)),
    )
    assert not np.allclose(
        plan.touchdown_targets_world_xy[0],
        plan.touchdown_targets_world_xy[3],
    )


def test_phase_freezes_for_missing_touchdown_support_before_next_swing() -> None:
    feet = np.asarray(
        [[0.3, 0.2, -0.5], [0.3, -0.2, -0.5], [-0.3, 0.2, -0.5], [-0.3, -0.2, -0.5]]
    )
    gait = ContinuousTrotGait(
        feet,
        GaitConfig(
            period=1.0,
            period_min=1.0,
            duty_factor=0.68,
            command_ramp_time=0.1,
        ),
    )
    target = BodyTarget(
        velocity_body=np.zeros(3),
        acceleration_body=np.zeros(3),
        height=0.55,
        rpy=np.zeros(3),
        angular_acceleration_body=np.zeros(3),
        centroidal_wrench=np.zeros(6),
    )
    initial = _state()
    stair_terrain = StairTerrain(rise=0.15, run=0.30, start_x=10.0)
    gait.plan(initial, BodyCommand(), target, stair_terrain)
    gait._motion_scale = 1.0
    gait._was_moving = True
    gait.phase = 0.31
    gait._was_swing[:] = np.asarray([True, False, False, True])

    no_touchdown = RobotState(
        time_s=0.02,
        q=initial.q,
        dq=initial.dq,
        base_position=initial.base_position,
        base_velocity_body=initial.base_velocity_body,
        base_rpy=initial.base_rpy,
        base_angular_velocity_body=initial.base_angular_velocity_body,
        foot_positions_body=initial.foot_positions_body,
        foot_velocities_body=initial.foot_velocities_body,
        contacts=ContactState(
            np.asarray([False, True, True, False]),
            np.asarray([0.0, 40.0, 40.0, 0.0]),
        ),
    )
    first = gait.plan(no_touchdown, BodyCommand(vx=0.4), target, stair_terrain)
    phase_after_detection = gait.phase
    second = gait.plan(
        RobotState(
            time_s=0.12,
            q=no_touchdown.q,
            dq=no_touchdown.dq,
            base_position=no_touchdown.base_position,
            base_velocity_body=no_touchdown.base_velocity_body,
            base_rpy=no_touchdown.base_rpy,
            base_angular_velocity_body=no_touchdown.base_angular_velocity_body,
            foot_positions_body=no_touchdown.foot_positions_body,
            foot_velocities_body=no_touchdown.foot_velocities_body,
            contacts=no_touchdown.contacts,
        ),
        BodyCommand(vx=0.4),
        target,
        stair_terrain,
    )

    assert first.swing_mask.tolist() == [True, False, False, True]
    assert second.swing_mask.tolist() == [True, False, False, True]
    assert gait.phase == pytest.approx(phase_after_detection)

    landed = RobotState(
        time_s=0.22,
        q=no_touchdown.q,
        dq=no_touchdown.dq,
        base_position=no_touchdown.base_position,
        base_velocity_body=no_touchdown.base_velocity_body,
        base_rpy=no_touchdown.base_rpy,
        base_angular_velocity_body=no_touchdown.base_angular_velocity_body,
        foot_positions_body=no_touchdown.foot_positions_body,
        foot_velocities_body=no_touchdown.foot_velocities_body,
        contacts=ContactState(np.ones(4, dtype=bool), np.full(4, 40.0)),
    )
    landed_plan = gait.plan(landed, BodyCommand(vx=0.4), target, stair_terrain)
    assert not landed_plan.swing_mask[[0, 3]].any()
    assert np.count_nonzero(landed_plan.swing_mask) <= 1
    gait.plan(
        RobotState(
            time_s=0.32,
            q=landed.q,
            dq=landed.dq,
            base_position=landed.base_position,
            base_velocity_body=landed.base_velocity_body,
            base_rpy=landed.base_rpy,
            base_angular_velocity_body=landed.base_angular_velocity_body,
            foot_positions_body=landed.foot_positions_body,
            foot_velocities_body=landed.foot_velocities_body,
            contacts=landed.contacts,
        ),
        BodyCommand(vx=0.4),
        target,
        stair_terrain,
    )
    assert gait.phase > phase_after_detection


def test_flat_missing_touchdown_is_world_anchored_for_capture() -> None:
    feet = np.asarray(
        [[0.3, 0.2, -0.5], [0.3, -0.2, -0.5], [-0.3, 0.2, -0.5], [-0.3, -0.2, -0.5]]
    )
    gait = ContinuousTrotGait(
        feet,
        GaitConfig(
            period=1.0,
            period_min=1.0,
            duty_factor=0.68,
            command_ramp_time=0.1,
        ),
    )
    initial = _state()
    target = BodyTarget(
        velocity_body=np.zeros(3),
        acceleration_body=np.zeros(3),
        height=0.55,
        rpy=np.zeros(3),
        angular_acceleration_body=np.zeros(3),
        centroidal_wrench=np.zeros(6),
    )
    gait.plan(initial, BodyCommand(), target, FlatTerrain())
    gait._motion_scale = 1.0
    gait._was_moving = True
    gait.phase = 0.31
    gait._was_swing[:] = np.asarray([True, False, False, True])
    no_touchdown = replace(
        initial,
        time_s=0.02,
        contacts=ContactState(
            np.asarray([False, True, True, False]),
            np.asarray([0.0, 40.0, 40.0, 0.0]),
        ),
    )
    plan = gait.plan(no_touchdown, BodyCommand(vx=0.4), target, FlatTerrain())

    assert plan.world_anchored_mask.tolist() == [True, False, False, True]


def test_flat_pending_touchdown_freezes_phase_and_consumes_swing_budget() -> None:
    feet = np.asarray(
        [[0.3, 0.2, -0.5], [0.3, -0.2, -0.5], [-0.3, 0.2, -0.5], [-0.3, -0.2, -0.5]]
    )
    gait = ContinuousTrotGait(
        feet,
        GaitConfig(
            period=1.0,
            period_min=1.0,
            duty_factor=0.68,
            command_ramp_time=0.1,
            max_swing_legs=1,
        ),
    )
    gait._last_time = 0.0
    gait._motion_scale = 1.0
    gait._was_moving = True
    gait._was_swing[3] = True
    gait._contact_hold[3] = True
    gait._height_initialized = True
    gait._swing_start_world_xy[3] = feet[3, :2]
    gait._swing_target_world_xy[3] = feet[3, :2]
    gait._swing_start_center_height[3] = 0.05
    gait._swing_target_center_height[3] = 0.05
    gait.phase = 0.33
    state = replace(
        _state(),
        time_s=0.2,
        foot_positions_body=feet,
        contacts=ContactState(
            np.asarray([True, True, True, False]),
            np.asarray([40.0, 40.0, 40.0, 0.0]),
        ),
    )
    target = BodyTarget(
        velocity_body=np.zeros(3),
        acceleration_body=np.zeros(3),
        height=0.55,
        rpy=np.zeros(3),
        angular_acceleration_body=np.zeros(3),
        centroidal_wrench=np.zeros(6),
    )

    plan = gait.plan(state, BodyCommand(vy=0.12), target, FlatTerrain())

    assert gait.phase == pytest.approx(0.33)
    assert plan.swing_mask.tolist() == [False, False, False, True]
    assert plan.contact_acquisition_mask.tolist() == [False, False, False, True]
    assert plan.world_anchored_mask[3]


def test_single_swing_budget_keeps_one_leg_for_the_whole_phase_window() -> None:
    feet = np.asarray(
        [[0.3, 0.2, -0.5], [0.3, -0.2, -0.5], [-0.3, 0.2, -0.5], [-0.3, -0.2, -0.5]]
    )
    gait = ContinuousTrotGait(
        feet,
        GaitConfig(period=1.0, period_min=1.0, duty_factor=0.68, max_swing_legs=1),
    )
    gait._motion_scale = 1.0
    gait._was_moving = True
    gait._height_initialized = True
    gait.phase = 0.10
    state = replace(_state(), foot_positions_body=feet)
    target = BodyTarget(
        velocity_body=np.zeros(3),
        acceleration_body=np.zeros(3),
        height=0.55,
        rpy=np.zeros(3),
        angular_acceleration_body=np.zeros(3),
        centroidal_wrench=np.zeros(6),
    )

    first = gait.plan(state, BodyCommand(vx=0.1), target, FlatTerrain())
    gait.phase = 0.10
    second = gait.plan(state, BodyCommand(vx=0.1), target, FlatTerrain())

    assert np.count_nonzero(first.swing_mask) == 1
    np.testing.assert_array_equal(second.swing_mask, first.swing_mask)


def test_single_swing_budget_is_fair_across_all_four_legs() -> None:
    feet = np.asarray(
        [[0.3, 0.2, -0.5], [0.3, -0.2, -0.5], [-0.3, 0.2, -0.5], [-0.3, -0.2, -0.5]]
    )
    gait = ContinuousTrotGait(
        feet,
        GaitConfig(period=1.0, period_min=1.0, duty_factor=0.68, max_swing_legs=1),
    )
    gait._motion_scale = 1.0
    gait._was_moving = True
    gait._height_initialized = True
    state = replace(_state(), foot_positions_body=feet)
    target = BodyTarget(
        velocity_body=np.zeros(3),
        acceleration_body=np.zeros(3),
        height=0.55,
        rpy=np.zeros(3),
        angular_acceleration_body=np.zeros(3),
        centroidal_wrench=np.zeros(6),
    )
    selected: list[int] = []

    for swing_phase, stance_phase in ((0.10, 0.40), (0.60, 0.90)) * 2:
        gait.phase = swing_phase
        plan = gait.plan(state, BodyCommand(vx=0.1), target, FlatTerrain())
        selected.append(int(np.flatnonzero(plan.swing_mask)[0]))
        gait.phase = stance_phase
        stance = gait.plan(state, BodyCommand(vx=0.1), target, FlatTerrain())
        assert not stance.swing_mask.any()

    assert set(selected) == {0, 1, 2, 3}
    assert gait._scheduled_swing_count.tolist() == [1, 1, 1, 1]


def test_single_swing_budget_preserves_the_feasible_measured_support_set() -> None:
    feet = np.asarray(
        [[0.3, 0.2, -0.5], [0.3, -0.2, -0.5], [-0.3, 0.2, -0.5], [-0.3, -0.2, -0.5]]
    )
    measured_feet = np.asarray(
        [[0.30, 0.20, -0.5], [0.00, -0.20, -0.5], [0.00, 0.20, -0.5], [0.00, -0.10, -0.5]]
    )
    gait = ContinuousTrotGait(
        feet,
        GaitConfig(period=1.0, period_min=1.0, duty_factor=0.68, max_swing_legs=1),
    )
    gait._motion_scale = 1.0
    gait._was_moving = True
    gait._height_initialized = True
    gait.phase = 0.10
    state = replace(
        _state(),
        foot_positions_body=measured_feet,
        contacts=ContactState(
            np.ones(4, dtype=bool),
            np.full(4, 40.0),
        ),
    )
    target = BodyTarget(
        velocity_body=np.zeros(3),
        acceleration_body=np.zeros(3),
        height=0.55,
        rpy=np.zeros(3),
        angular_acceleration_body=np.zeros(3),
        centroidal_wrench=np.zeros(6),
    )

    plan = gait.plan(state, BodyCommand(vx=0.1), target, FlatTerrain())

    assert plan.swing_mask.tolist() == [False, False, False, True]


def test_single_swing_selection_is_lateral_mirror_equivariant() -> None:
    feet = np.asarray(
        [[0.32, 0.23, -0.5], [0.28, -0.18, -0.5], [-0.31, 0.19, -0.5], [-0.27, -0.24, -0.5]]
    )
    contacts = ContactState(
        np.ones(4, dtype=bool),
        np.asarray([35.0, 55.0, 45.0, 25.0]),
    )
    state = replace(
        _state(),
        base_position=np.asarray([0.04, 0.03, 0.55]),
        foot_positions_body=feet,
        contacts=contacts,
    )
    mirror = np.asarray([1, 0, 3, 2])
    mirrored_feet = feet[mirror].copy()
    mirrored_feet[:, 1] *= -1.0
    mirrored_state = replace(
        state,
        base_position=state.base_position * np.asarray([1.0, -1.0, 1.0]),
        foot_positions_body=mirrored_feet,
        contacts=ContactState(
            contacts.in_contact[mirror],
            contacts.normal_force[mirror],
        ),
    )
    target = BodyTarget(
        velocity_body=np.zeros(3),
        acceleration_body=np.zeros(3),
        height=0.55,
        rpy=np.zeros(3),
        angular_acceleration_body=np.zeros(3),
        centroidal_wrench=np.zeros(6),
    )
    original = ContinuousTrotGait(
        feet,
        GaitConfig(period=1.0, period_min=1.0, duty_factor=0.68, max_swing_legs=1),
    )
    mirrored = ContinuousTrotGait(
        mirrored_feet,
        GaitConfig(period=1.0, period_min=1.0, duty_factor=0.68, max_swing_legs=1),
    )
    for gait, phase in ((original, 0.10), (mirrored, 0.60)):
        gait._motion_scale = 1.0
        gait._was_moving = True
        gait._height_initialized = True
        gait.phase = phase

    original_plan = original.plan(
        state,
        BodyCommand(vy=0.1),
        target,
        FlatTerrain(),
    )
    mirrored_plan = mirrored.plan(
        mirrored_state,
        BodyCommand(vy=-0.1),
        target,
        FlatTerrain(),
    )

    np.testing.assert_array_equal(
        mirrored_plan.swing_mask,
        original_plan.swing_mask[mirror],
    )


def test_lateral_recovery_does_not_exceed_physical_swing_budget() -> None:
    # Measured geometry from the flat-left failure at 2.00 s. The body is
    # 0.123 m from the diagonal support segment, just outside the 0.12 m
    # progress threshold. That must brake the body, not create a second swing.
    feet = np.asarray(
        [
            [0.2830237658, 0.1346629300, -0.4935380213],
            [0.2924429257, -0.3137796121, -0.5062611486],
            [-0.3181332035, 0.0433018909, -0.5040226728],
            [-0.3273076546, -0.3533554138, -0.4957364077],
        ]
    )
    gait = ContinuousTrotGait(
        feet,
        GaitConfig(
            period=1.0,
            period_min=1.0,
            duty_factor=0.68,
            command_ramp_time=0.1,
            max_swing_legs=1,
        ),
    )
    gait._last_time = 0.0
    gait._motion_scale = 1.0
    gait._was_moving = True
    gait.phase = 0.1194815995
    state = replace(
        _state(),
        foot_positions_body=feet,
        contacts=ContactState(
            np.asarray([False, True, True, False]),
            np.asarray([0.0, 40.0, 40.0, 0.0]),
        ),
    )
    target = BodyTarget(
        velocity_body=np.zeros(3),
        acceleration_body=np.zeros(3),
        height=0.55,
        rpy=np.zeros(3),
        angular_acceleration_body=np.zeros(3),
        centroidal_wrench=np.zeros(6),
    )

    plan = gait.plan(state, BodyCommand(vy=0.12), target, FlatTerrain())

    assert plan.swing_mask.tolist() == [True, False, False, False]
    assert not plan.contact_acquisition_mask.any()


def test_lateral_recovery_captures_missing_support_at_measured_world_xy() -> None:
    # Measured geometry from the flat-right failure at 2.02 s. Leg 2 is an
    # airborne stance leg selected only to restore support. Sending it to the
    # ordinary Raibert foothold would add 0.208 m of horizontal travel while
    # the base already has marginal support, so recovery must land vertically.
    nominal_feet = np.asarray(
        [
            [0.2939116450, 0.2082, -0.50550234],
            [0.2941410400, -0.2082, -0.50571686],
            [-0.3143683600, 0.2082, -0.50488988],
            [-0.3141357849, -0.2082, -0.50510784],
        ]
    )
    gait = ContinuousTrotGait(
        nominal_feet,
        GaitConfig(
            period=0.72,
            period_min=0.58,
            period_slope=0.05,
            duty_factor=0.68,
            command_ramp_time=0.30,
            foothold_velocity_gain=0.06,
            max_swing_legs=1,
        ),
    )
    gait._last_time = 2.02
    gait._motion_scale = 1.0
    gait._was_moving = True
    gait.phase = 0.963394
    state = RobotState(
        time_s=2.02,
        q=np.zeros(12),
        dq=np.zeros(12),
        base_position=np.asarray(
            [0.13367877068177897, -0.2308497786770929, 0.5325131162072422]
        ),
        base_velocity_body=np.asarray(
            [0.27561824506153754, -0.39148320485863364, -0.0016140553619828576]
        ),
        base_rpy=np.asarray(
            [0.00030227758158984567, -0.003909822420287065, -0.00012135823535885243]
        ),
        base_angular_velocity_body=np.asarray(
            [-0.12544594011975657, -0.15729572979538972, -0.048782668719043076]
        ),
        foot_positions_body=np.asarray(
            [
                [0.17422496624166423, 0.4239961735409213, -0.4912476388024559],
                [0.20280668595446835, -0.12876049591176592, -0.4979019353632984],
                [-0.42193298628291864, 0.372142803555328, -0.47137687719906407],
                [-0.44592426424411863, 0.009169639816084381, -0.48838531538410257],
            ]
        ),
        foot_velocities_body=np.zeros((4, 3)),
        contacts=ContactState(
            np.asarray([True, True, False, False]),
            np.asarray([66.6375474017013, 281.89336856378077, 0.0, 0.0]),
        ),
    )
    target = BodyTarget(
        velocity_body=np.zeros(3),
        acceleration_body=np.zeros(3),
        height=0.5471961541,
        rpy=np.zeros(3),
        angular_acceleration_body=np.zeros(3),
        centroidal_wrench=np.zeros(6),
    )

    plan = gait.plan(state, BodyCommand(vy=-0.12), target, FlatTerrain())
    yaw = float(state.base_rpy[2])
    yaw_rotation = np.asarray(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    )
    measured_world_xy = (
        state.base_position[:2]
        + yaw_rotation @ state.foot_positions_body[2, :2]
    )

    assert plan.swing_mask.tolist() == [False, False, True, False]
    assert plan.contact_acquisition_mask.tolist() == [False, False, True, False]
    assert plan.world_anchored_mask.tolist() == [False, False, True, False]
    np.testing.assert_allclose(plan.swing_start_world_xy[2], measured_world_xy)
    np.testing.assert_allclose(plan.touchdown_targets_world_xy[2], measured_world_xy)
    assert plan.terrain_heights[2] == pytest.approx(0.0)


def test_world_anchored_foot_is_exact_under_full_base_rotation() -> None:
    feet = np.asarray(
        [[0.3, 0.2, -0.5], [0.3, -0.2, -0.5], [-0.3, 0.2, -0.5], [-0.3, -0.2, -0.5]]
    )
    gait = ContinuousTrotGait(feet, GaitConfig(max_swing_legs=1))
    gait._last_time = 0.0
    gait._motion_scale = 1.0
    gait._was_moving = True
    gait._was_swing[0] = True
    gait._contact_hold[0] = True
    gait._height_initialized = True
    gait.phase = 0.55
    gait._swing_start_world_xy[0] = np.asarray([0.42, -0.05])
    gait._swing_target_world_xy[0] = np.asarray([0.42, -0.05])
    gait._swing_start_center_height[0] = 0.042
    gait._swing_target_center_height[0] = 0.042
    state = replace(
        _state(),
        time_s=0.1,
        base_position=np.asarray([0.2, -0.3, 0.55]),
        base_velocity_body=np.asarray([0.3, -0.2, 0.1]),
        base_rpy=np.asarray([0.25, -0.18, 0.31]),
        base_angular_velocity_body=np.asarray([0.4, -0.3, 0.2]),
        foot_positions_body=feet,
        contacts=ContactState(
            np.asarray([False, True, True, True]),
            np.asarray([0.0, 40.0, 40.0, 40.0]),
        ),
    )
    target = BodyTarget(
        velocity_body=np.zeros(3),
        acceleration_body=np.zeros(3),
        height=0.55,
        rpy=np.zeros(3),
        angular_acceleration_body=np.zeros(3),
        centroidal_wrench=np.zeros(6),
    )

    plan = gait.plan(state, BodyCommand(vy=0.12), target, FlatTerrain())
    roll, pitch, yaw = state.base_rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotation = np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    desired_world_position = state.base_position + rotation @ plan.positions_body[0]
    desired_world_velocity = (
        rotation @ state.base_velocity_body
        + np.cross(
            rotation @ state.base_angular_velocity_body,
            rotation @ plan.positions_body[0],
        )
        + rotation @ plan.velocities_body[0]
    )

    assert plan.world_anchored_mask.tolist() == [True, False, False, False]
    np.testing.assert_allclose(
        desired_world_position,
        np.asarray([0.42, -0.05, 0.042]),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(desired_world_velocity, np.zeros(3), atol=1.0e-12)


def test_stair_height_transfer_keeps_the_selected_leg_exclusive() -> None:
    feet = np.asarray(
        [[0.3, 0.2, -0.5], [0.3, -0.2, -0.5], [-0.3, 0.2, -0.5], [-0.3, -0.2, -0.5]]
    )
    gait = ContinuousTrotGait(
        feet,
        GaitConfig(
            period=1.0,
            period_min=1.0,
            duty_factor=0.68,
            stair_max_swing_legs=1,
        ),
    )
    initial = replace(
        _state(),
        contacts=ContactState(
            np.asarray([True, False, True, True]),
            np.asarray([40.0, 0.0, 40.0, 40.0]),
        ),
    )
    target = BodyTarget(
        velocity_body=np.zeros(3),
        acceleration_body=np.zeros(3),
        height=0.55,
        rpy=np.zeros(3),
        angular_acceleration_body=np.zeros(3),
        centroidal_wrench=np.zeros(6),
    )
    gait._motion_scale = 1.0
    gait._was_moving = True
    gait.phase = 0.55
    gait._height_initialized = True
    gait._stance_terrain_height[:] = 0.0
    gait._height_transition[1] = True
    gait._was_swing[1] = True

    plan = gait.plan(
        initial,
        BodyCommand(vx=0.2),
        target,
        StairTerrain(rise=0.15, run=0.30, start_x=10.0),
    )

    assert plan.swing_mask.tolist() == [False, True, False, False]
    assert plan.height_transition_mask[1]


def test_single_correct_stair_touchdown_waits_for_a_new_support_pair() -> None:
    feet = np.asarray(
        [[0.3, 0.2, -0.358], [0.3, -0.2, -0.5], [-0.3, 0.2, -0.5], [-0.3, -0.2, -0.5]]
    )
    gait = ContinuousTrotGait(
        feet,
        GaitConfig(period=1.0, period_min=1.0, duty_factor=0.68),
    )
    initial = _state()
    state = RobotState(
        time_s=0.02,
        q=initial.q,
        dq=initial.dq,
        base_position=initial.base_position,
        base_velocity_body=initial.base_velocity_body,
        base_rpy=initial.base_rpy,
        base_angular_velocity_body=initial.base_angular_velocity_body,
        foot_positions_body=feet,
        foot_velocities_body=initial.foot_velocities_body,
        contacts=ContactState(np.ones(4, dtype=bool), np.full(4, 40.0)),
    )
    target = BodyTarget(
        velocity_body=np.zeros(3),
        acceleration_body=np.zeros(3),
        height=0.55,
        rpy=np.zeros(3),
        angular_acceleration_body=np.zeros(3),
        centroidal_wrench=np.zeros(6),
    )
    gait._motion_scale = 1.0
    gait._was_moving = True
    gait.phase = 0.50
    gait._height_initialized = True
    gait._stance_terrain_height[:] = 0.0
    gait._height_transition[0] = True
    gait._was_swing[0] = True
    gait._swing_target_center_height[0] = 0.192
    gait._swing_target_world_xy[0] = np.asarray([0.3, 0.2])

    class SingleSupportEstimate:
        def support_height(self) -> float:
            return 0.0

        def corrected_height(self, leg: int, nominal_height: float) -> float:
            del leg
            return nominal_height

    plan = gait.plan(
        state,
        BodyCommand(vx=0.2),
        target,
        StairTerrain(rise=0.15, run=0.30, start_x=10.0),
        SingleSupportEstimate(),
    )

    assert plan.height_transition_mask[0]
    assert plan.world_anchored_mask[0]
    assert gait.should_hold_body_progress(
        state,
        BodyCommand(vx=0.2),
        StairTerrain(rise=0.15, run=0.30, start_x=10.0),
    )


def test_contacted_transition_leg_stays_in_stance_for_the_partner_swing() -> None:
    feet = np.asarray(
        [[0.3, 0.2, -0.5], [0.3, -0.2, -0.5], [-0.3, 0.2, -0.5], [-0.3, -0.2, -0.5]]
    )
    gait = ContinuousTrotGait(
        feet,
        GaitConfig(period=1.0, period_min=1.0, duty_factor=0.68),
    )
    state = _state()
    target = BodyTarget(
        velocity_body=np.zeros(3),
        acceleration_body=np.zeros(3),
        height=0.55,
        rpy=np.zeros(3),
        angular_acceleration_body=np.zeros(3),
        centroidal_wrench=np.zeros(6),
    )
    gait._motion_scale = 1.0
    gait._was_moving = True
    gait.phase = 0.10
    gait._height_initialized = True
    gait._stance_terrain_height[:] = 0.0
    gait._height_transition[0] = True
    gait._swing_target_center_height[0] = 0.192
    gait._was_swing[0] = True

    plan = gait.plan(
        state,
        BodyCommand(vx=0.2),
        target,
        StairTerrain(rise=0.15, run=0.30, start_x=10.0),
    )

    assert not plan.swing_mask[0]
    assert plan.swing_mask[3]


def test_established_stair_support_pair_is_kept_during_body_height_transfer() -> None:
    feet = np.asarray(
        [
            [0.3, 0.2, -0.358],
            [0.3, -0.2, -0.358],
            [-0.3, 0.2, -0.508],
            [-0.3, -0.2, -0.508],
        ]
    )
    gait = ContinuousTrotGait(
        feet,
        GaitConfig(period=1.0, period_min=1.0, duty_factor=0.68),
    )
    state = replace(_state(), foot_positions_body=feet)
    target = BodyTarget(
        velocity_body=np.zeros(3),
        acceleration_body=np.zeros(3),
        height=0.70,
        rpy=np.zeros(3),
        angular_acceleration_body=np.zeros(3),
        centroidal_wrench=np.zeros(6),
    )

    class EstablishedSupport:
        def support_height(self) -> float:
            return 0.15

        def corrected_height(self, leg: int, nominal_height: float) -> float:
            del leg
            return nominal_height

    gait._motion_scale = 1.0
    gait._was_moving = True
    gait.phase = 0.10
    plan = gait.plan(
        state,
        BodyCommand(vx=0.2),
        target,
        StairTerrain(rise=0.15, run=0.30, start_x=10.0),
        EstablishedSupport(),
    )

    assert plan.swing_mask.tolist() == [False, False, False, True]
    world_z = state.base_position[2] + plan.positions_body[:, 2]
    np.testing.assert_allclose(
        world_z[:3],
        state.base_position[2] + feet[:3, 2],
        atol=1.0e-6,
    )
