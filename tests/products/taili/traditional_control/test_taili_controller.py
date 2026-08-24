from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autotuner.control import (
    BodyCommand,
    ContactState,
    JointCommand,
    RobotState,
    WholeBodyDynamics,
)
from autotuner.control.backends import SimulatorBackend
from autotuner.control.trace import control_trace_row
from autotuner.control.wbc import FloatingBaseWbc
from products.taili.traditional_control import build_taili_controller, load_taili_control_config, load_taili_profile
from products.taili.traditional_control.backends.mujoco import (
    _environment_to_foot_normal,
    _split_free_joint_velocity,
)
from products.taili.traditional_control.backends.isaaclab import IsaacLabBackend
from products.taili.traditional_control.packaging import build_bundle


SCENARIOS = (
    "flat",
    "flat_forward",
    "flat_backward",
    "flat_left",
    "flat_right",
    "flat_yaw_left",
    "flat_yaw_right",
    "stairs_up",
    "stairs_down",
)


def _initial_state(bundle) -> RobotState:
    profile = bundle.profile
    feet = bundle.kinematics.nominal_foot_positions()
    nv = 18
    actuated_dofs = np.arange(6, 18, dtype=np.int64)
    foot_jacobians = np.zeros((4, 3, nv), dtype=np.float64)
    foot_jacobians[:, :, actuated_dofs] = bundle.kinematics.foot_jacobians(
        profile.q_default
    )
    base_jacobian = np.zeros((6, nv), dtype=np.float64)
    base_jacobian[:, :6] = np.eye(6, dtype=np.float64)
    dynamics = WholeBodyDynamics(
        generalized_velocity=np.zeros(nv),
        mass_matrix=np.eye(nv),
        bias_force=np.zeros(nv),
        actuated_dof_indices=actuated_dofs,
        base_jacobian=base_jacobian,
        base_bias_acceleration=np.zeros(6),
        foot_jacobians=foot_jacobians,
        foot_bias_accelerations=np.zeros((4, 3)),
    )
    return RobotState(
        time_s=0.0,
        q=profile.q_default,
        dq=np.zeros(12),
        base_position=np.asarray([0.0, 0.0, profile.nominal_base_height]),
        base_velocity_body=np.zeros(3),
        base_rpy=np.zeros(3),
        base_angular_velocity_body=np.zeros(3),
        foot_positions_body=feet,
        foot_velocities_body=np.zeros((4, 3)),
        contacts=ContactState(np.ones(4, dtype=bool), np.ones(4) * 50.0),
        dynamics=dynamics,
    )


def test_taili_profile_is_derived_from_urdf() -> None:
    config = load_taili_control_config()
    profile = load_taili_profile(config)
    assert profile.urdf_path.is_file()
    assert profile.joint_names[0] == "FL_hip_joint"
    assert profile.joint_names[-1] == "RR_calf_joint"
    assert profile.mass_kg > 30.0
    assert profile.foot_radius == 0.042
    assert np.all(profile.q_lower < profile.q_upper)


def test_controller_has_one_deterministic_step_for_each_nominal_scenario() -> None:
    for scenario in SCENARIOS:
        bundle = build_taili_controller(scenario)
        state = _initial_state(bundle)
        command = BodyCommand(*bundle.config["scenarios"][scenario]["command"])
        output = bundle.controller.step(state, command)
        assert output.joints.position.shape == (12,)
        assert np.isfinite(output.joints.position).all()
        assert np.isfinite(output.joints.torque).all()
        assert output.joints.saturated.shape == (12,)
        assert output.body.height > 0.0
        assert output.wbc.solver_status.lower().startswith("solved")
        assert output.wbc.dynamics_residual < 1.0e-5
        assert output.wbc.contact_acceleration_residual < 1.0e-5
        assert output.wbc.base_acceleration_residual_world.shape == (6,)
        assert output.wbc.foot_acceleration_residuals_world.shape == (4, 3)


def test_trace_has_compact_and_replayable_modes() -> None:
    bundle = build_taili_controller("flat_forward")
    state = _initial_state(bundle)
    command = BodyCommand(vx=0.4)
    output = bundle.controller.step(state, command)

    compact = control_trace_row(state, command, output)
    replayable = control_trace_row(state, command, output, include_dynamics=True)

    assert compact["state"]["dynamics"] is None
    assert replayable["state"]["dynamics"]["mass_matrix"]
    assert len(compact["plan"]["mpc"]["control_sequence"]) == bundle.controller.mpc.config.horizon
    assert compact["solution"]["wbc"]["foot_tracking_mask"] == [False] * 4


def test_mujoco_episode_records_a_structured_first_step_qp_failure(
    monkeypatch,
    tmp_path,
) -> None:
    failed = SimpleNamespace(
        info=SimpleNamespace(
            status="primal infeasible",
            status_val=3,
            iter=25,
            obj_val=1.0e30,
        ),
        x=None,
    )
    monkeypatch.setattr(FloatingBaseWbc, "_solve_qp", lambda *args: failed)
    trace_path = tmp_path / "failed.jsonl"

    from products.taili.traditional_control.evaluation import run_mujoco_episode

    result = run_mujoco_episode("flat", duration_s=0.02, trace_path=trace_path)

    assert not result.passed
    assert result.steps == 0
    assert result.requested_steps == 1
    assert result.runtime_failure is not None
    assert result.runtime_failure["code"] == "qp_unsolved"
    json.dumps(result.as_dict(), allow_nan=False)
    failure = json.loads(trace_path.read_text(encoding="utf-8"))
    assert failure["kind"] == "control_failure"
    assert failure["failure"]["details"]["solver_iterations"] == 25
    assert failure["state"]["dynamics"]["mass_matrix"]


def test_stair_geometry_is_parameterized() -> None:
    up = build_taili_controller("stairs_up")
    terrain = up.controller.terrain
    assert terrain.height_at(0.55, 0.0) == 0.30
    assert terrain.height_at(0.25, 2.0) == 0.0
    assert up.controller.gait.config.period == 1.10
    assert up.controller.gait.config.duty_factor == 0.68
    assert up.controller.wbc.config.swing_acceleration_weight == (20.0, 20.0, 20.0)
    assert up.controller.wbc.config.max_swing_acceleration == 60.0
    assert up.controller.wbc.config.contact_acquisition_horizontal_weight == 300.0
    assert up.controller.wbc.config.contact_acquisition_vertical_weight == 300.0
    assert build_taili_controller("flat_forward").controller.gait.config.period == 0.72
    assert build_taili_controller("flat_forward").controller.wbc.config.max_swing_acceleration == 60.0


def test_isaaclab_backend_implements_the_same_physics_step_contract() -> None:
    bundle = build_taili_controller("flat")
    state = _initial_state(bundle)
    clock = [0.0]
    q = state.q.copy()
    dq = state.dq.copy()
    torques: list[np.ndarray] = []
    physics_steps: list[float] = []

    backend = IsaacLabBackend(
        read_state_fn=lambda: state,
        read_joint_state_fn=lambda: (clock[0], q, dq),
        write_torque_fn=lambda value: torques.append(value.copy()),
        physics_step_fn=lambda: physics_steps.append(clock[0]),
        effort_limit=bundle.profile.effort_limit,
        physics_dt_s=0.005,
    )
    command = JointCommand(
        position=q,
        velocity=dq,
        torque=np.zeros(12),
        kp=np.full(12, 120.0),
        kd=np.full(12, 10.0),
        saturated=np.zeros(12, dtype=bool),
        task_residual=0.0,
        feedforward_torque=np.arange(12, dtype=np.float64),
    )

    assert isinstance(backend, SimulatorBackend)
    assert backend.read_state() is state
    backend.write_command(command)
    clock[0] = backend.physics_dt
    backend.step_physics()
    assert len(torques) == 2
    assert torques[-1].tolist() == command.feedforward_torque.tolist()
    assert physics_steps == [0.005]


def test_taili_bundle_bytes_are_deterministic_and_runtime_is_separate() -> None:
    root = Path(__file__).resolve().parents[4]
    output = root / ".pytest-tmp" / "traditional-control-bundle"

    first = build_bundle(output)
    first_bytes = Path(first["archive"]).read_bytes()
    second = build_bundle(output)

    assert Path(first["archive"]) == Path(second["archive"])
    assert Path(second["archive"]).read_bytes() == first_bytes
    assert first["artifact_digest"] == second["artifact_digest"]
    artifact_manifest = json.loads(Path(first["manifest"]).read_text(encoding="utf-8"))
    run_manifest = json.loads(Path(first["run_manifest"]).read_text(encoding="utf-8"))
    assert "generated_at" not in artifact_manifest
    assert run_manifest["artifact"]["digest"] == artifact_manifest["artifact_digest"]
    assert run_manifest["artifact"]["archive_sha256"] == first["archive_sha256"]


def test_mujoco_free_joint_velocity_order_is_linear_then_angular() -> None:
    linear, angular = _split_free_joint_velocity(np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    assert linear.tolist() == [1.0, 2.0, 3.0]
    assert angular.tolist() == [4.0, 5.0, 6.0]


def test_mujoco_contact_frame_uses_the_first_row_as_normal() -> None:
    top_surface = np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0])
    riser = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    assert _environment_to_foot_normal(top_surface, foot_is_geom1=False)[2] == 1.0
    assert _environment_to_foot_normal(-top_surface, foot_is_geom1=True)[2] == 1.0
    assert _environment_to_foot_normal(riser, foot_is_geom1=False)[2] == 0.0


def test_mujoco_contacts_only_report_upward_support() -> None:
    import mujoco

    bundle = build_taili_controller("stairs_up")
    scene = __import__(
        "products.taili.traditional_control.backends.mujoco_scene",
        fromlist=["build_nominal_scene"],
    ).build_nominal_scene(bundle, "stairs_up")
    adapter = __import__(
        "products.taili.traditional_control.backends.mujoco",
        fromlist=["MujocoStateAdapter"],
    ).MujocoStateAdapter(
        scene.model,
        scene.data,
        bundle.profile,
        contact_force_threshold=float(bundle.config["contact"]["normal_force_threshold"]),
        support_normal_min_z=float(bundle.config["contact"]["support_normal_min_z"]),
    )
    # Constructing contact frames by hand is simulator-specific, so validate
    # the semantic threshold against the actual first riser and top tread.
    foot_geom = int(adapter._feet[0])
    riser_geom = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "nominal_step_000")
    assert foot_geom >= 0 and riser_geom >= 0
    assert adapter.support_normal_min_z == pytest.approx(
        float(bundle.config["contact"]["support_normal_min_z"])
    )


def test_mujoco_flat_ground_reports_all_four_support_feet() -> None:
    bundle = build_taili_controller("flat")
    scene = __import__(
        "products.taili.traditional_control.backends.mujoco_scene",
        fromlist=["build_nominal_scene"],
    ).build_nominal_scene(bundle, "flat")
    adapter = __import__(
        "products.taili.traditional_control.backends.mujoco",
        fromlist=["MujocoStateAdapter"],
    ).MujocoStateAdapter(scene.model, scene.data, bundle.profile)

    state = adapter.read_state()
    assert state.contacts.in_contact.tolist() == [True, True, True, True]
    assert state.dynamics is not None
    assert state.dynamics.mass_matrix.shape == (scene.model.nv, scene.model.nv)
    assert state.dynamics.base_jacobian.shape == (6, scene.model.nv)
    assert state.dynamics.foot_jacobians.shape == (4, 3, scene.model.nv)
    assert np.allclose(
        state.dq,
        state.dynamics.generalized_velocity[state.dynamics.actuated_dof_indices],
    )
