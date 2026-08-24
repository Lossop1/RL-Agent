"""Serial nominal MuJoCo evaluation with task-level acceptance metrics."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np

from autotuner.control import BodyCommand, ControlError, MpcWbcController, StairTerrain
from autotuner.control.trace import (
    control_trace_row,
    failure_trace_row,
    write_trace_rows,
)

from .backends import MujocoBackend, build_nominal_scene
from .config import build_taili_controller


@dataclass(frozen=True)
class EpisodeResult:
    scenario: str
    steps: int
    requested_steps: int
    elapsed_s: float
    passed: bool
    failure_reasons: tuple[str, ...]
    fell: bool
    command: tuple[float, float, float]
    final_position_world: tuple[float, float, float]
    displacement_world: tuple[float, float, float]
    mean_velocity_body: tuple[float, float, float]
    velocity_tracking_rmse: float
    yaw_rate_tracking_rmse: float
    progress_ratio: float | None
    cross_track_error: float | None
    expected_terrain_height_change: float | None
    terrain_height_progress_ratio: float | None
    yaw_change_rad: float
    max_tilt_rad: float
    tilt_rms_rad: float
    base_height_min: float
    base_height_max: float
    base_height_error_rms: float
    torque_saturation_fraction: float
    mean_task_residual: float
    max_dynamics_residual: float
    max_contact_acceleration_residual: float
    max_contact_force_norm: float
    mean_solver_iterations: float
    supportless_fraction: float
    contacts_seen: int
    runtime_failure: dict[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "scenario": self.scenario,
            "steps": self.steps,
            "requested_steps": self.requested_steps,
            "elapsed_s": self.elapsed_s,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
            "fell": self.fell,
            "command": list(self.command),
            "final_position_world": list(self.final_position_world),
            "displacement_world": list(self.displacement_world),
            "mean_velocity_body": list(self.mean_velocity_body),
            "velocity_tracking_rmse": self.velocity_tracking_rmse,
            "yaw_rate_tracking_rmse": self.yaw_rate_tracking_rmse,
            "progress_ratio": self.progress_ratio,
            "cross_track_error": self.cross_track_error,
            "expected_terrain_height_change": self.expected_terrain_height_change,
            "terrain_height_progress_ratio": self.terrain_height_progress_ratio,
            "yaw_change_rad": self.yaw_change_rad,
            "max_tilt_rad": self.max_tilt_rad,
            "tilt_rms_rad": self.tilt_rms_rad,
            "base_height_min": self.base_height_min,
            "base_height_max": self.base_height_max,
            "base_height_error_rms": self.base_height_error_rms,
            "torque_saturation_fraction": self.torque_saturation_fraction,
            "mean_task_residual": self.mean_task_residual,
            "max_dynamics_residual": self.max_dynamics_residual,
            "max_contact_acceleration_residual": self.max_contact_acceleration_residual,
            "max_contact_force_norm": self.max_contact_force_norm,
            "mean_solver_iterations": self.mean_solver_iterations,
            "supportless_fraction": self.supportless_fraction,
            "contacts_seen": self.contacts_seen,
            "runtime_failure": self.runtime_failure,
        }


def _acceptance_failures(
    metrics: dict[str, float | bool | None],
    command: BodyCommand,
    thresholds: dict[str, float],
) -> tuple[str, ...]:
    failures: list[str] = []
    if bool(metrics["fell"]):
        failures.append("fell")
    if float(metrics["max_tilt_rad"]) > float(thresholds["max_tilt_rad"]):
        failures.append("tilt")
    if float(metrics["base_height_error_rms"]) > float(thresholds["max_height_error_rms"]):
        failures.append("height_tracking")
    if float(metrics["torque_saturation_fraction"]) > float(thresholds["max_torque_saturation_fraction"]):
        failures.append("torque_saturation")
    if float(metrics["max_dynamics_residual"]) > float(thresholds["max_dynamics_residual"]):
        failures.append("dynamics_residual")
    if float(metrics["max_contact_acceleration_residual"]) > float(
        thresholds["max_contact_acceleration_residual"]
    ):
        failures.append("contact_acceleration_residual")

    linear_speed = float(np.hypot(command.vx, command.vy))
    if linear_speed > 1.0e-6:
        if float(metrics["velocity_tracking_rmse"]) > float(thresholds["max_velocity_tracking_rmse"]):
            failures.append("linear_velocity_tracking")
        progress = metrics["progress_ratio"]
        if progress is None or float(progress) < float(thresholds["min_progress_ratio"]):
            failures.append("insufficient_progress")
        cross_track = metrics["cross_track_error"]
        if cross_track is None or float(cross_track) > float(thresholds["max_cross_track_error"]):
            failures.append("cross_track")
    elif abs(command.wz) > 1.0e-6:
        if float(metrics["yaw_rate_tracking_rmse"]) > float(thresholds["max_yaw_rate_tracking_rmse"]):
            failures.append("yaw_rate_tracking")
    else:
        if float(metrics["planar_displacement"]) > float(thresholds["max_standing_drift"]):
            failures.append("standing_drift")
        if float(metrics["mean_planar_speed"]) > float(thresholds["max_standing_mean_speed"]):
            failures.append("standing_motion")
    expected_height_change = metrics.get("expected_terrain_height_change")
    if expected_height_change is not None:
        if abs(float(expected_height_change)) < float(
            thresholds["min_stair_expected_height_change"]
        ):
            failures.append("insufficient_stair_exposure")
        else:
            terrain_progress = metrics.get("terrain_height_progress_ratio")
            if terrain_progress is None or float(terrain_progress) < float(
                thresholds["min_terrain_height_progress_ratio"]
            ):
                failures.append("insufficient_terrain_height_progress")
    return tuple(failures)


def run_mujoco_episode(
    scenario: str,
    *,
    duration_s: float | None = None,
    config_path: str | Path | None = None,
    trace_path: str | Path | None = None,
    trace_dynamics: bool = False,
    progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> EpisodeResult:
    bundle = build_taili_controller(scenario, config_path)
    if duration_s is None:
        durations = bundle.config["evaluation"].get("duration_s", {})
        duration_s = float(durations.get(scenario, durations.get("default", 3.0)))
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    scene = build_nominal_scene(bundle, scenario)
    backend = MujocoBackend(
        scene.model,
        scene.data,
        bundle.profile,
        contact_force_threshold=float(bundle.config["contact"]["normal_force_threshold"]),
        support_normal_min_z=float(bundle.config["contact"]["support_normal_min_z"]),
    )
    controller: MpcWbcController = bundle.controller
    command = BodyCommand(*bundle.config["scenarios"][scenario]["command"])
    physics_steps = max(1, int(round(scene.policy_dt / backend.physics_dt)))
    policy_steps = max(1, int(np.ceil(duration_s / scene.policy_dt)))
    initial_state = backend.read_state()
    initial_position = initial_state.base_position.copy()
    initial_yaw = float(initial_state.base_rpy[2])
    heights: list[float] = []
    height_errors: list[float] = []
    tilts: list[float] = []
    body_velocities: list[np.ndarray] = []
    yaw_rates: list[float] = []
    residuals: list[float] = []
    dynamics_residuals: list[float] = []
    contact_acceleration_residuals: list[float] = []
    contact_force_norms: list[float] = []
    solver_iterations: list[int] = []
    supportless_samples = 0
    saturated = 0
    saturation_samples = 0
    contacts_seen = 0
    trace_rows: list[dict[str, object]] = []
    runtime_failure: dict[str, object] | None = None
    completed_steps = 0
    cancelled_early = False
    for policy_index in range(policy_steps):
        if cancelled is not None and cancelled():
            cancelled_early = True
            break
        state = backend.read_state()
        try:
            output = controller.step(state, command)
        except ControlError as exc:
            runtime_failure = exc.as_dict()
            if trace_path is not None:
                trace_rows.append(failure_trace_row(state, command, exc))
            break
        backend.write_command(output.joints)
        desired_height = output.body.height
        heights.append(float(state.base_position[2]))
        height_errors.append(float(state.base_position[2] - desired_height))
        tilts.append(float(np.linalg.norm(state.base_rpy[:2])))
        body_velocities.append(state.base_velocity_body.copy())
        yaw_rates.append(float(state.base_angular_velocity_body[2]))
        residuals.append(float(output.joints.task_residual))
        dynamics_residuals.append(float(output.wbc.dynamics_residual))
        contact_acceleration_residuals.append(float(output.wbc.contact_acceleration_residual))
        contact_force_norms.append(
            float(np.max(np.linalg.norm(output.wbc.contact_forces_world, axis=1)))
        )
        solver_iterations.append(int(output.wbc.solver_iterations))
        supportless_samples += int(not np.any(output.wbc.active_contacts))
        saturated += int(np.count_nonzero(output.joints.saturated))
        saturation_samples += 12
        contacts_seen += int(np.count_nonzero(state.contacts.in_contact))
        if trace_path is not None:
            trace_rows.append(
                control_trace_row(
                    state,
                    command,
                    output,
                    include_dynamics=trace_dynamics,
                )
            )
        for _ in range(physics_steps):
            backend.step_physics()
        completed_steps += 1
        if progress is not None:
            progress((policy_index + 1) / policy_steps)

    final_state = backend.read_state()
    if not heights:
        heights.append(float(initial_state.base_position[2]))
        height_errors.append(
            float(initial_state.base_position[2] - bundle.profile.nominal_base_height)
        )
        tilts.append(float(np.linalg.norm(initial_state.base_rpy[:2])))
        body_velocities.append(initial_state.base_velocity_body.copy())
        yaw_rates.append(float(initial_state.base_angular_velocity_body[2]))
    displacement = final_state.base_position - initial_position
    velocities = np.asarray(body_velocities, dtype=np.float64)
    target_linear = np.asarray([command.vx, command.vy], dtype=np.float64)
    velocity_rmse = float(np.sqrt(np.mean(np.sum((velocities[:, :2] - target_linear) ** 2, axis=1))))
    yaw_rate_rmse = float(np.sqrt(np.mean((np.asarray(yaw_rates) - command.wz) ** 2)))
    command_norm = float(np.linalg.norm(target_linear))
    progress_ratio: float | None = None
    cross_track_error: float | None = None
    expected_terrain_height_change: float | None = None
    terrain_height_progress_ratio: float | None = None
    if command_norm > 1.0e-9:
        direction_body = target_linear / command_norm
        c, s = np.cos(initial_yaw), np.sin(initial_yaw)
        direction_world = np.asarray(
            [c * direction_body[0] - s * direction_body[1],
             s * direction_body[0] + c * direction_body[1]],
            dtype=np.float64,
        )
        planar_displacement = displacement[:2]
        progress_ratio = float(np.dot(planar_displacement, direction_world) / (command_norm * duration_s))
        lateral_direction = np.asarray([-direction_world[1], direction_world[0]], dtype=np.float64)
        cross_track_error = float(abs(np.dot(planar_displacement, lateral_direction)))
        terrain = bundle.controller.terrain
        if isinstance(terrain, StairTerrain):
            expected_endpoint = initial_position[:2] + direction_world * command_norm * duration_s
            initial_terrain_height = terrain.height_at(
                float(initial_position[0]), float(initial_position[1])
            )
            expected_terrain_height = terrain.height_at(
                float(expected_endpoint[0]), float(expected_endpoint[1])
            )
            expected_terrain_height_change = float(
                expected_terrain_height - initial_terrain_height
            )
            if abs(expected_terrain_height_change) > 1.0e-9:
                terrain_height_progress_ratio = float(
                    displacement[2] / expected_terrain_height_change
                )
    tilt_array = np.asarray(tilts, dtype=np.float64)
    height_error_array = np.asarray(height_errors, dtype=np.float64)
    fell = bool(min(heights) < 0.25 or max(tilts) > np.deg2rad(55.0))
    metrics: dict[str, float | bool | None] = {
        "fell": fell,
        "max_tilt_rad": float(max(tilts)),
        "base_height_error_rms": float(np.sqrt(np.mean(height_error_array ** 2))),
        "torque_saturation_fraction": float(saturated / max(saturation_samples, 1)),
        "max_dynamics_residual": float(max(dynamics_residuals, default=0.0)),
        "max_contact_acceleration_residual": float(
            max(contact_acceleration_residuals, default=0.0)
        ),
        "velocity_tracking_rmse": velocity_rmse,
        "yaw_rate_tracking_rmse": yaw_rate_rmse,
        "progress_ratio": progress_ratio,
        "cross_track_error": cross_track_error,
        "planar_displacement": float(np.linalg.norm(displacement[:2])),
        "mean_planar_speed": float(np.mean(np.linalg.norm(velocities[:, :2], axis=1))),
        "expected_terrain_height_change": expected_terrain_height_change,
        "terrain_height_progress_ratio": terrain_height_progress_ratio,
    }
    thresholds = {
        key: float(value)
        for key, value in bundle.config["evaluation"]["thresholds"].items()
    }
    failures = _acceptance_failures(metrics, command, thresholds)
    if runtime_failure is not None:
        failures = tuple(
            [f"controller_failure:{runtime_failure['code']}", *failures]
        )
    if trace_path is not None:
        write_trace_rows(trace_path, trace_rows)
    if cancelled_early:
        raise InterruptedError("traditional-control episode cancelled")
    return EpisodeResult(
        scenario=scenario,
        steps=completed_steps,
        requested_steps=policy_steps,
        elapsed_s=float(scene.data.time),
        passed=not failures,
        failure_reasons=failures,
        fell=fell,
        command=(command.vx, command.vy, command.wz),
        final_position_world=tuple(float(value) for value in final_state.base_position),
        displacement_world=tuple(float(value) for value in displacement),
        mean_velocity_body=tuple(float(value) for value in np.mean(velocities, axis=0)),
        velocity_tracking_rmse=velocity_rmse,
        yaw_rate_tracking_rmse=yaw_rate_rmse,
        progress_ratio=progress_ratio,
        cross_track_error=cross_track_error,
        expected_terrain_height_change=expected_terrain_height_change,
        terrain_height_progress_ratio=terrain_height_progress_ratio,
        yaw_change_rad=float(np.unwrap([initial_yaw, float(final_state.base_rpy[2])])[1] - initial_yaw),
        max_tilt_rad=float(max(tilts)),
        tilt_rms_rad=float(np.sqrt(np.mean(tilt_array ** 2))),
        base_height_min=float(min(heights)),
        base_height_max=float(max(heights)),
        base_height_error_rms=float(metrics["base_height_error_rms"]),
        torque_saturation_fraction=float(metrics["torque_saturation_fraction"]),
        mean_task_residual=float(np.mean(residuals)) if residuals else 0.0,
        max_dynamics_residual=float(metrics["max_dynamics_residual"]),
        max_contact_acceleration_residual=float(
            metrics["max_contact_acceleration_residual"]
        ),
        max_contact_force_norm=float(max(contact_force_norms, default=0.0)),
        mean_solver_iterations=float(np.mean(solver_iterations)) if solver_iterations else 0.0,
        supportless_fraction=float(supportless_samples / max(policy_steps, 1)),
        contacts_seen=contacts_seen,
        runtime_failure=runtime_failure,
    )


def write_episode_result(path: str | Path, result: EpisodeResult) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8")
    return target
