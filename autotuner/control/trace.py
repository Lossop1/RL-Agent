"""与仿真器无关的逐控制周期诊断记录。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .contracts import BodyCommand, RobotState
from .controller import ControlStep
from .errors import ControlError


TRACE_SCHEMA_VERSION = "traditional_control_trace_v1"


def _values(array: np.ndarray) -> list[object]:
    return np.asarray(array).tolist()


def _state_row(state: RobotState, *, include_dynamics: bool) -> dict[str, object]:
    dynamics = state.dynamics
    dynamics_row: dict[str, object] | None = None
    if include_dynamics and dynamics is not None:
        dynamics_row = {
            "generalized_velocity": _values(dynamics.generalized_velocity),
            "mass_matrix": _values(dynamics.mass_matrix),
            "bias_force": _values(dynamics.bias_force),
            "actuated_dof_indices": _values(dynamics.actuated_dof_indices),
            "base_jacobian": _values(dynamics.base_jacobian),
            "base_bias_acceleration": _values(dynamics.base_bias_acceleration),
            "foot_jacobians": _values(dynamics.foot_jacobians),
            "foot_bias_accelerations": _values(dynamics.foot_bias_accelerations),
        }
    return {
        "q": _values(state.q),
        "dq": _values(state.dq),
        "base_position_world": _values(state.base_position),
        "base_velocity_body": _values(state.base_velocity_body),
        "base_rpy": _values(state.base_rpy),
        "base_angular_velocity_body": _values(state.base_angular_velocity_body),
        "foot_positions_body": _values(state.foot_positions_body),
        "foot_velocities_body": _values(state.foot_velocities_body),
        "contacts": {
            "in_contact": _values(state.contacts.in_contact),
            "normal_force": _values(state.contacts.normal_force),
            "obstacle_contact": _values(state.contacts.obstacle_contact),
            "obstacle_force": _values(state.contacts.obstacle_force),
        },
        "dynamics": dynamics_row,
    }


def control_trace_row(
    state: RobotState,
    command: BodyCommand,
    step: ControlStep,
    *,
    include_dynamics: bool = False,
) -> dict[str, object]:
    """Serialize one controller input, plan and task-level result.

    The compact default is suitable for long evaluations. Full synchronized
    dynamics can be requested for exact offline WBC replay.
    """

    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "kind": "control_step",
        "time_s": float(state.time_s),
        "command_body": [float(command.vx), float(command.vy), float(command.wz)],
        "state": _state_row(state, include_dynamics=include_dynamics),
        "plan": {
            "phase": float(step.phase),
            "support_height": (
                None if step.support_height is None else float(step.support_height)
            ),
            "support_pitch": float(step.support_pitch),
            "body": {
                "velocity_body": _values(step.body.velocity_body),
                "acceleration_body": _values(step.body.acceleration_body),
                "height": float(step.body.height),
                "rpy": _values(step.body.rpy),
                "angular_acceleration_body": _values(
                    step.body.angular_acceleration_body
                ),
                "centroidal_wrench": _values(step.body.centroidal_wrench),
            },
            "mpc": {
                "predicted_states": _values(step.mpc.predicted_states),
                "control_sequence": _values(step.mpc.control_sequence),
            },
            "feet": {
                "positions_body": _values(step.feet.positions_body),
                "velocities_body": _values(step.feet.velocities_body),
                "swing_mask": _values(step.feet.swing_mask),
                "expected_contacts": _values(step.feet.expected_contacts),
                "terrain_heights": _values(step.feet.terrain_heights),
                "contact_weights": _values(step.feet.contact_weights),
                "height_transition_mask": _values(
                    step.feet.height_transition_mask
                ),
                "world_anchored_mask": _values(
                    step.feet.world_anchored_mask
                ),
                "contact_acquisition_mask": _values(
                    step.feet.contact_acquisition_mask
                ),
                "swing_start_world_xy": _values(step.feet.swing_start_world_xy),
                "touchdown_targets_world_xy": _values(
                    step.feet.touchdown_targets_world_xy
                ),
            },
        },
        "solution": {
            "joint_position": _values(step.joints.position),
            "joint_velocity": _values(step.joints.velocity),
            "joint_acceleration": _values(step.joints.acceleration),
            "joint_torque": _values(step.joints.torque),
            "torque_saturated": _values(step.joints.saturated),
            "task_residual": float(step.joints.task_residual),
            "wbc": step.wbc.as_dict(),
        },
    }


def failure_trace_row(
    state: RobotState,
    command: BodyCommand,
    failure: ControlError,
) -> dict[str, object]:
    """Record the state and command at a structured controller failure."""

    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "kind": "control_failure",
        "time_s": float(state.time_s),
        "command_body": [float(command.vx), float(command.vy), float(command.wz)],
        # A failure is rare and must remain exactly replayable, so it always
        # retains the synchronized dynamics snapshot.
        "state": _state_row(state, include_dynamics=True),
        "failure": failure.as_dict(),
    }


def write_trace_rows(
    path: str | Path,
    rows: Iterable[Mapping[str, object]],
) -> Path:
    """Atomically write deterministic JSONL inside a caller-owned run path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            handle.write("\n")
    temporary.replace(target)
    return target
