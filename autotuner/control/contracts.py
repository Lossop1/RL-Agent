"""Stable data contracts shared by the model controller and backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {result.shape}")
    return result.copy()


@dataclass(frozen=True)
class BodyCommand:
    """Body-frame command [vx, vy, wz] in m/s and rad/s."""

    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.asarray([self.vx, self.vy, self.wz], dtype=np.float64)


@dataclass(frozen=True)
class ContactState:
    """Per-foot support and non-support obstacle contact evidence in N."""

    in_contact: np.ndarray
    normal_force: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float64))
    obstacle_contact: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=bool))
    obstacle_force: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float64))

    def __post_init__(self) -> None:
        object.__setattr__(self, "in_contact", np.asarray(self.in_contact, dtype=bool).copy())
        object.__setattr__(self, "normal_force", _array(self.normal_force, (4,), "normal_force"))
        object.__setattr__(
            self,
            "obstacle_contact",
            np.asarray(self.obstacle_contact, dtype=bool).copy(),
        )
        object.__setattr__(self, "obstacle_force", _array(self.obstacle_force, (4,), "obstacle_force"))
        if self.in_contact.shape != (4,):
            raise ValueError("in_contact must have shape (4,)")
        if self.obstacle_contact.shape != (4,):
            raise ValueError("obstacle_contact must have shape (4,)")
        if np.any(self.normal_force < 0.0) or np.any(self.obstacle_force < 0.0):
            raise ValueError("contact forces must be non-negative")


@dataclass(frozen=True)
class WholeBodyDynamics:
    """与 ``RobotState`` 同步的浮基动力学快照。

    广义速度、质量矩阵和 Jacobian 均使用后端原生的同一组广义速度坐标。
    Jacobian 的空间顺序固定为线速度在前、角速度在后；四个足端 Jacobian
    只包含线速度。``actuated_dof_indices`` 按控制命令的 12 关节顺序排列。
    """

    generalized_velocity: np.ndarray
    mass_matrix: np.ndarray
    bias_force: np.ndarray
    actuated_dof_indices: np.ndarray
    base_jacobian: np.ndarray
    base_bias_acceleration: np.ndarray
    foot_jacobians: np.ndarray
    foot_bias_accelerations: np.ndarray

    def __post_init__(self) -> None:
        velocity = np.asarray(self.generalized_velocity, dtype=np.float64)
        if velocity.ndim != 1 or velocity.size < 18 or not np.isfinite(velocity).all():
            raise ValueError("generalized_velocity must be a finite vector with at least 18 entries")
        nv = int(velocity.size)
        mass = _array(self.mass_matrix, (nv, nv), "mass_matrix")
        if not np.allclose(mass, mass.T, atol=1.0e-9, rtol=1.0e-9):
            raise ValueError("mass_matrix must be symmetric")
        indices = np.asarray(self.actuated_dof_indices, dtype=np.int64)
        if indices.shape != (12,) or np.unique(indices).size != 12:
            raise ValueError("actuated_dof_indices must contain twelve unique entries")
        if np.any(indices < 0) or np.any(indices >= nv):
            raise ValueError("actuated_dof_indices contains an out-of-range entry")
        object.__setattr__(self, "generalized_velocity", velocity.copy())
        object.__setattr__(self, "mass_matrix", mass)
        object.__setattr__(self, "bias_force", _array(self.bias_force, (nv,), "bias_force"))
        object.__setattr__(self, "actuated_dof_indices", indices.copy())
        object.__setattr__(self, "base_jacobian", _array(self.base_jacobian, (6, nv), "base_jacobian"))
        object.__setattr__(
            self,
            "base_bias_acceleration",
            _array(self.base_bias_acceleration, (6,), "base_bias_acceleration"),
        )
        object.__setattr__(self, "foot_jacobians", _array(self.foot_jacobians, (4, 3, nv), "foot_jacobians"))
        object.__setattr__(
            self,
            "foot_bias_accelerations",
            _array(self.foot_bias_accelerations, (4, 3), "foot_bias_accelerations"),
        )


@dataclass(frozen=True)
class RobotState:
    """Simulator-independent measured state.

    Positions and velocities use the body frame for feet and the world frame
    for the floating base. ``base_rpy`` is [roll, pitch, yaw] in radians.
    """

    time_s: float
    q: np.ndarray
    dq: np.ndarray
    base_position: np.ndarray
    base_velocity_body: np.ndarray
    base_rpy: np.ndarray
    base_angular_velocity_body: np.ndarray
    foot_positions_body: np.ndarray
    foot_velocities_body: np.ndarray
    contacts: ContactState
    dynamics: WholeBodyDynamics | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.time_s)):
            raise ValueError("time_s must be finite")
        for name, value, shape in (
            ("q", self.q, (12,)),
            ("dq", self.dq, (12,)),
            ("base_position", self.base_position, (3,)),
            ("base_velocity_body", self.base_velocity_body, (3,)),
            ("base_rpy", self.base_rpy, (3,)),
            ("base_angular_velocity_body", self.base_angular_velocity_body, (3,)),
            ("foot_positions_body", self.foot_positions_body, (4, 3)),
            ("foot_velocities_body", self.foot_velocities_body, (4, 3)),
        ):
            object.__setattr__(self, name, _array(value, shape, name))
        if self.dynamics is not None and not isinstance(self.dynamics, WholeBodyDynamics):
            raise TypeError("dynamics must be a WholeBodyDynamics snapshot")


@dataclass(frozen=True)
class BodyTarget:
    """The first-step reference returned by the finite-horizon MPC."""

    velocity_body: np.ndarray
    acceleration_body: np.ndarray
    height: float
    rpy: np.ndarray
    angular_acceleration_body: np.ndarray
    centroidal_wrench: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "velocity_body", _array(self.velocity_body, (3,), "velocity_body"))
        object.__setattr__(self, "acceleration_body", _array(self.acceleration_body, (3,), "acceleration_body"))
        object.__setattr__(self, "rpy", _array(self.rpy, (3,), "rpy"))
        object.__setattr__(
            self,
            "angular_acceleration_body",
            _array(self.angular_acceleration_body, (3,), "angular_acceleration_body"),
        )
        object.__setattr__(self, "centroidal_wrench", _array(self.centroidal_wrench, (6,), "centroidal_wrench"))
        if not np.isfinite(float(self.height)):
            raise ValueError("height must be finite")


@dataclass(frozen=True)
class FootPlan:
    """Body-frame foot targets and their contact intent."""

    positions_body: np.ndarray
    velocities_body: np.ndarray
    swing_mask: np.ndarray
    expected_contacts: np.ndarray
    terrain_heights: np.ndarray
    contact_weights: np.ndarray = field(default_factory=lambda: np.ones(4, dtype=np.float64))
    height_transition_mask: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=bool)
    )
    world_anchored_mask: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=bool)
    )
    contact_acquisition_mask: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=bool)
    )
    swing_start_world_xy: np.ndarray = field(
        default_factory=lambda: np.zeros((4, 2), dtype=np.float64)
    )
    touchdown_targets_world_xy: np.ndarray = field(
        default_factory=lambda: np.zeros((4, 2), dtype=np.float64)
    )

    def __post_init__(self) -> None:
        for name, value, shape in (
            ("positions_body", self.positions_body, (4, 3)),
            ("velocities_body", self.velocities_body, (4, 3)),
            ("terrain_heights", self.terrain_heights, (4,)),
            ("contact_weights", self.contact_weights, (4,)),
            ("swing_start_world_xy", self.swing_start_world_xy, (4, 2)),
            ("touchdown_targets_world_xy", self.touchdown_targets_world_xy, (4, 2)),
        ):
            object.__setattr__(self, name, _array(value, shape, name))
        if np.any(self.contact_weights < 0.0) or np.any(self.contact_weights > 1.0):
            raise ValueError("contact_weights must be in [0, 1]")
        for name, value in (
            ("swing_mask", self.swing_mask),
            ("expected_contacts", self.expected_contacts),
            ("height_transition_mask", self.height_transition_mask),
            ("world_anchored_mask", self.world_anchored_mask),
            ("contact_acquisition_mask", self.contact_acquisition_mask),
        ):
            array = np.asarray(value, dtype=bool).copy()
            if array.shape != (4,):
                raise ValueError(f"{name} must have shape (4,)")
            object.__setattr__(self, name, array)


@dataclass(frozen=True)
class JointCommand:
    """Position/velocity/torque command emitted by WBC and the servo layer.

    ``position`` and ``velocity`` are the end-of-interval references;
    ``acceleration`` and ``reference_duration`` define the constant-
    acceleration path from the synchronized sample to that endpoint. A
    high-rate backend follows this path instead of applying the endpoint at
    the first physics substep. ``feedforward_torque`` is the inverse-dynamics
    QP torque and ``torque`` is the total torque at the synchronized sample.
    """

    position: np.ndarray
    velocity: np.ndarray
    torque: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    saturated: np.ndarray
    task_residual: float
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float64))
    reference_duration: float = 0.02
    feedforward_torque: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float64))

    def __post_init__(self) -> None:
        for name, value in (
            ("position", self.position),
            ("velocity", self.velocity),
            ("torque", self.torque),
            ("kp", self.kp),
            ("kd", self.kd),
            ("acceleration", self.acceleration),
            ("feedforward_torque", self.feedforward_torque),
        ):
            object.__setattr__(self, name, _array(value, (12,), name))
        saturated = np.asarray(self.saturated, dtype=bool).copy()
        if saturated.shape != (12,):
            raise ValueError("saturated must have shape (12,)")
        object.__setattr__(self, "saturated", saturated)
        if not np.isfinite(float(self.task_residual)):
            raise ValueError("task_residual must be finite")
        if not np.isfinite(float(self.reference_duration)) or self.reference_duration <= 0.0:
            raise ValueError("reference_duration must be positive and finite")
