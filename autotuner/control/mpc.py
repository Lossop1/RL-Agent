"""Finite-horizon centroidal motion controller for nominal locomotion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import BodyCommand, BodyTarget, RobotState
from .terrain import TerrainModel


def _rpy_rotation(rpy: np.ndarray) -> np.ndarray:
    """Return the body-to-world rotation for XYZ roll/pitch/yaw angles."""

    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class CentroidalMpcConfig:
    horizon: int = 12
    dt: float = 0.02
    mass_kg: float = 38.98
    inertia_kg_m2: tuple[float, float, float] = (0.1691, 0.5070, 0.5981)
    # Channels are world x/y/z and roll/pitch/yaw. Each weight applies to
    # both position and rate in the lifted double-integrator model.
    state_weight: tuple[float, ...] = (12.0, 12.0, 24.0, 16.0, 16.0, 8.0)
    terminal_weight: tuple[float, ...] = (24.0, 24.0, 40.0, 28.0, 28.0, 16.0)
    input_weight: tuple[float, ...] = (0.08, 0.08, 0.8, 0.3, 0.3, 0.06)
    max_acceleration: tuple[float, ...] = (5.0, 5.0, 8.0, 8.0, 8.0, 6.0)
    max_position_error: float = 0.15
    max_yaw_error: float = 0.35
    gravity: float = 9.81

    def __post_init__(self) -> None:
        if self.horizon < 2 or self.dt <= 0.0 or self.mass_kg <= 0.0:
            raise ValueError("invalid MPC horizon, dt, or mass")
        for name, value in (
            ("state_weight", self.state_weight),
            ("terminal_weight", self.terminal_weight),
            ("input_weight", self.input_weight),
            ("max_acceleration", self.max_acceleration),
        ):
            if len(value) != 6 or any(float(x) <= 0.0 for x in value):
                raise ValueError(f"{name} must contain six positive values")
        if self.max_position_error <= 0.0 or self.max_yaw_error <= 0.0:
            raise ValueError("reference anti-windup limits must be positive")


@dataclass(frozen=True)
class MpcOutput:
    target: BodyTarget
    predicted_states: np.ndarray
    control_sequence: np.ndarray


class CentroidalMpc:
    """Receding-horizon body trajectory controller.

    The optimization state contains six actual generalized positions
    ``[x, y, z, roll, pitch, yaw]`` and their rates. This matters: treating a
    commanded velocity itself as the position of a double integrator makes
    the optimized input a jerk while downstream code interprets it as force.
    Contact feasibility is enforced by the WBC force allocator, which owns
    the measured and planned contact set.
    """

    def __init__(self, config: CentroidalMpcConfig) -> None:
        self.config = config
        self._a, self._b = self._double_integrator(config.dt)
        self._sx, self._su = self._lifted_matrices(config.horizon)
        self._last_time: float | None = None
        self._target_position_xy: np.ndarray | None = None
        self._target_yaw: float | None = None

    def reset(self) -> None:
        self._last_time = None
        self._target_position_xy = None
        self._target_yaw = None

    @staticmethod
    def _double_integrator(dt: float) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray([[1.0, dt], [0.0, 1.0]], dtype=np.float64),
            np.asarray([[0.5 * dt * dt], [dt]], dtype=np.float64),
        )

    def _lifted_matrices(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        state_dim = 12
        input_dim = 6
        sx = np.zeros((state_dim * horizon, state_dim), dtype=np.float64)
        su = np.zeros((state_dim * horizon, input_dim * horizon), dtype=np.float64)
        block_a = np.kron(np.eye(6), self._a)
        block_b = np.kron(np.eye(6), self._b)
        for row in range(horizon):
            sx[row * state_dim : (row + 1) * state_dim] = np.linalg.matrix_power(block_a, row + 1)
            for col in range(row + 1):
                su[
                    row * state_dim : (row + 1) * state_dim,
                    col * input_dim : (col + 1) * input_dim,
                ] = np.linalg.matrix_power(block_a, row - col) @ block_b
        return sx, su

    def _solve(self, state: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        horizon = cfg.horizon
        state_dim = 12
        input_dim = 6
        if reference.shape != (horizon, state_dim):
            raise ValueError(f"reference must have shape {(horizon, state_dim)}")
        q = np.asarray(cfg.state_weight, dtype=np.float64)
        q_terminal = np.asarray(cfg.terminal_weight, dtype=np.float64)
        r = np.asarray(cfg.input_weight, dtype=np.float64)
        qbar = np.zeros((state_dim * horizon, state_dim * horizon), dtype=np.float64)
        rbar = np.zeros((input_dim * horizon, input_dim * horizon), dtype=np.float64)
        for index in range(horizon):
            weights = q_terminal if index == horizon - 1 else q
            qbar[
                index * state_dim : (index + 1) * state_dim,
                index * state_dim : (index + 1) * state_dim,
            ] = np.diag(np.repeat(weights, 2))
            rbar[
                index * input_dim : (index + 1) * input_dim,
                index * input_dim : (index + 1) * input_dim,
            ] = np.diag(r)
        hessian = self._su.T @ qbar @ self._su + rbar + np.eye(input_dim * horizon) * 1.0e-8
        gradient = self._su.T @ qbar @ (self._sx @ state - reference.reshape(-1))
        try:
            sequence = -np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            sequence = -np.linalg.pinv(hessian) @ gradient
        bounds = np.asarray(cfg.max_acceleration, dtype=np.float64)
        sequence = np.clip(sequence.reshape(horizon, input_dim), -bounds, bounds)
        predicted = (self._sx @ state + self._su @ sequence.reshape(-1)).reshape(horizon, state_dim)
        return predicted, sequence

    def step(
        self,
        state: RobotState,
        command: BodyCommand,
        terrain: TerrainModel,
        nominal_height: float,
        support_height: float | None = None,
        support_pitch: float = 0.0,
        hold_position: bool = False,
    ) -> MpcOutput:
        cfg = self.config
        rotation = _rpy_rotation(state.base_rpy)
        linear_velocity_world = rotation @ state.base_velocity_body
        angular_velocity_world = rotation @ state.base_angular_velocity_body
        command_velocity_world = rotation @ np.asarray([command.vx, command.vy, 0.0], dtype=np.float64)
        if self._last_time is None or self._target_position_xy is None or self._target_yaw is None:
            self._last_time = float(state.time_s)
            self._target_position_xy = state.base_position[:2].copy()
            self._target_yaw = float(state.base_rpy[2])
        dt = float(np.clip(float(state.time_s) - self._last_time, 0.0, 2.0 * cfg.dt))
        self._last_time = float(state.time_s)
        if hold_position:
            # A support transfer is a physical wait, not a zero-velocity
            # request that should keep chasing an already accumulated path.
            # Clear the stale position lead so the WBC does not inject a
            # delayed body shove while the new foothold is being established.
            self._target_position_xy = state.base_position[:2].copy()
        else:
            self._target_position_xy += command_velocity_world[:2] * dt
        self._target_yaw += float(command.wz) * dt
        position_error = self._target_position_xy - state.base_position[:2]
        position_error_norm = float(np.linalg.norm(position_error))
        if position_error_norm > cfg.max_position_error:
            self._target_position_xy = (
                state.base_position[:2]
                + position_error * (cfg.max_position_error / position_error_norm)
            )
        target_yaw_error = (
            self._target_yaw - float(state.base_rpy[2]) + np.pi
        ) % (2.0 * np.pi) - np.pi
        self._target_yaw = float(state.base_rpy[2]) + float(
            np.clip(target_yaw_error, -cfg.max_yaw_error, cfg.max_yaw_error)
        )
        yaw_error = (float(state.base_rpy[2]) - self._target_yaw + np.pi) % (2.0 * np.pi) - np.pi
        unwrapped_rpy = state.base_rpy.copy()
        unwrapped_rpy[2] = self._target_yaw + yaw_error
        pose = np.concatenate((state.base_position, unwrapped_rpy))
        rate = np.concatenate((linear_velocity_world, angular_velocity_world))
        current = np.column_stack((pose, rate)).reshape(-1)

        desired_height = terrain.body_height_target(
            state.base_position[0],
            command.vx,
            nominal_height,
            support_height=support_height,
        )
        if not np.isfinite(float(support_pitch)):
            raise ValueError("support_pitch must be finite")
        support_pitch = float(np.clip(support_pitch, -0.45, 0.45))
        reference = np.zeros((cfg.horizon, 12), dtype=np.float64)
        for index in range(cfg.horizon):
            horizon_time = (index + 1) * cfg.dt
            target_pose = np.asarray(
                [
                    self._target_position_xy[0] + command_velocity_world[0] * horizon_time,
                    self._target_position_xy[1] + command_velocity_world[1] * horizon_time,
                    desired_height,
                    0.0,
                    support_pitch,
                    self._target_yaw + command.wz * horizon_time,
                ],
                dtype=np.float64,
            )
            target_rate = np.asarray(
                [command_velocity_world[0], command_velocity_world[1], 0.0, 0.0, 0.0, command.wz],
                dtype=np.float64,
            )
            reference[index] = np.column_stack((target_pose, target_rate)).reshape(-1)

        predicted, sequence = self._solve(current, reference)
        first_state = predicted[0].reshape(6, 2)
        acceleration_world = sequence[0, :3]
        angular_acceleration_world = sequence[0, 3:]
        acceleration_body = rotation.T @ acceleration_world
        angular_acceleration_body = rotation.T @ angular_acceleration_world
        gravity_world = np.asarray([0.0, 0.0, -cfg.gravity], dtype=np.float64)
        ground_force_body = cfg.mass_kg * (acceleration_body - rotation.T @ gravity_world)
        inertia = np.asarray(cfg.inertia_kg_m2, dtype=np.float64)
        omega_body = state.base_angular_velocity_body
        ground_torque_body = (
            inertia * angular_acceleration_body
            + np.cross(omega_body, inertia * omega_body)
        )
        predicted_linear_body = rotation.T @ first_state[:3, 1]
        predicted_angular_body = rotation.T @ first_state[3:, 1]
        body_target = BodyTarget(
            velocity_body=np.asarray(
                [predicted_linear_body[0], predicted_linear_body[1], predicted_angular_body[2]],
                dtype=np.float64,
            ),
            acceleration_body=acceleration_body,
            height=float(desired_height),
            # Keep the orientation sent to WBC consistent with the pose used
            # by the MPC solve.  Omitting support_pitch here makes the
            # terrain estimate diagnostic-only and leaves the body controller
            # with a contradictory level target during mixed support.
            rpy=np.asarray([0.0, support_pitch, reference[0, 10]], dtype=np.float64),
            angular_acceleration_body=angular_acceleration_body,
            centroidal_wrench=np.concatenate((ground_force_body, ground_torque_body)),
        )
        return MpcOutput(body_target, predicted, sequence)
