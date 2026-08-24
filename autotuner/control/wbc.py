"""基于浮基逆动力学 QP 的全身控制器。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import osqp
from scipy import sparse

from .contracts import BodyTarget, FootPlan, JointCommand, RobotState, WholeBodyDynamics
from .errors import ControlSolveError


def _finite_array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result.copy()


def _rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:
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


def _orientation_error_world(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """返回从当前姿态旋转到目标姿态的世界系小角度误差。"""

    return 0.5 * sum(
        (np.cross(current[:, axis], desired[:, axis]) for axis in range(3)),
        start=np.zeros(3, dtype=np.float64),
    )


@dataclass(frozen=True)
class WholeBodyConfig:
    dt: float = 0.02
    base_acceleration_weight: tuple[float, ...] = (100.0, 150.0, 250.0, 300.0, 300.0, 100.0)
    base_acceleration_limit: tuple[float, ...] = (5.0, 5.0, 8.0, 8.0, 8.0, 6.0)
    swing_acceleration_weight: tuple[float, float, float] = (20.0, 20.0, 20.0)
    height_transition_vertical_weight: float = 100.0
    contact_acquisition_horizontal_weight: float = 300.0
    contact_acquisition_vertical_weight: float = 300.0
    contact_acquisition_downward_acceleration: float = 0.8
    contact_acquisition_posture_acceleration_weight: float = 4.0
    contact_acquisition_body_height_error_threshold: float = 0.03
    posture_acceleration_weight: float = 4.0
    generalized_acceleration_regularization: float = 1.0e-6
    contact_force_regularization: float = 1.0e-5
    joint_torque_regularization: float = 1.0e-6
    height_position_gain: float = 30.0
    height_velocity_gain: float = 8.0
    orientation_position_gain: float = 35.0
    orientation_velocity_gain: float = 7.0
    swing_position_gain: float = 140.0
    swing_velocity_gain: float = 22.0
    max_swing_acceleration: float = 60.0
    posture_position_gain: float = 18.0
    posture_velocity_gain: float = 4.0
    posture_acceleration_limit: tuple[float, ...] = (120.0,) * 12
    kp: tuple[float, ...] = (120.0,) * 12
    kd: tuple[float, ...] = (10.0,) * 12
    effort_limit: tuple[float, ...] = (320.0,) * 4 + (110.0,) * 4 + (220.0,) * 4
    velocity_limit: tuple[float, ...] = (13.1,) * 4 + (16.55,) * 4 + (8.27,) * 4
    contact_friction: float = 0.8
    minimum_contact_normal_force: float = 0.0
    max_contact_normal_force: float = 400.0
    osqp_absolute_tolerance: float = 1.0e-5
    osqp_relative_tolerance: float = 1.0e-5
    osqp_max_iterations: int = 4000

    def __post_init__(self) -> None:
        positive_scalars = (
            self.dt,
            self.contact_acquisition_horizontal_weight,
            self.contact_acquisition_vertical_weight,
            self.contact_acquisition_downward_acceleration,
            self.height_transition_vertical_weight,
            self.max_swing_acceleration,
            self.max_contact_normal_force,
            self.contact_friction,
            self.osqp_absolute_tolerance,
            self.osqp_relative_tolerance,
        )
        nonnegative_scalars = (
            self.posture_acceleration_weight,
            self.contact_acquisition_posture_acceleration_weight,
            self.contact_acquisition_body_height_error_threshold,
            self.generalized_acceleration_regularization,
            self.contact_force_regularization,
            self.joint_torque_regularization,
            self.height_position_gain,
            self.height_velocity_gain,
            self.orientation_position_gain,
            self.orientation_velocity_gain,
            self.swing_position_gain,
            self.swing_velocity_gain,
            self.posture_position_gain,
            self.posture_velocity_gain,
        )
        if any(float(value) <= 0.0 for value in positive_scalars):
            raise ValueError("positive whole-body configuration value is not positive")
        if not 0.0 <= float(self.minimum_contact_normal_force) <= float(
            self.max_contact_normal_force
        ):
            raise ValueError("minimum_contact_normal_force must be within contact force limits")
        if len(self.swing_acceleration_weight) != 3 or any(
            float(value) <= 0.0 for value in self.swing_acceleration_weight
        ):
            raise ValueError("swing_acceleration_weight must contain three positive values")
        if any(float(value) < 0.0 for value in nonnegative_scalars):
            raise ValueError("whole-body gains and regularization must be non-negative")
        if self.osqp_max_iterations <= 0:
            raise ValueError("osqp_max_iterations must be positive")
        if len(self.base_acceleration_weight) != 6 or any(
            float(value) <= 0.0 for value in self.base_acceleration_weight
        ):
            raise ValueError("base_acceleration_weight must contain six positive values")
        if len(self.base_acceleration_limit) != 6 or any(
            float(value) <= 0.0 for value in self.base_acceleration_limit
        ):
            raise ValueError("base_acceleration_limit must contain six positive values")
        for name, values in (
            ("posture_acceleration_limit", self.posture_acceleration_limit),
            ("kp", self.kp),
            ("kd", self.kd),
            ("effort_limit", self.effort_limit),
            ("velocity_limit", self.velocity_limit),
        ):
            if len(values) != 12 or any(float(value) <= 0.0 for value in values):
                raise ValueError(f"{name} must contain twelve positive values")


@dataclass(frozen=True)
class WholeBodyDiagnostics:
    solver_status: str
    solver_iterations: int
    objective_value: float
    dynamics_residual: float
    contact_acceleration_residual: float
    active_contacts: np.ndarray
    generalized_acceleration: np.ndarray
    contact_forces_world: np.ndarray
    desired_base_acceleration_world: np.ndarray
    achieved_base_acceleration_world: np.ndarray
    base_acceleration_residual_world: np.ndarray
    foot_tracking_mask: np.ndarray
    desired_foot_accelerations_world: np.ndarray
    achieved_foot_accelerations_world: np.ndarray
    foot_acceleration_residuals_world: np.ndarray

    def __post_init__(self) -> None:
        active = np.asarray(self.active_contacts, dtype=bool)
        acceleration = np.asarray(self.generalized_acceleration, dtype=np.float64)
        forces = np.asarray(self.contact_forces_world, dtype=np.float64)
        tracking = np.asarray(self.foot_tracking_mask, dtype=bool)
        if active.shape != (4,):
            raise ValueError("active_contacts must have shape (4,)")
        if acceleration.ndim != 1 or not np.isfinite(acceleration).all():
            raise ValueError("generalized_acceleration must be a finite vector")
        if forces.shape != (4, 3) or not np.isfinite(forces).all():
            raise ValueError("contact_forces_world must be finite with shape (4, 3)")
        if tracking.shape != (4,):
            raise ValueError("foot_tracking_mask must have shape (4,)")
        for name, value, shape in (
            ("desired_base_acceleration_world", self.desired_base_acceleration_world, (6,)),
            ("achieved_base_acceleration_world", self.achieved_base_acceleration_world, (6,)),
            ("base_acceleration_residual_world", self.base_acceleration_residual_world, (6,)),
            ("desired_foot_accelerations_world", self.desired_foot_accelerations_world, (4, 3)),
            ("achieved_foot_accelerations_world", self.achieved_foot_accelerations_world, (4, 3)),
            ("foot_acceleration_residuals_world", self.foot_acceleration_residuals_world, (4, 3)),
        ):
            object.__setattr__(self, name, _finite_array(value, shape, name))
        if self.solver_iterations < 0:
            raise ValueError("solver_iterations must be non-negative")
        for name in ("objective_value", "dynamics_residual", "contact_acceleration_residual"):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        object.__setattr__(self, "active_contacts", active.copy())
        object.__setattr__(self, "generalized_acceleration", acceleration.copy())
        object.__setattr__(self, "contact_forces_world", forces.copy())
        object.__setattr__(self, "foot_tracking_mask", tracking.copy())

    def as_dict(self) -> dict[str, object]:
        return {
            "solver_status": self.solver_status,
            "solver_iterations": self.solver_iterations,
            "objective_value": self.objective_value,
            "dynamics_residual": self.dynamics_residual,
            "contact_acceleration_residual": self.contact_acceleration_residual,
            "active_contacts": self.active_contacts.tolist(),
            "generalized_acceleration": self.generalized_acceleration.tolist(),
            "contact_forces_world": self.contact_forces_world.tolist(),
            "desired_base_acceleration_world": self.desired_base_acceleration_world.tolist(),
            "achieved_base_acceleration_world": self.achieved_base_acceleration_world.tolist(),
            "base_acceleration_residual_world": self.base_acceleration_residual_world.tolist(),
            "foot_tracking_mask": self.foot_tracking_mask.tolist(),
            "desired_foot_accelerations_world": self.desired_foot_accelerations_world.tolist(),
            "achieved_foot_accelerations_world": self.achieved_foot_accelerations_world.tolist(),
            "foot_acceleration_residuals_world": self.foot_acceleration_residuals_world.tolist(),
        }


@dataclass(frozen=True)
class WholeBodySolution:
    command: JointCommand
    diagnostics: WholeBodyDiagnostics


class FloatingBaseWbc:
    """求解浮基动力学、接触和执行器约束下的全身加速度与力矩。

    QP 变量依次是 ``[qdd, 四足世界系接触力, 12 关节力矩]``。浮基动力学、
    已承重支撑足的零加速度、单边接触、摩擦棱锥以及关节/力矩边界都是硬约束；
    MPC 基座加速度、摆腿加速度和名义关节姿态是软任务。求解失败会直接报错，
    不会退回旧的关节速度最小二乘或隐藏控制状态机。
    """

    def __init__(
        self,
        q_default: np.ndarray,
        q_lower: np.ndarray,
        q_upper: np.ndarray,
        config: WholeBodyConfig,
    ) -> None:
        self.q_default = np.asarray(q_default, dtype=np.float64).copy()
        self.q_lower = np.asarray(q_lower, dtype=np.float64).copy()
        self.q_upper = np.asarray(q_upper, dtype=np.float64).copy()
        self.config = config
        if any(array.shape != (12,) for array in (self.q_default, self.q_lower, self.q_upper)):
            raise ValueError("WBC joint vectors must have shape (12,)")
        if np.any(self.q_lower >= self.q_upper):
            raise ValueError("WBC joint lower limits must be below upper limits")
        self._previous_swing_mask = np.zeros(4, dtype=bool)
        self._swing_contact_armed = np.zeros(4, dtype=bool)

    def reset(self) -> None:
        self._previous_swing_mask.fill(False)
        self._swing_contact_armed.fill(False)

    @staticmethod
    def _add_task(
        hessian: np.ndarray,
        gradient: np.ndarray,
        matrix: np.ndarray,
        target: np.ndarray,
        weight: float | np.ndarray,
    ) -> None:
        weights = np.broadcast_to(np.asarray(weight, dtype=np.float64), target.shape)
        weighted_matrix = matrix * weights[:, None]
        hessian += 2.0 * matrix.T @ weighted_matrix
        gradient -= 2.0 * matrix.T @ (weights * target)

    def _solve_qp(
        self,
        hessian: np.ndarray,
        gradient: np.ndarray,
        constraint: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ):
        cfg = self.config
        solver = osqp.OSQP()
        solver.setup(
            P=sparse.csc_matrix(np.triu(0.5 * (hessian + hessian.T))),
            q=gradient,
            A=sparse.csc_matrix(constraint),
            l=lower,
            u=upper,
            verbose=False,
            polishing=True,
            eps_abs=cfg.osqp_absolute_tolerance,
            eps_rel=cfg.osqp_relative_tolerance,
            max_iter=cfg.osqp_max_iterations,
        )
        return solver.solve(raise_error=False)

    def _joint_acceleration_bounds(
        self,
        state: RobotState,
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        dt = cfg.dt
        velocity_limit = np.asarray(cfg.velocity_limit, dtype=np.float64)
        lower = np.full(12, -np.inf, dtype=np.float64)
        upper = np.full(12, np.inf, dtype=np.float64)
        lower = np.maximum(lower, (-velocity_limit - state.dq) / dt)
        upper = np.minimum(upper, (velocity_limit - state.dq) / dt)
        position_lower = 2.0 * (self.q_lower - state.q - dt * state.dq) / (dt * dt)
        position_upper = 2.0 * (self.q_upper - state.q - dt * state.dq) / (dt * dt)
        lower = np.maximum(lower, position_lower)
        upper = np.minimum(upper, position_upper)
        if np.any(lower > upper):
            indices = np.flatnonzero(lower > upper).tolist()
            raise ControlSolveError(
                "joint_acceleration_bounds_infeasible",
                f"joint acceleration bounds are infeasible at joints {indices}",
                details={
                    "joint_indices": indices,
                    "lower": lower.tolist(),
                    "upper": upper.tolist(),
                },
            )
        return lower, upper

    def _base_acceleration_target(
        self,
        state: RobotState,
        target: BodyTarget,
        rotation: np.ndarray,
        dynamics: WholeBodyDynamics,
    ) -> np.ndarray:
        cfg = self.config
        base_velocity_world = dynamics.base_jacobian[:3] @ dynamics.generalized_velocity
        angular_velocity_world = dynamics.base_jacobian[3:] @ dynamics.generalized_velocity
        linear = rotation @ target.acceleration_body
        linear[2] += (
            cfg.height_position_gain * (float(target.height) - float(state.base_position[2]))
            - cfg.height_velocity_gain * float(base_velocity_world[2])
        )
        desired_rotation = _rotation_from_rpy(target.rpy)
        orientation_error = _orientation_error_world(rotation, desired_rotation)
        desired_angular_velocity = rotation @ np.asarray(
            [0.0, 0.0, float(target.velocity_body[2])], dtype=np.float64
        )
        angular = (
            rotation @ target.angular_acceleration_body
            + cfg.orientation_position_gain * orientation_error
            + cfg.orientation_velocity_gain * (desired_angular_velocity - angular_velocity_world)
        )
        target_acceleration = np.concatenate((linear, angular))
        limit = np.asarray(cfg.base_acceleration_limit, dtype=np.float64)
        return np.clip(target_acceleration, -limit, limit)

    def _active_contact_mask(
        self,
        state: RobotState,
        plan: FootPlan,
    ) -> np.ndarray:
        """Select physical supports without allowing an unsafe phase gap.

        A nominal phase can still contain an old contact at the instant a
        swing begins. Event-based admission distinguishes that stale liftoff
        contact from a support reacquired during the swing; it does not
        prescribe a leg sequence or a foothold.
        """

        swing = plan.swing_mask
        newly_swinging = swing & (~self._previous_swing_mask)
        self._swing_contact_armed[newly_swinging] = ~state.contacts.in_contact[
            newly_swinging
        ]
        self._swing_contact_armed[
            swing & (~newly_swinging) & (~state.contacts.in_contact)
        ] = True
        self._swing_contact_armed[~swing] = False

        candidate = state.contacts.in_contact & (
            (~swing) | self._swing_contact_armed
        )

        self._previous_swing_mask = swing.copy()
        return candidate

    def admit_contacts(
        self,
        state: RobotState,
        plan: FootPlan,
    ) -> np.ndarray:
        """Return the event-admitted load-bearing contacts for this cycle.

        The controller uses the same admission result for terrain estimation
        and WBC.  Keeping this transition in one place prevents a stale
        contact at liftoff from becoming a global support-height measurement.
        """

        return self._active_contact_mask(state, plan)

    def _swing_acceleration_target(
        self,
        leg: int,
        state: RobotState,
        plan: FootPlan,
        base_acceleration: np.ndarray,
        rotation: np.ndarray,
        dynamics: WholeBodyDynamics,
    ) -> np.ndarray:
        cfg = self.config
        generalized_velocity = dynamics.generalized_velocity
        base_velocity_world = dynamics.base_jacobian[:3] @ generalized_velocity
        angular_velocity_world = dynamics.base_jacobian[3:] @ generalized_velocity
        current_offset_world = rotation @ state.foot_positions_body[leg]
        desired_offset_world = rotation @ plan.positions_body[leg]
        current_position_world = state.base_position + current_offset_world
        desired_position_world = state.base_position + desired_offset_world
        current_velocity_world = dynamics.foot_jacobians[leg] @ generalized_velocity
        desired_velocity_world = (
            base_velocity_world
            + np.cross(angular_velocity_world, desired_offset_world)
            + rotation @ plan.velocities_body[leg]
        )
        desired = (
            cfg.swing_position_gain * (desired_position_world - current_position_world)
            + cfg.swing_velocity_gain * (desired_velocity_world - current_velocity_world)
        )
        physical_world_anchor = plan.world_anchored_mask[leg] and (
            plan.contact_acquisition_mask[leg]
            or plan.height_transition_mask[leg]
        )
        if not physical_world_anchor:
            # Ordinary body-relative swing references benefit from the base
            # feed-forward term. A latched touchdown or stair transition is
            # already world anchored, so adding base acceleration would move
            # the foot away from its physical landing target.
            desired += (
                base_acceleration[:3]
                + np.cross(base_acceleration[3:], desired_offset_world)
                + np.cross(
                    angular_velocity_world,
                    np.cross(angular_velocity_world, desired_offset_world),
                )
            )
        norm = float(np.linalg.norm(desired))
        if norm > cfg.max_swing_acceleration:
            desired *= cfg.max_swing_acceleration / norm
        return desired

    def solve(
        self,
        state: RobotState,
        plan: FootPlan,
        body_target: BodyTarget,
        active_contacts: np.ndarray | None = None,
    ) -> WholeBodySolution:
        dynamics = state.dynamics
        if dynamics is None:
            raise ControlSolveError(
                "missing_dynamics",
                "floating-base WBC requires a synchronized WholeBodyDynamics snapshot",
            )
        if not np.allclose(
            state.dq,
            dynamics.generalized_velocity[dynamics.actuated_dof_indices],
            atol=1.0e-8,
            rtol=1.0e-8,
        ):
            raise ValueError("RobotState joint velocity is not synchronized with WholeBodyDynamics")

        cfg = self.config
        nv = int(dynamics.generalized_velocity.size)
        force_offset = nv
        torque_offset = nv + 12
        variable_count = nv + 24
        hessian = np.eye(variable_count, dtype=np.float64) * 2.0e-10
        gradient = np.zeros(variable_count, dtype=np.float64)

        rotation = _rotation_from_rpy(state.base_rpy)
        desired_base_acceleration = self._base_acceleration_target(
            state, body_target, rotation, dynamics
        )
        base_task_target = desired_base_acceleration - dynamics.base_bias_acceleration
        base_task = np.zeros((6, variable_count), dtype=np.float64)
        base_task[:, :nv] = dynamics.base_jacobian
        self._add_task(
            hessian,
            gradient,
            base_task,
            base_task_target,
            np.asarray(cfg.base_acceleration_weight, dtype=np.float64),
        )

        task_matrices = [base_task]
        task_targets = [base_task_target]
        if active_contacts is None:
            active_contacts = self._active_contact_mask(state, plan)
        else:
            active_contacts = np.asarray(active_contacts, dtype=bool).copy()
            if active_contacts.shape != (4,):
                raise ValueError("active_contacts must have shape (4,)")
        # 相位进入支撑并不等于已经建立物理接触。尚未落地的计划支撑腿继续
        # 跟踪落脚目标；只有测得承重后才切换为零足端加速度硬约束。
        # A physical foot can remain in contact with an old stair tread after
        # a higher/lower support cluster has become authoritative. It is no
        # longer a hard support, but it still needs an unloading/position task
        # until it leaves the surface. Otherwise the simulator supplies an
        # unmodelled collision force while the QP assumes free space.
        foot_tracking_mask = (
            plan.swing_mask | plan.expected_contacts
        ) & (~active_contacts)
        desired_foot_accelerations = np.zeros((4, 3), dtype=np.float64)
        for leg in np.flatnonzero(foot_tracking_mask):
            swing_task = np.zeros((3, variable_count), dtype=np.float64)
            swing_task[:, :nv] = dynamics.foot_jacobians[leg]
            desired_foot_acceleration = self._swing_acceleration_target(
                int(leg),
                state,
                plan,
                desired_base_acceleration,
                rotation,
                dynamics,
            )
            if (
                plan.contact_acquisition_mask[leg]
                and not active_contacts[leg]
            ):
                # Exact geometric touchdown is neutrally stable under a soft
                # task: the foot can hover at the contact manifold without
                # ever generating measurable load. A bounded downward bias
                # closes that physical contact; it is removed as soon as the
                # backend admits the foot as support.
                desired_foot_acceleration[2] -= (
                    cfg.contact_acquisition_downward_acceleration
                )
            desired_foot_accelerations[leg] = desired_foot_acceleration
            swing_target = desired_foot_acceleration - dynamics.foot_bias_accelerations[leg]
            swing_weight = np.asarray(
                cfg.swing_acceleration_weight, dtype=np.float64
            )
            tracking_weight: np.ndarray = swing_weight
            if plan.height_transition_mask[leg]:
                tracking_weight = np.asarray(
                    [
                        swing_weight[0],
                        swing_weight[1],
                        cfg.height_transition_vertical_weight,
                    ],
                    dtype=np.float64,
                )
            if not plan.swing_mask[leg] or (
                plan.world_anchored_mask[leg]
                and plan.contact_acquisition_mask[leg]
                and not plan.height_transition_mask[leg]
            ):
                # A held touchdown is still represented by the latched swing
                # target, but it is now a contact-acquisition task. Without
                # this branch the ordinary swing weight can leave the foot a
                # few millimetres above the tread indefinitely.
                tracking_weight = np.asarray(
                    [
                        cfg.contact_acquisition_horizontal_weight,
                        cfg.contact_acquisition_horizontal_weight,
                        cfg.contact_acquisition_vertical_weight,
                    ],
                    dtype=np.float64,
                )
            self._add_task(
                hessian,
                gradient,
                swing_task,
                swing_target,
                tracking_weight,
            )
            task_matrices.append(swing_task)
            task_targets.append(swing_target)

        posture_task = np.zeros((12, variable_count), dtype=np.float64)
        for joint, dof in enumerate(dynamics.actuated_dof_indices):
            posture_task[joint, int(dof)] = 1.0
        posture_target = (
            cfg.posture_position_gain * (self.q_default - state.q)
            - cfg.posture_velocity_gain * state.dq
        )
        acceleration_limit = np.asarray(cfg.posture_acceleration_limit, dtype=np.float64)
        posture_target = np.clip(posture_target, -acceleration_limit, acceleration_limit)
        posture_weights: float | np.ndarray = cfg.posture_acceleration_weight
        # A world-anchored touchdown is a temporary geometric obligation:
        # retaining the old nominal posture can make the leg unable to reach
        # the already-latched support point. Keep posture as a soft task for
        # ordinary motion, but let the explicit acquisition policy decide its
        # weight while an unconfirmed landing is being captured.
        acquisition_mask = (
            plan.contact_acquisition_mask
            & (~active_contacts)
        )
        body_height_transition = (
            abs(float(body_target.height) - float(state.base_position[2]))
            > cfg.contact_acquisition_body_height_error_threshold
        )
        acquisition_mask &= (
            (~plan.height_transition_mask) | body_height_transition
        )
        if np.any(acquisition_mask):
            posture_weights = np.full(
                12,
                cfg.posture_acceleration_weight,
                dtype=np.float64,
            )
            for leg in np.flatnonzero(acquisition_mask):
                posture_weights[[int(leg), 4 + int(leg), 8 + int(leg)]] = (
                    cfg.contact_acquisition_posture_acceleration_weight
                )
        if np.any(np.asarray(posture_weights, dtype=np.float64) > 0.0):
            self._add_task(
                hessian,
                gradient,
                posture_task,
                posture_target,
                posture_weights,
            )

        hessian[:nv, :nv] += np.eye(nv) * (2.0 * cfg.generalized_acceleration_regularization)
        hessian[force_offset:torque_offset, force_offset:torque_offset] += np.eye(12) * (
            2.0 * cfg.contact_force_regularization
        )
        hessian[torque_offset:, torque_offset:] += np.eye(12) * (
            2.0 * cfg.joint_torque_regularization
        )

        actuation = np.zeros((nv, 12), dtype=np.float64)
        for joint, dof in enumerate(dynamics.actuated_dof_indices):
            actuation[int(dof), joint] = 1.0
        stacked_foot_jacobian = dynamics.foot_jacobians.reshape(12, nv)

        constraint_matrices: list[np.ndarray] = []
        constraint_lower: list[np.ndarray] = []
        constraint_upper: list[np.ndarray] = []

        dynamics_constraint = np.zeros((nv, variable_count), dtype=np.float64)
        dynamics_constraint[:, :nv] = dynamics.mass_matrix
        dynamics_constraint[:, force_offset:torque_offset] = -stacked_foot_jacobian.T
        dynamics_constraint[:, torque_offset:] = -actuation
        dynamics_rhs = -dynamics.bias_force
        constraint_matrices.append(dynamics_constraint)
        constraint_lower.append(dynamics_rhs)
        constraint_upper.append(dynamics_rhs)

        for leg in np.flatnonzero(active_contacts):
            contact_constraint = np.zeros((3, variable_count), dtype=np.float64)
            contact_constraint[:, :nv] = dynamics.foot_jacobians[leg]
            contact_rhs = -dynamics.foot_bias_accelerations[leg]
            constraint_matrices.append(contact_constraint)
            constraint_lower.append(contact_rhs)
            constraint_upper.append(contact_rhs)

        variable_bounds = np.eye(variable_count, dtype=np.float64)
        variable_lower = np.full(variable_count, -np.inf, dtype=np.float64)
        variable_upper = np.full(variable_count, np.inf, dtype=np.float64)
        joint_lower, joint_upper = self._joint_acceleration_bounds(state)
        for joint, dof in enumerate(dynamics.actuated_dof_indices):
            variable_lower[int(dof)] = joint_lower[joint]
            variable_upper[int(dof)] = joint_upper[joint]
        for leg in range(4):
            force_slice = slice(force_offset + 3 * leg, force_offset + 3 * leg + 3)
            if active_contacts[leg]:
                variable_lower[force_slice] = [
                    -np.inf,
                    -np.inf,
                    cfg.minimum_contact_normal_force,
                ]
                variable_upper[force_slice] = [np.inf, np.inf, cfg.max_contact_normal_force]
            else:
                variable_lower[force_slice] = 0.0
                variable_upper[force_slice] = 0.0
        effort_limit = np.asarray(cfg.effort_limit, dtype=np.float64)
        variable_lower[torque_offset:] = -effort_limit
        variable_upper[torque_offset:] = effort_limit
        constraint_matrices.append(variable_bounds)
        constraint_lower.append(variable_lower)
        constraint_upper.append(variable_upper)

        friction_rows: list[np.ndarray] = []
        for leg in np.flatnonzero(active_contacts):
            fx = force_offset + 3 * int(leg)
            fy = fx + 1
            fz = fx + 2
            for tangent, sign in ((fx, 1.0), (fx, -1.0), (fy, 1.0), (fy, -1.0)):
                row = np.zeros(variable_count, dtype=np.float64)
                row[tangent] = sign
                row[fz] = -cfg.contact_friction
                friction_rows.append(row)
        if friction_rows:
            friction = np.asarray(friction_rows, dtype=np.float64)
            constraint_matrices.append(friction)
            constraint_lower.append(np.full(friction.shape[0], -np.inf, dtype=np.float64))
            constraint_upper.append(np.zeros(friction.shape[0], dtype=np.float64))

        constraint = np.vstack(constraint_matrices)
        lower = np.concatenate(constraint_lower)
        upper = np.concatenate(constraint_upper)

        result = self._solve_qp(hessian, gradient, constraint, lower, upper)
        if result.info.status_val not in (1, 2) or result.x is None:
            raise ControlSolveError(
                "qp_unsolved",
                "floating-base WBC QP failed: "
                f"status={result.info.status}, iterations={result.info.iter}",
                details={
                    "solver_status": str(result.info.status),
                    "solver_status_value": int(result.info.status_val),
                    "solver_iterations": int(result.info.iter),
                    "objective_value": float(result.info.obj_val),
                    "active_contacts": active_contacts.tolist(),
                    "foot_tracking_mask": foot_tracking_mask.tolist(),
                },
            )
        solution = np.asarray(result.x, dtype=np.float64)
        if solution.shape != (variable_count,) or not np.isfinite(solution).all():
            raise ControlSolveError(
                "nonfinite_solution",
                "floating-base WBC QP returned a non-finite solution",
                details={
                    "solver_status": str(result.info.status),
                    "solver_iterations": int(result.info.iter),
                    "solution_shape": list(solution.shape),
                },
            )

        generalized_acceleration = solution[:nv]
        contact_forces = solution[force_offset:torque_offset].reshape(4, 3)
        joint_torque = solution[torque_offset:]
        joint_acceleration = generalized_acceleration[dynamics.actuated_dof_indices]
        velocity = np.clip(
            state.dq + cfg.dt * joint_acceleration,
            -np.asarray(cfg.velocity_limit, dtype=np.float64),
            np.asarray(cfg.velocity_limit, dtype=np.float64),
        )
        position = np.clip(
            state.q + cfg.dt * state.dq + 0.5 * cfg.dt * cfg.dt * joint_acceleration,
            self.q_lower,
            self.q_upper,
        )
        kp = np.asarray(cfg.kp, dtype=np.float64)
        kd = np.asarray(cfg.kd, dtype=np.float64)
        task_matrix = np.vstack(task_matrices)
        task_target = np.concatenate(task_targets)
        task_residual = float(
            np.linalg.norm(task_matrix @ solution - task_target)
            / max(np.sqrt(task_target.size), 1.0)
        )
        dynamics_residual = float(
            np.max(np.abs(dynamics_constraint @ solution - dynamics_rhs))
        )
        contact_residuals = [
            dynamics.foot_jacobians[leg] @ generalized_acceleration
            + dynamics.foot_bias_accelerations[leg]
            for leg in np.flatnonzero(active_contacts)
        ]
        contact_residual = float(
            max((np.max(np.abs(value)) for value in contact_residuals), default=0.0)
        )
        achieved_base_acceleration = (
            dynamics.base_jacobian @ generalized_acceleration
            + dynamics.base_bias_acceleration
        )
        base_acceleration_residual = (
            achieved_base_acceleration - desired_base_acceleration
        )
        achieved_foot_accelerations = np.zeros((4, 3), dtype=np.float64)
        foot_acceleration_residuals = np.zeros((4, 3), dtype=np.float64)
        for leg in np.flatnonzero(foot_tracking_mask):
            achieved = (
                dynamics.foot_jacobians[leg] @ generalized_acceleration
                + dynamics.foot_bias_accelerations[leg]
            )
            achieved_foot_accelerations[leg] = achieved
            foot_acceleration_residuals[leg] = (
                achieved - desired_foot_accelerations[leg]
            )
        command = JointCommand(
            position=position,
            velocity=velocity,
            torque=joint_torque,
            kp=kp,
            kd=kd,
            saturated=np.abs(joint_torque) >= effort_limit - 1.0e-6,
            task_residual=task_residual,
            acceleration=joint_acceleration,
            reference_duration=cfg.dt,
            feedforward_torque=joint_torque,
        )
        diagnostics = WholeBodyDiagnostics(
            solver_status=str(result.info.status),
            solver_iterations=int(result.info.iter),
            objective_value=float(result.info.obj_val),
            dynamics_residual=dynamics_residual,
            contact_acceleration_residual=contact_residual,
            active_contacts=active_contacts,
            generalized_acceleration=generalized_acceleration,
            contact_forces_world=contact_forces,
            desired_base_acceleration_world=desired_base_acceleration,
            achieved_base_acceleration_world=achieved_base_acceleration,
            base_acceleration_residual_world=base_acceleration_residual,
            foot_tracking_mask=foot_tracking_mask,
            desired_foot_accelerations_world=desired_foot_accelerations,
            achieved_foot_accelerations_world=achieved_foot_accelerations,
            foot_acceleration_residuals_world=foot_acceleration_residuals,
        )
        return WholeBodySolution(command, diagnostics)
