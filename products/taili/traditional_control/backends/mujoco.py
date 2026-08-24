"""MuJoCo state/action adapter for the Taili model controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autotuner.control.contracts import (
    ContactState,
    JointCommand,
    RobotState,
    WholeBodyDynamics,
)
from autotuner.control.servo import HighRateImpedanceServo

from ..profile import TailiProfile


def _rotate_inverse(quat_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    w = float(quat_wxyz[0])
    xyz = np.asarray(quat_wxyz[1:], dtype=np.float64)
    return vector * (2.0 * w * w - 1.0) - 2.0 * w * np.cross(xyz, vector) + 2.0 * xyz * np.dot(xyz, vector)


def _quat_to_rpy(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(item) for item in quat)
    return np.asarray(
        [
            np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)),
            np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        ],
        dtype=np.float64,
    )


def _split_free_joint_velocity(qvel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split MuJoCo free-joint qvel into world linear and angular velocity."""

    value = np.asarray(qvel, dtype=np.float64)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError("free-joint qvel must be finite with shape (6,)")
    return value[:3].copy(), value[3:].copy()


def _environment_to_foot_normal(
    contact_frame: np.ndarray,
    *,
    foot_is_geom1: bool,
) -> np.ndarray:
    """Return the contact normal oriented from the environment to the foot.

    MuJoCo stores the normal in the first three consecutive values of
    ``mjContact.frame``.  Treating the flattened frame as a matrix column
    mixes the normal with the two tangent axes and can reject valid support.
    """

    frame = np.asarray(contact_frame, dtype=np.float64)
    if frame.size != 9 or not np.isfinite(frame).all():
        raise ValueError("MuJoCo contact frame must contain nine finite values")
    normal_geom1_to_geom2 = frame.reshape(3, 3)[0].copy()
    return -normal_geom1_to_geom2 if foot_is_geom1 else normal_geom1_to_geom2


@dataclass
class MujocoStateAdapter:
    """Read canonical Taili state from an already-loaded ``MjModel/MjData``."""

    model: object
    data: object
    profile: TailiProfile
    contact_force_threshold: float = 5.0
    support_normal_min_z: float = 0.5
    base_body_name: str = "base_link"

    def __post_init__(self) -> None:
        import mujoco

        self._mujoco = mujoco
        if not 0.0 <= self.support_normal_min_z <= 1.0:
            raise ValueError("support_normal_min_z must be in [0, 1]")
        self._qpos = np.asarray([
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in self.profile.joint_names
        ])
        self._dof = np.asarray([
            self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in self.profile.joint_names
        ])
        self._base_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.base_body_name)
        if self._base_body < 0:
            for candidate in ("base", "base_link"):
                self._base_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, candidate)
                if self._base_body >= 0:
                    break
        if self._base_body < 0:
            raise ValueError("Taili MuJoCo model has no base body")
        free_joints = np.flatnonzero(
            (np.asarray(self.model.jnt_type) == int(mujoco.mjtJoint.mjJNT_FREE))
            & (np.asarray(self.model.jnt_bodyid) == self._base_body)
        )
        if free_joints.size != 1:
            raise ValueError("Taili base body must own exactly one free joint")
        self._free_dof = int(self.model.jnt_dofadr[int(free_joints[0])])
        self._feet = np.asarray([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_foot")
            for leg in ("FL", "FR", "RL", "RR")
        ])
        if np.any(self._feet < 0):
            raise ValueError("Taili MuJoCo model is missing one or more foot bodies")
        self._previous_feet = None
        self._previous_time = None
        self._foot_geoms: dict[int, int] = {}
        for geom_id in range(int(self.model.ngeom)):
            body_id = int(self.model.geom_bodyid[geom_id])
            for foot_index, foot_body in enumerate(self._feet):
                if body_id == int(foot_body):
                    self._foot_geoms[geom_id] = foot_index

    def _contacts(self) -> ContactState:
        in_contact = np.zeros(4, dtype=bool)
        forces = np.zeros(4, dtype=np.float64)
        obstacle_contact = np.zeros(4, dtype=bool)
        obstacle_forces = np.zeros(4, dtype=np.float64)
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            foot_index = self._foot_geoms.get(geom1, self._foot_geoms.get(geom2))
            if foot_index is None:
                continue
            # MuJoCo's contact frame normal points from geom1 toward geom2.
            # Orient it from the environment toward the foot, then accept it
            # as load-bearing only when it has a meaningful upward component.
            support_normal = _environment_to_foot_normal(
                contact.frame,
                foot_is_geom1=geom1 in self._foot_geoms,
            )
            force = np.zeros(6, dtype=np.float64)
            self._mujoco.mj_contactForce(self.model, self.data, index, force)
            normal_force = abs(float(force[0]))
            # A geometric touch with negligible normal load is not yet a
            # support event. Keep it out of ``in_contact`` so WBC cannot turn
            # a grazing/corner contact into a hard zero-acceleration support.
            if normal_force < self.contact_force_threshold:
                continue
            if float(support_normal[2]) < self.support_normal_min_z:
                obstacle_contact[foot_index] = True
                obstacle_forces[foot_index] = max(
                    obstacle_forces[foot_index], normal_force
                )
                continue
            in_contact[foot_index] = True
            forces[foot_index] = max(forces[foot_index], normal_force)
        return ContactState(in_contact, forces, obstacle_contact, obstacle_forces)

    def _point_jacobian(self, body_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """读取刚体原点的世界系 Jacobian 和 ``Jdot*qdot``。"""

        nv = int(self.model.nv)
        linear = np.zeros((3, nv), dtype=np.float64)
        angular = np.zeros((3, nv), dtype=np.float64)
        linear_dot = np.zeros((3, nv), dtype=np.float64)
        angular_dot = np.zeros((3, nv), dtype=np.float64)
        self._mujoco.mj_jacBody(
            self.model,
            self.data,
            linear,
            angular,
            int(body_id),
        )
        self._mujoco.mj_jacDot(
            self.model,
            self.data,
            linear_dot,
            angular_dot,
            np.asarray(self.data.xpos[int(body_id)], dtype=np.float64),
            int(body_id),
        )
        generalized_velocity = np.asarray(self.data.qvel, dtype=np.float64)
        return (
            linear,
            angular,
            linear_dot @ generalized_velocity,
            angular_dot @ generalized_velocity,
        )

    def _dynamics(self) -> WholeBodyDynamics:
        """构造与当前 ``MjData`` 同步的完整浮基动力学快照。"""

        nv = int(self.model.nv)
        mass_matrix = np.zeros((nv, nv), dtype=np.float64)
        self._mujoco.mj_fullM(self.model, mass_matrix, self.data.qM)
        mass_matrix = 0.5 * (mass_matrix + mass_matrix.T)
        base_linear, base_angular, base_linear_bias, base_angular_bias = (
            self._point_jacobian(self._base_body)
        )
        foot_jacobians = np.zeros((4, 3, nv), dtype=np.float64)
        foot_bias = np.zeros((4, 3), dtype=np.float64)
        for leg, body_id in enumerate(self._feet):
            linear, _angular, linear_bias, _angular_bias = self._point_jacobian(int(body_id))
            foot_jacobians[leg] = linear
            foot_bias[leg] = linear_bias
        return WholeBodyDynamics(
            generalized_velocity=np.asarray(self.data.qvel, dtype=np.float64),
            mass_matrix=mass_matrix,
            bias_force=np.asarray(self.data.qfrc_bias, dtype=np.float64),
            actuated_dof_indices=self._dof,
            base_jacobian=np.vstack((base_linear, base_angular)),
            base_bias_acceleration=np.concatenate((base_linear_bias, base_angular_bias)),
            foot_jacobians=foot_jacobians,
            foot_bias_accelerations=foot_bias,
        )

    def read_state(self) -> RobotState:
        base_position = np.asarray(self.data.xpos[self._base_body], dtype=np.float64).copy()
        base_rotation = np.asarray(self.data.xmat[self._base_body], dtype=np.float64).reshape(3, 3)
        quat = np.asarray(self.data.xquat[self._base_body], dtype=np.float64)
        q = np.asarray(self.data.qpos[self._qpos], dtype=np.float64).copy()
        dq = np.asarray(self.data.qvel[self._dof], dtype=np.float64).copy()
        free_dof = np.asarray(self.data.qvel[self._free_dof : self._free_dof + 6], dtype=np.float64)
        base_linear_world, base_angular_world = _split_free_joint_velocity(free_dof)
        foot_world = np.asarray(self.data.xpos[self._feet], dtype=np.float64).copy()
        foot_body = (foot_world - base_position) @ base_rotation
        now = float(self.data.time)
        if self._previous_feet is None or self._previous_time is None or now <= self._previous_time:
            foot_velocity = np.zeros((4, 3), dtype=np.float64)
        else:
            foot_velocity = (foot_body - self._previous_feet) / (now - self._previous_time)
        self._previous_feet = foot_body.copy()
        self._previous_time = now
        return RobotState(
            time_s=now,
            q=q,
            dq=dq,
            base_position=base_position,
            base_velocity_body=_rotate_inverse(quat, base_linear_world),
            base_rpy=_quat_to_rpy(quat),
            base_angular_velocity_body=_rotate_inverse(quat, base_angular_world),
            foot_positions_body=foot_body,
            foot_velocities_body=foot_velocity,
            contacts=self._contacts(),
            dynamics=self._dynamics(),
        )


@dataclass
class MujocoActionAdapter:
    """Run the high-rate impedance layer and write named MuJoCo actuators."""

    model: object
    data: object
    profile: TailiProfile

    def __post_init__(self) -> None:
        import mujoco

        self._actuators = np.asarray([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name.removesuffix("_joint"))
            for name in self.profile.joint_names
        ])
        if np.any(self._actuators < 0):
            raise ValueError("Taili MuJoCo model is missing a named actuator")
        self._qpos = np.asarray([
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in self.profile.joint_names
        ])
        self._dof = np.asarray([
            self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in self.profile.joint_names
        ])
        self._servo = HighRateImpedanceServo(self.profile.effort_limit)

    def write_command(self, command: JointCommand) -> None:
        self._servo.set_command(
            command,
            np.asarray(self.data.qpos[self._qpos], dtype=np.float64),
            np.asarray(self.data.qvel[self._dof], dtype=np.float64),
            float(self.data.time),
        )
        self.apply_physics_step()

    def apply_physics_step(self) -> None:
        """Recompute impedance torque at the simulator's physics rate.

        MPC/WBC references are intentionally held between policy ticks, but a
        stale torque sample must not be held through contact impulses. The
        feed-forward term comes from the contact-consistent inverse-dynamics
        QP and the impedance correction is refreshed every physics step.
        """
        if not self._servo.has_command:
            return
        q = np.asarray(self.data.qpos[self._qpos], dtype=np.float64)
        dq = np.asarray(self.data.qvel[self._dof], dtype=np.float64)
        self.data.ctrl[self._actuators] = self._servo.torque(
            q,
            dq,
            float(self.data.time),
        )


@dataclass
class MujocoBackend:
    """Taili 的完整 MuJoCo 后端；一次调用只推进一个物理子步。"""

    model: object
    data: object
    profile: TailiProfile
    contact_force_threshold: float = 5.0
    support_normal_min_z: float = 0.5

    def __post_init__(self) -> None:
        import mujoco

        self._mujoco = mujoco
        self.state = MujocoStateAdapter(
            self.model,
            self.data,
            self.profile,
            contact_force_threshold=self.contact_force_threshold,
            support_normal_min_z=self.support_normal_min_z,
        )
        self.action = MujocoActionAdapter(self.model, self.data, self.profile)

    @property
    def physics_dt(self) -> float:
        return float(self.model.opt.timestep)

    def read_state(self) -> RobotState:
        return self.state.read_state()

    def write_command(self, command: JointCommand) -> None:
        self.action.write_command(command)

    def apply_physics_step(self) -> None:
        self.action.apply_physics_step()

    def step_physics(self) -> None:
        self.apply_physics_step()
        self._mujoco.mj_step(self.model, self.data)
