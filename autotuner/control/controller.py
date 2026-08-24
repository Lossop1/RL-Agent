"""Composition root for the model-based MPC/WBC controller."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

import numpy as np

from .contracts import BodyCommand, FootPlan, JointCommand, RobotState, BodyTarget
from .gait import ContinuousTrotGait
from .mpc import CentroidalMpc, MpcOutput
from .terrain import ContactTerrainEstimator, StairTerrain, TerrainModel
from .wbc import FloatingBaseWbc, WholeBodyDiagnostics


@dataclass(frozen=True)
class ControlStep:
    body: BodyTarget
    mpc: MpcOutput
    feet: FootPlan
    joints: JointCommand
    wbc: WholeBodyDiagnostics
    phase: float
    support_height: float | None
    support_pitch: float


class MpcWbcController:
    """MPC -> continuous gait -> contact adaptation -> WBC -> impedance."""

    def __init__(
        self,
        mpc: CentroidalMpc,
        gait: ContinuousTrotGait,
        wbc: FloatingBaseWbc,
        terrain: TerrainModel,
        nominal_height: float,
        contact_estimator: ContactTerrainEstimator | None = None,
        foot_radius: float = 0.042,
    ) -> None:
        self.mpc = mpc
        self.gait = gait
        self.wbc = wbc
        self.terrain = terrain
        self.nominal_height = float(nominal_height)
        self.contact_estimator = contact_estimator
        self.foot_radius = float(foot_radius)
        self._last_admitted_contacts = np.ones(4, dtype=bool)

    @staticmethod
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

    @staticmethod
    def _world_foot_positions(state: RobotState) -> np.ndarray:
        roll, pitch, yaw = (float(value) for value in state.base_rpy)
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rotation = np.asarray(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=np.float64,
        )
        return state.foot_positions_body @ rotation.T + state.base_position

    def reset(self) -> None:
        self.mpc.reset()
        self.gait.reset()
        self.wbc.reset()
        if self.contact_estimator is not None:
            self.contact_estimator.reset()
        self._last_admitted_contacts.fill(True)

    def _braking_target(self, state: RobotState, target: BodyTarget) -> BodyTarget:
        """Replace a held body reference with a bounded physical brake.

        ``hold_position`` clears the MPC position lead, but the first lifted
        state still contains the measured body velocity.  Feeding that state
        directly to WBC can let an unstable support set keep translating for
        several cycles.  A bounded brake makes the safety gate an actual
        closed-loop stop while the gait captures a replacement foothold.
        """

        rotation = self._rotation_from_rpy(state.base_rpy)
        max_linear = np.asarray(self.mpc.config.max_acceleration[:3], dtype=np.float64)
        max_angular = np.asarray(self.mpc.config.max_acceleration[3:], dtype=np.float64)
        brake_dt = max(float(self.wbc.config.dt), float(self.mpc.config.dt))
        velocity_world = rotation @ state.base_velocity_body
        angular_world = rotation @ state.base_angular_velocity_body
        linear_acceleration_world = np.clip(
            -velocity_world / brake_dt,
            -max_linear,
            max_linear,
        )
        angular_acceleration_world = np.clip(
            -angular_world / brake_dt,
            -max_angular,
            max_angular,
        )
        return replace(
            target,
            velocity_body=np.zeros(3, dtype=np.float64),
            acceleration_body=rotation.T @ linear_acceleration_world,
            angular_acceleration_body=rotation.T @ angular_acceleration_world,
        )

    def step(self, state: RobotState, command: BodyCommand) -> ControlStep:
        support_height: float | None = None
        support_pitch = 0.0
        effective_command = self.gait.effective_command(state, command)
        if self.contact_estimator is not None:
            # The estimator receives upward, force-bearing contacts from the
            # backend. Height clustering requires multiple feet at one level,
            # so a single grazing swing contact cannot change the body target.
            foot_world = self._world_foot_positions(state)
            self.contact_estimator.update(
                state.contacts,
                foot_world[:, 2],
                self.foot_radius,
                support_mask=self._last_admitted_contacts,
                foot_world_xy=foot_world[:, :2],
                reference_x=float(state.base_position[0]),
                support_direction=float(command.vx),
            )
            support_height = self.contact_estimator.support_height()
            estimated_support_pitch = self.contact_estimator.support_pitch()
            # StairTerrain is piecewise horizontal.  A mixed set of feet on
            # adjacent treads is not a physical support plane; feeding its
            # regression slope into the body target would make the controller
            # pitch toward the step transition.  Keep the estimate available
            # for continuous terrain models, but keep nominal stairs level.
            support_pitch = (
                0.0
                if isinstance(self.terrain, StairTerrain)
                else estimated_support_pitch
            )
        # A stair foothold transition is allowed to continue its continuous
        # swing trajectory, but MPC must not keep advancing the body while no
        # new support pair has been established. This prevents the controller
        # from converting a valid foothold attempt into an unsupported fall.
        hold_body_progress = self.gait.should_hold_body_progress(
            state, command, self.terrain
        )
        mpc_command = command
        if hold_body_progress:
            mpc_command = BodyCommand()
        else:
            mpc_command = effective_command
        mpc_output = self.mpc.step(
            state,
            mpc_command,
            self.terrain,
            self.nominal_height,
            support_height=support_height,
            support_pitch=support_pitch,
            hold_position=hold_body_progress,
        )
        if hold_body_progress:
            mpc_output = replace(
                mpc_output,
                target=self._braking_target(state, mpc_output.target),
            )
        foot_plan = self.gait.plan(
            state,
            command,
            mpc_output.target,
            self.terrain,
            self.contact_estimator,
        )
        admitted_contacts = self.wbc.admit_contacts(
            state,
            foot_plan,
        )
        whole_body = self.wbc.solve(
            state,
            foot_plan,
            mpc_output.target,
            active_contacts=admitted_contacts,
        )
        self._last_admitted_contacts = whole_body.diagnostics.active_contacts.copy()
        return ControlStep(
            mpc_output.target,
            mpc_output,
            foot_plan,
            whole_body.command,
            whole_body.diagnostics,
            self.gait.phase,
            support_height,
            support_pitch,
        )
