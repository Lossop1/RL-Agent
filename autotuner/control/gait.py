"""Continuous phase gait and terrain-aware foot references."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import BodyCommand, BodyTarget, FootPlan, RobotState
from .support import SupportGeometry, analyze_support_geometry
from .terrain import ContactTerrainEstimator, StairTerrain, TerrainModel


def _smoothstep(value: float) -> tuple[float, float]:
    u = float(np.clip(value, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u), 6.0 * u * (1.0 - u)


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


@dataclass(frozen=True)
class GaitConfig:
    period: float = 0.72
    period_min: float = 0.58
    period_slope: float = 0.05
    yaw_speed_equivalent: float = 0.30
    duty_factor: float = 0.55
    swing_clearance: float = 0.06
    foot_radius: float = 0.042
    command_ramp_time: float = 0.30
    contact_blend_phase: float = 0.08
    foothold_velocity_gain: float = 0.35
    foothold_yaw_rate_gain: float = 0.10
    max_foothold_correction: tuple[float, float] = (0.12, 0.06)
    start_phase: float = 0.0
    lateral_start_phase: float = 0.75
    height_transition_threshold: float = 0.03
    obstacle_lift_delay: float = 0.20
    # Maximum simultaneous moving swing legs for a stair profile.  The
    # generic gait default preserves diagonal trot; a product stair profile
    # can set this to one when three-support transfer is required.
    stair_max_swing_legs: int = 2
    # Optional generic swing cap for a product profile that needs a
    # crawl-like transfer on a narrow support margin. ``None`` preserves the
    # ordinary diagonal trot.
    max_swing_legs: int | None = None

    def __post_init__(self) -> None:
        if self.period <= 0.0 or self.period_min <= 0.0 or self.period_min > self.period:
            raise ValueError("invalid gait period")
        if not 0.0 < self.duty_factor < 1.0:
            raise ValueError("duty_factor must be in (0, 1)")
        if self.swing_clearance < 0.0 or self.foot_radius <= 0.0:
            raise ValueError("swing clearance and foot radius must be valid")
        if self.command_ramp_time <= 0.0:
            raise ValueError("command_ramp_time must be positive")
        if not 0.0 < self.contact_blend_phase < 0.5:
            raise ValueError("contact_blend_phase must be in (0, 0.5)")
        if (
            self.foothold_velocity_gain < 0.0
            or self.foothold_yaw_rate_gain < 0.0
            or len(self.max_foothold_correction) != 2
            or any(value <= 0.0 for value in self.max_foothold_correction)
        ):
            raise ValueError("foothold feedback gains must be non-negative with positive limits")
        if not 0.0 <= self.lateral_start_phase < 1.0:
            raise ValueError("lateral_start_phase must be in [0, 1)")
        if not 0.0 <= self.start_phase < 1.0:
            raise ValueError("start_phase must be in [0, 1)")
        if self.height_transition_threshold <= 0.0:
            raise ValueError("height_transition_threshold must be positive")
        if not 0.0 <= self.obstacle_lift_delay < 0.8:
            raise ValueError("obstacle_lift_delay must be in [0, 0.8)")
        if self.stair_max_swing_legs < 1:
            raise ValueError("stair_max_swing_legs must be positive")
        if self.max_swing_legs is not None and self.max_swing_legs < 1:
            raise ValueError("max_swing_legs must be positive when provided")


class ContinuousTrotGait:
    """Continuous body-relative trot with latched world horizontal footholds."""

    def __init__(self, nominal_feet_body: np.ndarray, config: GaitConfig) -> None:
        feet = np.asarray(nominal_feet_body, dtype=np.float64)
        if feet.shape != (4, 3) or not np.isfinite(feet).all():
            raise ValueError("nominal_feet_body must be finite with shape (4, 3)")
        self.nominal_feet_body = feet.copy()
        self.config = config
        self.phase = 0.0
        self._last_time: float | None = None
        self._motion_scale = 0.0
        self._was_moving = False
        self._was_swing = np.zeros(4, dtype=bool)
        self._height_initialized = False
        self._stance_terrain_height = np.zeros(4, dtype=np.float64)
        self._swing_start_center_height = np.zeros(4, dtype=np.float64)
        self._swing_target_center_height = np.zeros(4, dtype=np.float64)
        self._swing_start_world_xy = np.zeros((4, 2), dtype=np.float64)
        self._swing_target_world_xy = np.zeros((4, 2), dtype=np.float64)
        self._height_transition = np.zeros(4, dtype=bool)
        self._swing_lift_delay = np.zeros(4, dtype=np.float64)
        self._stance_anchor_world_xy = np.zeros((4, 2), dtype=np.float64)
        self._stance_anchor_valid = np.zeros(4, dtype=bool)
        self._contact_hold = np.zeros(4, dtype=bool)
        self._scheduled_swing_count = np.zeros(4, dtype=np.int64)

    def reset(self) -> None:
        self.phase = 0.0
        self._last_time = None
        self._motion_scale = 0.0
        self._was_moving = False
        self._was_swing.fill(False)
        self._height_initialized = False
        self._stance_terrain_height.fill(0.0)
        self._swing_start_center_height.fill(0.0)
        self._swing_target_center_height.fill(0.0)
        self._swing_start_world_xy.fill(0.0)
        self._swing_target_world_xy.fill(0.0)
        self._height_transition.fill(False)
        self._swing_lift_delay.fill(0.0)
        self._stance_anchor_world_xy.fill(0.0)
        self._stance_anchor_valid.fill(False)
        self._contact_hold.fill(False)
        self._scheduled_swing_count.fill(0)

    def should_hold_body_progress(
        self,
        state: RobotState,
        command: BodyCommand,
        terrain: TerrainModel,
    ) -> bool:
        """Whether body translation must wait for physical support.

        This is a continuous safety gate, not a stair state machine: the
        phase and foothold references remain continuous, while MPC is denied
        additional commanded translation until an actual support pair exists.
        """

        command_active = np.linalg.norm(command.as_array()[:2]) > 0.02 or abs(command.wz) > 0.02
        if not command_active:
            return False
        if np.count_nonzero(state.contacts.in_contact) < 2:
            return True
        # A raw contact count is not enough for a floating base.  During a
        # phase/contact race the two measured feet can be on one lateral
        # side (for example ``0101`` during a lateral trot).  Letting MPC
        # continue the commanded body motion in that snapshot creates a
        # persistent roll impulse while the opposite pair is still in swing.
        # Use the measured support geometry as a short-lived body-progress
        # brake; the swing trajectory remains continuous and can capture a
        # replacement foot on the next physics steps.
        measured_world = self._world_foot_positions(state)
        support_geometry = analyze_support_geometry(
            measured_world[:, :2],
            state.contacts.in_contact,
            state.base_position[:2],
            minimum_lateral_span=0.25,
        )
        if not support_geometry.stable:
            return True
        if isinstance(terrain, StairTerrain):
            # A single high foothold is useful evidence, not a reason to stop
            # the body forever.  Freeze only while the pending transition has
            # left fewer than three measured supports or a touchdown is still
            # missing.  This keeps the body moving through a valid three-foot
            # transfer without turning the oscillator into a step state machine.
            if np.any(self._contact_hold) and np.count_nonzero(state.contacts.in_contact) < 3:
                return True
            if np.any(self._height_transition):
                # A correct single-foot touchdown is not yet a new support
                # plane. Keep the body reference frozen until the contact
                # estimator has seen the new level; the oscillator may still
                # complete another geometrically selected swing.
                return True
            return False
        # Flat-ground touchdown capture can proceed while the body follows
        # its command; requiring a full stop for every transient grazing
        # contact causes a valid crawl to deadlock. The phase/contact repair
        # in ``plan`` protects WBC admission locally without turning flat
        # gait into a hidden support state machine.
        return False

    @staticmethod
    def _world_foot_positions(state: RobotState) -> np.ndarray:
        rotation = _rotation_from_rpy(state.base_rpy)
        return state.base_position[None, :] + state.foot_positions_body @ rotation.T

    def _period(self, command: BodyCommand) -> float:
        speed = np.linalg.norm(command.as_array()[:2])
        speed += self.config.yaw_speed_equivalent * abs(command.wz)
        return float(max(self.config.period_min, self.config.period - self.config.period_slope * speed))

    def _next_motion_scale(self, time_s: float, command: BodyCommand) -> float:
        """Predict the next continuous command-ramp value without mutation."""

        now = float(time_s)
        if not np.isfinite(now):
            raise ValueError("time_s must be finite")
        command_active = np.linalg.norm(command.as_array()[:2]) > 0.02 or abs(command.wz) > 0.02
        target_scale = 1.0 if command_active else 0.0
        dt = max(0.0, now - self._last_time) if self._last_time is not None else 0.0
        scale = float(self._motion_scale)
        if dt > 0.0:
            change = min(dt / self.config.command_ramp_time, abs(target_scale - scale))
            scale += float(np.sign(target_scale - scale) * change)
        return float(np.clip(scale, 0.0, 1.0))

    def effective_command(self, state: RobotState, command: BodyCommand) -> BodyCommand:
        """Return the command that the next gait step will actually execute.

        MPC and gait must see the same ramped command. Keeping this preview
        side-effect free lets the composition root query it before MPC while
        ``plan`` remains the sole owner of phase and ramp state mutation.
        """

        scale = self._next_motion_scale(state.time_s, command)
        return BodyCommand(command.vx * scale, command.vy * scale, command.wz * scale)

    @staticmethod
    def _phase_offsets() -> np.ndarray:
        return np.asarray([0.0, 0.5, 0.5, 0.0], dtype=np.float64)

    @staticmethod
    def _body_velocity_for_world_foot_target(
        rotation: np.ndarray,
        base_velocity_body: np.ndarray,
        base_angular_velocity_body: np.ndarray,
        foot_position_body: np.ndarray,
        desired_world_velocity: np.ndarray,
    ) -> np.ndarray:
        """将世界系足端目标速度反解为 WBC 使用的机体系相对速度。

        WBC 会重新加上基座平动和角速度项，因此这里必须补偿完整的
        三轴基座角速度，而不能只补偿 yaw。楼梯上的 roll/pitch 运动
        会在足端产生显著的横向和垂向速度。
        """

        base_velocity_world = rotation @ base_velocity_body
        angular_velocity_world = rotation @ base_angular_velocity_body
        foot_offset_world = rotation @ foot_position_body
        relative_velocity_world = (
            desired_world_velocity
            - base_velocity_world
            - np.cross(angular_velocity_world, foot_offset_world)
        )
        return rotation.T @ relative_velocity_world

    def _contact_weights(self, phases: np.ndarray, moving: bool) -> np.ndarray:
        if not moving:
            return np.ones(4, dtype=np.float64)
        swing_duration = 1.0 - self.config.duty_factor
        blend = min(
            self.config.contact_blend_phase,
            0.49 * swing_duration,
            0.49 * self.config.duty_factor,
        )
        weights = np.zeros(4, dtype=np.float64)
        for leg, phase in enumerate(phases):
            if swing_duration <= phase < swing_duration + blend:
                smooth, _ = _smoothstep((phase - swing_duration) / blend)
                weights[leg] = smooth
            elif swing_duration + blend <= phase <= 1.0 - blend:
                weights[leg] = 1.0
            elif phase > 1.0 - blend:
                smooth, _ = _smoothstep((1.0 - phase) / blend)
                weights[leg] = smooth
        return weights

    def _limit_flat_swings(
        self,
        swing: np.ndarray,
        state: RobotState,
        measured_world_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the generic swing budget without a leg-order bias.

        An unfinished touchdown and a swing already in flight retain their
        slots. New candidates may remove a measured contact only when the
        remaining support set is geometrically stable. Among candidates that
        satisfy that physical admission rule, the least-used leg wins; the
        geometry score resolves equal-use candidates.

        The returned admission mask contains only newly selected scheduled
        swings. The caller records an admission after the final support guard
        confirms that the swing actually starts.
        """

        if self.config.max_swing_legs is None:
            return swing.copy(), np.zeros(4, dtype=bool)

        budget = int(self.config.max_swing_legs)
        held = swing & self._contact_hold
        limited = held.copy()
        remaining = max(0, budget - int(np.count_nonzero(limited)))
        if remaining == 0:
            return limited, np.zeros(4, dtype=bool)

        # A selected leg must not alternate with its diagonal partner inside
        # one phase window merely because its fairness counter just changed.
        continuing = swing & (~held) & self._was_swing
        continuing_legs = np.flatnonzero(continuing)
        if continuing_legs.size > remaining:
            # This cannot arise through the configured budget, but preserving
            # the hard physical cap is safer than publishing an invalid plan
            # if state is restored from an inconsistent external snapshot.
            continuing_legs = continuing_legs[
                np.argsort(
                    self._scheduled_swing_count[continuing_legs],
                    kind="stable",
                )
            ][:remaining]
        limited[continuing_legs] = True
        remaining -= int(continuing_legs.size)

        admitted = np.zeros(4, dtype=bool)
        candidates = swing & (~held) & (~self._was_swing)
        while remaining > 0 and np.any(candidates):
            viable: list[tuple[int, SupportGeometry]] = []
            for leg in np.flatnonzero(candidates):
                candidate_swing = limited.copy()
                candidate_swing[leg] = True
                candidate_support = state.contacts.in_contact & (~candidate_swing)
                geometry = analyze_support_geometry(
                    measured_world_xy,
                    candidate_support,
                    state.base_position[:2],
                    minimum_lateral_span=0.25,
                )
                # Moving an already-airborne leg cannot remove support. A
                # loaded leg, however, is admitted only when its removal
                # leaves a support set the body-progress gate also accepts.
                if not state.contacts.in_contact[leg] or geometry.stable:
                    viable.append((int(leg), geometry))

            if not viable:
                break

            fewest_entries = min(
                int(self._scheduled_swing_count[leg]) for leg, _ in viable
            )
            equally_due = [
                (leg, geometry)
                for leg, geometry in viable
                if int(self._scheduled_swing_count[leg]) == fewest_entries
            ]
            # Exact geometric ties use the canonical leg index. For the two
            # diagonal phase pairs this is mirror-equivariant (FL<->FR and
            # RL<->RR) and is only a deterministic final tie break; it cannot
            # starve a leg because scheduled counts are the primary key.
            selected, _ = max(
                equally_due,
                key=lambda item: (
                    int(item[1].stable),
                    int(item[1].contact_count),
                    float(item[1].margin),
                    float(item[1].area),
                    float(item[1].lateral_span),
                    -float(state.contacts.normal_force[item[0]]),
                    -int(item[0]),
                ),
            )
            limited[selected] = True
            admitted[selected] = True
            candidates[selected] = False
            remaining -= 1

        return limited, admitted

    def plan(
        self,
        state: RobotState,
        command: BodyCommand,
        body_target: BodyTarget,
        terrain: TerrainModel,
        contact_estimator: ContactTerrainEstimator | None = None,
    ) -> FootPlan:
        now = float(state.time_s)
        dt = max(0.0, now - self._last_time) if self._last_time is not None else 0.0
        self._last_time = now
        command_active = np.linalg.norm(command.as_array()[:2]) > 0.02 or abs(command.wz) > 0.02
        target_scale = 1.0 if command_active else 0.0
        if dt > 0.0:
            change = min(dt / self.config.command_ramp_time, abs(target_scale - self._motion_scale))
            self._motion_scale += float(np.sign(target_scale - self._motion_scale) * change)
        self._motion_scale = float(np.clip(self._motion_scale, 0.0, 1.0))
        effective_command = BodyCommand(
            vx=command.vx * self._motion_scale,
            vy=command.vy * self._motion_scale,
            wz=command.wz * self._motion_scale,
        )
        moving = self._motion_scale > 1.0e-4
        started_moving = moving and not self._was_moving
        if started_moving:
            self.phase = self.config.start_phase
            if abs(effective_command.vy) > abs(effective_command.vx):
                self.phase = self.config.lateral_start_phase
        self._was_moving = moving
        # A touchdown that has not produced measured support is incomplete on
        # every terrain. Advancing the oscillator would schedule another
        # physical swing while the previous foot is still being acquired.
        if moving and not np.any(self._contact_hold):
            # The ramp limits stride and body velocity, while the oscillator
            # keeps its physical period. Scaling phase rate by command
            # amplitude stretches the first swing and desynchronizes its
            # trajectory from contact loading.
            phase_rate = 1.0 / self._period(effective_command)
            self.phase = (self.phase + dt * phase_rate) % 1.0
        phases = (self.phase + self._phase_offsets()) % 1.0
        swing_duration = 1.0 - self.config.duty_factor
        raw_swing = phases < swing_duration if moving else np.zeros(4, dtype=bool)

        # 相位候选必须保留至少两只实际承重脚。若当前接触脚恰好被
        # 候选摆腿覆盖，延后其中承重最大的脚；这是实时支撑保护，
        # 不规定固定腿序，也不把一次接触事件编码成状态机。
        recovery_capture = np.zeros(4, dtype=bool)
        if moving:
            support_mask = state.contacts.in_contact & ~raw_swing
            support_count = int(np.count_nonzero(support_mask))
            if support_count < 2:
                required = 2 - support_count
                swing_contacts = np.flatnonzero(
                    raw_swing & state.contacts.in_contact
                )
                if swing_contacts.size:
                    forces = state.contacts.normal_force[swing_contacts]
                    order = swing_contacts[np.argsort(-forces, kind="stable")]
                    raw_swing[order[:required]] = False
                support_mask = state.contacts.in_contact & ~raw_swing
                if np.count_nonzero(support_mask) < 2:
                    # When only two feet are measured, keep both available
                    # supports instead of asking WBC to create a new swing.
                    raw_swing[state.contacts.in_contact] = False
        swing = raw_swing.copy()
        scheduled_admission = np.zeros(4, dtype=bool)
        flat_capture = np.zeros(4, dtype=bool)
        if moving:
            for leg in range(4):
                if self._contact_hold[leg]:
                    # A held leg has already completed its swing but has not
                    # yet established upward support. Keep its touchdown
                    # target until contact is real and the phase is back in
                    # the ordinary stance interval.
                    if state.contacts.in_contact[leg] and not raw_swing[leg]:
                        self._contact_hold[leg] = False
                        swing[leg] = False
                    else:
                        # Keep the completed swing target active until a real
                        # upward support contact arrives. Falling back to a
                        # nominal stance reference here discards the latched
                        # touchdown point and can pull the body sideways while
                        # the phase is waiting.
                        swing[leg] = True
                elif (
                    self._was_swing[leg]
                    and not raw_swing[leg]
                    and not state.contacts.in_contact[leg]
                ):
                    self._contact_hold[leg] = True
                    # The just-finished swing has no measured touchdown yet.
                    # Keep its latched endpoint under the swing task while
                    # the oscillator waits for physical support.
                    swing[leg] = True
            flat_capture = (
                self._was_swing & (~raw_swing) & (~state.contacts.in_contact)
            )
        if isinstance(terrain, StairTerrain):
            # A transition leg that has physically touched down is already a
            # support candidate.  Keep it in stance until the support-level
            # estimator confirms the new pair; otherwise the next oscillator
            # phase can re-swing the same leg and suppress the partner needed
            # to establish that pair.
            swing[self._height_transition & state.contacts.in_contact] = False
        rotation = _rotation_from_rpy(state.base_rpy)
        yaw = float(state.base_rpy[2])
        c, s = np.cos(yaw), np.sin(yaw)
        yaw_rotation = np.asarray([[c, -s], [s, c]], dtype=np.float64)
        # Keep the validated flat path as yaw-only: flat-ground height is
        # measured independently from horizontal foot placement. Stair tread
        # transitions are the one case where body roll/pitch must be included
        # to reconstruct the actual world foot height.
        if isinstance(terrain, StairTerrain):
            measured_world = (
                state.base_position[None, :]
                + state.foot_positions_body @ rotation.T
            )
            measured_world_xy = measured_world[:, :2]
            measured_world_center_z = measured_world[:, 2]
        else:
            measured_world_xy = (
                state.base_position[:2][None, :]
                + state.foot_positions_body[:, :2] @ yaw_rotation.T
            )
            measured_world_center_z = (
                state.base_position[2] + state.foot_positions_body[:, 2]
            )
        if not self._height_initialized:
            self._stance_terrain_height[:] = measured_world_center_z - self.config.foot_radius
            self._swing_start_center_height[:] = measured_world_center_z
            self._swing_target_center_height[:] = measured_world_center_z
            self._swing_start_world_xy[:] = measured_world_xy
            self._swing_target_world_xy[:] = measured_world_xy
            self._height_initialized = True

        support_lock = np.zeros(4, dtype=bool)
        established_support_height: float | None = None
        if (
            moving
            and isinstance(terrain, StairTerrain)
            and contact_estimator is not None
        ):
            established_support_height = contact_estimator.support_height()
            if (
                established_support_height is not None
                and abs(float(body_target.height) - float(state.base_position[2]))
                > self.config.height_transition_threshold
            ):
                measured_terrain = measured_world_center_z - self.config.foot_radius
                support_lock = state.contacts.in_contact & (
                    np.abs(measured_terrain - float(established_support_height))
                    <= self.config.height_transition_threshold
                )

        if not moving:
            self._was_swing.fill(False)
            self._contact_hold.fill(False)
            for leg in range(4):
                nominal = terrain.height_at(
                    float(measured_world_xy[leg, 0]), float(measured_world_xy[leg, 1])
                )
                if contact_estimator is not None:
                    nominal = contact_estimator.corrected_height(leg, nominal)
                self._stance_terrain_height[leg] = nominal
            return FootPlan(
                self.nominal_feet_body.copy(),
                np.zeros((4, 3), dtype=np.float64),
                swing,
                np.ones(4, dtype=bool),
                self._stance_terrain_height.copy(),
                np.ones(4, dtype=np.float64),
                np.zeros(4, dtype=bool),
            )

        if isinstance(terrain, StairTerrain):
            swing[support_lock] = False

        world_anchored = self._height_transition.copy()
        # A held touchdown is a world-frame landing target on flat ground as
        # well as on stairs.  Restricting this to StairTerrain lets the base
        # acceleration feed-forward drag an unconfirmed foot away from the
        # floor during ordinary gait startup.
        world_anchored |= self._contact_hold
        if isinstance(terrain, StairTerrain):
            world_anchored |= support_lock

        period = self._period(effective_command)
        linear_velocity = np.asarray([effective_command.vx, effective_command.vy, 0.0])
        angular_velocity = np.asarray([0.0, 0.0, effective_command.wz])
        ground_velocity = linear_velocity + np.cross(
            np.broadcast_to(angular_velocity, self.nominal_feet_body.shape),
            self.nominal_feet_body,
        )
        strides = ground_velocity[:, :2] * (period * self.config.duty_factor)
        velocity_error = state.base_velocity_body[:2] - linear_velocity[:2]
        yaw_rate_error = float(state.base_angular_velocity_body[2] - effective_command.wz)

        if isinstance(terrain, StairTerrain):
            # Inspect only the swing legs that are starting now.  This keeps
            # the ordinary diagonal gait intact, while allowing a real tread
            # height change to retain the available support immediately.
            active_transition = (
                self._height_transition
                & swing
                & (~state.contacts.in_contact)
            )
            transition_delta = np.zeros(4, dtype=np.float64)
            if np.any(active_transition):
                transition_legs = np.flatnonzero(active_transition)
                transition_delta[transition_legs] = np.abs(
                    self._swing_target_center_height[transition_legs]
                    - (
                        self._stance_terrain_height[transition_legs]
                        + self.config.foot_radius
                    )
                )
                transition_candidates = transition_legs
            else:
                transition_candidates = np.zeros(0, dtype=np.int64)
                for leg in np.flatnonzero(swing & (~self._was_swing)):
                    correction = self.config.foothold_velocity_gain * velocity_error
                    correction += self.config.foothold_yaw_rate_gain * yaw_rate_error * np.asarray(
                        [-self.nominal_feet_body[leg, 1], self.nominal_feet_body[leg, 0]]
                    )
                    correction = np.clip(
                        correction,
                        -np.asarray(self.config.max_foothold_correction),
                        np.asarray(self.config.max_foothold_correction),
                    )
                    touchdown_body_xy = (
                        self.nominal_feet_body[leg, :2]
                        + 0.5 * strides[leg]
                        + correction
                    )
                    touchdown_world_xy = (
                        state.base_position[:2] + yaw_rotation @ touchdown_body_xy
                    )
                    preview = terrain.touchdown_target(
                        measured_world_xy[leg],
                        touchdown_world_xy,
                        yaw_rotation @ linear_velocity[:2],
                        self.config.foot_radius,
                        current_support_height=(
                            established_support_height
                            if established_support_height is not None
                            else self._stance_terrain_height[leg]
                        ),
                    )
                    transition_delta[leg] = abs(
                        float(preview[2]) - self._stance_terrain_height[leg]
                    )
                transition_candidates = np.flatnonzero(
                    transition_delta > self.config.height_transition_threshold
                )

            if transition_candidates.size > 0:
                order = transition_candidates[
                    np.argsort(-transition_delta[transition_candidates])
                ]
                selected = order[: self.config.stair_max_swing_legs]
                limited = np.zeros(4, dtype=bool)
                limited[selected] = True
                swing &= limited

            # Keep ordinary stair motion on the configured diagonal gait.  A
            # real tread-height change is detected when a touchdown target is
            # latched below; only an already-latched transition may narrow
            # the swing set to preserve support during that transfer.
            transition_swing = (
                self._height_transition
                & swing
                & (~state.contacts.in_contact)
            )
            if np.any(transition_swing):
                transition_legs = np.flatnonzero(transition_swing)
                if transition_legs.size > self.config.stair_max_swing_legs:
                    transition_delta = np.abs(
                        self._swing_target_center_height
                        - (self._stance_terrain_height + self.config.foot_radius)
                    )
                    order = transition_legs[np.argsort(-transition_delta[transition_legs])]
                    selected = order[: self.config.stair_max_swing_legs]
                    limited = np.zeros(4, dtype=bool)
                    limited[selected] = True
                    transition_swing = limited
                swing = transition_swing.copy()

            # 楼梯上的并发摆腿上限是整个步态的安全上限，而不只是跃迁腿
            # 的上限。否则一只正在捕获新台阶的腿会与普通摆腿叠加，实际
            # 支撑数会低于机身能够承受的范围。优先保留真实高度跃迁腿，
            # 再按当前相位保留其余摆腿，仍然不规定固定腿序。
            # A completed trajectory awaiting contact is still a physical
            # non-support leg under WBC tracking, so it consumes the same
            # swing budget as a moving trajectory. Preserve acquisitions and
            # admit only the remaining number of new swings.
            held_swing = swing & self._contact_hold
            swing_legs = np.flatnonzero(swing & ~held_swing)
            remaining = max(
                0,
                self.config.stair_max_swing_legs
                - int(np.count_nonzero(held_swing)),
            )
            if swing_legs.size > remaining:
                transition_priority = self._height_transition[swing_legs].astype(int)
                order = swing_legs[
                    np.argsort(-transition_priority, kind="stable")
                ]
                limited = held_swing.copy()
                limited[order[:remaining]] = True
                swing = limited
        elif moving and self.config.max_swing_legs is not None:
            swing, scheduled_admission = self._limit_flat_swings(
                swing,
                state,
                measured_world_xy,
            )

        # Re-check after touchdown capture and stair-specific limiting. A
        # missing stance foot may have been added to ``_contact_hold`` after
        # the first guard, and a held leg must never turn two valid supports
        # into a one-support WBC problem. Contacting swing legs are the only
        # candidates that can be deferred without inventing a new foothold.
        if moving:
            # A phase/contact race can leave two measured feet on one lateral
            # side as the only supports. Repair that local snapshot by
            # retaining a currently contacting swing foot when it produces a
            # stable support set. This is admission protection, not a gait
            # sequence or a terrain state machine.
            support_mask = state.contacts.in_contact & ~swing
            support_geometry = analyze_support_geometry(
                measured_world_xy,
                support_mask,
                state.base_position[:2],
                minimum_lateral_span=0.25,
            )
            if not support_geometry.stable:
                candidates_mask = swing & state.contacts.in_contact
                if isinstance(terrain, StairTerrain):
                    # A held stair touchdown can be a corner/riser contact;
                    # only the support estimator may release it. Flat-ground
                    # capture has no height-transfer ambiguity, so a measured
                    # held contact may be admitted immediately when it fixes
                    # a one-sided support set.
                    candidates_mask &= ~self._contact_hold
                candidates = np.flatnonzero(candidates_mask)
                best_leg: int | None = None
                best_score: tuple[float, float] | None = None
                for leg in candidates:
                    candidate_mask = support_mask.copy()
                    candidate_mask[leg] = True
                    candidate_geometry = analyze_support_geometry(
                        measured_world_xy,
                        candidate_mask,
                        state.base_position[:2],
                        minimum_lateral_span=0.25,
                    )
                    if not candidate_geometry.stable:
                        continue
                    score = (
                        float(candidate_geometry.margin),
                        float(state.contacts.normal_force[leg]),
                    )
                    if best_score is None or score > best_score:
                        best_leg = int(leg)
                        best_score = score
                if best_leg is not None:
                    self._contact_hold[best_leg] = False
                    raw_swing[best_leg] = False
                    swing[best_leg] = False
                else:
                    # No contacting swing leg can repair the set. Promote one
                    # currently airborne stance leg on the opposite lateral
                    # side into a capture swing. This is driven by the
                    # measured support snapshot, not by a prescribed leg
                    # sequence, and is the only way to recover when the
                    # oscillator's next swing is still in the future.
                    missing_stance = (
                        (~state.contacts.in_contact)
                        & (~swing)
                        & (~self._contact_hold)
                    )
                    missing_legs = np.flatnonzero(missing_stance)
                    swing_budget = (
                        self.config.stair_max_swing_legs
                        if isinstance(terrain, StairTerrain)
                        else (
                            self.config.max_swing_legs
                            if self.config.max_swing_legs is not None
                            else 2
                        )
                    )
                    if (
                        missing_legs.size
                        and np.any(support_mask)
                        and np.count_nonzero(swing) < swing_budget
                    ):
                        support_lateral = float(
                            np.mean(measured_world_xy[support_mask, 1])
                        )
                        order = missing_legs[
                            np.argsort(
                                -np.abs(
                                    measured_world_xy[missing_legs, 1]
                                    - support_lateral
                                ),
                                kind="stable",
                            )
                        ]
                        recovery_leg = int(order[0])
                        self._contact_hold[recovery_leg] = True
                        recovery_capture[recovery_leg] = True
                        swing[recovery_leg] = True

            support_mask = state.contacts.in_contact & ~swing
            support_count = int(np.count_nonzero(support_mask))
            if support_count < 2:
                required = 2 - support_count
                swing_contacts = np.flatnonzero(
                    swing & state.contacts.in_contact
                )
                if swing_contacts.size:
                    forces = state.contacts.normal_force[swing_contacts]
                    order = swing_contacts[np.argsort(-forces, kind="stable")]
                    swing[order[:required]] = False
                if np.count_nonzero(state.contacts.in_contact & ~swing) < 2:
                    swing[state.contacts.in_contact] = False

        confirmed_scheduled_entry = scheduled_admission & swing & (~self._was_swing)
        self._scheduled_swing_count[confirmed_scheduled_entry] += 1

        contact_weights = self._contact_weights(phases, moving)
        if moving:
            contact_weights[raw_swing & ~swing] = 1.0
        contact_weights[self._contact_hold] = 1.0

        # A leg retained in stance by the support-protection rule must carry
        # load immediately; the phase-derived blend still applies to the
        # genuinely swinging leg(s).
        positions = self.nominal_feet_body.copy()
        velocities = np.zeros((4, 3), dtype=np.float64)
        terrain_heights = np.zeros(4, dtype=np.float64)
        swing_time = max(period * swing_duration, 1.0e-6)

        for leg, phase in enumerate(phases):
            stride = strides[leg]
            vertical_center = self._stance_terrain_height[leg] + self.config.foot_radius
            vertical_velocity = 0.0
            if swing[leg]:
                progress = float(np.clip(phase / swing_duration, 0.0, 1.0))
                smooth, smooth_derivative = _smoothstep(progress)
                is_recovery_capture = bool(recovery_capture[leg])
                if not self._was_swing[leg] or is_recovery_capture:
                    self._swing_start_center_height[leg] = measured_world_center_z[leg]
                    if is_recovery_capture:
                        # A recovery leg is promoted because the measured
                        # support polygon is already marginal.  Repositioning
                        # it with a normal Raibert target can move it across
                        # the body and make contact acquisition impossible.
                        # Capture vertically at the measured world XY; terrain
                        # still determines the landing height below.
                        touchdown_world_xy = measured_world_xy[leg].copy()
                    else:
                        correction = self.config.foothold_velocity_gain * velocity_error
                        correction += self.config.foothold_yaw_rate_gain * yaw_rate_error * np.asarray(
                            [-self.nominal_feet_body[leg, 1], self.nominal_feet_body[leg, 0]]
                        )
                        correction_limit = np.asarray(self.config.max_foothold_correction)
                        correction = np.clip(correction, -correction_limit, correction_limit)
                        touchdown_body_xy = self.nominal_feet_body[leg, :2] + 0.5 * stride + correction
                        touchdown_world_xy = state.base_position[:2] + yaw_rotation @ touchdown_body_xy
                    touchdown = terrain.touchdown_target(
                        measured_world_xy[leg],
                        touchdown_world_xy,
                        np.zeros(2, dtype=np.float64)
                        if is_recovery_capture
                        else yaw_rotation @ linear_velocity[:2],
                        self.config.foot_radius,
                        current_support_height=(
                            established_support_height
                            if established_support_height is not None
                            else self._stance_terrain_height[leg]
                        ),
                    )
                    self._swing_start_world_xy[leg] = measured_world_xy[leg]
                    self._swing_target_world_xy[leg] = touchdown[:2]
                    self._stance_anchor_valid[leg] = False
                    target_terrain = float(touchdown[2])
                    if contact_estimator is not None:
                        target_terrain = contact_estimator.corrected_height(leg, target_terrain)
                    self._swing_target_center_height[leg] = target_terrain + self.config.foot_radius
                    self._height_transition[leg] = bool(
                        abs(target_terrain - self._stance_terrain_height[leg])
                        > self.config.height_transition_threshold
                    )
                    self._swing_lift_delay[leg] = (
                        self.config.obstacle_lift_delay
                        if self._height_transition[leg]
                        else 0.0
                    )

                world_delta = self._swing_target_world_xy[leg] - self._swing_start_world_xy[leg]
                horizontal_progress = smooth
                horizontal_derivative = smooth_derivative
                lift_delay = float(self._swing_lift_delay[leg])
                if lift_delay > 0.0:
                    horizontal_u = np.clip(
                        (progress - lift_delay) / (1.0 - lift_delay),
                        0.0,
                        1.0,
                    )
                    horizontal_progress, horizontal_derivative = _smoothstep(
                        horizontal_u
                    )
                    horizontal_derivative /= 1.0 - lift_delay
                desired_world_xy = (
                    self._swing_start_world_xy[leg] + horizontal_progress * world_delta
                )
                desired_world_xy_velocity = (
                    horizontal_derivative * world_delta / swing_time
                )
                start_height = self._swing_start_center_height[leg]
                target_height = self._swing_target_center_height[leg]
                apex = max(float(start_height), float(target_height)) + self.config.swing_clearance
                if progress < 0.5:
                    vertical, derivative = _smoothstep(2.0 * progress)
                    vertical_center = start_height + vertical * (apex - start_height)
                    vertical_velocity = 2.0 * derivative * (apex - start_height) / swing_time
                else:
                    vertical, derivative = _smoothstep(2.0 * progress - 1.0)
                    vertical_center = apex + vertical * (target_height - apex)
                    vertical_velocity = 2.0 * derivative * (target_height - apex) / swing_time
                terrain_heights[leg] = target_height - self.config.foot_radius

                positions[leg, :2] = yaw_rotation.T @ (
                    desired_world_xy - state.base_position[:2]
                )
            else:
                if (
                    not self._stance_anchor_valid[leg]
                    and state.contacts.in_contact[leg]
                ):
                    # Initial and reacquired stance contacts must be latched
                    # at their measured world point. Open-loop stride
                    # references can otherwise walk a support foot toward a
                    # stair edge before the next swing starts.
                    self._stance_anchor_world_xy[leg] = measured_world_xy[leg]
                    self._stance_anchor_valid[leg] = True
                if self._was_swing[leg]:
                    self._stance_anchor_world_xy[leg] = self._swing_target_world_xy[leg]
                    self._stance_anchor_valid[leg] = True
                if state.contacts.in_contact[leg] and self._stance_anchor_valid[leg]:
                    # A contact closes the loop at the measured world point.
                    # The planned touchdown is only a reference; retaining it
                    # as an anchor can drag an old-tread support toward an
                    # edge before the new tread is actually established.
                    measured_terrain = (
                        measured_world_center_z[leg] - self.config.foot_radius
                    )
                    expected_terrain = (
                        self._swing_target_center_height[leg]
                        - self.config.foot_radius
                    )
                    self._stance_terrain_height[leg] = measured_terrain
                    support_height = (
                        None
                        if contact_estimator is None
                        else contact_estimator.support_height()
                    )
                    if (
                        support_height is not None
                        and abs(measured_terrain - float(support_height))
                        <= self.config.height_transition_threshold
                    ):
                        # A foot can reach a higher tread before its old
                        # open-loop target is updated.  Once the estimator has
                        # confirmed that tread with a support pair, keep this
                        # physical support point as the local stance target.
                        self._swing_target_center_height[leg] = (
                            measured_terrain + self.config.foot_radius
                        )
                    support_established = (
                        support_height is None
                        or abs(support_height - expected_terrain)
                        <= self.config.height_transition_threshold
                    )
                    if (
                        abs(measured_terrain - expected_terrain)
                        <= self.config.height_transition_threshold
                        and support_established
                    ):
                        self._height_transition[leg] = False
                    elif self._height_transition[leg]:
                        if (
                            abs(measured_terrain - expected_terrain)
                            > self.config.height_transition_threshold
                        ):
                            # A contact on the old tread is not a successful
                            # transition. Make that measured surface
                            # authoritative and let a later swing retry the
                            # geometric target. A correct single-foot contact
                            # remains pending until the support pair agrees.
                            self._swing_target_center_height[leg] = (
                                measured_terrain + self.config.foot_radius
                            )
                            self._height_transition[leg] = False
                stance_progress = float(
                    np.clip((phase - swing_duration) / self.config.duty_factor, 0.0, 1.0)
                )
                if self._stance_anchor_valid[leg]:
                    positions[leg, :2] = yaw_rotation.T @ (
                        self._stance_anchor_world_xy[leg] - state.base_position[:2]
                    )
                    velocities[leg, :2] = -state.base_velocity_body[:2]
                    velocities[leg, :2] -= float(
                        state.base_angular_velocity_body[2]
                    ) * np.asarray([-positions[leg, 1], positions[leg, 0]])
                else:
                    positions[leg, :2] += stride * (0.5 - stance_progress)
                    velocities[leg, :2] = -stride / max(
                        period * self.config.duty_factor, 1.0e-6
                    )
                if self._stance_anchor_valid[leg]:
                    vertical_center = self._swing_target_center_height[leg]
                    terrain_heights[leg] = (
                        self._swing_target_center_height[leg] - self.config.foot_radius
                    )
                else:
                    vertical_center = self._stance_terrain_height[leg] + self.config.foot_radius
                    terrain_heights[leg] = self._stance_terrain_height[leg]

            physical_world_anchor = bool(
                world_anchored[leg] or self._height_transition[leg]
            )
            if physical_world_anchor:
                # ``positions_body`` is consumed by WBC through the complete
                # roll/pitch/yaw rotation. A yaw-only horizontal inverse plus
                # an independently solved Z does not represent a fixed world
                # point once the base tilts. Invert the full rigid transform
                # so the landing target remains exact in all three axes.
                desired_world_position = np.asarray(
                    [
                        *(
                            state.base_position[:2]
                            + yaw_rotation @ positions[leg, :2]
                        ),
                        vertical_center,
                    ],
                    dtype=np.float64,
                )
                positions[leg] = rotation.T @ (
                    desired_world_position - state.base_position
                )
                desired_world_velocity = np.asarray(
                    [0.0, 0.0, vertical_velocity], dtype=np.float64
                )
                if swing[leg]:
                    desired_world_velocity[:2] = desired_world_xy_velocity
                velocities[leg] = self._body_velocity_for_world_foot_target(
                    rotation,
                    state.base_velocity_body,
                    state.base_angular_velocity_body,
                    positions[leg],
                    desired_world_velocity,
                )
            else:
                # Foot positions are expressed relative to the measured base
                # pose consumed by WBC.  Subtracting the desired base height
                # moves an already-supported foot in world Z whenever MPC
                # raises the body for a new tread; keep the terrain target in
                # world coordinates instead.
                positions[leg, 2] = vertical_center - state.base_position[2]
                positions[leg, 2] += (
                    body_target.rpy[1] * positions[leg, 0]
                    - body_target.rpy[0] * positions[leg, 1]
                )
                velocities[leg, 2] = vertical_velocity

            if swing[leg] and not physical_world_anchor:
                desired_world_velocity = np.asarray(
                    [
                        desired_world_xy_velocity[0],
                        desired_world_xy_velocity[1],
                        vertical_velocity,
                    ],
                    dtype=np.float64,
                )
                body_velocity = yaw_rotation.T @ desired_world_xy_velocity
                body_velocity -= state.base_velocity_body[:2]
                body_velocity -= float(state.base_angular_velocity_body[2]) * np.asarray(
                    [-positions[leg, 1], positions[leg, 0]]
                )
                velocities[leg, :2] = body_velocity
                velocities[leg, 2] = vertical_velocity

        # The previous-cycle mask must be published only after every leg has
        # compared against the same snapshot. Updating it inside the loop
        # makes the later diagonal leg miss its swing-entry event and leaves
        # its touchdown target stale.
        self._was_swing[:] = swing
        expected_contacts = contact_weights > 0.02
        world_anchored |= self._height_transition
        published_world_anchored = world_anchored | self._contact_hold | flat_capture
        return FootPlan(
            positions,
            velocities,
            swing,
            expected_contacts,
            terrain_heights,
            contact_weights,
            self._height_transition.copy(),
            published_world_anchored,
            self._contact_hold.copy(),
            self._swing_start_world_xy.copy(),
            self._swing_target_world_xy.copy(),
        )
