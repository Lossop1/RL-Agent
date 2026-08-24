"""Nominal terrain geometry and continuous contact adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .contracts import ContactState


class TerrainModel(Protocol):
    def height_at(self, world_x: float, world_y: float) -> float:
        ...

    def body_height_target(
        self,
        base_x: float,
        command_vx: float,
        nominal_height: float,
        support_height: float | None = None,
    ) -> float:
        ...

    def touchdown_target(
        self,
        current_world_xy: np.ndarray,
        proposed_world_xy: np.ndarray,
        travel_world_xy: np.ndarray,
        foot_radius: float,
        current_support_height: float | None = None,
    ) -> np.ndarray:
        """Return a reachable world-frame ``[x, y, terrain_z]`` foothold."""
        ...


@dataclass(frozen=True)
class FlatTerrain:
    height: float = 0.0

    def height_at(self, world_x: float, world_y: float) -> float:
        del world_x, world_y
        return float(self.height)

    def body_height_target(
        self,
        base_x: float,
        command_vx: float,
        nominal_height: float,
        support_height: float | None = None,
    ) -> float:
        del base_x, command_vx, support_height
        return float(nominal_height + self.height)

    def touchdown_target(
        self,
        current_world_xy: np.ndarray,
        proposed_world_xy: np.ndarray,
        travel_world_xy: np.ndarray,
        foot_radius: float,
        current_support_height: float | None = None,
    ) -> np.ndarray:
        del current_world_xy, travel_world_xy, foot_radius, current_support_height
        proposed = np.asarray(proposed_world_xy, dtype=np.float64)
        if proposed.shape != (2,) or not np.isfinite(proposed).all():
            raise ValueError("proposed_world_xy must be finite with shape (2,)")
        return np.asarray([proposed[0], proposed[1], self.height], dtype=np.float64)


@dataclass(frozen=True)
class StairTerrain:
    """A parameterized staircase aligned with world X.

    ``start_x`` is the first riser. Heights are piecewise constant; the
    controller never embeds a height-specific branch. A negative command
    naturally samples the lower support surface when moving down.
    """

    rise: float
    run: float
    start_x: float = 0.0
    width: float = 2.0
    base_height: float = 0.0
    max_steps: int = 100

    def __post_init__(self) -> None:
        if self.rise <= 0.0 or self.run <= 0.0 or self.width <= 0.0:
            raise ValueError("stair rise, run, and width must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")

    def height_at(self, world_x: float, world_y: float) -> float:
        if abs(float(world_y)) > self.width * 0.5:
            return float(self.base_height)
        distance = float(world_x) - self.start_x
        if distance < 0.0:
            return float(self.base_height)
        # ``start_x`` is the face of the first riser. The first tread in the
        # generated scene therefore already has one rise at this boundary.
        index = int(np.floor(distance / self.run)) + 1
        index = int(np.clip(index, 1, self.max_steps))
        return float(self.base_height + index * self.rise)

    def body_height_target(
        self,
        base_x: float,
        command_vx: float,
        nominal_height: float,
        support_height: float | None = None,
    ) -> float:
        """Keep the body above the support that has physically been established.

        Base-position lookahead raises the body before a foot has landed on the
        next tread. That creates an infeasible request precisely when support is
        weakest. Contact height is therefore authoritative; scene geometry is a
        deterministic fallback for controllers without a contact estimator.
        """

        del command_vx
        if support_height is not None:
            support = float(support_height)
            if not np.isfinite(support):
                raise ValueError("support_height must be finite when provided")
        else:
            support = self.height_at(base_x, 0.0)
        return float(nominal_height + support)

    def touchdown_target(
        self,
        current_world_xy: np.ndarray,
        proposed_world_xy: np.ndarray,
        travel_world_xy: np.ndarray,
        foot_radius: float,
        current_support_height: float | None = None,
    ) -> np.ndarray:
        """Project an approaching foothold past the next stair edge.

        The projection is purely geometric. It does not prescribe a leg
        sequence or a stair phase; it only prevents a spherical foot whose
        nominal touchdown is immediately before an edge from being driven
        through the vertical riser during the following stance.
        """

        current = np.asarray(current_world_xy, dtype=np.float64)
        proposed = np.asarray(proposed_world_xy, dtype=np.float64).copy()
        travel = np.asarray(travel_world_xy, dtype=np.float64)
        if any(value.shape != (2,) for value in (current, proposed, travel)):
            raise ValueError("touchdown vectors must have shape (2,)")
        if not all(np.isfinite(value).all() for value in (current, proposed, travel)):
            raise ValueError("touchdown vectors must be finite")
        radius = float(foot_radius)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("foot_radius must be positive and finite")
        if abs(float(proposed[1])) > self.width * 0.5 or abs(float(travel[0])) <= 1.0e-6:
            return np.asarray(
                [proposed[0], proposed[1], self.height_at(proposed[0], proposed[1])],
                dtype=np.float64,
            )

        landing_margin = min(1.25 * radius, 0.25 * self.run)
        if current_support_height is None:
            support_height = self.height_at(float(current[0]), float(current[1]))
        else:
            support_height = float(current_support_height)
            if not np.isfinite(support_height):
                raise ValueError("current_support_height must be finite when provided")
        support_index = int(
            np.clip(
                np.rint((support_height - self.base_height) / self.rise),
                0,
                self.max_steps,
            )
        )
        proposed_index = int(
            np.clip(
                np.rint(
                    (self.height_at(float(proposed[0]), float(proposed[1])) - self.base_height)
                    / self.rise
                ),
                0,
                self.max_steps,
            )
        )
        if travel[0] > 1.0e-6:
            # A foothold may advance only one tread beyond the currently
            # established support.  The proposed body-relative target can
            # jump across several risers while the body is still slow; using
            # it directly would ask the leg to skip the very support needed
            # to make that transfer safe.
            next_boundary = self.start_x + support_index * self.run
            predicted_x = float(current[0] + travel[0])
            boundary_reached = predicted_x >= next_boundary - landing_margin
            target_index = min(proposed_index, support_index + 1)
            if proposed_index <= support_index and boundary_reached:
                target_index = support_index + 1
            target_index = max(target_index, support_index)
        elif travel[0] < -1.0e-6:
            # Apply the same one-tread limit while descending.  A foot that
            # has already established a lower tread must not be sent back
            # upward merely because the body-relative proposal lags behind.
            previous_boundary = self.start_x + (support_index - 1) * self.run
            predicted_x = float(current[0] + travel[0])
            boundary_reached = (
                support_index > 0
                and predicted_x <= previous_boundary + landing_margin
            )
            target_index = max(proposed_index, support_index - 1)
            if proposed_index >= support_index and boundary_reached:
                target_index = support_index - 1
            target_index = min(target_index, support_index)
        else:
            target_index = proposed_index

        if target_index == 0:
            tread_start = -np.inf
            tread_end = self.start_x
        else:
            tread_start = self.start_x + (target_index - 1) * self.run
            tread_end = self.start_x + target_index * self.run
        if np.isfinite(tread_start):
            lower_bound = tread_start + landing_margin
        else:
            lower_bound = -np.inf
        upper_bound = tread_end - landing_margin
        proposed[0] = float(np.clip(proposed[0], lower_bound, upper_bound))
        target_height = self.base_height + target_index * self.rise
        return np.asarray([proposed[0], proposed[1], target_height], dtype=np.float64)


class ContactTerrainEstimator:
    """Low-pass support-height estimate from contact timing and foot pose."""

    def __init__(
        self,
        alpha: float = 0.25,
        force_threshold: float = 5.0,
        max_correction: float = 0.025,
        minimum_support_contacts: int = 2,
        support_level_tolerance: float = 0.04,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if force_threshold < 0.0:
            raise ValueError("force_threshold must be non-negative")
        if max_correction < 0.0:
            raise ValueError("max_correction must be non-negative")
        if minimum_support_contacts < 1 or minimum_support_contacts > 4:
            raise ValueError("minimum_support_contacts must be in [1, 4]")
        if support_level_tolerance < 0.0:
            raise ValueError("support_level_tolerance must be non-negative")
        self.alpha = float(alpha)
        self.force_threshold = float(force_threshold)
        self.max_correction = float(max_correction)
        self.minimum_support_contacts = int(minimum_support_contacts)
        self.support_level_tolerance = float(support_level_tolerance)
        self._height = np.zeros(4, dtype=np.float64)
        self._initialized = np.zeros(4, dtype=bool)
        self._support_height = 0.0
        self._support_initialized = False
        self._support_pitch = 0.0
        self._support_pitch_initialized = False

    def reset(self) -> None:
        self._height.fill(0.0)
        self._initialized.fill(False)
        self._support_height = 0.0
        self._support_initialized = False
        self._support_pitch = 0.0
        self._support_pitch_initialized = False

    def update(
        self,
        contacts: ContactState,
        foot_world_z: np.ndarray,
        foot_radius: float,
        support_mask: np.ndarray | None = None,
        foot_world_xy: np.ndarray | None = None,
        reference_x: float | None = None,
        support_direction: float = 0.0,
    ) -> np.ndarray:
        z = np.asarray(foot_world_z, dtype=np.float64)
        if z.shape != (4,) or not np.isfinite(z).all():
            raise ValueError("foot_world_z must be finite with shape (4,)")
        intended = (
            np.ones(4, dtype=bool)
            if support_mask is None
            else np.asarray(support_mask, dtype=bool)
        )
        if intended.shape != (4,):
            raise ValueError("support_mask must have shape (4,)")
        if foot_world_xy is not None:
            xy = np.asarray(foot_world_xy, dtype=np.float64)
            if xy.shape != (4, 2) or not np.isfinite(xy).all():
                raise ValueError("foot_world_xy must be finite with shape (4, 2)")
        else:
            xy = None
        if reference_x is not None and not np.isfinite(float(reference_x)):
            raise ValueError("reference_x must be finite when provided")
        if not np.isfinite(float(support_direction)):
            raise ValueError("support_direction must be finite")
        active = (
            contacts.in_contact
            & (contacts.normal_force >= self.force_threshold)
            & intended
        )
        measured = z - float(foot_radius)
        for index in np.flatnonzero(active):
            if not self._initialized[index]:
                self._height[index] = measured[index]
                self._initialized[index] = True
            else:
                self._height[index] = (1.0 - self.alpha) * self._height[index] + self.alpha * measured[index]
        # Adjacent stair contacts are a transfer, not a support plane. A
        # directional transition is accepted only after a complete support
        # pair has been measured on the new level. One new foot is useful
        # foothold evidence, but it is not enough to move the whole-body
        # height target; doing so makes the remaining old support unload and
        # leaves the estimator stuck at an artificial intermediate height.
        measured_support: float | None = None
        if np.count_nonzero(active) >= self.minimum_support_contacts:
            active_measured = measured[active]
            if (
                active_measured.size > 0
                and float(np.max(active_measured) - np.min(active_measured))
                <= self.support_level_tolerance
            ):
                measured_support = float(np.mean(active_measured))
            elif self._support_initialized and abs(float(support_direction)) > 1.0e-6:
                current = float(self._support_height)
                candidate = (
                    float(np.max(active_measured))
                    if support_direction > 0.0
                    else float(np.min(active_measured))
                )
                candidate_members = np.abs(active_measured - candidate) <= self.support_level_tolerance
                if (
                    np.count_nonzero(candidate_members) >= self.minimum_support_contacts
                    and (
                        candidate > current + self.support_level_tolerance
                        if support_direction > 0.0
                        else candidate < current - self.support_level_tolerance
                    )
                ):
                    measured_support = float(np.mean(active_measured[candidate_members]))
        if measured_support is not None:
            if not self._support_initialized:
                self._support_height = measured_support
                self._support_initialized = True
            elif abs(measured_support - self._support_height) > self.support_level_tolerance:
                # A complete new support pair is a physical event. Move the
                # reference directly to that measured level; WBC still limits
                # the resulting body acceleration and therefore smooths the
                # actual motion.
                self._support_height = measured_support
            else:
                self._support_height = (
                    (1.0 - self.alpha) * self._support_height
                    + self.alpha * measured_support
                )
        # Pitch is a geometric diagnostic of the currently loaded contacts,
        # not the global body-height authority. It remains available for a
        # staggered pair even when no single height cluster is yet large
        # enough to support the body target.
        if np.count_nonzero(active) >= self.minimum_support_contacts and xy is not None and reference_x is not None:
                support_indices_for_pitch = np.flatnonzero(active)
                support_x = xy[support_indices_for_pitch, 0]
                centered_x = support_x - float(np.mean(support_x))
                denominator = float(np.dot(centered_x, centered_x))
                if denominator > 0.04:
                    support_z = measured[support_indices_for_pitch]
                    slope = float(
                        np.dot(
                            centered_x,
                            support_z - float(np.mean(support_z)),
                        )
                        / denominator
                    )
                    target_pitch = float(np.clip(-np.arctan(slope), -0.45, 0.45))
                    if abs(slope) > 0.05:
                        if not self._support_pitch_initialized:
                            self._support_pitch = target_pitch
                            self._support_pitch_initialized = True
                        else:
                            self._support_pitch = (
                                (1.0 - self.alpha) * self._support_pitch
                                + self.alpha * target_pitch
                            )
        return self.estimate()

    def estimate(self) -> np.ndarray:
        return self._height.copy()

    def support_height(self) -> float | None:
        """Return the filtered force-weighted height of measured support."""

        if not self._support_initialized:
            return None
        return float(self._support_height)

    def support_pitch(self) -> float:
        """Return the filtered pitch of the measured longitudinal support."""

        return float(self._support_pitch) if self._support_pitch_initialized else 0.0

    def corrected_height(self, leg: int, nominal_height: float) -> float:
        """Return a measured correction only when it agrees with the model.

        On known nominal terrain, a contact on the previous stair tread must
        not replace the planned touchdown height for the next tread. Small
        residuals still compensate collision-radius and model discrepancies.
        """

        if leg < 0 or leg >= 4:
            raise IndexError("leg must be in [0, 3]")
        nominal = float(nominal_height)
        if not np.isfinite(nominal):
            raise ValueError("nominal_height must be finite")
        if not self._initialized[leg]:
            return nominal
        residual = float(self._height[leg] - nominal)
        if abs(residual) > self.max_correction:
            return nominal
        return nominal + residual
