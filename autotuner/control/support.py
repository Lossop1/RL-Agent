"""Support-set geometry shared by gait admission and body-progress gating."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SupportGeometry:
    """A small, simulator-independent summary of the loaded support set.

    ``margin`` is positive when the projected body point is inside the convex
    support polygon.  For a two-foot support set it is the negative distance
    from the body point to the support segment, while ``lateral_span`` keeps
    a front/rear pair on one side from being treated as a stable diagonal.
    """

    contact_mask: np.ndarray
    contact_count: int
    area: float
    margin: float
    lateral_span: float
    stable: bool

    def __post_init__(self) -> None:
        mask = np.asarray(self.contact_mask, dtype=bool)
        if mask.shape != (4,):
            raise ValueError("contact_mask must have shape (4,)")
        object.__setattr__(self, "contact_mask", mask.copy())
        if self.contact_count != int(np.count_nonzero(mask)):
            raise ValueError("contact_count does not match contact_mask")
        for name in ("area", "margin", "lateral_span"):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Return point indices in counter-clockwise order for a tiny point set."""

    if points.shape[0] <= 1:
        return np.arange(points.shape[0], dtype=np.int64)
    ordered = sorted(
        range(points.shape[0]),
        key=lambda index: (float(points[index, 0]), float(points[index, 1])),
    )

    def cross(origin: int, first: int, second: int) -> float:
        a = points[first] - points[origin]
        b = points[second] - points[origin]
        return float(a[0] * b[1] - a[1] * b[0])

    lower: list[int] = []
    for index in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], index) <= 0.0:
            lower.pop()
        lower.append(index)
    upper: list[int] = []
    for index in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], index) <= 0.0:
            upper.pop()
        upper.append(index)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.int64)


def _segment_margin(point: np.ndarray, segment: np.ndarray) -> float:
    delta = segment[1] - segment[0]
    length = float(np.linalg.norm(delta))
    if length <= 1.0e-9:
        return -float(np.linalg.norm(point - segment[0]))
    projection = float(np.dot(point - segment[0], delta) / (length * length))
    closest = segment[0] + np.clip(projection, 0.0, 1.0) * delta
    return -float(np.linalg.norm(point - closest))


def analyze_support_geometry(
    foot_positions_world_xy: np.ndarray,
    contact_mask: np.ndarray,
    body_position_world_xy: np.ndarray,
    *,
    minimum_lateral_span: float = 0.12,
    minimum_polygon_area: float = 0.008,
    minimum_margin: float = -0.10,
    maximum_segment_distance: float = 0.12,
) -> SupportGeometry:
    """Classify whether measured contacts can safely carry body progress.

    The classifier is deliberately conservative and does not select a leg
    sequence. A two-foot set must span both lateral sides and place the body
    close to the support segment; a three/four-foot set must form a polygon
    with non-trivial area and contain the body projection up to a small edge
    tolerance.
    """

    feet = np.asarray(foot_positions_world_xy, dtype=np.float64)
    mask = np.asarray(contact_mask, dtype=bool)
    body = np.asarray(body_position_world_xy, dtype=np.float64)
    if feet.shape != (4, 2) or not np.isfinite(feet).all():
        raise ValueError("foot_positions_world_xy must be finite with shape (4, 2)")
    if mask.shape != (4,):
        raise ValueError("contact_mask must have shape (4,)")
    if body.shape != (2,) or not np.isfinite(body).all():
        raise ValueError("body_position_world_xy must be finite with shape (2,)")
    indices = np.flatnonzero(mask)
    count = int(indices.size)
    points = feet[indices]
    lateral_span = float(np.ptp(points[:, 1])) if count else 0.0
    if count < 2:
        return SupportGeometry(mask, count, 0.0, -1.0e9, lateral_span, False)
    if count == 2:
        margin = _segment_margin(body, points)
        stable = (
            lateral_span >= float(minimum_lateral_span)
            and margin >= -float(maximum_segment_distance)
        )
        return SupportGeometry(mask, count, 0.0, margin, lateral_span, stable)

    hull = points[_convex_hull(points)]
    area = 0.5 * abs(
        float(
            np.sum(
                hull[:, 0] * np.roll(hull[:, 1], -1)
                - hull[:, 1] * np.roll(hull[:, 0], -1)
            )
        )
    )
    edge_vectors = np.roll(hull, -1, axis=0) - hull
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    cross_values = (
        edge_vectors[:, 0] * (body[1] - hull[:, 1])
        - edge_vectors[:, 1] * (body[0] - hull[:, 0])
    )
    margin = float(np.min(cross_values / np.maximum(edge_lengths, 1.0e-9)))
    stable = (
        area >= float(minimum_polygon_area)
        and lateral_span >= float(minimum_lateral_span)
        and margin >= float(minimum_margin)
    )
    return SupportGeometry(mask, count, area, margin, lateral_span, stable)
