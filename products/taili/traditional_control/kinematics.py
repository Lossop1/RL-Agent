"""Taili four-leg forward kinematics and numerical foot Jacobians."""

from __future__ import annotations

import numpy as np

from .profile import LEGS, TailiProfile


def _rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


class TailiKinematics:
    """Kinematics derived from the same URDF used by the product."""

    def __init__(self, profile: TailiProfile, finite_difference_step: float = 1.0e-6) -> None:
        self.profile = profile
        self.finite_difference_step = float(finite_difference_step)
        if self.finite_difference_step <= 0.0:
            raise ValueError("finite_difference_step must be positive")

    def foot_positions(self, q: np.ndarray) -> np.ndarray:
        q_array = np.asarray(q, dtype=np.float64)
        if q_array.shape != (12,) or not np.isfinite(q_array).all():
            raise ValueError("q must be finite with shape (12,)")
        result = []
        for leg_index, _leg in enumerate(LEGS):
            hip = q_array[leg_index]
            thigh = q_array[4 + leg_index]
            calf = q_array[8 + leg_index]
            hip_rotation = _rot_x(hip)
            thigh_rotation = _rot_y(thigh)
            calf_rotation = _rot_y(calf)
            thigh_frame = hip_rotation @ thigh_rotation
            calf_frame = thigh_frame @ calf_rotation
            result.append(
                self.profile.hip_offsets[leg_index]
                + hip_rotation @ self.profile.thigh_offsets[leg_index]
                + thigh_frame @ self.profile.calf_offsets[leg_index]
                + calf_frame @ self.profile.foot_offsets[leg_index]
            )
        return np.asarray(result, dtype=np.float64)

    def foot_jacobians(self, q: np.ndarray) -> np.ndarray:
        q_array = np.asarray(q, dtype=np.float64)
        base = self.foot_positions(q_array)
        jacobians = np.zeros((4, 3, 12), dtype=np.float64)
        step = self.finite_difference_step
        for index in range(12):
            plus = q_array.copy()
            minus = q_array.copy()
            plus[index] += step
            minus[index] -= step
            jacobians[:, :, index] = (self.foot_positions(plus) - self.foot_positions(minus)) / (2.0 * step)
        del base
        return jacobians

    def nominal_foot_positions(self) -> np.ndarray:
        return self.foot_positions(self.profile.q_default)
