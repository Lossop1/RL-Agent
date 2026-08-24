"""仿真器无关的高频关节阻抗伺服。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import JointCommand


def _joint_vector(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (12,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be finite with shape (12,)")
    return vector.copy()


@dataclass
class HighRateImpedanceServo:
    """在物理步长上跟踪 WBC 给出的常加速度参考。

    WBC 每个控制周期只生成一次参考；伺服器在每个物理子步重新读取关节状态，
    因而 MuJoCo 与 IsaacLab 不会各自实现一套不同的插值和阻抗语义。
    """

    effort_limit: np.ndarray

    def __post_init__(self) -> None:
        self.effort_limit = _joint_vector(self.effort_limit, "effort_limit")
        if np.any(self.effort_limit <= 0.0):
            raise ValueError("effort_limit must be positive")
        self.reset()

    def reset(self) -> None:
        self._command: JointCommand | None = None
        self._initial_position: np.ndarray | None = None
        self._initial_velocity: np.ndarray | None = None
        self._initial_time_s: float | None = None

    @property
    def has_command(self) -> bool:
        return self._command is not None

    def set_command(
        self,
        command: JointCommand,
        position: np.ndarray,
        velocity: np.ndarray,
        time_s: float,
    ) -> None:
        if not isinstance(command, JointCommand):
            raise TypeError("command must be a JointCommand")
        if not np.isfinite(float(time_s)):
            raise ValueError("time_s must be finite")
        self._command = command
        self._initial_position = _joint_vector(position, "position")
        self._initial_velocity = _joint_vector(velocity, "velocity")
        self._initial_time_s = float(time_s)

    def torque(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        time_s: float,
    ) -> np.ndarray:
        if (
            self._command is None
            or self._initial_position is None
            or self._initial_velocity is None
            or self._initial_time_s is None
        ):
            raise RuntimeError("impedance servo has no active command")
        if not np.isfinite(float(time_s)):
            raise ValueError("time_s must be finite")
        q = _joint_vector(position, "position")
        dq = _joint_vector(velocity, "velocity")
        command = self._command
        elapsed = float(
            np.clip(
                float(time_s) - self._initial_time_s,
                0.0,
                command.reference_duration,
            )
        )
        position_reference = (
            self._initial_position
            + elapsed * self._initial_velocity
            + 0.5 * elapsed * elapsed * command.acceleration
        )
        velocity_reference = self._initial_velocity + elapsed * command.acceleration
        raw_torque = (
            command.feedforward_torque
            + command.kp * (position_reference - q)
            + command.kd * (velocity_reference - dq)
        )
        return np.clip(raw_torque, -self.effort_limit, self.effort_limit)
