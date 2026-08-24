"""不导入 IsaacLab 的 Taili 适配边界。

具体任务负责把 IsaacLab 张量转换为单环境的规范状态，并提供一个物理子步
回调；这里统一检查契约，并复用与 MuJoCo 完全相同的高频阻抗伺服。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from autotuner.control.contracts import JointCommand, RobotState
from autotuner.control.servo import HighRateImpedanceServo


JointStateReader = Callable[[], tuple[float, np.ndarray, np.ndarray]]


@dataclass
class IsaacLabStateAdapter:
    read_fn: Callable[[], RobotState]

    def read_state(self) -> RobotState:
        state = self.read_fn()
        if not isinstance(state, RobotState):
            raise TypeError("IsaacLab read_fn must return autotuner.control.RobotState")
        return state


@dataclass
class IsaacLabActionAdapter:
    """把规范 WBC 命令变为每个 IsaacLab 物理子步的关节力矩。"""

    read_joint_state_fn: JointStateReader
    write_torque_fn: Callable[[np.ndarray], None]
    effort_limit: np.ndarray

    def __post_init__(self) -> None:
        self._servo = HighRateImpedanceServo(self.effort_limit)

    def _joint_state(self) -> tuple[float, np.ndarray, np.ndarray]:
        time_s, position, velocity = self.read_joint_state_fn()
        if not np.isfinite(float(time_s)):
            raise ValueError("IsaacLab joint-state time must be finite")
        position_array = np.asarray(position, dtype=np.float64)
        velocity_array = np.asarray(velocity, dtype=np.float64)
        if (
            position_array.shape != (12,)
            or velocity_array.shape != (12,)
            or not np.isfinite(position_array).all()
            or not np.isfinite(velocity_array).all()
        ):
            raise ValueError("IsaacLab joint state must contain two finite 12-vectors")
        return float(time_s), position_array, velocity_array

    def write_command(self, command: JointCommand) -> None:
        time_s, position, velocity = self._joint_state()
        self._servo.set_command(command, position, velocity, time_s)
        self.apply_physics_step()

    def apply_physics_step(self) -> None:
        if not self._servo.has_command:
            return
        time_s, position, velocity = self._joint_state()
        torque = self._servo.torque(position, velocity, time_s)
        self.write_torque_fn(torque.copy())


@dataclass
class IsaacLabBackend:
    """统一的单环境 IsaacLab 后端。

    ``physics_step_fn`` 只推进一次 IsaacLab 世界；执行器力矩在调用前已经由
    ``IsaacLabActionAdapter`` 按当前关节状态刷新。
    """

    read_state_fn: Callable[[], RobotState]
    read_joint_state_fn: JointStateReader
    write_torque_fn: Callable[[np.ndarray], None]
    physics_step_fn: Callable[[], None]
    effort_limit: np.ndarray
    physics_dt_s: float

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.physics_dt_s)) or self.physics_dt_s <= 0.0:
            raise ValueError("physics_dt_s must be positive and finite")
        self.state = IsaacLabStateAdapter(self.read_state_fn)
        self.action = IsaacLabActionAdapter(
            self.read_joint_state_fn,
            self.write_torque_fn,
            self.effort_limit,
        )

    @property
    def physics_dt(self) -> float:
        return float(self.physics_dt_s)

    def read_state(self) -> RobotState:
        return self.state.read_state()

    def write_command(self, command: JointCommand) -> None:
        self.action.write_command(command)

    def apply_physics_step(self) -> None:
        self.action.apply_physics_step()

    def step_physics(self) -> None:
        self.apply_physics_step()
        self.physics_step_fn()
