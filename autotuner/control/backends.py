"""Simulator-independent backend contracts.

The control core only sees these protocols. Concrete MuJoCo and IsaacLab
adapters live under the product that owns their asset conventions.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .contracts import JointCommand, RobotState


@runtime_checkable
class StateBackend(Protocol):
    def read_state(self) -> RobotState:
        """Read one synchronized state sample."""


@runtime_checkable
class ActionBackend(Protocol):
    def write_command(self, command: JointCommand) -> None:
        """Apply one command without changing controller semantics."""

    def apply_physics_step(self) -> None:
        """Refresh the low-level impedance torque before one physics step."""


@runtime_checkable
class SimulatorBackend(StateBackend, ActionBackend, Protocol):
    """完整的单环境仿真后端契约。

    ``step_physics`` 必须先刷新高频阻抗力矩，再推进恰好一个物理子步；控制器
    和评测层不允许直接调用某个仿真器的 step API。
    """

    @property
    def physics_dt(self) -> float:
        """Return the duration of one physics step in seconds."""

    def step_physics(self) -> None:
        """Apply the held command and advance exactly one physics step."""
