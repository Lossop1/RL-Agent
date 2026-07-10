from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class IsaacLabAdapterConfig:
    task: str
    robot_spec_path: str | None = None
    device: str | None = None


class IsaacLabEnvAdapter:
    """Interface placeholder for Isaac Lab recorders.

    This class intentionally contains no target thresholds. Concrete projects can subclass it
    when their Direct env command API or terrain API differs from the default recorder.
    """

    def __init__(self, env: Any, robot_spec: Any | None = None):
        self.env = env
        self.robot_spec = robot_spec

    def set_command(self, cmd):  # pragma: no cover - Isaac Lab specific
        base = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        if hasattr(base, "commands"):
            base.commands[:] = cmd
        elif hasattr(base, "_cmd_target"):
            base._cmd_target[:] = cmd
        else:
            raise AttributeError("Cannot find commands or _cmd_target on this env. Provide a custom adapter.")
