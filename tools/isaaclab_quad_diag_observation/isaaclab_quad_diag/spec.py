from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .util import load_yaml


@dataclass
class RobotSpec:
    robot_name: str = "unknown_quad"
    joint_order: list[str] = field(default_factory=list)
    foot_body_names: dict[str, str] = field(default_factory=dict)
    leg_order: list[str] = field(default_factory=lambda: ["FL", "FR", "RL", "RR"])
    nominal_stand_height: float | None = None
    mirror: dict[str, Any] = field(default_factory=dict)
    effort_limits: list[float] | None = None

    @classmethod
    def load(cls, path: str | Path | None) -> "RobotSpec":
        if path is None:
            return cls()
        data = load_yaml(path) or {}
        return cls(
            robot_name=data.get("robot_name", data.get("robot", "unknown_quad")),
            joint_order=list(data.get("joint_order", [])),
            foot_body_names=dict(data.get("foot_body_names", {})),
            leg_order=list(data.get("leg_order", ["FL", "FR", "RL", "RR"])),
            nominal_stand_height=data.get("nominal_stand_height"),
            mirror=dict(data.get("mirror", {})),
            effort_limits=data.get("effort_limits"),
        )


def load_robot_spec(path: str | Path | None) -> RobotSpec:
    return RobotSpec.load(path)
