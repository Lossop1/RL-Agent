"""Taili robot profile loaded from the tracked URDF and control config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any, Mapping

import numpy as np


LEGS = ("FL", "FR", "RL", "RR")
JOINT_NAMES = tuple(
    [f"{leg}_hip_joint" for leg in LEGS]
    + [f"{leg}_thigh_joint" for leg in LEGS]
    + [f"{leg}_calf_joint" for leg in LEGS]
)


def _vector(text: str | None, size: int = 3) -> np.ndarray:
    values = [float(item) for item in str(text or "0 0 0").split()]
    if len(values) != size:
        raise ValueError(f"URDF origin must have {size} values")
    return np.asarray(values, dtype=np.float64)


def _origin(element: ET.Element) -> np.ndarray:
    origin = element.find("origin")
    return _vector(origin.get("xyz") if origin is not None else None)


def _joint_map(root: ET.Element) -> dict[str, ET.Element]:
    return {str(item.get("name")): item for item in root.findall("joint") if item.get("name")}


def _mass(root: ET.Element) -> float:
    total = 0.0
    for link in root.findall("link"):
        inertial = link.find("inertial")
        mass = inertial.find("mass") if inertial is not None else None
        if mass is not None and mass.get("value"):
            total += float(mass.get("value"))
    return total


def _foot_radius(root: ET.Element) -> float:
    link = root.find("link[@name='FL_foot']")
    if link is None:
        raise ValueError("Taili URDF has no FL_foot link")
    for collision in link.findall("collision"):
        sphere = collision.find("geometry/sphere")
        if sphere is not None and sphere.get("radius"):
            return float(sphere.get("radius"))
    raise ValueError("Taili foot collision sphere is missing")


@dataclass(frozen=True)
class TailiProfile:
    """Canonical Taili geometry and actuator limits.

    Arrays follow the established [all hips, all thighs, all calves] order.
    Geometry is extracted from URDF; controller gains and the default pose are
    supplied by the traditional-control configuration, not copied from RL.
    """

    urdf_path: Path
    joint_names: tuple[str, ...]
    q_default: np.ndarray
    q_lower: np.ndarray
    q_upper: np.ndarray
    effort_limit: np.ndarray
    velocity_limit: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    hip_offsets: np.ndarray
    thigh_offsets: np.ndarray
    calf_offsets: np.ndarray
    foot_offsets: np.ndarray
    foot_radius: float
    nominal_base_height: float
    mass_kg: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "q_default", np.asarray(self.q_default, dtype=np.float64).copy())
        for name in ("q_lower", "q_upper", "effort_limit", "velocity_limit", "kp", "kd"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=np.float64).copy())
        for name in ("hip_offsets", "thigh_offsets", "calf_offsets", "foot_offsets"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=np.float64).copy())
        if len(self.joint_names) != 12 or any(getattr(self, name).shape != (12,) for name in ("q_default", "q_lower", "q_upper", "effort_limit", "velocity_limit", "kp", "kd")):
            raise ValueError("Taili joint vectors must have length 12")
        if any(getattr(self, name).shape != (4, 3) for name in ("hip_offsets", "thigh_offsets", "calf_offsets", "foot_offsets")):
            raise ValueError("Taili leg geometry must have shape (4, 3)")
        if np.any(self.q_lower > self.q_upper) or self.foot_radius <= 0.0 or self.nominal_base_height <= 0.0:
            raise ValueError("invalid Taili profile limits or height")


def default_taili_urdf() -> Path:
    return Path(__file__).resolve().parents[3] / "locomotion-console-ui" / "public" / "robot" / "taili_dog_description" / "urdf" / "robot.urdf"


def load_taili_profile(config: Mapping[str, Any], urdf_path: str | Path | None = None) -> TailiProfile:
    path = Path(urdf_path or config.get("robot", {}).get("urdf", default_taili_urdf()))
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / path
    path = path.resolve()
    root = ET.parse(path).getroot()
    joints = _joint_map(root)
    missing = [name for name in JOINT_NAMES if name not in joints]
    if missing:
        raise ValueError(f"Taili URDF is missing joints: {missing}")

    robot_cfg = config.get("robot", {})
    actuator_cfg = config.get("actuator", {})
    q_default = np.asarray(robot_cfg.get("q_default", [0.0] * 4 + [0.7] * 4 + [-1.4] * 4), dtype=np.float64)
    kp_roles = actuator_cfg.get("kp", {"hip": 120.0, "thigh": 120.0, "calf": 120.0})
    kd_roles = actuator_cfg.get("kd", {"hip": 10.0, "thigh": 10.0, "calf": 10.0})
    role_values = lambda values: np.asarray(
        [float(values["hip"])] * 4 + [float(values["thigh"])] * 4 + [float(values["calf"])] * 4,
        dtype=np.float64,
    )
    q_lower, q_upper, effort, velocity = [], [], [], []
    for name in JOINT_NAMES:
        limit = joints[name].find("limit")
        if limit is None:
            raise ValueError(f"joint {name} has no URDF limit")
        q_lower.append(float(limit.get("lower", "-inf")))
        q_upper.append(float(limit.get("upper", "inf")))
        effort.append(float(limit.get("effort", "inf")))
        velocity.append(float(limit.get("velocity", "inf")))

    hip_offsets, thigh_offsets, calf_offsets, foot_offsets = [], [], [], []
    for leg in LEGS:
        hip_offsets.append(_origin(joints[f"{leg}_hip_joint"]))
        thigh_offsets.append(_origin(joints[f"{leg}_thigh_joint"]))
        calf_offsets.append(_origin(joints[f"{leg}_calf_joint"]))
        foot_joint = joints.get(f"{leg}_foot_joint")
        if foot_joint is None:
            raise ValueError(f"Taili URDF is missing {leg}_foot_joint")
        foot_offsets.append(_origin(foot_joint))

    return TailiProfile(
        urdf_path=path,
        joint_names=JOINT_NAMES,
        q_default=q_default,
        q_lower=np.asarray(q_lower, dtype=np.float64),
        q_upper=np.asarray(q_upper, dtype=np.float64),
        effort_limit=np.asarray(effort, dtype=np.float64),
        velocity_limit=np.asarray(velocity, dtype=np.float64),
        kp=role_values(kp_roles),
        kd=role_values(kd_roles),
        hip_offsets=np.asarray(hip_offsets, dtype=np.float64),
        thigh_offsets=np.asarray(thigh_offsets, dtype=np.float64),
        calf_offsets=np.asarray(calf_offsets, dtype=np.float64),
        foot_offsets=np.asarray(foot_offsets, dtype=np.float64),
        foot_radius=_foot_radius(root),
        nominal_base_height=float(robot_cfg.get("nominal_base_height", 0.5471961541)),
        mass_kg=float(robot_cfg.get("mass_kg", _mass(root))),
    )
