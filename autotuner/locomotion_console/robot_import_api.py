"""Console surface for robot-import validation (§3 C front door).

Lets the UI ask "is this URDF adaptable by the framework?" before assembling a ConfigSet, and lists
the URDFs the workspace already knows about. Read-only (URDF parse + derive; no SSH, no GPU).
"""
from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class RobotImportInfo(BaseModel):
    urdf: str
    available: bool = True
    message: str = ""
    adaptable: bool = False
    n_joints: int = 0
    mass_kg: float = 0.0
    leg_length_m: float = 0.0
    roles: List[str] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)
    derived_preview: Dict[str, object] = Field(default_factory=dict)


class KnownUrdfsInfo(BaseModel):
    urdfs: Dict[str, str] = Field(default_factory=dict)        # label/robot key → urdf path


def known_urdfs() -> KnownUrdfsInfo:
    from autotuner.adapter.__main__ import PRESETS
    return KnownUrdfsInfo(urdfs={robot: p["urdf"] for robot, p in PRESETS.items()})


def build_robot_import(urdf: str) -> RobotImportInfo:
    # Confine the (unauthenticated, GET) urdf path to the known robot-asset roots BEFORE any parse,
    # so this endpoint can't be used as an arbitrary-file parse / existence oracle (?urdf=/etc/passwd).
    from autotuner.adapter._safe_xml import resolve_within_roots
    try:
        safe_path = resolve_within_roots(urdf)
    except Exception:
        # Do not echo the raw path/exception (avoids a filesystem enumeration oracle).
        return RobotImportInfo(urdf=urdf, available=False,
                               message="URDF path is not an allowed robot-asset path")
    try:
        from autotuner.adapter.robot_import import validate_robot_urdf
        r = validate_robot_urdf(safe_path)
        return RobotImportInfo(
            urdf=urdf, available=True, adaptable=r.adaptable, n_joints=r.n_joints,
            mass_kg=r.mass_kg, leg_length_m=r.leg_length_m, roles=list(r.roles),
            issues=list(r.issues), derived_preview=r.derived_preview,
        )
    except Exception as exc:  # noqa: BLE001
        return RobotImportInfo(urdf=urdf, available=False, message=f"URDF not importable: {type(exc).__name__}")
