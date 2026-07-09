"""Robot import validation (§3 C / §4① — 导入机器人 → 配置 front door).

Before a URDF can become a ConfigSet, the system must answer: is this robot ADAPTABLE by the
framework? This runs the Adapter's derivation and checks the structural prerequisites (12-DoF
quadruped with hip/thigh/calf leg roles, positive effort/velocity limits, derivable mass + leg
length), returning an honest report — adaptable or a concrete list of why not. Pure URDF parse +
derive; no SSH, no GPU. The framework itself is never assumed to fit; this is the gate that says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from autotuner.adapter.derive import derive

_LEG_ROLES = ("hip", "thigh", "calf")


@dataclass
class RobotImportReport:
    urdf: str
    adaptable: bool
    n_joints: int = 0
    mass_kg: float = 0.0
    leg_length_m: float = 0.0
    roles: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    derived_preview: Dict[str, object] = field(default_factory=dict)


def validate_robot_urdf(urdf_path: str) -> RobotImportReport:
    try:
        d = derive(urdf_path)
    except Exception as exc:  # noqa: BLE001 — any parse/derive failure = not importable, reported honestly
        return RobotImportReport(urdf=urdf_path, adaptable=False,
                                 issues=[f"URDF not parseable/derivable: {type(exc).__name__}: {exc}"])

    issues: List[str] = []
    n = d.get("n_actuated_joints", 0)
    if n != 12:
        issues.append(f"expected 12 actuated joints (quadruped × hip/thigh/calf), got {n}")
    eff = d.get("effort_limit", {}) or {}
    for r in _LEG_ROLES:
        if r not in eff:
            issues.append(f"missing leg role '{r}' (URDF joint naming must yield hip/thigh/calf)")
        elif eff[r] <= 0:
            issues.append(f"leg role '{r}' has non-positive effort limit ({eff[r]})")
    vel = d.get("velocity_limit", {}) or {}
    for r in _LEG_ROLES:
        if r in vel and vel[r] <= 0:
            issues.append(f"leg role '{r}' has non-positive velocity limit ({vel[r]})")
    if d.get("mass_kg", 0) <= 0:
        issues.append("total mass not derivable from URDF inertials")
    if d.get("leg_length_m", 0) <= 0:
        issues.append("leg length not derivable from URDF kinematics")

    return RobotImportReport(
        urdf=urdf_path, adaptable=not issues, n_joints=n,
        mass_kg=float(d.get("mass_kg", 0.0)), leg_length_m=float(d.get("leg_length_m", 0.0)),
        roles=sorted(eff.keys()), issues=issues,
        derived_preview={"effort": eff, "Kp": d.get("Kp_per_joint", {}),
                         "base_h_target": d.get("base_h_target_m")},
    )


if __name__ == "__main__":
    import sys
    urdf = sys.argv[1] if len(sys.argv) > 1 else "assets/robots/taili-dog/robot.urdf"
    r = validate_robot_urdf(urdf)
    print(f"{urdf}\n  adaptable={r.adaptable}  joints={r.n_joints}  mass={r.mass_kg:.1f}kg  "
          f"leg={r.leg_length_m:.3f}m  roles={r.roles}")
    for i in r.issues:
        print(f"  ✗ {i}")
    if r.adaptable:
        print(f"  derived: effort={r.derived_preview['effort']} base_h={r.derived_preview['base_h_target']}")
