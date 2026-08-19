"""从 URDF 派生执行器与质量信息。

本模块不假设机器人名称、自由度或关节命名。产品通过 ``joint_roles`` 正则
声明需要聚合的关节角色；因此同一算法可用于不同型号，无法匹配时会明确失败。
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from autotuner.adapter._safe_xml import safe_parse_urdf


class DerivationError(ValueError):
    """URDF 无法按产品声明派生时抛出。"""


def _link_inertias(root) -> dict[str, float]:
    values: dict[str, float] = {}
    for link in root.iter("link"):
        name = str(link.attrib.get("name") or "")
        inertia = link.find("./inertial/inertia")
        if name and inertia is not None:
            values[name] = float(inertia.attrib.get("iyy", 0.0))
    return values


def _total_mass(root) -> float:
    total = 0.0
    for mass in root.findall("./link/inertial/mass"):
        total += float(mass.attrib.get("value", 0.0))
    return total


def _compile_roles(settings: Mapping[str, Any]) -> dict[str, re.Pattern[str]]:
    raw = settings.get("joint_roles")
    if not isinstance(raw, Mapping) or not raw:
        raise DerivationError("adaptation.actuator.joint_roles must be a non-empty mapping")
    result: dict[str, re.Pattern[str]] = {}
    for role, pattern in raw.items():
        try:
            result[str(role)] = re.compile(str(pattern))
        except re.error as exc:
            raise DerivationError(f"invalid joint role regex {role!r}: {exc}") from exc
    return result


def derive(urdf_path: str, settings: Mapping[str, Any]) -> dict[str, Any]:
    """按产品声明从 URDF 派生执行器限制、PD 建议、质量和维度。"""
    if not isinstance(settings, Mapping):
        raise DerivationError("adaptation.actuator must be a mapping")
    roles = _compile_roles(settings)
    kp_per_effort = float(settings.get("kp_per_effort", 0.0))
    inertia_floor = float(settings.get("damping_inertia_floor", 0.05))
    if kp_per_effort <= 0 or inertia_floor <= 0:
        raise DerivationError("kp_per_effort and damping_inertia_floor must be positive")

    root = safe_parse_urdf(urdf_path)
    inertias = _link_inertias(root)
    effort: dict[str, float] = {}
    velocity: dict[str, float] = {}
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    role_inertia: dict[str, list[float]] = {role: [] for role in roles}
    actuated = 0

    for joint in root.iter("joint"):
        if joint.attrib.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        actuated += 1
        name = str(joint.attrib.get("name") or "")
        child = joint.find("child")
        child_name = str(child.attrib.get("link") or "") if child is not None else ""
        matched = [role for role, pattern in roles.items() if pattern.fullmatch(name)]
        if len(matched) > 1:
            raise DerivationError(f"joint {name!r} matches multiple roles: {matched}")
        if not matched:
            continue
        role = matched[0]
        values = {
            "effort": float(limit.attrib.get("effort", 0.0)),
            "velocity": float(limit.attrib.get("velocity", 0.0)),
            "lower": float(limit.attrib.get("lower", 0.0)),
            "upper": float(limit.attrib.get("upper", 0.0)),
        }
        for field in ("effort", "velocity"):
            if values[field] <= 0:
                raise DerivationError(f"joint {name!r} has non-positive {field} limit")
        if role in effort:
            if not math.isclose(effort[role], values["effort"], rel_tol=1e-6):
                raise DerivationError(f"role {role!r} has inconsistent effort limits")
            if not math.isclose(velocity[role], values["velocity"], rel_tol=1e-6):
                raise DerivationError(f"role {role!r} has inconsistent velocity limits")
            lower[role] = min(lower[role], values["lower"])
            upper[role] = max(upper[role], values["upper"])
        else:
            effort[role] = values["effort"]
            velocity[role] = values["velocity"]
            lower[role] = values["lower"]
            upper[role] = values["upper"]
        role_inertia[role].append(inertias.get(child_name, 0.0))

    missing = sorted(set(roles) - set(effort))
    if missing:
        raise DerivationError("URDF is missing declared joint roles: " + ", ".join(missing))
    mass = _total_mass(root)
    if mass <= 0 or actuated <= 0:
        raise DerivationError("URDF must declare positive link masses and actuated joints")

    kp = {role: round(kp_per_effort * value) for role, value in effort.items()}
    kd = {
        role: round(
            2.0 * math.sqrt(kp[role] * max(sum(values) / max(len(values), 1), inertia_floor)),
            1,
        )
        for role, values in role_inertia.items()
    }
    return {
        "mass_kg": mass,
        "n_actuated_joints": actuated,
        "effort_limit": effort,
        "velocity_limit": velocity,
        "joint_range": {role: [lower[role], upper[role]] for role in effort},
        "Kp_per_joint": kp,
        "Kd_per_joint": kd,
        "note": "执行器限制和质量来自 URDF；PD 仅为按产品锚点派生的审计建议。",
    }


__all__ = ["DerivationError", "derive"]
