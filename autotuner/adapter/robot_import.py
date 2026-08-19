"""按产品合同验证机器人 URDF 是否可由当前适配器处理。"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from autotuner.adapter.derive import derive
from autotuner.product import call_product_plugin, resolve_product_adapter, resolve_product_contract


@dataclass
class RobotImportReport:
    urdf: str
    adaptable: bool
    product_id: str = ""
    n_joints: int = 0
    mass_kg: float = 0.0
    leg_length_m: float = 0.0
    roles: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    derived_preview: dict[str, object] = field(default_factory=dict)


def validate_robot_urdf(
    urdf_path: str,
    *,
    contract: Any | None = None,
    product_id: str | None = None,
) -> RobotImportReport:
    """验证 URDF 的自由度、关节角色和执行器限制是否满足产品声明。"""
    try:
        resolved = contract or resolve_product_contract(product_id)
        adapter = resolve_product_adapter(resolved)
        actuator = adapter.adaptation.get("actuator")
        reference_settings = adapter.adaptation.get("reference_geometry")
        if not isinstance(actuator, Mapping):
            raise ValueError("adaptation.actuator must be a mapping")
        if not isinstance(reference_settings, Mapping):
            raise ValueError("adaptation.reference_geometry must be a mapping")
        derived = derive(urdf_path, actuator)
        reference = call_product_plugin(
            resolved,
            "adaptation",
            "reference_geometry",
            urdf_path,
            reference_settings,
        )
        if not isinstance(reference, Mapping):
            raise ValueError("reference_geometry plugin must return a mapping")
    except Exception as exc:  # noqa: BLE001 - 失败原因属于导入报告
        return RobotImportReport(
            urdf=urdf_path,
            adaptable=False,
            product_id=str(product_id or ""),
            issues=[f"URDF not parseable/derivable: {type(exc).__name__}: {exc}"],
        )

    issues: list[str] = []
    expected_dof = int(resolved.robot.get("dof") or 0)
    actual_dof = int(derived.get("n_actuated_joints") or 0)
    if expected_dof and actual_dof != expected_dof:
        issues.append(f"expected {expected_dof} actuated joints, got {actual_dof}")
    effort = derived.get("effort_limit") or {}
    velocity = derived.get("velocity_limit") or {}
    for role in sorted(effort):
        if float(effort[role]) <= 0:
            issues.append(f"role {role!r} has a non-positive effort limit")
        if float(velocity.get(role, 0.0)) <= 0:
            issues.append(f"role {role!r} has a non-positive velocity limit")
    mass = float(derived.get("mass_kg") or 0.0)
    leg_length = float(reference.get("total_leg") or 0.0)
    if mass <= 0:
        issues.append("total mass is not derivable from URDF inertials")
    if leg_length <= 0:
        issues.append("leg length is not derivable from the product reference geometry")

    return RobotImportReport(
        urdf=urdf_path,
        adaptable=not issues,
        product_id=resolved.product_id,
        n_joints=actual_dof,
        mass_kg=mass,
        leg_length_m=leg_length,
        roles=sorted(effort),
        issues=issues,
        derived_preview={
            "effort": effort,
            "Kp": derived.get("Kp_per_joint", {}),
            "Kd": derived.get("Kd_per_joint", {}),
            "leg_length_m": leg_length,
        },
    )


__all__ = ["RobotImportReport", "validate_robot_urdf"]
