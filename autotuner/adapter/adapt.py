"""产品合同驱动的机器人适配入口。"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from autotuner.adapter.derive import derive
from autotuner.adapter.reward_scale import scale_rewards
from autotuner.product import (
    call_product_plugin,
    resolve_product_adapter,
    resolve_product_contract,
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def adapt(
    urdf_path: str,
    mass: Optional[float] = None,
    regenerate_clips_to: Optional[str] = None,
    *,
    contract: Any | None = None,
    product_id: str | None = None,
) -> dict[str, Any]:
    """把 URDF 与产品适配声明编译为可审计的适配结果。

    未显式传入合同时只允许注册表中存在唯一产品；注册多个产品后调用者必须
    指定 ``product_id``。该规则避免系统悄悄回退到某个机器人型号。
    """
    resolved = contract or resolve_product_contract(product_id)
    adapter = resolve_product_adapter(resolved)
    settings = _mapping(adapter.adaptation, "adaptation")
    actuator_settings = _mapping(settings.get("actuator"), "adaptation.actuator")
    reference_settings = _mapping(
        settings.get("reference_geometry"),
        "adaptation.reference_geometry",
    )
    reward_anchor = _mapping(settings.get("reward_anchor"), "adaptation.reward_anchor")
    health_ratios = _mapping(settings.get("health_ratios"), "adaptation.health_ratios")

    derived = derive(urdf_path, actuator_settings)
    resolved_mass = float(mass) if mass is not None else float(derived["mass_kg"])
    if resolved_mass <= 0:
        raise ValueError("mass must be positive")
    reference = call_product_plugin(
        resolved,
        "adaptation",
        "reference_geometry",
        urdf_path,
        reference_settings,
    )
    reference = _mapping(reference, "adaptation.reference_geometry result")
    leg = _positive(reference.get("total_leg"), "reference_geometry.total_leg")
    rewards = scale_rewards(leg, resolved_mass, reward_anchor)

    clips: list[str] | None = None
    if regenerate_clips_to:
        regenerated = call_product_plugin(
            resolved,
            "adaptation",
            "regenerate_reference",
            urdf_path,
            regenerate_clips_to,
            reference_settings,
            which="all",
        )
        regenerated = _mapping(regenerated, "adaptation.regenerate_reference result")
        raw_clips = regenerated.get("clips")
        if not isinstance(raw_clips, (list, tuple)):
            raise ValueError("adaptation.regenerate_reference must return a clips list")
        clips = [str(path) for path in raw_clips]

    height_target = _positive(
        health_ratios.get("base_height_target"),
        "health_ratios.base_height_target",
    )
    height_band = health_ratios.get("base_height_band")
    swing_band = health_ratios.get("swing_height_band")
    if not isinstance(height_band, (list, tuple)) or len(height_band) != 2:
        raise ValueError("health_ratios.base_height_band must contain two values")
    if not isinstance(swing_band, (list, tuple)) or len(swing_band) != 2:
        raise ValueError("health_ratios.swing_height_band must contain two values")

    return {
        "provenance": {
            "product_id": adapter.product_id,
            "product_contract_digest": adapter.contract_digest,
            "urdf": urdf_path,
            "mass_kg": resolved_mass,
            "leg_length_m": leg,
            "framework_composition": str(settings.get("composition") or ""),
        },
        "actuator": {
            "effort": derived["effort_limit"],
            "velocity": derived["velocity_limit"],
            "Kp": derived["Kp_per_joint"],
            "Kd": derived["Kd_per_joint"],
            "joint_range": derived["joint_range"],
        },
        "dims": {"n_actuated_joints": derived["n_actuated_joints"]},
        "reference_geom": reference.get("generator"),
        "reference_clips": clips,
        "reward_thresholds": rewards,
        "health_band": {
            "base_h_target": round(height_target * leg, 3),
            "base_h_band": [round(float(item) * leg, 3) for item in height_band],
            "swing_healthy_cm": [round(float(item) * leg * 100, 1) for item in swing_band],
            "drag_threshold_cm": round(
                _positive(health_ratios.get("drag_height"), "health_ratios.drag_height")
                * leg
                * 100,
                1,
            ),
        },
        "note": derived["note"],
    }


__all__ = ["adapt"]
