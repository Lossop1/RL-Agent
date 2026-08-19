"""按产品合同缩放机器人相关的数值阈值。

这里实现的是通用缩放算法，不保存任何机器人型号的经验锚点。锚点由产品
清单的 ``adaptation.reward_anchor`` 提供，并在审计记录中随合同摘要保存。
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


class RewardScaleError(ValueError):
    """奖励缩放锚点缺失或格式不合法。"""


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RewardScaleError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result <= 0:
        raise RewardScaleError(f"{name} must be a positive finite number")
    return result


def _range(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise RewardScaleError(f"{name} must be a two-item list")
    result = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in result) or result[0] > result[1]:
        raise RewardScaleError(f"{name} must be an ordered finite range")
    return result


def _rng(values: tuple[float, float], scale: float) -> tuple[float, float]:
    return (round(values[0] * scale, 4), round(values[1] * scale, 4))


def scale_rewards(leg: float, mass: float, anchor: Mapping[str, Any]) -> dict[str, Any]:
    """根据产品锚点缩放腿长、质量相关的阈值。

    ``anchor`` 需要包含 ``leg_length``、``mass_kg`` 和 ``thresholds``。算法
    只负责尺度变换，不判断某个阈值是否适合训练；后者属于产品验收与研究层。
    """
    if not isinstance(anchor, Mapping):
        raise RewardScaleError("reward anchor must be a mapping")
    leg_ref = _number(anchor.get("leg_length"), "reward_anchor.leg_length")
    mass_ref = _number(anchor.get("mass_kg"), "reward_anchor.mass_kg")
    thresholds = anchor.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise RewardScaleError("reward_anchor.thresholds must be a mapping")

    leg_value = _number(leg, "leg")
    mass_value = _number(mass, "mass")
    k_leg = leg_value / leg_ref
    k_mass = mass_value / mass_ref
    k_time = math.sqrt(k_leg)

    def threshold(name: str) -> Any:
        if name not in thresholds:
            raise RewardScaleError(f"reward_anchor.thresholds missing {name!r}")
        return thresholds[name]

    return {
        "stand_height": round(float(threshold("stand_height")) * k_leg, 4),
        "base_clearance": round(float(threshold("base_clearance")) * k_leg, 4),
        "clr_rough_bonus_max": round(float(threshold("clr_rough_bonus_max")) * k_leg, 4),
        "air_time_min": round(float(threshold("air_time_min")) * k_time, 4),
        "torque_limit_frac": float(threshold("torque_limit_frac")),
        "cmd_fwd_range": _rng(_range(threshold("cmd_fwd_range"), "cmd_fwd_range"), k_leg),
        "cmd_back_range": _rng(_range(threshold("cmd_back_range"), "cmd_back_range"), k_leg),
        "cmd_lat_range": _rng(_range(threshold("cmd_lat_range"), "cmd_lat_range"), k_leg),
        "cmd_yaw_range": _range(threshold("cmd_yaw_range"), "cmd_yaw_range"),
        "dr_mass_range_1": _rng(_range(threshold("dr_mass_range_1"), "dr_mass_range_1"), k_mass),
        "dr_mass_range_2": _rng(_range(threshold("dr_mass_range_2"), "dr_mass_range_2"), k_mass),
        "dr_mass_range_3": _rng(_range(threshold("dr_mass_range_3"), "dr_mass_range_3"), k_mass),
        "_scale": {
            "leg_k": round(k_leg, 4),
            "mass_k": round(k_mass, 4),
            "time_k": round(k_time, 4),
        },
    }


def scale_gait_balance(mass: float, anchor: Mapping[str, Any]) -> dict[str, Any]:
    """返回可选的质量相关步态建议；该函数不直接修改训练配置。"""
    mass_ref = _number(anchor.get("mass_kg"), "reward_anchor.mass_kg")
    k_mass = mass / mass_ref
    return {
        "rew_gait_phase": round(3.0 * k_mass, 2),
        "phase_gate_gait_1": round(max(0.78, 0.85 - 0.04 * (k_mass - 1.0)), 3),
        "phase_gate_slip_1": round(min(0.28, 0.20 + 0.05 * (k_mass - 1.0)), 3),
        "rew_feet_air_time": round(0.25 * k_mass**0.5, 3),
        "_note": f"advisory mass scaling k_mass={round(k_mass, 3)}; validate by training",
    }


__all__ = ["RewardScaleError", "scale_gait_balance", "scale_rewards"]
