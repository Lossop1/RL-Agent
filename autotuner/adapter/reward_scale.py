"""Taili reward-threshold scaling helpers.

The current adapter surface is Taili-only. `scale_rewards` keeps the historical
Taili-proven values exactly reproducible when called with the Taili URDF-derived
leg length and mass, while still leaving a small pure function for dry-run analysis.
"""
from __future__ import annotations

import math
from typing import Dict


LEG_PROVEN = 0.682
MASS_PROVEN = 38.98

_PROVEN = dict(
    stand_height=0.52,
    base_clearance=0.07,
    clr_rough_bonus_max=0.06,
    air_time_min=0.08,
    torque_limit_frac=0.85,
    cmd_fwd_range=(0.30, 0.70),
    cmd_back_range=(0.20, 0.45),
    cmd_lat_range=(0.15, 0.35),
    cmd_yaw_range=(0.25, 0.80),
    dr_mass_range_1=(-1.0, 2.0),
    dr_mass_range_2=(-3.0, 6.0),
    dr_mass_range_3=(-3.0, 10.0),
)


def _rng(values: tuple[float, float], scale: float) -> tuple[float, float]:
    return (round(values[0] * scale, 4), round(values[1] * scale, 4))


def scale_rewards(leg: float, mass: float) -> Dict:
    """Return reward thresholds scaled from the Taili-proven anchor."""
    k_leg = leg / LEG_PROVEN
    k_mass = mass / MASS_PROVEN
    k_time = math.sqrt(k_leg)

    return {
        "stand_height": round(_PROVEN["stand_height"] * k_leg, 4),
        "base_clearance": round(_PROVEN["base_clearance"] * k_leg, 4),
        "clr_rough_bonus_max": round(_PROVEN["clr_rough_bonus_max"] * k_leg, 4),
        "air_time_min": round(_PROVEN["air_time_min"] * k_time, 4),
        "torque_limit_frac": _PROVEN["torque_limit_frac"],
        "cmd_fwd_range": _rng(_PROVEN["cmd_fwd_range"], k_leg),
        "cmd_back_range": _rng(_PROVEN["cmd_back_range"], k_leg),
        "cmd_lat_range": _rng(_PROVEN["cmd_lat_range"], k_leg),
        "cmd_yaw_range": _PROVEN["cmd_yaw_range"],
        "dr_mass_range_1": _rng(_PROVEN["dr_mass_range_1"], k_mass),
        "dr_mass_range_2": _rng(_PROVEN["dr_mass_range_2"], k_mass),
        "dr_mass_range_3": _rng(_PROVEN["dr_mass_range_3"], k_mass),
        "_scale": {"leg_k": round(k_leg, 4), "mass_k": round(k_mass, 4), "time_k": round(k_time, 4)},
    }


def scale_gait_balance(mass: float) -> Dict:
    """Return the legacy mass-scaled gait-balance knobs for dry-run display.

    This remains empirical/advisory; it is not part of the current deployment path.
    """
    k_mass = mass / MASS_PROVEN
    return {
        "rew_gait_phase": round(3.0 * k_mass, 2),
        "phase_gate_gait_1": round(max(0.78, 0.85 - 0.04 * (k_mass - 1.0)), 3),
        "phase_gate_slip_1": round(min(0.28, 0.20 + 0.05 * (k_mass - 1.0)), 3),
        "rew_feet_air_time": round(0.25 * k_mass**0.5, 3),
        "_note": f"advisory mass scaling k_mass={round(k_mass, 3)}; validate by training",
    }


if __name__ == "__main__":
    import json

    out = scale_rewards(LEG_PROVEN, MASS_PROVEN)
    ok = (
        out["stand_height"] == 0.56
        and out["base_clearance"] == 0.07
        and out["clr_rough_bonus_max"] == 0.06
        and out["cmd_fwd_range"] == (0.3, 0.7)
        and out["dr_mass_range_3"] == (-3.0, 10.0)
        and out["torque_limit_frac"] == 0.85
    )
    print("Taili identity:", json.dumps(out, ensure_ascii=False))
    print(f"reward scaling identity test: {'PASS' if ok else 'FAIL'}")
