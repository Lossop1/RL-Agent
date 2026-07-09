"""⑥ NL-改框架 deterministic spine (§4⑥, §9 Suggestion, §10 LLM-proposes-deterministic-guards).

The chat-soul (locomotion_console) can PROPOSE framework-content edits in natural language
("腿抬高点" → raise swing clearance). This module is the deterministic guard the architecture
mandates: each editable framework knob has a PHYSICS/GEOMETRY-derived safe band (from the Adapter's
derived health_band / scaling, §8.0), and a proposed value is accepted ONLY if it stays in-band —
never "范围外瞎拍". Returns a typed Suggestion (§9) with the safe_range + rationale + risk, which
the operator/LLM sees before any write (the actual write goes through the gated edit-config path).

Bands are robot-specific because they come from the AdaptedConfig (adapt()/pipeline bundle.adapted).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class EditableField:
    name: str                       # user-facing knob
    rt_key: str                     # key in adapted["reward_thresholds"]
    unit: str
    range_kind: str                 # "health_band" | "factor" | "cap"
    range_arg: object               # health_band key | (lo_factor, hi_factor) | (lo_cap, hi_cap)
    tuple_index: Optional[int] = None   # for range-valued rt fields, which element this knob edits
    rationale: str = ""


# grounded in reward_scale.py keys + adapt() health_band
EDITABLE: Dict[str, EditableField] = {
    "base_height": EditableField(
        "base_height", "stand_height", "m", "health_band", "base_h_band",
        rationale="geometric base-height band derived from leg length (§8.2)"),
    "swing_clearance": EditableField(
        "swing_clearance", "base_clearance", "m", "factor", (0.6, 1.5),
        rationale="length-type, ∝ leg; edits within a calibrated factor band (§8.0)"),
    "rough_clearance_bonus": EditableField(
        "rough_clearance_bonus", "clr_rough_bonus_max", "m", "factor", (0.5, 1.6),
        rationale="rough-terrain swing bonus, ∝ leg; calibrated band"),
    "air_time_min": EditableField(
        "air_time_min", "air_time_min", "s", "factor", (0.6, 1.5),
        rationale="swing-cadence floor, ∝ sqrt(leg); calibrated band"),
    "fwd_speed_max": EditableField(
        "fwd_speed_max", "cmd_fwd_range", "m/s", "factor", (0.5, 1.5), tuple_index=1,
        rationale="forward command ceiling, ∝ leg (kinematic); calibrated band"),
    "torque_limit_frac": EditableField(
        "torque_limit_frac", "torque_limit_frac", "frac", "cap", (0.5, 0.9),
        rationale="dimensionless effort fraction; hard cap 0.9 keeps actuator margin (F2)"),
}


@dataclass
class Suggestion:
    target_field: str
    old: float
    new: float
    safe_range: Tuple[float, float]
    accepted: bool
    risk: str                       # "within_band" | "out_of_band" | "unknown_field" | "missing"
    rationale: str


def _current(adapted: dict, f: EditableField) -> Optional[float]:
    rt = adapted.get("reward_thresholds", {})
    if f.rt_key not in rt:
        return None
    v = rt[f.rt_key]
    if f.tuple_index is not None:
        if not isinstance(v, (tuple, list)) or len(v) <= f.tuple_index:
            return None
        return float(v[f.tuple_index])
    return float(v)


def safe_range(field: str, adapted: dict) -> Optional[Tuple[float, float]]:
    f = EDITABLE.get(field)
    if f is None:
        return None
    if f.range_kind == "health_band":
        band = adapted.get("health_band", {}).get(f.range_arg)
        if isinstance(band, (tuple, list)) and len(band) == 2:
            return (float(band[0]), float(band[1]))
        return None
    if f.range_kind == "cap":
        return (float(f.range_arg[0]), float(f.range_arg[1]))
    if f.range_kind == "factor":
        cur = _current(adapted, f)
        if cur is None:
            return None
        lo, hi = f.range_arg
        return (round(cur * lo, 4), round(cur * hi, 4))
    return None


def validate_edit(field: str, new_value: float, adapted: dict) -> Suggestion:
    """Propose-and-guard a framework-content edit. Accepted only if new_value is within the
    robot's derived safe band for that knob."""
    f = EDITABLE.get(field)
    if f is None:
        return Suggestion(field, 0.0, float(new_value), (0.0, 0.0), False, "unknown_field",
                          f"no editable framework knob named {field!r}; known: {', '.join(EDITABLE)}")
    cur = _current(adapted, f)
    rng = safe_range(field, adapted)
    if cur is None or rng is None:
        return Suggestion(field, cur or 0.0, float(new_value), rng or (0.0, 0.0), False, "missing",
                          f"cannot resolve current value / safe range for {field!r} from this AdaptedConfig")
    lo, hi = rng
    accepted = lo <= float(new_value) <= hi
    return Suggestion(
        target_field=f"reward_thresholds.{f.rt_key}" + (f"[{f.tuple_index}]" if f.tuple_index is not None else ""),
        old=cur, new=float(new_value), safe_range=(lo, hi), accepted=accepted,
        risk="within_band" if accepted else "out_of_band",
        rationale=(f"{f.rationale}; safe band [{lo}, {hi}] {f.unit}"
                   + ("" if accepted else f" — {new_value} is OUTSIDE; reject per §8.0 (no out-of-band guesses)")),
    )


def list_editable() -> Dict[str, str]:
    return {k: f"{v.unit} — {v.rationale}" for k, v in EDITABLE.items()}
