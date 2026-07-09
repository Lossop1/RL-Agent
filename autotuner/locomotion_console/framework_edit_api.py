"""Console surface for the ⑥ NL-edit safety guard (autotuner.framework_library.framework_edit).

Given a framework knob + a proposed value + a robot, derive the robot's safe band (via the Adapter)
and return a §9 Suggestion: accepted only if in-band. The chat-soul / UI calls this BEFORE proposing
a framework edit, so NL requests ("腿抬高点") are bounded by physics — never "范围外瞎拍" (§8.0/§10).
Read-only (runs adapt() for the derived band; no materialize, no remote, no GPU).
"""
from __future__ import annotations

from typing import List, Tuple

from pydantic import BaseModel, Field


class EditValidationInfo(BaseModel):
    field: str
    available: bool = True
    message: str = ""
    target_field: str = ""
    old: float = 0.0
    new: float = 0.0
    safe_range: Tuple[float, float] = (0.0, 0.0)
    accepted: bool = False
    risk: str = ""
    rationale: str = ""


class EditableKnobsInfo(BaseModel):
    knobs: dict[str, str] = Field(default_factory=dict)


def editable_knobs() -> EditableKnobsInfo:
    from autotuner.framework_library.framework_edit import list_editable
    return EditableKnobsInfo(knobs=list_editable())


def build_edit_validation(field: str, value: float, robot: str = "taili") -> EditValidationInfo:
    try:
        from autotuner.adapter.__main__ import build_config_set
        from autotuner.adapter.adapt import adapt
        from autotuner.framework_library.framework_edit import validate_edit
        cs = build_config_set(robot)
        adapted = adapt(cs.urdf)                     # derived health_band + reward_thresholds (no materialize)
        s = validate_edit(field, float(value), adapted)
        return EditValidationInfo(
            field=field, available=True, target_field=s.target_field, old=s.old, new=s.new,
            safe_range=(s.safe_range[0], s.safe_range[1]), accepted=s.accepted,
            risk=s.risk, rationale=s.rationale,
        )
    except Exception as exc:  # noqa: BLE001
        return EditValidationInfo(field=field, available=False, new=float(value),
                                  risk="error", message=f"{type(exc).__name__}: {exc}")
