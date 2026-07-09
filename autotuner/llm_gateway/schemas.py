"""
Schemas for LLM-bound I/O.

Strict dataclasses + JSON-schema dicts that constrain every LLM
input and output.  The validator rejects any LLM response missing
required fields or supplying disallowed values.

The schemas are deliberately MINIMAL — the LLM doesn't get to invent
new fields, propose cfg values, or output free-form text.  When the
LLM violates the schema we discard the response and use a deterministic
fallback rather than letting the violation propagate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


# ── Known enums ─────────────────────────────────────────────────────────

KNOWN_ROBOT_IDS: tuple[str, ...] = (
    "taili_dog_39kg",
)
KNOWN_TASK_CLASSES: tuple[str, ...] = (
    "velocity_tracking",
)
KNOWN_TERRAINS: tuple[str, ...] = (
    "flat", "rough", "slope", "stairs",
)
KNOWN_AMBITION_LEVELS: tuple[str, ...] = (
    "baseline_walking",          # robot moves and tracks moderately
    "robust_locomotion",         # walks on terrain, recovers from disturbance
    "industry_grade",            # meets the project's measurable robustness gates
)


# ── TaskSpec — output of task_intake ──────────────────────────────────

@dataclass(frozen=True)
class TaskSpec:
    """Result of translating user NL into a structured task."""
    robot_id: str                    # must be in KNOWN_ROBOT_IDS
    task_class: str                  # must be in KNOWN_TASK_CLASSES
    terrain: str                     # must be in KNOWN_TERRAINS
    ambition: str                    # must be in KNOWN_AMBITION_LEVELS
    max_command_lin_vel_mps: float   # 0 < v ≤ 3.0
    user_clarifications_needed: tuple[str, ...]  # questions the LLM still has
    # Audit fields
    raw_user_text: str
    llm_model: str
    confidence: float                # 0.0..1.0


TASK_SPEC_JSON_SCHEMA: Dict = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "robot_id", "task_class", "terrain", "ambition",
        "max_command_lin_vel_mps", "user_clarifications_needed",
        "confidence",
    ],
    "properties": {
        "robot_id":  {"type": "string", "enum": list(KNOWN_ROBOT_IDS)},
        "task_class": {"type": "string", "enum": list(KNOWN_TASK_CLASSES)},
        "terrain":   {"type": "string", "enum": list(KNOWN_TERRAINS)},
        "ambition":  {"type": "string", "enum": list(KNOWN_AMBITION_LEVELS)},
        "max_command_lin_vel_mps": {
            "type": "number", "minimum": 0.1, "maximum": 3.0,
        },
        "user_clarifications_needed": {
            "type": "array",
            "items": {"type": "string", "maxLength": 200},
            "maxItems": 4,
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


def validate_task_spec(d: dict, raw_text: str, llm_model: str) -> TaskSpec:
    """Validate dict against schema, then build typed TaskSpec.

    Raises ValueError on any schema violation.
    """
    for k in TASK_SPEC_JSON_SCHEMA["required"]:
        if k not in d:
            raise ValueError(f"TaskSpec missing field {k!r}")
    rid = d["robot_id"]
    if rid not in KNOWN_ROBOT_IDS:
        raise ValueError(f"robot_id {rid!r} not in known set")
    if d["task_class"] not in KNOWN_TASK_CLASSES:
        raise ValueError(f"task_class {d['task_class']!r} not known")
    if d["terrain"] not in KNOWN_TERRAINS:
        raise ValueError(f"terrain {d['terrain']!r} not known")
    if d["ambition"] not in KNOWN_AMBITION_LEVELS:
        raise ValueError(f"ambition {d['ambition']!r} not known")
    mv = d["max_command_lin_vel_mps"]
    if not (0.1 <= float(mv) <= 3.0):
        raise ValueError(f"max_command_lin_vel_mps out of range: {mv}")
    cl = d.get("user_clarifications_needed", [])
    if len(cl) > 4:
        raise ValueError("too many clarification questions")
    conf = d.get("confidence", 0.0)
    if not (0.0 <= float(conf) <= 1.0):
        raise ValueError(f"confidence out of range: {conf}")
    return TaskSpec(
        robot_id=rid,
        task_class=d["task_class"],
        terrain=d["terrain"],
        ambition=d["ambition"],
        max_command_lin_vel_mps=float(mv),
        user_clarifications_needed=tuple(str(c) for c in cl),
        raw_user_text=raw_text,
        llm_model=llm_model,
        confidence=float(conf),
    )


# ── Verdict abstract — input to verdict_narrate ───────────────────────
#
# The LLM gets ONLY this abstract, never the full verdict.json.

@dataclass(frozen=True)
class VerdictAbstract:
    """Heavily redacted snapshot of the staged_eval result for the LLM."""
    industry_grade: bool
    last_stage_reached: str
    pass_per_stage: Dict[str, bool]      # stage_name → pass?
    failure_summary: str
    top3_failed_dims: tuple[str, ...]    # human-readable dim names
    score: float                          # composite 0..1


VERDICT_NARRATION_JSON_SCHEMA: Dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary_chinese", "next_action_user_hint"],
    "properties": {
        "summary_chinese": {"type": "string", "maxLength": 500},
        "next_action_user_hint": {"type": "string", "maxLength": 200},
    },
}


@dataclass(frozen=True)
class VerdictNarration:
    summary_chinese: str
    next_action_user_hint: str
    llm_model: str


def validate_verdict_narration(d: dict, llm_model: str) -> VerdictNarration:
    for k in VERDICT_NARRATION_JSON_SCHEMA["required"]:
        if k not in d:
            raise ValueError(f"VerdictNarration missing {k!r}")
    s = str(d["summary_chinese"])
    n = str(d["next_action_user_hint"])
    if len(s) > 500 or len(n) > 200:
        raise ValueError("narration text too long")
    return VerdictNarration(
        summary_chinese=s,
        next_action_user_hint=n,
        llm_model=llm_model,
    )
