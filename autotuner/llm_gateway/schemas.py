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

from dataclasses import dataclass
from typing import Dict, Mapping


# ── Known enums ─────────────────────────────────────────────────────────

KNOWN_ROBOT_IDS: tuple[str, ...] = (
    "robot",
)
KNOWN_TASK_CLASSES: tuple[str, ...] = (
    "locomotion",
)
KNOWN_TERRAINS: tuple[str, ...] = (
    "flat",
)
KNOWN_AMBITION_LEVELS: tuple[str, ...] = (
    "baseline_walking",          # robot moves and tracks moderately
    "robust_locomotion",         # walks on terrain, recovers from disturbance
    "industry_grade",            # meets the project's measurable robustness gates
)


def _vocabulary_values(value: object, fallback: tuple[str, ...]) -> tuple[str, ...]:
    """规范化合同词表，去重并拒绝空值，避免 LLM schema 出现无效枚举。"""
    if not isinstance(value, (list, tuple)):
        return fallback
    result = tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
    return result or fallback


@dataclass(frozen=True)
class TaskSpecVocabulary:
    """一个产品允许任务接收器使用的最小枚举集合。"""

    robot_ids: tuple[str, ...] = ("robot",)
    task_classes: tuple[str, ...] = ("locomotion",)
    terrains: tuple[str, ...] = ("flat",)
    ambition_levels: tuple[str, ...] = KNOWN_AMBITION_LEVELS

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None) -> "TaskSpecVocabulary":
        data = value if isinstance(value, Mapping) else {}
        return cls(
            robot_ids=_vocabulary_values(data.get("robot_ids"), ("robot",)),
            task_classes=_vocabulary_values(data.get("task_classes"), ("locomotion",)),
            terrains=_vocabulary_values(data.get("terrains"), ("flat",)),
            ambition_levels=_vocabulary_values(data.get("ambition_levels"), KNOWN_AMBITION_LEVELS),
        )


DEFAULT_TASK_SPEC_VOCABULARY = TaskSpecVocabulary()


def task_spec_json_schema(vocabulary: TaskSpecVocabulary | None = None) -> Dict:
    """为指定产品词表生成 JSON schema，不把某个产品写入网关代码。"""
    values = vocabulary or DEFAULT_TASK_SPEC_VOCABULARY
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "robot_id", "task_class", "terrain", "ambition",
            "max_command_lin_vel_mps", "user_clarifications_needed", "confidence",
        ],
        "properties": {
            "robot_id": {"type": "string", "enum": list(values.robot_ids)},
            "task_class": {"type": "string", "enum": list(values.task_classes)},
            "terrain": {"type": "string", "enum": list(values.terrains)},
            "ambition": {"type": "string", "enum": list(values.ambition_levels)},
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


TASK_SPEC_JSON_SCHEMA: Dict = task_spec_json_schema(
    TaskSpecVocabulary(
        robot_ids=KNOWN_ROBOT_IDS,
        task_classes=KNOWN_TASK_CLASSES,
        terrains=KNOWN_TERRAINS,
        ambition_levels=KNOWN_AMBITION_LEVELS,
    )
)


def validate_task_spec(
    d: dict,
    raw_text: str,
    llm_model: str,
    vocabulary: TaskSpecVocabulary | None = None,
) -> TaskSpec:
    """Validate dict against schema, then build typed TaskSpec.

    Raises ValueError on any schema violation.
    """
    values = vocabulary or DEFAULT_TASK_SPEC_VOCABULARY
    schema = task_spec_json_schema(values)
    for k in schema["required"]:
        if k not in d:
            raise ValueError(f"TaskSpec missing field {k!r}")
    rid = d["robot_id"]
    if rid not in values.robot_ids:
        raise ValueError(f"robot_id {rid!r} not in product vocabulary")
    if d["task_class"] not in values.task_classes:
        raise ValueError(f"task_class {d['task_class']!r} not known")
    if d["terrain"] not in values.terrains:
        raise ValueError(f"terrain {d['terrain']!r} not known")
    if d["ambition"] not in values.ambition_levels:
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
