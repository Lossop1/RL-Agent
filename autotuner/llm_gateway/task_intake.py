"""
task_intake — Translate user natural-language task into TaskSpec.

THE ONLY decision the LLM makes here: map user words to enums in
`KNOWN_*` from `schemas.py`.  Everything else (cfg values, training
budget, mutation logic) stays in the deterministic core.

Fallback policy
===============

If the LLM returns garbage / fails schema validation after retries,
we DO NOT pretend it produced a valid spec.  Instead `translate()`
returns a TaskSpec with `confidence=0.0` and a populated
`user_clarifications_needed` list asking the user to disambiguate.
The caller (CLI) is expected to surface those questions, not silently
proceed.
"""

from __future__ import annotations

from typing import Optional

from autotuner.llm_gateway.client import LLMResponse, call_llm_with_schema
from autotuner.llm_gateway.schemas import (
    KNOWN_AMBITION_LEVELS,
    KNOWN_ROBOT_IDS,
    KNOWN_TASK_CLASSES,
    KNOWN_TERRAINS,
    TaskSpec,
    validate_task_spec,
)


_SYSTEM_PROMPT = """\
You are a strict natural-language → typed-spec translator for an RL
locomotion autotuner.  Your output is the ONLY signal the autotuner
sees about user intent.  You are NOT a decision-maker; you only
classify the user's request into a fixed schema.

You must output a valid JSON object with fields:
  - robot_id              one of: {robots}
  - task_class            one of: {tasks}
  - terrain               one of: {terrains}
  - ambition              one of: {ambitions}
  - max_command_lin_vel_mps   number in 0.1 .. 3.0
  - user_clarifications_needed  list of natural-language questions
                                (max 4) to ask the user when you
                                couldn't decide a field with high
                                confidence
  - confidence            number 0.0 .. 1.0

If the user's text is too ambiguous to decide a field, choose a
SENSIBLE DEFAULT and add the question to `user_clarifications_needed`,
then lower `confidence`.  Never invent fields, never output values
outside the allowed enums.  Never echo internal autotuner cfg
mechanics back to the user.
"""

_USER_TEMPLATE = """\
USER REQUEST:
\"\"\"{text}\"\"\"

Return your JSON object now.
"""


def _build_system_prompt() -> str:
    return _SYSTEM_PROMPT.format(
        robots=list(KNOWN_ROBOT_IDS),
        tasks=list(KNOWN_TASK_CLASSES),
        terrains=list(KNOWN_TERRAINS),
        ambitions=list(KNOWN_AMBITION_LEVELS),
    )


def _fallback_task_spec(user_text: str, reason: str) -> TaskSpec:
    """Deterministic fallback when LLM fails — always asks user."""
    return TaskSpec(
        robot_id=KNOWN_ROBOT_IDS[0],            # taili_dog_39kg
        task_class=KNOWN_TASK_CLASSES[0],
        terrain="rough",
        ambition=KNOWN_AMBITION_LEVELS[0],     # baseline_walking
        max_command_lin_vel_mps=0.5,
        user_clarifications_needed=(
            f"LLM fallback engaged ({reason}). Please confirm: "
            "robot? terrain? max forward speed?",
        ),
        raw_user_text=user_text,
        llm_model="(fallback)",
        confidence=0.0,
    )


def translate(user_text: str) -> TaskSpec:
    """Main entry point. Always returns a TaskSpec; check `confidence`."""
    if not user_text or len(user_text) > 4000:
        return _fallback_task_spec(user_text, "empty_or_too_long")

    system = _build_system_prompt()
    user = _USER_TEMPLATE.format(text=user_text)
    resp = call_llm_with_schema(
        system_prompt=system,
        user_prompt=user,
        schema_name="task_intake",
    )
    if resp.parsed is None:
        return _fallback_task_spec(user_text, resp.error or "llm_failed")

    try:
        return validate_task_spec(resp.parsed, user_text, resp.model)
    except ValueError as e:
        return _fallback_task_spec(user_text, f"schema_violation:{e}")
