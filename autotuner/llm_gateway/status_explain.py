"""
status_explain — Translate progress events into user-facing Chinese text.

Third PR37 frontend entry point.  Like task_intake and verdict_narrate,
the LLM is constrained to a strict schema and gets a deliberately
redacted abstraction of the event — never raw cfg / rollout / hyperparams.

Input  : ProgressAbstract (typed; built from controller events)
Output : ProgressNarration (≤300 char Chinese summary + tone tag)

Fallback policy: identical to siblings — if LLM violates schema or the
gateway is offline, build a deterministic readable line from the abstract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from autotuner.llm_gateway.client import call_llm_with_schema


# ── Schemas ──────────────────────────────────────────────────────────────

KNOWN_PHASE_LABELS = (
    "Phase0_audit",
    "Phase0_invariant_check",
    "Phase1_smoke",
    "Phase1B_smoke_audit",
    "Phase2_confirm",
    "Phase2_fixed_command_eval",
    "Phase3_reward_refine",
    "Phase4_deliver",
)
KNOWN_TONES = ("info", "good", "warn", "fail")


@dataclass(frozen=True)
class ProgressAbstract:
    """Redacted snapshot of one controller progress event."""
    phase: str                       # must be in KNOWN_PHASE_LABELS
    headline: str                    # 1-line summary (≤80 chars)
    metric_summary: Dict[str, float] # ≤5 KV pairs; keys are short labels
    is_terminal: bool                # True if this is the last event of a phase


@dataclass(frozen=True)
class ProgressNarration:
    text_chinese: str                # ≤300 chars
    tone: str                        # must be in KNOWN_TONES
    llm_model: str


PROGRESS_NARRATION_JSON_SCHEMA: Dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["text_chinese", "tone"],
    "properties": {
        "text_chinese": {"type": "string", "maxLength": 300},
        "tone": {"type": "string", "enum": list(KNOWN_TONES)},
    },
}


def validate_progress_narration(d: dict, llm_model: str) -> ProgressNarration:
    for k in PROGRESS_NARRATION_JSON_SCHEMA["required"]:
        if k not in d:
            raise ValueError(f"ProgressNarration missing field {k!r}")
    tone = d["tone"]
    if tone not in KNOWN_TONES:
        raise ValueError(f"tone {tone!r} not in known set")
    text = str(d["text_chinese"])
    if len(text) > 300:
        raise ValueError("text_chinese exceeds 300-char limit")
    return ProgressNarration(
        text_chinese=text, tone=tone, llm_model=llm_model,
    )


# ── Prompt ───────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You translate one phase of an RL locomotion autotuner's progress into
ONE short Chinese sentence (≤300 chars) for a user who is NOT a deep
RL engineer.

Output a JSON object with fields:
  - text_chinese  ≤300 chars; one or two short sentences in Chinese
  - tone          one of: {tones}

Tone selection:
  good  — phase passed
  info  — phase running / partial progress, no failure
  warn  — phase progressing but a metric is concerning
  fail  — phase failed; user may want to act

You must NEVER:
  - mention internal field names like `self.rewards.X`, `cfg_overrides`,
    or PPO hyperparameters
  - propose what the user should do mechanically — that's the system's job
  - invent numbers not in the metric_summary
  - exceed 300 chars (validation rejects)
"""

_USER_TEMPLATE = """\
PHASE: {phase}
HEADLINE: {headline}
METRICS: {metrics}
TERMINAL: {is_terminal}

Return your JSON now.
"""


# ── Fallback ─────────────────────────────────────────────────────────────

_PHASE_LABEL_CN = {
    "Phase0_audit":              "配置审计",
    "Phase0_invariant_check":    "不变量检查",
    "Phase1_smoke":              "短训练验证",
    "Phase1B_smoke_audit":       "短训失败归因",
    "Phase2_confirm":            "确认训练",
    "Phase2_fixed_command_eval": "固定命令评估",
    "Phase3_reward_refine":      "权重微调",
    "Phase4_deliver":            "交付门",
}


def _fallback_narration(abs_: ProgressAbstract) -> ProgressNarration:
    phase_cn = _PHASE_LABEL_CN.get(abs_.phase, abs_.phase)
    tone = "good" if abs_.is_terminal else "info"
    # Build a short readable summary deterministically.
    if abs_.metric_summary:
        metric_bits = "，".join(
            f"{k}={v:.2g}" for k, v in list(abs_.metric_summary.items())[:3]
        )
        text = f"{phase_cn}：{abs_.headline[:60]}（{metric_bits}）"
    else:
        text = f"{phase_cn}：{abs_.headline[:80]}"
    if len(text) > 300:
        text = text[:297] + "..."
    return ProgressNarration(
        text_chinese=text, tone=tone, llm_model="fallback_deterministic",
    )


# ── Public entry point ──────────────────────────────────────────────────

def explain(
    abstract: ProgressAbstract,
    *,
    llm_call=None,                   # injected for tests; None → real client
    llm_model: str = "claude-haiku-4-5-20251001",
) -> ProgressNarration:
    """Translate one ProgressAbstract into a ProgressNarration.

    Always returns a valid ProgressNarration — fallback to deterministic
    rendering when LLM call fails / violates schema.
    """
    if abstract.phase not in KNOWN_PHASE_LABELS:
        raise ValueError(
            f"phase {abstract.phase!r} not in KNOWN_PHASE_LABELS"
        )

    if llm_call is None:
        # LLM path: rely on client.call_llm_with_schema for retries + audit
        try:
            user_prompt = _USER_TEMPLATE.format(
                phase=abstract.phase,
                headline=abstract.headline,
                metrics=dict(abstract.metric_summary),
                is_terminal=abstract.is_terminal,
            )
            system_prompt = _SYSTEM_PROMPT.format(tones=list(KNOWN_TONES))
            resp = call_llm_with_schema(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_schema=PROGRESS_NARRATION_JSON_SCHEMA,
                model=llm_model,
                temperature=0.0,
                max_retries=2,
            )
        except Exception:
            return _fallback_narration(abstract)
    else:
        try:
            resp = llm_call(abstract)
        except Exception:
            return _fallback_narration(abstract)

    if not isinstance(resp, dict):
        return _fallback_narration(abstract)
    try:
        return validate_progress_narration(resp, llm_model)
    except (ValueError, TypeError):
        return _fallback_narration(abstract)
