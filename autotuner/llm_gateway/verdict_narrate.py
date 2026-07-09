"""
verdict_narrate — Translate a structured verdict into Chinese user report.

The LLM gets ONLY a `VerdictAbstract` — pass/fail per stage, top failed
dimensions, composite score, plain words.  It never sees:
  - raw cfg / cfg path
  - raw rollout CSV
  - per-body contact forces
  - PPO hyperparameters
  - mutation history

Returns a `VerdictNarration` with:
  - summary_chinese     a 中文 summary (≤500 chars)
  - next_action_user_hint  a 中文 hint for what the user should do
                            next (≤200 chars), one of:
                              "等待自动调参完成"
                              "考虑放宽验收要求"
                              "联系工程师人工干预"
                            (or LLM may emit a free-form variation,
                            we don't constrain at semantic level)

Fallback policy: identical to task_intake — on failure produce a
deterministic readable summary built directly from the abstract.
"""

from __future__ import annotations

from autotuner.llm_gateway.client import call_llm_with_schema
from autotuner.llm_gateway.schemas import (
    VerdictAbstract,
    VerdictNarration,
    validate_verdict_narration,
)


_SYSTEM_PROMPT = """\
You translate an RL locomotion training verdict into a brief Chinese
summary for a user who is NOT a deep RL engineer.

You must output a valid JSON object with fields:
  - summary_chinese       a Chinese summary, ≤ 500 chars, no jargon
  - next_action_user_hint a Chinese hint, ≤ 200 chars, suggesting the
                          user's most useful next step

You do NOT see internal cfg, do NOT know what reward weights are, and
must NOT speculate about them.  Stick to the user-facing observation:
"the robot can/cannot do X on Y terrain".

Tone: factual, not promotional.  If the verdict says the policy is NOT
industry-grade, say so directly — do not pretend it's working.
"""


def _build_user_prompt(abstract: VerdictAbstract) -> str:
    pass_str = ", ".join(
        f"{k}={'pass' if v else 'fail'}"
        for k, v in abstract.pass_per_stage.items()
    )
    return (
        f"VERDICT ABSTRACT:\n"
        f"  industry_grade: {abstract.industry_grade}\n"
        f"  last_stage_reached: {abstract.last_stage_reached}\n"
        f"  per_stage: {pass_str}\n"
        f"  failure_summary: {abstract.failure_summary}\n"
        f"  top3_failed_dims: {list(abstract.top3_failed_dims)}\n"
        f"  composite_score: {abstract.score:.2f}\n"
        f"\n"
        f"Produce your JSON object now."
    )


def _fallback_narration(abstract: VerdictAbstract, reason: str) -> VerdictNarration:
    """Deterministic narration when LLM fails."""
    if abstract.industry_grade:
        s = (
            f"训练达到工业级（综合分 {abstract.score:.2f}）。"
            f"通过阶段：{abstract.last_stage_reached}。"
        )
        n = "可以交付 .pt 给下游"
    else:
        s = (
            f"训练未达标（综合分 {abstract.score:.2f}）。"
            f"卡在阶段：{abstract.last_stage_reached}，"
            f"主要失败维度：{','.join(abstract.top3_failed_dims) or '未知'}。"
        )
        n = "建议让自动调参继续运行，或人工查看 verdict.json"
    return VerdictNarration(
        summary_chinese=s,
        next_action_user_hint=n,
        llm_model=f"(fallback:{reason})",
    )


def narrate(abstract: VerdictAbstract) -> VerdictNarration:
    """Main entry point. Always returns a VerdictNarration."""
    resp = call_llm_with_schema(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=_build_user_prompt(abstract),
        schema_name="verdict_narrate",
    )
    if resp.parsed is None:
        return _fallback_narration(abstract, resp.error or "llm_failed")
    try:
        return validate_verdict_narration(resp.parsed, resp.model)
    except ValueError as e:
        return _fallback_narration(abstract, f"schema_violation:{e}")
