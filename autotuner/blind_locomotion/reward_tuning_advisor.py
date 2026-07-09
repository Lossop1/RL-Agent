"""Reward-tuning advisor (⑨ advisory) — physeval verdict → env-var weight prescription.

Closes the iteration loop's last manual step: given the §2 verdict + the auto-classified deployment
behavior (from acceptance_aggregate), prescribe the specific `TAILI_RW_<FIELD>=val` adjustments for
the next training run. Encodes the tuning knowledge derived from the v1→v2→v3 journey:
  STANDS (vx~0 for move cmds)   → tracking reward too weak vs standing's safety → raise w_track_far
  CREEPS (~const speed)         → tracking gradient-starved → raise w_track_far (far-field)
  A2 yaw fails                  → raise w_yaw_far
  B3 over-lift                  → raise w_clearance_over (currently barely penalizes over-clearance)
  PARTIAL (tracks, imprecise)   → more training, or small w_track_far bump

This is ADVISORY (§10: LLM/heuristic proposes; human applies). The actual relaunch stays a
deliberate, gated step. Pure logic; no SSH/GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TuningSuggestion:
    env_var: str
    current: float
    suggested: float
    rationale: str


# strategy-book reward defaults (the knobs we tune)
_DEFAULTS = {"w_track_far": 0.6, "w_yaw_far": 0.3, "w_clearance_over": 0.1, "w_stand_far": 0.6}


def suggest_tuning(verdict: dict, current: Optional[Dict[str, float]] = None) -> List[TuningSuggestion]:
    """verdict = acceptance_aggregate output (has `failed`, `runs[i].behavior`). Returns env-var
    prescriptions for the next run; empty if the verdict passed or nothing is diagnosable."""
    cur = {**_DEFAULTS, **(current or {})}
    out: List[TuningSuggestion] = []
    if verdict.get("passed"):
        return out
    behavior = ""
    runs = verdict.get("runs") or []
    if runs:
        behavior = runs[0].get("behavior", "")
    failed = set(verdict.get("failed", []))

    # ── deployment behavior drives the primary tracking weight ──
    if "STANDS" in behavior:
        out.append(TuningSuggestion("TAILI_RW_W_TRACK_FAR", cur["w_track_far"], round(cur["w_track_far"] * 3.3, 2),
                                    "STANDS: move-tracking reward too weak vs standing's safety → strengthen far-field pull"))
    elif "CREEPS" in behavior:
        out.append(TuningSuggestion("TAILI_RW_W_TRACK_FAR", cur["w_track_far"], round(cur["w_track_far"] * 3.3, 2),
                                    "CREEPS: tracking gradient-starved → strengthen far-field pull (per-command)"))
    elif "PARTIAL" in behavior:
        out.append(TuningSuggestion("TAILI_RW_W_TRACK_FAR", cur["w_track_far"], round(cur["w_track_far"] * 1.5, 2),
                                    "PARTIAL: tracks but imprecise → modest bump (or more training to let the Gaussian sharpen)"))

    # ── per-gate refinements ──
    if "A2" in failed:
        out.append(TuningSuggestion("TAILI_RW_W_YAW_FAR", cur["w_yaw_far"], round(cur["w_yaw_far"] * 2.5, 2),
                                    "A2 yaw tracking fails → strengthen yaw far-field"))
    if "B3" in failed:
        out.append(TuningSuggestion("TAILI_RW_W_CLEARANCE_OVER", cur["w_clearance_over"], 0.8,
                                    "B3 over-lift: w_clearance_over=0.1 barely penalizes over-clearance → raise to penalize lifting too high"))
    return out


def render_prescription(suggestions: List[TuningSuggestion]) -> str:
    if not suggestions:
        return "no tuning suggested (verdict passed, or behavior not diagnosable)."
    lines = ["reward-tuning prescription (next run env vars):"]
    env = " ".join(f"{s.env_var}={s.suggested}" for s in suggestions)
    for s in suggestions:
        lines.append(f"  {s.env_var}: {s.current} → {s.suggested}  — {s.rationale}")
    lines.append(f"\n  launch: {env} python /root/tp_train_vN.py")
    return "\n".join(lines)
