"""Read-only tuning ledger assembled from the console's persisted LLM session.

This is not a new action system.  It gives the LLM a compact memory of recent
operator/assistant decisions, proposed actions, evidence tools, run ids and
diagnostic job ids so tuning discussions do not rely on fragile chat history.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import LocomotionConsoleSettings


_RUN_RE = re.compile(r"\b[a-zA-Z0-9_-]*20\d{6}_\d{6}[a-zA-Z0-9_-]*\b|\b[a-zA-Z0-9_-]*_\d{6,}\b")
_JOB_RE = re.compile(r"\b20\d{6}_\d{6}\b")
_CHECKPOINT_RE = re.compile(r"\bagent_\d+\.pt\b")


def build_tuning_ledger(settings: LocomotionConsoleSettings, *, limit: int = 24) -> dict[str, Any]:
    root = Path(settings.local_state_root) / "llm_session"
    turns = _read_jsonl(root / "turns.jsonl")[-max(1, min(limit, 200)):]
    proposals = _read_proposals(root / "proposals.json")[: max(1, min(limit, 200))]

    rows: list[dict[str, Any]] = []
    for turn in turns:
        rows.append(_turn_row(turn))

    recent_actions = [_proposal_row(item) for item in proposals[: max(1, min(limit, 80))]]
    runs = _collect_unique("runs", rows, recent_actions)
    jobs = _collect_unique("diagnostic_jobs", rows, recent_actions)
    checkpoints = _collect_unique("checkpoints", rows, recent_actions)

    return {
        "kind": "tuning_ledger",
        "permission_boundary": (
            "Local read-only summary of the console LLM session/proposal store. "
            "No remote access, no filesystem scan beyond the known session files, no writes."
        ),
        "state_root": str(root),
        "turns": rows,
        "proposals": recent_actions,
        "summary": {
            "turn_count": len(rows),
            "proposal_count": len(recent_actions),
            "runs_mentioned": runs[:12],
            "diagnostic_jobs_mentioned": jobs[:12],
            "checkpoints_mentioned": checkpoints[:12],
            "open_proposals": [
                item for item in recent_actions
                if item.get("status") == "pending"
            ][:8],
        },
        "limitations": [
            "This ledger records what the console discussed/proposed, not whether training improved.",
            "Use telemetry and diagnostic reports to verify outcomes.",
            "Old turns may include stale facts; prefer evidence timestamps and run ids.",
        ],
    }


def _turn_row(turn: dict[str, Any]) -> dict[str, Any]:
    content = str(turn.get("content") or "")
    transcript = [item for item in (turn.get("transcript") or []) if isinstance(item, dict)]
    evidence_tools = [str(item.get("tool") or "") for item in transcript if item.get("tool")]
    extracted = _extract_refs({"content": content, "transcript": transcript, "proposal": turn.get("proposed_action")})
    return {
        "id": turn.get("id"),
        "created_at": turn.get("created_at"),
        "role": turn.get("role"),
        "content_preview": content[:700],
        "evidence_tools": evidence_tools[:12],
        "proposed_action": turn.get("proposed_action"),
        "grounding": turn.get("grounding"),
        **extracted,
    }


def _proposal_row(item: dict[str, Any]) -> dict[str, Any]:
    extracted = _extract_refs(item)
    return {
        "id": item.get("id"),
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
        "status": item.get("status"),
        "name": item.get("name"),
        "args": item.get("args") or {},
        "risk": item.get("risk"),
        "evidence": item.get("evidence") or [],
        "requires": item.get("requires") or [],
        "expected_result": item.get("expected_result"),
        "rollback": item.get("rollback"),
        "reply_preview": str(item.get("reply") or "")[:700],
        "result_preview": str(item.get("result") or "")[:700],
        **extracted,
    }


def _extract_refs(value: Any) -> dict[str, list[str]]:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return {
        "runs": _dedupe(_RUN_RE.findall(text)),
        "diagnostic_jobs": _dedupe(_JOB_RE.findall(text)),
        "checkpoints": _dedupe(_CHECKPOINT_RE.findall(text)),
    }


def _collect_unique(key: str, *groups: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for group in groups:
        for item in group:
            for value in item.get(key) or []:
                if value not in out:
                    out.append(value)
    return out


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            rows.append(data)
    return rows


def _read_proposals(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    raw = data.get("items", data) if isinstance(data, dict) else data
    rows = [item for item in raw if isinstance(item, dict)]
    rows.sort(key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""), reverse=True)
    return rows


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out
