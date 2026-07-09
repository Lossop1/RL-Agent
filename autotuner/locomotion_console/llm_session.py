"""Persistent LLM conversation and proposal registry.

The console is currently single-user, so a compact JSONL store is enough. It keeps
operator-facing LLM behavior inspectable across backend restarts without making
the LLM part of the deterministic execution path.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import LocomotionConsoleSettings


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LLMSessionStore:
    def __init__(self, settings: LocomotionConsoleSettings):
        self.root = Path(settings.local_state_root) / "llm_session"
        self.turns_path = self.root / "turns.jsonl"
        self.proposals_path = self.root / "proposals.json"

    def recent_turns(self, limit: int = 16) -> list[dict[str, str]]:
        rows = self._read_jsonl(self.turns_path)
        result: list[dict[str, str]] = []
        for row in rows[-max(1, min(limit, 200)):]:
            role = str(row.get("role") or "")
            content = str(row.get("content") or "")
            if role in {"user", "assistant"} and content:
                result.append({"role": role, "content": content})
        return result

    def append_turn(
        self,
        *,
        role: str,
        content: str,
        transcript: list[dict[str, Any]] | None = None,
        proposed_action: dict[str, Any] | None = None,
        grounding: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        item = {
            "id": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"),
            "created_at": _utc_now(),
            "role": role,
            "content": content,
            "tools": [str(t.get("tool") or "") for t in (transcript or []) if t.get("tool")],
            "transcript": transcript or [],
            "proposed_action": proposed_action,
            "grounding": grounding,
        }
        self._append_jsonl(self.turns_path, item)
        return item

    def record_proposal(self, *, reply: str, proposed_action: dict[str, Any]) -> dict[str, Any]:
        proposals = self._read_proposals()
        action = proposed_action or {}
        item = {
            "id": datetime.now(timezone.utc).strftime("proposal_%Y%m%d_%H%M%S_%f"),
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "status": "pending",
            "name": str(action.get("name") or ""),
            "args": action.get("args") or {},
            "reply": reply,
            "result": "",
            "kind": str(action.get("kind") or "action"),
            "risk": str(action.get("risk") or "medium"),
            "evidence": list(action.get("evidence") or []),
            "requires": list(action.get("requires") or []),
            "expected_result": str(action.get("expected_result") or ""),
            "rollback": str(action.get("rollback") or ""),
        }
        proposals.append(item)
        self._write_proposals(proposals[-200:])
        return item

    def record_proposals(self, proposals_to_add: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        recorded = []
        for proposal in proposals_to_add:
            recorded.append(self.record_proposal(
                reply=str(proposal.get("reply") or proposal.get("expected_result") or ""),
                proposed_action=proposal,
            ))
        return recorded

    def update_latest_pending(self, *, name: str, ok: bool, detail: str) -> dict[str, Any] | None:
        proposals = self._read_proposals()
        for item in reversed(proposals):
            if item.get("status") == "pending" and item.get("name") == name:
                item["status"] = "executed" if ok else "failed"
                item["updated_at"] = _utc_now()
                item["result"] = detail
                self._write_proposals(proposals[-200:])
                return item
        return None

    def find_pending(self, *, name: str) -> dict[str, Any] | None:
        """Read-only: the latest still-pending proposal for this action name, or None. Used to BIND
        /chat/execute to an action the copilot actually proposed (defeats out-of-band execution)."""
        for item in reversed(self._read_proposals()):
            if item.get("status") == "pending" and item.get("name") == name:
                return item
        return None

    def mark_latest_pending_cancelled(self, *, name: str) -> dict[str, Any] | None:
        proposals = self._read_proposals()
        for item in reversed(proposals):
            if item.get("status") == "pending" and item.get("name") == name:
                item["status"] = "cancelled"
                item["updated_at"] = _utc_now()
                self._write_proposals(proposals[-200:])
                return item
        return None

    def proposals(self, limit: int = 30) -> list[dict[str, Any]]:
        rows = self._read_proposals()
        rows.sort(key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""), reverse=True)
        return rows[: max(1, min(limit, 200))]

    def reset(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.turns_path.write_text("", encoding="utf-8")
        self._write_proposals([])

    def _append_jsonl(self, path: Path, item: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows

    def _read_proposals(self) -> list[dict[str, Any]]:
        if not self.proposals_path.exists():
            return []
        try:
            data = json.loads(self.proposals_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        raw = data.get("items", data) if isinstance(data, dict) else data
        return [item for item in raw if isinstance(item, dict)]

    def _write_proposals(self, items: Iterable[dict[str, Any]]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "items": list(items)}
        tmp = self.proposals_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.proposals_path)
