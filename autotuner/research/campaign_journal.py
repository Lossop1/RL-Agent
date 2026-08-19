"""Campaign Journal — the durable decision memory the tuning workflow runs on.

The playbook workflow has two guards that need MEMORY across iterations (and across sessions/context resets):
  - NO-REPEAT: "before applying a lever, check if it was already tried and rolled back."
  - DECIDE:    "keep only if it improved; else rollback" — needs the prior verdict to compare against.

Neither works if the agent re-derives from scratch each turn. This module persists, per robot, one entry per
tuning iteration: the hypothesis (gate+lever), the evidence (before/after verdict), and the outcome
(kept/rolled-back). It is deliberately small, append-only JSONL, and robot-keyed so a many-robot system keeps
separate campaigns. Read it to answer "what have we tried, what's the best, what must I not repeat."
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

_ROOT = Path(os.environ.get("LOCOMOTION_CONSOLE_STATE_ROOT", "/tmp/lc_console_state")) / "campaign"


def _path(robot: str) -> Path:
    _ROOT.mkdir(parents=True, exist_ok=True)
    return _ROOT / f"{robot or 'taili'}.jsonl"


def record_iteration(robot: str, *, target_gate: str, lever: str, config_diff: str = "",
                     base_checkpoint: str = "", result_run: str = "",
                     score_before: str = "", score_after: str = "", gate_before: str = "",
                     gate_after: str = "", decision: str = "", evidence: str = "", note: str = "",
                     ts: float | None = None) -> dict[str, Any]:
    """Append one iteration's decision. `decision` in {kept, rolled_back, pending}. `ts` must be supplied by
    the caller (the console has the clock); defaults to now only as a fallback."""
    entry = {
        "ts": ts if ts is not None else time.time(),
        "target_gate": target_gate, "lever": lever, "config_diff": config_diff,
        "base_checkpoint": base_checkpoint, "result_run": result_run,
        "score_before": score_before, "score_after": score_after,
        "gate_before": gate_before, "gate_after": gate_after,
        "decision": decision, "evidence": evidence, "note": note,
    }
    with _path(robot).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def _load(robot: str) -> list[dict[str, Any]]:
    p = _path(robot)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:  # noqa: BLE001
            continue
    return out


def read_journal(robot: str = "taili", limit: int = 30) -> dict[str, Any]:
    """The campaign's decision memory, summarized for the two guards that consume it:
      - tried_and_rolled_back: (gate,lever) pairs the NO-REPEAT guard must reject.
      - best_so_far: the highest kept score — what a new run must beat.
      - recent: the last N entries.
    """
    entries = _load(robot)
    rolled_back = sorted({f"{e.get('target_gate')}::{e.get('lever')}"
                          for e in entries if e.get("decision") == "rolled_back"})
    kept = [e for e in entries if e.get("decision") == "kept"]

    def _score_num(s: str) -> int:
        try:
            return int(str(s).split("/")[0])
        except Exception:
            return -1
    best = max(kept, key=lambda e: _score_num(e.get("score_after", "")), default=None)
    return {
        "kind": "campaign_journal", "robot": robot, "n_iterations": len(entries),
        "best_so_far": {"score": best.get("score_after"), "run": best.get("result_run"),
                        "via": f"{best.get('target_gate')}::{best.get('lever')}"} if best else None,
        "tried_and_rolled_back": rolled_back,
        "guard_note": "NO-REPEAT: do not re-apply any (gate::lever) in tried_and_rolled_back. "
                      "DECIDE: a new run must beat best_so_far.score on its target gate with no regression.",
        "recent": entries[-limit:],
    }
