"""Tests for durable current research state and context-loss recovery."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from autotuner.research.research_state import (
    GoalState,
    ResearchCaseState,
    ResearchState,
    ResearchStateStore,
    RevisionConflict,
    StateIntegrityError,
)


def _state() -> ResearchState:
    return ResearchState(
        state_id="taili",
        program_ref="program:taili",
        contract_ref="contract:taili",
        goals=[GoalState(id="flat", statement="ideal flat locomotion", acceptance=["flat-suite passes"])],
    )


def test_compare_and_set_rejects_stale_context(tmp_path: Path):
    store = ResearchStateStore(tmp_path)
    first = store.initialize(_state(), actor="test")
    second = store.compare_and_set(
        first.revision,
        {"active_run_ref": "run:1"},
        actor="monitor",
        reason="bind active run",
    )
    assert second.revision == 1
    with pytest.raises(RevisionConflict) as exc:
        store.compare_and_set(first.revision, {"active_run_ref": "stale"}, actor="old-turn", reason="stale write")
    assert exc.value.actual == 1
    assert store.load().active_run_ref == "run:1"


def test_event_log_recovers_interrupted_snapshot_write(tmp_path: Path):
    store = ResearchStateStore(tmp_path)
    state = store.initialize(_state(), actor="test")
    state = store.compare_and_set(
        state.revision,
        lambda value: value.model_copy(
            update={
                "active_case_ref": "case:stand",
                "cases": {
                    "case:stand": ResearchCaseState(
                        id="case:stand",
                        problem_statement="zero-command stand is not quiet",
                    )
                },
            }
        ),
        actor="agent",
        reason="open case",
    )

    # Simulate a crash after the durable event append but before snapshot replace.
    stale = _state().model_copy(update={"revision": 0})
    store.state_path.write_text(stale.model_dump_json(), encoding="utf-8")
    recovered = store.load(recover=True)

    assert recovered.revision == state.revision
    assert recovered.active_case_ref == "case:stand"
    assert ResearchState.model_validate_json(store.state_path.read_text(encoding="utf-8")).revision == state.revision


def test_event_chain_detects_tampering(tmp_path: Path):
    store = ResearchStateStore(tmp_path)
    store.initialize(_state(), actor="test")
    event = json.loads(store.events_path.read_text(encoding="utf-8").splitlines()[0])
    event["state"]["contract_ref"] = "tampered"
    store.events_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    with pytest.raises(StateIntegrityError):
        store.history()


def test_summary_is_compact_and_chain_verified(tmp_path: Path):
    store = ResearchStateStore(tmp_path)
    store.initialize(_state(), actor="test")
    summary = store.summary()
    assert summary["state_id"] == "taili"
    assert summary["revision"] == 0
    assert summary["active_goals"] == ["flat"]
    assert summary["chain_verified"] is True
