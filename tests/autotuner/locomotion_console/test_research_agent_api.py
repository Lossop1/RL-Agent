"""Agent and API integration tests for the live research loop."""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("LOCOMOTION_CONSOLE_SOURCE", "fake")

from fastapi.testclient import TestClient  # noqa: E402

from autotuner.locomotion_console import agent  # noqa: E402
from autotuner.locomotion_console import app as app_module  # noqa: E402
from autotuner.locomotion_console.config import LocomotionConsoleSettings  # noqa: E402
from autotuner.research.research_state import ResearchState, ResearchStateStore  # noqa: E402


def test_research_read_tools_are_registered_and_state_is_durable(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCOMOTION_RESEARCH_ROOT", str(tmp_path))
    ResearchStateStore(tmp_path / "state").initialize(
        ResearchState(state_id="taili", baseline_ref="bundle:base"),
        actor="test",
    )
    assert {
        "get_research_state",
        "validate_research_mechanism",
        "run_decision_replay",
    } <= set(agent.TOOLS)
    state = agent._tool_get_research_state(None)
    assert state["available"] is True
    assert state["summary"]["state_id"] == "taili"
    assert state["summary"]["chain_verified"] is True
    replay = agent._tool_run_decision_replay(None, split="holdout")
    assert replay["case_count"] >= 1
    assert replay["pass_count"] == replay["case_count"]
    assert agent._intent_tool_hint("查看当前研究状态")["tool"] == "get_research_state"
    assert agent._intent_tool_hint("运行历史决策盲回放")["tool"] == "run_decision_replay"


def test_execute_research_action_accepts_only_registered_ids(monkeypatch):
    called = []

    class DummyResult:
        execution = SimpleNamespace(disposition="promote")

        def model_dump(self, mode="json"):
            return {"execution": {"disposition": "promote"}}

    class DummyManager:
        def execute_registered(self, **kwargs):
            called.append(kwargs)
            return DummyResult()

    monkeypatch.setattr(
        "autotuner.locomotion_console.research_service.build_research_cycle_manager",
        lambda **kwargs: DummyManager(),
    )
    settings = LocomotionConsoleSettings(source="fake")
    rejected = agent.execute_action(
        "execute_research_cycle",
        {"cycle_id": "cycle:1", "plan_id": "plan:1", "command": ["bad"]},
        settings,
    )
    assert rejected["ok"] is False
    assert called == []
    accepted = agent.execute_action(
        "execute_research_cycle",
        {"cycle_id": "cycle:1", "plan_id": "plan:1"},
        settings,
    )
    assert accepted["ok"] is True
    assert called == [{
        "cycle_id": "cycle:1",
        "plan_id": "plan:1",
        "actor": "operator-confirmed-agent",
    }]
    assert agent.action_risk("execute_research_cycle") == "destructive"


def test_research_api_state_cas_validation_and_execution_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCOMOTION_RESEARCH_ROOT", str(tmp_path))
    calls = []

    class ApiResult:
        def model_dump(self, mode="json"):
            return {"execution": {"disposition": "rollback"}}

    class DummyManager:
        def execute_registered(self, **kwargs):
            calls.append(kwargs)
            return ApiResult()

    with TestClient(app_module.app) as client:
        initialized = client.post("/research/state/initialize", json={
            "state": {"state_id": "taili", "baseline_ref": "bundle:base"},
            "actor": "test",
        })
        assert initialized.status_code == 200
        state = client.get("/research/state")
        assert state.status_code == 200
        assert state.json()["summary"]["revision"] == 0
        updated = client.post("/research/state/update", json={
            "expected_revision": 0,
            "changes": {"active_run_ref": "run:1"},
            "actor": "test",
            "reason": "bind run",
        })
        assert updated.status_code == 200
        stale = client.post("/research/state/update", json={
            "expected_revision": 0,
            "changes": {"active_run_ref": "run:stale"},
            "actor": "test",
            "reason": "stale update",
        })
        assert stale.status_code == 409
        validation = client.post("/research/mechanisms/validate", json={
            "bundle": {"id": "bundle:test", "contract_ref": "contract:test"},
        })
        assert validation.status_code == 200
        assert "checks" in validation.json()

        monkeypatch.setattr(app_module, "_research_cycle_manager", lambda: DummyManager())
        injected = client.post("/research/cycles/execute", json={
            "cycle_id": "cycle:1",
            "plan_id": "plan:1",
            "command": ["python", "train.py"],
        })
        assert injected.status_code == 422
        executed = client.post("/research/cycles/execute", json={
            "cycle_id": "cycle:1",
            "plan_id": "plan:1",
        })
        assert executed.status_code == 200
    assert calls == [{
        "cycle_id": "cycle:1",
        "plan_id": "plan:1",
        "actor": "operator-confirmed-api",
    }]
