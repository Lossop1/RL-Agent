"""Backend API smoke + secret-safety tests (fake source, no GPU, no SSH).

Guarantees the console read surface never 5xx's and never serializes a literal secret
(SSH password / LLM api key) into a response body.
"""
import json
import os

import pytest

os.environ.setdefault("LOCOMOTION_CONSOLE_SOURCE", "fake")

from fastapi.testclient import TestClient  # noqa: E402

from autotuner.locomotion_console.app import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


READ_ROUTES = [
    "/health", "/definitions", "/spec/coverage",
    "/config/frameworks", "/config/active", "/config/remote", "/config/llm", "/config/robot",
    "/config/framework/catalog",
    "/diagnostics/catalog", "/diagnostics/status", "/diagnostics/history",
    "/run/current", "/run/current/snapshot", "/run/current/telemetry",
    "/run/current/acceptance", "/config/strategy",
    "/remote/status", "/llm/readiness", "/chat/proposals", "/agent/workbench",
]


def test_strategy_view_exposes_tuning_settings():
    # The detailed-settings panel gap: reward weights + curriculum gates + AMP must be visible.
    from autotuner.locomotion_console.strategy_view import build_strategy_view
    v = build_strategy_view()
    assert v["available"] is True
    assert v["counts"]["reward_weights"] > 10 and v["counts"]["phases"] == 4
    assert "w_stance_slip" in v["reward"]["gait_quality"]          # a tuning-relevant weight present
    assert v["amp"]["style_reward_weight"] == 2.0                  # AMP hyperparams surfaced


def test_websocket_stream_connects(client):
    # Regression: the mutation-auth gate was once a GLOBAL dependency typed on Request, which made the
    # WebSocket handshake 500 (no Request in WS scope). It must be HTTP-scope only so the live stream
    # connects and self-authorizes via _ws_authorized.
    with client.websocket_connect("/run/current/stream") as ws:
        assert isinstance(ws.receive_json(), dict)


def test_websocket_same_origin_allowed_even_with_token(monkeypatch):
    # Regression: the WS origin allowlist only knew the dev :5173 origin, so the single-port :8000 UI
    # was silently rejected (and adding a server token 403'd every client). Policy now mirrors HTTP
    # reads: SAME-ORIGIN browser pages always stream; foreign origins need the token.
    from autotuner.locomotion_console import app as A
    monkeypatch.setattr(A, "_CONSOLE_TOKEN", "sekrit")
    with TestClient(A.app) as c:
        # same-origin page (Origin matches Host) — allowed without token
        with c.websocket_connect("/run/current/stream",
                                 headers={"Origin": "http://testserver", "Host": "testserver"}) as ws:
            assert isinstance(ws.receive_json(), dict)
        # foreign origin without token — rejected
        import pytest as _pytest
        with _pytest.raises(Exception):
            with c.websocket_connect("/run/current/stream",
                                     headers={"Origin": "http://evil.example", "Host": "testserver"}):
                pass
        # foreign origin WITH the token — allowed
        with c.websocket_connect("/run/current/stream?token=sekrit",
                                 headers={"Origin": "http://evil.example", "Host": "testserver"}) as ws:
            assert isinstance(ws.receive_json(), dict)


def test_acceptance_endpoint_shape(client):
    # Read-only benchmark verdict endpoint: always 200 with an `available` flag (never 5xx), and in the
    # fake source it is cleanly unavailable rather than erroring.
    body = client.get("/run/current/acceptance").json()
    assert body.get("available") is False
    assert "reason" in body


def test_agent_workbench_contract(client):
    body = client.get("/agent/workbench").json()
    assert body["objective"]
    assert body["service_loop"] == ["目标", "证据", "判断", "提案", "授权执行", "验证", "记录"]
    assert body["attempt"]["runtime_state"]
    assert body["judgement"]["title"]
    assert {item["id"] for item in body["evidence"]} >= {"llm", "run", "telemetry", "config", "remote"}
    assert {item["id"] for item in body["actions"]} >= {
        "ask-agent", "run-diagnostic", "deploy-payload", "start-training", "resume-training", "kill-training"
    }


def test_chat_context_route_manages_injected_sources(client):
    body = client.post("/chat/context/route", json={
        "message": "LLM 超时了，先看远程 GPU、tmux 和当前训练状态",
        "ui_mode": "training",
        "context": "run=fake | state=live | progress=0.2",
        "context_request": {
            "page": "training",
            "intent": "排查 LLM 超时和远程机器状态",
            "focus": ["remote", "gpu", "tmux", "timeout", "telemetry"],
            "visible": {"run": "fake", "state": "live", "progress": 0.2},
            "max_chars": 1800,
            "include_remote_status": True,
        },
    }).json()
    assert body["page"] == "training"
    assert "remote_status" in body["selected_sources"]
    assert "training_telemetry" in body["selected_sources"]
    assert body["injected_chars"] <= body["budget_chars"] <= 1800
    assert "selected_sources=" in body["prompt"]


def test_chat_execute_is_proposal_bound(client):
    # Safety envelope: /chat/execute must refuse an action the copilot never proposed (defeats an
    # out-of-band destructive POST by a token holder), and accept it once a matching proposal exists.
    from autotuner.locomotion_console.app import llm_session
    h = {"X-Console-Token": ""}  # loopback + no configured token → passes the mutation gate
    assert client.post("/chat/execute", json={"name": "kill_training", "args": {}}, headers=h).status_code == 409
    assert client.post("/chat/execute", json={"name": "nope", "args": {}}, headers=h).status_code == 400
    llm_session.record_proposal(reply="stop", proposed_action={"name": "kill_training", "args": {}})
    assert client.post("/chat/execute", json={"name": "kill_training", "args": {}}, headers=h).status_code != 409


def test_action_risk_tiers():
    from autotuner.locomotion_console import agent
    assert agent.action_risk("kill_training") == "destructive"
    assert agent.action_risk("edit_config") == "low"
    assert agent.action_risk("run_acceptance") == "medium"
    assert agent.action_risk("run_campaign") == "destructive"
    assert agent.action_risk("get_status") == "auto"  # read-only default


def test_mutation_lock_serializes_actions_but_not_reads(client):
    # Concurrency: a held mutation lock 409s a state-changing action but never blocks reads/telemetry.
    import asyncio
    from autotuner.locomotion_console import app as A
    h = {"X-Console-Token": ""}
    asyncio.get_event_loop().run_until_complete(A._ACTION_LOCK.acquire())
    try:
        assert client.get("/run/current").status_code == 200          # read unaffected
        assert client.get("/run/current/telemetry").status_code == 200
        assert client.post("/action/kill", headers=h).status_code == 409   # mutation serialized
    finally:
        A._ACTION_LOCK.release()
    assert client.post("/action/kill", headers=h).status_code != 409   # freed after release


@pytest.mark.parametrize("route", READ_ROUTES)
def test_read_routes_never_5xx(client, route):
    r = client.get(route)
    assert r.status_code < 500, f"{route} -> {r.status_code}: {r.text[:200]}"


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_remote_config_never_returns_literal_password(client):
    body = client.get("/config/remote").json()
    # The endpoint must expose a boolean flag, never the secret itself.
    assert "password_configured" in body
    assert isinstance(body["password_configured"], bool)
    assert "password" not in {k for k, v in body.items() if isinstance(v, str) and v}
    assert "ssh_pass" not in body


def test_llm_config_never_returns_literal_api_key(client):
    body = client.get("/config/llm").json()
    assert "api_key_resolved" in body and isinstance(body["api_key_resolved"], bool)
    # No field should contain an actual key value; only resolution flags.
    flat = json.dumps(body)
    assert "sk-" not in flat  # OpenAI-style literal key prefix must never appear
    # Safety invariants surfaced to the UI.
    assert body.get("advisory_only") is True
    assert body.get("actions_require_confirmation") is True


def test_unknown_route_is_404_not_500(client):
    assert client.get("/definitely/not/a/route").status_code == 404


def test_openapi_serves(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert len(r.json().get("paths", {})) >= 40


# ── auth / CSRF on state-changing endpoints ──
def test_mutating_route_without_token_header_allowed_by_local_default(client):
    # Local operator default: token gate is disabled unless LOCOMOTION_CONSOLE_REQUIRE_TOKEN=1.
    # This keeps the console usable on loopback during active lab work.
    assert client.post("/action/kill").status_code not in (401, 403)
    assert client.patch("/config/remote", json={}).status_code not in (401, 403)


def test_mutating_route_with_header_allowed_from_loopback(client):
    # Header present + loopback (TestClient) + no token configured → passes the auth gate
    # (returns a normal 2xx/4xx action result, never the 401/403 auth rejection).
    r = client.post("/action/kill", headers={"X-Console-Token": ""})
    assert r.status_code not in (401, 403)


def test_reads_are_open_without_token(client):
    # GET reads never require the header.
    assert client.get("/run/current").status_code < 400


def test_token_enforced_when_configured(monkeypatch):
    # With a token configured, the value must match (constant-time). Rebuild the app so the
    # module-level _CONSOLE_TOKEN picks up the env var.
    import importlib

    monkeypatch.setenv("LOCOMOTION_CONSOLE_TOKEN", "s3cret")
    monkeypatch.setenv("LOCOMOTION_CONSOLE_REQUIRE_TOKEN", "1")
    monkeypatch.setenv("LOCOMOTION_CONSOLE_SOURCE", "fake")
    from autotuner.locomotion_console import app as app_module

    app_module = importlib.reload(app_module)
    with TestClient(app_module.app) as c:
        assert c.post("/action/kill", headers={"X-Console-Token": "wrong"}).status_code == 401
        assert c.post("/action/kill", headers={"X-Console-Token": "s3cret"}).status_code not in (401, 403)
        assert c.get("/health").status_code == 200  # reads still open
    # restore the default app module for other tests
    monkeypatch.delenv("LOCOMOTION_CONSOLE_TOKEN", raising=False)
    monkeypatch.delenv("LOCOMOTION_CONSOLE_REQUIRE_TOKEN", raising=False)
    importlib.reload(app_module)


def test_autonomy_tiers_never_include_destructive(monkeypatch):
    # The BRAIN may auto-execute low/medium actions per LOCOMOTION_CONSOLE_LLM_AUTONOMY;
    # destructive (start/kill/deploy/campaign) must NEVER be auto-executable.
    from autotuner.locomotion_console import agent
    monkeypatch.delenv("LOCOMOTION_CONSOLE_LLM_AUTONOMY", raising=False)
    assert agent._autonomy_auto_tiers() == frozenset()          # default: propose-only
    monkeypatch.setenv("LOCOMOTION_CONSOLE_LLM_AUTONOMY", "assisted")
    assert agent._autonomy_auto_tiers() == frozenset({"low"})
    monkeypatch.setenv("LOCOMOTION_CONSOLE_LLM_AUTONOMY", "autonomous")
    tiers = agent._autonomy_auto_tiers()
    assert tiers == frozenset({"low", "medium"})
    for destructive in ("start_training", "kill_training", "deploy_payload", "run_campaign"):
        assert agent.action_risk(destructive) not in tiers
