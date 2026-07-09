"""The LLM owner can orchestrate an ULTRACODE workflow (investigate -> adversarially verify ->
synthesize) for high-stakes calls, grounded on one shared evidence pull. Locks the mechanism:
registration, the call fan-out shape, verification filtering, and the empty-task guard."""
from dataclasses import dataclass

from autotuner.locomotion_console import agent


@dataclass
class _R:
    parsed: dict
    error: str = ""
    raw_text: str = ""
    model: str = ""
    elapsed_s: float = 0.0
    attempt: int = 0


def _install_fakes(monkeypatch, *, holds=True, calls=None):
    calls = calls if calls is not None else []

    def fake_llm(system_prompt, user_prompt, schema_name, max_retries=1):
        calls.append(schema_name)
        if "investigate" in schema_name:
            return _R({"finding": f"finding-{len(calls)}", "confidence": "high", "grounded_on": "ev"})
        if "verify" in schema_name:
            return _R({"holds": holds, "reason": "checked"})
        if "synthesize" in schema_name:
            return _R({"conclusion": "do X", "next_action": "measure Y", "confidence": "high", "caveat": "z"})
        return _R({})

    monkeypatch.setattr("autotuner.llm_gateway.client.call_llm_with_schema", fake_llm)
    monkeypatch.setattr(agent, "_tool_get_evidence_context",
                        lambda src, query="", **k: {"evidence_for": query})
    return calls


def test_run_workflow_registered_and_documented():
    assert "run_workflow" in agent.TOOLS
    assert "run_workflow(" in agent._TOOLS_DOC
    assert "ULTRACODE escalation" in agent._SYSTEM


def test_run_workflow_needs_a_task():
    out = agent._tool_run_workflow(None, task="   ")
    assert "error" in out


def test_run_workflow_fans_out_verifies_and_synthesizes(monkeypatch):
    calls = _install_fakes(monkeypatch, holds=True)
    out = agent._tool_run_workflow(None, task="which lever next?", angles="a,b,c")
    assert out["n_angles"] == 3
    assert len(out["findings"]) == 3
    assert out["survived_verification"] == 3
    assert out["synthesis"]["next_action"] == "measure Y"
    # investigate x3, verify x3, synthesize x1
    assert calls.count("console_workflow_investigate") == 3
    assert calls.count("console_workflow_verify") == 3
    assert calls.count("console_workflow_synthesize") == 1


def test_run_workflow_verification_filters_unsupported(monkeypatch):
    _install_fakes(monkeypatch, holds=False)
    out = agent._tool_run_workflow(None, task="root cause?", angles="a,b")
    # every finding refuted -> none survive, but findings are still reported for transparency
    assert out["survived_verification"] == 0
    assert len(out["findings"]) == 2
    assert all(f["holds"] is False for f in out["findings"])


def test_run_workflow_caps_angles(monkeypatch):
    _install_fakes(monkeypatch, holds=True)
    out = agent._tool_run_workflow(None, task="t", angles="a,b,c,d,e,f,g")
    assert out["n_angles"] == 5  # capped at 5
