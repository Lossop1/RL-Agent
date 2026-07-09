"""The measured taili_spec §2 verdict is a first-class copilot tool + auto-routed evidence:
the LLM must ground tuning/spec answers in the measured gates, not hand-authored status strings."""
from autotuner.locomotion_console import agent, evidence_router


def test_get_acceptance_registered_and_documented():
    assert "get_acceptance" in agent.TOOLS
    assert "get_acceptance()" in agent._TOOLS_DOC


def test_compact_acceptance_extracts_failing_gates():
    verdict = {
        "available": True, "run": "r1", "passed": False, "n_present": 9, "n_hard": 16,
        "missing": ["D"], "failed": ["A1"],
        "families": {"A1": {"present": True, "ok": False}},
        "gates": {"A1[fwd05]": {"ok": False, "detail": "med=0.14 <= 0.10"},
                  "A1[fwd03]": {"ok": True, "detail": "med=0.02 <= 0.10"}},
        "scorecards_read": 1,
    }
    c = agent._compact_acceptance_for_llm(verdict)
    assert c["available"] is True and c["passed"] is False
    assert c["failing_gates"] == {"A1[fwd05]": "med=0.14 <= 0.10"}   # only the failing sub-gate
    assert c["missing_families"] == ["D"]


def test_compact_acceptance_unavailable_offers_measurement():
    c = agent._compact_acceptance_for_llm({"available": False, "reason": "no physeval logs yet"})
    assert c["available"] is False
    assert "run_acceptance" in c["instruction"]        # offers to measure, doesn't fabricate


def test_acceptance_verdict_is_a_known_evidence_source():
    assert "acceptance_verdict" in evidence_router._SOURCES


def test_spec_and_tuning_queries_route_the_measured_verdict():
    for q in ["spec coverage and acceptance", "为什么没达到 spec 验收标准，怎么调参",
              "which gates fail vs the taili_spec"]:
        r = evidence_router.route_evidence(query=q, ui_mode="")
        ids = [s.get("id") for s in (r.get("selected_sources") or [])]
        assert "acceptance_verdict" in ids, q
