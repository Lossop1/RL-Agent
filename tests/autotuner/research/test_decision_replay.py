"""Blind replay harness tests against recorded Taili decision patterns."""
from __future__ import annotations

from pathlib import Path

from autotuner.research.decision_replay import (
    DecisionReplayRunner,
    GlobalResearchDecisionPolicy,
    ReplayDecision,
    load_replay_cases,
)


CASES = Path("config/decision_replay/taili_core_cases.jsonl")


def test_global_policy_passes_recorded_holdout_cases():
    cases = load_replay_cases(CASES, split="holdout")
    report = DecisionReplayRunner().run(cases, GlobalResearchDecisionPolicy())
    assert report.case_count == 6
    assert report.pass_count == report.case_count, [
        (item.case_id, item.score, item.missing) for item in report.results if not item.passed
    ]


def test_policy_never_receives_withheld_outcome():
    case = load_replay_cases(CASES)[0]

    class InspectingPolicy:
        def decide(self, visible):
            assert "withheld_outcome" not in visible
            assert "correct_next_step" not in visible
            return ReplayDecision(
                actions=frozenset(), evidence=frozenset(), protected_capabilities=frozenset(),
            )

    result = DecisionReplayRunner().run_case(case, InspectingPolicy())
    assert result.outcome_after_decision == case.withheld_outcome
    assert result.passed is False
