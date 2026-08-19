"""Tests that outcomes alter durable state and become no-repeat knowledge."""
from __future__ import annotations

from pathlib import Path

from autotuner.mechanisms.mechanism_specs import Expression, MechanismBundle, RewardTermSpec, SignalSpec
from autotuner.mechanisms.mechanism_synthesis import CapabilityGap, MechanismSynthesizer, SynthesisRequest
from autotuner.research.outcome_learning import OutcomeLearner
from autotuner.research.research_ledger import ExperimentPlan, ResearchLedgerStore
from autotuner.research.research_state import (
    HypothesisState,
    ResearchCaseState,
    ResearchState,
    ResearchStateStore,
)
from autotuner.research.research_supervisor import ExperimentExecution


def _candidate():
    baseline = MechanismBundle(
        id="bundle:base", status="approved", contract_ref="contract:test",
        signals=(SignalSpec(name="progress", description="progress", source_ref="component.progress"),),
        rewards=(RewardTermSpec(
            id="reward:progress", name="progress", role="positive_drive",
            expression=Expression.signal("progress"), intended_effect="move",
            failure_region="still", success_region="moving",
        ),), evaluator_refs=("evaluator:flat",),
    )
    gap = CapabilityGap(
        id="slip", capability="low slip", symptom="feet slide",
        signal=SignalSpec(name="slip", description="stance slip", unit="m/s", source_ref="input.slip"),
        desired="lower", good=0.1, bad=0.5, hypothesis_ref="hypothesis:slip-drive",
    )
    request = SynthesisRequest(
        id="synthesis:slip", problem_ref="case:slip", problem_statement="feet slide",
        baseline=baseline, gaps=(gap,), protected_capabilities=("flat",),
        required_evaluators=("evaluator:flat",),
    )
    return MechanismSynthesizer().synthesize(request)[0]


def _plan():
    return ExperimentPlan(
        id="experiment:slip", problem_statement="feet slide", baseline_ref="bundle:base",
        intervention_diff={"patch": "slip"}, unchanged_fields=["contract"],
        protected_capabilities=["flat"], evaluation_plan={"suite": "flat"},
        success_condition="slip improves", rollback_condition="flat regresses",
        resource_budget={"gpu_hours": 1}, status="approved",
    )


def test_promoted_outcome_updates_hypothesis_ledger_and_no_repeat(tmp_path: Path):
    state_store = ResearchStateStore(tmp_path / "state")
    state = ResearchState(
        state_id="taili", active_case_ref="case:slip",
        cases={"case:slip": ResearchCaseState(id="case:slip", problem_statement="feet slide")},
        hypotheses={"hypothesis:slip-drive": HypothesisState(
            id="hypothesis:slip-drive", claim="slip lacks a positive quality path", confidence=0.4,
        )},
    )
    state_store.initialize(state, actor="test")
    ledger = ResearchLedgerStore(tmp_path / "ledger")
    candidate = _candidate()
    execution = ExperimentExecution(
        experiment_ref="experiment:slip", disposition="promote", workspace=str(tmp_path / "workspace"),
        backend_state="succeeded",
        evaluation={"success": True, "replicated": True, "resolves_case": True},
        protected_results={"flat": {"passed": True}},
    )
    result = OutcomeLearner(state_store, ledger).learn(
        expected_state_revision=0, plan=_plan(), candidate=candidate, execution=execution, actor="test",
    )
    updated = state_store.load()
    assert result.disposition == "promote"
    assert updated.hypotheses["hypothesis:slip-drive"].confidence > 0.4
    assert updated.hypotheses["hypothesis:slip-drive"].status == "validated"
    assert updated.cases["case:slip"].status == "resolved"
    assert candidate.candidate_bundle_fingerprint in updated.no_repeat_fingerprints
    assert ledger.records("experiment_outcome")[0]["disposition"] == "promote"
    assert ledger.records("knowledge_claim")[0]["status"] == "validated"

    duplicate = OutcomeLearner(state_store, ledger).learn(
        expected_state_revision=0, plan=_plan(), candidate=candidate, execution=execution, actor="test",
    )
    assert duplicate.idempotent is True
    assert ledger.summary()["record_event_counts"]["experiment_outcome"] == 1
