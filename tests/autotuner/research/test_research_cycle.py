"""End-to-end local research cycle from unknown gap to learned outcome."""
from __future__ import annotations

from pathlib import Path

import pytest

from autotuner.mechanisms.mechanism_specs import Expression, MechanismBundle, RewardTermSpec, SignalSpec
from autotuner.mechanisms.mechanism_synthesis import CapabilityGap, SynthesisRequest
from autotuner.research.research_cycle import ResearchCycleManager
from autotuner.research.research_ledger import ExperimentPlan, ResearchLedgerStore
from autotuner.research.research_state import ResearchCaseState, ResearchState, ResearchStateStore
from autotuner.research.research_supervisor import BackendHandle, BackendStatus, ResearchSupervisor


class PassingBackend:
    def start(self, plan, workspace, environment):
        assert environment["RL_RESEARCH_EXPERIMENT_REF"] == plan.id
        bundle = Path(environment["RL_MECHANISM_BUNDLE"])
        assert bundle.name == "mechanisms.json"
        assert bundle.parent.name == "candidate"
        return BackendHandle("passing", 0.0)

    def poll(self, handle):
        return BackendStatus("succeeded", step=100)

    def stop(self, handle, reason):
        raise AssertionError("passing backend must not stop")

    def evaluate(self, plan, workspace):
        return {
            "success": True, "evidence_complete": True, "replicated": False,
            "resolves_case": True, "protected": {"flat": {"passed": True}},
            "capability_delta": {"core_stability": 0.2},
        }


def _request():
    baseline = MechanismBundle(
        id="bundle:base", status="approved", contract_ref="contract:test",
        signals=(SignalSpec(
            name="progress", description="command progress", source_ref="component.progress",
            lower_bound=0.0, upper_bound=1.0,
        ),),
        rewards=(RewardTermSpec(
            id="reward:progress", name="progress", role="positive_drive",
            expression=Expression.signal("progress"), intended_effect="move",
            failure_region="still", success_region="moving",
        ),), evaluator_refs=("evaluator:flat",),
    )
    return SynthesisRequest(
        id="cycle:unknown-core-gap", problem_ref="case:core", problem_statement="unknown core oscillation",
        baseline=baseline,
        gaps=(CapabilityGap(
            id="core-wxy", capability="core stability", symptom="body oscillates",
            signal=SignalSpec(
                name="core.wxy", description="roll/pitch speed", unit="rad/s",
                source_ref="input.base_ang_vel", selector=(0, 1), shape="vector", lower_bound=0.0,
            ),
            signal_reduction="norm", desired="lower", good=0.2, bad=0.8,
            hypothesis_ref="hypothesis:new-core-path", reward_group="stab",
        ),),
        protected_capabilities=("flat",), required_evaluators=("evaluator:flat",),
    )


def test_complete_cycle_synthesizes_executes_promotes_and_learns(tmp_path: Path):
    state_store = ResearchStateStore(tmp_path / "state")
    state_store.initialize(ResearchState(
        state_id="taili", active_case_ref="case:core",
        cases={"case:core": ResearchCaseState(id="case:core", problem_statement="unknown core oscillation")},
    ), actor="test")
    ledger = ResearchLedgerStore(tmp_path / "ledger")
    supervisor = ResearchSupervisor(tmp_path / "experiments", backend=PassingBackend())
    manager = ResearchCycleManager(
        tmp_path / "cycles", state_store=state_store, ledger=ledger, supervisor=supervisor,
    )
    proposal = manager.propose(_request(), expected_state_revision=0, actor="test")
    assert proposal.status == "candidate_selected"
    assert proposal.selected_artifact_path
    assert len(proposal.assessments) == 2
    assert sum(item.selected for item in proposal.assessments) == 1

    plan = ExperimentPlan(
        id="experiment:unknown-core-gap", problem_statement="test core mechanism",
        baseline_ref=proposal.baseline_bundle_ref,
        intervention_diff={"patch_ref": proposal.selected_patch_ref},
        unchanged_fields=["contract", "evaluator"], protected_capabilities=["flat"],
        evaluation_plan={"suite": "evaluator:flat"}, success_condition="core improves",
        rollback_condition="flat regresses", resource_budget={"resources": ["gpu:0"]},
        authorization_ref="approval:test",
        status="approved",
    )
    manager.register_plan(plan, actor="operator")
    result = manager.execute_registered(
        cycle_id=proposal.cycle_id,
        plan_id=plan.id,
        actor="operator-confirmed",
        poll_interval_s=0,
    )
    assert result.execution.disposition == "promote"
    assert result.learning.disposition == "promote"
    assert state_store.load().cases["case:core"].status == "resolved"
    assert ledger.records("mechanism_candidate")
    assert ledger.records("experiment_outcome")
    with pytest.raises(RuntimeError, match="execution receipt"):
        manager.execute_registered(
            cycle_id=proposal.cycle_id,
            plan_id=plan.id,
            actor="operator-confirmed",
            poll_interval_s=0,
        )


def test_registered_execution_rejects_plan_not_bound_to_selected_patch(tmp_path: Path):
    state_store = ResearchStateStore(tmp_path / "state")
    state_store.initialize(ResearchState(
        state_id="taili",
        cases={"case:core": ResearchCaseState(id="case:core", problem_statement="core")},
    ), actor="test")
    ledger = ResearchLedgerStore(tmp_path / "ledger")
    manager = ResearchCycleManager(
        tmp_path / "cycles",
        state_store=state_store,
        ledger=ledger,
        supervisor=ResearchSupervisor(tmp_path / "experiments", backend=PassingBackend()),
    )
    proposal = manager.propose(_request(), expected_state_revision=0, actor="test")
    plan = ExperimentPlan(
        id="experiment:wrong-patch",
        problem_statement="must remain bound to reviewed candidate",
        baseline_ref=proposal.baseline_bundle_ref,
        intervention_diff={"patch_ref": "patch:not-selected"},
        unchanged_fields=["contract", "evaluator"],
        protected_capabilities=["flat"],
        evaluation_plan={"suite": "evaluator:flat"},
        success_condition="core improves",
        rollback_condition="flat regresses",
        resource_budget={"resources": ["gpu:0"]},
        authorization_ref="approval:test",
        status="approved",
    )
    manager.register_plan(plan, actor="operator")
    with pytest.raises(ValueError, match="not bound"):
        manager.execute_registered(
            cycle_id=proposal.cycle_id,
            plan_id=plan.id,
            actor="operator-confirmed",
            poll_interval_s=0,
        )
