"""Pure tests for lifecycle, evidence policy, and protected experiment guards."""
from __future__ import annotations

from autotuner.research.research_ledger import BaselineSet, ContractBundle, ExperimentPlan
from autotuner.research.research_supervisor import evidence_policy, experiment_readiness, lifecycle_transition


def test_lifecycle_does_not_allow_running_to_created():
    assert lifecycle_transition("run", "RUNNING", "CREATED")["status"] == "blocked"
    assert lifecycle_transition("run", "RUNNING", "STOPPING")["status"] == "allowed"


def test_evidence_policy_uses_small_window_for_obviously_bad_logs():
    decision = evidence_policy(log_quality="bad", behavior_quality="known")
    assert decision.default_action == "inspect_mechanism_without_full_diagnostic"
    assert "telemetry" in decision.required_kinds


def test_deployment_mismatch_blocks_before_reward_intervention():
    decision = evidence_policy(log_quality="good", behavior_quality="unknown", deployment_mismatch=True)
    assert decision.default_action == "block_intervention"
    assert decision.required_kinds == ["source_audit", "sim2sim"]


def test_experiment_readiness_protects_baseline_capabilities():
    contract = ContractBundle(
        id="contract:1", status="approved", approved_by="human", goals=["quality"], protected_capabilities=["flat", "stairs"]
    )
    baseline = BaselineSet(
        id="baseline:1", protected_capabilities=["flat", "stairs"], selection_reason="historical anchors"
    )
    plan = ExperimentPlan(
        id="experiment:1", problem_statement="test", baseline_ref="baseline:1",
        protected_capabilities=["flat"], intervention_diff={"reward.x": 1},
        unchanged_fields=["reward.y"], evaluation_plan={"suite": "flat"},
        success_condition="x", rollback_condition="y", resource_budget={"gpu_hours": 1}, status="approved"
    )
    result = experiment_readiness(plan, contract, baseline, policy_parity_status="blocked")
    assert result["status"] == "blocked"
    assert any("stairs" in blocker for blocker in result["blockers"])
    assert any("parity" in blocker for blocker in result["blockers"])
