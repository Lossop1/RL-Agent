from __future__ import annotations

import json
from pathlib import Path

import pytest

from autotuner.execution import DeploymentResult, RemoteArtifact, TrainingStartResult
from autotuner.product import (
    ProductPayload,
    TaskContractCompiler,
    TrainingLaunchRequest,
    resolve_product_contract,
)
from autotuner.mechanisms.mechanism_specs import Expression, MechanismBundle, RewardTermSpec
from autotuner.mechanisms.mechanism_synthesis import MechanismIntent, SynthesisRequest
from autotuner.research.coordinator import (
    EvaluationSubmission,
    EvaluatorResult,
    ResearchCoordinator,
    ResearchCoordinatorError,
)
from autotuner.research.research_ledger import EvidenceRecord
from autotuner.llm_gateway.research_proposal import ResearchProposal, ResearchProposalRequest
from autotuner.research.research_state import ResearchState, RevisionConflict


def _task_request() -> dict:
    return {
        "task": {
            "instance_id": "coordinator-test",
            "objective": "验证研究协调闭环",
            "goals": [{"id": "quality", "objective": "保持受保护能力"}],
            "protected_capabilities": ["quality"],
            "approved": True,
        }
    }


def _proposal() -> ResearchProposal:
    return ResearchProposal(
        proposal_id="proposal:core-adjustment",
        problem_ref="case:flat-quality",
        problem_statement="当前候选需要证据验证",
        evidence_refs=("evidence:baseline",),
        hypothesis_refs=("hypothesis:quality",),
        task_changes={"training": {"config_overlay": {"training_recipe": {"window": 1}}}},
        expected_effects=("质量保持",),
        falsifiable_predictions=("质量评估通过",),
        protected_capabilities=("quality",),
        required_evaluators=("flat-evaluator",),
        confidence=0.7,
    )


def _request() -> ResearchProposalRequest:
    return ResearchProposalRequest(
        problem_ref="case:flat-quality",
        problem_statement="当前候选需要证据验证",
        evidence_refs=("evidence:baseline",),
        evidence_summary={"quality": "baseline"},
        protected_capabilities=("quality",),
        required_evaluators=("flat-evaluator",),
        baseline_bundle_ref="mechanisms:baseline",
    )


def _launch_request(run_id: str) -> TrainingLaunchRequest:
    return TrainingLaunchRequest(
        payload_root="/remote/payload",
        run_id=run_id,
        total_steps=10,
        telemetry_interval=1,
        num_envs=1,
    )


class FakeDeployer:
    def __init__(self, *, deploy_fails: bool = False, rollback_fails: bool = False) -> None:
        self.deployed: list[str] = []
        self.rolled_back: list[str] = []
        self.deploy_fails = deploy_fails
        self.rollback_fails = rollback_fails

    def deploy_spec(self, spec):
        if self.deploy_fails:
            raise RuntimeError("remote deployment failed")
        self.deployed.append(spec.run_id)
        return DeploymentResult(
            status="activated",
            runtime=RemoteArtifact("runtime", spec.runtime_digest, "runtime", "activated"),
            payload=RemoteArtifact("payload", spec.payload_digest, "payload", "activated"),
            run=RemoteArtifact("run", spec.run_id, "runs/" + spec.run_id, "activated"),
        )

    def rollback_active(self, run_id: str):
        if self.rollback_fails:
            raise RuntimeError("remote is temporarily unavailable")
        self.rolled_back.append(run_id)
        return RemoteArtifact("run", run_id, "runs/" + run_id, "rolled_back")


class FakeStarter:
    def __init__(self, *, failed: bool = False) -> None:
        self.failed = failed
        self.plans = []

    def start(self, plan):
        self.plans.append(plan)
        return TrainingStartResult(
            status="failed" if self.failed else "started",
            run_id=plan.run_id,
            run_dir=plan.run_dir,
            handle_ref=f"tmux:{plan.tmux_session}",
            process_pattern=plan.process_pattern,
            error="simulated start failure" if self.failed else "",
        )


def _make_coordinator(monkeypatch, tmp_path: Path) -> tuple[ResearchCoordinator, object, object]:
    product = resolve_product_contract("taili")
    compiler = TaskContractCompiler()
    base = compiler.compile_bundle(product, _task_request())

    def fake_payload(*, contract, output_dir, task_bundle, task_artifacts, asset_bindings=()):
        return ProductPayload(
            archive=tmp_path / "payload.tar.gz",
            manifest=tmp_path / "payload.json",
            payload_digest="d" * 64,
        )

    monkeypatch.setattr("autotuner.product.task_pipeline.build_product_payload", fake_payload)
    coordinator = ResearchCoordinator(tmp_path / "output")
    coordinator.pipeline.contract_store.save(base, actor="test")
    coordinator.initialize(
        ResearchState(
            program_ref="program:test",
            contract_ref=base.contract.contract_id + "@1",
            baseline_ref="baseline-run",
            active_run_ref="baseline-run",
            active_checkpoint_ref="baseline/checkpoint.pt",
        ),
        actor="test",
    )
    baseline_evidence = EvidenceRecord(
        id="evidence:baseline",
        kind="telemetry",
        run_ref="baseline-run",
        contract_ref=base.contract.contract_id + "@1",
        raw_artifact_ref="baseline/telemetry.jsonl",
    )
    coordinator.ledger.append("evidence", baseline_evidence, actor="test")
    return coordinator, product, base


def test_coordinator_prepares_deploys_and_promotes_after_complete_evidence(monkeypatch, tmp_path: Path) -> None:
    coordinator, product, base = _make_coordinator(monkeypatch, tmp_path)
    proposal = _proposal()
    request = _request()

    applied = coordinator.approve_and_prepare(
        product,
        base,
        proposal,
        request,
        approved_by="human",
        run_id="run-candidate",
        expected_state_revision=0,
        launch_request=_launch_request("run-candidate"),
    )
    assert applied.pipeline_result.run_manifest.is_file()
    deployed = coordinator.deploy_prepared(
        "run-candidate",
        FakeDeployer(),
        expected_state_revision=1,
    )
    assert deployed.lifecycle.status == "deployed"
    assert deployed.state.observation_window is not None
    assert deployed.state.observation_window.status == "planned"
    started = coordinator.start_deployed(
        "run-candidate",
        FakeStarter(),
        expected_state_revision=deployed.state.revision,
    )
    assert started.lifecycle.status == "observing"

    evidence = EvidenceRecord(
        id="evidence:candidate:diag",
        kind="physical_diag",
        run_ref="run-candidate",
        contract_ref=deployed.lifecycle.contract_ref,
        raw_artifact_ref="candidate/diag.json",
        extracted_facts=[{"name": "quality", "value": 1.0, "confidence": 0.9}],
    )
    ingested = coordinator.ingest_evidence(evidence, expected_state_revision=started.state.revision)
    result = coordinator.evaluate(
        EvaluationSubmission(
            evaluation_id="evaluation:pass",
            run_ref="run-candidate",
            contract_ref=deployed.lifecycle.contract_ref,
            evidence_refs=(evidence.id,),
            evaluator_results={"flat-evaluator": EvaluatorResult(passed=True, evidence_refs=(evidence.id,))},
            protected_results={"quality": EvaluatorResult(passed=True, evidence_refs=(evidence.id,))},
            objective_success=True,
            checkpoint_ref="candidate/checkpoint.pt",
        ),
        expected_state_revision=ingested.state.revision,
    )

    assert result.assessment.disposition == "promote"
    assert result.lifecycle.status == "promoted"
    assert result.state.baseline_ref == "run-candidate"
    assert coordinator.load_run("run-candidate").status == "promoted"


def test_coordinator_rolls_back_protected_capability_regression(monkeypatch, tmp_path: Path) -> None:
    coordinator, product, base = _make_coordinator(monkeypatch, tmp_path)
    proposal = _proposal()
    coordinator.approve_and_prepare(
        product,
        base,
        proposal,
        _request(),
        approved_by="human",
        run_id="run-unsafe",
        expected_state_revision=0,
        launch_request=_launch_request("run-unsafe"),
    )
    deployer = FakeDeployer()
    deployed = coordinator.deploy_prepared("run-unsafe", deployer, expected_state_revision=1)
    started = coordinator.start_deployed(
        "run-unsafe",
        FakeStarter(),
        expected_state_revision=deployed.state.revision,
    )
    evidence = EvidenceRecord(
        id="evidence:unsafe",
        kind="physical_diag",
        run_ref="run-unsafe",
        contract_ref=deployed.lifecycle.contract_ref,
        raw_artifact_ref="unsafe/diag.json",
    )
    ingested = coordinator.ingest_evidence(evidence, expected_state_revision=started.state.revision)
    result = coordinator.evaluate(
        EvaluationSubmission(
            evaluation_id="evaluation:unsafe",
            run_ref="run-unsafe",
            contract_ref=deployed.lifecycle.contract_ref,
            evidence_refs=(evidence.id,),
            evaluator_results={"flat-evaluator": EvaluatorResult(passed=False, evidence_refs=(evidence.id,))},
            protected_results={"quality": EvaluatorResult(passed=False, evidence_refs=(evidence.id,))},
            objective_success=False,
            terminal_failure=True,
        ),
        expected_state_revision=ingested.state.revision,
        deployer=deployer,
    )

    assert result.assessment.disposition == "rollback"
    assert result.lifecycle.status == "rolled_back"
    assert deployer.rolled_back == ["baseline-run"]
    assert result.state.active_run_ref == "baseline-run"


def test_approval_rejects_evidence_that_is_not_in_ledger(monkeypatch, tmp_path: Path) -> None:
    coordinator, product, base = _make_coordinator(monkeypatch, tmp_path)
    proposal = _proposal().model_copy(update={"evidence_refs": ("evidence:missing",)})
    request = _request()

    with pytest.raises(ResearchCoordinatorError, match="absent from the research ledger"):
        coordinator.approve_and_prepare(
            product,
            base,
            proposal,
            request,
            approved_by="human",
            run_id="run-missing-evidence",
            expected_state_revision=0,
        )


def test_mechanism_cycle_uses_same_contract_and_run_pipeline(monkeypatch, tmp_path: Path) -> None:
    coordinator, product, base = _make_coordinator(monkeypatch, tmp_path)
    mechanism_baseline = MechanismBundle(
        id="bundle:cycle-base",
        status="approved",
        contract_ref="coordinator-test@1",
        signals=(
            {
                "name": "task.progress",
                "description": "progress",
                "source_ref": "input.progress",
                "lower_bound": 0.0,
                "upper_bound": 1.0,
            },
        ),
        rewards=(
            RewardTermSpec(
                id="reward:progress",
                name="progress",
                role="positive_drive",
                expression=Expression.signal("task.progress"),
                intended_effect="drive progress",
                failure_region="no progress",
                success_region="progress",
            ),
        ),
        evaluator_refs=("flat-evaluator",),
    )
    intent = MechanismIntent(
        action="replace",
        target_kind="reward",
        target_id="reward:progress",
        value=mechanism_baseline.rewards[0].model_dump(mode="json"),
        causal_rationale="preserve a measurable drive",
        expected_effect="retain progress",
        falsification="flat evaluator does not pass",
    )
    synthesis_request = SynthesisRequest(
        id="cycle:quality",
        problem_ref="case:flat-quality",
        problem_statement="机制候选需要验证",
        baseline=mechanism_baseline,
        custom_intents=(intent,),
        protected_capabilities=("quality",),
        required_evaluators=("flat-evaluator",),
    )
    cycle = coordinator.propose_mechanism_cycle(
        synthesis_request,
        expected_state_revision=0,
        actor="test",
    )
    assert cycle.status == "candidate_selected"

    applied = coordinator.approve_mechanism_cycle(
        product,
        base,
        synthesis_request,
        cycle_id=cycle.cycle_id,
        evidence_summary={"quality": "baseline"},
        evidence_refs=["evidence:baseline"],
        approved_by="human",
        run_id="run-cycle",
        expected_state_revision=1,
    )
    assert applied.pipeline_result.run_manifest.is_file()
    assert coordinator.load_run("run-cycle").proposal_ref == "cycle:quality:approved"


def test_deployment_failure_is_recorded_without_changing_active_run(monkeypatch, tmp_path: Path) -> None:
    coordinator, product, base = _make_coordinator(monkeypatch, tmp_path)
    coordinator.approve_and_prepare(
        product,
        base,
        _proposal(),
        _request(),
        approved_by="human",
        run_id="run-deploy-failure",
        expected_state_revision=0,
    )

    with pytest.raises(ResearchCoordinatorError, match="remote deployment failed"):
        coordinator.deploy_prepared(
            "run-deploy-failure",
            FakeDeployer(deploy_fails=True),
            expected_state_revision=1,
        )

    assert coordinator.load_run("run-deploy-failure").status == "failed"
    state = coordinator.state_store.load()
    assert state.active_run_ref == "baseline-run"
    assert state.revision == 1


def test_pending_rollback_survives_coordinator_restart(monkeypatch, tmp_path: Path) -> None:
    coordinator, product, base = _make_coordinator(monkeypatch, tmp_path)
    coordinator.approve_and_prepare(
        product,
        base,
        _proposal(),
        _request(),
        approved_by="human",
        run_id="run-pending-rollback",
        expected_state_revision=0,
        launch_request=_launch_request("run-pending-rollback"),
    )
    deployed = coordinator.deploy_prepared(
        "run-pending-rollback",
        FakeDeployer(),
        expected_state_revision=1,
    )
    started = coordinator.start_deployed(
        "run-pending-rollback",
        FakeStarter(),
        expected_state_revision=deployed.state.revision,
    )
    evidence = EvidenceRecord(
        id="evidence:pending-rollback",
        kind="physical_diag",
        run_ref="run-pending-rollback",
        contract_ref=deployed.lifecycle.contract_ref,
        raw_artifact_ref="pending/diag.json",
    )
    ingested = coordinator.ingest_evidence(evidence, expected_state_revision=started.state.revision)
    evaluated = coordinator.evaluate(
        EvaluationSubmission(
            evaluation_id="evaluation:pending-rollback",
            run_ref="run-pending-rollback",
            contract_ref=deployed.lifecycle.contract_ref,
            evidence_refs=(evidence.id,),
            evaluator_results={
                "flat-evaluator": EvaluatorResult(
                    passed=False,
                    evidence_refs=(evidence.id,),
                )
            },
            protected_results={
                "quality": EvaluatorResult(
                    passed=False,
                    evidence_refs=(evidence.id,),
                )
            },
            terminal_failure=True,
        ),
        expected_state_revision=ingested.state.revision,
    )
    assert evaluated.lifecycle.status == "rollback_required"
    assert "rollback:run-pending-rollback" in evaluated.state.pending_interventions

    recovered = ResearchCoordinator(tmp_path / "output")
    assert recovered.load_run("run-pending-rollback").status == "rollback_required"
    receipt = recovered.execute_pending_rollback(
        "run-pending-rollback",
        FakeDeployer(),
        expected_state_revision=evaluated.state.revision,
    )
    assert receipt.status == "rolled_back"
    assert recovered.state_store.load().active_run_ref == "baseline-run"


def test_deployed_launch_requires_verified_training_start(monkeypatch, tmp_path: Path) -> None:
    coordinator, product, base = _make_coordinator(monkeypatch, tmp_path)
    run_id = "run-with-launch"
    coordinator.approve_and_prepare(
        product,
        base,
        _proposal(),
        _request(),
        approved_by="human",
        run_id=run_id,
        expected_state_revision=0,
        launch_request=TrainingLaunchRequest(
            payload_root="/remote/payload",
            run_id=run_id,
            total_steps=10,
            telemetry_interval=1,
            num_envs=1,
        ),
    )
    deployed = coordinator.deploy_prepared(
        run_id,
        FakeDeployer(),
        expected_state_revision=1,
    )
    assert deployed.lifecycle.status == "deployed"

    recovered = ResearchCoordinator(tmp_path / "output")
    started = recovered.start_deployed(
        run_id,
        FakeStarter(),
        expected_state_revision=deployed.state.revision,
    )

    assert started.result.status == "started"
    assert started.lifecycle.status == "observing"
    assert started.receipt.status == "started"
    assert started.lifecycle.training_start_receipt_ref == started.receipt.id
    assert started.state.observation_window is not None
    assert started.state.observation_window.status == "observing"
    assert started.state.revision == deployed.state.revision + 1


def test_failed_training_start_keeps_deployed_run_and_active_state(monkeypatch, tmp_path: Path) -> None:
    coordinator, product, base = _make_coordinator(monkeypatch, tmp_path)
    run_id = "run-start-failure"
    coordinator.approve_and_prepare(
        product,
        base,
        _proposal(),
        _request(),
        approved_by="human",
        run_id=run_id,
        expected_state_revision=0,
        launch_request=TrainingLaunchRequest(
            payload_root="/remote/payload",
            run_id=run_id,
            total_steps=10,
            telemetry_interval=1,
            num_envs=1,
        ),
    )
    deployed = coordinator.deploy_prepared(
        run_id,
        FakeDeployer(),
        expected_state_revision=1,
    )
    started = coordinator.start_deployed(
        run_id,
        FakeStarter(failed=True),
        expected_state_revision=deployed.state.revision,
    )

    assert started.result.status == "failed"
    assert started.receipt.status == "failed"
    assert started.lifecycle.status == "deployed"
    assert started.state.revision == deployed.state.revision
    assert started.state.active_run_ref == run_id
    assert coordinator.load_run(run_id).status == "deployed"


def test_deployed_without_launch_plan_cannot_be_started(monkeypatch, tmp_path: Path) -> None:
    coordinator, product, base = _make_coordinator(monkeypatch, tmp_path)
    run_id = "run-deploy-only"
    coordinator.approve_and_prepare(
        product,
        base,
        _proposal(),
        _request(),
        approved_by="human",
        run_id=run_id,
        expected_state_revision=0,
    )
    deployed = coordinator.deploy_prepared(
        run_id,
        FakeDeployer(),
        expected_state_revision=1,
    )

    with pytest.raises(ResearchCoordinatorError, match="no training launch plan"):
        coordinator.start_deployed(
            run_id,
            FakeStarter(),
            expected_state_revision=deployed.state.revision,
        )


def test_start_rejects_tampered_materialized_launch_plan(monkeypatch, tmp_path: Path) -> None:
    coordinator, product, base = _make_coordinator(monkeypatch, tmp_path)
    run_id = "run-tampered-launch"
    coordinator.approve_and_prepare(
        product,
        base,
        _proposal(),
        _request(),
        approved_by="human",
        run_id=run_id,
        expected_state_revision=0,
        launch_request=_launch_request(run_id),
    )
    deployed = coordinator.deploy_prepared(run_id, FakeDeployer(), expected_state_revision=1)
    manifest_path = Path(deployed.lifecycle.pipeline_manifest_ref)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["launch_plan"]["argv"][0] = "tampered-python"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    class UnexpectedStarter:
        called = False

        def start(self, plan):
            self.called = True
            return FakeStarter().start(plan)

    starter = UnexpectedStarter()
    with pytest.raises(ResearchCoordinatorError, match="digest"):
        coordinator.start_deployed(
            run_id,
            starter,
            expected_state_revision=deployed.state.revision,
        )

    assert not starter.called
    assert coordinator.load_run(run_id).status == "deployed"


def test_invalid_training_starter_result_keeps_deployed_state(monkeypatch, tmp_path: Path) -> None:
    coordinator, product, base = _make_coordinator(monkeypatch, tmp_path)
    run_id = "run-invalid-start-result"
    coordinator.approve_and_prepare(
        product,
        base,
        _proposal(),
        _request(),
        approved_by="human",
        run_id=run_id,
        expected_state_revision=0,
        launch_request=_launch_request(run_id),
    )
    deployed = coordinator.deploy_prepared(run_id, FakeDeployer(), expected_state_revision=1)

    class InvalidStarter:
        def start(self, plan):
            return object()

    started = coordinator.start_deployed(
        run_id,
        InvalidStarter(),
        expected_state_revision=deployed.state.revision,
    )

    assert started.result.status == "failed"
    assert "invalid result type" in started.result.error
    assert started.lifecycle.status == "deployed"
    assert started.receipt.status == "failed"


def test_start_reconciles_after_state_revision_race_without_second_remote_start(
    monkeypatch,
    tmp_path: Path,
) -> None:
    coordinator, product, base = _make_coordinator(monkeypatch, tmp_path)
    run_id = "run-start-race"
    coordinator.approve_and_prepare(
        product,
        base,
        _proposal(),
        _request(),
        approved_by="human",
        run_id=run_id,
        expected_state_revision=0,
        launch_request=_launch_request(run_id),
    )
    deployed = coordinator.deploy_prepared(run_id, FakeDeployer(), expected_state_revision=1)

    original_compare_and_set = coordinator.state_store.compare_and_set
    first_call = True

    def race(expected_revision, update, *, actor, reason):
        nonlocal first_call
        if first_call:
            first_call = False

            def unrelated(current):
                current.unresolved_questions.append("race-observed")
                return current

            current = coordinator.state_store.load()
            raced = original_compare_and_set(
                current.revision,
                unrelated,
                actor="race-test",
                reason="simulate unrelated state update",
            )
            raise RevisionConflict(expected_revision, raced.revision)
        return original_compare_and_set(
            expected_revision,
            update,
            actor=actor,
            reason=reason,
        )

    monkeypatch.setattr(coordinator.state_store, "compare_and_set", race)
    starter = FakeStarter()
    started = coordinator.start_deployed(
        run_id,
        starter,
        expected_state_revision=deployed.state.revision,
    )

    assert started.lifecycle.status == "observing"
    assert started.state.observation_window is not None
    assert started.state.observation_window.status == "observing"
    assert len(starter.plans) == 1


def test_started_receipt_is_not_reused_to_reactivate_replaced_run(monkeypatch, tmp_path: Path) -> None:
    coordinator, product, base = _make_coordinator(monkeypatch, tmp_path)
    run_id = "run-replaced-before-reconcile"
    coordinator.approve_and_prepare(
        product,
        base,
        _proposal(),
        _request(),
        approved_by="human",
        run_id=run_id,
        expected_state_revision=0,
        launch_request=_launch_request(run_id),
    )
    deployed = coordinator.deploy_prepared(run_id, FakeDeployer(), expected_state_revision=1)

    original_compare_and_set = coordinator.state_store.compare_and_set
    first_call = True

    def race(expected_revision, update, *, actor, reason):
        nonlocal first_call
        if first_call:
            first_call = False

            def replace_active(current):
                current.active_run_ref = "newer-run"
                current.contract_ref = "newer-contract"
                return current

            current = coordinator.state_store.load()
            replaced = original_compare_and_set(
                current.revision,
                replace_active,
                actor="race-test",
                reason="simulate newer active run",
            )
            raise RevisionConflict(expected_revision, replaced.revision)
        return original_compare_and_set(
            expected_revision,
            update,
            actor=actor,
            reason=reason,
        )

    monkeypatch.setattr(coordinator.state_store, "compare_and_set", race)
    with pytest.raises(ResearchCoordinatorError, match="reconcile training start receipt"):
        coordinator.start_deployed(
            run_id,
            FakeStarter(),
            expected_state_revision=deployed.state.revision,
        )

    assert coordinator.load_run(run_id).status == "deployed"
    current = coordinator.state_store.load()
    assert current.active_run_ref == "newer-run"

    # A retry must fail before invoking a starter, rather than creating a second
    # tmux session or silently reactivating the replaced run.
    retry_starter = FakeStarter()
    with pytest.raises(ResearchCoordinatorError, match="no longer points"):
        coordinator.start_deployed(
            run_id,
            retry_starter,
            expected_state_revision=current.revision,
        )
    assert retry_starter.plans == []
