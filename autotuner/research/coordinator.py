"""产品无关的强化学习研究协调器。

协调器把受限 LLM 提案、任务合同流水线、远程部署、证据、裁决和回滚接成
一个可恢复闭环。它不实现机器人奖励、训练框架或远程命令，也不允许 LLM
越过合同审批与确定性评估直接执行。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence
import uuid

from pydantic import BaseModel, ConfigDict, Field

from autotuner.execution import (
    DeploymentResult,
    DeploymentSpec,
    RemoteArtifact,
    TRAINING_START_SCHEMA,
    TrainingStartResult,
)
from autotuner.llm_gateway.research_proposal import (
    AppliedResearchProposal,
    ResearchProposal,
    ResearchProposalRequest,
    ResearchProposalResult,
    apply_research_proposal,
    propose_research_change,
)
from autotuner.mechanisms.mechanism_synthesis import SynthesisRequest
from autotuner.product import (
    ResolvedProductContract,
    TaskContractBundle,
    TaskExecutionPipeline,
    TrainingLaunchPlan,
    TrainingLaunchRequest,
)
from .research_cycle import ResearchCycleManager, ResearchCycleProposal
from .research_ledger import (
    DecisionRecord,
    DeploymentReceipt,
    EvidenceRecord,
    ExperimentOutcome,
    LedgerEvent,
    ResearchLedgerStore,
    ResearchRunLifecycle,
    TrainingStartReceipt,
)
from .research_state import (
    FactState,
    ObservationWindowState,
    ProtectedCapabilityState,
    ResearchCaseState,
    ResearchState,
    ResearchStateStore,
    RevisionConflict,
)


COORDINATOR_SCHEMA = "rl-agent.research-coordinator/v1"


class ResearchCoordinatorError(RuntimeError):
    """研究闭环的身份、证据或状态转换不成立。"""


class ResearchDeployer(Protocol):
    def deploy_spec(self, spec: DeploymentSpec) -> DeploymentResult: ...

    def rollback_active(self, run_id: str) -> RemoteArtifact: ...


class ResearchTrainingStarter(Protocol):
    """Minimal boundary for starting a materialized training plan."""

    def start(self, plan: TrainingLaunchPlan) -> TrainingStartResult: ...


class EvaluatorResult(BaseModel):
    """一个独立评估器的结构化结论；``None`` 表示尚无结论。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool | None = None
    score: float | None = None
    threshold: float | None = None
    evidence_refs: tuple[str, ...] = ()
    details: dict[str, Any] = Field(default_factory=dict)


class EvaluationSubmission(BaseModel):
    """候选运行的一次可审计验收输入。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = COORDINATOR_SCHEMA
    evaluation_id: str
    run_ref: str
    contract_ref: str
    checkpoint_ref: str = ""
    evidence_refs: tuple[str, ...]
    evaluator_results: dict[str, EvaluatorResult] = Field(default_factory=dict)
    protected_results: dict[str, EvaluatorResult] = Field(default_factory=dict)
    objective_success: bool | None = None
    terminal_failure: bool = False
    notes: tuple[str, ...] = ()


class DispositionAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    disposition: Literal["promote", "continue", "rollback"]
    rationale: tuple[str, ...]
    missing_evaluators: tuple[str, ...] = ()
    missing_protected_results: tuple[str, ...] = ()
    failed_evaluators: tuple[str, ...] = ()
    regressed_capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class CoordinatedDeployment:
    lifecycle: ResearchRunLifecycle
    receipt: DeploymentReceipt
    result: DeploymentResult
    state: ResearchState


@dataclass(frozen=True)
class CoordinatedTrainingStart:
    lifecycle: ResearchRunLifecycle
    receipt: TrainingStartReceipt
    result: TrainingStartResult
    state: ResearchState


@dataclass(frozen=True)
class EvidenceIngestion:
    evidence: EvidenceRecord
    event: LedgerEvent
    lifecycle: ResearchRunLifecycle
    state: ResearchState


@dataclass(frozen=True)
class CoordinatedEvaluation:
    assessment: DispositionAssessment
    outcome: ExperimentOutcome
    decision: DecisionRecord
    lifecycle: ResearchRunLifecycle
    state: ResearchState
    rollback_receipt: DeploymentReceipt | None = None


def _canonical(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ResearchCoordinator:
    """协调一条合同驱动、证据闭环且可远程回滚的研究路线。"""

    def __init__(
        self,
        output_root: str | Path = "output",
        *,
        pipeline: TaskExecutionPipeline | None = None,
        state_store: ResearchStateStore | None = None,
        ledger: ResearchLedgerStore | None = None,
        cycle_manager: ResearchCycleManager | None = None,
    ) -> None:
        self.output_root = Path(output_root)
        pipeline_ledger = pipeline.ledger if pipeline is not None else None
        self.ledger = ledger or pipeline_ledger or ResearchLedgerStore(
            self.output_root / "research" / "ledger"
        )
        if pipeline_ledger is not None and Path(pipeline_ledger.root) != Path(self.ledger.root):
            raise ResearchCoordinatorError("pipeline and coordinator must share one research ledger")
        self.pipeline = pipeline or TaskExecutionPipeline(
            self.output_root,
            ledger=self.ledger,
        )
        self.state_store = state_store or ResearchStateStore(
            self.output_root / "research" / "state"
        )
        self.cycle_manager = cycle_manager or ResearchCycleManager(
            self.output_root / "research" / "cycles",
            state_store=self.state_store,
            ledger=self.ledger,
        )
        self.proposals_root = self.output_root / "research" / "proposals"

    def initialize(self, state: ResearchState, *, actor: str) -> ResearchState:
        """显式初始化权威状态；不从目录名或最近文件猜当前研究上下文。"""
        return self.state_store.initialize(state, actor=actor)

    @staticmethod
    def _require_revision(state: ResearchState, expected: int) -> None:
        if state.revision != expected:
            raise RevisionConflict(expected, state.revision)

    def _proposal_path(self, proposal_id: str) -> Path:
        digest = hashlib.sha256(proposal_id.encode("utf-8")).hexdigest()[:20]
        return self.proposals_root / f"{digest}.json"

    def _save_proposal(self, proposal: ResearchProposal) -> Path:
        path = self._proposal_path(proposal.proposal_id)
        payload = proposal.model_dump(mode="json")
        if path.is_file():
            if json.loads(path.read_text(encoding="utf-8")) != payload:
                raise ResearchCoordinatorError(
                    f"proposal id already exists with different content: {proposal.proposal_id}"
                )
            return path
        _atomic_json(path, payload)
        return path

    def load_proposal(self, proposal_id: str) -> ResearchProposal:
        path = self._proposal_path(proposal_id)
        if not path.is_file():
            raise KeyError(f"research proposal not found: {proposal_id}")
        return ResearchProposal.model_validate_json(path.read_text(encoding="utf-8"))

    def propose(
        self,
        request: ResearchProposalRequest,
        *,
        expected_state_revision: int,
        actor: str,
    ) -> ResearchProposalResult:
        """让 LLM 生成受限候选，并把候选而非执行动作写入研究状态。"""
        state = self.state_store.load()
        self._require_revision(state, expected_state_revision)
        result = propose_research_change(request)
        if not result.ready or result.proposal is None:
            return result
        proposal = result.proposal
        self._save_proposal(proposal)
        decision = DecisionRecord(
            id=f"decision:proposal:{proposal.proposal_id}",
            trigger="evidence-driven restricted LLM proposal",
            problem_statement=proposal.problem_statement,
            evidence_refs=list(proposal.evidence_refs),
            hypotheses_considered=list(proposal.hypothesis_refs),
            selected_experiment_ref=proposal.proposal_id,
            check_results=[
                {"prediction": value, "status": "pending"}
                for value in proposal.falsifiable_predictions
            ],
            status="proposed",
            source="research_coordinator",
        )
        self.ledger.append("decision", decision, actor=actor)

        def update(current: ResearchState) -> ResearchState:
            current.pending_interventions = list(
                dict.fromkeys(current.pending_interventions + [proposal.proposal_id])
            )
            case = current.cases.get(proposal.problem_ref)
            if case is None:
                case = ResearchCaseState(
                    id=proposal.problem_ref,
                    problem_statement=proposal.problem_statement,
                    evidence_refs=list(proposal.evidence_refs),
                    hypothesis_refs=list(proposal.hypothesis_refs),
                    candidate_refs=[proposal.proposal_id],
                    status="proposed",
                )
                current.cases[case.id] = case
            else:
                case.evidence_refs = list(dict.fromkeys(case.evidence_refs + list(proposal.evidence_refs)))
                case.hypothesis_refs = list(
                    dict.fromkeys(case.hypothesis_refs + list(proposal.hypothesis_refs))
                )
                case.candidate_refs = list(dict.fromkeys(case.candidate_refs + [proposal.proposal_id]))
                case.status = "proposed"
            current.active_case_ref = case.id
            current.last_decision_ref = decision.id
            return current

        self.state_store.compare_and_set(
            expected_state_revision,
            update,
            actor=actor,
            reason=f"record restricted research proposal {proposal.proposal_id}",
        )
        return result

    def propose_mechanism_cycle(
        self,
        request: SynthesisRequest,
        *,
        expected_state_revision: int,
        actor: str,
        validation_context: Any | None = None,
    ) -> ResearchCycleProposal:
        """调用确定性机制合成器，保留候选比较和拒绝理由。"""
        state = self.state_store.load()
        self._require_revision(state, expected_state_revision)
        return self.cycle_manager.propose(
            request,
            expected_state_revision=expected_state_revision,
            actor=actor,
            validation_context=validation_context,
        )

    def approve_mechanism_cycle(
        self,
        product: ResolvedProductContract,
        base: TaskContractBundle,
        request: SynthesisRequest,
        *,
        cycle_id: str,
        evidence_summary: Mapping[str, Any],
        evidence_refs: Sequence[str],
        approved_by: str,
        run_id: str,
        expected_state_revision: int,
        actor: str = "research-coordinator",
        launch_request: TrainingLaunchRequest | Mapping[str, Any] | None = None,
        asset_reuse_approval_refs: Sequence[str] = (),
    ) -> AppliedResearchProposal:
        """人工批准机制合成候选，并走同一合同/payload/deploy 入口。"""
        cycle = self.cycle_manager.load_proposal(cycle_id)
        if cycle.status != "candidate_selected" or not cycle.selected_candidate_ref:
            raise ResearchCoordinatorError(
                f"mechanism cycle is not executable: {cycle.status}"
            )
        if cycle.problem_ref != request.problem_ref or cycle.baseline_bundle_ref != request.baseline.id:
            raise ResearchCoordinatorError("mechanism cycle does not match the supplied synthesis request")
        selected = next(
            (
                item
                for item in cycle.assessments
                if item.selected and item.candidate.id == cycle.selected_candidate_ref
            ),
            None,
        )
        if selected is None or not selected.validation.ok:
            raise ResearchCoordinatorError("selected mechanism cycle candidate is not valid")
        patch = selected.candidate.patch
        if patch.baseline_fingerprint != request.baseline.fingerprint():
            raise ResearchCoordinatorError("mechanism cycle candidate baseline has changed")
        refs = tuple(
            dict.fromkeys(
                str(value)
                for value in (
                    *evidence_refs,
                    *(ref for gap in request.gaps for ref in gap.evidence_refs),
                )
                if str(value).strip()
            )
        )
        if not refs:
            raise ResearchCoordinatorError("mechanism cycle approval requires evidence_refs")
        if not selected.candidate.falsification:
            raise ResearchCoordinatorError("selected mechanism candidate has no falsifiable prediction")
        proposal = ResearchProposal(
            proposal_id=f"{cycle.cycle_id}:approved",
            problem_ref=request.problem_ref,
            problem_statement=request.problem_statement,
            evidence_refs=refs,
            hypothesis_refs=patch.hypothesis_refs,
            task_changes={
                "training": {
                    "mechanism_proposals": [
                        {
                            "patch": patch.model_dump(mode="json"),
                            "validation": selected.validation.model_dump(mode="json"),
                            "baseline_fingerprint": patch.baseline_fingerprint,
                        }
                    ]
                }
            },
            expected_effects=selected.candidate.predictions or patch.expected_effects,
            falsifiable_predictions=selected.candidate.falsification,
            protected_capabilities=request.protected_capabilities,
            required_evaluators=request.required_evaluators,
            confidence=1.0,
        )
        proposal_request = ResearchProposalRequest(
            problem_ref=request.problem_ref,
            problem_statement=request.problem_statement,
            evidence_refs=refs,
            evidence_summary=evidence_summary,
            protected_capabilities=request.protected_capabilities,
            required_evaluators=request.required_evaluators,
            baseline_bundle_ref=request.baseline.id,
            baseline_mechanism=request.baseline,
        )
        return self.approve_and_prepare(
            product,
            base,
            proposal,
            proposal_request,
            approved_by=approved_by,
            run_id=run_id,
            expected_state_revision=expected_state_revision,
            actor=actor,
            launch_request=launch_request,
            asset_reuse_approval_refs=asset_reuse_approval_refs,
        )

    def _append_lifecycle(self, lifecycle: ResearchRunLifecycle, *, actor: str) -> LedgerEvent:
        existing = self.ledger.latest("research_run", lifecycle.id)
        return self.ledger.append(
            "research_run",
            lifecycle,
            actor=actor,
            event_type="supersede" if existing is not None else "append",
        )

    def _load_lifecycle(self, run_ref: str) -> ResearchRunLifecycle:
        value = self.ledger.latest("research_run", f"research-run:{run_ref}")
        if value is None:
            raise KeyError(f"coordinated research run not found: {run_ref}")
        return ResearchRunLifecycle.model_validate(value)

    def load_run(self, run_ref: str) -> ResearchRunLifecycle:
        """从账本恢复一次候选运行，不依赖聊天上下文或内存对象。"""
        return self._load_lifecycle(run_ref)

    def approve_and_prepare(
        self,
        product: ResolvedProductContract,
        base: TaskContractBundle,
        proposal: ResearchProposal,
        request: ResearchProposalRequest,
        *,
        approved_by: str,
        run_id: str,
        expected_state_revision: int,
        actor: str = "research-coordinator",
        launch_request: TrainingLaunchRequest | Mapping[str, Any] | None = None,
        asset_reuse_approval_refs: Sequence[str] = (),
    ) -> AppliedResearchProposal:
        """批准候选并生成下一次运行；此步骤不触碰远程主机。"""
        state = self.state_store.load()
        self._require_revision(state, expected_state_revision)
        if self.ledger.latest("research_run", f"research-run:{run_id}") is not None:
            raise ResearchCoordinatorError(f"coordinated research run already exists: {run_id}")
        self._save_proposal(proposal)
        unknown_evidence = [
            evidence_ref
            for evidence_ref in proposal.evidence_refs
            if self.ledger.latest("evidence", evidence_ref) is None
        ]
        if unknown_evidence:
            raise ResearchCoordinatorError(
                "proposal cites evidence absent from the research ledger: "
                + ", ".join(unknown_evidence)
            )
        applied = apply_research_proposal(
            self.pipeline,
            product,
            base,
            proposal,
            request,
            approved_by=approved_by,
            run_id=run_id,
            actor=actor,
            launch_request=launch_request,
            asset_reuse_approval_refs=asset_reuse_approval_refs,
        )
        lifecycle = ResearchRunLifecycle(
            id=f"research-run:{run_id}",
            run_ref=run_id,
            contract_ref=applied.stored_contract.ref,
            parent_contract_ref=applied.stored_contract.parent_ref,
            proposal_ref=proposal.proposal_id,
            pipeline_manifest_ref=str(applied.pipeline_result.run_manifest),
            required_evaluators=list(proposal.required_evaluators),
            protected_capabilities=list(proposal.protected_capabilities),
            previous_active_run_ref=state.active_run_ref,
            previous_checkpoint_ref=state.active_checkpoint_ref,
            previous_contract_ref=state.contract_ref,
            status="prepared",
            launch_plan_digest=(
                self._training_plan_digest(applied.pipeline_result.launch_plan)
                if applied.pipeline_result.launch_plan is not None
                else ""
            ),
            source="research_coordinator",
        )
        self._append_lifecycle(lifecycle, actor=actor)
        decision_id = f"decision:proposal:{proposal.proposal_id}"
        previous_decision = self.ledger.latest("decision", decision_id)
        approved_decision = DecisionRecord(
            id=decision_id,
            trigger="human approval of restricted research proposal",
            problem_statement=proposal.problem_statement,
            evidence_refs=list(proposal.evidence_refs),
            hypotheses_considered=list(proposal.hypothesis_refs),
            selected_experiment_ref=run_id,
            check_results=[
                {
                    "mechanism_validation": (
                        applied.mechanism_validation.ok
                        if applied.mechanism_validation is not None
                        else "not_requested"
                    )
                }
            ],
            authorization_ref=f"human:{approved_by}",
            status="approved",
            source="research_coordinator",
        )
        self.ledger.append(
            "decision",
            approved_decision,
            actor=actor,
            event_type="supersede" if previous_decision is not None else "append",
        )

        def update(current: ResearchState) -> ResearchState:
            current.contract_ref = lifecycle.contract_ref
            current.pending_interventions = list(
                dict.fromkeys(current.pending_interventions + [proposal.proposal_id, run_id])
            )
            case = current.cases.get(proposal.problem_ref)
            if case is not None:
                case.experiment_refs = list(dict.fromkeys(case.experiment_refs + [run_id]))
                case.status = "proposed"
            current.last_decision_ref = approved_decision.id
            return current

        self.state_store.compare_and_set(
            expected_state_revision,
            update,
            actor=actor,
            reason=f"approve and prepare research run {run_id}",
        )
        return applied

    @staticmethod
    def _deployment_spec(manifest_ref: str) -> DeploymentSpec:
        manifest = json.loads(Path(manifest_ref).read_text(encoding="utf-8"))
        value = manifest.get("deployment")
        if not isinstance(value, Mapping):
            raise ResearchCoordinatorError("run manifest has no deployment specification")
        spec = DeploymentSpec(**dict(value))
        spec.validate()
        return spec

    @staticmethod
    def _deployment_receipt(
        lifecycle: ResearchRunLifecycle,
        result: DeploymentResult,
    ) -> DeploymentReceipt:
        return DeploymentReceipt(
            id=f"deployment:{lifecycle.run_ref}:{uuid.uuid4().hex}",
            run_ref=lifecycle.run_ref,
            contract_ref=lifecycle.contract_ref,
            operation="deploy",
            status="activated" if result.status == "activated" else "failed",
            previous_active_run_ref=lifecycle.previous_active_run_ref,
            runtime_ref=result.runtime.identity,
            payload_ref=result.payload.identity,
            remote_run_ref=result.run.identity,
            trace=[dict(item) for item in result.trace],
            error=(
                "; ".join(result.errors)
                if result.errors
                else ("" if result.status == "activated" else "remote deployment did not activate")
            ),
            source="research_coordinator",
        )

    @staticmethod
    def _training_plan(manifest_ref: str) -> TrainingLaunchPlan | None:
        """Load the immutable launch plan produced by the product pipeline."""
        manifest = json.loads(Path(manifest_ref).read_text(encoding="utf-8"))
        value = manifest.get("launch_plan")
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ResearchCoordinatorError("run manifest launch_plan must be a mapping")
        data = dict(value)
        argv = data.get("argv")
        environment = data.get("environment")
        if not isinstance(argv, (list, tuple)) or not argv or not isinstance(environment, Mapping):
            raise ResearchCoordinatorError("run manifest launch_plan is incomplete")
        if any(
            not isinstance(item, str) or any(char in item for char in ("\x00", "\r", "\n"))
            for item in argv
        ):
            raise ResearchCoordinatorError("run manifest launch_plan argv is invalid")
        for key, value in environment.items():
            if not isinstance(key, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None:
                raise ResearchCoordinatorError("run manifest launch_plan environment name is invalid")
            if not isinstance(value, str) or any(char in value for char in ("\x00", "\r", "\n")):
                raise ResearchCoordinatorError("run manifest launch_plan environment value is invalid")
        data["argv"] = tuple(str(item) for item in argv)
        data["environment"] = {
            str(key): str(item) for key, item in environment.items()
        }
        try:
            return TrainingLaunchPlan(**data)
        except (TypeError, ValueError) as exc:
            raise ResearchCoordinatorError(
                f"run manifest launch_plan is invalid: {type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _training_plan_digest(plan: TrainingLaunchPlan | None) -> str:
        if plan is None:
            return ""
        return hashlib.sha256(_canonical(plan.to_dict()).encode("utf-8")).hexdigest()

    @staticmethod
    def _failed_training_start(
        plan: TrainingLaunchPlan,
        error: str,
        *,
        trace: Sequence[Mapping[str, Any]] = (),
    ) -> TrainingStartResult:
        return TrainingStartResult(
            status="failed",
            run_id=plan.run_id,
            run_dir=plan.run_dir,
            handle_ref=f"tmux:{plan.tmux_session}",
            process_pattern=plan.process_pattern,
            trace=tuple(dict(item) for item in trace if isinstance(item, Mapping)),
            error=error,
        )

    @classmethod
    def _validate_training_start_result(
        cls,
        plan: TrainingLaunchPlan,
        result: object,
    ) -> TrainingStartResult:
        """将启动器输出收敛为与不可变计划一致的结果。"""
        if not isinstance(result, TrainingStartResult):
            return cls._failed_training_start(
                plan,
                "training starter returned an invalid result type",
            )
        if result.schema_version != TRAINING_START_SCHEMA:
            return cls._failed_training_start(
                plan,
                "training starter returned an unsupported result schema",
            )
        if result.status not in {"started", "failed"}:
            return cls._failed_training_start(plan, "training starter returned an invalid status")
        if result.run_id != plan.run_id:
            return cls._failed_training_start(
                plan,
                "training starter returned a different run_id",
                trace=result.trace if isinstance(result.trace, (list, tuple)) else (),
            )
        if result.run_dir != plan.run_dir or result.process_pattern != plan.process_pattern:
            return cls._failed_training_start(
                plan,
                "training starter returned plan identity fields that do not match",
                trace=result.trace if isinstance(result.trace, (list, tuple)) else (),
            )
        if not isinstance(result.trace, (list, tuple)) or any(
            not isinstance(item, Mapping) for item in result.trace
        ):
            return cls._failed_training_start(plan, "training starter returned an invalid trace")
        if result.status == "started" and result.handle_ref != f"tmux:{plan.tmux_session}":
            return cls._failed_training_start(
                plan,
                "training starter returned an unexpected remote handle",
                trace=result.trace,
            )
        if result.status == "failed" and not result.error:
            return cls._failed_training_start(
                plan,
                "training starter reported failure without an error",
                trace=result.trace,
            )
        return result

    @staticmethod
    def _training_start_receipt(
        lifecycle: ResearchRunLifecycle,
        plan: TrainingLaunchPlan,
        result: TrainingStartResult,
    ) -> TrainingStartReceipt:
        plan_digest = hashlib.sha256(
            _canonical(plan.to_dict()).encode("utf-8")
        ).hexdigest()
        return TrainingStartReceipt(
            id=f"training-start:{lifecycle.run_ref}:{uuid.uuid4().hex}",
            run_ref=lifecycle.run_ref,
            contract_ref=lifecycle.contract_ref,
            deployment_receipt_ref=lifecycle.deployment_receipt_ref,
            status=result.status,
            remote_run_ref=result.run_id,
            run_dir=result.run_dir,
            handle_ref=result.handle_ref,
            process_pattern=result.process_pattern,
            plan_digest=plan_digest,
            trace=[dict(item) for item in result.trace],
            error=result.error,
            source="research_coordinator",
        )

    def _latest_training_start_receipt(
        self,
        run_ref: str,
    ) -> TrainingStartReceipt | None:
        candidates = [
            TrainingStartReceipt.model_validate(value)
            for value in self.ledger.records("training_start_receipt")
            if str(value.get("run_ref") or "") == run_ref
        ]
        return candidates[-1] if candidates else None

    @staticmethod
    def _result_from_started_receipt(
        receipt: TrainingStartReceipt,
    ) -> TrainingStartResult:
        return TrainingStartResult(
            status="started",
            run_id=receipt.remote_run_ref,
            run_dir=receipt.run_dir,
            handle_ref=receipt.handle_ref,
            process_pattern=receipt.process_pattern,
            trace=tuple(dict(item) for item in receipt.trace),
        )

    def _finalize_started_training(
        self,
        lifecycle: ResearchRunLifecycle,
        plan: TrainingLaunchPlan,
        receipt: TrainingStartReceipt,
        result: TrainingStartResult,
        *,
        expected_state_revision: int,
        actor: str,
    ) -> CoordinatedTrainingStart:
        """把已验证的启动回执幂等地落到当前状态。"""
        plan_digest = self._training_plan_digest(plan)
        if receipt.plan_digest != plan_digest:
            raise ResearchCoordinatorError(
                "training start receipt does not match the materialized launch plan"
            )
        observing = lifecycle.model_copy(
            update={
                "training_start_receipt_ref": receipt.id,
                "status": "observing",
            }
        )

        def update(current: ResearchState) -> ResearchState:
            current.active_run_ref = observing.run_ref
            current.contract_ref = observing.contract_ref
            current.observation_window = ObservationWindowState(
                id=f"observation:{observing.run_ref}",
                run_ref=observing.run_ref,
                checkpoint_start_ref=current.active_checkpoint_ref,
                required_evidence=[
                    f"evaluator:{value}" for value in observing.required_evaluators
                ],
                status="observing",
            )
            case = current.cases.get(current.active_case_ref)
            if case is not None:
                case.status = "experimenting"
            return current

        state = self.state_store.load()
        self._require_revision(state, expected_state_revision)
        if (
            state.active_run_ref != lifecycle.run_ref
            or state.contract_ref != lifecycle.contract_ref
        ):
            raise ResearchCoordinatorError(
                "research state no longer points to this deployed run; "
                f"reconcile training start receipt {receipt.id} without reactivating it"
            )
        try:
            updated = self.state_store.compare_and_set(
                expected_state_revision,
                update,
                actor=actor,
                reason=f"start deployed research run {lifecycle.run_ref}",
            )
        except RevisionConflict as conflict:
            current = self.state_store.load()
            if (
                current.active_run_ref != observing.run_ref
                or current.contract_ref != observing.contract_ref
            ):
                raise ResearchCoordinatorError(
                    "training started but research state changed; "
                    f"reconcile training start receipt {receipt.id} before retrying"
                ) from conflict
            if (
                current.observation_window is not None
                and current.observation_window.status == "observing"
            ):
                updated = current
            else:
                try:
                    updated = self.state_store.compare_and_set(
                        current.revision,
                        update,
                        actor=actor,
                        reason=f"recover start of deployed research run {lifecycle.run_ref}",
                    )
                except RevisionConflict as retry_conflict:
                    raise ResearchCoordinatorError(
                        "training started but research state could not be reconciled; "
                        f"reconcile training start receipt {receipt.id}"
                    ) from retry_conflict

        # 状态成功后才追加 observing 生命周期。若进程在此处被中断，下一次
        # start_deployed 会依据成功回执走同一 finalize 路径，而不会再次启动远程进程。
        latest = self._load_lifecycle(lifecycle.run_ref)
        if latest.status == "deployed":
            self._append_lifecycle(observing, actor=actor)
            latest = observing
        elif latest.status == "observing":
            if latest.training_start_receipt_ref != receipt.id:
                raise ResearchCoordinatorError(
                    "research run already has a different verified training start receipt"
                )
            observing = latest
        else:
            raise ResearchCoordinatorError(
                f"research run lifecycle changed during training start: {latest.status}"
            )
        return CoordinatedTrainingStart(observing, receipt, result, updated)

    def deploy_prepared(
        self,
        run_ref: str,
        deployer: ResearchDeployer,
        *,
        expected_state_revision: int,
        actor: str = "research-coordinator",
    ) -> CoordinatedDeployment:
        """部署已物料化运行，并在远程激活成功后更新 active state。"""
        state = self.state_store.load()
        self._require_revision(state, expected_state_revision)
        lifecycle = self._load_lifecycle(run_ref)
        if lifecycle.status != "prepared":
            raise ResearchCoordinatorError(
                f"research run cannot deploy from status {lifecycle.status}"
            )
        try:
            result = deployer.deploy_spec(self._deployment_spec(lifecycle.pipeline_manifest_ref))
        except Exception as exc:
            receipt = DeploymentReceipt(
                id=f"deployment:{lifecycle.run_ref}:{uuid.uuid4().hex}",
                run_ref=lifecycle.run_ref,
                contract_ref=lifecycle.contract_ref,
                operation="deploy",
                status="failed",
                previous_active_run_ref=lifecycle.previous_active_run_ref,
                error=f"{type(exc).__name__}: {exc}",
                source="research_coordinator",
            )
            self.ledger.append("deployment_receipt", receipt, actor=actor)
            failed = lifecycle.model_copy(
                update={"deployment_receipt_ref": receipt.id, "status": "failed"}
            )
            self._append_lifecycle(failed, actor=actor)
            raise ResearchCoordinatorError(receipt.error) from exc
        receipt = self._deployment_receipt(lifecycle, result)
        self.ledger.append("deployment_receipt", receipt, actor=actor)
        if result.status != "activated":
            failed = lifecycle.model_copy(
                update={"deployment_receipt_ref": receipt.id, "status": "failed"}
            )
            self._append_lifecycle(failed, actor=actor)
            raise ResearchCoordinatorError(receipt.error or "remote deployment did not activate")
        deployed = lifecycle.model_copy(
            update={"deployment_receipt_ref": receipt.id, "status": "deployed"}
        )
        self._append_lifecycle(deployed, actor=actor)

        def update(current: ResearchState) -> ResearchState:
            current.active_run_ref = deployed.run_ref
            current.contract_ref = deployed.contract_ref
            current.observation_window = ObservationWindowState(
                id=f"observation:{deployed.run_ref}",
                run_ref=deployed.run_ref,
                checkpoint_start_ref=current.active_checkpoint_ref,
                required_evidence=[f"evaluator:{value}" for value in deployed.required_evaluators],
                status="planned",
            )
            case = current.cases.get(current.active_case_ref)
            if case is not None:
                case.status = "experimenting"
            return current

        updated = self.state_store.compare_and_set(
            expected_state_revision,
            update,
            actor=actor,
            reason=f"record deployed research run {run_ref}",
        )
        return CoordinatedDeployment(deployed, receipt, result, updated)

    def start_deployed(
        self,
        run_ref: str,
        starter: ResearchTrainingStarter,
        *,
        expected_state_revision: int,
        actor: str = "research-coordinator",
    ) -> CoordinatedTrainingStart:
        """Start a deployed plan and enter observation only after verification."""
        state = self.state_store.load()
        self._require_revision(state, expected_state_revision)
        lifecycle = self._load_lifecycle(run_ref)
        if lifecycle.status != "deployed":
            raise ResearchCoordinatorError(
                f"research run cannot start training from status {lifecycle.status}"
            )
        try:
            plan = self._training_plan(lifecycle.pipeline_manifest_ref)
        except (OSError, json.JSONDecodeError) as exc:
            raise ResearchCoordinatorError(
                f"cannot load training launch plan: {type(exc).__name__}: {exc}"
            ) from exc
        if plan is None:
            raise ResearchCoordinatorError(
                "deployed research run has no training launch plan"
            )
        if plan.run_id != lifecycle.run_ref:
            raise ResearchCoordinatorError(
                "training launch plan run_id does not match the coordinated run"
            )
        expected_plan_digest = lifecycle.launch_plan_digest
        actual_plan_digest = self._training_plan_digest(plan)
        if expected_plan_digest and expected_plan_digest != actual_plan_digest:
            raise ResearchCoordinatorError(
                "training launch plan digest does not match the prepared research run"
            )

        prior_receipt = self._latest_training_start_receipt(run_ref)
        if prior_receipt is not None and prior_receipt.status == "started":
            if prior_receipt.plan_digest != actual_plan_digest:
                raise ResearchCoordinatorError(
                    "existing training start receipt does not match the materialized launch plan"
                )
            prior_result = self._validate_training_start_result(
                plan,
                self._result_from_started_receipt(prior_receipt),
            )
            if prior_result.status != "started":
                raise ResearchCoordinatorError(
                    "existing training start receipt failed plan identity validation"
                )
            return self._finalize_started_training(
                lifecycle,
                plan,
                prior_receipt,
                prior_result,
                expected_state_revision=expected_state_revision,
                actor=actor,
            )

        try:
            result = starter.start(plan)
        except Exception as exc:  # noqa: BLE001
            result = self._failed_training_start(
                plan,
                f"{type(exc).__name__}: {exc}",
            )
        result = self._validate_training_start_result(plan, result)
        receipt = self._training_start_receipt(lifecycle, plan, result)
        self.ledger.append("training_start_receipt", receipt, actor=actor)

        if result.status != "started":
            failed_attempt = lifecycle.model_copy(
                update={"training_start_receipt_ref": receipt.id}
            )
            self._append_lifecycle(failed_attempt, actor=actor)
            return CoordinatedTrainingStart(
                failed_attempt,
                receipt,
                result,
                state,
            )
        return self._finalize_started_training(
            lifecycle,
            plan,
            receipt,
            result,
            expected_state_revision=expected_state_revision,
            actor=actor,
        )

    @staticmethod
    def _fact_statement(value: Any) -> str:
        if isinstance(value, Mapping):
            name = str(value.get("name") or value.get("id") or "fact")
            detail = {
                str(key): item
                for key, item in value.items()
                if str(key) not in {"name", "id", "confidence"}
            }
            return f"{name}: {_canonical(detail)}"
        return str(value)

    def ingest_evidence(
        self,
        evidence: EvidenceRecord,
        *,
        expected_state_revision: int,
        actor: str = "research-coordinator",
    ) -> EvidenceIngestion:
        """导入遥测、诊断、sim2sim 或真机证据并同步当前研究状态。"""
        state = self.state_store.load()
        self._require_revision(state, expected_state_revision)
        lifecycle = self._load_lifecycle(evidence.run_ref)
        if lifecycle.status not in {"observing", "continued"}:
            raise ResearchCoordinatorError(
                f"cannot attach current evidence to run in status {lifecycle.status}"
            )
        if evidence.contract_ref and evidence.contract_ref != lifecycle.contract_ref:
            raise ResearchCoordinatorError("evidence contract does not match the coordinated run")
        if self.ledger.latest("evidence", evidence.id) is not None:
            raise ResearchCoordinatorError(f"evidence id already exists: {evidence.id}")
        event = self.ledger.append("evidence", evidence, actor=actor)
        updated_lifecycle = lifecycle.model_copy(
            update={
                "evidence_refs": list(
                    dict.fromkeys(lifecycle.evidence_refs + [evidence.id])
                )
            }
        )
        self._append_lifecycle(updated_lifecycle, actor=actor)

        def update(current: ResearchState) -> ResearchState:
            case = current.cases.get(current.active_case_ref)
            if case is not None:
                case.evidence_refs = list(dict.fromkeys(case.evidence_refs + [evidence.id]))
                case.status = "evaluating"
            facts: list[Any] = list(evidence.observed_facts) + list(evidence.extracted_facts)
            for index, value in enumerate(facts):
                confidence = 0.5
                if isinstance(value, Mapping):
                    try:
                        confidence = max(0.0, min(1.0, float(value.get("confidence", 0.5))))
                    except (TypeError, ValueError):
                        confidence = 0.5
                fact = FactState(
                    id=f"fact:{evidence.id}:{index}",
                    statement=self._fact_statement(value),
                    evidence_refs=[evidence.id],
                    confidence=confidence,
                    status="observed",
                    scope=evidence.scene or evidence.scenario_ref,
                )
                current.facts[fact.id] = fact
            if current.observation_window is not None:
                current.observation_window.collected_evidence = list(
                    dict.fromkeys(
                        current.observation_window.collected_evidence + [evidence.id]
                    )
                )
            return current

        updated_state = self.state_store.compare_and_set(
            expected_state_revision,
            update,
            actor=actor,
            reason=f"ingest evidence {evidence.id}",
        )
        return EvidenceIngestion(evidence, event, updated_lifecycle, updated_state)

    def _validate_submission(
        self,
        lifecycle: ResearchRunLifecycle,
        submission: EvaluationSubmission,
    ) -> None:
        if submission.run_ref != lifecycle.run_ref or submission.contract_ref != lifecycle.contract_ref:
            raise ResearchCoordinatorError("evaluation target does not match the coordinated run")
        if not submission.evidence_refs:
            raise ResearchCoordinatorError("evaluation requires cited evidence")
        for evidence_ref in submission.evidence_refs:
            record = self.ledger.latest("evidence", evidence_ref)
            if record is None:
                raise ResearchCoordinatorError(f"evaluation cites unknown evidence: {evidence_ref}")
            if str(record.get("run_ref") or "") != lifecycle.run_ref:
                raise ResearchCoordinatorError(
                    f"evaluation evidence belongs to another run: {evidence_ref}"
                )
        cited = set(submission.evidence_refs)
        for name, result in {
            **submission.evaluator_results,
            **submission.protected_results,
        }.items():
            outside = set(result.evidence_refs) - cited
            if outside:
                raise ResearchCoordinatorError(
                    f"evaluator {name} cites evidence outside the submission: {sorted(outside)}"
                )

    @staticmethod
    def assess(
        lifecycle: ResearchRunLifecycle,
        submission: EvaluationSubmission,
    ) -> DispositionAssessment:
        required = set(lifecycle.required_evaluators)
        protected = set(lifecycle.protected_capabilities)
        missing_evaluators = sorted(
            name
            for name in required
            if name not in submission.evaluator_results
            or submission.evaluator_results[name].passed is None
        )
        failed_evaluators = sorted(
            name
            for name in required
            if name in submission.evaluator_results
            and submission.evaluator_results[name].passed is False
        )
        missing_protected = sorted(
            name
            for name in protected
            if name not in submission.protected_results
            or submission.protected_results[name].passed is None
        )
        regressed = sorted(
            name
            for name in protected
            if name in submission.protected_results
            and submission.protected_results[name].passed is False
        )
        rationale: list[str] = []
        if submission.terminal_failure:
            rationale.append("evaluation reports a terminal or unsafe failure")
        if regressed:
            rationale.append("protected capabilities regressed: " + ", ".join(regressed))
        if submission.terminal_failure or regressed:
            disposition: Literal["promote", "continue", "rollback"] = "rollback"
        elif (
            submission.objective_success is True
            and not missing_evaluators
            and not failed_evaluators
            and not missing_protected
        ):
            disposition = "promote"
            rationale.append("objective, required evaluators and protected capabilities all pass")
        else:
            disposition = "continue"
            if submission.objective_success is False:
                rationale.append("objective has not passed but no rollback condition is proven")
            if submission.objective_success is None:
                rationale.append("objective result is incomplete")
            if missing_evaluators:
                rationale.append("required evaluator results are missing")
            if failed_evaluators:
                rationale.append("required evaluators have not passed")
            if missing_protected:
                rationale.append("protected capability results are missing")
        return DispositionAssessment(
            disposition=disposition,
            rationale=tuple(rationale),
            missing_evaluators=tuple(missing_evaluators),
            missing_protected_results=tuple(missing_protected),
            failed_evaluators=tuple(failed_evaluators),
            regressed_capabilities=tuple(regressed),
        )

    def _rollback_remote(
        self,
        lifecycle: ResearchRunLifecycle,
        deployer: ResearchDeployer,
        *,
        actor: str,
    ) -> tuple[ResearchRunLifecycle, DeploymentReceipt]:
        target = lifecycle.previous_active_run_ref
        if not target:
            raise ResearchCoordinatorError("research run has no previous active run to roll back to")
        try:
            artifact = deployer.rollback_active(target)
        except Exception as exc:
            receipt = DeploymentReceipt(
                id=f"deployment:{lifecycle.run_ref}:rollback:{uuid.uuid4().hex}",
                run_ref=lifecycle.run_ref,
                contract_ref=lifecycle.contract_ref,
                operation="rollback",
                status="failed",
                target_run_ref=target,
                previous_active_run_ref=lifecycle.run_ref,
                error=f"{type(exc).__name__}: {exc}",
                source="research_coordinator",
            )
            self.ledger.append("deployment_receipt", receipt, actor=actor)
            required = lifecycle.model_copy(
                update={"status": "rollback_required", "disposition": "rollback"}
            )
            self._append_lifecycle(required, actor=actor)
            return required, receipt
        receipt = DeploymentReceipt(
            id=f"deployment:{lifecycle.run_ref}:rollback:{uuid.uuid4().hex}",
            run_ref=lifecycle.run_ref,
            contract_ref=lifecycle.contract_ref,
            operation="rollback",
            status="rolled_back",
            target_run_ref=target,
            previous_active_run_ref=lifecycle.run_ref,
            remote_run_ref=artifact.identity,
            trace=[{"status": artifact.status, "path": artifact.path, "detail": artifact.detail}],
            source="research_coordinator",
        )
        self.ledger.append("deployment_receipt", receipt, actor=actor)
        rolled_back = lifecycle.model_copy(
            update={"status": "rolled_back", "disposition": "rollback"}
        )
        self._append_lifecycle(rolled_back, actor=actor)
        return rolled_back, receipt

    def evaluate(
        self,
        submission: EvaluationSubmission,
        *,
        expected_state_revision: int,
        actor: str = "research-coordinator",
        deployer: ResearchDeployer | None = None,
    ) -> CoordinatedEvaluation:
        """依据独立证据裁决 promote/continue/rollback，并可立即执行回滚。"""
        state = self.state_store.load()
        self._require_revision(state, expected_state_revision)
        lifecycle = self._load_lifecycle(submission.run_ref)
        if lifecycle.status not in {"observing", "continued"}:
            raise ResearchCoordinatorError(
                f"research run cannot be evaluated from status {lifecycle.status}"
            )
        self._validate_submission(lifecycle, submission)
        assessment = self.assess(lifecycle, submission)
        outcome = ExperimentOutcome(
            id=f"outcome:{submission.evaluation_id}",
            experiment_ref=lifecycle.run_ref,
            actual_diff_ref=lifecycle.contract_ref,
            capability_delta={
                name: result.model_dump(mode="json")
                for name, result in submission.evaluator_results.items()
            },
            protected_capability_results={
                name: result.model_dump(mode="json")
                for name, result in submission.protected_results.items()
            },
            prediction_results=[
                {"evaluator": name, **result.model_dump(mode="json")}
                for name, result in submission.evaluator_results.items()
            ],
            disposition=assessment.disposition,
            knowledge_updates=list(submission.notes),
            source="research_coordinator",
        )
        self.ledger.append("experiment_outcome", outcome, actor=actor)
        decision = DecisionRecord(
            id=f"decision:evaluation:{submission.evaluation_id}",
            trigger="deterministic evidence disposition",
            problem_statement=f"evaluate research run {lifecycle.run_ref}",
            evidence_refs=list(submission.evidence_refs),
            selected_experiment_ref=lifecycle.run_ref,
            check_results=[assessment.model_dump(mode="json")],
            status="evaluated",
            source="research_coordinator",
        )
        self.ledger.append("decision", decision, actor=actor)

        rollback_receipt: DeploymentReceipt | None = None
        if assessment.disposition == "rollback" and deployer is not None:
            updated_lifecycle, rollback_receipt = self._rollback_remote(
                lifecycle,
                deployer,
                actor=actor,
            )
        elif assessment.disposition == "rollback":
            updated_lifecycle = lifecycle.model_copy(
                update={"status": "rollback_required", "disposition": "rollback"}
            )
            self._append_lifecycle(updated_lifecycle, actor=actor)
        elif assessment.disposition == "promote":
            updated_lifecycle = lifecycle.model_copy(
                update={"status": "promoted", "disposition": "promote"}
            )
            self._append_lifecycle(updated_lifecycle, actor=actor)
        else:
            updated_lifecycle = lifecycle.model_copy(
                update={"status": "continued", "disposition": "continue"}
            )
            self._append_lifecycle(updated_lifecycle, actor=actor)

        def update(current: ResearchState) -> ResearchState:
            for capability in updated_lifecycle.protected_capabilities:
                result = submission.protected_results.get(capability)
                status = "unknown"
                if result is not None and result.passed is True:
                    status = "passing"
                elif result is not None and result.passed is False:
                    status = "regressed"
                current.protected_capabilities[capability] = ProtectedCapabilityState(
                    id=capability,
                    baseline_ref=updated_lifecycle.previous_active_run_ref or current.baseline_ref,
                    evaluator_ref=capability,
                    last_result=result.model_dump(mode="json") if result is not None else {},
                    status=status,
                )
            case = current.cases.get(current.active_case_ref)
            if assessment.disposition == "promote":
                current.baseline_ref = updated_lifecycle.run_ref
                current.active_checkpoint_ref = submission.checkpoint_ref
                current.pending_interventions = [
                    value
                    for value in current.pending_interventions
                    if value not in {updated_lifecycle.run_ref, updated_lifecycle.proposal_ref}
                ]
                if case is not None:
                    case.status = "resolved"
                    case.resolution = f"promoted by {outcome.id}"
                if current.observation_window is not None:
                    current.observation_window.status = "complete"
            elif assessment.disposition == "rollback" and updated_lifecycle.status == "rolled_back":
                current.active_run_ref = updated_lifecycle.previous_active_run_ref
                current.active_checkpoint_ref = updated_lifecycle.previous_checkpoint_ref
                current.contract_ref = updated_lifecycle.previous_contract_ref
                current.pending_interventions = [
                    value
                    for value in current.pending_interventions
                    if value not in {updated_lifecycle.run_ref, updated_lifecycle.proposal_ref}
                ]
                if case is not None:
                    case.status = "open"
                    case.resolution = f"rolled back by {outcome.id}"
                if current.observation_window is not None:
                    current.observation_window.status = "cancelled"
            elif assessment.disposition == "rollback":
                current.pending_interventions = list(
                    dict.fromkeys(
                        current.pending_interventions
                        + [f"rollback:{updated_lifecycle.run_ref}"]
                    )
                )
                if current.observation_window is not None:
                    current.observation_window.status = "intervened"
                if case is not None:
                    case.status = "open"
            else:
                if case is not None:
                    case.status = "evaluating"
            current.last_outcome_ref = outcome.id
            current.last_decision_ref = decision.id
            return current

        updated_state = self.state_store.compare_and_set(
            expected_state_revision,
            update,
            actor=actor,
            reason=f"apply {assessment.disposition} disposition for {submission.run_ref}",
        )
        return CoordinatedEvaluation(
            assessment,
            outcome,
            decision,
            updated_lifecycle,
            updated_state,
            rollback_receipt,
        )

    def execute_pending_rollback(
        self,
        run_ref: str,
        deployer: ResearchDeployer,
        *,
        expected_state_revision: int,
        actor: str = "research-coordinator",
    ) -> DeploymentReceipt:
        """远程暂不可用时，恢复后执行账本中待处理的回滚。"""
        state = self.state_store.load()
        self._require_revision(state, expected_state_revision)
        lifecycle = self._load_lifecycle(run_ref)
        if lifecycle.status != "rollback_required":
            raise ResearchCoordinatorError(
                f"research run has no pending rollback: {lifecycle.status}"
            )
        rolled_back, receipt = self._rollback_remote(lifecycle, deployer, actor=actor)
        if rolled_back.status != "rolled_back":
            return receipt

        def update(current: ResearchState) -> ResearchState:
            current.active_run_ref = rolled_back.previous_active_run_ref
            current.active_checkpoint_ref = rolled_back.previous_checkpoint_ref
            current.contract_ref = rolled_back.previous_contract_ref
            current.pending_interventions = [
                value
                for value in current.pending_interventions
                if value not in {
                    rolled_back.run_ref,
                    rolled_back.proposal_ref,
                    f"rollback:{rolled_back.run_ref}",
                }
            ]
            if current.observation_window is not None:
                current.observation_window.status = "cancelled"
            return current

        self.state_store.compare_and_set(
            expected_state_revision,
            update,
            actor=actor,
            reason=f"complete pending rollback for {run_ref}",
        )
        return receipt


__all__ = [
    "COORDINATOR_SCHEMA",
    "CoordinatedDeployment",
    "CoordinatedTrainingStart",
    "CoordinatedEvaluation",
    "DispositionAssessment",
    "EvaluationSubmission",
    "EvaluatorResult",
    "EvidenceIngestion",
    "ResearchCoordinator",
    "ResearchCoordinatorError",
    "ResearchDeployer",
    "ResearchTrainingStarter",
]
