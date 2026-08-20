"""Executable vertical loop from a research problem to learned outcome."""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from autotuner.mechanisms.mechanism_artifacts import MechanismArtifactCompiler
from autotuner.mechanisms.mechanism_synthesis import MechanismSynthesizer, SynthesisCandidate, SynthesisRequest
from autotuner.mechanisms.mechanism_validation import ValidationContext, ValidationReport, validate_patch
from .outcome_learning import OutcomeLearner, OutcomeLearningResult
from .research_ledger import (
    DecisionRecord,
    ExperimentPlan,
    MechanismCandidateRecord,
    ResearchLedgerStore,
)
from .research_state import ResearchStateStore, RevisionConflict
from .research_supervisor import ExperimentExecution, ResearchSupervisor


class CandidateAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    candidate: SynthesisCandidate
    validation: ValidationReport
    artifact_path: str = ""
    selection_score: float
    selected: bool = False


class ResearchCycleProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cycle_id: str
    problem_ref: str
    baseline_bundle_ref: str
    selected_candidate_ref: str = ""
    selected_patch_ref: str = ""
    selected_artifact_path: str = ""
    assessments: tuple[CandidateAssessment, ...]
    state_revision: int
    status: str


class ResearchCycleResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    proposal: ResearchCycleProposal
    execution: ExperimentExecution
    learning: OutcomeLearningResult


def _failure_probe(request: SynthesisRequest) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for gap in request.gaps:
        if gap.desired == "lower":
            value = float(gap.bad)
        elif gap.desired == "higher":
            value = float(gap.bad)
        else:
            value = float(gap.bad[1])
        if gap.signal.shape in {"vector", "matrix", "per_leg", "per_joint"}:
            value = [value, value]
        values[gap.signal.name] = [value, value, value, value] if not isinstance(value, list) else [value] * 4
    return values


def _cycle_dir_name(cycle_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", cycle_id).strip("._")
    if not value:
        raise ValueError("cycle id has no safe filesystem representation")
    return value[:120]


class ResearchCycleManager:
    """Coordinates synthesis, validation, execution, and outcome learning."""

    def __init__(
        self,
        root: str | Path = "output/research/cycles",
        *,
        state_store: ResearchStateStore | None = None,
        ledger: ResearchLedgerStore | None = None,
        supervisor: ResearchSupervisor | None = None,
    ) -> None:
        self.root = Path(root)
        self.state_store = state_store or ResearchStateStore(self.root.parent / "state")
        self.ledger = ledger or ResearchLedgerStore(self.root.parent / "ledger")
        self.supervisor = supervisor or ResearchSupervisor(self.root.parent / "experiments")
        self.synthesizer = MechanismSynthesizer()
        self.compiler = MechanismArtifactCompiler()

    def propose(
        self,
        request: SynthesisRequest,
        *,
        expected_state_revision: int,
        actor: str,
        validation_context: ValidationContext | None = None,
    ) -> ResearchCycleProposal:
        state = self.state_store.load()
        if state.revision != expected_state_revision:
            raise RevisionConflict(expected_state_revision, state.revision)
        request = request.model_copy(update={
            "no_repeat_fingerprints": frozenset(
                set(request.no_repeat_fingerprints) | set(state.no_repeat_fingerprints)
            )
        })
        candidates = self.synthesizer.synthesize(request)
        if not candidates:
            return ResearchCycleProposal(
                cycle_id=request.id, problem_ref=request.problem_ref,
                baseline_bundle_ref=request.baseline.id, assessments=(),
                state_revision=state.revision, status="no_novel_candidates",
            )
        context = validation_context or ValidationContext(
            required_evaluator_refs=request.required_evaluators,
            protected_capabilities=request.protected_capabilities,
            runtime_signal_names=frozenset(
                {item.name for item in request.baseline.signals}
                | {gap.signal.name for gap in request.gaps}
            ),
            failure_probe_values=_failure_probe(request),
        )
        cycle_root = self.root / _cycle_dir_name(request.id)
        artifact_root = cycle_root / "artifacts"
        assessments: list[CandidateAssessment] = []
        for candidate in candidates:
            bundle, validation = validate_patch(request.baseline, candidate.patch, context)
            artifact_path = ""
            if validation.ok:
                artifact, _ = self.compiler.compile(
                    output_root=artifact_root, baseline=request.baseline, patch=candidate.patch,
                )
                artifact_path = str(artifact)
            guard_count = sum(
                operation.target_kind == "reward"
                and isinstance(operation.value, dict)
                and operation.value.get("role") == "guard"
                for operation in candidate.patch.operations
            )
            selection_score = (
                1000.0 * len(validation.errors)
                + 10.0 * len(validation.warnings)
                + len(candidate.patch.operations)
                + 0.5 * guard_count
            )
            assessments.append(CandidateAssessment(
                candidate=candidate, validation=validation,
                artifact_path=artifact_path, selection_score=selection_score,
            ))
        valid = [item for item in assessments if item.validation.ok]
        selected = min(valid, key=lambda item: (item.selection_score, item.candidate.id)) if valid else None
        if selected is not None:
            assessments = [item.model_copy(update={"selected": item.candidate.id == selected.candidate.id}) for item in assessments]

        for item in assessments:
            record = MechanismCandidateRecord(
                id=item.candidate.id,
                problem_ref=request.problem_ref,
                baseline_bundle_ref=request.baseline.id,
                baseline_fingerprint=request.baseline.fingerprint(),
                candidate_bundle_ref=item.candidate.patch.candidate_bundle_id,
                candidate_fingerprint=item.candidate.candidate_bundle_fingerprint,
                patch_ref=item.candidate.patch.id,
                artifact_ref=item.artifact_path,
                validation={
                    "ok": item.validation.ok,
                    "errors": [issue.model_dump(mode="json") for issue in item.validation.errors],
                    "warnings": [issue.model_dump(mode="json") for issue in item.validation.warnings],
                    "checks": item.validation.checks,
                },
                predictions=list(item.candidate.predictions),
                protected_capabilities=list(request.protected_capabilities),
                required_evaluators=list(request.required_evaluators),
                status="selected" if item.selected else ("validated" if item.validation.ok else "rejected"),
                source=request.generated_by,
            )
            self.ledger.append("mechanism_candidate", record, actor=actor)

        decision = DecisionRecord(
            id=f"decision:{request.id}",
            trigger="mechanism synthesis",
            problem_statement=request.problem_statement,
            evidence_refs=[ref for gap in request.gaps for ref in gap.evidence_refs],
            hypotheses_considered=[gap.hypothesis_ref for gap in request.gaps],
            selected_experiment_ref=selected.candidate.id if selected else "",
            rejected_options=[{
                "candidate_ref": item.candidate.id,
                "reason": "validation_failed" if not item.validation.ok else "higher_occam_score",
                "score": item.selection_score,
            } for item in assessments if not item.selected],
            check_results=[{
                "candidate_ref": item.candidate.id,
                "ok": item.validation.ok,
                "error_codes": [issue.code for issue in item.validation.errors],
            } for item in assessments],
            status="proposed" if selected else "abandoned",
            source="research_cycle",
        )
        self.ledger.append("decision", decision, actor=actor)

        def update_state(current):
            if selected is not None:
                current.pending_interventions = list(dict.fromkeys(
                    current.pending_interventions
                    + [selected.candidate.id, selected.candidate.patch.id]
                ))
                case = current.cases.get(request.problem_ref)
                if case is not None:
                    case.candidate_refs = list(dict.fromkeys(case.candidate_refs + [selected.candidate.id]))
                    case.status = "proposed"
                current.last_decision_ref = decision.id
            for item in assessments:
                if not item.validation.ok:
                    current.no_repeat_fingerprints[item.candidate.patch.fingerprint()] = f"validation-rejected:{decision.id}"
                    current.no_repeat_fingerprints[item.candidate.candidate_bundle_fingerprint] = f"validation-rejected:{decision.id}"
            return current

        updated = self.state_store.compare_and_set(
            expected_state_revision, update_state, actor=actor,
            reason=f"record synthesis decision {decision.id}",
        )
        proposal = ResearchCycleProposal(
            cycle_id=request.id,
            problem_ref=request.problem_ref,
            baseline_bundle_ref=request.baseline.id,
            selected_candidate_ref=selected.candidate.id if selected else "",
            selected_patch_ref=selected.candidate.patch.id if selected else "",
            selected_artifact_path=selected.artifact_path if selected else "",
            assessments=tuple(assessments),
            state_revision=updated.revision,
            status="candidate_selected" if selected else "all_candidates_rejected",
        )
        cycle_root.mkdir(parents=True, exist_ok=True)
        (cycle_root / "proposal.json").write_text(proposal.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return proposal

    def load_proposal(self, cycle_id: str) -> ResearchCycleProposal:
        path = self.root / _cycle_dir_name(cycle_id) / "proposal.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        return ResearchCycleProposal.model_validate_json(path.read_text(encoding="utf-8"))

    def load_plan(self, plan_id: str) -> ExperimentPlan:
        value = self.ledger.latest("experiment_plan", plan_id)
        if value is None:
            raise KeyError(f"approved experiment plan not found: {plan_id}")
        return ExperimentPlan.model_validate(value)

    def register_plan(self, plan: ExperimentPlan, *, actor: str) -> ExperimentPlan:
        if plan.status != "approved":
            raise ValueError("only an approved experiment plan can be registered for execution")
        if not plan.authorization_ref.strip():
            raise ValueError("approved experiment plan requires an explicit authorization_ref")
        existing = self.ledger.latest("experiment_plan", plan.id)
        if existing is not None:
            current = ExperimentPlan.model_validate(existing)
            if current != plan:
                raise ValueError("experiment plan id already exists with different content")
            return current
        self.ledger.append("experiment_plan", plan, actor=actor)
        return plan

    def execute_registered(
        self,
        *,
        cycle_id: str,
        plan_id: str,
        actor: str,
        environment: dict[str, str] | None = None,
        poll_interval_s: float = 1.0,
    ) -> ResearchCycleResult:
        proposal = self.load_proposal(cycle_id)
        plan = self.load_plan(plan_id)
        self._validate_plan_binding(proposal, plan)
        receipt = self._reserve_execution(proposal, plan, actor=actor)
        try:
            result = self.execute(
                proposal, plan, actor=actor,
                environment=environment, poll_interval_s=poll_interval_s,
            )
        except Exception as exc:
            self._finish_execution_receipt(
                receipt,
                status="failed",
                detail=f"{type(exc).__name__}: {exc}",
            )
            raise
        self._finish_execution_receipt(
            receipt,
            status="completed",
            detail=f"disposition={result.execution.disposition}",
        )
        return result

    def _execution_receipt_path(self, proposal: ResearchCycleProposal, plan: ExperimentPlan) -> Path:
        return (
            self.root
            / _cycle_dir_name(proposal.cycle_id)
            / "executions"
            / f"{_cycle_dir_name(plan.id)}.json"
        )

    def _reserve_execution(
        self,
        proposal: ResearchCycleProposal,
        plan: ExperimentPlan,
        *,
        actor: str,
    ) -> Path:
        path = self._execution_receipt_path(proposal, plan)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "execution_id": f"execution:{uuid.uuid4().hex}",
            "cycle_id": proposal.cycle_id,
            "plan_id": plan.id,
            "actor": actor,
            "status": "started",
            "started_at_epoch_s": time.time(),
        }
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                status = str(existing.get("status") or "unknown")
            except Exception:
                status = "unreadable"
            raise RuntimeError(
                f"registered experiment already has an execution receipt ({status}); "
                "use a new approved plan id for an explicit retry"
            ) from exc
        return path

    @staticmethod
    def _finish_execution_receipt(path: Path, *, status: str, detail: str) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update({
            "status": status,
            "detail": detail,
            "finished_at_epoch_s": time.time(),
        })
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)

    @staticmethod
    def _validate_plan_binding(
        proposal: ResearchCycleProposal,
        plan: ExperimentPlan,
    ) -> CandidateAssessment:
        if proposal.status != "candidate_selected":
            raise RuntimeError(f"proposal cannot execute in status {proposal.status}")
        selected = next(item for item in proposal.assessments if item.selected)
        if plan.baseline_ref != proposal.baseline_bundle_ref:
            raise ValueError("approved plan baseline does not match proposal baseline")
        declared_patch = str(plan.intervention_diff.get("patch_ref", ""))
        if declared_patch != proposal.selected_patch_ref:
            raise ValueError("approved plan is not bound to the selected mechanism patch")
        if not set(selected.candidate.patch.protected_capabilities) <= set(plan.protected_capabilities):
            raise ValueError("approved plan omits candidate protected capabilities")
        return selected

    def execute(
        self,
        proposal: ResearchCycleProposal,
        plan: ExperimentPlan,
        *,
        actor: str,
        environment: dict[str, str] | None = None,
        poll_interval_s: float = 1.0,
    ) -> ResearchCycleResult:
        selected = self._validate_plan_binding(proposal, plan)
        if self.ledger.latest("experiment_plan", plan.id) is None:
            raise PermissionError("experiment plan must be registered before execution")
        execution = self.supervisor.execute(
            plan, proposal.selected_artifact_path, actor=actor,
            environment=environment, poll_interval_s=poll_interval_s,
        )
        learning = OutcomeLearner(self.state_store, self.ledger).learn(
            expected_state_revision=proposal.state_revision,
            plan=plan,
            candidate=selected.candidate,
            execution=execution,
            actor=actor,
        )
        return ResearchCycleResult(proposal=proposal, execution=execution, learning=learning)
