"""Durable research records for the RL agent architecture.

The ledger is intentionally local and append-only.  It stores research facts,
plans, decisions, and handoff state without granting the agent permission to
change a contract, reward, checkpoint, or remote process.  A record is useful
only when its provenance and lifecycle are explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Type

from pydantic import BaseModel, Field


SCHEMA_VERSION = "rl-agent.research-ledger/v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dump(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_dump(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class LedgerRecord(BaseModel):
    schema_version: str = SCHEMA_VERSION
    id: str
    created_at: str = Field(default_factory=utc_now)
    source: str = ""


class ArtifactRef(BaseModel):
    id: str
    kind: str
    path: str = ""
    sha256: str = ""
    role: str = ""
    exists: bool | None = None


class ResearchProgram(LedgerRecord):
    robot_model_ref: str = ""
    control_stack_ref: str = ""
    task_contract_ref: str = ""
    training_world_ref: str = ""
    deployment_worlds: list[str] = Field(default_factory=list)
    budget: dict[str, Any] = Field(default_factory=dict)
    status: Literal["draft", "active", "retired"] = "draft"


class ContractBundle(LedgerRecord):
    version: str = ""
    status: Literal["draft", "approved", "retired"] = "draft"
    approved_by: str = ""
    goals: list[str] = Field(default_factory=list)
    qualitative_intent: list[str] = Field(default_factory=list)
    quantitative_gates: list[dict[str, Any]] = Field(default_factory=list)
    required_scenarios: list[str] = Field(default_factory=list)
    protected_capabilities: list[str] = Field(default_factory=list)
    deployment_constraints: list[str] = Field(default_factory=list)
    generated_artifacts: list[ArtifactRef] = Field(default_factory=list)
    supersedes: list[str] = Field(default_factory=list)


class EffectiveRunSnapshot(LedgerRecord):
    run_id: str
    source_info: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    assets: dict[str, Any] = Field(default_factory=dict)
    policy_contract_ref: str = ""
    distribution_manifest_ref: str = ""
    launch_command_redacted: list[str] = Field(default_factory=list)
    seeds: dict[str, Any] = Field(default_factory=dict)
    runtime_manifest_ref: str = ""
    task_contract_ref: str = ""
    task_contract_digest: str = ""
    task_bundle_digest: str = ""
    task_artifact_manifest_ref: str = ""
    runtime_digest: str = ""
    payload_digest: str = ""


class PolicyDeploymentContract(LedgerRecord):
    version: str = ""
    observation: dict[str, Any] = Field(default_factory=dict)
    action: dict[str, Any] = Field(default_factory=dict)
    controller: dict[str, Any] = Field(default_factory=dict)
    privileged_inputs: dict[str, list[str]] = Field(default_factory=dict)
    isaaclab_ref: str = ""
    mujoco_ref: str = ""
    real_adapter_ref: str = ""


class TrainingDistributionManifest(LedgerRecord):
    run_id: str = ""
    command_buckets: dict[str, Any] = Field(default_factory=dict)
    terrain_mix: dict[str, Any] = Field(default_factory=dict)
    curriculum_state: dict[str, Any] = Field(default_factory=dict)
    dr_channels: dict[str, Any] = Field(default_factory=dict)
    reset_sampling: dict[str, Any] = Field(default_factory=dict)
    realized_window: dict[str, Any] = Field(default_factory=dict)


class CapabilityProfile(LedgerRecord):
    checkpoint_ref: str
    capabilities: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    status: Literal["observed", "candidate", "validated", "counterexample", "retired"] = "observed"
    scope: str = ""


class BaselineSet(LedgerRecord):
    anchors: list[CapabilityProfile] = Field(default_factory=list)
    protected_capabilities: list[str] = Field(default_factory=list)
    selection_reason: str = ""


class ResumeEdge(LedgerRecord):
    parent_checkpoint_ref: str
    child_run_ref: str
    restored: dict[str, bool] = Field(default_factory=dict)
    reset_reason: dict[str, Any] = Field(default_factory=dict)
    compatibility_report_ref: str = ""
    parent_task_contract_ref: str = ""
    child_task_contract_ref: str = ""
    parent_task_contract_digest: str = ""
    child_task_contract_digest: str = ""
    parent_bundle_digest: str = ""
    child_bundle_digest: str = ""
    parent_payload_digest: str = ""
    child_payload_digest: str = ""
    proof: dict[str, Any] = Field(default_factory=dict)
    status: Literal["complete", "partial", "weights_only", "invalid", "unknown"] = "unknown"


class MechanismEntry(BaseModel):
    name: str
    formula_ref: str = ""
    intended_effect: str = ""
    activation: dict[str, Any] = Field(default_factory=dict)
    coverage: dict[str, Any] = Field(default_factory=dict)
    scale: dict[str, Any] = Field(default_factory=dict)
    saturation: dict[str, Any] = Field(default_factory=dict)
    contribution: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)


class RewardMechanismGraph(LedgerRecord):
    capability: str
    eligibility: str = ""
    positive_drives: list[MechanismEntry] = Field(default_factory=list)
    bridges: list[MechanismEntry] = Field(default_factory=list)
    guards: list[MechanismEntry] = Field(default_factory=list)
    couplings: list[dict[str, Any]] = Field(default_factory=list)
    exploit_hypotheses: list[str] = Field(default_factory=list)
    runtime_observables: list[str] = Field(default_factory=list)
    protected_capabilities: list[str] = Field(default_factory=list)


class MechanismCandidateRecord(LedgerRecord):
    problem_ref: str
    baseline_bundle_ref: str
    baseline_fingerprint: str
    candidate_bundle_ref: str
    candidate_fingerprint: str
    patch_ref: str
    artifact_ref: str = ""
    validation: dict[str, Any] = Field(default_factory=dict)
    predictions: list[str] = Field(default_factory=list)
    protected_capabilities: list[str] = Field(default_factory=list)
    required_evaluators: list[str] = Field(default_factory=list)
    status: Literal["generated", "validated", "rejected", "selected", "executed", "retired"] = "generated"


class EvidenceRecord(LedgerRecord):
    kind: Literal["telemetry", "physical_diag", "video_observation", "source_audit", "sim2sim", "real_test"]
    run_ref: str = ""
    checkpoint_ref: str = ""
    scenario_ref: str = ""
    contract_ref: str = ""
    measured_at: str = ""
    raw_artifact_ref: str = ""
    extracted_facts: list[dict[str, Any]] = Field(default_factory=list)
    observer: Literal["machine", "human", "hybrid"] = "machine"
    freshness: Literal["current", "stale", "historical"] = "current"
    scene: str = ""
    command_timeline: list[dict[str, Any]] = Field(default_factory=list)
    observed_facts: list[str] = Field(default_factory=list)


class HypothesisCandidate(BaseModel):
    claim: str
    confidence: float = 0.0
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    prediction: str = ""
    disambiguation: str = ""


class HypothesisSet(LedgerRecord):
    symptom_ref: str
    candidates: list[HypothesisCandidate] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)


class ExperimentPlan(LedgerRecord):
    problem_statement: str
    baseline_ref: str
    hypotheses_addressed: list[str] = Field(default_factory=list)
    intervention_diff: dict[str, Any] = Field(default_factory=dict)
    unchanged_fields: list[str] = Field(default_factory=list)
    protected_capabilities: list[str] = Field(default_factory=list)
    allowed_temporary_regression: dict[str, Any] = Field(default_factory=dict)
    training_window: dict[str, Any] = Field(default_factory=dict)
    evaluation_plan: dict[str, Any] = Field(default_factory=dict)
    seed_strategy: dict[str, Any] = Field(default_factory=dict)
    success_condition: str = ""
    rollback_condition: str = ""
    resource_budget: dict[str, Any] = Field(default_factory=dict)
    authorization_ref: str = ""
    risk_tier: Literal["low", "medium", "high"] = "medium"
    status: Literal["proposed", "approved", "executing", "evaluated", "abandoned"] = "proposed"


class DecisionRecord(LedgerRecord):
    trigger: str
    problem_statement: str
    evidence_refs: list[str] = Field(default_factory=list)
    hypotheses_considered: list[str] = Field(default_factory=list)
    selected_experiment_ref: str = ""
    rejected_options: list[dict[str, Any]] = Field(default_factory=list)
    check_results: list[dict[str, Any]] = Field(default_factory=list)
    authorization_ref: str = ""
    status: Literal["proposed", "approved", "executing", "evaluated", "abandoned"] = "proposed"


class ExperimentOutcome(LedgerRecord):
    experiment_ref: str
    actual_diff_ref: str = ""
    exposure_reached: dict[str, Any] = Field(default_factory=dict)
    capability_delta: dict[str, Any] = Field(default_factory=dict)
    protected_capability_results: dict[str, Any] = Field(default_factory=dict)
    prediction_results: list[dict[str, Any]] = Field(default_factory=list)
    disposition: Literal["promote", "continue", "rollback", "inconclusive"] = "inconclusive"
    knowledge_updates: list[str] = Field(default_factory=list)


class KnowledgeClaim(LedgerRecord):
    statement: str
    status: Literal["tentative", "validated", "rejected", "superseded"] = "tentative"
    scope: Literal["universal", "robot_family", "program", "config_range"] = "program"
    evidence_refs: list[str] = Field(default_factory=list)
    supersedes: list[str] = Field(default_factory=list)
    valid_from: str = ""
    last_reviewed: str = ""


class HandoffSnapshot(LedgerRecord):
    sealed_at: str = ""
    active_runs: list[str] = Field(default_factory=list)
    active_jobs: list[str] = Field(default_factory=list)
    current_contract_ref: str = ""
    established_facts: list[str] = Field(default_factory=list)
    active_hypotheses: list[str] = Field(default_factory=list)
    pending_experiments: list[str] = Field(default_factory=list)
    intervention_conditions: list[str] = Field(default_factory=list)


def approve_contract(bundle: ContractBundle, approver: str, *, new_id: str = "") -> ContractBundle:
    """Create an approved version; an existing object is never mutated in place."""
    if bundle.status != "draft":
        raise ValueError("only draft contract bundles can be approved")
    if not approver.strip():
        raise ValueError("approver is required")
    identifier = new_id.strip() or f"{bundle.id}@approved"
    return bundle.model_copy(update={"id": identifier, "status": "approved", "approved_by": approver, "supersedes": [bundle.id]})


class LedgerEvent(BaseModel):
    schema_version: str = SCHEMA_VERSION
    event_id: str
    sequence: int
    event_type: Literal["append", "supersede"] = "append"
    record_type: str
    record_id: str
    actor: str
    created_at: str
    payload: dict[str, Any]
    payload_hash: str
    previous_event_hash: str = ""
    event_hash: str


class LedgerValidationIssue(BaseModel):
    code: str
    message: str
    severity: Literal["error", "warn"] = "error"


RECORD_MODELS: dict[str, Type[BaseModel]] = {
    "research_program": ResearchProgram,
    "contract_bundle": ContractBundle,
    "effective_run_snapshot": EffectiveRunSnapshot,
    "policy_deployment_contract": PolicyDeploymentContract,
    "training_distribution_manifest": TrainingDistributionManifest,
    "capability_profile": CapabilityProfile,
    "baseline_set": BaselineSet,
    "resume_edge": ResumeEdge,
    "reward_mechanism_graph": RewardMechanismGraph,
    "mechanism_candidate": MechanismCandidateRecord,
    "evidence": EvidenceRecord,
    "hypothesis_set": HypothesisSet,
    "experiment_plan": ExperimentPlan,
    "decision": DecisionRecord,
    "experiment_outcome": ExperimentOutcome,
    "knowledge_claim": KnowledgeClaim,
    "handoff": HandoffSnapshot,
}


def validate_record(record_type: str, record: BaseModel | dict[str, Any]) -> list[LedgerValidationIssue]:
    """Validate cross-field invariants without claiming the research is true."""
    model = RECORD_MODELS.get(record_type)
    if model is None:
        return [LedgerValidationIssue(code="record.unknown_type", message=f"unknown record type: {record_type}")]
    try:
        value = record if isinstance(record, model) else model.model_validate(record)
    except Exception as exc:
        return [LedgerValidationIssue(code="record.schema_invalid", message=f"{type(exc).__name__}: {exc}")]

    issues: list[LedgerValidationIssue] = []
    if not str(getattr(value, "id", "")).strip():
        issues.append(LedgerValidationIssue(code="record.id_missing", message="record id is required"))
    if record_type == "contract_bundle" and getattr(value, "status", "") == "approved":
        if not str(getattr(value, "approved_by", "")).strip():
            issues.append(LedgerValidationIssue(code="contract.approver_missing", message="approved contract requires approved_by"))
        if not value.goals or not value.protected_capabilities:
            issues.append(LedgerValidationIssue(code="contract.scope_missing", message="approved contract requires goals and protected capabilities"))
    if record_type == "experiment_plan":
        required = {
            "baseline_ref": value.baseline_ref,
            "intervention_diff": value.intervention_diff,
            "unchanged_fields": value.unchanged_fields,
            "protected_capabilities": value.protected_capabilities,
            "evaluation_plan": value.evaluation_plan,
            "success_condition": value.success_condition,
            "rollback_condition": value.rollback_condition,
            "resource_budget": value.resource_budget,
        }
        for name, item in required.items():
            if not item:
                issues.append(LedgerValidationIssue(code=f"experiment.{name}_missing", message=f"experiment plan requires {name}"))
        if value.risk_tier == "high" and not value.authorization_ref:
            issues.append(LedgerValidationIssue(code="experiment.authorization_missing", message="high-risk experiment requires authorization_ref"))
    if record_type == "evidence":
        if value.observer in {"human", "hybrid"} and not value.scene:
            issues.append(LedgerValidationIssue(code="evidence.scene_missing", message="human evidence requires scene"))
        if value.observer in {"human", "hybrid"} and not value.command_timeline:
            issues.append(LedgerValidationIssue(code="evidence.timeline_missing", message="human evidence requires command timeline"))
        if value.observer in {"human", "hybrid"} and not value.observed_facts:
            issues.append(LedgerValidationIssue(code="evidence.facts_missing", message="human evidence requires structured observed facts"))
    if record_type == "experiment_outcome" and not value.experiment_ref:
        issues.append(LedgerValidationIssue(code="outcome.experiment_missing", message="outcome requires experiment_ref"))
    if record_type == "reward_mechanism_graph" and not value.positive_drives:
        issues.append(LedgerValidationIssue(code="mechanism.positive_drive_missing", message="mechanism graph requires at least one positive drive"))
    if record_type == "mechanism_candidate":
        for name in ("baseline_fingerprint", "candidate_fingerprint", "patch_ref"):
            if not str(getattr(value, name, "")).strip():
                issues.append(LedgerValidationIssue(code=f"candidate.{name}_missing", message=f"mechanism candidate requires {name}"))
        if value.status in {"validated", "selected", "executed"} and not value.artifact_ref:
            issues.append(LedgerValidationIssue(code="candidate.artifact_missing", message="validated candidate requires artifact_ref"))
    if record_type == "knowledge_claim" and value.status == "validated" and not value.evidence_refs:
        issues.append(LedgerValidationIssue(code="claim.evidence_missing", message="validated knowledge claim requires evidence_refs"))
    if record_type == "knowledge_claim" and value.status == "superseded" and not value.supersedes:
        issues.append(LedgerValidationIssue(code="claim.supersedes_missing", message="superseded knowledge claim requires supersedes"))
    if record_type == "handoff" and not value.sealed_at:
        issues.append(LedgerValidationIssue(code="handoff.seal_missing", message="handoff requires sealed_at"))
    return issues


def record_digest(record: BaseModel | dict[str, Any]) -> str:
    data = _dump(record)
    if isinstance(data, dict):
        data = {key: value for key, value in data.items() if key != "content_hash"}
    return content_hash(data)


def snapshot_from_runtime_manifest(manifest: dict[str, Any], manifest_path: str = "") -> EffectiveRunSnapshot:
    run = manifest.get("run") if isinstance(manifest.get("run"), dict) else {}
    execution = manifest.get("execution") if isinstance(manifest.get("execution"), dict) else {}
    task_contract_ref = str(
        execution.get("task_contract_ref")
        or manifest.get("contract_bundle_ref")
        or manifest.get("task_contract_ref")
        or ""
    )
    task_bundle_digest = str(
        execution.get("task_bundle_digest")
        or manifest.get("task_bundle_digest")
        or ""
    )
    task_artifact_manifest_ref = str(
        execution.get("task_artifact_manifest_ref")
        or manifest.get("task_artifact_manifest_ref")
        or ""
    )
    runtime_digest = str(execution.get("runtime_digest") or "")
    payload_digest = str(execution.get("payload_digest") or "")
    run_id = str(run.get("run_id") or manifest.get("run_id") or "")
    return EffectiveRunSnapshot(
        id=f"run:{run_id or manifest_path or uuid.uuid4().hex}",
        run_id=run_id,
        source_info={
            "source": manifest.get("runtime_execution", {}),
            "run": {key: run.get(key) for key in ("run_id", "task", "resume_checkpoint", "seed", "python_version") if key in run},
        },
        config=manifest.get("configuration") if isinstance(manifest.get("configuration"), dict) else {},
        environment=manifest.get("environment") if isinstance(manifest.get("environment"), dict) else {},
        assets=manifest.get("assets") if isinstance(manifest.get("assets"), dict) else {},
        policy_contract_ref=str(manifest.get("policy_contract_ref") or ""),
        distribution_manifest_ref=str(manifest.get("distribution_manifest_ref") or ""),
        launch_command_redacted=list((manifest.get("launch") or {}).get("command") or []) if isinstance(manifest.get("launch"), dict) else [],
        seeds={"seed": run.get("seed")} if "seed" in run else {},
        runtime_manifest_ref=str(manifest_path),
        task_contract_ref=task_contract_ref,
        task_contract_digest=str(execution.get("task_contract_digest") or ""),
        task_bundle_digest=task_bundle_digest,
        task_artifact_manifest_ref=task_artifact_manifest_ref,
        runtime_digest=runtime_digest,
        payload_digest=payload_digest,
        source="runtime_manifest",
    )


def resume_edge_from_runtime_manifest(manifest: dict[str, Any], manifest_path: str = "") -> ResumeEdge | None:
    edge = manifest.get("resume_edge")
    if not isinstance(edge, dict) or not edge.get("parent_checkpoint_ref"):
        return None
    run = manifest.get("run") if isinstance(manifest.get("run"), dict) else {}
    run_id = str(run.get("run_id") or manifest.get("run_id") or "")
    restored = edge.get("restored") if isinstance(edge.get("restored"), dict) else {}
    execution = manifest.get("execution") if isinstance(manifest.get("execution"), dict) else {}
    lineage = manifest.get("lineage") if isinstance(manifest.get("lineage"), dict) else {}
    status = str(edge.get("status") or "")
    return ResumeEdge(
        id=f"resume:{run_id or uuid.uuid4().hex}",
        parent_checkpoint_ref=str(edge.get("parent_checkpoint_ref")),
        child_run_ref=run_id,
        restored={str(key): bool(value) for key, value in restored.items()},
        reset_reason=edge.get("reset_reason") if isinstance(edge.get("reset_reason"), dict) else {},
        compatibility_report_ref=str(edge.get("parent_manifest") or manifest_path),
        parent_task_contract_ref=str(edge.get("parent_task_contract_ref") or ""),
        child_task_contract_ref=str(
            edge.get("child_task_contract_ref")
            or execution.get("task_contract_ref")
            or manifest.get("task_contract_ref")
            or ""
        ),
        parent_task_contract_digest=str(edge.get("parent_task_contract_digest") or ""),
        child_task_contract_digest=str(
            edge.get("child_task_contract_digest")
            or execution.get("task_contract_digest")
            or ""
        ),
        parent_bundle_digest=str(edge.get("parent_bundle_digest") or ""),
        child_bundle_digest=str(edge.get("child_bundle_digest") or execution.get("task_bundle_digest") or ""),
        parent_payload_digest=str(edge.get("parent_payload_digest") or ""),
        child_payload_digest=str(edge.get("child_payload_digest") or execution.get("payload_digest") or ""),
        proof=edge.get("proof") if isinstance(edge.get("proof"), dict) else {},
        status="complete" if status == "proven" else "partial",
        source="runtime_manifest",
    )


class LedgerValidationError(ValueError):
    def __init__(self, issues: Iterable[LedgerValidationIssue]):
        self.issues = list(issues)
        super().__init__("; ".join(issue.message for issue in self.issues))


class ResearchLedgerStore:
    """Append-only JSONL store with a tamper-evident hash chain."""

    _lock = threading.Lock()

    def __init__(self, root: str | os.PathLike[str] = "output/research/ledger") -> None:
        self.root = Path(root)
        self.events_path = self.root / "events.jsonl"

    def _read_events(self, verify: bool = True) -> list[LedgerEvent]:
        if not self.events_path.is_file():
            return []
        events: list[LedgerEvent] = []
        previous = ""
        for expected_sequence, line in enumerate(self.events_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            event = LedgerEvent.model_validate(json.loads(line))
            if verify:
                if event.sequence != expected_sequence:
                    raise LedgerValidationError([LedgerValidationIssue(code="ledger.sequence_gap", message=f"expected sequence {expected_sequence}, got {event.sequence}")])
                if event.previous_event_hash != previous:
                    raise LedgerValidationError([LedgerValidationIssue(code="ledger.chain_break", message=f"chain break at sequence {event.sequence}")])
                body = event.model_dump(mode="json", exclude={"event_hash"})
                if event.event_hash != content_hash(body):
                    raise LedgerValidationError([LedgerValidationIssue(code="ledger.hash_mismatch", message=f"hash mismatch at sequence {event.sequence}")])
            previous = event.event_hash
            events.append(event)
        return events

    def append(self, record_type: str, record: BaseModel | dict[str, Any], *, actor: str = "system", event_type: Literal["append", "supersede"] = "append") -> LedgerEvent:
        model = RECORD_MODELS.get(record_type)
        if model is None:
            raise LedgerValidationError([LedgerValidationIssue(code="record.unknown_type", message=f"unknown record type: {record_type}")])
        value = record if isinstance(record, model) else model.model_validate(record)
        issues = validate_record(record_type, value)
        if any(item.severity == "error" for item in issues):
            raise LedgerValidationError(issues)
        payload = _dump(value)
        if not isinstance(payload, dict):
            raise LedgerValidationError([LedgerValidationIssue(code="record.payload_invalid", message="record payload must be an object")])
        with self._lock:
            events = self._read_events(verify=True)
            previous_record = next(
                (event for event in reversed(events) if event.record_type == record_type and event.record_id == str(payload.get("id") or "")),
                None,
            )
            if previous_record is not None and record_type == "contract_bundle":
                previous_status = str(previous_record.payload.get("status") or "")
                if previous_status == "approved" and previous_record.payload != payload:
                    raise LedgerValidationError([
                        LedgerValidationIssue(
                            code="contract.approved_immutable",
                            message="approved ContractBundle is immutable; create a new version/id",
                        )
                    ])
            previous = events[-1].event_hash if events else ""
            event = LedgerEvent(
                event_id=f"event:{uuid.uuid4().hex}",
                sequence=len(events) + 1,
                event_type=event_type,
                record_type=record_type,
                record_id=str(payload.get("id") or ""),
                actor=actor,
                created_at=utc_now(),
                payload=payload,
                payload_hash=content_hash(payload),
                previous_event_hash=previous,
                event_hash="",
            )
            event.event_hash = content_hash(event.model_dump(mode="json", exclude={"event_hash"}))
            self.root.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical_json(event) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return event

    def import_runtime_manifest(self, path: str | os.PathLike[str], *, actor: str = "runtime-import") -> list[LedgerEvent]:
        manifest_path = Path(path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing = {(event.record_type, event.record_id) for event in self._read_events(verify=True)}
        events: list[LedgerEvent] = []

        def append_once(record_type: str, record: BaseModel) -> None:
            key = (record_type, str(getattr(record, "id", "")))
            if key in existing:
                return
            event = self.append(record_type, record, actor=actor)
            existing.add(key)
            events.append(event)

        run = manifest.get("run") if isinstance(manifest.get("run"), dict) else {}
        run_id = str(run.get("run_id") or manifest.get("run_id") or "")
        measured_at = str(manifest.get("updated_at") or manifest.get("generated_at") or "")
        append_once("effective_run_snapshot", snapshot_from_runtime_manifest(manifest, str(manifest_path)))

        evidence = manifest.get("evidence") if isinstance(manifest.get("evidence"), dict) else {}
        telemetry = evidence.get("telemetry") if isinstance(evidence.get("telemetry"), list) else []
        telemetry = [str(item) for item in telemetry if str(item).strip()]
        if not telemetry:
            telemetry_path = (manifest.get("paths") or {}).get("telemetry_jsonl") if isinstance(manifest.get("paths"), dict) else ""
            if telemetry_path:
                telemetry = [str(telemetry_path)]
        if run_id and telemetry:
            append_once(
                "evidence",
                EvidenceRecord(
                    id=f"evidence:{run_id}:telemetry",
                    kind="telemetry",
                    run_ref=run_id,
                    checkpoint_ref=str(run.get("resume_checkpoint") or ""),
                    measured_at=measured_at,
                    raw_artifact_ref=telemetry[0],
                    extracted_facts=[
                        {"name": "runtime_execution", "status": (manifest.get("runtime_execution") or {}).get("status", "unknown")},
                        {"name": "optimization_state", "status": (manifest.get("optimization_state") or {}).get("status", "unknown")},
                        {"name": "coverage_window", "value": manifest.get("coverage_window", "")},
                        {"name": "command_bucket_count", "value": len(manifest.get("command_coverage", [])) if isinstance(manifest.get("command_coverage"), list) else 0},
                    ],
                    observer="machine",
                    freshness="current",
                ),
            )
        if run_id and isinstance(manifest.get("runtime_execution"), dict):
            append_once(
                "evidence",
                EvidenceRecord(
                    id=f"evidence:{run_id}:source-audit",
                    kind="source_audit",
                    run_ref=run_id,
                    measured_at=measured_at,
                    raw_artifact_ref=str(manifest_path),
                    extracted_facts=[
                        {"name": "runtime_execution", "status": manifest["runtime_execution"].get("status", "unknown")},
                        {"name": "remote_verification", "status": manifest["runtime_execution"].get("remote_verification", "unknown")},
                        {"name": "runtime_preflight", "status": (manifest.get("runtime_preflight") or {}).get("status", "unknown")},
                    ],
                    observer="machine",
                    freshness="current",
                ),
            )
        edge = resume_edge_from_runtime_manifest(manifest, str(manifest_path))
        if edge is not None:
            append_once("resume_edge", edge)
        distribution = manifest.get("training_distribution")
        if isinstance(distribution, dict):
            distribution.setdefault("id", f"distribution:{manifest.get('run', {}).get('run_id') or uuid.uuid4().hex}")
            distribution.setdefault("run_id", run_id)
            append_once("training_distribution_manifest", TrainingDistributionManifest.model_validate(distribution))
        return events

    def events(self, *, verify: bool = True) -> list[LedgerEvent]:
        return self._read_events(verify=verify)

    def record_resume_proof(
        self,
        candidate_manifest: str | os.PathLike[str] | Mapping[str, Any],
        *,
        parent_manifest: str | os.PathLike[str] | Mapping[str, Any],
        checkpoint_inventory: Mapping[str, Any],
        runtime_preflight: Mapping[str, Any],
        restored: Mapping[str, Any],
        actor: str = "resume-proof",
    ) -> LedgerEvent:
        """仅凭运行时回执升级 resume edge；无法证明时保持 partial。"""
        from autotuner.execution.compatibility import evaluate_resume_proof

        def load(value: str | os.PathLike[str] | Mapping[str, Any]) -> dict[str, Any]:
            if isinstance(value, Mapping):
                return dict(value)
            return json.loads(Path(value).read_text(encoding="utf-8"))

        candidate = load(candidate_manifest)
        parent = load(parent_manifest)
        edge = resume_edge_from_runtime_manifest(candidate, str(candidate_manifest))
        if edge is None:
            raise LedgerValidationError([
                LedgerValidationIssue(
                    code="resume.edge_missing",
                    message="candidate runtime manifest has no resume edge",
                )
            ])
        proof = evaluate_resume_proof(
            parent,
            candidate,
            checkpoint=checkpoint_inventory,
            runtime_preflight=runtime_preflight,
            restored=restored,
        )
        current = self.latest("resume_edge", edge.id)
        if current is not None and current.get("status") == "complete" and not proof.proven:
            raise LedgerValidationError([
                LedgerValidationIssue(
                    code="resume.proven_immutable",
                    message="a proven resume edge cannot be downgraded by a later incomplete receipt",
                )
            ])
        updated = edge.model_copy(update={
            "status": "complete" if proof.proven else "partial",
            "proof": proof.to_dict(),
            "compatibility_report_ref": "resume-proof",
        })
        return self.append("resume_edge", updated, actor=actor, event_type="supersede")

    def latest(self, record_type: str, record_id: str) -> dict[str, Any] | None:
        matches = [event for event in self._read_events() if event.record_type == record_type and event.record_id == record_id]
        return matches[-1].payload if matches else None

    def records(self, record_type: str = "") -> list[dict[str, Any]]:
        latest: dict[tuple[str, str], LedgerEvent] = {}
        for event in self._read_events():
            latest[(event.record_type, event.record_id)] = event
        return [event.payload for event in latest.values() if not record_type or event.record_type == record_type]

    def summary(self) -> dict[str, Any]:
        events = self._read_events(verify=True)
        counts: dict[str, int] = {}
        for event in events:
            counts[event.record_type] = counts.get(event.record_type, 0) + 1
        return {
            "schema_version": SCHEMA_VERSION,
            "root": str(self.root.resolve()),
            "events_path": str(self.events_path.resolve()),
            "chain_verified": True,
            "event_count": len(events),
            "record_event_counts": counts,
            "latest_event": events[-1].event_id if events else "",
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect or import the local RL research ledger.")
    parser.add_argument("--root", default=os.environ.get("LOCOMOTION_RESEARCH_LEDGER_ROOT", "output/research_ledger"))
    parser.add_argument("--runtime-manifest", default="")
    parser.add_argument("--actor", default="cli")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    store = ResearchLedgerStore(args.root)
    if args.runtime_manifest:
        events = store.import_runtime_manifest(args.runtime_manifest, actor=args.actor)
        print(json.dumps({"imported": [event.event_id for event in events], **store.summary()}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(store.summary(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
