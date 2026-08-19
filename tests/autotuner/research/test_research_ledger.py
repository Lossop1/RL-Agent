"""Tests for the durable, append-only RL research ledger."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from autotuner.research.research_ledger import (
    ContractBundle,
    EvidenceRecord,
    ExperimentPlan,
    LedgerValidationError,
    ResearchLedgerStore,
    approve_contract,
    KnowledgeClaim,
    snapshot_from_runtime_manifest,
    validate_record,
)


def _approved_contract() -> ContractBundle:
    return ContractBundle(
        id="contract/taili@2026-08-15",
        version="2026-08-15",
        status="approved",
        approved_by="human",
        goals=["flat quality", "stairs quality", "robustness"],
        protected_capabilities=["flat", "stairs", "stand"],
    )


def _plan(**overrides) -> ExperimentPlan:
    data = {
        "id": "experiment:stand-brake",
        "problem_statement": "zero-command stop is too slow",
        "baseline_ref": "baseline:taili",
        "hypotheses_addressed": ["hypothesis:brake-drive"],
        "intervention_diff": {"reward.w_stop": {"old": 0.0, "new": 1.0}},
        "unchanged_fields": ["reward.tracking_lin", "terrain.curriculum"],
        "protected_capabilities": ["flat", "stairs"],
        "evaluation_plan": {"suite": "flat+stairs", "window": "full"},
        "success_condition": "stand and protected capabilities pass",
        "rollback_condition": "stairs or flat regresses beyond allowed band",
        "resource_budget": {"gpu_hours": 4},
        "risk_tier": "high",
        "authorization_ref": "lease:human-1",
    }
    data.update(overrides)
    return ExperimentPlan(**data)


def test_high_risk_experiment_requires_authorization_and_protected_fields():
    issues = validate_record("experiment_plan", _plan(authorization_ref=""))
    codes = {item.code for item in issues}
    assert "experiment.authorization_missing" in codes


def test_approval_creates_a_new_immutable_contract_version():
    draft = ContractBundle(id="contract:draft", goals=["quality"], protected_capabilities=["flat"])
    approved = approve_contract(draft, "human")
    assert draft.status == "draft"
    assert approved.status == "approved"
    assert approved.id != draft.id
    assert approved.supersedes == [draft.id]


def test_validated_knowledge_claim_requires_evidence():
    claim = KnowledgeClaim(id="claim:1", statement="claim", status="validated")
    assert "claim.evidence_missing" in {item.code for item in validate_record("knowledge_claim", claim)}


def test_human_evidence_requires_context_and_structured_facts():
    evidence = EvidenceRecord(
        id="evidence:video-1",
        kind="video_observation",
        observer="human",
        raw_artifact_ref="artifact:video-1",
    )
    codes = {item.code for item in validate_record("evidence", evidence)}
    assert {"evidence.scene_missing", "evidence.timeline_missing", "evidence.facts_missing"} <= codes


def test_store_is_append_only_and_detects_tampering(tmp_path: Path):
    store = ResearchLedgerStore(tmp_path)
    store.append("contract_bundle", _approved_contract(), actor="test")
    summary = store.summary()
    assert summary["chain_verified"] is True
    assert summary["event_count"] == 1

    lines = store.events_path.read_text(encoding="utf-8").splitlines()
    item = json.loads(lines[0])
    item["payload"]["goals"] = ["tampered"]
    store.events_path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    with pytest.raises(LedgerValidationError):
        store.summary()


def test_approved_contract_cannot_be_overwritten(tmp_path: Path):
    store = ResearchLedgerStore(tmp_path)
    approved = _approved_contract()
    store.append("contract_bundle", approved, actor="test")
    changed = approved.model_copy(update={"goals": ["changed"]})
    with pytest.raises(LedgerValidationError):
        store.append("contract_bundle", changed, actor="test")


def test_runtime_manifest_import_creates_snapshot_and_resume_edge(tmp_path: Path):
    manifest = {
        "run": {"run_id": "run-1", "task": "taili", "resume_checkpoint": "/runs/r1/checkpoints/agent_10.pt"},
        "configuration": {"effective_config": {"sha256": "abc"}},
        "resume_edge": {
            "parent_checkpoint_ref": "/runs/r0/checkpoints/agent_9.pt",
            "status": "proven",
            "restored": {"policy": True, "optimizer": True},
        },
    }
    path = tmp_path / "runtime_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    store = ResearchLedgerStore(tmp_path / "ledger")
    events = store.import_runtime_manifest(path)
    assert len(events) == 2
    assert store.records("effective_run_snapshot")[0]["run_id"] == "run-1"
    assert store.records("resume_edge")[0]["status"] == "complete"


def test_snapshot_from_runtime_manifest_keeps_provenance_boundary():
    snapshot = snapshot_from_runtime_manifest(
        {"run": {"run_id": "run-2"}, "runtime_execution": {"status": "declared"}},
        "runs/run-2/runtime_manifest.json",
    )
    assert snapshot.run_id == "run-2"
    assert snapshot.source_info["source"]["status"] == "declared"
    assert snapshot.runtime_manifest_ref.endswith("runtime_manifest.json")


def test_runtime_manifest_import_records_evidence_and_is_idempotent(tmp_path: Path):
    manifest = {
        "run": {"run_id": "run-evidence", "resume_checkpoint": ""},
        "paths": {"telemetry_jsonl": "run-evidence/train.telemetry.jsonl"},
        "runtime_execution": {"status": "proven", "remote_verification": "proven"},
        "runtime_preflight": {"status": "pass"},
        "optimization_state": {"status": "proven", "parity_check": True},
        "command_coverage": [{"bucket": "forward", "sample_count": 2}],
        "coverage_window": "steps:1-2",
        "training_distribution": {
            "schema_version": "rl-agent.training-distribution/v1",
            "command_buckets": {},
            "terrain_mix": {},
            "curriculum_state": {},
            "dr_channels": {},
            "reset_sampling": {},
            "realized_window": {},
        },
    }
    path = tmp_path / "runtime_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    store = ResearchLedgerStore(tmp_path / "ledger")

    first = store.import_runtime_manifest(path)
    second = store.import_runtime_manifest(path)

    assert len(first) == 4
    assert second == []
    assert store.summary()["event_count"] == 4
    evidence = store.records("evidence")
    assert {item["kind"] for item in evidence} == {"telemetry", "source_audit"}
    assert store.records("effective_run_snapshot")[0]["run_id"] == "run-evidence"
