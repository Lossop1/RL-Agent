from __future__ import annotations

from autotuner.execution.compatibility import (
    check_resume_compatibility,
    evaluate_resume_proof,
)
from autotuner.research.research_ledger import ResearchLedgerStore


def _manifest(*, run_id: str, contract_digest: str, bundle_digest: str, payload_digest: str, resume: bool = False) -> dict:
    manifest = {
        "run": {"run_id": run_id, "task": "taili"},
        "product": {"id": "taili", "version": "1"},
        "resolved_contract": {
            "product_id": "taili",
            "product_version": "1",
            "config_digest": "config",
            "source_digest": "source",
            "runtime": {"digest": "runtime"},
            "compatibility": {
                "observation_structure": "obs.v1",
                "action_structure": "act.v1",
                "network_structure": "net.v1",
                "normalization": "norm.v1",
                "physics_timestep": 0.0025,
            },
            "training": {"task_id": "taili"},
        },
        "execution": {
            "task_contract_ref": f"task.taili:{run_id}",
            "task_contract_digest": contract_digest,
            "task_bundle_digest": bundle_digest,
            "payload_digest": payload_digest,
            "runtime_digest": "runtime",
        },
    }
    if resume:
        manifest.update(
            {
                "runtime_execution": {
                    "status": "proven",
                    "remote_verification": "proven",
                },
                "resume_edge": {
                    "parent_checkpoint_ref": "/runs/parent/checkpoints/agent_10.pt",
                    "parent_task_contract_ref": "task.taili:parent",
                    "parent_task_contract_digest": "parent-contract",
                    "parent_bundle_digest": "parent-bundle",
                    "parent_payload_digest": "parent-payload",
                    "status": "declared",
                    "restored": {},
                },
            }
        )
    return manifest


def _checkpoint() -> dict:
    return {
        "status": "proven",
        "key_paths": [
            "models.policy",
            "models.value",
            "optimizers.policy",
            "state_preprocessor",
        ],
    }


def _preflight() -> dict:
    return {"status": "proven", "remote_boot_id": "boot-1"}


def test_resume_proof_stays_partial_until_all_state_is_confirmed(tmp_path):
    parent = _manifest(
        run_id="parent",
        contract_digest="parent-contract",
        bundle_digest="parent-bundle",
        payload_digest="parent-payload",
    )
    candidate = _manifest(
        run_id="child",
        contract_digest="child-contract",
        bundle_digest="child-bundle",
        payload_digest="child-payload",
        resume=True,
    )
    store = ResearchLedgerStore(tmp_path / "ledger")

    partial = store.record_resume_proof(
        candidate,
        parent_manifest=parent,
        checkpoint_inventory=_checkpoint(),
        runtime_preflight=_preflight(),
        restored={"policy": True, "value": True, "optimizer": True, "normalizer": True},
    )
    assert partial.payload["status"] == "partial"
    assert partial.payload["proof"]["status"] == "blocked"

    complete = store.record_resume_proof(
        candidate,
        parent_manifest=parent,
        checkpoint_inventory=_checkpoint(),
        runtime_preflight=_preflight(),
        restored={
            "policy": True,
            "value": True,
            "optimizer": True,
            "normalizer": True,
            "curriculum": True,
            "rng": True,
        },
    )
    assert complete.payload["status"] == "complete"
    assert complete.payload["proof"]["status"] == "proven"
    assert store.records("resume_edge")[0]["status"] == "complete"


def test_resume_proof_records_artifact_changes_but_blocks_abi_changes():
    parent = _manifest(
        run_id="parent",
        contract_digest="parent-contract",
        bundle_digest="parent-bundle",
        payload_digest="parent-payload",
    )
    candidate = _manifest(
        run_id="child",
        contract_digest="child-contract",
        bundle_digest="child-bundle",
        payload_digest="child-payload",
        resume=True,
    )
    proof = evaluate_resume_proof(
        parent,
        candidate,
        checkpoint=_checkpoint(),
        runtime_preflight=_preflight(),
        restored={
            "policy": True,
            "value": True,
            "optimizer": True,
            "normalizer": True,
            "curriculum": True,
            "rng": True,
        },
    )
    assert proof.proven is True
    assert proof.compatibility is not None
    assert proof.compatibility.compared["payload_digest"]["class"] == "provenance"
    assert proof.compatibility.compared["payload_digest"]["change"] == "recorded_provenance_change"

    changed = _manifest(
        run_id="child",
        contract_digest="child-contract",
        bundle_digest="child-bundle",
        payload_digest="child-payload",
        resume=True,
    )
    changed["resolved_contract"]["compatibility"]["network_structure"] = "net.v2"
    blocked = check_resume_compatibility(
        parent,
        changed,
        checkpoint=_checkpoint(),
    )
    assert blocked.compatible is False
    assert any("network_structure" in reason for reason in blocked.reasons)


def test_resume_compatibility_blocks_missing_product_version():
    parent = _manifest(
        run_id="parent",
        contract_digest="parent-contract",
        bundle_digest="parent-bundle",
        payload_digest="parent-payload",
    )
    candidate = _manifest(
        run_id="child",
        contract_digest="child-contract",
        bundle_digest="child-bundle",
        payload_digest="child-payload",
        resume=True,
    )
    del candidate["product"]["version"]
    del candidate["resolved_contract"]["product_version"]
    decision = check_resume_compatibility(
        parent,
        candidate,
        checkpoint=_checkpoint(),
    )
    assert decision.compatible is False
    assert any("product_version" in reason for reason in decision.reasons)
