from __future__ import annotations

import json
from pathlib import Path

import pytest

from autotuner.product import (
    ProductPayload,
    TaskBundleMaterializer,
    TaskContractCompiler,
    TaskContractError,
    TaskContractStore,
    TaskExecutionPipeline,
    TrainingLaunchRequest,
    resolve_product_contract,
)
from autotuner.product import task_materializer as task_materializer_module
from autotuner.research.research_ledger import snapshot_from_runtime_manifest


def _request(*, approved: bool = False, objective: str = "验证任务合同闭环") -> dict:
    return {
        "task": {
            "instance_id": "artifact-test",
            "objective": objective,
            "goals": [{"id": "quality", "objective": "保持产品能力并可追溯地生成产物"}],
            "protected_capabilities": ["quality"],
            "approved": approved,
        }
    }


def test_task_contract_store_keeps_versions_parent_refs_and_status_events(tmp_path: Path) -> None:
    product = resolve_product_contract("taili")
    compiler = TaskContractCompiler()
    first = compiler.compile_bundle(product, _request())
    store = TaskContractStore(tmp_path / "contracts")

    first_record = store.save(first, actor="test")
    assert first_record.ref == "artifact-test.taili@1"
    assert (first_record.path / "bundle_manifest.json").is_file()

    second = compiler.compile_bundle(
        product,
        _request(objective="修订任务合同但保留产品边界"),
        contract_version=2,
        contract_id=first.contract.contract_id,
    )
    second_record = store.save(second, parent_ref=first_record.ref, actor="test")
    assert second_record.parent_ref == first_record.ref

    rejected = store.set_status(second_record.ref, "rejected", actor="test", reason="测试回滚")
    assert rejected.status == "rejected"
    assert store.get(second_record.ref).status == "rejected"
    assert [item.contract_version for item in store.history("artifact-test.taili")] == [1, 2]

    restored = store.load_bundle(first_record.ref)
    assert restored.bundle_digest == first.bundle_digest
    assert restored.contract.contract_digest == first.contract.contract_digest


def test_approved_contract_is_explicit_and_immutable(tmp_path: Path) -> None:
    product = resolve_product_contract("taili")
    bundle = TaskContractCompiler().compile_bundle(product, _request(approved=True))
    store = TaskContractStore(tmp_path / "contracts")

    record = store.save(bundle, actor="reviewer")
    assert record.status == "approved"

    changed = TaskContractCompiler().compile_bundle(
        product,
        _request(approved=True, objective="不允许覆盖已批准版本"),
        contract_version=1,
        contract_id=bundle.contract.contract_id,
    )
    with pytest.raises(TaskContractError, match="different content"):
        store.save(changed)


def test_materializer_generates_five_checked_artifacts_and_rejects_tampering(tmp_path: Path) -> None:
    product = resolve_product_contract("taili")
    bundle = TaskContractCompiler().compile_bundle(product, _request())
    result = TaskBundleMaterializer().materialize(
        bundle,
        tmp_path / "artifacts",
        product=product,
        bundle_ref="artifact-test.taili@1",
    )

    assert {item.kind for item in result.artifacts} == {
        "training",
        "telemetry",
        "diagnostics",
        "simulation",
        "deployment",
    }
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["bundle_digest"] == bundle.bundle_digest
    assert all((result.root / item["path"]).is_file() for item in manifest["artifacts"])

    training = result.root / "artifacts" / "training.json"
    training.write_text(training.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    with pytest.raises(TaskContractError, match="missing or changed"):
        TaskBundleMaterializer().materialize(bundle, result.root, product=product)


def test_runtime_snapshot_keeps_task_bundle_provenance() -> None:
    snapshot = snapshot_from_runtime_manifest(
        {
            "execution": {
                "task_contract_ref": "artifact-test.taili@1",
                "task_bundle_digest": "b" * 64,
                "task_artifact_manifest_ref": "artifact-test.taili@1:manifest",
            },
            "run": {"run_id": "run-1"},
        },
        "run-1/runtime_manifest.json",
    )

    assert snapshot.task_contract_ref == "artifact-test.taili@1"
    assert snapshot.task_bundle_digest == "b" * 64
    assert snapshot.task_artifact_manifest_ref == "artifact-test.taili@1:manifest"


def test_materializer_accepts_a_product_specific_plugin(monkeypatch, tmp_path: Path) -> None:
    product = resolve_product_contract("taili")
    bundle = TaskContractCompiler().compile_bundle(product, _request())

    def virtual_materializer(*, bundle, output_dir):
        return {
            "artifacts": {
                kind: {"virtual_product": "beta", "kind": kind}
                for kind in ("training", "telemetry", "diagnostics", "simulation", "deployment")
            }
        }

    monkeypatch.setattr(
        task_materializer_module,
        "load_product_plugin",
        lambda *_args, **_kwargs: virtual_materializer,
    )
    result = TaskBundleMaterializer().materialize(
        bundle,
        tmp_path / "plugin-artifacts",
        product=product,
    )
    training = json.loads((result.root / "artifacts" / "training.json").read_text(encoding="utf-8"))
    assert training["source"] == "product_materializer"
    assert training["spec"]["virtual_product"] == "beta"


def test_task_pipeline_binds_payload_runtime_and_ledger_provenance(monkeypatch, tmp_path: Path) -> None:
    product = resolve_product_contract("taili")
    bundle = TaskContractCompiler().compile_bundle(product, _request())

    def fake_payload(*, contract, output_dir, task_bundle, task_artifacts):
        return ProductPayload(
            archive=tmp_path / "payload.tar.gz",
            manifest=tmp_path / "payload.json",
            payload_digest="p" * 64,
        )

    monkeypatch.setattr("autotuner.product.task_pipeline.build_product_payload", fake_payload)
    result = TaskExecutionPipeline(tmp_path / "pipeline").prepare(
        product,
        bundle,
        run_id="run-1",
    )

    manifest = json.loads(result.run_manifest.read_text(encoding="utf-8"))
    assert manifest["execution"]["task_contract_ref"] == result.stored_contract.ref
    assert manifest["execution"]["task_artifact_manifest_ref"] == result.materialized.manifest_ref
    assert result.deployment_spec.task_artifact_manifest_ref == result.materialized.manifest_ref
    assert {event.record_type for event in result.ledger_events} >= {
        "contract_bundle",
        "effective_run_snapshot",
    }


def test_evidence_revision_creates_parent_and_supersedes_edges(tmp_path: Path) -> None:
    product = resolve_product_contract("taili")
    compiler = TaskContractCompiler()
    base = compiler.compile_bundle(product, _request(approved=True))
    pipeline = TaskExecutionPipeline(tmp_path / "pipeline")
    pipeline.contract_store.save(base, actor="test")

    revised, stored = pipeline.revise(
        product,
        base,
        {"changes": {"training": {"experiment": {"mode": "evidence_driven"}}}},
        evidence_refs=["evidence:run-1:telemetry"],
        approved_by="human",
    )

    assert revised.contract.contract_version == 2
    assert revised.contract.provenance["revision"]["evidence_refs"] == [
        "evidence:run-1:telemetry"
    ]
    assert stored.parent_ref == "artifact-test.taili@1"
    assert stored.supersedes_ref == "artifact-test.taili@1"


def test_task_pipeline_records_declared_resume_edge(monkeypatch, tmp_path: Path) -> None:
    product = resolve_product_contract("taili")
    bundle = TaskContractCompiler().compile_bundle(product, _request(approved=True))

    def fake_payload(*, contract, output_dir, task_bundle, task_artifacts):
        return ProductPayload(
            archive=tmp_path / "payload.tar.gz",
            manifest=tmp_path / "payload.json",
            payload_digest="q" * 64,
        )

    monkeypatch.setattr("autotuner.product.task_pipeline.build_product_payload", fake_payload)
    request = TrainingLaunchRequest(
        payload_root="/remote/payload",
        run_id="run-resume",
        checkpoint="/remote/runs/parent/checkpoints/agent_10.pt",
        source_run="/remote/runs/parent",
        resume=True,
        resume_state={"phase": True},
    )
    result = TaskExecutionPipeline(tmp_path / "pipeline").prepare(
        product,
        bundle,
        run_id="run-resume",
        launch_request=request,
    )

    manifest = json.loads(result.run_manifest.read_text(encoding="utf-8"))
    assert manifest["resume_edge"]["parent_checkpoint_ref"].endswith("agent_10.pt")
    assert manifest["resume_edge"]["status"] == "declared"
    edge = next(event for event in result.ledger_events if event.record_type == "resume_edge")
    assert edge.payload["status"] == "partial"
