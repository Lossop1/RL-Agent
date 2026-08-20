from __future__ import annotations

from pathlib import Path

import pytest

from autotuner.artifacts import AssetCatalog, AssetCatalogError, AssetLineage


def _register(catalog: AssetCatalog):
    return catalog.register_asset(
        asset_id="flat-locomotion-baseline",
        kind="training_baseline",
        sha256="a" * 64,
        locator="archive/runs/run-17/checkpoints/agent_80000.pt",
        compatibility={"observation_abi": "obs-v3", "action_abi": "act-v2"},
        capability_scope=["flat_locomotion", "quiet_stand"],
        validation_evidence_refs=["evidence:diagnostic:run-17"],
        lineage=AssetLineage(
            source_product_ref="robot.alpha@2",
            source_task_contract_ref="task.alpha@4",
            source_run_ref="run-17",
        ),
        actor="test",
    )


def test_training_asset_requires_explicit_lineage(tmp_path: Path) -> None:
    catalog = AssetCatalog(tmp_path / "catalog")

    with pytest.raises(AssetCatalogError, match="explicit source_task_contract_ref"):
        catalog.register_asset(
            asset_id="untraced",
            kind="checkpoint",
            sha256="a" * 64,
            locator="agent.pt",
        )


def test_compatible_asset_requires_evidence_approval_and_binding(tmp_path: Path) -> None:
    catalog = AssetCatalog(tmp_path / "catalog")
    asset = _register(catalog)

    report = catalog.evaluate_compatibility(
        asset.asset_ref,
        expected_digest="a" * 64,
        target_context={"observation_abi": "obs-v3", "action_abi": "act-v2"},
        required_capabilities=["flat_locomotion"],
    )
    assert report.compatible

    approval = catalog.decide_reuse(
        asset.asset_ref,
        approval_ref="approval:baseline:18",
        expected_digest="a" * 64,
        target_contract_ref="task.beta@1",
        target_run_ref="run-18",
        role="initial_checkpoint",
        target_context={"observation_abi": "obs-v3", "action_abi": "act-v2"},
        required_capabilities=["flat_locomotion"],
        evidence_refs=["evidence:review:18"],
        approved_by="human",
        decision="approved",
        reason="ABI and protected capability checks passed",
        actor="test",
    )
    binding = catalog.bind_approved_reuse(
        approval.approval_ref,
        target_contract_ref="task.beta@1",
        target_run_ref="run-18",
        actor="test",
    )

    assert binding.asset_ref == asset.asset_ref
    assert catalog.get_binding(binding.binding_ref) == binding
    assert len(catalog.events()) == 3


def test_digest_or_compatibility_mismatch_cannot_be_approved(tmp_path: Path) -> None:
    catalog = AssetCatalog(tmp_path / "catalog")
    asset = _register(catalog)

    report = catalog.evaluate_compatibility(
        asset.asset_ref,
        expected_digest="b" * 64,
        target_context={"observation_abi": "obs-v4"},
        required_capabilities=["stairs"],
    )
    assert not report.compatible
    assert not report.digest_match
    assert report.missing_context == ("action_abi",)
    assert report.missing_capabilities == ("stairs",)

    with pytest.raises(AssetCatalogError, match="incompatible asset"):
        catalog.decide_reuse(
            asset.asset_ref,
            approval_ref="approval:bad",
            expected_digest="b" * 64,
            target_contract_ref="task.beta@1",
            target_run_ref="run-bad",
            role="initial_checkpoint",
            target_context={"observation_abi": "obs-v4"},
            required_capabilities=["stairs"],
            evidence_refs=["evidence:review:bad"],
            approved_by="human",
            decision="approved",
            reason="must not pass",
        )


def test_catalog_rejects_metadata_mutation_for_same_content_reference(tmp_path: Path) -> None:
    catalog = AssetCatalog(tmp_path / "catalog")
    _register(catalog)

    with pytest.raises(AssetCatalogError, match="different metadata"):
        catalog.register_asset(
            asset_id="flat-locomotion-baseline",
            kind="training_baseline",
            sha256="a" * 64,
            locator="different/location.pt",
            validation_evidence_refs=["evidence:diagnostic:run-17"],
            lineage=AssetLineage(
                source_task_contract_ref="task.alpha@4",
                source_run_ref="run-17",
            ),
        )
