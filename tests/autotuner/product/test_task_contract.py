from __future__ import annotations

import json

import pytest

from autotuner.execution import DeploymentSpec
from autotuner.product import (
    TaskContractCompiler,
    TaskContractError,
    TaskRequest,
    resolve_product_contract,
)


def _request(*, approved: bool = False) -> dict:
    return {
        "task": {
            "instance_id": "taili-research",
            "objective": "得到可部署且可追溯的盲态运动策略",
            "goals": [
                {"id": "flat_quality", "objective": "平地步态、核心和速度跟踪满足验收"},
                {"id": "stairs", "objective": "楼梯通过安全、稳定且不依赖特权信息"},
            ],
            "constraints": {"deployment_requires_privileged_terrain": False},
            "protected_capabilities": ["flat_quality", "stairs"],
            "training": {"experiment": {"mode": "resume_or_fresh_with_evidence"}},
            "simulation": {"sim2sim": {"required": True, "worlds": ["mujoco"]}},
            "approved": approved,
        }
    }


def test_task_contract_compiler_projects_all_runtime_specs(tmp_path) -> None:
    product = resolve_product_contract("taili")
    compiler = TaskContractCompiler()
    bundle = compiler.compile_bundle(product, _request())

    contract = bundle.contract
    assert contract.status == "draft"
    assert contract.product_contract_digest == product.contract_digest
    assert contract.training["task_id"] == "taili_blind_locomotion"
    assert contract.training["experiment"]["mode"] == "resume_or_fresh_with_evidence"
    assert contract.diagnostics["entrypoint"]
    assert contract.simulation["sim2sim"]["required"] is True
    assert contract.runtime["digest"] == product.runtime["digest"]
    assert set(bundle.specs) == {"training", "telemetry", "diagnostics", "simulation", "deployment"}
    assert all(spec["contract_digest"] == contract.contract_digest for spec in bundle.specs.values())

    output = bundle.write(tmp_path / "generated")
    manifest = json.loads((output / "bundle_manifest.json").read_text(encoding="utf-8"))
    assert manifest["bundle_digest"] == bundle.bundle_digest
    assert json.loads((output / "task_contract.json").read_text(encoding="utf-8"))["contract_digest"] == contract.contract_digest


def test_task_contract_compilation_is_deterministic() -> None:
    product = resolve_product_contract("taili")
    compiler = TaskContractCompiler()
    first = compiler.compile_bundle(product, _request())
    second = compiler.compile_bundle(product, _request())

    assert first.contract.to_dict() == second.contract.to_dict()
    assert first.bundle_digest == second.bundle_digest


def test_task_request_requires_explicit_objective_and_goal_text() -> None:
    with pytest.raises(TaskContractError, match="objective is required"):
        TaskRequest.from_mapping({"instance_id": "x"})
    with pytest.raises(TaskContractError, match="needs objective"):
        TaskRequest.from_mapping({"instance_id": "x", "objective": "x", "goals": [{"id": "g"}]})


def test_approved_request_changes_only_contract_status() -> None:
    product = resolve_product_contract("taili")
    compiler = TaskContractCompiler()
    draft = compiler.compile(product, _request(approved=False))
    approved = compiler.compile(product, _request(approved=True))

    assert draft.status == "draft"
    assert approved.status == "approved"
    assert draft.provenance["approval_required"] is True
    assert approved.provenance["approval_required"] is False


def test_bundle_exposes_product_neutral_execution_handoff() -> None:
    product = resolve_product_contract("taili")
    bundle = TaskContractCompiler().compile_bundle(product, _request(approved=True))
    spec = bundle.deployment_spec(
        payload_archive="payload.tar.gz",
        payload_manifest="payload_manifest.json",
        run_id="run-1",
        run_manifest={"run_id": "run-1"},
    )

    assert isinstance(spec, DeploymentSpec)
    spec.validate()
    assert spec.runtime_digest == product.runtime["digest"]


def test_draft_contract_cannot_create_execution_handoff() -> None:
    product = resolve_product_contract("taili")
    bundle = TaskContractCompiler().compile_bundle(product, _request())

    with pytest.raises(TaskContractError, match="approved contract required"):
        bundle.deployment_spec(
            payload_archive="payload.tar.gz",
            payload_manifest="payload_manifest.json",
            run_id="run-1",
            run_manifest={"run_id": "run-1"},
        )


@pytest.mark.parametrize(
    ("section", "value", "path"),
    [
        ("training", {"train_entrypoint": "evil.module:main"}, "training.train_entrypoint"),
        ("deployment", {"payload_builder": "evil.module:build"}, "deployment.payload_builder"),
        ("deployment", {"remote_root": "/tmp/other"}, "deployment.remote_root"),
    ],
)
def test_task_cannot_override_product_owned_fields(section: str, value: dict, path: str) -> None:
    product = resolve_product_contract("taili")
    request = _request()
    request["task"][section] = value

    with pytest.raises(TaskContractError, match=path):
        TaskContractCompiler().compile(product, request)


def test_product_requirements_are_hard_constraints() -> None:
    product = resolve_product_contract("taili")
    request = _request()
    contract = TaskContractCompiler().compile(product, request)

    assert contract.constraints["actor_observation"] == "proprioceptive_only"
    assert contract.training["constraints"] == contract.constraints

    request["task"]["constraints"]["actor_observation"] = "privileged_terrain"
    with pytest.raises(TaskContractError, match="constraints.actor_observation"):
        TaskContractCompiler().compile(product, request)
