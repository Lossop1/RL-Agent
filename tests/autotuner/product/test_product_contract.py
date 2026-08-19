from __future__ import annotations

import json

import pytest

from autotuner.product import ProductManifestError, ProductRegistry, resolve_product_contract
from autotuner.product.manifest import load_product_manifest


def test_taili_contract_is_resolved_from_manifest() -> None:
    contract = resolve_product_contract("taili")

    assert contract.product_id == "taili"
    assert contract.config_digest
    assert contract.asset_digest
    assert contract.contract_digest == contract.digest()
    assert contract.training["task_id"] == "taili_blind_locomotion"
    assert contract.training["runtime_task_ids"] == [
        "RobotLab-Isaac-Taili-Blind-Direct-v0",
        "RobotLab-Isaac-Taili-AMP-Blind-Direct-v0",
    ]
    assert contract.deployment["payload_package"] == "taili_blind_runtime"
    assert not contract.issues


def test_contract_json_is_deterministic(tmp_path) -> None:
    contract = resolve_product_contract("taili")
    first = contract.write(tmp_path / "contract.json")
    first_data = json.loads(first.read_text(encoding="utf-8"))
    second = contract.write(tmp_path / "nested" / "contract.json")
    second_data = json.loads(second.read_text(encoding="utf-8"))

    assert first_data == second_data
    assert first_data["contract_digest"] == contract.contract_digest


def test_contract_source_digest_ignores_python_cache(tmp_path) -> None:
    product_root = tmp_path / "config" / "products"
    product_root.mkdir(parents=True)
    source_root = tmp_path / "task"
    source_root.mkdir()
    (source_root / "config.yaml").write_text("task: true\n", encoding="utf-8")
    module = source_root / "module.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "product": {"id": "alpha", "version": "1", "label": "Alpha", "status": "draft"},
        "robot": {
            "id": "robot.alpha",
            "label": "Alpha",
            "status": "draft",
            "dof": 1,
            "base_link": "base",
            "joint_order": ["joint"],
        },
        "task": {
            "id": "alpha_task",
            "family": "test",
            "framework_id": "test",
            "train_entrypoint": "alpha.train:main",
            "diagnose_entrypoint": "alpha.diag:main",
        },
        "sources": {"config": "task/config.yaml", "roots": ["task"]},
    }
    import yaml

    (product_root / "alpha.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    registry = ProductRegistry(product_root, workspace_root=tmp_path)
    before = resolve_product_contract("alpha", root=tmp_path, registry=registry)

    cache = source_root / "__pycache__"
    cache.mkdir()
    (cache / "module.cpython-313.pyc").write_bytes(b"generated")
    after_cache = resolve_product_contract("alpha", root=tmp_path, registry=registry)
    assert after_cache.source_digest == before.source_digest

    module.write_text("VALUE = 2\n", encoding="utf-8")
    after_source = resolve_product_contract("alpha", root=tmp_path, registry=registry)
    assert after_source.source_digest != before.source_digest


def test_registry_never_falls_back_to_another_product(tmp_path) -> None:
    product_root = tmp_path / "config" / "products"
    product_root.mkdir(parents=True)
    source = tmp_path / "task.yaml"
    source.write_text("task: true\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "product": {"id": "alpha", "version": "1", "label": "Alpha", "status": "draft"},
        "robot": {
            "id": "robot.alpha",
            "label": "Alpha",
            "status": "draft",
            "dof": 1,
            "base_link": "base",
            "joint_order": ["joint"],
        },
        "task": {
            "id": "alpha_task",
            "family": "test",
            "framework_id": "test",
            "train_entrypoint": "alpha.train:main",
            "diagnose_entrypoint": "alpha.diag:main",
        },
        "sources": {"config": "task.yaml"},
    }
    import yaml

    (product_root / "alpha.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    registry = ProductRegistry(product_root, workspace_root=tmp_path)

    assert registry.get().product_id == "alpha"
    with pytest.raises(ProductManifestError, match="unknown product"):
        registry.get("robot.missing")


def test_product_manifest_rejects_invalid_joint_mapping(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
schema_version: 1
product: {id: bad, version: '1', label: Bad, status: draft}
robot:
  id: robot.bad
  label: Bad
  status: draft
  dof: 2
  base_link: base
  joint_order: [joint]
task:
  id: bad_task
  family: test
  framework_id: test
  train_entrypoint: bad:train
  diagnose_entrypoint: bad:diag
sources: {config: missing.yaml}
""",
        encoding="utf-8",
    )
    product = load_product_manifest(path)

    assert any("joint_order" in issue for issue in product.validate(tmp_path))
