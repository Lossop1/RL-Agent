from __future__ import annotations

import yaml
from types import SimpleNamespace

from autotuner.llm_gateway.schemas import TaskSpecVocabulary, validate_task_spec
from autotuner.locomotion_console import ops_state
from autotuner.product import ProductRegistry, load_product_plugin, resolve_product_contract
from autotuner.product.runtime import ProductRuntimeView


def test_virtual_product_owns_intake_framework_diagnostics_and_plugins(tmp_path, monkeypatch) -> None:
    product_root = tmp_path / "config" / "products"
    product_root.mkdir(parents=True)
    task_root = tmp_path / "virtual_task"
    task_root.mkdir()
    (task_root / "config.yaml").write_text("task: virtual\n", encoding="utf-8")
    (tmp_path / "virtual_plugins.py").write_text(
        "def build_context_pack(query='', include_docs=True):\n"
        "    return {'product': 'beta', 'query': query}\n"
        "\n"
        "def get_playbook(task='', gate='', robot=''):\n"
        "    return {'product': 'beta', 'task': task, 'gate': gate, 'robot': robot}\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    manifest = {
        "schema_version": 1,
        "product": {"id": "beta", "version": "1", "label": "Beta", "status": "draft"},
        "robot": {
            "id": "robot.beta",
            "label": "Beta robot",
            "status": "draft",
            "dof": 1,
            "base_link": "base",
            "joint_order": ["joint"],
        },
        "task": {
            "id": "beta_locomotion",
            "family": "locomotion",
            "framework_id": "beta_framework",
            "train_entrypoint": "virtual_plugins:train",
            "diagnose_entrypoint": "virtual_plugins:diagnose",
            "intake": {
                "robot_ids": ["robot.beta"],
                "task_classes": ["locomotion"],
                "terrains": ["flat", "stairs"],
                "ambition_levels": ["baseline_walking"],
            },
        },
        "sources": {"config": "virtual_task/config.yaml"},
        "diagnostics": {
            "payload": {
                "payload_roots": ["/beta/payloads"],
                "payload_globs": ["beta_*"],
                "required_files": ["beta/diagnose.py"],
            },
            "presets": [{"id": "quick", "stages": [{"id": "quick"}]}],
        },
        "framework": {
            "default_profile": "beta_framework",
            "profiles": {
                "beta_framework": {
                    "label": "Beta framework",
                    "run_globs": ["/beta/runs/*/"],
                    "checkpoint_roots": ["/beta/runs"],
                    "commands": {"train": "python -m virtual_plugins"},
                }
            },
        },
        "deployment": {
            "configuration_artifacts": {
                "source_config": "beta_config.yaml",
                "effective_config": "effective_config.yaml",
            }
        },
        "runtime": {"training_process_pattern": "beta_train"},
        "plugins": {
            "knowledge": {"context_pack": "virtual_plugins:build_context_pack"},
            "playbook": {"get_playbook": "virtual_plugins:get_playbook"},
        },
    }
    (product_root / "beta.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    registry = ProductRegistry(product_root, workspace_root=tmp_path)

    contract = resolve_product_contract("beta", root=tmp_path, registry=registry)
    assert not contract.issues
    view = ProductRuntimeView(contract)
    assert view.default_framework_id() == "beta_framework"
    assert view.diagnostics["payload"]["payload_globs"] == ["beta_*"]
    assert view.runtime["schema_version"] == "rl-agent.runtime-identity/v1"
    assert view.runtime["declaration"]["training_process_pattern"] == "beta_train"

    monkeypatch.setattr(ops_state, "resolve_product_runtime", lambda _product_id: view)
    declarations = ops_state._declarations(SimpleNamespace(settings=SimpleNamespace(product_id="beta")))
    assert declarations["product_id"] == "beta"
    assert declarations["process_pattern"] == "beta_train"
    assert declarations["run_glob"] == "/beta/runs/*/"

    vocabulary = TaskSpecVocabulary.from_mapping(contract.training["intake"])
    task = validate_task_spec(
        {
            "robot_id": "robot.beta",
            "task_class": "locomotion",
            "terrain": "stairs",
            "ambition": "baseline_walking",
            "max_command_lin_vel_mps": 0.5,
            "user_clarifications_needed": [],
            "confidence": 1.0,
        },
        raw_text="virtual product",
        llm_model="test",
        vocabulary=vocabulary,
    )
    assert task.robot_id == "robot.beta"
    assert load_product_plugin(contract, "knowledge", "context_pack")(query="x")["product"] == "beta"
    assert load_product_plugin(contract, "playbook", "get_playbook")(task="verify")["product"] == "beta"
