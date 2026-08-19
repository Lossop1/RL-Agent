"""Pure tests for runtime evidence helpers; no IsaacLab or torch process."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from products.taili.blind_locomotion.runtime_manifest import (
    capture_optimization_state,
    capture_runtime_execution,
    initial_manifest,
    write_manifest,
)


class _State:
    def __init__(self, value: str):
        self.value = value

    def state_dict(self):
        return {"value": self.value}


class _Model(_State):
    def named_parameters(self):
        return [("log_std", _State("std"))]


def test_runtime_snapshot_requires_real_components_and_records_fresh_state(tmp_path: Path):
    agent = SimpleNamespace(
        models={"policy": _Model("policy"), "value": _State("value"), "discriminator": _State("amp")},
        optimizer=_State("optimizer"),
        state_preprocessor=_State("normalizer"),
    )
    env = SimpleNamespace(_phase=0, _dr_level=0)
    snapshot = capture_optimization_state(
        agent,
        env,
        run_dir=tmp_path,
        amp_expected=True,
        parity_check=True,
        resume=False,
    )

    assert snapshot["status"] == "proven"
    assert snapshot["parity_check"] is True
    assert snapshot["fields"]["policy"]["status"] == "captured"
    assert snapshot["fields"]["curriculum"]["status"] == "fresh_initialization"


def test_runtime_import_proof_captures_resolved_file(tmp_path: Path):
    proof = capture_runtime_execution(
        Path(__file__).resolve().parents[4],
        ("products.taili.blind_locomotion.runtime_manifest",),
    )
    assert proof["status"] == "proven"
    assert proof["remote_verification"] == "proven"
    assert any(item["path"].endswith("runtime_manifest.py") for item in proof["source_files"])


def test_initial_manifest_is_written_as_a_declared_not_proven_record(tmp_path: Path):
    config = tmp_path / "config.yaml"
    effective = tmp_path / "effective.yaml"
    agent = tmp_path / "agent.yaml"
    for path in (config, effective, agent):
        path.write_text("key: value\n", encoding="utf-8")
    manifest = initial_manifest(
        run_id="fixture",
        run_dir=tmp_path,
        task="taili",
        payload_root=tmp_path,
        source_config=config,
        effective_config=effective,
        agent_config=agent,
    )
    path = tmp_path / "runtime_manifest.json"
    write_manifest(path, manifest)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["status"] == "running"
    assert loaded["runtime_execution"]["status"] == "declared"
    assert loaded["optimization_state"]["parity_check"] is False


def test_manifest_updates_preserve_nested_launch_facts(tmp_path: Path):
    config = tmp_path / "config.yaml"
    effective = tmp_path / "effective.yaml"
    agent = tmp_path / "agent.yaml"
    for path in (config, effective, agent):
        path.write_text("key: value\n", encoding="utf-8")
    path = tmp_path / "runtime_manifest.json"
    write_manifest(
        path,
        initial_manifest(
            run_id="run-preserved",
            run_dir=tmp_path,
            task="taili",
            payload_root=tmp_path,
            source_config=config,
            effective_config=effective,
            agent_config=agent,
        ),
    )

    from products.taili.blind_locomotion.runtime_manifest import update_manifest

    update_manifest(path, run={"seed": 42, "skrl_version": "1.4.3"})
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["run"]["run_id"] == "run-preserved"
    assert loaded["run"]["task"] == "taili"
    assert loaded["run"]["seed"] == 42
