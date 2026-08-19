from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from autotuner.execution.changeset import Change, ChangeSet, ChangeSetError
from autotuner.execution.compatibility import check_resume_compatibility
from autotuner.execution.payload import (
    build_payload_manifest,
    load_payload_manifest,
    verify_payload,
    verify_payload_archive,
)
from autotuner.execution.runtime import resolve_runtime_identity
from autotuner.execution.runtime import capture_runtime_identity


def test_runtime_identity_hashes_declared_lockfile(tmp_path: Path) -> None:
    lock = tmp_path / "runtime.lock"
    lock.write_text("isaaclab==1\n", encoding="utf-8")
    first = resolve_runtime_identity(
        {"backend": "isaaclab", "runtime_id": "test", "lockfiles": ["runtime.lock"]},
        root=tmp_path,
    )
    lock.write_text("isaaclab==2\n", encoding="utf-8")
    second = resolve_runtime_identity(
        {"backend": "isaaclab", "runtime_id": "test", "lockfiles": ["runtime.lock"]},
        root=tmp_path,
    )

    assert first.status == second.status == "declared"
    assert first.digest != second.digest
    assert first.lockfiles[0]["sha256"] != second.lockfiles[0]["sha256"]


def test_runtime_identity_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "runtime-outside.lock"
    outside.write_text("outside\n", encoding="utf-8")
    link = tmp_path / "runtime.lock"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available on this Windows runner")

    identity = resolve_runtime_identity({"lockfiles": ["runtime.lock"]}, root=tmp_path)
    assert identity.status == "invalid"
    assert any("escapes workspace" in item for item in identity.issues)


def test_runtime_evidence_does_not_promote_observation_to_match(monkeypatch) -> None:
    observed = capture_runtime_identity({"backend": "isaaclab", "runtime_id": "runtime", "digest": "declared"})
    assert observed["status"] == "observed"
    monkeypatch.setenv("RL_AGENT_RUNTIME_DIGEST", "declared")
    matched = capture_runtime_identity({"backend": "isaaclab", "runtime_id": "runtime", "digest": "declared"})
    assert matched["status"] == "matched"


def test_changeset_validates_all_old_values_before_writing(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    source = tmp_path / "module.py"
    config.write_text("gain: 1\nmode: nominal\n", encoding="utf-8")
    source.write_text("VALUE = 1\n", encoding="utf-8")
    changes = ChangeSet(
        id="change-1",
        base_contract_digest="contract",
        base_source_digest="source",
        changes=(
            Change("cfg", "config_path", "config.yaml", "adjust config", key="gain", expected_value=1, new_value=2),
            Change("src", "source_patch", "module.py", "adjust source", old_text="VALUE = 1", new_text="VALUE = 2"),
        ),
    )
    result = changes.apply(
        tmp_path,
        current_contract_digest="contract",
        current_source_digest="source",
        journal_root=tmp_path / "journals",
    )

    assert "gain: 2" in config.read_text(encoding="utf-8")
    assert source.read_text(encoding="utf-8") == "VALUE = 2\n"
    journal = json.loads(Path(result.journal_path).read_text(encoding="utf-8"))
    assert journal["state"] == "committed"
    ChangeSet.rollback(result.journal_path)
    assert config.read_text(encoding="utf-8") == "gain: 1\nmode: nominal\n"
    assert source.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_changeset_mismatch_is_atomic(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    changes = ChangeSet(
        id="change-mismatch",
        base_contract_digest="contract",
        base_source_digest="source",
        changes=(Change("src", "source_patch", "module.py", "stale", old_text="VALUE = 0", new_text="VALUE = 2"),),
    )
    with pytest.raises(ChangeSetError):
        changes.apply(tmp_path, current_contract_digest="contract", current_source_digest="source")
    assert target.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_changeset_refuses_an_existing_journal_directory(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    journal_dir = tmp_path / "journals" / "already-used"
    journal_dir.mkdir(parents=True)
    marker = journal_dir / "keep.txt"
    marker.write_text("do not delete\n", encoding="utf-8")
    changes = ChangeSet(
        id="already-used",
        base_contract_digest="",
        base_source_digest="",
        changes=(Change("src", "source_patch", "module.py", "update", old_text="VALUE = 1", new_text="VALUE = 2"),),
    )
    with pytest.raises(ChangeSetError, match="journal directory already exists"):
        changes.apply(tmp_path, journal_root=tmp_path / "journals")
    assert marker.read_text(encoding="utf-8") == "do not delete\n"


def test_changeset_can_guard_an_explicit_yaml_null(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("value: null\n", encoding="utf-8")
    changes = ChangeSet.from_mapping(
        {
            "id": "null-guard",
            "changes": [
                {
                    "id": "value",
                    "kind": "config_path",
                    "path": "config.yaml",
                    "reason": "replace an explicitly empty value",
                    "key": "value",
                    "expected_value": None,
                    "new_value": 1,
                }
            ],
        }
    )
    changes.apply(tmp_path)
    assert "value: 1" in config.read_text(encoding="utf-8")


def test_prepared_changeset_can_be_recovered_without_partial_rollback(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    changes = ChangeSet(
        id="recover-me",
        base_contract_digest="",
        base_source_digest="",
        changes=(Change("src", "source_patch", "module.py", "update", old_text="VALUE = 1", new_text="VALUE = 2"),),
    )
    result = changes.apply(tmp_path, journal_root=tmp_path / "journals")
    journal_path = Path(result.journal_path)
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["state"] = "prepared"
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    ChangeSet.recover_prepared(journal_path)
    assert target.read_text(encoding="utf-8") == "VALUE = 1\n"
    assert json.loads(journal_path.read_text(encoding="utf-8"))["state"] == "recovered"


def test_rollback_preflights_all_targets(tmp_path: Path) -> None:
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("A = 1\n", encoding="utf-8")
    second.write_text("B = 1\n", encoding="utf-8")
    changes = ChangeSet(
        id="rollback-preflight",
        base_contract_digest="",
        base_source_digest="",
        changes=(
            Change("first", "source_patch", "first.py", "update", old_text="A = 1", new_text="A = 2"),
            Change("second", "source_patch", "second.py", "update", old_text="B = 1", new_text="B = 2"),
        ),
    )
    result = changes.apply(tmp_path, journal_root=tmp_path / "journals")
    second.write_text("B = external\n", encoding="utf-8")
    with pytest.raises(ChangeSetError):
        ChangeSet.rollback(result.journal_path)
    assert first.read_text(encoding="utf-8") == "A = 2\n"


def test_runtime_replace_requires_and_checks_the_old_file_hash(tmp_path: Path) -> None:
    target = tmp_path / "runtime.cfg"
    target.write_text("old runtime\n", encoding="utf-8")
    changes = ChangeSet(
        id="runtime-replace",
        base_contract_digest="",
        base_source_digest="",
        changes=(
            Change(
                "runtime",
                "runtime_replace",
                "runtime.cfg",
                "replace pinned runtime declaration",
                expected_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                new_text="new runtime\n",
            ),
        ),
    )
    changes.apply(tmp_path)
    assert target.read_text(encoding="utf-8") == "new runtime\n"


def _manifest(*, runtime: str = "runtime", config: str = "config") -> dict:
    return {
        "product": {"id": "taili", "version": "1"},
        "run": {"task": "blind"},
        "resolved_contract": {
            "product_id": "taili",
            "product_version": "1",
            "config_digest": config,
            "source_digest": "source",
            "runtime": {"digest": runtime},
            "compatibility": {
                "observation_structure": "obs.v1",
                "action_structure": "act.v1",
                "network_structure": "net.v1",
                "normalization": "norm.v1",
                "physics_timestep": 0.0025,
            },
            "training": {"task_id": "blind"},
        },
        "payload": {"digest": "payload"},
    }


def test_resume_compatibility_requires_identity_and_checkpoint_components() -> None:
    checkpoint = {
        "status": "captured",
        "keys": [],
        "key_paths": ["models.policy", "models.value", "optimizers.policy", "state_preprocessor"],
    }
    decision = check_resume_compatibility(_manifest(), _manifest(), checkpoint=checkpoint)
    assert decision.compatible is True
    assert decision.decision == "resume"

    changed = check_resume_compatibility(_manifest(), _manifest(runtime="other"), checkpoint=checkpoint)
    assert changed.compatible is False
    assert any("runtime_digest" in reason for reason in changed.reasons)


def test_payload_manifest_is_content_addressed(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    contract = {
        "product_id": "alpha",
        "product_version": "1",
        "contract_digest": "contract",
        "source_digest": "source",
        "runtime": {"digest": "runtime"},
    }
    manifest = build_payload_manifest(tmp_path, contract=contract)
    path = manifest.write(tmp_path / "payload_manifest.json")
    loaded = load_payload_manifest(path)
    assert loaded.payload_digest == manifest.payload_digest
    assert verify_payload(tmp_path, loaded) == []
    (tmp_path / "pkg" / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert verify_payload(tmp_path, loaded)


def test_payload_archive_is_verified_before_transport(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    manifest = build_payload_manifest(
        staged,
        contract={"product_id": "a", "product_version": "1", "contract_digest": "c", "source_digest": "s"},
    )
    manifest.write(staged / "payload_manifest.json")
    archive = tmp_path / "payload.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for item in sorted(staged.iterdir()):
            handle.add(item, arcname=item.name)
    assert verify_payload_archive(archive, manifest) == []
    (staged / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    tampered = tmp_path / "tampered.tar.gz"
    with tarfile.open(tampered, "w:gz") as handle:
        for item in sorted(staged.iterdir()):
            handle.add(item, arcname=item.name)
    assert any("hash mismatch" in item for item in verify_payload_archive(tampered, manifest))


def test_payload_archive_checks_embedded_manifest_identity(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    manifest = build_payload_manifest(
        staged,
        contract={"product_id": "a", "product_version": "1", "contract_digest": "c", "source_digest": "s"},
    )
    manifest.write(staged / "payload_manifest.json")
    embedded = json.loads((staged / "payload_manifest.json").read_text(encoding="utf-8"))
    embedded["product_version"] = "tampered"
    (staged / "payload_manifest.json").write_text(json.dumps(embedded), encoding="utf-8")
    archive = tmp_path / "payload.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for item in sorted(staged.iterdir()):
            handle.add(item, arcname=item.name)
    assert any("embedded payload manifest" in item for item in verify_payload_archive(archive, manifest))
