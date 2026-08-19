from __future__ import annotations

import json
from pathlib import Path
import shlex
import shutil
import tarfile

from autotuner.execution.deployment import DeploymentSpec, RemoteLayout, VersionedRemoteDeployer
from autotuner.execution.payload import build_payload_manifest


class _FakeRemote:
    """Small command-aware transport; no shell or SSH is used by this test."""

    def __init__(self, root: Path):
        self.root = root

    def _path(self, value: str) -> Path:
        return self.root / value.lstrip("/")

    def put(self, local: str, remote: str) -> None:
        target = self._path(remote)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, target)

    def exec(self, cmd: str, timeout: int = 30) -> tuple[str, int]:
        tokens = shlex.split(cmd)
        if not tokens:
            return "", 0
        if tokens[0] == "mkdir" and tokens[1] == "-p":
            for value in tokens[2:]:
                self._path(value).mkdir(parents=True, exist_ok=True)
            return "", 0
        if tokens[0] == "test" and tokens[1] == "-f":
            return "", 0 if self._path(tokens[2]).is_file() else 1
        if tokens[0] == "sha256sum":
            path = self._path(tokens[1])
            if not path.is_file():
                return "", 1
            import hashlib

            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            return f"{digest}  {tokens[1]}\n", 0
        if tokens[0] == "cat":
            path = self._path(tokens[1])
            return (path.read_text(encoding="utf-8"), 0) if path.is_file() else ("", 1)
        if tokens[0] == "rm":
            values = [item for item in tokens[1:] if item != "--"]
            for value in values:
                target = self._path(value)
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
            return "", 0
        if tokens[0] == "mv":
            values = tokens[1:]
            if values and values[0] in {"-f", "-T"}:
                values = values[1:]
            source, target = self._path(values[-2]), self._path(values[-1])
            if not source.exists():
                return "", 1
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.is_dir():
                return "", 1
            shutil.move(str(source), str(target))
            return "", 0
        if tokens[0] == "tar" and tokens[1] == "-xzf":
            archive, destination = self._path(tokens[2]), self._path(tokens[4])
            destination.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive, "r:gz") as handle:
                handle.extractall(destination)
            return "", 0
        raise AssertionError(f"unsupported fake command: {cmd}")


def _payload(tmp_path: Path) -> tuple[Path, Path, str]:
    staged = tmp_path / "staged"
    (staged / "pkg").mkdir(parents=True)
    (staged / "pkg" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    contract = {
        "product_id": "alpha",
        "product_version": "1",
        "contract_digest": "contract",
        "source_digest": "source",
        "runtime": {"digest": "runtime-1"},
    }
    manifest = build_payload_manifest(staged, contract=contract)
    manifest_path = tmp_path / "payload_manifest.json"
    manifest.write(staged / "payload_manifest.json")
    manifest.write(manifest_path)
    archive = tmp_path / "payload.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for item in sorted(staged.rglob("*")):
            if item.is_file():
                handle.add(item, arcname=item.relative_to(staged))
    return archive, manifest_path, manifest.payload_digest


def test_remote_deployer_stages_immutable_artifacts_and_rolls_back(tmp_path: Path) -> None:
    archive, manifest, payload_digest = _payload(tmp_path)
    remote = _FakeRemote(tmp_path / "remote")
    deployer = VersionedRemoteDeployer(remote, layout=RemoteLayout("/data/rl-agent"))

    runtime = {"backend": "isaaclab", "runtime_id": "runtime-1", "digest": "runtime-1"}
    runtime_result = deployer.stage_runtime(runtime, runtime_digest="runtime-1")
    payload_result = deployer.stage_payload(archive, manifest)
    assert runtime_result.status == "activated"
    assert payload_result.identity == payload_digest
    assert deployer.stage_payload(archive, manifest).status == "reused"

    run_one = deployer.activate_run(
        "run-1",
        {"run_id": "run-1"},
        runtime_digest="runtime-1",
        payload_digest=payload_digest,
    )
    deployer.activate_run(
        "run-2",
        {"run_id": "run-2"},
        runtime_digest="runtime-1",
        payload_digest=payload_digest,
    )
    assert run_one.status == "activated"
    active = json.loads((tmp_path / "remote" / "data" / "rl-agent" / "active.json").read_text(encoding="utf-8"))
    assert active["run_id"] == "run-2"
    # Reusing an immutable run must still activate its pointer.
    reused = deployer.activate_run(
        "run-1",
        {"run_id": "run-1"},
        runtime_digest="runtime-1",
        payload_digest=payload_digest,
    )
    assert reused.status == "reused"
    active = json.loads((tmp_path / "remote" / "data" / "rl-agent" / "active.json").read_text(encoding="utf-8"))
    assert active["run_id"] == "run-1"
    assert active["previous"]["run_id"] == "run-2"
    assert "previous" not in active["previous"]
    deployer.rollback_active("run-1")
    active = json.loads((tmp_path / "remote" / "data" / "rl-agent" / "active.json").read_text(encoding="utf-8"))
    assert active["run_id"] == "run-1"


def test_deployment_spec_is_product_neutral() -> None:
    spec = DeploymentSpec(
        runtime={"digest": "runtime-1"},
        runtime_digest="runtime-1",
        payload_archive="payload.tar.gz",
        payload_manifest="payload.json",
        run_id="run-1",
        run_manifest={"run_id": "run-1"},
    )
    spec.validate()


def test_run_activation_rejects_payload_bound_to_another_runtime(tmp_path: Path) -> None:
    archive, manifest, payload_digest = _payload(tmp_path)
    remote = _FakeRemote(tmp_path / "remote")
    deployer = VersionedRemoteDeployer(remote, layout=RemoteLayout("/data/rl-agent"))
    deployer.stage_runtime({"backend": "isaaclab", "runtime_id": "runtime-1", "digest": "runtime-1"}, runtime_digest="runtime-1")
    deployer.stage_runtime({"backend": "isaaclab", "runtime_id": "runtime-2", "digest": "runtime-2"}, runtime_digest="runtime-2")
    deployer.stage_payload(archive, manifest)
    try:
        deployer.activate_run(
            "mismatch",
            {"run_id": "mismatch"},
            runtime_digest="runtime-2",
            payload_digest=payload_digest,
        )
    except RuntimeError as exc:
        assert "payload/runtime identity mismatch" in str(exc)
    else:
        raise AssertionError("activation unexpectedly accepted a mismatched runtime")
