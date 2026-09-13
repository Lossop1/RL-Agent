"""Content-addressed remote staging and atomic run activation.

This module owns transport orchestration only.  It does not know how an
IsaacLab task is trained or diagnosed; the product contract supplies the
payload/runtime references and the caller supplies the command.  Remote
artifacts are immutable once activated.  Rollback changes the active run
pointer and leaves every prior artifact available for audit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
from datetime import datetime, timezone
import json
import os
import re
import shlex
import secrets
import tempfile
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from .hashing import sha256_file
from .payload import PayloadManifest, load_payload_manifest, verify_payload_archive


DEPLOYMENT_SCHEMA = "rl-agent.remote-deployment/v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]*$")


@runtime_checkable
class RemoteTransport(Protocol):
    """Minimal transport used by the deployer and easy to fake in tests."""

    def exec(self, cmd: str, timeout: int = 30) -> tuple[str, int]:
        ...

    def put(self, local: str, remote: str) -> None:
        ...

    def get(self, remote: str, local: str) -> None:
        """Download file from remote to local."""
        ...


@dataclass(frozen=True)
class RemoteLayout:
    root: str = "/root/gpufree-data/rl-agent"

    def __post_init__(self) -> None:
        # Remote paths are shell-quoted later, but an absolute POSIX root keeps
        # the artifact layout unambiguous across SSH implementations.
        root_path = Path(self.root)
        if not self.root or not self.root.startswith("/") or ".." in root_path.parts:
            raise ValueError("remote artifact root must be an absolute POSIX path")

    @property
    def runtimes(self) -> str:
        return f"{self.root.rstrip('/')}/runtimes"

    @property
    def payloads(self) -> str:
        return f"{self.root.rstrip('/')}/payloads"

    @property
    def runs(self) -> str:
        return f"{self.root.rstrip('/')}/runs"

    @property
    def active(self) -> str:
        return f"{self.root.rstrip('/')}/active.json"


@dataclass(frozen=True)
class RemoteArtifact:
    kind: str
    identity: str
    path: str
    status: str
    sha256: str = ""
    detail: str = ""


@dataclass(frozen=True)
class DeploymentResult:
    status: str
    runtime: RemoteArtifact
    payload: RemoteArtifact
    run: RemoteArtifact
    trace: tuple[Mapping[str, Any], ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DEPLOYMENT_SCHEMA,
            "status": self.status,
            "runtime": asdict(self.runtime),
            "payload": asdict(self.payload),
            "run": asdict(self.run),
            "trace": [dict(item) for item in self.trace],
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class DeploymentSpec:
    """Product-neutral handoff from a resolved contract to the deployer.

    The product adapter owns how these paths and manifests are produced.  The
    execution layer only checks that the handoff is complete and immutable.
    """

    runtime: Mapping[str, Any]
    runtime_digest: str
    payload_archive: str | Path
    payload_manifest: str | Path | PayloadManifest
    run_id: str
    run_manifest: Mapping[str, Any] | str | Path
    payload_digest: str = ""
    task_contract_ref: str = ""
    task_bundle_digest: str = ""
    task_artifact_manifest_ref: str = ""

    def validate(self) -> None:
        _id(self.runtime_digest, "runtime_digest")
        _id(self.run_id, "run_id")
        if self.payload_digest:
            _id(self.payload_digest, "payload_digest")
        if self.task_contract_ref:
            _id(self.task_contract_ref, "task_contract_ref")
        if self.task_bundle_digest:
            _id(self.task_bundle_digest, "task_bundle_digest")
        if self.task_artifact_manifest_ref and "\x00" in self.task_artifact_manifest_ref:
            raise ValueError("task_artifact_manifest_ref contains a control character")
        if not isinstance(self.runtime, Mapping) or not self.runtime:
            raise ValueError("deployment runtime identity must be a non-empty mapping")


def _id(value: str, name: str) -> str:
    value = str(value or "").strip()
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(f"unsafe {name}: {value!r}")
    return value


def _exec(remote: RemoteTransport, command: str, *, timeout: int = 60) -> tuple[str, int]:
    result = remote.exec(command, timeout=timeout)
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("remote transport must return (output, return_code)")
    output, code = result
    return str(output), int(code)


def _remote_sha(remote: RemoteTransport, path: str) -> str:
    output, code = _exec(remote, f"sha256sum {shlex.quote(path)}", timeout=30)
    if code != 0:
        return ""
    token = output.strip().split()[0] if output.strip() else ""
    return token.lower() if re.fullmatch(r"[0-9a-fA-F]{64}", token) else ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _temporary_json(prefix: str, value: Mapping[str, Any]) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".json", delete=False, mode="w", encoding="utf-8")
    path = Path(handle.name)
    try:
        handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        handle.close()
    return path


class VersionedRemoteDeployer:
    """Deploy immutable runtime/payload artifacts and one independent run."""

    def __init__(self, remote: RemoteTransport, *, layout: RemoteLayout | None = None):
        self.remote = remote
        self.layout = layout or RemoteLayout()

    def deploy_spec(self, spec: DeploymentSpec) -> DeploymentResult:
        """Deploy one validated product-neutral specification."""
        spec.validate()
        return self.deploy(
            runtime=spec.runtime,
            runtime_digest=spec.runtime_digest,
            payload_archive=spec.payload_archive,
            payload_manifest=spec.payload_manifest,
            run_id=spec.run_id,
            run_manifest=spec.run_manifest,
            payload_digest=spec.payload_digest,
            task_contract_ref=spec.task_contract_ref,
            task_bundle_digest=spec.task_bundle_digest,
            task_artifact_manifest_ref=spec.task_artifact_manifest_ref,
        )

    def _mkdir(self, path: str) -> None:
        _, code = _exec(self.remote, f"mkdir -p {shlex.quote(path)}", timeout=30)
        if code != 0:
            raise RuntimeError(f"cannot create remote directory: {path}")

    def _token(self) -> str:
        return secrets.token_hex(8)

    def stage_runtime(self, runtime: Mapping[str, Any], *, runtime_digest: str) -> RemoteArtifact:
        identity = _id(runtime_digest, "runtime_digest")
        final = f"{self.layout.runtimes}/{identity}"
        declared_digest = str(runtime.get("digest") or "") if isinstance(runtime, Mapping) else ""
        if not declared_digest:
            raise ValueError("runtime descriptor must contain digest")
        if declared_digest != identity:
            raise ValueError("runtime descriptor digest does not match runtime_digest")
        local_descriptor = _temporary_json(f"rl-runtime-{identity[:12]}-", runtime)
        try:
            # The descriptor itself is checked independently of the content
            # address.  This catches a stale or hand-edited runtime record.
            expected_sha = sha256_file(local_descriptor)
            self._mkdir(self.layout.runtimes)
            if self._remote_file_exists(f"{final}/runtime_identity.json"):
                existing_sha = _remote_sha(self.remote, f"{final}/runtime_identity.json")
                if existing_sha != expected_sha:
                    raise RuntimeError(f"immutable runtime identity conflict: {identity}")
                return RemoteArtifact("runtime", identity, final, "reused", existing_sha)
            stage = f"{self.layout.runtimes}/.staging-{identity[:12]}-{self._token()}"
            try:
                self._mkdir(stage)
                remote_descriptor = f"{stage}/runtime_identity.json"
                self.remote.put(str(local_descriptor), remote_descriptor)
                if _remote_sha(self.remote, remote_descriptor) != expected_sha:
                    raise RuntimeError(f"runtime descriptor checksum mismatch: {identity}")
                _, code = _exec(self.remote, f"mv -T {shlex.quote(stage)} {shlex.quote(final)}", timeout=30)
                if code != 0:
                    # A concurrent deploy may have won the same immutable
                    # identity. Reuse it only after checking its checksum.
                    if self._remote_file_exists(f"{final}/runtime_identity.json") and _remote_sha(
                        self.remote, f"{final}/runtime_identity.json"
                    ) == expected_sha:
                        return RemoteArtifact("runtime", identity, final, "reused", expected_sha)
                    raise RuntimeError(f"runtime activation failed: {identity}")
                return RemoteArtifact("runtime", identity, final, "activated", expected_sha)
            except Exception:
                self._cleanup_remote(stage)
                raise
        finally:
            try:
                local_descriptor.unlink()
            except OSError:
                pass

    def stage_payload(
        self,
        archive: str | Path,
        manifest: str | Path | PayloadManifest,
        *,
        payload_digest: str = "",
    ) -> RemoteArtifact:
        archive_path = Path(archive).resolve()
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        temporary_manifest: Path | None = None
        if isinstance(manifest, PayloadManifest):
            payload_manifest = manifest
            fd, temporary_name = tempfile.mkstemp(prefix="rl-payload-manifest-", suffix=".json")
            os.close(fd)
            temporary_manifest = Path(temporary_name)
            try:
                payload_manifest.write(temporary_manifest)
                expected_manifest_sha = sha256_file(temporary_manifest)
            except Exception:
                temporary_manifest.unlink(missing_ok=True)
                raise
        else:
            payload_manifest = load_payload_manifest(manifest)
            expected_manifest_sha = sha256_file(manifest)
        payload_manifest.validate(require_digest=True)
        archive_errors = verify_payload_archive(archive_path, payload_manifest)
        if archive_errors:
            if temporary_manifest is not None:
                temporary_manifest.unlink(missing_ok=True)
            raise ValueError("payload archive failed local verification:\n" + "\n".join(archive_errors))
        identity = _id(payload_digest or payload_manifest.payload_digest, "payload_digest")
        if identity != payload_manifest.payload_digest:
            if temporary_manifest is not None:
                temporary_manifest.unlink(missing_ok=True)
            raise ValueError("payload digest does not match payload manifest")
        final = f"{self.layout.payloads}/{identity}"
        stage = ""
        try:
            self._mkdir(self.layout.payloads)
            if self._remote_file_exists(f"{final}/payload_manifest.json"):
                existing_sha = _remote_sha(self.remote, f"{final}/payload_manifest.json")
                if expected_manifest_sha and existing_sha != expected_manifest_sha:
                    raise RuntimeError(f"immutable payload identity conflict: {identity}")
                return RemoteArtifact("payload", identity, final, "reused", existing_sha)

            stage = f"{self.layout.payloads}/.staging-{identity[:12]}-{self._token()}"
            self._mkdir(stage)
            remote_archive = f"{stage}/payload.tar.gz"
            remote_manifest = f"{stage}/payload_manifest.json"
            self.remote.put(str(archive_path), remote_archive)
            if _remote_sha(self.remote, remote_archive) != sha256_file(archive_path):
                raise RuntimeError(f"payload archive checksum mismatch: {identity}")
            manifest_source = temporary_manifest or Path(manifest).resolve()
            self.remote.put(str(manifest_source), remote_manifest)
            if expected_manifest_sha and _remote_sha(self.remote, remote_manifest) != expected_manifest_sha:
                raise RuntimeError(f"payload manifest checksum mismatch: {identity}")
            unpacked = f"{stage}/unpacked"
            self._mkdir(unpacked)
            _, code = _exec(
                self.remote,
                f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(unpacked)}",
                timeout=180,
            )
            if code != 0:
                raise RuntimeError(f"payload archive extraction failed: {identity}")
            # Payload archives have one product-neutral root manifest.  The
            # execution layer must not infer a package name (or know Taili).
            embedded = f"{unpacked}/payload_manifest.json"
            if not self._remote_file_exists(embedded):
                embedded = ""
            if not embedded:
                raise RuntimeError(f"payload archive has no embedded manifest: {identity}")
            if expected_manifest_sha and _remote_sha(self.remote, embedded) != expected_manifest_sha:
                raise RuntimeError(f"embedded payload manifest checksum mismatch: {identity}")
            _, code = _exec(self.remote, f"mv -T {shlex.quote(unpacked)} {shlex.quote(final)}", timeout=60)
            if code != 0:
                if self._remote_file_exists(f"{final}/payload_manifest.json") and _remote_sha(
                    self.remote, f"{final}/payload_manifest.json"
                ) == expected_manifest_sha:
                    self._cleanup_remote(stage)
                    return RemoteArtifact("payload", identity, final, "reused", expected_manifest_sha)
                raise RuntimeError(f"payload activation failed: {identity}")
            # The extracted directory is now immutable at ``final``. Remove
            # only the staging siblings (archive and upload copy), retaining
            # the activated payload and its embedded manifest.
            self._cleanup_remote(stage)
            return RemoteArtifact("payload", identity, final, "activated", sha256_file(archive_path))
        except Exception:
            if stage:
                self._cleanup_remote(stage)
            raise
        finally:
            if temporary_manifest is not None:
                temporary_manifest.unlink(missing_ok=True)

    def activate_run(
        self,
        run_id: str,
        run_manifest: Mapping[str, Any] | str | Path,
        *,
        runtime_digest: str,
        payload_digest: str,
        task_contract_ref: str = "",
        task_bundle_digest: str = "",
        task_artifact_manifest_ref: str = "",
    ) -> RemoteArtifact:
        run_identity = _id(run_id, "run_id")
        runtime_identity = _id(runtime_digest, "runtime_digest")
        payload_identity = _id(payload_digest, "payload_digest")
        self._assert_runtime_artifact(runtime_identity)
        payload_manifest = self._assert_payload_artifact(payload_identity)
        bound_runtime = str(payload_manifest.get("runtime_digest") or "")
        if bound_runtime and bound_runtime != runtime_identity:
            raise RuntimeError(
                f"payload/runtime identity mismatch: payload={bound_runtime}, runtime={runtime_identity}"
            )
        if isinstance(run_manifest, Mapping):
            data = copy.deepcopy(dict(run_manifest))
        else:
            data = json.loads(Path(run_manifest).resolve().read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("run manifest must be an object")
        execution = data.setdefault("execution", {})
        if not isinstance(execution, dict):
            raise ValueError("run manifest execution section must be an object")
        execution.update(
            {
                "runtime_digest": runtime_identity,
                "payload_digest": payload_identity,
                "run_id": run_identity,
            }
        )
        if task_contract_ref:
            execution["task_contract_ref"] = task_contract_ref
        if task_bundle_digest:
            execution["task_bundle_digest"] = task_bundle_digest
        if task_artifact_manifest_ref:
            execution["task_artifact_manifest_ref"] = task_artifact_manifest_ref
        local_path = _temporary_json(f"rl-run-{run_identity}-", data)
        try:
            local_path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            expected_sha = sha256_file(local_path)
            final = f"{self.layout.runs}/{run_identity}"
            self._mkdir(self.layout.runs)
            if self._remote_file_exists(f"{final}/run.json"):
                existing_sha = _remote_sha(self.remote, f"{final}/run.json")
                if existing_sha != expected_sha:
                    raise RuntimeError(f"immutable run identity conflict: {run_identity}")
                # Reusing an immutable run still means making it the active
                # run.  The old implementation returned early and silently
                # left a different run active.
                self._atomic_active_pointer(run_identity, runtime_identity, payload_identity, event="activate-reused")
                return RemoteArtifact("run", run_identity, final, "reused", existing_sha)
            stage = f"{self.layout.runs}/.staging-{run_identity}-{self._token()}"
            try:
                self._mkdir(stage)
                remote_run = f"{stage}/run.json"
                self.remote.put(str(local_path), remote_run)
                if _remote_sha(self.remote, remote_run) != expected_sha:
                    raise RuntimeError(f"run manifest checksum mismatch: {run_identity}")
                _, code = _exec(self.remote, f"mv -T {shlex.quote(stage)} {shlex.quote(final)}", timeout=30)
                if code != 0:
                    if self._remote_file_exists(f"{final}/run.json") and _remote_sha(
                        self.remote, f"{final}/run.json"
                    ) == expected_sha:
                        self._atomic_active_pointer(run_identity, runtime_identity, payload_identity, event="activate-raced")
                        return RemoteArtifact("run", run_identity, final, "reused", expected_sha)
                    raise RuntimeError(f"run activation failed: {run_identity}")
                self._atomic_active_pointer(run_identity, runtime_identity, payload_identity, event="activate")
                return RemoteArtifact("run", run_identity, final, "activated", expected_sha)
            except Exception:
                self._cleanup_remote(stage)
                raise
        finally:
            try:
                local_path.unlink()
            except OSError:
                pass

    def deploy(
        self,
        *,
        runtime: Mapping[str, Any],
        runtime_digest: str,
        payload_archive: str | Path,
        payload_manifest: str | Path | PayloadManifest,
        run_id: str,
        run_manifest: Mapping[str, Any] | str | Path,
        payload_digest: str = "",
        task_contract_ref: str = "",
        task_bundle_digest: str = "",
        task_artifact_manifest_ref: str = "",
    ) -> DeploymentResult:
        trace: list[Mapping[str, Any]] = []
        try:
            runtime_result = self.stage_runtime(runtime, runtime_digest=runtime_digest)
            trace.append({"event": "runtime", **asdict(runtime_result)})
            payload_result = self.stage_payload(payload_archive, payload_manifest, payload_digest=payload_digest)
            trace.append({"event": "payload", **asdict(payload_result)})
            payload_identity = payload_result.identity
            run_result = self.activate_run(
                run_id,
                run_manifest,
                runtime_digest=runtime_result.identity,
                payload_digest=payload_identity,
                task_contract_ref=task_contract_ref,
                task_bundle_digest=task_bundle_digest,
                task_artifact_manifest_ref=task_artifact_manifest_ref,
            )
            trace.append({"event": "run", **asdict(run_result)})
            return DeploymentResult("activated", runtime_result, payload_result, run_result, tuple(trace))
        except Exception as exc:
            empty = RemoteArtifact("unknown", "", "", "failed", detail=str(exc))
            return DeploymentResult("failed", empty, empty, empty, tuple(trace), (f"{type(exc).__name__}: {exc}",))

    def _remote_file_exists(self, path: str) -> bool:
        _, code = _exec(self.remote, f"test -f {shlex.quote(path)}", timeout=30)
        return code == 0

    def _remote_json(self, path: str) -> Mapping[str, Any]:
        output, code = _exec(self.remote, f"cat {shlex.quote(path)}", timeout=30)
        if code != 0:
            raise RuntimeError(f"cannot read remote JSON: {path}")
        try:
            value = json.loads(output)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid remote JSON: {path}") from exc
        if not isinstance(value, Mapping):
            raise RuntimeError(f"remote JSON must be an object: {path}")
        return value

    def _assert_runtime_artifact(self, identity: str) -> None:
        path = f"{self.layout.runtimes}/{identity}/runtime_identity.json"
        if not self._remote_file_exists(path):
            raise ValueError(f"runtime artifact is not staged: {identity}")
        descriptor = self._remote_json(path)
        if str(descriptor.get("digest") or "") != identity:
            raise RuntimeError(f"runtime artifact digest mismatch: {identity}")

    def _assert_payload_artifact(self, identity: str) -> Mapping[str, Any]:
        path = f"{self.layout.payloads}/{identity}/payload_manifest.json"
        if not self._remote_file_exists(path):
            raise ValueError(f"payload artifact is not staged: {identity}")
        manifest = self._remote_json(path)
        if str(manifest.get("payload_digest") or "") != identity:
            raise RuntimeError(f"payload artifact digest mismatch: {identity}")
        return manifest

    def _cleanup_remote(self, path: str) -> None:
        if not path:
            return
        # Cleanup is best effort.  The immutable final directory is never
        # passed here; only a uniquely named staging path is eligible.
        try:
            _exec(self.remote, f"rm -rf -- {shlex.quote(path)}", timeout=30)
        except Exception:
            pass

    def read_active(self) -> Mapping[str, Any] | None:
        """Read the current pointer without importing a remote runtime."""
        output, code = _exec(self.remote, f"cat {shlex.quote(self.layout.active)}", timeout=30)
        if code != 0 or not output.strip():
            return None
        try:
            value = json.loads(output)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("active run pointer is not valid JSON") from exc
        if not isinstance(value, Mapping):
            raise RuntimeError("active run pointer must be an object")
        return value

    def _atomic_active_pointer(
        self,
        run_id: str,
        runtime_digest: str,
        payload_digest: str,
        *,
        event: str,
    ) -> None:
        self._mkdir(self.layout.root)
        previous = self.read_active()
        previous_summary = None
        if previous:
            previous_summary = {
                key: previous.get(key)
                for key in ("run_id", "runtime_digest", "payload_digest", "event_id", "activated_at")
                if previous.get(key) not in (None, "")
            }
        activated_at = _utc_now()
        pointer = {
            "schema_version": DEPLOYMENT_SCHEMA,
            "run_id": run_id,
            "runtime_digest": runtime_digest,
            "payload_digest": payload_digest,
            "event": event,
            "event_id": f"{_utc_now()}-{self._token()}",
            "activated_at": activated_at,
            "previous": previous_summary,
        }
        local = _temporary_json(f"rl-active-{run_id}-", pointer)
        stage = f"{self.layout.root}/.active-{self._token()}.json"
        activated = False
        try:
            self.remote.put(str(local), stage)
            _, code = _exec(self.remote, f"mv -f {shlex.quote(stage)} {shlex.quote(self.layout.active)}", timeout=30)
            if code != 0:
                raise RuntimeError("active run pointer activation failed")
            activated = True
        finally:
            try:
                local.unlink()
            except OSError:
                pass
            if not activated:
                self._cleanup_remote(stage)

    def rollback_active(self, run_id: str) -> RemoteArtifact:
        identity = _id(run_id, "run_id")
        run_path = f"{self.layout.runs}/{identity}/run.json"
        if not self._remote_file_exists(run_path):
            raise ValueError(f"cannot roll back to unknown run: {identity}")
        output, code = _exec(self.remote, f"cat {shlex.quote(run_path)}", timeout=30)
        if code != 0:
            raise RuntimeError(f"cannot read run manifest for rollback: {identity}")
        try:
            data = json.loads(output)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid remote run manifest: {identity}") from exc
        execution = data.get("execution") if isinstance(data, Mapping) else {}
        execution = execution if isinstance(execution, Mapping) else {}
        if str(execution.get("run_id") or identity) != identity:
            raise RuntimeError(f"run manifest identity mismatch: {identity}")
        runtime_digest = _id(str(execution.get("runtime_digest") or ""), "runtime_digest")
        payload_digest = _id(str(execution.get("payload_digest") or ""), "payload_digest")
        self._assert_runtime_artifact(runtime_digest)
        payload_manifest = self._assert_payload_artifact(payload_digest)
        bound_runtime = str(payload_manifest.get("runtime_digest") or "")
        if bound_runtime and bound_runtime != runtime_digest:
            raise RuntimeError(
                f"rollback payload/runtime identity mismatch: payload={bound_runtime}, runtime={runtime_digest}"
            )
        self._atomic_active_pointer(
            identity,
            runtime_digest,
            payload_digest,
            event="rollback",
        )
        return RemoteArtifact(
            "run",
            identity,
            f"{self.layout.runs}/{identity}",
            "rolled_back",
            detail=f"runtime={runtime_digest}; payload={payload_digest}",
        )


__all__ = [
    "DEPLOYMENT_SCHEMA",
    "DeploymentSpec",
    "DeploymentResult",
    "RemoteArtifact",
    "RemoteLayout",
    "RemoteTransport",
    "VersionedRemoteDeployer",
]
