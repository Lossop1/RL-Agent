"""Runtime identity resolution and runtime-environment evidence.

The identity is a reference to an immutable execution environment, not a
second task configuration.  A product may declare the backend, package
versions, lock files, and image/environment name.  Resolution hashes that
declaration and the referenced lock files.  A running payload can then record
observed package versions and compare them with the declared identity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

try:
    from .hashing import canonical_json, sha256_bytes, sha256_file
except ImportError:  # payload copy: the helper is renamed beside this module
    from .execution_hashing import canonical_json, sha256_bytes, sha256_file  # type: ignore


RUNTIME_SCHEMA = "rl-agent.runtime-identity/v1"
_PACKAGE_NAMES = (
    "isaaclab",
    "isaaclab_rl",
    "isaaclab_tasks",
    "isaacsim",
    "skrl",
    "torch",
    "gymnasium",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _relative_file(root: Path, value: str) -> Path:
    raw = Path(value)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"runtime lockfile must stay under workspace: {value!r}")
    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"runtime lockfile escapes workspace: {value!r}") from exc
    return candidate


@dataclass(frozen=True)
class RuntimeIdentity:
    schema_version: str
    backend: str
    runtime_id: str
    declaration: Mapping[str, Any]
    lockfiles: tuple[Mapping[str, Any], ...] = ()
    issues: tuple[str, ...] = ()
    status: str = "declared"
    digest: str = field(default="")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["lockfiles"] = [dict(item) for item in self.lockfiles]
        return data

    def _unsigned(self) -> dict[str, Any]:
        data = self.to_dict()
        data.pop("digest", None)
        return data

    def compute_digest(self) -> str:
        return sha256_bytes(canonical_json(self._unsigned()).encode("utf-8"))

    def with_digest(self) -> "RuntimeIdentity":
        return RuntimeIdentity(**{**self.to_dict(), "lockfiles": self.lockfiles, "digest": self.compute_digest()})


def resolve_runtime_identity(
    declaration: Mapping[str, Any] | None = None,
    *,
    root: str | Path,
) -> RuntimeIdentity:
    """Resolve a product runtime declaration into a stable identity."""
    values = _mapping(declaration)
    backend = str(values.get("backend") or "isaaclab").strip()
    runtime_id = str(values.get("runtime_id") or f"{backend}.default").strip()
    issues: list[str] = []
    lock_records: list[Mapping[str, Any]] = []
    raw_lockfiles = values.get("lockfiles", values.get("lock_files", ()))
    if isinstance(raw_lockfiles, str):
        raw_lockfiles = (raw_lockfiles,)
    if raw_lockfiles is None:
        raw_lockfiles = ()
    if not isinstance(raw_lockfiles, (list, tuple)):
        issues.append("runtime.lockfiles must be a list")
        raw_lockfiles = ()
    workspace = Path(root).resolve()
    for item in raw_lockfiles:
        if isinstance(item, Mapping):
            declared_path = str(item.get("path") or "").strip()
            declared_sha = str(item.get("sha256") or "").strip().lower()
        else:
            declared_path = str(item).strip()
            declared_sha = ""
        if not declared_path:
            issues.append("runtime.lockfiles contains an empty path")
            continue
        try:
            path = _relative_file(workspace, declared_path)
        except ValueError as exc:
            issues.append(str(exc))
            continue
        exists = path.is_file()
        actual_sha = sha256_file(path) if exists else ""
        if not exists:
            issues.append(f"missing runtime lockfile: {declared_path}")
        if declared_sha and actual_sha and declared_sha != actual_sha:
            issues.append(f"runtime lockfile sha256 mismatch: {declared_path}")
        lock_records.append(
            {
                "path": declared_path,
                "exists": exists,
                "sha256": actual_sha,
                "declared_sha256": declared_sha,
            }
        )

    # Keep only declared runtime facts in the digest.  Host-local observations
    # belong to a run manifest and must not silently change a payload identity.
    declaration_data = {str(key): value for key, value in values.items() if key not in {"lockfiles", "lock_files", "digest"}}
    identity = RuntimeIdentity(
        schema_version=RUNTIME_SCHEMA,
        backend=backend,
        runtime_id=runtime_id,
        declaration=declaration_data,
        lockfiles=tuple(lock_records),
        issues=tuple(issues),
        status="declared" if not issues else "invalid",
    ).with_digest()
    expected = str(values.get("digest") or "").strip().lower()
    if expected and expected != identity.digest:
        identity = RuntimeIdentity(
            **{**identity.to_dict(), "status": "invalid", "issues": (*identity.issues, "declared runtime digest mismatch"), "digest": ""}
        ).with_digest()
    return identity


def _installed_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = ""
    return versions


def capture_runtime_identity(expected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Capture host/runtime evidence without importing IsaacLab or torch."""
    expected_data = _mapping(expected)
    declared = _mapping(expected_data.get("declaration"))
    if declared:
        declared = {**declared, **expected_data}
        expected_data = declared
    observed = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": _installed_versions(),
        "environment": {
            key: os.environ.get(key, "")
            for key in ("CONDA_DEFAULT_ENV", "ISAACLAB_VERSION", "ISAAC_SIM_VERSION", "PHYSX_VERSION")
            if os.environ.get(key)
        },
    }
    observed_digest = sha256_bytes(canonical_json(observed).encode("utf-8"))
    expected_digest = str(expected_data.get("digest") or "")
    declared_packages = _mapping(expected_data.get("packages"))
    package_checks = {
        name: {
            "declared": str(version),
            "observed": observed["packages"].get(name, ""),
            "equal": not version or str(version).startswith("declared-") or str(version) == observed["packages"].get(name, ""),
        }
        for name, version in declared_packages.items()
    }
    explicit_runtime_digest = os.environ.get("RL_AGENT_RUNTIME_DIGEST", "").strip()
    if expected_digest and explicit_runtime_digest:
        status = "matched" if explicit_runtime_digest == expected_digest else "mismatch"
    elif package_checks and not all(item["equal"] for item in package_checks.values()):
        status = "mismatch"
    else:
        # A host cannot prove a declarative image/lockfile digest merely by
        # importing packages. Keep the distinction explicit for the caller.
        status = "observed" if expected_digest else "unbound"
    return {
        "schema_version": RUNTIME_SCHEMA,
        "status": status,
        "expected_digest": expected_digest,
        "observed_digest": observed_digest,
        "package_checks": package_checks,
        "backend": str(expected_data.get("backend") or "isaaclab"),
        "runtime_id": str(expected_data.get("runtime_id") or ""),
        "observed": observed,
    }


__all__ = ["RUNTIME_SCHEMA", "RuntimeIdentity", "capture_runtime_identity", "resolve_runtime_identity"]
