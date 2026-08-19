"""Content-addressed payload manifest utilities.

The product adapter decides which files belong in a payload.  This module only
records and verifies the resulting tree, so it remains independent of any
robot, simulator task, or training algorithm.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import tarfile
from typing import Any, Iterable, Mapping

from .hashing import canonical_json, sha256_bytes, sha256_file


PAYLOAD_SCHEMA = "rl-agent.payload-manifest/v1"


@dataclass(frozen=True)
class PayloadFile:
    path: str
    sha256: str
    size_bytes: int

    def validate(self) -> None:
        path = Path(self.path)
        if not self.path or path.is_absolute() or ".." in path.parts or "\\" in self.path:
            raise ValueError(f"payload file path must be relative and portable: {self.path!r}")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdefABCDEF" for char in self.sha256):
            raise ValueError(f"payload file has invalid sha256: {self.path!r}")
        if int(self.size_bytes) < 0:
            raise ValueError(f"payload file has negative size: {self.path!r}")


@dataclass(frozen=True)
class PayloadManifest:
    product_id: str
    product_version: str
    contract_digest: str
    runtime_digest: str
    source_digest: str
    files: tuple[PayloadFile, ...]
    schema_version: str = PAYLOAD_SCHEMA
    payload_digest: str = field(default="")

    def validate(self, *, require_digest: bool = False) -> None:
        if self.schema_version != PAYLOAD_SCHEMA:
            raise ValueError(f"unsupported payload manifest schema: {self.schema_version!r}")
        paths: set[str] = set()
        for item in self.files:
            item.validate()
            if item.path in paths:
                raise ValueError(f"duplicate payload file: {item.path}")
            paths.add(item.path)
        if require_digest and (not self.payload_digest or self.payload_digest != self.digest()):
            raise ValueError("payload manifest digest is missing or invalid")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["files"] = [asdict(item) for item in self.files]
        return data

    def _unsigned(self) -> dict[str, Any]:
        data = self.to_dict()
        data.pop("payload_digest", None)
        return data

    def digest(self) -> str:
        return sha256_bytes(canonical_json(self._unsigned()).encode("utf-8"))

    def with_digest(self) -> "PayloadManifest":
        return PayloadManifest(**{**self.to_dict(), "files": self.files, "payload_digest": self.digest()})

    def write(self, path: str | Path) -> Path:
        self.validate(require_digest=bool(self.payload_digest))
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target


def _contract_data(contract: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if hasattr(contract, "to_dict") and callable(contract.to_dict):
        contract = contract.to_dict()
    return contract if isinstance(contract, Mapping) else {}


def build_payload_manifest(
    root: str | Path,
    *,
    contract: Mapping[str, Any] | Any,
    exclude: Iterable[str] = ("payload_manifest.json",),
) -> PayloadManifest:
    """Build a deterministic manifest for all files already staged under root."""
    workspace = Path(root).resolve()
    if not workspace.is_dir():
        raise ValueError(f"payload root is not a directory: {workspace}")
    contract_data = _contract_data(contract)
    excluded = {str(item).replace("\\", "/") for item in exclude}
    files: list[PayloadFile] = []
    for path in sorted(item for item in workspace.rglob("*") if item.is_file()):
        if path.is_symlink():
            raise ValueError(f"payload cannot contain symlink: {path.relative_to(workspace).as_posix()}")
        relative = path.relative_to(workspace).as_posix()
        if relative in excluded:
            continue
        files.append(PayloadFile(relative, sha256_file(path), path.stat().st_size))
    runtime = contract_data.get("runtime") if isinstance(contract_data.get("runtime"), Mapping) else {}
    manifest = PayloadManifest(
        product_id=str(contract_data.get("product_id") or ""),
        product_version=str(contract_data.get("product_version") or ""),
        contract_digest=str(contract_data.get("contract_digest") or ""),
        runtime_digest=str(runtime.get("digest") or ""),
        source_digest=str(contract_data.get("source_digest") or ""),
        files=tuple(files),
    )
    result = manifest.with_digest()
    result.validate(require_digest=True)
    return result


def load_payload_manifest(path: str | Path) -> PayloadManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("payload manifest must be an object")
    files = tuple(PayloadFile(**dict(item)) for item in raw.get("files", ()) if isinstance(item, Mapping))
    manifest = PayloadManifest(
        product_id=str(raw.get("product_id") or ""),
        product_version=str(raw.get("product_version") or ""),
        contract_digest=str(raw.get("contract_digest") or ""),
        runtime_digest=str(raw.get("runtime_digest") or ""),
        source_digest=str(raw.get("source_digest") or ""),
        files=files,
        schema_version=str(raw.get("schema_version") or PAYLOAD_SCHEMA),
        payload_digest=str(raw.get("payload_digest") or ""),
    )
    manifest.validate(require_digest=True)
    return manifest


def verify_payload(root: str | Path, manifest: PayloadManifest) -> list[str]:
    """Return all missing, extra, or hash-mismatched staged files."""
    manifest.validate(require_digest=True)
    workspace = Path(root).resolve()
    expected = {item.path: item for item in manifest.files}
    actual_paths: set[str] = set()
    errors: list[str] = []
    for item in workspace.rglob("*"):
        if item.is_symlink():
            errors.append(f"symlink is not allowed in payload: {item.relative_to(workspace).as_posix()}")
        elif item.is_file():
            actual_paths.add(item.relative_to(workspace).as_posix())
    for relative, item in expected.items():
        path = workspace / relative
        if not path.is_file():
            errors.append(f"missing payload file: {relative}")
        elif sha256_file(path) != item.sha256 or path.stat().st_size != item.size_bytes:
            errors.append(f"payload file hash mismatch: {relative}")
    for relative in sorted(actual_paths - set(expected)):
        if relative != "payload_manifest.json":
            errors.append(f"unexpected payload file: {relative}")
    return errors


def verify_payload_archive(archive: str | Path, manifest: PayloadManifest) -> list[str]:
    """Verify archive members before an archive is uploaded to a remote host.

    Tar extraction is an outward-facing operation.  Reject absolute paths,
    traversal, links, missing files, and content mismatches locally so the
    remote transport never becomes the first integrity check.
    """
    manifest.validate(require_digest=True)
    expected = {item.path: item for item in manifest.files}
    errors: list[str] = []
    seen: set[str] = set()
    embedded_manifest: bytes | None = None
    try:
        handle = tarfile.open(Path(archive), "r:*")
    except (OSError, tarfile.TarError) as exc:
        return [f"cannot open payload archive: {exc}"]
    with handle:
        for member in handle.getmembers():
            name = member.name.replace("\\", "/")
            candidate = Path(name)
            if not name or candidate.is_absolute() or ".." in candidate.parts:
                errors.append(f"unsafe archive member: {member.name!r}")
                continue
            if member.issym() or member.islnk() or not member.isfile():
                errors.append(f"unsupported archive member: {member.name!r}")
                continue
            if name == "payload_manifest.json":
                if name in seen:
                    errors.append("duplicate archive file: payload_manifest.json")
                else:
                    extracted = handle.extractfile(member)
                    embedded_manifest = extracted.read() if extracted is not None else b""
                seen.add(name)
                continue
            if name not in expected:
                errors.append(f"unexpected archive file: {name}")
                continue
            if name in seen:
                errors.append(f"duplicate archive file: {name}")
                continue
            extracted = handle.extractfile(member)
            data = extracted.read() if extracted is not None else b""
            item = expected[name]
            if len(data) != item.size_bytes or sha256_bytes(data) != item.sha256:
                errors.append(f"archive file hash mismatch: {name}")
            seen.add(name)
    for name in sorted(set(expected) - seen):
        errors.append(f"missing archive file: {name}")
    if embedded_manifest is None:
        errors.append("missing archive file: payload_manifest.json")
    else:
        try:
            embedded_data = json.loads(embedded_manifest.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            errors.append(f"embedded payload manifest is invalid: {exc}")
        else:
            if canonical_json(embedded_data) != canonical_json(manifest.to_dict()):
                errors.append("embedded payload manifest does not match the supplied manifest")
    return errors


__all__ = [
    "PAYLOAD_SCHEMA",
    "PayloadFile",
    "PayloadManifest",
    "build_payload_manifest",
    "load_payload_manifest",
    "verify_payload",
    "verify_payload_archive",
]
