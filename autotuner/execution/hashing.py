"""Small deterministic hashing helpers used by the execution layer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(path: str | Path) -> tuple[str, int]:
    """Hash file content and relative names, independent of mtimes."""
    candidate = Path(path)
    if candidate.is_file():
        return sha256_file(candidate), 1
    if not candidate.is_dir():
        return "", 0
    digest = hashlib.sha256()
    count = 0
    for item in sorted(item for item in candidate.rglob("*") if item.is_file()):
        digest.update(item.relative_to(candidate).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
        count += 1
    return digest.hexdigest(), count


def paths_digest(root: str | Path, paths: Iterable[str | Path]) -> str:
    """Hash a declared set of workspace paths in stable order."""
    base = Path(root).resolve()
    records: list[dict[str, Any]] = []
    for raw in sorted({str(item) for item in paths}):
        item = (base / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
        try:
            relative = item.relative_to(base).as_posix()
        except ValueError:
            relative = str(item)
        digest, count = tree_digest(item)
        records.append({"path": relative, "sha256": digest, "file_count": count, "exists": item.exists()})
    return sha256_bytes(canonical_json(records).encode("utf-8"))


def safe_relative_path(root: str | Path, value: str | Path) -> Path:
    """Resolve a relative path and reject traversal outside *root*."""
    base = Path(root).resolve()
    raw = Path(value)
    if raw.is_absolute():
        raise ValueError(f"path must be relative to workspace: {value!s}")
    candidate = (base / raw).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"path escapes workspace: {value!s}") from exc
    return candidate
