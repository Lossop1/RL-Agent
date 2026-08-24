"""传统控制产物的稳定清单与追加式运行记录。"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Mapping, Sequence


ARTIFACT_MANIFEST_SCHEMA = "traditional_control_artifact_manifest_v2"
RUN_MANIFEST_SCHEMA = "traditional_control_run_manifest_v1"


def canonical_json_bytes(value: object) -> bytes:
    """返回不受字典插入顺序和本地编码影响的规范 JSON。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip()


def git_revision(root: Path) -> str:
    return _git(root, "rev-parse", "HEAD") or "unavailable"


def git_worktree_state(root: Path) -> str:
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status == "unavailable":
        return status
    return "dirty" if status else "clean"


def dependency_versions(distributions: Sequence[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in sorted(set(distributions)):
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = "unavailable"
    return versions


def _file_entries(root: Path, files: Iterable[str | Path]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in files:
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"manifest file is outside root: {path}") from exc
        if relative in seen:
            raise ValueError(f"manifest contains duplicate file: {relative}")
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(relative)
        entries.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size": int(path.stat().st_size),
            }
        )
    return sorted(entries, key=lambda item: str(item["path"]))


def build_manifest(
    root: str | Path,
    files: Iterable[str | Path],
    *,
    artifact_type: str,
    schema_version: str = ARTIFACT_MANIFEST_SCHEMA,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """构造只由稳定输入决定的 artifact manifest。

    时间、主机、Python 版本和 Git 工作树状态属于一次运行，不得进入 bundle，
    否则相同源码和配置无法生成相同字节。
    """

    root_path = Path(root).resolve()
    identity: dict[str, object] = {
        "schema_version": str(schema_version),
        "artifact_type": str(artifact_type),
        "files": _file_entries(root_path, files),
        "metadata": dict(metadata or {}),
    }
    return {
        **identity,
        "artifact_digest": sha256_bytes(canonical_json_bytes(identity)),
    }


def build_run_manifest(
    root: str | Path,
    artifact_manifest: Mapping[str, object],
    *,
    archive_path: str | Path,
    archive_sha256: str,
    operation: str,
    dependencies: Sequence[str] = (),
    metadata: Mapping[str, object] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, object]:
    """记录一次构建或运行；该记录有时间性，不进入稳定 bundle。"""

    root_path = Path(root).resolve()
    archive = Path(archive_path).resolve()
    try:
        archive_ref = archive.relative_to(root_path).as_posix()
    except ValueError as exc:
        raise ValueError(f"run artifact is outside root: {archive}") from exc
    timestamp = generated_at or datetime.now(timezone.utc)
    record: dict[str, object] = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "generated_at": timestamp.astimezone(timezone.utc).isoformat(),
        "operation": str(operation),
        "artifact": {
            "type": artifact_manifest["artifact_type"],
            "digest": artifact_manifest["artifact_digest"],
            "manifest_schema": artifact_manifest["schema_version"],
            "archive": archive_ref,
            "archive_sha256": str(archive_sha256),
        },
        "source": {
            "git_revision": git_revision(root_path),
            "git_worktree": git_worktree_state(root_path),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "dependencies": dependency_versions(dependencies),
        },
        "metadata": dict(metadata or {}),
    }
    record["run_digest"] = sha256_bytes(canonical_json_bytes(record))
    return record


def write_manifest(path: str | Path, manifest: Mapping[str, object]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(target)
    return target
