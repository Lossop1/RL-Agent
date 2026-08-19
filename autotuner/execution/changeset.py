"""Exact, hash-guarded changes for execution-layer inputs.

ChangeSet is intentionally narrower than a general patch system.  Every
change identifies one relative file and either changes one structured config
path or replaces an exact source/runtime text fragment.  All old values are
validated before any write occurs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import shutil
import tempfile
import re
from typing import Any, Mapping

import yaml

from .hashing import canonical_json, safe_relative_path, sha256_bytes, sha256_file


CHANGESET_SCHEMA = "rl-agent.changeset/v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_UNSET = object()


class ChangeSetError(ValueError):
    """A ChangeSet cannot be safely validated or applied."""


@dataclass(frozen=True)
class Change:
    id: str
    kind: str
    path: str
    reason: str
    expected_sha256: str = ""
    key: str = ""
    # A sentinel distinguishes "do not check" from an explicitly expected
    # YAML null.  That distinction matters when a stale config is otherwise
    # indistinguishable from the intended base state.
    expected_value: Any = _UNSET
    new_value: Any = _UNSET
    old_text: str = ""
    new_text: str = ""
    resume_impact: str = "unknown"

    def validate(self) -> None:
        if not _SAFE_ID.fullmatch(self.id.strip()):
            raise ChangeSetError(f"change id is unsafe: {self.id!r}")
        if self.kind not in {"config_path", "source_patch", "runtime_replace"}:
            raise ChangeSetError(f"unsupported change kind: {self.kind!r}")
        if not self.path or Path(self.path).is_absolute() or ".." in Path(self.path).parts:
            raise ChangeSetError(f"change path must be relative: {self.path!r}")
        if not self.reason.strip():
            raise ChangeSetError(f"change {self.id!r} requires a reason")
        if self.kind == "config_path" and not self.key:
            raise ChangeSetError(f"config change {self.id!r} requires a dotted key")
        if self.kind == "config_path" and self.new_value is _UNSET:
            raise ChangeSetError(f"config change {self.id!r} requires new_value")
        if self.kind != "config_path" and not self.new_text and self.new_value is _UNSET and not self.old_text:
            raise ChangeSetError(f"change {self.id!r} requires new_text or new_value")
        if self.kind == "source_patch" and not self.old_text and not self.expected_sha256:
            raise ChangeSetError(f"source patch {self.id!r} requires old_text or expected_sha256")
        if self.kind == "runtime_replace" and not self.expected_sha256:
            raise ChangeSetError(f"runtime replacement {self.id!r} requires expected_sha256")


@dataclass(frozen=True)
class AppliedChangeSet:
    changeset_id: str
    changeset_digest: str
    journal_path: str
    records: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CHANGESET_SCHEMA,
            "changeset_id": self.changeset_id,
            "changeset_digest": self.changeset_digest,
            "journal_path": self.journal_path,
            "records": [dict(item) for item in self.records],
        }


@dataclass(frozen=True)
class ChangeSet:
    id: str
    base_contract_digest: str
    base_source_digest: str
    changes: tuple[Change, ...]
    created_by: str = "execution"
    description: str = ""
    schema_version: str = CHANGESET_SCHEMA

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ChangeSet":
        raw_changes = value.get("changes", ())
        if not isinstance(raw_changes, (list, tuple)):
            raise ChangeSetError("changes must be a list")
        if any(not isinstance(item, Mapping) for item in raw_changes):
            raise ChangeSetError("each change must be an object")
        changes = tuple(_change_from_mapping(item) for item in raw_changes)
        result = cls(
            id=str(value.get("id") or ""),
            base_contract_digest=str(value.get("base_contract_digest") or ""),
            base_source_digest=str(value.get("base_source_digest") or ""),
            changes=changes,
            created_by=str(value.get("created_by") or "execution"),
            description=str(value.get("description") or ""),
            schema_version=str(value.get("schema_version") or CHANGESET_SCHEMA),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != CHANGESET_SCHEMA:
            raise ChangeSetError(f"unsupported changeset schema: {self.schema_version!r}")
        if not _SAFE_ID.fullmatch(self.id.strip()):
            raise ChangeSetError(f"changeset id is unsafe: {self.id!r}")
        if not self.changes:
            raise ChangeSetError("changeset must contain at least one change")
        ids: set[str] = set()
        paths: set[str] = set()
        for change in self.changes:
            change.validate()
            if change.id in ids:
                raise ChangeSetError(f"duplicate change id: {change.id}")
            if change.path in paths:
                raise ChangeSetError(f"multiple changes target the same path: {change.path}")
            ids.add(change.id)
            paths.add(change.path)

    def to_dict(self, *, include_digest: bool = False) -> dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "id": self.id,
            "base_contract_digest": self.base_contract_digest,
            "base_source_digest": self.base_source_digest,
            "created_by": self.created_by,
            "description": self.description,
            "changes": [_change_to_dict(change) for change in self.changes],
        }
        if include_digest:
            data["digest"] = self.digest()
        return data

    def digest(self) -> str:
        return sha256_bytes(canonical_json(self.to_dict()).encode("utf-8"))

    @staticmethod
    def _get_key(data: Any, key: str) -> Any:
        current = data
        for part in key.split("."):
            if not isinstance(current, Mapping) or part not in current:
                raise ChangeSetError(f"config key does not exist: {key}")
            current = current[part]
        return current

    @staticmethod
    def _set_key(data: Any, key: str, value: Any) -> None:
        parts = key.split(".")
        current = data
        for part in parts[:-1]:
            if not isinstance(current, dict) or part not in current:
                raise ChangeSetError(f"config key does not exist: {key}")
            current = current[part]
        if not isinstance(current, dict) or parts[-1] not in current:
            raise ChangeSetError(f"config key does not exist: {key}")
        current[parts[-1]] = value

    def _candidate_bytes(self, root: Path, change: Change) -> tuple[Path, bytes, bytes]:
        target = safe_relative_path(root, change.path)
        if not target.is_file():
            raise ChangeSetError(f"change target does not exist: {change.path}")
        before = target.read_bytes()
        actual = sha256_bytes(before)
        if change.expected_sha256 and actual.lower() != change.expected_sha256.lower():
            raise ChangeSetError(f"{change.id}: old sha256 mismatch for {change.path}")
        if change.kind == "config_path":
            try:
                data = yaml.safe_load(before.decode("utf-8"))
            except (UnicodeDecodeError, yaml.YAMLError) as exc:
                raise ChangeSetError(f"{change.id}: invalid YAML config {change.path}: {exc}") from exc
            actual_value = self._get_key(data, change.key)
            if change.expected_value is not _UNSET and actual_value != change.expected_value:
                raise ChangeSetError(f"{change.id}: old value mismatch at {change.path}#{change.key}")
            self._set_key(data, change.key, change.new_value)
            after_text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
            after = after_text.encode("utf-8")
        else:
            try:
                text = before.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ChangeSetError(f"{change.id}: target is not UTF-8 text: {change.path}") from exc
            if change.kind == "runtime_replace":
                # Runtime files are replaced as a whole, but the expected
                # hash still prevents changing an unreviewed base file.
                after = _replacement_bytes(change)
            elif change.old_text:
                old_text = change.old_text
                if text.count(old_text) != 1:
                    raise ChangeSetError(f"{change.id}: expected exactly one old_text match in {change.path}")
                text = text.replace(old_text, change.new_text, 1)
                after = text.encode("utf-8")
            elif change.new_text:
                after = change.new_text.encode("utf-8")
            else:
                after = json.dumps(change.new_value, ensure_ascii=False, indent=2).encode("utf-8")
        if before == after:
            raise ChangeSetError(f"{change.id}: change produces no content difference")
        return target, before, after

    def apply(
        self,
        root: str | Path,
        *,
        current_contract_digest: str = "",
        current_source_digest: str = "",
        journal_root: str | Path | None = None,
    ) -> AppliedChangeSet:
        """Validate every change, then commit all files with a rollback journal."""
        self.validate()
        if self.base_contract_digest and self.base_contract_digest != current_contract_digest:
            raise ChangeSetError("base contract digest does not match current contract")
        if self.base_source_digest and self.base_source_digest != current_source_digest:
            raise ChangeSetError("base source digest does not match current source")
        workspace = Path(root).resolve()
        prepared = [self._candidate_bytes(workspace, change) for change in self.changes]
        journal_dir = Path(journal_root) if journal_root is not None else workspace / "output" / "changesets"
        journal_dir = journal_dir / self.id
        if journal_dir.exists():
            if (journal_dir / "journal.json").exists():
                raise ChangeSetError(f"changeset has already been applied: {self.id}")
            raise ChangeSetError(f"changeset journal directory already exists: {journal_dir}")
        journal_dir.mkdir(parents=True, exist_ok=True)
        backup_dir = journal_dir / "before"
        journal_path = journal_dir / "journal.json"
        records: list[dict[str, Any]] = []
        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
            # All old values were checked above.  Copy the originals into the
            # rollback area before preparing the durable journal.
            for change, (target, before, after) in zip(self.changes, prepared):
                relative = target.relative_to(workspace)
                backup_target = backup_dir / relative
                backup_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup_target)
                records.append(
                    {
                        "change_id": change.id,
                        "kind": change.kind,
                        "path": relative.as_posix(),
                        "before_sha256": sha256_bytes(before),
                        "after_sha256": sha256_bytes(after),
                        "resume_impact": change.resume_impact,
                    }
                )
            journal = {
                "schema_version": CHANGESET_SCHEMA,
                "state": "prepared",
                "workspace": str(workspace),
                "changeset": self.to_dict(include_digest=True),
                "records": records,
                "backup_dir": str(backup_dir),
            }
            _write_json_durable(journal_path, journal)
        except Exception:
            # A failed backup is not an applied ChangeSet.  Remove only the
            # directory created for this id; never touch an existing journal.
            shutil.rmtree(journal_dir, ignore_errors=True)
            raise
        committed: list[dict[str, Any]] = []
        try:
            # The journal is durable before the first replacement. If a
            # process dies during this loop, recovery has the exact old/new
            # hashes and backup paths instead of an unexplained partial edit.
            with tempfile.TemporaryDirectory(prefix=f"changeset-{self.id}-", dir=str(journal_dir)) as temp_name:
                staged = Path(temp_name)
                for change, (_, _, after) in zip(self.changes, prepared):
                    relative = safe_relative_path(workspace, change.path).relative_to(workspace)
                    stage_target = staged / relative
                    stage_target.parent.mkdir(parents=True, exist_ok=True)
                    stage_target.write_bytes(after)
                for record in records:
                    target = safe_relative_path(workspace, record["path"])
                    staged_target = staged / Path(record["path"])
                    os.replace(staged_target, target)
                    committed.append(record)
        except Exception:
            for record in reversed(committed):
                backup = backup_dir / Path(record["path"])
                target = safe_relative_path(workspace, record["path"])
                if backup.is_file():
                    shutil.copy2(backup, target)
            journal["state"] = "aborted"
            _write_json_durable(journal_path, journal)
            raise
        journal["state"] = "committed"
        _write_json_durable(journal_path, journal)
        return AppliedChangeSet(self.id, self.digest(), str(journal_path), tuple(records))

    @staticmethod
    def rollback(journal_path: str | Path) -> None:
        journal_file = Path(journal_path)
        try:
            journal = json.loads(journal_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ChangeSetError(f"cannot read changeset journal: {journal_file}") from exc
        backup_dir = Path(str(journal.get("backup_dir") or ""))
        records = journal.get("records")
        if not backup_dir.is_dir() or not isinstance(records, list):
            raise ChangeSetError("invalid changeset rollback journal")
        if journal.get("state") != "committed":
            raise ChangeSetError("changeset is not in committed state")
        workspace = Path(str(journal.get("workspace") or journal_file.parent)).resolve()
        _rollback_records(journal, workspace, backup_dir, records, allow_prepared=False)
        journal["state"] = "rolled_back"
        journal["rollback"] = {"records": len(records)}
        _write_json_durable(journal_file, journal)

    @staticmethod
    def recover_prepared(journal_path: str | Path) -> None:
        """Recover a journal left in ``prepared`` state after a process crash.

        Recovery accepts a target that is either the recorded old or new hash,
        then restores every file from the durable backup tree.  Any unrelated
        edit blocks recovery rather than silently overwriting it.
        """
        journal_file = Path(journal_path)
        try:
            journal = json.loads(journal_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ChangeSetError(f"cannot read changeset journal: {journal_file}") from exc
        if journal.get("state") != "prepared":
            raise ChangeSetError("changeset is not in prepared state")
        records = journal.get("records")
        backup_dir = Path(str(journal.get("backup_dir") or ""))
        workspace = Path(str(journal.get("workspace") or journal_file.parent)).resolve()
        if not isinstance(records, list):
            raise ChangeSetError("invalid prepared changeset journal")
        _rollback_records(journal, workspace, backup_dir, records, allow_prepared=True)
        journal["state"] = "recovered"
        journal["recovery"] = {"records": len(records)}
        _write_json_durable(journal_file, journal)


def _change_from_mapping(value: Mapping[str, Any]) -> Change:
    raw = dict(value)
    # The marker keeps old JSON (which always serialized null fields) readable
    # while allowing a new ChangeSet to assert an explicit null value.
    if raw.pop("expected_value_set", True) is False:
        raw.pop("expected_value", None)
        raw["expected_value"] = _UNSET
    if raw.pop("new_value_set", True) is False:
        raw.pop("new_value", None)
        raw["new_value"] = _UNSET
    return Change(**raw)


def _change_to_dict(change: Change) -> dict[str, Any]:
    data = asdict(change)
    expected_set = change.expected_value is not _UNSET
    new_set = change.new_value is not _UNSET
    data["expected_value_set"] = expected_set
    data["new_value_set"] = new_set
    if not expected_set:
        data["expected_value"] = None
    if not new_set:
        data["new_value"] = None
    return data


def _replacement_bytes(change: Change) -> bytes:
    if change.new_text:
        return change.new_text.encode("utf-8")
    if change.new_value is not _UNSET:
        return json.dumps(change.new_value, ensure_ascii=False, indent=2).encode("utf-8")
    raise ChangeSetError(f"change {change.id!r} has no replacement content")


def _write_json_durable(path: Path, value: Mapping[str, Any]) -> None:
    """Write a small journal atomically so a crash cannot leave half JSON."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with temporary.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _safe_backup_path(backup_dir: Path, relative: str) -> Path:
    candidate = (backup_dir / Path(relative)).resolve()
    try:
        candidate.relative_to(backup_dir.resolve())
    except ValueError as exc:
        raise ChangeSetError(f"rollback path escapes backup directory: {relative!r}") from exc
    return candidate


def _rollback_records(
    journal: Mapping[str, Any],
    workspace: Path,
    backup_dir: Path,
    records: list[Mapping[str, Any]],
    *,
    allow_prepared: bool,
) -> None:
    if not backup_dir.is_dir():
        raise ChangeSetError("invalid changeset rollback journal")
    if not allow_prepared and journal.get("state") != "committed":
        raise ChangeSetError("changeset is not in committed state")

    # Preflight every target before changing any file.  A partial rollback is
    # worse than a refusal because it destroys the evidence needed to explain it.
    prepared: list[tuple[Path, Path, str]] = []
    for record in records:
        relative = str(record.get("path") or "")
        target = safe_relative_path(workspace, relative)
        backup = _safe_backup_path(backup_dir, relative)
        if not backup.is_file():
            raise ChangeSetError(f"rollback backup missing: {relative}")
        current = sha256_file(target) if target.is_file() else ""
        before = str(record.get("before_sha256") or "").lower()
        after = str(record.get("after_sha256") or "").lower()
        if current not in {before, after}:
            raise ChangeSetError(f"rollback refused: target changed after ChangeSet: {relative}")
        prepared.append((target, backup, relative))

    stage_root = Path(tempfile.mkdtemp(prefix="rollback-", dir=str(journal_file_parent(backup_dir))))
    try:
        staged: list[tuple[Path, Path, str]] = []
        for target, backup, relative in prepared:
            stage_target = stage_root / Path(relative)
            stage_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, stage_target)
            staged.append((target, stage_target, relative))
        for target, stage_target, _ in staged:
            os.replace(stage_target, target)
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def journal_file_parent(path: Path) -> Path:
    """Return a valid local temp parent even when a journal has no parent."""
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    return parent


__all__ = ["AppliedChangeSet", "CHANGESET_SCHEMA", "Change", "ChangeSet", "ChangeSetError"]
