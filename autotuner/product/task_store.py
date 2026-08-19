"""任务合同的本地版本仓库。

这个模块只保存合同和已经生成的 bundle，不执行训练，也不修改产品源码。
每个版本目录一旦写入就不再覆盖；状态变化通过追加事件记录，便于回滚、审计
以及在上下文压缩后重新定位一条任务谱系。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Mapping

from .task_contract import TaskContractBundle, TaskContractError, ResolvedTaskContract


STORE_SCHEMA = "rl-agent.task-contract-store/v1"
_SAFE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/-]*$")
_STATUSES = frozenset({"draft", "approved", "rejected", "superseded"})


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    import hashlib

    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _safe_component(value: str, name: str) -> str:
    result = str(value or "").strip()
    if not result or any(char in result for char in ("/", "\\", "\x00")):
        raise TaskContractError(f"{name} contains an unsafe path component: {result!r}")
    # 合同 id 允许冒号用于命名空间；文件系统目录使用无歧义编码，避免
    # Windows 将冒号解释成驱动器分隔符，也避免不同 id 映射到同一目录。
    return result.replace(":", "%3A")


def _atomic_write(path: Path, text: str) -> None:
    """先写临时文件并 fsync，再替换目标，避免留下半份合同。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@dataclass(frozen=True)
class StoredTaskContract:
    """已保存的合同版本及其稳定引用。"""

    ref: str
    contract_id: str
    contract_version: int
    status: str
    parent_ref: str
    supersedes_ref: str
    request_digest: str
    product_contract_digest: str
    contract_digest: str
    bundle_digest: str
    path: Path
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STORE_SCHEMA,
            "ref": self.ref,
            "contract_id": self.contract_id,
            "contract_version": self.contract_version,
            "status": self.status,
            "parent_ref": self.parent_ref,
            "supersedes_ref": self.supersedes_ref,
            "request_digest": self.request_digest,
            "product_contract_digest": self.product_contract_digest,
            "contract_digest": self.contract_digest,
            "bundle_digest": self.bundle_digest,
            "path": str(self.path),
            "created_at": self.created_at,
        }


class TaskContractStore:
    """以文件系统保存合同版本，并维护追加式状态历史。"""

    _lock = threading.Lock()

    def __init__(self, root: str | os.PathLike[str] = "output/task_contracts") -> None:
        self.root = Path(root)
        self.revisions_root = self.root / "revisions"
        self.index_path = self.root / "index.jsonl"

    @staticmethod
    def revision_ref(contract: ResolvedTaskContract | TaskContractBundle) -> str:
        contract_obj = contract.contract if isinstance(contract, TaskContractBundle) else contract
        return f"{contract_obj.contract_id}@{contract_obj.contract_version}"

    def _revision_path(self, contract_id: str, version: int) -> Path:
        return self.revisions_root / _safe_component(contract_id, "contract_id") / f"v{int(version):06d}"

    def _read_index(self, *, verify: bool = True) -> list[dict[str, Any]]:
        if not self.index_path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        previous = ""
        for sequence, line in enumerate(self.index_path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    if verify:
                        if value.get("sequence") != sequence:
                            raise TaskContractError(
                                f"task contract index sequence gap: expected {sequence}, got {value.get('sequence')}"
                            )
                        if value.get("previous_event_digest", "") != previous:
                            raise TaskContractError(f"task contract index chain break at sequence {sequence}")
                        event_digest = value.get("event_digest")
                        body = {key: item for key, item in value.items() if key != "event_digest"}
                        if event_digest != _digest(body):
                            raise TaskContractError(f"task contract index digest mismatch at sequence {sequence}")
                        previous = str(event_digest)
                    rows.append(value)
        return rows

    def _append_index(self, value: Mapping[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        rows = self._read_index(verify=True)
        event = dict(value)
        event["sequence"] = len(rows) + 1
        event["previous_event_digest"] = str(rows[-1].get("event_digest") or "") if rows else ""
        event["event_digest"] = _digest(event)
        line = _canonical(event) + "\n"
        with self.index_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def save(
        self,
        bundle: TaskContractBundle,
        *,
        parent_ref: str = "",
        supersedes_ref: str = "",
        status: str | None = None,
        actor: str = "system",
    ) -> StoredTaskContract:
        """原子保存一个新版本；同一合同版本内容不同会直接拒绝。"""
        contract = bundle.contract
        selected_status = str(status or contract.status or "draft")
        if selected_status not in _STATUSES:
            raise TaskContractError(f"unsupported task contract status: {selected_status!r}")
        if selected_status == "approved" and contract.status != "approved":
            raise TaskContractError("approved store record requires an approved task contract")
        ref = self.revision_ref(bundle)
        if not _SAFE_REF.fullmatch(ref):
            raise TaskContractError(f"unsafe task contract reference: {ref!r}")
        if parent_ref and self.find(str(parent_ref)) is None:
            raise TaskContractError(f"parent task contract revision not found: {parent_ref}")
        if supersedes_ref and self.find(str(supersedes_ref)) is None:
            raise TaskContractError(f"superseded task contract revision not found: {supersedes_ref}")
        revision_path = self._revision_path(contract.contract_id, contract.contract_version)
        record = StoredTaskContract(
            ref=ref,
            contract_id=contract.contract_id,
            contract_version=contract.contract_version,
            status=selected_status,
            parent_ref=str(parent_ref or ""),
            supersedes_ref=str(supersedes_ref or ""),
            request_digest=contract.request_digest,
            product_contract_digest=contract.product_contract_digest,
            contract_digest=contract.contract_digest,
            bundle_digest=bundle.bundle_digest,
            path=revision_path,
            created_at=_now(),
        )
        with self._lock:
            existing = self.find(ref)
            if existing is not None:
                if existing.contract_digest != record.contract_digest or existing.bundle_digest != record.bundle_digest:
                    raise TaskContractError(f"task contract revision already exists with different content: {ref}")
                return existing
            staging = revision_path.with_name(f".{revision_path.name}.staging-{os.getpid()}")
            if staging.exists():
                raise TaskContractError(f"stale task contract staging directory exists: {staging}")
            try:
                bundle.write(staging)
                _atomic_write(staging / "store_record.json", _canonical(record.to_dict()) + "\n")
                revision_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging, revision_path)
                event = {
                    "schema_version": STORE_SCHEMA,
                    "event": "save",
                    "actor": actor,
                    "created_at": record.created_at,
                    "record": record.to_dict(),
                }
                self._append_index(event)
            except Exception:
                if staging.exists():
                    import shutil

                    shutil.rmtree(staging, ignore_errors=True)
                raise
        return record

    def find(self, ref: str) -> StoredTaskContract | None:
        for row in reversed(self._read_index()):
            record = row.get("record") if isinstance(row.get("record"), dict) else {}
            if record.get("ref") == ref:
                return self._record_from_mapping(record)
        return None

    def get(self, ref: str) -> StoredTaskContract:
        result = self.find(ref)
        if result is None:
            raise KeyError(f"task contract revision not found: {ref}")
        return result

    def history(self, contract_id: str = "") -> list[StoredTaskContract]:
        latest: dict[str, StoredTaskContract] = {}
        for row in self._read_index():
            record = row.get("record") if isinstance(row.get("record"), dict) else {}
            ref = str(record.get("ref") or "")
            if ref and (not contract_id or record.get("contract_id") == contract_id):
                latest[ref] = self._record_from_mapping(record)
        return sorted(latest.values(), key=lambda item: (item.contract_id, item.contract_version))

    def set_status(self, ref: str, status: str, *, actor: str = "system", reason: str = "") -> StoredTaskContract:
        """追加状态事件，不改写历史版本文件。"""
        if status not in _STATUSES:
            raise TaskContractError(f"unsupported task contract status: {status!r}")
        current = self.get(ref)
        if status == "approved":
            contract = json.loads((current.path / "task_contract.json").read_text(encoding="utf-8"))
            if contract.get("status") != "approved":
                raise TaskContractError("only an approved contract can enter approved store status")
        updated = StoredTaskContract(**{**current.__dict__, "status": status})
        event = {
            "schema_version": STORE_SCHEMA,
            "event": "status",
            "actor": actor,
            "created_at": _now(),
            "reason": reason,
            "record": updated.to_dict(),
        }
        with self._lock:
            self._append_index(event)
        return updated

    def load_bundle(self, ref: str) -> TaskContractBundle:
        """从保存目录恢复 bundle，供回滚或重启后的研究周期继续使用。"""
        stored = self.get(ref)
        contract_data = json.loads((stored.path / "task_contract.json").read_text(encoding="utf-8"))
        contract = ResolvedTaskContract.from_mapping(contract_data)
        specs: dict[str, Mapping[str, Any]] = {}
        for path in sorted(stored.path.glob("*_spec.json")):
            specs[path.stem.removesuffix("_spec")] = json.loads(path.read_text(encoding="utf-8"))
        return TaskContractBundle(contract=contract, specs=specs, bundle_digest=stored.bundle_digest)

    @staticmethod
    def _record_from_mapping(value: Mapping[str, Any]) -> StoredTaskContract:
        data = dict(value)
        data.pop("schema_version", None)
        data["path"] = Path(str(data["path"]))
        return StoredTaskContract(**data)


__all__ = ["STORE_SCHEMA", "StoredTaskContract", "TaskContractStore"]
