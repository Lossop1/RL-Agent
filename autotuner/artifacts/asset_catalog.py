"""产品无关的内容寻址资产目录与复用审批。

目录只保存可验证的资产事实和谱系，不判断某个机器人应当使用什么资产。
登记、复用批准和运行绑定是三个独立步骤，避免把“文件存在”误当成“可以复用”。
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Literal, Mapping, Sequence
import uuid

from pydantic import BaseModel, ConfigDict, Field


ASSET_CATALOG_SCHEMA = "rl-agent.asset-catalog/v1"
ASSET_EVENT_SCHEMA = "rl-agent.asset-catalog-event/v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LINEAGE_REQUIRED_KINDS = frozenset(
    {
        "checkpoint",
        "policy_checkpoint",
        "training_baseline",
        "optimizer_state",
        "normalizer_state",
    }
)


class AssetCatalogError(ValueError):
    """资产身份、兼容性、审批或目录完整性不成立。"""


def _canonical(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_id(value: Any, name: str) -> str:
    result = str(value or "").strip()
    if not _SAFE_ID.fullmatch(result) or ".." in result.split("/"):
        raise AssetCatalogError(f"{name} is unsafe or empty: {result!r}")
    return result


def _sha256(value: Any, name: str = "digest") -> str:
    result = str(value or "").strip().lower()
    if not _SHA256.fullmatch(result):
        raise AssetCatalogError(f"{name} must be a lowercase sha256 digest")
    return result


def _path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "asset"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class AssetLineage(BaseModel):
    """资产的来源边；训练状态类资产必须能回到合同和运行。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_product_ref: str = ""
    source_task_contract_ref: str = ""
    source_run_ref: str = ""
    source_checkpoint_ref: str = ""
    parent_asset_refs: tuple[str, ...] = ()


class ReusableAsset(BaseModel):
    """一份内容寻址资产及其可验证适用范围。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = ASSET_CATALOG_SCHEMA
    asset_ref: str
    asset_id: str
    kind: str
    sha256: str
    locator: str
    compatibility: dict[str, str] = Field(default_factory=dict)
    capability_scope: tuple[str, ...] = ()
    validation_evidence_refs: tuple[str, ...] = ()
    lineage: AssetLineage = Field(default_factory=AssetLineage)
    metadata: dict[str, Any] = Field(default_factory=dict)
    registered_at: str = Field(default_factory=_now)


class AssetCompatibilityReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_ref: str
    expected_digest: str
    digest_match: bool
    compatible: bool
    missing_context: tuple[str, ...] = ()
    mismatches: tuple[str, ...] = ()
    missing_capabilities: tuple[str, ...] = ()


class AssetReuseApproval(BaseModel):
    """人对一次确定目标的复用决策；它不是跨任务的永久白名单。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = ASSET_CATALOG_SCHEMA
    approval_ref: str
    asset_ref: str
    expected_digest: str
    target_contract_ref: str
    target_run_ref: str
    role: str
    target_context: dict[str, str] = Field(default_factory=dict)
    required_capabilities: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    approved_by: str
    decision: Literal["approved", "rejected"]
    reason: str
    compatibility_report: AssetCompatibilityReport
    created_at: str = Field(default_factory=_now)


class AssetBinding(BaseModel):
    """经过批准的资产与目标合同、目标运行之间的不可变绑定。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = ASSET_CATALOG_SCHEMA
    binding_ref: str
    approval_ref: str
    asset_ref: str
    target_contract_ref: str
    target_run_ref: str
    role: str
    created_at: str = Field(default_factory=_now)


class AssetCatalogEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = ASSET_EVENT_SCHEMA
    event_id: str
    sequence: int = Field(ge=1)
    event_type: Literal["asset_registered", "reuse_decided", "asset_bound"]
    actor: str
    created_at: str
    payload: dict[str, Any]
    previous_event_hash: str = ""
    event_hash: str


class AssetCatalog:
    """持久化内容寻址资产、一次性复用审批和运行绑定。"""

    _lock = threading.RLock()

    def __init__(self, root: str | os.PathLike[str] = "output/assets/catalog") -> None:
        self.root = Path(root)
        self.events_path = self.root / "events.jsonl"

    def _asset_path(self, asset: ReusableAsset) -> Path:
        suffix = hashlib.sha256(asset.asset_ref.encode("utf-8")).hexdigest()[:16]
        return self.root / "assets" / _path_component(asset.kind) / asset.sha256 / f"{suffix}.json"

    def _approval_path(self, approval_ref: str) -> Path:
        suffix = hashlib.sha256(approval_ref.encode("utf-8")).hexdigest()[:16]
        return self.root / "approvals" / f"{suffix}.json"

    def _binding_path(self, binding_ref: str) -> Path:
        suffix = hashlib.sha256(binding_ref.encode("utf-8")).hexdigest()[:16]
        return self.root / "bindings" / f"{suffix}.json"

    @staticmethod
    def _event_body(value: AssetCatalogEvent | Mapping[str, Any]) -> dict[str, Any]:
        data = value.model_dump(mode="json") if isinstance(value, AssetCatalogEvent) else dict(value)
        data.pop("event_hash", None)
        return data

    def events(self) -> list[AssetCatalogEvent]:
        if not self.events_path.is_file():
            return []
        result: list[AssetCatalogEvent] = []
        previous = ""
        for sequence, line in enumerate(self.events_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                event = AssetCatalogEvent.model_validate_json(line)
            except Exception as exc:
                raise AssetCatalogError(f"invalid asset catalog event at line {sequence}: {exc}") from exc
            if event.sequence != sequence or event.previous_event_hash != previous:
                raise AssetCatalogError(f"asset catalog event chain breaks at line {sequence}")
            if event.event_hash != _digest(self._event_body(event)):
                raise AssetCatalogError(f"asset catalog event hash mismatch at line {sequence}")
            result.append(event)
            previous = event.event_hash
        return result

    def _append(self, event_type: str, payload: BaseModel, *, actor: str) -> AssetCatalogEvent:
        actor = _safe_id(actor, "actor")
        events = self.events()
        body = {
            "schema_version": ASSET_EVENT_SCHEMA,
            "event_id": f"asset-event:{uuid.uuid4().hex}",
            "sequence": len(events) + 1,
            "event_type": event_type,
            "actor": actor,
            "created_at": _now(),
            "payload": payload.model_dump(mode="json"),
            "previous_event_hash": events[-1].event_hash if events else "",
        }
        event = AssetCatalogEvent(**body, event_hash=_digest(body))
        self.root.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(event.model_dump_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return event

    def _latest(self, event_type: str, key: str, value: str) -> dict[str, Any] | None:
        for event in reversed(self.events()):
            if event.event_type == event_type and str(event.payload.get(key) or "") == value:
                return event.payload
        return None

    def get_asset(self, asset_ref: str) -> ReusableAsset:
        payload = self._latest("asset_registered", "asset_ref", asset_ref)
        if payload is None:
            raise KeyError(f"asset not found: {asset_ref}")
        return ReusableAsset.model_validate(payload)

    def get_approval(self, approval_ref: str) -> AssetReuseApproval:
        payload = self._latest("reuse_decided", "approval_ref", approval_ref)
        if payload is None:
            raise KeyError(f"asset reuse approval not found: {approval_ref}")
        return AssetReuseApproval.model_validate(payload)

    def get_binding(self, binding_ref: str) -> AssetBinding:
        payload = self._latest("asset_bound", "binding_ref", binding_ref)
        if payload is None:
            raise KeyError(f"asset binding not found: {binding_ref}")
        return AssetBinding.model_validate(payload)

    def register_asset(
        self,
        *,
        asset_id: str,
        kind: str,
        sha256: str,
        locator: str,
        compatibility: Mapping[str, Any] | None = None,
        capability_scope: Sequence[str] = (),
        validation_evidence_refs: Sequence[str] = (),
        lineage: AssetLineage | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        actor: str = "asset-catalog",
    ) -> ReusableAsset:
        asset_id = _safe_id(asset_id, "asset_id")
        kind = _safe_id(kind, "kind")
        sha256 = _sha256(sha256)
        locator = str(locator or "").strip()
        if not locator:
            raise AssetCatalogError("asset locator is required")
        asset_lineage = (
            lineage
            if isinstance(lineage, AssetLineage)
            else AssetLineage.model_validate(lineage or {})
        )
        if kind in _LINEAGE_REQUIRED_KINDS and (
            not asset_lineage.source_task_contract_ref.strip()
            or not asset_lineage.source_run_ref.strip()
        ):
            raise AssetCatalogError(
                f"{kind} requires explicit source_task_contract_ref and source_run_ref"
            )
        asset_ref = f"asset:{kind}:{sha256}:{asset_id}"
        record = ReusableAsset(
            asset_ref=asset_ref,
            asset_id=asset_id,
            kind=kind,
            sha256=sha256,
            locator=locator,
            compatibility={
                str(key): str(value)
                for key, value in (compatibility or {}).items()
                if str(key).strip() and str(value).strip()
            },
            capability_scope=tuple(dict.fromkeys(str(value) for value in capability_scope if str(value).strip())),
            validation_evidence_refs=tuple(
                dict.fromkeys(str(value) for value in validation_evidence_refs if str(value).strip())
            ),
            lineage=asset_lineage,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            existing_payload = self._latest("asset_registered", "asset_ref", asset_ref)
            if existing_payload is not None:
                existing = ReusableAsset.model_validate(existing_payload)
                if existing != record.model_copy(update={"registered_at": existing.registered_at}):
                    raise AssetCatalogError(f"asset reference already exists with different metadata: {asset_ref}")
                return existing
            _atomic_json(self._asset_path(record), record.model_dump(mode="json"))
            self._append("asset_registered", record, actor=actor)
        return record

    def evaluate_compatibility(
        self,
        asset_ref: str,
        *,
        expected_digest: str,
        target_context: Mapping[str, Any],
        required_capabilities: Sequence[str] = (),
    ) -> AssetCompatibilityReport:
        asset = self.get_asset(asset_ref)
        expected = _sha256(expected_digest, "expected_digest")
        context = {str(key): str(value) for key, value in target_context.items()}
        missing: list[str] = []
        mismatches: list[str] = []
        for key, wanted in asset.compatibility.items():
            if wanted == "*":
                continue
            actual = context.get(key, "")
            if not actual:
                missing.append(key)
            elif actual != wanted:
                mismatches.append(f"{key}: expected {wanted!r}, got {actual!r}")
        available = set(asset.capability_scope)
        missing_capabilities = sorted(
            str(value)
            for value in required_capabilities
            if str(value).strip() and str(value) not in available
        )
        digest_match = expected == asset.sha256
        compatible = digest_match and not missing and not mismatches and not missing_capabilities
        return AssetCompatibilityReport(
            asset_ref=asset.asset_ref,
            expected_digest=expected,
            digest_match=digest_match,
            compatible=compatible,
            missing_context=tuple(sorted(missing)),
            mismatches=tuple(mismatches),
            missing_capabilities=tuple(missing_capabilities),
        )

    def decide_reuse(
        self,
        asset_ref: str,
        *,
        approval_ref: str,
        expected_digest: str,
        target_contract_ref: str,
        target_run_ref: str,
        role: str,
        target_context: Mapping[str, Any],
        required_capabilities: Sequence[str] = (),
        evidence_refs: Sequence[str],
        approved_by: str,
        decision: Literal["approved", "rejected"],
        reason: str,
        actor: str = "asset-catalog",
    ) -> AssetReuseApproval:
        approval_ref = _safe_id(approval_ref, "approval_ref")
        target_contract_ref = _safe_id(target_contract_ref, "target_contract_ref")
        target_run_ref = _safe_id(target_run_ref, "target_run_ref")
        role = _safe_id(role, "role")
        approver = str(approved_by or "").strip()
        refs = tuple(dict.fromkeys(str(value) for value in evidence_refs if str(value).strip()))
        if not approver or not refs or not str(reason or "").strip():
            raise AssetCatalogError("approved_by, evidence_refs and reason are required")
        asset = self.get_asset(asset_ref)
        report = self.evaluate_compatibility(
            asset_ref,
            expected_digest=expected_digest,
            target_context=target_context,
            required_capabilities=required_capabilities,
        )
        if decision == "approved" and not asset.validation_evidence_refs:
            raise AssetCatalogError("an asset without validation evidence cannot be approved for reuse")
        if decision == "approved" and not report.compatible:
            raise AssetCatalogError("an incompatible asset cannot be approved for reuse")
        approval = AssetReuseApproval(
            approval_ref=approval_ref,
            asset_ref=asset_ref,
            expected_digest=report.expected_digest,
            target_contract_ref=target_contract_ref,
            target_run_ref=target_run_ref,
            role=role,
            target_context={str(key): str(value) for key, value in target_context.items()},
            required_capabilities=tuple(
                dict.fromkeys(str(value) for value in required_capabilities if str(value).strip())
            ),
            evidence_refs=refs,
            approved_by=approver,
            decision=decision,
            reason=str(reason).strip(),
            compatibility_report=report,
        )
        with self._lock:
            existing_payload = self._latest("reuse_decided", "approval_ref", approval_ref)
            if existing_payload is not None:
                existing = AssetReuseApproval.model_validate(existing_payload)
                if existing != approval.model_copy(update={"created_at": existing.created_at}):
                    raise AssetCatalogError(f"approval reference already exists with different content: {approval_ref}")
                return existing
            _atomic_json(self._approval_path(approval_ref), approval.model_dump(mode="json"))
            self._append("reuse_decided", approval, actor=actor)
        return approval

    def bind_approved_reuse(
        self,
        approval_ref: str,
        *,
        target_contract_ref: str,
        target_run_ref: str,
        actor: str = "asset-catalog",
    ) -> AssetBinding:
        approval = self.get_approval(approval_ref)
        if approval.decision != "approved":
            raise AssetCatalogError("rejected asset reuse cannot be bound")
        if approval.target_contract_ref != target_contract_ref or approval.target_run_ref != target_run_ref:
            raise AssetCatalogError("asset reuse approval target does not match the requested binding")
        identity = _digest(
            {
                "approval_ref": approval.approval_ref,
                "asset_ref": approval.asset_ref,
                "target_contract_ref": target_contract_ref,
                "target_run_ref": target_run_ref,
                "role": approval.role,
            }
        )
        binding = AssetBinding(
            binding_ref=f"asset-binding:{identity[:32]}",
            approval_ref=approval.approval_ref,
            asset_ref=approval.asset_ref,
            target_contract_ref=target_contract_ref,
            target_run_ref=target_run_ref,
            role=approval.role,
        )
        with self._lock:
            existing_payload = self._latest("asset_bound", "binding_ref", binding.binding_ref)
            if existing_payload is not None:
                return AssetBinding.model_validate(existing_payload)
            _atomic_json(self._binding_path(binding.binding_ref), binding.model_dump(mode="json"))
            self._append("asset_bound", binding, actor=actor)
        return binding


__all__ = [
    "ASSET_CATALOG_SCHEMA",
    "AssetBinding",
    "AssetCatalog",
    "AssetCatalogError",
    "AssetCompatibilityReport",
    "AssetLineage",
    "AssetReuseApproval",
    "ReusableAsset",
]
