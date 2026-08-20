"""Strict, explainable resume compatibility checks."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Mapping

try:
    from .hashing import canonical_json
except ImportError:  # payload 副本将辅助函数放在本模块旁边
    from .execution_hashing import canonical_json  # type: ignore


COMPATIBILITY_SCHEMA = "rl-agent.resume-compatibility/v1"


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    return value if isinstance(value, Mapping) else {}


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _first(mapping: Mapping[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value = _nested(mapping, *path)
        if value not in (None, "", [], {}):
            return value
    return None


def _normal(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normal(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_normal(item) for item in value]
    return value


def _fingerprint(manifest: Mapping[str, Any] | Any) -> dict[str, Any]:
    data = _as_mapping(manifest)
    contract = _as_mapping(data.get("resolved_contract") or data.get("contract") or data)
    execution = _as_mapping(data.get("execution"))
    training = _as_mapping(contract.get("training"))
    compatibility = _as_mapping(contract.get("compatibility") or data.get("compatibility"))
    runtime = _as_mapping(contract.get("runtime") or data.get("runtime"))
    payload = _as_mapping(data.get("payload"))
    configuration = _as_mapping(data.get("configuration"))
    return {
        "product": _first(data, ("product", "id"), ("product_id",)) or _first(contract, ("product_id",)),
        "product_version": _first(data, ("product", "version"), ("product_version",)) or _first(contract, ("product_version",)),
        "task": _first(data, ("run", "task"), ("task",)) or training.get("task_id"),
        "observation_structure": _first(compatibility, ("observation_structure",)) or _first(training, ("observation_structure",), ("observation",)),
        "action_structure": _first(compatibility, ("action_structure",)) or _first(training, ("action_structure",), ("action",)),
        "network_structure": _first(compatibility, ("network_structure",)) or _first(training, ("network_structure",), ("network",)),
        "normalization": _first(compatibility, ("normalization",)) or _first(training, ("normalization",), ("normalizer",)),
        "physics_timestep": _first(compatibility, ("physics_timestep",), ("timestep",)) or _first(runtime, ("physics_timestep",), ("timestep",)),
        "runtime_digest": runtime.get("digest")
        or execution.get("runtime_digest")
        or data.get("runtime_digest"),
        "task_contract_digest": execution.get("task_contract_digest"),
        "task_bundle_digest": execution.get("task_bundle_digest"),
        "config_digest": _first(contract, ("config_digest",)) or _first(configuration, ("effective_config", "sha256")),
        "source_digest": _first(contract, ("source_digest",)),
        # 合同身份与 payload 身份分别记录；不能用合同摘要冒充 payload 摘要。
        # 否则仅源码变化就会被误判为运行包仍然相同。
        "payload_digest": (
            payload.get("payload_digest")
            or payload.get("digest")
            or data.get("payload_digest")
            or execution.get("payload_digest")
            or _first(contract, ("payload_digest",))
        ),
    }


@dataclass(frozen=True)
class ResumeCompatibility:
    compatible: bool
    decision: str
    reasons: tuple[str, ...] = ()
    compared: Mapping[str, Any] = field(default_factory=dict)
    checkpoint: Mapping[str, Any] = field(default_factory=dict)
    fingerprint_digest: str = ""
    schema_version: str = COMPATIBILITY_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "compatible": self.compatible,
            "decision": self.decision,
            "reasons": list(self.reasons),
            "compared": dict(self.compared),
            "checkpoint": dict(self.checkpoint),
            "fingerprint_digest": self.fingerprint_digest,
        }


@dataclass(frozen=True)
class ResumeProof:
    """运行时恢复回执的确定性判定；声明本身不能产生 proven。"""

    proven: bool
    status: str
    reasons: tuple[str, ...] = ()
    compatibility: ResumeCompatibility | None = None
    checks: Mapping[str, bool] = field(default_factory=dict)
    state_components: Mapping[str, bool] = field(default_factory=dict)
    identity: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = COMPATIBILITY_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "proven": self.proven,
            "status": self.status,
            "reasons": list(self.reasons),
            "compatibility": self.compatibility.to_dict() if self.compatibility else {},
            "checks": dict(self.checks),
            "state_components": dict(self.state_components),
            "identity": dict(self.identity),
        }


_REQUIRED = (
    "product",
    "product_version",
    "task",
    "observation_structure",
    "action_structure",
    "network_structure",
    "normalization",
    "physics_timestep",
    "runtime_digest",
    "config_digest",
    "source_digest",
    "payload_digest",
)

# 这些字段决定检查点能否被当前运行时安全解释，变化时必须阻断 resume。
_RESUME_ABI_FIELDS = frozenset(
    {
        "product",
        "product_version",
        "task",
        "observation_structure",
        "action_structure",
        "network_structure",
        "normalization",
        "physics_timestep",
        "runtime_digest",
    }
)

# 这些字段用于谱系和复现实验记录。它们变化并不自动意味着策略状态
# 无法加载；例如奖励、课程或监控配置变化，可能正是一次合法的 resume
# 优化。但它们仍然必须存在，并且必须写入 proven proof 的 identity。
_PROVENANCE_FIELDS = frozenset(
    {
        "task_contract_digest",
        "task_bundle_digest",
        "config_digest",
        "source_digest",
        "payload_digest",
    }
)


def _checkpoint_components(checkpoint: Mapping[str, Any] | None) -> dict[str, Any]:
    data = _as_mapping(checkpoint)
    raw_keys = data.get("keys", ())
    raw_paths = data.get("key_paths", ())
    keys = {str(item).lower().replace("\\", ".") for item in raw_keys} if isinstance(raw_keys, (list, tuple, set)) else set()
    if isinstance(raw_paths, (list, tuple, set)):
        keys.update(str(item).lower().replace("\\", ".") for item in raw_paths)
    aliases = {
        "policy": {"policy", "actor", "models.policy", "models.actor"},
        "value": {"value", "critic", "models.value", "models.critic"},
        "optimizer": {"optimizer", "optimizers"},
        "normalizer": {"normalizer", "state_preprocessor", "value_preprocessor"},
    }
    return {
        name: any(any(alias == key or alias in key for alias in values) for key in keys)
        for name, values in aliases.items()
    }


def check_resume_compatibility(
    parent: Mapping[str, Any] | Any | None,
    candidate: Mapping[str, Any] | Any,
    *,
    checkpoint: Mapping[str, Any] | None = None,
    requested: bool = True,
    strict: bool = True,
) -> ResumeCompatibility:
    """Return a proof-oriented resume decision.

    严格模式下缺少身份字段会阻断 resume。运行时 ABI 变化也会阻断；
    合同、配置、源码和 payload 摘要的变化则作为谱系差异记录，不被
    错误地等同为网络结构不兼容。
    """
    current = _fingerprint(candidate)
    current_digest = hashlib.sha256(canonical_json(current).encode("utf-8")).hexdigest()
    if not requested or parent is None:
        return ResumeCompatibility(True, "fresh", compared={"candidate": current}, fingerprint_digest=current_digest)

    previous = _fingerprint(parent)
    reasons: list[str] = []
    compared: dict[str, Any] = {}
    for key in current:
        before = previous.get(key)
        after = current.get(key)
        compared[key] = {"parent": before, "candidate": after, "equal": before == after}
        compared[key]["class"] = (
            "resume_abi" if key in _RESUME_ABI_FIELDS else "provenance"
        )
        if before in (None, "", [], {}) or after in (None, "", [], {}):
            if strict and key in _REQUIRED:
                reasons.append(f"missing compatibility field: {key}")
        elif _normal(before) != _normal(after):
            if key in _RESUME_ABI_FIELDS:
                reasons.append(f"incompatible {key}: {before!r} != {after!r}")
            elif key in _PROVENANCE_FIELDS:
                compared[key]["change"] = "recorded_provenance_change"
            else:
                reasons.append(f"incompatible {key}: {before!r} != {after!r}")

    checkpoint_info = _as_mapping(checkpoint)
    if checkpoint_info:
        if checkpoint_info.get("status") not in {"captured", "proven"}:
            reasons.append("checkpoint inventory is not proven")
        components = _checkpoint_components(checkpoint_info)
        compared["checkpoint_components"] = components
        for name, present in components.items():
            if not present:
                reasons.append(f"checkpoint missing resumable component: {name}")
    elif strict:
        reasons.append("checkpoint inventory is required for resume")

    compatible = not reasons
    return ResumeCompatibility(
        compatible=compatible,
        decision="resume" if compatible else "blocked",
        reasons=tuple(reasons),
        compared=compared,
        checkpoint=checkpoint_info,
        fingerprint_digest=current_digest,
    )


def evaluate_resume_proof(
    parent: Mapping[str, Any] | Any | None,
    candidate: Mapping[str, Any] | Any,
    *,
    checkpoint: Mapping[str, Any] | None,
    runtime_preflight: Mapping[str, Any] | None,
    restored: Mapping[str, Any] | None,
    required_state_components: tuple[str, ...] = (
        "policy",
        "value",
        "optimizer",
        "normalizer",
        "curriculum",
        "rng",
    ),
) -> ResumeProof:
    """把远程回执收敛为 proven/blocked，避免凭日志声明升级恢复状态。"""
    compatibility = check_resume_compatibility(
        parent,
        candidate,
        checkpoint=checkpoint,
        requested=True,
        strict=True,
    )
    candidate_data = _as_mapping(candidate)
    parent_data = _as_mapping(parent)
    execution = _as_mapping(candidate_data.get("execution"))
    parent_execution = _as_mapping(parent_data.get("execution"))
    preflight = _as_mapping(runtime_preflight)
    restored_values = _as_mapping(restored)
    state_components = {
        name: bool(restored_values.get(name, False))
        for name in required_state_components
    }
    checks = {
        "compatibility": compatibility.compatible,
        "checkpoint_inventory": str((_as_mapping(checkpoint)).get("status") or "")
        in {"captured", "proven"},
        "runtime_preflight": str(preflight.get("status") or "") in {"pass", "proven"},
        "remote_verification": str(
            _as_mapping(candidate_data.get("runtime_execution")).get("remote_verification") or ""
        )
        in {"proven", "pass"},
        "contract_digest_present": bool(
            execution.get("task_contract_digest")
            and parent_execution.get("task_contract_digest")
        ),
        "bundle_digest_present": bool(
            execution.get("task_bundle_digest")
            and parent_execution.get("task_bundle_digest")
        ),
        "payload_digest_present": bool(
            execution.get("payload_digest")
            and parent_execution.get("payload_digest")
        ),
        "runtime_digest_present": bool(
            execution.get("runtime_digest")
            and parent_execution.get("runtime_digest")
        ),
    }
    reasons = list(compatibility.reasons)
    reasons.extend(
        f"resume proof check failed: {name}"
        for name, passed in checks.items()
        if not passed and name != "compatibility"
    )
    reasons.extend(
        f"resume state component was not restored: {name}"
        for name, passed in state_components.items()
        if not passed
    )
    proven = not reasons
    identity = {
        "parent_task_contract_ref": parent_execution.get("task_contract_ref", ""),
        "child_task_contract_ref": execution.get("task_contract_ref", ""),
        "parent_task_contract_digest": parent_execution.get("task_contract_digest", ""),
        "child_task_contract_digest": execution.get("task_contract_digest", ""),
        "parent_bundle_digest": parent_execution.get("task_bundle_digest", ""),
        "child_bundle_digest": execution.get("task_bundle_digest", ""),
        "parent_payload_digest": parent_execution.get("payload_digest", ""),
        "child_payload_digest": execution.get("payload_digest", ""),
        "parent_runtime_digest": parent_execution.get("runtime_digest", ""),
        "child_runtime_digest": execution.get("runtime_digest", ""),
    }
    return ResumeProof(
        proven=proven,
        status="proven" if proven else "blocked",
        reasons=tuple(dict.fromkeys(reasons)),
        compatibility=compatibility,
        checks=checks,
        state_components=state_components,
        identity=identity,
    )


__all__ = [
    "COMPATIBILITY_SCHEMA",
    "ResumeCompatibility",
    "ResumeProof",
    "check_resume_compatibility",
    "evaluate_resume_proof",
]
