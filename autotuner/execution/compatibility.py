"""Strict, explainable resume compatibility checks."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping

try:
    from .hashing import canonical_json
except ImportError:  # payload copy keeps the helper beside this module
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
        "runtime_digest": runtime.get("digest") or data.get("runtime_digest"),
        "config_digest": _first(contract, ("config_digest",)) or _first(configuration, ("effective_config", "sha256")),
        "source_digest": _first(contract, ("source_digest",)),
        # A contract digest is not a payload digest.  Treating it as a
        # fallback made a source-only change look resume-compatible with an
        # otherwise different packaged runtime.
        "payload_digest": (
            payload.get("payload_digest")
            or payload.get("digest")
            or data.get("payload_digest")
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


_REQUIRED = (
    "product",
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

    Missing identity fields are blocking in strict mode.  This prevents a
    checkpoint from being resumed merely because an actor weight file exists.
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
        if before in (None, "", [], {}) or after in (None, "", [], {}):
            if strict and key in _REQUIRED:
                reasons.append(f"missing compatibility field: {key}")
        elif _normal(before) != _normal(after):
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


__all__ = ["COMPATIBILITY_SCHEMA", "ResumeCompatibility", "check_resume_compatibility"]
