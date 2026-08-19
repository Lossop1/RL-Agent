"""Runtime evidence for the payload-owned Taili training entry point.

The launcher already records paths and effective configuration.  This module
records the facts that are only available after the real runtime has imported
the payload and constructed the skrl agent: resolved module paths, component
state digests, resume provenance, RNG state, and checkpoint inventory.

It is intentionally dependency-light.  IsaacLab/skrl objects are inspected
through small duck-typed helpers so the module can be imported by the payload
without adding a second training abstraction or changing reward semantics.
"""
from __future__ import annotations

import base64
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Mapping

try:
    # Prefer the copy shipped beside the payload.  An unrelated remote
    # ``autotuner`` checkout must not shadow the payload's runtime identity.
    from .runtime_identity import capture_runtime_identity  # type: ignore
except ImportError:  # source-tree compatibility: the helper lives in execution/
    from autotuner.execution.runtime import capture_runtime_identity


SCHEMA_VERSION = "rl-agent.runtime-manifest/v1"
REQUIRED_OPTIMIZATION_FIELDS = (
    "policy",
    "value",
    "optimizer",
    "scheduler",
    "normalizer",
    "log_std",
    "amp",
    "curriculum",
    "rng",
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _relative_or_absolute(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_digest(path: str | os.PathLike[str], *, root: Path | None = None) -> dict[str, Any]:
    candidate = Path(path)
    item: dict[str, Any] = {"path": str(candidate), "exists": False}
    try:
        stat = candidate.stat()
    except OSError:
        return item
    item.update(
        {
            "path": _relative_or_absolute(root, candidate) if root is not None else str(candidate.resolve()),
            "exists": candidate.is_file(),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    )
    if candidate.is_file():
        item["sha256"] = sha256_file(candidate)
    return item


def load_product_contract(payload_root: str | os.PathLike[str]) -> dict[str, Any]:
    """读取 payload 构建时生成的产品合同，不依赖仓库源码。"""
    root = Path(payload_root)
    candidates = (
        root / "taili_blind_runtime" / "product_contract.json",
        root / "product_contract.json",
    )
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError):
            continue
        if isinstance(data, Mapping):
            return dict(data)
    return {}


def load_payload_manifest(payload_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Read the content-addressed payload manifest without importing system code."""
    root = Path(payload_root)
    candidates = (root / "payload_manifest.json", root / "taili_blind_runtime" / "payload_manifest.json")
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError):
            continue
        if isinstance(data, Mapping):
            return dict(data)
    return {}


def _json_safe(value: Any, *, depth: int = 0, limit: int = 32) -> Any:
    """Keep metadata bounded and deterministic; never serialize whole tensors."""
    if depth > 4:
        return f"<{type(value).__name__}>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda pair: str(pair[0]))[:limit]
        return {str(key): _json_safe(item, depth=depth + 1, limit=limit) for key, item in items}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, depth=depth + 1, limit=limit) for item in list(value)[:limit]]
    return f"<{type(value).__module__}.{type(value).__name__}>"


def _hash_value(value: Any, digest: hashlib._Hash | None = None) -> str:
    """Hash nested state dictionaries without depending on torch or numpy."""
    digest = digest or hashlib.sha256()
    if value is None:
        digest.update(b"null")
    elif isinstance(value, Mapping):
        digest.update(b"map{")
        for key in sorted(value, key=lambda item: str(item)):
            digest.update(str(key).encode("utf-8", "replace"))
            digest.update(b":")
            _hash_value(value[key], digest)
            digest.update(b";")
        digest.update(b"}")
    elif isinstance(value, (list, tuple)):
        digest.update(b"list[")
        for item in value:
            _hash_value(item, digest)
            digest.update(b",")
        digest.update(b"]")
    elif isinstance(value, (str, int, float, bool)):
        digest.update(f"{type(value).__name__}:{value}".encode("utf-8", "replace"))
    elif hasattr(value, "detach") and hasattr(value, "shape"):
        # torch.Tensor-like object.  The import is intentionally avoided.
        try:
            tensor = value.detach().cpu().contiguous()
            digest.update(f"tensor:{tuple(tensor.shape)}:{tensor.dtype}".encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        except Exception:
            digest.update(repr(value).encode("utf-8", "replace"))
    elif hasattr(value, "tobytes") and hasattr(value, "shape"):
        try:
            digest.update(f"array:{tuple(value.shape)}:{getattr(value, 'dtype', '')}".encode("utf-8"))
            digest.update(value.tobytes())
        except Exception:
            digest.update(repr(value).encode("utf-8", "replace"))
    else:
        digest.update(repr(value).encode("utf-8", "replace"))
    return digest.hexdigest()


def _state_dict(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return obj
    method = getattr(obj, "state_dict", None)
    if callable(method):
        return method()
    return None


def _state_ref(label: str, obj: Any, *, source: str = "") -> dict[str, Any]:
    if obj is None:
        return {"status": "missing", "label": label, "source": source, "reason": "object_not_exposed"}
    try:
        state = _state_dict(obj)
        if state is None:
            return {"status": "missing", "label": label, "source": source, "reason": "state_dict_unavailable"}
        keys = len(state) if isinstance(state, Mapping) else 1
        return {
            "status": "captured",
            "label": label,
            "source": source,
            "sha256": _hash_value(state),
            "key_count": int(keys),
        }
    except Exception as exc:  # pragma: no cover - runtime-library dependent
        return {
            "status": "blocked",
            "label": label,
            "source": source,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _mapping_candidates(obj: Any) -> list[tuple[str, Any]]:
    result: list[tuple[str, Any]] = []
    for attr in ("models", "checkpoint_modules", "optimizers", "schedulers", "preprocessors"):
        value = getattr(obj, attr, None)
        if isinstance(value, Mapping):
            result.extend((f"{attr}.{key}", item) for key, item in value.items())
    return result


def _find_component(obj: Any, names: Iterable[str]) -> tuple[str, Any] | tuple[str, None]:
    wanted = {str(name).lower() for name in names}
    for attr in ("policy", "value", "optimizer", "scheduler", "learning_rate_scheduler", "state_preprocessor", "value_preprocessor", "normalizer", "amp", "discriminator"):
        value = getattr(obj, attr, None)
        if value is not None and attr.lower() in wanted:
            return attr, value
    for source, value in _mapping_candidates(obj):
        key = source.rsplit(".", 1)[-1].lower()
        if key in wanted:
            return source, value
    return "", None


def _find_model_parameter(model: Any, needle: str) -> tuple[str, Any] | tuple[str, None]:
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        return "", None
    try:
        for name, value in named_parameters():
            if needle in str(name).lower():
                return str(name), value
    except Exception:
        return "", None
    return "", None


def _env_object(env: Any) -> Any:
    current = env
    for _ in range(6):
        next_value = getattr(current, "unwrapped", None) or getattr(current, "env", None)
        if next_value is None or next_value is current:
            break
        current = next_value
    return current


def _curriculum_ref(env: Any, run_dir: Path) -> dict[str, Any]:
    sidecar = run_dir / "curriculum_state.json"
    if sidecar.is_file():
        return {"status": "captured", "source": str(sidecar), **file_digest(sidecar)}
    target = _env_object(env)
    values = {}
    for name in ("_phase", "_phase_count", "_terrain_level_peak", "_dr_level", "current_training_phase"):
        value = getattr(target, name, None)
        if value is not None:
            values[name] = _json_safe(value)
    if values:
        return {
            "status": "fresh_initialization",
            "source": "runtime_environment_attributes",
            "values": values,
            "sha256": _hash_value(values),
        }
    return {"status": "missing", "source": "", "reason": "curriculum_state_not_exposed"}


def _rng_ref(torch_module: Any = None) -> dict[str, Any]:
    states: dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np  # type: ignore

        states["numpy"] = np.random.get_state()
    except Exception:
        states["numpy"] = "not_available"
    if torch_module is not None:
        try:
            states["torch"] = torch_module.get_rng_state()
        except Exception:
            states["torch"] = "unavailable"
        try:
            states["torch_cuda"] = torch_module.cuda.get_rng_state_all() if torch_module.cuda.is_available() else "not_available"
        except Exception:
            states["torch_cuda"] = "unavailable"
    return {"status": "captured", "source": "process_rng", "sha256": _hash_value(states)}


def capture_optimization_state(
    agent: Any,
    env: Any,
    *,
    run_dir: str | os.PathLike[str],
    torch_module: Any = None,
    amp_expected: bool = True,
    parity_check: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Capture component hashes without mutating the agent or environment."""
    run_path = Path(run_dir)
    policy_source, policy = _find_component(agent, ("policy", "actor"))
    value_source, value = _find_component(agent, ("value", "critic"))
    optimizer_source, optimizer = _find_component(agent, ("optimizer",))
    scheduler_source, scheduler = _find_component(agent, ("scheduler", "learning_rate_scheduler"))
    normalizer_source, normalizer = _find_component(
        agent, ("state_preprocessor", "value_preprocessor", "normalizer")
    )
    amp_source, amp = _find_component(agent, ("amp", "discriminator"))

    log_std_name, log_std = _find_model_parameter(policy, "log_std") if policy is not None else ("", None)
    fields = {
        "policy": _state_ref("policy", policy, source=policy_source),
        "value": _state_ref("value", value, source=value_source),
        "optimizer": _state_ref("optimizer", optimizer, source=optimizer_source),
        "scheduler": (
            _state_ref("scheduler", scheduler, source=scheduler_source)
            if scheduler is not None
            else {"status": "not_configured", "source": "agent"}
        ),
        "normalizer": _state_ref("normalizer", normalizer, source=normalizer_source),
        "log_std": (
            _state_ref("log_std", {log_std_name: log_std}, source=policy_source)
            if log_std is not None
            else {"status": "not_configured", "source": "policy.named_parameters"}
        ),
        "amp": (
            _state_ref("amp", amp, source=amp_source)
            if amp is not None
            else {"status": "missing" if amp_expected else "not_configured", "source": "agent"}
        ),
        "curriculum": _curriculum_ref(env, run_path),
        "rng": _rng_ref(torch_module),
    }
    usable = all(item.get("status") in {"captured", "fresh_initialization", "not_configured"} for item in fields.values())
    return {
        "status": "proven" if usable and parity_check else "declared" if usable else "missing",
        "fields": fields,
        "parity_check": bool(parity_check),
        "resume": bool(resume),
        "source": "runtime_manifest.capture_optimization_state",
    }


def checkpoint_inventory(path: str | os.PathLike[str], *, torch_module: Any = None) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file():
        return {"status": "missing", "path": str(candidate), "keys": []}
    if torch_module is None:
        return {"status": "declared", "path": str(candidate), **file_digest(candidate), "keys": []}
    try:
        payload = torch_module.load(str(candidate), map_location="cpu", weights_only=False)
        keys = sorted(str(key) for key in payload) if isinstance(payload, Mapping) else []
        key_paths: list[str] = []

        def collect(value: Any, prefix: str = "", depth: int = 0) -> None:
            if depth > 3 or not isinstance(value, Mapping):
                return
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                key_paths.append(path)
                collect(child, path, depth + 1)

        collect(payload)
        return {
            "status": "captured",
            "path": str(candidate),
            **file_digest(candidate),
            "keys": keys,
            "key_paths": sorted(set(key_paths)),
        }
    except Exception as exc:  # pragma: no cover - depends on checkpoint version
        return {
            "status": "blocked",
            "path": str(candidate),
            **file_digest(candidate),
            "keys": [],
            "key_paths": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def capture_runtime_execution(
    payload_root: str | os.PathLike[str],
    expected_modules: Iterable[str],
    *,
    source_paths: Iterable[str | os.PathLike[str]] = (),
) -> dict[str, Any]:
    """Record actual module resolution, including the paths used by imports."""
    root = Path(payload_root).resolve()
    resolved: list[dict[str, Any]] = []
    missing: list[str] = []
    suspicious: list[str] = []
    for name in dict.fromkeys(str(item) for item in expected_modules):
        module = sys.modules.get(name)
        if module is None:
            try:
                module = importlib.import_module(name)
            except Exception:
                missing.append(name)
                continue
        module_file = getattr(module, "__file__", None)
        if not module_file:
            missing.append(name)
            continue
        path = Path(module_file).resolve()
        inside = _inside(root, path)
        if not inside:
            suspicious.append(f"{name}:{path}")
        resolved.append({"module": name, **file_digest(path)})

    for item in source_paths:
        path = Path(item)
        if not path.is_absolute():
            path = root / path
        resolved.append({"module": "source_file", **file_digest(path, root=root)})
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for item in resolved:
        unique[(str(item.get("module")), str(item.get("path")))] = item
    status = "proven" if not missing and not suspicious else "blocked"
    return {
        "status": status,
        "source_files": list(unique.values()),
        "symbols": {name: name not in missing for name in expected_modules},
        "duplicate_modules": {},
        "missing_files": missing,
        "remote_verification": "proven" if status == "proven" else "declared",
        "resolved_imports": list(unique.values()),
        "sys_path": [str(item) for item in sys.path],
        "shadowing_checked": not suspicious,
        "suspicious_imports": suspicious,
        "notes": [
            "resolved module paths were captured inside the payload process",
            "runtime source proof is separate from local strategy_backups scanning",
        ],
    }


def initial_manifest(
    *,
    run_id: str,
    run_dir: str | os.PathLike[str],
    task: str,
    payload_root: str | os.PathLike[str],
    source_config: str | os.PathLike[str],
    effective_config: str | os.PathLike[str],
    agent_config: str | os.PathLike[str],
    resume_checkpoint: str = "",
    seed: int | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    telemetry_path = os.environ.get("TAILI_TELEMETRY_JSONL") or str(run_path / "train.telemetry.jsonl")
    checkpoint_path = os.environ.get("TAILI_CHECKPOINT_DIR") or str(run_path / "checkpoints")
    product_contract = load_product_contract(payload_root)
    payload_manifest = load_payload_manifest(payload_root)
    product_data = {
        "id": str(product_contract.get("product_id") or ""),
        "version": str(product_contract.get("product_version") or ""),
        "digest": str(product_contract.get("contract_digest") or ""),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "program_id": "taili_blind_locomotion",
        "product": product_data,
        "status": "running",
        "run": {
            "run_id": str(run_id),
            "run_dir": str(run_path),
            "task": str(task),
            "resume_checkpoint": str(resume_checkpoint or ""),
            "seed": seed,
            "pid": os.getpid(),
            "python": sys.executable,
            "python_version": sys.version,
            "cwd": os.getcwd(),
        },
        "paths": {
            "payload_root": str(Path(payload_root).resolve()),
            "source_config": str(Path(source_config).resolve()),
            "effective_config": str(Path(effective_config).resolve()),
            "agent_config": str(Path(agent_config).resolve()),
            "telemetry_jsonl": telemetry_path,
            "checkpoint_dir": checkpoint_path,
            "runtime_manifest": str(run_path / "runtime_manifest.json"),
        },
        "configuration": {
            "source_config": file_digest(source_config),
            "effective_config": file_digest(effective_config),
            "agent_config": file_digest(agent_config),
        },
        "resolved_contract": {
            "status": "captured" if product_contract else "missing",
            "path": str(Path(payload_root) / "taili_blind_runtime" / "product_contract.json"),
            "digest": str(product_contract.get("contract_digest") or ""),
            "contract_digest": str(product_contract.get("contract_digest") or ""),
            "config_digest": str(product_contract.get("config_digest") or ""),
            "asset_digest": str(product_contract.get("asset_digest") or ""),
            "source_digest": str(product_contract.get("source_digest") or ""),
            "payload_digest": str(payload_manifest.get("payload_digest") or ""),
        },
        "payload": {
            "status": "captured" if payload_manifest else "missing",
            "path": str(Path(payload_root) / "payload_manifest.json"),
            "digest": str(payload_manifest.get("payload_digest") or ""),
            "runtime_digest": str(payload_manifest.get("runtime_digest") or ""),
        },
        "runtime": product_contract.get("runtime", {}),
        "compatibility": product_contract.get("compatibility", {}),
        "runtime_evidence": capture_runtime_identity(product_contract.get("runtime", {})),
        "policy_contract_ref": os.environ.get("TAILI_POLICY_CONTRACT_REF", ""),
        "distribution_manifest_ref": os.environ.get("TAILI_DISTRIBUTION_MANIFEST_REF", ""),
        "contract_bundle_ref": os.environ.get("TAILI_CONTRACT_BUNDLE_REF", ""),
        "environment": {},
        "assets": {},
        "runtime_execution": {"status": "declared", "remote_verification": "declared"},
        "optimization_state": {
            "status": "missing",
            "fields": {},
            "missing_fields": list(REQUIRED_OPTIMIZATION_FIELDS),
            "parity_check": False,
            "source": "runtime_not_initialized",
        },
        "command_coverage": [],
        "coverage_window": "",
        "command_coverage_source": "",
        "metric_alignment": [],
        "gate_calibration": [],
        "checkpoint_capabilities": [],
        "evidence": {"telemetry": [], "diagnostics": [], "scorecards": []},
        "resume_edge": {
            "parent_checkpoint_ref": str(resume_checkpoint or ""),
            "restored": {},
            "status": "fresh" if not resume_checkpoint else "unverified",
        },
        "errors": [],
    }


def update_manifest(path: str | os.PathLike[str], **updates: Any) -> dict[str, Any]:
    target = Path(path)
    try:
        current = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else {}
    except (OSError, ValueError, UnicodeError):
        current = {}
    # Runtime stages update one section at a time. Preserve launch facts such
    # as run_id/task when a later stage adds seed or package metadata to `run`.
    for key, value in updates.items():
        previous = current.get(key)
        if isinstance(previous, dict) and isinstance(value, Mapping):
            merged = dict(previous)
            merged.update(value)
            current[key] = merged
        else:
            current[key] = value
    current["updated_at"] = utc_now()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(target)
    return current


def write_manifest(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(target)


__all__ = [
    "SCHEMA_VERSION",
    "REQUIRED_OPTIMIZATION_FIELDS",
    "capture_optimization_state",
    "capture_runtime_execution",
    "checkpoint_inventory",
    "file_digest",
    "initial_manifest",
    "load_product_contract",
    "load_payload_manifest",
    "sha256_file",
    "update_manifest",
    "utc_now",
    "write_manifest",
]
