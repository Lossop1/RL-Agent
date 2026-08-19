"""Convert persisted run/evaluation artifacts into a research-audit manifest.

This is an offline evidence assembler.  It never launches IsaacLab, runs a
diagnostic, or turns a missing scorecard into a pass.  Telemetry can establish
command sampling facts; physical metric alignment requires an actual scorer
artifact; gate calibration requires an explicit reference-policy/effective
sample record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from autotuner.product import resolve_product_runtime

import yaml

from autotuner.research.gate_calibration import (
    GateDefinition,
    extract_gate_definitions,
    gate_trace_rows,
    quantiles,
)
from .research_audit import _DEFAULT_ALIGNMENT_SPECS


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _command_bucket(command: dict[str, Any]) -> str:
    vx = _number(command.get("cmd_vx")) or 0.0
    vy = _number(command.get("cmd_vy")) or 0.0
    wz = _number(command.get("cmd_wz")) or 0.0
    active = [abs(vx) > 0.05, abs(vy) > 0.05, abs(wz) > 0.05]
    if not any(active):
        return "stand"
    if sum(active) > 1:
        return "mixed"
    if abs(wz) > 0.05:
        return "yaw"
    if abs(vx) >= abs(vy):
        return "forward" if vx >= 0.0 else "backward"
    return "lateral"


def command_coverage_from_telemetry(rows: Iterable[dict[str, Any]], *, source: str) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = {}
    known_buckets = ("forward", "backward", "lateral", "yaw", "stand", "mixed")
    first_step: int | None = None
    last_step: int | None = None

    def ensure(bucket: str) -> dict[str, Any]:
        return buckets.setdefault(
            bucket,
            {
                "bucket": bucket,
                "target_count": 0,
                "applied_count": 0,
                "eligible_count": 0,
                "settled_count": 0,
                "terminal_count": 0,
                "timeout_count": 0,
                "sample_count": 0,
            },
        )

    for item in rows:
        command = item.get("command") if isinstance(item.get("command"), dict) else {}
        health = item.get("health") if isinstance(item.get("health"), dict) else {}
        stable = _number(health.get("stable_motion_gate"))
        terminal = _number(health.get("terminal_rate"))
        timeout = int(str(item.get("event") or "").lower() == "timeout")
        aggregate_present = any(
            f"bucket_{bucket}_target_samples" in command for bucket in known_buckets
        )
        if aggregate_present:
            # Sum counters over the actual environment batch. A mean command
            # vector cannot identify stand or mixed samples.
            for bucket in known_buckets:
                target = _number(command.get(f"bucket_{bucket}_target_samples"))
                applied = _number(command.get(f"bucket_{bucket}_applied_samples"))
                eligible = _number(command.get(f"bucket_{bucket}_eligible_samples"))
                if target is None and applied is None and eligible is None:
                    continue
                target_n = max(0.0, target or 0.0)
                applied_n = max(0.0, applied or 0.0)
                eligible_n = max(0.0, eligible if eligible is not None else applied_n)
                row = ensure(bucket)
                row["target_count"] += int(round(target_n))
                row["applied_count"] += int(round(applied_n))
                row["eligible_count"] += int(round(eligible_n))
                row["sample_count"] += int(round(max(target_n, applied_n)))
                settled = _number(command.get(f"bucket_{bucket}_settled_samples"))
                if settled is not None:
                    row["settled_count"] += int(round(max(0.0, settled)))
            # Global stability, terminal and timeout values are row-level in
            # the current payload. Do not distribute them to every bucket.
        elif {"cmd_vx", "cmd_vy", "cmd_wz"} <= set(command):
            bucket = _command_bucket(command)
            row = ensure(bucket)
            row["target_count"] += 1
            row["sample_count"] += 1
            row["applied_count"] += int(all(key in command for key in ("actual_vx", "actual_vy", "actual_wz")))
            row["eligible_count"] += 1
            row["settled_count"] += int(stable is not None and stable >= 0.5)
            row["terminal_count"] += int(terminal is not None and terminal > 0.5)
            row["timeout_count"] += timeout
        else:
            continue
        step = item.get("step")
        if isinstance(step, (int, float)):
            first_step = int(step) if first_step is None else min(first_step, int(step))
            last_step = int(step) if last_step is None else max(last_step, int(step))
    return {
        "coverage_window": f"steps:{first_step}-{last_step}" if first_step is not None else "",
        "command_coverage_source": source,
        "command_coverage": [buckets[key] for key in sorted(buckets)],
    }


def _terrain_bucket(item: dict[str, Any]) -> str:
    curriculum = item.get("curriculum") if isinstance(item.get("curriculum"), dict) else {}
    for key in ("terrain_type", "terrain", "terrain_name", "scene"):
        value = curriculum.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("terrain_level", "terrain_mean", "stairs_level", "stair_level"):
        value = _number(curriculum.get(key))
        if value is not None:
            return f"level:{value:g}"
    return "unknown"


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def pick(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
        return float(ordered[index])

    return {"min": float(ordered[0]), "p50": pick(0.5), "p90": pick(0.9), "max": float(ordered[-1])}


def distribution_manifest_from_telemetry(
    rows: Iterable[dict[str, Any]],
    *,
    source: str,
    target: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Separate configured targets from realized command/terrain/DR samples."""
    items = list(rows)
    command_rows = command_coverage_from_telemetry(items, source=source)
    terrain_counts: dict[str, int] = {}
    dr_values: dict[str, list[float]] = {}
    terminal_count = 0
    timeout_count = 0
    for item in items:
        terrain = _terrain_bucket(item)
        terrain_counts[terrain] = terrain_counts.get(terrain, 0) + 1
        curriculum = item.get("curriculum") if isinstance(item.get("curriculum"), dict) else {}
        health = item.get("health") if isinstance(item.get("health"), dict) else {}
        for mapping in (curriculum, health):
            for key, value in mapping.items():
                if not (str(key).startswith("dr_") or str(key) in {"mass_delta_kg", "friction", "push_velocity_m_s", "action_delay_policy_steps"}):
                    continue
                number = _number(value)
                if number is not None:
                    dr_values.setdefault(str(key), []).append(number)
        terminal_count += int((_number(health.get("terminal_rate")) or 0.0) > 0.5)
        timeout_count += int(str(item.get("event") or "").lower() == "timeout")
    final_curriculum = items[-1].get("curriculum") if items and isinstance(items[-1].get("curriculum"), dict) else {}
    return {
        "schema_version": "rl-agent.training-distribution/v1",
        "distribution_source": source,
        "command_buckets": {
            "target": target.get("command_buckets", {}) if isinstance(target, dict) else {},
            "realized": {row["bucket"]: row for row in command_rows["command_coverage"]},
        },
        "terrain_mix": {
            "target": target.get("terrain_mix", {}) if isinstance(target, dict) else {},
            "realized_counts": terrain_counts,
        },
        "curriculum_state": {
            "target": target.get("curriculum_state", {}) if isinstance(target, dict) else {},
            "last_observed": final_curriculum,
        },
        "dr_channels": {
            "target": target.get("dr_channels", {}) if isinstance(target, dict) else {},
            "realized_quantiles": {key: _quantiles(values) for key, values in sorted(dr_values.items())},
        },
        "reset_sampling": {
            "terminal_samples": terminal_count,
            "timeout_samples": timeout_count,
            "sample_count": len(items),
        },
        "realized_window": {
            "steps": command_rows["coverage_window"],
            "sample_count": len(items),
            "source": source,
        },
    }


def _scorecard_keys(
    paths: Iterable[Path],
    *,
    product_id: str | None = None,
    contract: Any = None,
) -> tuple[set[str], list[str]]:
    from autotuner.product import load_product_plugin, resolve_product_contract

    product = contract or resolve_product_contract(product_id)
    parse_scorecard = load_product_plugin(product, "acceptance", "parse_scorecard")
    keys: set[str] = set()
    evidence: list[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # 证据汇编与控制台必须使用同一产品解析器，避免同一日志产生两种语义。
        parsed = parse_scorecard(text)
        if parsed:
            keys.update(parsed)
            evidence.append(str(path))
    return keys, evidence


def _alignment_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("id")): item
        for item in _load_list(path, "metric_alignment")
        if item.get("id")
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_GENERIC_CONFIG_ARTIFACT_NAMES = {
    "source_config": "source_config.yaml",
    "effective_config": "effective_config.yaml",
    "agent_config": "agent.skrl.yaml",
}


def _file_ref(path: Path) -> dict[str, Any]:
    item: dict[str, Any] = {"path": str(path.resolve()), "exists": path.is_file()}
    if not path.is_file():
        return item
    stat = path.stat()
    item.update({
        "sha256": _sha256(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    })
    return item


def _configuration_artifact_names(product_id: str | None = None) -> dict[str, str]:
    """读取产品合同声明的运行配置文件名；没有产品时只使用通用命名。"""
    if not product_id:
        return dict(_GENERIC_CONFIG_ARTIFACT_NAMES)
    try:
        deployment = resolve_product_runtime(product_id).deployment
        declared = deployment.get("configuration_artifacts", {})
    except Exception:
        return {}
    if not isinstance(declared, dict):
        return {}
    result = {
        str(key): str(value).strip().lstrip("/")
        for key, value in declared.items()
        if str(key).strip() and isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", value.strip())
    }
    return result


def configuration_artifact_parity(
    run_path: Path,
    manifest: dict[str, Any],
    *,
    product_id: str | None = None,
) -> dict[str, Any]:
    """Pair downloaded configuration files with the runtime-recorded digests."""
    remote_config = manifest.get("configuration") if isinstance(manifest.get("configuration"), dict) else {}
    result: dict[str, Any] = {}
    for key, filename in _configuration_artifact_names(product_id).items():
        remote = remote_config.get(key) if isinstance(remote_config.get(key), dict) else {}
        local = _file_ref(run_path / filename)
        remote_hash = str(remote.get("sha256") or "")
        local_hash = str(local.get("sha256") or "")
        match = bool(remote_hash and local_hash and remote_hash == local_hash)
        result[key] = {
            "status": "proven" if match else "missing" if not local.get("exists") else "blocked",
            "remote": remote,
            "local": local,
            "sha256_match": match,
        }
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_gate_calibration_evidence(
    run_dir: str | Path,
    *,
    reference_policy_manifest: str | Path,
    scenario_contract: str | Path,
    effective_config: str | Path | None = None,
    telemetry: str | Path | None = None,
    output: str | Path | None = None,
    minimum_samples: int = 3,
) -> Path:
    """Build a reproducible gate sidecar from frozen-policy rollout telemetry.

    This reducer does not decide that a policy is capable.  It writes raw
    samples and a qualification mask, then summarizes only gates that the
    effective configuration actually contains.  ``research_audit`` reads the
    artifacts again and independently recomputes every claimed count/range.
    """
    run_path = Path(run_dir).resolve()
    config_path = Path(effective_config).resolve() if effective_config else run_path / "effective_config.yaml"
    telemetry_path = Path(telemetry).resolve() if telemetry else run_path / "train.telemetry.jsonl"
    reference_path = Path(reference_policy_manifest).resolve()
    scenario_path = Path(scenario_contract).resolve()
    output_path = Path(output).resolve() if output else run_path / "gate_calibration.json"
    raw_path = output_path.with_name("gate_calibration.raw.jsonl")
    mask_path = output_path.with_name("gate_calibration.mask.json")
    minimum = max(1, int(minimum_samples))

    config = _read_yaml(config_path)
    definitions = extract_gate_definitions(config)
    telemetry_rows = _read_jsonl(telemetry_path)
    trace = gate_trace_rows(telemetry_rows, definitions)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_tmp = raw_path.with_suffix(raw_path.suffix + ".tmp")
    raw_tmp.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in trace),
        encoding="utf-8",
    )
    raw_tmp.replace(raw_path)

    eligible_by_gate: dict[str, list[str]] = {}
    for item in trace:
        if item.get("eligible"):
            eligible_by_gate.setdefault(str(item.get("gate_id") or ""), []).append(str(item.get("sample_id") or ""))
    mask_payload = {
        "schema_version": "rl-agent.gate-qualification-mask/v1",
        "rules": {item.gate_id: item.qualification_rule for item in definitions},
        "eligible_sample_ids": eligible_by_gate,
    }
    _write_json(mask_path, mask_payload)

    artifacts = {
        "effective_config": config_path,
        "telemetry": telemetry_path,
        "raw_trace": raw_path,
        "qualification_mask": mask_path,
        "reference_policy": reference_path,
        "scenario_contract": scenario_path,
    }
    hashes = {
        name: _sha256(path)
        for name, path in artifacts.items()
        if path.is_file()
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in trace:
        grouped.setdefault(str(item.get("gate_id") or ""), []).append(item)

    rows: list[dict[str, Any]] = []
    for definition in definitions:
        samples = [item for item in grouped.get(definition.gate_id, []) if item.get("eligible")]
        values = [float(item["value"]) for item in samples]
        observed = quantiles(values)
        pass_count = sum(bool(item.get("passed")) for item in samples)
        gaps: list[str] = []
        if definition.active:
            if len(samples) < minimum:
                gaps.append(f"qualified samples {len(samples)} < {minimum}")
            if pass_count <= 0:
                gaps.append("reference policy never reached the configured threshold")
            for name, path in artifacts.items():
                if not path.is_file():
                    gaps.append(f"missing artifact: {name}")
        ceiling = None
        if values:
            ceiling = max(values) if definition.comparison == "gte" else min(values)
        rows.append({
            **definition.model_dump(),
            "code_default": None,
            "reference_policy_ref": str(reference_path),
            "distribution_ref": str(telemetry_path),
            "effective_config_ref": str(config_path),
            "raw_trace_ref": str(raw_path),
            "qualification_mask_ref": str(mask_path),
            "scenario_contract_ref": str(scenario_path),
            "sample_count": len(samples),
            "eligible_count": len(samples),
            "pass_count": pass_count,
            "pass_rate": pass_count / len(samples) if samples else 0.0,
            "minimum_samples": minimum,
            "observed_range": observed,
            "reachable_interval": (
                {"lower": observed["min"], "upper": observed["max"]}
                if observed else {}
            ),
            "observed_ceiling": ceiling,
            "aggregation": {
                "level": "distinct_gate_evaluation_window",
                "method": "empirical_min_p05_p50_p95_max",
                "duplicates": "deduplicated_by_gate_id_and_gate_eval_step",
            },
            "source_hashes": hashes,
            "status": (
                "declared" if not definition.active
                else "proven" if not gaps
                else "captured" if samples
                else "missing"
            ),
            "evidence": [str(path) for path in artifacts.values() if path.is_file()],
            "gaps": gaps,
        })

    _write_json(output_path, {
        "schema_version": "rl-agent.gate-calibration-evidence/v1",
        "generated_at": time_iso(),
        "effective_config_ref": str(config_path),
        "effective_config_sha256": hashes.get("effective_config", ""),
        "reference_policy_ref": str(reference_path),
        "scenario_contract_ref": str(scenario_path),
        "gate_calibration": rows,
    })
    return output_path


def _resolve_artifact(reference: Any, *, sidecar: Path) -> Path | None:
    if not isinstance(reference, str) or not reference.strip():
        return None
    candidate = Path(reference)
    return candidate.resolve() if candidate.is_absolute() else (sidecar.parent / candidate).resolve()


def metric_alignment_from_scorecards(
    scorecard_paths: Iterable[Path],
    *,
    telemetry_source: str,
    alignment_evidence: str | Path | None = None,
    product_id: str | None = None,
    contract: Any = None,
) -> list[dict[str, Any]]:
    keys, scorecard_evidence = _scorecard_keys(
        scorecard_paths,
        product_id=product_id,
        contract=contract,
    )
    sidecar = Path(alignment_evidence).resolve() if alignment_evidence else None
    sidecar_rows = _alignment_rows(sidecar)
    out: list[dict[str, Any]] = []
    for spec in _DEFAULT_ALIGNMENT_SPECS:
        identifier = str(spec["id"])
        if identifier == "D_STAIRS":
            present = any(key.startswith("D[stairs") for key in keys)
        else:
            # Scorecards may qualify a gate with a scenario, e.g. A2[yaw]
            # or B4[flat].  Alignment is by gate family, not one exact row.
            present = any(key == identifier or key.startswith(identifier + "[") for key in keys)
        evidence = [telemetry_source] if telemetry_source else []
        evidence.extend(scorecard_evidence)
        row = sidecar_rows.get(identifier, {})
        raw_trace = _resolve_artifact(row.get("raw_trace_ref"), sidecar=sidecar) if sidecar else None
        mask = _resolve_artifact(row.get("qualification_mask_ref"), sidecar=sidecar) if sidecar else None
        scenario = _resolve_artifact(row.get("scenario_contract_ref"), sidecar=sidecar) if sidecar else None
        sample_count = int(_number(row.get("sample_count")) or 0)
        aggregation = row.get("aggregation") if isinstance(row.get("aggregation"), dict) else {}
        threshold = row.get("threshold") if isinstance(row.get("threshold"), dict) else {}
        declared_hashes = row.get("source_hashes") if isinstance(row.get("source_hashes"), dict) else {}
        artifacts = {"raw_trace": raw_trace, "qualification_mask": mask, "scenario_contract": scenario}
        actual_hashes = {
            name: _sha256(path)
            for name, path in artifacts.items()
            if path is not None and path.is_file()
        }
        hash_ok = bool(actual_hashes) and all(
            declared_hashes.get(name) == digest for name, digest in actual_hashes.items()
        ) and set(actual_hashes) == set(artifacts)
        semantic_ok = bool(
            row.get("unit")
            and threshold
            and row.get("train_signals") == spec["train_signals"]
            and row.get("eval_signals") == spec["eval_signals"]
            and all(row.get(name) == spec[name] for name in ("frame", "event_window", "eligibility", "statistic"))
        )
        aligned = bool(
            present
            and sidecar
            and row
            and sample_count > 0
            and aggregation
            and hash_ok
            and semantic_ok
        )
        if sidecar and row:
            evidence.append(str(sidecar))
            evidence.extend(str(path) for path in artifacts.values() if path is not None and path.is_file())
        gaps: list[str] = []
        if not present:
            gaps.append("missing scorer scorecard coverage")
        if present and not aligned:
            gaps.append("missing or invalid raw metric alignment sidecar")
        out.append(
            {
                **spec,
                "status": "proven" if aligned else ("captured" if present else "missing"),
                "scorecard_coverage": bool(present),
                "unit": str(row.get("unit") or ""),
                "threshold": threshold,
                "raw_trace_ref": str(raw_trace) if raw_trace else "",
                "qualification_mask_ref": str(mask) if mask else "",
                "scenario_contract_ref": str(scenario) if scenario else "",
                "sample_count": sample_count,
                "aggregation": aggregation,
                "source_hashes": declared_hashes,
                "evidence": evidence if present and scorecard_evidence else [],
                "gaps": gaps,
            }
        )
    return out


def _load_list(path: Path | None, key: str) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    value = _read_json(path)
    if isinstance(value.get(key), list):
        return [item for item in value[key] if isinstance(item, dict)]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def assemble_manifest(
    run_dir: str | Path,
    *,
    output: str | Path | None = None,
    scorecards: Iterable[str | Path] = (),
    metric_alignment: str | Path | None = None,
    gate_calibration: str | Path | None = None,
    checkpoint_capabilities: str | Path | None = None,
    product_id: str | None = None,
) -> Path:
    run_path = Path(run_dir).resolve()
    manifest_path = run_path / "runtime_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["configuration_artifacts"] = configuration_artifact_parity(
        run_path,
        manifest,
        product_id=product_id,
    )
    telemetry_path = run_path / "train.telemetry.jsonl"
    if not telemetry_path.is_file():
        candidates = sorted(run_path.glob("*.telemetry.jsonl"))
        telemetry_path = candidates[0] if candidates else telemetry_path
    telemetry_rows = _read_jsonl(telemetry_path)
    manifest.update(command_coverage_from_telemetry(telemetry_rows, source=str(telemetry_path) if telemetry_rows else ""))
    if telemetry_rows:
        manifest["training_distribution"] = distribution_manifest_from_telemetry(
            telemetry_rows,
            source=str(telemetry_path),
            target=manifest.get("training_distribution") if isinstance(manifest.get("training_distribution"), dict) else None,
        )
    if scorecards:
        manifest["metric_alignment"] = metric_alignment_from_scorecards(
            [Path(item) for item in scorecards],
            telemetry_source=str(telemetry_path) if telemetry_rows else "",
            alignment_evidence=metric_alignment,
            product_id=product_id,
        )
        manifest.setdefault("evidence", {}).setdefault("scorecards", []).extend(str(item) for item in scorecards)
        if metric_alignment:
            manifest.setdefault("evidence", {}).setdefault("metric_alignment", []).append(str(metric_alignment))
    calibration = _load_list(Path(gate_calibration) if gate_calibration else None, "gate_calibration")
    if calibration:
        manifest["gate_calibration"] = calibration
        calibration_path = Path(gate_calibration).resolve()
        manifest.setdefault("evidence", {}).setdefault("gate_calibration", []).append({
            "path": str(calibration_path),
            "sha256": _sha256(calibration_path) if calibration_path.is_file() else "",
        })
    capabilities = _load_list(Path(checkpoint_capabilities) if checkpoint_capabilities else None, "checkpoint_capabilities")
    if capabilities:
        manifest["checkpoint_capabilities"] = capabilities
        manifest.setdefault("evidence", {}).setdefault("checkpoint_capabilities", []).append(str(checkpoint_capabilities))
    manifest["evidence"] = {
        **(manifest.get("evidence") if isinstance(manifest.get("evidence"), dict) else {}),
        "telemetry": [str(telemetry_path)] if telemetry_rows else [],
        "assembled_at": time_iso(),
    }
    target = Path(output).resolve() if output else manifest_path
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def time_iso() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assemble persisted product evidence into runtime_manifest.json")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--scorecard", action="append", default=[])
    parser.add_argument("--metric-alignment", default="")
    parser.add_argument("--gate-calibration", default="")
    parser.add_argument("--build-gate-calibration", action="store_true")
    parser.add_argument("--gate-reference-policy", default="")
    parser.add_argument("--gate-scenario-contract", default="")
    parser.add_argument("--minimum-gate-samples", type=int, default=3)
    parser.add_argument("--checkpoint-capabilities", default="")
    parser.add_argument("--product-id", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    gate_calibration = args.gate_calibration or None
    if args.build_gate_calibration:
        if not args.gate_reference_policy or not args.gate_scenario_contract:
            raise SystemExit("--build-gate-calibration requires --gate-reference-policy and --gate-scenario-contract")
        gate_calibration = str(build_gate_calibration_evidence(
            args.run_dir,
            reference_policy_manifest=args.gate_reference_policy,
            scenario_contract=args.gate_scenario_contract,
            minimum_samples=args.minimum_gate_samples,
        ))
    path = assemble_manifest(
        args.run_dir,
        output=args.output or None,
        scorecards=args.scorecard,
        metric_alignment=args.metric_alignment or None,
        gate_calibration=gate_calibration,
        checkpoint_capabilities=args.checkpoint_capabilities or None,
        product_id=args.product_id or None,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
