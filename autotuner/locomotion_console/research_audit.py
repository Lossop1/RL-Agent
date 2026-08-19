"""RL research audit primitives and a local, read-only preflight.

This module implements the first executable slice of the RL Agent proposal.  It
does not launch training, connect to a remote host, edit a reward, or claim
that a policy is good.  It answers a narrower question first: do we have the
facts needed to make an RL intervention safely and interpret its result?

The audit is deliberately split into static evidence (source/config hashes,
AST-derived reward structure) and runtime evidence supplied by an optional
manifest.  Static evidence alone never upgrades runtime claims to ``proven``.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field
import yaml

from .code_knowledge import _ROOT as PROJECT_ROOT
from autotuner.research.gate_calibration import extract_gate_definitions, gate_passes, quantiles
from .knowledge_model.reward_deriver import derive_reward_terms
from .knowledge_model.robot_sources import get_robot_sources


AuditStatus = Literal["ready", "incomplete", "blocked"]
# ``captured`` means a runtime artifact was recorded, but it is deliberately
# weaker than ``proven``: the artifact still needs calibration/alignment before
# it can authorize a research decision.
EvidenceStatus = Literal["proven", "captured", "declared", "unknown", "missing", "blocked"]
Severity = Literal["info", "warn", "error"]


class AuditFinding(BaseModel):
    id: str
    severity: Severity
    status: Literal["open", "resolved"] = "open"
    message: str
    evidence: list[str] = Field(default_factory=list)
    remediation: str = ""


class FileDigest(BaseModel):
    path: str
    exists: bool = False
    sha256: str = ""
    size_bytes: int = 0
    mtime_ns: int = 0


class RuntimeExecutionProof(BaseModel):
    status: EvidenceStatus = "unknown"
    source_files: list[FileDigest] = Field(default_factory=list)
    symbols: dict[str, bool] = Field(default_factory=dict)
    duplicate_modules: dict[str, list[str]] = Field(default_factory=dict)
    missing_files: list[str] = Field(default_factory=list)
    remote_verification: Literal["not_requested", "declared", "proven"] = "not_requested"
    notes: list[str] = Field(default_factory=list)


class MetricAlignmentContract(BaseModel):
    id: str
    capability: str
    train_signals: list[str] = Field(default_factory=list)
    eval_signals: list[str] = Field(default_factory=list)
    frame: str = ""
    event_window: str = ""
    eligibility: str = ""
    statistic: str = ""
    unit: str = ""
    threshold: dict[str, Any] = Field(default_factory=dict)
    scorecard_coverage: bool = False
    raw_trace_ref: str = ""
    qualification_mask_ref: str = ""
    scenario_contract_ref: str = ""
    sample_count: int = 0
    aggregation: dict[str, Any] = Field(default_factory=dict)
    source_hashes: dict[str, str] = Field(default_factory=dict)
    status: EvidenceStatus = "unknown"
    evidence: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class RewardAuditRecord(BaseModel):
    term: str
    sign: Literal["reward", "penalty", "mixed", "unknown"] = "unknown"
    weight_params: list[str] = Field(default_factory=list)
    gates: list[str] = Field(default_factory=list)
    measured_as: str = ""
    source: dict[str, Any] = Field(default_factory=dict)
    activation_status: EvidenceStatus = "unknown"
    gradient_status: EvidenceStatus = "unknown"
    saturation_status: EvidenceStatus = "unknown"
    findings: list[str] = Field(default_factory=list)


class GateCalibrationRecord(BaseModel):
    gate_id: str
    code_default: float | None = None
    config_path: str = ""
    configured_threshold: float | None = None
    comparison: Literal["gte", "lte", "unknown"] = "unknown"
    metric_field: str = ""
    purpose: Literal["curriculum_progress", "dr_progress", "final_acceptance", "unknown"] = "unknown"
    active: bool = True
    phase: int | None = None
    phase_min: int | None = None
    dr_level: int | None = None
    qualification_rule: str = ""
    reference_policy_ref: str = ""
    distribution_ref: str = ""
    effective_config_ref: str = ""
    raw_trace_ref: str = ""
    qualification_mask_ref: str = ""
    scenario_contract_ref: str = ""
    sample_count: int = 0
    eligible_count: int = 0
    pass_count: int = 0
    pass_rate: float = 0.0
    minimum_samples: int = 0
    observed_range: dict[str, float] = Field(default_factory=dict)
    reachable_interval: dict[str, float] = Field(default_factory=dict)
    observed_ceiling: float | None = None
    aggregation: dict[str, Any] = Field(default_factory=dict)
    source_hashes: dict[str, str] = Field(default_factory=dict)
    status: EvidenceStatus = "unknown"
    evidence: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class CommandCoverageLedger(BaseModel):
    status: EvidenceStatus = "missing"
    window: str = ""
    required_buckets: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    missing_buckets: list[str] = Field(default_factory=list)
    invalid_buckets: list[str] = Field(default_factory=list)
    source: str = ""


class OptimizationStateSnapshot(BaseModel):
    status: EvidenceStatus = "missing"
    fields: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    source: str = ""


class CheckpointCapabilityRecord(BaseModel):
    checkpoint: str
    step: int = 0
    path: str
    size_bytes: int = 0
    mtime_ns: int = 0
    role: Literal["newest", "best", "candidate", "counterexample", "unknown"] = "unknown"
    capability_status: EvidenceStatus = "unknown"
    capabilities: dict[str, Any] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)


class CheckpointCapabilityRegistry(BaseModel):
    status: EvidenceStatus = "missing"
    root: str = ""
    records: list[CheckpointCapabilityRecord] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ResearchAuditReport(BaseModel):
    schema_version: str = "rl-agent.research-audit/v1"
    generated_at: str
    program_id: str = "taili_blind_locomotion"
    root: str
    status: AuditStatus
    runtime_execution: RuntimeExecutionProof
    reward_terms: list[RewardAuditRecord] = Field(default_factory=list)
    metric_alignment: list[MetricAlignmentContract] = Field(default_factory=list)
    gate_calibration: list[GateCalibrationRecord] = Field(default_factory=list)
    command_coverage: CommandCoverageLedger
    optimization_state: OptimizationStateSnapshot
    checkpoints: CheckpointCapabilityRegistry
    findings: list[AuditFinding] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    permission_boundary: str = (
        "只读本地审计：不连接远端、不启动训练、不修改奖励/配置、不删除或覆盖 checkpoint。"
    )


_STEP_RE = re.compile(r"agent_(\d+)\.pt$")
_REQUIRED_COMMAND_BUCKETS = ("forward", "backward", "lateral", "yaw", "stand", "mixed")
_REQUIRED_OPTIMIZATION_FIELDS = (
    "policy", "value", "optimizer", "scheduler", "normalizer", "log_std", "amp", "curriculum", "rng"
)
_DEFAULT_ALIGNMENT_SPECS = (
    {
        "id": "A1",
        "capability": "flat_linear_tracking",
        "train_signals": ["tracking_lin", "tracking_lin_far"],
        "eval_signals": ["score_A1"],
        "frame": "body",
        "event_window": "steady moving-command frames",
        "eligibility": "nonzero linear command; no terminal frames",
        "statistic": "per-command bucket median and p90",
    },
    {
        "id": "A2",
        "capability": "flat_yaw_tracking",
        "train_signals": ["tracking_yaw", "tracking_yaw_far"],
        "eval_signals": ["score_A2"],
        "frame": "body yaw rate",
        "event_window": "steady yaw-command frames",
        "eligibility": "abs(cmd_wz) above yaw deadband",
        "statistic": "per-command bucket median and p90",
    },
    {
        "id": "A3",
        "capability": "stand_transition",
        "train_signals": ["stand", "stand_contact", "stand_far", "off_axis"],
        "eval_signals": ["score_A3", "score_C"],
        "frame": "body and local terrain frame",
        "event_window": "command nonzero -> zero through settled tail",
        "eligibility": "zero-command transition with known initial motion",
        "statistic": "settle time plus tail speed/yaw/duty/upright",
    },
    {
        "id": "B1",
        "capability": "touchdown_impact",
        "train_signals": ["landing_impact"],
        "eval_signals": ["score_B1"],
        "frame": "foot contact frame",
        "event_window": "swing -> settled stance touchdown window",
        "eligibility": "verified touchdown transition; exclude liftoff",
        "statistic": "per-touchdown-event p95",
    },
    {
        "id": "B2",
        "capability": "settled_stance_slip",
        "train_signals": ["stance_slip"],
        "eval_signals": ["score_B2"],
        "frame": "contact-point relative body frame",
        "event_window": "settled stance only",
        "eligibility": "firm support contact; exclude touchdown/liftoff transition",
        "statistic": "per-settled-frame p90",
    },
    {
        "id": "B3",
        "capability": "foot_clearance",
        "train_signals": ["clearance_under", "clearance_over"],
        "eval_signals": ["score_B3"],
        "frame": "local terrain frame",
        "event_window": "swing frames only",
        "eligibility": "verified swing phase; terrain obstacle height available when required",
        "statistic": "per-step swing peak / obstacle-relative margin",
    },
    {
        "id": "B4",
        "capability": "gait_symmetry",
        "train_signals": ["gait_anchor", "diagonal_contact", "duty_balance"],
        "eval_signals": ["score_B4"],
        "frame": "per-leg contact frame",
        "event_window": "full steady command window",
        "eligibility": "symmetric command and valid four-leg contact history",
        "statistic": "per-scene duty and clearance differences",
    },
    {
        "id": "D_STAIRS",
        "capability": "stairs_controlled_traversal",
        "train_signals": ["terrain_progress", "terrain_direction", "terrain_clearance"],
        "eval_signals": ["score_D", "score_D_ascend"],
        "frame": "body direction and local stair frame",
        "event_window": "full approach -> stair traversal -> final platform",
        "eligibility": "correct stair geometry, command direction, nonterminal trajectory",
        "statistic": "complete-case rate plus along-path/net-height/quality tails",
    },
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(root: Path, relative: str) -> FileDigest:
    path = root / relative
    if not path.is_file():
        return FileDigest(path=relative, exists=False)
    stat = path.stat()
    return FileDigest(
        path=relative,
        exists=True,
        sha256=_sha256(path),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


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
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _same_number(left: Any, right: Any, *, tolerance: float = 1e-9) -> bool:
    lhs = _number(left)
    rhs = _number(right)
    if lhs is None or rhs is None:
        return False
    return abs(lhs - rhs) <= tolerance * max(1.0, abs(lhs), abs(rhs))


def _source_files(root: Path) -> list[str]:
    sources = get_robot_sources()
    return list(dict.fromkeys([
        sources.reward_file,
        sources.env_reward_file,
        sources.curriculum_file,
        "autotuner/blind_locomotion/taili_blind_config.yaml",
        "autotuner/blind_locomotion/train_taili.py",
        "autotuner/blind_locomotion/acceptance_score.py",
        "autotuner/blind_locomotion/telemetry_emit.py",
        "autotuner/blind_locomotion/telemetry_payloads.py",
        "autotuner/blind_locomotion/runtime_manifest.py",
    ]))


def _duplicate_modules(root: Path) -> dict[str, list[str]]:
    backup_root = root / "strategy_backups"
    if not backup_root.is_dir():
        return {}
    names = ("taili_reward.py", "acceptance_score.py", "blind_tp_env.py", "taili_blind_config.yaml")
    duplicates: dict[str, list[str]] = {}
    for name in names:
        paths = sorted(_rel(root, p) for p in backup_root.rglob(name) if p.is_file())
        if paths:
            duplicates[name] = paths[:100]
    return duplicates


def build_runtime_execution_proof(
    root: Path = PROJECT_ROOT,
    manifest: dict[str, Any] | None = None,
) -> RuntimeExecutionProof:
    """Prove the local source path and surface shadowing risks without importing training code."""
    files = [_digest(root, rel) for rel in _source_files(root)]
    missing = [item.path for item in files if not item.exists]
    symbols: dict[str, bool] = {}
    reward_path = root / "autotuner/taili_core/taili_reward.py"
    env_path = root / "autotuner/blind_locomotion/blind_tp_env.py"
    for label, path, names in (
        ("reward.compute_reward_components", reward_path, ("compute_reward_components",)),
        ("env.TailiBlindTPEnv", env_path, ("TailiBlindTPEnv",)),
        ("env.TailiBlindTPEnv._get_rewards", env_path, ("_get_rewards",)),
        ("env.TailiBlindTPEnv._reset_idx", env_path, ("_reset_idx",)),
        ("env.TailiBlindTPEnv._resample_commands", env_path, ("_resample_commands",)),
    ):
        found = False
        if path.is_file():
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
                found = any(
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    and node.name in names
                    for node in ast.walk(tree)
                )
            except (OSError, SyntaxError, UnicodeError):
                found = False
        symbols[label] = found

    duplicates = _duplicate_modules(root)
    notes: list[str] = [
        "本证明只覆盖本地源码；远端 payload、sys.path 和实际部署 import 尚未核验。",
    ]
    if duplicates:
        notes.append("strategy_backups 中存在同名历史模块；它们不自动等于运行时冲突，但必须在远端启动证明中排除 shadowing。")
    if missing:
        status: EvidenceStatus = "blocked"
    elif duplicates:
        status = "declared"
    else:
        status = "declared"
    local = RuntimeExecutionProof(
        status=status,
        source_files=files,
        symbols=symbols,
        duplicate_modules=duplicates,
        missing_files=missing,
        notes=notes,
    )
    runtime = (manifest or {}).get("runtime_execution")
    runtime_files = runtime.get("source_files") if isinstance(runtime, dict) else None
    if not isinstance(runtime, dict) or runtime.get("status") != "proven" or runtime.get("remote_verification") != "proven":
        return local
    if not isinstance(runtime_files, list) or not runtime_files:
        return local
    return RuntimeExecutionProof(
        status="proven",
        source_files=[FileDigest(**item) for item in runtime_files if isinstance(item, dict)],
        symbols={str(key): bool(value) for key, value in (runtime.get("symbols") or {}).items()},
        duplicate_modules={},
        missing_files=[str(item) for item in runtime.get("missing_files", [])],
        remote_verification="proven",
        notes=list(local.notes)
        + [str(item) for item in runtime.get("notes", [])]
        + ["runtime manifest proved resolved import paths and checked shadowing"],
    )


def _reward_records(
    root: Path,
    manifest: dict[str, Any] | None = None,
) -> tuple[list[RewardAuditRecord], list[AuditFinding]]:
    relative = "autotuner/taili_core/taili_reward.py"
    path = root / relative
    findings: list[AuditFinding] = []
    if not path.is_file():
        return [], [AuditFinding(
            id="reward.source_missing", severity="error", message=f"奖励源文件不存在: {relative}",
            remediation="恢复权威奖励源或更新 RobotSources；禁止在源文件不明时启动调参。",
        )]
    try:
        text = path.read_text(encoding="utf-8")
        derived = derive_reward_terms(text, relative)
    except (OSError, UnicodeError, SyntaxError) as exc:
        return [], [AuditFinding(
            id="reward.parse_failed", severity="error", message=f"奖励 AST 推导失败: {type(exc).__name__}: {exc}",
            evidence=[relative], remediation="先修复奖励源的解析/版本问题，再进行实验。",
        )]

    edge_map: dict[str, dict[str, list[str]]] = {}
    for edge in derived.get("edges", []):
        src = getattr(edge, "src", "")
        edge_map.setdefault(src, {}).setdefault(getattr(edge, "rel", ""), []).append(getattr(edge, "dst", ""))
    runtime_rows = {
        str(item.get("term")): item
        for item in (manifest or {}).get("reward_terms", [])
        if isinstance(item, dict) and item.get("term")
    }
    records: list[RewardAuditRecord] = []
    for entity in derived.get("entities", []):
        if getattr(entity, "type", "") != "reward_term":
            continue
        attrs = getattr(entity, "attrs", {}) or {}
        name = str(getattr(entity, "name", ""))
        source = getattr(entity, "source", None)
        source_dict = source.model_dump(mode="json") if source is not None else {}
        key = f"reward_term:{name}"
        runtime_row = runtime_rows.get(name, {})
        record = RewardAuditRecord(
            term=name,
            sign=attrs.get("sign", "unknown"),
            weight_params=list(attrs.get("weight_params") or []),
            gates=[item.split(":", 1)[-1] for item in edge_map.get(key, {}).get("gated_by", [])],
            measured_as=next((item.split(":", 1)[-1] for item in edge_map.get(key, {}).get("measured_as", [])), ""),
            source=source_dict,
            activation_status=runtime_row.get("activation_status", "unknown"),
            gradient_status=runtime_row.get("gradient_status", "unknown"),
            saturation_status=runtime_row.get("saturation_status", "unknown"),
            findings=["运行窗口、有效样本、梯度和饱和率尚未通过 manifest 提供。"],
        )
        records.append(record)
        if record.sign == "unknown":
            findings.append(AuditFinding(
                id=f"reward.unknown_sign.{name}", severity="warn",
                message=f"奖励项 {name} 的正负语义无法由 AST 确认。",
                evidence=[relative], remediation="补充明确的正向驱动/质量惩罚标注或人工审查。",
            ))
    if not records:
        findings.append(AuditFinding(
            id="reward.no_terms", severity="error", message="奖励函数没有推导出任何 comp 项。",
            evidence=[relative], remediation="检查奖励函数名、RobotSources 和源版本。",
        ))
    if not runtime_rows:
        findings.append(AuditFinding(
            id="reward.gradient_not_measured", severity="warn",
        message="静态奖励结构已取得，但没有运行时梯度、饱和率或优势贡献证据。",
        evidence=[relative, "RewardAuditRecord.activation_status/gradient_status"],
        remediation="提供 rollout audit manifest 后再允许高风险权重/门控实验。",
        ))
    return records, findings


def build_metric_alignment(root: Path = PROJECT_ROOT, manifest: dict[str, Any] | None = None) -> tuple[list[MetricAlignmentContract], list[AuditFinding]]:
    """Build conservative train/eval mappings; a manifest is required for ``proven``."""
    manifest_rows = {str(item.get("id")): item for item in (manifest or {}).get("metric_alignment", []) if isinstance(item, dict)}
    reward_text = (root / "autotuner/taili_core/taili_reward.py").read_text(encoding="utf-8") if (root / "autotuner/taili_core/taili_reward.py").is_file() else ""
    score_text = (root / "autotuner/blind_locomotion/acceptance_score.py").read_text(encoding="utf-8") if (root / "autotuner/blind_locomotion/acceptance_score.py").is_file() else ""
    out: list[MetricAlignmentContract] = []
    findings: list[AuditFinding] = []
    for spec in _DEFAULT_ALIGNMENT_SPECS:
        row = manifest_rows.get(spec["id"], {})
        train_ok = all(signal in reward_text for signal in spec["train_signals"][:1])
        eval_ok = all(signal in score_text for signal in spec["eval_signals"])
        evidence = ["autotuner/taili_core/taili_reward.py"] if train_ok else []
        if eval_ok:
            evidence.append("autotuner/blind_locomotion/acceptance_score.py")
        gaps: list[str] = []
        if not train_ok:
            gaps.append("训练信号未在当前奖励源中确认。")
        if not eval_ok:
            gaps.append("验收函数未在当前 scorer 中确认。")
        status: EvidenceStatus = "declared" if train_ok and eval_ok else "missing"
        row_gaps: list[str] = [str(item) for item in row.get("gaps", []) if str(item)]
        sample_count = 0
        if row.get("status") in {"captured", "proven"}:
            status = "captured"
            evidence.extend(str(item) for item in row.get("evidence", []))
        if row.get("status") == "proven" and row.get("scorecard_coverage"):
            try:
                sample_count = int(row.get("sample_count", 0))
            except (TypeError, ValueError):
                sample_count = 0
            if sample_count <= 0:
                row_gaps.append("sample_count must be positive")
            if not row.get("aggregation"):
                row_gaps.append("aggregation")
            if not row.get("unit"):
                row_gaps.append("unit")
            if not row.get("threshold"):
                row_gaps.append("threshold")
            source_hashes = row.get("source_hashes") if isinstance(row.get("source_hashes"), dict) else {}
            for field, hash_key in (
                ("raw_trace_ref", "raw_trace"),
                ("qualification_mask_ref", "qualification_mask"),
                ("scenario_contract_ref", "scenario_contract"),
            ):
                reference = str(row.get(field) or "")
                path = Path(reference).resolve() if reference else Path()
                expected = str(source_hashes.get(hash_key) or "")
                if not reference or not _inside(root, path) or not path.is_file():
                    row_gaps.append(field)
                elif not expected or _sha256(path) != expected:
                    row_gaps.append(f"{field} hash mismatch")
            if not row_gaps and not gaps and row.get("evidence"):
                status = "proven"
        if row_gaps:
            gaps.extend(f"alignment evidence missing: {name}" for name in row_gaps)
        if status == "captured" and not row_gaps:
            gaps.append("scorecard coverage exists, but raw metric alignment proof is missing")
        out.append(MetricAlignmentContract(
            **spec,
            unit=str(row.get("unit") or ""),
            threshold=row.get("threshold") if isinstance(row.get("threshold"), dict) else {},
            scorecard_coverage=bool(row.get("scorecard_coverage")),
            raw_trace_ref=str(row.get("raw_trace_ref") or ""),
            qualification_mask_ref=str(row.get("qualification_mask_ref") or ""),
            scenario_contract_ref=str(row.get("scenario_contract_ref") or ""),
            sample_count=sample_count if row.get("status") == "proven" else 0,
            aggregation=row.get("aggregation") if isinstance(row.get("aggregation"), dict) else {},
            source_hashes=row.get("source_hashes") if isinstance(row.get("source_hashes"), dict) else {},
            status=status,
            evidence=list(dict.fromkeys(evidence)),
            gaps=list(dict.fromkeys(gaps)),
        ))
        if status != "proven":
            findings.append(AuditFinding(
                id=f"metric.not_proven.{spec['id']}", severity="error",
                message=f"{spec['id']} 的训练/验收物理量尚未完成运行时对齐证明。",
                evidence=evidence, remediation="为该指标提供同一物理量的窗口、资格、统计和 raw trace manifest。",
            ))
    return out, findings


def _gate_config_path(root: Path, manifest: dict[str, Any]) -> tuple[Path | None, list[str]]:
    """Resolve only a locally persisted effective config with proven parity."""
    gaps: list[str] = []
    artifacts = manifest.get("configuration_artifacts") if isinstance(manifest.get("configuration_artifacts"), dict) else {}
    effective = artifacts.get("effective_config") if isinstance(artifacts.get("effective_config"), dict) else {}
    local = effective.get("local") if isinstance(effective.get("local"), dict) else {}
    reference = str(local.get("path") or "")
    path = Path(reference).resolve() if reference else None
    if effective.get("status") != "proven" or not effective.get("sha256_match"):
        gaps.append("effective config local/remote parity is not proven")
    if path is None or not _inside(root, path) or not path.is_file():
        gaps.append("effective config is not a persisted local artifact inside the audit root")
        return None, gaps
    if str(local.get("sha256") or "") != _sha256(path):
        gaps.append("effective config local hash mismatch")
    return path, gaps


def _load_effective_config(path: Path | None) -> tuple[dict[str, Any], list[str]]:
    if path is None:
        return {}, []
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        return {}, [f"effective config parse failed: {type(exc).__name__}"]
    if not isinstance(value, dict):
        return {}, ["effective config root is not a mapping"]
    return value, []


def _resolve_local_artifact(root: Path, reference: Any) -> Path | None:
    if not isinstance(reference, str) or not reference.strip():
        return None
    path = Path(reference).resolve()
    return path if _inside(root, path) and path.is_file() else None


def _validate_frozen_reference(
    reference_path: Path | None,
    scenario_path: Path | None,
    effective_config_path: Path | None,
) -> list[str]:
    gaps: list[str] = []
    reference = _read_json(reference_path) if reference_path else {}
    scenario = _read_json(scenario_path) if scenario_path else {}
    checkpoint = reference.get("checkpoint") if isinstance(reference.get("checkpoint"), dict) else {}
    checkpoint_hash = str(checkpoint.get("sha256") or "")
    if reference.get("schema_version") != "rl-agent.gate-calibration-rollout/v1":
        gaps.append("reference policy manifest has the wrong schema")
    if reference.get("status") != "complete":
        gaps.append("reference rollout is not complete")
    if reference.get("training_enabled") is not False or reference.get("policy_mode") != "mean_action":
        gaps.append("reference rollout did not prove a frozen mean-action policy")
    runtime = reference.get("runtime_execution") if isinstance(reference.get("runtime_execution"), dict) else {}
    if runtime.get("status") != "proven" or runtime.get("remote_verification") != "proven":
        gaps.append("reference rollout runtime execution is not proven")
    if checkpoint.get("status") != "captured" or len(checkpoint_hash) != 64:
        gaps.append("reference checkpoint digest was not captured")

    effective_hash = _sha256(effective_config_path) if effective_config_path and effective_config_path.is_file() else ""
    if scenario.get("schema_version") != "rl-agent.gate-calibration-scenario/v1":
        gaps.append("gate scenario contract has the wrong schema")
    if scenario.get("training_enabled") is not False or scenario.get("policy_mode") != "mean_action":
        gaps.append("gate scenario does not require frozen mean actions")
    if scenario.get("command_source") != "environment_sampler":
        gaps.append("gate scenario did not use the runtime command sampler")
    if int(_number(scenario.get("num_envs")) or 0) <= 0 or int(_number(scenario.get("steps")) or 0) <= 0:
        gaps.append("gate scenario has no bounded environment/step count")
    if str(scenario.get("checkpoint_sha256") or "") != checkpoint_hash:
        gaps.append("scenario checkpoint hash differs from reference policy")
    if not effective_hash or str(scenario.get("effective_config_sha256") or "") != effective_hash:
        gaps.append("scenario effective-config hash differs from the audited run")

    phase_number = _number(scenario.get("phase_override"))
    dr_number = _number(scenario.get("dr_level_override"))
    phase_override = int(phase_number) if phase_number is not None else -1
    dr_override = int(dr_number) if dr_number is not None else -1
    resolved = scenario.get("resolved_curriculum") if isinstance(scenario.get("resolved_curriculum"), dict) else {}
    if phase_override >= 0 and (
        _number(resolved.get("phase")) != phase_override
        or _number(resolved.get("max_training_phase")) != phase_override
    ):
        gaps.append("runtime phase coordinates differ from the fixed scenario override")
    if dr_override >= 0 and _number(resolved.get("dr_level")) != dr_override:
        gaps.append("runtime DR level differs from the fixed scenario override")

    restore = scenario.get("curriculum_restore") if isinstance(scenario.get("curriculum_restore"), dict) else {}
    if restore.get("requested"):
        if restore.get("status") != "proven":
            gaps.append("requested curriculum restore was not proven at runtime")
        artifact_ref = str(restore.get("artifact_ref") or "")
        artifact_path = None
        if scenario_path and artifact_ref:
            candidate = Path(artifact_ref)
            artifact_path = candidate if candidate.is_absolute() else scenario_path.parent / candidate
        artifact = restore.get("artifact") if isinstance(restore.get("artifact"), dict) else {}
        source = restore.get("source") if isinstance(restore.get("source"), dict) else {}
        expected_hash = str(artifact.get("sha256") or "")
        if artifact_path is None or not artifact_path.is_file():
            gaps.append("restored curriculum artifact is not persisted beside the scenario")
        elif not expected_hash or _sha256(artifact_path) != expected_hash:
            gaps.append("restored curriculum artifact hash mismatch")
        if expected_hash != str(source.get("sha256") or ""):
            gaps.append("curriculum restore source and evidence copy hashes differ")
        reference_restore = (
            reference.get("curriculum_restore")
            if isinstance(reference.get("curriculum_restore"), dict)
            else {}
        )
        reference_artifact = (
            reference_restore.get("artifact")
            if isinstance(reference_restore.get("artifact"), dict)
            else {}
        )
        if str(reference_artifact.get("sha256") or "") != expected_hash:
            gaps.append("reference policy and scenario disagree on restored curriculum")
    return gaps


def _recompute_gate_row(
    root: Path,
    row: dict[str, Any],
    *,
    gate_id: str,
    threshold: float,
    comparison: str,
) -> tuple[dict[str, Any], list[str]]:
    """Re-read raw samples/mask and reject self-asserted sidecar claims."""
    gaps: list[str] = []
    hashes = row.get("source_hashes") if isinstance(row.get("source_hashes"), dict) else {}
    paths: dict[str, Path | None] = {
        name: _resolve_local_artifact(root, row.get(field))
        for name, field in (
            ("effective_config", "effective_config_ref"),
            ("telemetry", "distribution_ref"),
            ("raw_trace", "raw_trace_ref"),
            ("qualification_mask", "qualification_mask_ref"),
            ("reference_policy", "reference_policy_ref"),
            ("scenario_contract", "scenario_contract_ref"),
        )
    }
    for name, path in paths.items():
        if path is None:
            gaps.append(f"missing local artifact: {name}")
        elif str(hashes.get(name) or "") != _sha256(path):
            gaps.append(f"artifact hash mismatch: {name}")
    gaps.extend(_validate_frozen_reference(
        paths["reference_policy"],
        paths["scenario_contract"],
        paths["effective_config"],
    ))

    raw_rows = _read_jsonl(paths["raw_trace"]) if paths["raw_trace"] else []
    mask = _read_json(paths["qualification_mask"]) if paths["qualification_mask"] else {}
    eligible_map = mask.get("eligible_sample_ids") if isinstance(mask.get("eligible_sample_ids"), dict) else {}
    rules = mask.get("rules") if isinstance(mask.get("rules"), dict) else {}
    if str(rules.get(gate_id) or "") != str(row.get("qualification_rule") or ""):
        gaps.append("qualification mask rule differs from GateCalibrationRecord")
    eligible_ids = {
        str(item) for item in eligible_map.get(gate_id, [])
    } if isinstance(eligible_map.get(gate_id), list) else set()
    samples: list[dict[str, Any]] = []
    duplicate_tokens: set[tuple[str, int]] = set()
    for item in raw_rows:
        if str(item.get("gate_id") or "") != gate_id:
            continue
        sample_id = str(item.get("sample_id") or "")
        if sample_id not in eligible_ids or not item.get("eligible"):
            continue
        value = _number(item.get("value"))
        eval_step = _number(item.get("gate_eval_step"))
        if value is None or eval_step is None:
            continue
        token = (gate_id, int(eval_step))
        if token in duplicate_tokens:
            gaps.append("raw trace contains duplicate gate-evaluation windows")
            continue
        duplicate_tokens.add(token)
        if str(item.get("comparison") or "") != comparison:
            gaps.append("raw trace comparison differs from effective gate semantics")
        if not _same_number(item.get("threshold"), threshold):
            gaps.append("raw trace threshold differs from effective config")
        samples.append(item)
    values = [float(item["value"]) for item in samples]
    observed = quantiles(values)
    pass_count = sum(gate_passes(float(item["value"]), threshold, comparison) for item in samples)
    sample_count = len(samples)
    minimum_samples = int(_number(row.get("minimum_samples")) or 1)
    if sample_count < minimum_samples:
        gaps.append(f"qualified samples {sample_count} < {minimum_samples}")
    if pass_count <= 0:
        gaps.append("reference policy never reached the configured threshold")
    if int(_number(row.get("sample_count")) or 0) != sample_count:
        gaps.append("declared sample_count does not match raw trace")
    if int(_number(row.get("eligible_count")) or 0) != sample_count:
        gaps.append("declared eligible_count does not match qualification mask")
    if int(_number(row.get("pass_count")) or 0) != pass_count:
        gaps.append("declared pass_count does not match recomputed comparisons")
    declared_range = row.get("observed_range") if isinstance(row.get("observed_range"), dict) else {}
    if observed and any(not _same_number(declared_range.get(key), value) for key, value in observed.items()):
        gaps.append("declared observed_range does not match raw trace")
    return {
        "paths": paths,
        "sample_count": sample_count,
        "eligible_count": sample_count,
        "pass_count": pass_count,
        "pass_rate": pass_count / sample_count if sample_count else 0.0,
        "minimum_samples": minimum_samples,
        "observed_range": observed,
        "reachable_interval": (
            {"lower": observed["min"], "upper": observed["max"]}
            if observed else {}
        ),
        "observed_ceiling": (
            max(values) if comparison == "gte" else min(values)
        ) if values else None,
    }, list(dict.fromkeys(gaps))


def build_gate_calibration(root: Path = PROJECT_ROOT, manifest: dict[str, Any] | None = None) -> tuple[list[GateCalibrationRecord], list[AuditFinding]]:
    data = manifest or {}
    config_path, config_gaps = _gate_config_path(root, data)
    config, parse_gaps = _load_effective_config(config_path)
    definitions = extract_gate_definitions(config)
    manifest_rows = {
        str(item.get("gate_id")): item
        for item in data.get("gate_calibration", [])
        if isinstance(item, dict) and item.get("gate_id")
    }
    records: list[GateCalibrationRecord] = []
    findings: list[AuditFinding] = []
    shared_gaps = config_gaps + parse_gaps

    for definition in definitions:
        row = manifest_rows.get(definition.gate_id, {})
        gaps = list(shared_gaps)
        recomputed: dict[str, Any] = {}
        if not definition.active:
            status: EvidenceStatus = "declared"
            gaps.extend([definition.note or "configured value is not an active runtime gate"])
        elif not row:
            status = "missing"
            gaps.append("missing GateCalibrationRecord for effective gate")
        else:
            if str(row.get("config_path") or "") != definition.config_path:
                gaps.append("record config_path differs from effective config")
            if not _same_number(row.get("configured_threshold"), definition.configured_threshold):
                gaps.append("record threshold differs from effective config")
            if str(row.get("comparison") or "") != definition.comparison:
                gaps.append("record comparison differs from runtime semantics")
            if str(row.get("metric_field") or "") != definition.metric_field:
                gaps.append("record metric_field differs from runtime semantics")
            recomputed, row_gaps = _recompute_gate_row(
                root,
                row,
                gate_id=definition.gate_id,
                threshold=definition.configured_threshold,
                comparison=definition.comparison,
            )
            gaps.extend(row_gaps)
            status = "proven" if not gaps else "captured" if recomputed.get("sample_count") else "missing"
        evidence = [str(config_path)] if config_path else []
        evidence.extend(str(item) for item in row.get("evidence", []) if isinstance(item, str))
        records.append(GateCalibrationRecord(
            gate_id=definition.gate_id,
            config_path=definition.config_path,
            configured_threshold=definition.configured_threshold,
            comparison=definition.comparison,
            metric_field=definition.metric_field,
            purpose=definition.purpose if definition.purpose in {"curriculum_progress", "dr_progress", "final_acceptance"} else "unknown",
            active=definition.active,
            phase=definition.phase,
            phase_min=definition.phase_min,
            dr_level=definition.dr_level,
            qualification_rule=definition.qualification_rule,
            reference_policy_ref=str(row.get("reference_policy_ref") or ""),
            distribution_ref=str(row.get("distribution_ref") or ""),
            effective_config_ref=str(row.get("effective_config_ref") or ""),
            raw_trace_ref=str(row.get("raw_trace_ref") or ""),
            qualification_mask_ref=str(row.get("qualification_mask_ref") or ""),
            scenario_contract_ref=str(row.get("scenario_contract_ref") or ""),
            sample_count=int(recomputed.get("sample_count", 0)),
            eligible_count=int(recomputed.get("eligible_count", 0)),
            pass_count=int(recomputed.get("pass_count", 0)),
            pass_rate=float(recomputed.get("pass_rate", 0.0)),
            minimum_samples=int(recomputed.get("minimum_samples", _number(row.get("minimum_samples")) or 0)),
            observed_range=recomputed.get("observed_range", {}),
            reachable_interval=recomputed.get("reachable_interval", {}),
            observed_ceiling=recomputed.get("observed_ceiling"),
            aggregation=row.get("aggregation") if isinstance(row.get("aggregation"), dict) else {},
            source_hashes=row.get("source_hashes") if isinstance(row.get("source_hashes"), dict) else {},
            status=status,
            evidence=list(dict.fromkeys(evidence)),
            gaps=list(dict.fromkeys(gaps)),
        ))

    active_records = [item for item in records if item.active]
    if not definitions:
        findings.append(AuditFinding(
            id="curriculum.effective_gates_missing", severity="error",
            message="没有从本次 run 的 effective_config.yaml 得到可测量课程门槛。",
            evidence=[str(config_path)] if config_path else [],
            remediation="先组装并校验本次运行配置的本地/远端 hash parity。",
        ))
    elif any(item.status != "proven" for item in active_records):
        findings.append(AuditFinding(
            id="curriculum.gates_not_calibrated", severity="error",
            message="本次运行的有效课程门槛尚未全部用冻结参考策略和原始合格样本校准。",
            evidence=[str(config_path)] if config_path else [],
            remediation="对冻结 checkpoint 做有边界 rollout，保存 raw trace、资格 mask、场景契约和配置 hash 后重新审计。",
        ))
    return records, findings


def build_command_coverage(manifest: dict[str, Any] | None = None) -> tuple[CommandCoverageLedger, list[AuditFinding]]:
    data = manifest or {}
    rows = [item for item in data.get("command_coverage", []) if isinstance(item, dict)]
    row_by_bucket = {
        str(item.get("bucket") or item.get("direction") or ""): item
        for item in rows
        if str(item.get("bucket") or item.get("direction") or "")
    }
    missing = [
        bucket
        for bucket in _REQUIRED_COMMAND_BUCKETS
        if bucket not in row_by_bucket
        or any(
            not isinstance(row_by_bucket[bucket].get(key), (int, float))
            or row_by_bucket[bucket].get(key, 0) <= 0
            for key in ("target_count", "applied_count", "eligible_count")
        )
    ]
    invalid: list[str] = []
    for item in rows:
        bucket = str(item.get("bucket") or item.get("direction") or "")
        for key in ("target_count", "applied_count", "eligible_count"):
            value = item.get(key)
            if value is not None and (not isinstance(value, (int, float)) or value < 0):
                invalid.append(bucket or "<unnamed>")
                break
    status: EvidenceStatus = "proven" if rows and not missing and not invalid else ("blocked" if invalid else "missing")
    ledger = CommandCoverageLedger(
        status=status,
        window=str(data.get("coverage_window") or ""),
        required_buckets=list(_REQUIRED_COMMAND_BUCKETS),
        rows=rows,
        missing_buckets=missing,
        invalid_buckets=sorted(set(invalid)),
        source=str(data.get("command_coverage_source") or "manifest" if rows else ""),
    )
    findings: list[AuditFinding] = []
    if missing:
        findings.append(AuditFinding(
            id="commands.coverage_missing", severity="error",
            message=f"命令覆盖缺失: {', '.join(missing)}。",
            remediation="记录目标命令、实际应用、有效 rollout、settled、terminal 和 timeout 样本。",
        ))
    if invalid:
        findings.append(AuditFinding(
            id="commands.coverage_invalid", severity="error",
            message=f"命令覆盖存在非法计数: {', '.join(sorted(set(invalid)))}。",
            remediation="修复 manifest 生成器，禁止负数或非数值样本。",
        ))
    return ledger, findings


def build_optimization_state(manifest: dict[str, Any] | None = None) -> tuple[OptimizationStateSnapshot, list[AuditFinding]]:
    raw = (manifest or {}).get("optimization_state")
    if not isinstance(raw, dict):
        return OptimizationStateSnapshot(
            status="missing", missing_fields=list(_REQUIRED_OPTIMIZATION_FIELDS), source=""
        ), [AuditFinding(
            id="optimization.state_missing", severity="error",
            message="没有提供完整 PPO/resume 优化状态快照。",
            remediation="记录 policy/value/optimizer/scheduler/normalizer/log_std/AMP/curriculum/RNG 状态并做恢复 parity。",
        )]
    fields = raw.get("fields") if isinstance(raw.get("fields"), dict) else raw

    def field_ok(value: Any) -> bool:
        if isinstance(value, dict):
            return str(value.get("status") or "") in {
                "captured", "fresh_initialization", "not_configured", "proven"
            }
        return bool(value)

    missing = [key for key in _REQUIRED_OPTIMIZATION_FIELDS if not field_ok(fields.get(key))]
    parity = bool(raw.get("parity_check"))
    status: EvidenceStatus = "proven" if not missing and parity else ("declared" if not missing else "missing")
    snapshot = OptimizationStateSnapshot(
        status=status,
        fields={key: fields.get(key) for key in _REQUIRED_OPTIMIZATION_FIELDS if key in fields},
        missing_fields=missing,
        source=str(raw.get("source") or "manifest"),
    )
    findings: list[AuditFinding] = []
    if missing or not parity:
        findings.append(AuditFinding(
            id="optimization.state_not_proven", severity="error",
            message="优化器/课程/随机性状态仅声明或缺字段，不能证明 resume 路线连续。",
            remediation="完成恢复后状态 hash/parity 检查；缺失时将该实验标为非可复现。",
        ))
    return snapshot, findings


def build_checkpoint_registry(root: Path | None, manifest: dict[str, Any] | None = None) -> CheckpointCapabilityRegistry:
    if root is None or not root.is_dir():
        return CheckpointCapabilityRegistry(status="missing", root=str(root or ""), notes=["未提供本地 checkpoint 根目录；远端 checkpoint 不在本地审计范围。"])
    paths = sorted(root.rglob("agent_*.pt"))
    if not paths:
        return CheckpointCapabilityRegistry(status="missing", root=str(root), notes=["目录存在但没有 agent_*.pt。"])
    best_map: dict[str, Any] = {}
    for candidate in (root / "BEST_CHECKPOINT.json", root.parent / "BEST_CHECKPOINT.json"):
        if candidate.is_file():
            best_map = _read_json(candidate)
            break
    best_name = str(best_map.get("checkpoint") or best_map.get("path") or "")
    capability_rows = {
        str(item.get("checkpoint")): item
        for item in (manifest or {}).get("checkpoint_capabilities", [])
        if isinstance(item, dict)
    }
    records: list[CheckpointCapabilityRecord] = []
    newest = max(paths, key=lambda item: int(_STEP_RE.search(item.name).group(1)) if _STEP_RE.search(item.name) else -1)
    for path in paths[:500]:
        match = _STEP_RE.search(path.name)
        step = int(match.group(1)) if match else 0
        row = capability_rows.get(path.name, {})
        role: Literal["newest", "best", "candidate", "counterexample", "unknown"] = "candidate"
        if path == newest:
            role = "newest"
        if best_name and (path.name == best_name or str(path) == best_name):
            role = "best"
        records.append(CheckpointCapabilityRecord(
            checkpoint=path.name,
            step=step,
            path=str(path),
            size_bytes=path.stat().st_size,
            mtime_ns=path.stat().st_mtime_ns,
            role=role,
            capability_status="proven" if row.get("capabilities") and row.get("evidence") else "unknown",
            capabilities=row.get("capabilities") if isinstance(row.get("capabilities"), dict) else {},
            evidence=[str(item) for item in row.get("evidence", [])],
        ))
    status: EvidenceStatus = "proven" if any(item.capability_status == "proven" for item in records) else "declared"
    return CheckpointCapabilityRegistry(
        status=status,
        root=str(root),
        records=records,
        notes=["checkpoint 文件存在不等于能力已验证；没有诊断/验收证据的记录保持 unknown。"],
    )


def _load_manifest(root: Path, manifest_path: Path | None) -> dict[str, Any]:
    if manifest_path is None:
        return {}
    if not _inside(root, manifest_path):
        raise ValueError("manifest 必须位于审计 root 内")
    return _read_json(manifest_path)


def build_research_audit(
    root: Path = PROJECT_ROOT,
    *,
    manifest_path: Path | None = None,
    checkpoint_root: Path | None = None,
) -> ResearchAuditReport:
    root = root.resolve()
    manifest = _load_manifest(root, manifest_path)
    runtime = build_runtime_execution_proof(root, manifest)
    rewards, reward_findings = _reward_records(root, manifest)
    alignment, alignment_findings = build_metric_alignment(root, manifest)
    gates, gate_findings = build_gate_calibration(root, manifest)
    coverage, coverage_findings = build_command_coverage(manifest)
    optimization, optimization_findings = build_optimization_state(manifest)
    checkpoints = build_checkpoint_registry(checkpoint_root, manifest)
    findings = list(reward_findings + alignment_findings + gate_findings + coverage_findings + optimization_findings)
    if runtime.missing_files:
        findings.append(AuditFinding(
            id="runtime.source_missing", severity="error",
            message=f"关键运行源缺失: {', '.join(runtime.missing_files)}。",
            evidence=runtime.missing_files, remediation="恢复源文件或更新 RobotSources，禁止盲目执行。",
        ))
    if runtime.duplicate_modules:
        findings.append(AuditFinding(
            id="runtime.shadowing_risk", severity="warn",
            message="历史备份存在同名模块，当前本地审计不能证明远端 import 没有 shadowing。",
            evidence=list(runtime.duplicate_modules), remediation="生成远端 RuntimeExecutionProof，记录 sys.path 和 resolved module path。",
        ))
    p0_open = [item for item in findings if item.severity == "error" and item.status == "open"]
    status: AuditStatus = "ready" if not p0_open else "incomplete"
    summary = {
        "reward_terms": len(rewards),
        "metric_alignment_proven": sum(item.status == "proven" for item in alignment),
        "metric_alignment_total": len(alignment),
        "gates_calibrated": sum(item.status == "proven" for item in gates),
        "gates_total": len(gates),
        "command_buckets": len(coverage.rows),
        "optimization_state": optimization.status,
        "checkpoint_records": len(checkpoints.records),
        "open_errors": len(p0_open),
        "open_warnings": sum(item.severity == "warn" and item.status == "open" for item in findings),
        "strict_ready": not p0_open,
    }
    return ResearchAuditReport(
        generated_at=_utc_now(),
        root=str(root),
        status=status,
        runtime_execution=runtime,
        reward_terms=rewards,
        metric_alignment=alignment,
        gate_calibration=gates,
        command_coverage=coverage,
        optimization_state=optimization,
        checkpoints=checkpoints,
        findings=findings,
        summary=summary,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a local read-only RL research audit report.")
    parser.add_argument("--root", default=str(PROJECT_ROOT), help="repository root")
    parser.add_argument("--manifest", default="", help="optional manifest JSON under --root")
    parser.add_argument("--checkpoint-root", default="", help="optional local checkpoint directory")
    parser.add_argument("--output", default="", help="optional JSON output path under --root")
    parser.add_argument("--strict", action="store_true", help="return exit code 2 when P0 evidence is incomplete")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.root).resolve()
    try:
        manifest = Path(args.manifest).resolve() if args.manifest else None
        checkpoint_root = Path(args.checkpoint_root).resolve() if args.checkpoint_root else None
        report = build_research_audit(root, manifest_path=manifest, checkpoint_root=checkpoint_root)
        payload = json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
        if args.output:
            output = Path(args.output).resolve()
            if not _inside(root, output):
                raise ValueError("output 必须位于审计 root 内")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload, encoding="utf-8")
        else:
            sys.stdout.write(payload)
        if args.strict and report.status != "ready":
            return 2
        return 0
    except (OSError, ValueError, UnicodeError) as exc:
        print(f"research audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
