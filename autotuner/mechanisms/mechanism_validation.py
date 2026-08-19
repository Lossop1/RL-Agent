"""Static and numerical checks for synthesized RL mechanisms."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from .mechanism_runtime import MechanismRuntimeError, TensorExpressionEvaluator
from .mechanism_specs import (
    Expression,
    MechanismBundle,
    MechanismPatch,
    apply_mechanism_patch,
)


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    message: str
    severity: Literal["error", "warn"] = "error"
    mechanism_ref: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bundle_ref: str
    bundle_fingerprint: str
    issues: list[ValidationIssue] = Field(default_factory=list)
    checks: dict[str, Any] = Field(default_factory=dict)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [item for item in self.issues if item.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [item for item in self.issues if item.severity == "warn"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            detail = "; ".join(f"{item.code}: {item.message}" for item in self.errors)
            raise ValueError(detail)


class ValidationContext(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    execution_mode: Literal["train", "eval", "sim2sim", "real"] = "train"
    allow_privileged: bool = False
    required_evaluator_refs: tuple[str, ...] = ()
    protected_capabilities: tuple[str, ...] = ()
    runtime_signal_names: frozenset[str] = frozenset()
    probe_values: dict[str, Any] = Field(default_factory=dict)
    failure_probe_values: dict[str, Any] = Field(default_factory=dict)
    min_failure_gradient: float = Field(default=1e-7, ge=0.0)
    forbidden_reward_tokens: tuple[str, ...] = (
        "evaluator", "acceptance", "checkpoint", "curriculum_level", "gate_result", "human_score",
    )


class _SemanticInfo:
    def __init__(self, units: set[str] | None = None):
        self.units = units or set()
        self.signals: set[str] = set()


_DIMENSIONAL_UNIT = re.compile(r"^(?:1|m|s|rad|N|kg)(?:/[A-Za-z0-9]+)?(?:\^[+-]?\d+)?$")


def _is_dimensionless(unit: str) -> bool:
    return unit in {"", "1", "bool"}


def _compatible_units(left: set[str], right: set[str]) -> bool:
    left = {item for item in left if not _is_dimensionless(item)}
    right = {item for item in right if not _is_dimensionless(item)}
    return not left or not right or bool(left & right)


def _expression_semantics(
    expression: Expression,
    units: Mapping[str, str],
    *,
    issues: list[ValidationIssue],
    mechanism_ref: str,
) -> _SemanticInfo:
    if expression.op == "const":
        return _SemanticInfo({"1"})
    if expression.op == "signal":
        unit = units.get(expression.name, "")
        if unit and not _DIMENSIONAL_UNIT.fullmatch(unit):
            issues.append(ValidationIssue(
                code="semantic.unit_unknown", severity="warn", mechanism_ref=mechanism_ref,
                message=f"signal {expression.name} uses unrecognized unit {unit!r}; numerical checks remain required",
            ))
        info = _SemanticInfo({unit or "1"})
        info.signals.add(expression.name)
        return info
    if expression.op == "param":
        return _SemanticInfo({"1"})

    children = [
        _expression_semantics(item, units, issues=issues, mechanism_ref=mechanism_ref)
        for item in expression.args
    ]
    signals = set().union(*(item.signals for item in children)) if children else set()
    additive = {"add", "sub", "minimum", "maximum", "where"}
    comparisons = {"lt", "le", "gt", "ge", "eq", "and", "or"}
    if expression.op in additive:
        meaningful = [item.units for item in children if item.units]
        for left, right in zip(meaningful, meaningful[1:]):
            if not _compatible_units(left, right):
                issues.append(ValidationIssue(
                    code="semantic.unit_mismatch", mechanism_ref=mechanism_ref,
                    message=f"{expression.op} combines incompatible units: {sorted(left)} and {sorted(right)}",
                ))
        info = _SemanticInfo(set.union(*meaningful) if meaningful else {"1"})
    elif expression.op in comparisons:
        meaningful = [item.units for item in children]
        if len(meaningful) >= 2 and not _compatible_units(meaningful[0], meaningful[1]):
            issues.append(ValidationIssue(
                code="semantic.comparison_unit_mismatch", mechanism_ref=mechanism_ref,
                message=f"{expression.op} compares incompatible units: {sorted(meaningful[0])} and {sorted(meaningful[1])}",
            ))
        info = _SemanticInfo({"bool"})
    elif expression.op in {"gaussian", "laplace", "smoothstep"}:
        if len(children) >= 2 and not _compatible_units(children[0].units, children[1].units):
            issues.append(ValidationIssue(
                code="semantic.scale_unit_mismatch", mechanism_ref=mechanism_ref,
                message=f"{expression.op} scale/bounds do not match the measured signal",
            ))
        info = _SemanticInfo({"1"})
    elif expression.op in {"hinge", "clamp"}:
        meaningful = [item.units for item in children]
        if len(meaningful) >= 2 and not _compatible_units(meaningful[0], meaningful[1]):
            issues.append(ValidationIssue(
                code="semantic.threshold_unit_mismatch", mechanism_ref=mechanism_ref,
                message=f"{expression.op} threshold does not match the measured signal",
            ))
        info = _SemanticInfo(meaningful[0] if meaningful else {"1"})
    elif expression.op in {"mul", "div", "safe_ratio"}:
        info = _SemanticInfo({"1"} if expression.op == "safe_ratio" else {"derived"})
    else:
        info = _SemanticInfo({"1"})
    info.signals = signals
    return info


def _all_expressions(bundle: MechanismBundle) -> Iterable[tuple[str, Expression, str]]:
    for term in bundle.rewards:
        yield f"reward:{term.id}", term.expression, "reward"
        if term.activation is not None:
            yield f"reward:{term.id}:activation", term.activation, "activation"
    for metric in bundle.metrics:
        yield f"metric:{metric.id}", metric.expression, "metric"
    for gate in bundle.gates:
        if gate.expression is not None:
            yield f"gate:{gate.id}", gate.expression, "gate"


def _default_probe(bundle: MechanismBundle, context: ValidationContext) -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {}
    probe: dict[str, Any] = {}
    for signal in bundle.signals:
        if signal.name in context.probe_values:
            value = context.probe_values[signal.name]
        else:
            if signal.lower_bound is not None and signal.upper_bound is not None:
                value = (signal.lower_bound + signal.upper_bound) / 2.0
            elif signal.lower_bound is not None:
                value = signal.lower_bound + 0.1
            else:
                value = 0.1
            width = 1 if signal.shape in {"scalar", "per_env"} else 2
            value = [value] * width if signal.shape in {"vector", "matrix"} else value
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim == 0:
            tensor = tensor.repeat(4)
        elif tensor.shape[0] != 4:
            tensor = tensor.unsqueeze(0).repeat(4, *([1] * tensor.ndim))
        probe[signal.name] = tensor
    return probe


def _check_numerics(bundle: MechanismBundle, context: ValidationContext, report: ValidationReport) -> None:
    try:
        import torch
    except ImportError:
        report.issues.append(ValidationIssue(
            code="numeric.torch_missing", severity="warn",
            message="Torch is unavailable; numerical gradient checks were deferred",
        ))
        return
    probe = _default_probe(bundle, context)
    failure = dict(probe)
    failure.update(context.failure_probe_values)
    if not probe:
        report.issues.append(ValidationIssue(code="numeric.probe_missing", message="no executable tensor probe could be built"))
        return
    for name, expression, kind in _all_expressions(bundle):
        signals = {}
        for signal_name, value in (failure if kind == "reward" else probe).items():
            tensor = value if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.float32)
            signals[signal_name] = tensor.clone().detach().requires_grad_(True)
        try:
            evaluator = TensorExpressionEvaluator(signals, {item.name: item.value for item in bundle.parameters})
            result = evaluator.evaluate(expression)
        except Exception as exc:
            report.issues.append(ValidationIssue(
                code="numeric.evaluation_failed", mechanism_ref=name,
                message=f"{type(exc).__name__}: {exc}",
            ))
            continue
        if not bool(torch.isfinite(result).all()):
            report.issues.append(ValidationIssue(code="numeric.non_finite", mechanism_ref=name, message="expression produced non-finite output"))
        refs = expression.signal_refs()
        if refs:
            gradients = torch.autograd.grad(
                result.float().sum(), [signals[ref] for ref in refs], allow_unused=True, retain_graph=False,
            )
            norm = sum(float(grad.detach().abs().sum()) for grad in gradients if grad is not None)
            report.checks[f"gradient:{name}"] = norm
            if norm < context.min_failure_gradient and kind == "reward":
                report.issues.append(ValidationIssue(
                    code="numeric.no_failure_gradient", mechanism_ref=name,
                    message="reward has no measurable gradient in the supplied failure probe",
                    evidence={"gradient_l1": norm, "signals": sorted(refs)},
                ))


def validate_bundle(bundle: MechanismBundle, context: ValidationContext | None = None) -> ValidationReport:
    context = context or ValidationContext()
    report = ValidationReport(bundle_ref=bundle.id, bundle_fingerprint=bundle.fingerprint())
    signals = {item.name: item for item in bundle.signals}
    units = {name: item.unit for name, item in signals.items()}

    required_runtime = set(context.runtime_signal_names)
    for item in bundle.signals:
        if item.source_ref and required_runtime and item.name not in required_runtime:
            report.issues.append(ValidationIssue(
                code="source.runtime_signal_unknown", mechanism_ref=item.name,
                message=f"signal {item.name} is not present in the declared runtime registry",
            ))
        if item.privileged and not context.allow_privileged:
            report.issues.append(ValidationIssue(
                code="source.privileged_signal", mechanism_ref=item.name,
                message="privileged signal is not allowed in this execution context",
            ))
        if context.execution_mode not in item.availability:
            report.issues.append(ValidationIssue(
                code="source.availability_mismatch", mechanism_ref=item.name,
                message=f"signal is unavailable in execution mode {context.execution_mode}",
            ))

    for name, expression, kind in _all_expressions(bundle):
        info = _expression_semantics(expression, units, issues=report.issues, mechanism_ref=name)
        report.checks[f"signals:{name}"] = sorted(info.signals)
        if kind == "reward":
            forbidden = [
                ref for ref in info.signals
                if any(token in ref.lower() for token in context.forbidden_reward_tokens)
            ]
            if forbidden:
                report.issues.append(ValidationIssue(
                    code="anti_cheat.reward_evaluator_signal", mechanism_ref=name,
                    message=f"reward references evaluator/control bookkeeping signals: {', '.join(forbidden)}",
                ))

    metric_ids = {item.id for item in bundle.metrics}
    for gate in bundle.gates:
        if gate.metric_ref not in metric_ids and not gate.expression:
            report.issues.append(ValidationIssue(code="gate.metric_missing", mechanism_ref=gate.id, message="gate metric is missing"))

    for term in bundle.rewards:
        if term.role == "positive_drive" and not term.expression.signal_refs() and term.enabled:
            report.issues.append(ValidationIssue(
                code="reachability.constant_drive", mechanism_ref=term.id,
                message="positive drive is constant and cannot guide behavior",
            ))
    fingerprints: dict[str, str] = {}
    for term in bundle.rewards:
        fp = term.expression.fingerprint()
        if fp in fingerprints:
            report.issues.append(ValidationIssue(
                code="semantic.duplicate_reward", severity="warn", mechanism_ref=term.id,
                message=f"reward expression duplicates {fingerprints[fp]}",
            ))
        fingerprints[fp] = term.id

    train_eval_rewards = [
        item.id for item in bundle.rewards
        if item.enabled and any(
            context.execution_mode not in signals[ref].availability or signals[ref].privileged
            for ref in item.expression.signal_refs() if ref in signals
        )
    ]
    if train_eval_rewards:
        report.issues.append(ValidationIssue(
            code="parity.reward_not_deployable",
            message=f"enabled rewards depend on unavailable/privileged signals: {', '.join(train_eval_rewards)}",
        ))
    report.checks["positive_drive_count"] = sum(item.enabled and item.role == "positive_drive" for item in bundle.rewards)
    report.checks["evaluator_refs"] = list(bundle.evaluator_refs)
    _check_numerics(bundle, context, report)
    return report


def validate_patch(
    baseline: MechanismBundle,
    patch: MechanismPatch,
    context: ValidationContext | None = None,
) -> tuple[MechanismBundle, ValidationReport]:
    context = context or ValidationContext(
        required_evaluator_refs=baseline.evaluator_refs,
        protected_capabilities=patch.protected_capabilities,
    )
    candidate = apply_mechanism_patch(baseline, patch)
    report = validate_bundle(candidate, context)
    required_evaluators = set(context.required_evaluator_refs) | set(patch.required_evaluators)
    missing_evaluators = sorted(required_evaluators - set(candidate.evaluator_refs))
    if missing_evaluators:
        report.issues.append(ValidationIssue(
            code="protection.evaluator_removed",
            message=f"candidate removed required independent evaluators: {', '.join(missing_evaluators)}",
        ))
    baseline_owned_metrics = {item.id for item in baseline.metrics if item.evaluator_owned}
    baseline_owned_gates = {item.id for item in baseline.gates if item.evaluator_owned}
    candidate_ids = {item.id for item in candidate.metrics if item.evaluator_owned} | {item.id for item in candidate.gates if item.evaluator_owned}
    if not (baseline_owned_metrics | baseline_owned_gates) <= candidate_ids:
        report.issues.append(ValidationIssue(
            code="protection.evaluator_mechanism_changed",
            message="candidate removed an evaluator-owned metric or gate",
        ))
    if not set(patch.protected_capabilities):
        report.issues.append(ValidationIssue(code="protection.capability_list_empty", message="candidate has no protected capabilities"))
    return candidate, report
