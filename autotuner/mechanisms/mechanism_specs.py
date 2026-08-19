"""Declarative, versioned specifications for live RL research mechanisms.

Reward terms, telemetry metrics, and curriculum/safety gates share one small
expression language.  This lets the system create new mathematical semantics
without granting generated artifacts arbitrary Python execution privileges.
"""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


MECHANISM_SCHEMA_VERSION = "rl-agent.mechanisms/v1"
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]{0,127}$")

ExpressionOp = Literal[
    "const", "signal", "param",
    "add", "sub", "mul", "div", "neg", "abs", "square", "sqrt", "exp", "log",
    "minimum", "maximum", "clamp", "where",
    "lt", "le", "gt", "ge", "eq", "and", "or", "not",
    "mean", "sum", "min_reduce", "max_reduce", "norm",
    "gaussian", "laplace", "hinge", "smoothstep", "safe_ratio",
]


def _canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python", exclude_none=True)

    def normalize(item: Any) -> Any:
        if isinstance(item, BaseModel):
            return normalize(item.model_dump(mode="python", exclude_none=True))
        if isinstance(item, dict):
            return {str(key): normalize(child) for key, child in item.items()}
        if isinstance(item, (set, frozenset)):
            children = [normalize(child) for child in item]
            return sorted(
                children,
                key=lambda child: json.dumps(
                    child, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                ),
            )
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        return item

    return json.dumps(normalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class Expression(BaseModel):
    """An intentionally small tensor expression AST."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    op: ExpressionOp
    args: tuple["Expression", ...] = ()
    value: float | bool | None = None
    name: str = ""
    axis: int | tuple[int, ...] | None = None
    keepdim: bool = False

    @model_validator(mode="after")
    def _validate_shape(self) -> "Expression":
        nullary = {"const", "signal", "param"}
        unary = {"neg", "abs", "square", "sqrt", "exp", "log", "not", "mean", "sum", "min_reduce", "max_reduce", "norm"}
        binary = {"sub", "div", "lt", "le", "gt", "ge", "eq", "gaussian", "laplace", "hinge", "safe_ratio"}
        variadic = {"add", "mul", "minimum", "maximum", "and", "or"}
        expected = None
        if self.op in nullary:
            expected = 0
        elif self.op in unary:
            expected = 1
        elif self.op in binary:
            expected = 2
        elif self.op == "clamp" or self.op == "where" or self.op == "smoothstep":
            expected = 3
        elif self.op in variadic and len(self.args) < 2:
            raise ValueError(f"{self.op} requires at least two arguments")
        if expected is not None and len(self.args) != expected:
            raise ValueError(f"{self.op} requires {expected} arguments, got {len(self.args)}")
        if self.op == "const":
            if self.value is None:
                raise ValueError("const requires value")
            if self.name or self.axis is not None:
                raise ValueError("const cannot declare name or axis")
        elif self.op in {"signal", "param"}:
            if not self.name or not _IDENTIFIER.fullmatch(self.name):
                raise ValueError(f"{self.op} requires a valid name")
            if self.value is not None or self.axis is not None:
                raise ValueError(f"{self.op} cannot declare value or axis")
        elif self.value is not None or self.name:
            raise ValueError(f"{self.op} cannot declare value or name")
        if self.axis is not None and self.op not in {"mean", "sum", "min_reduce", "max_reduce", "norm"}:
            raise ValueError(f"axis is not supported by {self.op}")
        return self

    def signal_refs(self) -> set[str]:
        refs = {self.name} if self.op == "signal" else set()
        for arg in self.args:
            refs.update(arg.signal_refs())
        return refs

    def parameter_refs(self) -> set[str]:
        refs = {self.name} if self.op == "param" else set()
        for arg in self.args:
            refs.update(arg.parameter_refs())
        return refs

    def fingerprint(self) -> str:
        return content_fingerprint(self)

    @classmethod
    def const(cls, value: float | bool) -> "Expression":
        return cls(op="const", value=value)

    @classmethod
    def signal(cls, name: str) -> "Expression":
        return cls(op="signal", name=name)

    @classmethod
    def param(cls, name: str) -> "Expression":
        return cls(op="param", name=name)


class SignalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    unit: str = "1"
    frame: Literal["none", "body", "world", "command", "joint", "terrain"] = "none"
    shape: Literal["scalar", "per_env", "per_leg", "per_joint", "vector", "matrix"] = "per_env"
    dtype: Literal["float", "bool"] = "float"
    availability: set[Literal["train", "eval", "sim2sim", "real"]] = Field(default_factory=lambda: {"train", "eval"})
    privileged: bool = False
    lower_bound: float | None = None
    upper_bound: float | None = None
    source_ref: str = ""
    selector: int | tuple[int, ...] | None = None

    @model_validator(mode="after")
    def _validate_signal(self) -> "SignalSpec":
        if not _IDENTIFIER.fullmatch(self.name):
            raise ValueError(f"invalid signal name: {self.name}")
        if not self.description.strip():
            raise ValueError("signal description is required")
        if self.lower_bound is not None and self.upper_bound is not None and self.lower_bound > self.upper_bound:
            raise ValueError("signal lower_bound exceeds upper_bound")
        if self.source_ref and not re.fullmatch(r"(?:input|component)\.[A-Za-z][A-Za-z0-9_.-]*", self.source_ref):
            raise ValueError("signal source_ref must use input.<path> or component.<path>")
        if isinstance(self.selector, tuple) and not self.selector:
            raise ValueError("signal selector cannot be empty")
        return self


class ParameterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    value: float
    lower_bound: float
    upper_bound: float
    unit: str = "1"
    tunable: bool = True

    @model_validator(mode="after")
    def _validate_parameter(self) -> "ParameterSpec":
        if not _IDENTIFIER.fullmatch(self.name):
            raise ValueError(f"invalid parameter name: {self.name}")
        if self.lower_bound > self.upper_bound:
            raise ValueError("parameter lower_bound exceeds upper_bound")
        if not self.lower_bound <= self.value <= self.upper_bound:
            raise ValueError("parameter value is outside declared bounds")
        return self


class RewardTermSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    name: str
    role: Literal["positive_drive", "bridge", "guard"]
    expression: Expression
    weight: float = Field(default=1.0, ge=0.0)
    polarity: Literal["reward", "penalty"] = "reward"
    activation: Expression | None = None
    intended_effect: str
    failure_region: str
    success_region: str
    telemetry_key: str = ""
    reward_group: Literal["track", "turn", "gait", "stab", "style"] = "track"
    enabled: bool = True

    @model_validator(mode="after")
    def _validate_reward(self) -> "RewardTermSpec":
        if not _IDENTIFIER.fullmatch(self.id):
            raise ValueError(f"invalid reward id: {self.id}")
        if not self.intended_effect.strip() or not self.failure_region.strip() or not self.success_region.strip():
            raise ValueError("reward semantics require intended_effect, failure_region, and success_region")
        if self.activation is not None and self.activation.op == "const" and not isinstance(self.activation.value, bool):
            raise ValueError("constant reward activation must be boolean")
        return self


class MetricSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    name: str
    expression: Expression
    aggregation: Literal["mean", "min", "max", "p05", "p50", "p95", "rate", "sum"] = "mean"
    unit: str = "1"
    window_steps: int = Field(default=1, ge=1)
    intended_reading: str
    evaluator_owned: bool = False
    enabled: bool = True

    @model_validator(mode="after")
    def _validate_metric(self) -> "MetricSpec":
        if not _IDENTIFIER.fullmatch(self.id):
            raise ValueError(f"invalid metric id: {self.id}")
        if not self.intended_reading.strip():
            raise ValueError("metric intended_reading is required")
        return self


class GateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    name: str
    metric_ref: str = ""
    expression: Expression | None = None
    comparator: Literal["gt", "ge", "lt", "le", "between"]
    threshold: float | tuple[float, float]
    min_samples: int = Field(default=1, ge=1)
    consecutive_windows: int = Field(default=1, ge=1)
    hysteresis: float = Field(default=0.0, ge=0.0)
    action: Literal["advance", "hold", "rollback", "stop", "evaluate", "alert"] = "hold"
    scope: Literal["curriculum", "safety", "experiment", "monitor"] = "monitor"
    rationale: str
    evaluator_owned: bool = False
    enabled: bool = True

    @model_validator(mode="after")
    def _validate_gate(self) -> "GateSpec":
        if not _IDENTIFIER.fullmatch(self.id):
            raise ValueError(f"invalid gate id: {self.id}")
        if bool(self.metric_ref) == bool(self.expression is not None):
            raise ValueError("gate requires exactly one of metric_ref or expression")
        if self.comparator == "between":
            if not isinstance(self.threshold, tuple) or len(self.threshold) != 2 or self.threshold[0] > self.threshold[1]:
                raise ValueError("between gate requires an ordered two-value threshold")
        elif isinstance(self.threshold, tuple):
            raise ValueError(f"{self.comparator} gate requires a scalar threshold")
        if not self.rationale.strip():
            raise ValueError("gate rationale is required")
        return self


class MechanismBundle(BaseModel):
    """A complete executable mechanism set for one training candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = MECHANISM_SCHEMA_VERSION
    id: str
    version: int = Field(default=1, ge=1)
    status: Literal["draft", "candidate", "approved", "retired"] = "draft"
    contract_ref: str
    baseline_ref: str = ""
    supersedes: tuple[str, ...] = ()
    signals: tuple[SignalSpec, ...] = ()
    parameters: tuple[ParameterSpec, ...] = ()
    rewards: tuple[RewardTermSpec, ...] = ()
    metrics: tuple[MetricSpec, ...] = ()
    gates: tuple[GateSpec, ...] = ()
    evaluator_refs: tuple[str, ...] = ()
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_bundle(self) -> "MechanismBundle":
        if not _IDENTIFIER.fullmatch(self.id):
            raise ValueError(f"invalid mechanism bundle id: {self.id}")
        for label, values, key in (
            ("signal", self.signals, "name"),
            ("parameter", self.parameters, "name"),
            ("reward", self.rewards, "id"),
            ("metric", self.metrics, "id"),
            ("gate", self.gates, "id"),
        ):
            ids = [getattr(value, key) for value in values]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {label} identifiers")
        signal_ids = {item.name for item in self.signals}
        parameter_ids = {item.name for item in self.parameters}
        metric_ids = {item.id for item in self.metrics}
        expressions = [item.expression for item in self.rewards] + [item.expression for item in self.metrics]
        expressions += [item.activation for item in self.rewards if item.activation is not None]
        expressions += [item.expression for item in self.gates if item.expression is not None]
        unknown_signals = sorted(set().union(*(expr.signal_refs() for expr in expressions)) - signal_ids) if expressions else []
        unknown_parameters = sorted(set().union(*(expr.parameter_refs() for expr in expressions)) - parameter_ids) if expressions else []
        if unknown_signals:
            raise ValueError(f"expressions reference unknown signals: {', '.join(unknown_signals)}")
        if unknown_parameters:
            raise ValueError(f"expressions reference unknown parameters: {', '.join(unknown_parameters)}")
        missing_metrics = sorted({gate.metric_ref for gate in self.gates if gate.metric_ref} - metric_ids)
        if missing_metrics:
            raise ValueError(f"gates reference unknown metrics: {', '.join(missing_metrics)}")
        metrics_by_id = {item.id: item for item in self.metrics}
        unreachable_gates = [
            gate.id
            for gate in self.gates
            if gate.metric_ref
            and metrics_by_id[gate.metric_ref].window_steps < gate.min_samples
        ]
        if unreachable_gates:
            raise ValueError(
                "gate min_samples exceeds its metric history window: "
                + ", ".join(unreachable_gates)
            )
        if self.rewards and not any(item.enabled and item.role == "positive_drive" for item in self.rewards):
            raise ValueError("an executable reward bundle requires at least one enabled positive drive")
        return self

    def fingerprint(self) -> str:
        return content_fingerprint(self)


class MechanismOperation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["add", "replace", "remove"]
    target_kind: Literal["signal", "parameter", "reward", "metric", "gate"]
    target_id: str
    value: dict[str, Any] | None = None
    reason: str

    @model_validator(mode="after")
    def _validate_operation(self) -> "MechanismOperation":
        if not _IDENTIFIER.fullmatch(self.target_id):
            raise ValueError(f"invalid operation target: {self.target_id}")
        if self.action in {"add", "replace"} and self.value is None:
            raise ValueError(f"{self.action} operation requires value")
        if self.action == "remove" and self.value is not None:
            raise ValueError("remove operation cannot include value")
        if not self.reason.strip():
            raise ValueError("operation reason is required")
        return self


class MechanismPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    baseline_bundle_ref: str
    baseline_fingerprint: str
    candidate_bundle_id: str
    problem_ref: str
    hypothesis_refs: tuple[str, ...] = ()
    operations: tuple[MechanismOperation, ...]
    expected_effects: tuple[str, ...]
    protected_capabilities: tuple[str, ...]
    required_evaluators: tuple[str, ...]
    generated_by: str
    risk_tier: Literal["low", "medium", "high"] = "medium"

    @model_validator(mode="after")
    def _validate_patch(self) -> "MechanismPatch":
        if not self.operations:
            raise ValueError("mechanism patch requires at least one operation")
        if not self.expected_effects:
            raise ValueError("mechanism patch requires expected_effects")
        if not self.protected_capabilities or not self.required_evaluators:
            raise ValueError("mechanism patch must preserve capabilities and independent evaluators")
        return self

    def fingerprint(self) -> str:
        return content_fingerprint(self)


_COLLECTIONS = {
    "signal": ("signals", SignalSpec, "name"),
    "parameter": ("parameters", ParameterSpec, "name"),
    "reward": ("rewards", RewardTermSpec, "id"),
    "metric": ("metrics", MetricSpec, "id"),
    "gate": ("gates", GateSpec, "id"),
}


def apply_mechanism_patch(baseline: MechanismBundle, patch: MechanismPatch) -> MechanismBundle:
    """Apply a deterministic patch after checking its exact baseline."""
    if patch.baseline_bundle_ref != baseline.id:
        raise ValueError(f"patch baseline ref {patch.baseline_bundle_ref!r} does not match {baseline.id!r}")
    actual_fingerprint = baseline.fingerprint()
    if patch.baseline_fingerprint != actual_fingerprint:
        raise ValueError("patch baseline fingerprint does not match; candidate must be regenerated")

    data = deepcopy(baseline.model_dump(mode="json"))
    for operation in patch.operations:
        collection_name, model, key = _COLLECTIONS[operation.target_kind]
        collection = list(data[collection_name])
        positions = {str(item[key]): index for index, item in enumerate(collection)}
        exists = operation.target_id in positions
        if operation.action == "add":
            if exists:
                raise ValueError(f"cannot add existing {operation.target_kind}: {operation.target_id}")
            value = model.model_validate(operation.value).model_dump(mode="json")
            if str(value[key]) != operation.target_id:
                raise ValueError("operation target_id does not match value identifier")
            collection.append(value)
        elif operation.action == "replace":
            if not exists:
                raise ValueError(f"cannot replace missing {operation.target_kind}: {operation.target_id}")
            value = model.model_validate(operation.value).model_dump(mode="json")
            if str(value[key]) != operation.target_id:
                raise ValueError("operation target_id does not match value identifier")
            collection[positions[operation.target_id]] = value
        else:
            if not exists:
                raise ValueError(f"cannot remove missing {operation.target_kind}: {operation.target_id}")
            del collection[positions[operation.target_id]]
        data[collection_name] = collection

    data.update({
        "id": patch.candidate_bundle_id,
        "version": baseline.version + 1,
        "status": "candidate",
        "baseline_ref": baseline.id,
        "supersedes": [baseline.id],
        "provenance": {
            **baseline.provenance,
            "patch_id": patch.id,
            "patch_fingerprint": patch.fingerprint(),
            "generated_by": patch.generated_by,
        },
    })
    return MechanismBundle.model_validate(data)
