"""Torch runtime for declarative reward, metric, and gate mechanisms."""
from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .mechanism_specs import Expression, GateSpec, MechanismBundle, SignalSpec


class MechanismRuntimeError(RuntimeError):
    pass


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - only minimal web installs lack torch
        raise MechanismRuntimeError("mechanism execution requires torch") from exc
    return torch


def _nested_value(root: Any, dotted: str) -> Any:
    value = root
    for part in dotted.split("."):
        if isinstance(value, Mapping):
            if part not in value:
                raise MechanismRuntimeError(f"missing mapping signal path: {dotted}")
            value = value[part]
        else:
            if not hasattr(value, part):
                raise MechanismRuntimeError(f"missing object signal path: {dotted}")
            value = getattr(value, part)
    return value


def _resolve_signal(spec: SignalSpec, *, inputs: Any, components: Mapping[str, Any], supplied: Mapping[str, Any]) -> Any:
    if spec.name in supplied:
        value = supplied[spec.name]
    elif spec.source_ref.startswith("input."):
        value = _nested_value(inputs, spec.source_ref[len("input."):])
    elif spec.source_ref.startswith("component."):
        value = _nested_value(components, spec.source_ref[len("component."):])
    else:
        raise MechanismRuntimeError(f"signal {spec.name!r} has no runtime source")
    if spec.selector is not None:
        try:
            value = value[..., spec.selector]
        except Exception as exc:
            raise MechanismRuntimeError(f"cannot apply selector for signal {spec.name}: {spec.selector}") from exc
    return value


class TensorExpressionEvaluator:
    """Evaluate the safe expression AST with Torch tensor semantics."""

    def __init__(self, signals: Mapping[str, Any], parameters: Mapping[str, float]):
        self.torch = _torch()
        self.signals = signals
        self.parameters = parameters
        self.reference = next((value for value in signals.values() if self.torch.is_tensor(value)), None)
        if self.reference is None:
            raise MechanismRuntimeError("at least one tensor signal is required")

    def _constant(self, value: float | bool):
        if isinstance(value, bool):
            return self.torch.as_tensor(value, dtype=self.torch.bool, device=self.reference.device)
        return self.torch.as_tensor(value, dtype=self.reference.dtype, device=self.reference.device)

    def evaluate(self, expression: Expression):
        torch = self.torch
        op = expression.op
        if op == "const":
            return self._constant(expression.value)
        if op == "signal":
            if expression.name not in self.signals:
                raise MechanismRuntimeError(f"runtime signal is missing: {expression.name}")
            value = self.signals[expression.name]
            if torch.is_tensor(value):
                return value
            return torch.as_tensor(value, dtype=self.reference.dtype, device=self.reference.device)
        if op == "param":
            if expression.name not in self.parameters:
                raise MechanismRuntimeError(f"runtime parameter is missing: {expression.name}")
            return self._constant(float(self.parameters[expression.name]))

        args = [self.evaluate(arg) for arg in expression.args]
        if op == "add":
            return sum(args[1:], args[0])
        if op == "sub":
            return args[0] - args[1]
        if op == "mul":
            result = args[0]
            for value in args[1:]:
                result = result * value
            return result
        if op == "div":
            return args[0] / args[1]
        if op == "safe_ratio":
            epsilon = torch.as_tensor(1e-8, dtype=args[1].dtype, device=args[1].device)
            denominator = torch.where(args[1].abs() < epsilon, torch.copysign(epsilon, args[1] + epsilon), args[1])
            return args[0] / denominator
        if op == "neg":
            return -args[0]
        if op == "abs":
            return args[0].abs()
        if op == "square":
            return args[0].square()
        if op == "sqrt":
            return torch.sqrt(torch.clamp(args[0], min=0.0))
        if op == "exp":
            return torch.exp(args[0])
        if op == "log":
            return torch.log(torch.clamp(args[0], min=1e-12))
        if op == "minimum":
            result = args[0]
            for value in args[1:]:
                result = torch.minimum(result, value)
            return result
        if op == "maximum":
            result = args[0]
            for value in args[1:]:
                result = torch.maximum(result, value)
            return result
        if op == "clamp":
            return torch.maximum(torch.minimum(args[0], args[2]), args[1])
        if op == "where":
            return torch.where(args[0].bool(), args[1], args[2])
        if op in {"lt", "le", "gt", "ge", "eq"}:
            return {
                "lt": torch.lt, "le": torch.le, "gt": torch.gt,
                "ge": torch.ge, "eq": torch.eq,
            }[op](args[0], args[1])
        if op == "and":
            result = args[0].bool()
            for value in args[1:]:
                result = torch.logical_and(result, value.bool())
            return result
        if op == "or":
            result = args[0].bool()
            for value in args[1:]:
                result = torch.logical_or(result, value.bool())
            return result
        if op == "not":
            return torch.logical_not(args[0].bool())
        if op in {"mean", "sum", "min_reduce", "max_reduce", "norm"}:
            axis = expression.axis
            if op == "mean":
                return args[0].mean(dim=axis, keepdim=expression.keepdim)
            if op == "sum":
                return args[0].sum(dim=axis, keepdim=expression.keepdim)
            if op == "norm":
                return torch.linalg.vector_norm(args[0], dim=axis, keepdim=expression.keepdim)
            if axis is None:
                result = args[0].amin() if op == "min_reduce" else args[0].amax()
                return result
            reducer = torch.amin if op == "min_reduce" else torch.amax
            return reducer(args[0], dim=axis, keepdim=expression.keepdim)
        if op == "gaussian":
            sigma = torch.clamp(args[1].abs(), min=1e-9)
            return torch.exp(-torch.square(args[0] / sigma))
        if op == "laplace":
            scale = torch.clamp(args[1].abs(), min=1e-9)
            return torch.exp(-args[0].abs() / scale)
        if op == "hinge":
            return torch.clamp(args[0] - args[1], min=0.0)
        if op == "smoothstep":
            width = torch.clamp(args[2] - args[1], min=1e-9)
            x = torch.clamp((args[0] - args[1]) / width, 0.0, 1.0)
            return x * x * (3.0 - 2.0 * x)
        raise MechanismRuntimeError(f"unsupported expression op: {op}")


def _batch_vector(value: Any, *, batch_size: int, reference: Any, label: str):
    torch = _torch()
    value = value if torch.is_tensor(value) else torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if value.ndim == 0:
        value = value.expand(batch_size)
    if value.ndim != 1 or value.shape[0] != batch_size:
        raise MechanismRuntimeError(f"{label} must evaluate to one scalar per environment, got {tuple(value.shape)}")
    return value


def _aggregate(value: Any, kind: str) -> float:
    torch = _torch()
    flat = value.float().reshape(-1)
    if flat.numel() == 0:
        return math.nan
    if kind == "mean" or kind == "rate":
        result = flat.mean()
    elif kind == "min":
        result = flat.min()
    elif kind == "max":
        result = flat.max()
    elif kind == "sum":
        result = flat.sum()
    elif kind == "p05":
        result = torch.quantile(flat, 0.05)
    elif kind == "p50":
        result = torch.quantile(flat, 0.50)
    elif kind == "p95":
        result = torch.quantile(flat, 0.95)
    else:  # pragma: no cover - Pydantic prevents this
        raise MechanismRuntimeError(f"unsupported metric aggregation: {kind}")
    return float(result.detach().cpu())


@dataclass
class MechanismStepResult:
    reward_components: dict[str, Any] = field(default_factory=dict)
    reward_groups: dict[str, str] = field(default_factory=dict)
    metric_tensors: dict[str, Any] = field(default_factory=dict)
    metric_values: dict[str, float] = field(default_factory=dict)
    gate_results: dict[str, dict[str, Any]] = field(default_factory=dict)


class MechanismRuntime:
    """Stateful metric-window and gate evaluator for one bundle."""

    def __init__(self, bundle: MechanismBundle):
        self.bundle = bundle
        self.parameters = {item.name: item.value for item in bundle.parameters}
        self._histories = {item.id: deque(maxlen=item.window_steps) for item in bundle.metrics}
        self._gate_streaks = {item.id: 0 for item in bundle.gates}
        self._gate_passed = {item.id: False for item in bundle.gates}

    def _gate_pass(self, gate: GateSpec, value: float) -> bool:
        threshold = gate.threshold
        was_passed = self._gate_passed[gate.id]
        if gate.comparator == "between":
            low, high = threshold
            margin = gate.hysteresis if was_passed else 0.0
            return low - margin <= value <= high + margin
        threshold = float(threshold)
        if gate.comparator in {"ge", "gt"}:
            effective = threshold - gate.hysteresis if was_passed else threshold
            return value >= effective if gate.comparator == "ge" else value > effective
        effective = threshold + gate.hysteresis if was_passed else threshold
        return value <= effective if gate.comparator == "le" else value < effective

    def evaluate_step(
        self,
        *,
        inputs: Any = None,
        components: Mapping[str, Any] | None = None,
        signals: Mapping[str, Any] | None = None,
    ) -> MechanismStepResult:
        torch = _torch()
        components = components or {}
        supplied = signals or {}
        resolved = {
            spec.name: _resolve_signal(spec, inputs=inputs, components=components, supplied=supplied)
            for spec in self.bundle.signals
        }
        evaluator = TensorExpressionEvaluator(resolved, self.parameters)
        reference = evaluator.reference
        batch_size = next((int(value.shape[0]) for value in resolved.values() if torch.is_tensor(value) and value.ndim > 0), 1)
        result = MechanismStepResult()

        for term in self.bundle.rewards:
            if not term.enabled:
                continue
            value = _batch_vector(
                evaluator.evaluate(term.expression), batch_size=batch_size,
                reference=reference, label=f"reward {term.id}",
            )
            if term.activation is not None:
                activation = _batch_vector(
                    evaluator.evaluate(term.activation), batch_size=batch_size,
                    reference=reference, label=f"activation {term.id}",
                )
                value = value * activation.to(dtype=value.dtype)
            sign = 1.0 if term.polarity == "reward" else -1.0
            value = sign * term.weight * value
            if not bool(torch.isfinite(value).all()):
                raise MechanismRuntimeError(f"reward {term.id} produced non-finite values")
            key = term.telemetry_key or f"dynamic/{term.id}"
            result.reward_components[key] = value
            result.reward_groups[key] = term.reward_group

        for metric in self.bundle.metrics:
            if not metric.enabled:
                continue
            tensor = evaluator.evaluate(metric.expression)
            if not bool(torch.isfinite(tensor).all()):
                raise MechanismRuntimeError(f"metric {metric.id} produced non-finite values")
            scalar = _aggregate(tensor, metric.aggregation)
            result.metric_tensors[metric.id] = tensor
            result.metric_values[metric.id] = scalar
            self._histories[metric.id].append(scalar)

        for gate in self.bundle.gates:
            if not gate.enabled:
                continue
            if gate.metric_ref:
                history = self._histories[gate.metric_ref]
                samples = len(history)
                value = sum(history) / samples if samples else math.nan
            else:
                value = _aggregate(evaluator.evaluate(gate.expression), "mean")
                samples = 1
            ready = samples >= gate.min_samples and math.isfinite(value)
            passed_now = ready and self._gate_pass(gate, value)
            self._gate_streaks[gate.id] = self._gate_streaks[gate.id] + 1 if passed_now else 0
            passed = passed_now and self._gate_streaks[gate.id] >= gate.consecutive_windows
            self._gate_passed[gate.id] = passed
            result.gate_results[gate.id] = {
                "ready": ready,
                "passed": passed,
                "value": value,
                "samples": samples,
                "streak": self._gate_streaks[gate.id],
                "required_streak": gate.consecutive_windows,
                "action": gate.action if passed else "hold",
                "scope": gate.scope,
            }
        return result


def load_mechanism_bundle(path: str | Path) -> MechanismBundle:
    return MechanismBundle.model_validate_json(Path(path).read_text(encoding="utf-8"))


_RUNTIME_CACHE: tuple[str, int, int, MechanismRuntime] | None = None


def load_mechanism_runtime(path: str | Path, *, reload_if_changed: bool = False) -> MechanismRuntime:
    """Load once per process; optional stat checks support interactive research."""
    global _RUNTIME_CACHE
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    key = (str(resolved), int(stat.st_mtime_ns), int(stat.st_size))
    if _RUNTIME_CACHE is not None:
        cached_key = _RUNTIME_CACHE[:3]
        if cached_key[0] == key[0] and (not reload_if_changed or cached_key == key):
            return _RUNTIME_CACHE[3]
    runtime = MechanismRuntime(load_mechanism_bundle(resolved))
    _RUNTIME_CACHE = (*key, runtime)
    return runtime


def runtime_from_environment() -> MechanismRuntime | None:
    path = os.environ.get("TAILI_MECHANISM_BUNDLE", "").strip()
    if not path:
        return None
    reload_if_changed = os.environ.get("TAILI_MECHANISM_RELOAD", "0") == "1"
    return load_mechanism_runtime(path, reload_if_changed=reload_if_changed)


def clear_runtime_cache() -> None:
    global _RUNTIME_CACHE
    _RUNTIME_CACHE = None
