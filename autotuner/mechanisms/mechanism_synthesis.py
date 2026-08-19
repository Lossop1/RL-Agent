"""Hypothesis-driven synthesis of executable RL mechanism candidates."""
from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .mechanism_specs import (
    Expression,
    GateSpec,
    MechanismBundle,
    MechanismOperation,
    MechanismPatch,
    MetricSpec,
    RewardTermSpec,
    SignalSpec,
    apply_mechanism_patch,
    content_fingerprint,
)


class CapabilityGap(BaseModel):
    """A measured gap, expressed in a signal the runtime can consume."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    capability: str
    symptom: str
    signal: SignalSpec
    desired: Literal["lower", "higher", "between"]
    good: float | tuple[float, float]
    bad: float | tuple[float, float]
    severity: float = Field(default=0.5, ge=0.0, le=1.0)
    capability_kind: Literal["task", "quality", "safety"] = "quality"
    activation: Expression | None = None
    hypothesis_ref: str
    evidence_refs: tuple[str, ...] = ()
    use_for_curriculum: bool = False
    reward_group: Literal["track", "turn", "gait", "stab", "style"] = "stab"
    signal_reduction: Literal["none", "mean", "max", "sum", "norm"] = "none"

    @model_validator(mode="after")
    def _validate_ranges(self) -> "CapabilityGap":
        if self.desired == "between":
            if not isinstance(self.good, tuple) or not isinstance(self.bad, tuple):
                raise ValueError("between gap requires tuple good and bad ranges")
            if not (self.bad[0] <= self.good[0] <= self.good[1] <= self.bad[1]):
                raise ValueError("between ranges must satisfy bad_low <= good range <= bad_high")
        elif isinstance(self.good, tuple) or isinstance(self.bad, tuple):
            raise ValueError(f"{self.desired} gap requires scalar good and bad values")
        elif self.desired == "lower" and self.good >= self.bad:
            raise ValueError("lower gap requires good < bad")
        elif self.desired == "higher" and self.good <= self.bad:
            raise ValueError("higher gap requires good > bad")
        if not self.hypothesis_ref or not self.symptom.strip():
            raise ValueError("gap requires symptom and hypothesis_ref")
        if self.signal.shape in {"vector", "matrix", "per_leg", "per_joint"} and self.signal_reduction == "none":
            raise ValueError("non-scalar gap signals require an explicit signal_reduction")
        return self


class MechanismIntent(BaseModel):
    """Open-ended AST operation produced by a reasoner or human researcher."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["add", "replace", "remove"]
    target_kind: Literal["signal", "parameter", "reward", "metric", "gate"]
    target_id: str
    value: dict[str, Any] | None = None
    causal_rationale: str
    expected_effect: str
    falsification: str


class SynthesisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    id: str
    problem_ref: str
    problem_statement: str
    baseline: MechanismBundle
    gaps: tuple[CapabilityGap, ...] = ()
    custom_intents: tuple[MechanismIntent, ...] = ()
    protected_capabilities: tuple[str, ...]
    required_evaluators: tuple[str, ...]
    no_repeat_fingerprints: frozenset[str] = frozenset()
    max_candidates: int = Field(default=8, ge=1, le=32)
    generated_by: str = "mechanism_synthesizer"

    @model_validator(mode="after")
    def _validate_request(self) -> "SynthesisRequest":
        if not self.gaps and not self.custom_intents:
            raise ValueError("synthesis request needs measured gaps or custom intents")
        if not self.protected_capabilities or not self.required_evaluators:
            raise ValueError("synthesis requires protected capabilities and independent evaluators")
        return self


class SynthesisCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    variant: str
    patch: MechanismPatch
    candidate_bundle_fingerprint: str
    rationale: str
    predictions: tuple[str, ...]
    falsification: tuple[str, ...]
    novelty: tuple[str, ...]


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return value[:48] or "mechanism"


def _gap_signal_expression(gap: CapabilityGap) -> Expression:
    signal = Expression.signal(gap.signal.name)
    reductions = {
        "mean": "mean", "max": "max_reduce", "sum": "sum", "norm": "norm",
    }
    if gap.signal_reduction != "none":
        return Expression(op=reductions[gap.signal_reduction], args=(signal,), axis=-1)
    return signal


def _gap_quality(gap: CapabilityGap) -> tuple[Expression, Expression, Expression, float, Literal["ge", "le", "between"], float | tuple[float, float]]:
    signal = _gap_signal_expression(gap)
    if gap.desired == "lower":
        good, bad = float(gap.good), float(gap.bad)
        excess = Expression(op="hinge", args=(signal, Expression.const(good)))
        scale = max(bad - good, 1e-6)
        quality = Expression(op="laplace", args=(excess, Expression.const(scale)))
        return signal, quality, excess, scale, "le", good
    if gap.desired == "higher":
        good, bad = float(gap.good), float(gap.bad)
        deficit = Expression(op="maximum", args=(Expression(op="sub", args=(Expression.const(good), signal)), Expression.const(0.0)))
        scale = max(good - bad, 1e-6)
        quality = Expression(op="laplace", args=(deficit, Expression.const(scale)))
        return signal, quality, deficit, scale, "ge", good
    good_low, good_high = gap.good
    bad_low, bad_high = gap.bad
    low_fault = Expression(op="maximum", args=(Expression(op="sub", args=(Expression.const(good_low), signal)), Expression.const(0.0)))
    high_fault = Expression(op="maximum", args=(Expression(op="sub", args=(signal, Expression.const(good_high))), Expression.const(0.0)))
    fault = Expression(op="add", args=(low_fault, high_fault))
    scale = max(max(good_low - bad_low, bad_high - good_high), 1e-6)
    quality = Expression(op="laplace", args=(fault, Expression.const(scale)))
    return signal, quality, fault, scale, "between", (good_low, good_high)


class MechanismSynthesizer:
    """Generate competing drive/guard candidates rather than one opaque edit."""

    def _patch(
        self,
        request: SynthesisRequest,
        *,
        variant: str,
        operations: list[MechanismOperation],
        expected_effects: list[str],
        hypothesis_refs: list[str],
        risk_tier: Literal["low", "medium", "high"] = "medium",
    ) -> MechanismPatch:
        seed = {
            "request": request.id,
            "variant": variant,
            "operations": [item.model_dump(mode="json") for item in operations],
            "baseline": request.baseline.fingerprint(),
        }
        suffix = content_fingerprint(seed)[:12]
        return MechanismPatch(
            id=f"patch:{_slug(request.id)}:{_slug(variant)}:{suffix}",
            baseline_bundle_ref=request.baseline.id,
            baseline_fingerprint=request.baseline.fingerprint(),
            candidate_bundle_id=f"{request.baseline.id}:candidate:{suffix}",
            problem_ref=request.problem_ref,
            hypothesis_refs=tuple(dict.fromkeys(hypothesis_refs)),
            operations=tuple(operations),
            expected_effects=tuple(expected_effects),
            protected_capabilities=request.protected_capabilities,
            required_evaluators=request.required_evaluators,
            generated_by=request.generated_by,
            risk_tier=risk_tier,
        )

    @staticmethod
    def _ensure_signal_operation(baseline: MechanismBundle, gap: CapabilityGap) -> list[MechanismOperation]:
        existing = {item.name: item for item in baseline.signals}
        if gap.signal.name not in existing:
            return [MechanismOperation(
                action="add", target_kind="signal", target_id=gap.signal.name,
                value=gap.signal.model_dump(mode="json"), reason=f"expose measured gap {gap.id}",
            )]
        if existing[gap.signal.name] != gap.signal:
            raise ValueError(f"gap signal {gap.signal.name} conflicts with baseline signal declaration")
        return []

    def _gap_candidates(self, request: SynthesisRequest, gap: CapabilityGap) -> list[tuple[str, MechanismPatch, str, tuple[str, ...]]]:
        measured, quality, fault, scale, comparator, threshold = _gap_quality(gap)
        activation = gap.activation
        role = "positive_drive" if gap.capability_kind == "task" else "bridge"
        weight = 0.25 + 0.75 * gap.severity
        evidence_window_steps = 64
        prefix = _slug(gap.id)
        signal_ops = self._ensure_signal_operation(request.baseline, gap)
        metric = MetricSpec(
            id=f"metric:{prefix}", name=f"{gap.capability} gap metric",
            expression=measured, unit=gap.signal.unit,
            window_steps=evidence_window_steps,
            intended_reading=f"desired {gap.desired}; good={gap.good}; bad={gap.bad}",
        )
        gate = GateSpec(
            id=f"gate:{prefix}", name=f"{gap.capability} evidence gate",
            metric_ref=metric.id, comparator=comparator, threshold=threshold,
            min_samples=evidence_window_steps, consecutive_windows=3, hysteresis=0.05 * scale,
            action="advance" if gap.use_for_curriculum else "evaluate",
            scope="curriculum" if gap.use_for_curriculum else "experiment",
            rationale=f"measure whether {gap.symptom} actually improves before promotion",
        )
        observation_ops = [
            MechanismOperation(action="add", target_kind="metric", target_id=metric.id, value=metric.model_dump(mode="json"), reason="measure the intervention target"),
            MechanismOperation(action="add", target_kind="gate", target_id=gate.id, value=gate.model_dump(mode="json"), reason="require sustained evidence rather than one sample"),
        ]

        drive = RewardTermSpec(
            id=f"reward:{prefix}:drive", name=f"{gap.capability} positive quality drive",
            role=role, expression=quality, activation=activation, weight=weight,
            polarity="reward", reward_group=gap.reward_group,
            intended_effect=f"make {gap.capability} quality directly profitable",
            failure_region=f"{gap.signal.name} is in the measured bad region {gap.bad}",
            success_region=f"{gap.signal.name} reaches the desired region {gap.good}",
        )
        drive_ops = signal_ops + observation_ops + [MechanismOperation(
            action="add", target_kind="reward", target_id=drive.id,
            value=drive.model_dump(mode="json"), reason="add a positive mathematical path toward the desired behavior",
        )]
        drive_patch = self._patch(
            request, variant=f"{gap.id}:drive", operations=drive_ops,
            expected_effects=[f"increase {gap.capability} quality without requiring a failure penalty"],
            hypothesis_refs=[gap.hypothesis_ref], risk_tier="low" if gap.capability_kind == "quality" else "medium",
        )

        normalized_fault = Expression(op="square", args=(Expression(op="safe_ratio", args=(fault, Expression.const(scale))),))
        guard = RewardTermSpec(
            id=f"reward:{prefix}:guard", name=f"{gap.capability} failure guard",
            role="guard", expression=normalized_fault, activation=activation,
            weight=0.15 + 0.35 * gap.severity, polarity="penalty", reward_group=gap.reward_group,
            intended_effect=f"remove high-severity {gap.capability} failures after a positive path exists",
            failure_region=f"{gap.signal.name} exceeds the good region",
            success_region="zero penalty inside the good region",
        )
        guard_ops = drive_ops + [MechanismOperation(
            action="add", target_kind="reward", target_id=guard.id,
            value=guard.model_dump(mode="json"), reason="bound the failure tail while retaining the positive drive",
        )]
        guard_patch = self._patch(
            request, variant=f"{gap.id}:drive+guard", operations=guard_ops,
            expected_effects=[f"increase {gap.capability} quality", f"reduce the tail of {gap.symptom}"],
            hypothesis_refs=[gap.hypothesis_ref], risk_tier="medium",
        )
        falsification = (
            f"{metric.id} does not move toward {gap.good} in the declared observation window",
            "a protected capability regresses beyond its approved tolerance",
            "the reward rises while the physical symptom remains unchanged",
        )
        return [
            (f"{gap.id}:drive", drive_patch, "positive drive with aligned telemetry", falsification),
            (f"{gap.id}:drive+guard", guard_patch, "same positive path plus a bounded failure-tail guard", falsification),
        ]

    def _custom_candidate(self, request: SynthesisRequest) -> tuple[str, MechanismPatch, str, tuple[str, ...]]:
        operations = [MechanismOperation(
            action=intent.action,
            target_kind=intent.target_kind,
            target_id=intent.target_id,
            value=intent.value,
            reason=intent.causal_rationale,
        ) for intent in request.custom_intents]
        patch = self._patch(
            request,
            variant="reasoner-intent",
            operations=operations,
            expected_effects=[intent.expected_effect for intent in request.custom_intents],
            hypothesis_refs=[],
            risk_tier="high" if any(item.target_kind == "signal" for item in request.custom_intents) else "medium",
        )
        return (
            "reasoner-intent",
            patch,
            "open-ended reasoner AST candidate",
            tuple(intent.falsification for intent in request.custom_intents),
        )

    def synthesize(self, request: SynthesisRequest) -> list[SynthesisCandidate]:
        raw: list[tuple[str, MechanismPatch, str, tuple[str, ...]]] = []
        for gap in request.gaps:
            raw.extend(self._gap_candidates(request, gap))
        if request.custom_intents:
            raw.append(self._custom_candidate(request))

        candidates: list[SynthesisCandidate] = []
        seen: set[str] = set(request.no_repeat_fingerprints)
        for variant, patch, rationale, falsification in raw:
            bundle = apply_mechanism_patch(request.baseline, patch)
            fingerprint = bundle.fingerprint()
            if patch.fingerprint() in seen or fingerprint in seen:
                continue
            seen.add(patch.fingerprint())
            seen.add(fingerprint)
            novelty = tuple(
                f"{operation.action}:{operation.target_kind}:{operation.target_id}"
                for operation in patch.operations
            )
            candidates.append(SynthesisCandidate(
                id=f"candidate:{patch.id.removeprefix('patch:')}",
                variant=variant,
                patch=patch,
                candidate_bundle_fingerprint=fingerprint,
                rationale=rationale,
                predictions=patch.expected_effects,
                falsification=falsification,
                novelty=novelty,
            ))
            if len(candidates) >= request.max_candidates:
                break
        return candidates
