"""Anti-cheat, parity, and numerical validation tests."""
from __future__ import annotations

import pytest

from autotuner.mechanisms.mechanism_specs import (
    Expression,
    MechanismBundle,
    MechanismOperation,
    MechanismPatch,
    RewardTermSpec,
    SignalSpec,
)
from autotuner.mechanisms.mechanism_validation import ValidationContext, validate_bundle, validate_patch


def _bundle(signal: SignalSpec | None = None, expression: Expression | None = None) -> MechanismBundle:
    signal = signal or SignalSpec(
        name="error", description="tracking error", unit="m/s", source_ref="input.error",
        lower_bound=0.0, upper_bound=2.0,
    )
    expression = expression or Expression(op="laplace", args=(Expression.signal(signal.name), Expression.const(0.5)))
    return MechanismBundle(
        id="bundle:validation", status="approved", contract_ref="contract:test",
        signals=(signal,), rewards=(RewardTermSpec(
            id="reward:drive", name="drive", role="positive_drive", expression=expression,
            intended_effect="guide behavior", failure_region="large error", success_region="small error",
        ),), evaluator_refs=("evaluator:flat",),
    )


def test_validation_accepts_deployable_reward_with_failure_gradient():
    report = validate_bundle(_bundle(), ValidationContext(
        runtime_signal_names=frozenset({"error"}),
        failure_probe_values={"error": [1.5, 1.5, 1.5, 1.5]},
    ))
    assert report.ok, report.issues
    assert report.checks["gradient:reward:reward:drive"] > 0.0


def test_validation_propagates_units_through_synthesized_additive_tree():
    deficit = Expression(op="maximum", args=(
        Expression(op="sub", args=(Expression.const(0.5), Expression.signal("error"))),
        Expression.const(0.0),
    ))
    quality = Expression(op="laplace", args=(deficit, Expression.const(0.5)))
    report = validate_bundle(_bundle(expression=quality), ValidationContext(
        runtime_signal_names=frozenset({"error"}),
        failure_probe_values={"error": [0.1, 0.1, 0.1, 0.1]},
    ))
    assert report.ok, report.issues
    assert report.checks["gradient:reward:reward:drive"] > 0.0


def test_validation_rejects_privileged_and_evaluator_bookkeeping_signal():
    signal = SignalSpec(
        name="evaluator.score", description="forbidden score", source_ref="component.evaluator_score",
        privileged=True, availability={"eval"},
    )
    report = validate_bundle(_bundle(signal, Expression.signal(signal.name)), ValidationContext())
    codes = {item.code for item in report.errors}
    assert "source.privileged_signal" in codes
    assert "source.availability_mismatch" in codes
    assert "anti_cheat.reward_evaluator_signal" in codes


def test_validation_rejects_constant_positive_drive_and_no_failure_gradient():
    report = validate_bundle(_bundle(expression=Expression.const(1.0)))
    codes = {item.code for item in report.errors}
    assert "reachability.constant_drive" in codes


def test_patch_validation_preserves_evaluator_refs():
    baseline = _bundle()
    patch = MechanismPatch(
        id="patch:remove-evaluator", baseline_bundle_ref=baseline.id,
        baseline_fingerprint=baseline.fingerprint(), candidate_bundle_id="bundle:candidate",
        problem_ref="case:test",
        operations=(MechanismOperation(
            action="replace", target_kind="reward", target_id="reward:drive",
            value=baseline.rewards[0].model_copy(update={"weight": 0.5}).model_dump(mode="json"),
            reason="test replacement",
        ),), expected_effects=("test",), protected_capabilities=("flat",),
        required_evaluators=("evaluator:flat",), generated_by="test",
    )
    _, report = validate_patch(baseline, patch, ValidationContext(
        required_evaluator_refs=("evaluator:flat",), protected_capabilities=("flat",),
    ))
    assert report.ok, report.issues
