"""Schema and deterministic patch tests for executable mechanisms."""
from __future__ import annotations

import pytest

from autotuner.mechanisms.mechanism_specs import (
    Expression,
    MechanismBundle,
    MechanismOperation,
    MechanismPatch,
    MetricSpec,
    RewardTermSpec,
    SignalSpec,
    apply_mechanism_patch,
)


def _bundle() -> MechanismBundle:
    error = SignalSpec(
        name="tracking.error",
        description="absolute command tracking error",
        unit="m/s",
        frame="command",
        lower_bound=0.0,
    )
    quality = Expression(op="laplace", args=(Expression.signal("tracking.error"), Expression.const(0.3)))
    return MechanismBundle(
        id="bundle:base",
        status="approved",
        contract_ref="contract:taili",
        signals=(error,),
        rewards=(RewardTermSpec(
            id="reward:tracking",
            name="tracking drive",
            role="positive_drive",
            expression=quality,
            intended_effect="move toward commanded velocity",
            failure_region="tracking error is high",
            success_region="tracking error approaches zero",
        ),),
        metrics=(MetricSpec(
            id="metric:tracking-error",
            name="tracking error",
            expression=Expression.signal("tracking.error"),
            unit="m/s",
            intended_reading="lower is better",
        ),),
        evaluator_refs=("evaluator:flat",),
    )


def test_expression_rejects_invalid_arity_and_unknown_signal():
    with pytest.raises(ValueError, match="requires 2 arguments"):
        Expression(op="div", args=(Expression.const(1.0),))
    with pytest.raises(ValueError, match="unknown signals"):
        MechanismBundle(
            id="bundle:bad",
            contract_ref="contract:taili",
            rewards=(RewardTermSpec(
                id="reward:x", name="x", role="positive_drive",
                expression=Expression.signal("missing"), intended_effect="x",
                failure_region="bad", success_region="good",
            ),),
        )


def test_bundle_fingerprint_is_deterministic():
    first = _bundle()
    second = MechanismBundle.model_validate(first.model_dump(mode="json"))
    assert first.fingerprint() == second.fingerprint()


def test_bundle_rejects_gate_with_unreachable_sample_window():
    baseline = _bundle()
    with pytest.raises(ValueError, match="metric history window"):
        MechanismBundle.model_validate({
            **baseline.model_dump(mode="json"),
            "gates": [{
                "id": "gate:tracking",
                "name": "tracking evidence",
                "metric_ref": "metric:tracking-error",
                "comparator": "le",
                "threshold": 0.2,
                "min_samples": 64,
                "rationale": "require sustained tracking evidence",
            }],
        })


def test_patch_is_bound_to_baseline_and_can_create_reward():
    baseline = _bundle()
    smooth = RewardTermSpec(
        id="reward:smooth-core",
        name="smooth core bridge",
        role="bridge",
        expression=Expression(op="gaussian", args=(Expression.signal("tracking.error"), Expression.const(0.5))),
        weight=0.2,
        intended_effect="retain a broad gradient before precise tracking is reached",
        failure_region="large tracking error",
        success_region="controlled tracking",
    )
    patch = MechanismPatch(
        id="patch:add-smooth-core",
        baseline_bundle_ref=baseline.id,
        baseline_fingerprint=baseline.fingerprint(),
        candidate_bundle_id="bundle:candidate",
        problem_ref="case:tracking",
        operations=(MechanismOperation(
            action="add", target_kind="reward", target_id=smooth.id,
            value=smooth.model_dump(mode="json"), reason="add broad shaping",
        ),),
        expected_effects=("improve early tracking gradient",),
        protected_capabilities=("stand", "stairs"),
        required_evaluators=("evaluator:flat",),
        generated_by="test",
    )
    candidate = apply_mechanism_patch(baseline, patch)
    assert candidate.status == "candidate"
    assert {term.id for term in candidate.rewards} == {"reward:tracking", "reward:smooth-core"}
    assert candidate.provenance["patch_id"] == patch.id

    stale = patch.model_copy(update={"baseline_fingerprint": "0" * 64})
    with pytest.raises(ValueError, match="fingerprint"):
        apply_mechanism_patch(baseline, stale)


def test_patch_cannot_modify_evaluator_by_omission():
    baseline = _bundle()
    assert baseline.metrics[0].evaluator_owned is False
    with pytest.raises(ValueError, match="independent evaluators"):
        MechanismPatch(
            id="patch:unsafe", baseline_bundle_ref=baseline.id,
            baseline_fingerprint=baseline.fingerprint(), candidate_bundle_id="bundle:x",
            problem_ref="case:x",
            operations=(MechanismOperation(
                action="remove", target_kind="metric", target_id="metric:tracking-error", reason="unsafe",
            ),),
            expected_effects=("make score easier",), protected_capabilities=("flat",),
            required_evaluators=(), generated_by="test",
        )
