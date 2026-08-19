"""Candidate synthesis tests, including an open-ended reasoner intent."""
from __future__ import annotations

from autotuner.mechanisms.mechanism_specs import (
    Expression,
    MechanismBundle,
    RewardTermSpec,
    SignalSpec,
    apply_mechanism_patch,
)
from autotuner.mechanisms.mechanism_synthesis import (
    CapabilityGap,
    MechanismIntent,
    MechanismSynthesizer,
    SynthesisRequest,
)


def _baseline() -> MechanismBundle:
    return MechanismBundle(
        id="bundle:base",
        status="approved",
        contract_ref="contract:taili",
        signals=(SignalSpec(
            name="task.progress", description="normalized command progress",
            source_ref="component.direction_progress", lower_bound=0.0, upper_bound=1.0,
        ),),
        rewards=(RewardTermSpec(
            id="reward:progress", name="progress", role="positive_drive",
            expression=Expression.signal("task.progress"), reward_group="track",
            intended_effect="move in command direction", failure_region="no progress",
            success_region="commanded progress",
        ),),
        evaluator_refs=("evaluator:flat", "evaluator:stairs"),
    )


def _request(**updates) -> SynthesisRequest:
    data = dict(
        id="synthesis:core-wobble",
        problem_ref="case:core-wobble",
        problem_statement="base roll/pitch velocity remains high",
        baseline=_baseline(),
        gaps=(CapabilityGap(
            id="core-wxy",
            capability="moving core stability",
            symptom="high roll/pitch angular velocity",
            signal=SignalSpec(
                name="core.wxy", description="body roll/pitch angular speed",
                unit="rad/s", frame="body", source_ref="input.base_ang_vel", selector=(0, 1),
                lower_bound=0.0, shape="vector",
            ),
            desired="lower", good=0.25, bad=0.9, severity=0.8,
            hypothesis_ref="hypothesis:core-drive-missing", reward_group="stab",
            signal_reduction="norm",
        ),),
        protected_capabilities=("flat_tracking", "stand", "stairs"),
        required_evaluators=("evaluator:flat", "evaluator:stairs"),
    )
    data.update(updates)
    return SynthesisRequest(**data)


def test_synthesizer_produces_drive_first_and_drive_plus_guard():
    candidates = MechanismSynthesizer().synthesize(_request())
    assert [item.variant for item in candidates] == ["core-wxy:drive", "core-wxy:drive+guard"]
    first_bundle = apply_mechanism_patch(_baseline(), candidates[0].patch)
    first_terms = {term.id: term for term in first_bundle.rewards}
    assert first_terms["reward:core-wxy:drive"].polarity == "reward"
    assert not any(term.polarity == "penalty" for term in first_bundle.rewards)
    metric = next(item for item in first_bundle.metrics if item.id == "metric:core-wxy")
    gate = next(item for item in first_bundle.gates if item.id == "gate:core-wxy")
    assert metric.window_steps == gate.min_samples == 64

    guarded = apply_mechanism_patch(_baseline(), candidates[1].patch)
    guard = next(term for term in guarded.rewards if term.id.endswith(":guard"))
    assert guard.polarity == "penalty"
    assert any(term.id.endswith(":drive") for term in guarded.rewards)


def test_no_repeat_filters_previously_executed_candidate():
    first = MechanismSynthesizer().synthesize(_request())
    repeated = MechanismSynthesizer().synthesize(_request(
        no_repeat_fingerprints=frozenset({first[0].candidate_bundle_fingerprint}),
    ))
    assert len(repeated) == 1
    assert repeated[0].variant == "core-wxy:drive+guard"


def test_reasoner_can_create_novel_ast_not_present_in_template_catalog():
    baseline = _baseline()
    custom = RewardTermSpec(
        id="reward:novel-coupling",
        name="novel command-quality coupling",
        role="bridge",
        expression=Expression(op="minimum", args=(
            Expression.signal("task.progress"),
            Expression(op="square", args=(Expression.signal("task.progress"),)),
        )),
        reward_group="track",
        intended_effect="test a reasoner-created nonlinear coupling",
        failure_region="progress lacks sustained quality",
        success_region="both terms agree",
    )
    request = _request(
        gaps=(),
        custom_intents=(MechanismIntent(
            action="add", target_kind="reward", target_id=custom.id,
            value=custom.model_dump(mode="json"),
            causal_rationale="a competing hypothesis predicts this nonlinear coupling",
            expected_effect="separate transient progress from sustained progress",
            falsification="the candidate metric and independent evaluator do not improve",
        ),),
    )
    candidate = MechanismSynthesizer().synthesize(request)[0]
    bundle = apply_mechanism_patch(baseline, candidate.patch)
    assert next(term for term in bundle.rewards if term.id == custom.id).expression.op == "minimum"
    assert candidate.variant == "reasoner-intent"
