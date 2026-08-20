from __future__ import annotations

from types import SimpleNamespace

from autotuner.llm_gateway.client import LLMResponse
from autotuner.llm_gateway import research_proposal
from autotuner.mechanisms.mechanism_specs import (
    Expression,
    MechanismBundle,
    RewardTermSpec,
    SignalSpec,
)
from autotuner.mechanisms.mechanism_specs import MechanismPatch, MechanismOperation


def _baseline() -> MechanismBundle:
    return MechanismBundle(
        id="bundle:base",
        status="approved",
        contract_ref="task:test",
        signals=(
            SignalSpec(
                name="task.progress",
                description="progress",
                source_ref="input.progress",
                lower_bound=0.0,
                upper_bound=1.0,
            ),
        ),
        rewards=(
            RewardTermSpec(
                id="reward:progress",
                name="progress",
                role="positive_drive",
                expression=Expression.signal("task.progress"),
                reward_group="track",
                intended_effect="drive progress",
                failure_region="no progress",
                success_region="progress",
            ),
        ),
        evaluator_refs=("evaluator:flat",),
    )


def _request(*, baseline_mechanism=None):
    return research_proposal.ResearchProposalRequest(
        problem_ref="case:flat",
        problem_statement="速度跟踪在稳定核心约束下不足",
        evidence_refs=("evidence:run-1:diag",),
        evidence_summary={"tracking": 0.6, "core": 0.8},
        protected_capabilities=("flat", "stand"),
        required_evaluators=("evaluator:flat",),
        baseline_bundle_ref="task.taili@1",
        baseline_mechanism=baseline_mechanism,
    )


def _response(payload):
    return LLMResponse(
        parsed=payload,
        raw_text="{}",
        model="test-model",
        elapsed_s=0.01,
        attempt=0,
    )


def _payload(**updates):
    reward = RewardTermSpec(
        id="reward:quality_drive",
        name="quality drive",
        role="bridge",
        expression=Expression.signal("task.progress"),
        reward_group="track",
        intended_effect="retain progress while exposing quality",
        failure_region="quality is absent",
        success_region="quality is present",
    )
    data = {
        "proposal_id": "proposal:flat-quality",
        "problem_ref": "case:flat",
        "problem_statement": "速度跟踪在稳定核心约束下不足",
        "evidence_refs": ["evidence:run-1:diag"],
        "hypothesis_refs": ["hypothesis:quality-drive"],
        "task_changes": {
            "training": {
                "config_overlay": {"reward": {"style_reward_weight": 0.2}},
            },
        },
        "mechanism_intents": [
            {
                "action": "add",
                "target_kind": "reward",
                "target_id": reward.id,
                "value": reward.model_dump(mode="json"),
                "causal_rationale": "增加可测的连续驱动",
                "expected_effect": "质量和进度共同改善",
                "falsification": "独立评估器没有改善",
            },
        ],
        "expected_effects": ["速度跟踪改善且不损害核心"],
        "falsifiable_predictions": ["四方向跟踪窗口上升"],
        "protected_capabilities": ["flat", "stand"],
        "required_evaluators": ["evaluator:flat"],
        "confidence": 0.8,
    }
    data.update(updates)
    return data


def test_proposal_is_structured_and_evidence_bounded(monkeypatch):
    baseline = _baseline()
    monkeypatch.setattr(
        research_proposal,
        "call_llm_with_schema",
        lambda **_kwargs: _response(_payload()),
    )

    result = research_proposal.propose_research_change(
        _request(baseline_mechanism=baseline)
    )

    assert result.ready
    assert result.proposal is not None
    assert result.proposal.mechanism_intents[0].target_kind == "reward"


def test_proposal_rejects_evidence_not_supplied(monkeypatch):
    data = _payload(evidence_refs=["evidence:invented"])
    monkeypatch.setattr(
        research_proposal,
        "call_llm_with_schema",
        lambda **_kwargs: _response(data),
    )

    result = research_proposal.propose_research_change(_request())

    assert not result.ready
    assert "evidence" in result.error


def test_proposal_rejects_unbounded_task_fields(monkeypatch):
    data = _payload(
        mechanism_intents=[],
        task_changes={"training": {"arbitrary_python": "eval('x')"}},
    )
    monkeypatch.setattr(
        research_proposal,
        "call_llm_with_schema",
        lambda **_kwargs: _response(data),
    )

    result = research_proposal.propose_research_change(_request())

    assert not result.ready
    assert "unsupported task change fields" in result.error


def test_approval_validates_mechanism_before_contract_revision(monkeypatch):
    baseline = _baseline()
    proposal = research_proposal.ResearchProposal.model_validate(_payload())
    calls = []

    class FakePipeline:
        def revise_and_prepare(self, *args, **kwargs):
            calls.append(kwargs)
            stored = SimpleNamespace(ref="task.taili@2")
            pipeline_result = SimpleNamespace(stored_contract=stored)
            return SimpleNamespace(
                revised_bundle=args[2],
                stored_contract=stored,
                pipeline_result=pipeline_result,
            )

    applied = research_proposal.apply_research_proposal(
        FakePipeline(),
        SimpleNamespace(),
        baseline,
        proposal,
        _request(baseline_mechanism=baseline),
        approved_by="human",
        run_id="run-approved",
    )

    assert applied.mechanism_patch is not None
    assert applied.mechanism_validation is not None
    assert applied.mechanism_validation.ok
    assert calls[0]["approved_by"] == "human"
    assert calls[0]["run_id"] == "run-approved"
    assert applied.pipeline_result.stored_contract.ref == "task.taili@2"


def test_direct_mechanism_proposal_cannot_bypass_protection_validation():
    baseline = _baseline()
    patch = MechanismPatch(
        id="patch:untrusted",
        baseline_bundle_ref=baseline.id,
        baseline_fingerprint=baseline.fingerprint(),
        candidate_bundle_id="bundle:untrusted",
        problem_ref="case:flat",
        operations=(
            MechanismOperation(
                action="replace",
                target_kind="reward",
                target_id="reward:progress",
                value=baseline.rewards[0].model_dump(mode="json"),
                reason="test direct patch",
            ),
        ),
        expected_effects=("effect",),
        protected_capabilities=("different-capability",),
        required_evaluators=("evaluator:flat",),
        generated_by="untrusted",
    )
    proposal = research_proposal.ResearchProposal.model_validate(
        _payload(
            mechanism_intents=[],
            task_changes={
                "training": {
                    "mechanism_proposals": [{"patch": patch.model_dump(mode="json")}],
                }
            },
        )
    )
    class FakePipeline:
        def revise_and_prepare(self, *args, **kwargs):
            raise AssertionError("invalid mechanism must be rejected before materialization")

    try:
        research_proposal.apply_research_proposal(
            FakePipeline(),
            SimpleNamespace(),
            baseline,
            proposal,
            _request(baseline_mechanism=baseline),
            approved_by="human",
            run_id="run-untrusted",
        )
    except research_proposal.ResearchProposalError as exc:
        assert "changed protected capabilities" in str(exc)
    else:
        raise AssertionError("untrusted direct mechanism proposal was accepted")
