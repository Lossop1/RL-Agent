"""证据驱动的研究提案协调层。

LLM 只负责把受限的证据摘要翻译成候选变更。候选必须经过确定性的
mechanism AST 校验、产品能力校验和人工批准，才会进入任务合同谱系。
这里不执行 Python、不生成远程命令，也不直接修改任何训练源文件。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from autotuner.mechanisms.mechanism_specs import (
    MechanismBundle,
    MechanismPatch,
)
from autotuner.mechanisms.mechanism_synthesis import (
    MechanismIntent,
    MechanismSynthesizer,
    SynthesisRequest,
)
from autotuner.mechanisms.mechanism_validation import (
    ValidationContext,
    ValidationReport,
    validate_patch,
)
from autotuner.product import (
    ResolvedProductContract,
    TaskContractBundle,
    TaskExecutionPipeline,
    TaskPipelineResult,
    TrainingLaunchRequest,
)
from .client import LLMResponse, call_llm_with_schema


RESEARCH_PROPOSAL_SCHEMA = "rl-agent.research-proposal/v1"
_ALLOWED_TASK_CHANGE_SECTIONS = frozenset({"training", "telemetry"})
_ALLOWED_TRAINING_CHANGES = frozenset({"config_overlay", "mechanism_proposals"})
_ALLOWED_TELEMETRY_CHANGES = frozenset({"monitoring_proposals"})
_FORBIDDEN_TEXT = (
    "python",
    "powershell",
    "bash",
    "shell",
    "ssh",
    "subprocess",
    "os.system",
    "exec(",
    "eval(",
)


class ResearchProposalError(ValueError):
    """研究提案不满足安全边界或确定性校验时抛出。"""


class ResearchProposal(BaseModel):
    """LLM 或人工提交的结构化候选；不代表已经批准或可执行。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = RESEARCH_PROPOSAL_SCHEMA
    proposal_id: str
    problem_ref: str
    problem_statement: str
    evidence_refs: tuple[str, ...] = ()
    hypothesis_refs: tuple[str, ...] = ()
    task_changes: dict[str, dict[str, Any]] = Field(default_factory=dict)
    mechanism_intents: tuple[MechanismIntent, ...] = ()
    expected_effects: tuple[str, ...] = ()
    falsifiable_predictions: tuple[str, ...] = ()
    protected_capabilities: tuple[str, ...] = ()
    required_evaluators: tuple[str, ...] = ()
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


@dataclass(frozen=True)
class ResearchProposalRequest:
    """传给 LLM 的最小上下文；原始日志和源码不进入模型边界。"""

    problem_ref: str
    problem_statement: str
    evidence_refs: tuple[str, ...]
    evidence_summary: Mapping[str, Any]
    protected_capabilities: tuple[str, ...]
    required_evaluators: tuple[str, ...]
    baseline_bundle_ref: str
    baseline_mechanism: MechanismBundle | None = None

    def validate(self) -> None:
        if not self.problem_ref.strip() or not self.problem_statement.strip():
            raise ResearchProposalError("problem_ref and problem_statement are required")
        if not self.evidence_refs:
            raise ResearchProposalError("at least one evidence reference is required")
        if not self.protected_capabilities:
            raise ResearchProposalError("protected capabilities are required")
        if not self.required_evaluators:
            raise ResearchProposalError("independent evaluators are required")
        if len(str(self.evidence_summary)) > 20000:
            raise ResearchProposalError("evidence summary is too large for the proposal boundary")


@dataclass(frozen=True)
class ResearchProposalResult:
    """保留 LLM 原始响应，同时把无效提案明确表示为失败。"""

    proposal: ResearchProposal | None
    response: LLMResponse
    error: str = ""

    @property
    def ready(self) -> bool:
        return self.proposal is not None and not self.error


@dataclass(frozen=True)
class AppliedResearchProposal:
    """批准后的合同修订、机制校验和完整可训练交接。"""

    proposal: ResearchProposal
    revised_bundle: TaskContractBundle
    stored_contract: Any
    pipeline_result: TaskPipelineResult
    mechanism_patch: MechanismPatch | None
    mechanism_validation: ValidationReport | None


def _json_schema() -> dict[str, Any]:
    """给 LLM 的 schema 只描述候选字段，不暴露任意代码入口。"""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "proposal_id",
            "problem_ref",
            "problem_statement",
            "evidence_refs",
            "hypothesis_refs",
            "task_changes",
            "mechanism_intents",
            "expected_effects",
            "falsifiable_predictions",
            "protected_capabilities",
            "required_evaluators",
            "confidence",
        ],
        "properties": {
            "proposal_id": {"type": "string", "minLength": 1, "maxLength": 120},
            "problem_ref": {"type": "string", "minLength": 1, "maxLength": 200},
            "problem_statement": {"type": "string", "minLength": 1, "maxLength": 1000},
            "evidence_refs": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 240},
                "maxItems": 32,
            },
            "hypothesis_refs": {
                "type": "array",
                "items": {"type": "string", "maxLength": 240},
                "maxItems": 16,
            },
            "task_changes": {"type": "object", "maxProperties": 2},
            "mechanism_intents": {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "action",
                        "target_kind",
                        "target_id",
                        "value",
                        "causal_rationale",
                        "expected_effect",
                        "falsification",
                    ],
                    "properties": {
                        "action": {"type": "string", "enum": ["add", "replace", "remove"]},
                        "target_kind": {
                            "type": "string",
                            "enum": ["reward", "metric", "gate"],
                        },
                        "target_id": {"type": "string", "minLength": 1, "maxLength": 120},
                        "value": {"type": ["object", "null"]},
                        "causal_rationale": {"type": "string", "maxLength": 800},
                        "expected_effect": {"type": "string", "maxLength": 500},
                        "falsification": {"type": "string", "maxLength": 500},
                    },
                },
            },
            "expected_effects": {
                "type": "array",
                "items": {"type": "string", "maxLength": 500},
                "maxItems": 16,
            },
            "falsifiable_predictions": {
                "type": "array",
                "items": {"type": "string", "maxLength": 500},
                "maxItems": 16,
            },
            "protected_capabilities": {
                "type": "array",
                "items": {"type": "string", "maxLength": 160},
                "maxItems": 32,
            },
            "required_evaluators": {
                "type": "array",
                "items": {"type": "string", "maxLength": 160},
                "maxItems": 32,
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    }


def _prompt(request: ResearchProposalRequest) -> tuple[str, str]:
    baseline = {
        "bundle_ref": request.baseline_bundle_ref,
        "mechanism_bundle": (
            {
                "id": request.baseline_mechanism.id,
                "fingerprint": request.baseline_mechanism.fingerprint(),
                "signals": [item.name for item in request.baseline_mechanism.signals],
                "parameters": [item.name for item in request.baseline_mechanism.parameters],
                "rewards": [item.id for item in request.baseline_mechanism.rewards],
                "metrics": [item.id for item in request.baseline_mechanism.metrics],
                "gates": [item.id for item in request.baseline_mechanism.gates],
                "evaluator_refs": list(request.baseline_mechanism.evaluator_refs),
            }
            if request.baseline_mechanism is not None
            else None
        ),
    }
    system = (
        "你是强化学习研究协调器。只输出 JSON 候选提案，不输出代码、命令、路径操作或批准结论。"
        "奖励、metric、gate 只能使用受限机制 AST；无法从证据区分原因时输出空变更并降低 confidence。"
        "不得删除 protected capabilities 或 required evaluators。"
        "允许的 task_changes 只有 training.config_overlay、training.mechanism_proposals "
        "和 telemetry.monitoring_proposals。"
        f"基线摘要：{baseline}"
    )
    user = (
        f"problem_ref={request.problem_ref}\n"
        f"problem_statement={request.problem_statement}\n"
        f"evidence_refs={list(request.evidence_refs)}\n"
        f"evidence_summary={dict(request.evidence_summary)}\n"
        f"protected_capabilities={list(request.protected_capabilities)}\n"
        f"required_evaluators={list(request.required_evaluators)}"
    )
    return system, user


def _check_text(value: Any, field: str) -> None:
    text = str(value or "").lower()
    if any(token in text for token in _FORBIDDEN_TEXT):
        raise ResearchProposalError(f"{field} contains executable-code vocabulary")


def _validate_task_changes(changes: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(changes, Mapping):
        raise ResearchProposalError("task_changes must be a mapping")
    result: dict[str, dict[str, Any]] = {}
    for raw_section, raw_values in changes.items():
        section = str(raw_section)
        if section not in _ALLOWED_TASK_CHANGE_SECTIONS:
            raise ResearchProposalError(f"unsupported task change section: {section}")
        if not isinstance(raw_values, Mapping):
            raise ResearchProposalError(f"task_changes.{section} must be a mapping")
        allowed = (
            _ALLOWED_TRAINING_CHANGES
            if section == "training"
            else _ALLOWED_TELEMETRY_CHANGES
        )
        unknown = sorted(str(key) for key in raw_values if str(key) not in allowed)
        if unknown:
            raise ResearchProposalError(
                f"unsupported task change fields in {section}: {', '.join(unknown)}"
            )
        for key, value in raw_values.items():
            key = str(key)
            if key == "config_overlay" and not isinstance(value, Mapping):
                raise ResearchProposalError(f"task_changes.{section}.{key} must be a mapping")
            _check_text(value, f"task_changes.{section}.{key}")
        result[section] = dict(raw_values)
    return result


def _validate_proposal(
    proposal: ResearchProposal,
    request: ResearchProposalRequest,
) -> ResearchProposal:
    if proposal.problem_ref != request.problem_ref:
        raise ResearchProposalError("proposal problem_ref does not match request")
    if not set(proposal.evidence_refs) <= set(request.evidence_refs):
        raise ResearchProposalError("proposal cites evidence outside the supplied evidence set")
    if set(proposal.protected_capabilities) != set(request.protected_capabilities):
        raise ResearchProposalError("proposal changed protected capabilities")
    if set(proposal.required_evaluators) != set(request.required_evaluators):
        raise ResearchProposalError("proposal changed required evaluators")
    _validate_task_changes(proposal.task_changes)
    if proposal.mechanism_intents and request.baseline_mechanism is None:
        raise ResearchProposalError("mechanism intents require a baseline mechanism bundle")
    for index, intent in enumerate(proposal.mechanism_intents):
        if intent.target_kind not in {"reward", "metric", "gate"}:
            raise ResearchProposalError(f"mechanism intent {index} has unsupported target kind")
        _check_text(intent.causal_rationale, f"mechanism_intents[{index}].causal_rationale")
        _check_text(intent.expected_effect, f"mechanism_intents[{index}].expected_effect")
        _check_text(intent.falsification, f"mechanism_intents[{index}].falsification")
    if not proposal.expected_effects:
        raise ResearchProposalError("proposal requires expected_effects")
    if not proposal.falsifiable_predictions:
        raise ResearchProposalError("proposal requires falsifiable_predictions")
    return proposal


def propose_research_change(request: ResearchProposalRequest) -> ResearchProposalResult:
    """调用 LLM 生成候选；失败时不伪造合同或机制。"""
    request.validate()
    system, user = _prompt(request)
    response = call_llm_with_schema(
        system_prompt=system,
        user_prompt=user,
        schema_name="research_proposal",
    )
    if response.parsed is None:
        return ResearchProposalResult(None, response, response.error or "llm_failed")
    try:
        proposal = _validate_proposal(
            ResearchProposal.model_validate(response.parsed),
            request,
        )
        _validate_declared_mechanism_proposals(proposal, request)
    except Exception as exc:
        return ResearchProposalResult(None, response, f"proposal_invalid:{exc}")
    return ResearchProposalResult(proposal, response)


def _mechanism_patch(
    proposal: ResearchProposal,
    request: ResearchProposalRequest,
) -> tuple[MechanismPatch | None, ValidationReport | None]:
    if not proposal.mechanism_intents:
        return None, None
    baseline = request.baseline_mechanism
    if baseline is None:
        raise ResearchProposalError("mechanism baseline is required")
    synthesis = MechanismSynthesizer()
    synthesis_request = SynthesisRequest(
        id=proposal.proposal_id,
        problem_ref=proposal.problem_ref,
        problem_statement=proposal.problem_statement,
        baseline=baseline,
        custom_intents=proposal.mechanism_intents,
        protected_capabilities=proposal.protected_capabilities,
        required_evaluators=proposal.required_evaluators,
        generated_by="llm_research_proposal",
        max_candidates=1,
    )
    candidates = synthesis.synthesize(synthesis_request)
    if not candidates:
        raise ResearchProposalError("mechanism synthesizer produced no candidate")
    patch = candidates[0].patch
    context = ValidationContext(
        required_evaluator_refs=proposal.required_evaluators,
        protected_capabilities=proposal.protected_capabilities,
        runtime_signal_names=frozenset(item.name for item in baseline.signals),
    )
    _, report = validate_patch(baseline, patch, context)
    return patch, report


def _validate_declared_mechanism_proposals(
    proposal: ResearchProposal,
    request: ResearchProposalRequest,
) -> None:
    """校验提案中直接声明的 patch，防止绕过 intent 合成路径。"""
    training = proposal.task_changes.get("training", {})
    if not isinstance(training, Mapping) or "mechanism_proposals" not in training:
        return
    raw_proposals = training["mechanism_proposals"]
    if not isinstance(raw_proposals, (list, tuple)) or not raw_proposals:
        raise ResearchProposalError("training.mechanism_proposals must be a non-empty list")
    baseline = request.baseline_mechanism
    if baseline is None:
        raise ResearchProposalError(
            "direct mechanism proposals require a baseline mechanism bundle"
        )
    context = ValidationContext(
        required_evaluator_refs=request.required_evaluators,
        protected_capabilities=request.protected_capabilities,
        runtime_signal_names=frozenset(item.name for item in baseline.signals),
    )
    for index, raw in enumerate(raw_proposals):
        if not isinstance(raw, Mapping):
            raise ResearchProposalError(
                f"training.mechanism_proposals[{index}] must be a mapping"
            )
        patch_data = raw.get("patch", raw)
        try:
            patch = MechanismPatch.model_validate(patch_data)
        except Exception as exc:
            raise ResearchProposalError(
                f"training.mechanism_proposals[{index}] is not a valid mechanism patch: {exc}"
            ) from exc
        if set(patch.protected_capabilities) != set(request.protected_capabilities):
            raise ResearchProposalError(
                f"training.mechanism_proposals[{index}] changed protected capabilities"
            )
        if set(patch.required_evaluators) != set(request.required_evaluators):
            raise ResearchProposalError(
                f"training.mechanism_proposals[{index}] changed required evaluators"
            )
        _, report = validate_patch(baseline, patch, context)
        if not report.ok:
            raise ResearchProposalError(
                f"training.mechanism_proposals[{index}] failed validation: "
                + "; ".join(issue.message for issue in report.errors)
            )


def apply_research_proposal(
    pipeline: TaskExecutionPipeline,
    product: ResolvedProductContract,
    base: TaskContractBundle,
    proposal: ResearchProposal,
    request: ResearchProposalRequest,
    *,
    approved_by: str,
    run_id: str,
    actor: str = "research-proposal",
    launch_request: TrainingLaunchRequest | Mapping[str, Any] | None = None,
    asset_reuse_approval_refs: Sequence[str] = (),
) -> AppliedResearchProposal:
    """批准提案并生成新合同版本及其完整训练交接。"""
    request.validate()
    _validate_proposal(proposal, request)
    _validate_declared_mechanism_proposals(proposal, request)
    approver = str(approved_by or "").strip()
    if not approver:
        raise ResearchProposalError("approved_by is required")
    if not proposal.proposal_id.strip():
        raise ResearchProposalError("proposal_id is invalid")
    mechanism_patch, mechanism_report = _mechanism_patch(proposal, request)
    if mechanism_report is not None and not mechanism_report.ok:
        raise ResearchProposalError(
            "mechanism proposal failed validation: "
            + "; ".join(issue.message for issue in mechanism_report.errors)
        )
    changes: dict[str, Any] = {}
    for section, values in proposal.task_changes.items():
        changes[section] = dict(values)
    if mechanism_patch is not None:
        training = dict(changes.get("training", {}))
        training["mechanism_proposals"] = [
            {
                "patch": mechanism_patch.model_dump(mode="json"),
                "validation": mechanism_report.model_dump(mode="json") if mechanism_report else {},
                "baseline_fingerprint": request.baseline_mechanism.fingerprint()
                if request.baseline_mechanism
                else "",
            }
        ]
        changes["training"] = training
    if not changes:
        raise ResearchProposalError("approved proposal contains no task or mechanism change")
    prepared_revision = pipeline.revise_and_prepare(
        product,
        base,
        {"changes": changes},
        evidence_refs=list(proposal.evidence_refs),
        approved_by=approver,
        run_id=run_id,
        actor=actor,
        launch_request=launch_request,
        asset_reuse_approval_refs=asset_reuse_approval_refs,
    )
    return AppliedResearchProposal(
        proposal=proposal,
        revised_bundle=prepared_revision.revised_bundle,
        stored_contract=prepared_revision.stored_contract,
        pipeline_result=prepared_revision.pipeline_result,
        mechanism_patch=mechanism_patch,
        mechanism_validation=mechanism_report,
    )


__all__ = [
    "AppliedResearchProposal",
    "RESEARCH_PROPOSAL_SCHEMA",
    "ResearchProposal",
    "ResearchProposalError",
    "ResearchProposalRequest",
    "ResearchProposalResult",
    "apply_research_proposal",
    "propose_research_change",
]
