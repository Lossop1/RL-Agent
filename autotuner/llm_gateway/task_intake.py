"""
task_intake — Translate user natural-language task into TaskSpec.

THE ONLY decision the LLM makes here: map user words to enums in
`KNOWN_*` from `schemas.py`.  Everything else (cfg values, training
budget, mutation logic) stays in the deterministic core.

Fallback policy
===============

If the LLM returns garbage / fails schema validation after retries,
we DO NOT pretend it produced a valid spec.  Instead `translate()`
returns a TaskSpec with `confidence=0.0` and a populated
`user_clarifications_needed` list asking the user to disambiguate.
The caller (CLI) is expected to surface those questions, not silently
proceed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from autotuner.llm_gateway.client import call_llm_with_schema
from autotuner.llm_gateway.schemas import (
    TaskSpecVocabulary,
    TaskSpec,
    validate_task_spec,
)
from autotuner.product import (
    ProductRegistry,
    ResolvedProductContract,
    TaskContractStore,
    TaskContractBundle,
    TaskContractCompiler,
    TaskContractError,
    TaskRequest,
    TaskExecutionPipeline,
    resolve_product_contract,
)


_SYSTEM_PROMPT = """\
You are a strict natural-language → typed-spec translator for an RL
locomotion autotuner.  Your output is the ONLY signal the autotuner
sees about user intent.  You are NOT a decision-maker; you only
classify the user's request into a fixed schema.

You must output a valid JSON object with fields:
  - robot_id              one of: {robots}
  - task_class            one of: {tasks}
  - terrain               one of: {terrains}
  - ambition              one of: {ambitions}
  - max_command_lin_vel_mps   number in 0.1 .. 3.0
  - user_clarifications_needed  list of natural-language questions
                                (max 4) to ask the user when you
                                couldn't decide a field with high
                                confidence
  - confidence            number 0.0 .. 1.0

If the user's text is too ambiguous to decide a field, choose a
SENSIBLE DEFAULT and add the question to `user_clarifications_needed`,
then lower `confidence`.  Never invent fields, never output values
outside the allowed enums.  Never echo internal autotuner cfg
mechanics back to the user.
"""

_USER_TEMPLATE = """\
USER REQUEST:
\"\"\"{text}\"\"\"

Return your JSON object now.
"""


_DYNAMIC_SYSTEM_PROMPT = """\
你是强化学习系统的任务合同翻译器。你只把用户目标整理为结构化意图，
不能选择或改写产品源码、Python 入口、远程目录、运行包和部署身份。

当前已解析产品合同摘要：
{product_summary}

只返回一个 JSON 对象，字段如下：
  objective: 非空字符串
  goals: 目标列表，每项含唯一 id 和 objective，可附 acceptance 或 priority
  constraints: 用户不可妥协的硬约束对象
  protected_capabilities: 后续优化不可回退的能力 id 列表
  baseline: 用户明确指定的基线或谱系；未知时为空对象
  training: 任务级训练意图，不得包含产品入口、源码路径或 requirements
  telemetry: 任务级观测需求
  diagnostics: 任务级诊断需求
  simulation: 任务级仿真需求
  deployment: 任务级部署需求，不得包含包名、构建器或远程路径
  clarifications: 缺少关键信息时要询问的问题列表，最多 6 项
  confidence: 0 到 1

不要生成代码，不要虚构资产，不要把未知信息当作默认事实。产品 requirements
是不可削弱的硬约束。若用户要求与其冲突，保留用户原意并在 clarifications
明确指出冲突，后续合同编译器会拒绝执行。
"""


_DYNAMIC_USER_PROMPT = """用户任务：
{text}

请输出任务意图 JSON。
"""


@dataclass(frozen=True)
class DynamicTaskIntakeResult:
    """动态任务解析结果；失败时不伪造可执行合同。"""

    product_id: str
    request: TaskRequest | None
    bundle: TaskContractBundle | None
    clarifications: tuple[str, ...]
    confidence: float
    model: str
    error: str = ""
    contract_ref: str = ""
    materialization_manifest: str = ""
    pipeline: Any | None = None

    @property
    def ready(self) -> bool:
        return self.bundle is not None and not self.clarifications and not self.error


def _product_summary(contract: ResolvedProductContract) -> str:
    """只向 LLM 暴露任务理解所需的合同摘要，不发送源码和原始配置。"""
    robot = contract.robot
    training = contract.training
    summary = {
        "product_id": contract.product_id,
        "product_version": contract.product_version,
        "robot": {
            "id": robot.get("id"),
            "label": robot.get("label"),
            "dof": robot.get("dof"),
            "capabilities": robot.get("capabilities", []),
        },
        "task": {
            "id": training.get("task_id"),
            "family": training.get("task_family"),
            "framework": training.get("framework_id"),
            "requirements": training.get("requirements", {}),
        },
        "simulation": {
            "world_ids": [
                item.get("id")
                for item in contract.simulation.get("worlds", [])
                if isinstance(item, Mapping) and item.get("id")
            ],
            "sim2sim": contract.simulation.get("sim2sim", {}),
        },
    }
    return json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2)


def _dynamic_fallback(contract: ResolvedProductContract, reason: str) -> DynamicTaskIntakeResult:
    return DynamicTaskIntakeResult(
        product_id=contract.product_id,
        request=None,
        bundle=None,
        clarifications=(f"任务解析失败（{reason}），请确认目标、硬约束和验收标准。",),
        confidence=0.0,
        model="",
        error=reason,
    )


def _validated_dynamic_request(
    parsed: Mapping[str, Any],
    *,
    contract: ResolvedProductContract,
    user_text: str,
    model: str,
    approved: bool,
    approved_by: str = "",
) -> tuple[TaskRequest, tuple[str, ...], float]:
    clarifications_raw = parsed.get("clarifications", ())
    if not isinstance(clarifications_raw, (list, tuple)) or len(clarifications_raw) > 6:
        raise TaskContractError("clarifications must be a list with at most 6 items")
    clarifications = tuple(str(item).strip() for item in clarifications_raw if str(item).strip())
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError) as exc:
        raise TaskContractError("confidence must be numeric") from exc
    if not 0.0 <= confidence <= 1.0:
        raise TaskContractError("confidence must be between 0 and 1")

    task_data = {
        "instance_id": str(parsed.get("instance_id") or f"{contract.product_id}-task"),
        "objective": parsed.get("objective"),
        "goals": parsed.get("goals", ()),
        "constraints": parsed.get("constraints", {}),
        "protected_capabilities": parsed.get("protected_capabilities", ()),
        "baseline": parsed.get("baseline", {}),
        "training": parsed.get("training", {}),
        "telemetry": parsed.get("telemetry", {}),
        "diagnostics": parsed.get("diagnostics", {}),
        "simulation": parsed.get("simulation", {}),
        "deployment": parsed.get("deployment", {}),
        "metadata": {
            "source": "llm_dynamic_task_intake",
            "raw_user_text": user_text,
            "llm_model": model,
            "confidence": confidence,
            "clarifications": list(clarifications),
        },
        # 批准权来自调用方，绝不读取模型输出中的 approved。
        "approved": bool(approved),
    }
    if approved_by.strip():
        task_data["metadata"]["approved_by"] = approved_by.strip()
    return TaskRequest.from_mapping(task_data), clarifications, confidence


def translate_dynamic(
    user_text: str,
    *,
    product_id: str | None = None,
    approved: bool = False,
    approved_by: str = "",
    output_root: str | Path | None = None,
    run_id: str | None = None,
    launch_request: Mapping[str, Any] | Any | None = None,
    registry: ProductRegistry | None = None,
) -> DynamicTaskIntakeResult:
    """基于当前产品合同解析用户任务，并生成同源的派生规格。"""
    contract = resolve_product_contract(product_id, registry=registry)
    text = str(user_text or "").strip()
    if not text or len(text) > 12000:
        return _dynamic_fallback(contract, "empty_or_too_long")

    response = call_llm_with_schema(
        system_prompt=_DYNAMIC_SYSTEM_PROMPT.format(product_summary=_product_summary(contract)),
        user_prompt=_DYNAMIC_USER_PROMPT.format(text=text),
        schema_name="dynamic_task_contract",
    )
    if response.parsed is None:
        return _dynamic_fallback(contract, response.error or "llm_failed")
    try:
        request, clarifications, confidence = _validated_dynamic_request(
            response.parsed,
            contract=contract,
            user_text=text,
            model=response.model,
            approved=approved,
            approved_by=approved_by,
        )
        bundle = TaskContractCompiler().compile_bundle(contract, request)
        contract_ref = ""
        materialization_manifest = ""
        pipeline_result = None
        # 未解决的澄清只产生内存中的草案，不能提前物料化为可执行交接。
        if output_root is not None and not clarifications:
            output_path = Path(output_root)
            # 保留旧的直接输出布局，同时把同一 bundle 纳入版本仓库，避免
            # 旧工具和新系统分别生成两份无法对齐的合同。
            bundle.write(output_path)
            pipeline_result = TaskExecutionPipeline(
                output_path,
                contract_store=TaskContractStore(output_path / "contract_store"),
            ).prepare(
                contract,
                bundle,
                run_id=(
                    str(run_id).strip()
                    if str(run_id or "").strip()
                    else f"intake-{bundle.contract.contract_id}-{bundle.contract.contract_version}"
                ),
                actor="llm_task_intake",
                launch_request=launch_request,
            )
            contract_ref = pipeline_result.stored_contract.ref
            materialization_manifest = str(pipeline_result.materialized.manifest)
    except (TaskContractError, TypeError, ValueError) as exc:
        return _dynamic_fallback(contract, f"schema_or_contract:{exc}")
    return DynamicTaskIntakeResult(
        product_id=contract.product_id,
        request=request,
        bundle=bundle,
        clarifications=clarifications,
        confidence=confidence,
        model=response.model,
        contract_ref=contract_ref,
        materialization_manifest=materialization_manifest,
        pipeline=pipeline_result,
    )


def _task_vocabulary(contract: ResolvedProductContract | None) -> TaskSpecVocabulary:
    """从产品合同读取旧 TaskSpec 入口所需词表。"""
    if contract is None:
        return TaskSpecVocabulary()
    intake = contract.training.get("intake", {})
    return TaskSpecVocabulary.from_mapping(intake if isinstance(intake, Mapping) else None)


def _build_system_prompt(vocabulary: TaskSpecVocabulary | None = None) -> str:
    values = vocabulary or TaskSpecVocabulary()
    return _SYSTEM_PROMPT.format(
        robots=list(values.robot_ids),
        tasks=list(values.task_classes),
        terrains=list(values.terrains),
        ambitions=list(values.ambition_levels),
    )


def _fallback_task_spec(
    user_text: str,
    reason: str,
    vocabulary: TaskSpecVocabulary | None = None,
) -> TaskSpec:
    """Deterministic fallback when LLM fails — always asks user."""
    values = vocabulary or TaskSpecVocabulary()
    return TaskSpec(
        robot_id=values.robot_ids[0],
        task_class=values.task_classes[0],
        terrain=values.terrains[0],
        ambition=values.ambition_levels[0],
        max_command_lin_vel_mps=0.5,
        user_clarifications_needed=(
            f"LLM fallback engaged ({reason}). Please confirm: "
            "robot? terrain? max forward speed?",
        ),
        raw_user_text=user_text,
        llm_model="(fallback)",
        confidence=0.0,
    )


def translate(
    user_text: str,
    *,
    product_id: str | None = None,
    registry: ProductRegistry | None = None,
) -> TaskSpec:
    """Main entry point. Always returns a TaskSpec; check `confidence`."""
    contract: ResolvedProductContract | None = None
    try:
        # 兼容入口也必须绑定产品合同；失败时只返回澄清结果，不伪造产品身份。
        contract = resolve_product_contract(
            product_id,
            registry=registry,
            check_files=False,
        )
    except Exception:
        contract = None
    vocabulary = _task_vocabulary(contract)
    if not user_text or len(user_text) > 4000:
        return _fallback_task_spec(user_text, "empty_or_too_long", vocabulary)

    system = _build_system_prompt(vocabulary)
    user = _USER_TEMPLATE.format(text=user_text)
    resp = call_llm_with_schema(
        system_prompt=system,
        user_prompt=user,
        schema_name="task_intake",
    )
    if resp.parsed is None:
        return _fallback_task_spec(user_text, resp.error or "llm_failed", vocabulary)

    try:
        return validate_task_spec(resp.parsed, user_text, resp.model, vocabulary)
    except ValueError as e:
        return _fallback_task_spec(user_text, f"schema_violation:{e}", vocabulary)
