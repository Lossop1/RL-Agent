"""Structured LLM workflows over deterministic console evidence.

The LLM is used for narration and prioritization only. Evidence collection,
proposal safety fields, and readiness gates are deterministic so the console can
still return useful results when the model is unavailable.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from .config import LocomotionConsoleSettings
from .config_manager import llm_profile, recent_llm_audit, remote_profile
from .config_set import get_active_config_set
from .diagnostics import DiagnosticsController
from .framework_profile import list_framework_profiles
from .robot_profile import get_robot_profile, validate_robot_profile


def _evidence(id_: str, label: str, status: str, detail: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"id": id_, "label": label, "status": status, "detail": detail, "data": data or {}}


def _proposal(
    *,
    name: str,
    kind: str,
    risk: str,
    reply: str,
    evidence: list[str],
    requires: list[str] | None = None,
    expected_result: str = "",
    rollback: str = "",
    args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "risk": risk,
        "reply": reply,
        "args": args or {},
        "evidence": evidence,
        "requires": requires or [],
        "expected_result": expected_result,
        "rollback": rollback,
    }


def llm_readiness(settings: LocomotionConsoleSettings) -> dict[str, Any]:
    profile = llm_profile(settings)
    audit = recent_llm_audit(settings, limit=20)
    tools = [
        "get_workspace_config",
        "list_frameworks",
        "get_status",
        "get_evidence_context",
        "get_operator_context",
        "get_training_telemetry",
        "get_remote_status",
        "get_taili_context_pack",
        "explain_definition",
        "get_signal_map",
        "get_code_knowledge",
        "get_tuning_ledger",
        "get_run_ledger",
        "get_train_log",
        "list_diagnostic_history",
        "get_diagnostic_report",
        "get_config",
        "get_asset",
        "search_knowledge",
    ]
    issues: list[str] = []
    if not profile.configured:
        issues.append("LLM 档案未配置完整")
    if not profile.api_key_resolved:
        issues.append("API key 环境变量未解析")
    if profile.unsafe_literal_key:
        issues.append("旧配置中包含明文 API key；请迁移到环境变量")
    return {
        "configured": profile.configured,
        "api_key_resolved": profile.api_key_resolved,
        "unsafe_literal_key": profile.unsafe_literal_key,
        "model": profile.model,
        "base_url": profile.base_url,
        "audit_count": len(audit),
        "tools": tools,
        "workflow_count": 2,
        "ok": profile.configured and profile.api_key_resolved and not profile.unsafe_literal_key,
        "issues": issues,
    }


async def system_audit(settings: LocomotionConsoleSettings, source: Any, *, include_llm: bool = True) -> dict[str, Any]:
    diagnostics = DiagnosticsController(settings, source)
    evidence: list[dict[str, Any]] = []
    findings: list[str] = []
    proposals: list[dict[str, Any]] = []
    transcript: list[dict[str, Any]] = []
    issues: list[str] = []

    config_set = get_active_config_set(settings)
    remote = remote_profile(settings)
    robot = get_robot_profile()
    robot_issues = validate_robot_profile(robot)
    readiness = llm_readiness(settings)

    status = await source.get_status()
    try:
        catalog = await diagnostics.catalog()
    except Exception as exc:  # noqa: BLE001
        catalog = None
        issues.append(f"diagnostic catalog unavailable: {type(exc).__name__}: {exc}")
    history = diagnostics.history.list(limit=8)

    evidence.append(_evidence(
        "config_set",
        "当前 ConfigSet",
        "ok",
        f"{config_set.id}: {config_set.framework.label}",
        {"task_goal": config_set.task_goal, "notes": list(config_set.notes)},
    ))
    evidence.append(_evidence(
        "remote_status",
        "远程与训练状态",
        "ok" if status.remote_ok else "error",
        f"{status.run_id}; iter {status.latest_iter}; running={status.running}",
        status.model_dump(),
    ))
    evidence.append(_evidence(
        "remote_profile",
        "远程机档案",
        "ok" if remote.password_configured else "warning",
        f"{remote.host}:{remote.port} work_dir={remote.work_dir}",
        remote.__dict__,
    ))
    evidence.append(_evidence(
        "robot_profile",
        "机器人档案",
        "ok" if not robot_issues else "warning",
        f"{robot.label}; {robot.dof} DoF; spec={robot.diagnostic_spec}",
        {"issues": robot_issues, "capabilities": list(robot.capabilities)},
    ))
    evidence.append(_evidence(
        "framework_registry",
        "框架注册表",
        "ok",
        f"{settings.framework_id} active; {len(list_framework_profiles())} registered",
        {
            "frameworks": [
                {
                    "id": profile.id,
                    "status": profile.status,
                    "task_id": profile.task_id,
                    "diagnostic_task": profile.diagnostic_task,
                    "active": profile.id == settings.framework_id,
                }
                for profile in list_framework_profiles()
            ]
        },
    ))
    evidence.append(_evidence(
        "llm_readiness",
        "LLM 就绪状态",
        "ok" if readiness["ok"] else "warning",
        f"{readiness['model']} via {readiness['base_url']}",
        readiness,
    ))
    if catalog is not None:
        evidence.append(_evidence(
            "diagnostic_catalog",
            "诊断目录",
            "ok" if catalog.available else "warning",
            f"{len(catalog.checkpoints)} checkpoints; training_running={catalog.training_running}",
            catalog.model_dump(),
        ))
    evidence.append(_evidence(
        "diagnostic_history",
        "诊断历史",
        "ok" if history.items else "unknown",
        f"{len(history.items)} 条最近任务",
        {"items": [item.model_dump() for item in history.items[:5]]},
    ))

    if status.running:
        findings.append(
            f"训练正在运行：{status.run_id}，iter {status.latest_iter}；诊断不是硬禁止，但必须提示 GPU/IsaacLab 资源竞争风险并由操作者确认。"
        )
        proposals.append(_proposal(
            name="warn_before_concurrent_diagnostics",
            kind="diagnostic",
            risk="low",
            reply="训练正在占用远程运行环境。可以并发启动诊断，但必须先确认：诊断会抢 GPU，可能拖慢训练，IsaacLab 同时初始化也可能失败。",
            evidence=["remote_status", "diagnostic_catalog"],
            requires=["操作者确认接受 GPU/IsaacLab 并发风险"],
            expected_result="确认后仍可通过白名单诊断动作启动；如果远程资源检查失败，再由后端返回明确错误。",
            rollback="无状态变更；这是调度建议。",
        ))
    if not readiness["ok"]:
        findings.append("LLM 存在就绪问题，当前只能谨慎使用。")
        proposals.append(_proposal(
            name="repair_llm_profile",
            kind="llm",
            risk="low",
            reply="把 LLM API key 放入环境变量，配置文件只保存环境变量名。",
            evidence=["llm_readiness"],
            requires=["包含 API key 的环境变量"],
            expected_result="LLM readiness 显示 ok=true，且没有 unsafe_literal_key。",
            rollback="必要时恢复之前的本地 llm_profile.json。",
        ))
    if robot_issues:
        findings.append("机器人档案存在校验问题。")
    if catalog is not None and not catalog.checkpoints:
        findings.append("诊断目录中没有可用检查点。")
    if not findings:
        findings.append("核心控制台状态一致：远程机、机器人、框架、诊断和 LLM 接口均有响应。")

    summary = _deterministic_summary("system_audit", findings, proposals)
    if include_llm and readiness["configured"] and readiness["api_key_resolved"]:
        summary, transcript, llm_issues = _llm_narrate(
            workflow="system_audit",
            evidence=evidence,
            findings=findings,
            proposals=proposals,
            fallback=summary,
        )
        issues.extend(llm_issues)
        generated_by = "deterministic+llm" if transcript else "deterministic"
    else:
        generated_by = "deterministic"

    return {
        "workflow": "system_audit",
        "ok": not any(item["status"] == "error" for item in evidence),
        "generated_by": generated_by,
        "summary": summary,
        "evidence": evidence,
        "findings": findings,
        "proposals": proposals,
        "transcript": transcript,
        "issues": issues,
    }


async def diagnostic_brief(
    settings: LocomotionConsoleSettings,
    source: Any,
    *,
    job_id: str | None = None,
    include_llm: bool = True,
) -> dict[str, Any]:
    diagnostics = DiagnosticsController(settings, source)
    history = diagnostics.history.list(limit=10)
    selected = job_id or (history.items[0].job_id if history.items else "")
    evidence: list[dict[str, Any]] = [
        _evidence("diagnostic_history", "诊断历史", "ok" if history.items else "unknown", f"{len(history.items)} 条最近任务", {"items": [item.model_dump() for item in history.items[:5]]})
    ]
    findings: list[str] = []
    proposals: list[dict[str, Any]] = []
    issues: list[str] = []
    transcript: list[dict[str, Any]] = []

    if not selected:
        findings.append("当前还没有诊断任务。")
        proposals.append(_proposal(
            name="run_quick_diagnostic",
            kind="diagnostic",
            risk="medium",
            reply="可以先运行一次快速 checkpoint 诊断；如果训练正在运行，需要操作者确认 GPU/IsaacLab 并发风险。",
            evidence=["diagnostic_history"],
            requires=["远程可达", "存在可用检查点", "若训练正在运行则确认资源竞争风险"],
            expected_result="为所选 checkpoint 生成诊断报告和回放产物。",
            rollback="如果诊断仍在运行，可以取消该诊断任务。",
        ))
    else:
        try:
            report = await diagnostics.report_for_job(job_id=selected)
            evidence.append(_evidence(
                "diagnostic_report",
                "诊断报告",
                "ok",
                f"{report.preset_label}; {len(report.commands)} command rows",
                report.model_dump(),
            ))
            unstable = [
                cmd for cmd in report.commands
                if float(cmd.get("major_event_fraction") or 0.0) > 0 or float(cmd.get("stable_fraction") or 0.0) < 0.75
            ]
            if unstable:
                findings.append(f"诊断任务 {selected} 中有 {len(unstable)} 条命令结果需要关注。")
                proposals.append(_proposal(
                    name="inspect_diagnostic_playback",
                    kind="diagnostic",
                    risk="low",
                    reply="先查看不稳定命令段附近的回放，确认接触、reset 和姿态事件，再考虑改配置。",
                    evidence=["diagnostic_report"],
                    requires=["已完成且有回放产物的诊断任务"],
                    expected_result="操作者能区分映射/接触问题与 reward 或命令跟踪问题。",
                    rollback="无状态变更；这是只读检查。",
                    args={"job_id": selected},
                ))
            else:
                findings.append(f"诊断任务 {selected} 的归一化报告中没有明显不稳定命令行。")
        except Exception as exc:  # noqa: BLE001
            findings.append(f"诊断任务 {selected} 的报告不可读：{type(exc).__name__}: {exc}")
            issues.append(findings[-1])
            proposals.append(_proposal(
                name="rerun_quick_diagnostic",
                kind="diagnostic",
                risk="medium",
                reply="确认 checkpoint 和远程输出目录后，重新运行一次快速诊断。",
                evidence=["diagnostic_history"],
                requires=["远程可达", "若训练正在运行则确认资源竞争风险"],
                expected_result="生成新的报告/回放，并可被摘要。",
                rollback="如果诊断仍在运行，可以取消该诊断任务。",
                args={"job_id": selected},
            ))

    readiness = llm_readiness(settings)
    summary = _deterministic_summary("diagnostic_brief", findings, proposals)
    if include_llm and readiness["configured"] and readiness["api_key_resolved"]:
        summary, transcript, llm_issues = _llm_narrate(
            workflow="diagnostic_brief",
            evidence=evidence,
            findings=findings,
            proposals=proposals,
            fallback=summary,
        )
        issues.extend(llm_issues)
        generated_by = "deterministic+llm" if transcript else "deterministic"
    else:
        generated_by = "deterministic"
    return {
        "workflow": "diagnostic_brief",
        "ok": not issues,
        "generated_by": generated_by,
        "summary": summary,
        "evidence": evidence,
        "findings": findings,
        "proposals": proposals,
        "transcript": transcript,
        "issues": issues,
    }


def _deterministic_summary(workflow: str, findings: list[str], proposals: list[dict[str, Any]]) -> str:
    workflow_label = {
        "system_audit": "系统巡检",
        "diagnostic_brief": "诊断解读",
    }.get(workflow, workflow)
    head = f"{workflow_label}：" + (findings[0] if findings else "未发现明显问题。")
    if proposals:
        risk_label = {"low": "低风险", "medium": "中风险", "high": "高风险"}.get(proposals[0]["risk"], proposals[0]["risk"])
        head += f" 下一项建议：{proposals[0]['name']}（{risk_label}）。"
    return head


def _llm_narrate(
    *,
    workflow: str,
    evidence: list[dict[str, Any]],
    findings: list[str],
    proposals: list[dict[str, Any]],
    fallback: str,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    from autotuner.llm_gateway.client import call_llm_with_schema

    system = (
        "You summarize an RL locomotion console workflow. Use only the provided evidence. "
        "Return strict JSON: {\"summary\":\"...\"}. Answer in concise Chinese unless evidence "
        "is explicitly English-only. Mention risk gates. Use concrete system terms such as "
        "LLM, Framework, ConfigSet, Adapter, diagnostics, and gates. Do not imply the LLM can "
        "bypass evidence or confirmation."
    )
    user = json.dumps(
        {
            "workflow": workflow,
            "evidence": [{k: v for k, v in item.items() if k != "data"} | {"data_preview": str(item.get("data", {}))[:1200]} for item in evidence],
            "findings": findings,
            "proposals": proposals,
        },
        ensure_ascii=False,
    )
    resp = call_llm_with_schema(system, user, schema_name=f"{workflow}_summary", max_retries=1)
    transcript = [{"schema": f"{workflow}_summary", "model": resp.model, "elapsed_s": resp.elapsed_s, "error": resp.error}]
    if resp and resp.parsed and isinstance(resp.parsed.get("summary"), str):
        return resp.parsed["summary"], transcript, []
    return fallback, transcript, [resp.error or "LLM summary unavailable"]


def run_workflow_sync(
    workflow: str,
    settings: LocomotionConsoleSettings,
    source: Any,
    *,
    job_id: str | None = None,
    include_llm: bool = True,
) -> dict[str, Any]:
    if workflow == "system_audit":
        return asyncio.run(system_audit(settings, source, include_llm=include_llm))
    if workflow == "diagnostic_brief":
        return asyncio.run(diagnostic_brief(settings, source, job_id=job_id, include_llm=include_llm))
    raise ValueError(f"Unknown LLM workflow: {workflow}")
