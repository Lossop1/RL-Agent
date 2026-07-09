"""FastAPI locomotion console - Loop 0 spine.

Endpoints:
    GET  /health                 liveness
    GET  /run/current            RunStatus
    WS   /run/current/stream     live MetricPoint stream
    POST /action/resume|kill|physeval

The app holds one data source (fake or real, chosen by config). It never reimplements
remote/eval logic - see datasource.py.
"""
from __future__ import annotations

import asyncio
import contextlib
import hmac
import os
import time
from typing import Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .config_manager import (
    desired_framework_id,
    llm_profile,
    recent_llm_audit,
    remote_profile,
    select_framework,
    test_remote_connection,
    update_remote_profile,
    update_llm_profile,
)
from .config_set import get_active_config_set
from .adaptation_preview import AdaptationPreviewInfo, available_robots, build_adaptation_preview
from .framework_edit_api import EditableKnobsInfo, EditValidationInfo, build_edit_validation, editable_knobs
from .robot_import_api import KnownUrdfsInfo, RobotImportInfo, build_robot_import, known_urdfs
from .datasource import make_source
from .context_manager import build_context_envelope, should_probe_remote_for_context
from .definitions import list_definitions
from .diagnostics import DiagnosticsController
from .framework_catalog import FrameworkCatalogInfo, build_framework_catalog
from .framework_profile import list_framework_profiles
from .llm_session import LLMSessionStore
from .llm_workflows import diagnostic_brief, llm_readiness, system_audit
from .robot_profile import get_robot_profile, validate_robot_profile
from .slash_commands import handle_slash_command
from .spec_coverage import build_spec_coverage_report
from .schemas import (
    ActionResult,
    AgentWorkbenchActionInfo,
    AgentWorkbenchAttemptInfo,
    AgentWorkbenchEvidenceInfo,
    AgentWorkbenchInfo,
    AgentWorkbenchJudgementInfo,
    CancelProposalRequest,
    ContextEnvelopeInfo,
    ChatProposalHistory,
    ChatProposalInfo,
    ChatRequest,
    ChatResponse,
    ConfigSetInfo,
    DiagnosticCatalog,
    DiagnosticHistory,
    DiagnosticHistoryDeleteResult,
    DiagnosticJobStatus,
    DiagnosticHistoryItem,
    DiagnosticHistoryUpdate,
    DiagnosticPlayback,
    DiagnosticReport,
    DiagnosticRunRequest,
    DefinitionQueryResult,
    ExecuteRequest,
    ExecuteResponse,
    FrameworkSelectionRequest,
    FrameworkSelectionResult,
    FrameworkProfileInfo,
    LLMAuditHistory,
    LLMAuditItemInfo,
    LLMProfileInfo,
    LLMProfileUpdate,
    LLMProfileUpdateResult,
    LLMReadinessInfo,
    LLMWorkflowRequest,
    LLMWorkflowResult,
    PhysevalResult,
    ProfileSummaryInfo,
    RemoteConnectionTestInfo,
    RemoteMachineStatus,
    RemoteProfileInfo,
    RemoteProfileUpdate,
    RemoteProfileUpdateResult,
    RobotProfileInfo,
    RunSnapshot,
    RunStatus,
    SpecCoverageReport,
    TensorboardScalarCatalog,
    TensorboardSeriesResponse,
    TrainingTelemetry,
)

settings = get_settings()

# ── auth for state-changing endpoints ────────────────────────────────────────────────────
# Every mutating route (POST/PATCH/DELETE) drives real remote actions — kill/start training,
# deploy payloads, rewrite the SSH target, execute LLM-proposed actions. Two independent guards:
#   1. CSRF: an `X-Console-Token` header MUST be present. A cross-origin page can only set a custom
#      header via a CORS preflight, which the restricted allow_origins list rejects — so a bodyless
#      simple-POST CSRF ("fetch('http://localhost:8000/action/kill',{method:'POST'})") never reaches
#      the handler. GET reads are exempt (idempotent, no secrets in the body — verified).
#   2. Access: if LOCOMOTION_CONSOLE_TOKEN is set, the header value must match it (constant-time),
#      so exposing the port to a network still requires the shared secret. If it is NOT set, only
#      loopback clients may mutate (safe dev default); a clear 403 tells operators to set the token
#      before exposing the console.
_CONSOLE_TOKEN = os.environ.get("LOCOMOTION_CONSOLE_TOKEN", "").strip()
_MUTATING_METHODS = frozenset({"POST", "PATCH", "DELETE", "PUT"})
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})


def _console_auth_error(token_header: Optional[str], client_host: str) -> Optional[tuple[int, str]]:
    """Return (status, detail) if the request must be rejected, else None.

    令牌机制已按操作者要求去掉 (0708): the console-token gate is disabled — all state-changing requests
    pass. Set LOCOMOTION_CONSOLE_TOKEN + LOCOMOTION_CONSOLE_REQUIRE_TOKEN=1 to re-enable it if the port
    is ever exposed to a network."""
    if os.environ.get("LOCOMOTION_CONSOLE_REQUIRE_TOKEN", "").strip() not in ("1", "true", "True"):
        return None
    if token_header is None:
        return (403, "X-Console-Token header required for state-changing requests")
    if _CONSOLE_TOKEN and not hmac.compare_digest(token_header, _CONSOLE_TOKEN):
        return (401, "invalid X-Console-Token")
    return None


app = FastAPI(
    title="RL Locomotion Console",
    version="0.0.1-loop0",
)


@app.middleware("http")
async def _mutation_auth(request: Request, call_next):
    """Gate state-changing HTTP requests on the console token. Implemented as HTTP-scope middleware
    (not a global dependency) so it applies ONLY to HTTP — WebSocket connections (/run/current/stream)
    are out of scope here and self-authorize via _ws_authorized. Reads (GET/HEAD) always pass."""
    if request.method in _MUTATING_METHODS:
        err = _console_auth_error(
            request.headers.get("x-console-token"),
            request.client.host if request.client else "",
        )
        if err:
            return JSONResponse(status_code=err[0], content={"detail": err[1]})
    return await call_next(request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.ui_origin, "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

source = make_source(settings)
diagnostics = DiagnosticsController(settings, source)
llm_session = LLMSessionStore(settings)


def _proposal_info(item: dict) -> ChatProposalInfo:
    return ChatProposalInfo(
        id=str(item.get("id") or ""),
        created_at=str(item.get("created_at") or ""),
        updated_at=str(item.get("updated_at") or item.get("created_at") or ""),
        status=str(item.get("status") or "pending"),  # type: ignore[arg-type]
        name=str(item.get("name") or ""),
        args=item.get("args") if isinstance(item.get("args"), dict) else {},
        reply=str(item.get("reply") or ""),
        result=str(item.get("result") or ""),
        kind=str(item.get("kind") or "action"),
        risk=str(item.get("risk") or "medium"),  # type: ignore[arg-type]
        evidence=list(item.get("evidence") or []),
        requires=list(item.get("requires") or []),
        expected_result=str(item.get("expected_result") or ""),
        rollback=str(item.get("rollback") or ""),
    )


def _workflow_result(data: dict) -> LLMWorkflowResult:
    recorded = llm_session.record_proposals(data.get("proposals") or [])
    return LLMWorkflowResult(
        workflow=data["workflow"],
        ok=bool(data.get("ok")),
        generated_by=data.get("generated_by", "deterministic"),
        summary=str(data.get("summary") or ""),
        evidence=data.get("evidence") or [],
        findings=list(data.get("findings") or []),
        proposals=[_proposal_info(item) for item in recorded],
        transcript=list(data.get("transcript") or []),
        issues=list(data.get("issues") or []),
    )


def _number_value(data: dict, key: str) -> float | None:
    value = data.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _workbench_metric_values(telemetry: TrainingTelemetry) -> dict[str, object]:
    latest = telemetry.latest
    snapshot = telemetry.snapshot
    values: dict[str, object] = {
        "available": telemetry.available,
        "mode": telemetry.mode,
        "run_id": telemetry.run_id,
        "running": telemetry.running,
        "runtime_state": telemetry.runtime_state,
        "stale": telemetry.stale,
        "telemetry_age_s": telemetry.telemetry_age_s,
        "history_points": len(telemetry.history),
        "summary": telemetry.summary,
        "error": telemetry.error,
        "log_path": telemetry.log_path,
        "telemetry_path": telemetry.telemetry_path,
    }
    if snapshot is not None:
        values["snapshot"] = {
            "status": snapshot.status,
            "conclusion": snapshot.conclusion,
            "progress_pct": snapshot.progress_pct,
            "phase": snapshot.phase,
            "command_mode": snapshot.command_mode,
            "active_dirs": snapshot.active_dirs,
            "blocked_by": snapshot.blocked_by,
            "next_gate": snapshot.next_gate,
            "latest_checkpoint": snapshot.latest_checkpoint,
            "blockers": [item.model_dump() for item in snapshot.blockers[:5]],
        }
    if latest is None:
        return values
    reward = latest.reward or {}
    curriculum = latest.curriculum or {}
    command = latest.command or {}
    health = latest.health or {}
    progress = {
        "gate": _number_value(curriculum, "progress_gate"),
        "ratio": _number_value(command, "progress_ratio"),
        "fwd": _number_value(curriculum, "progress_fwd"),
        "back": _number_value(curriculum, "progress_back"),
        "lat": _number_value(curriculum, "progress_lat"),
        "yaw": _number_value(curriculum, "progress_yaw"),
        "lagging_dir": str(curriculum.get("progress_lagging_dir") or ""),
    }
    values.update({
        "step": latest.step,
        "total_steps": latest.total_steps,
        "elapsed": latest.elapsed,
        "eta": latest.eta,
        "fps": latest.fps,
        "checkpoint": latest.checkpoint,
        "reward_total": _number_value(reward, "total"),
        "phase": str(curriculum.get("phase") or ""),
        "blocked_by": str(curriculum.get("blocked_by") or ""),
        "progress_gate": progress["gate"],
        "terrain_mean": curriculum.get("terrain_mean"),
        "dr_level": curriculum.get("dr_level"),
        "cmd_vx": _number_value(command, "cmd_vx"),
        "actual_vx": _number_value(command, "actual_vx"),
        "lin_err": _number_value(command, "lin_err"),
        "progress_ratio": progress["ratio"],
        "fall_rate": _number_value(health, "fall_rate"),
        "base_h": _number_value(health, "base_h"),
        "timeline": {
            "step": latest.step,
            "total_steps": latest.total_steps,
            "elapsed": latest.elapsed,
            "eta": latest.eta,
            "fps": latest.fps,
            "checkpoint": latest.checkpoint,
        },
        "curriculum": {
            "phase": str(curriculum.get("phase") or ""),
            "command_mode": str(curriculum.get("command_mode") or ""),
            "active_dirs": str(curriculum.get("active_dirs") or ""),
            "blocked_by": str(curriculum.get("blocked_by") or ""),
            "next_gate": str(curriculum.get("next_gate") or ""),
            "progress_gate": progress["gate"],
            "progress_fwd": progress["fwd"],
            "progress_back": progress["back"],
            "progress_lat": progress["lat"],
            "progress_yaw": progress["yaw"],
            "progress_lagging_dir": progress["lagging_dir"],
            "terrain_mean": curriculum.get("terrain_mean"),
            "terrain_max": curriculum.get("terrain_max"),
            "dr_level": curriculum.get("dr_level"),
            "penalty_gate": curriculum.get("penalty_gate"),
            "clearance_gate": curriculum.get("clearance_gate"),
            "gait_gate": curriculum.get("gait_gate"),
            "diagonal_gate": curriculum.get("diagonal_gate"),
            "duty_balance_gate": curriculum.get("duty_balance_gate"),
            "slip_gate": curriculum.get("slip_gate"),
            "fall_gate": curriculum.get("fall_gate"),
        },
        "command": {
            "cmd_vx": _number_value(command, "cmd_vx"),
            "cmd_vy": _number_value(command, "cmd_vy"),
            "cmd_wz": _number_value(command, "cmd_wz"),
            "actual_vx": _number_value(command, "actual_vx"),
            "actual_vy": _number_value(command, "actual_vy"),
            "actual_wz": _number_value(command, "actual_wz"),
            "v_along": _number_value(command, "v_along"),
            "speed_xy": _number_value(command, "speed_xy"),
            "lin_err": _number_value(command, "lin_err"),
            "yaw_err": _number_value(command, "yaw_err"),
            "progress_ratio": progress["ratio"],
        },
        "gait": {
            "gait_match": _number_value(command, "gait_match"),
            "diagonal_contact": _number_value(command, "diagonal_contact"),
            "diagonal_pair_instant": _number_value(command, "diagonal_pair_instant"),
            "duty_balance": _number_value(command, "duty_balance"),
            "duty_spread_window": _number_value(command, "duty_spread_window"),
            "stance_slip": _number_value(command, "stance_slip"),
            "stance_slip_high_fraction": _number_value(command, "stance_slip_high_fraction"),
        },
        "health": {
            "fall_rate": _number_value(health, "fall_rate"),
            "reset_rate": _number_value(health, "reset_rate"),
            "terminal_rate": _number_value(health, "terminal_rate"),
            "base_h": _number_value(health, "base_h"),
            "base_h_min": _number_value(health, "base_h_min"),
            "upright": _number_value(health, "upright"),
            "tilt_deg": _number_value(health, "tilt_deg"),
            "tilt_deg_max": _number_value(health, "tilt_deg_max"),
            "height_low_risk_window": _number_value(health, "height_low_risk_window"),
            "tilt_high_risk_window": _number_value(health, "tilt_high_risk_window"),
            "contacts_mean": _number_value(health, "contacts_mean"),
            "torque_util": _number_value(health, "torque_util"),
        },
        "reward": {
            "total": _number_value(reward, "total"),
            "tracking_lin": _number_value(reward, "tracking_lin"),
            "tracking_yaw": _number_value(reward, "tracking_yaw"),
            "stand": _number_value(reward, "stand"),
            "terrain_progress": _number_value(reward, "terrain_progress"),
            "clearance_under": _number_value(reward, "clearance_under"),
            "clearance_over": _number_value(reward, "clearance_over"),
            "stance_slip": _number_value(reward, "stance_slip"),
            "landing_impact": _number_value(reward, "landing_impact"),
        },
        "paths": {
            "run_dir": latest.paths.get("run_dir") or "",
            "telemetry_jsonl": latest.paths.get("telemetry_jsonl") or telemetry.telemetry_path,
            "train_log": latest.paths.get("log") or telemetry.log_path,
            "console_log": latest.paths.get("console_log") or "",
            "checkpoint_dir": latest.paths.get("checkpoint_dir") or "",
            "effective_config": latest.paths.get("effective_config") or "",
        },
    })
    return values


def _workbench_remote_values(remote: RemoteMachineStatus) -> dict[str, object]:
    gpu = remote.gpus[0] if remote.gpus else None
    return {
        "host": remote.host,
        "available": remote.available,
        "load_avg": remote.load_avg,
        "memory_available_mb": remote.memory_available_mb,
        "gpu_used_mb": gpu.memory_used_mb if gpu else None,
        "gpu_total_mb": gpu.memory_total_mb if gpu else None,
        "gpu_util_pct": gpu.utilization_gpu_pct if gpu else None,
        "gpu_temp_c": gpu.temperature_c if gpu else None,
        "tmux_sessions": len(remote.tmux_sessions),
        "training_processes": len(remote.training_processes),
        "error": remote.error,
    }


def _remote_training_running(remote: RemoteMachineStatus) -> bool:
    return remote.available and bool(remote.training_processes)


def _remote_status_detail(remote: RemoteMachineStatus) -> str:
    if not remote.available:
        return remote.error or "远程状态不可用"
    gpu = remote.gpus[0] if remote.gpus else None
    gpu_text = "GPU 未发现"
    if gpu is not None:
        used = int(gpu.memory_used_mb or 0)
        total = int(gpu.memory_total_mb or 0)
        util = int(gpu.utilization_gpu_pct or 0)
        gpu_text = f"{gpu.name or 'GPU'} {used}/{total}MB {util}%"
    sessions = ",".join(item.name for item in remote.tmux_sessions[:3]) or "无 tmux"
    return (
        f"{remote.host or 'remote'} · {gpu_text} · "
        f"训练进程 {len(remote.training_processes)} · tmux {sessions} · load {remote.load_avg or '?'}"
    )


def _build_workbench_judgement(
    *,
    status: RunStatus,
    telemetry: TrainingTelemetry,
    evidence_ids: list[str],
) -> AgentWorkbenchJudgementInfo:
    gaps: list[str] = []
    snapshot = telemetry.snapshot
    latest = telemetry.latest
    if not status.remote_ok or status.degraded or status.runtime_state == "remote_unavailable":
        return AgentWorkbenchJudgementInfo(
            status="remote_unavailable",
            title="远端不可用，不能判断训练质量",
            summary=status.error or status.note or "远端 SSH 不可用；本地界面和已有记录仍可读取。",
            confidence="high",
            evidence_ids=evidence_ids,
            gaps=["需要恢复远端 SSH 后才能读取最新训练、部署上传包或启动诊断。"],
        )
    if not telemetry.available or latest is None:
        return AgentWorkbenchJudgementInfo(
            status="needs_evidence",
            title="缺少最新训练证据",
            summary=telemetry.error or telemetry.summary or "当前没有可用遥测，智能体不能可靠判断策略问题。",
            confidence="high",
            evidence_ids=evidence_ids,
            gaps=["需要最新 telemetry JSONL 或诊断报告。"],
        )

    blocked_by = (snapshot.blocked_by if snapshot else "") or str((latest.curriculum or {}).get("blocked_by") or "")
    phase = (snapshot.phase if snapshot else "") or str((latest.curriculum or {}).get("phase") or "")
    terrain_mean = (latest.curriculum or {}).get("terrain_mean")
    dr_level = (latest.curriculum or {}).get("dr_level")
    actual_vx = _number_value(latest.command or {}, "actual_vx")
    cmd_vx = _number_value(latest.command or {}, "cmd_vx")
    progress = _number_value(latest.curriculum or {}, "progress_gate")
    if progress is None:
        progress = _number_value(latest.command or {}, "progress_ratio")
    fall_rate = _number_value(latest.health or {}, "fall_rate")

    if snapshot and snapshot.status in {"blocked", "error"}:
        confidence: Literal["high", "medium", "low"] = "high"
        title = snapshot.conclusion or f"训练被 {blocked_by or snapshot.status} 阻塞"
        summary = snapshot.conclusion or telemetry.summary or "快照显示当前训练没有通过关键门槛。"
        if blocked_by == "progress" and (terrain_mean in (0, 0.0, None)) and (dr_level in (0, 0.0, None)):
            title = "基础前进失败，不是地形阶段失败"
            summary = (
                f"当前 phase={phase or '?'}，terrain_mean={terrain_mean or 0}，dr_level={dr_level or 0}，"
                f"blocked_by=progress；cmd_vx={cmd_vx}，actual_vx={actual_vx}。"
            )
        return AgentWorkbenchJudgementInfo(
            status="blocked",
            title=title,
            summary=summary,
            confidence=confidence,
            evidence_ids=evidence_ids,
            gaps=gaps,
        )

    if blocked_by:
        return AgentWorkbenchJudgementInfo(
            status="blocked",
            title=f"训练卡在 {blocked_by}",
            summary=(
                f"当前 blocked_by={blocked_by}，progress={progress}，fall_rate={fall_rate}。"
                "需要智能体基于证据提出下一轮动作。"
            ),
            confidence="medium",
            evidence_ids=evidence_ids,
            gaps=gaps,
        )

    if status.running:
        return AgentWorkbenchJudgementInfo(
            status="running",
            title="训练正在运行，先观察关键门槛",
            summary=telemetry.summary or "当前没有明确阻塞结论。继续观察 progress、fall、terrain 和 DR 课程。",
            confidence="medium",
            evidence_ids=evidence_ids,
            gaps=gaps,
        )

    return AgentWorkbenchJudgementInfo(
        status="ready",
        title="已停止，等待智能体给出下一步",
        summary=telemetry.summary or "训练当前未运行；可以请求智能体基于最近证据生成下一步提案。",
        confidence="medium",
        evidence_ids=evidence_ids,
        gaps=gaps,
    )


def _chat_elapsed(started: float) -> float:
    return round(time.perf_counter() - started, 3)


def _slash_response_mode(slash: dict) -> str:
    explicit = str(slash.get("mode") or "").strip()
    if explicit:
        return explicit
    transcript = list(slash.get("transcript") or [])
    if any(str(item.get("tool") or "").startswith("llm.") for item in transcript if isinstance(item, dict)):
        return "slash_llm_diagnostic"
    if slash.get("proposed_action"):
        return "slash_action_gate"
    return "slash_fast"


def _chat_agent_message(
    message: str,
    ui_mode: str = "",
    context: str = "",
    envelope: ContextEnvelopeInfo | None = None,
) -> str:
    prefix: list[str] = []
    if envelope is not None:
        prefix.append(envelope.prompt)
    elif ui_mode:
        prefix.append(f"UI mode: {ui_mode}")
    if envelope is None and context:
        prefix.append(f"Visible UI context: {context[:1200]}")
    if envelope is not None or ui_mode in {"training", "diagnostics"} or context:
        prefix.append(
            "Tool guidance: for any broad analysis, tuning, root-cause, current-state, "
            "or next-step answer, first call get_evidence_context with the operator question "
            "and this UI mode. It routes to allowlisted evidence sources, collects a compact "
            "context, and marks missing/stale facts as gaps. Do not answer from generic RL "
            "knowledge alone. Use narrower tools only when a specific missing fact is needed."
        )
    lowered = (message or "").lower()
    strategy_triggers = [
        "spec",
        "yaml",
        "策略",
        "配置",
        "奖励",
        "步态",
        "调参",
        "训练问题",
        "为什么",
        "怎么改",
        "taili",
    ]
    strategy_triggers.extend([
        "\u7b56\u7565",      # 策略
        "\u914d\u7f6e",      # 配置
        "\u5956\u52b1",      # 奖励
        "\u6b65\u6001",      # 步态
        "\u8c03\u53c2",      # 调参
        "\u8bad\u7ec3",      # 训练
        "\u4e3a\u4ec0\u4e48",  # 为什么
        "\u600e\u4e48\u6539",  # 怎么改
        "reward",
        "gait",
        "tuning",
    ])
    if any(token in lowered for token in strategy_triggers):
        prefix.append(
            "Additional guidance: use get_taili_context_pack for focused strategy/spec/YAML grounding "
            "if get_evidence_context says no live state is needed or a narrower follow-up is needed. "
            "Do not answer from generic RL knowledge alone."
        )
    if not prefix:
        return message
    return "\n".join(prefix) + f"\n\nOperator now says: {message}"


def _chat_suggestions(response_mode: str, transcript: list[dict], ui_mode: str = "") -> list[str]:
    tools = [str(item.get("tool") or "") for item in transcript if isinstance(item, dict)]
    if response_mode == "slash_llm_diagnostic" or "get_diagnostic_report" in tools or ui_mode == "diagnostics":
        return [
            "/ask 这次回放里最像哪种失败：拖行、滑移、蹲走还是方向没跟上？",
            "/ask 如果只改一个训练参数，应该优先看哪个证据？",
            "/run diag directions",
        ]
    if "get_training_telemetry" in tools or response_mode == "slash_fast":
        return [
            "/why blocked",
            "/ask 根据当前 telemetry 判断训练是不是卡住了",
            "/explain track lin_err progress_gate",
        ]
    if response_mode == "slash_action_gate":
        return ["确认前再解释风险", "/ask 这个动作会读写哪些远程文件？"]
    return [
        "/ask 结合当前页面上下文继续分析",
        "/status",
        "/why blocked",
        "/diag history",
    ]


def _fast_status_intent(message: str, ui_mode: str = "") -> bool:
    text = (message or "").strip().lower()
    if not text or text.startswith("/") or len(text) > 80:
        return False
    # In the training tab, short natural-language questions are overwhelmingly status
    # checks.  Do this before keyword matching so mojibake/IME punctuation does not send
    # the user into the slow full-agent path.
    if ui_mode == "training" and len(text) <= 24:
        return True
    terms = (
        "情况", "状态", "现在", "如何", "怎么样", "怎样", "进展", "跑到哪",
        "还在跑", "是否在跑", "训练到", "卡住", "卡在哪", "为何", "为什么",
        "status", "progress", "running", "stuck", "how is it",
    )
    return any(term in text for term in terms) and (
        ui_mode in {"", "training", "diagnostics"} or "训练" in text or "run" in text
    )


def _fmt_float(value: object, digits: int = 3) -> str:
    try:
        if value is None:
            return "?"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "?"


async def _fast_training_status_response(req: ChatRequest, started: float) -> ChatResponse:
    import asyncio as _asyncio
    import json as _json
    import shlex as _shlex

    def _probe_sync() -> dict:
        if not hasattr(source, "_get_remote"):
            return {"error": "fast status requires real data source"}
        remote = source._get_remote()  # type: ignore[attr-defined]
        run_glob = (
            source._run_glob_shell()  # type: ignore[attr-defined]
            if hasattr(source, "_run_glob_shell")
            else "/root/gpufree-data/taili_runs/*"
        )
        script = (
            "run=$(ls -dt " + run_glob + " 2>/dev/null | head -1 | sed 's:/*$::'); "
            "printf '__RUN__%s\\n' \"$run\"; "
            "if pgrep -fa 'taili_blind_runtime\\.train_taili|taili_blind_runtime\\.launch_taili_train' >/dev/null 2>&1; "
            "then echo '__RUNNING__1'; else echo '__RUNNING__0'; fi; "
            "date +__NOW__%s; "
            "if [ -n \"$run\" ]; then "
            "  if [ -s \"$run/train.telemetry.jsonl\" ]; then printf '__TELEMETRY__'; tail -n 1 \"$run/train.telemetry.jsonl\"; "
            "  elif [ -s \"$run/console.telemetry.jsonl\" ]; then printf '__TELEMETRY__'; tail -n 1 \"$run/console.telemetry.jsonl\"; "
            "  else echo '__TELEMETRY__'; fi; "
            "fi; "
            "printf '__GPU__'; "
            "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || true"
        )
        out = remote.exec_out("bash -lc " + _shlex.quote(script), timeout=8)
        result: dict = {"raw": out}
        for line in (out or "").splitlines():
            if line.startswith("__RUN__"):
                result["run"] = line[len("__RUN__"):].strip()
            elif line.startswith("__RUNNING__"):
                result["running"] = line[len("__RUNNING__"):].strip() == "1"
            elif line.startswith("__NOW__"):
                try:
                    result["remote_now_s"] = int(line[len("__NOW__"):].strip())
                except ValueError:
                    pass
            elif line.startswith("__TELEMETRY__"):
                text = line[len("__TELEMETRY__"):].strip()
                if text:
                    try:
                        result["telemetry"] = _json.loads(text)
                    except Exception as exc:  # noqa: BLE001
                        result["telemetry_error"] = f"{type(exc).__name__}: {exc}"
            elif line.startswith("__GPU__"):
                cells = [part.strip() for part in line[len("__GPU__"):].split(",")]
                if len(cells) >= 4:
                    try:
                        result["gpu"] = {
                            "memory_used_mb": float(cells[0]),
                            "memory_total_mb": float(cells[1]),
                            "utilization_gpu_pct": float(cells[2]),
                            "temperature_c": float(cells[3]),
                        }
                    except ValueError:
                        result["gpu_raw"] = cells
        return result

    try:
        probe = await _asyncio.wait_for(_asyncio.to_thread(_probe_sync), timeout=10.0)
    except Exception as exc:  # noqa: BLE001
        probe = {"error": f"{type(exc).__name__}: {exc}"}
    transcript: list[dict] = []
    transcript.append({"tool": "fast_remote_probe", "args": {"timeout_s": 10}, "result": probe})

    latest = probe.get("telemetry") if isinstance(probe, dict) else None
    if not isinstance(latest, dict):
        err = str(probe.get("error") or probe.get("telemetry_error") or "")
        run = str(probe.get("run") or "")
        state = "运行中" if probe.get("running") else "未运行"
        reply = f"现在没有拿到最新 telemetry。远程显示训练{state}，run=`{run or '?'}`。{('错误：' + err) if err else '可以稍后刷新或用 /probe telemetry。'}"
        return ChatResponse(
            reply=reply,
            steps=len(transcript),
            mode="fast_status",
            elapsed_s=_chat_elapsed(started),
            transcript=transcript,
            grounding={"checked": True, "grounded": True, "unsupported": [], "mode": "fast_status"},
            suggestions=["/probe telemetry", "/status", "/ask 为什么没有 telemetry？"],
        )

    reward = latest.get("reward") if isinstance(latest.get("reward"), dict) else {}
    curriculum = latest.get("curriculum") if isinstance(latest.get("curriculum"), dict) else {}
    health = latest.get("health") if isinstance(latest.get("health"), dict) else {}
    command = latest.get("command") if isinstance(latest.get("command"), dict) else {}
    gpu = probe.get("gpu") if isinstance(probe.get("gpu"), dict) else None
    gpu_text = (
        f"GPU {gpu['memory_used_mb']:.0f}/{gpu['memory_total_mb']:.0f} MB，util {gpu['utilization_gpu_pct']:.0f}%"
        if gpu and gpu.get("memory_used_mb") is not None and gpu.get("memory_total_mb") and gpu.get("utilization_gpu_pct") is not None
        else "GPU 状态未知"
    )
    step = int(latest.get("step") or 0)
    total = latest.get("total_steps") or "?"
    phase = str(curriculum.get("phase") or "?")
    blocked_by = str(curriculum.get("blocked_by") or "")
    progress = curriculum.get("progress_gate")
    running = bool(probe.get("running"))
    state_word = "运行中" if running else "未运行"
    problem_bits = []
    if blocked_by:
        problem_bits.append(f"当前卡点 `{blocked_by}`")
    if progress is not None:
        problem_bits.append(f"progress_gate={_fmt_float(progress)}（推进目标通常 ≥0.70）")
    if health.get("base_h") is not None:
        problem_bits.append(f"base_h={_fmt_float(health.get('base_h'))} m")
    problem_bits.append(f"fall_rate={_fmt_float(health.get('fall_rate'))}")
    dir_progress = []
    for name in ("fwd", "back", "lat", "yaw"):
        value = curriculum.get(f"progress_{name}")
        if value is None:
            value = command.get(f"progress_{name}")
        if value is not None:
            dir_progress.append(f"{name}={_fmt_float(value)}")
    dir_line = f"方向推进：{', '.join(dir_progress)}。" if dir_progress else ""
    motion_line = (
        f"命令/实际：cmd_vx={_fmt_float(command.get('cmd_vx'))}，actual_vx={_fmt_float(command.get('actual_vx'))}，"
        f"lin_err={_fmt_float(command.get('lin_err'))}，progress={_fmt_float(command.get('progress_ratio'))}。"
        if command else ""
    )
    reply = (
        f"训练{state_word}：`{str(probe.get('run') or '').rsplit('/', 1)[-1]}`，step {step} / {total}，"
        f"fps≈{_fmt_float(latest.get('fps'), 2)}，phase={phase}。\n"
        f"奖励 total≈{_fmt_float(reward.get('total'))}；gait_gate≈{_fmt_float(curriculum.get('gait_gate'))}，"
        f"slip_gate≈{_fmt_float(curriculum.get('slip_gate'))}。\n"
        f"{'；'.join(problem_bits)}。\n"
        f"{dir_line}\n"
        f"{motion_line}\n"
        f"{gpu_text}。这是 fast telemetry 回复，没有等待完整 LLM；要深入判断原因，用 `/ask 为什么 progress gate 上不去？`。"
    )
    return ChatResponse(
        reply=reply,
        steps=len(transcript),
        mode="fast_status",
        elapsed_s=_chat_elapsed(started),
        transcript=transcript,
        grounding={"checked": True, "grounded": True, "unsupported": [], "mode": "fast_status"},
        suggestions=["/why blocked", "/ask 为什么 progress gate 上不去？", "/probe telemetry"],
    )


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "source": settings.source, "framework_id": settings.framework_id}


@app.get("/config/frameworks", response_model=list[FrameworkProfileInfo])
async def config_frameworks() -> list[FrameworkProfileInfo]:
    desired_id = desired_framework_id(settings)
    return [
        FrameworkProfileInfo(
            id=profile.id,
            label=profile.label,
            status=profile.status,
            experiment=profile.experiment,
            task_id=profile.task_id,
            diagnostic_task=profile.diagnostic_task,
            note=profile.note,
            active=profile.id == settings.framework_id,
            desired=profile.id == desired_id,
        )
        for profile in list_framework_profiles()
    ]


@app.get("/config/framework/catalog", response_model=FrameworkCatalogInfo)
async def config_framework_catalog() -> FrameworkCatalogInfo:
    """The Framework Library ② mechanism catalog (M1-M9 + blind-TP) + named compositions with
    per-component adapt_plan. Distinct from /config/frameworks (run-level profiles)."""
    return build_framework_catalog()


@app.get("/config/adaptation/preview", response_model=AdaptationPreviewInfo)
async def config_adaptation_preview(robot: str = "taili") -> AdaptationPreviewInfo:
    """Dry-run the adapter chain (③④⑦) for a robot preset: derived actuator/geometry, ⑦ consistency
    (URDF-vs-asset, sim2real hazards), ④ deploy-readiness. Read-only; nothing remote is touched."""
    if robot not in available_robots():
        raise HTTPException(status_code=404, detail=f"unknown robot {robot!r}; known: {available_robots()}")
    return await asyncio.to_thread(build_adaptation_preview, robot)


@app.get("/config/robot/import/known", response_model=KnownUrdfsInfo)
async def config_robot_import_known() -> KnownUrdfsInfo:
    """URDFs the workspace already knows about (robot key → path)."""
    return known_urdfs()


@app.get("/config/robot/import", response_model=RobotImportInfo)
async def config_robot_import(urdf: str) -> RobotImportInfo:
    """§3 C front door: is this URDF adaptable by the framework? (12-DoF quadruped, hip/thigh/calf,
    derivable effort/mass/leg). Returns adaptable + honest issue list. Read-only."""
    return await asyncio.to_thread(build_robot_import, urdf)


@app.get("/config/framework/edit/knobs", response_model=EditableKnobsInfo)
async def config_framework_edit_knobs() -> EditableKnobsInfo:
    """The editable framework knobs (⑥) and their physical units/rationale."""
    return editable_knobs()


@app.get("/config/framework/edit/validate", response_model=EditValidationInfo)
async def config_framework_edit_validate(field: str, value: float, robot: str = "taili") -> EditValidationInfo:
    """⑥ guard: is this proposed framework-content edit within the robot's derived safe band?
    Accepted only if in-band (§8.0 no out-of-band guesses). Read-only."""
    if robot not in available_robots():
        raise HTTPException(status_code=404, detail=f"unknown robot {robot!r}; known: {available_robots()}")
    return await asyncio.to_thread(build_edit_validation, field, value, robot)


def _llm_info() -> LLMProfileInfo:
    state = llm_profile(settings)
    return LLMProfileInfo(
        configured=state.configured,
        model=state.model,
        base_url=state.base_url,
        api_key_env_var=state.api_key_env_var,
        api_key_resolved=state.api_key_resolved,
        unsafe_literal_key=state.unsafe_literal_key,
        advisory_only=state.advisory_only,
        actions_require_confirmation=state.actions_require_confirmation,
    )


def _remote_info() -> RemoteProfileInfo:
    state = remote_profile(settings)
    return RemoteProfileInfo(
        host=state.host,
        port=state.port,
        user=state.user,
        work_dir=state.work_dir,
        conda_env=state.conda_env,
        diagnostic_tool_root=state.diagnostic_tool_root,
        diagnostic_output_root=state.diagnostic_output_root,
        diagnostic_robot_root=state.diagnostic_robot_root,
        diagnostic_python=state.diagnostic_python,
        log_format=state.log_format,
        password_configured=state.password_configured,
        source=state.source,
    )


@app.get("/config/remote", response_model=RemoteProfileInfo)
async def config_remote() -> RemoteProfileInfo:
    return _remote_info()


@app.get("/config/strategy")
async def config_strategy() -> dict:
    """The detailed TUNING STRATEGY the operator otherwise couldn't see in the panel: reward weights
    (grouped tracking / gait-quality / thresholds), curriculum phases + advancement gates, and AMP
    hyperparameters — read-only, from taili_blind_config.yaml. Edits go through the edit_config action."""
    from .strategy_view import build_strategy_view

    return await asyncio.to_thread(build_strategy_view)


@app.patch("/config/remote", response_model=RemoteProfileUpdateResult)
async def config_remote_update(req: RemoteProfileUpdate) -> RemoteProfileUpdateResult:
    updates = {
        "ssh_host": req.host,
        "ssh_port": req.port,
        "ssh_user": req.user,
        "ssh_pass": req.password,
        "work_dir": req.work_dir,
        "conda_env": req.conda_env,
        "diagnostic_tool_root": req.diagnostic_tool_root,
        "diagnostic_output_root": req.diagnostic_output_root,
        "diagnostic_robot_root": req.diagnostic_robot_root,
        "diagnostic_python": req.diagnostic_python,
        "log_format": req.log_format,
    }
    try:
        result = update_remote_profile(settings, updates)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RemoteProfileUpdateResult(
        profile=_remote_info(),
        saved_path=result.saved_path,
        requires_restart=result.requires_restart,
        message=result.message,
    )


@app.post("/config/remote/test", response_model=RemoteConnectionTestInfo)
async def config_remote_test() -> RemoteConnectionTestInfo:
    result = await asyncio.to_thread(test_remote_connection, settings)
    return RemoteConnectionTestInfo(
        ok=result.ok,
        host=result.host,
        port=result.port,
        user=result.user,
        work_dir=result.work_dir,
        latency_ms=result.latency_ms,
        detail=result.detail,
        pwd=result.pwd,
    )


@app.get("/config/robot", response_model=RobotProfileInfo)
async def config_robot() -> RobotProfileInfo:
    profile = get_robot_profile()
    issues = validate_robot_profile(profile)
    return RobotProfileInfo(
        id=profile.id,
        label=profile.label,
        status=profile.status,
        dof=profile.dof,
        base_link=profile.base_link,
        joint_order=list(profile.joint_order),
        leg_order=list(profile.leg_order),
        foot_links=list(profile.foot_links),
        diagnostic_spec=profile.diagnostic_spec,
        capabilities=list(profile.capabilities),
        note=profile.note,
        valid=not issues,
        issues=issues,
    )


@app.get("/config/llm", response_model=LLMProfileInfo)
async def config_llm() -> LLMProfileInfo:
    return _llm_info()


@app.patch("/config/llm", response_model=LLMProfileUpdateResult)
async def config_llm_update(req: LLMProfileUpdate) -> LLMProfileUpdateResult:
    try:
        result = update_llm_profile(
            settings,
            model=req.model,
            base_url=req.base_url,
            api_key_env_var=req.api_key_env_var,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return LLMProfileUpdateResult(
        profile=_llm_info(),
        saved_path=result.saved_path,
        message=result.message,
    )


@app.post("/config/framework/select", response_model=FrameworkSelectionResult)
async def config_framework_select(req: FrameworkSelectionRequest) -> FrameworkSelectionResult:
    try:
        result = select_framework(settings, req.framework_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FrameworkSelectionResult(
        active_framework_id=result.active_framework_id,
        desired_framework_id=result.desired_framework_id,
        requires_restart=result.requires_restart,
        saved_path=result.saved_path,
        message=result.message,
    )


@app.get("/llm/audit", response_model=LLMAuditHistory)
async def llm_audit(limit: int = 20) -> LLMAuditHistory:
    return LLMAuditHistory(items=[
        LLMAuditItemInfo(
            file=item.file,
            schema_name=item.schema_name,
            model=item.model,
            elapsed_s=item.elapsed_s,
            attempt=item.attempt,
            ok=item.ok,
            error=item.error,
            created_at=item.created_at,
        )
        for item in recent_llm_audit(settings, limit=limit)
    ])


@app.get("/llm/readiness", response_model=LLMReadinessInfo)
async def llm_readiness_endpoint() -> LLMReadinessInfo:
    return LLMReadinessInfo(**llm_readiness(settings))


@app.post("/llm/workflows/system-audit", response_model=LLMWorkflowResult)
async def llm_workflow_system_audit(req: LLMWorkflowRequest) -> LLMWorkflowResult:
    data = await system_audit(settings, source, include_llm=req.include_llm)
    return _workflow_result(data)


@app.post("/llm/workflows/diagnostic-brief", response_model=LLMWorkflowResult)
async def llm_workflow_diagnostic_brief(req: LLMWorkflowRequest) -> LLMWorkflowResult:
    data = await diagnostic_brief(settings, source, job_id=req.job_id, include_llm=req.include_llm)
    return _workflow_result(data)


@app.get("/config/active", response_model=ConfigSetInfo)
async def config_active() -> ConfigSetInfo:
    config_set = get_active_config_set(settings)

    def summary(profile) -> ProfileSummaryInfo:
        return ProfileSummaryInfo(
            id=profile.id,
            label=profile.label,
            status=profile.status,
            detail=profile.detail,
        )

    return ConfigSetInfo(
        id=config_set.id,
        label=config_set.label,
        status=config_set.status,
        task_goal=config_set.task_goal,
        remote=summary(config_set.remote),
        robot=summary(config_set.robot),
        framework=summary(config_set.framework),
        llm=summary(config_set.llm),
        notes=list(config_set.notes),
    )


@app.get("/agent/workbench", response_model=AgentWorkbenchInfo)
async def agent_workbench() -> AgentWorkbenchInfo:
    config_set = get_active_config_set(settings)
    readiness = LLMReadinessInfo(**llm_readiness(settings))
    status, remote_status = await asyncio.gather(
        source.get_status(),
        source.remote_machine_status(),
        return_exceptions=True,
    )
    if isinstance(status, Exception):
        status = RunStatus(
            run_id="",
            task="",
            robot_id="",
            latest_iter=0,
            total_iter=0,
            running=False,
            source=settings.source,
            runtime_state="remote_unavailable",
            remote_ok=False,
            degraded=True,
            error=f"{type(status).__name__}: {status}",
        )
    if isinstance(remote_status, Exception):
        remote_status = RemoteMachineStatus(
            source=settings.source,
            available=False,
            error=f"{type(remote_status).__name__}: {remote_status}",
        )
    try:
        telemetry = await source.training_telemetry()
    except Exception as exc:  # noqa: BLE001 - workbench must stay readable when telemetry breaks
        telemetry = TrainingTelemetry(
            source=settings.source,
            available=False,
            runtime_state=status.runtime_state,
            running=status.running,
            run_id=status.run_id,
            telemetry_age_s=status.telemetry_age_s,
            mode="error",
            error=f"{type(exc).__name__}: {exc}",
            limitations=["telemetry read failed"],
        )

    snapshot = telemetry.snapshot
    latest = telemetry.latest
    remote_running = _remote_training_running(remote_status)
    effective_running = bool(status.running or telemetry.running or remote_running)
    effective_remote_ok = bool(status.remote_ok and not status.degraded and remote_status.available)
    evidence: list[AgentWorkbenchEvidenceInfo] = []
    evidence.append(AgentWorkbenchEvidenceInfo(
        id="llm",
        label="智能体模型",
        status="ok" if readiness.ok else "warning",
        detail=(f"{readiness.model} @ {readiness.base_url}" if readiness.ok else "; ".join(readiness.issues) or "LLM 未就绪"),
        source="/llm/readiness",
        values={
            "configured": readiness.configured,
            "api_key_resolved": readiness.api_key_resolved,
            "tools": len(readiness.tools),
            "workflows": readiness.workflow_count,
        },
    ))
    evidence.append(AgentWorkbenchEvidenceInfo(
        id="run",
        label="当前训练尝试",
        status="ok" if status.remote_ok and not status.degraded else "warning",
        detail=status.note or status.error or status.runtime_state,
        source="/run/current",
        values={
            "run_id": status.run_id,
            "running": status.running,
            "runtime_state": status.runtime_state,
            "latest_iter": status.latest_iter,
            "total_iter": status.total_iter,
            "telemetry_age_s": status.telemetry_age_s,
        },
    ))
    evidence.append(AgentWorkbenchEvidenceInfo(
        id="remote",
        label="远程机器状态",
        status="ok" if remote_status.available else "error",
        detail=_remote_status_detail(remote_status),
        source="/remote/status",
        values=_workbench_remote_values(remote_status),
    ))
    evidence.append(AgentWorkbenchEvidenceInfo(
        id="telemetry",
        label="训练遥测",
        status="ok" if telemetry.available and latest is not None else "missing",
        detail=telemetry.summary or telemetry.error or telemetry.mode,
        source="/run/current/telemetry",
        values=_workbench_metric_values(telemetry),
        provenance=snapshot.provenance if snapshot else [],
    ))
    evidence.append(AgentWorkbenchEvidenceInfo(
        id="config",
        label="配置与目标",
        status="ok" if config_set.status in {"draft", "validated"} else "unknown",
        detail=config_set.task_goal,
        source="/config/active",
        values={
            "config_set": config_set.id,
            "status": config_set.status,
            "remote": config_set.remote.id,
            "robot": config_set.robot.id,
            "framework": config_set.framework.id,
        },
    ))

    evidence_ids = [item.id for item in evidence]
    judgement = _build_workbench_judgement(status=status, telemetry=telemetry, evidence_ids=evidence_ids)
    proposals = [_proposal_info(item) for item in llm_session.proposals(limit=8)]
    pending = [item for item in proposals if item.status == "pending"]

    actions = [
        AgentWorkbenchActionInfo(
            id="ask-agent",
            label="让智能体基于证据判断下一步",
            kind="ask_agent",
            risk="low",
            enabled=readiness.ok,
            requires_confirmation=False,
            endpoint="/chat",
            reason="读取证据并生成解释或提案，不直接改变训练状态。",
            rollback="无状态变更。",
        ),
        AgentWorkbenchActionInfo(
            id="run-diagnostic",
            label="运行物理诊断",
            kind="diagnostic",
            risk="medium",
            enabled=effective_remote_ok,
            endpoint="/diagnostics/run",
            reason="用检查点回放验证方向、稳定性、地形和域随机化表现。",
            rollback="诊断只生成产物；可删除本地历史记录，不影响训练权重。",
        ),
        AgentWorkbenchActionInfo(
            id="deploy-payload",
            label="部署当前上传包",
            kind="deployment",
            risk="high",
            enabled=effective_remote_ok and not effective_running,
            endpoint="/action/deploy-payload",
            reason="把本地训练配置和运行代码同步到远端。",
            rollback="恢复上一版配置或上传包后重新部署。",
        ),
        AgentWorkbenchActionInfo(
            id="start-training",
            label="启动全新训练",
            kind="training",
            risk="high",
            enabled=effective_remote_ok and not effective_running,
            endpoint="/action/start",
            reason="用当前配置启动新训练尝试。",
            rollback="停止当前训练，并恢复上一版配置或检查点策略。",
        ),
        AgentWorkbenchActionInfo(
            id="resume-training",
            label="从检查点继续",
            kind="training",
            risk="high",
            enabled=effective_remote_ok and not effective_running,
            endpoint="/action/resume",
            reason="在已有检查点上继续训练。",
            rollback="停止当前训练；重新选择检查点或启动全新训练。",
        ),
        AgentWorkbenchActionInfo(
            id="kill-training",
            label="停止训练",
            kind="safety",
            risk="high",
            enabled=effective_remote_ok and effective_running,
            endpoint="/action/kill",
            reason="停止当前远端训练进程。",
            rollback="只能通过 resume/start 重新启动；已停止的进程不能原地恢复。",
        ),
    ]

    attempt = AgentWorkbenchAttemptInfo(
        run_id=status.run_id or telemetry.run_id,
        running=effective_running,
        runtime_state=status.runtime_state,
        source=status.source,
        step=snapshot.step if snapshot else status.latest_iter,
        total_steps=snapshot.total_steps if snapshot else status.total_iter,
        progress_pct=snapshot.progress_pct if snapshot else None,
        phase=snapshot.phase if snapshot else str((latest.curriculum or {}).get("phase") if latest else ""),
        command_mode=snapshot.command_mode if snapshot else "",
        blocked_by=snapshot.blocked_by if snapshot else str((latest.curriculum or {}).get("blocked_by") if latest else ""),
        latest_checkpoint=snapshot.latest_checkpoint if snapshot else (latest.checkpoint if latest else ""),
        telemetry_age_s=status.telemetry_age_s if status.telemetry_age_s is not None else telemetry.telemetry_age_s,
        remote_ok=effective_remote_ok,
        summary=snapshot.conclusion if snapshot else telemetry.summary,
    )

    notes = [
        "智能体工作台是主服务入口；训练、诊断、配置只是证据和工具。",
        "会改变训练状态的动作仍必须走确认门或显式按钮。",
    ]
    if pending:
        notes.append(f"当前有 {len(pending)} 个待授权提案。")
    if telemetry.limitations:
        notes.extend(telemetry.limitations[:3])

    return AgentWorkbenchInfo(
        generated_at=time.time(),
        objective=config_set.task_goal,
        service_loop=["目标", "证据", "判断", "提案", "授权执行", "验证", "记录"],
        attempt=attempt,
        llm=readiness,
        judgement=judgement,
        evidence=evidence,
        proposals=proposals,
        actions=actions,
        notes=notes,
    )


@app.get("/run/current", response_model=RunStatus)
async def run_current() -> RunStatus:
    return await source.get_status()


@app.get("/run/current/telemetry", response_model=TrainingTelemetry)
async def run_current_telemetry() -> TrainingTelemetry:
    return await source.training_telemetry()


@app.get("/run/current/snapshot", response_model=RunSnapshot)
async def run_current_snapshot() -> RunSnapshot:
    telemetry = await source.training_telemetry()
    if telemetry.snapshot is None:
        raise HTTPException(status_code=404, detail="No run snapshot available.")
    return telemetry.snapshot


@app.get("/run/current/acceptance")
async def run_current_acceptance() -> dict:
    """Benchmark verdict vs docs/taili_spec.md for the newest run — scored from persisted physeval
    logs (read-only; never launches physeval). Returns {available: False, reason} when none exist yet."""
    return await asyncio.to_thread(source.get_acceptance)


@app.get("/remote/status", response_model=RemoteMachineStatus)
async def remote_status() -> RemoteMachineStatus:
    return await source.remote_machine_status()


def _ws_authorized(ws: WebSocket) -> bool:
    """Cross-site WebSocket-hijack guard for the READ-ONLY telemetry stream. Policy mirrors HTTP
    reads: a SAME-ORIGIN browser page is always allowed (the old allowlist only knew the dev :5173
    origin, so the single-port :8000 UI was silently rejected); a foreign Origin is allowed only
    with the shared token (?token=, browsers can't set WS headers); a non-browser client (no
    Origin) needs loopback or the token."""
    origin = ws.headers.get("origin")
    token_ok = bool(_CONSOLE_TOKEN) and hmac.compare_digest(
        ws.query_params.get("token", ""), _CONSOLE_TOKEN)
    if origin is not None:
        same_origin = origin.split("://", 1)[-1] == ws.headers.get("host", "")
        allowed = {settings.ui_origin, "http://127.0.0.1:5173"}
        return same_origin or origin in allowed or token_ok
    host = ws.client.host if ws.client else ""
    return host in _LOOPBACK_HOSTS or token_ok


@app.websocket("/run/current/stream")
async def run_stream(ws: WebSocket) -> None:
    if not _ws_authorized(ws):
        await ws.close(code=1008)  # policy violation
        return
    await ws.accept()
    stop = asyncio.Event()
    try:
        async for point in source.stream(stop):
            await ws.send_json(point.model_dump())
    except WebSocketDisconnect:
        stop.set()
    except Exception as e:  # noqa: BLE001 - surface stream errors to the client, don't 500 silently
        stop.set()
        try:
            await ws.send_json({"error": type(e).__name__, "message": str(e)})
        except Exception:
            pass
    finally:
        stop.set()


@app.get("/run/current/tensorboard/scalars", response_model=TensorboardScalarCatalog)
async def run_tensorboard_scalars() -> TensorboardScalarCatalog:
    return await source.tensorboard_scalar_catalog()


@app.get("/run/current/tensorboard/series", response_model=TensorboardSeriesResponse)
async def run_tensorboard_series(
    tags: list[str] = Query(default_factory=list),
    max_points: int = 1200,
) -> TensorboardSeriesResponse:
    return await source.tensorboard_scalar_series(tags, max_points=max_points)


@app.get("/definitions", response_model=DefinitionQueryResult)
async def definitions(query: str = "") -> DefinitionQueryResult:
    return list_definitions(query)


@app.get("/spec/coverage", response_model=SpecCoverageReport)
async def spec_coverage() -> SpecCoverageReport:
    return build_spec_coverage_report()


# ── concurrency: serialize training-state mutations so two can't clobber each other (two starts
# racing the tmux session, a config edit racing a deploy). Reads/telemetry/stream stay OUTSIDE it. ──
_ACTION_LOCK = asyncio.Lock()


@contextlib.asynccontextmanager
async def _action_lock():
    try:
        await asyncio.wait_for(_ACTION_LOCK.acquire(), timeout=0.5)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=409,
                            detail="another state-changing operation is in progress; retry shortly")
    try:
        yield
    finally:
        _ACTION_LOCK.release()


@app.post("/action/deploy-payload", response_model=ActionResult)
async def action_deploy_payload() -> ActionResult:
    async with _action_lock():
        return await source.action_deploy_payload()


@app.post("/action/resume", response_model=ActionResult)
async def action_resume() -> ActionResult:
    async with _action_lock():
        return await source.action_resume()


@app.post("/action/start", response_model=ActionResult)
async def action_start() -> ActionResult:
    async with _action_lock():
        return await source.action_start()


@app.post("/action/kill", response_model=ActionResult)
async def action_kill() -> ActionResult:
    async with _action_lock():
        return await source.action_kill()


@app.post("/action/physeval", response_model=PhysevalResult)
async def action_physeval() -> PhysevalResult:
    return await source.action_physeval()


@app.get("/diagnostics/catalog", response_model=DiagnosticCatalog)
async def diagnostics_catalog() -> DiagnosticCatalog:
    return await diagnostics.catalog()


@app.post("/diagnostics/run", response_model=DiagnosticJobStatus)
async def diagnostics_run(req: DiagnosticRunRequest) -> DiagnosticJobStatus:
    try:
        return await diagnostics.start(req.preset, req.checkpoint, req.plan)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/diagnostics/status", response_model=DiagnosticJobStatus)
async def diagnostics_status() -> DiagnosticJobStatus:
    return await diagnostics.status()


@app.get("/diagnostics/history", response_model=DiagnosticHistory)
async def diagnostics_history(limit: int = 30) -> DiagnosticHistory:
    return diagnostics.history.list(limit=limit)


@app.patch("/diagnostics/history/{job_id}", response_model=DiagnosticHistoryItem)
async def diagnostics_history_update(job_id: str, req: DiagnosticHistoryUpdate) -> DiagnosticHistoryItem:
    item = diagnostics.history.update(job_id, note=req.note, archived=req.archived)
    if item is None:
        raise HTTPException(status_code=404, detail="Diagnostic job not found")
    return item


@app.delete("/diagnostics/history/{job_id}", response_model=DiagnosticHistoryDeleteResult)
async def diagnostics_history_delete(job_id: str) -> DiagnosticHistoryDeleteResult:
    ok = diagnostics.history.delete(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Diagnostic job not found")
    return DiagnosticHistoryDeleteResult(
        ok=True,
        job_id=job_id,
        message="Removed local diagnostic history entry. Remote artifacts were not deleted.",
    )


@app.post("/diagnostics/cancel", response_model=DiagnosticJobStatus)
async def diagnostics_cancel() -> DiagnosticJobStatus:
    return await diagnostics.cancel()


@app.get("/diagnostics/report", response_model=DiagnosticReport)
async def diagnostics_report(job_id: str | None = None) -> DiagnosticReport:
    try:
        return await diagnostics.report_for_job(job_id=job_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/diagnostics/playback", response_model=DiagnosticPlayback)
async def diagnostics_playback(max_frames: int = 900, job_id: str | None = None) -> DiagnosticPlayback:
    return await diagnostics.playback_for_job(max_frames=max_frames, job_id=job_id)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """The soul: the LLM operates the locomotion console's read-only tools to answer in plain language."""
    import asyncio as _asyncio

    from .agent import run_agent

    started = time.perf_counter()
    original_message = req.message or ""
    stripped_message = original_message.strip()
    force_llm = False
    agent_message = original_message
    lowered_message = stripped_message.lower()
    for command in ("/ask", "/llm"):
        if lowered_message == command or lowered_message.startswith(f"{command} "):
            force_llm = True
            agent_message = stripped_message[len(command):].strip()
            break
    if force_llm and not agent_message:
        return ChatResponse(
            reply="用法：/ask <问题> 或 /llm <问题>。这会跳过快速 slash 层，强制走 LLM Agent。",
            steps=0,
            mode="slash_usage",
            elapsed_s=_chat_elapsed(started),
            transcript=[{"tool": "slash_usage", "args": {"command": stripped_message}, "result": {"prefix": "/ask"}}],
            grounding={"checked": True, "grounded": True, "unsupported": [], "mode": "slash_usage"},
        )
    history = llm_session.recent_turns(limit=16)
    llm_session.append_turn(role="user", content=original_message)
    try:
        slash = None if force_llm else await _asyncio.to_thread(handle_slash_command, original_message, settings, source)
    except Exception as exc:  # noqa: BLE001
        reply = f"命令处理失败：{type(exc).__name__}: {exc}"
        transcript = [{
            "tool": "slash_command",
            "args": {"message": original_message},
            "result": {"ok": False, "error": reply},
        }]
        grounding = {
            "checked": True,
            "grounded": True,
            "unsupported": [],
            "mode": "slash_error",
        }
        llm_session.append_turn(role="assistant", content=reply, transcript=transcript, grounding=grounding)
        return ChatResponse(
            reply=reply,
            steps=0,
            transcript=transcript,
            grounding=grounding,
            mode="slash_error",
            elapsed_s=_chat_elapsed(started),
            suggestions=["/help", "/diag history", "/ask 解释刚才的错误"],
        )
    if slash is not None:
        response_mode = _slash_response_mode(slash)
        transcript = list(slash.get("transcript", []) or [])
        grounding = {
            "checked": True,
            "grounded": True,
            "unsupported": [],
            "mode": response_mode,
        }
        proposal_id = None
        if slash.get("proposed_action"):
            proposal = llm_session.record_proposal(
                reply=slash.get("reply", ""),
                proposed_action=slash.get("proposed_action") or {},
            )
            proposal_id = str(proposal.get("id") or "")
        llm_session.append_turn(
            role="assistant",
            content=str(slash.get("reply", "")),
            transcript=slash.get("transcript", []),
            proposed_action=slash.get("proposed_action"),
            grounding=grounding,
        )
        return ChatResponse(
            reply=str(slash.get("reply", "")),
            steps=int(slash.get("steps", 0) or 0),
            transcript=transcript,
            proposed_action=slash.get("proposed_action"),
            grounding=grounding,
            proposal_id=proposal_id,
            mode=response_mode,
            elapsed_s=_chat_elapsed(started),
            suggestions=_chat_suggestions(response_mode, transcript, req.ui_mode),
        )
    if not force_llm and _fast_status_intent(agent_message, req.ui_mode):
        response = await _fast_training_status_response(req, started)
        llm_session.append_turn(
            role="assistant",
            content=response.reply,
            transcript=response.transcript,
            grounding=response.grounding,
        )
        return response
    # run_agent does blocking SSH + LLM calls -> off the event loop; pass prior dialogue
    remote_for_context = None
    if should_probe_remote_for_context(message=agent_message, ui_mode=req.ui_mode, context_request=req.context_request):
        try:
            remote_for_context = await _asyncio.wait_for(source.remote_machine_status(), timeout=8.0)
        except Exception:  # noqa: BLE001 - the envelope can still route without live remote state
            remote_for_context = None
    context_envelope = build_context_envelope(
        message=agent_message,
        ui_mode=req.ui_mode,
        legacy_context=req.context,
        context_request=req.context_request,
        remote_status=remote_for_context,
    )
    agent_message_with_context = _chat_agent_message(
        agent_message,
        req.ui_mode,
        req.context,
        context_envelope,
    )
    max_agent_steps = int(os.environ.get("LOCOMOTION_CONSOLE_AGENT_MAX_STEPS", "5"))
    try:
        out = await _asyncio.wait_for(
            _asyncio.to_thread(
                run_agent,
                agent_message_with_context,
                settings,
                history,
                not force_llm,
                max_agent_steps,
                force_llm or req.ui_mode in {"training", "diagnostics"},
            ),
            timeout=float(os.environ.get("LOCOMOTION_CONSOLE_AGENT_TIMEOUT_S", "70")),
        )
    except _asyncio.TimeoutError:
        reply = (
            "LLM Agent 超时：模型或远程只读工具没有在时限内返回。"
            "当前请求没有执行任何动作。建议先用快速命令 `/why blocked`、`/status`、`/probe telemetry` 获取确定信息；"
            "需要深度分析再用 `/ask <具体问题>`。"
        )
        transcript = [{
            "tool": "agent_timeout",
            "args": {
                "timeout_s": float(os.environ.get("LOCOMOTION_CONSOLE_AGENT_TIMEOUT_S", "70")),
                "context_sources": context_envelope.selected_sources,
            },
            "result": {"ok": False, "message": reply},
        }]
        llm_session.append_turn(
            role="assistant",
            content=reply,
            transcript=transcript,
            grounding={"checked": False, "grounded": None, "unsupported": [], "mode": "agent_timeout"},
        )
        return ChatResponse(
            reply=reply,
            steps=0,
            transcript=transcript,
            grounding={"checked": False, "grounded": None, "unsupported": [], "mode": "agent_timeout"},
            mode="agent_timeout",
            context_envelope=context_envelope,
            elapsed_s=_chat_elapsed(started),
            suggestions=["/why blocked", "/status", "/probe telemetry"],
        )
    # grounding self-check (skipped when an action was proposed - those aren't factual claims)
    grounding = {
        "checked": False,
        "grounded": None,
        "unsupported": [],
        "mode": "skipped_for_latency",
    }
    if (
        not out.get("proposed_action")
        and os.environ.get("LOCOMOTION_CONSOLE_VERIFY_GROUNDING", "0").lower() in {"1", "true", "yes"}
    ):
        from .agent import verify_grounding
        try:
            grounding = await _asyncio.wait_for(
                _asyncio.to_thread(verify_grounding, out["reply"], out.get("transcript", []), settings),
                timeout=float(os.environ.get("LOCOMOTION_CONSOLE_GROUNDING_TIMEOUT_S", "12")),
            )
        except Exception as exc:  # noqa: BLE001
            grounding = {
                "checked": False,
                "grounded": None,
                "unsupported": [],
                "mode": "grounding_timeout",
                "error": f"{type(exc).__name__}: {exc}",
            }
    proposal_id = None
    # audit trail for AUTONOMY-executed actions (the brain ran low/medium-risk actions inline):
    # record each as an executed proposal so the proposal ledger shows what the copilot did.
    for done in out.get("executed_actions") or []:
        res = done.get("result") or {}
        rec = llm_session.record_proposal(
            reply=f"[autonomy] executed {done.get('action')}",
            proposed_action={"name": done.get("action"), "args": done.get("args") or {},
                             "kind": "action", "risk": "auto-executed"},
        )
        llm_session.update_latest_pending(
            name=str(done.get("action") or ""), ok=bool(res.get("ok")),
            detail=str(res.get("detail", ""))[:400])
    if out.get("proposed_action"):
        proposal = llm_session.record_proposal(
            reply=out.get("reply", ""),
            proposed_action=out.get("proposed_action") or {},
        )
        proposal_id = str(proposal.get("id") or "")
    llm_session.append_turn(
        role="assistant",
        content=out.get("reply", ""),
        transcript=out.get("transcript", []),
        proposed_action=out.get("proposed_action"),
        grounding=grounding,
    )
    response_mode = "agent_llm_forced" if force_llm else str(out.get("mode") or "agent_llm")
    transcript = list(out.get("transcript", []) or [])
    return ChatResponse(reply=out["reply"], steps=out.get("steps", 0),
                        transcript=transcript,
                        proposed_action=out.get("proposed_action"),
                        grounding=grounding,
                        proposal_id=proposal_id,
                        context_envelope=context_envelope,
                        mode=response_mode,
                        elapsed_s=_chat_elapsed(started),
                        suggestions=_chat_suggestions(response_mode, transcript, req.ui_mode))


@app.get("/chat/proposals", response_model=ChatProposalHistory)
async def chat_proposals(limit: int = 30) -> ChatProposalHistory:
    return ChatProposalHistory(items=[
        _proposal_info(item)
        for item in llm_session.proposals(limit=limit)
    ])


@app.post("/chat/context/route", response_model=ContextEnvelopeInfo)
async def chat_context_route(req: ChatRequest) -> ContextEnvelopeInfo:
    remote_for_context = None
    if should_probe_remote_for_context(message=req.message, ui_mode=req.ui_mode, context_request=req.context_request):
        try:
            remote_for_context = await asyncio.wait_for(source.remote_machine_status(), timeout=8.0)
        except Exception:  # noqa: BLE001 - route inspection should still work offline
            remote_for_context = None
    return build_context_envelope(
        message=req.message,
        ui_mode=req.ui_mode,
        legacy_context=req.context,
        context_request=req.context_request,
        remote_status=remote_for_context,
    )


@app.post("/chat/proposals/cancel", response_model=ChatProposalInfo)
async def chat_proposal_cancel(req: CancelProposalRequest) -> ChatProposalInfo:
    item = llm_session.mark_latest_pending_cancelled(name=req.name)
    if item is None:
        raise HTTPException(status_code=404, detail="No pending proposal found")
    return _proposal_info(item)


@app.post("/chat/reset")
async def chat_reset() -> dict:
    llm_session.reset()
    return {"ok": True}


@app.post("/chat/execute", response_model=ExecuteResponse)
async def chat_execute(req: ExecuteRequest) -> ExecuteResponse:
    """Run a confirmed action. Reached only when the operator confirms an LLM proposal in the UI.

    Two safety gates beyond the mutation-token middleware: (1) the action name must be known, and
    (2) PROPOSAL-BINDING — a still-pending proposal for exactly this action must exist, so a raw
    token holder cannot POST an un-proposed destructive action (e.g. kill_training on a converging
    run). The action + its risk tier + the bound proposal id are written to the audit trail."""
    import asyncio as _asyncio

    from .agent import execute_action, action_risk, ACTION_NAMES

    if req.name not in ACTION_NAMES:
        raise HTTPException(status_code=400, detail=f"unknown action: {req.name}")
    pending = llm_session.find_pending(name=req.name)
    if pending is None:
        raise HTTPException(
            status_code=409,
            detail=(f"no pending '{req.name}' proposal to confirm — the copilot must propose this "
                    f"action before it can be executed"),
        )
    tier = action_risk(req.name)
    # serialize the actual execution under the same mutation lock as /action/* so a copilot-confirmed
    # action can't race a direct one (e.g. two starts, or edit_config vs deploy).
    async with _action_lock():
        out = await _asyncio.to_thread(execute_action, req.name, req.args, settings)
    ok, detail = bool(out.get("ok")), str(out.get("detail", ""))
    # record the action + result so the soul remembers it did this
    llm_session.update_latest_pending(name=req.name, ok=ok, detail=detail)
    llm_session.append_turn(
        role="assistant",
        content=(f"[executed {req.name} risk={tier} proposal={pending.get('id')}] "
                 f"{'ok' if ok else 'failed'}: {detail}"),
    )
    return ExecuteResponse(ok=ok, detail=detail)


# ── serve the built UI single-origin (mounted LAST so every API route above wins) ──────────
# The built frontend uses a same-origin relative API base, so serving it from this app means the
# UI, REST, and WebSocket all share one origin — no CORS, no proxy, reachable via any address the
# server binds. Mounting only happens if the build exists (npm run build -> frontend/dist).
from pathlib import Path as _Path  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

_FRONTEND_DIST = _Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
