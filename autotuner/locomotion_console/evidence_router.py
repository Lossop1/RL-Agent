"""Evidence routing for the locomotion-console LLM.

The LLM should not decide by itself which project files or remote paths to scan.
This module is the deterministic front door: it maps an operator question to an
allowlisted set of evidence sources, plus explicit gap checks.  The answer can
remain natural and flexible; evidence collection is strict.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class EvidenceSource:
    id: str
    label: str
    kind: str
    capabilities: tuple[str, ...]
    limitations: tuple[str, ...]
    freshness: str


_SOURCES: dict[str, EvidenceSource] = {
    "training_telemetry": EvidenceSource(
        id="training_telemetry",
        label="Live training telemetry / RunSnapshot",
        kind="remote_readonly",
        capabilities=(
            "current run id, running/stale state, step, fps, ETA, checkpoint",
            "latest reward/curriculum/health/command fields",
            "metric groups, blockers, provenance, telemetry/log paths",
        ),
        limitations=(
            "training metrics are optimization proxies, not proof of real gait quality",
            "stale or missing telemetry cannot support current-state claims",
        ),
        freshness="live; must check stale/telemetry_age_s",
    ),
    "remote_status": EvidenceSource(
        id="remote_status",
        label="Remote machine status",
        kind="remote_readonly",
        capabilities=(
            "GPU memory/utilization/temperature/processes",
            "RAM, disk, tmux sessions, likely training processes",
            "SSH/backend availability symptoms",
        ),
        limitations=(
            "does not explain reward or gait quality by itself",
            "a remote failure is a system-state fact, not a training-quality fact",
        ),
        freshness="live; query when resources or connectivity are mentioned",
    ),
    "telemetry_provenance": EvidenceSource(
        id="telemetry_provenance",
        label="Telemetry/log/checkpoint provenance probe",
        kind="remote_probe",
        capabilities=(
            "exact train.log, train.telemetry.jsonl, console.log, checkpoint_dir paths",
            "presence, line counts, sizes, and discovery evidence",
        ),
        limitations=(
            "proves wiring and artifact presence, not whether the policy is good",
        ),
        freshness="live; use for source/path/log disputes",
    ),
    "train_log": EvidenceSource(
        id="train_log",
        label="Training log tail",
        kind="remote_readonly",
        capabilities=(
            "recent semantic [TP*] lines and console stdout/stderr tail",
            "launcher/runtime error clues",
        ),
        limitations=(
            "log tail is partial and can miss earlier causes",
            "telemetry JSONL remains the primary current-state source",
        ),
        freshness="live tail",
    ),
    "taili_context": EvidenceSource(
        id="taili_context",
        label="Taili spec/strategy/YAML knowledge pack",
        kind="local_allowlist",
        capabilities=(
            "taili_spec, strategy decisions, architecture notes, current Taili YAML contract",
            "network/reward/curriculum/control configuration intent",
        ),
        limitations=(
            "docs/YAML describe intent and configuration, not current observed behavior",
            "do not infer success from YAML without telemetry or diagnostics",
        ),
        freshness="stable until files change",
    ),
    "spec_coverage": EvidenceSource(
        id="spec_coverage",
        label="taili_spec coverage ledger",
        kind="local_allowlist",
        capabilities=(
            "acceptance rows mapped to current mechanisms, evaluation coverage, gaps",
            "what must exist before claiming spec compliance",
        ),
        limitations=(
            "coverage is a mechanism ledger, not a pass/fail result for the current checkpoint",
        ),
        freshness="stable until code/docs change",
    ),
    "acceptance_verdict": EvidenceSource(
        id="acceptance_verdict",
        label="Measured taili_spec §2 acceptance verdict",
        kind="remote_readonly",
        capabilities=(
            "MEASURED pass/fail per acceptance family (A1..F2) for the newest run's checkpoint",
            "failing sub-gates with the exact measured stat vs spec band",
            "the single metric the product is graded on — how far the policy is from benchmark",
        ),
        limitations=(
            "only as fresh as the last physeval scored for this run; unavailable until one is run",
            "reflects the scored terrains only (missing families = not yet evaluated, not passing)",
        ),
        freshness="per-checkpoint; scored from persisted physeval logs, never launches physeval",
    ),
    "definitions": EvidenceSource(
        id="definitions",
        label="Metric/reward definition registry",
        kind="local_registry",
        capabilities=(
            "formula, source, meaning, related fields for telemetry/reward/diagnostic terms",
            "current reading when paired with telemetry",
        ),
        limitations=(
            "definitions explain fields; they do not prove the field reflects reality",
        ),
        freshness="stable until code changes",
    ),
    "diagnostic_history": EvidenceSource(
        id="diagnostic_history",
        label="Diagnostic history",
        kind="local_registry",
        capabilities=(
            "recent diagnostic jobs, presets, checkpoints, states, artifacts",
            "which reports are available for behavior explanation",
        ),
        limitations=(
            "history alone does not describe behavior; completed reports are needed",
        ),
        freshness="local state; latest jobs first",
    ),
    "diagnostic_reports": EvidenceSource(
        id="diagnostic_reports",
        label="Completed diagnostic report(s)",
        kind="local_artifact",
        capabilities=(
            "observed robot behavior under commanded tests",
            "per-command tracking/stability/slip/impact/posture aggregates",
            "diagnostic value explanations and behavior context",
        ),
        limitations=(
            "diagnostics are checkpoint/test specific; do not generalize to other runs without evidence",
            "aggregate reports cannot replace watching playback when visual confirmation is required",
        ),
        freshness="stable artifact; must match selected/latest job",
    ),
    "workspace_config": EvidenceSource(
        id="workspace_config",
        label="Console workspace/config set",
        kind="local_registry",
        capabilities=(
            "active ConfigSet, framework, robot, remote profile, diagnostic task, LLM readiness",
            "whether system configuration explains UI/LLM/backend symptoms",
        ),
        limitations=(
            "does not contain live training-quality metrics",
        ),
        freshness="stable until operator changes settings",
    ),
    "frameworks": EvidenceSource(
        id="frameworks",
        label="Framework registry",
        kind="local_registry",
        capabilities=(
            "active/saved framework ids and diagnostic task bindings",
        ),
        limitations=(
            "registry entries are configuration metadata, not runtime proof",
        ),
        freshness="stable until settings change",
    ),
    "deployed_config": EvidenceSource(
        id="deployed_config",
        label="Remote deployed env config snippet",
        kind="remote_readonly",
        capabilities=(
            "current deployed env_cfg lines for a specific reward/parameter grep",
        ),
        limitations=(
            "grep snippets are incomplete; YAML is the local strategy contract",
            "requires a known profile path and remote connectivity",
        ),
        freshness="live remote file read",
    ),
    "asset": EvidenceSource(
        id="asset",
        label="Robot asset/actuator snippet",
        kind="remote_readonly",
        capabilities=(
            "actuator stiffness/damping/effort/velocity/mass/default-pose clues",
        ),
        limitations=(
            "asset snippets do not show current policy behavior by themselves",
        ),
        freshness="live remote file read",
    ),
    "run_ledger": EvidenceSource(
        id="run_ledger",
        label="Recent run ledger",
        kind="remote_readonly",
        capabilities=(
            "recent runs, max checkpoint iter, reward peak/final",
            "comparisons between recent training attempts",
        ),
        limitations=(
            "TensorBoard reward summaries are historical and can hide current proxy/diagnostic mismatch",
        ),
        freshness="live remote summary",
    ),
    "code_knowledge": EvidenceSource(
        id="code_knowledge",
        label="Taili code knowledge / implementation evidence",
        kind="local_allowlist",
        capabilities=(
            "allowlisted snippets from Taili reward, curriculum, diagnostics, telemetry, model, payload code",
            "literal evidence for whether YAML keys and telemetry fields are consumed in code",
            "source paths/line windows for implementation-level claims",
        ),
        limitations=(
            "not an arbitrary repository search and not a full static call graph",
            "literal occurrence is wiring evidence, not semantic proof",
            "must be paired with telemetry/diagnostics for runtime behavior",
        ),
        freshness="stable until source files change",
    ),
    "signal_map": EvidenceSource(
        id="signal_map",
        label="Spec/YAML/code/telemetry/diagnostic signal map",
        kind="local_curated_index",
        capabilities=(
            "maps spec items to YAML keys, code refs, telemetry fields, diagnostic fields",
            "shows what evidence is required before tuning claims",
            "helps prevent missing a source layer",
        ),
        limitations=(
            "curated index; update when new fields or files are added",
            "not a substitute for code tests or diagnostic runs",
        ),
        freshness="stable until source/docs change",
    ),
    "tuning_ledger": EvidenceSource(
        id="tuning_ledger",
        label="Tuning experiment ledger",
        kind="local_session_summary",
        capabilities=(
            "recent LLM session turns, proposals, evidence tools, runs, checkpoints, diagnostic job ids",
            "operator/assistant tuning memory across backend restarts",
            "open proposed actions and prior results",
        ),
        limitations=(
            "records discussion/proposals, not verified policy improvement",
            "old turns may contain stale facts; verify with current telemetry/diagnostics",
        ),
        freshness="local session state; newest turns first",
    ),
}


_LIVE_TERMS = (
    "现在", "当前", "情况", "状态", "训练", "运行", "跑", "step", "telemetry",
    "监控", "奖励", "reward", "phase", "phi", "phil", "gate", "卡住", "卡在",
    "收敛", "瓶颈", "进度", "checkpoint", "检查点", "fps", "eta",
)
_REMOTE_TERMS = (
    "显存", "gpu", "vram", "nvidia", "慢", "速度", "ssh", "连不上", "后端",
    "backend", "timeout", "超时", "tmux", "重启", "资源", "内存", "磁盘",
    "deploy", "部署", "上传", "启动训练",
)
_PROVENANCE_TERMS = (
    "日志", "log", "jsonl", "tfevents", "路径", "来源", "读的", "真数据",
    "显示", "不一致", "checkpoint", "检查点", "tmux attach", "终端",
)
_STRATEGY_TERMS = (
    "yaml", "配置", "策略", "调参", "参数", "权重", "奖励", "惩罚", "课程",
    "网络", "actor", "critic", "perceiver", "感知器", "amp", "模型", "结构",
    "height", "高度", "gait", "步态", "slip", "打滑", "impact", "冲击",
)
_CODE_TERMS = (
    "代码", "源码", "实现", "计算链", "调用链", "消费", "接线", "有没有用",
    "是否生效", "死键", "字段来源", "哪里计算", "哪里用", "映射", "对齐",
    "code", "source", "implementation", "consumer", "consumed", "wired", "dead key",
)
_LEDGER_TERMS = (
    "之前", "刚才", "上次", "历史", "记录", "ledger", "实验", "闭环",
    "改过", "做过", "提案", "proposal", "结果", "复盘",
)
_SPEC_TERMS = ("spec", "验收", "要求", "目标", "达标", "覆盖", "taili_spec")
_DIAG_TERMS = (
    "诊断", "回放", "测试", "怎么走", "表现", "前进", "后退", "forward",
    "backward", "lateral", "yaw", "偏航", "滑", "撞", "拖", "蹲", "步态差",
    "report", "job_id", "失败了",
)
_DEFINITION_TERMS = (
    "什么意思", "怎么算", "定义", "公式", "字段", "指标", "含义",
    "source", "formula", "metric",
)
_EXPLANATION_TERMS = ("为什么", "解释")
_SYSTEM_TERMS = (
    "系统", "ui", "界面", "页面", "llm", "对话", "知识路由", "上下文",
    "智能", "流畅", "丝滑", "后端", "configset", "framework", "adapter",
)
_ASSET_TERMS = (
    "urdf", "mesh", "meshes", "关节", "joint", "kp", "kd", "stiffness",
    "damping", "actuator", "执行器", "力矩", "effort", "mass", "质量",
)
_ACTION_TERMS = ("改", "修改", "怎么改", "修", "优化", "调参", "建议", "下一步")


def source_registry() -> dict[str, dict[str, Any]]:
    """Return source capability metadata in a JSON-friendly form."""
    return {
        key: {
            "id": source.id,
            "label": source.label,
            "kind": source.kind,
            "capabilities": list(source.capabilities),
            "limitations": list(source.limitations),
            "freshness": source.freshness,
        }
        for key, source in _SOURCES.items()
    }


def route_evidence(query: str = "", ui_mode: str = "") -> dict[str, Any]:
    """Map a user question to allowlisted evidence sources.

    This is intentionally deterministic.  The LLM may later phrase the answer in
    a flexible way, but it should not skip these selected sources or invent facts
    when they are unavailable.
    """
    q = _normalize(f"{ui_mode}\n{query}")
    selected: dict[str, dict[str, Any]] = {}
    decomposed: list[str] = []
    gap_checks: list[dict[str, str]] = []

    def add(source_id: str, reason: str, *, required: bool = True) -> None:
        source = _SOURCES[source_id]
        if source_id not in selected:
            selected[source_id] = {
                "id": source_id,
                "label": source.label,
                "kind": source.kind,
                "required": required,
                "reasons": [reason],
                "capabilities": list(source.capabilities),
                "limitations": list(source.limitations),
                "freshness": source.freshness,
            }
        elif reason not in selected[source_id]["reasons"]:
            selected[source_id]["reasons"].append(reason)
            selected[source_id]["required"] = bool(selected[source_id]["required"] or required)

    ui = _normalize(ui_mode or _extract_ui_mode(query))
    asks_live = _has(q, _LIVE_TERMS) or ui == "training"
    asks_remote = _has(q, _REMOTE_TERMS)
    asks_provenance = _has(q, _PROVENANCE_TERMS)
    asks_strategy = _has(q, _STRATEGY_TERMS)
    asks_spec = _has(q, _SPEC_TERMS)
    asks_diag = _has(q, _DIAG_TERMS) or ui == "diagnostics" or bool(_job_ids(query))
    asks_comparison = _has(q, ("对不上", "不一致", "差别", "差距", "误判", "看起来不错", "但是诊断"))
    asks_definition = _has(q, _DEFINITION_TERMS) or (
        _has(q, _EXPLANATION_TERMS)
        and (asks_live or asks_strategy or asks_diag or asks_spec or asks_comparison)
    )
    asks_system = _has(q, _SYSTEM_TERMS)
    asks_asset = _has(q, _ASSET_TERMS)
    asks_code = _has(q, _CODE_TERMS)
    asks_ledger = _has(q, _LEDGER_TERMS)
    asks_action = _has(q, _ACTION_TERMS)
    asks_tuning_root_cause = (
        asks_action
        and (asks_strategy or asks_live or asks_diag or asks_spec or "为什么" in q)
    ) or "调参" in q

    if asks_live:
        add("training_telemetry", "question refers to current training state, metrics, reward, phase, or dashboard")
        add("definitions", "live telemetry fields need formulas and limits to avoid proxy-metric confusion", required=False)
        decomposed.append("read the latest RunSnapshot/telemetry before making current-state claims")
        gap_checks.append({
            "source": "training_telemetry",
            "blocking": True,
            "gap": "telemetry unavailable or stale",
            "impact": "current run/phase/reward/fps conclusions are not supported",
        })

    if asks_remote:
        add("remote_status", "question mentions remote resources, speed, GPU/VRAM, tmux, SSH, backend, deploy, or timeout")
        decomposed.append("separate machine/resource state from policy-quality state")
        gap_checks.append({
            "source": "remote_status",
            "blocking": True,
            "gap": "remote status unavailable",
            "impact": "cannot conclude GPU/SSH/tmux/resource state",
        })

    if asks_provenance:
        add("telemetry_provenance", "question challenges log/checkpoint/telemetry source or monitor wiring")
        if "日志" in q or "log" in q or "终端" in q or "tmux attach" in q:
            add("train_log", "log tail is needed to explain terminal/log visibility", required=False)
        decomposed.append("verify exact artifacts before explaining what the monitor is showing")
        gap_checks.append({
            "source": "telemetry_provenance",
            "blocking": True,
            "gap": "artifact paths or presence checks missing",
            "impact": "cannot prove which log/jsonl/checkpoint the UI is using",
        })

    if asks_strategy:
        add("taili_context", "question is about Taili YAML, reward/curriculum/model/strategy, or tuning intent")
        add("definitions", "reward/gate/metric terms need deterministic formulas")
        add("signal_map", "strategy/config questions need the spec-YAML-code-telemetry-diagnostic mapping")
        decomposed.append("distinguish strategy/config intent from observed behavior")
        gap_checks.append({
            "source": "taili_context",
            "blocking": True,
            "gap": "YAML/docs unavailable",
            "impact": "cannot make project-specific strategy/config claims",
        })

    if asks_spec or asks_tuning_root_cause:
        add("spec_coverage", "spec acceptance and coverage gaps constrain tuning decisions")
        add("acceptance_verdict", "the MEASURED §2 verdict grounds which gates actually fail and by how much")
        decomposed.append("map claims to taili_spec coverage before saying a behavior is acceptable")

    if asks_diag:
        add("diagnostic_history", "behavior questions require knowing which diagnostic job/checkpoint/test was run")
        add("diagnostic_reports", "actual robot behavior should come from completed diagnostic reports")
        add("definitions", "diagnostic metrics need formula/source grounding", required=False)
        decomposed.append("use diagnostic reports for behavior; telemetry alone is only a proxy")
        gap_checks.append({
            "source": "diagnostic_reports",
            "blocking": True,
            "gap": "no relevant completed diagnostic report",
            "impact": "cannot accurately describe how the robot moved; can only discuss proxies",
        })

    if asks_comparison:
        add("training_telemetry", "comparison between monitor and diagnosis needs the monitor's current proxy metrics")
        add("diagnostic_history", "comparison needs the diagnostic job identity")
        add("diagnostic_reports", "comparison needs observed diagnostic behavior")
        add("definitions", "mismatch analysis needs metric definitions and limitations")
        decomposed.append("compare proxy metric definitions against diagnostic evidence, not the UI impression")

    if asks_tuning_root_cause:
        add("training_telemetry", "tuning/root-cause needs current optimization signal")
        add("diagnostic_history", "tuning/root-cause needs recent behavior tests when available", required=False)
        add("diagnostic_reports", "tuning/root-cause should check actual behavior before proposing changes", required=False)
        add("taili_context", "tuning/root-cause needs YAML and strategy constraints")
        add("definitions", "tuning/root-cause needs reward/gate formulas")
        add("signal_map", "tuning/root-cause must connect spec, YAML, code, telemetry, and diagnostics")
        add("code_knowledge", "tuning/root-cause needs implementation evidence for consumed fields and calculations", required=False)
        add("tuning_ledger", "tuning/root-cause should know recent changes/proposals and avoid repeating failed hypotheses", required=False)
        decomposed.append("only propose small reversible changes after current metrics, behavior, spec, and config are aligned")

    if asks_code:
        add("code_knowledge", "question asks for implementation, consumer, wiring, code calculation, or dead-key evidence")
        add("signal_map", "code questions need mapping from user terms to YAML keys, telemetry fields, diagnostics, and spec")
        decomposed.append("answer code-level claims from allowlisted snippets and mapping, not from YAML alone")
        gap_checks.append({
            "source": "code_knowledge",
            "blocking": True,
            "gap": "code knowledge unavailable or no matched snippets",
            "impact": "cannot claim a field is implemented/consumed/dead without code evidence",
        })

    if asks_ledger:
        add("tuning_ledger", "question refers to previous changes, experiment memory, or tuning loop history")
        decomposed.append("use ledger for memory, then verify outcomes with current telemetry/diagnostics")

    if asks_asset:
        add("asset", "question mentions robot geometry/actuator/joint/mass/URDF-level facts")
        add("taili_context", "asset facts need to be compared with the local strategy contract", required=False)

    if "当前值" in q or "deployed" in q or "env_cfg" in q or "远程配置" in q:
        add("deployed_config", "question asks for current deployed config rather than only local YAML", required=False)

    if "最近" in q or "对比" in q or "哪个run" in q or "哪个 run" in q or "回退" in q:
        add("run_ledger", "question compares recent runs or asks about rollback candidates", required=False)

    if asks_system:
        add("workspace_config", "question concerns console/LLM/UI/framework/config-set behavior")
        add("frameworks", "framework registry can explain active/saved framework and diagnostic task wiring", required=False)
        decomposed.append("separate console/LLM plumbing from training-policy facts")

    if asks_definition and "definitions" not in selected:
        add("definitions", "question asks what a field means or how it is computed")

    if not selected:
        add("workspace_config", "fallback: establish console boundary and avoid guessing from no evidence", required=False)
        decomposed.append("question does not clearly name a live/source-specific fact; answer only within available context")

    answer_rules = [
        "Use selected evidence sources only; do not scan arbitrary local or remote paths.",
        "Answer naturally; this is not a user-facing template.",
        "Every concrete number/path/run/checkpoint/state must come from collected evidence.",
        "If a selected source is missing or stale and it blocks the conclusion, state that gap instead of guessing.",
        "Treat telemetry/reward/gates as training proxies; use diagnostics for observed robot behavior.",
        "For changes/actions, propose only; execution must go through the console confirmation gate.",
    ]

    return {
        "query": query,
        "ui_mode": ui_mode,
        "profile": {
            "asks_live_training": asks_live,
            "asks_remote_status": asks_remote,
            "asks_provenance": asks_provenance,
            "asks_strategy_or_yaml": asks_strategy,
            "asks_spec": asks_spec,
            "asks_diagnostics": asks_diag,
            "asks_definition": asks_definition,
            "asks_system": asks_system,
            "asks_tuning_or_action": asks_tuning_root_cause,
        },
        "selected_sources": list(selected.values()),
        "decomposed_questions": decomposed,
        "gap_checks": gap_checks,
        "answer_rules": answer_rules,
        "permission_boundary": (
            "Read-only allowlist routing. No arbitrary filesystem scan, no arbitrary remote shell, "
            "no writes, no training/diagnostic execution."
        ),
    }


def selected_source_ids(route: dict[str, Any]) -> list[str]:
    return [str(item.get("id")) for item in route.get("selected_sources", []) if isinstance(item, dict)]


def _normalize(value: str) -> str:
    return (value or "").strip().lower()


def _has(text: str, terms: tuple[str, ...]) -> bool:
    return any(term.lower() in text for term in terms)


def _extract_ui_mode(query: str) -> str:
    match = re.search(r"ui\s*mode\s*:\s*([A-Za-z0-9_-]+)", query or "", flags=re.IGNORECASE)
    return match.group(1).lower() if match else ""


def _job_ids(query: str) -> list[str]:
    return re.findall(r"\b20\d{6}_\d{6}\b", query or "")
