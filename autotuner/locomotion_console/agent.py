"""The soul in the live loop: an agentic chat that OPERATES the locomotion console's own tools.

The user talks in natural language. The LLM reads the *real* box through read-only tools
and answers/explains in plain language - it never invents numbers, it cites tool output.

Built on the single-shot `call_llm_with_schema`: each turn the LLM is given the user message
plus the accumulated tool results, and returns JSON that is EITHER a tool call
(`{"tool": name, "args": {...}}`) or a final answer (`{"reply": "..."}`). Bounded steps.

Tools here are READ-ONLY (status / list runs / reward curve). Action tools (physeval, kill,
resume) are a deliberate next step with a confirmation guard - the soul should not actuate
training silently.
"""
from __future__ import annotations

import json
import re
import shlex
from typing import Any, Dict, List

# LLM-supplied identifiers that reach a remote shell must match this before use. repr()/f-string
# interpolation is NOT shell-safe (a value with a single quote makes repr emit a double-quoted
# string in which $()/backticks still expand → remote RCE). Everything shell-bound is shlex.quote'd
# AND allowlisted.
_RUN_TOKEN_RE = re.compile(r"^[A-Za-z0-9._/\-]+$")

from .config import LocomotionConsoleSettings
from .config_manager import desired_framework_id, llm_profile
from .config_set import get_active_config_set
from .datasource import RealDataSource
from .diagnostic_history import DiagnosticHistoryStore
from .framework_profile import list_framework_profiles


def _profile_summary(profile) -> Dict[str, Any]:
    return {
        "id": profile.id,
        "label": profile.label,
        "status": profile.status,
        "detail": profile.detail,
    }


# -- tools: each takes the RealDataSource, returns a JSON-serializable dict ----

def _tool_get_status(src: RealDataSource) -> Dict[str, Any]:
    r = src._get_remote()
    run = src._newest_run(r)
    it = src._latest_ckpt_iter(r, run) if run else 0
    return {
        "host": src.profile.hostname,
        "newest_run": run.split("/")[-1] if run else None,
        "latest_checkpoint_iter": it,
        "training_running": src._is_running(r),
    }


def _tool_list_runs(src: RealDataSource, limit: int = 8) -> Dict[str, Any]:
    r = src._get_remote()
    out = r.exec_out(f"ls -dt {src._run_glob_shell()} 2>/dev/null | head -{int(limit)}") or ""
    runs: List[Dict[str, Any]] = []
    for line in out.splitlines():
        run = line.strip().rstrip("/")
        if not run:
            continue
        runs.append({"run": run.split("/")[-1],
                     "max_checkpoint_iter": src._latest_ckpt_iter(r, run)})
    return {"runs": runs}


def _tool_get_reward_curve(src: RealDataSource, run: str = "") -> Dict[str, Any]:
    r = src._get_remote()
    if run:
        if not _RUN_TOKEN_RE.match(run):
            return {"error": f"invalid run name: {run!r}"}
        path = (r.exec_out(
            f"ls -dt {src._run_glob_shell()} 2>/dev/null | grep -- {shlex.quote(run)} | head -1"
        ) or "").strip().rstrip("/")
    else:
        path = src._newest_run(r)
    if not path:
        return {"error": "run not found"}
    pts, tags = src._fetch_reward_curve(r, path)
    name = path.split("/")[-1]
    if not pts:
        return {"run": name, "n_points": 0, "note": "no reward points logged",
                "available_tags": tags[:20]}
    vals = [v for _, v in pts]
    drop = None
    for i in range(1, len(pts)):
        d = pts[i][1] - pts[i - 1][1]
        if drop is None or d < drop[2]:
            drop = (pts[i - 1][0], pts[i][0], d)
    return {
        "run": name,
        "n_points": len(pts),
        "first": pts[0],
        "last": pts[-1],
        "max": round(max(vals), 1),
        "min": round(min(vals), 1),
        "biggest_drop": ({"from_iter": drop[0], "to_iter": drop[1],
                          "delta": round(drop[2], 1)} if drop else None),
        "curve": [(s, round(v, 1)) for s, v in pts],
    }


def _tool_get_asset(src: RealDataSource) -> Dict[str, Any]:
    """Read-only view of the robot ASSET cfg - actuator stiffness/damping/effort/velocity
    limits + default pose. The #1 root-cause lesson (stiffness too low for the robot's mass)
    lives HERE, not in env_cfg. locomotion console never writes it."""
    r = src._get_remote()
    p = src.profile.asset_cfg_path
    if not p:
        return {"error": "asset_cfg_path not discovered - run discovery"}
    out = r.exec_out(
        f"grep -nE 'stiffness|damping|effort_limit|velocity_limit|mass|joint_pos|"
        f"ImplicitActuator|DCMotor|usd_path|urdf' {p} 2>/dev/null | head -50") or ""
    return {"asset_cfg_path": p, "matched_lines": out[-3200:] if out else "(no matches)"}


def _tool_search_knowledge(src: RealDataSource, query: str = "") -> Dict[str, Any]:
    """Retrieve allowlisted project docs. No query = table of contents; query = matching sections."""
    from . import knowledge
    return knowledge.search(query)


def _tool_get_taili_context_pack(src: RealDataSource, query: str = "") -> Dict[str, Any]:
    """Read-only Taili strategy/spec/YAML context pack from a local allowlist.

    Use this before strategy/config/tuning answers. It gives the LLM the same
    project-specific background the operator expects: spec, strategy decisions,
    architecture notes, and current YAML contract.
    """
    from . import knowledge

    return knowledge.build_taili_context_pack(query=query, include_docs=True)


def _tool_get_campaign_journal(src: RealDataSource, robot: str = "taili", limit: int = 30) -> Dict[str, Any]:
    """The campaign's durable decision memory: best-so-far score, the (gate::lever) pairs already tried and
    ROLLED BACK (the NO-REPEAT guard must reject these), and recent iterations. Read this BEFORE proposing a
    lever (step 3 of run_tuning_loop) so you never repeat a failed experiment or claim a stale best."""
    from .campaign_journal import read_journal
    return read_journal(robot=robot, limit=limit)


def _tool_record_campaign_iteration(src: RealDataSource, robot: str = "taili", target_gate: str = "",
                                    lever: str = "", decision: str = "", score_before: str = "",
                                    score_after: str = "", config_diff: str = "", result_run: str = "",
                                    evidence: str = "", note: str = "") -> Dict[str, Any]:
    """Append this iteration's outcome to the campaign journal (decision in {kept, rolled_back, pending}).
    Do this at step 7 (decide) so the NO-REPEAT/DECIDE guards have the memory next iteration."""
    from .campaign_journal import record_iteration
    try:
        import datetime as _dt
        ts = _dt.datetime.now().timestamp()
    except Exception:
        ts = None
    entry = record_iteration(robot, target_gate=target_gate, lever=lever, decision=decision,
                             score_before=score_before, score_after=score_after, config_diff=config_diff,
                             result_run=result_run, evidence=evidence, note=note, ts=ts)
    return {"recorded": True, "entry": entry}


def _tool_get_playbook(src: RealDataSource, task: str = "", gate: str = "", robot: str = "") -> Dict[str, Any]:
    """SYSTEMATIZED operator knowledge, retrieved by task/gate. Returns the ROBOT-AGNOSTIC workflow (guarded
    state machine: establish→verify→select→apply→train→remeasure→decide) + grounding rules, plus the active
    robot PROFILE (gate→lever map with cautions, ops signatures, regime constants). Use this to DRIVE work as
    the system owner — obey each step's guard, never advance on a guess. `robot` selects the profile (a new
    robot = a new profile, same workflow)."""
    from .playbook import get_playbook
    return get_playbook(task=task, gate=gate, robot=robot)


def _tool_get_spec_coverage(src: RealDataSource) -> Dict[str, Any]:
    """Read-only taili_spec acceptance ledger: spec rows vs training/evaluation coverage."""
    from .spec_coverage import build_spec_coverage_report

    return build_spec_coverage_report().model_dump()


def _compact_acceptance_for_llm(verdict: Dict[str, Any]) -> Dict[str, Any]:
    """Shape the measured taili_spec §2 verdict for grounding: per-family present/ok plus the exact
    failing sub-gates with their measured stat-vs-band, so the copilot cites 'A1[fwd05] med=0.14>0.10'
    instead of narrating the hand-authored spec_coverage status."""
    if not isinstance(verdict, dict) or not verdict.get("available"):
        reason = (verdict or {}).get("reason", "no measured acceptance verdict")
        return {"available": False, "reason": reason,
                "instruction": ("No physeval has been scored for the current run yet. Say so plainly "
                                "and offer to run an acceptance measurement (run_acceptance).")}
    fams = verdict.get("families") or {}
    gates = verdict.get("gates") or {}
    families = {f: {"present": bool(s.get("present")), "ok": bool(s.get("ok"))}
                for f, s in fams.items() if isinstance(s, dict)}
    failing_gates = {k: v.get("detail", "") for k, v in gates.items()
                     if isinstance(v, dict) and not v.get("ok")}
    return {
        "available": True,
        "run": verdict.get("run"),
        "passed": verdict.get("passed"),
        "n_present": verdict.get("n_present"),
        "n_hard": verdict.get("n_hard"),
        "missing_families": verdict.get("missing"),
        "failed_families": verdict.get("failed"),
        "families": families,
        "failing_gates": failing_gates,
        "scorecards_read": verdict.get("scorecards_read"),
        "instruction": ("This is the MEASURED metric the product is graded on (taili_spec §2). Ground "
                        "every tuning hypothesis in these gates: cite the failing gate and its stat vs "
                        "band. missing_families were never evaluated (e.g. run terrain/push physeval)."),
    }


def _tool_get_operations_state(src: RealDataSource) -> Dict[str, Any]:
    """SYSTEM SELF-AWARENESS — use FIRST for any '卡住了吗/什么状态/为什么不动' question. Returns what
    the SYSTEM ITSELF is doing (auto_drive self-healing, campaigns, measurements), step freshness,
    and the ops-runbook stall classifier's diagnosis (known GPU-kernel-stall pattern etc.)."""
    from .ops_state import build_operations_state

    return build_operations_state(src)


def _tool_get_strategy(src: RealDataSource) -> Dict[str, Any]:
    """Read-only FULL tuning strategy: all reward weights (grouped tracking/gait-quality/thresholds),
    curriculum phases + advancement gates, AMP hyperparameters — the exact contract apply_tuning edits."""
    from .strategy_view import build_strategy_view

    return build_strategy_view()


def _tool_get_tuning_state(src: RealDataSource) -> Dict[str, Any]:
    """Read-only tuning-loop state: the rollback stack (recent apply_tuning entries, undoable),
    the BEST_CHECKPOINT registry (highest measured score + its checkpoint), and whether training
    is currently running. Check before proposing tune/train actions."""
    out: Dict[str, Any] = {}
    try:
        from autotuner.training.strategy_edit import rollback_stack
        out["rollback_stack"] = rollback_stack()[-5:]
    except Exception as e:  # noqa: BLE001
        out["rollback_stack_error"] = str(e)
    try:
        import json as _json
        remote = src._get_remote()
        raw = remote.exec_out("cat /root/gpufree-data/taili_runs/BEST_CHECKPOINT.json 2>/dev/null")
        out["best_checkpoint"] = _json.loads(raw) if raw.strip() else None
        out["training_running"] = bool(src._is_running(remote))
    except Exception as e:  # noqa: BLE001
        out["remote_error"] = str(e)
    return out


def _tool_analyze_acceptance(src: RealDataSource) -> Dict[str, Any]:
    """The copilot's TUNING BRAIN: rank the measured failing gates by margin, map each to its reward
    LEVER (the same empirically-guarded heuristic the autonomous campaign uses — metric-artifact
    gates skipped, unstable levers capped), and propose the next concrete apply_tuning change.
    Use this to DRIVE the loop yourself: run_acceptance → analyze_acceptance → propose apply_tuning
    → deploy_payload + resume_training → run_acceptance again."""
    from autotuner.blind_locomotion.taili_blind_config import get_config_value, load_taili_blind_config
    from autotuner.training.tune_orchestrator import GATE_LEVERS, SKIP_FAMILIES, analyze_gaps, propose_change

    verdict = src.get_acceptance()
    if not verdict.get("available"):
        return {"available": False, "reason": verdict.get("reason", "no measured verdict"),
                "instruction": "Run an acceptance measurement first (run_acceptance), then analyze."}
    gaps = analyze_gaps(verdict)
    cfg = load_taili_blind_config()
    current = {}
    for levers in GATE_LEVERS.values():
        for key, _step, _cap in levers:
            val = get_config_value(cfg, f"reward.{key}")
            if val is not None:
                current[key] = float(val)
    proposal = propose_change(gaps, current, [])
    return {
        "available": True,
        "run": verdict.get("run"),
        "score": sum(1 for v in (verdict.get("gates") or {}).values() if v.get("ok")),
        "ranked_gaps": [
            {"gate": g.gate, "family": g.family, "measured": g.measured, "threshold": g.threshold,
             "margin": g.margin, "levers": [{"key": k, "step": s, "cap": cap, "current": current.get(k)}
                                            for k, s, cap in GATE_LEVERS.get(g.family, [])]}
            for g in gaps[:8]
        ],
        "skipped_metric_artifact_families": sorted(SKIP_FAMILIES),
        "next_proposal": proposal or None,
        "instruction": ("ranked_gaps = failing gates worst-first with their reward levers (step size and "
                        "HARD cap — never exceed caps, they encode training-stability limits). "
                        "next_proposal is the heuristic's suggested apply_tuning change; you may refine it "
                        "with your own analysis, then PROPOSE apply_tuning + deploy_payload + "
                        "resume_training so the operator can confirm the full tune-train cycle."),
    }


def _tool_get_acceptance(src: RealDataSource) -> Dict[str, Any]:
    """Read-only MEASURED taili_spec §2 acceptance verdict for the newest run, scored from persisted
    physeval logs. Never launches physeval (separate isolated path). This is the single source of
    truth for 'how far from benchmark' — prefer it over spec_coverage status strings for any tuning,
    root-cause, or 'which gates fail' question. Returns available:False when no physeval scored yet."""
    return _compact_acceptance_for_llm(src.get_acceptance())


def _tool_get_code_knowledge(src: RealDataSource, query: str = "") -> Dict[str, Any]:
    """Read-only allowlisted Taili implementation evidence.

    Use for questions like: where is this metric calculated, is this YAML key consumed,
    does telemetry really reflect the reward/diagnostic mechanism, or is this a dead key.
    """
    from .code_knowledge import search_code_knowledge

    return search_code_knowledge(query=query)


def _tool_get_signal_map(src: RealDataSource, query: str = "") -> Dict[str, Any]:
    """Read-only spec/YAML/code/telemetry/diagnostic mapping."""
    from .code_knowledge import build_signal_map

    return build_signal_map(query=query)


def _tool_get_tuning_ledger(src: RealDataSource, limit: int = 24) -> Dict[str, Any]:
    """Read-only tuning memory: recent LLM turns/proposals/runs/diagnostic job ids."""
    from .tuning_ledger import build_tuning_ledger

    return build_tuning_ledger(src.settings, limit=limit)


def _tool_get_config(src: RealDataSource, grep: str = "") -> Dict[str, Any]:
    """Read-only view of the deployed env config - reward weights / key params.
    The locomotion console NEVER writes config; this only greps the file for relevant lines."""
    r = src._get_remote()
    p = src.profile.env_cfg_path
    if not p:
        return {"error": "env_cfg_path not discovered - run discovery"}
    pat = grep or ("rew_|weight|clearance|stiffness|damping|stand_h|base_h|"
                   "torque|action_scale|action_rate|_range|push|dr_")
    # pat is an LLM-supplied regex; shlex.quote both it and the path (repr() is not shell-safe).
    out = r.exec_out(f"grep -nE {shlex.quote(pat)} {shlex.quote(p)} 2>/dev/null | head -70") or ""
    return {"env_cfg_path": p, "matched_lines": out[-3200:] if out else "(no matches)"}


def _tool_get_run_ledger(src: RealDataSource, limit: int = 8) -> Dict[str, Any]:
    """Ledger of recent runs: max checkpoint iter + reward peak/final from each run's tfevents.
    Read-only - for comparing runs / choosing a version to roll back to."""
    r = src._get_remote()
    out = r.exec_out(f"ls -dt {src._run_glob_shell()} 2>/dev/null | head -{int(limit)}") or ""
    rows = []
    for line in out.splitlines():
        run = line.strip().rstrip("/")
        if not run:
            continue
        it = src._latest_ckpt_iter(r, run)
        try:
            pts, _ = src._fetch_reward_curve(r, run)
        except Exception:  # noqa: BLE001
            pts = []
        vals = [v for _, v in pts]
        rows.append({
            "run": run.split("/")[-1],
            "max_checkpoint_iter": it,
            "reward_peak": round(max(vals), 1) if vals else None,
            "reward_final": round(pts[-1][1], 1) if pts else None,
            "n_points": len(pts),
        })
    return {"runs": rows}


def _tool_get_train_log(src: RealDataSource, lines: int = 40) -> Dict[str, Any]:
    r = src._get_remote()
    run = src._newest_run(r)
    paths = src._resolve_telemetry_paths(r, run)
    import shlex

    train_tail = r.exec_out(f"tail -n {int(lines)} {shlex.quote(paths.log_path)} 2>/dev/null") if paths.log_path else ""
    console_tail = (
        r.exec_out(f"tail -n {int(lines)} {shlex.quote(paths.console_log_path)} 2>/dev/null")
        if paths.console_log_path and paths.console_log_path != paths.log_path
        else ""
    )
    return {
        "run": run,
        "log_path": paths.log_path,
        "telemetry_path": paths.telemetry_path,
        "console_log_path": paths.console_log_path,
        "checkpoint_dir": paths.checkpoint_dir,
        "path_evidence": list(paths.evidence),
        "running": src._is_running(r),
        "tail": train_tail[-2800:] if train_tail else "(train log empty or not found)",
        "console_tail": console_tail[-2800:] if console_tail else "",
        "note": "train.log is semantic [TP*] telemetry; console.log is complete stdout/stderr from the launcher when available.",
    }


def _tool_get_training_telemetry(src: RealDataSource) -> Dict[str, Any]:
    """Read-only structured training telemetry.

    Preferred source is the run-specific telemetry JSONL; fallback is [TPREW]/[TPSTAT]
    parsing. This is the first tool for "what is happening right now?" answers.
    """
    import asyncio

    telemetry = asyncio.run(src.training_telemetry())
    return telemetry.model_dump()


def _tool_get_operator_context(
    src: RealDataSource,
    query: str = "",
    include_diagnostics: bool = True,
) -> Dict[str, Any]:
    """One-shot read-only context for high-quality answers.

    This bundles the facts the operator expects the LLM to know without giving
    it arbitrary filesystem or shell access: controlled Taili docs/YAML,
    current telemetry, remote machine status, workspace config, definitions,
    and recent diagnostic history.
    """
    import asyncio

    from . import knowledge
    from .definitions import list_definitions
    from .diagnostics import DiagnosticsController

    async def _load_live():
        return await asyncio.gather(
            src.training_telemetry(),
            src.remote_machine_status(),
            return_exceptions=True,
        )

    telemetry, remote_status = asyncio.run(_load_live())

    def _dump(value: Any) -> Dict[str, Any]:
        if isinstance(value, Exception):
            return {"available": False, "error": f"{type(value).__name__}: {value}"}
        if hasattr(value, "model_dump"):
            return value.model_dump()
        return {"available": False, "error": f"unexpected value: {type(value).__name__}"}

    context: Dict[str, Any] = {
        "permission_boundary": (
            "Read-only bundle. Sources: controlled local Taili docs/YAML allowlist, "
            "current telemetry API, remote status API, config-set registry, definition registry, "
            "and local diagnostic history. No arbitrary local scan, no arbitrary remote path scan, no execution."
        ),
        "query": query,
        "taili_context_pack": knowledge.build_taili_context_pack(query=query, include_docs=True),
        "spec_coverage": _tool_get_spec_coverage(src),
        "training_telemetry": _dump(telemetry),
        "remote_status": _dump(remote_status),
        "workspace_config": _tool_get_workspace_config(src),
        "definitions": list_definitions(query or "reward gait progress_gate lin_err terrain").model_dump(),
    }
    if include_diagnostics:
        history = DiagnosticsController(src.settings, src).history.list(limit=5)
        context["diagnostic_history"] = history.model_dump()
    return context


def _tool_get_evidence_context(
    src: RealDataSource,
    query: str = "",
    ui_mode: str = "",
    include_optional: bool = True,
) -> Dict[str, Any]:
    """Route the operator question to allowlisted evidence, then collect a compact context.

    This is the preferred LLM entry point for broad/current/tuning/diagnostic answers.  It
    deliberately separates source routing from natural-language narration: the LLM gets the
    evidence and gaps, but no rigid report template.
    """
    import asyncio
    import re

    from . import knowledge
    from .definitions import definitions_for_telemetry, list_definitions
    from .evidence_router import route_evidence, selected_source_ids
    from .spec_coverage import build_spec_coverage_report

    operator_query = _operator_question_from_message(query)
    route = route_evidence(query=operator_query, ui_mode=ui_mode)
    source_ids = set(selected_source_ids(route))
    required = {
        str(item.get("id")): bool(item.get("required", True))
        for item in route.get("selected_sources", [])
        if isinstance(item, dict)
    }
    if not include_optional:
        source_ids = {sid for sid in source_ids if required.get(sid, True)}

    evidence: Dict[str, Any] = {}
    gaps: List[Dict[str, Any]] = []

    def _status_block(status: str, data: Any = None, *, summary: str = "", error: str = "") -> Dict[str, Any]:
        out: Dict[str, Any] = {"status": status}
        if summary:
            out["summary"] = summary
        if error:
            out["error"] = error
        if data is not None:
            out["data"] = data
        return out

    def _dump_model(value: Any) -> Dict[str, Any]:
        if isinstance(value, Exception):
            return {"available": False, "error": f"{type(value).__name__}: {value}"}
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, dict):
            return value
        return {"available": False, "error": f"unexpected value: {type(value).__name__}"}

    def _gap(source: str, message: str, *, blocking: bool | None = None, impact: str = "") -> None:
        gaps.append({
            "source": source,
            "blocking": required.get(source, True) if blocking is None else blocking,
            "message": message,
            "impact": impact,
        })

    async def _load_live() -> Dict[str, Any]:
        tasks: Dict[str, Any] = {}
        if "training_telemetry" in source_ids:
            tasks["training_telemetry"] = asyncio.create_task(src.training_telemetry())
        if "remote_status" in source_ids:
            tasks["remote_status"] = asyncio.create_task(src.remote_machine_status())
        out: Dict[str, Any] = {}
        for key, task in tasks.items():
            try:
                out[key] = await task
            except Exception as exc:  # noqa: BLE001
                out[key] = exc
        return out

    live: Dict[str, Any] = {}
    telemetry_model: Any = None
    if {"training_telemetry", "remote_status"} & source_ids:
        live = asyncio.run(_load_live())

    if "training_telemetry" in source_ids:
        telemetry_model = live.get("training_telemetry")
        telemetry = _dump_model(telemetry_model)
        compact = _compact_training_telemetry_for_llm(telemetry)
        evidence["training_telemetry"] = _status_block(
            "ok" if compact.get("available") else "missing",
            compact,
            summary=str(compact.get("summary") or compact.get("snapshot", {}).get("conclusion") or ""),
            error=str(compact.get("error") or ""),
        )
        if not compact.get("available"):
            _gap("training_telemetry", str(compact.get("error") or "telemetry unavailable"),
                 impact="current training state, reward, phase, and speed cannot be claimed")
        elif compact.get("stale"):
            _gap("training_telemetry", f"telemetry is stale; age_s={compact.get('telemetry_age_s')}",
                 impact="current-state claims must be phrased as last-known facts")

    if "remote_status" in source_ids:
        remote = _dump_model(live.get("remote_status"))
        compact = _compact_remote_status_for_llm(remote)
        evidence["remote_status"] = _status_block(
            "ok" if compact.get("available") else "missing",
            compact,
            summary=f"host={compact.get('host')}; gpus={len(compact.get('gpus') or [])}",
            error=str(compact.get("error") or ""),
        )
        if not compact.get("available"):
            _gap("remote_status", str(compact.get("error") or "remote status unavailable"),
                 impact="GPU/SSH/tmux/resource conclusions are not supported")

    if "telemetry_provenance" in source_ids:
        try:
            probe = _tool_probe_telemetry(src)
            evidence["telemetry_provenance"] = _status_block(
                "ok",
                probe,
                summary=f"run={probe.get('run')}; checks={len(probe.get('checks') or [])}",
            )
            checks = " ".join(str(item) for item in probe.get("checks") or [])
            if "missing" in checks:
                _gap("telemetry_provenance", f"some artifacts are missing: {checks[:400]}",
                     impact="monitor/log/checkpoint source may be incomplete")
        except Exception as exc:  # noqa: BLE001
            evidence["telemetry_provenance"] = _status_block("error", error=f"{type(exc).__name__}: {exc}")
            _gap("telemetry_provenance", f"{type(exc).__name__}: {exc}",
                 impact="cannot prove current log/jsonl/checkpoint wiring")

    if "train_log" in source_ids:
        try:
            log = _tool_get_train_log(src, lines=60)
            evidence["train_log"] = _status_block(
                "ok",
                {
                    "run": log.get("run"),
                    "log_path": log.get("log_path"),
                    "telemetry_path": log.get("telemetry_path"),
                    "console_log_path": log.get("console_log_path"),
                    "checkpoint_dir": log.get("checkpoint_dir"),
                    "running": log.get("running"),
                    "tail": str(log.get("tail") or "")[-1800:],
                    "console_tail": str(log.get("console_tail") or "")[-1800:],
                    "note": log.get("note"),
                },
                summary=f"run={log.get('run')}; running={log.get('running')}",
            )
        except Exception as exc:  # noqa: BLE001
            evidence["train_log"] = _status_block("error", error=f"{type(exc).__name__}: {exc}")
            _gap("train_log", f"{type(exc).__name__}: {exc}",
                 impact="cannot cite recent log tail")

    if "taili_context" in source_ids:
        pack = knowledge.build_taili_context_pack(query=operator_query, include_docs=True)
        compact = _compact_taili_context_pack_for_llm(pack)
        evidence["taili_context"] = _status_block(
            "ok" if compact.get("yaml", {}).get("available", True) else "missing",
            compact,
            summary="Taili YAML/spec/strategy allowlist loaded",
        )
        if compact.get("yaml", {}).get("available") is False:
            _gap("taili_context", str(compact.get("yaml", {}).get("error") or "Taili YAML unavailable"),
                 impact="cannot make project-specific YAML/config claims")

    if "spec_coverage" in source_ids:
        report = build_spec_coverage_report().model_dump()
        evidence["spec_coverage"] = _status_block(
            "ok",
            _compact_spec_coverage_for_llm(report),
            summary=str(report.get("summary") or ""),
        )

    if "acceptance_verdict" in source_ids:
        verdict = _tool_get_acceptance(src)
        if verdict.get("available"):
            n_fail = len(verdict.get("failing_gates") or {})
            summary = (f"measured §2: passed={verdict.get('passed')} "
                       f"present={verdict.get('n_present')}/{verdict.get('n_hard')} "
                       f"failing_gates={n_fail}")
            evidence["acceptance_verdict"] = _status_block("ok", verdict, summary=summary)
        else:
            evidence["acceptance_verdict"] = _status_block(
                "gap", verdict, summary=f"no measured verdict yet: {verdict.get('reason', '')}")

    if "signal_map" in source_ids:
        try:
            signal_map = _tool_get_signal_map(src, query=operator_query)
            evidence["signal_map"] = _status_block(
                "ok",
                signal_map,
                summary=f"{len(signal_map.get('rows') or [])} mapped signal row(s)",
            )
            if not signal_map.get("rows"):
                _gap("signal_map", "no spec/YAML/code/telemetry mapping row matched the query",
                     impact="source-layer coverage may be incomplete")
        except Exception as exc:  # noqa: BLE001
            evidence["signal_map"] = _status_block("error", error=f"{type(exc).__name__}: {exc}")
            _gap("signal_map", f"{type(exc).__name__}: {exc}",
                 impact="cannot map spec/YAML/code/telemetry/diagnostic layers")

    if "code_knowledge" in source_ids:
        try:
            code = _tool_get_code_knowledge(src, query=operator_query)
            evidence["code_knowledge"] = _status_block(
                "ok" if code.get("snippets") or code.get("consumption_checks") else "missing",
                _compact_code_knowledge_for_llm(code),
                summary=f"{len(code.get('snippets') or [])} snippet(s), {len(code.get('consumption_checks') or [])} consumption check(s)",
            )
            for gap in code.get("gaps") or []:
                if isinstance(gap, dict):
                    _gap("code_knowledge", str(gap.get("gap") or "code knowledge gap"),
                         impact=str(gap.get("impact") or "implementation claim may be unsupported"))
        except Exception as exc:  # noqa: BLE001
            evidence["code_knowledge"] = _status_block("error", error=f"{type(exc).__name__}: {exc}")
            _gap("code_knowledge", f"{type(exc).__name__}: {exc}",
                 impact="cannot ground implementation or code-consumption claims")

    if "tuning_ledger" in source_ids:
        try:
            ledger = _tool_get_tuning_ledger(src, limit=24)
            evidence["tuning_ledger"] = _status_block(
                "ok",
                _compact_tuning_ledger_for_llm(ledger),
                summary=(
                    f"{ledger.get('summary', {}).get('turn_count', 0)} turns; "
                    f"{ledger.get('summary', {}).get('proposal_count', 0)} proposals"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            evidence["tuning_ledger"] = _status_block("error", error=f"{type(exc).__name__}: {exc}")
            _gap("tuning_ledger", f"{type(exc).__name__}: {exc}",
                 impact="cannot use recent tuning memory; verify from current evidence")

    if "definitions" in source_ids:
        try:
            if telemetry_model is not None and not isinstance(telemetry_model, Exception):
                items = [item.model_dump() for item in definitions_for_telemetry(telemetry_model)[:24]]
                definitions = {"query": query, "items": items}
            else:
                definitions = list_definitions(operator_query or "reward gait progress_gate lin_err terrain diagnostic").model_dump()
            evidence["definitions"] = _status_block(
                "ok",
                _compact_definitions_for_llm(definitions),
                summary=f"{len(definitions.get('items') or [])} definitions",
            )
        except Exception as exc:  # noqa: BLE001
            evidence["definitions"] = _status_block("error", error=f"{type(exc).__name__}: {exc}")
            _gap("definitions", f"{type(exc).__name__}: {exc}",
                 impact="metric meanings/formulas may be ambiguous")

    history_payload: Dict[str, Any] | None = None
    if {"diagnostic_history", "diagnostic_reports"} & source_ids:
        try:
            history_payload = _tool_list_diagnostic_history(src, limit=8)
            if "diagnostic_history" in source_ids:
                evidence["diagnostic_history"] = _status_block(
                    "ok" if history_payload.get("items") else "missing",
                    history_payload,
                    summary=f"{len(history_payload.get('items') or [])} recent diagnostic jobs",
                )
            if not history_payload.get("items"):
                _gap("diagnostic_history", "no recent diagnostic history",
                     impact="cannot choose a completed report unless a job_id is supplied")
        except Exception as exc:  # noqa: BLE001
            history_payload = None
            if "diagnostic_history" in source_ids:
                evidence["diagnostic_history"] = _status_block("error", error=f"{type(exc).__name__}: {exc}")
            _gap("diagnostic_history", f"{type(exc).__name__}: {exc}",
                 impact="cannot select diagnostic jobs")

    if "diagnostic_reports" in source_ids:
        reports: List[Dict[str, Any]] = []
        report_errors: List[Dict[str, str]] = []
        report_ids = _select_diagnostic_report_ids(operator_query, history_payload)
        if not report_ids:
            report_ids = [""]
        for job_id in report_ids[:3]:
            try:
                reports.append(_tool_get_diagnostic_report(src, job_id=job_id))
            except Exception as exc:  # noqa: BLE001
                report_errors.append({"job_id": job_id or "<latest>", "error": f"{type(exc).__name__}: {exc}"})
        evidence["diagnostic_reports"] = _status_block(
            "ok" if reports else "missing",
            {"reports": [_compact_diagnostic_report_for_llm(item) for item in reports], "errors": report_errors},
            summary=f"{len(reports)} report(s) loaded",
        )
        if not reports:
            _gap("diagnostic_reports", "no completed diagnostic report could be loaded",
                 impact="do not describe actual robot behavior; say diagnostics are missing")
        for err in report_errors:
            _gap("diagnostic_reports", f"{err['job_id']}: {err['error']}", blocking=False,
                 impact="one requested diagnostic report was unavailable")

    if "workspace_config" in source_ids:
        try:
            cfg = _tool_get_workspace_config(src)
            evidence["workspace_config"] = _status_block(
                "ok",
                cfg,
                summary=f"framework={cfg.get('runtime', {}).get('active_framework_id')}; llm={cfg.get('llm_status', {}).get('model')}",
            )
        except Exception as exc:  # noqa: BLE001
            evidence["workspace_config"] = _status_block("error", error=f"{type(exc).__name__}: {exc}")
            _gap("workspace_config", f"{type(exc).__name__}: {exc}",
                 impact="console/framework/LLM configuration claims are unsupported")

    if "frameworks" in source_ids:
        try:
            frameworks = _tool_list_frameworks(src)
            evidence["frameworks"] = _status_block(
                "ok",
                frameworks,
                summary=f"active={frameworks.get('active_framework_id')}; desired={frameworks.get('desired_framework_id')}",
            )
        except Exception as exc:  # noqa: BLE001
            evidence["frameworks"] = _status_block("error", error=f"{type(exc).__name__}: {exc}")

    if "deployed_config" in source_ids:
        try:
            grep = _grep_hint_for_query(operator_query)
            cfg = _tool_get_config(src, grep=grep)
            evidence["deployed_config"] = _status_block(
                "ok" if not cfg.get("error") else "missing",
                cfg,
                summary=f"env_cfg={cfg.get('env_cfg_path')}",
                error=str(cfg.get("error") or ""),
            )
            if cfg.get("error"):
                _gap("deployed_config", str(cfg.get("error")),
                     impact="cannot compare local YAML with deployed env_cfg")
        except Exception as exc:  # noqa: BLE001
            evidence["deployed_config"] = _status_block("error", error=f"{type(exc).__name__}: {exc}")

    if "asset" in source_ids:
        try:
            asset = _tool_get_asset(src)
            evidence["asset"] = _status_block(
                "ok" if not asset.get("error") else "missing",
                asset,
                summary=f"asset_cfg={asset.get('asset_cfg_path')}",
                error=str(asset.get("error") or ""),
            )
            if asset.get("error"):
                _gap("asset", str(asset.get("error")),
                     impact="cannot make actuator/URDF-level claims")
        except Exception as exc:  # noqa: BLE001
            evidence["asset"] = _status_block("error", error=f"{type(exc).__name__}: {exc}")

    if "run_ledger" in source_ids:
        try:
            ledger = _tool_get_run_ledger(src, limit=8)
            evidence["run_ledger"] = _status_block(
                "ok",
                ledger,
                summary=f"{len(ledger.get('runs') or [])} recent runs",
            )
        except Exception as exc:  # noqa: BLE001
            evidence["run_ledger"] = _status_block("error", error=f"{type(exc).__name__}: {exc}")

    for check in route.get("gap_checks", []):
        if not isinstance(check, dict):
            continue
        sid = str(check.get("source") or "")
        if sid and sid in source_ids and sid not in evidence:
            _gap(sid, str(check.get("gap") or "selected source was not collected"),
                 impact=str(check.get("impact") or "conclusion may be incomplete"))

    return {
        "kind": "evidence_context",
        "query": operator_query,
        "raw_query": query if query != operator_query else "",
        "ui_mode": ui_mode,
        "permission_boundary": route.get("permission_boundary"),
        "route": {
            "profile": route.get("profile"),
            "selected_sources": route.get("selected_sources"),
            "decomposed_questions": route.get("decomposed_questions"),
            "answer_rules": route.get("answer_rules"),
        },
        "evidence": evidence,
        "gaps": gaps,
        "instruction": (
            "Use this evidence to answer naturally. Do not use a fixed report template. "
            "Do not claim facts not present in evidence. If a blocking gap prevents a conclusion, "
            "say exactly what is missing. Telemetry/rewards/gates are training proxies; diagnostic "
            "reports are the source for observed robot behavior. Actions may only be proposed."
        ),
    }


def _compact_training_telemetry_for_llm(telemetry: Dict[str, Any]) -> Dict[str, Any]:
    latest = telemetry.get("latest") if isinstance(telemetry.get("latest"), dict) else {}
    snapshot = telemetry.get("snapshot") if isinstance(telemetry.get("snapshot"), dict) else {}
    return {
        "available": telemetry.get("available"),
        "running": telemetry.get("running"),
        "runtime_state": telemetry.get("runtime_state"),
        "stale": telemetry.get("stale"),
        "telemetry_age_s": telemetry.get("telemetry_age_s"),
        "run_id": telemetry.get("run_id"),
        "mode": telemetry.get("mode"),
        "summary": telemetry.get("summary"),
        "log_path": telemetry.get("log_path"),
        "telemetry_path": telemetry.get("telemetry_path"),
        "source_stats": telemetry.get("source_stats"),
        "latest": {
            "step": latest.get("step"),
            "total_steps": latest.get("total_steps"),
            "elapsed": latest.get("elapsed"),
            "eta": latest.get("eta"),
            "fps": latest.get("fps"),
            "checkpoint": latest.get("checkpoint"),
            "reward": _first_dict_items(latest.get("reward"), 72),
            "curriculum": _first_dict_items(latest.get("curriculum"), 72),
            "health": _first_dict_items(latest.get("health"), 72),
            "command": _first_dict_items(latest.get("command"), 72),
            "counters": _first_dict_items(latest.get("counters"), 32),
            "paths": _first_dict_items(latest.get("paths"), 24),
        },
        "snapshot": {
            "status": snapshot.get("status"),
            "conclusion": snapshot.get("conclusion"),
            "progress_pct": snapshot.get("progress_pct"),
            "phase": snapshot.get("phase"),
            "command_mode": snapshot.get("command_mode"),
            "active_dirs": snapshot.get("active_dirs"),
            "blocked_by": snapshot.get("blocked_by"),
            "next_gate": snapshot.get("next_gate"),
            "latest_checkpoint": snapshot.get("latest_checkpoint"),
            "blockers": [
                _compact_blocker(item)
                for item in (snapshot.get("blockers") or [])[:10]
                if isinstance(item, dict)
            ],
            "metric_groups": [
                {
                    "id": group.get("id"),
                    "title": group.get("title"),
                    "summary": group.get("summary"),
                    "items": [
                        _compact_metric_item(item)
                        for item in (group.get("items") or [])[:24]
                        if isinstance(item, dict)
                    ],
                }
                for group in (snapshot.get("metric_groups") or [])[:8]
                if isinstance(group, dict)
            ],
            "provenance": [
                {
                    "key": item.get("key"),
                    "label": item.get("label"),
                    "path": item.get("path"),
                    "role": item.get("role"),
                    "available": item.get("available"),
                    "evidence": item.get("evidence"),
                }
                for item in (snapshot.get("provenance") or [])[:16]
                if isinstance(item, dict)
            ],
            "notes": snapshot.get("notes"),
        },
        "definitions": [
            {
                "key": item.get("key"),
                "label": item.get("label"),
                "meaning": _truncate(item.get("meaning"), 220),
                "formula": _truncate(item.get("formula"), 220),
                "current_reading": item.get("current_reading"),
                "source": item.get("source"),
            }
            for item in (telemetry.get("definitions") or [])[:18]
            if isinstance(item, dict)
        ],
        "limitations": telemetry.get("limitations"),
        "error": telemetry.get("error"),
    }


def _compact_remote_status_for_llm(remote: Dict[str, Any]) -> Dict[str, Any]:
    gpus = []
    for gpu in remote.get("gpus") or []:
        if not isinstance(gpu, dict):
            continue
        gpus.append({
            "index": gpu.get("index"),
            "name": gpu.get("name"),
            "memory_used_mb": gpu.get("memory_used_mb"),
            "memory_total_mb": gpu.get("memory_total_mb"),
            "utilization_gpu_pct": gpu.get("utilization_gpu_pct"),
            "utilization_memory_pct": gpu.get("utilization_memory_pct"),
            "temperature_c": gpu.get("temperature_c"),
            "power_w": gpu.get("power_w"),
            "processes": [
                {
                    "pid": proc.get("pid"),
                    "name": proc.get("name"),
                    "used_memory_mb": proc.get("used_memory_mb"),
                }
                for proc in (gpu.get("processes") or [])[:6]
                if isinstance(proc, dict)
            ],
        })
    return {
        "available": remote.get("available"),
        "source": remote.get("source"),
        "generated_at": remote.get("generated_at"),
        "host": remote.get("host"),
        "uptime": remote.get("uptime"),
        "load_avg": remote.get("load_avg"),
        "cpu_count": remote.get("cpu_count"),
        "memory_total_mb": remote.get("memory_total_mb"),
        "memory_used_mb": remote.get("memory_used_mb"),
        "memory_available_mb": remote.get("memory_available_mb"),
        "swap_used_mb": remote.get("swap_used_mb"),
        "swap_total_mb": remote.get("swap_total_mb"),
        "gpus": gpus,
        "disks": remote.get("disks"),
        "tmux_sessions": remote.get("tmux_sessions"),
        "training_processes": [
            {
                "pid": proc.get("pid"),
                "cpu_pct": proc.get("cpu_pct"),
                "mem_pct": proc.get("mem_pct"),
                "etime": proc.get("etime"),
                "command": _truncate(proc.get("command"), 220),
            }
            for proc in (remote.get("training_processes") or [])[:8]
            if isinstance(proc, dict)
        ],
        "error": remote.get("error"),
    }


def _compact_taili_context_pack_for_llm(pack: Dict[str, Any]) -> Dict[str, Any]:
    yaml_summary = pack.get("yaml") if isinstance(pack.get("yaml"), dict) else {}
    search_hits = pack.get("search_hits") if isinstance(pack.get("search_hits"), dict) else {}
    return {
        "kind": pack.get("kind"),
        "permission_boundary": pack.get("permission_boundary"),
        "sources": pack.get("sources"),
        "yaml": {
            "available": yaml_summary.get("available", True),
            "path": yaml_summary.get("path") or (pack.get("sources") or {}).get("strategy_yaml"),
            "error": yaml_summary.get("error"),
            "profile": yaml_summary.get("profile"),
            "task_default_id": yaml_summary.get("task_default_id"),
            "recipe": yaml_summary.get("recipe"),
            "model": yaml_summary.get("model"),
            "observation_contract": yaml_summary.get("observation_contract"),
            "amp": yaml_summary.get("amp"),
            "control": yaml_summary.get("control"),
            "actuator": yaml_summary.get("actuator"),
            "commands": yaml_summary.get("commands"),
            "curriculum_gates": yaml_summary.get("curriculum_gates"),
            "phases": yaml_summary.get("phases"),
            "reward_keys": yaml_summary.get("reward_keys"),
            "reward_core": yaml_summary.get("reward_core"),
            "skrl": yaml_summary.get("skrl"),
        },
        "matched_docs": [
            {
                "source": item.get("source"),
                "title": item.get("title"),
                "text": _truncate(item.get("text"), 900),
            }
            for item in (search_hits.get("matches") or [])[:5]
            if isinstance(item, dict)
        ],
        "doc_inventory": [
            {
                "name": item.get("name"),
                "role": item.get("role"),
                "available": item.get("available"),
                "path": item.get("path"),
            }
            for item in (pack.get("docs") or [])[:8]
            if isinstance(item, dict)
        ],
        "operator_hint": pack.get("operator_hint"),
    }


def _compact_spec_coverage_for_llm(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "spec_path": report.get("spec_path"),
        "summary": report.get("summary"),
        "legend": report.get("legend"),
        "items": [
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "group": item.get("group"),
                "status": item.get("status"),
                "training_mechanism": item.get("training_mechanism"),
                "evaluation": item.get("evaluation"),
                "gaps": item.get("gaps"),
                "next_actions": item.get("next_actions"),
                "evidence": item.get("evidence"),
            }
            for item in (report.get("items") or [])
            if isinstance(item, dict)
        ],
    }


def _compact_definitions_for_llm(definitions: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "query": definitions.get("query"),
        "items": [
            {
                "key": item.get("key"),
                "label": item.get("label"),
                "category": item.get("category"),
                "formula": item.get("formula"),
                "meaning": item.get("meaning"),
                "current_reading": item.get("current_reading"),
                "source": item.get("source"),
                "related": item.get("related"),
            }
            for item in (definitions.get("items") or [])[:28]
            if isinstance(item, dict)
        ],
    }


def _compact_code_knowledge_for_llm(code: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "kind": code.get("kind"),
        "permission_boundary": code.get("permission_boundary"),
        "query": code.get("query"),
        "matched_signal_rows": [
            {
                "id": row.get("id"),
                "concept": row.get("concept"),
                "spec_items": row.get("spec_items"),
                "yaml_keys": row.get("yaml_keys"),
                "telemetry_fields": row.get("telemetry_fields"),
                "diagnostic_fields": row.get("diagnostic_fields"),
                "code_refs": row.get("code_refs"),
                "notes": row.get("notes"),
            }
            for row in (code.get("matched_signal_rows") or [])[:8]
            if isinstance(row, dict)
        ],
        "files_considered": code.get("files_considered"),
        "snippets": [
            {
                "path": item.get("path"),
                "line_start": item.get("line_start"),
                "line_end": item.get("line_end"),
                "matched_terms": item.get("matched_terms"),
                "text": _truncate(item.get("text"), 1600),
            }
            for item in (code.get("snippets") or [])[:12]
            if isinstance(item, dict)
        ],
        "consumption_checks": [
            {
                "key": item.get("key"),
                "status": item.get("status"),
                "occurrences": item.get("occurrences"),
                "caveat": item.get("caveat"),
            }
            for item in (code.get("consumption_checks") or [])[:24]
            if isinstance(item, dict)
        ],
        "gaps": code.get("gaps"),
        "limitations": code.get("limitations"),
    }


def _compact_tuning_ledger_for_llm(ledger: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "kind": ledger.get("kind"),
        "permission_boundary": ledger.get("permission_boundary"),
        "summary": ledger.get("summary"),
        "turns": [
            {
                "id": item.get("id"),
                "created_at": item.get("created_at"),
                "role": item.get("role"),
                "content_preview": _truncate(item.get("content_preview"), 500),
                "evidence_tools": item.get("evidence_tools"),
                "proposed_action": item.get("proposed_action"),
                "grounding": item.get("grounding"),
                "runs": item.get("runs"),
                "diagnostic_jobs": item.get("diagnostic_jobs"),
                "checkpoints": item.get("checkpoints"),
            }
            for item in (ledger.get("turns") or [])[-16:]
            if isinstance(item, dict)
        ],
        "proposals": [
            {
                "id": item.get("id"),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "status": item.get("status"),
                "name": item.get("name"),
                "args": item.get("args"),
                "risk": item.get("risk"),
                "evidence": item.get("evidence"),
                "requires": item.get("requires"),
                "expected_result": _truncate(item.get("expected_result"), 500),
                "reply_preview": _truncate(item.get("reply_preview"), 500),
                "result_preview": _truncate(item.get("result_preview"), 500),
                "runs": item.get("runs"),
                "diagnostic_jobs": item.get("diagnostic_jobs"),
                "checkpoints": item.get("checkpoints"),
            }
            for item in (ledger.get("proposals") or [])[:16]
            if isinstance(item, dict)
        ],
        "limitations": ledger.get("limitations"),
    }


def _compact_diagnostic_report_for_llm(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "job_id": report.get("job_id"),
        "preset": report.get("preset"),
        "preset_label": report.get("preset_label"),
        "checkpoint": report.get("checkpoint"),
        "output_dir": report.get("output_dir"),
        "coverage": report.get("coverage"),
        "commands": report.get("commands"),
        "posture": report.get("posture"),
        "events": report.get("events"),
        "terrain": report.get("terrain"),
        "robustness": report.get("robustness"),
        "behavior_context": report.get("behavior_context"),
        "notes": report.get("notes"),
        "value_explanations": report.get("value_explanations"),
        "instruction": report.get("instruction"),
    }


def _select_diagnostic_report_ids(query: str, history_payload: Dict[str, Any] | None) -> List[str]:
    explicit = re.findall(r"\b20\d{6}_\d{6}\b", query or "")
    if explicit:
        return explicit
    if not history_payload:
        return []
    items = [item for item in (history_payload.get("items") or []) if isinstance(item, dict)]
    if not items:
        return []

    text = (query or "").lower()
    wanted_dirs: set[str] = set()
    direction_aliases = {
        "forward": ("forward", "前进", "fwd"),
        "backward": ("backward", "后退", "back"),
        "lateral": ("lateral", "横移", "侧向", "lat"),
        "yaw": ("yaw", "偏航", "转向"),
    }
    for direction, aliases in direction_aliases.items():
        if any(alias in text for alias in aliases):
            wanted_dirs.add(direction)

    completed = []
    for item in items:
        state = str(item.get("state") or "").lower()
        artifacts = item.get("artifacts") or []
        has_report = any(
            isinstance(artifact, dict)
            and artifact.get("available")
            and str(artifact.get("kind") or "").lower() in {"report", "diagnostics.report", "metrics"}
            for artifact in artifacts
        )
        if state in {"complete", "completed", "done", "success"} or has_report:
            completed.append(item)
    candidates = completed or items

    if wanted_dirs:
        filtered = []
        for item in candidates:
            haystack = " ".join(
                str(item.get(key) or "").lower()
                for key in ("preset", "preset_label", "message", "source")
            )
            if any(direction in haystack for direction in wanted_dirs):
                filtered.append(item)
        if filtered:
            candidates = filtered

    limit = 2 if any(token in text for token in ("最新两个", "最近两个", "latest two", "last two")) else 1
    return [str(item.get("job_id") or "") for item in candidates[:limit] if item.get("job_id")]


def _grep_hint_for_query(query: str) -> str:
    text = (query or "").lower()
    terms: list[str] = []
    if any(token in text for token in ("高度", "height", "base_h", "termination")):
        terms.extend(["base_h", "height", "h_ok", "termination"])
    if any(token in text for token in ("滑", "slip")):
        terms.extend(["slip", "stance"])
    if any(token in text for token in ("步态", "gait", "duty", "diagonal", "air")):
        terms.extend(["gait", "duty", "diagonal", "air"])
    if any(token in text for token in ("奖励", "reward", "权重", "惩罚")):
        terms.extend(["reward", "rew_", "weight", "w_"])
    if any(token in text for token in ("actor", "critic", "网络", "perceiver", "感知器")):
        terms.extend(["actor", "critic", "perceiver", "observation"])
    if any(token in text for token in ("执行器", "actuator", "kp", "kd", "stiffness", "damping")):
        terms.extend(["actuator", "stiffness", "damping"])
    if not terms:
        return ""
    # Keep it shell-safe for the existing grep wrapper: only literals and regex separators.
    return "|".join(dict.fromkeys(re.sub(r"[^A-Za-z0-9_]+", "", term) for term in terms if term))


def _first_dict_items(value: Any, limit: int) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(k): v for k, v in list(value.items())[:limit]}


def _compact_metric_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "key": item.get("key"),
        "label": item.get("label"),
        "value": item.get("value"),
        "numeric_value": item.get("numeric_value"),
        "unit": item.get("unit"),
        "tone": item.get("tone"),
        "direction": item.get("direction"),
        "target": item.get("target"),
        "target_label": item.get("target_label"),
        "source": item.get("source"),
        "formula": _truncate(item.get("formula"), 220),
        "meaning": _truncate(item.get("meaning"), 260),
        "reading": _truncate(item.get("reading"), 220),
    }


def _compact_blocker(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "key": item.get("key"),
        "label": item.get("label"),
        "value": item.get("value"),
        "target": item.get("target"),
        "direction": item.get("direction"),
        "gap": item.get("gap"),
        "severity": item.get("severity"),
        "source": item.get("source"),
        "detail": item.get("detail"),
    }


def _truncate(value: Any, limit: int = 400) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _tool_get_remote_status(src: RealDataSource) -> Dict[str, Any]:
    """Read-only remote machine snapshot: GPU memory/utilization, RAM, disk, tmux, training procs."""
    import asyncio

    status = asyncio.run(src.remote_machine_status())
    return status.model_dump()


def _compact_operator_context_for_llm(context: Dict[str, Any]) -> Dict[str, Any]:
    """Trim the one-shot context into a bounded payload the model can actually read.

    The full context intentionally contains auditable raw structures for API/UI consumers. The
    chat loop needs the decisive facts, not 1MB+ of telemetry history and doc previews.
    """
    taili = context.get("taili_context_pack") if isinstance(context.get("taili_context_pack"), dict) else {}
    telemetry = context.get("training_telemetry") if isinstance(context.get("training_telemetry"), dict) else {}
    remote = context.get("remote_status") if isinstance(context.get("remote_status"), dict) else {}
    latest = telemetry.get("latest") if isinstance(telemetry.get("latest"), dict) else {}
    snapshot = telemetry.get("snapshot") if isinstance(telemetry.get("snapshot"), dict) else {}
    yaml_summary = taili.get("yaml") if isinstance(taili.get("yaml"), dict) else {}
    search_hits = taili.get("search_hits") if isinstance(taili.get("search_hits"), dict) else {}

    def _first_dict_items(value: Any, limit: int) -> Dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        return {str(k): v for k, v in list(value.items())[:limit]}

    gpus = []
    for gpu in remote.get("gpus") or []:
        if not isinstance(gpu, dict):
            continue
        gpus.append({
            "index": gpu.get("index"),
            "name": gpu.get("name"),
            "memory_used_mb": gpu.get("memory_used_mb"),
            "memory_total_mb": gpu.get("memory_total_mb"),
            "utilization_gpu_pct": gpu.get("utilization_gpu_pct"),
            "temperature_c": gpu.get("temperature_c"),
            "power_w": gpu.get("power_w"),
            "processes": [
                {
                    "pid": proc.get("pid"),
                    "name": proc.get("name"),
                    "used_memory_mb": proc.get("used_memory_mb"),
                }
                for proc in (gpu.get("processes") or [])[:3]
                if isinstance(proc, dict)
            ],
        })

    diagnostics = context.get("diagnostic_history") if isinstance(context.get("diagnostic_history"), dict) else {}
    diag_items = []
    for item in diagnostics.get("items") or []:
        if not isinstance(item, dict):
            continue
        diag_items.append({
            "job_id": item.get("job_id"),
            "state": item.get("state"),
            "preset": item.get("preset"),
            "preset_label": item.get("preset_label"),
            "checkpoint": item.get("checkpoint"),
            "updated_at": item.get("updated_at"),
        })

    return {
        "permission_boundary": context.get("permission_boundary"),
        "query": context.get("query"),
        "taili_strategy": {
            "sources": taili.get("sources"),
            "yaml": {
                "profile": yaml_summary.get("profile"),
                "task_default_id": yaml_summary.get("task_default_id"),
                "recipe": yaml_summary.get("recipe"),
                "model": yaml_summary.get("model"),
                "observation_contract": yaml_summary.get("observation_contract"),
                "control": yaml_summary.get("control"),
                "actuator": yaml_summary.get("actuator"),
                "commands": yaml_summary.get("commands"),
                "curriculum_gates": yaml_summary.get("curriculum_gates"),
                "phases": yaml_summary.get("phases"),
                "reward_core": yaml_summary.get("reward_core"),
                "skrl": yaml_summary.get("skrl"),
            },
            "matched_docs": [
                {
                    "source": item.get("source"),
                    "title": item.get("title"),
                    "text": str(item.get("text") or "")[:700],
                }
                for item in (search_hits.get("matches") or [])[:3]
                if isinstance(item, dict)
            ],
        },
        "live_training": {
            "available": telemetry.get("available"),
            "running": telemetry.get("running"),
            "run_id": telemetry.get("run_id"),
            "mode": telemetry.get("mode"),
            "summary": telemetry.get("summary"),
            "log_path": telemetry.get("log_path"),
            "telemetry_path": telemetry.get("telemetry_path"),
            "source_stats": telemetry.get("source_stats"),
            "latest": {
                "step": latest.get("step"),
                "total_steps": latest.get("total_steps"),
                "elapsed": latest.get("elapsed"),
                "eta": latest.get("eta"),
                "fps": latest.get("fps"),
                "checkpoint": latest.get("checkpoint"),
                "reward": _first_dict_items(latest.get("reward"), 48),
                "curriculum": _first_dict_items(latest.get("curriculum"), 32),
                "health": _first_dict_items(latest.get("health"), 32),
                "command": _first_dict_items(latest.get("command"), 32),
                "counters": _first_dict_items(latest.get("counters"), 16),
            },
            "snapshot": {
                "status": snapshot.get("status"),
                "conclusion": snapshot.get("conclusion"),
                "progress_pct": snapshot.get("progress_pct"),
                "phase": snapshot.get("phase"),
                "command_mode": snapshot.get("command_mode"),
                "active_dirs": snapshot.get("active_dirs"),
                "blocked_by": snapshot.get("blocked_by"),
                "next_gate": snapshot.get("next_gate"),
                "latest_checkpoint": snapshot.get("latest_checkpoint"),
                "blockers": [
                    {
                        "key": blocker.get("key"),
                        "label": blocker.get("label"),
                        "value": blocker.get("value"),
                        "target": blocker.get("target"),
                        "direction": blocker.get("direction"),
                        "gap": blocker.get("gap"),
                        "severity": blocker.get("severity"),
                        "detail": blocker.get("detail"),
                        "source": blocker.get("source"),
                    }
                    for blocker in (snapshot.get("blockers") or [])[:6]
                    if isinstance(blocker, dict)
                ],
                "notes": snapshot.get("notes"),
            },
            "limitations": telemetry.get("limitations"),
            "error": telemetry.get("error"),
        },
        "remote_machine": {
            "available": remote.get("available"),
            "host": remote.get("host"),
            "uptime": remote.get("uptime"),
            "load_avg": remote.get("load_avg"),
            "cpu_count": remote.get("cpu_count"),
            "memory_total_mb": remote.get("memory_total_mb"),
            "memory_used_mb": remote.get("memory_used_mb"),
            "memory_available_mb": remote.get("memory_available_mb"),
            "gpus": gpus,
            "disks": remote.get("disks"),
            "tmux_sessions": remote.get("tmux_sessions"),
            "training_processes": [
                {
                    "pid": proc.get("pid"),
                    "cpu_pct": proc.get("cpu_pct"),
                    "mem_pct": proc.get("mem_pct"),
                    "etime": proc.get("etime"),
                    "command": str(proc.get("command") or "")[:160],
                }
                for proc in (remote.get("training_processes") or [])[:5]
                if isinstance(proc, dict)
            ],
            "error": remote.get("error"),
        },
        "workspace_config": context.get("workspace_config"),
        "definitions": context.get("definitions"),
        "diagnostic_history": {"items": diag_items},
    }


def _tool_probe_telemetry(src: RealDataSource) -> Dict[str, Any]:
    """Read-only telemetry wiring probe for LLM grounding.

    It reports where the console is reading from and whether each artifact is
    present. It does not run arbitrary shell supplied by the LLM.
    """
    r = src._get_remote()
    run = src._newest_run(r)
    paths = src._resolve_telemetry_paths(r, run)
    import shlex

    files = {
        "train_log": paths.log_path,
        "telemetry_jsonl": paths.telemetry_path,
        "console_log": paths.console_log_path,
        "checkpoint_dir": paths.checkpoint_dir,
    }
    script_lines = []
    for label, path in files.items():
        if not path:
            script_lines.append(f"echo {label}=missing:path-empty")
            continue
        q = shlex.quote(path)
        if label.endswith("_dir"):
            script_lines.append(f"if [ -d {q} ]; then echo {label}=present; else echo {label}=missing; fi")
        else:
            script_lines.append(
                f"if [ -s {q} ]; then "
                f"printf '{label}=present bytes='; stat -c '%s' {q}; "
                f"printf '{label}_lines='; wc -l < {q}; "
                f"else echo {label}=missing; fi"
            )
    raw = r.exec_out("bash -lc " + shlex.quote("; ".join(script_lines)), timeout=15) or ""
    return {
        "run": run,
        "paths": files,
        "path_evidence": list(paths.evidence),
        "running": src._is_running(r),
        "checks": [line.strip() for line in raw.splitlines() if line.strip()],
    }


def _tool_explain_definition(src: RealDataSource, query: str = "") -> Dict[str, Any]:
    """Read-only definition registry for reward/curriculum/diagnostic terms."""
    from .definitions import list_definitions

    return list_definitions(query).model_dump()


def _tool_get_eval_result(src: RealDataSource, lines: int = 60) -> Dict[str, Any]:
    """Read the physeval RESULT (the per-command motion table), skipping Isaac startup noise.
    For narrating the result of a run_physeval action back to the operator."""
    r = src._get_remote()
    log = src.profile.physeval_log
    running = bool((r.exec_out("pgrep -fa physeval.py | grep -v pgrep") or "").strip())
    # extract from the CHECKPOINT header to the end - that's the actual result block
    result = r.exec_out(f"sed -n '/CHECKPOINT/,$p' {log} 2>/dev/null | head -40") or ""
    if not result.strip():
        result = r.exec_out(f"tail -n {int(lines)} {log} 2>/dev/null") or ""
    note = ("eval still running - partial result" if running
            else ("done" if result.strip() else "no eval result yet - eval may not have run"))
    return {"log_path": log, "eval_running": running, "status": note,
            "result_table": result[-3000:] if result.strip() else "(none)"}


def _tool_get_workspace_config(src: RealDataSource) -> Dict[str, Any]:
    settings = src.settings
    config_set = get_active_config_set(settings)
    llm = llm_profile(settings)
    return {
        "config_set": {
            "id": config_set.id,
            "label": config_set.label,
            "status": config_set.status,
            "task_goal": config_set.task_goal,
            "remote": _profile_summary(config_set.remote),
            "robot": _profile_summary(config_set.robot),
            "framework": _profile_summary(config_set.framework),
            "llm": _profile_summary(config_set.llm),
            "notes": list(config_set.notes),
        },
        "runtime": {
            "source": settings.source,
            "active_framework_id": settings.framework_id,
            "desired_framework_id": desired_framework_id(settings),
            "diagnostic_task": settings.diagnostic_task,
        },
        "llm_status": {
            "configured": llm.configured,
            "model": llm.model,
            "base_url": llm.base_url,
            "api_key_env_var": llm.api_key_env_var,
            "api_key_resolved": llm.api_key_resolved,
            "unsafe_literal_key": llm.unsafe_literal_key,
            "advisory_only": llm.advisory_only,
            "actions_require_confirmation": llm.actions_require_confirmation,
        },
    }


def _tool_list_frameworks(src: RealDataSource) -> Dict[str, Any]:
    settings = src.settings
    desired_id = desired_framework_id(settings)
    return {
        "active_framework_id": settings.framework_id,
        "desired_framework_id": desired_id,
        "frameworks": [
            {
                "id": profile.id,
                "label": profile.label,
                "status": profile.status,
                "experiment": profile.experiment,
                "task_id": profile.task_id,
                "diagnostic_task": profile.diagnostic_task,
                "active": profile.id == settings.framework_id,
                "desired": profile.id == desired_id,
                "note": profile.note,
            }
            for profile in list_framework_profiles()
        ],
    }


def _tool_list_diagnostic_history(src: RealDataSource, limit: int = 5) -> Dict[str, Any]:
    history = DiagnosticHistoryStore(src.settings).list(limit=limit)
    return {
        "items": [
            {
                "job_id": item.job_id,
                "state": item.state,
                "preset": item.preset,
                "preset_label": item.preset_label,
                "checkpoint_name": item.checkpoint_name,
                "framework_id": item.framework_id,
                "diagnostic_task": item.diagnostic_task,
                "source": item.source,
                "progress": item.progress,
                "message": item.message,
                "updated_at": item.updated_at,
                "elapsed_s": item.elapsed_s,
                "artifacts": [
                    {"kind": artifact.kind, "available": artifact.available, "uri": artifact.uri}
                    for artifact in item.artifacts
                ],
            }
            for item in history.items
        ]
    }


def _tool_get_diagnostic_report(src: RealDataSource, job_id: str = "") -> Dict[str, Any]:
    from .diagnostics import DiagnosticsController
    from .slash_commands import _diagnostic_behavior_context

    async def _load() -> Dict[str, Any]:
        controller = DiagnosticsController(src.settings, src)
        report = await controller.report_for_job(job_id=job_id or None)
        behavior = _diagnostic_behavior_context(report)
        return {
            "job_id": report.job_id,
            "preset": report.preset,
            "preset_label": report.preset_label,
            "checkpoint": report.checkpoint.rsplit("/", 1)[-1] if report.checkpoint else "",
            "output_dir": report.output_dir,
            "coverage": report.coverage,
            "commands": report.commands,
            "posture": report.posture,
            "events": report.events,
            "notes": report.notes[:8],
            "terrain": report.terrain,
            "robustness": report.robustness,
            "behavior_context": behavior,
            "value_explanations": [
                {
                    "key": item.key,
                    "label": item.label,
                    "source": item.source,
                    "formula": item.formula,
                    "interpretation": item.interpretation,
                }
                for item in report.value_explanations[:16]
            ],
            "instruction": (
                "Use behavior_context for behavior-level explanation. Do not just recite fields. "
                "Describe the test intent, what the robot appears to do from aggregate evidence, "
                "what is evidence vs inference, and what the operator should inspect in playback."
            ),
        }

    import asyncio
    return asyncio.run(_load())


def _tool_run_workflow(src: RealDataSource, task: str = "", angles: str = "",
                       verify: bool = True, max_workers: int = 4) -> Dict[str, Any]:
    """ULTRACODE / multi-agent WORKFLOW orchestration for the LLM owner.

    The single-shot brain answers routine questions well, but a HIGH-STAKES call (which lever to
    tune next, the root cause of a plateau, whether a structural change is safe, "be thorough / 全面
    分析") deserves the same investigate -> adversarially-verify -> synthesize rigor the Claude Code
    operator runs as a Workflow. This gives the owner that mechanism INSIDE the console: it gathers
    grounded evidence once, fans out parallel angle-analyses (ThreadPoolExecutor over
    call_llm_with_schema), has each finding refuted by an adversarial verifier, then synthesizes a
    decisive conclusion + next action. Every sub-agent reasons ONLY from the shared evidence, so the
    whole workflow stays grounded (no free-association).

    COST-AWARE: ~1 + N + N + 1 LLM calls (N = angles). Use for substantive/high-stakes reasoning,
    NOT routine status — the remote box and the token budget both cost. Default 3 angles.
    """
    from autotuner.llm_gateway.client import call_llm_with_schema
    from concurrent.futures import ThreadPoolExecutor
    if not task.strip():
        return {"error": "run_workflow needs a task/question to reason about."}
    # 1) gather grounded evidence ONCE (shared by every sub-agent)
    try:
        evidence = _tool_get_evidence_context(src, query=task)
    except Exception as e:  # noqa: BLE001
        evidence = {"error": f"{type(e).__name__}: {e}"}
    ev_text = json.dumps(evidence, ensure_ascii=False, default=str)[:8000]
    # 2) fan out angle-analyses (parallel)
    angle_list = [a.strip() for a in angles.split(",") if a.strip()] or [
        "correctness / root-cause", "efficiency / cost", "risk / failure-mode"]
    angle_list = angle_list[:5]

    def _investigate(angle: str) -> Dict[str, Any]:
        sys_p = ("You are a grounded RL-locomotion-tuning analyst. Reason ONLY from the EVIDENCE "
                 "provided — never invent runs, numbers, or gate names. Return STRICT JSON: "
                 '{"finding": "<one specific claim>", "grounded_on": "<which evidence>", '
                 '"confidence": "high|med|low"}.')
        usr_p = (f"TASK: {task}\nANGLE: {angle}\n\nEVIDENCE:\n{ev_text}\n\n"
                 f"Give the single most important finding from the {angle} angle. If the evidence "
                 "is insufficient for this angle, say so and set confidence=low.")
        r = call_llm_with_schema(sys_p, usr_p, schema_name="console_workflow_investigate")
        p = (r.parsed if (r and r.parsed) else None) or {"finding": "(no response)", "confidence": "low"}
        return {"angle": angle, "finding": p.get("finding", ""),
                "grounded_on": p.get("grounded_on", ""), "confidence": p.get("confidence", "low")}

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(angle_list)))) as ex:
        findings = list(ex.map(_investigate, angle_list))

    # 3) adversarial verification (parallel) — refute-by-default keeps plausible-but-unsupported out
    if verify:
        def _verify(f: Dict[str, Any]) -> Dict[str, Any]:
            sys_p = ("You are an adversarial verifier. Try to REFUTE the finding using ONLY the "
                     "evidence. Return STRICT JSON: {\"holds\": true|false, \"reason\": \"<why>\"}. "
                     "Default holds=false if the evidence does not clearly support it.")
            usr_p = (f"FINDING ({f['angle']}): {f['finding']}\ngrounded_on: {f['grounded_on']}\n\n"
                     f"EVIDENCE:\n{ev_text}")
            r = call_llm_with_schema(sys_p, usr_p, schema_name="console_workflow_verify")
            v = (r.parsed if (r and r.parsed) else None) or {"holds": False, "reason": "no response"}
            return {**f, "holds": bool(v.get("holds", False)), "verify_reason": v.get("reason", "")}
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(findings)))) as ex:
            findings = list(ex.map(_verify, findings))
    survived = [f for f in findings if (not verify) or f.get("holds")]

    # 4) synthesize a decisive, grounded conclusion + next action
    sys_p = ("You are the synthesis lead for an RL-tuning console. Combine the VERIFIED findings "
             "into ONE decisive, grounded conclusion and a concrete next action (a specific lever, "
             "measurement, or experiment). Reason only from the findings/evidence. Return STRICT "
             'JSON: {"conclusion": "<...>", "next_action": "<...>", "confidence": "high|med|low", '
             '"caveat": "<the main uncertainty>"}.')
    usr_p = (f"TASK: {task}\n\nVERIFIED FINDINGS:\n"
             f"{json.dumps(survived or findings, ensure_ascii=False, default=str)[:8000]}")
    r = call_llm_with_schema(sys_p, usr_p, schema_name="console_workflow_synthesize")
    synth = (r.parsed if (r and r.parsed) else None) or {"conclusion": "(synthesis unavailable)"}
    return {
        "task": task,
        "n_angles": len(angle_list),
        "findings": findings,
        "survived_verification": len(survived),
        "synthesis": synth,
        "mechanism": "investigate -> adversarial-verify -> synthesize (ultracode workflow, grounded on shared evidence)",
    }


TOOLS = {
    "get_status": _tool_get_status,
    "run_workflow": _tool_run_workflow,
    "list_runs": _tool_list_runs,
    "get_reward_curve": _tool_get_reward_curve,
    "get_run_ledger": _tool_get_run_ledger,
    "get_evidence_context": _tool_get_evidence_context,
    "get_operator_context": _tool_get_operator_context,
    "get_training_telemetry": _tool_get_training_telemetry,
    "get_remote_status": _tool_get_remote_status,
    "probe_telemetry": _tool_probe_telemetry,
    "explain_definition": _tool_explain_definition,
    "get_code_knowledge": _tool_get_code_knowledge,
    "get_signal_map": _tool_get_signal_map,
    "get_tuning_ledger": _tool_get_tuning_ledger,
    "get_train_log": _tool_get_train_log,
    "get_eval_result": _tool_get_eval_result,
    "get_config": _tool_get_config,
    "get_asset": _tool_get_asset,
    "search_knowledge": _tool_search_knowledge,
    "get_taili_context_pack": _tool_get_taili_context_pack,
    "get_spec_coverage": _tool_get_spec_coverage,
    "get_acceptance": _tool_get_acceptance,
    "analyze_acceptance": _tool_analyze_acceptance,
    "get_operations_state": _tool_get_operations_state,
    "get_strategy": _tool_get_strategy,
    "get_tuning_state": _tool_get_tuning_state,
    "get_workspace_config": _tool_get_workspace_config,
    "list_frameworks": _tool_list_frameworks,
    "list_diagnostic_history": _tool_list_diagnostic_history,
    "get_diagnostic_report": _tool_get_diagnostic_report,
    "get_playbook": _tool_get_playbook,
    "get_campaign_journal": _tool_get_campaign_journal,
    "record_campaign_iteration": _tool_record_campaign_iteration,
}

_TOOLS_DOC = """Available tools (read-only):
- run_workflow(task, angles="", verify=true) -> ULTRACODE multi-agent workflow: investigate the task from
                                   several angles IN PARALLEL, adversarially verify each finding, then synthesize
                                   a decisive conclusion + next action — all grounded on one shared evidence pull.
                                   Use ONLY for HIGH-STAKES / "be thorough" reasoning (which lever next, root cause
                                   of a plateau, is a structural change safe, 全面分析/深入分析). It costs ~1+2N+1
                                   LLM calls, so do NOT use it for routine status/definition questions — answer
                                   those directly. This is how you bring operator-grade rigor to the big calls.
- get_playbook(task="", gate="", robot="") -> YOUR systematized operating knowledge + workflows. Call this
                                   FIRST when driving work: returns the robot-AGNOSTIC ordered workflow (guarded
                                   state machine), the active robot PROFILE's gate→lever map with cautions, ops
                                   signatures, and grounding rules. This is how you act as the system's owner.
- get_campaign_journal(robot="taili") -> durable decision memory: best-so-far score + the (gate::lever) pairs
                                   already tried and ROLLED BACK. Read at step 3 (select lever) to obey NO-REPEAT
                                   — never re-apply a rolled-back lever, never claim a stale best.
- record_campaign_iteration(target_gate, lever, decision, score_before, score_after, ...) -> append this
                                   iteration's outcome (decision in {kept,rolled_back,pending}) at step 7 (decide),
                                   so NO-REPEAT/DECIDE survive across turns, restarts, and sessions.
- get_status()                  -> host, newest run, latest checkpoint iter, whether training runs
- get_evidence_context(query="", ui_mode="", include_optional=true)
                                -> deterministic evidence router + compact context. Use FIRST for
                                   broad/current/tuning/root-cause/diagnostic behavior questions.
                                   It selects only allowlisted sources, collects them, marks missing
                                   or stale evidence as gaps, and leaves the final wording flexible.
- get_operator_context(query="", include_diagnostics=true)
                                -> legacy one-shot controlled context for high-quality answers: Taili
                                   spec/strategy/YAML pack, live training telemetry, remote GPU/RAM/
                                   disk/tmux state, active workspace config, definitions, and recent
                                   diagnostic history. Use only if get_evidence_context is unavailable
                                   or a legacy bundle is explicitly needed.
- get_training_telemetry()      -> structured live telemetry: step/ETA/fps, reward breakdown,
                                   curriculum/gates, health, checkpoint, JSONL/log provenance
- get_remote_status()           -> remote machine health: GPU memory/utilization/temperature,
                                   RAM, disk, tmux sessions, and likely training processes
- probe_telemetry()             -> read-only wiring probe: current run, exact train.log /
                                   train.telemetry.jsonl / console.log / checkpoint_dir paths,
                                   presence, sizes, line counts, and discovery evidence
- explain_definition(query="")  -> definition registry for reward/curriculum/diagnostic terms
- get_signal_map(query="")      -> curated spec/YAML/code/telemetry/diagnostic map for Taili
                                   signals. Use to avoid missing a source layer.
- get_code_knowledge(query="")  -> allowlisted Taili implementation evidence: source snippets,
                                   file/line windows, and YAML key consumption checks. Use for
                                   "where is this computed?", "is this field consumed?", "dead key",
                                   or YAML-code alignment questions.
- get_tuning_ledger(limit=24)   -> read-only tuning memory from recent LLM turns/proposals:
                                   runs, diagnostic jobs, checkpoints, proposed actions/results.
- list_runs(limit=8)            -> recent training runs with their max checkpoint iter
- get_reward_curve(run="")      -> reward curve + summary (max/min/biggest_drop) for a run
                                   (run = substring to pick a run; empty = newest)
- get_run_ledger(limit=8)       -> recent runs with max checkpoint iter + reward peak/final
                                   (for comparing runs / choosing a version)
- get_train_log(lines=40)       -> tail of the live training log + whether training is running
- get_eval_result(lines=60)     -> tail of the last physeval output + whether eval still running
- get_config(grep="")           -> read-only view of the deployed env config (reward weights /
                                   key params); optional grep regex to focus
- get_asset()                   -> read-only view of the robot asset (actuator stiffness/damping/
                                   effort/mass) - for actuator/hardware-root-cause diagnosis
- get_taili_context_pack(query="")
                                -> controlled local knowledge pack: Taili spec, strategy decisions,
                                   system architecture notes, and the current Taili YAML contract.
                                   Use before strategy/config/tuning questions.
- get_spec_coverage()           -> taili_spec acceptance ledger: each spec row mapped to current
                                   training mechanism, evaluation coverage, gaps, and next actions.
- get_acceptance()              -> MEASURED taili_spec §2 verdict for the newest run (scored from
                                   physeval logs; never launches physeval): passed, per-family
                                   present/ok, and failing sub-gates with stat-vs-band. THE metric
                                   the product is graded on — prefer over get_spec_coverage for any
                                   "which gates fail / how far from benchmark / what to tune" question.
- get_operations_state()        -> SYSTEM SELF-AWARENESS: what automations are running (auto_drive
                                   self-heal / campaign / measurement), step freshness, and the ops-
                                   runbook STALL CLASSIFIER diagnosis. Use FIRST for 卡住/状态/为什么
                                   不动 questions — quote its diagnosis, do NOT invent failure modes.
- get_strategy()                -> FULL tuning strategy: reward weights, curriculum phases +
                                   advancement gates, AMP hyperparams (what apply_tuning edits)
- get_tuning_state()            -> rollback stack, BEST_CHECKPOINT registry, training-running flag
- analyze_acceptance()          -> TUNING BRAIN: failing gates ranked worst-first, each mapped to
                                   its reward lever (step + HARD stability cap), metric-artifact
                                   gates excluded, plus the heuristic's next apply_tuning proposal.
                                   Use to DRIVE the measure->tune->train->re-measure loop yourself.
- get_workspace_config()        -> current ConfigSet: remote, robot, framework, LLM status,
                                   active vs saved framework, diagnostic task
- list_frameworks()             -> framework registry with active/saved markers
- list_diagnostic_history(limit=5)
                                -> recent diagnostic jobs and artifact availability
- get_diagnostic_report(job_id="")
                                -> summarized metrics for a completed diagnostic job; empty
                                   job_id means latest restorable job
- search_knowledge(query="")    -> allowlisted project docs search over spec / strategy /
                                   architecture notes. Use it for focused supporting evidence
                                   after get_taili_context_pack when needed."""

_ACTIONS_DOC = """Permission model:
- read-only: status, training telemetry, logs, TensorBoard-derived curves, diagnostics reports,
             workspace config, framework registry, asset/config snippets, knowledge definitions.
- probe: allowlisted discovery only, such as checkpoint catalog/diagnostic history/log tail.
- action: only propose actions; execution requires operator confirmation through the console.
- forbidden: arbitrary shell, arbitrary remote file writes, deleting data, editing remote code,
             or claiming you ran an action without a confirmed ExecuteResponse.

Actions you may PROPOSE (you must NEVER execute these yourself - propose, and
the operator confirms before anything runs):
- deploy_payload                  build local Taili payload, upload to remote data disk, extract, and verify
- start_training                  start a new payload-first Taili training run from the newest deployed payload
- run_diagnostic                  run an allowlisted diagnostic preset against a selected/latest checkpoint
- kill_training                   stop the currently-running training
- resume_training                 resume payload-first training from the latest checkpoint when available
- run_physeval                    legacy only; use run_diagnostic for the payload-first runtime
- run_acceptance {terrains,checkpoint}
                                  MEASURE the policy vs taili_spec §2 (physeval -> scored verdict).
                                  Self-refuses if training is active. Read the result afterwards with
                                  get_acceptance. This is the copilot's own measure step of the loop.
- edit_config {key, value}        change ONE env_cfg field's value (auto-backs-up first;
                                  rollback available). Use the exact field name from get_config.
- rollback_config                 restore env_cfg from the last locomotion console backup
- apply_tuning {changes:{key:value}, note}
                                  edit the tuning STRATEGY (reward weights / curriculum gates) in the
                                  local contract — allowlisted keys + bounds only, comment-preserving,
                                  with a rollback-stack push. Then deploy_payload + resume_training to
                                  train with it. This is your own "tune" step. Keys from get_config /
                                  /config/strategy, e.g. {"changes": {"w_stance_slip": 0.3}}.
- rollback_tuning                 undo the most recent apply_tuning (pops the rollback stack,
                                  restores prior values field-for-field)
- produce_policy {max_iters}      THE PRODUCT: end-to-end produce the best benchmark policy —
                                  bootstrap from best, loop full-battery measure->analyze->tune->
                                  train->re-measure until benchmark passes / levers exhausted, emit
                                  the deliverable report. Propose when asked to "deliver a policy".
- run_campaign {run, checkpoint, max_iters}
                                  AUTONOMOUS tuning campaign: the system runs the whole
                                  measure->analyze->tune->train->re-measure loop unattended, keeping
                                  improvements and rolling back regressions, with stall-recovery. This
                                  is how the system completes a tuning task on its own. Long-running.
For edit_config, put the field name + new value in args, e.g.
{"propose_action": {"name": "edit_config", "args": {"key": "rew_torque", "value": "-3.0e-4"}},
 "reply": "..."}. Always read the current value with get_config first."""

# action names the agent will surface as a confirmation, not auto-run
ACTION_NAMES = {"deploy_payload", "start_training", "run_physeval", "resume_training", "kill_training",
                "edit_config", "rollback_config", "run_diagnostic", "run_acceptance",
                "apply_tuning", "rollback_tuning", "run_campaign", "produce_policy"}

# Per-action RISK TIERS — the authorization surface for the active copilot. Higher tiers demand
# stronger confirmation and a tighter blast radius. 'destructive' actions change the training/box
# state (start/kill/deploy) or GPU occupancy; 'medium' launch GPU measurement; 'low' edit local
# config with rollback. Enforced in execute_action + chat_execute (both gate on a bound proposal).
ACTION_RISK = {
    "edit_config": "low", "rollback_config": "low",
    "apply_tuning": "low", "rollback_tuning": "low",
    "run_diagnostic": "medium", "run_physeval": "medium", "run_acceptance": "medium",
    "deploy_payload": "destructive", "start_training": "destructive",
    "resume_training": "destructive", "kill_training": "destructive",
    "run_campaign": "destructive",   # autonomously drives many training runs
    "produce_policy": "destructive",  # the PRODUCT action: hours of autonomous train/tune
}
_RISK_RANK = {"auto": 0, "low": 1, "medium": 2, "destructive": 3}


_AUTONOMY_TIERS = {"advisory": frozenset(), "assisted": frozenset({"low"}),
                   "autonomous": frozenset({"low", "medium"})}


def _autonomy_auto_tiers() -> frozenset:
    """Risk tiers the copilot may EXECUTE without confirmation. Set via
    LOCOMOTION_CONSOLE_LLM_AUTONOMY: advisory (propose-only, default) | assisted (low) |
    autonomous (low+medium). Destructive is NEVER auto-executable."""
    import os as _os
    return _AUTONOMY_TIERS.get(_os.environ.get("LOCOMOTION_CONSOLE_LLM_AUTONOMY", "advisory").strip().lower(),
                               frozenset())


def action_risk(name: str) -> str:
    """Risk tier for an action name ('low' | 'medium' | 'destructive'); 'auto' for read-only."""
    return ACTION_RISK.get(name, "auto")

_SYSTEM = """You are the BRAIN of this RL-locomotion training SYSTEM — its resident operator, not an
outside analyst. The system runs YOUR automations (auto_drive self-healing restarts, tuning
campaigns, measurements): consult get_operations_state for what the system is doing right now, and
the ops runbook (search_knowledge '卡住'/'运营手册') for this project's KNOWN failure signatures and
hard-won training facts BEFORE diagnosing from raw telemetry. A frozen step with a live process and
pinned GPU is the documented GPU-kernel stall with auto-recovery — answer from that knowledge, in
the first person, as the one running the system. The
operator may not know what's on screen - your job is to read the REAL training box through the
read tools and tell them, in plain language, what is happening and what they can do, and to
PROPOSE actions when they ask for one.

Role boundary:
- The LLM reads instruments, explains evidence, prioritizes next steps, and drafts proposals.
  It does NOT own final authority over facts, gates, or actions.
- A framework is a selectable, composable capability package made from AMP / curriculum / DR /
  reward / diagnostics / deployment mechanisms.
- Use the concrete system terms: LLM, Framework, ConfigSet, Adapter, diagnostics, and gates.

%s

%s

Each turn, reply with STRICT JSON, ONE of:
  {"tool": "<read-tool>", "args": {...}}                          call a read tool, or
  {"reply": "<answer>"}                                           answer the operator, or
  {"propose_action": {"name": "<action>", "args": {}}, "reply": "<explain + ask to confirm>"}
                                                                  propose an action.

Rules:
- Base every number and claim on tool results you actually received. Never invent run names,
  iterations, or reward values. If a tool hasn't given you a fact, call the read tool first.
- For broad, current-state, root-cause, tuning, "why", "what should we do next", or diagnostic
  behavior questions, call get_evidence_context(query=<operator question>) first. It handles
  knowledge routing and gap detection. Use narrower tools only for a specific missing fact.
- Do not treat get_evidence_context as a reply template. Use it as evidence, then answer naturally
  in the operator's language. If its gaps include a blocking missing/stale source, state that
  boundary instead of guessing.
- Do not answer tuning questions from YAML/docs alone when current telemetry or diagnostic
  behavior is relevant. If the route selected telemetry or diagnostics and they are missing,
  say what is missing before proposing a change.
- Do not answer implementation/wiring/dead-key questions from YAML/docs alone. Use
  get_evidence_context first; if implementation evidence is still missing, call get_code_knowledge
  or get_signal_map directly. A config key is "consumed" only when code evidence shows it.
- For tuning-loop questions, use tuning_ledger as memory, but verify outcomes with current
  telemetry/diagnostic evidence before saying a change worked.
- Never present telemetry/reward/gates as proof of real robot behavior. They are training proxies;
  diagnostic reports and playback artifacts are behavior evidence.
- Do NOT repeat a tool call you already made this turn - the prior results are shown to you; act
  on what you have. You have a limited step budget; once you have the lesson + the current value,
  decide (answer or propose) instead of searching again.
- Be concise and concrete. Answer in the operator's language (Chinese if they wrote Chinese).
- For non-trivial answers in Chinese, do not sound like a fixed template. Use short natural
  paragraphs or compact bullets. Lead with the useful conclusion, then cite the few decisive
  evidence points. Explicitly mark inference when the tool output is indirect. Avoid dumping
  every field you saw.
- When you explain a reward drop, reason from the curve; for this project a sharp mid-run drop
  that then recovers is typically domain-randomization onset, but say so as inference, not fact.
- To act, emit propose_action. AUTONOMY policy: low/medium-risk actions (apply_tuning,
  rollback_tuning, run_acceptance, run_diagnostic) may AUTO-EXECUTE — their results then appear in
  the transcript as {"action": ...} records, which you may report as DONE (cite the result detail).
  DESTRUCTIVE actions (start/kill/deploy/campaign) always pause for operator confirmation.
  You are the TUNING BRAIN: proactively drive measure (run_acceptance) -> analyze
  (analyze_acceptance) -> tune (apply_tuning) -> train (deploy+resume, confirmed) -> re-measure.
  Never claim an action ran unless its {"action": ...} record is in the transcript.
- When the operator asks to evaluate, call list_diagnostic_history / probe_telemetry as needed
  and propose run_diagnostic, not legacy run_physeval, unless the active framework explicitly
  exposes a physeval command. When asked to resume/stop, gather facts first, then propose the
  available action with a clear confirm message.
- When the operator asks "where is training now?", "what does this value mean?", "is it stuck?",
  or refers to a metric on the dashboard, call get_training_telemetry first. For definitions or
  formulas call explain_definition. Use slash-style wording for suggested next commands.
- When the operator asks whether the remote machine is busy, why it is slow, GPU/VRAM usage,
  tmux, disk, memory, or system resources, call get_remote_status. If the question mixes
  training quality and speed, call both get_training_telemetry and get_remote_status.
- When the operator asks where logs/checkpoints are, whether the monitor is reading the right
  files, or why telemetry is missing, call probe_telemetry first; then call get_train_log only
  if a tail is needed.
- When the operator asks about the whole system, configuration, framework choice, LLM readiness,
  or what remains to be done, call get_workspace_config and list_frameworks before answering.
- When the operator asks about diagnostics, prior tests, rendered verification, or a checkpoint's
  behavior, call list_diagnostic_history first. If a completed job is relevant, call
  get_diagnostic_report with that job_id before making claims about the behavior.
- For behavior/diagnostic explanation questions, prefer get_evidence_context. It will route to
  diagnostic history/report and the relevant definitions. Explain what test was run, what the
  robot appears to do from aggregate evidence, and what remains an inference.
- For broad analysis questions ("what is the biggest problem?", "what do you think?", "按 spec 分析",
  "why is this bad?", "what should we do next?", tuning/change/root-cause questions), use
  get_evidence_context rather than the legacy get_operator_context. Then call get_diagnostic_report/
  get_config/get_asset only if a specific missing fact is needed.
- ULTRACODE escalation: for a HIGH-STAKES or explicitly-thorough call — which lever to try next,
  the root cause of a plateau, whether a structural/architecture change is safe, or the operator
  says "全面/深入/彻底分析", "be thorough", "use a workflow" — call run_workflow(task=<the question>)
  AFTER you have the basic evidence. It fans out parallel angle-analyses, adversarially verifies each,
  and synthesizes a decisive conclusion + next_action, all grounded on shared evidence. Read its
  synthesis and answer from it (cite survived_verification vs total). Do NOT use run_workflow for
  routine status/definition/"is it stuck" questions — it costs many LLM calls; answer those directly.
- For narrow strategy/config/YAML questions that do not need live state, call get_taili_context_pack.
  For a specific metric/formula, call explain_definition. For YAML-code alignment, call
  get_signal_map/get_code_knowledge. Ground advice in the spec/YAML/strategy/code evidence
  and cite the source names (taili_spec, taili_strategy_decisions, taili_blind_config.yaml,
  and source file paths when relevant).
  If the pack has nothing relevant, say so plainly.
- Diagnosis -> fix: once your grounded diagnosis points to a fix, decide its TYPE:
  * if it's a single tunable config field (a reward weight / param that appears in get_config),
    call get_config to read its CURRENT value, then PROPOSE edit_config with the exact field name
    and a concrete new value, and in the reply cite the lesson + state old->new. (The edit auto-
    backs-up and is rollback-able, so it is safe to propose.)
  * if it's a structural / code change (e.g. gating logic in the reference generator, an obs
    change), DESCRIBE it - do NOT propose edit_config for things that aren't a single cfg value.
  Always read the current value before proposing, so the new value is sensible relative to it.""" % (
    _TOOLS_DOC, _ACTIONS_DOC)


def run_agent(message: str, settings: LocomotionConsoleSettings,
              history: List[Dict[str, str]] | None = None,
              allow_slash: bool = True,
              max_steps: int = 9,
              preload_operator_context: bool = False) -> Dict[str, Any]:
    """Run the agentic loop for one user message.

    `history` is the prior conversation ([{role, content}, ...]) so the soul has memory across
    messages; it resolves references like "the eval result" from earlier context. Returns
    {reply, transcript, steps, proposed_action?}.
    """
    from autotuner.llm_gateway.client import call_llm_with_schema

    src = RealDataSource(settings)
    from .slash_commands import handle_slash_command

    if allow_slash:
        slash = handle_slash_command(message, settings, src)
        if slash is not None:
            return slash
    transcript: List[Dict[str, Any]] = []
    if preload_operator_context:
        try:
            result = _tool_get_evidence_context(
                src,
                query=_operator_question_from_message(message),
                ui_mode=_ui_mode_from_message(message),
            )
        except Exception as e:  # noqa: BLE001
            result = {"error": f"{type(e).__name__}: {e}"}
        transcript.append({
            "tool": "get_evidence_context",
            "args": {
                "query": _operator_question_from_message(message),
                "ui_mode": _ui_mode_from_message(message),
                "preloaded": True,
            },
            "result": result,
        })

    executed_actions: List[Dict[str, Any]] = []
    executed_count = 0
    for _ in range(max_steps):
        user_prompt = _render(message, transcript, history or [])
        resp = call_llm_with_schema(_SYSTEM, user_prompt, schema_name="locomotion_console_agent")
        if not resp or not resp.parsed:
            err = resp.error if resp else "no response"
            return {"reply": f"(LLM unavailable: {err})", "transcript": transcript,
                    "steps": len(transcript)}
        d = resp.parsed
        # proposed action: AUTONOMY-TIERED. As the tuning BRAIN the copilot auto-executes low/medium
        # risk actions (rollback-able edits, measurements) up to a per-turn budget; DESTRUCTIVE
        # actions (start/kill/deploy/campaign) ALWAYS surface for operator confirmation.
        pa = d.get("propose_action")
        if pa and pa.get("name") in ACTION_NAMES:
            name, args_a = pa["name"], pa.get("args") or {}
            if action_risk(name) in _autonomy_auto_tiers() and executed_count < 3:
                out = execute_action(name, args_a, src.settings)
                executed_count += 1
                record = {"action": name, "args": args_a, "result": out}
                transcript.append(record)
                executed_actions.append(record)
                continue                                   # let the brain see the result and proceed
            return {"reply": d.get("reply", f"Confirm {pa['name']}?"),
                    "proposed_action": {"name": pa["name"], "args": args_a},
                    "executed_actions": executed_actions,
                    "transcript": transcript, "steps": len(transcript)}
        if d.get("reply"):
            return {"reply": d["reply"], "executed_actions": executed_actions,
                    "transcript": transcript, "steps": len(transcript)}
        tool = d.get("tool")
        args = d.get("args") or {}
        if tool in TOOLS:
            try:
                result = TOOLS[tool](src, **args)
                if tool == "get_operator_context" and isinstance(result, dict):
                    result = _compact_operator_context_for_llm(result)
            except Exception as e:  # noqa: BLE001
                result = {"error": f"{type(e).__name__}: {e}"}
        else:
            result = {"error": f"unknown tool: {tool}"}
        transcript.append({"tool": tool, "args": args, "result": result})

    return {"reply": "(too many tool steps; stopped)", "transcript": transcript,
            "steps": len(transcript)}


_VERIFY_SYSTEM = """You are a strict fact-checker for an RL-locomotion-console assistant. You are given the
TOOL OUTPUTS the assistant actually received, and the assistant's REPLY. Decide whether every
factual claim in the REPLY (numbers, run/checkpoint names, running/stopped states, file paths)
is supported by the tool outputs.

Return STRICT JSON: {"grounded": bool, "unsupported": ["<short claim>", ...]}.

Rules:
- A claim is UNSUPPORTED if its number/name/state does not appear in, or directly follow from,
  the tool outputs (e.g. saying "checkpoint iter 0 / untrained" when the tools show agent_65000
  with real motion data).
- Reasoning explicitly labeled as inference (e.g. "possibly DR", "roughly") is allowed.
- Be precise, not pedantic: rounding (15.4 -> ~15), unit phrasing, and summarizing are fine.
- If everything checks out, return {"grounded": true, "unsupported": []}."""


def verify_grounding(reply: str, transcript: List[Dict[str, Any]],
                     settings: LocomotionConsoleSettings) -> Dict[str, Any]:
    """Cheap second pass: does the reply's factual content trace to the tool outputs?

    Returns {checked, grounded, unsupported}. Skips when no tools were used (nothing to check
    against) or when the LLM check itself fails (never blocks the answer)."""
    if not transcript:
        return {"checked": False, "grounded": None, "unsupported": []}
    from autotuner.llm_gateway.client import call_llm_with_schema
    tool_text = "\n".join(
        f"{t['tool']}({json.dumps(t['args'], ensure_ascii=False)}): "
        f"{json.dumps(t['result'], ensure_ascii=False)[:1500]}" for t in transcript)
    user = f"TOOL OUTPUTS:\n{tool_text}\n\nASSISTANT REPLY:\n{reply}"
    resp = call_llm_with_schema(_VERIFY_SYSTEM, user, schema_name="grounding_check")
    if not resp or not resp.parsed:
        return {"checked": False, "grounded": None, "unsupported": []}
    d = resp.parsed
    return {"checked": True, "grounded": bool(d.get("grounded", True)),
            "unsupported": d.get("unsupported", []) or []}


def execute_action(name: str, args: Dict[str, Any], settings: LocomotionConsoleSettings) -> Dict[str, Any]:
    """Run a confirmed action. Called ONLY after the operator confirms - never from the agent
    loop. Maps to the RealDataSource action methods (kill is real; physeval/resume report
    honestly if their command isn't discovered into the profile yet)."""
    import asyncio

    if name not in ACTION_NAMES:
        return {"ok": False, "detail": f"unknown action: {name}"}
    src = RealDataSource(settings)

    async def _run():
        if name == "run_diagnostic":
            from .diagnostics import DiagnosticsController

            controller = DiagnosticsController(settings, src)
            status = await controller.start(
                preset_id=str(args.get("preset") or "quick"),
                requested_checkpoint=str(args.get("checkpoint") or "") or None,
            )
            return {"ok": bool(status.job_id), "detail": f"{status.preset_label or status.preset}: {status.message}"}
        if name == "kill_training":
            r = await src.action_kill()
            return {"ok": r.ok, "detail": r.message}
        if name == "deploy_payload":
            r = await src.action_deploy_payload()
            return {"ok": r.ok, "detail": r.message}
        if name == "start_training":
            r = await src.action_start()
            return {"ok": r.ok, "detail": r.message}
        if name == "resume_training":
            r = await src.action_resume()
            return {"ok": r.ok, "detail": r.message}
        if name == "run_physeval":
            r = await src.action_physeval(run=str(args.get("run", "")))
            return {"ok": r.ok, "detail": r.summary}
        if name == "run_acceptance":
            r = await src.action_run_acceptance(
                run=str(args.get("run", "")),
                terrains=str(args.get("terrains", "flat")),
                checkpoint=str(args.get("checkpoint", "best_agent.pt")),
            )
            return {"ok": r.ok, "detail": r.message}
        if name == "edit_config":
            r = await src.action_edit_config(key=str(args.get("key", "")),
                                             value=str(args.get("value", "")))
            return {"ok": r.ok, "detail": r.message}
        if name == "rollback_config":
            r = await src.action_rollback_config()
            return {"ok": r.ok, "detail": r.message}
        if name == "apply_tuning":
            # edit the LOCAL strategy contract (reward weights / curriculum gates) with allowlist +
            # bounds + a rollback-stack push; deploy_payload then ships it to the box. This is the
            # copilot's own "tune" step of the measure->tune->retrain loop.
            from autotuner.training.strategy_edit import apply_weight_changes
            changes = args.get("changes") if isinstance(args.get("changes"), dict) else {}
            res = await asyncio.to_thread(apply_weight_changes, changes, note=str(args.get("note", "")))
            if res["ok"]:
                diffs = ", ".join(f"{a['key']} {a['old']}->{a['new']}" for a in res["applied"])
                return {"ok": True, "detail": f"applied [{res['rollback_id']}]: {diffs or 'no-op'}. "
                        f"Deploy + resume to train with it; rollback_tuning to undo."}
            return {"ok": False, "detail": "; ".join(res.get("errors") or ["apply failed"])}
        if name == "produce_policy":
            r = await src.action_produce_policy(max_iters=int(args.get("max_iters", 8)))
            return {"ok": r.ok, "detail": r.message}
        if name == "run_campaign":
            r = await src.action_run_campaign(
                run=str(args.get("run", "")),
                checkpoint=str(args.get("checkpoint", "best_agent.pt")),
                max_iters=int(args.get("max_iters", 4)),
            )
            return {"ok": r.ok, "detail": r.message}
        if name == "rollback_tuning":
            from autotuner.training.strategy_edit import rollback_last
            res = await asyncio.to_thread(rollback_last)
            if res["ok"]:
                diffs = ", ".join(f"{a['key']}->{a['new']}" for a in res.get("restored", []))
                return {"ok": True, "detail": f"rolled back {res['rolled_back']}: restored {diffs}"}
            return {"ok": False, "detail": "; ".join(res.get("errors") or ["rollback failed"])}
        return {"ok": False, "detail": "unreachable"}

    return asyncio.run(_run())


def _render(message: str, transcript: List[Dict[str, Any]],
            history: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    if history:
        parts.append("Conversation so far (for context - resolve pronouns/references from it):")
        for h in history[-8:]:
            parts.append(f"  {h.get('role')}: {str(h.get('content',''))[:500]}")
        parts.append("")
    parts.append(f"Operator now says: {message}")
    parts.append("")
    if transcript:
        parts.append("Tool results gathered this turn:")
        for i, t in enumerate(transcript, 1):
            rendered = json.dumps(t['result'], ensure_ascii=False)
            budget = _tool_result_budget(str(t.get("tool") or ""))
            parts.append(f"[{i}] {t['tool']}({json.dumps(t['args'], ensure_ascii=False)}) -> "
                         f"{rendered[:budget]}")
    else:
        parts.append("(no tools called yet this turn - call one to gather facts before answering)")
    return "\n".join(parts)


def _tool_result_budget(tool: str) -> int:
    if tool == "get_evidence_context":
        return 9000
    if tool in {"get_operator_context", "get_diagnostic_report"}:
        return 5200
    if tool == "get_training_telemetry":
        return 4200
    return 1800


def _ui_mode_from_message(message: str) -> str:
    match = re.search(r"UI mode:\s*([A-Za-z0-9_-]+)", message or "", flags=re.IGNORECASE)
    return match.group(1).lower() if match else ""


def _operator_question_from_message(message: str) -> str:
    text = message or ""
    marker = "Operator now says:"
    if marker in text:
        return text.rsplit(marker, 1)[-1].strip()
    return text.strip()
