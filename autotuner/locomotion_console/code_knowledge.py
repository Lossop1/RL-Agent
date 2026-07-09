"""Allowlisted code knowledge for the locomotion-console LLM.

This is not an arbitrary grep endpoint.  It exposes a small, auditable index over
the Taili training/runtime files that matter for tuning: reward, gates,
telemetry, diagnostics, YAML loading, model/observation, and payload wiring.
The goal is to let the LLM answer "is this field really wired?" questions
without granting free filesystem or shell access.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_ROOT = Path(__file__).resolve().parents[2]

_ALLOWLIST_FILES: tuple[str, ...] = (
    "autotuner/blind_locomotion/taili_blind_config.yaml",
    "autotuner/blind_locomotion/taili_blind_config.py",
    "autotuner/blind_locomotion/blind_tp_env.py",
    "autotuner/blind_locomotion/telemetry_emit.py",
    "autotuner/blind_locomotion/train_taili.py",
    "autotuner/blind_locomotion/diagnose_taili_cases.py",
    "autotuner/blind_locomotion/diagnose_taili.py",
    "autotuner/blind_locomotion/acceptance_score.py",
    "autotuner/blind_locomotion/reward_tuning_advisor.py",
    "autotuner/blind_locomotion/terrain_perceiver_policy.py",
    "autotuner/blind_locomotion/terrain_perceiver_aux_patch.py",
    "autotuner/blind_locomotion/parametric_ref.py",
    "autotuner/blind_locomotion/assets/taili.py",
    "autotuner/taili_core/taili_reward.py",
    "autotuner/taili_core/taili_curriculum.py",
    "autotuner/taili_core/taili_obs.py",
    "autotuner/taili_core/taili_models.py",
    "autotuner/taili_core/taili_amp_reference.py",
    "autotuner/locomotion_console/telemetry.py",
    "autotuner/locomotion_console/definitions.py",
    "autotuner/locomotion_console/diagnostics.py",
    "autotuner/locomotion_console/spec_coverage.py",
    "autotuner/training_payloads/taili_blind_runtime/payload_manifest.py",
)


_SIGNAL_ROWS: list[dict[str, Any]] = [
    {
        "id": "a1_linear_tracking_progress",
        "concept": "linear velocity tracking and phase progress",
        "spec_items": ["A1", "A5"],
        "user_terms": ["前进", "后退", "progress", "progress_gate", "tracking", "lin_err", "actual_vx", "v_along"],
        "yaml_keys": ["sigma_lin_abs", "sigma_lin_rel", "w_tracking_lin", "w_track_far", "scale_track_far", "terrain_v_floor"],
        "code_refs": [
            "autotuner/blind_locomotion/blind_tp_env.py",
            "autotuner/taili_core/taili_reward.py",
            "autotuner/taili_core/taili_curriculum.py",
            "autotuner/locomotion_console/telemetry.py",
        ],
        "telemetry_fields": ["reward.tracking_lin", "reward.tracking_lin_far", "command.lin_err", "command.actual_vx", "command.v_along", "curriculum.progress_gate"],
        "diagnostic_fields": ["commands.tracking_error", "commands.controlled_progress", "commands.raw_progress"],
        "notes": "Telemetry progress is a training proxy. Use diagnostics to confirm commanded movement.",
    },
    {
        "id": "a2_yaw_tracking",
        "concept": "yaw command tracking",
        "spec_items": ["A2"],
        "user_terms": ["yaw", "偏航", "转向", "cmd_wz", "actual_wz", "tracking_yaw"],
        "yaml_keys": ["sigma_yaw", "w_tracking_yaw", "w_yaw_far"],
        "code_refs": [
            "autotuner/blind_locomotion/blind_tp_env.py",
            "autotuner/blind_locomotion/diagnose_taili_cases.py",
            "autotuner/locomotion_console/diagnostics.py",
        ],
        "telemetry_fields": ["reward.tracking_yaw", "reward.tracking_yaw_far", "command.cmd_wz", "command.actual_wz"],
        "diagnostic_fields": ["commands.mode=yaw", "commands.tracking_error"],
        "notes": "Yaw reward should be conditioned on nonzero yaw command; otherwise standing can receive a yaw bonus.",
    },
    {
        "id": "a3_stand_settle",
        "concept": "standing, stop, and settle behavior",
        "spec_items": ["A3", "C"],
        "user_terms": ["stand", "standing", "停", "立正", "settle", "recovery", "off_axis"],
        "yaml_keys": ["w_stand", "w_stand_far", "scale_stand_far", "sigma_stand_speed", "sigma_stand_yaw", "sigma_stand_pose", "sigma_stand_action", "w_off_axis"],
        "code_refs": [
            "autotuner/blind_locomotion/blind_tp_env.py",
            "autotuner/blind_locomotion/diagnose_taili_cases.py",
            "autotuner/locomotion_console/spec_coverage.py",
        ],
        "telemetry_fields": ["reward.stand", "reward.stand_far", "reward.off_axis", "curriculum.stand_gate", "health.base_h", "health.tilt_deg"],
        "diagnostic_fields": ["commands.mode=stand", "posture", "events.reset"],
        "notes": "Standing success is separate from walking success; do not use stand stability to claim gait quality.",
    },
    {
        "id": "b2_stance_slip",
        "concept": "settled stance slip",
        "spec_items": ["B2"],
        "user_terms": ["slip", "打滑", "滑", "stance_slip", "settled", "high_slip"],
        "yaml_keys": ["w_stance_slip", "w_stance_slip_late", "w_stance_slip_event", "slip_excess_cap", "slip_high_threshold", "phase_gate_slip_1"],
        "code_refs": [
            "autotuner/blind_locomotion/blind_tp_env.py",
            "autotuner/blind_locomotion/diagnose_taili_cases.py",
            "autotuner/locomotion_console/definitions.py",
            "autotuner/locomotion_console/spec_coverage.py",
        ],
        "telemetry_fields": ["reward.stance_slip", "reward.stance_slip_event", "command.stance_slip", "command.stance_slip_high_fraction"],
        "diagnostic_fields": ["events.high_slip_bout", "metrics.stance_slip", "behavior_context.slip"],
        "notes": "Settled-contact slip should exclude touchdown/liftoff transition frames.",
    },
    {
        "id": "b1_landing_impact",
        "concept": "touchdown impact and hard landing",
        "spec_items": ["B1"],
        "user_terms": ["impact", "冲击", "砸地", "landing", "touchdown", "hard_impact"],
        "yaml_keys": ["w_landing_impact", "w_landing_impact_late", "impact_excess_cap"],
        "code_refs": [
            "autotuner/blind_locomotion/blind_tp_env.py",
            "autotuner/blind_locomotion/diagnose_taili_cases.py",
            "autotuner/locomotion_console/spec_coverage.py",
        ],
        "telemetry_fields": ["reward.landing_impact", "command.touchdown_observed"],
        "diagnostic_fields": ["events.hard_impact_bout", "value_explanations.impact"],
        "notes": "Impact should be touchdown-only; continuously penalizing contact can create standing/walking traps.",
    },
    {
        "id": "b3_clearance",
        "concept": "foot clearance and obstacle-relative clearance",
        "spec_items": ["B3", "D"],
        "user_terms": ["clearance", "抬腿", "拖脚", "绊脚", "local_obstacle_h", "terrain clearance"],
        "yaml_keys": ["w_clearance_under", "w_clearance_over", "clearance_target", "clearance_margin", "clearance_gate"],
        "code_refs": [
            "autotuner/blind_locomotion/blind_tp_env.py",
            "autotuner/taili_core/taili_obs.py",
            "autotuner/locomotion_console/spec_coverage.py",
        ],
        "telemetry_fields": ["reward.clearance_under", "reward.clearance_over", "curriculum.clearance_gate"],
        "diagnostic_fields": ["commands.foot_clearance", "terrain.local_obstacle_h"],
        "notes": "Flat clearance and terrain-relative clearance are different; obstacle height wiring matters.",
    },
    {
        "id": "b4_gait_quality",
        "concept": "trot gait quality: diagonal pair, duty balance, air time",
        "spec_items": ["B4"],
        "user_terms": ["gait", "步态", "trot", "小跑", "diagonal", "duty", "air_time", "crawl", "拖行"],
        "yaml_keys": ["w_gait_anchor", "w_feet_air_time", "air_time_target", "w_diagonal_contact", "w_duty_balance", "phase_gate_diag_1", "phase_gate_duty_1", "phase_gate_air_1"],
        "code_refs": [
            "autotuner/blind_locomotion/blind_tp_env.py",
            "autotuner/taili_core/taili_amp_reference.py",
            "autotuner/locomotion_console/definitions.py",
        ],
        "telemetry_fields": ["command.gait_match", "command.diagonal_contact", "command.diagonal_pair_instant", "command.duty_balance", "command.duty_spread_window", "reward.gait_anchor", "reward.feet_air_time"],
        "diagnostic_fields": ["duties", "per-leg duty spread", "diagonal contact consistency"],
        "notes": "Gait proxies can be misleading if they reward contact patterns that do not produce controlled movement.",
    },
    {
        "id": "height_posture_gate",
        "concept": "single height/posture gate and termination",
        "spec_items": ["D4", "C"],
        "user_terms": ["height", "高度", "base_h", "机身高度", "蹲", "塌", "tilt", "upright", "termination"],
        "yaml_keys": ["nominal_base_h", "h_ok", "h_gate_close", "termination_height", "tilt_ok_rad", "tilt_gate_close_rad"],
        "code_refs": [
            "autotuner/blind_locomotion/blind_tp_env.py",
            "autotuner/blind_locomotion/taili_blind_config.py",
            "autotuner/locomotion_console/definitions.py",
        ],
        "telemetry_fields": ["health.base_h", "health.height_low_risk_window", "health.tilt_deg", "health.tilt_high_risk_window", "health.upright", "curriculum.stable_motion_gate"],
        "diagnostic_fields": ["posture.base_height_local.min", "posture.max_roll", "posture.max_pitch"],
        "notes": "Height is a gate/termination mechanism; avoid duplicate reward penalties unless the code explicitly consumes them.",
    },
    {
        "id": "terrain_dr_curriculum",
        "concept": "terrain curriculum and domain randomization",
        "spec_items": ["D1-D5", "E3", "E4"],
        "user_terms": ["terrain", "地形", "stairs", "slope", "rough", "boxes", "dr_level", "随机化", "friction", "mass"],
        "yaml_keys": ["terrain_start_phase", "dr_start_phase", "terrain_mean", "terrain_max", "friction_range_3", "mass_range"],
        "code_refs": [
            "autotuner/blind_locomotion/blind_tp_env.py",
            "autotuner/taili_core/taili_terrain_labels.py",
            "autotuner/locomotion_console/spec_coverage.py",
        ],
        "telemetry_fields": ["curriculum.terrain_mean", "curriculum.terrain_max", "curriculum.dr_level"],
        "diagnostic_fields": ["terrain", "robustness", "coverage.terrain"],
        "notes": "Terrain success needs terrain-specific diagnostics; flat telemetry is not enough.",
    },
    {
        "id": "model_observation_perceiver_amp",
        "concept": "actor/critic observation contract, terrain perceiver, AMP style",
        "spec_items": ["architecture"],
        "user_terms": ["actor", "critic", "网络", "observation", "观察", "history", "perceiver", "感知器", "amp", "style", "discriminator"],
        "yaml_keys": ["actor_hidden", "critic_hidden", "history_len", "observation_space", "privileged_dim", "terrain_perceiver_aux", "style_reward_weight", "discriminator_reward_scale"],
        "code_refs": [
            "autotuner/blind_locomotion/terrain_perceiver_policy.py",
            "autotuner/blind_locomotion/terrain_perceiver_aux_patch.py",
            "autotuner/taili_core/taili_models.py",
            "autotuner/taili_core/taili_obs.py",
            "autotuner/taili_core/taili_amp_reference.py",
        ],
        "telemetry_fields": ["reward.amp_style"],
        "diagnostic_fields": ["not directly measured; infer via behavior and style reward only"],
        "notes": "Architecture contract should be checked against code, not inferred from YAML alone.",
    },
    {
        "id": "actuator_torque_asset",
        "concept": "actuator gains, torque limits, saturation, and asset facts",
        "spec_items": ["B5", "F2"],
        "user_terms": ["actuator", "执行器", "kp", "kd", "stiffness", "damping", "torque", "力矩", "saturation", "asset", "urdf"],
        "yaml_keys": ["stiffness", "damping", "w_torque_margin", "w_torque_saturation", "torque_limit"],
        "code_refs": [
            "autotuner/blind_locomotion/assets/taili.py",
            "autotuner/blind_locomotion/blind_tp_env.py",
            "autotuner/locomotion_console/agent.py",
        ],
        "telemetry_fields": ["health.torque_util", "reward.torque_margin", "reward.torque_saturation"],
        "diagnostic_fields": ["torque health if diagnostic exports it"],
        "notes": "Remote asset snippets are live facts; YAML actuator values need code consumption checks.",
    },
    {
        "id": "diagnostic_runtime_provenance",
        "concept": "diagnostic runtime, payload, reports, and provenance",
        "spec_items": ["evaluation"],
        "user_terms": ["diagnostic", "诊断", "report", "payload", "provenance", "诊断失败", "回放", "checkpoint"],
        "yaml_keys": ["diagnostic_task"],
        "code_refs": [
            "autotuner/locomotion_console/diagnostics.py",
            "autotuner/blind_locomotion/diagnose_taili_cases.py",
            "autotuner/training_payloads/taili_blind_runtime/payload_manifest.py",
        ],
        "telemetry_fields": ["snapshot.latest_checkpoint", "paths.checkpoint_dir"],
        "diagnostic_fields": ["diagnostic_history", "diagnostics.report", "record_error.json"],
        "notes": "Use diagnostic reports for observed behavior and payload provenance for runtime failures.",
    },
    {
        "id": "telemetry_snapshot_ui",
        "concept": "RunSnapshot, telemetry parsing, dashboard metric source",
        "spec_items": ["system"],
        "user_terms": ["telemetry", "jsonl", "run_snapshot", "显示", "监控", "来源", "dashboard", "字段"],
        "yaml_keys": ["telemetry_interval"],
        "code_refs": [
            "autotuner/locomotion_console/telemetry.py",
            "autotuner/locomotion_console/definitions.py",
            "autotuner/blind_locomotion/telemetry_emit.py",
        ],
        "telemetry_fields": ["TrainingTelemetry.snapshot", "DataProvenanceItem", "MetricSnapshotGroup"],
        "diagnostic_fields": ["not diagnostic; compare separately"],
        "notes": "Dashboard values need provenance and definitions; they are not a substitute for diagnostic results.",
    },
]


_QUERY_ALIASES = {
    "走": "progress tracking actual_vx v_along gait",
    "不走": "progress tracking actual_vx v_along",
    "前进": "forward progress tracking actual_vx",
    "后退": "backward progress tracking",
    "步态": "gait trot diagonal duty air_time stance_slip",
    "打滑": "slip stance_slip high_slip settled",
    "滑": "slip stance_slip high_slip",
    "高度": "height base_h h_ok termination tilt",
    "配置": "yaml config reward curriculum",
    "调参": "reward yaml telemetry diagnostic spec",
    "诊断": "diagnostic report behavior_context",
    "感知器": "perceiver privileged observation",
    "网络": "actor critic model observation",
    "显存": "remote gpu",
    "来源": "provenance telemetry jsonl paths",
    "消费": "yaml_keys code_refs consumed",
    "死键": "yaml_keys consumed code_refs",
}


def build_signal_map(query: str = "", limit: int = 12) -> dict[str, Any]:
    terms = _query_terms(query)
    rows = _rank_signal_rows(terms)
    selected = rows if terms else [(1, row) for row in _SIGNAL_ROWS]
    return {
        "kind": "taili_signal_map",
        "permission_boundary": _permission_boundary(),
        "query": query,
        "rows": [row for _score, row in selected[: max(1, min(limit, 50))]],
        "total_rows": len(_SIGNAL_ROWS),
        "matched_rows": len(rows) if terms else len(_SIGNAL_ROWS),
        "limitations": [
            "This is a curated mapping, not a substitute for tests.",
            "Code references are allowlisted source locations; they do not grant arbitrary repository search.",
            "A key occurrence proves wiring evidence, not necessarily correct semantics.",
        ],
    }


def search_code_knowledge(query: str = "", max_snippets: int = 16) -> dict[str, Any]:
    terms = _query_terms(query)
    ranked_rows = _rank_signal_rows(terms)
    matched_rows = [row for _score, row in ranked_rows[:8]]
    if not terms:
        matched_rows = _SIGNAL_ROWS[:8]
    expanded_terms = _expand_terms(terms, matched_rows)
    files = _files_for_rows(matched_rows) or list(_ALLOWLIST_FILES)

    snippets: list[dict[str, Any]] = []
    for rel in files:
        text = _read_allowlisted(rel)
        if text is None:
            continue
        snippets.extend(_snippets_for_text(rel, text, expanded_terms, max_snippets=max_snippets - len(snippets)))
        if len(snippets) >= max_snippets:
            break

    consumption = _consumption_checks(matched_rows, files)
    return {
        "kind": "taili_code_knowledge",
        "permission_boundary": _permission_boundary(),
        "query": query,
        "matched_signal_rows": matched_rows,
        "terms_used": expanded_terms[:40],
        "files_considered": files,
        "snippets": snippets[:max_snippets],
        "consumption_checks": consumption,
        "gaps": _code_knowledge_gaps(matched_rows, snippets),
        "limitations": [
            "Snippets come from a fixed local allowlist only.",
            "This is literal source evidence, not a full static call graph.",
            "If a question requires proving runtime behavior, pair this with telemetry and diagnostics.",
        ],
    }


def source_inventory() -> dict[str, Any]:
    return {
        "kind": "taili_code_source_inventory",
        "permission_boundary": _permission_boundary(),
        "files": [
            {"path": rel, "available": (_ROOT / rel).is_file()}
            for rel in _ALLOWLIST_FILES
        ],
        "signal_rows": [
            {"id": row["id"], "concept": row["concept"], "spec_items": row["spec_items"]}
            for row in _SIGNAL_ROWS
        ],
    }


def _permission_boundary() -> str:
    return (
        "Local read-only allowlist over Taili training/runtime/source files. "
        "No arbitrary filesystem scan, no shell, no remote access, no writes."
    )


def _query_terms(query: str) -> list[str]:
    text = (query or "").lower()
    expanded = text
    for needle, alias in _QUERY_ALIASES.items():
        if needle.lower() in text:
            expanded += " " + alias
    terms = [
        token
        for token in re.split(r"[^0-9a-zA-Z_\u4e00-\u9fff]+", expanded)
        if len(token) >= 2
    ]
    return list(dict.fromkeys(terms))


def _rank_signal_rows(terms: list[str]) -> list[tuple[int, dict[str, Any]]]:
    if not terms:
        return []
    ranked: list[tuple[int, dict[str, Any]]] = []
    for row in _SIGNAL_ROWS:
        haystack = json.dumps(row, ensure_ascii=False).lower()
        score = sum(1 for term in terms if term.lower() in haystack)
        if score:
            ranked.append((score, row))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def _expand_terms(terms: list[str], rows: list[dict[str, Any]]) -> list[str]:
    expanded = list(terms)
    for row in rows:
        for field in ("yaml_keys", "telemetry_fields", "diagnostic_fields", "user_terms"):
            for value in row.get(field) or []:
                for part in re.split(r"[^0-9a-zA-Z_]+", str(value)):
                    if len(part) >= 3:
                        expanded.append(part.lower())
    return list(dict.fromkeys(expanded))


def _files_for_rows(rows: list[dict[str, Any]]) -> list[str]:
    files: list[str] = []
    for row in rows:
        for ref in row.get("code_refs") or []:
            rel = str(ref).split("::", 1)[0]
            if rel in _ALLOWLIST_FILES and rel not in files:
                files.append(rel)
    return files


def _read_allowlisted(rel: str) -> str | None:
    if rel not in _ALLOWLIST_FILES:
        return None
    path = _ROOT / rel
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _snippets_for_text(rel: str, text: str, terms: list[str], *, max_snippets: int) -> list[dict[str, Any]]:
    if max_snippets <= 0:
        return []
    lines = text.splitlines()
    lower_terms = [term.lower() for term in terms if len(term) >= 3]
    out: list[dict[str, Any]] = []
    used_windows: set[tuple[int, int]] = set()
    for idx, line in enumerate(lines):
        lower = line.lower()
        matched = [term for term in lower_terms if term in lower]
        if not matched:
            continue
        start = max(0, idx - 3)
        end = min(len(lines), idx + 4)
        window = (start, end)
        if window in used_windows:
            continue
        used_windows.add(window)
        out.append({
            "path": rel,
            "line_start": start + 1,
            "line_end": end,
            "matched_terms": matched[:8],
            "text": "\n".join(f"{line_no + 1}: {lines[line_no]}" for line_no in range(start, end))[:2400],
        })
        if len(out) >= max_snippets:
            break
    return out


def _consumption_checks(rows: list[dict[str, Any]], files: list[str]) -> list[dict[str, Any]]:
    keys: list[str] = []
    for row in rows:
        for key in row.get("yaml_keys") or []:
            if key not in keys:
                keys.append(str(key))
    checks: list[dict[str, Any]] = []
    code_files = [rel for rel in files if not rel.endswith(".yaml")]
    for key in keys[:40]:
        occurrences: list[dict[str, Any]] = []
        for rel in code_files:
            text = _read_allowlisted(rel)
            if not text:
                continue
            for idx, line in enumerate(text.splitlines()):
                if key in line:
                    occurrences.append({"path": rel, "line": idx + 1, "preview": line.strip()[:220]})
                    if len(occurrences) >= 8:
                        break
            if len(occurrences) >= 8:
                break
        checks.append({
            "key": key,
            "status": "has_code_occurrence" if occurrences else "not_found_in_allowlisted_code",
            "occurrences": occurrences,
            "caveat": "Literal occurrence is wiring evidence; inspect snippets before claiming semantic correctness.",
        })
    return checks


def _code_knowledge_gaps(rows: list[dict[str, Any]], snippets: list[dict[str, Any]]) -> list[dict[str, str]]:
    gaps: list[dict[str, str]] = []
    if not rows:
        gaps.append({
            "gap": "no signal-map row matched the query",
            "impact": "code snippets may be incomplete; ask a more specific field or concept",
        })
    if rows and not snippets:
        gaps.append({
            "gap": "matched signal rows but no literal snippets were found in allowlisted files",
            "impact": "do not claim a code-level implementation detail without further inspection",
        })
    return gaps
