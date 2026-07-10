from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import DiagnosticThresholds
from .events import add_derived_event_columns
from .util import bool_series, group_key_columns, numeric_col, load_record, write_json


def build_diagnostic_notes(df: pd.DataFrame, timeline: list[dict[str, Any]] | None = None, segments: list[dict[str, Any]] | None = None, record_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build prioritized observation notes.

    Notes are not pass/fail judgments. They are reading-priority hints for
    unusual, concentrated, or coverage-limiting observations.
    """
    thresholds = DiagnosticThresholds()
    d = add_derived_event_columns(df, thresholds)
    timeline = timeline or []
    segments = segments or []
    record_meta = record_meta or {}
    push_requested = _push_requested(record_meta)

    data_quality = []
    physical_events = []
    behavior_patterns = []

    # Data quality / coverage notes.
    if len(d) == 0:
        data_quality.append(_note("high", "empty_record", "record contains no rows"))
    if "time" in d.columns:
        t = numeric_col(d, "time", np.nan)
        keys = group_key_columns(d, include_segment=False)
        dt = t.groupby([d[c] for c in keys], sort=False).diff().dropna() if keys else t.diff().dropna()
        if len(dt) and (dt < -1e-9).any():
            data_quality.append(_note("high", "non_monotonic_time", "time decreases within an env/episode group"))
        if len(dt) and dt.std() > max(1e-6, 0.2 * abs(dt.median())):
            data_quality.append(_note("medium", "variable_dt_observed", f"dt variability observed: median={dt.median():.4g}, std={dt.std():.4g}"))
    field_groups = {
        "terrain_height": ["foot_FL_terrain_height", "foot_FL_clearance_local", "terrain_height_source"],
        "torque": ["torque_applied_0", "torque_limit_0", "torque_utilization_0"],
        "dr": ["dr_level", "dr_friction", "dr_mass"],
        "push": ["push_event", "push_equivalent_delta_v"],
    }
    if not push_requested:
        field_groups.pop("push")
    for group, cols in field_groups.items():
        missing = [c for c in cols if c not in d.columns]
        empty = [c for c in cols if c in d.columns and not _column_has_data(d[c])]
        if missing or empty:
            data_quality.append(_note(
                "medium",
                f"missing_{group}_fields",
                f"missing fields: {missing}; present but empty: {empty}",
            ))
    if "terrain_height_source" in d.columns:
        sources = sorted([str(x) for x in d["terrain_height_source"].dropna().unique()])
        if any(s in ["env_origin_fallback", "unavailable"] for s in sources):
            data_quality.append(_note("high", "terrain_height_approximate_or_unavailable", f"terrain_height_source includes {sources}; local clearance may be unreliable on non-flat terrain"))
    if all(c in d.columns for c in ["terrain_type_requested", "terrain_type"]):
        mismatch = d["terrain_type_requested"].astype(str) != d["terrain_type"].astype(str)
        aliases = {
            "plane": "flat", "slope_up": "slope", "slope_down": "slope_inv",
            "stairs_up": "stairs", "stairs_down": "stairs",
        }
        requested = d["terrain_type_requested"].astype(str).map(lambda x: aliases.get(x, x))
        mismatch = requested != d["terrain_type"].astype(str)
        if mismatch.any():
            data_quality.append(_note(
                "high",
                "terrain_request_execution_mismatch",
                f"{int(mismatch.sum())} rows have requested terrain different from observed terrain",
            ))
    if all(c in d.columns for c in ["dr_level_requested", "dr_level"]):
        mismatch = d["dr_level_requested"].astype(str) != d["dr_level"].astype(str)
        if mismatch.any():
            data_quality.append(_note(
                "high",
                "dr_request_execution_mismatch",
                f"{int(mismatch.sum())} rows have requested DR level different from observed DR level",
            ))
    if "done_observed" in d.columns and "terminal_state_available" in d.columns:
        done = bool_series(d, "done_observed")
        unavailable = done & ~bool_series(d, "terminal_state_available")
        if unavailable.any():
            data_quality.append(_note(
                "high",
                "terminal_state_unavailable",
                f"{int(unavailable.sum())} done transitions lack a true pre-reset terminal state",
            ))

    # Coverage notes.
    terrain_types = sorted([str(x) for x in d.get("terrain_type", pd.Series([], dtype=str)).dropna().unique()])
    if terrain_types and set(terrain_types).issubset({"flat", "plane", "unknown"}):
        data_quality.append(_note("medium", "terrain_coverage_flat_only", f"observed terrain types: {terrain_types}"))
    dr_levels = sorted([str(x) for x in d.get("dr_level", pd.Series([], dtype=str)).dropna().unique()])
    if dr_levels and set(dr_levels).issubset({"0", "0.0", "none", "unknown"}):
        data_quality.append(_note("medium", "dr_coverage_nominal_only", f"observed DR levels: {dr_levels}"))
    if push_requested and not (bool_series(d, "push_event").any() or bool_series(d, "push_observed").any()):
        data_quality.append(_note("low", "no_push_observed", "no push frames/events observed in this record"))

    # Physical event notes from bouts.
    for typ in ["large_pitch_bout", "large_roll_bout", "height_drop_bout", "torque_clamp_bout", "hard_impact_bout", "high_slip_bout"]:
        bouts = [e for e in timeline if e.get("event_type") == typ]
        if bouts:
            top = sorted(bouts, key=lambda e: (e.get("duration_s") or 0, e.get("frame_count") or 0), reverse=True)[0]
            physical_events.append(_note("high" if typ in ["large_pitch_bout", "height_drop_bout", "torque_clamp_bout"] else "medium", typ, _bout_message(top)))
    reset_events = [e for e in timeline if e.get("event_type") in ["reset_observed", "done_observed", "terminated_observed", "truncated_observed"]]
    if reset_events:
        e = reset_events[0]
        physical_events.append(_note("high", "reset_or_done_observed", f"{len(reset_events)} reset/done-like observations; first at t={e.get('time')} segment={e.get('cmd_segment_id')} mode={e.get('cmd_target_mode')}"))

    # Segment patterns.
    for s in sorted(segments, key=lambda x: (x.get("first_major_event_time_since_segment_start_s") is None, x.get("first_major_event_time_since_segment_start_s") or 9999))[:10]:
        if s.get("first_major_event_time_since_segment_start_s") is not None:
            behavior_patterns.append(_note("high", "early_major_event_in_segment", f"mode={s.get('cmd_target_mode')} terrain={s.get('terrain_type')} segment={s.get('cmd_segment_id')} first_major_event={s.get('first_major_event_time_since_segment_start_s'):.3f}s events={s.get('observed_event_labels')} raw_progress={_fmt(s.get('raw_progress_along_command'))} stable_pre_event_progress={_fmt(s.get('pose_stable_pre_first_major_event_progress_along_command'))}"))
    # Raw vs stable progress gap.
    for s in segments:
        raw = _num(s.get("raw_progress_along_command"))
        st = _num(s.get("pose_stable_pre_first_major_event_progress_along_command"))
        if raw is not None and st is not None and abs(raw) > 0.2 and abs(st) < 0.25 * abs(raw):
            behavior_patterns.append(_note("medium", "raw_vs_stable_progress_gap", f"mode={s.get('cmd_target_mode')} segment={s.get('cmd_segment_id')} raw_progress={raw:.3f}, stable_pre_event_progress={st:.3f}"))
    # Per-leg duty imbalance.
    duty = {}
    for leg in ["FL", "FR", "RL", "RR"]:
        c = f"foot_{leg}_contact"
        if c in d.columns:
            duty[leg] = float(bool_series(d, c).mean())
    if len(duty) == 4 and max(duty.values()) - min(duty.values()) > 0.25:
        behavior_patterns.append(_note("medium", "per_leg_duty_spread", f"per-leg duty spread={max(duty.values()) - min(duty.values()):.3f}, duties={duty}"))

    top = sorted(data_quality + physical_events + behavior_patterns, key=lambda n: {"high":0,"medium":1,"low":2}.get(n["priority"], 3))
    return {
        "data_quality_notices": data_quality,
        "physical_event_notices": physical_events,
        "behavior_pattern_notices": behavior_patterns,
        "top_observed_patterns": top[:20],
    }


def _push_requested(record_meta: dict[str, Any]) -> bool:
    suite = record_meta.get("requested_suite_config", record_meta.get("suite", {}))
    value = suite.get("pushes") if isinstance(suite, dict) else None
    if not value:
        return False
    if isinstance(value, dict):
        if value.get("enabled") in [False, "false", "disabled"]:
            return False
        if "events" in value:
            return bool(value["events"])
        if "cases" in value:
            return bool(value["cases"])
    return True


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _column_has_data(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").notna().any()
    cleaned = series.dropna().astype(str).str.strip()
    return bool((cleaned != "").any())


def _fmt(x):
    v = _num(x)
    return "n/a" if v is None else f"{v:.3f}"


def _note(priority: str, typ: str, message: str, related: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"priority": priority, "type": typ, "message": message, "related_metrics": related or {}}


def _bout_message(e: dict[str, Any]) -> str:
    return f"{e.get('event_type')} observed in mode={e.get('cmd_target_mode')} terrain={e.get('terrain_type')} segment={e.get('cmd_segment_id')} start={e.get('start_time')} duration={e.get('duration_s')} frames={e.get('frame_count')} peak={e.get('peak_value')}"


def main(argv: list[str] | None = None) -> None:
    from .events import extract_event_timeline, segment_event_summary
    p = argparse.ArgumentParser(description="Generate observation notes from a record.")
    p.add_argument("record")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    df = load_record(args.record)
    th = DiagnosticThresholds()
    timeline = extract_event_timeline(df, th)
    segs = segment_event_summary(df, timeline, th)
    notes = build_diagnostic_notes(df, timeline, segs)
    if args.out:
        write_json(args.out, notes)
    else:
        print(json.dumps(notes, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
