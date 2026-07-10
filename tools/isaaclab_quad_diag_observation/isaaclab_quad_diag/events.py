from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from .schema import DiagnosticThresholds, LEGS
from .util import (
    bool_series,
    command_components,
    command_segment_key_columns,
    group_key_columns,
    numeric_col,
    stats,
)


MAJOR_EVENT_COLUMNS = [
    "reset_observed",
    "done_observed",
    "large_roll_observed",
    "large_pitch_observed",
    "height_drop_observed",
]

EVENT_BOOL_COLUMNS = [
    "done_observed",
    "reset_observed",
    "terminated_observed",
    "truncated_observed",
    "large_roll_observed",
    "large_pitch_observed",
    "height_drop_observed",
    "hard_impact_observed",
    "high_slip_observed",
    "torque_spike_observed",
    "torque_clamp_observed",
    "push_observed",
]


def add_derived_event_columns(df: pd.DataFrame, thresholds: DiagnosticThresholds | None = None) -> pd.DataFrame:
    """Add neutral, observation-style event booleans.

    These columns are *not* pass/fail. They mark observed conditions and allow
    downstream metrics to produce slices such as all frames, pose-stable frames,
    and before/after first major event.
    """
    thresholds = thresholds or DiagnosticThresholds()
    d = df.copy()

    if "base_roll" in d.columns:
        d["base_roll_abs"] = numeric_col(d, "base_roll", 0.0).abs()
    if "base_pitch" in d.columns:
        d["base_pitch_abs"] = numeric_col(d, "base_pitch", 0.0).abs()

    d["terminated_observed"] = bool_series(d, "terminated") | bool_series(d, "terminated_observed")
    d["truncated_observed"] = bool_series(d, "truncated") | bool_series(d, "truncated_observed")
    d["done_observed"] = bool_series(d, "done") | d["terminated_observed"] | d["truncated_observed"]
    # reset_observed is separate from done; the recorder may have explicit reset flags.
    d["reset_observed"] = bool_series(d, "reset_observed") | bool_series(d, "transition_done_after_action")

    if "base_roll_abs" in d.columns:
        d["large_roll_observed"] = d["base_roll_abs"] >= thresholds.large_roll_deg
    else:
        d["large_roll_observed"] = False
    if "base_pitch_abs" in d.columns:
        d["large_pitch_observed"] = d["base_pitch_abs"] >= thresholds.large_pitch_deg
    else:
        d["large_pitch_observed"] = False

    if "base_height_local" in d.columns:
        height = numeric_col(d, "base_height_local", np.nan)
        nominal = numeric_col(d, "nominal_stand_height", np.nan)
        if nominal.notna().any():
            nom = nominal.fillna(nominal.dropna().median() if nominal.dropna().size else np.nan)
        else:
            nom = pd.Series(np.nan, index=d.index)
        d["height_drop_observed"] = (height < thresholds.height_ratio_low * nom).fillna(False)
    else:
        d["height_drop_observed"] = False

    # Contact/impact/slip derived from per-foot columns.
    impact_any = pd.Series(False, index=d.index)
    high_slip_any = pd.Series(False, index=d.index)
    for leg in LEGS:
        tvz = numeric_col(d, f"foot_{leg}_touchdown_vz", np.nan).abs()
        if tvz.notna().any():
            impact_any |= tvz >= thresholds.hard_impact_vz
        slip = numeric_col(d, f"foot_{leg}_stance_slip_xy", np.nan)
        if slip.notna().any():
            high_slip_any |= slip >= thresholds.high_slip_xy
    d["hard_impact_observed"] = impact_any
    d["high_slip_observed"] = high_slip_any

    torque_spike = pd.Series(False, index=d.index)
    torque_clamp = pd.Series(False, index=d.index)
    torque_cols = [c for c in d.columns if c.startswith("torque_utilization_")]
    for c in torque_cols:
        v = numeric_col(d, c, np.nan).abs()
        torque_spike |= v >= thresholds.torque_spike_util
        torque_clamp |= v >= thresholds.torque_clamp_util
    d["torque_spike_observed"] = torque_spike
    d["torque_clamp_observed"] = torque_clamp

    d["push_observed"] = bool_series(d, "push_event") | bool_series(d, "push_observed")
    d["major_event_observed"] = pd.Series(False, index=d.index)
    for c in MAJOR_EVENT_COLUMNS:
        d["major_event_observed"] |= bool_series(d, c)

    # Recorder versions may carry the column but leave it empty.
    derived_segment_time = time_since_segment_start(d)
    if "time_since_command_switch" not in d.columns:
        d["time_since_command_switch"] = derived_segment_time
    else:
        supplied = numeric_col(d, "time_since_command_switch", np.nan)
        d["time_since_command_switch"] = supplied.where(supplied.notna(), derived_segment_time)
    return d


def time_since_segment_start(df: pd.DataFrame) -> pd.Series:
    if "time" not in df.columns:
        return pd.Series(0.0, index=df.index)
    t = numeric_col(df, "time", 0.0)
    keys = command_segment_key_columns(df)
    if not keys:
        return t - float(t.min())
    t0 = t.groupby([df[c] for c in keys], sort=False).transform("min")
    return (t - t0).clip(lower=0.0)


def pose_stable_mask(df: pd.DataFrame, thresholds: DiagnosticThresholds | None = None) -> pd.Series:
    thresholds = thresholds or DiagnosticThresholds()
    d = add_derived_event_columns(df, thresholds)
    mask = pd.Series(True, index=d.index)
    if "large_roll_observed" in d.columns:
        mask &= ~bool_series(d, "large_roll_observed")
    if "large_pitch_observed" in d.columns:
        mask &= ~bool_series(d, "large_pitch_observed")
    if "height_drop_observed" in d.columns:
        mask &= ~bool_series(d, "height_drop_observed")
    mask &= ~bool_series(d, "done_observed")
    mask &= ~bool_series(d, "reset_observed")
    return mask.fillna(False)


def before_first_major_event_mask(df: pd.DataFrame, thresholds: DiagnosticThresholds | None = None) -> pd.Series:
    thresholds = thresholds or DiagnosticThresholds()
    d = add_derived_event_columns(df, thresholds)
    out = pd.Series(True, index=d.index)
    if "time" not in d.columns:
        return out
    group_cols = command_segment_key_columns(d)
    if not group_cols:
        group_cols = []
    for _, g in d.groupby(group_cols, dropna=False, sort=False) if group_cols else [(None, d)]:
        major = bool_series(g, "major_event_observed")
        if major.any():
            first_idx = major[major].index[0]
            first_t = float(numeric_col(g.loc[[first_idx]], "time", np.nan).iloc[0])
            out.loc[g.index] = numeric_col(g, "time", np.nan) < first_t
        else:
            out.loc[g.index] = True
    return out.fillna(False)


def after_first_major_event_mask(df: pd.DataFrame, thresholds: DiagnosticThresholds | None = None) -> pd.Series:
    b = before_first_major_event_mask(df, thresholds)
    d = add_derived_event_columns(df, thresholds)
    has_major_by_group = pd.Series(False, index=d.index)
    group_cols = command_segment_key_columns(d)
    for _, g in d.groupby(group_cols, dropna=False, sort=False) if group_cols else [(None, d)]:
        has = bool_series(g, "major_event_observed").any()
        has_major_by_group.loc[g.index] = bool(has)
    return (~b & has_major_by_group).fillna(False)


def command_settled_mask(df: pd.DataFrame, thresholds: DiagnosticThresholds | None = None) -> pd.Series:
    thresholds = thresholds or DiagnosticThresholds()
    d = add_derived_event_columns(df, thresholds)
    return numeric_col(d, "time_since_command_switch", 0.0) >= thresholds.command_settle_s


def _bout_records_for_bool(df: pd.DataFrame, col: str, value_col: str | None = None) -> list[dict[str, Any]]:
    if col not in df.columns:
        return []
    records: list[dict[str, Any]] = []
    group_cols = group_key_columns(df)
    groups = df.groupby(group_cols, dropna=False, sort=False) if group_cols else [(None, df)]
    for key, g in groups:
        mask = bool_series(g, col)
        if not mask.any():
            continue
        # Consecutive true runs inside the group.
        run_id = (mask != mask.shift(fill_value=False)).cumsum()
        for _, rg in g[mask].groupby(run_id[mask], sort=False):
            t = numeric_col(rg, "time", np.nan)
            start_t = float(t.iloc[0]) if t.notna().any() else None
            end_t = float(t.iloc[-1]) if t.notna().any() else None
            val_peak = None
            if value_col and value_col in rg.columns:
                v = numeric_col(rg, value_col, np.nan).abs().dropna()
                val_peak = None if v.empty else float(v.max())
            first = rg.iloc[0]
            rec = {
                "event_type": col.replace("_observed", "") + "_bout",
                "case_id": int(first.get("case_id", 0)) if pd.notna(first.get("case_id", 0)) else 0,
                "env_id": int(first.get("env_id", 0)) if pd.notna(first.get("env_id", 0)) else 0,
                "episode_id": int(first.get("episode_id", 0)) if pd.notna(first.get("episode_id", 0)) else 0,
                "cmd_segment_id": int(first.get("cmd_segment_id", 0)) if pd.notna(first.get("cmd_segment_id", 0)) else 0,
                "cmd_target_mode": str(first.get("cmd_target_mode", first.get("cmd_mode", first.get("__mode", "unknown")))),
                "terrain_type": str(first.get("terrain_type", "unknown")),
                "terrain_level": _maybe_float(first.get("terrain_level", None)),
                "dr_level": str(first.get("dr_level", "unknown")),
                "start_time": start_t,
                "end_time": end_t,
                "duration_s": None if start_t is None or end_t is None else max(0.0, end_t - start_t),
                "frame_count": int(len(rg)),
                "peak_value": val_peak,
            }
            records.append(rec)
    return records


def _maybe_float(x: Any) -> float | None:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return None


def extract_event_timeline(df: pd.DataFrame, thresholds: DiagnosticThresholds | None = None) -> list[dict[str, Any]]:
    """Return neutral event/bout observations.

    Continuous conditions are represented as bouts, not one event per frame.
    Discrete transitions (command switch, reset, touchdown, liftoff, push) are
    represented as instantaneous observations.
    """
    thresholds = thresholds or DiagnosticThresholds()
    d = add_derived_event_columns(df, thresholds)
    timeline: list[dict[str, Any]] = []

    # Continuous bout observations.
    value_map = {
        "large_roll_observed": "base_roll_abs",
        "large_pitch_observed": "base_pitch_abs",
        "height_drop_observed": "base_height_local",
        "hard_impact_observed": None,
        "high_slip_observed": None,
        "torque_spike_observed": None,
        "torque_clamp_observed": None,
    }
    for col, val in value_map.items():
        timeline.extend(_bout_records_for_bool(d, col, val))

    # Discrete observations: start of every segment and explicit flags.
    group_cols = command_segment_key_columns(d)
    for _, g in d.groupby(group_cols, dropna=False, sort=False) if group_cols else [(None, d)]:
        if len(g) == 0:
            continue
        r = g.iloc[0]
        timeline.append(_instant_event("command_segment_started", r, numeric_col(g.iloc[[0]], "time", np.nan).iloc[0], None))

    for col in ["done_observed", "reset_observed", "terminated_observed", "truncated_observed", "push_observed"]:
        if col not in d.columns:
            continue
        keys = group_key_columns(d)
        prev = bool_series(d, col).groupby([d[c] for c in keys], sort=False).shift(fill_value=False) if keys else bool_series(d, col).shift(fill_value=False)
        starts = bool_series(d, col) & ~prev
        for idx in d[starts].index:
            r = d.loc[idx]
            timeline.append(_instant_event(col, r, r.get("time", None), r.get(col, None)))

    for leg in LEGS:
        for kind in ["touchdown", "liftoff"]:
            col = f"foot_{leg}_{kind}"
            if col not in d.columns:
                continue
            starts = bool_series(d, col)
            for idx in d[starts].index:
                r = d.loc[idx]
                timeline.append(_instant_event(f"{kind}_observed", r, r.get("time", None), leg, extra={"leg": leg}))

    timeline.sort(
        key=lambda x: (
            x.get("case_id", 0),
            x.get("start_time") is None,
            x.get("start_time", x.get("time", 0)) or 0,
            x.get("env_id", 0),
        )
    )
    return timeline


def _instant_event(event_type: str, row: pd.Series, time_value: Any, value: Any, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    rec = {
        "event_type": event_type,
        "case_id": int(row.get("case_id", 0)) if pd.notna(row.get("case_id", 0)) else 0,
        "env_id": int(row.get("env_id", 0)) if pd.notna(row.get("env_id", 0)) else 0,
        "episode_id": int(row.get("episode_id", 0)) if pd.notna(row.get("episode_id", 0)) else 0,
        "cmd_segment_id": int(row.get("cmd_segment_id", 0)) if pd.notna(row.get("cmd_segment_id", 0)) else 0,
        "cmd_target_mode": str(row.get("cmd_target_mode", row.get("cmd_mode", row.get("__mode", "unknown")))),
        "terrain_type": str(row.get("terrain_type", "unknown")),
        "terrain_level": _maybe_float(row.get("terrain_level", None)),
        "dr_level": str(row.get("dr_level", "unknown")),
        "time": _maybe_float(time_value),
        "start_time": _maybe_float(time_value),
        "event_value": _maybe_float(value),
        "time_since_command_switch": _maybe_float(row.get("time_since_command_switch", None)),
    }
    if extra:
        rec.update(extra)
    return rec


def event_counts(timeline: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in timeline:
        typ = str(e.get("event_type", "unknown"))
        out[typ] = out.get(typ, 0) + 1
    return out


def event_frame_counts(df: pd.DataFrame, thresholds: DiagnosticThresholds | None = None) -> dict[str, int]:
    d = add_derived_event_columns(df, thresholds)
    out = {}
    for col in EVENT_BOOL_COLUMNS + ["major_event_observed"]:
        if col in d.columns:
            out[col.replace("_observed", "")] = int(bool_series(d, col).sum())
    return out


def first_major_event_time_by_segment(df: pd.DataFrame, thresholds: DiagnosticThresholds | None = None) -> pd.Series:
    d = add_derived_event_columns(df, thresholds)
    out = pd.Series(np.nan, index=d.index, dtype=float)
    group_cols = command_segment_key_columns(d)
    for _, g in d.groupby(group_cols, dropna=False, sort=False) if group_cols else [(None, d)]:
        major = bool_series(g, "major_event_observed")
        if major.any():
            ft = float(numeric_col(g[major].iloc[[0]], "time", np.nan).iloc[0])
            out.loc[g.index] = ft
    return out


def progress_along_command(df: pd.DataFrame, mask: pd.Series | None = None) -> float | None:
    if len(df) == 0 or not all(c in df.columns for c in ["base_lin_vel_b_x", "base_lin_vel_b_y", "time"]):
        return None
    vx, vy, _ = command_components(df, prefer_target=True)
    norm = np.sqrt(vx**2 + vy**2)
    if pd.Series(norm).dropna().max() <= 1e-6:
        return None
    dir_x = pd.Series(np.where(norm > 1e-6, vx / norm, 0.0), index=df.index)
    dir_y = pd.Series(np.where(norm > 1e-6, vy / norm, 0.0), index=df.index)
    along_vel = numeric_col(df, "base_lin_vel_b_x", 0.0) * dir_x + numeric_col(df, "base_lin_vel_b_y", 0.0) * dir_y
    if mask is not None:
        along_vel = along_vel.where(mask.reindex(df.index).fillna(False), 0.0)
    t = numeric_col(df, "time", np.nan)
    keys = command_segment_key_columns(df)
    if keys:
        dt = t.groupby([df[c] for c in keys], sort=False).diff().fillna(0).clip(lower=0)
    else:
        dt = t.diff().fillna(0).clip(lower=0)
    return float((along_vel * dt).sum())


def segment_event_summary(df: pd.DataFrame, timeline: list[dict[str, Any]], thresholds: DiagnosticThresholds | None = None) -> list[dict[str, Any]]:
    thresholds = thresholds or DiagnosticThresholds()
    d = add_derived_event_columns(df, thresholds)
    pose_stable = pose_stable_mask(d, thresholds)
    before_major = before_first_major_event_mask(d, thresholds)
    after_major = after_first_major_event_mask(d, thresholds)
    events_by_key: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for e in timeline:
        key = (
            int(e.get("case_id", 0)),
            int(e.get("env_id", 0)),
            int(e.get("cmd_segment_id", 0)),
        )
        events_by_key.setdefault(key, []).append(e)

    summaries: list[dict[str, Any]] = []
    group_cols = command_segment_key_columns(d)
    for key, g in d.groupby(group_cols, dropna=False, sort=True) if group_cols else [(None, d)]:
        first = g.iloc[0]
        case_id = int(first.get("case_id", 0)) if pd.notna(first.get("case_id", 0)) else 0
        env_id = int(first.get("env_id", 0)) if pd.notna(first.get("env_id", 0)) else 0
        episode_id = int(first.get("episode_id", 0)) if pd.notna(first.get("episode_id", 0)) else 0
        seg_id = int(first.get("cmd_segment_id", 0)) if pd.notna(first.get("cmd_segment_id", 0)) else 0
        evs = [
            e
            for e in events_by_key.get((case_id, env_id, seg_id), [])
            if e.get("event_type") != "command_segment_started"
        ]
        labels: dict[str, int] = {}
        for e in evs:
            labels[e["event_type"]] = labels.get(e["event_type"], 0) + 1
        t = numeric_col(g, "time", np.nan)
        t0 = float(t.min()) if t.notna().any() else None
        t1 = float(t.max()) if t.notna().any() else None
        ev_times = [e.get("start_time", e.get("time")) for e in evs if e.get("start_time", e.get("time")) is not None]
        first_event_time = float(min(ev_times)) if ev_times else None
        first_major = first_major_event_time_by_segment(g, thresholds).dropna()
        first_major_t = float(first_major.iloc[0]) if len(first_major) else None
        idx = g.index
        ps = pose_stable.loc[idx]
        bm = before_major.loc[idx]
        am = after_major.loc[idx]
        summaries.append({
            "case_id": case_id,
            "env_id": env_id,
            "episode_id": episode_id,
            "cmd_segment_id": seg_id,
            "cmd_target_mode": str(first.get("cmd_target_mode", first.get("cmd_mode", first.get("__mode", "unknown")))),
            "cmd_mode_first": str(first.get("cmd_mode", first.get("__mode", "unknown"))),
            "terrain_type": str(first.get("terrain_type", "unknown")),
            "terrain_level": _maybe_float(first.get("terrain_level", None)),
            "dr_level": str(first.get("dr_level", "unknown")),
            "rows": int(len(g)),
            "duration_s": None if t0 is None or t1 is None else max(0.0, t1 - t0),
            "observed_event_count": int(len(evs)),
            "observed_event_labels": sorted(labels.keys()),
            "observed_event_counts": labels,
            "first_event_time_s": first_event_time,
            "first_event_time_since_segment_start_s": None if first_event_time is None or t0 is None else float(first_event_time - t0),
            "first_major_event_time_s": first_major_t,
            "first_major_event_time_since_segment_start_s": None if first_major_t is None or t0 is None else float(first_major_t - t0),
            "reset_observed": bool(bool_series(g, "reset_observed").any()),
            "done_observed": bool(bool_series(g, "done_observed").any()),
            "pose_stable_fraction_all": float(ps.mean()) if len(ps) else None,
            "pose_stable_fraction_before_first_major_event": float((ps & bm).sum() / max(1, bm.sum())) if len(ps) and bm.sum() > 0 else None,
            "raw_progress_along_command": progress_along_command(g),
            "pose_stable_progress_along_command": progress_along_command(g, ps),
            "pre_first_major_event_progress_along_command": progress_along_command(g, bm),
            "pose_stable_pre_first_major_event_progress_along_command": progress_along_command(g, ps & bm),
            "post_first_major_event_progress_along_command": progress_along_command(g, am),
        })
    return summaries
