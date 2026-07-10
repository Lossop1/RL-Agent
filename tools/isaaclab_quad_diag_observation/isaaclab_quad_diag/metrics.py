from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .events import (
    add_derived_event_columns,
    extract_event_timeline,
    segment_event_summary,
    event_counts,
    event_frame_counts,
    before_first_major_event_mask,
    pose_stable_mask,
    progress_along_command,
)
from .notes import build_diagnostic_notes
from .schema import DiagnosticThresholds, LEGS
from .slices import build_slices
from .util import (
    bool_series,
    command_segment_key_columns,
    command_components,
    ensure_dir,
    group_key_columns,
    load_record,
    numeric_col,
    stats,
    write_json,
)


def compute_all_metrics(record_path: str | Path, out_dir: str | Path | None = None, meta_path: str | Path | None = None) -> dict[str, Any]:
    df = load_record(record_path)
    meta = _load_meta(meta_path or Path(record_path).with_name("record_meta.json"))
    return compute_metrics(df, source=str(record_path), out_dir=out_dir, record_meta=meta)


def _load_meta(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def compute_metrics(df: pd.DataFrame, source: str | None = None, out_dir: str | Path | None = None, record_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    thresholds = DiagnosticThresholds()
    record_meta = record_meta or {}
    df = add_derived_event_columns(df, thresholds)
    timeline = extract_event_timeline(df, thresholds)
    segs = segment_event_summary(df, timeline, thresholds)
    slices = build_slices(df, thresholds)
    notes = build_diagnostic_notes(df, timeline, segs, record_meta=record_meta)

    metrics: dict[str, Any] = {
        "schema_version": "ilqd_observation_v0.5.1",
        "source": source,
        "record_meta": record_meta,
        "coverage": compute_coverage(df, record_meta),
        "event_counts_by_bout_or_instant": event_counts(timeline),
        "event_frame_counts": event_frame_counts(df, thresholds),
        "tracking": compute_tracking(df, slices.masks),
        "stand": compute_stand(df, slices.masks),
        "gait": compute_gait(df, slices.masks),
        "contact": compute_contact(df, slices.masks),
        "slip": compute_slip(df, slices.masks),
        "posture": compute_posture(df, slices.masks),
        "terrain": compute_terrain_breakdown(df, segs, slices.masks, record_meta),
        "robustness": compute_robustness_breakdown(df, segs, timeline, slices.masks, record_meta),
        "hardware": compute_hardware(df),
        "symmetry": compute_symmetry(df),
        "segment_event_summary": segs,
        "command_breakdown": compute_command_breakdown(segs),
        "diagnostic_notes": notes,
        "missing_fields": missing_field_summary(df),
        "slice_descriptions": slices.description,
    }
    if out_dir is not None:
        out = ensure_dir(out_dir)
        write_json(out / "metrics.json", metrics)
        write_json(out / "event_timeline.json", timeline)
        write_json(out / "segment_event_summary.json", segs)
        write_json(out / "notes.json", notes)
        write_json(out / "terrain_breakdown.json", metrics["terrain"])
        write_json(out / "robustness_breakdown.json", metrics["robustness"])
        (out / "summary.md").write_text(render_summary(metrics), encoding="utf-8")
    return metrics


def compute_coverage(df: pd.DataFrame, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = meta or {}
    time = numeric_col(df, "time", np.nan)
    if time.notna().any() and "case_id" in df.columns:
        duration = float(sum(
            numeric_col(group, "time", np.nan).max() - numeric_col(group, "time", np.nan).min()
            for _, group in df.groupby("case_id", dropna=False, sort=False)
        ))
    else:
        duration = float(time.max() - time.min()) if time.notna().any() else None
    observed_terrain_types = sorted([str(x) for x in df.get("terrain_type", pd.Series([], dtype=str)).dropna().unique()])
    observed_dr_levels = sorted([str(x) for x in df.get("dr_level", pd.Series([], dtype=str)).dropna().unique()])
    push_count = int((bool_series(df, "push_event") | bool_series(df, "push_observed")).sum())
    requested_suite = meta.get("requested_suite_config", meta.get("suite", {}))
    executed = meta.get("executed_runtime_config", {})
    terrain_height_sources = sorted([str(x) for x in df.get("terrain_height_source", pd.Series([], dtype=str)).dropna().unique()]) if "terrain_height_source" in df.columns else []
    return {
        "rows": int(len(df)),
        "cases": int(df["case_id"].nunique()) if "case_id" in df.columns else 1,
        "envs": int(df["env_id"].nunique()) if "env_id" in df.columns else 1,
        "episodes": int(df[group_key_columns(df, include_segment=False)].drop_duplicates().shape[0])
        if group_key_columns(df, include_segment=False)
        else None,
        "segments": int(
            df[[c for c in ["case_id", "env_id", "cmd_segment_id"] if c in df.columns]]
            .drop_duplicates()
            .shape[0]
        ) if any(c in df.columns for c in ["case_id", "env_id", "cmd_segment_id"]) else None,
        "segment_attempts": int(df[command_segment_key_columns(df)].drop_duplicates().shape[0])
        if command_segment_key_columns(df) else None,
        "episode_segment_fragments": int(df[group_key_columns(df)].drop_duplicates().shape[0])
        if group_key_columns(df) else None,
        "duration_s": duration,
        "command_modes_observed": sorted([str(x) for x in df.get("cmd_target_mode", df.get("__mode", pd.Series([], dtype=str))).dropna().unique()]),
        "terrain_types_observed": observed_terrain_types,
        "terrain_levels_observed": sorted([float(x) for x in pd.to_numeric(df.get("terrain_level", pd.Series([], dtype=float)), errors="coerce").dropna().unique()]),
        "dr_levels_observed": observed_dr_levels,
        "push_frames_observed": push_count,
        "terrain_height_sources_observed": terrain_height_sources,
        "requested_suite_config_present": bool(requested_suite),
        "executed_runtime_config_present": bool(executed),
        "requested_suite_config": requested_suite,
        "executed_runtime_config": executed,
        "recording_notes": meta.get("recording_notes", []),
    }


def _base_velocity_and_cmd(df: pd.DataFrame, command_source: str = "target") -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    avx = numeric_col(df, "base_lin_vel_b_x", np.nan)
    avy = numeric_col(df, "base_lin_vel_b_y", np.nan)
    awz = numeric_col(df, "base_ang_vel_b_z", np.nan)
    if command_source == "applied":
        cvx = numeric_col(df, "cmd_vx", np.nan)
        cvy = numeric_col(df, "cmd_vy", np.nan)
        cwz = numeric_col(df, "cmd_wz", np.nan)
    else:
        cvx = numeric_col(df, "cmd_target_vx", np.nan)
        cvy = numeric_col(df, "cmd_target_vy", np.nan)
        cwz = numeric_col(df, "cmd_target_wz", np.nan)
    # Fallbacks when applied command is not recorded.
    if cvx.isna().all(): cvx = numeric_col(df, "cmd_target_vx", 0.0)
    if cvy.isna().all(): cvy = numeric_col(df, "cmd_target_vy", 0.0)
    if cwz.isna().all(): cwz = numeric_col(df, "cmd_target_wz", 0.0)
    return avx, avy, awz, cvx, cvy, cwz


def tracking_stats_for_df(df: pd.DataFrame, command_source: str = "target") -> dict[str, Any]:
    avx, avy, awz, cvx, cvy, cwz = _base_velocity_and_cmd(df, command_source=command_source)
    vx_err = (avx - cvx).abs()
    vy_err = (avy - cvy).abs()
    wz_err = (awz - cwz).abs()
    xy_err = np.sqrt(vx_err**2 + vy_err**2)
    norm = np.sqrt(cvx**2 + cvy**2)
    dir_x = pd.Series(np.where(norm > 1e-6, cvx / norm, 0.0), index=df.index)
    dir_y = pd.Series(np.where(norm > 1e-6, cvy / norm, 0.0), index=df.index)
    along = avx * dir_x + avy * dir_y
    lateral = -avx * dir_y + avy * dir_x
    return {
        "rows": int(len(df)),
        "command_source": command_source,
        "vx_abs_error": stats(vx_err),
        "vy_abs_error": stats(vy_err),
        "wz_abs_error": stats(wz_err),
        "xy_error_norm": stats(xy_err),
        "along_command_speed": stats(along[norm > 1e-6]),
        "off_axis_speed_abs": stats(lateral[norm > 1e-6].abs()),
    }


def _by_bucket(df: pd.DataFrame, func: Callable[[pd.DataFrame], dict[str, Any]], cols: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    use_cols = [c for c in cols if c in df.columns]
    if not use_cols:
        return out
    for key, g in df.groupby(use_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        name = "/".join(f"{c}={v}" for c, v in zip(use_cols, key))
        out[name] = func(g)
    return out


def compute_tracking(df: pd.DataFrame, masks: dict[str, pd.Series]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for slice_name, mask in masks.items():
        sub = df[mask.reindex(df.index).fillna(False)]
        if len(sub) == 0:
            continue
        out[slice_name] = {
            "target_command": tracking_stats_for_df(sub, "target"),
            "applied_command": tracking_stats_for_df(sub, "applied"),
            "by_cmd_mode": _by_bucket(sub, lambda g: tracking_stats_for_df(g, "target"), ["cmd_target_mode"]),
            "by_cmd_terrain": _by_bucket(sub, lambda g: tracking_stats_for_df(g, "target"), ["cmd_target_mode", "terrain_type"]),
            "by_cmd_dr": _by_bucket(sub, lambda g: tracking_stats_for_df(g, "target"), ["cmd_target_mode", "dr_level"]),
        }
    return out


def compute_stand(df: pd.DataFrame, masks: dict[str, pd.Series]) -> dict[str, Any]:
    mode = df.get("cmd_target_mode", df.get("__mode", pd.Series("", index=df.index))).astype(str)
    stand_mask = mode.eq("stand") | ((numeric_col(df, "cmd_target_vx", 0).abs() + numeric_col(df, "cmd_target_vy", 0).abs() + numeric_col(df, "cmd_target_wz", 0).abs()) <= 0.05)
    out = {}
    for name in ["all_frames", "command_settled", "pose_stable", "pose_stable_before_first_major_event"]:
        m = masks.get(name, pd.Series(True, index=df.index)) & stand_mask
        g = df[m]
        if len(g) == 0:
            continue
        lin = np.sqrt(numeric_col(g, "base_lin_vel_b_x", np.nan)**2 + numeric_col(g, "base_lin_vel_b_y", np.nan)**2)
        out[name] = {
            "rows": int(len(g)),
            "residual_lin_speed": stats(lin),
            "residual_yaw_rate_abs": stats(numeric_col(g, "base_ang_vel_b_z", np.nan).abs()),
            "upright_proxy": stats(numeric_col(g, "projected_gravity_b_z", np.nan).abs()),
            "base_height_local": stats(numeric_col(g, "base_height_local", np.nan)),
            "four_foot_contact_fraction": _four_foot_fraction(g),
            "base_xy_drift_observed": _xy_drift(g),
        }
    return out


def _xy_drift(df: pd.DataFrame) -> float | None:
    if not all(c in df.columns for c in ["base_pos_w_x", "base_pos_w_y"]):
        return None
    x = numeric_col(df, "base_pos_w_x", np.nan).dropna()
    y = numeric_col(df, "base_pos_w_y", np.nan).dropna()
    if len(x) == 0 or len(y) == 0:
        return None
    return float(np.hypot(x.iloc[-1] - x.iloc[0], y.iloc[-1] - y.iloc[0]))


def _four_foot_fraction(df: pd.DataFrame) -> float | None:
    cols = [f"foot_{l}_contact" for l in LEGS if f"foot_{l}_contact" in df.columns]
    if len(cols) < 4:
        return None
    c = sum(bool_series(df, col).astype(int) for col in cols)
    return float((c == 4).mean())


def compute_gait(df: pd.DataFrame, masks: dict[str, pd.Series]) -> dict[str, Any]:
    moving = (numeric_col(df, "cmd_target_vx", 0).abs() + numeric_col(df, "cmd_target_vy", 0).abs() + numeric_col(df, "cmd_target_wz", 0).abs()) > 0.05
    g = df[moving]
    out: dict[str, Any] = {"rows": int(len(g))}
    duties = {}
    clear_peaks = {}
    for leg in LEGS:
        c = f"foot_{leg}_contact"
        if c in g.columns:
            duties[leg] = float(bool_series(g, c).mean()) if len(g) else None
            out[f"{leg}_duty_factor"] = duties[leg]
        clr = numeric_col(g, f"foot_{leg}_clearance_local", np.nan)
        if clr.notna().any():
            out[f"{leg}_clearance_local"] = stats(clr)
            clear_peaks[leg] = float(clr.max())
        elif f"foot_{leg}_pos_w_z" in g.columns:
            # Fallback is intentionally labeled as world_z, not local clearance.
            out[f"{leg}_foot_world_z"] = stats(numeric_col(g, f"foot_{leg}_pos_w_z", np.nan))
        air = numeric_col(g, f"foot_{leg}_air_time", np.nan)
        if air.notna().any():
            out[f"{leg}_air_time"] = stats(air)
    if duties:
        out["per_leg_duty_range"] = None if any(v is None for v in duties.values()) else float(max(duties.values()) - min(duties.values()))
        if all(k in duties for k in ["FL", "FR"]): out["front_left_right_duty_diff"] = _absdiff(duties["FL"], duties["FR"])
        if all(k in duties for k in ["RL", "RR"]): out["rear_left_right_duty_diff"] = _absdiff(duties["RL"], duties["RR"])
        diagonal_legs = ["FL", "RR", "FR", "RL"]
        if all(k in duties and duties[k] is not None for k in diagonal_legs):
            out["diagonal_duty_diff"] = abs(
                (duties["FL"] + duties["RR"]) / 2
                - (duties["FR"] + duties["RL"]) / 2
            )
    if clear_peaks:
        out["per_leg_clearance_peak_range"] = float(max(clear_peaks.values()) - min(clear_peaks.values()))
    # Contact count distribution.
    cols = [f"foot_{l}_contact" for l in LEGS if f"foot_{l}_contact" in g.columns]
    if cols:
        cc = sum(bool_series(g, c).astype(int) for c in cols)
        out["contact_count_distribution"] = {str(k): float(v) for k, v in cc.value_counts(normalize=True).sort_index().items()}
        diag = ((bool_series(g, "foot_FL_contact") & bool_series(g, "foot_RR_contact")) | (bool_series(g, "foot_FR_contact") & bool_series(g, "foot_RL_contact"))) if all(f"foot_{l}_contact" in g.columns for l in LEGS) else None
        if diag is not None:
            out["diagonal_pair_fraction"] = float(diag.mean())
    return out


def _absdiff(a, b):
    if a is None or b is None: return None
    return float(abs(a - b))


def compute_contact(df: pd.DataFrame, masks: dict[str, pd.Series]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    tvz_vals = []
    force_vals = []
    for leg in LEGS:
        tvz = numeric_col(df, f"foot_{leg}_touchdown_vz", np.nan).abs().dropna()
        if len(tvz): tvz_vals.extend(tvz.tolist())
        f = numeric_col(df, f"foot_{leg}_force_norm", np.nan).dropna()
        if len(f): force_vals.extend(f.tolist())
    out["touchdown_vz_abs"] = stats(tvz_vals)
    out["force_norm"] = stats(force_vals)
    if len(df) and "cmd_target_mode" in df.columns:
        out["by_cmd_mode"] = {}
        for name, g in df.groupby("cmd_target_mode", dropna=False, sort=True):
            tv = []
            fv = []
            for leg in LEGS:
                a = numeric_col(g, f"foot_{leg}_touchdown_vz", np.nan).abs().dropna()
                b = numeric_col(g, f"foot_{leg}_force_norm", np.nan).dropna()
                tv.extend(a.tolist()); fv.extend(b.tolist())
            out["by_cmd_mode"][str(name)] = {"touchdown_vz_abs": stats(tv), "force_norm": stats(fv)}
    return out


def compute_slip(df: pd.DataFrame, masks: dict[str, pd.Series]) -> dict[str, Any]:
    vals = []
    by_leg = {}
    for leg in LEGS:
        s = numeric_col(df, f"foot_{leg}_stance_slip_xy", np.nan).dropna()
        if len(s):
            vals.extend(s.tolist())
            by_leg[leg] = stats(s)
    return {"stance_xy_speed": stats(vals), "by_leg": by_leg}


def compute_posture(df: pd.DataFrame, masks: dict[str, pd.Series]) -> dict[str, Any]:
    h = numeric_col(df, "base_height_local", np.nan)
    return {
        "base_roll_abs": stats(numeric_col(df, "base_roll_abs", np.nan)),
        "base_pitch_abs": stats(numeric_col(df, "base_pitch_abs", np.nan)),
        "base_height_local": stats(h),
        "base_height_drop_from_median_max": None if h.dropna().empty else float(h.dropna().median() - h.dropna().min()),
        "pose_stable_fraction_all": float(masks.get("pose_stable", pd.Series(False, index=df.index)).mean()) if len(df) else None,
        "pose_stable_before_first_major_fraction": float(masks.get("pose_stable_before_first_major_event", pd.Series(False, index=df.index)).mean()) if len(df) else None,
    }


def compute_terrain_breakdown(df: pd.DataFrame, segs: list[dict[str, Any]], masks: dict[str, pd.Series], meta: dict[str, Any] | None = None) -> dict[str, Any]:
    seg_df = pd.DataFrame(segs)
    out: dict[str, Any] = {
        "coverage": {
            "terrain_types_observed": sorted([str(x) for x in df.get("terrain_type", pd.Series([], dtype=str)).dropna().unique()]),
            "terrain_levels_observed": sorted([float(x) for x in pd.to_numeric(df.get("terrain_level", pd.Series([], dtype=float)), errors="coerce").dropna().unique()]),
            "terrain_height_sources_observed": sorted([str(x) for x in df.get("terrain_height_source", pd.Series([], dtype=str)).dropna().unique()]) if "terrain_height_source" in df.columns else [],
        },
        "by_terrain_type": {},
        "by_terrain_level": {},
        "by_terrain_and_command": {},
    }
    if len(seg_df):
        out["segment_progress_all"] = _segment_progress_stats(seg_df)
        for name, g in seg_df.groupby("terrain_type", dropna=False, sort=True):
            out["by_terrain_type"][str(name)] = _segment_progress_stats(g)
        if "terrain_level" in seg_df.columns:
            for name, g in seg_df.groupby("terrain_level", dropna=False, sort=True):
                out["by_terrain_level"][str(name)] = _segment_progress_stats(g)
        if all(c in seg_df.columns for c in ["terrain_type", "cmd_target_mode"]):
            for key, g in seg_df.groupby(["terrain_type", "cmd_target_mode"], dropna=False, sort=True):
                out["by_terrain_and_command"][f"terrain={key[0]}/cmd={key[1]}"] = _segment_progress_stats(g)
    # Per-frame terrain bucket metrics.
    if "terrain_type" in df.columns:
        out["frame_metrics_by_terrain"] = _by_bucket(df, _terrain_frame_metrics, ["terrain_type"])
    return out


def _segment_progress_stats(seg_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "segments_observed": int(len(seg_df)),
        "raw_progress_along_command": stats(seg_df.get("raw_progress_along_command", pd.Series([], dtype=float))),
        "pose_stable_progress_along_command": stats(seg_df.get("pose_stable_progress_along_command", pd.Series([], dtype=float))),
        "pre_first_major_event_progress_along_command": stats(seg_df.get("pre_first_major_event_progress_along_command", pd.Series([], dtype=float))),
        "pose_stable_pre_first_major_event_progress_along_command": stats(seg_df.get("pose_stable_pre_first_major_event_progress_along_command", pd.Series([], dtype=float))),
        "pose_stable_fraction_all": stats(seg_df.get("pose_stable_fraction_all", pd.Series([], dtype=float))),
        "first_major_event_time_since_segment_start_s": stats(seg_df.get("first_major_event_time_since_segment_start_s", pd.Series([], dtype=float))),
        "reset_observed_segments": int(seg_df.get("reset_observed", pd.Series([], dtype=bool)).astype(bool).sum()) if "reset_observed" in seg_df else 0,
        "done_observed_segments": int(seg_df.get("done_observed", pd.Series([], dtype=bool)).astype(bool).sum()) if "done_observed" in seg_df else 0,
    }


def compute_command_breakdown(segs: list[dict[str, Any]]) -> dict[str, Any]:
    seg_df = pd.DataFrame(segs)
    if len(seg_df) == 0 or "cmd_target_mode" not in seg_df.columns:
        return {}
    out: dict[str, Any] = {}
    for mode, group in seg_df.groupby("cmd_target_mode", dropna=False, sort=True):
        first_major = pd.to_numeric(
            group.get("first_major_event_time_since_segment_start_s", pd.Series(dtype=float)),
            errors="coerce",
        )
        major_observed = first_major.notna()
        out[str(mode)] = {
            "attempts_observed": int(len(group)),
            "major_event_attempt_fraction": float(major_observed.mean()),
            "done_attempt_fraction": float(group.get("done_observed", pd.Series(False, index=group.index)).astype(bool).mean()),
            "reset_attempt_fraction": float(group.get("reset_observed", pd.Series(False, index=group.index)).astype(bool).mean()),
            "first_major_event_time_s": stats(first_major),
            "pose_stable_fraction_all": stats(group.get("pose_stable_fraction_all", pd.Series(dtype=float))),
            "raw_progress_along_command": stats(group.get("raw_progress_along_command", pd.Series(dtype=float))),
            "pose_stable_pre_first_major_event_progress_along_command": stats(
                group.get("pose_stable_pre_first_major_event_progress_along_command", pd.Series(dtype=float))
            ),
        }
    return out


def _terrain_frame_metrics(g: pd.DataFrame) -> dict[str, Any]:
    posture_masks = {
        "pose_stable": pose_stable_mask(g),
        "pose_stable_before_first_major_event": pose_stable_mask(g) & before_first_major_event_mask(g),
    }
    return {
        "rows": int(len(g)),
        "posture": compute_posture(g, posture_masks),
        "contact": {"touchdown_vz_abs": compute_contact(g, {})["touchdown_vz_abs"]},
        "slip": compute_slip(g, {})["stance_xy_speed"],
        "hardware": compute_hardware(g),
    }


def compute_robustness_breakdown(df: pd.DataFrame, segs: list[dict[str, Any]], timeline: list[dict[str, Any]], masks: dict[str, pd.Series], meta: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "coverage": {
            "dr_levels_observed": sorted([str(x) for x in df.get("dr_level", pd.Series([], dtype=str)).dropna().unique()]),
            "dr_columns_observed": [c for c in df.columns if c.startswith("dr_")],
            "push_frames_observed": int((bool_series(df, "push_event") | bool_series(df, "push_observed")).sum()),
            "push_events_observed": len([e for e in timeline if e.get("event_type") == "push_observed"]),
        },
        "by_dr_level": {},
        "by_friction_bucket": {},
        "by_mass_bucket": {},
        "push_windows": compute_push_windows(df),
    }
    if "dr_level" in df.columns:
        out["by_dr_level"] = _by_bucket(df, _robustness_frame_metrics, ["dr_level"])
    if "dr_friction" in df.columns:
        temp = df.copy(); temp["__friction_bucket"] = pd.cut(numeric_col(temp, "dr_friction", np.nan), bins=[-np.inf,0.5,0.8,1.2,np.inf], labels=["<=0.5","0.5-0.8","0.8-1.2",">1.2"])
        out["by_friction_bucket"] = _by_bucket(temp, _robustness_frame_metrics, ["__friction_bucket"])
    if "dr_mass" in df.columns:
        temp = df.copy(); temp["__mass_bucket"] = pd.cut(numeric_col(temp, "dr_mass", np.nan), bins=4)
        out["by_mass_bucket"] = _by_bucket(temp, _robustness_frame_metrics, ["__mass_bucket"])
    return out


def _robustness_frame_metrics(g: pd.DataFrame) -> dict[str, Any]:
    posture_masks = {
        "pose_stable": pose_stable_mask(g),
        "pose_stable_before_first_major_event": pose_stable_mask(g) & before_first_major_event_mask(g),
    }
    return {
        "rows": int(len(g)),
        "tracking_target_all_frames": tracking_stats_for_df(g, "target"),
        "posture": compute_posture(g, posture_masks),
        "slip": compute_slip(g, {})["stance_xy_speed"],
        "contact_touchdown_vz_abs": compute_contact(g, {})["touchdown_vz_abs"],
        "hardware": compute_hardware(g),
    }


def compute_push_windows(df: pd.DataFrame) -> list[dict[str, Any]]:
    if not ("push_event" in df.columns or "push_observed" in df.columns):
        return []
    push = bool_series(df, "push_event") | bool_series(df, "push_observed")
    if not push.any():
        return []
    t = numeric_col(df, "time", np.nan)
    out = []
    for idx in df[push].index:
        r = df.loc[idx]
        case = r.get("case_id", 0); env = r.get("env_id", 0); ep = r.get("episode_id", 0)
        et = float(r.get("time", np.nan))
        same = df
        if "case_id" in df.columns:
            same = same[same["case_id"] == case]
        if "env_id" in df.columns:
            same = same[same["env_id"] == env]
        if "episode_id" in df.columns:
            same = same[same["episode_id"] == ep]
        for name, lo, hi in [("pre_push_0p5s", -0.5, 0), ("post_push_0p5s", 0, 0.5), ("post_push_1p0s", 0, 1.0), ("post_push_2p0s", 0, 2.0)]:
            g = same[(numeric_col(same, "time", np.nan) >= et + lo) & (numeric_col(same, "time", np.nan) <= et + hi)]
            posture_masks = {
                "pose_stable": pose_stable_mask(g),
                "pose_stable_before_first_major_event": pose_stable_mask(g) & before_first_major_event_mask(g),
            }
            out.append({"push_time": et, "window": name, "rows": int(len(g)), "posture": compute_posture(g, posture_masks), "tracking": tracking_stats_for_df(g, "target")})
    return out


def compute_hardware(df: pd.DataFrame) -> dict[str, Any]:
    util_cols = [c for c in df.columns if c.startswith("torque_utilization_")]
    torque_cols = [c for c in df.columns if c.startswith("torque_applied_")]
    action_cols = [c for c in df.columns if c.startswith("action_mean_")]
    out: dict[str, Any] = {}
    if torque_cols:
        vals = pd.concat([numeric_col(df, c, np.nan).abs() for c in torque_cols], ignore_index=True)
        out["torque_abs"] = stats(vals, qs=(0.5,0.9,0.95,0.99,0.995))
    else:
        out["torque_abs"] = {"available": False}
    if util_cols:
        vals = pd.concat([numeric_col(df, c, np.nan).abs() for c in util_cols], ignore_index=True)
        out["torque_utilization"] = stats(vals, qs=(0.5,0.9,0.95,0.99,0.995))
    else:
        out["torque_utilization"] = {"available": False}
    if action_cols and "time" in df.columns:
        # Norm of finite-difference action rate, per row.
        a = pd.concat([numeric_col(df, c, np.nan) for c in action_cols], axis=1)
        t = numeric_col(df, "time", np.nan)
        keys = group_key_columns(df, include_segment=False)
        if keys:
            groups = [df[c] for c in keys]
            dt = t.groupby(groups, sort=False).diff().replace(0, np.nan)
            da = a.groupby(groups, sort=False).diff()
        else:
            dt = t.diff().replace(0, np.nan)
            da = a.diff()
        rate = np.sqrt((da**2).sum(axis=1)) / dt
        out["action_rate_norm"] = stats(rate.replace([np.inf,-np.inf], np.nan), qs=(0.5,0.9,0.95,0.99))
    else:
        out["action_rate_norm"] = {"available": False}
    return out


def compute_symmetry(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    duties = {}
    for leg in LEGS:
        c = f"foot_{leg}_contact"
        if c in df.columns:
            duties[leg] = float(bool_series(df, c).mean())
    if duties:
        out["per_leg_duty"] = duties
        out["per_leg_duty_range"] = float(max(duties.values()) - min(duties.values())) if len(duties) == 4 else None
        if all(k in duties for k in ["FL","FR"]): out["front_left_right_duty_diff"] = abs(duties["FL"]-duties["FR"])
        if all(k in duties for k in ["RL","RR"]): out["rear_left_right_duty_diff"] = abs(duties["RL"]-duties["RR"])
    out["mirror_equivariance_error"] = {"available": False, "reason": "requires live policy adapter and mirror map"}
    return out


def missing_field_summary(df: pd.DataFrame) -> dict[str, Any]:
    groups = {
        "torque": ["torque_applied_0", "torque_limit_0", "torque_utilization_0"],
        "action_applied": ["action_applied_0"],
        "local_terrain_height": ["foot_FL_terrain_height", "foot_FL_clearance_local", "terrain_height_source"],
        "touchdown_liftoff": ["foot_FL_touchdown", "foot_FL_liftoff", "foot_FL_touchdown_vz"],
        "dr_metadata": ["dr_level", "dr_friction", "dr_mass"],
        "push_metadata": ["push_event", "push_equivalent_delta_v"],
    }
    out = {}
    for name, cols in groups.items():
        present = [c for c in cols if c in df.columns]
        empty = [
            c for c in present
            if (
                pd.to_numeric(df[c], errors="coerce").notna().sum() == 0
                if pd.api.types.is_numeric_dtype(df[c])
                else df[c].dropna().astype(str).str.strip().eq("").all()
            )
        ]
        out[name] = {
            "missing": [c for c in cols if c not in df.columns],
            "present": present,
            "present_but_empty": empty,
        }
    return out


def _format_num(x: Any, nd=3) -> str:
    if x is None: return "n/a"
    try:
        if not np.isfinite(float(x)): return "n/a"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def render_summary(metrics: dict[str, Any]) -> str:
    lines: list[str] = []
    cov = metrics.get("coverage", {})
    lines += [
        "# IsaacLabQuadDiag Observation Summary",
        "",
        "This report describes observed behavior only. External targets are not applied in this report.",
        "",
        "## Data Coverage",
        f"- source: `{metrics.get('source')}`",
        f"- rows: {cov.get('rows')}",
        f"- cases: {cov.get('cases')}",
        f"- envs: {cov.get('envs')}",
        f"- episodes: {cov.get('episodes')}",
        f"- segments: {cov.get('segments')}",
        f"- segment_attempts: {cov.get('segment_attempts')}",
        f"- duration_s: {_format_num(cov.get('duration_s'))}",
        f"- command_modes_observed: {cov.get('command_modes_observed')}",
        f"- terrain_types_observed: {cov.get('terrain_types_observed')}",
        f"- dr_levels_observed: {cov.get('dr_levels_observed')}",
        f"- push_frames_observed: {cov.get('push_frames_observed')}",
        f"- terrain_height_sources_observed: {cov.get('terrain_height_sources_observed')}",
        "",
    ]
    notes = metrics.get("diagnostic_notes", {})
    lines += ["## Top Observed Patterns", ""]
    top = notes.get("top_observed_patterns", []) if isinstance(notes, dict) else []
    if not top:
        lines.append("- none")
    else:
        for n in top[:12]:
            lines.append(f"- [{n.get('priority','')}] {n.get('type')}: {n.get('message')}")
    lines.append("")

    lines += ["## Event Timeline Overview", "", "### Bout/Instant Counts", "```json", json.dumps(metrics.get("event_counts_by_bout_or_instant", {}), indent=2, ensure_ascii=False), "```", "", "### Frame Counts", "```json", json.dumps(metrics.get("event_frame_counts", {}), indent=2, ensure_ascii=False), "```", ""]

    # Segment table.
    lines += ["## Command Segment Event Summary", "", "| case | env | ep | seg | mode | terrain | DR | rows | first major event s | events | raw progress | stable pre-event progress | pose-stable frac |", "|---:|---:|---:|---:|---|---|---|---:|---:|---|---:|---:|---:|"]
    for s in metrics.get("segment_event_summary", [])[:80]:
        lines.append("| {case_id} | {env_id} | {episode_id} | {cmd_segment_id} | {cmd_target_mode} | {terrain_type} | {dr_level} | {rows} | {fme} | {events} | {raw} | {stablepre} | {frac} |".format(
            **s,
            fme=_format_num(s.get("first_major_event_time_since_segment_start_s")),
            events=",".join(s.get("observed_event_labels", [])) or "-",
            raw=_format_num(s.get("raw_progress_along_command")),
            stablepre=_format_num(s.get("pose_stable_pre_first_major_event_progress_along_command")),
            frac=_format_num(s.get("pose_stable_fraction_all")),
        ))
    lines.append("")

    # Key sections as JSON; concise but complete.
    for title, key in [
        ("Command Breakdown", "command_breakdown"),
        ("Tracking Slices", "tracking"),
        ("Stand", "stand"),
        ("Gait", "gait"),
        ("Contact", "contact"),
        ("Slip", "slip"),
        ("Posture", "posture"),
        ("Terrain Breakdown", "terrain"),
        ("Robustness Breakdown", "robustness"),
        ("Hardware", "hardware"),
        ("Symmetry", "symmetry"),
        ("Missing / Approximate Fields", "missing_fields"),
    ]:
        lines += [f"## {title}", "```json", json.dumps(metrics.get(key, {}), indent=2, ensure_ascii=False)[:20000], "```", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Compute observation-only diagnostics from an IsaacLabQuadDiag record.")
    p.add_argument("record", help="record.csv or record.parquet")
    p.add_argument("--out", default=None, help="output directory")
    p.add_argument("--meta", default=None, help="optional record_meta.json")
    args = p.parse_args(argv)
    metrics = compute_all_metrics(args.record, args.out, args.meta)
    if args.out is None:
        print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
