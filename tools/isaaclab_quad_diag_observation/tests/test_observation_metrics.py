from __future__ import annotations

import numpy as np
import pandas as pd

from isaaclab_quad_diag.metrics import compute_metrics
from isaaclab_quad_diag.events import add_derived_event_columns, extract_event_timeline, segment_event_summary
from isaaclab_quad_diag.schema import DiagnosticThresholds


def make_record():
    rows = []
    dt = 0.02
    # backward segment: early large pitch after 0.08s, then reset-like done.
    for i in range(50):
        t = i * dt
        pitch = 0 if i < 4 else 48
        rows.append({
            "env_id": 0, "episode_id": 0, "cmd_segment_id": 0, "step": i, "time": t,
            "cmd_target_mode": "backward", "cmd_target_vx": -0.5, "cmd_target_vy": 0, "cmd_target_wz": 0,
            "cmd_mode": "backward", "cmd_vx": -0.5, "cmd_vy": 0, "cmd_wz": 0,
            "base_lin_vel_b_x": -0.6, "base_lin_vel_b_y": 0.0, "base_ang_vel_b_z": 0,
            "base_roll": 0, "base_pitch": pitch, "base_height_local": 0.55 if i < 45 else 0.3,
            "nominal_stand_height": 0.55,
            "done": 1 if i == 49 else 0,
            "terrain_type": "flat", "terrain_level": 0, "dr_level": 0,
            "foot_FL_contact": 1, "foot_FR_contact": 1, "foot_RL_contact": 1, "foot_RR_contact": 1,
            "foot_FL_stance_slip_xy": 0.1, "foot_FR_stance_slip_xy": 0.1, "foot_RL_stance_slip_xy": 0.1, "foot_RR_stance_slip_xy": 0.1,
            "foot_FL_touchdown_vz": np.nan, "foot_FR_touchdown_vz": np.nan, "foot_RL_touchdown_vz": np.nan, "foot_RR_touchdown_vz": np.nan,
            "torque_utilization_0": 0.2,
        })
    # lateral segment starts in next episode; no large event.
    for i in range(50):
        t = 1.0 + i * dt
        rows.append({
            "env_id": 0, "episode_id": 1, "cmd_segment_id": 1, "step": 50+i, "time": t,
            "cmd_target_mode": "lateral", "cmd_target_vx": 0, "cmd_target_vy": 0.3, "cmd_target_wz": 0,
            "cmd_mode": "lateral", "cmd_vx": 0, "cmd_vy": 0.3, "cmd_wz": 0,
            "base_lin_vel_b_x": 0, "base_lin_vel_b_y": 0.2, "base_ang_vel_b_z": 0,
            "base_roll": 0, "base_pitch": 0, "base_height_local": 0.55,
            "nominal_stand_height": 0.55,
            "done": 0,
            "terrain_type": "flat", "terrain_level": 0, "dr_level": 0,
            "foot_FL_contact": 1, "foot_FR_contact": 1, "foot_RL_contact": 0, "foot_RR_contact": 0,
            "foot_FL_stance_slip_xy": 0.02, "foot_FR_stance_slip_xy": 0.02, "foot_RL_stance_slip_xy": np.nan, "foot_RR_stance_slip_xy": np.nan,
            "foot_FL_touchdown_vz": np.nan, "foot_FR_touchdown_vz": np.nan, "foot_RL_touchdown_vz": np.nan, "foot_RR_touchdown_vz": np.nan,
            "torque_utilization_0": 0.2,
        })
    return pd.DataFrame(rows)


def test_event_bout_count_not_frame_count():
    df = make_record()
    th = DiagnosticThresholds()
    d = add_derived_event_columns(df, th)
    timeline = extract_event_timeline(d, th)
    pitch_bouts = [e for e in timeline if e["event_type"] == "large_pitch_bout"]
    assert len(pitch_bouts) == 1
    assert pitch_bouts[0]["frame_count"] > 1


def test_first_major_event_and_stable_progress():
    df = make_record()
    th = DiagnosticThresholds()
    timeline = extract_event_timeline(df, th)
    segs = segment_event_summary(df, timeline, th)
    back = [s for s in segs if s["cmd_target_mode"] == "backward"][0]
    assert abs(back["first_major_event_time_since_segment_start_s"] - 0.08) < 1e-6
    assert back["raw_progress_along_command"] > back["pose_stable_pre_first_major_event_progress_along_command"]


def test_metrics_outputs_neutral_names():
    m = compute_metrics(make_record())
    assert "segment_event_summary" in m
    assert m["command_breakdown"]["backward"]["major_event_attempt_fraction"] == 1.0
    assert m["command_breakdown"]["lateral"]["major_event_attempt_fraction"] == 0.0
    text = str(m).lower()
    assert "failed_segments" not in text
    assert "failure_reason" not in text
    assert "pass_rate" not in text
    assert "event_timeline" not in m  # written as separate file, not embedded by default


def test_reset_inside_segment_remains_one_command_attempt():
    rows = []
    for episode_id, start in [(0, 0.0), (1, 0.04)]:
        for i in range(2):
            rows.append({
                "case_id": 0,
                "env_id": 0,
                "episode_id": episode_id,
                "cmd_segment_id": 7,
                "step": len(rows),
                "time": start + i * 0.02,
                "cmd_target_mode": "yaw",
                "cmd_target_vx": 0.0,
                "cmd_target_vy": 0.0,
                "cmd_target_wz": 0.6,
                "base_lin_vel_b_x": 0.0,
                "base_lin_vel_b_y": 0.0,
                "base_roll": 0.0,
                "base_pitch": 0.0,
                "base_height_local": 0.5,
                "nominal_stand_height": 0.5,
                "done": int(episode_id == 0 and i == 1),
                "reset_observed": int(episode_id == 0 and i == 1),
            })
    metrics = compute_metrics(pd.DataFrame(rows))
    yaw = metrics["command_breakdown"]["yaw"]
    assert yaw["attempts_observed"] == 1
    assert yaw["major_event_attempt_fraction"] == 1.0
    assert metrics["coverage"]["segment_attempts"] == 1
    assert metrics["coverage"]["episode_segment_fragments"] == 2


def test_push_notices_only_when_push_was_requested():
    record = make_record()
    no_push = compute_metrics(record, record_meta={"requested_suite_config": {}})
    no_push_types = {n["type"] for n in no_push["diagnostic_notes"]["data_quality_notices"]}
    assert "missing_push_fields" not in no_push_types
    assert "no_push_observed" not in no_push_types

    requested = compute_metrics(
        record,
        record_meta={"requested_suite_config": {"pushes": {"enabled": True, "events": [{"time": 0.5}]}}},
    )
    requested_types = {n["type"] for n in requested["diagnostic_notes"]["data_quality_notices"]}
    assert "missing_push_fields" in requested_types
    assert "no_push_observed" in requested_types


def test_stand_only_record_has_empty_gait_without_crashing():
    record = make_record()
    for column in ["cmd_target_vx", "cmd_target_vy", "cmd_target_wz", "cmd_vx", "cmd_vy", "cmd_wz"]:
        record[column] = 0.0
    metrics = compute_metrics(record)
    assert metrics["gait"]["rows"] == 0
    assert (
        "diagonal_duty_diff" not in metrics["gait"]
        or metrics["gait"]["diagonal_duty_diff"] is None
    )
