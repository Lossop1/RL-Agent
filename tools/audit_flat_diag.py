#!/usr/bin/env python3
"""Summarize raw IsaacLab flat-diagnostic physics by command segment."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict


LEGS = ("FL", "FR", "RL", "RR")
NOMINAL_HEIGHT = 0.5471961541


def number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo = int(math.floor(index))
    hi = int(math.ceil(index))
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "p95": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def posture(row: dict[str, str]) -> tuple[float, float]:
    gx = number(row, "projected_gravity_b_x")
    gy = number(row, "projected_gravity_b_y")
    gz = number(row, "projected_gravity_b_z", -1.0)
    roll = math.degrees(math.atan2(gy, -gz))
    pitch = math.degrees(math.atan2(-gx, math.sqrt(gy * gy + gz * gz)))
    return abs(roll), abs(pitch)


def action_delta(rows: list[dict[str, str]]) -> dict[int, float]:
    result: dict[int, float] = {}
    last: list[float] | None = None
    for index, row in enumerate(rows):
        action = [number(row, f"action_applied_{joint}") for joint in range(12)]
        if last is not None:
            result[index] = math.sqrt(sum((a - b) ** 2 for a, b in zip(action, last)) / 12.0)
        last = action
    return result


def analyze(rows: list[dict[str, str]], label: str) -> dict[str, object]:
    if not rows:
        return {"label": label, "rows": 0}

    deltas = action_delta(rows)
    arrays: dict[str, list[float]] = defaultdict(list)
    foot_contact: dict[str, list[float]] = {leg: [] for leg in LEGS}
    touchdown_count = 0
    liftoff_count = 0
    for index, row in enumerate(rows):
        vx = number(row, "base_lin_vel_b_x")
        vy = number(row, "base_lin_vel_b_y")
        vz = number(row, "base_lin_vel_b_z")
        wx = number(row, "base_ang_vel_b_x")
        wy = number(row, "base_ang_vel_b_y")
        wz = number(row, "base_ang_vel_b_z")
        roll, pitch = posture(row)
        arrays["speed_xy"].append(math.hypot(vx, vy))
        arrays["vz_abs"].append(abs(vz))
        arrays["wxy"].append(math.hypot(wx, wy))
        arrays["wz_abs"].append(abs(wz))
        arrays["roll_deg_abs"].append(roll)
        arrays["pitch_deg_abs"].append(pitch)
        arrays["tilt_deg"].append(math.hypot(roll, pitch))
        height = number(row, "base_height_local")
        arrays["height_m"].append(height)
        arrays["height_error_m"].append(abs(height - NOMINAL_HEIGHT))
        arrays["action_delta_rms"].append(deltas.get(index, 0.0))
        hip_errors = [abs(number(row, f"joint_error_{joint}")) for joint in range(4)]
        arrays["hip_target_error_rad"].append(max(hip_errors))
        arrays["hip_target_error_mean_rad"].append(statistics.fmean(hip_errors))
        all_errors = [abs(number(row, f"joint_error_{joint}")) for joint in range(12)]
        arrays["joint_target_error_rad"].append(max(all_errors))
        joint_vel = [number(row, f"joint_vel_{joint}") for joint in range(12)]
        arrays["joint_velocity_rms"].append(math.sqrt(sum(value * value for value in joint_vel) / 12.0))
        per_leg = [
            math.sqrt(sum(joint_vel[offset] ** 2 for offset in (leg, leg + 4, leg + 8)) / 3.0)
            for leg in range(4)
        ]
        arrays["joint_velocity_worst_leg_rms"].append(max(per_leg))
        torque_util = [abs(number(row, f"torque_utilization_{joint}")) for joint in range(12)]
        arrays["torque_utilization"].append(max(torque_util))

        for leg in LEGS:
            contact = number(row, f"foot_{leg}_contact")
            foot_contact[leg].append(contact)
            fx = number(row, f"foot_{leg}_vel_w_x")
            fy = number(row, f"foot_{leg}_vel_w_y")
            fz = number(row, f"foot_{leg}_vel_w_z")
            arrays["foot_speed_3d"].append(math.sqrt(fx * fx + fy * fy + fz * fz))
            if contact > 0.5:
                arrays["stance_slip_xy"].append(number(row, f"foot_{leg}_stance_slip_xy"))
            if number(row, f"foot_{leg}_touchdown") > 0.5:
                touchdown_count += 1
                arrays["touchdown_vz_abs"].append(abs(number(row, f"foot_{leg}_touchdown_vz")))
            if number(row, f"foot_{leg}_liftoff") > 0.5:
                liftoff_count += 1

        cmd_x = number(row, "cmd_target_vx")
        cmd_y = number(row, "cmd_target_vy")
        cmd_w = number(row, "cmd_target_wz")
        cmd_xy = math.hypot(cmd_x, cmd_y)
        if cmd_xy > 1.0e-6:
            along = (vx * cmd_x + vy * cmd_y) / cmd_xy
            off_axis = abs(-vx * cmd_y + vy * cmd_x) / cmd_xy
            arrays["tracking_xy_error_mps"].append(math.hypot(vx - cmd_x, vy - cmd_y))
            arrays["progress_ratio"].append(along / cmd_xy)
            arrays["off_axis_speed_mps"].append(off_axis)
        if abs(cmd_w) > 1.0e-6:
            arrays["tracking_yaw_error_rps"].append(abs(wz - cmd_w))
            arrays["yaw_planar_leak_mps"].append(math.hypot(vx, vy))

    result: dict[str, object] = {"label": label, "rows": len(rows)}
    for key, values in arrays.items():
        result[key] = summary(values)
    result["height_span_m"] = max(arrays["height_m"]) - min(arrays["height_m"])
    duties = {leg: statistics.fmean(values) if values else 0.0 for leg, values in foot_contact.items()}
    result["duty"] = duties
    result["duty_range"] = max(duties.values()) - min(duties.values())
    result["touchdown_count"] = touchdown_count
    result["liftoff_count"] = liftoff_count
    return result


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} RECORD.csv")
    with open(sys.argv[1], newline="", encoding="utf-8") as source:
        all_rows = list(csv.DictReader(source))
    by_mode: dict[str, list[dict[str, str]]] = defaultdict(list)
    braking: list[dict[str, str]] = []
    for row in all_rows:
        mode = row.get("cmd_target_mode", "unknown")
        elapsed = number(row, "time_since_command_switch")
        if mode == "stand":
            if elapsed >= 0.60:
                by_mode["stand_settled"].append(row)
            if number(row, "cmd_segment_id") > 0.0 and elapsed <= 0.60:
                braking.append(row)
        elif elapsed >= 0.50:
            by_mode[mode].append(row)
    output = {"file": sys.argv[1], "all_rows": len(all_rows)}
    output["braking_window"] = analyze(braking, "braking_window")
    output["segments"] = {mode: analyze(rows, mode) for mode, rows in sorted(by_mode.items())}
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
