from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

LEGS = ["FL", "FR", "RL", "RR"]

CORE_COLUMNS = [
    "case_id", "env_id", "episode_id", "step", "time", "cmd_segment_id",
    "cmd_target_vx", "cmd_target_vy", "cmd_target_wz", "cmd_target_mode",
    "cmd_vx", "cmd_vy", "cmd_wz", "cmd_mode",
    "base_pos_w_x", "base_pos_w_y", "base_pos_w_z",
    "base_quat_w", "base_quat_x", "base_quat_y", "base_quat_z",
    "base_lin_vel_b_x", "base_lin_vel_b_y", "base_lin_vel_b_z",
    "base_ang_vel_b_x", "base_ang_vel_b_y", "base_ang_vel_b_z",
]

OPTIONAL_GROUPS = {
    "base_height": ["base_height_local", "base_terrain_height"],
    "events": ["terminated", "truncated", "done", "fall_flag"],
    "terrain": ["terrain_type", "terrain_level", "terrain_params"],
    "dr": ["dr_level", "dr_mass", "dr_friction", "dr_com_x", "dr_com_y", "dr_com_z", "dr_latency"],
    "push": ["push_event", "push_vector_x", "push_vector_y", "push_vector_z", "push_equivalent_delta_v"],
}


def joint_columns(n: int = 12) -> list[str]:
    cols: list[str] = []
    for i in range(n):
        cols += [
            f"joint_pos_{i}", f"joint_vel_{i}", f"joint_pos_des_{i}", f"joint_error_{i}",
            f"action_mean_{i}", f"action_applied_{i}", f"torque_cmd_{i}",
            f"torque_applied_{i}", f"torque_limit_{i}", f"torque_utilization_{i}",
        ]
    return cols


def foot_columns(legs: Iterable[str] = LEGS) -> list[str]:
    cols: list[str] = []
    for leg in legs:
        cols += [
            f"foot_{leg}_pos_w_x", f"foot_{leg}_pos_w_y", f"foot_{leg}_pos_w_z",
            f"foot_{leg}_vel_w_x", f"foot_{leg}_vel_w_y", f"foot_{leg}_vel_w_z",
            f"foot_{leg}_terrain_height", f"foot_{leg}_clearance_local",
            f"foot_{leg}_contact", f"foot_{leg}_normal_force", f"foot_{leg}_tangent_force",
            f"foot_{leg}_force_w_x", f"foot_{leg}_force_w_y", f"foot_{leg}_force_w_z", f"foot_{leg}_force_norm",
            f"foot_{leg}_air_time", f"foot_{leg}_stance_time",
            f"foot_{leg}_touchdown", f"foot_{leg}_liftoff", f"foot_{leg}_touchdown_vz", f"foot_{leg}_stance_slip_xy",
        ]
    return cols


@dataclass(frozen=True)
class DiagnosticThresholds:
    """Thresholds used to extract noteworthy events/slices, not acceptance criteria."""

    command_settle_s: float = 0.30
    event_window_s: float = 0.50
    large_roll_deg: float = 35.0
    large_pitch_deg: float = 35.0
    height_ratio_low: float = 0.65
    hard_impact_vz: float = 0.40
    high_slip_xy: float = 0.50
    torque_spike_util: float = 0.95
    torque_clamp_util: float = 0.999


def expected_schema_summary() -> dict:
    return {
        "core_columns": CORE_COLUMNS,
        "optional_groups": OPTIONAL_GROUPS,
        "joint_columns_12dof": joint_columns(12),
        "foot_columns": foot_columns(),
    }
