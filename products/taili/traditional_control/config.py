"""Configuration-driven construction of the Taili nominal controller."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from autotuner.control import (
    CentroidalMpc,
    CentroidalMpcConfig,
    ContactTerrainEstimator,
    ContinuousTrotGait,
    FloatingBaseWbc,
    FlatTerrain,
    GaitConfig,
    MpcWbcController,
    StairTerrain,
    WholeBodyConfig,
)

from .kinematics import TailiKinematics
from .profile import TailiProfile, load_taili_profile


@dataclass(frozen=True)
class TailiControllerBundle:
    controller: MpcWbcController
    profile: TailiProfile
    kinematics: TailiKinematics
    terrain_name: str
    config: dict[str, Any]


def default_taili_control_config() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "traditional_control" / "taili_nominal.yaml"


def load_taili_control_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or default_taili_control_config())
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[3] / config_path
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if data.get("schema_version") != "taili_traditional_control_v1":
        raise ValueError("unsupported Taili traditional-control configuration")
    return data


def _terrain(config: dict[str, Any], name: str):
    scene = config.get("scenarios", {}).get(name)
    if not isinstance(scene, dict):
        raise ValueError(f"unknown Taili traditional-control scenario: {name}")
    terrain = scene.get("terrain", {})
    if terrain.get("type") == "flat":
        return FlatTerrain(float(terrain.get("height", 0.0)))
    if terrain.get("type") == "stairs":
        return StairTerrain(
            rise=float(terrain["rise"]),
            run=float(terrain["run"]),
            start_x=float(terrain.get("start_x", 0.0)),
            width=float(terrain.get("width", 2.0)),
            base_height=float(terrain.get("base_height", 0.0)),
            max_steps=int(terrain.get("max_steps", 100)),
        )
    raise ValueError(f"unsupported terrain type for scenario {name}")


def build_taili_controller(
    scenario: str = "flat",
    config_path: str | Path | None = None,
) -> TailiControllerBundle:
    config = load_taili_control_config(config_path)
    profile = load_taili_profile(config)
    kinematics = TailiKinematics(profile)
    nominal_feet = kinematics.nominal_foot_positions()
    # A stair scene may deliberately trade position catch-up for bounded
    # support demand.  Keep the controller class unchanged; only the
    # scenario profile may override its numeric MPC configuration.
    mpc_cfg = {
        **config["mpc"],
        **config["scenarios"][scenario].get("mpc", {}),
    }
    mpc = CentroidalMpc(
        CentroidalMpcConfig(
            horizon=int(mpc_cfg["horizon"]),
            dt=float(mpc_cfg["dt"]),
            mass_kg=profile.mass_kg,
            inertia_kg_m2=tuple(float(x) for x in mpc_cfg["inertia_kg_m2"]),
            state_weight=tuple(float(x) for x in mpc_cfg["state_weight"]),
            terminal_weight=tuple(float(x) for x in mpc_cfg["terminal_weight"]),
            input_weight=tuple(float(x) for x in mpc_cfg["input_weight"]),
            max_acceleration=tuple(float(x) for x in mpc_cfg["max_acceleration"]),
            max_position_error=float(mpc_cfg["max_position_error"]),
            max_yaw_error=float(mpc_cfg["max_yaw_error"]),
        )
    )
    gait_cfg = {
        **config["gait"],
        **config["scenarios"][scenario].get("gait", {}),
    }
    gait = ContinuousTrotGait(
        nominal_feet,
        GaitConfig(
            period=float(gait_cfg["period"]),
            period_min=float(gait_cfg["period_min"]),
            period_slope=float(gait_cfg["period_slope"]),
            yaw_speed_equivalent=float(gait_cfg["yaw_speed_equivalent"]),
            duty_factor=float(gait_cfg["duty_factor"]),
            swing_clearance=float(gait_cfg["swing_clearance"]),
            foot_radius=profile.foot_radius,
            command_ramp_time=float(gait_cfg["command_ramp_time"]),
            contact_blend_phase=float(gait_cfg["contact_blend_phase"]),
            foothold_velocity_gain=float(gait_cfg["foothold_velocity_gain"]),
            foothold_yaw_rate_gain=float(gait_cfg["foothold_yaw_rate_gain"]),
            max_foothold_correction=tuple(float(x) for x in gait_cfg["max_foothold_correction"]),
            start_phase=float(gait_cfg.get("start_phase", 0.0)),
            lateral_start_phase=float(gait_cfg["lateral_start_phase"]),
            height_transition_threshold=float(
                gait_cfg["height_transition_threshold"]
            ),
            obstacle_lift_delay=float(gait_cfg["obstacle_lift_delay"]),
            stair_max_swing_legs=int(
                gait_cfg.get("stair_max_swing_legs", 1)
            ),
            max_swing_legs=(
                None
                if gait_cfg.get("max_swing_legs") is None
                else int(gait_cfg["max_swing_legs"])
            ),
        ),
    )
    wbc_cfg = {
        **config["wbc"],
        **config["scenarios"][scenario].get("wbc", {}),
    }
    wbc = FloatingBaseWbc(
        profile.q_default,
        profile.q_lower,
        profile.q_upper,
        WholeBodyConfig(
            dt=float(wbc_cfg["dt"]),
            base_acceleration_weight=tuple(float(x) for x in wbc_cfg["base_acceleration_weight"]),
            base_acceleration_limit=tuple(float(x) for x in wbc_cfg["base_acceleration_limit"]),
            swing_acceleration_weight=tuple(
                float(x) for x in wbc_cfg["swing_acceleration_weight"]
            ),
            height_transition_vertical_weight=float(
                wbc_cfg["height_transition_vertical_weight"]
            ),
            contact_acquisition_horizontal_weight=float(
                wbc_cfg["contact_acquisition_horizontal_weight"]
            ),
            contact_acquisition_vertical_weight=float(wbc_cfg["contact_acquisition_vertical_weight"]),
            contact_acquisition_downward_acceleration=float(
                wbc_cfg.get("contact_acquisition_downward_acceleration", 0.8)
            ),
            contact_acquisition_posture_acceleration_weight=float(
                wbc_cfg.get("contact_acquisition_posture_acceleration_weight", 4.0)
            ),
            contact_acquisition_body_height_error_threshold=float(
                wbc_cfg.get("contact_acquisition_body_height_error_threshold", 0.03)
            ),
            posture_acceleration_weight=float(wbc_cfg["posture_acceleration_weight"]),
            generalized_acceleration_regularization=float(wbc_cfg["generalized_acceleration_regularization"]),
            contact_force_regularization=float(wbc_cfg["contact_force_regularization"]),
            joint_torque_regularization=float(wbc_cfg["joint_torque_regularization"]),
            height_position_gain=float(wbc_cfg["height_position_gain"]),
            height_velocity_gain=float(wbc_cfg["height_velocity_gain"]),
            orientation_position_gain=float(wbc_cfg["orientation_position_gain"]),
            orientation_velocity_gain=float(wbc_cfg["orientation_velocity_gain"]),
            swing_position_gain=float(wbc_cfg["swing_position_gain"]),
            swing_velocity_gain=float(wbc_cfg["swing_velocity_gain"]),
            max_swing_acceleration=float(wbc_cfg["max_swing_acceleration"]),
            posture_position_gain=float(wbc_cfg["posture_position_gain"]),
            posture_velocity_gain=float(wbc_cfg["posture_velocity_gain"]),
            posture_acceleration_limit=tuple(float(x) for x in wbc_cfg["posture_acceleration_limit"]),
            contact_friction=float(wbc_cfg["contact_friction"]),
            minimum_contact_normal_force=float(
                wbc_cfg.get("minimum_contact_normal_force", 0.0)
            ),
            max_contact_normal_force=float(wbc_cfg["max_contact_normal_force"]),
            osqp_absolute_tolerance=float(wbc_cfg["osqp_absolute_tolerance"]),
            osqp_relative_tolerance=float(wbc_cfg["osqp_relative_tolerance"]),
            osqp_max_iterations=int(wbc_cfg["osqp_max_iterations"]),
            kp=tuple(profile.kp),
            kd=tuple(profile.kd),
            effort_limit=tuple(profile.effort_limit),
            velocity_limit=tuple(profile.velocity_limit),
        ),
    )
    return TailiControllerBundle(
        controller=MpcWbcController(
            mpc,
            gait,
            wbc,
            _terrain(config, scenario),
            profile.nominal_base_height,
            ContactTerrainEstimator(
                alpha=float(config["contact"]["height_filter_alpha"]),
                force_threshold=float(config["contact"]["normal_force_threshold"]),
                max_correction=float(config["contact"]["height_max_correction"]),
                minimum_support_contacts=int(
                    {
                        **config["contact"],
                        **config["scenarios"][scenario].get("contact", {}),
                    }["minimum_support_contacts"]
                ),
                support_level_tolerance=float(
                    config["contact"]["support_level_tolerance"]
                ),
            ),
            profile.foot_radius,
        ),
        profile=profile,
        kinematics=kinematics,
        terrain_name=scenario,
        config=config,
    )
