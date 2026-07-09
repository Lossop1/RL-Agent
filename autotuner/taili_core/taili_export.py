"""Deployment export manifest — strict per taili_strategy_decisions.md
(Runtime IO Contract v1 "export manifest" + TerrainPerceiver Contract export).

Everything the real-robot side needs to reproduce the training obs BIT-FOR-BIT. Values are
pulled from the single-source modules (taili_symmetry, taili_amp_reference) — NOT re-hardcoded
— so a drift in a constant fails the manifest consistency test. Weights are saved separately
(this is the metadata/recipe). real_to_canonical_map is robot-specific (adapter fills it).
"""
from __future__ import annotations

from . import taili_symmetry as sym
from . import taili_amp_reference as ref

# pinned Runtime IO constants (verified remote URDF/asset; see strategy doc)
DT_POLICY = 0.02
DECIMATION = 4
PHYSICS_DT = 0.005
ACTION_SCALE = 0.35
CMD_SMOOTH_ALPHA = 0.93
HISTORY_LEN = 25
NOMINAL_BASE_H = 0.52
EFFORT_LIMIT = {"hip": 320.0, "thigh": 110.0, "calf": 220.0}

# ordered layouts (field -> width); each must sum to its declared total
BODY53_LAYOUT = [("angvel", 3), ("gravity", 3), ("command", 3), ("jpos", 12),
                 ("jvel", 12), ("last_action", 12), ("gait_clock", 8)]
TICK54_LAYOUT = [("q_rel", 12), ("dq", 12), ("q_des_rel", 12), ("q_error", 12),
                 ("gyro_body", 3), ("projected_gravity", 3)]
AMP_FRAME51_LAYOUT = [("jp", 12), ("jv", 12), ("base_height", 1), ("tangent_normal", 6),
                      ("foot_rel", 12), ("command", 3), ("mode_onehot", 5)]
GAIT_CLOCK_LAYOUT = ["sin_FL", "sin_FR", "sin_RL", "sin_RR", "cos_FL", "cos_FR", "cos_RL", "cos_RR"]


def build_export_manifest():
    """Return the deployment recipe (constants + layouts + mirror metadata)."""
    return {
        "timing": {"dt_policy": DT_POLICY, "decimation": DECIMATION, "physics_dt": PHYSICS_DT,
                   "history_len": HISTORY_LEN},
        "constants": {
            "q_default": list(ref.Q_DEFAULT12),
            "action_scale": ACTION_SCALE,
            "cmd_smooth_alpha": CMD_SMOOTH_ALPHA,
            "nominal_base_h": NOMINAL_BASE_H,
            "base_height_ref": ref.BASE_HEIGHT_REF,
            "effort_limit": EFFORT_LIMIT,
        },
        "q_des_recipe": "q_des = q_default + action_scale * last_action",
        "q_error_recipe": "q_error = q_des - q_current (servo lag)",
        "command_recipe": "command_effective += (1 - cmd_smooth_alpha) * (cmd_target - command_effective)",
        "joint_order": {
            "canonical": list(sym.CANONICAL_JOINT_ORDER),
            "real_to_canonical_map": None,   # adapter fills: real SDK order -> canonical index
        },
        "layouts": {
            "body53": BODY53_LAYOUT,
            "tick54": TICK54_LAYOUT,
            "amp_frame51": AMP_FRAME51_LAYOUT,
            "gait_clock": GAIT_CLOCK_LAYOUT,
        },
        "frames": {
            "imu": "IsaacLab body frame; gyro=root_ang_vel_b; "
                   "projected_gravity=gravity@body, upright ~= (0,0,-1)",
            "command": "body frame [vx fwd, vy left, wz yaw]",
        },
        "mirror_metadata": {
            "JP": list(sym.JP), "JS": list(sym.JS),
            "GYRO_S": list(sym.GYRO_S), "GRAV_S": list(sym.GRAV_S), "CMD_S": list(sym.CMD_S),
            "GAIT_P": list(sym.GAIT_P), "z_mirror": "swap_halves(z_terrain[32])",
        },
        "deploy_note": "actor + TerrainPerceiver weights saved separately; aux heads NOT deployed.",
    }
