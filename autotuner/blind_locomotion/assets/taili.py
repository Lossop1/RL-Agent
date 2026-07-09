"""Package-local Taili asset configuration.

The deployable runtime keeps the IsaacLab articulation config inside the
`taili_blind_runtime` package and resolves the URDF from the package payload.
"""
from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg


def _taili_urdf_path() -> str:
    asset_dir = Path(__file__).resolve().parent
    candidates = [
        asset_dir / "robots" / "taili-dog" / "robot.urdf",
        Path(__file__).resolve().parents[3] / "assets" / "robots" / "taili-dog" / "robot.urdf",
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    raise FileNotFoundError("Taili URDF not found in runtime package or repository assets")


TAILI_DOG_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=False,
        asset_path=_taili_urdf_path(),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.52),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": 0.0,
            "F.*_thigh_joint": 0.7,
            "R.*_thigh_joint": 0.7,
            "F.*_calf_joint": -1.4,
            "R.*_calf_joint": -1.4,
        },
    ),
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit={
                ".*_hip_joint": 320,
                ".*_thigh_joint": 110,
                ".*_calf_joint": 220,
            },
            velocity_limit={
                ".*_hip_joint": 13.1,
                ".*_thigh_joint": 16.55,
                ".*_calf_joint": 8.27,
            },
            saturation_effort=320.0,
            stiffness=120.0,
            damping=10.0,
        ),
    },
)
