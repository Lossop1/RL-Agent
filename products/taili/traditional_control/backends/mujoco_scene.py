"""Build an ephemeral nominal MuJoCo scene from the tracked Taili URDF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import TailiControllerBundle


@dataclass
class MujocoNominalScene:
    model: object
    data: object
    scenario: str
    policy_dt: float


def build_nominal_scene(bundle: TailiControllerBundle, scenario: str) -> MujocoNominalScene:
    """Create a model with no hidden RL assets or generated source files."""

    import mujoco

    profile = bundle.profile
    config = bundle.config
    spec = mujoco.MjSpec.from_file(str(profile.urdf_path))
    root = spec.worldbody.bodies[0]
    root.add_freejoint()
    spec.option.timestep = float(config["simulation"]["physics_dt"])
    # Keep the nominal base height as the profile's physical reference.  The
    # URDF feet differ by less than a millimetre, so align the plane with the
    # highest nominal sole.  This is scene geometry, not a controller offset.
    nominal_foot_z = float(np.max(bundle.kinematics.nominal_foot_positions()[:, 2]))
    sole_offset = profile.nominal_base_height + nominal_foot_z - profile.foot_radius
    terrain_cfg = config["scenarios"][scenario]["terrain"]
    terrain_base = float(terrain_cfg.get("base_height", terrain_cfg.get("height", 0.0)))
    ground_level = terrain_base + sole_offset
    spec.worldbody.add_geom(
        name="nominal_ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[20.0, 20.0, 0.1],
        pos=[0.0, 0.0, ground_level],
        friction=[1.0, 0.005, 0.0001],
    )

    terrain = bundle.controller.terrain
    if hasattr(terrain, "rise"):
        for index in range(int(terrain.max_steps)):
            height = float(ground_level + (index + 1) * terrain.rise)
            center_x = float(terrain.start_x + (index + 0.5) * terrain.run)
            spec.worldbody.add_geom(
                name=f"nominal_step_{index:03d}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[center_x, 0.0, (ground_level + height) * 0.5],
                size=[float(terrain.run) * 0.5, float(terrain.width) * 0.5, (height - ground_level) * 0.5],
                friction=[1.0, 0.005, 0.0001],
            )
        # The staircase is a course, not an isolated set of risers.  Add a
        # landing at the top so the down-course spawn and the up-course exit
        # remain on the intended support surface.
        top_height = ground_level + float(terrain.max_steps) * float(terrain.rise)
        landing_start = float(terrain.start_x + terrain.max_steps * terrain.run)
        spec.worldbody.add_geom(
            name="nominal_upper_landing",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[landing_start + 0.5, 0.0, 0.5 * (ground_level + top_height)],
            size=[0.5, float(terrain.width) * 0.5, 0.5 * (top_height - ground_level)],
            friction=[1.0, 0.005, 0.0001],
        )

    for name, effort in zip(profile.joint_names, profile.effort_limit):
        spec.add_actuator(
            name=name.removesuffix("_joint"),
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target=name,
            gear=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ctrllimited=True,
            ctrlrange=[-float(effort), float(effort)],
        )

    model = spec.compile()
    data = mujoco.MjData(model)
    scenario_cfg = config["scenarios"][scenario]
    terrain_cfg = scenario_cfg["terrain"]
    if terrain_cfg.get("type") == "stairs" and scenario == "stairs_down":
        start_x = float(terrain_cfg["start_x"]) + float(terrain_cfg["run"]) * int(terrain_cfg["max_steps"]) + 0.45
        start_height = float(terrain_cfg.get("base_height", 0.0)) + float(terrain_cfg["rise"]) * int(terrain_cfg["max_steps"])
    elif terrain_cfg.get("type") == "stairs":
        start_x = float(terrain_cfg["start_x"]) - 0.45
        start_height = float(terrain_cfg.get("base_height", 0.0))
    else:
        start_x = 0.0
        start_height = float(terrain_cfg.get("height", 0.0))
    data.qpos[0:3] = [start_x, 0.0, profile.nominal_base_height + start_height]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for name, value in zip(profile.joint_names, profile.q_default):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint_id]] = float(value)
    mujoco.mj_forward(model, data)
    return MujocoNominalScene(model, data, scenario, float(config["simulation"]["policy_dt"]))
