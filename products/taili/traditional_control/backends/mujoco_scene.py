"""Build an ephemeral nominal MuJoCo scene from the tracked Taili URDF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from autotuner.control import FlatTerrain, StairTerrain

from ..config import TailiControllerBundle
from ..kinematics import TailiKinematics
from ..profile import TailiProfile, load_taili_profile


@dataclass
class MujocoNominalScene:
    model: object
    data: object
    scenario: str
    policy_dt: float


@dataclass(frozen=True)
class MujocoScenePrimitive:
    """Full-size scene geometry shared by MuJoCo and playback adapters."""

    id: str
    type: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]


def terrain_model_from_config(config: Mapping[str, Any], scenario: str):
    """Build the product terrain model without importing a simulator."""

    scenarios = config.get("scenarios")
    scene = scenarios.get(scenario) if isinstance(scenarios, Mapping) else None
    if not isinstance(scene, Mapping):
        raise ValueError(f"unknown traditional-control scenario: {scenario}")
    terrain = scene.get("terrain")
    terrain_cfg = terrain if isinstance(terrain, Mapping) else {}
    terrain_type = str(terrain_cfg.get("type") or "")
    if terrain_type == "flat":
        return FlatTerrain(float(terrain_cfg.get("height", 0.0)))
    if terrain_type == "stairs":
        return StairTerrain(
            rise=float(terrain_cfg["rise"]),
            run=float(terrain_cfg["run"]),
            start_x=float(terrain_cfg.get("start_x", 0.0)),
            width=float(terrain_cfg.get("width", 2.0)),
            base_height=float(terrain_cfg.get("base_height", 0.0)),
            max_steps=int(terrain_cfg.get("max_steps", 100)),
        )
    raise ValueError(f"unsupported terrain type for scenario {scenario}: {terrain_type!r}")


def nominal_terrain_primitives_from_config(
    config: Mapping[str, Any],
    scenario: str,
    *,
    profile: TailiProfile | None = None,
    kinematics: TailiKinematics | None = None,
    terrain: Any | None = None,
) -> tuple[MujocoScenePrimitive, ...]:
    """Describe the configured course for both simulation and playback.

    The helper deliberately depends only on the product URDF/configuration and
    control terrain model.  A replay loader can therefore reconstruct scene
    geometry even when the simulator package is not installed.
    """

    profile = profile or load_taili_profile(config)
    kinematics = kinematics or TailiKinematics(profile)
    scenarios = config.get("scenarios")
    scene = scenarios.get(scenario) if isinstance(scenarios, Mapping) else None
    if not isinstance(scene, Mapping):
        raise ValueError(f"unknown traditional-control scenario: {scenario}")
    terrain_cfg = scene.get("terrain")
    terrain_cfg = terrain_cfg if isinstance(terrain_cfg, Mapping) else {}
    terrain = terrain or terrain_model_from_config(config, scenario)
    nominal_foot_z = float(np.max(kinematics.nominal_foot_positions()[:, 2]))
    sole_offset = profile.nominal_base_height + nominal_foot_z - profile.foot_radius
    terrain_base = float(terrain_cfg.get("base_height", terrain_cfg.get("height", 0.0)))
    ground_level = terrain_base + sole_offset
    primitives = [
        MujocoScenePrimitive(
            id="nominal_ground",
            type="plane",
            center=(0.0, 0.0, ground_level),
            size=(40.0, 40.0, 0.2),
        )
    ]
    if isinstance(terrain, StairTerrain):
        for index in range(int(terrain.max_steps)):
            height = float(ground_level + (index + 1) * terrain.rise)
            center_x = float(terrain.start_x + (index + 0.5) * terrain.run)
            primitives.append(
                MujocoScenePrimitive(
                    id=f"nominal_step_{index:03d}",
                    type="box",
                    center=(center_x, 0.0, (ground_level + height) * 0.5),
                    size=(float(terrain.run), float(terrain.width), height - ground_level),
                )
            )
        top_height = ground_level + float(terrain.max_steps) * float(terrain.rise)
        landing_start = float(terrain.start_x + terrain.max_steps * terrain.run)
        primitives.append(
            MujocoScenePrimitive(
                id="nominal_upper_landing",
                type="box",
                center=(landing_start + 0.5, 0.0, 0.5 * (ground_level + top_height)),
                size=(1.0, float(terrain.width), top_height - ground_level),
            )
        )
    return tuple(primitives)


def nominal_terrain_primitives(
    bundle: TailiControllerBundle,
    scenario: str,
) -> tuple[MujocoScenePrimitive, ...]:
    """Describe the nominal collision course without depending on a renderer."""
    return nominal_terrain_primitives_from_config(
        bundle.config,
        scenario,
        profile=bundle.profile,
        kinematics=bundle.kinematics,
        terrain=bundle.controller.terrain,
    )


def build_nominal_scene(bundle: TailiControllerBundle, scenario: str) -> MujocoNominalScene:
    """Create a model with no hidden RL assets or generated source files."""

    import mujoco

    profile = bundle.profile
    config = bundle.config
    spec = mujoco.MjSpec.from_file(str(profile.urdf_path))
    root = spec.worldbody.bodies[0]
    root.add_freejoint()
    spec.option.timestep = float(config["simulation"]["physics_dt"])
    for primitive in nominal_terrain_primitives(bundle, scenario):
        if primitive.type == "plane":
            spec.worldbody.add_geom(
                name=primitive.id,
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                pos=list(primitive.center),
                size=[value * 0.5 for value in primitive.size],
                friction=[1.0, 0.005, 0.0001],
            )
        elif primitive.type == "box":
            spec.worldbody.add_geom(
                name=primitive.id,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=list(primitive.center),
                size=[value * 0.5 for value in primitive.size],
                friction=[1.0, 0.005, 0.0001],
            )
        else:  # pragma: no cover - descriptors are constructed above
            raise ValueError(f"unsupported nominal scene primitive: {primitive.type}")

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
