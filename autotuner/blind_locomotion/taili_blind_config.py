"""Single-source Taili blind training configuration.

`taili_blind_config.yaml` is the only editable training configuration. Runtime
files such as `agent.skrl.yaml` and `effective_config.yaml` are generated from
it per run; they are evidence artifacts, not independent source configs.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import os
from typing import Any, Mapping

import yaml


CONFIG_FILENAME = "taili_blind_config.yaml"


def default_config_path() -> Path:
    return Path(__file__).resolve().with_name(CONFIG_FILENAME)


def resolve_config_path(path: str | os.PathLike[str] | None = None) -> Path:
    if path:
        return Path(path)
    env_path = os.environ.get("TAILI_CONFIG_PATH")
    if env_path:
        return Path(env_path)
    return default_config_path()


def load_taili_blind_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    cfg_path = resolve_config_path(path)
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Taili config root must be a mapping: {cfg_path}")
    return data


def write_taili_blind_config(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False, allow_unicode=True)


def get_config_value(data: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def as_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def _role_joint_values(section: Mapping[str, Any] | None, default: Mapping[str, float]) -> dict[str, float]:
    src = section if isinstance(section, Mapping) else {}
    out: dict[str, float] = {}
    for role, value in default.items():
        try:
            out[role] = float(src.get(role, value))
        except (TypeError, ValueError):
            out[role] = float(value)
    return out


def _expand_leg_joint_values(role_values: Mapping[str, float]) -> list[float]:
    values: list[float] = []
    for _leg in ("FL", "FR", "RL", "RR"):
        values.extend([
            float(role_values.get("hip", 120.0)),
            float(role_values.get("thigh", 120.0)),
            float(role_values.get("calf", 120.0)),
        ])
    return values


def _first_config_value(data: Mapping[str, Any], paths: tuple[str, ...], default: Any = None) -> Any:
    for path in paths:
        value = get_config_value(data, path, None)
        if value is not None:
            return value
    return default


def build_skrl_config(data: Mapping[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(get_config_value(data, "skrl", {}))
    if not isinstance(cfg, dict) or not cfg:
        raise ValueError("taili_blind_config.yaml must contain a non-empty skrl section")

    policy = cfg.setdefault("models", {}).setdefault("policy", {})
    if not isinstance(policy, dict):
        raise ValueError("skrl.models.policy must be a mapping")
    policy["actor_hidden"] = list(_first_config_value(data, ("model.actor.hidden", "model.actor_hidden"), [1024, 512]))
    policy["dropout"] = float(_first_config_value(data, ("model.terrain_perceiver.dropout", "model.perceiver_dropout"), 0.0))
    return cfg


def effective_config_with_overrides(
    data: Mapping[str, Any],
    *,
    total_steps: int = 0,
    write_interval: str | int | None = None,
    checkpoint_interval: int | None = None,
) -> dict[str, Any]:
    cfg = deepcopy(dict(data))
    skrl = cfg.setdefault("skrl", {})
    trainer = skrl.setdefault("trainer", {})
    if total_steps > 0:
        trainer["timesteps"] = int(total_steps)
    exp = skrl.setdefault("agent", {}).setdefault("experiment", {})
    if write_interval is not None:
        exp["write_interval"] = int(write_interval) if str(write_interval).isdigit() else write_interval
    if checkpoint_interval is not None:
        exp["checkpoint_interval"] = int(checkpoint_interval)
    return cfg


def reward_config_mapping(data: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = data if data is not None else load_taili_blind_config()
    reward = get_config_value(cfg, "reward", {})
    if not isinstance(reward, dict):
        raise ValueError("taili reward section must be a mapping")
    out = dict(reward)
    if "nominal_base_h" in out:
        nominal = float(out["nominal_base_h"])
        out["h_ok"] = nominal - 0.05
        out["h_gate_close"] = nominal - 0.10
    return out


def _mapping_from_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _obj_value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def phase_command_spec(cfg: Any, phase: int | None = None) -> dict[str, Any]:
    """Resolve the command recipe active for the current training phase.

    The single YAML can describe a phased curriculum:

    training_recipe:
      command_mode: phase_curriculum
      phases:
        0: {command_mode: fixed_forward, fixed_vx: 0.5}
        1: {command_mode: single_axis, prob_fwd: 0.35, ...}

    The env keeps the live phase on ``self._phase``.  This helper is deliberately
    pure and tolerant of both dict configs and IsaacLab config objects, so tests
    and runtime use the same code path.
    """
    base_mode = str(_obj_value(cfg, "training_command_mode", "normal") or "normal")
    if base_mode not in {"phase_curriculum", "phased_curriculum", "phased"}:
        return {"command_mode": base_mode}

    phase_map = _mapping_from_obj(_obj_value(cfg, "training_phase_commands", {}))
    if not phase_map:
        return {"command_mode": "normal"}
    current = int(phase if phase is not None else _obj_value(cfg, "current_training_phase", _obj_value(cfg, "init_phase", 0)))

    parsed: list[tuple[int, dict[str, Any]]] = []
    for key, value in phase_map.items():
        try:
            parsed.append((int(key), _mapping_from_obj(value)))
        except (TypeError, ValueError):
            continue
    if not parsed:
        return {"command_mode": "normal"}
    parsed.sort(key=lambda item: item[0])
    selected = parsed[0][1]
    for phase_id, spec in parsed:
        if phase_id <= current:
            selected = spec
        else:
            break
    mode = str(selected.get("command_mode") or selected.get("mode") or "normal")
    out = dict(selected)
    out["command_mode"] = mode
    return out


def active_command_directions(cfg: Any, phase: int | None = None) -> tuple[str, ...]:
    """Return the command directions that are actually sampled by this recipe.

    Curriculum gates must not take the min over directions that the current
    recipe never trains. For example, bootstrap_forward fixes all commands to
    forward velocity, so back/lat/yaw progress staying at zero is not evidence
    of failure. In normal mixed-command mode, use the configured sampling
    probabilities.
    """
    spec = phase_command_spec(cfg, phase)
    explicit = spec.get("active_dirs")
    if isinstance(explicit, (list, tuple)):
        active = tuple(str(item) for item in explicit if str(item) in {"fwd", "back", "lat", "yaw"})
        if active:
            return active

    mode = str(spec.get("command_mode") or _obj_value(cfg, "training_command_mode", "normal") or "normal")
    if mode in {"fixed_forward", "forward_range"}:
        return ("fwd",)
    if mode == "stand_only":
        return ("stand",)
    if mode == "mixed":
        return ("fwd", "back", "lat", "yaw")

    probs = {
        "fwd": float(spec.get("prob_fwd", _obj_value(cfg, "cmd_prob_fwd", 0.25))),
        "back": float(spec.get("prob_back", _obj_value(cfg, "cmd_prob_back", 0.25))),
        "lat": float(spec.get("prob_lat", _obj_value(cfg, "cmd_prob_lat", 0.25))),
        "yaw": float(spec.get("prob_yaw", _obj_value(cfg, "cmd_prob_yaw", 0.25))),
    }
    active = tuple(name for name, prob in probs.items() if prob > 1e-9)
    return active or ("fwd", "back", "lat", "yaw")


def phase_progress_directions(cfg: Any, phase: int | None = None) -> tuple[str, ...]:
    """Directions used by phase/curriculum progress gates.

    Command sampling can still train every direction, including yaw. This helper only
    controls which directions are allowed to block curriculum progress. It lets terrain
    unlock depend on traversable linear motion while yaw remains an independently
    logged and rewarded objective.
    """
    active = tuple(name for name in active_command_directions(cfg, phase) if name != "stand")
    phase_id = int(phase if phase is not None else _obj_value(cfg, "current_training_phase", _obj_value(cfg, "init_phase", 0)))
    raw = _obj_value(cfg, f"phase_progress_dirs_{phase_id}", None)
    if raw is None:
        raw = _obj_value(cfg, "phase_progress_dirs", None)
    if raw is None:
        return active
    if isinstance(raw, str):
        items = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, (list, tuple, set)):
        items = [str(part) for part in raw]
    else:
        return active
    selected = tuple(name for name in items if name in active and name in {"fwd", "back", "lat", "yaw"})
    return selected or active


def active_direction_progress(progress: Mapping[str, float], cfg: Any, phase: int | None = None) -> tuple[float, tuple[str, ...]]:
    """Return min progress over phase-gated directions plus that direction list."""
    active = phase_progress_directions(cfg, phase)
    values = [float(progress[name]) for name in active if name in progress]
    if not values:
        values = [float(v) for v in progress.values()]
    return (min(values) if values else 0.0), active


def _set_if_present(obj: Any, attr: str, mapping: Mapping[str, Any], key: str | None = None) -> None:
    source_key = key or attr
    if source_key in mapping:
        setattr(obj, attr, as_tuple(mapping[source_key]))


def apply_env_config_to_cfg(env_cfg: Any, data: Mapping[str, Any] | None = None) -> None:
    cfg = data if data is not None else load_taili_blind_config()
    env = get_config_value(cfg, "env", {})
    if not isinstance(env, Mapping):
        raise ValueError("taili env section must be a mapping")
    reward = get_config_value(cfg, "reward", {})
    nominal_base_h = None
    if isinstance(reward, Mapping) and reward.get("nominal_base_h") is not None:
        nominal_base_h = float(reward["nominal_base_h"])

    obs = env.get("observation", {})
    if isinstance(obs, Mapping):
        _set_if_present(env_cfg, "obs_history_len", obs, "history_len")
        _set_if_present(env_cfg, "obs_history_dim", obs, "tick_dim")
        _set_if_present(env_cfg, "observation_space", obs)
        _set_if_present(env_cfg, "amp_observation_space", obs)
        _set_if_present(env_cfg, "num_amp_observations", obs, "amp_frames")
    contract = env.get("observation_contract", {})
    deployable = contract.get("deployable", {}) if isinstance(contract, Mapping) else {}
    amp = env.get("amp", {})
    if isinstance(deployable, Mapping):
        _set_if_present(env_cfg, "obs_history_len", deployable, "history_len")
        _set_if_present(env_cfg, "obs_history_dim", deployable, "tick_dim")
    if isinstance(contract, Mapping):
        _set_if_present(env_cfg, "observation_space", contract, "policy_tensor_dim")
    if isinstance(amp, Mapping):
        _set_if_present(env_cfg, "num_amp_observations", amp, "frames")
        _set_if_present(env_cfg, "amp_observation_space", amp, "frame_dim")

    control = env.get("control", {})
    if isinstance(control, Mapping):
        for attr, key in (
            ("action_scale", "action_scale"),
            ("action_delay_steps", "action_delay_steps"),
            ("episode_length_s", "episode_length_s"),
            ("decimation", "decimation"),
            ("dt", "dt"),
            ("contact_force_threshold", "contact_force_threshold"),
        ):
            _set_if_present(env_cfg, attr, control, key)
    if nominal_base_h is not None:
        setattr(env_cfg, "stand_height", nominal_base_h)
        setattr(env_cfg, "termination_height", nominal_base_h * 0.67)

    gait = env.get("gait", {})
    if isinstance(gait, Mapping):
        for attr, key in (
            ("gait_period", "period"),
            ("gait_period_min", "period_min"),
            ("gait_period_slope", "period_slope"),
            ("gait_duty", "duty"),
        ):
            _set_if_present(env_cfg, attr, gait, key)

    actuator = env.get("actuator", {})
    if isinstance(actuator, Mapping):
        stiffness = _role_joint_values(actuator.get("stiffness"), {"hip": 120.0, "thigh": 120.0, "calf": 120.0})
        damping = _role_joint_values(actuator.get("damping"), {"hip": 10.0, "thigh": 10.0, "calf": 10.0})
        setattr(env_cfg, "actuator_stiffness", stiffness)
        setattr(env_cfg, "actuator_damping", damping)
        setattr(env_cfg, "actuator_stiffness_by_joint", _expand_leg_joint_values(stiffness))
        setattr(env_cfg, "actuator_damping_by_joint", _expand_leg_joint_values(damping))
        try:
            legs = env_cfg.robot.actuators["legs"]
            legs.stiffness = {
                ".*_hip_joint": stiffness["hip"],
                ".*_thigh_joint": stiffness["thigh"],
                ".*_calf_joint": stiffness["calf"],
            }
            legs.damping = {
                ".*_hip_joint": damping["hip"],
                ".*_thigh_joint": damping["thigh"],
                ".*_calf_joint": damping["calf"],
            }
        except Exception:
            pass

    scene = env.get("scene", {})
    if isinstance(scene, Mapping):
        if "num_envs" in scene and hasattr(env_cfg, "scene"):
            env_cfg.scene.num_envs = int(scene["num_envs"])
        if "env_spacing" in scene and hasattr(env_cfg, "scene"):
            env_cfg.scene.env_spacing = float(scene["env_spacing"])

    commands = env.get("commands", {})
    if isinstance(commands, Mapping):
        mapping = {
            "cmd_resample_s": "resample_s",
            "cmd_resample_s_min": "resample_s_min",
            "cmd_resample_s_max": "resample_s_max",
            "cmd_smooth_alpha": "smooth_alpha",
            "cmd_transition_enable": "transition_enable",
            "cmd_transition_cycles": "transition_cycles",
            "cmd_transition_min_s": "transition_min_s",
            "cmd_transition_max_s": "transition_max_s",
            "cmd_transition_fast_s": "transition_fast_s",
            "cmd_transition_sign_flip_v": "transition_sign_flip_v",
            "cmd_transition_sign_flip_w": "transition_sign_flip_w",
            "cmd_transition_low_speed_v": "transition_low_speed_v",
            "cmd_transition_low_speed_w": "transition_low_speed_w",
            "cmd_transition_contact_feet": "transition_contact_feet",
            "cmd_transition_zero_frac_min": "transition_zero_frac_min",
            "cmd_transition_zero_frac_max": "transition_zero_frac_max",
            "cmd_transition_stop_zero_frac": "transition_stop_zero_frac",
            "cmd_fwd_max": "fwd_max",
            "cmd_back_max": "back_max",
            "cmd_lat_max": "lat_max",
            "cmd_yaw_max": "yaw_max",
        }
        for attr, key in mapping.items():
            _set_if_present(env_cfg, attr, commands, key)

    curriculum = env.get("curriculum", {})
    if isinstance(curriculum, Mapping):
        # forward EVERY curriculum key onto the env cfg: gate thresholds are per-phase and sparse
        # (phase_gate_<name>_<p> with walk-down in the env), so a fixed key list silently drops
        # newly added phase knobs (e.g. penalty_budget_ratio_max, phase_gate_air_2).
        for attr in curriculum:
            _set_if_present(env_cfg, str(attr), curriculum)

    dr = env.get("domain_randomization", {})
    if isinstance(dr, Mapping):
        mapping = {
            "dr_enable": "enable",
            "dr_start_level": "start_level",
            "dr_unlock_terrain": "unlock_terrain",
            "dr_push_interval_s_0": "push_interval_s_0",
            "dr_push_vel_0": "push_vel_0",
            "dr_push_interval_s_1": "push_interval_s_1",
            "dr_push_vel_1": "push_vel_1",
            "dr_mass_range_1": "mass_range_1",
            "dr_stiffness_scale_1": "stiffness_scale_1",
            "dr_damping_scale_1": "damping_scale_1",
            "dr_push_interval_s_2": "push_interval_s_2",
            "dr_push_vel_2": "push_vel_2",
            "dr_mass_range_2": "mass_range_2",
            "dr_stiffness_scale_2": "stiffness_scale_2",
            "dr_damping_scale_2": "damping_scale_2",
            "dr_push_interval_s_3": "push_interval_s_3",
            "dr_push_vel_3": "push_vel_3",
            "dr_push_ang_scale": "push_ang_scale",
            "dr_mass_range_3": "mass_range_3",
            "dr_stiffness_scale_3": "stiffness_scale_3",
            "dr_damping_scale_3": "damping_scale_3",
            "dr_friction_range_3": "friction_range_3",
            "dr_com_offset_3": "com_offset_3",
            "dr_imu_gyro_bias_3": "imu_gyro_bias_3",
            "dr_imu_grav_bias_3": "imu_grav_bias_3",
            "dr_gate_progress": "gate_progress",
            "dr_gate_progress_l2": "gate_progress_l2",
            "dr_gate_progress_l3": "gate_progress_l3",
            "dr_gate_intervals": "gate_intervals",
        }
        for attr, key in mapping.items():
            _set_if_present(env_cfg, attr, dr, key)

    overrides = env.get("blind_overrides", {})
    if isinstance(overrides, Mapping):
        for attr in (
            "sym_augment",
            "rew_gait_enforce",
            "rew_swing_drag",
            "rew_stance_slip",
            "rew_land_decel",
            "rew_clearance",
            "rew_clearance_heavy",
            "rew_overshoot",
            "rew_underspeed",
            "rew_backward_underspeed",
            "rew_backward_wrong_dir",
            "rew_lateral_underspeed",
            "directional_aux_backward_only",
            "rew_wrong_dir",
            "w_imitate_live",
            "imitate_sigma_live",
            "imitate_fwd_floor",
            "w_swing_dir",
            "swing_dir_margin",
            "swing_dir_sigma",
            "rew_climb",
            "climb_slip_gate",
            "climb_slip_soft_span",
            "climb_vz_cap",
            "rew_terrain_up",
            "rew_terrain_down",
            "rew_terrain_support_transfer",
            "rew_terrain_contact_quality",
            "rew_terrain_event_collapse",
            "rew_ang_vel_xy",
            "rew_hip_neutral",
            "hip_neutral_lat_scale",
            "strict_blind_terrain_reward",
            "w_lateral_foot_excursion",
            "lateral_foot_margin",
            "lateral_foot_scale",
            "terrain_transition_eps",
            "terrain_transition_span",
            "terrain_event_latch_s",
            "terrain_up_vz_cap",
            "terrain_down_vz_cap",
            "terrain_down_vz_target",
            "terrain_event_quality_floor",
            "terrain_front_duty_margin",
            "terrain_rear_duty_floor",
            "terrain_support_scale",
            "terrain_torque_soft_frac",
            "terrain_event_collapse_height",
            "terrain_event_collapse_wxy",
            "terrain_event_collapse_speed_ratio",
            "terrain_event_collapse_speed_min",
            "terrain_event_collapse_speed_scale",
            "terrain_curriculum_height_gain",
            "terrain_curriculum_height_loss",
            "terrain_curriculum_forward_min",
            "terrain_curriculum_stable_h",
            "terrain_curriculum_stable_upright",
            "terrain_curriculum_stable_contact_min",
            "terrain_curriculum_stable_wxy_max",
            "terrain_curriculum_speed_cap_ratio",
            "terrain_curriculum_speed_cap_min",
            "terrain_curriculum_failure_h",
            "terrain_curriculum_failure_upright",
            "terrain_curriculum_failure_wxy",
            "terrain_curriculum_floor_after_peak",
            "terrain_curriculum_peak_drop",
            "terrain_curriculum_move_down_patience",
            "discrete_clearance",
            "speed_tol_abs",
            "speed_tol_rel",
            "base_clearance",
            "stance_dx",
            "clr_rough_flat",
            "clr_rough_span",
            "clr_rough_bonus_max",
        ):
            _set_if_present(env_cfg, attr, overrides)

    terrain = env.get("terrain", {})
    if isinstance(terrain, Mapping) and hasattr(env_cfg, "terrain"):
        sub = env_cfg.terrain.terrain_generator.sub_terrains
        for name, values in terrain.items():
            if name not in sub or not isinstance(values, Mapping):
                continue
            target = sub[name]
            for attr, value in values.items():
                setattr(target, attr, as_tuple(value))

    recipe = get_config_value(cfg, "training_recipe", {})
    if isinstance(recipe, Mapping):
        mapping = {
            "training_recipe_id": "id",
            "training_command_mode": "command_mode",
            "training_fixed_vx": "fixed_vx",
            "training_forward_range": "forward_range",
            "touchdown_impact_only": "touchdown_impact_only",
            "init_phase": "init_phase",
        }
        for attr, key in mapping.items():
            _set_if_present(env_cfg, attr, recipe, key)
        phases = recipe.get("phases")
        if isinstance(phases, Mapping):
            setattr(env_cfg, "training_phase_commands", deepcopy(dict(phases)))
            parsed_phase_ids = []
            for key in phases:
                try:
                    parsed_phase_ids.append(int(key))
                except (TypeError, ValueError):
                    continue
            if parsed_phase_ids and not hasattr(env_cfg, "max_training_phase"):
                setattr(env_cfg, "max_training_phase", max(parsed_phase_ids))
