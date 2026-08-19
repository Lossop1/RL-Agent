"""课程和配置加载器的内部一致性测试，纯 Python，不依赖仿真。

这些测试锁定每次策略编辑都会经过的 YAML -> runtime 映射：
阶段解析、活跃方向门控、奖励配置透传，以及配置字段是否真正落到运行对象。
"""
from dataclasses import fields
from types import SimpleNamespace

import pytest

from products.taili.blind_locomotion import taili_blind_config as C
from products.taili.core import taili_geometry
from products.taili.core.taili_reward import RewardConfig


def _cfg_phase_curriculum():
    return SimpleNamespace(
        training_command_mode="phase_curriculum",
        training_phase_commands={
            0: {"command_mode": "single_axis", "active_dirs": ["fwd", "back", "lat", "yaw"], "prob_yaw": 0.40},
            1: {"command_mode": "mixed"},
            2: {"command_mode": "mixed"},
            3: {"command_mode": "mixed"},
        },
        init_phase=0,
    )


def test_phase_spec_selects_highest_phase_at_or_below_current():
    cfg = _cfg_phase_curriculum()
    assert C.phase_command_spec(cfg, 0)["command_mode"] == "single_axis"
    assert C.phase_command_spec(cfg, 2)["command_mode"] == "mixed"
    # 超过最后定义阶段时，沿用最后一个阶段配置。
    assert C.phase_command_spec(cfg, 9)["command_mode"] == "mixed"


def test_mixed_phase_keeps_balanced_single_axis_commands_inside_moving_samples():
    cfg = SimpleNamespace(
        training_command_mode="phase_curriculum",
        training_phase_commands={
            1: {
                "command_mode": "mixed",
                "stand_prob": 0.10,
                "near_zero_prob": 0.08,
                "single_axis_fraction": 0.40,
            },
        },
        init_phase=1,
    )

    probs = C.phase_command_spec(cfg, 1)["bucket_probs"]
    moving = 1.0 - probs["stand"] - probs["near_zero"]
    single = sum(probs[name] for name in ("fwd", "back", "lat", "yaw"))
    mixed = sum(probs[name] for name in ("linear_yaw", "lat_yaw", "mixed_linear"))

    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["near_zero"] == pytest.approx((1.0 - 0.10) * 0.08)
    assert single / moving == pytest.approx(0.40)
    assert mixed / moving == pytest.approx(0.60)
    for name in ("fwd", "back", "lat", "yaw"):
        assert probs[name] / moving == pytest.approx(0.10)


def test_non_phase_mode_passthrough():
    cfg = SimpleNamespace(training_command_mode="normal")
    assert C.phase_command_spec(cfg)["command_mode"] == "normal"


def test_active_directions_from_explicit_active_dirs():
    cfg = _cfg_phase_curriculum()
    assert C.active_command_directions(cfg, 0) == ("fwd", "back", "lat", "yaw")


def test_active_directions_mixed_is_all_four():
    cfg = _cfg_phase_curriculum()
    assert set(C.active_command_directions(cfg, 1)) == {"fwd", "back", "lat", "yaw"}


def test_active_directions_ignores_untrained_dirs_via_probs():
    # 只采样 fwd/yaw 的单轴阶段，不能让 back/lat progress 阻塞课程。
    cfg = SimpleNamespace(
        training_command_mode="phase_curriculum",
        training_phase_commands={0: {"command_mode": "single_axis",
                                     "prob_fwd": 0.5, "prob_yaw": 0.5,
                                     "prob_back": 0.0, "prob_lat": 0.0}},
        init_phase=0,
    )
    active = C.active_command_directions(cfg, 0)
    assert set(active) == {"fwd", "yaw"}
    assert "back" not in active and "lat" not in active


def test_single_axis_occupancy_refills_the_direction_removed_by_resets():
    deficits = C.single_axis_occupancy_deficits(
        total_envs=20,
        stand_prob=0.20,
        direction_weights=[0.25, 0.25, 0.25, 0.25],
        surviving_counts=[4, 1, 4, 4, 4],
    )

    assert deficits == (0, 3, 0, 0, 0)


def test_single_axis_occupancy_does_not_refill_an_overrepresented_survivor():
    deficits = C.single_axis_occupancy_deficits(
        total_envs=20,
        stand_prob=0.20,
        direction_weights=[0.25, 0.25, 0.25, 0.25],
        surviving_counts=[4, 0, 0, 0, 12],
    )

    assert deficits[0] == 0 and deficits[4] == 0
    assert deficits[1:4] == (4, 4, 4)


def test_non_stair_stratum_preserves_its_own_forward_quota():
    # 20 个楼梯环境会在外层额外固定为前进；这里的 80 个非楼梯环境仍应
    # 独立获得完整的平地前进配额，而不是被楼梯样本抵消。
    deficits = C.single_axis_occupancy_deficits(
        total_envs=80,
        stand_prob=0.10,
        direction_weights=[0.25, 0.25, 0.25, 0.25],
        surviving_counts=[0, 0, 0, 0, 0],
    )

    assert deficits == (8, 18, 18, 18, 18)


def test_active_direction_progress_takes_min_over_active_only():
    cfg = _cfg_phase_curriculum()
    progress = {"fwd": 0.9, "back": 0.8, "lat": 0.7, "yaw": 0.5}
    val, active = C.active_direction_progress(progress, cfg, 0)
    assert val == 0.5  # yaw 是最小值，并且当前阶段会训练 yaw。
    # 如果 yaw 不参与训练，它的低 progress 不能拖低阶段门控。
    cfg_noyaw = SimpleNamespace(
        training_command_mode="phase_curriculum",
        training_phase_commands={0: {"command_mode": "single_axis", "prob_fwd": 1.0,
                                     "prob_back": 0.0, "prob_lat": 0.0, "prob_yaw": 0.0}},
        init_phase=0,
    )
    val2, active2 = C.active_direction_progress(progress, cfg_noyaw, 0)
    assert active2 == ("fwd",) and val2 == 0.9


def test_reward_config_mapping_derives_height_gates():
    data = {"reward": {"nominal_base_h": 0.52, "w_tracking_lin": 2.5}}
    out = C.reward_config_mapping(data)
    assert out["w_tracking_lin"] == 2.5
    # 高度只有 nominal_base_h 一个来源，其余门槛由它派生。
    assert abs(out["h_ok"] - 0.47) < 1e-9
    assert abs(out["h_gate_close"] - 0.42) < 1e-9


def test_real_yaml_loads_and_reward_maps():
    # 随包策略契约必须能加载，并暴露结构完整的 reward 区块。
    data = C.load_taili_blind_config()
    rc = C.reward_config_mapping(data)
    assert rc["sigma_lin_abs"] == 0.10
    assert rc["sigma_yaw"] == 0.10
    assert rc["w_tracking_yaw"] == 2.25
    assert rc["w_yaw_progress"] == 0.75
    assert rc["w_yaw_far"] == 1.0
    assert rc["w_supported_progress"] == 0.75
    assert rc["w_linear_underspeed"] == 0.0
    assert rc["w_track_far"] == 1.0
    assert rc["direction_progress_full_ratio"] == 0.60
    assert rc["direction_progress_support_floor"] == 0.0
    assert rc["direction_progress_quality_floor"] == 0.0
    assert rc["w_terrain_progress"] == 0.0
    assert rc["w_yaw_support_moment"] == 0.0
    assert rc["w_yaw_wrong_moment"] == 0.0
    assert data["env"]["observation"]["history_stride"] == 2
    assert data["env"]["observation"]["history_order"] == "oldest_first"
    assert data["env"]["commands"]["transition_policy_managed"] is True
    assert data["env"]["curriculum"]["progress_ema_alpha"] == 0.01
    assert data["env"]["curriculum"]["progress_ema_min_samples"] == 8
    assert data["env"]["curriculum"]["progress_ema_reference_samples"] == 16
    assert data["env"]["curriculum"]["penalty_budget_controls_quality"] is False
    assert data["env"]["curriculum"]["phase_gate_flat_false_terrain_response_0"] == 0.01
    assert data["env"]["curriculum"]["terrain_curriculum_state_schema"] == 4
    phase3 = data["training_recipe"]["phases"][3]
    assert phase3["single_axis_fraction"] == 0.40
    phase3_probs = C.phase_command_spec(
        SimpleNamespace(
            training_command_mode="phase_curriculum",
            training_phase_commands=data["training_recipe"]["phases"],
            init_phase=3,
        ),
        3,
    )["bucket_probs"]
    phase3_moving = 1.0 - phase3_probs["stand"] - phase3_probs["near_zero"]
    assert sum(phase3_probs[name] for name in ("fwd", "back", "lat", "yaw")) / phase3_moving == pytest.approx(0.40)
    terrain = data["env"]["terrain"]
    assert terrain["flat"]["proportion"] == 0.30
    assert terrain["stairs"]["proportion"] == 0.15
    assert terrain["stairs_up"]["proportion"] == 0.20
    assert terrain["stairs"]["step_height_range"][-1] == 0.30
    assert terrain["stairs_up"]["step_height_range"][-1] == 0.30
    assert terrain["stairs"]["platform_width"] == 1.5
    assert terrain["stairs_up"]["platform_width"] == 1.5
    assert sum(float(spec["proportion"]) for spec in terrain.values()) == pytest.approx(1.0)
    assert rc["w_flat_move_height"] == 2.0
    assert rc["w_core_quality"] == 1.20
    assert rc["core_tilt_scale"] == 0.035
    assert rc["core_ang_vel_scale"] == 0.20
    assert rc["core_height_scale"] == 0.020
    assert rc["core_vz_scale"] == 0.12
    assert rc["core_progress_floor"] == 0.20
    assert rc["core_terrain_relief"] == 0.70
    assert rc["core_tail_gain"] == 0.40
    assert rc["tracking_posture_floor"] == 0.20
    assert rc["task_credit_floor"] == 0.20
    assert rc["task_credit_start_s"] == 1.50
    assert rc["task_credit_full_s"] == 3.00
    assert rc["tracking_motion_floor"] == 0.20
    assert rc["flat_move_height_target"] == pytest.approx(taili_geometry.NOMINAL_BASE_HEIGHT)
    assert rc["base_ang_accel_free"] == 2.5
    assert rc["base_ang_accel_scale"] == 6.0
    assert rc["yaw_duty_quality_floor"] == 0.50
    assert rc["nominal_base_h"] == pytest.approx(taili_geometry.NOMINAL_BASE_HEIGHT)
    assert data["env"]["blind_overrides"]["rew_climb"] == 0.0
    assert "rew_terrain_up" not in data["env"]["blind_overrides"]
    assert "rew_terrain_down" not in data["env"]["blind_overrides"]
    assert data["env"]["blind_overrides"]["terrain_response_height_scale"] == 0.04
    assert data["env"]["blind_overrides"]["terrain_response_height_deadband"] == 0.008
    assert data["env"]["blind_overrides"]["terrain_response_delta_deadband"] == 0.008
    assert data["env"]["blind_overrides"]["terrain_collision_trace_decay_time"] == 0.40
    assert data["env"]["blind_overrides"]["terrain_collision_response_full_scale"] == 0.35
    assert data["env"]["blind_overrides"]["terrain_curriculum_stair_height_min"] == 0.035
    assert "rew_terrain_direction_progress" not in data["env"]["blind_overrides"]
    assert "rew_terrain_layer_hold" not in data["env"]["blind_overrides"]
    assert data["env"]["blind_overrides"]["rew_terrain_support_loss"] == 0.0
    assert data["env"]["blind_overrides"]["rew_terrain_overspeed"] == 0.0
    assert data["env"]["blind_overrides"]["terrain_probe_height"] == 0.12
    assert data["env"]["blind_overrides"]["terrain_overspeed_start_ratio"] == 1.15
    assert data["env"]["blind_overrides"]["rew_terrain_collapse"] == 0.0
    assert data["env"]["curriculum"]["phase_gate_stairs_up_success_2"] == 0.65
    assert data["env"]["curriculum"]["phase_gate_stairs_down_collapse_2"] == 0.10
    assert data["env"]["actuator"]["stiffness"] == {"hip": 100, "thigh": 100, "calf": 100}
    assert data["env"]["actuator"]["damping"] == {"hip": 5.0, "thigh": 5.0, "calf": 5.0}
    assert rc["w_landing_impact_tail"] == 1.35
    assert rc["w_landing_impact_tail_late"] == 0.55
    assert rc["foot_trajectory_terminal_vz_scale"] == 0.28
    assert rc["w_terminal_swing_velocity"] == 0.80
    assert rc["w_clearance_under"] == 1.8
    assert rc["w_clearance_over"] == 1.0
    assert rc["clearance_worst_leg_mix"] == 0.65
    assert rc["linear_front_rear_support_scale"] == 1.0
    assert rc["terrain_clearance_margin_min"] == 0.02
    assert rc["terrain_clearance_margin_max"] == 0.035
    assert rc["h_ok"] == pytest.approx(taili_geometry.NOMINAL_BASE_HEIGHT - 0.05)
    assert rc["h_gate_close"] == pytest.approx(taili_geometry.NOMINAL_BASE_HEIGHT - 0.10)
    phases = data["training_recipe"]["phases"]
    assert phases[0]["command_mode"] == "single_axis"
    assert phases[1]["command_mode"] == "mixed"
    assert data["env"]["curriculum"]["phase_progress_dirs_0"] == ["fwd", "back", "lat", "yaw"]
    assert data["env"]["curriculum"]["phase_progress_dirs_1"] == ["fwd", "back", "lat", "yaw"]
    assert data["env"]["curriculum"]["phase_progress_dirs_2"] == ["fwd", "back", "lat", "yaw"]
    assert data["env"]["curriculum"]["phase_timeout_enable"] is False
    assert data["env"]["gait"] == {"period": 0.76, "period_min": 0.62, "period_slope": 0.05, "yaw_speed_equiv": 0.3, "duty": 0.5}
    assert data["reward"]["duty_linear_min"] == 0.42
    assert data["reward"]["duty_linear_max"] == 0.66
    assert data["reward"]["duty_symmetry_tolerance"] == 0.20
    assert data["reward"]["duty_cycle_ema_beta"] == 0.60
    assert data["env"]["curriculum"]["phase_gate_duty_valid_0"] == 0.70
    assert data["env"]["curriculum"]["phase_gate_terminal_rate_0"] == 0.01
    assert data["env"]["curriculum"]["phase_gate_terminal_rate_1"] == 0.008
    assert "phase_gate_flat_ang_accel_p95_0" not in data["env"]["curriculum"]
    assert data["env"]["curriculum"]["phase_gate_flat_touchdown_vz_p95_0"] == 0.60
    assert data["reward"]["gait_period_tolerance"] == 0.22
    assert data["env"]["commands"]["resample_s_min"] == 5.0
    assert data["env"]["commands"]["resample_s_max"] == 8.0
    assert data["skrl"]["agent"]["entropy_loss_scale"] == 0.02
    assert data["skrl"]["models"]["policy"]["max_log_std"] == -0.4
    assert data["model"]["terrain_perceiver_aux"]["ramp_steps"] == 30000
    assert data["model"]["terrain_perceiver_aux"]["exploration_entropy_final"] == 0.005
    assert data["model"]["terrain_perceiver_aux"]["exploration_entropy_hold_fraction"] == 0.75
    assert data["env"]["observation_contract"]["training_only"]["aux_label_dim"] == 34
    assert data["env"]["observation_contract"]["policy_tensor_dim"] == 1638
    assert data["env"]["observation"]["observation_space"] == 1638
    assert data["env"]["curriculum"]["terrain_start_phase"] == 0
    assert data["env"]["curriculum"]["terrain_reward_start_phase"] == 0
    assert data["env"]["curriculum"]["dr_start_phase"] == 0
    assert data["skrl"]["agent"]["rollouts"] == 48
    assert data["skrl"]["agent"]["style_reward_weight"] == 1.00
    assert data["env"]["scene"]["num_envs"] == 4096
    assert data["env"]["commands"]["transition_enable"] is False
    assert rc["quality_reward_floor"] == 0.55
    assert rc["clearance_worst_leg_mix"] == 0.65
    assert rc["w_touchdown_slip_tail"] == 1.00
    assert rc["orient_terrain_relief"] == 0.70
    assert rc["terrain_progress_full_ratio"] == 0.35
    assert rc["terrain_direction_alignment_floor"] == 0.35
    assert rc["w_base_ang_accel_raw_tail"] == 0.0
    assert rc["w_cycle_vz_bias"] == 0.0
    assert rc["w_stand_posture"] == 0.90
    assert rc["w_yaw_translation"] == 0.0
    assert rc["w_front_rear_extension"] == 0.75
    assert rc["w_contact_chatter"] == 0.65
    assert data["env"]["commands"]["transition_low_speed_v"] == 0.10
    assert data["env"]["commands"]["transition_stable_s"] == 0.08
    assert data["env"]["commands"]["transition_phase_window"] == 0.10
    assert data["env"]["commands"]["transition_handoff_s"] == 0.40
    assert data["env"]["commands"]["transition_tilt_deg"] == 6.0
    assert data["env"]["curriculum"]["phase_gate_transition_start_phase"] == 2
    assert data["env"]["curriculum"]["phase_gate_transition_fail_0"] == 0.45
    assert data["env"]["curriculum"]["phase_gate_transition_fail_1"] == 0.45
    assert data["env"]["curriculum"]["phase_gate_transition_min_events"] == 128
    assert data["env"]["curriculum"]["dr_gate_transition_start_level"] == 2
    assert data["env"]["curriculum"]["dr_gate_transition_stop_fail_final"] == 0.35
    assert data["env"]["curriculum"]["dr_gate_transition_reverse_fail_final"] == 0.30
    assert data["env"]["curriculum"]["dr_gate_transition_axis_fail_final"] == 0.40
    assert data["env"]["curriculum"]["dr_gate_transition_handoff_fail_final"] == 0.10
    assert data["env"]["curriculum"]["phase_progress_thresholds_0"]["back"] == 0.60
    assert data["env"]["curriculum"]["phase_progress_thresholds_0"]["lat"] == 0.60
    assert data["env"]["curriculum"]["phase_progress_thresholds_0"]["yaw"] == 0.35
    assert data["env"]["curriculum"]["phase_progress_thresholds_1"]["yaw"] == 0.35


def test_yaw_progress_threshold_is_normalized_for_phase_gate():
    cfg = SimpleNamespace(
        training_command_mode="phase_curriculum",
        training_phase_commands={0: {"command_mode": "single_axis", "active_dirs": ["fwd", "back", "lat", "yaw"]}},
        init_phase=0,
        phase_gate_prog_0=0.65,
        phase_progress_dirs_0=["fwd", "back", "lat", "yaw"],
        phase_progress_thresholds_0={"fwd": 0.65, "back": 0.60, "lat": 0.60, "yaw": 0.35},
    )
    progress = {"fwd": 0.80, "back": 0.78, "lat": 0.76, "yaw": 0.35}
    val, active = C.active_direction_progress(progress, cfg, 0)

    assert active == ("fwd", "back", "lat", "yaw")
    assert abs(val - 0.65) < 1e-9


def _runtime_cfg_shell():
    return SimpleNamespace(
        robot=SimpleNamespace(
            actuators={
                "legs": SimpleNamespace(stiffness={}, damping={}),
            },
        ),
        scene=SimpleNamespace(num_envs=0, env_spacing=0.0),
        terrain=SimpleNamespace(
            terrain_generator=SimpleNamespace(
                sub_terrains={
                    "stairs": SimpleNamespace(),
                    "boxes": SimpleNamespace(),
                },
            ),
        ),
    )


def _expect_runtime_value(value):
    return C.as_tuple(value)


def test_real_yaml_env_sections_reach_runtime_cfg():
    data = C.load_taili_blind_config()
    env = data["env"]
    cfg = _runtime_cfg_shell()
    C.apply_env_config_to_cfg(cfg, data)

    for key, value in env["control"].items():
        assert getattr(cfg, key) == _expect_runtime_value(value)

    assert cfg.gait_period == env["gait"]["period"]
    assert cfg.gait_period_min == env["gait"]["period_min"]
    assert cfg.gait_period_slope == env["gait"]["period_slope"]
    assert cfg.gait_yaw_speed_equiv == env["gait"]["yaw_speed_equiv"]
    assert cfg.gait_duty == env["gait"]["duty"]

    assert cfg.actuator_stiffness == env["actuator"]["stiffness"]
    assert cfg.actuator_damping == env["actuator"]["damping"]
    assert cfg.robot.actuators["legs"].stiffness[".*_hip_joint"] == env["actuator"]["stiffness"]["hip"]
    assert cfg.robot.actuators["legs"].damping[".*_calf_joint"] == env["actuator"]["damping"]["calf"]

    assert cfg.scene.num_envs == env["scene"]["num_envs"]
    assert cfg.scene.env_spacing == env["scene"]["env_spacing"]

    command_mapping = {
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
        "cmd_transition_zero_frac_min": "transition_zero_frac_min",
        "cmd_transition_zero_frac_max": "transition_zero_frac_max",
        "cmd_transition_stop_zero_frac": "transition_stop_zero_frac",
        "cmd_transition_stable_s": "transition_stable_s",
        "cmd_transition_wxy_max": "transition_wxy_max",
        "cmd_transition_tilt_deg": "transition_tilt_deg",
        "cmd_transition_joint_speed_max": "transition_joint_speed_max",
        "cmd_transition_action_rate_max": "transition_action_rate_max",
        "cmd_transition_handoff_s": "transition_handoff_s",
        "cmd_fwd_max": "fwd_max",
        "cmd_back_max": "back_max",
        "cmd_lat_max": "lat_max",
        "cmd_yaw_max": "yaw_max",
    }
    for attr, key in command_mapping.items():
        assert getattr(cfg, attr) == _expect_runtime_value(env["commands"][key])

    for key, value in env["curriculum"].items():
        assert getattr(cfg, key) == _expect_runtime_value(value)

    dr_mapping = {
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
    for attr, key in dr_mapping.items():
        assert getattr(cfg, attr) == _expect_runtime_value(env["domain_randomization"][key])

    for key, value in env["blind_overrides"].items():
        assert getattr(cfg, key) == _expect_runtime_value(value)

    assert cfg.terrain.terrain_generator.sub_terrains["stairs"].step_height_range == tuple(
        env["terrain"]["stairs"]["step_height_range"]
    )
    assert cfg.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range == tuple(
        env["terrain"]["boxes"]["grid_height_range"]
    )

    recipe = data["training_recipe"]
    assert cfg.training_recipe_id == recipe["id"]
    assert cfg.training_command_mode == recipe["command_mode"]
    assert cfg.touchdown_impact_only == recipe["touchdown_impact_only"]
    assert cfg.init_phase == recipe["init_phase"]
    assert cfg.training_phase_commands == recipe["phases"]


def test_real_yaml_reward_keys_are_known_reward_config_fields():
    data = C.load_taili_blind_config()
    known = {field.name for field in fields(RewardConfig)}
    unknown = sorted(set(data["reward"]) - known)
    assert unknown == []


def test_command_transition_is_a_short_soft_quality_window():
    data = C.load_taili_blind_config()
    assert data["env"]["commands"]["transition_max_s"] == 0.35


def test_discrete_terrain_recovery_keeps_flat_termination_unchanged():
    overrides = C.load_taili_blind_config()["env"]["blind_overrides"]
    assert overrides["terrain_body_collision_force_start"] == 5.0
    assert overrides["terrain_body_collision_force_span"] == 45.0
    assert overrides["terrain_recovery_termination_height"] == 0.18
    assert overrides["terrain_recovery_termination_tilt_deg"] == 60.0


def test_current_stability_thresholds_and_yaw_duty_semantics():
    reward = C.load_taili_blind_config()["reward"]
    assert reward["orient_soft_rad"] == 0.025
    assert reward["tracking_wxy_target"] == 0.15
    assert reward["cycle_wxy_rms_free"] == 0.10
    assert reward["duty_balance_yaw_scale"] == 0.50


def test_domain_randomization_starts_with_light_actuator_gain_variation():
    data = C.load_taili_blind_config()
    env = data["env"]
    dr = env["domain_randomization"]
    assert env["curriculum"]["dr_start_phase"] == 0
    assert dr["start_level"] == 0
    assert dr["unlock_terrain"] == 6.0
    assert dr["stiffness_scale_1"] == [0.9, 1.1]
    assert dr["damping_scale_1"] == [0.85, 1.15]
    assert dr["apply_prob_1"] == 0.5


def test_current_global_reward_coupling_values():
    reward = C.load_taili_blind_config()["reward"]
    assert reward["validated_tracking_floor"] == 0.15
    assert reward["yaw_tracking_base_fraction"] == 0.15
    assert reward["yaw_progress_quality_floor"] == 0.50
    assert reward["yaw_tracking_wxy_target"] == 0.22
    assert reward["yaw_tracking_wxy_width"] == 0.50
    assert reward["linear_underspeed_min_ratio"] == 0.85
    assert reward["w_cycle_wxy_bias"] == 0.0
    assert reward["w_cycle_wxy_bias_late"] == 0.0
    assert reward["quality_reward_progress_start"] == 0.05
    assert reward["quality_reward_progress_full"] == 0.45
    assert reward["quality_reward_floor"] == 0.55
    assert reward["refinement_reward_floor"] == 0.25
    assert reward["w_supported_progress"] == 0.75
    assert reward["direction_progress_full_ratio"] == 0.60
    assert reward["w_terrain_progress"] == 0.0
    assert reward["terrain_clearance_drive"] == 0.0
    assert reward["terrain_tracking_min_scale"] == 0.30
    assert reward["w_base_ang_accel"] == 0.0
    assert reward["w_base_ang_accel_raw_tail"] == 0.0
    assert reward["base_ang_accel_scale"] == 6.0
    assert reward["base_ang_vel_filter_beta"] == 0.55
    assert reward["w_lateral_coordinated_progress"] == 0.0
    assert reward["w_lateral_pair_velocity"] == 0.65
    assert reward["lateral_quality_floor"] == 0.20
    assert reward["w_directional_support_balance"] == 1.35
    assert reward["w_transition_readiness"] == 0.02
    assert reward["transition_tracking_floor"] == 1.0
    assert reward["transition_gait_floor"] == 1.0
    assert reward["w_foot_trajectory"] == 0.0
    assert reward["w_terminal_swing_velocity"] == 0.80
    assert reward["terminal_swing_vz_free"] == 0.12
    assert reward["w_landing_impact_tail"] == 1.35
    assert reward["w_landing_impact_tail_late"] == 0.55
    assert reward["touchdown_hold_scale"] == 0.65
    assert reward["w_cycle_vz_bias"] == 0.0
    assert reward["w_stand_posture"] == 0.90
    assert reward["w_yaw_translation"] == 0.0
    assert reward["w_front_rear_extension"] == 0.75
    assert reward["w_contact_chatter"] == 0.65
    assert reward["touchdown_vz_free"] == 0.15
    assert reward["w_gait_anchor"] == 1.20
    assert reward["w_gait_phase_mismatch"] == 0.45
    assert reward["w_excess_support"] == 0.60
    assert reward["w_contact_exchange"] == 0.80
    assert reward["w_contact_period"] == 0.50
    assert reward["w_feet_air_time"] == 0.60
    assert reward["w_action_magnitude"] == 0.0
    assert reward["w_yaw_underspeed"] == 0.0
    assert reward["yaw_underspeed_min_ratio"] == 0.75
    assert reward["off_axis_free"] == 0.03
    assert reward["off_axis_scale"] == 0.25
    assert reward["w_orient"] == 1.40
    assert reward["w_base_vz"] == 2.0
    assert reward["w_base_wxy"] == 1.40
    assert reward["w_support_integrity"] == 3.0
    assert reward["w_diagonal_contact"] == 1.3
    assert reward["w_terminal"] == 60.0
    overrides = C.load_taili_blind_config()["env"]["blind_overrides"]
    assert overrides["w_settle_brake"] == 0.08
    assert overrides["w_transition_failure"] == 0.0


def test_direction_drive_budgets_are_equal_and_terrain_does_not_double_pay():
    reward = C.load_taili_blind_config()["reward"]
    linear_budget = (
        reward["w_tracking_lin"]
        + reward["w_track_far"]
        + reward["w_supported_progress"]
    )
    lateral_budget = linear_budget + reward["w_lateral_coordinated_progress"]
    yaw_budget = (
        reward["w_tracking_yaw"]
        + reward["w_yaw_far"]
        + reward["w_yaw_progress"]
    )

    assert linear_budget == pytest.approx(4.0)
    assert lateral_budget == pytest.approx(4.0)
    assert yaw_budget == pytest.approx(4.0)
    assert reward["w_terrain_progress"] == 0.0


def test_duplicate_reward_owners_are_disabled_in_the_runtime_config():
    data = C.load_taili_blind_config()
    reward = data["reward"]
    disabled = (
        "w_linear_underspeed",
        "w_yaw_underspeed",
        "w_wrong_dir",
        "w_terrain_progress",
        "w_lateral_coordinated_progress",
        "w_yaw_support_moment",
        "w_yaw_wrong_moment",
        "w_planar_purity",
        "w_yaw_translation",
        "w_foot_trajectory",
        "w_action_magnitude",
        "w_base_ang_accel",
        "w_base_ang_accel_raw_tail",
        "w_cycle_wxy_bias",
        "w_cycle_wxy_bias_late",
        "w_cycle_vz_bias",
        "w_cycle_yaw_residual",
    )
    assert {name: reward[name] for name in disabled} == {
        name: 0.0 for name in disabled
    }
    assert data["env"]["blind_overrides"]["w_imitate_live"] == 0.60
    assert data["skrl"]["agent"]["style_reward_weight"] == 1.00


def test_full_command_envelope_is_present_from_phase0():
    phases = C.load_taili_blind_config()["training_recipe"]["phases"]
    expected = {
        "fwd_range": [0.15, 1.0],
        "back_range": [0.15, 0.8],
        "lat_range": [0.12, 0.5],
        "yaw_range": [0.15, 1.0],
    }
    for phase in phases.values():
        for name, bounds in expected.items():
            assert phase[name] == bounds


def test_stair_clearance_uses_only_a_short_contact_trace_without_potential_credit():
    data = C.load_taili_blind_config()
    overrides = data["env"]["blind_overrides"]
    assert "terrain_event_duration_s" not in overrides
    assert 0.0 < overrides["terrain_collision_trace_decay_time"] < 1.0
    assert "terrain_progress_height_scale" not in overrides
    assert "terrain_progress_regression_weight" not in overrides
    assert "terrain_progress_delta_cap" not in overrides


def test_skrl_config_uses_model_section_values():
    data = C.load_taili_blind_config()
    skrl = C.build_skrl_config(data)
    assert skrl["models"]["policy"]["actor_hidden"] == data["model"]["actor"]["hidden"]
    assert skrl["models"]["policy"]["dropout"] == data["model"]["terrain_perceiver"]["dropout"]
