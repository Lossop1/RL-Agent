"""Taili 生产奖励与稳定门控。

这是训练环境和奖励预算工具共用的唯一奖励实现：训练环境直接调用
compute_reward_components，预算工具通过 adapter 调同一套函数，避免出现第二套公式。

约定：
- 普通惩罚项先规约成每个机器人的标量，通常用均值或比例，不直接按关节/足端求和。
- 速度跟踪使用以验收带宽为 sigma 的指数核。
- collapse/terminal 与 timeout 区分处理；timeout 不作为失败惩罚。
- inp / cfg 遵循预算工具的数据结构，因此同一函数可以被测试覆盖。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch

from . import taili_geometry as geometry


@dataclass
class RewardConfig:
    """生产奖励配置。

    字段名与预算后端的 RewardBudgetConfig 保持一致，使训练和预算测试共用同一函数。
    """
    # 跟踪带宽：等价于速度验收带宽。
    sigma_lin_abs: float = 0.10
    sigma_lin_rel: float = 0.15
    sigma_yaw: float = 0.15
    sigma_stand_speed: float = 0.05
    sigma_stand_yaw: float = 0.05
    # 站立质量核：动作尺度不能过窄，否则正常动作会让 stand Gaussian 近似为 0。
    sigma_stand_pose: float = 0.20
    sigma_stand_action: float = 0.30
    # 单一机身高度来源。h_ok / h_gate_close 由 nominal_base_h 派生：
    #   h_ok = nominal_base_h - 0.05，h_gate_close = nominal_base_h - 0.10。
    # 低高度不单独加惩罚，而是通过 stable_motion_gate 和终止逻辑表达。
    nominal_base_h: float = geometry.NOMINAL_BASE_HEIGHT
    h_ok: float = geometry.NOMINAL_BASE_HEIGHT - 0.05
    h_gate_close: float = geometry.NOMINAL_BASE_HEIGHT - 0.10
    tilt_ok_rad: float = 0.2617993878          # 15 度。
    tilt_gate_close_rad: float = 0.6981317008  # 40 度。
    # 落脚、滑移、抬脚和地形相关尺度。
    flat_clearance_target: float = 0.08
    terrain_clearance_margin: float = 0.04
    terrain_clearance_margin_min: float = 0.02
    terrain_clearance_margin_gain: float = 0.05
    terrain_clearance_margin_max: float = 0.035
    clearance_band: float = 0.02
    terrain_v_floor: float = 0.10
    # 能力项的软稳定下限。真实终止仍硬置零，正常但尚不稳定的探索保留梯度。
    capability_gate_floor: float = 0.25
    # 奖励权重。
    w_tracking_lin: float = 2.0
    w_tracking_yaw: float = 1.0
    # yaw_progress 直接奖励“按命令方向实际转起来”，避免 yaw 只靠误差核和远场 shaping。
    w_yaw_progress: float = 0.0
    yaw_progress_quality_floor: float = 0.35
    # 欠速按命令比例归一化，使低速命令不会退化成几乎无代价的原地站立。
    w_yaw_underspeed: float = 0.0
    yaw_underspeed_min_ratio: float = 0.75
    w_stand: float = 1.5
    # 站立高斯乘积用于精确收敛；独立姿态代价在任一分量较差时仍保留恢复梯度。
    w_stand_posture: float = 0.0
    stand_pose_free: float = 0.03
    stand_pose_scale: float = 0.20
    stand_height_band: float = 0.015
    stand_height_scale: float = 0.04
    # 四足站稳：stand 项只看机身静止，不能防止脚下碎步；该项要求零命令下四足接触。
    # 它受 stand_gate 门控，不会在行走时形成站立陷阱。
    w_stand_contact: float = 0.8
    # 命令方向推进服务所有线性命令，并在不规则接触后保持完整强度。精确速度
    # 大小由 near/far tracking 单独负责，地形上允许连续退让。
    w_supported_progress: float = 0.5
    direction_progress_full_ratio: float = 0.60
    direction_progress_support_floor: float = 0.0
    # 平地推进必须同时保持可控机身和轻触地；真实地形响应后退回方向主驱动。
    direction_progress_quality_floor: float = 0.0
    # 旧的接触后专属推进项保留兼容键，默认不参与总奖励。
    w_terrain_progress: float = 0.0
    # 地形允许低速稳定通过，但方向推进在碰障静止点仍需保留非零梯度。
    terrain_progress_full_ratio: float = 0.35
    terrain_direction_alignment_floor: float = 0.35
    # 地形条件下的进展耦合：保留进展 floor，避免早期无梯度。
    # 完整进展奖励要求低滑移、真实支撑和受控机身运动。
    healthy_progress_floor: float = 0.35
    quality_reward_progress_start: float = 0.20
    quality_reward_progress_full: float = 0.65
    quality_reward_floor: float = 0.35
    refinement_reward_progress_start: float = 0.15
    refinement_reward_progress_full: float = 0.55
    refinement_reward_floor: float = 0.08
    healthy_progress_slip_target: float = 0.12
    healthy_progress_slip_width: float = 0.18
    terrain_clearance_drive: float = 0.35
    # 真实离散地形事件发生后，立即释放平地运动学风格预算。
    terrain_style_max_relief: float = 0.65
    tracking_touchdown_clean_floor: float = 0.65
    gait_anchor_progress_floor: float = 0.20
    gait_anchor_min_ratio: float = 0.60
    linear_underspeed_min_ratio: float = 0.65
    w_linear_underspeed: float = 0.0
    # 跟踪质量门只用于诊断和课程验收，不再缩放基础任务奖励。
    validated_tracking_floor: float = 0.40
    tracking_validity_terrain_mix: float = 1.0
    terrain_tracking_min_scale: float = 0.30
    tracking_posture_floor: float = 0.55
    # 连续执行年龄只用于诊断，不能延迟基础任务信用。
    task_credit_floor: float = 0.20
    task_credit_start_s: float = 1.50
    task_credit_full_s: float = 3.00
    yaw_tracking_drift_target: float = 0.10
    yaw_tracking_drift_width: float = 0.25
    yaw_tracking_wxy_target: float = 0.35
    yaw_tracking_wxy_width: float = 0.65
    yaw_tracking_support_floor: float = 0.35
    yaw_support_contact_target: float = 2.25
    yaw_support_contact_width: float = 0.75
    # 机身运动质量只用于诊断和质量验收；姿态代价以加法进入总奖励。
    tracking_motion_floor: float = 0.45
    tracking_wxy_target: float = 0.25
    tracking_wxy_width: float = 0.45
    tracking_vz_target: float = 0.12
    tracking_vz_width: float = 0.30
    # yaw 不强制直线对角步态，但支撑占空质量过差时不能获得完整转向收益。
    yaw_duty_quality_floor: float = 0.45
    yaw_tracking_base_fraction: float = 0.60
    w_yaw_support_moment: float = 0.0
    w_yaw_wrong_moment: float = 0.0
    w_torque_margin: float = 1.0
    w_torque_saturation: float = 5.0
    w_clearance_under: float = 0.4
    w_clearance_over: float = 0.05
    clearance_worst_leg_mix: float = 0.50
    w_off_axis: float = 0.5
    off_axis_free: float = 0.03
    off_axis_scale: float = 0.25
    w_planar_purity: float = 0.0
    planar_purity_yaw_scale: float = 0.40
    w_heading_hold: float = 0.0
    heading_hold_scale: float = 0.35
    # 纯 yaw 的角速度跟踪不能掩盖伴随平移。
    w_yaw_translation: float = 0.0
    yaw_translation_free: float = 0.03
    yaw_translation_scale: float = 0.25
    w_action_rate: float = 0.02
    # 平地动作目标超过正常关节行程时持续惩罚；真实碰障后允许扩大动作幅度。
    w_action_magnitude: float = 0.0
    action_magnitude_free: float = 1.0
    action_magnitude_scale: float = 1.0
    action_magnitude_terrain_relief: float = 0.75
    w_transition_readiness: float = 0.0
    transition_tracking_floor: float = 1.0
    transition_gait_floor: float = 1.0
    transition_trajectory_floor: float = 0.15
    w_foot_trajectory: float = 0.0
    foot_trajectory_error_scale: float = 0.35
    foot_trajectory_back_scale: float = 1.20
    w_terminal_swing_velocity: float = 0.0
    terminal_swing_vz_free: float = 0.15
    terminal_swing_vz_scale: float = 0.35
    w_stance_slip: float = 0.05
    w_stance_slip_late: float = 0.20
    # 滑移尾部惩罚：除均值外，直接惩罚稳定接触脚超过高滑移阈值的比例。
    # 该项通过 quality_gate 渐入，避免基础步态尚未形成时过早压制。
    w_stance_slip_high: float = 0.0
    w_stance_slip_high_late: float = 0.60
    w_worst_leg_slip: float = 0.0
    # 同一只脚过早再次触地表示碎步或接触抖动；阈值按当前目标周期缩放。
    w_contact_chatter: float = 0.0
    contact_chatter_min_period_ratio: float = 0.65
    w_directional_support_balance: float = 0.0
    linear_front_rear_support_scale: float = 0.35
    w_lateral_coordinated_progress: float = 0.0
    lateral_quality_floor: float = 0.30
    w_lateral_pair_velocity: float = 0.0
    lateral_pair_velocity_free: float = 0.12
    lateral_pair_velocity_scale: float = 0.60
    w_cycle_wxy_bias: float = 0.0
    w_cycle_wxy_bias_late: float = 0.0
    cycle_wxy_rms_free: float = 0.18
    cycle_wxy_rms_scale: float = 0.45
    # 周期垂向速度能量直接约束机身上下起伏，避免均值高度掩盖弹跳。
    w_cycle_vz_bias: float = 0.0
    cycle_vz_rms_free: float = 0.08
    cycle_vz_rms_scale: float = 0.22
    w_cycle_yaw_residual: float = 0.0
    cycle_yaw_rms_free: float = 0.08
    cycle_yaw_rms_scale: float = 0.35
    # 落脚冲击尾部惩罚：关注最重的一只脚，而不是只看四足均值。
    w_landing_impact_tail: float = 0.0
    w_landing_impact_tail_late: float = 0.50
    # 脚前勾修正：惩罚触地瞬间足端水平速度，补上只看竖直落脚速度的盲区。
    # 默认权重为 0 时不改变行为；非零时通过 quality_gate 后期渐入。
    w_touchdown_slip: float = 0.0
    w_touchdown_slip_late: float = 0.60
    w_touchdown_slip_tail: float = 0.0
    w_touchdown_force_rate: float = 0.0
    touchdown_force_rate_scale: float = 0.45
    touchdown_hold_s: float = 0.10
    touchdown_hold_scale: float = 0.30
    w_swing_action_accel: float = 0.0
    swing_action_accel_scale: float = 0.30
    foot_trajectory_velocity_mix: float = 0.45
    foot_trajectory_terminal_vz_scale: float = 0.35
    # 足端每控制步速度变化的软尺度（m/s）。与 action 二阶差分共用一个平滑权重，
    # 避免再增加一套独立奖励预算。
    swing_foot_velocity_delta_scale: float = 0.80
    # 分段滑移形状：稳定支撑脚速度高于 slip_free_speed 后线性惩罚，到 slip_speed_scale 饱和。
    # slip_free_speed 是测量代理的底噪，不表示允许滑移；低于该值惩罚会直接伤害正常行走。
    # 饱和值限制总量，线性段保留脱离滑移的梯度。
    slip_free_speed: float = 0.15
    slip_speed_scale: float = 1.0
    # 落脚冲击使用连续渐近形状：轻触地不惩罚，重落脚持续保留梯度而不硬截断。
    impact_speed_scale: float = 1.0
    touchdown_vz_free: float = 0.20
    w_landing_impact: float = 0.15
    w_landing_impact_late: float = 0.85
    # 平地步态姿态和速率质量，全部有界：
    # orient：倾斜超过约 5 度后惩罚；base_vz：竖直弹跳；base_wxy：滚转/俯仰角速度。
    # hip_deviation：髋关节偏离默认外展位置的幅度。
    w_orient: float = 0.30
    orient_soft_rad: float = 0.09
    orient_scale_rad: float = 0.35
    core_tail_gain: float = 0.25
    orient_terrain_relief: float = 0.70
    w_base_vz: float = 1.0
    base_vz_free: float = 0.05
    base_vz_scale: float = 0.75
    w_base_wxy: float = 0.10
    w_base_ang_accel: float = 0.0
    base_ang_accel_free: float = 1.5
    base_ang_accel_scale: float = 8.0
    base_ang_vel_filter_beta: float = 0.75
    # 低通与原始角加速度只保留诊断尺度，用于区分连续摆动和高频冲击。
    w_base_ang_accel_raw_tail: float = 0.0
    base_ang_accel_raw_free: float = 10.0
    base_ang_accel_raw_scale: float = 20.0
    # 平地移动时保持接近站立高度；局部障碍出现后自动放松，允许上下台阶时降低重心。
    w_flat_move_height: float = 0.0
    flat_move_height_target: float = geometry.NOMINAL_BASE_HEIGHT
    flat_move_height_band: float = 0.03
    flat_move_height_terrain_relief: float = 0.06
    # 统一核心质量正奖励：稳定姿态、非命令角速度、机身高度和垂向速度共同构成
    # 一个可达吸引子。现有加性代价仍负责错误状态的纠偏，二者不互相门控。
    w_core_quality: float = 0.0
    core_tilt_scale: float = 0.035
    core_ang_vel_scale: float = 0.20
    core_height_scale: float = 0.020
    core_vz_scale: float = 0.12
    core_progress_floor: float = 0.20
    core_terrain_relief: float = 0.70
    w_support_integrity: float = 0.0
    support_integrity_min_contacts: float = 2.0
    support_integrity_wxy_soft: float = 0.9
    support_integrity_wxy_hard: float = 2.4
    w_hip_deviation: float = 0.40
    hip_deviation_free: float = 0.04
    hip_deviation_scale: float = 0.50
    hip_deviation_lat_scale: float = 0.35
    hip_deviation_yaw_scale: float = 0.25
    # 平地前后腿平均伸展应一致；横移和 yaw 保留必要的载荷重排空间。
    w_front_rear_extension: float = 0.0
    front_rear_extension_free: float = 0.015
    front_rear_extension_scale: float = 0.05
    front_rear_extension_lat_yaw_scale: float = 0.50
    w_diagonal_contact: float = 0.20
    w_duty_balance: float = 0.20
    duty_target: float = 0.50
    duty_linear_min: float = 0.42
    duty_linear_max: float = 0.68
    duty_linear_margin: float = 0.18
    duty_symmetry_tolerance: float = 0.25
    duty_cycle_ema_beta: float = 0.60
    gait_period_tolerance: float = 0.22
    diagonal_contact_yaw_scale: float = 0.15
    duty_balance_yaw_scale: float = 0.50
    quality_window_ema_beta: float = 0.98
    slip_high_threshold: float = 0.20
    # 正向步态锚点：奖励接触节律匹配 gait clock，使小跑节律成为吸引子。
    w_gait_anchor: float = 1.0
    gait_false_contact_scale: float = 2.0
    # 正向锚点负责吸引，密集错配代价负责把错误接触结构推出零梯度区。
    w_gait_phase_mismatch: float = 0.0
    w_excess_support: float = 0.0
    # 正向接触交换：只在真实完成过一个测量周期后，奖励对角接触与 duty 同时成立。
    # 这复用现有质量窗口，避免 reset 默认值或单帧碰巧接触获得虚假信用。
    w_contact_exchange: float = 0.0
    # 实际接触周期必须直接参与优化，不能只作为课程 gate。
    w_contact_period: float = 0.0
    contact_period_free: float = 0.05
    contact_period_scale: float = 0.15
    w_feet_air_time: float = 1.0
    air_time_target: float = 0.30
    # 反向运动压力：惩罚与线速度命令或 yaw 符号相反的运动。
    w_wrong_dir: float = 0.5
    rear_duty_tolerance: float = 0.05
    w_terminal: float = 10.0
    # 远场 shaping：Laplacian exp(-err/scale) 在大误差区仍有梯度。
    # 它负责把策略从差初始化拉向目标，窄 Gaussian 负责最后精度。
    w_track_far: float = 1.0
    w_yaw_far: float = 0.5
    w_stand_far: float = 0.8
    scale_track_far: float = 0.35
    scale_yaw_far: float = 0.35
    scale_stand_far: float = 0.3


def default_reward_cfg() -> RewardConfig:
    try:
        try:
            from ..blind_locomotion.taili_blind_config import reward_config_mapping
        except Exception:
            from ..taili_blind_config import reward_config_mapping
        values = reward_config_mapping()
        valid = {field.name for field in fields(RewardConfig)}
        return RewardConfig(**{k: v for k, v in values.items() if k in valid})
    except Exception:
        return RewardConfig()


def reward_cfg_from_env(base: "RewardConfig | None" = None) -> RewardConfig:
    """从环境变量覆盖奖励配置。

    任意数值字段都可以用 TAILI_RW_<FIELD> 覆盖，例如：
    `TAILI_RW_W_TRACK_FAR=2.0 python tp_train.py`。
    未设置的字段保留 dataclass 默认值。
    """
    import dataclasses
    import os
    cfg = base if base is not None else default_reward_cfg()
    applied = {}
    for f in dataclasses.fields(cfg):
        env = os.environ.get("TAILI_RW_" + f.name.upper())
        if env is None:
            continue
        cur = getattr(cfg, f.name)
        try:
            setattr(cfg, f.name, type(cur)(env) if isinstance(cur, (int, float)) and not isinstance(cur, bool) else cur)
            applied[f.name] = env
        except (ValueError, TypeError):
            pass
    if applied:
        try:
            print(f"[taili_reward] env overrides: {applied}", flush=True)
        except Exception:
            pass
    # 即使使用 TAILI_RW_* 覆盖，也保持高度门槛由 nominal_base_h 单一派生。
    cfg.h_ok = float(cfg.nominal_base_h) - 0.05
    cfg.h_gate_close = float(cfg.nominal_base_h) - 0.10
    return cfg


def exp_kernel(err, sigma):
    sigma = torch.clamp(torch.as_tensor(sigma, dtype=err.dtype, device=err.device), min=1e-9)
    return torch.exp(-((err / sigma) ** 2))


def bounded_overspeed_quality(v_along, command_speed, tolerance):
    """只压低超速的正收益；欠速仍由原有推进比例表达。"""
    overspeed = torch.clamp(v_along - command_speed, min=0.0)
    return exp_kernel(overspeed, tolerance)


def terrain_amp_style_scale(
    terrain_response,
    quality_gate,
    max_relief=0.80,
    *,
    relief_progress=None,
    early_relief=0.10,
):
    """按真实接触后事件退让风格，平地始终保持满额 AMP。

    ``relief_progress`` 为空时保留连续地形原有语义。离散障碍传入碰撞后事件
    作用域时立即使用完整退让预算，不能要求策略先完成探步才解除平地参考。
    """
    response = torch.clamp(terrain_response, 0.0, 1.0)
    quality = torch.clamp(quality_gate, 0.0, 1.0)
    max_relief = min(max(float(max_relief), 0.0), 1.0)
    if relief_progress is None:
        relief = torch.full_like(response, max_relief)
        relief_gate = quality
    else:
        progress = torch.clamp(
            torch.as_tensor(relief_progress, dtype=response.dtype, device=response.device),
            0.0,
            1.0,
        )
        early_relief = min(max(float(early_relief), 0.0), max_relief)
        relief = early_relief + (max_relief - early_relief) * progress
        # 事件已经由真实碰撞限定作用域，退让不能再受平地成熟度限制。
        relief_gate = torch.ones_like(quality)
    return 1.0 - relief * response * relief_gate


def adaptive_terrain_clearance_margin(
    obstacle_height,
    *,
    minimum: float = 0.02,
    gain: float = 0.05,
    maximum: float = 0.035,
):
    """按障碍高度给出小幅安全余量，避免低台阶过抬和高台阶余量不足。"""
    height = torch.clamp(torch.as_tensor(obstacle_height), min=0.0)
    lo = max(float(minimum), 0.0)
    hi = max(float(maximum), lo)
    return torch.clamp(lo + float(gain) * height, min=lo, max=hi)


def terrain_clearance_targets(
    foot_clearance,
    obstacle_height,
    *,
    flat_target,
    margin,
    margin_min=None,
    margin_gain=None,
    margin_max=None,
    lead_foot_mask=None,
    event_direction=None,
    event_ramp=None,
    event_probe_target=None,
):
    """碰障后提高当前上行响应脚目标，其余脚保持普通 clearance。

    严格盲态训练传入 ``event_probe_target``：它是能力上限而不是扫描到的台阶高度，
    只在被动碰障事件后生效。未传入时保留连续粗糙地形的既有目标语义。
    """
    flat = torch.full_like(foot_clearance, float(flat_target))
    obstacle_height = torch.as_tensor(
        obstacle_height, dtype=foot_clearance.dtype, device=foot_clearance.device
    ).reshape(-1, 1)
    if margin_min is None or margin_gain is None or margin_max is None:
        adaptive_margin = torch.full_like(obstacle_height, float(margin))
    else:
        adaptive_margin = adaptive_terrain_clearance_margin(
            obstacle_height,
            minimum=float(margin_min),
            gain=float(margin_gain),
            maximum=float(margin_max),
        )
    elevated = torch.maximum(obstacle_height + adaptive_margin, flat)
    if event_probe_target is not None:
        probe_target = torch.as_tensor(
            event_probe_target,
            dtype=foot_clearance.dtype,
            device=foot_clearance.device,
        ).reshape(-1, 1)
        elevated = torch.maximum(probe_target, flat)
    if lead_foot_mask is None:
        lead = torch.ones_like(foot_clearance)
    else:
        lead = lead_foot_mask.to(dtype=foot_clearance.dtype, device=foot_clearance.device)
    if event_direction is None:
        up_event = torch.ones_like(obstacle_height)
    else:
        direction = torch.as_tensor(
            event_direction, dtype=foot_clearance.dtype, device=foot_clearance.device
        ).reshape(-1, 1)
        up_event = (direction > 0).to(foot_clearance.dtype)
    if event_ramp is None:
        ramp = torch.ones_like(obstacle_height)
    else:
        ramp = torch.clamp(
            torch.as_tensor(event_ramp, dtype=foot_clearance.dtype, device=foot_clearance.device),
            0.0,
            1.0,
        ).reshape(-1, 1)
    target_mix = lead * up_event * ramp
    return flat + target_mix * (elevated - flat)


def tolerance_score(error, tolerance):
    """把非负误差映射为分数，tolerance 对应 0.5 分。"""
    half_score_error = max(float(tolerance), 1e-6)
    return 1.0 - torch.clamp(error / (2.0 * half_score_error), 0.0, 1.0)


def trajectory_quality_score(error, scope, scale, terrain_pattern_scale=None):
    """把统一足端轨迹误差映射为可复用的有效性分数。

    平地稳态完整使用轨迹质量；过渡窗口不施加新方向参考；真实地形响应后按现有
    ``terrain_pattern_scale`` 退让。这样任务奖励、轨迹惩罚和阶段验收使用同一语义。
    """
    error = torch.clamp(torch.as_tensor(error), min=0.0)
    scope = torch.clamp(
        torch.as_tensor(scope, dtype=error.dtype, device=error.device), 0.0, 1.0
    )
    half_score_error = max(float(scale), 1e-6)
    raw_quality = half_score_error / (error + half_score_error)
    scoped_quality = 1.0 - scope * (1.0 - raw_quality)
    if terrain_pattern_scale is None:
        return scoped_quality
    pattern_scale = torch.clamp(
        torch.as_tensor(terrain_pattern_scale, dtype=error.dtype, device=error.device),
        0.0,
        1.0,
    )
    return 1.0 - pattern_scale * (1.0 - scoped_quality)


def duty_symmetry_score(error, tolerance):
    """支撑占空对称分；保留具名入口以明确遥测语义。"""
    return tolerance_score(error, tolerance)


def settle_brake_potential_delta(previous_speed, current_speed):
    """过渡制动的有符号速度势差：减速为正，加速为负。"""
    return previous_speed - current_speed


def settle_brake_signal(previous_speed, current_speed, near_zero_scale=0.18):
    """组合有符号减速势差与近零速保持奖励。"""
    delta = settle_brake_potential_delta(previous_speed, current_speed)
    near_zero = torch.exp(-torch.square(current_speed / max(float(near_zero_scale), 1e-6)))
    return 2.0 * delta + 0.20 * near_zero


def transition_style_weight(braking, releasing, release_progress):
    """过渡期风格锚点权重：制动时关闭，释放时用五次曲线平滑恢复。"""
    progress = torch.clamp(
        torch.as_tensor(release_progress),
        0.0,
        1.0,
    )
    smooth = progress.pow(3) * (10.0 + progress * (-15.0 + 6.0 * progress))
    braking_b = torch.as_tensor(braking, device=progress.device).bool()
    releasing_b = torch.as_tensor(releasing, device=progress.device).bool()
    return torch.where(
        braking_b,
        torch.zeros_like(progress),
        torch.where(releasing_b, smooth, torch.ones_like(progress)),
    )


def transition_motion_fault(base_lin_vel_xy, base_yaw_rate, old_axis, require_stop,
                            low_linear_speed, low_yaw_speed, old_command=None):
    """统一过渡动量语义。

    停车需要清除全部平面和 yaw 动量；反向与换轴只清除沿旧命令方向的
    不兼容动量。传入 old_command 后，平滑穿越零点不会因为新方向速度已经
    建立而再次被判成旧动量故障。
    """
    low_v = max(float(low_linear_speed), 1e-6)
    low_w = max(float(low_yaw_speed), 1e-6)
    axis_velocity = torch.stack((
        base_lin_vel_xy[:, 0], base_lin_vel_xy[:, 1], base_yaw_rate,
    ), dim=-1).gather(1, old_axis.long().unsqueeze(-1)).squeeze(-1)
    axis_scale = torch.where(
        old_axis.long() == 2,
        torch.full_like(axis_velocity, low_w),
        torch.full_like(axis_velocity, low_v),
    )
    if old_command is None:
        axis_fault = axis_velocity.abs() / axis_scale
    else:
        old_component = old_command.to(dtype=axis_velocity.dtype, device=axis_velocity.device).gather(
            1, old_axis.long().unsqueeze(-1)
        ).squeeze(-1)
        old_sign = torch.where(old_component.abs() > 1e-6, torch.sign(old_component), torch.ones_like(old_component))
        axis_fault = torch.clamp(axis_velocity * old_sign, min=0.0) / axis_scale
    stop_fault = torch.maximum(
        torch.linalg.norm(base_lin_vel_xy, dim=-1) / low_v,
        base_yaw_rate.abs() / low_w,
    )
    return torch.where(require_stop.bool(), stop_fault, axis_fault)


def transition_failure_penalty(failed_event, weight, gate):
    """只在过渡超时发生的当帧施加代价，避免失败状态长期锁存污染奖励。"""
    return -float(weight) * failed_event.to(gate.dtype) * gate


def transition_readiness_signal(previous_fault, current_fault, settled_bonus=0.25):
    """结果驱动的过渡势差；fault 应规约到 [0, 1]，越小越接近安全释放。"""
    delta = torch.clamp(previous_fault - current_fault, -0.25, 0.25) / 0.25
    bounded_fault = torch.clamp(current_fault, min=0.0, max=1.0)
    near_ready = torch.exp(-4.0 * bounded_fault)
    absolute_drive = 0.5 * (0.5 - bounded_fault)
    return delta + absolute_drive + float(settled_bonus) * near_ready


def transition_core_fault(terms):
    """聚合过渡核心误差，同时保留所有维度和最差维度的梯度。"""
    bounded = torch.clamp(terms, 0.0, 1.0)
    return 0.5 * bounded.mean(dim=-1) + 0.5 * bounded.amax(dim=-1)


def transition_support_phase_ready(support_count, gait_phase, phase_window, required_support=4.0):
    """判断是否处于四足接地的对角交换相位，用于软质量评价。"""
    phase = torch.remainder(gait_phase, 1.0)
    phase_distance = torch.minimum(
        torch.minimum(phase, 1.0 - phase),
        torch.abs(phase - 0.5),
    )
    return (support_count >= float(required_support)) & (phase_distance <= float(phase_window))


def update_transition_failure_ema(events, failures, value, valid, beta):
    """按完整事件批次更新失败率，空批次保持原值。"""
    events = int(events)
    if events <= 0:
        return float(value), bool(valid)
    sample = float(failures) / float(events)
    if not valid:
        return sample, True
    beta = min(max(float(beta), 0.0), 0.999)
    return beta * float(value) + (1.0 - beta) * sample, True


def smooth_bounded_excess(value, free, scale):
    """连续、有界且无硬截断的超限代价。"""
    x = torch.clamp((value - float(free)) / max(float(scale), 1e-6), min=0.0)
    return x.square() / (1.0 + x.square())


def masked_event_mean(value, event_mask):
    """只在真实事件上求均值；无事件时返回零。"""
    value = torch.as_tensor(value)
    weight = torch.clamp(
        torch.as_tensor(event_mask, dtype=value.dtype, device=value.device),
        0.0,
        1.0,
    )
    event_count = weight.sum(dim=-1)
    return (value * weight).sum(dim=-1) / event_count.clamp(min=1.0)


def huber_excess(value, free, scale):
    """超限后的 Huber 代价；大误差区保持线性梯度，不在严重摇晃时饱和。"""
    x = torch.clamp((value - float(free)) / max(float(scale), 1e-6), min=0.0)
    return torch.where(x <= 1.0, 0.5 * x.square(), x - 0.5)


def normalized_underspeed_cost(signed_speed, command_magnitude, min_ratio):
    """按命令比例计算欠速代价，站立时在所有非零速度档保持同等压力。"""
    target = command_magnitude * max(float(min_ratio), 0.0)
    deficit_ratio = torch.clamp(
        (target - signed_speed) / target.clamp(min=0.05),
        min=0.0,
    )
    return 2.0 * huber_excess(deficit_ratio, 0.0, 1.0)


def far_kernel(err, scale):
    """Laplacian shaping 核 exp(-err/scale)。

    输出有界、单调递减，并且在大误差区仍保留非零梯度，用作远场吸引项。
    """
    scale = torch.clamp(torch.as_tensor(scale, dtype=err.dtype, device=err.device), min=1e-9)
    return torch.exp(-err.abs() / scale)


def balanced_progress_geometric(values, eps: float = 1e-4) -> float:
    """用几何平均汇总分方向成熟度，防止单个易方向主导质量渐入。

    它比硬最小值平滑：所有方向都会贡献，最弱方向也不会一票否决；但只有各方向
    共同提高时结果才会明显上升，符合全方向能力共同承担质量约束的语义。
    """
    clipped = [min(max(float(value), float(eps)), 1.0) for value in values]
    if not clipped:
        return 0.0
    return math.exp(sum(math.log(value) for value in clipped) / len(clipped))


def budgeted_quality_gate(floor: float, latched: float, budget_gate: float) -> float:
    """保留质量 floor，仅让预算控制 floor 以上的成熟增量。"""
    floor_value = min(max(float(floor), 0.0), 1.0)
    latched_value = min(max(float(latched), floor_value), 1.0)
    budget_value = min(max(float(budget_gate), 0.0), 1.0)
    return floor_value + budget_value * (latched_value - floor_value)


def _smoothstep(edge0, edge1, x):
    t = torch.clamp((x - edge0) / (edge1 - edge0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def sustained_task_credit_gate(age_s, floor=0.20, start_s=1.50, full_s=3.00):
    """估计连续命令执行成熟度；该值仅用于诊断，不缩放基础任务信用。"""
    floor = min(max(float(floor), 0.0), 1.0)
    maturity = _smoothstep(float(start_s), max(float(full_s), float(start_s) + 1e-6), age_s)
    return torch.clamp(floor + (1.0 - floor) * maturity, 0.0, 1.0)


def _metric_gate(value, target, *, higher=True, width=0.10):
    """软门控，用于把进展奖励与步态健康耦合，同时保留梯度。"""
    target_t = torch.as_tensor(target, dtype=value.dtype, device=value.device)
    width_t = torch.as_tensor(width, dtype=value.dtype, device=value.device).clamp_min(1e-6)
    if higher:
        return _smoothstep(target_t - width_t, target_t, value)
    return 1.0 - _smoothstep(target_t, target_t + width_t, value)


def tracking_motion_quality(base_ang_vel, base_lin_vel, cfg, *, floor=None):
    """同时要求横滚/俯仰角速度和垂向速度可控。

    使用最差轴而不是平均值，避免一个健康分量把正在下坠的另一个分量稀释掉。
    ``floor`` 只保留早期探索梯度；传入 0 可得到严格物理质量。
    """
    wxy = torch.linalg.norm(base_ang_vel[:, :2], dim=-1)
    wxy_gate = _metric_gate(
        wxy,
        float(getattr(cfg, "tracking_wxy_target", 0.25)),
        higher=False,
        width=float(getattr(cfg, "tracking_wxy_width", 0.45)),
    )
    vz_gate = _metric_gate(
        base_lin_vel[:, 2].abs(),
        float(getattr(cfg, "tracking_vz_target", 0.12)),
        higher=False,
        width=float(getattr(cfg, "tracking_vz_width", 0.30)),
    )
    raw = torch.minimum(wxy_gate, vz_gate)
    if floor is None:
        floor = float(getattr(cfg, "tracking_motion_floor", 0.45))
    floor = min(max(float(floor), 0.0), 1.0)
    return torch.clamp(floor + (1.0 - floor) * raw, 0.0, 1.0)


def support_structure_gate(foot_contact, cmd):
    """当前命令下的最低支撑结构门控。

    这里只表达物理底线，不表达步态是否漂亮：
    - 前进/后退需要前后都有支撑，避免前脚全飞、后脚推滑导致前翻。
    - 横移需要左右都有支撑，避免单侧支撑拖成转圈。
    - 纯 yaw 只要求至少两只脚支撑，避免把 yaw 重新绑到线性步态质量上。
    """
    if foot_contact is None:
        return None
    f = torch.float64 if cmd.dtype == torch.float64 else torch.float32
    contact = foot_contact.to(dtype=f, device=cmd.device)
    contact_count = contact.sum(dim=-1)
    general = torch.clamp(contact_count - 1.0, 0.0, 1.0)
    if contact.shape[-1] < 4:
        return general

    fl, fr, rl, rr = contact[:, 0], contact[:, 1], contact[:, 2], contact[:, 3]
    front = torch.clamp(fl + fr, 0.0, 1.0)
    rear = torch.clamp(rl + rr, 0.0, 1.0)
    left = torch.clamp(fl + rl, 0.0, 1.0)
    right = torch.clamp(fr + rr, 0.0, 1.0)
    fore_aft = front * rear
    lateral = left * right

    xy_abs = cmd[:, :2].abs()
    lin_mag = xy_abs.sum(dim=-1)
    x_w = xy_abs[:, 0] / lin_mag.clamp(min=1e-6)
    y_w = xy_abs[:, 1] / lin_mag.clamp(min=1e-6)
    linear_gate = torch.where(
        lin_mag > 0.05,
        x_w * fore_aft + y_w * lateral,
        general,
    )
    yaw_mag = cmd[:, 2].abs()
    yaw_mix = yaw_mag / (yaw_mag + lin_mag + 1e-6)
    return torch.where(
        yaw_mag > 0.05,
        (1.0 - yaw_mix) * linear_gate + yaw_mix * general,
        linear_gate,
    )


def stable_motion_gate(base_h_above_terrain, tilt_rel, support_instability,
                       severe_body_collision, terminal_window, cfg, alpha=(1.0, 1.0, 1.0)):
    """两段式稳定门控：灰区软乘积，collapse/terminal 时硬置 0。

    环境和奖励共用这套定义，避免稳定性语义分叉。
    """
    g_height = _smoothstep(cfg.h_gate_close, cfg.h_ok, base_h_above_terrain)
    g_tilt = 1.0 - _smoothstep(cfg.tilt_ok_rad, cfg.tilt_gate_close_rad, tilt_rel)
    g_support = torch.clamp(1.0 - support_instability, 0.0, 1.0)
    g_soft = (g_height ** alpha[0]) * (g_tilt ** alpha[1]) * (g_support ** alpha[2])
    hard = (base_h_above_terrain < cfg.h_gate_close) | (tilt_rel > cfg.tilt_gate_close_rad) \
        | (severe_body_collision > 0.5) | (terminal_window > 0.5)
    return torch.where(hard, torch.zeros_like(g_soft), torch.clamp(g_soft, 0.0, 1.0))


# 这些量用于门控、归一化和遥测，不是独立奖励。标量 total 与多 critic 必须共用
# 同一清单，否则训练目标会因运行模式不同而变化，诊断比例也会意外成为正奖励。
REWARD_DIAGNOSTIC_COMPONENTS = (
    "stable_motion_gate", "hard_survival_gate", "capability_gate",
    "stand_gate", "moving_gate", "quality_gate", "refinement_gate",
    "support_structure_gate",
    "healthy_progress_gate", "validated_tracking_gate", "validated_yaw_tracking_gate",
    "terrain_tracking_scale", "tracking_posture_gate", "task_credit_gate", "effective_tracking_gate",
    "yaw_progress_quality_gate", "yaw_progress_overspeed_gate",
    "direction_progress_quality_gate",
    "yaw_quality_gate", "yaw_drift_gate", "yaw_wxy_gate", "yaw_duty_gate",
    "tracking_motion_gate", "touchdown_clean_gate", "progress_style_gate",
    "terrain_response", "terrain_pattern_scale", "terrain_clearance_success",
    "direction_progress_ratio", "terrain_direction_progress_ratio",
    "direction_alignment", "terrain_direction_alignment_credit",
    # 角加速度保留为诊断量，用于识别尖峰和仿真异常；核心稳定由姿态、角速度、
    # 垂向速度和高度直接负责，避免同一摇晃被高阶导数重复收费。
    "base_ang_accel",
    # 地形类型专属安全量和首次足端碰障只用于诊断。统一任务、核心、支撑与触地
    # 语义已经覆盖真实结果，不能再通过不可观测的地形标签改变 Actor 的目标。
    "terrain_contact_quality", "terrain_support_loss", "terrain_collapse",
    "terrain_overspeed", "stumble",
)


def compute_reward_components(inp, cfg):
    """计算每个机器人的奖励分量、门控诊断和 total。"""
    comp = {}
    f = torch.float64 if inp.cmd.dtype == torch.float64 else torch.float32

    gate = inp.stable_motion_gate
    stand_gate, moving_gate = inp.stand_gate, inp.moving_gate
    quality_gate = getattr(inp, "quality_gate", 1.0)
    quality_gate = torch.as_tensor(quality_gate, dtype=gate.dtype, device=gate.device)
    if quality_gate.ndim == 0:
        quality_gate = quality_gate.expand_as(gate)
    refinement_gate = getattr(inp, "refinement_gate", quality_gate)
    refinement_gate = torch.as_tensor(refinement_gate, dtype=gate.dtype, device=gate.device)
    if refinement_gate.ndim == 0:
        refinement_gate = refinement_gate.expand_as(gate)
    hard_survival_gate = getattr(inp, "hard_survival_gate", None)
    if hard_survival_gate is None:
        base_h_for_survival = torch.as_tensor(
            getattr(inp, "base_h_above_terrain", torch.full_like(gate, cfg.nominal_base_h)),
            dtype=gate.dtype,
            device=gate.device,
        )
        tilt_for_survival = torch.as_tensor(
            getattr(inp, "tilt_rel", torch.zeros_like(gate)),
            dtype=gate.dtype,
            device=gate.device,
        )
        terminal_for_survival = torch.as_tensor(
            getattr(inp, "terminal_window", torch.zeros_like(gate)),
            dtype=gate.dtype,
            device=gate.device,
        )
        # 高度和倾斜已经由 stable_motion_gate 连续表达。只有真实 terminal 或数值
        # 非法才硬关闭能力梯度，避免在终止阈值之前制造“还能恢复但没有任务奖励”的死区。
        hard_survival_gate = (
            torch.isfinite(base_h_for_survival)
            & torch.isfinite(tilt_for_survival)
            & (terminal_for_survival <= 0.5)
        ).to(gate.dtype)
    hard_survival_gate = torch.as_tensor(hard_survival_gate, dtype=gate.dtype, device=gate.device)
    if hard_survival_gate.ndim == 0:
        hard_survival_gate = hard_survival_gate.expand_as(gate)
    capability_floor = min(max(float(getattr(cfg, "capability_gate_floor", 0.25)), 0.0), 1.0)
    capability_gate = hard_survival_gate * torch.clamp(
        capability_floor + (1.0 - capability_floor) * gate,
        0.0,
        1.0,
    )
    comp["stable_motion_gate"] = gate
    comp["hard_survival_gate"] = hard_survival_gate
    comp["capability_gate"] = capability_gate
    comp["stand_gate"] = stand_gate
    comp["moving_gate"] = moving_gate
    comp["quality_gate"] = quality_gate
    comp["refinement_gate"] = refinement_gate

    # 地形响应来自上一帧已经发生的接触、绊脚或支撑面变化，不能由前方高度扫描提前触发。
    terrain_response = getattr(inp, "terrain_response", torch.zeros_like(gate))
    terrain_response = torch.clamp(
        torch.as_tensor(terrain_response, dtype=gate.dtype, device=gate.device),
        0.0,
        1.0,
    )
    local_obstacle_h = getattr(inp, "local_obstacle_h", torch.zeros_like(gate))
    local_obstacle_h = torch.as_tensor(local_obstacle_h, dtype=gate.dtype, device=gate.device)
    terrain_obstacle_h = local_obstacle_h * terrain_response
    terrain_pattern_scale = terrain_amp_style_scale(
        terrain_response,
        quality_gate,
        max_relief=float(getattr(cfg, "terrain_style_max_relief", 0.65)),
        relief_progress=getattr(inp, "terrain_style_relief_progress", None),
    )
    comp["terrain_response"] = terrain_response
    comp["terrain_pattern_scale"] = terrain_pattern_scale

    trajectory_error = getattr(inp, "foot_trajectory_error", None)
    trajectory_scope = getattr(inp, "foot_trajectory_scope", None)
    if trajectory_error is not None and trajectory_scope is not None:
        trajectory_error = trajectory_error.to(dtype=f, device=gate.device)
        trajectory_scope = trajectory_scope.to(dtype=f, device=gate.device)
        trajectory_scale = max(float(getattr(cfg, "foot_trajectory_error_scale", 0.35)), 1e-6)
        trajectory_shape = trajectory_error / (trajectory_error + trajectory_scale)
        trajectory_quality = trajectory_quality_score(
            trajectory_error,
            trajectory_scope,
            trajectory_scale,
            terrain_pattern_scale,
        )
    else:
        trajectory_shape = torch.zeros_like(gate)
        trajectory_scope = torch.zeros_like(gate)
        trajectory_quality = torch.ones_like(gate)

    cmd = inp.cmd
    vel = inp.base_lin_vel
    cmd_xy = cmd[:, :2]
    cmd_xy_norm = torch.linalg.norm(cmd_xy, dim=-1)
    base_h_above_terrain = getattr(inp, "base_h_above_terrain", torch.full_like(gate, cfg.nominal_base_h))
    tilt_rel = getattr(inp, "tilt_rel", torch.zeros_like(gate))
    foot_contact = getattr(inp, "foot_contact", None)
    if foot_contact is not None:
        foot_contact = foot_contact.to(dtype=f, device=gate.device)
        contact_count = foot_contact.sum(dim=-1)
    else:
        contact_count = torch.full_like(gate, 2.0)
    support_structure = support_structure_gate(foot_contact, cmd)
    if support_structure is None:
        support_structure = torch.ones_like(gate)
    height_score = _smoothstep(cfg.h_gate_close, cfg.h_ok, base_h_above_terrain)
    tilt_score = 1.0 - _smoothstep(cfg.tilt_ok_rad, cfg.tilt_gate_close_rad, tilt_rel)
    min_contacts = float(getattr(cfg, "support_integrity_min_contacts", 2.0))
    contact_score = torch.clamp((contact_count - (min_contacts - 1.0)) / 1.0, 0.0, 1.0)
    wxy_mag = torch.sqrt(inp.base_ang_vel[:, 0] ** 2 + inp.base_ang_vel[:, 1] ** 2)
    wxy_fault = _smoothstep(
        float(getattr(cfg, "support_integrity_wxy_soft", 0.9)),
        float(getattr(cfg, "support_integrity_wxy_hard", 2.4)),
        wxy_mag,
    )
    support_fault = torch.maximum(
        torch.maximum(1.0 - height_score, 1.0 - tilt_score),
        torch.maximum(torch.maximum(1.0 - contact_score, 1.0 - support_structure), wxy_fault),
    )
    support_quality = torch.clamp(1.0 - support_fault, 0.0, 1.0)
    comp["support_structure_gate"] = support_structure

    # 命令跟踪是主任务，不能再被接触质量、步态成熟度或地形响应乘掉。
    # 打滑、拖脚、碰撞和失稳由下方独立的加性代价约束；只有真实终止或数值
    # 非法才关闭任务收益。否则策略在最需要恢复驱动的台阶前反而会失去梯度。
    sigma_v = torch.maximum(torch.full_like(cmd_xy_norm, cfg.sigma_lin_abs), cfg.sigma_lin_rel * cmd_xy_norm)
    lin_err = torch.linalg.norm(vel[:, :2] - cmd_xy, dim=-1)
    tracking_lin_raw = exp_kernel(lin_err, sigma_v)
    yaw_err = (inp.base_ang_vel[:, 2] - cmd[:, 2]).abs()
    yaw_cmd_gate = (cmd[:, 2].abs() > 0.05).to(f)
    tracking_yaw_raw = exp_kernel(yaw_err, cfg.sigma_yaw)

    # 站立项：低速度、默认姿态和低动作幅度，由 stand_gate 控制。
    speed = torch.linalg.norm(vel[:, :2], dim=-1)
    yaw_rate = inp.base_ang_vel[:, 2].abs()
    action_norm = torch.linalg.norm(inp.action, dim=-1) / (inp.action.shape[-1] ** 0.5)
    stand = (exp_kernel(speed, cfg.sigma_stand_speed) * exp_kernel(yaw_rate, cfg.sigma_stand_yaw)
             * exp_kernel(inp.default_pose_error.abs(), getattr(cfg, "sigma_stand_pose", 0.20))
             * exp_kernel(action_norm, getattr(cfg, "sigma_stand_action", 0.30)))
    comp["stand"] = cfg.w_stand * stand * stand_gate * capability_gate

    # 乘法高斯适合精确站稳后的收敛，但任一分量较差时会整体接近零。独立的姿态
    # 代价继续提供默认姿态和标称高度的恢复梯度，并用最差关节尾部防止单腿取巧。
    default_pose_tail = getattr(inp, "default_pose_error_tail", inp.default_pose_error)
    default_pose_tail = torch.as_tensor(default_pose_tail, dtype=f, device=gate.device)
    pose_metric = 0.5 * inp.default_pose_error.abs() + 0.5 * default_pose_tail.abs()
    stand_pose_cost = huber_excess(
        pose_metric,
        float(getattr(cfg, "stand_pose_free", 0.03)),
        float(getattr(cfg, "stand_pose_scale", 0.20)),
    )
    stand_height_cost = huber_excess(
        (base_h_above_terrain - float(cfg.nominal_base_h)).abs(),
        float(getattr(cfg, "stand_height_band", 0.015)),
        float(getattr(cfg, "stand_height_scale", 0.04)),
    )
    stand_posture_scope = torch.as_tensor(
        getattr(inp, "stand_posture_scope", torch.ones_like(gate)),
        dtype=f,
        device=gate.device,
    )
    comp["stand_posture"] = (
        -float(getattr(cfg, "w_stand_posture", 0.0))
        * (0.55 * stand_pose_cost + 0.45 * stand_height_cost)
        * stand_gate
        * stand_posture_scope
        * terrain_pattern_scale
        * hard_survival_gate
    )

    # 四足站稳：零命令下奖励四足接触，防止“机身静止但脚下碎步”拿到高 stand 分。
    # 该项只在 stand_gate 下生效，不影响行走。
    if foot_contact is not None:
        contact_mean = foot_contact.mean(dim=-1)
        contact_worst = foot_contact.min(dim=-1).values
        # 单足长期悬空时四足均值仍有 0.75，足以被其他站立奖励掩盖。最差足占
        # 主要权重，均值只保留接触建立过程中的连续梯度。
        four_foot = 0.25 * contact_mean + 0.75 * contact_worst
        comp["stand_contact"] = (
            getattr(cfg, "w_stand_contact", 0.0) * four_foot * stand_gate * capability_gate
        )
    else:
        comp["stand_contact"] = torch.zeros_like(gate)
    support_scope = torch.clamp(moving_gate + 0.35 * stand_gate, 0.0, 1.0)
    # support_integrity 只负责“有没有可靠支撑”。高度、倾斜和 wxy 已分别由
    # flat_move_height、orient 和 base_wxy 负责，不能在这里再次叠加。
    support_contact_fault = torch.maximum(1.0 - contact_score, 1.0 - support_structure)
    comp["support_integrity"] = (
        -getattr(cfg, "w_support_integrity", 0.0)
        * (support_contact_fault ** 2)
        * support_scope
    )

    # 远场 shaping：在大误差区提供梯度，把策略拉向命令目标。
    # 它使用与高斯跟踪相同的门控，collapse 时关闭，不引入新的退化最优。
    s_far = getattr(cfg, "scale_track_far", 0.5)
    s_stand_far = getattr(cfg, "scale_stand_far", 0.3)
    tracking_lin_far_raw = far_kernel(lin_err, s_far)
    s_yaw_far = getattr(cfg, "scale_yaw_far", s_far)
    tracking_yaw_far_raw = far_kernel(yaw_err, s_yaw_far)
    comp["stand_far"] = (
        getattr(cfg, "w_stand_far", 0.6)
        * far_kernel(speed, s_stand_far)
        * far_kernel(yaw_rate, s_stand_far)
        * stand_gate
        * capability_gate
    )

    # 地形稳定进展：只按线速度命令方向计算。
    # 原始进展 = dot(v_xy, cmd_xy) / |cmd_xy|；纯 yaw 不拥有线性地形进展。
    # 这样避免横移/yaw 命令下把前向漂移误判为进展。
    cmd_xy_norm_safe = torch.clamp(cmd_xy_norm, min=1e-6)
    v_along = torch.sum(vel[:, :2] * cmd_xy, dim=-1) / cmd_xy_norm_safe
    has_linear_cmd = cmd_xy_norm > 0.05
    ratio = torch.clamp(v_along / cmd_xy_norm_safe, 0.0, 1.0)
    style_min_ratio = max(float(getattr(cfg, "gait_anchor_min_ratio", 0.45)), 1e-6)
    style_progress = torch.clamp(v_along / (cmd_xy_norm_safe * style_min_ratio), 0.0, 1.0)
    style_floor = float(getattr(cfg, "gait_anchor_progress_floor", 0.55))
    progress_style_gate = torch.where(
        has_linear_cmd,
        torch.clamp(style_floor + (1.0 - style_floor) * style_progress, 0.0, 1.0),
        torch.ones_like(gate),
    )
    clearance_contact = getattr(inp, "clearance_support_contact", inp.foot_contact)
    clearance_contact = clearance_contact.to(dtype=f, device=gate.device)
    swing_for_progress = (clearance_contact <= 0.5).to(f)
    clearance_target = terrain_clearance_targets(
        inp.foot_clearance,
        terrain_obstacle_h,
        flat_target=cfg.flat_clearance_target,
        margin=cfg.terrain_clearance_margin,
        margin_min=getattr(cfg, "terrain_clearance_margin_min", None),
        margin_gain=getattr(cfg, "terrain_clearance_margin_gain", None),
        margin_max=getattr(cfg, "terrain_clearance_margin_max", None),
        lead_foot_mask=getattr(
            inp,
            "terrain_clearance_foot_mask",
            getattr(inp, "terrain_lead_foot_mask", None),
        ),
        event_direction=getattr(inp, "terrain_event_direction", None),
        event_ramp=getattr(inp, "terrain_clearance_ramp", None),
        event_probe_target=getattr(inp, "terrain_probe_target", None),
    )
    event_clearance_mask = getattr(inp, "terrain_clearance_foot_mask", None)
    reactive_foot_strength = torch.zeros_like(inp.foot_clearance)
    if event_clearance_mask is not None:
        event_clearance_mask = torch.clamp(
            event_clearance_mask.to(dtype=f, device=gate.device), 0.0, 1.0
        )
        event_ramp = torch.clamp(
            torch.as_tensor(
                getattr(inp, "terrain_clearance_ramp", torch.zeros_like(gate)),
                dtype=f,
                device=gate.device,
            ),
            0.0,
            1.0,
        )
        event_direction = torch.as_tensor(
            getattr(inp, "terrain_event_direction", torch.zeros_like(gate)),
            dtype=f,
            device=gate.device,
        )
        reactive_foot_strength = (
            event_clearance_mask
            * event_ramp[:, None]
            * (event_direction > 0.0).to(f)[:, None]
        )
    # 普通步态保留 clearance 容差；碰障脚必须满足完整物理目标。否则 9 cm
    # 目标会被 2 cm 容差降成 7 cm，使仍顶着台阶的 5-6 cm 足端获得过高成功率。
    effective_clearance_band = float(cfg.clearance_band) * (1.0 - reactive_foot_strength)
    under_for_progress = torch.clamp(
        (clearance_target - inp.foot_clearance - effective_clearance_band) / 0.05,
        min=0.0,
    )
    clearance_success_by_foot = torch.clamp(1.0 - under_for_progress, 0.0, 1.0)
    placement_success = getattr(inp, "terrain_placement_success_by_foot", None)
    if placement_success is not None:
        # 足端已经在更高承重层落稳时，抬脚任务已经完成；不能继续要求该脚悬空。
        placement_success = torch.clamp(
            placement_success.to(dtype=f, device=gate.device),
            0.0,
            1.0,
        )
        clearance_success_by_foot = torch.maximum(
            clearance_success_by_foot,
            placement_success,
        )
    swing_clearance_success = (
        clearance_success_by_foot * swing_for_progress
    ).sum(dim=-1) / swing_for_progress.sum(dim=-1).clamp(min=1.0)
    # 碰障腿可能仍处于支撑态；若继续只统计摆动足，就会在真正撞台阶的腿尚未抬开时
    # 错发 terrain_progress。真实上行响应存在时改为评价响应足自身，平地和下行保持原语义。
    if event_clearance_mask is not None:
        event_weight = reactive_foot_strength.sum(dim=-1)
        event_clearance_success = (
            clearance_success_by_foot * reactive_foot_strength
        ).sum(dim=-1) / event_weight.clamp(min=1e-6)
        clearance_success = torch.where(
            event_weight > 1e-6,
            event_clearance_success,
            swing_clearance_success,
        )
    else:
        clearance_success = swing_clearance_success
    comp["terrain_clearance_success"] = clearance_success
    terrain_clearance_response = torch.clamp(
        torch.as_tensor(
            getattr(inp, "terrain_clearance_response", terrain_response),
            dtype=f,
            device=gate.device,
        ),
        0.0,
        1.0,
    )
    terrain_progress_clearance_gate = torch.clamp(
        1.0
        - terrain_clearance_response
        + terrain_clearance_response * clearance_success,
        0.0,
        1.0,
    )
    terrain_difficulty = torch.maximum(terrain_response, terrain_clearance_response)

    diag_window = getattr(inp, "diagonal_pair_window", None)
    duty_window = getattr(inp, "duty_quality_window", None)
    slip_high_window = getattr(inp, "stance_slip_high_fraction", None)
    diag_score = diag_window.to(dtype=f, device=gate.device) if diag_window is not None else torch.full_like(gate, 0.70)
    duty_score = duty_window.to(dtype=f, device=gate.device) if duty_window is not None else torch.full_like(gate, 0.70)
    slip_tail = slip_high_window.to(dtype=f, device=gate.device) if slip_high_window is not None else torch.zeros_like(gate)
    slip_gate = _metric_gate(
        slip_tail,
        float(getattr(cfg, "healthy_progress_slip_target", 0.12)),
        higher=False,
        width=float(getattr(cfg, "healthy_progress_slip_width", 0.18)),
    )
    touchdown_clean_gate = torch.ones_like(gate)
    td_gate_mask = getattr(inp, "touchdown_mask", None)
    td_gate_vz = getattr(inp, "touchdown_vz", None)
    if td_gate_mask is not None and td_gate_vz is not None:
        td_gate_mask = td_gate_mask.to(dtype=f, device=gate.device)
        td_gate_vz = td_gate_vz.to(dtype=f, device=gate.device)
        td_event = td_gate_mask.sum(dim=-1) > 0.5
        impact_scale = max(float(getattr(cfg, "impact_speed_scale", 1.0)), 1e-6)
        td_bad_by_foot = torch.clamp(
            (
                td_gate_vz.abs()
                - float(getattr(cfg, "touchdown_vz_free", 0.12))
            ) / impact_scale,
            0.0,
            1.0,
        )
        td_gate_xy = getattr(inp, "touchdown_xy_speed", None)
        if td_gate_xy is not None:
            td_gate_xy = td_gate_xy.to(dtype=f, device=gate.device)
            td_xy_bad = torch.clamp((td_gate_xy - 0.05) / impact_scale, 0.0, 1.0)
            td_bad_by_foot = torch.maximum(td_bad_by_foot, td_xy_bad)
        td_gate_force = getattr(inp, "touchdown_force_rate", None)
        if td_gate_force is not None:
            td_gate_force = td_gate_force.to(dtype=f, device=gate.device)
            force_scale = max(float(getattr(cfg, "touchdown_force_rate_scale", 0.45)), 1e-6)
            td_force_bad = torch.clamp(td_gate_force / force_scale, 0.0, 1.0)
            td_bad_by_foot = torch.maximum(td_bad_by_foot, td_force_bad)
        td_bad = (td_bad_by_foot * td_gate_mask).max(dim=-1).values
        # 平地完整执行轻脚约束；真实碰障后沿既有阶段退让，避免压制楼梯支撑和换层发力。
        td_bad = td_bad * terrain_pattern_scale
        td_floor = float(getattr(cfg, "tracking_touchdown_clean_floor", 0.65))
        td_clean = torch.clamp(td_floor + (1.0 - td_floor) * (1.0 - td_bad), 0.0, 1.0)
        touchdown_clean_gate = torch.where(td_event, td_clean, touchdown_clean_gate)
    abs_wz_cmd = cmd[:, 2].abs()
    has_yaw_cmd = abs_wz_cmd > 0.05
    pure_yaw = (abs_wz_cmd > 0.05) & (cmd_xy_norm <= 0.05)
    yaw_mix = torch.clamp(abs_wz_cmd / (abs_wz_cmd + cmd_xy_norm + 1e-6), 0.0, 1.0)
    rhythm_gate = 0.5 * (
        _metric_gate(diag_score, 0.52, higher=True, width=0.18)
        + _metric_gate(duty_score, 0.58, higher=True, width=0.18)
    )
    tracking_motion_gate = tracking_motion_quality(inp.base_ang_vel, vel, cfg)
    tracking_motion_raw = tracking_motion_quality(inp.base_ang_vel, vel, cfg, floor=0.0)
    wxy_mag_for_yaw = torch.sqrt(inp.base_ang_vel[:, 0] ** 2 + inp.base_ang_vel[:, 1] ** 2)
    yaw_drift_gate = _metric_gate(
        speed,
        float(getattr(cfg, "yaw_tracking_drift_target", 0.10)),
        higher=False,
        width=float(getattr(cfg, "yaw_tracking_drift_width", 0.25)),
    )
    yaw_wxy_gate = _metric_gate(
        wxy_mag_for_yaw,
        float(getattr(cfg, "yaw_tracking_wxy_target", 0.35)),
        higher=False,
        width=float(getattr(cfg, "yaw_tracking_wxy_width", 0.65)),
    )
    support_floor = float(getattr(cfg, "yaw_tracking_support_floor", 0.35))
    yaw_contact_target = float(getattr(cfg, "yaw_support_contact_target", 2.25))
    yaw_contact_width = max(float(getattr(cfg, "yaw_support_contact_width", 0.75)), 1e-6)
    # 两足交替支撑附近质量最高；三足仍保留可用梯度，四足拖动不能再获得满额
    # yaw 跟踪收益。
    yaw_contact_quality = torch.exp(
        -0.5 * ((contact_count - yaw_contact_target) / yaw_contact_width).square()
    )
    support_present = torch.clamp(contact_count - 1.0, 0.0, 1.0)
    support_gate_raw = yaw_contact_quality * support_present
    support_gate = torch.clamp(
        support_floor + (1.0 - support_floor) * support_gate_raw,
        0.0,
        1.0,
    )
    yaw_duty_floor = float(getattr(cfg, "yaw_duty_quality_floor", 0.45))
    yaw_duty_gate = torch.clamp(
        yaw_duty_floor + (1.0 - yaw_duty_floor) * support_gate,
        0.0,
        1.0,
    )
    yaw_quality_gate = (
        slip_gate
        * touchdown_clean_gate
        * support_gate
        * support_quality
        * yaw_duty_gate
        * tracking_motion_gate
        * 0.5
        * (yaw_drift_gate + yaw_wxy_gate)
    )
    # 足端轨迹误差由独立加性惩罚纠偏，不能再乘掉 yaw 的任务收益。
    # 否则早期尚未成型的轨迹会同时失去 tracking 驱动，形成自锁。
    terrain_mix = torch.clamp(
        terrain_difficulty * float(getattr(cfg, "tracking_validity_terrain_mix", 1.0)),
        0.0,
        1.0,
    )
    valid_contact = slip_gate * touchdown_clean_gate * support_quality * tracking_motion_gate * (
        (1.0 - terrain_mix) * rhythm_gate + terrain_mix * clearance_success
    )
    yaw_valid_contact = torch.where(
        has_yaw_cmd,
        (1.0 - yaw_mix) * valid_contact + yaw_mix * yaw_quality_gate,
        valid_contact,
    )
    healthy_floor = float(getattr(cfg, "healthy_progress_floor", 0.35))
    healthy_progress_gate = torch.clamp(healthy_floor + (1.0 - healthy_floor) * valid_contact, 0.0, 1.0)
    tracking_floor = float(getattr(cfg, "validated_tracking_floor", healthy_floor))
    validated_tracking_gate = torch.clamp(tracking_floor + (1.0 - tracking_floor) * valid_contact, 0.0, 1.0)
    yaw_base = float(getattr(cfg, "yaw_tracking_base_fraction", 0.60))
    validated_yaw_tracking_gate = torch.clamp(yaw_base + (1.0 - yaw_base) * yaw_valid_contact, 0.0, 1.0)
    # clearance、支撑和稳定性均由独立加性代价负责，不能乘掉碰障后的方向驱动。
    # 保留 terrain_clearance_drive 配置键仅用于旧 payload 兼容，不再进入任务奖励。
    comp["healthy_progress_gate"] = healthy_progress_gate
    comp["validated_tracking_gate"] = validated_tracking_gate
    comp["validated_yaw_tracking_gate"] = validated_yaw_tracking_gate
    comp["yaw_quality_gate"] = yaw_quality_gate
    comp["yaw_drift_gate"] = yaw_drift_gate
    comp["yaw_wxy_gate"] = yaw_wxy_gate
    comp["yaw_duty_gate"] = yaw_duty_gate
    comp["tracking_motion_gate"] = tracking_motion_gate
    comp["touchdown_clean_gate"] = touchdown_clean_gate
    comp["progress_style_gate"] = progress_style_gate
    # 纯 yaw 不能靠“平移速度为零”领取线速度任务正奖励；平移纯净度由独立惩罚约束。
    linear_task_gate = has_linear_cmd.to(f)
    # 地形接触后只放松速度“大小”的精确误差；方向推进在下方保持完整强度。
    # 该响应来自已经发生的接触，碰障前仍按平地命令严格跟踪。
    terrain_tracking_min = min(max(
        float(getattr(cfg, "terrain_tracking_min_scale", 0.30)), 0.0
    ), 1.0)
    terrain_tracking_scale = 1.0 - terrain_difficulty * (1.0 - terrain_tracking_min)
    # 以下两个门只做诊断，不能乘入基础任务驱动。姿态或命令年龄越差，Actor 越
    # 需要明确的目标方向；若此时削弱 tracking，恢复动作反而没有正向梯度。
    posture_floor = min(max(float(getattr(cfg, "tracking_posture_floor", 0.55)), 0.0), 1.0)
    tracking_posture_gate = torch.clamp(
        posture_floor + (1.0 - posture_floor) * torch.minimum(gate, tracking_motion_raw),
        0.0,
        1.0,
    )
    tracking_posture_gate = torch.lerp(
        tracking_posture_gate,
        torch.ones_like(tracking_posture_gate),
        terrain_difficulty,
    )
    task_credit_age_s = torch.as_tensor(
        getattr(
            inp,
            "task_credit_age_s",
            torch.full_like(gate, float(getattr(cfg, "task_credit_full_s", 3.0))),
        ),
        dtype=f,
        device=gate.device,
    )
    if task_credit_age_s.ndim == 0:
        task_credit_age_s = task_credit_age_s.expand_as(gate)
    flat_task_credit_gate = sustained_task_credit_gate(
        task_credit_age_s,
        floor=float(getattr(cfg, "task_credit_floor", 0.20)),
        start_s=float(getattr(cfg, "task_credit_start_s", 1.50)),
        full_s=float(getattr(cfg, "task_credit_full_s", 3.00)),
    )
    task_credit_gate = torch.lerp(
        flat_task_credit_gate,
        torch.ones_like(flat_task_credit_gate),
        terrain_difficulty,
    )
    effective_tracking_gate = hard_survival_gate
    comp["terrain_tracking_scale"] = terrain_tracking_scale
    comp["tracking_posture_gate"] = tracking_posture_gate
    comp["task_credit_gate"] = task_credit_gate
    comp["effective_tracking_gate"] = effective_tracking_gate
    comp["tracking_lin"] = (
        cfg.w_tracking_lin * tracking_lin_raw * linear_task_gate * moving_gate
        * hard_survival_gate * terrain_tracking_scale
    )
    comp["tracking_yaw"] = (
        cfg.w_tracking_yaw * tracking_yaw_raw * moving_gate
        * yaw_cmd_gate * hard_survival_gate
    )
    comp["tracking_lin_far"] = (
        getattr(cfg, "w_track_far", 0.6) * tracking_lin_far_raw * linear_task_gate
        * moving_gate * hard_survival_gate * terrain_tracking_scale
    )
    comp["tracking_yaw_far"] = (
        getattr(cfg, "w_yaw_far", 0.3) * tracking_yaw_far_raw * moving_gate
        * hard_survival_gate * yaw_cmd_gate
    )
    yaw_cmd_sq = torch.clamp(cmd[:, 2] * cmd[:, 2], min=1e-6)
    yaw_progress_ratio = torch.clamp(inp.base_ang_vel[:, 2] * cmd[:, 2] / yaw_cmd_sq, 0.0, 1.0)
    signed_yaw_rate = inp.base_ang_vel[:, 2] * torch.sign(cmd[:, 2])
    yaw_progress_overspeed_gate = bounded_overspeed_quality(
        signed_yaw_rate,
        abs_wz_cmd,
        float(getattr(cfg, "scale_yaw_far", 0.60)),
    )
    yaw_progress_floor = float(getattr(cfg, "yaw_progress_quality_floor", 0.35))
    yaw_progress_quality = torch.clamp(
        yaw_progress_floor + (1.0 - yaw_progress_floor) * yaw_quality_gate,
        0.0,
        1.0,
    )
    comp["yaw_progress_quality_gate"] = yaw_progress_quality
    comp["yaw_progress_overspeed_gate"] = yaw_progress_overspeed_gate
    comp["yaw_progress"] = (
        getattr(cfg, "w_yaw_progress", 0.0)
        * yaw_progress_ratio
        * yaw_progress_overspeed_gate
        * moving_gate
        * yaw_cmd_gate
        * hard_survival_gate
    )
    yaw_signed_speed = inp.base_ang_vel[:, 2] * torch.sign(cmd[:, 2])
    yaw_underspeed = normalized_underspeed_cost(
        yaw_signed_speed,
        abs_wz_cmd,
        float(getattr(cfg, "yaw_underspeed_min_ratio", 0.75)),
    )
    comp["yaw_underspeed"] = (
        -float(getattr(cfg, "w_yaw_underspeed", 0.0))
        * yaw_underspeed
        * yaw_cmd_gate
        * moving_gate
        * hard_survival_gate
        * terrain_tracking_scale
    )
    yaw_moment = getattr(inp, "yaw_support_moment_norm", None)
    if yaw_moment is not None:
        yaw_moment = yaw_moment.to(dtype=f, device=gate.device)
        signed_moment = yaw_moment * torch.sign(cmd[:, 2])
        comp["yaw_support_moment"] = (
            getattr(cfg, "w_yaw_support_moment", 0.0)
            * torch.clamp(signed_moment, 0.0, 1.0)
            * yaw_cmd_gate
            * moving_gate
            * capability_gate
            * task_credit_gate
        )
        comp["yaw_wrong_moment"] = (
            -getattr(cfg, "w_yaw_wrong_moment", 0.0)
            * torch.clamp(-signed_moment, 0.0, 1.0).pow(2)
            * yaw_cmd_gate
            * moving_gate
            * capability_gate
        )
    else:
        comp["yaw_support_moment"] = torch.zeros_like(gate)
        comp["yaw_wrong_moment"] = torch.zeros_like(gate)
    # 一般方向推进从零速度开始连续给梯度，并保留固定支撑 floor。这样失去支撑
    # 不会获得满额收益，但早期探索和碰障恢复也不会被质量门乘成零。
    direction_full_ratio = max(float(getattr(
        cfg,
        "direction_progress_full_ratio",
        getattr(cfg, "terrain_progress_full_ratio", 0.60),
    )), 1e-6)
    direction_progress_ratio = torch.clamp(ratio / direction_full_ratio, 0.0, 1.0)
    terrain_full_ratio = max(float(getattr(
        cfg,
        "terrain_progress_full_ratio",
        direction_full_ratio,
    )), 1e-6)
    terrain_direction_progress_ratio = torch.clamp(ratio / terrain_full_ratio, 0.0, 1.0)
    direction_alignment = torch.clamp(
        v_along / torch.linalg.norm(vel[:, :2], dim=-1).clamp(min=1e-6),
        0.0,
        1.0,
    )
    comp["direction_progress_ratio"] = direction_progress_ratio
    comp["terrain_direction_progress_ratio"] = terrain_direction_progress_ratio
    comp["direction_alignment"] = direction_alignment

    # 核心稳定不再只依赖一组零散负代价。这个正奖励把“沿命令产生进展”和
    # “机身保持为稳定平台”放在同一条加性路径中，但不乘回 tracking 主驱动。
    # 直线运动约束 wx/wy 及非命令 wz；包含 yaw 命令时只把 wx/wy 视为无关转动。
    linear_active = (cmd_xy_norm > 0.05).to(f)
    yaw_active = (abs_wz_cmd > 0.05).to(f)
    active_modes = (linear_active + yaw_active).clamp(min=1.0)
    core_progress_ratio = (
        direction_progress_ratio * linear_active
        + yaw_progress_ratio * yaw_active
    ) / active_modes
    unwanted_wz = inp.base_ang_vel[:, 2] * (1.0 - yaw_active)
    unwanted_ang_vel = torch.sqrt(
        inp.base_ang_vel[:, 0].square()
        + inp.base_ang_vel[:, 1].square()
        + unwanted_wz.square()
    )
    core_height_target = float(getattr(
        cfg, "flat_move_height_target", getattr(cfg, "nominal_base_h", geometry.NOMINAL_BASE_HEIGHT)
    ))
    core_quality = (
        0.30 * exp_kernel(tilt_rel.abs(), float(getattr(cfg, "core_tilt_scale", 0.035)))
        + 0.25 * exp_kernel(unwanted_ang_vel, float(getattr(cfg, "core_ang_vel_scale", 0.20)))
        + 0.25 * exp_kernel(
            torch.abs(base_h_above_terrain - core_height_target),
            float(getattr(cfg, "core_height_scale", 0.020)),
        )
        + 0.20 * exp_kernel(
            vel[:, 2].abs(),
            float(getattr(cfg, "core_vz_scale", 0.12)),
        )
    )
    core_progress_floor = min(max(float(getattr(cfg, "core_progress_floor", 0.20)), 0.0), 1.0)
    core_terrain_relief = min(max(float(getattr(cfg, "core_terrain_relief", 0.70)), 0.0), 0.95)
    # 静止和移动共享同一条核心稳定信用；stand 只是不再把它截断掉。
    core_scope = torch.clamp(moving_gate + stand_gate, 0.0, 1.0)
    core_progress_floor_t = torch.where(
        stand_gate > 0.5,
        torch.ones_like(gate),
        torch.full_like(gate, core_progress_floor),
    )
    comp["core_quality"] = (
        float(getattr(cfg, "w_core_quality", 0.0))
        * core_scope
        * (core_progress_floor_t + (1.0 - core_progress_floor_t) * core_progress_ratio)
        * core_quality
        * (1.0 - core_terrain_relief * terrain_response)
        * hard_survival_gate
    )
    direction_support_floor = min(max(
        float(getattr(cfg, "direction_progress_support_floor", 0.0)), 0.0
    ), 1.0)
    direction_support_quality = torch.clamp(
        direction_support_floor + (1.0 - direction_support_floor) * support_quality,
        0.0,
        1.0,
    )
    # 平地上的方向信用同时要求机身角速度、竖直运动和真实触地处于可控范围。
    # 探索梯度由 tracking/far/欠速项负责；progress 本身不再给失稳速度保底信用。
    # 被动碰障后连续退回地形方向驱动，楼梯主驱动不等待平地质量先达标。
    direction_yaw_quality = _metric_gate(
        (inp.base_ang_vel[:, 2] - cmd[:, 2]).abs(),
        float(getattr(cfg, "cycle_yaw_rms_free", 0.08)),
        higher=False,
        width=float(getattr(cfg, "heading_hold_scale", 0.35)),
    )
    flat_motion_quality = torch.clamp(
        tracking_motion_raw * direction_yaw_quality * touchdown_clean_gate,
        0.0,
        1.0,
    ).pow(1.0 / 3.0)
    direction_quality_floor = min(max(
        float(getattr(cfg, "direction_progress_quality_floor", 0.0)), 0.0
    ), 1.0)
    flat_progress_quality = torch.clamp(
        direction_quality_floor
        + (1.0 - direction_quality_floor) * flat_motion_quality,
        0.0,
        1.0,
    )
    terrain_quality_relief = torch.clamp(terrain_difficulty, 0.0, 1.0)
    direction_progress_quality = torch.lerp(
        flat_progress_quality,
        torch.ones_like(flat_progress_quality),
        terrain_quality_relief,
    )
    comp["direction_progress_quality_gate"] = direction_progress_quality
    # 地形方向信用只要求：非平地真实响应已经发生、沿命令方向产生进展、尚未真实
    # 终止。它不等待 clearance、换层、支撑质量或步态模板先达标；这些均是加性约束。
    # 碰障后速度可能瞬时降到零。若把 alignment² 直接乘到推进信用上，零速点的
    # 方向驱动梯度也会归零。保留非零方向下限负责脱困，离轴速度仍由 off_axis、
    # planar_purity 和 alignment 的剩余部分约束。
    terrain_alignment_floor = min(max(float(getattr(
        cfg,
        "terrain_direction_alignment_floor",
        0.35,
    )), 0.0), 1.0)
    terrain_alignment_credit = torch.clamp(
        terrain_alignment_floor
        + (1.0 - terrain_alignment_floor) * direction_alignment.square(),
        0.0,
        1.0,
    )
    comp["terrain_direction_alignment_credit"] = terrain_alignment_credit
    # 一个方向推进项同时覆盖平地和已发生接触后的地形响应。地形只放宽速度大小，
    # 不增加第二份任务奖励；碰障静止点用非零对齐下限保留脱困梯度。
    unified_progress_ratio = torch.lerp(
        direction_progress_ratio,
        terrain_direction_progress_ratio,
        terrain_difficulty,
    )
    unified_alignment_credit = torch.lerp(
        direction_alignment.square(),
        terrain_alignment_credit,
        terrain_difficulty,
    )
    comp["supported_progress"] = (
        getattr(cfg, "w_supported_progress", 0.0)
        * unified_progress_ratio
        * unified_alignment_credit
        * bounded_overspeed_quality(v_along, cmd_xy_norm, sigma_v)
        * has_linear_cmd.to(f)
        * moving_gate
        * hard_survival_gate
    )
    comp["terrain_progress"] = (
        cfg.w_terrain_progress
        * terrain_direction_progress_ratio
        * terrain_alignment_credit
        * bounded_overspeed_quality(v_along, cmd_xy_norm, sigma_v)
        * has_linear_cmd.to(f)
        * moving_gate
        * hard_survival_gate
        * terrain_difficulty
    )
    underspeed_w = float(getattr(cfg, "w_linear_underspeed", 0.0))
    if underspeed_w != 0.0:
        min_ratio = max(float(getattr(cfg, "linear_underspeed_min_ratio", 0.65)), 0.0)
        underspeed = normalized_underspeed_cost(v_along, cmd_xy_norm, min_ratio)
        comp["linear_underspeed"] = (
            -underspeed_w
            * underspeed
            * has_linear_cmd.to(f)
            * moving_gate
            * hard_survival_gate
            * terrain_tracking_scale
        )
    else:
        comp["linear_underspeed"] = torch.zeros_like(gate)

    # 反向运动压力：主动惩罚与线速度命令方向或 yaw 符号相反的运动。
    # 只扣掉跟踪奖励不够，因为反向吸引子仍可能没有直接成本。
    w_wd = getattr(cfg, "w_wrong_dir", 0.0)
    lin_wrong = torch.clamp(-v_along, min=0.0) * has_linear_cmd.to(f)
    has_yaw_cmd = cmd[:, 2].abs() > 0.05
    yaw_wrong = torch.clamp(
        -(inp.base_ang_vel[:, 2] * torch.sign(cmd[:, 2])) / cmd[:, 2].abs().clamp(min=0.05),
        min=0.0,
    ) * has_yaw_cmd.to(f)
    lin_wrong_cost = 2.0 * huber_excess(lin_wrong, 0.0, 1.0)
    yaw_wrong_cost = 2.0 * huber_excess(yaw_wrong, 0.0, 1.0)
    comp["wrong_dir"] = (
        -w_wd * (lin_wrong_cost + yaw_wrong_cost) * moving_gate * hard_survival_gate
    )

    # 非命令轴残差：只惩罚接近零命令的轴。
    # 该项不受 moving_gate 控制；站立时所有轴都接近零命令，因此它也是静止抗漂移压力。
    eps = 0.05
    off = torch.zeros_like(speed)
    off = off + torch.where(cmd[:, 0].abs() <= eps, vel[:, 0].abs(), torch.zeros_like(off))
    off = off + torch.where(cmd[:, 1].abs() <= eps, vel[:, 1].abs(), torch.zeros_like(off))
    off = off + torch.where(cmd[:, 2].abs() <= eps, inp.base_ang_vel[:, 2].abs(), torch.zeros_like(off))
    off_axis_cost = 2.0 * huber_excess(
        off,
        float(getattr(cfg, "off_axis_free", 0.03)),
        float(getattr(cfg, "off_axis_scale", 0.25)),
    )
    comp["off_axis"] = -cfg.w_off_axis * off_axis_cost * hard_survival_gate

    pure_x = (cmd[:, 0].abs() > eps) & (cmd[:, 1].abs() <= eps) & (cmd[:, 2].abs() <= eps)
    pure_y = (cmd[:, 1].abs() > eps) & (cmd[:, 0].abs() <= eps) & (cmd[:, 2].abs() <= eps)
    pure_yaw = (cmd[:, 2].abs() > eps) & (cmd_xy_norm <= eps)
    pure_yaw_f = pure_yaw.to(f)
    linear_gait_gate = has_linear_cmd.to(f)
    diag_cmd_gate = torch.clamp(linear_gait_gate + pure_yaw_f * float(getattr(cfg, "diagonal_contact_yaw_scale", 0.15)), 0.0, 1.0)
    duty_cmd_gate = torch.clamp(linear_gait_gate + pure_yaw_f * float(getattr(cfg, "duty_balance_yaw_scale", 0.50)), 0.0, 1.0)
    yaw_scale = float(getattr(cfg, "planar_purity_yaw_scale", 0.40))
    purity_err = (
        pure_x.to(f) * (vel[:, 1] ** 2 + yaw_scale * inp.base_ang_vel[:, 2] ** 2)
        + pure_y.to(f) * (vel[:, 0] ** 2 + yaw_scale * inp.base_ang_vel[:, 2] ** 2)
        + pure_yaw.to(f) * (vel[:, 0] ** 2 + vel[:, 1] ** 2)
    )
    comp["planar_purity"] = (
        -getattr(cfg, "w_planar_purity", 0.0) * purity_err * moving_gate * capability_gate
    )

    heading_error = getattr(inp, "heading_error", torch.zeros_like(gate))
    heading_error = torch.as_tensor(heading_error, dtype=f, device=gate.device)
    no_yaw_linear = ((cmd[:, 2].abs() <= eps) & (cmd_xy_norm > eps)).to(f)
    heading_scale = max(float(getattr(cfg, "heading_hold_scale", 0.35)), 1e-6)
    heading_angle_cost = 2.0 * huber_excess(
        heading_error.abs(), 0.0, heading_scale
    )
    comp["heading_hold"] = (
        -getattr(cfg, "w_heading_hold", 0.0)
        * heading_angle_cost * no_yaw_linear * moving_gate * capability_gate
    )
    yaw_translation_cost = 2.0 * huber_excess(
        speed,
        float(getattr(cfg, "yaw_translation_free", 0.03)),
        float(getattr(cfg, "yaw_translation_scale", 0.25)),
    )
    comp["yaw_translation"] = (
        -float(getattr(cfg, "w_yaw_translation", 0.0))
        * yaw_translation_cost
        * pure_yaw.to(f)
        * moving_gate
        * terrain_pattern_scale
        * capability_gate
    )

    # 抬脚高度只在摆动中段评价，并混入最差腿，避免一条低抬脚腿被其余腿稀释。
    swing = (clearance_contact <= 0.5).to(f)
    apex_weight = getattr(inp, "swing_apex_weight", swing)
    apex_weight = apex_weight.to(dtype=f, device=gate.device) * swing
    # 普通平地只在摆动中点检查 clearance。上楼碰障是被动感知事件，若仍要求
    # 事件恰好与预设摆动中点重合，碰撞足会得到零梯度。当前响应足因此按连续
    # 碰撞强度临时扩大评价权重；响应消失后立即恢复，不引入锁存或阶段状态。
    event_foot = getattr(
        inp,
        "terrain_clearance_foot_mask",
        getattr(inp, "terrain_lead_foot_mask", None),
    )
    event_direction = getattr(inp, "terrain_event_direction", None)
    event_ramp = getattr(inp, "terrain_clearance_ramp", None)
    if event_foot is not None and event_direction is not None and event_ramp is not None:
        event_foot = event_foot.to(dtype=f, device=gate.device)
        event_direction = torch.as_tensor(
            event_direction, dtype=f, device=gate.device
        ).reshape(-1, 1)
        event_ramp = torch.clamp(
            torch.as_tensor(event_ramp, dtype=f, device=gate.device),
            0.0,
            1.0,
        ).reshape(-1, 1)
        event_weight = (
            event_foot
            * (event_direction > 0).to(f)
            * event_ramp
            * swing
        )
        apex_weight = torch.maximum(apex_weight, event_weight)
        collision_foot = getattr(inp, "terrain_collision_foot_mask", None)
        if collision_foot is not None:
            # 碰撞腿即使仍在支撑也必须获得离地梯度；普通高差承重腿仍受上面的
            # swing 条件保护，避免把已经踩稳上层的足错误地再次抬起。
            collision_weight = (
                collision_foot.to(dtype=f, device=gate.device)
                * (event_direction > 0).to(f)
                * event_ramp
            )
            apex_weight = torch.maximum(apex_weight, collision_weight)
    target = clearance_target
    under = torch.clamp(
        (target - inp.foot_clearance - effective_clearance_band) / 0.05,
        min=0.0,
    )
    over = torch.clamp(
        (inp.foot_clearance - target - effective_clearance_band) / 0.10,
        min=0.0,
    )
    # 平地窄误差仍使用平方代价；碰障后目标可能瞬间提高到能力上限，远端改用
    # Huber 线性尾部，避免稀疏单帧产生数十倍梯度并污染共享的平地策略。
    under_cost = torch.where(under <= 1.0, under.square(), 1.5 * under - 0.5)
    over_cost = torch.where(over <= 1.0, over.square(), 1.5 * over - 0.5)
    under_weighted = under_cost * apex_weight
    over_weighted = over_cost * apex_weight
    apex_denom = apex_weight.sum(dim=-1).clamp(min=1.0)
    under_mean = under_weighted.sum(dim=-1) / apex_denom
    over_mean = over_weighted.sum(dim=-1) / apex_denom
    under_worst = under_weighted.max(dim=-1).values
    over_worst = over_weighted.max(dim=-1).values
    worst_mix = min(max(float(getattr(cfg, "clearance_worst_leg_mix", 0.50)), 0.0), 1.0)
    comp["clearance_under"] = -cfg.w_clearance_under * (
        (1.0 - worst_mix) * under_mean + worst_mix * under_worst
    )
    comp["clearance_over"] = -cfg.w_clearance_over * (
        (1.0 - worst_mix) * over_mean + worst_mix * over_worst
    )

    # 步态结构：只在运动时生效。单纯速度跟踪可能被爬行/拖脚捷径满足，
    # 这些项把对角小跑接触节律、前后支撑和左右平衡显式交给 PPO。
    stance = (inp.foot_contact > 0.5).to(f)
    desired = getattr(inp, "desired_foot_contact", None)
    if desired is not None:
        desired = desired.to(dtype=f, device=stance.device)
        event_active = torch.as_tensor(
            getattr(inp, "terrain_event_active", torch.zeros_like(gate)),
            dtype=f,
            device=gate.device,
        )
        lead_foot = getattr(inp, "terrain_lead_foot_mask", None)
        if lead_foot is None:
            gait_leg_weight = torch.ones_like(stance)
        else:
            lead_foot = lead_foot.to(dtype=f, device=stance.device)
            gait_leg_weight = 1.0 - event_active[:, None] * lead_foot
        gait_denom = gait_leg_weight.sum(dim=-1).clamp(min=1.0)
        gait_mismatch = (
            (stance - desired).abs() * gait_leg_weight
        ).sum(dim=-1) / gait_denom
        gait_match = 1.0 - gait_mismatch
        false_contact = (
            torch.clamp(stance - desired, min=0.0) * gait_leg_weight
        ).sum(dim=-1) / gait_denom
        gait_score = torch.clamp(
            2.0 * gait_match
            - 1.0
            - float(getattr(cfg, "gait_false_contact_scale", 2.0)) * false_contact,
            min=0.0,
        )
        comp["gait_phase_mismatch"] = (
            -float(getattr(cfg, "w_gait_phase_mismatch", 0.0))
            * gait_mismatch
            * linear_gait_gate
            * terrain_pattern_scale
            * moving_gate
            * gate
        )
        # 正向步态锚点不走 quality_gate 渐入；从一开始就奖励符合 gait clock 的接触节律，
        # 使干净小跑成为吸引子。额外支撑脚不能继续领取半额正奖。
        comp["gait_anchor"] = (
            getattr(cfg, "w_gait_anchor", 0.0)
            * gait_score
            * progress_style_gate
            * support_quality
            * terrain_pattern_scale
            * moving_gate
            * capability_gate
        )
        desired_count = desired.sum(dim=-1)
        excess_support = torch.clamp(
            (contact_count - desired_count)
            / torch.clamp(4.0 - desired_count, min=1.0),
            0.0,
            1.0,
        )
        steady_scope = torch.as_tensor(
            getattr(inp, "foot_trajectory_scope", moving_gate),
            dtype=f,
            device=gate.device,
        )
        comp["excess_support"] = (
            -float(getattr(cfg, "w_excess_support", 0.0))
            * excess_support.square()
            * steady_scope
            * terrain_pattern_scale
            * gate
        )
    else:
        comp["gait_anchor"] = torch.zeros_like(gate)
        comp["gait_phase_mismatch"] = torch.zeros_like(gate)
        comp["excess_support"] = torch.zeros_like(gate)
    diagonal_pair_window = getattr(inp, "diagonal_pair_window", None)
    duty_quality_window = getattr(inp, "duty_quality_window", None)
    duty_cycle_valid_window = getattr(inp, "duty_cycle_valid_window", None)
    if diagonal_pair_window is not None and duty_quality_window is not None:
        diagonal_pair_window = diagonal_pair_window.to(dtype=f, device=gate.device)
        duty_quality_window = duty_quality_window.to(dtype=f, device=gate.device)
        comp["diagonal_contact"] = (
            -getattr(cfg, "w_diagonal_contact", 0.0)
            * torch.clamp(1.0 - diagonal_pair_window, 0.0, 1.0)
            * diag_cmd_gate
            * terrain_pattern_scale
            * moving_gate
            * gate
        )
        if duty_cycle_valid_window is not None:
            duty_cycle_valid_window = duty_cycle_valid_window.to(dtype=f, device=gate.device)
            comp["contact_exchange"] = (
                getattr(cfg, "w_contact_exchange", 0.0)
                * torch.clamp(diagonal_pair_window, 0.0, 1.0)
                * torch.clamp(duty_quality_window, 0.0, 1.0)
                * torch.clamp(duty_cycle_valid_window, 0.0, 1.0)
                * diag_cmd_gate
                * terrain_pattern_scale
                * moving_gate
                * capability_gate
            )
        else:
            comp["contact_exchange"] = torch.zeros_like(gate)
        comp["duty_balance"] = (
            -getattr(cfg, "w_duty_balance", 0.0)
            * (torch.clamp(1.0 - duty_quality_window, 0.0, 1.0) ** 2)
            * duty_cmd_gate
            * terrain_pattern_scale
            * moving_gate
            * gate
        )
    elif stance.shape[-1] >= 4:
        fl, fr, rl, rr = stance[:, 0], stance[:, 1], stance[:, 2], stance[:, 3]
        diagonal_mismatch = 0.5 * ((fl - rr).abs() + (fr - rl).abs())
        front_duty = 0.5 * (fl + fr)
        rear_duty = 0.5 * (rl + rr)
        left_duty = 0.5 * (fl + rl)
        right_duty = 0.5 * (fr + rr)
        fb_bias = torch.clamp((front_duty - rear_duty).abs() - getattr(cfg, "rear_duty_tolerance", 0.05), min=0.0)
        lr_bias = (left_duty - right_duty).abs()
        comp["diagonal_contact"] = (
            -getattr(cfg, "w_diagonal_contact", 0.0)
            * diagonal_mismatch
            * diag_cmd_gate
            * terrain_pattern_scale
            * moving_gate
            * gate
        )
        comp["duty_balance"] = (
            -getattr(cfg, "w_duty_balance", 0.0)
            * (fb_bias ** 2 + lr_bias ** 2)
            * duty_cmd_gate
            * terrain_pattern_scale
            * moving_gate
            * gate
        )
        comp["contact_exchange"] = torch.zeros_like(gate)
    else:
        comp["diagonal_contact"] = torch.zeros_like(gate)
        comp["duty_balance"] = torch.zeros_like(gate)
        comp["contact_exchange"] = torch.zeros_like(gate)

    period_error = getattr(inp, "contact_period_error_window", None)
    period_valid = getattr(inp, "contact_period_valid_window", None)
    if period_error is not None and period_valid is not None:
        period_error = period_error.to(dtype=f, device=gate.device)
        period_valid = torch.clamp(
            period_valid.to(dtype=f, device=gate.device), 0.0, 1.0
        )
        period_cost = huber_excess(
            period_error,
            float(getattr(cfg, "contact_period_free", 0.05)),
            float(getattr(cfg, "contact_period_scale", 0.15)),
        )
        comp["contact_period"] = (
            -float(getattr(cfg, "w_contact_period", 0.0))
            * period_cost
            * period_valid
            * linear_gait_gate
            * terrain_pattern_scale
            * moving_gate
            * gate
        )
    else:
        comp["contact_period"] = torch.zeros_like(gate)

    # 支撑相滑移：分段、有界。稳定支撑脚速度超过 slip_free_speed 后线性惩罚，
    # 到 slip_speed_scale 饱和。窗口已经由环境按稳定接触和运动状态处理，
    # 不再乘当前帧门控，避免短暂 collapse/stand 掩盖最近滑移。
    slip_w = getattr(cfg, "w_stance_slip", 0.05) + quality_gate * getattr(cfg, "w_stance_slip_late", 0.0)
    slip_scale = max(float(getattr(cfg, "slip_speed_scale", 1.0)), 1e-6)
    slip_free = float(getattr(cfg, "slip_free_speed", 0.15))
    speed_window = getattr(inp, "stance_slip_speed_window", None)
    if speed_window is not None:
        speed_window = speed_window.to(dtype=f, device=gate.device)
        comp["stance_slip"] = -slip_w * torch.clamp((speed_window - slip_free) / slip_scale, 0.0, 1.0)
    else:
        slip_stance = getattr(inp, "stance_slip_contact", stance)
        slip_stance = slip_stance.to(dtype=f, device=stance.device)
        denom = slip_stance.sum(dim=-1).clamp(min=1.0)
        speed_mean = (inp.foot_vel_xy * slip_stance).sum(dim=-1) / denom
        comp["stance_slip"] = -slip_w * torch.clamp((speed_mean - slip_free) / slip_scale, 0.0, 1.0)
    slip_by_leg = getattr(inp, "stance_slip_by_leg_window", None)
    if slip_by_leg is not None:
        slip_by_leg = slip_by_leg.to(dtype=f, device=gate.device)
        worst_slip = slip_by_leg.max(dim=-1).values
        comp["worst_leg_slip"] = -getattr(cfg, "w_worst_leg_slip", 0.0) * torch.clamp(
            (worst_slip - slip_free) / slip_scale, 0.0, 1.0
        ) * moving_gate * gate
    else:
        comp["worst_leg_slip"] = torch.zeros_like(gate)
    contact_chatter = getattr(inp, "contact_chatter_window", None)
    if contact_chatter is not None:
        contact_chatter = torch.clamp(
            contact_chatter.to(dtype=f, device=gate.device), 0.0, 1.0
        )
        comp["contact_chatter"] = (
            -float(getattr(cfg, "w_contact_chatter", 0.0))
            * contact_chatter
            * moving_gate
            * terrain_pattern_scale
            * gate
        )
    else:
        comp["contact_chatter"] = torch.zeros_like(gate)

    duty_by_leg = getattr(inp, "duty_by_leg_window", None)
    if duty_by_leg is not None:
        duty_by_leg = duty_by_leg.to(dtype=f, device=gate.device)
        front = 0.5 * (duty_by_leg[:, 0] + duty_by_leg[:, 1])
        rear = 0.5 * (duty_by_leg[:, 2] + duty_by_leg[:, 3])
        left = 0.5 * (duty_by_leg[:, 0] + duty_by_leg[:, 2])
        right = 0.5 * (duty_by_leg[:, 1] + duty_by_leg[:, 3])
        diag_a = 0.5 * (duty_by_leg[:, 0] + duty_by_leg[:, 3])
        diag_b = 0.5 * (duty_by_leg[:, 1] + duty_by_leg[:, 2])
        # 直行关注左右支撑，横移关注前后支撑；纯 yaw 同时需要前后与左右均衡。
        x_weight = cmd[:, 0].abs()
        y_weight = cmd[:, 1].abs()
        z_weight = cmd[:, 2].abs()
        norm = (x_weight + y_weight + z_weight).clamp(min=1e-6)
        linear_fb_scale = max(
            float(getattr(cfg, "linear_front_rear_support_scale", 0.35)),
            0.0,
        )
        duty_imbalance = (
            x_weight * (
                (left - right).abs()
                + linear_fb_scale * (front - rear).abs()
            )
            + y_weight * (front - rear).abs()
            + z_weight / 3.0 * (
                (left - right).abs()
                + (front - rear).abs()
                + (diag_a - diag_b).abs()
            )
        ) / norm
        support_force = getattr(inp, "support_force_by_leg_window", None)
        if support_force is not None:
            support_force = support_force.to(dtype=f, device=gate.device)
            force_sum = support_force.sum(dim=-1).clamp(min=1e-6)
            force_front = support_force[:, 0] + support_force[:, 1]
            force_rear = support_force[:, 2] + support_force[:, 3]
            force_left = support_force[:, 0] + support_force[:, 2]
            force_right = support_force[:, 1] + support_force[:, 3]
            force_diag_a = support_force[:, 0] + support_force[:, 3]
            force_diag_b = support_force[:, 1] + support_force[:, 2]
            force_imbalance = (
                x_weight * (
                    (force_left - force_right).abs()
                    + linear_fb_scale * (force_front - force_rear).abs()
                )
                + y_weight * (force_front - force_rear).abs()
                + z_weight / 3.0 * (
                    (force_left - force_right).abs()
                    + (force_front - force_rear).abs()
                    + (force_diag_a - force_diag_b).abs()
                )
            ) / (norm * force_sum)
            imbalance = 0.4 * duty_imbalance + 0.6 * torch.clamp(force_imbalance, 0.0, 1.0)
        else:
            imbalance = duty_imbalance
        # 纯横移时逐足比较“相对机身速度”与真实机身速度对应的无滑移运动学。
        # 世界系足速旋转到机体系后仍是绝对速度，不能与 -cmd_vy 比较；那会惩罚
        # 正确的世界静止支撑足并鼓励滑动。逐足均值与最差腿同时使用，避免组平均
        # 掩盖前足前后摆动、单侧碎触地或前后腿走不同轨迹。
        foot_relative_velocity_body = getattr(inp, "foot_relative_velocity_body", None)
        if foot_relative_velocity_body is not None:
            foot_relative_velocity_body = foot_relative_velocity_body.to(dtype=f, device=gate.device)
            settled_stance = getattr(inp, "stance_slip_contact", stance)
            settled_stance = settled_stance.to(dtype=f, device=gate.device)
            expected_relative_xy = -inp.base_lin_vel[:, None, :2]
            relative_error_xy = (
                foot_relative_velocity_body[:, :, :2] - expected_relative_xy
            ).abs()
            lateral_scale = inp.base_lin_vel[:, 1].abs() + 0.10
            per_leg_error = torch.clamp(
                (relative_error_xy[:, :, 1] + 0.5 * relative_error_xy[:, :, 0])
                / (1.5 * lateral_scale[:, None]),
                0.0,
                1.0,
            )
            stance_count = settled_stance.sum(dim=1)
            error_mean = (per_leg_error * settled_stance).sum(dim=1) / stance_count.clamp(min=1.0)
            error_worst = (per_leg_error * settled_stance).max(dim=1).values
            lateral_kinematic_error = torch.where(
                stance_count > 0.5,
                0.5 * error_mean + 0.5 * error_worst,
                torch.ones_like(error_mean),
            )
            steady_scope = getattr(inp, "foot_trajectory_scope", torch.ones_like(gate))
            lateral_scope = (
                pure_y.to(f)
                * steady_scope.to(dtype=f, device=gate.device)
                * terrain_pattern_scale
            )
            imbalance = torch.clamp(
                (1.0 - 0.25 * lateral_scope) * imbalance
                + 0.25 * lateral_scope * lateral_kinematic_error,
                0.0,
                1.0,
            )
        comp["directional_support_balance"] = -getattr(cfg, "w_directional_support_balance", 0.0) * (
            imbalance ** 2
        ) * terrain_pattern_scale * moving_gate * gate
        lat_cmd = cmd[:, 1].abs()
        lat_progress = torch.clamp(
            inp.base_lin_vel[:, 1] * torch.sign(cmd[:, 1]) / lat_cmd.clamp(min=0.05),
            0.0,
            1.0,
        )
        yaw_residual = torch.clamp(inp.base_ang_vel[:, 2].abs() / (lat_cmd + 0.10), 0.0, 2.0)
        wxy_residual = torch.clamp(wxy_mag / 0.8, 0.0, 2.0)
        cross_residual = torch.clamp(inp.base_lin_vel[:, 0].abs() / (lat_cmd + 0.10), 0.0, 2.0)
        lat_quality = torch.exp(
            -3.0 * imbalance.pow(2)
            - yaw_residual.pow(2)
            - 0.5 * wxy_residual.pow(2)
            - cross_residual.pow(2)
        )
        lat_floor = min(max(float(getattr(cfg, "lateral_quality_floor", 0.30)), 0.0), 1.0)
        comp["lateral_coordinated_progress"] = (
            getattr(cfg, "w_lateral_coordinated_progress", 0.0)
            * lat_progress
            * (lat_floor + (1.0 - lat_floor) * lat_quality)
            * pure_y.to(f)
            * moving_gate
            * capability_gate
            * task_credit_gate
        )
    else:
        comp["directional_support_balance"] = torch.zeros_like(gate)
        comp["lateral_coordinated_progress"] = torch.zeros_like(gate)

    foot_relative_velocity_body = getattr(inp, "foot_relative_velocity_body", None)
    if foot_relative_velocity_body is not None:
        foot_relative_velocity_body = foot_relative_velocity_body.to(dtype=f, device=gate.device)
        pair_a = torch.linalg.norm(
            foot_relative_velocity_body[:, 0, :2] - foot_relative_velocity_body[:, 3, :2],
            dim=-1,
        )
        pair_b = torch.linalg.norm(
            foot_relative_velocity_body[:, 1, :2] - foot_relative_velocity_body[:, 2, :2],
            dim=-1,
        )
        lat_scale = cmd[:, 1].abs() + 0.10
        pair_error = 0.5 * (pair_a + pair_b) / lat_scale
        pair_cost = huber_excess(
            pair_error,
            float(getattr(cfg, "lateral_pair_velocity_free", 0.12)),
            float(getattr(cfg, "lateral_pair_velocity_scale", 0.60)),
        )
        steady_scope = torch.as_tensor(
            getattr(inp, "foot_trajectory_scope", moving_gate),
            dtype=f,
            device=gate.device,
        )
        comp["lateral_pair_velocity"] = (
            -float(getattr(cfg, "w_lateral_pair_velocity", 0.0))
            * pair_cost
            * pure_y.to(f)
            * steady_scope
            * terrain_pattern_scale
            * gate
        )
    else:
        comp["lateral_pair_velocity"] = torch.zeros_like(gate)

    cycle_wxy = getattr(inp, "cycle_wxy_energy", None)
    if cycle_wxy is not None:
        cycle_wxy = cycle_wxy.to(dtype=f, device=gate.device)
        cycle_rms = torch.sqrt(torch.clamp(cycle_wxy, min=0.0))
        cycle_excess = 2.0 * huber_excess(
            cycle_rms,
            float(getattr(cfg, "cycle_wxy_rms_free", 0.18)),
            float(getattr(cfg, "cycle_wxy_rms_scale", 0.45)),
        )
        cycle_w = (
            getattr(cfg, "w_cycle_wxy_bias", 0.0)
            + refinement_gate * getattr(cfg, "w_cycle_wxy_bias_late", 0.0)
        )
        comp["cycle_wxy_bias"] = (
            -cycle_w * cycle_excess * terrain_pattern_scale * moving_gate * gate
        )
    else:
        comp["cycle_wxy_bias"] = torch.zeros_like(gate)
    cycle_vz = getattr(inp, "cycle_vz_energy", None)
    if cycle_vz is not None:
        cycle_vz_rms = torch.sqrt(torch.clamp(
            cycle_vz.to(dtype=f, device=gate.device), min=0.0
        ))
        cycle_vz_excess = 2.0 * huber_excess(
            cycle_vz_rms,
            float(getattr(cfg, "cycle_vz_rms_free", 0.08)),
            float(getattr(cfg, "cycle_vz_rms_scale", 0.22)),
        )
        comp["cycle_vz_bias"] = (
            -float(getattr(cfg, "w_cycle_vz_bias", 0.0))
            * cycle_vz_excess
            * terrain_pattern_scale
            * moving_gate
            * gate
        )
    else:
        comp["cycle_vz_bias"] = torch.zeros_like(gate)
    cycle_yaw = getattr(inp, "cycle_yaw_residual_energy", None)
    if cycle_yaw is not None:
        cycle_yaw_rms = torch.sqrt(torch.clamp(cycle_yaw.to(dtype=f, device=gate.device), min=0.0))
        yaw_excess = 2.0 * huber_excess(
            cycle_yaw_rms,
            float(getattr(cfg, "cycle_yaw_rms_free", 0.08)),
            float(getattr(cfg, "cycle_yaw_rms_scale", 0.35)),
        )
        linear_scope = (
            (cmd[:, :2].norm(dim=-1) > 0.05) & (cmd[:, 2].abs() <= 0.05)
        ).to(f)
        comp["cycle_yaw_residual"] = (
            -getattr(cfg, "w_cycle_yaw_residual", 0.0)
            * yaw_excess
            * linear_scope
            * terrain_pattern_scale
            * gate
        )
    else:
        comp["cycle_yaw_residual"] = torch.zeros_like(gate)
    # 滑移尾部惩罚：在均值窗口之外，额外惩罚高滑移脚的比例。
    # 默认权重为 0 时不生效；非零时通过 quality_gate 后期渐入。
    slip_high = getattr(inp, "stance_slip_high_fraction", None)
    if slip_high is not None:
        slip_high = slip_high.to(dtype=f, device=gate.device)
        slip_high_w = getattr(cfg, "w_stance_slip_high", 0.0) + quality_gate * getattr(cfg, "w_stance_slip_high_late", 0.0)
        comp["stance_slip"] = comp["stance_slip"] - slip_high_w * slip_high

    # 足端滞空时间：反碎步，并与 gait clock 一致。
    # 目标来自当前步态时钟的摆动时长；短摆动被惩罚，过长摆动不给额外奖励，
    # 上界由 gait_anchor 约束。
    td_mask_air = getattr(inp, "touchdown_event_mask", getattr(inp, "touchdown_mask", None))
    foot_air_time = getattr(inp, "foot_air_time", None)
    if td_mask_air is not None and foot_air_time is not None:
        td_mask_air = td_mask_air.to(dtype=f, device=gate.device)
        foot_air_time = foot_air_time.to(dtype=f, device=gate.device)
        air_target = getattr(inp, "air_time_target", None)
        if air_target is None:
            air_target = getattr(cfg, "air_time_target", 0.30)
        else:
            air_target = air_target.to(dtype=f, device=gate.device)
            if air_target.ndim == 1:
                air_target = air_target[:, None]
        air_short = torch.clamp(foot_air_time - air_target, max=0.0)   # 只惩罚短摆动。
        comp["feet_air_time"] = getattr(cfg, "w_feet_air_time", 0.0) * (
            air_short * td_mask_air).sum(dim=-1) * moving_gate * gate
    else:
        comp["feet_air_time"] = torch.zeros_like(gate)

    # 落脚冲击：优先只在真实触地事件上计算。
    # 如果环境提供 touchdown_mask，则只惩罚摆动到支撑的触地瞬间，
    # 避免把正常摆腿竖直速度也当成落脚冲击。
    impact_scale = max(float(getattr(cfg, "impact_speed_scale", 1.0)), 1e-6)
    impact_over = smooth_bounded_excess(
        inp.touchdown_vz,
        getattr(cfg, "touchdown_vz_free", 0.20),
        impact_scale,
    )
    td_mask = getattr(inp, "touchdown_mask", None)
    touchdown_scale = float(getattr(inp, "touchdown_persistence_scale", 1.0))
    touchdown_cost_scale = touchdown_scale * terrain_pattern_scale
    impact_w = getattr(cfg, "w_landing_impact", 1.0) + quality_gate * getattr(cfg, "w_landing_impact_late", 0.0)
    if td_mask is not None:
        td_weight = td_mask.to(dtype=impact_over.dtype, device=impact_over.device)
        event_mean = masked_event_mean(impact_over, td_weight)
        comp["landing_impact"] = -touchdown_cost_scale * impact_w * event_mean
    else:
        td_weight = None
        comp["landing_impact"] = -impact_w * impact_over.mean(dim=-1) * terrain_pattern_scale
    # 落脚冲击尾部惩罚：关注最重的一只脚，避免均值掩盖单脚重砸。
    if getattr(cfg, "w_landing_impact_tail", 0.0) != 0.0 or getattr(cfg, "w_landing_impact_tail_late", 0.0) != 0.0:
        tail_w = getattr(cfg, "w_landing_impact_tail", 0.0) + quality_gate * getattr(cfg, "w_landing_impact_tail_late", 0.0)
        # 均值项保持有界，避免正常触地压倒任务收益；最差足尾部使用 Huber，
        # 让 1 m/s 以上的重砸仍有可用梯度，而不是全部挤在同一个饱和值上。
        impact_tail = huber_excess(
            inp.touchdown_vz,
            getattr(cfg, "touchdown_vz_free", 0.20),
            impact_scale,
        )
        worst = (
            (impact_tail * td_weight).max(dim=-1).values
            if td_weight is not None else impact_tail.max(dim=-1).values
        )
        comp["landing_impact"] = comp["landing_impact"] - touchdown_cost_scale * tail_w * worst

    # 触地滑移：与竖直落脚冲击对应，但作用在触地瞬间的足端水平速度。
    # 它补上支撑相滑移不覆盖触地帧的盲区，抑制脚前勾和侧向挂脚。
    # 只看足端世界速度，不直接惩罚机身速度，因此不压制正常推进。
    ts_w = getattr(cfg, "w_touchdown_slip", 0.0) + quality_gate * getattr(cfg, "w_touchdown_slip_late", 0.0)
    tail_w = float(getattr(cfg, "w_touchdown_slip_tail", 0.0))
    if (
        getattr(cfg, "w_touchdown_slip", 0.0) != 0.0
        or getattr(cfg, "w_touchdown_slip_late", 0.0) != 0.0
        or tail_w != 0.0
    ):
        touchdown_xy = getattr(inp, "touchdown_xy_speed", inp.foot_vel_xy)
        ts_over = smooth_bounded_excess(touchdown_xy, 0.05, impact_scale)
        comp["touchdown_slip"] = -touchdown_cost_scale * ts_w * (
            masked_event_mean(ts_over, td_weight) if td_weight is not None else ts_over.mean(dim=-1)
        )
        if tail_w != 0.0:
            touchdown_xy_tail = huber_excess(touchdown_xy, 0.05, impact_scale)
            worst_xy = (
                (touchdown_xy_tail * td_weight).max(dim=-1).values
                if td_weight is not None else touchdown_xy_tail.max(dim=-1).values
            )
            comp["touchdown_slip"] -= touchdown_cost_scale * tail_w * worst_xy

    if trajectory_error is not None:
        pure_back = (
            (cmd[:, 0] < -0.05)
            & (cmd[:, 1].abs() <= 0.05)
            & (cmd[:, 2].abs() <= 0.05)
        ).to(f)
        direction_scale = 1.0 + pure_back * (float(getattr(cfg, "foot_trajectory_back_scale", 1.20)) - 1.0)
        comp["foot_trajectory"] = (
            -float(getattr(cfg, "w_foot_trajectory", 0.0))
            * trajectory_shape
            * trajectory_scope
            * direction_scale
            * terrain_pattern_scale
            * gate
        )
    else:
        comp["foot_trajectory"] = torch.zeros_like(gate)

    terminal_swing_velocity = getattr(inp, "terminal_swing_velocity_cost", None)
    if terminal_swing_velocity is not None:
        terminal_swing_velocity = terminal_swing_velocity.to(dtype=f, device=gate.device)
        comp["terminal_swing_velocity"] = (
            -float(getattr(cfg, "w_terminal_swing_velocity", 0.0))
            * terminal_swing_velocity
            * terrain_pattern_scale
            * gate
        )
    else:
        comp["terminal_swing_velocity"] = torch.zeros_like(gate)

    # 轻脚只约束触地瞬间的承重跃迁，不能通过全局减小力矩牺牲支撑和爬台阶发力。
    touchdown_force_rate = getattr(inp, "touchdown_force_rate", None)
    if touchdown_force_rate is not None and td_mask is not None:
        force_over = torch.clamp(
            touchdown_force_rate.to(dtype=f, device=gate.device)
            / max(float(getattr(cfg, "touchdown_force_rate_scale", 0.45)), 1e-6),
            0.0,
            1.0,
        )
        comp["touchdown_force_rate"] = (
            -touchdown_cost_scale * getattr(cfg, "w_touchdown_force_rate", 0.0)
            * masked_event_mean(force_over.pow(2), td_weight)
        )
    else:
        comp["touchdown_force_rate"] = torch.zeros_like(gate)

    # 摆动轨迹二阶平滑仅作用于非支撑腿；支撑腿保留快速闭环和足够力矩。
    swing_action_accel = getattr(inp, "swing_action_accel", None)
    swing_joint_mask = getattr(inp, "swing_joint_mask", None)
    if swing_action_accel is not None and swing_joint_mask is not None:
        action_scale = max(float(getattr(cfg, "swing_action_accel_scale", 0.30)), 1e-6)
        accel_cost = huber_excess(
            swing_action_accel.to(dtype=f, device=gate.device).abs(),
            0.10 * action_scale,
            action_scale,
        )
        joint_mask = swing_joint_mask.to(dtype=f, device=gate.device)
        denom = joint_mask.sum(dim=-1).clamp(min=1.0)
        action_cost = (accel_cost * joint_mask).sum(dim=-1) / denom
        foot_delta = getattr(inp, "swing_foot_velocity_delta", None)
        swing_foot_mask = getattr(inp, "swing_foot_mask", None)
        if foot_delta is not None and swing_foot_mask is not None:
            foot_scale = max(float(getattr(cfg, "swing_foot_velocity_delta_scale", 0.80)), 1e-6)
            # 约 0.2*scale 的变化视为控制/估计底噪；Huber 尾部持续区分严重突变，
            # 避免旧的硬截断让 60 m/s^2 与更糟的摆动得到相同代价。
            foot_over = huber_excess(
                foot_delta.to(dtype=f, device=gate.device),
                0.20 * foot_scale,
                foot_scale,
            )
            foot_mask = swing_foot_mask.to(dtype=f, device=gate.device)
            foot_denom = foot_mask.sum(dim=-1).clamp(min=1.0)
            foot_mean = (foot_over * foot_mask).sum(dim=-1) / foot_denom
            foot_tail = (foot_over * foot_mask).max(dim=-1).values
            terrain_response = torch.clamp(
                getattr(inp, "terrain_response", torch.zeros_like(gate)).to(dtype=f, device=gate.device), 0.0, 1.0
            )
            foot_cost = (0.5 * foot_mean + 0.5 * foot_tail) * (1.0 - 0.75 * terrain_response)
            smooth_cost = 0.35 * action_cost + 0.65 * foot_cost
        else:
            smooth_cost = action_cost
        comp["swing_action_accel"] = (
            -getattr(cfg, "w_swing_action_accel", 0.0) * smooth_cost
        )
    else:
        comp["swing_action_accel"] = torch.zeros_like(gate)

    # 平地步态姿态和速率质量：机身保持水平，减少弹跳、滚转/俯仰抖动和髋关节外翻。
    # 所有项有界，并由稳定门控避免在 collapse 上叠加无意义惩罚。
    # 正常倾斜区间保留旧线性语义；超过 scale 后保留较小的持续尾部梯度。
    # 直接使用无界斜率会在 fresh 随机策略阶段压倒方向驱动，重新形成近静止吸引子。
    core_tail_gain = min(max(float(getattr(cfg, "core_tail_gain", 0.25)), 0.0), 1.0)
    tilt_excess = torch.clamp(
        (tilt_rel - getattr(cfg, "orient_soft_rad", 0.09))
        / max(getattr(cfg, "orient_scale_rad", 0.35), 1e-6),
        min=0.0,
    )
    tilt_over = torch.clamp(tilt_excess, max=1.0) + core_tail_gain * torch.clamp(
        tilt_excess - 1.0, min=0.0
    )
    # 平地参考接近水平；接触后确认地形时允许必要的低频倾斜，但角速度和角加速度
    # 仍保持约束，使机身像稳定平台而不是随每次触地摆动。
    orient_relief = min(max(float(getattr(cfg, "orient_terrain_relief", 0.70)), 0.0), 0.90)
    orient_scope = 1.0 - orient_relief * terrain_response
    # 安全代价必须覆盖“已经失稳但尚未终止”的区间。若继续使用 stable_motion_gate，
    # 机身越接近倒地，这些代价反而越接近零，而方向收益会一直保留到 terminal，
    # 策略便可能通过短暂冲速后倒地获得净收益。hard_survival_gate 只在真实终止
    # 或数值非法时关闭，因此既保留终止前的纠偏梯度，也不惩罚倒地后的恢复动作。
    safety_gate = hard_survival_gate
    comp["orient"] = -getattr(cfg, "w_orient", 0.0) * tilt_over * orient_scope * safety_gate
    base_vz_cost = 2.0 * huber_excess(
        vel[:, 2].abs(),
        float(getattr(cfg, "base_vz_free", 0.05)),
        float(getattr(cfg, "base_vz_scale", 0.75)),
    )
    comp["base_vz"] = (
        -getattr(cfg, "w_base_vz", 0.0) * base_vz_cost * safety_gate
    )
    wxy_sq = inp.base_ang_vel[:, 0] ** 2 + inp.base_ang_vel[:, 1] ** 2
    # 1 rad/s 以下与旧平方代价完全相同；更大的摇晃改为线性 Huber 尾部，
    # 避免 2 rad/s 以上全部落入同一饱和值。
    wxy_cost = 2.0 * huber_excess(torch.sqrt(wxy_sq), 0.0, 1.0)
    comp["base_wxy"] = (
        -getattr(cfg, "w_base_wxy", 0.0)
        * wxy_cost
        * safety_gate
    )
    base_ang_accel = getattr(inp, "base_ang_accel", None)
    if base_ang_accel is not None:
        accel = base_ang_accel.to(dtype=f, device=gate.device)
        steady_scope = torch.clamp(
            torch.as_tensor(
                getattr(inp, "foot_trajectory_scope", moving_gate),
                dtype=f,
                device=gate.device,
            ),
            0.0,
            1.0,
        )
        # 诊断中始终记录 roll/pitch；平地稳态再计入 yaw。切换窗口排除必要的
        # yaw 加减速，便于把真实高频冲击与正常命令响应分开。
        accel_metric = torch.sqrt(
            accel[:, 0].square()
            + accel[:, 1].square()
            + steady_scope * accel[:, 2].square()
        )
        accel_cost = huber_excess(
            accel_metric,
            getattr(cfg, "base_ang_accel_free", 1.5),
            getattr(cfg, "base_ang_accel_scale", 8.0),
        )
        comp["base_ang_accel"] = (
            -getattr(cfg, "w_base_ang_accel", 0.0) * accel_cost * safety_gate
        )
        base_ang_accel_raw = getattr(inp, "base_ang_accel_raw", None)
        raw_tail_weight = float(getattr(cfg, "w_base_ang_accel_raw_tail", 0.0))
        if base_ang_accel_raw is not None and raw_tail_weight != 0.0:
            raw_accel = base_ang_accel_raw.to(dtype=f, device=gate.device)
            raw_metric = torch.sqrt(
                raw_accel[:, 0].square()
                + raw_accel[:, 1].square()
                + steady_scope * raw_accel[:, 2].square()
            )
            raw_tail = huber_excess(
                raw_metric,
                float(getattr(cfg, "base_ang_accel_raw_free", 10.0)),
                float(getattr(cfg, "base_ang_accel_raw_scale", 20.0)),
            )
            comp["base_ang_accel"] -= raw_tail_weight * raw_tail * safety_gate
    else:
        comp["base_ang_accel"] = torch.zeros_like(gate)
    terrain_relief = terrain_response
    height_target = float(getattr(cfg, "flat_move_height_target", cfg.nominal_base_h))
    height_band = max(float(getattr(cfg, "flat_move_height_band", 0.03)), 1e-6)
    # 带宽外先保持旧平方形状，超过旧饱和点后以较小线性尾部继续区分塌腰。
    height_excess = torch.clamp(
        (torch.abs(base_h_above_terrain - height_target) - height_band) / height_band,
        min=0.0,
    )
    height_error = torch.clamp(height_excess, max=1.0).square() + core_tail_gain * torch.clamp(
        height_excess - 1.0, min=0.0
    )
    height_relief = min(max(float(getattr(cfg, "flat_move_height_terrain_relief", 0.80)), 0.0), 0.90)
    comp["flat_move_height"] = (
        -getattr(cfg, "w_flat_move_height", 0.0)
        * height_error
        * moving_gate
        * (1.0 - height_relief * terrain_relief)
        * safety_gate
    )
    front_rear_extension = getattr(inp, "front_rear_extension_error", None)
    if front_rear_extension is not None:
        front_rear_extension = front_rear_extension.to(dtype=f, device=gate.device)
        extension_cost = huber_excess(
            front_rear_extension,
            float(getattr(cfg, "front_rear_extension_free", 0.015)),
            float(getattr(cfg, "front_rear_extension_scale", 0.05)),
        )
        command_total = cmd[:, :2].abs().sum(dim=-1) + cmd[:, 2].abs()
        lat_yaw_mix = (cmd[:, 1].abs() + cmd[:, 2].abs()) / command_total.clamp(min=1e-6)
        lat_yaw_scale = min(max(float(getattr(
            cfg, "front_rear_extension_lat_yaw_scale", 0.50
        )), 0.0), 1.0)
        extension_scope = torch.where(
            command_total <= 0.05,
            torch.ones_like(command_total),
            1.0 - (1.0 - lat_yaw_scale) * lat_yaw_mix,
        )
        comp["front_rear_extension"] = (
            -float(getattr(cfg, "w_front_rear_extension", 0.0))
            * extension_cost
            * extension_scope
            * terrain_pattern_scale
            * torch.clamp(moving_gate + stand_gate, 0.0, 1.0)
            * gate
        )
    else:
        comp["front_rear_extension"] = torch.zeros_like(gate)
    # 髋关节偏差：前进、后退和站立使用最差半数腿的尾部，防止四足均值掩盖
    # 单腿外翻；横移/yaw 保留必要自由度，真实地形响应后再连续退让。
    hip_dev = getattr(inp, "hip_deviation", None)
    if hip_dev is not None:
        hip_dev = hip_dev.to(dtype=f, device=gate.device)
        hip_by_leg = getattr(inp, "hip_deviation_by_leg", None)
        if hip_by_leg is not None:
            hip_by_leg = hip_by_leg.to(dtype=f, device=gate.device)
            tail_count = max(1, (hip_by_leg.shape[-1] + 1) // 2)
            hip_tail = torch.topk(hip_by_leg, k=tail_count, dim=-1).values.mean(dim=-1)
        else:
            hip_tail = getattr(inp, "hip_deviation_max", hip_dev)
            hip_tail = hip_tail.to(dtype=f, device=gate.device)
        cmd_mag = cmd[:, :2].abs().sum(dim=-1) + cmd[:, 2].abs()
        pure_x_mix = cmd[:, 0].abs() / cmd_mag.clamp(min=1e-6)
        straight_or_stand = (pure_x | (cmd_mag <= 0.05)).to(f)
        straight_mix = torch.maximum(pure_x_mix, straight_or_stand)
        hip_metric = hip_dev * (1.0 - straight_mix) + hip_tail * straight_mix
        lat_mix = cmd[:, 1].abs() / cmd_mag.clamp(min=1e-6)
        yaw_mix = cmd[:, 2].abs() / cmd_mag.clamp(min=1e-6)
        lat_scale = float(getattr(cfg, "hip_deviation_lat_scale", 0.35))
        yaw_scale = float(getattr(cfg, "hip_deviation_yaw_scale", 0.25))
        hip_scope = torch.clamp(
            1.0 - (1.0 - lat_scale) * lat_mix - (1.0 - yaw_scale) * yaw_mix,
            min=0.0,
            max=1.0,
        )
        hip_cost = huber_excess(
            hip_metric,
            float(getattr(cfg, "hip_deviation_free", 0.04)),
            float(getattr(cfg, "hip_deviation_scale", 0.15)),
        )
        comp["hip_deviation"] = (
            -getattr(cfg, "w_hip_deviation", 0.0)
            * hip_cost
            * hip_scope
            * terrain_pattern_scale
            * gate
        )
    else:
        comp["hip_deviation"] = torch.zeros_like(gate)

    # 力矩：margin 使用关节均值；饱和项使用饱和关节比例。
    limits = inp.torque_limit
    if limits.ndim == 1:
        limits = limits.unsqueeze(0).expand_as(inp.torque)
    util = inp.torque.abs() / torch.clamp(limits, min=1e-6)
    margin_excess = torch.clamp((util - 0.85) / 0.15, min=0.0)
    comp["torque_margin"] = -cfg.w_torque_margin * (margin_excess ** 2).mean(dim=-1)
    sat = ((util >= 1.0) | (inp.torque_clamped > 0.5)).to(f)
    comp["torque_saturation"] = -cfg.w_torque_saturation * sat.mean(dim=-1)

    # 动作平滑：每个机器人上的归一化动作变化率。
    comp["action_rate"] = -cfg.w_action_rate * ((inp.action - inp.last_action) ** 2).mean(dim=-1)
    action_cost = huber_excess(
        inp.action.abs(),
        float(getattr(cfg, "action_magnitude_free", 1.0)),
        float(getattr(cfg, "action_magnitude_scale", 1.0)),
    )
    action_tail = 0.5 * action_cost.mean(dim=-1) + 0.5 * action_cost.max(dim=-1).values
    action_relief = min(max(
        float(getattr(cfg, "action_magnitude_terrain_relief", 0.75)), 0.0
    ), 0.95)
    comp["action_magnitude"] = (
        -float(getattr(cfg, "w_action_magnitude", 0.0))
        * action_tail
        * moving_gate
        * (1.0 - action_relief * terrain_response)
        * hard_survival_gate
    )

    # 核心稳定和安全项始终完整生效。它们均已归一化并有界，不能再因为当前代价大
    # 而由预算反馈削弱，否则质量越差、纠偏压力反而越小。
    # 接触与步态质量保留非零 floor，再随四方向能力成熟单调增强。
    contact_quality_components = (
        "clearance_under",
        "clearance_over",
        "diagonal_contact",
        "duty_balance",
        "gait_phase_mismatch",
        "contact_period",
        "excess_support",
        "stance_slip",
        "worst_leg_slip",
        "contact_chatter",
        "directional_support_balance",
        "feet_air_time",
        "landing_impact",
        "touchdown_slip",
        "touchdown_force_rate",
        "terminal_swing_velocity",
    )
    for name in contact_quality_components:
        if name in comp:
            comp[name] = comp[name] * quality_gate

    # 机身高阶动态、轨迹和姿态精修只在四方向基础运动形成后渐入，避免压制步态启动。
    refinement_components = (
        "cycle_wxy_bias",
        "cycle_vz_bias",
        "cycle_yaw_residual",
        "foot_trajectory",
        "swing_action_accel",
        "lateral_pair_velocity",
        "hip_deviation",
        "front_rear_extension",
        "action_rate",
        "planar_purity",
        "heading_hold",
        "yaw_translation",
    )
    for name in refinement_components:
        if name in comp:
            comp[name] = comp[name] * refinement_gate

    # 终止惩罚：只惩罚真实 terminal，不惩罚 timeout。
    is_terminal = 1.0 if (inp.terminal_reason is not None and inp.terminal_reason != "timeout") else 0.0
    comp["terminal_penalty"] = -cfg.w_terminal * torch.full_like(gate, is_terminal)

    total = torch.zeros_like(gate)
    for k, v in comp.items():
        if k not in REWARD_DIAGNOSTIC_COMPONENTS:
            total = total + v
    comp["total"] = total
    return comp


# ---------------------------------------------------------------------------
# 多 critic 奖励分组（TAILI_MULTI_CRITIC）。
# ---------------------------------------------------------------------------
# 将标量奖励分量拆到 K 个目标列中，使多 critic PPO 可以按目标分别做 GAE。
# 这是一个分区：每个非 gate、非 total 分量只进入一个组；未知分量进入 fallback 组。
# 因此 group_reward_vector(comp).sum(dim=-1) 必须等于 comp["total"] 加上外部注入项。
# turn 单独成组，避免 yaw 在 track 组里被更大、更容易的线速度优势淹没。
REWARD_GROUP_NAMES = ("track", "turn", "gait", "stab", "style")
REWARD_GROUP_GATES = REWARD_DIAGNOSTIC_COMPONENTS + ("total",)
REWARD_GROUP_FALLBACK = "stab"
_REWARD_GROUP_OF = {
    # 任务/命令跟踪，主要是正向任务奖励。
    "tracking_lin": "track",
    "tracking_yaw": "turn", "tracking_yaw_far": "turn", "yaw_progress": "turn",
    "yaw_underspeed": "turn",
    "yaw_support_moment": "turn",   # yaw 拥有独立归一化优势。
    "tracking_lin_far": "track",
    "lateral_coordinated_progress": "track",
    "supported_progress": "track", "terrain_progress": "track", "stand": "track", "stand_contact": "track",
    "stand_far": "track", "climb": "track", "terrain_up": "track",
    "terrain_down": "track", "terrain_layer_hold": "track", "terrain_overspeed": "track",
    "terrain_direction_progress": "track", "feet_air_time": "track", "linear_underspeed": "track",
    # 步态质量：节律、接触时序、滑移和抬脚高度带宽。
    "gait_anchor": "gait", "gait_phase_mismatch": "gait", "contact_exchange": "gait",
    "contact_period": "gait", "diagonal_contact": "gait",
    "duty_balance": "gait", "excess_support": "gait",
    "foot_trajectory": "gait",
    "terminal_swing_velocity": "gait",
    "stance_slip": "gait", "worst_leg_slip": "gait", "contact_chatter": "gait",
    "directional_support_balance": "gait",
    "lateral_pair_velocity": "gait",
    "clearance_under": "gait", "clearance_over": "gait",
    "terrain_support_loss": "stab",
    # 稳定性、姿态和安全。
    "orient": "stab", "base_vz": "stab", "base_wxy": "stab", "base_ang_accel": "stab",
    "cycle_wxy_bias": "stab", "cycle_vz_bias": "stab",
    "cycle_yaw_residual": "stab",
    "flat_move_height": "stab", "core_quality": "stab", "support_integrity": "stab", "off_axis": "stab",
    "planar_purity": "stab", "heading_hold": "stab", "yaw_translation": "stab",
    "wrong_dir": "stab", "yaw_wrong_moment": "stab", "backward_underspeed": "stab", "lateral_underspeed": "stab",
    "transition_readiness": "stab", "transition_motion_profile": "stab",
    "stand_posture": "stab", "hip_deviation": "stab", "front_rear_extension": "stab",
    "landing_impact": "stab", "touchdown_force_rate": "stab",
    "lateral_foot_excursion": "stab",
    "terrain_contact_quality": "stab", "terrain_collapse": "stab",
    "stumble": "stab", "terminal_penalty": "stab",
    # 风格、力矩和动作平滑。
    "torque_margin": "style", "torque_saturation": "style", "action_rate": "style",
    "action_magnitude": "style",
    "swing_action_accel": "style",
}


def num_reward_groups() -> int:
    return len(REWARD_GROUP_NAMES)


def group_reward_vector(comp, extra_by_group=None):
    """把奖励分量字典拆成 [N, K] 的分组向量。

    gate 诊断和聚合 total 不进入分组；其余分量按映射累加到目标列。
    未知 key 进入 fallback 组。extra_by_group 用于注入环境侧额外分组项。
    不变量：分组向量按列求和应等于 comp["total"] 加外部注入项之和。
    """
    names = REWARD_GROUP_NAMES
    col = {g: i for i, g in enumerate(names)}
    ref = comp.get("total")
    if ref is None:
        for v in comp.values():
            if torch.is_tensor(v):
                ref = v
                break
    n = ref.shape[0]
    vec = torch.zeros(n, len(names), dtype=ref.dtype, device=ref.device)
    for k, v in comp.items():
        if k in REWARD_GROUP_GATES or not torch.is_tensor(v):
            continue
        g = _REWARD_GROUP_OF.get(k, REWARD_GROUP_FALLBACK)
        vec[:, col[g]] = vec[:, col[g]] + v.to(dtype=ref.dtype)
    if extra_by_group:
        for g, v in extra_by_group.items():
            if v is None:
                continue
            vec[:, col[g]] = vec[:, col[g]] + torch.as_tensor(v, dtype=ref.dtype, device=ref.device)
    return vec
