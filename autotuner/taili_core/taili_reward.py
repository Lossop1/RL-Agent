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
    nominal_base_h: float = 0.52
    h_ok: float = 0.47
    h_gate_close: float = 0.42
    tilt_ok_rad: float = 0.2617993878          # 15 度。
    tilt_gate_close_rad: float = 0.6981317008  # 40 度。
    # 落脚、滑移、抬脚和地形相关尺度。
    flat_clearance_target: float = 0.08
    terrain_clearance_margin: float = 0.04
    clearance_band: float = 0.02
    terrain_v_floor: float = 0.10
    # 奖励权重。
    w_tracking_lin: float = 2.0
    w_tracking_yaw: float = 1.0
    w_stand: float = 1.5
    # 四足站稳：stand 项只看机身静止，不能防止脚下碎步；该项要求零命令下四足接触。
    # 它受 stand_gate 门控，不会在行走时形成站立陷阱。
    w_stand_contact: float = 0.8
    w_terrain_progress: float = 0.5
    # 地形条件下的进展耦合：保留进展 floor，避免早期无梯度。
    # 完整进展奖励要求低滑移；平地偏向规则节律，粗糙/高地形偏向足够抬脚。
    # 这样用一个能力门控表达全局质量，避免堆出大量局部姿态开关。
    healthy_progress_floor: float = 0.35
    healthy_progress_slip_target: float = 0.12
    healthy_progress_slip_width: float = 0.18
    terrain_clearance_drive: float = 0.35
    tracking_touchdown_clean_floor: float = 0.65
    gait_anchor_progress_floor: float = 0.55
    gait_anchor_min_ratio: float = 0.45
    linear_underspeed_min_ratio: float = 0.65
    w_linear_underspeed: float = 0.0
    # 跟踪应表示可用运动，而不仅是机身速度数值匹配。
    # floor 保证早期 PPO 仍有命令跟踪梯度；剩余奖励由接触、滑移和抬脚质量决定。
    validated_tracking_floor: float = 0.40
    tracking_validity_terrain_mix: float = 1.0
    w_torque_margin: float = 1.0
    w_torque_saturation: float = 5.0
    w_clearance_under: float = 0.4
    w_clearance_over: float = 0.05
    w_off_axis: float = 0.5
    w_planar_purity: float = 0.0
    planar_purity_yaw_scale: float = 0.40
    w_heading_hold: float = 0.0
    heading_hold_scale: float = 0.35
    w_action_rate: float = 0.02
    w_stance_slip: float = 0.05
    w_stance_slip_late: float = 0.20
    # 滑移尾部惩罚：除均值外，直接惩罚稳定接触脚超过高滑移阈值的比例。
    # 该项通过 quality_gate 渐入，避免基础步态尚未形成时过早压制。
    w_stance_slip_high: float = 0.0
    w_stance_slip_high_late: float = 0.60
    # 落脚冲击尾部惩罚：关注最重的一只脚，而不是只看四足均值。
    w_landing_impact_tail: float = 0.0
    w_landing_impact_tail_late: float = 0.50
    # 脚前勾修正：惩罚触地瞬间足端水平速度，补上只看竖直落脚速度的盲区。
    # 默认权重为 0 时不改变行为；非零时通过 quality_gate 后期渐入。
    w_touchdown_slip: float = 0.0
    w_touchdown_slip_late: float = 0.60
    # 分段滑移形状：稳定支撑脚速度高于 slip_free_speed 后线性惩罚，到 slip_speed_scale 饱和。
    # slip_free_speed 是测量代理的底噪，不表示允许滑移；低于该值惩罚会直接伤害正常行走。
    # 饱和值限制总量，线性段保留脱离滑移的梯度。
    slip_free_speed: float = 0.15
    slip_speed_scale: float = 1.0
    # 落脚冲击使用与滑移类似的分段形状：超过 0.10 后线性惩罚，到 impact_speed_scale 饱和。
    impact_speed_scale: float = 1.0
    w_landing_impact: float = 0.15
    w_landing_impact_late: float = 0.85
    # 平地步态姿态和速率质量，全部有界：
    # orient：倾斜超过约 5 度后惩罚；base_vz：竖直弹跳；base_wxy：滚转/俯仰角速度。
    # hip_deviation：髋关节偏离默认外展位置的幅度。
    w_orient: float = 0.30
    orient_soft_rad: float = 0.09
    orient_scale_rad: float = 0.35
    w_base_vz: float = 1.0
    w_base_wxy: float = 0.10
    w_hip_deviation: float = 0.40
    hip_deviation_scale: float = 0.50
    w_diagonal_contact: float = 0.20
    w_duty_balance: float = 0.20
    duty_target: float = 0.50
    quality_window_ema_beta: float = 0.98
    slip_high_threshold: float = 0.20
    # 正向步态锚点：奖励接触节律匹配 gait clock，使小跑节律成为吸引子。
    w_gait_anchor: float = 1.0
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


def far_kernel(err, scale):
    """Laplacian shaping 核 exp(-err/scale)。

    输出有界、单调递减，并且在大误差区仍保留非零梯度，用作远场吸引项。
    """
    scale = torch.clamp(torch.as_tensor(scale, dtype=err.dtype, device=err.device), min=1e-9)
    return torch.exp(-err.abs() / scale)


def _smoothstep(edge0, edge1, x):
    t = torch.clamp((x - edge0) / (edge1 - edge0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _metric_gate(value, target, *, higher=True, width=0.10):
    """软门控，用于把进展奖励与步态健康耦合，同时保留梯度。"""
    target_t = torch.as_tensor(target, dtype=value.dtype, device=value.device)
    width_t = torch.as_tensor(width, dtype=value.dtype, device=value.device).clamp_min(1e-6)
    if higher:
        return _smoothstep(target_t - width_t, target_t, value)
    return 1.0 - _smoothstep(target_t, target_t + width_t, value)


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
    comp["stable_motion_gate"] = gate
    comp["stand_gate"] = stand_gate
    comp["moving_gate"] = moving_gate
    comp["quality_gate"] = quality_gate

    cmd = inp.cmd
    vel = inp.base_lin_vel
    cmd_xy = cmd[:, :2]
    cmd_xy_norm = torch.linalg.norm(cmd_xy, dim=-1)
    foot_contact = getattr(inp, "foot_contact", None)
    if foot_contact is not None:
        foot_contact = foot_contact.to(dtype=f, device=gate.device)
        contact_count = foot_contact.sum(dim=-1)
    else:
        contact_count = torch.full_like(gate, 2.0)

    # 原始命令跟踪信号。真正发放奖励前还会乘以接触/地形有效性门控，
    # 使“跟踪”表示可用运动，而不是打滑、拖脚或撞击时的机身速度匹配。
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
    comp["stand"] = cfg.w_stand * stand * stand_gate * gate

    # 四足站稳：零命令下奖励四足接触，防止“机身静止但脚下碎步”拿到高 stand 分。
    # 该项只在 stand_gate 下生效，不影响行走。
    if foot_contact is not None:
        four_foot = foot_contact.mean(dim=-1)
        comp["stand_contact"] = getattr(cfg, "w_stand_contact", 0.0) * four_foot * stand_gate * gate
    else:
        comp["stand_contact"] = torch.zeros_like(gate)

    # 远场 shaping：在大误差区提供梯度，把策略拉向命令目标。
    # 它使用与高斯跟踪相同的门控，collapse 时关闭，不引入新的退化最优。
    s_far = getattr(cfg, "scale_track_far", 0.5)
    s_stand_far = getattr(cfg, "scale_stand_far", 0.3)
    tracking_lin_far_raw = far_kernel(lin_err, s_far)
    s_yaw_far = getattr(cfg, "scale_yaw_far", s_far)
    tracking_yaw_far_raw = far_kernel(yaw_err, s_yaw_far)
    comp["stand_far"] = (getattr(cfg, "w_stand_far", 0.6) * far_kernel(speed, s_stand_far)
                         * far_kernel(yaw_rate, s_stand_far) * stand_gate * gate)

    # 地形稳定进展：只按线速度命令方向计算。
    # 原始进展 = dot(v_xy, cmd_xy) / |cmd_xy|；纯 yaw 不拥有线性地形进展。
    # 这样避免横移/yaw 命令下把前向漂移误判为进展。
    cmd_xy_norm_safe = torch.clamp(cmd_xy_norm, min=1e-6)
    v_along = torch.sum(vel[:, :2] * cmd_xy, dim=-1) / cmd_xy_norm_safe
    has_linear_cmd = cmd_xy_norm > 0.05
    ratio = torch.clamp(v_along / cmd_xy_norm_safe, 0.0, 1.0)
    floor_ok = ((v_along >= cfg.terrain_v_floor) & has_linear_cmd).to(f)
    style_min_ratio = max(float(getattr(cfg, "gait_anchor_min_ratio", 0.45)), 1e-6)
    style_progress = torch.clamp(v_along / (cmd_xy_norm_safe * style_min_ratio), 0.0, 1.0)
    style_floor = float(getattr(cfg, "gait_anchor_progress_floor", 0.55))
    progress_style_gate = torch.where(
        has_linear_cmd,
        torch.clamp(style_floor + (1.0 - style_floor) * style_progress, 0.0, 1.0),
        torch.ones_like(gate),
    )
    swing_for_progress = (inp.foot_contact <= 0.5).to(f)
    target_for_progress = torch.where(
        inp.local_obstacle_h[:, None] > 0,
        inp.local_obstacle_h[:, None] + cfg.terrain_clearance_margin,
        torch.full_like(inp.foot_clearance, cfg.flat_clearance_target),
    )
    under_for_progress = torch.clamp(
        (target_for_progress - inp.foot_clearance - cfg.clearance_band) / 0.05,
        min=0.0,
    )
    clearance_success = torch.clamp(1.0 - under_for_progress, 0.0, 1.0)
    clearance_success = (
        clearance_success * swing_for_progress
    ).sum(dim=-1) / swing_for_progress.sum(dim=-1).clamp(min=1.0)
    terrain_difficulty = torch.clamp(inp.local_obstacle_h / 0.25, 0.0, 1.0)

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
        td_over = torch.clamp((td_gate_vz.abs() - 0.10) / impact_scale, 0.0, 1.0)
        td_bad = (td_over * td_gate_mask).max(dim=-1).values
        td_floor = float(getattr(cfg, "tracking_touchdown_clean_floor", 0.65))
        td_clean = torch.clamp(td_floor + (1.0 - td_floor) * (1.0 - td_bad), 0.0, 1.0)
        touchdown_clean_gate = torch.where(td_event, td_clean, touchdown_clean_gate)
    rhythm_gate = 0.5 * (
        _metric_gate(diag_score, 0.52, higher=True, width=0.18)
        + _metric_gate(duty_score, 0.58, higher=True, width=0.18)
    )
    terrain_mix = torch.clamp(
        terrain_difficulty * float(getattr(cfg, "tracking_validity_terrain_mix", 1.0)),
        0.0,
        1.0,
    )
    valid_contact = slip_gate * touchdown_clean_gate * (
        (1.0 - terrain_mix) * rhythm_gate + terrain_mix * clearance_success
    )
    healthy_floor = float(getattr(cfg, "healthy_progress_floor", 0.35))
    healthy_progress_gate = torch.clamp(healthy_floor + (1.0 - healthy_floor) * valid_contact, 0.0, 1.0)
    tracking_floor = float(getattr(cfg, "validated_tracking_floor", healthy_floor))
    validated_tracking_gate = torch.clamp(tracking_floor + (1.0 - tracking_floor) * valid_contact, 0.0, 1.0)
    terrain_drive = 1.0 + float(getattr(cfg, "terrain_clearance_drive", 0.35)) * terrain_difficulty * clearance_success
    comp["healthy_progress_gate"] = healthy_progress_gate
    comp["validated_tracking_gate"] = validated_tracking_gate
    comp["touchdown_clean_gate"] = touchdown_clean_gate
    comp["progress_style_gate"] = progress_style_gate
    comp["tracking_lin"] = cfg.w_tracking_lin * tracking_lin_raw * moving_gate * gate * validated_tracking_gate
    comp["tracking_yaw"] = cfg.w_tracking_yaw * tracking_yaw_raw * moving_gate * gate * yaw_cmd_gate * validated_tracking_gate
    comp["tracking_lin_far"] = getattr(cfg, "w_track_far", 0.6) * tracking_lin_far_raw * moving_gate * gate * validated_tracking_gate
    comp["tracking_yaw_far"] = getattr(cfg, "w_yaw_far", 0.3) * tracking_yaw_far_raw * moving_gate * gate * yaw_cmd_gate * validated_tracking_gate
    comp["terrain_progress"] = cfg.w_terrain_progress * ratio * floor_ok * gate * healthy_progress_gate * terrain_drive
    underspeed_w = float(getattr(cfg, "w_linear_underspeed", 0.0))
    if underspeed_w != 0.0:
        min_ratio = max(float(getattr(cfg, "linear_underspeed_min_ratio", 0.65)), 0.0)
        min_speed = cmd_xy_norm * min_ratio
        underspeed = torch.clamp(min_speed - v_along, min=0.0, max=1.5)
        comp["linear_underspeed"] = -underspeed_w * (underspeed ** 2) * has_linear_cmd.to(f) * moving_gate * gate * quality_gate
    else:
        comp["linear_underspeed"] = torch.zeros_like(gate)

    # 反向运动压力：主动惩罚与线速度命令方向或 yaw 符号相反的运动。
    # 只扣掉跟踪奖励不够，因为反向吸引子仍可能没有直接成本。
    w_wd = getattr(cfg, "w_wrong_dir", 0.0)
    lin_wrong = torch.clamp(-v_along, min=0.0, max=1.5) * has_linear_cmd.to(f)
    has_yaw_cmd = cmd[:, 2].abs() > 0.05
    yaw_wrong = torch.clamp(-(inp.base_ang_vel[:, 2] * torch.sign(cmd[:, 2])), min=0.0, max=1.5) * has_yaw_cmd.to(f)
    comp["wrong_dir"] = -w_wd * (lin_wrong ** 2 + yaw_wrong ** 2) * moving_gate * gate

    # 非命令轴残差：只惩罚接近零命令的轴。
    # 该项不受 moving_gate 控制；站立时所有轴都接近零命令，因此它也是静止抗漂移压力。
    eps = 0.05
    off = torch.zeros_like(speed)
    off = off + torch.where(cmd[:, 0].abs() <= eps, vel[:, 0].abs(), torch.zeros_like(off))
    off = off + torch.where(cmd[:, 1].abs() <= eps, vel[:, 1].abs(), torch.zeros_like(off))
    off = off + torch.where(cmd[:, 2].abs() <= eps, inp.base_ang_vel[:, 2].abs(), torch.zeros_like(off))
    comp["off_axis"] = -cfg.w_off_axis * (off ** 2) * gate

    pure_x = (cmd[:, 0].abs() > eps) & (cmd[:, 1].abs() <= eps) & (cmd[:, 2].abs() <= eps)
    pure_y = (cmd[:, 1].abs() > eps) & (cmd[:, 0].abs() <= eps) & (cmd[:, 2].abs() <= eps)
    pure_yaw = (cmd[:, 2].abs() > eps) & (cmd_xy_norm <= eps)
    yaw_scale = float(getattr(cfg, "planar_purity_yaw_scale", 0.40))
    purity_err = (
        pure_x.to(f) * (vel[:, 1] ** 2 + yaw_scale * inp.base_ang_vel[:, 2] ** 2)
        + pure_y.to(f) * (vel[:, 0] ** 2 + yaw_scale * inp.base_ang_vel[:, 2] ** 2)
        + pure_yaw.to(f) * (vel[:, 0] ** 2 + vel[:, 1] ** 2)
    )
    comp["planar_purity"] = -getattr(cfg, "w_planar_purity", 0.0) * purity_err * moving_gate * gate

    heading_error = getattr(inp, "heading_error", torch.zeros_like(gate))
    heading_error = torch.as_tensor(heading_error, dtype=f, device=gate.device)
    no_yaw_linear = ((cmd[:, 2].abs() <= eps) & (cmd_xy_norm > eps)).to(f)
    heading_scale = max(float(getattr(cfg, "heading_hold_scale", 0.35)), 1e-6)
    heading_cost = torch.clamp(heading_error.abs() / heading_scale, 0.0, 1.0) ** 2
    comp["heading_hold"] = -getattr(cfg, "w_heading_hold", 0.0) * heading_cost * no_yaw_linear * moving_gate * gate

    # 抬脚高度带宽：只在摆动脚上取均值，不按足端原始求和。
    swing = (inp.foot_contact <= 0.5).to(f)
    target = torch.where(inp.local_obstacle_h[:, None] > 0,
                         inp.local_obstacle_h[:, None] + cfg.terrain_clearance_margin,
                         torch.full_like(inp.foot_clearance, cfg.flat_clearance_target))
    under = torch.clamp((target - inp.foot_clearance - cfg.clearance_band) / 0.05, min=0.0)
    over = torch.clamp((inp.foot_clearance - target - cfg.clearance_band) / 0.10, min=0.0)
    comp["clearance_under"] = -cfg.w_clearance_under * (under ** 2 * swing).mean(dim=-1)
    comp["clearance_over"] = -cfg.w_clearance_over * (over ** 2 * swing).mean(dim=-1)

    # 步态结构：只在运动时生效。单纯速度跟踪可能被爬行/拖脚捷径满足，
    # 这些项把对角小跑接触节律、前后支撑和左右平衡显式交给 PPO。
    stance = (inp.foot_contact > 0.5).to(f)
    desired = getattr(inp, "desired_foot_contact", None)
    if desired is not None:
        desired = desired.to(dtype=f, device=stance.device)
        gait_mismatch = (stance - desired).abs().mean(dim=-1)
        gait_match = 1.0 - gait_mismatch
        # 正向步态锚点不走 quality_gate 渐入；从一开始就奖励符合 gait clock 的接触节律，
        # 使干净小跑成为吸引子，而不是只靠 mismatch 惩罚。
        comp["gait_anchor"] = (
            getattr(cfg, "w_gait_anchor", 0.0)
            * torch.clamp(2.0 * gait_match - 1.0, min=0.0)
            * progress_style_gate
            * moving_gate
            * gate
        )
    else:
        comp["gait_anchor"] = torch.zeros_like(gate)
    diagonal_pair_window = getattr(inp, "diagonal_pair_window", None)
    duty_quality_window = getattr(inp, "duty_quality_window", None)
    if diagonal_pair_window is not None and duty_quality_window is not None:
        diagonal_pair_window = diagonal_pair_window.to(dtype=f, device=gate.device)
        duty_quality_window = duty_quality_window.to(dtype=f, device=gate.device)
        comp["diagonal_contact"] = (
            -getattr(cfg, "w_diagonal_contact", 0.0)
            * torch.clamp(1.0 - diagonal_pair_window, 0.0, 1.0)
            * moving_gate
            * gate
            * quality_gate
        )
        comp["duty_balance"] = (
            -getattr(cfg, "w_duty_balance", 0.0)
            * (torch.clamp(1.0 - duty_quality_window, 0.0, 1.0) ** 2)
            * moving_gate
            * gate
            * quality_gate
        )
    elif stance.shape[-1] >= 4:
        fl, fr, rl, rr = stance[:, 0], stance[:, 1], stance[:, 2], stance[:, 3]
        diagonal_mismatch = 0.5 * ((fl - rr).abs() + (fr - rl).abs())
        front_duty = 0.5 * (fl + fr)
        rear_duty = 0.5 * (rl + rr)
        left_duty = 0.5 * (fl + rl)
        right_duty = 0.5 * (fr + rr)
        rear_bias = torch.clamp(rear_duty - front_duty - getattr(cfg, "rear_duty_tolerance", 0.05), min=0.0)
        lr_bias = (left_duty - right_duty).abs()
        comp["diagonal_contact"] = -getattr(cfg, "w_diagonal_contact", 0.0) * diagonal_mismatch * moving_gate * gate * quality_gate
        comp["duty_balance"] = -getattr(cfg, "w_duty_balance", 0.0) * (rear_bias ** 2 + lr_bias ** 2) * moving_gate * gate * quality_gate
    else:
        comp["diagonal_contact"] = torch.zeros_like(gate)
        comp["duty_balance"] = torch.zeros_like(gate)

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
    td_mask_air = getattr(inp, "touchdown_mask", None)
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
    impact_over = torch.clamp((inp.touchdown_vz.abs() - 0.10) / impact_scale, 0.0, 1.0)
    td_mask = getattr(inp, "touchdown_mask", None)
    impact_w = getattr(cfg, "w_landing_impact", 1.0) + quality_gate * getattr(cfg, "w_landing_impact_late", 0.0)
    if td_mask is not None:
        comp["landing_impact"] = -impact_w * (impact_over * td_mask).mean(dim=-1)
    else:
        comp["landing_impact"] = -impact_w * impact_over.mean(dim=-1)
    # 落脚冲击尾部惩罚：关注最重的一只脚，避免均值掩盖单脚重砸。
    if getattr(cfg, "w_landing_impact_tail", 0.0) != 0.0 or getattr(cfg, "w_landing_impact_tail_late", 0.0) != 0.0:
        tail_w = getattr(cfg, "w_landing_impact_tail", 0.0) + quality_gate * getattr(cfg, "w_landing_impact_tail_late", 0.0)
        worst = (impact_over * td_mask).max(dim=-1).values if td_mask is not None else impact_over.max(dim=-1).values
        comp["landing_impact"] = comp["landing_impact"] - tail_w * worst

    # 触地滑移：与竖直落脚冲击对应，但作用在触地瞬间的足端水平速度。
    # 它补上支撑相滑移不覆盖触地帧的盲区，抑制脚前勾和侧向挂脚。
    # 只看足端世界速度，不直接惩罚机身速度，因此不压制正常推进。
    ts_w = getattr(cfg, "w_touchdown_slip", 0.0) + quality_gate * getattr(cfg, "w_touchdown_slip_late", 0.0)
    if (getattr(cfg, "w_touchdown_slip", 0.0) != 0.0 or getattr(cfg, "w_touchdown_slip_late", 0.0) != 0.0):
        ts_over = torch.clamp((inp.foot_vel_xy - 0.05) / impact_scale, 0.0, 1.0)      # 每只脚超过水平底噪的部分。
        comp["touchdown_slip"] = -ts_w * ((ts_over * td_mask).mean(dim=-1) if td_mask is not None
                                          else ts_over.mean(dim=-1))

    # 平地步态姿态和速率质量：机身保持水平，减少弹跳、滚转/俯仰抖动和髋关节外翻。
    # 所有项有界，并由稳定门控避免在 collapse 上叠加无意义惩罚。
    tilt_over = torch.clamp(
        (inp.tilt_rel - getattr(cfg, "orient_soft_rad", 0.09)) / max(getattr(cfg, "orient_scale_rad", 0.35), 1e-6),
        0.0, 1.0)
    comp["orient"] = -getattr(cfg, "w_orient", 0.0) * tilt_over * gate
    # 与 orient 一样受稳定门控控制：collapse 时关闭，正常步态下近似保持原有 shaping。
    comp["base_vz"] = -getattr(cfg, "w_base_vz", 0.0) * torch.clamp(vel[:, 2] ** 2, max=1.0) * gate
    wxy_sq = inp.base_ang_vel[:, 0] ** 2 + inp.base_ang_vel[:, 1] ** 2
    comp["base_wxy"] = -getattr(cfg, "w_base_wxy", 0.0) * torch.clamp(wxy_sq, max=4.0) * gate
    # 髋关节偏差：按命令门控。前进、后退和站立时要求髋关节接近默认位置；
    # 横移和 yaw 本身可能需要髋外展，因此不在这些命令下强行压回默认。
    hip_dev = getattr(inp, "hip_deviation", None)
    if hip_dev is not None:
        hip_dev = hip_dev.to(dtype=f, device=gate.device)
        no_lat_yaw = ((cmd[:, 1].abs() <= 0.05) & (cmd[:, 2].abs() <= 0.05)).to(f)
        comp["hip_deviation"] = -getattr(cfg, "w_hip_deviation", 0.0) * torch.clamp(
            hip_dev / max(getattr(cfg, "hip_deviation_scale", 0.5), 1e-6), 0.0, 1.0) * no_lat_yaw * gate
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

    # 终止惩罚：只惩罚真实 terminal，不惩罚 timeout。
    is_terminal = 1.0 if (inp.terminal_reason is not None and inp.terminal_reason != "timeout") else 0.0
    comp["terminal_penalty"] = -cfg.w_terminal * torch.full_like(gate, is_terminal)

    total = torch.zeros_like(gate)
    for k, v in comp.items():
        if k not in (
            "stable_motion_gate",
            "stand_gate",
            "moving_gate",
            "quality_gate",
            "healthy_progress_gate",
            "validated_tracking_gate",
            "touchdown_clean_gate",
            "progress_style_gate",
        ):
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
REWARD_GROUP_GATES = (
    "stable_motion_gate", "stand_gate", "moving_gate", "quality_gate",
    "healthy_progress_gate", "validated_tracking_gate", "touchdown_clean_gate",
    "progress_style_gate", "total",
)
REWARD_GROUP_FALLBACK = "stab"
_REWARD_GROUP_OF = {
    # 任务/命令跟踪，主要是正向任务奖励。
    "tracking_lin": "track",
    "tracking_yaw": "turn", "tracking_yaw_far": "turn",   # yaw 拥有独立归一化优势。
    "tracking_lin_far": "track",
    "terrain_progress": "track", "stand": "track", "stand_contact": "track",
    "stand_far": "track", "climb": "track", "terrain_up": "track",
    "terrain_down": "track", "feet_air_time": "track", "linear_underspeed": "track",
    # 步态质量：节律、接触时序、滑移和抬脚高度带宽。
    "gait_anchor": "gait", "diagonal_contact": "gait", "duty_balance": "gait",
    "stance_slip": "gait", "clearance_under": "gait", "clearance_over": "gait",
    "terrain_support_transfer": "gait",
    # 稳定性、姿态和安全。
    "orient": "stab", "base_vz": "stab", "base_wxy": "stab", "off_axis": "stab",
    "planar_purity": "stab", "heading_hold": "stab",
    "wrong_dir": "stab", "lateral_underspeed": "stab", "hip_deviation": "stab", "landing_impact": "stab",
    "lateral_foot_excursion": "stab",
    "terrain_contact_quality": "stab", "terrain_event_collapse": "stab",
    "stumble": "stab", "terminal_penalty": "stab",
    # 风格、力矩和动作平滑。
    "torque_margin": "style", "torque_saturation": "style", "action_rate": "style",
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
