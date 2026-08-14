"""Taili 地形奖励和课程使用的纯张量计算。

环境负责地形生成与 reset；本模块只描述可以单独测试的物理语义。楼梯能力
以承重足所在真实地面高度的连续变化表达，不再使用先导足、阶段、锁存或超时状态机。
"""
from __future__ import annotations

import math
from typing import Any

import torch

from . import taili_geometry as geometry


def classify_support_contact(
    contact_force_w: torch.Tensor,
    support_normal_w: torch.Tensor,
    *,
    normal_force_min: float = 8.0,
    normal_ratio_min: float = 0.35,
) -> dict[str, torch.Tensor]:
    """区分承重接触与立面碰撞。

    只有沿当前支撑面法向具有足够分量的接触才算承重。立面碰撞可以作为碰障后的
    本体响应，但不能伪装成落脚或支撑高度变化。
    """
    force = contact_force_w
    normal = torch.nn.functional.normalize(
        support_normal_w.to(dtype=force.dtype, device=force.device),
        dim=-1,
    )
    signed_normal_force = (force * normal[:, None, :]).sum(dim=-1)
    normal_force = signed_normal_force.abs()
    force_norm = torch.linalg.norm(force, dim=-1)
    normal_ratio = normal_force / force_norm.clamp(min=1e-6)
    contact = (
        (normal_force >= float(normal_force_min))
        & (normal_ratio >= float(normal_ratio_min))
    )
    return {
        "contact": contact,
        "normal_force": normal_force,
        "normal_ratio": torch.clamp(normal_ratio, 0.0, 1.0),
        "force_norm": force_norm,
    }


def unexpected_body_contact_score(
    contact_force_w: torch.Tensor,
    *,
    force_start: float = 5.0,
    force_span: float = 45.0,
) -> torch.Tensor:
    """把非足端刚体接触力映射为连续碰障强度。

    hip、thigh、calf 和 base 在正常步态中不应承重；它们一旦接触地形，就是
    已经发生的被动碰障证据。这里只读取当前接触力，不使用地形类型、扫描高度
    或任何锁存状态。
    """
    force = torch.as_tensor(contact_force_w)
    magnitude = torch.linalg.norm(force, dim=-1)
    start = max(float(force_start), 0.0)
    span = max(float(force_span), 1e-6)
    return torch.clamp((magnitude - start) / span, 0.0, 1.0)


def command_leading_leg_weights(command_xy: torch.Tensor) -> torch.Tensor:
    """按当前平移命令把机身碰撞分配给迎障腿，顺序为 FL/FR/RL/RR。"""
    command = torch.as_tensor(command_xy)
    if command.shape[-1] != 2:
        raise ValueError(f"command_xy must end in 2 values, got {tuple(command.shape)}")
    vx, vy = command[..., 0], command[..., 1]
    denom = (vx.abs() + vy.abs()).clamp(min=1e-6)
    front = torch.clamp(vx, min=0.0) / denom
    rear = torch.clamp(-vx, min=0.0) / denom
    left = torch.clamp(vy, min=0.0) / denom
    right = torch.clamp(-vy, min=0.0) / denom
    moving = ((vx.abs() + vy.abs()) > 1e-6).to(command.dtype)
    return torch.stack(
        [front + left, front + right, rear + left, rear + right],
        dim=-1,
    ).clamp(0.0, 1.0) * moving[..., None]


def map_body_collision_to_legs(
    *,
    limb_collision_score: torch.Tensor,
    base_collision_score: torch.Tensor,
    command_xy: torch.Tensor,
) -> torch.Tensor:
    """合并逐腿碰撞与按命令方向分配的机身碰撞。"""
    limb = torch.as_tensor(limb_collision_score)
    if limb.shape[-1] != 4:
        raise ValueError(
            f"limb_collision_score must end in 4 legs, got {tuple(limb.shape)}"
        )
    base = torch.as_tensor(
        base_collision_score,
        dtype=limb.dtype,
        device=limb.device,
    )
    weights = command_leading_leg_weights(
        torch.as_tensor(command_xy, dtype=limb.dtype, device=limb.device)
    )
    return torch.maximum(
        torch.clamp(limb, 0.0, 1.0),
        torch.clamp(base, 0.0, 1.0)[..., None] * weights,
    )


def terrain_recovery_termination(
    *,
    base_height_local: torch.Tensor,
    projected_gravity_z: torch.Tensor,
    finite: torch.Tensor,
    min_height: float = 0.18,
    max_tilt_deg: float = 60.0,
) -> torch.Tensor:
    """离散地形恢复空间的无状态终止条件。"""
    height = torch.as_tensor(base_height_local)
    gravity_z = torch.as_tensor(
        projected_gravity_z,
        dtype=height.dtype,
        device=height.device,
    )
    finite_mask = torch.as_tensor(finite, device=height.device).bool()
    tilt_deg = min(max(float(max_tilt_deg), 0.0), 180.0)
    tilt_limit_z = -math.cos(math.radians(tilt_deg))
    return (
        (~finite_mask)
        | (height < float(min_height))
        | (gravity_z > tilt_limit_z)
    )


def loaded_support_height(
    *,
    ground_z: torch.Tensor,
    support_contact: torch.Tensor,
    support_force: torch.Tensor,
    fallback_height: torch.Tensor | None = None,
    force_floor: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """计算法向承重力加权的支撑地面高度。

    非接触足、未承重足和机身高度都不参与，因此跳起、伸腿或立面碰撞不能制造
    地形进展。没有有效承重足时返回 fallback，同时把 ``valid`` 置为假。
    """
    dtype = ground_z.dtype
    contact = support_contact.bool()
    force = torch.clamp(
        support_force.to(dtype=dtype, device=ground_z.device),
        min=0.0,
    ) * contact.to(dtype)
    force_sum = force.sum(dim=1)
    valid = force_sum > float(force_floor)
    weights = force / force_sum[:, None].clamp(min=float(force_floor))
    observed = (ground_z * weights).sum(dim=1)
    if fallback_height is None:
        fallback = ground_z.median(dim=1).values
    else:
        fallback = fallback_height.to(dtype=dtype, device=ground_z.device)
    height = torch.where(valid, observed, fallback)
    centered = ground_z - height[:, None]
    dispersion = torch.sqrt(
        torch.clamp((weights * centered.square()).sum(dim=1), min=0.0)
    )
    loaded_count = (force > float(force_floor)).sum(dim=1)
    return {
        "height": height,
        "valid": valid,
        "weights": weights,
        "dispersion": torch.where(valid, dispersion, torch.zeros_like(dispersion)),
        "force_sum": force_sum,
        "loaded_count": loaded_count,
    }


def filter_loaded_support_height(
    *,
    observed_height: torch.Tensor,
    previous_height: torch.Tensor,
    valid: torch.Tensor,
    alpha: float = 0.20,
) -> torch.Tensor:
    """低通承重比例快速重排造成的高度抖动。

    这是固定系数的物理量估计器，不识别地形类型，也没有阶段、锁存或超时。
    无有效承重时保持上一值，避免腾空帧用 fallback 制造虚假高度变化。
    """
    blend = min(max(float(alpha), 0.0), 1.0)
    filtered = previous_height + blend * (observed_height - previous_height)
    return torch.where(valid.bool(), filtered, previous_height)


def continuous_terrain_contact_response(
    *,
    ground_z: torch.Tensor,
    support_contact: torch.Tensor,
    support_force: torch.Tensor,
    current_height: torch.Tensor,
    previous_height: torch.Tensor,
    landing_mask: torch.Tensor | None = None,
    collision_score_by_foot: torch.Tensor | None = None,
    height_scale: float = 0.04,
    delta_scale: float = 0.025,
    height_deadband: float = 0.008,
    delta_deadband: float = 0.008,
) -> dict[str, torch.Tensor]:
    """由当前接触后的物理量生成连续地形响应。

    该响应没有阶段记忆。承重足位于不同高度、支撑高度真实变化或当前足端发生
    水平主导碰撞时才增大；平地作用域由环境在外层屏蔽。
    """
    dtype = ground_z.dtype
    h_scale = max(float(height_scale), 1e-6)
    d_scale = max(float(delta_scale), 1e-6)
    h_deadband = max(float(height_deadband), 0.0)
    d_deadband = max(float(delta_deadband), 0.0)
    contact = support_contact.to(dtype=dtype, device=ground_z.device)
    force = torch.clamp(support_force.to(dtype=dtype, device=ground_z.device), min=0.0)
    force = force * contact
    force_sum = force.sum(dim=1, keepdim=True)
    weights = force / force_sum.clamp(min=1e-6)

    support_offset = torch.abs(ground_z - current_height[:, None])
    support_offset_excess = torch.clamp(support_offset - h_deadband, min=0.0)
    support_leg = torch.clamp(support_offset_excess / h_scale, 0.0, 1.0) * contact
    dispersion = torch.sqrt(torch.clamp((weights * support_offset.square()).sum(dim=1), min=0.0))
    dispersion_response = torch.clamp((dispersion - h_deadband) / h_scale, 0.0, 1.0)
    delta = current_height - previous_height
    delta_response = torch.clamp((delta.abs() - d_deadband) / d_scale, 0.0, 1.0)

    if landing_mask is None:
        landing_leg = torch.zeros_like(ground_z)
    else:
        landing = landing_mask.to(dtype=dtype, device=ground_z.device)
        landing_leg = torch.clamp(
            (
                torch.abs(ground_z - previous_height[:, None])
                - h_deadband
            ) / h_scale,
            0.0,
            1.0,
        ) * landing
    if collision_score_by_foot is None:
        collision_leg = torch.zeros_like(ground_z)
    else:
        collision_leg = torch.clamp(
            collision_score_by_foot.to(dtype=dtype, device=ground_z.device),
            0.0,
            1.0,
        )

    leg_response = torch.maximum(torch.maximum(support_leg, landing_leg), collision_leg)
    response = torch.maximum(
        torch.maximum(dispersion_response, delta_response),
        leg_response.max(dim=1).values,
    )
    return {
        "response": torch.clamp(response, 0.0, 1.0),
        "leg_response": torch.clamp(leg_response, 0.0, 1.0),
        "support_delta": delta,
        "dispersion": dispersion,
        "collision_response": collision_leg.max(dim=1).values,
    }


def update_terrain_collision_trace(
    *,
    current_response: torch.Tensor,
    previous_trace: torch.Tensor,
    dt: float,
    decay_time: float,
    active: torch.Tensor | None = None,
) -> torch.Tensor:
    """平滑保存最近一次真实碰障，覆盖策略产生反应所需的短暂延迟。

    该量只由已经发生的接触更新，不读取前方地形，也不编码足序或动作阶段。
    作用域失效时立即清零，避免跨命令或跨地形残留。
    """
    tau = max(float(decay_time), 1e-6)
    decay = math.exp(-max(float(dt), 0.0) / tau)
    current = torch.clamp(
        current_response.to(dtype=previous_trace.dtype, device=previous_trace.device),
        0.0,
        1.0,
    )
    trace = torch.maximum(current, previous_trace * decay)
    if active is not None:
        scope = active.to(device=trace.device).bool()
        while scope.ndim < trace.ndim:
            scope = scope.unsqueeze(-1)
        trace = torch.where(scope, trace, torch.zeros_like(trace))
    return torch.clamp(trace, 0.0, 1.0)


def terrain_parameter_upper_bound(
    level: torch.Tensor,
    *,
    num_rows: int,
    value_min: float,
    value_max: float,
) -> torch.Tensor:
    """返回当前课程行可能生成的参数上界。

    IsaacLab 在第 ``level`` 行使用 ``(level + U[0, 1)) / num_rows`` 采样
    difficulty。奖励侧取该区间上界，既覆盖本行真实台阶，又不会在最低行直接要求
    最高难度的抬脚高度。
    """
    rows = max(int(num_rows), 1)
    lo = float(value_min)
    hi = max(float(value_max), lo)
    difficulty_upper = torch.clamp(
        (torch.as_tensor(level).to(dtype=torch.float32) + 1.0) / float(rows),
        0.0,
        1.0,
    )
    return lo + difficulty_upper * (hi - lo)


def terrain_motion_credit(
    *,
    v_along: torch.Tensor,
    command_speed: torch.Tensor,
    progress_full_ratio: float = 0.35,
    overspeed_start_ratio: float = 1.15,
    overspeed_span_ratio: float = 0.35,
) -> dict[str, torch.Tensor]:
    """把沿命令方向速度拆成慢速推进信用和非饱和超速代价。"""
    speed = torch.as_tensor(command_speed, dtype=v_along.dtype, device=v_along.device)
    valid = speed > 1e-6
    ratio = torch.where(valid, v_along / speed.clamp(min=1e-6), torch.zeros_like(v_along))
    progress = torch.clamp(
        ratio / max(float(progress_full_ratio), 1e-6),
        0.0,
        1.0,
    )
    overspeed_x = torch.clamp(
        (ratio - float(overspeed_start_ratio))
        / max(float(overspeed_span_ratio), 1e-6),
        min=0.0,
    )
    overspeed = torch.where(
        overspeed_x <= 1.0,
        0.5 * overspeed_x.square(),
        overspeed_x - 0.5,
    )
    return {
        "ratio": torch.where(valid, ratio, torch.zeros_like(ratio)),
        "progress": torch.where(valid, progress, torch.zeros_like(progress)),
        "overspeed": torch.where(valid, overspeed, torch.zeros_like(overspeed)),
    }


def update_contact_support_reference(
    *,
    foot_position_w: torch.Tensor,
    contact: torch.Tensor,
    support_force: torch.Tensor,
    previous_point_w: torch.Tensor,
    previous_normal_w: torch.Tensor,
    alpha: float = 0.10,
    foot_radius: float = geometry.FOOT_RADIUS,
    min_plane_contacts: int = 3,
) -> dict[str, torch.Tensor]:
    """仅用已经承重的足端更新支撑面参考。

    两足支撑时只更新支撑点并保持旧法向；三足及以上才重新拟合法向。该参考只服务
    奖励、课程和遥测，不进入 Actor，也不读取未接触区域。
    """
    dtype = foot_position_w.dtype
    contact_b = contact.bool()
    force = torch.clamp(support_force.to(dtype=dtype), min=0.0) * contact_b.to(dtype)
    force_sum = force.sum(dim=1, keepdim=True)
    weight = force / force_sum.clamp(min=1e-6)
    contact_count = contact_b.sum(dim=1)

    world_up = torch.zeros_like(previous_normal_w)
    world_up[:, 2] = 1.0
    sole_position = foot_position_w - float(foot_radius) * world_up[:, None, :]
    observed_point = (sole_position * weight[:, :, None]).sum(dim=1)
    point_valid = force_sum[:, 0] > 1e-6
    observed_point = torch.where(point_valid[:, None], observed_point, previous_point_w)

    centered = sole_position - observed_point[:, None, :]
    covariance = torch.einsum("ni,nij,nik->njk", weight, centered, centered)
    covariance = covariance + 1e-8 * torch.eye(
        3, dtype=dtype, device=foot_position_w.device
    )[None, :, :]
    _, eigenvectors = torch.linalg.eigh(covariance)
    observed_normal = eigenvectors[:, :, 0]
    observed_normal = torch.where(
        (observed_normal * world_up).sum(dim=-1, keepdim=True) < 0.0,
        -observed_normal,
        observed_normal,
    )
    plane_valid = (contact_count >= int(min_plane_contacts)) & (observed_normal[:, 2] >= 0.766)
    observed_normal = torch.where(plane_valid[:, None], observed_normal, previous_normal_w)

    blend = min(max(float(alpha), 0.0), 1.0)
    point = torch.where(
        point_valid[:, None],
        (1.0 - blend) * previous_point_w + blend * observed_point,
        previous_point_w,
    )
    normal = torch.nn.functional.normalize(
        (1.0 - blend) * previous_normal_w + blend * observed_normal,
        dim=-1,
    )
    return {
        "point_w": point,
        "normal_w": normal,
        "point_valid": point_valid,
        "plane_valid": plane_valid,
        "contact_count": contact_count,
    }


def scope_terrain_response(raw_response: torch.Tensor, flat_mask: torch.Tensor) -> torch.Tensor:
    """平地始终使用严格动作约束，物理地形响应只在非平地样本中生效。"""
    return torch.clamp(raw_response, 0.0, 1.0) * (~flat_mask.bool()).to(raw_response.dtype)


def compute_terrain_curriculum_moves(
    *,
    cmd_mag: torch.Tensor,
    forward_dist: torch.Tensor,
    support_height_delta: torch.Tensor,
    expected_height_direction: torch.Tensor,
    support_height_valid: torch.Tensor,
    valid_episode: torch.Tensor,
    terrain_curriculum_active: bool,
    terrain_unlocked: bool,
    terminal_now: torch.Tensor,
    base_h_local: torch.Tensor,
    upright_score: torch.Tensor,
    contact_count: torch.Tensor,
    body_wxy: torch.Tensor,
    v_along: torch.Tensor,
    max_episode_length_s: float,
    terrain_move_up_dist: float,
    stair_height_min: float = 0.08,
    stair_forward_min: float = 0.75,
    stable_h: float = 0.42,
    stable_upright: float = 0.85,
    stable_contact_min: float = 2.0,
    stable_wxy_max: float = 1.50,
    speed_cap_ratio: float = 1.60,
    speed_cap_min: float = 0.75,
    failure_h: float = 0.42,
    failure_upright: float = 0.75,
    failure_wxy: float = 2.00,
    eligible_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """按整段真实结果决定地形升降级。

    楼梯必须同时满足承重高度净变化、沿命令方向位移、稳定终点和受控速度；其他
    地形沿用稳定穿越距离。root z、最大地形等级和单次碰撞都不能代替楼梯成功。
    """
    zeros = torch.zeros_like(valid_episode, dtype=torch.bool)
    direction = torch.sign(
        expected_height_direction.to(
            dtype=support_height_delta.dtype,
            device=support_height_delta.device,
        )
    )
    requires_height = direction.ne(0.0)
    evaluation_eligible = ~requires_height | (cmd_mag > 0.10)
    signed_height = direction * support_height_delta
    height_ok = (
        requires_height
        & support_height_valid.bool()
        & (signed_height >= float(stair_height_min))
    )

    if not terrain_curriculum_active:
        return {
            "move_up": zeros,
            "low_progress_down": zeros,
            "failure_down": zeros,
            "move_down": zeros,
            "controlled_up": zeros,
            "controlled_down": zeros,
            "distance_success": zeros,
            "stable_end": zeros,
            "speed_controlled": zeros,
            "catastrophic": zeros,
            "height_ok": height_ok,
            "requires_height": requires_height,
            "signed_height": signed_height,
            "evaluation_eligible": evaluation_eligible,
        }

    # 站立和纯旋转没有跨层任务，不能在楼梯成功率分母中记作失败，也不能因此降级。
    valid = valid_episode.bool() & evaluation_eligible
    if eligible_mask is not None:
        valid = valid & eligible_mask.bool()
    terminal = terminal_now.bool()
    stable_end = (
        valid
        & ~terminal
        & (base_h_local >= float(stable_h))
        & (upright_score >= float(stable_upright))
        & (contact_count >= float(stable_contact_min))
        & (body_wxy <= float(stable_wxy_max))
    )
    speed_limit = torch.maximum(
        cmd_mag * float(speed_cap_ratio),
        torch.full_like(cmd_mag, float(speed_cap_min)),
    )
    speed_controlled = v_along.abs() <= speed_limit
    stair_forward_ok = forward_dist >= float(stair_forward_min)
    controlled_height = height_ok & stair_forward_ok & stable_end & speed_controlled
    controlled_up = controlled_height & (direction > 0.0)
    controlled_down = controlled_height & (direction < 0.0)
    distance_success = (
        ~requires_height
        & (cmd_mag > 0.10)
        & (forward_dist >= float(terrain_move_up_dist))
        & stable_end
        & speed_controlled
    )
    move_up = (
        distance_success | controlled_up | controlled_down
    ) & valid & bool(terrain_unlocked)

    catastrophic = (
        terminal
        | (base_h_local < float(failure_h))
        | (upright_score < float(failure_upright))
        | (body_wxy > float(failure_wxy))
    ) & valid
    failure_down = catastrophic

    expected_dist = cmd_mag * float(max_episode_length_s) * 0.5
    ordinary_low_progress = ~requires_height & (forward_dist < expected_dist)
    stair_incomplete = requires_height & ~controlled_height
    low_progress_down = (
        (ordinary_low_progress | stair_incomplete)
        & ~move_up
        & valid
        & ~catastrophic
    )
    move_down = (low_progress_down | failure_down) & ~move_up
    return {
        "move_up": move_up,
        "low_progress_down": low_progress_down,
        "failure_down": failure_down,
        "move_down": move_down,
        "controlled_up": controlled_up,
        "controlled_down": controlled_down,
        "distance_success": distance_success,
        "stable_end": stable_end,
        "speed_controlled": speed_controlled,
        "catastrophic": catastrophic,
        "height_ok": height_ok,
        "requires_height": requires_height,
        "signed_height": signed_height,
        "evaluation_eligible": evaluation_eligible,
    }
