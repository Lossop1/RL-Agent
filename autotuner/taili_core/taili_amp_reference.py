"""AMP online reference generator — strict per taili_strategy_decisions.md
("AMP Online Reference Generator Contract v1" + D).

Analytic, command-conditioned FLAT style reference. Differences from the old
parametric_ref (which is reference-only):
  * nominal pose = q_default = [0, 0.7, -1.4] per leg (NOT old 0.6/-1.2);
    base_height_ref is DERIVED by FK(q_default) (NOT a hardcoded BASE_Z).
  * NO roughness / terrain input — terrain adaptation belongs to TerrainPerceiver +
    terrain reward + curriculum. Feeding terrain here = implicit privileged shaping.
  * outputs frame51 = motion43 + command3 + mode_onehot5 (mode derived from command).

Kinematic constants are URDF-derived (same as gen_taili_gaits / parametric_ref). Pure torch.
"""
from __future__ import annotations

import math

import torch

from . import taili_geometry as geometry
from . import taili_symmetry as sym

# ── URDF kinematic constants ─────────────────────────────────────────────────
L1 = 0.36385                               # thigh length
FOOT_OFF = (0.0252765, 0.0, -0.3179635)    # calf->foot offset (sagittal x,z used)
HIPX, HIPY = 0.30414, 0.065                # hip joint offset from base (x, y)
THIGHY = 0.1432                            # hip-abduction-axis -> thigh sagittal y offset
LEGS = ["FL", "FR", "RL", "RR"]
SGN = {"FL": (1, 1), "FR": (1, -1), "RL": (-1, 1), "RR": (-1, -1)}   # (x,y) sign per leg
TROT = {"FL": 0.0, "FR": 0.5, "RL": 0.5, "RR": 0.0}                  # diagonal trot offsets

# ── nominal pose = q_default (Runtime IO / asset; NOT the old 0.6/-1.2) ───────
Q_DEFAULT_THIGH = 0.7
Q_DEFAULT_CALF = -1.4
Q_DEFAULT_HIP = 0.0
# canonical 12-vector [hip x4, thigh x4, calf x4] legs [FL,FR,RL,RR]
Q_DEFAULT12 = [Q_DEFAULT_HIP] * 4 + [Q_DEFAULT_THIGH] * 4 + [Q_DEFAULT_CALF] * 4


def _fk2(th_t, th_c):
    """2-link sagittal FK: thigh th_t, calf th_c -> foot (x,z) rel hip."""
    s_t, c_t = torch.sin(th_t), torch.cos(th_t)
    s2, c2 = torch.sin(th_t + th_c), torch.cos(th_t + th_c)
    ox, oz = FOOT_OFF[0], FOOT_OFF[2]
    fx = -L1 * s_t + c2 * ox + s2 * oz
    fz = -L1 * c_t - s2 * ox + c2 * oz
    return fx, fz


def _ik2(tx, tz, iters=40):
    """2-link sagittal IK (vectorized Newton) for foot (tx,tz) rel hip -> (th_t, th_c).
    Initialized at q_default so the stand target returns q_default exactly."""
    th_t = torch.full_like(tx, Q_DEFAULT_THIGH)
    th_c = torch.full_like(tx, Q_DEFAULT_CALF)
    ox, oz = FOOT_OFF[0], FOOT_OFF[2]
    for _ in range(iters):
        s_t, c_t = torch.sin(th_t), torch.cos(th_t)
        s2, c2 = torch.sin(th_t + th_c), torch.cos(th_t + th_c)
        fx = -L1 * s_t + c2 * ox + s2 * oz
        fz = -L1 * c_t - s2 * ox + c2 * oz
        ex, ez = tx - fx, tz - fz
        j11 = -L1 * c_t - s2 * ox + c2 * oz
        j12 = -s2 * ox + c2 * oz
        j21 = L1 * s_t - c2 * ox - s2 * oz
        j22 = -c2 * ox - s2 * oz
        det = (j11 * j22 - j12 * j21).clamp_min(1e-9)
        th_t = th_t + (j22 * ex - j12 * ez) / det
        th_c = th_c + (-j21 * ex + j11 * ez) / det
    return th_t, th_c


# nominal foot (x,z) rel hip at q_default -> X0, H0。H0 是足端球心高度，
# base 高度还必须加上 URDF 碰撞球半径。
with torch.no_grad():
    _nx, _nz = _fk2(torch.tensor(Q_DEFAULT_THIGH), torch.tensor(Q_DEFAULT_CALF))
    X0 = float(_nx)
    H0 = -float(_nz)
BASE_HEIGHT_REF = geometry.NOMINAL_BASE_HEIGHT


def apply_flat_stand_reset(
    root_state,
    joint_pos,
    joint_vel,
    commands,
    env_origins,
    default_joint_pos,
    flat_mask,
    *,
    sole_clearance: float = 0.003,
):
    """把平地零命令 RSI 覆盖为静态默认姿态，避免从移动 clip 开始刹车。"""
    stand_mask = (
        flat_mask.bool()
        & (torch.linalg.norm(commands[:, :2], dim=-1) <= 0.05)
        & (commands[:, 2].abs() <= 0.05)
    )
    if not bool(stand_mask.any()):
        return stand_mask

    root_state[stand_mask, :2] = env_origins[stand_mask, :2]
    root_state[stand_mask, 2] = (
        env_origins[stand_mask, 2]
        + float(geometry.NOMINAL_BASE_HEIGHT)
        + float(sole_clearance)
    )
    root_state[stand_mask, 3:7] = 0.0
    root_state[stand_mask, 3] = 1.0
    root_state[stand_mask, 7:13] = 0.0
    joint_pos[stand_mask] = default_joint_pos[stand_mask]
    joint_vel[stand_mask] = 0.0
    return stand_mask


def neutralize_reset_velocities(root_state, joint_vel):
    """保留 RSI 姿态，但清除参考动作携带的免费运动信用。"""
    root_state[:, 7:13] = 0.0
    joint_vel.zero_()


def preserve_failed_command_targets(sampled_targets, previous_targets, failed_mask):
    """真实失败后的下一回合重试原命令，timeout 和正常 reset 仍使用新采样。"""
    mask = failed_mask.to(dtype=torch.bool, device=sampled_targets.device)
    previous = previous_targets.to(
        dtype=sampled_targets.dtype,
        device=sampled_targets.device,
    )
    return torch.where(mask[:, None], previous, sampled_targets)


def _foot_traj(p, sdx, sdy, clearance):
    """trot stance/swing foot trajectory at per-leg phase p in [0,1). Smooth, no-slam, no-scuff."""
    stance = p < 0.5
    s_st = p / 0.5
    s_sw = (p - 0.5) / 0.5
    smooth5 = s_sw ** 3 * (10.0 + s_sw * (-15.0 + 6.0 * s_sw))
    hsw = 2.0 * smooth5 - s_sw
    fx = torch.where(stance, X0 + sdx / 2 - sdx * s_st, X0 - sdx / 2 + sdx * hsw)
    fy = torch.where(stance, sdy / 2 - sdy * s_st, -sdy / 2 + sdy * hsw)
    swing_bump = 64.0 * s_sw ** 3 * (1.0 - s_sw) ** 3
    fz = torch.where(stance, torch.full_like(p, -H0), -H0 + clearance * swing_bump)
    return fx, fy, fz


def period_for_speed(speed, base=0.55, slope=0.075, pmin=0.40):
    return torch.clamp(base - slope * speed, min=pmin, max=base)


def clearance_for_speed(speed, clearance_base, clearance_gain):
    """低速缩短步幅时同步降低平地抬脚高度，避免小步高抬后重落脚。"""
    speed_ratio = torch.clamp(speed / 0.50, 0.0, 1.0)
    base_scale = 0.55 + 0.45 * speed_ratio
    return clearance_base * base_scale + clearance_gain * torch.clamp(speed / 2.0, 0.0, 1.0)


def flat_reference(commands, times, *, gait_period=0.55, gait_period_slope=0.075,
                   gait_period_min=0.40, yaw_speed_equiv=0.15,
                   clearance_base=0.07, clearance_gain=0.03,
                   stance_dx=0.0, iters=40):
    """Analytic trot reference, command-conditioned, terrain-AGNOSTIC.
    commands (M,3) [vx,vy,wz], times (M,). Returns jp,jv,bh,tn,foot_rel."""
    dev = commands.device
    vx, vy, wz = commands[:, 0], commands[:, 1], commands[:, 2]
    speed = torch.norm(commands[:, :2], dim=1)
    # yaw 按足端旋转半径折算成等效线速度，只影响步态周期；步幅仍由刚体足端速度公式决定。
    period_speed = speed + float(yaw_speed_equiv) * wz.abs()
    T = period_for_speed(period_speed, gait_period, gait_period_slope, gait_period_min)
    clearance = clearance_for_speed(period_speed, clearance_base, clearance_gain)
    # NO roughness term (D: AMP terrain-agnostic).
    # gate clearance by command magnitude -> stand command yields a STATIC planted stance.
    cmd_mag = torch.norm(commands, dim=1)
    clearance = clearance * torch.clamp(cmd_mag / 0.1, 0.0, 1.0)

    def joints_at(tt):
        hip, thigh, calf, foot = [], [], [], []
        for lg in LEGS:
            sx, sy = SGN[lg]
            hx, hy = sx * HIPX, sy * HIPY
            dx = stance_dx if lg in ("FL", "FR") else -stance_dx
            foot_x0 = hx + X0 + dx
            foot_y0 = hy + sy * THIGHY
            sdx = (vx - wz * foot_y0) * T / 2.0
            sdy = (vy + wz * foot_x0) * T / 2.0
            p = ((tt / T) + TROT[lg]) % 1.0
            fx, fy, fz = _foot_traj(p, sdx, sdy, clearance)
            fx = fx + dx
            th_h = torch.atan2(fy, -fz)
            r = torch.hypot(fy, fz)
            th_t, th_c = _ik2(fx, -r, iters)
            hip.append(th_h); thigh.append(th_t); calf.append(th_c)
            fy_b = hy + fy + sy * THIGHY * torch.cos(th_h)
            fz_b = fz + sy * THIGHY * torch.sin(th_h)
            foot.append(torch.stack([hx + fx, fy_b, fz_b], dim=-1))
        jp = torch.stack(hip + thigh + calf, dim=-1)       # (M,12) canonical dof order
        foot_rel = torch.stack(foot, dim=1)                # (M,4,3)
        return jp, foot_rel

    jp, foot_rel = joints_at(times)
    dt = 1.0 / 50.0
    jp2, _ = joints_at(times + dt)
    jv = (jp2 - jp) / dt
    M = commands.shape[0]
    bh = torch.full((M, 1), BASE_HEIGHT_REF, device=dev, dtype=commands.dtype)
    yaw = wz * times
    tn = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw),
                      torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)], dim=-1)
    return jp, jv, bh, tn, foot_rel


def motion43(commands, times, **kw):
    jp, jv, bh, tn, foot_rel = flat_reference(commands, times, **kw)
    return torch.cat([jp, jv, bh, tn, foot_rel.reshape(foot_rel.shape[0], 12)], dim=-1)


def mode_onehot(commands, c_move=0.1, w_move=0.1):
    """Single-axis mode from command: 0 stand, 1 fwd, 2 back, 3 lat, 4 yaw."""
    vx, vy, wz = commands[:, 0], commands[:, 1], commands[:, 2]
    spd_xy = torch.norm(commands[:, :2], dim=1)
    stand = (spd_xy <= c_move) & (wz.abs() <= w_move)
    idx = torch.zeros(commands.shape[0], dtype=torch.long, device=commands.device)
    idx = torch.where(vx > c_move, torch.full_like(idx, 1), idx)
    idx = torch.where(vx < -c_move, torch.full_like(idx, 2), idx)
    idx = torch.where(vy.abs() > c_move, torch.full_like(idx, 3), idx)
    idx = torch.where(wz.abs() > w_move, torch.full_like(idx, 4), idx)
    idx = torch.where(stand, torch.zeros_like(idx), idx)
    return torch.nn.functional.one_hot(idx, num_classes=5).to(commands.dtype)


def conditioned_frame51(motion, commands):
    """把 motion43 与可部署命令条件组装成判别器单帧。"""
    if motion.shape[-1] != 43:
        raise ValueError(f"AMP motion width must be 43, got {motion.shape[-1]}")
    if commands.shape[-1] != 3 or motion.shape[:-1] != commands.shape[:-1]:
        raise ValueError("AMP motion and command batch shapes do not match")
    return torch.cat([motion, commands, mode_onehot(commands)], dim=-1)


def frame51(commands, times, **kw):
    """frame51 = motion43 + command3 + mode_onehot5 (for the conditional discriminator)."""
    return conditioned_frame51(motion43(commands, times, **kw), commands)
