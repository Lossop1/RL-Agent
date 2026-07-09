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


# nominal foot (x,z) rel hip at q_default -> X0, H0 ; base_height_ref = H0 (FK-derived)
with torch.no_grad():
    _nx, _nz = _fk2(torch.tensor(Q_DEFAULT_THIGH), torch.tensor(Q_DEFAULT_CALF))
    X0 = float(_nx)
    H0 = -float(_nz)
BASE_HEIGHT_REF = H0          # derived from FK(q_default); cross-checked ~ nominal_base_h 0.52


def _foot_traj(p, sdx, sdy, clearance):
    """trot stance/swing foot trajectory at per-leg phase p in [0,1). Smooth, no-slam, no-scuff."""
    stance = p < 0.5
    s_st = p / 0.5
    s_sw = (p - 0.5) / 0.5
    hsw = -4.0 * s_sw ** 3 + 6.0 * s_sw ** 2 - s_sw
    fx = torch.where(stance, X0 + sdx / 2 - sdx * s_st, X0 - sdx / 2 + sdx * hsw)
    fy = torch.where(stance, sdy / 2 - sdy * s_st, -sdy / 2 + sdy * hsw)
    fz = torch.where(stance, torch.full_like(p, -H0), -H0 + clearance * torch.sin(math.pi * s_sw) ** 2)
    return fx, fy, fz


def period_for_speed(speed, base=0.55, slope=0.075, pmin=0.40):
    return torch.clamp(base - slope * speed, min=pmin, max=base)


def flat_reference(commands, times, *, gait_period=0.55, gait_period_slope=0.075,
                   gait_period_min=0.40, clearance_base=0.07, clearance_gain=0.03,
                   stance_dx=0.0, iters=40):
    """Analytic trot reference, command-conditioned, terrain-AGNOSTIC.
    commands (M,3) [vx,vy,wz], times (M,). Returns jp,jv,bh,tn,foot_rel."""
    dev = commands.device
    vx, vy, wz = commands[:, 0], commands[:, 1], commands[:, 2]
    speed = torch.norm(commands[:, :2], dim=1)
    # yaw-aware effective speed for the PERIOD only (0705 D3m): a high-yaw reference gait steps faster
    # (0.30*|wz| ~ |wz|*r_foot), matching the env air-time target so the style reward doesn't fight the
    # faster-stepping the yaw-tracking reward asks for. clearance/stride still use the linear speed.
    period_speed = speed + 0.15 * wz.abs()
    T = period_for_speed(period_speed, gait_period, gait_period_slope, gait_period_min)
    clearance = clearance_base + clearance_gain * torch.clamp(speed / 2.0, 0.0, 1.0)
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


def frame51(commands, times, **kw):
    """frame51 = motion43 + command3 + mode_onehot5 (for the conditional discriminator)."""
    return torch.cat([motion43(commands, times, **kw), commands, mode_onehot(commands)], dim=-1)
