"""Parametric (analytic) AMP reference for FLAT locomotion — a CONTINUOUS function of the command.

Instead of pre-baked discrete-speed clips, this computes the trot reference frame analytically for ANY
commanded (vx, vy, wz) at any gait phase: step frequency + stride scale with the command (period shortens
with speed, matching the env's speed-adaptive gait clock). Vectorized in torch over a batch of samples.

It reproduces gen_taili_gaits' kinematics (per-leg ground velocity -> stance/swing foot trajectory ->
3-DOF IK), so the AMP discriminator sees the SAME style as the clips but with seamless command coverage.
Validated against the discrete clips (see validate_parametric_ref.py).

Returned style features match the clip convention used by collect_reference_motions:
  jp (M,12) joint pos in CLIP dof order [hip x4, thigh x4, calf x4] over legs [FL,FR,RL,RR],
  jv (M,12) joint vel (finite diff), bh (M,1) base height, tn (M,6) base tangent+normal,
  foot_rel (M,4,3) foot position relative to base in base frame, legs [FL,FR,RL,RR].
"""
from __future__ import annotations
import math
import numpy as np
import torch

try:
    from .taili_core import taili_geometry as geometry
except ImportError:
    from products.taili.core import taili_geometry as geometry

# --- URDF kinematic constants (identical to gen_taili_gaits.py) ---
L1 = 0.36385                                          # thigh length
FOOT_OFF = (0.0252765, 0.0, -0.3179635)              # calf->foot offset (sagittal x,z used)
HIPX, HIPY = 0.30414, 0.065                          # hip joint offset from base (x, y)
THIGHY = 0.1432                                       # hip-abduction-axis -> thigh sagittal-plane y offset
BASE_Z = geometry.NOMINAL_BASE_HEIGHT                # 足端球心高度加 URDF 碰撞球半径。
LEGS = ["FL", "FR", "RL", "RR"]
SGN = {"FL": (1, 1), "FR": (1, -1), "RL": (-1, 1), "RR": (-1, -1)}   # (x,y) sign per leg
TROT = {"FL": 0.0, "FR": 0.5, "RL": 0.5, "RR": 0.0}                  # diagonal trot phase offsets


def _fk2(th_t, th_c):
    """2-link sagittal FK (torch): thigh angle th_t, calf angle th_c -> foot (x,z) rel hip. Matches fk_foot."""
    s_t, c_t = torch.sin(th_t), torch.cos(th_t)
    s2, c2 = torch.sin(th_t + th_c), torch.cos(th_t + th_c)
    ox, oz = FOOT_OFF[0], FOOT_OFF[2]
    fx = -L1 * s_t + c2 * ox + s2 * oz
    fz = -L1 * c_t - s2 * ox + c2 * oz
    return fx, fz


def _ik2(tx, tz, iters=40):
    """2-link sagittal IK (vectorized Newton) for target foot (tx,tz) rel hip -> (th_t, th_c)."""
    th_t = torch.full_like(tx, 0.7)   # 从默认站立姿态开始，保证 IK 分支连续。
    th_c = torch.full_like(tx, -1.4)
    ox, oz = FOOT_OFF[0], FOOT_OFF[2]
    for _ in range(iters):
        s_t, c_t = torch.sin(th_t), torch.cos(th_t)
        s2, c2 = torch.sin(th_t + th_c), torch.cos(th_t + th_c)
        fx = -L1 * s_t + c2 * ox + s2 * oz
        fz = -L1 * c_t - s2 * ox + c2 * oz
        ex, ez = tx - fx, tz - fz
        # analytic 2x2 Jacobian d(fx,fz)/d(th_t,th_c)
        j11 = -L1 * c_t - s2 * ox + c2 * oz
        j12 = -s2 * ox + c2 * oz
        j21 = L1 * s_t - c2 * ox - s2 * oz
        j22 = -c2 * ox - s2 * oz
        det = (j11 * j22 - j12 * j21).clamp_min(1e-9)
        dth_t = (j22 * ex - j12 * ez) / det
        dth_c = (-j21 * ex + j11 * ez) / det
        th_t = th_t + dth_t
        th_c = th_c + dth_c
    return th_t, th_c


# nominal foot (x,z) rel hip at the standing joints — q_default (0, 0.7, -1.4), matching the task reward's
# default pose so AMP style, imitation and the style gate share one standing reference. H0 是默认姿态下
# base 到足端球心的距离；BASE_Z 还需
# 加上真实足球半径，才能让碰撞球足底落在地面而不是让球心落在地面。
with torch.no_grad():
    _nx, _nz = _fk2(torch.tensor(0.7), torch.tensor(-1.4))
    X0 = float(_nx)
    H0 = -float(_nz)
    BASE_Z = geometry.NOMINAL_BASE_HEIGHT


def _foot_traj(p, sdx, sdy, clearance):
    """trot stance/swing foot trajectory (torch) at per-leg phase p in [0,1). Matches foot_traj_rel_hip.
    stance (p<0.5): planted, sliding back rel body; swing (p>=0.5): lift + return to front foothold.

    平滑摆动：
      * vertical 使用 64*s^3*(1-s)^3，离地和落地端点的速度、加速度均为 0；
      * horizontal 使用 h(s)=2*smoothstep5(s)-s。它保持 h'(0)=h'(1)=-1，
        使足端在离地/触地时仍与支撑相的机体系后滑速度匹配，同时把端点加速度降为 0。
      Foothold endpoints (+/-sdx/2) unchanged -> stance & no-slip body-velocity consistency preserved."""
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


def period_for_speed(speed, base, slope, pmin):
    """speed-adaptive gait period (same law as the env clock)."""
    return torch.clamp(base - slope * speed, min=pmin, max=base)


def clearance_for_speed(speed, clearance_base, clearance_gain):
    """低速缩短步幅时同步降低平地抬脚高度，避免小步高抬后重落脚。"""
    speed_ratio = torch.clamp(speed / 0.50, 0.0, 1.0)
    base_scale = 0.55 + 0.45 * speed_ratio
    return clearance_base * base_scale + clearance_gain * torch.clamp(speed / 2.0, 0.0, 1.0)


def foot_reference(commands, times, *, gait_period=0.55, gait_period_slope=0.075,
                   gait_period_min=0.40, yaw_speed_equiv=0.15,
                   clearance_base=0.09, clearance_gain=0.03, stance_dx=0.0):
    """生成命令条件化的四足相对机身轨迹，不执行 IK。"""
    vx, vy, wz = commands[:, 0], commands[:, 1], commands[:, 2]
    speed = torch.norm(commands[:, :2], dim=1)
    period_speed = speed + float(yaw_speed_equiv) * wz.abs()
    period = period_for_speed(period_speed, gait_period, gait_period_slope, gait_period_min)
    clearance = clearance_for_speed(period_speed, clearance_base, clearance_gain)
    clearance = clearance * torch.clamp(torch.norm(commands, dim=1) / 0.1, 0.0, 1.0)

    feet = []
    for leg in LEGS:
        sx, sy = SGN[leg]
        hx, hy = sx * HIPX, sy * HIPY
        dx = stance_dx if leg in ("FL", "FR") else -stance_dx
        foot_x0 = hx + X0 + dx
        foot_y0 = hy + sy * THIGHY
        stride_x = (vx - wz * foot_y0) * period / 2.0
        stride_y = (vy + wz * foot_x0) * period / 2.0
        phase = ((times / period) + TROT[leg]) % 1.0
        fx, fy, fz = _foot_traj(phase, stride_x, stride_y, clearance)
        fx = fx + dx
        hip_angle = torch.atan2(fy, -fz)
        fy_body = hy + fy + sy * THIGHY * torch.cos(hip_angle)
        fz_body = fz + sy * THIGHY * torch.sin(hip_angle)
        feet.append(torch.stack((hx + fx, fy_body, fz_body), dim=-1))
    return torch.stack(feet, dim=1)


def flat_reference(commands, times, *, gait_period=0.55, gait_period_slope=0.075, gait_period_min=0.40,
                   yaw_speed_equiv=0.15, clearance_base=0.09, clearance_gain=0.03,
                   roughness=None, clearance_rough_gain=0.30,
                   stance_dx=0.0, iters=40, jp_only=False):
    """Analytic trot reference. commands (M,3) [vx,vy,wz], times (M,) local time (s).
    roughness (M,) optional terrain roughness in [0,~0.3]: raises swing clearance so the AMP discriminator
    rewards a HIGH-LIFT climbing style on rough/stair terrain (flat terrain keeps the low-energy ~9cm lift).
    Returns jp (M,12 clip dof order), jv (M,12), bh (M,1), tn (M,6), foot_rel (M,4,3)."""
    dev = commands.device
    vx, vy, wz = commands[:, 0], commands[:, 1], commands[:, 2]
    speed = torch.norm(commands[:, :2], dim=1)
    # yaw 按足端旋转半径折算成等效线速度，只影响步态周期；步幅仍由刚体足端速度公式决定。
    period_speed = speed + float(yaw_speed_equiv) * wz.abs()
    T = period_for_speed(period_speed, gait_period, gait_period_slope, gait_period_min)    # (M,)
    clearance = clearance_for_speed(period_speed, clearance_base, clearance_gain)             # (M,)
    if roughness is not None:
        clearance = clearance + clearance_rough_gain * torch.clamp(roughness, 0.0, 0.3)     # higher lift on rough
    # STAND reference: at ~zero command the stride is already 0, but the swing still lifts -> the reference is
    # "marching in place", so the discriminator rewards the standing policy for lifting feet -> FIDGET. Gate the
    # swing clearance by command magnitude so a stand command yields a STATIC planted stance (feet at nominal,
    # zero joint velocity) -> the discriminator instead rewards STILLNESS when commanded to stand.
    cmd_mag = torch.norm(commands, dim=1)                                                   # (M,) incl. yaw
    clearance = clearance * torch.clamp(cmd_mag / 0.1, 0.0, 1.0)                            # 0 at stand -> planted

    def joints_at(tt):
        hip = []; thigh = []; calf = []; foot = []
        for lg in LEGS:
            sx, sy = SGN[lg]
            hx, hy = sx * HIPX, sy * HIPY
            # YAW LEVER = the actual foot ground-contact point, NOT the hip. The stance foot sweeps at the
            # rigid-body velocity AT THE FOOT (v = v_cmd + wz x r_foot). Using the hip lever r_hip (shorter than
            # r_foot by the nominal foot offset: X0 fore-aft + sy*THIGHY lateral) under-rotated the commanded yaw
            # by ~22% (implied wz = 0.78*cmd). Use the nominal stance-foot base-frame position as the lever.
            # STANCE-WIDEN (match gen_taili_gaits.foot_traj_rel_hip): front feet +stance_dx, rear -stance_dx ->
            # wider fore-aft support polygon (pitch-stable). stance_dx=0.0 keeps the original narrow stance.
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
            if jp_only:
                continue
            # foot rel BASE = hip offset + Rx(th_h)@([0, sy*THIGHY, 0] + 2-link foot). The decoupled IK target
            # (fx,fy,fz) omits the THIGHY lateral offset; add it back (rotated by abduction) to match real FK.
            fy_b = hy + fy + sy * THIGHY * torch.cos(th_h)
            fz_b = fz + sy * THIGHY * torch.sin(th_h)
            foot.append(torch.stack([hx + fx, fy_b, fz_b], dim=-1))
        jp = torch.stack(hip + thigh + calf, dim=-1)                   # (M,12) clip dof order
        foot_rel = None if jp_only else torch.stack(foot, dim=1)       # (M,4,3)
        return jp, foot_rel

    # JP-ONLY FAST PATH (imitation reward): one IK pass, fewer iters, skip jv/bh/tn/foot_rel. ~6x cheaper than the
    # full discriminator reference, so it can run on ALL envs every step without tanking throughput.
    if jp_only:
        return joints_at(times)[0]

    jp, foot_rel = joints_at(times)
    dt = 1.0 / 50.0
    jp2, _ = joints_at(times + dt)
    jv = (jp2 - jp) / dt

    M = commands.shape[0]
    bh = torch.full((M, 1), BASE_Z, device=dev)
    yaw = wz * times                                                  # base yaw accumulates for turning
    tn = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw),
                      torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)], dim=-1)
    return jp, jv, bh, tn, foot_rel
