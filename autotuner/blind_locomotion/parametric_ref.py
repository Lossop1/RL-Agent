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

# --- URDF kinematic constants (identical to gen_taili_gaits.py) ---
L1 = 0.36385                                          # thigh length
FOOT_OFF = (0.0252765, 0.0, -0.3179635)              # calf->foot offset (sagittal x,z used)
HIPX, HIPY = 0.30414, 0.065                          # hip joint offset from base (x, y)
THIGHY = 0.1432                                       # hip-abduction-axis -> thigh sagittal-plane y offset
BASE_Z = 0.52                                        # nominal standing base height; YAML reward.nominal_base_h is the source
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
    th_t = torch.full_like(tx, 0.7)   # OCCAM/ROOT-1 (0707): q_default seed (was stale 0.6/-1.2)
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


# nominal foot (x,z) rel hip at the standing joints — q_default (0, 0.7, -1.4), matching the TASK reward's
# default pose so AMP style + imitation + the phi-gate style_err no longer fight the task at every stance/stand
# frame (ROOT-1, 0707). BASE_Z is DERIVED from FK(q_default) so the nominal foot is grounded (was stale 0.52
# with a 0.548 foot depth -> foot 0.028 m below ground; now 0.5052 consistent).
with torch.no_grad():
    _nx, _nz = _fk2(torch.tensor(0.7), torch.tensor(-1.4))
    X0 = float(_nx)
    H0 = -float(_nz)
    BASE_Z = H0   # FK-derived; overrides the stale 0.52 literal above, matches taili_amp_reference.BASE_HEIGHT_REF


def _foot_traj(p, sdx, sdy, clearance):
    """trot stance/swing foot trajectory (torch) at per-leg phase p in [0,1). Matches foot_traj_rel_hip.
    stance (p<0.5): planted, sliding back rel body; swing (p>=0.5): lift + return to front foothold.

    SOFT, NO-SLAM, NO-SCUFF swing (spec: smooth/no-slam/no-slip):
      * vertical  clearance*sin^2(pi*s): dz/dt -> 0 at touchdown (sin gave -clearance*2pi/T ~ -0.5 m/s = SLAM).
      * horizontal velocity-matched RETRACTION h(s)=-4s^3+6s^2-s (h(0)=0,h(1)=1, h'(0)=h'(1)=-1): the foot's
        body-frame velocity == the stance slide rate (-sdx*2/T) at BOTH liftoff & touchdown -> the foot is
        already world-stationary when it plants (no scuff; linear swing landed at +0.6 vs stance -0.6 = skid).
      Foothold endpoints (+/-sdx/2) unchanged -> stance & no-slip body-velocity consistency preserved."""
    stance = p < 0.5
    s_st = p / 0.5
    s_sw = (p - 0.5) / 0.5
    hsw = -4.0 * s_sw ** 3 + 6.0 * s_sw ** 2 - s_sw
    fx = torch.where(stance, X0 + sdx / 2 - sdx * s_st, X0 - sdx / 2 + sdx * hsw)
    fy = torch.where(stance, sdy / 2 - sdy * s_st, -sdy / 2 + sdy * hsw)
    fz = torch.where(stance, torch.full_like(p, -H0), -H0 + clearance * torch.sin(math.pi * s_sw) ** 2)
    return fx, fy, fz


def period_for_speed(speed, base, slope, pmin):
    """speed-adaptive gait period (same law as the env clock)."""
    return torch.clamp(base - slope * speed, min=pmin, max=base)


def flat_reference(commands, times, *, gait_period=0.55, gait_period_slope=0.075, gait_period_min=0.40,
                   clearance_base=0.09, clearance_gain=0.03, roughness=None, clearance_rough_gain=0.30,
                   stance_dx=0.0, iters=40, jp_only=False):
    """Analytic trot reference. commands (M,3) [vx,vy,wz], times (M,) local time (s).
    roughness (M,) optional terrain roughness in [0,~0.3]: raises swing clearance so the AMP discriminator
    rewards a HIGH-LIFT climbing style on rough/stair terrain (flat terrain keeps the low-energy ~9cm lift).
    Returns jp (M,12 clip dof order), jv (M,12), bh (M,1), tn (M,6), foot_rel (M,4,3)."""
    dev = commands.device
    vx, vy, wz = commands[:, 0], commands[:, 1], commands[:, 2]
    speed = torch.norm(commands[:, :2], dim=1)
    T = period_for_speed(speed, gait_period, gait_period_slope, gait_period_min)           # (M,)
    clearance = clearance_base + clearance_gain * torch.clamp(speed / 2.0, 0.0, 1.0)        # (M,)
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
