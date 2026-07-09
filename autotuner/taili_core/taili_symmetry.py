"""Canonical L/R mirror for Taili — THE single source.

actor, TerrainPerceiver, AMP reference/discriminator, log_std tying, tests and any
data-augmentation MUST import the maps from here; no duplication anywhere else
(taili_strategy_decisions.md, section E).

Sagittal-plane reflection (y -> -y). Canonical joint order = Isaac
[FL, FR, RL, RR] x [hip, thigh, calf]:

    [FL_hip, FR_hip, RL_hip, RR_hip,
     FL_thigh, FR_thigh, RL_thigh, RR_thigh,
     FL_calf, FR_calf, RL_calf, RR_calf]

Physical correctness is NOT implied by involution / z-consistency / actor-equivariance
tests — a wrong-but-involutive map passes all three. It is anchored by the physical
round-trip / concrete-example tests (test_taili_symmetry.py).
"""
from __future__ import annotations

import torch

# ── canonical joint order ────────────────────────────────────────────────────
CANONICAL_JOINT_ORDER = [
    "FL_hip", "FR_hip", "RL_hip", "RR_hip",
    "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
    "FL_calf", "FR_calf", "RL_calf", "RR_calf",
]

# ── mirror permutations / signs (all derived from the y -> -y reflection) ─────
# joint vector: swap L<->R within each (type, front/rear) pair; negate hip (abduction)
JP = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
JS = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# 3-vectors under y->-y: axial (pseudo) flips x,z ; polar flips y
GYRO_S = [-1.0, 1.0, -1.0]          # body angular velocity (axial)
GRAV_S = [1.0, -1.0, 1.0]           # projected gravity (polar)
CMD_S = [1.0, -1.0, -1.0]           # command [vx, vy, wz]: vx keep, vy/wz flip
LINV_S = [1.0, -1.0, 1.0]           # linear velocity (polar)
TCTX_S = [1.0, -1.0, 1.0]           # terrain_ctx [fore_slope, lat_slope, rough]
# gait clock = (sin4, cos4) BLOCK layout [sinFL,sinFR,sinRL,sinRR, cosFL,cosFR,cosRL,cosRR]
# swap legs within each block, no sign flip
GAIT_P = [1, 0, 3, 2, 5, 4, 7, 6]
# per-foot quantities [FL, FR, RL, RR]: swap FL<->FR, RL<->RR
FOOT_P = [1, 0, 3, 2]
FOOT_XYZ_S = [1.0, -1.0, 1.0]       # each foot xyz: polar y-flip
# tangent_normal6 = [tangent(3), normal(3)], both polar -> y-flip each
TN_S = [1.0, -1.0, 1.0, 1.0, -1.0, 1.0]
# mode_onehot5 [stand, fwd, back, lat, yaw]: identity (direction is in command)
MODE_P = [0, 1, 2, 3, 4]
# geom_label9 [slope_x, slope_y, log_rough, foot_h FL/FR/RL/RR, edge_up, edge_down]
# under y->-y: slope_y flips; foot_h swaps FL<->FR, RL<->RR; rest unchanged
GEOM_P = [0, 1, 2, 4, 3, 6, 5, 7, 8]
GEOM_S = [1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# risk_label2 [impact_score, support_instability]: both unchanged
RISK_P = [0, 1]
RISK_S = [1.0, 1.0]


def _seg(x, perm, sign=None):
    """Mirror a trailing segment by permutation (+ optional per-channel sign)."""
    out = x[..., perm]
    if sign is not None:
        out = out * torch.as_tensor(sign, dtype=out.dtype, device=out.device)
    return out


def _sgn(x, sign):
    return x * torch.as_tensor(sign, dtype=x.dtype, device=x.device)


# ── primitive mirrors ────────────────────────────────────────────────────────
def mirror_joint_vec12(x):
    return _seg(x, JP, JS)


def mirror_action12(a):
    return _seg(a, JP, JS)


def mirror_command3(c):
    return _sgn(c, CMD_S)


def mirror_gyro3(w):
    return _sgn(w, GYRO_S)


def mirror_projected_gravity3(g):
    return _sgn(g, GRAV_S)


def mirror_gait_clock8(clk):
    return _seg(clk, GAIT_P)            # permutation only


def mirror_geom_label9(g):
    return _seg(g, GEOM_P, GEOM_S)      # aux geom label (slope_y flips, foot_h swaps)


def mirror_risk_label2(r):
    return _seg(r, RISK_P, RISK_S)      # aux risk label (impact, support_instability unchanged)


def mirror_z_terrain32(z):
    half = z.shape[-1] // 2
    return torch.cat([z[..., half:], z[..., :half]], dim=-1)   # swap halves


def _mirror_foot_rel12(f):
    """f (..., 12) = [FL_xyz, FR_xyz, RL_xyz, RR_xyz] -> swap feet + per-foot y-flip."""
    lead = f.shape[:-1]
    f = f.reshape(*lead, 4, 3)[..., FOOT_P, :]
    f = _sgn(f, FOOT_XYZ_S)
    return f.reshape(*lead, 12)


# ── composite layouts ────────────────────────────────────────────────────────
# body53: angvel3 | grav3 | cmd3 | jpos12 | jvel12 | lastact12 | gait8
def mirror_actor_body53(b):
    return torch.cat([
        mirror_gyro3(b[..., 0:3]),
        mirror_projected_gravity3(b[..., 3:6]),
        mirror_command3(b[..., 6:9]),
        mirror_joint_vec12(b[..., 9:21]),
        mirror_joint_vec12(b[..., 21:33]),
        mirror_action12(b[..., 33:45]),
        mirror_gait_clock8(b[..., 45:53]),
    ], dim=-1)


# actor_input85 = body53 | z_terrain32
def mirror_actor_input85(x):
    return torch.cat([mirror_actor_body53(x[..., :53]), mirror_z_terrain32(x[..., 53:85])], dim=-1)


# tick54: q_rel12 | dq12 | q_des_rel12 | q_error12 | gyro3 | grav3
def mirror_perceiver_tick54(t):
    return torch.cat([
        mirror_joint_vec12(t[..., 0:12]),
        mirror_joint_vec12(t[..., 12:24]),
        mirror_joint_vec12(t[..., 24:36]),
        mirror_joint_vec12(t[..., 36:48]),
        mirror_gyro3(t[..., 48:51]),
        mirror_projected_gravity3(t[..., 51:54]),
    ], dim=-1)


def mirror_perceiver_history_25x54(h):
    """h (..., T, 54): mirror each tick; time order NOT reversed."""
    return mirror_perceiver_tick54(h)


# amp_motion43: jp12 | jv12 | bh1 | tn6 | foot_rel12
def mirror_amp_motion43(m):
    return torch.cat([
        mirror_joint_vec12(m[..., 0:12]),
        mirror_joint_vec12(m[..., 12:24]),
        m[..., 24:25],                       # base height unchanged
        _sgn(m[..., 25:31], TN_S),           # tangent+normal (polar)
        _mirror_foot_rel12(m[..., 31:43]),
    ], dim=-1)


# amp_frame51: motion43 | cmd3 | mode5
def mirror_amp_frame51(fr):
    return torch.cat([
        mirror_amp_motion43(fr[..., 0:43]),
        mirror_command3(fr[..., 43:46]),
        fr[..., 46:51],                      # mode onehot unchanged
    ], dim=-1)


# amp two-frame discriminator input: frame51 | frame51 (102)
def mirror_amp_frame102(x):
    return torch.cat([mirror_amp_frame51(x[..., 0:51]), mirror_amp_frame51(x[..., 51:102])], dim=-1)


# ── actor helpers ────────────────────────────────────────────────────────────
def structural_mean(net, x, mirror_in=mirror_actor_input85, mirror_act=mirror_action12):
    """mean = 0.5 [net(x) + M_act net(M_in x)] — L/R equivariant for ANY net weights."""
    return 0.5 * (net(x) + mirror_act(net(mirror_in(x))))


def tie_log_std(log_std):
    """Tie L/R action pairs by JP so log_std[i] == log_std[JP[i]] (std is sign-free)."""
    return 0.5 * (log_std + log_std[..., JP])
