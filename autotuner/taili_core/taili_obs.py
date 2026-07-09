"""Observation assembly — strict per taili_strategy_decisions.md (Runtime IO Contract v1).

Builds the model inputs from raw per-tick runtime quantities (canonical joint order), with the
exact q_des / q_error recipe. Identical in sim / real / export. The SAMPLING of those raw
quantities (encoders, IMU, sim) is env/adapter-side; this is the pure-torch assembly recipe.
"""
from __future__ import annotations

import math

import torch

from . import taili_amp_reference as ref   # Q_DEFAULT12

Q_DEFAULT = torch.tensor(ref.Q_DEFAULT12)
ACTION_SCALE = 0.35


def q_des_rel(last_action, action_scale=ACTION_SCALE):
    return action_scale * last_action


def q_des(last_action, action_scale=ACTION_SCALE, q_default=None):
    qd = Q_DEFAULT.to(last_action) if q_default is None else q_default
    return qd + q_des_rel(last_action, action_scale)


def assemble_tick54(q, dq, last_action, gyro, gravity, action_scale=ACTION_SCALE, q_default=None):
    """tick54 = q_rel | dq | q_des_rel | q_error | gyro | projected_gravity.
    q_rel=q-q_default; q_des_rel=action_scale*last_action; q_error=q_des-q (servo lag)."""
    qd = Q_DEFAULT.to(q) if q_default is None else q_default
    qr = q - qd
    qdr = action_scale * last_action
    qerr = (qd + qdr) - q
    return torch.cat([qr, dq, qdr, qerr, gyro, gravity], dim=-1)


def gait_clock8(phase4):
    """phase4 (...,4) per-leg phase in [0,1) -> (sin4, cos4) block layout."""
    ang = 2.0 * math.pi * phase4
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


def assemble_body53(gyro, gravity, command, jpos, jvel, last_action, gait_clock):
    """body53 = angvel | gravity | command | jpos | jvel | last_action | gait_clock."""
    return torch.cat([gyro, gravity, command, jpos, jvel, last_action, gait_clock], dim=-1)
