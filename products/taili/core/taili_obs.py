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


def update_tick_history(history, tick, update_mask=None, order="oldest_first"):
    """更新固定长度历史；旧检查点可选择其训练时使用的newest_first语义。"""
    if order == "oldest_first":
        shifted = torch.cat([history[:, 1:], tick.unsqueeze(1)], dim=1)
    elif order == "newest_first":
        shifted = torch.cat([tick.unsqueeze(1), history[:, :-1]], dim=1)
    else:
        raise ValueError(f"unsupported history order: {order}")
    if update_mask is None:
        return shifted
    return torch.where(update_mask[:, None, None], shifted, history)


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


def assemble_body57(gyro, gravity, command, previous_command, command_age,
                    jpos, jvel, last_action, gait_clock):
    """组装部署侧 Actor 观测。

    command_age 是相对最大过渡窗口归一化到 [0, 1] 的命令年龄。上一命令和命令年龄
    都来自控制器，不依赖仿真特权信息，使策略能够辨认瞬时换向而不需要篡改用户命令。
    """
    if command_age.ndim == command.ndim - 1:
        command_age = command_age.unsqueeze(-1)
    return torch.cat([
        gyro, gravity, command, previous_command, command_age,
        jpos, jvel, last_action, gait_clock,
    ], dim=-1)


def foot_mask_to_joint12(foot_mask):
    """把 FL/FR/RL/RR 足端 mask 映射到 hip[4] | thigh[4] | calf[4] 关节顺序。"""
    return torch.cat([foot_mask, foot_mask, foot_mask], dim=-1)
