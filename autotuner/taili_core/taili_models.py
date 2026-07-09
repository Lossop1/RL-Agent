"""Taili blind-deployable model core — TerrainPerceiver + structurally-equivariant actor.

Strict implementation of taili_strategy_decisions.md ("TerrainPerceiver Contract v1" + A + B).
This is the BLIND DOG draft: z_terrain is extracted from deployable proprioceptive history,
trained JOINTLY with PPO + aux (NOT a teacher-student / privileged-encoder distillation).

Pure torch; imports the single-source mirror from taili_symmetry. No skrl / isaaclab here —
the env/skrl wrapper imports these. Unit-testable on CPU (test_taili_models.py).
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import taili_symmetry as sym

# layout constants (Runtime IO / Contract v1)
TICK_DIM = 54
HISTORY_LEN = 25
BODY_DIM = 53
ACTION_DIM = 12
Z_HALF = 16
Z_DIM = 2 * Z_HALF          # 32
GEOM_DIM = 9
RISK_DIM = 2


# ── causal TCN encoder u(h): [B,25,54] -> [B,16] ─────────────────────────────
class _CausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel, dilation=dilation)

    def forward(self, x):                       # x: [B, C, T]
        return self.conv(F.pad(x, (self.pad, 0)))   # left-pad only -> causal, T preserved


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = _CausalConv1d(channels, kernel, dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(F.elu(self.conv(x)))


class TCNEncoder(nn.Module):
    """u(h): [25,54] -> Linear(54,128)+ELU -> 4x residual causal conv (dil 1,2,4,8)
    -> last timestep -> Linear(128,64)+ELU -> Linear(64,16). Receptive field 31 >= 25."""

    def __init__(self, in_dim=TICK_DIM, channels=128, out_dim=Z_HALF, dropout=0.0):
        super().__init__()
        self.inp = nn.Linear(in_dim, channels)
        self.blocks = nn.ModuleList(
            [_ResidualBlock(channels, 3, d, dropout) for d in (1, 2, 4, 8)]
        )
        self.head = nn.Sequential(nn.Linear(channels, 64), nn.ELU(), nn.Linear(64, out_dim))

    def forward(self, h):                        # h: [B, 25, 54]
        x = F.elu(self.inp(h)).transpose(1, 2)   # [B, 128, 25]
        for blk in self.blocks:
            x = blk(x)
        return self.head(x[:, :, -1])            # last timestep -> [B, 16]


# ── TerrainPerceiver: history -> z_terrain[32] (+ training aux heads) ─────────
class TerrainPerceiver(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.u = TCNEncoder(TICK_DIM, 128, Z_HALF, dropout)
        self.geom_head = nn.Linear(Z_DIM, GEOM_DIM)   # training-only
        self.risk_head = nn.Linear(Z_DIM, RISK_DIM)   # training-only

    def encode(self, history):                   # [B,25,54] -> [B,32]
        z_left = self.u(history)
        z_right = self.u(sym.mirror_perceiver_history_25x54(history))
        return torch.cat([z_left, z_right], dim=-1)   # = swap_halves under history mirror

    def aux(self, z):                            # training only; deploy ignores
        return self.geom_head(z), self.risk_head(z)


def apply_grad_scale(z, grad_scale: float):
    """policy->perceiver gradient ramp: forward value == z, gradient scaled by grad_scale (B §4)."""
    return grad_scale * z + (1.0 - grad_scale) * z.detach()


# ── structurally-equivariant actor: body53 + z32 -> action12 ─────────────────
class EquivariantActor(nn.Module):
    def __init__(self, hidden=(1024, 512), initial_log_std=-1.0):
        super().__init__()
        layers, d = [], BODY_DIM + Z_DIM
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, ACTION_DIM)]
        self.net = nn.Sequential(*layers)
        self.log_std_param = nn.Parameter(torch.full((ACTION_DIM,), float(initial_log_std)))

    def mean(self, body, z):
        """Structurally L/R-equivariant mean for any weights: 1/2[net(x)+M_a net(M_in x)].

        A/B DIAGNOSTIC (0704): hard equivariance averages 1/2[net(x)+M net(Mx)], which for the
        ANTI-symmetric (turning) action component averages two initially-uncorrelated halves →
        suppresses turning output at init and slows yaw learning while symmetric (forward) is
        preserved. Set TAILI_NO_EQUIV=1 to bypass and use the raw net(x): if yaw then learns,
        the hard equivariance is the yaw blocker → replace with SOFT equivariance (mirror loss).
        """
        x = torch.cat([body, z], dim=-1)         # [B, 85]
        if os.environ.get("TAILI_NO_EQUIV") == "1":
            return self.net(x)                   # raw, non-equivariant (diagnostic only)
        return sym.structural_mean(self.net, x, sym.mirror_actor_input85, sym.mirror_action12)

    def log_std(self):
        return sym.tie_log_std(self.log_std_param)   # L/R-tied exploration


# ── AMP discriminator: two-frame style prior, command/mode-conditioned (D) ────
AMP_FRAME_DIM = 51
AMP_NUM_FRAMES = 2


class AMPDiscriminator(nn.Module):
    """Plain MLP on frame51 x 2 = 102 -> 1 logit. NOT equivariant by construction;
    L/R symmetry of the style prior is enforced by mirror-augmenting the training frames
    (sym.mirror_amp_frame102), applied identically to reference and policy samples."""

    def __init__(self, hidden=(512, 256)):
        super().__init__()
        layers, d = [], AMP_FRAME_DIM * AMP_NUM_FRAMES
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, frames102):
        return self.net(frames102)
