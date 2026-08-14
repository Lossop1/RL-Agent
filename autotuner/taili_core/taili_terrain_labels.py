"""Privileged geom/risk label kernels — strict per taili_strategy_decisions.md
(「地形感知器」slope/roughness/impact/support_instability + 共享特权信号契约).

Pure-math kernels the training env feeds with privileged sim quantities (terrain-height
samples, contact forces, support geometry). Training-ONLY: these produce the geom[9] and
base risk[2] labels; the env appends six post-contact event labels. They supervise z_terrain
and are NEVER deployment inputs.

Pure torch; CPU-testable with synthetic inputs (test_taili_terrain_labels.py), including a
physical mirror round-trip that pins the slope/foot_h signs to the y->-y reflection.
"""
from __future__ import annotations

import math

import torch

G = 9.81
TAN25 = math.tan(math.radians(25.0))


# ── slope: least-squares local support plane  z = a x + b y + c ──────────────
def fit_local_plane(xy, z):
    """xy (...,N,2), z (...,N) -> (a, b, c). slope_x=a, slope_y=b."""
    ones = torch.ones_like(z)
    A = torch.stack([xy[..., 0], xy[..., 1], ones], dim=-1)        # (...,N,3)
    # normal equations  (A^T A) p = A^T z
    AtA = A.transpose(-1, -2) @ A
    Atz = (A.transpose(-1, -2) @ z.unsqueeze(-1)).squeeze(-1)
    p = torch.linalg.solve(AtA, Atz)
    return p[..., 0], p[..., 1], p[..., 2]


def slope_norm(slope):
    return torch.clamp(slope / TAN25, -1.0, 1.0)


# ── roughness: residual std after plane fit ──────────────────────────────────
def roughness(xy, z):
    a, b, c = fit_local_plane(xy, z)
    z_plane = a.unsqueeze(-1) * xy[..., 0] + b.unsqueeze(-1) * xy[..., 1] + c.unsqueeze(-1)
    resid = z - z_plane
    return resid.std(dim=-1)


def log_roughness(xy, z):
    return torch.log(roughness(xy, z) + 1e-4)


def log_roughness_norm(xy, z, rough_mu=-5.0, rough_std=1.5):
    lr = log_roughness(xy, z)
    return torch.clamp((lr - rough_mu) / rough_std, -3.0, 3.0) / 3.0


# ── foot_h: terrain height under foot minus support-plane height ─────────────
def foot_h_norm(terrain_h, support_plane_h):
    return torch.clamp((terrain_h - support_plane_h) / 0.30, -1.0, 1.0)


# ── edge_up / edge_down: fore-aft height step in a base-frame near-body grid ──
def edge_up_down(heights):
    """heights (..., NX, NY) terrain heights on a base-frame grid, NX = fore-aft (x).
    dz = z(x+dx, y) - z(x, y) along fore-aft. edge_up = max(dz), edge_down = max(-dz)."""
    dz = heights[..., 1:, :] - heights[..., :-1, :]            # fore-aft diff
    flat = dz.reshape(*dz.shape[:-2], -1)
    edge_up = torch.clamp(flat.max(dim=-1).values, min=0.0)
    edge_down = torch.clamp((-flat).max(dim=-1).values, min=0.0)
    return edge_up, edge_down


def edge_up_norm(edge_up):
    return torch.clamp(edge_up / 0.30, 0.0, 1.0)


def edge_down_norm(edge_down):
    return torch.clamp(edge_down / 0.30, 0.0, 1.0)


# ── impact_score: recent max contact force, normalized by body weight ────────
def impact_score(f_max, mass):
    return torch.log(1.0 + f_max / (mass * G))


def impact_score_norm(f_max, mass):
    return torch.clamp(impact_score(f_max, mass) / 3.0, 0.0, 1.0)


# ── support_instability: support-polygon margin + attitude tilt ──────────────
def support_instability(contact_count, polygon_margin, diagonal_pair, dist_to_support_line,
                        roll, pitch):
    """Piecewise margin badness (per doc) max attitude tilt badness. Scalars or batched."""
    cc = torch.as_tensor(contact_count)
    margin_ge3 = torch.clamp((0.03 - polygon_margin) / 0.08, 0.0, 1.0)
    margin_2diag = torch.clamp((dist_to_support_line - 0.02) / 0.08, 0.0, 1.0)
    margin_2 = torch.where(torch.as_tensor(diagonal_pair, dtype=torch.bool),
                           margin_2diag, torch.full_like(margin_2diag, 0.7))
    margin_bad = torch.where(cc >= 3, margin_ge3,
                  torch.where(cc == 2, margin_2,
                   torch.where(cc == 1, torch.full_like(margin_ge3, 0.9),
                               torch.full_like(margin_ge3, 1.0))))
    tilt = torch.sqrt(roll ** 2 + pitch ** 2)
    tilt_bad = torch.clamp((tilt - math.radians(10.0)) / math.radians(20.0), 0.0, 1.0)
    return torch.maximum(margin_bad, tilt_bad)


# ── 组装 geom[9]、基础 risk[2] 和碰触后事件 risk[8] ─────────────────────────
def assemble_geom9(slope_x_n, slope_y_n, log_rough_n, foot_h_n4, edge_up_n, edge_down_n):
    """foot_h_n4 (...,4) in FOOT order FL,FR,RL,RR."""
    return torch.cat([
        slope_x_n.unsqueeze(-1), slope_y_n.unsqueeze(-1), log_rough_n.unsqueeze(-1),
        foot_h_n4, edge_up_n.unsqueeze(-1), edge_down_n.unsqueeze(-1),
    ], dim=-1)


def assemble_risk8(
    impact_n,
    support_instability,
    event_direction,
    lead_foot4,
    event_visible,
):
    """组装基础风险和碰触后事件监督，同时返回逐维有效掩码。

    `event_visible` 必须由已发生的碰撞、预期触地失败或真实换层触发。事件不可见
    时后六维完全不参与损失，避免用训练期地形类型给盲态策略提供前视信息。
    """
    dtype = impact_n.dtype
    base = torch.stack([impact_n, support_instability.to(dtype=dtype)], dim=-1)
    direction = event_direction.to(device=impact_n.device)
    lead = lead_foot4.to(dtype=dtype, device=impact_n.device)
    event = torch.cat([
        (direction > 0).to(dtype)[:, None],
        (direction < 0).to(dtype)[:, None],
        lead,
    ], dim=-1)
    label = torch.cat([base, event], dim=-1)
    visible = event_visible.to(dtype=dtype, device=impact_n.device)
    mask = torch.cat([
        torch.ones_like(base),
        visible[:, None].expand_as(event),
    ], dim=-1)
    return label, mask


def assemble_risk2(impact_n, support_instab):
    return torch.stack([impact_n, support_instab], dim=-1)
