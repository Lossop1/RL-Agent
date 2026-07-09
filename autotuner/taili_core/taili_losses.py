"""TerrainPerceiver aux + smoothness losses and the grad_scale schedule — strict per
taili_strategy_decisions.md (B + TerrainPerceiver Contract aux/loss + smoothness pairing).

The pure-torch loss MATH (masked Huber aux, smoothness with detached z_prev, grad_scale ramp).
The skrl trainer PLUMBING (optimizer groups, hooking into the AMP update, z_prev pairing through
the rollout shuffle) is remote/integration — this is the part that is unit-testable on CPU.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

GEOM_W = 1.0
RISK_W = 0.5
SMOOTH_W = 0.03


def masked_huber(pred, label, mask, delta=1.0):
    """Per-element Huber, masked, averaged over VALID entries only (mask in {0,1})."""
    per = F.huber_loss(pred, label, reduction="none", delta=delta)
    denom = mask.sum().clamp_min(1.0)
    return (per * mask).sum() / denom


def smoothness_loss(z_t, z_prev, steady_mask):
    """0.03 * steady_mask * ||z_t - stopgrad(z_prev)||^2 (z_prev is the detached target)."""
    d = ((z_t - z_prev.detach()) ** 2).sum(dim=-1)        # per sample
    denom = steady_mask.sum().clamp_min(1.0)
    return SMOOTH_W * (steady_mask * d).sum() / denom


def aux_loss(geom_pred, geom_label, geom_mask, risk_pred, risk_label, risk_mask,
             z_t=None, z_prev=None, steady_mask=None):
    """L_aux = 1.0*Huber(geom) + 0.5*Huber(risk) + 0.03*smoothness (if z_t/z_prev given)."""
    loss = GEOM_W * masked_huber(geom_pred, geom_label, geom_mask) \
        + RISK_W * masked_huber(risk_pred, risk_label, risk_mask)
    if z_t is not None and z_prev is not None and steady_mask is not None:
        loss = loss + smoothness_loss(z_t, z_prev, steady_mask)
    return loss


def grad_scale_schedule(step: int, ramp_steps: int):
    """policy->perceiver gradient ramp 0 -> 1 over the first ramp_steps updates (B)."""
    if ramp_steps <= 0:
        return 1.0
    return float(min(max(step / ramp_steps, 0.0), 1.0))
