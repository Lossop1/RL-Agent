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
EVENT_RISK_W = 0.75
SMOOTH_W = 0.03


def masked_huber(pred, label, mask, delta=1.0):
    """只在掩码有效且输入有限的位置计算 Huber 均值。

    仅在损失之后乘零不能隔离 NaN，因为 ``NaN * 0`` 仍然是 NaN。
    无效标签必须在进入 Huber 之前替换掉。
    """
    valid = (mask > 0) & torch.isfinite(mask) & torch.isfinite(pred) & torch.isfinite(label)
    safe_pred = torch.where(valid, pred, torch.zeros_like(pred))
    safe_label = torch.where(valid, label, torch.zeros_like(label))
    per = F.huber_loss(safe_pred, safe_label, reduction="none", delta=delta)
    valid_f = valid.to(dtype=per.dtype)
    denom = valid_f.sum().clamp_min(1.0)
    return (per * valid_f).sum() / denom


def smoothness_loss(z_t, z_prev, steady_mask):
    """0.03 * steady_mask * ||z_t - stopgrad(z_prev)||^2 (z_prev is the detached target)."""
    d = ((z_t - z_prev.detach()) ** 2).sum(dim=-1)        # per sample
    denom = steady_mask.sum().clamp_min(1.0)
    return SMOOTH_W * (steady_mask * d).sum() / denom


def aux_loss(geom_pred, geom_label, geom_mask, risk_pred, risk_label, risk_mask,
             z_t=None, z_prev=None, steady_mask=None):
    """辅助损失；稀疏事件通道独立归一化，避免被常驻基础风险样本稀释。"""
    if risk_pred.shape != risk_label.shape or risk_pred.shape != risk_mask.shape:
        raise ValueError("risk prediction, label and mask shapes must match")
    loss = GEOM_W * masked_huber(geom_pred, geom_label, geom_mask) \
        + RISK_W * masked_huber(risk_pred[..., :2], risk_label[..., :2], risk_mask[..., :2])
    if risk_pred.shape[-1] > 2:
        loss = loss + EVENT_RISK_W * masked_huber(
            risk_pred[..., 2:],
            risk_label[..., 2:],
            risk_mask[..., 2:],
        )
    if z_t is not None and z_prev is not None and steady_mask is not None:
        loss = loss + smoothness_loss(z_t, z_prev, steady_mask)
    return loss


def grad_scale_schedule(step: int, ramp_steps: int):
    """policy->perceiver gradient ramp 0 -> 1 over the first ramp_steps updates (B)."""
    if ramp_steps <= 0:
        return 1.0
    return float(min(max(step / ramp_steps, 0.0), 1.0))
