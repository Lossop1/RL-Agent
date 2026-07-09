"""Unit tests for the pure-torch aux/smoothness loss math (taili_losses.py).

The trainer PLUMBING (optimizer groups, z_prev pairing through the rollout shuffle) is remote
integration; the MATH here is CPU-unit-testable and is what the TerrainPerceiver aux head trains
against, so a regression here silently corrupts the learned terrain latent.
"""
import pytest

torch = pytest.importorskip("torch")

from autotuner.taili_core import taili_losses as L


def test_masked_huber_averages_over_valid_only():
    pred = torch.zeros(4)
    label = torch.tensor([1.0, 1.0, 1.0, 1.0])   # each Huber(|1|,delta=1)=0.5
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])    # only first two valid
    # mean over valid entries = 0.5, NOT summed/4 = 0.25
    assert torch.isclose(L.masked_huber(pred, label, mask), torch.tensor(0.5))


def test_masked_huber_all_masked_out_is_zero_not_nan():
    pred = torch.zeros(3)
    label = torch.ones(3)
    mask = torch.zeros(3)
    out = L.masked_huber(pred, label, mask)
    assert torch.isfinite(out) and float(out) == 0.0   # clamp_min(1.0) denom guards div-by-zero


def test_smoothness_uses_detached_prev_and_steady_mask():
    z_t = torch.tensor([[1.0, 0.0]], requires_grad=True)
    z_prev = torch.tensor([[0.0, 0.0]], requires_grad=True)
    steady = torch.tensor([1.0])
    loss = L.smoothness_loss(z_t, z_prev, steady)
    # 0.03 * ||[1,0]||^2 = 0.03
    assert torch.isclose(loss, torch.tensor(0.03))
    loss.backward()
    # gradient flows to z_t but NOT to z_prev (it is the detached target)
    assert z_t.grad is not None and z_t.grad.abs().sum() > 0
    assert z_prev.grad is None or float(z_prev.grad.abs().sum()) == 0.0


def test_smoothness_zero_when_not_steady():
    z_t = torch.tensor([[5.0, 5.0]])
    z_prev = torch.zeros(1, 2)
    assert float(L.smoothness_loss(z_t, z_prev, torch.tensor([0.0]))) == 0.0


def test_aux_loss_weights_geom_and_risk():
    # geom Huber 0.5 (weight 1.0) + risk Huber 0.5 (weight 0.5) = 0.5 + 0.25 = 0.75
    z = torch.zeros(1, 2)
    geom_pred, geom_label, geom_mask = torch.zeros(2), torch.ones(2), torch.ones(2)
    risk_pred, risk_label, risk_mask = torch.zeros(2), torch.ones(2), torch.ones(2)
    loss = L.aux_loss(geom_pred, geom_label, geom_mask, risk_pred, risk_label, risk_mask)
    assert torch.isclose(loss, torch.tensor(0.75))
    assert (L.GEOM_W, L.RISK_W, L.SMOOTH_W) == (1.0, 0.5, 0.03)
