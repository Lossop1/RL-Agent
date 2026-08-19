"""Unit tests for the pure-torch aux/smoothness loss math (taili_losses.py).

The trainer PLUMBING (optimizer groups, z_prev pairing through the rollout shuffle) is remote
integration; the MATH here is CPU-unit-testable and is what the TerrainPerceiver aux head trains
against, so a regression here silently corrupts the learned terrain latent.
"""
import pytest

torch = pytest.importorskip("torch")

from products.taili.core import taili_losses as L
from products.taili.blind_locomotion.terrain_perceiver_aux_patch import (
    _exploration_schedule,
    _optimizer_grads_are_finite,
)


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


def test_masked_huber_ignores_nonfinite_values_before_loss_math():
    pred = torch.tensor([0.0, float("nan"), 0.0, float("inf")])
    label = torch.tensor([1.0, 2.0, float("nan"), 4.0])
    mask = torch.tensor([1.0, 0.0, 0.0, 0.0])
    out = L.masked_huber(pred, label, mask)
    assert torch.isfinite(out)
    assert torch.isclose(out, torch.tensor(0.5))


def test_masked_huber_excludes_nonfinite_values_even_with_bad_mask():
    pred = torch.tensor([0.0, 0.0])
    label = torch.tensor([1.0, float("nan")])
    mask = torch.tensor([1.0, float("nan")])
    assert torch.isclose(L.masked_huber(pred, label, mask), torch.tensor(0.5))


def test_optimizer_gradient_guard_rejects_nan_without_changing_parameters():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    parameter.grad = torch.tensor([float("nan")])
    before = parameter.detach().clone()
    assert not _optimizer_grads_are_finite(optimizer)
    assert torch.equal(parameter.detach(), before)


def test_optimizer_gradient_guard_accepts_finite_or_missing_gradients():
    first = torch.nn.Parameter(torch.tensor([1.0]))
    second = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.Adam([first, second], lr=0.1)
    first.grad = torch.tensor([0.25])
    assert _optimizer_grads_are_finite(optimizer)


def test_exploration_schedule_uses_one_bounded_maturity_axis():
    kwargs = dict(
        entropy_initial=0.02,
        entropy_final=0.005,
        hold_fraction=0.75,
    )
    assert _exploration_schedule(0.0, **kwargs) == pytest.approx(0.02)
    assert _exploration_schedule(0.75, **kwargs) == pytest.approx(0.02)
    assert _exploration_schedule(0.875, **kwargs) == pytest.approx(0.0125)
    assert _exploration_schedule(1.0, **kwargs) == pytest.approx(0.005)
    assert _exploration_schedule(2.0, **kwargs) == pytest.approx(0.005)


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


def test_sparse_event_risk_is_normalized_separately_from_base_risk():
    geom = torch.zeros(1, 9)
    risk_pred = torch.zeros(1, 8)
    risk_label = torch.zeros(1, 8)
    risk_label[:, 2:] = 1.0
    risk_mask = torch.ones(1, 8)

    loss = L.aux_loss(
        geom,
        geom,
        torch.zeros_like(geom),
        risk_pred,
        risk_label,
        risk_mask,
    )
    assert loss == pytest.approx(L.EVENT_RISK_W * 0.5)

    hidden = L.aux_loss(
        geom,
        geom,
        torch.zeros_like(geom),
        risk_pred,
        risk_label,
        torch.cat([torch.ones(1, 2), torch.zeros(1, 6)], dim=-1),
    )
    assert hidden == 0.0


def test_aux_loss_rejects_a_stale_risk_dimension_contract():
    with pytest.raises(ValueError, match="shapes must match"):
        L.aux_loss(
            torch.zeros(1, 9),
            torch.zeros(1, 9),
            torch.ones(1, 9),
            torch.zeros(1, 8),
            torch.zeros(1, 2),
            torch.ones(1, 2),
        )
