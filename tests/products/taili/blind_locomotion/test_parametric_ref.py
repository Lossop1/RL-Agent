# Torch is optional in the test environment; importorskip must run first.
# ruff: noqa: E402
import pytest

torch = pytest.importorskip("torch")

from products.taili.blind_locomotion.parametric_ref import H0, X0, _foot_traj, flat_reference, foot_reference
from products.taili.core.taili_amp_reference import _foot_traj as amp_foot_traj


def test_quintic_swing_has_velocity_and_acceleration_continuity_at_endpoints():
    dtype = torch.float64
    eps = 1e-4
    s = torch.tensor([0.0, eps, 2.0 * eps, 1.0 - 2.0 * eps, 1.0 - eps, 1.0], dtype=dtype)
    p = 0.5 + 0.5 * s
    fx, _, fz = _foot_traj(
        p,
        torch.ones_like(p),
        torch.zeros_like(p),
        torch.full_like(p, 0.1),
    )

    # 机体系端点速度与支撑相后滑速度匹配；五次曲线同时让端点加速度趋近 0。
    ds = eps
    start_velocity = (fx[1] - fx[0]) / ds
    end_velocity = (fx[-1] - fx[-2]) / ds
    start_accel = (fx[2] - 2.0 * fx[1] + fx[0]) / (ds * ds)
    end_accel = (fx[-1] - 2.0 * fx[-2] + fx[-3]) / (ds * ds)
    assert start_velocity.item() == pytest.approx(-1.0, abs=2e-3)
    assert end_velocity.item() == pytest.approx(-1.0, abs=2e-3)
    assert abs(start_accel.item()) < 0.05
    assert abs(end_accel.item()) < 0.05

    # 垂直 bump 在离地和落地端点回到名义高度，且一阶变化趋近 0。
    assert fz[0].item() == pytest.approx(-H0)
    assert fz[-1].item() == pytest.approx(-H0)
    assert abs(((fz[1] - fz[0]) / ds).item()) < 1e-4
    assert abs(((fz[-1] - fz[-2]) / ds).item()) < 1e-4
    assert fx[0].item() == pytest.approx(X0 - 0.5)
    assert fx[-1].item() == pytest.approx(X0 + 0.5)


def test_live_and_amp_reference_use_the_same_foot_trajectory():
    phase = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)[:-1]
    stride_x = torch.linspace(-0.25, 0.35, phase.numel(), dtype=torch.float64)
    stride_y = torch.linspace(0.10, -0.10, phase.numel(), dtype=torch.float64)
    clearance = torch.linspace(0.06, 0.18, phase.numel(), dtype=torch.float64)

    live = _foot_traj(phase, stride_x, stride_y, clearance)
    amp = amp_foot_traj(phase, stride_x, stride_y, clearance)

    for live_axis, amp_axis in zip(live, amp):
        assert torch.allclose(live_axis, amp_axis)


def test_fast_foot_reference_matches_full_reference_positions():
    commands = torch.tensor([
        [0.5, 0.0, 0.0],
        [-0.4, 0.0, 0.0],
        [0.0, 0.3, 0.0],
        [0.0, 0.0, 0.6],
    ])
    times = torch.tensor([0.10, 0.20, 0.30, 0.40])
    expected = flat_reference(commands, times, iters=8)[-1]
    actual = foot_reference(commands, times)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_forward_and_backward_foot_references_are_sign_symmetric():
    times = torch.linspace(0.0, 0.70, 32)
    forward = foot_reference(torch.tensor([[0.4, 0.0, 0.0]]).repeat(32, 1), times)
    backward = foot_reference(torch.tensor([[-0.4, 0.0, 0.0]]).repeat(32, 1), times)
    midpoint = 0.5 * (forward[..., 0] + backward[..., 0])
    assert torch.allclose(midpoint, midpoint[:1].expand_as(midpoint), atol=1e-6, rtol=1e-6)
    assert torch.allclose(forward[..., 1:], backward[..., 1:], atol=1e-6, rtol=1e-6)


def test_low_speed_reference_reduces_clearance_with_stride():
    times = torch.linspace(0.0, 0.70, 128)
    slow = foot_reference(torch.tensor([[-0.15, 0.0, 0.0]]).repeat(128, 1), times)
    fast = foot_reference(torch.tensor([[-0.50, 0.0, 0.0]]).repeat(128, 1), times)
    slow_lift = slow[..., 2].amax() - slow[..., 2].amin()
    fast_lift = fast[..., 2].amax() - fast[..., 2].amin()
    assert slow_lift < fast_lift
