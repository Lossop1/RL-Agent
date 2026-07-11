"""Regression tests for terrain curriculum success/failure semantics."""
import pytest

torch = pytest.importorskip("torch")

from autotuner.taili_core.terrain_curriculum import compute_terrain_curriculum_moves


def _base(**over):
    data = dict(
        dist=torch.tensor([1.0]),
        cmd_mag=torch.tensor([0.5]),
        forward_dist=torch.tensor([0.6]),
        height_delta=torch.tensor([0.12]),
        valid_episode=torch.tensor([True]),
        terrain_curriculum_active=True,
        terrain_unlocked=True,
        terminal_now=torch.tensor([False]),
        base_h_local=torch.tensor([0.52]),
        upright_score=torch.tensor([0.98]),
        contact_count=torch.tensor([3.0]),
        body_wxy=torch.tensor([0.4]),
        v_along=torch.tensor([0.5]),
        max_episode_length_s=8.0,
        terrain_move_up_dist=4.0,
    )
    data.update(over)
    return compute_terrain_curriculum_moves(**data)


def test_stable_height_gain_can_advance():
    out = _base()
    assert bool(out["controlled_up"][0])
    assert bool(out["move_up"][0])
    assert not bool(out["move_down"][0])


def test_rushing_down_after_height_loss_is_failure_not_success():
    out = _base(
        height_delta=torch.tensor([-0.18]),
        v_along=torch.tensor([1.45]),
        base_h_local=torch.tensor([0.30]),
        body_wxy=torch.tensor([2.6]),
        terminal_now=torch.tensor([True]),
    )
    assert not bool(out["controlled_down"][0])
    assert not bool(out["move_up"][0])
    assert bool(out["failure_down"][0])
    assert bool(out["move_down"][0])


def test_distance_alone_does_not_advance_after_terminal_collapse():
    out = _base(
        dist=torch.tensor([5.5]),
        height_delta=torch.tensor([0.0]),
        terminal_now=torch.tensor([True]),
        base_h_local=torch.tensor([0.25]),
    )
    assert not bool(out["distance_success"][0])
    assert not bool(out["move_up"][0])
    assert bool(out["failure_down"][0])


def test_ineligible_terrain_does_not_change_level():
    out = _base(eligible_mask=torch.tensor([False]))
    assert not bool(out["controlled_up"][0])
    assert not bool(out["move_up"][0])
    assert not bool(out["move_down"][0])
    assert not bool(out["stable_end"][0])
