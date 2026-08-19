import torch

from products.taili.core.taili_obs import update_tick_history


def test_history_is_oldest_to_newest():
    history = torch.tensor([[[1.0], [2.0], [3.0]]])
    tick = torch.tensor([[4.0]])
    updated = update_tick_history(history, tick)
    assert updated.flatten().tolist() == [2.0, 3.0, 4.0]


def test_history_stride_mask_does_not_shift_unsampled_env():
    history = torch.tensor([[[1.0], [2.0]], [[5.0], [6.0]]])
    tick = torch.tensor([[3.0], [7.0]])
    updated = update_tick_history(history, tick, torch.tensor([True, False]))
    assert updated[0].flatten().tolist() == [2.0, 3.0]
    assert updated[1].flatten().tolist() == [5.0, 6.0]


def test_legacy_checkpoint_history_is_newest_first():
    history = torch.tensor([[[3.0], [2.0], [1.0]]])
    tick = torch.tensor([[4.0]])
    updated = update_tick_history(history, tick, order="newest_first")
    assert updated.flatten().tolist() == [4.0, 3.0, 2.0]
