import pytest

from products.taili.core.taili_curriculum import update_sample_weighted_ema


def test_progress_ema_ignores_small_batches():
    value, initialized, effective_alpha = update_sample_weighted_ema(
        0.7, 0.2, 7, alpha=0.01, min_samples=8, reference_samples=16, initialized=True
    )
    assert value == 0.7
    assert initialized
    assert effective_alpha == 0.0


def test_progress_ema_initializes_from_first_reliable_batch():
    value, initialized, effective_alpha = update_sample_weighted_ema(
        0.0, 0.68, 16, alpha=0.01, min_samples=8, reference_samples=16, initialized=False
    )
    assert value == 0.68
    assert initialized
    assert effective_alpha == pytest.approx(0.01)


def test_progress_ema_weights_partial_batches_less():
    full, _, full_alpha = update_sample_weighted_ema(
        0.6, 0.8, 16, alpha=0.01, min_samples=8, reference_samples=16, initialized=True
    )
    partial, _, partial_alpha = update_sample_weighted_ema(
        0.6, 0.8, 8, alpha=0.01, min_samples=8, reference_samples=16, initialized=True
    )
    assert 0.6 < partial < full < 0.8
    assert 0.0 < partial_alpha < full_alpha
