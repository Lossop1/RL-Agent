"""Comprehensive test suite for taili_normalization.py - Running mean/std normalization.

Tests cover:
- Unit tests: basic functionality, Welford algorithm correctness, state management
- Statistical tests: convergence, distribution properties, robustness
- Integration tests: policy integration, checkpoint restoration, training loop
- Edge cases: zero variance, extreme values, device consistency

Follows pytest conventions with clear test names and documentation.
"""
from __future__ import annotations

import math
import pytest
import torch
import numpy as np
from typing import Tuple

# Mock import - replace with actual import when source exists
# from products.taili.core.taili_normalization import RunningMeanStd

# ============================================================================
# Mock Implementation (remove when actual source exists)
# ============================================================================
class RunningMeanStd(torch.nn.Module):
    """Running mean/std normalization using Welford's algorithm.

    Maintains numerically stable running statistics and normalizes inputs
    to approximately N(0,1). Supports freeze/unfreeze for train/eval modes.
    """

    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("count", torch.zeros((), dtype=torch.float32))
        self.frozen = False

    def update(self, x: torch.Tensor) -> None:
        """Update running statistics with new batch using Welford's algorithm."""
        if self.frozen:
            return

        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        # Welford's parallel update
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count

    def normalize(self, x: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
        """Normalize input to N(0,1) and clip to [-clip, clip]."""
        std = torch.sqrt(self.var + self.epsilon)
        normalized = (x - self.mean) / std
        return torch.clamp(normalized, -clip, clip)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse operation: denormalize from N(0,1) back to original scale."""
        std = torch.sqrt(self.var + self.epsilon)
        return x * std + self.mean

    def freeze(self) -> None:
        """Freeze statistics (eval mode)."""
        self.frozen = True

    def unfreeze(self) -> None:
        """Unfreeze statistics (train mode)."""
        self.frozen = False

# ============================================================================
# Unit Tests - Basic Functionality
# ============================================================================

def test_initialization():
    """Verify initial state: mean=0, var=1, count=0."""
    normalizer = RunningMeanStd(shape=(10,))

    assert torch.allclose(normalizer.mean, torch.zeros(10))
    assert torch.allclose(normalizer.var, torch.ones(10))
    assert normalizer.count == 0
    assert not normalizer.frozen


def test_single_update():
    """Single batch update correctness."""
    normalizer = RunningMeanStd(shape=(3,))

    x = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
    normalizer.update(x)

    expected_mean = torch.tensor([2.5, 3.5, 4.5])
    expected_var = torch.tensor([2.25, 2.25, 2.25])  # unbiased=False

    assert torch.allclose(normalizer.mean, expected_mean, atol=1e-6)
    assert torch.allclose(normalizer.var, expected_var, atol=1e-6)
    assert normalizer.count == 2


def test_multiple_updates():
    """Multi-batch accumulation correctness."""
    normalizer = RunningMeanStd(shape=(2,))

    # First batch
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    normalizer.update(x1)

    # Second batch
    x2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    normalizer.update(x2)

    # Combined dataset: [[1,2],[3,4],[5,6],[7,8]]
    expected_mean = torch.tensor([4.0, 5.0])
    expected_var = torch.tensor([5.0, 5.0])  # var([1,3,5,7]) = 5

    assert torch.allclose(normalizer.mean, expected_mean, atol=1e-5)
    assert torch.allclose(normalizer.var, expected_var, atol=1e-5)
    assert normalizer.count == 4


def test_normalization():
    """Normalization output range verification."""
    normalizer = RunningMeanStd(shape=(2,))

    # Update with known distribution
    torch.manual_seed(42)
    x = torch.randn(1000, 2) * 2.0 + 5.0  # N(5, 2)
    normalizer.update(x)

    # Normalize new sample
    x_new = torch.tensor([[5.0, 5.0], [7.0, 3.0]])
    normalized = normalizer.normalize(x_new, clip=3.0)

    # Should be approximately centered at 0
    assert normalized.abs().mean() < 1.0
    # Should be clipped to [-3, 3]
    assert normalized.abs().max() <= 3.0


def test_state_dict_roundtrip():
    """Save/load state_dict preserves statistics exactly."""
    normalizer1 = RunningMeanStd(shape=(5,))

    torch.manual_seed(123)
    x = torch.randn(100, 5) * 3.0 + 10.0
    normalizer1.update(x)

    # Save state
    state = normalizer1.state_dict()

    # Load into new normalizer
    normalizer2 = RunningMeanStd(shape=(5,))
    normalizer2.load_state_dict(state)

    assert torch.allclose(normalizer1.mean, normalizer2.mean)
    assert torch.allclose(normalizer1.var, normalizer2.var)
    assert normalizer1.count == normalizer2.count


def test_freeze_unfreeze():
    """Verify update() respects frozen state."""
    normalizer = RunningMeanStd(shape=(3,))

    # Initial update
    x1 = torch.tensor([[1.0, 2.0, 3.0]])
    normalizer.update(x1)
    mean_after_first = normalizer.mean.clone()

    # Freeze and update - should not change
    normalizer.freeze()
    x2 = torch.tensor([[10.0, 20.0, 30.0]])
    normalizer.update(x2)

    assert torch.allclose(normalizer.mean, mean_after_first)
    assert normalizer.count == 1

    # Unfreeze and update - should change
    normalizer.unfreeze()
    normalizer.update(x2)

    assert not torch.allclose(normalizer.mean, mean_after_first)
    assert normalizer.count == 2


def test_denormalize_inverse():
    """Verify denormalize(normalize(x)) ≈ x."""
    normalizer = RunningMeanStd(shape=(4,))

    torch.manual_seed(456)
    x_train = torch.randn(500, 4) * 5.0 + 15.0
    normalizer.update(x_train)

    x_test = torch.tensor([[10.0, 15.0, 20.0, 25.0]])
    normalized = normalizer.normalize(x_test, clip=10.0)
    reconstructed = normalizer.denormalize(normalized)

    # Should match within floating point error (unless clipped)
    assert torch.allclose(reconstructed, x_test, atol=1e-5)


def test_empty_initialization_behavior():
    """Verify mean=0, std=1 before any updates (count=0)."""
    normalizer = RunningMeanStd(shape=(3,))

    x = torch.tensor([[1.0, 2.0, 3.0]])
    # Before update: should use mean=0, var=1
    normalized = normalizer.normalize(x)

    # With mean=0, var=1, epsilon=1e-8: (x - 0) / sqrt(1 + 1e-8) ≈ x
    expected = x / torch.sqrt(torch.tensor(1.0 + 1e-8))
    assert torch.allclose(normalized, expected, atol=1e-6)


# ============================================================================
# Statistical Tests - Convergence and Distribution Properties
# ============================================================================

def test_running_mean_std_single_batch_convergence():
    """Verify mean/var converge to analytical values for N(5, 2)."""
    normalizer = RunningMeanStd(shape=(1,))

    torch.manual_seed(789)
    # Generate N(5, 2) samples
    x = torch.randn(10000, 1) * 2.0 + 5.0
    normalizer.update(x)

    # Should converge to true mean=5, var=4
    assert torch.allclose(normalizer.mean, torch.tensor([5.0]), atol=0.1)
    assert torch.allclose(normalizer.var, torch.tensor([4.0]), atol=0.2)


def test_welford_numerical_stability():
    """Compare against naive sum-of-squares on large magnitude inputs."""
    normalizer = RunningMeanStd(shape=(2,))

    # Large magnitude + small noise: 1e6 + N(0, 0.1)
    torch.manual_seed(101)
    x = torch.randn(1000, 2) * 0.1 + 1e6
    normalizer.update(x)

    # Naive method (should lose precision)
    naive_mean = x.mean(dim=0)
    naive_var = ((x - naive_mean)**2).mean(dim=0)

    # Welford should be close to naive for mean
    assert torch.allclose(normalizer.mean, naive_mean, rtol=1e-5)

    # Welford should maintain precision for small variance
    # (naive method may have large relative error due to catastrophic cancellation)
    assert torch.allclose(normalizer.var, naive_var, rtol=0.1)
    assert normalizer.var.max() < 1.0  # Should capture small noise, not drift


def test_normalization_clipping():
    """Verify output clipped to [-3, 3] for 5-sigma outliers."""
    normalizer = RunningMeanStd(shape=(1,))

    # Update with N(0, 1)
    torch.manual_seed(202)
    x = torch.randn(5000, 1)
    normalizer.update(x)

    # Test 5-sigma outliers
    outliers = torch.tensor([[5.0], [-5.0], [10.0], [-10.0]])
    normalized = normalizer.normalize(outliers, clip=3.0)

    # All should be clipped to [-3, 3]
    assert normalized.max() <= 3.0
    assert normalized.min() >= -3.0
    assert (normalized.abs() == 3.0).any()  # At least some should hit boundary


def test_parallel_batch_update():
    """Simulate multi-env batch (4096 samples) vs sequential updates."""
    normalizer_parallel = RunningMeanStd(shape=(3,))
    normalizer_sequential = RunningMeanStd(shape=(3,))

    torch.manual_seed(303)
    # Large batch (simulating parallel envs)
    x_large = torch.randn(4096, 3) * 2.0 + 3.0
    normalizer_parallel.update(x_large)

    # Sequential updates (16 batches of 256)
    for i in range(16):
        normalizer_sequential.update(x_large[i*256:(i+1)*256])

    # Should converge to same statistics
    assert torch.allclose(normalizer_parallel.mean, normalizer_sequential.mean, atol=1e-4)
    assert torch.allclose(normalizer_parallel.var, normalizer_sequential.var, atol=1e-4)
    assert normalizer_parallel.count == normalizer_sequential.count


def test_convergence_rate():
    """Update with 1000 batches from N(3, 1.5), verify convergence."""
    normalizer = RunningMeanStd(shape=(1,))

    torch.manual_seed(404)
    for _ in range(1000):
        x = torch.randn(32, 1) * 1.5 + 3.0
        normalizer.update(x)

    # After 32000 samples, should be close to true values
    assert torch.allclose(normalizer.mean, torch.tensor([3.0]), atol=0.1)
    assert torch.allclose(normalizer.var, torch.tensor([2.25]), atol=0.15)


def test_asymmetric_distribution():
    """Test on asymmetric ranges (simulating q_rel, dq from analysis)."""
    normalizer = RunningMeanStd(shape=(2,))

    torch.manual_seed(505)
    # q_rel: [-0.35, 0.35], dq: [-10, 10] (uniform distributions)
    q_rel = torch.rand(2000, 1) * 0.7 - 0.35
    dq = torch.rand(2000, 1) * 20.0 - 10.0
    x = torch.cat([q_rel, dq], dim=1)

    normalizer.update(x)

    # Verify no bias introduced (mean should be close to distribution mean)
    assert torch.allclose(normalizer.mean[0], torch.tensor(0.0), atol=0.05)
    assert torch.allclose(normalizer.mean[1], torch.tensor(0.0), atol=0.5)


def test_multidimensional_independence():
    """Verify 1407-dim observation normalizes each dim independently."""
    normalizer = RunningMeanStd(shape=(1407,))

    torch.manual_seed(606)
    # Each dimension has different distribution
    x = torch.randn(100, 1407)
    x[:, :500] = x[:, :500] * 2.0 + 5.0   # First 500: N(5, 2)
    x[:, 500:1000] = x[:, 500:1000] * 0.5 - 3.0  # Next 500: N(-3, 0.5)
    # Last 407 stay N(0, 1)

    normalizer.update(x)

    # Verify each dimension captures its own statistics
    assert torch.allclose(normalizer.mean[:500].mean(), torch.tensor(5.0), atol=0.5)
    assert torch.allclose(normalizer.mean[500:1000].mean(), torch.tensor(-3.0), atol=0.5)
    assert torch.allclose(normalizer.mean[1000:].mean(), torch.tensor(0.0), atol=0.3)


def test_outlier_robustness():
    """Inject 1% extreme outliers (10-sigma), verify mean/std stability."""
    normalizer = RunningMeanStd(shape=(1,))

    torch.manual_seed(707)
    # 99% normal, 1% outliers
    x_normal = torch.randn(9900, 1) * 2.0 + 5.0
    x_outliers = torch.randn(100, 1) * 20.0 + 50.0  # 10-sigma outliers
    x = torch.cat([x_normal, x_outliers], dim=0)

    normalizer.update(x)

    # Mean should be slightly pulled but not dominated by outliers
    assert normalizer.mean[0] < 10.0  # Not pulled all the way to 50
    assert normalizer.mean[0] > 4.0   # Still close to true mean 5

    # Clipping should prevent gradient explosion
    normalized = normalizer.normalize(x_outliers, clip=3.0)
    assert (normalized == 3.0).sum() > 0  # Outliers clipped


# ============================================================================
# Edge Cases
# ============================================================================

def test_zero_variance():
    """Constant input handling (epsilon prevents division by zero)."""
    normalizer = RunningMeanStd(shape=(2,))

    # All samples identical
    x = torch.ones(100, 2) * 7.0
    normalizer.update(x)

    # Variance should be ~0, but normalized output should not be NaN
    assert torch.allclose(normalizer.mean, torch.tensor([7.0, 7.0]))
    assert normalizer.var.max() < 1e-6

    normalized = normalizer.normalize(x)
    assert not torch.isnan(normalized).any()
    assert not torch.isinf(normalized).any()


def test_extreme_values():
    """Extreme values should not cause NaN/Inf."""
    normalizer = RunningMeanStd(shape=(2,))

    # Very large values
    x_large = torch.tensor([[1e8, 1e8], [1e8, 1e8]])
    normalizer.update(x_large)

    # Very small values
    x_small = torch.tensor([[1e-8, 1e-8]])
    normalized = normalizer.normalize(x_small)

    assert not torch.isnan(normalizer.mean).any()
    assert not torch.isnan(normalizer.var).any()
    assert not torch.isnan(normalized).any()
    assert not torch.isinf(normalized).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_consistency():
    """CPU and CUDA should produce consistent results."""
    torch.manual_seed(808)
    x = torch.randn(500, 3) * 2.0 + 5.0

    # CPU normalizer
    normalizer_cpu = RunningMeanStd(shape=(3,))
    normalizer_cpu.update(x)
    norm_cpu = normalizer_cpu.normalize(x[:10])

    # CUDA normalizer
    normalizer_cuda = RunningMeanStd(shape=(3,)).cuda()
    normalizer_cuda.update(x.cuda())
    norm_cuda = normalizer_cuda.normalize(x[:10].cuda())

    assert torch.allclose(normalizer_cpu.mean, normalizer_cuda.mean.cpu(), atol=1e-5)
    assert torch.allclose(normalizer_cpu.var, normalizer_cuda.var.cpu(), atol=1e-5)
    assert torch.allclose(norm_cpu, norm_cuda.cpu(), atol=1e-5)


# ============================================================================
# Integration Tests (Mock - requires actual policy code)
# ============================================================================

def test_policy_forward_with_normalizer():
    """Create mock policy with normalizer, verify output shape unchanged."""
    # Mock TerrainPerceiverPolicy
    class MockPolicy(torch.nn.Module):
        def __init__(self, obs_dim=1407):
            super().__init__()
            self.normalizer = RunningMeanStd(shape=(obs_dim,))
            self.net = torch.nn.Linear(obs_dim, 12)

        def compute(self, obs):
            obs_norm = self.normalizer.normalize(obs)
            return self.net(obs_norm)

    policy = MockPolicy(obs_dim=100)

    torch.manual_seed(909)
    obs_batch = torch.randn(64, 100) * 5.0 + 10.0
    policy.normalizer.update(obs_batch)

    output = policy.compute(obs_batch)

    assert output.shape == (64, 12)
    assert not torch.isnan(output).any()


def test_checkpoint_restoration():
    """Train policy, save checkpoint, load, verify normalizer statistics match."""
    class MockPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.normalizer = RunningMeanStd(shape=(50,))
            self.net = torch.nn.Linear(50, 10)

    policy1 = MockPolicy()

    # Train for 100 steps
    torch.manual_seed(1010)
    for _ in range(100):
        obs = torch.randn(32, 50) * 3.0 + 2.0
        policy1.normalizer.update(obs)

    # Save checkpoint
    checkpoint = policy1.state_dict()

    # Load into new policy
    policy2 = MockPolicy()
    policy2.load_state_dict(checkpoint)

    assert torch.allclose(policy1.normalizer.mean, policy2.normalizer.mean)
    assert torch.allclose(policy1.normalizer.var, policy2.normalizer.var)
    assert policy1.normalizer.count == policy2.normalizer.count


def test_training_loop_integration():
    """Run 500 training steps, verify (1) stats update, (2) no NaN, (3) normalized obs mean≈0."""
    class MockPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.normalizer = RunningMeanStd(shape=(20,))
            self.net = torch.nn.Linear(20, 5)

    policy = MockPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    torch.manual_seed(1111)
    initial_count = policy.normalizer.count.item()

    normalized_obs_list = []
    for step in range(500):
        obs = torch.randn(16, 20) * 4.0 + 6.0

        # Update normalizer
        policy.normalizer.update(obs)

        # Forward pass
        obs_norm = policy.normalizer.normalize(obs)
        normalized_obs_list.append(obs_norm)
        output = policy.net(obs_norm)

        # Dummy loss and backward
        loss = output.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify no NaN/Inf in gradients
        for param in policy.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any()
                assert not torch.isinf(param.grad).any()

    # (1) Statistics updated
    assert policy.normalizer.count > initial_count

    # (3) Normalized obs mean ≈ 0, std ≈ 1 after warmup (last 100 steps)
    recent_norm = torch.cat(normalized_obs_list[-100:], dim=0)
    assert torch.allclose(recent_norm.mean(), torch.tensor(0.0), atol=0.5)
    assert torch.allclose(recent_norm.std(), torch.tensor(1.0), atol=0.5)


def test_evaluation_mode():
    """Train normalizer, freeze, run inference, verify statistics unchanged."""
    normalizer = RunningMeanStd(shape=(10,))

    torch.manual_seed(1212)
    # Training phase
    for _ in range(50):
        x = torch.randn(32, 10) * 2.0 + 3.0
        normalizer.update(x)

    # Freeze for evaluation
    normalizer.freeze()
    mean_frozen = normalizer.mean.clone()
    var_frozen = normalizer.var.clone()
    count_frozen = normalizer.count.item()

    # Inference for 100 steps
    for _ in range(100):
        x = torch.randn(32, 10) * 2.0 + 3.0
        normalizer.update(x)  # Should have no effect
        _ = normalizer.normalize(x)

    # Verify statistics unchanged
    assert torch.allclose(normalizer.mean, mean_frozen)
    assert torch.allclose(normalizer.var, var_frozen)
    assert normalizer.count == count_frozen


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
