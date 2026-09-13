"""Unit tests for GPUResourcePool.

Tests cover:
1. Single GPU allocation
2. Multi-GPU allocation (multi-card training)
3. Insufficient GPU resources returns None
4. GPU release and reallocation
5. Conflict detection (duplicate allocation)
6. Priority reservation mechanism
"""

import pytest
from autotuner.product.gpu_pool import GPUResourcePool, AllocationRecord


class TestGPUResourcePool:
    """Test suite for GPU resource pool management."""

    def test_single_gpu_allocation(self):
        """Test allocating a single GPU."""
        pool = GPUResourcePool(total_gpus=4)

        gpu_ids = pool.allocate(task_id="task_1", num_gpus=1)

        assert gpu_ids is not None
        assert len(gpu_ids) == 1
        assert 0 <= gpu_ids[0] < 4
        assert pool.get_available_count() == 3

    def test_multi_gpu_allocation(self):
        """Test allocating multiple GPUs for multi-card training."""
        pool = GPUResourcePool(total_gpus=8)

        # Allocate 4 GPUs for distributed training
        gpu_ids = pool.allocate(task_id="distributed_task", num_gpus=4)

        assert gpu_ids is not None
        assert len(gpu_ids) == 4
        assert len(set(gpu_ids)) == 4  # All unique
        assert all(0 <= gpu_id < 8 for gpu_id in gpu_ids)
        assert pool.get_available_count() == 4

    def test_insufficient_resources_returns_none(self):
        """Test that allocation returns None when insufficient GPUs available."""
        pool = GPUResourcePool(total_gpus=2)

        # Allocate all GPUs
        gpu_ids_1 = pool.allocate(task_id="task_1", num_gpus=2)
        assert gpu_ids_1 is not None

        # Try to allocate more - should fail
        gpu_ids_2 = pool.allocate(task_id="task_2", num_gpus=1)
        assert gpu_ids_2 is None
        assert pool.get_available_count() == 0

    def test_release_and_reallocate(self):
        """Test releasing GPUs and reallocating them."""
        pool = GPUResourcePool(total_gpus=4)

        # Allocate GPUs
        gpu_ids_1 = pool.allocate(task_id="task_1", num_gpus=2)
        assert gpu_ids_1 is not None
        assert pool.get_available_count() == 2

        # Release GPUs
        released = pool.release(task_id="task_1")
        assert released is True
        assert pool.get_available_count() == 4

        # Reallocate same number
        gpu_ids_2 = pool.allocate(task_id="task_2", num_gpus=2)
        assert gpu_ids_2 is not None
        assert len(gpu_ids_2) == 2
        assert pool.get_available_count() == 2

    def test_duplicate_allocation_conflict(self):
        """Test conflict detection when allocating to same task_id twice."""
        pool = GPUResourcePool(total_gpus=4)

        # First allocation succeeds
        gpu_ids_1 = pool.allocate(task_id="task_1", num_gpus=1)
        assert gpu_ids_1 is not None

        # Duplicate allocation fails
        gpu_ids_2 = pool.allocate(task_id="task_1", num_gpus=1)
        assert gpu_ids_2 is None

    def test_priority_reservation_basic(self):
        """Test priority mechanism for task allocation."""
        pool = GPUResourcePool(total_gpus=4)

        # Allocate with different priorities
        low_priority = pool.allocate(task_id="low", num_gpus=2, priority=1)
        high_priority = pool.allocate(task_id="high", num_gpus=2, priority=10)

        assert low_priority is not None
        assert high_priority is not None

        # Verify allocations are tracked with priority
        allocation_low = pool.get_allocation("low")
        allocation_high = pool.get_allocation("high")
        assert allocation_low is not None
        assert allocation_high is not None

    def test_release_nonexistent_task(self):
        """Test releasing a task that doesn't exist."""
        pool = GPUResourcePool(total_gpus=4)

        released = pool.release(task_id="nonexistent")
        assert released is False

    def test_get_allocation(self):
        """Test retrieving allocation for a task."""
        pool = GPUResourcePool(total_gpus=4)

        gpu_ids = pool.allocate(task_id="task_1", num_gpus=2)
        assert gpu_ids is not None

        retrieved = pool.get_allocation("task_1")
        assert retrieved is not None
        assert retrieved == gpu_ids
        assert retrieved is not gpu_ids  # Should be a copy

    def test_get_allocation_nonexistent(self):
        """Test getting allocation for nonexistent task."""
        pool = GPUResourcePool(total_gpus=4)

        allocation = pool.get_allocation("nonexistent")
        assert allocation is None

    def test_is_available(self):
        """Test checking if specific GPU is available."""
        pool = GPUResourcePool(total_gpus=4)

        # All GPUs initially available
        assert pool.is_available(0) is True
        assert pool.is_available(3) is True
        assert pool.is_available(4) is False  # Out of range

        # Allocate GPU 0
        gpu_ids = pool.allocate(task_id="task_1", num_gpus=1, preferred_ids=[0])
        assert gpu_ids == [0]

        # GPU 0 no longer available
        assert pool.is_available(0) is False
        assert pool.is_available(1) is True

    def test_preferred_gpu_ids(self):
        """Test allocating preferred GPU IDs."""
        pool = GPUResourcePool(total_gpus=8)

        # Request specific GPUs
        gpu_ids = pool.allocate(task_id="task_1", num_gpus=2, preferred_ids=[3, 5])

        assert gpu_ids is not None
        assert 3 in gpu_ids
        assert 5 in gpu_ids

    def test_preferred_ids_partially_available(self):
        """Test when some preferred IDs are unavailable."""
        pool = GPUResourcePool(total_gpus=4)

        # Occupy GPU 1
        pool.allocate(task_id="task_1", num_gpus=1, preferred_ids=[1])

        # Request GPUs 1 and 2, but 1 is occupied
        gpu_ids = pool.allocate(task_id="task_2", num_gpus=2, preferred_ids=[1, 2])

        assert gpu_ids is not None
        assert len(gpu_ids) == 2
        assert 2 in gpu_ids  # Should get 2
        assert 1 not in gpu_ids  # Should not get 1 (occupied)

    def test_invalid_num_gpus(self):
        """Test allocation with invalid GPU count."""
        pool = GPUResourcePool(total_gpus=4)

        # Zero GPUs
        gpu_ids_zero = pool.allocate(task_id="task_1", num_gpus=0)
        assert gpu_ids_zero is None

        # More than available
        gpu_ids_over = pool.allocate(task_id="task_2", num_gpus=5)
        assert gpu_ids_over is None

    def test_concurrent_allocations(self):
        """Test multiple concurrent allocations."""
        pool = GPUResourcePool(total_gpus=8)

        allocations = []
        for i in range(4):
            gpu_ids = pool.allocate(task_id=f"task_{i}", num_gpus=2)
            assert gpu_ids is not None
            allocations.append(gpu_ids)

        # All 8 GPUs should be allocated
        assert pool.get_available_count() == 0

        # All allocations should be unique
        all_gpus = [gpu for alloc in allocations for gpu in alloc]
        assert len(all_gpus) == 8
        assert len(set(all_gpus)) == 8

    def test_partial_release_and_reallocate(self):
        """Test releasing some tasks and reallocating."""
        pool = GPUResourcePool(total_gpus=6)

        # Allocate to 3 tasks
        pool.allocate(task_id="task_1", num_gpus=2)
        pool.allocate(task_id="task_2", num_gpus=2)
        pool.allocate(task_id="task_3", num_gpus=2)
        assert pool.get_available_count() == 0

        # Release middle task
        pool.release(task_id="task_2")
        assert pool.get_available_count() == 2

        # Reallocate
        gpu_ids = pool.allocate(task_id="task_4", num_gpus=2)
        assert gpu_ids is not None
        assert pool.get_available_count() == 0

    def test_allocation_immutability(self):
        """Test that returned GPU IDs cannot affect internal state."""
        pool = GPUResourcePool(total_gpus=4)

        gpu_ids = pool.allocate(task_id="task_1", num_gpus=2)
        assert gpu_ids is not None

        original_ids = gpu_ids.copy()

        # Modify returned list
        gpu_ids.append(999)

        # Should not affect stored allocation
        retrieved = pool.get_allocation("task_1")
        assert retrieved == original_ids
        assert 999 not in retrieved


@pytest.fixture
def pool():
    """Fixture providing a GPU pool with 4 GPUs."""
    return GPUResourcePool(total_gpus=4)


class TestGPUResourcePoolFixture:
    """Tests using pytest fixtures."""

    def test_with_fixture(self, pool):
        """Test using fixture."""
        gpu_ids = pool.allocate(task_id="test", num_gpus=1)
        assert gpu_ids is not None
        assert pool.get_available_count() == 3
