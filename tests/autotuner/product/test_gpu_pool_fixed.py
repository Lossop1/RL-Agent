"""GPU资源池单元测试 - 适配实际实现接口。

测试覆盖：
1. GPU自动探测
2. 单GPU分配
3. 多GPU分配
4. 资源不足返回空列表
5. GPU释放和再分配
6. 冲突检测（重复分配）
7. 优先级预留机制
"""

import pytest
from unittest.mock import Mock, patch
from autotuner.product.gpu_pool import GPUResourcePool, GPUResource, AllocationRecord


class MockGPUPool:
    """测试用GPU池模拟器 - 不依赖nvidia-smi。"""

    def __init__(self, num_gpus: int = 4):
        """创建模拟GPU池。

        Args:
            num_gpus: 模拟的GPU数量
        """
        self.pool = GPUResourcePool(auto_discover=False)
        # 手动填充GPU资源
        for i in range(num_gpus):
            gpu = GPUResource(
                gpu_id=str(i),
                name=f"Test GPU {i}",
                memory_total_mb=24000.0,
                memory_used_mb=100.0,
                utilization_pct=5.0,
            )
            self.pool._gpus[str(i)] = gpu

    def allocate(self, experiment_id: str, count: int, priority: int = 50) -> list[str]:
        """分配GPU。"""
        return self.pool.allocate(count, experiment_id, priority)

    def release(self, experiment_id: str):
        """释放GPU。"""
        self.pool.release(experiment_id)

    def get_available_count(self) -> int:
        """获取可用GPU数量。"""
        return self.pool.get_available_gpu_count()


class TestGPUResourcePool:
    """GPU资源池测试套件。"""

    def test_single_gpu_allocation(self):
        """测试分配单个GPU。"""
        pool = MockGPUPool(num_gpus=4)

        gpu_ids = pool.allocate(experiment_id="exp_1", count=1)

        assert len(gpu_ids) == 1
        assert gpu_ids[0] in ["0", "1", "2", "3"]
        assert pool.get_available_count() == 3

    def test_multi_gpu_allocation(self):
        """测试分配多个GPU（多卡训练）。"""
        pool = MockGPUPool(num_gpus=8)

        gpu_ids = pool.allocate(experiment_id="exp_multi", count=4)

        assert len(gpu_ids) == 4
        assert len(set(gpu_ids)) == 4  # 无重复
        assert pool.get_available_count() == 4

    def test_insufficient_resources(self):
        """测试资源不足时返回空列表。"""
        pool = MockGPUPool(num_gpus=2)

        # 先分配2个GPU
        gpu_ids = pool.allocate(experiment_id="exp_1", count=2)
        assert len(gpu_ids) == 2

        # 尝试再分配1个（应失败）
        gpu_ids_fail = pool.allocate(experiment_id="exp_2", count=1)
        assert len(gpu_ids_fail) == 0

    def test_release_and_reallocate(self):
        """测试释放GPU后可以重新分配。"""
        pool = MockGPUPool(num_gpus=4)

        # 分配全部4个GPU
        pool.allocate(experiment_id="exp_1", count=2)
        pool.allocate(experiment_id="exp_2", count=2)
        assert pool.get_available_count() == 0

        # 释放exp_1的2个GPU
        pool.release(experiment_id="exp_1")
        assert pool.get_available_count() == 2

        # 重新分配
        gpu_ids = pool.allocate(experiment_id="exp_3", count=2)
        assert len(gpu_ids) == 2

    def test_duplicate_allocation_ignored(self):
        """测试重复分配同一实验ID会被忽略（幂等性）。"""
        pool = MockGPUPool(num_gpus=4)

        gpu_ids_1 = pool.allocate(experiment_id="exp_dup", count=2)
        assert len(gpu_ids_1) == 2

        # 重复分配应返回空（已分配）
        gpu_ids_2 = pool.allocate(experiment_id="exp_dup", count=1)
        assert len(gpu_ids_2) == 0

        # 仍然只占用2个GPU
        assert pool.get_available_count() == 2

    def test_priority_reservation(self):
        """测试优先级预留机制（baseline优先）。"""
        pool = MockGPUPool(num_gpus=4)

        # 先分配3个给普通实验
        pool.allocate(experiment_id="regular_1", count=3, priority=50)
        assert pool.get_available_count() == 1

        # 高优先级实验应该能分配（即使只剩1个）
        gpu_ids_baseline = pool.allocate(experiment_id="baseline", count=1, priority=100)
        assert len(gpu_ids_baseline) == 1

    def test_release_nonexistent_experiment(self):
        """测试释放不存在的实验（应安全忽略）。"""
        pool = MockGPUPool(num_gpus=4)

        # 不应抛出异常
        pool.release(experiment_id="nonexistent")
        assert pool.get_available_count() == 4

    def test_get_allocation_info(self):
        """测试查询分配信息。"""
        pool = MockGPUPool(num_gpus=4)

        gpu_ids = pool.allocate(experiment_id="exp_query", count=2)

        allocation = pool.pool.get_allocation(experiment_id="exp_query")
        assert allocation is not None
        assert allocation.experiment_id == "exp_query"
        assert len(allocation.gpu_ids) == 2
        assert set(allocation.gpu_ids) == set(gpu_ids)

    def test_all_gpus_exhausted(self):
        """测试所有GPU耗尽后的行为。"""
        pool = MockGPUPool(num_gpus=3)

        pool.allocate(experiment_id="exp_a", count=3)
        assert pool.get_available_count() == 0

        # 新分配应失败
        gpu_ids = pool.allocate(experiment_id="exp_b", count=1)
        assert len(gpu_ids) == 0


class TestGPUDiscovery:
    """GPU探测测试。"""

    @patch("subprocess.run")
    def test_local_gpu_discovery(self, mock_run):
        """测试本地nvidia-smi探测。"""
        # 模拟nvidia-smi输出
        mock_run.return_value = Mock(
            returncode=0,
            stdout="0, NVIDIA RTX 4090, 1024, 24576, 15\n1, NVIDIA RTX 4090, 512, 24576, 5\n"
        )

        pool = GPUResourcePool(auto_discover=True)
        gpus = pool.discover_gpus()

        assert len(gpus) == 2
        assert gpus[0].gpu_id == "0"
        assert gpus[0].name == "NVIDIA RTX 4090"
        assert gpus[0].memory_used_mb == 1024.0
        assert gpus[1].gpu_id == "1"

    @patch("subprocess.run")
    def test_discovery_failure_returns_empty(self, mock_run):
        """测试探测失败时返回空列表。"""
        mock_run.return_value = Mock(returncode=1, stdout="")

        pool = GPUResourcePool(auto_discover=True)
        gpus = pool.discover_gpus()

        assert len(gpus) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
