#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""研究调度器集成测试：验证真实的ExperimentQueue和GPUResourcePool实现。"""
import pytest
from pathlib import Path
from autotuner.product.research_scheduler import (
    ExperimentQueue,
    ExperimentJob,
    JobStatus,
    GPUResource,
    GPUResourcePool,
)
from autotuner.research.research_ledger import ExperimentPlan


class TestExperimentQueueIntegration:
    """ExperimentQueue真实实现的集成测试"""

    def test_enqueue_and_dequeue_basic(self, tmp_path: Path):
        """基本入队出队"""
        queue = ExperimentQueue()

        plan = ExperimentPlan(
            id="plan1",
            problem_statement="测试问题陈述",
            baseline_ref="baseline_v1",
        )

        job = ExperimentJob(
            job_id="job1",
            plan=plan,
            candidate_artifact=tmp_path / "candidate.tar",
            priority=1,
            status=JobStatus.PENDING,
        )

        queue.enqueue(job)
        assert queue.size() == 1

        dequeued = queue.dequeue()
        assert dequeued is not None
        assert dequeued.job_id == "job1"
        assert dequeued.status == JobStatus.QUEUED
        assert queue.size() == 0

    def test_priority_ordering(self, tmp_path: Path):
        """优先级排序：priority=0(baseline)优先于priority>0(candidate)"""
        queue = ExperimentQueue()

        plan = ExperimentPlan(
            id="plan1",
            problem_statement="测试问题",
            baseline_ref="baseline_v1",
        )

        # 创建不同优先级的任务
        candidate_job = ExperimentJob(
            job_id="candidate1",
            plan=plan,
            candidate_artifact=tmp_path / "c1.tar",
            priority=1,  # candidate
            status=JobStatus.PENDING,
        )

        baseline_job = ExperimentJob(
            job_id="baseline1",
            plan=plan,
            candidate_artifact=tmp_path / "baseline.tar",
            priority=0,  # baseline优先
            status=JobStatus.PENDING,
        )

        low_priority_job = ExperimentJob(
            job_id="candidate2",
            plan=plan,
            candidate_artifact=tmp_path / "c2.tar",
            priority=2,
            status=JobStatus.PENDING,
        )

        # 乱序入队
        queue.enqueue(candidate_job)
        queue.enqueue(low_priority_job)
        queue.enqueue(baseline_job)

        # 验证按优先级出队：baseline(0) > candidate1(1) > candidate2(2)
        job1 = queue.dequeue()
        assert job1 is not None
        assert job1.job_id == "baseline1"

        job2 = queue.dequeue()
        assert job2 is not None
        assert job2.job_id == "candidate1"

        job3 = queue.dequeue()
        assert job3 is not None
        assert job3.job_id == "candidate2"

    def test_peek_without_removal(self, tmp_path: Path):
        """peek查看队首但不移除"""
        queue = ExperimentQueue()

        plan = ExperimentPlan(
            id="plan1",
            problem_statement="测试问题",
            baseline_ref="baseline_v1",
        )

        job = ExperimentJob(
            job_id="job1",
            plan=plan,
            candidate_artifact=tmp_path / "c.tar",
            priority=1,
            status=JobStatus.PENDING,
        )

        queue.enqueue(job)

        # peek不移除
        peeked = queue.peek()
        assert peeked is not None
        assert peeked.job_id == "job1"
        assert queue.size() == 1

        # dequeue移除
        dequeued = queue.dequeue()
        assert dequeued is not None
        assert dequeued.job_id == "job1"
        assert queue.size() == 0

    def test_remove_job(self, tmp_path: Path):
        """取消队列中的任务"""
        queue = ExperimentQueue()

        plan = ExperimentPlan(
            id="plan1",
            problem_statement="测试问题",
            baseline_ref="baseline_v1",
        )

        job1 = ExperimentJob(
            job_id="job1",
            plan=plan,
            candidate_artifact=tmp_path / "c1.tar",
            priority=1,
            status=JobStatus.PENDING,
        )

        job2 = ExperimentJob(
            job_id="job2",
            plan=plan,
            candidate_artifact=tmp_path / "c2.tar",
            priority=1,
            status=JobStatus.PENDING,
        )

        queue.enqueue(job1)
        queue.enqueue(job2)
        assert queue.size() == 2

        # 移除job1
        removed = queue.remove("job1")
        assert removed is True
        assert queue.size() == 1

        # 剩余的是job2
        remaining = queue.dequeue()
        assert remaining is not None
        assert remaining.job_id == "job2"


class TestGPUResourcePoolIntegration:
    """GPUResourcePool真实实现的集成测试"""

    def test_static_gpu_discovery(self):
        """静态GPU列表配置"""
        pool = GPUResourcePool(
            discovery_mode="static",
            static_gpu_list=["gpu:0", "gpu:1", "gpu:2"],
        )

        available = pool.get_available()
        assert len(available) == 3
        assert all(gpu.is_available() for gpu in available)

    def test_allocate_and_release(self):
        """分配和释放GPU"""
        pool = GPUResourcePool(
            discovery_mode="static",
            static_gpu_list=["gpu:0", "gpu:1"],
        )

        # 分配1个GPU给实验exp1
        allocated_ids = pool.allocate(experiment_id="exp1", count=1)
        assert len(allocated_ids) == 1
        assert isinstance(allocated_ids[0], str)

        # 检查可用GPU数量减少
        available = pool.get_available()
        assert len(available) == 1

        # 释放GPU
        pool.release(gpu_ids=allocated_ids, experiment_id="exp1")

        # 检查GPU重新可用
        available = pool.get_available()
        assert len(available) == 2

    def test_allocate_multiple_gpus(self):
        """分配多个GPU给同一实验"""
        pool = GPUResourcePool(
            discovery_mode="static",
            static_gpu_list=["gpu:0", "gpu:1", "gpu:2", "gpu:3"],
        )

        # 分配2个GPU
        allocated_ids = pool.allocate(experiment_id="exp1", count=2)
        assert len(allocated_ids) == 2
        assert all(isinstance(gpu_id, str) for gpu_id in allocated_ids)

        # 可用GPU剩余2个
        available = pool.get_available()
        assert len(available) == 2

        # 验证分配记录
        allocation = pool.get_allocation("exp1")
        assert len(allocation) == 2
        assert set(allocation) == set(allocated_ids)

    def test_insufficient_resources(self):
        """资源不足时抛出异常"""
        pool = GPUResourcePool(
            discovery_mode="static",
            static_gpu_list=["gpu:0"],
        )

        # 只有1个GPU，尝试分配2个应该失败
        from autotuner.product.research_scheduler import ResourceUnavailable

        with pytest.raises(ResourceUnavailable):
            pool.allocate(experiment_id="exp1", count=2)

    def test_get_stats(self):
        """获取资源池统计信息"""
        pool = GPUResourcePool(
            discovery_mode="static",
            static_gpu_list=["gpu:0", "gpu:1", "gpu:2"],
        )

        stats = pool.get_stats()
        assert stats["total"] == 3
        assert stats["available"] == 3
        assert stats["allocated"] == 0

        # 分配后统计更新
        allocated_ids = pool.allocate(experiment_id="exp1", count=2)
        stats = pool.get_stats()
        assert stats["total"] == 3
        assert stats["available"] == 1
        assert stats["allocated"] == 2
