"""并行实验调度系统。

提供GPU资源池管理、实验队列调度、并发执行协调，支持baseline优先策略和错误隔离。
核心组件：
- GPUResourcePool: GPU资源分配和释放
- ExperimentQueue: 优先级队列管理
- ResearchScheduler: 并行调度协调器
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol
from collections.abc import Mapping, Sequence

from autotuner.research.research_ledger import ExperimentPlan
from autotuner.research.research_supervisor import (
    BackendHandle,
    BackendStatus,
    ExperimentBackend,
    ExperimentExecution,
    ResourceLease,
    ResourceLeaseStore,
)


def _now_epoch() -> float:
    return time.time()


class JobStatus(str, Enum):
    """实验作业状态枚举"""
    PENDING = "PENDING"  # 已创建，未入队
    QUEUED = "QUEUED"  # 在队列中等待
    RUNNING = "RUNNING"  # 正在执行
    COMPLETED = "COMPLETED"  # 成功完成
    FAILED = "FAILED"  # 执行失败
    CANCELLED = "CANCELLED"  # 用户取消


@dataclass
class GPUResource:
    """单个GPU资源的状态和属性"""
    id: str  # 如 'gpu:0'
    device_index: int
    memory_total_mb: int = 0
    allocated_to: str | None = None  # experiment_id
    allocated_at: float | None = None

    def is_available(self) -> bool:
        """检查GPU是否可用"""
        return self.allocated_to is None


@dataclass
class ExperimentJob:
    """单个实验的完整信息：计划、候选、优先级、状态"""
    job_id: str
    plan: ExperimentPlan
    candidate_artifact: Path
    priority: int = 1  # 0=baseline(最高), >0=candidate
    status: JobStatus = JobStatus.PENDING
    gpu_allocation: list[str] = field(default_factory=list)
    workspace: Path | None = None
    backend_handle: BackendHandle | None = None
    created_at: float = field(default_factory=_now_epoch)
    started_at: float | None = None
    completed_at: float | None = None
    error_message: str = ""
    execution_result: ExperimentExecution | None = None
    actor: str = "scheduler"


@dataclass
class ExperimentBatch:
    """批量实验提交的分组（baseline + candidates）"""
    batch_id: str
    baseline_job_id: str
    candidate_job_ids: list[str]
    created_at: float = field(default_factory=_now_epoch)
    wait_for_baseline: bool = False

    @property
    def all_job_ids(self) -> list[str]:
        return [self.baseline_job_id] + self.candidate_job_ids


@dataclass
class SchedulerState:
    """调度器运行时状态快照"""
    running: bool
    max_parallel: int
    jobs_pending: int
    jobs_running: int
    jobs_completed: int
    jobs_failed: int
    available_gpus: int
    allocated_gpus: int
    last_schedule_time: float
    active_job_ids: list[str]


class ResourceUnavailable(Exception):
    """资源不足异常"""
    pass


class ResourceExhausted(Exception):
    """资源池和队列已满异常"""
    pass


class GPUResourcePool:
    """追踪可用GPU资源，分配和释放GPU给实验，避免资源冲突"""

    def __init__(
        self,
        discovery_mode: Literal["local", "static"] = "local",
        static_gpu_list: list[str] | None = None,
    ) -> None:
        self.discovery_mode = discovery_mode
        self.static_gpu_list = static_gpu_list or []
        self._gpus: dict[str, GPUResource] = {}
        self._lock = threading.RLock()
        self._discover_gpus()

    def _discover_gpus(self) -> None:
        """探测可用GPU（本地nvidia-smi或静态配置）"""
        with self._lock:
            if self.discovery_mode == "static":
                for idx, gpu_id in enumerate(self.static_gpu_list):
                    self._gpus[gpu_id] = GPUResource(
                        id=gpu_id,
                        device_index=idx,
                        memory_total_mb=0,
                    )
            elif self.discovery_mode == "local":
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0:
                        for line in result.stdout.strip().split("\n"):
                            if line.strip():
                                parts = line.split(",")
                                idx = int(parts[0].strip())
                                mem = int(parts[1].strip())
                                gpu_id = f"gpu:{idx}"
                                self._gpus[gpu_id] = GPUResource(
                                    id=gpu_id,
                                    device_index=idx,
                                    memory_total_mb=mem,
                                )
                except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
                    # Fallback: 假设有gpu:0
                    self._gpus["gpu:0"] = GPUResource(id="gpu:0", device_index=0, memory_total_mb=0)

    def allocate(
        self,
        count: int,
        experiment_id: str,
        timeout: float = 0.0,
    ) -> list[str]:
        """为实验分配指定数量的GPU

        Args:
            count: 需要的GPU数量
            experiment_id: 实验ID
            timeout: 等待超时（秒），0表示不等待

        Returns:
            分配的GPU ID列表，如['gpu:0', 'gpu:1']

        Raises:
            ResourceUnavailable: 可用GPU不足
        """
        deadline = _now_epoch() + timeout if timeout > 0 else _now_epoch()

        while True:
            with self._lock:
                available = [gpu for gpu in self._gpus.values() if gpu.is_available()]
                if len(available) >= count:
                    allocated = available[:count]
                    allocated_ids = []
                    now = _now_epoch()
                    for gpu in allocated:
                        gpu.allocated_to = experiment_id
                        gpu.allocated_at = now
                        allocated_ids.append(gpu.id)
                    return allocated_ids

            if _now_epoch() >= deadline:
                raise ResourceUnavailable(
                    f"需要{count}个GPU，但只有{len(available)}个可用"
                )

            if timeout > 0:
                time.sleep(0.1)
            else:
                raise ResourceUnavailable(
                    f"需要{count}个GPU，但只有{len(available)}个可用"
                )

    def release(self, gpu_ids: list[str], experiment_id: str) -> None:
        """释放实验占用的GPU

        Args:
            gpu_ids: 之前allocate返回的GPU ID列表
            experiment_id: 实验ID
        """
        with self._lock:
            for gpu_id in gpu_ids:
                gpu = self._gpus.get(gpu_id)
                if gpu and gpu.allocated_to == experiment_id:
                    gpu.allocated_to = None
                    gpu.allocated_at = None

    def get_available(self) -> list[GPUResource]:
        """获取当前可用GPU列表"""
        with self._lock:
            return [gpu for gpu in self._gpus.values() if gpu.is_available()]

    def get_allocation(self, experiment_id: str) -> list[str]:
        """查询实验的GPU分配"""
        with self._lock:
            return [
                gpu.id
                for gpu in self._gpus.values()
                if gpu.allocated_to == experiment_id
            ]

    def get_stats(self) -> dict[str, int]:
        """获取资源池统计信息"""
        with self._lock:
            total = len(self._gpus)
            available = len([gpu for gpu in self._gpus.values() if gpu.is_available()])
            return {
                "total": total,
                "available": available,
                "allocated": total - available,
            }


class ExperimentQueue:
    """管理待执行实验的优先级队列，支持baseline优先调度策略"""

    def __init__(self) -> None:
        self._queue: list[ExperimentJob] = []
        self._lock = threading.RLock()

    def enqueue(self, job: ExperimentJob) -> None:
        """添加实验到队列

        Args:
            job: 状态必须为PENDING的实验作业
        """
        if job.status != JobStatus.PENDING:
            raise ValueError(f"只能将PENDING状态的job入队，当前状态：{job.status}")

        with self._lock:
            job.status = JobStatus.QUEUED
            self._queue.append(job)
            self._reorder_by_priority()

    def _reorder_by_priority(self) -> None:
        """内部优先级排序（baseline first，priority小的优先）"""
        self._queue.sort(key=lambda j: (j.priority, j.created_at))

    def dequeue(self) -> ExperimentJob | None:
        """按优先级取出下一个实验

        Returns:
            优先级最高的实验，队列为空时返回None
        """
        with self._lock:
            if not self._queue:
                return None
            return self._queue.pop(0)

    def peek(self) -> ExperimentJob | None:
        """查看队列头部但不移除"""
        with self._lock:
            return self._queue[0] if self._queue else None

    def remove(self, job_id: str) -> bool:
        """取消/移除指定实验

        Returns:
            是否找到并移除
        """
        with self._lock:
            for i, job in enumerate(self._queue):
                if job.job_id == job_id:
                    self._queue.pop(i)
                    return True
            return False

    def list_pending(self) -> list[ExperimentJob]:
        """列出所有待执行实验"""
        with self._lock:
            return list(self._queue)

    def size(self) -> int:
        """队列长度"""
        with self._lock:
            return len(self._queue)


@dataclass
class SchedulerConfig:
    """调度器配置参数"""
    max_parallel_experiments: int = 2
    poll_interval_s: float = 1.0
    gpu_discovery_mode: Literal["local", "static"] = "local"
    static_gpu_list: list[str] = field(default_factory=list)
    lease_ttl_multiplier: float = 1.0
    enable_auto_retry: bool = False
    max_retries: int = 0


class ParallelResearchSupervisor:
    """适配层：包装现有ResearchSupervisor，提供非阻塞执行接口"""

    def __init__(
        self,
        backend: ExperimentBackend,
        lease_store: ResourceLeaseStore,
        root: Path,
    ) -> None:
        self.backend = backend
        self.lease_store = lease_store
        self.root = Path(root)

    def _prepare_workspace(self, job: ExperimentJob) -> Path:
        """准备工作目录"""
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", job.plan.id).strip("_")[:100]
        workspace = self.root / "runs" / f"{safe_id}-{uuid.uuid4().hex[:10]}"
        workspace.mkdir(parents=True, exist_ok=False)
        return workspace

    def _acquire_lease(self, job: ExperimentJob, ttl_multiplier: float = 1.0) -> ResourceLease:
        """获取资源租约"""
        resources = job.plan.resource_budget.get("resources", ["gpu:0"])
        if not isinstance(resources, list):
            raise ValueError("resource_budget.resources must be a list")

        ttl = float(job.plan.resource_budget.get("lease_ttl_s", 3600.0)) * ttl_multiplier
        return self.lease_store.acquire(resources, owner=job.actor, ttl_s=ttl)

    def _copy_candidate(self, candidate_artifact: Path, workspace: Path) -> Path:
        """复制候选文件到workspace"""
        import shutil
        source = candidate_artifact.resolve()
        if not source.exists():
            raise FileNotFoundError(source)

        destination = workspace / "candidate"
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            destination.mkdir(parents=True)
            shutil.copy2(source, destination / source.name)
        return destination

    def start_experiment_async(
        self,
        job: ExperimentJob,
        environment: dict[str, str],
    ) -> tuple[BackendHandle, Path]:
        """启动实验但不阻塞等待完成

        Args:
            job: 实验作业
            environment: 环境变量

        Returns:
            (BackendHandle, workspace路径) 元组

        Raises:
            异常时不返回None，直接抛出
        """
        workspace = self._prepare_workspace(job)
        candidate = self._copy_candidate(job.candidate_artifact, workspace)

        # 写入execution.json
        (workspace / "execution.json").write_text(
            json.dumps({
                "experiment_ref": job.plan.id,
                "candidate_artifact": str(job.candidate_artifact),
                "job_id": job.job_id,
                "actor": job.actor,
            }, indent=2) + "\n",
            encoding="utf-8"
        )

        # 准备环境变量
        env = dict(environment)
        env["RL_RESEARCH_EXPERIMENT_REF"] = job.plan.id
        env["RL_RESEARCH_CANDIDATE_ROOT"] = str(candidate)

        mechanism_bundle = candidate / "mechanisms.json"
        if not mechanism_bundle.is_file():
            raise FileNotFoundError(f"candidate mechanism bundle is missing: {mechanism_bundle}")
        env["RL_MECHANISM_BUNDLE"] = str(mechanism_bundle)

        # 启动backend
        handle = self.backend.start(job.plan, workspace, env)
        return handle, workspace

    def poll_experiment(self, handle: BackendHandle) -> BackendStatus:
        """轮询状态"""
        return self.backend.poll(handle)

    def evaluate_experiment(self, job: ExperimentJob) -> ExperimentExecution:
        """评估完成的实验"""
        if not job.workspace:
            raise ValueError("job.workspace is None")

        workspace = job.workspace
        evaluation = self.backend.evaluate(job.plan, workspace)

        # 计算disposition
        protected = self._protected_results(evaluation, job.plan)
        disposition = self._decide(evaluation, protected)

        return ExperimentExecution(
            experiment_ref=job.plan.id,
            disposition=disposition,
            workspace=str(workspace),
            backend_state="succeeded",
            evaluation=evaluation,
            protected_results=protected,
        )

    def stop_experiment(self, handle: BackendHandle, reason: str) -> None:
        """停止实验"""
        self.backend.stop(handle, reason)

    @staticmethod
    def _protected_results(evaluation: Mapping[str, Any], plan: ExperimentPlan) -> dict[str, Any]:
        """提取protected capability结果"""
        declared = evaluation.get("protected", {})
        results: dict[str, Any] = {}
        for capability in plan.protected_capabilities:
            item = declared.get(capability) if isinstance(declared, Mapping) else None
            results[capability] = item if isinstance(item, Mapping) else {
                "passed": False,
                "reason": "missing evaluator result"
            }
        return results

    @staticmethod
    def _decide(
        evaluation: Mapping[str, Any],
        protected: Mapping[str, Any]
    ) -> Literal["promote", "rollback", "inconclusive"]:
        """决定实验结果"""
        if not evaluation:
            return "inconclusive"
        if any(not bool(item.get("passed")) for item in protected.values()):
            return "rollback"
        if evaluation.get("success") is True and evaluation.get("evidence_complete", True) is True:
            return "promote"
        if evaluation.get("success") is False:
            return "rollback"
        return "inconclusive"


class ResearchScheduler:
    """并行实验调度的核心协调器：管理队列、资源池、执行器，协调多实验生命周期"""

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        root: str | os.PathLike[str] = "output/research_scheduler",
        *,
        lease_store: ResourceLeaseStore | None = None,
        backend: ExperimentBackend | None = None,
    ) -> None:
        self.config = config or SchedulerConfig()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        # 资源管理
        self.gpu_pool = GPUResourcePool(
            discovery_mode=self.config.gpu_discovery_mode,
            static_gpu_list=self.config.static_gpu_list,
        )

        # 队列和作业管理
        self.queue = ExperimentQueue()
        self._jobs: dict[str, ExperimentJob] = {}
        self._jobs_lock = threading.RLock()

        # 后端
        from autotuner.research.research_supervisor import CommandExperimentBackend
        self.lease_store = lease_store or ResourceLeaseStore(self.root / "leases")
        self.backend = backend or CommandExperimentBackend()
        self.supervisor = ParallelResearchSupervisor(
            backend=self.backend,
            lease_store=self.lease_store,
            root=self.root,
        )

        # 调度器状态
        self._running = False
        self._scheduler_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running_jobs: dict[str, ExperimentJob] = {}
        self._running_lock = threading.RLock()
        self._batches: dict[str, ExperimentBatch] = {}

    def submit_experiment(
        self,
        plan: ExperimentPlan,
        candidate_artifact: Path,
        priority: int = 1,
        actor: str = "scheduler",
    ) -> str:
        """提交单个实验到调度队列

        Args:
            plan: 实验计划
            candidate_artifact: 候选文件路径
            priority: 优先级（0=baseline最高，>0=candidate）
            actor: 提交者

        Returns:
            唯一job_id

        Raises:
            ResourceExhausted: 资源池已满且队列已满
        """
        job = ExperimentJob(
            job_id=f"job:{uuid.uuid4().hex}",
            plan=plan,
            candidate_artifact=Path(candidate_artifact),
            priority=priority,
            actor=actor,
        )

        with self._jobs_lock:
            self._jobs[job.job_id] = job

        self.queue.enqueue(job)
        return job.job_id

    def submit_batch(
        self,
        baseline: tuple[ExperimentPlan, Path],
        candidates: list[tuple[ExperimentPlan, Path]],
        wait_baseline: bool = False,
        actor: str = "scheduler",
    ) -> ExperimentBatch:
        """批量提交baseline和多个candidates

        Args:
            baseline: (ExperimentPlan, Path) 元组
            candidates: [(ExperimentPlan, Path), ...] 列表
            wait_baseline: 是否等baseline完成后再调度candidates
            actor: 提交者

        Returns:
            ExperimentBatch跟踪整批任务
        """
        baseline_plan, baseline_path = baseline
        baseline_job_id = self.submit_experiment(baseline_plan, baseline_path, priority=0, actor=actor)

        candidate_job_ids = []
        for idx, (cand_plan, cand_path) in enumerate(candidates, start=1):
            job_id = self.submit_experiment(cand_plan, cand_path, priority=idx, actor=actor)
            candidate_job_ids.append(job_id)

        batch = ExperimentBatch(
            batch_id=f"batch:{uuid.uuid4().hex}",
            baseline_job_id=baseline_job_id,
            candidate_job_ids=candidate_job_ids,
            wait_for_baseline=wait_baseline,
        )

        self._batches[batch.batch_id] = batch
        return batch

    def start_scheduler(self, max_parallel: int | None = None) -> None:
        """启动调度循环

        Args:
            max_parallel: 最大并行实验数，None使用config默认值
        """
        if self._running:
            raise RuntimeError("调度器已在运行")

        if max_parallel is not None:
            self.config.max_parallel_experiments = max_parallel

        self._running = True
        self._stop_event.clear()
        self._scheduler_thread = threading.Thread(target=self._schedule_loop, daemon=True)
        self._scheduler_thread.start()

    def stop_scheduler(self) -> None:
        """停止调度器，清理资源"""
        if not self._running:
            return

        self._stop_event.set()
        self._running = False

        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=10)
            self._scheduler_thread = None

    def _schedule_loop(self) -> None:
        """主调度循环（后台线程）"""
        while not self._stop_event.is_set():
            try:
                # 轮询运行中的任务
                self._poll_running_jobs()

                # 尝试启动新任务
                self._try_start_next()

                # 休眠
                time.sleep(self.config.poll_interval_s)

            except Exception as e:
                # 错误隔离：记录但不中断调度循环
                print(f"调度循环出错: {e}")
                time.sleep(self.config.poll_interval_s)

    def _try_start_next(self) -> None:
        """尝试从队列启动下一个实验"""
        with self._running_lock:
            # 检查并行限制
            if len(self._running_jobs) >= self.config.max_parallel_experiments:
                return

            # 从队列取出下一个
            job = self.queue.dequeue()
            if not job:
                return

            # 检查batch约束
            if not self._can_start_job(job):
                # 放回队列
                job.status = JobStatus.PENDING
                self.queue.enqueue(job)
                return

            # 尝试分配GPU
            required_gpus = len(job.plan.resource_budget.get("resources", ["gpu:0"]))
            try:
                gpu_ids = self.gpu_pool.allocate(required_gpus, job.job_id, timeout=0.0)
                job.gpu_allocation = gpu_ids
            except ResourceUnavailable:
                # GPU不足，放回队列
                job.status = JobStatus.PENDING
                self.queue.enqueue(job)
                return

            # 启动实验
            try:
                self._start_job(job)
            except Exception as e:
                # 启动失败，释放GPU
                self.gpu_pool.release(job.gpu_allocation, job.job_id)
                job.status = JobStatus.FAILED
                job.error_message = f"启动失败: {e}"
                job.completed_at = _now_epoch()

    def _can_start_job(self, job: ExperimentJob) -> bool:
        """检查是否可以启动job（考虑batch约束）"""
        # 查找包含此job的batch
        for batch in self._batches.values():
            if job.job_id in batch.candidate_job_ids:
                if batch.wait_for_baseline:
                    # 需要等待baseline完成
                    baseline_job = self._jobs.get(batch.baseline_job_id)
                    if baseline_job and baseline_job.status != JobStatus.COMPLETED:
                        return False
        return True

    def _start_job(self, job: ExperimentJob) -> None:
        """启动单个实验"""
        job.status = JobStatus.RUNNING
        job.started_at = _now_epoch()

        # 准备环境变量
        env = os.environ.copy()
        if job.gpu_allocation:
            # 设置CUDA_VISIBLE_DEVICES
            device_indices = [
                self.gpu_pool._gpus[gpu_id].device_index
                for gpu_id in job.gpu_allocation
            ]
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(idx) for idx in device_indices)

        # 启动实验
        handle, workspace = self.supervisor.start_experiment_async(job, env)
        job.backend_handle = handle
        job.workspace = workspace

        with self._running_lock:
            self._running_jobs[job.job_id] = job

    def _poll_running_jobs(self) -> None:
        """轮询运行中实验状态"""
        with self._running_lock:
            completed_job_ids = []

            for job_id, job in self._running_jobs.items():
                if not job.backend_handle:
                    continue

                # 检查超时
                max_seconds = float(job.plan.training_window.get("max_seconds", job.plan.resource_budget.get("max_seconds", 3600.0)))
                if job.started_at and (_now_epoch() - job.started_at >= max_seconds):
                    self.supervisor.stop_experiment(job.backend_handle, "training_window_timeout")
                    job.status = JobStatus.FAILED
                    job.error_message = "训练超时"
                    job.completed_at = _now_epoch()
                    completed_job_ids.append(job_id)
                    continue

                # 轮询状态
                status = self.supervisor.poll_experiment(job.backend_handle)

                if status.state != "running":
                    if status.state == "succeeded":
                        # 评估实验
                        try:
                            execution = self.supervisor.evaluate_experiment(job)
                            job.execution_result = execution
                            job.status = JobStatus.COMPLETED
                        except Exception as e:
                            job.status = JobStatus.FAILED
                            job.error_message = f"评估失败: {e}"
                    else:
                        job.status = JobStatus.FAILED
                        job.error_message = status.message

                    job.completed_at = _now_epoch()
                    completed_job_ids.append(job_id)

            # 处理完成的任务
            for job_id in completed_job_ids:
                job = self._running_jobs.pop(job_id)
                self._handle_job_completion(job)

    def _handle_job_completion(self, job: ExperimentJob) -> None:
        """处理完成/失败"""
        self._cleanup_job(job)

    def _cleanup_job(self, job: ExperimentJob) -> None:
        """释放资源"""
        if job.gpu_allocation:
            self.gpu_pool.release(job.gpu_allocation, job.job_id)
            job.gpu_allocation = []

    def get_job_status(self, job_id: str) -> ExperimentJob:
        """查询实验当前状态

        Args:
            job_id: 作业ID

        Returns:
            完整的ExperimentJob对象

        Raises:
            KeyError: job_id不存在
        """
        with self._jobs_lock:
            if job_id not in self._jobs:
                raise KeyError(f"job_id不存在: {job_id}")
            return self._jobs[job_id]

    def cancel_job(self, job_id: str) -> None:
        """取消实验"""
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if not job:
                raise KeyError(f"job_id不存在: {job_id}")

            if job.status == JobStatus.QUEUED:
                self.queue.remove(job_id)
                job.status = JobStatus.CANCELLED
                job.completed_at = _now_epoch()
            elif job.status == JobStatus.RUNNING:
                with self._running_lock:
                    if job.backend_handle:
                        self.supervisor.stop_experiment(job.backend_handle, "user_cancelled")
                    job.status = JobStatus.CANCELLED
                    job.completed_at = _now_epoch()
                    if job_id in self._running_jobs:
                        self._running_jobs.pop(job_id)
                    self._cleanup_job(job)

    def list_jobs(self, status: JobStatus | None = None) -> list[ExperimentJob]:
        """列出实验

        Args:
            status: 过滤状态，None返回全部

        Returns:
            ExperimentJob列表
        """
        with self._jobs_lock:
            if status is None:
                return list(self._jobs.values())
            return [job for job in self._jobs.values() if job.status == status]

    def wait_all_complete(self, timeout: float | None = None) -> SchedulerState:
        """阻塞等待所有实验完成（或超时）

        Args:
            timeout: 超时秒数，None=无限等待

        Returns:
            最终调度器状态

        Raises:
            RuntimeError: 调度器未启动
        """
        if not self._running:
            raise RuntimeError("调度器未启动")

        deadline = (_now_epoch() + timeout) if timeout is not None else None

        while True:
            with self._jobs_lock:
                pending = [j for j in self._jobs.values() if j.status in {JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING}]
                if not pending:
                    break

            if deadline and _now_epoch() >= deadline:
                break

            time.sleep(self.config.poll_interval_s)

        return self.get_scheduler_state()

    def get_scheduler_state(self) -> SchedulerState:
        """获取调度器状态快照"""
        with self._jobs_lock:
            jobs_pending = len([j for j in self._jobs.values() if j.status == JobStatus.QUEUED])
            jobs_running = len([j for j in self._jobs.values() if j.status == JobStatus.RUNNING])
            jobs_completed = len([j for j in self._jobs.values() if j.status == JobStatus.COMPLETED])
            jobs_failed = len([j for j in self._jobs.values() if j.status == JobStatus.FAILED])

            with self._running_lock:
                active_job_ids = list(self._running_jobs.keys())

        gpu_stats = self.gpu_pool.get_stats()

        return SchedulerState(
            running=self._running,
            max_parallel=self.config.max_parallel_experiments,
            jobs_pending=jobs_pending,
            jobs_running=jobs_running,
            jobs_completed=jobs_completed,
            jobs_failed=jobs_failed,
            available_gpus=gpu_stats["available"],
            allocated_gpus=gpu_stats["allocated"],
            last_schedule_time=_now_epoch(),
            active_job_ids=active_job_ids,
        )
