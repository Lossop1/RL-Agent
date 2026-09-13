"""ResearchScheduler 单元测试

测试场景：
1. 串行提交3个实验，验证状态转换
2. 并行运行2个实验，验证并发执行
3. GPU资源耗尽时任务排队
4. 某个实验失败不影响其他
5. 优先级调度（baseline先于候选）
6. 崩溃恢复（从状态文件恢复）
"""
import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# 配置pytest-asyncio
pytest_plugins = ('pytest_asyncio',)


class ExperimentState:
    """实验状态枚举"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Experiment:
    """实验定义"""
    def __init__(
        self,
        experiment_id: str,
        name: str,
        config: Dict[str, Any],
        priority: int = 0,
        gpu_required: int = 1,
    ):
        self.experiment_id = experiment_id
        self.name = name
        self.config = config
        self.priority = priority
        self.gpu_required = gpu_required
        self.state = ExperimentState.PENDING
        self.result: Dict[str, Any] | None = None
        self.error: str | None = None


class ResearchScheduler:
    """研究实验调度器

    负责管理多个实验的调度、执行和资源分配
    """
    def __init__(
        self,
        max_parallel: int = 2,
        gpu_total: int = 2,
        state_file: Path | None = None,
    ):
        self.max_parallel = max_parallel
        self.gpu_total = gpu_total
        self.gpu_available = gpu_total
        self.state_file = state_file

        self.experiments: Dict[str, Experiment] = {}
        self.running: List[str] = []
        self.queue: List[str] = []
        self._lock = asyncio.Lock()
        self._tasks: Dict[str, asyncio.Task] = {}

    async def submit(self, experiment: Experiment) -> None:
        """提交实验"""
        async with self._lock:
            self.experiments[experiment.experiment_id] = experiment
            experiment.state = ExperimentState.QUEUED
            self.queue.append(experiment.experiment_id)
            await self._save_state()

        # 尝试启动排队的实验
        await self._process_queue()

    async def _process_queue(self) -> None:
        """处理队列中的实验"""
        async with self._lock:
            # 按优先级排序（降序）
            self.queue.sort(
                key=lambda eid: self.experiments[eid].priority,
                reverse=True,
            )

            # 启动可以运行的实验
            to_start = []
            for exp_id in self.queue[:]:
                exp = self.experiments[exp_id]

                # 检查资源和并发限制
                if (
                    len(self.running) < self.max_parallel
                    and self.gpu_available >= exp.gpu_required
                ):
                    to_start.append(exp_id)
                    self.queue.remove(exp_id)
                    self.running.append(exp_id)
                    exp.state = ExperimentState.RUNNING
                    self.gpu_available -= exp.gpu_required

        # 在锁外启动任务
        for exp_id in to_start:
            task = asyncio.create_task(self._run_experiment(exp_id))
            self._tasks[exp_id] = task

    async def _run_experiment(self, experiment_id: str) -> None:
        """运行单个实验"""
        exp = self.experiments[experiment_id]

        try:
            # 模拟实验执行
            result = await self._execute_experiment(exp)

            async with self._lock:
                exp.state = ExperimentState.COMPLETED
                exp.result = result
                self.gpu_available += exp.gpu_required
                if experiment_id in self.running:
                    self.running.remove(experiment_id)
                await self._save_state()

        except Exception as e:
            async with self._lock:
                exp.state = ExperimentState.FAILED
                exp.error = str(e)
                self.gpu_available += exp.gpu_required
                if experiment_id in self.running:
                    self.running.remove(experiment_id)
                await self._save_state()

        finally:
            # 继续处理队列
            await self._process_queue()

    async def _execute_experiment(self, experiment: Experiment) -> Dict[str, Any]:
        """执行实验（由子类或mock实现）"""
        raise NotImplementedError

    async def wait_all(self, timeout: float | None = None) -> None:
        """等待所有实验完成"""
        tasks = list(self._tasks.values())
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)

    async def get_status(self, experiment_id: str) -> Dict[str, Any]:
        """获取实验状态"""
        async with self._lock:
            exp = self.experiments.get(experiment_id)
            if not exp:
                return {"error": "experiment not found"}

            return {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "state": exp.state,
                "priority": exp.priority,
                "result": exp.result,
                "error": exp.error,
            }

    async def _save_state(self) -> None:
        """保存状态到文件"""
        if not self.state_file:
            return

        state = {
            "experiments": {
                eid: {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "config": exp.config,
                    "priority": exp.priority,
                    "gpu_required": exp.gpu_required,
                    "state": exp.state,
                    "result": exp.result,
                    "error": exp.error,
                }
                for eid, exp in self.experiments.items()
            },
            "queue": self.queue,
            "running": self.running,
            "gpu_available": self.gpu_available,
        }

        self.state_file.write_text(json.dumps(state, indent=2))

    @classmethod
    async def restore_from_state(cls, state_file: Path) -> "ResearchScheduler":
        """从状态文件恢复"""
        if not state_file.exists():
            raise FileNotFoundError(f"State file not found: {state_file}")

        state = json.loads(state_file.read_text())

        scheduler = cls(
            max_parallel=2,
            gpu_total=2,
            state_file=state_file,
        )

        # 恢复实验
        for eid, exp_data in state["experiments"].items():
            exp = Experiment(
                experiment_id=exp_data["experiment_id"],
                name=exp_data["name"],
                config=exp_data["config"],
                priority=exp_data["priority"],
                gpu_required=exp_data["gpu_required"],
            )
            exp.state = exp_data["state"]
            exp.result = exp_data["result"]
            exp.error = exp_data["error"]
            scheduler.experiments[eid] = exp

        scheduler.queue = state["queue"]
        scheduler.running = state["running"]
        scheduler.gpu_available = state["gpu_available"]

        return scheduler


# ============= 测试 fixtures =============

@pytest.fixture
def temp_state_file():
    """临时状态文件"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        path = Path(f.name)
    yield path
    if path.exists():
        path.unlink()


@pytest.fixture
def scheduler(temp_state_file):
    """默认调度器"""
    return ResearchScheduler(
        max_parallel=2,
        gpu_total=2,
        state_file=temp_state_file,
    )


@pytest.fixture
def mock_execute():
    """Mock实验执行"""
    async def _execute(experiment: Experiment) -> Dict[str, Any]:
        await asyncio.sleep(0.1)  # 模拟执行时间
        return {"score": 0.85, "iterations": 1000}

    return _execute


# ============= 测试场景 =============

@pytest.mark.asyncio
async def test_serial_submission_state_transitions(scheduler, mock_execute):
    """测试1: 串行提交3个实验，验证状态转换"""
    # Mock实验执行
    scheduler._execute_experiment = mock_execute

    experiments = [
        Experiment("exp1", "baseline", {"lr": 0.001}, priority=0),
        Experiment("exp2", "variant_a", {"lr": 0.002}, priority=0),
        Experiment("exp3", "variant_b", {"lr": 0.003}, priority=0),
    ]

    # 串行提交
    for exp in experiments:
        await scheduler.submit(exp)
        await asyncio.sleep(0.05)  # 短暂延迟以观察状态

    # 等待所有完成
    await scheduler.wait_all(timeout=5.0)

    # 验证所有实验都完成了
    for exp_id in ["exp1", "exp2", "exp3"]:
        status = await scheduler.get_status(exp_id)
        assert status["state"] == ExperimentState.COMPLETED
        assert status["result"] is not None
        assert "score" in status["result"]


@pytest.mark.asyncio
async def test_parallel_execution(scheduler, mock_execute):
    """测试2: 并行运行2个实验，验证并发执行"""
    scheduler._execute_experiment = mock_execute

    experiments = [
        Experiment("exp1", "exp_parallel_1", {"param": 1}),
        Experiment("exp2", "exp_parallel_2", {"param": 2}),
    ]

    # 同时提交
    await asyncio.gather(*[scheduler.submit(exp) for exp in experiments])

    # 检查并发限制：最多2个同时运行
    await asyncio.sleep(0.05)
    async with scheduler._lock:
        assert len(scheduler.running) <= scheduler.max_parallel

    # 等待完成
    await scheduler.wait_all(timeout=5.0)

    # 验证都成功了
    for exp_id in ["exp1", "exp2"]:
        status = await scheduler.get_status(exp_id)
        assert status["state"] == ExperimentState.COMPLETED


@pytest.mark.asyncio
async def test_gpu_resource_queueing(scheduler, mock_execute):
    """测试3: GPU资源耗尽时任务排队"""
    scheduler._execute_experiment = mock_execute

    # 提交3个实验，每个需要1 GPU，总共只有2 GPU
    experiments = [
        Experiment(f"exp{i}", f"exp_{i}", {"id": i}, gpu_required=1)
        for i in range(1, 4)
    ]

    # 快速提交所有实验
    await asyncio.gather(*[scheduler.submit(exp) for exp in experiments])

    # 短暂等待后检查状态
    await asyncio.sleep(0.05)

    async with scheduler._lock:
        # 应该有2个在运行，1个在队列
        assert len(scheduler.running) == 2
        assert len(scheduler.queue) == 1
        assert scheduler.gpu_available == 0

    # 等待所有完成
    await scheduler.wait_all(timeout=5.0)

    # 给一点时间让最后的清理完成
    await asyncio.sleep(0.1)

    # 验证最终GPU被释放
    async with scheduler._lock:
        assert scheduler.gpu_available == scheduler.gpu_total
        assert len(scheduler.running) == 0
        assert len(scheduler.queue) == 0


@pytest.mark.asyncio
async def test_experiment_failure_isolation(scheduler):
    """测试4: 某个实验失败不影响其他"""

    # Mock: 第一个实验失败，其他成功
    call_count = [0]

    async def _execute_with_failure(experiment: Experiment) -> Dict[str, Any]:
        call_count[0] += 1
        if call_count[0] == 1:  # 第一个调用失败
            raise RuntimeError("Simulated experiment failure")
        await asyncio.sleep(0.1)
        return {"score": 0.9}

    scheduler._execute_experiment = _execute_with_failure

    experiments = [
        Experiment("exp1", "will_fail", {}),
        Experiment("exp2", "will_succeed_1", {}),
        Experiment("exp3", "will_succeed_2", {}),
    ]

    await asyncio.gather(*[scheduler.submit(exp) for exp in experiments])
    await scheduler.wait_all(timeout=5.0)

    # 验证状态
    status1 = await scheduler.get_status("exp1")
    assert status1["state"] == ExperimentState.FAILED
    assert status1["error"] is not None

    status2 = await scheduler.get_status("exp2")
    assert status2["state"] == ExperimentState.COMPLETED

    status3 = await scheduler.get_status("exp3")
    assert status3["state"] == ExperimentState.COMPLETED


@pytest.mark.asyncio
async def test_priority_scheduling(scheduler, mock_execute):
    """测试5: 优先级调度（baseline先于候选）"""
    scheduler._execute_experiment = mock_execute

    # 提交3个实验，baseline优先级更高
    experiments = [
        Experiment("candidate1", "candidate_1", {}, priority=1),
        Experiment("baseline", "baseline", {}, priority=10),  # 高优先级
        Experiment("candidate2", "candidate_2", {}, priority=1),
    ]

    # 快速提交
    await asyncio.gather(*[scheduler.submit(exp) for exp in experiments])

    # 短暂延迟后检查运行顺序
    await asyncio.sleep(0.05)

    async with scheduler._lock:
        # baseline应该在运行中（因为优先级高）
        running_exp_ids = scheduler.running
        assert "baseline" in running_exp_ids or scheduler.experiments["baseline"].state == ExperimentState.COMPLETED

    await scheduler.wait_all(timeout=5.0)

    # 给一点时间让所有任务完成和清理
    await asyncio.sleep(0.2)

    # 所有实验都应完成
    for exp_id in ["candidate1", "baseline", "candidate2"]:
        status = await scheduler.get_status(exp_id)
        assert status["state"] == ExperimentState.COMPLETED


@pytest.mark.asyncio
async def test_crash_recovery(temp_state_file, mock_execute):
    """测试6: 崩溃恢复（从状态文件恢复）"""

    # 阶段1: 创建调度器并提交实验
    scheduler1 = ResearchScheduler(
        max_parallel=2,
        gpu_total=2,
        state_file=temp_state_file,
    )
    scheduler1._execute_experiment = mock_execute

    experiments = [
        Experiment("exp1", "exp_1", {"param": 1}),
        Experiment("exp2", "exp_2", {"param": 2}),
    ]

    await asyncio.gather(*[scheduler1.submit(exp) for exp in experiments])

    # 不等待完成，直接保存状态（模拟崩溃前的状态）
    await asyncio.sleep(0.05)
    async with scheduler1._lock:
        await scheduler1._save_state()

    # 验证状态文件存在
    assert temp_state_file.exists()

    # 阶段2: 从状态文件恢复
    scheduler2 = await ResearchScheduler.restore_from_state(temp_state_file)

    # 验证恢复的状态
    assert len(scheduler2.experiments) == 2
    assert "exp1" in scheduler2.experiments
    assert "exp2" in scheduler2.experiments

    # 验证GPU和队列状态被恢复
    assert scheduler2.gpu_available >= 0
    assert scheduler2.gpu_available <= scheduler2.gpu_total


@pytest.mark.asyncio
async def test_empty_queue_handling(scheduler, mock_execute):
    """边界测试: 空队列处理"""
    scheduler._execute_experiment = mock_execute

    # 不提交任何实验
    await scheduler._process_queue()

    async with scheduler._lock:
        assert len(scheduler.queue) == 0
        assert len(scheduler.running) == 0
        assert scheduler.gpu_available == scheduler.gpu_total


@pytest.mark.asyncio
async def test_concurrent_submissions(scheduler, mock_execute):
    """边界测试: 并发提交"""
    scheduler._execute_experiment = mock_execute

    # 大量并发提交
    experiments = [
        Experiment(f"exp{i}", f"concurrent_{i}", {"id": i})
        for i in range(10)
    ]

    # 并发提交
    await asyncio.gather(*[scheduler.submit(exp) for exp in experiments])

    # 验证所有实验都被记录
    async with scheduler._lock:
        assert len(scheduler.experiments) == 10
        total = len(scheduler.running) + len(scheduler.queue)
        assert total <= 10

    # 等待所有完成，增加超时时间
    await scheduler.wait_all(timeout=30.0)

    # 多次等待确保所有异步任务真正完成
    for _ in range(5):
        await asyncio.sleep(0.2)
        # 检查是否还有运行中的任务
        if not scheduler._tasks:
            break

    # 验证最终状态
    completed = 0
    failed = 0
    for i in range(10):
        status = await scheduler.get_status(f"exp{i}")
        if status["state"] == ExperimentState.COMPLETED:
            completed += 1
        elif status["state"] == ExperimentState.FAILED:
            failed += 1

    # 所有实验应该完成（成功或失败）
    assert completed + failed == 10, f"Only {completed} completed and {failed} failed out of 10"


@pytest.mark.asyncio
async def test_get_status_nonexistent(scheduler):
    """边界测试: 查询不存在的实验"""
    status = await scheduler.get_status("nonexistent")
    assert "error" in status


@pytest.mark.asyncio
async def test_resource_release_on_failure(scheduler):
    """边界测试: 失败时资源正确释放"""

    async def _failing_execute(experiment: Experiment) -> Dict[str, Any]:
        await asyncio.sleep(0.1)
        raise RuntimeError("Intentional failure")

    scheduler._execute_experiment = _failing_execute

    exp = Experiment("exp1", "will_fail", {}, gpu_required=1)
    await scheduler.submit(exp)
    await scheduler.wait_all(timeout=5.0)

    # GPU应该被释放
    async with scheduler._lock:
        assert scheduler.gpu_available == scheduler.gpu_total
        assert len(scheduler.running) == 0

    status = await scheduler.get_status("exp1")
    assert status["state"] == ExperimentState.FAILED
