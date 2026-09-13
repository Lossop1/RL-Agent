"""实验追踪器测试。"""
import json
import tempfile
from pathlib import Path

import pytest

from autotuner.product.experiment_tracker import (
    ExperimentTracker,
    ExperimentStatus,
    ExperimentMetadata,
)


@pytest.fixture
def temp_storage():
    """创建临时存储文件。"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name
    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def tracker(temp_storage):
    """创建实验追踪器实例。"""
    return ExperimentTracker(storage_path=temp_storage)


def test_create_experiment(tracker):
    """测试创建实验。"""
    tracker.create_experiment("exp_001", "candidate_123", priority=1)

    exp = tracker.get_experiment("exp_001")
    assert exp is not None
    assert exp.experiment_id == "exp_001"
    assert exp.candidate_id == "candidate_123"
    assert exp.status == ExperimentStatus.PENDING
    assert exp.priority == 1
    assert exp.created_at != ""
    assert exp.started_at is None
    assert exp.completed_at is None


def test_create_duplicate_experiment(tracker):
    """测试创建重复实验应抛出异常。"""
    tracker.create_experiment("exp_001", "candidate_123")

    with pytest.raises(ValueError, match="已存在"):
        tracker.create_experiment("exp_001", "candidate_456")


def test_update_status_to_running(tracker):
    """测试更新状态到 RUNNING。"""
    tracker.create_experiment("exp_001", "candidate_123")
    tracker.update_status("exp_001", ExperimentStatus.RUNNING, gpu_allocation=0)

    exp = tracker.get_experiment("exp_001")
    assert exp.status == ExperimentStatus.RUNNING
    assert exp.gpu_allocation == 0
    assert exp.started_at is not None
    assert exp.completed_at is None


def test_update_status_to_completed(tracker):
    """测试更新状态到 COMPLETED。"""
    tracker.create_experiment("exp_001", "candidate_123")
    tracker.update_status("exp_001", ExperimentStatus.RUNNING, gpu_allocation=0)
    tracker.update_status("exp_001", ExperimentStatus.COMPLETED)

    exp = tracker.get_experiment("exp_001")
    assert exp.status == ExperimentStatus.COMPLETED
    assert exp.completed_at is not None


def test_update_status_to_failed(tracker):
    """测试更新状态到 FAILED。"""
    tracker.create_experiment("exp_001", "candidate_123")
    tracker.update_status("exp_001", ExperimentStatus.RUNNING, gpu_allocation=0)
    tracker.update_status("exp_001", ExperimentStatus.FAILED,
                         error_message="GPU out of memory")

    exp = tracker.get_experiment("exp_001")
    assert exp.status == ExperimentStatus.FAILED
    assert exp.error_message == "GPU out of memory"
    assert exp.completed_at is not None


def test_update_nonexistent_experiment(tracker):
    """测试更新不存在的实验应抛出异常。"""
    with pytest.raises(KeyError, match="不存在"):
        tracker.update_status("exp_999", ExperimentStatus.RUNNING)


def test_get_by_status(tracker):
    """测试按状态查询实验。"""
    tracker.create_experiment("exp_001", "candidate_123")
    tracker.create_experiment("exp_002", "candidate_456")
    tracker.create_experiment("exp_003", "candidate_789")

    tracker.update_status("exp_001", ExperimentStatus.RUNNING, gpu_allocation=0)
    tracker.update_status("exp_002", ExperimentStatus.COMPLETED)

    pending = tracker.get_by_status(ExperimentStatus.PENDING)
    assert len(pending) == 1
    assert pending[0].experiment_id == "exp_003"

    running = tracker.get_by_status(ExperimentStatus.RUNNING)
    assert len(running) == 1
    assert running[0].experiment_id == "exp_001"

    completed = tracker.get_by_status(ExperimentStatus.COMPLETED)
    assert len(completed) == 1
    assert completed[0].experiment_id == "exp_002"


def test_get_by_priority(tracker):
    """测试按优先级查询实验。"""
    tracker.create_experiment("exp_001", "candidate_123", priority=3)
    tracker.create_experiment("exp_002", "candidate_456", priority=1)
    tracker.create_experiment("exp_003", "candidate_789", priority=2)

    all_sorted = tracker.get_by_priority()
    assert len(all_sorted) == 3
    assert all_sorted[0].experiment_id == "exp_002"  # priority 1
    assert all_sorted[1].experiment_id == "exp_003"  # priority 2
    assert all_sorted[2].experiment_id == "exp_001"  # priority 3


def test_get_by_priority_with_status_filter(tracker):
    """测试按优先级查询并过滤状态。"""
    tracker.create_experiment("exp_001", "candidate_123", priority=3)
    tracker.create_experiment("exp_002", "candidate_456", priority=1)
    tracker.create_experiment("exp_003", "candidate_789", priority=2)

    tracker.update_status("exp_002", ExperimentStatus.RUNNING, gpu_allocation=0)

    pending_sorted = tracker.get_by_priority(ExperimentStatus.PENDING)
    assert len(pending_sorted) == 2
    assert pending_sorted[0].experiment_id == "exp_003"  # priority 2
    assert pending_sorted[1].experiment_id == "exp_001"  # priority 3


def test_persistence_save_and_load(temp_storage):
    """测试状态持久化和加载。"""
    # 创建追踪器并添加实验
    tracker1 = ExperimentTracker(storage_path=temp_storage)
    tracker1.create_experiment("exp_001", "candidate_123", priority=1)
    tracker1.update_status("exp_001", ExperimentStatus.RUNNING, gpu_allocation=0)

    # 创建新实例，应该加载之前的状态
    tracker2 = ExperimentTracker(storage_path=temp_storage)
    exp = tracker2.get_experiment("exp_001")

    assert exp is not None
    assert exp.experiment_id == "exp_001"
    assert exp.candidate_id == "candidate_123"
    assert exp.status == ExperimentStatus.RUNNING
    assert exp.gpu_allocation == 0
    assert exp.priority == 1


def test_status_change_callback(tracker):
    """测试状态变更回调。"""
    callback_log = []

    def on_status_change(exp_id, old_status, new_status, metadata):
        callback_log.append((exp_id, old_status, new_status))

    tracker.register_callback(on_status_change)

    tracker.create_experiment("exp_001", "candidate_123")
    tracker.update_status("exp_001", ExperimentStatus.RUNNING, gpu_allocation=0)
    tracker.update_status("exp_001", ExperimentStatus.COMPLETED)

    assert len(callback_log) == 2
    assert callback_log[0] == ("exp_001", ExperimentStatus.PENDING, ExperimentStatus.RUNNING)
    assert callback_log[1] == ("exp_001", ExperimentStatus.RUNNING, ExperimentStatus.COMPLETED)


def test_callback_not_triggered_on_same_status(tracker):
    """测试相同状态不触发回调。"""
    callback_log = []

    def on_status_change(exp_id, old_status, new_status, metadata):
        callback_log.append((exp_id, old_status, new_status))

    tracker.register_callback(on_status_change)

    tracker.create_experiment("exp_001", "candidate_123")
    tracker.update_status("exp_001", ExperimentStatus.PENDING)  # 相同状态

    assert len(callback_log) == 0


def test_delete_experiment(tracker):
    """测试删除实验。"""
    tracker.create_experiment("exp_001", "candidate_123")

    assert tracker.delete_experiment("exp_001") is True
    assert tracker.get_experiment("exp_001") is None
    assert tracker.delete_experiment("exp_001") is False  # 再次删除返回 False


def test_update_custom_metadata(tracker):
    """测试更新自定义元数据。"""
    tracker.create_experiment("exp_001", "candidate_123",
                            metadata={"config": "v1", "notes": "initial"})

    tracker.update_metadata("exp_001", {"notes": "updated", "iteration": 5})

    exp = tracker.get_experiment("exp_001")
    assert exp.metadata["config"] == "v1"
    assert exp.metadata["notes"] == "updated"
    assert exp.metadata["iteration"] == 5


def test_count_by_status(tracker):
    """测试统计各状态数量。"""
    tracker.create_experiment("exp_001", "candidate_123")
    tracker.create_experiment("exp_002", "candidate_456")
    tracker.create_experiment("exp_003", "candidate_789")

    tracker.update_status("exp_001", ExperimentStatus.RUNNING, gpu_allocation=0)
    tracker.update_status("exp_002", ExperimentStatus.COMPLETED)

    counts = tracker.count_by_status()
    assert counts[ExperimentStatus.PENDING] == 1
    assert counts[ExperimentStatus.RUNNING] == 1
    assert counts[ExperimentStatus.COMPLETED] == 1
    assert counts[ExperimentStatus.FAILED] == 0


def test_get_all_experiments(tracker):
    """测试获取所有实验。"""
    tracker.create_experiment("exp_001", "candidate_123")
    tracker.create_experiment("exp_002", "candidate_456")
    tracker.create_experiment("exp_003", "candidate_789")

    all_exps = tracker.get_all()
    assert len(all_exps) == 3
    exp_ids = {exp.experiment_id for exp in all_exps}
    assert exp_ids == {"exp_001", "exp_002", "exp_003"}


def test_corrupted_file_handling(temp_storage):
    """测试损坏文件的处理。"""
    # 写入无效的 JSON
    with open(temp_storage, "w") as f:
        f.write("invalid json content {{{")

    # 应该能够创建追踪器并从空状态开始
    tracker = ExperimentTracker(storage_path=temp_storage)
    tracker.create_experiment("exp_001", "candidate_123")

    exp = tracker.get_experiment("exp_001")
    assert exp is not None
