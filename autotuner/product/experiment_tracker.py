"""实验追踪器 — 追踪实验状态、元数据和持久化。

追踪实验的完整生命周期（PENDING → RUNNING → COMPLETED/FAILED），记录候选ID、GPU分配、
时间戳等元数据，并持久化到 JSON 文件以支持崩溃恢复。提供按状态、优先级查询的接口，
以及状态变更回调机制。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any


class ExperimentStatus(Enum):
    """实验状态枚举。"""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class ExperimentMetadata:
    """实验元数据记录。

    Args:
        experiment_id: 实验唯一标识符
        candidate_id: 候选配置ID
        status: 实验状态
        gpu_allocation: 分配的GPU编号（可选）
        priority: 实验优先级（数字越小优先级越高）
        created_at: 创建时间戳
        started_at: 开始时间戳（可选）
        completed_at: 完成时间戳（可选）
        error_message: 失败时的错误信息（可选）
        metadata: 额外的自定义元数据
    """
    experiment_id: str
    candidate_id: str
    status: ExperimentStatus
    gpu_allocation: Optional[int] = None
    priority: int = 0
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# 状态变更回调类型：(experiment_id, old_status, new_status, metadata) -> None
StatusCallback = Callable[[str, ExperimentStatus, ExperimentStatus, ExperimentMetadata], None]


class ExperimentTracker:
    """实验追踪器 — 管理实验状态和持久化。

    管理所有实验的状态，持久化到 JSON 文件以支持崩溃恢复，提供查询接口和状态变更回调。

    Example:
        tracker = ExperimentTracker(storage_path="experiments.json")

        # 创建新实验
        tracker.create_experiment("exp_001", "candidate_123", priority=1)

        # 更新状态
        tracker.update_status("exp_001", ExperimentStatus.RUNNING, gpu_allocation=0)

        # 查询
        running = tracker.get_by_status(ExperimentStatus.RUNNING)
        pending_sorted = tracker.get_by_priority(ExperimentStatus.PENDING)
    """

    def __init__(self, storage_path: str = "experiments_state.json"):
        """初始化实验追踪器。

        Args:
            storage_path: JSON 存储文件路径
        """
        self.storage_path = Path(storage_path)
        self._experiments: Dict[str, ExperimentMetadata] = {}
        self._callbacks: List[StatusCallback] = []
        self._load_state()

    def _load_state(self) -> None:
        """从 JSON 文件加载状态。"""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for exp_id, exp_data in data.items():
                # 将状态字符串转换为枚举
                exp_data["status"] = ExperimentStatus(exp_data["status"])
                self._experiments[exp_id] = ExperimentMetadata(**exp_data)
        except Exception as e:
            # 文件损坏时从空状态开始
            print(f"警告: 无法加载状态文件 {self.storage_path}: {e}")
            self._experiments = {}

    def _save_state(self) -> None:
        """持久化状态到 JSON 文件。"""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # 将实验数据转换为可序列化格式
        data = {}
        for exp_id, metadata in self._experiments.items():
            exp_dict = asdict(metadata)
            # 将枚举转换为字符串
            exp_dict["status"] = metadata.status.value
            data[exp_id] = exp_dict

        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def register_callback(self, callback: StatusCallback) -> None:
        """注册状态变更回调函数。

        Args:
            callback: 回调函数，签名为 (experiment_id, old_status, new_status, metadata) -> None
        """
        self._callbacks.append(callback)

    def _trigger_callbacks(self, experiment_id: str, old_status: ExperimentStatus,
                          new_status: ExperimentStatus) -> None:
        """触发所有注册的回调函数。"""
        metadata = self._experiments[experiment_id]
        for callback in self._callbacks:
            try:
                callback(experiment_id, old_status, new_status, metadata)
            except Exception as e:
                print(f"警告: 回调执行失败: {e}")

    @staticmethod
    def _timestamp() -> str:
        """生成当前时间戳字符串。"""
        return datetime.now().isoformat()

    def create_experiment(self, experiment_id: str, candidate_id: str,
                         priority: int = 0, metadata: Optional[Dict[str, Any]] = None) -> None:
        """创建新实验记录。

        Args:
            experiment_id: 实验唯一标识符
            candidate_id: 候选配置ID
            priority: 实验优先级（数字越小优先级越高）
            metadata: 额外的自定义元数据

        Raises:
            ValueError: 如果实验ID已存在
        """
        if experiment_id in self._experiments:
            raise ValueError(f"实验 {experiment_id} 已存在")

        exp_metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            candidate_id=candidate_id,
            status=ExperimentStatus.PENDING,
            priority=priority,
            created_at=self._timestamp(),
            metadata=metadata or {}
        )

        self._experiments[experiment_id] = exp_metadata
        self._save_state()

    def update_status(self, experiment_id: str, new_status: ExperimentStatus,
                     gpu_allocation: Optional[int] = None,
                     error_message: Optional[str] = None) -> None:
        """更新实验状态。

        Args:
            experiment_id: 实验ID
            new_status: 新状态
            gpu_allocation: GPU分配（可选，通常在RUNNING时设置）
            error_message: 错误信息（可选，通常在FAILED时设置）

        Raises:
            KeyError: 如果实验ID不存在
        """
        if experiment_id not in self._experiments:
            raise KeyError(f"实验 {experiment_id} 不存在")

        metadata = self._experiments[experiment_id]
        old_status = metadata.status

        # 更新状态
        metadata.status = new_status

        # 根据状态更新时间戳
        if new_status == ExperimentStatus.RUNNING and metadata.started_at is None:
            metadata.started_at = self._timestamp()

        if new_status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED):
            metadata.completed_at = self._timestamp()

        # 更新 GPU 分配
        if gpu_allocation is not None:
            metadata.gpu_allocation = gpu_allocation

        # 更新错误信息
        if error_message is not None:
            metadata.error_message = error_message

        # 保存并触发回调
        self._save_state()
        if old_status != new_status:
            self._trigger_callbacks(experiment_id, old_status, new_status)

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """获取实验元数据。

        Args:
            experiment_id: 实验ID

        Returns:
            实验元数据，如果不存在返回 None
        """
        return self._experiments.get(experiment_id)

    def get_by_status(self, status: ExperimentStatus) -> List[ExperimentMetadata]:
        """查询指定状态的所有实验。

        Args:
            status: 实验状态

        Returns:
            符合条件的实验列表
        """
        return [exp for exp in self._experiments.values() if exp.status == status]

    def get_by_priority(self, status: Optional[ExperimentStatus] = None) -> List[ExperimentMetadata]:
        """按优先级排序查询实验（可选过滤状态）。

        Args:
            status: 可选的状态过滤条件，None表示查询所有状态

        Returns:
            按优先级排序的实验列表（优先级数字越小排在前面）
        """
        experiments: List[ExperimentMetadata] = list(self._experiments.values())

        if status is not None:
            experiments = [exp for exp in experiments if exp.status == status]

        return sorted(experiments, key=lambda exp: exp.priority)

    def get_all(self) -> List[ExperimentMetadata]:
        """获取所有实验。

        Returns:
            所有实验的列表
        """
        return list(self._experiments.values())

    def delete_experiment(self, experiment_id: str) -> bool:
        """删除实验记录。

        Args:
            experiment_id: 实验ID

        Returns:
            如果删除成功返回 True，实验不存在返回 False
        """
        if experiment_id not in self._experiments:
            return False

        del self._experiments[experiment_id]
        self._save_state()
        return True

    def update_metadata(self, experiment_id: str, metadata_updates: Dict[str, Any]) -> None:
        """更新实验的自定义元数据。

        Args:
            experiment_id: 实验ID
            metadata_updates: 要更新的元数据字典

        Raises:
            KeyError: 如果实验ID不存在
        """
        if experiment_id not in self._experiments:
            raise KeyError(f"实验 {experiment_id} 不存在")

        self._experiments[experiment_id].metadata.update(metadata_updates)
        self._save_state()

    def count_by_status(self) -> Dict[ExperimentStatus, int]:
        """统计各状态的实验数量。

        Returns:
            状态到数量的映射
        """
        counts = {status: 0 for status in ExperimentStatus}
        for exp in self._experiments.values():
            counts[exp.status] += 1
        return counts
