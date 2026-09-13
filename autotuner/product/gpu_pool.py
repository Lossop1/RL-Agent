"""GPU资源池 - 追踪和分配GPU资源给实验。

提供细粒度的GPU级别管理，避免多个实验抢占同一GPU。支持资源预留机制（baseline优先）。
本地nvidia-smi探测或远程SSH探测。
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class GPUResource:
    """GPU资源信息。

    Attributes:
        gpu_id: GPU标识符（通常是索引，如"0", "1"）
        name: GPU型号名称（如"NVIDIA RTX 4090"）
        memory_total_mb: 总显存（MB）
        memory_used_mb: 已用显存（MB）
        utilization_pct: GPU使用率（0-100）
        available: 是否可用于分配
        allocated_to: 当前分配给哪个实验ID（空表示未分配）
        last_updated: 最后更新时间戳
    """
    gpu_id: str
    name: str = ""
    memory_total_mb: float = 0.0
    memory_used_mb: float = 0.0
    utilization_pct: float = 0.0
    available: bool = True
    allocated_to: str = ""
    last_updated: float = field(default_factory=time.time)

    def is_idle(self, memory_threshold_mb: float = 500.0, utilization_threshold: float = 10.0) -> bool:
        """判断GPU是否空闲（基于显存和使用率阈值）。"""
        return (
            self.memory_used_mb < memory_threshold_mb
            and self.utilization_pct < utilization_threshold
        )


@dataclass
class AllocationRecord:
    """GPU分配记录。

    Attributes:
        experiment_id: 实验ID
        gpu_ids: 分配的GPU ID列表
        allocated_at: 分配时间戳
        priority: 优先级（baseline=100, regular=50）
    """
    experiment_id: str
    gpu_ids: List[str]
    allocated_at: float
    priority: int = 50  # baseline=100, regular=50


class GPUResourcePool:
    """GPU资源池管理器。

    追踪可用GPU，分配/释放GPU给实验，避免资源冲突。
    支持本地nvidia-smi探测或远程SSH探测。
    """

    def __init__(
        self,
        remote_ssh=None,
        state_file: Optional[str] = None,
        auto_discover: bool = True,
    ):
        """初始化GPU资源池。

        Args:
            remote_ssh: 远程SSH连接对象（可选，用于远程GPU探测）
            state_file: 状态持久化文件路径（可选）
            auto_discover: 是否自动探测GPU
        """
        self.remote_ssh = remote_ssh
        self.state_file = Path(state_file) if state_file else None
        self._lock = threading.Lock()

        # GPU资源字典: {gpu_id: GPUResource}
        self._gpus: Dict[str, GPUResource] = {}

        # 分配记录: {experiment_id: AllocationRecord}
        self._allocations: Dict[str, AllocationRecord] = {}

        # 加载持久化状态
        if self.state_file and self.state_file.exists():
            self._load_state()

        # 自动探测GPU
        if auto_discover:
            self.discover_gpus()

    def discover_gpus(self) -> List[GPUResource]:
        """探测可用GPU（本地nvidia-smi或远程probe）。

        Returns:
            GPU资源列表
        """
        with self._lock:
            if self.remote_ssh is not None:
                gpus = self._discover_remote_gpus()
            else:
                gpus = self._discover_local_gpus()

            # 更新GPU资源字典，保留已分配状态
            for gpu in gpus:
                if gpu.gpu_id in self._gpus:
                    # 保留分配信息
                    gpu.allocated_to = self._gpus[gpu.gpu_id].allocated_to
                    gpu.available = self._gpus[gpu.gpu_id].available
                self._gpus[gpu.gpu_id] = gpu

            self._save_state()
            return list(self._gpus.values())

    def _discover_local_gpus(self) -> List[GPUResource]:
        """通过本地nvidia-smi探测GPU。"""
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

            if result.returncode != 0:
                return []

            gpus: List[GPUResource] = []
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpu = GPUResource(
                        gpu_id=parts[0],
                        name=parts[1],
                        memory_used_mb=float(parts[2]) if parts[2] else 0.0,
                        memory_total_mb=float(parts[3]) if parts[3] else 0.0,
                        utilization_pct=float(parts[4]) if parts[4] else 0.0,
                        last_updated=time.time(),
                    )
                    gpus.append(gpu)

            return gpus
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
            # nvidia-smi不可用或超时
            return []

    def _discover_remote_gpus(self) -> List[GPUResource]:
        """通过远程SSH探测GPU。"""
        try:
            cmd = (
                "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu "
                "--format=csv,noheader,nounits 2>/dev/null || true"
            )
            output = self.remote_ssh.exec_out(cmd, timeout=10) or ""

            gpus: List[GPUResource] = []
            for line in output.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpu = GPUResource(
                        gpu_id=parts[0],
                        name=parts[1],
                        memory_used_mb=float(parts[2]) if parts[2] else 0.0,
                        memory_total_mb=float(parts[3]) if parts[3] else 0.0,
                        utilization_pct=float(parts[4]) if parts[4] else 0.0,
                        last_updated=time.time(),
                    )
                    gpus.append(gpu)

            return gpus
        except Exception:
            return []

    def allocate(
        self,
        count: int,
        experiment_id: str,
        priority: int = 50,
        allow_busy: bool = False,
    ) -> List[str]:
        """分配指定数量的GPU给实验。

        Args:
            count: 需要的GPU数量
            experiment_id: 实验ID
            priority: 优先级（baseline=100, regular=50）
            allow_busy: 是否允许分配忙碌的GPU（默认只分配空闲GPU）

        Returns:
            分配的GPU ID列表

        Raises:
            RuntimeError: 可用GPU不足
            ValueError: 实验已有分配
        """
        with self._lock:
            # 检查是否已分配
            if experiment_id in self._allocations:
                raise ValueError(
                    f"实验 {experiment_id} 已经分配了GPU: "
                    f"{self._allocations[experiment_id].gpu_ids}"
                )

            # 刷新GPU状态
            self.discover_gpus()

            # 查找可用GPU
            available_gpus: List[GPUResource] = []
            for gpu in self._gpus.values():
                if not gpu.allocated_to:
                    if allow_busy or gpu.is_idle():
                        available_gpus.append(gpu)

            # 检查可用GPU数量
            if len(available_gpus) < count:
                raise RuntimeError(
                    f"可用GPU不足：需要 {count} 个，但只有 {len(available_gpus)} 个可用"
                )

            # 按显存使用率排序，优先分配空闲GPU
            available_gpus.sort(key=lambda g: g.memory_used_mb)

            # 分配GPU
            allocated_ids = [gpu.gpu_id for gpu in available_gpus[:count]]
            for gpu_id in allocated_ids:
                self._gpus[gpu_id].allocated_to = experiment_id
                self._gpus[gpu_id].available = False

            # 记录分配
            self._allocations[experiment_id] = AllocationRecord(
                experiment_id=experiment_id,
                gpu_ids=allocated_ids,
                allocated_at=time.time(),
                priority=priority,
            )

            self._save_state()
            return allocated_ids

    def release(self, gpu_ids: List[str], experiment_id: str) -> None:
        """释放已分配的GPU。

        Args:
            gpu_ids: 要释放的GPU ID列表
            experiment_id: 实验ID

        Raises:
            ValueError: GPU未分配给该实验
        """
        with self._lock:
            # 验证分配记录
            if experiment_id not in self._allocations:
                raise ValueError(f"实验 {experiment_id} 没有GPU分配记录")

            record = self._allocations[experiment_id]
            for gpu_id in gpu_ids:
                if gpu_id not in record.gpu_ids:
                    raise ValueError(
                        f"GPU {gpu_id} 未分配给实验 {experiment_id}"
                    )

                if gpu_id in self._gpus:
                    self._gpus[gpu_id].allocated_to = ""
                    self._gpus[gpu_id].available = True

            # 更新分配记录
            record.gpu_ids = [gid for gid in record.gpu_ids if gid not in gpu_ids]
            if not record.gpu_ids:
                del self._allocations[experiment_id]

            self._save_state()

    def release_all(self, experiment_id: str) -> None:
        """释放实验的所有GPU。

        Args:
            experiment_id: 实验ID
        """
        with self._lock:
            if experiment_id not in self._allocations:
                return

            gpu_ids = self._allocations[experiment_id].gpu_ids.copy()
            self.release(gpu_ids, experiment_id)

    def get_available(self, only_idle: bool = True) -> List[GPUResource]:
        """获取当前可用GPU列表。

        Args:
            only_idle: 是否只返回空闲GPU（默认True）

        Returns:
            可用GPU资源列表
        """
        with self._lock:
            self.discover_gpus()

            available = []
            for gpu in self._gpus.values():
                if not gpu.allocated_to:
                    if not only_idle or gpu.is_idle():
                        available.append(gpu)

            return available

    def get_allocation(self, experiment_id: str) -> List[str]:
        """查询实验的GPU分配。

        Args:
            experiment_id: 实验ID

        Returns:
            分配的GPU ID列表（空列表表示未分配）
        """
        with self._lock:
            record = self._allocations.get(experiment_id)
            return record.gpu_ids.copy() if record else []

    def get_all_allocations(self) -> Dict[str, List[str]]:
        """获取所有分配记录。

        Returns:
            {experiment_id: [gpu_ids]}
        """
        with self._lock:
            return {
                exp_id: record.gpu_ids.copy()
                for exp_id, record in self._allocations.items()
            }

    def _update_usage(self) -> None:
        """后台更新GPU使用率（可选，用于智能调度）。

        这是一个辅助方法，可由外部定时调用或在需要时手动调用。
        """
        self.discover_gpus()

    def _save_state(self) -> None:
        """持久化状态到文件。"""
        if not self.state_file:
            return

        try:
            state = {
                "gpus": {
                    gpu_id: {
                        "gpu_id": gpu.gpu_id,
                        "name": gpu.name,
                        "allocated_to": gpu.allocated_to,
                        "available": gpu.available,
                    }
                    for gpu_id, gpu in self._gpus.items()
                },
                "allocations": {
                    exp_id: {
                        "experiment_id": record.experiment_id,
                        "gpu_ids": record.gpu_ids,
                        "allocated_at": record.allocated_at,
                        "priority": record.priority,
                    }
                    for exp_id, record in self._allocations.items()
                },
            }

            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception:
            # 持久化失败不影响核心功能
            pass

    def _load_state(self) -> None:
        """从文件加载状态。"""
        if not self.state_file or not self.state_file.exists():
            return

        try:
            state = json.loads(self.state_file.read_text(encoding="utf-8"))

            # 加载分配记录
            for exp_id, rec_data in state.get("allocations", {}).items():
                self._allocations[exp_id] = AllocationRecord(
                    experiment_id=rec_data["experiment_id"],
                    gpu_ids=rec_data["gpu_ids"],
                    allocated_at=rec_data["allocated_at"],
                    priority=rec_data.get("priority", 50),
                )

            # 加载GPU状态（仅分配信息，实际状态需要重新探测）
            for gpu_id, gpu_data in state.get("gpus", {}).items():
                if gpu_id not in self._gpus:
                    self._gpus[gpu_id] = GPUResource(
                        gpu_id=gpu_id,
                        name=gpu_data.get("name", ""),
                    )
                self._gpus[gpu_id].allocated_to = gpu_data.get("allocated_to", "")
                self._gpus[gpu_id].available = gpu_data.get("available", True)
        except Exception:
            # 加载失败使用空状态
            pass


def create_pool(
    remote_ssh=None,
    state_file: Optional[str] = None,
) -> GPUResourcePool:
    """工厂函数：创建GPU资源池实例。

    Args:
        remote_ssh: 远程SSH连接对象（可选）
        state_file: 状态文件路径（可选）

    Returns:
        GPUResourcePool实例
    """
    return GPUResourcePool(
        remote_ssh=remote_ssh,
        state_file=state_file,
        auto_discover=True,
    )
