"""仿真器后端协议定义 - P6.1第一步。

定义统一的仿真器接口协议，支持IsaacLab/MuJoCo多后端切换。
设计目标：
- 协议定义核心操作契约（reset/step/get_observations等）
- 类型安全：使用Protocol实现鸭子类型检查
- 最小侵入：不改变现有环境实现，仅定义接口
- 可扩展：支持新后端通过实现协议接入

架构原则：
- Layer 6物理仿真层抽象
- 上层（训练/诊断）通过协议接口消费仿真器
- 下层（IsaacLab/MuJoCo）通过适配器实现协议
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch


class SimulatorBackend(Protocol):
    """仿真器后端协议接口。

    所有仿真器后端（IsaacLab/MuJoCo）必须实现此协议的所有方法。
    使用Protocol而非ABC，支持结构化子类型（structural subtyping）。

    设计原则：
    - 最小接口：仅定义训练/评估必需的核心操作
    - 类型明确：所有返回值类型显式声明
    - 状态透明：不隐藏内部状态，通过get_*方法暴露
    - 错误明确：异常情况通过返回值或异常传递
    """

    # ==================== 环境生命周期 ====================

    @property
    def num_envs(self) -> int:
        """环境实例数量（并行环境数）。"""
        ...

    @property
    def device(self) -> torch.device:
        """张量所在设备（cuda或cpu）。"""
        ...

    def reset(self, env_ids: Optional[Sequence[int]] = None) -> Dict[str, torch.Tensor]:
        """重置指定环境实例。

        Args:
            env_ids: 要重置的环境索引序列，None表示重置所有环境

        Returns:
            观测字典，包含：
                - "policy": 策略观测 (num_reset_envs, obs_dim)
                - "critic": 评论家观测（如果存在）
                - 其他后端特定观测

        Note:
            重置后环境状态应与初始状态一致（位置/速度/接触状态）
        """
        ...

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """执行一步仿真。

        Args:
            actions: 动作张量 (num_envs, action_dim)

        Returns:
            (observations, rewards, dones, info) 元组：
                - observations: 观测字典，同reset()返回格式
                - rewards: 奖励标量 (num_envs,)
                - dones: 终止标志 (num_envs,)，True表示回合结束
                - info: 附加信息字典，包含统计量/调试信息

        Note:
            dones=True的环境会自动重置（或由上层调用reset()）
        """
        ...

    def close(self) -> None:
        """关闭仿真器并释放资源。

        清理GPU内存、关闭渲染窗口、释放文件句柄等。
        调用后不应再调用其他方法。
        """
        ...

    # ==================== 观测与状态查询 ====================

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """获取当前观测（不推进仿真）。

        Returns:
            观测字典，格式同reset()返回值

        Note:
            用于诊断/可视化，不影响仿真状态
        """
        ...

    # ==================== 渲染（可选） ====================

    def render(self) -> Optional[np.ndarray]:
        """渲染当前帧（可选功能）。

        Returns:
            RGB图像数组 (H, W, 3)，uint8类型
            如果后端不支持渲染，返回None

        Note:
            渲染可能很慢，仅用于可视化调试
        """
        ...


class SimulatorMetadata(Protocol):
    """仿真器元数据协议（可选）。

    提供环境配置和能力查询接口，用于运行时检查和日志记录。
    不是所有后端都需要实现，仅用于增强可观测性。
    """

    @property
    def backend_name(self) -> str:
        """后端名称（如'IsaacLab'/'MuJoCo'）。"""
        ...

    @property
    def simulator_dt(self) -> float:
        """仿真时间步长（秒）。"""
        ...

    @property
    def observation_space_dim(self) -> Dict[str, int]:
        """观测空间维度字典。

        Returns:
            {"policy": 1407, "critic": 1512, ...}
        """
        ...

    @property
    def action_space_dim(self) -> int:
        """动作空间维度。"""
        ...


def validate_backend(backend: Any) -> bool:
    """运行时验证对象是否实现SimulatorBackend协议。

    Args:
        backend: 待验证的后端对象

    Returns:
        True如果实现了所有必需方法，否则False

    Note:
        Protocol类型检查在静态分析时进行，此函数用于运行时验证
    """
    required_methods = [
        "reset",
        "step",
        "close",
        "get_observations",
        "render",
    ]
    required_properties = [
        "num_envs",
        "device",
    ]

    for method in required_methods:
        if not hasattr(backend, method) or not callable(getattr(backend, method)):
            return False

    for prop in required_properties:
        if not hasattr(backend, prop):
            return False

    return True


__all__ = [
    "SimulatorBackend",
    "SimulatorMetadata",
    "validate_backend",
]
