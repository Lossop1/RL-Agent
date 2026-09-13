"""仿真器后端工厂函数和包装器。

P6.1第三步：重构训练入口使用后端抽象。

设计原则：
- gym.make()作为内部实现：上层通过SimulatorBackend协议消费
- 支持多后端切换：backend_type参数控制使用IsaacLab/MuJoCo
- 保持原有接口：现有训练代码最小化修改
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from autotuner.simulation.simulator_protocol import SimulatorBackend
from autotuner.simulation.isaaclab_adapter import IsaacLabAdapter

# 延迟导入gymnasium，避免测试时必须安装
try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore


def create_backend(
    task: str,
    cfg: Any,
    backend_type: str = "isaaclab",
    render_mode: Optional[str] = None,
) -> SimulatorBackend:
    """工厂函数：创建仿真器后端。

    Args:
        task: 环境任务名称（如"RobotLab-Isaac-Taili-AMP-Blind-Direct-v0"）
        cfg: 环境配置对象（DirectRLEnvCfg或等效配置）
        backend_type: 后端类型，"isaaclab"或"mujoco"
        render_mode: 渲染模式，传递给环境构造函数

    Returns:
        实现SimulatorBackend协议的后端对象

    Raises:
        ValueError: 如果backend_type不支持

    示例：
        from isaaclab_tasks.utils import parse_env_cfg
        env_cfg = parse_env_cfg("RobotLab-Isaac-Taili-AMP-Blind-Direct-v0", device="cuda:0")
        backend = create_backend("RobotLab-Isaac-Taili-AMP-Blind-Direct-v0", env_cfg)
        obs = backend.reset()
        obs, rewards, dones, info = backend.step(actions)
    """
    if backend_type == "isaaclab":
        if gym is None:
            raise ImportError(
                "gymnasium is required for IsaacLab backend. "
                "Install it with: pip install gymnasium"
            )
        # gym.make()是IsaacLab的标准环境创建入口
        env = gym.make(task, cfg=cfg, render_mode=render_mode)
        return IsaacLabAdapter(env)
    elif backend_type == "mujoco":
        # MuJoCo后端待实现（P6.1步骤5）
        raise NotImplementedError("MuJoCo backend not yet implemented")
    else:
        raise ValueError(f"Unsupported backend_type: {backend_type}")


__all__ = [
    "create_backend",
]
