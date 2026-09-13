"""IsaacLab适配器 - 包装现有IsaacLab环境实现SimulatorBackend协议。

P6.1第二步：实现IsaacLabAdapter，验证等价性（MAE<1e-6）。

设计原则：
- 零侵入：不修改现有TailiAmpEnv/TailiBlindTPEnv实现
- 薄包装：仅实现协议方法到IsaacLab环境方法的映射
- 完全等价：保证行为与直接使用IsaacLab环境一致
- 类型安全：返回值类型与协议定义严格匹配

集成策略：
- 现有训练代码直接使用TailiBlindTPEnv（继承DirectRLEnv）
- 适配器包装现有环境，暴露SimulatorBackend接口
- 未来MuJoCo后端通过MuJoCoAdapter实现相同接口
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from autotuner.simulation.simulator_protocol import SimulatorBackend

# IsaacLab导入延迟到运行时，避免测试时必须安装IsaacLab
try:
    from isaaclab.envs import DirectRLEnv
except ImportError:
    DirectRLEnv = None  # 类型检查时使用，运行时ImportError会在__init__中抛出


class IsaacLabAdapter:
    """IsaacLab环境适配器，实现SimulatorBackend协议。

    包装任何继承isaaclab.envs.DirectRLEnv的环境类，提供统一的SimulatorBackend接口。

    示例用法：
        env_cfg = TailiAmpEnvCfg(...)
        isaaclab_env = TailiBlindTPEnv(env_cfg)
        backend = IsaacLabAdapter(isaaclab_env)

        # 通过SimulatorBackend协议使用
        obs = backend.reset()
        obs, rewards, dones, info = backend.step(actions)
        backend.close()

    设计说明：
    - 不修改原始env对象，仅转发方法调用
    - 观测字典格式与IsaacLab保持一致（"policy"/"critic"键）
    - info字典直接转发，包含IsaacLab提供的所有统计量
    - 渲染功能取决于原始env是否实现render()方法
    """

    def __init__(self, env):
        """初始化IsaacLab适配器。

        Args:
            env: 任何继承DirectRLEnv的IsaacLab环境实例

        Raises:
            TypeError: 如果env不是DirectRLEnv的实例
            ImportError: 如果IsaacLab未安装且env不符合协议
        """
        # 测试环境下允许鸭子类型：只要对象有必需的方法和属性即可
        if DirectRLEnv is not None and not isinstance(env, DirectRLEnv):
            raise TypeError(
                f"IsaacLabAdapter requires DirectRLEnv, got {type(env).__name__}"
            )

        # 验证env至少有必需的接口（支持测试时的mock对象）
        required_attrs = ["num_envs", "device", "reset", "step", "close", "_get_observations"]
        missing = [attr for attr in required_attrs if not hasattr(env, attr)]
        if missing:
            raise TypeError(
                f"Environment missing required attributes: {missing}. "
                f"Expected DirectRLEnv-compatible interface."
            )

        self._env = env

    # ==================== SimulatorBackend协议实现 ====================

    @property
    def num_envs(self) -> int:
        """环境实例数量。"""
        return self._env.num_envs

    @property
    def device(self) -> torch.device:
        """张量所在设备。"""
        return self._env.device

    def reset(
        self, env_ids: Optional[Sequence[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """重置指定环境。

        Args:
            env_ids: 要重置的环境索引，None表示重置所有环境

        Returns:
            观测字典，包含"policy"键（必需）和可能的"critic"键

        Note:
            DirectRLEnv.reset()接受env_ids参数，None时重置所有环境
        """
        obs, _ = self._env.reset(env_ids=env_ids)
        return obs

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """执行一步仿真。

        Args:
            actions: 动作张量 (num_envs, action_dim)

        Returns:
            (observations, rewards, dones, info) 元组

        Note:
            DirectRLEnv.step()返回(obs, reward, terminated, truncated, info)
            终止条件合并为dones = terminated | truncated
        """
        obs, rewards, terminated, truncated, info = self._env.step(actions)
        dones = terminated | truncated
        return obs, rewards, dones, info

    def close(self) -> None:
        """关闭仿真器并释放资源。"""
        self._env.close()

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """获取当前观测。

        Returns:
            观测字典，格式同reset()返回值

        Note:
            DirectRLEnv._get_observations()方法返回当前观测而不推进仿真
        """
        return self._env._get_observations()

    def render(self) -> Optional[np.ndarray]:
        """渲染当前帧。

        Returns:
            RGB图像数组 (H, W, 3)，如果不支持渲染则返回None

        Note:
            IsaacLab环境可能通过sim.render()提供渲染，但DirectRLEnv
            不强制要求render()方法存在，因此此方法可能返回None
        """
        if hasattr(self._env, "render") and callable(self._env.render):
            return self._env.render()
        return None


def create_isaaclab_backend(env) -> SimulatorBackend:
    """工厂函数：创建IsaacLab后端。

    Args:
        env: IsaacLab环境实例

    Returns:
        实现SimulatorBackend协议的适配器

    Note:
        返回值满足SimulatorBackend协议，可用于类型检查和运行时验证
    """
    return IsaacLabAdapter(env)


__all__ = [
    "IsaacLabAdapter",
    "create_isaaclab_backend",
]
