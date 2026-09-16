"""测试backend_factory工厂函数。

验证create_backend()正确创建不同类型的后端并返回协议兼容对象。
"""
import pytest
import torch

from autotuner.simulation.backend_factory import create_backend
from autotuner.simulation.simulator_protocol import validate_backend
from autotuner.simulation.isaaclab_adapter import IsaacLabAdapter


# 测试时不依赖完整IsaacLab，使用模拟环境
try:
    from isaaclab.envs import DirectRLEnv as _DirectRLEnvBase
except ImportError:
    class _DirectRLEnvBase:
        """最小DirectRLEnv基类定义，仅用于测试。"""
        pass


class MockEnvCfg:
    """模拟环境配置对象。"""
    def __init__(self):
        self.seed = 42
        self.num_envs = 4
        self.device = "cpu"


class MockDirectRLEnv(_DirectRLEnvBase):
    """模拟DirectRLEnv用于测试backend_factory。"""

    def __init__(self, cfg=None, render_mode=None):
        self.cfg = cfg or MockEnvCfg()
        self.render_mode = render_mode
        self.num_envs = getattr(self.cfg, "num_envs", 4)
        self.device = torch.device(getattr(self.cfg, "device", "cpu"))
        self._obs_dim = 10
        self._action_dim = 12

    def reset(self, env_ids=None):
        n = len(env_ids) if env_ids is not None else self.num_envs
        obs = {
            "policy": torch.zeros(n, self._obs_dim, device=self.device),
            "critic": torch.zeros(n, self._obs_dim + 5, device=self.device),
        }
        return obs, {}

    def step(self, actions):
        obs = {
            "policy": torch.zeros(self.num_envs, self._obs_dim, device=self.device),
            "critic": torch.zeros(self.num_envs, self._obs_dim + 5, device=self.device),
        }
        rewards = torch.zeros(self.num_envs, device=self.device)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return obs, rewards, terminated, truncated, {}

    def close(self):
        pass

    def _get_observations(self):
        return {
            "policy": torch.zeros(self.num_envs, self._obs_dim, device=self.device),
            "critic": torch.zeros(self.num_envs, self._obs_dim + 5, device=self.device),
        }

    def render(self):
        return None


class TestBackendFactory:
    """测试create_backend()工厂函数。"""

    def test_create_isaaclab_backend_direct(self):
        """测试直接使用适配器创建IsaacLab后端。"""
        # 不依赖gymnasium，直接测试适配器包装
        env = MockDirectRLEnv()
        backend = IsaacLabAdapter(env)

        assert validate_backend(backend)
        assert backend.num_envs == 4
        assert backend.device == torch.device("cpu")

    def test_adapter_unwrapped_is_the_raw_env(self):
        """unwrapped 原样交出底层环境。

        IsaacLab 自带的 skrl 包装器按 env.unwrapped 判断底层类型，并把 step/reset 直接
        发给传进来的对象。本适配器的 reset 只返回观测、step 返回四元组，契约与 skrl 期望的
        不同，所以入口应当取 .unwrapped 交出去——这条断言盯的就是那个出口没有被改坏。
        """
        env = MockDirectRLEnv()
        backend = IsaacLabAdapter(env)
        assert backend.unwrapped is env

    def test_isaaclab_backend_functional(self):
        """测试IsaacLab后端功能完整性。"""
        env = MockDirectRLEnv()
        backend = IsaacLabAdapter(env)

        # 测试reset
        obs = backend.reset()
        assert "policy" in obs
        assert obs["policy"].shape == (4, 10)

        # 测试step
        actions = torch.zeros(4, 12)
        obs, rewards, dones, info = backend.step(actions)
        assert obs["policy"].shape == (4, 10)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)

        # 测试get_observations
        obs = backend.get_observations()
        assert "policy" in obs

        # 测试close
        backend.close()

    def test_backend_with_render_mode(self):
        """测试创建带渲染模式的后端。"""
        env = MockDirectRLEnv(render_mode="rgb_array")
        backend = IsaacLabAdapter(env)
        assert validate_backend(backend)
        assert env.render_mode == "rgb_array"

    def test_mujoco_backend_not_implemented(self):
        """测试MuJoCo后端尚未实现。"""
        cfg = MockEnvCfg()
        with pytest.raises(NotImplementedError, match="MuJoCo backend not yet implemented"):
            create_backend("RobotLab-Isaac-Mock-Test-v0", cfg, backend_type="mujoco")

    def test_unsupported_backend_type(self):
        """测试不支持的后端类型。"""
        cfg = MockEnvCfg()
        with pytest.raises(ValueError, match="Unsupported backend_type"):
            create_backend("RobotLab-Isaac-Mock-Test-v0", cfg, backend_type="pybullet")
