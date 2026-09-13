"""测试IsaacLabAdapter适配器实现。

验证适配器正确包装IsaacLab环境并满足SimulatorBackend协议。
"""
import torch
import pytest

from autotuner.simulation.isaaclab_adapter import IsaacLabAdapter, create_isaaclab_backend
from autotuner.simulation.simulator_protocol import validate_backend

# 测试时不依赖完整IsaacLab，使用模拟基类
try:
    from isaaclab.envs import DirectRLEnv as _DirectRLEnvBase
except ImportError:
    # IsaacLab未安装时，定义最小基类用于测试
    class _DirectRLEnvBase:
        """最小DirectRLEnv基类定义，仅用于测试。"""
        pass


class MockDirectRLEnv(_DirectRLEnvBase):
    """模拟DirectRLEnv用于测试，避免依赖完整IsaacLab环境。"""

    def __init__(self, num_envs: int = 4, device: str = "cpu"):
        # 不调用super().__init__，避免依赖完整IsaacLab场景配置
        self.num_envs = num_envs
        self.device = torch.device(device)
        self._obs_dim = 10
        self._action_dim = 12
        self._step_count = 0

    def reset(self, env_ids=None):
        """重置环境。"""
        n = len(env_ids) if env_ids is not None else self.num_envs
        obs = {
            "policy": torch.zeros(n, self._obs_dim, device=self.device),
            "critic": torch.zeros(n, self._obs_dim + 5, device=self.device),
        }
        info = {"reset_count": n}
        return obs, info

    def step(self, actions):
        """执行一步仿真。"""
        self._step_count += 1
        obs = {
            "policy": torch.zeros(self.num_envs, self._obs_dim, device=self.device),
            "critic": torch.zeros(self.num_envs, self._obs_dim + 5, device=self.device),
        }
        rewards = torch.ones(self.num_envs, device=self.device) * 0.5
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        info = {"step_count": self._step_count}
        return obs, rewards, terminated, truncated, info

    def close(self):
        """关闭环境。"""
        pass

    def _get_observations(self):
        """获取当前观测（不推进仿真）。"""
        return {
            "policy": torch.zeros(self.num_envs, self._obs_dim, device=self.device),
            "critic": torch.zeros(self.num_envs, self._obs_dim + 5, device=self.device),
        }

    def render(self):
        """渲染当前帧（模拟不支持渲染）。"""
        return None


class TestIsaacLabAdapter:
    """测试IsaacLabAdapter适配器。"""

    def test_adapter_implements_protocol(self):
        """测试适配器实现SimulatorBackend协议。"""
        env = MockDirectRLEnv(num_envs=4)
        adapter = IsaacLabAdapter(env)
        assert validate_backend(adapter)

    def test_adapter_initialization(self):
        """测试适配器初始化。"""
        env = MockDirectRLEnv(num_envs=8, device="cpu")
        adapter = IsaacLabAdapter(env)
        assert adapter.num_envs == 8
        assert adapter.device == torch.device("cpu")

    def test_adapter_initialization_type_check(self):
        """测试适配器初始化类型检查。"""
        # 非DirectRLEnv对象应抛出TypeError
        try:
            IsaacLabAdapter("not_an_env")
            assert False, "应该抛出TypeError"
        except TypeError as e:
            assert "DirectRLEnv" in str(e)

    def test_adapter_reset_all_envs(self):
        """测试重置所有环境。"""
        env = MockDirectRLEnv(num_envs=4)
        adapter = IsaacLabAdapter(env)

        obs = adapter.reset()
        assert "policy" in obs
        assert "critic" in obs
        assert obs["policy"].shape == (4, 10)
        assert obs["critic"].shape == (4, 15)

    def test_adapter_reset_partial_envs(self):
        """测试重置部分环境。"""
        env = MockDirectRLEnv(num_envs=4)
        adapter = IsaacLabAdapter(env)

        obs = adapter.reset(env_ids=[0, 2])
        assert "policy" in obs
        assert obs["policy"].shape == (2, 10)

    def test_adapter_step(self):
        """测试step方法。"""
        env = MockDirectRLEnv(num_envs=4)
        adapter = IsaacLabAdapter(env)

        actions = torch.zeros(4, 12)
        obs, rewards, dones, info = adapter.step(actions)

        assert "policy" in obs
        assert obs["policy"].shape == (4, 10)
        assert rewards.shape == (4,)
        assert torch.allclose(rewards, torch.tensor(0.5))
        assert dones.shape == (4,)
        assert dones.dtype == torch.bool
        assert "step_count" in info

    def test_adapter_step_terminated_or_truncated(self):
        """测试terminated和truncated合并为dones。"""
        env = MockDirectRLEnv(num_envs=4)

        # 修改环境使某些实例terminated
        original_step = env.step

        def step_with_terminated(actions):
            obs, rewards, terminated, truncated, info = original_step(actions)
            terminated[0] = True
            truncated[2] = True
            return obs, rewards, terminated, truncated, info

        env.step = step_with_terminated

        adapter = IsaacLabAdapter(env)
        actions = torch.zeros(4, 12)
        obs, rewards, dones, info = adapter.step(actions)

        # dones应为terminated | truncated
        assert dones[0]  # terminated[0] = True
        assert not dones[1]
        assert dones[2]  # truncated[2] = True
        assert not dones[3]

    def test_adapter_get_observations(self):
        """测试get_observations方法。"""
        env = MockDirectRLEnv(num_envs=4)
        adapter = IsaacLabAdapter(env)

        obs = adapter.get_observations()
        assert "policy" in obs
        assert obs["policy"].shape == (4, 10)

    def test_adapter_render(self):
        """测试render方法。"""
        env = MockDirectRLEnv(num_envs=4)
        adapter = IsaacLabAdapter(env)

        # MockDirectRLEnv.render()返回None
        img = adapter.render()
        assert img is None

    def test_adapter_render_when_not_implemented(self):
        """测试环境未实现render时的行为。"""
        # 创建没有render方法的环境对象
        class EnvWithoutRender:
            def __init__(self):
                self.num_envs = 4
                self.device = torch.device("cpu")

            def reset(self, env_ids=None):
                n = len(env_ids) if env_ids is not None else self.num_envs
                return {"policy": torch.zeros(n, 10)}, {}

            def step(self, actions):
                obs = {"policy": torch.zeros(self.num_envs, 10)}
                rewards = torch.zeros(self.num_envs)
                terminated = torch.zeros(self.num_envs, dtype=torch.bool)
                truncated = torch.zeros(self.num_envs, dtype=torch.bool)
                return obs, rewards, terminated, truncated, {}

            def close(self):
                pass

            def _get_observations(self):
                return {"policy": torch.zeros(self.num_envs, 10)}

        env = EnvWithoutRender()
        adapter = IsaacLabAdapter(env)
        img = adapter.render()
        assert img is None

    def test_adapter_close(self):
        """测试close方法。"""
        env = MockDirectRLEnv(num_envs=4)
        adapter = IsaacLabAdapter(env)
        adapter.close()  # 不应抛出异常

    def test_adapter_properties(self):
        """测试适配器属性。"""
        env = MockDirectRLEnv(num_envs=16, device="cpu")
        adapter = IsaacLabAdapter(env)

        assert adapter.num_envs == 16
        assert adapter.device == torch.device("cpu")


class TestCreateIsaacLabBackend:
    """测试create_isaaclab_backend工厂函数。"""

    def test_factory_creates_valid_backend(self):
        """测试工厂函数创建有效后端。"""
        env = MockDirectRLEnv(num_envs=4)
        backend = create_isaaclab_backend(env)

        assert validate_backend(backend)
        assert backend.num_envs == 4

    def test_factory_returns_protocol_compatible_object(self):
        """测试工厂函数返回协议兼容对象。"""
        env = MockDirectRLEnv(num_envs=4)
        backend = create_isaaclab_backend(env)

        # 应可作为SimulatorBackend使用
        obs = backend.reset()
        assert "policy" in obs

        actions = torch.zeros(4, 12)
        obs, rewards, dones, info = backend.step(actions)
        assert rewards.shape == (4,)

        backend.close()


class TestAdapterEquivalence:
    """测试适配器与原始环境的等价性。"""

    def test_reset_equivalence(self):
        """测试reset结果与原始环境一致。"""
        env = MockDirectRLEnv(num_envs=4)
        adapter = IsaacLabAdapter(env)

        obs_direct, _ = env.reset()
        obs_adapter = adapter.reset()

        assert torch.equal(obs_direct["policy"], obs_adapter["policy"])
        assert torch.equal(obs_direct["critic"], obs_adapter["critic"])

    def test_step_equivalence(self):
        """测试step结果与原始环境一致。"""
        # 创建两个独立环境实例避免共享状态
        env1 = MockDirectRLEnv(num_envs=4)
        env2 = MockDirectRLEnv(num_envs=4)
        adapter = IsaacLabAdapter(env2)

        actions = torch.zeros(4, 12)

        # 直接调用环境
        obs_direct, rewards_direct, term_direct, trunc_direct, info_direct = env1.step(
            actions
        )
        dones_direct = term_direct | trunc_direct

        # 通过适配器调用
        obs_adapter, rewards_adapter, dones_adapter, info_adapter = adapter.step(
            actions
        )

        assert torch.equal(obs_direct["policy"], obs_adapter["policy"])
        assert torch.equal(rewards_direct, rewards_adapter)
        assert torch.equal(dones_direct, dones_adapter)

    def test_get_observations_equivalence(self):
        """测试get_observations与原始环境一致。"""
        env = MockDirectRLEnv(num_envs=4)
        adapter = IsaacLabAdapter(env)

        obs_direct = env._get_observations()
        obs_adapter = adapter.get_observations()

        assert torch.equal(obs_direct["policy"], obs_adapter["policy"])
        assert torch.equal(obs_direct["critic"], obs_adapter["critic"])
