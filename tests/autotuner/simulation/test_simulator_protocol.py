"""测试仿真器后端协议定义。

验证SimulatorBackend协议的完整性和validate_backend()函数的正确性。
"""
import torch

from autotuner.simulation.simulator_protocol import SimulatorBackend, validate_backend


class MockSimulator:
    """模拟仿真器，完整实现SimulatorBackend协议。"""

    def __init__(self, num_envs: int = 4, device: str = "cpu"):
        self._num_envs = num_envs
        self._device = torch.device(device)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> torch.device:
        return self._device

    def reset(self, env_ids=None):
        n = len(env_ids) if env_ids is not None else self._num_envs
        return {"policy": torch.zeros(n, 10, device=self._device)}

    def step(self, actions):
        obs = {"policy": torch.zeros(self._num_envs, 10, device=self._device)}
        rewards = torch.zeros(self._num_envs, device=self._device)
        dones = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        info = {"step_count": 0}
        return obs, rewards, dones, info

    def close(self):
        pass

    def get_observations(self):
        return {"policy": torch.zeros(self._num_envs, 10, device=self._device)}

    def render(self):
        return None


class IncompleteSimulator:
    """不完整的仿真器，缺少部分方法。"""

    @property
    def num_envs(self) -> int:
        return 4

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def reset(self, env_ids=None):
        return {"policy": torch.zeros(4, 10)}

    # 缺少step/close/get_observations/render


class TestSimulatorProtocol:
    """测试SimulatorBackend协议定义。"""

    def test_mock_simulator_implements_protocol(self):
        """测试模拟仿真器实现协议。"""
        sim = MockSimulator(num_envs=4, device="cpu")
        assert validate_backend(sim)

    def test_incomplete_simulator_fails_validation(self):
        """测试不完整仿真器验证失败。"""
        sim = IncompleteSimulator()
        assert not validate_backend(sim)

    def test_mock_simulator_reset(self):
        """测试reset方法返回正确格式。"""
        sim = MockSimulator(num_envs=4)

        # 重置所有环境
        obs = sim.reset()
        assert "policy" in obs
        assert obs["policy"].shape == (4, 10)

        # 重置部分环境
        obs = sim.reset(env_ids=[0, 2])
        assert obs["policy"].shape == (2, 10)

    def test_mock_simulator_step(self):
        """测试step方法返回正确格式。"""
        sim = MockSimulator(num_envs=4)
        actions = torch.zeros(4, 12)

        obs, rewards, dones, info = sim.step(actions)

        assert "policy" in obs
        assert obs["policy"].shape == (4, 10)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        assert dones.dtype == torch.bool
        assert isinstance(info, dict)

    def test_mock_simulator_get_observations(self):
        """测试get_observations方法。"""
        sim = MockSimulator(num_envs=4)
        obs = sim.get_observations()
        assert "policy" in obs
        assert obs["policy"].shape == (4, 10)

    def test_mock_simulator_render(self):
        """测试render方法。"""
        sim = MockSimulator(num_envs=4)
        img = sim.render()
        # MockSimulator不支持渲染
        assert img is None

    def test_mock_simulator_properties(self):
        """测试属性访问。"""
        sim = MockSimulator(num_envs=8, device="cpu")
        assert sim.num_envs == 8
        assert sim.device == torch.device("cpu")

    def test_mock_simulator_close(self):
        """测试close方法。"""
        sim = MockSimulator(num_envs=4)
        sim.close()  # 不应抛出异常


class TestValidateBackend:
    """测试validate_backend()验证函数。"""

    def test_validate_complete_backend(self):
        """测试完整后端验证通过。"""
        sim = MockSimulator()
        assert validate_backend(sim)

    def test_validate_incomplete_backend(self):
        """测试不完整后端验证失败。"""
        sim = IncompleteSimulator()
        assert not validate_backend(sim)

    def test_validate_missing_properties(self):
        """测试缺少属性的后端验证失败。"""

        class NoProperties:
            def reset(self, env_ids=None):
                return {}

            def step(self, actions):
                return {}, torch.zeros(4), torch.zeros(4, dtype=torch.bool), {}

            def close(self):
                pass

            def get_observations(self):
                return {}

            def render(self):
                return None

        sim = NoProperties()
        assert not validate_backend(sim)

    def test_validate_non_callable_methods(self):
        """测试方法不可调用的后端验证失败。"""

        class NonCallable:
            num_envs = 4
            device = torch.device("cpu")
            reset = "not_a_function"  # 不是可调用对象
            step = None
            close = None
            get_observations = None
            render = None

        sim = NonCallable()
        assert not validate_backend(sim)

    def test_validate_none_object(self):
        """测试None对象验证失败。"""
        assert not validate_backend(None)

    def test_validate_arbitrary_object(self):
        """测试任意对象验证失败。"""
        assert not validate_backend(42)
        assert not validate_backend("string")
        assert not validate_backend([1, 2, 3])


class TestProtocolTypeHints:
    """测试协议类型提示的正确性。"""

    def test_protocol_as_type_hint(self):
        """测试协议可作为类型提示使用。"""

        def use_simulator(sim: SimulatorBackend) -> int:
            """接受SimulatorBackend协议的函数。"""
            return sim.num_envs

        sim = MockSimulator(num_envs=16)
        result = use_simulator(sim)
        assert result == 16

    def test_protocol_does_not_require_inheritance(self):
        """测试协议不需要显式继承。

        MockSimulator未继承SimulatorBackend，但实现了所有方法，
        因此在类型检查时应被认为是SimulatorBackend的子类型。
        """
        sim = MockSimulator()
        # 运行时验证
        assert validate_backend(sim)
        # 静态类型检查器应接受此赋值（需要mypy/pyright验证）
        backend: SimulatorBackend = sim  # noqa: F841
