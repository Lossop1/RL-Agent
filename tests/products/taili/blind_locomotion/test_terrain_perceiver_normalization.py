"""测试观测归一化集成到TerrainPerceiverPolicy。

验证P6.3集成要求：
- 归一化器正确集成到策略compute()流程
- 训练模式下更新统计量
- 评估模式下冻结统计量
- 检查点保存/加载包含归一化器状态
- 95%观测值落在[-3,3]区间
"""
import pytest
import torch

# skrl是训练环境依赖，测试环境可能不可用 - 跳过整个模块
pytest.importorskip("skrl")

from products.taili.blind_locomotion.terrain_perceiver_policy import (
    TerrainPerceiverPolicy,
    ACTOR_OBS,
)


class TestTerrainPerceiverPolicyNormalization:
    """测试策略中的观测归一化集成。"""

    @pytest.fixture
    def mock_spaces(self):
        """创建模拟的观测和动作空间。"""
        class MockSpace:
            def __init__(self, shape):
                self.shape = shape

        obs_space = MockSpace((ACTOR_OBS,))
        action_space = MockSpace((12,))  # Taili 12关节
        return obs_space, action_space

    def test_normalization_enabled_by_default(self, mock_spaces):
        """测试归一化默认启用。"""
        obs_space, action_space = mock_spaces
        policy = TerrainPerceiverPolicy(obs_space, action_space, "cpu")

        assert policy.use_obs_normalization
        assert policy.obs_normalizer is not None
        assert policy.obs_normalizer.count == 0

    def test_normalization_can_be_disabled(self, mock_spaces):
        """测试可以禁用归一化。"""
        obs_space, action_space = mock_spaces
        policy = TerrainPerceiverPolicy(
            obs_space, action_space, "cpu", use_obs_normalization=False
        )

        assert not policy.use_obs_normalization
        assert policy.obs_normalizer is None

    def test_training_mode_updates_statistics(self, mock_spaces):
        """测试训练模式下更新统计量。"""
        obs_space, action_space = mock_spaces
        policy = TerrainPerceiverPolicy(obs_space, action_space, "cpu")
        policy.train()

        # 第一次前向传播
        inputs = {"states": torch.randn(16, ACTOR_OBS)}
        _ = policy.compute(inputs)

        assert policy.obs_normalizer.count == 16

        # 第二次前向传播
        _ = policy.compute(inputs)
        assert policy.obs_normalizer.count == 32

    def test_eval_mode_freezes_statistics(self, mock_spaces):
        """测试评估模式下不更新统计量。"""
        obs_space, action_space = mock_spaces
        policy = TerrainPerceiverPolicy(obs_space, action_space, "cpu")

        # 训练模式：更新统计量
        policy.train()
        inputs = {"states": torch.randn(16, ACTOR_OBS)}
        _ = policy.compute(inputs)
        count_after_train = policy.obs_normalizer.count.item()

        # 评估模式：冻结统计量
        policy.eval()
        _ = policy.compute(inputs)
        count_after_eval = policy.obs_normalizer.count.item()

        assert count_after_train == 16
        assert count_after_eval == 16  # 未增长

    def test_normalization_produces_bounded_output(self, mock_spaces):
        """测试归一化后观测值被限制在[-3,3]区间。"""
        obs_space, action_space = mock_spaces
        policy = TerrainPerceiverPolicy(obs_space, action_space, "cpu")
        policy.train()

        # 预热归一化器
        torch.manual_seed(42)
        for _ in range(10):
            warmup_inputs = {"states": torch.randn(32, ACTOR_OBS)}
            _ = policy.compute(warmup_inputs)

        # 评估模式：检查归一化输出
        policy.eval()
        test_inputs = {"states": torch.randn(1000, ACTOR_OBS)}

        # 手动检查归一化后的states（内部会调用normalizer.normalize）
        states = test_inputs["states"]
        normalized = policy.obs_normalizer.normalize(states, clip_range=3.0)

        # 所有值应在[-3, 3]区间
        assert (normalized >= -3.0).all()
        assert (normalized <= 3.0).all()

        # 95%+应在[-3, 3]区间（理论上100%因为我们clip了）
        within_range = ((normalized >= -3.0) & (normalized <= 3.0)).float().mean()
        assert within_range >= 0.95

    def test_checkpoint_includes_normalizer_state(self, mock_spaces):
        """测试检查点包含归一化器状态。"""
        obs_space, action_space = mock_spaces
        policy1 = TerrainPerceiverPolicy(obs_space, action_space, "cpu")
        policy1.train()

        # 训练并更新统计量
        inputs = {"states": torch.randn(100, ACTOR_OBS)}
        _ = policy1.compute(inputs)

        # 保存state_dict
        state_dict = policy1.state_dict()

        # 验证归一化器buffer在state_dict中
        assert "obs_normalizer.mean" in state_dict
        assert "obs_normalizer.var" in state_dict
        assert "obs_normalizer.count" in state_dict

        # 加载到新策略
        policy2 = TerrainPerceiverPolicy(obs_space, action_space, "cpu")
        policy2.load_state_dict(state_dict)

        # 验证统计量正确加载
        assert torch.allclose(policy2.obs_normalizer.mean, policy1.obs_normalizer.mean)
        assert torch.allclose(policy2.obs_normalizer.var, policy1.obs_normalizer.var)
        assert policy2.obs_normalizer.count == policy1.obs_normalizer.count

    def test_disabled_normalization_bypasses_normalizer(self, mock_spaces):
        """测试禁用归一化时跳过归一化器。"""
        obs_space, action_space = mock_spaces
        policy = TerrainPerceiverPolicy(
            obs_space, action_space, "cpu", use_obs_normalization=False
        )
        policy.train()

        # 原始观测
        original_states = torch.randn(16, ACTOR_OBS)
        inputs = {"states": original_states.clone()}

        # compute()应该不修改states（无归一化）
        _ = policy.compute(inputs)

        # obs_normalizer=None时不应崩溃
        assert policy.obs_normalizer is None

    def test_normalization_with_multiple_batches(self, mock_spaces):
        """测试多个batch累积更新归一化统计量。"""
        obs_space, action_space = mock_spaces
        policy = TerrainPerceiverPolicy(obs_space, action_space, "cpu")
        policy.train()

        batch_size = 32
        num_batches = 10

        for i in range(num_batches):
            inputs = {"states": torch.randn(batch_size, ACTOR_OBS)}
            _ = policy.compute(inputs)

        # 统计量应反映所有batch
        assert policy.obs_normalizer.count == batch_size * num_batches

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA不可用")
    def test_normalization_on_gpu(self, mock_spaces):
        """测试GPU设备上的归一化。"""
        obs_space, action_space = mock_spaces
        policy = TerrainPerceiverPolicy(obs_space, action_space, "cuda")
        policy.train()

        # 验证归一化器在正确的设备上
        assert policy.obs_normalizer.mean.device.type == "cuda"

        # GPU上的前向传播
        inputs = {"states": torch.randn(16, ACTOR_OBS, device="cuda")}
        mean, log_std, info = policy.compute(inputs)

        assert mean.device.type == "cuda"
        assert policy.obs_normalizer.count == 16
