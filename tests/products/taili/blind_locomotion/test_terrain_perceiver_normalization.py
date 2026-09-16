"""测试观测归一化集成到TerrainPerceiverPolicy。

验证P6.3集成要求：
- 归一化器正确集成到策略compute()流程
- 训练模式下更新统计量
- 评估模式下冻结统计量
- 检查点保存/加载包含归一化器状态
- 95%观测值落在[-3,3]区间
"""
import gymnasium
import pytest
import torch

# skrl是训练环境依赖，测试环境可能不可用 - 跳过整个模块
pytest.importorskip("skrl")


def _box(size: int) -> gymnasium.spaces.Box:
    """一个无限范围的观测/动作空间。

    2026-09-16 之前本文件用的是只有 ``.shape`` 的 MockSpace，那样在**真的** skrl 下也跑不起来：
    skrl 的 ``Model.__init__`` 会走 ``compute_space_size()``，最终读到 ``is_np_flattenable``。
    也就是说这些用例在两种环境下都从未真正执行过（本机没装 skrl，整模块被上面那行跳过）。
    """
    return gymnasium.spaces.Box(
        low=-float("inf"), high=float("inf"), shape=(size,), dtype="float32"
    )

from products.taili.blind_locomotion.terrain_perceiver_policy import (
    TerrainPerceiverPolicy,
    ACTOR_OBS,
)

# 环境真正交给策略的 policy 张量宽度：可部署盲态切片 + 训练期特权量 + 辅助标签。
# 出处：taili_blind_config.yaml 的 deployable.actor_raw_dim=1407、
# training_only.privileged_dim=197、aux_label_dim=34；blind_tp_env_cfg.py:25 把它算作
# 57 + 25*54 + (3 + 17*11 + 3 + 4) + 34 = 1638，blind_tp_env.py:578 按这个宽度拼张量。
POLICY_TENSOR_DIM = ACTOR_OBS + 197 + 34


class TestTerrainPerceiverPolicyNormalization:
    """测试策略中的观测归一化集成。"""

    @pytest.fixture
    def mock_spaces(self):
        """可部署盲态切片的观测空间与 12 关节动作空间。"""
        return _box(ACTOR_OBS), _box(12)  # Taili 12关节

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

    def test_accepts_the_full_policy_tensor(self):
        """策略收到的是完整的 1638 维张量，不是 1407，归一化不许被这个宽度绊倒。

        生产里模型是用 env 的 observation_space 构造的（1638），skrl 的 init_state_dict
        与训练时的 "states" 也都是这个宽度。本模块此前所有用例都用 MockSpace((ACTOR_OBS,))
        造空间、喂 (N, 1407)，于是"归一化器按 1407 建、却收到 1638"在单测里永远碰不到——
        一直到 3060 机器上第一次跑真实训练才炸在
        terrain_perceiver_policy.py:85（RuntimeError: tensor a (1638) vs b (1407)）。
        """
        policy = TerrainPerceiverPolicy(_box(POLICY_TENSOR_DIM), _box(12), "cpu")
        policy.train()

        inputs = {"states": torch.randn(16, POLICY_TENSOR_DIM)}
        mean, log_std, _info = policy.compute(inputs)

        assert policy.obs_normalizer.count == 16
        assert mean.shape == (16, 12)

        # 归一化只统计盲态切片那 1407 列，多出来的 231 列不参与
        assert policy.obs_normalizer.mean.shape == (ACTOR_OBS,)

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
