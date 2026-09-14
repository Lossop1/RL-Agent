"""CheckpointSelector单元测试"""
from __future__ import annotations

import pytest

from autotuner.product.checkpoint_curator import CheckpointRegistry, CheckpointSelector


@pytest.fixture
def sample_registry():
    """创建示例检查点注册表"""
    registry = CheckpointRegistry()

    checkpoints = [
        ("agent_10000.pt", 10000, 0.5, 0.15, 80.0, 1, 1726234567.0),
        ("agent_20000.pt", 20000, 1.0, 0.10, 100.0, 1, 1726234580.0),
        ("agent_30000.pt", 30000, 1.5, 0.08, 120.0, 2, 1726234600.0),
        ("agent_40000.pt", 40000, 1.8, 0.05, 150.0, 2, 1726234620.0),
        ("agent_50000.pt", 50000, 2.0, 0.03, 180.0, 3, 1726234640.0),
        ("agent_60000.pt", 60000, 1.2, 0.25, 70.0, 3, 1726234660.0),  # 高terminal_rate
    ]

    for ref, step, reward, term_rate, ep_len, phase, mtime in checkpoints:
        perf = {
            "reward_mean": reward,
            "terminal_rate": term_rate,
            "episode_length_mean": ep_len,
            "curriculum_phase": phase,
            "checkpoint_mtime": mtime,
        }
        registry.register(ref, step, perf)

    return registry


def test_score_calculation():
    """测试评分计算逻辑"""
    selector = CheckpointSelector()

    performance = {
        "step": 50000,
        "reward_mean": 1.5,
        "terminal_rate": 0.1,
        "episode_length_mean": 120.0,
    }

    weights = {
        "reward": 0.5,
        "stability": 0.3,
        "episode": 0.1,
        "recency": 0.1,
    }

    score = selector.score(performance, weights, recency_base_step=60000)

    # 验证得分在合理范围内
    assert 0.0 <= score <= 1.0
    assert score > 0.5  # reward和stability都较高


def test_filter_candidates_by_reward(sample_registry):
    """测试基于reward阈值过滤"""
    selector = CheckpointSelector(
        min_reward_ratio=0.5,  # 50%最大值
        max_terminal_rate=0.2,
        min_training_step=10000,
    )

    criteria = {
        "reward_max": 2.0,  # 最大reward
        "exclude_refs": set(),
    }

    candidates = selector.filter_candidates(sample_registry, criteria)

    # reward_mean >= 1.0的检查点应该通过（除了terminal_rate过高的）
    assert "agent_10000.pt" not in candidates  # reward=0.5 < 1.0
    assert "agent_20000.pt" in candidates  # reward=1.0
    assert "agent_30000.pt" in candidates  # reward=1.5
    assert "agent_60000.pt" not in candidates  # terminal_rate=0.25 > 0.2


def test_filter_candidates_by_terminal_rate(sample_registry):
    """测试基于terminal_rate过滤"""
    selector = CheckpointSelector(
        min_reward_ratio=0.0,  # 不限制reward
        max_terminal_rate=0.1,  # 严格的稳定性要求
        min_training_step=10000,
    )

    criteria = {"reward_max": 2.0, "exclude_refs": set()}

    candidates = selector.filter_candidates(sample_registry, criteria)

    # 只有terminal_rate <= 0.1的应该通过
    assert "agent_10000.pt" not in candidates  # terminal_rate=0.15
    assert "agent_20000.pt" in candidates  # terminal_rate=0.10
    assert "agent_30000.pt" in candidates  # terminal_rate=0.08
    assert "agent_60000.pt" not in candidates  # terminal_rate=0.25


def test_filter_excludes_milestones(sample_registry):
    """测试过滤器排除里程碑检查点"""
    selector = CheckpointSelector()

    criteria = {
        "reward_max": 2.0,
        "exclude_refs": {"agent_50000.pt"},  # 排除最佳检查点
    }

    candidates = selector.filter_candidates(sample_registry, criteria)
    assert "agent_50000.pt" not in candidates


def test_rank_top_k(sample_registry):
    """测试top-K排序"""
    selector = CheckpointSelector()

    # 获取所有通过基本过滤的候选
    candidates = [
        (ref, perf)
        for ref, perf in sample_registry.all_checkpoints()
        if perf["terminal_rate"] <= 0.2 and perf["step"] >= 50000
    ]

    weights = {
        "reward": 0.5,
        "stability": 0.3,
        "episode": 0.1,
        "recency": 0.1,
    }

    top_3 = selector.rank_top_k(candidates, k=3, weights=weights)

    # 验证返回数量
    assert len(top_3) <= 3

    # agent_50000.pt应该排在前面（高reward、低terminal_rate、高episode_length）
    assert "agent_50000.pt" in top_3


def test_rank_top_k_empty_candidates():
    """测试空候选列表的top-K排序"""
    selector = CheckpointSelector()
    result = selector.rank_top_k([], k=5, weights={})
    assert result == []


def test_select_milestones_phase_transitions(sample_registry):
    """测试识别phase转换里程碑"""
    selector = CheckpointSelector()

    milestones = selector.select_milestones(sample_registry)

    # 应该包含每个phase的首个和末个
    # phase 1: agent_10000 (首), agent_20000 (末)
    # phase 2: agent_30000 (首), agent_40000 (末)
    # phase 3: agent_50000 (首), agent_60000 (末)
    assert "agent_10000.pt" in milestones  # phase 1首个
    assert "agent_20000.pt" in milestones  # phase 1末个
    assert "agent_30000.pt" in milestones  # phase 2首个


def test_select_milestones_best_reward(sample_registry):
    """测试识别历史最佳reward里程碑"""
    selector = CheckpointSelector()

    milestones = selector.select_milestones(sample_registry)

    # agent_50000.pt有最高reward=2.0
    assert "agent_50000.pt" in milestones


def test_select_milestones_empty_registry():
    """测试空注册表的里程碑选择"""
    registry = CheckpointRegistry()
    selector = CheckpointSelector()

    milestones = selector.select_milestones(registry)
    assert milestones == []


def test_filter_candidates_min_training_step(sample_registry):
    """测试最小训练步数过滤"""
    selector = CheckpointSelector(
        min_reward_ratio=0.0,
        max_terminal_rate=1.0,
        min_training_step=40000,  # 只保留40000步以上
    )

    criteria = {"reward_max": 2.0, "exclude_refs": set()}

    candidates = selector.filter_candidates(sample_registry, criteria)

    # 只有step >= 40000的应该通过
    assert "agent_10000.pt" not in candidates
    assert "agent_20000.pt" not in candidates
    assert "agent_30000.pt" not in candidates
    assert "agent_40000.pt" in candidates
    assert "agent_50000.pt" in candidates


def test_score_recency_decay():
    """测试新鲜度衰减计算"""
    selector = CheckpointSelector()

    # 旧检查点
    old_perf = {
        "step": 10000,
        "reward_mean": 1.0,
        "terminal_rate": 0.1,
        "episode_length_mean": 100.0,
    }

    # 新检查点
    new_perf = {
        "step": 50000,
        "reward_mean": 1.0,
        "terminal_rate": 0.1,
        "episode_length_mean": 100.0,
    }

    weights = {
        "reward": 0.0,
        "stability": 0.0,
        "episode": 0.0,
        "recency": 1.0,  # 只考虑新鲜度
    }

    old_score = selector.score(old_perf, weights, recency_base_step=50000)
    new_score = selector.score(new_perf, weights, recency_base_step=50000)

    # 新检查点应该得分更高
    assert new_score > old_score


def test_rank_top_k_considers_all_weights(sample_registry):
    """测试排序考虑所有权重维度"""
    selector = CheckpointSelector()

    candidates = sample_registry.all_checkpoints()

    # 只看reward
    reward_weights = {"reward": 1.0, "stability": 0.0, "episode": 0.0, "recency": 0.0}
    reward_top = selector.rank_top_k(candidates, k=1, weights=reward_weights)
    assert reward_top[0] == "agent_50000.pt"  # 最高reward=2.0

    # 只看stability
    stability_weights = {"reward": 0.0, "stability": 1.0, "episode": 0.0, "recency": 0.0}
    stability_top = selector.rank_top_k(candidates, k=1, weights=stability_weights)
    assert stability_top[0] == "agent_50000.pt"  # 最低terminal_rate=0.03
