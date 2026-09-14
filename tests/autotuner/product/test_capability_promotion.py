"""CapabilityPromotionService单元测试"""
from __future__ import annotations

import time

import pytest

from autotuner.product.checkpoint_curator import (
    CapabilityPromotionService,
    CheckpointRegistry,
    CheckpointSelector,
)


def test_extract_capabilities_high_performance():
    """测试从高性能检查点提取能力特征"""
    service = CapabilityPromotionService()

    performance = {
        "reward_mean": 2.0,
        "terminal_rate": 0.05,
        "episode_length_mean": 150.0,
        "curriculum_phase": 3,
    }
    config = {"command_modes": ["forward", "turn", "strafe"]}

    capabilities = service.extract_capabilities("agent_50000.pt", performance, config)

    # 验证能力提取
    assert capabilities["speed_range"] == "medium_to_high"
    assert "phase_3" in capabilities["terrain_levels"]
    assert "obstacles" in capabilities["terrain_levels"]
    assert "forward" in capabilities["command_modes"]
    assert capabilities["quality_gates"]["stability"] == "pass"
    assert capabilities["quality_gates"]["endurance"] == "pass"


def test_extract_capabilities_low_performance():
    """测试从低性能检查点提取能力特征"""
    service = CapabilityPromotionService()

    performance = {
        "reward_mean": 0.3,
        "terminal_rate": 0.3,
        "episode_length_mean": 50.0,
        "curriculum_phase": 1,
    }
    config = {}

    capabilities = service.extract_capabilities("agent_10000.pt", performance, config)

    # 验证能力提取
    assert capabilities["speed_range"] == "low"
    assert capabilities["terrain_levels"] == ["phase_1"]
    assert capabilities["quality_gates"]["stability"] == "fail"
    assert capabilities["quality_gates"]["endurance"] == "fail"


def test_create_profile():
    """测试创建CapabilityProfile记录"""
    service = CapabilityPromotionService()

    capabilities = {
        "speed_range": "medium_to_high",
        "terrain_levels": ["phase_2", "uneven_terrain"],
        "command_modes": ["forward", "turn"],
        "quality_gates": {"stability": "pass"},
    }

    profile = service.create_profile("agent_30000.pt", capabilities, status="candidate")

    # 验证profile结构
    assert profile.checkpoint_ref == "agent_30000.pt"
    assert profile.capabilities == capabilities
    assert profile.status == "candidate"


def test_promote_top_performers_no_ledger():
    """测试批量提升top-K检查点（无ledger）"""
    registry = CheckpointRegistry()
    current_time = time.time()

    # 注册多个检查点
    checkpoints = [
        ("agent_10000.pt", 10000, 0.5, 0.15, 80.0, 1),
        ("agent_20000.pt", 20000, 1.0, 0.10, 100.0, 1),
        ("agent_30000.pt", 30000, 1.5, 0.08, 120.0, 2),
        ("agent_40000.pt", 40000, 1.8, 0.05, 150.0, 2),
        ("agent_50000.pt", 50000, 2.0, 0.03, 180.0, 3),
    ]

    for ref, step, reward, term_rate, ep_len, phase in checkpoints:
        perf = {
            "reward_mean": reward,
            "terminal_rate": term_rate,
            "episode_length_mean": ep_len,
            "curriculum_phase": phase,
            "checkpoint_mtime": current_time,
        }
        registry.register(ref, step, perf)

    service = CapabilityPromotionService()
    selector = CheckpointSelector()

    # 提升top-3
    promoted_ids = service.promote_top_performers(
        registry,
        selector,
        ledger_store=None,
        top_k=3,
        config={"command_modes": ["forward"]},
    )

    # 验证提升结果
    assert len(promoted_ids) == 3
    # 高性能检查点应该被提升
    assert "agent_50000.pt" in promoted_ids
    assert "agent_40000.pt" in promoted_ids


def test_promote_top_performers_empty_registry():
    """测试空注册表的提升操作"""
    registry = CheckpointRegistry()
    service = CapabilityPromotionService()
    selector = CheckpointSelector()

    promoted_ids = service.promote_top_performers(
        registry,
        selector,
        ledger_store=None,
        top_k=5,
    )

    assert promoted_ids == []


def test_extract_capabilities_medium_performance():
    """测试从中等性能检查点提取能力特征"""
    service = CapabilityPromotionService()

    performance = {
        "reward_mean": 1.0,
        "terminal_rate": 0.12,
        "episode_length_mean": 90.0,
        "curriculum_phase": 2,
    }
    config = {"command_modes": ["forward", "turn"]}

    capabilities = service.extract_capabilities("agent_20000.pt", performance, config)

    assert capabilities["speed_range"] == "low_to_medium"
    assert "phase_2" in capabilities["terrain_levels"]
    assert "uneven_terrain" in capabilities["terrain_levels"]
    assert capabilities["quality_gates"]["stability"] == "pass"
    assert capabilities["quality_gates"]["endurance"] == "pass"
    assert capabilities["quality_gates"]["reward_threshold"] == "pass"


def test_create_profile_default_status():
    """测试创建profile时的默认状态"""
    service = CapabilityPromotionService()

    capabilities = {"speed_range": "low"}

    profile = service.create_profile("agent_10000.pt", capabilities)

    # 默认状态应为candidate
    assert profile.status == "candidate"


def test_create_profile_validated_status():
    """测试创建validated状态的profile"""
    service = CapabilityPromotionService()

    capabilities = {"speed_range": "high"}

    profile = service.create_profile("agent_60000.pt", capabilities, status="validated")

    assert profile.status == "validated"
