"""CheckpointRegistry单元测试"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from autotuner.product.checkpoint_curator import CheckpointRegistry, PerformanceSnapshot


def test_register_and_get_performance():
    """测试注册和查询检查点性能"""
    registry = CheckpointRegistry()

    perf = {
        "reward_mean": 1.5,
        "terminal_rate": 0.1,
        "episode_length_mean": 100.0,
        "curriculum_phase": 2,
        "curriculum_level": 3,
        "checkpoint_mtime": 1726234567.0,
    }

    registry.register("checkpoints/agent_50000.pt", 50000, perf)

    retrieved = registry.get_performance("checkpoints/agent_50000.pt")
    assert retrieved is not None
    assert retrieved["step"] == 50000
    assert retrieved["reward_mean"] == 1.5
    assert retrieved["terminal_rate"] == 0.1


def test_get_nonexistent_checkpoint():
    """测试查询不存在的检查点返回None"""
    registry = CheckpointRegistry()
    result = registry.get_performance("nonexistent.pt")
    assert result is None


def test_query_by_step_range():
    """测试按步数范围查询"""
    registry = CheckpointRegistry()

    # 注册3个检查点
    for step in [10000, 20000, 30000]:
        perf = {
            "reward_mean": step / 10000.0,
            "terminal_rate": 0.1,
            "episode_length_mean": 100.0,
            "curriculum_phase": 1,
            "checkpoint_mtime": 1726234567.0,
        }
        registry.register(f"agent_{step}.pt", step, perf)

    # 查询15000-25000范围
    results = registry.query_by_step_range(15000, 25000)
    assert len(results) == 1
    assert results[0][0] == "agent_20000.pt"
    assert results[0][1]["step"] == 20000


def test_backfill_from_telemetry(tmp_path: Path):
    """测试从telemetry JSONL回填历史数据"""
    registry = CheckpointRegistry()

    # 创建telemetry JSONL文件
    telemetry_path = tmp_path / "telemetry.jsonl"
    telemetry_data = [
        {
            "step": 10000,
            "reward": {"mean": 0.5},
            "health": {"terminal_rate": 0.15, "episode_length_mean": 80.0},
            "curriculum": {"phase": 1, "level": 1},
        },
        {
            "step": 20000,
            "reward": {"mean": 1.2},
            "health": {"terminal_rate": 0.08, "episode_length_mean": 120.0},
            "curriculum": {"phase": 2, "level": 3},
        },
    ]
    telemetry_path.write_text(
        "\n".join(json.dumps(entry) for entry in telemetry_data),
        encoding="utf-8",
    )

    # 创建检查点清单
    checkpoint_inventory = [
        {"path": "agent_10000.pt", "step": 10000, "mtime": 1726234567.0},
        {"path": "agent_20000.pt", "step": 20000, "mtime": 1726234580.0},
        {"path": "agent_30000.pt", "step": 30000, "mtime": 1726234600.0},  # 无对应telemetry
    ]

    # 执行回填
    count = registry.backfill_from_telemetry(str(telemetry_path), checkpoint_inventory)

    # 验证回填结果
    assert count == 2  # 只有10000和20000有telemetry数据

    perf_10k = registry.get_performance("agent_10000.pt")
    assert perf_10k is not None
    assert perf_10k["reward_mean"] == 0.5
    assert perf_10k["terminal_rate"] == 0.15

    perf_20k = registry.get_performance("agent_20000.pt")
    assert perf_20k is not None
    assert perf_20k["reward_mean"] == 1.2
    assert perf_20k["curriculum_phase"] == 2


def test_backfill_infers_step_from_filename(tmp_path: Path):
    """测试从文件名推断step进行回填"""
    registry = CheckpointRegistry()

    telemetry_path = tmp_path / "telemetry.jsonl"
    telemetry_path.write_text(
        json.dumps({
            "step": 15000,
            "reward": {"mean": 0.8},
            "health": {"terminal_rate": 0.12, "episode_length_mean": 90.0},
            "curriculum": {"phase": 1, "level": 2},
        }),
        encoding="utf-8",
    )

    # 检查点清单未提供step，依赖文件名推断
    checkpoint_inventory = [
        {"path": "checkpoints/agent_15000.pt", "mtime": 1726234570.0},
    ]

    count = registry.backfill_from_telemetry(str(telemetry_path), checkpoint_inventory)
    assert count == 1

    perf = registry.get_performance("checkpoints/agent_15000.pt")
    assert perf is not None
    assert perf["step"] == 15000
    assert perf["reward_mean"] == 0.8


def test_export_to_manifest(tmp_path: Path):
    """测试导出检查点清单到JSON文件"""
    registry = CheckpointRegistry()

    perf1 = {
        "reward_mean": 1.0,
        "terminal_rate": 0.1,
        "episode_length_mean": 100.0,
        "curriculum_phase": 1,
        "checkpoint_mtime": 1726234567.0,
    }
    perf2 = {
        "reward_mean": 1.5,
        "terminal_rate": 0.05,
        "episode_length_mean": 150.0,
        "curriculum_phase": 2,
        "checkpoint_mtime": 1726234580.0,
    }

    registry.register("agent_10000.pt", 10000, perf1)
    registry.register("agent_20000.pt", 20000, perf2)

    # 导出
    output_path = tmp_path / "manifest.json"
    registry.export_to_manifest(str(output_path))

    # 验证文件内容
    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert manifest["version"] == "checkpoint_registry/v1"
    assert manifest["count"] == 2
    assert "agent_10000.pt" in manifest["checkpoints"]
    assert manifest["checkpoints"]["agent_10000.pt"]["reward_mean"] == 1.0


def test_all_checkpoints():
    """测试获取所有检查点"""
    registry = CheckpointRegistry()

    for i, step in enumerate([10000, 20000, 30000]):
        perf = {
            "reward_mean": float(i + 1),
            "terminal_rate": 0.1,
            "episode_length_mean": 100.0,
            "curriculum_phase": 1,
            "checkpoint_mtime": 1726234567.0,
        }
        registry.register(f"agent_{step}.pt", step, perf)

    all_ckpts = registry.all_checkpoints()
    assert len(all_ckpts) == 3

    # 验证返回格式
    assert all(isinstance(item, tuple) and len(item) == 2 for item in all_ckpts)
    refs = [ref for ref, _ in all_ckpts]
    assert "agent_10000.pt" in refs
    assert "agent_20000.pt" in refs
    assert "agent_30000.pt" in refs


def test_performance_snapshot_serialization():
    """测试PerformanceSnapshot序列化和反序列化"""
    snapshot = PerformanceSnapshot(
        step=10000,
        reward_mean=1.5,
        terminal_rate=0.1,
        episode_length_mean=100.0,
        curriculum_phase=2,
        curriculum_level=3,
        checkpoint_mtime=1726234567.0,
        raw_metrics={"tracking_lin": 0.8, "gait_match": 0.9},
    )

    # 序列化
    data = snapshot.to_dict()
    assert data["step"] == 10000
    assert data["raw_metrics"]["tracking_lin"] == 0.8

    # 反序列化
    restored = PerformanceSnapshot.from_dict(data)
    assert restored.step == 10000
    assert restored.reward_mean == 1.5
    assert restored.raw_metrics["gait_match"] == 0.9


def test_backfill_handles_missing_telemetry(tmp_path: Path):
    """测试回填处理telemetry文件不存在的情况"""
    registry = CheckpointRegistry()

    checkpoint_inventory = [
        {"path": "agent_10000.pt", "step": 10000, "mtime": 1726234567.0},
    ]

    # telemetry文件不存在
    count = registry.backfill_from_telemetry(
        str(tmp_path / "nonexistent.jsonl"),
        checkpoint_inventory,
    )

    assert count == 0
    assert registry.get_performance("agent_10000.pt") is None


def test_backfill_handles_malformed_telemetry(tmp_path: Path):
    """测试回填处理格式错误的telemetry文件"""
    registry = CheckpointRegistry()

    telemetry_path = tmp_path / "malformed.jsonl"
    telemetry_path.write_text("not valid json", encoding="utf-8")

    checkpoint_inventory = [
        {"path": "agent_10000.pt", "step": 10000, "mtime": 1726234567.0},
    ]

    count = registry.backfill_from_telemetry(str(telemetry_path), checkpoint_inventory)
    assert count == 0
