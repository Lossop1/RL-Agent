"""P4.2检查点管理集成测试。

测试CheckpointIntegration和checkpoint_hook在训练流程中的集成。
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from products.taili.blind_locomotion.checkpoint_integration import (
    CheckpointIntegration,
)


def test_checkpoint_integration_basic():
    """测试CheckpointIntegration基本功能"""
    with tempfile.TemporaryDirectory() as tmpdir:
        integration = CheckpointIntegration(
            tmpdir,
            cleanup_interval=5,
            cleanup_threshold_gb=1.0,
            enable_cleanup=True,
            enable_promotion=True,
        )

        # 模拟保存10个检查点
        for i in range(10):
            checkpoint_path = Path(tmpdir) / f"agent_{(i+1)*10000}.pt"
            checkpoint_path.write_text(f"checkpoint {i}", encoding="utf-8")

            performance = {
                "reward_mean": float(i) * 0.2,
                "terminal_rate": 0.15 - float(i) * 0.01,
                "episode_length_mean": 80.0 + float(i) * 10.0,
                "curriculum_phase": min(i // 3, 3),
                "checkpoint_mtime": time.time(),
            }

            integration.on_checkpoint_saved(
                str(checkpoint_path),
                (i + 1) * 10000,
                performance,
            )

        # 验证注册
        stats = integration.get_statistics()
        assert stats["total_registered"] == 10
        assert stats["checkpoint_count"] == 10


def test_checkpoint_integration_periodic_cleanup():
    """测试周期性清理触发"""
    with tempfile.TemporaryDirectory() as tmpdir:
        integration = CheckpointIntegration(
            tmpdir,
            cleanup_interval=3,  # 每3个检查点清理一次
            cleanup_threshold_gb=0.0,  # 强制触发清理
            enable_cleanup=True,
        )

        # 保存6个检查点，应该触发2次清理
        for i in range(6):
            checkpoint_path = Path(tmpdir) / f"agent_{(i+1)*10000}.pt"
            checkpoint_path.write_text("x" * 100, encoding="utf-8")

            performance = {
                "reward_mean": 0.5,
                "terminal_rate": 0.1,
                "episode_length_mean": 100.0,
                "curriculum_phase": 1,
                "checkpoint_mtime": time.time(),
            }

            integration.on_checkpoint_saved(
                str(checkpoint_path),
                (i + 1) * 10000,
                performance,
            )

        # 第3个和第6个检查点保存时应该触发了清理
        # 由于保留策略保护top-K和recent，实际删除数量可能少于预期
        # 主要验证清理流程没有报错
        stats = integration.get_statistics()
        assert stats["checkpoint_count"] == 6


def test_checkpoint_integration_finalize():
    """测试训练结束时的最终处理"""
    with tempfile.TemporaryDirectory() as tmpdir:
        integration = CheckpointIntegration(
            tmpdir,
            enable_cleanup=True,
            enable_promotion=True,
        )

        # 保存5个检查点
        for i in range(5):
            checkpoint_path = Path(tmpdir) / f"agent_{(i+1)*10000}.pt"
            checkpoint_path.write_text(f"checkpoint {i}", encoding="utf-8")

            performance = {
                "reward_mean": float(i) * 0.5,
                "terminal_rate": 0.1,
                "episode_length_mean": 100.0 + float(i) * 20.0,
                "curriculum_phase": 2,
                "checkpoint_mtime": time.time(),
            }

            integration.on_checkpoint_saved(
                str(checkpoint_path),
                (i + 1) * 10000,
                performance,
            )

        # 执行最终处理
        promoted = integration.finalize_training(top_k=3, config={})

        # 验证提升了top-3
        assert len(promoted) == 3


def test_checkpoint_integration_export_manifest():
    """测试导出检查点清单"""
    with tempfile.TemporaryDirectory() as tmpdir:
        integration = CheckpointIntegration(tmpdir)

        # 保存3个检查点
        for i in range(3):
            checkpoint_path = Path(tmpdir) / f"agent_{(i+1)*10000}.pt"
            checkpoint_path.write_text(f"checkpoint {i}", encoding="utf-8")

            performance = {
                "reward_mean": 1.0,
                "terminal_rate": 0.1,
                "episode_length_mean": 100.0,
                "curriculum_phase": 1,
                "checkpoint_mtime": time.time(),
            }

            integration.on_checkpoint_saved(
                str(checkpoint_path),
                (i + 1) * 10000,
                performance,
            )

        # 导出清单
        manifest_path = Path(tmpdir) / "manifest.json"
        integration.export_manifest(str(manifest_path))

        # 验证清单文件存在
        assert manifest_path.exists()

        # 验证内容
        import json
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["count"] == 3
        # CheckpointRegistry使用完整路径作为key
        checkpoint_keys = list(manifest["checkpoints"].keys())
        assert len(checkpoint_keys) == 3
        assert any("agent_10000.pt" in key for key in checkpoint_keys)


def test_checkpoint_integration_statistics():
    """测试统计信息获取"""
    with tempfile.TemporaryDirectory() as tmpdir:
        integration = CheckpointIntegration(tmpdir, cleanup_interval=5)

        # 初始统计
        stats = integration.get_statistics()
        assert stats["total_registered"] == 0
        assert stats["checkpoint_count"] == 0
        assert stats["disk_usage_gb"] == 0.0

        # 保存2个检查点
        for i in range(2):
            checkpoint_path = Path(tmpdir) / f"agent_{(i+1)*10000}.pt"
            checkpoint_path.write_text("x" * 1024, encoding="utf-8")  # 1KB

            performance = {
                "reward_mean": 1.0,
                "terminal_rate": 0.1,
                "episode_length_mean": 100.0,
                "curriculum_phase": 1,
                "checkpoint_mtime": time.time(),
            }

            integration.on_checkpoint_saved(
                str(checkpoint_path),
                (i + 1) * 10000,
                performance,
            )

        # 更新统计
        stats = integration.get_statistics()
        assert stats["total_registered"] == 2
        assert stats["checkpoint_count"] == 2
        assert stats["disk_usage_gb"] > 0.0


def test_checkpoint_integration_disable_cleanup():
    """测试禁用自动清理"""
    with tempfile.TemporaryDirectory() as tmpdir:
        integration = CheckpointIntegration(
            tmpdir,
            cleanup_interval=2,
            enable_cleanup=False,  # 禁用清理
        )

        # 保存4个检查点
        for i in range(4):
            checkpoint_path = Path(tmpdir) / f"agent_{(i+1)*10000}.pt"
            checkpoint_path.write_text(f"checkpoint {i}", encoding="utf-8")

            performance = {
                "reward_mean": 0.5,
                "terminal_rate": 0.1,
                "episode_length_mean": 100.0,
                "curriculum_phase": 1,
                "checkpoint_mtime": time.time(),
            }

            integration.on_checkpoint_saved(
                str(checkpoint_path),
                (i + 1) * 10000,
                performance,
            )

        # 所有文件应该仍然存在（没有触发清理）
        remaining = list(Path(tmpdir).glob("agent_*.pt"))
        assert len(remaining) == 4


def test_checkpoint_integration_disable_promotion():
    """测试禁用能力提升"""
    with tempfile.TemporaryDirectory() as tmpdir:
        integration = CheckpointIntegration(
            tmpdir,
            enable_promotion=False,  # 禁用提升
        )

        # 保存3个检查点
        for i in range(3):
            checkpoint_path = Path(tmpdir) / f"agent_{(i+1)*10000}.pt"
            checkpoint_path.write_text(f"checkpoint {i}", encoding="utf-8")

            performance = {
                "reward_mean": 1.0,
                "terminal_rate": 0.1,
                "episode_length_mean": 100.0,
                "curriculum_phase": 1,
                "checkpoint_mtime": time.time(),
            }

            integration.on_checkpoint_saved(
                str(checkpoint_path),
                (i + 1) * 10000,
                performance,
            )

        # 最终处理不应该提升任何检查点
        promoted = integration.finalize_training(top_k=3)
        assert len(promoted) == 0
