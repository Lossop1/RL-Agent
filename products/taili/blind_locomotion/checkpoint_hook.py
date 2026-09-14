"""P4.2检查点保存钩子，用于集成CheckpointIntegration到skrl训练循环。

使用方式：
在train_taili.py中通过猴子补丁拦截agent.save()调用：
    from checkpoint_hook import install_checkpoint_hook
    install_checkpoint_hook(agent, env, checkpoint_dir)
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any


def install_checkpoint_hook(agent, env, checkpoint_dir: str) -> None:
    """为skrl agent安装检查点保存钩子。

    Args:
        agent: skrl Agent实例
        env: 训练环境实例（必须有_latest_performance_snapshot属性）
        checkpoint_dir: 检查点保存目录
    """
    from products.taili.blind_locomotion.checkpoint_integration import (
        CheckpointIntegration,
    )

    # 创建CheckpointIntegration实例并附加到环境
    if not hasattr(env, "_checkpoint_integration"):
        env._checkpoint_integration = CheckpointIntegration(
            checkpoint_dir,
            cleanup_interval=10,
            cleanup_threshold_gb=90.0,
            enable_cleanup=True,
            enable_promotion=True,
        )
        print("[CheckpointHook] installed checkpoint management", flush=True)

    # 保存原始save方法
    original_save = agent.save

    @functools.wraps(original_save)
    def save_with_hook(path: str, *args, **kwargs):
        """包装后的save方法，在保存后触发CheckpointIntegration。"""
        # 调用原始save方法
        result = original_save(path, *args, **kwargs)

        # 触发CheckpointIntegration回调
        try:
            # 获取当前训练步数
            timestep = getattr(agent, "timestep", 0)

            # 从环境获取最新性能快照
            performance_snapshot = getattr(env, "_latest_performance_snapshot", None)

            if performance_snapshot is not None and env._checkpoint_integration is not None:
                env._checkpoint_integration.on_checkpoint_saved(
                    path,
                    timestep,
                    performance_snapshot,
                )
        except Exception as e:
            # 钩子失败不应中断训练
            print(f"[CheckpointHook] hook failed: {e}", flush=True)

        return result

    # 替换agent.save方法
    agent.save = save_with_hook


def finalize_checkpoint_management(env, config: dict[str, Any] | None = None) -> None:
    """训练结束时执行最终清理和能力提升。

    Args:
        env: 训练环境实例
        config: 训练配置（可选）
    """
    if hasattr(env, "_checkpoint_integration") and env._checkpoint_integration is not None:
        try:
            promoted = env._checkpoint_integration.finalize_training(
                top_k=10,
                config=config,
                ledger_store=None,
            )
            print(
                f"[CheckpointHook] finalized checkpoint management, promoted {len(promoted)} checkpoints",
                flush=True,
            )

            # 导出清单
            try:
                manifest_path = Path(env._checkpoint_integration.checkpoint_dir) / "checkpoint_manifest.json"
                env._checkpoint_integration.export_manifest(str(manifest_path))
            except Exception as e:
                print(f"[CheckpointHook] manifest export failed: {e}", flush=True)

            # 打印统计信息
            stats = env._checkpoint_integration.get_statistics()
            print(f"[CheckpointHook] statistics: {stats}", flush=True)
        except Exception as e:
            print(f"[CheckpointHook] finalize failed: {e}", flush=True)
