"""P4.2检查点保存钩子，用于集成CheckpointIntegration到skrl训练循环。

使用方式：
在train_taili.py中通过猴子补丁拦截检查点落盘调用：
    from checkpoint_hook import install_checkpoint_hook
    install_checkpoint_hook(agent, env, checkpoint_dir)

拦截的是 ``agent.write_checkpoint``，不是 ``agent.save``。
skrl 1.4.3 的训练循环里，检查点是 ``agents/torch/base.py:677`` 的
``post_interaction`` 调 ``self.write_checkpoint(timestep, timesteps)`` 写的，
落盘文件名是 ``agent_<timestep>.pt``。``save(path)`` 在 1.4.3 里虽然还在
（同文件 :367），但训练循环一次都不调它——只包 save 的话，检查点照写，
登记表永远是空的。2026-09-16 的 hooksave_n4 就是这么演的：
磁盘上躺着 46 MB 的 agent_200.pt，registry 里 0 条。

``save`` 那一支保留着，因为直接调 agent.save 的下游和测试还在用。
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any


def install_checkpoint_hook(agent, env, checkpoint_dir: str) -> None:
    """为skrl agent安装检查点保存钩子。

    Args:
        agent: skrl Agent实例
        env: 训练环境实例（必须有_latest_performance_snapshot属性）
        checkpoint_dir: 检查点保存目录
    """
    try:  # 载荷内：与 checkpoint_integration 同包
        from .checkpoint_integration import CheckpointIntegration
    except ImportError as _payload_exc:  # 源码树 / 被当顶层脚本导入（没有 __package__）
        try:
            from products.taili.blind_locomotion.checkpoint_integration import (
                CheckpointIntegration,
            )
        except ImportError:
            # 两支都失败时报载荷那一支，否则真因会被 "No module named 'products'" 盖掉。
            raise _payload_exc from None

    # 创建CheckpointIntegration实例并附加到环境。
    # 判据必须是"是不是 None"，不能是 hasattr：环境在 __init__ 里就把
    # self._checkpoint_integration 置成了 None（blind_tp_env.py:179），
    # 于是 hasattr 恒为 True，集成器永远不会被创建——注册、清理、清单导出
    # 全部静默失效，而且因为走的是"已装过"那支，连那行 installed 日志都不打。
    # 2026-09-16 在 3060 容器里跑的 fixstep_n4（576 步跑到 complete）就是如此：
    # 日志里一条 CheckpointHook 都没有。
    if getattr(env, "_checkpoint_integration", None) is None:
        env._checkpoint_integration = CheckpointIntegration(
            checkpoint_dir,
            cleanup_interval=10,
            cleanup_threshold_gb=90.0,
            enable_cleanup=True,
            enable_promotion=True,
        )
        print("[CheckpointHook] installed checkpoint management", flush=True)

    def _register(path, step) -> None:
        """把一次落盘登记进集成器。失败只打印，不中断训练。"""
        integration = getattr(env, "_checkpoint_integration", None)
        if integration is None:
            return
        snapshot = getattr(env, "_latest_performance_snapshot", None)
        if snapshot is None:
            # 快照由环境的奖励日志路径每步刷新；到这儿还没有说明那一步没跑过，
            # 硬塞一个假快照会把 registry 污染成看起来有数据。
            return
        # checkpoint_mtime 记的是快照生成时刻，不是文件自己的时间。
        # 回填（backfill_from_telemetry）拿它跟检查点清单的 mtime 对齐，
        # 差几秒就可能配错档，所以这里以文件为准。
        snapshot = dict(snapshot)
        try:
            snapshot["checkpoint_mtime"] = os.path.getmtime(path)
        except OSError:
            pass
        integration.on_checkpoint_saved(path, step, snapshot)

    def _checkpoint_dir() -> Path:
        """skrl 把检查点写在 <experiment_dir>/checkpoints 下。

        以 agent 自己的 experiment_dir 为准：它才是真正落盘的地方。
        取不到时退回安装时传进来的目录。
        """
        experiment_dir = getattr(agent, "experiment_dir", None)
        if experiment_dir:
            return Path(experiment_dir) / "checkpoints"
        return Path(checkpoint_dir)

    if getattr(agent, "write_checkpoint", None) is None:
        # 不抛异常：调用点的 except 会把它降级成一行日志，等于又静默失效一次。
        # 训练照跑，只是检查点不被登记——这件事必须吼出来。
        print(
            "[CheckpointHook] WARNING: agent 没有 write_checkpoint（skrl 版本不符？），"
            "训练循环写出的检查点不会被登记",
            flush=True,
        )
        return

    original_write_checkpoint = agent.write_checkpoint

    @functools.wraps(original_write_checkpoint)
    def write_checkpoint_with_hook(timestep, timesteps, *args, **kwargs):
        """包装 skrl 训练循环真正调用的那个方法。

        用"前后比目录"而不是自己拼文件名：skrl 的名字随
        ``checkpoint_store_separately`` 变（``agent_<t>.pt`` 或 ``<module>_<t>.pt``），
        拼错了会静默登记一个不存在的文件。比目录只多一次 listdir，
        而它每 checkpoint_interval 步才跑一次。
        """
        directory = _checkpoint_dir()
        try:
            before = {p.name for p in directory.iterdir()} if directory.is_dir() else set()
        except OSError:
            before = set()

        result = original_write_checkpoint(timestep, timesteps, *args, **kwargs)

        try:
            if directory.is_dir():
                for path in sorted(directory.iterdir()):
                    if path.name not in before and path.is_file():
                        _register(str(path), timestep)
        except Exception as e:
            print(f"[CheckpointHook] hook failed: {e}", flush=True)

        return result

    agent.write_checkpoint = write_checkpoint_with_hook

    # save() 不在 skrl 1.4.3 的训练循环里，但下游和测试会直接调，一并包上。
    original_save = agent.save

    @functools.wraps(original_save)
    def save_with_hook(path: str, *args, **kwargs):
        """包装后的save方法，在保存后触发CheckpointIntegration。"""
        result = original_save(path, *args, **kwargs)
        try:
            _register(path, getattr(agent, "timestep", 0))
        except Exception as e:
            # 钩子失败不应中断训练
            print(f"[CheckpointHook] hook failed: {e}", flush=True)
        return result

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
