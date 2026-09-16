"""P4.2检查点管理训练集成模块。

职责：
- 在训练过程中捕获检查点保存事件并注册到CheckpointRegistry
- 周期性触发CheckpointCurator清理策略控制磁盘占用
- 训练结束时调用CapabilityPromotionService提升top-K检查点

使用方式：
1. 训练开始时创建CheckpointIntegration实例
2. 每次检查点保存后调用on_checkpoint_saved()
3. 训练结束时调用finalize_training()
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

try:  # 载荷内：两个模块被拍平到 taili_blind_runtime/
    from .checkpoint_curator import (
        CapabilityPromotionService,
        CheckpointCurator,
        CheckpointRegistry,
        CheckpointSelector,
        ResearchLedgerUnavailable,
    )
except ImportError as _payload_exc:  # 源码树：checkpoint_curator 住在 autotuner/product/
    try:
        from autotuner.product.checkpoint_curator import (
            CapabilityPromotionService,
            CheckpointCurator,
            CheckpointRegistry,
            CheckpointSelector,
            ResearchLedgerUnavailable,
        )
    except ImportError:
        # 两支都失败时先报载荷那一支，否则真因（例如容器里没有 pydantic）会被
        # "No module named 'autotuner'" 盖掉。
        raise _payload_exc from None


class CheckpointIntegration:
    """检查点管理训练集成器。"""

    def __init__(
        self,
        checkpoint_dir: str,
        *,
        cleanup_interval: int = 10,
        cleanup_threshold_gb: float = 90.0,
        enable_cleanup: bool = True,
        enable_promotion: bool = True,
    ):
        """初始化集成器。

        Args:
            checkpoint_dir: 检查点保存目录
            cleanup_interval: 每保存N个检查点触发一次清理，默认10
            cleanup_threshold_gb: 磁盘占用阈值（GB），超过时触发强制清理，默认90GB
            enable_cleanup: 是否启用自动清理，默认True
            enable_promotion: 是否启用能力提升，默认True
        """
        self.checkpoint_dir = checkpoint_dir
        self.cleanup_interval = cleanup_interval
        self.cleanup_threshold_gb = cleanup_threshold_gb
        self.enable_cleanup = enable_cleanup
        self.enable_promotion = enable_promotion

        self.registry = CheckpointRegistry()
        self.curator = CheckpointCurator()
        self.promotion_service = CapabilityPromotionService()
        self.selector = CheckpointSelector()

        self._checkpoint_count = 0
        self._last_cleanup_step = 0

    def on_checkpoint_saved(
        self,
        checkpoint_path: str,
        step: int,
        performance_snapshot: dict[str, Any],
    ) -> None:
        """检查点保存回调，注册到Registry并触发周期性清理。

        Args:
            checkpoint_path: 检查点文件路径（相对或绝对）
            step: 训练步数
            performance_snapshot: 性能快照，必须包含：
                - reward_mean: 平均奖励
                - terminal_rate: 终止率
                - episode_length_mean: 平均episode长度
                - curriculum_phase: 课程阶段
                - checkpoint_mtime: 检查点修改时间戳
        """
        # 注册到registry
        self.registry.register(checkpoint_path, step, performance_snapshot)
        self._checkpoint_count += 1

        # 周期性触发清理
        if self.enable_cleanup and self._checkpoint_count % self.cleanup_interval == 0:
            self._trigger_cleanup(step)

    def _trigger_cleanup(self, current_step: int) -> None:
        """触发检查点清理策略。

        Args:
            current_step: 当前训练步数
        """
        try:
            cleaned = self.curator.trigger_cleanup_if_needed(
                self.checkpoint_dir,
                self.registry,
                threshold_gb=self.cleanup_threshold_gb,
            )
            if cleaned > 0:
                print(
                    f"[CheckpointIntegration] step={current_step} cleaned={cleaned} checkpoints",
                    flush=True,
                )
            self._last_cleanup_step = current_step
        except Exception as e:
            # 清理失败不应中断训练
            print(f"[CheckpointIntegration] cleanup failed: {e}", flush=True)

    def finalize_training(
        self,
        *,
        top_k: int = 10,
        config: dict[str, Any] | None = None,
        ledger_store=None,
    ) -> list[str]:
        """训练结束时执行最终清理和能力提升。

        Args:
            top_k: 提升前K个高性能检查点，默认10
            config: 训练配置（可选），用于提取command_modes等能力特征
            ledger_store: 研究账本存储（可选），用于持久化CapabilityProfile

        Returns:
            提升的检查点ID列表
        """
        promoted_ids = []

        # 执行最终清理
        if self.enable_cleanup:
            try:
                self._trigger_cleanup(current_step=-1)
            except Exception as e:
                print(f"[CheckpointIntegration] final cleanup failed: {e}", flush=True)

        # 提升top-K检查点
        if self.enable_promotion:
            try:
                promoted_ids = self.promotion_service.promote_top_performers(
                    self.registry,
                    self.selector,
                    ledger_store=ledger_store,
                    top_k=top_k,
                    config=config or {},
                )
                if promoted_ids:
                    print(
                        f"[CheckpointIntegration] promoted {len(promoted_ids)} checkpoints to capability profiles",
                        flush=True,
                    )
            except ResearchLedgerUnavailable as e:
                # 训练容器里没有 pydantic，这里必然走到。能力提升只服务研究台账，
                # 注册表/清理/清单导出都不依赖它，所以这是一句说明而不是故障。
                print(
                    f"[CheckpointIntegration] 跳过能力提升：研究台账不可用（{e}）。"
                    "检查点清理与清单导出不受影响。",
                    flush=True,
                )
            except Exception as e:
                print(f"[CheckpointIntegration] promotion failed: {e}", flush=True)

        return promoted_ids

    def export_manifest(self, output_path: str) -> None:
        """导出检查点清单到JSON文件。

        Args:
            output_path: 输出文件路径
        """
        self.registry.export_to_manifest(output_path)
        print(
            f"[CheckpointIntegration] exported checkpoint manifest to {output_path}",
            flush=True,
        )

    def get_statistics(self) -> dict[str, Any]:
        """获取当前统计信息。

        Returns:
            统计信息字典，包含：
            - total_registered: 已注册检查点数量
            - checkpoint_count: 保存检查点计数
            - last_cleanup_step: 上次清理步数
            - disk_usage_gb: 当前磁盘占用（GB）
        """
        disk_usage = self.curator.compute_disk_usage(self.checkpoint_dir)
        total_registered = len(list(self.registry.all_checkpoints()))

        return {
            "total_registered": total_registered,
            "checkpoint_count": self._checkpoint_count,
            "last_cleanup_step": self._last_cleanup_step,
            "disk_usage_gb": disk_usage,
        }
