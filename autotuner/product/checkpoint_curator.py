"""检查点管理模块：注册、筛选、清理和能力提升

核心功能：
1. CheckpointRegistry - 维护检查点到性能快照的映射
2. CheckpointSelector - 基于多维度评分筛选高性能检查点
3. CheckpointCurator - 执行清理策略，控制磁盘占用
4. CapabilityPromotionService - 提取能力特征并创建CapabilityProfile
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autotuner.research.research_ledger import CapabilityProfile, ResearchLedgerStore


@dataclass
class PerformanceSnapshot:
    """检查点性能快照

    Attributes:
        step: 训练步数
        reward_mean: 平均奖励
        terminal_rate: 终止率（越低越稳定）
        episode_length_mean: 平均episode长度
        curriculum_phase: 课程阶段
        curriculum_level: 课程级别
        checkpoint_mtime: 检查点修改时间（Unix时间戳）
        raw_metrics: 原始指标字典
    """
    step: int
    reward_mean: float
    terminal_rate: float
    episode_length_mean: float
    curriculum_phase: int
    curriculum_level: int
    checkpoint_mtime: float
    raw_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "step": self.step,
            "reward_mean": self.reward_mean,
            "terminal_rate": self.terminal_rate,
            "episode_length_mean": self.episode_length_mean,
            "curriculum_phase": self.curriculum_phase,
            "curriculum_level": self.curriculum_level,
            "checkpoint_mtime": self.checkpoint_mtime,
            "raw_metrics": self.raw_metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerformanceSnapshot:
        """从字典反序列化"""
        return cls(
            step=data["step"],
            reward_mean=data["reward_mean"],
            terminal_rate=data["terminal_rate"],
            episode_length_mean=data["episode_length_mean"],
            curriculum_phase=data["curriculum_phase"],
            curriculum_level=data["curriculum_level"],
            checkpoint_mtime=data["checkpoint_mtime"],
            raw_metrics=data.get("raw_metrics", {}),
        )


class CheckpointRegistry:
    """检查点注册表

    维护检查点文件路径到性能快照的映射关系。
    支持实时注册、历史数据回填、按步数范围查询和导出清单功能。
    """

    def __init__(self):
        """初始化空注册表"""
        self._registry: dict[str, PerformanceSnapshot] = {}

    def register(
        self,
        checkpoint_ref: str,
        step: int,
        performance_snapshot: dict[str, Any],
    ) -> None:
        """注册检查点性能快照

        Args:
            checkpoint_ref: 检查点文件路径（相对或绝对）
            step: 训练步数
            performance_snapshot: 性能快照字典，包含：
                - reward_mean: 平均奖励
                - terminal_rate: 终止率
                - episode_length_mean: 平均episode长度
                - curriculum_phase: 课程阶段
                - curriculum_level: 课程级别（可选）
                - checkpoint_mtime: 检查点修改时间
                - raw_metrics: 其他原始指标（可选）
        """
        snapshot = PerformanceSnapshot(
            step=step,
            reward_mean=performance_snapshot["reward_mean"],
            terminal_rate=performance_snapshot["terminal_rate"],
            episode_length_mean=performance_snapshot["episode_length_mean"],
            curriculum_phase=performance_snapshot["curriculum_phase"],
            curriculum_level=performance_snapshot.get("curriculum_level", 0),
            checkpoint_mtime=performance_snapshot["checkpoint_mtime"],
            raw_metrics=performance_snapshot.get("raw_metrics", {}),
        )
        self._registry[checkpoint_ref] = snapshot

    def get_performance(self, checkpoint_ref: str) -> dict[str, Any] | None:
        """获取检查点性能快照

        Args:
            checkpoint_ref: 检查点文件路径

        Returns:
            性能快照字典，如果不存在返回None
        """
        snapshot = self._registry.get(checkpoint_ref)
        return snapshot.to_dict() if snapshot else None

    def backfill_from_telemetry(
        self,
        telemetry_jsonl_path: str,
        checkpoint_inventory: list[dict[str, Any]],
    ) -> int:
        """从telemetry JSONL回填历史检查点性能数据

        通过step匹配checkpoint文件的mtime，建立性能映射。

        Args:
            telemetry_jsonl_path: telemetry JSONL文件路径
            checkpoint_inventory: 检查点清单列表，每项包含：
                - path: 检查点路径
                - mtime: 修改时间
                - step: 训练步数（可选，从文件名推断）

        Returns:
            成功回填的检查点数量
        """
        # 读取telemetry数据构建step到性能的映射
        step_to_perf: dict[int, dict[str, Any]] = {}
        try:
            with open(telemetry_jsonl_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    step = entry.get("step")
                    if step is None:
                        continue
                    reward = entry.get("reward", {})
                    health = entry.get("health", {})
                    curriculum = entry.get("curriculum", {})
                    step_to_perf[step] = {
                        "reward_mean": reward.get("mean", 0.0),
                        "terminal_rate": health.get("terminal_rate", 0.0),
                        "episode_length_mean": health.get("episode_length_mean", 0.0),
                        "curriculum_phase": curriculum.get("phase", 0),
                        "curriculum_level": curriculum.get("level", 0),
                    }
        except (FileNotFoundError, json.JSONDecodeError):
            return 0

        # 匹配检查点到性能
        backfilled_count = 0
        for ckpt_info in checkpoint_inventory:
            ckpt_path = ckpt_info["path"]
            ckpt_step = ckpt_info.get("step")
            ckpt_mtime = ckpt_info.get("mtime", 0.0)

            # 如果step未提供，尝试从文件名提取（agent_XXXXX.pt格式）
            if ckpt_step is None:
                ckpt_filename = Path(ckpt_path).stem
                if ckpt_filename.startswith("agent_"):
                    try:
                        ckpt_step = int(ckpt_filename.split("_")[1])
                    except (IndexError, ValueError):
                        continue
                else:
                    continue

            # 查找最接近的telemetry记录
            perf = step_to_perf.get(ckpt_step)
            if perf:
                perf["checkpoint_mtime"] = ckpt_mtime
                self.register(ckpt_path, ckpt_step, perf)
                backfilled_count += 1

        return backfilled_count

    def query_by_step_range(
        self,
        min_step: int,
        max_step: int,
    ) -> list[tuple[str, dict[str, Any]]]:
        """按步数范围查询检查点

        Args:
            min_step: 最小步数（包含）
            max_step: 最大步数（包含）

        Returns:
            符合条件的(checkpoint_ref, performance_dict)列表
        """
        results = []
        for ckpt_ref, snapshot in self._registry.items():
            if min_step <= snapshot.step <= max_step:
                results.append((ckpt_ref, snapshot.to_dict()))
        return results

    def export_to_manifest(self, output_path: str) -> None:
        """导出检查点性能清单到JSON文件

        Args:
            output_path: 输出文件路径
        """
        manifest = {
            "version": "checkpoint_registry/v1",
            "count": len(self._registry),
            "checkpoints": {
                ckpt_ref: snapshot.to_dict()
                for ckpt_ref, snapshot in self._registry.items()
            },
        }
        Path(output_path).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def all_checkpoints(self) -> list[tuple[str, dict[str, Any]]]:
        """获取所有检查点及其性能快照

        Returns:
            所有(checkpoint_ref, performance_dict)列表
        """
        return [(ref, snap.to_dict()) for ref, snap in self._registry.items()]


class CheckpointSelector:
    """检查点筛选器

    基于多维度加权评分函数筛选高性能检查点。
    评分函数综合考虑：reward_mean (50%)、稳定性1-terminal_rate (30%)、
    episode_length (10%)、新鲜度exp衰减 (10%)。
    """

    def __init__(
        self,
        *,
        min_reward_ratio: float = 0.3,
        max_terminal_rate: float = 0.2,
        min_training_step: int = 50000,
    ):
        """初始化筛选器

        Args:
            min_reward_ratio: 最小reward相对最大值的比例（0-1）
            max_terminal_rate: 最大终止率阈值
            min_training_step: 最小训练步数要求
        """
        self.min_reward_ratio = min_reward_ratio
        self.max_terminal_rate = max_terminal_rate
        self.min_training_step = min_training_step

    def score(
        self,
        performance: dict[str, Any],
        weights: dict[str, float],
        recency_base_step: int,
        reward_max: float = 1.0,
        episode_max: float = 1.0,
    ) -> float:
        """计算检查点综合得分

        Args:
            performance: 性能快照字典
            weights: 权重字典，包含reward、stability、episode、recency
            recency_base_step: 新鲜度计算基准步数（通常为最新步数）
            reward_max: 最大奖励值（用于归一化）
            episode_max: 最大episode长度（用于归一化）

        Returns:
            综合得分（0-1范围）
        """
        reward_weight = weights.get("reward", 0.5)
        stability_weight = weights.get("stability", 0.3)
        episode_weight = weights.get("episode", 0.1)
        recency_weight = weights.get("recency", 0.1)

        # 归一化奖励到[0,1]
        reward_score = performance.get("reward_mean", 0.0) / max(reward_max, 1e-6)

        # 稳定性得分：1 - terminal_rate
        stability_score = 1.0 - performance.get("terminal_rate", 0.0)

        # episode长度归一化到[0,1]
        episode_score = performance.get("episode_length_mean", 0.0) / max(episode_max, 1e-6)

        # 新鲜度得分：基于步数差距的指数衰减（24小时半衰期，假设2000步/小时）
        step = performance.get("step", 0)
        age_steps = max(0, recency_base_step - step)
        age_hours = age_steps / 2000.0  # 假设2000步/小时
        recency_score = math.exp(-age_hours / 24.0)

        # 加权综合得分
        score = (
            reward_weight * reward_score
            + stability_weight * stability_score
            + episode_weight * episode_score
            + recency_weight * recency_score
        )

        return max(0.0, min(1.0, score))  # 限制在[0,1]

    def filter_candidates(
        self,
        registry: CheckpointRegistry,
        criteria: dict[str, Any],
    ) -> list[str]:
        """过滤符合条件的检查点

        Args:
            registry: 检查点注册表
            criteria: 过滤条件字典，包含：
                - reward_max: 最大奖励值（用于比例计算）
                - exclude_refs: 排除的检查点引用集合（里程碑检查点）

        Returns:
            符合条件的检查点引用列表
        """
        reward_max = criteria.get("reward_max", 1.0)
        min_reward_threshold = self.min_reward_ratio * reward_max
        exclude_refs = criteria.get("exclude_refs", set())

        candidates = []
        for ref, perf in registry.all_checkpoints():
            # 跳过排除列表中的检查点
            if ref in exclude_refs:
                continue

            # 应用硬性过滤条件
            if perf["step"] < self.min_training_step:
                continue
            if perf["reward_mean"] < min_reward_threshold:
                continue
            if perf["terminal_rate"] > self.max_terminal_rate:
                continue

            candidates.append(ref)

        return candidates

    def rank_top_k(
        self,
        candidates: list[tuple[str, dict[str, Any]]],
        k: int,
        weights: dict[str, float],
    ) -> list[str]:
        """对候选检查点评分并返回top-K

        Args:
            candidates: 候选检查点列表，每项为(ref, performance)
            k: 保留数量
            weights: 评分权重字典

        Returns:
            top-K检查点引用列表，按得分降序
        """
        if not candidates:
            return []

        # 计算最新步数作为新鲜度基准
        max_step = max(perf["step"] for _, perf in candidates)

        # 计算reward和episode的最大值用于归一化
        reward_max = max(perf["reward_mean"] for _, perf in candidates)
        episode_max = max(perf["episode_length_mean"] for _, perf in candidates)

        # 计算所有候选的得分
        scored = []
        for ref, perf in candidates:
            score = self.score(perf, weights, max_step, reward_max, episode_max)
            scored.append((ref, score))

        # 按得分降序排序，取top-K
        scored.sort(key=lambda x: x[1], reverse=True)
        return [ref for ref, _ in scored[:k]]

    def select_milestones(
        self,
        registry: CheckpointRegistry,
    ) -> list[str]:
        """识别里程碑检查点

        里程碑包括：
        - curriculum phase转换点（每个phase的首个和末个检查点）
        - 历史最佳reward检查点

        Args:
            registry: 检查点注册表

        Returns:
            里程碑检查点引用列表
        """
        all_ckpts = registry.all_checkpoints()
        if not all_ckpts:
            return []

        milestones = []

        # 按step排序
        sorted_ckpts = sorted(all_ckpts, key=lambda x: x[1]["step"])

        # 识别phase转换点
        phase_transitions: dict[int, list[str]] = {}
        for ref, perf in sorted_ckpts:
            phase = perf["curriculum_phase"]
            if phase not in phase_transitions:
                phase_transitions[phase] = []
            phase_transitions[phase].append(ref)

        # 每个phase的首个和末个
        for phase, refs in phase_transitions.items():
            if refs:
                milestones.append(refs[0])  # 首个
                if len(refs) > 1:
                    milestones.append(refs[-1])  # 末个

        # 历史最佳reward
        best_reward_ref = max(all_ckpts, key=lambda x: x[1]["reward_mean"])[0]
        if best_reward_ref not in milestones:
            milestones.append(best_reward_ref)

        return list(set(milestones))  # 去重


class CheckpointCurator:
    """检查点管理器

    执行分层保留策略安全删除低性能检查点并控制磁盘占用。
    """

    def __init__(self, archive_dir: str = "archive"):
        """初始化管理器

        Args:
            archive_dir: 归档目录名（相对于checkpoint_dir）
        """
        self.archive_dir = archive_dir

    def apply_retention_policy(
        self,
        checkpoint_dir: str,
        registry: CheckpointRegistry,
        policy: dict[str, Any],
    ) -> dict[str, Any]:
        """应用保留策略

        Args:
            checkpoint_dir: 检查点目录路径
            registry: 检查点注册表
            policy: 保留策略字典，包含：
                - top_k: 保留top-K高性能检查点数量
                - recent_k: 保留最近K个检查点数量
                - recent_hours: 保留最近N小时内的所有检查点
                - archive_age_days: 归档低性能检查点的年龄阈值（天）
                - delete_age_days: 物理删除归档检查点的年龄阈值（天）
                - score_threshold_percentile: 评分百分位阈值（低于此值的归档）

        Returns:
            执行报告字典，包含kept/archived/deleted列表
        """
        import time

        checkpoint_path = Path(checkpoint_dir)
        all_ckpts = registry.all_checkpoints()

        if not all_ckpts:
            return {"kept": [], "archived": [], "deleted": []}

        # 策略参数
        top_k = policy.get("top_k", 10)
        recent_k = policy.get("recent_k", 5)
        recent_hours = policy.get("recent_hours", 24)
        archive_age_days = policy.get("archive_age_days", 7)
        delete_age_days = policy.get("delete_age_days", 30)
        score_percentile = policy.get("score_threshold_percentile", 50)

        # 识别里程碑检查点（自动保留）
        selector = CheckpointSelector()
        milestones = set(selector.select_milestones(registry))

        # 计算当前时间
        current_time = time.time()

        # 按步数排序，获取最近的recent_k个
        sorted_by_step = sorted(all_ckpts, key=lambda x: x[1]["step"], reverse=True)
        recent_refs = {ref for ref, _ in sorted_by_step[:recent_k]}

        # 获取最近recent_hours小时内的检查点
        recent_time_threshold = current_time - (recent_hours * 3600)
        recent_by_time = {
            ref for ref, perf in all_ckpts
            if perf["checkpoint_mtime"] >= recent_time_threshold
        }

        # 计算所有检查点的评分并选择top-K
        weights = {"reward": 0.5, "stability": 0.3, "episode": 0.1, "recency": 0.1}
        top_performers = set(selector.rank_top_k(all_ckpts, top_k, weights))

        # 计算评分百分位阈值
        if all_ckpts:
            max_step = max(perf["step"] for _, perf in all_ckpts)
            reward_max = max(perf["reward_mean"] for _, perf in all_ckpts)
            episode_max = max(perf["episode_length_mean"] for _, perf in all_ckpts)

            scores = [
                selector.score(perf, weights, max_step, reward_max, episode_max)
                for _, perf in all_ckpts
            ]
            scores.sort()
            threshold_idx = int(len(scores) * score_percentile / 100.0)
            score_threshold = scores[threshold_idx] if threshold_idx < len(scores) else 0.0
        else:
            score_threshold = 0.0

        # 分类检查点
        kept = []
        to_archive = []
        to_delete = []

        for ref, perf in all_ckpts:
            ckpt_path = checkpoint_path / ref
            age_days = (current_time - perf["checkpoint_mtime"]) / 86400.0

            # 保留条件：里程碑、top-K、最近的
            if (
                ref in milestones
                or ref in top_performers
                or ref in recent_refs
                or ref in recent_by_time
            ):
                kept.append(ref)
                continue

            # 删除条件（已归档且超过delete_age_days）
            archive_path = checkpoint_path / self.archive_dir / Path(ref).name
            if archive_path.exists() and age_days >= delete_age_days:
                to_delete.append(ref)
                continue

            # 归档条件（低性能且age >= archive_age_days）
            score = selector.score(perf, weights, max_step, reward_max, episode_max)
            if score < score_threshold and age_days >= archive_age_days:
                to_archive.append(ref)
            else:
                kept.append(ref)

        return {
            "kept": kept,
            "archived": to_archive,
            "deleted": to_delete,
            "policy": policy,
            "stats": {
                "total": len(all_ckpts),
                "milestones": len(milestones),
                "top_performers": len(top_performers),
                "recent_by_step": len(recent_refs),
                "recent_by_time": len(recent_by_time),
            },
        }

    def archive_checkpoint(self, checkpoint_path: str, archive_dir: str) -> str:
        """归档检查点到archive目录

        Args:
            checkpoint_path: 检查点文件路径
            archive_dir: 归档目录路径

        Returns:
            归档后的文件路径
        """
        import shutil

        src = Path(checkpoint_path)
        archive_path = Path(archive_dir)
        archive_path.mkdir(parents=True, exist_ok=True)

        dest = archive_path / src.name
        shutil.move(str(src), str(dest))

        return str(dest)

    def delete_checkpoint(
        self,
        checkpoint_path: str,
        ledger_store: ResearchLedgerStore | None = None,
    ) -> None:
        """物理删除检查点文件

        Args:
            checkpoint_path: 检查点文件路径
            ledger_store: 研究账本存储（可选，记录删除操作）
        """
        ckpt_path = Path(checkpoint_path)
        if ckpt_path.exists():
            ckpt_path.unlink()

            # 记录删除操作到账本（如果提供）
            if ledger_store is not None:
                # TODO: 添加删除记录到ledger
                pass

    def compute_disk_usage(self, checkpoint_dir: str) -> float:
        """计算检查点目录磁盘占用

        Args:
            checkpoint_dir: 检查点目录路径

        Returns:
            磁盘占用大小（GB）
        """
        total_size = 0
        checkpoint_path = Path(checkpoint_dir)

        if not checkpoint_path.exists():
            return 0.0

        for item in checkpoint_path.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size

        return total_size / (1024**3)  # 转换为GB

    def trigger_cleanup_if_needed(
        self,
        checkpoint_dir: str,
        registry: CheckpointRegistry,
        threshold_gb: float,
    ) -> int:
        """当磁盘占用超过阈值时触发强制清理

        Args:
            checkpoint_dir: 检查点目录路径
            registry: 检查点注册表
            threshold_gb: 磁盘占用阈值（GB）

        Returns:
            清理的检查点数量
        """
        current_usage = self.compute_disk_usage(checkpoint_dir)

        if current_usage < threshold_gb:
            return 0

        # 强制清理策略：仅保留top-5 + 最近3个 + 里程碑
        emergency_policy = {
            "top_k": 5,
            "recent_k": 3,
            "recent_hours": 24,
            "archive_age_days": 0,  # 立即归档
            "delete_age_days": 0,   # 立即删除
            "score_threshold_percentile": 70,  # 提高阈值
        }

        report = self.apply_retention_policy(checkpoint_dir, registry, emergency_policy)

        # 执行归档
        checkpoint_path = Path(checkpoint_dir)
        archive_path = checkpoint_path / self.archive_dir

        archived_count = 0
        for ref in report["archived"]:
            src = checkpoint_path / ref
            if src.exists():
                self.archive_checkpoint(str(src), str(archive_path))
                archived_count += 1

        # 执行删除
        deleted_count = 0
        for ref in report["deleted"]:
            # 优先删除归档文件
            archive_file = archive_path / Path(ref).name
            if archive_file.exists():
                archive_file.unlink()
                deleted_count += 1
            # 如果主文件也在删除列表，一并删除
            main_file = checkpoint_path / ref
            if main_file.exists():
                main_file.unlink()
                deleted_count += 1

        # 额外清理：如果仍未达到目标，删除不在kept列表中的文件
        if archived_count + deleted_count == 0:
            all_ckpts = registry.all_checkpoints()
            kept_set = set(report["kept"])

            for ref, _ in all_ckpts:
                if ref not in kept_set:
                    main_file = checkpoint_path / ref
                    if main_file.exists():
                        # 先归档后删除
                        self.archive_checkpoint(str(main_file), str(archive_path))
                        archived_count += 1

        return archived_count + deleted_count


class CapabilityPromotionService:
    """检查点能力提升服务

    从高性能检查点提取能力特征并创建CapabilityProfile记录写入研究台账。
    """

    def extract_capabilities(
        self,
        checkpoint_ref: str,
        performance: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """从检查点提取能力特征

        Args:
            checkpoint_ref: 检查点文件引用
            performance: 性能快照
            config: 训练配置

        Returns:
            能力特征字典
        """
        capabilities = {}

        # 从性能快照提取关键能力
        reward_mean = performance.get("reward_mean", 0.0)
        terminal_rate = performance.get("terminal_rate", 1.0)
        episode_length = performance.get("episode_length_mean", 0.0)
        curriculum_phase = performance.get("curriculum_phase", 0)

        # 速度范围能力（基于reward和episode_length推断）
        if reward_mean > 1.5 and episode_length > 100:
            capabilities["speed_range"] = "medium_to_high"
        elif reward_mean > 0.8:
            capabilities["speed_range"] = "low_to_medium"
        else:
            capabilities["speed_range"] = "low"

        # 地形等级（基于curriculum phase）
        capabilities["terrain_levels"] = [f"phase_{curriculum_phase}"]
        if curriculum_phase >= 2:
            capabilities["terrain_levels"].append("uneven_terrain")
        if curriculum_phase >= 3:
            capabilities["terrain_levels"].append("obstacles")

        # 命令模式（从config推断或默认）
        command_modes = config.get("command_modes", ["forward", "turn"])
        capabilities["command_modes"] = command_modes

        # 质量门控（基于稳定性）
        quality_gates = {}
        quality_gates["stability"] = "pass" if terminal_rate < 0.15 else "fail"
        quality_gates["endurance"] = "pass" if episode_length > 80 else "fail"
        quality_gates["reward_threshold"] = "pass" if reward_mean > 0.5 else "fail"

        capabilities["quality_gates"] = quality_gates

        return capabilities

    def create_profile(
        self,
        checkpoint_ref: str,
        capabilities: dict[str, Any],
        status: str = "candidate",
    ) -> CapabilityProfile:
        """创建CapabilityProfile记录

        Args:
            checkpoint_ref: 检查点文件引用
            capabilities: 能力特征字典
            status: 状态（observed/candidate/validated）

        Returns:
            CapabilityProfile实例
        """
        import uuid
        from autotuner.research.research_ledger import CapabilityProfile

        profile = CapabilityProfile(
            id=str(uuid.uuid4()),
            checkpoint_ref=checkpoint_ref,
            capabilities=capabilities,
            status=status,  # type: ignore
        )

        return profile

    def promote_top_performers(
        self,
        registry: CheckpointRegistry,
        selector: CheckpointSelector,
        ledger_store: ResearchLedgerStore | None,
        top_k: int = 5,
        config: dict[str, Any] | None = None,
    ) -> list[str]:
        """批量提升top-K检查点为CapabilityProfile

        Args:
            registry: 检查点注册表
            selector: 检查点选择器
            ledger_store: 研究账本存储
            top_k: 提升top-K个检查点
            config: 训练配置（用于提取能力）

        Returns:
            提升的profile ID列表
        """
        if config is None:
            config = {}

        all_ckpts = registry.all_checkpoints()
        if not all_ckpts:
            return []

        # 选择top-K检查点
        weights = {"reward": 0.5, "stability": 0.3, "episode": 0.1, "recency": 0.1}
        top_refs = selector.rank_top_k(all_ckpts, top_k, weights)

        promoted_ids = []
        for ref in top_refs:
            perf = registry.get_performance(ref)
            if perf is None:
                continue

            # 提取能力特征
            capabilities = self.extract_capabilities(ref, perf, config)

            # 创建profile
            profile = self.create_profile(ref, capabilities, status="candidate")

            # 写入ledger（如果提供）
            if ledger_store is not None:
                profile_id = ledger_store.append(profile)
                promoted_ids.append(profile_id)
            else:
                # 无ledger时返回checkpoint_ref作为ID
                promoted_ids.append(ref)

        return promoted_ids

    def update_status(
        self,
        profile_id: str,
        new_status: str,
        ledger_store: ResearchLedgerStore,
    ) -> None:
        """更新CapabilityProfile状态

        Args:
            profile_id: profile记录ID
            new_status: 新状态（observed/candidate/validated）
            ledger_store: 研究账本存储
        """
        # TODO: 实现状态更新逻辑
        # 需要从ledger读取profile，更新status字段，重新写入
        pass

