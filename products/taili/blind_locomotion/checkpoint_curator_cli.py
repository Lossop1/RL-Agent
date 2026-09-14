"""P4.2检查点管理CLI工具。

提供命令行接口用于手动执行检查点选择、清理、提升等操作。

使用方式:
    python -m products.taili.blind_locomotion.checkpoint_curator_cli \\
        --checkpoint-dir ./checkpoints \\
        --telemetry ./telemetry.jsonl \\
        select --top-k 10

    python -m products.taili.blind_locomotion.checkpoint_curator_cli \\
        --checkpoint-dir ./checkpoints \\
        cleanup --threshold-gb 50.0

    python -m products.taili.blind_locomotion.checkpoint_curator_cli \\
        --checkpoint-dir ./checkpoints \\
        --telemetry ./telemetry.jsonl \\
        promote --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from autotuner.product.checkpoint_curator import (
    CapabilityPromotionService,
    CheckpointCurator,
    CheckpointRegistry,
    CheckpointSelector,
)


def cmd_select(args) -> int:
    """选择top-K高性能检查点。"""
    registry = CheckpointRegistry()

    # 从telemetry回填性能数据
    if args.telemetry:
        checkpoint_dir = Path(args.checkpoint_dir)
        inventory = [
            {"path": str(p), "step": None, "mtime": p.stat().st_mtime}
            for p in sorted(checkpoint_dir.glob("agent_*.pt"))
        ]
        count = registry.backfill_from_telemetry(args.telemetry, inventory)
        print(f"Backfilled {count} checkpoints from telemetry", file=sys.stderr)

    # 选择top-K
    selector = CheckpointSelector(
        min_reward_ratio=args.min_reward_ratio,
        max_terminal_rate=args.max_terminal_rate,
        min_training_step=args.min_step,
    )

    all_checkpoints = list(registry.all_checkpoints())
    if not all_checkpoints:
        print("No checkpoints registered", file=sys.stderr)
        return 1

    # 计算reward_max用于过滤
    reward_max = max(
        (perf.get("reward_mean", 0.0) for _, perf in all_checkpoints),
        default=1.0,
    )

    candidates = selector.filter_candidates(
        registry,
        {"reward_max": reward_max, "exclude_refs": set()},
    )

    weights = {
        "reward": 0.5,
        "stability": 0.3,
        "episode": 0.1,
        "recency": 0.1,
    }

    top_k = selector.rank_top_k(
        [(ref, registry.get_performance(ref)) for ref in candidates],
        k=args.top_k,
        weights=weights,
    )

    # 输出JSON格式
    result = {
        "top_k": top_k,
        "count": len(top_k),
        "total_registered": len(all_checkpoints),
        "candidates": len(candidates),
    }
    print(json.dumps(result, indent=2))
    return 0


def cmd_cleanup(args) -> int:
    """执行检查点清理策略。"""
    registry = CheckpointRegistry()
    curator = CheckpointCurator()

    # 从telemetry回填性能数据
    if args.telemetry:
        checkpoint_dir = Path(args.checkpoint_dir)
        inventory = [
            {"path": str(p), "step": None, "mtime": p.stat().st_mtime}
            for p in sorted(checkpoint_dir.glob("agent_*.pt"))
        ]
        count = registry.backfill_from_telemetry(args.telemetry, inventory)
        print(f"Backfilled {count} checkpoints from telemetry", file=sys.stderr)

    # 执行清理
    if args.threshold_gb:
        cleaned = curator.trigger_cleanup_if_needed(
            args.checkpoint_dir,
            registry,
            threshold_gb=args.threshold_gb,
        )
        result = {
            "mode": "threshold",
            "threshold_gb": args.threshold_gb,
            "cleaned": cleaned,
        }
    else:
        policy = {
            "top_k": args.keep_top_k,
            "recent_k": args.keep_recent_k,
            "recent_hours": args.recent_hours,
            "archive_age_days": args.archive_age_days,
            "delete_age_days": args.delete_age_days,
            "score_threshold_percentile": args.score_percentile,
        }
        report = curator.apply_retention_policy(args.checkpoint_dir, registry, policy)
        result = {
            "mode": "policy",
            "policy": policy,
            "kept": report["kept"],
            "archived": report["archived"],
            "deleted": report["deleted"],
            "stats": report.get("stats", {}),
        }

    print(json.dumps(result, indent=2))
    return 0


def cmd_promote(args) -> int:
    """提升top-K检查点为CapabilityProfile。"""
    registry = CheckpointRegistry()
    selector = CheckpointSelector()
    service = CapabilityPromotionService()

    # 从telemetry回填性能数据
    if args.telemetry:
        checkpoint_dir = Path(args.checkpoint_dir)
        inventory = [
            {"path": str(p), "step": None, "mtime": p.stat().st_mtime}
            for p in sorted(checkpoint_dir.glob("agent_*.pt"))
        ]
        count = registry.backfill_from_telemetry(args.telemetry, inventory)
        print(f"Backfilled {count} checkpoints from telemetry", file=sys.stderr)

    # 提升top-K
    promoted_ids = service.promote_top_performers(
        registry,
        selector,
        ledger_store=None,
        top_k=args.top_k,
        config={},
    )

    result = {
        "promoted": promoted_ids,
        "count": len(promoted_ids),
    }
    print(json.dumps(result, indent=2))
    return 0


def cmd_status(args) -> int:
    """显示检查点管理状态。"""
    registry = CheckpointRegistry()
    curator = CheckpointCurator()

    # 从telemetry回填性能数据
    if args.telemetry:
        checkpoint_dir = Path(args.checkpoint_dir)
        inventory = [
            {"path": str(p), "step": None, "mtime": p.stat().st_mtime}
            for p in sorted(checkpoint_dir.glob("agent_*.pt"))
        ]
        count = registry.backfill_from_telemetry(args.telemetry, inventory)
        print(f"Backfilled {count} checkpoints from telemetry", file=sys.stderr)

    # 统计信息
    all_checkpoints = list(registry.all_checkpoints())
    disk_usage = curator.compute_disk_usage(args.checkpoint_dir)

    selector = CheckpointSelector()
    milestones = selector.select_milestones(registry)

    result = {
        "checkpoint_dir": args.checkpoint_dir,
        "total_registered": len(all_checkpoints),
        "disk_usage_gb": disk_usage,
        "milestones": milestones,
        "milestone_count": len(milestones),
    }

    # 性能统计
    if all_checkpoints:
        rewards = [perf.get("reward_mean", 0.0) for _, perf in all_checkpoints]
        terminal_rates = [perf.get("terminal_rate", 0.0) for _, perf in all_checkpoints]

        result["performance"] = {
            "reward_mean_avg": sum(rewards) / len(rewards),
            "reward_mean_max": max(rewards),
            "terminal_rate_avg": sum(terminal_rates) / len(terminal_rates),
            "terminal_rate_min": min(terminal_rates),
        }

    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Checkpoint management CLI for P4.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Checkpoint directory path",
    )
    parser.add_argument(
        "--telemetry",
        type=str,
        default="",
        help="Telemetry JSONL file path for backfilling performance data",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # select子命令
    select_parser = subparsers.add_parser("select", help="Select top-K checkpoints")
    select_parser.add_argument("--top-k", type=int, default=10, help="Number of top checkpoints to select")
    select_parser.add_argument("--min-reward-ratio", type=float, default=0.3, help="Minimum reward ratio threshold")
    select_parser.add_argument("--max-terminal-rate", type=float, default=0.2, help="Maximum terminal rate threshold")
    select_parser.add_argument("--min-step", type=int, default=50000, help="Minimum training step threshold")

    # cleanup子命令
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up checkpoints")
    cleanup_parser.add_argument("--threshold-gb", type=float, default=None, help="Disk usage threshold (GB) for forced cleanup")
    cleanup_parser.add_argument("--keep-top-k", type=int, default=10, help="Keep top K high-performance checkpoints")
    cleanup_parser.add_argument("--keep-recent-k", type=int, default=5, help="Keep recent K checkpoints")
    cleanup_parser.add_argument("--recent-hours", type=float, default=24.0, help="Keep all checkpoints within N hours")
    cleanup_parser.add_argument("--archive-age-days", type=float, default=7.0, help="Archive checkpoints older than N days")
    cleanup_parser.add_argument("--delete-age-days", type=float, default=30.0, help="Delete checkpoints older than N days")
    cleanup_parser.add_argument("--score-percentile", type=float, default=50.0, help="Archive below this percentile")

    # promote子命令
    promote_parser = subparsers.add_parser("promote", help="Promote top-K checkpoints to CapabilityProfile")
    promote_parser.add_argument("--top-k", type=int, default=10, help="Number of top checkpoints to promote")

    # status子命令
    subparsers.add_parser("status", help="Show checkpoint management status")

    args = parser.parse_args(argv)

    if args.command == "select":
        return cmd_select(args)
    elif args.command == "cleanup":
        return cmd_cleanup(args)
    elif args.command == "promote":
        return cmd_promote(args)
    elif args.command == "status":
        return cmd_status(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
