#!/usr/bin/env python3
"""Remap a Taili curriculum state after changing terrain column proportions.

IsaacLab stores a column index in ``terrain_types``. The semantic terrain type
for that column is derived from the current terrain proportions, so restoring a
state under a different mix can silently reinterpret old flat/stair columns.
This tool transfers levels by semantic terrain type and preserves the separate
fixed-replay and frontier distributions for stair terrains.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml


STAIR_TYPES = ("stairs", "stairs_up")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _load_state(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"state root must be a mapping: {path}")
    return value


def _terrain_proportions(config: dict[str, Any]) -> list[tuple[str, float]]:
    terrain = config.get("env", {}).get("terrain", {})
    if not isinstance(terrain, dict) or not terrain:
        raise ValueError("config is missing env.terrain")
    result: list[tuple[str, float]] = []
    for name, section in terrain.items():
        if not isinstance(section, dict):
            continue
        result.append((str(name), max(0.0, float(section.get("proportion", 0.0)))))
    total = sum(value for _, value in result)
    if total <= 0.0:
        raise ValueError("terrain proportions must contain a positive value")
    return [(name, value / total) for name, value in result]


def _column_semantics(config: dict[str, Any], num_cols: int) -> list[str]:
    proportions = _terrain_proportions(config)
    cumulative: list[float] = []
    running = 0.0
    for _, proportion in proportions:
        running += proportion
        cumulative.append(running)

    semantics: list[str] = []
    for column in range(num_cols):
        probe = column / num_cols + 1.0e-3
        index = next(
            (i for i, upper in enumerate(cumulative) if probe < upper),
            len(proportions) - 1,
        )
        semantics.append(proportions[index][0])
    return semantics


def _env_ids_for_type(
    terrain_types: list[int], column_semantics: list[str], terrain_name: str
) -> list[int]:
    return [
        env_id
        for env_id, column in enumerate(terrain_types)
        if column_semantics[column] == terrain_name
    ]


def _replay_partition(
    env_ids: list[int], *, fraction: float, level_min: int, level_max: int
) -> tuple[list[int], dict[int, int]]:
    if not env_ids or fraction <= 0.0:
        return [], {}
    ratio = min(max(float(fraction), 0.0), 1.0)
    count = min(len(env_ids), max(1, int(round(len(env_ids) * ratio))))
    positions = [
        min(len(env_ids) - 1, math.floor((index + 0.5) * len(env_ids) / count))
        for index in range(count)
    ]
    chosen = [env_ids[position] for position in positions]
    lo = max(0, int(level_min))
    hi = max(lo, int(level_max))
    span = hi - lo + 1
    return chosen, {env_id: lo + index % span for index, env_id in enumerate(chosen)}


def _replay_config(config: dict[str, Any]) -> tuple[float, int, int]:
    curriculum = config.get("env", {}).get("curriculum", {})
    if not isinstance(curriculum, dict):
        curriculum = {}
    return (
        float(curriculum.get("terrain_replay_fraction", 0.0)),
        int(curriculum.get("terrain_replay_level_min", 3)),
        int(curriculum.get("terrain_replay_level_max", 5)),
    )


def _quantile_transfer(
    source_ids: list[int],
    target_ids: list[int],
    *,
    source_levels: list[int],
    source_peaks: list[int],
    source_streaks: list[int],
    target_levels: list[int],
    target_peaks: list[int],
    target_streaks: list[int],
) -> None:
    if not target_ids:
        return
    if not source_ids:
        raise ValueError("cannot populate a target terrain type with no source samples")
    ordered_source = sorted(
        source_ids,
        key=lambda env_id: (
            source_levels[env_id],
            source_peaks[env_id],
            source_streaks[env_id],
            env_id,
        ),
    )
    for index, target_id in enumerate(sorted(target_ids)):
        source_position = min(
            len(ordered_source) - 1,
            math.floor((index + 0.5) * len(ordered_source) / len(target_ids)),
        )
        source_id = ordered_source[source_position]
        target_levels[target_id] = source_levels[source_id]
        target_peaks[target_id] = source_peaks[source_id]
        target_streaks[target_id] = source_streaks[source_id]


def _mean(values: list[int], ids: list[int]) -> float:
    return sum(values[index] for index in ids) / len(ids) if ids else 0.0


def remap_state(
    source_state: dict[str, Any],
    source_config: dict[str, Any],
    target_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    terrain_types = [int(value) for value in source_state.get("terrain_types", [])]
    source_levels = [int(value) for value in source_state.get("terrain_levels", [])]
    if not terrain_types or len(source_levels) != len(terrain_types):
        raise ValueError("terrain_types and terrain_levels must be non-empty and equal length")
    source_peaks = [int(value) for value in source_state.get("terrain_level_peak", source_levels)]
    source_streaks = [
        int(value)
        for value in source_state.get("terrain_move_down_streak", [0] * len(source_levels))
    ]
    if len(source_peaks) != len(source_levels) or len(source_streaks) != len(source_levels):
        raise ValueError("curriculum state arrays must have equal length")

    num_cols = max(terrain_types) + 1
    source_semantics = _column_semantics(source_config, num_cols)
    target_semantics = _column_semantics(target_config, num_cols)
    target_levels = [0] * len(source_levels)
    target_peaks = [0] * len(source_levels)
    target_streaks = [0] * len(source_levels)

    source_replay_fraction, source_replay_min, source_replay_max = _replay_config(source_config)
    target_replay_fraction, target_replay_min, target_replay_max = _replay_config(target_config)
    terrain_names = list(dict.fromkeys(source_semantics + target_semantics))
    report: dict[str, Any] = {
        "num_envs": len(terrain_types),
        "num_cols": num_cols,
        "source_columns": source_semantics,
        "target_columns": target_semantics,
        "types": {},
    }

    for terrain_name in terrain_names:
        source_ids = _env_ids_for_type(terrain_types, source_semantics, terrain_name)
        target_ids = _env_ids_for_type(terrain_types, target_semantics, terrain_name)
        type_report: dict[str, Any] = {
            "source_count": len(source_ids),
            "target_count": len(target_ids),
            "source_mean": _mean(source_levels, source_ids),
        }
        if not target_ids:
            report["types"][terrain_name] = type_report
            continue

        if terrain_name in STAIR_TYPES:
            source_replay, _ = _replay_partition(
                source_ids,
                fraction=source_replay_fraction,
                level_min=source_replay_min,
                level_max=source_replay_max,
            )
            target_replay, target_fixed_levels = _replay_partition(
                target_ids,
                fraction=target_replay_fraction,
                level_min=target_replay_min,
                level_max=target_replay_max,
            )
            source_replay_set = set(source_replay)
            target_replay_set = set(target_replay)
            source_frontier = [env_id for env_id in source_ids if env_id not in source_replay_set]
            target_frontier = [env_id for env_id in target_ids if env_id not in target_replay_set]
            _quantile_transfer(
                source_frontier or source_ids,
                target_frontier,
                source_levels=source_levels,
                source_peaks=source_peaks,
                source_streaks=source_streaks,
                target_levels=target_levels,
                target_peaks=target_peaks,
                target_streaks=target_streaks,
            )
            for target_id in target_replay:
                fixed_level = target_fixed_levels[target_id]
                target_levels[target_id] = fixed_level
                target_peaks[target_id] = fixed_level
                target_streaks[target_id] = 0
            type_report.update(
                {
                    "source_replay_count": len(source_replay),
                    "source_frontier_count": len(source_frontier),
                    "source_frontier_mean": _mean(source_levels, source_frontier),
                    "target_replay_count": len(target_replay),
                    "target_frontier_count": len(target_frontier),
                    "target_replay_mean": _mean(target_levels, target_replay),
                    "target_frontier_mean": _mean(target_levels, target_frontier),
                }
            )
        else:
            _quantile_transfer(
                source_ids,
                target_ids,
                source_levels=source_levels,
                source_peaks=source_peaks,
                source_streaks=source_streaks,
                target_levels=target_levels,
                target_peaks=target_peaks,
                target_streaks=target_streaks,
            )

        type_report["target_mean"] = _mean(target_levels, target_ids)
        report["types"][terrain_name] = type_report

    output = dict(source_state)
    output["terrain_levels"] = target_levels
    output["terrain_level_peak"] = target_peaks
    output["terrain_move_down_streak"] = target_streaks
    output["terrain_layout_remap"] = {
        "method": "semantic_quantile_with_stair_replay_frontier",
        "source_columns": source_semantics,
        "target_columns": target_semantics,
    }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    output, report = remap_state(
        _load_state(args.source_state),
        _load_yaml(args.source_config),
        _load_yaml(args.target_config),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, separators=(",", ":")), encoding="utf-8")
    report_text = json.dumps(report, ensure_ascii=True, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report_text, encoding="utf-8")
    print(report_text, end="")


if __name__ == "__main__":
    main()
