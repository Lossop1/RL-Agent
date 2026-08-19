"""基于诊断 record.csv 的楼梯完整换层验收。"""
from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


LEGS = ("FL", "FR", "RL", "RR")


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _median(values: Iterable[Any]) -> float | None:
    clean = [value for item in values if (value := _float(item)) is not None]
    return statistics.median(clean) if clean else None


def _percentile(values: Iterable[Any], q: float) -> float | None:
    clean = sorted(value for item in values if (value := _float(item)) is not None)
    if not clean:
        return None
    pos = min(max(float(q), 0.0), 1.0) * (len(clean) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    return clean[lo] * (hi - pos) + clean[hi] * (pos - lo)


def _terrain_specs(suite: dict[str, Any]) -> list[dict[str, Any]]:
    terrains = [item for item in suite.get("terrains", []) if isinstance(item, dict)]
    dr_count = max(1, len([item for item in suite.get("dr_cases", []) if isinstance(item, dict)]))
    return [terrain for terrain in terrains for _ in range(dr_count)]


def _direction(terrain_type: str) -> int:
    return -1 if terrain_type in {"stairs_down", "stairs"} else 1


def _first_contact_time(
    rows: list[dict[str, str]],
    leg: str,
    baseline: float,
    direction: int,
    threshold: float,
) -> float | None:
    for row in rows:
        contact = _float(row.get(f"foot_{leg}_contact")) or 0.0
        height = _float(row.get(f"foot_{leg}_terrain_height"))
        if contact > 0.5 and height is not None and direction * (height - baseline) >= threshold:
            return _float(row.get("time"))
    return None


def _case_result(
    case_id: int,
    rows: list[dict[str, str]],
    terrain: dict[str, Any],
    criteria: dict[str, Any],
) -> dict[str, Any]:
    terrain_type = str(terrain.get("type") or rows[0].get("terrain_type_requested") or "")
    params = terrain.get("params") if isinstance(terrain.get("params"), dict) else {}
    step_height = _float(params.get("step_height")) or 0.0
    if step_height > 1.0:
        step_height /= 100.0
    direction = _direction(terrain_type)
    moving = [row for row in rows if str(row.get("cmd_target_mode") or row.get("cmd_mode")) == "forward"]
    if not moving:
        return {
            "case_id": case_id,
            "terrain": terrain_type,
            "step_height_m": step_height,
            "success": False,
            "failure_reasons": ["missing_forward_segment"],
        }

    moving.sort(key=lambda row: _float(row.get("time")) or 0.0)
    sample_dt = _median(
        (b - a for a, b in zip(
            [_float(row.get("time")) for row in moving[:-1]],
            [_float(row.get("time")) for row in moving[1:]],
        ) if a is not None and b is not None)
    ) or 0.02
    baseline_count = max(5, int(round(0.4 / max(sample_dt, 1e-6))))
    final_count = max(5, int(round(1.0 / max(sample_dt, 1e-6))))
    baseline_rows = moving[:baseline_count]
    final_rows = moving[-final_count:]
    base_z_start = _median(row.get("base_pos_w_z") for row in baseline_rows)
    base_z_final = _median(row.get("base_pos_w_z") for row in final_rows)
    base_delta = (
        direction * (base_z_final - base_z_start)
        if base_z_start is not None and base_z_final is not None
        else None
    )
    threshold = max(0.01, 0.8 * step_height)

    leg_results: dict[str, dict[str, Any]] = {}
    first_times: dict[str, float | None] = {}
    for leg in LEGS:
        baseline = _median(row.get(f"foot_{leg}_terrain_height") for row in baseline_rows)
        contacted_heights = [
            row.get(f"foot_{leg}_terrain_height")
            for row in final_rows
            if (_float(row.get(f"foot_{leg}_contact")) or 0.0) > 0.5
        ]
        if baseline is None:
            final_delta = None
        elif direction > 0:
            target = max((v for item in contacted_heights if (v := _float(item)) is not None), default=None)
            final_delta = None if target is None else target - baseline
        else:
            target = min((v for item in contacted_heights if (v := _float(item)) is not None), default=None)
            final_delta = None if target is None else baseline - target
        complete = final_delta is not None and final_delta >= threshold
        first_time = None if baseline is None else _first_contact_time(moving, leg, baseline, direction, threshold)
        first_times[leg] = first_time
        leg_results[leg] = {
            "support_height_delta_m": final_delta,
            "completed_layer_change": complete,
            "first_target_contact_s": first_time,
        }

    all_feet_complete = all(item["completed_layer_change"] for item in leg_results.values())
    body_complete = base_delta is not None and base_delta >= threshold
    would_terminate_fraction = sum(
        1 for row in final_rows if (_float(row.get("would_terminate")) or 0.0) > 0.5
    ) / max(1, len(final_rows))
    min_height = min(
        (value for row in final_rows if (value := _float(row.get("base_height_local"))) is not None),
        default=None,
    )
    min_height_required = _float(criteria.get("min_height_m")) or 0.40
    stable = would_terminate_fraction == 0.0 and min_height is not None and min_height >= min_height_required

    torque_columns = [f"torque_utilization_{index}" for index in range(12)]
    torque_values = [_float(row.get(column)) for row in moving for column in torque_columns]
    torque_p95 = _percentile(torque_values, 0.95)
    torque_max = max((value for value in torque_values if value is not None), default=None)
    saturated_frames = sum(
        1
        for row in moving
        if any((_float(row.get(column)) or 0.0) >= 0.999 for column in torque_columns)
    )
    torque_saturation_fraction = saturated_frames / max(1, len(moving))
    slip_values = [
        _float(row.get(f"foot_{leg}_stance_slip_xy"))
        for row in moving
        for leg in LEGS
    ]
    slip_p95 = _percentile(slip_values, 0.95)

    front_times = [first_times[leg] for leg in ("FL", "FR") if first_times[leg] is not None]
    rear_times = [first_times[leg] for leg in ("RL", "RR") if first_times[leg] is not None]
    transfer_latency = (
        max(rear_times) - max(front_times)
        if len(front_times) == 2 and len(rear_times) == 2
        else None
    )
    failure_reasons: list[str] = []
    if not body_complete:
        failure_reasons.append("base_did_not_change_layer")
    if not all_feet_complete:
        missing = [leg for leg, item in leg_results.items() if not item["completed_layer_change"]]
        failure_reasons.append("feet_did_not_change_layer:" + ",".join(missing))
    if not stable:
        failure_reasons.append("final_state_unstable")
    if torque_saturation_fraction > 0.05:
        failure_reasons.append("torque_saturation_over_5pct")

    return {
        "case_id": case_id,
        "terrain": terrain_type,
        "direction": "down" if direction < 0 else "up",
        "step_height_m": step_height,
        "success": body_complete and all_feet_complete and stable,
        "base_height_delta_m": base_delta,
        "required_layer_delta_m": threshold,
        "all_feet_completed_layer_change": all_feet_complete,
        "rear_support_transfer_latency_s": transfer_latency,
        "final_stable": stable,
        "would_terminate_fraction_final": would_terminate_fraction,
        "min_base_height_local_final_m": min_height,
        "torque_utilization_p95": torque_p95,
        "torque_utilization_max": torque_max,
        "torque_saturation_frame_fraction": torque_saturation_fraction,
        "stance_slip_p95_mps": slip_p95,
        "legs": leg_results,
        "failure_reasons": failure_reasons,
    }

def compute_stair_validation(record_path: str | Path, suite: dict[str, Any]) -> dict[str, Any]:
    """计算每个楼梯case的完整换层结果，并返回可写入metrics.json的字典。"""
    with Path(record_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        case_id = int(_float(row.get("case_id")) or 0)
        grouped[case_id].append(row)
    specs = _terrain_specs(suite)
    criteria = suite.get("criteria") if isinstance(suite.get("criteria"), dict) else {}
    cases = []
    for case_id, case_rows in sorted(grouped.items()):
        terrain = specs[case_id] if case_id < len(specs) else {
            "type": case_rows[0].get("terrain_type_requested", ""),
            "params": {},
        }
        if str(terrain.get("type") or "") not in {"stairs", "stairs_up", "stairs_down"}:
            continue
        cases.append(_case_result(case_id, case_rows, terrain, criteria))
    passed = sum(1 for case in cases if case.get("success"))
    return {
        "definition": (
            "成功要求机身永久跨越至少0.8倍台阶高度、四脚均在目标层形成接触，"
            "并在末1秒保持高度安全且无would_terminate。"
        ),
        "cases": cases,
        "summary": {
            "cases": len(cases),
            "passed": passed,
            "pass_rate": passed / len(cases) if cases else None,
        },
    }
