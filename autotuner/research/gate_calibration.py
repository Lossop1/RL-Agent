"""Deterministic curriculum-gate definitions and trace reduction.

Gate calibration must use the configuration that a run actually loaded.  This
module deliberately contains only stable semantics: where a configured gate is
read from, which runtime telemetry quantity it compares, and the comparison
direction.  Threshold values always come from ``effective_config.yaml``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Iterable, Mapping


_PHASE_GATE_RE = re.compile(r"^phase_gate_(.+)_(\d+)$")
_PROGRESS_THRESHOLDS_RE = re.compile(r"^phase_progress_thresholds_(\d+)$")


@dataclass(frozen=True)
class GateDefinition:
    gate_id: str
    config_path: str
    configured_threshold: float
    comparison: str
    metric_field: str
    purpose: str = "curriculum_progress"
    phase: int | None = None
    phase_min: int | None = None
    dr_level: int | None = None
    active: bool = True
    qualification_rule: str = "distinct runtime gate-evaluation window"
    note: str = ""

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


# ``metric_field`` names are keys in the runtime telemetry ``curriculum``
# payload.  These mappings mirror the comparisons in
# 产品环境负责产出门控遥测；本层不复制任何任务阈值。
_PHASE_METRICS: dict[str, tuple[str, str]] = {
    "prog": ("phase_gate_progress_value", "gte"),
    "slip": ("phase_gate_slip_value", "lte"),
    "diag": ("phase_gate_diagonal_value", "gte"),
    "duty_target": ("phase_gate_duty_target_value", "gte"),
    "duty_symmetry": ("phase_gate_duty_symmetry_value", "gte"),
    "period": ("phase_gate_period_value", "gte"),
    "yaw_gait": ("phase_gate_yaw_gait_value", "gte"),
    "duty_valid": ("phase_gate_duty_valid_value", "gte"),
    "execution": ("phase_gate_execution_value", "gte"),
    "terminal_rate": ("phase_gate_terminal_rate_value", "lte"),
    "air": ("phase_gate_air_value", "gte"),
    "flat_tilt_p95": ("phase_gate_flat_tilt_p95_value", "lte"),
    "flat_wxy": ("phase_gate_flat_wxy_value", "lte"),
    "flat_height_error_p95": ("phase_gate_flat_height_error_p95_value", "lte"),
    "flat_touchdown_vz_p95": ("phase_gate_flat_touchdown_vz_p95_value", "lte"),
    "flat_slip_high": ("phase_gate_flat_slip_high_value", "lte"),
    "flat_trajectory_p95": ("phase_gate_flat_trajectory_p95_value", "lte"),
    "flat_false_terrain_response": ("phase_gate_flat_false_terrain_response_value", "lte"),
    "terrain": ("phase_gate_terrain_level_value", "gte"),
    "discrete_terrain": ("phase_gate_terrain_discrete_value", "gte"),
    "boxes": ("phase_gate_terrain_boxes_value", "gte"),
    "stairs": ("phase_gate_terrain_stairs_down_value", "gte"),
    "stairs_up": ("phase_gate_terrain_stairs_up_value", "gte"),
    "boxes_success": ("phase_gate_terrain_boxes_success_value", "gte"),
    "stairs_down_success": ("phase_gate_terrain_stairs_down_success_value", "gte"),
    "stairs_up_success": ("phase_gate_terrain_stairs_up_success_value", "gte"),
    "boxes_collapse": ("phase_gate_terrain_boxes_collapse_value", "lte"),
    "stairs_down_collapse": ("phase_gate_terrain_stairs_down_collapse_value", "lte"),
    "stairs_up_collapse": ("phase_gate_terrain_stairs_up_collapse_value", "lte"),
    "fall": ("fall_gate", "lte"),
}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _add(
    out: list[GateDefinition],
    *,
    gate_id: str,
    config_path: str,
    value: Any,
    comparison: str,
    metric_field: str,
    purpose: str = "curriculum_progress",
    phase: int | None = None,
    phase_min: int | None = None,
    dr_level: int | None = None,
    active: bool = True,
    qualification_rule: str = "distinct runtime gate-evaluation window",
    note: str = "",
) -> None:
    threshold = _number(value)
    if threshold is None:
        return
    out.append(GateDefinition(
        gate_id=gate_id,
        config_path=config_path,
        configured_threshold=threshold,
        comparison=comparison,
        metric_field=metric_field,
        purpose=purpose,
        phase=phase,
        phase_min=phase_min,
        dr_level=dr_level,
        active=active,
        qualification_rule=qualification_rule,
        note=note,
    ))


def extract_gate_definitions(config: Mapping[str, Any]) -> list[GateDefinition]:
    """Extract measurable gates from a parsed effective configuration.

    Scheduling constants such as ``phase_intervals`` and ``gate_intervals`` are
    intentionally excluded: they control dwell/ramp timing rather than compare
    a policy metric against a capability threshold.
    """
    env = config.get("env") if isinstance(config.get("env"), Mapping) else {}
    curriculum = env.get("curriculum") if isinstance(env.get("curriculum"), Mapping) else {}
    dr = env.get("domain_randomization") if isinstance(env.get("domain_randomization"), Mapping) else {}
    terrain_start_phase = int(_number(curriculum.get("terrain_start_phase")) or 0)
    dr_start_phase = int(_number(curriculum.get("dr_start_phase")) or terrain_start_phase)
    out: list[GateDefinition] = []

    for key, value in curriculum.items():
        progress_match = _PROGRESS_THRESHOLDS_RE.match(str(key))
        if progress_match and isinstance(value, Mapping):
            phase = int(progress_match.group(1))
            configured_dirs = curriculum.get(f"phase_progress_dirs_{phase}")
            active_dirs = {
                str(item) for item in configured_dirs
            } if isinstance(configured_dirs, (list, tuple, set)) else set(value)
            for direction, threshold in value.items():
                name = str(direction)
                _add(
                    out,
                    gate_id=f"env.curriculum.{key}.{name}",
                    config_path=f"env.curriculum.{key}.{name}",
                    value=threshold,
                    comparison="gte",
                    metric_field=f"progress_{name}",
                    phase=phase,
                    active=name in active_dirs,
                    qualification_rule=f"phase={phase}; direction={name} active; distinct gate-evaluation window",
                )
            continue

        phase_match = _PHASE_GATE_RE.match(str(key))
        if not phase_match:
            continue
        metric_name, phase_text = phase_match.groups()
        phase = int(phase_text)
        metric = _PHASE_METRICS.get(metric_name)
        if metric is not None:
            _add(
                out,
                gate_id=f"env.curriculum.{key}",
                config_path=f"env.curriculum.{key}",
                value=value,
                comparison=metric[1],
                metric_field=metric[0],
                phase=phase,
                qualification_rule=f"phase={phase}; distinct gate-evaluation window",
            )
            continue
        # ``duty`` is only a fallback used when the explicit duty-target and
        # symmetry gates are absent.  Transition failure is currently measured
        # for diagnosis but deliberately does not block phase advancement.
        if metric_name in {"duty", "transition_fail"}:
            _add(
                out,
                gate_id=f"env.curriculum.{key}",
                config_path=f"env.curriculum.{key}",
                value=value,
                comparison="gte" if metric_name == "duty" else "lte",
                metric_field="" if metric_name == "duty" else "transition_failure_gate",
                phase=phase,
                active=False,
                note=(
                    "fallback shadowed by explicit duty_target/duty_symmetry gates"
                    if metric_name == "duty"
                    else "runtime records this diagnostic, but transition_safety_ok is not an active phase gate"
                ),
            )

    for key, metric_field, comparison in (
        ("regress_fall", "fall_gate", "lte"),
        ("regress_prog", "progress_gate", "gte"),
        ("regress_execution", "execution_gate", "gte"),
        ("dr_gate_execution", "execution_gate", "gte"),
        ("terrain_gate_slip_high", "terrain_health_slip_high_value", "lte"),
    ):
        if key in curriculum:
            _add(
                out,
                gate_id=f"env.curriculum.{key}",
                config_path=f"env.curriculum.{key}",
                value=curriculum[key],
                comparison=comparison,
                metric_field=metric_field,
                purpose="dr_progress" if key.startswith("dr_gate_") else "curriculum_progress",
                phase_min=(
                    dr_start_phase if key.startswith("dr_gate_")
                    else terrain_start_phase if key == "terrain_gate_slip_high"
                    else None
                ),
                qualification_rule="distinct gate-evaluation window with the corresponding curriculum path active",
            )

    # These thresholds are currently diagnostic only: the environment computes
    # them but sets dr_transition_ok=True.  Preserve that fact instead of
    # pretending they calibrate an active mathematical gate.
    for key in (
        "dr_gate_transition_stop_fail_final",
        "dr_gate_transition_reverse_fail_final",
        "dr_gate_transition_axis_fail_final",
        "dr_gate_transition_handoff_fail_final",
    ):
        if key in curriculum:
            _add(
                out,
                gate_id=f"env.curriculum.{key}",
                config_path=f"env.curriculum.{key}",
                value=curriculum[key],
                comparison="lte",
                metric_field="",
                purpose="dr_progress",
                active=False,
                note="runtime records transition diagnostics, but they do not currently block DR advancement",
            )

    if "unlock_terrain" in dr:
        _add(
            out,
            gate_id="env.domain_randomization.unlock_terrain",
            config_path="env.domain_randomization.unlock_terrain",
            value=dr["unlock_terrain"],
            comparison="gte",
            metric_field="dr_gate_terrain_level",
            purpose="dr_progress",
            phase_min=dr_start_phase,
            qualification_rule="DR not complete; distinct gate-evaluation window",
        )
    for key, level in (("gate_progress", 0), ("gate_progress_l2", 1), ("gate_progress_l3", 2)):
        if key in dr:
            _add(
                out,
                gate_id=f"env.domain_randomization.{key}",
                config_path=f"env.domain_randomization.{key}",
                value=dr[key],
                comparison="gte",
                metric_field="progress_gate",
                purpose="dr_progress",
                phase_min=dr_start_phase,
                dr_level=level,
                qualification_rule=f"dr_level={level}; distinct gate-evaluation window",
            )

    # Legacy HistoryActorPolicy payloads use these four quality gates at every
    # non-terminal DR level.  They remain part of the runtime's mathematical
    # DR transition even though newer payloads moved quality checks elsewhere.
    for key, metric_field, comparison in (
        ("gate_gait_min", "dr_gate_gait_value", "gte"),
        ("gate_duty_min", "dr_gate_duty_value", "gte"),
        ("gate_slip_max", "dr_gate_slip_value", "lte"),
        ("gate_tilt_deg_max", "dr_gate_tilt_deg_value", "lte"),
    ):
        if key in dr:
            _add(
                out,
                gate_id=f"env.domain_randomization.{key}",
                config_path=f"env.domain_randomization.{key}",
                value=dr[key],
                comparison=comparison,
                metric_field=metric_field,
                purpose="dr_progress",
                phase_min=dr_start_phase,
                qualification_rule=(
                    "phase at or beyond DR start; DR not complete; "
                    "distinct gate-evaluation window"
                ),
            )

    return sorted(out, key=lambda item: item.gate_id)


def phase_index(value: Any) -> int | None:
    if isinstance(value, str):
        match = re.search(r"(\d+)$", value.strip())
        return int(match.group(1)) if match else None
    number = _number(value)
    return int(number) if number is not None else None


def gate_passes(value: float, threshold: float, comparison: str) -> bool:
    if comparison == "gte":
        return value >= threshold
    if comparison == "lte":
        return value <= threshold
    return False


def gate_trace_rows(
    telemetry_rows: Iterable[Mapping[str, Any]],
    definitions: Iterable[GateDefinition],
) -> list[dict[str, Any]]:
    """Reduce telemetry to one sample per distinct runtime gate evaluation."""
    definitions = [item for item in definitions if item.active and item.metric_field]
    seen: set[tuple[str, int]] = set()
    out: list[dict[str, Any]] = []
    for row in telemetry_rows:
        curriculum = row.get("curriculum") if isinstance(row.get("curriculum"), Mapping) else {}
        current_phase = phase_index(curriculum.get("phase"))
        current_dr = phase_index(curriculum.get("dr_level"))
        eval_step = phase_index(curriculum.get("phase_gate_eval_step"))
        row_step = phase_index(row.get("step"))
        if eval_step is None or eval_step <= 0:
            continue
        sample_step = row_step if row_step is not None else eval_step
        active_dirs = {
            item.strip()
            for item in str(curriculum.get("active_dirs") or "").split(",")
            if item.strip()
        }
        for definition in definitions:
            if definition.phase is not None and definition.phase != current_phase:
                continue
            if definition.phase_min is not None and (
                current_phase is None or current_phase < definition.phase_min
            ):
                continue
            if definition.dr_level is not None and definition.dr_level != current_dr:
                continue
            if ".phase_progress_thresholds_" in definition.gate_id:
                direction = definition.gate_id.rsplit(".", 1)[-1]
                if direction not in active_dirs:
                    continue
            value = _number(curriculum.get(definition.metric_field))
            if value is None:
                continue
            token = (definition.gate_id, eval_step)
            if token in seen:
                continue
            seen.add(token)
            sample_id = f"{definition.gate_id}@{eval_step}"
            out.append({
                "sample_id": sample_id,
                "gate_id": definition.gate_id,
                "step": sample_step,
                "gate_eval_step": eval_step,
                "phase": current_phase,
                "dr_level": current_dr,
                "metric_field": definition.metric_field,
                "value": value,
                "threshold": definition.configured_threshold,
                "comparison": definition.comparison,
                "eligible": True,
                "passed": gate_passes(value, definition.configured_threshold, definition.comparison),
            })
    return out


def quantiles(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(float(item) for item in values if math.isfinite(float(item)))
    if not ordered:
        return {}

    def pick(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
        return ordered[index]

    return {
        "min": ordered[0],
        "p05": pick(0.05),
        "p50": pick(0.5),
        "p95": pick(0.95),
        "max": ordered[-1],
    }
