"""Training telemetry parsing and shaping.

The preferred remote format is JSONL (`*.telemetry.jsonl`). The console also
parses the currently deployed human log fallback, for example:

  [TPREW] step 42500 rew 1.108 lin_err 0.278 speed 0.206 gate 0.97 track 0.187 stand 0.000
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from .schemas import (
    DataProvenanceItem,
    MetricSnapshotGroup,
    MetricSnapshotItem,
    RunSnapshot,
    SnapshotBlocker,
    TrainingTelemetry,
    TrainingTelemetryPoint,
)


_TPREW_RE = re.compile(r"\[TPREW\]\s*(?P<body>.*)$")
_TPSTAT_RE = re.compile(r"\[TPSTAT\]\s*(?P<body>.*)$")
_TPCURR_RE = re.compile(r"\[TPCURR\]\s*(?P<body>.*)$")
_TPCKPT_RE = re.compile(r"\[TPCKPT\]\s*(?P<body>.*)$")
_TPCMD_RE = re.compile(r"\[TPCMD\]\s*(?P<body>.*)$")
_TPPEN_RE = re.compile(r"\[TPPEN\]\s*(?P<body>.*)$")
_TPGATE_RE = re.compile(r"\[TPGATE\]\s*(?P<body>.*)$")
_ENV_PROGRESS_RE = re.compile(
    r"fwd/back/lat/yaw=(?P<fwd>[-+0-9.]+)/(?P<back>[-+0-9.]+)/(?P<lat>[-+0-9.]+)/(?P<yaw>[-+0-9.]+)"
)
_PROGRESS_RE = re.compile(
    r"(?P<pct>\d+)%\|.*?\|\s*(?P<iter>\d+)/(?P<total>\d+)\s*\[(?P<elapsed>[^<\]]+)<(?P<eta>[^,\]]+),\s*(?P<fps>[-+0-9.]+)it/s\]"
)

_FALLBACK_GATE_THRESHOLDS = {
    "progress": {
        0: 0.70,
        1: 0.70,
        2: 0.70,
        3: 0.70,
        4: 0.65,
        5: 0.55,
        6: 0.70,
        7: 0.65,
        8: 0.70,
    },
    "gait": 0.85,
    "slip": 0.20,
    "diagonal": 0.82,
    "duty": 0.82,
    "fall": 0.05,
}


def _parse_kv(body: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    parts = body.strip().split()
    i = 0
    while i < len(parts):
        token = parts[i]
        if "=" in token:
            key, value = token.split("=", 1)
        elif i + 1 < len(parts):
            key, value = token, parts[i + 1]
            i += 1
        else:
            i += 1
            continue
        key = key.strip().lower()
        value = value.strip().rstrip(",")
        if not key:
            i += 1
            continue
        try:
            if re.fullmatch(r"[-+]?\d+", value):
                values[key] = int(value)
            else:
                values[key] = float(value)
        except ValueError:
            values[key] = value
        i += 1
    return values


def _point_from_json(data: dict[str, Any], raw: str = "") -> TrainingTelemetryPoint | None:
    if not isinstance(data, dict):
        return None
    if data.get("type") not in {None, "train_tick", "tp_tick", "telemetry"}:
        return None
    reward = data.get("reward") if isinstance(data.get("reward"), dict) else {}
    curriculum = data.get("curriculum") if isinstance(data.get("curriculum"), dict) else {}
    health = data.get("health") if isinstance(data.get("health"), dict) else {}
    command = data.get("command") if isinstance(data.get("command"), dict) else {}
    counters = data.get("counters") if isinstance(data.get("counters"), dict) else {}
    paths = data.get("paths") if isinstance(data.get("paths"), dict) else {}
    step = int(data.get("step") or data.get("iter") or 0)
    if step <= 0 and not reward and not curriculum:
        return None
    return TrainingTelemetryPoint(
        step=step,
        total_steps=int(data["total_steps"]) if data.get("total_steps") is not None else None,
        ts=float(data.get("ts") or data.get("time") or time.time()),
        elapsed=str(data.get("elapsed") or ""),
        eta=str(data.get("eta") or ""),
        fps=float(data["fps"]) if data.get("fps") is not None else None,
        reward={str(k): float(v) for k, v in reward.items() if isinstance(v, (int, float))},
        curriculum=dict(curriculum),
        health={str(k): float(v) for k, v in health.items() if isinstance(v, (int, float))},
        command={str(k): float(v) for k, v in command.items() if isinstance(v, (int, float))},
        counters={str(k): float(v) for k, v in counters.items() if isinstance(v, (int, float))},
        paths={str(k): str(v) for k, v in paths.items() if v is not None},
        checkpoint=str(data.get("checkpoint") or ""),
        raw=raw,
    )


def _latest_progress_by_dir_from_log(text: str) -> dict[str, float]:
    """Parse the last human ENV progress line as a compatibility fallback.

    New payloads write progress_fwd/back/lat/yaw into JSONL. Existing live runs
    may only have the richer per-direction values in console.log, so this
    helper augments snapshots without making console.log the primary source.
    """
    found: dict[str, float] = {}
    for line in (text or "").splitlines():
        match = _ENV_PROGRESS_RE.search(line)
        if not match:
            continue
        try:
            found = {key: float(match.group(key)) for key in ("fwd", "back", "lat", "yaw")}
        except ValueError:
            continue
    return found


def _augment_points_from_side_log(points: list[TrainingTelemetryPoint], text: str) -> None:
    if not points:
        return
    progress = _latest_progress_by_dir_from_log(text)
    if not progress:
        return
    latest = points[-1]
    for key, value in progress.items():
        latest.command.setdefault(f"progress_{key}", value)
        latest.curriculum.setdefault(f"progress_{key}", value)
    active_text = str(latest.curriculum.get("active_dirs") or "")
    active = [item.strip() for item in active_text.split(",") if item.strip()] or list(progress)
    active_values = {name: progress[name] for name in active if name in progress}
    if active_values:
        lagging_dir, lagging_value = min(active_values.items(), key=lambda item: item[1])
        latest.command.setdefault("progress_min_active", lagging_value)
        latest.curriculum.setdefault("progress_min_active", lagging_value)
        latest.curriculum.setdefault("progress_lagging_dir", lagging_dir)
        latest.curriculum.setdefault("progress_gate", lagging_value)


def parse_telemetry_jsonl(text: str) -> list[TrainingTelemetryPoint]:
    points: list[TrainingTelemetryPoint] = []
    for line in (text or "").splitlines():
        raw = line.strip()
        if not raw:
            continue
        # A single malformed line (bad JSON OR a non-numeric value in a numeric field, e.g.
        # {"step":"N/A"}) must be DROPPED, not raised: the caller treats any exception here as a
        # remote-SSH failure and puts a healthy box into a self-inflicted cooldown outage. Guard the
        # point construction too, matching the JSONDecodeError intent already present.
        try:
            data = json.loads(raw)
            point = _point_from_json(data, raw=raw)
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
        if point is not None:
            points.append(point)
    return points


def parse_training_log(text: str) -> list[TrainingTelemetryPoint]:
    by_step: dict[int, TrainingTelemetryPoint] = {}
    last_step = 0
    for line in (text or "").splitlines():
        raw = line.rstrip()
        if not raw:
            continue
        progress = _PROGRESS_RE.search(raw)
        progress_values: dict[str, Any] = {}
        if progress:
            progress_values = {
                "step": int(progress.group("iter")),
                "total_steps": int(progress.group("total")),
                "elapsed": progress.group("elapsed"),
                "eta": progress.group("eta"),
                "fps": float(progress.group("fps")),
            }
            last_step = int(progress.group("iter"))
        match = _TPREW_RE.search(raw)
        if match:
            values = _parse_kv(match.group("body"))
            step = int(values.get("step") or progress_values.get("step") or last_step or 0)
            reward: dict[str, float] = {}
            for key, value in values.items():
                if key != "step" and isinstance(value, (int, float)):
                    reward[key] = float(value)
            for source, target in [
                ("rew", "total"),
                ("total", "total"),
                ("lin_err", "lin_err"),
                ("speed", "speed"),
                ("gate", "gait"),
                ("gait", "gait"),
                ("track", "track"),
                ("stand", "stand"),
                ("clearance", "clearance"),
                ("slip", "slip"),
                ("torque", "torque"),
                ("base_h", "base_h"),
                ("upright", "upright"),
                ("offaxis", "offaxis"),
                ("overshoot", "overshoot"),
                ("torque_sat", "torque_sat"),
                ("height_low", "height_low"),
                ("h_collapse", "height_collapse"),
                ("tilt_collapse", "tilt_collapse"),
            ]:
                if isinstance(values.get(source), (int, float)):
                    reward[target] = float(values[source])
            point = by_step.get(step) or TrainingTelemetryPoint(step=step, ts=time.time())
            point.reward.update(reward)
            point.raw = raw
            if progress_values:
                point.total_steps = progress_values.get("total_steps")
                point.elapsed = str(progress_values.get("elapsed") or "")
                point.eta = str(progress_values.get("eta") or "")
                point.fps = progress_values.get("fps")
            by_step[step] = point
            last_step = step
            continue
        for regex, section in [
            (_TPSTAT_RE, "status"),
            (_TPCMD_RE, "command"),
            (_TPPEN_RE, "penalty"),
            (_TPGATE_RE, "gate"),
            (_TPCURR_RE, "curriculum"),
            (_TPCKPT_RE, "checkpoint"),
        ]:
            match = regex.search(raw)
            if not match:
                continue
            values = _parse_kv(match.group("body"))
            step = int(values.get("step") or progress_values.get("step") or last_step or 0)
            point = by_step.get(step) or TrainingTelemetryPoint(step=step, ts=time.time())
            point.raw = raw
            if section == "status":
                for key in ["phase", "terrain_mean", "terrain_max", "dr_level"]:
                    if key in values:
                        point.curriculum[key] = values[key]
                for source, target in [("falls", "fall_rate"), ("fall_rate", "fall_rate"), ("resets", "reset_rate"), ("reset_rate", "reset_rate")]:
                    if isinstance(values.get(source), (int, float)):
                        point.health[target] = float(values[source])
                if isinstance(values.get("total"), int):
                    point.total_steps = int(values["total"])
                if isinstance(values.get("fps"), (int, float)):
                    point.fps = float(values["fps"])
                if values.get("elapsed") is not None:
                    point.elapsed = str(values["elapsed"])
                if values.get("eta") is not None:
                    point.eta = str(values["eta"])
            elif section == "command":
                for key, value in values.items():
                    if key != "step" and isinstance(value, (int, float)):
                        point.command[key] = float(value)
                for short, long in [("act_vx", "actual_vx"), ("act_vy", "actual_vy"), ("act_wz", "actual_wz")]:
                    if isinstance(values.get(short), (int, float)):
                        point.command[long] = float(values[short])
                if isinstance(values.get("lin_err"), (int, float)):
                    point.reward.setdefault("lin_err", float(values["lin_err"]))
                if isinstance(values.get("speed_xy"), (int, float)):
                    point.reward.setdefault("speed", float(values["speed_xy"]))
            elif section == "penalty":
                for key, value in values.items():
                    if key != "step" and isinstance(value, (int, float)):
                        point.reward[key] = float(value)
            elif section == "gate":
                for key, value in values.items():
                    if key != "step" and isinstance(value, (int, float)):
                        point.health[key] = float(value)
                if isinstance(values.get("stable"), (int, float)):
                    point.reward.setdefault("gait", float(values["stable"]))
            elif section == "curriculum":
                for key, value in values.items():
                    if key != "step":
                        point.curriculum[key] = value
            elif section == "checkpoint":
                if values.get("path"):
                    point.checkpoint = str(values["path"])
            if progress_values:
                point.total_steps = progress_values.get("total_steps")
                point.elapsed = str(progress_values.get("elapsed") or "")
                point.eta = str(progress_values.get("eta") or "")
                point.fps = progress_values.get("fps")
            by_step[step] = point
            last_step = step
            break
    return [by_step[key] for key in sorted(by_step) if key > 0]


def summarize_telemetry(points: list[TrainingTelemetryPoint]) -> str:
    if not points:
        return "没有可解析的训练 telemetry。"
    latest = points[-1]
    parts = [f"step {latest.step}"]
    if latest.total_steps:
        parts.append(f"/ {latest.total_steps} ({latest.step / latest.total_steps:.0%})")
    if latest.eta:
        parts.append(f"ETA {latest.eta}")
    if latest.fps is not None:
        parts.append(f"{latest.fps:.2f} it/s")
    if latest.reward:
        reward_bits = []
        for key in ["total", "track", "lin_err", "speed", "gait", "stand"]:
            if key in latest.reward:
                reward_bits.append(f"{key}={latest.reward[key]:.3f}")
        if reward_bits:
            parts.append("; " + ", ".join(reward_bits))
    blocked_by = latest.curriculum.get("blocked_by")
    if blocked_by:
        parts.append(f"; blocked_by={blocked_by}")
    return " ".join(parts)


def _num(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _phase_index(phase: Any) -> int | None:
    if isinstance(phase, int):
        return phase
    if isinstance(phase, float):
        return int(phase)
    match = re.search(r"(\d+)", str(phase or ""))
    return int(match.group(1)) if match else None


def _progress_threshold(phase: Any) -> float:
    idx = _phase_index(phase)
    if idx is None:
        return 0.70
    return float(_FALLBACK_GATE_THRESHOLDS["progress"].get(idx, 0.70))


def _nested_get(data: Any, *keys: str) -> Any:
    cur = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _runtime_telemetry_interval_from_config(effective_config_text: str) -> int | None:
    if not effective_config_text.strip():
        return None
    try:
        import yaml

        raw = yaml.safe_load(effective_config_text) or {}
    except Exception:
        return None
    value = _nested_get(raw, "runtime", "telemetry_interval")
    if value is None:
        value = _nested_get(raw, "observability", "telemetry_interval_steps")
    num = _as_float(value)
    return int(num) if num and num > 0 else None


def _gate_thresholds_from_config(effective_config_text: str) -> tuple[dict[str, Any], list[str]]:
    """Read gate thresholds from the run's effective config.

    The UI must not silently invent acceptance thresholds. In real mode the
    datasource passes `effective_config.yaml`; only when that is unavailable do
    we use conservative display fallbacks and mark the snapshot note.
    """
    thresholds = {
        "progress": dict(_FALLBACK_GATE_THRESHOLDS["progress"]),
        "gait": float(_FALLBACK_GATE_THRESHOLDS["gait"]),
        "slip": float(_FALLBACK_GATE_THRESHOLDS["slip"]),
        "diagonal": float(_FALLBACK_GATE_THRESHOLDS["diagonal"]),
        "duty": float(_FALLBACK_GATE_THRESHOLDS["duty"]),
        "fall": float(_FALLBACK_GATE_THRESHOLDS["fall"]),
    }
    notes: list[str] = []
    if not effective_config_text.strip():
        notes.append("gate thresholds fallback: effective_config.yaml was not available in this telemetry response.")
        return thresholds, notes
    try:
        import yaml

        raw = yaml.safe_load(effective_config_text) or {}
    except Exception as exc:  # noqa: BLE001
        notes.append(f"gate thresholds fallback: failed to parse effective_config.yaml ({type(exc).__name__}).")
        return thresholds, notes

    curriculum = _nested_get(raw, "env", "curriculum")
    if not isinstance(curriculum, dict):
        curriculum = _nested_get(raw, "curriculum")
    if not isinstance(curriculum, dict):
        notes.append("gate thresholds fallback: effective_config.yaml has no env.curriculum section.")
        return thresholds, notes

    loaded = False
    progress: dict[int, float] = {}
    for key, value in curriculum.items():
        match = re.fullmatch(r"phase_gate_prog_(\d+)", str(key))
        if not match:
            continue
        num = _as_float(value)
        if num is not None:
            progress[int(match.group(1))] = num
    if progress:
        thresholds["progress"] = {**thresholds["progress"], **progress}
        loaded = True
    for config_key, threshold_key in [
        ("phase_gate_gait_1", "gait"),
        ("gait_curriculum_gate", "gait"),
        ("phase_gate_slip_1", "slip"),
        ("phase_gate_diag_1", "diagonal"),
        ("phase_gate_duty_1", "duty"),
        ("phase_gate_fall_2", "fall"),
    ]:
        num = _as_float(curriculum.get(config_key))
        if num is not None:
            thresholds[threshold_key] = num
            loaded = True
    if loaded:
        notes.append("gate thresholds loaded from effective_config.yaml.")
    else:
        notes.append("gate thresholds fallback: effective_config.yaml did not contain phase_gate_* values.")
    return thresholds, notes


def _progress_threshold_from(thresholds: dict[str, Any], phase: Any) -> float:
    idx = _phase_index(phase)
    progress = thresholds.get("progress") if isinstance(thresholds, dict) else None
    if isinstance(progress, dict) and idx is not None:
        return float(progress.get(idx, progress.get(str(idx), 0.70)))
    return _progress_threshold(phase)


def _dir_progress_values(latest: TrainingTelemetryPoint) -> dict[str, float]:
    """Return per-direction progress from the strongest available source.

    New payloads write these into JSONL. During a live run that was started
    before the payload change, `_augment_points_from_side_log` may fill the
    same keys from console.log. Keeping this helper centralized makes blocker
    reporting deterministic for UI and LLM.
    """
    values: dict[str, float] = {}
    for direction in ("fwd", "back", "lat", "yaw"):
        key = f"progress_{direction}"
        num = _num(latest.curriculum.get(key), latest.command.get(key))
        if num is not None:
            values[direction] = num
    return values


def _active_dir_names(active_dirs: str, progress_by_dir: dict[str, float]) -> list[str]:
    names = [
        item.strip()
        for item in re.split(r"[,/ ]+", active_dirs or "")
        if item.strip()
    ]
    normalized: list[str] = []
    aliases = {
        "forward": "fwd",
        "forwards": "fwd",
        "backward": "back",
        "backwards": "back",
        "lateral": "lat",
        "left": "lat",
        "right": "lat",
    }
    for name in names:
        key = aliases.get(name.lower(), name.lower())
        if key in progress_by_dir and key not in normalized:
            normalized.append(key)
    return normalized or list(progress_by_dir)


def _directional_progress_blocker(
    *,
    progress_by_dir: dict[str, float],
    active_dirs: str,
    progress_gate: float | None,
    progress_target: float,
    blocked_by: str,
) -> SnapshotBlocker | None:
    if not progress_by_dir:
        if progress_gate is not None and (
            blocked_by == "progress" or progress_gate < progress_target
        ):
            return SnapshotBlocker(
                key="progress_gate",
                label="progress gate",
                value=progress_gate,
                target=progress_target,
                direction="higher",
                gap=max(0.0, progress_target - progress_gate),
                severity="bad" if blocked_by == "progress" else "warn",
                source="curriculum.progress_gate",
                detail="Per-direction progress is not available in this run; deploy the latest payload to expose fwd/back/lat/yaw.",
            )
        return None
    active = _active_dir_names(active_dirs, progress_by_dir)
    active_values = {name: progress_by_dir[name] for name in active if name in progress_by_dir}
    if not active_values:
        return None
    lagging_dir, lagging_value = min(active_values.items(), key=lambda item: item[1])
    if blocked_by != "progress" and lagging_value >= progress_target:
        return None
    severity = "bad" if lagging_value < progress_target * 0.85 or blocked_by == "progress" else "warn"
    return SnapshotBlocker(
        key=f"progress_{lagging_dir}",
        label=f"{lagging_dir} progress",
        value=lagging_value,
        target=progress_target,
        direction="higher",
        gap=max(0.0, progress_target - lagging_value),
        severity=severity,  # type: ignore[arg-type]
        source="curriculum.progress_* / command.progress_*; console.log fallback for pre-deploy live runs",
        detail=f"{lagging_dir} is the weakest active direction ({', '.join(active)}). This is the phase gate bottleneck.",
    )


def _blocker_from_gap(
    gap: dict[str, Any],
    *,
    source: str,
    detail: str,
) -> SnapshotBlocker | None:
    if gap.get("tone") not in {"bad", "warn"}:
        return None
    value = _as_float(gap.get("value"))
    target = _as_float(gap.get("target"))
    direction = str(gap.get("direction") or "info")
    raw_gap = _as_float(gap.get("gap"))
    return SnapshotBlocker(
        key=str(gap.get("key") or ""),
        label=str(gap.get("label") or gap.get("key") or ""),
        value=value,
        target=target,
        direction=direction if direction in {"higher", "lower", "target", "info"} else "info",  # type: ignore[arg-type]
        gap=raw_gap,
        severity="bad" if gap.get("tone") == "bad" else "warn",
        source=source,
        detail=detail,
    )


def _tone_for(value: float | None, *, target: float | None = None, direction: str = "higher") -> str:
    if value is None or target is None:
        return "neutral"
    if direction == "lower":
        if value <= target:
            return "good"
        if value <= target * 1.25:
            return "warn"
        return "bad"
    if value >= target:
        return "good"
    if value >= target * 0.85:
        return "warn"
    return "bad"


def _fmt(value: Any, precision: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    if isinstance(value, int):
        return str(value)
    if value is None or value == "":
        return "—"
    return str(value)


def _gap_status(
    *,
    key: str,
    label: str,
    value: float | None,
    target: float | None,
    direction: str,
    precision: int = 3,
) -> dict[str, Any]:
    tone = _tone_for(value, target=target, direction=direction)
    if value is None or target is None:
        reading = "—"
        gap = None
    elif direction == "lower":
        gap = value - target
        reading = f"{value:.{precision}f} / ≤{target:.{precision}f}"
    else:
        gap = target - value
        reading = f"{value:.{precision}f} / ≥{target:.{precision}f}"
    return {
        "key": key,
        "label": label,
        "value": value,
        "target": target,
        "direction": direction,
        "tone": tone,
        "reading": reading,
        "gap": gap,
    }


def _worst_gaps(gaps: list[dict[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
    severity = {"bad": 0, "warn": 1, "neutral": 2, "good": 3}
    candidates = [gap for gap in gaps if gap.get("tone") in {"bad", "warn"}]
    return sorted(candidates, key=lambda item: (severity.get(str(item.get("tone")), 9), str(item.get("key"))))[:limit]


def _gap_summary(gaps: list[dict[str, Any]]) -> str:
    worst = _worst_gaps(gaps)
    if not worst:
        return ""
    return "、".join(f"{item['label']} {item['reading']}" for item in worst)


def _item(
    *,
    key: str,
    label: str,
    value: Any,
    section: str,
    source: str,
    formula: str,
    meaning: str,
    unit: str = "",
    precision: int = 3,
    direction: str = "info",
    target: float | None = None,
    target_label: str = "",
    tone: str | None = None,
) -> MetricSnapshotItem:
    numeric = _num(value)
    resolved_tone = tone or _tone_for(
        numeric,
        target=target,
        direction="lower" if direction == "lower" else "higher",
    )
    reading = _fmt(value, precision)
    if target_label:
        reading += f" · 目标 {target_label}"
    elif target is not None:
        op = "≤" if direction == "lower" else "≥"
        reading += f" · 目标 {op}{target:.{precision}f}"
    return MetricSnapshotItem(
        key=key,
        label=label,
        value=value,
        numeric_value=numeric,
        unit=unit,
        precision=precision,
        tone=resolved_tone,  # type: ignore[arg-type]
        direction=direction,  # type: ignore[arg-type]
        target=target,
        target_label=target_label,
        section=section,
        source=source,
        formula=formula,
        meaning=meaning,
        reading=reading,
    )


def _path_from_latest(latest: TrainingTelemetryPoint | None, *keys: str) -> str:
    if latest is None:
        return ""
    for key in keys:
        value = latest.paths.get(key)
        if value:
            return value
    return ""


def _provenance(
    *,
    latest: TrainingTelemetryPoint | None,
    log_path: str,
    telemetry_path: str,
    console_log_path: str,
    event_file: str,
    checkpoint_dir: str,
    effective_config_path: str,
    evidence: list[str],
) -> list[DataProvenanceItem]:
    run_dir = _path_from_latest(latest, "run_dir")
    console_log = console_log_path or _path_from_latest(latest, "console_log")
    checkpoint = latest.checkpoint if latest else ""
    items = [
        DataProvenanceItem(
            key="run_dir",
            label="当前 run",
            path=run_dir,
            role="训练输出根目录；其他路径必须落在这个 run contract 内或来自显式配置。",
            available=bool(run_dir),
            evidence=evidence,
        ),
        DataProvenanceItem(
            key="telemetry_jsonl",
            label="实时主事实源",
            path=telemetry_path or _path_from_latest(latest, "telemetry_jsonl"),
            role="train.telemetry.jsonl；监控当前状态优先读取它。",
            available=bool(telemetry_path),
            evidence=evidence,
        ),
        DataProvenanceItem(
            key="train_log",
            label="语义日志",
            path=log_path or _path_from_latest(latest, "log", "train_log"),
            role="[TPSTAT]/[TPCMD]/[TPREW]/[TPGATE]/[TPCURR] 人类可读日志；JSONL 缺失时 fallback。",
            available=bool(log_path),
            evidence=evidence,
        ),
        DataProvenanceItem(
            key="console_log",
            label="完整终端日志",
            path=console_log,
            role="训练进程 stdout/stderr 汇总，用于排错，不作为指标主事实源。",
            available=bool(console_log),
            evidence=evidence,
        ),
        DataProvenanceItem(
            key="tensorboard",
            label="TensorBoard 历史曲线",
            path=event_file or _path_from_latest(latest, "tensorboard_dir"),
            role="tfevents scalar；用于历史曲线，不覆盖 JSONL 的最新状态。",
            available=bool(event_file or _path_from_latest(latest, "tensorboard_dir")),
            evidence=evidence,
        ),
        DataProvenanceItem(
            key="checkpoint_dir",
            label="检查点目录",
            path=checkpoint_dir or _path_from_latest(latest, "checkpoint_dir"),
            role="训练保存的 policy artifacts；诊断应从 catalog/快照选择，不硬编码。",
            available=bool(checkpoint_dir or _path_from_latest(latest, "checkpoint_dir")),
            evidence=evidence,
        ),
        DataProvenanceItem(
            key="latest_checkpoint",
            label="最新检查点",
            path=checkpoint,
            role="telemetry 或 checkpoint catalog 报告的最新保存文件。",
            available=bool(checkpoint),
            evidence=evidence,
        ),
        DataProvenanceItem(
            key="effective_config",
            label="有效配置",
            path=effective_config_path or _path_from_latest(latest, "effective_config"),
            role="本次 run 展开的最终配置；阈值/课程/奖励定义应以它为准。",
            available=bool(effective_config_path or _path_from_latest(latest, "effective_config")),
            evidence=evidence,
        ),
    ]
    return items


def build_run_snapshot(
    telemetry: TrainingTelemetry,
    *,
    console_log_path: str = "",
    checkpoint_dir: str = "",
    event_file: str = "",
    effective_config_path: str = "",
    effective_config_text: str = "",
    path_evidence: list[str] | None = None,
) -> RunSnapshot:
    latest = telemetry.latest
    evidence = list(path_evidence or [])
    if not latest:
        status = "error" if telemetry.error else "missing"
        _thresholds, threshold_notes = _gate_thresholds_from_config(effective_config_text)
        return RunSnapshot(
            run_id=telemetry.run_id,
            source=telemetry.source,
            mode=telemetry.mode,
            running=telemetry.running,
            runtime_state=telemetry.runtime_state,
            stale=telemetry.stale,
            telemetry_age_s=telemetry.telemetry_age_s,
            status=status,  # type: ignore[arg-type]
            conclusion=telemetry.error or "没有可解析的训练 telemetry；需要检查 JSONL/日志路径或训练是否已开始输出。",
            generated_at=time.time(),
            provenance=_provenance(
                latest=None,
                log_path=telemetry.log_path,
                telemetry_path=telemetry.telemetry_path,
                console_log_path=console_log_path,
                event_file=event_file,
                checkpoint_dir=checkpoint_dir,
                effective_config_path=effective_config_path,
                evidence=evidence,
            ),
            notes=list(telemetry.limitations) + threshold_notes,
        )

    reward = latest.reward
    curr = latest.curriculum
    health = latest.health
    command = latest.command
    phase = _text(curr.get("phase"))
    gate_thresholds, threshold_notes = _gate_thresholds_from_config(effective_config_text)
    progress_target = _progress_threshold_from(gate_thresholds, phase)
    gait_target = float(gate_thresholds["gait"])
    slip_target = float(gate_thresholds["slip"])
    diag_target = float(gate_thresholds["diagonal"])
    duty_target = float(gate_thresholds["duty"])
    fall_target = float(gate_thresholds["fall"])

    progress_gate = _num(curr.get("progress_gate"), command.get("progress_ratio"))
    gait_gate = _num(curr.get("gait_gate"), command.get("gait_match"), reward.get("gait"))
    diagonal = _num(command.get("diagonal_contact"), curr.get("diagonal_gate"))
    duty_balance = _num(command.get("duty_balance"), curr.get("duty_balance_gate"))
    stance_slip = _num(command.get("stance_slip"))
    if stance_slip is None:
        stance_slip = abs(_num(reward.get("stance_slip"), reward.get("slip")) or 0.0)
    diag_pair_instant = _num(command.get("diagonal_pair_instant"))
    duty_spread_window = _num(command.get("duty_spread_window"))
    stance_slip_high_fraction = _num(command.get("stance_slip_high_fraction"))
    height_low_risk_window = _num(health.get("height_low_risk_window"))
    tilt_high_risk_window = _num(health.get("tilt_high_risk_window"))
    fall_rate = _num(health.get("fall_rate"), curr.get("fall_gate"))
    lin_err = _num(command.get("lin_err"), reward.get("lin_err"))
    duty_spread_target = 0.20
    slip_event_target = 0.05
    height_risk_target = 0.05
    tilt_risk_target = 0.05
    blocked_by = _text(curr.get("blocked_by"))
    next_gate = _text(curr.get("next_gate") or curr.get("next"))
    command_mode = _text(curr.get("command_mode"))
    active_dirs = _text(curr.get("active_dirs"))
    progress_by_dir = _dir_progress_values(latest)
    progress_lagging_dir = _text(curr.get("progress_lagging_dir"))
    constraint_gaps = [
        _gap_status(
            key="progress_gate",
            label="推进",
            value=progress_gate,
            target=progress_target,
            direction="higher",
        ),
        _gap_status(
            key="diagonal_contact",
            label="对角支撑",
            value=diagonal,
            target=diag_target,
            direction="higher",
        ),
        _gap_status(
            key="duty_balance",
            label="占空平衡",
            value=duty_balance,
            target=duty_target,
            direction="higher",
        ),
        _gap_status(
            key="stance_slip",
            label="平均打滑",
            value=stance_slip,
            target=slip_target,
            direction="lower",
        ),
        _gap_status(
            key="duty_spread_window",
            label="占空差",
            value=duty_spread_window,
            target=duty_spread_target,
            direction="lower",
        ),
        _gap_status(
            key="stance_slip_high_fraction",
            label="高滑移比例",
            value=stance_slip_high_fraction,
            target=slip_event_target,
            direction="lower",
        ),
        _gap_status(
            key="height_low_risk_window",
            label="低高度风险",
            value=height_low_risk_window,
            target=height_risk_target,
            direction="lower",
        ),
        _gap_status(
            key="tilt_high_risk_window",
            label="大倾角风险",
            value=tilt_high_risk_window,
            target=tilt_risk_target,
            direction="lower",
        ),
        _gap_status(
            key="fall_rate",
            label="摔倒率",
            value=fall_rate,
            target=fall_target,
            direction="lower",
        ),
    ]
    lagging_summary = _gap_summary(constraint_gaps)
    blockers: list[SnapshotBlocker] = []
    progress_blocker = _directional_progress_blocker(
        progress_by_dir=progress_by_dir,
        active_dirs=active_dirs,
        progress_gate=progress_gate,
        progress_target=progress_target,
        blocked_by=blocked_by,
    )
    if progress_blocker is not None:
        blockers.append(progress_blocker)
    for gap in _worst_gaps(constraint_gaps, limit=4):
        if gap.get("key") == "progress_gate" and progress_blocker is not None:
            continue
        blocker = _blocker_from_gap(
            gap,
            source=f"{gap.get('key')} snapshot metric",
            detail=f"{gap.get('label')} is behind its current training gate; inspect source/formula before changing weights.",
        )
        if blocker is not None:
            blockers.append(blocker)

    if telemetry.mode == "log_fallback":
        status = "watch"
        conclusion = "当前使用日志 fallback，指标不如 JSONL 完整；先确认 JSONL 是否持续写入。"
    elif blocked_by and blocked_by not in {"none", "ok", "-"}:
        status = "blocked"
        if blocked_by == "progress":
            conclusion = (
                f"{phase or '当前阶段'} 被 progress gate 卡住："
                f"{_fmt(progress_gate)} / 目标 ≥{progress_target:.2f}。"
            )
        else:
            conclusion = f"{phase or '当前阶段'} blocked_by={blocked_by}，下一门控={next_gate or '未知'}。"
    elif progress_gate is not None and progress_gate < progress_target:
        status = "watch"
        conclusion = f"推进能力尚未达到阶段门槛：progress={progress_gate:.3f}，目标 ≥{progress_target:.2f}。"
    elif lagging_summary:
        status = "watch"
        conclusion = f"推进 gate 已接近/通过，但真实步态约束仍落后：{lagging_summary}。"
    else:
        status = "ok"
        conclusion = "当前 telemetry 可读，推进、步态、滑移和稳定性主要约束暂未显示明显阻塞。"

    groups = [
        MetricSnapshotGroup(
            id="constraints",
            title="当前约束落后项",
            summary="把训练目标拆成可核对的约束：推进过线不代表步态真实过线，优先看这里的红/黄项。",
            items=[
                _item(
                    key="progress_gate",
                    label="推进 gate",
                    value=progress_gate,
                    precision=3,
                    direction="higher",
                    target=progress_target,
                    section="curriculum",
                    source="curriculum.progress_gate; fallback command.progress_ratio",
                    formula=f"recent-window progress over active_dirs; target phase_gate_prog_{_phase_index(phase) if _phase_index(phase) is not None else '?'}.",
                    meaning="当前阶段是否认为机器人已经有足够命令方向推进。它不能单独证明步态正确。",
                ),
                _item(
                    key="diagonal_contact",
                    label="对角支撑",
                    value=diagonal,
                    precision=3,
                    direction="higher",
                    target=diag_target,
                    section="command",
                    source="command.diagonal_contact; fallback curriculum.diagonal_gate",
                    formula="对角腿接触组合与 trot 支撑模式的一致性。",
                    meaning="低值说明不是稳定对角小跑；可能是 crawl、拖行、三脚支撑或乱步。",
                ),
                _item(
                    key="duty_balance",
                    label="占空平衡",
                    value=duty_balance,
                    precision=3,
                    direction="higher",
                    target=duty_target,
                    section="command",
                    source="command.duty_balance; fallback curriculum.duty_balance_gate",
                    formula="前后腿/左右腿站立占空比平衡评分。",
                    meaning="低值说明腿之间承担支撑不均，常见于后腿长撑、前腿拖行或偏置步态。",
                ),
                _item(
                    key="stance_slip",
                    label="平均打滑",
                    value=stance_slip,
                    precision=3,
                    direction="lower",
                    target=slip_target,
                    section="command/reward",
                    source="command.stance_slip; fallback abs(reward.stance_slip)",
                    formula="支撑脚水平滑动代理；reward 中对应负惩罚项。",
                    meaning="高值说明脚在地面上滑，实际运动可能不是干净步态推进。",
                ),
                _item(
                    key="stance_slip_high_fraction",
                    label="高滑移比例",
                    value=stance_slip_high_fraction,
                    precision=3,
                    direction="lower",
                    target=slip_event_target,
                    section="command",
                    source="command.stance_slip_high_fraction",
                    formula="EMA(fraction of settled stance feet with slip speed > slip_high_threshold).",
                    meaning="局部高滑移事件比例；能暴露平均 slip 被稀释时的拖滑问题。",
                ),
                _item(
                    key="duty_spread_window",
                    label="占空差",
                    value=duty_spread_window,
                    precision=3,
                    direction="lower",
                    target=duty_spread_target,
                    section="command",
                    source="command.duty_spread_window",
                    formula="max(per_leg_duty_ema) - min(per_leg_duty_ema).",
                    meaning="各腿长期站立占空差；越高越像 crawl/拖行/前后腿偏置。",
                ),
                _item(
                    key="height_low_risk_window",
                    label="低高度风险",
                    value=height_low_risk_window,
                    precision=3,
                    direction="lower",
                    target=height_risk_target,
                    section="health",
                    source="health.height_low_risk_window",
                    formula="EMA(clamp((h_ok - base_h) / (h_ok - h_gate_close), 0, 1)).",
                    meaning="近期低身高风险；平均身高正常时也可能隐藏短时蹲塌。",
                ),
                _item(
                    key="tilt_high_risk_window",
                    label="大倾角风险",
                    value=tilt_high_risk_window,
                    precision=3,
                    direction="lower",
                    target=tilt_risk_target,
                    section="health",
                    source="health.tilt_high_risk_window",
                    formula="EMA(clamp((tilt - tilt_ok) / (tilt_gate_close - tilt_ok), 0, 1)).",
                    meaning="近期大倾角风险；姿态风险会折扣正奖励并影响接触质量。",
                ),
            ],
        ),
        MetricSnapshotGroup(
            id="directional_progress",
            title="各方向推进 gate",
            summary="显示 active_dirs 中每个方向的近期推进能力；phase 卡住时优先看最弱方向，而不是只看总 progress_gate。",
            items=[
                _item(
                    key=f"progress_{direction}",
                    label=f"{direction} progress",
                    value=progress_by_dir.get(direction),
                    precision=3,
                    direction="higher",
                    target=progress_target,
                    tone="bad" if progress_lagging_dir == direction and (blocked_by == "progress" or (progress_by_dir.get(direction) or 0.0) < progress_target) else None,
                    section="curriculum/command",
                    source="curriculum.progress_* / command.progress_*; may be augmented from console.log for old live runs",
                    formula=f"recent-window commanded progress score for {direction}; phase target ≥{progress_target:.2f}.",
                    meaning="这是课程推进的方向分项。最小的 active direction 会决定 progress gate，能解释为什么总体看起来接近但 phase 不推进。",
                )
                for direction in ("fwd", "back", "lat", "yaw")
            ],
        ),
        MetricSnapshotGroup(
            id="progress",
            title="推进与命令跟踪",
            summary="判断机器人是否按命令产生有效速度；这是“会不会走”的主面板。",
            items=[
                _item(
                    key="cmd_vx",
                    label="命令 vx",
                    value=_num(command.get("cmd_vx")),
                    unit="m/s",
                    precision=3,
                    section="command",
                    source="train.telemetry.jsonl command.cmd_vx",
                    formula="训练采样的目标前向速度。",
                    meaning="当前策略被要求达到的前向速度。",
                ),
                _item(
                    key="actual_vx",
                    label="实际 vx",
                    value=_num(command.get("actual_vx")),
                    unit="m/s",
                    precision=3,
                    section="command",
                    source="train.telemetry.jsonl command.actual_vx",
                    formula="仿真 base 速度在 x 方向的测量值。",
                    meaning="实际走出来的前向速度；应接近命令 vx。",
                ),
                _item(
                    key="v_along",
                    label="命令方向速度",
                    value=_num(command.get("v_along")),
                    unit="m/s",
                    precision=3,
                    section="command",
                    source="train.telemetry.jsonl command.v_along",
                    formula="实际速度投影到当前命令方向。",
                    meaning="比单独看 vx 更适合 mixed/yaw/lateral 阶段。",
                ),
                _item(
                    key="lin_err",
                    label="线速度误差",
                    value=lin_err,
                    unit="m/s",
                    precision=3,
                    direction="lower",
                    section="command/reward",
                    source="command.lin_err; fallback reward.lin_err",
                    formula="||commanded_xy_velocity - measured_xy_velocity||。",
                    meaning="越低越好；高误差说明速度跟踪不足或方向错误。",
                ),
                _item(
                    key="progress_gate",
                    label="阶段推进 gate",
                    value=progress_gate,
                    precision=3,
                    direction="higher",
                    target=progress_target,
                    section="curriculum",
                    source="curriculum.progress_gate; fallback command.progress_ratio",
                    formula=f"最近窗口内当前 active_dirs 的推进能力；当前阶段目标 phase_gate_prog_{_phase_index(phase) if _phase_index(phase) is not None else '?'}。",
                    meaning="课程是否允许进入下一阶段的主 gate。",
                ),
                _item(
                    key="progress_lagging_dir",
                    label="最弱推进方向",
                    value=progress_lagging_dir or None,
                    section="curriculum/command",
                    source="curriculum.progress_lagging_dir; fallback derived from progress_*",
                    formula="argmin(progress_fwd, progress_back, progress_lat, progress_yaw over active_dirs).",
                    meaning="当 blocked_by=progress 时，它指出真正拖住阶段推进的方向。",
                ),
                _item(
                    key="tracking_lin",
                    label="近场跟踪奖励",
                    value=_num(reward.get("tracking_lin"), reward.get("track")),
                    precision=3,
                    direction="higher",
                    section="reward",
                    source="reward.tracking_lin; fallback reward.track",
                    formula="训练 reward 中的线速度跟踪项。",
                    meaning="越高通常表示命令速度误差越小，但必须和 progress/actual_vx 一起看。",
                ),
            ],
        ),
        MetricSnapshotGroup(
            id="gait",
            title="步态质量",
            summary="解释为什么 tracking 可能不低但前端看起来仍然爬行、拖脚或步态差。",
            items=[
                _item(
                    key="gait_match",
                    label="步态匹配",
                    value=gait_gate,
                    precision=3,
                    direction="higher",
                    target=gait_target,
                    section="command/curriculum",
                    source="command.gait_match; fallback curriculum.gait_gate/reward.gait",
                    formula="当前接触节律与期望步态的匹配度。",
                    meaning="越高越接近期望步态；高步态分不等于真实前进好。",
                ),
                _item(
                    key="diagonal_contact",
                    label="对角支撑一致性",
                    value=diagonal,
                    precision=3,
                    direction="higher",
                    target=diag_target,
                    section="command",
                    source="command.diagonal_contact; fallback curriculum.diagonal_gate",
                    formula="对角腿接触组合与 trot 支撑模式的一致性。",
                    meaning="低值说明可能不是稳定对角 trot，而是 crawl/拖行/乱步。",
                ),
                _item(
                    key="duty_balance",
                    label="占空比平衡",
                    value=duty_balance,
                    precision=3,
                    direction="higher",
                    target=duty_target,
                    section="command",
                    source="command.duty_balance; fallback curriculum.duty_balance_gate",
                    formula="前后腿/左右腿站立占空比平衡评分。",
                    meaning="低值说明可能出现后腿长时间撑地、前腿拖行等模式。",
                ),
                _item(
                    key="stance_slip",
                    label="支撑期打滑",
                    value=stance_slip,
                    precision=3,
                    direction="lower",
                    target=slip_target,
                    section="command/reward",
                    source="command.stance_slip; fallback abs(reward.stance_slip)",
                    formula="支撑脚水平滑动代理，惩罚项在 reward 中常为负值。",
                    meaning="越低越好；高值说明脚在地上滑，不是真正稳态推进。",
                ),
                _item(
                    key="diagonal_pair_instant",
                    label="瞬时严格对角",
                    value=diag_pair_instant,
                    precision=3,
                    direction="higher",
                    section="command",
                    source="command.diagonal_pair_instant",
                    formula="1 only when contact is exactly FL+RR or FR+RL; otherwise 0.",
                    meaning="当前帧是否是真正对角支撑。四脚拖行、三脚支撑不会得分。",
                ),
                _item(
                    key="duty_spread_window",
                    label="占空差窗口",
                    value=duty_spread_window,
                    precision=3,
                    direction="lower",
                    target=0.20,
                    section="command",
                    source="command.duty_spread_window",
                    formula="max(per_leg_duty_ema) - min(per_leg_duty_ema).",
                    meaning="腿之间长期占空差。越高越像 crawl、拖行或单侧/前后腿偏置。",
                ),
                _item(
                    key="stance_slip_high_fraction",
                    label="高滑移比例",
                    value=stance_slip_high_fraction,
                    precision=3,
                    direction="lower",
                    target=0.05,
                    section="command",
                    source="command.stance_slip_high_fraction",
                    formula="EMA(fraction of settled stance feet with slip speed > slip_high_threshold).",
                    meaning="局部滑移事件比例；用于补足平均 slip 被稀释的问题。",
                ),
                _item(
                    key="stable_motion_gate",
                    label="稳定运动 gate",
                    value=_num(health.get("stable_motion_gate"), reward.get("stable_motion_gate"), reward.get("gait")),
                    precision=3,
                    direction="higher",
                    section="health/reward",
                    source="health.stable_motion_gate; fallback reward.stable_motion_gate",
                    formula="姿态稳定、移动、接触等组合的稳定性门控。",
                    meaning="用于判断能否安全提高质量惩罚或课程难度。",
                ),
                _item(
                    key="landing_impact",
                    label="落地冲击惩罚",
                    value=abs(_num(reward.get("landing_impact")) or 0.0) if reward.get("landing_impact") is not None else None,
                    precision=3,
                    direction="lower",
                    section="reward",
                    source="abs(reward.landing_impact)",
                    formula="触地瞬间冲击相关惩罚的绝对值。",
                    meaning="高值通常意味着脚落地硬、姿态或摆腿轨迹不稳。",
                ),
            ],
        ),
        MetricSnapshotGroup(
            id="health",
            title="稳定性与安全",
            summary="先排除摔倒、姿态塌陷、力矩饱和，再解释 reward/tracking。",
            items=[
                _item(
                    key="fall_rate",
                    label="摔倒率",
                    value=fall_rate,
                    precision=3,
                    direction="lower",
                    target=fall_target,
                    section="health",
                    source="health.fall_rate; fallback curriculum.fall_gate",
                    formula="最近窗口 falls / episodes 或终止窗口代理。",
                    meaning="高于阈值时应优先修稳定性，而不是继续加速度命令。",
                ),
                _item(
                    key="terminal_rate",
                    label="终止率",
                    value=_num(health.get("terminal_rate")),
                    precision=3,
                    direction="lower",
                    section="health",
                    source="health.terminal_rate",
                    formula="最近窗口触发 episode 终止的比例。",
                    meaning="用于区分 reward 低是因为动作差还是频繁终止。",
                ),
                _item(
                    key="base_h",
                    label="机身高度",
                    value=_num(health.get("base_h"), reward.get("base_h")),
                    unit="m",
                    precision=3,
                    direction="higher",
                    target=_num(reward.get("cfg_height_min_ok")) or 0.47,
                    section="health",
                    source="health.base_h",
                    formula="base link 高度。",
                    meaning="过低说明塌陷/蹲伏，过高可能跳跃或姿态异常。",
                ),
                _item(
                    key="upright",
                    label="直立度",
                    value=_num(health.get("upright"), reward.get("upright")),
                    precision=3,
                    direction="higher",
                    section="health",
                    source="health.upright",
                    formula="机身 up 方向与世界 z 的对齐程度。",
                    meaning="越高越直立；低值说明姿态倾斜。",
                ),
                _item(
                    key="base_h_min",
                    label="最低机身高度",
                    value=_num(health.get("base_h_min")),
                    unit="m",
                    precision=3,
                    direction="higher",
                    target=0.47,
                    section="health",
                    source="health.base_h_min",
                    formula="min(base_h) over the current telemetry batch.",
                    meaning="均值高度会隐藏短时塌陷；最低高度更接近诊断里的低身高风险。",
                ),
                _item(
                    key="height_low_risk_window",
                    label="低高度风险窗口",
                    value=height_low_risk_window,
                    precision=3,
                    direction="lower",
                    target=0.05,
                    section="health",
                    source="health.height_low_risk_window",
                    formula="EMA(clamp((h_ok - base_h) / (h_ok - h_gate_close), 0, 1)).",
                    meaning="训练 gate 使用的近期低身高风险。非零说明近期有蹲塌/低高度片段。",
                ),
                _item(
                    key="tilt_deg",
                    label="倾角",
                    value=_num(health.get("tilt_deg")),
                    unit="deg",
                    precision=1,
                    direction="lower",
                    section="health",
                    source="health.tilt_deg",
                    formula="base 姿态倾角角度。",
                    meaning="持续偏大时会影响接触和推进。",
                ),
                _item(
                    key="tilt_deg_max",
                    label="最大倾角",
                    value=_num(health.get("tilt_deg_max")),
                    unit="deg",
                    precision=1,
                    direction="lower",
                    target=15.0,
                    section="health",
                    source="health.tilt_deg_max",
                    formula="max(tilt_deg) over the current telemetry batch.",
                    meaning="均值倾角会隐藏短时大摆动；最大值更接近诊断里的姿态风险。",
                ),
                _item(
                    key="tilt_high_risk_window",
                    label="大倾角风险窗口",
                    value=tilt_high_risk_window,
                    precision=3,
                    direction="lower",
                    target=0.05,
                    section="health",
                    source="health.tilt_high_risk_window",
                    formula="EMA(clamp((tilt - tilt_ok) / (tilt_gate_close - tilt_ok), 0, 1)).",
                    meaning="训练 gate 使用的近期大倾角风险。非零说明姿态约束已在打折正奖励。",
                ),
                _item(
                    key="torque_util",
                    label="力矩利用率",
                    value=_num(health.get("torque_util")),
                    precision=3,
                    direction="lower",
                    section="health",
                    source="health.torque_util",
                    formula="实际力矩 / 力矩限制的归一化代理。",
                    meaning="高值说明策略可能靠硬顶力矩运动。",
                ),
            ],
        ),
        MetricSnapshotGroup(
            id="curriculum",
            title="课程与阶段",
            summary="解释为什么训练还在当前阶段，以及下一步被哪个 gate 控制。",
            items=[
                _item(
                    key="phase",
                    label="阶段",
                    value=phase,
                    section="curriculum",
                    source="curriculum.phase",
                    formula="训练课程状态机当前 phase。",
                    meaning="决定当前命令分布、地形/DR/惩罚是否开启。",
                ),
                _item(
                    key="command_mode",
                    label="命令模式",
                    value=command_mode,
                    section="curriculum",
                    source="curriculum.command_mode",
                    formula="当前 phase 的命令采样模式。",
                    meaning="fixed_forward/single_axis/mixed 会改变指标解释方式。",
                ),
                _item(
                    key="active_dirs",
                    label="验收方向",
                    value=active_dirs,
                    section="curriculum",
                    source="curriculum.active_dirs",
                    formula="当前 gate 统计的方向集合。",
                    meaning="progress_gate 只应按 active_dirs 判断，不应混入未开放方向。",
                ),
                _item(
                    key="penalty_gate",
                    label="质量惩罚渐入",
                    value=_num(curr.get("penalty_gate")),
                    precision=3,
                    section="curriculum",
                    source="curriculum.penalty_gate",
                    formula="质量惩罚权重随阶段/interval 从 0 到 1 渐入。",
                    meaning="早期为 0 时，步态质量差不一定会强烈惩罚。",
                ),
                _item(
                    key="terrain_mean",
                    label="地形均值等级",
                    value=_num(curr.get("terrain_mean")),
                    precision=2,
                    section="curriculum",
                    source="curriculum.terrain_mean",
                    formula="当前地形 curriculum 平均等级。",
                    meaning="用于解释 reward 突变是否来自地形升级。",
                ),
                _item(
                    key="dr_level",
                    label="DR 等级",
                    value=curr.get("dr_level"),
                    precision=0,
                    section="curriculum",
                    source="curriculum.dr_level",
                    formula="Domain Randomization 当前等级。",
                    meaning="DR 升级会增加训练难度和曲线波动。",
                ),
            ],
        ),
        MetricSnapshotGroup(
            id="reward",
            title="奖励分解",
            summary="奖励只作为训练目标分解看，不再把一个来源不明的 reward 曲线当结论。",
            items=[
                _item(
                    key="reward_total",
                    label="总奖励",
                    value=_num(reward.get("total"), reward.get("rew")),
                    precision=3,
                    section="reward",
                    source="reward.total",
                    formula="训练环境汇总后的当前 reward 标量。",
                    meaning="综合目标；必须和 progress、gait、health 分项一起判断。",
                ),
                _item(
                    key="terrain_progress",
                    label="地形推进奖励",
                    value=_num(reward.get("terrain_progress")),
                    precision=3,
                    section="reward",
                    source="reward.terrain_progress",
                    formula="命令方向上的推进/地形相关 reward 分量。",
                    meaning="高值说明有推进趋势，但仍需检查实际速度和打滑。",
                ),
                _item(
                    key="tracking_lin_far",
                    label="远场跟踪奖励",
                    value=_num(reward.get("tracking_lin_far")),
                    precision=3,
                    section="reward",
                    source="reward.tracking_lin_far",
                    formula="速度误差较大时仍提供学习梯度的远场跟踪项。",
                    meaning="早期训练有用；如果远场高而近场低，说明还没精确跟踪。",
                ),
                _item(
                    key="clearance_under",
                    label="清高不足惩罚",
                    value=abs(_num(reward.get("clearance_under")) or 0.0) if reward.get("clearance_under") is not None else None,
                    precision=3,
                    direction="lower",
                    section="reward",
                    source="abs(reward.clearance_under)",
                    formula="摆动脚低于目标清高带的惩罚绝对值。",
                    meaning="高值说明容易拖脚/绊脚。",
                ),
                _item(
                    key="clearance_over",
                    label="清高过高惩罚",
                    value=abs(_num(reward.get("clearance_over")) or 0.0) if reward.get("clearance_over") is not None else None,
                    precision=3,
                    direction="lower",
                    section="reward",
                    source="abs(reward.clearance_over)",
                    formula="摆动脚高于目标清高带的惩罚绝对值。",
                    meaning="高值说明抬腿过高、能耗和冲击风险上升。",
                ),
                _item(
                    key="height_low",
                    label="低身高惩罚",
                    value=abs(_num(reward.get("height_low")) or 0.0) if reward.get("height_low") is not None else None,
                    precision=3,
                    direction="lower",
                    target=0.0,
                    section="reward",
                    source="abs(reward.height_low)",
                    formula="w_height_low * max((height_min_ok - base_h) / (nominal_base_h - height_min_ok), 0)^2.",
                    meaning="0 means base height is above the spec lower bound; non-zero means crouching is being penalized.",
                ),
                _item(
                    key="torque_saturation",
                    label="力矩饱和惩罚",
                    value=abs(_num(reward.get("torque_saturation"), reward.get("torque_sat")) or 0.0)
                    if (reward.get("torque_saturation") is not None or reward.get("torque_sat") is not None)
                    else None,
                    precision=3,
                    direction="lower",
                    section="reward",
                    source="abs(reward.torque_saturation)",
                    formula="接近或超过力矩能力边界的惩罚绝对值。",
                    meaning="持续高值说明动作可能不可部署。",
                ),
            ],
        ),
    ]

    total_steps = latest.total_steps
    progress_pct = (latest.step / total_steps) if total_steps else None
    return RunSnapshot(
        run_id=telemetry.run_id,
        source=telemetry.source,
        mode=telemetry.mode,
        running=telemetry.running,
        status=status,  # type: ignore[arg-type]
        conclusion=conclusion,
        generated_at=time.time(),
        step=latest.step,
        total_steps=total_steps,
        progress_pct=progress_pct,
        elapsed=latest.elapsed,
        eta=latest.eta,
        fps=latest.fps,
        phase=phase,
        command_mode=command_mode,
        active_dirs=active_dirs,
        blocked_by=blocked_by,
        next_gate=next_gate,
        latest_checkpoint=latest.checkpoint,
        provenance=_provenance(
            latest=latest,
            log_path=telemetry.log_path,
            telemetry_path=telemetry.telemetry_path,
            console_log_path=console_log_path,
            event_file=event_file,
            checkpoint_dir=checkpoint_dir,
            effective_config_path=effective_config_path,
            evidence=evidence,
        ),
        metric_groups=groups,
        blockers=blockers,
        notes=list(telemetry.limitations) + threshold_notes,
    )


def build_telemetry(
    *,
    source: str,
    run_id: str,
    running: bool,
    log_path: str,
    telemetry_path: str,
    console_log_path: str = "",
    checkpoint_dir: str = "",
    event_file: str = "",
    effective_config_path: str = "",
    effective_config_text: str = "",
    path_evidence: list[str] | None = None,
    source_stats: dict[str, Any] | None = None,
    jsonl_text: str = "",
    log_text: str = "",
    side_log_text: str = "",
    error: str = "",
) -> TrainingTelemetry:
    points = parse_telemetry_jsonl(jsonl_text)
    mode = "jsonl" if points else "missing"
    limitations: list[str] = []
    if not points and log_text:
        points = parse_training_log(log_text)
        mode = "log_fallback" if points else "missing"
        if points:
            limitations.append("当前使用 [TPREW]/[TPSTAT] 日志 fallback；缺少 JSONL 时无法可靠判断 terrain/DR/phase/checkpoint 等完整状态。")
    if points and side_log_text:
        _augment_points_from_side_log(points, side_log_text)
    latest = points[-1] if points else None
    if not points and not error:
        limitations.append("远程训练脚本尚未输出 telemetry JSONL，且日志里没有可解析的 [TPREW] 行。")
    resolved_source_stats: dict[str, Any] = dict(source_stats or {})
    generated_at = time.time()
    configured_interval = _runtime_telemetry_interval_from_config(effective_config_text)
    resolved_source_stats.setdefault("source_kind", mode)
    resolved_source_stats.setdefault("history_points", len(points))
    if configured_interval is not None:
        resolved_source_stats.setdefault("configured_interval_steps", configured_interval)
    if latest:
        resolved_source_stats.setdefault("latest_step", latest.step)
        if latest.ts:
            resolved_source_stats.setdefault("latest_ts", latest.ts)
            resolved_source_stats.setdefault("latest_age_s", max(0.0, generated_at - float(latest.ts)))
        if latest.fps and latest.fps > 0 and configured_interval:
            resolved_source_stats.setdefault("expected_wall_interval_s", float(configured_interval) / float(latest.fps))
        if len(points) >= 2:
            previous = points[-2]
            step_delta = latest.step - previous.step
            wall_delta = latest.ts - previous.ts
            if step_delta > 0:
                resolved_source_stats.setdefault("observed_step_interval", step_delta)
            if wall_delta > 0:
                resolved_source_stats.setdefault("observed_wall_interval_s", wall_delta)
                if step_delta > 0:
                    resolved_source_stats.setdefault("observed_steps_per_second", step_delta / wall_delta)
    remote_now = _as_float(resolved_source_stats.get("remote_now_s"))
    mtime = _as_float(resolved_source_stats.get("mtime_s"))
    if remote_now is not None and mtime is not None:
        resolved_source_stats.setdefault("file_mtime_age_s", max(0.0, remote_now - mtime))
    latest_age = _as_float(resolved_source_stats.get("latest_age_s"))
    expected_wall = _as_float(resolved_source_stats.get("expected_wall_interval_s"))
    stale_threshold = max(30.0, (expected_wall or 2.0) * 5.0)
    stale = bool(latest and ((not running) or (latest_age is not None and latest_age > stale_threshold)))
    if latest_age is not None:
        resolved_source_stats.setdefault("stale_threshold_s", stale_threshold)
        resolved_source_stats.setdefault("stale", stale)
    runtime_state = "unknown"
    if error:
        runtime_state = "remote_unavailable"
    elif stale and running:
        runtime_state = "stale"
    elif latest and not running:
        runtime_state = "stopped"
    elif latest and running:
        runtime_state = "live"
    elif not latest:
        runtime_state = "stopped" if not running else "stale"

    telemetry = TrainingTelemetry(
        available=bool(points),
        source="real" if source == "real" else "fake",  # type: ignore[arg-type]
        run_id=run_id,
        running=running,
        runtime_state=runtime_state,  # type: ignore[arg-type]
        stale=stale,
        telemetry_age_s=latest_age,
        log_path=log_path,
        telemetry_path=telemetry_path,
        mode=mode,  # type: ignore[arg-type]
        source_stats=resolved_source_stats,
        latest=latest,
        history=points[-240:],
        summary=summarize_telemetry(points),
        limitations=limitations,
        error=error,
    )
    telemetry.snapshot = build_run_snapshot(
        telemetry,
        console_log_path=console_log_path,
        checkpoint_dir=checkpoint_dir,
        event_file=event_file,
        effective_config_path=effective_config_path,
        effective_config_text=effective_config_text,
        path_evidence=list(path_evidence or []),
    )
    if stale:
        telemetry.limitations.append(
            "Telemetry is stale: the latest sample is not live; displayed values are the last recorded state."
        )
        if telemetry.snapshot is not None:
            telemetry.snapshot.stale = True
            telemetry.snapshot.telemetry_age_s = latest_age
            telemetry.snapshot.runtime_state = runtime_state  # type: ignore[assignment]
            telemetry.snapshot.notes.append(
                "Telemetry is stale: training is not currently writing this stream."
            )
            if not running:
                telemetry.snapshot.conclusion = "训练已停止；以下指标是最后一次 telemetry，不代表当前实时状态。 " + telemetry.snapshot.conclusion
    elif telemetry.snapshot is not None:
        telemetry.snapshot.runtime_state = runtime_state  # type: ignore[assignment]
        telemetry.snapshot.telemetry_age_s = latest_age
    return telemetry
