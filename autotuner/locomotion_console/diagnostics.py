"""One-click orchestration and report shaping for Isaac Lab physical diagnostics."""
from __future__ import annotations

import asyncio
import csv
from difflib import SequenceMatcher
import io
import json
import math
import re
import shlex
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .config import LocomotionConsoleSettings
from .config_set import get_active_config_set
from .diagnostic_history import DiagnosticHistoryStore
from .framework_profile import FrameworkProfile, get_framework_profile, list_framework_profiles
from .schemas import (
    ArtifactRefInfo,
    DiagnosticCatalog,
    DiagnosticCheckpoint,
    DiagnosticJobStatus,
    DiagnosticLogInfo,
    DiagnosticPlayback,
    DiagnosticPlaybackFoot,
    DiagnosticPlaybackFrame,
    DiagnosticPlan,
    DiagnosticPreset,
    DiagnosticReport,
    DiagnosticStageStatus,
    DiagnosticValueInfo,
)


async def probe_concurrent_headroom(remote, n_env: int = 4) -> "tuple[bool, str]":
    """Whether a num_envs=n_env diagnostic fits ALONGSIDE a running training: checks BOTH the memory-cgroup
    RAM headroom AND GPU free VRAM. The operator wants mid-training probes to run whenever there is space —
    block ONLY on a real OOM / VRAM-contention risk that would kill both processes."""
    import asyncio as _asyncio
    need_ram = 12.0 + 0.03 * n_env + 4.0
    need_vram = 3.0 + 0.05 * n_env
    ram_free = -1.0
    vram_free = -1.0
    try:
        raw = await _asyncio.to_thread(
            remote.exec_out,
            "lim=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null); "
            "cur=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null); "
            "vram=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'); "
            "echo \"$lim $cur ${vram:-X}\"")
        parts = (raw or "").split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            ram_free = (int(parts[0]) - int(parts[1])) / (1024.0 ** 3)
        if len(parts) >= 3 and parts[2].isdigit():
            vram_free = int(parts[2]) / 1024.0
    except Exception:
        return (False, "无法读取内存/显存余量")
    if ram_free < 0:
        return (False, "无法读取 cgroup 内存余量")
    ok = (ram_free >= need_ram) and (vram_free < 0 or vram_free >= need_vram)
    detail = (f"RAM空闲~{ram_free:.0f}GB(需~{need_ram:.0f}) 显存空闲~{vram_free:.0f}GB(需~{need_vram:.0f}) num_envs={n_env}")
    return (ok, detail)


@dataclass(frozen=True)
class _Stage:
    id: str
    label: str
    suite: str
    expected_segments: int


@dataclass(frozen=True)
class _Preset:
    id: str
    label: str
    description: str
    category: str
    estimated_minutes: int
    stages: tuple[_Stage, ...]


@dataclass
class _Job:
    id: str
    preset: _Preset
    checkpoint: str
    framework_id: str
    diagnostic_task: str
    output_dir: str
    started_at: float
    plan: dict[str, Any] | None = None
    state: str = "starting"
    message: str = ""


PRESETS: tuple[_Preset, ...] = (
    _Preset(
        "quick",
        "Quick check",
        "Verify the diagnostic task, checkpoint, and record pipeline.",
        "quick",
        1,
        (_Stage("quick", "Quick check", "remote_probe.yaml", 2),),
    ),
    _Preset(
        "forward",
        "Forward",
        "Run a stand baseline and then switch to a forward command.",
        "direction",
        2,
        (_Stage("forward", "Forward", "direction_forward.yaml", 2),),
    ),
    _Preset(
        "backward",
        "Backward",
        "Run a stand baseline and then switch to a backward command.",
        "direction",
        2,
        (_Stage("backward", "Backward", "direction_backward.yaml", 2),),
    ),
    _Preset(
        "lateral",
        "Lateral",
        "Run a stand baseline and then switch to a lateral command.",
        "direction",
        2,
        (_Stage("lateral", "Lateral", "direction_lateral.yaml", 2),),
    ),
    _Preset(
        "yaw",
        "Yaw",
        "Run a stand baseline and then switch to a yaw command.",
        "direction",
        2,
        (_Stage("yaw", "Yaw", "direction_yaw.yaml", 2),),
    ),
    _Preset(
        "directions",
        "All directions",
        "Run forward, backward, lateral, and yaw in one IsaacLab session.",
        "direction",
        3,
        (_Stage("directions", "All directions", "directions_combined.yaml", 5),),
    ),
    _Preset(
        "terrain",
        "Terrain coverage",
        "Check flat, slope, rough, and stair terrain coverage.",
        "environment",
        5,
        (_Stage("terrain", "Terrain coverage", "terrain_probe.yaml", 2),),
    ),
    _Preset(
        "dr",
        "Domain randomization",
        "Compare observed behavior across DR0 through DR3 settings.",
        "robustness",
        3,
        (_Stage("dr", "Domain randomization", "dr_probe.yaml", 2),),
    ),
    _Preset(
        "push",
        "External push",
        "Apply a known lateral delta-v and observe recovery.",
        "robustness",
        2,
        (_Stage("push", "External push", "push_probe.yaml", 1),),
    ),
)

_PRESET_BY_ID = {preset.id: preset for preset in PRESETS}
_SESSION = "locomotion_console_diag"
_TASK_MARKER = "__LOCOMOTION_CONSOLE_DIAGNOSTIC_TASKS__"
_PAYLOAD_DIAG_FRAMEWORKS = {"taili_amp_blind"}
_PAYLOAD_GLOB = "taili_blind_runtime_*"
_TASK_GENERIC_TOKENS = {
    "direct",
    "env",
    "isaac",
    "lab",
    "robot",
    "robotlab",
    "v0",
}
_TASK_STRONG_TOKENS = {"blind", "teacher"}
_JOINT_ORDER = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]
_LEG_ORDER = ["FL", "FR", "RL", "RR"]
_DIRECTION_STAGE_IDS = {"forward", "backward", "lateral", "yaw"}
_TERRAIN_ALIASES = {
    "flat": "flat",
    "plane": "flat",
    "slope": "slope",
    "slope_up": "slope",
    "uphill": "slope",
    "slope_down": "slope_inv",
    "downhill": "slope_inv",
    "slope_inv": "slope_inv",
    "rough": "rough",
    "boxes": "boxes",
    "box": "boxes",
    "stairs": "stairs",
    "stairs_up": "stairs_up",
    "stair": "stairs",
    "stairs_down": "stairs_down",
}
_MAX_PLAN_SECONDS = 18 * 60


def _task_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^A-Za-z0-9]+", value.lower())
        if token and token not in _TASK_GENERIC_TOKENS
    }


def _clamp_float(value: Any, low: float, high: float, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    if not math.isfinite(result):
        result = default
    return max(low, min(high, result))


def _clamp_int(value: Any, low: int, high: int, default: int = 0) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    return max(low, min(high, result))


def _mode_for_command(vx: float, vy: float, wz: float, fallback: str = "") -> str:
    fallback = str(fallback or "").strip().lower()
    if fallback and fallback not in {"unknown", "custom"}:
        return fallback
    if abs(vx) <= 0.1 and abs(vy) <= 0.1 and abs(wz) <= 0.1:
        return "stand"
    if abs(wz) > max(abs(vx), abs(vy), 0.1):
        return "yaw"
    if abs(vy) > max(abs(vx), 0.1):
        return "lateral"
    return "forward" if vx >= 0 else "backward"


def _direction_command(stage_id: str) -> dict[str, Any]:
    if stage_id == "backward":
        return {"mode": "backward", "vx": -0.4, "vy": 0.0, "wz": 0.0, "duration_s": 3.0}
    if stage_id == "lateral":
        return {"mode": "lateral", "vx": 0.0, "vy": 0.3, "wz": 0.0, "duration_s": 3.0}
    if stage_id == "yaw":
        return {"mode": "yaw", "vx": 0.0, "vy": 0.0, "wz": 0.6, "duration_s": 3.0}
    return {"mode": "forward", "vx": 0.5, "vy": 0.0, "wz": 0.0, "duration_s": 3.0}


def _default_plan_for_preset(preset: _Preset) -> dict[str, Any]:
    commands: list[dict[str, Any]] = []
    terrains = [{"type": "flat", "level": 0, "params": {}}]
    dr_cases = [{"level": 0, "label": "DR0"}]
    pushes: list[dict[str, Any]] = []
    if preset.id == "quick":
        commands = [
            {"id": "stand", "label": "Stand", "mode": "stand", "vx": 0.0, "vy": 0.0, "wz": 0.0, "duration_s": 1.0, "repeats": 1},
            {"id": "forward", "label": "Forward 0.3", "mode": "forward", "vx": 0.3, "vy": 0.0, "wz": 0.0, "duration_s": 1.0, "repeats": 1},
        ]
    elif preset.id in _DIRECTION_STAGE_IDS:
        moving = _direction_command(preset.id)
        commands = [
            {"id": "stand", "label": "Stand baseline", "mode": "stand", "vx": 0.0, "vy": 0.0, "wz": 0.0, "duration_s": 1.0, "repeats": 1},
            {"id": preset.id, "label": preset.label, **moving, "repeats": 1},
        ]
    elif preset.id == "directions":
        commands = [
            {"id": "stand", "label": "Stand baseline", "mode": "stand", "vx": 0.0, "vy": 0.0, "wz": 0.0, "duration_s": 1.0, "repeats": 1},
            {"id": "forward", "label": "Forward", **_direction_command("forward"), "settle_s": 0.4, "repeats": 1},
            {"id": "backward", "label": "Backward", **_direction_command("backward"), "settle_s": 0.4, "repeats": 1},
            {"id": "lateral", "label": "Lateral", **_direction_command("lateral"), "settle_s": 0.4, "repeats": 1},
            {"id": "yaw", "label": "Yaw", **_direction_command("yaw"), "settle_s": 0.4, "repeats": 1},
        ]
    elif preset.id == "terrain":
        commands = [
            {"id": "stand", "label": "Stand", "mode": "stand", "vx": 0.0, "vy": 0.0, "wz": 0.0, "duration_s": 0.8, "repeats": 1},
            {"id": "forward", "label": "Forward", "mode": "forward", "vx": 0.5, "vy": 0.0, "wz": 0.0, "duration_s": 2.0, "repeats": 1},
        ]
        terrains = [
            {"type": "flat", "level": 0, "params": {}},
            {"type": "slope", "level": 5, "params": {}},
            {"type": "rough", "level": 5, "params": {}},
            {"type": "stairs_up", "level": 5, "params": {"direction": "up", "step_height": 0.18}},
        ]
    elif preset.id == "dr":
        commands = [
            {"id": "stand", "label": "Stand", "mode": "stand", "vx": 0.0, "vy": 0.0, "wz": 0.0, "duration_s": 0.8, "repeats": 1},
            {"id": "forward", "label": "Forward", "mode": "forward", "vx": 0.5, "vy": 0.0, "wz": 0.0, "duration_s": 2.0, "repeats": 1},
        ]
        dr_cases = [{"level": 0, "label": "DR0"}, {"level": 3, "label": "DR3"}]
    elif preset.id == "push":
        commands = [
            {"id": "stand", "label": "Stand", "mode": "stand", "vx": 0.0, "vy": 0.0, "wz": 0.0, "duration_s": 2.0, "repeats": 1},
            {"id": "forward", "label": "Forward", "mode": "forward", "vx": 0.5, "vy": 0.0, "wz": 0.0, "duration_s": 2.0, "repeats": 1},
        ]
        pushes = [{"enabled": True, "segment": 0, "time_s": 1.0, "vector": [0.0, 1.0, 0.0]}]
    else:
        commands = [
            {"id": "stand", "label": "Stand", "mode": "stand", "vx": 0.0, "vy": 0.0, "wz": 0.0, "duration_s": 1.0, "repeats": 1},
        ]
    return {
        "name": preset.label,
        "template": preset.id,
        "init_phase": 0,
        "num_envs": 1,
        "reset_policy": "per_case",
        "reset_initialization": "command_start",
        "commands": commands,
        "terrains": terrains,
        "dr_cases": dr_cases,
        "pushes": pushes,
        "recording": {"sample_hz": 50.0, "max_rows": 5000, "save_playback": True},
        "criteria": {
            "tracking_error_max": 0.20,
            "min_height_m": 0.47,
            "stable_fraction_min": 0.90,
            "slip_max": 0.20,
            "hard_impact_max": 0,
        },
        "notes": ["Generated from the selected preset; edit before launch if the test intent is different."],
    }


def _merge_plan(default: Any, override: Any) -> Any:
    if override is None:
        return default
    if isinstance(default, dict) and isinstance(override, dict):
        result = dict(default)
        for key, value in override.items():
            if value is None:
                continue
            result[key] = _merge_plan(result.get(key), value)
        return result
    return override


def _plan_to_dict(plan: DiagnosticPlan | dict[str, Any] | None) -> dict[str, Any]:
    if plan is None:
        return {}
    if isinstance(plan, DiagnosticPlan):
        if hasattr(plan, "model_dump"):
            return plan.model_dump()
        return plan.dict()
    if hasattr(plan, "model_dump"):
        return plan.model_dump()
    if hasattr(plan, "dict"):
        return plan.dict()
    if isinstance(plan, dict):
        return dict(plan)
    return {}


def _normalize_plan(preset: _Preset, requested_plan: DiagnosticPlan | dict[str, Any] | None = None) -> dict[str, Any]:
    raw = _merge_plan(_default_plan_for_preset(preset), _plan_to_dict(requested_plan))
    commands_raw = raw.get("commands") if isinstance(raw.get("commands"), list) else []
    commands: list[dict[str, Any]] = []
    total_command_seconds = 0.0
    for index, cmd in enumerate(commands_raw):
        if not isinstance(cmd, dict):
            continue
        vx = _clamp_float(cmd.get("vx"), -1.5, 1.5, 0.0)
        vy = _clamp_float(cmd.get("vy"), -0.7, 0.7, 0.0)
        wz = _clamp_float(cmd.get("wz"), -1.5, 1.5, 0.0)
        duration = _clamp_float(cmd.get("duration_s", cmd.get("duration")), 0.2, 30.0, 3.0)
        settle = _clamp_float(cmd.get("settle_s"), 0.0, 10.0, 0.0)
        ramp = _clamp_float(cmd.get("ramp_s"), 0.0, 10.0, 0.0)
        repeats = _clamp_int(cmd.get("repeats"), 1, 10, 1)
        mode = _mode_for_command(vx, vy, wz, str(cmd.get("mode") or ""))
        total_command_seconds += (duration + settle) * repeats
        commands.append(
            {
                "id": str(cmd.get("id") or mode or f"cmd_{index}"),
                "label": str(cmd.get("label") or cmd.get("id") or mode or f"Command {index + 1}"),
                "mode": mode,
                "vx": vx,
                "vy": vy,
                "wz": wz,
                "duration_s": duration,
                "settle_s": settle,
                "ramp_s": ramp,
                "repeats": repeats,
            }
        )
    if not commands:
        commands = _default_plan_for_preset(preset)["commands"]
        total_command_seconds = sum((float(c.get("duration_s", 1.0)) + float(c.get("settle_s", 0.0))) * int(c.get("repeats", 1)) for c in commands)

    terrains_raw = raw.get("terrains") if isinstance(raw.get("terrains"), list) else []
    terrains: list[dict[str, Any]] = []
    for item in terrains_raw:
        if not isinstance(item, dict):
            continue
        params = item.get("params") if isinstance(item.get("params"), dict) else {}
        requested_type = str(item.get("type") or "flat").strip().lower()
        direction = str(params.get("direction") or "").strip().lower()
        if requested_type == "stairs" and direction == "up":
            requested_type = "stairs_up"
        elif requested_type == "stairs" and direction == "down":
            requested_type = "stairs_down"
        terrain_type = _TERRAIN_ALIASES.get(requested_type, "flat")
        normalized_params = {str(key): value for key, value in params.items()}
        if terrain_type == "stairs_up":
            normalized_params["direction"] = "up"
        elif terrain_type == "stairs_down":
            normalized_params["direction"] = "down"
        if terrain_type in {"stairs", "stairs_up", "stairs_down"}:
            normalized_params["step_height"] = _clamp_float(normalized_params.get("step_height"), 0.02, 0.40, 0.18)
        terrains.append(
            {
                "type": terrain_type,
                "level": _clamp_int(item.get("level"), 0, 9, 0),
                "params": normalized_params,
            }
        )
    if not terrains:
        terrains = [{"type": "flat", "level": 0, "params": {}}]

    dr_raw = raw.get("dr_cases") if isinstance(raw.get("dr_cases"), list) else []
    dr_cases: list[dict[str, Any]] = []
    for item in dr_raw:
        if not isinstance(item, dict):
            continue
        case = {
            "level": _clamp_int(item.get("level"), 0, 3, 0),
            "label": str(item.get("label") or f"DR{_clamp_int(item.get('level'), 0, 3, 0)}"),
        }
        for key, low, high in [
            ("friction", 0.2, 2.0),
            ("mass_scale", 0.5, 1.8),
            ("stiffness_scale", 0.5, 1.5),
            ("damping_scale", 0.5, 1.5),
        ]:
            if item.get(key) is not None:
                case[key] = _clamp_float(item.get(key), low, high, 1.0)
        if item.get("latency_steps") is not None:
            case["latency_steps"] = _clamp_int(item.get("latency_steps"), 0, 8, 0)
        dr_cases.append(case)
    if not dr_cases:
        dr_cases = [{"level": 0, "label": "DR0"}]

    pushes_raw = raw.get("pushes") if isinstance(raw.get("pushes"), list) else []
    pushes: list[dict[str, Any]] = []
    for item in pushes_raw:
        if not isinstance(item, dict):
            continue
        vector_raw = item.get("vector") if isinstance(item.get("vector"), list) else [0.0, 0.0, 0.0]
        vector = [
            _clamp_float(vector_raw[i] if i < len(vector_raw) else 0.0, -3.0, 3.0, 0.0)
            for i in range(3)
        ]
        pushes.append(
            {
                "enabled": bool(item.get("enabled", False)),
                "segment": _clamp_int(item.get("segment"), 0, max(0, len(commands) - 1), 0),
                "time_s": _clamp_float(item.get("time_s"), 0.0, 30.0, 1.0),
                "vector": vector,
            }
        )

    recording_raw = raw.get("recording") if isinstance(raw.get("recording"), dict) else {}
    criteria_raw = raw.get("criteria") if isinstance(raw.get("criteria"), dict) else {}
    case_count = max(1, len(terrains) * len(dr_cases))
    estimated_seconds = total_command_seconds * case_count
    if estimated_seconds > _MAX_PLAN_SECONDS:
        raise ValueError(
            f"Diagnostic plan is too large ({estimated_seconds:.0f}s before IsaacLab startup); "
            f"reduce commands, repeats, terrains, or DR cases below {_MAX_PLAN_SECONDS}s."
        )
    return {
        "name": str(raw.get("name") or preset.label),
        "template": str(raw.get("template") or preset.id),
        "init_phase": _clamp_int(raw.get("init_phase"), 0, 9, 0),
        "num_envs": _clamp_int(raw.get("num_envs"), 1, 64, 1),
        "reset_policy": str(raw.get("reset_policy") or "per_case"),
        "reset_initialization": str(raw.get("reset_initialization") or "command_start"),
        "commands": commands,
        "terrains": terrains,
        "dr_cases": dr_cases,
        "pushes": pushes,
        "recording": {
            "sample_hz": _clamp_float(recording_raw.get("sample_hz"), 10.0, 100.0, 50.0),
            "max_rows": _clamp_int(recording_raw.get("max_rows"), 100, 200000, 5000),
            "save_playback": bool(recording_raw.get("save_playback", True)),
        },
        "criteria": {
            "tracking_error_max": _clamp_float(criteria_raw.get("tracking_error_max"), 0.02, 2.0, 0.20),
            "min_height_m": _clamp_float(criteria_raw.get("min_height_m"), 0.20, 0.80, 0.47),
            "stable_fraction_min": _clamp_float(criteria_raw.get("stable_fraction_min"), 0.0, 1.0, 0.90),
            "slip_max": _clamp_float(criteria_raw.get("slip_max"), 0.0, 2.0, 0.20),
            "hard_impact_max": _clamp_int(criteria_raw.get("hard_impact_max"), 0, 10000, 0),
        },
        "estimated_rollout_s": estimated_seconds,
        "notes": [str(item) for item in (raw.get("notes") if isinstance(raw.get("notes"), list) else [])],
    }


def _plan_summary(plan: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(plan, dict):
        return {}
    commands = plan.get("commands") if isinstance(plan.get("commands"), list) else []
    terrains = plan.get("terrains") if isinstance(plan.get("terrains"), list) else []
    dr_cases = plan.get("dr_cases") if isinstance(plan.get("dr_cases"), list) else []
    pushes = plan.get("pushes") if isinstance(plan.get("pushes"), list) else []
    modes = []
    for command in commands:
        if isinstance(command, dict):
            mode = str(command.get("mode") or "")
            if mode and mode not in modes:
                modes.append(mode)
    return {
        "name": plan.get("name", ""),
        "template": plan.get("template", ""),
        "init_phase": plan.get("init_phase", 0),
        "commands": len(commands),
        "modes": modes,
        "terrains": [item.get("type") for item in terrains if isinstance(item, dict)],
        "dr_levels": [item.get("level") for item in dr_cases if isinstance(item, dict)],
        "pushes": sum(1 for item in pushes if isinstance(item, dict) and item.get("enabled")),
        "estimated_rollout_s": plan.get("estimated_rollout_s"),
    }


def _plan_to_suite_yaml(plan: dict[str, Any], suite_name: str) -> str:
    import yaml

    commands: list[dict[str, Any]] = []
    for command in plan.get("commands", []):
        if not isinstance(command, dict):
            continue
        for repeat in range(int(command.get("repeats", 1) or 1)):
            if float(command.get("settle_s", 0.0) or 0.0) > 0:
                commands.append(
                    {
                        "mode": "stand",
                        "vx": 0.0,
                        "vy": 0.0,
                        "wz": 0.0,
                        "duration": float(command.get("settle_s", 0.0) or 0.0),
                        "label": f"settle_before_{command.get('id') or command.get('mode')}_{repeat + 1}",
                        "source_command_id": command.get("id", ""),
                    }
                )
            commands.append(
                {
                    "mode": str(command.get("mode") or "stand"),
                    "vx": float(command.get("vx", 0.0) or 0.0),
                    "vy": float(command.get("vy", 0.0) or 0.0),
                    "wz": float(command.get("wz", 0.0) or 0.0),
                    "duration": float(command.get("duration_s", 1.0) or 1.0),
                    "ramp": float(command.get("ramp_s", 0.0) or 0.0),
                    "label": str(command.get("label") or command.get("mode") or "command"),
                    "source_command_id": command.get("id", ""),
                    "repeat": repeat + 1,
                }
            )
    payload = {
        "name": suite_name,
        "num_envs": int(plan.get("num_envs", 1) or 1),
        "init_phase": int(plan.get("init_phase", 0) or 0),
        "reset_policy": str(plan.get("reset_policy") or "per_case"),
        "reset_initialization": str(plan.get("reset_initialization") or "command_start"),
        "terrains": plan.get("terrains") or [{"type": "flat", "level": 0}],
        "dr_cases": plan.get("dr_cases") or [{"level": 0}],
        "commands": commands or [{"mode": "stand", "vx": 0.0, "vy": 0.0, "wz": 0.0, "duration": 1.0}],
    }
    enabled_pushes = [item for item in plan.get("pushes", []) if isinstance(item, dict) and item.get("enabled")]
    if enabled_pushes:
        payload["pushes"] = {
            "enabled": True,
            "events": [
                {
                    "segment": int(item.get("segment", 0) or 0),
                    "time": float(item.get("time_s", 0.0) or 0.0),
                    "vector": item.get("vector") or [0.0, 0.0, 0.0],
                }
                for item in enabled_pushes
            ],
        }
    return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)


def _select_diagnostic_task_from_candidates(
    configured_task: str,
    framework_id: str,
    candidates: list[str],
) -> tuple[str, str]:
    """Return a remote-registered diagnostic task, resolving stale local names.

    The local framework profile is a desired/default task, not a source of truth.
    The remote Gym registry is authoritative because RobotLab task ids differ
    across branches (for example, some blind tasks omit the training framework
    token such as "AMP").
    """
    unique = sorted({item for item in candidates if item})
    if configured_task in unique:
        return configured_task, ""
    if not unique:
        raise RuntimeError(
            f"Remote Gym registry did not report any diagnostic tasks while validating {configured_task!r}"
        )

    configured_tokens = _task_tokens(configured_task)
    framework_tokens = _task_tokens(framework_id.replace("_", "-"))
    desired_tokens = configured_tokens | framework_tokens

    # Strong semantic tokens must not silently cross framework families:
    # blind checkpoints must not resolve to teacher tasks and vice versa.
    required_strong = sorted(desired_tokens & _TASK_STRONG_TOKENS)
    viable = [
        candidate
        for candidate in unique
        if all(token in _task_tokens(candidate) for token in required_strong)
    ] or unique

    def score(candidate: str) -> float:
        tokens = _task_tokens(candidate)
        shared = tokens & desired_tokens
        result = 100.0 * SequenceMatcher(None, configured_task.lower(), candidate.lower()).ratio()
        result += 14.0 * len(shared)
        if "taili" in desired_tokens and "taili" in tokens:
            result += 30.0
        for token in required_strong:
            if token in tokens:
                result += 45.0
        for token in _TASK_STRONG_TOKENS - set(required_strong):
            if token in tokens:
                result -= 60.0
        # "AMP" is often a training/checkpoint family rather than a Gym task id.
        # Keep it as a mild positive match, not a hard requirement.
        if "amp" in desired_tokens and "amp" in tokens:
            result += 8.0
        return result

    ranked = sorted(((score(candidate), candidate) for candidate in viable), reverse=True)
    best_score, best = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else -1.0
    if best_score < 55.0 or best_score - second_score < 3.0:
        preview = ", ".join(unique[:12])
        if len(unique) > 12:
            preview += ", ..."
        raise RuntimeError(
            "Configured diagnostic task is not registered on the remote host "
            f"({configured_task!r}), and no unambiguous replacement could be selected. "
            f"Remote candidates: {preview}"
        )
    return best, f"resolved remote diagnostic task {configured_task!r} -> {best!r}"


class DiagnosticsController:
    def __init__(self, settings: LocomotionConsoleSettings, source: Any):
        self.settings = settings
        self.source = source
        self.framework = get_framework_profile(settings.framework_id)
        self.config_set = get_active_config_set(settings)
        self.history = DiagnosticHistoryStore(settings)
        self._job: _Job | None = None

    async def catalog(self) -> DiagnosticCatalog:
        available = True
        message = ""
        if self.settings.source == "fake":
            checkpoints = [
                DiagnosticCheckpoint(
                    path=f"fake/agent_{iteration}.pt",
                    name=f"agent_{iteration}.pt",
                    run_name="fake_2026-06-28",
                    iteration=iteration,
                    kind="latest" if iteration == 70000 else "iteration",
                    is_default=iteration == 70000,
                    note="latest saved iteration" if iteration == 70000 else "older saved iteration",
                    framework_id=self.framework.id,
                    framework_label=self.framework.label,
                    framework_status=self.framework.status,
                    diagnostic_task=self.framework.diagnostic_task,
                )
                for iteration in [70000, 65000, 60000]
            ]
            training_running = False
        else:
            try:
                remote = self.source._get_remote()
                checkpoints = await asyncio.to_thread(self._recent_checkpoints, remote)
                training_running = await asyncio.to_thread(self.source._is_running, remote)
                if not checkpoints:
                    message = "Remote is reachable, but no diagnostic checkpoint was found."
            except Exception as exc:  # noqa: BLE001 - remote power/network is operator-controlled
                checkpoints = []
                training_running = False
                available = False
                message = f"Remote is unavailable; checkpoint catalog cannot be refreshed. {type(exc).__name__}: {exc}"
        checkpoint = checkpoints[0].path if checkpoints else ""
        return DiagnosticCatalog(
            presets=[
                DiagnosticPreset(
                    id=p.id,
                    label=p.label,
                    description=p.description,
                    category=p.category,
                    estimated_minutes=p.estimated_minutes,
                )
                for p in PRESETS
            ],
            plan_templates={p.id: _default_plan_for_preset(p) for p in PRESETS},
            checkpoints=checkpoints,
            checkpoint=checkpoint or None,
            checkpoint_name=checkpoint.rsplit("/", 1)[-1] if checkpoint else None,
            training_running=training_running,
            source="real" if self.settings.source == "real" else "fake",
            framework_id=self.framework.id,
            framework_label=self.framework.label,
            framework_status=self.framework.status,
            framework_note=self.framework.note,
            available=available,
            message=message,
        )

    async def start(
        self,
        preset_id: str,
        requested_checkpoint: str | None = None,
        requested_plan: DiagnosticPlan | dict[str, Any] | None = None,
    ) -> DiagnosticJobStatus:
        preset = _PRESET_BY_ID.get(preset_id)
        if preset is None:
            raise ValueError(f"Unknown diagnostic preset: {preset_id}")
        plan = _normalize_plan(preset, requested_plan)
        current = await self.status()
        if current.state in {"starting", "running"}:
            raise RuntimeError("A diagnostic job is already running")

        if self.settings.source == "fake":
            available = (await self.catalog()).checkpoints
            paths = {item.path for item in available}
            checkpoint = requested_checkpoint or available[0].path
            if checkpoint not in paths:
                raise ValueError("Selected checkpoint is not in the available checkpoint list")
        else:
            training_running = False
            try:
                remote = self.source._get_remote()
                training_running = await asyncio.to_thread(self.source._is_running, remote)
                available = await asyncio.to_thread(self._recent_checkpoints, remote)
            except RuntimeError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Remote is unavailable; diagnostics cannot start. {type(exc).__name__}: {exc}") from exc
            if not available:
                raise RuntimeError("No checkpoint was found on the remote host")
            paths = {item.path for item in available}
            if (requested_checkpoint or "").strip().lower() == "best":
                # the BEST_CHECKPOINT.json registry is maintained by acceptance_run whenever a
                # measured score improves — diagnose the best-known policy, not merely the newest
                # (which once picked an under-converged fragment of a stalled run).
                raw = await asyncio.to_thread(
                    remote.exec_out,
                    "cat /root/gpufree-data/taili_runs/BEST_CHECKPOINT.json 2>/dev/null")
                try:
                    best = str((json.loads(raw or "{}") or {}).get("checkpoint") or "")
                except Exception:
                    best = ""
                if not best:
                    raise RuntimeError("no BEST_CHECKPOINT registry yet — run an acceptance measurement first")
                exists = await asyncio.to_thread(
                    remote.exec_out, f"[ -f {shlex.quote(best)} ] && echo yes || echo no")
                if (exists or "").strip() != "yes":
                    raise RuntimeError(f"registered best checkpoint is missing on disk: {best}")
                checkpoint = best
            else:
                checkpoint = requested_checkpoint or available[0].path
                if checkpoint not in paths:
                    raise ValueError("Selected checkpoint is not in the recent checkpoint list; refresh and retry")
            if training_running:
                # Training + a diagnostic are two Isaac procs sharing the box RAM cgroup + one GPU. Block ONLY on
                # a real OOM/VRAM risk; otherwise run the diagnostic CONCURRENTLY (operator wants mid-training probes
                # whenever there is space). Headroom = cgroup RAM free AND GPU VRAM free vs the num_envs footprint.
                n_env = int((plan or {}).get("num_envs", 1) or 1)
                _ok, _detail = await probe_concurrent_headroom(remote, n_env)
                if not _ok:
                    raise RuntimeError(
                        f"训练运行中且余量不足({_detail})——诊断暂停以避免 OOM/显存争用同时杀掉训练和诊断。"
                        f"降低 num_envs,或先停训练。")
                # enough headroom → let the diagnostic run concurrently with training

        job_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = f"{self.settings.diagnostic_output_root}/locomotion_console_{job_id}_{preset.id}"
        selected_checkpoint = next((item for item in available if item.path == checkpoint), None)
        framework_id = selected_checkpoint.framework_id if selected_checkpoint else self.framework.id
        diagnostic_task = selected_checkpoint.diagnostic_task if selected_checkpoint else self.framework.diagnostic_task
        task_note = ""
        if self.settings.source == "real":
            try:
                diagnostic_task, task_note = await asyncio.to_thread(
                    self._resolve_remote_diagnostic_task,
                    remote,
                    diagnostic_task or self.settings.diagnostic_task,
                    framework_id,
                )
            except RuntimeError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Remote diagnostic task validation failed. {type(exc).__name__}: {exc}"
                ) from exc

        self._job = _Job(job_id, preset, checkpoint, framework_id, diagnostic_task, output_dir, time.time(), plan=plan)

        if self.settings.source == "real":
            try:
                remote = self.source._get_remote()
                await asyncio.to_thread(remote.exec_out, f"mkdir -p {shlex.quote(output_dir)}")
                await asyncio.to_thread(self._write_remote_plan_artifacts, remote, self._job)
                command = await asyncio.to_thread(self._install_remote_run_script, remote, self._job)
                await asyncio.to_thread(self.source._launch_tmux, remote, _SESSION, command)
            except Exception as exc:  # noqa: BLE001
                self._job.state = "error"
                self._job.message = f"Remote is unavailable; diagnostics did not start. {type(exc).__name__}: {exc}"
                return await self.status()
        self._job.state = "running"
        if task_note:
            self._job.message = f"{self._job.message} {task_note}".strip()
        if not self._job.message:
            self._job.message = "Diagnostic job started"
        return await self.status()

    async def cancel(self) -> DiagnosticJobStatus:
        if self._job is None:
            return DiagnosticJobStatus(state="idle", message="No diagnostic job is running")
        if self.settings.source == "real":
            try:
                remote = self.source._get_remote()
                await asyncio.to_thread(
                    remote.exec_out,
                    f"tmux kill-session -t {_SESSION} 2>/dev/null; true",
                )
            except Exception as exc:  # noqa: BLE001
                self._job.state = "error"
                self._job.message = f"Remote is unavailable; cancel command was not confirmed. {type(exc).__name__}: {exc}"
                return await self.status()
        self._job.state = "cancelled"
        self._job.message = "Diagnostic job cancelled"
        return await self.status()

    async def status(self) -> DiagnosticJobStatus:
        self._restore_latest_job_if_needed()
        job = self._job
        if job is None:
            return DiagnosticJobStatus(state="idle", message="No diagnostic job has been started")
        elapsed = max(0.0, time.time() - job.started_at)
        if self.settings.source == "fake":
            return self._record_status(self._fake_status(job, elapsed))

        if job.state in {"complete", "cancelled", "error"}:
            if self.settings.source == "real" and job.state in {"complete", "error"}:
                # A diagnostic tmux session can disappear between polls after the
                # recorder has already written record.csv + metrics.json.  Do not
                # freeze the UI in a stale "exited early" state; let the remote
                # artifacts be authoritative.
                try:
                    remote = self.source._get_remote()
                    recovered_stages = await asyncio.to_thread(self._read_stage_statuses, remote, job)
                    if job.state == "complete" or all(stage.state == "complete" for stage in recovered_stages):
                        job.state = "complete"
                        job.message = "Diagnostics and report generation completed"
                        log_tail = await asyncio.to_thread(self._read_log_tail, remote, job)
                        return self._record_status(
                            self._status_model(job, recovered_stages, 1.0, elapsed, log_tail)
                        )
                except Exception:
                    pass
            stage_state = "complete" if job.state == "complete" else job.state
            stage_statuses = [
                DiagnosticStageStatus(
                    id=stage.id,
                    label=stage.label,
                    state=stage_state,  # type: ignore[arg-type]
                    progress=1.0 if job.state == "complete" else 0.0,
                    rows_written=0,
                )
                for stage in job.preset.stages
            ]
            return self._record_status(
                self._status_model(
                    job,
                    stage_statuses,
                    1.0 if job.state == "complete" else 0.0,
                    elapsed,
                    [],
                )
            )

        try:
            remote = self.source._get_remote()
            stage_statuses = await asyncio.to_thread(self._read_stage_statuses, remote, job)
            complete = all(stage.state == "complete" for stage in stage_statuses)
            failed = any(stage.state == "error" for stage in stage_statuses)
            active = await asyncio.to_thread(
                remote.exec_out,
                f"tmux has-session -t {_SESSION} 2>/dev/null && echo active || true",
            )
            log_tail = await asyncio.to_thread(self._read_log_tail, remote, job)
        except Exception as exc:  # noqa: BLE001
            job.state = "error"
            job.message = f"Remote is unavailable while reading diagnostic status. {type(exc).__name__}: {exc}"
            stage_statuses = [
                DiagnosticStageStatus(id=stage.id, label=stage.label, state="error", progress=0.0, rows_written=0)
                for stage in job.preset.stages
            ]
            return self._record_status(self._status_model(job, stage_statuses, 0.0, elapsed, []))
        if complete:
            job.state = "complete"
            job.message = "Diagnostics and report generation completed"
        elif job.state == "cancelled":
            pass
        elif failed:
            job.state = "error"
            error_message = await asyncio.to_thread(self._read_stage_error_message, remote, job)
            job.message = error_message or (
                "Remote diagnostic failed; see record_error.json and job.log in the output directory."
            )
        elif not (active or "").strip():
            job.state = "error"
            error_message = await asyncio.to_thread(self._read_stage_error_message, remote, job)
            job.message = error_message or (
                "Remote diagnostic process is gone before the report was complete. "
                "If the remote machine/container was restarted, this job was interrupted and must be rerun."
            )
        else:
            job.state = "running"
            running_stage = next((s.label for s in stage_statuses if s.state == "running"), "")
            status_message = f"Running: {running_stage}" if running_stage else "Preparing remote environment"
            if "GPU contention" in job.message or "training is running" in job.message:
                job.message = f"{status_message}. {job.message}"
            else:
                job.message = status_message
        progress = sum(stage.progress for stage in stage_statuses) / max(1, len(stage_statuses))
        return self._record_status(self._status_model(job, stage_statuses, progress, elapsed, log_tail))

    async def report(self) -> DiagnosticReport:
        return await self.report_for_job()

    async def report_for_job(self, job_id: str | None = None) -> DiagnosticReport:
        job = self._job_for_artifact(job_id)
        if job is None:
            raise RuntimeError("Diagnostics are not complete yet")
        if not job_id:
            status = await self.status()
            if status.state != "complete":
                raise RuntimeError("Diagnostics are not complete yet")
            job = self._job
        if job is None or job.state != "complete":
            if self.settings.source != "real" or job is None:
                raise RuntimeError("Diagnostics are not complete yet")
            try:
                remote = self.source._get_remote()
                stage_statuses = await asyncio.to_thread(self._read_stage_statuses, remote, job)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Remote is unavailable; diagnostic report cannot be refreshed. {type(exc).__name__}: {exc}"
                ) from exc
            if not all(stage.state == "complete" for stage in stage_statuses):
                raise RuntimeError("Diagnostics are not complete yet")
            job.state = "complete"
            job.message = "Diagnostics and report generation completed"
            self._record_status(self._status_model(job, stage_statuses, 1.0, time.time() - job.started_at, []))
        if self.settings.source == "fake":
            return _fake_report(job)
        try:
            remote = self.source._get_remote()
            metrics, log = await asyncio.to_thread(self._read_report_inputs, remote, job)
            plan = await asyncio.to_thread(self._read_remote_plan, remote, job)
            return _normalize_report(job, metrics, log=log, artifacts=self._report_artifacts(remote, job, log), plan=plan)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Remote is unavailable; diagnostic report cannot be read. {type(exc).__name__}: {exc}"
            ) from exc

    async def playback(self, max_frames: int = 900) -> DiagnosticPlayback:
        return await self.playback_for_job(max_frames=max_frames)

    async def playback_for_job(
        self,
        max_frames: int = 900,
        job_id: str | None = None,
    ) -> DiagnosticPlayback:
        """Return downsampled rollout frames from diagnostic record.csv files."""
        max_frames = max(1, min(int(max_frames or 900), 4000))
        job = self._job_for_artifact(job_id)
        if job is None:
            return DiagnosticPlayback(
                available=False,
                source="real" if self.settings.source == "real" else "fake",
                message="No diagnostic job has been started; no record.csv is available for playback.",
                joint_order=_JOINT_ORDER,
                leg_order=_LEG_ORDER,
            )
        if self.settings.source == "fake":
            return _fake_playback(job, max_frames=max_frames)
        try:
            remote = self.source._get_remote()
            stage_records = await asyncio.to_thread(self._read_playback_records, remote, job)
        except Exception as exc:  # noqa: BLE001
            return DiagnosticPlayback(
                available=False,
                source="real",
                message=f"Remote is unavailable; playback cannot be read. {type(exc).__name__}: {exc}",
                output_dir=job.output_dir,
                joint_order=_JOINT_ORDER,
                leg_order=_LEG_ORDER,
            )
        return _playback_from_record_texts(
            job,
            stage_records,
            source="real",
            max_frames=max_frames,
        )

    def _latest_checkpoint(self, remote: Any) -> str:
        checkpoints = self._recent_checkpoints(remote)
        return checkpoints[0].path if checkpoints else ""

    def _recent_checkpoints(self, remote: Any, limit: int = 20) -> list[DiagnosticCheckpoint]:
        checkpoints: list[DiagnosticCheckpoint] = []
        seen: set[str] = set()
        for framework in list_framework_profiles():
            for item in self._recent_checkpoints_for_framework(remote, framework, limit):
                if item.path in seen:
                    continue
                seen.add(item.path)
                checkpoints.append(item)
        latest_by_run: dict[tuple[str, str], int] = {}
        for item in checkpoints:
            if item.iteration is None:
                continue
            key = (item.framework_id, item.run_name)
            latest_by_run[key] = max(latest_by_run.get(key, -1), item.iteration)

        normalized: list[DiagnosticCheckpoint] = []
        for item in checkpoints:
            kind = item.kind
            note = item.note
            if item.name == "best_agent.pt":
                kind = "best"
                note = note or (
                    "best_agent.pt is a training-selected snapshot; curriculum runs may leave it behind the latest phase."
                )
            elif item.iteration is not None:
                latest = latest_by_run.get((item.framework_id, item.run_name), item.iteration)
                kind = "latest" if item.iteration == latest else "iteration"
                note = note or ("latest saved iteration for this run" if kind == "latest" else "older saved iteration")
            normalized.append(item.model_copy(update={"kind": kind, "note": note, "is_default": False}))
        checkpoints = normalized
        checkpoints.sort(
            key=lambda item: (
                0 if item.kind == "latest" else 1 if item.kind == "iteration" else 2 if item.kind == "best" else 3,
                -(item.iteration or -1),
                item.framework_id,
                item.run_name,
                item.name,
            )
        )
        if checkpoints:
            checkpoints[0] = checkpoints[0].model_copy(update={"is_default": True})
        return checkpoints[:limit]

    def _recent_checkpoints_for_framework(
        self,
        remote: Any,
        framework: FrameworkProfile,
        limit: int = 20,
    ) -> list[DiagnosticCheckpoint]:
        roots = framework.checkpoint_roots or (
            f"{self.settings.diagnostic_robot_root}/logs/skrl/{framework.experiment}",
        )
        quoted_roots = " ".join(shlex.quote(root.rstrip("/")) for root in roots)
        command = (
            f"find -L {quoted_roots} -type f \\( -path '*/checkpoints/agent_*.pt' "
            f"-o -path '*/checkpoints/best_agent.pt' \\) "
            f"-printf '%T@|%p\\n' 2>/dev/null | sort -t'|' -k1,1nr | head -{int(max(limit * 3, limit))}"
        )
        return self._parse_checkpoint_lines((remote.exec_out(command) or "").strip(), framework)

    @staticmethod
    def _parse_checkpoint_lines(raw: str, framework: FrameworkProfile | None = None) -> list[DiagnosticCheckpoint]:
        import re

        result = []
        seen = set()
        for line in raw.splitlines():
            _, separator, path = line.partition("|")
            path = path.strip() if separator else line.strip()
            if not path or path in seen:
                continue
            seen.add(path)
            parts = path.rstrip("/").split("/")
            name = parts[-1]
            run_name = parts[-3] if len(parts) >= 3 and parts[-2] == "checkpoints" else ""
            match = re.fullmatch(r"agent_(\d+)\.pt", name)
            is_best = name == "best_agent.pt"
            result.append(
                DiagnosticCheckpoint(
                    path=path,
                    name=name,
                    run_name=run_name,
                    iteration=int(match.group(1)) if match else None,
                    kind="iteration" if match else "best" if is_best else "unknown",
                    note=(
                        "best_agent.pt is a training-selected snapshot; use latest agent_<step>.pt for phase-curriculum diagnostics unless intentionally comparing best."
                        if is_best
                        else ""
                    ),
                    framework_id=framework.id if framework else "",
                    framework_label=framework.label if framework else "",
                    framework_status=framework.status if framework else "",
                    diagnostic_task=framework.diagnostic_task if framework else "",
                )
            )
        return result

    def _build_remote_script(self, job: _Job) -> str:
        if job.framework_id in _PAYLOAD_DIAG_FRAMEWORKS:
            return self._build_payload_remote_script(job)
        return self._build_legacy_remote_script(job)

    def _payload_root_shell(self) -> str:
        roots = []
        root = str(self.settings.diagnostic_tool_root or "").rstrip("/")
        if root:
            roots.append(root)
        roots.append("/root/gpufree-data/training_payloads")
        unique = []
        for item in roots:
            if item and item not in unique:
                unique.append(item)
        clauses = " ".join(f"{shlex.quote(root.rstrip('/'))}/{_PAYLOAD_GLOB}" for root in unique)
        return (
            "PAYLOAD=''\n"
            f"for candidate in $(ls -td {clauses} 2>/dev/null || true); do\n"
            "  if [ -f \"$candidate/taili_blind_runtime/diagnose_taili_cases.py\" ] && [ -f \"$candidate/taili_blind_runtime/isaaclab_quad_diag/metrics.py\" ]; then\n"
            "    PAYLOAD=\"$candidate\"\n"
            "    break\n"
            "  fi\n"
            "  if [ -f \"$candidate/taili_blind_runtime/diagnose_taili_cases.py\" ]; then\n"
            "    echo \"[diagnostics] skipping incomplete payload without taili_blind_runtime/isaaclab_quad_diag/metrics.py: $candidate\" >&2\n"
            "  fi\n"
            "done\n"
            "if [ -z \"$PAYLOAD\" ]; then\n"
            "  echo '[diagnostics] compatible taili_blind_runtime payload was not found under configured payload roots; expected diagnose_taili_cases.py and isaaclab_quad_diag/metrics.py' >&2\n"
            "  exit 31\n"
            "fi\n"
            "echo \"[diagnostics] payload=$PAYLOAD\"\n"
            "export PAYLOAD\n"
            "export PYTHONPATH=\"$PAYLOAD:${PYTHONPATH:-}\"\n"
            "cd \"$PAYLOAD\"\n"
        )

    def _build_payload_remote_script(self, job: _Job) -> str:
        s = self.settings
        commands = [
            "set -e",
            self._payload_root_shell(),
            "if [ -L /tmp/IsaacLab ]; then mkdir -p \"$(readlink -f /tmp/IsaacLab)\"; else mkdir -p /tmp/IsaacLab; fi",
        ]
        for stage in job.preset.stages:
            stage_out = self._stage_output(job, stage)
            suite = f"{stage_out}/diagnostic_suite.yaml"
            commands.append(f"mkdir -p {shlex.quote(stage_out)}")
            stage_plan = self._stage_plan(job, stage)
            init_phase = _clamp_int(stage_plan.get("init_phase"), 0, 9, 0)
            diag_cmd = " ".join(
                [
                    f"TAILI_INIT_PHASE={init_phase}",
                    shlex.quote(s.diagnostic_python),
                    "-u",
                    "-m",
                    "taili_blind_runtime.diagnose_taili_cases",
                    "--task",
                    shlex.quote(job.diagnostic_task or s.diagnostic_task),
                    "--checkpoint",
                    shlex.quote(job.checkpoint),
                    "--suite",
                    shlex.quote(suite),
                    "--out",
                    shlex.quote(stage_out),
                    "--device",
                    "cuda:0",
                    "--headless",
                ]
            )
            commands.append(
                "\n".join(
                    [
                        "set +e",
                        " ".join(
                            [
                                "timeout",
                                "--preserve-status",
                                "20m",
                                "bash",
                                "-lc",
                                shlex.quote(diag_cmd),
                            ]
                        ),
                        "diag_rc=$?",
                        "set -e",
                        "echo \"[diagnostics] diagnostic command exit_code=$diag_rc\"",
                        f"if [ -s {shlex.quote(stage_out + '/record_error.json')} ]; then",
                        "  echo '[diagnostics] record_error.json was written by payload diagnostic' >&2",
                        "  exit 22",
                        "fi",
                        "if [ \"$diag_rc\" -ne 0 ]; then",
                        "  echo '[diagnostics] diagnostic command failed before metrics; see traceback above and record_error.json if present' >&2",
                        "  exit \"$diag_rc\"",
                        "fi",
                    ]
                )
            )
            record_csv = f"{stage_out}/record.csv"
            quoted_record_csv = shlex.quote(record_csv)
            commands.append(
                f"test -s {quoted_record_csv} && [ \"$(wc -l < {quoted_record_csv})\" -gt 1 ] "
                "|| (echo '[diagnostics] record.csv is empty or has no data rows; "
                "payload diagnostic failed before metrics' >&2; exit 20)"
            )
            commands.append(
                f"test -s {shlex.quote(stage_out + '/metrics/metrics.json')} "
                "|| (echo '[diagnostics] metrics.json is missing or empty after payload diagnostic' >&2; exit 21)"
            )
        body = "\n".join(commands)
        return "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -e",
                f"exec > {shlex.quote(job.output_dir + '/job.log')} 2>&1",
                "echo \"[diagnostics] payload-local remote run script started at $(date -Is)\"",
                body,
                "echo \"[diagnostics] remote run script completed at $(date -Is)\"",
                "",
            ]
        )

    def _build_legacy_remote_script(self, job: _Job) -> str:
        s = self.settings
        commands = [
            "set -e",
            f"cd {shlex.quote(s.diagnostic_robot_root)}",
            f"export PYTHONPATH={shlex.quote(s.diagnostic_tool_root)}:$PYTHONPATH",
            "if [ -L /tmp/IsaacLab ]; then mkdir -p \"$(readlink -f /tmp/IsaacLab)\"; else mkdir -p /tmp/IsaacLab; fi",
        ]
        for stage in job.preset.stages:
            stage_out = self._stage_output(job, stage)
            spec = f"{s.diagnostic_tool_root}/specs/taili.yaml"
            commands.append(f"mkdir -p {shlex.quote(stage_out)}")
            commands.append(
                " ".join(
                    [
                        shlex.quote(s.diagnostic_python),
                        "-c",
                        shlex.quote("from isaaclab_quad_diag.record import main; main()"),
                        "--task",
                        shlex.quote(job.diagnostic_task or s.diagnostic_task),
                        "--robot-spec",
                        shlex.quote(spec),
                        "--policy-backend",
                        "skrl",
                        "--checkpoint",
                        shlex.quote(job.checkpoint),
                        "--suite",
                        shlex.quote(f"{stage_out}/diagnostic_suite.yaml"),
                        "--out",
                        shlex.quote(stage_out),
                        "--headless",
                    ]
                )
            )
            record_csv = f"{stage_out}/record.csv"
            quoted_record_csv = shlex.quote(record_csv)
            commands.append(
                f"test -s {quoted_record_csv} && [ \"$(wc -l < {quoted_record_csv})\" -gt 1 ] "
                "|| (echo '[diagnostics] record.csv is empty or has no data rows; "
                "recorder failed before metrics' >&2; exit 20)"
            )
            commands.append(
                " ".join(
                    [
                        shlex.quote(s.diagnostic_python),
                        "-c",
                        shlex.quote("from isaaclab_quad_diag.metrics import main; main()"),
                        shlex.quote(f"{stage_out}/record.csv"),
                        "--out",
                        shlex.quote(f"{stage_out}/metrics"),
                    ]
                )
            )
        body = "\n".join(commands)
        return "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -e",
                f"exec > {shlex.quote(job.output_dir + '/job.log')} 2>&1",
                "echo \"[diagnostics] remote run script started at $(date -Is)\"",
                body,
                "echo \"[diagnostics] remote run script completed at $(date -Is)\"",
                "",
            ]
        )

    @staticmethod
    def _stage_plan(job: _Job, stage: _Stage) -> dict[str, Any]:
        plan = dict(job.plan or _default_plan_for_preset(job.preset))
        if len(job.preset.stages) <= 1 or stage.id == "directions":
            return plan
        commands = plan.get("commands") if isinstance(plan.get("commands"), list) else []
        filtered = []
        for command in commands:
            if not isinstance(command, dict):
                continue
            mode = str(command.get("mode") or "")
            cid = str(command.get("id") or "")
            if mode == "stand" or mode == stage.id or cid == stage.id:
                filtered.append(command)
        if not any(isinstance(item, dict) and str(item.get("mode") or "") == stage.id for item in filtered):
            filtered.append({**_direction_command(stage.id), "id": stage.id, "label": stage.label, "repeats": 1})
        plan["commands"] = filtered
        plan["name"] = f"{plan.get('name') or job.preset.label} / {stage.label}"
        plan["template"] = f"{plan.get('template') or job.preset.id}:{stage.id}"
        return plan

    def _write_remote_plan_artifacts(self, remote: Any, job: _Job) -> None:
        import os
        import yaml

        plan = job.plan or _default_plan_for_preset(job.preset)
        with tempfile.TemporaryDirectory() as tmp:
            plan_json = os.path.join(tmp, "diagnostic_plan.json")
            plan_yaml = os.path.join(tmp, "diagnostic_plan.yaml")
            with open(plan_json, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(plan, handle, ensure_ascii=False, indent=2)
            with open(plan_yaml, "w", encoding="utf-8", newline="\n") as handle:
                yaml.safe_dump(plan, handle, allow_unicode=True, sort_keys=False)
            remote.put(plan_json, f"{job.output_dir}/diagnostic_plan.json")
            remote.put(plan_yaml, f"{job.output_dir}/diagnostic_plan.yaml")
            for stage in job.preset.stages:
                stage_out = self._stage_output(job, stage)
                remote.exec_out(f"mkdir -p {shlex.quote(stage_out)}")
                stage_plan = self._stage_plan(job, stage)
                suite_path = os.path.join(tmp, f"{stage.id}_diagnostic_suite.yaml")
                plan_path = os.path.join(tmp, f"{stage.id}_diagnostic_plan.json")
                with open(suite_path, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(_plan_to_suite_yaml(stage_plan, f"{job.id}_{stage.id}"))
                with open(plan_path, "w", encoding="utf-8", newline="\n") as handle:
                    json.dump(stage_plan, handle, ensure_ascii=False, indent=2)
                remote.put(suite_path, f"{stage_out}/diagnostic_suite.yaml")
                remote.put(plan_path, f"{stage_out}/diagnostic_plan.json")

    def _install_remote_run_script(self, remote: Any, job: _Job) -> str:
        script = self._build_remote_script(job)
        remote_path = f"{job.output_dir}/run.sh"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", delete=False) as handle:
            handle.write(script)
            local_path = handle.name
        try:
            remote.put(local_path, remote_path)
        finally:
            try:
                import os

                os.unlink(local_path)
            except OSError:
                pass
        remote.exec_out(f"chmod +x {shlex.quote(remote_path)}")
        return (
            f"set +e; bash {shlex.quote(remote_path)}; "
            f"status=$?; tmux kill-session -t {_SESSION} 2>/dev/null || true; exit $status"
        )

    def _build_remote_command(self, job: _Job) -> str:
        """Backward-compatible helper for tests/inspection."""
        script_path = f"{job.output_dir}/run.sh"
        return (
            f"set +e; bash {shlex.quote(script_path)}; "
            f"status=$?; tmux kill-session -t {_SESSION} 2>/dev/null || true; exit $status"
        )

    def _remote_diagnostic_tasks(self, remote: Any) -> list[str]:
        return self._legacy_remote_diagnostic_tasks(remote)

    def _remote_payload_diagnostic_tasks(self, remote: Any) -> list[str]:
        s = self.settings
        script = f"""
import argparse
import json
try:
    from isaaclab.app import AppLauncher
    launch_args = argparse.Namespace(headless=True, enable_cameras=False, device="cuda:0")
    app = AppLauncher(launch_args).app
    try:
        import gymnasium as gym  # noqa: F401
        import taili_blind_runtime  # noqa: F401
        from gymnasium.envs.registration import registry
        names = sorted(str(key) for key in registry.keys() if "RobotLab" in str(key) or "Taili" in str(key))
        print("{_TASK_MARKER}" + json.dumps(names))
    finally:
        app.close()
except BaseException as exc:
    print("{_TASK_MARKER}" + json.dumps({{"error": type(exc).__name__ + ": " + str(exc)}}))
"""
        shell = "\n".join(
            [
                "set -e",
                self._payload_root_shell(),
                f"{shlex.quote(s.diagnostic_python)} -u -c {shlex.quote(script)}",
            ]
        )
        # Isaac cold-boot on the rented box takes 3-5 min; 120s timed out mid-boot and surfaced as an
        # opaque "not parseable" error. Give it headroom, and on failure show the probe's actual tail.
        raw = remote.exec_out(f"bash -lc {shlex.quote(shell)} 2>&1", timeout=420)
        marker_index = raw.rfind(_TASK_MARKER)
        if marker_index < 0:
            tail = (raw or "").strip()[-300:] or "(probe produced no output — likely SSH/Isaac-boot timeout)"
            raise RuntimeError(f"payload task registry probe did not return a parseable task list; probe tail: {tail}")
        payload = raw[marker_index + len(_TASK_MARKER):].strip().splitlines()[0]
        data = json.loads(payload)
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(str(data["error"]))
        if not isinstance(data, list):
            raise RuntimeError("payload task registry probe returned an unexpected payload")
        return [str(item) for item in data]

    def _legacy_remote_diagnostic_tasks(self, remote: Any) -> list[str]:
        s = self.settings
        script = f"""
import argparse
import json
try:
    from isaaclab.app import AppLauncher
    launch_args = argparse.Namespace(headless=True, enable_cameras=False, device="cuda:0")
    app = AppLauncher(launch_args).app
    try:
        import gymnasium as gym  # noqa: F401
        import robot_lab.tasks  # noqa: F401
        from gymnasium.envs.registration import registry
        names = sorted(str(key) for key in registry.keys() if "RobotLab" in str(key) or "Taili" in str(key))
        print("{_TASK_MARKER}" + json.dumps(names))
    finally:
        app.close()
except BaseException as exc:
    print("{_TASK_MARKER}" + json.dumps({{"error": type(exc).__name__ + ": " + str(exc)}}))
"""
        shell = " && ".join(
            [
                f"cd {shlex.quote(s.diagnostic_robot_root)}",
                f"export PYTHONPATH={shlex.quote(s.diagnostic_tool_root)}:$PYTHONPATH",
                f"{shlex.quote(s.diagnostic_python)} -u -c {shlex.quote(script)}",
            ]
        )
        raw = remote.exec_out(f"bash -lc {shlex.quote(shell)} 2>&1", timeout=420)
        marker_index = raw.rfind(_TASK_MARKER)
        if marker_index < 0:
            tail = (raw or "").strip()[-300:] or "(probe produced no output — likely SSH/Isaac-boot timeout)"
            raise RuntimeError(f"remote task registry probe did not return a parseable task list; probe tail: {tail}")
        payload = raw[marker_index + len(_TASK_MARKER):].strip().splitlines()[0]
        data = json.loads(payload)
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(str(data["error"]))
        if not isinstance(data, list):
            raise RuntimeError("remote task registry probe returned an unexpected payload")
        return [str(item) for item in data]

    def _resolve_remote_diagnostic_task(self, remote: Any, configured_task: str, framework_id: str) -> tuple[str, str]:
        # The registry probe boots Isaac (~3-5 min on this box) just to list task ids, which change
        # only when a payload adds/renames envs — cache per framework so repeat diagnostics start
        # immediately. A failed probe is not cached.
        cache = getattr(self, "_task_registry_cache", None)
        if cache is None:
            cache = self._task_registry_cache = {}
        candidates = cache.get(framework_id)
        if not candidates:
            if framework_id in _PAYLOAD_DIAG_FRAMEWORKS:
                candidates = self._remote_payload_diagnostic_tasks(remote)
            else:
                candidates = self._remote_diagnostic_tasks(remote)
            cache[framework_id] = candidates
        return _select_diagnostic_task_from_candidates(configured_task, framework_id, candidates)

    def _stage_output(self, job: _Job, stage: _Stage) -> str:
        if len(job.preset.stages) == 1:
            return job.output_dir
        return f"{job.output_dir}/{stage.id}"

    def _read_stage_statuses(self, remote: Any, job: _Job) -> list[DiagnosticStageStatus]:
        return [
            self._read_stage_status(remote, job, stage)
            for stage in job.preset.stages
        ]

    def _read_stage_status(self, remote: Any, job: _Job, stage: _Stage) -> DiagnosticStageStatus:
        output = self._stage_output(job, stage)
        meta = self._remote_json(remote, f"{output}/record_meta.json")
        progress_data = self._remote_json(remote, f"{output}/record_progress.json")
        metrics_exists = bool(
            (remote.exec_out(f"test -s {shlex.quote(output + '/metrics/metrics.json')} && echo yes") or "").strip()
        )
        error_exists = bool(
            (remote.exec_out(f"test -s {shlex.quote(output + '/record_error.json')} && echo yes") or "").strip()
        )
        record_lines_raw = (
            remote.exec_out(f"test -f {shlex.quote(output + '/record.csv')} && wc -l < {shlex.quote(output + '/record.csv')}")
            or ""
        ).strip()
        try:
            record_rows = max(0, int(record_lines_raw or "0") - 1)
        except ValueError:
            record_rows = 0
        rows = int(progress_data.get("rows_written", meta.get("rows_written", 0)) or 0)
        rows = max(rows, record_rows)
        meta_state = str(meta.get("status", ""))
        progress_state = str(progress_data.get("status", ""))
        if meta_state == "failed" or progress_state == "error" or error_exists:
            return DiagnosticStageStatus(
                id=stage.id, label=stage.label, state="error", progress=0.0, rows_written=rows
            )
        if metrics_exists:
            return DiagnosticStageStatus(
                id=stage.id, label=stage.label, state="complete", progress=1.0, rows_written=rows
            )
        if progress_data:
            if progress_data.get("requested_cases"):
                value = float(progress_data.get("completed_cases", 0)) / max(
                    1.0, float(progress_data["requested_cases"])
                )
            else:
                value = float(progress_data.get("completed_segments", 0)) / max(
                    1.0, float(stage.expected_segments)
                )
            return DiagnosticStageStatus(
                id=stage.id,
                label=stage.label,
                state="running",
                progress=min(0.92, max(0.08, value * 0.92)),
                rows_written=rows,
            )
        if meta_state == "running":
            return DiagnosticStageStatus(
                id=stage.id, label=stage.label, state="running", progress=0.05, rows_written=rows
            )
        return DiagnosticStageStatus(
            id=stage.id, label=stage.label, state="pending", progress=0.0, rows_written=rows
        )

    def _read_stage_error_message(self, remote: Any, job: _Job) -> str:
        for stage in job.preset.stages:
            output = self._stage_output(job, stage)
            data = self._remote_json(remote, f"{output}/record_error.json")
            if not data:
                continue
            error_type = str(data.get("error_type") or "").strip()
            error = str(data.get("error") or "").strip()
            case_id = data.get("case_id")
            case_dir = str(data.get("case_dir") or "").strip()
            prefix = f"{stage.label} failed"
            if case_id is not None:
                prefix += f" at case {case_id}"
            detail = f"{error_type + ': ' if error_type else ''}{error}".strip()
            suffix = f" case_dir={case_dir}" if case_dir else ""
            return f"{prefix}: {detail or 'see record_error.json'}{suffix}"
        return ""

    @staticmethod
    def _remote_json(remote: Any, path: str) -> dict[str, Any]:
        raw = remote.exec_out(f"cat {shlex.quote(path)} 2>/dev/null")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _read_remote_plan(self, remote: Any, job: _Job) -> dict[str, Any]:
        if isinstance(job.plan, dict) and job.plan:
            return job.plan
        return self._remote_json(remote, f"{job.output_dir}/diagnostic_plan.json")

    @staticmethod
    def _read_log_tail(remote: Any, job: _Job) -> list[str]:
        raw = remote.exec_out(f"tail -n 8 {shlex.quote(job.output_dir + '/job.log')} 2>/dev/null")
        return [line for line in (raw or "").splitlines() if line.strip()][-8:]

    def _read_report_inputs(self, remote: Any, job: _Job) -> tuple[list[tuple[str, dict[str, Any]]], DiagnosticLogInfo]:
        return self._read_metrics(remote, job), self._read_log_info(remote, job)

    @staticmethod
    def _read_log_info(remote: Any, job: _Job, tail_lines: int = 80) -> DiagnosticLogInfo:
        path = job.output_dir + "/job.log"
        quoted = shlex.quote(path)
        stat_raw = (remote.exec_out(
            f"test -f {quoted} && "
            f"printf '%s|%s|%s' \"$(wc -c < {quoted})\" \"$(wc -l < {quoted})\" "
            f"\"$(grep -c '\\[diagnostics\\] remote run script completed' {quoted} 2>/dev/null || true)\""
        ) or "").strip()
        if not stat_raw:
            return DiagnosticLogInfo(path=path, available=False, note="job.log is not present on the remote output directory.")
        parts = stat_raw.split("|")
        try:
            byte_count = int(parts[0] or 0)
            line_count = int(parts[1] or 0)
            complete = int(parts[2] or 0) > 0
        except (IndexError, ValueError):
            byte_count, line_count, complete = 0, 0, False
        raw_tail = remote.exec_out(f"tail -n {int(tail_lines)} {quoted} 2>/dev/null") or ""
        tail = [line for line in raw_tail.splitlines() if line.strip()]
        error_raw = remote.exec_out(
            "grep -nEi 'traceback|error|exception|failed|empty|no such file|missing|jsondecode|emptydata' "
            f"{quoted} 2>/dev/null | tail -n 20"
        ) or ""
        error_excerpt = [line for line in error_raw.splitlines() if line.strip()]
        return DiagnosticLogInfo(
            path=path,
            available=True,
            complete=complete,
            truncated=line_count > len(tail),
            bytes=byte_count,
            lines=line_count,
            shown_lines=len(tail),
            tail=tail,
            error_excerpt=error_excerpt,
            note=(
                "Tail excerpt only; full job.log remains on the remote output directory."
                if line_count > len(tail)
                else "Complete job.log content is shown in the excerpt."
            ),
        )

    def _report_artifacts(self, remote: Any, job: _Job, log: DiagnosticLogInfo) -> list[ArtifactRefInfo]:
        artifacts = [
            ArtifactRefInfo(
                kind="output_dir",
                label="Output directory",
                uri=job.output_dir,
                available=bool(job.output_dir),
                note="Remote directory containing all diagnostic artifacts.",
            ),
            ArtifactRefInfo(
                kind="job_log",
                label="job.log",
                uri=log.path,
                available=log.available,
                note=f"{log.lines} lines, {log.bytes} bytes. {log.note}" if log.available else log.note,
            ),
            ArtifactRefInfo(
                kind="run_script",
                label="run.sh",
                uri=f"{job.output_dir}/run.sh",
                available=bool((remote.exec_out(f"test -f {shlex.quote(job.output_dir + '/run.sh')} && echo yes") or "").strip()),
                note="The exact remote script executed by tmux.",
            ),
            ArtifactRefInfo(
                kind="diagnostic_plan",
                label="diagnostic_plan.json",
                uri=f"{job.output_dir}/diagnostic_plan.json",
                available=bool((remote.exec_out(f"test -f {shlex.quote(job.output_dir + '/diagnostic_plan.json')} && echo yes") or "").strip()),
                note="User-visible diagnostic intent: commands, terrains, DR cases, pushes, recording, and criteria.",
            ),
        ]
        for stage in job.preset.stages:
            output = self._stage_output(job, stage)
            for kind, label, rel, note in [
                ("record_csv", f"{stage.label} record.csv", "record.csv", "Raw rollout rows used for playback and metrics."),
                ("record_meta", f"{stage.label} record_meta.json", "record_meta.json", "Recorder status and row counts."),
                ("metrics_json", f"{stage.label} metrics.json", "metrics/metrics.json", "Metric output normalized into the report."),
            ]:
                path = f"{output}/{rel}"
                artifacts.append(
                    ArtifactRefInfo(
                        kind=f"{stage.id}_{kind}",
                        label=label,
                        uri=path,
                        available=bool((remote.exec_out(f"test -f {shlex.quote(path)} && echo yes") or "").strip()),
                        note=note,
                    )
                )
        return artifacts

    def _read_metrics(self, remote: Any, job: _Job) -> list[tuple[str, dict[str, Any]]]:
        reports = []
        for stage in job.preset.stages:
            output = self._stage_output(job, stage)
            metrics_path = output + "/metrics/metrics.json"
            raw = remote.exec_out(f"cat {shlex.quote(metrics_path)} 2>/dev/null")
            if not raw or not raw.strip():
                log_tail = self._read_log_tail(remote, job)
                detail = f"Diagnostic metrics are missing or empty for stage {stage.label}: {metrics_path}"
                if log_tail:
                    detail += ". Recent job.log: " + " | ".join(log_tail[-4:])
                raise RuntimeError(detail)
            try:
                reports.append((stage.id, json.loads(raw)))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Diagnostic metrics are not valid JSON for stage {stage.label}: {metrics_path}"
                ) from exc
        return reports

    def _read_playback_records(self, remote: Any, job: _Job) -> list[tuple[str, str]]:
        records: list[tuple[str, str]] = []
        for stage in job.preset.stages:
            output = self._stage_output(job, stage)
            # Bound both time and memory: a runaway record.csv must not be slurped whole into RAM
            # or hang the request. 32 MB is far more than any downsampled playback needs; a
            # truncated final line is tolerated by the CSV parser (it skips malformed rows).
            raw = remote.exec_out(
                f"head -c 33554432 {shlex.quote(output + '/record.csv')} 2>/dev/null", timeout=30)
            if raw and raw.strip():
                records.append((stage.id, raw))
        return records

    def _fake_status(self, job: _Job, elapsed: float) -> DiagnosticJobStatus:
        if job.state == "cancelled":
            progress = 0.0
        else:
            progress = min(1.0, elapsed / 5.0)
            if progress >= 1.0:
                job.state = "complete"
                job.message = "Demo diagnostics completed"
            else:
                job.state = "running"
                job.message = "Running demo diagnostics"
        count = len(job.preset.stages)
        stage_statuses = []
        scaled = progress * count
        for index, stage in enumerate(job.preset.stages):
            stage_progress = min(1.0, max(0.0, scaled - index))
            state = "complete" if stage_progress >= 1 else "running" if stage_progress > 0 else "pending"
            stage_statuses.append(
                DiagnosticStageStatus(
                    id=stage.id,
                    label=stage.label,
                    state=state,
                    progress=stage_progress,
                    rows_written=int(stage_progress * 800),
                )
            )
        return self._status_model(job, stage_statuses, progress, elapsed, ["Demo mode: remote GPU is not connected"])

    @staticmethod
    def _status_model(
        job: _Job,
        stages: list[DiagnosticStageStatus],
        progress: float,
        elapsed: float,
        log_tail: list[str],
    ) -> DiagnosticJobStatus:
        return DiagnosticJobStatus(
            state=job.state,
            job_id=job.id,
            preset=job.preset.id,
            preset_label=job.preset.label,
            checkpoint=job.checkpoint,
            framework_id=job.framework_id,
            output_dir=job.output_dir,
            progress=progress,
            elapsed_s=elapsed,
            message=job.message,
            stages=stages,
            log_tail=log_tail,
            plan_summary=_plan_summary(job.plan),
        )

    def _record_status(self, status: DiagnosticJobStatus) -> DiagnosticJobStatus:
        diagnostic_task = ""
        if self._job and self._job.id == status.job_id:
            diagnostic_task = self._job.diagnostic_task
        self.history.upsert_status(
            status,
            source="real" if self.settings.source == "real" else "fake",
            config_set_id=self.config_set.id,
            diagnostic_task=diagnostic_task or self.framework.diagnostic_task,
        )
        return status

    def _restore_latest_job_if_needed(self) -> None:
        if self._job is not None:
            return
        for item in self.history.list(limit=10).items:
            if item.state not in {"starting", "running"}:
                continue
            if not item.output_dir or not item.preset:
                continue
            self._job = self._job_from_history_item(item)
            if self._job is not None:
                return

    def _job_for_artifact(self, job_id: str | None = None) -> _Job | None:
        if job_id:
            item = self.history.get(job_id)
            return self._job_from_history_item(item) if item is not None else None
        self._restore_latest_job_if_needed()
        return self._job

    def _job_from_history_item(self, item) -> _Job | None:
        if item is None or not item.output_dir or not item.preset:
            return None
        preset = _PRESET_BY_ID.get(item.preset)
        if preset is None:
            return None
        return _Job(
            id=item.job_id,
            preset=preset,
            checkpoint=item.checkpoint,
            framework_id=item.framework_id or self.framework.id,
            diagnostic_task=item.diagnostic_task or self.framework.diagnostic_task,
            output_dir=item.output_dir,
            started_at=max(0.0, time.time() - float(item.elapsed_s or 0.0)),
            state=item.state,
            message=item.message,
        )


def _stat_value(value: Any, key: str = "mean") -> float | None:
    if not isinstance(value, dict):
        return None
    result = value.get(key)
    return float(result) if isinstance(result, (int, float)) else None


def _finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bool_cell(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "1.0", "true", "yes", "y"}


def _row_key(row: dict[str, str]) -> tuple[int, int, int]:
    return (
        int(_finite_float(row.get("case_id")) or 0),
        int(_finite_float(row.get("env_id")) or 0),
        int(_finite_float(row.get("episode_id")) or 0),
    )


def _available_env_ids(rows: list[dict[str, str]]) -> list[int]:
    env_ids: set[int] = set()
    for row in rows:
        env_ids.add(int(_finite_float(row.get("env_id")) or 0))
    return sorted(env_ids)


def _choose_playback_env(rows: list[dict[str, str]]) -> int | None:
    groups: dict[tuple[int, int, int], int] = {}
    for row in rows:
        if _bool_cell(row.get("post_step_state_may_be_after_reset")):
            continue
        key = _row_key(row)
        groups[key] = groups.get(key, 0) + 1
    if not groups:
        return None
    by_env: dict[int, dict[int, int]] = {}
    for (case_id, env_id, _episode_id), length in groups.items():
        cases = by_env.setdefault(env_id, {})
        cases[case_id] = max(cases.get(case_id, 0), length)
    return max(
        by_env,
        key=lambda env_id: (
            len(by_env[env_id]),
            sum(by_env[env_id].values()),
            -env_id,
        ),
    )


def _playback_rows_for_env(rows: list[dict[str, str]], env_id: int | None = None) -> list[dict[str, str]]:
    selected_env_id = _choose_playback_env(rows) if env_id is None else env_id
    if selected_env_id is None:
        return []
    selected = [
        row for row in rows
        if int(_finite_float(row.get("env_id")) or 0) == selected_env_id
    ]
    return sorted(selected, key=lambda item: (
        _finite_float(item.get("case_id")) or 0.0,
        _finite_float(item.get("time")) or 0.0,
        _finite_float(item.get("step")) or 0.0,
    ))


def _best_continuous_rows(rows: list[dict[str, str]], env_id: int | None = None) -> list[dict[str, str]]:
    return _playback_rows_for_env(rows, env_id)


def _downsample(rows: list[dict[str, str]], max_frames: int) -> tuple[list[dict[str, str]], int]:
    if len(rows) <= max_frames:
        return rows, 1
    stride = int(math.ceil(len(rows) / max_frames))
    return rows[::stride], stride


def _frame_from_row(stage_id: str, row: dict[str, str], t_offset: float) -> DiagnosticPlaybackFrame | None:
    base = [
        _finite_float(row.get("base_pos_w_x")),
        _finite_float(row.get("base_pos_w_y")),
        _finite_float(row.get("base_pos_w_z")),
    ]
    quat = [
        _finite_float(row.get("base_quat_w")),
        _finite_float(row.get("base_quat_x")),
        _finite_float(row.get("base_quat_y")),
        _finite_float(row.get("base_quat_z")),
    ]
    joints = [_finite_float(row.get(f"joint_pos_{i}")) for i in range(len(_JOINT_ORDER))]
    if any(value is None for value in base + quat + joints):
        return None
    row_t = _finite_float(row.get("time")) or 0.0
    feet: dict[str, DiagnosticPlaybackFoot] = {}
    for leg in _LEG_ORDER:
        pos = [
            _finite_float(row.get(f"foot_{leg}_pos_w_x")),
            _finite_float(row.get(f"foot_{leg}_pos_w_y")),
            _finite_float(row.get(f"foot_{leg}_pos_w_z")),
        ]
        if any(value is None for value in pos):
            continue
        feet[leg] = DiagnosticPlaybackFoot(
            position=[float(value) for value in pos if value is not None],
            contact=_bool_cell(row.get(f"foot_{leg}_contact")),
            force_norm=_finite_float(row.get(f"foot_{leg}_force_norm")),
            clearance=_finite_float(row.get(f"foot_{leg}_clearance_local")),
        )
    return DiagnosticPlaybackFrame(
        t=t_offset + row_t,
        stage=stage_id,
        case_id=int(_finite_float(row.get("case_id")) or 0),
        env_id=int(_finite_float(row.get("env_id")) or 0),
        segment_id=int(_finite_float(row.get("cmd_segment_id")) or 0),
        command_mode=str(row.get("cmd_target_mode") or row.get("cmd_mode") or "unknown"),
        base_position=[float(value) for value in base if value is not None],
        base_quaternion_wxyz=[float(value) for value in quat if value is not None],
        joints=[float(value) for value in joints if value is not None],
        feet=feet,
        reset_observed=_bool_cell(row.get("reset_observed")),
        done=_bool_cell(row.get("done")),
        terrain_height=_finite_float(row.get("base_terrain_height")),
        terrain=str(row.get("terrain_type") or row.get("terrain_type_requested") or ""),
        terrain_level=(int(_finite_float(row.get("terrain_level")))
                       if _finite_float(row.get("terrain_level")) is not None else None),
    )


def _playback_from_record_texts(
    job: _Job,
    stage_records: list[tuple[str, str]],
    source: str,
    max_frames: int,
) -> DiagnosticPlayback:
    source_rows = 0
    all_rows: list[dict[str, str]] = []
    rows_by_stage: list[tuple[str, list[dict[str, str]]]] = []
    selected_rows: list[tuple[str, dict[str, str]]] = []
    for stage_id, text in stage_records:
        rows = list(csv.DictReader(io.StringIO(text)))
        source_rows += len(rows)
        all_rows.extend(rows)
        rows_by_stage.append((stage_id, rows))
    selected_env_id = _choose_playback_env(all_rows)
    for stage_id, rows in rows_by_stage:
        for row in _best_continuous_rows(rows, selected_env_id):
            selected_rows.append((stage_id, row))
    available_env_ids = _available_env_ids(all_rows)
    if not selected_rows:
        return DiagnosticPlayback(
            available=False,
            message="Diagnostic output did not contain playable record.csv frames.",
            source=source,  # type: ignore[arg-type]
            output_dir=job.output_dir,
            joint_order=_JOINT_ORDER,
            leg_order=_LEG_ORDER,
            source_rows=source_rows,
            selected_env_id=selected_env_id,
            available_env_ids=available_env_ids,
        )
    rows_only = [row for _, row in selected_rows]
    sampled_rows, stride = _downsample(rows_only, max_frames)
    sampled_ids = {id(row) for row in sampled_rows}
    frames: list[DiagnosticPlaybackFrame] = []
    t_offset = 0.0
    previous_chunk = None          # (stage_id, case_id): per-case time restarts at 0, so offset on
    previous_raw_t = 0.0           # CASE boundaries too or the timeline goes non-monotonic
    for stage_id, row in selected_rows:
        if id(row) not in sampled_ids:
            continue
        raw_t = _finite_float(row.get("time")) or 0.0
        chunk = (stage_id, int(_finite_float(row.get("case_id")) or 0))
        if previous_chunk is not None and chunk != previous_chunk:
            t_offset += previous_raw_t + 0.5
        frame = _frame_from_row(stage_id, row, t_offset)
        if frame is not None:
            frames.append(frame)
            previous_raw_t = raw_t
            previous_chunk = chunk
    if len(frames) >= 2:
        deltas = [b.t - a.t for a, b in zip(frames, frames[1:]) if b.t > a.t]
        fps = 1.0 / (sum(deltas) / len(deltas)) if deltas else 50.0
    else:
        fps = 50.0
    return DiagnosticPlayback(
        available=bool(frames),
        message="Actual IsaacLab rollout frames from diagnostic record.csv" if frames else "No valid playback frames",
        source=source,  # type: ignore[arg-type]
        output_dir=job.output_dir,
        fps=max(1.0, min(120.0, fps)),
        joint_order=_JOINT_ORDER,
        leg_order=_LEG_ORDER,
        frames=frames,
        source_rows=source_rows,
        stride=stride,
        selected_env_id=selected_env_id,
        available_env_ids=available_env_ids,
    )


def _fake_playback(job: _Job, max_frames: int = 900) -> DiagnosticPlayback:
    frames: list[DiagnosticPlaybackFrame] = []
    dt = 1.0 / 50.0
    total = min(max_frames, 240)
    for i in range(total):
        t = i * dt
        phase = t * math.tau * 1.8
        moving = t > 0.8
        x = max(0.0, t - 0.8) * 0.35 if moving else 0.0
        z = 0.55 + (0.015 * math.sin(phase * 2.0) if moving else 0.0)
        joints = [0.0] * len(_JOINT_ORDER)
        for leg_index in range(4):
            leg_phase = phase + (math.pi if leg_index in {0, 3} else 0.0)
            hip = 0.10 * math.sin(leg_phase) if moving else 0.0
            thigh = 0.45 + (0.20 * math.sin(leg_phase) if moving else 0.0)
            calf = -0.90 + (0.30 * math.sin(leg_phase + 0.7) if moving else 0.0)
            joints[leg_index] = hip
            joints[4 + leg_index] = thigh
            joints[8 + leg_index] = calf
        foot_offsets = {
            "FL": (0.28, 0.18),
            "FR": (0.28, -0.18),
            "RL": (-0.28, 0.18),
            "RR": (-0.28, -0.18),
        }
        feet = {}
        for leg_index, leg in enumerate(_LEG_ORDER):
            leg_phase = phase + (math.pi if leg_index in {0, 3} else 0.0)
            swing = moving and math.sin(leg_phase) > 0.0
            ox, oy = foot_offsets[leg]
            feet[leg] = DiagnosticPlaybackFoot(
                position=[x + ox + 0.04 * math.sin(leg_phase), oy, 0.04 if swing else 0.0],
                contact=not swing,
                force_norm=55.0 if not swing else 2.0,
                clearance=0.04 if swing else 0.0,
            )
        frames.append(
            DiagnosticPlaybackFrame(
                t=t,
                stage=job.preset.stages[0].id if job.preset.stages else job.preset.id,
                case_id=0,
                env_id=0,
                segment_id=1 if moving else 0,
                command_mode="forward" if moving else "stand",
                base_position=[x, 0.0, z],
                base_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
                joints=joints,
                feet=feet,
                reset_observed=False,
                done=False,
                terrain_height=0.0,
            )
        )
    return DiagnosticPlayback(
        available=True,
        message="Demo diagnostic frames; real mode reads IsaacLab record.csv",
        source="fake",
        output_dir=job.output_dir,
        fps=50.0,
        joint_order=_JOINT_ORDER,
        leg_order=_LEG_ORDER,
        frames=frames,
        source_rows=len(frames),
        stride=1,
        selected_env_id=0,
        available_env_ids=[0],
    )


def _metrics_source_for_job(job: _Job) -> str:
    if len(job.preset.stages) == 1:
        return f"{job.output_dir}/metrics/metrics.json"
    return f"{job.output_dir}/<stage>/metrics/metrics.json"


def _value_info(
    *,
    key: str,
    label: str,
    group: str,
    value: Any,
    unit: str = "",
    source: str,
    fields: list[str],
    formula: str,
    interpretation: str,
    confidence: str = "high",
) -> DiagnosticValueInfo:
    return DiagnosticValueInfo(
        key=key,
        label=label,
        group=group,
        value=value,
        unit=unit,
        source=source,
        fields=fields,
        formula=formula,
        interpretation=interpretation,
        confidence=confidence,  # type: ignore[arg-type]
    )


def _build_value_explanations(job: _Job, report_data: dict[str, Any]) -> list[DiagnosticValueInfo]:
    source = _metrics_source_for_job(job)
    coverage = report_data["coverage"]
    posture = report_data["posture"]
    commands = report_data["commands"]
    major_modes = [cmd["mode"] for cmd in commands if float(cmd.get("major_event_fraction") or 0.0) > 0.0]
    explanations = [
        _value_info(
            key="coverage.attempts",
            label="尝试段数",
            group="summary",
            value=coverage.get("attempts"),
            unit="segments",
            source=source,
            fields=["coverage.segment_attempts"],
            formula="sum(stage.metrics.coverage.segment_attempts)",
            interpretation="诊断实际观察到的命令段尝试数；它描述覆盖量，不直接等于好坏。",
        ),
        _value_info(
            key="posture.stable_fraction",
            label="稳定姿态占比",
            group="summary",
            value=posture.get("stable_fraction"),
            unit="fraction",
            source=source,
            fields=["posture.pose_stable_fraction_all", "coverage.rows"],
            formula="sum(stage.pose_stable_fraction_all * stage.rows) / sum(stage.rows)",
            interpretation="所有采样行中姿态稳定的比例。低于 0.75 时优先看姿态、接触和 reset 事件。",
            confidence="medium" if posture.get("stable_fraction") is None else "high",
        ),
        _value_info(
            key="summary.major_event_modes",
            label="出现主要事件的方向数",
            group="summary",
            value=len(major_modes),
            unit="modes",
            source=source,
            fields=["command_breakdown.*.major_event_attempt_fraction"],
            formula="count(command rows where major_event_fraction > 0)",
            interpretation=(
                "这些方向出现过 reset/done/大姿态异常等主要事件；先看这些方向的回放和日志。"
                if major_modes else
                "归一化结果里没有方向出现主要事件；仍需看 tracking/error 是否异常。"
            ),
        ),
        _value_info(
            key="coverage.rows",
            label="原始数据行数",
            group="summary",
            value=coverage.get("rows"),
            unit="rows",
            source=source,
            fields=["coverage.rows"],
            formula="sum(stage.metrics.coverage.rows)",
            interpretation="record.csv 的原始采样行数。行数为 0 或过小通常表示 recorder 没真正跑起来。",
        ),
        _value_info(
            key="posture.max_roll_deg",
            label="最大横滚",
            group="posture",
            value=posture.get("max_roll_deg"),
            unit="deg",
            source=source,
            fields=["posture.base_roll_abs.max"],
            formula="max(stage.metrics.posture.base_roll_abs.max)",
            interpretation="诊断中观察到的最大横滚绝对值；大于 30° 通常意味着姿态风险明显。",
            confidence="medium" if posture.get("max_roll_deg") is None else "high",
        ),
        _value_info(
            key="posture.max_pitch_deg",
            label="最大俯仰",
            group="posture",
            value=posture.get("max_pitch_deg"),
            unit="deg",
            source=source,
            fields=["posture.base_pitch_abs.max"],
            formula="max(stage.metrics.posture.base_pitch_abs.max)",
            interpretation="诊断中观察到的最大俯仰绝对值；大于 30° 时优先看起步/换向瞬间。",
            confidence="medium" if posture.get("max_pitch_deg") is None else "high",
        ),
        _value_info(
            key="posture.min_height_m",
            label="最低机身高度",
            group="posture",
            value=posture.get("min_height_m"),
            unit="m",
            source=source,
            fields=["posture.base_height_local.min"],
            formula="min(stage.metrics.posture.base_height_local.min)",
            interpretation="局部地形系下的最低机身高度；过低通常对应蹲塌、触地或 reset 前兆。",
            confidence="medium" if posture.get("min_height_m") is None else "high",
        ),
    ]
    for command in commands:
        mode = str(command.get("mode") or "unknown")
        attempts = max(1, int(command.get("attempts") or 0))
        base_fields = [f"command_breakdown.{mode}"]
        explanations.extend([
            _value_info(
                key=f"commands.{mode}.stable_fraction",
                label=f"{mode} 稳定占比",
                group="command",
                value=command.get("stable_fraction"),
                unit="fraction",
                source=source,
                fields=base_fields + ["pose_stable_fraction_all.mean", "attempts_observed"],
                formula="weighted_mean(stage.command_breakdown[mode].pose_stable_fraction_all.mean, attempts_observed)",
                interpretation="该命令方向下姿态稳定的比例；低值说明该方向本身不稳或事件过早发生。",
            ),
            _value_info(
                key=f"commands.{mode}.major_event_fraction",
                label=f"{mode} 主要事件比例",
                group="command",
                value=command.get("major_event_fraction"),
                unit="fraction",
                source=source,
                fields=base_fields + ["major_event_attempt_fraction", "attempts_observed"],
                formula="weighted_mean(stage.command_breakdown[mode].major_event_attempt_fraction, attempts_observed)",
                interpretation="该方向尝试中出现主要事件的比例；越高越应优先检查该方向回放。",
            ),
            _value_info(
                key=f"commands.{mode}.controlled_progress_m",
                label=f"{mode} 受控位移",
                group="command",
                value=command.get("controlled_progress_m"),
                unit="m",
                source=source,
                fields=base_fields + ["pose_stable_pre_first_major_event_progress_along_command.mean"],
                formula=(
                    "weighted_mean("
                    "stage.command_breakdown[mode].pose_stable_pre_first_major_event_progress_along_command.mean, "
                    "attempts_observed)"
                ),
                interpretation="只统计第一处主要事件前、姿态稳定阶段的沿命令方向位移；比 raw progress 更接近可控能力。",
            ),
            _value_info(
                key=f"commands.{mode}.raw_progress_m",
                label=f"{mode} 原始位移",
                group="command",
                value=command.get("raw_progress_m"),
                unit="m",
                source=source,
                fields=base_fields + ["raw_progress_along_command.mean"],
                formula="weighted_mean(stage.command_breakdown[mode].raw_progress_along_command.mean, attempts_observed)",
                interpretation="不扣除异常事件的沿命令方向位移；摔倒/滑行也可能让它偏大。",
            ),
            _value_info(
                key=f"commands.{mode}.tracking_error",
                label=f"{mode} 跟踪误差",
                group="command",
                value=command.get("tracking_error"),
                unit="rad/s" if mode == "yaw" else "m/s",
                source=source,
                fields=["tracking.pose_stable_before_first_major_event.by_cmd_mode", "tracking.pose_stable.by_cmd_mode"],
                formula="weighted_mean(selected p50 tracking error for mode, attempts_observed)",
                interpretation="优先从姿态稳定且事件前的切片取 p50 误差；用于判断能不能跟住命令。",
                confidence="medium" if command.get("tracking_error") is None else "high",
            ),
            _value_info(
                key=f"commands.{mode}.first_major_event_s",
                label=f"{mode} 首个主要事件时间",
                group="command",
                value=command.get("first_major_event_s"),
                unit="s",
                source=source,
                fields=base_fields + ["first_major_event_time_s.mean", "first_major_event_time_s.n"],
                formula="weighted_mean(first_major_event_time_s.mean, first_major_event_time_s.n)",
                interpretation=(
                    "越早说明该方向越快进入异常；为空表示没有观测到主要事件。"
                    if command.get("first_major_event_s") is not None else
                    "没有观测到主要事件，因此该值为空。"
                ),
                confidence="medium" if command.get("first_major_event_s") is None else "high",
            ),
        ])
        explanations.append(
            _value_info(
                key=f"commands.{mode}.attempts",
                label=f"{mode} 尝试数",
                group="command",
                value=attempts,
                unit="segments",
                source=source,
                fields=base_fields + ["attempts_observed"],
                formula="sum(stage.command_breakdown[mode].attempts_observed)",
                interpretation="该方向实际参与统计的命令段数量；数量少时结论置信度会下降。",
                confidence="low" if attempts < 2 else "high",
            )
        )
    return explanations


def _normalize_report(
    job: _Job,
    reports: list[tuple[str, dict[str, Any]]],
    *,
    log: DiagnosticLogInfo | None = None,
    artifacts: list[ArtifactRefInfo] | None = None,
    plan: dict[str, Any] | None = None,
) -> DiagnosticReport:
    plan = plan or job.plan or {}
    coverage = {
        "rows": 0,
        "cases": 0,
        "attempts": 0,
        "terrain_types": set(),
        "terrain_types_requested": set(),
        "dr_levels": set(),
        "dr_levels_requested": set(),
        "push_frames": 0,
    }
    command_acc: dict[str, dict[str, float]] = {}
    posture = {
        "max_roll_deg": None,
        "max_pitch_deg": None,
        "min_height_m": None,
        "stable_fraction": None,
    }
    event_counts: dict[str, int] = {}
    notes: list[dict[str, Any]] = []
    note_seen: set[tuple[str, str]] = set()
    terrain_sources: set[str] = set()
    dr_columns: set[str] = set()
    stable_weight = 0
    stable_total = 0.0

    for stage_id, metrics in reports:
        cov = metrics.get("coverage", {})
        rows = int(cov.get("rows", 0) or 0)
        coverage["rows"] += rows
        coverage["cases"] += int(cov.get("cases", 0) or 0)
        coverage["attempts"] += int(cov.get("segment_attempts", 0) or 0)
        coverage["terrain_types"].update(cov.get("terrain_types_observed", []))
        coverage["terrain_types_requested"].update(cov.get("terrain_types_requested", []))
        coverage["dr_levels"].update(cov.get("dr_levels_observed", []))
        coverage["dr_levels_requested"].update(cov.get("dr_levels_requested", []))
        coverage["push_frames"] += int(cov.get("push_frames_observed", 0) or 0)
        terrain_sources.update(cov.get("terrain_height_sources_observed", []))

        for mode, data in metrics.get("command_breakdown", {}).items():
            attempts = int(data.get("attempts_observed", 0) or 0)
            if attempts <= 0:
                continue
            acc = command_acc.setdefault(
                mode,
                {
                    "attempts": 0.0,
                    "major": 0.0,
                    "done": 0.0,
                    "reset": 0.0,
                    "stable": 0.0,
                    "raw": 0.0,
                    "controlled": 0.0,
                    "first_major": 0.0,
                    "first_major_n": 0.0,
                    "tracking_error": 0.0,
                    "tracking_n": 0.0,
                },
            )
            acc["attempts"] += attempts
            for source, target in [
                ("major_event_attempt_fraction", "major"),
                ("done_attempt_fraction", "done"),
                ("reset_attempt_fraction", "reset"),
            ]:
                acc[target] += float(data.get(source, 0.0) or 0.0) * attempts
            for source, target in [
                ("pose_stable_fraction_all", "stable"),
                ("raw_progress_along_command", "raw"),
                ("pose_stable_pre_first_major_event_progress_along_command", "controlled"),
            ]:
                value = _stat_value(data.get(source))
                if value is not None:
                    acc[target] += value * attempts
            first = data.get("first_major_event_time_s", {})
            first_n = int(first.get("n", 0) or 0) if isinstance(first, dict) else 0
            first_mean = _stat_value(first)
            if first_n and first_mean is not None:
                acc["first_major"] += first_mean * first_n
                acc["first_major_n"] += first_n
            tracking_error = _tracking_error(metrics, mode)
            if tracking_error is not None:
                acc["tracking_error"] += tracking_error * attempts
                acc["tracking_n"] += attempts

        post = metrics.get("posture", {})
        roll = _stat_value(post.get("base_roll_abs"), "max")
        pitch = _stat_value(post.get("base_pitch_abs"), "max")
        height = _stat_value(post.get("base_height_local"), "min")
        posture["max_roll_deg"] = _max_optional(posture["max_roll_deg"], roll)
        posture["max_pitch_deg"] = _max_optional(posture["max_pitch_deg"], pitch)
        posture["min_height_m"] = _min_optional(posture["min_height_m"], height)
        stable = post.get("pose_stable_fraction_all")
        if isinstance(stable, (int, float)) and rows:
            stable_total += float(stable) * rows
            stable_weight += rows

        for event, count in metrics.get("event_counts_by_bout_or_instant", {}).items():
            event_counts[event] = event_counts.get(event, 0) + int(count or 0)

        diag_notes = metrics.get("diagnostic_notes", {})
        for item in diag_notes.get("top_observed_patterns", []):
            key = (str(item.get("type")), str(item.get("message")))
            if key not in note_seen:
                note_seen.add(key)
                notes.append(item)
        dr_columns.update(metrics.get("robustness", {}).get("coverage", {}).get("dr_columns_observed", []))

    if isinstance(plan, dict):
        for terrain in plan.get("terrains", []) if isinstance(plan.get("terrains"), list) else []:
            if isinstance(terrain, dict) and terrain.get("type"):
                coverage["terrain_types_requested"].add(str(terrain["type"]))
        for dr_case in plan.get("dr_cases", []) if isinstance(plan.get("dr_cases"), list) else []:
            if isinstance(dr_case, dict) and dr_case.get("level") is not None:
                coverage["dr_levels_requested"].add(str(dr_case["level"]))

    posture["stable_fraction"] = stable_total / stable_weight if stable_weight else None
    commands = []
    for mode, acc in command_acc.items():
        attempts = max(1.0, acc["attempts"])
        commands.append(
            {
                "mode": mode,
                "attempts": int(acc["attempts"]),
                "major_event_fraction": acc["major"] / attempts,
                "done_fraction": acc["done"] / attempts,
                "reset_fraction": acc["reset"] / attempts,
                "stable_fraction": acc["stable"] / attempts,
                "raw_progress_m": acc["raw"] / attempts,
                "controlled_progress_m": acc["controlled"] / attempts,
                "first_major_event_s": (
                    acc["first_major"] / acc["first_major_n"] if acc["first_major_n"] else None
                ),
                "tracking_error": (
                    acc["tracking_error"] / acc["tracking_n"] if acc["tracking_n"] else None
                ),
            }
        )
    mode_order = {"stand": 0, "forward": 1, "backward": 2, "lateral": 3, "yaw": 4}
    commands.sort(key=lambda item: mode_order.get(item["mode"], 99))
    events = [
        {"type": event, "count": count}
        for event, count in sorted(event_counts.items(), key=lambda item: (-item[1], item[0]))[:12]
    ]
    terrain_types = sorted(coverage["terrain_types"])
    terrain_requested = sorted(coverage["terrain_types_requested"])
    dr_requested = sorted(coverage["dr_levels_requested"])
    terrain_height_sources = sorted(terrain_sources)
    terrain_gaps: list[str] = []
    if not terrain_types:
        terrain_gaps.append("No terrain type was reported by diagnostics metrics.")
    elif terrain_types == ["flat"]:
        terrain_gaps.append("Only flat terrain was covered; this cannot validate slope/rough/boxes/stairs behavior.")
    missing_terrains = sorted(set(terrain_requested) - set(terrain_types))
    if missing_terrains:
        terrain_gaps.append(f"Requested terrain(s) were not observed in metrics: {', '.join(missing_terrains)}.")
    if not terrain_height_sources:
        terrain_gaps.append("No terrain height source was reported; terrain-relative height/clearance interpretation is weak.")
    if "body_collision" not in event_counts:
        terrain_gaps.append("body_collision was not observed/reported; terrain obstacle collision wiring may still be missing.")
    if "hard_impact_bout" not in event_counts:
        terrain_gaps.append("hard_impact_bout was not reported; landing impact coverage may be absent or uneventful.")
    terrain_command_rows = [
        {
            "label": "terrain types",
            "value": ", ".join(terrain_types) if terrain_types else "missing",
            "tone": "good" if len(terrain_types) >= 2 else "warn",
        },
        {
            "label": "height source",
            "value": ", ".join(terrain_height_sources) if terrain_height_sources else "missing",
            "tone": "good" if terrain_height_sources else "bad",
        },
        {
            "label": "body collision signal",
            "value": str(event_counts.get("body_collision", 0)),
            "tone": "good" if "body_collision" in event_counts else "warn",
        },
        {
            "label": "hard impact bouts",
            "value": str(event_counts.get("hard_impact_bout", 0)),
            "tone": "warn" if event_counts.get("hard_impact_bout", 0) else "good",
        },
        {
            "label": "high slip bouts",
            "value": str(event_counts.get("high_slip_bout", 0)),
            "tone": "warn" if event_counts.get("high_slip_bout", 0) else "good",
        },
    ]
    terrain_readiness = "complete" if not terrain_gaps else "partial"
    report_data = {
        "coverage": {
            "rows": coverage["rows"],
            "cases": coverage["cases"],
            "attempts": coverage["attempts"],
            "terrain_types_requested": terrain_requested,
            "terrain_types": terrain_types,
            "terrain_height_sources": terrain_height_sources,
            "dr_levels_requested": dr_requested,
            "dr_levels": sorted(coverage["dr_levels"]),
            "push_frames": coverage["push_frames"],
        },
        "commands": commands,
        "posture": posture,
        "events": events,
        "notes": notes[:10],
        "terrain": {
            "requested_types": terrain_requested,
            "types": terrain_types,
            "height_sources": terrain_height_sources,
            "readiness": terrain_readiness,
            "gaps": terrain_gaps,
            "terrain_command_rows": terrain_command_rows,
        },
        "robustness": {
            "dr_levels_requested": dr_requested,
            "dr_levels": sorted(coverage["dr_levels"]),
            "dr_columns": sorted(dr_columns),
            "push_frames": coverage["push_frames"],
        },
        "plan": plan,
    }
    return DiagnosticReport(
        job_id=job.id,
        preset=job.preset.id,
        preset_label=job.preset.label,
        checkpoint=job.checkpoint,
        output_dir=job.output_dir,
        artifacts=artifacts or [],
        log=log or DiagnosticLogInfo(
            path=f"{job.output_dir}/job.log",
            available=False,
            note="No job.log metadata was loaded for this report.",
        ),
        value_explanations=_build_value_explanations(job, report_data),
        coverage=report_data["coverage"],
        commands=report_data["commands"],
        posture=report_data["posture"],
        events=report_data["events"],
        notes=report_data["notes"],
        terrain=report_data["terrain"],
        robustness=report_data["robustness"],
        plan=report_data["plan"],
    )


def _tracking_error(metrics: dict[str, Any], mode: str) -> float | None:
    tracking = metrics.get("tracking", {})
    for slice_name in ["pose_stable_command_settled", "pose_stable_before_first_major_event", "pose_stable"]:
        buckets = tracking.get(slice_name, {}).get("by_cmd_mode", {})
        bucket = buckets.get(f"cmd_target_mode={mode}")
        if bucket is None:
            bucket = next(
                (value for key, value in buckets.items() if str(key).endswith(f"={mode}")),
                None,
            )
        if not bucket:
            continue
        stat = bucket.get("wz_abs_error" if mode == "yaw" else "xy_error_norm")
        value = _stat_value(stat, "p50")
        if value is not None:
            return value
    return None


def _max_optional(current: float | None, value: float | None) -> float | None:
    if value is None:
        return current
    return value if current is None else max(current, value)


def _min_optional(current: float | None, value: float | None) -> float | None:
    if value is None:
        return current
    return value if current is None else min(current, value)


def _fake_report(job: _Job) -> DiagnosticReport:
    plan = job.plan or _default_plan_for_preset(job.preset)
    if job.preset.id in {"forward", "backward", "lateral", "yaw"}:
        modes = [job.preset.id]
    elif job.preset.id == "directions":
        modes = ["forward", "backward", "lateral", "yaw"]
    else:
        modes = ["forward"]
    raw = {
        "forward": (0.0, 1.0, 0.19, 0.19, 0.46, None),
        "backward": (1.0, 0.18, 1.08, 0.04, 0.09, 0.53),
        "lateral": (0.25, 0.99, 0.36, 0.30, 0.27, 1.42),
        "yaw": (1.0, 0.92, 0.0, 0.0, 0.33, 1.72),
    }
    commands = [
        {
            "mode": mode,
            "attempts": 4,
            "major_event_fraction": raw[mode][0],
            "done_fraction": raw[mode][0] if mode != "backward" else 0.0,
            "reset_fraction": raw[mode][0] if mode != "backward" else 0.0,
            "stable_fraction": raw[mode][1],
            "raw_progress_m": raw[mode][2],
            "controlled_progress_m": raw[mode][3],
            "tracking_error": raw[mode][4],
            "first_major_event_s": raw[mode][5],
        }
        for mode in modes
    ]
    base = DiagnosticReport(
        job_id=job.id,
        preset=job.preset.id,
        preset_label=job.preset.label,
        checkpoint=job.checkpoint,
        output_dir=job.output_dir,
        artifacts=[
            ArtifactRefInfo(
                kind="output_dir",
                label="Output directory",
                uri=job.output_dir,
                available=True,
                note="Demo output directory.",
            ),
            ArtifactRefInfo(
                kind="job_log",
                label="job.log",
                uri=f"{job.output_dir}/job.log",
                available=False,
                note="Demo mode does not write a remote job log.",
            ),
            ArtifactRefInfo(
                kind="metrics_json",
                label="metrics.json",
                uri=f"{job.output_dir}/metrics/metrics.json",
                available=True,
                note="Synthetic metrics used by demo mode.",
            ),
        ],
        log=DiagnosticLogInfo(
            path=f"{job.output_dir}/job.log",
            available=False,
            complete=True,
            note="Demo mode has no remote IsaacLab job.log.",
        ),
        coverage={
            "rows": 3200,
            "cases": 4,
            "attempts": 16,
            "terrain_types": ["flat"],
            "terrain_height_sources": ["height_scanner_nearest"],
            "dr_levels": ["0"],
            "push_frames": 0,
        },
        commands=commands,
        posture={
            "max_roll_deg": 38.7,
            "max_pitch_deg": 48.4,
            "min_height_m": 0.33,
            "stable_fraction": 0.77,
        },
        events=[
            {"type": "high_slip_bout", "count": 57},
            {"type": "touchdown_observed", "count": 224},
            {"type": "large_pitch_bout", "count": 8},
            {"type": "reset_observed", "count": 8},
        ],
        notes=[
            {
                "priority": "high",
                "type": "early_major_event_in_segment",
                "message": "A large pitch event appears about 0.53 s after the backward command; controlled displacement is near zero.",
            },
            {
                "priority": "high",
                "type": "reset_or_done_observed",
                "message": "All four yaw command attempts observed reset.",
            },
        ],
        terrain={"types": ["flat"], "height_sources": ["height_scanner_nearest"]},
        robustness={"dr_levels": ["0"], "dr_columns": [], "push_frames": 0},
        plan=plan,
    )
    report_data = {
        "coverage": base.coverage,
        "commands": base.commands,
        "posture": base.posture,
        "events": base.events,
        "notes": base.notes,
        "terrain": base.terrain,
        "robustness": base.robustness,
    }
    base.value_explanations = _build_value_explanations(job, report_data)
    for item in base.value_explanations:
        item.source = "fake metrics fixture"
        if item.confidence == "high":
            item.confidence = "medium"
    return base
