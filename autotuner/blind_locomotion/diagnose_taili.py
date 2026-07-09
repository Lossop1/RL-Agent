"""Payload-local Taili blind diagnostic runner.

This is the diagnostics counterpart of ``train_taili.py``: it runs entirely from
the self-contained ``taili_blind_runtime`` payload and does not import
``robot_lab``.  It emits the same ILQD-style artifacts the console already knows
how to read:

    record.csv
    record_meta.json
    record_progress.json
    metrics/metrics.json

The runner is intentionally observation-first.  It executes fixed command
batteries with the policy mean action, records command-vs-actual motion and
hardware/contact state, then computes metrics from those rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run payload-local Taili blind command diagnostics.")
    p.add_argument("--task", default="RobotLab-Isaac-Taili-AMP-Blind-Direct-v0")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--agent-yaml", default="")
    return p


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - requires IsaacLab runtime
    args = _parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TAILI_RUN_DIR", str(out_dir))
    os.environ.setdefault("TAILI_RUN_ID", out_dir.name)
    os.environ.setdefault("TAILI_TRAIN_LOG", str(out_dir / "diagnostic.train.log"))
    os.environ.setdefault("TAILI_TELEMETRY_JSONL", str(out_dir / "diagnostic.telemetry.jsonl"))
    os.environ.setdefault("TAILI_CONSOLE_LOG", str(out_dir / "diagnostic.console.log"))
    os.environ.setdefault("TAILI_CHECKPOINT_DIR", str(out_dir / "checkpoints"))
    try:
        from isaaclab.app import AppLauncher
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("diagnose_taili must run inside an IsaacLab Python environment") from exc

    launch_args = argparse.Namespace(**vars(args))
    launch_args.headless = bool(args.headless)
    launch_args.enable_cameras = False
    app_launcher = AppLauncher(launch_args)
    app = app_launcher.app
    exit_code = 0
    try:
        run_diagnostic(args)
    except BaseException as exc:  # noqa: BLE001 - diagnostics must leave a readable failure artifact
        exit_code = int(exc.code) if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 1
        _write_json(
            out_dir / "record_error.json",
            {
                "status": "error",
                "stage": "diagnostic_runner",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "updated_at": _now_iso(),
            },
        )
        _progress(out_dir, stage="error", rows_written=0, status="error", error_type=type(exc).__name__)
        print(f"[TAILI_DIAG_ERROR] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
    finally:
        try:
            app.close()
        except Exception as close_exc:  # noqa: BLE001
            if exit_code == 0:
                exit_code = 1
            print(f"[TAILI_DIAG_ERROR] app.close failed: {type(close_exc).__name__}: {close_exc}", flush=True)
            traceback.print_exc()
    if exit_code:
        raise SystemExit(exit_code)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"suite root must be a mapping: {path}")
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _progress(out_dir: Path, *, stage: str, rows_written: int = 0, **extra: Any) -> None:
    payload = {
        "completed_segments": int(extra.pop("completed_segments", 0)),
        "rows_written": int(rows_written),
        "status": str(extra.pop("status", "running")),
        "stage": stage,
        "updated_at": _now_iso(),
    }
    payload.update(extra)
    _write_json(out_dir / "record_progress.json", payload)
    print(f"[TAILI_DIAG] stage={stage} rows={rows_written}", flush=True)


def _clean_command(cmd: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": str(cmd.get("mode") or "unknown"),
        "vx": float(cmd.get("vx", 0.0) or 0.0),
        "vy": float(cmd.get("vy", 0.0) or 0.0),
        "wz": float(cmd.get("wz", 0.0) or 0.0),
        "duration": float(cmd.get("duration", 1.0) or 1.0),
        "ramp": float(cmd.get("ramp", 0.0) or 0.0),
        "label": str(cmd.get("label") or cmd.get("mode") or "command"),
    }


def _clean_terrain(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": str(item.get("type") or "flat"),
        "level": int(float(item.get("level", 0) or 0)),
        "params": item.get("params") if isinstance(item.get("params"), dict) else {},
    }


def _clean_dr_case(item: dict[str, Any]) -> dict[str, Any]:
    out = {
        "level": int(float(item.get("level", 0) or 0)),
        "label": str(item.get("label") or f"DR{item.get('level', 0)}"),
    }
    for key in ("friction", "mass_scale", "stiffness_scale", "damping_scale", "latency_steps"):
        if item.get(key) is not None:
            out[key] = item[key]
    return out


def _clean_push_events(pushes: Any) -> list[dict[str, Any]]:
    if not isinstance(pushes, dict) or not pushes.get("enabled"):
        return []
    result = []
    for item in pushes.get("events") or []:
        if not isinstance(item, dict):
            continue
        vector = item.get("vector") if isinstance(item.get("vector"), list) else [0.0, 0.0, 0.0]
        result.append(
            {
                "segment": int(float(item.get("segment", 0) or 0)),
                "time": float(item.get("time", 0.0) or 0.0),
                "vector": [float(vector[i] if i < len(vector) else 0.0) for i in range(3)],
            }
        )
    return result


def _enrich_metrics_with_suite(metrics_path: Path, suite: dict[str, Any]) -> None:
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"[TAILI_DIAG] metrics enrichment skipped: {type(exc).__name__}: {exc}", flush=True)
        return
    if not isinstance(payload, dict):
        return
    coverage = payload.setdefault("coverage", {})
    if not isinstance(coverage, dict):
        coverage = {}
        payload["coverage"] = coverage
    coverage["terrain_types_requested"] = sorted(
        {
            str(item.get("type"))
            for item in suite.get("terrains", [])
            if isinstance(item, dict) and item.get("type") is not None
        }
    )
    coverage["dr_levels_requested"] = sorted(
        {
            str(item.get("level"))
            for item in suite.get("dr_cases", [])
            if isinstance(item, dict) and item.get("level") is not None
        }
    )
    payload["suite_plan"] = {
        "name": suite.get("name", ""),
        "commands": suite.get("commands", []),
        "terrains": suite.get("terrains", []),
        "dr_cases": suite.get("dr_cases", []),
        "pushes": suite.get("pushes", {}),
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_experiment_cfg(args) -> dict[str, Any]:
    import yaml
    from isaaclab_tasks.utils import load_cfg_from_registry

    try:
        from .taili_blind_config import build_skrl_config
    except ImportError:
        from taili_blind_config import build_skrl_config

    if args.agent_yaml:
        with Path(args.agent_yaml).open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    else:
        loaded = load_cfg_from_registry(args.task, "skrl_amp_cfg_entry_point")
    if isinstance(loaded, dict) and "skrl" in loaded:
        return build_skrl_config(loaded)
    return loaded


def _load_evaluation_checkpoint(agent, path: str) -> None:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"unsupported checkpoint payload: {type(checkpoint).__name__}")
    for name in ("policy", "state_preprocessor"):
        if name not in checkpoint:
            raise KeyError(f"checkpoint is missing evaluation module: {name}")
        module = agent.checkpoint_modules.get(name)
        if module is None or not hasattr(module, "load_state_dict"):
            raise KeyError(f"agent is missing evaluation module: {name}")
        print(f"[TAILI_DIAG] loading module: {name}", flush=True)
        module.load_state_dict(checkpoint[name])
        if hasattr(module, "eval"):
            module.eval()
    print("[TAILI_DIAG] checkpoint loaded", flush=True)


_TERRAIN_CANONICAL = {
    "flat": "flat",
    "plane": "flat",
    "slope": "slope",
    "slope_up": "slope",
    "uphill": "slope",
    "slope_inv": "slope_inv",
    "slope_down": "slope_inv",
    "downhill": "slope_inv",
    "rough": "rough",
    "boxes": "boxes",
    "box": "boxes",
    "stairs": "stairs",
    "stairs_up": "stairs_up",
    "stair": "stairs",
    "stairs_down": "stairs",
}


def _configure_terrain(env_cfg, terrain: dict[str, Any]) -> str:
    requested = str(terrain.get("type") or "flat")
    level = max(0, int(terrain.get("level", 0) or 0))
    params = terrain.get("params") if isinstance(terrain.get("params"), dict) else {}
    canonical = _TERRAIN_CANONICAL.get(requested, requested)
    if canonical == "flat":
        env_cfg.terrain.terrain_type = "plane"
        env_cfg.terrain.max_init_terrain_level = 0
        return "flat"
    try:
        sub_key = "stairs" if canonical == "stairs_up" else canonical
        # DEEP-COPY isolation (0706): terrain_generator and its sub_terrains dict are CLASS-LEVEL
        # shared defaults — parse_env_cfg hands back the SAME objects on every call.  Mutating them
        # per case (proportion, direction swap) leaked into the next case; on the 2nd gym.make the
        # accumulated/half-swapped config made configclass._validate recurse to a RecursionError,
        # crashing any diagnostic with >1 terrain (e.g. 上/下楼梯对照).  Copy the generator onto THIS
        # env_cfg so every case mutates an isolated graph and the shared default stays pristine.
        import copy as _copy
        generator = _copy.deepcopy(env_cfg.terrain.terrain_generator)
        env_cfg.terrain.terrain_generator = generator
        sub_terrains = generator.sub_terrains
        if sub_key not in sub_terrains:
            env_cfg.terrain.terrain_type = "plane"
            return "flat"
        # On the isolated copy it is now safe to select one terrain by proportion (and to swap in an
        # inverted-stairs cfg below) without corrupting later gym.make calls.
        for name, cfg in sub_terrains.items():
            cfg.proportion = 1.0 if name == sub_key else 0.0
        sub = sub_terrains[sub_key]
        # DIRECTION: stairs_down / params.direction=down -> inverted pyramid stairs (real descent),
        # not the same ascending tile. Value-level swap only (never replace the sub_terrains dict).
        direction = str(params.get("direction")
                        or ("up" if requested.endswith("_up") or canonical == "stairs_up" else "")
                        or ("down" if requested.endswith(("_down", "downhill")) else "")).lower()
        # 几何真相(经实测确认): 普通金字塔楼梯出生点在塔顶中央平台 -> 前进=下楼梯;
        # 倒金字塔是坑、出生在坑底 -> 前进=上楼梯。所以 direction=up 用 Inverted。
        if sub_key == "stairs" and direction == "up":
            try:
                import isaaclab.terrains as _tg
                inv = _tg.MeshInvertedPyramidStairsTerrainCfg(
                    proportion=1.0,
                    step_height_range=getattr(sub, "step_height_range", (0.05, 0.18)),
                    step_width=getattr(sub, "step_width", 0.3),
                    platform_width=getattr(sub, "platform_width", 3.0),
                    border_width=getattr(sub, "border_width", 1.0),
                    holes=False)
                sub_terrains[sub_key] = inv
                sub = inv
            except Exception as _exc:  # noqa: BLE001
                print(f"[TAILI_DIAG] stairs_up(inverted) unavailable ({type(_exc).__name__}); using default stairs (descend)", flush=True)
        # EXPLICIT params beat curriculum levels: step_height (exact meters -> degenerate range),
        # then any other attr the sub-terrain cfg actually has (noise_range, slope_range, ...).
        if params.get("step_height") is not None and hasattr(sub, "step_height_range"):
            try:
                _h = float(params["step_height"])
                # 单位容错: >1.0 视为厘米输入(30 -> 0.30m);再夹紧到物理可行域 [0.02, 0.40]m。
                # 曾发生: step_height=30 被原样当 30米 -> 210米高塔,机器人"掉悬崖"。
                if _h > 1.0:
                    print(f"[TAILI_DIAG] step_height={_h} 视为厘米 -> {_h/100:.2f}m", flush=True)
                    _h = _h / 100.0
                _h = min(0.40, max(0.02, _h))
                sub.step_height_range = (_h, _h)
            except (TypeError, ValueError):
                pass
        for _k, _v in params.items():
            if _k in ("direction", "step_height") or not hasattr(sub, _k):
                continue
            try:
                _vals = _v if isinstance(_v, (list, tuple)) else [_v]
                if any(isinstance(x, (int, float)) and abs(float(x)) > 5.0 for x in _vals):
                    print(f"[TAILI_DIAG] 参数 {_k}={_v} 超出合理量级(>5m),已忽略", flush=True)
                    continue
                setattr(sub, _k, tuple(_v) if isinstance(_v, (list, tuple)) else _v)
            except Exception:  # noqa: BLE001
                pass
        env_cfg.terrain.terrain_type = "generator"
        generator.curriculum = True
        env_cfg.terrain.max_init_terrain_level = level
        return "stairs_up" if sub_key == "stairs" and direction == "up" else sub_key
    except Exception as exc:  # noqa: BLE001 - keep diagnostics available on minimal IsaacLab installs
        print(f"[TAILI_DIAG] terrain config fallback to plane for {requested}: {type(exc).__name__}: {exc}", flush=True)
        env_cfg.terrain.terrain_type = "plane"
        return "flat"


def _apply_dr_case(base, dr_case: dict[str, Any]) -> dict[str, Any]:
    requested_level = int(dr_case.get("level", 0) or 0)
    applied: dict[str, Any] = {"requested_level": requested_level, "level": requested_level}
    try:
        setattr(base, "_dr_level", requested_level)
    except Exception as exc:  # noqa: BLE001
        applied["level_error"] = f"{type(exc).__name__}: {exc}"
    if dr_case.get("latency_steps") is not None and hasattr(base.cfg, "action_delay_steps"):
        try:
            base.cfg.action_delay_steps = int(dr_case["latency_steps"])
            applied["latency_steps"] = int(dr_case["latency_steps"])
        except Exception as exc:  # noqa: BLE001
            applied["latency_error"] = f"{type(exc).__name__}: {exc}"
    for key in ("friction", "mass_scale", "stiffness_scale", "damping_scale"):
        if dr_case.get(key) is not None:
            applied[f"{key}_requested"] = dr_case[key]
            applied[f"{key}_status"] = "recorded_not_directly_overridden"
    return applied


def _set_external_command(base, target) -> None:
    if hasattr(base, "_cmd_target"):
        base._cmd_target[:] = target.unsqueeze(0).expand(base.num_envs, -1)
    if hasattr(base, "commands"):
        base.commands[:] = target.unsqueeze(0).expand(base.num_envs, -1)
    if not hasattr(base, "_cmd_target") and not hasattr(base, "commands"):
        raise AttributeError("Taili env does not expose commands/_cmd_target")


def _read_action_applied(base, fallback):
    for name in ("_delayed_action", "actions"):
        value = getattr(base, name, None)
        if value is not None:
            try:
                return value.detach().clone()
            except Exception:
                pass
    return fallback.detach().clone()


def _terrain_height(base, env_id: int, x: float, y: float, fallback: float = 0.0) -> tuple[float, str]:
    scanner = getattr(base, "_height_scanner", None)
    if scanner is not None:
        try:
            import torch

            hits = scanner.data.ray_hits_w[env_id]
            finite = torch.isfinite(hits).all(dim=-1)
            valid = hits[finite]
            if valid.numel() > 0:
                xy = valid[:, :2]
                q = torch.tensor([x, y], device=xy.device, dtype=xy.dtype)
                idx = torch.argmin(torch.sum((xy - q) ** 2, dim=-1))
                return float(valid[idx, 2]), "height_scanner_nearest"
        except Exception:
            pass
    try:
        return float(base._terrain.env_origins[env_id, 2]), "env_origin_fallback"
    except Exception:
        return float(fallback), "unavailable"


def _terrain_identity(base, env_id: int) -> tuple[str, float]:
    if getattr(getattr(base.cfg, "terrain", None), "terrain_type", None) == "plane":
        return "flat", 0.0
    try:
        import numpy as np

        terrain = base._terrain
        level = float(terrain.terrain_levels[env_id].detach().cpu().item())
        col = int(terrain.terrain_types[env_id].detach().cpu().item())
        generator = base.cfg.terrain.terrain_generator
        names = list(generator.sub_terrains.keys())
        proportions = np.asarray([float(generator.sub_terrains[name].proportion) for name in names], dtype=float)
        total = float(proportions.sum())
        if not names or total <= 0:
            return "unknown", level
        proportions = proportions / total
        num_cols = max(1, int(getattr(generator, "num_cols", len(names)) or len(names)))
        type_by_column = np.searchsorted(
            np.cumsum(proportions),
            (np.arange(num_cols, dtype=float) + 0.5) / num_cols,
            side="right",
        )
        if 0 <= col < len(type_by_column):
            idx = int(type_by_column[col])
            return (names[idx] if 0 <= idx < len(names) else "unknown"), level
        return "unknown", level
    except Exception:
        return "unknown", float("nan")


def _effort_limits(base):
    try:
        return base.robot.actuators["legs"].effort_limit
    except Exception:
        return getattr(base.robot.data, "joint_effort_limits", None)


def _build_columns(n_joints: int = 12, legs: list[str] | None = None) -> list[str]:
    legs = legs or ["FL", "FR", "RL", "RR"]
    cols = [
        "run_id", "case_id", "env_id", "episode_id", "step", "time", "control_dt", "physics_dt", "decimation",
        "task_name", "robot_name", "nominal_stand_height",
        "terrain_type_requested", "terrain_type", "terrain_level", "terrain_height_source",
        "dr_level_requested", "dr_level",
        "capture_stage", "terminal_state_available", "post_step_state_may_be_after_reset", "transition_done_after_action",
        "terminated", "truncated", "done", "reset_observed",
        "cmd_target_vx", "cmd_target_vy", "cmd_target_wz", "cmd_target_mode",
        "cmd_vx", "cmd_vy", "cmd_wz", "cmd_mode", "cmd_segment_id", "time_since_command_switch",
        "push_event", "push_vector_x", "push_vector_y", "push_vector_z", "push_equivalent_delta_v",
        "base_pos_w_x", "base_pos_w_y", "base_pos_w_z", "base_terrain_height", "base_height_local",
        "base_quat_w", "base_quat_x", "base_quat_y", "base_quat_z",
        "base_lin_vel_b_x", "base_lin_vel_b_y", "base_lin_vel_b_z",
        "base_ang_vel_b_x", "base_ang_vel_b_y", "base_ang_vel_b_z",
        "projected_gravity_b_x", "projected_gravity_b_y", "projected_gravity_b_z",
        "dr_mass", "dr_friction", "dr_com_x", "dr_com_y", "dr_com_z",
        "dr_stiffness_scale", "dr_damping_scale", "dr_latency",
    ]
    for i in range(n_joints):
        cols += [
            f"joint_pos_{i}", f"joint_vel_{i}", f"joint_pos_des_{i}", f"joint_error_{i}",
            f"torque_applied_{i}", f"torque_limit_{i}", f"torque_utilization_{i}",
            f"action_mean_{i}", f"action_applied_{i}",
        ]
    for leg in legs:
        cols += [
            f"foot_{leg}_pos_w_x", f"foot_{leg}_pos_w_y", f"foot_{leg}_pos_w_z",
            f"foot_{leg}_vel_w_x", f"foot_{leg}_vel_w_y", f"foot_{leg}_vel_w_z",
            f"foot_{leg}_terrain_height", f"foot_{leg}_clearance_local", f"foot_{leg}_contact",
            f"foot_{leg}_force_w_x", f"foot_{leg}_force_w_y", f"foot_{leg}_force_w_z", f"foot_{leg}_force_norm",
            f"foot_{leg}_normal_force", f"foot_{leg}_tangent_force",
            f"foot_{leg}_air_time", f"foot_{leg}_stance_time",
            f"foot_{leg}_touchdown", f"foot_{leg}_liftoff", f"foot_{leg}_touchdown_vz",
            f"foot_{leg}_stance_slip_xy",
        ]
    return cols


def _derive_mode(vx: float, vy: float, wz: float) -> str:
    if abs(vx) <= 0.1 and abs(vy) <= 0.1 and abs(wz) <= 0.1:
        return "stand"
    if abs(wz) > max(abs(vx), abs(vy), 0.1):
        return "yaw"
    if abs(vy) > max(abs(vx), 0.1):
        return "lateral"
    return "forward" if vx >= 0 else "backward"


def _rows_from_state(
    *,
    base,
    action_mean,
    action_applied,
    target,
    stage: str,
    case_id: int,
    segment_id: int,
    step: int,
    t: float,
    dt: float,
    mode: str,
    time_since_command_switch: float,
    episode_id,
    prev_contact,
    contact_valid,
    done,
    terminated,
    truncated,
    terrain_requested: str = "flat",
    terrain_level_requested: int = 0,
    dr_case: dict[str, Any] | None = None,
    push_event: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    import math
    import numpy as np
    import torch

    robot = base.robot
    data = robot.data
    legs = ["FL", "FR", "RL", "RR"]
    joint_names = list(data.joint_names)
    joint_order = [f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf") for leg in legs]
    joint_idx = [joint_names.index(name) for name in joint_order]
    body_names = list(data.body_names)
    foot_idx = [body_names.index(f"{leg}_foot") for leg in legs]
    contact_sensor = getattr(base, "_contact_sensor", None)
    contact_idx = None
    if contact_sensor is not None:
        try:
            contact_idx, _ = contact_sensor.find_bodies([f"{leg}_foot" for leg in legs])
        except Exception:
            contact_idx = None

    q = data.joint_pos[:, joint_idx].detach().cpu().numpy()
    dq = data.joint_vel[:, joint_idx].detach().cpu().numpy()
    qdes = getattr(data, "joint_pos_target", None)
    qdes_np = qdes[:, joint_idx].detach().cpu().numpy() if qdes is not None else np.full_like(q, np.nan)
    torque = getattr(data, "applied_torque", getattr(data, "computed_torque", None))
    torque_np = torque[:, joint_idx].detach().cpu().numpy() if torque is not None else np.full_like(q, np.nan)
    limits = _effort_limits(base)
    if limits is None:
        limit_np = np.full_like(q, np.nan)
    else:
        limit_t = limits.detach().cpu() if hasattr(limits, "detach") else torch.as_tensor(limits)
        if limit_t.ndim == 1:
            limit_np = np.broadcast_to(limit_t.numpy()[joint_idx], q.shape)
        else:
            limit_np = limit_t[:, joint_idx].numpy()

    foot_pos = data.body_pos_w[:, foot_idx, :].detach().cpu().numpy()
    foot_vel_t = getattr(data, "body_lin_vel_w", None)
    foot_vel = foot_vel_t[:, foot_idx, :].detach().cpu().numpy() if foot_vel_t is not None else np.full_like(foot_pos, np.nan)
    if contact_sensor is not None and contact_idx is not None:
        force = contact_sensor.data.net_forces_w[:, contact_idx, :].detach().cpu().numpy()
        force_norm = np.linalg.norm(force, axis=-1)
        contact = force_norm > 5.0
        try:
            air_time = contact_sensor.data.current_air_time[:, contact_idx].detach().cpu().numpy()
        except Exception:
            air_time = np.full(contact.shape, np.nan)
    else:
        force = np.full_like(foot_pos, np.nan)
        force_norm = np.full((base.num_envs, len(legs)), np.nan)
        contact = np.zeros((base.num_envs, len(legs)), dtype=bool)
        air_time = np.full((base.num_envs, len(legs)), np.nan)

    command = getattr(base, "commands", target.unsqueeze(0).expand(base.num_envs, -1)).detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    action_mean_np = action_mean.detach().cpu().numpy()
    action_applied_np = action_applied.detach().cpu().numpy()
    root_pos = data.root_pos_w.detach().cpu().numpy()
    root_quat = data.root_quat_w.detach().cpu().numpy()
    root_lin = data.root_lin_vel_b.detach().cpu().numpy()
    root_ang = data.root_ang_vel_b.detach().cpu().numpy()
    gravity = data.projected_gravity_b.detach().cpu().numpy()
    dr_case = dr_case or {"level": 0}
    push_event = push_event or {}
    push_vector = push_event.get("vector") if isinstance(push_event.get("vector"), list) else [float("nan")] * 3
    push_mag = math.sqrt(sum(float(push_vector[i]) ** 2 for i in range(3))) if push_event else float("nan")

    rows: list[dict[str, Any]] = []
    for env_id in range(base.num_envs):
        terrain_type, terrain_level = _terrain_identity(base, env_id)
        terrain_h, terrain_source = _terrain_height(base, env_id, root_pos[env_id, 0], root_pos[env_id, 1])
        cmd = command[env_id]
        row: dict[str, Any] = {
            "run_id": f"case_{case_id}",
            "case_id": case_id,
            "env_id": env_id,
            "episode_id": int(episode_id[env_id]),
            "step": step,
            "time": t,
            "control_dt": dt,
            "physics_dt": float(getattr(base, "physics_dt", float("nan"))),
            "decimation": int(getattr(base.cfg, "decimation", 1)),
            "task_name": getattr(base.cfg, "task_name", "taili_blind_runtime"),
            "robot_name": "taili",
            "nominal_stand_height": float(getattr(base.cfg, "stand_height", 0.52)),
            "terrain_type_requested": terrain_requested,
            "terrain_type": terrain_type,
            "terrain_level": terrain_level,
            "terrain_height_source": terrain_source,
            "dr_level_requested": str(dr_case.get("level", 0)),
            "dr_level": str(getattr(base, "_dr_level", 0)),
            "capture_stage": stage,
            "terminal_state_available": 1,
            "post_step_state_may_be_after_reset": 0,
            "transition_done_after_action": int(done[env_id]),
            "terminated": int(terminated[env_id]),
            "truncated": int(truncated[env_id]),
            "done": int(done[env_id]),
            "reset_observed": int(done[env_id]),
            "cmd_target_vx": float(target_np[0]),
            "cmd_target_vy": float(target_np[1]),
            "cmd_target_wz": float(target_np[2]),
            "cmd_target_mode": mode,
            "cmd_vx": float(cmd[0]),
            "cmd_vy": float(cmd[1]),
            "cmd_wz": float(cmd[2]),
            "cmd_mode": _derive_mode(float(cmd[0]), float(cmd[1]), float(cmd[2])),
            "cmd_segment_id": segment_id,
            "time_since_command_switch": time_since_command_switch,
            "push_event": int(bool(push_event)),
            "push_vector_x": float(push_vector[0]),
            "push_vector_y": float(push_vector[1]),
            "push_vector_z": float(push_vector[2]),
            "push_equivalent_delta_v": push_mag,
            "base_pos_w_x": float(root_pos[env_id, 0]),
            "base_pos_w_y": float(root_pos[env_id, 1]),
            "base_pos_w_z": float(root_pos[env_id, 2]),
            "base_terrain_height": terrain_h,
            "base_height_local": float(root_pos[env_id, 2]) - terrain_h,
            "base_quat_w": float(root_quat[env_id, 0]),
            "base_quat_x": float(root_quat[env_id, 1]),
            "base_quat_y": float(root_quat[env_id, 2]),
            "base_quat_z": float(root_quat[env_id, 3]),
            "base_lin_vel_b_x": float(root_lin[env_id, 0]),
            "base_lin_vel_b_y": float(root_lin[env_id, 1]),
            "base_lin_vel_b_z": float(root_lin[env_id, 2]),
            "base_ang_vel_b_x": float(root_ang[env_id, 0]),
            "base_ang_vel_b_y": float(root_ang[env_id, 1]),
            "base_ang_vel_b_z": float(root_ang[env_id, 2]),
            "projected_gravity_b_x": float(gravity[env_id, 0]),
            "projected_gravity_b_y": float(gravity[env_id, 1]),
            "projected_gravity_b_z": float(gravity[env_id, 2]),
            "dr_mass": float(dr_case.get("mass_scale", 1.0) or 1.0) - 1.0,
            "dr_friction": float(dr_case.get("friction", 1.0) or 1.0),
            "dr_com_x": 0.0,
            "dr_com_y": 0.0,
            "dr_com_z": 0.0,
            "dr_stiffness_scale": float(dr_case.get("stiffness_scale", 1.0) or 1.0),
            "dr_damping_scale": float(dr_case.get("damping_scale", 1.0) or 1.0),
            "dr_latency": float(dr_case.get("latency_steps", getattr(base.cfg, "action_delay_steps", 0)) or 0) * dt,
        }
        for i in range(12):
            row[f"joint_pos_{i}"] = float(q[env_id, i])
            row[f"joint_vel_{i}"] = float(dq[env_id, i])
            row[f"joint_pos_des_{i}"] = float(qdes_np[env_id, i])
            row[f"joint_error_{i}"] = float(qdes_np[env_id, i] - q[env_id, i]) if np.isfinite(qdes_np[env_id, i]) else float("nan")
            row[f"torque_applied_{i}"] = float(torque_np[env_id, i])
            row[f"torque_limit_{i}"] = float(limit_np[env_id, i])
            row[f"torque_utilization_{i}"] = (
                abs(float(torque_np[env_id, i])) / max(abs(float(limit_np[env_id, i])), 1e-9)
                if np.isfinite(limit_np[env_id, i]) else float("nan")
            )
            row[f"action_mean_{i}"] = float(action_mean_np[env_id, i])
            row[f"action_applied_{i}"] = float(action_applied_np[env_id, i])
        for li, leg in enumerate(legs):
            fx, fy, fz = foot_pos[env_id, li]
            th, thsrc = _terrain_height(base, env_id, float(fx), float(fy), terrain_h)
            contact_now = bool(contact[env_id, li])
            touchdown = int(contact_valid[env_id] and (not prev_contact[env_id, li]) and contact_now)
            liftoff = int(contact_valid[env_id] and prev_contact[env_id, li] and (not contact_now))
            vx, vy, vz = foot_vel[env_id, li]
            fwx, fwy, fwz = force[env_id, li]
            row.update({
                f"foot_{leg}_pos_w_x": float(fx),
                f"foot_{leg}_pos_w_y": float(fy),
                f"foot_{leg}_pos_w_z": float(fz),
                f"foot_{leg}_vel_w_x": float(vx),
                f"foot_{leg}_vel_w_y": float(vy),
                f"foot_{leg}_vel_w_z": float(vz),
                f"foot_{leg}_terrain_height": th,
                f"foot_{leg}_clearance_local": float(fz) - th,
                f"foot_{leg}_contact": int(contact_now),
                f"foot_{leg}_force_w_x": float(fwx),
                f"foot_{leg}_force_w_y": float(fwy),
                f"foot_{leg}_force_w_z": float(fwz),
                f"foot_{leg}_force_norm": float(force_norm[env_id, li]),
                f"foot_{leg}_normal_force": float(fwz),
                f"foot_{leg}_tangent_force": math.sqrt(float(fwx) ** 2 + float(fwy) ** 2) if np.isfinite(fwx) and np.isfinite(fwy) else float("nan"),
                f"foot_{leg}_air_time": float(air_time[env_id, li]),
                f"foot_{leg}_stance_time": float("nan"),
                f"foot_{leg}_touchdown": touchdown,
                f"foot_{leg}_liftoff": liftoff,
                f"foot_{leg}_touchdown_vz": abs(float(vz)) if touchdown and np.isfinite(vz) else float("nan"),
                f"foot_{leg}_stance_slip_xy": math.sqrt(float(vx) ** 2 + float(vy) ** 2) if contact_now and np.isfinite(vx) and np.isfinite(vy) else float("nan"),
            })
            if row["terrain_height_source"] == "env_origin_fallback" and thsrc != "env_origin_fallback":
                row["terrain_height_source"] = thsrc
        rows.append(row)
    return rows


def run_diagnostic(args) -> None:  # pragma: no cover - requires IsaacLab runtime
    import numpy as np
    import torch
    import gymnasium as gym
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from skrl.utils.runner.torch import Runner

    import taili_blind_runtime  # noqa: F401 - register task and skrl policy component

    suite = _load_yaml(args.suite)
    commands = [_clean_command(cmd) for cmd in (suite.get("commands") or [])]
    if not commands:
        commands = [_clean_command({"mode": "stand", "duration": 1.0})]
    terrains = [_clean_terrain(item) for item in (suite.get("terrains") or []) if isinstance(item, dict)]
    if not terrains:
        terrains = [_clean_terrain({"type": "flat", "level": 0})]
    dr_cases = [_clean_dr_case(item) for item in (suite.get("dr_cases") or []) if isinstance(item, dict)]
    if not dr_cases:
        dr_cases = [_clean_dr_case({"level": 0})]
    push_events = _clean_push_events(suite.get("pushes"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    record_path = out_dir / "record.csv"
    num_envs = int(args.num_envs or suite.get("num_envs") or 1)

    total_cases = max(1, len(terrains) * len(dr_cases))
    meta_base = {
        "schema_version": "ilqd_observation_record_v0.5.1",
        "task": args.task,
        "checkpoint": args.checkpoint,
        "suite_path": args.suite,
        "requested_suite_config": suite,
        "executed_runtime_config": {
            "cases": [],
            "record_capture": "post_step_terminal_safe",
            "reset_initialization": "command_start",
            "payload_local": True,
            "terrain_runtime_note": (
                "Each requested terrain is executed in its own IsaacLab environment so terrain generator settings "
                "are physically applied at construction time."
            ),
        },
        "num_envs_requested": num_envs,
        "recording_notes": [
            "payload-local Taili diagnostic; no robot_lab imports",
            "policy action uses mean_actions when available",
        ],
        "semantics": "Observation-only. No pass/fail labels are emitted by the recorder.",
    }

    rows_written = 0

    def write_meta(status: str) -> None:
        payload = dict(meta_base)
        payload.update(status=status, rows_written=rows_written)
        _write_json(out_dir / "record_meta.json", payload)

    write_meta("running")
    _progress(out_dir, stage="parse_env_cfg", rows_written=rows_written)

    experiment_cfg = _load_experiment_cfg(args)
    experiment_cfg.setdefault("trainer", {})["close_environment_at_exit"] = False
    experiment_cfg.setdefault("agent", {}).setdefault("experiment", {})["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    cols = _build_columns()
    global_step = 0
    t = 0.0

    with record_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        case_id = 0
        for terrain_index, terrain in enumerate(terrains):
            terrain_requested = str(terrain.get("type") or "flat")
            terrain_level_requested = int(terrain.get("level", 0) or 0)
            env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=num_envs)
            terrain_effective = _configure_terrain(env_cfg, terrain)
            if hasattr(env_cfg, "reset_strategy"):
                env_cfg.reset_strategy = "start"
            _progress(out_dir, stage="gym_make", rows_written=rows_written, active_terrain=terrain_requested)
            env = gym.make(args.task, cfg=env_cfg, render_mode=None)
            try:
                _progress(out_dir, stage="skrl_wrapper", rows_written=rows_written, active_terrain=terrain_requested)
                env = SkrlVecEnvWrapper(env, ml_framework="torch")
                base = env.unwrapped
                if hasattr(base, "use_external_commands"):
                    base.use_external_commands = True
                _progress(out_dir, stage="build_runner", rows_written=rows_written, active_terrain=terrain_requested)
                runner = Runner(env, experiment_cfg)
                _progress(out_dir, stage="load_checkpoint", rows_written=rows_written, active_terrain=terrain_requested)
                _load_evaluation_checkpoint(runner.agent, args.checkpoint)
                runner.agent.set_running_mode("eval")
                _progress(out_dir, stage="checkpoint_loaded", rows_written=rows_written, active_terrain=terrain_requested)

                dt = float(getattr(base, "step_dt", getattr(base, "physics_dt", 0.005) * getattr(base.cfg, "decimation", 1)))
                episode_id = np.zeros(base.num_envs, dtype=int)
                prev_contact = np.zeros((base.num_envs, 4), dtype=bool)
                contact_valid = np.zeros(base.num_envs, dtype=bool)

                def apply_push(vector: list[float]) -> None:
                    try:
                        vel = torch.cat([base.robot.data.root_lin_vel_w, base.robot.data.root_ang_vel_w], dim=-1).clone()
                        for i in range(3):
                            vel[:, i] += float(vector[i])
                        base.robot.write_root_com_velocity_to_sim(vel)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[TAILI_DIAG] push failed: {type(exc).__name__}: {exc}", flush=True)

                first = commands[0]
                first_target = torch.tensor([first["vx"], first["vy"], first["wz"]], device=base.device, dtype=torch.float32)
                _set_external_command(base, first_target)
                _progress(out_dir, stage="env_reset", rows_written=rows_written, active_terrain=terrain_requested)
                obs, _ = env.reset()
                _set_external_command(base, first_target)
                print(f"[TAILI_DIAG] reset complete terrain={terrain_requested} effective={terrain_effective} num_envs={base.num_envs} dt={dt:.4f}", flush=True)
                for dr_case in dr_cases:
                    dr_applied = _apply_dr_case(base, dr_case)
                    obs, _ = env.reset()
                    episode_id += 1
                    prev_contact[:] = False
                    contact_valid[:] = False
                    meta_base["executed_runtime_config"]["cases"].append(
                        {
                            "case_id": case_id,
                            "status": "executed",
                            "num_envs": num_envs,
                            "runtime": "taili_blind_runtime",
                            "terrain_requested": terrain,
                            "terrain_effective": terrain_effective,
                            "dr_requested": dr_case,
                            "dr_applied": dr_applied,
                        }
                    )
                    write_meta("running")
                    print(
                        f"[TAILI_DIAG] case={case_id} terrain={terrain_requested}@{terrain_level_requested} "
                        f"effective={terrain_effective} dr={dr_case.get('level', 0)}",
                        flush=True,
                    )
                    for segment_id, cmd in enumerate(commands):
                        target = torch.tensor([cmd["vx"], cmd["vy"], cmd["wz"]], device=base.device, dtype=torch.float32)
                        _set_external_command(base, target)
                        if str(suite.get("reset_policy", "per_case")) == "per_segment" and segment_id > 0:
                            obs, _ = env.reset()
                            _set_external_command(base, target)
                            episode_id += 1
                            prev_contact[:] = False
                            contact_valid[:] = False
                        mode = cmd["mode"]
                        steps = max(1, int(round(float(cmd["duration"]) / dt)))
                        print(f"[TAILI_DIAG] segment={segment_id} mode={mode} target=({cmd['vx']},{cmd['vy']},{cmd['wz']}) steps={steps}", flush=True)
                        _progress(
                            out_dir,
                            stage="rollout",
                            rows_written=rows_written,
                            active_case=case_id,
                            requested_cases=total_cases,
                            completed_cases=case_id,
                            active_segment=segment_id,
                            requested_segments=len(commands),
                            last_mode=mode,
                        )
                        fired_pushes: set[int] = set()
                        for k in range(steps):
                            active_push: dict[str, Any] | None = None
                            elapsed_in_segment = (k + 1) * dt
                            for push_index, event in enumerate(push_events):
                                if int(event.get("segment", 0)) != segment_id or push_index in fired_pushes:
                                    continue
                                if elapsed_in_segment >= float(event.get("time", 0.0) or 0.0):
                                    apply_push(event.get("vector") or [0.0, 0.0, 0.0])
                                    fired_pushes.add(push_index)
                                    active_push = event
                                    print(f"[TAILI_DIAG] push segment={segment_id} vector={event.get('vector')}", flush=True)
                            _set_external_command(base, target)
                            with torch.inference_mode():
                                act_out = runner.agent.act(obs, timestep=0, timesteps=0)
                                action_mean = act_out[-1].get("mean_actions", act_out[0])
                            action_applied = _read_action_applied(base, action_mean)
                            obs, _, terminated, truncated, _info = env.step(action_mean)
                            terminated_np = terminated.detach().cpu().numpy().astype(bool).reshape(-1)
                            truncated_np = truncated.detach().cpu().numpy().astype(bool).reshape(-1)
                            done = terminated_np | truncated_np
                            rows = _rows_from_state(
                                base=base,
                                action_mean=action_mean,
                                action_applied=action_applied,
                                target=target,
                                stage="post_step",
                                case_id=case_id,
                                segment_id=segment_id,
                                step=global_step,
                                t=t + dt,
                                dt=dt,
                                mode=mode,
                                time_since_command_switch=(k + 1) * dt,
                                episode_id=episode_id,
                                prev_contact=prev_contact,
                                contact_valid=contact_valid,
                                done=done,
                                terminated=terminated_np,
                                truncated=truncated_np,
                                terrain_requested=terrain_requested,
                                terrain_level_requested=terrain_level_requested,
                                dr_case=dr_case,
                                push_event=active_push,
                            )
                            for row in rows:
                                writer.writerow(row)
                                rows_written += 1
                            for env_id, row in enumerate(rows):
                                prev_contact[env_id, :] = [bool(row[f"foot_{leg}_contact"]) for leg in ("FL", "FR", "RL", "RR")]
                                contact_valid[env_id] = True
                                if done[env_id]:
                                    episode_id[env_id] += 1
                                    prev_contact[env_id, :] = False
                                    contact_valid[env_id] = False
                            t += dt
                            global_step += 1
                        handle.flush()
                        _write_json(
                            out_dir / "record_progress.json",
                            {
                                "completed_cases": case_id,
                                "requested_cases": total_cases,
                                "completed_segments": segment_id + 1,
                                "requested_segments": len(commands),
                                "rows_written": rows_written,
                                "last_mode": mode,
                                "status": "running",
                                "stage": "rollout",
                                "updated_at": _now_iso(),
                            },
                        )
                        write_meta("running")
                    case_id += 1
            finally:
                try:
                    env.close()
                except Exception:
                    pass

    write_meta("complete")
    _write_json(
        out_dir / "record_progress.json",
        {
            "completed_cases": total_cases,
            "requested_cases": total_cases,
            "completed_segments": len(commands),
            "requested_segments": len(commands),
            "rows_written": rows_written,
            "status": "complete",
        },
    )

    # Compute metrics in-process so the console can keep using the existing report reader.
    try:
        from .isaaclab_quad_diag.metrics import compute_all_metrics
    except ImportError:
        from isaaclab_quad_diag.metrics import compute_all_metrics

    compute_all_metrics(record_path, out_dir / "metrics", out_dir / "record_meta.json")
    _enrich_metrics_with_suite(out_dir / "metrics" / "metrics.json", suite)
    print(f"[TAILI_DIAG] record written to: {record_path} rows={rows_written}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
