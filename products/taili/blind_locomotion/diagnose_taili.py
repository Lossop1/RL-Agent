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
import subprocess
import sys
import time
import traceback
from typing import Any

try:
    from .taili_core import taili_geometry
except ImportError:
    try:
        from products.taili.core import taili_geometry
    except ImportError:
        from taili_core import taili_geometry


RECORD_SCHEMA_VERSION = "ilqd_observation_record_v0.6.0"


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
    suite = _load_yaml(args.suite)
    terrains = [item for item in (suite.get("terrains") or []) if isinstance(item, dict)]
    if len(terrains) > 1 and os.environ.get("TAILI_DIAG_SINGLE_TERRAIN") != "1":
        _run_isolated_terrain_suite(args, suite, out_dir)
        return
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


def _run_isolated_terrain_suite(args, suite: dict[str, Any], out_dir: Path) -> None:
    """每个地形使用独立 IsaacLab 进程，随后合并无判定的物理记录。"""
    import yaml

    terrains = [item for item in (suite.get("terrains") or []) if isinstance(item, dict)]
    case_root = out_dir / "terrain_cases"
    case_root.mkdir(parents=True, exist_ok=True)
    child_dirs: list[Path] = []
    for terrain_index, terrain in enumerate(terrains):
        child_dir = case_root / f"terrain_{terrain_index:03d}"
        child_dir.mkdir(parents=True, exist_ok=True)
        child_suite = dict(suite)
        child_suite["name"] = f"{suite.get('name', 'diagnostic')}_terrain_{terrain_index:03d}"
        child_suite["terrains"] = [terrain]
        child_suite_path = child_dir / "suite.yaml"
        child_suite_path.write_text(
            yaml.safe_dump(child_suite, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            "-m",
            "taili_blind_runtime.diagnose_taili",
            "--task",
            args.task,
            "--checkpoint",
            args.checkpoint,
            "--suite",
            str(child_suite_path),
            "--out",
            str(child_dir),
            "--device",
            args.device,
        ]
        if args.num_envs is not None:
            command.extend(["--num-envs", str(args.num_envs)])
        if args.agent_yaml:
            command.extend(["--agent-yaml", args.agent_yaml])
        if args.headless:
            command.append("--headless")
        child_env = os.environ.copy()
        child_env["TAILI_DIAG_SINGLE_TERRAIN"] = "1"
        child_env["TAILI_RUN_DIR"] = str(child_dir)
        child_env["TAILI_RUN_ID"] = child_dir.name
        child_env["TAILI_TRAIN_LOG"] = str(child_dir / "diagnostic.train.log")
        child_env["TAILI_TELEMETRY_JSONL"] = str(child_dir / "diagnostic.telemetry.jsonl")
        child_env["TAILI_CONSOLE_LOG"] = str(child_dir / "diagnostic.console.log")
        child_env["TAILI_CHECKPOINT_DIR"] = str(child_dir / "checkpoints")
        print(
            f"[TAILI_DIAG] isolated terrain {terrain_index + 1}/{len(terrains)}: "
            f"{terrain.get('type', 'flat')}",
            flush=True,
        )
        completed = subprocess.run(command, env=child_env, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"isolated terrain diagnostic failed: index={terrain_index} rc={completed.returncode}"
            )
        child_dirs.append(child_dir)

    record_path = out_dir / "record.csv"
    fieldnames: list[str] | None = None
    rows_written = 0
    case_offset = 0
    step_offset = 0
    time_offset = 0.0
    merged_cases: list[dict[str, Any]] = []
    first_meta: dict[str, Any] | None = None
    with record_path.open("w", newline="", encoding="utf-8") as output:
        writer = None
        for child_dir in child_dirs:
            meta = json.loads((child_dir / "record_meta.json").read_text(encoding="utf-8"))
            if first_meta is None:
                first_meta = meta
            for case in meta.get("executed_runtime_config", {}).get("cases", []):
                merged = dict(case)
                merged["case_id"] = int(case.get("case_id", 0)) + case_offset
                merged_cases.append(merged)
            with (child_dir / "record.csv").open("r", newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                if fieldnames is None:
                    fieldnames = list(reader.fieldnames or [])
                    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                child_rows = list(reader)
            max_case = -1
            max_step = -1
            max_time = 0.0
            for row in child_rows:
                original_case = int(float(row.get("case_id", 0) or 0))
                original_step = int(float(row.get("step", 0) or 0))
                original_time = float(row.get("time", 0.0) or 0.0)
                row["case_id"] = original_case + case_offset
                row["step"] = original_step + step_offset
                row["time"] = original_time + time_offset
                writer.writerow(row)
                rows_written += 1
                max_case = max(max_case, original_case)
                max_step = max(max_step, original_step)
                max_time = max(max_time, original_time)
            case_offset += max_case + 1
            step_offset += max_step + 1
            time_offset += max_time + (float(child_rows[-1].get("control_dt", 0.0)) if child_rows else 0.0)

    if first_meta is None:
        raise RuntimeError("isolated terrain diagnostics produced no metadata")
    merged_meta = dict(first_meta)
    merged_meta["suite_path"] = args.suite
    merged_meta["requested_suite_config"] = suite
    merged_meta["status"] = "complete"
    merged_meta["rows_written"] = rows_written
    runtime_config = dict(merged_meta.get("executed_runtime_config", {}))
    runtime_config["cases"] = merged_cases
    runtime_config["terrain_process_isolation"] = True
    runtime_config["terrain_case_outputs"] = [str(path) for path in child_dirs]
    merged_meta["executed_runtime_config"] = runtime_config
    _write_json(out_dir / "record_meta.json", merged_meta)
    dr_count = max(1, len([item for item in (suite.get("dr_cases") or []) if isinstance(item, dict)]))
    command_count = max(1, len([item for item in (suite.get("commands") or []) if isinstance(item, dict)]))
    total_cases = len(terrains) * dr_count
    _write_json(
        out_dir / "record_progress.json",
        {
            "completed_cases": total_cases,
            "requested_cases": total_cases,
            "completed_segments": total_cases * command_count,
            "requested_segments": total_cases * command_count,
            "rows_written": rows_written,
            "status": "complete",
            "terrain_process_isolation": True,
        },
    )
    try:
        from .isaaclab_quad_diag.metrics import compute_all_metrics
    except ImportError:
        from isaaclab_quad_diag.metrics import compute_all_metrics
    compute_all_metrics(record_path, out_dir / "metrics", out_dir / "record_meta.json")
    _enrich_metrics_with_suite(out_dir / "metrics" / "metrics.json", suite, record_path)
    print(f"[TAILI_DIAG] merged isolated terrains: rows={rows_written} cases={total_cases}", flush=True)


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


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _extend_diagnostic_episode_horizon(base: Any, planned_steps: int, margin_steps: int) -> dict[str, int]:
    """Keep a routine episode timeout from splitting one diagnostic plan."""
    previous = max(1, int(getattr(base, "max_episode_length", 1) or 1))
    current_step = 0
    episode_length_buf = getattr(base, "episode_length_buf", None)
    if episode_length_buf is not None:
        try:
            current_step = max(0, int(episode_length_buf.max().item()))
        except (AttributeError, TypeError, ValueError):
            current_step = 0
    planned_steps = max(1, int(planned_steps))
    margin_steps = max(1, int(margin_steps))
    required = max(previous, current_step + planned_steps + margin_steps + 1)
    cfg = getattr(base, "cfg", None)
    step_dt = float(getattr(base, "step_dt", 0.0) or 0.0)
    if cfg is None or not hasattr(cfg, "episode_length_s") or step_dt <= 0.0:
        raise RuntimeError("diagnostic cannot extend episode horizon: cfg.episode_length_s or step_dt is unavailable")
    cfg.episode_length_s = max(float(cfg.episode_length_s), required * step_dt)
    effective = int(base.max_episode_length)
    return {
        "previous_max_episode_steps": previous,
        "starting_episode_step": current_step,
        "planned_steps": planned_steps,
        "margin_steps": margin_steps,
        "effective_max_episode_steps": effective,
    }


def _force_env_reset(env: Any):
    """强制 skrl IsaacLab wrapper 再次调用底层环境 reset。"""
    if hasattr(env, "_reset_once"):
        env._reset_once = True
    return env.reset()


def _enrich_metrics_with_suite(metrics_path: Path, suite: dict[str, Any], record_path: Path | None = None) -> None:
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
    if record_path is not None and record_path.is_file():
        try:
            from .stair_validation import compute_stair_validation
        except ImportError:
            from stair_validation import compute_stair_validation
        payload["stair_validation"] = compute_stair_validation(record_path, suite)
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
        # 优先保留训练配置中的真实地形类型。若把 stairs_up 几何挂到 stairs 键下，
        # 环境会把物理上行误标为下行，进而锁定错误的事件方向和奖励链路。
        sub_key = canonical
        # IsaacLab 配置中的 terrain_generator 可能由多个环境共享。每个诊断 case 都复制一份，
        # 后续的比例和方向调整不会污染其他 case，也不会改变默认配置对象。
        import copy as _copy
        generator = _copy.deepcopy(env_cfg.terrain.terrain_generator)
        env_cfg.terrain.terrain_generator = generator
        sub_terrains = generator.sub_terrains
        # 兼容没有独立 stairs_up 键的旧运行包：仅在确实缺失时才复用并反转 stairs。
        if sub_key not in sub_terrains and canonical == "stairs_up" and "stairs" in sub_terrains:
            sub_key = "stairs"
        if sub_key not in sub_terrains:
            env_cfg.terrain.terrain_type = "plane"
            return "flat"
        # 现在可以在隔离副本上选择单一地形或替换倒置楼梯，而不会影响后续 gym.make。
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
    expanded = target.unsqueeze(0).expand(base.num_envs, -1)
    updated = False
    if hasattr(base, "_cmd_target"):
        base._cmd_target[:] = expanded
        updated = True
    if hasattr(base, "commands"):
        # 诊断协议测量原始命令响应，不经过训练命令采样器或过渡器。
        base.commands[:] = expanded
        updated = True
    if not updated:
        raise AttributeError("Taili env does not expose commands/_cmd_target")


def _sync_external_command_observation(observation, base):
    """把刚切换的外部命令同步到当前 actor 观测，避免首帧仍使用旧命令。"""
    import torch

    commands = getattr(base, "commands", None)
    if commands is None:
        return observation

    def _sync_tensor(value):
        if torch.is_tensor(value) and value.ndim >= 2 and value.shape[-1] >= 9:
            value[..., 6:9].copy_(commands.to(device=value.device, dtype=value.dtype))

    if isinstance(observation, dict):
        for key in ("policy", "states", "observation"):
            if key in observation:
                _sync_tensor(observation[key])
    else:
        _sync_tensor(observation)
    return observation


def _enable_external_commands(base) -> None:
    """诊断必须显式创建该开关；训练环境默认不会预先声明这个属性。"""
    base.use_external_commands = True


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
        "task_name", "robot_name", "foot_radius", "nominal_stand_height",
        "terrain_type_requested", "terrain_type", "terrain_level", "terrain_height_source",
        "dr_level_requested", "dr_level",
        "capture_stage", "terminal_state_available", "post_step_state_may_be_after_reset", "transition_done_after_action",
        "terminated", "truncated", "done", "reset_observed",
        "would_terminate", "would_reset", "diagnostic_reset_suppressed", "termination_height",
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
            f"foot_{leg}_terrain_height", f"foot_{leg}_center_clearance_local",
            f"foot_{leg}_clearance_local", f"foot_{leg}_contact",
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
    prev_foot_vz,
    contact_valid,
    done,
    terminated,
    truncated,
    terrain_requested: str = "flat",
    terrain_level_requested: int = 0,
    dr_case: dict[str, Any] | None = None,
    push_event: dict[str, Any] | None = None,
    suppress_strict_reset: bool = False,
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
    foot_ang_t = getattr(data, "body_ang_vel_w", None)
    if foot_ang_t is not None and foot_vel_t is not None:
        foot_ang = foot_ang_t[:, foot_idx, :].detach().cpu().numpy()
        support_normal = getattr(base, "_support_reference_normal", None)
        if support_normal is not None:
            normal_np = support_normal.detach().cpu().numpy()
        else:
            normal_np = np.zeros((base.num_envs, 3), dtype=foot_vel.dtype)
            normal_np[:, 2] = 1.0
        contact_offset = -taili_geometry.FOOT_RADIUS * normal_np[:, None, :]
        foot_contact_vel = foot_vel + np.cross(foot_ang, contact_offset)
    else:
        foot_contact_vel = np.full_like(foot_pos, np.nan)
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
        base_height_local = float(root_pos[env_id, 2]) - terrain_h
        termination_height = float(getattr(base.cfg, "termination_height", float("nan")))
        finite_root = bool(
            np.isfinite(root_pos[env_id]).all()
            and np.isfinite(root_lin[env_id]).all()
            and np.isfinite(root_ang[env_id]).all()
        )
        would_terminate_height = (
            np.isfinite(base_height_local)
            and np.isfinite(termination_height)
            and base_height_local < termination_height
        )
        would_terminate = bool(would_terminate_height or (not finite_root))
        would_reset = bool(would_terminate or truncated[env_id])
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
            "foot_radius": float(getattr(base.cfg, "foot_radius", taili_geometry.FOOT_RADIUS)),
            "nominal_stand_height": float(
                getattr(base.cfg, "stand_height", taili_geometry.NOMINAL_BASE_HEIGHT)
            ),
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
            "would_terminate": int(would_terminate),
            "would_reset": int(would_reset),
            "diagnostic_reset_suppressed": int(bool(suppress_strict_reset and would_terminate and not done[env_id])),
            "termination_height": termination_height,
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
            "base_height_local": base_height_local,
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
            cp_vx, cp_vy, _ = foot_contact_vel[env_id, li]
            precontact_down_vz = max(-float(prev_foot_vz[env_id, li]), 0.0)
            fwx, fwy, fwz = force[env_id, li]
            row.update({
                f"foot_{leg}_pos_w_x": float(fx),
                f"foot_{leg}_pos_w_y": float(fy),
                f"foot_{leg}_pos_w_z": float(fz),
                f"foot_{leg}_vel_w_x": float(vx),
                f"foot_{leg}_vel_w_y": float(vy),
                f"foot_{leg}_vel_w_z": float(vz),
                f"foot_{leg}_terrain_height": th,
                f"foot_{leg}_center_clearance_local": float(fz) - th,
                f"foot_{leg}_clearance_local": taili_geometry.sole_clearance(float(fz), th),
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
                f"foot_{leg}_touchdown_vz": (
                    precontact_down_vz if touchdown and np.isfinite(prev_foot_vz[env_id, li]) else float("nan")
                ),
                f"foot_{leg}_stance_slip_xy": (
                    math.sqrt(float(cp_vx) ** 2 + float(cp_vy) ** 2)
                    if contact_now and np.isfinite(cp_vx) and np.isfinite(cp_vy)
                    else float("nan")
                ),
            })
            if row["terrain_height_source"] == "env_origin_fallback" and thsrc != "env_origin_fallback":
                row["terrain_height_source"] = thsrc
        rows.append(row)
    return rows


def run_diagnostic(args) -> None:  # pragma: no cover - requires IsaacLab runtime
    import copy
    import numpy as np
    import torch
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from skrl.utils.runner.torch import Runner
    from autotuner.simulation.backend_factory import create_backend

    import taili_blind_runtime  # noqa: F401 - register task and skrl policy component

    suite = _load_yaml(args.suite)
    init_phase = int(float(suite.get("init_phase", os.environ.get("TAILI_INIT_PHASE", 0)) or 0))
    os.environ["TAILI_INIT_PHASE"] = str(max(0, min(9, init_phase)))
    print(f"[TAILI_DIAG] init_phase={os.environ['TAILI_INIT_PHASE']}", flush=True)
    strict_reset = _as_bool(
        suite.get("diagnostic_strict_reset", suite.get("strict_reset", os.environ.get("TAILI_DIAG_STRICT_RESET"))),
        default=False,
    )
    suppress_strict_reset = not strict_reset
    reset_policy_note = "training_strict_reset" if strict_reset else "observe_without_routine_resets"
    print(f"[TAILI_DIAG] diagnostic_reset_policy={reset_policy_note}", flush=True)
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
        "schema_version": RECORD_SCHEMA_VERSION,
        "task": args.task,
        "checkpoint": args.checkpoint,
        "suite_path": args.suite,
        "requested_suite_config": suite,
        "executed_runtime_config": {
            "cases": [],
            "record_capture": "post_step_terminal_safe",
            "reset_initialization": "command_start",
            "diagnostic_reset_policy": reset_policy_note,
            "diagnostic_strict_reset": strict_reset,
            "height_termination_recorded_as": "would_terminate",
            "episode_timeout_policy": "training_timeout" if strict_reset else "extended_to_cover_diagnostic_plan",
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
            "foot clearance is sole clearance; center clearance is recorded separately",
            (
                "diagnostic suppresses training height termination and records would_terminate/diagnostic_reset_suppressed"
                if suppress_strict_reset
                else "diagnostic uses training-strict reset behavior"
            ),
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
    # parse_env_cfg 的部分 configclass 默认对象由类级共享。保留一份尚未交给
    # IsaacLab 的完整模板，每个地形 case 深拷贝整个对象图，避免首个环境构建
    # 写入运行时引用后污染下一次 cfg.validate()。
    env_cfg_template = parse_env_cfg(args.task, device=args.device, num_envs=num_envs)

    with record_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        case_id = 0
        for terrain_index, terrain in enumerate(terrains):
            terrain_requested = str(terrain.get("type") or "flat")
            terrain_level_requested = int(terrain.get("level", 0) or 0)
            env_cfg = copy.deepcopy(env_cfg_template)
            terrain_effective = _configure_terrain(env_cfg, terrain)
            if suppress_strict_reset and hasattr(env_cfg, "early_termination"):
                env_cfg.early_termination = False
            if hasattr(env_cfg, "reset_strategy"):
                env_cfg.reset_strategy = "start"
            _progress(out_dir, stage="gym_make", rows_written=rows_written, active_terrain=terrain_requested)
            env = create_backend(args.task, env_cfg, backend_type="isaaclab")
            try:
                _progress(out_dir, stage="skrl_wrapper", rows_written=rows_written, active_terrain=terrain_requested)
                env = SkrlVecEnvWrapper(env, ml_framework="torch")
                base = env.unwrapped
                if suppress_strict_reset and hasattr(base.cfg, "early_termination"):
                    base.cfg.early_termination = False
                _enable_external_commands(base)
                _progress(out_dir, stage="build_runner", rows_written=rows_written, active_terrain=terrain_requested)
                runner = Runner(env, experiment_cfg)
                _progress(out_dir, stage="load_checkpoint", rows_written=rows_written, active_terrain=terrain_requested)
                _load_evaluation_checkpoint(runner.agent, args.checkpoint)
                runner.agent.set_running_mode("eval")
                _progress(out_dir, stage="checkpoint_loaded", rows_written=rows_written, active_terrain=terrain_requested)

                dt = float(getattr(base, "step_dt", getattr(base, "physics_dt", 0.005) * getattr(base.cfg, "decimation", 1)))
                planned_steps = sum(
                    max(1, int(round(float(command["duration"]) / dt)))
                    for command in commands
                )
                timeout_margin_steps = max(10, int(round(2.0 / dt)))
                episode_id = np.zeros(base.num_envs, dtype=int)
                prev_contact = np.zeros((base.num_envs, 4), dtype=bool)
                prev_foot_vz = np.full((base.num_envs, 4), np.nan, dtype=float)
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
                obs, _ = _force_env_reset(env)
                _set_external_command(base, first_target)
                obs = _sync_external_command_observation(obs, base)
                print(f"[TAILI_DIAG] reset complete terrain={terrain_requested} effective={terrain_effective} num_envs={base.num_envs} dt={dt:.4f}", flush=True)
                for dr_case in dr_cases:
                    dr_applied = _apply_dr_case(base, dr_case)
                    obs, _ = _force_env_reset(env)
                    episode_horizon = None
                    if suppress_strict_reset:
                        episode_horizon = _extend_diagnostic_episode_horizon(
                            base,
                            planned_steps=planned_steps,
                            margin_steps=timeout_margin_steps,
                        )
                        print(f"[TAILI_DIAG] episode_horizon={episode_horizon}", flush=True)
                    episode_id += 1
                    prev_contact[:] = False
                    prev_foot_vz[:] = np.nan
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
                            "diagnostic_episode_horizon": episode_horizon,
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
                        obs = _sync_external_command_observation(obs, base)
                        if str(suite.get("reset_policy", "per_case")) == "per_segment" and segment_id > 0:
                            obs, _ = _force_env_reset(env)
                            _set_external_command(base, target)
                            obs = _sync_external_command_observation(obs, base)
                            episode_id += 1
                            prev_contact[:] = False
                            prev_foot_vz[:] = np.nan
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
                            obs = _sync_external_command_observation(obs, base)
                            with torch.inference_mode():
                                act_out = runner.agent.act(obs, timestep=0, timesteps=0)
                                action_mean = act_out[-1].get("mean_actions", act_out[0])
                            if not bool(torch.isfinite(action_mean).all()):
                                bad_count = int((~torch.isfinite(action_mean)).sum().item())
                                raise RuntimeError(
                                    f"checkpoint produced non-finite actions before env.step: "
                                    f"{bad_count}/{action_mean.numel()} values are NaN or Inf"
                                )
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
                                prev_foot_vz=prev_foot_vz,
                                contact_valid=contact_valid,
                                done=done,
                                terminated=terminated_np,
                                truncated=truncated_np,
                                terrain_requested=terrain_requested,
                                terrain_level_requested=terrain_level_requested,
                                dr_case=dr_case,
                                push_event=active_push,
                                suppress_strict_reset=suppress_strict_reset,
                            )
                            for row in rows:
                                writer.writerow(row)
                                rows_written += 1
                            for env_id, row in enumerate(rows):
                                prev_contact[env_id, :] = [bool(row[f"foot_{leg}_contact"]) for leg in ("FL", "FR", "RL", "RR")]
                                prev_foot_vz[env_id, :] = [float(row[f"foot_{leg}_vel_w_z"]) for leg in ("FL", "FR", "RL", "RR")]
                                contact_valid[env_id] = True
                                if done[env_id]:
                                    episode_id[env_id] += 1
                                    prev_contact[env_id, :] = False
                                    prev_foot_vz[env_id, :] = np.nan
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
    _enrich_metrics_with_suite(out_dir / "metrics" / "metrics.json", suite, record_path)
    print(f"[TAILI_DIAG] record written to: {record_path} rows={rows_written}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
