"""Small telemetry emitter for Taili blind training.

This module is deliberately dependency-light so it can be copied to the remote
runtime package. It writes the observability streams the console expects:

- human-readable lines: [TPSTAT] / [TPCMD] / [TPREW] / [TPPEN] / [TPGATE] / [TPCURR] / [TPCKPT]
- machine-readable JSONL: train.telemetry.jsonl
- optional TensorBoard scalars when a writer is provided

The emitter is fail-open by design: telemetry must never crash training.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping


DATA_ROOT = os.environ.get("TAILI_DATA_ROOT", "/root/gpufree-data")
DEFAULT_RUN_ROOT = os.environ.get("TAILI_RUN_ROOT", f"{DATA_ROOT}/taili_runs")
RUN_ID = os.environ.get("TAILI_RUN_ID", "").strip()


def _default_log_path() -> str:
    if os.environ.get("TAILI_TRAIN_LOG"):
        return os.environ["TAILI_TRAIN_LOG"]
    if os.environ.get("TAILI_RUN_DIR"):
        return f"{os.environ['TAILI_RUN_DIR'].rstrip('/')}/train.log"
    name = f"{RUN_ID}.log" if RUN_ID else "taili_train.log"
    return f"{DATA_ROOT}/logs/{name}"


def _default_jsonl_path(log_path: str) -> str:
    if log_path.endswith(".log"):
        return log_path[:-4] + ".telemetry.jsonl"
    return log_path + ".telemetry.jsonl"


DEFAULT_LOG = _default_log_path()
DEFAULT_JSONL = os.environ.get("TAILI_TELEMETRY_JSONL", _default_jsonl_path(DEFAULT_LOG))


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_float_map(data: Mapping[str, Any]) -> dict[str, float]:
    return {str(k): _f(v) for k, v in data.items() if v is not None}


def _kv(items: list[tuple[str, Any]], *, digits: int = 3) -> str:
    parts: list[str] = []
    for key, value in items:
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            parts.append(f"{key}={1 if value else 0}")
        elif isinstance(value, int):
            parts.append(f"{key}={value}")
        elif isinstance(value, float):
            parts.append(f"{key}={value:.{digits}f}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


class TrainingTelemetryEmitter:
    def __init__(self, jsonl_path: str | None = None, writer: Any | None = None):
        self.log_path = os.environ.get("TAILI_TRAIN_LOG", DEFAULT_LOG)
        self.jsonl_path = jsonl_path or os.environ.get("TAILI_TELEMETRY_JSONL", DEFAULT_JSONL)
        self.run_root = os.environ.get("TAILI_RUN_ROOT", DEFAULT_RUN_ROOT)
        self.run_id = os.environ.get("TAILI_RUN_ID", RUN_ID)
        self.run_dir = os.environ.get("TAILI_RUN_DIR", "")
        self.checkpoint_dir = os.environ.get(
            "TAILI_CHECKPOINT_DIR",
            f"{self.run_dir.rstrip('/')}/checkpoints" if self.run_dir else f"{self.run_root.rstrip('/')}/checkpoints",
        )
        self.console_log = os.environ.get(
            "TAILI_CONSOLE_LOG",
            f"{self.run_dir.rstrip('/')}/console.log" if self.run_dir else "",
        )
        self.writer = writer
        self._started_at = time.time()
        self._last_wall = self._started_at
        self._last_step = 0
        self._paths_printed = False
        self._reward_cfg_printed = False

    def emit(
        self,
        *,
        step: int,
        total_steps: int | None = None,
        reward: Mapping[str, Any] | None = None,
        curriculum: Mapping[str, Any] | None = None,
        health: Mapping[str, Any] | None = None,
        command: Mapping[str, Any] | None = None,
        checkpoint: str = "",
        counters: Mapping[str, Any] | None = None,
        paths: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            now = time.time()
            dt = max(1e-6, now - self._last_wall)
            fps = (int(step) - int(self._last_step)) / dt if self._last_step else None
            self._last_wall = now
            self._last_step = int(step)

            reward_data = _clean_float_map(reward or {})
            curriculum_data = dict(curriculum or {})
            health_data = _clean_float_map(health or {})
            command_data = _clean_float_map(command or {})
            counter_data = _clean_float_map(counters or {})
            path_data = {
                "data_root": DATA_ROOT,
                "run_root": self.run_root,
                "run_id": self.run_id,
                "run_dir": self.run_dir,
                "log": self.log_path,
                "telemetry_jsonl": self.jsonl_path,
                "console_log": self.console_log,
                "checkpoint_dir": self.checkpoint_dir,
            }
            path_data.update({str(k): str(v) for k, v in (paths or {}).items() if v is not None})
            elapsed_s = max(0.0, now - self._started_at)
            eta = ""
            if total_steps and fps and fps > 0:
                eta_s = max(0.0, (int(total_steps) - int(step)) / fps)
                eta = _fmt_duration(eta_s)
            payload = {
                "type": "train_tick",
                "step": int(step),
                "total_steps": int(total_steps) if total_steps else None,
                "ts": now,
                "elapsed": _fmt_duration(elapsed_s),
                "eta": eta,
                "fps": fps,
                "reward": reward_data,
                "curriculum": curriculum_data,
                "health": health_data,
                "command": command_data,
                "counters": counter_data,
                "paths": path_data,
                "checkpoint": checkpoint,
            }
            self._print_human(payload)
            self._write_jsonl(payload)
            self._write_tensorboard(payload)
        except Exception:
            # Observability must not break training.
            return

    def checkpoint(self, *, step: int, path: str, saved: bool = True) -> None:
        try:
            line = f"[TPCKPT] step={int(step)} path={path} saved={1 if saved else 0}"
            print(line, flush=True)
            self._append_log_line(line)
        except Exception:
            return

    def _print_human(self, payload: dict[str, Any]) -> None:
        reward = payload["reward"]
        curr = payload["curriculum"]
        health = payload["health"]
        command = payload.get("command", {})
        counters = payload.get("counters", {})
        paths = payload.get("paths", {})
        step = payload["step"]
        total = payload.get("total_steps") or 0
        pct = (100.0 * step / total) if total else None
        if not self._paths_printed:
            self._paths_printed = True
            path_line = (
                "[TPPATH] "
                f"run_root={paths.get('run_root', '')} "
                f"run_id={paths.get('run_id', '')} "
                f"run_dir={paths.get('run_dir', '')} "
                f"log={paths.get('log', '')} "
                f"jsonl={paths.get('telemetry_jsonl', '')} "
                f"console_log={paths.get('console_log', '')} "
                f"checkpoint_dir={paths.get('checkpoint_dir', '')}"
            )
            print(path_line, flush=True)
            self._append_log_line(path_line)
        if not self._reward_cfg_printed:
            cfg_items = [
                (key[4:], value)
                for key, value in reward.items()
                if isinstance(key, str) and key.startswith("cfg_")
            ]
            if cfg_items:
                self._reward_cfg_printed = True
                self._emit_line("[TPRCFG] " + _kv(sorted(cfg_items)))
        self._emit_line("[TPSTAT] " + _kv([
            ("step", step),
            ("total", total if total else None),
            ("pct", pct),
            ("fps", _f(payload.get("fps")) if payload.get("fps") is not None else None),
            ("elapsed", payload.get("elapsed") or None),
            ("eta", payload.get("eta") or None),
            ("run_id", paths.get("run_id") or None),
        ], digits=2))
        self._emit_line("[TPCMD] " + _kv([
            ("cmd_vx", command.get("cmd_vx")),
            ("cmd_vy", command.get("cmd_vy")),
            ("cmd_wz", command.get("cmd_wz")),
            ("act_vx", command.get("actual_vx")),
            ("act_vy", command.get("actual_vy")),
            ("act_wz", command.get("actual_wz")),
            ("v_along", command.get("v_along")),
            ("speed_xy", command.get("speed_xy", reward.get("speed"))),
            ("lin_err", command.get("lin_err", reward.get("lin_err"))),
            ("yaw_err", command.get("yaw_err")),
            ("progress", command.get("progress_ratio")),
            ("diag_win", command.get("diagonal_contact")),
            ("diag_inst", command.get("diagonal_pair_instant")),
            ("duty_win", command.get("duty_balance")),
            ("duty_spread", command.get("duty_spread_window")),
            ("slip_win", command.get("stance_slip")),
            ("slip_high", command.get("stance_slip_high_fraction")),
        ]))
        self._emit_line("[TPREW] " + _kv([
            ("step", step),
            ("total", reward.get("total", reward.get("rew"))),
            ("tracking_lin", reward.get("tracking_lin", reward.get("track"))),
            ("tracking_lin_far", reward.get("tracking_lin_far")),
            ("tracking_yaw", reward.get("tracking_yaw")),
            ("tracking_yaw_far", reward.get("tracking_yaw_far")),
            ("stand", reward.get("stand")),
            ("stand_far", reward.get("stand_far")),
            ("terrain_progress", reward.get("terrain_progress")),
            ("gait_anchor", reward.get("gait_anchor")),
            ("imitate", reward.get("imitate")),   # LIVE joint-imitation attractor (the gait-cluster fix) — surface it to watch the policy match the clean reference
            ("touchdown_slip", reward.get("touchdown_slip")),
            ("swing_dir", reward.get("swing_dir")),   # 脚在空中往滑 fix: positive attractor, watch it climb toward w_swing_dir*moving (=~0.5) as the mid-swing slide vanishes
            ("feet_air_time", reward.get("feet_air_time")),
        ]))
        self._emit_line("[TPPEN] " + _kv([
            ("clearance_under", reward.get("clearance_under")),
            ("clearance_over", reward.get("clearance_over")),
            ("stance_slip", reward.get("stance_slip", reward.get("slip"))),
            ("stance_slip_event", reward.get("stance_slip_event")),
            ("landing_impact", reward.get("landing_impact")),
            ("diagonal_contact", reward.get("diagonal_contact")),
            ("duty_balance", reward.get("duty_balance")),
            ("torque_margin", reward.get("torque_margin")),
            ("torque_saturation", reward.get("torque_saturation", reward.get("torque_sat"))),
            ("off_axis", reward.get("off_axis", reward.get("offaxis"))),
            ("wrong_dir", reward.get("wrong_dir")),
            ("action_rate", reward.get("action_rate")),
        ]))
        self._emit_line("[TPGATE] " + _kv([
            ("stable", health.get("stable_motion_gate", reward.get("gait", reward.get("gate")))),
            ("moving", health.get("moving_gate")),
            ("stand_gate", health.get("stand_gate")),
            ("base_h", health.get("base_h", reward.get("base_h"))),
            ("base_h_min", health.get("base_h_min")),
            ("height_low_risk", health.get("height_low_risk_window")),
            ("upright", health.get("upright", reward.get("upright"))),
            ("tilt_deg", health.get("tilt_deg")),
            ("tilt_deg_max", health.get("tilt_deg_max")),
            ("tilt_high_risk", health.get("tilt_high_risk_window")),
            ("support_instability", health.get("support_instability")),
            ("contacts_mean", health.get("contacts_mean")),
            ("fall_rate", health.get("fall_rate")),
            ("reset_rate", health.get("reset_rate")),
            ("terminal_rate", health.get("terminal_rate")),
            ("torque_util", health.get("torque_util")),
        ]))
        self._emit_line("[TPCURR] " + _kv([
            ("step", step),
            ("phase", curr.get("phase")),
            ("command_mode", curr.get("command_mode")),
            ("active_dirs", curr.get("active_dirs")),
            ("phase_count", curr.get("phase_count")),
            ("penalty_gate", curr.get("penalty_gate")),
            ("clearance_gate", curr.get("clearance_gate")),
            ("phase_gate", curr.get("phase_gate")),
            ("progress_gate", curr.get("progress_gate")),
            # per-direction progress — surface ALL of them, not just yaw: the blocker (min direction) is often
            # a LINEAR direction, and over-tuning yaw can starve linear locomotion. Seeing all four prevents
            # the yaw-tunnel-vision of chasing yaw while lat/fwd/back are the real laggards.
            ("prog_fwd", curr.get("progress_fwd")),
            ("prog_back", curr.get("progress_back")),
            ("prog_lat", curr.get("progress_lat")),
            ("prog_yaw", curr.get("progress_yaw")),
            ("yaw_ceil", curr.get("yaw_ceil")),            # reachable-ramp diagnostic: is the yaw range growing?
            ("lagging_dir", curr.get("progress_lagging_dir")),
            ("gait_gate", curr.get("gait_gate")),
            ("diag_gate", curr.get("diagonal_gate")),
            ("duty_gate", curr.get("duty_balance_gate")),
            ("slip_gate", curr.get("slip_gate")),
            ("duty_spread_gate", curr.get("duty_spread_gate")),
            ("fall_gate", curr.get("fall_gate")),
            ("terrain_mean", curr.get("terrain_mean")),
            ("terrain_max", curr.get("terrain_max")),
            ("dr_level", curr.get("dr_level")),
            ("next", curr.get("next_gate")),
            ("blocked_by", curr.get("blocked_by")),
        ]))

    def _write_jsonl(self, payload: dict[str, Any]) -> None:
        path = Path(self.jsonl_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _write_tensorboard(self, payload: dict[str, Any]) -> None:
        if self.writer is None:
            return
        step = payload["step"]
        for key, value in payload["reward"].items():
            self.writer.add_scalar(f"Telemetry/reward/{key}", value, step)
        for key, value in payload.get("command", {}).items():
            self.writer.add_scalar(f"Telemetry/command/{key}", value, step)
        for key, value in payload["health"].items():
            self.writer.add_scalar(f"Telemetry/health/{key}", value, step)
        for key, value in payload["curriculum"].items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"Telemetry/curriculum/{key}", float(value), step)
        for key, value in payload.get("counters", {}).items():
            self.writer.add_scalar(f"Telemetry/counters/{key}", value, step)

    def _emit_line(self, line: str) -> None:
        print(line, flush=True)
        self._append_log_line(line)

    def _append_log_line(self, line: str) -> None:
        path = Path(self.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line.rstrip() + "\n")


def _fmt_duration(seconds: float) -> str:
    total = int(max(0.0, seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"
