"""Locomotion console runtime settings.

Kept deliberately tiny — environment-overridable so the fake/real switch and the box it
talks to never require code edits.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .framework_profile import DEFAULT_FRAMEWORK_ID, get_framework_profile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_CONFIG_DIR = PROJECT_ROOT / "config"
PROJECT_SSH_CONFIG = PROJECT_CONFIG_DIR / "ssh.json"


def project_config_path(name: str) -> Path:
    return PROJECT_CONFIG_DIR / name


def _default_state_root() -> str:
    return os.environ.get(
        "LOCOMOTION_CONSOLE_STATE_ROOT",
        os.path.join(tempfile.gettempdir(), "locomotion_console_state"),
    )


def _saved_framework_id(local_state_root: str) -> str:
    path = Path(local_state_root) / "console_config.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        value = str(raw.get("framework_id") or "").strip()
        if value:
            get_framework_profile(value)
            return value
    except Exception:
        pass
    return ""


def _read_json(path: Path) -> dict:
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _effective_remote_config(local_state_root: str) -> dict:
    cfg = _read_json(PROJECT_SSH_CONFIG)
    override = _read_json(Path(local_state_root) / "remote_profile.json")
    for key, value in override.items():
        if value not in (None, ""):
            cfg[key] = value
    return cfg


def _default_source(local_state_root: str | None = None) -> str:
    explicit = os.environ.get("LOCOMOTION_CONSOLE_SOURCE")
    if explicit:
        value = explicit.strip().lower()
        return value if value in {"fake", "real"} else "fake"
    cfg = _effective_remote_config(local_state_root or _default_state_root())
    return "real" if str(cfg.get("ssh_host") or "").strip() else "fake"


def _remote_or_env(cfg: dict, env_name: str, key: str, default: str) -> str:
    return os.environ.get(env_name, str(cfg.get(key) or default))


@dataclass(frozen=True)
class LocomotionConsoleSettings:
    # "fake" = synthetic data source (no GPU, no SSH). "real" = RemoteSSH against the box.
    source: str = field(default_factory=_default_source)
    # WS metric stream cadence (seconds).
    poll_interval_s: float = float(os.environ.get("LOCOMOTION_CONSOLE_POLL_S", "1.0"))
    # Optional run-dir substring to view a specific run; empty = newest run.
    run_filter: str = os.environ.get("LOCOMOTION_CONSOLE_RUN", "")
    # Active framework profile. The default is the current non-teacher draft framework.
    framework_id: str = os.environ.get("LOCOMOTION_CONSOLE_FRAMEWORK", DEFAULT_FRAMEWORK_ID)
    # Remote log path the real source tails (the training pipe). Overridable per box.
    remote_log_path: str = os.environ.get("LOCOMOTION_CONSOLE_REMOTE_LOG", "")
    # Log format key under config/log_formats/ used by parse_log.
    log_format: str = os.environ.get("LOCOMOTION_CONSOLE_LOG_FORMAT", "skrl")
    # CORS origin for the Vite dev server.
    ui_origin: str = os.environ.get("LOCOMOTION_CONSOLE_UI_ORIGIN", "http://localhost:5173")
    # Remote observation-diagnostic package and output roots.
    diagnostic_tool_root: str = os.environ.get(
        "LOCOMOTION_CONSOLE_DIAG_TOOL_ROOT",
        "/root/gpufree-data/tools/isaaclab_quad_diag_v051",
    )
    diagnostic_output_root: str = os.environ.get(
        "LOCOMOTION_CONSOLE_DIAG_OUTPUT_ROOT",
        "/root/gpufree-data/diag_runs",
    )
    diagnostic_robot_root: str = os.environ.get(
        "LOCOMOTION_CONSOLE_DIAG_ROBOT_ROOT",
        "/root/gpufree-data/robot_lab",
    )
    diagnostic_python: str = os.environ.get(
        "LOCOMOTION_CONSOLE_DIAG_PYTHON",
        "/opt/conda/envs/isaaclab/bin/python",
    )
    local_state_root: str = _default_state_root()
    diagnostic_task: str = os.environ.get(
        "LOCOMOTION_CONSOLE_DIAG_TASK",
        get_framework_profile(os.environ.get("LOCOMOTION_CONSOLE_FRAMEWORK", "taili_amp_blind")).diagnostic_task,
    )


def get_settings() -> LocomotionConsoleSettings:
    local_state_root = _default_state_root()
    remote_cfg = _effective_remote_config(local_state_root)
    framework_id = (
        os.environ.get("LOCOMOTION_CONSOLE_FRAMEWORK")
        or _saved_framework_id(local_state_root)
        or DEFAULT_FRAMEWORK_ID
    )
    diagnostic_task = os.environ.get(
        "LOCOMOTION_CONSOLE_DIAG_TASK",
        get_framework_profile(framework_id).diagnostic_task,
    )
    return LocomotionConsoleSettings(
        source=_default_source(local_state_root),
        poll_interval_s=float(os.environ.get("LOCOMOTION_CONSOLE_POLL_S", "1.0")),
        run_filter=os.environ.get("LOCOMOTION_CONSOLE_RUN", ""),
        framework_id=framework_id,
        remote_log_path=_remote_or_env(remote_cfg, "LOCOMOTION_CONSOLE_REMOTE_LOG", "remote_log_path", ""),
        log_format=_remote_or_env(remote_cfg, "LOCOMOTION_CONSOLE_LOG_FORMAT", "log_format", "skrl"),
        ui_origin=os.environ.get("LOCOMOTION_CONSOLE_UI_ORIGIN", "http://localhost:5173"),
        diagnostic_tool_root=_remote_or_env(
            remote_cfg,
            "LOCOMOTION_CONSOLE_DIAG_TOOL_ROOT",
            "diagnostic_tool_root",
            "/root/gpufree-data/tools/isaaclab_quad_diag_v051",
        ),
        diagnostic_output_root=_remote_or_env(
            remote_cfg,
            "LOCOMOTION_CONSOLE_DIAG_OUTPUT_ROOT",
            "diagnostic_output_root",
            "/root/gpufree-data/diag_runs",
        ),
        diagnostic_robot_root=_remote_or_env(
            remote_cfg,
            "LOCOMOTION_CONSOLE_DIAG_ROBOT_ROOT",
            "diagnostic_robot_root",
            "/root/gpufree-data/robot_lab",
        ),
        diagnostic_python=_remote_or_env(
            remote_cfg,
            "LOCOMOTION_CONSOLE_DIAG_PYTHON",
            "diagnostic_python",
            "/opt/conda/envs/isaaclab/bin/python",
        ),
        local_state_root=local_state_root,
        diagnostic_task=diagnostic_task,
    )
