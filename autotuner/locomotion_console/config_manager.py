"""Safe profile configuration management for the locomotion console.

This module handles operator-editable console preferences. It deliberately keeps
runtime control separate from saved intent: framework selection is persisted and
reported as pending when it differs from the already-started backend settings.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import LocomotionConsoleSettings, project_config_path
from .framework_profile import get_framework_profile


PROJECT_LLM_CONFIG = project_config_path("llm.json")
PROJECT_SSH_CONFIG = project_config_path("ssh.json")
ENV_VAR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class LLMProfileState:
    configured: bool
    model: str
    base_url: str
    api_key_env_var: str
    api_key_resolved: bool
    unsafe_literal_key: bool
    advisory_only: bool = True
    actions_require_confirmation: bool = True


@dataclass(frozen=True)
class LLMProfileUpdateResult:
    profile: LLMProfileState
    saved_path: str
    message: str


@dataclass(frozen=True)
class FrameworkSelectionState:
    active_framework_id: str
    desired_framework_id: str
    requires_restart: bool
    saved_path: str
    message: str


@dataclass(frozen=True)
class RemoteProfileState:
    host: str
    port: int
    user: str
    work_dir: str
    conda_env: str
    diagnostic_tool_root: str
    diagnostic_output_root: str
    diagnostic_robot_root: str
    diagnostic_python: str
    log_format: str
    password_configured: bool
    source: str


@dataclass(frozen=True)
class RemoteProfileUpdateResult:
    profile: RemoteProfileState
    saved_path: str
    requires_restart: bool
    message: str


@dataclass(frozen=True)
class RemoteConnectionTestResult:
    ok: bool
    host: str
    port: int
    user: str
    work_dir: str
    latency_ms: int
    detail: str
    pwd: str = ""


@dataclass(frozen=True)
class LLMAuditItem:
    file: str
    schema_name: str
    model: str
    elapsed_s: float
    attempt: int
    ok: bool
    error: str
    created_at: str


def state_root(settings: LocomotionConsoleSettings) -> Path:
    return Path(settings.local_state_root)


def console_config_path(settings: LocomotionConsoleSettings) -> Path:
    return state_root(settings) / "console_config.json"


def llm_config_path(settings: LocomotionConsoleSettings) -> Path:
    return state_root(settings) / "llm_profile.json"


def remote_config_path(settings: LocomotionConsoleSettings) -> Path:
    return state_root(settings) / "remote_profile.json"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    # Atomic (temp-in-same-dir + os.replace) so a crash mid-write never leaves a truncated,
    # unparseable profile — this file is the remote-target / credential source of truth.
    # 0600 because it can hold ssh_pass in cleartext (state dir, single-operator).
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        os.chmod(tmp, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _project_remote_fields(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "ssh_host": str(cfg.get("ssh_host") or "").strip(),
        "ssh_port": str(cfg.get("ssh_port") or "22").strip(),
        "ssh_user": str(cfg.get("ssh_user") or "root").strip(),
        "work_dir": str(cfg.get("work_dir") or "/root/robot_lab").strip(),
        "conda_env": str(cfg.get("conda_env") or "isaaclab").strip(),
        "diagnostic_tool_root": str(cfg.get("diagnostic_tool_root") or "").strip(),
        "diagnostic_output_root": str(cfg.get("diagnostic_output_root") or "").strip(),
        "diagnostic_robot_root": str(cfg.get("diagnostic_robot_root") or "").strip(),
        "diagnostic_python": str(cfg.get("diagnostic_python") or "").strip(),
        "log_format": str(cfg.get("log_format") or "skrl").strip(),
    }


def effective_remote_config(settings: LocomotionConsoleSettings) -> dict[str, Any]:
    base = _read_json(PROJECT_SSH_CONFIG)
    override = _read_json(remote_config_path(settings))
    result = dict(base)
    for key, value in override.items():
        if value not in (None, ""):
            result[key] = value
    return result


def remote_profile(settings: LocomotionConsoleSettings) -> RemoteProfileState:
    cfg = effective_remote_config(settings)
    fields = _project_remote_fields(cfg)
    return RemoteProfileState(
        host=fields["ssh_host"],
        port=int(fields["ssh_port"] or 22),
        user=fields["ssh_user"],
        work_dir=fields["work_dir"],
        conda_env=fields["conda_env"],
        diagnostic_tool_root=fields["diagnostic_tool_root"] or settings.diagnostic_tool_root,
        diagnostic_output_root=fields["diagnostic_output_root"] or settings.diagnostic_output_root,
        diagnostic_robot_root=fields["diagnostic_robot_root"] or settings.diagnostic_robot_root,
        diagnostic_python=fields["diagnostic_python"] or settings.diagnostic_python,
        log_format=fields["log_format"] or settings.log_format,
        password_configured=bool(str(cfg.get("ssh_pass") or "")),
        source=settings.source,
    )


def update_remote_profile(settings: LocomotionConsoleSettings, updates: dict[str, Any]) -> RemoteProfileUpdateResult:
    current = effective_remote_config(settings)
    next_cfg = _project_remote_fields(current)
    allowed = set(next_cfg) | {"ssh_pass"}
    for key, value in updates.items():
        if key not in allowed or value is None:
            continue
        if key == "ssh_pass":
            if str(value):
                next_cfg[key] = str(value)
            continue
        next_cfg[key] = str(value).strip()
    if not next_cfg["ssh_host"]:
        raise ValueError("ssh_host is required")
    try:
        port = int(next_cfg["ssh_port"])
    except ValueError as exc:
        raise ValueError("ssh_port must be an integer") from exc
    if port < 1 or port > 65535:
        raise ValueError("ssh_port must be in 1..65535")
    if not next_cfg["ssh_user"]:
        raise ValueError("ssh_user is required")
    if not next_cfg["work_dir"].startswith("/"):
        raise ValueError("work_dir must be an absolute remote path")
    if "ssh_pass" not in next_cfg and current.get("ssh_pass"):
        next_cfg["ssh_pass"] = current["ssh_pass"]
    path = remote_config_path(settings)
    _write_json(path, next_cfg)
    return RemoteProfileUpdateResult(
        profile=remote_profile(settings),
        saved_path=str(path),
        requires_restart=True,
        message="Saved remote profile. Restart the backend for all runtime paths to use it.",
    )


def test_remote_connection(settings: LocomotionConsoleSettings) -> RemoteConnectionTestResult:
    import time as _time

    from autotuner.training.remote import RemoteSSH

    cfg = effective_remote_config(settings)
    profile = remote_profile(settings)
    t0 = _time.time()
    try:
        remote = RemoteSSH(cfg)
        out = remote.exec_out("pwd && test -d . && echo ok", timeout=12)
        try:
            remote.close()
        except Exception:
            pass
        latency_ms = int((_time.time() - t0) * 1000)
        lines = [line.strip() for line in (out or "").splitlines() if line.strip()]
        return RemoteConnectionTestResult(
            ok=bool(lines),
            host=profile.host,
            port=profile.port,
            user=profile.user,
            work_dir=profile.work_dir,
            latency_ms=latency_ms,
            detail="SSH command completed" if lines else "SSH connected but returned no output",
            pwd=lines[0] if lines else "",
        )
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((_time.time() - t0) * 1000)
        return RemoteConnectionTestResult(
            ok=False,
            host=profile.host,
            port=profile.port,
            user=profile.user,
            work_dir=profile.work_dir,
            latency_ms=latency_ms,
            detail=f"{type(exc).__name__}: {exc}",
        )


def read_console_config(settings: LocomotionConsoleSettings) -> dict[str, Any]:
    return _read_json(console_config_path(settings))


def desired_framework_id(settings: LocomotionConsoleSettings) -> str:
    saved = read_console_config(settings).get("framework_id")
    if isinstance(saved, str) and saved.strip():
        try:
            get_framework_profile(saved.strip())
            return saved.strip()
        except ValueError:
            pass
    return settings.framework_id


def select_framework(settings: LocomotionConsoleSettings, framework_id: str) -> FrameworkSelectionState:
    framework_id = framework_id.strip()
    get_framework_profile(framework_id)
    path = console_config_path(settings)
    cfg = read_console_config(settings)
    cfg["framework_id"] = framework_id
    cfg["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json(path, cfg)
    requires_restart = framework_id != settings.framework_id
    return FrameworkSelectionState(
        active_framework_id=settings.framework_id,
        desired_framework_id=framework_id,
        requires_restart=requires_restart,
        saved_path=str(path),
        message=(
            "Saved. Restart the backend for this framework to become active."
            if requires_restart else
            "Saved. The selected framework is already active."
        ),
    )


def _merged_llm_config(settings: LocomotionConsoleSettings) -> dict[str, Any]:
    cfg = _read_json(PROJECT_LLM_CONFIG)
    override = _read_json(llm_config_path(settings))
    cfg.update({k: v for k, v in override.items() if k in {"model", "base_url", "api_key_env_var"}})
    return cfg


def llm_profile(settings: LocomotionConsoleSettings) -> LLMProfileState:
    cfg = _merged_llm_config(settings)
    model = str(cfg.get("model") or "")
    base_url = str(cfg.get("base_url") or "")
    raw_key_ref = str(cfg.get("api_key_env_var") or "")
    unsafe_literal = raw_key_ref.startswith(("sk-", "key-"))
    env_var = "" if unsafe_literal else raw_key_ref
    resolved = bool(raw_key_ref) if unsafe_literal else bool(os.environ.get(env_var, ""))
    return LLMProfileState(
        configured=bool(model and base_url and (env_var or unsafe_literal)),
        model=model,
        base_url=base_url,
        api_key_env_var=env_var,
        api_key_resolved=resolved,
        unsafe_literal_key=unsafe_literal,
    )


def update_llm_profile(
    settings: LocomotionConsoleSettings,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env_var: str | None = None,
) -> LLMProfileUpdateResult:
    current = _merged_llm_config(settings)
    next_cfg = {
        "model": str(current.get("model") or ""),
        "base_url": str(current.get("base_url") or ""),
        "api_key_env_var": "" if str(current.get("api_key_env_var") or "").startswith(("sk-", "key-")) else str(current.get("api_key_env_var") or ""),
    }
    if model is not None:
        next_cfg["model"] = model.strip()
    if base_url is not None:
        next_cfg["base_url"] = base_url.strip()
    if api_key_env_var is not None:
        value = api_key_env_var.strip()
        if value.startswith(("sk-", "key-")):
            raise ValueError("API keys must be stored in an environment variable; save the variable name here.")
        if value and not ENV_VAR_RE.fullmatch(value):
            raise ValueError("api_key_env_var must be a valid environment variable name.")
        next_cfg["api_key_env_var"] = value
    if not next_cfg["model"]:
        raise ValueError("model is required")
    if not next_cfg["base_url"]:
        raise ValueError("base_url is required")
    if not next_cfg["api_key_env_var"]:
        raise ValueError("api_key_env_var is required and must be an environment variable name")

    path = llm_config_path(settings)
    _write_json(path, next_cfg)
    return LLMProfileUpdateResult(
        profile=llm_profile(settings),
        saved_path=str(path),
        message="Saved LLM profile. The next LLM call will use this profile.",
    )


def llm_audit_dirs(settings: LocomotionConsoleSettings) -> list[Path]:
    override = os.environ.get("LOCOMOTION_CONSOLE_LLM_AUDIT_DIR", "")
    if override:
        return [Path(override)]
    primary = state_root(settings) / "llm_gateway_audit"
    legacy = Path("out_v07_taili_dog/llm_gateway_audit")
    return [primary, legacy] if primary != legacy else [primary]


def recent_llm_audit(settings: LocomotionConsoleSettings, limit: int = 20) -> list[LLMAuditItem]:
    paths: list[Path] = []
    for root in llm_audit_dirs(settings):
        if root.exists():
            paths.extend(root.glob("llm_call_*.json"))
    if not paths:
        return []
    items: list[LLMAuditItem] = []
    files = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files[: max(0, min(limit, 100))]:
        data = _read_json(path)
        stat = path.stat()
        items.append(LLMAuditItem(
            file=path.name,
            schema_name=str(data.get("schema_name") or ""),
            model=str(data.get("model") or ""),
            elapsed_s=float(data.get("elapsed_s") or 0.0),
            attempt=int(data.get("attempt") or 0),
            ok=not bool(data.get("error")) and data.get("parsed") is not None,
            error=str(data.get("error") or ""),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
        ))
    return items
