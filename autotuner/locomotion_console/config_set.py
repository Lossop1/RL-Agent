"""Minimal ConfigSet assembly for the locomotion console.

V0 keeps profiles read-only and explicit. The framework can stay draft while the
console still has a stable center object for jobs, artifacts, diagnostics, and
future deployment actions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .config import LocomotionConsoleSettings
from .config_manager import effective_remote_config, llm_profile
from .framework_profile import FrameworkProfile, get_framework_profile
from .robot_profile import get_robot_profile


ProfileStatus = Literal["draft", "validated", "reference", "missing", "configured"]


@dataclass(frozen=True)
class ProfileSummary:
    id: str
    label: str
    status: ProfileStatus
    detail: str = ""


@dataclass(frozen=True)
class ConfigSet:
    id: str
    label: str
    status: Literal["draft", "validated"]
    task_goal: str
    remote: ProfileSummary
    robot: ProfileSummary
    framework: ProfileSummary
    llm: ProfileSummary
    notes: tuple[str, ...] = field(default_factory=tuple)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _remote_summary(settings: LocomotionConsoleSettings) -> ProfileSummary:
    cfg = effective_remote_config(settings)
    host = cfg.get("ssh_host", "")
    port = cfg.get("ssh_port", "")
    label = f"{host}:{port}" if host else "No remote configured"
    return ProfileSummary(
        id="remote.default",
        label=label,
        status="configured" if host else "missing",
        detail=str(cfg.get("work_dir") or "/root/robot_lab"),
    )


def _llm_summary(settings: LocomotionConsoleSettings) -> ProfileSummary:
    cfg = llm_profile(settings)
    return ProfileSummary(
        id="llm.default",
        label=cfg.model or "No LLM configured",
        status="configured" if cfg.configured else "missing",
        detail=cfg.base_url,
    )


def _robot_summary() -> ProfileSummary:
    profile = get_robot_profile()
    return ProfileSummary(
        id=profile.id,
        label=profile.label,
        status=profile.status,
        detail=f"{profile.dof} DoF; diagnostic spec {profile.diagnostic_spec}; {profile.note}",
    )


def _framework_summary(profile: FrameworkProfile) -> ProfileSummary:
    return ProfileSummary(
        id=profile.id,
        label=profile.label,
        status=profile.status,
        detail=profile.task_id,
    )


def get_active_config_set(settings: LocomotionConsoleSettings) -> ConfigSet:
    framework = get_framework_profile(settings.framework_id)
    return ConfigSet(
        id=f"taili-default:{framework.id}",
        label="Taili 默认框架适配工作区",
        status="draft",
        task_goal=(
            "构建 Taili 运动框架适配闭环：用框架组件承载能力，"
            "由 Adapter 派生机器人专属配置，再用检查点诊断和可视化验证交付。"
        ),
        remote=_remote_summary(settings),
        robot=_robot_summary(),
        framework=_framework_summary(framework),
        llm=_llm_summary(settings),
        notes=(
            "LLM 只负责建议、发现和叙述；确定性代码负责执行、守门和验证。",
            "当前框架仍是 draft 档案；诊断以 checkpoint 为中心，也可能包含 reference checkpoint。",
        ),
    )
