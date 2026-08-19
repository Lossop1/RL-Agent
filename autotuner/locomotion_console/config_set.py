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
from autotuner.product import ContractResolutionError, ProductManifest, get_product, resolve_product_contract


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
    contract: ProfileSummary | None = None


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


def _robot_summary(product: ProductManifest) -> ProfileSummary:
    profile = product.robot_profile()
    return ProfileSummary(
        id=profile.id,
        label=profile.label,
        status=profile.status,
        detail=f"{profile.dof} DoF; diagnostic spec {profile.diagnostic_spec}; {profile.note}",
    )


def _contract_summary(product: ProductManifest) -> ProfileSummary:
    try:
        contract = resolve_product_contract(product)
    except (ContractResolutionError, OSError, ValueError) as exc:
        return ProfileSummary(
            id=f"contract:{product.product_id}",
            label="Resolved product contract",
            status="missing",
            detail=str(exc),
        )
    return ProfileSummary(
        id=f"contract:{product.product_id}:{contract.contract_digest[:12]}",
        label="Resolved product contract",
        status="validated" if not contract.issues else "draft",
        detail=(
            f"config={contract.config_digest[:12] or 'missing'}; "
            f"assets={contract.asset_digest[:12]}; "
            f"contract={contract.contract_digest[:12]}"
        ),
    )


def _framework_summary(profile: FrameworkProfile) -> ProfileSummary:
    return ProfileSummary(
        id=profile.id,
        label=profile.label,
        status=profile.status,
        detail=profile.task_id,
    )


def get_active_config_set(settings: LocomotionConsoleSettings) -> ConfigSet:
    product = get_product(settings.product_id)
    framework = get_framework_profile(
        settings.framework_id,
        product_id=settings.product_id or None,
    )
    contract = _contract_summary(product)
    return ConfigSet(
        id=f"{product.product_id}:{framework.id}",
        label=f"{product.label} product workspace",
        status="validated" if contract.status == "validated" else "draft",
        task_goal=(
            f"为 {product.label} 解析产品资产和任务要求，生成训练、监控、诊断与部署合同，"
            "再由确定性运行链执行并记录可追溯证据。"
        ),
        remote=_remote_summary(settings),
        robot=_robot_summary(product),
        framework=_framework_summary(framework),
        llm=_llm_summary(settings),
        contract=contract,
        notes=(
            "LLM 只负责建议、发现和叙述；确定性代码负责执行、守门和验证。",
            "产品特定配置和诊断必须通过 resolved contract 绑定，不能由控制台猜测。",
        ),
    )
