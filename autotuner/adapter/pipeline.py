"""Adapter pipeline — close the offline chain into one call (SYSTEM_ARCHITECTURE §3 A, §14.1-2).

`ConfigSet → AdaptationBundle` in one step:
    adapt()            URDF → AdaptedConfig (derived actuator/dims/geom + scaled reward thresholds)
    materialize()      AdaptedConfig → typed edits on COPIES in work_dir (env_cfg + asset)
    roundtrip_verify() re-read the materialized env_cfg, confirm the fields actually wrote
    build_plan()       work_dir + remote_map → DEPLOY PLAN (dry-run; backup-then-put, checksums)

This is the backend of the "A 一键产出" path: hand it a robot + its remote destinations, get back
the full provenance bundle (what was derived, what changed vs proven, and the exact deploy plan) —
all CPU, zero GPU, nothing written remote (execute() stays gated in deploy.py). The framework
itself (reward structure / curriculum / A-C / gait mode) is untouched: only robot-specific numbers
and references are derived (§6).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

from autotuner.adapter.adapt import adapt
from autotuner.adapter.materialize import materialize, roundtrip_verify, Edit
from autotuner.adapter.deploy import build_plan, render_plan, DeployPlan
from autotuner.framework_library import (
    get_composition, validate_composition, adapt_plan as _component_adapt_plan,
)
from autotuner.product import resolve_product_adapter, resolve_product_contract


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ConfigSet:
    """一次产品适配的确定性输入，不携带产品实现对象。"""
    urdf: str
    env_cfg_src: str
    asset_src: str
    product_id: str = ""
    product_contract_digest: str = ""
    remote_env_cfg: str = ""
    remote_asset: str = ""
    mass: Optional[float] = None
    regenerate_clips: bool = False
    clip_remote_dir: Optional[str] = None
    launch_cmd: Optional[str] = None
    framework_composition: str = ""
    robot_capabilities: tuple[str, ...] = ()
    legacy_deploy_enabled: bool = False


@dataclass
class AdaptationBundle:
    adapted: dict
    edits: List[Edit]
    roundtrip_ok: bool
    roundtrip_bad: List[str]
    plan: DeployPlan
    work_dir: str
    derived_summary: Dict[str, object] = field(default_factory=dict)
    composition_id: str = ""
    composition_valid: bool = True
    composition_issues: List[str] = field(default_factory=list)
    component_adapt_plan: Dict[str, List[str]] = field(default_factory=dict)
    references_required: bool = False
    legacy_deploy_enabled: bool = False


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _workspace_path(value: Any, name: str) -> str:
    relative = Path(str(value or ""))
    if not str(relative) or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} must be a workspace-relative path")
    resolved = (_PROJECT_ROOT / relative).resolve()
    try:
        resolved.relative_to(_PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"{name} escapes the workspace") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} does not exist: {relative.as_posix()}")
    return str(resolved)


def _asset_path(contract: Any, asset_id: str) -> str:
    for asset in contract.assets:
        if asset.id == asset_id:
            if not asset.exists:
                raise FileNotFoundError(f"product asset is missing: {asset_id}")
            return _workspace_path(asset.resolved_path, f"asset {asset_id}")
    raise ValueError(f"product adaptation references unknown asset: {asset_id!r}")


def build_config_set(product_id: str | None = None, composition: str | None = None) -> ConfigSet:
    """从产品合同构造适配输入；系统中不再维护机器人 preset 字典。"""
    contract = resolve_product_contract(product_id)
    adapter = resolve_product_adapter(contract)
    settings = _mapping(adapter.adaptation, "adaptation")
    materialization = _mapping(settings.get("materialization"), "adaptation.materialization")
    legacy = _mapping(settings.get("legacy_deploy", {}), "adaptation.legacy_deploy")
    capabilities = contract.robot.get("capabilities")
    if not isinstance(capabilities, (list, tuple)):
        raise ValueError("robot.capabilities must be a list")
    selected_composition = str(composition or settings.get("composition") or "").strip()
    if not selected_composition:
        raise ValueError("adaptation.composition is required")
    return ConfigSet(
        product_id=contract.product_id,
        product_contract_digest=contract.contract_digest,
        urdf=_asset_path(contract, str(settings.get("urdf_asset") or "")),
        env_cfg_src=_workspace_path(materialization.get("env_source"), "materialization.env_source"),
        asset_src=_workspace_path(materialization.get("asset_source"), "materialization.asset_source"),
        remote_env_cfg=str(legacy.get("remote_env_cfg") or ""),
        remote_asset=str(legacy.get("remote_asset") or ""),
        clip_remote_dir=str(legacy.get("clip_remote_dir") or "") or None,
        launch_cmd=str(legacy.get("launch_cmd") or "") or None,
        framework_composition=selected_composition,
        robot_capabilities=tuple(str(item) for item in capabilities),
        legacy_deploy_enabled=bool(legacy.get("enabled", False)),
    )


def plan_adaptation(cs: ConfigSet, work_dir: str, stamp: str,
                    require_refs: bool | None = None) -> AdaptationBundle:
    """Run the full offline adaptation→deploy-plan chain. No remote writes.

    Resolves the ② framework composition first (§5 ConfigSet{framework} → Adapter): validates the
    robot supports every component's morphology, and records the per-component adapt_plan
    (invariant/derive/regenerate/scale) so the bundle documents what the Adapter touched and why.
    """
    contract = resolve_product_contract(cs.product_id or None)
    if cs.product_contract_digest and cs.product_contract_digest != contract.contract_digest:
        raise ValueError(
            "product contract changed after ConfigSet creation; rebuild the ConfigSet before adaptation"
        )
    adapter = resolve_product_adapter(contract)
    settings = _mapping(adapter.adaptation, "adaptation")
    materialization = _mapping(settings.get("materialization"), "adaptation.materialization")
    legacy = _mapping(settings.get("legacy_deploy", {}), "adaptation.legacy_deploy")
    references_required = (
        bool(legacy.get("require_reference_clips", False))
        if require_refs is None
        else bool(require_refs)
    )
    composition_id = cs.framework_composition or str(settings.get("composition") or "")
    comp = get_composition(composition_id, contract.product_id)
    comp_issues = validate_composition(comp, tuple(cs.robot_capabilities))
    comp_plan = _component_adapt_plan(comp)

    wd = Path(work_dir)
    adapted = adapt(
        cs.urdf,
        mass=cs.mass,
        regenerate_clips_to=str(wd) if cs.regenerate_clips else None,
        contract=contract,
    )
    edits = materialize(
        adapted,
        cs.env_cfg_src,
        cs.asset_src,
        work_dir,
        spec=materialization,
    )
    bad = roundtrip_verify(
        adapted,
        str(wd / Path(cs.env_cfg_src).name),
        spec=materialization,
    )

    remote_map: dict[str, str] = {}
    if cs.remote_env_cfg:
        remote_map[Path(cs.env_cfg_src).name] = cs.remote_env_cfg
    if cs.remote_asset:
        remote_map[Path(cs.asset_src).name] = cs.remote_asset
    # regenerated reference clips (if any) ship to clip_remote_dir under their own basename
    for clip in (adapted.get("reference_clips") or []):
        name = Path(clip).name
        if cs.clip_remote_dir:
            remote_map[name] = f"{cs.clip_remote_dir.rstrip('/')}/{name}"

    plan = build_plan(work_dir, remote_map, launch_cmd=cs.launch_cmd, stamp=stamp,
                      require_refs=references_required)

    a, r, h = adapted["actuator"], adapted["reward_thresholds"], adapted["health_band"]
    derived = {
        "effort": a["effort"], "Kp": a["Kp"], "Kd": a["Kd"],
        "n_joints": adapted["dims"]["n_actuated_joints"],
        "stand_height": r.get("stand_height"), "base_h_target": h.get("base_h_target"),
        "n_edits_applied": sum(1 for e in edits if e.changed and e.applied),
        "n_flagged_advisory": sum(1 for e in edits if not e.applied),
    }
    return AdaptationBundle(adapted=adapted, edits=edits, roundtrip_ok=not bad,
                            roundtrip_bad=bad, plan=plan, work_dir=str(wd),
                            derived_summary=derived,
                            composition_id=comp.id, composition_valid=not comp_issues,
                            composition_issues=comp_issues, component_adapt_plan=comp_plan,
                            references_required=references_required,
                            legacy_deploy_enabled=cs.legacy_deploy_enabled)


def consistency_report(b: AdaptationBundle) -> dict:
    """⑦ honesty layer (§4⑦ 配置三方校验): classify URDF-derived vs current asset/env_cfg values.

      agree     — current value == URDF-derived (framework reproduced; no change)
      corrected — current value disagreed with URDF and was fixed; effort/velocity mismatches are
                  sim2real hazards (sim asked for torque/speed the real motor can't give)
      flagged   — derived but NOT auto-applied (per-motor Kp/Kd, §8.1: advisory, human/LLM decides)
    """
    agree, corrected, flagged = [], [], []
    for e in b.edits:
        tag = f"{e.file}:{e.field} {e.old}→{e.new}"
        if not e.applied:
            flagged.append(f"{tag} ({e.kind})")
        elif e.changed:
            corrected.append(tag)
        else:
            agree.append(f"{e.file}:{e.field}")
    hazards = [c for c in corrected if "velocity" in c.lower() or "effort" in c.lower()]
    verdict = "clean" if not corrected and not flagged else (
        "corrections_applied" if corrected else "advisory_only")
    return {
        "verdict": verdict,
        "n_agree": len(agree), "n_corrected": len(corrected), "n_flagged": len(flagged),
        "corrected": corrected, "flagged": flagged,
        "sim2real_hazards_fixed": hazards,
    }


def deploy_readiness(b: AdaptationBundle) -> dict:
    """④ '写走守门' gate before deploy.execute: is this ConfigSet safe to push to remote?

    Blocks if the materialized env_cfg failed round-trip (fields didn't write / are missing on this
    robot), the framework composition is invalid for the robot, mapped files are missing from the
    work_dir, or no reference clips are being shipped (§6: regenerate the full reference set before
    deploy — the committed clip has drifted). Returns {ready, blockers}.
    """
    blockers: List[str] = []
    if not b.roundtrip_ok:
        blockers.append(f"env_cfg round-trip failed (fields missing/unwritten): {b.roundtrip_bad}")
    if not b.composition_valid:
        blockers.append(f"framework composition invalid for robot: {b.composition_issues}")
    if b.plan.missing:
        blockers.append(f"mapped files missing from work_dir: {b.plan.missing}")
    if b.references_required and not any(it.role == "reference_clip" for it in b.plan.items):
        blockers.append("no reference clips in plan (§6: regenerate the full reference set before deploy)")
    if not b.legacy_deploy_enabled:
        blockers.append("legacy source-file deployment is disabled; build and deploy the product payload")
    return {"ready": not blockers, "blockers": blockers}


def render_bundle(b: AdaptationBundle) -> str:
    d = b.derived_summary
    cp = b.component_adapt_plan
    lines = [
        "ADAPTATION BUNDLE",
        f"  framework [{b.composition_id}]: {'VALID' if b.composition_valid else 'ISSUES ' + str(b.composition_issues)}",
        f"     invariant={cp.get('invariant', [])} derive={cp.get('derive', [])} "
        f"regenerate={cp.get('regenerate', [])} scale={cp.get('scale', [])}",
        f"  derived: effort={d['effort']} Kp={d['Kp']} joints={d['n_joints']}",
        f"           stand_height={d['stand_height']} base_h={d['base_h_target']}",
        f"  materialize: {d['n_edits_applied']} real edit(s), {d['n_flagged_advisory']} flagged advisory, "
        f"rest no-op (==proven)",
        f"  roundtrip: {'PASS' if b.roundtrip_ok else 'FAIL ' + str(b.roundtrip_bad)}",
        "",
        render_plan(b.plan),
    ]
    return "\n".join(lines)


__all__ = [
    "AdaptationBundle",
    "ConfigSet",
    "build_config_set",
    "consistency_report",
    "deploy_readiness",
    "plan_adaptation",
    "render_bundle",
]
