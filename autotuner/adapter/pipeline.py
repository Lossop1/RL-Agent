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
from typing import Dict, List, Optional

from autotuner.adapter.adapt import adapt
from autotuner.adapter.materialize import materialize, roundtrip_verify, Edit
from autotuner.adapter.deploy import build_plan, render_plan, DeployPlan
from autotuner.framework_library import (
    get_composition, validate_composition, adapt_plan as _component_adapt_plan,
)

_DEFAULT_CAPS = ("quadruped_12dof", "blind_actor", "amp_reference")


@dataclass
class ConfigSet:
    """Operational unit (§2), backend twin of the console's ConfigSet: robot + framework + remote dests."""
    urdf: str
    env_cfg_src: str                                   # local source env_cfg to adapt+ship
    asset_src: str                                     # local source asset to adapt+ship
    remote_env_cfg: str                                # remote destination for env_cfg
    remote_asset: str                                  # remote destination for asset
    mass: Optional[float] = None
    regenerate_clips: bool = False                     # regenerate AMP references into work_dir (§6)
    clip_remote_dir: Optional[str] = None              # remote dir for regenerated clips
    launch_cmd: Optional[str] = None
    framework_composition: str = "taili_amp_proven"    # ② framework instance (Taili strategy)
    robot_capabilities: tuple = _DEFAULT_CAPS          # morphology caps for §② applicability check


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


def plan_adaptation(cs: ConfigSet, work_dir: str, stamp: str,
                    require_refs: bool = True) -> AdaptationBundle:
    """Run the full offline adaptation→deploy-plan chain. No remote writes.

    Resolves the ② framework composition first (§5 ConfigSet{framework} → Adapter): validates the
    robot supports every component's morphology, and records the per-component adapt_plan
    (invariant/derive/regenerate/scale) so the bundle documents what the Adapter touched and why.
    """
    comp = get_composition(cs.framework_composition)
    comp_issues = validate_composition(comp, tuple(cs.robot_capabilities))
    comp_plan = _component_adapt_plan(comp)

    wd = Path(work_dir)
    adapted = adapt(cs.urdf, mass=cs.mass,
                    regenerate_clips_to=str(wd) if cs.regenerate_clips else None)
    edits = materialize(adapted, cs.env_cfg_src, cs.asset_src, work_dir)
    bad = roundtrip_verify(adapted, str(wd / Path(cs.env_cfg_src).name))

    remote_map = {
        Path(cs.env_cfg_src).name: cs.remote_env_cfg,
        Path(cs.asset_src).name: cs.remote_asset,
    }
    # regenerated reference clips (if any) ship to clip_remote_dir under their own basename
    for clip in (adapted.get("reference_clips") or []):
        name = Path(clip).name
        if cs.clip_remote_dir:
            remote_map[name] = f"{cs.clip_remote_dir.rstrip('/')}/{name}"

    plan = build_plan(work_dir, remote_map, launch_cmd=cs.launch_cmd, stamp=stamp,
                      require_refs=require_refs)

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
                            composition_issues=comp_issues, component_adapt_plan=comp_plan)


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
    if not any(it.role == "reference_clip" for it in b.plan.items):
        blockers.append("no reference clips in plan (§6: regenerate the full reference set before deploy)")
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


if __name__ == "__main__":
    import tempfile
    from autotuner.adapter.adapt import _DEFAULT_URDF

    cs = ConfigSet(
        urdf=_DEFAULT_URDF,
        env_cfg_src="autotuner/blind_locomotion/env_edit/taili_amp_env_cfg.py",
        asset_src="autotuner/blind_locomotion/assets/taili.py",
        remote_env_cfg=("/root/gpufree-data/robot_lab/source/robot_lab/robot_lab/tasks/"
                        "direct/taili_amp/taili_amp_env_cfg.py"),
        remote_asset="/root/gpufree-data/robot_lab/.../asset/taili.py",
        launch_cmd="/opt/conda/envs/isaaclab/bin/python3 /root/tp_train_real.py",
    )
    wd = tempfile.mkdtemp(prefix="adapter_pipeline_")
    bundle = plan_adaptation(cs, wd, stamp="20260629-231500")
    print(render_bundle(bundle))
    print("\n(Taili identity: env_cfg edits all no-op, asset calf velocity already URDF-correct at 8.27, "
          "roundtrip PASS. DRY-RUN — deploy.execute(confirm=True) required to write remote.)")
