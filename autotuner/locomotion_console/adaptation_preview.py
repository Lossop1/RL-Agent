"""Console presentation for the adapter chain (③④⑦) — a read-only dry-run preview.

Runs `autotuner.adapter.pipeline.plan_adaptation` for a robot preset and summarizes, for the UI:
what the Adapter would DERIVE (effort/Kp/mass/base_h), the ⑦ consistency verdict (URDF-vs-asset,
sim2real hazards), and the ④ deploy-readiness gate. Pure dry-run: copies are written to a temp
work_dir; nothing remote is touched. No SSH, no GPU.
"""
from __future__ import annotations

import tempfile
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DeployItemInfo(BaseModel):
    remote: str
    role: str
    size: int


class AdaptationPreviewInfo(BaseModel):
    robot: str
    available: bool = False
    message: str = ""
    composition_id: str = ""
    composition_valid: bool = True
    effort: Dict[str, float] = Field(default_factory=dict)
    kp: Dict[str, float] = Field(default_factory=dict)
    n_joints: int = 0
    mass_kg: float = 0.0
    base_h_target: Optional[float] = None
    roundtrip_ok: bool = True
    roundtrip_bad: List[str] = Field(default_factory=list)
    consistency_verdict: str = ""
    n_agree: int = 0
    n_corrected: int = 0
    n_flagged: int = 0
    sim2real_hazards_fixed: List[str] = Field(default_factory=list)
    ready: bool = False
    blockers: List[str] = Field(default_factory=list)
    plan_items: List[DeployItemInfo] = Field(default_factory=list)


def available_robots() -> List[str]:
    from autotuner.adapter.__main__ import PRESETS
    return list(PRESETS)


def build_adaptation_preview(robot: str = "taili") -> AdaptationPreviewInfo:
    try:
        from autotuner.adapter.__main__ import build_config_set
        from autotuner.adapter.pipeline import (
            plan_adaptation, consistency_report, deploy_readiness,
        )
        cs = build_config_set(robot)
        wd = tempfile.mkdtemp(prefix=f"preview_{robot}_")
        bundle = plan_adaptation(cs, wd, stamp="preview")
        rep = consistency_report(bundle)
        rd = deploy_readiness(bundle)
        a = bundle.adapted["actuator"]
        prov = bundle.adapted["provenance"]
        return AdaptationPreviewInfo(
            robot=robot, available=True,
            composition_id=bundle.composition_id, composition_valid=bundle.composition_valid,
            effort={k: float(v) for k, v in a["effort"].items()},
            kp={k: float(v) for k, v in a["Kp"].items()},
            n_joints=bundle.adapted["dims"]["n_actuated_joints"],
            mass_kg=float(prov.get("mass_kg", 0.0)),
            base_h_target=bundle.adapted["health_band"].get("base_h_target"),
            roundtrip_ok=bundle.roundtrip_ok, roundtrip_bad=list(bundle.roundtrip_bad),
            consistency_verdict=rep["verdict"], n_agree=rep["n_agree"],
            n_corrected=rep["n_corrected"], n_flagged=rep["n_flagged"],
            sim2real_hazards_fixed=list(rep["sim2real_hazards_fixed"]),
            ready=rd["ready"], blockers=list(rd["blockers"]),
            plan_items=[DeployItemInfo(remote=it.remote, role=it.role, size=it.size)
                        for it in bundle.plan.items],
        )
    except Exception as exc:  # noqa: BLE001 — surface adaptation failures as a displayable state
        return AdaptationPreviewInfo(robot=robot, available=False,
                                     message=f"{type(exc).__name__}: {exc}")
