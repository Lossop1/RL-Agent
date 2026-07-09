"""Adaptation audit trail (§9 'every output carries provenance, all into audit_trail'; §12 可审计/有出处).

The adapter chain computes a rich AdaptationBundle, but transparency is a non-negotiable quality —
so this serializes the WHOLE thing into one structured, persistable record: provenance (URDF / mass /
leg / framework), the framework composition + adapt_plan, every materialize edit (old→new, whether it
changed/was applied), the ⑦ consistency verdict, the ④ readiness verdict, and the exact deploy plan
(with checksums). You can't audit what you don't record. Pure dict + JSON; no SSH, no GPU.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional


def build_audit_record(bundle, consistency: dict, readiness: dict, stamp: str,
                       configset: Optional[dict] = None) -> dict:
    """Full auditable record of one adaptation. `stamp` supplied by caller (no wall-clock)."""
    plan = bundle.plan
    return {
        "stamp": stamp,
        "provenance": bundle.adapted.get("provenance", {}),
        "framework": {
            "composition": bundle.composition_id,
            "valid": bundle.composition_valid,
            "issues": list(bundle.composition_issues),
            "adapt_plan": bundle.component_adapt_plan,
        },
        "derived": bundle.derived_summary,
        "materialize": {
            "roundtrip_ok": bundle.roundtrip_ok,
            "roundtrip_bad": list(bundle.roundtrip_bad),
            "edits": [asdict(e) for e in bundle.edits],
        },
        "consistency": consistency,
        "readiness": readiness,
        "deploy_plan": {
            "work_dir": plan.work_dir,
            "backup_suffix": plan.backup_suffix,
            "launch_cmd": plan.launch_cmd,
            "warnings": list(plan.warnings),
            "items": [{"remote": it.remote, "role": it.role, "sha256": it.sha256, "size": it.size}
                      for it in plan.items],
        },
        "configset": configset,
    }


def write_audit(record: dict, path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return str(p)
