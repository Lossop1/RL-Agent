"""§3 A 一键产出 orchestrator — the deterministic backbone (§5 data flow, §10 控制流).

Sequences the whole flow: ConfigSet → adapt+materialize+regenerate → ④readiness gate → deploy
(gated) → train (gated) → evaluate (gated) → §2 verdict. Each ONLINE step (deploy/train/eval) is
OUTWARD-FACING / GPU, so it runs ONLY when explicitly authorized (allow_deploy/allow_train/
allow_eval) AND its injected executor succeeds; otherwise the flow stops at a clear outcome with the
full offline bundle + audit. The control flow (sequence, gates, decisions) is deterministic and
unit-tested with mock executors; the executors plug in the real paramiko deploy / training launch /
physeval. The LLM is nowhere in this control path (§10 free).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from autotuner.adapter.pipeline import (
    ConfigSet, plan_adaptation, consistency_report, deploy_readiness,
)
from autotuner.adapter.audit import build_audit_record


@dataclass
class StepResult:
    step: str
    ok: bool
    detail: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class OrchestrationResult:
    outcome: str                 # see OUTCOMES below
    steps: List[StepResult]
    bundle_summary: dict
    readiness: dict
    consistency: dict
    verdict: Optional[dict] = None
    audit: dict = field(default_factory=dict)


# terminal outcomes
OUTCOMES = (
    "ADAPT_FAILED",        # roundtrip failed → can't even materialize correctly
    "DEPLOYED_DRYRUN",     # offline complete, deploy not authorized (default safe stop)
    "NOT_READY",           # readiness gate blocked (missing refs / fields) and deploy authorized
    "DEPLOY_FAILED",       # deploy executor reported failure
    "DEPLOYED",            # deployed, training not authorized
    "TRAIN_FAILED",        # training executor reported failure
    "TRAINED",             # trained, evaluation not authorized
    "EVAL_FAILED",         # evaluation executor errored
    "EVAL_FAILED_GATES",   # evaluated but did NOT pass the §2 benchmark
    "DELIVERED",           # full flow, §2 benchmark passed
)


class Executors(Protocol):
    def deploy(self, plan, stamp: str) -> StepResult: ...                 # backup-then-put + verify
    def train(self, config_set: ConfigSet, stamp: str) -> StepResult: ...  # data: {"checkpoint": path}
    def evaluate(self, checkpoint: str) -> StepResult: ...                 # data: {"scorecard_text": str}


def orchestrate(cs: ConfigSet, work_dir: str, stamp: str, *,
                executors: Optional[Executors] = None,
                allow_deploy: bool = False, allow_train: bool = False, allow_eval: bool = False,
                require_ready: bool = True) -> OrchestrationResult:
    steps: List[StepResult] = []

    # ── offline: adapt + materialize (+ regenerate if cs.regenerate_clips) + deploy plan ──
    bundle = plan_adaptation(cs, work_dir, stamp)
    cons = consistency_report(bundle)
    rd = deploy_readiness(bundle)
    audit = build_audit_record(bundle, cons, rd, stamp)
    steps.append(StepResult("adapt+materialize", bundle.roundtrip_ok,
                            f"roundtrip_ok={bundle.roundtrip_ok}; {bundle.derived_summary.get('n_edits_applied', 0)} edits",
                            {"composition": bundle.composition_id}))
    steps.append(StepResult("readiness", rd["ready"], "; ".join(rd["blockers"]) or "ready"))

    def result(outcome, verdict=None):
        return OrchestrationResult(outcome, steps, bundle.derived_summary, rd, cons, verdict, audit)

    if not bundle.roundtrip_ok and require_ready:
        return result("ADAPT_FAILED")
    if not allow_deploy:
        return result("DEPLOYED_DRYRUN")
    if require_ready and not rd["ready"]:
        return result("NOT_READY")
    if executors is None:
        return result("DEPLOYED_DRYRUN")

    # ── deploy (gated, outward-facing) ──
    dep = executors.deploy(bundle.plan, stamp)
    steps.append(dep)
    if not dep.ok:
        return result("DEPLOY_FAILED")
    if not allow_train:
        return result("DEPLOYED")

    # ── train (gated, GPU) ──
    tr = executors.train(cs, stamp)
    steps.append(tr)
    if not tr.ok:
        return result("TRAIN_FAILED")
    checkpoint = tr.data.get("checkpoint", "")
    if not allow_eval:
        return result("TRAINED")

    # ── evaluate (gated, GPU) → §2 verdict ──
    ev = executors.evaluate(checkpoint)
    steps.append(ev)
    if not ev.ok:
        return result("EVAL_FAILED")
    from autotuner.blind_locomotion.acceptance_aggregate import aggregate
    verdict = aggregate([ev.data.get("scorecard_text", "")])
    return result("DELIVERED" if verdict["passed"] else "EVAL_FAILED_GATES", verdict)


def render(r: OrchestrationResult) -> str:
    lines = [f"ORCHESTRATION → {r.outcome}", f"  framework: {r.bundle_summary.get('effort', '')}"]
    for s in r.steps:
        lines.append(f"  [{'OK ' if s.ok else 'XX '}] {s.step}: {s.detail}")
    if r.verdict is not None:
        lines.append(f"  §2 benchmark: {'PASS' if r.verdict['passed'] else 'FAIL'} "
                     f"({r.verdict['n_present']}/{r.verdict['n_hard']} hard families)")
    return "\n".join(lines)
