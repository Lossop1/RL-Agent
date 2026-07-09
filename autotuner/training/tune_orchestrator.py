"""Autonomous tuning orchestrator: given a policy checkpoint, run the full measure→analyze→tune→
train→re-measure loop UNATTENDED until the acceptance score stops improving or a budget is hit.

This is the capability that lets the system "complete a user-assigned tuning task" without a human
confirming each step. It chains the primitives built this session — strategy_edit (apply/rollback),
acceptance_run (measure+score), and the hardened RemoteSSH (deploy/launch/monitor) — and encodes the
gate→lever tuning heuristic learned empirically (docs/taili_tuning_followups.md D3h–D3l), including the
guard-rails discovered the hard way (F2 can't be forced past ~1.8; B2 is a metric artifact to skip).

The pure decision logic (analyze_gaps, propose_change) is unit-tested off-box; run_campaign drives the
GPU box with stall-detection + bounded auto-restart so a transient hang can't wedge the campaign.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ── gate → reward-lever heuristic (key, per-iteration step, hard cap) — learned empirically ─────────
# Each failing family maps to the reward weight(s) that move it. Steps are deliberately SMALL and
# capped: the F2 cap is 1.8 because w_torque_margin=2.2 deterministically destabilized training (D3l).
GATE_LEVERS: dict[str, list[tuple[str, float, float]]] = {
    "B3": [("w_clearance_over", 0.4, 3.0)],                                 # over-lift
    "B4": [("w_diagonal_contact", 0.1, 0.8), ("w_duty_balance", 0.1, 0.8)], # gait symmetry
    "F2": [("w_torque_margin", 0.15, 1.8)],                                 # torque — GENTLE (D3l)
    # A2 yaw: w_tracking_yaw is PROVEN INEFFECTIVE (weight-immune, 2.5→3.1 no effect, D3). The lever that
    # actually moved yaw p90 0.7→0.47 is rew_imitate (turn-gated) + pure-yaw sampling. Bump rew_imitate.
    "A2": [("rew_imitate", 0.3, 3.0)],
    # A3 duty_min<0.95 (foot lifts during stand): w_stand_contact is the validated lever, NOT w_stand
    # (which rewards body stillness only, not four-foot planting). w_stand_contact is stand-gated.
    "A3": [("w_stand_contact", 0.3, 3.0)],
    "C":  [("w_stand_contact", 0.3, 3.0)],                                  # C is coupled to A3
    "B1": [("w_landing_impact_late", 0.2, 1.6)],                            # landing impact
    "A1": [("w_wrong_dir", 0.1, 1.2)],                                      # backward tracking tail
    # D[stairs_up] ascending climb: w_climb rewards upward base velocity on discrete terrain. Structural
    # levers (foot_h4 perception, stairs_up curriculum) are code-level, not weight-bumpable here.
    "D":  [("w_climb", 0.3, 2.5)],
}
# Families to SKIP as tuning targets — proven metric artifacts, not policy deficiencies (D3j).
SKIP_FAMILIES = {"B2"}


@dataclass
class GapAnalysis:
    family: str
    gate: str
    measured: float | None
    threshold: float | None
    margin: float | None            # how far over the spec (measured-threshold), None if unparseable


def _parse_stat(detail: str) -> tuple[float | None, float | None]:
    """Pull the (measured, threshold) pair from a physeval detail line, e.g.
    'stance slip p90=0.226 m/s <= 0.05' -> (0.226, 0.05); 'p90=0.279 <= 0.150' -> (0.279, 0.150)."""
    thr = None
    mt = re.search(r"<=\s*([0-9]*\.?[0-9]+)", detail)
    if mt:
        thr = float(mt.group(1))
    meas = None
    mm = re.search(r"(?:p90|p95|=|clearance=|slip\s+p90=|settle=)\s*([0-9]*\.?[0-9]+)", detail)
    # prefer an explicit pNN= or =value that is NOT the threshold
    cands = [float(x) for x in re.findall(r"=\s*([0-9]*\.?[0-9]+)", detail)]
    cands = [c for c in cands if thr is None or abs(c - thr) > 1e-9]
    if cands:
        meas = max(cands)                                   # the worst (largest) measured stat
    elif mm:
        meas = float(mm.group(1))
    return meas, thr


def analyze_gaps(verdict: dict[str, Any]) -> list[GapAnalysis]:
    """Rank the failing hard gates by how far over spec they are (worst first). Metric-artifact
    families and unmovable gates are excluded from the tuning targets."""
    gates = verdict.get("gates") or {}
    out: list[GapAnalysis] = []
    for gate, v in gates.items():
        if v.get("ok"):
            continue
        fam = re.match(r"[A-F]\d", gate)
        fam = fam.group(0) if fam else gate.split("[")[0]
        if fam in SKIP_FAMILIES or fam not in GATE_LEVERS:
            continue
        meas, thr = _parse_stat(str(v.get("detail", "")))
        margin = (meas - thr) if (meas is not None and thr is not None) else None
        out.append(GapAnalysis(fam, gate, meas, thr, margin))
    # rank: largest normalized margin first (unparseable margins sink to the bottom)
    out.sort(key=lambda g: (g.margin / g.threshold) if (g.margin and g.threshold) else -1, reverse=True)
    return out


def propose_change(gaps: list[GapAnalysis], current: dict[str, float],
                   history: list[dict[str, Any]]) -> dict[str, float]:
    """Propose the next {key: value} tuning delta for the worst *movable* gap whose lever is not yet
    capped and whose last step didn't regress. Returns {} if nothing productive remains."""
    # exclude BOTH regressed AND apply_failed: a rejected apply (bounds/hang) must be skipped too, else the
    # loop re-proposes the same lever every iteration and livelocks, burning the whole campaign budget (audit 0706).
    tried_and_failed = {h["key"] for h in history if h.get("outcome") in ("regressed", "apply_failed")}
    for gap in gaps:
        for key, step, cap in GATE_LEVERS.get(gap.family, []):
            cur = float(current.get(key, 0.0))
            if cur >= cap or key in tried_and_failed:
                continue
            return {key: round(min(cur + step, cap), 4)}
    return {}


@dataclass
class Campaign:
    goal: str = "benchmark"
    max_iters: int = 6
    steps_per_iter: int = 18000    # policy peaks ~18k (more overtrains B4/A2); D3m
    num_envs: int = 1024           # 2048 hit a memory-stall wall ~16k; 1024 runs clean
    max_stall_restarts: int = 2
    iterations: list[dict[str, Any]] = field(default_factory=list)
    best_score: int = -1
    best_checkpoint: str = ""

    def summary(self) -> dict[str, Any]:
        return {"goal": self.goal, "iterations": self.iterations,
                "best_score": self.best_score, "best_checkpoint": self.best_checkpoint,
                "n_iters": len(self.iterations)}


# ── driver: chains strategy_edit + RemoteSSH + acceptance_run into the unattended loop ──────────────
def _score(verdict: dict[str, Any]) -> int:
    """Passing hard gate-instances (the number the campaign maximizes)."""
    return sum(1 for v in (verdict.get("gates") or {}).values() if v.get("ok"))


class TuningDriver:
    """Remote operations for a campaign, on the hardened transport. Every step is best-effort and
    logged; a stall triggers bounded kill+restart so the box can't wedge the campaign."""

    def __init__(self, ssh_json: str = "config/ssh.json",
                 python: str = "/opt/conda/envs/isaaclab/bin/python",
                 data_root: str = "/root/gpufree-data", log=print):
        from autotuner.training.acceptance_run import _ssh_from_json
        self.remote = _ssh_from_json(ssh_json)
        self.ssh_json, self.python, self.data_root, self.log = ssh_json, python, data_root, log

    def close(self):
        try:
            self.remote.close()
        except Exception:
            pass

    def apply_and_deploy(self, changes: dict[str, float], note: str, stamp: str) -> str | None:
        """Apply the tuning to the local strategy, ship a fresh payload with it, return payload path."""
        import shlex
        from autotuner.training.strategy_edit import apply_weight_changes
        res = apply_weight_changes(changes, note=note, stamp=stamp)
        if not res["ok"]:
            self.log(f"[orchestrator] apply rejected: {res['errors']}")
            return None
        base = self.remote.exec_out(
            f"ls -1td {self.data_root}/training_payloads/taili_blind_runtime_* | head -1").strip()
        new = f"{self.data_root}/training_payloads/taili_blind_runtime_campaign_{stamp}"
        self.remote.exec(f"cp -r {shlex.quote(base)} {shlex.quote(new)}", timeout=200)
        sftp = self.remote._client.open_sftp()
        # ship the current strategy AND the current source (reward/env/physeval), so structural code
        # fixes — not just weights — reach the box each iteration.
        sftp.put("autotuner/blind_locomotion/taili_blind_config.yaml",
                 f"{new}/taili_blind_runtime/taili_blind_config.yaml")
        for f in ("blind_tp_env.py", "physeval_blind.py", "acceptance_score.py"):
            sftp.put(f"autotuner/blind_locomotion/{f}", f"{new}/taili_blind_runtime/{f}")
        sftp.put("autotuner/blind_locomotion/acceptance_score.py", f"{new}/acceptance_score.py")
        sftp.close()
        self.remote.exec(
            f"sed -i -E 's/^(\\s*init_phase:\\s*)[0-9]+/\\13/' "
            f"{shlex.quote(new)}/taili_blind_runtime/taili_blind_config.yaml", timeout=20)
        self.log(f"[orchestrator] applied {res['applied']}, deployed {new.split('/')[-1]}")
        return new

    def launch(self, payload: str, run_id: str, checkpoint: str, num_envs: int) -> None:
        import shlex
        self.remote.exec("tmux kill-session -t taili_train 2>/dev/null; "
                         "pkill -9 -f train_taili 2>/dev/null; sleep 4; true", timeout=30)
        self.remote.exec("tmux new-session -d -s taili_train", timeout=15)
        cmd = (f"cd {self.data_root} && export PYTHONPATH={payload} && "
               f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
               f"{self.python} -m taili_blind_runtime.launch_taili_train --run-id {run_id} "
               f"--checkpoint {shlex.quote(checkpoint)} --total-steps 1500000 -- "
               f"--num_envs {int(num_envs)} 2>&1 | tee /tmp/{run_id}.log")
        self.remote.exec(f"tmux send-keys -t taili_train {shlex.quote(cmd)} Enter", timeout=15)

    def latest_step(self, run_id: str) -> int:
        out = self.remote.exec_out(
            f"grep -oE 'step=[0-9]+' {self.data_root}/taili_runs/{run_id}/train.log 2>/dev/null | tail -1")
        m = re.search(r"(\d+)", out or "")
        return int(m.group(1)) if m else 0

    def alive(self) -> bool:
        return bool(self.remote.exec_out("pgrep -f train_taili | grep -v pgrep | head -1").strip())

    def latest_checkpoint(self, run_id: str) -> str:
        return self.remote.exec_out(
            f"ls -1t {self.data_root}/taili_runs/{run_id}/checkpoints/agent_*.pt 2>/dev/null | head -1").strip()

    def monitor_to_target(self, run_id: str, target_step: int, *, payload: str, checkpoint: str,
                          num_envs: int, max_restarts: int, poll_s: int = 200, sleep=None) -> dict[str, Any]:
        """Poll until target_step, restarting on a detected stall (frozen step) or death up to
        max_restarts. Returns {reached, stalls, last_step, checkpoint}."""
        import time
        sleep = sleep or time.sleep
        last, frozen, stalls = -1, 0, 0
        while True:
            sleep(poll_s)
            step, alive = self.latest_step(run_id), self.alive()
            self.log(f"[orchestrator] {run_id} step={step} alive={alive}")
            if step >= target_step:
                return {"reached": True, "stalls": stalls, "last_step": step,
                        "checkpoint": self.latest_checkpoint(run_id) or checkpoint}
            frozen = frozen + 1 if step == last else 0
            last = step
            if (not alive) or frozen >= 3:                       # death or ~3-poll stall
                if stalls >= max_restarts:
                    return {"reached": False, "stalls": stalls, "last_step": step,
                            "checkpoint": self.latest_checkpoint(run_id) or checkpoint}
                stalls += 1
                ck = self.latest_checkpoint(run_id) or checkpoint
                self.log(f"[orchestrator] STALL/death at {step}; restart {stalls}/{max_restarts} from {ck.split('/')[-1]}")
                self.launch(payload, run_id, ck, num_envs)
                last, frozen = -1, 0

    def measure(self, run_id: str, checkpoint: str, terrains: list[str]) -> dict[str, Any]:
        """Pause training and score the checkpoint vs the spec (physeval → §2 verdict)."""
        from autotuner.training.acceptance_run import run_acceptance
        self.remote.exec("tmux kill-session -t taili_train 2>/dev/null; "
                         "pkill -9 -f train_taili 2>/dev/null; sleep 3; true", timeout=30)
        ck_name = checkpoint.split("/")[-1] if checkpoint else "best_agent.pt"
        return run_acceptance(run_id, terrains, ck_name, 64, self.ssh_json, True, self.python, self.data_root)


def run_campaign(camp: Campaign, from_run: str, from_checkpoint: str, *,
                 ssh_json: str = "config/ssh.json", terrains: list[str] | None = None,
                 stamp_fn=None, log=print) -> dict[str, Any]:
    """Autonomous measure→analyze→tune→train→re-measure loop. Keeps a change that improves the score,
    ROLLS BACK one that regresses (via strategy_edit), and stops when no productive lever remains or
    the iteration budget is hit. Drives the box with stall-recovery; a wedged box ends the iteration,
    not the process."""
    import time
    from autotuner.blind_locomotion.taili_blind_config import load_taili_blind_config, get_config_value
    from autotuner.training.strategy_edit import rollback_last
    terrains = terrains or ["flat"]
    stamp_fn = stamp_fn or (lambda: time.strftime("%Y%m%d_%H%M%S"))
    d = TuningDriver(ssh_json=ssh_json, log=log)
    # launch() needs an ABSOLUTE checkpoint path; a bare name (fine for measure(), which resolves it
    # against the run dir) made train_taili crash with FileNotFoundError and the campaign never stepped.
    if from_checkpoint and not from_checkpoint.startswith("/"):
        from_checkpoint = f"{d.data_root}/taili_runs/{from_run}/checkpoints/{from_checkpoint}"
    history: list[dict[str, Any]] = []
    # DURABLE NO-REPEAT (audit 0706): seed this campaign's exclusion set from the cross-campaign journal so a
    # lever rolled back in a PRIOR campaign (even a prior session) is not re-proposed. The two loops now share
    # one memory. Best-effort: a missing journal never blocks the campaign.
    try:
        from autotuner.locomotion_console.campaign_journal import read_journal, record_iteration
    except Exception:  # noqa: BLE001
        read_journal = record_iteration = None
    if read_journal:
        try:
            for pair in (read_journal("taili").get("tried_and_rolled_back") or []):
                _g, _, _lev = pair.partition("::")
                history.append({"key": (_lev.split()[0] if _lev else pair), "outcome": "regressed"})
        except Exception:  # noqa: BLE001
            pass
    try:
        base = d.measure(from_run, from_checkpoint, terrains)
        camp.best_score, camp.best_checkpoint = _score(base), from_checkpoint
        log(f"[campaign] baseline score {camp.best_score}/19")
        cur_ckpt, cur_run = from_checkpoint, from_run
        for i in range(camp.max_iters):
            gaps = analyze_gaps(base)
            cfg = load_taili_blind_config()
            current = {k: get_config_value(cfg, f"reward.{k}") for _, lv in GATE_LEVERS.items() for k, *_ in lv}
            change = propose_change(gaps, {k: (v or 0.0) for k, v in current.items()}, history)
            if not change:
                log("[campaign] no productive lever remains — stopping"); break
            stamp = stamp_fn()
            log(f"[campaign] iter {i+1}: targeting {gaps[0].gate if gaps else '?'} with {change}")
            payload = d.apply_and_deploy(change, note=f"campaign iter{i+1}", stamp=stamp)
            if not payload:
                history.append({"key": list(change)[0], "outcome": "apply_failed"}); continue
            run_id = f"campaign_{stamp}"
            d.launch(payload, run_id, cur_ckpt, camp.num_envs)
            mon = d.monitor_to_target(run_id, camp.steps_per_iter, payload=payload, checkpoint=cur_ckpt,
                                      num_envs=camp.num_envs, max_restarts=camp.max_stall_restarts)
            verdict = d.measure(run_id, mon["checkpoint"], terrains)
            score = _score(verdict)
            if verdict.get("passed"):
                camp.best_score, camp.best_checkpoint = score, mon["checkpoint"]
                camp.iterations.append({"iter": i + 1, "change": change, "score": score,
                                        "reached": mon["reached"], "checkpoint": mon["checkpoint"],
                                        "benchmark_passed": True})
                log(f"[campaign] BENCHMARK PASSED at iter {i+1} — stopping with the deliverable")
                break
            improved = score > camp.best_score
            log(f"[campaign] iter {i+1} score {score}/19 (best {camp.best_score}) reached={mon['reached']}")
            history.append({"key": list(change)[0], "change": change, "score": score,
                            "outcome": "improved" if improved else "regressed", "reached": mon["reached"]})
            if record_iteration:  # durable memory for NO-REPEAT/DECIDE across campaigns (audit 0706)
                try:
                    record_iteration("taili", target_gate=(gaps[0].gate if gaps else ""),
                                     lever=str(list(change)[0]), decision=("kept" if improved else "rolled_back"),
                                     score_before=f"{camp.best_score}/19", score_after=f"{score}/19",
                                     result_run=run_id, config_diff=str(change), note="run_campaign")
                except Exception:  # noqa: BLE001
                    pass
            camp.iterations.append({"iter": i + 1, "change": change, "score": score,
                                    "reached": mon["reached"], "checkpoint": mon["checkpoint"]})
            if improved:
                camp.best_score, camp.best_checkpoint = score, mon["checkpoint"]
                base, cur_ckpt, cur_run = verdict, mon["checkpoint"], run_id
            else:
                rollback_last()                                   # undo the non-improving change
                log(f"[campaign] iter {i+1} did not improve — rolled back {change}")
        return camp.summary()
    finally:
        d.close()


FULL_BATTERY = ["flat", "rough", "stairs", "stairs_up", "slope", "boxes", "push"]   # stairs_up = 爬楼梯(0706)


def produce_policy(*, from_run: str = "", from_checkpoint: str = "", max_iters: int = 8,
                   steps_per_iter: int = 18000, num_envs: int = 1024,
                   ssh_json: str = "config/ssh.json", report_path: str = "/tmp/policy_report.json",
                   log=print) -> dict[str, Any]:
    """END-TO-END product pipeline: produce the best-possible benchmark policy and a deliverable
    report. Bootstraps from the BEST_CHECKPOINT registry (or the given run/checkpoint), then runs the
    autonomous measure(full battery)->analyze->tune->train->re-measure loop until the benchmark
    passes, no productive lever remains, or the iteration budget ends. Emits a report with the final
    per-gate table, the tuning history, and the best checkpoint path — the product's deliverable."""
    import json as _json
    from pathlib import Path as _Path

    if not (from_run and from_checkpoint):
        # bootstrap from the best-known measured policy
        from autotuner.training.acceptance_run import _ssh_from_json
        remote = _ssh_from_json(ssh_json)
        try:
            raw = remote.exec_out("cat /root/gpufree-data/taili_runs/BEST_CHECKPOINT.json 2>/dev/null")
            best = _json.loads(raw or "{}") if raw.strip() else {}
        finally:
            remote.close()
        ck = str(best.get("checkpoint") or "")
        if not ck:
            raise SystemExit("no BEST_CHECKPOINT registry and no explicit run/checkpoint — "
                             "run one acceptance measurement first to seed the registry")
        parts = ck.split("/")
        from_run, from_checkpoint = parts[-3], parts[-1]
        log(f"[produce] bootstrapping from registry best: {from_run}/{from_checkpoint} "
            f"(score {best.get('score')})")
    camp = Campaign(goal="benchmark", max_iters=max_iters, steps_per_iter=steps_per_iter,
                    num_envs=num_envs)
    summary = run_campaign(camp, from_run, from_checkpoint, ssh_json=ssh_json,
                           terrains=FULL_BATTERY, log=log)
    summary["benchmark_passed"] = any(it.get("benchmark_passed") for it in summary.get("iterations", []))
    summary["deliverable"] = {
        "best_checkpoint": summary.get("best_checkpoint"),
        "best_score": summary.get("best_score"),
        "battery": FULL_BATTERY,
        "report": report_path,
    }
    _Path(report_path).write_text(_json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[produce] deliverable report -> {report_path}")
    return summary


def main(argv=None) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Autonomous tuning campaign")
    ap.add_argument("from_run")
    ap.add_argument("from_checkpoint")
    ap.add_argument("--max-iters", type=int, default=6)
    # SAFE REGIME defaults (audit 0706): the CLI defaults were 30000/2048 — the exact overtrain(>18k peak)+OOM
    # (2048 hits a memory wall ~16k) regime the campaign was tuned to AVOID, and action_run_campaign inherits
    # these. Match the safe Campaign dataclass defaults so the one-click autonomous path can't overtrain/OOM.
    ap.add_argument("--steps-per-iter", type=int, default=18000)
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--config", default="config/ssh.json")
    ap.add_argument("--terrains", nargs="*", default=["flat"])
    ap.add_argument("--out", default="")
    ap.add_argument("--produce", action="store_true",
                    help="END-TO-END product mode: full battery, bootstrap from BEST registry, deliverable report")
    a = ap.parse_args(argv)
    if a.produce:
        summary = produce_policy(from_run=a.from_run if a.from_run != "auto" else "",
                                 from_checkpoint=a.from_checkpoint if a.from_checkpoint != "auto" else "",
                                 max_iters=a.max_iters, steps_per_iter=a.steps_per_iter,
                                 num_envs=a.num_envs, ssh_json=a.config,
                                 report_path=a.out or "/tmp/policy_report.json")
        print(json.dumps(summary.get("deliverable", {}), indent=2))
        return 0
    camp = Campaign(max_iters=a.max_iters, steps_per_iter=a.steps_per_iter, num_envs=a.num_envs)
    summary = run_campaign(camp, a.from_run, a.from_checkpoint, ssh_json=a.config, terrains=a.terrains)
    print(json.dumps(summary, indent=2))
    if a.out:
        from pathlib import Path
        Path(a.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
