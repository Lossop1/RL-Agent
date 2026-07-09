"""Run physeval on the remote box for a trained checkpoint and score it against docs/taili_spec.md.

Product-grade benchmark-measurement loop: the deterministic core (SSH transport, spec scorer) is the
hardened RemoteSSH (accept-new host key, shlex-safe interpolation) plus the unit-tested
acceptance_aggregate/acceptance_score. Given a run id it reads the run's run.json for the payload +
agent config, launches `taili_blind_runtime.physeval_blind` for each terrain, and merges the
scorecards into the §2 verdict. Refuses to contend with an active training process unless --force.

    python -m autotuner.training.acceptance_run <run_id|newest> [--terrains flat rough ...]
                                                [--checkpoint agent_50000.pt] [--num-envs 64]
                                                [--config config/ssh.json] [--force] [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from autotuner.training.remote import RemoteSSH
from autotuner.blind_locomotion import acceptance_aggregate as AGG

DEFAULT_PY = "/opt/conda/envs/isaaclab/bin/python"
DEFAULT_DATA_ROOT = "/root/gpufree-data"


def _ssh_from_json(path: str) -> RemoteSSH:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if not str(cfg.get("ssh_host") or "").strip():
        raise SystemExit(f"no ssh_host in {path}")
    r = RemoteSSH(cfg)
    r.connect_timeout = r.banner_timeout = r.auth_timeout = 15
    return r


def _resolve_run(remote: RemoteSSH, run_id: str, data_root: str) -> str:
    runs = f"{data_root}/taili_runs"
    if run_id in ("newest", "latest", ""):
        out = remote.exec_out(f"ls -1td {shlex.quote(runs)}/*/ 2>/dev/null | head -1")
        return out.strip().rstrip("/")
    # allowlist the run id (reaches a remote shell)
    import re
    if not re.fullmatch(r"[A-Za-z0-9._\-]+", run_id):
        raise SystemExit(f"invalid run id: {run_id!r}")
    return f"{runs}/{run_id}"


def _maybe_update_best_registry(remote: RemoteSSH, data_root: str, verdict: dict) -> bool:
    """Keep <data_root>/taili_runs/BEST_CHECKPOINT.json pointing at the highest-scoring measured
    checkpoint. Diagnostics/tooling can then target the alias 'best' instead of guessing 'newest'
    (which today picked an under-converged fragment). Best-effort: never fails the measurement."""
    try:
        import re as _re
        gates = verdict.get("gates") or {}
        score = sum(1 for v in gates.values() if v.get("ok"))
        # HEALTH TIEBREAK (0706): gate COUNT alone is a lossy quality signal — two 10/25 policies can differ
        # sharply in fall rate (dirclr agent_18000 had stairs 9.4% vs the registered July-5 policy's 18.8%),
        # yet strict `score > prev` never promotes the healthier one. Break ties on total fall_rate (lower
        # is better), parsed from the D-gate details ("... fall_rate=0.094=0 ...").
        fall = 0.0
        for v in gates.values():
            m = _re.search(r"fall_rate=([0-9.]+)", str(v.get("detail", "")))
            if m:
                fall += float(m.group(1))
        reg = f"{data_root}/taili_runs/BEST_CHECKPOINT.json"
        prev_score, prev_fall = -1, 1e9
        try:
            _p = json.loads(remote.exec_out(f"cat {shlex.quote(reg)} 2>/dev/null") or "{}")
            prev_score = int(_p.get("score", -1))
            prev_fall = float(_p.get("total_fall_rate", 1e9))
        except Exception:
            prev_score, prev_fall = -1, 1e9
        # promote on strictly higher score, OR equal score with strictly lower total fall rate
        better = (score > prev_score) or (score == prev_score and fall < prev_fall - 1e-9)
        if not better:
            return False
        entry = json.dumps({
            "score": score,
            "total_fall_rate": round(fall, 4),
            "checkpoint": f"{verdict.get('run', '')}/checkpoints/{verdict.get('checkpoint', '')}",
            "run": verdict.get("run", ""),
            "gates_passed": sorted(k for k, v in gates.items() if v.get("ok")),
        }, ensure_ascii=False)
        remote.exec_out(f"printf '%s' {shlex.quote(entry)} > {shlex.quote(reg)}")
        return True
    except Exception:
        return False


def run_acceptance(run_id: str, terrains: list[str], checkpoint: str, num_envs: int,
                   ssh_json: str, force: bool, python: str, data_root: str) -> dict:
    remote = _ssh_from_json(ssh_json)
    try:
        if not force:
            busy = remote.exec_out("pgrep -f train_taili | grep -v pgrep | head -1")
            if busy.strip():
                raise SystemExit("training is ACTIVE on the box — physeval would contend/OOM. "
                                 "Pass --force or pause training first.")
        run_dir = _resolve_run(remote, run_id, data_root)
        if not run_dir:
            raise SystemExit(f"no run dir for {run_id!r}")
        meta = json.loads(remote.exec_out(f"cat {shlex.quote(run_dir + '/run.json')} 2>/dev/null") or "{}")
        payload = meta.get("payload_root", "")
        agent_yaml = f"{run_dir}/agent.skrl.yaml"
        ckpt = f"{run_dir}/checkpoints/{checkpoint}"
        if remote.exec_out(f"[ -f {shlex.quote(ckpt)} ] && echo yes || echo no").strip() != "yes":
            raise SystemExit(f"checkpoint not found: {ckpt}")
        print(f"run={run_dir}\npayload={payload}\ncheckpoint={ckpt}\nterrains={terrains}\n", flush=True)

        scorecards = []
        for terr in terrains:
            log = f"{run_dir}/physeval_{terr}.log"
            if terr == "push":
                # E-battery (spec E1/E2): lateral base-push recovery — a different runner from the
                # terrain physeval. Its "E1 PASS/FAIL ..." lines merge into the same §2 verdict.
                runner = (f"{shlex.quote(python)} -m taili_blind_runtime.physeval_blind_e "
                          f"--checkpoint {shlex.quote(ckpt)} --agent-yaml {shlex.quote(agent_yaml)} "
                          f"--push 1.0 --num_envs {int(num_envs)} --headless")
            else:
                runner = (f"{shlex.quote(python)} -m taili_blind_runtime.physeval_blind "
                          f"--checkpoint {shlex.quote(ckpt)} --agent-yaml {shlex.quote(agent_yaml)} "
                          f"--terrain {shlex.quote(terr)} --num_envs {int(num_envs)} --headless")
            cmd = (f"cd {shlex.quote(data_root)} && export PYTHONPATH={shlex.quote(payload)} && "
                   f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
                   f"{runner} 2>&1 | tee {shlex.quote(log)}")
            print(f"=== physeval {terr} (Isaac boot + eval, ~3-5 min) ===", flush=True)
            out, _ = remote.exec(f"bash -lc {shlex.quote(cmd)}", timeout=900)
            print("\n".join(out.splitlines()[-6:]), flush=True)
            scorecards.append(out)

        verdict = AGG.aggregate(scorecards)
        # per-gate detail (A1[fwd05] -> {ok, detail}) so callers (the orchestrator's score/analysis,
        # the console view) can reason per-gate, not just per-family. Matches datasource.get_acceptance.
        merged = AGG.merge_runs(AGG.parse_scorecard(t) for t in scorecards)
        verdict["gates"] = {k: {"ok": bool(v.get("ok")), "detail": str(v.get("detail", ""))}
                            for k, v in merged.items()}
        verdict["run"] = run_dir
        verdict["checkpoint"] = checkpoint
        _maybe_update_best_registry(remote, data_root, verdict)
        return verdict
    finally:
        remote.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", nargs="?", default="newest")
    ap.add_argument("--terrains", nargs="*", default=["flat"])
    ap.add_argument("--checkpoint", default="best_agent.pt")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--config", default="config/ssh.json")
    ap.add_argument("--python", default=DEFAULT_PY)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    verdict = run_acceptance(args.run, args.terrains, args.checkpoint, args.num_envs,
                             args.config, args.force, args.python, args.data_root)
    print("\n" + "=" * 70)
    print(AGG.render_verdict(verdict))
    print("gates seen:", ", ".join(verdict.get("gates_seen", [])) or "none")
    if args.out:
        Path(args.out).write_text(json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"report -> {args.out}")
    return 0 if verdict.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
