"""Aggregate physeval_blind SCORECARD outputs → the taili_spec §2 benchmark verdict.

physeval_blind prints, per run (flat / each terrain / push / DR), lines of the form
    <gate_key>  PASS|FAIL  <detail>
(e.g. "A1[fwd05] PASS ...", "F2 FAIL ...", "D[stairs] PASS ..."). The full acceptance is several
such runs over different terrains/conditions. This parses each run's scorecard, MERGES the gate
results (a gate must pass in EVERY run that produced it), and applies acceptance_score.final_verdict
(§2: all hard families present + ok; missing = not passed). Pure text + logic; no sim, no GPU.

This is the missing link between the raw physeval runs and the single "does this policy pass the
benchmark?" answer.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List

from autotuner.blind_locomotion import acceptance_score as ACC

# matches a scorecard line: leading gate key, PASS/FAIL, then detail
_LINE = re.compile(r"^\s*([A-Z]\d?(?:\[[^\]]+\])?)\s+(PASS|FAIL)\s+(.*?)\s*$")
# physeval_blind header: "BLIND ACCEPTANCE  ckpt=best_agent.pt  terrain=flat@4  num_envs=64 ..."
_PROV = re.compile(r"ckpt=(\S+)\s+terrain=(\S+)\s+num_envs=(\d+)")


# physeval_blind battery row: "fwd05 | 0.393m | +0.00/+0.00/+0.00 (+0.5/+0.0/+0.0) | ..."
_BATTERY = re.compile(
    r"^\s*(\w+)\s*\|\s*[\d.]+m?\s*\|\s*([+\-]?[\d.]+)/([+\-]?[\d.]+)/([+\-]?[\d.]+)\s*"
    r"\(([+\-]?[\d.]+)/([+\-]?[\d.]+)/([+\-]?[\d.]+)\)")


def parse_battery(log_text: str) -> list:
    """Extract the per-command battery: [{label, cmd:(vx,vy,wz), actual:(vx,vy,wz)}]."""
    rows = []
    for line in log_text.splitlines():
        m = _BATTERY.match(line)
        if m:
            rows.append({"label": m.group(1),
                         "actual": (float(m.group(2)), float(m.group(3)), float(m.group(4))),
                         "cmd": (float(m.group(5)), float(m.group(6)), float(m.group(7)))})
    return rows


def classify_behavior(rows: list) -> str:
    """Diagnose the deployment failure MODE from the move-command rows — the key tuning signal:
    STANDS (vx~0 for all move cmds → tracking too weak) vs CREEPS (~const speed regardless of cmd →
    tracking gradient-starved) vs TRACKS (actual varies with cmd)."""
    import math
    move = [r for r in rows if max(abs(c) for c in r["cmd"][:2]) > 0.1]   # linear move commands
    if not move:
        return "no_move_commands_in_battery"
    sp = [math.hypot(r["actual"][0], r["actual"][1]) for r in move]
    err = [math.hypot(r["actual"][0] - r["cmd"][0], r["actual"][1] - r["cmd"][1]) for r in move]
    if all(s < 0.10 for s in sp):
        return "STANDS — vx~0 for all move commands (tracking reward too weak vs standing)"
    if (max(sp) - min(sp)) < 0.12:
        return "CREEPS — ~constant speed regardless of command (tracking gradient-starved)"
    if max(err) < 0.12:
        return "TRACKS — actual matches commanded speed"
    return "PARTIAL — actual varies with command but with error (tracking imprecise)"


def parse_provenance(log_text: str) -> dict:
    """Pull {checkpoint, terrain, num_envs} from a physeval_blind run header (provenance for audit)."""
    m = _PROV.search(log_text)
    if not m:
        return {"checkpoint": "", "terrain": "", "num_envs": 0}
    return {"checkpoint": m.group(1), "terrain": m.group(2), "num_envs": int(m.group(3))}


def parse_scorecard(log_text: str) -> Dict[str, dict]:
    """Extract {gate_key: {ok, detail}} from one physeval_blind run's output."""
    out: Dict[str, dict] = {}
    for line in log_text.splitlines():
        m = _LINE.match(line)
        if not m:
            continue
        key, verdict, detail = m.group(1), m.group(2), m.group(3)
        out[key] = {"ok": verdict == "PASS", "detail": detail}
    return out


def merge_runs(run_results: Iterable[Dict[str, dict]]) -> Dict[str, dict]:
    """Merge gate results across runs. A gate present in multiple runs passes only if it passes in
    ALL of them (AND semantics — e.g. F2 must hold on flat AND every terrain)."""
    merged: Dict[str, dict] = {}
    for res in run_results:
        for key, val in res.items():
            if key not in merged:
                merged[key] = dict(val)
            elif not val["ok"]:                       # any failing occurrence fails the gate
                merged[key] = {"ok": False, "detail": val["detail"] + " (failed in one run)"}
    return merged


def aggregate(log_texts: Iterable[str]) -> dict:
    """Parse + merge several physeval_blind scorecards → taili_spec §2 final verdict."""
    merged = merge_runs(parse_scorecard(t) for t in log_texts)
    verdict = ACC.final_verdict(merged)
    verdict["gates_seen"] = sorted(merged)
    return verdict


def aggregate_files(paths: Iterable[str]) -> dict:
    """Read several physeval_blind log files → taili_spec §2 verdict. Missing/unreadable files are
    skipped with a note in the result (honest — a skipped terrain run = its gates not evaluated)."""
    from pathlib import Path
    texts, skipped = [], []
    for p in paths:
        try:
            texts.append(Path(p).read_text(encoding="utf-8", errors="replace"))
        except OSError as e:
            skipped.append(f"{p}: {e}")
    v = aggregate(texts)
    v["runs_read"] = len(texts)
    v["runs_skipped"] = skipped
    v["runs"] = [{"path": str(p), **parse_provenance(t),
                  "behavior": classify_behavior(parse_battery(t))}
                 for p, t in zip(paths, texts)]
    return v


def write_report(verdict: dict, path: str, stamp: str = "") -> str:
    """Persist the §2 verdict + per-run provenance as an auditable acceptance report (§12)."""
    import json
    from pathlib import Path
    report = {"stamp": stamp, "passed": verdict["passed"], "n_present": verdict["n_present"],
              "n_hard": verdict["n_hard"], "failed": verdict["failed"], "missing": verdict["missing"],
              "runs": verdict.get("runs", []), "families": verdict["families"], "soft": verdict["soft"]}
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(p)


def render_verdict(v: dict) -> str:
    head = "BENCHMARK: PASS ✅" if v["passed"] else "BENCHMARK: FAIL ✗"
    lines = [f"{head}  ({v['n_present']}/{v['n_hard']} hard families present)"]
    if v["missing"]:
        lines.append(f"  not evaluated (treated as fail): {', '.join(v['missing'])}")
    if v["failed"]:
        lines.append(f"  FAILED: {', '.join(v['failed'])}")
    soft_fail = [s for s, st in v["soft"].items() if st["present"] and not st["ok"]]
    if soft_fail:
        lines.append(f"  soft (report-only) below target: {', '.join(soft_fail)}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Aggregate physeval_blind log file(s) → §2 verdict. Usage:
    #   python -m autotuner.blind_locomotion.acceptance_aggregate <physeval_log>... [--out report.json]
    import sys
    args = sys.argv[1:]
    out_path = None
    if "--out" in args:
        i = args.index("--out")
        out_path = args[i + 1] if i + 1 < len(args) else None
        args = args[:i] + args[i + 2:]
    if not args:
        print("usage: python -m autotuner.blind_locomotion.acceptance_aggregate <physeval_log>... [--out report.json]")
        sys.exit(2)
    v = aggregate_files(args)
    print(render_verdict(v))
    print(f"\n(read {v['runs_read']} run(s); gates seen: {', '.join(v['gates_seen']) or 'none'})")
    for r in v["runs"]:
        print(f"  run {r['path']}: ckpt={r['checkpoint']} terrain={r['terrain']}")
    for s in v["runs_skipped"]:
        print(f"  ! skipped {s}")
    if out_path:
        print(f"\nreport → {write_report(v, out_path)}")
