"""physeval_suite.py — one-shot acceptance sweep across ALL terrain families + the E battery.

taili_spec's final bar is "A1-A4, B1-B4, C, D1-D4, E1-E5, F2 all pass, measured mean-action
under the deployment protocol, on EVERY terrain family". physeval_blind measures one terrain
per process (terrain is baked at env creation), so this driver runs the battery sequentially
and merges the per-run JSON into a single spec-item scorecard.

Each sub-run pays the Isaac boot cost (~2-4 min); a full sweep is ~30-45 min on one GPU.
Never run while training occupies the GPU.

  python physeval_suite.py --checkpoint /path/agent_50000.pt --out suite_report.json
  python physeval_suite.py --checkpoint ... --terrains flat rough      # subset
  python physeval_suite.py --checkpoint ... --skip-e                   # flat/terrain only

Report layout (suite_report.json):
  {"checkpoint": ..., "runs": {"flat": {...}, "rough": {...}, ..., "push": {...}},
   "spec_items": {"A1": "PASS|FAIL|NOT-EVALUATED", ..., "F2": ...},
   "verdict": "PASS|FAIL", "not_evaluated": [...]}
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

TERRAINS = ["flat", "slope", "rough", "boxes", "stairs"]

# spec item -> (which run's report carries it, key in that report). Single-run items only.
# flat carries the A/B/C hard gates; E1/E2 come from the push run. A4 (mixed-command bucket) has no
# measurement yet and surfaces as NOT-EVALUATED — never silently passed.
# D and F2 are aggregated across MULTIPLE runs below (not here): D = every terrain run's D[terrain]
# must pass (spec "controlled progress on every terrain family"), F2 = the hardware torque gate must
# hold on EVERY run (flat AND terrain), matching acceptance_score.final_verdict's AND semantics.
SPEC_FROM_RUN = {
    "A1": ("flat", "A1"), "A2": ("flat", "A2"), "A3": ("flat", "A3"), "A4": ("flat", "A4"),
    "B1": ("flat", "B1"), "B2": ("flat", "B2"), "B3": ("flat", "B3"), "B4": ("flat", "B4"),
    "C":  ("flat", "C"),
    "E1": ("push", "E1"), "E2": ("push", "E2"),
}
# terrain runs whose D[terrain] result feeds the aggregate D family (all must pass)
D_TERRAIN_RUNS = ("slope", "rough", "boxes", "stairs")


def _run(cmd: list[str], log_path: Path, timeout_s: int) -> int:
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
        try:
            return proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            return -9


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="suite_report.json")
    ap.add_argument("--terrains", nargs="*", default=TERRAINS)
    ap.add_argument("--num_envs", type=int, default=64)
    ap.add_argument("--push", type=float, default=1.0)
    ap.add_argument("--skip-e", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--timeout", type=int, default=1800, help="per-sub-run timeout (s)")
    args = ap.parse_args()

    out = Path(args.out)
    workdir = out.parent / (out.stem + "_runs")
    workdir.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict] = {}

    def collect(tag: str, report_path: Path, rc: int):
        if rc == 0 and report_path.is_file():
            try:
                runs[tag] = json.loads(report_path.read_text(encoding="utf-8"))
                return
            except Exception as e:  # noqa: BLE001
                runs[tag] = {"error": f"report unreadable: {e}"}
        runs[tag] = {"error": f"sub-run failed rc={rc} (see {workdir / (tag + '.log')})"}

    for terrain in args.terrains:
        report = workdir / f"{terrain}.json"
        cmd = [args.python, str(here / "physeval_blind.py"),
               "--checkpoint", args.checkpoint, "--terrain", terrain,
               "--num_envs", str(args.num_envs), "--headless",
               "--report", str(report)]
        t0 = time.time()
        rc = _run(cmd, workdir / f"{terrain}.log", args.timeout)
        print(f"[suite] {terrain}: rc={rc} ({time.time()-t0:.0f}s)", flush=True)
        collect(terrain, report, rc)

    if not args.skip_e:
        report = workdir / "push.json"
        cmd = [args.python, str(here / "physeval_blind_e.py"),
               "--checkpoint", args.checkpoint, "--num_envs", str(args.num_envs),
               "--push", str(args.push), "--headless", "--report", str(report)]
        t0 = time.time()
        rc = _run(cmd, workdir / "push.log", args.timeout)
        print(f"[suite] push: rc={rc} ({time.time()-t0:.0f}s)", flush=True)
        collect("push", report, rc)

    # merge into the spec scorecard; anything unmeasured is NOT-EVALUATED, never silently passed
    def _item(tag: str, key: str):
        run = runs.get(tag) or {}
        v = (run.get("items") or {}).get(key) if isinstance(run.get("items"), dict) else run.get(key)
        if isinstance(v, bool):
            v = "PASS" if v else "FAIL"
        return v if v in ("PASS", "FAIL") else None

    def _aggregate(results: list) -> str:
        # AND semantics across runs (matches acceptance_score.final_verdict / merge_runs): every
        # measured occurrence must PASS; if none was measured it is NOT-EVALUATED (never silently passed).
        seen = [r for r in results if r in ("PASS", "FAIL")]
        if not seen:
            return "NOT-EVALUATED"
        return "PASS" if all(r == "PASS" for r in seen) else "FAIL"

    spec_items: dict[str, str] = {}
    for item, (tag, key) in SPEC_FROM_RUN.items():
        spec_items[item] = _item(tag, key) or "NOT-EVALUATED"
    # D = controlled progress on EVERY terrain family (slope/rough/boxes/stairs); all must pass.
    spec_items["D"] = _aggregate([_item(terr, f"D[{terr}]") for terr in D_TERRAIN_RUNS])
    # F2 = the hardware torque gate must hold on EVERY run (flat AND every terrain), not flat only.
    spec_items["F2"] = _aggregate([_item(tag, "F2") for tag in list(args.terrains)])
    # E3-E5 need the full-DR eval battery (not built yet) — stated, not skipped silently
    for item in ("E3", "E4", "E5"):
        spec_items[item] = "NOT-EVALUATED"

    not_eval = sorted(k for k, v in spec_items.items() if v == "NOT-EVALUATED")
    verdict = "FAIL" if any(v == "FAIL" for v in spec_items.values()) else (
        "PASS" if not not_eval else "INCOMPLETE")
    payload = {"checkpoint": args.checkpoint, "runs": runs,
               "spec_items": spec_items, "verdict": verdict, "not_evaluated": not_eval}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[suite] verdict={verdict} not_evaluated={not_eval}")
    print(f"[suite] report -> {out}")
    return 0 if verdict != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
