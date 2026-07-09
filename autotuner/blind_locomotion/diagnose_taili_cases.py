"""Process-isolated Taili diagnostic case runner.

IsaacLab/Omniverse keeps enough global simulator/config state that repeatedly
constructing Taili envs inside one Python process can fail on the second
``gym.make``.  The single-case runner in ``diagnose_taili.py`` is therefore kept
as the unit of execution; this module splits a suite into one terrain/DR case
per child process, then merges the records back into the stage-level artifacts
that the console already consumes.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Taili diagnostics with process isolation per terrain/DR case.")
    p.add_argument("--task", default="RobotLab-Isaac-Taili-AMP-Blind-Direct-v0")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--agent-yaml", default="")
    return p


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"suite root must be a mapping: {path}")
    return data


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _progress(out_dir: Path, *, status: str, stage: str, rows_written: int = 0, **extra: Any) -> None:
    payload = {
        "status": status,
        "stage": stage,
        "rows_written": int(rows_written),
        "updated_at": _now_iso(),
    }
    payload.update(extra)
    _write_json(out_dir / "record_progress.json", payload)
    print(f"[TAILI_DIAG_CASES] stage={stage} status={status} rows={rows_written}", flush=True)


def _safe_name(value: Any) -> str:
    text = str(value or "case").strip().lower()
    keep = []
    for ch in text:
        if ch.isalnum():
            keep.append(ch)
        elif ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("._-") or "case"


def _case_error(case_dir: Path) -> dict[str, Any]:
    path = case_dir / "record_error.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"error_type": type(exc).__name__, "error": f"record_error.json could not be parsed: {exc}"}
    return data if isinstance(data, dict) else {"error": str(data)}


def _summarize_child_error(case_dir: Path, returncode: int) -> str:
    data = _case_error(case_dir)
    error_type = str(data.get("error_type") or "").strip()
    error = str(data.get("error") or "").strip()
    if error_type or error:
        return f"{error_type + ': ' if error_type else ''}{error}".strip()
    return f"child diagnostic exited with code {returncode}"


def _build_child_command(args: argparse.Namespace, suite_path: Path, out_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "taili_blind_runtime.diagnose_taili",
        "--task",
        args.task,
        "--checkpoint",
        args.checkpoint,
        "--suite",
        str(suite_path),
        "--out",
        str(out_dir),
        "--device",
        args.device,
    ]
    if args.headless:
        cmd.append("--headless")
    if args.num_envs is not None:
        cmd.extend(["--num-envs", str(args.num_envs)])
    if args.agent_yaml:
        cmd.extend(["--agent-yaml", args.agent_yaml])
    return cmd


def _merge_records(case_dirs: list[Path], out_dir: Path) -> int:
    record_path = out_dir / "record.csv"
    fieldnames: list[str] | None = None
    rows_written = 0
    with record_path.open("w", newline="", encoding="utf-8") as out_handle:
        writer: csv.DictWriter[str] | None = None
        for global_case_id, case_dir in enumerate(case_dirs):
            case_record = case_dir / "record.csv"
            if not case_record.exists() or case_record.stat().st_size <= 0:
                raise RuntimeError(f"case record is missing or empty: {case_record}")
            with case_record.open("r", newline="", encoding="utf-8") as in_handle:
                reader = csv.DictReader(in_handle)
                if not reader.fieldnames:
                    raise RuntimeError(f"case record has no header: {case_record}")
                if fieldnames is None:
                    fieldnames = list(reader.fieldnames)
                    writer = csv.DictWriter(out_handle, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                elif list(reader.fieldnames) != fieldnames:
                    raise RuntimeError(f"case record schema differs from first case: {case_record}")
                assert writer is not None
                for row in reader:
                    row["case_id"] = str(global_case_id)
                    writer.writerow(row)
                    rows_written += 1
    return rows_written


def _load_case_meta(case_dir: Path) -> dict[str, Any]:
    path = case_dir / "record_meta.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_merged_meta(
    *,
    args: argparse.Namespace,
    suite: dict[str, Any],
    out_dir: Path,
    case_dirs: list[Path],
    case_specs: list[dict[str, Any]],
    rows_written: int,
    status: str,
) -> None:
    executed_cases: list[dict[str, Any]] = []
    for case_id, (case_dir, spec) in enumerate(zip(case_dirs, case_specs)):
        child_meta = _load_case_meta(case_dir)
        child_executed = child_meta.get("executed_runtime_config", {}).get("cases", [])
        executed_cases.append(
            {
                "case_id": case_id,
                "status": "executed",
                "runtime": "taili_blind_runtime.process_isolated",
                "case_dir": str(case_dir),
                "terrain_requested": spec.get("terrain", {}),
                "dr_requested": spec.get("dr_case", {}),
                "child_executed_runtime_config": child_executed,
            }
        )
    payload = {
        "schema_version": "ilqd_observation_record_v0.5.1",
        "task": args.task,
        "checkpoint": args.checkpoint,
        "suite_path": args.suite,
        "requested_suite_config": suite,
        "executed_runtime_config": {
            "process_isolated_cases": True,
            "cases": executed_cases,
            "record_capture": "post_step_terminal_safe",
            "reset_initialization": suite.get("reset_initialization", "command_start"),
            "payload_local": True,
        },
        "num_envs_requested": int(args.num_envs or suite.get("num_envs") or 1),
        "rows_written": int(rows_written),
        "status": status,
        "recording_notes": [
            "payload-local Taili diagnostic; no robot_lab imports",
            "one terrain/DR case per child process to avoid IsaacLab global-state reuse",
            "policy action uses mean_actions when available",
        ],
        "semantics": "Observation-only. No pass/fail labels are emitted by the recorder.",
    }
    _write_json(out_dir / "record_meta.json", payload)


def _enrich_metrics(metrics_path: Path, suite: dict[str, Any]) -> None:
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"[TAILI_DIAG_CASES] metrics enrichment skipped: {type(exc).__name__}: {exc}", flush=True)
        return
    if not isinstance(payload, dict):
        return
    coverage = payload.setdefault("coverage", {})
    if not isinstance(coverage, dict):
        coverage = {}
        payload["coverage"] = coverage
    coverage["terrain_types_requested"] = sorted(
        {
            str(item.get("type"))
            for item in suite.get("terrains", [])
            if isinstance(item, dict) and item.get("type") is not None
        }
    )
    coverage["dr_levels_requested"] = sorted(
        {
            str(item.get("level"))
            for item in suite.get("dr_cases", [])
            if isinstance(item, dict) and item.get("level") is not None
        }
    )
    payload["suite_plan"] = {
        "name": suite.get("name", ""),
        "commands": suite.get("commands", []),
        "terrains": suite.get("terrains", []),
        "dr_cases": suite.get("dr_cases", []),
        "pushes": suite.get("pushes", {}),
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_cases(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suite = _load_yaml(args.suite)
    terrains = [item for item in (suite.get("terrains") or []) if isinstance(item, dict)] or [{"type": "flat", "level": 0}]
    dr_cases = [item for item in (suite.get("dr_cases") or []) if isinstance(item, dict)] or [{"level": 0}]
    total_cases = max(1, len(terrains) * len(dr_cases))
    _progress(out_dir, status="running", stage="split_cases", requested_cases=total_cases, completed_cases=0)

    case_root = out_dir / "cases"
    suite_root = out_dir / "case_suites"
    case_dirs: list[Path] = []
    case_specs: list[dict[str, Any]] = []

    case_id = 0
    for terrain in terrains:
        for dr_case in dr_cases:
            case_name = f"case_{case_id:03d}_{_safe_name(terrain.get('type', 'flat'))}_dr{_safe_name(dr_case.get('level', 0))}"
            case_dir = case_root / case_name
            case_suite_path = suite_root / f"{case_name}.yaml"
            case_suite = dict(suite)
            case_suite["name"] = f"{suite.get('name') or 'diagnostic'} / {case_name}"
            case_suite["terrains"] = [terrain]
            case_suite["dr_cases"] = [dr_case]
            _write_yaml(case_suite_path, case_suite)
            case_dirs.append(case_dir)
            case_specs.append({"terrain": terrain, "dr_case": dr_case, "suite": str(case_suite_path)})
            case_id += 1

    rows_so_far = 0
    completed = 0
    for case_id, (case_dir, spec) in enumerate(zip(case_dirs, case_specs)):
        case_dir.mkdir(parents=True, exist_ok=True)
        terrain = spec["terrain"]
        dr_case = spec["dr_case"]
        _progress(
            out_dir,
            status="running",
            stage="case_running",
            rows_written=rows_so_far,
            requested_cases=total_cases,
            completed_cases=completed,
            active_case=case_id,
            active_case_dir=str(case_dir),
            active_terrain=str(terrain.get("type", "flat")),
            active_dr_level=str(dr_case.get("level", 0)),
        )
        cmd = _build_child_command(args, Path(spec["suite"]), case_dir)
        print(f"[TAILI_DIAG_CASES] start case={case_id} terrain={terrain} dr={dr_case}", flush=True)
        print("[TAILI_DIAG_CASES] command=" + " ".join(cmd), flush=True)
        result = subprocess.run(cmd, cwd=Path.cwd(), env=os.environ.copy(), check=False)
        child_error = _case_error(case_dir)
        if result.returncode != 0 or child_error:
            message = _summarize_child_error(case_dir, result.returncode)
            payload = {
                "status": "error",
                "stage": "case_runner",
                "case_id": case_id,
                "case_dir": str(case_dir),
                "terrain": terrain,
                "dr_case": dr_case,
                "error_type": "ChildDiagnosticError",
                "error": message,
                "child_returncode": result.returncode,
                "child_error": child_error,
                "updated_at": _now_iso(),
            }
            _write_json(out_dir / "record_error.json", payload)
            _write_merged_meta(
                args=args,
                suite=suite,
                out_dir=out_dir,
                case_dirs=case_dirs[:case_id],
                case_specs=case_specs[:case_id],
                rows_written=rows_so_far,
                status="failed",
            )
            _progress(
                out_dir,
                status="error",
                stage="case_error",
                rows_written=rows_so_far,
                requested_cases=total_cases,
                completed_cases=completed,
                active_case=case_id,
                error=message,
            )
            raise RuntimeError(message)
        try:
            rows_so_far += max(0, sum(1 for _ in (case_dir / "record.csv").open("r", encoding="utf-8")) - 1)
        except Exception:
            pass
        completed += 1
        _progress(
            out_dir,
            status="running",
            stage="case_complete",
            rows_written=rows_so_far,
            requested_cases=total_cases,
            completed_cases=completed,
            active_case=case_id,
        )

    rows_written = _merge_records(case_dirs, out_dir)
    _write_merged_meta(
        args=args,
        suite=suite,
        out_dir=out_dir,
        case_dirs=case_dirs,
        case_specs=case_specs,
        rows_written=rows_written,
        status="complete",
    )

    try:
        from .isaaclab_quad_diag.metrics import compute_all_metrics
    except ImportError:
        from isaaclab_quad_diag.metrics import compute_all_metrics

    compute_all_metrics(out_dir / "record.csv", out_dir / "metrics", out_dir / "record_meta.json")
    _enrich_metrics(out_dir / "metrics" / "metrics.json", suite)
    _progress(
        out_dir,
        status="complete",
        stage="complete",
        rows_written=rows_written,
        requested_cases=total_cases,
        completed_cases=total_cases,
    )
    print(f"[TAILI_DIAG_CASES] merged record written to: {out_dir / 'record.csv'} rows={rows_written}", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_cases(args)
    except BaseException as exc:  # noqa: BLE001
        error_path = out_dir / "record_error.json"
        if not error_path.exists():
            _write_json(
                error_path,
                {
                    "status": "error",
                    "stage": "case_runner",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "updated_at": _now_iso(),
                },
            )
        _progress(out_dir, status="error", stage="error", error_type=type(exc).__name__, error=str(exc))
        print(f"[TAILI_DIAG_CASES_ERROR] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
