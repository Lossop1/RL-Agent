"""Run the fixed serial nominal MuJoCo acceptance suite."""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from autotuner.control.manifest import build_run_manifest, sha256_file, write_manifest
from products.taili.traditional_control.evaluation import (
    EpisodeResult,
    run_mujoco_episode,
    write_episode_result,
)
from products.taili.traditional_control.packaging import (
    RUNTIME_DEPENDENCIES,
    build_bundle,
    project_root,
)


ACCEPTANCE_SCHEMA = "taili_traditional_control_acceptance_v1"
ACCEPTANCE_SCENARIOS = (
    "flat",
    "flat_forward",
    "flat_backward",
    "flat_left",
    "flat_right",
    "flat_yaw_left",
    "flat_yaw_right",
    "stairs_up",
    "stairs_down",
)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def run_acceptance(
    *,
    output_root: str | Path | None = None,
    config_path: str | Path | None = None,
    trace: bool = False,
) -> dict[str, object]:
    """Run every acceptance scenario serially and persist one auditable verdict."""

    root = project_root().resolve()
    output = Path(output_root or root / "output" / "traditional_control").resolve()
    output.relative_to(root)
    archive_root = output / "archive"
    bundle = build_bundle(archive_root)
    timestamp = datetime.now(timezone.utc)
    run_name = f"{timestamp.strftime('%Y%m%dT%H%M%S.%fZ')}_{str(bundle['artifact_digest'])[:12]}"
    run_dir = output / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    results: list[EpisodeResult] = []
    for scenario in ACCEPTANCE_SCENARIOS:
        result_path = run_dir / f"{scenario}.json"
        trace_path = run_dir / f"{scenario}.jsonl" if trace else None
        result = run_mujoco_episode(
            scenario,
            config_path=config_path,
            trace_path=trace_path,
        )
        write_episode_result(result_path, result)
        results.append(result)

    summary = {
        "schema_version": ACCEPTANCE_SCHEMA,
        "passed": all(result.passed for result in results),
        "artifact_digest": str(bundle["artifact_digest"]),
        "scenarios": [result.as_dict() for result in results],
    }
    results_path = run_dir / "results.json"
    _write_json(results_path, summary)

    run_manifest = build_run_manifest(
        root,
        bundle,
        archive_path=bundle["archive"],
        archive_sha256=str(bundle["archive_sha256"]),
        operation="acceptance",
        dependencies=RUNTIME_DEPENDENCIES,
        generated_at=timestamp,
        metadata={
            "acceptance_schema": ACCEPTANCE_SCHEMA,
            "run_directory": run_dir.relative_to(root).as_posix(),
            "results": results_path.relative_to(root).as_posix(),
            "results_sha256": sha256_file(results_path),
            "scenarios": list(ACCEPTANCE_SCENARIOS),
            "passed": bool(summary["passed"]),
            "trace": bool(trace),
        },
    )
    manifest_path = run_dir / "run_manifest.json"
    write_manifest(manifest_path, run_manifest)
    summary["run_digest"] = run_manifest["run_digest"]
    summary_path = run_dir / "summary.json"
    summary["run_manifest"] = manifest_path.relative_to(root).as_posix()
    _write_json(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    summary = run_acceptance(
        output_root=args.output_root,
        config_path=args.config,
        trace=args.trace,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if bool(summary["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
