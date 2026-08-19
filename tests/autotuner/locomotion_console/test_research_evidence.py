"""Pure tests for converting persisted telemetry and scorecards to evidence."""
from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path

import yaml

from autotuner.locomotion_console.research_audit import build_gate_calibration
from autotuner.locomotion_console.research_evidence import build_gate_calibration_evidence
from autotuner.locomotion_console.research_evidence import assemble_manifest
from autotuner.locomotion_console.research_audit import _DEFAULT_ALIGNMENT_SPECS


def _write_alignment_sidecar(root: Path) -> Path:
    artifacts = {
        "raw_trace": root / "raw_trace.jsonl",
        "qualification_mask": root / "qualification_mask.json",
        "scenario_contract": root / "scenario_contract.json",
    }
    artifacts["raw_trace"].write_text('{"sample": 1}\n', encoding="utf-8")
    artifacts["qualification_mask"].write_text('{"eligible": [true]}\n', encoding="utf-8")
    artifacts["scenario_contract"].write_text('{"scenario": "test"}\n', encoding="utf-8")
    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in artifacts.items()
    }
    rows = []
    for spec in _DEFAULT_ALIGNMENT_SPECS:
        rows.append({
            **spec,
            "unit": "test-unit",
            "threshold": {"operator": "<=", "value": 1.0},
            "raw_trace_ref": artifacts["raw_trace"].name,
            "qualification_mask_ref": artifacts["qualification_mask"].name,
            "scenario_contract_ref": artifacts["scenario_contract"].name,
            "sample_count": 1,
            "aggregation": {"method": "declared-test-statistic", "empty_samples": "fail"},
            "source_hashes": hashes,
        })
    sidecar = root / "metric_alignment.json"
    sidecar.write_text(json.dumps({"metric_alignment": rows}), encoding="utf-8")
    return sidecar


def test_evidence_assembler_separates_command_coverage_from_scorer_alignment():
    root = Path.cwd() / f".research_evidence_test_{uuid.uuid4().hex}"
    root.mkdir()
    try:
        commands = {
            "forward": (0.5, 0.0, 0.0),
            "backward": (-0.5, 0.0, 0.0),
            "lateral": (0.0, 0.3, 0.0),
            "yaw": (0.0, 0.0, 0.4),
            "stand": (0.0, 0.0, 0.0),
            "mixed": (0.4, 0.2, 0.2),
        }
        telemetry = [
            {
                "step": index,
                "command": {
                    "cmd_vx": values[0],
                    "cmd_vy": values[1],
                    "cmd_wz": values[2],
                    "actual_vx": values[0],
                    "actual_vy": values[1],
                    "actual_wz": values[2],
                },
                "health": {"stable_motion_gate": 1.0, "terminal_rate": 0.0},
            }
            for index, values in enumerate(commands.values())
        ]
        (root / "train.telemetry.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in telemetry), encoding="utf-8"
        )
        scorecard = root / "scorecard.log"
        scorecard.write_text(
            "\n".join(
                [
                    "A1[fwd] PASS ok",
                    "A2[yaw] PASS ok",
                    "A3 PASS ok",
                    "B1 PASS ok",
                    "B2 PASS ok",
                    "B3 PASS ok",
                    "B4[flat] PASS ok",
                    "C PASS ok",
                    "D[stairs] PASS ok",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        alignment = _write_alignment_sidecar(root)
        path = assemble_manifest(root, scorecards=[scorecard], metric_alignment=alignment)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        buckets = {item["bucket"] for item in manifest["command_coverage"]}
        assert buckets == set(commands)
        assert all(item["target_count"] == 1 for item in manifest["command_coverage"])
        aligned = {item["id"]: item["status"] for item in manifest["metric_alignment"]}
        assert all(value == "proven" for value in aligned.values())
        distribution = manifest["training_distribution"]
        assert distribution["command_buckets"]["realized"]["forward"]["sample_count"] == 1
        assert distribution["terrain_mix"]["realized_counts"] == {"unknown": len(commands)}
        assert distribution["realized_window"]["sample_count"] == len(commands)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_scorecard_only_captures_coverage_but_does_not_prove_metric_alignment():
    root = Path.cwd() / f".research_evidence_test_{uuid.uuid4().hex}"
    root.mkdir()
    try:
        (root / "train.telemetry.jsonl").write_text(
            json.dumps({"step": 1, "command": {"cmd_vx": 0.5, "cmd_vy": 0.0, "cmd_wz": 0.0}}) + "\n",
            encoding="utf-8",
        )
        scorecard = root / "scorecard.log"
        scorecard.write_text(
            "\n".join(("A1 PASS ok", "A2 PASS ok", "A3 PASS ok", "B1 PASS ok", "B2 PASS ok", "B3 PASS ok", "B4 PASS ok", "D[stairs] PASS ok")) + "\n",
            encoding="utf-8",
        )
        path = assemble_manifest(root, scorecards=[scorecard])
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert {item["status"] for item in manifest["metric_alignment"]} == {"captured"}
        assert all(item["scorecard_coverage"] for item in manifest["metric_alignment"])
        assert all(item["gaps"] for item in manifest["metric_alignment"])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_evidence_assembler_does_not_prove_metrics_without_scorecard():
    root = Path.cwd() / f".research_evidence_test_{uuid.uuid4().hex}"
    root.mkdir()
    try:
        (root / "train.telemetry.jsonl").write_text(
            json.dumps(
                {
                    "step": 1,
                    "command": {"cmd_vx": 0.5, "cmd_vy": 0.0, "cmd_wz": 0.0},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        path = assemble_manifest(root)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["command_coverage"][0]["bucket"] == "forward"
        assert "metric_alignment" not in manifest
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_evidence_assembler_uses_disjoint_batch_command_counters():
    root = Path.cwd() / f".research_evidence_test_{uuid.uuid4().hex}"
    root.mkdir()
    try:
        buckets = ("forward", "backward", "lateral", "yaw", "stand", "mixed")
        command = {
            "cmd_vx": 0.1,
            "cmd_vy": 0.0,
            "cmd_wz": 0.0,
            **{
                f"bucket_{bucket}_{suffix}_samples": 1.0
                for bucket in buckets
                for suffix in ("target", "applied", "eligible")
            },
        }
        (root / "train.telemetry.jsonl").write_text(
            json.dumps(
                {
                    "step": 10,
                    "command": command,
                    "health": {"stable_motion_gate": 0.5, "terminal_rate": 0.0},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        path = assemble_manifest(root)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        rows = {item["bucket"]: item for item in manifest["command_coverage"]}
        assert set(rows) == set(buckets)
        assert all(rows[bucket]["target_count"] == 1 for bucket in buckets)
        assert all(rows[bucket]["eligible_count"] == 1 for bucket in buckets)
        assert all(rows[bucket]["settled_count"] == 0 for bucket in buckets)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_evidence_assembler_accepts_explicit_per_bucket_settled_counts():
    root = Path.cwd() / f".research_evidence_test_{uuid.uuid4().hex}"
    root.mkdir()
    try:
        command = {
            "bucket_stand_target_samples": 4.0,
            "bucket_stand_applied_samples": 4.0,
            "bucket_stand_eligible_samples": 3.0,
            "bucket_stand_settled_samples": 2.0,
        }
        (root / "train.telemetry.jsonl").write_text(
            json.dumps({"step": 1, "command": command}) + "\n",
            encoding="utf-8",
        )
        path = assemble_manifest(root)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        row = manifest["command_coverage"][0]
        assert row["bucket"] == "stand"
        assert row["settled_count"] == 2
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_effective_gate_calibration_is_recomputed_from_raw_trace():
    root = Path.cwd() / f".research_evidence_test_{uuid.uuid4().hex}"
    root.mkdir()
    try:
        config = {
            "env": {
                "curriculum": {
                    "phase_progress_dirs_0": ["fwd"],
                    "phase_progress_thresholds_0": {"fwd": 0.6},
                    "phase_gate_prog_0": 0.6,
                    "phase_gate_execution_0": 0.8,
                },
                "domain_randomization": {},
            }
        }
        config_paths = {
            "source_config": root / "taili_blind_config.yaml",
            "effective_config": root / "effective_config.yaml",
            "agent_config": root / "agent.skrl.yaml",
        }
        config_paths["source_config"].write_text(yaml.safe_dump(config), encoding="utf-8")
        config_paths["effective_config"].write_text(yaml.safe_dump(config), encoding="utf-8")
        config_paths["agent_config"].write_text("trainer: {}\n", encoding="utf-8")
        runtime_manifest = {
            "configuration": {
                key: {
                    "path": f"/remote/{path.name}",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "exists": True,
                }
                for key, path in config_paths.items()
            }
        }
        (root / "runtime_manifest.json").write_text(json.dumps(runtime_manifest), encoding="utf-8")

        rows = []
        for step, progress, execution in ((10, 0.61, 0.81), (20, 0.65, 0.84), (30, 0.70, 0.87)):
            rows.append({
                "step": step,
                "curriculum": {
                    "phase": "phi0",
                    "dr_level": 0,
                    "active_dirs": "fwd",
                    "phase_gate_eval_step": step,
                    "progress_fwd": progress,
                    "phase_gate_progress_value": progress,
                    "phase_gate_execution_value": execution,
                },
            })
        (root / "train.telemetry.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8"
        )
        checkpoint_hash = "a" * 64
        reference = root / "reference_policy.json"
        reference.write_text(json.dumps({
            "schema_version": "rl-agent.gate-calibration-rollout/v1",
            "status": "complete",
            "training_enabled": False,
            "policy_mode": "mean_action",
            "runtime_execution": {"status": "proven", "remote_verification": "proven"},
            "checkpoint": {"status": "captured", "sha256": checkpoint_hash},
        }), encoding="utf-8")
        scenario = root / "scenario_contract.json"
        scenario.write_text(json.dumps({
            "schema_version": "rl-agent.gate-calibration-scenario/v1",
            "training_enabled": False,
            "policy_mode": "mean_action",
            "command_source": "environment_sampler",
            "num_envs": 64,
            "steps": 30,
            "checkpoint_sha256": checkpoint_hash,
            "effective_config_sha256": hashlib.sha256(config_paths["effective_config"].read_bytes()).hexdigest(),
        }), encoding="utf-8")

        sidecar = build_gate_calibration_evidence(
            root,
            reference_policy_manifest=reference,
            scenario_contract=scenario,
            minimum_samples=3,
        )
        assembled_path = assemble_manifest(root, gate_calibration=sidecar)
        manifest = json.loads(assembled_path.read_text(encoding="utf-8"))
        assert manifest["configuration_artifacts"]["effective_config"]["status"] == "proven"

        records, findings = build_gate_calibration(Path.cwd(), manifest)
        active = [item for item in records if item.active]
        assert {item.gate_id for item in active} == {
            "env.curriculum.phase_gate_execution_0",
            "env.curriculum.phase_gate_prog_0",
            "env.curriculum.phase_progress_thresholds_0.fwd",
        }
        assert all(item.status == "proven" for item in active)
        assert not findings

        raw = root / "gate_calibration.raw.jsonl"
        raw.write_text(raw.read_text(encoding="utf-8") + json.dumps({
            "sample_id": "tampered",
            "gate_id": "env.curriculum.phase_gate_prog_0",
            "eligible": True,
            "gate_eval_step": 40,
            "value": 999,
            "threshold": 0.6,
            "comparison": "gte",
        }) + "\n", encoding="utf-8")
        records, findings = build_gate_calibration(Path.cwd(), manifest)
        assert any(item.status != "proven" for item in records if item.active)
        assert any(item.id == "curriculum.gates_not_calibrated" for item in findings)
    finally:
        shutil.rmtree(root, ignore_errors=True)
