"""Pure tests for the RL Agent research-audit vertical slice."""
from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path

from autotuner.locomotion_console import agent
from autotuner.locomotion_console.research_audit import (
    _validate_frozen_reference,
    build_checkpoint_registry,
    build_command_coverage,
    build_optimization_state,
    build_research_audit,
)


def test_local_audit_is_conservative_and_derives_reward_structure():
    report = build_research_audit()

    assert report.schema_version == "rl-agent.research-audit/v1"
    assert report.status == "incomplete"
    assert any(item.term == "tracking_lin" for item in report.reward_terms)
    assert report.runtime_execution.symbols["reward.compute_reward_components"] is True
    assert report.summary["open_errors"] > 0
    assert any(item.id == "optimization.state_missing" for item in report.findings)


def test_command_coverage_requires_all_regimes():
    ledger, findings = build_command_coverage({
        "command_coverage": [
            {"bucket": "forward", "target_count": 10, "applied_count": 10, "eligible_count": 9},
        ]
    })

    assert ledger.status == "missing"
    assert set(ledger.missing_buckets) == {"backward", "lateral", "yaw", "stand", "mixed"}
    assert any(item.id == "commands.coverage_missing" for item in findings)


def test_command_coverage_rejects_named_buckets_without_eligible_samples():
    ledger, findings = build_command_coverage({
        "command_coverage": [
            {
                "bucket": bucket,
                "target_count": 0 if bucket == "mixed" else 1,
                "applied_count": 0 if bucket == "mixed" else 1,
                "eligible_count": 0 if bucket == "mixed" else 1,
            }
            for bucket in ("forward", "backward", "lateral", "yaw", "stand", "mixed")
        ]
    })

    assert ledger.status == "missing"
    assert ledger.missing_buckets == ["mixed"]
    assert any(item.id == "commands.coverage_missing" for item in findings)


def test_optimization_state_only_becomes_proven_with_parity():
    raw = {key: f"{key}-hash" for key in ("policy", "value", "optimizer", "scheduler", "normalizer", "log_std", "amp", "curriculum", "rng")}
    raw["parity_check"] = False
    snapshot, findings = build_optimization_state({"optimization_state": raw})
    assert snapshot.status == "declared"
    assert any(item.id == "optimization.state_not_proven" for item in findings)
    raw["parity_check"] = True
    snapshot, findings = build_optimization_state({"optimization_state": raw})
    assert snapshot.status == "proven"
    assert not findings


def test_checkpoint_registry_does_not_invent_capabilities():
    # The execution environment denies pytest's default %TEMP% directory. Keep
    # the fixture local to the repo and remove it deterministically instead.
    tmp_path = Path.cwd() / f".research_audit_test_{uuid.uuid4().hex}"
    tmp_path.mkdir()
    try:
        (tmp_path / "agent_10.pt").write_bytes(b"one")
        (tmp_path / "agent_20.pt").write_bytes(b"two")
        registry = build_checkpoint_registry(tmp_path, {})

        assert registry.status == "declared"
        assert registry.records[0].role == "candidate"
        assert all(item.capability_status == "unknown" for item in registry.records)

        manifest = {"checkpoint_capabilities": [{
            "checkpoint": "agent_20.pt",
            "capabilities": {"flat": "strong"},
            "evidence": ["diag:20"],
        }]}
        registry = build_checkpoint_registry(tmp_path, manifest)
        assert registry.status == "proven"
        assert any(item.capability_status == "proven" for item in registry.records)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_research_audit_tool_is_registered_and_documented():
    assert "get_research_audit" in agent.TOOLS
    assert "get_research_audit(manifest" in agent._TOOLS_DOC
    assert agent._intent_tool_hint("请做研究 Agent 架构缺口审计")["tool"] == "get_research_audit"


def test_frozen_reference_rehashes_explicit_curriculum_restore():
    root = Path.cwd() / f".research_audit_restore_test_{uuid.uuid4().hex}"
    root.mkdir()
    try:
        effective = root / "effective_config.yaml"
        effective.write_text("env: {}\n", encoding="utf-8")
        curriculum = root / "restored_curriculum_state.json"
        curriculum.write_text('{"version":4,"terrain_levels":[0]}\n', encoding="utf-8")
        curriculum_hash = hashlib.sha256(curriculum.read_bytes()).hexdigest()
        checkpoint_hash = "a" * 64
        restore = {
            "requested": True,
            "status": "proven",
            "artifact_ref": curriculum.name,
            "source": {"sha256": curriculum_hash},
            "artifact": {"sha256": curriculum_hash},
        }
        reference = root / "reference_policy.json"
        reference.write_text(json.dumps({
            "schema_version": "rl-agent.gate-calibration-rollout/v1",
            "status": "complete",
            "training_enabled": False,
            "policy_mode": "mean_action",
            "runtime_execution": {"status": "proven", "remote_verification": "proven"},
            "checkpoint": {"status": "captured", "sha256": checkpoint_hash},
            "curriculum_restore": restore,
        }), encoding="utf-8")
        scenario = root / "gate_scenario_contract.json"
        scenario.write_text(json.dumps({
            "schema_version": "rl-agent.gate-calibration-scenario/v1",
            "training_enabled": False,
            "policy_mode": "mean_action",
            "command_source": "environment_sampler",
            "num_envs": 1,
            "steps": 100,
            "phase_override": 2,
            "dr_level_override": 1,
            "resolved_curriculum": {"phase": 2, "max_training_phase": 2, "dr_level": 1},
            "checkpoint_sha256": checkpoint_hash,
            "effective_config_sha256": hashlib.sha256(effective.read_bytes()).hexdigest(),
            "curriculum_restore": restore,
        }), encoding="utf-8")

        assert _validate_frozen_reference(reference, scenario, effective) == []
        curriculum.write_text('{"version":4,"terrain_levels":[9]}\n', encoding="utf-8")
        assert "restored curriculum artifact hash mismatch" in _validate_frozen_reference(
            reference, scenario, effective
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)
