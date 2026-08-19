"""Serial, resource-safe experiment execution tests."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

import pytest

from autotuner.research.research_ledger import ExperimentPlan
from autotuner.research.research_supervisor import (
    BackendHandle,
    BackendStatus,
    CommandExperimentBackend,
    ResearchSupervisor,
    ResourceLeaseStore,
)


class FakeBackend:
    def __init__(self, evaluation):
        self.evaluation = evaluation
        self.polls = 0
        self.stopped = False

    def start(self, plan, workspace, environment):
        self.workspace = workspace
        self.environment = dict(environment)
        return BackendHandle("fake", 0.0)

    def poll(self, handle):
        self.polls += 1
        return BackendStatus("succeeded")

    def stop(self, handle, reason):
        self.stopped = True

    def evaluate(self, plan, workspace):
        return self.evaluation


def _plan(**updates):
    value = dict(
        id="experiment:test",
        problem_statement="test a candidate",
        baseline_ref="baseline:1",
        intervention_diff={"mechanism": "candidate"},
        unchanged_fields=["contract"],
        protected_capabilities=["flat", "stairs"],
        evaluation_plan={"suite": "independent"},
        success_condition="evaluator success",
        rollback_condition="protected regression",
        resource_budget={"resources": ["gpu:0"], "lease_ttl_s": 60},
        status="approved",
    )
    value.update(updates)
    return ExperimentPlan(**value)


def test_lease_blocks_second_experiment(tmp_path: Path):
    leases = ResourceLeaseStore(tmp_path / "leases")
    first = leases.acquire(["gpu:0"], owner="first", ttl_s=60)
    with pytest.raises(RuntimeError, match="already leased"):
        leases.acquire(["gpu:0"], owner="second", ttl_s=60)
    leases.release(first)
    second = leases.acquire(["gpu:0"], owner="second", ttl_s=60)
    leases.release(second)


def test_non_overlapping_leases_coexist_and_release_independently(tmp_path: Path):
    leases = ResourceLeaseStore(tmp_path / "leases")
    gpu = leases.acquire(["gpu:0"], owner="trainer", ttl_s=60)
    cpu = leases.acquire(["cpu:evaluator"], owner="evaluator", ttl_s=60)
    leases.release(gpu)
    with pytest.raises(RuntimeError, match="already leased"):
        leases.acquire(["cpu:evaluator"], owner="other", ttl_s=60)
    leases.release(cpu)
    assert not leases.path.exists()


def test_command_backend_streams_large_output_to_workspace_log(tmp_path: Path):
    backend = CommandExperimentBackend()
    plan = _plan(training_window={
        "command": [sys.executable, "-c", "print('x' * 250000)"],
        "max_seconds": 10,
    })
    environment = dict(os.environ)
    environment["TAILI_MECHANISM_BUNDLE"] = str(tmp_path / "candidate" / "mechanisms.json")
    handle = backend.start(plan, tmp_path, environment)
    deadline = time.time() + 10
    status = backend.poll(handle)
    while status.state == "running" and time.time() < deadline:
        time.sleep(0.02)
        status = backend.poll(handle)
    assert status.state == "succeeded"
    assert (tmp_path / "training.log").stat().st_size > 200000

    evaluator = _plan(evaluation_plan={
        "command": [sys.executable, "-c", "import json,os; print(json.dumps({'success': os.getenv('TAILI_MECHANISM_BUNDLE') is not None}))"],
    })
    assert backend.evaluate(evaluator, tmp_path)["success"] is True


def test_success_promotes_only_after_all_protected_capabilities_pass(tmp_path: Path):
    artifact = tmp_path / "candidate"
    artifact.mkdir()
    (artifact / "mechanisms.json").write_text("{}", encoding="utf-8")
    backend = FakeBackend({
        "success": True, "evidence_complete": True,
        "protected": {"flat": {"passed": True}, "stairs": {"passed": True}},
    })
    supervisor = ResearchSupervisor(tmp_path / "research", backend=backend)
    result = supervisor.execute(_plan(), artifact, actor="test", poll_interval_s=0)
    assert result.disposition == "promote"
    assert (tmp_path / "research" / "ACTIVE_MECHANISM.json").is_file()
    assert result.workspace != str(artifact)


def test_protected_regression_rolls_back_and_keeps_candidate_for_review(tmp_path: Path):
    artifact = tmp_path / "candidate"
    artifact.mkdir()
    marker = artifact / "checkpoint.pt"
    marker.write_bytes(b"valuable")
    (artifact / "mechanisms.json").write_text("{}", encoding="utf-8")
    backend = FakeBackend({
        "success": True,
        "protected": {"flat": {"passed": True}, "stairs": {"passed": False}},
    })
    supervisor = ResearchSupervisor(tmp_path / "research", backend=backend)
    result = supervisor.execute(_plan(), artifact, actor="test", poll_interval_s=0)
    assert result.disposition == "rollback"
    assert Path(result.workspace, "candidate", "checkpoint.pt").read_bytes() == b"valuable"
    assert not (tmp_path / "research" / "ACTIVE_MECHANISM.json").exists()
