"""End-to-end integration tests for parallel research candidate execution.

Scenarios covered:
1. Submit 3 candidates and verify parallel execution
2. Simulate GPU resource constraints (only 2 GPUs available)
3. Simulate candidate crash and verify others continue
4. Verify experiment result collection and decision logic
5. Verify state persistence and recovery
"""
import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest


@pytest.fixture
def temp_state_dir():
    """Create a temporary directory for test state that works on Windows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# -- Mock structures for parallel research --

class MockGPUPool:
    """Simulates a GPU resource pool with limited capacity."""

    def __init__(self, capacity: int = 2):
        self.capacity = capacity
        self.allocated: Dict[str, int] = {}
        self.lock = asyncio.Lock()

    async def acquire(self, candidate_id: str) -> int:
        """Acquire a GPU slot, blocking if none available."""
        async with self.lock:
            while len(self.allocated) >= self.capacity:
                await asyncio.sleep(0.1)

            # Assign next available GPU index
            used = set(self.allocated.values())
            for idx in range(self.capacity):
                if idx not in used:
                    self.allocated[candidate_id] = idx
                    return idx
            raise RuntimeError("GPU pool exhausted")

    async def release(self, candidate_id: str):
        """Release GPU slot."""
        async with self.lock:
            if candidate_id in self.allocated:
                del self.allocated[candidate_id]

    def get_available_count(self) -> int:
        """Return number of available GPUs."""
        return self.capacity - len(self.allocated)


class MockCandidate:
    """Represents a research candidate (e.g., hyperparameter configuration)."""

    def __init__(
        self,
        candidate_id: str,
        config: Dict[str, Any],
        should_crash: bool = False,
        execution_time: float = 1.0,
    ):
        self.candidate_id = candidate_id
        self.config = config
        self.should_crash = should_crash
        self.execution_time = execution_time
        self.state = "pending"  # pending -> running -> completed / crashed
        self.result: Dict[str, Any] | None = None
        self.assigned_gpu: int | None = None
        self.start_time: float | None = None
        self.end_time: float | None = None


class ParallelResearchOrchestrator:
    """Orchestrates parallel execution of research candidates with GPU resource management."""

    def __init__(self, gpu_pool: MockGPUPool, state_dir: Path):
        self.gpu_pool = gpu_pool
        self.state_dir = state_dir
        self.candidates: Dict[str, MockCandidate] = {}
        self.results: Dict[str, Dict[str, Any]] = {}

    def submit_candidate(self, candidate: MockCandidate):
        """Submit a candidate for execution."""
        self.candidates[candidate.candidate_id] = candidate
        self._persist_state()

    async def execute_candidate(self, candidate: MockCandidate) -> Dict[str, Any]:
        """Execute a single candidate with GPU resource management."""
        try:
            # Acquire GPU
            candidate.state = "acquiring_gpu"
            self._persist_state()

            gpu_idx = await self.gpu_pool.acquire(candidate.candidate_id)
            candidate.assigned_gpu = gpu_idx
            candidate.state = "running"
            candidate.start_time = time.time()
            self._persist_state()

            # Simulate training
            await asyncio.sleep(candidate.execution_time)

            # Simulate crash
            if candidate.should_crash:
                candidate.state = "crashed"
                candidate.end_time = time.time()
                self._persist_state()
                raise RuntimeError(f"Candidate {candidate.candidate_id} crashed during execution")

            # Generate result
            result = {
                "candidate_id": candidate.candidate_id,
                "status": "completed",
                "config": candidate.config,
                "metrics": {
                    "reward_mean": 10.0 + hash(candidate.candidate_id) % 5,
                    "success_rate": 0.8 + (hash(candidate.candidate_id) % 20) / 100,
                },
                "execution_time_s": time.time() - candidate.start_time,
                "gpu_idx": gpu_idx,
            }

            candidate.result = result
            candidate.state = "completed"
            candidate.end_time = time.time()
            self._persist_state()

            return result

        except Exception as e:
            result = {
                "candidate_id": candidate.candidate_id,
                "status": "crashed",
                "error": str(e),
                "execution_time_s": (time.time() - candidate.start_time) if candidate.start_time else 0,
            }
            candidate.result = result
            if candidate.state != "crashed":
                candidate.state = "crashed"
                candidate.end_time = time.time()
            self._persist_state()
            return result

        finally:
            # Always release GPU
            await self.gpu_pool.release(candidate.candidate_id)

    async def run_all(self) -> Dict[str, Dict[str, Any]]:
        """Execute all submitted candidates in parallel."""
        tasks = [
            self.execute_candidate(candidate)
            for candidate in self.candidates.values()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results (including exceptions)
        for i, candidate_id in enumerate(self.candidates.keys()):
            result = results[i]
            if isinstance(result, Exception):
                self.results[candidate_id] = {
                    "candidate_id": candidate_id,
                    "status": "exception",
                    "error": str(result),
                }
            else:
                self.results[candidate_id] = result

        self._persist_state()
        return self.results

    def _persist_state(self):
        """Persist orchestrator state to disk."""
        self.state_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "candidates": {
                cid: {
                    "candidate_id": c.candidate_id,
                    "config": c.config,
                    "state": c.state,
                    "assigned_gpu": c.assigned_gpu,
                    "start_time": c.start_time,
                    "end_time": c.end_time,
                    "result": c.result,
                }
                for cid, c in self.candidates.items()
            },
            "results": self.results,
        }

        state_file = self.state_dir / "orchestrator_state.json"
        state_file.write_text(json.dumps(state, indent=2))

    @classmethod
    def restore_from_state(cls, gpu_pool: MockGPUPool, state_dir: Path) -> "ParallelResearchOrchestrator":
        """Restore orchestrator from persisted state."""
        orchestrator = cls(gpu_pool, state_dir)

        state_file = state_dir / "orchestrator_state.json"
        if not state_file.exists():
            return orchestrator

        state = json.loads(state_file.read_text())

        # Restore candidates
        for cid, c_data in state.get("candidates", {}).items():
            candidate = MockCandidate(
                candidate_id=c_data["candidate_id"],
                config=c_data["config"],
            )
            candidate.state = c_data["state"]
            candidate.assigned_gpu = c_data["assigned_gpu"]
            candidate.start_time = c_data["start_time"]
            candidate.end_time = c_data["end_time"]
            candidate.result = c_data["result"]
            orchestrator.candidates[cid] = candidate

        orchestrator.results = state.get("results", {})

        return orchestrator


# -- Test suite --

@pytest.mark.asyncio
async def test_parallel_execution_3_candidates(temp_state_dir):
    """Test submitting 3 candidates and verifying parallel execution."""
    gpu_pool = MockGPUPool(capacity=2)
    orchestrator = ParallelResearchOrchestrator(gpu_pool, temp_state_dir)

    # Submit 3 candidates
    candidates = [
        MockCandidate("cand_a", {"lr": 0.001}, execution_time=0.2),
        MockCandidate("cand_b", {"lr": 0.003}, execution_time=0.2),
        MockCandidate("cand_c", {"lr": 0.01}, execution_time=0.2),
    ]

    for candidate in candidates:
        orchestrator.submit_candidate(candidate)

    # Verify all submitted
    assert len(orchestrator.candidates) == 3

    # Execute all in parallel
    start_time = time.time()
    results = await orchestrator.run_all()
    elapsed = time.time() - start_time

    # Verify completion
    assert len(results) == 3
    assert all(r["status"] == "completed" for r in results.values())

    # Verify parallel execution: with 2 GPUs and 3 candidates (0.2s each),
    # total time should be ~0.4s (two batches), not 0.6s (sequential)
    assert elapsed < 0.5, f"Expected parallel execution ~0.4s, got {elapsed:.2f}s"

    # Verify GPU assignment
    gpu_assignments = [r["gpu_idx"] for r in results.values()]
    assert all(gpu_idx in {0, 1} for gpu_idx in gpu_assignments)

    # Verify state persisted
    state_file = temp_state_dir / "orchestrator_state.json"
    assert state_file.exists()
    state = json.loads(state_file.read_text())
    assert len(state["candidates"]) == 3
    assert all(c["state"] == "completed" for c in state["candidates"].values())


@pytest.mark.asyncio
async def test_gpu_resource_constraint(temp_state_dir):
    """Test GPU resource constraints with limited capacity."""
    gpu_pool = MockGPUPool(capacity=2)
    orchestrator = ParallelResearchOrchestrator(gpu_pool, temp_state_dir)

    # Submit 5 candidates with longer execution time
    candidates = [
        MockCandidate(f"cand_{i}", {"param": i}, execution_time=0.3)
        for i in range(5)
    ]

    for candidate in candidates:
        orchestrator.submit_candidate(candidate)

    # Execute
    start_time = time.time()
    results = await orchestrator.run_all()
    elapsed = time.time() - start_time

    # Verify all completed
    assert len(results) == 5
    assert all(r["status"] == "completed" for r in results.values())

    # With 2 GPUs and 5 candidates (0.3s each):
    # Batch 1: cand_0, cand_1 (0.3s)
    # Batch 2: cand_2, cand_3 (0.3s)
    # Batch 3: cand_4 (0.3s)
    # Total: ~0.9s
    assert 0.8 < elapsed < 1.2, f"Expected ~0.9s with 2 GPUs, got {elapsed:.2f}s"

    # Verify no more than 2 GPUs used at once
    for candidate in orchestrator.candidates.values():
        assert candidate.assigned_gpu in {0, 1, None}


@pytest.mark.asyncio
async def test_candidate_crash_others_continue(temp_state_dir):
    """Test that when one candidate crashes, others continue execution."""
    gpu_pool = MockGPUPool(capacity=2)
    orchestrator = ParallelResearchOrchestrator(gpu_pool, temp_state_dir)

    # Submit 3 candidates, middle one crashes
    candidates = [
        MockCandidate("cand_a", {"lr": 0.001}, execution_time=0.2),
        MockCandidate("cand_b", {"lr": 0.003}, should_crash=True, execution_time=0.15),
        MockCandidate("cand_c", {"lr": 0.01}, execution_time=0.2),
    ]

    for candidate in candidates:
        orchestrator.submit_candidate(candidate)

    # Execute
    results = await orchestrator.run_all()

    # Verify results
    assert len(results) == 3
    assert results["cand_a"]["status"] == "completed"
    assert results["cand_b"]["status"] == "crashed"
    assert "crashed during execution" in results["cand_b"]["error"]
    assert results["cand_c"]["status"] == "completed"

    # Verify crashed candidate state
    crashed = orchestrator.candidates["cand_b"]
    assert crashed.state == "crashed"
    assert crashed.result is not None
    assert crashed.end_time is not None

    # Verify successful candidates have metrics
    assert "metrics" in results["cand_a"]
    assert "metrics" in results["cand_c"]
    assert "reward_mean" in results["cand_a"]["metrics"]


@pytest.mark.asyncio
async def test_experiment_result_collection_and_decision(temp_state_dir):
    """Test result collection and decision logic for best candidate selection."""
    gpu_pool = MockGPUPool(capacity=2)
    orchestrator = ParallelResearchOrchestrator(gpu_pool, temp_state_dir)

    # Submit candidates with different performance
    candidates = [
        MockCandidate("cand_low", {"lr": 0.001}, execution_time=0.1),
        MockCandidate("cand_high", {"lr": 0.005}, execution_time=0.1),
        MockCandidate("cand_mid", {"lr": 0.003}, execution_time=0.1),
    ]

    for candidate in candidates:
        orchestrator.submit_candidate(candidate)

    # Execute
    results = await orchestrator.run_all()

    # All should complete successfully
    assert all(r["status"] == "completed" for r in results.values())

    # Decision logic: select best based on reward_mean
    completed_results = [r for r in results.values() if r["status"] == "completed"]
    best_candidate = max(completed_results, key=lambda r: r["metrics"]["reward_mean"])

    # Verify best selection
    assert best_candidate["candidate_id"] in {"cand_low", "cand_high", "cand_mid"}
    assert "metrics" in best_candidate
    assert best_candidate["metrics"]["reward_mean"] >= 10.0

    # Verify all results have required fields
    for result in completed_results:
        assert "candidate_id" in result
        assert "config" in result
        assert "metrics" in result
        assert "execution_time_s" in result
        assert "gpu_idx" in result
        assert result["gpu_idx"] in {0, 1}


@pytest.mark.asyncio
async def test_state_persistence_and_recovery(temp_state_dir):
    """Test that orchestrator state persists and can be recovered."""
    gpu_pool = MockGPUPool(capacity=2)

    # Phase 1: Create orchestrator and submit candidates
    orchestrator1 = ParallelResearchOrchestrator(gpu_pool, temp_state_dir)

    candidates = [
        MockCandidate("cand_x", {"param": 1}, execution_time=0.1),
        MockCandidate("cand_y", {"param": 2}, execution_time=0.1),
    ]

    for candidate in candidates:
        orchestrator1.submit_candidate(candidate)

    # Execute
    results1 = await orchestrator1.run_all()

    # Verify state file exists
    state_file = temp_state_dir / "orchestrator_state.json"
    assert state_file.exists()

    # Phase 2: Restore from state
    gpu_pool2 = MockGPUPool(capacity=2)
    orchestrator2 = ParallelResearchOrchestrator.restore_from_state(gpu_pool2, temp_state_dir)

    # Verify restored state matches
    assert len(orchestrator2.candidates) == 2
    assert "cand_x" in orchestrator2.candidates
    assert "cand_y" in orchestrator2.candidates

    # Verify candidate states restored
    for cid in ["cand_x", "cand_y"]:
        restored = orchestrator2.candidates[cid]
        original = orchestrator1.candidates[cid]

        assert restored.state == original.state == "completed"
        assert restored.result is not None
        assert restored.result["status"] == "completed"
        assert restored.config == original.config

    # Verify results restored
    assert orchestrator2.results == results1

    # Verify persisted state is valid JSON
    state = json.loads(state_file.read_text())
    assert "candidates" in state
    assert "results" in state
    assert len(state["candidates"]) == 2


@pytest.mark.asyncio
async def test_mixed_scenario_crash_and_constraint(temp_state_dir):
    """Integration test: combine crash, resource constraint, and recovery."""
    gpu_pool = MockGPUPool(capacity=2)
    orchestrator = ParallelResearchOrchestrator(gpu_pool, temp_state_dir)

    # Submit 4 candidates: 1 crashes, others succeed
    candidates = [
        MockCandidate("cand_1", {"lr": 0.001}, execution_time=0.15),
        MockCandidate("cand_2", {"lr": 0.002}, should_crash=True, execution_time=0.1),
        MockCandidate("cand_3", {"lr": 0.003}, execution_time=0.15),
        MockCandidate("cand_4", {"lr": 0.004}, execution_time=0.15),
    ]

    for candidate in candidates:
        orchestrator.submit_candidate(candidate)

    # Execute
    start_time = time.time()
    results = await orchestrator.run_all()
    elapsed = time.time() - start_time

    # Verify mixed results
    assert len(results) == 4
    assert results["cand_1"]["status"] == "completed"
    assert results["cand_2"]["status"] == "crashed"
    assert results["cand_3"]["status"] == "completed"
    assert results["cand_4"]["status"] == "completed"

    # Verify resource constraint respected (2 GPUs max)
    # With staggered execution times and 2 GPUs, should complete in ~0.3s
    assert elapsed < 0.5, f"Expected <0.5s with 2 GPUs, got {elapsed:.2f}s"

    # Verify state persisted correctly
    state_file = temp_state_dir / "orchestrator_state.json"
    state = json.loads(state_file.read_text())

    assert state["candidates"]["cand_2"]["state"] == "crashed"
    assert all(
        state["candidates"][cid]["state"] == "completed"
        for cid in ["cand_1", "cand_3", "cand_4"]
    )

    # Recovery: restore and verify
    gpu_pool_new = MockGPUPool(capacity=2)
    orchestrator_restored = ParallelResearchOrchestrator.restore_from_state(gpu_pool_new, temp_state_dir)

    assert len(orchestrator_restored.candidates) == 4
    assert orchestrator_restored.candidates["cand_2"].state == "crashed"

    # Select best from completed candidates
    completed = [
        r for r in orchestrator_restored.results.values()
        if r["status"] == "completed"
    ]
    assert len(completed) == 3

    best = max(completed, key=lambda r: r["metrics"]["reward_mean"])
    assert best["candidate_id"] in {"cand_1", "cand_3", "cand_4"}
